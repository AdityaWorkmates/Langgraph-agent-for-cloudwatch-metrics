

import json
import logging
import time
from datetime import datetime, timezone
from typing import Annotated, TypedDict, Any, List, Dict
import io
import base64

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from langgraph.graph import State

from src.config import (
    MODEL_ID,
    AWS_REGION,
    MODEL_PROVIDER,
    MODEL_TEMPERATURE,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN
)

logger = logging.getLogger(__name__)

# State TypedDict for LangGraph
class State(TypedDict):
    messages: Annotated[list, add_messages]
    raw_input: dict
    analysis: dict
    output: dict

llm_kwargs = {
    "model": MODEL_ID,
    "model_provider": MODEL_PROVIDER,
    "region_name": AWS_REGION,
    "temperature": MODEL_TEMPERATURE,
}

if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
    llm_kwargs.update({
        "aws_access_key_id": AWS_ACCESS_KEY_ID,
        "aws_secret_access_key": AWS_SECRET_ACCESS_KEY,
    })
    if AWS_SESSION_TOKEN:
        llm_kwargs["aws_session_token"] = AWS_SESSION_TOKEN

llm = init_chat_model(**llm_kwargs)

SYSTEM_PROMPT = '''
You are a cloud reliability assistant. You will be given an input JSON payload containing monitoring, metrics, or alarm information.
The payload structure may vary. Tasks:
1) Extract key facts (resource id, metric name, timestamps, values, threshold, alarm state, actions).
2) Produce a short SUMMARY (2-4 sentences) describing what happened.
3) Provide ADVICE (one-paragraph) and a prioritized list of REMEDIATION recommendations (immediate, short-term, long-term).
4) Provide DIAGNOSTIC COMMANDS to run (concrete aws/ssm/journalctl/top/ps commands).
5) Assign SEVERITY (CRITICAL/HIGH/MEDIUM/LOW) and a numeric CONFIDENCE 0.0-1.0.
6) Output strictly valid JSON (no extra plaintext), matching the schema below.

RESPONSE_SCHEMA:
{
  "summary": "<string>",
  "advice": "<string>",
  "severity": "<CRITICAL|HIGH|MEDIUM|LOW>",
  "confidence": <0.0-1.0>,
  "recommendations": [
    {
      "title": "<short>",
      "what": "<steps>",
      "why": "<reason>",
      "effort": "<low|medium|high>",
      "priority": "<P0|P1|P2|P3>"
    },...
  ],
  "diagnostics": ["<command1>", "<command2>", ...],
  "raw_findings": { ... }
}
'''


def extract_json_from_text(text: str):
    if not text:
        return None
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            if "\n" in block:
                first_line, rest = block.split("\n", 1)
                lang = first_line.strip().lower()
                candidate = rest if lang in (
                    "json", "javascript", "js") else block
            else:
                candidate = block
            candidate = candidate.strip()
            try:
                return json.loads(candidate)
            except Exception:
                continue
    start = text.find("{")
    if start != -1:
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:idx + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        break
    return None


def parse_timestamp(value):
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            if value > 1e12:
                return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
            if value > 1e9:
                return datetime.fromtimestamp(value, tz=timezone.utc)
        if isinstance(value, str) and value.isdigit():
            n = int(value)
            if n > 1e12:
                return datetime.fromtimestamp(n / 1000.0, tz=timezone.utc)
            if n > 1e9:
                return datetime.fromtimestamp(n, tz=timezone.utc)
    except Exception:
        pass

    if isinstance(value, str):
        s = value.strip()
        try:
            if s.endswith("Z"):
                s2 = s.replace("Z", "+00:00")
                return datetime.fromisoformat(s2)
            return datetime.fromisoformat(s)
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
            except Exception:
                continue
    return None


def is_number(v):
    return isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit())


def extract_from_metric_data_by_range(payload: dict) -> list:
    series = []
    if isinstance(payload, dict) and isinstance(payload.get("metric_data_by_range"), dict):
        ranges = payload.get("metric_data_by_range") or {}
        for range_name, range_obj in ranges.items():
            datapoints = (range_obj or {}).get("datapoints") or []
            if not isinstance(datapoints, list) or not datapoints:
                continue
            times = []
            values = []
            for dp in datapoints:
                if not isinstance(dp, dict):
                    continue
                t = parse_timestamp(dp.get("timestamp"))
                v = dp.get("value")
                try:
                    v = float(v) if is_number(v) else None
                except Exception:
                    v = None
                times.append(t)
                values.append(v)
            series.append({"name": f"cpu_{range_name}", "times": times, "values": values})
    return series

def extract_from_timestamps_and_values(payload: dict) -> list:
    series = []
    if isinstance(payload, dict):
        if "timestamps" in payload and "values" in payload:
            ts = payload.get("timestamps", [])
            vs = payload.get("values", [])
            if isinstance(ts, list) and isinstance(vs, list) and len(ts) == len(vs):
                times = [parse_timestamp(t) for t in ts]
                values = [float(v) if is_number(v) else None for v in vs]
                series.append({"name": "series", "times": times, "values": values})
    return series

def extract_from_list_of_dicts(payload: list) -> list:
    series = []
    if isinstance(payload, list) and payload and all(isinstance(x, dict) for x in payload):
        ts_keys = ("timestamp", "time", "date", "t", "ts")
        val_keys = None
        for d in payload:
            for k, v in d.items():
                if k.lower() in ts_keys:
                    continue
                if is_number(v):
                    val_keys = k
                    break
            if val_keys:
                break
        if val_keys:
            times = []
            values = []
            for d in payload:
                t = None
                for tk in ts_keys:
                    if tk in d:
                        t = parse_timestamp(d[tk])
                        break
                if t is None:
                    for k in d:
                        if "date" in k.lower() or "time" in k.lower() or k.lower().endswith("_at"):
                            t = parse_timestamp(d[k])
                            if t:
                                break
                times.append(t)
                v = d.get(val_keys)
                try:
                    values.append(float(v) if is_number(v) else None)
                except Exception:
                    values.append(None)
            series.append({"name": val_keys, "times": times, "values": values})
    return series

def extract_from_nested_metrics(payload: dict) -> list:
    series = []
    if isinstance(payload, dict) and "metrics" in payload and isinstance(payload["metrics"], list):
        for m in payload["metrics"]:
            if isinstance(m, dict):
                points = m.get("points") or m.get("data") or m.get("values")
                if isinstance(points, list) and points:
                    times = []
                    values = []
                    for p in points:
                        if isinstance(p, dict):
                            t = p.get("t") or p.get("timestamp") or p.get("time") or p.get("date")
                            v = p.get("v") or p.get("value") or p.get("val")
                            times.append(parse_timestamp(t))
                            try:
                                values.append(float(v) if is_number(v) else None)
                            except Exception:
                                values.append(None)
                        elif isinstance(p, list) and len(p) >= 2:
                            times.append(parse_timestamp(p[0]))
                            try:
                                values.append(float(p[1]) if is_number(p[1]) else None)
                            except Exception:
                                values.append(None)
                    name = m.get("name") or m.get("metric") or "metric"
                    series.append({"name": name, "times": times, "values": values})
    return series

def extract_time_series(payload: Any) -> List[Dict[str, Any]]:
    series = []
    series.extend(extract_from_metric_data_by_range(payload))
    series.extend(extract_from_timestamps_and_values(payload))
    series.extend(extract_from_list_of_dicts(payload))
    series.extend(extract_from_nested_metrics(payload))

    cleaned = []
    for s in series:
        pts = [(t, v) for t, v in zip(s.get("times", []), s.get("values", [])) if t is not None and v is not None]
        if len(pts) >= 2:
            times, values = zip(*pts)
            cleaned.append({"name": s.get("name", "series"), "times": list(times), "values": list(values)})
    return cleaned


def plot_series_to_base64(times: List[datetime], values: List[float], title: str = None) -> str:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(times, values, marker="o", linewidth=1)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    if title:
        ax.set_title(title)
    fig.autofmt_xdate()
    buf = io.BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


# parse input node
def parse_input(state: State) -> dict:    
    logger.debug("parse_input: start")
    raw = state.get("raw_input") or {}
    if (not raw) and state.get("messages"):
        last = state["messages"][-1]
        if isinstance(last, dict):
            content = last.get("content", "")
        else:
            content = str(last)
        try:
            raw = json.loads(content)
        except Exception:
            try:
                idx = content.index("{")
                raw = json.loads(content[idx:])
            except Exception:
                raw = {"text": content}
    if not isinstance(raw, dict):
        raw = {"value": raw}
    state["raw_input"] = raw
    try:
        keys = list(raw.keys()) if isinstance(raw, dict) else []
        logger.debug("parse_input: parsed keys=%s", keys[:20])
    except Exception:
        logger.debug("parse_input: parsed payload (keys unavailable)")
    return {"raw_input": state["raw_input"]}

# analyze with LLM node
def analyze_with_llm(state: State) -> dict:
    
    logger.debug("analyze_with_llm: start")
    payload = state.get("raw_input", {})
    user_content = f"Input payload (JSON):\n{json.dumps(payload, indent=2)}\n\nFollow the system instructions exactly and return the RESPONSE_SCHEMA JSON."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content},
    ]
    response_text = None
    try:
        t0 = time.time()
        if hasattr(llm, "invoke"):
            resp = llm.invoke(messages)
            if hasattr(resp, "content"):
                response_text = resp.content
            elif isinstance(resp, dict):
                response_text = resp.get(
                    "content") or resp.get("text") or str(resp)
            else:
                response_text = str(resp)
        else:
            resp = llm(messages)
            if hasattr(resp, "content"):
                response_text = resp.content
            elif isinstance(resp, dict):
                response_text = resp.get(
                    "content") or resp.get("text") or str(resp)
            else:
                response_text = str(resp)
        dt = (time.time() - t0) * 1000.0
        logger.info("LLM call completed in %.1f ms", dt)
        if response_text:
            logger.debug("LLM raw text length=%d", len(response_text))
    except Exception as e:
        logger.exception("LLM call failed: %s", e)
        state["analysis"] = {"error": str(e), "raw_text": None}
        return {"analysis": state["analysis"]}
    parsed = None
    if not response_text:
        parsed = {"raw_text": None}
    else:
        text = response_text.strip()
        obj = None
        try:
            obj = json.loads(text)
        except Exception:
            obj = extract_json_from_text(text)
        parsed = obj if obj is not None else {"raw_text": text}
        if isinstance(parsed, dict):
            logger.debug("LLM parsed keys=%s", list(parsed.keys()))
    state["analysis"] = parsed
    logger.debug("analyze_with_llm: parsed=%s", list(parsed.keys())
                 if isinstance(parsed, dict) else type(parsed).__name__)
    return {"analysis": parsed}


def get_plot_candidates(state: State) -> list:
    payload = state.get("raw_input") or {}
    analysis = state.get("analysis") or {}
    candidates = []
    if isinstance(analysis, dict) and analysis.get("raw_findings"):
        candidates.append(analysis["raw_findings"])
    if payload:
        candidates.append(payload)

    for c in list(candidates):
        if isinstance(c, dict) and c.get("metric_data_by_range"):
            candidates.append(c.get("metric_data_by_range"))

    if isinstance(payload, dict) and payload.get("metric_data_by_range"):
        candidates.append(payload.get("metric_data_by_range"))
    uniq_candidates = []
    seen_ids = set()
    for c in candidates:
        cid = id(c)
        if cid not in seen_ids:
            seen_ids.add(cid)
            uniq_candidates.append(c)
    return uniq_candidates


def extract_and_merge_series(candidates: list) -> list:

    series_found = []
    for idx, cand in enumerate(candidates):
        try:
            s = extract_time_series(cand)
            logger.debug("generate_plots: candidate[%d] -> %d series", idx, len(s))
            series_found.extend(s)
        except Exception as e:
            logger.exception(
                "generate_plots: extract_time_series failed on candidate[%d]: %s", idx, e
            )
    merged = []
    seen_keys = set()
    for s in series_found:
        try:
            times = s.get("times", [])
            vals = s.get("values", [])
            if not times or not vals or len(times) < 2:
                continue
            key = (
                s.get("name"),
                len(times),
                times[0].isoformat() if times[0] else None,
                times[-1].isoformat() if times[-1] else None,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(s)
        except Exception:
            continue
    return merged


def create_plots(series: list) -> list:
    plots = []
    for s in series:
        try:
            name = s.get("name", "series")
            times = s.get("times", [])
            values = s.get("values", [])
            if len(times) < 2 or len(values) < 2:
                continue
            data_uri = plot_series_to_base64(times, values, title=name)
            plots.append({"name": name, "data_uri": data_uri})
        except Exception as e:
            logger.exception(
                "generate_plots: plotting failed for %s: %s", s.get("name"), e
            )
            continue
    return plots

# generate plots node
def generate_plots(state: State) -> dict:
    logger.debug("generate_plots: start")
    candidates = get_plot_candidates(state)
    series = extract_and_merge_series(candidates)
    plots = create_plots(series)
    state.setdefault("analysis", {})
    state["analysis"]["plots"] = plots
    logger.debug(
        "generate_plots: created %d plots from %d candidates",
        len(plots),
        len(candidates),
    )
    return {"analysis": state["analysis"]}

# format output node
def format_output(state: State) -> dict:
    logger.debug("format_output: start")
    analysis = state.get("analysis", {})
    out = {
        "summary": analysis.get("summary") if isinstance(analysis, dict) else None,
        "advice": analysis.get("advice") if isinstance(analysis, dict) else None,
        "severity": analysis.get("severity") if isinstance(analysis, dict) else "MEDIUM",
        "confidence": analysis.get("confidence") if isinstance(analysis, dict) else 0.5,
        "recommendations": analysis.get("recommendations") if isinstance(analysis, dict) else [],
        "diagnostics": analysis.get("diagnostics") if isinstance(analysis, dict) else [],
        "raw_findings": analysis.get("raw_findings") if isinstance(analysis, dict) else state.get("raw_input", {}),
        "plots": analysis.get("plots") if isinstance(analysis, dict) else [],
    }
    if not out["summary"] and isinstance(analysis, dict) and "raw_text" in analysis:
        out["summary"] = "LLM returned non-JSON content. See raw_findings.raw_text."
        out["raw_findings"] = {"raw_text": analysis.get("raw_text")}
    state["output"] = out
    logger.debug("format_output: done with keys=%s", list(out.keys()))
    return {"output": out}

# main function to run the graph
def run_graph(graph, payload: dict):
    logger.info("run_graph: received payload with keys=%s", list(payload.keys())
                if isinstance(payload, dict) else type(payload).__name__)
    initial_state = {
        "messages": [{"role": "user", "content": json.dumps(payload)}],
        "raw_input": payload
    }
    t0 = time.time()
    result = graph.invoke(initial_state)
    dt = (time.time() - t0) * 1000.0
    logger.info("run_graph: completed in %.1f ms", dt)
    return result.get("output") or result
