import json
import logging
from datetime import datetime, timezone
from typing import Annotated, TypedDict, Any, List, Dict
import io
import base64
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from langchain.chat_models import init_chat_model
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from src.config import (
    MODEL_ID,
    AWS_REGION,
    MODEL_PROVIDER,
    MODEL_TEMPERATURE,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)

# State Cladss for LangGraph
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
                candidate = rest if lang in ("json") else block
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
    s = value.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    else:
        return None

def parse_time_series_from_payload(payload: Any) -> List[Dict[str, Any]]:
    series_found = []
    if payload.get("metric_data_by_range"):
        ranges = payload.get("metric_data_by_range") or {}
        for range_name, range_obj in ranges.items():
            datapoints = (range_obj or {}).get("datapoints") or []
            if not datapoints:
                continue
            times = []
            values = []
            for dp in datapoints:
                t = parse_timestamp(dp.get("timestamp"))
                v = dp.get("value")
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    v = None
                times.append(t)
                values.append(v)
            series_found.append({"name": f"cpu_{range_name}", "times": times, "values": values})
    if "timestamps" in payload and "values" in payload:
        ts = payload.get("timestamps", [])
        vs = payload.get("values", [])
        if len(ts) == len(vs):
            times = [parse_timestamp(t) for t in ts]
            try:
                values = [float(v) for v in vs]
            except (ValueError, TypeError):
                values = [None for v in vs]
            series_found.append({"name": "series", "times": times, "values": values})
    if "metrics" in payload:
        for metric in payload["metrics"]:
            points = metric.get("points") or metric.get("data") or metric.get("values")
            if points:
                times = []
                values = []
                for point in points:
                    t = point.get("t") or point.get("timestamp") or point.get("time") or point.get("date")
                    v = point.get("v") or point.get("value") or point.get("val")
                    times.append(parse_timestamp(t))
                    try:
                        values.append(float(v))
                    except (ValueError, TypeError):
                        values.append(None)
                name = metric.get("name") or metric.get("metric") or "metric"
                series_found.append({"name": name, "times": times, "values": values})
    cleaned = []
    for series in series:
        valid_points = [(t, v) for t, v in zip(series.get("times", []), series.get("values", [])) if t is not None and v is not None]
        if len(valid_points) >= 2:
            times, values = zip(*valid_points)
            cleaned.append({"name": series.get("name", "series"), "times": list(times), "values": list(values)})
    return cleaned

def convert_timeseries_to_base64_plot(times: List[datetime], values: List[float], title: str = None) -> str:
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
        content = last.get("content", "")
        try:
            raw = json.loads(content)
        except Exception:
            try:
                idx = content.index("{")
                raw = json.loads(content[idx:])
            except Exception:
                raw = {"text": content}
    state["raw_input"] = raw
    try:
        keys = list(raw.keys())
        logger.debug("parse_input: parsed keys=%s", keys)
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
        resp = llm.invoke(messages)
        response_text = resp.get("content") or resp.get("text") or str(resp)
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
        logger.debug("LLM parsed keys=%s", list(parsed.keys()))
    state["analysis"] = parsed
    logger.debug("analyze_with_llm: parsed=%s", list(parsed.keys()))
    return {"analysis": parsed}

def gather_plotting_data_sources(state: State) -> list:
    payload = state.get("raw_input") or {}
    analysis = state.get("analysis") or {}
    potential_data_sources = []
    if analysis.get("raw_findings"):
        potential_data_sources.append(analysis["raw_findings"])
    if payload:
        potential_data_sources.append(payload)
    for c in list(potential_data_sources):
        if c.get("metric_data_by_range"):
            potential_data_sources.append(c.get("metric_data_by_range"))
    if payload.get("metric_data_by_range"):
        potential_data_sources.append(payload.get("metric_data_by_range"))
    unique_data_sources = []
    seen_ids = set()
    for source in potential_data_sources:
        cid = id(source)
        if cid not in seen_ids:
            seen_ids.add(cid)
            unique_data_sources.append(source)
    return unique_data_sources

def merge_duplicate_series(data_sources: list) -> list:
    series_found = []
    for idx, source in enumerate(data_sources):
        try:
            extracted_series = parse_time_series_from_payload(source)
            logger.debug("merge_duplicate_series: candidate[%d] -> %d series", idx, len(extracted_series))
            series_found.extend(extracted_series)
        except Exception as e:
            logger.exception("merge_duplicate_series: parse_time_series_from_payload failed on candidate[%d]: %s", idx, e)
    merged = []
    seen_keys = set()
    for series in series_found:
        try:
            times = series.get("times", [])
            vals = series.get("values", [])
            if not times or not vals or len(times) < 2:
                continue
            key = (
                series.get("name"),
                len(times),
                times[0].isoformat() if times[0] else None,
                times[-1].isoformat() if times[-1] else None,
            )
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(series)
        except Exception:
            continue
    return merged

def render_plots_as_base64(series: list) -> list:
    plots = []
    for series_item in series:
        try:
            name = series_item.get("name", "series")
            times = series_item.get("times", [])
            values = series_item.get("values", [])
            if len(times) < 2 or len(values) < 2:
                continue
            data_uri = convert_timeseries_to_base64_plot(times, values, title=name)
            plots.append({"name": name, "data_uri": data_uri})
        except Exception as e:
            logger.exception(
                "render_plots_as_base64: plotting failed for %s: %s", series_item.get("name"), e
            )
            continue
    return plots

# generate plots node
def generate_plots(state: State) -> dict:
    logger.debug("generate_plots: start")
    potential_data_sources = gather_plotting_data_sources(state)
    series = merge_duplicate_series(potential_data_sources)
    plots = render_plots_as_base64(series)
    state.setdefault("analysis", {})
    state["analysis"]["plots"] = plots
    logger.debug(
        "generate_plots: created %d plots from %d candidates",
        len(plots),
        len(potential_data_sources),
    )
    return {"analysis": state["analysis"]}

# format output node
def format_output(state: State) -> dict:
    logger.debug("format_output: start")
    analysis = state.get("analysis", {})
    out = {
        "summary": analysis.get("summary"),
        "advice": analysis.get("advice"),
        "severity": analysis.get("severity", "MEDIUM"),
        "confidence": analysis.get("confidence", 0.5),
        "recommendations": analysis.get("recommendations", []),
        "diagnostics": analysis.get("diagnostics", []),
        "raw_findings": analysis.get("raw_findings", state.get("raw_input", {})),
        "plots": analysis.get("plots", []),
    }
    if not out["summary"] and "raw_text" in analysis:
        out["summary"] = "LLM returned non-JSON content. See raw_findings.raw_text."
        out["raw_findings"] = {"raw_text": analysis.get("raw_text")}
    state["output"] = out
    logger.debug("format_output: done with keys=%s", list(out.keys()))
    return {"output": out}

# run_graph() function to run the graph
def run_graph(graph, payload: dict):
    logger.info("run_graph: received payload with keys=%s", list(payload.keys()))
    initial_state = {
        "messages": [{"role": "user", "content": json.dumps(payload)}],
        "raw_input": payload
    }
    result = graph.invoke(initial_state)
    return result.get("output") or result
