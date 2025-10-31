import os
import json
import logging
import time
import io
import base64
from typing import Annotated, TypedDict, List, Dict, Any
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from langchain.chat_models import init_chat_model
except Exception:
    init_chat_model = None

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from flask_cors import CORS
from dotenv import load_dotenv


MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

MODEL_PROVIDER = os.getenv("BEDROCK_PROVIDER", "bedrock_converse")
MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "64000"))

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")

if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
if os.getenv("LANGSMITH_ENDPOINT") and not os.getenv("LANGCHAIN_ENDPOINT"):
    os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT")

if os.getenv("LANGCHAIN_TRACING_V2") is None and os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

if MODEL_ID is None:
    raise RuntimeError("Set BEDROCK_MODEL_ID environment variable to the Bedrock model id you want to use.")



load_dotenv()
app = Flask(__name__)
CORS(app)



from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.runtime.context import RequestContext


app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("api")

def _extract_json_from_text(text: str):
    """Try multiple strategies to extract a single valid JSON object from free-form text.

    1) Look for fenced code blocks ```...``` and try to parse contents (skip optional language tag like 'json').
    2) Fallback to brace-matching from the first '{' to the corresponding closing '}'.
    """
    if not text:
        return None
    if "```" in text:
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            if "\n" in block:
                first_line, rest = block.split("\n", 1)
                lang = first_line.strip().lower()
                candidate = rest if lang in ("json", "javascript", "js") else block
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

def _parse_timestamp(value):
    """
    Try numeric (epoch seconds or ms) or ISO-like strings.
    Returns a datetime or None.
    """
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

def _is_number(v):
    return isinstance(v, (int, float)) or (isinstance(v, str) and v.replace(".", "", 1).isdigit())

def extract_time_series(payload: Any) -> List[Dict[str, Any]]:
    """
    Heuristic extractor that returns a list of time-series dicts:
    [{'name': name, 'times': [datetime, ...], 'values': [num, ...]}, ...]
    It tries several common payload shapes:
     - list of dicts with keys like 'timestamp' / 'time' and a numeric value key
     - dict with 'timestamps' and 'values' lists
     - dict with nested metrics e.g. {'metrics': [{'name':..., 'points': [{'t':..., 'v':...}, ...]}, ...]}`
    """
    series = []

    try:
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
                    t = _parse_timestamp(dp.get("timestamp"))
                    v = dp.get("value")
                    try:
                        v = float(v) if _is_number(v) else None
                    except Exception:
                        v = None
                    times.append(t)
                    values.append(v)
                series.append({"name": f"cpu_{range_name}", "times": times, "values": values})
    except Exception:
        pass

    if isinstance(payload, dict):
        if "timestamps" in payload and "values" in payload:
            ts = payload.get("timestamps", [])
            vs = payload.get("values", [])
            if isinstance(ts, list) and isinstance(vs, list) and len(ts) == len(vs):
                times = [_parse_timestamp(t) for t in ts]
                values = [float(v) if _is_number(v) else None for v in vs]
                series.append({"name": "series", "times": times, "values": values})

    if isinstance(payload, list) and payload and all(isinstance(x, dict) for x in payload):
        ts_keys = ("timestamp", "time", "date", "t", "ts")
        val_keys = None
        for d in payload:
            for k, v in d.items():
                if k.lower() in ts_keys:
                    continue
                if _is_number(v):
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
                        t = _parse_timestamp(d[tk])
                        break
                if t is None:
                    for k in d:
                        if "date" in k.lower() or "time" in k.lower() or k.lower().endswith("_at"):
                            t = _parse_timestamp(d[k])
                            if t:
                                break
                times.append(t)
                v = d.get(val_keys)
                try:
                    values.append(float(v) if _is_number(v) else None)
                except Exception:
                    values.append(None)
            series.append({"name": val_keys, "times": times, "values": values})

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
                            times.append(_parse_timestamp(t))
                            try:
                                values.append(float(v) if _is_number(v) else None)
                            except Exception:
                                values.append(None)
                        elif isinstance(p, list) and len(p) >= 2:
                            times.append(_parse_timestamp(p[0]))
                            try:
                                values.append(float(p[1]) if _is_number(p[1]) else None)
                            except Exception:
                                values.append(None)
                    name = m.get("name") or m.get("metric") or "metric"
                    series.append({"name": name, "times": times, "values": values})

    cleaned = []
    for s in series:
        pts = [(t, v) for t, v in zip(s.get("times", []), s.get("values", [])) if t is not None and v is not None]
        if len(pts) >= 2:
            times, values = zip(*pts)
            cleaned.append({"name": s.get("name", "series"), "times": list(times), "values": list(values)})
    return cleaned

def _plot_series_to_base64(times: List[datetime], values: List[float], title: str = None) -> str:
    """
    Plot the series and return a data: URI containing base64 PNG.
    """
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

class State(TypedDict):
    messages: Annotated[list, add_messages]
    raw_input: dict
    analysis: dict
    output: dict

graph_builder = StateGraph(State)

if init_chat_model is None:
    raise RuntimeError("langchain not found or import changed; install/adjust langchain before running.")

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

def analyze_with_llm(state: State) -> dict:
    logger.debug("analyze_with_llm: start")
    payload = state.get("raw_input", {})
    user_content = f"Input payload (JSON):\n{json.dumps(payload, indent=2)}\n\nFollow the system instructions exactly and return the RESPONSE_SCHEMA JSON."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content}
    ]

    response_text = None
    try:
        t0 = time.time()
        if hasattr(llm, "invoke"):
            resp = llm.invoke(messages)
            if hasattr(resp, "content"):
                response_text = resp.content
            elif isinstance(resp, dict):
                response_text = resp.get("content") or resp.get("text") or str(resp)
            else:
                response_text = str(resp)
        else:
            resp = llm(messages)
            if hasattr(resp, "content"):
                response_text = resp.content
            elif isinstance(resp, dict):
                response_text = resp.get("content") or resp.get("text") or str(resp)
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
            obj = _extract_json_from_text(text)
        parsed = obj if obj is not None else {"raw_text": text}
        if isinstance(parsed, dict):
            logger.debug("LLM parsed keys=%s", list(parsed.keys()))

    state["analysis"] = parsed
    logger.debug("analyze_with_llm: parsed=%s", list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__)
    return {"analysis": parsed}

def generate_plots(state: State) -> dict:
    """
    Look for timestamped series in multiple candidate locations (analysis.raw_findings,
    raw_input, and metric_data_by_range) and produce PNG plots.
    Adds analysis['plots'] = [{'name':..., 'data_uri': 'data:image/png;base64,...'}, ...]`
    """
    logger.debug("generate_plots: start (improved)")
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

    series_found = []
    try:
        for idx, cand in enumerate(uniq_candidates):
            try:
                s = extract_time_series(cand)
                logger.debug("generate_plots: candidate[%d] -> %d series", idx, len(s))
                series_found.extend(s)
            except Exception as e:
                logger.exception("generate_plots: extract_time_series failed on candidate[%d]: %s", idx, e)
    except Exception as e:
        logger.exception("generate_plots: extraction loop failure: %s", e)

    merged = []
    seen_keys = set()
    for s in series_found:
        try:
            times = s.get("times", [])
            vals = s.get("values", [])
            if not times or not vals or len(times) < 2:
                continue
            key = (s.get("name"), len(times), times[0].isoformat() if times[0] else None, times[-1].isoformat() if times[-1] else None)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(s)
        except Exception:
            continue

    plots = []
    for s in merged:
        try:
            name = s.get("name", "series")
            times = s.get("times", [])
            values = s.get("values", [])
            if len(times) < 2 or len(values) < 2:
                continue
            data_uri = _plot_series_to_base64(times, values, title=name)
            plots.append({"name": name, "data_uri": data_uri})
        except Exception as e:
            logger.exception("generate_plots: plotting failed for %s: %s", s.get("name"), e)
            continue

    state.setdefault("analysis", {})
    state["analysis"]["plots"] = plots
    logger.debug("generate_plots: created %d plots from %d candidate payloads (series_found=%d, merged=%d)",
                 len(plots), len(uniq_candidates), len(series_found), len(merged))
    return {"analysis": state["analysis"]}

    """
    Look for timestamped series in raw_input (or analysis.raw_findings) and produce PNG plots.
    Adds analysis['plots'] = [{'name':..., 'data_uri': 'data:image/png;base64,...'}, ...]`
    """
    logger.debug("generate_plots: start")
    payload = state.get("raw_input") or {}
    analysis = state.get("analysis") or {}

    candidate_payload = payload
    if isinstance(analysis, dict) and "raw_findings" in analysis and analysis["raw_findings"]:
        candidate_payload = analysis["raw_findings"]

    try:
        series_list = extract_time_series(candidate_payload)
    except Exception as e:
        logger.exception("generate_plots: extraction failed: %s", e)
        state.setdefault("analysis", {})
        state["analysis"]["plots"] = []
        return {"analysis": state["analysis"]}

    plots = []
    for s in series_list:
        try:
            name = s.get("name", "series")
            times = s.get("times", [])
            values = s.get("values", [])
            if len(times) < 2 or len(values) < 2:
                continue
            data_uri = _plot_series_to_base64(times, values, title=name)
            plots.append({"name": name, "data_uri": data_uri})
        except Exception as e:
            logger.exception("generate_plots: plotting failed for %s: %s", s.get("name"), e)
            continue

    state.setdefault("analysis", {})
    state["analysis"]["plots"] = plots
    logger.debug("generate_plots: created %d plots", len(plots))
    return {"analysis": state["analysis"]}

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

graph_builder.add_node("parse_input", parse_input)
graph_builder.add_node("analyze_with_llm", analyze_with_llm)
graph_builder.add_node("generate_plots", generate_plots)
graph_builder.add_node("format_output", format_output)

graph_builder.add_edge(START, "parse_input")
graph_builder.add_edge("parse_input", "analyze_with_llm")
graph_builder.add_edge("analyze_with_llm", "generate_plots")
graph_builder.add_edge("generate_plots", "format_output")
graph_builder.add_edge("format_output", END)

graph = graph_builder.compile()

def run_graph(payload: dict):
    logger.info("run_graph: received payload with keys=%s", list(payload.keys())[:20] if isinstance(payload, dict) else type(payload).__name__)
    initial_state = {
        "messages": [{"role": "user", "content": json.dumps(payload)}],
        "raw_input": payload
    }
    t0 = time.time()
    result = graph.invoke(initial_state)
    dt = (time.time() - t0) * 1000.0
    logger.info("run_graph: completed in %.1f ms", dt)
    return result.get("output") or result

@app.route("/health", methods=["GET"])
def health():
    logger.debug("health: ok")
    return jsonify({"status": "ok", "model_id": MODEL_ID})

@app.route("/analyze", methods=["POST"])
def analyze():
    req_id = getattr(request, "environ", {}).get("REQUEST_ID") or str(int(time.time() * 1000))
    logger.info("analyze[%s]: start content_type=%s", req_id, request.content_type)
    """
    Accepts:
    - multipart/form-data with 'file' field containing a JSON file
    OR
    - application/json body with the payload
    Returns LLM-produced analysis JSON (and any generated plots).
    """
    payload = None
    if "file" in request.files:
        f = request.files["file"]
        filename = secure_filename(f.filename or "upload.json")
        try:
            payload = json.load(f)
        except Exception as e:
            logger.warning("analyze[%s]: invalid json file: %s", req_id, e)
            return jsonify({"error": "invalid json file", "detail": str(e)}), 400
    else:
        try:
            payload = request.get_json(force=True)
        except Exception as e:
            logger.warning("analyze[%s]: invalid/empty JSON body: %s", req_id, e)
            return jsonify({"error": "no file and invalid/empty JSON body", "detail": str(e)}), 400

    if not payload:
        logger.warning("analyze[%s]: empty payload", req_id)
        return jsonify({"error": "empty payload"}), 400

    try:
        output = run_graph(payload)
    except Exception as e:
        logger.exception("analyze[%s]: graph run failed: %s", req_id, e)
        return jsonify({"error": "graph run failed", "detail": str(e)}), 500

    logger.info("analyze[%s]: success", req_id)
    return jsonify(output), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 6000)), debug=True)