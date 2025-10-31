from flasgger import swag_from
import json
import logging
import time
from flask import request, jsonify
from werkzeug.utils import secure_filename
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

try:
    from langchain.chat_models import init_chat_model
except Exception:
    init_chat_model = None

from src.utils import _extract_json_from_text, extract_time_series, _plot_series_to_base64
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


class State(TypedDict):
    messages: Annotated[list, add_messages]
    raw_input: dict
    analysis: dict
    output: dict


graph_builder = StateGraph(State)

if init_chat_model is None:
    raise RuntimeError(
        "langchain not found or import changed; install/adjust langchain before running.")

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
    """Parses the input from the state and prepares it for the LLM."""

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
    """Analyzes the input with the LLM and returns the analysis."""

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

            obj = _extract_json_from_text(text)

        parsed = obj if obj is not None else {"raw_text": text}

        if isinstance(parsed, dict):

            logger.debug("LLM parsed keys=%s", list(parsed.keys()))

    state["analysis"] = parsed

    logger.debug("analyze_with_llm: parsed=%s", list(parsed.keys())
                 if isinstance(parsed, dict) else type(parsed).__name__)

    return {"analysis": parsed}


def generate_plots(state: State) -> dict:
    """Generates plots from the time series data in the analysis."""

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

                logger.debug(
                    "generate_plots: candidate[%d] -> %d series", idx, len(s))

                series_found.extend(s)

            except Exception as e:

                logger.exception(
                    "generate_plots: extract_time_series failed on candidate[%d]: %s", idx, e)

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

            key = (s.get("name"), len(times), times[0].isoformat(
            ) if times[0] else None, times[-1].isoformat() if times[-1] else None)

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

            logger.exception(
                "generate_plots: plotting failed for %s: %s", s.get("name"), e)

            continue

    state.setdefault("analysis", {})

    state["analysis"]["plots"] = plots

    logger.debug("generate_plots: created %d plots from %d candidate payloads (series_found=%d, merged=%d)",

                 len(plots), len(uniq_candidates), len(series_found), len(merged))

    return {"analysis": state["analysis"]}


def format_output(state: State) -> dict:
    """Formats the output of the analysis."""

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
    """Runs the graph with the given payload."""

    logger.info("run_graph: received payload with keys=%s", list(payload.keys())[
                :20] if isinstance(payload, dict) else type(payload).__name__)

    initial_state = {

        "messages": [{"role": "user", "content": json.dumps(payload)}],

        "raw_input": payload

    }

    t0 = time.time()

    result = graph.invoke(initial_state)

    dt = (time.time() - t0) * 1000.0

    logger.info("run_graph: completed in %.1f ms", dt)

    return result.get("output") or result


def health():
    """Health check endpoint.



    ---



    responses:



      200:



        description: The application is healthy.



    """

    logger.info("Health check requested")

    response = jsonify({"status": "ok", "model_id": MODEL_ID})

    logger.info(f"Health check response: {response.get_data(as_text=True)}")

    return response


def analyze():
    """Analyzes the given payload.



    ---



    parameters:



      - name: file



        in: formData



        type: file



        required: false



        description: The JSON file to analyze.



      - name: body



        in: body



        required: false



        schema:



          type: object



    responses:



      200:



        description: The analysis was successful.



      400:



        description: The request was invalid.



      500:



        description: An error occurred during the analysis.



    """

    req_id = getattr(request, "environ", {}).get(
        "REQUEST_ID") or str(int(time.time() * 1000))

    logger.info(f"Analyze request {req_id} received")

    logger.info(f"Request headers: {request.headers}")

    logger.info(f"Request body: {request.get_data(as_text=True)}")

    payload = None

    if "file" in request.files:

        f = request.files["file"]

        filename = secure_filename(f.filename or "upload.json")

        logger.info(f"Processing uploaded file: {filename}")

        try:

            payload = json.load(f)

        except Exception as e:

            logger.warning(f"Invalid JSON file uploaded: {e}")

            return jsonify({"error": "invalid json file", "detail": str(e)}), 400

    else:

        try:

            payload = request.get_json(force=True)

        except Exception as e:

            logger.warning(f"Invalid JSON body: {e}")

            return jsonify({"error": "no file and invalid/empty JSON body", "detail": str(e)}), 400

    if not payload:

        logger.warning("Empty payload received")

        return jsonify({"error": "empty payload"}), 400

    try:

        output = run_graph(payload)

        logger.info(f"Analysis successful for request {req_id}")

        logger.debug(f"Analysis output: {output}")

        return jsonify(output), 200

    except Exception as e:

        logger.exception(f"Graph run failed for request {req_id}: {e}")

        return jsonify({"error": "graph run failed", "detail": str(e)}), 500
