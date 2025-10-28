# app.py
import os
import json
import logging
import time
from typing import Annotated, TypedDict
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# LangChain / LangGraph imports
try:
    from langchain.chat_models import init_chat_model
except Exception:
    # If you use a newer/older langchain, adjust import accordingly.
    init_chat_model = None

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


# MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")




# ---------- Configuration ----------
MODEL_PROVIDER = os.getenv("BEDROCK_PROVIDER", "bedrock_converse")  # change if needed
# MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-5-20250929-v1:0")
MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
# Optional model settings you can tune
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "64000"))

if MODEL_ID is None:
    raise RuntimeError("Set BEDROCK_MODEL_ID environment variable to the Bedrock model id you want to use.")

# ---------- Flask app ----------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB max upload (tweak as needed)

# ---------- Logging ----------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("api")

# ---------- Helpers ----------
def _extract_json_from_text(text: str):
    """Try multiple strategies to extract a single valid JSON object from free-form text.

    1) Look for fenced code blocks ```...``` and try to parse contents (skip optional language tag like 'json').
    2) Fallback to brace-matching from the first '{' to the corresponding closing '}'.
    """
    if not text:
        return None
    # Strategy 1: fenced code blocks
    if "```" in text:
        parts = text.split("```")
        # code blocks are in odd indices: 1,3,5,...
        for i in range(1, len(parts), 2):
            block = parts[i]
            # Remove optional language tag on the first line
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
    # Strategy 2: brace matching
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

# ---------- LangGraph State ----------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    raw_input: dict
    analysis: dict
    output: dict

graph_builder = StateGraph(State)

# ---------- Init LLM (LangChain -> Bedrock) ----------
if init_chat_model is None:
    raise RuntimeError("langchain not found or import changed; install/adjust langchain before running.")

# Initialize chat model for Bedrock using explicit keyword args
llm = init_chat_model(
    model=MODEL_ID,
    model_provider=MODEL_PROVIDER,
    region_name=AWS_REGION,
    temperature=MODEL_TEMPERATURE,
)

# ---------- System prompt enforced schema ----------
SYSTEM_PROMPT = """
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
"""

# ---------- parse node ----------
def parse_input(state: State) -> dict:
    logger.debug("parse_input: start")
    raw = state.get("raw_input") or {}
    # If messages exist and raw_input empty, try parse JSON from last user message
    if (not raw) and state.get("messages"):
        last = state["messages"][-1]
        if isinstance(last, dict):
            content = last.get("content", "")
        else:
            content = str(last)
        try:
            raw = json.loads(content)
        except Exception:
            # fallback: try to locate a JSON substring
            try:
                idx = content.index("{")
                raw = json.loads(content[idx:])
            except Exception:
                raw = {"text": content}
    if not isinstance(raw, dict):
        raw = {"value": raw}
    state["raw_input"] = raw
    # Avoid logging entire payload; log keys only
    try:
        keys = list(raw.keys()) if isinstance(raw, dict) else []
        logger.debug("parse_input: parsed keys=%s", keys[:20])
    except Exception:
        logger.debug("parse_input: parsed payload (keys unavailable)")
    return {"raw_input": state["raw_input"]}

# ---------- analyze node (calls Bedrock via LangChain) ----------
def analyze_with_llm(state: State) -> dict:
    logger.debug("analyze_with_llm: start")
    payload = state.get("raw_input", {})
    user_content = f"Input payload (JSON):\n{json.dumps(payload, indent=2)}\n\nFollow the system instructions exactly and return the RESPONSE_SCHEMA JSON."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content}
    ]

    # Try calling common LangChain chat model interfaces
    response_text = None
    try:
        t0 = time.time()
        # Many LangChain chat models support .invoke(messages) or __call__ or .generate
        if hasattr(llm, "invoke"):
            resp = llm.invoke(messages)
            # Prefer message.content when available
            if hasattr(resp, "content"):
                response_text = resp.content
            elif isinstance(resp, dict):
                response_text = resp.get("content") or resp.get("text") or str(resp)
            else:
                response_text = str(resp)
        else:
            # fallback to calling llm(messages)
            resp = llm(messages)
            # resp could be an object with generations
            if hasattr(resp, "content"):
                response_text = resp.content
            elif isinstance(resp, dict):
                response_text = resp.get("content") or resp.get("text") or str(resp)
            else:
                response_text = str(resp)
        dt = (time.time() - t0) * 1000.0
        logger.info("LLM call completed in %.1f ms", dt)
    except Exception as e:
        # store error in analysis for observability
        logger.exception("LLM call failed: %s", e)
        state["analysis"] = {"error": str(e), "raw_text": None}
        return {"analysis": state["analysis"]}

    # try to extract JSON from response_text
    parsed = None
    if not response_text:
        parsed = {"raw_text": None}
    else:
        text = response_text.strip()
        # direct parse, then fenced code/brace-matching fallback
        obj = None
        try:
            obj = json.loads(text)
        except Exception:
            obj = _extract_json_from_text(text)
        parsed = obj if obj is not None else {"raw_text": text}

    state["analysis"] = parsed
    logger.debug("analyze_with_llm: parsed=%s", list(parsed.keys()) if isinstance(parsed, dict) else type(parsed).__name__)
    return {"analysis": parsed}

# ---------- format node ----------
def format_output(state: State) -> dict:
    logger.debug("format_output: start")
    analysis = state.get("analysis", {})
    # If analysis already matches schema, pass through. Otherwise try to create a minimal schema.
    out = {
        "summary": analysis.get("summary") if isinstance(analysis, dict) else None,
        "advice": analysis.get("advice") if isinstance(analysis, dict) else None,
        "severity": analysis.get("severity") if isinstance(analysis, dict) else "MEDIUM",
        "confidence": analysis.get("confidence") if isinstance(analysis, dict) else 0.5,
        "recommendations": analysis.get("recommendations") if isinstance(analysis, dict) else [],
        "diagnostics": analysis.get("diagnostics") if isinstance(analysis, dict) else [],
        "raw_findings": analysis.get("raw_findings") if isinstance(analysis, dict) else state.get("raw_input", {})
    }
    # Fill sensible defaults if LLM returned raw_text only
    if not out["summary"] and isinstance(analysis, dict) and "raw_text" in analysis:
        out["summary"] = "LLM returned non-JSON content. See raw_findings.raw_text."
        out["raw_findings"] = {"raw_text": analysis.get("raw_text")}
    # ensure types are serializable
    state["output"] = out
    logger.debug("format_output: done with keys=%s", list(out.keys()))
    return {"output": out}

# ---------- Wire nodes into LangGraph ----------
graph_builder.add_node("parse_input", parse_input)
graph_builder.add_node("analyze_with_llm", analyze_with_llm)
graph_builder.add_node("format_output", format_output)

graph_builder.add_edge(START, "parse_input")
graph_builder.add_edge("parse_input", "analyze_with_llm")
graph_builder.add_edge("analyze_with_llm", "format_output")
graph_builder.add_edge("format_output", END)

graph = graph_builder.compile()

# ---------- Runner helper ----------
def run_graph(payload: dict):
    logger.info("run_graph: received payload with keys=%s", list(payload.keys())[:20] if isinstance(payload, dict) else type(payload).__name__)
    initial_state = {
        "messages": [{"role": "user", "content": json.dumps(payload)}],
        "raw_input": payload
    }
    t0 = time.time()
    # Compiled graphs use invoke()/stream() instead of run()
    result = graph.invoke(initial_state)
    dt = (time.time() - t0) * 1000.0
    logger.info("run_graph: completed in %.1f ms", dt)
    # graph.run returns final state depending on version
    return result.get("output") or result

# ---------- Flask endpoints ----------
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
    Returns LLM-produced analysis JSON.
    """
    # 1) Get payload
    payload = None
    # multipart file
    if "file" in request.files:
        f = request.files["file"]
        filename = secure_filename(f.filename or "upload.json")
        try:
            payload = json.load(f)
        except Exception as e:
            logger.warning("analyze[%s]: invalid json file: %s", req_id, e)
            return jsonify({"error": "invalid json file", "detail": str(e)}), 400
    else:
        # try JSON body
        try:
            payload = request.get_json(force=True)
        except Exception as e:
            logger.warning("analyze[%s]: invalid/empty JSON body: %s", req_id, e)
            return jsonify({"error": "no file and invalid/empty JSON body", "detail": str(e)}), 400

    if not payload:
        logger.warning("analyze[%s]: empty payload", req_id)
        return jsonify({"error": "empty payload"}), 400

    # 2) Run LangGraph analysis
    try:
        output = run_graph(payload)
    except Exception as e:
        logger.exception("analyze[%s]: graph run failed: %s", req_id, e)
        return jsonify({"error": "graph run failed", "detail": str(e)}), 500

    # 3) Return structured output
    logger.info("analyze[%s]: success", req_id)
    return jsonify(output), 200

# ---------- Run app ----------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 6900)), debug=True)
