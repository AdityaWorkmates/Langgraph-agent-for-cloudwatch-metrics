

from typing import Annotated, TypedDict, Any, Dict
import json
import os

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ----- State schema for LangGraph -----
class State(TypedDict):
    messages: Annotated[list, add_messages]
    raw_input: dict
    analysis: dict
    summary: str
    recommendations: dict
    output: dict

graph_builder = StateGraph(State)

# ----- Configure Bedrock model via LangChain init_chat_model -----
# Replace with actual model id present in your AWS Bedrock account.
MODEL_PROVIDER = "bedrock_converse"   # use bedrock_converse or 'bedrock' according to your langchain version
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "<REPLACE_WITH_BEDROCK_MODEL_ID>")
REGION_NAME = os.getenv("AWS_REGION", "us-west-2")  # if needed

llm = init_chat_model(
    MODEL_PROVIDER,
    model_id=MODEL_ID,
    region_name=REGION_NAME,
    # you can add other bedrock-specific kwargs here if your langchain version supports them
)

# ----- Prompt template (system + user) -----
SYSTEM_PROMPT = """
You are a reliable cloud reliability assistant. You receive an input payload describing monitoring/metrics/alarms
(CloudWatch-like or similar). The payload structure may vary; be resilient and focus on the data available.

Tasks:
1. Carefully inspect the input JSON. Extract the most important facts (timestamps, metric name & namespace,
   threshold, observed values, alarm state, evaluation periods, dimensions / resource identifiers, actions).
2. Produce a concise human-readable SUMMARY (2-6 sentences) describing what happened and why the alarm is firing (if it is).
3. Produce ACTIONABLE RECOMMENDATIONS: immediate actions, medium-term fixes, and long-term improvements.
   For each recommendation include: "what to do", "why", "estimated effort" (low/medium/high), and "priority" (P0..P3).
4. Assign a SEVERITY: one of CRITICAL / HIGH / MEDIUM / LOW and provide a short justification.
5. Provide concrete diagnostics/commands/checks to run (examples: check instance CPU, check process list, check memory, inspect disk I/O, verify autoscaling activity, check Lambda logs).
6. Output MUST be valid JSON only, matching the RESPONSE_SCHEMA below.

RESPONSE_SCHEMA:
{
  "summary": "<string>",
  "severity": "<CRITICAL|HIGH|MEDIUM|LOW>",
  "severity_reason": "<string>",
  "recommendations": [
    {
      "title": "<short>",
      "what": "<specific steps>",
      "why": "<reason>",
      "effort": "<low|medium|high>",
      "priority": "<P0|P1|P2|P3>"
    }, ...
  ],
  "diagnostics": ["<command or check 1>", "..."],
  "confidence": "<0.0-1.0>",
  "raw_findings": { ... }   // JSON fragment of extracted fields you relied on
}
"""

# ----- Node: parse_input -----
def parse_input(state: State) -> dict:
    """
    Accepts state['messages'] or state['raw_input'] and normalizes into raw_input (dict).
    If messages contains JSON string, parse it.
    """
    raw = state.get("raw_input") or {}
    # if user provided messages, try extracting JSON from last user message
    if not raw and state.get("messages"):
        # assume last message contains the JSON
        last = state["messages"][-1]
        content = last.get("content") if isinstance(last, dict) else str(last)
        # attempt to parse JSON substring
        try:
            raw = json.loads(content)
        except Exception:
            # fallback: try to find JSON object in content by simple heuristics
            try:
                start = content.index("{")
                raw = json.loads(content[start:])
            except Exception:
                raw = {"text": content}
    # ensure we always return a dict
    state["raw_input"] = raw if isinstance(raw, dict) else {"value": raw}
    return {"raw_input": state["raw_input"]}

# ----- Node: analyze_with_llm -----
def analyze_with_llm(state: State) -> dict:
    """
    Calls the Bedrock LLM with the system prompt and the raw_input.
    Stores the LLM response parsed as JSON into state['analysis'].
    """
    payload = state["raw_input"]
    user_content = f"Input payload (JSON):\n{json.dumps(payload, indent=2)}\n\nPlease follow the system instructions and produce the RESPONSE_SCHEMA JSON."
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_content}
    ]

    # LLM invocation (LangChain init_chat_model returns a chat model with .invoke in many setups)
    response = llm.invoke(messages)  # may be .generate or .invoke depending on langchain version
    # response likely contains a 'content' or 'text' field. Try to extract
    text = None
    if isinstance(response, dict):
        # try common keys
        text = response.get("content") or response.get("text") or response.get("message") or str(response)
    else:
        text = str(response)

    # Try to parse returned JSON (best-effort)
    try:
        parsed = json.loads(text)
    except Exception:
        # if model returned extra backticks or explanation, attempt to extract {...}
        try:
            start = text.index("{")
            parsed = json.loads(text[start:])
        except Exception:
            parsed = {"raw_text": text}

    state["analysis"] = parsed
    return {"analysis": parsed}


# ----- Node: format_output -----
def format_output(state: State) -> dict:
    """
    Validates & canonicalizes the analysis into a final 'output' dict the consumer will get.
    If the LLM analysis misses fields, fill default structure.
    """
    analysis = state.get("analysis", {})
    # Basic guards and defaults
    out = {
        "summary": analysis.get("summary", "No summary provided."),
        "severity": analysis.get("severity", "MEDIUM"),
        "severity_reason": analysis.get("severity_reason", ""),
        "recommendations": analysis.get("recommendations", []),
        "diagnostics": analysis.get("diagnostics", []),
        "confidence": analysis.get("confidence", 0.5),
        "raw_findings": analysis.get("raw_findings", state.get("raw_input", {}))
    }
    state["output"] = out
    return {"output": out}


# ----- Wire nodes into LangGraph -----
graph_builder.add_node("parse_input", parse_input)
graph_builder.add_node("analyze_with_llm", analyze_with_llm)
graph_builder.add_node("format_output", format_output)

graph_builder.add_edge(START, "parse_input")
graph_builder.add_edge("parse_input", "analyze_with_llm")
graph_builder.add_edge("analyze_with_llm", "format_output")
graph_builder.add_edge("format_output", END)

graph = graph_builder.compile()

# ----- Example runner function -----
def run_analysis(payload: dict) -> dict:
    """
    Run the LangGraph with a JSON payload and return the structured output.
    """
    initial_state = {
        "messages": [{"role": "user", "content": json.dumps(payload)}],
        "raw_input": payload
    }
    result = graph.run(initial_state)
    # graph.run returns final state (depending on langgraph version) — normalize:
    return result.get("output") or result

# ----- If run as script, demo with sample payload (replace by real model creds to actually call Bedrock) -----
if __name__ == "__main__":
    sample = {
      "instance_id": "i-097bf8bf5e21f9d52",
      "region": "us-west-2",
      "metric": {
        "namespace": "AWS/EC2",
        "metric_name": "CPUUtilization",
        "statistic": "Average",
        "unit": "Percent",
        "period_seconds": 60
      },
      "datapoints": [
        {"timestamp": "2025-10-28T07:05:00Z", "value": 0.0004632515893455},
        {"timestamp": "2025-10-28T07:10:00Z", "value": 94.4040188257285}
      ],
      "alarm": {
        "alarm_name": "TEST_CLOUDWATCH_SERVER_TEAM4_CPU#80",
        "state": "ALARM",
        "threshold": 80.0,
        "comparison_operator": "GreaterThanOrEqualToThreshold",
        "evaluation_periods": 1,
        "period_seconds": 300
      },
      "query_time_range": {"start_iso":"2025-10-28T06:11:06Z","end_iso":"2025-10-28T07:11:06Z"},
      "generated_at": "2025-10-28T07:11:07Z"
    }

    # This will call Bedrock when llm is properly configured. If you don't want to call Bedrock yet,
    # you can run run_analysis(sample) but expect the model call to happen.
    output = run_analysis(sample)
    print(json.dumps(output, indent=2))
