from typing import Annotated, TypedDict, Any, Dict
import json
import os

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]
    raw_input: dict
    analysis: dict
    summary: str
    recommendations: dict
    output: dict

graph_builder = StateGraph(State)

MODEL_PROVIDER = "bedrock_converse"
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "<REPLACE_WITH_BEDROCK_MODEL_ID>")
REGION_NAME = os.getenv("AWS_REGION", "us-west-2")

llm = init_chat_model(
    MODEL_PROVIDER,
    model_id=MODEL_ID,
    region_name=REGION_NAME,
)

SYSTEM_PROMPT = '''
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
  "raw_findings": { ... }
}
'''

def parse_input(state: State) -> dict:
    """
    Accepts state['messages'] or state['raw_input'] and normalizes into raw_input (dict).
    If messages contains JSON string, parse it.
    """
    raw = state.get("raw_input") or {}
    if not raw and state.get("messages"):
        last = state["messages"][-1]
        content = last.get("content") if isinstance(last, dict) else str(last)
        try:
            raw = json.loads(content)
        except Exception:
            try:
                start = content.index("{")
                raw = json.loads(content[start:])
            except Exception:
                raw = {"text": content}
    state["raw_input"] = raw if isinstance(raw, dict) else {"value": raw}
    return {"raw_input": state["raw_input"]}

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

    response = llm.invoke(messages)
    text = None
    if isinstance(response, dict):
        text = response.get("content") or response.get("text") or response.get("message") or str(response)
    else:
        text = str(response)

    try:
        parsed = json.loads(text)
    except Exception:
        try:
            start = text.index("{")
            parsed = json.loads(text[start:])
        except Exception:
            parsed = {"raw_text": text}

    state["analysis"] = parsed
    return {"analysis": parsed}


def format_output(state: State) -> dict:
    """
    Validates & canonicalizes the analysis into a final 'output' dict the consumer will get.
    If the LLM analysis misses fields, fill default structure.
    """
    analysis = state.get("analysis", {})
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

graph_builder.add_node("parse_input", parse_input)
graph_builder.add_node("analyze_with_llm", analyze_with_llm)
graph_builder.add_node("format_output", format_output)

graph_builder.add_edge(START, "parse_input")
graph_builder.add_edge("parse_input", "analyze_with_llm")
graph_builder.add_edge("analyze_with_llm", "format_output")
graph_builder.add_edge("format_output", END)

graph = graph_builder.compile()

def run_analysis(payload: dict) -> dict:
    """
    Run the LangGraph with a JSON payload and return the structured output.
    """
    initial_state = {
        "messages": [{"role": "user", "content": json.dumps(payload)}],
        "raw_input": payload
    }
    result = graph.run(initial_state)
    return result.get("output") or result

if __name__ == "__main__":
    payload = json.load(open("payload.json"))
    output = run_analysis(payload)
    print(json.dumps(output, indent=2))