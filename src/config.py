
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL_PROVIDER = os.getenv("BEDROCK_PROVIDER", "bedrock_converse")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "64000"))

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT") or os.getenv("LANGSMITH_ENDPOINT")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2") or ("true" if os.getenv("LANGCHAIN_API_KEY") else None)

if MODEL_ID is None:
    raise RuntimeError(
        "Set BEDROCK_MODEL_ID environment variable to the Bedrock model id you want to use.")


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