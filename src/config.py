
import os

MODEL_ID = os.getenv("MODEL_ID", "us.anthropic.claude-sonnet-4-20250514-v1:0")
AWS_REGION = os.getenv("AWS_REGION", "us-west-2")
MODEL_PROVIDER = os.getenv("BEDROCK_PROVIDER", "bedrock_converse")
MODEL_TEMPERATURE = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
MODEL_MAX_TOKENS = int(os.getenv("MODEL_MAX_TOKENS", "64000"))

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")

LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()

if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT")
if os.getenv("LANGSMITH_ENDPOINT") and not os.getenv("LANGCHAIN_ENDPOINT"):
    langchain_endpoint = os.getenv("LANGSMITH_ENDPOINT")
    if langchain_endpoint:
        os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint

if os.getenv("LANGCHAIN_TRACING_V2") is None and os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

if MODEL_ID is None:
    raise RuntimeError(
        "Set BEDROCK_MODEL_ID environment variable to the Bedrock model id you want to use.")
