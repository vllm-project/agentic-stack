import os

from openai import OpenAI

client = OpenAI(
    base_url=os.environ.get("AGENTIC_STACK_BASE_URL", "http://127.0.0.1:5969/v1"),
    api_key="dummy",
)
