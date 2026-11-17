"""Native tool/function calling (the modern way to build agents) — OpenAI-compatible API.

pip install openai ; export OPENAI_API_KEY=...
    python examples/04_tool_calling_native.py
"""
import json

from openai import OpenAI

client = OpenAI()

# 1) Describe tools with JSON Schema
tools = [{
    "type": "function",
    "function": {
        "name": "get_pipeline_status",
        "description": "Get the latest run status of a data pipeline by name",
        "parameters": {"type": "object",
                       "properties": {"pipeline": {"type": "string", "description": "pipeline name"}},
                       "required": ["pipeline"]},
    },
}]


# 2) Implement them (normally: call ADF / Databricks Jobs REST API)
def get_pipeline_status(pipeline: str) -> dict:
    fake = {"pl_ingest_sales": {"status": "Failed", "error": "2200 - invalid credentials", "duration_s": 42}}
    return fake.get(pipeline, {"status": "Unknown"})


messages = [{"role": "user", "content": "Did pl_ingest_sales succeed last night? If not, why?"}]

# 3) The agent loop
for _ in range(5):
    resp = client.chat.completions.create(model="gpt-4o-mini", messages=messages, tools=tools)
    msg = resp.choices[0].message
    messages.append(msg)
    if not msg.tool_calls:
        print(msg.content)
        break
    for call in msg.tool_calls:
        args = json.loads(call.function.arguments)
        result = get_pipeline_status(**args)  # dispatch (validate args in production!)
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
