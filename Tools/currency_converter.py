import os
import json
import requests
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# ------------------------------------------------------------------
# 1. Load .env and tokens
# ------------------------------------------------------------------
load_dotenv()   # Loads variables from your .env file

HF_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
EXCHANGERATE_API_KEY = os.environ.get("EXCHANGERATE_API_KEY")

if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file.")
if not EXCHANGERATE_API_KEY:
    raise ValueError("EXCHANGERATE_API_KEY not found in .env file.")

# ------------------------------------------------------------------
# 2. Define tools
# ------------------------------------------------------------------
@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> dict:
    """Fetch the currency conversion factor between a base currency and a target currency."""
    # Using your actual API key from the .env file
    url = (
        f"https://v6.exchangerate-api.com/v6/"
        f"{EXCHANGERATE_API_KEY}/pair/{base_currency}/{target_currency}"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()


@tool
def convert(base_currency_value: float, conversion_rate: float) -> float:
    """Given a currency conversion rate, calculate the target currency value
    from a base currency value."""
    return base_currency_value * conversion_rate


tools = [get_conversion_factor, convert]
TOOL_MAP = {t.name: t for t in tools}

# ------------------------------------------------------------------
# 3. Load Hugging Face model + bind tools
# ------------------------------------------------------------------
llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b:cerebras",   # <-- CHANGED: supported model + provider
    task="text-generation",
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=512,
    temperature=0.1,
)

chat_model = ChatHuggingFace(llm=llm)
llm_with_tools = chat_model.bind_tools(tools)

# ------------------------------------------------------------------
# 4. Agent loop (handles multi-turn tool calling)
# ------------------------------------------------------------------
messages = [HumanMessage(content="Convert 10 USD to INR")]

conversion_rate = None
MAX_STEPS = 10

for step in range(MAX_STEPS):
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)

    # If no more tool calls, the model is ready to give the final answer
    if not getattr(ai_message, "tool_calls", None):
        break

    for tool_call in ai_message.tool_calls:
        name = tool_call["name"]
        args = dict(tool_call["args"])

        # Inject the rate we got from the previous tool call
        if name == "convert":
            if conversion_rate is None:
                fallback = get_conversion_factor.invoke(
                    {"base_currency": "USD", "target_currency": "INR"}
                )
                conversion_rate = json.loads(fallback.content)["conversion_rate"]
            args["conversion_rate"] = conversion_rate

        result = TOOL_MAP[name].invoke(args)

        # Capture conversion_rate when get_conversion_factor runs
        if name == "get_conversion_factor":
            payload = result if isinstance(result, dict) else json.loads(result.content)
            conversion_rate = payload["conversion_rate"]

        messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

# ------------------------------------------------------------------
# 5. Final answer
# ------------------------------------------------------------------
final = messages[-1]
print("\n=== Final Answer ===")
print(final.content if hasattr(final, "content") else final)