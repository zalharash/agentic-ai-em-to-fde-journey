from ast import main
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import json
import random

# Define the exact shape we want back
class PersonInfo(BaseModel):
    name: str
    role: str
    years_experience: int

# A fake tool/function that passed to the model, the model never touches it, it just calls it and get the result.
def get_weather(city: str) -> str:
    fake_data = {
        "Stockholm": "9°C, cloudy",
        "Malmö": "11°C, rainy",
    }
    return fake_data.get(city, "unknown city")


# Describe that function to the model, in a schema it understands so we can act on its response instead of getting text response that can't be handled to branch the logic
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            },
        },
    }
]

# ===============================
#          Tool Calling
# the exact mechanism every agent framework wraps in convenience the model can't act, only request. The code stayed in control the entire time, I could have inspected call.function.arguments, rejected it, logged it, or asked a human before running it.
# calling_tools_once
def main() -> None:
    load_dotenv()
    client = OpenAI()

    messages = [{"role": "user", "content": "What's the weather in Stockholm right now?"}]

    # First call, give the model the question AND the list of tools it may use, and leave it to the LLM model to decide if that tool is needed to answer the prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=tools,
    )

    reply = response.choices[0].message

    if reply.tool_calls:
        call = reply.tool_calls[0]
        print(f"Model wants to call: {call.function.name}({call.function.arguments})")

        # My code decides to wether run the asked tool or not, not the LLM model. The LLM model just sent back that this tools is needed.
        args = json.loads(call.function.arguments)
        result = get_weather(**args)
        print(f"Function returned: {result}")

        # Send the result back so the model so it can form a real answer
        messages.append(reply)  # the model's tool request
        messages.append({
            "role": "tool",
            "tool_call_id": call.id,
            "content": result,
        })

        final = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        print(final.choices[0].message.content)
    else:
        print(reply.content)

if __name__=="__main__":
    main()



# Example of the model response:
# main⚡ ⇒ uv run 03-tool-calling/main.py
# Model wants to call: get_weather({"city":"Stockholm"})
# Function returned: 9°C, cloudy
# The current weather in Stockholm is 9°C and cloudy.