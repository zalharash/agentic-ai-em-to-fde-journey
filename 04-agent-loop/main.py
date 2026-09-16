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
    # Here we are returning a random error that implies a retry for the LLM. In real world, we would call a real API and get the result.
    return random.choice([
        "ERROR: timeout, please retry",
        "ERROR: rate limited, try again",
    ])
    # fake_data = {
    #     "Stockholm": "9°C, cloudy",
    #     "Malmö": "11°C, rainy",
    # }
    # return fake_data.get(city, "unknown city")


def get_forecast(city: str, days: int) -> str:
    return f"{city} will be similar for the next {days} days: mild and cloudy."


# Describe that functions/tools to the model, in a schema it understands
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
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": "Get a multi-day forecast for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                },
                "required": ["city", "days"],
            },
        },
    },
]


def run_agent(question: str, max_steps: int = 4) -> None:
    load_dotenv()
    client = OpenAI()
    FUNCTIONS = {"get_weather": get_weather, "get_forecast": get_forecast}
    messages = [{"role": "user", "content": question}]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, tools=tools,
        )
        reply = response.choices[0].message

        if not reply.tool_calls:
            print(f"[done after {step} tool calls]")
            print(reply.content)
            return

        messages.append(reply)
        for call in reply.tool_calls:
            args = json.loads(call.function.arguments)
            print(f"Step {step}: calling {call.function.name}({args})")
            result = FUNCTIONS[call.function.name](**args)
            messages.append({
                "role": "tool", "tool_call_id": call.id, "content": str(result),
            })

    print("[stopped: hit max_steps without finishing]")


def main() -> None:
    # run_agent("Should I bring a jacket to Stockholm this week?", max_steps=6)
    run_agent("What's the weather on Stockholm?", max_steps=4)


if __name__ == "__main__":
    main()

# Example of the model response:
# main⚡ ⇒ uv run 04-agent-loop/main.py
# Step 0: calling get_weather({'city': 'Stockholm'})
# Step 1: calling get_weather({'city': 'Stockholm'})
# [done after 2 tool calls]
# I am currently unable to retrieve the weather information for Stockholm due to a temporary issue. Please try again later.