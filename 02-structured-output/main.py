from ast import main
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

# Define the exact shape we want back
class PersonInfo(BaseModel):
    name: str
    role: str
    years_experience: int

# ===============================
# Structured Response Message
def main() -> None:
    text_to_parse: str = "Ziad is a Senior Software Engineer with 18 years of experience."
    load_dotenv()
    client = OpenAI()

    prompt = f"Extract the person's info from: {text_to_parse}"

    print(f"Sending message: {prompt}")

    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ],
        response_format=PersonInfo,  # Tell the API to fill this shape
    )

    person = response.choices[0].message.parsed  # Already a validated PersonInfo object

    print(person) # PersonInfo(name='Ziad', role='Senior Software Engineer', years_experience=18)
    print(person.name) # Ziad
    print(person.years_experience + 1) # 19

if __name__ == "__main__":
    main()

# Example of the model response:
# main⚡ ⇒ uv run 02-structured-output/main.py
# Sending message: Extract the person's info from: Ziad is a Senior Software Engineer with 18 years of experience.
# name='Ziad' role='Senior Software Engineer' years_experience=18
# Ziad
# 19