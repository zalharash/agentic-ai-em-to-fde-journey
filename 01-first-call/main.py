import os
from dotenv import load_dotenv
from openai import OpenAI

# unstructured_response_message
def main() -> None:
    load_dotenv()  # reads .env into the environment

    # client = OpenAI()  # automatically picks up OPENAI_API_KEY from the environment
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )

    prompt = "In one sentence, what is a Forward Deployed Engineer?"
    print(f"Sending message: {prompt}")

    response = client.chat.completions.create(
        # model="gpt-4o-mini",
        model="deepseek-flash",
        messages=[
            {"role": "user", "content": prompt}
        ],
        stream=False,
        max_tokens=500,
        reasoning_effort="medium",
        extra_body={"thinking": {"type": "disabled"}}
    )

    message = response.choices[0].message
    # print(f"Reasoning: {getattr(message, 'reasoning_content', None)}")
    answer = message.content
    print(f"Received answer: {answer}")

# only run this block when the file is executed directly, not when it's imported
if __name__ == "__main__":
    main()
