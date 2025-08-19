import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response= client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "you are Sanuwar's assistnat."},
        {"role": "user", "content": "Hello! Can you briefly introduce yourself?"}
    ]
)

print (response.choices[0].message.content)