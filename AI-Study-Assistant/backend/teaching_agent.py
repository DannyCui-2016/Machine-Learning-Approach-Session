import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_teaching(question):

    prompt = f"""
You are a kind primary school math teacher.

Rules:
1. Do NOT give final answer directly.
2. Teach step by step.
3. After each step ask: Do you understand?
4. Use simple child-friendly language.

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt}
        ]
    )

    return response.choices[0].message.content
