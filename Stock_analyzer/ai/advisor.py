# ==============================
# ai/advisor.py
# ==============================
from openai import OpenAI
from config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY)

def ai_advice(data):
    prompt = f"Analyze stock data and give BUY/HOLD/SELL:\n{data}"

    res = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return res.choices[0].message.content