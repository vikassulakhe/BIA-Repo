#pip install -U langchain langchain-core langchain-groq langdetect python-dotenv

import os
from dotenv import load_dotenv
from langdetect import detect

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------
# Load API Key
# -----------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# -----------------------------------
# Initialize LLM (Streaming Enabled)
# -----------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.3,
    streaming=True
)

# -----------------------------------
# Output Parser
# -----------------------------------
parser = StrOutputParser()

# -----------------------------------
# Prompt Templates
# -----------------------------------
english_prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Reply naturally in English.\nUser: {text}"
)

hindi_prompt = PromptTemplate.from_template(
    "आप एक सहायक हैं। स्वाभाविक हिंदी में उत्तर दें।\nप्रश्न: {text}"
)

spanish_prompt = PromptTemplate.from_template(
    "Eres un asistente útil. Responde naturalmente en español.\nUsuario: {text}"
)
# -----------------------------------
# Chains (LCEL)
# -----------------------------------
english_chain = english_prompt | llm | parser
hindi_chain = hindi_prompt | llm | parser
spanish_chain = spanish_prompt | llm | parser

# -----------------------------------
# Language Router
# -----------------------------------
def route_chain(user_input):
    text = user_input.lower()

    # Rule-based overrides (handles mixed language)
    if any(word in text for word in ["hola", "gracias", "adios"]):
        return spanish_chain

    if any(char in text for char in "अआइईउऊएऐओऔकखगघचछजझ"):
        return hindi_chain

    # Fallback to detection
    try:
        lang = detect(user_input)
    except:
        lang = "en"

    if lang == "hi":
        return hindi_chain
    elif lang == "es":
        return spanish_chain
    else:
        return english_chain

# -----------------------------------
# Invoke (Normal Response)
# -----------------------------------
def chat_invoke(user_input):
    chain = route_chain(user_input)
    response = chain.invoke({"text": user_input})
    return response

# -----------------------------------
# Streaming Response
# -----------------------------------
def chat_stream(user_input):
    chain = route_chain(user_input)

    print("Bot: ", end="", flush=True)
    for chunk in chain.stream({"text": user_input}):
        print(chunk, end="", flush=True)
    print()

# -----------------------------------
# CLI Interface
# -----------------------------------
if __name__ == "__main__":
    print("🌍 Multilingual Chatbot (type 'exit' to quit)")
    print("Type 'stream: your text' for streaming mode\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        # Streaming mode
        if user_input.startswith("stream:"):
            text = user_input.replace("stream:", "").strip()
            chat_stream(text)

        # Normal invoke mode
        else:
            response = chat_invoke(user_input)
            print("Bot:", response)