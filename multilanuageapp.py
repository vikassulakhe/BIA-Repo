import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# -----------------------------------
# Load API Key
# -----------------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# -----------------------------------
# Initialize LLM
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
# Prompts
# -----------------------------------
english_prompt = PromptTemplate.from_template(
    "You are a helpful assistant. Reply in English.\nUser: {text}"
)

hindi_prompt = PromptTemplate.from_template(
    "आप एक सहायक हैं। हिंदी में उत्तर दें।\nप्रश्न: {text}"
)

spanish_prompt = PromptTemplate.from_template(
    "Eres un asistente útil. Responde en español.\nUsuario: {text}"
)

# -----------------------------------
# Chains
# -----------------------------------
english_chain = english_prompt | llm | parser
hindi_chain = hindi_prompt | llm | parser
spanish_chain = spanish_prompt | llm | parser

# -----------------------------------
# LLM Language Detection
# -----------------------------------
def detect_language_llm(text):
    prompt = f"""
    Detect language. Only return: hi, es, or en.

    Text: {text}
    """
    response = llm.invoke(prompt)
    # `invoke` may return an object with `content` or a raw string.
    text = getattr(response, "content", response) or ""
    return str(text).strip().lower()

# -----------------------------------
# Router
# -----------------------------------
def route_chain(user_input):
    try:
        lang = detect_language_llm(user_input)
    except:
        lang = "en"

    if "hi" in lang:
        return hindi_chain
    elif "es" in lang:
        return spanish_chain
    else:
        return english_chain

# -----------------------------------
# Stream Function
# -----------------------------------
def stream_response(chain, user_input):
    full_response = ""
    for chunk in chain.stream({"text": user_input}):
        full_response += chunk
        yield full_response

# -----------------------------------
# Streamlit UI
# -----------------------------------
st.set_page_config(page_title="Multilingual Chatbot", page_icon="🌍")

st.title("🌍 Multilingual Chatbot (LangChain + Groq)")
st.write("Supports English, Hindi, Spanish")

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input box
user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Get chain
    chain = route_chain(user_input)

    # Bot response (streaming)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        full_response = ""
        for chunk in chain.stream({"text": user_input}):
            full_response += chunk
            response_placeholder.markdown(full_response)

    # Save bot response
    st.session_state.messages.append({"role": "assistant", "content": full_response})