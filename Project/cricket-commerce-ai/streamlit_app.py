import streamlit as st
from main import orchestrator

st.set_page_config(page_title="Cricket Commerce AI", page_icon="🏏")

st.title("🏏 Cricket Commerce Support Agent")

if "chat" not in st.session_state:
    st.session_state.chat = []

user_input = st.chat_input("Ask about bats, orders, or refunds")

if user_input:

    response = orchestrator(user_input)

    st.session_state.chat.append(("user", user_input))
    st.session_state.chat.append(("assistant", response))

for role, msg in st.session_state.chat:

    with st.chat_message(role):
        st.write(msg)