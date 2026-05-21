# 🌍 Multilingual Chatbot (LangChain + OpenAI)

A Streamlit-based multilingual chatbot that supports **English**, **Hindi**, and **Spanish**. It uses LangChain with OpenAI's `gpt-4o-mini` model to automatically detect the user's language and respond accordingly.

---

## Features

- Auto language detection using the LLM itself
- Streaming responses for a real-time feel
- Persistent chat history within the session
- Clean Streamlit web UI
- Supports: English 🇬🇧 | Hindi 🇮🇳 | Spanish 🇪🇸

---

## Tech Stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI |
| [LangChain](https://python.langchain.com) | LLM orchestration (LCEL chains) |
| [OpenAI GPT-4o-mini](https://platform.openai.com) | Language model |
| [python-dotenv](https://pypi.org/project/python-dotenv/) | API key management |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/your-username/your-repo.git
cd Multi-lauguage-chatbot
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install streamlit langchain langchain-openai langchain-core python-dotenv
```

### 4. Add your OpenAI API key

Create a `.env` file in this folder (or copy `.env.example`):

```bash
cp .env.example .env
```

Then open `.env` and replace the placeholder:

```
OPENAI_API_KEY=your_openai_api_key_here
```

Get your key at [platform.openai.com/api-keys](https://platform.openai.com/api-keys).

> ⚠️ Never commit your `.env` file to Git. It is already listed in `.gitignore`.

### 5. Run the app

```bash
streamlit run multilanuageapp.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Project Structure

```
Multi-lauguage-chatbot/
├── multilanuageapp.py   # Main Streamlit app
├── .env                 # Your API key (not committed to Git)
├── .env.example         # Template for the .env file
└── README.md            # This file
```

---

## How It Works

1. User types a message in any supported language
2. The LLM detects the language (`en`, `hi`, or `es`)
3. The appropriate LangChain prompt chain is selected
4. A streaming response is returned and displayed in real time

---

## License

MIT
