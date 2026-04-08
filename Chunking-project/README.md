# Chunking-project

This folder contains small Streamlit apps and helpers that use the Groq API. To avoid committing secrets, the projects read the Groq API key from the environment variable `GROQ_API_KEY`.

Quick setup
1. Create a virtual environment and install dependencies (see `requirements.txt`).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy the example env file and add your Groq API key:

```bash
cp .env.example .env
# then edit .env and set GROQ_API_KEY=your_real_key
```

Alternatively you can export the key in your shell:

```bash
export GROQ_API_KEY="your_real_key"
```

Notes
- The apps prefer `GROQ_API_KEY` from the environment and allow overriding via the Streamlit sidebar where implemented.
- `.env` is ignored by `.gitignore`; never commit your real key.

Running the Streamlit apps

```bash
streamlit run app.py
streamlit run app-png.py
```

If you don't have `python-dotenv` installed, the apps still read environment variables exported in your shell.
