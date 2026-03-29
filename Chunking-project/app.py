import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader
import docx

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("📚 RAG Chatbot (ChromaDB)")

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# INIT CHROMA DB
# -----------------------------
@st.cache_resource
def init_db():
    client = chromadb.Client()
    return client.get_or_create_collection(name="rag_docs")

collection = init_db()

# -----------------------------
# FILE READERS
# -----------------------------
def read_txt(file):
    return file.read().decode("utf-8")

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def read_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text(file):
    if file.type == "text/plain":
        return read_txt(file)
    elif file.type == "application/pdf":
        return read_pdf(file)
    elif "word" in file.type:
        return read_docx(file)
    return ""

# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(text, chunk_size=300, overlap=80):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# -----------------------------
# STORE IN DB
# -----------------------------
def store_in_db(chunks):
    embeddings = model.encode(chunks).tolist()
    ids = [f"id_{i}" for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

# -----------------------------
# RETRIEVE
# -----------------------------
def retrieve(query, k=5):
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )

    return results["documents"][0]

# -----------------------------
# VALIDATE API KEY ✅
# -----------------------------
def validate_api_key(api_key):
    try:
        client = Groq(api_key=api_key)

        # small test call
        client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="llama-3.3-70b-versatile",
            max_tokens=1
        )
        return True
    except Exception:
        return False

# -----------------------------
# API KEY UI
# -----------------------------
st.sidebar.title("🔐 Settings")

groq_api_key = st.sidebar.text_input(
    "Enter Groq API Key",
    type="password"
)

# Validate once per session
if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key")
    st.stop()

if "api_valid" not in st.session_state:
    with st.spinner("Validating API key..."):
        if validate_api_key(groq_api_key):
            st.session_state.api_valid = True
            st.sidebar.success("✅ Valid API Key")
        else:
            st.sidebar.error("❌ Invalid API Key")
            st.stop()

# -----------------------------
# GENERATE ANSWER
# -----------------------------
def generate_answer(query, context, api_key):
    try:
        client = Groq(api_key=api_key)

        prompt = f"""
You are a helpful AI assistant.

Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload files (.txt, .pdf, .docx)",
    type=["txt", "pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    all_text = ""

    for file in uploaded_files:
        text = extract_text(file)
        all_text += text + "\n"

    st.success(f"{len(uploaded_files)} file(s) uploaded!")

    if "loaded" not in st.session_state:
        chunks = chunk_text(all_text)
        store_in_db(chunks)
        st.session_state.loaded = True
        st.success("✅ Stored in Vector DB")

# -----------------------------
# CHAT MEMORY
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------------
# CHAT INPUT
# -----------------------------
query = st.chat_input("Ask something...")

if query:
    if "loaded" not in st.session_state:
        st.warning("⚠️ Upload files first!")
    else:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retrieved = retrieve(query)
                context = "\n".join(retrieved)

                # Debug view
                with st.expander("🔍 Retrieved Context"):
                    st.write(context)

                answer = generate_answer(query, context, groq_api_key)
                st.write(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
        