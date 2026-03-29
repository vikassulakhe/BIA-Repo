import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader
import docx
import base64

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("🧠 Multimodal RAG Chatbot (Text + Image)")

# -----------------------------
# MODEL LOAD
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------
# CHROMA DB
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

# -----------------------------
# IMAGE → LLM (VISION)
# -----------------------------
def read_image_with_llm(file, api_key):
    client = Groq(api_key=api_key)

    image_bytes = file.read()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image in detail and extract all useful information"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=500
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Image processing failed: {str(e)}"

# -----------------------------
# EXTRACT TEXT
# -----------------------------
def extract_text(file, api_key):
    if file.type == "text/plain":
        return read_txt(file)

    elif file.type == "application/pdf":
        return read_pdf(file)

    elif "word" in file.type:
        return read_docx(file)

    elif file.type in ["image/png", "image/jpeg"]:
        return read_image_with_llm(file, api_key)

    return ""

# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(text, chunk_size=300, overlap=80):
    chunks = []
    start = 0

    while start < len(text):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

# -----------------------------
# STORE IN DB
# -----------------------------
def store_in_db(chunks):
    if not chunks:
        st.error("❌ No valid chunks to store")
        return

    embeddings = model.encode(chunks)

    if len(embeddings) == 0:
        st.error("❌ Embeddings empty")
        return

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"id_{i}" for i in range(len(chunks))]
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
# VALIDATE API KEY
# -----------------------------
def validate_api_key(api_key):
    try:
        client = Groq(api_key=api_key)
        client.chat.completions.create(
            messages=[{"role": "user", "content": "test"}],
            model="llama-3.3-70b-versatile",
            max_tokens=1
        )
        return True
    except:
        return False

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🔐 Settings")

api_key = st.sidebar.text_input("Enter Groq API Key", type="password")

if not api_key:
    st.warning("Enter API key")
    st.stop()

if "valid" not in st.session_state:
    with st.spinner("Validating key..."):
        if validate_api_key(api_key):
            st.session_state.valid = True
            st.sidebar.success("✅ Valid API Key")
        else:
            st.sidebar.error("❌ Invalid API Key")
            st.stop()

# -----------------------------
# GENERATE ANSWER
# -----------------------------
def generate_answer(query, context, api_key):
    client = Groq(api_key=api_key)

    prompt = f"""
You are a RAG assistant.

Rules:
- Answer ONLY from context
- If not found, say "I don't know"

Context:
{context}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content

# -----------------------------
# FILE UPLOAD
# -----------------------------
files = st.file_uploader(
    "Upload files (.txt, .pdf, .docx, .png, .jpg)",
    type=["txt", "pdf", "docx", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

if files:
    all_text = ""

    for f in files:
        text = extract_text(f, api_key)

        if text and text.strip():
            all_text += text + "\n"
        else:
            st.warning(f"No usable text in {f.name}")

    chunks = chunk_text(all_text)

    if chunks:
        store_in_db(chunks)
        st.success("✅ Data stored in vector DB")
    else:
        st.error("❌ No valid content found")

# -----------------------------
# CHAT
# -----------------------------
if "chat" not in st.session_state:
    st.session_state.chat = []

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask something...")

if query:
    st.session_state.chat.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        retrieved = retrieve(query)
        context = "\n".join(retrieved)

        with st.expander("🔍 Context"):
            st.write(context)

        answer = generate_answer(query, context, api_key)
        st.write(answer)

    st.session_state.chat.append({"role": "assistant", "content": answer})