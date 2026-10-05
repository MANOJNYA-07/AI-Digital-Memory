import json
from pathlib import Path

import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="AI Digital Memory",
    page_icon="🧠",
    layout="wide"
)

KNOWLEDGE_FOLDER = Path("knowledge")
MEMORY_STORE_PATH = KNOWLEDGE_FOLDER / "memory_store.json"


@st.cache_resource
def get_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


def ensure_knowledge_folder():
    KNOWLEDGE_FOLDER.mkdir(exist_ok=True)


def load_saved_memory():
    if not MEMORY_STORE_PATH.exists():
        return []

    try:
        with MEMORY_STORE_PATH.open("r", encoding="utf-8") as f:
            memory = json.load(f)
        return memory if isinstance(memory, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def save_memory(memory_chunks):
    ensure_knowledge_folder()
    with MEMORY_STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump(memory_chunks, f, ensure_ascii=False, indent=2)


def clean_text(text):
    return " ".join(text.split())


def extract_pdf_text(file_path):
    reader = PdfReader(str(file_path))
    pages = []

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            pages.append(extracted)

    return "\n".join(pages)


def split_text_into_chunks(text, chunk_size=500):
    cleaned = clean_text(text)
    if not cleaned:
        return []

    return [
        cleaned[i:i + chunk_size]
        for i in range(0, len(cleaned), chunk_size)
    ]


def add_uploaded_memory(uploaded_file):
    ensure_knowledge_folder()

    file_path = KNOWLEDGE_FOLDER / uploaded_file.name
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if uploaded_file.name.lower().endswith(".pdf"):
        text = extract_pdf_text(file_path)
    elif uploaded_file.name.lower().endswith(".txt"):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
    else:
        st.warning("Unsupported file type. Please upload a PDF or TXT file.")
        return

    new_chunks = [
        {"id": len(st.session_state.memory_chunks) + idx, "text": chunk}
        for idx, chunk in enumerate(split_text_into_chunks(text))
    ]

    existing_chunks = st.session_state.get("memory_chunks", [])
    merged_chunks = existing_chunks + new_chunks

    seen_texts = set()
    unique_chunks = []
    for chunk in merged_chunks:
        chunk_text = chunk["text"].strip()
        if chunk_text and chunk_text not in seen_texts:
            seen_texts.add(chunk_text)
            unique_chunks.append(chunk)

    st.session_state.memory_chunks = unique_chunks
    save_memory(unique_chunks)

    return unique_chunks


def search_memory(question, memory_chunks, top_k=3):
    if not memory_chunks or not question:
        return []

    model = get_embedding_model()
    texts = [chunk["text"] for chunk in memory_chunks]
    question_vector = model.encode([question], show_progress_bar=False)
    memory_vectors = model.encode(texts, show_progress_bar=False)

    similarities = cosine_similarity(question_vector, memory_vectors)[0]
    ranked_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in ranked_indices:
        score = float(similarities[idx])
        if score <= 0:
            continue
        results.append({
            "score": score,
            "text": memory_chunks[int(idx)]["text"],
        })

    return results


def build_memory_answer(question, relevant_results):
    if not relevant_results:
        return (
            "I couldn't find a strong match for that question in your stored memory. "
            "Try uploading a document or ask about something that appears in it."
        )

    answer = "Based on your saved memory, here is the most relevant information:\n\n"
    for index, match in enumerate(relevant_results[:3], start=1):
        cleaned = match["text"].strip()
        answer += f"{index}. {cleaned}\n\n"

    return answer


# -----------------------------
# TITLE
# -----------------------------
st.title("🧠 AI-Powered Digital Memory")
st.write("Your personal AI system for storing and retrieving knowledge.")

if "memory_chunks" not in st.session_state:
    st.session_state.memory_chunks = load_saved_memory()

st.divider()

# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.header("📚 My Memory")

    uploaded_file = st.file_uploader(
        "Upload your knowledge",
        type=["pdf", "txt"]
    )

    st.info(
        "Upload notes, documents or other information "
        "you want your AI memory to remember."
    )

# -----------------------------
# MAIN PAGE
# -----------------------------
st.subheader("📤 Upload Your Knowledge")

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")

    with st.spinner("Processing your document and indexing it into memory..."):
        chunks = add_uploaded_memory(uploaded_file)

    if chunks:
        st.subheader("🧠 Memory Processing")
        st.metric("Memory Chunks", len(chunks))
        st.success("✅ Your document has been processed and saved in memory!")

        st.subheader("📖 Memory Preview")
        preview_text = "\n\n".join(item["text"] for item in chunks[:5])
        st.text_area("Extracted information", preview_text[:3000], height=300)
else:
    st.info("👈 Upload a PDF or TXT file from the sidebar to begin.")

# -----------------------------
# ASK YOUR MEMORY
# -----------------------------
st.divider()

st.subheader("🔍 Ask Your Digital Memory")

question = st.text_input(
    "What do you want to remember?",
    placeholder="Example: What did I save about Java?"
)

if st.button("🔍 Search Memory"):
    if not question:
        st.warning("Please enter a question.")
    else:
        memory_chunks = st.session_state.get("memory_chunks", [])
        if not memory_chunks:
            st.warning("Please upload a PDF or TXT file before asking a question.")
        else:
            with st.spinner("Searching your memory..."):
                relevant_results = search_memory(question, memory_chunks, top_k=3)

            st.markdown("### Answer")
            st.write(build_memory_answer(question, relevant_results))

            if relevant_results:
                st.markdown("### Best matches")
                for idx, result in enumerate(relevant_results, start=1):
                    st.write(f"{idx}. {result['text'][:400]}")