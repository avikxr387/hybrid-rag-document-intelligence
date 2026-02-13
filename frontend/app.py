

import streamlit as st
import sys
from pathlib import Path
import time

# -----------------------------
# Backend Import
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR / "backend"))

from rag_pipeline import RAGSystem


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="HybridRAG",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Styling
# -----------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1.5rem;
}

.chat-user {
    background-color: #1f2937;
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 10px;
}

.chat-bot {
    background-color: #111827;
    padding: 14px;
    border-radius: 10px;
    border: 1px solid #2a2f3a;
    margin-bottom: 15px;
}

.source-box {
    background-color: #0b0f14;
    padding: 8px;
    border-radius: 6px;
    border: 1px solid #2a2f3a;
    margin-bottom: 6px;
    font-size: 14px;
    color: #cbd5e1;
}

.confidence {
    font-size: 13px;
    color: #9ca3af;
    margin-top: 5px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.title("HybridRAG")
st.caption("Hybrid Retrieval Augmented Document Intelligence")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Documents")

docs_path = BASE_DIR / "docs"
docs_path.mkdir(exist_ok=True)

uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    type=["pdf", "docx", "csv"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        with open(docs_path / file.name, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("Files uploaded")

# -----------------------------
# Initialize System
# -----------------------------
@st.cache_resource
def load_rag():
    return RAGSystem(str(docs_path))

if st.sidebar.button("Initialize System"):
    with st.spinner("Building indexes..."):
        st.session_state.rag = load_rag()
    st.sidebar.success("System ready")

# -----------------------------
# Chat History
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Query Input
# -----------------------------
query = st.chat_input("Ask something about your documents...")

if query and "rag" in st.session_state:
    # Save user message
    st.session_state.chat_history.append(("user", query))

    # Loading animation
    with st.spinner("Thinking..."):
        time.sleep(0.3)
        answer, sources = st.session_state.rag.answer_question(query)

        # Simple confidence score
        confidence = min(len(sources) / 3, 1.0)

    # Save bot response
    st.session_state.chat_history.append(("bot", answer, sources, confidence))

# -----------------------------
# Display Chat
# -----------------------------
for msg in st.session_state.chat_history:
    if msg[0] == "user":
        st.markdown(
            f'<div class="chat-user"><b>You:</b><br>{msg[1]}</div>',
            unsafe_allow_html=True
        )

    elif msg[0] == "bot":
        answer = msg[1]
        sources = msg[2]
        confidence = msg[3]

        st.markdown(
            f'<div class="chat-bot"><b>HybridRAG:</b><br>{answer}</div>',
            unsafe_allow_html=True
        )

        # Confidence
        st.markdown(
            f'<div class="confidence">Confidence: {confidence:.2f}</div>',
            unsafe_allow_html=True
        )

        # Source Preview
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(
                        f'<div class="source-box">{src}</div>',
                        unsafe_allow_html=True
                    )

# -----------------------------
# Warning if not initialized
# -----------------------------
if "rag" not in st.session_state:
    st.info("Upload documents and initialize the system from the sidebar.")
