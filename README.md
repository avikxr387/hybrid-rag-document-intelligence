# HybridRAG: Intelligent Document Retrieval with Hybrid Search and Reliability Controls

HybridRAG is an end-to-end Retrieval-Augmented Generation (RAG) system for accurate question answering over multiple documents.
The system focuses on **retrieval quality, grounding, and transparency** using hybrid search, cross-encoder reranking, and confidence-based hallucination control.

Unlike basic “chat with PDF” tools, this project emphasizes **reliability, interpretability, and practical system design** for real-world document intelligence.

---

## Overview

Traditional document QA systems rely only on semantic search, which may miss exact keyword matches or retrieve weak context.

HybridRAG improves accuracy by combining:

* Dense semantic retrieval (embeddings)
* Sparse keyword retrieval (BM25)
* Cross-encoder reranking
* Source diversity control
* Confidence-based answer validation
* Source transparency

The result is a more robust and trustworthy document question-answering pipeline.

---

## Key Features

### Hybrid Retrieval

* Semantic search using **Sentence Transformers**
* Keyword search using **BM25**
* Combined candidate pool for improved recall

### Cross-Encoder Reranking

* Query–document relevance scoring
* Improves final context quality before generation

### Hallucination Control

* Answers generated **only when sufficient supporting context exists**
* Otherwise returns:

  *“Information not found in the documents.”*

### Source Transparency

* Displays source documents for each answer
* Expandable source previews in the web interface

### Context Diversity

* Limits number of chunks per document
* Prevents dominance from a single source

### Multi-Document Support

Supports:

* PDF
* DOCX
* CSV

---

## Web Interface

The system includes a **Streamlit-based web application** for interactive document querying.

### Features

* Chat-style interface for question answering
* Multi-document upload from sidebar
* Real-time Hybrid RAG retrieval and response generation
* Expandable source previews for transparency
* Confidence score based on retrieval strength
* Session-based chat history
* Clean dark-mode layout for professional usability

### Run the Web App

From the project root:

```
streamlit run frontend/app.py
```

Then open the local URL shown in the terminal.

---

## Project Structure

```
HYBRIDRAG/
│
├── backend/
│   ├── rag_pipeline.py      # Main RAG pipeline
│   ├── retriever.py         # Semantic, BM25, hybrid retrieval & reranking
│   ├── utils.py             # Document loading, chunking, evaluation utilities
│   └── run.py               # CLI interface
│
├── frontend/
│   └── app.py               # Streamlit web interface
│
├── docs/                    # User documents
│
├── notebooks/
│   └── hybridrag.ipynb      # Experiments and development
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

### 1. Clone the repository

```
git clone https://github.com/avikxr387/hybrid-rag-document-intelligence.git
cd hybrid-rag-document-intelligence
```

### 2. Create a virtual environment (recommended)

Windows:

```
python -m venv venv
venv\Scripts\activate
```

Linux / Mac:

```
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Usage

### Option 1 — Web Interface (Recommended)

```
streamlit run frontend/app.py
```

Workflow:

1. Upload documents from the sidebar
2. Click **Initialize System**
3. Ask questions in the chat interface
4. View answers along with sources and confidence score

---

### Option 2 — Command Line Interface

Place documents inside:

```
docs/
```

Then run:

```
cd backend
python run.py
```

Type your questions directly in the terminal.

---

## Retrieval Pipeline

1. Load and chunk documents
2. Build:

   * Semantic index (MiniLM embeddings)
   * BM25 keyword index
3. Retrieve top candidates from both methods
4. Merge and remove duplicates
5. Apply cross-encoder reranking
6. Limit chunks per source (diversity control)
7. Perform confidence check
8. Generate grounded answer using **FLAN-T5**

---

## Evaluation and Analysis

The system includes utilities for:

* Retrieval source distribution reports
* Recall@k comparison (Semantic vs BM25 vs Hybrid)
* Confidence-based answer filtering

These components help analyze retrieval quality and reduce hallucinations.

---

## Technologies Used

* Python
* Sentence Transformers
* HuggingFace Transformers (FLAN-T5)
* Cross-Encoder (MS-MARCO MiniLM)
* ChromaDB
* BM25 (rank-bm25)
* LangChain (document loaders & text splitting)
* Streamlit (web interface)

---

## Future Improvements

* FastAPI-based REST backend
* Persistent vector database
* Conversation memory across sessions
* Advanced evaluation metrics (MRR, nDCG)
* Cloud deployment

---

## Why This Project?

Most document QA projects focus primarily on UI.
This project focuses on:

* Retrieval optimization
* Grounded generation
* Transparency and interpretability
* Reliability in real-world scenarios

It demonstrates practical system design for production-oriented RAG applications.

---

## Author

**AVIK HALDER**
GitHub: [https://github.com/avikxr387](https://github.com/avikxr387)

