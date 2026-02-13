# HybridRAG: Intelligent Document Retrieval with Hybrid Search and Hallucination Control

A Retrieval-Augmented Generation (RAG) system that enables accurate question answering over multiple documents using **hybrid retrieval (semantic + keyword search)**, **cross-encoder reranking**, and **confidence-based hallucination prevention**.

This project focuses on **retrieval quality, transparency, and reliability**, rather than simple “chat with PDF” functionality.

---

## Overview

Traditional document chat systems rely only on semantic search, which can miss exact matches or retrieve irrelevant context.
This system improves accuracy by combining:

* Dense semantic retrieval (embeddings)
* Sparse keyword retrieval (BM25)
* Cross-encoder reranking
* Source diversity control
* Confidence-based answer validation

The result is a more reliable and interpretable document question-answering pipeline.

---

## Features

### Hybrid Retrieval

* Semantic search using **Sentence Transformers**
* Keyword search using **BM25**
* Combined candidate pool for better recall

### Cross-Encoder Reranking

* Query–document relevance scoring
* Improves final context quality

### Hallucination Control

* Answers generated **only when sufficient document support exists**
* Otherwise returns: *“Information not found in the documents.”*

### Source Transparency

* Retrieval report showing document distribution
* Final answers include source documents

### Context Diversity

* Limits number of chunks per document to avoid source dominance

### Multi-Document Support

Supports:

* PDF
* DOCX
* TXT

---

## Project Structure

```
HYBRIDRAG/
│
├── backend/
│   ├── rag_pipeline.py      # Main RAG system
│   ├── retriever.py         # Semantic, BM25, hybrid, reranking
│   ├── utils.py             # Loading, chunking, reporting, checks
│   └── run.py               # CLI entry point
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
git clone https://github.com/<your-username>/hybridrag.git
cd hybridrag
```

### 2. Create virtual environment (recommended)

```
python -m venv venv
venv\Scripts\activate    # Windows
source venv/bin/activate # Linux/Mac
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## Usage

### Add documents

Place your files inside:

```
docs/
```

Supported formats: `.pdf`, `.docx`, `.txt`

---

### Run the system

```
cd backend
python run.py
```

You will see:

```
HybridRAG System
System ready!
Question:
```

Ask any question related to your documents.

Type `exit` to quit.

---

## Example

**Query**

```
What items are restricted during air travel?
```

**Output**

```
===== ANSWER =====
Sharp objects and flammable materials are prohibited in cabin baggage.

===== SOURCES =====
- Baggage Guidelines.pdf
```

If information is not present:

```
Information not found in the documents.
```

---

## Retrieval Pipeline

1. Load and chunk documents
2. Build:

   * Semantic index (MiniLM embeddings)
   * BM25 keyword index
3. Retrieve top candidates from both
4. Merge and remove duplicates
5. Cross-encoder reranking
6. Limit chunks per source
7. Confidence check
8. Generate grounded answer using FLAN-T5

---

## Evaluation

The system includes:

* Retrieval source distribution reports
* Recall@k comparison (semantic vs BM25 vs hybrid)
* Confidence-based answer filtering

These components help analyze retrieval quality and reduce hallucinations.

---

## Technologies Used

* Python
* Sentence Transformers
* HuggingFace Transformers (FLAN-T5)
* ChromaDB
* BM25 (rank-bm25)
* LangChain (loaders & text splitting)

---

## Future Improvements

* Web interface for document upload and querying
* REST API (FastAPI)
* Larger embedding models
* Persistent vector storage
* Advanced evaluation metrics

---

## Why This Project?

Most document chat systems focus only on UI.
This project focuses on:

* Retrieval optimization
* Model grounding
* Transparency
* Reliability

It demonstrates practical design considerations for real-world RAG systems.

---

## Author

AVIK HALDER GitHub: [https://github.com/avikxr387]

