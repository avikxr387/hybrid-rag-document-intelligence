import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(folder_path):
    all_docs = []
    
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            docs = loader.load()
        elif file.endswith(".docx"):
            loader = Docx2txtLoader(path)
            docs = loader.load()
        elif file.endswith(".csv"):
            loader = CSVLoader(path)
            docs = loader.load()
        else:
            continue
        
        for d in docs:
            d.metadata["source"] = file
        
        all_docs.extend(docs)
    
    return all_docs

def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)

from collections import Counter

def retrieval_report(retrieved_docs):
    """
    Shows how many chunks came from each source document
    """
    sources = [d["metadata"]["source"] for d in retrieved_docs]
    count = Counter(sources)
    
    print("\n===== RETRIEVAL REPORT =====")
    total = len(retrieved_docs)
    
    for src, c in count.items():
        percent = (c / total) * 100
        print(f"{src}: {c} chunks ({percent:.1f}%)")
    
    print("Total retrieved chunks:", total)

from collections import Counter

def relevance_check(results, min_support=2):
    """
    Returns True if enough chunks support the answer.
    Prevents hallucination when retrieval is weak.
    """
    if len(results) == 0:
        return False
    
    sources = [d["metadata"]["source"] for d in results]
    count = Counter(sources)
    top_count = count.most_common(1)[0][1]
    
    return top_count >= min_support

def limit_per_source(docs, max_per_source=3):
    result = []
    count = {}
    
    for d in docs:
        src = d["metadata"]["source"]
        if count.get(src, 0) < max_per_source:
            result.append(d)
            count[src] = count.get(src, 0) + 1
    
    return result
