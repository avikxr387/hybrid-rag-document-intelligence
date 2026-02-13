from transformers import pipeline
from retriever import SemanticIndex, BM25Index, Reranker, HybridRetriever
from utils import (
    load_documents,
    split_documents,
    retrieval_report,
    relevance_check,
    limit_per_source
)

class RAGSystem:
    def __init__(self, data_path):
        # Load data
        docs = load_documents(data_path)
        chunks = split_documents(docs)

        # Build indexes
        self.semantic = SemanticIndex()
        self.semantic.build(chunks)

        self.bm25 = BM25Index()
        self.bm25.build(chunks)

        self.reranker = Reranker()
        self.hybrid = HybridRetriever(self.semantic, self.bm25, self.reranker)

        # QA model
        self.qa_model = pipeline(
            "text-generation",
            model="google/flan-t5-base"
        )

    def answer_question(self, query):
        results = self.hybrid.retrieve(query)
        results = limit_per_source(results, max_per_source=3)

        retrieval_report(results)

        if not relevance_check(results):
            return "Information not found in the documents.", []

        context = "\n".join([r["text"] for r in results])

        prompt = f"""
Answer using only the context below.
If the answer is not present, say: Information not found.

Context:
{context}

Question: {query}
"""

        answer = self.qa_model(prompt, max_length=200)[0]["generated_text"]
        sources = list(set([r["metadata"]["source"] for r in results]))

        return answer, sources
