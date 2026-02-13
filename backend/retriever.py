from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class SemanticIndex:
    def __init__(self, persist_dir="db"):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("docs")
    
    def build(self, docs):
        texts = [d.page_content for d in docs]
        metadatas = [d.metadata for d in docs]
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        ids = [str(i) for i in range(len(texts))]
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def search(self, query, k=5):
        q_emb = self.model.encode([query])[0]
        
        results = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k
        )
        
        docs = []
        for i in range(len(results["documents"][0])):
            docs.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })
        
        return docs

class BM25Index:
    def build(self, docs):
        self.texts = [d.page_content for d in docs]
        self.metadatas = [d.metadata for d in docs]
        tokenized = [t.split() for t in self.texts]
        self.bm25 = BM25Okapi(tokenized)
    
    def search(self, query, k=5):
        scores = self.bm25.get_scores(query.split())
        top_k = np.argsort(scores)[-k:][::-1]
        
        results = []
        for idx in top_k:
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadatas[idx]
            })
        
        return results

class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    def rerank(self, query, docs, top_k=3):
        pairs = [(query, d["text"]) for d in docs]
        scores = self.model.predict(pairs)
        
        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        return [r[0] for r in ranked[:top_k]]
    
class HybridRetriever:
    def __init__(self, semantic_index, bm25_index, reranker):
        self.semantic = semantic_index
        self.bm25 = bm25_index
        self.reranker = reranker
    
    def retrieve(self, query, k=5):
        sem = self.semantic.search(query, k)
        kw = self.bm25.search(query, k)
        
        combined = sem + kw
        
        # Remove duplicates
        unique = []
        seen = set()
        for d in combined:
            if d["text"] not in seen:
                unique.append(d)
                seen.add(d["text"])
        
        return self.reranker.rerank(query, unique, top_k=k)
        # return unique[:3]

