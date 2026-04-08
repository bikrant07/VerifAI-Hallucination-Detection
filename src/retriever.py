import json
import os
import pandas as pd
import faiss
import numpy as np
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

class SciQRetriever:
    def __init__(self, 
                 facts_path="data/processed/combined_edu_kb.jsonl", 
                 index_dir="data/index",
                 model_name="all-MiniLM-L6-v2"):
        self.facts_path = facts_path
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "edu_combined_faiss.index")
        self.facts_meta_path = os.path.join(index_dir, "edu_combined_meta.jsonl")
        
        # We will load the model lazily if needed
        self.model_name = model_name
        self.model = None
        self.index = None
        self.facts = []
        
        os.makedirs(self.index_dir, exist_ok=True)
        
    def _load_model(self):
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
    def build_index(self):
        """Reads facts, encodes them, and builds a FAISS index."""
        print("Building vector index...")
        self._load_model()
        
        facts = []
        with open(self.facts_path, 'r', encoding='utf-8') as f:
            for line in f:
                facts.append(json.loads(line))
        
        self.facts = facts
        texts = [fact['text'] for fact in facts]
        
        print(f"Encoding {len(texts)} facts...")
        embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Save index and metadata
        faiss.write_index(self.index, self.index_path)
        with open(self.facts_meta_path, 'w', encoding='utf-8') as f:
            for fact in facts:
                f.write(json.dumps(fact) + '\n')
                
        print(f"Index built and saved to {self.index_path}")
        
    def load_index(self):
        """Loads an existing FAISS index and metadata."""
        if not os.path.exists(self.index_path) or not os.path.exists(self.facts_meta_path):
            print("Index not found. Building a new one...")
            self.build_index()
            return

        print("Loading existing vector index...")
        self.index = faiss.read_index(self.index_path)
        
        self.facts = []
        with open(self.facts_meta_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.facts.append(json.loads(line))
                
        self._load_model()
        print(f"Loaded {len(self.facts)} facts from index.")
        
    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves top k facts relevant to the query."""
        if self.index is None:
            self.load_index()
            
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            fact = self.facts[idx]
            results.append({
                "fact_id": fact["fact_id"],
                "text": fact["text"],
                "source": fact.get("source", "Unknown"),
                "distance": float(distances[0][i])
            })
            
        return results

if __name__ == "__main__":
    # Test building the index
    retriever = SciQRetriever()
    retriever.build_index()
    
    # Test retrieval
    results = retriever.retrieve("What is the boiling point of water?")
    print("Test retrieval results:")
    for r in results:
        print(f" - {r['text']} (Dist: {r['distance']:.2f})")
