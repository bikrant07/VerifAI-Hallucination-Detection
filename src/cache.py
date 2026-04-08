import sqlite3
import hashlib
import json
import os
import time
import re
from typing import Dict, Any, Optional

class LLMCache:
    def __init__(self, db_path: str = "verifai_cache.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the cache table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS llm_results (
                        id TEXT PRIMARY KEY,
                        query TEXT,
                        claim TEXT,
                        result_json TEXT,
                        timestamp REAL
                    )
                """)
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_query_claim ON llm_results (query, claim)")
                conn.commit()
        except Exception as e:
            print(f"❌ Cache initialization error: {e}")

    def _normalize_text(self, text: str) -> str:
        if not text:
            return ""
        # Lowercase and strip
        text = text.lower().strip()
        
        # Standardize common units and symbols
        replacements = {
            r'\bdegree celsius\b': 'c',
            r'\bdegrees celsius\b': 'c',
            r'\bcelsius\b': 'c',
            r'°c': 'c',
            r'\bpercent\b': '%',
        }
        for pattern, repl in replacements.items():
            text = re.sub(pattern, repl, text)
            
        # Remove punctuation, keeping only alphanumeric and spaces
        text = re.sub(r'[^\w\s]', '', text)
        
        # Collapse multiple spaces into one
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _generate_id(self, query: str, claim: str) -> str:
        """Generate a unique ID for a query-claim pair."""
        norm_query = self._normalize_text(query)
        norm_claim = self._normalize_text(claim)
        combined = f"{norm_query}|{norm_claim}"
        return hashlib.sha256(combined.encode()).hexdigest()

    def get(self, query: str, claim: str) -> Optional[Dict[str, Any]]:
        """Retrieve a cached result if it exists."""
        cache_id = self._generate_id(query, claim)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT result_json FROM llm_results WHERE id = ?", (cache_id,))
                row = cursor.fetchone()
                if row:
                    print(f"✅ Cache Hit for: {query[:30]}...")
                    return json.loads(row[0])
        except Exception as e:
            print(f"⚠️ Cache read error: {e}")
        return None

    def set(self, query: str, claim: str, result: Dict[str, Any]):
        """Store a result in the cache."""
        cache_id = self._generate_id(query, claim)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO llm_results (id, query, claim, result_json, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (cache_id, query, claim, json.dumps(result), time.time()))
                conn.commit()
            print(f"💾 Results cached for: {query[:30]}...")
        except Exception as e:
            print(f"❌ Cache write error: {e}")
