import os
from typing import List, Dict, Any
from src.live_retrieval import FactVerifier

class WebRetriever:
    def __init__(self, max_results: int = 3):
        self.max_results = max_results
        # FactVerifier automatically handles .env loading
        self.verifier = FactVerifier()

    def search(self, query: str) -> List[Dict[str, Any]]:
        """
        Performs a multi-source web search (Tavily, Wikipedia, Wikidata) 
        and returns consolidated evidence snippets.
        """
        print(f"🌎 Performing multi-source web search fallback for: {query}")
        results = []
        try:
            # 1. Use the new FactVerifier to pull evidence
            # We include Wikipedia as primary source, Wikidata is used for structured facts
            verification = self.verifier.verify(query)
            
            # 2. Map Evidence objects to the detector's expected format
            # Truncate all snippets to 300 chars for conciseness
            for i, e in enumerate(verification.evidence):
                results.append({
                    "fact_id": f"web_{i}",
                    "text": e.snippet[:300] + ("..." if len(e.snippet) > 300 else ""),
                    "source": f"{e.source.capitalize()}: {e.title}",
                    "url": e.url,
                    "distance": 0.0,
                    "is_web": True
                })
                
            # 3. Add Wikipedia summary as a high-priority fact (truncated)
            if verification.wikipedia_summary:
                results.insert(0, {
                    "fact_id": "web_wiki_summary",
                    "text": verification.wikipedia_summary[:300] + ("..." if len(verification.wikipedia_summary) > 300 else ""),
                    "source": "Wikipedia Summary",
                    "url": f"https://en.wikipedia.org/wiki/{query.replace(' ', '_')}",
                    "distance": 0.0,
                    "is_web": True
                })

            # 4. Add key Wikidata facts as evidence string
            if verification.wikidata_facts:
                fact_str = " | ".join([f"{k}: {v}" for k, v in list(verification.wikidata_facts.items()) if not k.startswith("_")])
                results.append({
                    "fact_id": "web_wikidata",
                    "text": f"Wikidata Structured Facts: {fact_str}",
                    "source": "Wikidata Knowledge Graph",
                    "url": f"https://www.wikidata.org/wiki/{verification.wikidata_facts.get('_qid', '')}",
                    "distance": 0.0,
                    "is_web": True
                })

        except Exception as e:
            print(f"❌ Multi-source web search failed: {e}")
            
        return results

if __name__ == "__main__":
    retriever = WebRetriever()
    res = retriever.search("Albert Einstein Nobel Prize")
    for r in res:
        print(f"\nSource: {r['source']}\nSnippet: {r['text']}")
