"""
LLM Fact Verifier — Tavily + Wikipedia + Wikidata
--------------------------------------------------
Retrieves evidence from multiple sources to validate LLM outputs.

Install:
    pip install tavily-python wikipedia-api requests

Usage:
    verifier = FactVerifier(tavily_api_key="tvly-...")
    result = verifier.verify("The Eiffel Tower was built in 1889")
    print(result.summary())
"""

import os
import re
import requests
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()




# ─────────────────────────────────────────
# Data models
# ─────────────────────────────────────────

@dataclass
class Evidence:
    source: str          # "tavily" | "wikipedia" | "wikidata"
    title: str
    url: str
    snippet: str
    score: float = 0.0   # relevance score where available


@dataclass
class VerificationResult:
    claim: str
    evidence: list[Evidence] = field(default_factory=list)
    wikipedia_summary: Optional[str] = None
    wikidata_facts: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Claim : {self.claim}",
            f"Sources found : {len(self.evidence)}",
            "",
        ]
        if self.wikipedia_summary:
            lines += ["── Wikipedia ──", self.wikipedia_summary[:400] + "...", ""]

        if self.wikidata_facts:
            lines.append("── Wikidata facts ──")
            for k, v in list(self.wikidata_facts.items())[:8]:
                lines.append(f"  {k}: {v}")
            lines.append("")

        if self.evidence:
            lines.append("── Tavily sources ──")
            for e in self.evidence[:5]:
                lines.append(f"  [{e.score:.2f}] {e.title}")
                lines.append(f"       {e.url}")
                lines.append(f"       {e.snippet[:140]}...")
                lines.append("")

        return "\n".join(lines)


# ─────────────────────────────────────────
# Wikipedia client
# ─────────────────────────────────────────

class WikipediaClient:
    BASE = "https://en.wikipedia.org/api/rest_v1"
    SEARCH = "https://en.wikipedia.org/w/api.php"

    def search(self, query: str, limit: int = 3) -> list[dict]:
        """Return top page titles matching a query."""
        headers = {"User-Agent": "HallucinationDetector (Educational Research)"}
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }
        r = requests.get(self.SEARCH, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json().get("query", {}).get("search", [])

    def get_summary(self, title: str) -> Optional[str]:
        """Return the plain-text intro summary for a page."""
        headers = {"User-Agent": "HallucinationDetector (Educational Research)"}
        url = f"{self.BASE}/page/summary/{quote(title)}"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        return data.get("extract")

    def get_sections(self, title: str) -> list[dict]:
        """Return page sections (title + text) for deeper fact-checking."""
        headers = {"User-Agent": "HallucinationDetector (Educational Research)"}
        params = {
            "action": "parse",
            "page": title,
            "prop": "sections",
            "format": "json",
        }
        r = requests.get(self.SEARCH, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json().get("parse", {}).get("sections", [])


# ─────────────────────────────────────────
# Wikidata client
# ─────────────────────────────────────────

class WikidataClient:
    SPARQL = "https://query.wikidata.org/sparql"
    SEARCH = "https://www.wikidata.org/w/api.php"

    def search_entity(self, query: str) -> Optional[str]:
        """Return the QID (e.g. Q243) for the best-matching entity."""
        headers = {"User-Agent": "HallucinationDetector (Educational Research)"}
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "format": "json",
            "limit": 1,
        }
        r = requests.get(self.SEARCH, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        results = r.json().get("search", [])
        return results[0]["id"] if results else None

    def get_facts(self, qid: str) -> dict:
        """
        Return a dict of human-readable label → value for key properties.
        Covers: description, instance of, country, inception date,
                occupation, notable work, coordinates, and more.
        """
        headers = {
            "User-Agent": "HallucinationDetector (Educational Research)",
            "Accept": "application/sparql-results+json"
        }
        query = f"""
        SELECT DISTINCT ?propLabel ?valueLabel WHERE {{
          wd:{qid} ?p ?value .
          ?propEntity wikibase:directClaim ?p ;
                      rdfs:label ?propLabel .
          FILTER(LANG(?propLabel) = "en")
          
          SERVICE wikibase:label {{ 
            bd:serviceParam wikibase:language "en". 
            ?value rdfs:label ?valueLabel .
          }}
          FILTER(!isBlank(?value))
        }}
        LIMIT 50
        """
        r = requests.get(
            self.SPARQL,
            params={"query": query},
            headers=headers,
            timeout=30,
        )
        if r.status_code != 200:
            print(f"❌ Wikidata SPARQL Error {r.status_code}: {r.text[:200]}")
            return {}

        facts = {}
        bindings = r.json().get("results", {}).get("bindings", [])
        if not bindings:
            print(f"⚠️ No Wikidata facts found for {qid} with current query.")
            
        for row in bindings:
            prop = row["propLabel"]["value"]
            val = row.get("valueLabel", {}).get("value", "")
            if val and not val.startswith("http"):  # skip raw URIs
                facts[prop] = val
        return facts

    def entity_description(self, qid: str) -> Optional[str]:
        """Return the short English description for a QID."""
        headers = {"User-Agent": "HallucinationDetector (Educational Research)"}
        url = f"https://www.wikidata.org/wiki/Special:EntityData/{qid}.json"
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            return None
        entities = r.json().get("entities", {})
        entity = entities.get(qid, {})
        descs = entity.get("descriptions", {})
        return descs.get("en", {}).get("value")


# ─────────────────────────────────────────
# Main verifier
# ─────────────────────────────────────────

class FactVerifier:
    """
    Verify a claim by pulling evidence from:
      - Tavily  (live web search, scored snippets)
      - Wikipedia  (human-readable summary)
      - Wikidata  (structured entity facts)
    """

    def __init__(self):
        self.wiki = WikipediaClient()
        self.wikidata = WikidataClient()

    # ── helpers ──────────────────────────

    def _extract_main_entity(self, claim: str) -> str:
        """
        Passes the entire query/claim as requested by user.
        """
        return claim[:100] # Limit length slightly for API safety

    # ── individual retrievers ─────────────



    def fetch_wikipedia(self, entity: str) -> tuple[Optional[str], list[Evidence]]:
        """Return (summary_text, list_of_Evidence) for the best Wikipedia match."""
        results = self.wiki.search(entity, limit=3)
        if not results:
            return None, []

        top_title = results[0]["title"]
        summary = self.wiki.get_summary(top_title)

        evidence = []
        for r in results:
            evidence.append(Evidence(
                source="wikipedia",
                title=r["title"],
                url=f"https://en.wikipedia.org/wiki/{quote(r['title'])}",
                snippet=r.get("snippet", "").replace('<span class="searchmatch">', "").replace("</span>", ""),
                score=1.0,
            ))
        return summary, evidence

    def fetch_wikidata(self, entity: str) -> dict:
        """Return structured facts dict for the entity from Wikidata."""
        qid = self.wikidata.search_entity(entity)
        if not qid:
            return {}
        facts = self.wikidata.get_facts(qid)
        desc = self.wikidata.entity_description(qid)
        if desc:
            facts["_description"] = desc
        facts["_qid"] = qid
        return facts

    # ── main entry point ──────────────────

    def verify(
        self,
        claim: str,
        include_wikidata: bool = True,
    ) -> VerificationResult:
        """
        Retrieve evidence for a claim from remaining sources:
          - Wikipedia  (human-readable summary)
          - Wikidata  (structured entity facts)
        """
        result = VerificationResult(claim=claim)
        entity = self._extract_main_entity(claim)

        # 1 — Wikipedia
        print(f"[wikipedia] entity: {entity!r}")
        wiki_summary, wiki_evidence = self.fetch_wikipedia(entity)
        result.wikipedia_summary = wiki_summary
        result.evidence.extend(wiki_evidence)
        print(f"            → {'found' if wiki_summary else 'not found'}")

        # 2 — Wikidata
        if include_wikidata:
            print(f"[wikidata]  entity: {entity!r}")
            facts = self.fetch_wikidata(entity)
            result.wikidata_facts = facts
            print(f"            → {len(facts)} properties")

        return result

    def verify_batch(self, claims: list[str], **kwargs) -> list[VerificationResult]:
        """Verify multiple claims sequentially."""
        return [self.verify(claim, **kwargs) for claim in claims]


# ─────────────────────────────────────────
# Quick demo
# ─────────────────────────────────────────

if __name__ == "__main__":
    # Set your key here or export TAVILY_API_KEY=tvly-...
    API_KEY = os.environ.get("TAVILY_API_KEY", "tvly-your-key-here")

    verifier = FactVerifier(tavily_api_key=API_KEY)

    test_claims = [
        "The Eiffel Tower was built in 1889",
        "Albert Einstein won the Nobel Prize in Physics in 1921",
    ]

    for claim in test_claims:
        print("\n" + "=" * 60)
        result = verifier.verify(claim)
        print()
        print(result.summary())