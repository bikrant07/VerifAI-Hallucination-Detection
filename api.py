# uvicorn api:app --reload --port 8000
import os
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.detector import HallucinationDetector

# Load environment variables
load_dotenv()

# --- Models ---
class VerifyRequest(BaseModel):
    query: str
    claim: str

class VerifyResponse(BaseModel):
    verdict: str
    confidence: float
    reason: str
    highlighted_span: str = ""
    error_type: str = ""
    correction: str = ""
    counterfactual: str = ""
    evidence: List[Dict[str, Any]] = []

# --- Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the HallucinationDetector once on startup."""
    print("Initialising Hallucination Detector...")
    detector = HallucinationDetector()
    detector._load_nli_model()
    app.state.detector = detector
    yield
    # Clean up if needed
    print("Shutting down...")

# --- App Instance ---
app = FastAPI(
    title="VerifED API",
    description="API for Educational Hallucination Detection",
    version="1.0.0",
    lifespan=lifespan
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for Chrome extension
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

@app.post("/verify", response_model=VerifyResponse)
async def verify_claim(request: VerifyRequest):
    """
    Verifies a claim against a query using the HallucinationDetector.
    Returns verdict, confidence, reason, XAI details, and evidence.
    """
    detector: HallucinationDetector = app.state.detector
    
    try:
        # 1. Perform detection
        # Enabling all judges for the API by default since keys are expected in .env
        result = detector.check_hallucination(
            user_query=request.query,
            llm_output=request.claim,
            use_gemini=False,
            use_openai=True,
            use_groq=True,
            allow_web_fallback=True
        )
        
        verdict = result.get("verdict", "Unverified")
        confidence = result.get("confidence", 0.0)
        reason = result.get("reason", "Verification failed.")
        evidence = result.get("evidence", [])
        
        # 2. Generate correction and XAI if hallucination detected
        highlighted_span = ""
        error_type = ""
        correction = ""
        counterfactual = ""
        
        if verdict in ["Hallucinated", "Mismatched"]:
            # Generate Correction
            correction = detector.generate_correction(
                user_query=request.query,
                llm_output=request.claim,
                evidence=evidence,
                use_gemini=False,
                use_openai=True,
                use_groq=True
            )
            
            # Generate XAI Details
            xai_data = detector.get_detailed_explanation(
                user_query=request.query,
                llm_output=request.claim,
                evidence=evidence,
                use_gemini=False,
                use_openai=True,
                use_groq=True
            )
            
            if xai_data and "error" not in xai_data:
                highlighted_span = xai_data.get("wrong_phrase", "")
                error_type = xai_data.get("error_type", "")
                counterfactual = xai_data.get("counterfactual", "")
                # Some models might return corrected_version in XAI; 
                # we prefer the one from generate_correction but fallback if needed
                if not correction:
                    correction = xai_data.get("corrected_version", "")
        
        return VerifyResponse(
            verdict=verdict,
            confidence=confidence,
            reason=reason,
            highlighted_span=highlighted_span,
            error_type=error_type,
            correction=correction,
            counterfactual=counterfactual,
            evidence=evidence
        )
        
    except Exception as e:
        print(f"Error during verification: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
