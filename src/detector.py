import os
import json
import torch
import numpy as np
import concurrent.futures

# Suppress HuggingFace tokenizer fork warnings when using threading
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from typing import Dict, Any, List, Tuple
from transformers import pipeline
from google import genai
from dotenv import load_dotenv
from src.retriever import SciQRetriever
from src.web_retriever import WebRetriever
from src.cache import LLMCache
from concurrent.futures import ThreadPoolExecutor

load_dotenv() # Load environment variables from .env if present

class GeminiJudge:
    def __init__(self, api_key: str = None):
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or passed as argument.")
        
        # Use the new google-genai Client
        self.client = genai.Client(api_key=api_key)
        # Using the model verified as working in user's environment
        self.model_name = "gemini-2.5-flash" 

    def verify(self, user_query: str, llm_output: str, evidence: List[Dict[str, Any]], num_samples: int = 2) -> Dict[str, Any]:
        """
        Uses Gemini to verify the llm_output against retrieved evidence.
        Implements Self-Consistency by taking a majority vote over multiple generations.
        """
        evidence_text = "\n".join([f"- {e['source']}: {e['text']}" for e in evidence if e['status'] != "Irrelevant"])
        
        if not evidence_text:
            prompt = f"""
            You are a Fact-Checking Assistant. Your task is to verify if a "Student Claim" is factually correct or a hallucination based on YOUR INTERNAL KNOWLEDGE.

            USER QUERY: {user_query}
            STUDENT CLAIM TO VERIFY: {llm_output}

            INSTRUCTIONS:
            1. Evaluate the Claim using your internal knowledge.
            2. Determine if the Claim is factually correct or hallucinated.
            3. Provide a final verdict: [Factually Correct, Hallucinated, or Unverified].
            4. Provide a brief, concise reason for your verdict.
            5. Provide a confidence score between 0.0 and 1.0.

            FORMAT YOUR RESPONSE AS JSON:
            {{
                "verdict": "Factually Correct" | "Hallucinated" | "Unverified",
                "reason": "Clear explanation based on your internal knowledge...",
                "confidence": 0.95
            }}
            """
        else:
            prompt = f"""
            You are a Fact-Checking Assistant. Your task is to verify if a "Student Claim" is factually correct or a hallucination based ONLY on the provided "Ground Truth Evidence".

            USER QUERY: {user_query}
            STUDENT CLAIM TO VERIFY: {llm_output}

            GROUND TRUTH EVIDENCE:
            {evidence_text}

            INSTRUCTIONS:
            1. Compare the Claim against the Evidence.
            2. Determine if the Evidence supports, contradicts, or is neutral towards the Claim.
            3. Provide a final verdict: [Factually Correct, Hallucinated, or Unverified].
            4. Provide a brief, concise reason for your verdict.
            5. Provide a confidence score between 0.0 and 1.0.

            FORMAT YOUR RESPONSE AS JSON:
            {{
                "verdict": "Factually Correct" | "Hallucinated" | "Unverified",
                "reason": "Clear explanation...",
                "confidence": 0.95
            }}
            """
        
        def _generate_single_sample(i):
            try:
                # We use a higher temperature to encourage different reasoning paths
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        temperature=0.7,
                        response_mime_type="application/json",
                    )
                )
                
                text = response.text
                if text:
                    # Fallback parsing in case the model wraps response in markdown code blocks
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif "{" in text:
                        text = text[text.find("{"):text.rfind("}")+1]
                        
                    result = json.loads(text)
                    
                    # Validate structure
                    if "verdict" in result and result["verdict"] in ["Factually Correct", "Hallucinated", "Unverified"]:
                        return result
                    else:
                        print(f"Skipping malformed Gemini response: {text}")
            except Exception as e:
                print(f"Gemini internal generation error: {e}")
            return None

        results = []
        try:
            with ThreadPoolExecutor(max_workers=num_samples) as executor:
                futures = [executor.submit(_generate_single_sample, i) for i in range(num_samples)]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res:
                        results.append(res)
        except Exception as e:
            print(f"Gemini Parallel Verification Error: {e}")
            if not results:
                return {
                    "verdict": "Unverified",
                    "reason": f"Gemini API error: {str(e)}",
                    "confidence": 0.0
                }
        
        if not results:
            return {
                "verdict": "Unverified",
                "reason": "Failed to generate valid JSON responses from Gemini.",
                "confidence": 0.0
            }

        # --- Self-Consistency Aggregation ---
        verdicts = {}
        for r in results:
            v_key = r["verdict"]
            if v_key not in verdicts:
                verdicts[v_key] = {"count": 0, "confidences": [], "reasons": []}
            
            verdicts[v_key]["count"] += 1
            # Add valid confidence values safely
            conf = float(r.get("confidence", 0.0))
            if 0.0 <= conf <= 1.0:
                verdicts[v_key]["confidences"].append(conf)
            verdicts[v_key]["reasons"].append(r.get("reason", ""))
        
        # Step 1: Find the majority vote
        majority_verdict = max(verdicts.keys(), key=lambda v_k: verdicts[v_k]["count"])
        
        # Step 2: Average confidence for the majority vote
        maj_confidences = verdicts[majority_verdict]["confidences"]
        avg_confidence = sum(maj_confidences) / len(maj_confidences) if maj_confidences else 0.0
        
        # Step 3: Use the reason from the first response that matches the majority verdict
        majority_reason = verdicts[majority_verdict]["reasons"][0] if verdicts[majority_verdict]["reasons"] else "Majority consensus."
        
        # If there's disagreement among samples, mention it explicitly
        total_valid = len(results)
        majority_count = verdicts[majority_verdict]["count"]
        
        if majority_count < total_valid:
            majority_reason += f" (Gemini Self-Consistency note: Majority {majority_count}/{total_valid} vote. Some internal disagreement detected.)"

        return {
            "verdict": majority_verdict,
            "reason": majority_reason,
            "confidence": avg_confidence,
            "samples": results # Include individual generations
        }

    def correct(self, user_query: str, llm_output: str, evidence: List[Dict[str, Any]]) -> str:
        """
        Generates a factually correct version of the llm_output based on evidence.
        """
        evidence_text = "\n".join([f"- {e['source']}: {e['text']}" for e in evidence if e['status'] != "Irrelevant"])
        
        prompt = f"""
        You are an Expert Fact-Correction Assistant. 
        The following "Student Claim" has been identified as a hallucination or factually incorrect.
        Your task is to provide a FACTUALLY CORRECT version of the claim based ONLY on the provided "Ground Truth Evidence".
        If no evidence is provided, use your internal knowledge, but maintain a helpful and educational tone.

        USER QUERY: {user_query}
        HALLUCINATED CLAIM: {llm_output}

        GROUND TRUTH EVIDENCE:
        {evidence_text if evidence_text else "No specific external evidence provided. Use internal knowledge."}

        INSTRUCTIONS:
        1. Rewrite the claim so it is factually accurate.
        2. Keep the correction concise and direct.
        3. Do not add conversational filler.
        4. Focus on correcting the specific errors identified.

        CORRECTED STATEMENT:
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(temperature=0.3)
            )
            return response.text.strip()
        except Exception as e:
            return f"Error generating correction: {str(e)}"

    def generate_xai(self, user_query: str, claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates structured XAI explanation including span highlighting, error type, and counterfactuals.
        """
        evidence_text = "\n".join([f"- {e['source']}: {e['text']}" for e in evidence if e['status'] != "Irrelevant"])
        
        prompt = f"""
        You are an Explainable AI (XAI) Assistant. Analyze the following "Hallucinated Claim" against the "Evidence".
        
        USER QUERY: {user_query}
        HALLUCINATED CLAIM: {claim}
        EVIDENCE:
        {evidence_text if evidence_text else "Use internal knowledge."}
        
        TASK:
        Identify the specific part of the claim that is wrong and categorize the error.
        
        FORMAT YOUR RESPONSE AS JSON:
        {{
            "wrong_phrase": "The exact substring of the claim that is incorrect",
            "error_type": "Factual Substitution | Fabrication | Overclaiming | Underclaiming | Other",
            "corrected_version": "A rewrite concentrating ONLY on fixing the hallucinated part",
            "counterfactual": "Explanation of what change (e.g., 'changing X to Y') would make the claim correct"
        }}
        """
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"Gemini XAI Error: {e}")
            return {"error": str(e)}

class OpenAIJudge:
    def __init__(self, api_key: str = None):
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or passed as argument.")
        
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
        self.model_name = "gpt-4o-mini" 

    def verify(self, user_query: str, llm_output: str, evidence: List[Dict[str, Any]], num_samples: int = 2) -> Dict[str, Any]:
        evidence_text = "\n".join([f"- {e['source']}: {e['text']}" for e in evidence if e.get('status') != "Irrelevant"])
        
        if not evidence_text:
            prompt = f"""
            You are a Fact-Checking Assistant. Your task is to verify if a "Student Claim" is factually correct or a hallucination based on YOUR INTERNAL KNOWLEDGE.

            USER QUERY: {user_query}
            STUDENT CLAIM TO VERIFY: {llm_output}

            INSTRUCTIONS:
            1. Evaluate the Claim using your internal knowledge.
            2. Determine if the Claim is factually correct or hallucinated.
            3. Provide a final verdict: [Factually Correct, Hallucinated, or Unverified].
            4. Provide a brief, concise reason for your verdict.
            5. Provide a confidence score between 0.0 and 1.0.

            FORMAT YOUR RESPONSE AS JSON:
            {{
                "verdict": "Factually Correct" | "Hallucinated" | "Unverified",
                "reason": "Clear explanation based on your internal knowledge...",
                "confidence": 0.95
            }}
            """
        else:
            prompt = f"""
            You are a Fact-Checking Assistant. Your task is to verify if a "Student Claim" is factually correct or a hallucination based ONLY on the provided "Ground Truth Evidence".

            USER QUERY: {user_query}
            STUDENT CLAIM TO VERIFY: {llm_output}

            GROUND TRUTH EVIDENCE:
            {evidence_text}

            INSTRUCTIONS:
            1. Compare the Claim against the Evidence.
            2. Determine if the Evidence supports, contradicts, or is neutral towards the Claim.
            3. Provide a final verdict: [Factually Correct, Hallucinated, or Unverified].
            4. Provide a brief, concise reason for your verdict.
            5. Provide a confidence score between 0.0 and 1.0.

            FORMAT YOUR RESPONSE AS JSON:
            {{
                "verdict": "Factually Correct" | "Hallucinated" | "Unverified",
                "reason": "Clear explanation...",
                "confidence": 0.95
            }}
            """
        
        def _generate_single_sample(i):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    response_format={ "type": "json_object" },
                    messages=[
                        {"role": "system", "content": "You output JSON matching the requested schema exactly."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                
                text = response.choices[0].message.content
                if text:
                    result = json.loads(text)
                    if "verdict" in result and result["verdict"] in ["Factually Correct", "Hallucinated", "Unverified"]:
                        return result
                    else:
                        print(f"Skipping malformed OpenAI response: {text}")
            except Exception as e:
                print(f"OpenAI internal generation error: {e}")
            return None

        results = []
        try:
            with ThreadPoolExecutor(max_workers=num_samples) as executor:
                futures = [executor.submit(_generate_single_sample, i) for i in range(num_samples)]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res:
                        results.append(res)
        except Exception as e:
            print(f"OpenAI Parallel Verification Error: {e}")
            if not results:
                return {
                    "verdict": "Unverified",
                    "reason": f"OpenAI API error: {str(e)}",
                    "confidence": 0.0
                }
        
        if not results:
            return {
                "verdict": "Unverified",
                "reason": "Failed to generate valid JSON responses from OpenAI.",
                "confidence": 0.0
            }

        verdicts = {}
        for r in results:
            v_key = r["verdict"]
            if v_key not in verdicts:
                verdicts[v_key] = {"count": 0, "confidences": [], "reasons": []}
            
            verdicts[v_key]["count"] += 1
            conf = float(r.get("confidence", 0.0))
            if 0.0 <= conf <= 1.0:
                verdicts[v_key]["confidences"].append(conf)
            verdicts[v_key]["reasons"].append(r.get("reason", ""))
        
        majority_verdict = max(verdicts.keys(), key=lambda v_k: verdicts[v_k]["count"])
        maj_confidences = verdicts[majority_verdict]["confidences"]
        avg_confidence = sum(maj_confidences) / len(maj_confidences) if maj_confidences else 0.0
        majority_reason = verdicts[majority_verdict]["reasons"][0] if verdicts[majority_verdict]["reasons"] else "Majority consensus."
        
        total_valid = len(results)
        majority_count = verdicts[majority_verdict]["count"]
        if majority_count < total_valid:
            majority_reason += f" (OpenAI Self-Consistency note: Majority {majority_count}/{total_valid} vote. Some internal disagreement detected.)"

        return {
            "verdict": majority_verdict,
            "reason": majority_reason,
            "confidence": avg_confidence,
            "samples": results # Include individual generations
        }

    def correct(self, user_query: str, llm_output: str, evidence: List[Dict[str, Any]]) -> str:
        evidence_text = "\n".join([f"- {e['source']}: {e['text']}" for e in evidence if e.get('status') != "Irrelevant"])
        
        prompt = f"""
        You are an Expert Fact-Correction Assistant. 
        The following "Student Claim" has been identified as a hallucination.
        Provide a FACTUALLY CORRECT version of the claim based on the provided "Evidence".

        USER QUERY: {user_query}
        HALLUCINATED CLAIM: {llm_output}

        EVIDENCE:
        {evidence_text if evidence_text else "No external evidence. Use internal knowledge."}

        INSTRUCTIONS:
        - Rewrite the claim to be factually accurate.
        - Be concise and objective.
        
        CORRECTED STATEMENT:
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Error generating OpenAI correction: {str(e)}"

    def generate_xai(self, user_query: str, claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        evidence_text = "\n".join([f"- {e['source']}: {e['text']}" for e in evidence if e.get('status') != "Irrelevant"])
        
        prompt = f"""
        Analyze the "Hallucinated Claim" based on "Evidence". Identify wrong span, error type, and counterfactual.
        
        USER QUERY: {user_query}
        HALLUCINATED CLAIM: {claim}
        EVIDENCE:
        {evidence_text}
        
        OUTPUT JSON:
        {{
            "wrong_phrase": "string",
            "error_type": "string",
            "corrected_version": "string",
            "counterfactual": "string"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                response_format={ "type": "json_object" },
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": str(e)}

class GroqJudge:
    def __init__(self, api_key: str = None):
        if not api_key:
            api_key = os.getenv("GROK_API_KEY") # User named it GROK_API_KEY in .env
        if not api_key:
            print("⚠️ Warning: GROK_API_KEY not found in environment.")
            self.api_key = None
        else:
            self.api_key = api_key
        
        self.url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"

    def verify(self, user_query: str, llm_output: str, evidence: List[Dict[str, Any]], num_samples: int = 2) -> Dict[str, Any]:
        """
        Uses Groq (Llama-3) to verify a claim against evidence.
        Uses self-consistency (majority vote) for robustness.
        """
        if not self.api_key:
            return {"verdict": "Unverified", "reason": "Groq API key missing.", "confidence": 0.0}

        evidence_text = "\n".join([f"- {e['source']}: {e['text']}" for e in evidence if e.get('status') != "Irrelevant"])
        
        if not evidence_text:
            prompt = f"""
            You are a Fact-Checking Assistant. Your task is to verify if a "Student Claim" is factually correct or a hallucination based on YOUR INTERNAL KNOWLEDGE.

            USER QUERY: {user_query}
            STUDENT CLAIM TO VERIFY: {llm_output}

            INSTRUCTIONS:
            1. Evaluate the Claim using your internal knowledge.
            2. Determine if the Claim is factually correct or hallucinated.
            3. Provide a final verdict: [Factually Correct, Hallucinated, or Unverified].
            4. Provide a brief, concise reason for your verdict.
            5. Provide a confidence score between 0.0 and 1.0.

            FORMAT YOUR RESPONSE AS JSON:
            {{
                "verdict": "Factually Correct" | "Hallucinated" | "Unverified",
                "reason": "Clear explanation based on your internal knowledge...",
                "confidence": 0.95
            }}
            """
        else:
            prompt = f"""
            You are a Fact-Checking Assistant. Your task is to verify if a "Student Claim" is factually correct or a hallucination based ONLY on the provided "Ground Truth Evidence".

            USER QUERY: {user_query}
            STUDENT CLAIM TO VERIFY: {llm_output}

            GROUND TRUTH EVIDENCE:
            {evidence_text}

            INSTRUCTIONS:
            1. Compare the Claim against the Evidence.
            2. Determine if the Evidence supports, contradicts, or is neutral towards the Claim.
            3. Provide a final verdict: [Factually Correct, Hallucinated, or Unverified].
            4. Provide a brief, concise reason for your verdict.
            5. Provide a confidence score between 0.0 and 1.0.

            FORMAT YOUR RESPONSE AS JSON:
            {{
                "verdict": "Factually Correct" | "Hallucinated" | "Unverified",
                "reason": "Clear explanation...",
                "confidence": 0.95
            }}
            """

        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        def _generate_single_sample(i):
            try:
                data = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.7, # Higher temp for diverse internal reasoning
                    "response_format": {"type": "json_object"}
                }
                response = requests.post(self.url, json=data, headers=headers, timeout=15)
                if response.status_code == 200:
                    raw_content = response.json()['choices'][0]['message']['content']
                    import json
                    return json.loads(raw_content)
                else:
                    print(f"Groq API Error: {response.status_code}. {response.text}")
            except Exception as e:
                print(f"Groq internal generation failure: {e}")
            return None

        results = []
        # Run multiple samples in parallel for self-consistency
        try:
            with ThreadPoolExecutor(max_workers=num_samples) as executor:
                futures = [executor.submit(_generate_single_sample, i) for i in range(num_samples)]
                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    if res:
                        results.append(res)
        except Exception as e:
            print(f"Groq Parallel Verification Exception: {e}")

        if not results:
            return {"verdict": "Unverified", "reason": "Groq API failed or timed out.", "confidence": 0.0}

        # Majority Vote Consensus Logic
        verdicts = {}
        for r in results:
            v_key = r.get("verdict", "Unverified")
            if v_key not in verdicts:
                verdicts[v_key] = {"count": 0, "confidences": [], "reasons": []}
            
            verdicts[v_key]["count"] += 1
            conf = float(r.get("confidence", 0.0))
            if 0.0 <= conf <= 1.0:
                verdicts[v_key]["confidences"].append(conf)
            verdicts[v_key]["reasons"].append(r.get("reason", ""))
        
        majority_verdict = max(verdicts.keys(), key=lambda v_k: verdicts[v_k]["count"])
        maj_confidences = verdicts[majority_verdict]["confidences"]
        avg_confidence = sum(maj_confidences) / len(maj_confidences) if maj_confidences else 0.0
        majority_reason = verdicts[majority_verdict]["reasons"][0] if verdicts[majority_verdict]["reasons"] else "Majority consensus."
        
        total_valid = len(results)
        majority_count = verdicts[majority_verdict]["count"]
        if majority_count < total_valid:
            majority_reason += f" (Groq Self-Consistency note: Majority {majority_count}/{total_valid} vote.)"

        return {
            "verdict": majority_verdict,
            "reason": majority_reason,
            "confidence": avg_confidence,
            "samples": results
        }

    def correct(self, user_query: str, llm_output: str, evidence: List[Dict[str, Any]]) -> str:
        evidence_text = "\n".join([f"- {e['source']}: {e['text']}" for e in evidence if e.get('status') != "Irrelevant"])
        
        prompt = f"""
        You are an Expert Fact-Correction Assistant.
        Rewrite the "Hallucinated Claim" to be factually correct based on the "Evidence".
        
        USER QUERY: {user_query}
        HALLUCINATED CLAIM: {llm_output}
        
        EVIDENCE:
        {evidence_text if evidence_text else "Use internal knowledge."}
        
        Corrected Statement:
        """

        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
            response = requests.post(self.url, json=data, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
            return f"Groq Correction Error: {response.status_code}"
        except Exception as e:
            return f"Groq Correction Exception: {e}"

    def generate_xai(self, user_query: str, claim: str, evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
        evidence_text = "\n".join([f"- {e['source']}: {e['text']}" for e in evidence if e.get('status') != "Irrelevant"])
        
        prompt = f"""
        Identify the error in the following claim using the evidence provided.
        Respond ONLY with a JSON object.
        
        USER QUERY: {user_query}
        HALLUCINATED CLAIM: {claim}
        EVIDENCE: {evidence_text}
        
        SCHEMA:
        {{
            "wrong_phrase": "exact span that is wrong",
            "error_type": "Factual Substitution | Fabrication | Overclaiming | Underclaiming",
            "corrected_version": "corrected part",
            "counterfactual": "how to make it correct"
        }}
        """
        
        import requests
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.2,
                "response_format": {"type": "json_object"}
            }
            response = requests.post(self.url, json=data, headers=headers, timeout=30)
            if response.status_code == 200:
                return json.loads(response.json()['choices'][0]['message']['content'])
            return {"error": f"Groq XAI Error: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}


class HallucinationDetector:
    def __init__(self, 
                 retriever: SciQRetriever = None, 
                 nli_model_name: str = "cross-encoder/nli-deberta-base"):
        """
        Initializes the Hallucination Detector.
        """
        self.retriever = retriever
        if self.retriever is None:
            self.retriever = SciQRetriever()
            self.retriever.load_index()
        
        self.web_retriever = WebRetriever()
        self.nli_model_name = nli_model_name
        self.nli_pipeline = None
        self.cache = LLMCache()

    def _load_nli_model(self):
        if self.nli_pipeline is None:
            print(f"Loading NLI model: {self.nli_model_name}")
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            try:
                self.nli_pipeline = pipeline("text-classification", model=self.nli_model_name, device=device)
            except Exception as e:
                print(f"Failed to load on {device}, falling back to CPU. Error: {e}")
                self.nli_pipeline = pipeline("text-classification", model=self.nli_model_name, device="cpu")

    def check_hallucination(self, user_query: str, llm_output: str, k: int = 3, use_gemini: bool = False, use_openai: bool = False, use_groq: bool = False, allow_web_fallback: bool = True) -> Dict[str, Any]:
        """
        Checks if the `llm_output` is hallucinated based on retrieved facts for `user_query`.
        Includes a 3-way LLM consensus (Gemini, OpenAI, Groq) with self-consistency.
        """
        # 0. Check Cache First
        cached_result = self.cache.get(user_query, llm_output)
        if cached_result:
            return cached_result

        # 1. Retrieve local evidence
        search_query = f"{user_query} {llm_output}"
        print(f"Retrieving top {k} local facts for query...")
        retrieved_facts = self.retriever.retrieve(search_query, k=k)
        
        RELEVANCE_THRESHOLD = 1.1 
        relevant_local_count = sum(1 for f in retrieved_facts if f["distance"] <= RELEVANCE_THRESHOLD)
        
        # 2. Web Fallback if local search fails
        is_fallback_active = False
        if relevant_local_count == 0 and allow_web_fallback:
            print("⚠️ Local Knowledge Base returned no relevant results. Triggering Web Fallback...")
            is_fallback_active = True
            retrieved_facts = self.web_retriever.search(user_query)
            # Re-evaluate relevance (all web search results are considered 'relevant' for cross-check)
            relevant_facts_count = len(retrieved_facts)
        else:
            relevant_facts_count = relevant_local_count


            
        # 3. Verify all facts in a single batch using NLI
        self._load_nli_model()
        
        individual_results = []
        best_entailment_score = 0.0
        best_contradiction_score = 0.0
        
        # Filter relevant facts for batching
        facts_to_process = []
        for fact in retrieved_facts:
            if not fact.get("is_web") and fact["distance"] > RELEVANCE_THRESHOLD:
                individual_results.append({**fact, "status": "Irrelevant", "nli_score": 0.0})
            else:
                facts_to_process.append(fact)

        if facts_to_process:
            print(f"Running batch NLI verification on {len(facts_to_process)} facts...")
            batch_inputs = [{"text": f["text"], "text_pair": llm_output} for f in facts_to_process]
            # Use pipeline batching
            batch_results = self.nli_pipeline(batch_inputs, truncation=True, batch_size=8)
            
            for fact, res in zip(facts_to_process, batch_results):
                label = res['label'].lower()
                score = res['score']
                
                fact_status = "Neutral"
                if "entailment" in label:
                    fact_status = "Supports"
                    best_entailment_score = max(best_entailment_score, score)
                elif "contradiction" in label:
                    fact_status = "Contradicts"
                    best_contradiction_score = max(best_contradiction_score, score)
                
                individual_results.append({
                    **fact,
                    "nli_label": label,
                    "nli_score": score,
                    "status": fact_status
                })
            
        # 4. Determine NLI verdict
        if relevant_facts_count == 0:
            nli_verdict = "Evidence Not Found"
            nli_reason = "No relevant facts found to verify the claim."
            nli_confidence = 0.0
        elif best_entailment_score > 0.5:
            nli_verdict = "Factually Correct"
            nli_reason = f"Supporting evidence found (Confidence: {best_entailment_score:.1%})."
            nli_confidence = best_entailment_score
        elif best_contradiction_score > 0.5:
            nli_verdict = "Hallucinated"
            nli_reason = f"Contradictory evidence found (Confidence: {best_contradiction_score:.1%})."
            nli_confidence = best_contradiction_score
        else:
            nli_verdict = "Hallucinated"
            nli_reason = "No strong evidence found to support the claim."
            nli_confidence = max(best_entailment_score, best_contradiction_score)

        # 5. Concurrent Cross-verification with LLMs
        llm_consensus = None
        openai_consensus = None
        groq_consensus = None
        final_verdict = nli_verdict
        final_confidence = nli_confidence
        final_reason = nli_reason
        used_internal_knowledge = False
        
        def run_gemini():
            try:
                print("Verifying with Gemini...")
                return GeminiJudge().verify(user_query, llm_output, individual_results)
            except Exception as e:
                return {"verdict": "Unverified", "reason": str(e), "confidence": 0.0}

        def run_openai():
            try:
                print("Verifying with OpenAI...")
                return OpenAIJudge().verify(user_query, llm_output, individual_results)
            except Exception as e:
                return {"verdict": "Unverified", "reason": str(e), "confidence": 0.0}

        def run_groq():
            try:
                print("Verifying with Groq...")
                return GroqJudge().verify(user_query, llm_output, individual_results)
            except Exception as e:
                return {"verdict": "Unverified", "reason": str(e), "confidence": 0.0}

        if relevant_facts_count > 0 and (use_gemini or use_openai or use_groq):
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                if use_gemini: futures[executor.submit(run_gemini)] = "gemini"
                if use_openai: futures[executor.submit(run_openai)] = "openai"
                if use_groq:   futures[executor.submit(run_groq)] = "groq"
                
                for future in concurrent.futures.as_completed(futures):
                    target = futures[future]
                    if target == "gemini": llm_consensus = future.result()
                    elif target == "openai": openai_consensus = future.result()
                    elif target == "groq": groq_consensus = future.result()

        # Multi-Model Consensus Logic for external evidence
        if relevant_facts_count > 0:
            all_verdicts = [{"source": "NLI", "verdict": nli_verdict, "confidence": nli_confidence, "reason": nli_reason}]
            if llm_consensus: all_verdicts.append({"source": "Gemini", **llm_consensus})
            if openai_consensus: all_verdicts.append({"source": "OpenAI", **openai_consensus})
            if groq_consensus: all_verdicts.append({"source": "Groq", **groq_consensus})

            # Filter for meaningful "successful" votes
            # We exclude "Unverified" (API errors) and NLI neutrality
            active_verdicts = []
            for v in all_verdicts:
                if v["verdict"] == "Unverified":
                    continue
                if v["verdict"] == "Evidence Not Found":
                    continue
                # If NLI returned a default hallucination because of "no strong evidence", 
                # we don't count it as a "strong" vote if we have any other definitive LLM vote.
                if v["source"] == "NLI" and "No strong evidence found" in v["reason"]:
                    # Check if we have at least one successful LLM vote
                    has_llm_vote = any(l for l in all_verdicts if l["source"] in ["Gemini", "OpenAI", "Groq"] and l["verdict"] != "Unverified")
                    if has_llm_vote:
                        continue # Skip this weak NLI vote
                
                active_verdicts.append(v)
            
            # If no conclusive LLM/NLI votes but we have "Evidence Not Found" from NLI, keep it as base
            if not active_verdicts:
                active_verdicts = [v for v in all_verdicts if v["verdict"] != "Unverified"]

            if len(active_verdicts) > 1:
                vote_tally = {}
                for v in active_verdicts:
                    v_key = v["verdict"]
                    if v_key not in vote_tally:
                        vote_tally[v_key] = {"count": 0, "conf_sum": 0.0, "sources": []}
                    vote_tally[v_key]["count"] += 1
                    vote_tally[v_key]["conf_sum"] += v["confidence"]
                    vote_tally[v_key]["sources"].append(v["source"])
                
                # Winner by majority count, then by avg confidence in case of ties
                majority_v_key = max(vote_tally.keys(), key=lambda k: (vote_tally[k]["count"], vote_tally[k]["conf_sum"]/vote_tally[k]["count"]))
                majority_info = vote_tally[majority_v_key]
                
                # Check for absolute disagreement (e.g. 1 Support vs 1 Contradict)
                is_tie = (len(active_verdicts) == 2 and majority_info["count"] == 1 and active_verdicts[0]["verdict"] != active_verdicts[1]["verdict"])
                    
                if is_tie:
                    final_verdict = "Mismatched"
                    votes_str = ", ".join([v["source"] + "=" + v["verdict"] for v in active_verdicts])
                    final_reason = f"Models disagreed. Votes: {votes_str}."
                    final_confidence = 0.0
                else:
                    final_verdict = majority_v_key
                    final_confidence = majority_info["conf_sum"] / majority_info["count"]
                    if majority_info["count"] == len(active_verdicts):
                        final_reason = f"Unanimous Agreement ({', '.join(majority_info['sources'])}). Base Logic: {nli_reason}"
                    else:
                        dissenters = [v for v in active_verdicts if v['verdict'] != majority_v_key]
                        dissent_text = ", ".join([f"{d['source']} ({d['verdict']})" for d in dissenters])
                        final_reason = f"Majority Vote ({', '.join(majority_info['sources'])}). Dissenting: {dissent_text}. Base: {nli_reason}"
            elif len(active_verdicts) == 1:
                # Only one model actually gave a definitive answer
                v = active_verdicts[0]
                final_verdict = v["verdict"]
                final_confidence = v["confidence"]
                # Mention the failover if necessary
                if v["source"] != "NLI" and nli_verdict == "Evidence Not Found":
                    final_reason = f"Verified via {v['source']} after evidence was missing for NLI. {v['reason']}"
                else:
                    final_reason = f"Verified via {v['source']}. {v['reason']}"
        
        # Internal Knowledge Fallback (Parallelized)
        if not retrieved_facts and (use_gemini or use_openai or use_groq):
            print("No evidence found. Running Multi-Model Internal Knowledge fallback...")
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {}
                if use_gemini: futures[executor.submit(run_gemini)] = "gemini"
                if use_openai: futures[executor.submit(run_openai)] = "openai"
                if use_groq:   futures[executor.submit(run_groq)] = "groq"
                
                for future in concurrent.futures.as_completed(futures):
                    target = futures[future]
                    if target == "gemini": llm_consensus = future.result()
                    elif target == "openai": openai_consensus = future.result()
                    elif target == "groq": groq_consensus = future.result()

            fallback_verdicts = []
            if llm_consensus and llm_consensus["verdict"] != "Unverified":
                fallback_verdicts.append({"source": "Gemini", **llm_consensus})
            if openai_consensus and openai_consensus["verdict"] != "Unverified":
                fallback_verdicts.append({"source": "OpenAI", **openai_consensus})
            if groq_consensus and groq_consensus["verdict"] != "Unverified":
                fallback_verdicts.append({"source": "Groq", **groq_consensus})

            if fallback_verdicts:
                used_internal_knowledge = True
                if len(fallback_verdicts) > 1:
                    vote_tally = {}
                    for v in fallback_verdicts:
                        v_key = v["verdict"]
                        if v_key not in vote_tally:
                            vote_tally[v_key] = {"count": 0, "conf_sum": 0.0, "sources": []}
                        vote_tally[v_key]["count"] += 1
                        vote_tally[v_key]["conf_sum"] += v["confidence"]
                        vote_tally[v_key]["sources"].append(v["source"])
                    
                    majority_v_key = max(vote_tally.keys(), key=lambda k: (vote_tally[k]["count"], vote_tally[k]["conf_sum"]/vote_tally[k]["count"]))
                    majority_info = vote_tally[majority_v_key]
                    
                    is_tie = (len(fallback_verdicts) == 2 and majority_info["count"] == 1 and fallback_verdicts[0]["verdict"] != fallback_verdicts[1]["verdict"])
                        
                    if is_tie:
                        final_verdict = "Mismatched"
                        votes_str = ", ".join([v["source"] + "=" + v["verdict"] for v in fallback_verdicts])
                        final_reason = f"Internal Knowledge Mismatch: {votes_str}"
                        final_confidence = 0.0
                    else:
                        final_verdict = majority_v_key
                        final_confidence = majority_info["conf_sum"] / majority_info["count"]
                        final_reason = f"Fallback to Internal Knowledge ({', '.join(majority_info['sources'])}): {fallback_verdicts[0]['reason']}"
                else:
                    # Only one
                    v = fallback_verdicts[0]
                    final_verdict = v["verdict"]
                    final_confidence = v["confidence"]
                    final_reason = f"Fallback to Internal Knowledge ({v['source']}): {v['reason']}"
            else:
                final_verdict = "Evidence Not Found"
                final_confidence = 0.0
                final_reason = "External evidence failed and LLM Internal Knowledge was unavailable."
        elif not retrieved_facts:
            final_verdict = "Evidence Not Found"
            final_confidence = 0.0
            final_reason = "The query appears to be outside both local knowledge and web reach."

        result = {
            "verdict": final_verdict,
            "confidence": final_confidence,
            "reason": final_reason,
            "evidence": individual_results,
            "nli_result": {
                "verdict": nli_verdict,
                "confidence": nli_confidence,
                "reason": nli_reason
            },
            "llm_result": llm_consensus,
            "openai_result": openai_consensus,
            "groq_result": groq_consensus,
            "is_web_fallback": is_fallback_active,
            "used_internal_knowledge": used_internal_knowledge
        }
        
        # 5. Save to Cache
        self.cache.set(user_query, llm_output, result)
        
        return result

    def generate_correction(self, user_query: str, llm_output: str, evidence: List[Dict[str, Any]], 
                           use_gemini: bool = False, use_openai: bool = False, use_groq: bool = False) -> str:
        """
        Generates a correction using the prioritized enabled judge.
        Priority: Groq -> OpenAI -> Gemini
        """
        # 0. Check Cache
        cache_key = f"corr|{user_query}|{llm_output}"
        cached = self.cache.get("correction", cache_key)
        if cached:
            return cached.get("correction_text", "Error retrieving cached correction")

        try:
            correction_text = ""
            if use_groq:
                print("Generating correction with Groq...")
                correction_text = GroqJudge().correct(user_query, llm_output, evidence)
            elif use_openai:
                print("Generating correction with OpenAI...")
                correction_text = OpenAIJudge().correct(user_query, llm_output, evidence)
            elif use_gemini:
                print("Generating correction with Gemini...")
                correction_text = GeminiJudge().correct(user_query, llm_output, evidence)
            else:
                return "No LLM judge enabled for correction."
            
            # 5. Save to Cache
            if correction_text:
                self.cache.set("correction", cache_key, {"correction_text": correction_text})
            
            return correction_text
        except Exception as e:
            return f"Failed to generate correction: {str(e)}"

    def get_detailed_explanation(self, user_query: str, llm_output: str, evidence: List[Dict[str, Any]], 
                                use_gemini: bool = False, use_openai: bool = False, use_groq: bool = False) -> Dict[str, Any]:
        """
        Generates a granular XAI explanation using the prioritized enabled judge.
        """
        # 0. Check Cache
        cache_key = f"xai|{user_query}|{llm_output}"
        cached = self.cache.get("xai", cache_key)
        if cached:
            return cached

        try:
            xai_data = {}
            if use_groq:
                print("Generating Granular XAI with Groq...")
                xai_data = GroqJudge().generate_xai(user_query, llm_output, evidence)
            elif use_openai:
                print("Generating Granular XAI with OpenAI...")
                xai_data = OpenAIJudge().generate_xai(user_query, llm_output, evidence)
            elif use_gemini:
                print("Generating Granular XAI with Gemini...")
                xai_data = GeminiJudge().generate_xai(user_query, llm_output, evidence)
            else:
                return {"error": "No LLM judge enabled for XAI."}
            
            # 5. Save to Cache
            if xai_data and "error" not in xai_data:
                self.cache.set("xai", cache_key, xai_data)
                
            return xai_data
        except Exception as e:
            return {"error": f"Failed to generate XAI: {str(e)}"}

if __name__ == "__main__":
    detector = HallucinationDetector()
    res = detector.check_hallucination("Who is the current CEO of Tesla?", "Elon Musk is the CEO of Tesla.")
    print(json.dumps(res, indent=2))
