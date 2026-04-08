import streamlit as st
import os
import json

# Force tokenizer parallelism to false to avoid deadlock warnings with ThreadPoolExecutor
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
from typing import Dict, Any, List
from src.retriever import SciQRetriever
from src.detector import HallucinationDetector
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px

load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Educational Hallucination Detector",
    page_icon="🕵️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #238636;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        border: none;
    }
    .fact-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-bottom: 10px;
    }
    .verdict-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 25px;
        font-size: 24px;
        font-weight: bold;
    }
    .verdict-correct {
        background: linear-gradient(135deg, #1b4d3e 0%, #0d2a22 100%);
        border-left: 5px solid #3fb950;
        color: #3fb950;
    }
    .verdict-hallucinated {
        background: linear-gradient(135deg, #4d1b1b 0%, #2a0d0d 100%);
        border-left: 5px solid #f85149;
        color: #f85149;
    }
    .distance-badge {
        font-size: 12px;
        padding: 2px 8px;
        border-radius: 10px;
        background-color: #30363d;
        color: #8b949e;
    }
    /* Premium Shadows and Borders */
    .premium-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: bold;
        color: #58a6ff;
    }
    .pipeline-step {
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        background: #21262d;
        border: 1px solid #30363d;
        margin-right: 10px;
        font-size: 12px;
        color: #8b949e;
    }
    .pipeline-active {
        background: #1f6feb;
        color: white;
        border-color: #58a6ff;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if 'detector' not in st.session_state:
    with st.spinner("Initializing AI Models and Knowledge Base..."):
        st.session_state.detector = HallucinationDetector()
        # Pre-load NLI to avoid delay during first check
        st.session_state.detector._load_nli_model()

# --- Sidebar ---
with st.sidebar:
    st.title("🕵️Detector Settings")
    st.markdown("---")
    
    st.header("Settings")
    if st.button("🔄 Clear Cache & Reload Index"):
        st.session_state.detector.retriever.load_index()
        st.success("Index Reloaded!")

    st.markdown("---")
    st.header("Consensus Models")
    use_gemini = st.toggle("Enable Gemini Judge", value=False, help="Uses Gemini 2.5 to cross-verify the output (Requires GEMINI_API_KEY in .env)")
    use_openai = st.toggle("Enable OpenAI Judge", value=False, help="Uses OpenAI GPT to cross-verify the output (Requires OPENAI_API_KEY in .env)")
    use_groq = st.toggle("Enable Groq Judge", value=True, help="Uses Groq (Llama-3) for ultra-fast verification (Requires GROK_API_KEY in .env)")
    
    st.markdown("---")
    st.header("KB Stats:")
    st.markdown("- SciQ Facts: ~12,200")
    st.markdown("- Wikipedia Educational: ~150")
    st.markdown("- Total facts: ~12,400")

# --- Main UI ---
st.title("VerifED: Educational Hallucination Detector")
st.markdown("Verify LLM statements against **12.4k+ facts** from SciQ and Wikipedia educational sources.")

col1, col2 = st.columns(2)

with col1:
    st.header("1. The Context")
    user_query = st.text_area("User Query or Question:", 
                               placeholder="e.g., What is the boiling point of water?",
                               height=150)

with col2:
    st.header("2. The Claim")
    llm_output = st.text_area("LLM Generated Output to Verify:", 
                               placeholder="e.g., Water boils at 100 degrees Celsius.",
                               height=150)

# --- Visual Pipeline Tracker ---
if st.button("🔍 Analyze for Hallucinations"):
    if not user_query or not llm_output:
        st.warning("Please provide both a query and the output to verify.")
    else:
        with st.spinner("Searching Knowledge Base and Running Consensus Verification..."):
            result = st.session_state.detector.check_hallucination(
                user_query, 
                llm_output, 
                use_gemini=use_gemini, 
                use_openai=use_openai,
                use_groq=use_groq
            )
            st.session_state.check_done = True
            
            verdict = result["verdict"]
            confidence = result["confidence"]
            reason = result["reason"]
            evidence = result["evidence"]
            
            # --- NEW: Define Tabs ---
            tab_detect, tab_correct, tab_explain = st.tabs([
                "📊 Detection & Evidence", 
                "✨ Fact-Checked Correction", 
                "🔬 Granular XAI Analysis"
            ])

            with tab_detect:
                # Display Final Verdict Box
                if verdict == "Factually Correct":
                    box_class = "verdict-correct"
                    icon = "✅"
                elif verdict == "Hallucinated":
                    box_class = "verdict-hallucinated"
                    icon = "🚨"
                elif verdict == "Mismatched":
                    box_class = "verdict-unverified"
                    icon = "🔀"
                    st.markdown("""
                    <style>
                        .verdict-unverified {
                            background: linear-gradient(135deg, #442200 0%, #221100 100%);
                            border-left: 5px solid #ffaa00;
                            color: #ffcc00;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                elif "Not Found" in verdict:
                    box_class = "verdict-unverified"
                    icon = "🔍"
                    st.markdown("""
                    <style>
                        .verdict-unverified {
                            background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
                            border-left: 5px solid #58a6ff;
                            color: #58a6ff;
                        }
                    </style>
                    """, unsafe_allow_html=True)
                else:
                    box_class = "verdict-unverified"
                    icon = "⚠️"
                    
                st.markdown(f"""
                <div class='verdict-box {box_class}'>
                    {icon} Final Verdict: {verdict.upper()}
                </div>
                """, unsafe_allow_html=True)

                # Populate models_to_show before the metric dashboard
                models_to_show = []
                if result.get("nli_result"): models_to_show.append(("Local NLI", result["nli_result"]))
                if result.get("llm_result"): models_to_show.append(("Gemini Judge", result["llm_result"]))
                if result.get("openai_result"): models_to_show.append(("OpenAI Judge", result["openai_result"]))
                if result.get("groq_result"): models_to_show.append(("Groq Judge", result["groq_result"]))

                # --- Metric Dashboard ---
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.metric("Final Confidence", f"{confidence:.1%}", delta=f"{confidence*10:.2f} pts")
                with m_col2:
                    st.metric("Evidence Clips", len(evidence), delta=f"{len(evidence)} facts", delta_color="normal")
                with m_col3:
                    consensus_count = len(models_to_show)
                    st.metric("Judge Consensus", f"{consensus_count}/4", help="NLI + LLMs participating in final vote")

                if result.get("is_web_fallback"):
                    st.warning("🌐 **Web Fallback Active:** No relevant local facts were found. Results are grounded in real-time Web Search snippets.")
                
                if len(models_to_show) > 0:
                    vi_col1, vi_col2 = st.columns([1, 1.5])
                    with vi_col1:
                        st.markdown("#### 📊 Consensus Map")
                        categories = [m[0] for m in models_to_show]
                        values = [m[1].get('confidence', 0.0) for m in models_to_show]
                        fig = go.Figure(data=go.Scatterpolar(r=values, theta=categories, fill='toself', line_color='#58a6ff', fillcolor='rgba(88, 166, 255, 0.3)'))
                        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1]), bgcolor="#0d1117"), showlegend=False, paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=40, r=40, t=10, b=10), height=250)
                        st.plotly_chart(fig, use_container_width=True)
                    with vi_col2:
                        st.markdown("#### ⚖️ Voting Distribution")
                        for name, res_dict in models_to_show:
                            st.markdown(f"**{name}:** {res_dict.get('verdict', 'N/A')}")
                            st.progress(res_dict.get('confidence', 0.0))
                
                st.info(reason)

                if result.get("used_internal_knowledge"):
                    st.markdown("### 🧠 Internal Knowledge Fallback")
                    st.info("No external evidence found. Verification performed using LLM internal knowledge.")
                else:
                    st.markdown("### 📚 Retrieved Evidence")
                    if evidence:
                        for fact in evidence:
                            text = fact.get('text', 'No text found')
                            dist = fact.get('distance', 0.0)
                            status = fact.get('status', 'Neutral')
                            nli_score = fact.get('nli_score', 0.0)
                            source = fact.get('source', 'Unknown Source')
                            is_web = fact.get('is_web', False)
                            status_color = "#3fb950" if status == "Supports" else "#f85149" if status == "Contradicts" else "#8b949e"
                            status_icon = "✅" if status == "Supports" else "🚨" if status == "Contradicts" else "⚪"
                            dist_badge = f"<span class='distance-badge'>L2 Dist: {dist:.2f}</span>" if not is_web else "<span class='distance-badge' style='background: #238636; color: white;'>Web Result</span>"
                            st.markdown(f"""<div class='fact-card'><div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;'>{dist_badge}<span style='background: #1f6feb; color: white; padding: 2px 8px; border-radius: 12px; font-size: 11px; font-weight: bold;'>{source}</span></div><p style='margin-bottom: 5px;'>{text}</p><div style='font-size: 14px; border-top: 1px solid #30363d; padding-top: 5px; color: {status_color};'>{status_icon} <strong>{status}</strong> (Confidence: {nli_score:.1%})</div></div>""", unsafe_allow_html=True)

            with tab_correct:
                if verdict in ["Hallucinated", "Mismatched"]:
                    with st.spinner("Generating Fact-Checked Correction..."):
                        correction = st.session_state.detector.generate_correction(user_query, llm_output, evidence, use_gemini=use_gemini, use_openai=use_openai, use_groq=use_groq)
                        st.session_state.correction = correction
                else:
                    st.session_state.correction = None

                if st.session_state.get("correction"):
                    st.markdown("### ✨ Recommended Fact-Checked Version")
                    st.markdown(f"""<div style='padding: 20px; border-radius: 10px; border: 1px solid #2ea043; border-left: 5px solid #2ea043;'><p style='color: #8b949e; font-size: 12px; margin-bottom: 5px; text-transform: uppercase; font-weight: bold;'>Proposed Correction:</p><p style='font-size: 18px; color: #c9d1d9; font-style: italic;'>"{st.session_state.correction}"</p></div>""", unsafe_allow_html=True)
                else:
                    st.success("No correction needed for this claim.")

            with tab_explain:
                if verdict in ["Hallucinated", "Mismatched"]:
                    with st.spinner("🔬 Generating Granular XAI Analysis..."):
                        xai_data = st.session_state.detector.get_detailed_explanation(user_query, llm_output, evidence, use_gemini=use_gemini, use_openai=use_openai, use_groq=use_groq)
                    
                    if xai_data and "error" not in xai_data:
                        st.markdown("### 🔬 Precision Error Breakdown")
                        wrong_phrase = xai_data.get("wrong_phrase", "")
                        highlighted_claim = llm_output.replace(wrong_phrase, f"<span style='background-color: #f85149; color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold;'>{wrong_phrase}</span>") if wrong_phrase and wrong_phrase in llm_output else llm_output
                        st.markdown(f"**Identified Error Span:**")
                        st.markdown(f"<div style='padding: 15px; border-radius: 10px; border: 1px solid #30363d;'>{highlighted_claim}</div>", unsafe_allow_html=True)
                        col_x1, col_x2 = st.columns(2)
                        with col_x1:
                            st.markdown(f"**Error Category:**")
                            st.info(f"🏷️ {xai_data.get('error_type', 'Unknown')}")
                        with col_x2:
                            st.markdown(f"**Counterfactual Explanation:**")
                            st.success(f"🔄 {xai_data.get('counterfactual', 'N/A')}")
                
                with st.expander("🔬 Detailed Model Audit (Self-Consistency)"):
                    for name, res_dict in models_to_show:
                        if "samples" in res_dict and res_dict["samples"]:
                            st.markdown(f"#### {name} Reasoning Log")
                            for idx, sample in enumerate(res_dict["samples"]):
                                status_emoji = "✅" if sample["verdict"] == "Factually Correct" else "❌" if sample["verdict"] == "Hallucinated" else "⚠️"
                                st.write(f"**Sample {idx+1}:** {status_emoji} {sample['verdict']} | Reasoning: {sample.get('reason', 'N/A')}")
                            st.divider()

# Footer
st.markdown("---")
st.caption("Educational Hallucination Detector v1.4 | Evidence grounded in SciQ & Wikipedia | Multimodal Consensus Engine")
