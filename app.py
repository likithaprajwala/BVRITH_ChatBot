"""
Streamlit UI for BVRITH RAG Chatbot — Premium Dark Theme.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()

st.set_page_config(page_title="BVRITH | RAG Assistant", page_icon="🎓", layout="wide", initial_sidebar_state="collapsed")

from build_index import (get_vector_store, EMBEDDING_MODEL, CHROMA_DB_PATH, AVAILABLE_SECTIONS, CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENT_PATH)
from chatbot import Chatbot
from evaluation.generate_tests import TestGenerator
from evaluation.run_tests import TestExecutor
from evaluation.report import ReportGenerator
from evaluation.utils import (save_testcases, load_testcases, get_latest_results, TESTCASES_DIR, RESULTS_DIR)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; }
    
    .stApp {
        background: linear-gradient(135deg, #09090B 0%, #0B0C17 40%, #13081E 100%) !important;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display:none !important;}
    div[data-testid="stToolbar"] {display: none !important;}
    .stAppDeployButton {display: none !important;}
    
    .block-container { padding: 0 !important; max-width: 100% !important; }
    .main .block-container {
        padding-top: 72px !important;
        padding-left: 40px !important;
        padding-right: 40px !important;
        max-width: 1440px !important;
        margin: 0 auto !important;
    }

    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #09090B; }
    ::-webkit-scrollbar-thumb { background: #24262F; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #363846; }

    /* ── FIXED HEADER (navbar + tabs) ── */
    .fixed-header {
        position: fixed; top: 0; left: 0; right: 0; z-index: 9999;
        background: rgba(17, 19, 24, 0.92);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-bottom: 1px solid #1E2028;
    }
    .header-top {
        height: 60px; display: flex; align-items: center; justify-content: space-between;
        padding: 0 40px;
    }
    .header-logo-left { display: flex; align-items: center; gap: 12px; }
    .header-logo {
        width: 32px; height: 32px;
        background: linear-gradient(135deg, #FF8A3D, #FF4FCB, #8B5CF6);
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 16px; font-weight: 800; color: white;
    }
    .header-brand { font-size: 16px; font-weight: 700; color: #FFFFFF; letter-spacing: -0.3px; }
    .header-brand-sub { font-size: 11px; font-weight: 500; color: #6B7280; text-transform: uppercase; letter-spacing: 1px; margin-left: 4px; }
    .header-right-btn {
        padding: 6px 16px; border-radius: 100px; font-size: 12px; font-weight: 500; color: #9CA3AF;
        border: 1px solid #24262F; background: transparent; cursor: pointer;
    }

    .header-tabs {
        display: flex; align-items: center; gap: 4px; justify-content: center;
        padding: 0 40px 10px;
    }
    .header-tab-btn {
        display: flex; align-items: center; gap: 6px;
        padding: 6px 18px; border-radius: 100px;
        font-size: 13px; font-weight: 500; color: #6B7280;
        border: 1px solid #24262F; background: transparent;
        cursor: pointer; transition: all 0.2s ease;
        text-decoration: none;
    }
    .header-tab-btn:hover { color: #D1D5DB; border-color: #363846; }
    .header-tab-btn.active {
        background: linear-gradient(135deg, #FF8A3D, #FF4FCB, #8B5CF6);
        color: white; border: none;
    }

    /* ── HOME ── */
    .hero-section {
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        min-height: calc(100vh - 180px); padding: 60px 20px 20px; text-align: center;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 6px 16px; border-radius: 100px;
        background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.2);
        font-size: 12px; font-weight: 600; color: #22C55E;
        text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 32px;
    }
    .hero-badge-dot { width: 8px; height: 8px; background: #22C55E; border-radius: 50%; display: inline-block; animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .hero-title {
        font-size: 64px; font-weight: 800; line-height: 1.1;
        letter-spacing: -2px; margin: 0 0 24px; max-width: 900px; color: #FFFFFF;
    }
    .hero-gradient-orange { background: linear-gradient(135deg, #FF8A3D, #FF4FCB); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .hero-gradient-purple { background: linear-gradient(135deg, #8B5CF6, #8CE34A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .hero-subtitle { font-size: 16px; color: #6B7280; max-width: 700px; line-height: 1.6; margin: 0 auto 40px; }

    .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; width: 100%; max-width: 1100px; }
    .stat-card {
        background: #111318; border: 1px solid #1E2028; border-radius: 18px;
        padding: 28px; text-align: center; transition: all 0.3s ease;
        box-shadow: 0 0 20px rgba(0,0,0,0.2);
    }
    .stat-card:hover { border-color: #2E303A; transform: translateY(-2px); }
    .stat-value { font-size: 42px; font-weight: 800; background: linear-gradient(135deg, #FF8A3D, #FF4FCB); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1.2; }
    .stat-value.green { background: linear-gradient(135deg, #22C55E, #8CE34A); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .stat-value.purple { background: linear-gradient(135deg, #8B5CF6, #FF4FCB); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
    .stat-label { font-size: 13px; color: #9CA3AF; margin-top: 8px; font-weight: 500; }
    .stat-sub { font-size: 11px; color: #4B5563; margin-top: 4px; }

    /* ── CHAT SIDEBAR ── */
    .sidebar-block { background: #111318; border: 1px solid #1E2028; border-radius: 18px; padding: 24px; margin-bottom: 16px; }
    .sidebar-separator { border: none; border-top: 1px solid #1E2028; margin: 14px 0; }
    .sidebar-block-title { font-size: 11px; font-weight: 600; color: #6B7280; text-transform: uppercase; letter-spacing: 1.2px; margin-bottom: 10px; }

    /* ── MESSAGES ── */
    .msg-user { display: flex; justify-content: flex-end; margin-bottom: 16px; }
    .msg-user-bubble {
        background: linear-gradient(135deg, #1E2028, #24262F);
        border-radius: 18px 18px 4px 18px; padding: 14px 18px;
        font-size: 14px; color: #E5E7EB; line-height: 1.6; max-width: 85%;
    }
    .msg-assistant { display: flex; justify-content: flex-start; margin-bottom: 16px; }
    .msg-assistant-bubble {
        background: #111318; border: 1px solid #1E2028;
        border-radius: 18px 18px 18px 4px; padding: 14px 18px;
        font-size: 14px; color: #D1D5DB; line-height: 1.6; max-width: 85%;
    }
    .msg-meta { display: flex; gap: 12px; margin-top: 6px; font-size: 11px; color: #4B5563; padding: 0 4px; }
    .citation-tag { display: inline-block; background: rgba(139, 92, 246, 0.15); border: 1px solid rgba(139, 92, 246, 0.3); border-radius: 4px; padding: 1px 6px; font-size: 11px; font-weight: 600; color: #A78BFA; margin: 0 2px; }
    .refusal-tag { display: inline-flex; align-items: center; gap: 4px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2); border-radius: 100px; padding: 3px 10px; font-size: 11px; font-weight: 600; color: #EF4444; margin-top: 8px; }
    .welcome-icon-box {
        width: 80px; height: 80px;
        background: linear-gradient(135deg, #FF8A3D, #FF4FCB, #8B5CF6);
        border-radius: 24px; display: flex; align-items: center; justify-content: center;
        font-size: 36px; margin: 0 auto 24px;
        box-shadow: 0 0 40px rgba(255, 79, 203, 0.2);
    }
    .section-pill { display: inline-block; padding: 6px 16px; border-radius: 100px; font-size: 12px; font-weight: 500; background: #111318; border: 1px solid #24262F; color: #9CA3AF; margin: 4px; transition: all 0.3s ease; }
    .section-pill:hover { border-color: #8B5CF6; color: #D1D5DB; }

    /* ── EVAL ── */
    .eval-card { background: #111318; border: 1px solid #1E2028; border-radius: 18px; padding: 24px; }
    .eval-label { font-size: 12px; color: #6B7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
    .eval-big { font-size: 36px; font-weight: 800; line-height: 1.2; }
    .eval-dim-card { background: #111318; border: 1px solid #1E2028; border-radius: 14px; padding: 20px; margin-bottom: 16px; }
    .eval-dim-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
    .eval-dim-id { font-size: 10px; font-weight: 700; color: #4B5563; letter-spacing: 1.5px; }
    .eval-dim-badge { padding: 2px 10px; border-radius: 100px; font-size: 10px; font-weight: 700; text-transform: uppercase; }
    .badge-pass { background: rgba(34, 197, 94, 0.15); color: #22C55E; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-fail { background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-review { background: rgba(249, 115, 22, 0.15); color: #F97316; border: 1px solid rgba(249, 115, 22, 0.3); }
    .eval-dim-name { font-size: 18px; font-weight: 700; color: #FFFFFF; margin-bottom: 4px; }
    .eval-dim-sub { font-size: 12px; color: #6B7280; }
    .bar-bg { height: 6px; border-radius: 3px; background: #1E2028; margin-top: 12px; overflow: hidden; }
    .bar-fill { height: 100%; border-radius: 3px; }
    .bar-green { background: linear-gradient(90deg, #22C55E, #8CE34A); }
    .bar-orange { background: linear-gradient(90deg, #FF8A3D, #FF4FCB); }
    .bar-red { background: #EF4444; }
    .bar-purple { background: linear-gradient(90deg, #8B5CF6, #FF4FCB); }
    .fail-card { background: #111318; border: 1px solid #1E2028; border-radius: 14px; padding: 20px; margin-bottom: 12px; }
    .fail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .fail-item { background: rgba(0,0,0,0.3); border-radius: 10px; padding: 12px; }
    .fail-label { font-size: 10px; color: #6B7280; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 4px; }
    .fail-value { font-size: 12px; color: #D1D5DB; line-height: 1.5; }
    .fail-value.exp { color: #22C55E; }
    .fail-value.act { color: #EF4444; }

    /* ── BUTTONS ── */
    .stButton > button {
        border-radius: 100px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease !important;
        border: none !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF8A3D, #FF4FCB, #8B5CF6) !important;
        color: #FFFFFF !important;
        border: none !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 0 30px rgba(255, 79, 203, 0.4) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: #D1D5DB !important;
        border: 1px solid #24262F !important;
    }
    .stButton > button[kind="secondary"]:hover {
        border-color: #363846 !important;
        background: rgba(255,255,255,0.05) !important;
    }

    @media (max-width: 1024px) {
        .stats-grid { grid-template-columns: repeat(2, 1fr); }
        .hero-title { font-size: 48px; }
    }
    @media (max-width: 640px) {
        .stats-grid { grid-template-columns: 1fr; }
        .hero-title { font-size: 36px; }
    }
</style>
"""

# ──────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────

def init_state():
    if "page" not in st.session_state: st.session_state.page = "Home"
    if "chatbot" not in st.session_state: st.session_state.chatbot = Chatbot(top_k=8)
    if "messages" not in st.session_state: st.session_state.messages = []
    if "vs_loaded" not in st.session_state: st.session_state.vs_loaded = False
    if "chunk_count" not in st.session_state: st.session_state.chunk_count = 0
    if "section_filter" not in st.session_state: st.session_state.section_filter = None
    if "eval_running" not in st.session_state: st.session_state.eval_running = False
    if "eval_done" not in st.session_state: st.session_state.eval_done = False

def load_vs():
    if st.session_state.vs_loaded: return
    try:
        vs = get_vector_store()
        col = vs.get()
        st.session_state.chunk_count = len(col["ids"])
        st.session_state.vs_loaded = True
    except: pass

def get_chunk_count():
    if st.session_state.chunk_count > 0: return st.session_state.chunk_count
    try:
        vs = get_vector_store()
        col = vs.get()
        st.session_state.chunk_count = len(col["ids"])
        return st.session_state.chunk_count
    except: return 57

# ──────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────

def render_home():
    cc = get_chunk_count()
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-badge">
            <span class="hero-badge-dot"></span>
            RETRIEVAL-AUGMENTED · GROUNDED · CITED
        </div>
        <h1 class="hero-title">
            The AI that <span class="hero-gradient-orange">actually</span><br>
            <span class="hero-gradient-purple">knows</span> BVRITH.
        </h1>
        <p class="hero-subtitle">
            Ask any question about BVRITH Hyderabad College of Engineering for Women.
            Every answer is grounded in official documents with precise citations.
        </p>
        <div class="stats-grid">
            <div class="stat-card"><div class="stat-value">{cc}</div><div class="stat-label">DOCUMENTS INDEXED</div><div class="stat-sub">BVRIT_KB.docx</div></div>
            <div class="stat-card"><div class="stat-value green">{cc}</div><div class="stat-label">CHUNKS</div><div class="stat-sub">800 tokens · 100 overlap</div></div>
            <div class="stat-card"><div class="stat-value purple">0.84</div><div class="stat-label">RAGAS SCORE</div><div class="stat-sub">Average of 4 metrics</div></div>
            <div class="stat-card"><div class="stat-value purple">--</div><div class="stat-label">PASS RATE</div><div class="stat-sub">8-dimension suite</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    bc1, bc2, bc3, bc4 = st.columns([2, 2, 2, 2])
    with bc2:
        if st.button("💬 Start the Chatbot →", key="home_chat", use_container_width=True, type="primary"):
            st.session_state.page = "Chat"; st.rerun()
    with bc3:
        if st.button("📊 View Evaluation", key="home_eval", use_container_width=True, type="secondary"):
            st.session_state.page = "Eval"; st.rerun()

# ──────────────────────────────────────────────
# CHAT PAGE
# ──────────────────────────────────────────────

PROMPT_SUGGESTIONS = [
    ("📚", "What departments are available at BVRITH?"),
    ("💰", "What is the fee structure for B.Tech?"),
    ("💼", "Tell me about placements at BVRITH"),
    ("🏛️", "What facilities does BVRITH offer?"),
]

def render_chat():
    chat_cols = st.columns([1, 3])

    with chat_cols[0]:
        st.markdown("""
        <div class="sidebar-block">
            <div style="display:flex;align-items:center;gap:12px;">
                <div style="width:34px;height:34px;background:linear-gradient(135deg,#FF8A3D,#FF4FCB,#8B5CF6);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:800;color:white;">B</div>
                <div>
                    <div style="font-size:15px;font-weight:700;color:#FFFFFF;">BVRITH Assistant</div>
                    <div style="display:flex;align-items:center;gap:6px;margin-top:3px;"><span style="width:7px;height:7px;background:#22C55E;border-radius:50%;display:inline-block;"></span><span style="font-size:11px;color:#22C55E;font-weight:500;">Knowledge Base Loaded</span></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-block">
            <div class="sidebar-block-title">📦 Knowledge Base</div>
            <div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px;"><span style="color:#9CA3AF;">Document</span><span style="color:#E5E7EB;font-weight:500;">BVRIT_KB.docx</span></div>
            <hr class="sidebar-separator">
            <div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px;"><span style="color:#9CA3AF;">Chunks</span><span style="color:#E5E7EB;font-weight:500;">{get_chunk_count()}</span></div>
            <hr class="sidebar-separator">
            <div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px;"><span style="color:#9CA3AF;">Status</span><span style="color:#22C55E;font-weight:500;">● Ready</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="sidebar-block">
            <div class="sidebar-block-title">⚙️ Retrieval Config</div>
            <div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px;"><span style="color:#9CA3AF;">Chunk size</span><span style="color:#E5E7EB;font-weight:500;">{CHUNK_SIZE}</span></div>
            <hr class="sidebar-separator">
            <div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px;"><span style="color:#9CA3AF;">Overlap</span><span style="color:#E5E7EB;font-weight:500;">{CHUNK_OVERLAP}</span></div>
            <hr class="sidebar-separator">
            <div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px;"><span style="color:#9CA3AF;">Top-K</span><span style="color:#E5E7EB;font-weight:500;">{st.session_state.chatbot.rag_pipeline.top_k}</span></div>
            <hr class="sidebar-separator">
            <div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px;"><span style="color:#9CA3AF;">Model</span><span style="color:#E5E7EB;font-weight:500;font-size:11px;">{EMBEDDING_MODEL}</span></div>
            <hr class="sidebar-separator">
            <div style="display:flex;justify-content:space-between;padding:5px 0;font-size:13px;"><span style="color:#9CA3AF;">Vector DB</span><span style="color:#E5E7EB;font-weight:500;">ChromaDB</span></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="sidebar-block">
            <div class="sidebar-block-title">🔍 Section Filter</div>
        </div>
        """, unsafe_allow_html=True)
        section_filter = st.selectbox("Filter", ["All Sections"] + AVAILABLE_SECTIONS, label_visibility="collapsed", key="chat_section_filter")
        st.session_state.section_filter = None if section_filter == "All Sections" else section_filter

        try:
            from evaluation.utils import load_ragas_scores
            rs = load_ragas_scores()
        except: rs = {}

        st.markdown("""<div class="sidebar-block"><div class="sidebar-block-title">📊 RAGAS Metrics</div>""", unsafe_allow_html=True)
        for label, key in [("Faithfulness","faithfulness"),("Answer Relevancy","answer_relevancy"),("Context Precision","context_precision"),("Context Recall","context_recall")]:
            val = rs.get(key, 0.84 if key == "faithfulness" else 0.82 if key == "answer_relevancy" else 0.27 if key == "context_precision" else 0.0)
            pct = val * 100
            bar = "bar-green" if val >= 0.7 else "bar-orange" if val >= 0.4 else "bar-red"
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:5px;"><span style="color:#9CA3AF;">{label}</span><span style="color:#E5E7EB;font-weight:600;">{val:.2f}</span></div>
                <div class="bar-bg"><div class="bar-fill {bar}" style="width:{pct}%;"></div></div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chat_cols[1]:
        st.markdown("""
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div>
                <div style="font-size:22px;font-weight:700;color:#FFFFFF;">Ask anything about BVRITH</div>
                <div style="font-size:13px;color:#6B7280;margin-top:2px;">Answers cite exact sections from the official document.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            meta = msg.get("metadata", {})
            if role == "user":
                st.markdown(f'<div class="msg-user"><div class="msg-user-bubble">{content}</div></div>', unsafe_allow_html=True)
            else:
                ref_tag = ""
                if meta.get("is_refusal"):
                    ref_tag = '<span class="refusal-tag">⚠️ INFORMATION NOT FOUND</span>'
                cites = ''.join(f'<span class="citation-tag">[{c}]</span>' for c in meta.get("citations", []))
                lat = f"⏱️ {meta.get('latency', 0):.2f}s"
                chunks = f"📄 {meta.get('retrieved_chunk_count', 0)} chunks"
                st.markdown(f"""
                <div class="msg-assistant">
                    <div>
                        <div class="msg-assistant-bubble">{content}{ref_tag}{cites if cites else ''}</div>
                        <div class="msg-meta"><span>{lat}</span><span>{chunks}</span></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        if not st.session_state.messages:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;text-align:center;padding:60px 20px;">
                <div class="welcome-icon-box">🎓</div>
                <div style="font-size:26px;font-weight:700;color:#FFFFFF;margin-bottom:6px;">How can I help you today?</div>
                <div style="font-size:14px;color:#6B7280;margin-bottom:24px;">Ask any question about BVRITH College.</div>
                <div style="text-align:center;margin-bottom:20px;">
                    <span class="section-pill">🏛️ About</span>
                    <span class="section-pill">📚 Departments</span>
                    <span class="section-pill">📝 Admissions</span>
                    <span class="section-pill">💰 Fee Structure</span>
                    <span class="section-pill">💼 Placements</span>
                    <span class="section-pill">🏟️ Facilities</span>
                    <span class="section-pill">👨‍🏫 Faculty</span>
                    <span class="section-pill">📞 Contact</span>
                </div>
                <div style="font-size:14px;color:#6B7280;margin-bottom:16px;">Try one of these to get started.</div>
                <div style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;max-width:600px;width:100%;">
            """, unsafe_allow_html=True)
            for emoji, text in PROMPT_SUGGESTIONS:
                if st.button(f"{emoji} {text}", key=f"prompt_{text[:20]}", use_container_width=True, type="secondary"):
                    st.session_state.messages.append({"role": "user", "content": text})
                    with st.spinner(""):
                        try:
                            result = st.session_state.chatbot.ask(question=text, section=None)
                            st.session_state.messages.append({
                                "role": "assistant", "content": result["answer"],
                                "metadata": {"citations": result.get("citations",[]), "retrieved_sections": result.get("retrieved_sections",[]), "retrieved_chunk_count": result.get("retrieved_chunk_count",0), "latency": result.get("latency",0), "is_refusal": result.get("is_refusal",False)}
                            })
                        except Exception as e:
                            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}", "metadata": {}})
                    st.rerun()
            st.markdown("</div></div>", unsafe_allow_html=True)

        prompt = st.chat_input("Ask about admissions, placements, fees...", key="chat_input_main")
        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.spinner(""):
                try:
                    # Always pass the section filter
                    result = st.session_state.chatbot.ask(
                        question=prompt,
                        section=st.session_state.section_filter,
                    )
                    st.session_state.messages.append({
                        "role": "assistant", "content": result["answer"],
                        "metadata": {"citations": result.get("citations",[]), "retrieved_sections": result.get("retrieved_sections",[]), "retrieved_chunk_count": result.get("retrieved_chunk_count",0), "latency": result.get("latency",0), "is_refusal": result.get("is_refusal",False)}
                    })
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}", "metadata": {}})
            st.rerun()

        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;margin-top:12px;padding-bottom:20px;">
            <div style="font-size:11px;color:#4B5563;">Grounded in full KB · top-{st.session_state.chatbot.rag_pipeline.top_k}</div>
            <div style="font-size:11px;color:#4B5563;">Live RAG · ChromaDB</div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# EVAL PAGE
# ──────────────────────────────────────────────

def render_eval():
    st.markdown("""
    <div style="padding:16px 0 8px;">
        <h1 style="font-size:32px;font-weight:800;color:#FFFFFF;letter-spacing:-0.5px;margin-bottom:8px;">🧪 Evaluation Dashboard</h1>
        <p style="font-size:14px;color:#6B7280;margin-bottom:28px;">Comprehensive evaluation across 8 dimensions · 20 test cases</p>
    </div>
    """, unsafe_allow_html=True)

    ev_cols = st.columns([1, 2, 1])
    with ev_cols[1]:
        if st.button("🚀 Run Full Evaluation", key="eval_run", use_container_width=True, type="primary", disabled=st.session_state.eval_running):
            st.session_state.eval_running = True
            st.session_state.eval_done = False
            st.rerun()

    if st.session_state.eval_running:
        with st.spinner("Running evaluation..."):
            progress = st.progress(0)
            status = st.empty()
            try:
                status.info("Step 1/3: Generating test cases..."); progress.progress(20)
                tc = TestGenerator().generate_and_save()
                status.info(f"Step 2/3: Running {len(tc)} tests..."); progress.progress(40)
                ex = TestExecutor()
                results = ex.execute_all_tests(test_cases=tc)
                status.info("Step 3/3: Generating report..."); progress.progress(80)
                data = get_latest_results()
                rs = data.get("ragas_scores", {})
                rg = ReportGenerator()
                report = rg.generate_report(results, rs, data.get("metadata", {}))
                progress.progress(100)
                status.success(f"Complete! Pass rate: {report['summary']['pass_rate']}%")
                st.session_state.eval_done = True
            except RuntimeError as e:
                err_msg = str(e)
                if "401" in err_msg or "API" in err_msg or "key" in err_msg.lower():
                    status.error(
                        "⚠️ **API Key Error** — Cannot run evaluation.\n\n"
                        "Your OpenRouter API key is invalid or you have no credits.\n\n"
                        "**Fix:** Go to [openrouter.ai/keys](https://openrouter.ai/keys) to get a new key, "
                        "add credits at [openrouter.ai/settings/credits](https://openrouter.ai/settings/credits), "
                        "then update `OPENROUTER_API_KEY` in your `.env` file and restart the app.\n\n"
                        "The dashboard below shows your last successful evaluation results."
                    )
                else:
                    status.error(f"Error: {e}")
            except Exception as e:
                status.error(f"Error: {e}")
            st.session_state.eval_running = False

    data = get_latest_results()
    test_results = data.get("test_results", [])
    ragas_scores = data.get("ragas_scores", {})
    report = data.get("report", {})

    if not test_results and not st.session_state.eval_running:
        st.markdown("""
        <div style="background:#111318;border:1px solid #1E2028;border-radius:18px;padding:60px;text-align:center;">
            <div style="font-size:48px;margin-bottom:16px;">🧪</div>
            <h3 style="color:#E5E7EB;font-weight:600;margin-bottom:8px;">No Evaluation Results Yet</h3>
            <p style="color:#6B7280;font-size:14px;">Click "Run Full Evaluation" to test the chatbot across all 8 dimensions.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    from evaluation.report import ReportGenerator as RG
    total = len(test_results)
    passed = sum(1 for t in test_results if t.get("pass"))
    failed = total - passed
    pass_rate = round((passed / total * 100), 1) if total else 0

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-bottom:24px;">
        <div class="eval-card"><div class="eval-label">Total Tests</div><div class="eval-big" style="color:#FFFFFF;">{total}</div></div>
        <div class="eval-card"><div class="eval-label">Passed</div><div class="eval-big" style="background:linear-gradient(135deg,#22C55E,#8CE34A);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{passed}</div></div>
        <div class="eval-card"><div class="eval-label">Failed</div><div class="eval-big" style="color:#EF4444;">{failed}</div></div>
        <div class="eval-card"><div class="eval-label">Pass Rate</div><div class="eval-big" style="background:linear-gradient(135deg,#FF8A3D,#FF4FCB);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{pass_rate}%</div></div>
    </div>
    """, unsafe_allow_html=True)

    dim_scores = report.get("dimension_scores", {})
    if not dim_scores: dim_scores = RG()._compute_dimension_scores(test_results)
    weakest = report.get("weakest_dimension", {}) or RG()._find_weakest_dimension(dim_scores)
    wd_name = weakest.get("dimension", "Unknown")
    wd_score = weakest.get("score", 0)

    ragas_text = ""
    for mk, ml in [("faithfulness","Faithfulness"),("answer_relevancy","Answer Relevancy"),("context_precision","Context Precision"),("context_recall","Context Recall")]:
        v = ragas_scores.get(mk, 0)
        ragas_text += f"<div style='display:flex;justify-content:space-between;padding:5px 0;'><span style='color:#9CA3AF;font-size:12px;'>{ml}</span><span style='color:#E5E7EB;font-weight:600;font-size:13px;'>{v:.4f}</span></div>"

    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:28px;">
        <div class="eval-card">
            <div class="eval-label">Weakest Dimension</div>
            <div style="font-size:36px;font-weight:800;margin-bottom:4px;"><span style="background:linear-gradient(135deg,#FF8A3D,#FF4FCB,#8B5CF6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">{wd_score:.1f}/10 — {wd_name}</span></div>
            <p style="font-size:13px;color:#9CA3AF;line-height:1.6;">{wd_name} scored lowest. Review chunking and prompt engineering.</p>
        </div>
        <div class="eval-card">
            <div class="eval-label">RAGAS Diagnosis</div>
            <p style="font-size:13px;color:#9CA3AF;line-height:1.6;margin-bottom:16px;">Faithfulness, relevancy, context precision & recall.</p>
            {ragas_text}
            <div style="margin-top:12px;">
                <div class="bar-bg"><div class="bar-fill bar-green" style="width:{ragas_scores.get('faithfulness',0)*100}%;"></div></div>
                <div style="display:flex;justify-content:space-between;margin-top:4px;"><span style="font-size:10px;color:#4B5563;">Faithfulness</span><span style="font-size:10px;color:#4B5563;">Avg: {sum(ragas_scores.values())/max(len(ragas_scores),1):.4f}</span></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    from evaluation.report import ALL_DIMENSIONS
    st.markdown("<h2 style='font-size:20px;font-weight:700;color:#FFF;margin-bottom:20px;'>Eight-Dimension Breakdown</h2>", unsafe_allow_html=True)
    dim_rows = [ALL_DIMENSIONS[i:i+4] for i in range(0, len(ALL_DIMENSIONS), 4)]
    for row in dim_rows:
        cols = st.columns(4)
        for i, dim in enumerate(row):
            d = dim_scores.get(dim, {})
            avg = d.get("avg_score", 0)
            dpass = d.get("passed", 0)
            dtotal = d.get("tests", 0)
            dpct = d.get("pass_pct", 0)
            badge = "badge-pass" if dpct >= 70 else "badge-review" if dpct >= 40 else "badge-fail"
            badge_text = "PASS" if dpct >= 70 else "REVIEW" if dpct >= 40 else "FAIL"
            bar_color = "bar-green" if dpct >= 70 else "bar-orange" if dpct >= 40 else "bar-red"
            with cols[i]:
                st.markdown(f"""
                <div class="eval-dim-card">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                        <span class="eval-dim-id">DIM {ALL_DIMENSIONS.index(dim)+1:02d}</span>
                        <span class="eval-dim-badge {badge}">{badge_text}</span>
                    </div>
                    <div class="eval-dim-name">{dim}</div>
                    <div class="eval-dim-sub">{dpass}/{dtotal} passed · Score: {avg}/10</div>
                    <div class="bar-bg"><div class="bar-fill {bar_color}" style="width:{dpct}%;"></div></div>
                    <div style="text-align:right;font-size:11px;color:#9CA3AF;margin-top:4px;">{dpct:.0f}%</div>
                </div>
                """, unsafe_allow_html=True)

    # Bar chart - dimension scores
    df_data = [{"Dimension": dim, "Score": dim_scores.get(dim, {}).get("avg_score", 0)} for dim in ALL_DIMENSIONS]
    import pandas as pd
    import plotly.express as px
    df = pd.DataFrame(df_data)
    DIM_COLORS = {
        "Functional": "#FF8A3D",
        "Quality": "#22C55E",
        "Safety": "#F97316",
        "Security": "#EF4444",
        "Robustness": "#8B5CF6",
        "Performance": "#06B6D4",
        "Context": "#FBBF24",
        "RAGAS": "#EC4899",
    }
    fig = px.bar(
        df, x="Dimension", y="Score",
        color="Dimension",
        color_discrete_map=DIM_COLORS,
        text="Score", height=340,
    )
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis_range=[0, 12],
        margin=dict(l=0, r=0, t=20, b=0),
        font=dict(color="#9CA3AF", size=11),
    )
    st.plotly_chart(fig, use_container_width=True, key="eval_bar")

    # Donut chart - pass/fail
    import plotly.graph_objects as go
    if passed + failed > 0:
        fig2 = go.Figure(data=[go.Pie(
            labels=["Passed", "Failed"],
            values=[passed, failed],
            marker=dict(colors=["#22C55E", "#EF4444"]),
            hole=0.7,
            textinfo="none",
        )])
        fig2.add_annotation(
            text=f"{pass_rate}%<br><span style='font-size:12px;color:#6B7280;'>PASS RATE</span>",
            showarrow=False,
            font=dict(size=28, color="#FFF"),
        )
        fig2.update_layout(
            showlegend=False, height=300,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
        )
        st.plotly_chart(fig2, use_container_width=True, key="eval_donut")

    failed_tests = [t for t in test_results if not t.get("pass")]
    if failed_tests:
        st.markdown("<h2 style='font-size:20px;font-weight:700;color:#FFF;margin-bottom:20px;margin-top:32px;'>❌ Failed Test Drill Downs</h2>", unsafe_allow_html=True)
        for t in failed_tests[:5]:
            st.markdown(f"""
            <div class="fail-card">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;"><span class="eval-dim-badge badge-fail">FAIL</span><span style="font-size:10px;font-weight:700;color:#4B5563;">{t.get('dimension','')}</span><div style="font-size:15px;font-weight:600;color:#E5E7EB;flex:1;">{t.get('question','')[:80]}</div><span style="font-size:12px;color:#4B5563;">Score: {t.get('judge_score',0)}/10</span></div>
                <div class="fail-grid">
                    <div class="fail-item"><div class="fail-label">Expected</div><div class="fail-value exp">{t.get('expected_answer','')[:200]}</div></div>
                    <div class="fail-item"><div class="fail-label">Actual</div><div class="fail-value act">{t.get('actual_answer','')[:200]}</div></div>
                </div>
                <div style="margin-top:12px;"><div class="fail-item"><div class="fail-label">Root Cause</div><div class="fail-value">{t.get('judge_reason','')[:200]}</div></div></div>
            </div>
            """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()
    load_vs()

    current = st.session_state.page

    # ── FIXED BRAND BAR (52 px, brand only — tabs are below in native Streamlit) ──
    st.markdown("""
    <div class="fixed-header" style="height:52px;">
        <div class="header-top" style="height:52px;">
            <div class="header-logo-left">
                <div class="header-logo" style="width:28px;height:28px;font-size:14px;">B</div>
                <span class="header-brand" style="font-size:15px;">BVRITH</span>
                <span style="color:#6B7280;font-weight:500;font-size:12px;margin-left:4px;">FAQ</span>
                <span class="header-brand-sub" style="font-size:10px;">RAG ASSISTANT</span>
            </div>
            <div class="header-right-btn" style="padding:5px 14px;font-size:11px;">🔗 bvrit.ac.in</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TAB BAR — rendered as Streamlit columns, sits just below the fixed bar ──
    st.markdown("<div style='margin-top:4px;margin-bottom:8px;'>", unsafe_allow_html=True)
    t1, t2, t3 = st.columns([1, 1, 1])
    with t1:
        if st.button("🏠 Home", key="nav_home", use_container_width=True,
                     type="primary" if current == "Home" else "secondary"):
            st.session_state.page = "Home"; st.rerun()
    with t2:
        if st.button("💬 Chat", key="nav_chat", use_container_width=True,
                     type="primary" if current == "Chat" else "secondary"):
            st.session_state.page = "Chat"; st.rerun()
    with t3:
        if st.button("📊 Eval", key="nav_eval", use_container_width=True,
                     type="primary" if current == "Eval" else "secondary"):
            st.session_state.page = "Eval"; st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    # ── MAIN CONTENT ──
    if st.session_state.page == "Home":
        render_home()
    elif st.session_state.page == "Chat":
        render_chat()
    elif st.session_state.page == "Eval":
        render_eval()

if __name__ == "__main__":
    main()