"""FastAPI Web Server for Voice-Enabled RAG System (Indic MSMARCO-XI)."""

import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from src.generation.generator import get_generator
from src.guardrails.confidence import ConfidenceGuardrail
from src.guardrails.safety import CompositeGuardrail
from src.pipeline.harness import RobustExecutionHarness
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.schemas import AudioInputRequest, RAGResponse, TextInputRequest
from src.stt.transcriber import get_transcriber
from src.utils.logging import logger

# Global pipeline instance
pipeline_instance: Optional[RAGPipeline] = None
index_manager_instance: Optional[Any] = None
config_data: Dict[str, Any] = {}


def load_app_config(config_path: str = "configs/config.yaml") -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline_instance, index_manager_instance, config_data
    logger.info("Initializing Voice RAG Pipeline for Web Service...")

    config_data = load_app_config()
    retrieval_cfg = config_data.get("retrieval", {})
    stt_cfg = config_data.get("stt", {})
    guard_cfg = config_data.get("guardrails", {})
    gen_cfg = config_data.get("generation", {})

    device = retrieval_cfg.get("device", "cuda")
    embed_model_name = retrieval_cfg.get("embedding_model", "intfloat/multilingual-e5-small")
    index_path = retrieval_cfg.get("index_path", "data/processed/faiss_index.bin")
    metadata_path = retrieval_cfg.get("metadata_path", "data/processed/passage_metadata.json")

    retrieval_mode = retrieval_cfg.get("mode", "dense")
    if retrieval_mode == "bm25":
        logger.info("Initializing BM25 Sparse Retriever (Low Memory Mode)...")
        from src.retrieval.bm25_retriever import BM25Retriever
        retriever = BM25Retriever(metadata_path=metadata_path, top_k=retrieval_cfg.get("top_k", 5))
    else:
        # 1. Initialize Embedder
        logger.info(f"Loading embedder: {embed_model_name} on {device}...")
        from src.retrieval.embedder import MultilingualE5Embedder
        embedder = MultilingualE5Embedder(model_name_or_path=embed_model_name, device=device, warmup=True)
    
        # 2. Initialize / Load FAISS Index
        from src.retrieval.index import FAISSIndexManager
        index_manager = FAISSIndexManager(dimension=embedder.dimension, index_type=retrieval_cfg.get("index_type", "FlatIP"))
        if Path(index_path).exists() and Path(metadata_path).exists():
            logger.info(f"Loading FAISS index from {index_path}...")
            index_manager.load(index_path, metadata_path)
        else:
            logger.warning("FAISS index not found on disk. Initializing bootstrap index...")
            from benchmarks.run_latency_bench import ensure_index_exists
            index_manager = ensure_index_exists(embedder, index_path, metadata_path)
    
        index_manager_instance = index_manager
    
        # 3. Retriever
        from src.retrieval.retriever import Retriever as DenseRetriever
        retriever = DenseRetriever(embedder=embedder, index_manager=index_manager, top_k=retrieval_cfg.get("top_k", 5))

    # 4. Guardrail
    min_confidence = guard_cfg.get("min_confidence_threshold", 0.75)
    guardrail = CompositeGuardrail(
        confidence_threshold=min_confidence,
        enable_input_safety=guard_cfg.get("enable_input_safety", True),
        enable_off_topic=guard_cfg.get("enable_off_topic_detection", True),
        enable_groundedness=guard_cfg.get("enable_groundedness_check", True),
    )

    # 5. Generator
    generator = get_generator(
        provider=gen_cfg.get("provider", "mock"),
        model_name=gen_cfg.get("model_name", "mock-llm-indic"),
        temperature=gen_cfg.get("temperature", 0.1),
    )

    # 6. STT Transcriber
    stt_provider = stt_cfg.get("provider", "sarvam")
    logger.info(f"Setting up STT Transcriber ({stt_provider})...")
    transcriber = get_transcriber(
        provider=stt_provider,
        model_size=stt_cfg.get("model_size", "tiny"),
        device=device,
        compute_type=stt_cfg.get("compute_type", "int8"),
        beam_size=stt_cfg.get("beam_size", 1),
        vad_filter=stt_cfg.get("vad_filter", False),
    )

    # 7. Orchestration Harness & Pipeline
    harness = RobustExecutionHarness(
        max_retries=config_data.get("harness", {}).get("max_retries", 3),
        backoff_factor=config_data.get("harness", {}).get("backoff_factor", 1.5),
        enable_circuit_breaker=config_data.get("harness", {}).get("enable_circuit_breaker", True),
    )

    pipeline_instance = RAGPipeline(
        transcriber=transcriber,
        retriever=retriever,
        guardrail=guardrail,
        generator=generator,
        harness=harness,
        default_language=config_data.get("dataset", {}).get("language", "hi"),
    )

    logger.info("Voice RAG Web Pipeline successfully initialized.")
    yield
    logger.info("Shutting down Voice RAG Web Pipeline...")


app = FastAPI(
    title="Indic Voice-Enabled RAG API (HH Goa 2026)",
    description="Sub-200ms Voice & Text Indic Retrieval-Augmented Generation system on MSMARCO-XI.",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


HTML_PAGE = """<!DOCTYPE html>
<html lang="hi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live Voice & Text Indic RAG (HH Goa Task #2)</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Noto+Sans+Devanagari:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --surface-color: #111827;
            --surface-card: #1f2937;
            --primary: #38bdf8;
            --primary-hover: #0284c7;
            --accent: #a855f7;
            --text: #f9fafb;
            --text-muted: #9ca3af;
            --border: #374151;
            --success: #22c55e;
            --refusal: #ef4444;
            --warning: #f59e0b;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            background-color: var(--bg-color);
            color: var(--text);
            font-family: 'Inter', 'Noto Sans Devanagari', sans-serif;
            min-height: 100vh;
            padding: 24px 16px;
            display: flex;
            justify-content: center;
        }
        .container {
            width: 100%;
            max-width: 920px;
        }
        header {
            text-align: center;
            margin-bottom: 28px;
        }
        .badge {
            display: inline-block;
            background: rgba(56, 189, 248, 0.1);
            color: var(--primary);
            border: 1px solid rgba(56, 189, 248, 0.3);
            border-radius: 9999px;
            padding: 4px 14px;
            font-size: 0.82rem;
            font-weight: 600;
            margin-bottom: 12px;
        }
        h1 {
            font-size: 2.1rem;
            font-weight: 700;
            background: linear-gradient(135deg, #38bdf8, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }
        p.subtitle {
            color: var(--text-muted);
            font-size: 0.95rem;
        }
        .card {
            background: var(--surface-color);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        }
        .controls-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
            flex-wrap: wrap;
            gap: 10px;
        }
        .input-tabs {
            display: flex;
            gap: 8px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
            flex: 1;
        }
        .tab-btn {
            background: transparent;
            border: none;
            color: var(--text-muted);
            font-size: 0.92rem;
            font-weight: 600;
            cursor: pointer;
            padding: 8px 16px;
            border-radius: 6px;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        .tab-btn.active {
            background: var(--surface-card);
            color: var(--primary);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .lang-picker {
            display: flex;
            align-items: center;
            gap: 8px;
            background: var(--surface-card);
            padding: 6px 12px;
            border: 1px solid var(--border);
            border-radius: 8px;
        }
        .lang-picker select {
            background: transparent;
            color: var(--text);
            border: none;
            font-size: 0.9rem;
            font-weight: 600;
            outline: none;
            cursor: pointer;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        textarea, input[type="text"] {
            width: 100%;
            background: var(--surface-card);
            border: 1px solid var(--border);
            color: var(--text);
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 1rem;
            font-family: inherit;
            outline: none;
            resize: vertical;
        }
        textarea:focus, input[type="text"]:focus {
            border-color: var(--primary);
        }
        .samples {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }
        .sample-chip {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border);
            color: var(--text-muted);
            font-size: 0.8rem;
            padding: 4px 10px;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .sample-chip:hover {
            border-color: var(--primary);
            color: var(--text);
        }
        
        /* Live Voice Command Box */
        .live-voice-box {
            text-align: center;
            padding: 30px 20px;
            background: rgba(31, 41, 55, 0.4);
            border: 2px dashed var(--border);
            border-radius: 12px;
            transition: border-color 0.3s;
        }
        .live-voice-box.recording {
            border-color: var(--refusal);
            background: rgba(239, 68, 68, 0.05);
        }
        .mic-record-btn {
            width: 84px;
            height: 84px;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--primary), var(--accent));
            border: none;
            color: #000;
            font-size: 2rem;
            cursor: pointer;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            margin-bottom: 16px;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
        }
        .mic-record-btn:hover {
            transform: scale(1.06);
            box-shadow: 0 0 30px rgba(56, 189, 248, 0.7);
        }
        .mic-record-btn.recording {
            background: var(--refusal);
            color: #fff;
            animation: pulse-red 1.2s infinite;
            box-shadow: 0 0 25px rgba(239, 68, 68, 0.8);
        }
        @keyframes pulse-red {
            0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
            70% { transform: scale(1.08); box-shadow: 0 0 0 16px rgba(239, 68, 68, 0); }
            100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
        }
        .voice-status-text {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 6px;
        }
        .voice-timer {
            font-size: 0.9rem;
            color: var(--text-muted);
            font-family: monospace;
        }
        .visualizer {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 4px;
            height: 28px;
            margin-top: 14px;
        }
        .v-bar {
            width: 4px;
            height: 6px;
            background: var(--primary);
            border-radius: 2px;
            transition: height 0.1s ease;
        }
        .live-voice-box.recording .v-bar {
            background: var(--refusal);
            animation: bounce-bar 0.6s ease-in-out infinite alternate;
        }
        .live-voice-box.recording .v-bar:nth-child(2) { animation-delay: 0.1s; }
        .live-voice-box.recording .v-bar:nth-child(3) { animation-delay: 0.2s; }
        .live-voice-box.recording .v-bar:nth-child(4) { animation-delay: 0.3s; }
        .live-voice-box.recording .v-bar:nth-child(5) { animation-delay: 0.15s; }
        .live-voice-box.recording .v-bar:nth-child(6) { animation-delay: 0.25s; }
        .live-voice-box.recording .v-bar:nth-child(7) { animation-delay: 0.05s; }
        @keyframes bounce-bar {
            0% { height: 6px; }
            100% { height: 26px; }
        }

        .action-row {
            display: flex;
            justify-content: flex-end;
            margin-top: 18px;
        }
        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--accent));
            color: #000;
            font-weight: 700;
            border: none;
            padding: 12px 28px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.95rem;
            display: inline-flex;
            align-items: center;
            gap: 8px;
            transition: opacity 0.2s;
        }
        .btn-primary:hover { opacity: 0.9; }
        .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
        .results-section { display: none; }
        .result-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 16px;
        }
        .status-pill {
            font-size: 0.8rem;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 9999px;
            text-transform: uppercase;
        }
        .status-pill.pass { background: rgba(34, 197, 94, 0.15); color: var(--success); border: 1px solid var(--success); }
        .status-pill.refusal { background: rgba(239, 68, 68, 0.15); color: var(--refusal); border: 1px solid var(--refusal); }
        .answer-box {
            background: var(--surface-card);
            border-left: 4px solid var(--primary);
            border-radius: 0 8px 8px 0;
            padding: 16px;
            font-size: 1.05rem;
            line-height: 1.6;
            margin-bottom: 20px;
        }
        .answer-box.refusal {
            border-left-color: var(--refusal);
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 12px;
            margin-bottom: 24px;
        }
        .metric-card {
            background: var(--surface-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px;
            text-align: center;
        }
        .metric-card.sla-highlight {
            border-color: var(--primary);
            background: rgba(56, 189, 248, 0.05);
        }
        .metric-label {
            font-size: 0.72rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 4px;
        }
        .metric-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text);
        }
        .metric-value.sla-pass { color: var(--success); }
        .metric-sub {
            font-size: 0.7rem;
            color: var(--text-muted);
            margin-top: 2px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
            font-size: 0.88rem;
        }
        th, td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }
        th { color: var(--text-muted); font-size: 0.75rem; text-transform: uppercase; }
        .score-cell { font-family: monospace; font-weight: 700; color: var(--primary); }
        .loader {
            display: none;
            text-align: center;
            padding: 24px;
        }
        .spinner {
            width: 36px;
            height: 36px;
            border: 3px solid rgba(255,255,255,0.1);
            border-radius: 50%;
            border-top-color: var(--primary);
            animation: spin 0.8s ease-in-out infinite;
            margin: 0 auto 12px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <span class="badge">⚡ Sub-200ms Retrieval SLA | HH Goa 2026</span>
            <h1>Indic Voice RAG Pipeline</h1>
            <p class="subtitle">Live Voice & Text Retrieval-Augmented Generation on MSMARCO-XI with Guardrails & Latency Analytics</p>
        </header>

        <div class="card">
            <div class="controls-bar">
                <div class="input-tabs">
                    <button class="tab-btn active" onclick="switchTab('live')">🎙️ Live Voice</button>
                    <button class="tab-btn" onclick="switchTab('text')">✍️ Text Query</button>
                    <button class="tab-btn" onclick="switchTab('file')">📁 Audio File</button>
                </div>
                <div class="lang-picker">
                    <label style="margin:0; font-size:0.75rem;">🌐 Lang:</label>
                    <select id="langSelect">
                        <option value="hi" selected>Hindi (हिन्दी)</option>
                        <option value="en">English</option>
                        <option value="mr">Marathi (मराठी)</option>
                        <option value="bn">Bengali (বাংলা)</option>
                        <option value="ta">Tamil (தமிழ்)</option>
                        <option value="te">Telugu (తెలుగు)</option>
                        <option value="gu">Gujarati (ગુજરાતી)</option>
                        <option value="kn">Kannada (ಕನ್ನಡ)</option>
                        <option value="ml">Malayalam (മലയാളം)</option>
                        <option value="pa">Punjabi (ਪੰਜਾਬੀ)</option>
                        <option value="ur">Urdu (اردو)</option>
                    </select>
                </div>
            </div>

            <!-- Live Voice Command Tab -->
            <div id="liveTab" class="tab-content active">
                <div id="liveVoiceBox" class="live-voice-box">
                    <button id="micRecordBtn" class="mic-record-btn" onclick="toggleLiveVoiceRecording()">
                        <span id="micIcon">🎙️</span>
                    </button>
                    <div id="voiceStatus" class="voice-status-text">Click microphone to speak your command</div>
                    <div id="voiceTimer" class="voice-timer">Ready • 16kHz PCM</div>
                    <div class="visualizer">
                        <div class="v-bar"></div>
                        <div class="v-bar"></div>
                        <div class="v-bar"></div>
                        <div class="v-bar"></div>
                        <div class="v-bar"></div>
                        <div class="v-bar"></div>
                        <div class="v-bar"></div>
                    </div>
                </div>
                <div class="samples" style="margin-top: 14px;">
                    <span style="font-size: 0.8rem; color: var(--text-muted); width: 100%;">💡 Voice Command Ideas:</span>
                    <span class="sample-chip">"भारत की राजधानी क्या है?"</span>
                    <span class="sample-chip">"सौर ऊर्जा के क्या लाभ हैं?"</span>
                    <span class="sample-chip">"पौधों में प्रकाश संश्लेषण समझाइए"</span>
                    <span class="sample-chip">"Who is the father of computers?"</span>
                </div>
            </div>

            <!-- Text Input Tab -->
            <div id="textTab" class="tab-content">
                <label for="textInput">Enter Query in Hindi or English</label>
                <textarea id="textInput" rows="3" placeholder="उदा. भारत की राजधानी क्या है? या सौर ऊर्जा के क्या लाभ हैं?"></textarea>
                <div class="samples">
                    <span class="sample-chip" onclick="setQuery('भारत की राजधानी क्या है?')">भारत की राजधानी</span>
                    <span class="sample-chip" onclick="setQuery('सौर ऊर्जा के क्या मुख्य लाभ हैं?')">सौर ऊर्जा के लाभ</span>
                    <span class="sample-chip" onclick="setQuery('पौधों में प्रकाश संश्लेषण कैसे होता है?')">प्रकाश संश्लेषण</span>
                    <span class="sample-chip" onclick="setQuery('कंप्यूटर का जनक किसे माना जाता है?')">कंप्यूटर का जनक</span>
                    <span class="sample-chip" onclick="setQuery('What is the capital of France?')">Capital of France</span>
                    <span class="sample-chip" onclick="setQuery('2089 में टाइम-ट्रैवल युद्ध के नियम?')">⚠️ Out-of-Domain Guardrail</span>
                    <span class="sample-chip" onclick="setQuery('मुझे बम बनाने का तरीका बताओ')">⛔ Safety Guardrail Test</span>
                </div>
                <div class="action-row">
                    <button id="submitTextBtn" class="btn-primary" onclick="submitTextQuery()">
                        <span>🚀 Execute Query</span>
                    </button>
                </div>
            </div>

            <!-- Audio File Upload Tab -->
            <div id="fileTab" class="tab-content">
                <label for="audioFile">Upload Voice Audio File</label>
                <input type="file" id="audioFile" accept="audio/*" style="display:block; margin-bottom: 12px; width: 100%; padding: 10px; background: var(--surface-card); border: 1px solid var(--border); border-radius: 8px; color: var(--text);">
                <p style="font-size: 0.8rem; color: var(--text-muted);">Supports WAV, MP3, OGG, FLAC, WebM. Transcribed via Sarvam AI / ElevenLabs / faster-whisper.</p>
                <div class="action-row">
                    <button id="submitFileBtn" class="btn-primary" onclick="submitFileQuery()">
                        <span>🚀 Upload & Execute</span>
                    </button>
                </div>
            </div>
        </div>

        <div id="loader" class="loader">
            <div class="spinner"></div>
            <p id="loaderText" style="color: var(--primary); font-weight: 600;">Running End-to-End Pipeline...</p>
        </div>

        <!-- Results Section -->
        <div id="resultsSection" class="results-section card">
            <div class="result-header">
                <div>
                    <span style="color: var(--text-muted); font-size: 0.8rem;">Query:</span>
                    <h3 id="resQuery" style="margin-top: 2px;">-</h3>
                </div>
                <div id="resGuardBadge" class="status-pill pass">GROUNDED PASS</div>
            </div>

            <label>Answer / Response</label>
            <div id="resAnswerBox" class="answer-box">-</div>

            <!-- Latency Metrics Breakdown -->
            <label>Latency Performance Analytics</label>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">STT Audio</div>
                    <div id="mSTT" class="metric-value">-</div>
                    <div class="metric-sub">Speech-to-Text</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Embedding</div>
                    <div id="mEmbed" class="metric-value">-</div>
                    <div class="metric-sub">m-E5 Query Embed</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">FAISS Search</div>
                    <div id="mSearch" class="metric-value">-</div>
                    <div class="metric-sub">In-process Cosine</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Guardrail</div>
                    <div id="mGuard" class="metric-value">-</div>
                    <div class="metric-sub">Confidence Check</div>
                </div>
                <div class="metric-card sla-highlight">
                    <div class="metric-label">⚡ RETRIEVAL LEG</div>
                    <div id="mRetLeg" class="metric-value sla-pass">-</div>
                    <div class="metric-sub">SLA Budget: &lt; 200ms</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Generation</div>
                    <div id="mGen" class="metric-value">-</div>
                    <div class="metric-sub">Token Generation</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Total Pipeline</div>
                    <div id="mTotal" class="metric-value">-</div>
                    <div class="metric-sub">End-to-End</div>
                </div>
            </div>

            <!-- Retrieved Passages Table -->
            <label>Retrieved Grounding Passages (Top-K)</label>
            <table id="passagesTable">
                <thead>
                    <tr>
                        <th style="width: 50px;">Rank</th>
                        <th style="width: 80px;">Score</th>
                        <th style="width: 120px;">Passage ID</th>
                        <th>Text Excerpt</th>
                    </tr>
                </thead>
                <tbody id="passagesTbody">
                </tbody>
            </table>
        </div>
    </div>

    <script>
        let currentTab = 'live';
        let isRecording = false;
        let audioContext = null;
        let mediaStream = null;
        let scriptProcessor = null;
        let audioBuffers = [];
        let recordStartTime = 0;
        let timerInterval = null;

        function switchTab(tab) {
            currentTab = tab;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            if (tab === 'live') {
                document.querySelectorAll('.tab-btn')[0].classList.add('active');
                document.getElementById('liveTab').classList.add('active');
            } else if (tab === 'text') {
                document.querySelectorAll('.tab-btn')[1].classList.add('active');
                document.getElementById('textTab').classList.add('active');
            } else {
                document.querySelectorAll('.tab-btn')[2].classList.add('active');
                document.getElementById('fileTab').classList.add('active');
            }
        }

        function setQuery(text) {
            document.getElementById('textInput').value = text;
        }

        // Live Voice Command Recorder (PCM WAV 16kHz)
        async function toggleLiveVoiceRecording() {
            if (!isRecording) {
                await startRecording();
            } else {
                await stopRecordingAndSubmit();
            }
        }

        async function startRecording() {
            try {
                mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { sampleRate: 16000, channelCount: 1 } });
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(mediaStream);
                
                // Ensure audio context is active
                if (audioContext.state === 'suspended') {
                    await audioContext.resume();
                }
                
                audioBuffers = [];
                scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                scriptProcessor.onaudioprocess = (e) => {
                    if (!isRecording) return;
                    const inputData = e.inputBuffer.getChannelData(0);
                    audioBuffers.push(new Float32Array(inputData));
                };

                // Prevent echo/feedback loop by muting the output
                const gainNode = audioContext.createGain();
                gainNode.gain.value = 0;
                source.connect(scriptProcessor);
                scriptProcessor.connect(gainNode);
                gainNode.connect(audioContext.destination);

                isRecording = true;
                recordStartTime = Date.now();
                
                const btn = document.getElementById('micRecordBtn');
                btn.classList.add('recording');
                document.getElementById('micIcon').textContent = '⏹️';
                document.getElementById('liveVoiceBox').classList.add('recording');
                document.getElementById('voiceStatus').textContent = '🔴 Listening... Speak your query now!';

                timerInterval = setInterval(() => {
                    const elapsed = ((Date.now() - recordStartTime) / 1000).toFixed(1);
                    document.getElementById('voiceTimer').textContent = `Recording • ${elapsed}s / max 15.0s`;
                    if (elapsed >= 15.0) {
                        stopRecordingAndSubmit();
                    }
                }, 100);

            } catch (err) {
                alert('Microphone access error: ' + err.message + '. Please ensure microphone permission is granted in browser.');
            }
        }

        async function stopRecordingAndSubmit() {
            if (!isRecording) return;
            isRecording = false;
            clearInterval(timerInterval);

            const btn = document.getElementById('micRecordBtn');
            btn.classList.remove('recording');
            document.getElementById('micIcon').textContent = '🎙️';
            document.getElementById('liveVoiceBox').classList.remove('recording');
            document.getElementById('voiceStatus').textContent = 'Processing live voice command...';
            document.getElementById('voiceTimer').textContent = 'Finalizing audio...';

            if (scriptProcessor) scriptProcessor.disconnect();
            if (mediaStream) mediaStream.getTracks().forEach(t => t.stop());
            if (audioContext && audioContext.state !== 'closed') audioContext.close();

            // Flatten audio buffers
            let totalLength = audioBuffers.reduce((acc, buf) => acc + buf.length, 0);
            if (totalLength === 0) {
                document.getElementById('voiceStatus').textContent = 'No audio detected. Click to try again.';
                return;
            }

            let merged = new Float32Array(totalLength);
            let offset = 0;
            for (let b of audioBuffers) {
                merged.set(b, offset);
                offset += b.length;
            }

            // Encode to standard 16kHz 16-bit Mono WAV
            const wavBlob = encodeWAV(merged, 16000);
            await sendAudioBlob(wavBlob);
        }

        function encodeWAV(samples, sampleRate) {
            const buffer = new ArrayBuffer(44 + samples.length * 2);
            const view = new DataView(buffer);

            function writeString(view, offset, string) {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            }

            /* RIFF identifier */
            writeString(view, 0, 'RIFF');
            /* RIFF chunk length */
            view.setUint32(4, 36 + samples.length * 2, true);
            /* RIFF type */
            writeString(view, 8, 'WAVE');
            /* format chunk identifier */
            writeString(view, 12, 'fmt ');
            /* format chunk length */
            view.setUint32(16, 16, true);
            /* sample format (raw PCM) */
            view.setUint16(20, 1, true);
            /* channel count (1 = mono) */
            view.setUint16(22, 1, true);
            /* sample rate */
            view.setUint32(24, sampleRate, true);
            /* byte rate (sample rate * block align) */
            view.setUint32(28, sampleRate * 2, true);
            /* block align (channel count * bytes per sample) */
            view.setUint16(32, 2, true);
            /* bits per sample */
            view.setUint16(34, 16, true);
            /* data chunk identifier */
            writeString(view, 36, 'data');
            /* data chunk length */
            view.setUint32(40, samples.length * 2, true);

            // Write 16-bit PCM samples
            let idx = 44;
            for (let i = 0; i < samples.length; i++) {
                let s = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(idx, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                idx += 2;
            }

            return new Blob([view], { type: 'audio/wav' });
        }

        async function sendAudioBlob(blob) {
            const loader = document.getElementById('loader');
            const results = document.getElementById('resultsSection');
            const lang = document.getElementById('langSelect').value;

            loader.style.display = 'block';
            document.getElementById('loaderText').textContent = 'Transcribing voice & executing sub-200ms RAG retrieval...';
            results.style.display = 'none';

            try {
                const formData = new FormData();
                formData.append('file', blob, 'live_command.wav');
                formData.append('language', lang);

                const resp = await fetch(`/api/query/voice?language=${lang}`, {
                    method: 'POST',
                    body: formData
                });
                if (!resp.ok) throw new Error(await resp.text());
                const resData = await resp.json();
                renderResults(resData);
                document.getElementById('voiceStatus').textContent = '✓ Voice command answered! Click to speak again.';
                document.getElementById('voiceTimer').textContent = 'Ready';
            } catch (err) {
                alert('Pipeline error: ' + err.message);
                document.getElementById('voiceStatus').textContent = 'Error processing command. Click to retry.';
            } finally {
                loader.style.display = 'none';
            }
        }

        async function submitTextQuery() {
            const btn = document.getElementById('submitTextBtn');
            const loader = document.getElementById('loader');
            const results = document.getElementById('resultsSection');
            const query = document.getElementById('textInput').value.trim();
            const lang = document.getElementById('langSelect').value;

            if (!query) { alert('Please enter a query.'); return; }

            btn.disabled = true;
            loader.style.display = 'block';
            document.getElementById('loaderText').textContent = 'Executing retrieval & guardrails...';
            results.style.display = 'none';

            try {
                const resp = await fetch('/api/query/text', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, language: lang })
                });
                if (!resp.ok) throw new Error(await resp.text());
                const resData = await resp.json();
                renderResults(resData);
            } catch (err) {
                alert('Error running pipeline: ' + err.message);
            } finally {
                btn.disabled = false;
                loader.style.display = 'none';
            }
        }

        async function submitFileQuery() {
            const btn = document.getElementById('submitFileBtn');
            const loader = document.getElementById('loader');
            const fileInput = document.getElementById('audioFile');
            const lang = document.getElementById('langSelect').value;

            if (!fileInput.files.length) { alert('Please select an audio file.'); return; }

            btn.disabled = true;
            loader.style.display = 'block';
            document.getElementById('loaderText').textContent = 'Transcribing audio file & retrieving...';

            try {
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('language', lang);

                const resp = await fetch(`/api/query/voice?language=${lang}`, {
                    method: 'POST',
                    body: formData
                });
                if (!resp.ok) throw new Error(await resp.text());
                const resData = await resp.json();
                renderResults(resData);
            } catch (err) {
                alert('Error running pipeline: ' + err.message);
            } finally {
                btn.disabled = false;
                loader.style.display = 'none';
            }
        }

        function renderResults(data) {
            const results = document.getElementById('resultsSection');
            results.style.display = 'block';

            document.getElementById('resQuery').textContent = data.query;
            const answerBox = document.getElementById('resAnswerBox');
            answerBox.textContent = data.answer;

            const badge = document.getElementById('resGuardBadge');
            if (data.is_refusal) {
                badge.className = 'status-pill refusal';
                badge.textContent = `⛔ REFUSAL (Score: ${data.confidence_score.toFixed(3)})`;
                answerBox.className = 'answer-box refusal';
            } else {
                badge.className = 'status-pill pass';
                badge.textContent = `✓ GROUNDED PASS (Score: ${data.confidence_score.toFixed(3)})`;
                answerBox.className = 'answer-box';
            }

            // Latencies
            const lat = data.latencies || {};
            document.getElementById('mSTT').textContent = (lat.stt_ms || 0).toFixed(1) + ' ms';
            document.getElementById('mEmbed').textContent = (lat.embed_ms || 0).toFixed(1) + ' ms';
            document.getElementById('mSearch').textContent = (lat.retrieve_ms || 0).toFixed(1) + ' ms';
            document.getElementById('mGuard').textContent = (lat.guardrail_ms || 0).toFixed(1) + ' ms';

            const retLeg = lat.retrieval_leg_total_ms || 0;
            const mRet = document.getElementById('mRetLeg');
            mRet.textContent = retLeg.toFixed(1) + ' ms';
            mRet.className = retLeg < 200.0 ? 'metric-value sla-pass' : 'metric-value';

            document.getElementById('mGen').textContent = (lat.generation_ms || 0).toFixed(1) + ' ms';
            document.getElementById('mTotal').textContent = (lat.total_pipeline_ms || 0).toFixed(1) + ' ms';

            // Passages
            const tbody = document.getElementById('passagesTbody');
            tbody.innerHTML = '';
            if (data.retrieved_passages && data.retrieved_passages.length > 0) {
                data.retrieved_passages.forEach(p => {
                    const tr = document.createElement('tr');
                    tr.innerHTML = `
                        <td style="text-align: center;">${p.rank}</td>
                        <td class="score-cell">${p.score.toFixed(4)}</td>
                        <td><small style="color: var(--text-muted);">${p.passage_id}</small></td>
                        <td>${p.text}</td>
                    `;
                    tbody.appendChild(tr);
                });
            } else {
                tbody.innerHTML = '<tr><td colspan="4" style="text-align:center; color: var(--text-muted);">No passages retrieved.</td></tr>';
            }

            results.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    """Serves the interactive web interface."""
    return HTML_PAGE


class TextQueryRequestModel(BaseModel):
    query: str = Field(..., description="Query text in Hindi or English")
    language: str = Field("hi", description="ISO language code ('hi' or 'en')")


@app.post("/api/query/text", response_model=RAGResponse)
async def api_query_text(req: TextQueryRequestModel):
    """Processes a text query through Retrieval -> Guardrails -> Generation."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline is not initialized")
    try:
        response = pipeline_instance.process_text(TextInputRequest(query=req.query, language=req.language))
        return response
    except Exception as e:
        logger.error(f"[API] Error in query_text: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query/voice", response_model=RAGResponse)
async def api_query_voice(file: UploadFile = File(...), language: str = "hi"):
    """Processes a voice audio file through STT -> Retrieval -> Guardrails -> Generation."""
    if not pipeline_instance:
        raise HTTPException(status_code=503, detail="Pipeline is not initialized")

    temp_path = None
    try:
        content = await file.read()
        suffix = Path(file.filename or "audio.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            temp_path = tmp.name

        response = pipeline_instance.process_voice(AudioInputRequest(audio_path=temp_path, language=language))
        return response
    except Exception as e:
        logger.error(f"[API] Error in query_voice: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline_instance is not None,
        "index_vectors": index_manager_instance.total_vectors if index_manager_instance else 0,
    }


@app.get("/api/stats")
async def get_stats():
    """Returns vector index and system metadata statistics."""
    if not index_manager_instance:
        raise HTTPException(status_code=503, detail="Index manager not initialized")
    return {
        "total_vectors": index_manager_instance.total_vectors,
        "dimension": index_manager_instance.dimension,
        "index_type": index_manager_instance.index_type,
        "config": {
            "dataset": config_data.get("dataset", {}),
            "chunking": config_data.get("chunking", {}),
            "retrieval": config_data.get("retrieval", {}),
            "stt": config_data.get("stt", {}),
        },
    }
