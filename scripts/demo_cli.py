"""Interactive CLI Demonstration for Voice-Enabled Indic RAG."""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional
import numpy as np
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.generation.generator import MockGenerator, get_generator
from src.guardrails.confidence import ConfidenceGuardrail
from src.guardrails.safety import CompositeGuardrail
from src.pipeline.harness import RobustExecutionHarness
from src.pipeline.rag_pipeline import RAGPipeline
from src.pipeline.schemas import AudioInputRequest, TextInputRequest
from src.retrieval.embedder import MultilingualE5Embedder
from src.retrieval.index import FAISSIndexManager
from src.retrieval.retriever import Retriever
from src.stt.live_capture import list_audio_input_devices, record_live_voice
from src.stt.transcriber import get_transcriber
from src.utils.logging import logger

console = Console()


def render_response(resp, is_voice: bool = False):
    """Renders a formatted RAG response with stage latencies to the terminal."""
    console.rule("[bold cyan]RAG Query Result[/bold cyan]")

    if is_voice:
        console.print(f"[bold cyan]Transcribed Voice Input:[/bold cyan] {resp.query}")
    else:
        console.print(f"[bold cyan]Query:[/bold cyan] {resp.query}")

    # Guardrail status
    if resp.is_refusal:
        guard_text = f"[bold red]⛔ REFUSAL / ABSTAIN (Confidence: {resp.confidence_score:.4f})[/bold red]"
    else:
        guard_text = f"[bold green]✓ GROUNDED PASS (Confidence: {resp.confidence_score:.4f})[/bold green]"
    console.print(f"[bold]Guardrail Status:[/bold] {guard_text}")

    # Answer panel
    border_color = "red" if resp.is_refusal else "green"
    title = "Refusal Notice" if resp.is_refusal else "Generated Answer"
    console.print(Panel(f"[bold]{resp.answer}[/bold]", title=title, border_style=border_color))

    # Retrieved passages table
    if resp.retrieved_passages:
        p_table = Table(title="Retrieved Grounding Passages (Top-K)", show_header=True, header_style="bold blue")
        p_table.add_column("Rank", justify="center", width=6)
        p_table.add_column("Score", justify="center", width=10)
        p_table.add_column("Passage ID", width=16)
        p_table.add_column("Text Excerpt", style="dim")

        for p in resp.retrieved_passages:
            snippet = p.text[:120] + "..." if len(p.text) > 120 else p.text
            p_table.add_row(str(p.rank), f"{p.score:.4f}", p.passage_id, snippet)
        console.print(p_table)

    # Latency Breakdown Table
    lat_table = Table(title="Stage Latency Breakdown", show_header=True, header_style="bold magenta")
    lat_table.add_column("Stage", style="bold")
    lat_table.add_column("Latency (ms)", justify="right")
    lat_table.add_column("Budget / Target", justify="center")

    if is_voice:
        lat_table.add_row("1. Speech-to-Text (STT)", f"{resp.latencies.stt_ms:.2f} ms", "-")

    lat_table.add_row("2. Query Embedding", f"{resp.latencies.embed_ms:.2f} ms", "-")
    lat_table.add_row("3. FAISS Vector Search", f"{resp.latencies.retrieve_ms:.2f} ms", "-")
    lat_table.add_row("4. Guardrail Check", f"{resp.latencies.guardrail_ms:.2f} ms", "-")

    # Retrieval Leg Highlight
    ret_sla = "✓ <200ms" if resp.latencies.retrieval_leg_total_ms < 200.0 else "✗ >200ms"
    lat_table.add_row(
        "[bold yellow]⚡ RETRIEVAL LEG TOTAL[/bold yellow]",
        f"[bold yellow]{resp.latencies.retrieval_leg_total_ms:.2f} ms[/bold yellow]",
        f"[bold yellow]{ret_sla}[/bold yellow]",
    )

    lat_table.add_row("5. LLM Answer Generation", f"{resp.latencies.generation_ms:.2f} ms", "Reported Separately")
    lat_table.add_row("[bold]TOTAL PIPELINE[/bold]", f"[bold]{resp.latencies.total_pipeline_ms:.2f} ms[/bold]", "-")

    console.print(lat_table)
    console.print()


def show_devices():
    """Lists available microphone devices in a formatted table."""
    devices = list_audio_input_devices()
    table = Table(title="Available Audio Input Devices (Microphones)", show_header=True, header_style="bold cyan")
    table.add_column("Index", justify="center", width=8)
    table.add_column("Device Name", style="bold")
    table.add_column("Input Channels", justify="center", width=16)
    table.add_column("Sample Rate (Hz)", justify="right", width=18)

    if not devices:
        table.add_row("-", "No active microphone devices detected", "-", "-")
    else:
        for d in devices:
            table.add_row(str(d["index"]), str(d["name"]), str(d["channels"]), f"{d['default_samplerate']:.0f}")

    console.print(table)


def main():
    parser = argparse.ArgumentParser(description="Voice-Enabled RAG Live Voice & Text Demo CLI")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--query-text", type=str, default=None, help="Direct text query")
    parser.add_argument("--audio-file", type=str, default=None, help="Path to audio file (WAV/MP3)")
    parser.add_argument("--live", "--mic", dest="live", action="store_true", help="Start Live Voice Command mode (records from microphone)")
    parser.add_argument("--duration", "-d", type=float, default=5.0, help="Live voice recording duration in seconds (default: 5.0)")
    parser.add_argument("--auto-stop", "--vad", dest="auto_stop", action="store_true", help="Automatically stop recording when speech pause is detected")
    parser.add_argument("--language", "-l", type=str, default="hi", help="ISO language code for speech & retrieval (e.g. 'hi', 'en', 'mr', 'ta')")
    parser.add_argument("--audio-device", type=str, default=None, help="Microphone device index or name substring")
    parser.add_argument("--list-devices", action="store_true", help="List all available audio input devices and exit")
    parser.add_argument("--single-shot", "--once", dest="single_shot", action="store_true", help="Record a single voice command, execute, and exit")
    parser.add_argument("--device", type=str, default=None, help="Computation device ('cuda' or 'cpu')")
    parser.add_argument("--stt-provider", type=str, default=None, choices=["sarvam", "elevenlabs", "faster_whisper", "mock"], help="STT provider")
    parser.add_argument("--stt-model", type=str, default=None, help="STT model size (e.g. tiny, base, large-v3-turbo)")
    parser.add_argument("--compute-type", type=str, default=None, help="STT compute type (int8, float16, float32)")
    args = parser.parse_args()

    if args.list_devices:
        show_devices()
        return

    # Load configuration
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve compute devices with fallback
    import torch
    cuda_available = torch.cuda.is_available()
    
    device = args.device or os.environ.get("RETRIEVAL_DEVICE") or cfg.get("retrieval", {}).get("device", "cuda")
    if device == "cuda" and not cuda_available:
        device = "cpu"

    embed_model_name = cfg.get("retrieval", {}).get("embedding_model", "intfloat/multilingual-e5-small")
    index_path = cfg.get("retrieval", {}).get("index_path", "data/processed/faiss_index.bin")
    metadata_path = cfg.get("retrieval", {}).get("metadata_path", "data/processed/passage_metadata.json")
    min_confidence = cfg.get("guardrails", {}).get("min_confidence_threshold", 0.75)

    console.rule("[bold cyan]Initializing Voice-Enabled Indic RAG Pipeline[/bold cyan]")

    # Setup Embedder and Index
    embedder = MultilingualE5Embedder(model_name_or_path=embed_model_name, device=device, warmup=True)
    index_mgr = FAISSIndexManager(dimension=embedder.dimension, index_type="FlatIP")

    if not (Path(index_path).exists() and Path(metadata_path).exists()):
        console.print("[yellow]Index not found on disk. Bootstrapping index...[/yellow]")
        from benchmarks.run_latency_bench import ensure_index_exists
        index_mgr = ensure_index_exists(embedder, index_path, metadata_path)
    else:
        index_mgr.load(index_path, metadata_path)

    retriever = Retriever(embedder=embedder, index_manager=index_mgr, top_k=cfg.get("retrieval", {}).get("top_k", 5))
    
    # Use CompositeGuardrail
    guardrail = CompositeGuardrail(
        confidence_threshold=min_confidence,
        enable_input_safety=cfg.get("guardrails", {}).get("enable_input_safety", True),
        enable_off_topic=cfg.get("guardrails", {}).get("enable_off_topic_detection", True),
        enable_groundedness=cfg.get("guardrails", {}).get("enable_groundedness_check", True),
    )
    generator = get_generator(provider=cfg.get("generation", {}).get("provider", "mock"))

    # Setup STT
    stt_cfg = cfg.get("stt", {})
    stt_provider = args.stt_provider or os.environ.get("STT_PROVIDER") or stt_cfg.get("provider", "sarvam")
    stt_device = args.device or os.environ.get("WHISPER_DEVICE") or stt_cfg.get("device", "cuda")
    if stt_device == "cuda" and not cuda_available:
        stt_device = "cpu"

    transcriber = get_transcriber(
        provider=stt_provider,
        model_size=args.stt_model or stt_cfg.get("model_size", "tiny"),
        device=stt_device,
        compute_type=args.compute_type or stt_cfg.get("compute_type", "int8"),
        beam_size=stt_cfg.get("beam_size", 1),
        vad_filter=stt_cfg.get("vad_filter", False),
    )

    pipeline = RAGPipeline(
        transcriber=transcriber,
        retriever=retriever,
        guardrail=guardrail,
        generator=generator,
        harness=RobustExecutionHarness(),
        default_language=args.language or cfg.get("dataset", {}).get("language", "hi"),
    )

    console.print(f"[bold green]✓ Pipeline initialized successfully! (STT: {stt_provider}, Lang: {args.language})[/bold green]\n")

    # 1. Text Query mode
    if args.query_text:
        resp = pipeline.process_text(TextInputRequest(query=args.query_text, language=args.language))
        render_response(resp, is_voice=False)
        return

    # 2. Audio File mode
    if args.audio_file:
        resp = pipeline.process_voice(AudioInputRequest(audio_path=args.audio_file, language=args.language))
        render_response(resp, is_voice=True)
        return

    # 3. Live Voice Command Mode
    if args.live:
        console.rule("[bold magenta]🎙️ Live Voice Command Mode[/bold magenta]")
        console.print(f"[dim]Language: {args.language} | Max Duration: {args.duration:.1f}s | Auto-Stop: {args.auto_stop}[/dim]")
        console.print("[dim]Press Enter to trigger recording. Speak clearly into your microphone.[/dim]")
        if not args.single_shot:
            console.print("[dim]Press Ctrl+C at any time to exit.\n[/dim]")

        try:
            while True:
                console.input("\n[bold yellow]👉 Press [Enter] to start recording voice command...[/bold yellow] ")
                audio_file = record_live_voice(
                    duration=args.duration,
                    device=args.audio_device,
                    auto_stop_silence=args.auto_stop,
                )
                if audio_file:
                    try:
                        resp = pipeline.process_voice(AudioInputRequest(audio_path=audio_file, language=args.language))
                        render_response(resp, is_voice=True)
                    finally:
                        if os.path.exists(audio_file):
                            try:
                                os.remove(audio_file)
                            except Exception:
                                pass

                if args.single_shot:
                    break

        except KeyboardInterrupt:
            console.print("\n[bold green]Exiting live voice mode. Goodbye![/bold green]")
        return

    # 4. Interactive Text REPL Loop
    console.print("[bold]Interactive Mode[/bold] (Type a query in Hindi/English, or type 'exit' to quit):\n")
    while True:
        try:
            query = console.input("[bold yellow]Enter Query > [/bold yellow]").strip()
            if not query or query.lower() in ("exit", "quit", "q"):
                break
            resp = pipeline.process_text(TextInputRequest(query=query, language=args.language))
            render_response(resp, is_voice=False)
        except (KeyboardInterrupt, EOFError):
            break

    console.print("\n[bold green]Goodbye![/bold green]")


if __name__ == "__main__":
    main()

