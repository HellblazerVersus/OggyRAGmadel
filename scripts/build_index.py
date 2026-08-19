"""One-off script: Load MSMARCO-XI dataset, chunk passages, embed, and build FAISS index."""

import argparse
import sys
import time
from pathlib import Path
import yaml
from src.ingestion.chunkers import get_chunker
from src.ingestion.loaders import MSMARCOLoader
from src.retrieval.embedder import MultilingualE5Embedder
from src.retrieval.index import FAISSIndexManager
from src.utils.logging import logger, console
from rich.table import Table


def parse_args():
    parser = argparse.ArgumentParser(description="Build FAISS vector index for Voice RAG")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--language", type=str, default=None, help="Dataset language code (e.g. 'hi')")
    parser.add_argument("--max-passages", type=int, default=None, help="Maximum number of passages to index")
    parser.add_argument("--chunker", type=str, default=None, choices=["fixed", "sentence"], help="Chunking strategy")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda' or 'cpu')")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--index-path", type=str, default=None, help="Target FAISS index output path")
    parser.add_argument("--metadata-path", type=str, default=None, help="Target metadata output path")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    p = Path(config_path)
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Resolve arguments with config fallbacks
    dataset_name = cfg.get("dataset", {}).get("name", "ai4bharat/MSMARCO-XI")
    language = args.language or cfg.get("dataset", {}).get("language", "hi")
    split = cfg.get("dataset", {}).get("split", "train")
    max_passages = args.max_passages or cfg.get("dataset", {}).get("max_passages", 1000)

    chunker_strategy = args.chunker or cfg.get("chunking", {}).get("strategy", "sentence")
    device = args.device or cfg.get("retrieval", {}).get("device", "cuda")
    embed_model_name = cfg.get("retrieval", {}).get("embedding_model", "intfloat/multilingual-e5-base")
    batch_size = args.batch_size or cfg.get("retrieval", {}).get("batch_size", 64)

    index_path = args.index_path or cfg.get("retrieval", {}).get("index_path", "data/processed/faiss_index.bin")
    metadata_path = args.metadata_path or cfg.get("retrieval", {}).get("metadata_path", "data/processed/passage_metadata.json")

    console.rule("[bold cyan]Voice RAG: FAISS Index Builder[/bold cyan]")
    logger.info(f"Dataset: {dataset_name} | Language: {language} | Max Passages: {max_passages}")
    logger.info(f"Chunker: {chunker_strategy} | Model: {embed_model_name} | Device: {device}")

    # 1. Load Raw Passages
    t0 = time.time()
    loader = MSMARCOLoader(
        dataset_name=dataset_name,
        language=language,
        split=split,
        cache_dir=cfg.get("dataset", {}).get("raw_cache_dir", "data/raw"),
    )
    raw_passages = loader.load_passages(max_passages=max_passages)
    logger.info(f"Loaded {len(raw_passages)} raw passages in {time.time() - t0:.2f}s")

    # 2. Chunk Passages
    t1 = time.time()
    chunker = get_chunker(
        strategy=chunker_strategy,
        window_size=cfg.get("chunking", {}).get("fixed_window_size", 256),
        overlap=cfg.get("chunking", {}).get("fixed_window_overlap", 32),
        max_tokens=cfg.get("chunking", {}).get("sentence_max_tokens", 300),
        overlap_sentences=cfg.get("chunking", {}).get("sentence_overlap_sentences", 1),
    )
    chunks = chunker.chunk_passages(raw_passages)
    logger.info(f"Generated {len(chunks)} chunks using strategy '{chunker_strategy}' in {time.time() - t1:.2f}s")

    if not chunks:
        logger.error("No chunks were generated. Aborting index creation.")
        sys.exit(1)

    # 3. Compute Embeddings
    t2 = time.time()
    embedder = MultilingualE5Embedder(model_name_or_path=embed_model_name, device=device)
    chunk_texts = [c.text for c in chunks]
    logger.info(f"Embedding {len(chunk_texts)} chunks (batch size: {batch_size})...")
    embeddings = embedder.embed_passages(chunk_texts, batch_size=batch_size)
    embed_time = time.time() - t2
    logger.info(f"Computed embeddings shape {embeddings.shape} in {embed_time:.2f}s ({len(chunk_texts)/embed_time:.1f} chunks/sec)")

    # 4. Build and Persist FAISS Index
    t3 = time.time()
    index_mgr = FAISSIndexManager(dimension=embedder.dimension, index_type="FlatIP")
    index_mgr.add_chunks(chunks, embeddings)
    index_mgr.save(index_path=index_path, metadata_path=metadata_path)
    logger.info(f"Saved FAISS index ({index_mgr.total_vectors} vectors) to {index_path} in {time.time() - t3:.2f}s")

    table = Table(title="Index Build Summary", show_header=True, header_style="bold green")
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")
    table.add_row("Raw Passages", str(len(raw_passages)))
    table.add_row("Total Chunks", str(len(chunks)))
    table.add_row("Vector Dimension", str(embedder.dimension))
    table.add_row("FAISS Index Type", "IndexFlatIP (Cosine Similarity)")
    table.add_row("Index Output File", index_path)
    table.add_row("Metadata Output File", metadata_path)
    table.add_row("Total Time", f"{time.time() - t0:.2f}s")
    console.print(table)
    console.print("[bold green]✓ Index build completed successfully![/bold green]")


if __name__ == "__main__":
    main()
