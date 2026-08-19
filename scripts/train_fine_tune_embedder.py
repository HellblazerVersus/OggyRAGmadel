"""Fine-tune a lightweight multilingual embedding model (e.g. multilingual-e5-small)
on the ai4bharat/MSMARCO-XI Indic dataset for ultra-low latency (< 50ms) retrieval.
"""

import argparse
import os
from pathlib import Path
import time
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import MultipleNegativesRankingLoss
from src.ingestion.loaders import MSMARCOLoader
from src.utils.logging import logger, console


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune embedding model on Indic MSMARCO-XI")
    parser.add_argument(
        "--base-model",
        type=str,
        default="intfloat/multilingual-e5-small",
        help="Base HuggingFace model (e.g., intfloat/multilingual-e5-small)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/indic_e5_small_finetuned",
        help="Output directory to save fine-tuned model checkpoint",
    )
    parser.add_argument("--language", type=str, default="hi", help="Language code (default: 'hi')")
    parser.add_argument("--max-samples", type=int, default=1000, help="Max training query-passage pairs")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda' or 'cpu')")
    return parser.parse_args()


def prepare_training_dataset(loader: MSMARCOLoader, max_samples: int = 1000) -> Dataset:
    """Prepares (anchor, positive) pairs formatted with E5 prefixes as a HuggingFace Dataset."""
    raw_passages = loader.load_passages(max_passages=max_samples)
    anchors = []
    positives = []

    for idx, passage in enumerate(raw_passages):
        query = passage.metadata.get("query")
        text = passage.text
        if not text:
            continue
        
        # If dataset row has paired query, use it; otherwise generate anchor
        if query and str(query).strip():
            q_text = f"query: {str(query).strip()}"
        else:
            first_sentence = text.split("।")[0].strip() if "।" in text else text[:80]
            q_text = f"query: {first_sentence}"

        p_text = f"passage: {text.strip()}"
        anchors.append(q_text)
        positives.append(p_text)

    # Fallback curated pairs if empty
    if not anchors:
        curated = [
            ("query: भारत की राजधानी क्या है?", "passage: नई दिल्ली भारत की आधिकारिक राजधानी और प्रशासनिक केंद्र है।"),
            ("query: सौर ऊर्जा क्या है?", "passage: सौर ऊर्जा सूर्य से प्राप्त होने वाली अक्षय और स्वच्छ ऊर्जा है।"),
            ("query: कंप्यूटर के जनक कौन हैं?", "passage: चार्ल्स बैबेज को कंप्यूटर का जनक माना जाता है जिन्होंने एनालिटिकल इंजन बनाया था।"),
            ("query: जल संरक्षण क्यों जरूरी है?", "passage: जल संरक्षण से भूजल स्तर में सुधार होता है और सूखे से राहत मिलती है।"),
        ]
        for q, p in curated * 50:
            anchors.append(q)
            positives.append(p)

    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def fine_tune():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    console.rule("[bold cyan]Fine-Tuning Lightweight Indic Embedder[/bold cyan]")
    logger.info(f"Base Model: {args.base_model}")
    logger.info(f"Target Checkpoint: {args.output_dir}")
    logger.info(f"Device: {device} | Epochs: {args.epochs} | Batch Size: {args.batch_size}")

    # 1. Load Base Model from HuggingFace
    logger.info(f"Downloading/Loading base model: {args.base_model}...")
    model = SentenceTransformer(args.base_model, device=device)

    # 2. Prepare Dataset from MSMARCO-XI
    loader = MSMARCOLoader(
        dataset_name="ai4bharat/MSMARCO-XI",
        language=args.language,
        split="train",
        cache_dir="data/raw",
    )
    logger.info("Preparing training examples...")
    train_dataset = prepare_training_dataset(loader, max_samples=args.max_samples)
    logger.info(f"Prepared {len(train_dataset)} training pairs.")

    # 3. Loss Function (MultipleNegativesRankingLoss is SOTA for dense retrieval)
    train_loss = MultipleNegativesRankingLoss(model=model)

    # 4. Training Arguments
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        fp16=(device == "cuda"),
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
    )

    # 5. Train and Save
    logger.info(f"Starting fine-tuning ({args.epochs} epochs)...")
    t0 = time.time()
    trainer.train()
    model.save_pretrained(args.output_dir)

    elapsed = time.time() - t0
    logger.info(f"Fine-tuning complete in {elapsed:.2f}s! Checkpoint saved to: {args.output_dir}")
    console.print(f"[bold green]✓ Fine-tuned model saved to {args.output_dir}[/bold green]")
    console.print("[yellow]To use this model in RAG, update configs/config.yaml -> retrieval.embedding_model[/yellow]")


if __name__ == "__main__":
    fine_tune()

