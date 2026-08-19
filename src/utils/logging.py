"""Logging and diagnostic utilities for the Voice-Enabled RAG pipeline."""

import logging
import sys
import time
from contextlib import contextmanager
from typing import Generator, Optional
from rich.console import Console
from rich.logging import RichHandler

console = Console()

_logger_initialized = False


def setup_logger(name: str = "voice_rag", level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured logger with rich formatting."""
    global _logger_initialized
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    if not logger.handlers:
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
            markup=True,
        )
        handler.setLevel(numeric_level)
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False

    return logger


logger = setup_logger()


@contextmanager
def timer(stage_name: str, logger_instance: Optional[logging.Logger] = None) -> Generator[dict, None, None]:
    """Context manager for measuring high-precision stage execution time in milliseconds."""
    log = logger_instance or logger
    res = {"elapsed_ms": 0.0}
    start = time.perf_counter_ns()
    try:
        yield res
    finally:
        elapsed_ns = time.perf_counter_ns() - start
        elapsed_ms = elapsed_ns / 1_000_000.0
        res["elapsed_ms"] = elapsed_ms
        log.debug(f"[timer] {stage_name} took {elapsed_ms:.2f}ms")
