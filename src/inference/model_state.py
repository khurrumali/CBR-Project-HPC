"""
model_state.py — Singleton model holder for MedGemma.

Loaded once into the MCP server process memory; all subsequent cbr_adapt
tool calls reuse the same model instance without any reload overhead.

Thread safety: a lock guards load/unload; generate() is assumed to be
called sequentially by the MCP server (FastMCP is single-threaded by default).
"""

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_tokenizer: Optional[Any] = None
_model: Optional[Any] = None
_model_path: Optional[str] = None
_device: str = "cpu"


def is_loaded() -> bool:
    return _model is not None


def current_model_path() -> Optional[str]:
    return _model_path


def load(model_path: str, *, use_gpu: bool = True) -> str:
    """Load model into memory. Idempotent if same path is already loaded."""
    global _tokenizer, _model, _model_path, _device

    with _lock:
        if _model is not None and _model_path == model_path:
            return f"Already loaded: {model_path}"

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise RuntimeError(f"torch/transformers not available: {e}") from e

        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        logger.info("Loading model from %s on %s ...", model_path, device)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        )

        _tokenizer = tokenizer
        _model = model
        _model_path = model_path
        _device = device

        logger.info("Model ready on %s.", device)
        return f"Loaded {model_path} on {device}"


def generate(prompt: str, *, max_new_tokens: int = 512, temperature: float = 0.2) -> str:
    """Run inference with the resident model. Raises RuntimeError if not loaded."""
    if _model is None or _tokenizer is None:
        raise RuntimeError("Model not loaded. Call load_medgemma() first.")

    import torch

    inputs = _tokenizer(prompt, return_tensors="pt").to(_device)
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )

    return _tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )


def unload() -> str:
    """Release model from memory and free GPU cache."""
    global _tokenizer, _model, _model_path

    with _lock:
        if _model is None:
            return "No model currently loaded."

        import torch

        del _model, _tokenizer
        _model = None
        _tokenizer = None
        _model_path = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded.")
        return "Model unloaded and GPU cache cleared."
