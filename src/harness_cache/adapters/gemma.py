"""Local EmbeddingGemma 2 embedder (onnxruntime and tokenizers, both installed with fastembed).

Unlike the paraphrase model, EmbeddingGemma is asymmetric: a query and a stored text get
different prefixes, as its model card prescribes. Its vectors can be cut to 512, 256 or 128
dimensions (Matryoshka) and renormalised. The model is read from a local directory, never
downloaded. Inference runs in a worker thread, one at a time per event loop, like OnnxEmbedder.
"""

from __future__ import annotations

import asyncio
import os
import weakref
from typing import Literal

import numpy as np

from harness_cache.adapters.onnx import _default_threads

PREFIXES = {"query": "task: search result | query: ", "document": "title: none | text: "}
DIMS = (768, 512, 256, 128)
MAX_TOKENS = 2048
_BATCH = 32
# The ONNX export also takes media inputs; text only means empty arrays of 512 features.
_MEDIA = ("image_features", "video_features", "audio_features")


class GemmaEmbedder:
    def __init__(self, model_dir: str, dim: int = 768, quantized: bool = False, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if dim not in DIMS:
            raise ValueError(f"EmbeddingGemma dimension must be one of {DIMS}, got {dim}")
        path = os.path.join(model_dir, "onnx", "model_quantized.onnx" if quantized else "model.onnx")
        if not os.path.isfile(path):
            raise RuntimeError(f"EmbeddingGemma model not found at {path}; mount it (see compose.yaml)")
        self.threads = threads or _default_threads()
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        self._session = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self._tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        self._tok.enable_padding()
        self._tok.enable_truncation(MAX_TOKENS)
        self.name = f"gemma:embeddinggemma-2{'-int8' if quantized else ''}:{dim}"
        self.dim = dim
        self._gates: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # one gate per event loop

    def _run(self, texts: list[str]) -> np.ndarray:
        enc = self._tok.encode_batch(texts)
        feeds = {"input_ids": np.array([e.ids for e in enc], np.int64),
                 "attention_mask": np.array([e.attention_mask for e in enc], np.int64)}
        for name in _MEDIA:
            feeds[name] = np.zeros((0, 512), np.float32)
        return self._session.run(["sentence_embedding"], feeds)[0]

    def _embed(self, texts: list[str], kind: str) -> np.ndarray:
        prefixed = [PREFIXES[kind] + t for t in texts]
        m = np.concatenate([self._run(prefixed[i:i + _BATCH]) for i in range(0, len(prefixed), _BATCH)])
        return truncate(m.astype(np.float32), self.dim)

    async def embed(self, texts: list[str], kind: Literal["query", "document"]) -> np.ndarray:
        loop = asyncio.get_running_loop()
        gate = self._gates.get(loop)
        if gate is None:
            gate = self._gates[loop] = asyncio.Semaphore(1)
        async with gate:
            return await asyncio.to_thread(self._embed, texts, kind)


def truncate(m: np.ndarray, dim: int) -> np.ndarray:
    """Keep the first `dim` coordinates (Matryoshka) and bring each row back to unit norm."""
    m = m[:, :dim]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero embedding vector")
    return m / norms
