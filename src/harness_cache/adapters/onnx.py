"""Local ONNX embedder (fastembed), the same model as Akili and Urithi.

The model is read from a local cache and never downloaded at runtime: on a device
without connectivity a download attempt hangs instead of failing. Inference runs
in a worker thread so the event loop (and the cold-path scheduler) stays responsive.
"""

from __future__ import annotations

import asyncio
import os
from typing import Literal

import numpy as np

from harness_cache.metrics import resource_limits


def _default_threads() -> int:
    cpus = resource_limits().get("cpus_limit")
    # onnxruntime sizes its pool to the host's cores; inside a CPU-limited
    # container that oversubscribes the quota and inflates latency.
    return max(1, int(cpus)) if cpus else (os.cpu_count() or 1)


class OnnxEmbedder:
    def __init__(self, model: str, dim: int, cache_dir: str, threads: int | None = None):
        from fastembed import TextEmbedding

        if not os.path.isdir(cache_dir):
            raise RuntimeError(
                f"embedding model cache not found at {cache_dir}; mount it (see compose.yaml)"
            )
        self.threads = threads or _default_threads()
        self._model = TextEmbedding(
            model_name=model, cache_dir=cache_dir, local_files_only=True, threads=self.threads
        )
        self.name = f"onnx:{model}"
        self.dim = dim
        # One inference at a time per process. Measured: sixteen agents seeding at once on one CPU
        # ran sixteen inference threads, starved the event loop, and the worker stopped accepting
        # connections (0 of 192 requests answered).
        import weakref
        self._gates: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()  # one gate per event loop

    def _embed(self, texts: list[str]) -> np.ndarray:
        m = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        if m.shape != (len(texts), self.dim):
            raise ValueError(f"unexpected embedding shape {m.shape}, wanted ({len(texts)}, {self.dim})")
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("zero embedding vector")
        return m / norms

    async def embed(self, texts: list[str], kind: Literal["query", "document"]) -> np.ndarray:
        # Symmetric paraphrase model: no query/document prefix.
        loop = asyncio.get_running_loop()
        gate = self._gates.get(loop)
        if gate is None:
            gate = self._gates[loop] = asyncio.Semaphore(1)
        async with gate:
            return await asyncio.to_thread(self._embed, texts)
