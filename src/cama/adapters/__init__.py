"""Adapter factory: the only place that knows concrete backends."""

from __future__ import annotations

import os

from cama.config import Config
from cama.ports import LLM, Archive, Embedder


def make_llm(cfg: Config) -> LLM:
    if cfg.llm == "echo":
        from cama.adapters.local import EchoLLM

        return EchoLLM(cfg.echo_ms_per_prompt_token)
    if cfg.llm == "gemini":
        from cama.adapters.gemini import GeminiLLM

        return GeminiLLM(cfg.model, _api_key())
    raise ValueError(f"unknown CAMA_LLM={cfg.llm!r}")


def make_embedder(cfg: Config) -> Embedder:
    if cfg.embedder == "hash":
        from cama.adapters.local import HashEmbedder

        return HashEmbedder()
    if cfg.embedder == "gemini":
        from cama.adapters.gemini import GeminiEmbedder

        return GeminiEmbedder(cfg.embed_model, cfg.embed_dim, _api_key())
    raise ValueError(f"unknown CAMA_EMBEDDER={cfg.embedder!r}")


def make_archive(cfg: Config) -> Archive:
    from cama.adapters.local import FileArchive

    return FileArchive(cfg.archive_dir, cfg.store_name)


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (pass it with --env-file .env)")
    return key
