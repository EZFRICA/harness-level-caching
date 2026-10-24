"""Adapter factory: the only place that knows concrete backends."""

from __future__ import annotations

import os

from cama.config import Config
from cama.ports import LLM, Archive, Embedder


def make_llm(cfg: Config) -> LLM:
    if cfg.llm == "echo":
        from cama.adapters.local import EchoLLM

        return EchoLLM(cfg.echo_ms_per_prompt_token, cfg.echo_base_ms, cfg.echo_ms_per_output_token,
                       cfg.echo_cached_extra_ms, cfg.llm_error_rate)
    if cfg.llm == "gemini":
        from cama.adapters.gemini import GeminiLLM

        return GeminiLLM(cfg.model, _api_key())
    if cfg.llm == "openai_compat":
        from cama.adapters.openai_compat import OpenAICompatLLM

        key = os.environ.get(cfg.oai_key_env, "") if cfg.oai_key_env else ""
        if cfg.oai_key_env and not key:
            raise RuntimeError(f"{cfg.oai_key_env} is not set")
        return OpenAICompatLLM(cfg.model, cfg.oai_base_url, key, cfg.oai_salt)
    raise ValueError(f"unknown CAMA_LLM={cfg.llm!r}")


_EMBEDDERS: dict[tuple, Embedder] = {}


def make_embedder(cfg: Config) -> Embedder:
    """One embedder per process and space: a worker serving many agents must load a model once.
    Measured: one ONNX model per agent pushed a 1 GB worker out of memory with three agents."""
    key = (cfg.embedder, cfg.embed_model, cfg.embed_dim, cfg.onnx_model, cfg.onnx_dim, cfg.onnx_cache_dir)
    if key not in _EMBEDDERS:
        _EMBEDDERS[key] = _new_embedder(cfg)
    return _EMBEDDERS[key]


def _new_embedder(cfg: Config) -> Embedder:
    if cfg.embedder == "hash":
        from cama.adapters.local import HashEmbedder

        return HashEmbedder()
    if cfg.embedder == "gemini":
        from cama.adapters.gemini import GeminiEmbedder

        return GeminiEmbedder(cfg.embed_model, cfg.embed_dim, _api_key())
    if cfg.embedder == "onnx":
        from cama.adapters.onnx import OnnxEmbedder

        return OnnxEmbedder(cfg.onnx_model, cfg.onnx_dim, cfg.onnx_cache_dir)
    raise ValueError(f"unknown CAMA_EMBEDDER={cfg.embedder!r}")


def make_archive(cfg: Config) -> Archive:
    from cama.adapters.local import FileArchive

    return FileArchive(cfg.archive_dir, cfg.store_name)


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (pass it with --env-file .env)")
    return key
