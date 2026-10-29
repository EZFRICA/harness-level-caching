"""Adapter factory: the only place that knows concrete backends."""

from __future__ import annotations

import os

from harness_cache.config import Config
from harness_cache.ports import LLM, Archive, Embedder


def make_llm(cfg: Config) -> LLM:
    if cfg.llm == "echo":
        from harness_cache.adapters.local import EchoLLM

        return EchoLLM(cfg.echo_ms_per_prompt_token, cfg.echo_base_ms, cfg.echo_ms_per_output_token,
                       cfg.echo_cached_extra_ms, cfg.llm_error_rate)
    if cfg.llm == "gemini":
        from harness_cache.adapters.gemini import GeminiLLM

        if cfg.gemini_vertex_project:
            return GeminiLLM(cfg.model, "", vertex_project=cfg.gemini_vertex_project, token_file=cfg.gemini_token_file)
        return GeminiLLM(cfg.model, _api_key())
    if cfg.llm == "openai_compat":
        from harness_cache.adapters.openai_compat import OpenAICompatLLM

        key = os.environ.get(cfg.oai_key_env, "") if cfg.oai_key_env else ""
        if cfg.oai_key_env and not key:
            raise RuntimeError(f"{cfg.oai_key_env} is not set")
        return OpenAICompatLLM(cfg.model, cfg.oai_base_url, key, cfg.oai_salt, key_file=cfg.oai_key_file)
    raise ValueError(f"unknown HC_LLM={cfg.llm!r}")


_EMBEDDERS: dict[tuple, Embedder] = {}


def make_embedder(cfg: Config) -> Embedder:
    """One embedder per process and space: a worker serving many agents must load a model once.
    Measured: one ONNX model per agent pushed a 1 GB worker out of memory with three agents."""
    key = (cfg.embedder, cfg.embed_model, cfg.embed_dim, cfg.onnx_model, cfg.onnx_dim, cfg.onnx_cache_dir,
           cfg.gemma_dir, cfg.gemma_dim, cfg.gemma_quantized)
    if key not in _EMBEDDERS:
        _EMBEDDERS[key] = _new_embedder(cfg)
    return _EMBEDDERS[key]


def _new_embedder(cfg: Config) -> Embedder:
    if cfg.embedder == "hash":
        from harness_cache.adapters.local import HashEmbedder

        return HashEmbedder()
    if cfg.embedder == "gemini":
        from harness_cache.adapters.gemini import GeminiEmbedder

        return GeminiEmbedder(cfg.embed_model, cfg.embed_dim, _api_key())
    if cfg.embedder == "onnx":
        from harness_cache.adapters.onnx import OnnxEmbedder

        return OnnxEmbedder(cfg.onnx_model, cfg.onnx_dim, cfg.onnx_cache_dir)
    if cfg.embedder == "gemma":
        from harness_cache.adapters.gemma import GemmaEmbedder

        return GemmaEmbedder(cfg.gemma_dir, cfg.gemma_dim, cfg.gemma_quantized)
    raise ValueError(f"unknown HC_EMBEDDER={cfg.embedder!r}")


def make_archive(cfg: Config) -> Archive:
    from harness_cache.adapters.local import FileArchive

    return FileArchive(cfg.archive_dir, cfg.store_name)


def _api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (pass it with --env-file .env)")
    return key
