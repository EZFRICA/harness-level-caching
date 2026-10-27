"""Configuration from environment variables (CAMA_*). Every value lands in the run manifest."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# Routing thresholds depend on the embedding space; never reuse one embedder's
# thresholds for another (different models live in unrelated geometries).
_DEFAULT_TAUS = {
    "hash": {"active": 0.35, "l3": 0.30},
    "gemini": {"active": 0.70, "l3": 0.65},
    "onnx": {"active": 0.40, "l3": 0.50},
}


def _default_pricing() -> Path:
    return Path(__file__).resolve().parents[2] / "pricing.toml"


@dataclass
class Config:
    mode: str = "cama"  # cama | topk | naive
    llm: str = "echo"  # echo | gemini | openai_compat
    embedder: str = "hash"  # hash | gemini | onnx
    model: str = "gemini-3.5-flash-lite"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    onnx_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    onnx_dim: int = 384
    onnx_cache_dir: str = "/models"
    temperature: float = 0.7
    oai_base_url: str = ""  # openai_compat: server base URL (Nebius, vLLM)
    oai_key_env: str = ""  # openai_compat: name of the variable holding the key (never the key)
    oai_salt: str = ""  # openai_compat: run salt opening the system prompt (shared caches)
    oai_key_file: str = ""  # openai_compat: file holding a short-lived token, re-read per request
    gemini_vertex_project: str = ""  # gemini through Vertex AI (location global) instead of the Gemini API
    gemini_token_file: str = ""  # gemini on Vertex: file holding a short-lived access token

    capacity: int = 12  # max blocks in the active set
    max_memory_tokens: int = 1500  # max approx tokens of the rendered active set
    victim_window: int = 3  # LRU candidates considered for eviction (1 = pure LRU)
    window_turns: int = 2  # recent exchanges sent verbatim (cama/topk)
    window_epoch: int = 1  # 1: slide every turn | E: grow to window_turns+E-1, then drop E at once (append-only between)
    extract_every: int = 1  # extract after every N exchanges, in one call (cama/topk)
    extract_replace: bool = False  # hypothesis H5: the extractor may replace a corrected fact's line
    memory_refresh: str = "turn"  # turn | epoch (hypothesis H7: rendered memory frozen until the window drops)
    route_policy: str = "cascade"  # cascade | urithi (whole-store search, per-kind thresholds, words beside meaning)
    compact_at: int = 6000  # compact mode: fold the history once its unfolded part passes this (approx tokens)
    compact_keep: int = 4  # compact mode: exchanges kept verbatim after a fold
    page_in_tail: bool = False  # hypothesis H8 (with epoch refresh): a paged-in block is sent after the window
    render_order: str = "stable"  # stable (decision 0002) | recency: ablation, blocks in order of last access
    extract_policy: str = "every"  # every: batch by extract_every | statements: a stated fact is extracted at once

    tau_lexical: float = 0.6  # query-term containment for a tier-a hit
    min_lexical_overlap: int = 2
    tau_active: float = 0.35
    tau_l3: float = 0.30
    page_in_k: int = 2

    tools: bool = True  # offer the simulated tools to the model (agentic turns)
    tool_cache: bool = True  # H3
    embed_cache: bool = True  # H2
    max_steps: int = 4  # model calls per turn at most
    tool_error_rate: float = 0.0  # injected tool failures
    tool_latency_scale: float = 1.0
    web_search: bool = False  # offer the real web_search tool (Tavily); changes the tool prefix
    web_mode: str = "tape"  # tape: replay recorded queries, record new ones | replay: never call live
    web_tape: Path = field(default_factory=lambda: Path("/runs/web-tapes/default.jsonl"))
    web_max_calls: int = 300  # live Tavily calls per process at most
    web_semantic_tau: float = 0.0  # hypothesis: serve a web query from a close one (cosine >= tau); 0 = off
    prefix_cache: str = "auto"  # auto | explicit | implicit | none (auto: from the model's capability)
    max_usd: float = 2.0  # budget cap per run for paid calls
    allow_unpriced: bool = False  # refuse a paid model with no price, since the cap could not hold
    pricing_path: Path = field(default_factory=lambda: _default_pricing())
    cache_ttl_s: int = 1800
    cache_admission: str = "stable"  # stable: create a prefix cache only once it has been sent unchanged | always

    online: bool = True  # whether the L4 sync task may run
    snapshot_every: int = 50  # journal rows between two snapshots in the shared tier
    sync_interval_s: float = 1.0  # replication tick (bounds the RPO); 0 = sync on the cold path when idle
    extract_batch_max: int = 8  # exchanges per extraction call when the cold path falls behind
    echo_ms_per_prompt_token: float = 0.0  # echo LLM latency model (see adapters/local.py)
    echo_base_ms: float = 0.0
    echo_ms_per_output_token: float = 0.0
    echo_cached_extra_ms: float = 0.0
    llm_error_rate: float = 0.0  # injected model failures (echo)
    llm_attempts: int = 2  # model call attempts within a step

    store_name: str = "cama"  # store file = data_dir/<store_name>.sqlite (one per experiment)
    data_dir: Path = field(default_factory=lambda: Path("/data"))
    runs_dir: Path = field(default_factory=lambda: Path("/runs"))
    archive_dir: Path = field(default_factory=lambda: Path("/archive"))

    @classmethod
    def from_env(cls) -> Config:
        embedder = _env("CAMA_EMBEDDER", "hash")
        taus = _DEFAULT_TAUS.get(embedder, _DEFAULT_TAUS["hash"])
        return cls(
            mode=_env("CAMA_MODE", "cama"),
            llm=_env("CAMA_LLM", "echo"),
            embedder=embedder,
            model=_env("CAMA_MODEL", "gemini-3.5-flash-lite"),
            embed_model=_env("CAMA_EMBED_MODEL", "gemini-embedding-001"),
            embed_dim=int(_env("CAMA_EMBED_DIM", "768")),
            onnx_model=_env("CAMA_ONNX_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"),
            onnx_dim=int(_env("CAMA_ONNX_DIM", "384")),
            onnx_cache_dir=_env("CAMA_ONNX_CACHE_DIR", "/models"),
            temperature=float(_env("CAMA_TEMPERATURE", "0.7")),
            oai_base_url=_env("CAMA_OAI_BASE_URL", ""),
            oai_key_env=_env("CAMA_OAI_KEY_ENV", ""),
            oai_salt=_env("CAMA_OAI_SALT", ""),
            oai_key_file=_env("CAMA_OAI_KEY_FILE", ""),
            gemini_vertex_project=_env("CAMA_GEMINI_VERTEX_PROJECT", ""),
            gemini_token_file=_env("CAMA_GEMINI_TOKEN_FILE", ""),
            capacity=int(_env("CAMA_CAPACITY", "12")),
            max_memory_tokens=int(_env("CAMA_MAX_MEMORY_TOKENS", "1500")),
            victim_window=int(_env("CAMA_VICTIM_WINDOW", "3")),
            window_turns=int(_env("CAMA_WINDOW_TURNS", "2")),
            window_epoch=int(_env("CAMA_WINDOW_EPOCH", "1")),
            extract_every=int(_env("CAMA_EXTRACT_EVERY", "1")),
            extract_replace=_env("CAMA_EXTRACT_REPLACE", "off") == "on",
            memory_refresh=_env("CAMA_MEMORY_REFRESH", "turn"),
            route_policy=_env("CAMA_ROUTE_POLICY", "cascade"),
            compact_at=int(_env("CAMA_COMPACT_AT", "6000")),
            compact_keep=int(_env("CAMA_COMPACT_KEEP", "4")),
            page_in_tail=_env("CAMA_PAGE_IN_TAIL", "off") == "on",
            render_order=_env("CAMA_RENDER_ORDER", "stable"),
            extract_policy=_env("CAMA_EXTRACT_POLICY", "every"),
            tau_lexical=float(_env("CAMA_TAU_LEXICAL", "0.6")),
            min_lexical_overlap=int(_env("CAMA_MIN_LEXICAL_OVERLAP", "2")),
            tau_active=float(_env("CAMA_TAU_ACTIVE", str(taus["active"]))),
            tau_l3=float(_env("CAMA_TAU_L3", str(taus["l3"]))),
            page_in_k=int(_env("CAMA_PAGE_IN_K", "2")),
            tools=_env("CAMA_TOOLS", "on") == "on",
            tool_cache=_env("CAMA_TOOL_CACHE", "on") == "on",
            embed_cache=_env("CAMA_EMBED_CACHE", "on") == "on",
            max_steps=int(_env("CAMA_MAX_STEPS", "4")),
            tool_error_rate=float(_env("CAMA_FAULT_TOOL_ERROR_RATE", "0")),
            tool_latency_scale=float(_env("CAMA_TOOL_LATENCY_SCALE", "1")),
            web_search=_env("CAMA_WEB_SEARCH", "off") == "on",
            web_mode=_env("CAMA_WEB_MODE", "tape"),
            web_tape=Path(_env("CAMA_WEB_TAPE", "/runs/web-tapes/default.jsonl")),
            web_max_calls=int(_env("CAMA_WEB_MAX_CALLS", "300")),
            web_semantic_tau=float(_env("CAMA_WEB_SEMANTIC_TAU", "0")),
            prefix_cache=_env("CAMA_PREFIX_CACHE", "auto"),
            max_usd=float(_env("CAMA_MAX_USD", "2")),
            allow_unpriced=_env("CAMA_ALLOW_UNPRICED", "0") == "1",
            pricing_path=Path(_env("CAMA_PRICING", str(_default_pricing()))),
            cache_ttl_s=int(_env("CAMA_CACHE_TTL_S", "1800")),
            cache_admission=_env("CAMA_CACHE_ADMISSION", "stable"),
            online=_env("CAMA_ONLINE", "1") == "1",
            snapshot_every=int(_env("CAMA_SNAPSHOT_EVERY", "50")),
            sync_interval_s=float(_env("CAMA_SYNC_INTERVAL_S", "1.0")),
            extract_batch_max=int(_env("CAMA_EXTRACT_BATCH_MAX", "8")),
            echo_ms_per_prompt_token=float(_env("CAMA_ECHO_MS_PER_PROMPT_TOKEN", "0")),
            echo_base_ms=float(_env("CAMA_ECHO_BASE_MS", "0")),
            echo_ms_per_output_token=float(_env("CAMA_ECHO_MS_PER_OUTPUT_TOKEN", "0")),
            echo_cached_extra_ms=float(_env("CAMA_ECHO_CACHED_EXTRA_MS", "0")),
            llm_error_rate=float(_env("CAMA_FAULT_LLM_ERROR_RATE", "0")),
            llm_attempts=int(_env("CAMA_LLM_ATTEMPTS", "2")),
            store_name=_env("CAMA_STORE", "cama"),
            data_dir=Path(_env("CAMA_DATA_DIR", "/data")),
            runs_dir=Path(_env("CAMA_RUNS_DIR", "/runs")),
            archive_dir=Path(_env("CAMA_ARCHIVE_DIR", "/archive")),
        )

    def as_dict(self) -> dict:
        d = asdict(self)
        return {k: str(v) if isinstance(v, Path) else v for k, v in d.items()}

    @property
    def store_path(self) -> Path:
        return self.data_dir / f"{self.store_name}.sqlite"
