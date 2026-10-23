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
}


@dataclass
class Config:
    mode: str = "cama"  # cama | topk | naive
    llm: str = "echo"  # echo | gemini
    embedder: str = "hash"  # hash | gemini
    model: str = "gemini-flash-lite-latest"
    embed_model: str = "gemini-embedding-001"
    embed_dim: int = 768
    temperature: float = 0.7

    capacity: int = 12  # max blocks in the active set
    max_memory_tokens: int = 1500  # max approx tokens of the rendered active set
    victim_window: int = 3  # LRU candidates considered for eviction (1 = pure LRU)
    window_turns: int = 2  # recent exchanges sent verbatim (cama/topk)

    tau_lexical: float = 0.6  # query-term containment for a tier-a hit
    min_lexical_overlap: int = 2
    tau_active: float = 0.35
    tau_l3: float = 0.30
    page_in_k: int = 2

    online: bool = True  # whether the L4 sync task may run
    echo_ms_per_prompt_token: float = 0.0  # echo LLM: simulated prefill cost

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
            model=_env("CAMA_MODEL", "gemini-flash-lite-latest"),
            embed_model=_env("CAMA_EMBED_MODEL", "gemini-embedding-001"),
            embed_dim=int(_env("CAMA_EMBED_DIM", "768")),
            temperature=float(_env("CAMA_TEMPERATURE", "0.7")),
            capacity=int(_env("CAMA_CAPACITY", "12")),
            max_memory_tokens=int(_env("CAMA_MAX_MEMORY_TOKENS", "1500")),
            victim_window=int(_env("CAMA_VICTIM_WINDOW", "3")),
            window_turns=int(_env("CAMA_WINDOW_TURNS", "2")),
            tau_lexical=float(_env("CAMA_TAU_LEXICAL", "0.6")),
            min_lexical_overlap=int(_env("CAMA_MIN_LEXICAL_OVERLAP", "2")),
            tau_active=float(_env("CAMA_TAU_ACTIVE", str(taus["active"]))),
            tau_l3=float(_env("CAMA_TAU_L3", str(taus["l3"]))),
            page_in_k=int(_env("CAMA_PAGE_IN_K", "2")),
            online=_env("CAMA_ONLINE", "1") == "1",
            echo_ms_per_prompt_token=float(_env("CAMA_ECHO_MS_PER_PROMPT_TOKEN", "0")),
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
