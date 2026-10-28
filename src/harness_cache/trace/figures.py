"""Figures are generated, never typed.

A run is frozen into `docs/measured/runs/<run_id>.json` (its pin, its metrics, a compact
per-turn series). `docs/measured/figures.toml` declares each figure: the question, the runs,
the columns, the limits, the command that reproduces it. `harness-cache figures` writes the Markdown
pages from those two sources only; a test regenerates them and fails on any difference, so a
number in the docs cannot drift from the run that produced it.
"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

from harness_cache.cost import load_pricing
from harness_cache.report import LEGACY_MODES, load_jsonl, metrics

MAX_LINES = 100


def snapshot(run_dir: Path, dest_dir: Path, skip_first: int = 1, pricing_path: Path | None = None,
             price_as: str | None = None) -> Path:
    if (run_dir / "load_summary.json").exists():
        return _snapshot_load(run_dir, dest_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    turns = load_jsonl(run_dir / "turns.jsonl")
    tasks = load_jsonl(run_dir / "tasks.jsonl")
    series = [
        {"turn": t["turn"], "tier": (t.get("route") or {}).get("tier"), "hot_ms": round(t["hot_ms"], 1),
         "llm_ms": round(t["timings_ms"]["llm"], 1), "prompt": t["llm"]["prompt_tokens"],
         "cached": t["llm"]["cached_tokens"], "output": t["llm"]["output_tokens"],
         "request_est": t["context"]["total_tokens_est"], "stable_est": t["context"]["stable_prefix_tokens_est"],
         "cache_hit": (t.get("prefix_cache") or {}).get("hit"),
         "calls": t["llm"].get("calls", 1), "h3_hits": (t.get("h3") or {}).get("hits"),
         "spent_usd": t.get("budget_spent_usd")}
        for t in turns
    ]
    pricing = load_pricing(pricing_path) if pricing_path and pricing_path.exists() else None
    data = {"metrics": metrics(manifest, turns, tasks, skip_first, pricing), "series": series}
    if price_as and pricing:
        from harness_cache.cost import run_cost
        as_if = {**manifest, "config": {**manifest["config"], "llm": "gemini", "model": price_as}}
        proj = run_cost(as_if, turns, tasks, pricing)
        data["metrics"]["cost_projected"] = {k: v for k, v in proj.items() if k != "per_turn"} | {"as": price_as}
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{manifest['run_id']}.json"
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return out


def _snapshot_load(run_dir: Path, dest_dir: Path) -> Path:
    """A load or fault run: its summary, its settings, and any fault facts recorded beside it."""
    conf = json.loads((run_dir / "load_config.json").read_text()) if (run_dir / "load_config.json").exists() else {}
    facts = json.loads((run_dir / "facts.json").read_text()) if (run_dir / "facts.json").exists() else {}
    data = {"kind": "load", "metrics": {"run_id": run_dir.name, "summary": json.loads((run_dir / "load_summary.json").read_text()),
            "config": {k: conf.get(k) for k in ("agents", "turns", "concurrency", "scenario", "workers", "started_at")},
            "facts": facts}}
    dest_dir.mkdir(parents=True, exist_ok=True)
    out = dest_dir / f"{run_dir.name}.json"
    out.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    return out


def _ms(d: dict | None) -> str:
    return "n/a" if not d else f"{d['median']:.0f} [{d['ci_lo']:.0f}, {d['ci_hi']:.0f}]"


def _lw(m: dict) -> str:
    return str(len(str(m["config"].get("workers") or "").split(",")))


LOAD_COLUMNS = {
    "run": ("run", lambda m: f"`{m['run_id']}`"),
    "workers": ("workers", _lw),
    "agents": ("agents", lambda m: str(m["config"].get("agents"))),
    "ok": ("answered", lambda m: f"{m['summary']['ok']}/{m['summary']['requests']}"),
    "failovers": ("failovers", lambda m: str(m["summary"]["failovers"])),
    "takeovers": ("takeovers", lambda m: str(m["summary"].get("takeovers", "n/a"))),
    "rps": ("requests/s", lambda m: f"{m['summary']['throughput_rps']:.2f}"),
    "p50": ("p50 ms", lambda m: f"{m['summary']['p50_ms']:.0f}" if m["summary"]["p50_ms"] else "n/a"),
    "p95": ("p95 ms", lambda m: f"{m['summary']['p95_ms']:.0f}" if m["summary"]["p95_ms"] else "n/a"),
    "p99": ("p99 ms", lambda m: f"{m['summary']['p99_ms']:.0f}" if m["summary"]["p99_ms"] else "n/a"),
    "fault": ("fault", lambda m: m["facts"].get("fault", "none")),
    "rto": ("RTO s", lambda m: str(m["facts"].get("rto_s", "n/a"))),
    "rpo": ("RPO rows", lambda m: str(m["facts"].get("rpo_rows", "n/a"))),
}

COLUMNS = {
    "run": ("run", lambda m: f"`{m['pin']['run_id']}`"),
    "date": ("date", lambda m: m["pin"]["date"]),
    "code": ("code", lambda m: f"`{m['pin']['code_sha']}`"),
    "mode": ("mode", lambda m: LEGACY_MODES.get(m["pin"]["mode"], m["pin"]["mode"])),
    "model": ("model", lambda m: m["pin"]["model"]),
    "embedder": ("embedder", lambda m: m["pin"]["embedder"]),
    "prefix_cache": ("prefix cache", lambda m: m["pin"]["prefix_cache"]),
    "turns": ("turns", lambda m: str(m["n_turns"])),
    "hot": ("turn ms, median [95% CI]", lambda m: _ms(m.get("hot_ms"))),
    "memory": ("memory ms, median [95% CI]", lambda m: _ms(m.get("memory_ms"))),
    "prompt": ("prompt tokens", lambda m: str(m["tokens"]["prompt_sum"])),
    "cached": ("cached tokens", lambda m: str(m["tokens"]["cached_sum"])),
    "cached_share": ("cached share", lambda m: f"{100 * m['tokens']['cached_share']:.1f}%"),
    "no_embed": ("turns without embedding call",
                 lambda m: f"{100 * m['routing']['no_embed_share']:.0f}%" if "routing" in m else "n/a"),
    "cache_turns": ("turns served from prefix cache",
                    lambda m: f"{m['prefix_cache']['turns_hit']}/{m['prefix_cache']['turns']}"
                    if "prefix_cache" in m else "n/a"),
    "rss": ("peak RSS MB", lambda m: f"{m['peak_rss_mb']:.0f}"),
    "cost": ("cost USD", lambda m: f"{m['cost']['total_usd']:.4f}" if m.get("cost", {}).get("priced") else "n/a"),
    "cost_cold": ("of which cold path", lambda m: f"{m['cost']['cold_usd']:.4f}" if m.get("cost", {}).get("priced") else "n/a"),
    "model_calls": ("model calls", lambda m: str(m["harness"]["model_calls"]) if "harness" in m else "n/a"),
    "h3": ("tool cache hits / calls", lambda m: f"{m['harness']['h3_hits']}/{m['harness']['tool_calls']}"
           if "harness" in m else "n/a"),
    "cost_projected": ("cost if on the priced model, USD",
                       lambda m: f"{m['cost_projected']['total_usd']:.4f}" if m.get("cost_projected", {}).get("priced") else "n/a"),
    "request_last": ("last request tokens", lambda m: str(m["tokens"]["request_est_last"])),
    "page_ins": ("page faults", lambda m: str(m["routing"]["page_ins"]) if "routing" in m else "n/a"),
    "cache_mode": ("prefix cache", lambda m: m["pin"].get("cache_mode", m["pin"]["prefix_cache"])),
    "recall": ("recall facts found", lambda m: f"{m['recall']['found']}/{m['recall']['facts']}" if "recall" in m else "n/a"),
}


def render_figure(fig: dict, snapshots: dict[str, dict]) -> str:
    missing = [r for r in fig["runs"] if r not in snapshots]
    if missing:
        raise KeyError(f"figure {fig['id']}: no snapshot for {missing}")
    table = LOAD_COLUMNS if fig.get("kind") == "load" else COLUMNS
    cols = [table[c] for c in fig["columns"]]
    lines = [f"# {fig['title']}", "",
             "Generated by `harness-cache figures` from `docs/measured/runs/`. Do not edit.", "",
             f"**Question.** {fig['question']}", "",
             "| " + " | ".join(h for h, _ in cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for run in fig["runs"]:
        m = snapshots[run]["metrics"]
        lines.append("| " + " | ".join(f(m) for _, f in cols) + " |")
    note = ("Latencies are client side, per request, every request included." if fig.get("kind") == "load"
            else "Latency statistics skip the first turn of each run.")
    lines += ["", f"{note} Reproduce: `{fig['reproduce']}`", ""]
    if fig.get("reading"):
        lines += ["**Reading.**", ""] + [f"- {x}" for x in fig["reading"]] + [""]
    lines += ["**Limits.**", ""] + [f"- {x}" for x in fig["limits"]]
    text = "\n".join(lines) + "\n"
    if text.count("\n") > MAX_LINES:
        raise ValueError(f"figure {fig['id']} renders past {MAX_LINES} lines; split it")
    return text


def load_snapshots(runs_dir: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text()) for p in sorted(runs_dir.glob("*.json"))}


def generate(measured_dir: Path) -> dict[str, str]:
    """Return {file name: content} for every figure page; does not write."""
    spec = tomllib.loads((measured_dir / "figures.toml").read_text())
    snaps = load_snapshots(measured_dir / "runs")
    pages: dict[str, list[str]] = {}
    for fig in spec["figure"]:
        pages.setdefault(fig["file"], []).append(render_figure(fig, snaps))
    return {name: "\n".join(parts) for name, parts in pages.items()}


def write(measured_dir: Path) -> list[Path]:
    out = []
    for name, text in generate(measured_dir).items():
        p = measured_dir / name
        p.write_text(text)
        out.append(p)
    return out
