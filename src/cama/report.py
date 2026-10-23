"""Run analysis: summary statistics with bootstrap confidence intervals."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

_RNG = np.random.default_rng(20261023)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap_ci(values: list[float], stat=np.median, n: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    if len(values) < 2:
        v = values[0] if values else float("nan")
        return v, v
    arr = np.asarray(values, dtype=float)
    stats = np.array([stat(_RNG.choice(arr, size=arr.size, replace=True)) for _ in range(n)])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def _fmt(values: list[float], unit: str = "") -> str:
    if not values:
        return "n/a"
    lo, hi = bootstrap_ci(values)
    return (f"median {np.median(values):.1f}{unit} [95% CI {lo:.1f}–{hi:.1f}]  "
            f"p95 {np.percentile(values, 95):.1f}{unit}  mean {np.mean(values):.1f}{unit}")


def summarize(run_dir: Path, skip_first: int = 0) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    turns = _load(run_dir / "turns.jsonl")[skip_first:]
    tasks = _load(run_dir / "tasks.jsonl")
    cfg = manifest["config"]
    lim = manifest["limits"]
    lines = [
        f"# run {manifest['run_id']}  mode={cfg['mode']}  llm={cfg['llm']}  embedder={cfg['embedder']}",
        f"limits: cpus={lim.get('cpus_limit')}  mem={lim.get('memory_max_bytes')}  "
        f"capacity={cfg['capacity']}  max_memory_tokens={cfg['max_memory_tokens']}  "
        f"victim_window={cfg['victim_window']}",
        f"turns: {len(turns)} (skipped first {skip_first})",
    ]
    if not turns:
        return "\n".join(lines)

    hot = [t["hot_ms"] for t in turns]
    llm = [t["timings_ms"]["llm"] for t in turns]
    non_llm = [h - l for h, l in zip(hot, llm)]
    lines += [
        "",
        "## hot path (what the user waits for)",
        f"total      {_fmt(hot, 'ms')}",
        f"llm        {_fmt(llm, 'ms')}",
        f"memory/os  {_fmt(non_llm, 'ms')}   (hot - llm)",
    ]
    share = [n / h for n, h in zip(non_llm, hot) if h > 0]
    lines.append(f"memory/os share of turn: median {100 * np.median(share):.2f}%")

    routed = [t for t in turns if t.get("route")]
    if routed:
        tiers = Counter(t["route"]["tier"] for t in routed)
        n = len(routed)
        avoided = sum(1 for t in routed if t["route"]["l3_rows_scanned"] == 0)
        no_embed = sum(1 for t in routed if t["route"]["embed_calls"] == 0)
        lines += [
            "",
            "## routing cascade",
            "tiers: " + "  ".join(f"{k}={v} ({100 * v / n:.0f}%)" for k, v in tiers.most_common()),
            f"turns without L3 scan: {avoided}/{n} ({100 * avoided / n:.0f}%)",
            f"turns without embedding call: {no_embed}/{n} ({100 * no_embed / n:.0f}%)",
            f"page-ins: {sum(len(t['route']['paged_in']) for t in routed)}  "
            f"evictions (hot): {sum(len(t['route']['evicted']) for t in routed)}",
        ]

    tot = [t["context"]["total_tokens_est"] for t in turns]
    stable = [t["context"]["stable_prefix_tokens_est"] for t in turns]
    uncached = [a - b for a, b in zip(tot, stable)]
    ptok = [t["llm"]["prompt_tokens"] for t in turns]
    ctok = [t["llm"]["cached_tokens"] for t in turns]
    lines += [
        "",
        "## context and cache",
        f"request tokens (est)        {_fmt(tot)}   first={tot[0]} last={tot[-1]}",
        f"stable prefix (est)         {_fmt(stable)}",
        f"non-reusable tokens (est)   {_fmt(uncached)}",
        f"provider prompt tokens      sum={sum(ptok)}  last={ptok[-1]}",
        f"provider cached tokens      sum={sum(ctok)}  ({100 * sum(ctok) / max(1, sum(ptok)):.1f}% of prompt)",
        f"active set size             last={turns[-1]['context']['active_size']}",
        f"peak RSS                    {max(t['peak_rss_mb'] for t in turns):.1f} MB",
    ]

    if tasks:
        by = Counter(t["task"].split("#")[0] for t in tasks)
        fails = [t for t in tasks if not t.get("ok")]
        ext = [t for t in tasks if t["task"].startswith("extract") and t.get("ok")]
        lines += [
            "",
            "## cold path (off the user's critical path)",
            "tasks: " + "  ".join(f"{k}={v}" for k, v in by.items()) + f"  failures={len(fails)}",
            f"extraction run time  {_fmt([t['run_ms'] for t in ext], 'ms')}",
            f"blocks created={sum(len(t.get('created', [])) for t in ext)}  "
            f"appended={sum(len(t.get('appended', [])) for t in ext)}  "
            f"evicted={sum(len(t.get('evicted', [])) for t in ext)}  "
            f"parse_errors={sum(1 for t in ext if t.get('parse_error'))}",
        ]
    return "\n".join(lines)
