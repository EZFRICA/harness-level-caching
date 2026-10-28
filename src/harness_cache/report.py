"""Run analysis.

`metrics()` is the single computation of a run's figures: the text report, the snapshots
under docs/measured, the generated figure tables and the dashboard all read it, so a number
cannot differ between two places that show it.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

BOOTSTRAP_SEED = 20261023
BOOTSTRAP_N = 2000
# Mode names recorded by runs made before the study was renamed; frozen runs keep them.
LEGACY_MODES = {"cama": "extract"}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def bootstrap_ci(values: list[float], stat=np.median, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI. Seeded per call, so a figure is reproducible on its own."""
    if len(values) < 2:
        v = float(values[0]) if values else float("nan")
        return v, v
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arr = np.asarray(values, dtype=float)
    stats = np.array([stat(rng.choice(arr, size=arr.size, replace=True)) for _ in range(BOOTSTRAP_N)])
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def dist(values: list[float]) -> dict | None:
    if not values:
        return None
    lo, hi = bootstrap_ci(values)
    return {"n": len(values), "median": float(np.median(values)), "ci_lo": lo, "ci_hi": hi,
            "p95": float(np.percentile(values, 95)), "mean": float(np.mean(values))}


def pin(manifest: dict) -> dict:
    c, lim = manifest["config"], manifest.get("limits", {})
    return {
        "run_id": manifest["run_id"], "date": manifest["started_at"][:10], "code_sha": manifest.get("code_sha"),
        "mode": LEGACY_MODES.get(c["mode"], c["mode"]), "llm": c["llm"], "model": c.get("model") if c["llm"] != "echo" else "echo",
        "embedder": c["embedder"], "prefix_cache": c.get("prefix_cache", "none"),
        "cpus": lim.get("cpus_limit"), "memory_bytes": lim.get("memory_max_bytes"),
        "capacity": c["capacity"], "max_memory_tokens": c["max_memory_tokens"],
        "victim_window": c["victim_window"], "scenario": manifest.get("scenario"),
    }


def recall_of(probes: dict) -> dict:
    """Answers stay in the run directory; only the scores are frozen with a figure."""
    return {k: probes[k] for k in ("probes", "facts", "found", "answered")} | {
        "by_turn": [{"turn": i["turn"], "found": i["found"]} for i in probes["items"]]}


def metrics(manifest: dict, turns: list[dict], tasks: list[dict], skip_first: int = 0,
            pricing: dict | None = None) -> dict:
    all_turns = turns
    turns = turns[skip_first:]
    out: dict = {"pin": pin(manifest), "n_turns": len(all_turns), "skip_first": skip_first}
    if not turns:
        return out
    hot = [t["hot_ms"] for t in turns]
    llm = [t["timings_ms"]["llm"] for t in turns]
    mem = [h - m for h, m in zip(hot, llm)]
    out.update(hot_ms=dist(hot), llm_ms=dist(llm), memory_ms=dist(mem))

    routed = [t for t in turns if t.get("route")]
    if routed:
        n = len(routed)
        out["routing"] = {
            "tiers": dict(Counter(t["route"]["tier"] for t in routed)),
            "no_l3_share": sum(t["route"]["l3_rows_scanned"] == 0 for t in routed) / n,
            "no_embed_share": sum(t["route"]["embed_calls"] == 0 for t in routed) / n,
            "page_ins": sum(len(t["route"]["paged_in"]) for t in routed),
            "evictions": sum(len(t["route"]["evicted"]) for t in routed),
        }

    # Token totals cover every turn: they are what was paid, warm-up included.
    ptok = [t["llm"]["prompt_tokens"] for t in all_turns]
    ctok = [t["llm"]["cached_tokens"] for t in all_turns]
    otok = [t["llm"]["output_tokens"] for t in all_turns]
    est = [t["context"]["total_tokens_est"] for t in all_turns]
    out["tokens"] = {
        "prompt_sum": sum(ptok), "cached_sum": sum(ctok), "output_sum": sum(otok),
        "cached_share": sum(ctok) / sum(ptok) if sum(ptok) else 0.0,
        "request_est_first": est[0], "request_est_last": est[-1],
        "stable_prefix_est": dist([t["context"]["stable_prefix_tokens_est"] for t in turns]),
    }
    out["peak_rss_mb"] = max(t["peak_rss_mb"] for t in all_turns)

    pc = [t["prefix_cache"] for t in all_turns if t.get("prefix_cache")]
    if pc:
        ct = [t for t in tasks if t["task"].startswith("cache") and t.get("ok")]
        acts = Counter(t.get("action") for t in ct)
        out["prefix_cache"] = {
            "turns_hit": sum(p["hit"] for p in pc), "turns": len(pc),
            "creates": acts.get("create", 0), "reuses": acts.get("reuse", 0), "skips": acts.get("skip", 0),
            "create_ms": dist([t["create_ms"] for t in ct if t.get("action") == "create"]),
        }

    agentic = [t for t in all_turns if "h3" in t]
    if agentic:
        out["harness"] = {
            "model_calls": sum(t["llm"]["calls"] for t in agentic),
            "tool_calls": sum(len(t["tools"]) for t in agentic),
            "h3_hits": sum(t["h3"]["hits"] for t in agentic), "h3_misses": sum(t["h3"]["misses"] for t in agentic),
            "h3_bypass": sum(t["h3"]["bypass"] for t in agentic), "h3_errors": sum(t["h3"]["errors"] for t in agentic),
            "h3_saved_ms": sum(t["h3"]["saved_ms"] for t in agentic),
            "h2_hits": sum(t["h2"]["hits"] for t in agentic), "h2_misses": sum(t["h2"]["misses"] for t in agentic),
            "tools_ms": dist([t["timings_ms"].get("tools", 0.0) for t in turns]),
        }
    if pricing is not None:
        from harness_cache.cost import run_cost
        out["cost"] = {k: v for k, v in run_cost(manifest, all_turns, tasks, pricing).items() if k != "per_turn"}
        out["pin"]["pricing_hash"] = manifest.get("pricing_hash")
    out["pin"]["cache_mode"] = manifest.get("cache_mode", out["pin"]["prefix_cache"])
    out["pin"]["scenario_sha"] = manifest.get("scenario_sha")
    if manifest.get("stopped"):
        out["stopped"] = manifest["stopped"]
    if manifest.get("probes"):
        out["recall"] = recall_of(manifest["probes"])

    if tasks:
        ext = [t for t in tasks if t["task"].startswith("extract") and t.get("ok")]
        out["cold"] = {
            "tasks": dict(Counter(t["task"].split("#")[0] for t in tasks)),
            "failures": sum(1 for t in tasks if not t.get("ok")),
            "extract_ms": dist([t["run_ms"] for t in ext]),
            "extract_prompt_tokens": sum(t.get("prompt_tokens", 0) for t in ext),
            "extract_output_tokens": sum(t.get("output_tokens", 0) for t in ext),
            "blocks_created": sum(len(t.get("created", [])) for t in ext),
            "blocks_appended": sum(len(t.get("appended", [])) for t in ext),
        }
    return out


def run_metrics(run_dir: Path, skip_first: int = 0, pricing: dict | None = None) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return metrics(manifest, load_jsonl(run_dir / "turns.jsonl"), load_jsonl(run_dir / "tasks.jsonl"), skip_first,
                   pricing)


def _d(d: dict | None, unit: str = "") -> str:
    if not d:
        return "n/a"
    return (f"median {d['median']:.1f}{unit} [95% CI {d['ci_lo']:.1f} to {d['ci_hi']:.1f}]  "
            f"p95 {d['p95']:.1f}{unit}  n={d['n']}")


def summarize(run_dir: Path, skip_first: int = 0, pricing: dict | None = None) -> str:
    m = run_metrics(run_dir, skip_first, pricing)
    p = m["pin"]
    lines = [
        f"# run {p['run_id']}  mode={p['mode']}  model={p['model']}  embedder={p['embedder']}  "
        f"prefix_cache={p['prefix_cache']}",
        f"pin: date={p['date']} code={p['code_sha']} cpus={p['cpus']} mem={p['memory_bytes']} "
        f"capacity={p['capacity']} max_memory_tokens={p['max_memory_tokens']} victim_window={p['victim_window']}",
        f"turns: {m['n_turns']} (latency stats skip the first {skip_first})",
    ]
    if "hot_ms" not in m:
        return "\n".join(lines)
    lines += ["", "## hot path", f"total   {_d(m['hot_ms'], 'ms')}", f"model   {_d(m['llm_ms'], 'ms')}",
              f"memory  {_d(m['memory_ms'], 'ms')}  (total minus model)"]
    if "routing" in m:
        r = m["routing"]
        lines += ["", "## routing cascade", "tiers: " + "  ".join(f"{k}={v}" for k, v in r["tiers"].items()),
                  f"no L3 scan {100 * r['no_l3_share']:.0f}%  no embedding call {100 * r['no_embed_share']:.0f}%  "
                  f"page-ins {r['page_ins']}  evictions {r['evictions']}"]
    t = m["tokens"]
    lines += ["", "## tokens (all turns)",
              f"prompt {t['prompt_sum']}  cached {t['cached_sum']} ({100 * t['cached_share']:.1f}%)  output {t['output_sum']}",
              f"request estimate first {t['request_est_first']} last {t['request_est_last']}",
              f"stable prefix estimate {_d(t['stable_prefix_est'])}", f"peak RSS {m['peak_rss_mb']:.1f} MB"]
    if "prefix_cache" in m:
        c = m["prefix_cache"]
        lines += ["", "## prefix cache", f"turns served {c['turns_hit']}/{c['turns']}  creates {c['creates']}  "
                  f"reuses {c['reuses']}  skips {c['skips']}", f"creation {_d(c['create_ms'], 'ms')}"]
    if "harness" in m:
        h = m["harness"]
        lines += ["", "## harness caches", f"model calls {h['model_calls']}  tool calls {h['tool_calls']}",
                  f"H3 tools: hits {h['h3_hits']}  misses {h['h3_misses']}  bypass {h['h3_bypass']}  "
                  f"errors {h['h3_errors']}  saved {h['h3_saved_ms']:.0f} ms",
                  f"H2 embeddings: hits {h['h2_hits']}  misses {h['h2_misses']}"]
    if "cost" in m:
        c = m["cost"]
        if c["priced"]:
            lines += ["", f"## cost ({c['model']}, prices {c['status']} read {c['read_on']})",
                      f"total {c['total_usd']:.5f} USD  hot {c['hot_usd']:.5f}  cold {c['cold_usd']:.5f}",
                      "hot: " + "  ".join(f"{k} {v:.5f}" for k, v in c["hot"].items()),
                      "cold: " + "  ".join(f"{k} {v:.5f}" for k, v in c["cold"].items())]
            lines += [f"assumption: {a}" for a in c["assumptions"]]
        else:
            lines += ["", f"## cost: not priced ({c['reason']})"]
    if "cold" in m:
        c = m["cold"]
        lines += ["", "## cold path", "tasks: " + "  ".join(f"{k}={v}" for k, v in c["tasks"].items())
                  + f"  failures={c['failures']}", f"extraction {_d(c['extract_ms'], 'ms')}",
                  f"blocks created {c['blocks_created']}  appended {c['blocks_appended']}"]
    return "\n".join(lines)
