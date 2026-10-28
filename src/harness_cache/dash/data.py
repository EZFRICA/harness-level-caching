"""Read side of the dashboard. Everything here reads; nothing writes, nothing calls a model.

Sources: live run directories (`runs/`), frozen runs (`docs/measured/runs/`), the stores of
live runs (opened query-only), the ledger, and in cluster mode the coordinator. Every figure
is computed by `harness_cache.report.metrics` and `harness_cache.cost.run_cost`, the same code as the reports
and the generated pages, so the dashboard cannot show a number the docs would not.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from harness_cache.config import Config
from harness_cache.cost import load_pricing, run_cost
from harness_cache.memory.active_set import block_tokens, render_key
from harness_cache.report import LEGACY_MODES, load_jsonl, metrics
from harness_cache.types import Block, Kind


class DashData:
    def __init__(self, cfg: Config, measured_dir: Path | None, ledger_path: Path | None):
        self.cfg = cfg
        self.measured = measured_dir
        self.ledger_path = ledger_path

    def pricing(self) -> dict:
        try:
            return load_pricing(self.cfg.pricing_path)
        except OSError:
            return {}

    # ── runs ───────────────────────────────────────────────────────────────
    def runs(self) -> list[dict]:
        out: dict[str, dict] = {}
        if self.measured and (self.measured / "runs").exists():
            for p in (self.measured / "runs").glob("*.json"):
                snap = json.loads(p.read_text())
                if snap.get("kind") == "load":
                    out[p.stem] = {"id": p.stem, "source": "frozen", "kind": "load", "mtime": p.stat().st_mtime}
                    continue
                m = snap["metrics"]
                out[p.stem] = {"id": p.stem, "source": "frozen", "kind": "turns", "mtime": p.stat().st_mtime,
                               **_label(m["pin"]), "turns": m["n_turns"]}
        if self.cfg.runs_dir.exists():
            for d in self.cfg.runs_dir.iterdir():
                if (d / "INVALID.json").exists():  # kept for the trace, never shown as a measurement
                    continue
                man = d / "manifest.json"
                if man.exists():
                    try:
                        mf = json.loads(man.read_text())
                    except json.JSONDecodeError:
                        continue
                    turns = d / "turns.jsonl"
                    mtime = max(man.stat().st_mtime, turns.stat().st_mtime if turns.exists() else 0)
                    c = mf.get("config", {})
                    out[d.name] = {"id": d.name, "source": "live", "kind": "turns", "mtime": mtime,
                                   "mode": LEGACY_MODES.get(c.get("mode"), c.get("mode")), "model": c.get("model") if c.get("llm") != "echo" else "echo",
                                   "embedder": c.get("embedder"), "cache_mode": mf.get("cache_mode", c.get("prefix_cache")),
                                   "date": mf.get("started_at", "")[:10], "worker": mf.get("worker")}
                elif (d / "load_summary.json").exists():
                    out[d.name] = {"id": d.name, "source": "live", "kind": "load",
                                   "mtime": (d / "load_summary.json").stat().st_mtime}
        return sorted(out.values(), key=lambda r: -r["mtime"])

    def run(self, run_id: str) -> dict:
        d = self.cfg.runs_dir / run_id
        pricing = self.pricing()
        if (d / "manifest.json").exists():
            manifest = json.loads((d / "manifest.json").read_text())
            turns, tasks = load_jsonl(d / "turns.jsonl"), load_jsonl(d / "tasks.jsonl")
            m = metrics(manifest, turns, tasks, skip_first=1, pricing=pricing)
            cost = run_cost(manifest, turns, tasks, pricing) if turns else {"priced": False}
            series = [_point(t) for t in turns]
            for p, c in zip(series, cost.get("per_turn", [])):
                p["usd"] = c["usd"]
            store_name = manifest["config"].get("store_name", "cama")
            store = self._store(Path(manifest["config"]["data_dir"]) / f"{store_name}.sqlite")
            last = turns[-1] if turns else None
            return {"id": run_id, "source": "live", "metrics": m, "series": series, "store": store,
                    "last_turn": last, "tasks": tasks[-30:], "manifest": _manifest_view(manifest)}
        if self.measured and (self.measured / "runs" / f"{run_id}.json").exists():
            snap = json.loads((self.measured / "runs" / f"{run_id}.json").read_text())
            return {"id": run_id, "source": "frozen", "metrics": snap["metrics"], "series": snap["series"],
                    "store": None, "last_turn": None, "tasks": [], "manifest": None}
        raise KeyError(run_id)

    def compare(self, a: str, b: str) -> dict:
        ra, rb = self.run(a), self.run(b)
        return {"a": ra, "b": rb, "verdicts": verdicts(ra["metrics"], rb["metrics"])}

    # ── stress, ledger ─────────────────────────────────────────────────────
    def loads(self) -> list[dict]:
        out = []
        if self.measured and (self.measured / "runs").exists():
            for p in (self.measured / "runs").glob("*.json"):
                snap = json.loads(p.read_text())
                if snap.get("kind") == "load":
                    m = snap["metrics"]
                    out.append({"id": p.stem, "summary": m["summary"], "config": m["config"], "facts": m.get("facts", {}),
                                "mtime": p.stat().st_mtime, "source": "frozen"})
        seen = {r["id"] for r in out}
        if self.cfg.runs_dir.exists():
            for d in self.cfg.runs_dir.iterdir():
                if (d / "INVALID.json").exists():
                    continue
                s = d / "load_summary.json"
                if s.exists() and d.name not in seen:
                    conf = d / "load_config.json"
                    out.append({"id": d.name, "summary": json.loads(s.read_text()),
                                "config": json.loads(conf.read_text()) if conf.exists() else {},
                                "mtime": s.stat().st_mtime})
        return sorted(out, key=lambda r: -r["mtime"])

    def ledger(self, n: int = 60) -> list[dict]:
        if not self.ledger_path or not self.ledger_path.exists():
            return []
        rows = [json.loads(x) for x in self.ledger_path.read_text().splitlines() if x]
        return rows[-n:][::-1]

    # ── live store ─────────────────────────────────────────────────────────
    def _store(self, path: Path) -> dict | None:
        if not path.exists():
            return None
        db = sqlite3.connect(path, timeout=2)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA query_only=1")
        try:
            rows = db.execute("SELECT id, kind, title, content, pinned, created_at, modified_at, accessed_at, "
                              "version, active FROM blocks").fetchall()
            counts = dict(db.execute("SELECT COUNT(*) journal, COALESCE(SUM(synced=0),0) unsynced FROM journal")
                          .fetchone())
        finally:
            db.close()
        blocks = [(Block(id=r["id"], kind=Kind.parse(r["kind"]), title=r["title"], content=r["content"],
                         pinned=bool(r["pinned"]), created_at=r["created_at"], modified_at=r["modified_at"],
                         accessed_at=r["accessed_at"], version=r["version"]), bool(r["active"])) for r in rows]
        active = sorted([b for b, a in blocks if a], key=render_key)
        recency = {b.id: i for i, b in enumerate(sorted(active, key=lambda b: -b.accessed_at))}

        def view(b: Block) -> dict:
            return {"id": b.id, "kind": b.kind.label, "title": b.title, "content": b.content, "pinned": b.pinned,
                    "version": b.version, "tokens": block_tokens(b), "recency": recency.get(b.id)}

        return {"active": [view(b) for b in active],
                "l3": [view(b) for b, a in sorted(blocks, key=lambda x: -x[0].accessed_at) if not a][:40],
                "counts": counts}


def _label(pin: dict) -> dict:
    return {"mode": LEGACY_MODES.get(pin["mode"], pin["mode"]), "model": pin["model"], "embedder": pin["embedder"],
            "cache_mode": pin.get("cache_mode", pin.get("prefix_cache")), "date": pin["date"], "worker": None}


def _manifest_view(m: dict) -> dict:
    keys = ("run_id", "started_at", "code_sha", "limits", "cache_mode", "model_info", "embedding_space",
            "pricing_hash", "scenario", "scenario_sha", "budget", "stopped", "renamed_from", "rename_reason")
    return {k: m[k] for k in keys if k in m}


def _point(t: dict) -> dict:
    return {"turn": t["turn"], "tier": (t.get("route") or {}).get("tier"), "hot_ms": t["hot_ms"],
            "llm_ms": t["timings_ms"]["llm"], "tools_ms": t["timings_ms"].get("tools", 0.0),
            "prompt": t["llm"]["prompt_tokens"], "cached": t["llm"]["cached_tokens"],
            "output": t["llm"]["output_tokens"], "request_est": t["context"]["total_tokens_est"],
            "stable_est": t["context"]["stable_prefix_tokens_est"], "calls": t["llm"].get("calls", 1),
            "cache_hit": (t.get("prefix_cache") or {}).get("hit"), "h3_hits": (t.get("h3") or {}).get("hits"),
            "paged_in": len((t.get("route") or {}).get("paged_in", [])),
            "evicted": len((t.get("route") or {}).get("evicted", [])), "block_keys": t["context"].get("block_keys")}


def verdicts(a: dict, b: dict) -> list[dict]:
    """Plain sentences, each stating a difference only when the evidence supports it."""
    out = []
    ha, hb = a.get("hot_ms"), b.get("hot_ms")
    if ha and hb:
        apart = ha["ci_hi"] < hb["ci_lo"] or hb["ci_hi"] < ha["ci_lo"]
        if apart:
            faster, slower = ("A", "B") if ha["median"] < hb["median"] else ("B", "A")
            ratio = max(ha["median"], hb["median"]) / max(1e-9, min(ha["median"], hb["median"]))
            out.append({"topic": "latency", "strength": "measured",
                        "text": f"{faster} answers faster than {slower}: median turn {min(ha['median'], hb['median']):.0f} ms "
                                f"against {max(ha['median'], hb['median']):.0f} ms ({ratio:.2f}x), 95% intervals apart."})
        else:
            out.append({"topic": "latency", "strength": "none",
                        "text": "No measured latency difference: the 95% intervals of the median turn overlap."})
    ca, cb = a.get("cost", {}), b.get("cost", {})
    if ca.get("priced") and cb.get("priced") and ca["total_usd"] and cb["total_usd"]:
        cheap, dear = ("A", "B") if ca["total_usd"] < cb["total_usd"] else ("B", "A")
        lo, hi = sorted([ca["total_usd"], cb["total_usd"]])
        out.append({"topic": "cost", "strength": "single run",
                    "text": f"{cheap} cost {100 * (1 - lo / hi):.0f}% less than {dear} ({lo:.4f} against {hi:.4f} USD). "
                            "One run each: a run to run spread is not included."})
        for name, side in (("A", ca), ("B", cb)):
            if side["cold_usd"] > side["hot_usd"]:
                out.append({"topic": "cost", "strength": "measured",
                            "text": f"In {name}, the cold path (memory extraction, cache creation) costs more than the turns themselves."})
    elif not (ca.get("priced") and cb.get("priced")):
        out.append({"topic": "cost", "strength": "absent", "text": "Cost not comparable: at least one run is not priced."})
    ta, tb = a.get("tokens"), b.get("tokens")
    if ta and tb:
        out.append({"topic": "cache", "strength": "measured",
                    "text": f"Share of prompt tokens served from the provider cache: A {100 * ta['cached_share']:.0f}%, "
                            f"B {100 * tb['cached_share']:.0f}%."})
    if a["pin"].get("scenario_sha") and b["pin"].get("scenario_sha") and a["pin"]["scenario_sha"] != b["pin"]["scenario_sha"]:
        out.insert(0, {"topic": "warning", "strength": "warning",
                       "text": "The two runs used different scenarios: differences may come from the scenario."})
    out.append({"topic": "quality", "strength": "absent", "text": "Answer quality is not measured: a cheaper run is not shown to be as good."})
    return out
