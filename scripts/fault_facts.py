"""Derive the facts of a fault run from its raw files, so RTO and RPO are computed, never typed.

    uv run python scripts/fault_facts.py RUN_DIR --fault "kill -9 of w2" [--kill-ts FILE] [--local JSON ...]

RTO: for each agent whose epoch moved up, time from the fault to its first answer on the new owner.
RPO: journal rows not yet replicated at the fault, read from the dead worker's local stores.
"""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("run_dir")
p.add_argument("--fault", required=True)
p.add_argument("--kill-ts")
p.add_argument("--local", nargs="*", default=[], help="JSON files of local store counts read after the fault")
a = p.parse_args()
run = Path(a.run_dir)
rows = [json.loads(x) for x in (run / "load.jsonl").read_text().splitlines() if x]
facts = {"fault": a.fault}
if a.kill_ts:
    t0 = float(Path(a.kill_ts).read_text())
    rtos = []
    for agent in sorted({r["agent"] for r in rows}):
        rs = sorted((r for r in rows if r["agent"] == agent), key=lambda r: r["i"])
        if max((r["epoch"] or 0) for r in rs) > 1:
            first = next(r for r in rs if (r["epoch"] or 0) > 1)
            rtos.append(round(first["ts"] - t0, 1))
    facts["rto_s"] = max(rtos) if rtos else "no takeover"
    facts["rto_per_agent_s"] = rtos
rpo = 0
for f in a.local:
    rpo += sum(v["unsynced_at_crash"] for v in json.loads(Path(f).read_text()).values())
facts["rpo_rows"] = rpo if a.local else "not measured"
(run / "facts.json").write_text(json.dumps(facts, indent=1))
print(json.dumps(facts))
