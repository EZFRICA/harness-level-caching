"""harness-cache: command line interface.

  harness-cache chat                 interactive session (docker exec -it ... harness-cache chat)
  harness-cache run SCENARIO.jsonl   scripted, reproducible session
  harness-cache state                active set and store counters
  harness-cache report RUN_DIR...    statistics for one or more runs
  harness-cache reset --yes          wipe the data volume's store
  harness-cache dashboard            live read-only dashboard on :8797
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from harness_cache import logging as run_logging
from harness_cache.config import Config
from harness_cache.engine import Engine
from harness_cache.memory.active_set import block_tokens
from harness_cache.metrics import RunRecorder
from harness_cache.report import summarize
from harness_cache.store.sqlite_store import SQLiteStore


def _cfg(args: argparse.Namespace) -> Config:
    cfg = Config.from_env()
    if getattr(args, "mode", None):
        cfg.mode = args.mode
    if getattr(args, "store", None):
        cfg.store_name = args.store
    return cfg


def _engine(cfg: Config, run_id: str, session: str, extra: dict | None = None) -> Engine:
    rec = RunRecorder(cfg.runs_dir, run_id)
    run_logging.setup(rec.dir / "run.log")
    rec.manifest(cfg.as_dict(), extra)
    engine = Engine(cfg, rec, session=session)
    rec.note(engine.manifest_extra())
    return engine


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _print_state(engine_or_store) -> None:
    if isinstance(engine_or_store, Engine):
        store, active = engine_or_store.store, engine_or_store.active.render_order()
    else:
        store = engine_or_store
        from harness_cache.memory.active_set import render_key

        active = sorted(store.active_blocks(), key=render_key)
    print(json.dumps(store.count()))
    print(f"{'#':>2} {'kind':<12}{'pin':<4}{'v':>3} {'tok':>5}  id / title")
    for i, b in enumerate(active):
        print(f"{i:>2} {b.kind.label:<12}{'*' if b.pinned else '':<4}{b.version:>3} {block_tokens(b):>5}  "
              f"{b.id}: {b.title}")


async def cmd_chat(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    engine = _engine(cfg, args.run_id or f"{_stamp()}-chat-{cfg.mode}", args.session)
    await engine.start()
    loop = asyncio.get_running_loop()
    print(f"harness-cache chat, mode={cfg.mode} llm={engine.llm.name} run={engine.recorder.run_id}")
    print("commands: /state  /drain  /quit")
    try:
        while True:
            line = await loop.run_in_executor(None, _readline)
            if line is None or line.strip() == "/quit":
                break
            line = line.strip()
            if not line:
                continue
            if line == "/state":
                _print_state(engine)
                continue
            if line == "/drain":
                await engine.drain()
                print("cold path drained")
                continue
            res = await engine.turn(line)
            r = res.record
            tier = r["route"]["tier"] if r["route"] else "-"
            print(f"\n{res.text}\n  [{tier} | hot {r['hot_ms']:.0f}ms | llm {r['timings_ms']['llm']:.0f}ms | "
                  f"prompt {r['llm']['prompt_tokens']} cached {r['llm']['cached_tokens']}]\n")
        await engine.drain()
    finally:
        await engine.close()


def _readline() -> str | None:
    try:
        return input("> ")
    except EOFError:
        return None


async def cmd_run(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    scenario = Path(args.scenario)
    steps = [json.loads(line) for line in scenario.read_text().splitlines() if line.strip()]
    run_id = args.run_id or f"{_stamp()}-{cfg.mode}-{scenario.stem}"
    if args.fresh:
        cfg.store_name = args.store or f"run-{run_id}"
        for suffix in ("", "-wal", "-shm"):
            Path(f"{cfg.store_path}{suffix}").unlink(missing_ok=True)
    engine = _engine(cfg, run_id, args.session or run_id, {"scenario": str(scenario), "scenario_sha": __import__("hashlib").blake2b(scenario.read_bytes(), digest_size=6).hexdigest(),
                                                   "drain_between_turns": not args.no_drain})
    await engine.start()
    from harness_cache.cost import BudgetExceeded
    from harness_cache.probes import score, summary
    probes: list[dict] = []
    try:
        for step in steps:
            if "seed" in step:
                out = await engine.seed(step["seed"])
                print(f"seeded {len(out['inserted'])} blocks, evicted {out['evicted']}")
            elif "user" in step:
                try:
                    res = await engine.turn(step["user"])
                except BudgetExceeded as e:
                    print(f"\nstopped: budget cap reached ({e})")
                    engine.recorder.note({"stopped": "budget", "stop_detail": str(e)})
                    break
                r = res.record
                if step.get("expect"):  # a recall probe: scored on the answer, recorded in the manifest
                    probes.append({"turn": r["turn"], "expect": step["expect"],
                                   "found": score(res.text, step["expect"]), "answer": res.text[:400]})
                tier = r["route"]["tier"] if r["route"] else "-"
                print(f"[{r['turn']:>3}] {tier:<9} hot={r['hot_ms']:7.0f}ms  "
                      f"tok={r['context']['total_tokens_est']:>5}  stable={r['context']['stable_prefix_tokens_est']:>5}  "
                      f"cached={r['llm']['cached_tokens']:>5}  | {step['user'][:60]}")
                if not args.no_drain:
                    await engine.drain()  # simulate idle time between user messages
            elif step.get("drain"):
                await engine.drain()
        await engine.drain()
        engine.recorder.note({"budget": {"cap_usd": engine.budget.max_usd, "spent_usd": engine.budget.spent_usd,
                                         "paid_calls": engine.budget.calls}})
        if probes:
            s = summary(probes)
            engine.recorder.note({"probes": s})
            print(f"recall probes: {s['found']}/{s['facts']} facts, {s['answered']}/{s['probes']} probes fully answered")
        print(f"\nrun dir: {engine.recorder.dir}  spent {engine.budget.spent_usd:.4f} USD (cap {engine.budget.max_usd})")
    finally:
        await engine.close()


def cmd_state(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    store = SQLiteStore(cfg.store_path)
    _print_state(store)
    store.close()


def cmd_report(args: argparse.Namespace) -> None:
    from harness_cache.cost import load_pricing

    pricing = load_pricing(_cfg(args).pricing_path)
    for d in args.run_dirs:
        print(summarize(Path(d), skip_first=args.skip_first, pricing=pricing))
        print()


def cmd_reset(args: argparse.Namespace) -> None:
    if not args.yes:
        sys.exit("refusing to reset without --yes")
    cfg = _cfg(args)
    store = SQLiteStore(cfg.store_path)
    store.reset()
    store.close()
    print(f"store reset: {cfg.store_path}")


def cmd_ledger(args: argparse.Namespace) -> None:
    from harness_cache.trace import repo_root
    from harness_cache.trace.ledger import Ledger, write_pages

    root = repo_root()
    ledger = Ledger(root / "docs" / "ledger.jsonl")
    if args.ledger_cmd == "add":
        event = {k: v for k, v in {
            "who": args.who, "kind": args.kind, "title": args.title, "why": args.why,
            "files": args.files, "verified": args.verified, "live": args.live, "runs": args.runs,
            "decisions": args.decisions, "data": args.data, "result": args.result, "detail": args.detail,
        }.items() if v}
        e = ledger.append(event)
        write_pages(ledger, root / "docs" / "journal")
        print(f"ledger #{e['seq']} {e['kind']}: {e['title']}")
    elif args.ledger_cmd == "render":
        for p in write_pages(ledger, root / "docs" / "journal"):
            print(p.relative_to(root))
    elif args.ledger_cmd == "check":
        problems = ledger.check()
        print("\n".join(problems) if problems else f"ledger ok: {len(ledger.events())} events, chain intact")
        sys.exit(1 if problems else 0)


def cmd_figures(args: argparse.Namespace) -> None:
    from harness_cache.trace import figures, repo_root

    measured = repo_root() / "docs" / "measured"
    if args.figures_cmd == "snapshot":
        for d in args.run_dirs:
            print(figures.snapshot(Path(d), measured / "runs", skip_first=args.skip_first,
                                   pricing_path=_cfg(args).pricing_path, price_as=args.price_as))
    elif args.figures_cmd == "write":
        for p in figures.write(measured):
            print(p)
    elif args.figures_cmd == "check":
        stale = [n for n, t in figures.generate(measured).items()
                 if not (measured / n).exists() or (measured / n).read_text() != t]
        print("stale: " + ", ".join(stale) if stale else "figures ok")
        sys.exit(1 if stale else 0)


def cmd_decisions(args: argparse.Namespace) -> None:
    from harness_cache.trace import decisions, repo_root

    problems = decisions.check(repo_root())
    print("\n".join(problems) if problems else "decisions ok")
    sys.exit(1 if problems else 0)


def cmd_serve(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    from harness_cache.cluster.coordinator import PostgresCoordinator
    from harness_cache.cluster.service import Worker, create_app

    cfg = _cfg(args)
    dsn = os.environ.get("HC_COORDINATOR")
    if not dsn:
        sys.exit("HC_COORDINATOR (a Postgres DSN) is required to serve as a cluster worker")
    run_id = os.environ.get("HC_CLUSTER_RUN", _stamp())
    run_logging.setup(cfg.runs_dir / f"{run_id}-{args.worker}.log")
    worker = Worker(cfg, args.worker, PostgresCoordinator(dsn), run_id, lease_ttl_s=args.lease_ttl)
    uvicorn.run(create_app(worker), host=args.host, port=args.port, log_level="warning")


async def cmd_load(args: argparse.Namespace) -> None:
    from harness_cache.cluster.load import Load, LoadConfig

    cfg = _cfg(args)
    run_id = args.run_id or f"{_stamp()}-load"
    out_dir = cfg.runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    lc = LoadConfig(workers=args.workers.split(","), agents=args.agents, turns=args.turns,
                    concurrency=args.concurrency, scenario=Path(args.scenario), out=out_dir / "load.jsonl",
                    think_ms=args.think_ms)
    (out_dir / "load_config.json").write_text(json.dumps({**vars(args), "started_at": _stamp()}, indent=2, default=str))
    summary = await Load(lc).run()
    (out_dir / "load_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def cmd_gen(args: argparse.Namespace) -> None:
    from harness_cache.stress.gen import write

    print(write(args.kind, args.turns, Path(args.out), args.seed))


def cmd_dashboard(args: argparse.Namespace) -> None:
    from harness_cache.dash.app import serve

    serve(_cfg(args), args.host, args.port)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="harness-cache")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat")
    c.add_argument("--session", default="default")
    c.add_argument("--mode", choices=["extract", "topk", "naive", "compact"])
    c.add_argument("--store")
    c.add_argument("--run-id")

    r = sub.add_parser("run")
    r.add_argument("scenario")
    r.add_argument("--session", help="default: the run id")
    r.add_argument("--mode", choices=["extract", "topk", "naive", "compact"])
    r.add_argument("--store", help="store name under the data volume (default: HC_STORE)")
    r.add_argument("--fresh", action="store_true",
                   help="start from an empty store (default name run-<run_id>), for comparable experiments")
    r.add_argument("--run-id")
    r.add_argument("--no-drain", action="store_true", help="do not wait for the cold path between turns")

    st = sub.add_parser("state")
    st.add_argument("--store")

    rp = sub.add_parser("report")
    rp.add_argument("run_dirs", nargs="+")
    rp.add_argument("--skip-first", type=int, default=0, help="exclude warm-up turns")

    rs = sub.add_parser("reset")
    rs.add_argument("--yes", action="store_true")
    rs.add_argument("--store")

    lg = sub.add_parser("ledger", help="the trace of every action")
    lsub = lg.add_subparsers(dest="ledger_cmd", required=True)
    la = lsub.add_parser("add")
    la.add_argument("--who", default="claude")
    la.add_argument("--kind", required=True, choices=["action", "run", "decision", "measurement", "fault", "review"])
    la.add_argument("--title", required=True)
    la.add_argument("--why", required=True)
    la.add_argument("--files", nargs="*")
    la.add_argument("--verified")
    la.add_argument("--live")
    la.add_argument("--runs", nargs="*")
    la.add_argument("--decisions", nargs="*")
    la.add_argument("--data")
    la.add_argument("--result", required=True, choices=["done", "partial", "failed"])
    la.add_argument("--detail")
    lsub.add_parser("render")
    lsub.add_parser("check")

    fg = sub.add_parser("figures", help="generated measurement pages")
    fsub = fg.add_subparsers(dest="figures_cmd", required=True)
    fs = fsub.add_parser("snapshot")
    fs.add_argument("run_dirs", nargs="+")
    fs.add_argument("--skip-first", type=int, default=1)
    fs.add_argument("--price-as", help="also price the run as if it had used this model (a projection)")
    fsub.add_parser("write")
    fsub.add_parser("check")

    sub.add_parser("decisions", help="check decisions against their evidence")

    sv = sub.add_parser("serve", help="run as a cluster worker (needs HC_COORDINATOR)")
    sv.add_argument("--worker", required=True)
    sv.add_argument("--host", default="0.0.0.0")
    sv.add_argument("--port", type=int, default=8797)
    sv.add_argument("--lease-ttl", type=float, default=6.0)

    ld = sub.add_parser("load", help="drive many agents against the cluster workers")
    ld.add_argument("--workers", required=True, help="comma separated base URLs")
    ld.add_argument("--agents", type=int, default=8)
    ld.add_argument("--turns", type=int, default=24)
    ld.add_argument("--concurrency", type=int, default=8)
    ld.add_argument("--scenario", default="/scenarios/agentic_fr_v1.jsonl")
    ld.add_argument("--think-ms", type=float, default=0.0)
    ld.add_argument("--run-id")

    gn = sub.add_parser("gen", help="generate a long synthetic scenario for stress profiles")
    gn.add_argument("--kind", choices=["zipf", "churn"], required=True)
    gn.add_argument("--turns", type=int, default=200)
    gn.add_argument("--seed", type=int, default=20261023)
    gn.add_argument("--out", default="scenarios/generated")

    db = sub.add_parser("dashboard", help="live read-only dashboard")
    db.add_argument("--host", default="0.0.0.0")
    db.add_argument("--port", type=int, default=8797)

    args = p.parse_args(argv)
    if args.cmd == "chat":
        asyncio.run(cmd_chat(args))
    elif args.cmd == "run":
        asyncio.run(cmd_run(args))
    elif args.cmd == "state":
        cmd_state(args)
    elif args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "reset":
        cmd_reset(args)
    elif args.cmd == "dashboard":
        cmd_dashboard(args)
    elif args.cmd == "gen":
        cmd_gen(args)
    elif args.cmd == "serve":
        cmd_serve(args)
    elif args.cmd == "load":
        asyncio.run(cmd_load(args))
    elif args.cmd == "ledger":
        cmd_ledger(args)
    elif args.cmd == "figures":
        cmd_figures(args)
    elif args.cmd == "decisions":
        cmd_decisions(args)


if __name__ == "__main__":
    main()
