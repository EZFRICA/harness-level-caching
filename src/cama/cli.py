"""cama — command line interface.

  cama chat                 interactive session (docker exec -it ... cama chat)
  cama run SCENARIO.jsonl   scripted, reproducible session
  cama state                active set and store counters
  cama report RUN_DIR...    statistics for one or more runs
  cama reset --yes          wipe the data volume's store
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from cama import logging as cama_logging
from cama.config import Config
from cama.engine import Engine
from cama.memory.active_set import block_tokens
from cama.metrics import RunRecorder
from cama.report import summarize
from cama.store.sqlite_store import SQLiteStore


def _cfg(args: argparse.Namespace) -> Config:
    cfg = Config.from_env()
    if getattr(args, "mode", None):
        cfg.mode = args.mode
    if getattr(args, "store", None):
        cfg.store_name = args.store
    return cfg


def _engine(cfg: Config, run_id: str, session: str, extra: dict | None = None) -> Engine:
    rec = RunRecorder(cfg.runs_dir, run_id)
    cama_logging.setup(rec.dir / "cama.log")
    rec.manifest(cfg.as_dict(), extra)
    return Engine(cfg, rec, session=session)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _print_state(engine_or_store) -> None:
    if isinstance(engine_or_store, Engine):
        store, active = engine_or_store.store, engine_or_store.active.render_order()
    else:
        store = engine_or_store
        from cama.memory.active_set import render_key

        active = sorted(store.active_blocks(), key=render_key)
    print(json.dumps(store.count()))
    print(f"{'#':>2} {'kind':<12}{'pin':<4}{'v':>3} {'tok':>5}  id / title")
    for i, b in enumerate(active):
        print(f"{i:>2} {b.kind.label:<12}{'*' if b.pinned else '':<4}{b.version:>3} {block_tokens(b):>5}  "
              f"{b.id} — {b.title}")


async def cmd_chat(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    engine = _engine(cfg, args.run_id or f"{_stamp()}-chat-{cfg.mode}", args.session)
    await engine.start()
    loop = asyncio.get_running_loop()
    print(f"cama chat — mode={cfg.mode} llm={engine.llm.name} run={engine.recorder.run_id}")
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
    engine = _engine(cfg, run_id, args.session or run_id, {"scenario": str(scenario), "drain_between_turns": not args.no_drain})
    await engine.start()
    try:
        for step in steps:
            if "seed" in step:
                out = await engine.seed(step["seed"])
                print(f"seeded {len(out['inserted'])} blocks, evicted {out['evicted']}")
            elif "user" in step:
                res = await engine.turn(step["user"])
                r = res.record
                tier = r["route"]["tier"] if r["route"] else "-"
                print(f"[{r['turn']:>3}] {tier:<9} hot={r['hot_ms']:7.0f}ms  "
                      f"tok={r['context']['total_tokens_est']:>5}  stable={r['context']['stable_prefix_tokens_est']:>5}  "
                      f"cached={r['llm']['cached_tokens']:>5}  | {step['user'][:60]}")
                if not args.no_drain:
                    await engine.drain()  # simulate idle time between user messages
            elif step.get("drain"):
                await engine.drain()
        await engine.drain()
        print(f"\nrun dir: {engine.recorder.dir}")
    finally:
        await engine.close()


def cmd_state(args: argparse.Namespace) -> None:
    cfg = _cfg(args)
    store = SQLiteStore(cfg.store_path)
    _print_state(store)
    store.close()


def cmd_report(args: argparse.Namespace) -> None:
    for d in args.run_dirs:
        print(summarize(Path(d), skip_first=args.skip_first))
        print()


def cmd_reset(args: argparse.Namespace) -> None:
    if not args.yes:
        sys.exit("refusing to reset without --yes")
    cfg = _cfg(args)
    store = SQLiteStore(cfg.store_path)
    store.reset()
    store.close()
    print(f"store reset: {cfg.store_path}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cama")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("chat")
    c.add_argument("--session", default="default")
    c.add_argument("--mode", choices=["cama", "topk", "naive"])
    c.add_argument("--store")
    c.add_argument("--run-id")

    r = sub.add_parser("run")
    r.add_argument("scenario")
    r.add_argument("--session", help="default: the run id")
    r.add_argument("--mode", choices=["cama", "topk", "naive"])
    r.add_argument("--store", help="store name under the data volume (default: CAMA_STORE)")
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


if __name__ == "__main__":
    main()
