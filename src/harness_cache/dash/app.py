"""The CAMA dashboard: a read-only FastAPI app on port 8797, assets served locally (no CDN)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from harness_cache.config import Config
from harness_cache.dash.data import DashData

STATIC = Path(__file__).parent / "static"


def create_app(cfg: Config):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    from fastapi.staticfiles import StaticFiles

    measured = Path(os.environ.get("HC_MEASURED_DIR", "/measured"))
    ledger = Path(os.environ.get("HC_LEDGER", "/docs/ledger.jsonl"))
    data = DashData(cfg, measured if measured.exists() else None, ledger if ledger.exists() else None)
    coord_dsn = os.environ.get("HC_COORDINATOR")
    state = {"coord": None}

    app = FastAPI(title="CAMA dashboard")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def nocache(payload, status: int = 200):
        return JSONResponse(payload, status_code=status, headers={"Cache-Control": "no-store"})

    @app.get("/")
    async def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-store"})

    @app.get("/healthz")
    async def health():
        return {"ok": True}

    @app.get("/api/runs")
    async def runs():
        return nocache(data.runs())

    def known(run_id: str) -> str:
        """Only ids the dashboard listed: no path outside runs/ or docs/measured/runs (review finding)."""
        if run_id not in {r["id"] for r in data.runs()}:
            raise HTTPException(404, "unknown run")
        return run_id

    @app.get("/api/run/{run_id}")
    async def run(run_id: str):
        known(run_id)
        try:
            return nocache(await asyncio.to_thread(data.run, run_id))
        except KeyError:
            raise HTTPException(404, f"no run {run_id}") from None

    @app.get("/api/compare")
    async def compare(a: str, b: str):
        known(a), known(b)
        try:
            return nocache(await asyncio.to_thread(data.compare, a, b))
        except KeyError as e:
            raise HTTPException(404, f"no run {e}") from None

    @app.get("/api/loads")
    async def loads():
        return nocache(data.loads())

    @app.get("/api/ledger")
    async def ledger_view():
        return nocache(data.ledger())

    @app.get("/api/pricing")
    async def pricing():
        return nocache(data.pricing())

    @app.get("/api/cluster")
    async def cluster():
        if not coord_dsn:
            return nocache({"available": False, "reason": "single worker mode (no HC_COORDINATOR)"})
        try:
            if state["coord"] is None:
                from harness_cache.cluster.coordinator import PostgresCoordinator
                c = PostgresCoordinator(coord_dsn)
                await asyncio.wait_for(c.connect(create_schema=False), 3)
                state["coord"] = c
            view = await asyncio.wait_for(state["coord"].view(), 3)
            return nocache({"available": True, **json.loads(json.dumps(view, default=str))})
        except Exception as e:  # noqa: BLE001: the cluster may be down; say so
            if state["coord"] is not None:
                await state["coord"].close()
            state["coord"] = None
            return nocache({"available": False, "reason": f"coordinator unreachable: {type(e).__name__}"})

    @app.get("/api/stream")
    async def stream():
        """Server-sent events: a signature of the run list every second, so pages refresh on change."""
        async def gen():
            last = None
            while True:
                sig = json.dumps([(r["id"], round(r["mtime"], 1)) for r in data.runs()[:15]])
                if sig != last:
                    last = sig
                    yield f"event: runs\ndata: {sig}\n\n"
                else:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(1.0)
        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})

    return app


def serve(cfg: Config, host: str, port: int) -> None:
    import uvicorn

    uvicorn.run(create_app(cfg), host=host, port=port, log_level="warning")
