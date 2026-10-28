"""The dashboard serves every view's data from the same computations as the reports, read only."""

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from harness_cache.config import Config

ROOT = Path(__file__).resolve().parent.parent


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("HC_MEASURED_DIR", str(ROOT / "docs" / "measured"))
    monkeypatch.setenv("HC_LEDGER", str(ROOT / "docs" / "ledger.jsonl"))
    monkeypatch.delenv("HC_COORDINATOR", raising=False)
    from harness_cache.dash.app import create_app
    return TestClient(create_app(Config(runs_dir=tmp_path / "runs", data_dir=tmp_path / "d")))


def test_the_page_and_its_assets_are_served_without_any_external_url(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    html = c.get("/").text
    assert "/static/app.js" in html and "http" not in html.replace("http-equiv", "")
    for asset in ("app.js", "style.css"):
        body = c.get(f"/static/{asset}").text
        assert "cdn" not in body.lower() and "https://" not in body


def test_frozen_runs_are_listed_and_compared_with_honest_verdicts(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    runs = {r["id"]: r for r in c.get("/api/runs").json()}
    assert "g35lite-onnx-agentic-cama-v1" in runs and runs["g35lite-onnx-agentic-cama-v1"]["source"] == "frozen"
    d = c.get("/api/compare", params={"a": "g35lite-onnx-agentic-nopc-v1", "b": "g35lite-onnx-agentic-cama-v1"}).json()
    texts = " ".join(v["text"] for v in d["verdicts"])
    assert "faster" in texts, "the latency intervals of these two runs are apart"
    assert any(v["topic"] == "quality" and v["strength"] == "absent" for v in d["verdicts"])
    assert d["a"]["metrics"]["pin"]["cache_mode"] == "none" and d["b"]["metrics"]["pin"]["cache_mode"] == "explicit"


def test_cluster_view_says_unavailable_rather_than_failing(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/cluster").json() == {"available": False,
                                            "reason": "single worker mode (no HC_COORDINATOR)"}


def test_the_ledger_is_served_newest_first(tmp_path, monkeypatch):
    ev = _client(tmp_path, monkeypatch).get("/api/ledger").json()
    assert ev and ev[0]["seq"] > ev[-1]["seq"]


def _m(median, lo, hi, total=None, cold=0.0, share=0.0, sha="s"):
    m = {"hot_ms": {"median": median, "ci_lo": lo, "ci_hi": hi}, "tokens": {"cached_share": share},
         "pin": {"scenario_sha": sha}, "cost": {"priced": total is not None}}
    if total is not None:
        m["cost"].update(total_usd=total, hot_usd=total - cold, cold_usd=cold)
    return m


def test_a_verdict_never_claims_a_latency_difference_inside_overlapping_intervals():
    from harness_cache.dash.data import verdicts
    v = verdicts(_m(1000, 900, 1100), _m(1050, 950, 1150))
    assert any(x["topic"] == "latency" and x["strength"] == "none" for x in v)
    v = verdicts(_m(1000, 950, 1050), _m(2000, 1900, 2100))
    assert any(x["topic"] == "latency" and x["strength"] == "measured" and x["text"].startswith("A answers faster") for x in v)


def test_verdicts_flag_unpriced_runs_different_scenarios_and_unmeasured_quality():
    from harness_cache.dash.data import verdicts
    v = verdicts(_m(1, 0, 2, total=0.02, cold=0.015, sha="x"), _m(1, 0, 2, sha="y"))
    topics = {(x["topic"], x["strength"]) for x in v}
    assert ("warning", "warning") in topics and ("cost", "absent") in topics and ("quality", "absent") in topics
    v = verdicts(_m(1, 0, 2, total=0.02, cold=0.015), _m(1, 0, 2, total=0.01))
    assert any("cold path" in x["text"] and x["text"].startswith("In A") for x in v)


def test_a_run_id_outside_the_listed_runs_is_refused(tmp_path, monkeypatch):
    """Review: /api/compare?a=../.. read paths outside runs/."""
    c = _client(tmp_path, monkeypatch)
    assert c.get("/api/run/..%2F..%2Fetc").status_code == 404
    assert c.get("/api/compare", params={"a": "../../etc", "b": "x"}).status_code == 404


def test_a_run_marked_invalid_is_never_shown(tmp_path, monkeypatch):
    d = tmp_path / "runs" / "broken"
    d.mkdir(parents=True)
    (d / "load_summary.json").write_text('{"requests": 1, "ok": 0}')
    (d / "INVALID.json").write_text('{"invalid": "mixed code versions"}')
    c = _client(tmp_path, monkeypatch)
    assert "broken" not in {r["id"] for r in c.get("/api/runs").json()}
    assert "broken" not in {r["id"] for r in c.get("/api/loads").json()}
