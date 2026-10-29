"""EmbeddingGemma 2: prefixes per kind, Matryoshka cut, its own thresholds. The model file is
gitignored; the test that loads it is skipped when it is absent, the others need no model."""

import os

import numpy as np
import pytest

from harness_cache.adapters.gemma import PREFIXES, GemmaEmbedder, truncate
from harness_cache.config import Config
from harness_cache.engine import Engine
from harness_cache.memory.router import URITHI_THRESHOLDS, URITHI_THRESHOLDS_BY_EMBEDDER
from harness_cache.metrics import RunRecorder
from harness_cache.types import Kind

MODEL = os.path.join(os.path.dirname(__file__), "..", "models", "embeddinggemma-2-onnx")


def test_truncate_keeps_the_first_coordinates_and_unit_norm():
    m = np.array([[3.0, 4.0, 12.0], [0.0, 2.0, 5.0]], np.float32)
    out = truncate(m, 2)
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[0], [0.6, 0.8], rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-6)
    with pytest.raises(ValueError):
        truncate(np.array([[0.0, 0.0, 1.0]], np.float32), 2)


class _Stub(GemmaEmbedder):
    def __init__(self, dim=768):  # no model: records what would be encoded
        self.dim, self.seen = dim, []
        self._gates = {}

    def _run(self, texts):
        self.seen += texts
        return np.ones((len(texts), 768), np.float32)


async def test_a_query_and_a_stored_text_get_different_prefixes():
    e = _Stub(dim=256)
    q = await e.embed(["my locker code"], "query")
    d = await e.embed(["My locker code is 4831."], "document")
    assert e.seen == [PREFIXES["query"] + "my locker code", PREFIXES["document"] + "My locker code is 4831."]
    assert PREFIXES["query"] != PREFIXES["document"]
    assert q.shape == d.shape == (1, 256)


def test_an_unknown_dimension_is_refused_before_loading():
    with pytest.raises(ValueError):
        GemmaEmbedder("/nowhere", dim=300)
    with pytest.raises(RuntimeError):
        GemmaEmbedder("/nowhere", dim=768)


def test_gemma_has_its_own_routing_and_search_thresholds(monkeypatch):
    monkeypatch.setenv("HC_EMBEDDER", "gemma")
    cfg = Config.from_env()
    assert (cfg.tau_active, cfg.tau_l3) == (0.748, 0.775)
    g = URITHI_THRESHOLDS_BY_EMBEDDER["gemma"]
    assert set(g) == set(Kind) and all(g[k] > URITHI_THRESHOLDS[k] for k in Kind)


async def test_the_search_reads_the_threshold_of_its_embedder(tmp_path):
    seed = [{"id": "trip", "kind": "projet", "title": "Trip to Abidjan", "content": "- Flight booked.",
             "active": False}]
    found = {}
    for embedder in ("hash", "gemma"):
        cfg = Config(data_dir=tmp_path / embedder, runs_dir=tmp_path / "r", archive_dir=tmp_path / "a",
                     tools=False, route_policy="urithi")
        eng = Engine(cfg, RunRecorder(tmp_path / "r", embedder))
        await eng.seed(seed)
        eng.cfg.embedder = embedder  # same hash vectors, only the threshold table changes

        async def near(texts, kind):  # cosine 0.6 to the stored block, no shared word
            v = eng.store.get("trip").vector
            w = np.zeros_like(v)
            w[np.argmin(np.abs(v))] = 1.0
            w = 0.6 * v + 0.8 * (w - (w @ v) * v) / np.linalg.norm(w - (w @ v) * v)
            return np.array([w], np.float32)

        eng.router.embedder = type("E", (), {"embed": staticmethod(near)})()
        res = await eng.router.recall("zzz")
        found[embedder] = [b for b, _ in res[1].hits]
    assert found == {"hash": ["trip"], "gemma": []}


@pytest.mark.skipif(not os.path.isfile(os.path.join(MODEL, "onnx", "model.onnx")), reason="model not downloaded")
async def test_the_real_model_ranks_the_matching_statement_first():
    e = GemmaEmbedder(MODEL, dim=768, threads=1)
    q = await e.embed(["Which planet is known as the Red Planet?"], "query")
    d = await e.embed(["Venus is often called Earth's twin.",
                       "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
                       "Jupiter has a prominent red spot."], "document")
    assert int(np.argmax(d @ q[0])) == 1
    one = await e.embed(["Jupiter has a prominent red spot."], "document")
    np.testing.assert_allclose(one[0], d[2], atol=1e-4)  # padding in a batch changes nothing
