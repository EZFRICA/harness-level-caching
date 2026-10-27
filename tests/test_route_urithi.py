"""The urithi routing policy: whole-store search, per-kind thresholds, words beside meaning."""

from cama.config import Config
from cama.engine import Engine
from cama.memory.keywords import bm25, fuse
from cama.metrics import RunRecorder


def _engine(tmp_path, policy):
    cfg = Config(data_dir=tmp_path / "d", runs_dir=tmp_path / "r", archive_dir=tmp_path / "a",
                 tools=False, route_policy=policy, capacity=3)
    return Engine(cfg, RunRecorder(tmp_path / "r", "x"))


SEED = [
    {"id": "trip", "kind": "projet", "title": "Trip to Abidjan", "content": "- Flight booked, hotel on the Plateau."},
    {"id": "talk", "kind": "projet", "title": "DevFest talk", "content": "- Slides due December 1."},
    {"id": "health", "kind": "projet", "title": "Health check", "content": "- Annual check in spring."},
    {"id": "course", "kind": "projet", "title": "Japanese calligraphy course",
     "content": "- Signed up for a Japanese calligraphy course on Saturdays.", "active": False},
]


async def test_a_word_shared_with_a_stored_block_pages_it_in(tmp_path):
    eng = _engine(tmp_path, "urithi")
    await eng.seed(SEED)
    assert "course" not in eng.active.ids()
    res = await eng.router.route("What course did I sign up for?")
    assert "course" in res.paged_in and "course" in eng.active.ids()


async def test_the_cascade_default_is_unchanged(tmp_path):
    eng = _engine(tmp_path, "cascade")
    await eng.seed(SEED)
    res = await eng.router.route("What course did I sign up for?")
    assert res.tier in {"lexical", "active", "l3_fault", "miss"}  # the old path still answers


def test_bm25_weighs_rare_words_and_fusion_keeps_meaning_order_on_ties():
    s = bm25("calligraphy course", ["a japanese calligraphy course", "a course on trains", "nothing here"])
    assert s[0] > s[1] > s[2] == 0
    assert fuse(["a", "b"], ["b", "a"])[0] == "a"
