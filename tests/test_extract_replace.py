"""H5: a corrected fact replaces its line, only when the switch is on."""

import json

from harness_cache.adapters.local import HashEmbedder
from harness_cache.memory.active_set import ActiveSet
from harness_cache.memory.extractor import Extractor, replace_line
from harness_cache.store.sqlite_store import SQLiteStore
from harness_cache.types import Block, Kind, LLMResult

TALK = "- Talk de 30 minutes le 14 decembre.\n- Duree revisee : 25 minutes, creneau 11h."


def test_the_line_holding_the_old_value_is_replaced_in_place():
    out, found = replace_line(TALK, "Duree revisee : 25 minutes, creneau 11h.", "Duree : 30 minutes, creneau 11h.")
    assert found and out.splitlines() == ["- Talk de 30 minutes le 14 decembre.", "- Duree : 30 minutes, creneau 11h."]


def test_an_absent_old_line_appends_instead_of_losing_the_fact():
    out, found = replace_line(TALK, "no such line", "new fact")
    assert not found and out.endswith("\n- new fact") and "25 minutes" in out


def test_an_empty_old_line_never_matches_the_first_line():
    assert replace_line(TALK, "  ", "x")[1] is False


class _Fixed:
    name = "fixed"

    def __init__(self, ops):
        self.ops, self.systems = ops, []

    async def generate(self, system, messages, **kw):
        self.systems.append(system)
        return LLMResult(text=json.dumps({"ops": self.ops}))


async def _extract(tmp_path, replace):
    store = SQLiteStore(tmp_path / f"s{replace}.sqlite")
    emb = HashEmbedder()
    store.bind_embedding_space(emb.name, emb.dim)
    active = ActiveSet(12, 5000, 3)
    block = Block("talk", Kind.PROJET, "Talk", TALK)
    block.vector = (await emb.embed([block.text], "document"))[0]
    active.admit(block)
    store.upsert(block, "create", active=True)
    llm = _Fixed([{"op": "replace", "id": "talk", "old": "Duree revisee : 25 minutes", "text": "Duree : 30 minutes"}])
    rep = await Extractor(llm, emb, store, active, replace=replace).run("Mon talk repasse a 30 minutes.", "Note.")
    return rep, active.get("talk"), llm.systems[0]


async def test_with_the_switch_on_a_correction_leaves_no_stale_line(tmp_path):
    rep, block, system = await _extract(tmp_path, True)
    assert rep.replaced == ["talk"] and "25 minutes" not in block.content and '"replace"' in system


async def test_with_the_switch_off_a_replace_op_is_refused_and_the_prompt_unchanged(tmp_path):
    rep, block, system = await _extract(tmp_path, False)
    assert rep.invalid_ops == 1 and block.content == TALK and '"replace"' not in system
