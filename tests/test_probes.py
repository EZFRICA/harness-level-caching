"""Recall probes: deterministic scoring of planted facts in an answer."""

from cama.probes import fold, score, summary
from cama.report import recall_of


def test_a_fact_is_found_whatever_the_case_accents_and_spacing():
    assert fold("  Le 1er  DÉCEMBRE ") == "le 1er decembre"
    assert score("Les slides sont dues le 1er Décembre.", ["1er decembre|1 decembre"]) == [True]


def test_each_fact_is_scored_apart_and_any_alternative_counts():
    assert score("Vol AF 702, porte 4719", ["702", "4719", "awa|konate"]) == [True, True, False]


def test_an_empty_alternative_never_matches_everything():
    assert score("anything", ["|"]) == [False]


def test_the_summary_counts_facts_and_fully_answered_probes():
    s = summary([{"turn": 1, "found": [True, False]}, {"turn": 2, "found": [True]}])
    assert (s["probes"], s["facts"], s["found"], s["answered"]) == (2, 3, 2, 1)


def test_scores_are_frozen_with_the_run_but_answers_are_not():
    m = recall_of(summary([{"turn": 3, "found": [True, False], "answer": "secret text", "expect": ["x", "y"]}]))
    assert (m["found"], m["facts"]) == (1, 2) and m["by_turn"] == [{"turn": 3, "found": [True, False]}]
    assert "secret text" not in str(m)
