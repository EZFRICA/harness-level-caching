"""Freeze the probe set HC_PIN_JUDGE=decisions replays before each fold: eight texts from the
hand-written training set (none from a test scenario), each asked twice to the Decisions API to
check the answer is repeatable, written with its probability to scenarios/labels/pin_probes_v1.jsonl.
A paid call (16 small requests). The key is read from .env by its variable name, never printed.

    uv run python scripts/freeze_pin_probes.py [--key-var GEMINI_API_KEY_2]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from eval_fact_decisions import key_from_env_file  # noqa: E402

from harness_cache.memory.pin_judge import MODEL, PROBES, QUESTION, gemini_asker  # noqa: E402

TEXTS = [  # clear and borderline cases, both labels
    "My passport expires on October 3, 2028.",
    "The apartment door code is 1907B.",
    "Correction: the launch moved to October 14.",
    "I prefer meetings in the morning, never after 17:00.",
    "Things slipped a little on the move.",
    "Progress on the website: halfway through.",
    "A small change to the schedule, nothing big.",
    "I worked a bit on the slides today.",
]


async def main(key_var: str):
    ask = gemini_asker(key_from_env_file(key_var))
    first, second = await ask(TEXTS), await ask(TEXTS)
    spread = max(abs(first[t][0] - second[t][0]) for t in TEXTS)
    lines = [json.dumps({"note": f"Frozen {date.today()} on {MODEL} with pin_judge.QUESTION; replayed before each "
                                 f"fold, a move over the tolerance sends the run back to the rule. Asked twice, "
                                 f"largest difference {spread:.6f}."})]
    lines += [json.dumps({"text": t, "p": round(first[t][0], 6)}) for t in TEXTS]
    PROBES.write_text("\n".join(lines) + "\n")
    for t in TEXTS:
        print(f"{first[t][0]:.4f}  {second[t][0]:.4f}  {t}")
    print("largest difference between the two calls:", f"{spread:.6f}")
    assert QUESTION  # the frozen values belong to this question


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--key-var", default="GEMINI_API_KEY_2")
    asyncio.run(main(ap.parse_args().key_var))
