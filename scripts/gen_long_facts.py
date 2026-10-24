"""Generate the decisive long-session scenario: facts planted over hundreds of turns, some
corrected, each probed long after it was stated. Seeded, so the file can be regenerated.

Usage: uv run python scripts/gen_long_facts.py [turns] [seed] > scenarios/long_facts_en_v1.jsonl

Filler turns are the user lines of scenarios/generated/zipf_200_s20261023.jsonl, cycled. The seed
memory is empty, so every arm learns the facts from the conversation only. A probe's expected
value is the latest one stated before the probe.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

FACTS = [  # key, statement, question, value, corrected value (or None)
    ("locker", "My gym locker code is {v}.", "What is my gym locker code?", "4831", "7290"),
    ("manager", "My new manager is called {v}.", "What is my manager's name?", "Priya Raman", None),
    ("flight", "My flight to Lisbon is {v}.", "What is my flight number to Lisbon?", "TP 1438", "TP 1452"),
    ("dentist", "My dentist appointment is on {v}.", "When is my dentist appointment?", "March 14", None),
    ("bike", "The serial number of my bike is {v}.", "What is my bike's serial number?", "WTU-58219", None),
    ("cat", "My cat is named {v}.", "What is my cat's name?", "Biscotte", None),
    ("budget", "The research budget for the quarter is {v} euros.", "What is the quarterly research budget?", "12400", "9800"),
    ("wifi", "The office wifi password hint is {v}.", "What is the office wifi password hint?", "blue heron", None),
    ("allergy", "I am allergic to {v}.", "What am I allergic to?", "shellfish", None),
    ("room", "The workshop is in room {v}.", "Which room is the workshop in?", "B214", "C307"),
    ("sister", "My sister's birthday is on {v}.", "When is my sister's birthday?", "July 9", None),
    ("plate", "My car plate is {v}.", "What is my car plate?", "GH-417-KT", None),
    ("insurance", "My insurance policy number is {v}.", "What is my insurance policy number?", "POL-88213", None),
    ("coach", "My running coach is {v}.", "Who is my running coach?", "Malik Osei", None),
    ("deadline", "The paper deadline is {v}.", "When is the paper deadline?", "May 2", "May 16"),
    ("pin", "The printer PIN is {v}.", "What is the printer PIN?", "6604", None),
    ("hotel", "My hotel booking reference is {v}.", "What is my hotel booking reference?", "HX-39021", None),
    ("doctor", "My doctor is Dr {v}.", "What is my doctor's name?", "Lemaire", None),
    ("server", "The staging server is called {v}.", "What is the staging server called?", "kestrel", "osprey"),
    ("bus", "I take bus line {v} to work.", "Which bus line do I take to work?", "62", None),
    ("plant", "My office plant is a {v}.", "What kind of plant is in my office?", "monstera", None),
    ("parking", "My parking spot is {v}.", "What is my parking spot?", "P3-118", None),
    ("course", "I signed up for a course on {v}.", "What course did I sign up for?", "Japanese calligraphy", None),
    ("neighbour", "My neighbour's name is {v}.", "What is my neighbour's name?", "Ines Fadel", None),
]
GAP = 60  # turns at least between a statement and its probe


def generate(turns: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    root = Path(__file__).resolve().parents[1]
    filler = [json.loads(l)["user"] for l in (root / "scenarios/generated/zipf_200_s20261023.jsonl").read_text()
              .splitlines()[1:] if "user" in l]
    slots: dict[int, dict] = {}
    last_plant = int(turns * 0.55)
    for key, stmt, question, v1, v2 in FACTS:
        t0 = rng.randint(5, last_plant)
        while t0 in slots:
            t0 += 1
        slots[t0] = {"user": stmt.format(v=v1)}
        latest, latest_t = v1, t0
        if v2:
            t1 = rng.randint(t0 + 40, min(t0 + 160, turns - GAP - 20))
            while t1 in slots:
                t1 += 1
            slots[t1] = {"user": f"Correction: {stmt.format(v=v2)[0].lower() + stmt.format(v=v2)[1:]}"}
            latest, latest_t = v2, t1
        tp = rng.randint(latest_t + GAP, turns - 1)
        while tp in slots:
            tp += 1
        if tp < turns:
            slots[tp] = {"user": question, "expect": [latest]}
    out, f = [{"seed": []}], 0
    for t in range(turns):
        if t in slots:
            out.append(slots[t])
        else:
            out.append({"user": filler[f % len(filler)]})
            f += 1
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 20261025
    for line in generate(n, s):
        print(json.dumps(line, ensure_ascii=False))
