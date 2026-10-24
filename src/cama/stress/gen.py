"""Synthetic long scenarios for stress profiles, deterministic from a seed.

`zipf`: topics recur with a Zipf law, as real usage does (a few topics often, many rarely).
`churn`: a different topic every turn, the worst case for the active set and the prefix cache.
Every topic has a seed block stored in L3 only, so a first mention is a page fault.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SUBJECTS = [
    ("trip-abidjan", "Trip to Abidjan", "departure 8 December, return 20 December, hotel on the Plateau"),
    ("devfest-talk", "DevFest talk", "40 minute talk on offline agents, slides due 1 December"),
    ("research-cache", "Cache-aware memory research", "measure cost and latency of harness caches"),
    ("phone-tests", "Pixel 4 XL tests", "Termux, llama.cpp, pin to A55 cores for a low-end profile"),
    ("funding", "Research funding", "French Tech Emergence, i-Lab, CIFRE thesis"),
    ("half-marathon", "Half marathon plan", "Nantes in April, under 1h50, long run on Sundays"),
    ("taxes", "Tax return", "conference income as non commercial profits, keep travel receipts"),
    ("recipe-attieke", "Attieke recipe", "grilled fish, onions, tomatoes, two hours of marinade"),
    ("community", "Developer community", "GDG Cloud Abidjan meetups, speakers for the next season"),
    ("hiring", "Hiring an intern", "Python and data, six months, start in February"),
    ("car", "Car maintenance", "service due at 60000 km, winter tyres in November"),
    ("apartment", "Apartment move", "lease ends in March, boxes, moving company quotes"),
    ("reading", "Reading list", "MemGPT, vLLM PagedAttention, LongMemEval, LoCoMo"),
    ("health", "Health check", "annual check up booked, blood test fasting"),
    ("family", "Family visit", "parents arriving in Nantes on 2 January for two weeks"),
    ("budget", "Monthly budget", "rent, savings goal 500 euros, cloud bill under 30 euros"),
    ("course", "Online course", "distributed systems course, lecture 6 on consensus"),
    ("garden", "Balcony garden", "tomatoes and basil, water every two days in summer"),
    ("guitar", "Guitar practice", "fingerpicking, 20 minutes a day, song for the wedding"),
    ("wedding", "Friend's wedding", "Cotonou in June, best man speech, flights to book"),
    ("visa", "Visa paperwork", "e-visa for Cote d'Ivoire, passport valid until 2031"),
    ("cloud-bill", "Cloud bill", "BigQuery scans, set a budget alert at 20 euros"),
    ("paper", "Paper draft", "workshop deadline in January, related work section first"),
    ("patent", "Patent filing", "INPI provisional before any publication, 13 euros"),
    ("laptop", "Laptop upgrade", "MacBook with 16 GB RAM, keep the M2 for tests"),
    ("language", "Learning Swahili", "ten words a day, Akili and Urithi vocabulary"),
    ("newsletter", "Newsletter", "monthly issue on agent memory, 400 subscribers"),
    ("dentist", "Dentist", "appointment on 14 November at 9 am"),
    ("gift", "Birthday gift", "sister's birthday on 30 November, a book and headphones"),
    ("conference-cfp", "Conference CFP", "submit to the cloud summit, abstract of 300 words"),
]

ASKS = ["What did I note about {t}?", "Remind me where I am with {t}.", "Any update needed on {t}?",
        "Summarise {t} in two lines.", "What is the next step for {t}?"]
TELLS = ["Note for {t}: done the first part today.", "New fact for {t}: it moved by one week.",
         "Update {t}: I added a constraint about the budget."]


def generate(kind: str, turns: int, seed: int = 20261023, zipf_s: float = 1.1) -> list[dict]:
    rng = random.Random(seed)
    n = len(SUBJECTS)
    weights = [1 / (r + 1) ** zipf_s for r in range(n)]
    seed_blocks = [{"id": "profile", "kind": "fondamental", "pinned": True, "title": "User profile",
                    "content": "- Applied AI engineer between Nantes and Abidjan.\n- Prefers short answers."}]
    for bid, title, content in SUBJECTS:
        seed_blocks.append({"id": bid, "kind": "projet", "active": False, "title": title, "content": f"- {content}."})
    steps: list[dict] = [{"seed": seed_blocks}]
    for i in range(turns):
        k = i % n if kind == "churn" else rng.choices(range(n), weights=weights)[0]
        title = SUBJECTS[k][1].lower()
        template = rng.choice(TELLS if rng.random() < 0.25 else ASKS)
        steps.append({"user": template.format(t=title)})
    return steps


def write(kind: str, turns: int, out_dir: Path, seed: int = 20261023) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{kind}_{turns}_s{seed}.jsonl"
    path.write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in generate(kind, turns, seed)) + "\n")
    return path
