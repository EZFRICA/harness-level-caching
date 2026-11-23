"""Which folded messages should H10 pin? The hand-written rule (states_a_fact) against a local
classifier on EmbeddingGemma 2 vectors, every part of it versioned in this repository.

    uv run python scripts/eval_fact_classifier.py [--json out.json]

Training: scenarios/labels/statement_train_v1.jsonl, written by hand, independent of the test
scenario (none of its 24 facts, none of its filler templates). Test: the distinct user messages of
scenarios/long_facts_en_1500_v1.jsonl, where the positives are the 24 planted statements and their
6 corrections, taken from scripts/gen_long_facts.py, and everything else is negative. No network,
no paid call; the model is read from models/embeddinggemma-2-onnx.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gen_long_facts import FACTS  # noqa: E402

from harness_cache.adapters.gemma import GemmaEmbedder  # noqa: E402
from harness_cache.engine import states_a_fact  # noqa: E402
from harness_cache.tokens import approx_tokens  # noqa: E402

THRESHOLD = 0.5
L2 = 1e-2


def planted() -> set[str]:
    out = set()
    for _, stmt, _, v1, v2 in FACTS:
        out.add(stmt.format(v=v1))
        if v2:
            s = stmt.format(v=v2)
            out.add(f"Correction: {s[0].lower() + s[1:]}")
    return out


def fit(x: np.ndarray, y: np.ndarray, steps: int = 3000, lr: float = 0.5) -> tuple[np.ndarray, float]:
    """Logistic regression by gradient descent, L2 on the weights; deterministic."""
    w, b = np.zeros(x.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(x @ w + b)))
        g = p - y
        w -= lr * (x.T @ g / len(y) + L2 * w)
        b -= lr * g.mean()
    return w, b


def prob(x, w, b):
    return 1 / (1 + np.exp(-(x @ w + b)))


def cross_val(x, y, k=5):
    idx = np.random.default_rng(20261117).permutation(len(y))
    correct = 0
    for f in range(k):
        test = idx[f::k]
        train = np.setdiff1d(idx, test)
        w, b = fit(x[train], y[train])
        correct += int(((prob(x[test], w, b) >= THRESHOLD) == y[test]).sum())
    return correct / len(y)


def score(name, flagged: set[str], positives: set[str], counts: Counter):
    kept = positives & flagged
    noise = flagged - positives
    return {"method": name, "facts_kept": len(kept), "facts": len(positives),
            "missed": sorted(positives - flagged), "noise_distinct": len(noise),
            "noise_occurrences": sum(counts[m] for m in noise),
            "pinned_distinct": len(flagged), "pinned_tokens_est": sum(approx_tokens(m) for m in flagged),
            "noise_examples": sorted(noise)[:6]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in (ROOT / "scenarios/labels/statement_train_v1.jsonl").read_text().splitlines()
            if l.strip() and '"label"' in l]
    msgs = [json.loads(l)["user"] for l in (ROOT / "scenarios/long_facts_en_1500_v1.jsonl").read_text().splitlines()
            if '"user"' in l]
    counts = Counter(msgs)
    distinct = sorted(counts)
    positives = planted() & set(distinct)
    assert len(positives) == 30, len(positives)
    train_texts = [r["text"] for r in rows]
    assert not set(train_texts) & set(distinct), "a training text appears in the test scenario"

    emb = GemmaEmbedder(str(ROOT / "models" / "embeddinggemma-2-onnx"), dim=768)
    x_train = emb._embed(train_texts, "document")
    y_train = np.array([r["label"] for r in rows], float)
    t0 = time.perf_counter()
    x_test = emb._embed(distinct, "document")
    ms_per_message = (time.perf_counter() - t0) * 1000 / len(distinct)

    cv = cross_val(x_train, y_train)
    w, b = fit(x_train, y_train)
    p = prob(x_test, w, b)
    clf = {m for m, q in zip(distinct, p) if q >= THRESHOLD}
    rule = {m for m in distinct if states_a_fact(m)}
    result = {
        "model": emb.name, "revision": "onnx-community/embeddinggemma-2-ONNX daa72c51243991dfcaf9f9137d2c573d8f7790c0",
        "train": {"examples": len(rows), "positives": int(y_train.sum()), "cross_validated_accuracy": round(cv, 3)},
        "test": {"distinct_messages": len(distinct), "turns": len(msgs), "positives": len(positives)},
        "classifier_ms_per_message": round(ms_per_message, 1), "threshold": THRESHOLD,
        "results": [score("rule states_a_fact", rule, positives, counts),
                    score("classifier on EmbeddingGemma 2", clf, positives, counts)],
        "lowest_positive_probability": round(float(min(q for m, q in zip(distinct, p) if m in positives)), 3),
        "highest_negative_probability": round(float(max(q for m, q in zip(distinct, p) if m not in positives)), 3),
    }
    print(json.dumps(result, indent=1, ensure_ascii=False))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
