# Harness-Level Caching for LLM Agents

Private research code. Nothing here is published before a patent filing. The code is the
`harness_cache` package; runs and ledger events recorded before 2026-10-28 carry its former
name, cama, and the mode `cama` is read as `extract`.

A measurement study: what an LLM agent gains, in cost and latency, from caching in its own
harness (memory routing, embeddings, tool results, the prompt prefix sent to the provider),
and where it breaks. The deliverable is measured trade-offs and limits, not a product.

Where to read: design in `docs/architecture.md`, choices in `docs/decisions/`, results in
`docs/measured/` (generated from frozen runs), what does not work in `docs/limits.md`,
existing work in `docs/prior-art.md`, every action in `docs/ledger.jsonl` (rendered to
`docs/journal/`).

## Main results so far

- `docs/measured/decisive-1500.md`: over 1500 turns on DeepSeek-V4-Flash the running summary
  recalls every fact and stays the cheapest arm; verbatim recall and the extracted memory
  with apu-urithi's routing also recall every fact, at a higher cost.
- `docs/measured/decisive.md` and `decisive-urithi.md`: over 500 turns a history folded into
  a running summary (`compact`) recalls every fact and stays the cheapest arm; an extracted
  memory with a stable render order catches up on recall only with apu-urithi's routing.
- `docs/measured/probes/`: Gemini's implicit cache gave 0 hits on 2026-10-27 on both the
  Gemini API and Vertex AI; provider caching cannot be assumed, it has to be measured.
- `docs/measured/web.md`: an exact-key tool cache rarely hits on model-written web queries.

## Install as a package

With access to this repository, pip installs the code from GitHub (`[local]` adds the ONNX
embedder; append `@<tag or commit>` to pin a version):

```bash
pip install "harness-cache[local] @ git+ssh://git@github.com/EZFRICA/harness-level-caching.git"
harness-cache --help
```

The price table is not part of the package: point `HC_PRICING` to a copy of `pricing.toml`,
or paid runs are refused.

## Run it

```bash
cp .env.example .env                       # set the keys of the providers you use
HC_CODE_SHA=$(scripts/code_id.sh) docker compose up -d --build
```

The local embedding model (`qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q`) is mounted
from `models/`, which is not in the repository: place the Hugging Face cache there. Limits of
the container: `HC_CPUS` and `HC_MEM` (default 1 CPU and 1 GB).

```bash
docker compose exec -it hc harness-cache chat                            # interactive
docker compose exec hc harness-cache run /scenarios/assistant_fr_v1.jsonl --fresh --mode extract
docker compose exec hc harness-cache report /runs/<run_id>
```

Modes: `naive` (whole history), `compact` (running summary, add `HC_VERBATIM_RECALL=on` for
verbatim recall), `topk` and `extract` (extracted memory, paged by a router). Each figure page in `docs/measured/` gives the exact
command and switches that reproduce it. Long-session scenarios: `scripts/gen_long_facts.py`.

Dashboard: http://localhost:8797 (read only, live): overview, live run, A/B comparison with
verdicts, costs with price provenance, cluster, stress, and the ledger.

Cluster (Postgres and three workers): `./scripts/cluster_fresh.sh <run id>`, then
`docker compose exec hc harness-cache load --workers http://hc-w1:8797,http://hc-w2:8797,http://hc-w3:8797`.
Stress profiles: `scripts/stress_scale.sh`, `scripts/fault_facts.py`, skill `run-a-stress`.

## Trace and figures

```bash
uv run pytest -q                           # offline, no key, no network
uv run harness-cache ledger add --kind action --title "..." --why "..." --result done
uv run harness-cache ledger check                   # the hash chain holds
uv run harness-cache figures snapshot runs/<run_id> # freeze a run into docs/measured/runs
uv run harness-cache figures write                  # regenerate the figure pages
uv run harness-cache decisions                      # every decision cites frozen runs and real tests
```

## Volumes

| volume | holds |
|---|---|
| `hc-data` at `/data` | SQLite stores, one per experiment with `--fresh` |
| `hc-archive` at `/archive` | the L4 replica fed by the cold path |
| `./runs` at `/runs` | manifest, turns, tasks and log of every run |
| `./scenarios` at `/scenarios` | scripted conversations, read only |
| `./models` at `/models` | the local ONNX embedding model, read only |
