# Harness-Level Caching for LLM Agents

Code name `cama`. Private research code. Nothing here is published before a patent filing.

A measurement study: what an LLM agent gains, in cost and latency, from caching in its own
harness (memory routing, embeddings, tool results, the prompt prefix sent to the provider),
and where it breaks. The deliverable is measured trade-offs and limits, not a product.

Where to read: design in `docs/architecture.md`, choices in `docs/decisions/`, results in
`docs/measured/` (generated from frozen runs), what does not work in `docs/limits.md`,
existing work in `docs/prior-art.md`, every action in `docs/ledger.jsonl` (rendered to
`docs/journal/`).

## Main results so far

- `docs/measured/decisive.md` and `decisive-urithi.md`: over 500 turns a history folded into
  a running summary (`compact`) recalls every fact and stays the cheapest arm; an extracted
  memory with a stable render order catches up on recall only with apu-urithi's routing.
- `docs/measured/probes/`: Gemini's implicit cache gave 0 hits on 2026-10-27 on both the
  Gemini API and Vertex AI; provider caching cannot be assumed, it has to be measured.
- `docs/measured/web.md`: an exact-key tool cache rarely hits on model-written web queries.

## Run it

```bash
cp .env.example .env                       # set the keys of the providers you use
CAMA_CODE_SHA=$(scripts/code_id.sh) docker compose up -d --build
```

The local embedding model (`qdrant/paraphrase-multilingual-MiniLM-L12-v2-onnx-Q`) is mounted
from `models/`, which is not in the repository: place the Hugging Face cache there. Limits of
the container: `CAMA_CPUS` and `CAMA_MEM` (default 1 CPU and 1 GB).

```bash
docker compose exec -it cama cama chat                            # interactive
docker compose exec cama cama run /scenarios/assistant_fr_v1.jsonl --fresh --mode cama
docker compose exec cama cama report /runs/<run_id>
```

Modes: `naive`, `compact`, `topk`, `cama`. Each figure page in `docs/measured/` gives the exact
command and switches that reproduce it. Long-session scenarios: `scripts/gen_long_facts.py`.

Dashboard: http://localhost:8797 (read only, live): overview, live run, A/B comparison with
verdicts, costs with price provenance, cluster, stress, and the ledger.

Cluster (Postgres and three workers): `./scripts/cluster_fresh.sh <run id>`, then
`docker compose exec cama cama load --workers http://cama-w1:8797,http://cama-w2:8797,http://cama-w3:8797`.
Stress profiles: `scripts/stress_scale.sh`, `scripts/fault_facts.py`, skill `run-a-stress`.

## Trace and figures

```bash
uv run pytest -q                           # offline, no key, no network
uv run cama ledger add --kind action --title "..." --why "..." --result done
uv run cama ledger check                   # the hash chain holds
uv run cama figures snapshot runs/<run_id> # freeze a run into docs/measured/runs
uv run cama figures write                  # regenerate the figure pages
uv run cama decisions                      # every decision cites frozen runs and real tests
```

## Volumes

| volume | holds |
|---|---|
| `cama-data` at `/data` | SQLite stores, one per experiment with `--fresh` |
| `cama-archive` at `/archive` | the L4 replica fed by the cold path |
| `./runs` at `/runs` | manifest, turns, tasks and log of every run |
| `./scenarios` at `/scenarios` | scripted conversations, read only |
| `./models` at `/models` | the local ONNX embedding model, read only |
