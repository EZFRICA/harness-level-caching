# CAMA: Cache-Aware Memory Architecture

Private research code. Nothing here is published before a patent filing.

CAMA is the cache of the agent harness: routing, embeddings, tool results and the prompt
prefix, organised so that what is stable stays reusable across turns and steps. Design in
`docs/architecture.md`, choices in `docs/decisions/`, measurements in `docs/measured/`
(generated from frozen runs), every action in `docs/ledger.jsonl` (rendered to
`docs/journal/`).

## Run it

```bash
cp .env.example .env                       # set GEMINI_API_KEY for the online profile
CAMA_CODE_SHA=$(git rev-parse --short HEAD) docker compose up -d --build
```

Online profile: `CAMA_LLM=gemini CAMA_EMBEDDER=onnx docker compose up -d`. Modest device:
`CAMA_CPUS=0.5 CAMA_MEM=512m`. The local embedding model is mounted from `models/`.

```bash
docker compose exec -it cama cama chat                            # interactive
docker compose exec cama cama run /scenarios/assistant_fr_v1.jsonl --fresh --mode cama
docker compose exec cama cama report /runs/<run_id>
```

Dashboard: http://localhost:8797 (read only, live): overview, live run, A/B comparison with
verdicts, costs with price provenance, cluster, stress, and the ledger.

Cluster (Postgres and three workers): `./scripts/cluster_fresh.sh <run id>`, then
`docker compose exec cama cama load --workers http://cama-w1:8797,http://cama-w2:8797,http://cama-w3:8797`.
Stress profiles: `scripts/stress_scale.sh`, `scripts/fault_facts.py`, skill `run-a-stress`.
What does not work yet: `docs/limits.md`.

## Trace and figures

```bash
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
