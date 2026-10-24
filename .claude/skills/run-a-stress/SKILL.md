---
name: run-a-stress
description: How CAMA is pushed until it breaks (long and churning generated scenarios, many agents on the cluster, a killed worker, a stopped Postgres, resource squeeze) and how the observed limits are recorded with RPO and RTO computed from raw files. Use when asked whether CAMA scales or survives faults, after a change to the scheduler, the cluster or a cache level, or to find the next limit.
when_to_use: Running a stress profile; killing a worker or stopping Postgres; reading a breaking point; writing a stress figure.
---

# Running a stress

The question is never "does it hold" but "where does it stop, and what does CAMA still add
there". Every limit found so far came from a run, not from reading code (ledger, fault events).

## Profiles and how to run them

| id | how |
|---|---|
| S1 long session | `uv run cama gen --kind zipf --turns 200`, then `cama run` in the three modes |
| S2 churn | `uv run cama gen --kind churn --turns 200`, same |
| S3 many agents | `cama load --agents N --concurrency N` against one worker, N from 1 to 64 |
| S4 squeeze | `CAMA_CPUS=0.5 CAMA_MEM=512m`; the ONNX model alone is about 700 MB |
| S5 faults | `./scripts/cluster_fresh.sh <id>`, a load in the background, then `docker kill -s KILL cama-w2` or `docker stop cama-postgres`; `CAMA_FAULT_LLM_ERROR_RATE`, `CAMA_FAULT_TOOL_ERROR_RATE` |
| S6 scale | the same load against 1, 2 then 3 workers |

## Rules learnt the hard way

- **Start the cluster fresh** (`scripts/cluster_fresh.sh`): worker volumes outlive a run, and a
  worker once trusted a previous run's store.
- **Pass flags so the shell cannot eat them**: zsh does not split `$var`; two ablations ran
  without their flag. Check `config` in each manifest before reading a result.
- **Measure, then compute**: RTO and RPO come from `scripts/fault_facts.py` reading
  `load.jsonl`, the fault timestamp and the dead worker's volume, never from memory.
- **The echo model is for load**, with its latency model (about 1 s per call, fitted on real
  calls with a weak fit); real models run small, under `CAMA_MAX_USD`.
- **A faster run is suspect** until its cold path is checked: a starved cold path looks fast.
- **One code version per series**: `cluster_fresh.sh` rebuilds the image, so editing `src/`
  during a series mixes versions. `scripts/code_id.sh` puts a hash of `src/` in every manifest;
  stop the series before changing code and rerun it under a new id.

## Recording

Freeze each run (`uv run cama figures snapshot runs/<id>`, load runs included), declare it in
`docs/measured/figures.toml` with `kind = "load"` for cluster runs, generate, and add a
`fault` or `measurement` event to the ledger. A limit that calls for a design change gets a
decision record.
