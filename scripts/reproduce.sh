#!/usr/bin/env bash
# Reproduce the memory comparison of the study on DeepSeek-V4-Flash at Nebius, one arm at a time.
#
#   scripts/reproduce.sh setup                 fetch the embedding model, build and start the container
#   scripts/reproduce.sh <arm> [scenario]      run one arm, then print its report
#
# Arms, with the settings of the frozen runs in docs/measured/decisive-1500.md:
#   naive               the whole history on every call
#   summary             history folded into a running summary past 6000 tokens (mode compact)
#   verbatim            summary, plus the user's own past messages recalled at the end of the prompt
#   extract             extracted memory, decision 0013 settings, apu-urithi's routing
#   extract-statements  the same, notes written only for messages that state a fact
#
# Scenario: /scenarios/long_facts_en_1500_v1.jsonl by default (1500 turns, 24 facts, 30 to 60 min,
# 0.13 to 1.2 USD). For a first try: /scenarios/generated/smoke_facts_100_v1.jsonl (100 turns,
# 6 facts, a few minutes, about 0.02 USD; the summary then folds past 2000 tokens instead of 6000,
# or it would never fold in 100 turns). Needs NEBIUS_API_KEY in .env; spending is capped per run.
set -euo pipefail
cd "$(dirname "$0")/.."

arm=${1:?usage: scripts/reproduce.sh setup | naive | summary | verbatim | extract | extract-statements [scenario]}

if [ "$arm" = setup ]; then
  [ -f .env ] || { cp .env.example .env; echo "created .env: set NEBIUS_API_KEY in it, then run setup again"; exit 1; }
  if [ ! -d models/models--qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q ]; then
    uv run --extra local python -c "from fastembed import TextEmbedding; \
TextEmbedding('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', cache_dir='models')"
  fi
  HC_CODE_SHA=$(scripts/code_id.sh) docker compose up -d --build
  echo "ready: dashboard on http://localhost:8797"
  exit 0
fi

scenario=${2:-/scenarios/long_facts_en_1500_v1.jsonl}
run_id="repro-$arm-$(basename "$scenario" .jsonl)-$(date +%Y%m%d-%H%M%S)"
provider=(-e HC_EMBEDDER=onnx -e HC_LLM=openai_compat -e HC_OAI_KEY_ENV=NEBIUS_API_KEY
          -e HC_OAI_BASE_URL=https://api.tokenfactory.nebius.com/v1
          -e HC_MODEL=deepseek-ai/DeepSeek-V4-Flash-0731 -e HC_OAI_SALT="$run_id" -e HC_LOG_LEVEL=WARNING)
case $scenario in *smoke*) provider+=(-e HC_COMPACT_AT=2000) ;; esac
extract=(-e HC_EXTRACT_REPLACE=on -e HC_MEMORY_REFRESH=epoch -e HC_PAGE_IN_TAIL=on
         -e HC_WINDOW_EPOCH=20 -e HC_ROUTE_POLICY=urithi)
case $arm in
  naive)              mode=naive;   cap=2.5; extra=() ;;
  summary)            mode=compact; cap=1.0; extra=() ;;
  verbatim)           mode=compact; cap=1.0; extra=(-e HC_VERBATIM_RECALL=on) ;;
  extract)            mode=extract; cap=1.0; extra=("${extract[@]}" -e HC_EXTRACT_EVERY=1) ;;
  extract-statements) mode=extract; cap=1.0
                      extra=("${extract[@]}" -e HC_EXTRACT_EVERY=10 -e HC_EXTRACT_POLICY=statements) ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

docker compose exec -T "${provider[@]}" -e HC_MAX_USD="$cap" ${extra[@]+"${extra[@]}"} \
  hc harness-cache run "$scenario" --fresh --mode "$mode" --run-id "$run_id" | tail -3
docker compose exec -T hc harness-cache report "/runs/$run_id"
