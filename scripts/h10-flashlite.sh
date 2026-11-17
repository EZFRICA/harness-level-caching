#!/usr/bin/env bash
# Paid H10 test on Gemini 3.5 Flash-Lite through the Gemini API, 500 turns, one arm at a time.
#
#   scripts/h10-flashlite.sh            run every arm not finished yet, in order, then stop
#   scripts/h10-flashlite.sh status     say which arms are finished
#   KEY_VAR=GEMINI_API_KEY_2 scripts/h10-flashlite.sh   read the key from another variable of .env
#
# An arm is finished when its manifest holds the probe scores (written at the end of a run) and
# it has every turn. A run cannot resume mid way (the summary lives in memory), so an unfinished
# arm is moved to runs/aborted/interrupted/ and started again from its first turn. Stopping with
# Ctrl+C loses at most the arm in progress. Each arm is capped at 2 USD (HC_MAX_USD).
set -euo pipefail
cd "$(dirname "$0")/.."

SCENARIO=/scenarios/long_facts_en_v1.jsonl
TURNS=$(grep -c '"user"' "scenarios/long_facts_en_v1.jsonl")
ARMS=("compact:off:all" "compact-pin:on:all" "compact-pin-rest:on:rest")
KEY_VAR=${KEY_VAR:-GEMINI_API_KEY}  # the variable's name only; its value is read inside the container
[[ "$KEY_VAR" =~ ^GEMINI_API_KEY(_[0-9]+)?$ ]] || { echo "KEY_VAR must name a GEMINI_API_KEY variable"; exit 2; }

finished() {
  local d="runs/$1"
  [ -f "$d/manifest.json" ] && [ -f "$d/turns.jsonl" ] &&
    python3 -c "import json,sys; sys.exit(0 if 'probes' in json.load(open('$d/manifest.json')) else 1)" &&
    [ "$(wc -l < "$d/turns.jsonl" | tr -d ' ')" -ge "$TURNS" ]
}

for arm in "${ARMS[@]}"; do
  IFS=: read -r name pin rest <<< "$arm"
  id="gemapi-g35lite-lf500-$name-v1"
  if finished "$id"; then echo "done     $id"; continue; fi
  if [ "${1:-}" = "status" ]; then echo "to run   $id"; continue; fi
  if [ -d "runs/$id" ]; then
    mkdir -p runs/aborted/interrupted
    mv "runs/$id" "runs/aborted/interrupted/$id-$(date +%Y%m%d%H%M%S)"
    echo "moved the unfinished $id to runs/aborted/interrupted/"
  fi
  echo "running  $id (key from $KEY_VAR)"
  docker compose exec -T -e HC_LLM=gemini -e HC_MODEL=gemini-3.5-flash-lite -e HC_EMBEDDER=onnx \
    -e HC_PREFIX_CACHE=implicit -e HC_MAX_USD=2 -e HC_LOG_LEVEL=WARNING \
    -e HC_VERBATIM_PIN="$pin" -e HC_PIN_SUMMARY="$rest" -e KEY_VAR="$KEY_VAR" \
    hc sh -c 'GEMINI_API_KEY="$(printenv "$KEY_VAR")" exec harness-cache run "$@"' sh \
    "$SCENARIO" --fresh --mode compact --run-id "$id" | tail -3
done
