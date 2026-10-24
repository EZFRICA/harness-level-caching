#!/usr/bin/env bash
# S3 (agents per worker) and S6 (workers) on a fresh cluster each time, echo model with the fitted
# latency model. Every run lands in runs/<id> with its load summary; freeze them afterwards with
#   uv run cama figures snapshot runs/<id>
set -euo pipefail
cd "$(dirname "$0")/.."
W1=http://cama-w1:8797; W2=http://cama-w2:8797; W3=http://cama-w3:8797
load() {  # id workers agents turns
  ./scripts/cluster_fresh.sh "$1" >/dev/null
  docker compose exec -T cama cama load --workers "$2" --agents "$3" --concurrency "$3" --turns "$4" --run-id "$1" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);print('$1', {k:d[k] for k in ('ok','errors','throughput_rps','p50_ms','p95_ms')})"
  echo '{"fault": "none"}' > "runs/$1/facts.json"
}
for n in 1 4 16 32; do load "cluster-s3-1w-${n}a-v2" "$W1" "$n" 12; done
for spec in "1:$W1" "2:$W1,$W2" "3:$W1,$W2,$W3"; do
  w=${spec%%:*}; urls=${spec#*:}
  load "cluster-s6-${w}w-18a-v2" "$urls" 18 12
done
