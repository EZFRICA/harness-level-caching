#!/usr/bin/env bash
# Start the cluster from empty shared and worker stores, so no run inherits another's state.
set -euo pipefail
docker compose --profile cluster rm -sf postgres hc-w1 hc-w2 hc-w3 >/dev/null 2>&1 || true
docker volume rm -f cama_pg-data cama_w1-data cama_w2-data cama_w3-data >/dev/null 2>&1 || true
HC_CLUSTER_RUN="${1:-cluster}" HC_CODE_SHA="$(./scripts/code_id.sh)" docker compose --profile cluster up -d --build >/dev/null 2>&1
for w in w1 w2 w3; do
  until docker compose exec -T hc python -c "import httpx,sys; httpx.get('http://hc-$w:8797/healthz').raise_for_status()" >/dev/null 2>&1; do sleep 1; done
done
echo "cluster fresh: postgres + w1 w2 w3 ready"
