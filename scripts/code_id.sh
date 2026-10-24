#!/usr/bin/env bash
# Identity of the code a run used: the commit, plus a hash of src/ when the tree has uncommitted
# changes, so two dirty builds can never share an id (a stress series once mixed two versions).
cd "$(dirname "$0")/.."
sha=$(git rev-parse --short HEAD)
if [ -n "$(git status --porcelain -- src pyproject.toml)" ]; then
  h=$(find src pyproject.toml -type f \( -name '*.py' -o -name '*.js' -o -name '*.css' -o -name '*.html' -o -name '*.toml' \) -print0 | sort -z | xargs -0 shasum | shasum | cut -c1-8)
  echo "${sha}-dirty-${h}"
else
  echo "$sha"
fi
