#!/usr/bin/env bash
# Measure one pipeline change end to end: baseline at the committed settings, then the changed run.
#   bash experiment.sh 250 40      # CHUNK_SIZE=250, CHUNK_OVERLAP=40 against the committed baseline
# Needs `ollama serve` running. Restores rag.py afterwards.
set -u
cd "$(dirname "$0")"
PY=../../.venv/Scripts/python
RC=../../.venv/Scripts/ragcheck
mkdir -p out
cp rag.py out/rag.py.orig
trap 'cp out/rag.py.orig rag.py; kill $APP 2>/dev/null' EXIT

serve() {
  $PY -m uvicorn app:app --port 8000 > "out/app-$1.log" 2>&1 &
  APP=$!
  until curl -s localhost:8000/health > /dev/null; do sleep 2; done
  curl -s localhost:8000/health; echo
}

start=$(date +%s)
serve baseline
$RC run -c ragcheck.yaml -l "chunk-1000" --html out/baseline.html --md out/baseline.md
$RC baseline -c ragcheck.yaml
kill $APP; wait $APP 2>/dev/null
echo "baseline done in $(( $(date +%s) - start ))s"

sed -i "s/^CHUNK_SIZE = .*/CHUNK_SIZE = $1  # characters per chunk/; s/^CHUNK_OVERLAP = .*/CHUNK_OVERLAP = $2  # characters repeated between neighbouring chunks/" rag.py
serve "chunk-$1"
$RC run -c ragcheck.yaml -l "chunk-$1" --html "out/chunk-$1.html" --md "out/chunk-$1.md"
echo "exit $? · total $(( $(date +%s) - start ))s"
