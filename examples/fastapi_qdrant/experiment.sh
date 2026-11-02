#!/usr/bin/env bash
# Measure one pipeline change end to end: a baseline at the committed settings, then a run with the change.
#   bash experiment.sh CHUNK_SIZE=250 CHUNK_OVERLAP=40
#   bash experiment.sh TEMPERATURE=1.0
# Each NAME=VALUE replaces that setting at the top of rag.py. Needs `ollama serve`. Restores rag.py afterwards.
set -u
cd "$(dirname "$0")"
PY=../../.venv/Scripts/python
AG=../../.venv/Scripts/answergate
NAME=$(echo "$*" | tr ' =' '_-' | tr '[:upper:]' '[:lower:]')
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
$AG run -c answergate.yaml -l "baseline" --html out/baseline.html --md out/baseline.md
$AG baseline -c answergate.yaml
kill $APP; wait $APP 2>/dev/null
echo "baseline done in $(( $(date +%s) - start ))s"

for kv in "$@"; do
  sed -i "s/^${kv%%=*} = [^#]*/${kv%%=*} = ${kv#*=}  /" rag.py
done
serve "$NAME"
$AG run -c answergate.yaml -l "$NAME" --html "out/$NAME.html" --md "out/$NAME.md"
echo "exit $? · total $(( $(date +%s) - start ))s"
