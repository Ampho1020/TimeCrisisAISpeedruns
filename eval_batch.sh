#!/usr/bin/env bash
# Run N back-to-back eval episodes, each with a fresh listener + BizHawk, appending to a pooled JSONL.
#   ./eval_batch.sh <tag> <episodes> [run_eval.py args...]
# e.g. ./eval_batch.sh base 5 theta_interrupt.npy --dump-frames-root captures/base
# Pass --dump-frames-root DIR to also capture each episode's frames into DIR/epK.
set -u
cd "$(dirname "$0")"
TAG=$1; N=$2; shift 2
OUT=${EVAL_OUT:-evals.jsonl}
BIZ=/home/ampho/Downloads/BizHawk/BizHawk-2.11.1-linux-x64
ROM="/home/ampho/Downloads/TimeCrisis_NTSC/Time Crisis.cue"
ARGS=(); DUMP_ROOT=""
while [ $# -gt 0 ]; do
  if [ "$1" = "--dump-frames-root" ]; then DUMP_ROOT=$2; shift 2; else ARGS+=("$1"); shift; fi
done
mkdir -p logs
for i in $(seq 1 "$N"); do
  pkill -f EmuHawk 2>/dev/null; sleep 2
  LOG=logs/eval_${TAG}_$i.log
  EXTRA=()
  [ -n "$DUMP_ROOT" ] && EXTRA=(--dump-frames "$DUMP_ROOT/ep$i")
  DISPLAY=:0 PYTHONPATH=. .venv/bin/python run_eval.py "${ARGS[@]}" "${EXTRA[@]}" \
    --shot-diag-out "$OUT" --tag "$TAG" --speed "${EVAL_SPEED:-3200}" > "$LOG" 2>&1 &
  PY=$!
  until grep -q "listening" "$LOG" 2>/dev/null || ! kill -0 $PY 2>/dev/null; do sleep 1; done
  (cd "$BIZ" && DISPLAY=:0 nohup ./EmuHawkMono.sh "$ROM" --socket_ip=127.0.0.1 --socket_port=8765 \
     --lua=/home/ampho/TimeCrisisAISpeedruns/bizhawk_bridge.lua > /dev/null 2>&1 &)
  wait $PY
  echo "[$TAG $i/$N] $(grep -E '^accuracy|^screens|^cleared' "$LOG" | tr '\n' ' ')"
done
pkill -f EmuHawk 2>/dev/null
.venv/bin/python pool_eval.py "$OUT" 2>/dev/null | tail -n 40
