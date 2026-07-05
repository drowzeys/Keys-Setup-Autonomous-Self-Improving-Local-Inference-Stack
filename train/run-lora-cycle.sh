#!/usr/bin/env bash
# One self-improvement cycle Hermes can invoke (or cron nightly):
#   mine Hermes' own log -> if enough new pairs, LoRA-train Gemma -> eval-gate -> stage adapter.
# Hermes triggers this; it is how the stack self-trains from its own routing data.
set -euo pipefail
cd "$(dirname "$0")"
MIN_PAIRS="${MIN_PAIRS:-50}"          # don't train on noise; wait for a real batch
IMAGE="${IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate:latest}"
STAMP="${STAMP:-cycle}"               # pass a date/tag in (Date.now() unavailable here)

echo "[cycle] 1/3 mine signal from Hermes history"
python3 mine_signal.py --out train_pairs.jsonl
N=$(wc -l < train_pairs.jsonl || echo 0)
echo "[cycle] mined $N pairs (threshold $MIN_PAIRS)"
if [ "$N" -lt "$MIN_PAIRS" ]; then
  echo "[cycle] below threshold — skip training this cycle (accumulate more log). OK."
  exit 0
fi

echo "[cycle] 2/3 LoRA-train Gemma-4-12B in container"
docker run --rm --gpus all --network host --shm-size=16g \
  -v "$HOME/models-gemma4-12b-it:/model:ro" \
  -v "$PWD:/work" -w /work --entrypoint bash "$IMAGE" -c '
    pip install -q peft trl datasets accelerate 2>/dev/null || true
    python3 gemma_lora_train.py --model /model --data train_pairs.jsonl \
      --out "adapters/gemma-moa-'"$STAMP"'"'

echo "[cycle] 3/3 eval-gate + stage (manual gate for now)"
echo "  adapter at train/adapters/gemma-moa-$STAMP"
echo "  GATE before hot-swap: run held-out eval; only promote if >= current adapter."
echo "  hot-swap: serve Gemma+adapter (or merge into the target specialist) and repoint the router."
echo "[cycle] done."
