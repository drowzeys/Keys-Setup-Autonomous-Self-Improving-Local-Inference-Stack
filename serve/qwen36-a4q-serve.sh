#!/usr/bin/env bash
# Qwen3.6-27B-NVFP4 (official nvidia weights) + A4Q native-fp4 attention on AEON vLLM 0.24.
# A4Q port image aeon-vllm-a4q:port (jethac FlashInfer 0.6.13 fork + patched flashinfer.py).
# Local head node spark-13b3 == r0. Serve on :8000, served-model-name qwen36-nvfp4-a4q.
set -euo pipefail

A4Q="${A4Q:-1}"                       # VLLM_NVFP4_A4Q (1=on, 0=A/B baseline)
MAXLEN="${MAXLEN:-262144}"            # 256K native (max_position_embeddings=262144, no rope_scaling)
UTIL="${UTIL:-0.6}"                   # user-requested gpu-mem-util 0.6 (single-node safe per OOM rules)
SEQS="${SEQS:-8}"
NAME="${NAME:-qwen36-aeon}"
PORT="${PORT:-8000}"
SNAP=/models/models--nvidia--Qwen3.6-27B-NVFP4/snapshots/0893e1606ff3d5f97a441f405d5fc541a6bdf404

~/gpu-clear.sh || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
mkdir -p ~/.cache/flashinfer

docker run -d --name "$NAME" --gpus all --network host --ipc host \
  --shm-size=16g \
  -v ~/models-qwen36:/models:ro \
  -v ~/.cache/flashinfer:/root/.cache/flashinfer \
  -e VLLM_NVFP4_A4Q="$A4Q" \
  -e VLLM_ATTENTION_BACKEND=FLASHINFER \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e VLLM_USE_FLASHINFER_SAMPLER=1 \
  --entrypoint vllm \
  aeon-vllm-a4q:port \
  serve "$SNAP" \
    --served-model-name qwen36-nvfp4-a4q \
    --trust-remote-code \
    --language-model-only \
    --quantization modelopt \
    --kv-cache-dtype nvfp4 \
    --default-chat-template-kwargs '{"enable_thinking":false}' \
    --gpu-memory-utilization "$UTIL" \
    --max-num-seqs "$SEQS" \
    --max-model-len "$MAXLEN" \
    --host 0.0.0.0 --port "$PORT"

echo "launched $NAME (A4Q=$A4Q util=$UTIL maxlen=$MAXLEN seqs=$SEQS); tail: docker logs -f $NAME"
