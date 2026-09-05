#!/usr/bin/env bash
# Qwen3.6-35B-A3B on a 6 GB RTX 3050 laptop with 16 GB RAM.
# Non-expert weights + KV on the GPU, expert weights on the host (mmap; the part that
# doesn't fit in RAM pages from NVMe), and a per-layer LRU cache of hot expert slices in
# the remaining VRAM (llama.cpp PR #27861, --moe-expert-cache).
#
# Layout (see README): $LLM_DIR/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
#                      $LLM_DIR/cuda-12.8/
#                      $LLM_DIR/llama.cpp-src/build-cuda/bin/llama-server
# LLM_DIR defaults to the directory this script lives in.
#
# Tunables (env):
#   CACHE    --moe-expert-cache slots per layer. 40 ≈ 2.3 GB VRAM (measured best on 6 GB). 0 = off.
#   INSERTS  --moe-expert-cache-inserts (expert uploads per layer per step). 2 and 4 measured identical.
#   NCM      --n-cpu-moe. 99 = every expert layer on the host. Pinning layers (lower values)
#            measured worse than spending the same VRAM on CACHE.
#   UB       -ub/-b micro-batch. 2048 → ~300 tok/s prefill vs ~180 at 512; costs ~280 MB VRAM.
#   THREADS  decode threads (4 = P-cores only, best on a 4P+4E i5-12450H).  TB  prefill threads (8).
#   CTX      context. KV at 32K q8_0 is ~0.4 GB on this architecture (only 10 of 40 layers hold KV).
#   LV       --log-verbosity: 3 = info, 5 = debug (prints "moe-cache: ... hit-rate=" every 512 steps).
#   PORT, EXTRA
set -euo pipefail
LLM_DIR="${LLM_DIR:-$(cd "$(dirname "$0")" && pwd)}"
MODEL="$LLM_DIR/models/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
BIN="$LLM_DIR/llama.cpp-src/build-cuda/bin/llama-server"
export LD_LIBRARY_PATH="$LLM_DIR/cuda-12.8/lib64:$LLM_DIR/llama.cpp-src/build-cuda/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

CACHE="${CACHE:-40}"
INSERTS="${INSERTS:-2}"
NCM="${NCM:-99}"
UB="${UB:-2048}"
THREADS="${THREADS:-4}"
TB="${TB:-8}"
CTX="${CTX:-32768}"
LV="${LV:-3}"
PORT="${PORT:-8091}"

CACHEARGS=()
[ "$CACHE" != 0 ] && CACHEARGS=(--moe-expert-cache "$CACHE" --moe-expert-cache-inserts "$INSERTS")

exec "$BIN" -m "$MODEL" \
  -ngl 99 --n-cpu-moe "$NCM" "${CACHEARGS[@]}" \
  -t "$THREADS" -tb "$TB" -c "$CTX" -ub "$UB" -b "$UB" -fa on -ctk q8_0 -ctv q8_0 \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 \
  --jinja -np 1 --log-verbosity "$LV" --host 127.0.0.1 --port "$PORT" ${EXTRA:-}
