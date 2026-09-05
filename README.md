# Qwen3.6-35B-A3B on a 6 GB RTX 3050 laptop with 16 GB RAM

Running an 18 GB Q4 MoE model as a local coding-agent backend on a laptop that can't hold it in RAM or VRAM.

Measured on an RTX 3050 Laptop 6 GB / i5-12450H / 16 GB DDR5, Ubuntu 22.04 (September 2026):

| | stock (Vulkan prebuilt, experts on CPU) | this setup |
|---|---|---|
| decode | 13-16 tok/s | **~25 tok/s** |
| prefill | — | **~300 tok/s** |
| context | 32K | 32K |
| VRAM used | 2.7 GB | 5.5 GB of 5.8 |

Three ingredients: a CUDA build of llama.cpp (instead of the Vulkan prebuilt), llama.cpp **PR #27861** — a
per-layer LRU cache that keeps recently-used expert weights in spare VRAM (hit rate ~59 % with 15 % of the experts
cached) — and flags from a measured sweep. No RAM upgrade, no smaller quant.

Files: `serve.sh` (launcher with the tuned defaults), `pi/models.json` and `pi/settings.json` (config for the [pi](https://pi.dev) coding agent).

## Requirements

- NVIDIA GPU with **6 GB** VRAM (RTX 3050 Laptop; the VRAM budget below is sized for 5.8 GB usable). Driver ≥ 570.
- **16 GB RAM** with ≥ 12 GB available while the model runs (close the browser). The model does not fit; the
  coldest ~4 GB of experts page from NVMe on every run — less free RAM means slower decode.
- ~40 GB free disk on one filesystem: model 18.2 GB, CUDA runfile 5.4 GB + toolkit 8.7 GB, llama.cpp build ~3 GB,
  plus ~10 GB of scratch during the toolkit install.
- Ubuntu 22.04 or 24.04: `sudo apt install -y build-essential cmake ninja-build git aria2 curl python3`

Everything lives under one directory; `serve.sh` finds its files relative to that directory:

```bash
export LLM_DIR=/path/with/40GB/free/llm
mkdir -p "$LLM_DIR"/{models,dl,tmp,logs} && cd "$LLM_DIR"
```

## 1. Model

Use the **MTP** repo. Both unsloth repos have a file with this exact name; only the MTP one (18,209,036,576 bytes)
includes the next-token-prediction layer, which is inert unless speculative decoding is enabled later.

```bash
cd "$LLM_DIR/models"
aria2c -x8 -s8 -c --file-allocation=none \
  "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf"
sha256sum Qwen3.6-35B-A3B-UD-IQ4_XS.gguf
# df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf
```

Stay on IQ4_XS. Any larger Q4 adds gigabytes that page from disk. Do not use `--no-mmap` or `--mlock`: with a model
larger than free RAM they fail to load or get OOM-killed.

## 2. CUDA 12.8 toolkit

There is no prebuilt Linux CUDA llama.cpp release, so a local build is required, and it needs a CUDA 12.x toolkit
(an Ubuntu-packaged `nvcc` 11.x is too old; CUDA 13.2 has reports of broken output on Qwen3.6).

```bash
cd "$LLM_DIR/dl"
aria2c -x8 -s8 -c --file-allocation=none \
  https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
chmod +x cuda_12.8.1_570.124.06_linux.run && ./cuda_12.8.1_570.124.06_linux.run --check
sudo env TMPDIR="$LLM_DIR/tmp" ./cuda_12.8.1_570.124.06_linux.run \
  --silent --toolkit --toolkitpath="$LLM_DIR/cuda-12.8" --no-man-page --override
"$LLM_DIR/cuda-12.8/bin/nvcc" --version | tail -1
```

Must run as root (as a normal user the silent installer exits 0 without installing). `TMPDIR` must point at a disk
with ≥ 10 GB free; the old `--tmpdir` flag no longer exists in 12.8. The installer also creates a `/usr/local/cuda`
symlink to the toolkit path. Takes ~5 minutes.

## 3. Build llama.cpp from PR #27861

```bash
cd "$LLM_DIR"
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp-src && cd llama.cpp-src
git fetch origin pull/27861/head:pr27861 && git checkout pr27861
git log -1 --format='%h %s'   # bccbacdb8 MoE expert cache: GPU-resident LRU cache for host-offloaded expert weights

export PATH="$LLM_DIR/cuda-12.8/bin:$PATH"
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_CUDA_COMPILER="$LLM_DIR/cuda-12.8/bin/nvcc" \
  -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=ON
cmake --build build-cuda --target llama-server -j8      # ~7 min

export LD_LIBRARY_PATH="$LLM_DIR/cuda-12.8/lib64:$LLM_DIR/llama.cpp-src/build-cuda/bin"
./build-cuda/bin/llama-server --list-devices           # CUDA0: ... RTX 3050 6GB Laptop GPU (5806 MiB, ...)
```

Commit `bccbacdb8` is what was benchmarked; `git checkout bccbacdb8` pins it if the PR head has moved. The PR is a
draft (not merged as of September 2026); if it has been merged since, build `master` and confirm
`llama-server --help | grep moe-expert-cache` still works. `-DCMAKE_CUDA_ARCHITECTURES=86` is the RTX 30 series.

## 4. Run

```bash
cp serve.sh "$LLM_DIR"/ && cd "$LLM_DIR"
./serve.sh > logs/server.log 2>&1 &
until curl -sf http://127.0.0.1:8091/health >/dev/null; do sleep 2; done
grep "MoE expert cache enabled" logs/server.log   # 40 layers x 40 slots, 2 inserts/step, 2324.5 MiB device memory
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # ~5450 MiB
```

First start reads 18 GB from disk (1-3 min); warm restarts take ~15 s. Smoke test:

```bash
curl -s http://127.0.0.1:8091/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages":[{"role":"user","content":"Reply with exactly: laptop ok"}],
  "max_tokens":200, "chat_template_kwargs":{"enable_thinking":false}}'
```

What the flags do (all overridable via env, see the header of `serve.sh`):

- `-ngl 99 --n-cpu-moe 99` — everything except expert weights on the GPU; experts on the host.
- `--moe-expert-cache 40` — 40 of 256 experts per layer cached in VRAM by recency (2.3 GB). Decode-only.
- `-np 1` — one slot. The default of 4 would allocate 4× KV; mandatory on 6 GB.
- `-fa on -ctk q8_0 -ctv q8_0 -c 32768` — 8-bit KV; only ~0.4 GB at 32K on this hybrid-attention architecture.
- `-ub 2048 -b 2048` — large micro-batch for prefill (~300 vs ~180 tok/s), ~280 MB VRAM.
- `-t 4 -tb 8` — decode on the 4 P-cores only; prefill on all 8 physical cores.
- `--jinja --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0` — Qwen3.6 thinking-mode template and sampling. Reasoning is
  returned as `reasoning_content`; tool calling works.

Stop with `kill "$(pgrep -f 'build-cuda/bin/llama-serve[r]')"` and wait for VRAM to drop before starting another
instance.

## 5. pi coding agent

```bash
npm install -g @mariozechner/pi-coding-agent
mkdir -p ~/.pi/agent && cp pi/models.json ~/.pi/agent/models.json   # or merge the provider into an existing file
```

Then `pi` in a project → `/model` → "Qwen3.6-35B-A3B (laptop)". The `compat` flags are required (llama.cpp rejects
the `developer` role and `reasoning_effort`); `thinkingFormat: "qwen-chat-template"` lets pi toggle thinking per
request. Keep `maxTokens` ≥ 8192 — thinking uses a few thousand tokens before the answer.

Web search for the agent: `pi install npm:pi-web-access` — works with no API keys out of the box. `pi/settings.json`
is the resulting settings file with this model as the default and the extension registered; copy it to
`~/.pi/agent/settings.json` if starting fresh. A self-hosted SearXNG can be pointed at via `searxngBaseUrl` in
`~/.pi/web-search.json` (optional; see the extension's README).

## 6. Tuning for a different VRAM/RAM

VRAM at 32K ctx, `-ub 2048`, cache 40: 5.45-5.5 GB. Budget: non-expert weights ~1.3 GB, KV + state ~0.4 GB, compute
buffers ~1 GB, cache = slots × 40 layers × 1.39 MiB (40 → 2.3 GB, 32 → 1.9 GB, 48 → 2.8 GB). Keep ≥ 300 MB free.

- Startup dies with `cudaMalloc failed: out of memory` → lower `CACHE` by 8, or `UB=1024`.
- More VRAM → raise `CACHE`, not pinned layers: at equal VRAM the cache beat static layer pinning (`--n-cpu-moe 34`)
  by 16 %, and a pin+cache mix was worse than pure cache. 48 slots on 6 GB fits but gains nothing.
- Less RAM → slower decode (more experts paged from disk); nothing else to change.
- Hit rate: `LV=5 ./serve.sh` logs `moe-cache: ... hit-rate=` every 512 tokens (expect ~58-60 % at 40 slots).

## Measured results

RTX 3050 Laptop 6 GB, i5-12450H, ~14 GB RAM available, 32K ctx, 3 fixed prompts × 512 tokens, warm:

| config | VRAM | decode tok/s |
|---|---|---|
| Vulkan prebuilt (b10448), experts on CPU | 2.7 GB | 16.4 |
| CUDA build, experts on CPU | 2.8 GB | 18.7 |
| CUDA, 6 expert layers pinned on GPU | 5.2 GB | 21.3 |
| CUDA, expert cache 24 / 32 / 40 slots | 4.3 / 4.7 / 5.2 GB | 22.4 / 23.8 / 24.7 |
| **cache 40 + `-t 4 -tb 8`** | 5.2 GB | **25.4** |
| cache 40 + `--spec-type ngram-mod` | 5.2 GB | 23.5 |
| cache 48 | 5.6 GB | 25.2 |

Prefill with cache 40: 180 tok/s at `-ub 512`, **307 tok/s at `-ub 2048`** (15.5K-token prompt); a 31.6K-token
prompt fit with no allocation errors. Decode immediately after a 15-30K prefill drops to 13-18 tok/s.

## Not covered

- MTP speculative decoding (`--spec-type draft-mtp`) on top of the cache: reported ~1.8× elsewhere on this model, but
  the PR's batched path crashes above 8 draft tokens (`--spec-draft-n-max 3` or lower) and the drafter costs ~0.9 GB
  VRAM. Untested here.
- A 32 GB RAM upgrade removes the disk paging entirely; people with the same GPU class and 32 GB report 27-36 tok/s.
