# Deployment & Performance Testing

Documents the cloud GPU deployment setup and load test results comparing on-GPU vs remote (local → cloud) latency.

---

## Infrastructure

**Provider:** RunPod
**GPU:** RTX 4090 24GB
**Services:**
- `api.py` — FastAPI REST server on port 8000
- `serve.py` — Gradio demo UI on port 7860

RunPod exposes each port via a public HTTPS proxy:
```
https://{pod-id}-{port}.proxy.runpod.net/
```
Direct port access (e.g. `ip:8000`) is firewalled. All external traffic goes through the RunPod proxy. SSH remains the only direct connection channel (used by VSCode Remote).

**Starting the API:**
```bash
# Reuse pre-built FAISS index (faster startup)
LOAD_INDEX=1 uvicorn api:app --host 0.0.0.0 --port 8000

# Gradio UI with auth
python serve.py --load-index --username <user> --password <pass>
```

---

## Load Testing

**Tool:** [Locust](https://locust.io/) (`locustfile.py`)
**Config:** 5 concurrent users, 1 user/s ramp-up, 60s duration
**Question types:**
- `bridge` — 2-hop reasoning (classify → decompose → retrieve × 2 → answer → verify)
- `comparison` — yes/no entity comparison (classify → rewrite → retrieve → answer → verify)
- `random` — sampled from full pool

### Test 1 — On-GPU (locust running on the GPU server itself)

Baseline: measures pure inference latency with no network overhead.

```
locust -f locustfile.py --host http://localhost:8000 \
    --headless -u 5 -r 1 --run-time 60s
```

| Type | p50 | p75 | p95 | p99 | # reqs |
|------|-----|-----|-----|-----|--------|
| bridge | 7600ms | 9400ms | 12000ms | 14000ms | 52 |
| comparison | 6300ms | 8800ms | 12000ms | 15000ms | 63 |
| random | 6100ms | 9600ms | 13000ms | 13000ms | 18 |
| /health | 2ms | 3ms | 5ms | 17ms | 25 |
| **Aggregated** | **6100ms** | **8500ms** | **12000ms** | **14000ms** | **158** |

Errors: 1 × `HTTP 0` (isolated connection reset, <1%).

### Test 2 — Remote (locust running on local MacBook → RunPod proxy)

Measures end-to-end latency as experienced by an external user.

```
locust -f locustfile.py --host https://{pod-id}-8000.proxy.runpod.net \
    --headless -u 5 -r 1 --run-time 60s
```

| Type | p50 | p75 | p95 | p99 | # reqs |
|------|-----|-----|-----|-----|--------|
| bridge | 9600ms | 11000ms | 16000ms | 18000ms | 42 |
| comparison | 6900ms | 8600ms | 13000ms | 14000ms | 57 |
| random | 6700ms | 8500ms | 13000ms | 16000ms | 25 |
| /health | 940ms | 1300ms | 2500ms | 2500ms | 18 |
| **Aggregated** | **7100ms** | **9500ms** | **14000ms** | **16000ms** | **142** |

Errors: 0.

---

## Analysis

### Network overhead

`/health` latency jumps from 2ms (on-GPU) to 940ms (remote), establishing the baseline network + RunPod proxy cost at roughly **~1 second**.

| Metric | On-GPU | Remote | Delta |
|--------|--------|--------|-------|
| /health p50 | 2ms | 940ms | +938ms |
| Aggregated p50 | 6100ms | 7100ms | +1000ms |
| Aggregated p95 | 12000ms | 14000ms | +2000ms |

### Key findings

- **Network overhead is ~1s** (flat across percentiles), which is the RunPod proxy round-trip from Europe/US.
- **Inference dominates latency.** Network adds ~15% to p50 — GPU inference is the real bottleneck.
- **Bridge questions are slowest** (p50 +700ms vs comparison) due to two retrieval hops and more LLM calls.
- **Tail latency (p99) increases ~2s** over baseline, consistent with the fixed network overhead.
- **Zero errors in remote test**, confirming the RunPod proxy is stable under this load.

### Bottleneck

With BGE retrieval being fast, the dominant cost is LLM inference (Qwen2.5-3B on A100). Each `/ask` request triggers 3–5 model forward passes depending on question type and whether verify triggers a retry. Batching or speculative decoding would be the next lever for latency reduction.
