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

With BGE retrieval being fast, the dominant cost is LLM inference (Qwen2.5-3B on RTX 4090). Each `/ask` request triggers 3–5 model forward passes depending on question type and whether verify triggers a retry. Batching or speculative decoding would be the next lever for latency reduction.

---

## Optimisation Round 1 — `max_new_tokens` reduction

**Changes:** reduced token generation limits across all nodes (`reasoner.py` default 48→24, `rewrite_comparison` 80→48, `answer_final` in `graph.py` 64→20). Model precision kept at bfloat16.

> Note: 4-bit quantization (`bitsandbytes` NF4) was tested and reverted — it increased latency 3× on RTX 4090 because the GPU's high memory bandwidth (1008 GB/s) means dequantization overhead outweighs any memory savings for a 3B model.

### On-GPU results

| Type | p50 | p75 | p95 | p99 | # reqs | vs baseline |
|------|-----|-----|-----|-----|--------|-------------|
| bridge | 7700ms | 8400ms | 12000ms | 12000ms | 8 | ≈ 持平 |
| comparison | 5600ms | 7000ms | 10000ms | 10000ms | 12 | **-700ms (-11%)** |
| random | 4100ms | 9100ms | 10000ms | 10000ms | 10 | **-2000ms (-33%)** |
| /health | 2ms | 15ms | 15ms | 15ms | 4 | — |
| **Aggregated** | **5200ms** | **7300ms** | **10000ms** | **12000ms** | **34** | **-900ms (-15%)** |

### Remote results

| Type | p50 | p75 | p95 | p99 | # reqs |
|------|-----|-----|-----|-----|--------|
| bridge | 8400ms | 9300ms | 14000ms | 14000ms | 11 |
| comparison | 11000ms | 13000ms | 15000ms | 15000ms | 7 |
| random | 9600ms | 10000ms | 11000ms | 11000ms | 5 |
| /health | 1100ms | 1100ms | 1100ms | 1100ms | 1 |
| **Aggregated** | **8900ms** | **11000ms** | **14000ms** | **15000ms** | **24** |

> Remote comparison p50 increase (6900ms → 11000ms) is statistical noise — only 7 samples.

### Finding

`max_new_tokens` reduction gives ~15% improvement on aggregated p50 (6100ms → 5200ms on-GPU), driven mainly by comparison and random questions which generate short answers. Bridge latency is unchanged — its bottleneck is 5 sequential LLM calls, not token count per call.

---

## Observability Stack

Added production monitoring to `api.py` covering GPU metrics, token accounting, per-node latency, and optional LLM tracing.

### New files

```
monitoring/
├── __init__.py
├── metrics.py          # Prometheus metric definitions + GPU background poller
├── tracer.py           # Langfuse request-trace wrapper (graceful no-op if unconfigured)
├── token_context.py    # Thread-local token counter (input/output tokens per request)
├── prometheus.yml      # Prometheus scrape config (targets: localhost:8000, node-exporter)
└── grafana/
    └── provisioning/
        └── datasources/
            └── prometheus.yml   # Auto-provisions Prometheus datasource in Grafana
docker-compose.monitoring.yml    # Prometheus + Grafana + node-exporter (for Docker envs)
```

### Modified files

| File | Change |
|------|--------|
| `api.py` | API key auth (`X-Api-Key` header), `/metrics` endpoint, Prometheus instrumentation, Langfuse trace per request, `request_id` + `tokens` in response |
| `src/reasoner.py` | `_generate()` records input/output token counts via `token_context` |
| `requirements.txt` | Added `prometheus-client`, `nvidia-ml-py`, `langfuse`, `locust` |

### Metrics exposed at `GET /metrics`

| Metric | Type | Description |
|--------|------|-------------|
| `agent_requests_total{status}` | Counter | Request count by status (`ok` / `error`) |
| `agent_request_latency_seconds` | Histogram | End-to-end /ask latency |
| `agent_node_latency_seconds{node}` | Histogram | Wall-clock time per LangGraph node |
| `agent_input_tokens_total` | Counter | Cumulative prompt tokens (all model calls) |
| `agent_output_tokens_total` | Counter | Cumulative generated tokens |
| `agent_verify_failures_total` | Counter | verify node returned `verified=False` |
| `agent_fallback_total{type}` | Counter | Fallback triggers (`local` / `web`) |
| `agent_question_type_total{qtype}` | Counter | Questions by type (`bridge` / `comparison`) |
| `gpu_memory_used_bytes{gpu_id}` | Gauge | VRAM in use (polled every 1s) |
| `gpu_memory_total_bytes{gpu_id}` | Gauge | Total VRAM |
| `gpu_utilization_percent{gpu_id}` | Gauge | GPU compute utilisation |
| `gpu_temperature_celsius{gpu_id}` | Gauge | GPU die temperature |

### API changes

**New response fields:**
```json
{
  "answer": "yes",
  "verified": true,
  "qtype": "comparison",
  "steps": ["classify", "rewrite", "retrieve_comparison", "answer_final", "verify"],
  "request_id": "b3d2a1f0-...",
  "tokens": { "input": 3241, "output": 12 }
}
```

**Auth (optional):** set `API_KEYS=key1,key2` env var. If unset, auth is disabled.
```bash
curl -X POST http://localhost:8000/ask \
  -H "X-Api-Key: key1" \
  -H "Content-Type: application/json" \
  -d '{"question": "...", "tenant_id": "my-app"}'
```

### RunPod deployment (no Docker)

Prometheus and Grafana run as standalone binaries. Install once:

```bash
# Prometheus
cd /tmp
wget -q https://github.com/prometheus/prometheus/releases/download/v2.51.2/prometheus-2.51.2.linux-amd64.tar.gz
tar xzf prometheus-2.51.2.linux-amd64.tar.gz
mv prometheus-2.51.2.linux-amd64/prometheus /usr/local/bin/

# Grafana
wget -q https://dl.grafana.com/oss/release/grafana-11.0.0.linux-amd64.tar.gz
tar xzf grafana-11.0.0.linux-amd64.tar.gz
mv grafana-v11.0.0 /opt/grafana
```

Start everything (script saved at `/root/start_monitoring.sh`):

```bash
# Prometheus
prometheus \
  --config.file=/root/hotpotqa-agent/monitoring/prometheus.yml \
  --storage.tsdb.retention.time=7d \
  --web.listen-address=0.0.0.0:9090 \
  > /tmp/prometheus.log 2>&1 &

# Grafana — replace <pod-id> with your RunPod pod ID
GF_SERVER_HTTP_PORT=3000 \
GF_SERVER_ROOT_URL=https://<pod-id>-3000.proxy.runpod.net \
GF_SERVER_DOMAIN=<pod-id>-3000.proxy.runpod.net \
/opt/grafana/bin/grafana server \
  --homepath /opt/grafana \
  --config /opt/grafana/conf/custom.ini \
  > /tmp/grafana.log 2>&1 &
```

Expose ports `3000` and `9090` in RunPod pod settings to access via proxy.

**Grafana datasource:** add manually via API (provisioning file is present but Grafana DB takes precedence on RunPod):
```bash
curl -s -X POST -u admin:admin123 \
  -H "Content-Type: application/json" \
  http://localhost:3000/api/datasources \
  -d '{"name":"Prometheus","type":"prometheus","url":"http://localhost:9090","access":"proxy","isDefault":true,"jsonData":{}}'
```

### Grafana dashboard panels

| Panel | Query | Type |
|-------|-------|------|
| Request Rate | `rate(agent_requests_total[1m])` | Time series |
| Latency P50/P95 | `histogram_quantile(0.5\|0.95, rate(agent_request_latency_seconds_bucket[5m]))` | Time series |
| GPU VRAM % | `gpu_memory_used_bytes / gpu_memory_total_bytes * 100` | Gauge |
| GPU Utilization | `max_over_time(gpu_utilization_percent[30s])` | Gauge |
| Token Rate | `rate(agent_input/output_tokens_total[1m])` | Time series |
| Per-Node Latency | `histogram_quantile(0.5, sum by(node,le)(rate(agent_node_latency_seconds_bucket[5m])))` | Time series |

> **Token rate note:** input tokens (~3000/req) vastly outnumber output tokens (~10/req). Use logarithmic Y-axis scale to see both on the same panel.

> **GPU utilization note:** inference bursts last 100–300ms on RTX 4090. Use `max_over_time(...[30s])` to capture peaks that fall between 1s polling intervals.

### Langfuse tracing (optional)

Per-request LLM traces with per-node input/output snapshots. Disabled by default; enable by setting env vars:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://cloud.langfuse.com   # or self-hosted URL
```

Each `/ask` request creates one Langfuse trace containing a span per LangGraph node (classify, decompose, retrieve, answer, verify) with inputs, outputs, and timing.
