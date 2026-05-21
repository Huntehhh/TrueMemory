# Resource Budgets by Tier

TrueMemory's memory footprint varies by tier. These budgets were established
in v0.7.0 after extensive benchmarking on Apple Silicon Macs.

## Architecture

All tiers use a **shared model server** (`truememory-model-server`) that loads
models once and serves all MCP sessions via a Unix domain socket. Each MCP
session is a lightweight proxy (~80 MB) that delegates model inference to the
shared server.

## Per-Tier Budgets

| Tier | Model Server | Each MCP Session | 5 Sessions Total |
|------|-------------|-----------------|------------------|
| **Edge** | ~500 MB | ~80 MB | ~900 MB |
| **Base** | ~1.5 GB | ~80 MB | ~1.9 GB |
| **Pro** | ~1.5 GB | ~80 MB | ~1.9 GB |

## MPS Watermark

The PyTorch MPS memory watermark is clamped to prevent over-allocation:

```
ceiling = max(1.5 GB, min(ram * 0.08, 2.5 GB))
```

| Machine RAM | MPS Ceiling |
|-------------|------------|
| 8 GB | 1.5 GB |
| 16 GB | 1.3 GB |
| 24 GB | 1.9 GB |
| 32+ GB | 2.5 GB |

Users can override via `PYTORCH_MPS_HIGH_WATERMARK_RATIO` environment variable.

## What Consumes Memory

- **PyTorch runtime**: ~800 MB (loaded by model server)
- **Embedding model** (Base/Pro): Qwen3-Embedding-0.6B ~600 MB
- **Embedding model** (Edge): model2vec/potion-base-8M ~30 MB
- **Reranker** (Base/Pro): gte-reranker-modernbert-base ~300 MB on MPS
- **Reranker** (Edge): ms-marco-MiniLM-L-6-v2 ~22 MB
- **MPS GPU workspace**: varies by watermark ratio
- **Each MCP session**: Python + SQLite + protocol handling ~80 MB

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | auto | MPS memory ceiling as fraction of RAM |
| `TRUEMEMORY_MODEL_SERVER_IDLE` | 300 | Seconds before idle model server exits |
| `TRUEMEMORY_NO_MODEL_SERVER` | 0 | Set to 1 to disable shared model server |
| `TRUEMEMORY_MAX_RSS_MB` | 0 | Reported in stats (informational) |
