# Scoring API latency (Phase 7.5)

200 `POST /score` requests with the real model (average of 3 XGBoost boosters, 49 features), in-process via FastAPI's TestClient, after 10 warm-up requests. Machine: arm64, Darwin, Python 3.15.0. Payloads are varied realistic payments from the Simulator's sender profiles. A hosted deployment adds network time.

| Measured | p50 | p95 | p99 | max |
|---|---|---|---|---|
| End to end (client side) | 1.6 ms | 2.1 ms | 2.4 ms | 2.8 ms |
| Inside the service (X-Process-Time-ms) | 0.9 ms | 1.2 ms | 1.5 ms | 1.6 ms |

Mean end to end 1.6 ms; 0 of 200 test payments flagged. Reproduce: `python scripts/api_latency.py`.
