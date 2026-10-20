"""Phase 7.5: latency of the scoring API with the real model (3 XGBoost boosters from data/demo/model/).

    python scripts/api_latency.py            # 200 requests, writes reports/latency.md

In-process (FastAPI TestClient, no network), so this measures the service itself: request parsing, validation, the
3-model ensemble and the response. A hosted deployment adds network time on top. Payloads are varied, realistic
payments built from the Simulator's sender profiles (app.logic.whatif_features), not one repeated request.
"""
from __future__ import annotations

import platform
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient

from api.main import create_app
from app import logic

N, SEED = 200, 0


def main() -> None:
    b = logic.load_bundle(ROOT / "data" / "demo")
    rng = random.Random(SEED)
    profiles, cat = list(b.profiles["profiles"]), b.categories
    payloads = []
    for _ in range(N):
        f = logic.whatif_features(b, amount=round(10 ** rng.uniform(1, 6), 2), pay_format=rng.choice(cat["pay_format"]),
                                  pay_currency=rng.choice(list(cat["currencies"])), different_receive_currency=rng.random() < 0.1,
                                  same_bank=rng.random() < 0.3, hour=rng.randrange(24), profile=rng.choice(profiles),
                                  out_cnt_24h=rng.randrange(0, 50), uniq_out=rng.randrange(0, 200))
        payloads.append({"features": {k: v for k, v in f.items() if v is not None}})
    with TestClient(create_app(demo_dir=ROOT / "data" / "demo")) as client:
        assert client.get("/health").json()["model_loaded"], "the model did not load (on a Mac: XGBoost needs libomp)"
        for p in payloads[:10]:                                           # warm-up, not measured
            client.post("/score", json=p)
        wall, server, flagged = [], [], 0
        for p in payloads:
            t0 = time.perf_counter()
            r = client.post("/score", json=p)
            wall.append((time.perf_counter() - t0) * 1000)
            assert r.status_code == 200, r.text
            server.append(float(r.headers["X-Process-Time-ms"]))
            flagged += r.json()["flagged"]

    def pct(xs, q):
        return sorted(xs)[min(len(xs) - 1, round(q * (len(xs) - 1)))]

    rows = [("End to end (client side)", wall), ("Inside the service (X-Process-Time-ms)", server)]
    lines = ["# Scoring API latency (Phase 7.5)", "",
             (f"{N} `POST /score` requests with the real model (average of 3 XGBoost boosters, {len(b.meta['feature_names'])} features), in-process via FastAPI's "
             f"TestClient, after 10 warm-up requests. Machine: {platform.machine()}, {platform.system()}, Python {platform.python_version()}. "
             "Payloads are varied realistic payments from the Simulator's sender profiles. A hosted deployment adds network time."), "",
             "| Measured | p50 | p95 | p99 | max |", "|---|---|---|---|---|"]
    lines += [f"| {name} | {pct(xs, .5):.1f} ms | {pct(xs, .95):.1f} ms | {pct(xs, .99):.1f} ms | {max(xs):.1f} ms |" for name, xs in rows]
    lines += ["", f"Mean end to end {statistics.mean(wall):.1f} ms; {flagged} of {N} test payments flagged. Reproduce: `python scripts/api_latency.py`."]
    (ROOT / "reports" / "latency.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
