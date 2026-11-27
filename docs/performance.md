# API Performance and Load Test Report

**Test date:** 2026-11-27
**Tool:** Locust 2.47.0
**Python:** 3.14.6
**Target:** Local Docker container (`127.0.0.1:8001`)
**Docker image:** `cwp-api`

## Test Environment

| Component | Specification |
|---|---|
| CPU | AMD Ryzen 3 3250U with Radeon Graphics |
| System RAM | 9.88 GB |
| Docker CPUs | 4 |
| Docker memory | Approximately 4.743 GiB |

## Test 1 — 50-user stress test

**Configuration:** 50 users, spawn rate 5 users/second, duration 60 seconds. Historical Locust wait time: 1–2 seconds per user.

| Metric | Result |
|---|---:|
| Total requests | 1,544 |
| Failed requests | 1,106 |
| Failure rate | 71.63% |
| Aggregate throughput | 26.92 requests/second |
| Aggregate p50 latency | 16 ms |
| Aggregate p95 latency | 2,200 ms |
| Prediction requests | 1,166 |
| Prediction failures | 1,106 |
| Health-check requests | 378 |
| Health-check failures | 0 |

### Observations

- All 1,106 failures were HTTP 429 responses from `/predict/chase`.
- The H4 per-IP rate limiter enforced its configured 60 requests/minute limit.
- All `/health` requests succeeded.
- This was a rate-limit stress test, not a measurement of maximum successful prediction throughput.
- The high tail latency indicates contention under this workload.

## Test 2 — controlled prediction test

**Configuration:** 2 users, spawn rate 1 user/second, duration 60 seconds.

| Metric | Result |
|---|---:|
| Total requests | 30 |
| Failed requests | 0 |
| Failure rate | 0% |
| Prediction requests | 22 |
| Health-check requests | 8 |
| Prediction average latency | 149 ms |
| Prediction p50 latency | 69 ms |
| Prediction p95 latency | 330 ms |
| Prediction maximum latency | 1,326 ms |
| Locust exit code | 0 |

### Observations

- All requests succeeded.
- No rate-limit responses occurred.
- The controlled test provides a baseline for prediction latency under light load.
- The maximum latency is retained to document occasional slow responses.

## Conclusion

The local Dockerized API remained healthy during both tests.

The stress test confirmed that the H4 rate limiter rejects excessive prediction traffic with HTTP 429 responses while health checks remain available.

The controlled test completed with zero failures and a prediction p95 latency of 330 ms.
