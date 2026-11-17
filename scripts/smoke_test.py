import sys
import time

import requests


base = sys.argv[1].rstrip("/")


def ok(cond, msg):
    print(("PASS " if cond else "FAIL ") + msg)

    if not cond:
        sys.exit(1)


t0 = time.time()

# 1. Health check
health = requests.get(base + "/health", timeout=120).json()

ok(health["status"] == "ok", "health")

# 2. Model info
mi = requests.get(base + "/model-info", timeout=30).json()

ok("version" in mi, "model-info " + str(mi.get("version")))

# 3. Chase prediction
body = {
    "batting_team": "Chennai Super Kings",
    "bowling_team": "Mumbai Indians",
    "venue": "MA Chidambaram Stadium",
    "runs": 90,
    "wickets": 3,
    "balls_bowled": 60,
}

r = requests.post(
    base + "/predict/chase",
    json={**body, "target": 180},
    timeout=30,
).json()

a = r["batting_team_win_prob"]
b = r["bowling_team_win_prob"]

ok(
    0 <= a <= 1 and abs(a + b - 1) < 1e-6,
    "chase probabilities sum to 1",
)

# 4. First innings prediction
r = requests.post(
    base + "/predict/first-innings",
    json=body,
    timeout=30,
).json()

lo, mid, hi = (r["projected_score"][k] for k in ("q10", "q50", "q90"))

ok(lo <= mid <= hi, "projected score band is ordered")

print("all passed in %.1fs" % (time.time() - t0))
