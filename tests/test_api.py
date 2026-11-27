from fastapi.testclient import TestClient

from cricketwp.api.main import app

BASE = {
    "batting_team": "Chennai Super Kings",
    "bowling_team": "Mumbai Indians",
    "venue": "MA Chidambaram Stadium",
    "target": 180,
    "runs": 90,
    "wickets": 3,
    "balls_bowled": 60,
}


# -----------------------------
# Request helpers
# -----------------------------


def call(c, **kw):
    return c.post("/predict/chase", json={**BASE, **kw})


def call_what_if(c, **kw):
    return c.post("/what-if/chase", json={**BASE, **kw})


# -----------------------------
# Existing chase API tests
# -----------------------------


def test_sums_to_one():
    with TestClient(app) as c:
        r = call(c).json()

        total = r["batting_team_win_prob"] + r["bowling_team_win_prob"]

        assert abs(total - 1) < 1e-6


def test_validation():
    with TestClient(app) as c:
        r = call(c, wickets=11)

        assert r.status_code == 422


def test_target_already_reached():
    with TestClient(app) as c:
        r = call(c, target=180, runs=180).json()

        assert r["batting_team_win_prob"] == 1.0


def test_lower_target_never_lowers_probability():
    with TestClient(app) as c:
        baseline = call(c, target=180).json()["batting_team_win_prob"]

        easier_target = call(c, target=150).json()["batting_team_win_prob"]

        assert easier_target >= baseline


def test_more_wickets_lost_never_raises_probability():
    with TestClient(app) as c:
        baseline = call(c, wickets=3).json()["batting_team_win_prob"]

        more_wickets_lost = call(c, wickets=6).json()["batting_team_win_prob"]

        assert more_wickets_lost <= baseline


# -----------------------------
# New what-if API tests
# -----------------------------


def test_what_if_returns_six_outcomes():
    with TestClient(app) as c:
        r = call_what_if(c)

        assert r.status_code == 200

        data = r.json()

        assert len(data["outcomes"]) == 6

        labels = [item["outcome"] for item in data["outcomes"]]

        assert labels == [
            "Dot",
            "1",
            "2",
            "4",
            "6",
            "Wicket",
        ]


def test_what_if_probabilities_are_valid():
    with TestClient(app) as c:
        data = call_what_if(c).json()

        base = data["current_batting_team_win_prob"]

        assert 0 <= base <= 1

        for item in data["outcomes"]:
            probability = item["batting_team_win_prob"]
            change = item["change"]

            assert 0 <= probability <= 1

            assert abs(change - (probability - base)) < 1e-6


def test_what_if_terminal_state_returns_empty_outcomes():
    with TestClient(app) as c:
        r = call_what_if(c, runs=180)

        assert r.status_code == 200

        data = r.json()

        assert data["current_batting_team_win_prob"] == 1.0
        assert data["outcomes"] == []


def test_what_if_validation():
    with TestClient(app) as c:
        r = call_what_if(c, wickets=11)

        assert r.status_code == 422


# -----------------------------
# Trust API tests
# -----------------------------


def test_trust_endpoint_returns_report():
    with TestClient(app) as c:
        r = c.get("/trust")

        assert r.status_code == 200

        data = r.json()

        assert data["version"] == "1.0.0"
        assert data["n_states"] == 114440
        assert data["n_matches"] == 1133

        assert len(data["bins"]) == 9


def test_trust_metrics_are_valid():
    with TestClient(app) as c:
        data = c.get("/trust").json()

        metrics = data["metrics"]
        baseline = data["baseline"]

        for metric in ["logloss", "brier", "auc", "ece"]:
            assert metric in metrics
            assert metric in baseline

        assert 0 <= metrics["auc"] <= 1
        assert 0 <= metrics["ece"] <= 1

        assert metrics["logloss"] < baseline["logloss"]
        assert metrics["brier"] < baseline["brier"]
        assert metrics["auc"] > baseline["auc"]
