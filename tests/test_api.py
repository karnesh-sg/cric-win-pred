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


def call(c, **kw):
    return c.post("/predict/chase", json={**BASE, **kw})


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
