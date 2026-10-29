from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Cricket Win Probability")


class ChaseState(BaseModel):
    target: int
    runs: int
    wickets: int
    balls_bowled: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/chase")
def predict_chase(s: ChaseState):
    need = s.target - s.runs
    left = 120 - s.balls_bowled

    p = 0.5 - need / 200 + (10 - s.wickets) / 40 + left / 400

    p = max(0.0, min(1.0, p))

    return {
        "batting_team_win_prob": p,
        "bowling_team_win_prob": 1 - p,
    }
