from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import FastAPI
from pydantic import BaseModel, Field

from cricketwp.predictor import Predictor


# Locate artifacts from project root
ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts"

# Load models once
pred = Predictor(folder=str(ARTIFACTS))

app = FastAPI(title="Cricket Win Probability")

# Load replay data once
REPLAY_PATH = ARTIFACTS / "replay.csv.gz"

replay_df = pd.read_csv(REPLAY_PATH)

# Extract match-level metadata from first innings
matches_df = (
    replay_df[replay_df["innings"] == 1][
        [
            "match_id",
            "date",
            "venue",
            "batting_team",
            "bowling_team",
        ]
    ]
    .drop_duplicates(subset=["match_id"])
    .sort_values("date", ascending=False)
)

print("Replay matches loaded:", len(matches_df))


# -----------------------------
# Request models
# -----------------------------


class ChaseState(BaseModel):
    batting_team: str
    bowling_team: str
    venue: str

    target: int = Field(gt=0, le=300)
    runs: int = Field(ge=0, le=300)
    wickets: int = Field(ge=0, le=10)
    balls_bowled: int = Field(ge=0, le=120)

    runs_last12: int = Field(default=0, ge=0)
    wkts_last12: int = Field(default=0, ge=0, le=10)
    dot_pct_last12: float = Field(default=0.4166667, ge=0, le=1)
    partnership_runs: int = Field(default=0, ge=0)


class FirstInningsState(BaseModel):
    batting_team: str
    bowling_team: str
    venue: str

    runs: int = Field(ge=0, le=300)
    wickets: int = Field(ge=0, le=10)
    balls_bowled: int = Field(ge=0, le=120)

    runs_last12: int = Field(default=0, ge=0)
    wkts_last12: int = Field(default=0, ge=0, le=10)
    dot_pct_last12: float = Field(default=0.4166667, ge=0, le=1)
    partnership_runs: int = Field(default=0, ge=0)


# -----------------------------
# GET endpoints
# -----------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": pred.model_info()["version"],
    }


@app.get("/teams")
def get_teams():
    return {"teams": pred.teams()}


@app.get("/venues")
def get_venues():
    return {"venues": pred.venues()}


@app.get("/model-info")
def get_model_info():
    return pred.model_info()


@app.get("/matches")
def get_matches():
    return {"matches": matches_df.to_dict(orient="records")}


@app.get("/matches/{match_id}/replay")
def get_match_replay(match_id: int):
    match_replay = replay_df[replay_df["match_id"] == match_id].copy()

    if match_replay.empty:
        raise HTTPException(
            status_code=404,
            detail="Match not found",
        )

    match_replay = match_replay.sort_values(["innings", "balls_bowled"])

    records = (
        match_replay.astype(object).where(pd.notna(match_replay), None).to_dict(orient="records")
    )

    return {
        "match_id": match_id,
        "total_deliveries": len(records),
        "replay": records,
    }


# -----------------------------
# POST endpoints
# -----------------------------


@app.post("/predict/chase")
def predict_chase(s: ChaseState):
    p, factors = pred.chase(
        batting_team=s.batting_team,
        bowling_team=s.bowling_team,
        venue=s.venue,
        runs=s.runs,
        wickets=s.wickets,
        balls_bowled=s.balls_bowled,
        target=s.target,
        runs_last12=s.runs_last12,
        wkts_last12=s.wkts_last12,
        dot_pct_last12=s.dot_pct_last12,
        partnership_runs=s.partnership_runs,
    )

    return {
        "batting_team": s.batting_team,
        "batting_team_win_prob": p,
        "bowling_team": s.bowling_team,
        "bowling_team_win_prob": 1 - p,
        "top_factors": factors,
    }


@app.post("/predict/first-innings")
def predict_first_innings(s: FirstInningsState):
    p = pred.first_innings(
        batting_team=s.batting_team,
        bowling_team=s.bowling_team,
        venue=s.venue,
        runs=s.runs,
        wickets=s.wickets,
        balls_bowled=s.balls_bowled,
        runs_last12=s.runs_last12,
        wkts_last12=s.wkts_last12,
        dot_pct_last12=s.dot_pct_last12,
        partnership_runs=s.partnership_runs,
    )

    return {
        "batting_team": s.batting_team,
        "batting_team_win_prob": p,
        "bowling_team": s.bowling_team,
        "bowling_team_win_prob": 1 - p,
    }
