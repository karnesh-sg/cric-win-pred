import json
import logging
import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from cricketwp.predictor import Predictor
from collections import deque
from threading import Lock


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cricketwp.api")

stats_lock = Lock()

api_stats = {
    "prediction_count": 0,
    "probability_sum": 0.0,
}

request_latencies = deque(maxlen=10000)


def record_prediction(probability: float, innings: str, venue: str):
    with stats_lock:
        api_stats["prediction_count"] += 1
        api_stats["probability_sum"] += float(probability)

    logger.info(
        json.dumps(
            {
                "event": "prediction",
                "innings": innings,
                "venue": venue,
                "probability": round(float(probability), 6),
            }
        )
    )


# Locate artifacts from project root
ARTIFACTS = Path(__file__).resolve().parents[3] / "artifacts"

# Load trust report once
TRUST_PATH = ARTIFACTS / "trust.json"

if TRUST_PATH.exists():
    with open(TRUST_PATH, encoding="utf-8") as f:
        trust_data = json.load(f)
else:
    trust_data = None

# Load models once
pred = Predictor(folder=str(ARTIFACTS))

app = FastAPI(title="Cricket Win Probability")


@app.middleware("http")
async def request_logging(request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        latency_ms = (time.perf_counter() - start_time) * 1000

        with stats_lock:
            request_latencies.append(latency_ms)

        response.headers["x-request-id"] = request_id

        log_record = {
            "event": "http_request",
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": round(latency_ms, 2),
        }

        logger.info(json.dumps(log_record))

        return response

    except Exception:
        latency_ms = (time.perf_counter() - start_time) * 1000

        logger.exception(
            json.dumps(
                {
                    "event": "http_request_error",
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": 500,
                    "latency_ms": round(latency_ms, 2),
                }
            )
        )

        raise


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


@app.get("/ready")
def readiness():
    try:
        info = pred.model_info()

        return {
            "status": "ready",
            "model_version": info["version"],
        }

    except Exception:
        logger.exception("Readiness check failed")

        raise HTTPException(
            status_code=503,
            detail="Model is not ready",
        )


@app.get("/stats")
def get_stats():
    with stats_lock:
        count = api_stats["prediction_count"]

        mean_probability = api_stats["probability_sum"] / count if count > 0 else None

        latencies = list(request_latencies)

    p95_latency = float(np.percentile(latencies, 95)) if latencies else None

    return {
        "prediction_count": count,
        "mean_probability": (round(mean_probability, 6) if mean_probability is not None else None),
        "p95_latency_ms": (round(p95_latency, 2) if p95_latency is not None else None),
        "latency_sample_size": len(latencies),
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


@app.get("/trust")
def get_trust():
    if trust_data is None:
        raise HTTPException(
            status_code=503,
            detail="Trust report is not available",
        )

    return trust_data


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

    record_prediction(p, "chase", s.venue)

    return {
        "batting_team": s.batting_team,
        "batting_team_win_prob": p,
        "bowling_team": s.bowling_team,
        "bowling_team_win_prob": 1 - p,
        "top_factors": factors,
        "explanation_type": "xgboost_treeshap_raw_margin",
        "explanation_note": (
            "TreeSHAP contributions explain the uncalibrated "
            "XGBoost raw margin (log-odds). The displayed win "
            "probability is isotonic-calibrated and then adjusted "
            "by cricket terminal-state rules."
        ),
    }


@app.post("/what-if/chase")
def what_if_chase(s: ChaseState):
    base, outcomes = pred.what_if_chase(
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
        "bowling_team": s.bowling_team,
        "current_batting_team_win_prob": base,
        "outcomes": outcomes,
    }


@app.post("/predict/first-innings")
def predict_first_innings(s: FirstInningsState):
    p, projected_score, factors = pred.first_innings(
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

    record_prediction(p, "first_innings", s.venue)

    return {
        "batting_team": s.batting_team,
        "batting_team_win_prob": p,
        "bowling_team": s.bowling_team,
        "bowling_team_win_prob": 1 - p,
        "projected_score": projected_score,
        "top_factors": factors,
        "explanation_type": "xgboost_treeshap_raw_margin",
        "explanation_note": (
            "TreeSHAP contributions explain the uncalibrated "
            "XGBoost raw margin (log-odds) for the win-probability "
            "classifier. The displayed win probability is "
            "isotonic-calibrated. Score quantiles are predictions "
            "from separate trained quantile models."
        ),
    }
