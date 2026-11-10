import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from cricketwp.features import add_state_features
from cricketwp.predict import apply_rules, top_factors


class Predictor:
    def __init__(self, folder="artifacts"):
        f = Path(folder)
        self.folder = f

        # Load lookup tables and defaults
        self.look = json.loads((f / "lookups.json").read_text())

        # Load chase model
        self.chase_model = xgb.XGBClassifier()
        self.chase_model.load_model(str(f / "chase_xgb.json"))
        self.iso_c = json.loads((f / "chase_iso.json").read_text())

        # Load batting-first model
        self.first = xgb.XGBClassifier()
        self.first.load_model(str(f / "first_xgb.json"))
        self.iso_f = json.loads((f / "first_iso.json").read_text())

        # Load score prediction quantile models
        self.q10 = xgb.XGBRegressor()
        self.q10.load_model(str(f / "score_q10.json"))

        self.q50 = xgb.XGBRegressor()
        self.q50.load_model(str(f / "score_q50.json"))

        self.q90 = xgb.XGBRegressor()
        self.q90.load_model(str(f / "score_q90.json"))

    def elo(self, team):
        return self.look["elo"].get(team, 1500.0)

    def chase(
        self,
        batting_team,
        bowling_team,
        venue,
        runs,
        wickets,
        balls_bowled,
        target,
        runs_last12=0,
        wkts_last12=0,
        dot_pct_last12=None,
        partnership_runs=0,
    ):
        # 1. Calculate current match state
        balls_left = 120 - balls_bowled
        wickets_in_hand = 10 - wickets
        runs_needed = target - runs

        # 2. Get venue statistics
        venue_stats = self.look["venues"].get(venue, self.look["venue_default"])

        venue_avg_first = venue_stats["avg_first"]
        venue_chase_rate = venue_stats["chase_rate"]

        # 3. Get ELO difference
        elo_diff = self.elo(batting_team) - self.elo(bowling_team)

        # 4. Get default rolling dot-ball percentage
        if dot_pct_last12 is None:
            dot_pct_last12 = self.look["defaults"]["dot_pct_last12"]

        # 5. Build the raw match-state DataFrame
        s = pd.DataFrame(
            [
                {
                    "runs": runs,
                    "wickets": wickets,
                    "balls_bowled": balls_bowled,
                    "balls_left": balls_left,
                    "wickets_in_hand": wickets_in_hand,
                    "target": target,
                    "runs_last12": runs_last12,
                    "wkts_last12": wkts_last12,
                    "dot_pct_last12": dot_pct_last12,
                    "partnership_runs": partnership_runs,
                    "venue_avg_first": venue_avg_first,
                    "venue_chase_rate": venue_chase_rate,
                    "elo_diff": elo_diff,
                    "target_vs_venue": target - venue_avg_first,
                }
            ]
        )

        # 6. Add engineered features
        s = add_state_features(s, chase=True)

        # 7. Select features in the exact training order
        features = [
            "runs_needed",
            "balls_left",
            "wickets_in_hand",
            "crr",
            "rrr",
            "rrr_minus_crr",
            "target",
            "runs_last12",
            "wkts_last12",
            "dot_pct_last12",
            "partnership_runs",
            "venue_avg_first",
            "venue_chase_rate",
            "elo_diff",
            "target_vs_venue",
            "phase",
        ]

        X = s[features]

        # 8. Predict raw probability
        p = self.chase_model.predict_proba(X)[:, 1]

        # 9. Apply isotonic calibration
        iso = self.iso_c
        p = np.interp(p, iso["x"], iso["y"])

        # 10. Apply cricket terminal-state rules
        p = apply_rules(
            p,
            runs_needed=np.array([runs_needed]),
            balls_left=np.array([balls_left]),
            wickets_in_hand=np.array([wickets_in_hand]),
        )

        # 11. Return probability as a Python float
        # 11. Calculate the top 3 factors
        factors = top_factors(
            self.chase_model,
            X,
            features,
            k=3,
            pos="batting team",
        )

        # 12. Return probability and top factors
        return float(p[0]), factors

    def first_innings(
        self,
        batting_team,
        bowling_team,
        venue,
        runs,
        wickets,
        balls_bowled,
        runs_last12=0,
        wkts_last12=0,
        dot_pct_last12=None,
        partnership_runs=0,
    ):
        # 1. Calculate match state
        balls_left = 120 - balls_bowled
        wickets_in_hand = 10 - wickets

        # 2. Get venue statistics
        venue_stats = self.look["venues"].get(venue, self.look["venue_default"])

        venue_avg_first = venue_stats["avg_first"]
        venue_chase_rate = venue_stats["chase_rate"]

        # 3. Calculate ELO difference
        elo_diff_bat = self.elo(batting_team) - self.elo(bowling_team)

        # 4. Default dot-ball percentage
        if dot_pct_last12 is None:
            dot_pct_last12 = self.look["defaults"]["dot_pct_last12"]

        # 5. Build match-state DataFrame
        s = pd.DataFrame(
            [
                {
                    "runs": runs,
                    "wickets": wickets,
                    "balls_bowled": balls_bowled,
                    "balls_left": balls_left,
                    "wickets_in_hand": wickets_in_hand,
                    "runs_last12": runs_last12,
                    "wkts_last12": wkts_last12,
                    "dot_pct_last12": dot_pct_last12,
                    "partnership_runs": partnership_runs,
                    "venue_avg_first": venue_avg_first,
                    "venue_chase_rate": venue_chase_rate,
                    "elo_diff_bat": elo_diff_bat,
                }
            ]
        )

        # 6. Add engineered features
        s = add_state_features(s, chase=False)

        # 7. Select features in exact training order
        features = [
            "runs",
            "wickets_in_hand",
            "balls_left",
            "crr",
            "runs_last12",
            "wkts_last12",
            "dot_pct_last12",
            "partnership_runs",
            "venue_avg_first",
            "venue_chase_rate",
            "elo_diff_bat",
            "phase",
        ]

        X = s[features]

        # 8. Predict raw probability
        p = self.first.predict_proba(X)[:, 1]

        # 9. Apply isotonic calibration
        iso = self.iso_f

        p = np.interp(p, iso["x"], iso["y"])

        # 10. Return Python float
        return float(p[0])

    def teams(self):
        return sorted(self.look["elo"].keys())

    def venues(self):
        return sorted(self.look["venues"].keys())

    def model_info(self):
        meta_path = self.folder / "meta.json"

        meta = json.loads(meta_path.read_text())

        return meta
