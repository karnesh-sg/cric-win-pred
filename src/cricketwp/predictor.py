import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from cricketwp.features import add_state_features
from cricketwp.predict import apply_rules

OUTCOMES = [
    ("Dot", 0, 0),
    ("1", 1, 0),
    ("2", 2, 0),
    ("4", 4, 0),
    ("6", 6, 0),
    ("Wicket", 0, 1),
]


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

    @staticmethod
    def _top_shap_factors(model, X, k=3):
        """
        Return the top k TreeSHAP feature contributions.

        XGBoost pred_contribs returns contributions in raw-margin
        (log-odds) space. The final column is the bias/base value,
        which is excluded from the ranked feature list.

        These values explain the raw classifier output before isotonic
        calibration and cricket terminal-state rules. They are not
        percentage-point contributions to the final calibrated probability.
        """
        booster = model.get_booster()

        dmatrix = xgb.DMatrix(
            X,
            feature_names=list(X.columns),
        )

        contributions = booster.predict(
            dmatrix,
            pred_contribs=True,
        )

        # The predictor currently passes one match-state row.
        feature_values = contributions[0, :-1]
        feature_names = list(X.columns)

        # Rank features by absolute SHAP contribution.
        ranked_indices = np.argsort(np.abs(feature_values))[::-1][:k]

        factors = []

        for index in ranked_indices:
            shap_value = float(feature_values[index])

            factors.append(
                {
                    "feature": feature_names[index],
                    "name": feature_names[index],
                    "shap_value": shap_value,
                    "impact": round(shap_value, 4),
                    "direction": (
                        "favors batting team"
                        if shap_value > 0
                        else "favors bowling team"
                        if shap_value < 0
                        else "neutral"
                    ),
                    "space": "raw_model_log_odds",
                }
            )

        return factors

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
        venue_stats = self.look["venues"].get(
            venue,
            self.look["venue_default"],
        )

        venue_avg_first = venue_stats["avg_first"]
        venue_chase_rate = venue_stats["chase_rate"]

        # 3. Get ELO difference
        elo_diff = self.elo(batting_team) - self.elo(bowling_team)

        # 4. Get default rolling dot-ball percentage
        if dot_pct_last12 is None:
            dot_pct_last12 = self.look["defaults"]["dot_pct_last12"]

        # 5. Build raw match-state DataFrame
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
                    "target_vs_venue": (target - venue_avg_first),
                }
            ]
        )

        # 6. Add engineered features
        s = add_state_features(s, chase=True)

        # 7. Select features in exact training order
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
        raw_p = self.chase_model.predict_proba(X)[:, 1]

        # 9. Apply isotonic calibration
        iso = self.iso_c

        p = np.interp(
            raw_p,
            iso["x"],
            iso["y"],
        )

        # 10. Apply cricket terminal-state rules
        p = apply_rules(
            p,
            runs_needed=np.array([runs_needed]),
            balls_left=np.array([balls_left]),
            wickets_in_hand=np.array([wickets_in_hand]),
        )

        # 11. Calculate model-specific TreeSHAP factors
        factors = self._top_shap_factors(
            self.chase_model,
            X,
            k=3,
        )

        # 12. Return probability and factors
        return float(p[0]), factors

    def what_if_chase(
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

        if dot_pct_last12 is None:
            dot_pct_last12 = self.look["defaults"]["dot_pct_last12"]

        base, _ = self.chase(
            batting_team=batting_team,
            bowling_team=bowling_team,
            venue=venue,
            runs=runs,
            wickets=wickets,
            balls_bowled=balls_bowled,
            target=target,
            runs_last12=runs_last12,
            wkts_last12=wkts_last12,
            dot_pct_last12=dot_pct_last12,
            partnership_runs=partnership_runs,
        )

        if runs >= target or balls_bowled >= 120 or wickets >= 10:
            return base, []

        results = []
        for label, runs_scored, wicket in OUTCOMES:
            new_runs = runs + runs_scored
            new_wickets = wickets + wicket
            new_balls_bowled = balls_bowled + 1
            new_runs_last12 = round(runs_last12 * 11 / 12) + runs_scored
            new_wkts_last12 = round(wkts_last12 * 11 / 12) + wicket
            is_dot = 1 if runs_scored == 0 else 0
            new_dot_pct_last12 = (dot_pct_last12 * 11 + is_dot) / 12
            new_partnership_runs = 0 if wicket else partnership_runs + runs_scored
            probability, _ = self.chase(
                batting_team=batting_team,
                bowling_team=bowling_team,
                venue=venue,
                runs=new_runs,
                wickets=new_wickets,
                balls_bowled=new_balls_bowled,
                target=target,
                runs_last12=new_runs_last12,
                wkts_last12=new_wkts_last12,
                dot_pct_last12=new_dot_pct_last12,
                partnership_runs=new_partnership_runs,
            )

            results.append(
                {
                    "outcome": label,
                    "batting_team_win_prob": probability,
                    "change": probability - base,
                }
            )

        return base, results

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
        venue_stats = self.look["venues"].get(
            venue,
            self.look["venue_default"],
        )

        venue_avg_first = venue_stats["avg_first"]
        venue_chase_rate = venue_stats["chase_rate"]

        # 3. Calculate ELO difference
        elo_diff_bat = self.elo(batting_team) - self.elo(bowling_team)

        # 4. Get default dot-ball percentage
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
        raw_p = self.first.predict_proba(X)[:, 1]

        # 9. Apply isotonic calibration
        iso = self.iso_f

        p = np.interp(
            raw_p,
            iso["x"],
            iso["y"],
        )

        # 10. Predict projected score using trained quantile models
        q10_pred = float(self.q10.predict(X)[0])
        q50_pred = float(self.q50.predict(X)[0])
        q90_pred = float(self.q90.predict(X)[0])

        # Keep the quantile range ordered in case models cross
        q10_pred, q50_pred, q90_pred = sorted([q10_pred, q50_pred, q90_pred])

        projected_score = {
            "q10": int(round(q10_pred)),
            "q50": int(round(q50_pred)),
            "q90": int(round(q90_pred)),
        }

        # 11. Explain win-probability classifier using TreeSHAP
        # Note: quantile score models are separate models.
        factors = self._top_shap_factors(
            self.first,
            X,
            k=3,
        )

        # 12. Return probability, score projection, and factors
        return float(p[0]), projected_score, factors

    def teams(self):
        return sorted(self.look["elo"].keys())

    def venues(self):
        return sorted(self.look["venues"].keys())

    def model_info(self):
        meta_path = self.folder / "meta.json"

        meta = json.loads(meta_path.read_text())

        return meta
