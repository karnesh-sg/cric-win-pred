import numpy as np
from cricketwp.predict import apply_rules


def test_chase_completed():
    p = np.array([0.65])
    result = apply_rules(
        p, runs_needed=np.array([0]), balls_left=np.array([0]), wickets_in_hand=np.array([0])
    )

    assert result[0] == 1.0


def test_no_balls_remaining():
    p = np.array([0.70])
    result = apply_rules(
        p, runs_needed=np.array([5]), balls_left=np.array([0]), wickets_in_hand=np.array([3])
    )

    assert result[0] == 0.0


def test_all_out_before_chase_completed():
    p = np.array([0.80])
    result = apply_rules(
        p, runs_needed=np.array([10]), balls_left=np.array([12]), wickets_in_hand=np.array([0])
    )

    assert result[0] == 0.0


import pandas as pd
from cricketwp.predict import chase_curve
from cricketwp.features import CHASE_FEATURES


class DummyClassifier:
    def predict_proba(self, X):
        raw = np.array([0.4, 0.6, 0.9])

        return np.column_stack([1 - raw, raw])


def test_chase_curve_and_wpa():
    # Create a DataFrame containing all model features
    df = pd.DataFrame({feature: np.zeros(3) for feature in CHASE_FEATURES})

    # Add the columns needed by chase_curve
    df["balls_bowled"] = [1, 2, 3]
    df["runs_needed"] = [10, 5, 0]
    df["balls_left"] = [20, 0, 0]
    df["wickets_in_hand"] = [5, 2, 0]

    clf = DummyClassifier()

    # Identity calibration for this test
    iso_x = np.array([0.0, 1.0])
    iso_y = np.array([0.0, 1.0])

    result = chase_curve(df, clf, iso_x, iso_y)

    # Check win probabilities after rules
    np.testing.assert_allclose(result["p_chase"].values, [0.4, 0.0, 1.0])

    # Check WPA
    np.testing.assert_allclose(result["wpa"].values, [0.0, -0.4, 1.0])
