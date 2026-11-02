import numpy as np
import pytest

from cricketwp.eval import evaluate, ece


def test_perfect_predictions_have_near_zero_logloss():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])

    metrics = evaluate(y, p)

    assert metrics["logloss"] < 1e-5


def test_perfect_predictions_have_perfect_auc():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])

    metrics = evaluate(y, p)

    assert metrics["auc"] == 1.0


def test_perfect_predictions_have_near_zero_brier():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])

    metrics = evaluate(y, p)

    assert metrics["brier"] < 1e-5


def test_ece_for_perfect_predictions_is_zero():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])

    assert ece(y, p) == 0.0


def test_evaluate_rejects_invalid_probabilities():
    y = np.array([0, 1])
    p = np.array([-0.1, 1.1])

    with pytest.raises(ValueError):
        evaluate(y, p)
