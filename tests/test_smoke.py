import pandas as pd


def test_import():
    import cricketwp

    assert cricketwp is not None


import json
from pathlib import Path

from cricketwp.data.parse import parse_match

FILES = sorted(Path("tests/fixtures").glob("*.json"))


def test_totals_match_raw_json():
    assert FILES, "Put sample match files in tests/fixtures"

    for f in FILES:
        raw = json.loads(f.read_text(encoding="utf-8"))
        df = parse_match(f)

        for i, inn in enumerate(raw["innings"], start=1):
            if inn.get("super_over"):
                continue

            expected = sum(d["runs"]["total"] for ov in inn["overs"] for d in ov["deliveries"])

            got = df[df.innings == i]["runs_total"].sum()

            assert got == expected, (
                f"Total mismatch in {f.name}, innings {i}: expected {expected}, got {got}"
            )


from cricketwp.data.build_balls import build_balls
from cricketwp.pipeline import build_match_states


def test_two_fixture_pipeline(tmp_path):
    fixture_dir = Path("tests/fixtures")

    balls_path = tmp_path / "interim" / "balls.parquet"
    states_dir = tmp_path / "states"

    # Stage 1: Parse both fixtures into a balls dataset.
    build_balls(
        raw_dir=fixture_dir,
        output_path=balls_path,
    )

    assert balls_path.exists()

    balls = pd.read_parquet(balls_path)
    assert not balls.empty
    assert balls["match_id"].nunique() == 2

    # Stage 2: Build match and innings states.
    summary = build_match_states(
        balls_path=balls_path,
        output_dir=states_dir,
    )

    assert summary["matches"] == 2
    assert summary["first_rows"] > 0
    assert summary["chase_rows"] > 0

    assert (states_dir / "matches.parquet").exists()
    assert (states_dir / "first_states.parquet").exists()
    assert (states_dir / "chase_states.parquet").exists()
