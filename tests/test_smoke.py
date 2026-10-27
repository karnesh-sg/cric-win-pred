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
