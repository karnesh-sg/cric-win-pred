import json
from pathlib import Path

import pandas as pd

IGNORED = {"retired hurt", "retired not out"}


def parse_match(path):
    path = Path(path)
    m = json.loads(path.read_text(encoding="utf-8"))
    info = m["info"]

    out = info.get("outcome", {})

    meta = {
        "match_id": path.stem,
        "date": info["dates"][0],
        "venue": info.get("venue"),
        "match_type": info.get("match_type"),
        "gender": info.get("gender"),
        "winner": out.get("winner"),
        "result": out.get("result"),
        "method": out.get("method"),
    }

    rows = []

    for inn_no, inn in enumerate(m["innings"], start=1):
        if inn.get("super_over"):
            continue

        bat = inn["team"]
        bowl = next(t for t in info["teams"] if t != bat)

        for ov in inn["overs"]:
            for d in ov["deliveries"]:
                ex = d.get("extras", {})

                wk = [w for w in d.get("wickets", []) if w["kind"] not in IGNORED]

                rows.append(
                    {
                        **meta,
                        "innings": inn_no,
                        "over": ov["over"],
                        "batting_team": bat,
                        "bowling_team": bowl,
                        "runs_batter": d["runs"]["batter"],
                        "runs_total": d["runs"]["total"],
                        "is_wicket": len(wk),
                        "is_legal": int("wides" not in ex and "noballs" not in ex),
                    }
                )

    return pd.DataFrame(rows)
