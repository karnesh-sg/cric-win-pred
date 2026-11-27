import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

from cricketwp.predict import chase_curve

# Official held-out test data
te = pd.read_parquet("data/processed/chase_test.parquet")

# Final model and saved calibration
clf = xgb.XGBClassifier()
clf.load_model("artifacts/chase_xgb.json")

with open("artifacts/chase_iso.json", encoding="utf-8") as f:
    iso = json.load(f)

curve = chase_curve(
    te,
    clf,
    np.asarray(iso["x"]),
    np.asarray(iso["y"]),
)

t = te.copy()
t["p_chase"] = curve["p_chase"].to_numpy()

# Five selected cases
cases = {
    1488332: "Bermuda vs Bahamas",
    1523004: "Bhutan vs Thailand",
    1557533: "Mozambique vs Zambia",
    1473468: "Punjab Kings vs KKR",
    1502748: "Tanzania vs Uganda",
}

output_dir = Path("docs/figs")
output_dir.mkdir(parents=True, exist_ok=True)

for match_id, title in cases.items():
    m = t[t["match_id"].astype(str) == str(match_id)]
    m = m.sort_values("balls_bowled").copy()

    if m.empty:
        print("Missing:", match_id)
        continue

    # Actual winner's probability
    label = int(m["label"].iloc[0])

    if label == 1:
        m["p_winner"] = m["p_chase"]
    else:
        m["p_winner"] = 1 - m["p_chase"]

    # Restrict displayed curve to post-powerplay
    m = m[m["balls_bowled"] > 36]

    worst = m.loc[m["p_winner"].idxmin()]

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        m["balls_bowled"],
        m["p_winner"] * 100,
        linewidth=2,
        label="Eventual winner probability",
    )

    ax.axhline(50, linestyle="--", linewidth=1)

    ax.scatter(
        [worst["balls_bowled"]],
        [worst["p_winner"] * 100],
        s=80,
        marker="x",
        label="Lowest probability given to winner",
    )

    ax.set_title(f"{title} | Match ID {match_id}\nWinner: {m['winner'].iloc[0]}")

    ax.set_xlabel("Legal balls bowled")
    ax.set_ylabel("Actual winner probability (%)")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()

    path = output_dir / f"failure_{match_id}.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)

    print("Saved:", path)
