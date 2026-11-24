from pathlib import Path

import pandas as pd

from cricketwp.data.parse import parse_match


def build_balls(
    raw_dir=Path("data/raw"),
    output_path=Path("data/interim/balls.parquet"),
):

    frames = []
    skipped = 0
    files = sorted(raw_dir.rglob("*.json"))

    if not files:
        raise FileNotFoundError(f"No JSON files found under {raw_dir.resolve()}")

    print(f"Found {len(files)} JSON files.")

    for f in files:
        try:
            frames.append(parse_match(f))
        except Exception as e:
            skipped += 1
            print("skip", f.name, e)

    if not frames:
        raise RuntimeError("No matches could be parsed successfully.")

    df = pd.concat(frames, ignore_index=True)

    # Keep men's T20 and T20 International matches.
    df = df[df["match_type"].isin(["T20", "IT20"]) & (df["gender"] == "male")].copy()

    # A standard T20 innings should not exceed 120 legal balls.
    legal = df.groupby(["match_id", "innings"])["is_legal"].sum().sort_values(ascending=False)

    print("\nInnings with the most legal balls:")
    print(legal.head(10))

    print("\nInnings exceeding 120 legal balls:")
    print(legal[legal > 120])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    match_count = df["match_id"].nunique()

    print(f"Skipped files: {skipped}")
    print(f"Saved to: {output_path}")
    print(f"{match_count} matches, {len(df)} rows")


def main():
    build_balls()


if __name__ == "__main__":
    main()
