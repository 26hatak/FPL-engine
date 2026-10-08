"""Download past FPL seasons from vaastav/Fantasy-Premier-League.

Run once: python src/download_past_seasons.py

Files come from a pinned commit, so re-downloading always gives identical data.
Saves gzipped CSV under data/raw/vaastav/<season>/. Never edit these files.
"""
import gzip
import time
from pathlib import Path

import requests

# Pinned commit (2026-08-28). Change only on purpose, and note why in DECISIONS.md.
SHA = "9779cdbc0c07f6c900c2d0c181ddf6bb9c800f88"
BASE = f"https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/{SHA}/data"
SEASONS = ["2024-25", "2025-26"]
FILES = {
    "merged_gw": "gws/merged_gw.csv",   # one row per player per match
    "players_raw": "players_raw.csv",   # season id -> stable 'code'
    "teams": "teams.csv",               # season team id -> stable 'code'
    "fixtures": "fixtures.csv",         # kickoffs, used to approximate deadlines
}
RAW = Path("data/raw/vaastav")

session = requests.Session()
session.headers.update({"User-Agent": "fpl-engine student project (26hatak@gmail.com)"})


def save(text: str, path: Path) -> None:
    """Write text gzipped, matching the rest of data/raw."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    for season in SEASONS:
        for name, remote in FILES.items():
            out = RAW / season / f"{name}.csv.gz"
            if out.exists():
                continue
            r = session.get(f"{BASE}/{season}/{remote}", timeout=60)
            r.raise_for_status()
            save(r.text, out)
            print(f"saved {out}")
            time.sleep(1)
