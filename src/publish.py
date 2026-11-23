"""Collect the tables the app needs into publish/, ready to upload.

Run from the project root, after build_all.py: python src/publish.py

Why: data/processed/ is git-ignored (it is rebuilt from data/raw/), so a hosted
app can't read it from the repo. The daily GitHub job runs this script and
uploads publish/ to the GitHub Release tagged "data-latest"; the app downloads
from there (src/app_data.py).

publish/ holds:
  xp, fixtures, teams, players   copied as they are from data/processed/
  prices                         the rows of the LATEST snapshot in players_daily
                                 (price, status, news, ownership right now);
                                 the full history isn't needed by the app
  gw_values                      each player's price at each gameweek this season
                                 (from player_gw): fpl_sync's purchase price for
                                 players in a team's first squad
  xp_components                  only if data/processed/ has it (the Phase 1
                                 model will write it); skipped quietly if not
  manifest.json                  as_of, row counts and the git commit of the data,
                                 so the app can say how fresh its data is
"""
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from build_player_match import CURRENT_SEASON

PROCESSED = Path("data/processed")
PUBLISH = Path("publish")
TABLES = ["xp", "fixtures", "teams", "players"]   # must exist
OPTIONAL = ["xp_components"]                      # published when present


def latest_prices(players_daily: pd.DataFrame) -> pd.DataFrame:
    """The rows of the most recent snapshot: one row per player, as of now."""
    latest = players_daily["snapshot_time"].max()
    return players_daily[players_daily["snapshot_time"] == latest].reset_index(drop=True)


def gw_values(player_gw: pd.DataFrame) -> pd.DataFrame:
    """season, player_code, gw, value (tenths of £m) for the current season.

    fpl_sync needs a player's price at a team's first deadline: players in the
    first squad have no transfer record to read a purchase price from. Only
    these 4 columns and this season, so the hosted app downloads ~10k rows
    instead of the whole player_gw table."""
    cur = player_gw[player_gw["season"] == CURRENT_SEASON]
    return (cur[["season", "player_code", "gw", "value"]]
            .dropna(subset=["value"]).reset_index(drop=True))


def git_sha() -> str | None:
    """The commit the data came from, or None outside a git checkout.

    In the GitHub job this is the snapshot commit the job just pushed, not
    GITHUB_SHA (the commit that existed when the run started)."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def publish(processed: Path = PROCESSED, out: Path = PUBLISH,
            sha: str | None = None) -> dict:
    """Write the app's tables + manifest.json into `out`; return the manifest.

    `out` is emptied first so a table that stopped existing (say
    xp_components) can't linger and get uploaded stale."""
    tables = {name: pd.read_parquet(processed / f"{name}.parquet") for name in TABLES}
    for name in OPTIONAL:
        path = processed / f"{name}.parquet"
        if path.exists():
            tables[name] = pd.read_parquet(path)
    tables["prices"] = latest_prices(pd.read_parquet(processed / "players_daily.parquet"))
    tables["gw_values"] = gw_values(pd.read_parquet(
        processed / "player_gw.parquet", columns=["season", "player_code", "gw", "value"]))

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)

    manifest = {
        # The snapshot everything was built from: the newest thing the app knows.
        "as_of": tables["prices"]["snapshot_time"].max().isoformat(),
        "git_sha": sha if sha is not None else git_sha(),
        "rows": {name: len(df) for name, df in sorted(tables.items())},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    manifest = publish()
    print(f"publish/ as of {manifest['as_of']} (commit {manifest['git_sha']})")
    for name, n in manifest["rows"].items():
        print(f"  {name:<14} {n:>6} rows")


if __name__ == "__main__":
    main()
