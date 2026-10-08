"""Checks against the real data. Skipped where the needed files don't exist yet."""
import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from build_player_match import CURRENT_SEASON, approx_deadlines, current_deadlines

RAW = Path("data/raw")
PROCESSED = Path("data/processed")
needs_processed = pytest.mark.skipif(
    not (PROCESSED / "player_match.parquet").exists(),
    reason="run python src/build_player_match.py first")


def load_json(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def test_approx_deadline_rule_matches_real_deadlines():
    """'Deadline = first kickoff - 90 min' must hold this season, where we know
    the real deadlines, before we trust it for past seasons."""
    latest = sorted((RAW / "daily").iterdir())[-1]
    fixtures = pd.DataFrame(load_json(latest / "fixtures.json.gz")).dropna(subset=["event"])
    events = pd.DataFrame(load_json(latest / "bootstrap-static.json.gz")["events"])
    finished = events.loc[events["finished"], "id"]

    real = current_deadlines().set_index("gw")["deadline_time"].loc[finished]
    approx = approx_deadlines(fixtures, CURRENT_SEASON).set_index("gw")["deadline_time"].loc[finished]
    assert (real == approx).all(), (approx - real)[real != approx]


@needs_processed
def test_every_match_has_codes_and_position():
    pm = pd.read_parquet(PROCESSED / "player_match.parquet")
    for col in ["player_code", "team_code", "opponent_code", "position"]:
        assert pm[col].notna().all(), col
    assert set(pm["position"]) == {"GK", "DEF", "MID", "FWD"}


@needs_processed
def test_current_season_points_match_live():
    """player_match is built from element_summary; event/live is FPL's official
    per-GW total. Summed per GW they must agree exactly."""
    pm = pd.read_parquet(PROCESSED / "player_match.parquet")
    pm = pm[pm["season"] == CURRENT_SEASON]
    ours = pm.groupby(["element", "gw"])["total_points"].sum()

    live = pd.DataFrame(
        [{"element": e["id"], "gw": int(p.name[2:4]), "total_points": e["stats"]["total_points"]}
         for p in sorted((RAW / "gameweeks").glob("gw*_live.json.gz"))
         for e in load_json(p)["elements"]]
    ).set_index(["element", "gw"])["total_points"]

    joined = pd.concat([ours.rename("ours"), live.rename("live")], axis=1)
    assert joined.notna().all().all(), "rows missing on one side"
    assert (joined["ours"] == joined["live"]).all()
