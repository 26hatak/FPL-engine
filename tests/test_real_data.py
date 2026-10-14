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


def live_points():
    """FPL's official points per player per finished GW, from event/live."""
    return pd.DataFrame(
        [{"element": e["id"], "gw": int(p.name[2:4]), "total_points": e["stats"]["total_points"]}
         for p in sorted((RAW / "gameweeks").glob("gw*_live.json.gz"))
         for e in load_json(p)["elements"]]
    ).set_index(["element", "gw"])["total_points"]


def current_player_match():
    pm = pd.read_parquet(PROCESSED / "player_match.parquet")
    return pm[pm["season"] == CURRENT_SEASON]


@needs_processed
def test_current_season_points_match_live():
    """player_match is built from element_summary; event/live is FPL's official
    per-GW total. Wherever both have a GW, summed points must agree exactly."""
    ours = current_player_match().groupby(["element", "gw"])["total_points"].sum()
    live = live_points()
    shared_gws = set(ours.index.get_level_values("gw")) & set(live.index.get_level_values("gw"))
    assert shared_gws

    joined = pd.concat([ours.rename("ours"), live.rename("live")], axis=1)
    joined = joined[joined.index.get_level_values("gw").isin(shared_gws)]
    assert joined.notna().all().all(), "players missing on one side"
    assert (joined["ours"] == joined["live"]).all()


@needs_processed
def test_player_match_covers_every_finished_gw():
    """The daily job saves gwNN_live as each GW finishes and re-downloads
    element_summary in the same run. Fails if that refresh didn't happen."""
    ours = set(current_player_match()["gw"])
    live = set(live_points().index.get_level_values("gw"))
    missing = sorted(live - ours)
    assert not missing, (f"player_match lacks GW {missing}: the history refresh "
                         "didn't run. Run `python src/snapshot.py --backfill`, then rebuild")


needs_player_gw = pytest.mark.skipif(
    not (PROCESSED / "player_gw.parquet").exists(),
    reason="run python src/build_all.py first")


@needs_player_gw
def test_snapshots_are_from_before_each_deadline():
    """The as-of join's guarantee, checked on the real table: no snapshot used
    for a gameweek was taken at or after that gameweek's deadline."""
    gw = pd.read_parquet(PROCESSED / "player_gw.parquet")
    used = gw[gw["snap_time"].notna()]
    assert len(used) > 0
    assert (used["snap_time"] < used["deadline_time"]).all()
