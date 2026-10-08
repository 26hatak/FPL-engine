import numpy as np
import pandas as pd
import pytest

from build_player_gw import GW_KEYS, build
from build_player_match import OUTCOMES

WINDOWS = (3, 5)
FEATURES = [f"f_{name}_{n}" for n in WINDOWS
            for name in ["minutes_mean", "start_rate", "xg_per90", "xa_per90", "defcon_per90"]]


def row(df, season, player, gw):
    """The single player_gw row for one key."""
    hit = df[(df["season"] == season) & (df["player_code"] == player) & (df["gw"] == gw)]
    assert len(hit) == 1, f"expected one row for {season} {player} GW{gw}, got {len(hit)}"
    return hit.iloc[0]


@pytest.fixture
def player_gw(player_match, gameweeks):
    return build(player_match, gameweeks)


def f_cols(df):
    return [c for c in df.columns if c.startswith("f_")]


# --- shape of the table -----------------------------------------------------

def test_keys_unique(player_gw):
    assert not player_gw.duplicated(GW_KEYS).any()


def test_grid_starts_at_first_appearance(player_gw):
    p300 = player_gw[(player_gw["player_code"] == 300)]
    assert sorted(p300["gw"]) == [3, 4, 5, 6]


def test_blank_gameweek_row(player_gw):
    blank = row(player_gw, "2025-26", 200, 4)
    assert blank["n_fixtures"] == 0
    assert pd.isna(blank["total_points"])        # no match happened


def test_double_gameweek_summed(player_gw):
    dgw = row(player_gw, "2025-26", 100, 3)
    assert dgw["n_fixtures"] == 2
    assert dgw["goals_scored"] == 3
    assert dgw["minutes"] == 180
    assert dgw["expected_goals"] == pytest.approx(1.6)


def test_defcon_nan_preserved(player_gw):
    assert pd.isna(row(player_gw, "2024-25", 100, 1)["defensive_contribution"])
    assert row(player_gw, "2025-26", 100, 3)["defensive_contribution"] == 8


def test_deadline_attached(player_gw):
    assert player_gw["deadline_time"].notna().all()


def test_numeric_dtypes(player_gw):
    for col in OUTCOMES + FEATURES + ["n_fixtures"]:
        assert pd.api.types.is_numeric_dtype(player_gw[col]), col


# --- features ---------------------------------------------------------------

def test_feature_columns_present(player_gw):
    missing = [c for c in FEATURES if c not in player_gw.columns]
    assert not missing, f"missing features: {missing}"


def test_first_row_has_no_features(player_gw):
    # Nothing happened before a player's first ever gameweek.
    for season, player, gw in [("2024-25", 100, 1), ("2025-26", 200, 1), ("2025-26", 300, 3)]:
        assert row(player_gw, season, player, gw)[FEATURES].isna().all(), (season, player, gw)


def test_rolling_mean_skips_blank_and_excludes_current(player_gw):
    # Player 200 minutes: GW1 10, GW2 20, GW3 30, GW4 blank, GW5 50, GW6 60.
    assert row(player_gw, "2025-26", 200, 2)["f_minutes_mean_3"] == pytest.approx(10)
    assert row(player_gw, "2025-26", 200, 4)["f_minutes_mean_3"] == pytest.approx(20)  # GW1-3
    assert row(player_gw, "2025-26", 200, 5)["f_minutes_mean_3"] == pytest.approx(20)  # GW1-3, blank skipped
    assert row(player_gw, "2025-26", 200, 6)["f_minutes_mean_3"] == pytest.approx(100 / 3)  # GW2,3,5


def test_rolling_crosses_season_boundary(player_gw):
    # 2025-26 GW1 for player 100 sees both 2024-25 gameweeks: xG 0.6 in 180 minutes.
    assert row(player_gw, "2025-26", 100, 1)["f_xg_per90_3"] == pytest.approx(0.3)


def test_defcon_per90_ignores_unrecorded(player_gw):
    # Before 2025-26 GW2 the only recorded DefCon is GW1: 4 in 90 minutes.
    assert row(player_gw, "2025-26", 100, 2)["f_defcon_per90_3"] == pytest.approx(4)


# --- the leakage test -------------------------------------------------------

@pytest.mark.parametrize("gw", [1, 3, 5])
def test_no_leakage_perturbation(player_match, gameweeks, gw):
    """Make every outcome in one gameweek absurd and rebuild. If any feature at
    or before that gameweek changes, a feature is reading the future (or itself).
    This works however the features are implemented, which is the point."""
    before = build(player_match, gameweeks).set_index(GW_KEYS).sort_index()

    poisoned = player_match.copy()
    hit = (poisoned["season"] == "2025-26") & (poisoned["gw"] == gw)
    poisoned.loc[hit, OUTCOMES] = 999
    after = build(poisoned, gameweeks).set_index(GW_KEYS).sort_index()

    feats = f_cols(before)
    assert feats, "no f_* feature columns built yet"
    past = (before.index.get_level_values("season") == "2024-25") | (
        before.index.get_level_values("gw") <= gw)
    pd.testing.assert_frame_equal(before.loc[past, feats], after.loc[past, feats])

    # Sanity: the poison must show up somewhere later, or the test proves nothing.
    future = ~past
    assert not np.allclose(before.loc[future, feats].fillna(-1),
                           after.loc[future, feats].fillna(-1))
