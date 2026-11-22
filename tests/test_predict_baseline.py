"""v0 xp baseline on hand-built data: every expected number checks with a calculator."""
import numpy as np
import pandas as pd
import pytest

from predict_baseline import predict

SEASON = "2025-26"


def gw_row(player, gw, points, n_fixtures=1, season=SEASON):
    return dict(season=season, player_code=player, gw=gw, n_fixtures=n_fixtures,
                total_points=np.nan if n_fixtures == 0 else points)


@pytest.fixture
def player_gw():
    rows = [
        # Player 1: 2 GWs last season, then GW1-7 this season with a blank in GW4.
        gw_row(1, 37, 10, season="2024-25"), gw_row(1, 38, 10, season="2024-25"),
        gw_row(1, 1, 1), gw_row(1, 2, 2), gw_row(1, 3, 3), gw_row(1, 4, 0, n_fixtures=0),
        gw_row(1, 5, 5), gw_row(1, 6, 6), gw_row(1, 7, 7),
        # Player 2: a double in GW6 (10 points over 2 matches), then 2 points in GW7.
        gw_row(2, 6, 10, n_fixtures=2), gw_row(2, 7, 2),
        # Player 3: only last season, 2 matches -> the window crosses the season.
        gw_row(3, 37, 4, season="2024-25"), gw_row(3, 38, 6, season="2024-25"),
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def fixtures():
    """GW8-10 this season. Team 1: single, double, blank. Team 2: one match each."""
    matches = [(8, 1, 2), (9, 1, 3), (9, 4, 1), (10, 2, 3)]
    return pd.DataFrame([dict(season=SEASON, gw=gw, team_h_code=h, team_a_code=a)
                         for gw, h, a in matches])


@pytest.fixture
def roster():
    return pd.DataFrame({"player_code": [1, 2, 3, 4], "team_code": [1, 2, 2, 1],
                         "position": ["MID", "FWD", "DEF", "GK"]})


def xp_of(xp, player, gw):
    hit = xp[(xp["player_code"] == player) & (xp["gw"] == gw)]
    assert len(hit) == 1
    return hit.iloc[0]


def test_rate_is_last_5_gameweeks_with_a_match(player_gw, fixtures, roster):
    # Last 5 with a match before GW8: GW2, 3, 5, 6, 7 (GW4 blank skipped, GW1 too old).
    xp = predict(player_gw, fixtures, roster, SEASON, first_gw=8)
    r = xp_of(xp, 1, 8)
    assert r["n_matches"] == 5
    assert r["pts_per_match"] == pytest.approx((2 + 3 + 5 + 6 + 7) / 5)


def test_rate_is_per_match_not_per_gameweek(player_gw, fixtures, roster):
    # 12 points over 3 matches in 2 gameweeks: 4 per match, not 6 per gameweek.
    xp = predict(player_gw, fixtures, roster, SEASON, first_gw=8)
    assert xp_of(xp, 2, 8)["pts_per_match"] == pytest.approx(4.0)


def test_window_crosses_season(player_gw, fixtures, roster):
    xp = predict(player_gw, fixtures, roster, SEASON, first_gw=8)
    assert xp_of(xp, 3, 8)["pts_per_match"] == pytest.approx(5.0)


def test_double_counts_twice_blank_counts_zero(player_gw, fixtures, roster):
    xp = predict(player_gw, fixtures, roster, SEASON, first_gw=8)
    rate = (2 + 3 + 5 + 6 + 7) / 5
    assert xp_of(xp, 1, 8)["xp"] == pytest.approx(rate)        # single
    assert xp_of(xp, 1, 9)["xp"] == pytest.approx(2 * rate)    # double
    assert xp_of(xp, 1, 10)["n_fixtures"] == 0                 # blank
    assert xp_of(xp, 1, 10)["xp"] == 0


def test_no_history_predicts_zero(player_gw, fixtures, roster):
    xp = predict(player_gw, fixtures, roster, SEASON, first_gw=8)
    assert (xp.loc[xp["player_code"] == 4, ["n_matches", "xp"]] == 0).all().all()


def test_horizon_stops_at_season_end(player_gw, fixtures, roster):
    xp = predict(player_gw, fixtures, roster, SEASON, first_gw=8, horizon=6)
    assert sorted(xp["gw"].unique()) == [8, 9, 10]
    assert not xp.duplicated(["player_code", "gw"]).any()


def test_availability_next_gameweek_only_and_u_for_all(player_gw, fixtures, roster):
    snapshot = pd.DataFrame({"player_code": [1, 2, 3],
                             "status": ["d", "u", "a"],
                             "chance_of_playing_next_round": [50.0, 0.0, np.nan]})
    xp = predict(player_gw, fixtures, roster, SEASON, first_gw=8, snapshot=snapshot)
    assert xp_of(xp, 1, 8)["availability"] == 0.5     # doubtful: next GW halved
    assert xp_of(xp, 1, 9)["availability"] == 1.0     # ... later GWs untouched
    assert (xp.loc[xp["player_code"] == 2, "xp"] == 0).all()   # left the club: 0 throughout
    assert xp_of(xp, 3, 8)["availability"] == 1.0     # no news = fit
    assert xp_of(xp, 4, 8)["availability"] == 1.0     # not in the snapshot = fit


@pytest.mark.parametrize("first_gw", [5, 8])
def test_no_leakage_perturbation(player_gw, fixtures, roster, first_gw):
    """Changing any outcome from first_gw on must not change the prediction."""
    before = predict(player_gw, fixtures, roster, SEASON, first_gw)
    changed = player_gw.copy()
    later = (changed["season"] == SEASON) & (changed["gw"] >= first_gw)
    changed.loc[later, "total_points"] += 100
    after = predict(changed, fixtures, roster, SEASON, first_gw)
    pd.testing.assert_frame_equal(before, after)
