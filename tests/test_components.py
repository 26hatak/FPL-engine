"""v1 decomposed xP (player_fixture -> predict_components) on hand-built data.

Season 2025-26 plays the role of "this season": GW1-3 are played, GW4-6 are
in the future. Teams 1 and 2 have players; team 3 has none (a promoted side
we know nothing about). The schedule:

    GW1  1 v 2                      played
    GW2  2 v 1                      played
    GW3  1 v 2                      played
    GW4  1 v 2,  3 v 1              team 1 has a DOUBLE
    GW5  2 v 3                      team 1 has a BLANK
    GW6  1 v 2
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import build_player_fixture
import build_player_gw
from build_player_match import COLUMNS, OUTCOMES
from predict_components import (COMPONENTS, expected_floor_div, load_scoring, predict_fixtures,
                                prob_at_least, shrink, to_gameweeks, upcoming)

SEASON = "2025-26"
BASE = pd.Timestamp("2025-08-01", tz="UTC")
SCHEDULE = [  # fixture, gw, home, away, days after the GW deadline
    (1, 1, 1, 2, 1), (2, 2, 2, 1, 1), (3, 3, 1, 2, 1),
    (4, 4, 1, 2, 1), (5, 4, 3, 1, 4), (6, 5, 2, 3, 1), (7, 6, 1, 2, 1)]
PLAYED_GWS = 3


def deadline(gw):
    return BASE + pd.Timedelta(days=7 * gw)


def kickoff(fixture):
    _, gw, _, _, days = SCHEDULE[fixture - 1]
    return deadline(gw) + pd.Timedelta(days=days)


@pytest.fixture
def gameweeks():
    return pd.DataFrame([{"season": SEASON, "gw": gw, "deadline_time": deadline(gw),
                          "deadline_is_approx": False} for gw in range(1, 7)])


@pytest.fixture
def fixtures():
    return pd.DataFrame([dict(season=SEASON, fixture=f, gw=gw, kickoff_time=kickoff(f),
                              finished=gw <= PLAYED_GWS, team_h_code=h, team_a_code=a)
                         for f, gw, h, a, _ in SCHEDULE])


def match(player, fixture, team, position, minutes=90, **stats):
    """One player_match row; every stat not given is 0."""
    _, gw, h, a, _ = SCHEDULE[fixture - 1]
    row = dict.fromkeys(COLUMNS, 0)
    row.update(season=SEASON, player_code=player, element=player, fixture=fixture, gw=gw,
               kickoff_time=kickoff(fixture), team_code=team, opponent_code=a if team == h else h,
               was_home=team == h, position=position, minutes=minutes, starts=int(minutes > 0))
    row.update(stats)
    return row


@pytest.fixture
def player_match():
    rows = []
    for f in (1, 2, 3):
        rows += [
            match(10, f, 1, "MID", expected_goals=0.3, expected_assists=0.2, goals_scored=int(f == 2),
                  bonus=2 * int(f == 2), defensive_contribution=8),
            match(11, f, 1, "DEF", expected_goals=0.05, defensive_contribution=11,
                  clean_sheets=int(f != 1), goals_conceded=int(f == 1), bonus=int(f == 3)),
            match(12, f, 1, "GK", saves=3, clean_sheets=int(f != 1), goals_conceded=int(f == 1)),
            match(21, f, 2, "MID", minutes=[90, 30, 0][f - 1], expected_goals=0.1,
                  defensive_contribution=5, yellow_cards=int(f == 1)),
            match(22, f, 2, "FWD", expected_goals=0.1, expected_assists=0.1),
        ]
    # Player 20: one match only (GW3), 2 xG in 90 minutes. The one-week wonder.
    rows.append(match(20, 3, 2, "FWD", expected_goals=2.0, goals_scored=2, bonus=3))
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture
def roster():
    # Player 30 is a new signing for team 1 with no history at all.
    return pd.DataFrame({"player_code": [10, 11, 12, 20, 21, 22, 30],
                         "team_code": [1, 1, 1, 2, 2, 2, 1],
                         "position": ["MID", "DEF", "GK", "FWD", "MID", "FWD", "MID"]})


def run(player_match, gameweeks, fixtures, roster, players_daily=None):
    """The whole chain, as build_all runs it: player_gw -> player_fixture -> predictions."""
    player_gw = build_player_gw.build(player_match, gameweeks)
    pf = build_player_fixture.build(player_match, player_gw, fixtures, roster, gameweeks,
                                    players_daily, season=SEASON)
    return pf, predict_fixtures(pf, gameweeks, load_scoring())


@pytest.fixture
def predicted(player_match, gameweeks, fixtures, roster):
    return run(player_match, gameweeks, fixtures, roster)


def one(df, **keys):
    hit = df
    for k, v in keys.items():
        hit = hit[hit[k] == v]
    assert len(hit) == 1, f"expected one row for {keys}, got {len(hit)}"
    return hit.iloc[0]


def upcoming_table(fx, roster, first_gw=4):
    future = fx[fx["is_future"]]
    return upcoming(to_gameweeks(future), roster, SEASON, first_gw=first_gw, last_gw=6)


# --- player_fixture ---------------------------------------------------------

def test_player_fixture_one_row_per_player_per_fixture(predicted):
    pf, _ = predicted
    assert not pf.duplicated(["season", "player_code", "fixture"]).any()
    # Future: every roster player x his club's unplayed fixtures (team 1: f4, f5, f7).
    assert sorted(pf.loc[pf["player_code"] == 10, "fixture"]) == [1, 2, 3, 4, 5, 7]
    assert pf.loc[pf["is_future"], "minutes"].isna().all()


def test_double_shares_gameweek_features_and_has_own_opponent(predicted):
    pf, _ = predicted
    dgw = pf[(pf["player_code"] == 10) & (pf["gw"] == 4)].sort_values("kickoff_time")
    assert dgw["n_fixtures_gw"].tolist() == [2, 2]
    assert dgw["opponent_code"].tolist() == [2, 3]
    assert dgw["home"].tolist() == [True, False]
    assert dgw["rest_days_before"].tolist() == [7, 3]      # GW3 kickoff -> f4 -> f5
    f_cols = [c for c in pf.columns if c.startswith("f_")]
    pd.testing.assert_series_equal(dgw[f_cols].iloc[0], dgw[f_cols].iloc[1], check_names=False)


def test_future_features_are_the_state_after_the_last_played_gw(predicted):
    pf, _ = predicted
    # Player 21: 90, 30, 0 minutes in GW1-3 -> appears 2 of 3, 60+ 1 of 3, every future GW.
    future = pf[(pf["player_code"] == 21) & pf["is_future"]]
    assert future["f_appear_rate_5"].tolist() == pytest.approx([2 / 3] * len(future))
    assert future["f_60_rate_5"].tolist() == pytest.approx([1 / 3] * len(future))
    assert future["f_mins_per_match_5"].tolist() == pytest.approx([40] * len(future))


# --- components -------------------------------------------------------------

def test_components_sum_to_xp_exactly(predicted, roster):
    _, fx = predicted
    out = upcoming_table(fx, roster)
    assert (out[COMPONENTS].sum(axis=1) == out["xp"]).all()
    assert out[COMPONENTS + ["xp"]].notna().all().all()


def test_double_counts_both_fixtures(predicted, roster):
    _, fx = predicted
    out = upcoming_table(fx, roster)
    per_fixture = fx[(fx["player_code"] == 10) & (fx["gw"] == 4)]
    assert len(per_fixture) == 2 and (per_fixture["xp_appear"] > 0).all()
    dgw = one(out, player_code=10, gw=4)
    assert dgw["n_fixtures"] == 2
    assert dgw["xp"] == pytest.approx(per_fixture[COMPONENTS].to_numpy().sum())
    assert dgw["xp"] > one(out, player_code=10, gw=6)["xp"] * 1.5    # two matches beat one
    assert list(dgw["opponent"]) == [2, 3]


def test_blank_is_zero(predicted, roster):
    _, fx = predicted
    blank = one(upcoming_table(fx, roster), player_code=10, gw=5)
    assert blank["n_fixtures"] == 0
    assert blank["xp"] == 0 and (blank[COMPONENTS] == 0).all()
    assert list(blank["opponent"]) == []


def test_no_history_predicts_zero(predicted, roster):
    _, fx = predicted
    new = upcoming_table(fx, roster).query("player_code == 30")
    assert (new["xp"] == 0).all() and (new["n_matches"] == 0).all()


def test_position_rules_from_scoring_yaml(predicted, roster):
    _, fx = predicted
    out = upcoming_table(fx, roster)
    mid, gk, fwd = (one(out, player_code=p, gw=6) for p in (10, 12, 20))
    assert mid["xp_saves"] == 0 and mid["xp_conceded"] == 0    # outfield: no saves, MID concedes nothing
    assert gk["xp_saves"] > 0 and gk["xp_conceded"] < 0 and gk["xp_defcon"] == 0
    assert fwd["xp_cs"] == 0                                    # FWD clean sheet = 0 points
    p = mid["p60"]
    assert mid["xp_appear"] == pytest.approx(1 * (mid["p_play"] - p) + 2 * p)


def test_shrink_pulls_a_one_match_player_toward_the_position_mean():
    # 2 xG in his only 90 minutes, position mean 0.2, prior worth 6 matches:
    # (2 + 6 x 0.2) / (1 + 6) = 0.457, much nearer 0.2 than 2.0.
    assert shrink(pd.Series([2.0]), pd.Series([90.0]), 0.2, 6).iloc[0] == pytest.approx(3.2 / 7)
    assert shrink(pd.Series([np.nan]), pd.Series([0.0]), 0.2, 6).iloc[0] == pytest.approx(0.2)
    # With lots of minutes the player's own rate wins.
    assert shrink(pd.Series([2.0]), pd.Series([90.0 * 600]), 0.2, 6).iloc[0] == pytest.approx(2.0, rel=0.02)


def test_one_match_wonder_is_shrunk_in_the_pipeline(predicted):
    _, fx = predicted
    wonder = fx[(fx["player_code"] == 20) & fx["is_future"]].iloc[0]
    prior = wonder["prior_xg90"]
    assert prior < wonder["xg90"] < 2.0
    assert wonder["xg90"] - prior < (2.0 - prior) / 2        # pulled more than halfway back


def test_poisson_helpers_match_brute_force():
    lam = np.array([0.0, 0.7, 3.0, 11.0])
    k = np.arange(61)
    log_fact = np.array([math.lgamma(j + 1) for j in k])
    with np.errstate(divide="ignore", invalid="ignore"):   # 0 x log(0) for lambda = 0
        pmf = np.exp(k * np.log(lam[:, None]) - lam[:, None] - log_fact)
    pmf[0] = (k == 0)                                       # lambda = 0: X is always 0
    assert expected_floor_div(lam, 3) == pytest.approx((pmf * (k // 3)).sum(axis=1))
    thr = np.array([10, 10, 12, 12])
    assert prob_at_least(lam, thr) == pytest.approx((pmf * (k >= thr[:, None])).sum(axis=1))


def test_availability_next_gameweek_only_and_u_for_all(player_match, gameweeks, fixtures, roster):
    snap_time = deadline(3) + pd.Timedelta(days=3)           # after the GW3 deadline: about GW4
    daily = pd.DataFrame({"season": SEASON, "player_code": [10, 21], "snapshot_time": snap_time,
                          "status": ["d", "u"], "chance_of_playing_next_round": [50.0, 0.0],
                          "news": "", "now_cost": 50, "selected_by_percent": 1.0, "ep_next": 2.0})
    _, base = run(player_match, gameweeks, fixtures, roster)
    _, flagged = run(player_match, gameweeks, fixtures, roster, daily)
    b, f = (upcoming_table(x, roster) for x in (base, flagged))
    assert one(f, player_code=10, gw=4)["p_play"] == pytest.approx(0.5 * one(b, player_code=10, gw=4)["p_play"])
    assert one(f, player_code=10, gw=6)["xp"] == pytest.approx(one(b, player_code=10, gw=6)["xp"])
    assert (f.loc[f["player_code"] == 21, "xp"] == 0).all()   # left the club: 0 throughout


# --- the leakage test -------------------------------------------------------

PREDICTIONS = COMPONENTS + ["p_play", "p60", "exp_mins", "lambda_g", "lambda_a", "p_cs",
                            "p_defcon", "xg90", "xa90"]


@pytest.mark.parametrize("gw", [1, 2, 3])
def test_no_leakage_perturbation(player_match, gameweeks, fixtures, roster, gw):
    """Make every outcome in one gameweek absurd and rerun the whole chain.
    No prediction for that gameweek or earlier may change: a prediction is
    made at the deadline, before the matches. Covers the player features AND
    every constant estimated in predict_components (team ratings, league
    average, home factor, position priors, bonus coefficients)."""
    keys = ["season", "player_code", "fixture"]
    _, before = run(player_match, gameweeks, fixtures, roster)

    poisoned = player_match.copy()
    poisoned.loc[poisoned["gw"] == gw, OUTCOMES] = 999
    _, after = run(poisoned, gameweeks, fixtures, roster)

    before, after = (x.set_index(keys).sort_index() for x in (before, after))
    past = before["gw"] <= gw
    pd.testing.assert_frame_equal(before.loc[past, PREDICTIONS], after.loc[past, PREDICTIONS])
    # Sanity: the poison must reach some later prediction, or the test proves nothing.
    assert not np.allclose(before.loc[~past, PREDICTIONS].fillna(-1),
                           after.loc[~past, PREDICTIONS].fillna(-1))


# --- real data --------------------------------------------------------------

XP_COMPONENTS = Path("data/processed/xp_components.parquet")


@pytest.mark.skipif(not XP_COMPONENTS.exists(), reason="run python src/build_all.py first")
def test_real_xp_components_is_complete_and_sums():
    out = pd.read_parquet(XP_COMPONENTS)
    baseline = pd.read_parquet("data/processed/xp.parquet")
    keys = ["player_code", "gw"]
    assert not out.duplicated(keys).any()
    assert set(map(tuple, out[keys].to_numpy())) == set(map(tuple, baseline[keys].to_numpy()))
    assert (out[COMPONENTS].sum(axis=1) == out["xp"]).all()
    assert out[COMPONENTS + ["xp"]].notna().all().all()
    assert (out.loc[out["n_fixtures"] == 0, "xp"] == 0).all()
    probs = out[["p_play", "p60", "p_cs", "p_defcon"]].stack()
    assert ((probs >= 0) & (probs <= 1)).all()
