"""Wildcard optimizer on a hand-built pool of 26 players.

Two kinds of test:
  - check_plan / score_plan: written already, they run now.
  - solve_wildcard: the spec for your MILP. They skip until it stops raising
    NotImplementedError, then they must all pass.
"""
import numpy as np
import pandas as pd
import pytest

from wildcard import BUDGET, Plan, check_plan, score_plan, solve_wildcard

GWS = [10, 11]
POSITIONS = ["GK"] * 4 + ["DEF"] * 8 + ["MID"] * 8 + ["FWD"] * 6
# The first 2 GK, 5 DEF, 5 MID, 3 FWD form a legal, cheap squad (see `legal_plan`).
HAND_SQUAD = [1, 2, 5, 6, 7, 8, 9, 13, 14, 15, 16, 17, 21, 22, 23]


@pytest.fixture
def players():
    """26 players over 8 clubs. Club = position in the list mod 8, so the hand
    squad has at most 3 from any club. Hand squad costs 45 each, others 40-99."""
    codes = list(range(1, len(POSITIONS) + 1))
    price = [45 if c in HAND_SQUAD else 40 + (c * 13) % 60 for c in codes]
    return pd.DataFrame({"position": POSITIONS, "team_code": [(c - 1) % 8 + 1 for c in codes],
                         "price": price, "web_name": [f"p{c}" for c in codes]},
                        index=pd.Index(codes, name="player_code"))


@pytest.fixture
def xp_wide(players):
    rng = np.random.default_rng(0)
    values = rng.uniform(0, 8, size=(len(players), len(GWS))).round(2)
    return pd.DataFrame(values, index=players.index, columns=GWS)


@pytest.fixture
def legal_plan():
    xi = [1, 5, 6, 7, 8, 13, 14, 15, 16, 21, 22]          # 1 GK, 4 DEF, 4 MID, 2 FWD
    return Plan(squad=list(HAND_SQUAD), starters={gw: list(xi) for gw in GWS},
                captain={gw: 13 for gw in GWS})


def solve(players, xp_wide, **kwargs):
    try:
        return solve_wildcard(players, xp_wide, **kwargs)
    except NotImplementedError:
        pytest.skip("solve_wildcard not written yet")


# --- the checker (runs now) ------------------------------------------------------

def test_hand_plan_is_legal(players, legal_plan):
    assert check_plan(legal_plan, players, GWS) == []


def _sixteen(plan, players):
    plan.squad.append(3)
    return plan, players, BUDGET

def _wrong_quota(plan, players):
    plan.squad[plan.squad.index(17)] = 10            # a MID out, a DEF in: 6 DEF, 4 MID
    return plan, players, BUDGET

def _four_from_one_club(plan, players):
    players.loc[[5, 6, 7, 8], "team_code"] = 99
    return plan, players, BUDGET

def _over_budget(plan, players):
    return plan, players, 15 * 45 - 1

def _two_keepers_start(plan, players):
    plan.starters[10] = [1, 2, 5, 6, 7, 13, 14, 15, 16, 21, 22]
    return plan, players, BUDGET

def _captain_on_bench(plan, players):
    plan.captain[11] = 9
    return plan, players, BUDGET

def _starter_not_in_squad(plan, players):
    plan.starters[11] = [3 if p == 1 else p for p in plan.starters[11]]
    return plan, players, BUDGET


@pytest.mark.parametrize("break_it", [_sixteen, _wrong_quota, _four_from_one_club, _over_budget,
                                      _two_keepers_start, _captain_on_bench, _starter_not_in_squad])
def test_checker_catches_each_rule(players, legal_plan, break_it):
    plan, pl, budget = break_it(legal_plan, players.copy())
    assert check_plan(plan, pl, GWS, budget=budget), f"{break_it.__name__} was not caught"


def test_score_by_hand():
    pl = pd.DataFrame(index=pd.Index([1, 2], name="player_code"))
    xp = pd.DataFrame({10: [4.0, 1.0], 11: [2.0, 3.0]}, index=pl.index)
    plan = Plan(squad=[1, 2], starters={10: [1], 11: [1]}, captain={10: 1, 11: 1})
    # GW10: 4 + 0.1*1 + 4 (captain) = 8.1.  GW11: (2 + 0.1*3 + 2) * 0.9 = 3.87
    assert score_plan(plan, xp, delta=0.9, bench_weight=0.1) == pytest.approx(8.1 + 3.87)


# --- the solver (skips until written) --------------------------------------------

def test_solution_is_legal(players, xp_wide):
    plan = solve(players, xp_wide)
    assert check_plan(plan, players, GWS) == []


def test_solution_is_legal_on_a_tight_budget(players, xp_wide):
    budget = 15 * 45 + 30                             # only just above the cheapest legal squad
    plan = solve(players, xp_wide, budget=budget)
    assert check_plan(plan, players, GWS, budget=budget) == []


def test_beats_the_hand_built_squad(players, xp_wide, legal_plan):
    """Optimal means no legal plan scores more. The hand plan is one legal plan."""
    plan = solve(players, xp_wide)
    assert score_plan(plan, xp_wide) >= score_plan(legal_plan, xp_wide) - 1e-6


def test_cheap_star_is_bought_and_captained(players, xp_wide):
    xp = xp_wide.copy()
    xp.loc[20] = 30.0                                 # a 40-something MID who scores 30 a week
    players = players.copy()
    players.loc[20, "price"] = 40
    plan = solve(players, xp)
    assert 20 in plan.squad
    assert all(plan.captain[gw] == 20 for gw in GWS)


def test_captain_is_the_best_starter_each_week(players, xp_wide):
    plan = solve(players, xp_wide)
    for gw in GWS:
        best = xp_wide.loc[plan.starters[gw], gw].idxmax()
        assert xp_wide.at[plan.captain[gw], gw] == pytest.approx(xp_wide.at[best, gw])


def test_club_limit_binds(players, xp_wide):
    """Club 1's five DEFs are the best defenders by far: only 3 can come."""
    players = players.copy()
    xp = xp_wide.copy()
    defs = players.index[players["position"] == "DEF"][:5]
    players.loc[defs, "team_code"] = 1
    others_at_club_1 = players.index[players["team_code"] == 1].difference(defs)
    players.loc[others_at_club_1, "team_code"] = 2
    xp.loc[defs] = 20.0
    plan = solve(players, xp)
    assert len(set(plan.squad) & set(defs)) == 3
