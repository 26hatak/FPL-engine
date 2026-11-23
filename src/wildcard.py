"""Wildcard optimizer: the best 15-man squad for the next 6 gameweeks (MILP).

Run from the project root: python src/wildcard.py

Stage 2 of the end-to-end v0. Reads data/processed/xp.parquet and today's prices,
then picks the squad, each gameweek's starting XI and captain that maximize

    sum over t of  delta^(t-1) * sum over i of  xp[i,t] * (s[i,t] + c[i,t] + eps * (x[i] - s[i,t]))

    x[i]   = 1 if player i is in the squad              (binary)
    s[i,t] = 1 if player i starts in gameweek t          (binary)
    c[i,t] = 1 if player i is captain in gameweek t      (binary)
    delta  = discount per gameweek ahead (later weeks are less certain:
             within-position Spearman falls 0.746 -> 0.639 from 1 to 6 GWs ahead)
    eps    = weight of bench points (a bench player only scores if a starter drops out)

Split of work (CLAUDE.md "let me try first"):
  - written: constants, load_inputs, check_plan, score_plan, print_plan, main
  - YOU write: solve_wildcard (the MILP). tests/test_wildcard.py is the spec:
    its solver tests skip until solve_wildcard stops raising NotImplementedError.

xp is the optimizer's INPUT. A squad is judged on the points it actually goes on
to score (Phase 4), never on its own xp: that's circular and biased upward
(the optimizer's curse, INTERVIEW.md section 6).
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROCESSED = Path("data/processed")

BUDGET = 1000          # tenths of £m: £100.0m, a fresh wildcard squad
SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
XI_SIZE = 11
XI_LIMITS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}   # (min, max) starters
DELTA = 0.9
BENCH_WEIGHT = 0.1


@dataclass
class Plan:
    """What the optimizer returns. Gameweeks are the columns of xp_wide."""
    squad: list[int]                 # 15 player_codes
    starters: dict[int, list[int]]   # gw -> 11 player_codes
    captain: dict[int, int]          # gw -> player_code (one of that gw's starters)


def load_inputs(processed: Path = PROCESSED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """players: one row per buyable player, indexed by player_code, with
    web_name, team_code, position, price (tenths of £m, from the latest snapshot).
    xp_wide: index player_code, one column per gameweek, values = xp.

    Only players in both xp and the latest snapshot are kept: a player with no
    price can't be bought, and one with no xp can't be valued.
    """
    xp = pd.read_parquet(processed / "xp.parquet")
    daily = pd.read_parquet(processed / "players_daily.parquet")
    names = pd.read_parquet(processed / "players.parquet")

    # Same snapshot xp was made from, so prices and predictions describe the same moment.
    as_of = xp["as_of"].iloc[0]
    prices = (daily[daily["snapshot_time"] == as_of]
              .set_index("player_code")["now_cost"].rename("price"))

    xp_wide = xp.pivot(index="player_code", columns="gw", values="xp")
    info = xp.drop_duplicates("player_code").set_index("player_code")[["team_code", "position"]]
    season_names = names[names["season"] == xp["season"].iloc[0]].set_index("player_code")["web_name"]

    players = info.join(prices, how="inner").join(season_names)
    players["price"] = players["price"].astype(int)
    return players, xp_wide.loc[players.index]


def check_plan(plan: Plan, players: pd.DataFrame, gws: list[int],
               budget: int = BUDGET) -> list[str]:
    """Every FPL rule the plan breaks, in words. An empty list = legal.

    Written as a plain checker, separate from the solver, so the tests don't
    trust the MILP to police itself: if a constraint is missing from the model,
    this catches it.
    """
    problems = []
    squad = list(plan.squad)
    if len(squad) != 15 or len(set(squad)) != 15:
        problems.append(f"squad must be 15 different players, got {len(set(squad))} of {len(squad)}")
    unknown = set(squad) - set(players.index)
    if unknown:
        return problems + [f"unknown players {sorted(unknown)}"]

    sq = players.loc[squad]
    for pos, n in SQUAD.items():
        if (sq["position"] == pos).sum() != n:
            problems.append(f"need {n} {pos}, got {(sq['position'] == pos).sum()}")
    clubs = sq["team_code"].value_counts()
    for club, n in clubs[clubs > MAX_PER_CLUB].items():
        problems.append(f"{n} players from club {club} (max {MAX_PER_CLUB})")
    cost = int(sq["price"].sum())
    if cost > budget:
        problems.append(f"cost {cost} over budget {budget}")

    for gw in gws:
        xi = list(plan.starters.get(gw, []))
        if len(xi) != XI_SIZE or len(set(xi)) != XI_SIZE:
            problems.append(f"GW{gw}: need {XI_SIZE} different starters, got {len(set(xi))}")
        if not set(xi) <= set(squad):
            problems.append(f"GW{gw}: starters not in the squad {sorted(set(xi) - set(squad))}")
            continue
        pos_counts = players.loc[xi, "position"].value_counts()
        for pos, (lo, hi) in XI_LIMITS.items():
            if not lo <= pos_counts.get(pos, 0) <= hi:
                problems.append(f"GW{gw}: {pos_counts.get(pos, 0)} {pos} starting (allowed {lo}-{hi})")
        if plan.captain.get(gw) not in xi:
            problems.append(f"GW{gw}: captain {plan.captain.get(gw)} is not a starter")
    return problems


def score_plan(plan: Plan, xp_wide: pd.DataFrame, delta: float = DELTA,
               bench_weight: float = BENCH_WEIGHT) -> float:
    """The objective at the top of this file, computed by hand for a given plan.
    Lets the tests compare the solver's answer with squads built by hand."""
    total = 0.0
    for t, gw in enumerate(xp_wide.columns):
        xi = set(plan.starters[gw])
        week = 0.0
        for p in plan.squad:
            pts = xp_wide.at[p, gw]
            week += pts if p in xi else bench_weight * pts
        week += xp_wide.at[plan.captain[gw], gw]          # captain counts twice
        total += delta ** t * week
    return total


def solve_wildcard(players: pd.DataFrame, xp_wide: pd.DataFrame, budget: int = BUDGET,
                   delta: float = DELTA, bench_weight: float = BENCH_WEIGHT) -> Plan:
    """YOUR EXERCISE: build and solve the MILP, return a Plan.

    Inputs: players (index player_code; position, team_code, price) and xp_wide
    (index player_code, columns = gameweeks in order). Output: a Plan that
    check_plan accepts and that maximizes score_plan.

    Suggested order. Run `pytest tests/test_wildcard.py -v` after each step:
      1. One gameweek, squad only: binary x[i], maximize sum xp[i] * x[i],
         subject to budget, the 2/5/5/3 quotas and <= 3 per club.
         (Return every squad player as a starter for now: the tests will tell you
         what's still missing.)
      2. Add s[i,t] for every gameweek: 11 starters, the XI_LIMITS formation,
         and s[i,t] <= x[i] (only squad players can start).
      3. Add c[i,t]: one captain per gameweek and c[i,t] <= s[i,t].
      4. Use the full objective with delta and bench_weight.

    PuLP in 6 lines (a toy, not this problem):
        import pulp
        prob = pulp.LpProblem("toy", pulp.LpMaximize)
        y = pulp.LpVariable.dicts("y", ["a", "b", "c"], cat="Binary")
        prob += pulp.lpSum(value[k] * y[k] for k in y)      # first += is the objective
        prob += pulp.lpSum(y[k] for k in y) <= 2            # later += are constraints
        prob.solve(pulp.HiGHS(msg=False)); chosen = [k for k in y if y[k].value() > 0.5]

    PuLP 3.3 warns that LpVariable.dicts is deprecated (PuLP 4 renames it to
    prob.add_variable_dicts). It works fine; requirements.txt pins 3.3.2.

    Check pulp.LpStatus[prob.status] == "Optimal" before reading values, and
    read binaries with `> 0.5`, not `== 1` (solvers return 0.9999999).

    Think about first: why can't you just sort by xp per £m and take the top
    ones? (Answer in INTERVIEW.md section 6, once you've tried.)
    """
    raise NotImplementedError("solve_wildcard: your turn, see the docstring")


def print_plan(plan: Plan, players: pd.DataFrame, xp_wide: pd.DataFrame) -> None:
    sq = players.loc[plan.squad].copy()
    sq["xp_total"] = xp_wide.loc[plan.squad].sum(axis=1)
    first = xp_wide.columns[0]
    sq["role"] = ["C" if p == plan.captain[first] else ("XI" if p in plan.starters[first] else "bench")
                  for p in sq.index]
    order = {pos: k for k, pos in enumerate(SQUAD)}
    sq = sq.sort_values(["position", "xp_total"], key=lambda s: s.map(order) if s.name == "position" else -s)
    print(sq[["web_name", "position", "team_code", "price", "xp_total", "role"]].to_string())
    print(f"cost {sq['price'].sum() / 10:.1f}m   objective {score_plan(plan, xp_wide):.1f}"
          f"   (role = GW{first})")


def main() -> None:
    players, xp_wide = load_inputs()
    plan = solve_wildcard(players, xp_wide)
    problems = check_plan(plan, players, list(xp_wide.columns))
    assert not problems, problems
    print_plan(plan, players, xp_wide)


if __name__ == "__main__":
    main()
