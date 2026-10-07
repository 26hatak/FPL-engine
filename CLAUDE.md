# FPL Engine

A Fantasy Premier League engine built in three stages, each feeding the next:

1. **Points prediction**: expected points (xP) for every player for the next 6 gameweeks.
2. **Wildcard optimizer**: the best 15-man squad from those predictions (mixed-integer programming).
3. **Transfer planner**: suggested transfers for my actual team over a 4–6 gameweek horizon.

Full project plan (phases, formulas, timeline, resources):
https://claude.ai/code/artifact/552b9df2-e6af-4f4f-b688-02f3eacee9cf

## How to help me

I'm a student building this to learn skills I can talk about in internship interviews (quant, ML engineering, data science). So:

- **Explain the why** behind each decision, not just the code. I need to be able to defend every line in an interview.
- **Let me try first** when a step is a good learning exercise. Give hints or a skeleton before a full solution, unless I ask for the full thing.
- **Keep it simple.** Build the crude end-to-end version of all three stages first, then improve the weakest piece.
- **Flag leakage risks** whenever we touch features, joins or evaluation.
- I'm new to some tooling (git, terminals, virtual environments, GitHub Actions). Give exact commands and say what each one does.

## Current status (update this as we go)

Phase 0 (data), target finish Oct 25, 2026:

- [x] `src/snapshot.py` saves the FPL API daily; GitHub Actions job `.github/workflows/snapshot.yml` runs it every day at 07:17 UTC and commits the data (confirmed working).
- [x] Backfilled the 2026/27 season so far (`python src/snapshot.py --backfill`).
- [ ] Explore the data in `notebooks/01_explore.ipynb` and write `DATA.md` (in progress).
- [ ] Download past seasons (2024-25, 2025-26) from https://github.com/vaastav/Fantasy-Premier-League
- [ ] Build `player_gw` table + a test that fails on leakage.

Open questions for `DATA.md`:
1. Which file gives player name, position, team? (gameweek files only have IDs)
2. What units is price stored in? (`now_cost` is tenths of £m)
3. Double gameweeks: one row or two per player?
4. Which fields are outcomes vs. pre-deadline features?
5. Does `element_summary` history include xG/xA per match?

## Repo layout

```
fpl-engine/
├── .github/workflows/snapshot.yml   daily data job
├── src/snapshot.py                  API snapshot + backfill script
├── notebooks/                       exploration only
├── data/raw/                        raw API data, gzipped JSON (never edit)
│   ├── daily/<UTC timestamp>/       bootstrap-static + fixtures, one folder per day
│   ├── gameweeks/gwNN_live.json.gz  every player's stats per finished gameweek
│   └── element_summary/<date>/      per-player season history (incl. price, ownership per GW)
├── data/processed/                  Parquet tables built from raw (to do)
├── DATA.md                          field notes (to do)
└── .gitignore                       .venv/, __pycache__/, .DS_Store
```

## Conventions

- **Raw data is immutable.** Everything downstream is rebuilt from `data/raw/`.
- **Python environment:** `.venv` (`source .venv/bin/activate`). Install packages there, never globally.
- **Paths in notebooks:** notebooks run from `notebooks/`, so build paths from the project root:
  `ROOT = Path.cwd().parent; RAW = ROOT / "data" / "raw"`.
- **Always `git pull` before working**: the snapshot bot commits every day.
- **Storage:** Parquet + DuckDB for processed tables.
- **Tools planned:** pandas, DuckDB, scikit-learn / LightGBM, statsmodels, PuLP (HiGHS/CBC solver), Streamlit.

## FPL API endpoints (base: https://fantasy.premierleague.com/api/)

| Endpoint | Contents |
| --- | --- |
| `bootstrap-static/` | all players (price, position, team, status, chance of playing, ownership, `ep_next`), teams, gameweeks |
| `element-summary/{id}/` | one player's GW history this season + upcoming fixtures |
| `event/{gw}/live/` | every player's stats for one gameweek |
| `fixtures/` | all fixtures, kickoffs, difficulty |
| `entry/{team_id}/`, `entry/{team_id}/event/{gw}/picks/` | my squad, bank, transfers |

The API only shows the current state, which is why daily snapshots matter: past prices, ownership and injury flags are otherwise lost. Be polite: one request at a time, ~1 s apart.

## The plan, phase by phase

**Phase 0: Data.** One core table `player_gw`: one row per player per gameweek, features as known *before that deadline*, plus actual outcomes (minutes, goals, assists, clean sheet, points). Rule: a feature for GW t may only use data from before the GW t deadline. Rolling form features must be shifted by one match per player so a match never sees itself.

**Phase 1: Points prediction.** Don't predict points directly; predict the pieces and apply the scoring rules from a `scoring.yaml` config:

E[pts] = 2·p60 + (p_play − p60) + E[mins/90]·(G_pos·λ_g + 3·λ_a) + p60·C_pos·P(CS) + 2·P(DefCon) + E[bonus] − E[penalties]

Sub-models, in build order:
1. Minutes model: P(start), P(60+ | start). Logistic regression, then LightGBM. Most important piece.
2. Team strength: Poisson regression on xG for/against (Dixon–Coles idea). P(CS) = exp(−λ_against).
3. Player shares of team xG/xA, shrunk toward position averages (empirical Bayes).
4. Defensive contributions: per-90 counts (Poisson / negative binomial) → P(crossing threshold).
5. Bonus: crude average given goals/assists/CS; improve last.

Output: `xp` table, mean + standard deviation per player per GW for the next 6 GWs.
Baselines to beat: last-5-match average, then FPL's own `ep_next`.

**Phase 2: Wildcard optimizer (MILP).** Variables: x_i (in squad), s_i,t (starts GW t), c_i,t (captain GW t).
Maximize Σ_t δ^(t−1) Σ_i xp_i,t · (s_i,t + c_i,t + ε·(x_i − s_i,t)), with δ ≈ 0.85–0.9 and ε ≈ 0.1.
Constraints: budget ≤ bank + selling value; 2 GK / 5 DEF / 5 MID / 3 FWD; ≤ 3 per club; each GW 11 starters (1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD); one captain who starts.
Exercises: compare to greedy; re-solve with ±£0.5m (shadow price); watch the optimizer's curse (picked squads underperform their xP; fix with shrinkage or a std penalty).
xP is the optimizer's *input*. Picks are *evaluated* on actual points in Phase 4, never on their own xP (circular and biased upward).

**Phase 3: Transfer planner.** Phase 2 plus time: x_i,t, in_i,t, out_i,t, hits h_t. Squad flow x_i,t = x_i,t−1 + in − out. Free transfers bank up to 5; each extra transfer costs 4 points. Selling price = purchase price + half of any rise, rounded down to £0.1m. Rolling horizon: plan 4–6 GWs, execute only this week, re-plan next week. Show top 3 options by re-solving with constraints excluding previous answers. Chips last (2026/27 has two chip sets; first set expires at the GW19 deadline; confirm in official rules).

**Phase 4: Evaluation.** Walk-forward (train GWs 1..t−1, predict t). Metrics: MAE per position, within-position Spearman, calibration plots for P(start) and P(CS). Backtest past seasons with point-in-time data only; compare against an `ep_next`-driven planner and a most-owned "template" team. Ablations: remove one component at a time. Watch for: season-total stats as features, end-of-season prices, dropped players, rule changes across seasons (DefCon points from 2025/26; bonus system reworked for 2026/27), tuning on the reported season.

**Phase 5: Shipping.** Weekly scheduled pipeline (fetch → features → predict → optimize); Streamlit app (enter team ID → transfers, xP table, wildcard squad); tests that every returned squad is legal; weekly public decision log; final write-up with calibration, backtest, ablations, and what failed.

## Timeline

| Dates (2026–27) | Work |
| --- | --- |
| Oct 12 – Oct 25 | Phase 0: data + snapshots |
| Oct 26 – Nov 8 | End-to-end v0 (baseline xP + wildcard MILP) — **gate Nov 8: all stages run** |
| Nov 9 – Nov 29 | Points model v1 (minutes, team strength, shares) |
| Nov 30 – Dec 13 | Transfer planner |
| Dec 14 – Dec 22 | Exams (light) |
| Dec 23 – Jan 10 | Backtest + ablations |
| Jan 4 – Jan 24 | App + write-up — **public launch Jan 24** |

If behind, cut in this order: bonus model, chips, alternative suggestions, the app. Never cut the backtest.

## FPL 2026/27 rule notes

- Defensive contribution (DefCon) points carried over unchanged from 2025/26.
- Bonus Points System reworked (no penalty for being tackled; CBI bonus now per 3 instead of per 2; goalkeeper save bonuses changed).
- Chips in two sets; Assistant Manager chip removed.
- Up to 5 free transfers can be banked.
- Always check https://fantasy.premierleague.com/help/rules before hard-coding a value.

## Interview angles to keep in mind

- Leakage prevention (point-in-time features, walk-forward evaluation).
- Decomposed points model and why (predictability, rule changes become config edits).
- Optimizer's curse = why quants shrink forecasts before portfolio optimization.
- Transfer planner = rebalancing with transaction costs.
- Record real numbers from ablations and backtests in `DECISIONS.md` as we go (what was tried, why, what happened).
