# INTERVIEW.md: what to know cold

Notes for talking about this project in quant, ML engineering and data science interviews. Every number here comes from `DECISIONS.md` or `DATA.md`; check there before quoting it.

**Labels**
- ⭐ **Important to know for interviews.** Be able to explain it without notes.
- ✅ Built: you can say you did it.
- 🔜 Planned: say "the next step is…", never "I built…".

---

## The 30-second pitch ⭐

> I'm building a Fantasy Premier League engine in three stages: predict every player's expected points for the next 6 gameweeks, pick the best 15-man squad with a mixed-integer program, then plan transfers for my real team over a rolling horizon with transaction costs. The hard part isn't the model, it's the data: the FPL API only shows the present, so I snapshot it daily to get point-in-time features, and I test for leakage automatically.

## Numbers to memorize ⭐

| What | Number | Where it came from |
| --- | --- | --- |
| Rows in the baseline comparison | 32,068 player-gameweeks (2024-25 + 2025-26) | DECISIONS 2026-10-14 |
| FPL's own prediction (`fpl_xp`) | MAE **0.855**, within-position Spearman **0.759** | The bar to beat |
| Last-5 average baseline | MAE **1.079**, Spearman **0.705** | The floor |
| v0 xP (walk-forward, 1 GW ahead) | MAE **1.071**, Spearman **0.705** | Reproduces the floor, so the code is right |
| Horizon decay, Spearman | **0.746** (1 GW ahead) → **0.639** (6 GWs ahead), about −14% | Justifies the discount δ ≈ 0.85–0.9 |
| MAE on players who actually played | **2.23** vs 1.04 on all rows | Most rows are "predicted 0, scored 0" |
| Treating unrecorded `fpl_xp` as NaN, not 0 | Pearson 0.51 → **0.73** (+0.22) | One data-cleaning decision, big effect |
| Zero-minute match rows (2026-27) | **52%** | Why minutes is modelled as P(start), not a regression |
| Doubles | 1.3% of player-gameweeks | Rare, but they're when chips and hits get played |

---

## 1. Point-in-time data ✅ ⭐

**What:** a GitHub Actions job saves the FPL API every day at 07:17 UTC and commits it (`src/snapshot.py`, `.github/workflows/snapshot.yml`). Raw data is never edited. Every table is rebuilt from it (`src/build_all.py`).

**Why:** the API only shows *now*. Yesterday's price, ownership and injury flags are gone unless you saved them. Without snapshots you can't know what a manager could have known at a past deadline.

**Interview line:** "Point-in-time correctness is the same problem as survivorship and look-ahead bias in a backtest. I fixed it at the source by recording the state myself."

**Detail worth knowing:** snapshot features are joined with `pd.merge_asof` (backward) using the latest snapshot **strictly before** the deadline. A later one leaks: FPL's `ep_next` switches to the next gameweek the moment a deadline passes (Haaland went 7.5 → 5.0 on Oct 10).

## 2. Leakage prevention ✅ ⭐

**The rule:** a feature for gameweek t may only use data from before the GW t deadline. Rolling form is shifted by one match, so a match never sees itself.

**The test I'm proudest of** (`tests/test_player_gw.py::test_no_leakage_perturbation`):
1. Build the features.
2. Set every outcome in one gameweek to 999 and rebuild.
3. Assert every feature at or before that gameweek is unchanged.
4. Sanity check: assert something *after* it did change, or the test proves nothing.

**Why it's good:** it tests the behaviour, not the implementation. It catches leaks however the feature was written, including ones nobody thought of. The same test exists for the prediction (`tests/test_predict_baseline.py`).

**Leaks I found and documented** (DATA.md):
- The `fixtures` table holds today's schedule, so a backtest "knows" about a rescheduled double before it was announced.
- `players.team_code` is today's club, so a January signing would be predicted at his new club in August.
- Season totals and end-of-season prices can't be used as features for past gameweeks.

## 3. Data engineering choices ✅

- **Stable keys:** join seasons on `player_code` / `team_code`. FPL renumbers `element` and team `id` every season. Joining on those silently mixes up players.
- **Grain:** one row per player per gameweek (`player_gw`). Blanks are added back as rows with `n_fixtures = 0`. Doubles are summed. Club and position are forward-filled, never back-filled (back-filling copies a later transfer into the past).
- **Missing ≠ zero:** DefCon is NaN for 2024-25 (not recorded), never 0. The +0.22 Pearson result above is the payoff.
- **Reproducible:** "now" = the latest snapshot, not the wall clock. The same raw data always rebuilds the same tables.
- **Automated checks:** 40 pytest tests (synthetic + real data) run daily after the rebuild. A red run emails me.

## 4. Evaluation ✅ ⭐

**Walk-forward:** to predict GW t, use only GWs 1..t−1. Never a random train/test split: rows from the same player and season are correlated, and a random split lets the model see the future.

**Why these metrics:**
- **MAE:** points are heavy-tailed (hauls), so absolute error is more robust than squared error.
- **Within-position Spearman:** rank correlation inside each (season, gameweek, position), averaged. The optimizer only cares about *ranking* players it could swap, and squad quotas mean you swap a defender for a defender. Positions score differently, so a pooled correlation would be inflated.

**Honest caveat to volunteer:** most rows are fringe players (predicted ~0, scored 0), which flatters every metric. MAE doubles to 2.23 on players who played. Always report both.

**Baselines before models:** I set the bar (FPL's own prediction) and the floor (last-5 average) before building anything, so every model result has context.

## 5. v0 expected points ✅

`src/predict_baseline.py`: **xp = points per match (last 5) × n_fixtures × availability**

| Choice | Why |
| --- | --- |
| Per match, not per gameweek | A double shouldn't inflate the rate |
| × `n_fixtures` | A double = 2×, a blank = 0. That's the one thing a 6-GW plan must get right |
| Availability on the **next** GW only | The injury flag is about next week. Applying it to all 6 writes off a player out for one match |
| Status `u` → 0 for all 6 | He's left the club |

**Why build something this crude?** To get all three stages running end to end first, then improve the weakest piece. It also becomes the baseline Phase 1 has to beat.

---

## 6. The wildcard optimizer (MILP) 🔜 ⭐⭐

The most important section for quant interviews. Next to build: `src/wildcard.py` with PuLP (HiGHS/CBC solver).

### The formulation

Decision variables, all binary (0 or 1):
- x_i = 1 if player i is in the 15-man squad
- s_i,t = 1 if player i starts in gameweek t
- c_i,t = 1 if player i is captain in gameweek t

Objective (maximize):

  Σ_t δ^(t−1) · Σ_i xp_i,t · ( s_i,t + c_i,t + ε · (x_i − s_i,t) )

- s_i,t counts a starter's points once. c_i,t adds them again: the captain scores double.
- ε ≈ 0.1 gives bench players a little value, so the bench isn't pure filler.
- δ ≈ 0.85–0.9 discounts later gameweeks because they're less certain. My horizon-decay numbers back this up (Spearman 0.746 → 0.639).

Constraints:
- Budget: Σ price_i · x_i ≤ bank + selling value
- Squad: exactly 2 GK, 5 DEF, 5 MID, 3 FWD
- At most 3 players per club
- Each gameweek: 11 starters, 1 GK, 3–5 DEF, 2–5 MID, 1–3 FWD
- s_i,t ≤ x_i (only squad players start), c_i,t ≤ s_i,t (the captain must start), Σ_i c_i,t = 1

### Questions to be ready for ⭐

**"Why not just sort by points per £ and pick greedily?"**
Ratio greedy is optimal for the *fractional* knapsack, where you can buy half an item. Here you can't buy half a player, and there are position quotas and club limits on top of the budget. Greedy leaves budget unspent, or hits a club limit and has to make a bad substitution. The quotas couple the choices: the best set isn't the set of individually best picks.

**"Why not try every squad?"**
About 670 players choose 15 is astronomically many. A MILP solver uses **branch and bound**: solve the LP relaxation (let variables be fractional), which gives an upper bound, then branch on a fractional variable and prune any branch whose bound is worse than the best squad found so far. It proves optimality without enumerating.

**"Why linear?"**
Everything is a sum of xp × a binary variable, so the problem stays linear and exact solvers work. Captaincy looks nonlinear (double one player's points), but splitting it into a separate c variable keeps it linear.

**"What's a shadow price here?"**
The value of one more unit of a constraint: how many expected points another £0.5m buys. In a MILP there are no clean LP duals, so I re-solve with the budget ±£0.5m and take the difference. That tells you whether money in the bank is worth anything.

**"What's the optimizer's curse?"** ⭐⭐
Every xp is an estimate with noise. The optimizer picks the players with the *highest estimates*, and those are disproportionately the ones whose noise was positive. So the chosen squad's predicted points are biased upward, and it underperforms its own xP. My v0 table already shows it: the top of the table is hot streaks (one player at 7.8 points per match → 46.8 over 6 GWs).
Fixes: shrink estimates toward a position average (empirical Bayes, the same idea as James–Stein), or penalize uncertainty in the objective (needs an xp standard deviation, Phase 1).
**Quant link:** this is why portfolio managers shrink expected returns before mean–variance optimization. Raw estimates produce extreme, overconfident portfolios.

**"How do you evaluate the optimizer?"**
On *actual* points the squad went on to score, in a walk-forward backtest. Never on its own xP: that's circular and biased upward by the curse above.

**"How do you know the solver's answer is legal?"** ✅ A separate checker (`check_plan` in `src/wildcard.py`) re-checks every rule in plain Python, independently of the MILP. The tests break each rule one at a time and confirm the checker catches it. If a constraint is missing from the model, the checker fails the test, so the solver never gets to police itself.

**Exercises planned:** compare against greedy, the ±£0.5m shadow price, and measure the gap between predicted and actual points of the chosen squads.

## 7. The transfer planner 🔜 ⭐

Phase 2 plus time. Variables per gameweek: squad x_i,t, transfers in_i,t / out_i,t, hits h_t.
- Squad flow: x_i,t = x_i,t−1 + in_i,t − out_i,t
- Free transfers bank up to 5. Each extra transfer costs 4 points.
- Selling price = purchase price + half of any rise, rounded down to £0.1m.
- **Rolling horizon:** plan 4–6 GWs, execute only this week, re-plan next week with new information. This is the same idea as model predictive control.

**Quant link:** this is portfolio rebalancing with transaction costs. A transfer is only worth it if the expected gain over the horizon beats the cost (a hit = −4 points), the same way a trade must beat its fees and slippage. Doing nothing is often optimal.

## 8. The decomposed points model 🔜 ⭐

Don't predict points directly. Predict the pieces and apply the scoring rules from a config file:

E[pts] = 2·p60 + (p_play − p60) + E[mins/90]·(G_pos·λ_g + 3·λ_a) + p60·C_pos·P(CS) + 2·P(DefCon) + E[bonus] − E[penalties]

**Why:**
- Each piece is more predictable than total points, and each has its own good model: minutes (logistic regression → LightGBM), team strength (Poisson regression on xG, the Dixon–Coles idea, P(CS) = exp(−λ_against)), player shares of team xG shrunk to position averages (empirical Bayes).
- **Rule changes become config edits.** DefCon points arrived in 2025-26 and bonus was reworked for 2026-27. A model trained on total points would have learned the old rules.
- Minutes is the most important piece: 52% of match rows are 0 minutes, so it's zero-inflated, and modelled as P(start) then P(60+ | start).

---

## Weaknesses to volunteer (they make you sound senior)

- v0 xP is only as good as the floor baseline; FPL's own prediction is still much better (0.855 vs 1.071 MAE).
- No uncertainty (standard deviation) on xp yet, which the optimizer's-curse fix needs.
- Availability flags only exist from 2026-10-07, so the injury signal can't be backtested.
- The backtest's fixture schedule isn't point-in-time for past seasons.
- Missed my own Nov 8 gate for all stages running, by about two weeks. The plan has a cut list for this (bonus model, chips, alternative suggestions, the app), and the backtest is never cut.

## Keeping this file current

When a 🔜 item is built, flip it to ✅ and add its real numbers from `DECISIONS.md`.
