# DATA.md: field notes

What's in `data/raw/`, how the pieces fit together, and the traps. Every claim here was checked against the data on 2026-10-07.

## Sources

| Path | Source | Grain | Covers |
| --- | --- | --- | --- |
| `data/raw/daily/<UTC ts>/bootstrap-static.json.gz` | FPL API, daily job | one file per day | players, teams, positions, gameweeks **as of that day** |
| `data/raw/daily/<UTC ts>/fixtures.json.gz` | FPL API, daily job | one row per fixture | all 380 fixtures, kickoffs, scores, difficulty |
| `data/raw/gameweeks/gwNN_live.json.gz` | FPL API, daily job | one row per player per **gameweek** | finished GWs this season |
| `data/raw/element_summary/<date>/<id>.json.gz` | FPL API, daily job (re-downloaded once a GW finishes) | `history`: one row per player per **match** | this season up to `<date>` |
| `data/raw/vaastav/<season>/*.csv.gz` | [vaastav](https://github.com/vaastav/Fantasy-Premier-League), pinned commit `9779cdb` | `merged_gw`: one row per player per **match** | 2024-25, 2025-26 (all 38 GWs) |

## Processed tables (`data/processed/`, rebuilt by `python src/build_all.py`)

| Table | One row = | Use |
| --- | --- | --- |
| `player_gw` | player × gameweek | Main modelling table: targets, `f_*` form, `snap_*` snapshot state, team, position, `fpl_xp` |
| `player_match` | player × match | Per-match stats; team totals (team strength), shares, per-fixture models |
| `fixtures` | match, **played and future** | Opponent, home/away, kickoff, FPL difficulty, scores once played. Future rows are what predictions are made for |
| `players` | player × season | Names (`web_name`), club and position: lookup by `player_code` |
| `teams` | team × season | Season id → stable `team_code` → names: lookup |
| `players_daily` | player × daily snapshot | Point-in-time flags, price, ownership, `ep_next` (joined into `player_gw`) |
| `gameweeks` | season × gameweek | Deadlines (joined into `player_gw`) |

Tables keep codes, not names; join `players` / `teams` when a person needs to read the result. `players.team_code` is the club at season end (past) or today (current): for the club in a given match use `player_match.team_code`.

## Keys

- **Players:** `element` / `id` is renumbered every season. **`code` is stable across seasons**; use it (`player_code`) to join seasons. Every `element` in the data maps to a `code`.
- **Teams:** the team `id` (1–20) is also per season. Team `code` is stable (Arsenal = 3).
- **Positions:** `element_type` 1–4 = GK/DEF/MID/FWD. The API says `GKP`, vaastav says `GK`; we use `GK`. vaastav 2024-25 also has `AM` rows (Assistant Manager chip), which we drop.
- `players.team` is the player's **current** club. For a past match, the club comes from the fixture (`team_h` if `was_home`, else `team_a`). vaastav's `team` column is already the club at the time.

## Units and types

- Price (`now_cost`, `history.value`, vaastav `value`) is in **tenths of £m**: 39–156 = £3.9m–£15.6m.
- `expected_goals`, `expected_assists`, `expected_goal_involvements`, `expected_goals_conceded`, `influence`, `creativity`, `threat`, `ict_index` and `form` are **strings** in the API. Cast them with `pd.to_numeric`.
- All timestamps are UTC (`...Z`).

## Grain and double gameweeks

- `history` (element_summary) and vaastav `merged_gw` have **one row per match**, so two rows per player in a double gameweek.
- `gwNN_live` has **one row per gameweek** with the stats summed.
- Seen so far: none in 2026-27 GW 1–5. There are 374 player-doubles in 2024-25 and 419 in 2025-26.
- Blank gameweeks (team has no fixture) have **no row at all**. `player_gw` adds them back with `n_fixtures = 0`.
- Cross-check: summed per GW, `history.total_points` equals `live.total_points` for all 3,216 rows this season.
- In `player_gw`, a double has `n_fixtures = 2` and `n_home` = 0, 1 or 2. A blank keeps the player's `team_code` and `position`, forward-filled from his last match. They're never back-filled, because that would copy a later transfer into the past.

## Double gameweeks: predict per fixture (design for Phase 1)

Doubles are 1.3% of player-gameweeks (783 of 60,242), but they're exactly when chips and hits are played, so the optimizer must value them correctly.

- **Don't average the two opponents.** City + Southampton is not "a mid-table opponent", and clean-sheet odds don't respond to strength in a straight line.
- **Don't add `opponent_1` / `opponent_2` columns.** The order is arbitrary, the second column is NaN 98.7% of the time, and triples would need a third.
- **Predict per fixture, then sum to the gameweek:** E[pts_GW] = Σ over the GW's fixtures of E[pts | player, fixture]. A blank sums to 0 with no special case. This needs a `player_fixture` table: one row per player per fixture, future fixtures included.
- **Fixture-specific features**, all known before the deadline: opponent strength, home/away, days of rest before and after the match, `n_fixtures` in the GW. Rest days matter because managers rotate in doubles, so P(start) differs between the two matches.
- **Leakage rule:** `f_*` features are frozen at the gameweek deadline and shared by every fixture in that gameweek. Match 2 of a double must not see match 1, because the transfer decision is made before both. Rolling per match instead of per GW would break this.
- **Correlation:** a start in match 1 makes a start in match 2 somewhat less likely. Summing expected points is still exact (linearity of expectation), but the gameweek's standard deviation is not the simple sum of variances.

## Outcomes vs pre-deadline fields

The same row mixes both kinds, so the question is per **field**, not per file.

| Known only after kickoff (outcomes) | Known before the deadline (usable for that GW) |
| --- | --- |
| `minutes`, `starts`, `goals_scored`, `assists`, `clean_sheets`, `goals_conceded`, `saves`, `bonus`, `bps`, `influence`/`creativity`/`threat`/`ict_index`, `expected_*`, `defensive_contribution` (and parts), cards, `total_points`; fixture scores | `opponent_team`, `was_home`, `kickoff_time`, fixture difficulty, `value` (price that GW), `transfers_in`/`transfers_out` for that GW; from daily snapshots: `status`, `chance_of_playing_next_round`, `ep_next`, `now_cost` |

- An outcome becomes a feature only for **later** gameweeks, through a shifted rolling window (`f_*` columns in `player_gw`, enforced by `tests/test_player_gw.py::test_no_leakage_perturbation`).
- `selected` (ownership) for a GW is counted at the deadline. That's effectively pre-deadline, but treat it with care.
- **Leaks if used for past GWs:** anything from `bootstrap-static` `elements` (`total_points`, `form`, `now_cost`, `selected_by_percent`, `ep_next`) and `teams` (`strength_*`). These are values as of the snapshot day.

## Deadlines (the leakage clock)

- 2026-27: real `deadline_time` from `bootstrap-static` `events`.
- Past seasons have no deadline file. Deadline = first kickoff of the GW − 90 min. This holds exactly for every 2026-27 GW (tested).
- Caveat: vaastav fixtures are end-of-season, so a postponed opening match makes that GW's approximate deadline too late.

## Known gaps

- **Availability flags start on 2026-10-07.** `status` and `chance_of_playing` only exist in daily snapshots, so there are none for 2026-27 GW 1–5 or for past seasons. The minutes model must learn from past minutes and starts until enough snapshots build up.
- **Snapshot features (`snap_*` in `player_gw`).** `players_daily` stacks every daily snapshot: one row per player per snapshot. `player_gw` takes the latest snapshot **strictly before** each deadline (`pd.merge_asof`, backward), so GW 6 uses Oct 9 14:22 for the Oct 10 10:00 deadline. A later snapshot would leak: `ep_next` switches to the next gameweek the moment a deadline passes (Haaland 7.5 → 5.0 on Oct 10). `snap_*` is NaN before GW 6 and for past seasons.
- **Zero-inflated minutes:** 52% of 2026-27 match rows have `minutes == 0`. That's why the model predicts P(start) and P(60+ | start) instead of regressing on minutes.
- **Missing rows ≠ 0 minutes:** players who join mid-season have no rows before they arrive (667 players × 5 GWs = 3,335, but there are 3,216 rows).
- **DefCon only from 2025-26:** `defensive_contribution` is NaN (not recorded) for 2024-25, never 0.
- `history_past` in element_summary is season totals only. Per-GW past data comes from vaastav.
- **`fpl_xp`** (vaastav `xP`) = FPL's own expected points for the gameweek, past seasons only; the same value on both rows of a double, so `player_gw` takes it once. vaastav didn't capture it for 3 gameweeks of 2024-25 and 27 of 2025-26 and wrote 0 for everyone: those are NaN (tested). Correlation with actual points: 0.72 where recorded (0.51 with the fake zeros). Use it as a **baseline**, not a feature: its capture time relative to the deadline isn't documented. For 2026-27 the point-in-time equivalent is `snap_ep_next`.

## Rule changes across seasons

- 2024-25: Assistant Manager chip (`AM` rows, `mng_*` columns), dropped.
- 2025-26: DefCon points introduced.
- 2026-27: bonus points system reworked (CBI bonus per 3, no penalty for being tackled, new GK save bonuses), chips in two sets, up to 5 banked free transfers. Check https://fantasy.premierleague.com/help/rules before hard-coding anything.
