# DATA.md: field notes

What's in `data/raw/`, how the pieces fit together, and the traps. Every claim here was checked against the data on 2026-10-07.

## Sources

| Path | Source | Grain | Covers |
| --- | --- | --- | --- |
| `data/raw/daily/<UTC ts>/bootstrap-static.json.gz` | FPL API, daily job | one file per day | players, teams, positions, gameweeks **as of that day** |
| `data/raw/daily/<UTC ts>/fixtures.json.gz` | FPL API, daily job | one row per fixture | all 380 fixtures, kickoffs, scores, difficulty |
| `data/raw/gameweeks/gwNN_live.json.gz` | FPL API, daily job | one row per player per **gameweek** | finished GWs this season |
| `data/raw/element_summary/<date>/<id>.json.gz` | FPL API, one-off backfill | `history`: one row per player per **match** | this season up to `<date>` |
| `data/raw/vaastav/<season>/*.csv.gz` | [vaastav](https://github.com/vaastav/Fantasy-Premier-League), pinned commit `9779cdb` | `merged_gw`: one row per player per **match** | 2024-25, 2025-26 (all 38 GWs) |

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

- **Availability flags start on 2026-10-07.** `status` and `chance_of_playing` only exist in daily snapshots, so there are none for 2026-27 GW 1–5 or for past seasons. The minutes model must learn from past minutes and starts until enough snapshots build up. A later as-of join of snapshots onto `player_gw` (latest snapshot before each deadline) is a TODO.
- **Zero-inflated minutes:** 52% of 2026-27 match rows have `minutes == 0`. That's why the model predicts P(start) and P(60+ | start) instead of regressing on minutes.
- **Missing rows ≠ 0 minutes:** players who join mid-season have no rows before they arrive (667 players × 5 GWs = 3,335, but there are 3,216 rows).
- **DefCon only from 2025-26:** `defensive_contribution` is NaN (not recorded) for 2024-25, never 0.
- `history_past` in element_summary is season totals only. Per-GW past data comes from vaastav.
- vaastav `xP` = FPL's own expected points for that GW. Keep it in mind as the past-season baseline for Phase 4.

## Rule changes across seasons

- 2024-25: Assistant Manager chip (`AM` rows, `mng_*` columns), dropped.
- 2025-26: DefCon points introduced.
- 2026-27: bonus points system reworked (CBI bonus per 3, no penalty for being tackled, new GK save bonuses), chips in two sets, up to 5 banked free transfers. Check https://fantasy.premierleague.com/help/rules before hard-coding anything.
