"""fpl_sync on hand-built histories: every expected number checks with a calculator.

No test touches the internet: `no_network` (autouse) makes any real HTTP call
fail, and the HTTP tests swap in fake responses.
"""
import pandas as pd
import pytest
import requests

import fpl_sync as fs
from fpl_sync import (FplApiError, FplClient, TeamNotFound, chips_available, fetch_team,
                      reconstruct_free_transfers, reconstruct_purchase_prices,
                      reconstruct_squad_elements, reconstruct_team, selling_price)

SEASON = "2026-27"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("a test tried to reach the network")
    monkeypatch.setattr(requests.Session, "request", boom)


# --------------------------------------------------------------------------- #
# Selling price
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("purchase, current, expected", [
    (75, 78, 76),   # the rules' own example: 7.5 -> 7.8 sells for 7.6
    (50, 51, 50),   # +0.1: half is 0.05, rounded down -> no gain at all
    (50, 52, 51),   # +0.2 -> +0.1
    (50, 53, 51),   # +0.3: half is 0.15, rounded down -> +0.1
    (50, 50, 50),   # no change
    (50, 48, 48),   # a drop is taken in full
    (100, 89, 89),
])
def test_selling_price(purchase, current, expected):
    assert selling_price(purchase, current) == expected


# --------------------------------------------------------------------------- #
# Free transfers
# --------------------------------------------------------------------------- #

def hist(event, made=0, cost=0, bank=0):
    return dict(event=event, event_transfers=made, event_transfers_cost=cost,
                bank=bank, value=1000)


def test_free_transfers_bank_up_to_five():
    # GW1 is the first deadline (unlimited), then nine weeks without a transfer:
    # 1, 2, 3, 4, 5, 5, ... -> capped at 5.
    rows = [hist(g) for g in range(1, 11)]
    ft, warnings = reconstruct_free_transfers(rows, [], started_event=1)
    assert ft == 5
    assert warnings == []


def test_free_transfers_count_down_and_hits():
    rows = [hist(1, made=7),          # before the first deadline: free, ignored
            hist(2),                  # ft 1 unused -> 2
            hist(3, made=1),          # 2 - 1 + 1 = 2
            hist(4, made=3, cost=4),  # one hit; max(2 - 3, 0) + 1 = 1
            hist(5, made=1)]          # 1 - 1 + 1 = 1
    ft, warnings = reconstruct_free_transfers(rows, [], started_event=1)
    assert ft == 1
    assert warnings == []


def test_wildcard_and_free_hit_keep_saved_transfers():
    # Rules: "If you had 2 saved free transfers, you will still have 2 saved
    # free transfers the Gameweek after playing the chip." -> no +1 that week.
    # FPL reports 0 transfers in a chip week, but any number would be ignored.
    rows = [hist(1), hist(2), hist(3, made=12), hist(4), hist(5, made=9), hist(6)]
    chips = [{"name": "wildcard", "event": 3}, {"name": "freehit", "event": 5}]
    # ft: GW2 1 -> GW3 2 (wildcard) -> GW4 2 -> GW5 3 (free hit) -> GW6 3 -> GW7 4
    ft, warnings = reconstruct_free_transfers(rows, chips, started_event=1)
    assert ft == 4
    assert warnings == []


def test_wildcard_at_five_stays_five():
    rows = [hist(g) for g in range(1, 7)] + [hist(7, made=15), hist(8, made=2)]
    chips = [{"name": "wildcard", "event": 7}]
    # 5 at GW6 -> GW7 5 (WC, kept) -> GW8 5 - 2 + 1 = 4
    ft, _ = reconstruct_free_transfers(rows, chips, started_event=1)
    assert ft == 4


def test_bench_boost_does_not_reset():
    rows = [hist(1), hist(2), hist(3)]
    chips = [{"name": "bboost", "event": 2}]
    ft, _ = reconstruct_free_transfers(rows, chips, started_event=1)
    assert ft == 3


def test_mid_season_starter():
    # Team created for GW5: GW5 is its unlimited week, GW6 is its first free transfer.
    rows = [hist(5, made=4), hist(6), hist(7)]
    ft, _ = reconstruct_free_transfers(rows, [], started_event=5)
    assert ft == 3


def test_free_transfer_mismatch_is_reported():
    # 2 transfers with only 1 free should cost 4; FPL charged 0, so FPL must
    # have granted an extra transfer the rules don't describe.
    rows = [hist(1), hist(2, made=2, cost=0)]
    _, warnings = reconstruct_free_transfers(rows, [], started_event=1)
    assert len(warnings) == 1 and "GW2" in warnings[0]


# --------------------------------------------------------------------------- #
# Squad and purchase prices
# --------------------------------------------------------------------------- #

def picks(elements, chip=None):
    return {"active_chip": chip, "picks": [{"element": e} for e in elements]}


def tr(event, e_in, cost_in, e_out, cost_out, day):
    return dict(event=event, element_in=e_in, element_in_cost=cost_in,
                element_out=e_out, element_out_cost=cost_out,
                time=f"2026-09-{day:02d}T10:00:00Z")


def test_pending_transfer_is_applied():
    # GW5 picks already include the GW5 transfer (3 -> 16); only GW6's is pending.
    gw5 = [16 if e == 3 else e for e in range(1, 16)]
    squad, pending = reconstruct_squad_elements(
        picks(gw5), None, [tr(5, 16, 60, 3, 50, 1), tr(6, 17, 45, 4, 44, 9)],
        as_of_gw=5)
    assert 17 in squad and 4 not in squad and 16 in squad and len(squad) == 15
    assert [t["element_in"] for t in pending] == [17]


def test_free_hit_squad_reverts():
    fh_squad = list(range(20, 35))
    squad, _ = reconstruct_squad_elements(picks(fh_squad, "freehit"), picks(range(1, 16)),
                                          [], as_of_gw=5)
    assert squad == list(range(1, 16))


def test_purchase_prices():
    transfers = [
        tr(3, 16, 50, 2, 45, 1),   # bought 16 at 5.0 ...
        tr(4, 2, 46, 16, 51, 5),   # ... sold him, re-bought 2 at 4.6
        tr(6, 16, 55, 3, 40, 10),  # re-bought 16 at 5.5: the price that counts
        tr(7, 4, 99, 5, 60, 15),   # free-hit week: reverted, must be ignored
    ]
    chips = [{"name": "freehit", "event": 7}]
    squad = [1, 2, 16, 5]
    first = {1: 40, 2: 45, 5: 61}
    got = reconstruct_purchase_prices(squad, transfers, chips, first)
    assert got[16] == (55, "transfer")
    assert got[2] == (46, "transfer")          # re-bought, not the first-deadline price
    assert got[1] == (40, "first_deadline")
    assert got[5] == (61, "first_deadline")    # the free-hit sale didn't happen


# --------------------------------------------------------------------------- #
# Chips
# --------------------------------------------------------------------------- #

def test_chips_available():
    used = [{"name": "wildcard", "event": 4}, {"name": "freehit", "event": 11}]
    assert chips_available(used, 12) == ["bboost", "3xc"]   # no FH in back-to-back GWs
    assert chips_available(used, 13) == ["bboost", "3xc"]   # first-half FH already used
    assert chips_available([{"name": "freehit", "event": 18}], 19) == \
        ["wildcard", "bboost", "3xc"]                       # back-to-back across a GW
    # Second half: everything is back.
    assert chips_available(used, 20) == ["wildcard", "freehit", "bboost", "3xc"]
    assert chips_available([], 38) == ["wildcard", "freehit", "bboost", "3xc"]
    assert chips_available([], None) == []


# --------------------------------------------------------------------------- #
# Whole team
# --------------------------------------------------------------------------- #

POS = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3


@pytest.fixture
def tables():
    elements = range(1, 21)
    players = pd.DataFrame({
        "season": SEASON, "element": list(elements),
        "player_code": [1000 + e for e in elements],
        "web_name": [f"P{e}" for e in elements],
        "team_code": [e % 4 + 1 for e in elements],
        "position": [POS[(e - 1) % 15] for e in elements],
    })
    teams = pd.DataFrame({"season": SEASON, "team_code": [1, 2, 3, 4],
                          "team_short": ["AAA", "BBB", "CCC", "DDD"]})
    t_old, t_new = pd.Timestamp("2026-11-20", tz="UTC"), pd.Timestamp("2026-11-21", tz="UTC")
    prices = pd.DataFrame({
        "season": SEASON, "player_code": [1000 + e for e in elements] * 2,
        "snapshot_time": [t_old] * 20 + [t_new] * 20,
        # every player 5.0 yesterday; today 5.3 for elements 1-10, 4.8 for 11-20
        "now_cost": [50] * 20 + [53] * 10 + [48] * 10,
    })
    first = pd.DataFrame({"season": SEASON, "player_code": [1000 + e for e in elements],
                          "gw": 1, "value": 50})
    return dict(players=players, teams=teams, prices=prices, first_deadline_values=first)


def raw_team(latest_chip=None):
    gw = 4
    history = {"current": [hist(1, bank=0), hist(2, bank=0),
                           hist(3, made=1, bank=5), hist(4, bank=99 if latest_chip else 5)],
               "chips": [{"name": "freehit", "event": 4}] if latest_chip else [],
               "past": []}
    transfers = [tr(3, 16, 45, 1, 50, 1),        # GW3: 1 -> 16 at 4.5, bank 0 -> 5
                 tr(5, 17, 52, 2, 50, 20)]       # pending for GW5: 2 -> 17, bank -2
    squad_gw4 = [16] + list(range(2, 16))
    return {"entry": {"id": 42, "name": "Test FC", "started_event": 1, "current_event": gw},
            "history": history, "transfers": transfers,
            "picks": picks(range(5, 20), "freehit") if latest_chip else picks(squad_gw4),
            "prev_picks": picks(squad_gw4) if latest_chip else None}


@pytest.mark.parametrize("latest_chip", [None, "freehit"])
def test_reconstruct_team(tables, latest_chip):
    team = reconstruct_team(raw_team(latest_chip), **tables)
    df = team.to_frame().set_index("element")
    assert len(df) == 15 and 1 not in df.index and 2 not in df.index
    assert {16, 17} <= set(df.index)
    assert df["player_code"].tolist() == [1000 + e for e in df.index]   # never `element` as key
    assert team.as_of_gw == 4 and team.next_gw == 5
    assert team.bank == 5 + 50 - 52       # GW4 bank (pre free hit) + pending sale - buy
    # 16: bought 4.5, now 4.8 -> +0.3 rise, keep 0.1 -> 4.6
    assert df.loc[16, ["purchase_price", "current_price", "selling_price", "price_source"]
                  ].tolist() == [45, 48, 46, "transfer"]
    # 3: first squad at 5.0, now 5.3 -> 5.1
    assert df.loc[3, ["purchase_price", "selling_price", "price_source"]].tolist() == \
        [50, 51, "first_deadline"]
    # 12: first squad at 5.0, now 4.8 -> drop taken in full
    assert df.loc[12, "selling_price"] == 48
    assert df.loc[3, "club"] == "DDD" and df.loc[3, "position"] == "DEF"   # team 3 % 4 + 1
    # ft: GW2 1 -> GW3 2 -> GW4 2 (1 used, +1) -> GW5 3, or 2 if GW4 was the free hit
    assert team.free_transfers == (2 if latest_chip else 3)
    assert team.free_transfers_left == team.free_transfers - 1     # one pending
    assert team.prices_as_of.startswith("2026-11-21")
    assert team.warnings == []


def test_unknown_element_is_a_clear_error(tables):
    raw = raw_team()
    raw["picks"] = picks(list(range(2, 16)) + [999])
    raw["transfers"] = []
    with pytest.raises(ValueError, match="999"):
        reconstruct_team(raw, **tables)


# --------------------------------------------------------------------------- #
# HTTP layer, with the network replaced by fakes
# --------------------------------------------------------------------------- #

class FakeClient:
    def __init__(self, responses):
        self.responses, self.calls = responses, []

    def get(self, path):
        self.calls.append(path)
        if path not in self.responses:
            raise TeamNotFound(path)
        return self.responses[path]


def test_fetch_team_free_hit_fetches_previous_picks():
    r = raw_team("freehit")
    client = FakeClient({"entry/42/": r["entry"], "entry/42/history/": r["history"],
                         "entry/42/transfers/": r["transfers"],
                         "entry/42/event/4/picks/": r["picks"],
                         "entry/42/event/3/picks/": r["prev_picks"]})
    got = fetch_team(42, client)
    assert client.calls[-1] == "entry/42/event/3/picks/"
    assert got["prev_picks"] == r["prev_picks"]


def test_fetch_team_before_first_deadline():
    client = FakeClient({"entry/7/": {"id": 7, "current_event": None, "started_event": 12}})
    with pytest.raises(FplApiError, match="first deadline"):
        fetch_team(7, client)


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


@pytest.mark.parametrize("response, error, text", [
    (FakeResponse(404, {"detail": "No Entry matches the given query."}), TeamNotFound, "No FPL team"),
    (FakeResponse(503), FplApiError, "being updated"),
    (FakeResponse(500), FplApiError, "HTTP 500"),
    (FakeResponse(200), FplApiError, "non-JSON"),
    (requests.ConnectionError("down"), FplApiError, "Could not reach"),
])
def test_http_errors(monkeypatch, response, error, text):
    def fake_get(self, url, timeout):
        assert timeout == fs.TIMEOUT
        assert self.headers["User-Agent"] == fs.USER_AGENT
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(requests.Session, "get", fake_get)
    with pytest.raises(error, match=text):
        fs.fetch_entry(FplClient(pause=0), 404404)
