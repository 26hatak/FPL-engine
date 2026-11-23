"""publish.py (manifest, latest prices) and app_data.py (local vs release switch).

Everything runs on tiny tables in pytest's tmp_path; the download is mocked,
so these tests never touch the network."""
import io
import json

import pandas as pd
import pytest

import app_data
import fpl_sync
from build_player_match import CURRENT_SEASON
from publish import gw_values, latest_prices, publish

T0 = pd.Timestamp("2026-11-21T07:17", tz="UTC")
T1 = pd.Timestamp("2026-11-22T07:17", tz="UTC")


@pytest.fixture
def processed(tmp_path):
    """A fake data/processed/ with the required tables and a 2-snapshot players_daily."""
    folder = tmp_path / "processed"
    folder.mkdir()
    pd.DataFrame({"player_code": [1, 2], "gw": [13, 13], "xp": [4.0, 2.5]}).to_parquet(folder / "xp.parquet")
    pd.DataFrame({"fixture": [1, 2, 3]}).to_parquet(folder / "fixtures.parquet")
    pd.DataFrame({"team_code": [3]}).to_parquet(folder / "teams.parquet")
    pd.DataFrame({"player_code": [1, 2]}).to_parquet(folder / "players.parquet")
    pd.DataFrame({"snapshot_time": [T0, T0, T1, T1], "player_code": [1, 2, 1, 2],
                  "now_cost": [50, 60, 51, 59]}).to_parquet(folder / "players_daily.parquet")
    # 2 rows this season (one with no price: a blank), 1 from last season.
    pd.DataFrame({"season": [CURRENT_SEASON, CURRENT_SEASON, "2025-26"], "player_code": [1, 2, 1],
                  "gw": [1, 1, 38], "value": [50.0, None, 70.0],
                  "total_points": [2, 0, 9]}).to_parquet(folder / "player_gw.parquet")
    return folder


def test_latest_prices_keeps_only_newest_snapshot(processed):
    prices = latest_prices(pd.read_parquet(processed / "players_daily.parquet"))
    assert (prices["snapshot_time"] == T1).all()
    assert prices["now_cost"].tolist() == [51, 59]


def test_manifest(processed, tmp_path):
    out = tmp_path / "publish"
    manifest = publish(processed, out, sha="abc123")
    assert manifest == {
        "as_of": T1.isoformat(),
        "git_sha": "abc123",
        "rows": {"fixtures": 3, "gw_values": 1, "players": 2, "prices": 2, "teams": 1, "xp": 2},
    }
    assert json.loads((out / "manifest.json").read_text()) == manifest
    assert sorted(p.name for p in out.iterdir()) == [
        "fixtures.parquet", "gw_values.parquet", "manifest.json", "players.parquet", "prices.parquet",
        "teams.parquet", "xp.parquet"]


def test_optional_table_and_stale_files(processed, tmp_path):
    """xp_components is published when present; an old file in publish/ is removed."""
    out = tmp_path / "publish"
    out.mkdir()
    (out / "stale.parquet").write_bytes(b"old")
    pd.DataFrame({"player_code": [1]}).to_parquet(processed / "xp_components.parquet")
    manifest = publish(processed, out, sha="abc123")
    assert manifest["rows"]["xp_components"] == 1
    assert not (out / "stale.parquet").exists()


def test_missing_required_table_fails(processed, tmp_path):
    (processed / "xp.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        publish(processed, tmp_path / "publish", sha="abc123")


def parquet_bytes(df):
    buffer = io.BytesIO()
    df.to_parquet(buffer)
    return buffer.getvalue()


@pytest.fixture
def fake_download(monkeypatch):
    """Replace the HTTP call: record every URL asked for, serve a 1-row table."""
    urls = []

    def download(url):
        urls.append(url)
        return parquet_bytes(pd.DataFrame({"from": ["release"]}))

    monkeypatch.setattr(app_data, "_download", download)
    return urls


def test_load_table_reads_local_file_when_present(processed, fake_download):
    xp = app_data.load_table("xp", processed=processed, base_url="https://example.test/rel")
    assert xp["xp"].tolist() == [4.0, 2.5]
    assert fake_download == []          # development: no network


def test_load_table_prices_locally_from_players_daily(processed, fake_download):
    prices = app_data.load_table("prices", processed=processed, base_url="https://example.test/rel")
    assert prices["now_cost"].tolist() == [51, 59]
    assert fake_download == []


def test_load_table_downloads_when_no_local_data(tmp_path, fake_download):
    """Hosted: no data/processed/ at all, so every table comes from the release."""
    empty = tmp_path / "nothing_here"
    for name in ["xp", "prices"]:
        df = app_data.load_table(name, processed=empty, base_url="https://example.test/rel")
        assert df["from"].tolist() == ["release"]
    assert fake_download == ["https://example.test/rel/xp.parquet",
                             "https://example.test/rel/prices.parquet"]


def test_load_manifest(monkeypatch):
    monkeypatch.setattr(app_data, "_download", lambda url: b'{"as_of": "x", "rows": {}}')
    assert app_data.load_manifest("https://example.test/rel") == {"as_of": "x", "rows": {}}


def test_gw_values_is_this_season_with_a_price(processed):
    values = gw_values(pd.read_parquet(processed / "player_gw.parquet"))
    assert values.to_dict("records") == [
        {"season": CURRENT_SEASON, "player_code": 1, "gw": 1, "value": 50.0}]


def test_load_table_gw_values_locally_from_player_gw(processed, fake_download):
    values = app_data.load_table("gw_values", processed=processed, base_url="https://example.test/rel")
    assert values["value"].tolist() == [50.0]
    assert fake_download == []


def test_team_sync_tables_load_from_the_release_alone(processed, tmp_path, monkeypatch):
    """Hosted: no data/processed/. Publish, serve the release from that folder,
    and check fpl_sync gets every table it needs, the same as locally."""
    out = tmp_path / "publish"
    publish(processed, out, sha="abc123")
    # The "release" is the publish folder: serve each URL's file name from it.
    monkeypatch.setattr(app_data, "_download",
                        lambda url: (out / url.rsplit("/", 1)[1]).read_bytes())

    hosted = fpl_sync.load_tables(processed=tmp_path / "nothing_here")
    local = fpl_sync.load_tables(processed=processed)
    assert hosted.keys() == local.keys()
    for name in local:
        pd.testing.assert_frame_equal(hosted[name], local[name], check_dtype=False)
