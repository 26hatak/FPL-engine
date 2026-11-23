"""Load the app's tables: local files in development, the GitHub Release when hosted.

    from app_data import load_table
    xp = load_table("xp")

On your laptop, data/processed/ exists (build_all.py wrote it), so the table is
read from there and the app always shows what you just built. On Streamlit
Community Cloud the repo is checked out without data/processed/ (git-ignored),
so the table is downloaded from the "data-latest" release that the daily
GitHub job uploads (src/publish.py).

No Streamlit here on purpose: this stays testable with plain pytest, and the
app adds the caching itself, e.g.

    @st.cache_data(ttl=3600)        # re-download at most once an hour
    def table(name): return load_table(name)

Names: xp, fixtures, teams, players, prices, gw_values, xp_components (once Phase 1 writes it).
"""
import io
import json
import os
from pathlib import Path

import pandas as pd
import requests

from publish import PROCESSED, gw_values, latest_prices

# Public repo, so release assets download without a token. FPL_DATA_URL
# overrides it (a fork, or a test release).
# Published tables that aren't files in data/processed/: publish.py cuts them
# from a bigger table, so do the same locally and development matches hosting.
DERIVED = {"prices": ("players_daily", latest_prices),
           "gw_values": ("player_gw", gw_values)}

RELEASE_URL = os.environ.get(
    "FPL_DATA_URL", "https://github.com/26hatak/FPL-engine/releases/download/data-latest")


def _download(url: str) -> bytes:
    """GET one release asset. GitHub redirects to its file storage; requests follows."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()   # a 404 (asset missing) fails loudly, not as an empty table
    return response.content


def load_table(name: str, processed: Path = PROCESSED,
               base_url: str = RELEASE_URL) -> pd.DataFrame:
    """One table as a DataFrame, local if data/processed/ has it, else from the release."""
    local = processed / f"{name}.parquet"
    if local.exists():
        return pd.read_parquet(local)
    if name in DERIVED:
        source, cut = DERIVED[name]
        if (processed / f"{source}.parquet").exists():
            return cut(pd.read_parquet(processed / f"{source}.parquet"))
    return pd.read_parquet(io.BytesIO(_download(f"{base_url}/{name}.parquet")))


def load_manifest(base_url: str = RELEASE_URL) -> dict:
    """The published manifest (as_of, rows, git_sha), always from the release:
    it describes what the hosted app is serving."""
    return json.loads(_download(f"{base_url}/manifest.json"))
