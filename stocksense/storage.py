"""Persistence: parquet cache for cleaned data and a small SQLite store for saved results."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from io import StringIO
from pathlib import Path


import os
import streamlit as st
from huggingface_hub import hf_hub_download

import pandas as pd

from . import config as C
from .preprocessing import QualityReport, list_csv_files, load_and_clean


def _fingerprint(path) -> str:
    files = list_csv_files(path)
    return "_".join(f"{f.stem}-{f.stat().st_size}" for f in files)[:150]


HF_REPO_ID = "ParthaviW/stocksense-nse-data"
HF_FILENAME = "nse_2000000.csv"


def resolve_data_path(path):
    """Use the local file when available; otherwise download the private HF dataset."""
    p = Path(path).expanduser()

    # Preserve the user's ability to select another local file or folder.
    if p.exists():
        return p

    # Only auto-download when the default dataset path is requested.
    if p.resolve() != C.DEFAULT_DATA_PATH.resolve():
        return p

    token = os.environ.get("HF_TOKEN")
    if not token:
        try:
            token = st.secrets.get("HF_TOKEN")
        except Exception:
            token = None

    if not token:
        raise RuntimeError(
            "HF_TOKEN is missing. Add it under Streamlit Cloud → App settings → Secrets."
        )

    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            token=token,
        )
        return Path(downloaded_path)
    except Exception as exc:
        raise RuntimeError(
            "Could not download the NSE dataset from Hugging Face. "
            "Check HF_TOKEN and repository access."
        ) from exc



def load_with_cache(path, series=("EQ",), cache_dir: Path = C.CACHE_DIR):
    """Load data, downloading the default private HF dataset when needed."""
    resolved_path = resolve_data_path(path)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    key = _fingerprint(resolved_path) + "_" + "-".join(series)
    pq = cache_dir / f"{key}.parquet"
    js = cache_dir / f"{key}.json"

    if pq.exists() and js.exists():
        try:
            rep = QualityReport(**json.loads(js.read_text()))
            return pd.read_parquet(pq), rep, True
        except Exception:
            pass  # Corrupt cache: rebuild it.

    df, rep = load_and_clean(resolved_path, series)
    df.to_parquet(pq, index=False)
    js.write_text(json.dumps(rep.to_dict(), default=str))
    return df, rep, False



class ResultStore:
    """Saves result tables (e.g. mined rules) with their parameters so runs can be revisited."""

    def __init__(self, db_path: Path | None = None):
        self.db = Path(db_path or C.CACHE_DIR / "results.db")
        self.db.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db) as con:
            con.execute("CREATE TABLE IF NOT EXISTS runs (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "created TEXT, kind TEXT, params TEXT, data TEXT)")

    def save(self, kind: str, params: dict, table: pd.DataFrame) -> int:
        with sqlite3.connect(self.db) as con:
            cur = con.execute("INSERT INTO runs (created, kind, params, data) VALUES (?,?,?,?)",
                              (datetime.now().isoformat(timespec="seconds"), kind,
                               json.dumps(params, default=str),
                               table.to_json(orient="split", default_handler=str)))
            return int(cur.lastrowid)

    def list_runs(self, kind: str | None = None) -> pd.DataFrame:
        q = "SELECT id, created, kind, params FROM runs" + (" WHERE kind=?" if kind else "") + " ORDER BY id DESC"
        with sqlite3.connect(self.db) as con:
            return pd.read_sql_query(q, con, params=(kind,) if kind else ())

    def load(self, run_id: int) -> pd.DataFrame:
        with sqlite3.connect(self.db) as con:
            row = con.execute("SELECT data FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"No saved run with id {run_id}")
        return pd.read_json(StringIO(row[0]), orient="split")
