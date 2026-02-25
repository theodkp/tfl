import logging
from pathlib import Path
from typing import List

import pandas as pd

from src.logging_config import setup_logging

RAW_DIR = Path("data/raw/line_status")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "line_status_5min.parquet"


def _find_snapshot_files(raw_dir: Path) -> List[Path]:
    """Return all daily snapshot parquet files under data/raw/line_status/date=*/"""
    files: List[Path] = []
    if not raw_dir.exists():
        logging.error(f"Raw directory {raw_dir} does not exist.")
        return files

    for day_dir in sorted(raw_dir.glob("date=*")):
        snap_file = day_dir / "snapshots.parquet"
        if snap_file.exists():
            files.append(snap_file)
        else:
            logging.warning(
                f"Missing snapshots.parquet in {day_dir}, skipping.")
    return files


def _load_raw_snapshots(files: List[Path]) -> pd.DataFrame:
    """Load and concatenate all raw snapshot parquet files"""
    if not files:
        raise FileNotFoundError("No snapshot parquet files found")

    dfs = []
    for f in files:
        logging.info(f"Reading {f}")
        df = pd.read_parquet(f)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    df_all["snapshot_time_utc"] = pd.to_datetime(
        df_all["snapshot_time_utc"], utc=True)
    df_all["snapshot_time_utc"] = df_all["snapshot_time_utc"].dt.round("5min")
    df_all["line_id"] = df_all["line_id"].astype(str).str.lower()
    df_all["status_severity"] = df_all["status_severity"].astype("int64")

    before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["snapshot_time_utc", "line_id"])
    dropped = before - len(df_all)
    if dropped:
        logging.warning(
            f"Dropped {dropped} duplicate (snapshot_time_utc, line_id) rows.")

    return df_all


def _build_regular_time_grid(df_all: pd.DataFrame) -> pd.DataFrame:
    """
    Build a regular 5-minute time grid for all observed lines.
    """
    all_lines = sorted(df_all["line_id"].unique())

    t_min = df_all["snapshot_time_utc"].min().floor("5min")
    t_max = df_all["snapshot_time_utc"].max().ceil("5min")

    full_times = pd.date_range(t_min, t_max, freq="5min", tz="UTC")
    logging.info(
        f"Building 5-minute grid from {t_min} to {t_max} "
        f"({len(full_times)} time points) for {len(all_lines)} lines."
    )

    full_index = pd.MultiIndex.from_product(
        [full_times, all_lines],
        names=["snapshot_time_utc", "line_id"],
    )

    df_all = df_all.set_index(["snapshot_time_utc", "line_id"])
    df_grid = df_all.reindex(full_index)

    df_grid["is_observed"] = ~df_grid["status_severity"].isna()

    return df_grid.reset_index()


def _filter_to_fully_observed_timestamps(df_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only timestamps where we have an observation for every line.
    """
    obs_rate = (
        df_grid.groupby("snapshot_time_utc")[
            "is_observed"].mean().rename("obs_rate")
    )

    summary = obs_rate.describe()
    logging.info("Per-timestamp observation rate summary:")
    logging.info(summary.to_string())

    fully_observed_times = obs_rate[obs_rate == 1.0].index
    kept = len(fully_observed_times)
    total = len(obs_rate)

    logging.info(
        f"Keeping {kept} fully-observed timestamps out of {total} "
        f"({kept / total:.1%} of all 5-minute slots)."
    )

    if kept == 0:
        logging.error(
            "No timestamps have full coverage across all lines. "
            "Consider relaxing the policy or inspecting raw data."
        )
        raise RuntimeError("No fully-observed timestamps available.")

    df_filtered = df_grid[df_grid["snapshot_time_utc"].isin(
        fully_observed_times)].copy()

    return df_filtered


def _add_disruption_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add disruption indicators.
      - is_disrupted_any: any disruption vs good service
      - is_disrupted_main: alias for the main "any disruption" flag
      - is_planned_closure: status_severity == 4
      - is_disrupted_unplanned: disrupted but not planned closure
      - is_disrupted_major: clear major service issues
      - disruption_level: categorical {none, minor, major}
    """

    if df["status_severity"].isna().any():
        raise RuntimeError(
            "status_severity contains NA after filtering; check pipeline.")

    sev = df["status_severity"]

    df["is_disrupted_any"] = sev != 10

    df["is_disrupted_main"] = df["is_disrupted_any"]

    df["is_planned_closure"] = sev == 4

    df["is_disrupted_unplanned"] = (sev != 10) & (sev != 4)

    # no service, suspended, part suspended, part closure, severe delays
    major_codes = {20, 2, 3, 5, 6}
    df["is_disrupted_major"] = sev.isin(major_codes)

    def classify_level(code: int) -> str:
        if code == 10:
            return "none"
        elif code == 9:
            return "minor"
        else:
            return "major"

    df["disruption_level"] = sev.map(classify_level)

    return df


def _enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["snapshot_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True)
    df["line_id"] = df["line_id"].astype("category")
    df["status_severity"] = df["status_severity"].astype("int64")

    df["is_disrupted_any"] = df["is_disrupted_any"].astype(bool)
    df["is_disrupted_main"] = df["is_disrupted_main"].astype(bool)
    df["is_planned_closure"] = df["is_planned_closure"].astype(bool)
    df["is_disrupted_unplanned"] = df["is_disrupted_unplanned"].astype(bool)
    df["is_disrupted_major"] = df["is_disrupted_major"].astype(bool)

    df["disruption_level"] = df["disruption_level"].astype("category")

    return df


def build_status_table() -> pd.DataFrame:
    """Build the line_status_5min df."""
    files = _find_snapshot_files(RAW_DIR)
    df_raw = _load_raw_snapshots(files)

    logging.info(
        f"Loaded {len(df_raw)} raw rows spanning "
        f"{df_raw['snapshot_time_utc'].min()} to {df_raw['snapshot_time_utc'].max()}."
    )

    df_grid = _build_regular_time_grid(df_raw)
    df_full = _filter_to_fully_observed_timestamps(df_grid)

    keep_cols = [
        "snapshot_time_utc",
        "line_id",
        "status_severity",
    ]
    df_full = df_full[keep_cols]

    df_full = _add_disruption_columns(df_full)
    df_full = _enforce_dtypes(df_full)

    df_full = df_full.sort_values(
        ["snapshot_time_utc", "line_id"]).reset_index(drop=True)

    return df_full


if __name__ == "__main__":
    setup_logging()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    table = build_status_table()

    n_times = table["snapshot_time_utc"].nunique()
    n_lines = table["line_id"].nunique()
    logging.info(
        f"Final table has {len(table)} rows, "
        f"{n_times} timestamps, {n_lines} lines."
    )

    disruption_rate = table["is_disrupted_any"].mean()
    logging.info(
        f"Overall disruption rate (main): {table['is_disrupted_main'].mean():.1%}")
    logging.info(
        f"Overall disruption rate (unplanned): {table['is_disrupted_unplanned'].mean():.1%}")
    logging.info(
        f"Planned closure rate: {table['is_planned_closure'].mean():.1%}")

    table.to_parquet(OUT_FILE, index=False)
    logging.info(f"Wrote analysis-ready table to {OUT_FILE}")
