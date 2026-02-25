import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from src.logging_config import setup_logging


STATUS_FILE = Path("data/processed/line_status_5min.parquet")
DIST_FILE = Path("data/processed/line_distances.parquet")
OUT_DIR = Path("results")
OUT_FILE = OUT_DIR / "ablation_effects_time_matched.parquet"


def load_status_table() -> pd.DataFrame:
    """Load the status table and pivot to wide format."""
    if not STATUS_FILE.exists():
        raise FileNotFoundError(
            f"Status file {STATUS_FILE} not found. "
            "Run 05_build_status_table.py first."
        )

    df = pd.read_parquet(STATUS_FILE)

    df["snapshot_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True)
    df["is_disrupted"] = df["is_disrupted_any"].astype(int)

    wide = df.pivot(
        index="snapshot_time_utc",
        columns="line_id",
        values="is_disrupted",
    ).sort_index()

    logging.info(
        f"Loaded status table with {len(wide)} timestamps and {wide.shape[1]} lines."
    )

    return wide


def load_distance_matrix(lines: List[str]) -> pd.DataFrame:
    """Load line distances and return a square distance matrix aligned to lines."""
    if not DIST_FILE.exists():
        raise FileNotFoundError(
            f"Distance file {DIST_FILE} not found. "
            "Run 06_compute_distances.py first."
        )

    dist = pd.read_parquet(DIST_FILE)
    mat = dist.pivot(
        index="source_line", columns="target_line", values="distance"
    )

    mat = mat.reindex(index=lines, columns=lines)

    if mat.isna().any().any():
        raise RuntimeError("Distance matrix contains NaNs after reindexing; check data.")

    logging.info(
        f"Loaded distance matrix for {mat.shape[0]} lines. "
        f"Distance summary (off-diagonal):\n"
        + mat.where(~np.eye(len(mat), dtype=bool)).stack().describe().to_string()
    )

    return mat


def _diff_in_means(
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> float:
    """Simple difference in means E[Y|T=1] - E[Y|T=0]."""
    mask_t = treatment == 1
    mask_c = treatment == 0

    if not mask_t.any() or not mask_c.any():
        return np.nan

    return float(outcome[mask_t].mean() - outcome[mask_c].mean())


def run_time_matched() -> pd.DataFrame:
    """
    Time-matched ablation.

    Within each (day_of_week x hour_of_day) stratum, compute Δ as in the baseline,
    then aggregate across strata with weights proportional to the number of treated
    timestamps in each stratum.
    """
    wide = load_status_table()
    lines = list(wide.columns)
    dist_mat = load_distance_matrix(lines)

    times = pd.to_datetime(wide.index)
    dow = times.dayofweek
    hour = times.hour
    strata = dow.astype(str) + "_" + hour.astype(str)

    unique_strata = np.unique(strata)

    stratum_to_idx: Dict[str, np.ndarray] = {}
    for s in unique_strata:
        stratum_to_idx[s] = np.where(strata == s)[0]

    records: List[Dict[str, object]] = []

    for line in lines:
        T = wide[line].to_numpy(dtype=int)

        mask_all = np.array([l != line for l in lines])

        row = dist_mat.loc[line]
        mask_d1 = (row == 1).to_numpy()
        mask_d1 &= mask_all

        mask_d2p = (row >= 2).to_numpy()
        mask_d2p &= mask_all

        Y_all_full = wide.loc[:, mask_all].mean(axis=1).to_numpy(dtype=float)
        Y_d1_full = (
            wide.loc[:, mask_d1].mean(axis=1).to_numpy(dtype=float)
            if mask_d1.any()
            else np.full(len(wide), np.nan)
        )
        Y_d2p_full = (
            wide.loc[:, mask_d2p].mean(axis=1).to_numpy(dtype=float)
            if mask_d2p.any()
            else np.full(len(wide), np.nan)
        )

        num_treated_total = 0.0
        num_treated_all = 0.0
        num_treated_d1 = 0.0
        num_treated_d2p = 0.0

        sum_w_delta_all = 0.0
        sum_w_delta_d1 = 0.0
        sum_w_delta_d2p = 0.0

        for s, idx in stratum_to_idx.items():
            T_s = T[idx]

            if not ((T_s == 1).any() and (T_s == 0).any()):
                continue

            n_treated_s = float(T_s.sum())
            if n_treated_s == 0.0:
                continue

            num_treated_total += n_treated_s

            Y_all_s = Y_all_full[idx]
            d_all_s = _diff_in_means(T_s, Y_all_s)
            if not np.isnan(d_all_s):
                sum_w_delta_all += n_treated_s * d_all_s
                num_treated_all += n_treated_s

            # Distance 1 
            if not np.isnan(Y_d1_full).all():
                Y_d1_s = Y_d1_full[idx]
                if not np.isnan(Y_d1_s).all():
                    d_d1_s = _diff_in_means(T_s, Y_d1_s)
                    if not np.isnan(d_d1_s):
                        sum_w_delta_d1 += n_treated_s * d_d1_s
                        num_treated_d1 += n_treated_s

            # Distance ≥2 
            if not np.isnan(Y_d2p_full).all():
                Y_d2p_s = Y_d2p_full[idx]
                if not np.isnan(Y_d2p_s).all():
                    d_d2p_s = _diff_in_means(T_s, Y_d2p_s)
                    if not np.isnan(d_d2p_s):
                        sum_w_delta_d2p += n_treated_s * d_d2p_s
                        num_treated_d2p += n_treated_s

        effect_all = (
            sum_w_delta_all / num_treated_all if num_treated_all > 0 else np.nan
        )
        effect_d1 = (
            sum_w_delta_d1 / num_treated_d1 if num_treated_d1 > 0 else np.nan
        )
        effect_d2p = (
            sum_w_delta_d2p / num_treated_d2p if num_treated_d2p > 0 else np.nan
        )

        treat_rate = float(T.mean())

        records.append(
            {
                "line_id": line,
                "effect_all": effect_all,
                "effect_d1": effect_d1,
                "effect_d2p": effect_d2p,
                "treat_rate": treat_rate,
                "n_treated_used": num_treated_total,
            }
        )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    setup_logging()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    effects_tm = run_time_matched()

    logging.info("Time-matched ablation effects:")
    logging.info(
        effects_tm.sort_values("effect_all", ascending=False).to_string(index=False)
    )

    effects_tm.to_parquet(OUT_FILE, index=False)
    logging.info(f"Wrote time-matched ablation effects to {OUT_FILE}")

