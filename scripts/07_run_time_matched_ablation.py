import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils.logging_config import setup_logging


STATUS_FILE = Path("data/processed/line_status_5min.parquet")
DIST_FILE = Path("data/processed/line_distances.parquet")
OUT_DIR = Path("results")
OUT_FILE = OUT_DIR / "ablation_effects_time_matched.parquet"

B_BOOT = 500
RNG_SEED = 42


# Load the status table and pivot to wide format (rows=timestamps, cols=lines).
def load_status_table() -> pd.DataFrame:
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


# Load line distances and return a square distance matrix aligned to lines.
def load_distance_matrix(lines: List[str]) -> pd.DataFrame:
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


# Simple difference in means: E[Y|T=1] - E[Y|T=0].
def _diff_in_means(
    treatment: np.ndarray,
    outcome: np.ndarray,
) -> float:
    mask_t = treatment == 1
    mask_c = treatment == 0

    if not mask_t.any() or not mask_c.any():
        return np.nan

    return float(outcome[mask_t].mean() - outcome[mask_c].mean())


# Compute time-matched effects: within each (day-of-week x hour) stratum, compute Δ,
# then average across strata weighted by number of treated timestamps.
# Reduces confounding from rush-hour vs off-peak disruption patterns.
def compute_time_matched_effects(
    wide: pd.DataFrame,
    dist_mat: pd.DataFrame,
) -> pd.DataFrame:
    lines = list(wide.columns)
    times = pd.to_datetime(wide.index)
    # Stratify by day-of-week and hour so we compare like-with-like (e.g. Mon 9am vs Mon 9am).
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
        mask_d1 = (row == 1).to_numpy() & mask_all
        mask_d2p = (row >= 2).to_numpy() & mask_all

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
            # Skip strata with no treated or no control (can't compute diff-in-means).
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

            if not np.isnan(Y_d1_full).all():
                Y_d1_s = Y_d1_full[idx]
                if not np.isnan(Y_d1_s).all():
                    d_d1_s = _diff_in_means(T_s, Y_d1_s)
                    if not np.isnan(d_d1_s):
                        sum_w_delta_d1 += n_treated_s * d_d1_s
                        num_treated_d1 += n_treated_s

            if not np.isnan(Y_d2p_full).all():
                Y_d2p_s = Y_d2p_full[idx]
                if not np.isnan(Y_d2p_s).all():
                    d_d2p_s = _diff_in_means(T_s, Y_d2p_s)
                    if not np.isnan(d_d2p_s):
                        sum_w_delta_d2p += n_treated_s * d_d2p_s
                        num_treated_d2p += n_treated_s

        # Weighted average of stratum-level Δ by number of treated in each stratum.
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


# Block bootstrap by day; return 95% CIs for effect_all, effect_d1, effect_d2p per line.
def bootstrap_time_matched(
    wide: pd.DataFrame,
    dist_mat: pd.DataFrame,
    b: int = B_BOOT,
    seed: int = RNG_SEED,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    rng = np.random.default_rng(seed)
    times = wide.index
    days = pd.to_datetime(times).date
    unique_days = np.array(sorted(set(days)))
    day_to_idx: Dict[object, np.ndarray] = {}
    for d in unique_days:
        day_to_idx[d] = np.where(days == d)[0]

    lines = list(wide.columns)
    draws_all: Dict[str, List[float]] = {ln: [] for ln in lines}
    draws_d1: Dict[str, List[float]] = {ln: [] for ln in lines}
    draws_d2p: Dict[str, List[float]] = {ln: [] for ln in lines}

    wide_np = wide.to_numpy(dtype=float)
    wide_index = wide.index

    for _ in range(b):
        sampled_days = rng.choice(unique_days, size=len(unique_days), replace=True)
        idx_list = [day_to_idx[d] for d in sampled_days]
        boot_idx = np.concatenate(idx_list)
        boot_wide = pd.DataFrame(
            wide_np[boot_idx, :],
            index=wide_index[boot_idx],
            columns=lines,
        )
        boot_df = compute_time_matched_effects(boot_wide, dist_mat)
        for _, row in boot_df.iterrows():
            ln = row["line_id"]
            if not np.isnan(row["effect_all"]):
                draws_all[ln].append(row["effect_all"])
            if not np.isnan(row["effect_d1"]):
                draws_d1[ln].append(row["effect_d1"])
            if not np.isnan(row["effect_d2p"]):
                draws_d2p[ln].append(row["effect_d2p"])

    ci_results: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for ln in lines:
        def _ci(draws: List[float]) -> Tuple[float, float]:
            if not draws:
                return (np.nan, np.nan)
            qs = np.quantile(draws, [0.025, 0.975])
            return float(qs[0]), float(qs[1])

        ci_results[ln] = {
            "all": _ci(draws_all[ln]),
            "d1": _ci(draws_d1[ln]),
            "d2p": _ci(draws_d2p[ln]),
        }
    return ci_results


# Main pipeline: load data, compute time-matched effects, bootstrap CIs.
def run_time_matched() -> pd.DataFrame:
    wide = load_status_table()
    lines = list(wide.columns)
    dist_mat = load_distance_matrix(lines)

    logging.info("Computing time-matched point estimates.")
    effects_df = compute_time_matched_effects(wide, dist_mat)

    logging.info(f"Running block bootstrap with B={B_BOOT} replications.")
    ci_est = bootstrap_time_matched(wide, dist_mat, b=B_BOOT, seed=RNG_SEED)

    effects_df["effect_all_ci_lower"] = effects_df["line_id"].map(
        lambda ln: ci_est[ln]["all"][0]
    )
    effects_df["effect_all_ci_upper"] = effects_df["line_id"].map(
        lambda ln: ci_est[ln]["all"][1]
    )
    effects_df["effect_d1_ci_lower"] = effects_df["line_id"].map(
        lambda ln: ci_est[ln]["d1"][0]
    )
    effects_df["effect_d1_ci_upper"] = effects_df["line_id"].map(
        lambda ln: ci_est[ln]["d1"][1]
    )
    effects_df["effect_d2p_ci_lower"] = effects_df["line_id"].map(
        lambda ln: ci_est[ln]["d2p"][0]
    )
    effects_df["effect_d2p_ci_upper"] = effects_df["line_id"].map(
        lambda ln: ci_est[ln]["d2p"][1]
    )
    effects_df["B"] = B_BOOT

    return effects_df


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

