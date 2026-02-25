import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.logging_config import setup_logging


STATUS_FILE = Path("data/processed/line_status_5min.parquet")
DIST_FILE = Path("data/processed/line_distances.parquet")
OUT_DIR = Path("results")
OUT_FILE = OUT_DIR / "ablation_effects.parquet"

B_BOOT = 500
RNG_SEED = 42


@dataclass
class LineEffects:
    line_id: str
    effect_all: float
    effect_d1: float
    effect_d2p: float
    effect_all_ci: Tuple[float, float]
    effect_d1_ci: Tuple[float, float]
    effect_d2p_ci: Tuple[float, float]
    treat_rate: float


def load_status_table() -> pd.DataFrame:
    """Load the line status table and pivot to wide format."""
    if not STATUS_FILE.exists():
        raise FileNotFoundError(
            f"Status file {STATUS_FILE} not found. "
            "Run 05_build_status_table.py first."
        )

    df = pd.read_parquet(STATUS_FILE)

    df["snapshot_time_utc"] = pd.to_datetime(df["snapshot_time_utc"], utc=True)
    df["is_disrupted"] = df["is_disrupted_any"].astype(int)

    # pivot table
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
        raise RuntimeError(
            "Distance matrix contains NaNs after reindexing; check data.")

    logging.info(
        f"Loaded distance matrix for {mat.shape[0]} lines. "
        f"Distance summary (off-diagonal):\n"
        + mat.where(~np.eye(len(mat), dtype=bool)
                    ).stack().describe().to_string()
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


def _ols_with_time_fixed_effects(
    outcome: np.ndarray,
    treatment: np.ndarray,
    times: pd.DatetimeIndex,
) -> float:
    """
    OLS regression: Y = alpha + beta * T + gamma_{dowxhour} + e.

    Implemented via least squares with one-hot dummies for day_of_week x hour_of_day.
    Returns the coefficient beta on T.
    """
    if len(outcome) != len(treatment) or len(outcome) != len(times):
        raise ValueError(
            "Outcome, treatment, and times must have the same length.")

    dow = times.dayofweek
    hour = times.hour
    dow_hour = dow.astype(str) + "_" + hour.astype(str)

    fe = pd.get_dummies(dow_hour, drop_first=True)

    X = pd.concat(
        [
            pd.Series(1.0, index=fe.index, name="intercept"),
            pd.Series(treatment.astype(float), index=fe.index, name="T"),
            fe.astype(float),
        ],
        axis=1,
    ).to_numpy()

    y = outcome.astype(float)

    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta_hat[1])


def compute_point_estimates(
    wide: pd.DataFrame,
    dist_mat: pd.DataFrame,
) -> Dict[str, Dict[str, float]]:
    """
    Compute point estimates Δ_all, Δ_d1, Δ_d2p for each line.

    Returns dict[line_id] -> {"all": Δ_all, "d1": Δ_d1, "d2p": Δ_d2p, "treat_rate": p(T=1)}.
    """
    lines = list(wide.columns)
    results: Dict[str, Dict[str, float]] = {}

    for line in lines:
        T = wide[line].to_numpy(dtype=int)
        treat_rate = T.mean()

        mask_all = np.array([l != line for l in lines])
        Y_all = wide.loc[:, mask_all].mean(axis=1).to_numpy(dtype=float)

        row = dist_mat.loc[line]
        mask_d1 = (row == 1).to_numpy()
        mask_d1 &= mask_all

        mask_d2p = (row >= 2).to_numpy()
        mask_d2p &= mask_all

        Y_d1 = (
            wide.loc[:, mask_d1].mean(axis=1).to_numpy(dtype=float)
            if mask_d1.any()
            else np.full(len(wide), np.nan)
        )
        Y_d2p = (
            wide.loc[:, mask_d2p].mean(axis=1).to_numpy(dtype=float)
            if mask_d2p.any()
            else np.full(len(wide), np.nan)
        )

        d_all = _diff_in_means(T, Y_all)
        d_d1 = _diff_in_means(T, Y_d1) if not np.isnan(Y_d1).all() else np.nan
        d_d2p = _diff_in_means(T, Y_d2p) if not np.isnan(
            Y_d2p).all() else np.nan

        results[line] = {
            "all": d_all,
            "d1": d_d1,
            "d2p": d_d2p,
            "treat_rate": float(treat_rate),
        }

    return results


def bootstrap_effects(
    wide: pd.DataFrame,
    dist_mat: pd.DataFrame,
    b: int = B_BOOT,
    seed: int = RNG_SEED,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """
    Block bootstrap by day to obtain CIs for Δ estimates.

    For each bootstrap replicate:
      - Resample days with replacement.
      - Recompute Δ_all, Δ_d1, Δ_d2p for each line on the resampled panel.
    """
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

    wide_np = wide.to_numpy(dtype=int)

    for b_ix in range(b):
        sampled_days = rng.choice(
            unique_days, size=len(unique_days), replace=True)
        idx_list = [day_to_idx[d] for d in sampled_days]
        boot_idx = np.concatenate(idx_list)

        boot_wide = pd.DataFrame(
            wide_np[boot_idx, :],
            columns=lines,
        )

        boot_results = compute_point_estimates(boot_wide, dist_mat)

        for ln in lines:
            res = boot_results[ln]
            if not np.isnan(res["all"]):
                draws_all[ln].append(res["all"])
            if not np.isnan(res["d1"]):
                draws_d1[ln].append(res["d1"])
            if not np.isnan(res["d2p"]):
                draws_d2p[ln].append(res["d2p"])

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


def run_ablation() -> pd.DataFrame:
    wide = load_status_table()
    lines = list(wide.columns)
    dist_mat = load_distance_matrix(lines)

    logging.info("Computing point estimates.")
    point_est = compute_point_estimates(wide, dist_mat)

    logging.info(f"Running block bootstrap with B={B_BOOT} replications.")
    ci_est = bootstrap_effects(wide, dist_mat, b=B_BOOT, seed=RNG_SEED)

    records: List[Dict[str, object]] = []
    for ln in lines:
        pe = point_est[ln]
        ci = ci_est[ln]

        records.append(
            {
                "line_id": ln,
                "effect_all": pe["all"],
                "effect_all_ci_lower": ci["all"][0],
                "effect_all_ci_upper": ci["all"][1],
                "effect_d1": pe["d1"],
                "effect_d1_ci_lower": ci["d1"][0],
                "effect_d1_ci_upper": ci["d1"][1],
                "effect_d2p": pe["d2p"],
                "effect_d2p_ci_lower": ci["d2p"][0],
                "effect_d2p_ci_upper": ci["d2p"][1],
                "treat_rate": pe["treat_rate"],
                "B": B_BOOT,
            }
        )

    df_out = pd.DataFrame.from_records(records)
    return df_out


def run_regression_crosscheck() -> pd.DataFrame:
    """
    Panel regression cross-check for spillover effects.

    For each line L, estimate:
      Y_all(t) = alpha + beta_all * T_L(t) + gamma_{dowxhour} + e_t
      Y_d1(t)  = alpha + beta_d1 * T_L(t) + gamma_{dowxhour} + e_t
      Y_d2p(t) = alpha + beta_d2p * T_L(t) + gamma_{dowxhour} + e_t

    where Y_* are defined as in the ablation (means across other lines at different
    network distances). We report beta_* as regression-style cross-checks.
    """
    wide = load_status_table()
    lines = list(wide.columns)
    dist_mat = load_distance_matrix(lines)

    times = pd.to_datetime(wide.index)

    records: List[Dict[str, object]] = []

    for line in lines:
        T = wide[line].to_numpy(dtype=int)

        mask_all = np.array([l != line for l in lines])
        Y_all = wide.loc[:, mask_all].mean(axis=1).to_numpy(dtype=float)

        row = dist_mat.loc[line]
        mask_d1 = (row == 1).to_numpy()
        mask_d1 &= mask_all

        mask_d2p = (row >= 2).to_numpy()
        mask_d2p &= mask_all

        Y_d1 = (
            wide.loc[:, mask_d1].mean(axis=1).to_numpy(dtype=float)
            if mask_d1.any()
            else np.full(len(wide), np.nan)
        )
        Y_d2p = (
            wide.loc[:, mask_d2p].mean(axis=1).to_numpy(dtype=float)
            if mask_d2p.any()
            else np.full(len(wide), np.nan)
        )

        beta_all = _ols_with_time_fixed_effects(Y_all, T, times)
        beta_d1 = (
            _ols_with_time_fixed_effects(Y_d1, T, times)
            if not np.isnan(Y_d1).all()
            else np.nan
        )
        beta_d2p = (
            _ols_with_time_fixed_effects(Y_d2p, T, times)
            if not np.isnan(Y_d2p).all()
            else np.nan
        )

        records.append(
            {
                "line_id": line,
                "beta_all": beta_all,
                "beta_d1": beta_d1,
                "beta_d2p": beta_d2p,
            }
        )

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    setup_logging()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    effects = run_ablation()

    logging.info("Ablation effects (preview):")
    logging.info(
        effects.sort_values(
            "effect_all", ascending=False).to_string(index=False)
    )

    effects.to_parquet(OUT_FILE, index=False)
    logging.info(f"Wrote ablation effects to {OUT_FILE}")

    reg_effects = run_regression_crosscheck()
    reg_out_file = OUT_DIR / "ablation_effects_regression.parquet"

    logging.info("Regression cross-check effects:")
    logging.info(reg_effects.to_string(index=False))

    reg_effects.to_parquet(reg_out_file, index=False)
    logging.info(f"Wrote regression cross-check effects to {reg_out_file}")
