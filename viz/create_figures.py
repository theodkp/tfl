"""
Generate ablation figures:
1. Spillover ranking (bar plot, time-matched, error bars)
2. Baseline vs time-matched scatter (45° line)
3. Distance decay (average effect_d1 vs effect_d2p)
4. Network structure (node size = spillover, edge width = weight)
"""
from pathlib import Path

import pickle
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results"
DATA_DIR = BASE / "data" / "processed"
FIG_DIR = BASE / "viz" / "figures"
GRAPH_PATH = DATA_DIR / "line_graph.pkl"


# Load time-matched ablation results.
def _load_time_matched() -> pd.DataFrame:
    p = RESULTS_DIR / "ablation_effects_time_matched.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Run 08_run_time_matched_ablation.py first. Missing {p}")
    return pd.read_parquet(p)


# Load baseline ablation results.
def _load_baseline() -> pd.DataFrame:
    p = RESULTS_DIR / "ablation_effects.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Run 07_run_ablation.py first. Missing {p}")
    return pd.read_parquet(p)


# Load line graph.
def _load_graph() -> nx.Graph:
    if not GRAPH_PATH.exists():
        raise FileNotFoundError(f"Line graph not found. Run 03_build_line_graph.py. Missing {GRAPH_PATH}")
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


# Bar plot of spillover effect by line, sorted smallest to largest.
def fig1_spillover_ranking() -> None:
    df = _load_time_matched()
    df = df.sort_values("effect_all", ascending=True)

    has_ci = "effect_all_ci_lower" in df.columns and "effect_all_ci_upper" in df.columns
    yerr = None
    if has_ci:
        yerr = np.array([
            df["effect_all"] - df["effect_all_ci_lower"],
            df["effect_all_ci_upper"] - df["effect_all"],
        ])

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(df))
    ax.barh(x, df["effect_all"], color="steelblue", alpha=0.85, edgecolor="navy", linewidth=0.5)
    if yerr is not None:
        ax.errorbar(
            np.array(df["effect_all"].values),
            x,
            xerr=yerr,
            fmt="none",
            color="black",
            capsize=3,
            capthick=1,
        )
    ax.set_yticks(x)
    ax.set_yticklabels(df["line_id"].values, fontsize=10)
    ax.set_xlabel("Spillover effect (time-matched)")
    ax.set_ylabel("Line")
    ax.set_title("Spillover effect by line (time-matched)")
    ax.axvline(0, color="gray", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG_DIR / "fig1_spillover_ranking.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR / 'fig1_spillover_ranking.png'}")


# Scatter of baseline vs time-matched spillover with 45° line.
def fig2_baseline_vs_timematched() -> None:
    base = _load_baseline()
    tm = _load_time_matched()
    m = base.merge(tm[["line_id", "effect_all"]], on="line_id", suffixes=("_baseline", "_tm"))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(m["effect_all_baseline"], m["effect_all_tm"], s=80, alpha=0.8, color="steelblue", edgecolor="navy")
    lims = [
        min(m["effect_all_baseline"].min(), m["effect_all_tm"].min()) - 0.02,
        max(m["effect_all_baseline"].max(), m["effect_all_tm"].max()) + 0.02,
    ]
    ax.plot(lims, lims, "k--", alpha=0.7, label="45° line")
    ax.set_xlim(tuple(lims))
    ax.set_ylim(tuple(lims))
    ax.set_xlabel("Baseline spillover effect")
    ax.set_ylabel("Time-matched spillover effect")
    ax.set_title("Baseline vs time-matched spillover estimates")
    ax.set_aspect("equal")
    for _, row in m.iterrows():
        ax.annotate(
            row["line_id"],
            (row["effect_all_baseline"], row["effect_all_tm"]),
            fontsize=8,
            alpha=0.8,
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.legend(loc="upper left")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG_DIR / "fig2_baseline_vs_timematched.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR / 'fig2_baseline_vs_timematched.png'}")


# Two bars: average spillover at distance 1 vs distance 2+.
def fig3_distance_decay() -> None:
    df = _load_time_matched()
    mean_d1 = df["effect_d1"].dropna().mean()
    mean_d2p = df["effect_d2p"].dropna().mean()
    se_d1 = df["effect_d1"].dropna().sem() if df["effect_d1"].notna().sum() > 1 else 0.0
    se_d2p = df["effect_d2p"].dropna().sem() if df["effect_d2p"].notna().sum() > 1 else 0.0

    fig, ax = plt.subplots(figsize=(5, 5))
    x = [0, 1]
    means = [mean_d1, mean_d2p]
    errs = [se_d1, se_d2p]
    ax.bar(x, means, yerr=errs, capsize=8, color=["steelblue", "coral"], alpha=0.85, edgecolor="navy")
    ax.set_xticks(x)
    ax.set_xticklabels(["Distance 1", "Distance 2+"])
    ax.set_ylabel("Average spillover effect")
    ax.set_title("Structural decay: average spillover by network distance")
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.5)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG_DIR / "fig3_distance_decay.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR / 'fig3_distance_decay.png'}")


# Line network graph: node size = spillover, edge width = shared stations.
def fig4_network_structure() -> None:
    G = _load_graph()
    df = _load_time_matched()
    effect_map = df.set_index("line_id")["effect_all"].to_dict()

    effects = np.array([effect_map.get(n, 0) for n in G.nodes()])
    effects_min, effects_max = effects.min(), effects.max()
    if effects_max > effects_min:
        normalized = (effects - effects_min) / (effects_max - effects_min)
    else:
        normalized = np.ones_like(effects) * 0.5
    node_sizes = 300 + 1200 * normalized

    pos = nx.spring_layout(G, k=1.5, seed=42, weight="weight")
    edge_widths = [G[u][v].get("weight", 1) * 0.5 for u, v in G.edges()]

    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="steelblue", alpha=0.85, edgecolors="navy")
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color="gray", alpha=0.5)
    ax.set_title("Line network: node size = spillover effect, edge width = shared stations")
    ax.axis("off")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(FIG_DIR / "fig4_network_structure.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {FIG_DIR / 'fig4_network_structure.png'}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig1_spillover_ranking()
    fig2_baseline_vs_timematched()
    fig3_distance_decay()
    fig4_network_structure()
    print("All figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
