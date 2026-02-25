import logging
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

from src.logging_config import setup_logging


IN_GRAPH = Path("data/processed/line_graph.pkl")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "line_distances.parquet"


def load_graph(path: Path) -> nx.Graph:
    """Load the line graph pkl file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Line graph pickle not found at {path}. "
            "Run 03_build_line_graph.py first."
        )

    with open(path, "rb") as f:
        G: nx.Graph = pickle.load(f)

    return G


def compute_unweighted_shortest_paths(G: nx.Graph) -> pd.DataFrame:
    """
    Compute unweighted shortest path distances between all pairs of lines.
    """
    records = []
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        for tgt, d in lengths.items():
            records.append(
                {
                    "source_line": src,
                    "target_line": tgt,
                    "distance": int(d),
                }
            )

    df = pd.DataFrame.from_records(records)

    return df


def compute_distances() -> None:
    logging.info(f"Loading line graph from {IN_GRAPH}")
    G = load_graph(IN_GRAPH)

    logging.info(
        f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )

    df_dist = compute_unweighted_shortest_paths(G)

    # checks
    n_pairs = len(df_dist)
    logging.info(f"Computed distances for {n_pairs} pairs.")

    mask_offdiag = df_dist["source_line"] != df_dist["target_line"]
    offdiag = df_dist[mask_offdiag]
    if not offdiag.empty:
        logging.info(
            "Off-diagonal distance summary:\n"
            + offdiag["distance"].describe().to_string()
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_dist.to_parquet(OUT_FILE, index=False)
    logging.info(f"Wrote distances to {OUT_FILE}")


if __name__ == "__main__":
    setup_logging()
    compute_distances()

