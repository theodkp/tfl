import logging
import pickle
from pathlib import Path

import pandas as pd

from src.logging_config import setup_logging


IN_GRAPH = Path("data/processed/line_graph.pkl")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "line_weighted_degree.parquet"


def load_graph(path: Path):
    """Load the line graph pkl file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Line graph pickle not found at {path}. "
            "Run 03_build_line_graph.py first."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def compute_weighted_degree(G) -> pd.DataFrame:
    """For each node, weighted_degree = sum of edge weights to neighbors."""
    records = []
    for node in G.nodes():
        w_deg = sum(
            G[node][nb].get("weight", 1)
            for nb in G.neighbors(node)
        )
        records.append({"line_id": node, "weighted_degree": float(w_deg)})
    return pd.DataFrame.from_records(records)


def line_weights() -> None:
    setup_logging()
    logging.info(f"Loading line graph from {IN_GRAPH}")
    G = load_graph(IN_GRAPH)
    logging.info(
        f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges."
    )
    df = compute_weighted_degree(G)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)
    logging.info(f"Wrote line weights to {OUT_FILE}")
    logging.info(df.sort_values("weighted_degree", ascending=False).to_string(index=False))


if __name__ == "__main__":
    line_weights()
