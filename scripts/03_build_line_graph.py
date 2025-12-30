import pandas as pd
import networkx as nx
import pickle
import logging
from pathlib import Path
from itertools import combinations
from typing import Tuple

IN_FILE = Path("data/raw/stations/line_station.parquet")
OUT_DIR = Path("data/processed")
OUT_GRAPH = OUT_DIR / "line_graph.pkl" 
OUT_EDGES = OUT_DIR / "line_graph_edges.parquet"

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def build_line_graph(df: pd.DataFrame) -> Tuple[nx.Graph, pd.DataFrame]:
    """
    Builds graph
    """
    df = df.drop_duplicates(subset=["station_naptan", "line_id"])
    
    station_groups = df.groupby("station_naptan")["line_id"].apply(list)
    
    edge_list = []
    for lines in station_groups:
        if len(lines) >= 2:
            for u, v in combinations(sorted(lines), 2):
                edge_list.append({"source": u, "target": v})
    
    edge_df = pd.DataFrame(edge_list)
    weights = edge_df.groupby(["source", "target"]).size().reset_index(name="weight")
    
    G = nx.from_pandas_edgelist(weights, source="source", target="target", edge_attr="weight")
    
    all_lines = df["line_id"].unique()
    G.add_nodes_from(all_lines)

    return G, weights

if __name__ == "__main__":
    if not IN_FILE.exists():
        logging.error(f"Input file {IN_FILE} not found. Ensure station data is fetched first.")
        exit(1)

    logging.info(f"Reading station data from {IN_FILE}")
    raw_df = pd.read_parquet(IN_FILE)
    
    logging.info("Building tube line network graph...")
    G, weights_df = build_line_graph(raw_df)
    
    logging.info(f"Graph successfully built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(OUT_GRAPH, 'wb') as f:
        pickle.dump(G, f)
        
    weights_df.to_parquet(OUT_EDGES, index=False)
    
    logging.info(f"Graph saved to {OUT_GRAPH}")
    logging.info(f"Edges saved to {OUT_EDGES}")