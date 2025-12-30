import networkx as nx
import pickle
import matplotlib.pyplot as plt

def plot_static_graph(graph_path):
    with open(graph_path, 'rb') as f:
        G = pickle.load(f)
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=1.5, seed=42) # k=spread, seed=consistent layout
    
    # Scale edge widths by weight
    weights = [G[u][v]['weight'] * 0.5 for u, v in G.edges()]
    
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2000)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', alpha=0.5)
    
    plt.title("Structural Network of Tube Lines (Edge thickness = Shared Stations)")
    plt.axis('off')
    plt.savefig("viz/line_network_graph.png")
    plt.show()


plot_static_graph("data/processed/line_graph.pkl")