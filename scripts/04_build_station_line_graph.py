from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import pandas as pd


line_to_stations = pd.read_csv("")

# station -> list of lines
station_to_lines = defaultdict(list)

for line, stations in line_to_stations.items():
    for station in stations:
        station_to_lines[station].append(line)

print("\nStation → Lines (sample):")
for k in list(station_to_lines)[:3]:
    print(k, station_to_lines[k])



G = nx.Graph()

for station, lines in station_to_lines.items():
    if len(lines) < 2:
        continue

    for a, b in combinations(lines, 2):
        if G.has_edge(a, b):
            G[a][b]["weight"] += 1
        else:
            G.add_edge(a, b, weight=1)


print(G)
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

print("\nEdges with weights:")
for u, v, d in G.edges(data=True):
    print(f"{u} — {v}: {d['weight']}")


# plt.figure(figsize=(10, 10))

# pos = nx.spring_layout(G, seed=42, weight="weight")

# # Node sizes = degree
# node_sizes = [300 + 300 * G.degree(n) for n in G.nodes()]

# # Edge widths = weight
# edge_widths = [G[u][v]["weight"] for u, v in G.edges()]

# nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="lightblue")
# nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.7)
# nx.draw_networkx_labels(G, pos, font_size=10)

# plt.title("London Underground Line Interchange Network")
# plt.axis("off")
# plt.show()