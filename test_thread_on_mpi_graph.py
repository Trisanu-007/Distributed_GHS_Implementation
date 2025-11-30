"""
Test thread-based GHS on the same graph as MPI version
"""

import json
import networkx as nx
from ghs_implementation import GHSAlgorithm, create_random_graph

# Load the graph from graph_data
with open("graph_data/graph_metadata.json", "r") as f:
    metadata = json.load(f)

# Create NetworkX graph
G = nx.Graph()
for u, v, w in metadata["edges"]:
    G.add_edge(u, v, weight=w)

print("=" * 70)
print("Testing Thread-based GHS on MPI graph")
print("=" * 70)
print(f"Nodes: {metadata['num_nodes']}")
print(f"Edges: {metadata['num_edges']}")
print("Edge list:")
for u, v, w in sorted(metadata["edges"]):
    print(f"  ({u}, {v}): {w}")

# Run GHS
ghs = GHSAlgorithm(metadata["num_nodes"], metadata["edges"])
mst_edges = ghs.run(timeout=5)

print(f"\nMST Edges found ({len(mst_edges)}):")
total_weight = 0
for u, v in sorted(mst_edges):
    weight = G[u][v]["weight"]
    print(f"  ({u}, {v}): weight = {weight}")
    total_weight += weight

print(f"\nTotal MST weight: {total_weight}")
print(f"Expected edges: {metadata['num_nodes'] - 1}")

if len(mst_edges) == metadata["num_nodes"] - 1:
    print("Correct number of edges!")
else:
    print("WARNING: Incorrect number of edges!")
    ghs.print_debug_info()
