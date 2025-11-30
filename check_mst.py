import json
import networkx as nx

meta = json.load(open("test_graph_4/graph_metadata.json"))
G = nx.Graph()
for u, v, w in meta["edges"]:
    G.add_edge(u, v, weight=w)

mst = nx.minimum_spanning_tree(G)
print("Expected MST edges:")
for u, v in sorted(mst.edges()):
    print(f'  ({u},{v}): {G[u][v]["weight"]}')
print(f'\nTotal weight: {sum(d["weight"] for _, _, d in mst.edges(data=True))}')
print(f"Number of edges: {mst.number_of_edges()}")

# Check connectivity
print(f"\nOriginal graph connected: {nx.is_connected(G)}")
print(f"MST connected: {nx.is_connected(mst)}")
