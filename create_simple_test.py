"""
Create a simple test graph for debugging MPI GHS
"""

import json
import os


def create_simple_test():
    """Create a simple 3-node test graph"""
    # Graph: 0 -- 1 (weight 1)
    #        |
    #        2 (weight 2)
    #        Expected MST: 0-1(1), 0-2(2)

    graph_dir = "test_graph_data"
    os.makedirs(graph_dir, exist_ok=True)

    # Node 0 neighbors: 1(weight 1), 2(weight 2)
    with open(os.path.join(graph_dir, "node_0.json"), "w") as f:
        json.dump(
            {"node_id": 0, "neighbors": {"1": 1, "2": 2}, "num_neighbors": 2},
            f,
            indent=2,
        )

    # Node 1 neighbors: 0(weight 1)
    with open(os.path.join(graph_dir, "node_1.json"), "w") as f:
        json.dump(
            {"node_id": 1, "neighbors": {"0": 1}, "num_neighbors": 1}, f, indent=2
        )

    # Node 2 neighbors: 0(weight 2)
    with open(os.path.join(graph_dir, "node_2.json"), "w") as f:
        json.dump(
            {"node_id": 2, "neighbors": {"0": 2}, "num_neighbors": 1}, f, indent=2
        )

    # Metadata
    with open(os.path.join(graph_dir, "graph_metadata.json"), "w") as f:
        json.dump(
            {"num_nodes": 3, "num_edges": 2, "edges": [[0, 1, 1], [0, 2, 2]]},
            f,
            indent=2,
        )

    print("Created simple test graph:")
    print("  Nodes: 3")
    print("  Edges: 0-1(1), 0-2(2)")
    print("  Expected MST weight: 3")


if __name__ == "__main__":
    create_simple_test()
