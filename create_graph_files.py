"""
Create graph files for MPI-based GHS algorithm
Each node gets a separate file with its neighbor information
"""

import networkx as nx
import random
import json
import os
import matplotlib.pyplot as plt


def create_random_graph(num_nodes=6, edge_probability=0.5, seed=42):
    """Create a random connected graph with random weights"""
    random.seed(seed)

    # Generate random graph using Erdos-Renyi model
    G = nx.erdos_renyi_graph(num_nodes, edge_probability, seed=seed)

    # Ensure the graph is connected
    attempts = 0
    while not nx.is_connected(G) and attempts < 100:
        G = nx.erdos_renyi_graph(
            num_nodes, edge_probability, seed=random.randint(0, 10000)
        )
        attempts += 1

    if not nx.is_connected(G):
        # Force connectivity by adding edges
        components = list(nx.connected_components(G))
        for i in range(len(components) - 1):
            node1 = list(components[i])[0]
            node2 = list(components[i + 1])[0]
            G.add_edge(node1, node2)

    # Assign random weights to edges
    for u, v in G.edges():
        G[u][v]["weight"] = random.randint(1, 10)

    return G


def create_node_files(graph, output_dir="graph_data"):
    """
    Create individual files for each node containing neighbor information
    Format: node_<id>.json with neighbor information
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    num_nodes = graph.number_of_nodes()

    print(f"Creating graph files for {num_nodes} nodes...")
    print(f"Output directory: {output_dir}")

    # Create file for each node
    for node_id in range(num_nodes):
        # Get neighbor information
        neighbors = {}
        for neighbor in graph.neighbors(node_id):
            weight = graph[node_id][neighbor]["weight"]
            neighbors[neighbor] = weight

        # Create node data
        node_data = {
            "node_id": node_id,
            "neighbors": neighbors,
            "num_neighbors": len(neighbors),
        }

        # Write to file
        filename = os.path.join(output_dir, f"node_{node_id}.json")
        with open(filename, "w") as f:
            json.dump(node_data, f, indent=2)

        print(f"  Created {filename}: Node {node_id} with {len(neighbors)} neighbors")

    # Create graph metadata file
    metadata = {
        "num_nodes": num_nodes,
        "num_edges": graph.number_of_edges(),
        "edges": [(u, v, graph[u][v]["weight"]) for u, v in graph.edges()],
    }

    metadata_file = os.path.join(output_dir, "graph_metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Created {metadata_file}: Graph metadata")

    # Visualize the graph
    visualize_graph(graph, output_dir)

    return output_dir


def visualize_graph(graph, output_dir):
    """Visualize the graph and save to file"""
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph, seed=42)

    # Draw graph
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_color="lightblue",
        node_size=700,
        font_size=12,
        font_weight="bold",
        edge_color="gray",
        width=2,
    )

    # Draw edge labels
    edge_labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=10)

    plt.title("Input Graph for GHS Algorithm (MPI)", fontsize=14, fontweight="bold")

    output_file = os.path.join(output_dir, "input_graph.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"\n  Visualization saved to {output_file}")
    plt.close()


def print_graph_summary(graph):
    """Print summary of the graph"""
    print("\n" + "=" * 70)
    print("Graph Summary")
    print("=" * 70)
    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")
    print(f"Is connected: {nx.is_connected(graph)}")

    print("\nEdge list (with weights):")
    for u, v, data in sorted(graph.edges(data=True)):
        print(f"  ({u}, {v}): weight = {data['weight']}")

    # Calculate expected MST weight using NetworkX
    mst = nx.minimum_spanning_tree(graph, weight="weight")
    mst_weight = sum(data["weight"] for _, _, data in mst.edges(data=True))
    print(f"\nExpected MST weight (NetworkX): {mst_weight}")
    print("=" * 70)


def main():
    """Main function to create graph files"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate graph files for MPI-based GHS algorithm"
    )
    parser.add_argument(
        "--nodes", type=int, default=6, help="Number of nodes (default: 6)"
    )
    parser.add_argument(
        "--edge-prob", type=float, default=0.5, help="Edge probability (default: 0.5)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="graph_data",
        help="Output directory (default: graph_data)",
    )

    args = parser.parse_args()

    num_nodes = args.nodes
    edge_probability = args.edge_prob
    seed = args.seed
    output_dir = args.output_dir

    print("=" * 70)
    print("Graph File Generator for MPI-based GHS Algorithm")
    print("=" * 70)

    # Create random graph
    print(f"\nGenerating random graph...")
    print(f"  Nodes: {num_nodes}")
    print(f"  Edge probability: {edge_probability}")
    print(f"  Random seed: {seed}")

    graph = create_random_graph(num_nodes, edge_probability, seed)

    # Print graph summary
    print_graph_summary(graph)

    # Create node files
    print("\n" + "=" * 70)
    create_node_files(graph, output_dir)

    print("\n" + "=" * 70)
    print("Graph files created successfully!")
    print("=" * 70)
    print(f"\nTo run GHS algorithm with MPI:")
    print(f"  mpiexec -n {num_nodes} python ghs_implementation_mpi.py")
    print(f"\nFor SLURM cluster:")
    print(f"  sbatch --nodes={num_nodes} --ntasks={num_nodes} run_ghs.slurm")
    print("=" * 70)


if __name__ == "__main__":
    main()
