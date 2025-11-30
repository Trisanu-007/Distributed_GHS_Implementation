"""
Simplified GHS Algorithm Implementation for Distributed MST
Uses threads to simulate distributed nodes
"""

import os
import threading
import queue
import time
import random
import networkx as nx
import matplotlib.pyplot as plt
import json
from enum import Enum


class MessageType(Enum):
    CONNECT = "CONNECT"
    INITIATE = "INITIATE"
    TEST = "TEST"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REPORT = "REPORT"
    CHANGEROOT = "CHANGEROOT"


class EdgeState(Enum):
    BASIC = "BASIC"
    BRANCH = "BRANCH"
    REJECTED = "REJECTED"


class NodeState(Enum):
    SLEEPING = "SLEEPING"
    FIND = "FIND"
    FOUND = "FOUND"


class Message:
    def __init__(self, msg_type, sender, **kwargs):
        self.msg_type = msg_type
        self.sender = sender
        self.data = kwargs


class Node(threading.Thread):
    def __init__(self, node_id, neighbors, graph, termination_event):
        super().__init__()
        self.daemon = True
        self.node_id = node_id
        self.graph = graph
        self.neighbors = neighbors  # dict: {neighbor_id: weight}
        self.termination_event = termination_event

        # GHS state
        self.state = NodeState.SLEEPING
        self.level = 0
        self.fragment_id = node_id
        self.find_count = 0
        self.best_edge = None
        self.best_weight = float("inf")
        self.test_edge = None
        self.in_branch = None

        # Edge states
        self.edge_states = {neighbor: EdgeState.BASIC for neighbor in neighbors}

        # Message queue
        self.message_queue = queue.Queue()
        self.pending_messages = {}  # Track requeued messages to avoid infinite loops
        self.max_requeue = 10  # Maximum times to requeue a message

        # Control
        self.running = True
        self.terminated = False
        self.idle_count = 0
        self.max_idle = 50

        # Lock
        self.lock = threading.RLock()

    def send_message(self, target, msg_type, **kwargs):
        """Send a message to another node"""
        if target in self.graph.nodes:
            msg = Message(msg_type, self.node_id, **kwargs)
            self.graph.nodes[target]["node"].message_queue.put(msg)

    def requeue_message(self, msg):
        """Requeue a message with tracking to prevent infinite loops"""
        msg_key = (msg.msg_type, msg.sender, tuple(sorted(msg.data.items())))

        requeue_count = self.pending_messages.get(msg_key, 0)
        if requeue_count < self.max_requeue:
            self.pending_messages[msg_key] = requeue_count + 1
            time.sleep(0.02)
            self.message_queue.put(msg)
            return True
        else:
            # Message requeued too many times
            return False

    def run(self):
        """Main thread execution"""
        # Wake up and start
        self.wakeup()

        # Process messages
        while self.running and not self.termination_event.is_set():
            try:
                msg = self.message_queue.get(timeout=0.05)
                self.process_message(msg)
                self.idle_count = 0
            except queue.Empty:
                self.idle_count += 1
                if self.check_local_termination():
                    continue

    def wakeup(self):
        """Initialize by finding minimum weight edge"""
        with self.lock:
            if self.state == NodeState.SLEEPING:
                # Find minimum weight edge
                if not self.neighbors:
                    self.state = NodeState.FOUND
                    self.terminated = True
                    return

                min_neighbor = min(self.neighbors.items(), key=lambda x: x[1])[0]

                # Mark as branch
                self.edge_states[min_neighbor] = EdgeState.BRANCH
                self.level = 0
                self.state = NodeState.FOUND
                self.find_count = 0

                # Send CONNECT
                self.send_message(min_neighbor, MessageType.CONNECT, level=0)

    def process_message(self, msg):
        """Process incoming messages"""
        handlers = {
            MessageType.CONNECT: self.handle_connect,
            MessageType.INITIATE: self.handle_initiate,
            MessageType.TEST: self.handle_test,
            MessageType.ACCEPT: self.handle_accept,
            MessageType.REJECT: self.handle_reject,
            MessageType.REPORT: self.handle_report,
            MessageType.CHANGEROOT: self.handle_changeroot,
        }

        handler = handlers.get(msg.msg_type)
        if handler:
            handler(msg)

    def handle_connect(self, msg):
        """Handle CONNECT message"""
        sender = msg.sender
        level = msg.data["level"]

        with self.lock:
            if self.state == NodeState.SLEEPING:
                self.wakeup()

            if level < self.level:
                # Absorb lower level fragment
                self.edge_states[sender] = EdgeState.BRANCH
                self.send_message(
                    sender,
                    MessageType.INITIATE,
                    level=self.level,
                    fragment_id=self.fragment_id,
                    state=self.state.value,
                )
            elif self.edge_states[sender] == EdgeState.BASIC:
                # Edge still BASIC, wait for it to be processed
                if not self.requeue_message(msg):
                    # Too many requeues, treat as merge
                    self.edge_states[sender] = EdgeState.BRANCH
                    self.send_message(
                        sender,
                        MessageType.INITIATE,
                        level=self.level + 1,
                        fragment_id=max(self.node_id, sender),
                        state=NodeState.FIND.value,
                    )
            elif level == self.level:
                # Both at same level, merge fragments
                self.edge_states[sender] = EdgeState.BRANCH
                new_fragment_id = max(self.fragment_id, sender)
                self.send_message(
                    sender,
                    MessageType.INITIATE,
                    level=self.level + 1,
                    fragment_id=new_fragment_id,
                    state=NodeState.FIND.value,
                )
            else:
                # Requeue if level > self.level
                self.requeue_message(msg)

    def handle_initiate(self, msg):
        """Handle INITIATE message"""
        sender = msg.sender
        level = msg.data["level"]
        fragment_id = msg.data["fragment_id"]
        state_value = msg.data["state"]

        with self.lock:
            self.level = level
            self.fragment_id = fragment_id
            self.state = NodeState(state_value)
            self.in_branch = sender
            self.best_edge = None
            self.best_weight = float("inf")

            # Propagate to branches
            for neighbor in self.neighbors:
                if (
                    neighbor != sender
                    and self.edge_states[neighbor] == EdgeState.BRANCH
                ):
                    self.send_message(
                        neighbor,
                        MessageType.INITIATE,
                        level=level,
                        fragment_id=fragment_id,
                        state=state_value,
                    )
                    if self.state == NodeState.FIND:
                        self.find_count += 1

            if self.state == NodeState.FIND:
                self.test()

    def test(self):
        """Test for minimum weight outgoing edge"""
        # Find minimum BASIC edge
        basic_edges = [
            (n, w)
            for n, w in self.neighbors.items()
            if self.edge_states[n] == EdgeState.BASIC
        ]

        if basic_edges:
            self.test_edge = min(basic_edges, key=lambda x: x[1])[0]
            self.send_message(
                self.test_edge,
                MessageType.TEST,
                level=self.level,
                fragment_id=self.fragment_id,
            )
        else:
            self.test_edge = None
            self.report()

    def handle_test(self, msg):
        """Handle TEST message"""
        sender = msg.sender
        level = msg.data["level"]
        fragment_id = msg.data["fragment_id"]

        with self.lock:
            if self.state == NodeState.SLEEPING:
                self.wakeup()

            if level > self.level:
                # Higher level test, wait for our level to catch up
                if not self.requeue_message(msg):
                    # If requeued too many times, reject
                    self.send_message(sender, MessageType.REJECT)
            elif fragment_id == self.fragment_id:
                # Same fragment - internal edge
                if self.edge_states[sender] == EdgeState.BASIC:
                    self.edge_states[sender] = EdgeState.REJECTED
                if sender != self.test_edge:
                    self.send_message(sender, MessageType.REJECT)
                else:
                    self.test()
            else:
                # Different fragment - potential MST edge
                self.send_message(sender, MessageType.ACCEPT)

    def handle_accept(self, msg):
        """Handle ACCEPT message"""
        sender = msg.sender

        with self.lock:
            self.test_edge = None
            if self.neighbors[sender] < self.best_weight:
                self.best_edge = sender
                self.best_weight = self.neighbors[sender]
            self.report()

    def handle_reject(self, msg):
        """Handle REJECT message"""
        sender = msg.sender

        with self.lock:
            if self.edge_states[sender] == EdgeState.BASIC:
                self.edge_states[sender] = EdgeState.REJECTED
            self.test()

    def report(self):
        """Report best edge to parent"""
        # Ensure find_count doesn't go negative
        if self.find_count < 0:
            self.find_count = 0

        if self.find_count == 0 and self.test_edge is None:
            self.state = NodeState.FOUND
            if self.in_branch is not None:
                self.send_message(
                    self.in_branch, MessageType.REPORT, weight=self.best_weight
                )
            else:
                # Root node - initiate changeroot if we have an outgoing edge
                if self.best_weight < float("inf"):
                    self.changeroot()
                else:
                    self.terminated = True

    def handle_report(self, msg):
        """Handle REPORT message"""
        sender = msg.sender
        weight = msg.data["weight"]

        with self.lock:
            if sender != self.in_branch:
                # Report from child
                if self.find_count > 0:
                    self.find_count -= 1
                if weight < self.best_weight:
                    self.best_weight = weight
                    self.best_edge = sender
                self.report()
            elif self.state == NodeState.FIND:
                # Still in FIND state, wait
                if not self.requeue_message(msg):
                    # Requeued too many times, force transition to FOUND
                    self.state = NodeState.FOUND
                    if weight > self.best_weight:
                        self.changeroot()
            elif self.state == NodeState.FOUND:
                # Already in FOUND, check if we should change root
                if weight > self.best_weight:
                    self.changeroot()
                elif weight == self.best_weight and weight < float("inf"):
                    # Both have same weight, use node ID to break tie
                    if self.node_id < sender:
                        self.changeroot()
            elif weight == float("inf") and self.best_weight == float("inf"):
                # No more outgoing edges
                self.terminated = True

    def changeroot(self):
        """Change root of fragment"""
        with self.lock:
            # Check if best_edge exists
            if self.best_edge is None:
                return

            # Ensure best_edge is valid
            if self.best_edge not in self.edge_states:
                return

            edge_state = self.edge_states[self.best_edge]

            if edge_state == EdgeState.BRANCH:
                # Already a branch, propagate changeroot
                self.send_message(self.best_edge, MessageType.CHANGEROOT)
            elif edge_state == EdgeState.BASIC:
                # Mark as branch and connect
                self.edge_states[self.best_edge] = EdgeState.BRANCH
                self.send_message(self.best_edge, MessageType.CONNECT, level=self.level)
                # Reset for next phase
                self.best_edge = None
                self.best_weight = float("inf")
            elif edge_state == EdgeState.REJECTED:
                # Edge was rejected, find another
                self.best_edge = None
                self.best_weight = float("inf")
                self.state = NodeState.FIND
                self.test()

    def handle_changeroot(self, msg):
        """Handle CHANGEROOT message"""
        self.changeroot()

    def check_local_termination(self):
        """Check if this node believes algorithm might be done"""
        with self.lock:
            # Periodic check for stuck root nodes
            if (
                self.state == NodeState.FOUND
                and self.in_branch is None
                and self.best_weight < float("inf")
                and self.idle_count > 20
            ):
                # Root node stuck with outgoing edge - try changeroot again
                self.changeroot()
                self.idle_count = 0
                return False

            if (
                self.state == NodeState.FOUND
                and self.test_edge is None
                and self.find_count == 0
                and self.best_weight == float("inf")
                and self.idle_count > self.max_idle
            ):
                self.terminated = True
                return True
            return False


class GHSAlgorithm:
    def __init__(self, num_nodes, edges):
        """
        Initialize GHS algorithm
        edges: list of tuples (u, v, weight)
        """
        self.num_nodes = num_nodes
        self.edges = edges
        self.graph = nx.Graph()
        self.termination_event = threading.Event()

        # Build graph
        for u, v, w in edges:
            self.graph.add_edge(u, v, weight=w)

        # Create nodes
        self.nodes = {}
        for node_id in range(num_nodes):
            neighbors = {}
            for neighbor in self.graph.neighbors(node_id):
                neighbors[neighbor] = self.graph[node_id][neighbor]["weight"]

            node = Node(node_id, neighbors, self.graph, self.termination_event)
            self.nodes[node_id] = node
            self.graph.nodes[node_id]["node"] = node

    def run(self, timeout=10):
        """Run the GHS algorithm"""
        print("Starting GHS Algorithm...")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of edges: {len(self.edges)}")

        # Start all threads
        for node in self.nodes.values():
            node.start()

        # Wait for termination or timeout
        start_time = time.time()
        check_interval = 0.2
        last_check = 0

        while time.time() - start_time < timeout:
            current_time = time.time() - start_time

            # Periodically check for global termination
            if current_time - last_check >= check_interval:
                if self.check_global_termination():
                    elapsed = time.time() - start_time
                    print(f"Algorithm completed in {elapsed:.2f} seconds")
                    self.termination_event.set()
                    break
                last_check = current_time

            time.sleep(0.1)
        else:
            print(f"Timeout after {timeout} seconds - collecting results")
            self.termination_event.set()

        # Stop all nodes
        for node in self.nodes.values():
            node.running = False

        # Give threads time to finish
        time.sleep(0.2)

        # Collect MST edges from BRANCH edges
        mst_edges = set()
        for node_id in self.nodes:
            node = self.nodes[node_id]
            for neighbor, state in node.edge_states.items():
                if state == EdgeState.BRANCH and node_id < neighbor:
                    mst_edges.add((node_id, neighbor))

        print(f"Found {len(mst_edges)} MST edges")
        return list(mst_edges)

    def check_global_termination(self):
        """Check if all nodes have reached a stable state"""
        stable_nodes = 0
        fragment_ids = set()

        for node in self.nodes.values():
            with node.lock:
                fragment_ids.add(node.fragment_id)
                if (
                    node.state == NodeState.FOUND
                    and node.test_edge is None
                    and node.find_count == 0
                ):
                    stable_nodes += 1

        all_stable = stable_nodes == self.num_nodes
        single_fragment = len(fragment_ids) == 1

        # Check if we have n-1 branch edges (complete MST)
        branch_count = 0
        for node in self.nodes.values():
            with node.lock:
                for state in node.edge_states.values():
                    if state == EdgeState.BRANCH:
                        branch_count += 1
        branch_count //= 2

        complete_mst = branch_count == self.num_nodes - 1

        # Verify MST is connected if we have the right number of edges
        if complete_mst:
            mst_connected = self.verify_mst_connected()
            return mst_connected

        return (all_stable and single_fragment) or complete_mst

    def verify_mst_connected(self):
        """Verify that the MST forms a connected spanning tree"""
        # Build adjacency list from BRANCH edges
        mst_adj = {i: set() for i in range(self.num_nodes)}

        for node_id in self.nodes:
            node = self.nodes[node_id]
            with node.lock:
                for neighbor, state in node.edge_states.items():
                    if state == EdgeState.BRANCH:
                        mst_adj[node_id].add(neighbor)

        # BFS to check if all nodes are reachable from node 0
        visited = set([0])
        queue_bfs = [0]

        while queue_bfs:
            current = queue_bfs.pop(0)
            for neighbor in mst_adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue_bfs.append(neighbor)

        # MST is connected if all nodes are visited
        return len(visited) == self.num_nodes

    def print_debug_info(self):
        """Print detailed debug information about node states"""
        print("\nNode States:")
        print(
            f"{'Node':<6} {'State':<10} {'Level':<7} {'Fragment':<10} {'FindCnt':<9} {'BestWt':<9} {'TestEdge':<10}"
        )
        print("-" * 75)

        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            with node.lock:
                state_str = (
                    node.state.value
                    if hasattr(node.state, "value")
                    else str(node.state)
                )
                print(
                    f"{node_id:<6} {state_str:<10} {node.level:<7} {node.fragment_id:<10} "
                    f"{node.find_count:<9} {node.best_weight:<9.1f} {str(node.test_edge):<10}"
                )

        print("\nEdge States:")
        print(f"{'Node':<6} {'Neighbor':<10} {'State':<12} {'Weight':<8}")
        print("-" * 40)

        edge_info = []
        for node_id in sorted(self.nodes.keys()):
            node = self.nodes[node_id]
            with node.lock:
                for neighbor, state in sorted(node.edge_states.items()):
                    if node_id < neighbor:  # Print each edge only once
                        weight = self.graph[node_id][neighbor]["weight"]
                        state_str = (
                            state.value if hasattr(state, "value") else str(state)
                        )
                        edge_info.append((node_id, neighbor, state_str, weight))

        for node_id, neighbor, state_str, weight in sorted(edge_info):
            print(f"{node_id:<6} {neighbor:<10} {state_str:<12} {weight:<8}")

        # Fragment analysis
        print("\nFragment Analysis:")
        fragments = {}
        for node_id in self.nodes:
            node = self.nodes[node_id]
            with node.lock:
                frag_id = node.fragment_id
                if frag_id not in fragments:
                    fragments[frag_id] = []
                fragments[frag_id].append(node_id)

        print(f"Number of fragments: {len(fragments)}")
        for frag_id, nodes_list in sorted(fragments.items()):
            print(f"  Fragment {frag_id}: nodes {sorted(nodes_list)}")

        # BRANCH edge connectivity
        print("\nBRANCH Edge Connectivity:")
        branch_edges = []
        for node_id in self.nodes:
            node = self.nodes[node_id]
            with node.lock:
                for neighbor, state in node.edge_states.items():
                    if state == EdgeState.BRANCH and node_id < neighbor:
                        branch_edges.append((node_id, neighbor))

        print(f"BRANCH edges: {sorted(branch_edges)}")

        # Check which nodes are unreachable
        if branch_edges:
            adj = {i: set() for i in range(self.num_nodes)}
            for u, v in branch_edges:
                adj[u].add(v)
                adj[v].add(u)

            visited = set([0])
            queue = [0]
            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)

            unreachable = set(range(self.num_nodes)) - visited
            if unreachable:
                print(f"⚠ Unreachable nodes from node 0: {sorted(unreachable)}")
            else:
                print("✓ All nodes reachable via BRANCH edges")

    def visualize(self, mst_edges, save_path="ghs_mst.png"):
        """Visualize the graph and MST"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Layout
        pos = nx.spring_layout(self.graph, seed=42)

        # Original graph
        ax1.set_title("Original Graph", fontsize=14, fontweight="bold")
        nx.draw(
            self.graph,
            pos,
            ax=ax1,
            with_labels=True,
            node_color="lightblue",
            node_size=700,
            font_size=12,
            font_weight="bold",
        )
        edge_labels = nx.get_edge_attributes(self.graph, "weight")
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, ax=ax1)

        # MST
        ax2.set_title("MST (GHS Algorithm)", fontsize=14, fontweight="bold")
        mst_graph = nx.Graph()
        for u, v in mst_edges:
            weight = self.graph[u][v]["weight"]
            mst_graph.add_edge(u, v, weight=weight)

        # Add all nodes
        for node in self.graph.nodes():
            if node not in mst_graph.nodes():
                mst_graph.add_node(node)

        nx.draw(
            mst_graph,
            pos,
            ax=ax2,
            with_labels=True,
            node_color="lightgreen",
            node_size=700,
            font_size=12,
            font_weight="bold",
            edge_color="red",
            width=3,
        )

        if mst_edges:
            edge_labels = nx.get_edge_attributes(mst_graph, "weight")
            nx.draw_networkx_edge_labels(mst_graph, pos, edge_labels, ax=ax2)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to {save_path}")
        plt.close()

        return mst_graph


def create_random_graph(num_nodes=8, edge_probability=0.4, seed=42):
    """Create a random connected graph with random weights"""
    random.seed(seed)

    # Generate random graph using Erdos-Renyi model
    G = nx.erdos_renyi_graph(num_nodes, edge_probability, seed=seed)

    # Ensure the graph is connected
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(
            num_nodes, edge_probability, seed=random.randint(0, 1000)
        )

    # Assign random weights to edges
    edges = []
    for u, v in G.edges():
        weight = random.randint(1, 10)
        edges.append((u, v, weight))

    return num_nodes, edges


def run_experiment(num_nodes, edges, experiment_num, debug=False):
    """Run GHS algorithm on a single graph configuration"""
    print(f"\n{'=' * 70}")
    print(f"Experiment {experiment_num}: {num_nodes} nodes, {len(edges)} edges")
    print("=" * 70)

    # Run GHS
    ghs = GHSAlgorithm(num_nodes, edges)
    mst_edges = ghs.run(timeout=15)

    # Debug information if enabled
    if debug or len(mst_edges) != num_nodes - 1:
        print(f"\n--- Debug Info for Experiment {experiment_num} ---")
        ghs.print_debug_info()

    # Calculate results
    total_weight = 0
    for u, v in sorted(mst_edges):
        weight = ghs.graph[u][v]["weight"]
        total_weight += weight

    # Verify with NetworkX
    nx_mst = nx.minimum_spanning_tree(ghs.graph, weight="weight")
    nx_edges = sorted([(min(u, v), max(u, v)) for u, v in nx_mst.edges()])
    nx_weight = sum(ghs.graph[u][v]["weight"] for u, v in nx_edges)

    is_correct = set(mst_edges) == set(nx_edges)

    # Print results
    print(f"\nMST Weight: {total_weight}")
    print(f"MST Edges Found: {len(mst_edges)}/{num_nodes - 1} expected")
    print(f"NetworkX MST Weight: {nx_weight}")
    print(f"Status: {'✓ CORRECT' if is_correct else '✗ INCORRECT'}")

    root_folder = "mst_visualizations"
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)

    # Save visualization
    filename = f"{root_folder}/ghs_mst_exp{experiment_num}.png"
    ghs.visualize(mst_edges, filename)

    return {
        "experiment": experiment_num,
        "num_nodes": num_nodes,
        "num_edges": len(edges),
        "mst_edges": [(u, v, ghs.graph[u][v]["weight"]) for u, v in sorted(mst_edges)],
        "mst_weight": total_weight,
        "networkx_weight": nx_weight,
        "is_correct": is_correct,
        "edges_found": len(mst_edges),
        "edges_expected": num_nodes - 1,
    }


def main():
    """Main function - Loop through multiple graph configurations"""

    print("=" * 70)
    print(" " * 15 + "GHS Algorithm - Multiple Experiments")
    print("=" * 70)

    # Define 6 different graph configurations
    graph_configs = [
        {"num_nodes": 5, "edge_probability": 0.5, "seed": 42},
        {"num_nodes": 6, "edge_probability": 0.4, "seed": 100},
        {"num_nodes": 7, "edge_probability": 0.6, "seed": 200},
        {"num_nodes": 6, "edge_probability": 0.7, "seed": 300},
        {"num_nodes": 10, "edge_probability": 0.8, "seed": 400},
        {"num_nodes": 20, "edge_probability": 0.3, "seed": 500},
    ]

    all_results = []

    # Run experiments for each configuration
    for i, config in enumerate(graph_configs, 1):
        # Generate random graph
        num_nodes, edges = create_random_graph(
            num_nodes=config["num_nodes"],
            edge_probability=config["edge_probability"],
            seed=config["seed"],
        )

        # Run experiment
        result = run_experiment(num_nodes, edges, i)
        all_results.append(result)

    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "SUMMARY")
    print("=" * 70)
    print(
        f"{'Exp':<5} {'Nodes':<7} {'Edges':<7} {'MST Wt':<9} {'Found':<10} {'Status':<10}"
    )
    print("-" * 70)

    for result in all_results:
        status = "✓ PASS" if result["is_correct"] else "✗ FAIL"
        found_str = f"{result['edges_found']}/{result['edges_expected']}"
        print(
            f"{result['experiment']:<5} {result['num_nodes']:<7} {result['num_edges']:<7} "
            f"{result['mst_weight']:<9} {found_str:<10} {status:<10}"
        )

    # Save all results
    with open("ghs_experiments.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 70)
    print("All results saved to: ghs_experiments.json")
    print("Visualizations saved as: ghs_mst_exp1.png, ghs_mst_exp2.png, etc.")
    print("=" * 70)


if __name__ == "__main__":
    main()
