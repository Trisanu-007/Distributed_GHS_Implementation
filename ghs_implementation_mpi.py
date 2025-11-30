"""
MPI-based GHS Algorithm Implementation
Each MPI process represents a node in the graph
"""

from mpi4py import MPI
import json
import os
import time
import sys
from enum import Enum


class MessageType(Enum):
    CONNECT = 0
    INITIATE = 1
    TEST = 2
    ACCEPT = 3
    REJECT = 4
    REPORT = 5
    CHANGEROOT = 6
    TERMINATE = 7


class EdgeState(Enum):
    BASIC = 0
    BRANCH = 1
    REJECTED = 2


class NodeState(Enum):
    SLEEPING = 0
    FIND = 1
    FOUND = 2


class MPINode:
    """GHS Node implementation using MPI"""

    def __init__(self, comm, rank, size, graph_dir="graph_data"):
        self.comm = comm
        self.rank = rank  # Node ID
        self.size = size  # Total number of nodes
        self.graph_dir = graph_dir

        # Load node information from file
        self.load_node_info()

        # GHS state
        self.state = NodeState.SLEEPING
        self.level = 0
        self.fragment_id = self.rank
        self.find_count = 0
        self.best_edge = None
        self.best_weight = float("inf")
        self.test_edge = None
        self.in_branch = None

        # Edge states
        self.edge_states = {neighbor: EdgeState.BASIC for neighbor in self.neighbors}

        # Message queue for deferred messages
        self.deferred_messages = []
        self.defer_count = {}

        # Termination - scale with graph size
        self.terminated = False
        self.idle_iterations = 0
        self.max_idle = max(100, self.size * 20)

    def load_node_info(self):
        """Load node information from file"""
        filename = os.path.join(self.graph_dir, f"node_{self.rank}.json")

        try:
            with open(filename, "r") as f:
                data = json.load(f)

            self.node_id = data["node_id"]
            # Convert string keys to integers
            self.neighbors = {int(k): v for k, v in data["neighbors"].items()}

            if self.rank == 0:
                print(
                    f"[Node {self.rank}] Loaded info: {len(self.neighbors)} neighbors"
                )
        except FileNotFoundError:
            print(f"[Node {self.rank}] ERROR: File {filename} not found!")
            self.neighbors = {}

    def send_message(self, target, msg_type, **data):
        """Send a message to another node"""
        if target < 0 or target >= self.size:
            return

        message = {"type": msg_type.value, "sender": self.rank, "data": data}
        print(
            f"[Node {self.rank}] >> Sending {msg_type.name} to Node {target} | Data: {data}"
        )
        self.comm.send(message, dest=target, tag=msg_type.value)

    def receive_message(self, timeout=0.01):
        """Try to receive a message (non-blocking)"""
        status = MPI.Status()

        if self.comm.iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status):
            message = self.comm.recv(source=status.Get_source(), tag=status.Get_tag())
            msg_type = MessageType(message["type"])
            sender = message["sender"]
            print(f"[Node {self.rank}] << Received {msg_type.name} from Node {sender}")
            return message
        return None

    def wakeup(self):
        """Initialize the algorithm"""
        if self.state == NodeState.SLEEPING and self.neighbors:
            # Find minimum weight edge
            min_neighbor = min(self.neighbors.items(), key=lambda x: x[1])[0]
            min_weight = self.neighbors[min_neighbor]

            print(f"\n[Node {self.rank}] *** WAKEUP | State: SLEEPING -> FOUND")
            print(
                f"[Node {self.rank}] +++ MST EDGE ADDED: ({self.rank}, {min_neighbor}) weight={min_weight}"
            )

            # Mark as branch
            self.edge_states[min_neighbor] = EdgeState.BRANCH
            self.level = 0
            self.state = NodeState.FOUND
            self.find_count = 0

            print(
                f"[Node {self.rank}] State: level=0, fragment={self.fragment_id}, state=FOUND"
            )

            # Send CONNECT
            self.send_message(
                min_neighbor, MessageType.CONNECT, level=0, fragment_id=self.fragment_id
            )

    def process_message(self, msg):
        """Process incoming message"""
        msg_type = MessageType(msg["type"])
        sender = msg["sender"]
        data = msg["data"]

        if msg_type == MessageType.CONNECT:
            self.handle_connect(sender, data)
        elif msg_type == MessageType.INITIATE:
            self.handle_initiate(sender, data)
        elif msg_type == MessageType.TEST:
            self.handle_test(sender, data)
        elif msg_type == MessageType.ACCEPT:
            self.handle_accept(sender, data)
        elif msg_type == MessageType.REJECT:
            self.handle_reject(sender, data)
        elif msg_type == MessageType.REPORT:
            self.handle_report(sender, data)
        elif msg_type == MessageType.CHANGEROOT:
            self.handle_changeroot(sender, data)
        elif msg_type == MessageType.TERMINATE:
            self.terminated = True

    def handle_connect(self, sender, data):
        """Handle CONNECT message"""
        level = data["level"]
        sender_fragment_id = data.get("fragment_id", sender)  # Fallback to sender ID

        print(f"\n[Node {self.rank}] === HANDLE CONNECT from Node {sender}")
        print(
            f"[Node {self.rank}] Sender: level={level}, fragment={sender_fragment_id}"
        )
        print(
            f"[Node {self.rank}] My state: level={self.level}, fragment={self.fragment_id}, state={self.state.name}"
        )

        if self.state == NodeState.SLEEPING:
            print(f"[Node {self.rank}] I was sleeping, waking up first...")
            self.wakeup()

        if level < self.level:
            print(
                f"[Node {self.rank}] Sender level < my level -> Absorbing into my fragment"
            )
            print(
                f"[Node {self.rank}] +++ MST EDGE ADDED: ({self.rank}, {sender}) weight={self.neighbors[sender]}"
            )
            self.edge_states[sender] = EdgeState.BRANCH
            self.send_message(
                sender,
                MessageType.INITIATE,
                level=self.level,
                fragment_id=self.fragment_id,
                state=self.state.value,
            )
            if self.state == NodeState.FIND:
                self.find_count += 1
        elif level == self.level:
            # Wait before processing to avoid race conditions
            time.sleep(0.001)

            print(f"[Node {self.rank}] Same level merge detected")
            # Only mark as BRANCH if not already marked
            if self.edge_states.get(sender) == EdgeState.BASIC:
                print(
                    f"[Node {self.rank}] +++ MST EDGE ADDED: ({self.rank}, {sender}) weight={self.neighbors[sender]}"
                )
                self.edge_states[sender] = EdgeState.BRANCH

            new_fragment_id = max(self.fragment_id, sender_fragment_id)
            new_level = self.level + 1

            # Determine priority: higher fragment_id, or if equal, higher node_id
            # The node with higher priority becomes the "parent" and sends INITIATE
            my_priority = (self.fragment_id, self.rank)
            sender_priority = (sender_fragment_id, sender)
            should_initiate = my_priority > sender_priority

            print(
                f"[Node {self.rank}] Priority check: my={my_priority} vs sender={sender_priority}"
            )
            if should_initiate:
                print(
                    f"[Node {self.rank}] I have priority -> Initiating merge to level {new_level}"
                )
                # We initiate the merge - update our own state first
                self.level = new_level
                self.fragment_id = new_fragment_id
                self.state = NodeState.FIND
                self.in_branch = None  # We're the root of the new fragment
                self.best_edge = None
                self.best_weight = float("inf")
                self.find_count = 0

                # Send INITIATE to sender
                self.send_message(
                    sender,
                    MessageType.INITIATE,
                    level=new_level,
                    fragment_id=new_fragment_id,
                    state=NodeState.FIND.value,
                )

                # Count branches for find_count (sender + any other branches)
                for neighbor in self.neighbors:
                    if self.edge_states[neighbor] == EdgeState.BRANCH:
                        self.find_count += 1

                # Propagate to other branches
                for neighbor in self.neighbors:
                    if (
                        neighbor != sender
                        and self.edge_states[neighbor] == EdgeState.BRANCH
                    ):
                        self.send_message(
                            neighbor,
                            MessageType.INITIATE,
                            level=new_level,
                            fragment_id=new_fragment_id,
                            state=NodeState.FIND.value,
                        )

                # Start testing
                time.sleep(0.002)
                self.test()

    def handle_initiate(self, sender, data):
        """Handle INITIATE message"""
        level = data["level"]
        fragment_id = data["fragment_id"]
        state_value = data["state"]

        old_level = self.level
        old_fragment = self.fragment_id

        print(f"\n[Node {self.rank}] ~~~ HANDLE INITIATE from Node {sender}")
        print(
            f"[Node {self.rank}] State change: level {old_level}->{level}, fragment {old_fragment}->{fragment_id}"
        )
        print(
            f"[Node {self.rank}] New state: {NodeState(state_value).name}, in_branch={sender}"
        )

        self.level = level
        self.fragment_id = fragment_id
        self.state = NodeState(state_value)
        self.in_branch = sender
        self.best_edge = None
        self.best_weight = float("inf")
        self.find_count = 0

        # Count branches (excluding sender)
        branch_neighbors = [
            n
            for n in self.neighbors
            if n != sender and self.edge_states[n] == EdgeState.BRANCH
        ]

        print(
            f"[Node {self.rank}] Branch neighbors (excluding sender): {branch_neighbors}"
        )

        # Propagate to branches
        for neighbor in branch_neighbors:
            print(
                f"[Node {self.rank}] Propagating INITIATE to branch neighbor {neighbor}"
            )
            self.send_message(
                neighbor,
                MessageType.INITIATE,
                level=level,
                fragment_id=fragment_id,
                state=state_value,
            )
            if self.state == NodeState.FIND:
                self.find_count += 1

        print(f"[Node {self.rank}] find_count={self.find_count}")

        # Small delay to ensure fragment membership propagates before testing
        if self.state == NodeState.FIND:
            print(f"[Node {self.rank}] State is FIND -> Starting test phase")
            time.sleep(0.002)  # 2ms delay for synchronization
            self.test()
        elif self.state == NodeState.FOUND:
            # If propagating FOUND state, report immediately if no children
            if self.find_count == 0:
                print(f"[Node {self.rank}] State is FOUND, no children -> Reporting")
                self.report()

    def test(self):
        """Test for minimum weight outgoing edge"""
        basic_edges = [
            (n, w)
            for n, w in self.neighbors.items()
            if self.edge_states[n] == EdgeState.BASIC
        ]

        print(f"\n[Node {self.rank}] ??? TEST phase")
        print(f"[Node {self.rank}] Available BASIC edges: {basic_edges}")

        if basic_edges:
            # Sort by weight, then by node ID for deterministic tie-breaking
            self.test_edge = min(basic_edges, key=lambda x: (x[1], x[0]))[0]
            print(
                f"[Node {self.rank}] Testing edge to Node {self.test_edge} (weight={self.neighbors[self.test_edge]})"
            )
            self.send_message(
                self.test_edge,
                MessageType.TEST,
                level=self.level,
                fragment_id=self.fragment_id,
            )
        else:
            print(f"[Node {self.rank}] No BASIC edges left -> Reporting to parent")
            self.test_edge = None
            self.report()

    def handle_test(self, sender, data):
        """Handle TEST message"""
        level = data["level"]
        fragment_id = data["fragment_id"]

        print(f"\n[Node {self.rank}] ??? HANDLE TEST from Node {sender}")
        print(f"[Node {self.rank}] Sender: level={level}, fragment={fragment_id}")
        print(
            f"[Node {self.rank}] My state: level={self.level}, fragment={self.fragment_id}"
        )

        if self.state == NodeState.SLEEPING:
            print(f"[Node {self.rank}] I was sleeping, waking up first...")
            self.wakeup()

        if level > self.level:
            # Defer until our level catches up
            msg_key = (MessageType.TEST.value, sender, level)
            defer_count = self.defer_count.get(msg_key, 0)
            if defer_count < 5:  # Allow more deferrals for larger graphs
                print(
                    f"[Node {self.rank}] Sender level > my level -> Deferring TEST (count={defer_count + 1})"
                )
                self.deferred_messages.append(
                    {"type": MessageType.TEST.value, "sender": sender, "data": data}
                )
                self.defer_count[msg_key] = defer_count + 1
            else:
                # Too many deferrals - accept to prevent deadlock
                print(
                    f"[Node {self.rank}] Too many deferrals -> Accepting to prevent deadlock"
                )
                self.send_message(sender, MessageType.ACCEPT)
            return
        elif fragment_id == self.fragment_id:
            # Same fragment - reject the edge
            print(
                f"[Node {self.rank}] Same fragment detected -> REJECTING edge ({self.rank}, {sender})"
            )
            if self.edge_states[sender] == EdgeState.BASIC:
                self.edge_states[sender] = EdgeState.REJECTED
            if sender != self.test_edge:
                self.send_message(sender, MessageType.REJECT)
            else:
                # We were testing this edge, need to test another
                print(
                    f"[Node {self.rank}] This was my test_edge -> Testing another edge"
                )
                self.test()
        else:
            # Different fragment - accept as potential MST edge
            print(
                f"[Node {self.rank}] Different fragment -> ACCEPTING edge as potential MST edge"
            )
            self.send_message(sender, MessageType.ACCEPT)

    def handle_accept(self, sender, data):
        """Handle ACCEPT message"""
        print(f"\n[Node {self.rank}] [OK] HANDLE ACCEPT from Node {sender}")
        print(
            f"[Node {self.rank}] Edge ({self.rank}, {sender}) weight={self.neighbors[sender]}"
        )

        self.test_edge = None
        if self.neighbors[sender] < self.best_weight:
            print(
                f"[Node {self.rank}] New best edge: {sender} (weight {self.neighbors[sender]} < {self.best_weight})"
            )
            self.best_edge = sender
            self.best_weight = self.neighbors[sender]
        elif self.neighbors[sender] == self.best_weight and self.best_edge is not None:
            # Tie-breaking: prefer lower node ID for determinism
            if sender < self.best_edge:
                print(
                    f"[Node {self.rank}] Tie-breaking: choosing {sender} over {self.best_edge}"
                )
                self.best_edge = sender
        else:
            print(
                f"[Node {self.rank}] Not better than current best (weight={self.best_weight})"
            )

        print(
            f"[Node {self.rank}] Current best: edge to {self.best_edge}, weight={self.best_weight}"
        )
        self.report()

    def handle_reject(self, sender, data):
        """Handle REJECT message"""
        print(f"\n[Node {self.rank}] [X] HANDLE REJECT from Node {sender}")
        print(f"[Node {self.rank}] Marking edge ({self.rank}, {sender}) as REJECTED")

        if self.edge_states[sender] == EdgeState.BASIC:
            self.edge_states[sender] = EdgeState.REJECTED

        print(f"[Node {self.rank}] Testing another edge...")
        self.test()

    def report(self):
        """Report best edge to parent"""
        if self.find_count < 0:
            self.find_count = 0

        print(
            f"\n[Node {self.rank}] ||| REPORT check: find_count={self.find_count}, test_edge={self.test_edge}"
        )

        if self.find_count == 0 and self.test_edge is None:
            old_state = self.state
            self.state = NodeState.FOUND
            print(f"[Node {self.rank}] State: {old_state.name} -> FOUND")
            print(
                f"[Node {self.rank}] Best edge: {self.best_edge}, weight={self.best_weight}"
            )

            if self.in_branch is not None:
                print(f"[Node {self.rank}] Reporting to parent (Node {self.in_branch})")
                self.send_message(
                    self.in_branch, MessageType.REPORT, weight=self.best_weight
                )
            else:
                # Root node - initiate merge if outgoing edge exists
                print(f"[Node {self.rank}] I am ROOT of fragment {self.fragment_id}")
                if self.best_weight < float("inf"):
                    print(
                        f"[Node {self.rank}] ROOT: Found outgoing edge, initiating CHANGEROOT"
                    )
                    self.changeroot()
                else:
                    # No outgoing edges found - algorithm complete for this fragment
                    print(
                        f"[Node {self.rank}] ROOT: No outgoing edges -> Fragment complete"
                    )
                    self.terminated = True
        else:
            print(
                f"[Node {self.rank}] Not ready to report yet (waiting for children or test result)"
            )

    def handle_report(self, sender, data):
        """Handle REPORT message"""
        weight = data["weight"]

        print(f"\n[Node {self.rank}] ||| HANDLE REPORT from Node {sender}")
        print(
            f"[Node {self.rank}] Reported weight: {weight}, my best: {self.best_weight}"
        )

        if sender != self.in_branch:
            # Report from child
            print(f"[Node {self.rank}] Report from child (not in_branch)")
            if self.find_count > 0:
                self.find_count -= 1
                print(f"[Node {self.rank}] Decremented find_count to {self.find_count}")

            if weight < self.best_weight:
                print(
                    f"[Node {self.rank}] Child's edge is better: {weight} < {self.best_weight}"
                )
                self.best_weight = weight
                self.best_edge = sender
            elif weight == self.best_weight and weight < float("inf"):
                # Tie-breaking: prefer lower node ID
                if self.best_edge is None or sender < self.best_edge:
                    print(f"[Node {self.rank}] Tie-breaking: choosing {sender}")
                    self.best_edge = sender

            print(
                f"[Node {self.rank}] Current best: edge to {self.best_edge}, weight={self.best_weight}"
            )
            self.report()
        elif self.state == NodeState.FOUND:
            # Already in FOUND, check if we should change root
            print(f"[Node {self.rank}] Report from in_branch while in FOUND state")
            if weight > self.best_weight:
                print(
                    f"[Node {self.rank}] Parent's weight > mine -> Initiating CHANGEROOT"
                )
                self.changeroot()
            elif weight == self.best_weight and weight < float("inf"):
                # Both have same weight, use node ID to break tie
                if self.rank < sender:
                    print(
                        f"[Node {self.rank}] Tie + lower rank -> Initiating CHANGEROOT"
                    )
                    self.changeroot()

    def changeroot(self):
        """Change root of fragment"""
        print(f"\n[Node {self.rank}] <-> CHANGEROOT initiated")

        if self.best_edge is None or self.best_edge not in self.edge_states:
            print(f"[Node {self.rank}] No valid best_edge, returning")
            return

        edge_state = self.edge_states[self.best_edge]
        print(
            f"[Node {self.rank}] Best edge: {self.best_edge} (weight={self.best_weight}), state={edge_state.name}"
        )

        # Prevent infinite loops - don't call changeroot on already processed edges
        if (
            hasattr(self, "_last_changeroot_edge")
            and self._last_changeroot_edge == self.best_edge
        ):
            if hasattr(self, "_changeroot_count"):
                self._changeroot_count += 1
                print(
                    f"[Node {self.rank}] Repeated changeroot on same edge (count={self._changeroot_count})"
                )
                if self._changeroot_count > 5:
                    # Tried too many times, likely done
                    print(f"[Node {self.rank}] Too many retries -> Terminating")
                    self.terminated = True
                    return
            else:
                self._changeroot_count = 1
        else:
            self._last_changeroot_edge = self.best_edge
            self._changeroot_count = 1

        if edge_state == EdgeState.BRANCH:
            print(
                f"[Node {self.rank}] Best edge is BRANCH -> Forwarding CHANGEROOT to {self.best_edge}"
            )
            self.send_message(self.best_edge, MessageType.CHANGEROOT)
        elif edge_state == EdgeState.BASIC:
            print(f"[Node {self.rank}] Best edge is BASIC -> Converting to BRANCH")
            print(
                f"[Node {self.rank}] +++ MST EDGE ADDED: ({self.rank}, {self.best_edge}) weight={self.best_weight}"
            )
            self.edge_states[self.best_edge] = EdgeState.BRANCH
            self.send_message(
                self.best_edge,
                MessageType.CONNECT,
                level=self.level,
                fragment_id=self.fragment_id,
            )
            # After sending CONNECT, mark as processed and wait for new INITIATE
            self._last_changeroot_edge = self.best_edge
            self.best_edge = None
            self.best_weight = float("inf")
            self.state = NodeState.FIND
            print(f"[Node {self.rank}] State changed to FIND, waiting for new INITIATE")
        elif edge_state == EdgeState.REJECTED:
            # Edge was rejected, reset and continue
            print(
                f"[Node {self.rank}] Best edge was REJECTED -> Resetting and testing again"
            )
            self.best_edge = None
            self.best_weight = float("inf")
            self.test()

    def handle_changeroot(self, sender, data):
        """Handle CHANGEROOT message"""
        print(f"\n[Node {self.rank}] <-> HANDLE CHANGEROOT from Node {sender}")
        print(f"[Node {self.rank}] My best_edge: {self.best_edge}")

        # Only propagate if we have a valid best_edge that's not already processed
        if (
            self.best_edge is not None
            and self.edge_states.get(self.best_edge) == EdgeState.BASIC
        ):
            print(f"[Node {self.rank}] Best edge is BASIC -> Calling changeroot")
            self.changeroot()
        elif (
            self.best_edge is not None
            and self.edge_states.get(self.best_edge) == EdgeState.BRANCH
        ):
            # Already processed, forward the changeroot
            print(
                f"[Node {self.rank}] Best edge is BRANCH -> Forwarding CHANGEROOT to {self.best_edge}"
            )
            self.send_message(self.best_edge, MessageType.CHANGEROOT)
        else:
            print(f"[Node {self.rank}] No valid best_edge to propagate CHANGEROOT")

    def run(self, max_iterations=5000):
        """Main execution loop"""
        # Stagger wakeup based on rank to reduce simultaneous CONNECT messages
        time.sleep(0.001 * self.rank)

        # Wake up
        self.wakeup()

        iteration = 0
        messages_processed = 0

        while iteration < max_iterations and not self.terminated:
            # Try to receive and process messages
            msg = self.receive_message(timeout=0.001)
            if msg:
                self.process_message(msg)
                self.idle_iterations = 0
                messages_processed += 1
            else:
                self.idle_iterations += 1

                # Process deferred messages in deterministic order
                if self.deferred_messages and self.idle_iterations % 10 == 0:
                    # Sort deferred messages by sender for deterministic ordering
                    self.deferred_messages.sort(key=lambda m: m.get("sender", 0))
                    deferred_msg = self.deferred_messages.pop(0)
                    self.process_message(deferred_msg)
                    messages_processed += 1
                else:
                    time.sleep(0.001)

            # Check for stuck state - root should try to changeroot more aggressively
            stuck_threshold = max(30, self.size * 3)
            if (
                self.state == NodeState.FOUND
                and self.in_branch is None
                and self.best_weight < float("inf")
                and self.best_edge is not None
                and self.idle_iterations > stuck_threshold
                and self.idle_iterations % 30 == 0  # Retry every 30 idle iterations
            ):
                self.changeroot()
                self.idle_iterations = (
                    stuck_threshold  # Reset to threshold to continue trying
                )

            # Stop if completely idle (likely done) - scale with graph size
            max_idle_threshold = max(1000, self.size * 100)
            if self.idle_iterations > max_idle_threshold:
                break

            # Don't use per-node branch count for early termination
            # as it can cause premature stopping in disconnected fragments

            iteration += 1

        if self.rank == 0:
            print(
                f"[Node {self.rank}] Finished after {iteration} iterations, processed {messages_processed} messages"
            )

    def get_mst_edges(self):
        """Get MST edges from this node"""
        mst_edges = []
        for neighbor, state in self.edge_states.items():
            if state == EdgeState.BRANCH and self.rank < neighbor:
                weight = self.neighbors[neighbor]
                mst_edges.append((self.rank, neighbor, weight))
        return mst_edges


def collect_results(comm, rank, size, node):
    """Collect MST results from all nodes"""
    # Each node sends its BRANCH edges
    local_edges = node.get_mst_edges()

    # Gather all edges to rank 0
    all_edges = comm.gather(local_edges, root=0)

    if rank == 0:
        # Combine all edges
        mst_edges = []
        for edges in all_edges:
            mst_edges.extend(edges)

        # Remove duplicates
        mst_edges = list(set(mst_edges))

        return mst_edges

    return None


def print_results(mst_edges, graph_dir="graph_data"):
    """Print MST results"""
    print("\n" + "=" * 70)
    print("GHS Algorithm Results (MPI Implementation)")
    print("=" * 70)

    # Load metadata for verification
    metadata_file = os.path.join(graph_dir, "graph_metadata.json")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    num_nodes = metadata["num_nodes"]

    print(f"\nMST Edges found by GHS:")
    total_weight = 0
    for u, v, w in sorted(mst_edges):
        print(f"  ({u}, {v}): weight = {w}")
        total_weight += w

    print(f"\nTotal MST weight: {total_weight}")
    print(f"Number of edges: {len(mst_edges)}")
    print(f"Expected edges: {num_nodes - 1}")

    if len(mst_edges) == num_nodes - 1:
        print("Correct number of edges!")
    else:
        print("WARNING: Incorrect number of edges!")

    # Save results
    results = {
        "mst_edges": mst_edges,
        "total_weight": total_weight,
        "num_edges": len(mst_edges),
        "algorithm": "GHS (MPI)",
    }

    output_file = os.path.join(graph_dir, "mst_result_mpi.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    # Visualize the MST
    try:
        import networkx as nx
        import matplotlib.pyplot as plt

        # Load original graph
        G = nx.Graph()
        for u, v, w in metadata["edges"]:
            G.add_edge(u, v, weight=w)

        # Create MST graph
        MST = nx.Graph()
        for u, v, w in mst_edges:
            MST.add_edge(u, v, weight=w)

        # Draw graph
        plt.figure(figsize=(12, 5))

        # Original graph
        plt.subplot(1, 2, 1)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            node_size=500,
            font_size=12,
            font_weight="bold",
        )
        edge_labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edge_labels(G, pos, edge_labels)
        plt.title("Original Graph")

        # MST
        plt.subplot(1, 2, 2)
        nx.draw(
            MST,
            pos,
            with_labels=True,
            node_color="lightgreen",
            node_size=500,
            font_size=12,
            font_weight="bold",
        )
        mst_labels = nx.get_edge_attributes(MST, "weight")
        nx.draw_networkx_edge_labels(MST, pos, mst_labels)
        plt.title(f"MST (weight={total_weight})")

        plt.tight_layout()
        output_image = os.path.join(graph_dir, "mst_result_mpi.png")
        plt.savefig(output_image, dpi=150, bbox_inches="tight")
        print(f"Visualization saved to: {output_image}")
        plt.close()
    except ImportError:
        pass

    print("=" * 70)


def main():
    """Main MPI execution"""
    import argparse

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Only rank 0 parses arguments
    if rank == 0:
        parser = argparse.ArgumentParser(description="Run MPI-based GHS algorithm")
        parser.add_argument(
            "--graph-dir",
            type=str,
            default="graph_data",
            help="Graph data directory (default: graph_data)",
        )
        args = parser.parse_args()
        graph_dir = args.graph_dir
    else:
        graph_dir = None

    # Broadcast graph_dir to all processes
    graph_dir = comm.bcast(graph_dir, root=0)

    if rank == 0:
        print("=" * 70)
        print("MPI-based GHS Algorithm")
        print("=" * 70)
        print(f"Number of MPI processes: {size}")
        print(f"Graph directory: {graph_dir}")
        print("=" * 70)

    # Create node
    node = MPINode(comm, rank, size, graph_dir)

    start_time = time.time()

    # Run GHS algorithm (scale with graph size)
    max_iters = max(5000, size * 1000)
    node.run(max_iterations=max_iters)

    # Barrier to ensure all nodes finish
    comm.Barrier()

    if rank == 0:
        elapsed = time.time() - start_time
        print(f"\nAlgorithm completed in {elapsed:.2f} seconds")

    # Global termination check: collect idle status from all nodes
    local_status = {
        "idle": node.idle_iterations,
        "state": node.state.value,
        "best_weight": node.best_weight,
    }
    all_status = comm.gather(local_status, root=0)

    if rank == 0 and all_status:
        # Check if all nodes are idle with no outgoing edges
        all_idle = all([s["idle"] > 100 for s in all_status])
        avg_idle = sum(s["idle"] for s in all_status) / len(all_status)
        print(f"Global status: avg_idle={avg_idle:.0f}, all_nodes_idle={all_idle}")

    # Collect results
    mst_edges = collect_results(comm, rank, size, node)

    # Print results (only rank 0)
    if rank == 0 and mst_edges is not None:
        print_results(mst_edges, graph_dir)


if __name__ == "__main__":
    main()
