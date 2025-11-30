# GHS Algorithm Implementation - README

## Overview
This is a working implementation of the **Gallager-Humblet-Spira (GHS) algorithm** for finding the Minimum Spanning Tree (MST) in a distributed setting using Python threads to simulate distributed nodes.

## Files Created
- **ghs_implementation.py** - Main implementation file
- **ghs_mst.json** - MST results in JSON format
- **ghs_mst.png** - Visualization comparing the original graph and the MST

## Algorithm Details

### GHS Algorithm
The GHS algorithm is a distributed algorithm that finds an MST in a network where:
- Each node only knows about its immediate neighbors
- Nodes communicate via message passing
- The algorithm proceeds in phases, merging fragments

### Implementation Features
1. **Thread-based Simulation**: Each node runs as a separate thread
2. **Message Passing**: Nodes communicate via queues
3. **Seven Message Types**:
   - CONNECT: Initiates fragment merging
   - INITIATE: Propagates fragment information
   - TEST: Tests if an edge is outgoing
   - ACCEPT: Accepts an edge as outgoing
   - REJECT: Rejects an edge (same fragment)
   - REPORT: Reports best outgoing edge
   - CHANGEROOT: Changes fragment root

### Node States
- **SLEEPING**: Initial state
- **FIND**: Searching for minimum outgoing edge
- **FOUND**: Found the minimum edge for current level

### Edge States
- **BASIC**: Unknown state
- **BRANCH**: Part of MST
- **REJECTED**: Not part of MST

## Results

### Sample Graph (6 nodes, 9 edges)
```
Edges:
(0,1): weight=1, (0,2): weight=4, (1,2): weight=2
(1,3): weight=5, (2,3): weight=3, (2,4): weight=7
(3,4): weight=6, (3,5): weight=8, (4,5): weight=9
```

### MST Found by GHS Algorithm
```
MST Edges:
- (0,1): weight=1
- (1,2): weight=2
- (2,3): weight=3
- (3,4): weight=6
- (3,5): weight=8

Total MST Weight: 20
```

### Verification
✓ **Algorithm produces CORRECT MST** - Verified against NetworkX's MST implementation

## How to Run

```bash
python ghs_implementation.py
```

## Output
1. **Console**: Shows algorithm progress and MST edges
2. **ghs_mst.json**: Complete MST data in JSON format
3. **ghs_mst.png**: Side-by-side visualization of original graph and MST

## Algorithm Complexity
- **Time Complexity**: O(n log n + m) where n=nodes, m=edges
- **Message Complexity**: O(n log n + m)
- **Optimal**: Yes, for distributed MST algorithms

## Key Implementation Details

1. **Thread Safety**: Uses locks (RLock) for thread-safe operations
2. **Message Queue**: Each node has its own queue for incoming messages
3. **Daemon Threads**: Threads are daemonized for clean shutdown
4. **Timeout Handling**: Algorithm has a configurable timeout
5. **Graph Generation**: Uses NetworkX for graph creation and verification

## Distributed Simulation
Each node operates independently and:
- Only knows about its immediate neighbors
- Communicates only via messages
- Makes local decisions based on received messages
- No global coordinator

This truly simulates a distributed environment where nodes work asynchronously.

## Extensions Possible
1. Add network delays to simulate real network latency
2. Add message failures/retries
3. Visualize message passing in real-time
4. Support dynamic graph changes
5. Add more metrics (message count, execution time per node)
