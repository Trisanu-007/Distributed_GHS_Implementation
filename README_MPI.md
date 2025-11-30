# MPI-based GHS Algorithm Implementation

## Overview
This implementation of the GHS (Gallager-Humblet-Spira) algorithm uses MPI (Message Passing Interface) for distributed execution, suitable for running on SLURM clusters.

## Files

### 1. `create_graph_files.py`
Creates the input graph and generates individual node files.
- Generates a random connected graph using NetworkX
- Creates one JSON file per node with neighbor information
- Creates `graph_data/` directory with:
  - `node_<id>.json` - Individual node data files
  - `graph_metadata.json` - Overall graph information
  - `input_graph.png` - Visualization of the input graph

### 2. `ghs_implementation_mpi.py`
MPI-based implementation of the GHS algorithm.
- Each MPI process represents one node in the graph
- Nodes read their data from files (simulates distributed storage)
- Uses message passing for inter-node communication
- Produces MST using distributed GHS algorithm

### 3. `run_ghs.slurm`
SLURM batch script for running on a cluster.
- Configurable for different cluster sizes
- Handles job scheduling and resource allocation

## Installation

### Local Setup (Windows/Linux/Mac)

1. **Install Python packages:**
```bash
pip install mpi4py networkx matplotlib
```

2. **Install MPI (if not already installed):**
   - **Windows**: Download and install Microsoft MPI or MPICH
   - **Linux**: `sudo apt-get install mpich` or `sudo yum install mpich`
   - **Mac**: `brew install mpich`

### SLURM Cluster Setup

On your cluster, ensure the following modules are available:
```bash
module load python/3.9    # or your Python version
module load openmpi/4.1.0  # or your MPI implementation
```

Install Python packages in your user environment:
```bash
pip install --user mpi4py networkx matplotlib
```

## Usage

### Step 1: Create Graph Files

```bash
python create_graph_files.py
```

This creates:
- `graph_data/` directory with node files
- `input_graph.png` visualization
- Expected MST weight for verification

**Configuration** (edit in `create_graph_files.py`):
```python
num_nodes = 6            # Number of nodes
edge_probability = 0.5   # Edge density
seed = 42                # Random seed
```

### Step 2: Run GHS Algorithm

#### Local Execution:
```bash
mpiexec -n 6 python ghs_implementation_mpi.py
```
(Replace `6` with your number of nodes)

#### SLURM Cluster Execution:
```bash
sbatch --nodes=6 --ntasks=6 run_ghs.slurm
```

Or edit `run_ghs.slurm` to set default values and run:
```bash
sbatch run_ghs.slurm
```

### Step 3: Check Results

Results are saved in `graph_data/mst_result_mpi.json`:
```json
{
  "mst_edges": [[0, 3, 1], [0, 2, 2], ...],
  "total_weight": 11,
  "num_edges": 5,
  "algorithm": "GHS (MPI)"
}
```

## File Structure

```
DS_project/
├── create_graph_files.py          # Graph generator
├── ghs_implementation_mpi.py      # MPI-based GHS algorithm
├── run_ghs.slurm                  # SLURM batch script
├── graph_data/                    # Generated graph files
│   ├── node_0.json                # Node 0 data
│   ├── node_1.json                # Node 1 data
│   ├── ...
│   ├── graph_metadata.json        # Graph metadata
│   ├── input_graph.png            # Input visualization
│   └── mst_result_mpi.json        # Algorithm results
└── README_MPI.md                  # This file
```

## Node File Format

Each `node_<id>.json` file contains:
```json
{
  "node_id": 0,
  "neighbors": {
    "1": 5,    // neighbor_id: edge_weight
    "2": 3
  },
  "num_neighbors": 2
}
```

## Algorithm Overview

The MPI implementation follows the GHS algorithm:

1. **Initialization**: Each node wakes up and connects to its minimum weight edge
2. **Fragment Formation**: Nodes merge into fragments at increasing levels
3. **Edge Testing**: Fragments test edges to find minimum outgoing edges
4. **Fragment Merging**: Fragments merge using CONNECT messages
5. **Termination**: Algorithm terminates when no outgoing edges remain

### Message Types:
- `CONNECT`: Initiate fragment merging
- `INITIATE`: Propagate fragment information
- `TEST`: Test if edge is outgoing
- `ACCEPT`: Accept edge as outgoing
- `REJECT`: Reject edge (internal to fragment)
- `REPORT`: Report best outgoing edge
- `CHANGEROOT`: Change fragment root

## SLURM Configuration

Edit `run_ghs.slurm` for your cluster:

```bash
#SBATCH --nodes=6              # Number of nodes (= graph nodes)
#SBATCH --ntasks=6             # Number of MPI tasks
#SBATCH --time=00:10:00        # Time limit
#SBATCH --mem-per-cpu=1G       # Memory per CPU
```

**Important**: `--nodes` and `--ntasks` should match the number of nodes in your graph!

## Troubleshooting

### mpi4py not found
```bash
pip install mpi4py
# or on cluster:
pip install --user mpi4py
```

### MPI not installed
Install MPI implementation (MPICH, OpenMPI, or Microsoft MPI)

### Wrong number of processes
Ensure `mpiexec -n N` matches the number of nodes in the graph

### File not found errors
Run `create_graph_files.py` first to generate node files

### SLURM job fails
Check:
- Module availability: `module avail`
- Job status: `squeue -u $USER`
- Error logs: `cat ghs_mpi_<jobid>.err`

## Comparison with Thread-based Implementation

| Feature | Thread-based | MPI-based |
|---------|-------------|-----------|
| Execution | Single machine | Distributed cluster |
| Scalability | Limited by cores | Scales to many nodes |
| Communication | Shared memory | Message passing |
| File I/O | Optional | Required (simulates distributed storage) |
| Best for | Local testing | Production clusters |

## Performance Notes

- Algorithm complexity: O(N log N + E) where N=nodes, E=edges
- Message complexity: O(N log N + E)
- Typical execution time: 0.2-1.0 seconds for 6-7 nodes
- Scales well to larger graphs on distributed systems

## Example Output

```
======================================================================
MPI-based GHS Algorithm
======================================================================
Number of MPI processes: 6
Graph directory: graph_data
======================================================================
[Node 0] Loaded info: 3 neighbors
[Node 0] Woke up, connecting to node 3

Algorithm completed in 0.45 seconds

======================================================================
GHS Algorithm Results (MPI Implementation)
======================================================================

MST Edges found by GHS:
  (0, 2): weight = 2
  (0, 3): weight = 1
  (1, 4): weight = 4
  (2, 4): weight = 2
  (3, 5): weight = 2

Total MST weight: 11
Number of edges: 5
Expected edges: 5
✓ Correct number of edges!

Results saved to: graph_data/mst_result_mpi.json
======================================================================
```

## References

- Gallager, R. G., Humblet, P. A., & Spira, P. M. (1983). A distributed algorithm for minimum-weight spanning trees. ACM Transactions on Programming Languages and Systems, 5(1), 66-77.
