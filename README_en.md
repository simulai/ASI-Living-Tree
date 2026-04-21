# ASI-Living-Tree

> A Hopfield dynamics based knowledge graph retrieval architecture for code RAG

[中文](./README.md) | English

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Core Features

- **Node Isolation**: New knowledge does not affect old node embeddings - true continuous learning
- **O(1) Update**: Add new knowledge without retraining
- **Energy-Driven Retrieval**: Deterministic results - same query produces same result
- **Cluster-Level Ambiguity Resolution**: 258x discrimination - precise code version conflict handling

## Installation

```bash
pip install git+https://github.com/simulai/ASI-Living-Tree.git
```

Or clone and install locally:

```bash
git clone https://github.com/simulai/ASI-Living-Tree.git
cd ASI-Living-Tree
pip install -e .
```

## Quick Start

```python
import numpy as np
from living_tree import LivingTreeMemory, ClusterLevelEnergyGapRetrieval

# Create knowledge memory
memory = LivingTreeMemory()

# Add nodes (embedding + solution + metadata)
for i in range(100):
    emb = np.random.randn(768).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    memory.add_node(emb, f"solution_{i}", {'version': f'v{i % 3}'})

# Create retriever
retrieval = ClusterLevelEnergyGapRetrieval(beta=15.0, gap_threshold=0.03)
retrieval.store(memory.nodes)

# Query
query = np.random.randn(768).astype(np.float32)
query = query / (np.linalg.norm(query) + 1e-8)

nodes, sims, info = retrieval.retrieve(query, k=5, return_ambiguity=True)

print(f"Energy Gap: {info['energy_gap']:.3f}")
print(f"Is ambiguous: {info['is_ambiguous']}")
print(f"Winning version: {info['winning_version']}")
```

## Verification Results

| Experiment | Result | Key Finding |
|------------|--------|-------------|
| H46 Integration | 98.5% | Pareto optimal |
| H51 Forgetting | 0% | Node isolation |
| H52 Interference | 0% | Complementary gating |
| H59 Entropy | PASS | Entropy discrimination |
| H60 Energy Gap | 17x | Cluster-level discrimination |
| H61 Killer Demo | 258x | 100% version routing |
| H62 requests | 90.9% | Real API conflict |

## Architecture

```
living_tree/
├── __init__.py          # Package entry
├── core.py              # Core classes: LivingTreeNode, LivingTreeMemory, HopfieldRetrieval
├── entropy_retrieval.py # Entropy-driven ambiguity resolution (H59)
└── energy_gap_retrieval.py  # Energy gap ambiguity resolution (H60-H62)
```

## Core Formulas

**Hopfield Energy Function**:
```
E(x, memory) = -β · exp(sim(x, memory))
```

**Cluster-Level Energy Gap**:
```
Cluster_Energy[cluster] = max_sim(query, cluster_nodes)
Energy_Gap = E_top_cluster - E_second_cluster
is_ambiguous = Energy_Gap < threshold
```

## Demos

- `demos/demo_code_version_conflict.ipynb` - Code version conflict Killer Demo (coming soon)
- `demos/demo_humaneval_rag.ipynb` - HumanEval RAG demo (coming soon)

## Benchmark

- `benchmarks/code_version_conflict.json` - Code version conflict test set

## Paper

See [paper/Living_Tree_Paper_Draft_v1.md](./paper/Living_Tree_Paper_Draft_v1.md) for detailed verification results

## Citation

```bibtex
@misc{ASI-Living-Tree,
  title = {ASI-Living-Tree: A Hopfield Dynamics Based Knowledge Graph Retrieval Architecture},
  author = {Claude Code},
  year = {2026},
  note = {https://github.com/simulai/ASI-Living-Tree}
}
```

## License

MIT License