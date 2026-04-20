# ASI-Living-Tree

> 一种基于Hopfield动力学的知识图谱检索架构，专为代码领域的RAG设计

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 核心特性

- **节点隔离**：新知识不影响旧节点嵌入，真正的持续学习
- **O(1) 更新**：添加新知识无需重训练
- **能量驱动检索**：确定性检索结果，相同查询产生相同结果
- **聚类级歧义解析**：258x 区分度，精准处理代码版本冲突

## 安装

```bash
pip install git+https://github.com/simulai/ASI-Living-Tree.git
```

或克隆后本地安装：

```bash
git clone https://github.com/simulai/ASI-Living-Tree.git
cd ASI-Living-Tree
pip install -e .
```

## 快速开始

```python
import numpy as np
from living_tree import LivingTreeMemory, HopfieldRetrieval, ClusterLevelEnergyGapRetrieval

# 创建知识记忆
memory = LivingTreeMemory()

# 添加节点（嵌入向量 + 解决方案 + 元数据）
for i in range(100):
    emb = np.random.randn(768).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    memory.add_node(emb, f"solution_{i}", {'version': f'v{i % 3}'})

# 创建检索器
retrieval = ClusterLevelEnergyGapRetrieval(beta=15.0, gap_threshold=0.03)
retrieval.store(memory.nodes)

# 查询
query = np.random.randn(768).astype(np.float32)
query = query / (np.linalg.norm(query) + 1e-8)

nodes, sims, info = retrieval.retrieve(query, k=5, return_ambiguity=True)

print(f"Energy Gap: {info['energy_gap']:.3f}")
print(f"Is ambiguous: {info['is_ambiguous']}")
print(f"Winning version: {info['winning_version']}")
```

## 验证结果

| 实验 | 结果 | 关键发现 |
|------|------|----------|
| H46 集成验证 | 98.5% | Pareto最优 |
| H51 遗忘率 | 0% | 节点隔离特性 |
| H52 模块干扰 | 0% | 互补门控有效 |
| H59 熵歧义解析 | PASS | 熵区分有效 |
| H60 能量间隔 | 17x | 聚类级区分度 |
| H61 Killer Demo | 258x | 版本路由100% |
| H62 requests库 | 90.9% | 真实API冲突 |

## 架构

```
living_tree/
├── __init__.py          # 包入口
├── core.py              # 核心类：LivingTreeNode, LivingTreeMemory, HopfieldRetrieval
├── entropy_retrieval.py # 熵驱动的歧义解析（H59）
└── energy_gap_retrieval.py  # 能量间隔歧义解析（H60-H62）
```

## 核心公式

**Hopfield能量函数**：
```
E(x, memory) = -β · exp(sim(x, memory))
```

**聚类级能量间隔**：
```
Cluster_Energy[cluster] = max_sim(query, cluster_nodes)
Energy_Gap = E_top_cluster - E_second_cluster
is_ambiguous = Energy_Gap < threshold
```

## Demos

- `demos/demo_code_version_conflict.ipynb` - 代码版本冲突Killer Demo
- `demos/demo_humaneval_rag.ipynb` - HumanEval RAG演示

## Benchmark

- `benchmarks/code_version_conflict.json` - 代码版本冲突测试集

## 论文

详细验证结果见 [paper/Living_Tree_Paper_Draft_v1.md](./paper/Living_Tree_Paper_Draft_v1.md)

## 引用

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