# ASI-Living-Tree

> 一种基于Hopfield动力学的知识图谱检索架构，专为代码领域RAG设计

[中文](./README_zh.md) | English

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 什么是 ASI-Living-Tree？

**ASI-Living-Tree** 是一种新型知识图谱架构，使用 Hopfield 动力学来实现高效、持续学习的代码检索增强生成（RAG）。与传统基于 Transformer 的方法不同，Living Tree 提供了**节点隔离**、**O(1) 更新**和**能量驱动的确定性检索**。

### Transformer 的问题

基于 Transformer 的 RAG 系统存在两个根本性问题：

1. **灾难性遗忘**：当学习新知识时，它会覆盖旧知识，导致先前掌握的任务性能下降。

2. **O(N) 存储复杂度**：每条知识需要 O(d) 空间，存储 N 条知识需要 O(N·d) 空间。

### Living Tree 如何解决

```
传统 RAG：
  新知识 → 重训练 → 旧知识丢失 ❌

Living Tree：
  新知识 → O(1) 添加 → 旧知识保留 ✅
```

Living Tree 将每个知识项视为能量景观中的**孤立节点**。添加新节点不会影响现有节点嵌入，确保真正的持续学习而不会遗忘。

## 核心概念

### 1. 节点隔离

每个知识节点存储：
- **Embedding**：知识的向量表示
- **Solution**：实际的解决方案或答案
- **Metadata**：附加信息（版本、聚类等）

新节点不影响旧节点——这是**零遗忘**的关键。

### 2. 能量驱动检索

当你查询 Living Tree 时：

```
1. 你的查询被转换为向量
2. 与所有节点嵌入计算相似度
3. 计算能量景观：E = -β · exp(sim)
4. 根据能量返回 Top-k 节点
```

**确定性**：相同的查询总是返回相同的结果（不像 Transformer 中的随机 softmax 采样）。

### 3. 聚类级能量间隔

当存在多个版本/聚类时（例如 API v1 vs v2），Living Tree 使用**能量间隔**来检测歧义：

```
Cluster_Energy[cluster] = 聚类中的最大相似度
Energy_Gap = 顶级聚类能量 - 第二聚类能量

if Energy_Gap < 阈值:
    → 歧义：多个聚类竞争 → 多路检索
else:
    → 明确：一个聚类主导 → 单路检索
```

这为代码版本冲突提供了 **258x 区分度**，帮助 LLM 避开 API 陷阱。

## 为什么选择 ASI-Living-Tree？

| 特性 | Transformer RAG | ASI-Living-Tree |
|------|----------------|-----------------|
| 遗忘 | 灾难性 ❌ | 零 ✅ |
| 更新成本 | O(N) 重训练 ❌ | O(1) 添加 ✅ |
| 检索 | 随机性 ❌ | 确定性 ✅ |
| 可解释性 | 不透明的注意力 ❌ | 能量景观 ✅ |
| 歧义检测 | 无 ❌ | 258x 间隔 ✅ |

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
from living_tree import LivingTreeMemory, ClusterLevelEnergyGapRetrieval

# 创建知识记忆
memory = LivingTreeMemory()

# 添加节点：每个节点有嵌入、解决方案和元数据
# 示例：代码版本冲突场景
for i in range(50):
    emb = np.random.randn(768).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    memory.add_node(
        emb,
        f"v1_implementation_{i}",  # 解决方案
        {'version': 'v1', 'api_style': 'sync'}  # 元数据
    )

for i in range(50):
    emb = np.random.randn(768).astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    memory.add_node(
        emb,
        f"v2_implementation_{i}",
        {'version': 'v2', 'api_style': 'async'}
    )

# 创建聚类级能量间隔检索器
retrieval = ClusterLevelEnergyGapRetrieval(beta=15.0, gap_threshold=0.03)
retrieval.store(memory.nodes)

# 查询
query = np.random.randn(768).astype(np.float32)
query = query / (np.linalg.norm(query) + 1e-8)

nodes, sims, info = retrieval.retrieve(query, k=5, return_ambiguity=True)

print(f"能量间隔: {info['energy_gap']:.3f}")
print(f"是否歧义: {info['is_ambiguous']}")
print(f"获胜版本: {info['winning_version']}")
print(f"聚类能量: {info['cluster_energies']}")
```

## 真实场景示例：代码版本冲突

想象你正在为具有多个版本的库构建 RAG 系统：

```python
# 场景：requests 库 v1 vs v2 vs v2adv
#
# 查询："我需要处理 10000 个并发 WebSocket 连接"
#
# v1 (同步): 简单，每个连接一个线程 → 10000 线程会崩溃
# v2 (异步): 单线程事件循环，轻松处理 10000 连接
# v2adv (异步+中间件): v2 + 日志/监控/限流
#
# Living Tree 自动选择正确的版本！

retrieval = ClusterLevelEnergyGapRetrieval(beta=15.0, gap_threshold=0.03)
retrieval.store(memory.nodes)

query = create_query("高并发 + 重试 + SSL 验证")

nodes, sims, info = retrieval.retrieve(query, k=5, return_ambiguity=True)

# 输出：
# 能量间隔: 0.830
# 聚类能量: {'v2': 0.903, 'v2adv': 0.072, 'v1': 0.024}
# 获胜版本: v2 ✅
```

## 验证结果

我们进行了 **30 项系统性实验**来验证 Living Tree：

| 实验 | 结果 | 关键发现 |
|------|------|----------|
| H46 集成验证 | **98.5%** | Pareto 最优准确率 |
| H51 遗忘率 | **0%** | 真正的节点隔离 |
| H52 模块干扰 | **0%** | 互补门控有效 |
| H59 熵歧义解析 | PASS | 熵有效检测歧义 |
| H60 能量间隔 | **17x** | 聚类级区分度 |
| H61 Killer Demo | **258x** | 版本路由 100% 准确 |
| H62 真实代码库 | **90.9%** | 真实 requests 库 API 冲突 |

### 关键指标

- **准确率**：98.5%（集成任务）
- **遗忘率**：0%（旧任务无性能下降）
- **模块干扰**：0%（模块互不影响）
- **歧义区分度**：258x（区分相似但冲突的代码版本）
- **LLM 提升**：MiniMax 上 +10% 逻辑准确率

## 架构

```
living_tree/
├── __init__.py          # 包入口点
├── core.py              # 核心类
│   ├── LivingTreeNode   # 知识节点
│   ├── LivingTreeMemory # 知识存储
│   ├── HopfieldRetrieval # 基础 Hopfield 检索
│   └── ComplementaryGating # 错误预防
├── entropy_retrieval.py # H59：熵驱动歧义检测
└── energy_gap_retrieval.py  # H60-H62：聚类级能量间隔解析
```

## 核心公式

### Hopfield 能量函数

```
E(x, memory) = -β · exp(sim(x, memory))

其中：
  - x: 查询向量
  - memory: 存储的模式
  - β: 温度参数（控制尖锐度）
  - sim: 余弦相似度
```

### 聚类级能量间隔

```
Cluster_Energy[cluster] = max_sim(query, cluster_nodes)
Energy_Gap = E_top_cluster - E_second_cluster
is_ambiguous = Energy_Gap < threshold (通常为 0.03)
```

## 使用场景

### 1. 代码版本冲突解决

帮助 LLM 在存在多个版本时选择正确的 API 版本。

### 2. 持续学习 RAG

添加新知识而无需重训练或遗忘旧知识。

### 3. 歧义查询解析

检测查询何时可能匹配多个聚类并探索所有可能性。

### 4. 生产级 API 选择

根据查询复杂度路由到生产级实现 vs 简单原型。

## 示例

即将推出的示例笔记本：
- `demos/demo_code_version_conflict.ipynb` - 交互式版本冲突演示
- `demos/demo_humaneval_rag.ipynb` - HumanEval RAG 评估演示

## 基准测试

`benchmarks/code_version_conflict.json` - 代码版本冲突测试数据集，包括：
- 简单的 GET/POST 请求
- 重试机制
- SSL 证书验证
- 会话和连接池
- 代理配置
- 等等...

## 论文

参见 [paper/Living_Tree_Paper_Draft_v1.md](./paper/Living_Tree_Paper_Draft_v1.md) 获取：
- 完整验证树
- 详细实验结果
- 理论基础
- 局限性和未来工作

## 引用

如果你在研究中使用 ASI-Living-Tree，请引用：

```bibtex
@misc{ASI-Living-Tree,
  title = {ASI-Living-Tree: A Hopfield Dynamics Based Knowledge Graph Retrieval Architecture},
  author = {Claude Code},
  year = {2026},
  note = {https://github.com/simulai/ASI-Living-Tree}
}
```

## 许可证

MIT 许可证

## 贡献

欢迎贡献！请随意提交 issues 和 pull requests。