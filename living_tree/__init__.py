"""
ASI-Living-Tree: 基于Hopfield动力学的知识图谱检索架构
=====================================================

一个用于代码RAG的持续学习知识图谱系统。

核心特性：
- 节点隔离：新知识不影响旧节点
- O(1) 更新：无需重训练
- 能量驱动检索：确定性结果
- 聚类级歧义解析：258x区分度

Author: Claude Code
Date: 2026-04-20
"""

__version__ = "0.1.0"
__author__ = "ASI-Living-Tree Team"

from .core import LivingTreeNode, LivingTreeMemory, HopfieldRetrieval
from .entropy_retrieval import EntropyBasedRetrieval
from .energy_gap_retrieval import ClusterLevelEnergyGapRetrieval

__all__ = [
    "LivingTreeNode",
    "LivingTreeMemory",
    "HopfieldRetrieval",
    "EntropyBasedRetrieval",
    "ClusterLevelEnergyGapRetrieval",
]