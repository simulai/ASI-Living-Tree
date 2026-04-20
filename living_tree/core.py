"""
Living Tree 核心模块
===================

包含：
- LivingTreeNode: 知识图谱节点
- LivingTreeMemory: 知识存储
- HopfieldRetrieval: Hopfield动力学检索
- ComplementaryGating: 互补门控

Author: Claude Code
Date: 2026-04-20
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class LivingTreeNode:
    """知识图谱节点"""

    def __init__(self, node_id: int, embedding: np.ndarray,
                 solution: str, metadata: Optional[Dict] = None):
        self.node_id = node_id
        self.embedding = embedding
        self.solution = solution
        self.metadata = metadata or {}


class LivingTreeMemory:
    """知识存储系统 - 支持O(1)节点添加"""

    def __init__(self):
        self.nodes: List[LivingTreeNode] = []

    def add_node(self, embedding: np.ndarray, solution: str,
                 metadata: Optional[Dict] = None) -> int:
        """添加节点，返回node_id"""
        node = LivingTreeNode(len(self.nodes), embedding, solution, metadata)
        self.nodes.append(node)
        return node.node_id

    def store_batch(self, embeddings: List[np.ndarray],
                    solutions: List[str],
                    metadata: Optional[List[Dict]] = None):
        """批量存储节点"""
        for i, (emb, sol) in enumerate(zip(embeddings, solutions)):
            meta = metadata[i] if metadata else None
            self.add_node(emb, sol, meta)

    def get_node(self, node_id: int) -> Optional[LivingTreeNode]:
        """获取节点"""
        if 0 <= node_id < len(self.nodes):
            return self.nodes[node_id]
        return None

    def __len__(self) -> int:
        return len(self.nodes)


class HopfieldRetrieval:
    """
    Hopfield动力学检索

    核心公式: E(x, memory) = -β · exp(sim(x, memory))
    """

    def __init__(self, beta: float = 15.0):
        self.beta = beta
        self.nodes: List[LivingTreeNode] = []

    def store(self, nodes: List[LivingTreeNode]):
        """存储节点"""
        self.nodes = nodes

    def retrieve(self, query_emb: np.ndarray, k: int = 5) -> Tuple[List[LivingTreeNode], List[float]]:
        """
        检索top-k节点

        Args:
            query_emb: 查询向量
            k: 返回节点数

        Returns:
            (节点列表, 相似度列表)
        """
        # 归一化查询
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # 计算所有相似度
        patterns = np.array([n.embedding for n in self.nodes])
        sims = patterns @ q

        # 排序取top-k
        top_indices = np.argsort(-sims)[:k]
        return [self.nodes[i] for i in top_indices], [float(sims[i]) for i in top_indices]

    def retrieve_with_energy(self, query_emb: np.ndarray, k: int = 5) -> Tuple[List[LivingTreeNode], List[float], Dict]:
        """带能量的检索"""
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        patterns = np.array([n.embedding for n in self.nodes])
        sims = patterns @ q

        # 计算能量分布
        exp_sims = np.exp((sims - np.max(sims)) * self.beta)
        probs = exp_sims / exp_sims.sum()

        top_indices = np.argsort(-sims)[:k]
        top_nodes = [self.nodes[i] for i in top_indices]
        top_sims = [float(sims[i]) for i in top_indices]
        top_probs = [float(probs[i]) for i in top_indices]

        return top_nodes, top_sims, {
            'energies': top_probs,
            'top_indices': top_indices.tolist()
        }


class ComplementaryGating:
    """
    互补门控 - 防止错误传播

    原理：Top-K选择 + 能量门控
    """

    def __init__(self, top_k: int = 2, energy_threshold: float = 0.1):
        self.top_k = top_k
        self.energy_threshold = energy_threshold

    def filter(self, nodes: List[LivingTreeNode],
               energies: List[float]) -> List[LivingTreeNode]:
        """互补门控过滤"""
        if len(nodes) <= self.top_k:
            return nodes

        # Top-K 选择
        top_indices = np.argsort(-np.array(energies))[:self.top_k]
        return [nodes[i] for i in top_indices]

    def select(self, nodes: List[LivingTreeNode],
               energies: List[float]) -> List[LivingTreeNode]:
        """选择能量高于阈值的节点"""
        threshold = np.mean(energies) * self.energy_threshold
        return [n for n, e in zip(nodes, energies) if e >= threshold]


def create_memory_from_embeddings(embeddings: np.ndarray,
                                  solutions: List[str],
                                  metadata: Optional[List[Dict]] = None) -> LivingTreeMemory:
    """
    从嵌入向量创建知识记忆

    Args:
        embeddings: NxD 嵌入矩阵
        solutions: N 个解决方案文本
        metadata: N 个元数据字典

    Returns:
        LivingTreeMemory
    """
    memory = LivingTreeMemory()
    memory.store_batch(embeddings, solutions, metadata)
    return memory


# 示例用法
if __name__ == "__main__":
    # 创建简单的知识记忆
    memory = LivingTreeMemory()

    # 添加一些节点
    for i in range(10):
        emb = np.random.randn(768).astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        memory.add_node(emb, f"solution_{i}", {'cluster': i % 3})

    print(f"Memory size: {len(memory)}")

    # 创建检索器
    retrieval = HopfieldRetrieval(beta=15.0)
    retrieval.store(memory.nodes)

    # 查询
    query = np.random.randn(768).astype(np.float32)
    query = query / (np.linalg.norm(query) + 1e-8)

    nodes, sims = retrieval.retrieve(query, k=3)
    print(f"Retrieved {len(nodes)} nodes")
    print(f"Top similarities: {sims}")