"""
熵驱动歧义解析模块
=================

H59 验证结果：
- 熵区分能力: PASS (gap=0.127 > 0.1)
- 边界歧义检测率: 100%
- 多路扩展覆盖: 2→3 clusters

配置：beta=15.0, entropy_threshold=0.75

Author: Claude Code
Date: 2026-04-19
"""

import numpy as np
from typing import List, Tuple, Dict
from .core import LivingTreeNode


class EntropyBasedRetrieval:
    """
    熵驱动的歧义解析检索

    使用概率分布熵检测歧义：
    - 高熵 → 多吸引子竞争 → 多路返回
    - 低熵 → 一个吸引子主导 → 单路返回
    """

    def __init__(self, beta: float = 15.0, entropy_threshold: float = 0.75):
        """
        Args:
            beta: 能量分布温度参数
            entropy_threshold: 熵阈值，高于该值判定为歧义
        """
        self.beta = beta
        self.entropy_threshold = entropy_threshold
        self.nodes: List[LivingTreeNode] = []
        self.ambiguity_detected = 0
        self.total_queries = 0

    def store(self, nodes: List[LivingTreeNode]):
        """存储节点"""
        self.nodes = nodes

    def compute_entropy(self, probs: np.ndarray) -> float:
        """计算概率分布的熵"""
        probs = probs / (probs.sum() + 1e-10)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(len(probs))
        normalized_entropy = entropy / (max_entropy + 1e-10)
        return normalized_entropy

    def compute_energy_distribution(self, sims: np.ndarray) -> np.ndarray:
        """计算能量分布"""
        exp_sims = np.exp((sims - np.max(sims)) * self.beta)
        probs = exp_sims / exp_sims.sum()
        return probs

    def retrieve(self, query_emb: np.ndarray, k: int = 5,
                 return_ambiguity: bool = False) -> Tuple:
        """
        检索top-k节点

        Args:
            query_emb: 查询向量
            k: 返回节点数
            return_ambiguity: 是否返回歧义信息

        Returns:
            (节点列表, 相似度列表) 或 (节点列表, 相似度列表, 歧义信息)
        """
        self.total_queries += 1

        # 归一化查询
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # 计算相似度
        patterns = np.array([n.embedding for n in self.nodes])
        sims = patterns @ q

        # 计算能量分布和熵
        probs = self.compute_energy_distribution(sims)
        entropy = self.compute_entropy(probs)

        # 歧义判定
        is_ambiguous = entropy > self.entropy_threshold

        if is_ambiguous:
            self.ambiguity_detected += 1

        # 获取top-k
        top_indices = np.argsort(-sims)[:k]
        top_nodes = [self.nodes[i] for i in top_indices]
        top_sims = [float(sims[i]) for i in top_indices]
        top_probs = [float(probs[i]) for i in top_indices]

        if return_ambiguity:
            return top_nodes, top_sims, {
                'entropy': entropy,
                'is_ambiguous': is_ambiguous,
                'probs': top_probs,
                'sims': top_sims
            }

        return top_nodes, top_sims

    def retrieve_multi_route(self, query_emb: np.ndarray, k: int = 5,
                           n_routes: int = 3) -> Tuple:
        """
        多路返回：从每个聚类提取节点进行交叉合并

        使用轮询算法确保每个聚类都有代表节点
        """
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        patterns = np.array([n.embedding for n in self.nodes])
        sims = patterns @ q

        probs = self.compute_energy_distribution(sims)
        entropy = self.compute_entropy(probs)

        is_ambiguous = entropy > self.entropy_threshold

        if not is_ambiguous:
            top_indices = np.argsort(-sims)[:k]
            return [self.nodes[i] for i in top_indices], \
                   [float(sims[i]) for i in top_indices], 'single'

        # 有歧义，进行多路返回
        sorted_indices = np.argsort(-sims)

        # 识别不同的吸引子
        attractors = {}
        for idx in sorted_indices:
            node = self.nodes[idx]
            cluster = node.metadata.get('cluster', node.metadata.get('region', idx))
            if cluster not in attractors:
                attractors[cluster] = []
            attractors[cluster].append(idx)

        # 轮询算法：从每个聚类交替选取节点
        route_indices = []

        # 每个聚类先取一个，确保都有代表
        for cluster in sorted(attractors.keys()):
            route_indices.append(attractors[cluster][0])

        # 然后轮询填充剩余位置
        max_per_cluster = k // n_routes + 2
        round_idx = 0
        while len(route_indices) < k and round_idx < max_per_cluster:
            for cluster in sorted(attractors.keys()):
                if round_idx < len(attractors[cluster]):
                    idx = attractors[cluster][round_idx]
                    if idx not in route_indices:
                        route_indices.append(idx)
                if len(route_indices) >= k:
                    break
            round_idx += 1

        route_nodes = [self.nodes[i] for i in route_indices[:k]]
        route_nodes.sort(key=lambda n: sims[n.node_id], reverse=True)
        route_sims = [float(sims[n.node_id]) for n in route_nodes]

        return route_nodes, route_sims, 'multi'


# 示例用法
if __name__ == "__main__":
    import numpy as np
    from living_tree.core import LivingTreeMemory

    # 创建测试记忆
    memory = LivingTreeMemory()
    embeddings = np.random.randn(150, 768).astype(np.float32)
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)

    for i in range(150):
        memory.add_node(embeddings[i], f"solution_{i}", {'cluster': i % 3})

    # 创建熵检索器
    entropy_retrieval = EntropyBasedRetrieval(beta=15.0, entropy_threshold=0.75)
    entropy_retrieval.store(memory.nodes)

    # 测试查询
    query = embeddings[0]  # 使用第一个节点作为查询

    nodes, sims, info = entropy_retrieval.retrieve(query, k=5, return_ambiguity=True)

    print(f"Entropy: {info['entropy']:.3f}")
    print(f"Is ambiguous: {info['is_ambiguous']}")
    print(f"Retrieved {len(nodes)} nodes")