"""
聚类级能量间隔歧义解析模块
=========================

H60 v2 验证结果：
- 聚类级间隔 17x 区分度
- 最优 gap_threshold = 0.03

H61 Killer Demo 结果：
- 258x 区分度
- 版本路由 100% 准确

H62 真实代码库结果：
- requests 库 API 冲突 90.9% 准确

核心公式：
- Cluster_Energy[cluster] = max_sim(query, cluster_nodes)
- Energy_Gap = E_top_cluster - E_second_cluster
- is_ambiguous = Energy_Gap < threshold

Author: Claude Code
Date: 2026-04-20
"""

import numpy as np
from typing import List, Tuple, Dict
from collections import defaultdict, Counter
from .core import LivingTreeNode


class ClusterLevelEnergyGapRetrieval:
    """
    聚类级能量间隔驱动的歧义解析检索

    与 node-level gap 的区别：
    - Node-level: gap = E_top_node - E_second_node
    - Cluster-level: gap = max_sim(Cluster_A) - max_sim(Cluster_B)

    优势：
    - 对节点噪声鲁棒
    - 物理意义更清晰
    """

    def __init__(self, beta: float = 15.0, gap_threshold: float = 0.03):
        """
        Args:
            beta: 能量分布温度参数
            gap_threshold: 能量间隔阈值，小于该值判定为歧义
        """
        self.beta = beta
        self.gap_threshold = gap_threshold
        self.nodes: List[LivingTreeNode] = []
        self.ambiguity_detected = 0
        self.total_queries = 0

    def store(self, nodes: List[LivingTreeNode]):
        """存储节点"""
        self.nodes = nodes

    def compute_cluster_energies(self, sims: np.ndarray) -> Dict:
        """
        计算每个聚类的最高能量（吸引子势能）
        """
        cluster_energies = {}
        for i, sim in enumerate(sims):
            cluster = self.nodes[i].metadata.get('version',
                                                  self.nodes[i].metadata.get('cluster', i))
            if cluster not in cluster_energies or sim > cluster_energies[cluster]:
                cluster_energies[cluster] = sim
        return cluster_energies

    def compute_energy_gap(self, cluster_energies: Dict) -> Tuple[float, float, float]:
        """
        计算聚类间的能量间隔

        Returns:
            (gap, top_energy, second_energy)
        """
        sorted_energies = sorted(cluster_energies.values(), reverse=True)
        top_e = sorted_energies[0]
        second_e = sorted_energies[1] if len(sorted_energies) > 1 else 0
        return top_e - second_e, top_e, second_e

    def retrieve(self, query_emb: np.ndarray, k: int = 5,
                 return_ambiguity: bool = False) -> Tuple:
        """
        基于聚类级能量间隔的检索

        Returns:
            (节点列表, 相似度列表) 或 (节点列表, 相似度列表, 歧义信息)
        """
        self.total_queries += 1

        # 归一化查询
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        # 计算相似度
        patterns = np.array([n.embedding for n in self.nodes])
        sims = patterns @ q

        # 计算每个聚类的最高能量
        cluster_energies = self.compute_cluster_energies(sims)

        # 计算能量间隔
        energy_gap, top_e, second_e = self.compute_energy_gap(cluster_energies)

        # 歧义判定
        is_ambiguous = energy_gap < self.gap_threshold

        if is_ambiguous:
            self.ambiguity_detected += 1

        # 找出获胜的版本/聚类
        winning_version = max(cluster_energies.items(), key=lambda x: x[1])[0]

        # 获取top-k
        top_indices = np.argsort(-sims)[:k]
        top_nodes = [self.nodes[i] for i in top_indices]
        top_sims = [float(sims[i]) for i in top_indices]

        if return_ambiguity:
            return top_nodes, top_sims, {
                'energy_gap': float(energy_gap),
                'top_cluster_e': float(top_e),
                'second_cluster_e': float(second_e),
                'is_ambiguous': is_ambiguous,
                'winning_version': winning_version,
                'cluster_energies': {k: float(v) for k, v in cluster_energies.items()},
                'top_clusters': [self.nodes[i].metadata.get('version',
                                                          self.nodes[i].metadata.get('cluster', -1))
                                for i in top_indices]
            }

        return top_nodes, top_sims

    def retrieve_multi_route(self, query_emb: np.ndarray, k: int = 5,
                           n_routes: int = 3) -> Tuple:
        """
        多路返回：基于聚类级能量间隔触发

        使用轮询算法确保每个聚类都有代表节点
        """
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        patterns = np.array([n.embedding for n in self.nodes])
        sims = patterns @ q

        cluster_energies = self.compute_cluster_energies(sims)
        energy_gap, _, _ = self.compute_energy_gap(cluster_energies)

        is_ambiguous = energy_gap < self.gap_threshold

        if not is_ambiguous:
            top_indices = np.argsort(-sims)[:k]
            return [self.nodes[i] for i in top_indices], \
                   [float(sims[i]) for i in top_indices], 'single'

        # 多路返回：按聚类能量排序，轮询选取
        sorted_clusters = sorted(cluster_energies.items(), key=lambda x: x[1], reverse=True)

        cluster_nodes = defaultdict(list)

        # 收集每个聚类的节点（按相似度排序）
        for idx in np.argsort(-sims):
            cluster = self.nodes[idx].metadata.get('version',
                                                   self.nodes[idx].metadata.get('cluster', -1))
            if len(cluster_nodes[cluster]) < k // n_routes + 2:
                cluster_nodes[cluster].append(self.nodes[idx])

        # 轮询选取
        route_nodes = []
        max_per_cluster = k // n_routes + 2
        round_idx = 0
        while len(route_nodes) < k and round_idx < max_per_cluster:
            for cluster, _ in sorted_clusters:
                if round_idx < len(cluster_nodes[cluster]):
                    route_nodes.append(cluster_nodes[cluster][round_idx])
                if len(route_nodes) >= k:
                    break
            round_idx += 1

        route_nodes = route_nodes[:k]
        route_sims = [float(sims[n.node_id]) for n in route_nodes]

        return route_nodes, route_sims, 'multi'


class EnergyGapRetrieval:
    """
    Node-level 能量间隔检索（来自 H60）

    注意：推荐使用 ClusterLevelEnergyGapRetrieval（更高区分度）
    """

    def __init__(self, beta: float = 15.0, gap_threshold: float = 0.05):
        self.beta = beta
        self.gap_threshold = gap_threshold
        self.nodes: List[LivingTreeNode] = []

    def store(self, nodes: List[LivingTreeNode]):
        self.nodes = nodes

    def compute_energy_distribution(self, sims: np.ndarray) -> np.ndarray:
        exp_sims = np.exp((sims - np.max(sims)) * self.beta)
        probs = exp_sims / exp_sims.sum()
        return probs

    def retrieve(self, query_emb: np.ndarray, k: int = 5,
                 return_ambiguity: bool = False) -> Tuple:
        q = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        patterns = np.array([n.embedding for n in self.nodes])
        sims = patterns @ q

        probs = self.compute_energy_distribution(sims)

        top_indices = np.argsort(-sims)
        E_top = sims[top_indices[0]]
        E_second = sims[top_indices[1]]
        gap_absolute = E_top - E_second

        is_ambiguous = gap_absolute < self.gap_threshold

        top_k_indices = top_indices[:k]
        top_nodes = [self.nodes[i] for i in top_k_indices]
        top_sims = [float(sims[i]) for i in top_k_indices]

        if return_ambiguity:
            return top_nodes, top_sims, {
                'energy_gap': float(gap_absolute),
                'is_ambiguous': is_ambiguous,
                'top_e': float(E_top),
                'second_e': float(E_second)
            }

        return top_nodes, top_sims


# 示例用法
if __name__ == "__main__":
    import numpy as np
    from living_tree.core import LivingTreeMemory

    # 创建测试记忆（3个版本）
    memory = LivingTreeMemory()

    for version in ['v1', 'v2', 'v2adv']:
        for i in range(50):
            emb = np.random.randn(768).astype(np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-8)
            memory.add_node(emb, f"{version}_impl_{i}", {'version': version})

    # 创建聚类级能量间隔检索器
    retrieval = ClusterLevelEnergyGapRetrieval(beta=15.0, gap_threshold=0.03)
    retrieval.store(memory.nodes)

    # 测试查询
    query = np.random.randn(768).astype(np.float32)
    query = query / (np.linalg.norm(query) + 1e-8)

    nodes, sims, info = retrieval.retrieve(query, k=5, return_ambiguity=True)

    print(f"Energy Gap: {info['energy_gap']:.3f}")
    print(f"Is ambiguous: {info['is_ambiguous']}")
    print(f"Winning version: {info['winning_version']}")
    print(f"Cluster energies: {info['cluster_energies']}")
    print(f"Retrieved {len(nodes)} nodes")