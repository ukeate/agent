"""记忆关联图网络"""

import asyncio
from typing import List, Dict, Optional, Tuple, Set, Any
import networkx as nx
import numpy as np
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from .models import Memory, MemoryType
from .storage import MemoryStorage

logger = get_logger(__name__)

class MemoryAssociationGraph:
    """记忆关联网络管理器"""
    
    def __init__(self, storage: MemoryStorage):
        self.storage = storage
        self.graph = nx.DiGraph()  # 有向图
        self._edge_weights: Dict[Tuple[str, str], float] = {}
        self._access_patterns: Dict[str, List[datetime]] = {}
        
    def add_memory_node(self, memory: Memory):
        """添加记忆节点"""
        self.graph.add_node(
            memory.id,
            memory_type=memory.type.value if hasattr(memory.type, 'value') else str(memory.type),
            importance=memory.importance,
            created_at=memory.created_at,
            content_summary=memory.content[:100]  # 存储内容摘要
        )
        
        # 初始化访问模式
        if memory.id not in self._access_patterns:
            self._access_patterns[memory.id] = []
            
    def add_association(
        self,
        memory1: Memory,
        memory2: Memory,
        weight: float = 0.5,
        association_type: str = "related"
    ):
        """添加记忆关联"""
        # 确保节点存在
        if memory1.id not in self.graph:
            self.add_memory_node(memory1)
        if memory2.id not in self.graph:
            self.add_memory_node(memory2)
            
        # 添加边
        self.graph.add_edge(
            memory1.id,
            memory2.id,
            weight=weight,
            type=association_type,
            created_at=utc_now()
        )
        
        # 存储权重
        self._edge_weights[(memory1.id, memory2.id)] = weight
        
    def update_association_weight(
        self,
        memory1_id: str,
        memory2_id: str,
        delta: float = 0.1
    ):
        """更新关联权重"""
        if self.graph.has_edge(memory1_id, memory2_id):
            current_weight = self.graph[memory1_id][memory2_id].get('weight', 0.5)
            new_weight = min(1.0, max(0.0, current_weight + delta))
            self.graph[memory1_id][memory2_id]['weight'] = new_weight
            self._edge_weights[(memory1_id, memory2_id)] = new_weight
            
    async def activate_related(
        self,
        memory_id: str,
        depth: int = 2,
        min_weight: float = 0.3
    ) -> List[Tuple[Memory, float]]:
        """激活相关记忆链"""
        if memory_id not in self.graph:
            return []
            
        # 记录访问模式
        self._access_patterns.setdefault(memory_id, []).append(utc_now())
        
        # 获取ego网络(以memory_id为中心的子图)
        try:
            ego_graph = nx.ego_graph(self.graph, memory_id, radius=depth)
        except:
            return []
            
        # 计算节点的激活强度
        activation_scores = {}
        
        # BFS遍历计算激活强度
        visited = set()
        queue = [(memory_id, 1.0, 0)]  # (节点ID, 激活强度, 深度)
        
        while queue:
            node_id, strength, current_depth = queue.pop(0)
            
            if node_id in visited or current_depth > depth:
                continue
                
            visited.add(node_id)
            
            # 记录激活强度
            if node_id != memory_id:
                activation_scores[node_id] = max(
                    activation_scores.get(node_id, 0),
                    strength
                )
                
            # 传播激活到邻居
            if current_depth < depth:
                for neighbor in self.graph.successors(node_id):
                    edge_weight = self.graph[node_id][neighbor].get('weight', 0.5)
                    
                    if edge_weight >= min_weight:
                        # 激活强度衰减
                        propagated_strength = strength * edge_weight * (0.8 ** current_depth)
                        queue.append((neighbor, propagated_strength, current_depth + 1))
                        
        # 加载记忆对象
        results = []
        for node_id, score in activation_scores.items():
            memory = await self.storage.get_memory(node_id)
            if memory:
                results.append((memory, score))
                
        # 按激活强度排序
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
        
    def find_memory_clusters(self, min_cluster_size: int = 3) -> List[Set[str]]:
        """发现记忆簇(紧密关联的记忆组)"""
        # 转换为无向图以找社区
        undirected = self.graph.to_undirected()
        
        # 使用Louvain算法检测社区
        try:
            import community as community_louvain
            partition = community_louvain.best_partition(undirected)
            
            # 组织簇
            clusters: Dict[int, Set[str]] = {}
            for node, cluster_id in partition.items():
                clusters.setdefault(cluster_id, set()).add(node)
                
            # 过滤小簇
            return [
                cluster for cluster in clusters.values()
                if len(cluster) >= min_cluster_size
            ]
        except ImportError:
            # 如果没有community库，使用简单的连通分量
            components = nx.connected_components(undirected)
            return [
                comp for comp in components
                if len(comp) >= min_cluster_size
            ]
            
    def get_memory_importance_rank(self) -> List[Tuple[str, float]]:
        """获取记忆重要性排名(基于PageRank)"""
        if not self.graph.nodes():
            return []
            
        try:
            # 计算PageRank
            pagerank_scores = nx.pagerank(
                self.graph,
                weight='weight',
                alpha=0.85
            )
            
            # 结合节点自身的重要性
            final_scores = {}
            for node_id, pr_score in pagerank_scores.items():
                node_importance = self.graph.nodes[node_id].get('importance', 0.5)
                final_scores[node_id] = pr_score * 0.7 + node_importance * 0.3
                
            # 排序
            ranked = sorted(
                final_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            
            return ranked
            
        except Exception as e:
            logger.error(f"计算记忆重要性失败: {e}")
            return []
            
    def find_shortest_path(
        self,
        start_memory_id: str,
        end_memory_id: str
    ) -> Optional[List[str]]:
        """找到两个记忆之间的最短路径"""
        if start_memory_id not in self.graph or end_memory_id not in self.graph:
            return None
            
        try:
            path = nx.shortest_path(
                self.graph,
                start_memory_id,
                end_memory_id,
                weight=lambda u, v, d: 1.0 - d.get('weight', 0.5)  # 权重越高，距离越短
            )
            return path
        except nx.NetworkXNoPath:
            return None
            
    def get_memory_context(
        self,
        memory_id: str,
        max_neighbors: int = 5
    ) -> Dict[str, Any]:
        """获取记忆的上下文信息"""
        if memory_id not in self.graph:
            return {}
            
        context = {
            "memory_id": memory_id,
            "in_degree": self.graph.in_degree(memory_id),
            "out_degree": self.graph.out_degree(memory_id),
            "predecessors": [],
            "successors": [],
            "access_pattern": self._access_patterns.get(memory_id, [])
        }
        
        # 获取前驱节点(指向该记忆的)
        predecessors = list(self.graph.predecessors(memory_id))[:max_neighbors]
        for pred in predecessors:
            weight = self.graph[pred][memory_id].get('weight', 0.5)
            context["predecessors"].append({
                "id": pred,
                "weight": weight,
                "type": self.graph[pred][memory_id].get('type', 'related')
            })
            
        # 获取后继节点(该记忆指向的)
        successors = list(self.graph.successors(memory_id))[:max_neighbors]
        for succ in successors:
            weight = self.graph[memory_id][succ].get('weight', 0.5)
            context["successors"].append({
                "id": succ,
                "weight": weight,
                "type": self.graph[memory_id][succ].get('type', 'related')
            })
            
        return context
        
    def detect_memory_patterns(self) -> Dict[str, List[str]]:
        """检测记忆访问模式"""
        patterns = {
            "frequently_accessed": [],
            "recently_accessed": [],
            "co_accessed": [],
            "central_memories": []
        }
        
        current_time = utc_now()
        
        # 频繁访问的记忆
        access_counts = {
            mid: len(accesses)
            for mid, accesses in self._access_patterns.items()
        }
        patterns["frequently_accessed"] = [
            mid for mid, count in sorted(
                access_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            if count > 5
        ]
        
        # 最近访问的记忆
        recent_accesses = []
        for mid, accesses in self._access_patterns.items():
            if accesses:
                last_access = max(accesses)
                if (current_time - last_access).total_seconds() < 3600:  # 1小时内
                    recent_accesses.append((mid, last_access))
                    
        patterns["recently_accessed"] = [
            mid for mid, _ in sorted(
                recent_accesses,
                key=lambda x: x[1],
                reverse=True
            )[:10]
        ]
        
        # 中心记忆(高连接度)
        if self.graph.nodes():
            degree_centrality = nx.degree_centrality(self.graph)
            patterns["central_memories"] = [
                mid for mid, _ in sorted(
                    degree_centrality.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:10]
                if degree_centrality[mid] > 0.1
            ]
            
        return patterns
        
    def prune_weak_associations(self, min_weight: float = 0.1):
        """修剪弱关联"""
        edges_to_remove = []
        
        for u, v, data in self.graph.edges(data=True):
            if data.get('weight', 0.5) < min_weight:
                edges_to_remove.append((u, v))
                
        for edge in edges_to_remove:
            self.graph.remove_edge(*edge)
            if edge in self._edge_weights:
                del self._edge_weights[edge]
                
        logger.info(f"修剪了 {len(edges_to_remove)} 个弱关联")
        
    def save_graph_state(self) -> Dict:
        """保存图状态"""
        return {
            "nodes": list(self.graph.nodes(data=True)),
            "edges": list(self.graph.edges(data=True)),
            "edge_weights": self._edge_weights,
            "access_patterns": {
                k: [dt.isoformat() for dt in v]
                for k, v in self._access_patterns.items()
            }
        }
        
    def load_graph_state(self, state: Dict):
        """加载图状态"""
        self.graph.clear()
        
        # 恢复节点
        for node_id, node_data in state.get("nodes", []):
            self.graph.add_node(node_id, **node_data)
            
        # 恢复边
        for u, v, edge_data in state.get("edges", []):
            self.graph.add_edge(u, v, **edge_data)
            
        # 恢复权重
        self._edge_weights = state.get("edge_weights", {})
        
        # 恢复访问模式
        self._access_patterns = {
            k: [datetime.fromisoformat(dt) for dt in v]
            for k, v in state.get("access_patterns", {}).items()
        }
from src.core.logging import get_logger
