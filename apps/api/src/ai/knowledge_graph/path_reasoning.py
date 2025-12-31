"""
多跳路径推理器 - 基于图遍历的多步推理和路径发现

实现功能:
- 1-5跳的多步推理路径搜索
- 路径置信度计算和最优路径选择
- 推理路径的可解释性输出
- 路径剪枝和性能优化
- 异步图遍历和并行搜索

技术栈:
- 广度优先搜索(BFS)和深度优先搜索(DFS)
- 启发式搜索和路径剪枝
- 图数据库集成
- 路径缓存和优化
"""

import asyncio
import heapq
from collections import defaultdict, deque
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional, Any, Union, Callable
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import threading

from src.core.logging import get_logger
logger = get_logger(__name__)

class SearchStrategy(str, Enum):
    """搜索策略"""
    BREADTH_FIRST = "bfs"
    DEPTH_FIRST = "dfs" 
    BEST_FIRST = "best_first"
    A_STAR = "a_star"
    BIDIRECTIONAL = "bidirectional"

class PathType(str, Enum):
    """路径类型"""
    SHORTEST = "shortest"
    MOST_CONFIDENT = "most_confident"
    DIVERSE = "diverse"
    ALL_PATHS = "all"

@dataclass
class PathNode:
    """路径节点"""
    entity: str
    relation: Optional[str] = None
    confidence: float = 1.0
    depth: int = 0
    
    def __hash__(self):
        return hash((self.entity, self.relation))

@dataclass
class ReasoningPath:
    """推理路径数据结构"""
    id: str
    start_entity: str
    end_entity: str
    nodes: List[PathNode] = field(default_factory=list)
    total_confidence: float = 1.0
    length: int = 0
    explanation: str = ""
    execution_time_ms: float = 0.0
    search_strategy: SearchStrategy = SearchStrategy.BREADTH_FIRST
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)
    
    @property
    def path_relations(self) -> List[str]:
        """获取路径中的关系序列"""
        return [node.relation for node in self.nodes[1:] if node.relation]
    
    @property
    def path_entities(self) -> List[str]:
        """获取路径中的实体序列"""
        return [node.entity for node in self.nodes]
    
    def to_triple_sequence(self) -> List[Tuple[str, str, str]]:
        """转换为三元组序列"""
        triples = []
        for i in range(len(self.nodes) - 1):
            current = self.nodes[i]
            next_node = self.nodes[i + 1]
            if next_node.relation:
                triples.append((current.entity, next_node.relation, next_node.entity))
        return triples

@dataclass
class PathSearchConfig:
    """路径搜索配置"""
    max_hops: int = 5
    max_paths: int = 100
    min_confidence: float = 0.1
    search_strategy: SearchStrategy = SearchStrategy.BREADTH_FIRST
    path_type: PathType = PathType.MOST_CONFIDENT
    enable_caching: bool = True
    timeout_seconds: int = 30
    prune_threshold: float = 0.05
    diversity_threshold: float = 0.8

class PathSearchResult:
    """路径搜索结果"""
    def __init__(self):
        self.paths: List[ReasoningPath] = []
        self.total_searched: int = 0
        self.cache_hits: int = 0
        self.execution_time: float = 0.0
        self.converged: bool = False
        self.error: Optional[str] = None
        self.metadata: Dict[str, Any] = {}

class HeuristicFunction:
    """启发式函数接口"""
    
    def __init__(self, graph_db=None, embedding_engine=None):
        self.graph_db = graph_db
        self.embedding_engine = embedding_engine
    
    async def estimate_distance(self, current: str, target: str) -> float:
        """估计从当前节点到目标节点的距离"""
        if self.embedding_engine:
            return await self._embedding_based_heuristic(current, target)
        return 0.0  # 默认启发式函数
    
    async def _embedding_based_heuristic(self, entity1: str, entity2: str) -> float:
        """基于嵌入向量的启发式函数"""
        try:
            similar_entities = await self.embedding_engine.find_similar_entities(entity1, top_k=100)
            for similar_entity, similarity in similar_entities:
                if similar_entity == entity2:
                    return 1.0 - similarity  # 相似度越高，距离越小
            return 1.0  # 默认距离
        except Exception:
            return 1.0

class PathReasoner:
    """多跳路径推理器"""
    
    def __init__(self, graph_db=None, embedding_engine=None, config: PathSearchConfig = None):
        self.graph_db = graph_db
        self.embedding_engine = embedding_engine
        self.config = config or PathSearchConfig()
        
        # 路径缓存
        self.path_cache: Dict[str, List[ReasoningPath]] = {}
        self.neighbor_cache: Dict[str, List[Tuple[str, str, float]]] = {}
        
        # 统计信息
        self.total_searches = 0
        self.cache_hits = 0
        self.total_paths_found = 0
        
        # 启发式函数
        self.heuristic = HeuristicFunction(graph_db, embedding_engine)
        
        # 异步执行器
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()
    
    async def find_reasoning_paths(self,
                                 start_entity: str,
                                 end_entity: str,
                                 relation_constraints: List[str] = None,
                                 config: PathSearchConfig = None) -> PathSearchResult:
        """查找推理路径"""
        search_config = config or self.config
        start_time = utc_now()
        
        try:
            # 检查缓存
            cache_key = self._generate_cache_key(start_entity, end_entity, relation_constraints)
            if search_config.enable_caching and cache_key in self.path_cache:
                self.cache_hits += 1
                result = PathSearchResult()
                result.paths = self.path_cache[cache_key]
                result.cache_hits = 1
                result.execution_time = 0.0
                return result
            
            # 执行路径搜索
            if search_config.search_strategy == SearchStrategy.BREADTH_FIRST:
                paths = await self._breadth_first_search(start_entity, end_entity, relation_constraints, search_config)
            elif search_config.search_strategy == SearchStrategy.DEPTH_FIRST:
                paths = await self._depth_first_search(start_entity, end_entity, relation_constraints, search_config)
            elif search_config.search_strategy == SearchStrategy.BEST_FIRST:
                paths = await self._best_first_search(start_entity, end_entity, relation_constraints, search_config)
            elif search_config.search_strategy == SearchStrategy.A_STAR:
                paths = await self._a_star_search(start_entity, end_entity, relation_constraints, search_config)
            elif search_config.search_strategy == SearchStrategy.BIDIRECTIONAL:
                paths = await self._bidirectional_search(start_entity, end_entity, relation_constraints, search_config)
            else:
                paths = await self._breadth_first_search(start_entity, end_entity, relation_constraints, search_config)
            
            # 后处理路径
            paths = await self._post_process_paths(paths, search_config)
            
            # 缓存结果
            if search_config.enable_caching:
                self.path_cache[cache_key] = paths
            
            # 构建结果
            result = PathSearchResult()
            result.paths = paths
            result.execution_time = (utc_now() - start_time).total_seconds() * 1000
            result.converged = True
            
            self.total_searches += 1
            self.total_paths_found += len(paths)
            
            return result
            
        except Exception as e:
            logger.error(f"Path search failed: {str(e)}")
            result = PathSearchResult()
            result.error = str(e)
            result.execution_time = (utc_now() - start_time).total_seconds() * 1000
            return result
    
    async def _breadth_first_search(self,
                                  start: str,
                                  end: str,
                                  constraints: List[str],
                                  config: PathSearchConfig) -> List[ReasoningPath]:
        """广度优先搜索"""
        queue = deque([(start, [PathNode(entity=start)], 1.0)])
        visited = set()
        found_paths = []
        
        while queue and len(found_paths) < config.max_paths:
            current_entity, path, confidence = queue.popleft()
            
            if len(path) > config.max_hops + 1:
                continue
            
            if current_entity == end and len(path) > 1:
                # 找到目标路径
                reasoning_path = ReasoningPath(
                    id=self._generate_path_id(),
                    start_entity=start,
                    end_entity=end,
                    nodes=path.copy(),
                    total_confidence=confidence,
                    length=len(path) - 1,
                    search_strategy=SearchStrategy.BREADTH_FIRST
                )
                reasoning_path.explanation = self._generate_explanation(reasoning_path)
                found_paths.append(reasoning_path)
                continue
            
            # 获取邻居节点
            neighbors = await self._get_entity_neighbors(current_entity, constraints)
            
            for neighbor_entity, relation, rel_confidence in neighbors:
                if neighbor_entity not in visited or len(path) < 3:  # 允许短路径重访
                    new_path = path + [PathNode(entity=neighbor_entity, relation=relation, confidence=rel_confidence)]
                    new_confidence = confidence * rel_confidence
                    
                    if new_confidence >= config.min_confidence:
                        queue.append((neighbor_entity, new_path, new_confidence))
            
            visited.add(current_entity)
        
        return found_paths
    
    async def _depth_first_search(self,
                                start: str,
                                end: str,
                                constraints: List[str],
                                config: PathSearchConfig) -> List[ReasoningPath]:
        """深度优先搜索"""
        found_paths = []
        visited = set()
        
        async def dfs_recursive(current_entity: str, path: List[PathNode], confidence: float, depth: int):
            if len(found_paths) >= config.max_paths or depth > config.max_hops:
                return
            
            if current_entity == end and depth > 0:
                reasoning_path = ReasoningPath(
                    id=self._generate_path_id(),
                    start_entity=start,
                    end_entity=end,
                    nodes=path.copy(),
                    total_confidence=confidence,
                    length=depth,
                    search_strategy=SearchStrategy.DEPTH_FIRST
                )
                reasoning_path.explanation = self._generate_explanation(reasoning_path)
                found_paths.append(reasoning_path)
                return
            
            if current_entity in visited:
                return
            
            visited.add(current_entity)
            
            neighbors = await self._get_entity_neighbors(current_entity, constraints)
            
            for neighbor_entity, relation, rel_confidence in neighbors:
                if neighbor_entity not in visited:
                    new_path = path + [PathNode(entity=neighbor_entity, relation=relation, confidence=rel_confidence)]
                    new_confidence = confidence * rel_confidence
                    
                    if new_confidence >= config.min_confidence:
                        await dfs_recursive(neighbor_entity, new_path, new_confidence, depth + 1)
            
            visited.remove(current_entity)
        
        await dfs_recursive(start, [PathNode(entity=start)], 1.0, 0)
        return found_paths
    
    async def _best_first_search(self,
                               start: str,
                               end: str,
                               constraints: List[str],
                               config: PathSearchConfig) -> List[ReasoningPath]:
        """最佳优先搜索"""
        # 优先队列：(-confidence, path_length, current_entity, path, confidence)
        priority_queue = [(-1.0, 0, start, [PathNode(entity=start)], 1.0)]
        visited = set()
        found_paths = []
        
        while priority_queue and len(found_paths) < config.max_paths:
            neg_conf, path_len, current_entity, path, confidence = heapq.heappop(priority_queue)
            
            if path_len > config.max_hops:
                continue
            
            if current_entity == end and path_len > 0:
                reasoning_path = ReasoningPath(
                    id=self._generate_path_id(),
                    start_entity=start,
                    end_entity=end,
                    nodes=path.copy(),
                    total_confidence=confidence,
                    length=path_len,
                    search_strategy=SearchStrategy.BEST_FIRST
                )
                reasoning_path.explanation = self._generate_explanation(reasoning_path)
                found_paths.append(reasoning_path)
                continue
            
            if current_entity in visited:
                continue
            
            visited.add(current_entity)
            
            neighbors = await self._get_entity_neighbors(current_entity, constraints)
            
            for neighbor_entity, relation, rel_confidence in neighbors:
                if neighbor_entity not in visited:
                    new_path = path + [PathNode(entity=neighbor_entity, relation=relation, confidence=rel_confidence)]
                    new_confidence = confidence * rel_confidence
                    
                    if new_confidence >= config.min_confidence:
                        heapq.heappush(priority_queue, (
                            -new_confidence,  # 负数用于最大堆
                            path_len + 1,
                            neighbor_entity,
                            new_path,
                            new_confidence
                        ))
        
        return found_paths
    
    async def _a_star_search(self,
                           start: str,
                           end: str,
                           constraints: List[str],
                           config: PathSearchConfig) -> List[ReasoningPath]:
        """A*搜索算法"""
        # 优先队列：(f_score, current_entity, path, g_score)
        priority_queue = [(0.0, start, [PathNode(entity=start)], 0.0)]
        visited = set()
        found_paths = []
        
        while priority_queue and len(found_paths) < config.max_paths:
            f_score, current_entity, path, g_score = heapq.heappop(priority_queue)
            
            if len(path) > config.max_hops + 1:
                continue
            
            if current_entity == end and len(path) > 1:
                reasoning_path = ReasoningPath(
                    id=self._generate_path_id(),
                    start_entity=start,
                    end_entity=end,
                    nodes=path.copy(),
                    total_confidence=1.0 - g_score,  # 转换为置信度
                    length=len(path) - 1,
                    search_strategy=SearchStrategy.A_STAR
                )
                reasoning_path.explanation = self._generate_explanation(reasoning_path)
                found_paths.append(reasoning_path)
                continue
            
            if current_entity in visited:
                continue
            
            visited.add(current_entity)
            
            neighbors = await self._get_entity_neighbors(current_entity, constraints)
            
            for neighbor_entity, relation, rel_confidence in neighbors:
                if neighbor_entity not in visited:
                    # g(n): 从起点到当前节点的实际成本
                    new_g_score = g_score + (1.0 - rel_confidence)
                    
                    # h(n): 启发式函数估计的成本
                    h_score = await self.heuristic.estimate_distance(neighbor_entity, end)
                    
                    # f(n) = g(n) + h(n)
                    new_f_score = new_g_score + h_score
                    
                    if new_g_score < 1.0:  # 置信度阈值检查
                        new_path = path + [PathNode(entity=neighbor_entity, relation=relation, confidence=rel_confidence)]
                        heapq.heappush(priority_queue, (new_f_score, neighbor_entity, new_path, new_g_score))
        
        return found_paths
    
    async def _bidirectional_search(self,
                                  start: str,
                                  end: str,
                                  constraints: List[str],
                                  config: PathSearchConfig) -> List[ReasoningPath]:
        """双向搜索"""
        # 前向搜索
        forward_queue = deque([(start, [PathNode(entity=start)], 1.0)])
        forward_visited = {start: ([PathNode(entity=start)], 1.0)}
        
        # 后向搜索
        backward_queue = deque([(end, [PathNode(entity=end)], 1.0)])
        backward_visited = {end: ([PathNode(entity=end)], 1.0)}
        
        found_paths = []
        max_depth = config.max_hops // 2 + 1
        
        for depth in range(max_depth):
            if len(found_paths) >= config.max_paths:
                break
            
            # 前向搜索一层
            await self._expand_search_layer(forward_queue, forward_visited, constraints, config)
            
            # 后向搜索一层
            await self._expand_search_layer(backward_queue, backward_visited, constraints, config)
            
            # 检查交集
            intersection = set(forward_visited.keys()) & set(backward_visited.keys())
            
            for meeting_point in intersection:
                if meeting_point == start or meeting_point == end:
                    continue
                
                forward_path, forward_conf = forward_visited[meeting_point]
                backward_path, backward_conf = backward_visited[meeting_point]
                
                # 合并路径（需要反转后向路径）
                merged_path = forward_path + backward_path[::-1][1:]  # 去除重复的交汇点
                merged_confidence = forward_conf * backward_conf
                
                if merged_confidence >= config.min_confidence:
                    reasoning_path = ReasoningPath(
                        id=self._generate_path_id(),
                        start_entity=start,
                        end_entity=end,
                        nodes=merged_path,
                        total_confidence=merged_confidence,
                        length=len(merged_path) - 1,
                        search_strategy=SearchStrategy.BIDIRECTIONAL
                    )
                    reasoning_path.explanation = self._generate_explanation(reasoning_path)
                    found_paths.append(reasoning_path)
        
        return found_paths
    
    async def _expand_search_layer(self,
                                 queue: deque,
                                 visited: Dict,
                                 constraints: List[str],
                                 config: PathSearchConfig):
        """扩展搜索层"""
        queue_size = len(queue)
        
        for _ in range(queue_size):
            if not queue:
                break
            
            current_entity, path, confidence = queue.popleft()
            neighbors = await self._get_entity_neighbors(current_entity, constraints)
            
            for neighbor_entity, relation, rel_confidence in neighbors:
                if neighbor_entity not in visited:
                    new_path = path + [PathNode(entity=neighbor_entity, relation=relation, confidence=rel_confidence)]
                    new_confidence = confidence * rel_confidence
                    
                    if new_confidence >= config.min_confidence:
                        queue.append((neighbor_entity, new_path, new_confidence))
                        visited[neighbor_entity] = (new_path, new_confidence)
    
    async def _get_entity_neighbors(self,
                                  entity: str,
                                  constraints: List[str] = None) -> List[Tuple[str, str, float]]:
        """获取实体的邻居节点"""
        cache_key = f"{entity}:{':'.join(constraints or [])}"
        
        # 检查缓存
        if cache_key in self.neighbor_cache:
            return self.neighbor_cache[cache_key]
        
        neighbors = []
        
        try:
            if self.graph_db:
                # 从图数据库查询邻居
                constraint_clause = ""
                if constraints:
                    constraint_clause = f"AND type(r) IN {constraints}"
                
                query = f"""
                MATCH (e1)-[r]->(e2)
                WHERE e1.name = $entity {constraint_clause}
                RETURN e2.name as neighbor, type(r) as relation, 
                       coalesce(r.confidence, 1.0) as confidence
                LIMIT 100
                """
                
                results = await self.graph_db.execute_query(query, {"entity": entity})
                
                for record in results:
                    neighbors.append((
                        record["neighbor"],
                        record["relation"],
                        float(record["confidence"])
                    ))
            
            # 缓存结果
            self.neighbor_cache[cache_key] = neighbors
            
        except Exception as e:
            logger.error(f"Failed to get neighbors for {entity}: {str(e)}")
        
        return neighbors
    
    async def _post_process_paths(self,
                                paths: List[ReasoningPath],
                                config: PathSearchConfig) -> List[ReasoningPath]:
        """后处理路径"""
        if not paths:
            return paths
        
        # 按置信度排序
        paths.sort(key=lambda p: p.total_confidence, reverse=True)
        
        # 根据路径类型过滤
        if config.path_type == PathType.SHORTEST:
            # 保留最短路径
            min_length = min(p.length for p in paths)
            paths = [p for p in paths if p.length == min_length]
        elif config.path_type == PathType.MOST_CONFIDENT:
            # 已经按置信度排序
            paths = paths
        elif config.path_type == PathType.DIVERSE:
            # 选择多样化路径
            paths = await self._select_diverse_paths(paths, config)
        
        # 限制返回数量
        paths = paths[:config.max_paths]
        
        # 添加执行时间和其他元数据
        for path in paths:
            path.metadata["post_processed"] = True
            path.metadata["total_candidates"] = len(paths)
        
        return paths
    
    async def _select_diverse_paths(self,
                                  paths: List[ReasoningPath],
                                  config: PathSearchConfig) -> List[ReasoningPath]:
        """选择多样化路径"""
        if not paths:
            return paths
        
        diverse_paths = [paths[0]]  # 总是包含最好的路径
        
        for candidate in paths[1:]:
            is_diverse = True
            
            for selected in diverse_paths:
                similarity = self._calculate_path_similarity(candidate, selected)
                if similarity > config.diversity_threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_paths.append(candidate)
                
                if len(diverse_paths) >= config.max_paths:
                    break
        
        return diverse_paths
    
    def _calculate_path_similarity(self, path1: ReasoningPath, path2: ReasoningPath) -> float:
        """计算路径相似度"""
        if path1.length != path2.length:
            return 0.0
        
        common_relations = 0
        total_relations = max(len(path1.path_relations), len(path2.path_relations))
        
        if total_relations == 0:
            return 1.0
        
        for i, rel1 in enumerate(path1.path_relations):
            if i < len(path2.path_relations) and rel1 == path2.path_relations[i]:
                common_relations += 1
        
        return common_relations / total_relations
    
    def _generate_explanation(self, path: ReasoningPath) -> str:
        """生成路径解释"""
        if len(path.nodes) < 2:
            return "Empty path"
        
        explanation_parts = []
        explanation_parts.append(f"从 {path.start_entity} 到 {path.end_entity} 的推理路径:")
        
        for i in range(len(path.nodes) - 1):
            current = path.nodes[i]
            next_node = path.nodes[i + 1]
            
            if next_node.relation:
                confidence_str = f"(置信度: {next_node.confidence:.3f})"
                explanation_parts.append(
                    f"  步骤 {i + 1}: {current.entity} --{next_node.relation}--> {next_node.entity} {confidence_str}"
                )
        
        explanation_parts.append(f"总体置信度: {path.total_confidence:.3f}")
        explanation_parts.append(f"路径长度: {path.length} 跳")
        
        return "\n".join(explanation_parts)
    
    def _generate_cache_key(self, start: str, end: str, constraints: List[str]) -> str:
        """生成缓存键"""
        constraint_str = ":".join(sorted(constraints or []))
        return f"{start}|{end}|{constraint_str}"
    
    def _generate_path_id(self) -> str:
        """生成路径ID"""
        import uuid
        return str(uuid.uuid4())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_searches": self.total_searches,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / self.total_searches if self.total_searches > 0 else 0.0,
            "total_paths_found": self.total_paths_found,
            "average_paths_per_search": self.total_paths_found / self.total_searches if self.total_searches > 0 else 0.0,
            "cache_size": len(self.path_cache),
            "neighbor_cache_size": len(self.neighbor_cache)
        }
    
    def clear_cache(self):
        """清除缓存"""
        self.path_cache.clear()
        self.neighbor_cache.clear()
        logger.info("Path reasoner cache cleared")
    
    async def explain_path(self, path_id: str) -> Optional[Dict[str, Any]]:
        """解释特定路径"""
        for paths in self.path_cache.values():
            for path in paths:
                if path.id == path_id:
                    return {
                        "path_id": path.id,
                        "start_entity": path.start_entity,
                        "end_entity": path.end_entity,
                        "explanation": path.explanation,
                        "confidence": path.total_confidence,
                        "length": path.length,
                        "strategy": path.search_strategy,
                        "triples": path.to_triple_sequence(),
                        "metadata": path.metadata
                    }
        return None
    
    async def find_shortest_path(self, start: str, end: str) -> Optional[ReasoningPath]:
        """查找最短路径"""
        config = PathSearchConfig(
            path_type=PathType.SHORTEST,
            max_paths=1,
            search_strategy=SearchStrategy.BREADTH_FIRST
        )
        
        result = await self.find_reasoning_paths(start, end, config=config)
        return result.paths[0] if result.paths else None
    
    async def find_most_confident_path(self, start: str, end: str) -> Optional[ReasoningPath]:
        """查找最高置信度路径"""
        config = PathSearchConfig(
            path_type=PathType.MOST_CONFIDENT,
            max_paths=1,
            search_strategy=SearchStrategy.BEST_FIRST
        )
        
        result = await self.find_reasoning_paths(start, end, config=config)
        return result.paths[0] if result.paths else None
