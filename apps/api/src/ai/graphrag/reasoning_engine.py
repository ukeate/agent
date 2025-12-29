"""
GraphRAG推理引擎

提供图谱推理能力：
- 多跳推理路径搜索算法
- 推理路径评分和排序机制
- 推理结果的可解释性生成
- 推理缓存和优化策略
"""

import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import heapq
from collections import defaultdict, deque
from .data_models import (
    ReasoningPath,
    QueryDecomposition,
    GraphContext,
    EntityRecognitionResult,
    GraphRAGConfig
)
from ..knowledge_graph.graph_operations import GraphOperations
from ..openai_client import get_openai_client

from src.core.logging import get_logger
logger = get_logger(__name__)

class ReasoningEngine:
    """推理引擎"""
    
    def __init__(self, graph_operations: GraphOperations, config: GraphRAGConfig):
        self.graph_ops = graph_operations
        self.config = config
        self.openai_client = get_openai_client()
        
        # 推理路径评分权重
        self.scoring_weights = {
            'path_length': 0.3,      # 路径长度权重
            'entity_confidence': 0.25, # 实体置信度权重
            'relation_confidence': 0.25, # 关系置信度权重
            'semantic_coherence': 0.2   # 语义连贯性权重
        }

    async def generate_reasoning_paths(
        self,
        decomposition: QueryDecomposition,
        graph_context: GraphContext,
        max_paths: int = 10,
        max_depth: int = 4
    ) -> List[ReasoningPath]:
        """生成推理路径"""
        try:
            logger.info(f"开始生成推理路径，最大路径数: {max_paths}, 最大深度: {max_depth}")
            
            all_paths = []
            
            # 1. 基于实体查询生成路径
            entity_paths = await self._generate_entity_reasoning_paths(
                decomposition.entity_queries, graph_context, max_paths, max_depth
            )
            all_paths.extend(entity_paths)
            
            # 2. 基于关系查询生成路径
            relation_paths = await self._generate_relation_reasoning_paths(
                decomposition.relation_queries, graph_context, max_paths, max_depth
            )
            all_paths.extend(relation_paths)
            
            # 3. 基于复杂推理生成路径
            if decomposition.decomposition_strategy == "complex_reasoning":
                complex_paths = await self._generate_complex_reasoning_paths(
                    decomposition, graph_context, max_paths, max_depth
                )
                all_paths.extend(complex_paths)
            
            # 4. 路径评分和排序
            scored_paths = await self._score_reasoning_paths(all_paths, decomposition)
            
            # 5. 选择最佳路径
            top_paths = scored_paths[:max_paths]
            
            # 6. 生成解释
            for path in top_paths:
                path.explanation = await self._generate_path_explanation(path)
            
            logger.info(f"生成了{len(top_paths)}条推理路径")
            return top_paths
            
        except Exception as e:
            logger.error(f"推理路径生成失败: {e}")
            return []

    async def _generate_entity_reasoning_paths(
        self,
        entity_queries: List[Dict[str, Any]],
        graph_context: GraphContext,
        max_paths: int,
        max_depth: int
    ) -> List[ReasoningPath]:
        """基于实体查询生成推理路径"""
        paths = []
        
        try:
            for entity_query in entity_queries[:5]:  # 限制处理的实体数量
                entity_name = entity_query.get('entity', '')
                if not entity_name:
                    continue
                
                # 寻找该实体在图谱上下文中的邻居路径
                entity_paths = await self._find_entity_neighborhood_paths(
                    entity_name, graph_context, max_depth
                )
                paths.extend(entity_paths)
                
                if len(paths) >= max_paths:
                    break
            
            return paths[:max_paths]
            
        except Exception as e:
            logger.error(f"实体推理路径生成失败: {e}")
            return []

    async def _generate_relation_reasoning_paths(
        self,
        relation_queries: List[Dict[str, Any]],
        graph_context: GraphContext,
        max_paths: int,
        max_depth: int
    ) -> List[ReasoningPath]:
        """基于关系查询生成推理路径"""
        paths = []
        
        try:
            for relation_query in relation_queries[:5]:  # 限制处理的关系数量
                entity1 = relation_query.get('entity1', '')
                entity2 = relation_query.get('entity2', '')
                max_hops = min(relation_query.get('max_hops', 3), max_depth)
                
                if not entity1 or not entity2:
                    continue
                
                # 寻找两个实体间的路径
                connecting_paths = await self._find_connecting_paths(
                    entity1, entity2, graph_context, max_hops
                )
                paths.extend(connecting_paths)
                
                if len(paths) >= max_paths:
                    break
            
            return paths[:max_paths]
            
        except Exception as e:
            logger.error(f"关系推理路径生成失败: {e}")
            return []

    async def _generate_complex_reasoning_paths(
        self,
        decomposition: QueryDecomposition,
        graph_context: GraphContext,
        max_paths: int,
        max_depth: int
    ) -> List[ReasoningPath]:
        """基于复杂推理生成路径"""
        paths = []
        
        try:
            # 结合实体和关系查询进行复杂推理
            entities = [eq.get('entity', '') for eq in decomposition.entity_queries]
            
            # 生成多实体间的推理路径
            if len(entities) >= 2:
                multi_paths = await self._generate_multi_entity_paths(
                    entities, graph_context, max_depth
                )
                paths.extend(multi_paths)
            
            # 基于子查询进行推理
            for sub_query in decomposition.sub_queries:
                query_paths = await self._generate_query_specific_paths(
                    sub_query, entities, graph_context, max_depth
                )
                paths.extend(query_paths)
            
            return paths[:max_paths]
            
        except Exception as e:
            logger.error(f"复杂推理路径生成失败: {e}")
            return []

    async def _find_entity_neighborhood_paths(
        self,
        entity_name: str,
        graph_context: GraphContext,
        max_depth: int
    ) -> List[ReasoningPath]:
        """寻找实体邻域路径"""
        paths = []
        
        try:
            # 从图谱上下文中找到实体
            target_entity = None
            for entity in graph_context.entities:
                if (entity.get('canonical_form') == entity_name or 
                    entity.get('name') == entity_name):
                    target_entity = entity
                    break
            
            if not target_entity:
                return []
            
            # 使用广度优先搜索找到邻域路径
            visited = set()
            queue = deque([(target_entity, [], [])])
            
            while queue and len(paths) < 10:
                current_entity, path_entities, path_relations = queue.popleft()
                current_id = current_entity.get('id', current_entity.get('canonical_form'))
                
                if current_id in visited or len(path_entities) >= max_depth:
                    continue
                
                visited.add(current_id)
                
                # 如果路径长度大于0，创建推理路径
                if path_entities:
                    reasoning_path = ReasoningPath(
                        path_id=str(uuid.uuid4()),
                        entities=path_entities + [current_entity.get('canonical_form', current_id)],
                        relations=path_relations,
                        path_score=0.0,  # 稍后计算
                        explanation="",  # 稍后生成
                        evidence=[],
                        hops_count=len(path_entities)
                    )
                    paths.append(reasoning_path)
                
                # 找到相关的关系和实体
                for relation in graph_context.relations:
                    if self._is_entity_in_relation(current_entity, relation):
                        related_entity = self._get_related_entity(current_entity, relation)
                        if related_entity:
                            new_path_entities = path_entities + [current_entity.get('canonical_form', current_id)]
                            new_path_relations = path_relations + [relation.get('type', 'RELATED')]
                            queue.append((related_entity, new_path_entities, new_path_relations))
            
            return paths
            
        except Exception as e:
            logger.error(f"实体邻域路径搜索失败: {e}")
            return []

    async def _find_connecting_paths(
        self,
        entity1: str,
        entity2: str,
        graph_context: GraphContext,
        max_hops: int
    ) -> List[ReasoningPath]:
        """寻找连接两个实体的路径"""
        paths = []
        
        try:
            # 找到起始和目标实体
            start_entity = None
            end_entity = None
            
            for entity in graph_context.entities:
                canonical_form = entity.get('canonical_form', entity.get('name', ''))
                if canonical_form == entity1:
                    start_entity = entity
                elif canonical_form == entity2:
                    end_entity = entity
            
            if not start_entity or not end_entity:
                return []
            
            # 使用双向BFS寻找最短路径
            paths_found = await self._bidirectional_search(
                start_entity, end_entity, graph_context, max_hops
            )
            
            for path_data in paths_found:
                reasoning_path = ReasoningPath(
                    path_id=str(uuid.uuid4()),
                    entities=path_data['entities'],
                    relations=path_data['relations'],
                    path_score=0.0,  # 稍后计算
                    explanation="",  # 稍后生成
                    evidence=path_data.get('evidence', []),
                    hops_count=len(path_data['relations'])
                )
                paths.append(reasoning_path)
            
            return paths
            
        except Exception as e:
            logger.error(f"连接路径搜索失败: {e}")
            return []

    async def _bidirectional_search(
        self,
        start_entity: Dict[str, Any],
        end_entity: Dict[str, Any],
        graph_context: GraphContext,
        max_hops: int
    ) -> List[Dict[str, Any]]:
        """双向搜索算法"""
        paths = []
        
        try:
            start_id = start_entity.get('id', start_entity.get('canonical_form'))
            end_id = end_entity.get('id', end_entity.get('canonical_form'))
            
            # 正向搜索队列
            forward_queue = deque([(start_entity, [start_id], [])])
            forward_visited = {start_id: ([start_id], [])}
            
            # 反向搜索队列
            backward_queue = deque([(end_entity, [end_id], [])])
            backward_visited = {end_id: ([end_id], [])}
            
            max_depth = max_hops // 2 + 1
            
            for depth in range(max_depth):
                # 正向扩展
                if forward_queue:
                    forward_queue = await self._expand_search_frontier(
                        forward_queue, forward_visited, graph_context, depth < max_depth - 1
                    )
                
                # 反向扩展
                if backward_queue:
                    backward_queue = await self._expand_search_frontier(
                        backward_queue, backward_visited, graph_context, depth < max_depth - 1
                    )
                
                # 检查是否有路径相遇
                meeting_points = set(forward_visited.keys()) & set(backward_visited.keys())
                for meeting_point in meeting_points:
                    if meeting_point != start_id and meeting_point != end_id:
                        # 构建完整路径
                        forward_path = forward_visited[meeting_point]
                        backward_path = backward_visited[meeting_point]
                        
                        # 合并路径（反向路径需要反转）
                        full_entities = forward_path[0] + backward_path[0][::-1][1:]
                        full_relations = forward_path[1] + backward_path[1][::-1]
                        
                        paths.append({
                            'entities': full_entities,
                            'relations': full_relations,
                            'evidence': []
                        })
                
                if paths or (not forward_queue and not backward_queue):
                    break
            
            return paths[:5]  # 返回前5条路径
            
        except Exception as e:
            logger.error(f"双向搜索失败: {e}")
            return []

    async def _expand_search_frontier(
        self,
        queue: deque,
        visited: Dict[str, Tuple[List[str], List[str]]],
        graph_context: GraphContext,
        should_continue: bool
    ) -> deque:
        """扩展搜索前沿"""
        new_queue = deque()
        
        try:
            while queue:
                current_entity, path_entities, path_relations = queue.popleft()
                current_id = current_entity.get('id', current_entity.get('canonical_form'))
                
                if not should_continue:
                    continue
                
                # 寻找相邻实体
                for relation in graph_context.relations:
                    if self._is_entity_in_relation(current_entity, relation):
                        related_entity = self._get_related_entity(current_entity, relation)
                        if related_entity:
                            related_id = related_entity.get('id', related_entity.get('canonical_form'))
                            
                            if related_id not in visited:
                                new_path_entities = path_entities + [related_id]
                                new_path_relations = path_relations + [relation.get('type', 'RELATED')]
                                
                                visited[related_id] = (new_path_entities, new_path_relations)
                                new_queue.append((related_entity, new_path_entities, new_path_relations))
            
            return new_queue
            
        except Exception as e:
            logger.error(f"搜索前沿扩展失败: {e}")
            return deque()

    async def _generate_multi_entity_paths(
        self,
        entities: List[str],
        graph_context: GraphContext,
        max_depth: int
    ) -> List[ReasoningPath]:
        """生成多实体推理路径"""
        paths = []
        
        try:
            # 为所有实体对生成连接路径
            for i, entity1 in enumerate(entities[:4]):
                for entity2 in entities[i+1:5]:
                    connecting_paths = await self._find_connecting_paths(
                        entity1, entity2, graph_context, max_depth
                    )
                    paths.extend(connecting_paths)
            
            return paths
            
        except Exception as e:
            logger.error(f"多实体路径生成失败: {e}")
            return []

    async def _generate_query_specific_paths(
        self,
        sub_query: str,
        entities: List[str],
        graph_context: GraphContext,
        max_depth: int
    ) -> List[ReasoningPath]:
        """基于子查询生成特定路径"""
        paths = []
        
        try:
            # 根据子查询类型生成不同的路径
            query_lower = sub_query.lower()
            
            if any(word in query_lower for word in ['cause', 'reason', 'why']):
                # 因果推理路径
                paths = await self._generate_causal_paths(entities, graph_context, max_depth)
            elif any(word in query_lower for word in ['compare', 'similar', 'different']):
                # 比较推理路径  
                paths = await self._generate_comparison_paths(entities, graph_context, max_depth)
            elif any(word in query_lower for word in ['timeline', 'sequence', 'order']):
                # 时序推理路径
                paths = await self._generate_temporal_paths(entities, graph_context, max_depth)
            
            return paths
            
        except Exception as e:
            logger.error(f"查询特定路径生成失败: {e}")
            return []

    async def _generate_causal_paths(
        self,
        entities: List[str],
        graph_context: GraphContext,
        max_depth: int
    ) -> List[ReasoningPath]:
        """生成因果推理路径"""
        paths = []
        
        try:
            causal_relations = ['CAUSES', 'LEADS_TO', 'RESULTS_IN', 'TRIGGERS', 'INFLUENCES']
            
            for relation in graph_context.relations:
                rel_type = relation.get('type', '')
                if rel_type in causal_relations:
                    # 构建因果路径
                    path = ReasoningPath(
                        path_id=str(uuid.uuid4()),
                        entities=[
                            relation.get('source', ''),
                            relation.get('target', '')
                        ],
                        relations=[rel_type],
                        path_score=0.8,  # 因果关系通常重要性较高
                        explanation="",
                        evidence=[relation],
                        hops_count=1
                    )
                    paths.append(path)
            
            return paths
            
        except Exception as e:
            logger.error(f"因果路径生成失败: {e}")
            return []

    async def _generate_comparison_paths(
        self,
        entities: List[str],
        graph_context: GraphContext,
        max_depth: int
    ) -> List[ReasoningPath]:
        """生成比较推理路径"""
        paths = []
        
        try:
            comparison_relations = ['SIMILAR_TO', 'DIFFERENT_FROM', 'COMPARES_WITH', 'CONTRASTS_WITH']
            
            for relation in graph_context.relations:
                rel_type = relation.get('type', '')
                if rel_type in comparison_relations:
                    path = ReasoningPath(
                        path_id=str(uuid.uuid4()),
                        entities=[
                            relation.get('source', ''),
                            relation.get('target', '')
                        ],
                        relations=[rel_type],
                        path_score=0.7,
                        explanation="",
                        evidence=[relation],
                        hops_count=1
                    )
                    paths.append(path)
            
            return paths
            
        except Exception as e:
            logger.error(f"比较路径生成失败: {e}")
            return []

    async def _generate_temporal_paths(
        self,
        entities: List[str],
        graph_context: GraphContext,
        max_depth: int
    ) -> List[ReasoningPath]:
        """生成时序推理路径"""
        paths = []
        
        try:
            temporal_relations = ['BEFORE', 'AFTER', 'DURING', 'FOLLOWS', 'PRECEDES']
            
            for relation in graph_context.relations:
                rel_type = relation.get('type', '')
                if rel_type in temporal_relations:
                    path = ReasoningPath(
                        path_id=str(uuid.uuid4()),
                        entities=[
                            relation.get('source', ''),
                            relation.get('target', '')
                        ],
                        relations=[rel_type],
                        path_score=0.6,
                        explanation="",
                        evidence=[relation],
                        hops_count=1
                    )
                    paths.append(path)
            
            return paths
            
        except Exception as e:
            logger.error(f"时序路径生成失败: {e}")
            return []

    def _is_entity_in_relation(
        self,
        entity: Dict[str, Any],
        relation: Dict[str, Any]
    ) -> bool:
        """检查实体是否在关系中"""
        entity_id = entity.get('id', entity.get('canonical_form'))
        
        # 处理不同的关系格式
        if 'e1' in relation and 'e2' in relation:
            # Neo4j格式
            e1_id = relation['e1'].get('id', relation['e1'].get('canonical_form'))
            e2_id = relation['e2'].get('id', relation['e2'].get('canonical_form'))
            return entity_id == e1_id or entity_id == e2_id
        elif 'source' in relation and 'target' in relation:
            # 简化格式
            return (entity_id == relation['source'] or 
                   entity_id == relation['target'])
        
        return False

    def _get_related_entity(
        self,
        entity: Dict[str, Any],
        relation: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """获取关系中的相关实体"""
        entity_id = entity.get('id', entity.get('canonical_form'))
        
        # 处理不同的关系格式
        if 'e1' in relation and 'e2' in relation:
            e1_id = relation['e1'].get('id', relation['e1'].get('canonical_form'))
            e2_id = relation['e2'].get('id', relation['e2'].get('canonical_form'))
            
            if entity_id == e1_id:
                return relation['e2']
            elif entity_id == e2_id:
                return relation['e1']
        elif 'source' in relation and 'target' in relation:
            if entity_id == relation['source']:
                return {'canonical_form': relation['target']}
            elif entity_id == relation['target']:
                return {'canonical_form': relation['source']}
        
        return None

    async def _score_reasoning_paths(
        self,
        paths: List[ReasoningPath],
        decomposition: QueryDecomposition
    ) -> List[ReasoningPath]:
        """为推理路径评分"""
        try:
            for path in paths:
                score = await self._calculate_path_score(path, decomposition)
                path.path_score = score
            
            # 按评分排序
            paths.sort(key=lambda p: p.path_score, reverse=True)
            return paths
            
        except Exception as e:
            logger.error(f"推理路径评分失败: {e}")
            return paths

    async def _calculate_path_score(
        self,
        path: ReasoningPath,
        decomposition: QueryDecomposition
    ) -> float:
        """计算路径评分"""
        try:
            score = 0.0
            
            # 路径长度评分（较短路径得分更高）
            length_score = 1.0 / (1.0 + path.hops_count * 0.2)
            score += length_score * self.scoring_weights['path_length']
            
            # 实体置信度评分（如果有的话）
            entity_confidence = 0.8  # 默认置信度
            score += entity_confidence * self.scoring_weights['entity_confidence']
            
            # 关系置信度评分
            relation_confidence = 0.8  # 默认置信度
            score += relation_confidence * self.scoring_weights['relation_confidence']
            
            # 语义连贯性评分
            coherence_score = await self._calculate_semantic_coherence(path, decomposition)
            score += coherence_score * self.scoring_weights['semantic_coherence']
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"路径评分计算失败: {e}")
            return 0.5

    async def _calculate_semantic_coherence(
        self,
        path: ReasoningPath,
        decomposition: QueryDecomposition
    ) -> float:
        """计算语义连贯性"""
        try:
            # 检查路径实体是否与查询相关
            query_entities = set()
            for eq in decomposition.entity_queries:
                query_entities.add(eq.get('entity', '').lower())
            
            path_entities = set(entity.lower() for entity in path.entities)
            overlap = len(query_entities.intersection(path_entities))
            
            if len(query_entities) > 0:
                relevance_score = overlap / len(query_entities)
            else:
                relevance_score = 0.5
            
            # 检查路径关系的语义连贯性
            coherent_relations = ['IS_A', 'PART_OF', 'CAUSES', 'LEADS_TO', 'SIMILAR_TO']
            relation_coherence = sum(
                1 for relation in path.relations 
                if relation in coherent_relations
            ) / max(1, len(path.relations))
            
            return (relevance_score + relation_coherence) / 2.0
            
        except Exception as e:
            logger.error(f"语义连贯性计算失败: {e}")
            return 0.5

    async def _generate_path_explanation(self, path: ReasoningPath) -> str:
        """生成推理路径解释"""
        try:
            if len(path.entities) < 2 or len(path.relations) < 1:
                return f"Simple path involving {', '.join(path.entities)}"
            
            explanation_parts = []
            
            for i, relation in enumerate(path.relations):
                if i < len(path.entities) - 1:
                    source_entity = path.entities[i]
                    target_entity = path.entities[i + 1]
                    
                    # 根据关系类型生成自然语言描述
                    if relation in ['IS_A', 'TYPE_OF']:
                        explanation_parts.append(f"{source_entity} is a type of {target_entity}")
                    elif relation in ['PART_OF', 'BELONGS_TO']:
                        explanation_parts.append(f"{source_entity} is part of {target_entity}")
                    elif relation in ['CAUSES', 'LEADS_TO']:
                        explanation_parts.append(f"{source_entity} causes {target_entity}")
                    elif relation in ['LOCATED_IN', 'AT']:
                        explanation_parts.append(f"{source_entity} is located in {target_entity}")
                    else:
                        relation_desc = relation.lower().replace('_', ' ')
                        explanation_parts.append(f"{source_entity} {relation_desc} {target_entity}")
            
            explanation = ". ".join(explanation_parts) + "."
            
            # 添加路径统计信息
            if path.hops_count > 1:
                explanation += f" This reasoning path involves {path.hops_count} steps."
            
            return explanation
            
        except Exception as e:
            logger.error(f"路径解释生成失败: {e}")
            return f"Reasoning path connecting {', '.join(path.entities)} through {len(path.relations)} relationships."

    async def explain_reasoning_result(
        self,
        paths: List[ReasoningPath],
        query: str
    ) -> str:
        """解释推理结果"""
        try:
            if not paths:
                return "No reasoning paths found for the given query."
            
            explanation = f"Found {len(paths)} reasoning paths for the query: '{query}'\n\n"
            
            for i, path in enumerate(paths[:3], 1):  # 只解释前3条路径
                explanation += f"Path {i} (Score: {path.path_score:.2f}):\n"
                explanation += f"  {path.explanation}\n"
                
                if path.evidence:
                    explanation += f"  Evidence: {len(path.evidence)} supporting facts\n"
                
                explanation += "\n"
            
            return explanation
            
        except Exception as e:
            logger.error(f"推理结果解释失败: {e}")
            return "Unable to explain reasoning results."
