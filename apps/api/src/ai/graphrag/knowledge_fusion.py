"""
GraphRAG知识融合处理器

提供多源知识融合能力：
- 多源知识的置信度计算
- 冲突检测和解决算法  
- 知识一致性验证机制
- 融合结果的可解释性支持
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
import numpy as np
from .data_models import (
    KnowledgeSource, 
    GraphContext, 
    ReasoningPath,
    FusionResult,
    GraphRAGConfig
)
from ..openai_client import get_openai_client

from src.core.logging import get_logger
logger = get_logger(__name__)

class KnowledgeFusion:
    """知识融合引擎"""
    
    def __init__(self, config: GraphRAGConfig):
        self.config = config
        self.openai_client = get_openai_client()
        
        # 融合策略映射
        self.fusion_strategies = {
            'weighted_average': self._weighted_average_fusion,
            'rank_aggregation': self._rank_aggregation_fusion,
            'evidence_based': self._evidence_based_fusion,
            'confidence_threshold': self._confidence_threshold_fusion,
            'weighted_evidence': self._weighted_evidence_fusion
        }
        
        # 冲突解决策略
        self.conflict_resolution_strategies = {
            'highest_confidence': self._resolve_by_highest_confidence,
            'majority_vote': self._resolve_by_majority_vote,
            'source_priority': self._resolve_by_source_priority,
            'evidence_weight': self._resolve_by_evidence_weight
        }

    async def fuse_knowledge_sources(
        self,
        retrieval_results: Dict[str, Any],
        graph_context: GraphContext,
        reasoning_results: List[ReasoningPath],
        confidence_threshold: float = 0.6,
        max_sources: int = 50
    ) -> FusionResult:
        """融合多源知识"""
        start_time = utc_now()
        
        try:
            logger.info("开始知识融合处理")
            
            # 1. 准备知识源
            knowledge_sources = await self._prepare_knowledge_sources(
                retrieval_results, graph_context, reasoning_results, max_sources
            )
            
            logger.info(f"准备了{len(knowledge_sources)}个知识源")
            
            # 2. 冲突检测
            conflicts = await self._detect_conflicts(knowledge_sources)
            
            # 3. 冲突解决
            resolved_sources = knowledge_sources
            if conflicts and self.config.enable_conflict_resolution:
                resolved_sources = await self._resolve_conflicts(
                    knowledge_sources, conflicts
                )
            
            # 4. 知识融合和排序
            fused_results = await self._fuse_and_rank(
                resolved_sources, confidence_threshold
            )
            
            # 5. 一致性检查
            consistency_score = await self._check_consistency(fused_results)
            
            # 6. 构建融合结果
            fusion_time = (utc_now() - start_time).total_seconds()
            
            result = FusionResult(
                final_ranking=fused_results['documents'],
                confidence_scores=fused_results.get('confidence_scores', {}),
                conflicts_detected=conflicts,
                resolution_strategy=self.config.fusion_strategy,
                consistency_score=consistency_score
            )
            
            logger.info(f"知识融合完成，融合时间: {fusion_time:.2f}秒，一致性评分: {consistency_score:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"知识融合失败: {e}")
            # 返回基础融合结果
            return FusionResult(
                final_ranking=[],
                confidence_scores={},
                conflicts_detected=[],
                resolution_strategy="fallback",
                consistency_score=0.0
            )

    async def _prepare_knowledge_sources(
        self,
        retrieval_results: Dict[str, Any],
        graph_context: GraphContext,
        reasoning_results: List[ReasoningPath],
        max_sources: int
    ) -> List[KnowledgeSource]:
        """准备知识源"""
        sources = []
        
        try:
            # 1. 处理向量检索结果
            if 'vector' in retrieval_results:
                vector_results = retrieval_results['vector']
                for i, doc in enumerate(vector_results[:max_sources//3]):
                    # 确保文档有必要的字段
                    content = doc.get('content', '') or doc.get('page_content', '')
                    if not content:
                        continue
                    
                    source = KnowledgeSource(
                        source_type='vector',
                        content=content,
                        confidence=doc.get('score', 0.5),
                        metadata={
                            'document_id': doc.get('id', f'vector_{i}'),
                            'distance': doc.get('distance', 1.0),
                            'source': doc.get('source', 'vector_store'),
                            **doc.get('metadata', {})
                        }
                    )
                    sources.append(source)
            
            # 2. 处理图谱事实
            for i, relation in enumerate(graph_context.relations[:max_sources//3]):
                try:
                    # 处理不同格式的关系数据
                    if isinstance(relation, dict):
                        if 'r' in relation and 'e1' in relation and 'e2' in relation:
                            # Neo4j格式
                            fact_content = self._format_neo4j_relation_as_text(relation)
                            confidence = relation['r'].get('confidence', 0.8)
                            relation_type = relation['r'].get('type', 'UNKNOWN')
                        else:
                            # 简化格式
                            fact_content = self._format_simple_relation_as_text(relation)
                            confidence = relation.get('confidence', 0.8)
                            relation_type = relation.get('type', 'UNKNOWN')
                    else:
                        # 字符串格式
                        fact_content = str(relation)
                        confidence = 0.7
                        relation_type = 'UNKNOWN'
                    
                    if fact_content:
                        source = KnowledgeSource(
                            source_type='graph',
                            content=fact_content,
                            confidence=confidence,
                            metadata={
                                'relation_id': f'graph_{i}',
                                'relation_type': relation_type,
                                'source': 'knowledge_graph'
                            },
                            graph_context=graph_context
                        )
                        sources.append(source)
                        
                except Exception as e:
                    logger.warning(f"处理图谱关系失败: {e}")
                    continue
            
            # 3. 处理推理结果
            for reasoning in reasoning_results[:max_sources//3]:
                if reasoning.explanation:
                    source = KnowledgeSource(
                        source_type='reasoning',
                        content=reasoning.explanation,
                        confidence=reasoning.path_score,
                        metadata={
                            'reasoning_path_id': reasoning.path_id,
                            'hops_count': reasoning.hops_count,
                            'entities': reasoning.entities,
                            'relations': reasoning.relations,
                            'evidence_count': len(reasoning.evidence),
                            'source': 'reasoning_engine'
                        }
                    )
                    sources.append(source)
            
            logger.info(f"成功准备了{len(sources)}个知识源")
            return sources
            
        except Exception as e:
            logger.error(f"准备知识源失败: {e}")
            return sources

    def _format_neo4j_relation_as_text(self, relation: Dict[str, Any]) -> str:
        """将Neo4j格式的关系转换为文本"""
        try:
            entity1 = relation['e1'].get('canonical_form', 'Entity1')
            entity2 = relation['e2'].get('canonical_form', 'Entity2')
            relation_type = relation['r'].get('type', 'RELATED_TO')
            
            # 根据关系类型生成自然语言描述
            if relation_type in ['IS_A', 'TYPE_OF']:
                return f"{entity1} is a type of {entity2}."
            elif relation_type in ['PART_OF', 'BELONGS_TO']:
                return f"{entity1} is part of {entity2}."
            elif relation_type in ['LOCATED_IN', 'AT']:
                return f"{entity1} is located in {entity2}."
            elif relation_type in ['WORKS_AT', 'EMPLOYED_BY']:
                return f"{entity1} works at {entity2}."
            elif relation_type in ['FOUNDED', 'CREATED']:
                return f"{entity1} founded {entity2}."
            else:
                return f"{entity1} is related to {entity2} through {relation_type.lower().replace('_', ' ')}."
                
        except Exception as e:
            logger.warning(f"格式化Neo4j关系失败: {e}")
            return f"Relationship: {relation}"

    def _format_simple_relation_as_text(self, relation: Dict[str, Any]) -> str:
        """将简化格式的关系转换为文本"""
        try:
            source = relation.get('source', 'Entity1')
            target = relation.get('target', 'Entity2')
            rel_type = relation.get('type', 'RELATED_TO')
            
            return f"{source} {rel_type.lower().replace('_', ' ')} {target}."
        except Exception:
            return str(relation)

    async def _detect_conflicts(
        self, 
        knowledge_sources: List[KnowledgeSource]
    ) -> List[Dict[str, Any]]:
        """检测知识冲突"""
        conflicts = []
        
        try:
            # 1. 基于内容相似度的冲突检测
            conflicts.extend(await self._detect_content_conflicts(knowledge_sources))
            
            # 2. 基于事实冲突的检测
            conflicts.extend(await self._detect_factual_conflicts(knowledge_sources))
            
            # 3. 基于置信度差异的冲突检测
            conflicts.extend(await self._detect_confidence_conflicts(knowledge_sources))
            
            logger.info(f"检测到{len(conflicts)}个知识冲突")
            return conflicts
            
        except Exception as e:
            logger.error(f"冲突检测失败: {e}")
            return []

    async def _detect_content_conflicts(
        self,
        sources: List[KnowledgeSource]
    ) -> List[Dict[str, Any]]:
        """检测内容冲突"""
        conflicts = []
        
        for i, source1 in enumerate(sources):
            for j, source2 in enumerate(sources[i+1:], i+1):
                try:
                    # 计算内容相似度
                    similarity = await self._calculate_content_similarity(
                        source1.content, source2.content
                    )
                    
                    # 如果内容相似但置信度差异较大，或者来源不同但内容矛盾
                    confidence_diff = abs(source1.confidence - source2.confidence)
                    
                    if similarity > 0.8 and confidence_diff > 0.3:
                        conflict = {
                            'type': 'confidence_conflict',
                            'sources': [i, j],
                            'similarity': similarity,
                            'confidence_diff': confidence_diff,
                            'description': f"相似内容但置信度差异较大: {confidence_diff:.2f}"
                        }
                        conflicts.append(conflict)
                    elif similarity > 0.6 and self._detect_contradiction(source1.content, source2.content):
                        conflict = {
                            'type': 'content_contradiction',
                            'sources': [i, j],
                            'similarity': similarity,
                            'description': "相似内容但存在矛盾"
                        }
                        conflicts.append(conflict)
                        
                except Exception as e:
                    logger.warning(f"内容冲突检测失败: {e}")
                    continue
        
        return conflicts

    async def _detect_factual_conflicts(
        self,
        sources: List[KnowledgeSource]
    ) -> List[Dict[str, Any]]:
        """检测事实冲突"""
        conflicts = []
        
        # 提取关键事实声明
        factual_claims = {}
        
        for i, source in enumerate(sources):
            try:
                # 简单的事实提取 - 寻找"X is Y"模式
                claims = self._extract_factual_claims(source.content)
                for claim in claims:
                    if claim not in factual_claims:
                        factual_claims[claim] = []
                    factual_claims[claim].append((i, source))
            except Exception as e:
                logger.warning(f"事实提取失败: {e}")
                continue
        
        # 检测矛盾的事实声明
        for claim, source_list in factual_claims.items():
            if len(source_list) > 1:
                # 检查是否有矛盾的声明
                confidences = [source.confidence for _, source in source_list]
                if max(confidences) - min(confidences) > 0.4:
                    conflict = {
                        'type': 'factual_conflict',
                        'claim': claim,
                        'sources': [i for i, _ in source_list],
                        'confidences': confidences,
                        'description': f"关于'{claim}'的事实声明存在冲突"
                    }
                    conflicts.append(conflict)
        
        return conflicts

    async def _detect_confidence_conflicts(
        self,
        sources: List[KnowledgeSource]
    ) -> List[Dict[str, Any]]:
        """检测置信度冲突"""
        conflicts = []
        
        # 按源类型分组
        source_groups = {}
        for i, source in enumerate(sources):
            if source.source_type not in source_groups:
                source_groups[source.source_type] = []
            source_groups[source.source_type].append((i, source))
        
        # 检测同类型源之间的置信度异常
        for source_type, source_list in source_groups.items():
            if len(source_list) > 2:
                confidences = [source.confidence for _, source in source_list]
                mean_conf = np.mean(confidences)
                std_conf = np.std(confidences)
                
                # 找出异常的置信度
                for i, (idx, source) in enumerate(source_list):
                    if abs(source.confidence - mean_conf) > 2 * std_conf:
                        conflict = {
                            'type': 'confidence_anomaly',
                            'source_index': idx,
                            'source_type': source_type,
                            'confidence': source.confidence,
                            'expected_range': [mean_conf - std_conf, mean_conf + std_conf],
                            'description': f"{source_type}源的置信度异常"
                        }
                        conflicts.append(conflict)
        
        return conflicts

    def _extract_factual_claims(self, content: str) -> List[str]:
        """提取事实声明"""
        import re
        claims = []
        
        # 简单的模式匹配
        patterns = [
            r'(\w+(?:\s+\w+)*)\s+is\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+was\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+are\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+were\s+(\w+(?:\s+\w+)*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                claim = f"{match[0]} is {match[1]}".lower()
                claims.append(claim)
        
        return claims

    def _detect_contradiction(self, content1: str, content2: str) -> bool:
        """检测内容矛盾"""
        # 简单的否定词检测
        negation_words = ['not', 'no', 'never', 'none', 'nothing', 'neither', 'nor']
        
        content1_lower = content1.lower()
        content2_lower = content2.lower()
        
        # 检查一个内容是否包含否定形式
        has_negation_1 = any(word in content1_lower for word in negation_words)
        has_negation_2 = any(word in content2_lower for word in negation_words)
        
        # 如果一个有否定一个没有，可能存在矛盾
        return has_negation_1 != has_negation_2

    async def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似度"""
        try:
            # 简单的词汇重叠相似度
            words1 = set(content1.lower().split())
            words2 = set(content2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            jaccard_similarity = intersection / union if union > 0 else 0.0
            
            # 也可以使用余弦相似度或其他更复杂的方法
            return jaccard_similarity
            
        except Exception as e:
            logger.warning(f"相似度计算失败: {e}")
            return 0.0

    async def _resolve_conflicts(
        self,
        knowledge_sources: List[KnowledgeSource],
        conflicts: List[Dict[str, Any]]
    ) -> List[KnowledgeSource]:
        """解决知识冲突"""
        try:
            resolved_sources = knowledge_sources.copy()
            
            # 按冲突类型分组处理
            for conflict in conflicts:
                conflict_type = conflict['type']
                
                if conflict_type in ['confidence_conflict', 'factual_conflict']:
                    # 使用置信度解决策略
                    resolved_sources = await self.conflict_resolution_strategies['highest_confidence'](
                        resolved_sources, conflict
                    )
                elif conflict_type == 'content_contradiction':
                    # 使用证据权重解决策略
                    resolved_sources = await self.conflict_resolution_strategies['evidence_weight'](
                        resolved_sources, conflict
                    )
                elif conflict_type == 'confidence_anomaly':
                    # 调整异常的置信度
                    source_idx = conflict['source_index']
                    if source_idx < len(resolved_sources):
                        expected_conf = np.mean(conflict['expected_range'])
                        resolved_sources[source_idx].confidence = expected_conf
            
            logger.info(f"冲突解决完成，处理了{len(conflicts)}个冲突")
            return resolved_sources
            
        except Exception as e:
            logger.error(f"冲突解决失败: {e}")
            return knowledge_sources

    async def _resolve_by_highest_confidence(
        self,
        sources: List[KnowledgeSource],
        conflict: Dict[str, Any]
    ) -> List[KnowledgeSource]:
        """通过最高置信度解决冲突"""
        source_indices = conflict.get('sources', [])
        if not source_indices:
            return sources
        
        # 找到置信度最高的源
        max_confidence = -1
        keep_index = -1
        
        for idx in source_indices:
            if idx < len(sources) and sources[idx].confidence > max_confidence:
                max_confidence = sources[idx].confidence
                keep_index = idx
        
        # 保留最高置信度的源，降低其他源的置信度
        for idx in source_indices:
            if idx < len(sources) and idx != keep_index:
                sources[idx].confidence *= 0.7  # 降低置信度
        
        return sources

    async def _resolve_by_majority_vote(
        self,
        sources: List[KnowledgeSource],
        conflict: Dict[str, Any]
    ) -> List[KnowledgeSource]:
        """通过多数投票解决冲突"""
        # 这里可以实现基于多数投票的冲突解决
        return sources

    async def _resolve_by_source_priority(
        self,
        sources: List[KnowledgeSource],
        conflict: Dict[str, Any]
    ) -> List[KnowledgeSource]:
        """通过源优先级解决冲突"""
        # 定义源类型优先级
        source_priority = {
            'graph': 0.9,
            'reasoning': 0.8,
            'vector': 0.7
        }
        
        source_indices = conflict.get('sources', [])
        
        # 根据源类型调整置信度
        for idx in source_indices:
            if idx < len(sources):
                source_type = sources[idx].source_type
                priority = source_priority.get(source_type, 0.5)
                sources[idx].confidence *= priority
        
        return sources

    async def _resolve_by_evidence_weight(
        self,
        sources: List[KnowledgeSource],
        conflict: Dict[str, Any]
    ) -> List[KnowledgeSource]:
        """通过证据权重解决冲突"""
        source_indices = conflict.get('sources', [])
        
        # 计算每个源的证据权重
        for idx in source_indices:
            if idx < len(sources):
                source = sources[idx]
                evidence_weight = 1.0
                
                # 基于元数据计算证据权重
                if source.source_type == 'reasoning':
                    evidence_count = source.metadata.get('evidence_count', 1)
                    evidence_weight = min(1.0, evidence_count / 3.0)
                elif source.source_type == 'graph':
                    # 图谱源的证据权重基于关系类型
                    relation_type = source.metadata.get('relation_type', '')
                    if relation_type in ['IS_A', 'TYPE_OF']:
                        evidence_weight = 0.9
                    elif relation_type in ['RELATED_TO']:
                        evidence_weight = 0.6
                
                sources[idx].confidence *= evidence_weight
        
        return sources

    async def _fuse_and_rank(
        self,
        knowledge_sources: List[KnowledgeSource],
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """融合和排序知识源"""
        try:
            # 过滤低置信度源
            filtered_sources = [
                source for source in knowledge_sources 
                if source.confidence >= confidence_threshold
            ]
            
            logger.info(f"应用置信度阈值{confidence_threshold}后，保留{len(filtered_sources)}个源")
            
            # 选择融合策略
            strategy = self.config.fusion_strategy
            if strategy in self.fusion_strategies:
                return await self.fusion_strategies[strategy](filtered_sources)
            else:
                # 默认使用加权证据融合
                return await self._weighted_evidence_fusion(filtered_sources)
                
        except Exception as e:
            logger.error(f"知识融合和排序失败: {e}")
            return {'documents': [], 'confidence_scores': {}}

    async def _weighted_evidence_fusion(
        self,
        sources: List[KnowledgeSource]
    ) -> Dict[str, Any]:
        """加权证据融合策略"""
        scored_documents = []
        confidence_scores = {}
        
        # 定义源类型权重
        type_weights = {
            'vector': 0.4,
            'graph': 0.3,
            'reasoning': 0.3
        }
        
        for i, source in enumerate(sources):
            # 基础分数
            base_score = source.confidence
            
            # 源类型权重
            type_weight = type_weights.get(source.source_type, 0.2)
            
            # 图谱上下文奖励
            graph_bonus = 0.0
            if source.graph_context:
                graph_bonus = min(0.2, source.graph_context.confidence_score * 0.1)
            
            # 推理路径奖励
            reasoning_bonus = 0.0
            if source.source_type == 'reasoning':
                hops_count = source.metadata.get('hops_count', 1)
                evidence_count = source.metadata.get('evidence_count', 1)
                # 更短的路径和更多的证据获得更高分数
                reasoning_bonus = min(0.15, (evidence_count / max(hops_count, 1)) * 0.05)
            
            # 最终分数
            final_score = (base_score * type_weight) + graph_bonus + reasoning_bonus
            final_score = min(1.0, final_score)  # 确保不超过1.0
            
            document = {
                'content': source.content,
                'source_type': source.source_type,
                'confidence': source.confidence,
                'final_score': final_score,
                'metadata': source.metadata.copy()
            }
            
            scored_documents.append(document)
            confidence_scores[f'source_{i}'] = final_score
        
        # 按最终分数排序
        scored_documents.sort(key=lambda x: x['final_score'], reverse=True)
        
        return {
            'documents': scored_documents,
            'confidence_scores': confidence_scores
        }

    async def _weighted_average_fusion(
        self,
        sources: List[KnowledgeSource]
    ) -> Dict[str, Any]:
        """加权平均融合策略"""
        # 实现加权平均融合逻辑
        return await self._weighted_evidence_fusion(sources)

    async def _rank_aggregation_fusion(
        self,
        sources: List[KnowledgeSource]
    ) -> Dict[str, Any]:
        """排序聚合融合策略"""
        # 实现排序聚合融合逻辑
        return await self._weighted_evidence_fusion(sources)

    async def _evidence_based_fusion(
        self,
        sources: List[KnowledgeSource]
    ) -> Dict[str, Any]:
        """基于证据的融合策略"""
        # 实现基于证据的融合逻辑
        return await self._weighted_evidence_fusion(sources)

    async def _confidence_threshold_fusion(
        self,
        sources: List[KnowledgeSource]
    ) -> Dict[str, Any]:
        """置信度阈值融合策略"""
        # 实现置信度阈值融合逻辑
        return await self._weighted_evidence_fusion(sources)

    async def _check_consistency(self, fused_results: Dict[str, Any]) -> float:
        """检查融合结果的一致性"""
        try:
            documents = fused_results.get('documents', [])
            if len(documents) < 2:
                return 1.0  # 单个文档认为是一致的
            
            # 计算文档间的一致性
            consistency_scores = []
            
            for i, doc1 in enumerate(documents[:5]):  # 只检查前5个文档
                for doc2 in documents[i+1:6]:
                    similarity = await self._calculate_content_similarity(
                        doc1['content'], doc2['content']
                    )
                    
                    # 如果相似度高但置信度差异大，一致性较低
                    confidence_diff = abs(doc1['confidence'] - doc2['confidence'])
                    consistency = similarity * (1 - confidence_diff)
                    consistency_scores.append(max(0.0, consistency))
            
            if consistency_scores:
                return np.mean(consistency_scores)
            else:
                return 1.0
                
        except Exception as e:
            logger.error(f"一致性检查失败: {e}")
            return 0.5
