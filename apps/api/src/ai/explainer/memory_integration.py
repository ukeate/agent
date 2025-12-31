"""记忆系统集成解释器

本模块提供记忆召回解释、记忆使用情况解释和学习过程跟踪功能。
"""

from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from src.ai.memory.context_recall import ContextAwareRecall
from src.ai.memory.models import Memory, MemoryType, MemoryStatus
from src.ai.memory.storage import MemoryStorage
from src.ai.explainer.decision_tracker import DecisionTracker
from src.models.schemas.explanation import (
    DecisionExplanation,
    ExplanationComponent,
    ExplanationLevel,
    ExplanationType,
    EvidenceType,
    ConfidenceMetrics,
    CounterfactualScenario
)

class MemoryExplainer:
    """记忆系统解释器"""
    
    def __init__(self, memory_storage: Optional[MemoryStorage] = None):
        """初始化记忆解释器"""
        self.memory_storage = memory_storage
        self.context_recall = ContextAwareRecall(memory_storage) if memory_storage else None
        
    async def explain_memory_recall(
        self,
        query_context: str,
        recalled_memories: List[Tuple[Memory, float]],
        session_id: Optional[str] = None,
        explanation_level: ExplanationLevel = ExplanationLevel.DETAILED
    ) -> DecisionExplanation:
        """解释记忆召回过程"""
        
        explanation_id = uuid4()
        
        # 创建决策跟踪器
        tracker = DecisionTracker(
            decision_id=f"memory_recall_{explanation_id.hex[:8]}",
            decision_context=f"记忆召回查询: {query_context}"
        )
        
        # 记录查询处理步骤
        query_node = tracker.create_node(
            node_type="memory_query",
            description="处理记忆召回查询",
            input_data={
                "query": query_context,
                "session_id": session_id,
                "timestamp": utc_now().isoformat()
            }
        )
        
        # 记录向量搜索步骤
        vector_search_node = tracker.create_node(
            node_type="vector_search",
            description="执行向量相似度搜索",
            input_data={
                "query_embedding_dimension": 1536,  # OpenAI embedding dimension
                "search_type": "semantic_similarity"
            },
            parent_id=query_node
        )
        
        # 记录时间衰减处理
        temporal_node = tracker.create_node(
            node_type="temporal_filtering",
            description="应用时间衰减权重",
            input_data={
                "decay_function": "exponential",
                "time_window": "24_hours"
            },
            parent_id=query_node
        )
        
        # 记录实体匹配
        entity_node = tracker.create_node(
            node_type="entity_matching",
            description="实体相关性匹配",
            input_data={
                "entity_extraction_method": "llm_based",
                "matching_algorithm": "set_intersection"
            },
            parent_id=query_node
        )
        
        # 处理召回结果
        recall_results = []
        total_relevance = 0
        memory_types_used = set()
        
        for memory, relevance_score in recalled_memories:
            recall_results.append({
                "memory_id": str(memory.id),
                "content_preview": memory.content[:100] + "..." if len(memory.content) > 100 else memory.content,
                "relevance_score": relevance_score,
                "memory_type": memory.type.value,
                "importance": memory.importance,
                "created_at": memory.created_at.isoformat(),
                "last_accessed": memory.last_accessed.isoformat()
            })
            total_relevance += relevance_score
            memory_types_used.add(memory.type.value)
        
        # 完成处理节点
        tracker.complete_node(query_node, {
            "processed_query": query_context,
            "memories_found": len(recalled_memories)
        })
        
        tracker.complete_node(vector_search_node, {
            "vector_matches": len([m for m, s in recalled_memories if s > 0.7]),
            "max_similarity": max([s for _, s in recalled_memories], default=0)
        })
        
        tracker.complete_node(temporal_node, {
            "temporal_boost_applied": True,
            "recent_memories": len([m for m, _ in recalled_memories 
                                 if (utc_now() - m.created_at).days <= 1])
        })
        
        tracker.complete_node(entity_node, {
            "entity_matches": len([m for m, _ in recalled_memories if m.tags]),
            "unique_entities": len(set().union(*[m.tags for m, _ in recalled_memories]))
        })
        
        # 添加置信度因子
        for i, (memory, score) in enumerate(recalled_memories[:5]):  # 前5个最相关的记忆
            tracker.add_confidence_factor(
                factor_name=f"memory_relevance_{i+1}",
                factor_value=score,
                weight=max(0.1, 1.0 - i * 0.2),  # 递减权重
                impact=score,
                source="memory_system"
            )
        
        # 最终化决策
        avg_relevance = total_relevance / len(recalled_memories) if recalled_memories else 0
        tracker.finalize_decision(
            final_decision=f"召回了{len(recalled_memories)}个相关记忆",
            confidence_score=min(1.0, avg_relevance * 1.5),
            reasoning=f"基于语义相似度、时间相关性和实体匹配召回记忆"
        )
        
        # 生成解释组件
        components = self._generate_memory_components(recalled_memories, query_context)
        
        # 生成置信度指标
        confidence_metrics = self._calculate_memory_confidence(recalled_memories, query_context)
        
        # 生成反事实场景
        counterfactuals = self._generate_memory_counterfactuals(recalled_memories, query_context)
        
        # 生成解释文本
        explanations = self._generate_memory_explanations(
            recalled_memories, query_context, explanation_level
        )
        
        # 创建解释对象
        explanation = DecisionExplanation(
            id=explanation_id,
            decision_id=tracker.decision_id,
            explanation_type=ExplanationType.MEMORY_RECALL,
            explanation_level=explanation_level,
            decision_description=f"为查询 '{query_context}' 召回相关记忆",
            decision_outcome=f"成功召回{len(recalled_memories)}个记忆，平均相关性{avg_relevance:.2f}",
            decision_context=tracker.decision_context,
            summary_explanation=explanations["summary"],
            detailed_explanation=explanations.get("detailed"),
            technical_explanation=explanations.get("technical"),
            components=components,
            confidence_metrics=confidence_metrics,
            counterfactuals=counterfactuals,
            visualization_data=self._generate_memory_visualization(recalled_memories),
            metadata={
                "query_context": query_context,
                "memories_count": len(recalled_memories),
                "memory_types": list(memory_types_used),
                "session_id": session_id,
                "recall_method": "hybrid_search",
                "processing_time": len(tracker.processing_steps)
            }
        )
        
        return explanation
    
    def _generate_memory_components(
        self,
        recalled_memories: List[Tuple[Memory, float]],
        query_context: str
    ) -> List[ExplanationComponent]:
        """生成记忆解释组件"""
        
        components = []
        
        # 为每个召回的记忆创建组件
        for i, (memory, relevance_score) in enumerate(recalled_memories[:5]):  # 前5个最重要的
            component = ExplanationComponent(
                factor_name=f"recalled_memory_{i+1}",
                factor_value=f"记忆内容: {memory.content[:50]}...",
                weight=max(0.1, 1.0 - i * 0.15),  # 递减权重
                impact_score=relevance_score,
                evidence_type=EvidenceType.MEMORY,
                evidence_source=f"memory_id_{memory.id}",
                evidence_content=memory.content,
                causal_relationship=f"该记忆与查询上下文的相关性为{relevance_score:.2f}",
                metadata={
                    "memory_type": memory.type.value,
                    "memory_importance": memory.importance,
                    "created_at": memory.created_at.isoformat(),
                    "last_accessed": memory.last_accessed.isoformat(),
                    "tags": memory.tags
                }
            )
            components.append(component)
        
        # 添加查询处理组件
        query_component = ExplanationComponent(
            factor_name="query_processing",
            factor_value=query_context,
            weight=0.3,
            impact_score=0.8,
            evidence_type=EvidenceType.INPUT_DATA,
            evidence_source="user_query",
            evidence_content=f"用户查询: {query_context}",
            causal_relationship="查询上下文决定了记忆召回的方向和相关性评估",
            metadata={
                "query_length": len(query_context),
                "query_type": "context_based_recall"
            }
        )
        components.append(query_component)
        
        return components
    
    def _calculate_memory_confidence(
        self,
        recalled_memories: List[Tuple[Memory, float]],
        query_context: str
    ) -> ConfidenceMetrics:
        """计算记忆召回的置信度指标"""
        
        if not recalled_memories:
            return ConfidenceMetrics(
                overall_confidence=0.1,
                uncertainty_score=0.9,
                confidence_sources=[]
            )
        
        # 计算各项指标
        relevance_scores = [score for _, score in recalled_memories]
        avg_relevance = sum(relevance_scores) / len(relevance_scores)
        max_relevance = max(relevance_scores)
        
        # 计算记忆质量
        importance_scores = [memory.importance for memory, _ in recalled_memories]
        avg_importance = sum(importance_scores) / len(importance_scores)
        
        # 计算时间新鲜度
        current_time = utc_now()
        freshness_scores = []
        for memory, _ in recalled_memories:
            hours_ago = (current_time - memory.last_accessed).total_seconds() / 3600
            freshness = max(0, 1 - hours_ago / 168)  # 一周内的记忆认为是新鲜的
            freshness_scores.append(freshness)
        avg_freshness = sum(freshness_scores) / len(freshness_scores)
        
        # 计算整体置信度
        overall_confidence = (
            avg_relevance * 0.5 +  # 相关性最重要
            avg_importance * 0.2 +  # 记忆重要性
            avg_freshness * 0.2 +   # 时间新鲜度
            min(1.0, len(recalled_memories) / 10) * 0.1  # 记忆数量多样性
        )
        
        # 计算不确定性
        relevance_variance = sum((s - avg_relevance) ** 2 for s in relevance_scores) / len(relevance_scores)
        uncertainty_score = min(1.0, relevance_variance * 2 + (1 - max_relevance) * 0.5)
        
        return ConfidenceMetrics(
            overall_confidence=overall_confidence,
            prediction_confidence=avg_relevance,
            evidence_confidence=avg_importance,
            model_confidence=avg_freshness,
            uncertainty_score=uncertainty_score,
            variance=relevance_variance,
            confidence_interval_lower=max(0, overall_confidence - uncertainty_score * 0.5),
            confidence_interval_upper=min(1.0, overall_confidence + uncertainty_score * 0.5),
            confidence_sources=[],
            calibration_score=avg_relevance
        )
    
    def _generate_memory_counterfactuals(
        self,
        recalled_memories: List[Tuple[Memory, float]],
        query_context: str
    ) -> List[CounterfactualScenario]:
        """生成记忆相关的反事实场景"""
        
        scenarios = []
        
        if not recalled_memories:
            return scenarios
        
        # 场景1: 如果没有历史记忆
        no_memory_scenario = CounterfactualScenario(
            scenario_name="无历史记忆场景",
            changed_factors={"historical_memories": "none"},
            predicted_outcome="将无法提供上下文相关的个性化回应",
            probability=0.3,
            impact_difference=-0.7,
            explanation="如果系统没有历史记忆，回应质量和个性化程度将显著降低"
        )
        scenarios.append(no_memory_scenario)
        
        # 场景2: 如果只使用最新记忆
        if len(recalled_memories) > 1:
            recent_only_scenario = CounterfactualScenario(
                scenario_name="仅使用最新记忆",
                changed_factors={"memory_selection": "recent_only"},
                predicted_outcome="可能错过重要的历史上下文信息",
                probability=0.5,
                impact_difference=-0.3,
                explanation="如果只考虑最新记忆，可能会忽略重要的长期模式和深层上下文"
            )
            scenarios.append(recent_only_scenario)
        
        # 场景3: 如果提高记忆重要性阈值
        high_importance_scenario = CounterfactualScenario(
            scenario_name="提高重要性阈值",
            changed_factors={"importance_threshold": 0.8},
            predicted_outcome="召回的记忆更少但质量更高",
            probability=0.7,
            impact_difference=0.2,
            explanation="提高重要性阈值会减少召回的记忆数量，但提高平均质量"
        )
        scenarios.append(high_importance_scenario)
        
        return scenarios
    
    def _generate_memory_explanations(
        self,
        recalled_memories: List[Tuple[Memory, float]],
        query_context: str,
        explanation_level: ExplanationLevel
    ) -> Dict[str, str]:
        """生成记忆召回解释文本"""
        
        explanations = {}
        
        if not recalled_memories:
            explanations["summary"] = f"未找到与查询 '{query_context}' 相关的记忆"
            return explanations
        
        # 概要解释
        memory_count = len(recalled_memories)
        avg_relevance = sum(score for _, score in recalled_memories) / memory_count
        top_relevance = recalled_memories[0][1] if recalled_memories else 0
        
        explanations["summary"] = (
            f"为查询 '{query_context}' 召回了{memory_count}个相关记忆，"
            f"平均相关性{avg_relevance:.1%}，最高相关性{top_relevance:.1%}"
        )
        
        # 详细解释
        if explanation_level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]:
            memory_types = set(memory.type.value for memory, _ in recalled_memories)
            recent_count = len([m for m, _ in recalled_memories 
                              if (utc_now() - m.created_at).days <= 1])
            
            explanations["detailed"] = (
                f"记忆召回过程使用了混合搜索策略，结合了语义相似度、时间相关性和实体匹配。"
                f"召回的{memory_count}个记忆涵盖了{len(memory_types)}种类型：{', '.join(memory_types)}。"
                f"其中{recent_count}个是最近24小时内的记忆，体现了时间新鲜度的重要性。"
                f"最相关的记忆相关性达到{top_relevance:.1%}，说明找到了高质量的匹配。"
            )
        
        # 技术解释
        if explanation_level == ExplanationLevel.TECHNICAL:
            vector_dim = 1536  # OpenAI embedding dimension
            high_relevance_count = len([s for _, s in recalled_memories if s > 0.7])
            
            explanations["technical"] = (
                f"技术实现细节：使用{vector_dim}维度的嵌入向量进行语义搜索，"
                f"结合指数衰减的时间权重（衰减常数配置）和基于标签的实体匹配。"
                f"向量搜索使用余弦相似度计算，时间权重采用exp(-t/λ)函数。"
                f"最终评分 = 0.5×向量相似度 + 0.2×时间权重 + 0.2×实体匹配 + 0.1×重要性。"
                f"高相关性记忆（>0.7）数量：{high_relevance_count}个。"
            )
        
        return explanations
    
    def _generate_memory_visualization(
        self,
        recalled_memories: List[Tuple[Memory, float]]
    ) -> Dict[str, Any]:
        """生成记忆可视化数据"""
        
        if not recalled_memories:
            return {}
        
        # 相关性分布
        relevance_data = [
            {
                "memory_id": str(memory.id),
                "relevance": score,
                "importance": memory.importance,
                "type": memory.type.value,
                "age_hours": (utc_now() - memory.created_at).total_seconds() / 3600
            }
            for memory, score in recalled_memories
        ]
        
        # 记忆类型分布
        type_counts = {}
        for memory, score in recalled_memories:
            memory_type = memory.type.value
            if memory_type not in type_counts:
                type_counts[memory_type] = {"count": 0, "avg_relevance": 0, "total_relevance": 0}
            type_counts[memory_type]["count"] += 1
            type_counts[memory_type]["total_relevance"] += score
            type_counts[memory_type]["avg_relevance"] = (
                type_counts[memory_type]["total_relevance"] / type_counts[memory_type]["count"]
            )
        
        # 时间分布
        time_buckets = {"recent": 0, "day": 0, "week": 0, "month": 0, "older": 0}
        current_time = utc_now()
        
        for memory, _ in recalled_memories:
            hours_ago = (current_time - memory.created_at).total_seconds() / 3600
            if hours_ago <= 1:
                time_buckets["recent"] += 1
            elif hours_ago <= 24:
                time_buckets["day"] += 1
            elif hours_ago <= 168:  # 一周
                time_buckets["week"] += 1
            elif hours_ago <= 720:  # 一个月
                time_buckets["month"] += 1
            else:
                time_buckets["older"] += 1
        
        return {
            "relevance_scatter": {
                "chart_type": "scatter",
                "data": relevance_data,
                "x_axis": "age_hours",
                "y_axis": "relevance",
                "color_by": "type"
            },
            "type_distribution": {
                "chart_type": "bar",
                "data": [
                    {
                        "type": memory_type,
                        "count": data["count"],
                        "avg_relevance": data["avg_relevance"]
                    }
                    for memory_type, data in type_counts.items()
                ]
            },
            "time_distribution": {
                "chart_type": "pie",
                "data": [
                    {"label": "最近1小时", "value": time_buckets["recent"]},
                    {"label": "今天", "value": time_buckets["day"]},
                    {"label": "本周", "value": time_buckets["week"]},
                    {"label": "本月", "value": time_buckets["month"]},
                    {"label": "更早", "value": time_buckets["older"]}
                ]
            }
        }
    
    async def explain_learning_process(
        self,
        memories_learned: List[Memory],
        learning_context: str,
        explanation_level: ExplanationLevel = ExplanationLevel.DETAILED
    ) -> DecisionExplanation:
        """解释学习过程和记忆形成"""
        
        explanation_id = uuid4()
        
        # 创建决策跟踪器
        tracker = DecisionTracker(
            decision_id=f"learning_process_{explanation_id.hex[:8]}",
            decision_context=f"学习过程: {learning_context}"
        )
        
        # 记录学习步骤
        for i, memory in enumerate(memories_learned):
            learning_node = tracker.create_node(
                node_type="memory_formation",
                description=f"形成记忆 {i+1}: {memory.type.value}",
                input_data={
                    "memory_content": memory.content[:100],
                    "memory_type": memory.type.value,
                    "importance": memory.importance,
                    "tags": memory.tags
                }
            )
            
            tracker.complete_node(learning_node, {
                "memory_id": str(memory.id),
                "processing_successful": True,
                "embedding_generated": memory.embedding is not None
            })
            
            # 添加学习因子
            tracker.add_confidence_factor(
                factor_name=f"memory_formation_{i+1}",
                factor_value=memory.importance,
                weight=0.8,
                impact=memory.importance,
                source="learning_system"
            )
        
        # 最终化学习决策
        avg_importance = sum(m.importance for m in memories_learned) / len(memories_learned) if memories_learned else 0
        tracker.finalize_decision(
            final_decision=f"成功学习并存储了{len(memories_learned)}个记忆",
            confidence_score=avg_importance,
            reasoning="基于内容重要性和上下文相关性形成记忆"
        )
        
        # 生成解释组件
        components = []
        for i, memory in enumerate(memories_learned):
            component = ExplanationComponent(
                factor_name=f"learned_memory_{i+1}",
                factor_value=memory.content[:50] + "...",
                weight=memory.importance,
                impact_score=memory.importance,
                evidence_type=EvidenceType.MEMORY,
                evidence_source="learning_process",
                evidence_content=memory.content,
                causal_relationship=f"该记忆的重要性评分为{memory.importance:.2f}",
                metadata={
                    "memory_type": memory.type.value,
                    "tags": memory.tags,
                    "created_at": memory.created_at.isoformat()
                }
            )
            components.append(component)
        
        # 生成置信度指标
        confidence_metrics = ConfidenceMetrics(
            overall_confidence=avg_importance,
            prediction_confidence=avg_importance,
            evidence_confidence=0.8,
            model_confidence=0.7,
            uncertainty_score=1 - avg_importance,
            confidence_sources=[]
        )
        
        # 生成解释文本
        memory_types = set(m.type.value for m in memories_learned)
        summary = (
            f"学习过程中形成了{len(memories_learned)}个记忆，"
            f"涵盖{len(memory_types)}种类型，平均重要性{avg_importance:.1%}"
        )
        
        detailed = (
            f"学习过程分析了输入内容的语义和上下文，识别了关键信息并评估重要性。"
            f"形成的记忆类型包括：{', '.join(memory_types)}。"
            f"每个记忆都被赋予了重要性评分，用于后续的召回排序。"
            f"学习过程还提取了关键标签和实体，以支持更好的检索和关联。"
        )
        
        # 创建解释对象
        explanation = DecisionExplanation(
            id=explanation_id,
            decision_id=tracker.decision_id,
            explanation_type=ExplanationType.MEMORY_RECALL,
            explanation_level=explanation_level,
            decision_description=f"学习过程: {learning_context}",
            decision_outcome=f"成功形成{len(memories_learned)}个记忆",
            decision_context=learning_context,
            summary_explanation=summary,
            detailed_explanation=detailed if explanation_level != ExplanationLevel.SUMMARY else None,
            components=components,
            confidence_metrics=confidence_metrics,
            counterfactuals=[],
            metadata={
                "learning_context": learning_context,
                "memories_formed": len(memories_learned),
                "memory_types": list(memory_types),
                "avg_importance": avg_importance
            }
        )
        
        return explanation
