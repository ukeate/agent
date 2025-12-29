"""
Agentic RAG服务层

提供智能检索系统的核心业务逻辑，包括：
- 智能查询处理和工作流编排
- 多代理协作和任务分发
- 检索过程监控和解释
- 用户反馈学习和系统优化
- 统计分析和性能监控
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import List, Dict, Any, Optional, AsyncGenerator
from src.models.schemas.agentic_rag import (
    # 枚举类型
    QueryIntentType, ExpansionStrategyType, RetrievalStrategyType, 
    QualityDimensionType, FailureType, StreamEventType,
    # 请求响应模型
    AgenticQueryRequest, AgenticQueryResponse, ExplanationRequest, 
    ExplanationResponse, FeedbackRequest, FeedbackResponse,
    # 数据模型
    QueryAnalysisInfo, ExpandedQueryInfo, RetrievalResultInfo,
    ValidationResultInfo, ComposedContext, RetrievalExplanation,
    FallbackResultInfo, StreamEvent, AgenticRagStats, HealthCheckResponse,
)
from src.ai.agentic_rag.query_analyzer import QueryAnalyzer, QueryAnalysis, QueryIntent
from src.ai.agentic_rag.query_expander import QueryExpander, ExpandedQuery, ExpansionStrategy
from src.ai.agentic_rag.retrieval_agents import MultiAgentRetriever, RetrievalStrategy
from src.ai.agentic_rag.result_validator import ResultValidator
from src.ai.agentic_rag.context_composer import ContextComposer, CompositionStrategy
from src.ai.agentic_rag.explainer import RetrievalExplainer
from src.ai.agentic_rag.fallback_handler import FallbackHandler

from src.core.logging import get_logger
logger = get_logger(__name__)

# 导入Agentic RAG核心组件

class AgenticRagService:
    """Agentic RAG服务"""
    
    def __init__(self):
        """初始化服务"""
        self.query_analyzer = QueryAnalyzer()
        self.query_expander = QueryExpander()
        self.multi_agent_retriever = MultiAgentRetriever()
        self.result_validator = ResultValidator()
        self.context_composer = ContextComposer()
        self.explainer = RetrievalExplainer()
        self.fallback_handler = FallbackHandler()
        
        # 性能统计
        self.stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_response_time": 0.0,
            "total_quality_score": 0.0,
            "strategy_usage": {},
            "failure_patterns": {},
        }
        
        # 会话管理
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.explanations_by_query_id: Dict[str, Any] = {}
        self.explanations_by_path_id: Dict[str, Any] = {}
        
        logger.info("Agentic RAG Service 初始化完成")

    # ==================== 核心智能查询方法 ====================

    async def intelligent_query(
        self,
        query: str,
        context_history: Optional[List[str]] = None,
        expansion_strategies: Optional[List[ExpansionStrategyType]] = None,
        retrieval_strategies: Optional[List[RetrievalStrategyType]] = None,
        max_results: int = 10,
        include_explanation: bool = True,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        执行完整的智能检索流程
        
        Returns:
            包含success, query_id和所有检索结果的字典
        """
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            # 1. 查询分析
            logger.info(f"开始查询分析: {query}")
            query_analysis = await self._analyze_query(query, context_history, session_id)
            
            # 2. 查询扩展
            logger.info(f"开始查询扩展，策略: {expansion_strategies}")
            expanded_queries = await self._expand_query(
                query_analysis, context_history, expansion_strategies
            )
            
            # 3. 多代理检索
            logger.info(f"开始多代理检索，策略: {retrieval_strategies}")
            retrieval_results = await self._multi_agent_retrieve(
                query_analysis, expanded_queries, retrieval_strategies, max_results
            )
            
            # 4. 结果验证
            logger.info("开始结果验证")
            validation_result = await self._validate_results(query_analysis, retrieval_results)
            
            # 5. 上下文组合
            logger.info("开始上下文组合")
            composed_context = await self._compose_context(
                query_analysis, validation_result, max_tokens=2000
            )
            
            # 6. 检索解释
            explanation = None
            if include_explanation:
                logger.info("生成检索解释")
                explanation = await self._generate_explanation(
                    query_id, query_analysis, retrieval_results, validation_result
                )
            
            # 7. 检查是否需要fallback处理
            processing_time = time.time() - start_time
            fallback_result = await self._check_and_handle_failure(
                query_analysis, retrieval_results, validation_result, processing_time
            )
            
            # 8. 更新统计信息
            await self._update_statistics(
                success=True,
                processing_time=processing_time,
                quality_score=validation_result.overall_quality if validation_result else 0.5,
                strategies_used=self._get_strategies_used(expansion_strategies, retrieval_strategies),
            )
            
            # 9. 更新会话状态
            if session_id:
                await self._update_session(session_id, query_id, query, retrieval_results)
            
            return {
                "success": True,
                "query_id": query_id,
                "query_analysis": self._convert_query_analysis(query_analysis),
                "expanded_queries": self._convert_expanded_queries(expanded_queries),
                "retrieval_results": self._convert_retrieval_results(retrieval_results),
                "validation_result": self._convert_validation_result(validation_result),
                "composed_context": self._convert_composed_context(composed_context),
                "explanation": self._convert_explanation(explanation),
                "fallback_result": self._convert_fallback_result(fallback_result),
                "processing_time": processing_time,
                "timestamp": utc_now(),
                "session_id": session_id,
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"智能查询失败: {e}")
            
            # 更新失败统计
            await self._update_statistics(
                success=False,
                processing_time=processing_time,
                quality_score=0.0,
                strategies_used=[],
                error_type=type(e).__name__,
            )
            
            return {
                "success": False,
                "query_id": query_id,
                "error": str(e),
                "processing_time": processing_time,
                "timestamp": utc_now(),
            }

    async def intelligent_query_stream(
        self,
        query: str,
        context_history: Optional[List[str]] = None,
        expansion_strategies: Optional[List[ExpansionStrategyType]] = None,
        retrieval_strategies: Optional[List[RetrievalStrategyType]] = None,
        max_results: int = 10,
        include_explanation: bool = True,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        执行智能检索流程并流式返回进度
        """
        query_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # 发送开始事件
            yield {
                "event_type": StreamEventType.QUERY_ANALYSIS,
                "data": {"query_id": query_id, "query": query},
                "progress": 0.0,
                "message": "开始查询分析",
                "timestamp": utc_now(),
            }
            
            # 1. 查询分析
            query_analysis = await self._analyze_query(query, context_history, session_id)
            yield {
                "event_type": StreamEventType.QUERY_ANALYSIS,
                "data": {"query_analysis": self._convert_query_analysis(query_analysis)},
                "progress": 0.15,
                "message": "查询分析完成",
                "timestamp": utc_now(),
            }
            
            # 2. 查询扩展
            yield {
                "event_type": StreamEventType.QUERY_EXPANSION,
                "data": {},
                "progress": 0.2,
                "message": "开始查询扩展",
                "timestamp": utc_now(),
            }
            
            expanded_queries = await self._expand_query(
                query_analysis, context_history, expansion_strategies
            )
            yield {
                "event_type": StreamEventType.QUERY_EXPANSION,
                "data": {"expanded_queries": self._convert_expanded_queries(expanded_queries)},
                "progress": 0.35,
                "message": "查询扩展完成",
                "timestamp": utc_now(),
            }
            
            # 3. 多代理检索
            yield {
                "event_type": StreamEventType.RETRIEVAL_START,
                "data": {},
                "progress": 0.4,
                "message": "开始多代理检索",
                "timestamp": utc_now(),
            }
            
            retrieval_results = await self._multi_agent_retrieve(
                query_analysis, expanded_queries, retrieval_strategies, max_results
            )
            
            yield {
                "event_type": StreamEventType.RETRIEVAL_COMPLETE,
                "data": {
                    "results_count": sum(len(result.results) for result in retrieval_results),
                    "agents_used": len(retrieval_results),
                },
                "progress": 0.65,
                "message": "多代理检索完成",
                "timestamp": utc_now(),
            }
            
            # 4. 结果验证
            yield {
                "event_type": StreamEventType.VALIDATION_START,
                "data": {},
                "progress": 0.7,
                "message": "开始结果验证",
                "timestamp": utc_now(),
            }
            
            validation_result = await self._validate_results(query_analysis, retrieval_results)
            yield {
                "event_type": StreamEventType.VALIDATION_COMPLETE,
                "data": {"overall_quality": validation_result.overall_quality},
                "progress": 0.8,
                "message": "结果验证完成",
                "timestamp": utc_now(),
            }
            
            # 5. 上下文组合
            yield {
                "event_type": StreamEventType.CONTEXT_COMPOSITION,
                "data": {},
                "progress": 0.85,
                "message": "开始上下文组合",
                "timestamp": utc_now(),
            }
            
            composed_context = await self._compose_context(
                query_analysis, validation_result, max_tokens=2000
            )
            yield {
                "event_type": StreamEventType.CONTEXT_COMPOSITION,
                "data": {"fragment_count": len(getattr(composed_context, 'selected_fragments', []))},
                "progress": 0.9,
                "message": "上下文组合完成",
                "timestamp": utc_now(),
            }
            
            # 6. 检索解释
            explanation = None
            if include_explanation:
                explanation = await self._generate_explanation(
                    query_id, query_analysis, retrieval_results, validation_result
                )
                yield {
                    "event_type": StreamEventType.EXPLANATION_GENERATED,
                    "data": {},
                    "progress": 0.95,
                    "message": "检索解释生成完成",
                    "timestamp": utc_now(),
                }
            
            # 7. 完成
            processing_time = time.time() - start_time
            yield {
                "event_type": StreamEventType.COMPLETE,
                "data": {
                    "query_id": query_id,
                    "processing_time": processing_time,
                    "results": {
                        "query_analysis": self._convert_query_analysis(query_analysis),
                        "expanded_queries": self._convert_expanded_queries(expanded_queries),
                        "retrieval_results": self._convert_retrieval_results(retrieval_results),
                        "validation_result": self._convert_validation_result(validation_result),
                        "composed_context": self._convert_composed_context(composed_context),
                        "explanation": self._convert_explanation(explanation),
                    }
                },
                "progress": 1.0,
                "message": "智能检索完成",
                "timestamp": utc_now(),
            }
            
        except Exception as e:
            logger.error(f"流式智能查询失败: {e}")
            yield {
                "event_type": StreamEventType.ERROR,
                "data": {"error": str(e)},
                "progress": None,
                "message": f"检索失败: {str(e)}",
                "timestamp": utc_now(),
            }

    # ==================== 核心业务逻辑方法 ====================

    async def _analyze_query(
        self, 
        query: str, 
        context_history: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> QueryAnalysis:
        """执行查询分析"""
        return await self.query_analyzer.analyze_query(query, context_history)

    async def _expand_query(
        self,
        query_analysis: QueryAnalysis,
        context_history: Optional[List[str]] = None,
        strategies: Optional[List[ExpansionStrategyType]] = None,
    ) -> List[ExpandedQuery]:
        """执行查询扩展"""
        # 转换策略类型
        expansion_strategies = None
        if strategies:
            expansion_strategies = [ExpansionStrategy(s.value) for s in strategies]
        
        return await self.query_expander.expand_query(
            query_analysis, context_history, expansion_strategies
        )

    async def _multi_agent_retrieve(
        self,
        query_analysis: QueryAnalysis,
        expanded_queries: List[ExpandedQuery],
        strategies: Optional[List[RetrievalStrategyType]] = None,
        max_results: int = 10,
    ):
        """执行多代理检索"""
        # 转换策略类型
        retrieval_strategies = None
        if strategies:
            retrieval_strategies = [RetrievalStrategy(s.value) for s in strategies]
        
        # 获取最佳扩展查询
        best_queries = self.query_expander.get_best_expansions(expanded_queries, max_results=5)
        all_queries = [query_analysis.query_text] + best_queries
        
        return await self.multi_agent_retriever.retrieve(
            query_analysis, 
            limit=max_results,
            strategies=retrieval_strategies
        )

    async def _validate_results(self, query_analysis: QueryAnalysis, retrieval_results):
        """验证检索结果"""
        return await self.result_validator.validate_results(query_analysis, retrieval_results)

    async def _compose_context(
        self,
        query_analysis: QueryAnalysis,
        validation_result,
        max_tokens: int = 2000,
    ):
        """组合上下文"""
        return await self.context_composer.compose_context(
            query_analysis, validation_result, 
            max_tokens=max_tokens,
            composition_strategy=CompositionStrategy.BALANCED.value
        )

    async def _generate_explanation(
        self,
        query_id: str,
        query_analysis: QueryAnalysis,
        retrieval_results,
        validation_result,
    ):
        """生成检索解释"""
        # 记录检索路径
        path_id = self.explainer.start_path_recording(query_analysis)
        
        # 记录决策点
        from src.ai.agentic_rag.explainer import DecisionRecord, DecisionPoint
        import time
        
        decision_record = DecisionRecord(
            decision_point=DecisionPoint.QUERY_ANALYSIS,
            timestamp=time.time(),
            input_data={"query": query_analysis.query_text, "intent": query_analysis.intent_type.value},
            decision_made={"intent_type": query_analysis.intent_type.value, "confidence": query_analysis.confidence},
            reasoning=f"基于查询内容和结构分析，识别查询意图为{query_analysis.intent_type.value}，置信度{query_analysis.confidence:.2f}",
            confidence=query_analysis.confidence,
            execution_time=0.1,
            success=True
        )
        
        self.explainer.record_decision(path_id, decision_record)
        
        # 完成路径记录
        total_results = sum(len(result.results) for result in retrieval_results)
        self.explainer.finish_path_recording(
            path_id, 
            total_time=1.0,  # 估算总时间
            results_count=total_results
        )
        
        # 生成解释
        from src.ai.agentic_rag.explainer import ExplanationLevel
        explanation = await self.explainer.explain_retrieval_process(
            path_id=path_id,
            retrieval_results=retrieval_results,
            validation_result=validation_result,
            composed_context=None,
            explanation_level=ExplanationLevel.DETAILED
        )
        
        self.explanations_by_query_id[query_id] = explanation
        self.explanations_by_path_id[explanation.retrieval_path.path_id] = explanation
        
        return explanation

    async def _check_and_handle_failure(
        self,
        query_analysis: QueryAnalysis,
        retrieval_results,
        validation_result,
        processing_time: float,
    ):
        """检查并处理检索失败"""
        return await self.fallback_handler.handle_retrieval_failure(
            query_analysis, retrieval_results, validation_result, processing_time
        )

    # ==================== 转换方法 ====================

    def _convert_query_analysis(self, analysis: QueryAnalysis) -> QueryAnalysisInfo:
        """转换查询分析结果"""
        return QueryAnalysisInfo(
            intent_type=QueryIntentType(analysis.intent_type.value),
            confidence=analysis.confidence,
            complexity_score=analysis.complexity_score,
            entities=analysis.entities,
            keywords=analysis.keywords,
            domain=analysis.domain,
            language=analysis.language,
        )

    def _convert_expanded_queries(self, expanded_queries: List[ExpandedQuery]) -> List[ExpandedQueryInfo]:
        """转换查询扩展结果"""
        return [
            ExpandedQueryInfo(
                original_query=eq.original_query,
                expanded_queries=eq.expanded_queries,
                strategy=ExpansionStrategyType(eq.strategy.value),
                confidence=eq.confidence,
                sub_questions=eq.sub_questions,
                language_variants=eq.language_variants,
                explanation=eq.explanation,
            )
            for eq in expanded_queries
        ]

    def _convert_retrieval_results(self, results) -> List[RetrievalResultInfo]:
        """转换检索结果"""
        converted_results = []
        for result in results:
            # 转换knowledge items
            knowledge_items = []
            for item in result.results:
                if isinstance(item, dict):
                    knowledge_items.append({
                        "id": item.get("id", str(uuid.uuid4())),
                        "content": item.get("content", ""),
                        "file_path": item.get("file_path"),
                        "content_type": item.get("content_type"),
                        "metadata": item.get("metadata", {}),
                        "score": item.get("score", 0.0),
                    })
                else:
                    # 如果是其他格式，尝试转换
                    knowledge_items.append({
                        "id": str(uuid.uuid4()),
                        "content": str(item),
                        "score": 0.5,
                    })
            
            converted_results.append(RetrievalResultInfo(
                agent_type=RetrievalStrategyType(result.agent_type.value),
                results=knowledge_items,
                score=result.score,
                confidence=result.confidence,
                processing_time=result.processing_time,
                explanation=result.explanation,
            ))
        
        return converted_results

    def _convert_validation_result(self, validation_result):
        """转换验证结果"""
        if not validation_result:
            return None
        
        # 转换质量评分
        quality_scores = {}
        for dimension, score in validation_result.quality_scores.items():
            quality_scores[QualityDimensionType(dimension.value)] = {
                "dimension": QualityDimensionType(dimension.value),
                "score": score.score,
                "confidence": score.confidence,
                "explanation": score.explanation,
            }
        
        # 转换冲突检测结果为字符串列表
        conflicts = []
        for conflict in validation_result.conflicts:
            conflict_description = f"{conflict.conflict_type.value}: {conflict.explanation}"
            if hasattr(conflict, 'resolution_suggestion') and conflict.resolution_suggestion:
                conflict_description += f" (建议: {conflict.resolution_suggestion})"
            conflicts.append(conflict_description)
        
        return ValidationResultInfo(
            quality_scores=quality_scores,
            conflicts=conflicts,
            overall_quality=validation_result.overall_quality,
            overall_confidence=validation_result.overall_confidence,
            recommendations=validation_result.recommendations,
        )

    def _convert_composed_context(self, context):
        """转换组合上下文"""
        if not context:
            return None

        fragment_type_map = {
            "code": "code",
            "definition": "definition",
            "example": "example",
            "procedure": "procedure",
            "reference": "reference",
            "explanation": "reference",
            "context": "reference",
        }
        relationship_type_map = {
            "dependency": "dependency",
            "sequence": "sequence",
            "hierarchy": "hierarchy",
            "contrast": "contrast",
            "similarity": "supplement",
            "complement": "supplement",
        }

        fragments = []
        id_to_index: Dict[str, int] = {}
        selected = getattr(context, "selected_fragments", []) or []

        for idx, frag in enumerate(selected):
            frag_id = str(getattr(frag, "id", idx))
            id_to_index[frag_id] = idx

            metadata = getattr(frag, "metadata", None) or {}
            original_item = metadata.get("original_item") or {}
            score = original_item.get("score", getattr(frag, "relevance_score", 0.0))
            try:
                score_f = float(score)
            except Exception:
                score_f = 0.0
            score_f = max(0.0, min(1.0, score_f))

            frag_type = getattr(getattr(frag, "fragment_type", None), "value", None) or str(
                getattr(frag, "fragment_type", "reference")
            )

            fragments.append(
                {
                    "content": str(getattr(frag, "content", "")),
                    "fragment_type": fragment_type_map.get(frag_type, "reference"),
                    "relevance_score": max(
                        0.0, min(1.0, float(getattr(frag, "relevance_score", 0.0) or 0.0))
                    ),
                    "information_density": max(
                        0.0, min(1.0, float(getattr(frag, "information_density", 0.0) or 0.0))
                    ),
                    "token_count": int(getattr(frag, "tokens", 0) or 0),
                    "source": {
                        "id": str(original_item.get("id", frag_id)),
                        "content": str(original_item.get("content", getattr(frag, "content", ""))),
                        "file_path": original_item.get("file_path", getattr(frag, "source", None)),
                        "content_type": original_item.get("file_type", original_item.get("content_type")),
                        "metadata": original_item.get("metadata") or {},
                        "score": score_f,
                    },
                }
            )

        relationships = []
        for rel in getattr(context, "relationships", []) or []:
            a = str(getattr(rel, "fragment_a", ""))
            b = str(getattr(rel, "fragment_b", ""))
            if a not in id_to_index or b not in id_to_index:
                continue

            rel_type = getattr(getattr(rel, "relationship_type", None), "value", None) or str(
                getattr(rel, "relationship_type", "supplement")
            )

            relationships.append(
                {
                    "relationship_type": relationship_type_map.get(rel_type, "supplement"),
                    "source_fragment": id_to_index[a],
                    "target_fragment": id_to_index[b],
                    "strength": max(0.0, min(1.0, float(getattr(rel, "strength", 0.0) or 0.0))),
                    "explanation": getattr(rel, "explanation", None),
                }
            )

        return {
            "fragments": fragments,
            "relationships": relationships,
            "total_tokens": int(getattr(context, "total_tokens", 0) or 0),
            "diversity_score": max(0.0, min(1.0, float(getattr(context, "diversity_score", 0.0) or 0.0))),
            "coherence_score": max(0.0, min(1.0, float(getattr(context, "coherence_score", 0.0) or 0.0))),
            "information_density": max(0.0, min(1.0, float(getattr(context, "information_density", 0.0) or 0.0))),
        }

    def _convert_explanation(self, explanation):
        """转换检索解释"""
        if not explanation:
            return None
        
        return {
            "path_record": {
                "path_id": explanation.retrieval_path.path_id,
                "query": explanation.query,
                "decision_points": [
                    {
                        "step": d.decision_point.value,
                        "decision": f"{d.decision_point.value}_decision",
                        "rationale": d.reasoning,
                        "confidence": d.confidence,
                        "alternatives": [],
                        "timestamp": datetime.fromtimestamp(d.timestamp),
                    }
                    for d in explanation.retrieval_path.decisions
                ],
                "total_time": explanation.retrieval_path.total_time,
                "success": explanation.retrieval_path.success_rate >= 0.8,
                "final_results_count": explanation.retrieval_path.final_results_count,
                "created_at": explanation.generated_at,
            },
            "confidence_analysis": {
                "overall_confidence": explanation.confidence_analysis.overall_confidence,
                "confidence_level": explanation.confidence_analysis.confidence_level.value,
                "uncertainty_factors": explanation.confidence_analysis.uncertainty_sources,
                "confidence_explanation": explanation.confidence_analysis.confidence_explanation,
            },
            "summary": explanation.summary,
            "detailed_explanation": explanation.detailed_explanation,
            "improvement_suggestions": explanation.improvement_suggestions,
            "visualization_data": {
                "flow_diagram": explanation.flow_diagram,
                "metrics_chart": explanation.metrics_chart,
                "timeline": explanation.timeline,
            },
        }

    def _convert_fallback_result(self, fallback_result):
        """转换fallback结果"""
        if not fallback_result:
            return None
        
        return {
            "original_failure": {
                "failure_type": FailureType(fallback_result.original_failure.failure_type.value),
                "severity": fallback_result.original_failure.severity.value,
                "confidence": fallback_result.original_failure.confidence,
                "evidence": fallback_result.original_failure.evidence,
                "metrics": fallback_result.original_failure.metrics,
            },
            "actions_taken": [
                {
                    "strategy": action.strategy.value,
                    "description": action.description,
                    "parameters": action.parameters,
                    "expected_improvement": action.expected_improvement,
                    "success_probability": action.success_probability,
                }
                for action in fallback_result.actions_taken
            ],
            "user_guidance": {
                "message": fallback_result.user_guidance.message if fallback_result.user_guidance else "",
                "suggestions": fallback_result.user_guidance.suggestions if fallback_result.user_guidance else [],
                "examples": fallback_result.user_guidance.examples if fallback_result.user_guidance else [],
                "severity_level": fallback_result.user_guidance.severity_level if fallback_result.user_guidance else "info",
            },
            "success": fallback_result.success,
            "improvement_metrics": fallback_result.improvement_metrics,
            "total_time": fallback_result.total_time,
        }

    # ==================== 辅助方法 ====================

    def _get_strategies_used(self, expansion_strategies, retrieval_strategies) -> List[str]:
        """获取使用的策略列表"""
        strategies = []
        if expansion_strategies:
            strategies.extend([s.value for s in expansion_strategies])
        if retrieval_strategies:
            strategies.extend([s.value for s in retrieval_strategies])
        return strategies

    async def _update_statistics(
        self,
        success: bool,
        processing_time: float,
        quality_score: float,
        strategies_used: List[str],
        error_type: Optional[str] = None,
    ):
        """更新统计信息"""
        self.stats["total_queries"] += 1
        self.stats["total_response_time"] += processing_time
        self.stats["total_quality_score"] += quality_score
        
        if success:
            self.stats["successful_queries"] += 1
        else:
            self.stats["failed_queries"] += 1
            if error_type:
                self.stats["failure_patterns"][error_type] = (
                    self.stats["failure_patterns"].get(error_type, 0) + 1
                )
        
        # 策略使用统计
        for strategy in strategies_used:
            self.stats["strategy_usage"][strategy] = (
                self.stats["strategy_usage"].get(strategy, 0) + 1
            )

    async def _update_session(
        self,
        session_id: str,
        query_id: str,
        query: str,
        results,
    ):
        """更新会话状态"""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "created_at": utc_now(),
                "query_history": [],
                "last_active": utc_now(),
            }
        
        session = self.active_sessions[session_id]
        session["query_history"].append({
            "query_id": query_id,
            "query": query,
            "timestamp": utc_now(),
            "results_count": sum(len(result.results) for result in results),
        })
        session["last_active"] = utc_now()
        
        # 限制历史记录数量
        if len(session["query_history"]) > 50:
            session["query_history"] = session["query_history"][-50:]

    # ==================== 公开API方法 ====================

    async def get_explanation(self, request: ExplanationRequest) -> Dict[str, Any]:
        """获取检索解释"""
        try:
            if request.query_id:
                # 根据query_id获取解释
                explanation = self.explanations_by_query_id.get(request.query_id)
                if not explanation:
                    return {
                        "success": False,
                        "error": f"Query ID {request.query_id} not found"
                    }
                
            elif request.path_id:
                # 根据path_id获取解释
                explanation = self.explanations_by_path_id.get(request.path_id)
                if not explanation:
                    return {
                        "success": False,
                        "error": f"Path ID {request.path_id} not found"
                    }
            else:
                return {
                    "success": False,
                    "error": "Either query_id or path_id must be provided"
                }
            
            return {
                "success": True,
                "explanation": self._convert_explanation(explanation)
            }
            
        except Exception as e:
            logger.error(f"获取解释失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def submit_feedback(self, request: FeedbackRequest) -> Dict[str, Any]:
        """提交用户反馈"""
        try:
            feedback_id = str(uuid.uuid4())
            
            # 记录反馈（这里可以扩展到数据库存储）
            logger.info(f"收到用户反馈 {feedback_id}: {request.ratings}")
            
            return {
                "success": True,
                "message": "反馈提交成功",
                "feedback_id": feedback_id
            }
            
        except Exception as e:
            logger.error(f"提交反馈失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def process_feedback_learning(self, request: FeedbackRequest):
        """处理反馈学习（后台任务）"""
        try:
            # 这里可以实现基于反馈的学习逻辑
            # 例如：调整检索策略权重、更新质量评估模型等
            logger.info(f"开始处理反馈学习: {request.query_id}")
            
            # 模拟学习过程
            await asyncio.sleep(1)
            
            logger.info(f"反馈学习处理完成: {request.query_id}")
            
        except Exception as e:
            logger.error(f"反馈学习处理失败: {e}")

    async def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        try:
            avg_response_time = (
                self.stats["total_response_time"] / max(self.stats["total_queries"], 1)
            )
            avg_quality_score = (
                self.stats["total_quality_score"] / max(self.stats["successful_queries"], 1)
            )
            
            return {
                "success": True,
                "data": {
                    "total_queries": self.stats["total_queries"],
                    "successful_queries": self.stats["successful_queries"],
                    "failed_queries": self.stats["failed_queries"],
                    "average_response_time": avg_response_time,
                    "average_quality_score": avg_quality_score,
                    "strategy_usage": self.stats["strategy_usage"],
                    "failure_patterns": self.stats["failure_patterns"],
                    "performance_metrics": {
                        "success_rate": (
                            self.stats["successful_queries"] / max(self.stats["total_queries"], 1)
                        ),
                        "avg_response_time": avg_response_time,
                        "avg_quality_score": avg_quality_score,
                    },
                    "updated_at": utc_now(),
                }
            }
            
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            components = {}
            
            # 检查各个组件
            components["query_analyzer"] = "healthy" if self.query_analyzer else "unhealthy"
            components["query_expander"] = "healthy" if self.query_expander else "unhealthy"
            components["multi_agent_retriever"] = "healthy" if self.multi_agent_retriever else "unhealthy"
            components["result_validator"] = "healthy" if self.result_validator else "unhealthy"
            components["context_composer"] = "healthy" if self.context_composer else "unhealthy"
            components["explainer"] = "healthy" if self.explainer else "unhealthy"
            components["fallback_handler"] = "healthy" if self.fallback_handler else "unhealthy"
            
            # 整体状态
            overall_status = "healthy" if all(
                status == "healthy" for status in components.values()
            ) else "unhealthy"
            
            # 获取统计信息
            stats_result = await self.get_statistics()
            stats = stats_result.get("data") if stats_result.get("success") else None
            
            return {
                "status": overall_status,
                "components": components,
                "stats": stats,
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "components": {},
                "error": str(e)
            }

# 创建全局服务实例
agentic_rag_service = AgenticRagService()
