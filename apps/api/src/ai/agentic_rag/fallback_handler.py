"""
检索失败处理器

提供检索失败情况下的备用策略和用户提示功能：
1. 检索失败检测和分类机制
2. 备用检索策略自动切换
3. 智能用户提示和建议生成
4. 查询重构和再次尝试机制
5. 失败原因分析和改进建议
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from src.ai.agentic_rag.query_analyzer import QueryAnalysis, QueryIntent
from src.ai.agentic_rag.query_expander import QueryExpander
from src.ai.agentic_rag.retrieval_agents import RetrievalResult, RetrievalStrategy, MultiAgentRetriever
from src.ai.agentic_rag.result_validator import ValidationResult
from src.models.schemas.agentic_rag import FailureType, FallbackStrategyType
from src.core.config import get_settings
from src.ai.openai_client import get_openai_client


class FailureSeverity(Enum):
    """失败严重程度"""
    LOW = "low"        # 轻微失败，可以继续
    MEDIUM = "medium"  # 中等失败，需要调整策略
    HIGH = "high"      # 严重失败，需要重大改变
    CRITICAL = "critical"  # 致命失败，无法继续


@dataclass
class FailureDetection:
    """失败检测结果"""
    failure_type: FailureType
    severity: FailureSeverity
    confidence: float  # 检测置信度
    evidence: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class FallbackAction:
    """备用行动"""
    strategy: FallbackStrategyType
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_improvement: float = 0.0  # 预期改进程度
    execution_time: float = 0.0  # 预计执行时间
    success_probability: float = 0.0  # 成功概率


@dataclass
class UserGuidance:
    """用户指导"""
    message: str
    suggestions: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    severity_level: str = "info"  # info, warning, error
    actions_required: bool = False


@dataclass
class FallbackResult:
    """备用策略执行结果"""
    original_failure: FailureDetection
    actions_taken: List[FallbackAction]
    new_query_analysis: Optional[QueryAnalysis] = None
    new_retrieval_results: List[RetrievalResult] = field(default_factory=list)
    new_validation_result: Optional[ValidationResult] = None
    user_guidance: Optional[UserGuidance] = None
    success: bool = False
    improvement_metrics: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0


class FallbackHandler:
    """检索失败处理器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = get_openai_client()
        self.query_expander = QueryExpander()
        
        # 失败检测阈值配置
        self.failure_thresholds = {
            "min_results": 1,          # 最少结果数量
            "min_quality": 0.3,        # 最低质量分数
            "min_coverage": 0.4,       # 最小覆盖度
            "max_response_time": 30.0, # 最大响应时间(秒)
            "min_confidence": 0.2      # 最低置信度
        }
        
        # 策略优先级配置
        self.strategy_priorities = {
            FailureType.NO_RESULTS: [
                FallbackStrategyType.QUERY_EXPANSION,
                FallbackStrategyType.LOWER_THRESHOLD,
                FallbackStrategyType.BROADER_SEARCH
            ],
            FailureType.LOW_QUALITY: [
                FallbackStrategyType.STRATEGY_SWITCH,
                FallbackStrategyType.ALTERNATIVE_SOURCES,
                FallbackStrategyType.QUERY_SIMPLIFICATION
            ],
            FailureType.QUERY_TOO_VAGUE: [
                FallbackStrategyType.QUERY_EXPANSION,
                FallbackStrategyType.HUMAN_ASSISTANCE
            ],
            FailureType.QUERY_TOO_COMPLEX: [
                FallbackStrategyType.QUERY_SIMPLIFICATION,
                FallbackStrategyType.STRATEGY_SWITCH
            ]
        }
        
        # 用户提示模板
        self.guidance_templates = {
            FailureType.NO_RESULTS: {
                "message": "没有找到相关结果，建议尝试以下方法：",
                "suggestions": [
                    "使用更通用的关键词",
                    "检查是否有拼写错误", 
                    "尝试使用英文关键词",
                    "描述问题的具体场景"
                ]
            },
            FailureType.LOW_QUALITY: {
                "message": "找到的结果质量较低，建议：",
                "suggestions": [
                    "提供更多上下文信息",
                    "使用更精确的术语",
                    "指定特定的技术栈或版本"
                ]
            },
            FailureType.QUERY_TOO_VAGUE: {
                "message": "您的查询比较模糊，建议：",
                "suggestions": [
                    "提供具体的使用场景",
                    "明确技术需求和约束",
                    "添加相关的背景信息"
                ]
            }
        }
        
        # 失败历史记录
        self.failure_history: List[FallbackResult] = []
    
    async def handle_retrieval_failure(
        self,
        query_analysis: QueryAnalysis,
        retrieval_results: List[RetrievalResult],
        validation_result: Optional[ValidationResult] = None,
        processing_time: float = 0.0
    ) -> FallbackResult:
        """处理检索失败"""
        start_time = time.time()
        
        # 1. 检测失败类型和严重程度
        failure_detection = await self._detect_failure(
            query_analysis, retrieval_results, validation_result, processing_time
        )
        
        if not failure_detection:
            # 没有检测到失败，返回成功结果
            return FallbackResult(
                original_failure=FailureDetection(
                    failure_type=FailureType.SYSTEM_ERROR,
                    severity=FailureSeverity.LOW,
                    confidence=0.0
                ),
                actions_taken=[],
                success=True,
                total_time=time.time() - start_time
            )
        
        # 2. 生成备用策略
        fallback_actions = await self._generate_fallback_actions(failure_detection, query_analysis)
        
        # 3. 执行备用策略
        new_query_analysis = query_analysis
        new_retrieval_results = retrieval_results
        new_validation_result = validation_result
        actions_taken = []
        
        for action in fallback_actions:
            try:
                execution_result = await self._execute_fallback_action(
                    action, new_query_analysis, new_retrieval_results
                )
                
                if execution_result:
                    new_query_analysis = execution_result.get("query_analysis", new_query_analysis)
                    new_retrieval_results = execution_result.get("retrieval_results", new_retrieval_results)
                    new_validation_result = execution_result.get("validation_result", new_validation_result)
                    actions_taken.append(action)
                    
                    # 检查是否已经解决问题
                    if await self._check_improvement(failure_detection, new_retrieval_results, new_validation_result):
                        break
                        
            except Exception as e:
                # 记录执行失败但继续尝试其他策略
                action.parameters["error"] = str(e)
                continue
        
        # 4. 生成用户指导
        user_guidance = await self._generate_user_guidance(
            failure_detection, query_analysis, actions_taken
        )
        
        # 5. 计算改进指标
        improvement_metrics = self._calculate_improvement_metrics(
            retrieval_results, new_retrieval_results, validation_result, new_validation_result
        )
        
        # 6. 判断是否成功
        success = await self._evaluate_success(failure_detection, new_retrieval_results, new_validation_result)
        
        total_time = time.time() - start_time
        
        result = FallbackResult(
            original_failure=failure_detection,
            actions_taken=actions_taken,
            new_query_analysis=new_query_analysis,
            new_retrieval_results=new_retrieval_results,
            new_validation_result=new_validation_result,
            user_guidance=user_guidance,
            success=success,
            improvement_metrics=improvement_metrics,
            total_time=total_time
        )
        
        # 记录到历史
        self.failure_history.append(result)
        
        return result
    
    async def _detect_failure(
        self,
        query_analysis: QueryAnalysis,
        retrieval_results: List[RetrievalResult],
        validation_result: Optional[ValidationResult],
        processing_time: float
    ) -> Optional[FailureDetection]:
        """检测检索失败"""
        
        failures = []
        
        # 1. 检查结果数量
        total_results = sum(len(r.results) for r in retrieval_results)
        if total_results < self.failure_thresholds["min_results"]:
            failures.append(FailureDetection(
                failure_type=FailureType.NO_RESULTS,
                severity=FailureSeverity.HIGH if total_results == 0 else FailureSeverity.MEDIUM,
                confidence=1.0,
                evidence=[f"检索到{total_results}个结果，低于最低要求{self.failure_thresholds['min_results']}个"],
                metrics={"result_count": total_results}
            ))
        
        # 2. 检查质量分数
        if validation_result and validation_result.overall_quality < self.failure_thresholds["min_quality"]:
            failures.append(FailureDetection(
                failure_type=FailureType.LOW_QUALITY,
                severity=FailureSeverity.HIGH if validation_result.overall_quality < 0.1 else FailureSeverity.MEDIUM,
                confidence=0.9,
                evidence=[f"质量评分{validation_result.overall_quality:.2f}，低于最低要求{self.failure_thresholds['min_quality']}"],
                metrics={"quality_score": validation_result.overall_quality}
            ))
        
        # 3. 检查响应时间
        if processing_time > self.failure_thresholds["max_response_time"]:
            failures.append(FailureDetection(
                failure_type=FailureType.TIMEOUT,
                severity=FailureSeverity.MEDIUM,
                confidence=1.0,
                evidence=[f"处理时间{processing_time:.2f}秒，超过最大允许时间{self.failure_thresholds['max_response_time']}秒"],
                metrics={"processing_time": processing_time}
            ))
        
        # 4. 检查查询复杂度和模糊性
        if query_analysis.complexity_score > 0.8:
            failures.append(FailureDetection(
                failure_type=FailureType.QUERY_TOO_COMPLEX,
                severity=FailureSeverity.MEDIUM,
                confidence=query_analysis.complexity_score,
                evidence=[f"查询复杂度{query_analysis.complexity_score:.2f}过高"],
                metrics={"complexity_score": query_analysis.complexity_score}
            ))
        
        if query_analysis.confidence < 0.5:
            failures.append(FailureDetection(
                failure_type=FailureType.QUERY_TOO_VAGUE,
                severity=FailureSeverity.MEDIUM,
                confidence=1.0 - query_analysis.confidence,
                evidence=[f"查询意图置信度{query_analysis.confidence:.2f}过低"],
                metrics={"query_confidence": query_analysis.confidence}
            ))
        
        # 5. 检查覆盖度（基于关键词匹配）
        if retrieval_results:
            coverage_score = await self._calculate_coverage_score(query_analysis, retrieval_results)
            if coverage_score < self.failure_thresholds["min_coverage"]:
                failures.append(FailureDetection(
                    failure_type=FailureType.INSUFFICIENT_COVERAGE,
                    severity=FailureSeverity.MEDIUM,
                    confidence=0.8,
                    evidence=[f"覆盖度{coverage_score:.2f}，低于最低要求{self.failure_thresholds['min_coverage']}"],
                    metrics={"coverage_score": coverage_score}
                ))
        
        # 返回最严重的失败
        if failures:
            failures.sort(key=lambda x: (x.severity.value, x.confidence), reverse=True)
            return failures[0]
        
        return None
    
    async def _calculate_coverage_score(
        self,
        query_analysis: QueryAnalysis,
        retrieval_results: List[RetrievalResult]
    ) -> float:
        """计算覆盖度评分"""
        if not query_analysis.keywords or not retrieval_results:
            return 0.0
        
        all_content = " ".join([
            " ".join([item.get("content", "") for item in result.results])
            for result in retrieval_results
        ]).lower()
        
        matched_keywords = sum(
            1 for keyword in query_analysis.keywords
            if keyword.lower() in all_content
        )
        
        return matched_keywords / len(query_analysis.keywords)
    
    async def _generate_fallback_actions(
        self,
        failure: FailureDetection,
        query_analysis: QueryAnalysis
    ) -> List[FallbackAction]:
        """生成备用行动"""
        actions = []
        
        # 根据失败类型选择策略
        strategies = self.strategy_priorities.get(failure.failure_type, [])
        
        for strategy in strategies[:3]:  # 最多尝试3个策略
            action = await self._create_fallback_action(strategy, failure, query_analysis)
            if action:
                actions.append(action)
        
        return actions
    
    async def _create_fallback_action(
        self,
        strategy: FallbackStrategyType,
        failure: FailureDetection,
        query_analysis: QueryAnalysis
    ) -> Optional[FallbackAction]:
        """创建具体的备用行动"""
        
        if strategy == FallbackStrategyType.QUERY_EXPANSION:
            return FallbackAction(
                strategy=strategy,
                description="扩展查询关键词以提高召回率",
                parameters={
                    "expansion_method": "synonym_and_related",
                    "max_expansions": 5
                },
                expected_improvement=0.3,
                success_probability=0.7,
                execution_time=2.0
            )
        
        elif strategy == FallbackStrategyType.QUERY_SIMPLIFICATION:
            return FallbackAction(
                strategy=strategy,
                description="简化查询以减少复杂性",
                parameters={
                    "keep_main_keywords": True,
                    "remove_modifiers": True
                },
                expected_improvement=0.4,
                success_probability=0.6,
                execution_time=1.0
            )
        
        elif strategy == FallbackStrategyType.STRATEGY_SWITCH:
            return FallbackAction(
                strategy=strategy,
                description="切换到替代检索策略",
                parameters={
                    "new_strategies": ["semantic", "keyword", "hybrid"],
                    "exclude_failed": True
                },
                expected_improvement=0.5,
                success_probability=0.8,
                execution_time=3.0
            )
        
        elif strategy == FallbackStrategyType.LOWER_THRESHOLD:
            return FallbackAction(
                strategy=strategy,
                description="降低相关性阈值以包含更多结果",
                parameters={
                    "new_threshold": max(0.1, self.failure_thresholds["min_quality"] * 0.7),
                    "max_results": 20
                },
                expected_improvement=0.2,
                success_probability=0.9,
                execution_time=0.5
            )
        
        elif strategy == FallbackStrategyType.BROADER_SEARCH:
            return FallbackAction(
                strategy=strategy,
                description="扩大搜索范围和数据源",
                parameters={
                    "include_all_sources": True,
                    "expand_time_range": True
                },
                expected_improvement=0.3,
                success_probability=0.6,
                execution_time=2.5
            )
        
        return None
    
    async def _execute_fallback_action(
        self,
        action: FallbackAction,
        query_analysis: QueryAnalysis,
        current_results: List[RetrievalResult]
    ) -> Optional[Dict[str, Any]]:
        """执行备用行动"""
        
        try:
            if action.strategy == FallbackStrategyType.QUERY_EXPANSION:
                return await self._execute_query_expansion(action, query_analysis)
            
            elif action.strategy == FallbackStrategyType.QUERY_SIMPLIFICATION:
                return await self._execute_query_simplification(action, query_analysis)
            
            elif action.strategy == FallbackStrategyType.STRATEGY_SWITCH:
                return await self._execute_strategy_switch(action, query_analysis)
            
            elif action.strategy == FallbackStrategyType.LOWER_THRESHOLD:
                return await self._execute_lower_threshold(action, current_results)
            
            elif action.strategy == FallbackStrategyType.BROADER_SEARCH:
                return await self._execute_broader_search(action, query_analysis)
        
        except Exception as e:
            action.parameters["execution_error"] = str(e)
            return None
        
        return None
    
    async def _execute_query_expansion(
        self,
        action: FallbackAction,
        query_analysis: QueryAnalysis
    ) -> Dict[str, Any]:
        """执行查询扩展"""
        expanded_queries = await self.query_expander.expand_query(
            query_analysis,
            max_expansions=action.parameters.get("max_expansions", 5)
        )
        
        if expanded_queries:
            # 使用扩展后的查询重新进行检索
            new_query_analysis = QueryAnalysis(
                query_text=expanded_queries[0],  # 使用最佳扩展
                intent_type=query_analysis.intent_type,
                confidence=query_analysis.confidence,
                complexity_score=query_analysis.complexity_score,
                entities=query_analysis.entities,
                keywords=query_analysis.keywords + [expanded_queries[0]],
                domain=query_analysis.domain,
                sentiment=query_analysis.sentiment,
                language=query_analysis.language
            )
            
            # 这里应该调用检索系统，但为了简化，我们返回修改后的查询
            return {
                "query_analysis": new_query_analysis,
                "expanded_queries": expanded_queries
            }
        
        return {}
    
    async def _execute_query_simplification(
        self,
        action: FallbackAction,
        query_analysis: QueryAnalysis
    ) -> Dict[str, Any]:
        """执行查询简化"""
        # 保留主要关键词，移除修饰词
        main_keywords = query_analysis.keywords[:3]  # 保留前3个关键词
        simplified_query = " ".join(main_keywords)
        
        new_query_analysis = QueryAnalysis(
            query_text=simplified_query,
            intent_type=query_analysis.intent_type,
            confidence=min(query_analysis.confidence + 0.1, 1.0),  # 简化后置信度稍提高
            complexity_score=max(query_analysis.complexity_score - 0.2, 0.0),  # 复杂度降低
            entities=query_analysis.entities[:2],  # 保留主要实体
            keywords=main_keywords,
            domain=query_analysis.domain,
            sentiment=query_analysis.sentiment,
            language=query_analysis.language
        )
        
        return {"query_analysis": new_query_analysis}
    
    async def _execute_strategy_switch(
        self,
        action: FallbackAction,
        query_analysis: QueryAnalysis
    ) -> Dict[str, Any]:
        """执行策略切换"""
        # 这里应该调用不同的检索策略
        # 为了简化，我们返回策略切换的参数
        return {
            "strategy_changed": True,
            "new_strategies": action.parameters.get("new_strategies", [])
        }
    
    async def _execute_lower_threshold(
        self,
        action: FallbackAction,
        current_results: List[RetrievalResult]
    ) -> Dict[str, Any]:
        """执行降低阈值"""
        new_threshold = action.parameters.get("new_threshold", 0.1)
        
        # 重新筛选结果，使用更低的阈值
        filtered_results = []
        for result in current_results:
            new_items = [
                item for item in result.results
                if item.get("score", 0) >= new_threshold
            ]
            if new_items:
                new_result = RetrievalResult(
                    agent_type=result.agent_type,
                    query=result.query,
                    results=new_items,
                    score=result.score,
                    confidence=result.confidence,
                    processing_time=result.processing_time,
                    explanation=f"降低阈值至{new_threshold}后的结果"
                )
                filtered_results.append(new_result)
        
        return {"retrieval_results": filtered_results}
    
    async def _execute_broader_search(
        self,
        action: FallbackAction,
        query_analysis: QueryAnalysis
    ) -> Dict[str, Any]:
        """执行扩大搜索"""
        # 这里应该调用更广泛的检索策略
        # 为了简化，我们返回扩大搜索的标记
        return {
            "search_broadened": True,
            "include_all_sources": action.parameters.get("include_all_sources", True)
        }
    
    async def _check_improvement(
        self,
        original_failure: FailureDetection,
        new_results: List[RetrievalResult],
        new_validation: Optional[ValidationResult]
    ) -> bool:
        """检查是否有改进"""
        
        # 检查结果数量改进
        if original_failure.failure_type == FailureType.NO_RESULTS:
            total_results = sum(len(r.results) for r in new_results)
            return total_results > 0
        
        # 检查质量改进
        if original_failure.failure_type == FailureType.LOW_QUALITY and new_validation:
            original_quality = original_failure.metrics.get("quality_score", 0)
            return new_validation.overall_quality > original_quality
        
        return False
    
    async def _generate_user_guidance(
        self,
        failure: FailureDetection,
        query_analysis: QueryAnalysis,
        actions_taken: List[FallbackAction]
    ) -> UserGuidance:
        """生成用户指导"""
        
        template = self.guidance_templates.get(failure.failure_type, {
            "message": "检索遇到了一些问题，请尝试以下建议：",
            "suggestions": ["重新表述您的问题", "提供更多上下文信息"]
        })
        
        # 基础指导
        message = template["message"]
        suggestions = template["suggestions"].copy()
        examples = []
        
        # 根据查询意图添加特定建议
        if query_analysis.intent_type == QueryIntent.CODE:
            suggestions.extend([
                "包含具体的编程语言或框架名称",
                "提供代码上下文或错误信息",
                "指定版本号或技术栈"
            ])
            examples = [
                "如何在Python Flask中实现用户认证？",
                "React hooks useEffect的最佳实践有哪些？"
            ]
        
        elif query_analysis.intent_type == QueryIntent.FACTUAL:
            suggestions.extend([
                "使用更具体的术语",
                "指定时间范围或版本",
                "添加相关的背景信息"
            ])
            
        # 根据执行的行动调整建议
        if actions_taken:
            action_messages = []
            for action in actions_taken:
                if action.strategy == FallbackStrategyType.QUERY_EXPANSION:
                    action_messages.append("已尝试扩展您的查询关键词")
                elif action.strategy == FallbackStrategyType.QUERY_SIMPLIFICATION:
                    action_messages.append("已尝试简化您的查询")
            
            if action_messages:
                message += f" 系统已自动{', '.join(action_messages)}。"
        
        # 确定严重程度
        severity_level = "info"
        if failure.severity in [FailureSeverity.HIGH, FailureSeverity.CRITICAL]:
            severity_level = "error"
        elif failure.severity == FailureSeverity.MEDIUM:
            severity_level = "warning"
        
        return UserGuidance(
            message=message,
            suggestions=suggestions,
            examples=examples,
            severity_level=severity_level,
            actions_required=(failure.severity in [FailureSeverity.HIGH, FailureSeverity.CRITICAL])
        )
    
    def _calculate_improvement_metrics(
        self,
        original_results: List[RetrievalResult],
        new_results: List[RetrievalResult],
        original_validation: Optional[ValidationResult],
        new_validation: Optional[ValidationResult]
    ) -> Dict[str, float]:
        """计算改进指标"""
        metrics = {}
        
        # 结果数量改进
        original_count = sum(len(r.results) for r in original_results)
        new_count = sum(len(r.results) for r in new_results)
        metrics["result_count_change"] = new_count - original_count
        
        if original_count > 0:
            metrics["result_count_improvement"] = (new_count - original_count) / original_count
        else:
            metrics["result_count_improvement"] = 1.0 if new_count > 0 else 0.0
        
        # 质量改进
        if original_validation and new_validation:
            metrics["quality_change"] = new_validation.overall_quality - original_validation.overall_quality
            metrics["quality_improvement"] = metrics["quality_change"] / max(original_validation.overall_quality, 0.1)
        
        # 置信度改进
        original_confidence = sum(r.confidence for r in original_results) / len(original_results) if original_results else 0
        new_confidence = sum(r.confidence for r in new_results) / len(new_results) if new_results else 0
        metrics["confidence_change"] = new_confidence - original_confidence
        
        return metrics
    
    async def _evaluate_success(
        self,
        original_failure: FailureDetection,
        new_results: List[RetrievalResult],
        new_validation: Optional[ValidationResult]
    ) -> bool:
        """评估备用策略是否成功"""
        
        # 基本成功条件
        total_results = sum(len(r.results) for r in new_results)
        
        if total_results == 0:
            return False
        
        # 根据原始失败类型评估
        if original_failure.failure_type == FailureType.NO_RESULTS:
            return total_results >= self.failure_thresholds["min_results"]
        
        if original_failure.failure_type == FailureType.LOW_QUALITY and new_validation:
            return new_validation.overall_quality >= self.failure_thresholds["min_quality"]
        
        if original_failure.failure_type in [FailureType.QUERY_TOO_VAGUE, FailureType.QUERY_TOO_COMPLEX]:
            # 如果有结果且质量可接受，认为成功
            return total_results > 0 and (
                not new_validation or new_validation.overall_quality >= 0.4
            )
        
        return total_results > 0
    
    def get_failure_statistics(self) -> Dict[str, Any]:
        """获取失败统计信息"""
        if not self.failure_history:
            return {"total_failures": 0}
        
        stats = {
            "total_failures": len(self.failure_history),
            "success_rate": sum(1 for r in self.failure_history if r.success) / len(self.failure_history),
            "failure_types": {},
            "common_strategies": {},
            "avg_improvement": 0.0
        }
        
        for result in self.failure_history:
            # 统计失败类型
            failure_type = result.original_failure.failure_type.value
            stats["failure_types"][failure_type] = stats["failure_types"].get(failure_type, 0) + 1
            
            # 统计使用的策略
            for action in result.actions_taken:
                strategy = action.strategy.value
                stats["common_strategies"][strategy] = stats["common_strategies"].get(strategy, 0) + 1
        
        # 计算平均改进
        improvements = [
            r.improvement_metrics.get("result_count_improvement", 0)
            for r in self.failure_history
            if r.improvement_metrics
        ]
        if improvements:
            stats["avg_improvement"] = sum(improvements) / len(improvements)
        
        return stats
    
    def clear_failure_history(self):
        """清空失败历史"""
        self.failure_history.clear()