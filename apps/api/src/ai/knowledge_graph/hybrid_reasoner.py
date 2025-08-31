"""
混合推理引擎模块

整合多种推理方法：规则推理、嵌入推理、路径推理、不确定性推理
提供统一的推理接口和结果融合机制
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import numpy as np
import json

from .rule_engine import RuleEngine, InferenceResult as RuleInferenceResult
from .embedding_engine import EmbeddingEngine, SimilarityResult
from .path_reasoning import PathReasoner, PathSearchResult, ReasoningPath
from .uncertainty_reasoning import UncertaintyReasoner, UncertaintyQuantification
from .reasoning_optimizer import ReasoningOptimizer, ReasoningPriority

logger = logging.getLogger(__name__)


class ReasoningStrategy(Enum):
    """推理策略"""
    RULE_ONLY = "rule_only"                    # 仅规则推理
    EMBEDDING_ONLY = "embedding_only"          # 仅嵌入推理
    PATH_ONLY = "path_only"                    # 仅路径推理
    UNCERTAINTY_ONLY = "uncertainty_only"      # 仅不确定性推理
    ENSEMBLE = "ensemble"                      # 集成所有方法
    ADAPTIVE = "adaptive"                      # 自适应策略选择
    CASCADING = "cascading"                    # 级联推理
    VOTING = "voting"                         # 投票机制


class ConfidenceWeights(Enum):
    """置信度权重策略"""
    EQUAL = "equal"                           # 等权重
    PERFORMANCE_BASED = "performance_based"    # 基于性能的权重
    DOMAIN_SPECIFIC = "domain_specific"       # 领域特定权重
    DYNAMIC = "dynamic"                       # 动态调整权重


@dataclass
class ReasoningRequest:
    """推理请求"""
    query: str
    query_type: str
    context: Dict[str, Any] = field(default_factory=dict)
    entities: List[str] = field(default_factory=list)
    relations: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    strategy: ReasoningStrategy = ReasoningStrategy.ADAPTIVE
    max_depth: int = 3
    top_k: int = 10
    confidence_threshold: float = 0.5
    timeout: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningEvidence:
    """推理证据"""
    source: str                              # 推理源
    method: str                              # 推理方法
    evidence_type: str                       # 证据类型
    content: Any                             # 证据内容
    confidence: float                        # 置信度
    support_count: int = 0                   # 支持计数
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HybridReasoningResult:
    """混合推理结果"""
    query: str
    results: List[Dict[str, Any]]
    confidence: float
    evidences: List[ReasoningEvidence]
    strategy_used: ReasoningStrategy
    execution_time: float
    method_contributions: Dict[str, float]    # 各方法贡献度
    uncertainty_analysis: Optional[UncertaintyQuantification] = None
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """策略性能统计"""
    strategy: ReasoningStrategy
    total_queries: int = 0
    success_queries: int = 0
    avg_confidence: float = 0.0
    avg_execution_time: float = 0.0
    accuracy_score: float = 0.0
    last_updated: datetime = field(default_factory=utc_factory)


class HybridReasoner:
    """混合推理引擎"""
    
    def __init__(self, 
                 rule_engine: RuleEngine,
                 embedding_engine: EmbeddingEngine,
                 path_reasoner: PathReasoner,
                 uncertainty_reasoner: UncertaintyReasoner,
                 optimizer: ReasoningOptimizer):
        """初始化混合推理引擎"""
        self.rule_engine = rule_engine
        self.embedding_engine = embedding_engine
        self.path_reasoner = path_reasoner
        self.uncertainty_reasoner = uncertainty_reasoner
        self.optimizer = optimizer
        
        # 策略性能统计
        self.strategy_performance: Dict[ReasoningStrategy, StrategyPerformance] = {}
        self._initialize_strategy_performance()
        
        # 权重配置
        self.confidence_weights = {
            "rule": 0.3,
            "embedding": 0.25,
            "path": 0.25,
            "uncertainty": 0.2
        }
        
        # 自适应阈值
        self.adaptive_thresholds = {
            "high_confidence": 0.8,
            "medium_confidence": 0.6,
            "low_confidence": 0.4
        }
        
        logger.info("混合推理引擎初始化完成")
    
    def _initialize_strategy_performance(self):
        """初始化策略性能统计"""
        for strategy in ReasoningStrategy:
            self.strategy_performance[strategy] = StrategyPerformance(strategy=strategy)
    
    async def reason(self, request: ReasoningRequest) -> HybridReasoningResult:
        """执行混合推理"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 优化推理请求
            optimized_request = await self.optimizer.optimize_reasoning_request(
                request.__dict__, 
                ReasoningPriority.HIGH
            )
            
            # 选择推理策略
            if request.strategy == ReasoningStrategy.ADAPTIVE:
                strategy = await self._select_adaptive_strategy(request)
            else:
                strategy = request.strategy
            
            # 执行推理
            result = await self._execute_reasoning_strategy(strategy, request)
            
            # 记录性能
            execution_time = asyncio.get_event_loop().time() - start_time
            await self._update_strategy_performance(strategy, result, execution_time)
            
            result.execution_time = execution_time
            result.strategy_used = strategy
            
            logger.info(f"混合推理完成: {strategy.value}, 置信度: {result.confidence:.3f}, 耗时: {execution_time:.3f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"混合推理失败: {str(e)}")
            # 返回空结果
            return HybridReasoningResult(
                query=request.query,
                results=[],
                confidence=0.0,
                evidences=[],
                strategy_used=request.strategy,
                execution_time=asyncio.get_event_loop().time() - start_time,
                method_contributions={},
                explanation=f"推理失败: {str(e)}"
            )
    
    async def _select_adaptive_strategy(self, request: ReasoningRequest) -> ReasoningStrategy:
        """自适应策略选择"""
        # 分析查询特征
        query_features = await self._analyze_query_features(request)
        
        # 获取历史性能
        best_strategy = ReasoningStrategy.ENSEMBLE
        best_score = 0.0
        
        for strategy, perf in self.strategy_performance.items():
            if perf.total_queries > 0:
                # 计算综合评分
                score = (perf.accuracy_score * 0.4 + 
                        perf.avg_confidence * 0.3 + 
                        (1.0 / max(perf.avg_execution_time, 0.001)) * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy
        
        # 根据查询特征调整策略
        if query_features.get("has_rules", False):
            if best_strategy == ReasoningStrategy.EMBEDDING_ONLY:
                best_strategy = ReasoningStrategy.CASCADING
        
        if query_features.get("needs_uncertainty", False):
            if best_strategy in [ReasoningStrategy.RULE_ONLY, ReasoningStrategy.EMBEDDING_ONLY]:
                best_strategy = ReasoningStrategy.ENSEMBLE
        
        logger.debug(f"自适应策略选择: {best_strategy.value}, 评分: {best_score:.3f}")
        return best_strategy
    
    async def _analyze_query_features(self, request: ReasoningRequest) -> Dict[str, Any]:
        """分析查询特征"""
        features = {
            "has_rules": False,
            "needs_similarity": False,
            "needs_path": False,
            "needs_uncertainty": False,
            "query_complexity": "medium"
        }
        
        query_lower = request.query.lower()
        
        # 规则相关关键词
        rule_keywords = ["if", "then", "when", "implies", "follows", "因为", "所以", "如果", "那么"]
        features["has_rules"] = any(keyword in query_lower for keyword in rule_keywords)
        
        # 相似性相关关键词
        similarity_keywords = ["similar", "like", "related", "类似", "相似", "相关"]
        features["needs_similarity"] = any(keyword in query_lower for keyword in similarity_keywords)
        
        # 路径相关关键词
        path_keywords = ["path", "connect", "relationship", "路径", "连接", "关系"]
        features["needs_path"] = any(keyword in query_lower for keyword in path_keywords)
        
        # 不确定性相关关键词
        uncertainty_keywords = ["maybe", "probably", "uncertain", "可能", "大概", "不确定"]
        features["needs_uncertainty"] = any(keyword in query_lower for keyword in uncertainty_keywords)
        
        # 查询复杂度
        if len(request.entities) > 5 or len(request.relations) > 3:
            features["query_complexity"] = "high"
        elif len(request.entities) <= 2 and len(request.relations) <= 1:
            features["query_complexity"] = "low"
        
        return features
    
    async def _execute_reasoning_strategy(self, strategy: ReasoningStrategy, request: ReasoningRequest) -> HybridReasoningResult:
        """执行特定推理策略"""
        if strategy == ReasoningStrategy.RULE_ONLY:
            return await self._execute_rule_reasoning(request)
        elif strategy == ReasoningStrategy.EMBEDDING_ONLY:
            return await self._execute_embedding_reasoning(request)
        elif strategy == ReasoningStrategy.PATH_ONLY:
            return await self._execute_path_reasoning(request)
        elif strategy == ReasoningStrategy.UNCERTAINTY_ONLY:
            return await self._execute_uncertainty_reasoning(request)
        elif strategy == ReasoningStrategy.ENSEMBLE:
            return await self._execute_ensemble_reasoning(request)
        elif strategy == ReasoningStrategy.CASCADING:
            return await self._execute_cascading_reasoning(request)
        elif strategy == ReasoningStrategy.VOTING:
            return await self._execute_voting_reasoning(request)
        else:
            # 默认使用集成方法
            return await self._execute_ensemble_reasoning(request)
    
    async def _execute_rule_reasoning(self, request: ReasoningRequest) -> HybridReasoningResult:
        """执行规则推理"""
        try:
            # 构造初始事实
            initial_facts = []
            for entity in request.entities:
                initial_facts.append(f"entity({entity})")
            for relation in request.relations:
                initial_facts.append(f"relation({relation})")
            
            # 执行前向链式推理
            inference_results = await self.rule_engine.forward_chaining(initial_facts, max_iterations=10)
            
            # 转换结果格式
            results = []
            evidences = []
            total_confidence = 0.0
            
            for result in inference_results:
                results.append({
                    "conclusion": result.conclusion,
                    "confidence": result.confidence,
                    "applied_rules": result.applied_rules
                })
                
                evidences.append(ReasoningEvidence(
                    source="rule_engine",
                    method="forward_chaining",
                    evidence_type="rule_application",
                    content=result.applied_rules,
                    confidence=result.confidence,
                    support_count=len(result.applied_rules)
                ))
                
                total_confidence += result.confidence
            
            avg_confidence = total_confidence / max(len(inference_results), 1)
            
            return HybridReasoningResult(
                query=request.query,
                results=results,
                confidence=avg_confidence,
                evidences=evidences,
                strategy_used=ReasoningStrategy.RULE_ONLY,
                execution_time=0.0,
                method_contributions={"rule": 1.0},
                explanation=f"通过规则推理得出{len(results)}个结论"
            )
            
        except Exception as e:
            logger.error(f"规则推理失败: {str(e)}")
            return self._create_empty_result(request, f"规则推理失败: {str(e)}")
    
    async def _execute_embedding_reasoning(self, request: ReasoningRequest) -> HybridReasoningResult:
        """执行嵌入推理"""
        try:
            results = []
            evidences = []
            total_confidence = 0.0
            
            # 对每个实体进行相似性搜索
            for entity in request.entities:
                similar_results = await self.embedding_engine.find_similar_entities(
                    entity, top_k=request.top_k
                )
                
                for sim_result in similar_results.similar_entities:
                    results.append({
                        "entity": sim_result.entity,
                        "similarity": sim_result.similarity,
                        "confidence": sim_result.similarity
                    })
                    
                    evidences.append(ReasoningEvidence(
                        source="embedding_engine",
                        method="similarity_search",
                        evidence_type="entity_similarity",
                        content={"entity": sim_result.entity, "similarity": sim_result.similarity},
                        confidence=sim_result.similarity
                    ))
                    
                    total_confidence += sim_result.similarity
            
            avg_confidence = total_confidence / max(len(results), 1)
            
            return HybridReasoningResult(
                query=request.query,
                results=results,
                confidence=avg_confidence,
                evidences=evidences,
                strategy_used=ReasoningStrategy.EMBEDDING_ONLY,
                execution_time=0.0,
                method_contributions={"embedding": 1.0},
                explanation=f"通过嵌入相似性搜索得出{len(results)}个相关实体"
            )
            
        except Exception as e:
            logger.error(f"嵌入推理失败: {str(e)}")
            return self._create_empty_result(request, f"嵌入推理失败: {str(e)}")
    
    async def _execute_path_reasoning(self, request: ReasoningRequest) -> HybridReasoningResult:
        """执行路径推理"""
        try:
            results = []
            evidences = []
            total_confidence = 0.0
            
            # 在实体对之间寻找推理路径
            for i, start_entity in enumerate(request.entities):
                for end_entity in request.entities[i+1:]:
                    path_result = await self.path_reasoner.find_reasoning_paths(
                        start_entity, end_entity, 
                        relation_constraints=request.relations
                    )
                    
                    for path in path_result.paths:
                        results.append({
                            "start_entity": start_entity,
                            "end_entity": end_entity,
                            "path": path.path_entities,
                            "relations": path.path_relations,
                            "confidence": path.confidence
                        })
                        
                        evidences.append(ReasoningEvidence(
                            source="path_reasoner",
                            method="path_search",
                            evidence_type="reasoning_path",
                            content={
                                "path": path.path_entities,
                                "relations": path.path_relations
                            },
                            confidence=path.confidence,
                            support_count=len(path.path_entities)
                        ))
                        
                        total_confidence += path.confidence
            
            avg_confidence = total_confidence / max(len(results), 1)
            
            return HybridReasoningResult(
                query=request.query,
                results=results,
                confidence=avg_confidence,
                evidences=evidences,
                strategy_used=ReasoningStrategy.PATH_ONLY,
                execution_time=0.0,
                method_contributions={"path": 1.0},
                explanation=f"通过路径推理发现{len(results)}条推理路径"
            )
            
        except Exception as e:
            logger.error(f"路径推理失败: {str(e)}")
            return self._create_empty_result(request, f"路径推理失败: {str(e)}")
    
    async def _execute_uncertainty_reasoning(self, request: ReasoningRequest) -> HybridReasoningResult:
        """执行不确定性推理"""
        try:
            # 构造证据
            evidence = {}
            for i, entity in enumerate(request.entities):
                evidence[f"entity_{i}"] = 0.8  # 假设实体存在的先验概率
            
            # 执行贝叶斯推理
            uncertainty_result = await self.uncertainty_reasoner.calculate_inference_confidence(
                evidence, request.query
            )
            
            results = [{
                "hypothesis": request.query,
                "posterior_probability": uncertainty_result.posterior_probability,
                "confidence_interval": uncertainty_result.confidence_interval,
                "uncertainty_score": uncertainty_result.uncertainty_score
            }]
            
            evidences = [ReasoningEvidence(
                source="uncertainty_reasoner",
                method="bayesian_inference",
                evidence_type="uncertainty_quantification",
                content={
                    "evidence": evidence,
                    "posterior": uncertainty_result.posterior_probability,
                    "uncertainty": uncertainty_result.uncertainty_score
                },
                confidence=uncertainty_result.posterior_probability
            )]
            
            return HybridReasoningResult(
                query=request.query,
                results=results,
                confidence=uncertainty_result.posterior_probability,
                evidences=evidences,
                strategy_used=ReasoningStrategy.UNCERTAINTY_ONLY,
                execution_time=0.0,
                method_contributions={"uncertainty": 1.0},
                uncertainty_analysis=uncertainty_result,
                explanation=f"通过不确定性推理计算假设概率为{uncertainty_result.posterior_probability:.3f}"
            )
            
        except Exception as e:
            logger.error(f"不确定性推理失败: {str(e)}")
            return self._create_empty_result(request, f"不确定性推理失败: {str(e)}")
    
    async def _execute_ensemble_reasoning(self, request: ReasoningRequest) -> HybridReasoningResult:
        """执行集成推理"""
        try:
            # 并行执行所有推理方法
            tasks = [
                self._execute_rule_reasoning(request),
                self._execute_embedding_reasoning(request),
                self._execute_path_reasoning(request),
                self._execute_uncertainty_reasoning(request)
            ]
            
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 融合结果
            fused_result = await self._fuse_reasoning_results(request, results_list)
            fused_result.strategy_used = ReasoningStrategy.ENSEMBLE
            
            return fused_result
            
        except Exception as e:
            logger.error(f"集成推理失败: {str(e)}")
            return self._create_empty_result(request, f"集成推理失败: {str(e)}")
    
    async def _execute_cascading_reasoning(self, request: ReasoningRequest) -> HybridReasoningResult:
        """执行级联推理"""
        try:
            # 级联顺序：规则 -> 路径 -> 嵌入 -> 不确定性
            current_request = request
            cascade_results = []
            all_evidences = []
            
            # 第一层：规则推理
            rule_result = await self._execute_rule_reasoning(current_request)
            cascade_results.append(rule_result)
            all_evidences.extend(rule_result.evidences)
            
            # 如果规则推理置信度足够，可以继续
            if rule_result.confidence > self.adaptive_thresholds["medium_confidence"]:
                # 第二层：路径推理
                path_result = await self._execute_path_reasoning(current_request)
                cascade_results.append(path_result)
                all_evidences.extend(path_result.evidences)
                
                # 第三层：嵌入推理
                if path_result.confidence > self.adaptive_thresholds["low_confidence"]:
                    embedding_result = await self._execute_embedding_reasoning(current_request)
                    cascade_results.append(embedding_result)
                    all_evidences.extend(embedding_result.evidences)
            
            # 融合级联结果
            fused_result = await self._fuse_reasoning_results(request, cascade_results)
            fused_result.strategy_used = ReasoningStrategy.CASCADING
            fused_result.evidences = all_evidences
            
            return fused_result
            
        except Exception as e:
            logger.error(f"级联推理失败: {str(e)}")
            return self._create_empty_result(request, f"级联推理失败: {str(e)}")
    
    async def _execute_voting_reasoning(self, request: ReasoningRequest) -> HybridReasoningResult:
        """执行投票推理"""
        try:
            # 并行执行多个推理方法
            tasks = [
                self._execute_rule_reasoning(request),
                self._execute_embedding_reasoning(request),
                self._execute_path_reasoning(request)
            ]
            
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            valid_results = [r for r in results_list if isinstance(r, HybridReasoningResult)]
            
            # 投票决策
            voted_result = await self._vote_on_results(request, valid_results)
            voted_result.strategy_used = ReasoningStrategy.VOTING
            
            return voted_result
            
        except Exception as e:
            logger.error(f"投票推理失败: {str(e)}")
            return self._create_empty_result(request, f"投票推理失败: {str(e)}")
    
    async def _fuse_reasoning_results(self, request: ReasoningRequest, results_list: List[HybridReasoningResult]) -> HybridReasoningResult:
        """融合推理结果"""
        valid_results = [r for r in results_list if isinstance(r, HybridReasoningResult) and r.confidence > 0]
        
        if not valid_results:
            return self._create_empty_result(request, "所有推理方法都失败")
        
        # 合并所有结果
        all_results = []
        all_evidences = []
        method_contributions = {}
        
        for result in valid_results:
            all_results.extend(result.results)
            all_evidences.extend(result.evidences)
            
            # 计算方法贡献
            for method, contribution in result.method_contributions.items():
                if method in method_contributions:
                    method_contributions[method] += contribution * result.confidence
                else:
                    method_contributions[method] = contribution * result.confidence
        
        # 归一化方法贡献
        total_contribution = sum(method_contributions.values())
        if total_contribution > 0:
            for method in method_contributions:
                method_contributions[method] /= total_contribution
        
        # 计算加权平均置信度
        weighted_confidence = sum(r.confidence * len(r.results) for r in valid_results) / max(sum(len(r.results) for r in valid_results), 1)
        
        # 去重和排序结果
        unique_results = self._deduplicate_results(all_results)
        sorted_results = sorted(unique_results, key=lambda x: x.get("confidence", 0), reverse=True)
        
        return HybridReasoningResult(
            query=request.query,
            results=sorted_results[:request.top_k],
            confidence=weighted_confidence,
            evidences=all_evidences,
            strategy_used=ReasoningStrategy.ENSEMBLE,
            execution_time=0.0,
            method_contributions=method_contributions,
            explanation=f"融合{len(valid_results)}种推理方法，得到{len(sorted_results)}个结果"
        )
    
    async def _vote_on_results(self, request: ReasoningRequest, results_list: List[HybridReasoningResult]) -> HybridReasoningResult:
        """对结果进行投票"""
        if not results_list:
            return self._create_empty_result(request, "没有有效的推理结果")
        
        # 收集所有候选结果
        candidate_results = {}
        vote_counts = {}
        
        for result in results_list:
            for item in result.results:
                # 简化结果表示用于投票
                key = str(item)
                if key not in candidate_results:
                    candidate_results[key] = item
                    vote_counts[key] = 0
                
                # 根据置信度加权投票
                vote_counts[key] += result.confidence
        
        # 按投票数排序
        sorted_candidates = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
        
        # 构造最终结果
        final_results = []
        for key, votes in sorted_candidates[:request.top_k]:
            result = candidate_results[key].copy()
            result["vote_score"] = votes
            result["confidence"] = min(votes / len(results_list), 1.0)
            final_results.append(result)
        
        # 合并证据
        all_evidences = []
        for result in results_list:
            all_evidences.extend(result.evidences)
        
        avg_confidence = sum(item["confidence"] for item in final_results) / max(len(final_results), 1)
        
        return HybridReasoningResult(
            query=request.query,
            results=final_results,
            confidence=avg_confidence,
            evidences=all_evidences,
            strategy_used=ReasoningStrategy.VOTING,
            execution_time=0.0,
            method_contributions={"voting": 1.0},
            explanation=f"通过{len(results_list)}种方法投票，选出{len(final_results)}个最佳结果"
        )
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去重结果"""
        seen = set()
        unique_results = []
        
        for result in results:
            # 创建结果的简化表示用于去重
            key = str(sorted(result.items()) if isinstance(result, dict) else result)
            
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results
    
    def _create_empty_result(self, request: ReasoningRequest, explanation: str) -> HybridReasoningResult:
        """创建空结果"""
        return HybridReasoningResult(
            query=request.query,
            results=[],
            confidence=0.0,
            evidences=[],
            strategy_used=request.strategy,
            execution_time=0.0,
            method_contributions={},
            explanation=explanation
        )
    
    async def _update_strategy_performance(self, strategy: ReasoningStrategy, result: HybridReasoningResult, execution_time: float):
        """更新策略性能统计"""
        perf = self.strategy_performance[strategy]
        perf.total_queries += 1
        
        if result.confidence > 0:
            perf.success_queries += 1
        
        # 更新平均值
        perf.avg_confidence = (perf.avg_confidence * (perf.total_queries - 1) + result.confidence) / perf.total_queries
        perf.avg_execution_time = (perf.avg_execution_time * (perf.total_queries - 1) + execution_time) / perf.total_queries
        
        # 简单的准确率估算（基于置信度）
        perf.accuracy_score = perf.success_queries / perf.total_queries
        perf.last_updated = utc_now()
    
    async def get_strategy_performance_stats(self) -> Dict[str, Any]:
        """获取策略性能统计"""
        stats = {}
        
        for strategy, perf in self.strategy_performance.items():
            stats[strategy.value] = {
                "total_queries": perf.total_queries,
                "success_rate": perf.success_queries / max(perf.total_queries, 1),
                "avg_confidence": perf.avg_confidence,
                "avg_execution_time": perf.avg_execution_time,
                "accuracy_score": perf.accuracy_score,
                "last_updated": perf.last_updated.isoformat()
            }
        
        return stats
    
    async def update_confidence_weights(self, weights: Dict[str, float]):
        """更新置信度权重"""
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.confidence_weights = {k: v / total_weight for k, v in weights.items()}
            logger.info(f"置信度权重已更新: {self.confidence_weights}")
    
    async def explain_reasoning(self, result: HybridReasoningResult) -> str:
        """生成推理解释"""
        explanation_parts = [
            f"查询: {result.query}",
            f"使用策略: {result.strategy_used.value}",
            f"总体置信度: {result.confidence:.3f}",
            f"执行时间: {result.execution_time:.3f}秒"
        ]
        
        if result.method_contributions:
            contribution_str = ", ".join([f"{k}: {v:.2f}" for k, v in result.method_contributions.items()])
            explanation_parts.append(f"方法贡献度: {contribution_str}")
        
        explanation_parts.append(f"证据数量: {len(result.evidences)}")
        explanation_parts.append(f"结果数量: {len(result.results)}")
        
        if result.uncertainty_analysis:
            explanation_parts.append(f"不确定性分析: 后验概率 {result.uncertainty_analysis.posterior_probability:.3f}")
        
        return "\n".join(explanation_parts)