"""
检索过程解释器

提供检索决策过程的可解释性和透明度展示功能：
1. 检索决策过程记录和展示
2. 检索路径可视化和流程图生成  
3. 置信度评分和不确定性标记
4. 用户友好的检索结果解释界面
5. 检索失败原因分析和改进建议
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

from src.ai.agentic_rag.query_analyzer import QueryAnalysis, QueryIntent
from src.ai.agentic_rag.retrieval_agents import RetrievalResult, RetrievalStrategy, MultiAgentRetriever
from src.ai.agentic_rag.result_validator import ValidationResult, QualityScore, QualityDimension
from src.ai.agentic_rag.context_composer import ComposedContext
from src.core.config import get_settings
from src.ai.openai_client import get_openai_client


class ExplanationLevel(Enum):
    """解释详细程度级别"""
    SIMPLE = "simple"      # 简单解释，适合普通用户
    DETAILED = "detailed"  # 详细解释，包含技术细节
    TECHNICAL = "technical"  # 技术解释，适合开发者
    DEBUG = "debug"        # 调试信息，包含完整过程


class DecisionPoint(Enum):
    """关键决策点类型"""
    QUERY_ANALYSIS = "query_analysis"          # 查询分析决策
    STRATEGY_SELECTION = "strategy_selection"  # 策略选择决策
    RETRIEVAL_EXECUTION = "retrieval_execution"  # 检索执行决策
    RESULT_FUSION = "result_fusion"            # 结果融合决策
    QUALITY_VALIDATION = "quality_validation"  # 质量验证决策
    CONTEXT_COMPOSITION = "context_composition"  # 上下文组合决策


class ConfidenceLevel(Enum):
    """置信度级别"""
    VERY_HIGH = "very_high"  # 0.8-1.0
    HIGH = "high"            # 0.6-0.8
    MEDIUM = "medium"        # 0.4-0.6
    LOW = "low"              # 0.2-0.4
    VERY_LOW = "very_low"    # 0.0-0.2


@dataclass
class DecisionRecord:
    """决策记录"""
    decision_point: DecisionPoint
    timestamp: float
    input_data: Dict[str, Any]
    decision_made: Dict[str, Any]
    reasoning: str
    confidence: float
    alternatives_considered: List[Dict[str, Any]] = field(default_factory=list)
    execution_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class RetrievalPath:
    """检索路径"""
    path_id: str
    query_analysis: QueryAnalysis
    decisions: List[DecisionRecord] = field(default_factory=list)
    total_time: float = 0.0
    success_rate: float = 1.0
    final_results_count: int = 0
    path_visualization: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfidenceAnalysis:
    """置信度分析"""
    overall_confidence: float
    confidence_level: ConfidenceLevel
    confidence_breakdown: Dict[str, float] = field(default_factory=dict)
    uncertainty_sources: List[str] = field(default_factory=list)
    confidence_explanation: str = ""
    reliability_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class ExplanationOutput:
    """解释输出"""
    explanation_id: str
    query: str
    explanation_level: ExplanationLevel
    retrieval_path: RetrievalPath
    confidence_analysis: ConfidenceAnalysis
    
    # 用户友好的解释
    summary: str
    detailed_explanation: str
    decision_rationale: List[str] = field(default_factory=list)
    
    # 可视化数据
    flow_diagram: Dict[str, Any] = field(default_factory=dict)
    metrics_chart: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)
    
    # 改进建议
    improvement_suggestions: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)
    
    generated_at: datetime = field(default_factory=utc_factory)
    generation_time: float = 0.0


class RetrievalExplainer:
    """检索过程解释器"""
    
    def __init__(self):
        self.settings = get_settings()
        self.client = get_openai_client()
        
        # 解释模板配置
        self.explanation_templates = {
            ExplanationLevel.SIMPLE: {
                "summary_template": "根据您的查询\"{query}\"，系统执行了{strategy_count}种检索策略，找到了{result_count}个相关结果，整体质量{quality_level}。",
                "decision_template": "选择{strategy}策略是因为{reasoning}。"
            },
            ExplanationLevel.DETAILED: {
                "summary_template": "查询分析：{analysis_summary}。采用{strategy_list}策略进行检索，耗时{total_time:.2f}秒，检索到{result_count}个结果，经过质量验证后整体评分{quality_score:.2f}。",
                "decision_template": "在{decision_point}阶段，基于{input_summary}，决定{decision_made}，置信度{confidence:.2f}，原因：{reasoning}。"
            },
            ExplanationLevel.TECHNICAL: {
                "summary_template": "技术分析：查询意图{intent_type}，复杂度{complexity_score:.2f}。多代理协作使用{agent_types}，并行执行耗时{parallel_time:.2f}秒。结果融合采用{fusion_method}，最终上下文组合{context_tokens}个token。",
                "decision_template": "决策点{decision_point}：输入参数{input_params}，算法选择{algorithm}，输出{output_summary}，性能指标{performance_metrics}。"
            }
        }
        
        # 置信度阈值配置
        self.confidence_thresholds = {
            ConfidenceLevel.VERY_HIGH: 0.8,
            ConfidenceLevel.HIGH: 0.6,
            ConfidenceLevel.MEDIUM: 0.4,
            ConfidenceLevel.LOW: 0.2,
            ConfidenceLevel.VERY_LOW: 0.0
        }
        
        # 路径记录存储
        self.path_records: Dict[str, RetrievalPath] = {}
    
    def start_path_recording(self, query_analysis: QueryAnalysis) -> str:
        """开始记录检索路径"""
        import uuid
        path_id = f"path_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        self.path_records[path_id] = RetrievalPath(
            path_id=path_id,
            query_analysis=query_analysis
        )
        
        return path_id
    
    def record_decision(self, path_id: str, decision_record: DecisionRecord):
        """记录决策过程"""
        if path_id in self.path_records:
            self.path_records[path_id].decisions.append(decision_record)
    
    def finish_path_recording(self, path_id: str, total_time: float, results_count: int):
        """完成路径记录"""
        if path_id in self.path_records:
            path = self.path_records[path_id]
            path.total_time = total_time
            path.final_results_count = results_count
            
            # 计算成功率
            successful_decisions = sum(1 for d in path.decisions if d.success)
            path.success_rate = successful_decisions / len(path.decisions) if path.decisions else 0.0
            
            # 生成路径可视化数据
            path.path_visualization = self._generate_path_visualization(path)
    
    async def explain_retrieval_process(
        self,
        path_id: str,
        retrieval_results: List[RetrievalResult],
        validation_result: Optional[ValidationResult] = None,
        composed_context: Optional[ComposedContext] = None,
        explanation_level: ExplanationLevel = ExplanationLevel.DETAILED
    ) -> ExplanationOutput:
        """解释检索过程"""
        start_time = time.time()
        
        if path_id not in self.path_records:
            raise ValueError(f"Path record not found: {path_id}")
        
        path = self.path_records[path_id]
        explanation_id = f"explanation_{path_id}_{int(time.time())}"
        
        # 分析置信度
        confidence_analysis = await self._analyze_confidence(
            path, retrieval_results, validation_result
        )
        
        # 生成解释内容
        summary = self._generate_summary(path, retrieval_results, validation_result, explanation_level)
        detailed_explanation = await self._generate_detailed_explanation(
            path, retrieval_results, validation_result, composed_context, explanation_level
        )
        
        # 生成决策理由
        decision_rationale = self._extract_decision_rationale(path, explanation_level)
        
        # 生成可视化数据
        flow_diagram = self._generate_flow_diagram(path, retrieval_results)
        metrics_chart = self._generate_metrics_chart(path, retrieval_results, validation_result)
        timeline = self._generate_timeline(path)
        
        # 生成改进建议
        improvement_suggestions = await self._generate_improvement_suggestions(
            path, retrieval_results, validation_result, confidence_analysis
        )
        alternative_approaches = self._suggest_alternative_approaches(path, retrieval_results)
        
        generation_time = time.time() - start_time
        
        return ExplanationOutput(
            explanation_id=explanation_id,
            query=path.query_analysis.query_text,
            explanation_level=explanation_level,
            retrieval_path=path,
            confidence_analysis=confidence_analysis,
            summary=summary,
            detailed_explanation=detailed_explanation,
            decision_rationale=decision_rationale,
            flow_diagram=flow_diagram,
            metrics_chart=metrics_chart,
            timeline=timeline,
            improvement_suggestions=improvement_suggestions,
            alternative_approaches=alternative_approaches,
            generation_time=generation_time
        )
    
    async def _analyze_confidence(
        self,
        path: RetrievalPath,
        retrieval_results: List[RetrievalResult],
        validation_result: Optional[ValidationResult]
    ) -> ConfidenceAnalysis:
        """分析置信度"""
        
        # 计算各组件的置信度
        confidence_breakdown = {}
        uncertainty_sources = []
        
        # 查询分析置信度
        query_confidence = path.query_analysis.confidence
        confidence_breakdown["query_analysis"] = query_confidence
        if query_confidence < 0.7:
            uncertainty_sources.append("查询意图识别不够确定")
        
        # 检索结果置信度  
        if retrieval_results:
            retrieval_confidence = sum(r.confidence for r in retrieval_results) / len(retrieval_results)
            confidence_breakdown["retrieval"] = retrieval_confidence
            if retrieval_confidence < 0.6:
                uncertainty_sources.append("检索结果置信度较低")
        else:
            confidence_breakdown["retrieval"] = 0.0
            uncertainty_sources.append("没有检索到任何结果")
        
        # 验证结果置信度
        if validation_result:
            validation_confidence = validation_result.overall_confidence
            confidence_breakdown["validation"] = validation_confidence
            if validation_confidence < 0.6:
                uncertainty_sources.append("结果质量验证存在问题")
        else:
            confidence_breakdown["validation"] = 0.5
            uncertainty_sources.append("缺少质量验证信息")
        
        # 决策过程置信度
        if path.decisions:
            decision_confidence = sum(d.confidence for d in path.decisions) / len(path.decisions)
            confidence_breakdown["decisions"] = decision_confidence
            if decision_confidence < 0.7:
                uncertainty_sources.append("决策过程存在不确定性")
        else:
            confidence_breakdown["decisions"] = 0.5
        
        # 计算整体置信度（加权平均）
        weights = {
            "query_analysis": 0.25,
            "retrieval": 0.35,
            "validation": 0.25,
            "decisions": 0.15
        }
        
        overall_confidence = sum(
            confidence_breakdown.get(key, 0.0) * weight 
            for key, weight in weights.items()
        )
        
        # 确定置信度级别
        confidence_level = self._get_confidence_level(overall_confidence)
        
        # 分析可靠性因素
        reliability_factors = {
            "result_count": min(len(retrieval_results) / 10, 1.0),  # 结果数量
            "strategy_diversity": len(set(r.agent_type for r in retrieval_results)) / 3,  # 策略多样性
            "execution_success": path.success_rate,  # 执行成功率
            "time_efficiency": min(2.0 / path.total_time, 1.0) if path.total_time > 0 else 0.5  # 时间效率
        }
        
        # 生成置信度解释
        confidence_explanation = await self._generate_confidence_explanation(
            overall_confidence, confidence_breakdown, uncertainty_sources, reliability_factors
        )
        
        return ConfidenceAnalysis(
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            confidence_breakdown=confidence_breakdown,
            uncertainty_sources=uncertainty_sources,
            confidence_explanation=confidence_explanation,
            reliability_factors=reliability_factors
        )
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """获取置信度级别"""
        for level in [ConfidenceLevel.VERY_HIGH, ConfidenceLevel.HIGH, 
                     ConfidenceLevel.MEDIUM, ConfidenceLevel.LOW]:
            if confidence >= self.confidence_thresholds[level]:
                return level
        return ConfidenceLevel.VERY_LOW
    
    async def _generate_confidence_explanation(
        self,
        overall_confidence: float,
        breakdown: Dict[str, float],
        uncertainty_sources: List[str],
        reliability_factors: Dict[str, float]
    ) -> str:
        """生成置信度解释"""
        
        level_desc = {
            ConfidenceLevel.VERY_HIGH: "非常高",
            ConfidenceLevel.HIGH: "较高", 
            ConfidenceLevel.MEDIUM: "中等",
            ConfidenceLevel.LOW: "较低",
            ConfidenceLevel.VERY_LOW: "很低"
        }
        
        confidence_level = self._get_confidence_level(overall_confidence)
        
        explanation = f"整体置信度为{overall_confidence:.2f}（{level_desc[confidence_level]}）。"
        
        # 添加主要影响因素
        if breakdown["retrieval"] > 0.7:
            explanation += "检索结果质量良好，"
        elif breakdown["retrieval"] < 0.5:
            explanation += "检索结果质量有待提升，"
            
        if breakdown["validation"] > 0.7:
            explanation += "质量验证通过，"
        elif breakdown["validation"] < 0.5:
            explanation += "质量验证发现问题，"
        
        # 添加不确定性说明
        if uncertainty_sources:
            explanation += f"主要不确定性来源：{', '.join(uncertainty_sources[:3])}。"
        
        # 添加可靠性因素
        high_reliability = [k for k, v in reliability_factors.items() if v > 0.7]
        if high_reliability:
            factor_names = {
                "result_count": "结果数量充足",
                "strategy_diversity": "策略覆盖全面", 
                "execution_success": "执行成功率高",
                "time_efficiency": "响应时间理想"
            }
            explanation += f"优势：{', '.join(factor_names.get(f, f) for f in high_reliability)}。"
        
        return explanation
    
    def _generate_summary(
        self,
        path: RetrievalPath,
        retrieval_results: List[RetrievalResult],
        validation_result: Optional[ValidationResult],
        explanation_level: ExplanationLevel
    ) -> str:
        """生成摘要"""
        
        template = self.explanation_templates[explanation_level]["summary_template"]
        
        # 准备模板参数
        if explanation_level == ExplanationLevel.SIMPLE:
            strategy_count = len(set(r.agent_type for r in retrieval_results))
            result_count = sum(len(r.results) for r in retrieval_results)
            
            quality_level = "优秀"
            if validation_result:
                if validation_result.overall_quality >= 0.8:
                    quality_level = "优秀"
                elif validation_result.overall_quality >= 0.6:
                    quality_level = "良好"
                elif validation_result.overall_quality >= 0.4:
                    quality_level = "一般"
                else:
                    quality_level = "待改进"
            
            return template.format(
                query=path.query_analysis.query_text,
                strategy_count=strategy_count,
                result_count=result_count,
                quality_level=quality_level
            )
            
        elif explanation_level == ExplanationLevel.DETAILED:
            analysis_summary = f"{path.query_analysis.intent_type.value}类型查询，复杂度{path.query_analysis.complexity_score:.1f}"
            strategy_list = "、".join(set(r.agent_type.value for r in retrieval_results))
            result_count = sum(len(r.results) for r in retrieval_results)
            quality_score = validation_result.overall_quality if validation_result else 0.0
            
            return template.format(
                analysis_summary=analysis_summary,
                strategy_list=strategy_list,
                total_time=path.total_time,
                result_count=result_count,
                quality_score=quality_score
            )
            
        elif explanation_level == ExplanationLevel.TECHNICAL:
            agent_types = list(set(r.agent_type.value for r in retrieval_results))
            parallel_time = path.total_time  # 实际应该是并行时间
            fusion_method = "加权分数融合"  # 从MultiAgentRetriever获取
            context_tokens = 0  # 从ComposedContext获取
            
            return template.format(
                intent_type=path.query_analysis.intent_type.value,
                complexity_score=path.query_analysis.complexity_score,
                agent_types=", ".join(agent_types),
                parallel_time=parallel_time,
                fusion_method=fusion_method,
                context_tokens=context_tokens
            )
        
        return "检索过程摘要生成失败"
    
    async def _generate_detailed_explanation(
        self,
        path: RetrievalPath,
        retrieval_results: List[RetrievalResult],
        validation_result: Optional[ValidationResult],
        composed_context: Optional[ComposedContext],
        explanation_level: ExplanationLevel
    ) -> str:
        """生成详细解释"""
        
        explanation_parts = []
        
        # 查询分析阶段
        explanation_parts.append("## 查询分析阶段")
        explanation_parts.append(
            f"系统首先分析了您的查询\"{path.query_analysis.query_text}\"，"
            f"识别为{path.query_analysis.intent_type.value}类型，"
            f"提取了{len(path.query_analysis.keywords)}个关键词和{len(path.query_analysis.entities)}个实体。"
        )
        
        # 策略选择阶段
        strategy_decisions = [d for d in path.decisions if d.decision_point == DecisionPoint.STRATEGY_SELECTION]
        if strategy_decisions:
            explanation_parts.append("## 策略选择阶段")
            for decision in strategy_decisions:
                explanation_parts.append(f"- {decision.reasoning}")
        
        # 检索执行阶段
        explanation_parts.append("## 检索执行阶段")
        for result in retrieval_results:
            explanation_parts.append(
                f"- **{result.agent_type.value}代理**：检索到{len(result.results)}个结果，"
                f"平均得分{result.score:.2f}，置信度{result.confidence:.2f}"
            )
        
        # 结果验证阶段
        if validation_result:
            explanation_parts.append("## 结果验证阶段")
            explanation_parts.append(f"系统对检索结果进行了6个维度的质量评估：")
            for dimension, score in validation_result.quality_scores.items():
                explanation_parts.append(f"- {dimension.value}：{score.score:.2f} ({score.explanation})")
        
        # 上下文组合阶段
        if composed_context:
            explanation_parts.append("## 上下文组合阶段")
            total_fragments = composed_context.metadata.get('total_fragments', len(composed_context.selected_fragments))
            explanation_parts.append(
                f"从{total_fragments}个知识片段中选择了{len(composed_context.selected_fragments)}个最相关的片段，"
                f"总计{composed_context.total_tokens}个token，信息密度{composed_context.information_density:.2f}。"
            )
        
        return "\n\n".join(explanation_parts)
    
    def _extract_decision_rationale(self, path: RetrievalPath, explanation_level: ExplanationLevel) -> List[str]:
        """提取决策理由"""
        rationale = []
        
        template = self.explanation_templates[explanation_level]["decision_template"]
        
        for decision in path.decisions:
            if explanation_level == ExplanationLevel.SIMPLE:
                rationale.append(template.format(
                    strategy=decision.decision_made.get("strategy", "未知"),
                    reasoning=decision.reasoning
                ))
            elif explanation_level == ExplanationLevel.DETAILED:
                rationale.append(template.format(
                    decision_point=decision.decision_point.value,
                    input_summary=str(decision.input_data)[:100] + "...",
                    decision_made=str(decision.decision_made)[:100] + "...",
                    confidence=decision.confidence,
                    reasoning=decision.reasoning
                ))
            elif explanation_level == ExplanationLevel.TECHNICAL:
                rationale.append(template.format(
                    decision_point=decision.decision_point.value,
                    input_params=json.dumps(decision.input_data, ensure_ascii=False)[:200],
                    algorithm=decision.decision_made.get("algorithm", "default"),
                    output_summary=str(decision.decision_made)[:100],
                    performance_metrics=f"耗时{decision.execution_time:.3f}s"
                ))
        
        return rationale
    
    def _generate_flow_diagram(self, path: RetrievalPath, retrieval_results: List[RetrievalResult]) -> Dict[str, Any]:
        """生成流程图数据"""
        nodes = []
        edges = []
        
        # 添加查询节点
        nodes.append({
            "id": "query",
            "label": "用户查询",
            "type": "start",
            "data": {
                "query": path.query_analysis.query_text,
                "intent": path.query_analysis.intent_type.value
            }
        })
        
        # 添加分析节点
        nodes.append({
            "id": "analysis",
            "label": "查询分析",
            "type": "process",
            "data": {
                "complexity": path.query_analysis.complexity_score,
                "keywords": path.query_analysis.keywords,
                "entities": path.query_analysis.entities
            }
        })
        
        edges.append({"from": "query", "to": "analysis"})
        
        # 添加策略选择节点
        strategies = list(set(r.agent_type for r in retrieval_results))
        for i, strategy in enumerate(strategies):
            strategy_id = f"strategy_{i}"
            nodes.append({
                "id": strategy_id,
                "label": f"{strategy.value}检索",
                "type": "agent",
                "data": {
                    "strategy": strategy.value,
                    "results": [r for r in retrieval_results if r.agent_type == strategy]
                }
            })
            edges.append({"from": "analysis", "to": strategy_id})
        
        # 添加融合节点
        nodes.append({
            "id": "fusion",
            "label": "结果融合",
            "type": "process",
            "data": {
                "total_results": sum(len(r.results) for r in retrieval_results)
            }
        })
        
        for i, _ in enumerate(strategies):
            edges.append({"from": f"strategy_{i}", "to": "fusion"})
        
        # 添加输出节点
        nodes.append({
            "id": "output",
            "label": "最终结果",
            "type": "end",
            "data": {
                "success": path.success_rate >= 0.8
            }
        })
        
        edges.append({"from": "fusion", "to": "output"})
        
        return {
            "nodes": nodes,
            "edges": edges,
            "layout": "hierarchical"
        }
    
    def _generate_metrics_chart(
        self,
        path: RetrievalPath,
        retrieval_results: List[RetrievalResult],
        validation_result: Optional[ValidationResult]
    ) -> Dict[str, Any]:
        """生成指标图表数据"""
        
        charts = {}
        
        # 性能指标图表
        charts["performance"] = {
            "type": "bar",
            "title": "检索性能指标",
            "data": {
                "labels": ["查询分析", "策略选择", "并行检索", "结果融合", "质量验证"],
                "values": [
                    0.1,  # 查询分析时间
                    0.05, # 策略选择时间
                    max(r.processing_time for r in retrieval_results) if retrieval_results else 0,
                    0.2,  # 融合时间（估算）
                    0.1   # 验证时间（估算）
                ]
            }
        }
        
        # 质量评分图表
        if validation_result:
            charts["quality"] = {
                "type": "radar",
                "title": "结果质量评估",
                "data": {
                    "labels": [dim.value for dim in validation_result.quality_scores.keys()],
                    "values": [score.score for score in validation_result.quality_scores.values()]
                }
            }
        
        # 策略贡献图表
        if retrieval_results:
            strategy_contributions = {}
            for result in retrieval_results:
                strategy_contributions[result.agent_type.value] = len(result.results)
            
            charts["strategy_contribution"] = {
                "type": "pie", 
                "title": "各策略结果贡献",
                "data": {
                    "labels": list(strategy_contributions.keys()),
                    "values": list(strategy_contributions.values())
                }
            }
        
        return charts
    
    def _generate_timeline(self, path: RetrievalPath) -> List[Dict[str, Any]]:
        """生成时间线数据"""
        timeline = []
        
        # 添加查询开始事件
        timeline.append({
            "timestamp": 0,
            "event": "查询开始",
            "description": f"接收查询：{path.query_analysis.query_text}",
            "type": "start"
        })
        
        # 添加决策事件
        for decision in path.decisions:
            timeline.append({
                "timestamp": decision.timestamp,
                "event": decision.decision_point.value,
                "description": decision.reasoning,
                "success": decision.success,
                "confidence": decision.confidence,
                "type": "decision"
            })
        
        # 添加完成事件
        timeline.append({
            "timestamp": path.total_time,
            "event": "检索完成",
            "description": f"总耗时{path.total_time:.2f}秒，获得{path.final_results_count}个结果",
            "type": "end"
        })
        
        return sorted(timeline, key=lambda x: x["timestamp"])
    
    async def _generate_improvement_suggestions(
        self,
        path: RetrievalPath,
        retrieval_results: List[RetrievalResult],
        validation_result: Optional[ValidationResult],
        confidence_analysis: ConfidenceAnalysis
    ) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        # 基于置信度的建议
        if confidence_analysis.overall_confidence < 0.6:
            suggestions.append("整体置信度较低，建议：")
            
            if confidence_analysis.confidence_breakdown.get("query_analysis", 0) < 0.7:
                suggestions.append("- 尝试使用更明确的查询词语，避免歧义表达")
            
            if confidence_analysis.confidence_breakdown.get("retrieval", 0) < 0.6:
                suggestions.append("- 考虑扩展查询范围或使用同义词")
            
            if confidence_analysis.confidence_breakdown.get("validation", 0) < 0.6:
                suggestions.append("- 建议人工复查结果的准确性和相关性")
        
        # 基于质量评估的建议
        if validation_result:
            low_quality_dimensions = [
                dim for dim, score in validation_result.quality_scores.items() 
                if score.score < 0.6
            ]
            
            if QualityDimension.RELEVANCE in low_quality_dimensions:
                suggestions.append("相关性不足，建议使用更精确的关键词或添加上下文信息")
            
            if QualityDimension.COMPLETENESS in low_quality_dimensions:
                suggestions.append("信息完整性待提升，建议尝试多角度查询或细化查询范围")
            
            if QualityDimension.CONSISTENCY in low_quality_dimensions:
                suggestions.append("检测到信息不一致，建议查看冲突检测结果并选择可信来源")
        
        # 基于检索结果的建议
        if not retrieval_results:
            suggestions.append("未检索到结果，建议：")
            suggestions.append("- 简化查询词语，使用更通用的表达")
            suggestions.append("- 检查是否存在拼写错误")
            suggestions.append("- 尝试英文关键词查询")
        elif sum(len(r.results) for r in retrieval_results) < 3:
            suggestions.append("结果数量较少，建议扩大检索范围或使用更宽泛的查询词")
        
        # 基于性能的建议
        if path.total_time > 5.0:
            suggestions.append("检索耗时较长，建议简化查询或在非高峰时段使用")
        
        return suggestions
    
    def _suggest_alternative_approaches(
        self,
        path: RetrievalPath,
        retrieval_results: List[RetrievalResult]
    ) -> List[str]:
        """建议替代方法"""
        alternatives = []
        
        # 基于查询意图的建议
        if path.query_analysis.intent_type == QueryIntent.CODE:
            alternatives.append("尝试直接搜索具体的API或函数名")
            alternatives.append("查看官方文档或代码示例库")
            
        elif path.query_analysis.intent_type == QueryIntent.FACTUAL:
            alternatives.append("尝试搜索相关的定义或概念解释")
            alternatives.append("查找权威资料或学术论文")
            
        elif path.query_analysis.intent_type == QueryIntent.PROCEDURAL:
            alternatives.append("寻找步骤化的教程或指南")
            alternatives.append("查看相关的最佳实践文档")
        
        # 基于检索策略的建议
        used_strategies = set(r.agent_type for r in retrieval_results)
        all_strategies = set(RetrievalStrategy)
        unused_strategies = all_strategies - used_strategies
        
        if unused_strategies:
            strategy_names = [s.value for s in unused_strategies]
            alternatives.append(f"尝试使用{', '.join(strategy_names)}检索策略")
        
        # 基于查询复杂度的建议
        if path.query_analysis.complexity_score > 0.7:
            alternatives.append("将复杂查询拆分为多个简单的子查询")
            alternatives.append("使用分层查询方法，先查找概念再查找细节")
        
        return alternatives
    
    def _generate_path_visualization(self, path: RetrievalPath) -> Dict[str, Any]:
        """生成路径可视化数据"""
        
        visualization = {
            "path_id": path.path_id,
            "total_steps": len(path.decisions),
            "success_rate": path.success_rate,
            "execution_time": path.total_time,
            "steps": []
        }
        
        # 转换决策为可视化步骤
        for i, decision in enumerate(path.decisions):
            step = {
                "step_number": i + 1,
                "decision_point": decision.decision_point.value,
                "timestamp": decision.timestamp,
                "duration": decision.execution_time,
                "success": decision.success,
                "confidence": decision.confidence,
                "description": decision.reasoning[:100] + "..." if len(decision.reasoning) > 100 else decision.reasoning
            }
            
            if not decision.success and decision.error_message:
                step["error"] = decision.error_message
            
            visualization["steps"].append(step)
        
        return visualization
    
    def get_path_record(self, path_id: str) -> Optional[RetrievalPath]:
        """获取路径记录"""
        return self.path_records.get(path_id)
    
    def list_path_records(self) -> List[str]:
        """列出所有路径记录ID"""
        return list(self.path_records.keys())
    
    def clear_path_records(self):
        """清空路径记录"""
        self.path_records.clear()
    
    async def generate_debug_report(self, path_id: str) -> Dict[str, Any]:
        """生成调试报告"""
        if path_id not in self.path_records:
            return {"error": "Path record not found"}
        
        path = self.path_records[path_id]
        
        return {
            "path_id": path_id,
            "query": path.query_analysis.query_text,
            "analysis": {
                "intent_type": path.query_analysis.intent_type.value,
                "confidence": path.query_analysis.confidence,
                "complexity_score": path.query_analysis.complexity_score,
                "entities": path.query_analysis.entities,
                "keywords": path.query_analysis.keywords
            },
            "execution": {
                "total_time": path.total_time,
                "success_rate": path.success_rate,
                "final_results_count": path.final_results_count,
                "decision_count": len(path.decisions)
            },
            "decisions": [
                {
                    "decision_point": d.decision_point.value,
                    "timestamp": d.timestamp,
                    "success": d.success,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning,
                    "execution_time": d.execution_time,
                    "error": d.error_message
                }
                for d in path.decisions
            ],
            "path_visualization": path.path_visualization
        }