"""解释生成引擎

本模块集成OpenAI API生成自然语言解释，支持多种解释风格和详细程度。
"""

import json
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Any, Dict, List, Optional, Union, Tuple
from uuid import uuid4
from src.ai.openai_client import OpenAIClient
from src.ai.explainer.decision_tracker import DecisionTracker
from src.ai.explainer.confidence_calculator import ConfidenceCalculator
from src.ai.explainer.models import ExplanationTemplate
from src.ai.explainer.cot_reasoning_explainer import CoTReasoningExplainer
from src.ai.explainer.workflow_explainer import WorkflowExplainer
from src.models.schemas.explanation import (
    DecisionExplanation,
    ExplanationComponent,
    ExplanationLevel,
    ExplanationType,
    EvidenceType,
    ConfidenceMetrics,
    CounterfactualScenario
)

class ExplanationGenerator:
    """解释生成引擎"""
    
    def __init__(self, openai_client: Optional[OpenAIClient] = None):
        """初始化解释生成器"""
        self.openai_client = openai_client or OpenAIClient()
        self.template_manager = ExplanationTemplate()
        self.confidence_calculator = ConfidenceCalculator()
        self.cot_explainer = CoTReasoningExplainer(openai_client)
        self.workflow_explainer = WorkflowExplainer(openai_client)
        
        # 解释生成配置
        self.generation_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.3,  # 较低温度确保一致性
            "max_tokens": 2000,
            "top_p": 0.9
        }
        
        # 解释风格模板
        self.style_templates = {
            "technical": {
                "tone": "技术性和精确",
                "audience": "技术专家或开发者",
                "focus": "算法逻辑、数据流和技术细节",
                "language": "专业术语和精确描述"
            },
            "business": {
                "tone": "商业导向和结果驱动",
                "audience": "业务决策者和管理人员",
                "focus": "业务影响、ROI和战略考量",
                "language": "商业术语和价值导向描述"
            },
            "user_friendly": {
                "tone": "友好和易懂",
                "audience": "普通用户",
                "focus": "实际影响和简单解释",
                "language": "日常语言和类比"
            },
            "regulatory": {
                "tone": "正式和合规导向",
                "audience": "审计人员和监管机构",
                "focus": "合规性、可审计性和风险控制",
                "language": "正式术语和规范描述"
            }
        }
    
    def generate_explanation(
        self,
        decision_tracker: DecisionTracker,
        explanation_type: ExplanationType,
        explanation_level: ExplanationLevel,
        style: str = "user_friendly",
        custom_context: Optional[Dict[str, Any]] = None,
        use_cot_reasoning: bool = False,
        reasoning_mode: str = "analytical"
    ) -> DecisionExplanation:
        """生成决策解释"""
        
        explanation_id = uuid4()
        generation_start = utc_now()
        
        try:
            # 如果启用CoT推理，使用CoT解释器
            if use_cot_reasoning or explanation_type == ExplanationType.REASONING:
                reasoning_chain, cot_explanation = self.cot_explainer.generate_cot_explanation(
                    decision_tracker, reasoning_mode, explanation_level
                )
                
                # 增强CoT解释的元数据
                if cot_explanation.metadata is None:
                    cot_explanation.metadata = {}
                cot_explanation.metadata.update({
                    "enhanced_by_standard_generator": True,
                    "reasoning_chain_id": reasoning_chain.chain_id,
                    "generation_style": style,
                    "generation_time_ms": int(
                        (utc_now() - generation_start).total_seconds() * 1000
                    )
                })
                
                return cot_explanation
            # 1. 收集决策数据
            decision_summary = decision_tracker.get_summary()
            decision_path = decision_tracker.get_decision_path()
            confidence_factors = decision_tracker.confidence_factors
            
            # 2. 计算置信度指标
            confidence_metrics = self._calculate_explanation_confidence(
                decision_tracker, custom_context
            )
            
            # 3. 生成解释组件
            explanation_components = self._generate_explanation_components(
                decision_tracker, confidence_metrics
            )
            
            # 4. 生成不同层次的解释文本
            explanations = self._generate_explanation_texts(
                decision_tracker,
                explanation_type,
                explanation_level,
                style,
                confidence_metrics,
                custom_context
            )
            
            # 5. 生成反事实场景
            counterfactuals = self._generate_counterfactual_scenarios(
                decision_tracker, confidence_metrics
            )
            
            # 6. 生成可视化数据
            visualization_data = self._generate_visualization_data(
                decision_tracker, confidence_metrics, explanation_components
            )
            
            # 7. 创建解释对象
            explanation = DecisionExplanation(
                id=explanation_id,
                decision_id=decision_tracker.decision_id,
                explanation_type=explanation_type,
                explanation_level=explanation_level,
                decision_description=decision_summary.get("decision_context", ""),
                decision_outcome=decision_summary.get("final_decision", ""),
                decision_context=self._format_decision_context(decision_tracker),
                summary_explanation=explanations["summary"],
                detailed_explanation=explanations.get("detailed"),
                technical_explanation=explanations.get("technical"),
                components=explanation_components,
                confidence_metrics=confidence_metrics,
                counterfactuals=counterfactuals,
                visualization_data=visualization_data,
                metadata={
                    "generation_style": style,
                    "generation_time_ms": int(
                        (utc_now() - generation_start).total_seconds() * 1000
                    ),
                    "model_version": self.generation_config["model"],
                    "decision_path_length": len(decision_path),
                    "confidence_factors_count": len(confidence_factors)
                }
            )
            
            return explanation
            
        except Exception as e:
            # 降级处理：生成基础解释
            return self._generate_fallback_explanation(
                explanation_id,
                decision_tracker,
                explanation_type,
                explanation_level,
                str(e)
            )
    
    def _generate_explanation_texts(
        self,
        decision_tracker: DecisionTracker,
        explanation_type: ExplanationType,
        explanation_level: ExplanationLevel,
        style: str,
        confidence_metrics: ConfidenceMetrics,
        custom_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """生成不同层次的解释文本"""
        
        explanations = {}
        
        # 准备上下文数据
        context_data = self._prepare_explanation_context(
            decision_tracker, confidence_metrics, custom_context
        )
        
        # 生成概要解释
        if explanation_level in [ExplanationLevel.SUMMARY, ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]:
            summary_prompt = self._build_explanation_prompt(
                ExplanationLevel.SUMMARY, explanation_type, style, context_data
            )
            explanations["summary"] = self._call_openai_for_explanation(summary_prompt)
        
        # 生成详细解释
        if explanation_level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]:
            detailed_prompt = self._build_explanation_prompt(
                ExplanationLevel.DETAILED, explanation_type, style, context_data
            )
            explanations["detailed"] = self._call_openai_for_explanation(detailed_prompt)
        
        # 生成技术解释
        if explanation_level == ExplanationLevel.TECHNICAL:
            technical_prompt = self._build_explanation_prompt(
                ExplanationLevel.TECHNICAL, explanation_type, style, context_data
            )
            explanations["technical"] = self._call_openai_for_explanation(technical_prompt)
        
        return explanations
    
    def _build_explanation_prompt(
        self,
        level: ExplanationLevel,
        explanation_type: ExplanationType,
        style: str,
        context_data: Dict[str, Any]
    ) -> str:
        """构建解释生成提示"""
        
        style_config = self.style_templates.get(style, self.style_templates["user_friendly"])
        
        prompt = f"""作为一个AI决策解释专家，请根据以下信息生成{level.value}级别的{explanation_type.value}解释。

解释风格要求：
- 目标受众：{style_config['audience']}
- 语调：{style_config['tone']}
- 重点关注：{style_config['focus']}
- 语言风格：{style_config['language']}

决策信息：
- 决策ID：{context_data['decision_id']}
- 最终决策：{context_data['final_decision']}
- 决策上下文：{context_data['decision_context']}
- 置信度：{context_data['overall_confidence']:.2%}

关键因素：
{self._format_key_factors(context_data['confidence_factors'])}

决策路径：
{self._format_decision_path(context_data['decision_path'])}

置信度分析：
{self._format_confidence_analysis(context_data['confidence_metrics'])}

请生成一个清晰、准确、有说服力的解释，说明：
1. 为什么做出这个决策
2. 关键影响因素及其重要性
3. 决策的可信度和局限性
"""

        # 根据解释层次添加特定要求
        if level == ExplanationLevel.SUMMARY:
            prompt += "\n要求：简洁明了，控制在150字以内，突出最关键的2-3个要点。"
        elif level == ExplanationLevel.DETAILED:
            prompt += "\n要求：详细分析，包含完整的推理过程，控制在500字以内。"
        elif level == ExplanationLevel.TECHNICAL:
            prompt += "\n要求：技术性解释，包含算法细节、数据流和技术参数，控制在800字以内。"
        
        return prompt
    
    def _call_openai_for_explanation(self, prompt: str) -> str:
        """调用OpenAI API生成解释"""
        try:
            response = self.openai_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **self.generation_config
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            # 降级到模板解释
            return f"解释生成失败，请稍后重试。错误：{str(e)}"
    
    def _generate_explanation_components(
        self,
        decision_tracker: DecisionTracker,
        confidence_metrics: ConfidenceMetrics
    ) -> List[ExplanationComponent]:
        """生成解释组件"""
        
        components = []
        
        # 从置信度因子生成组件
        for factor in decision_tracker.confidence_factors:
            component = ExplanationComponent(
                factor_name=factor["factor_name"],
                factor_value=factor["factor_value"],
                weight=factor["weight"],
                impact_score=factor["impact"],
                evidence_type=self._determine_evidence_type(factor),
                evidence_source=factor.get("source", "unknown"),
                evidence_content=self._generate_factor_explanation(factor),
                causal_relationship=f"{factor['factor_name']}影响决策结果",
                metadata={
                    "reliability_score": self._calculate_factor_reliability(factor),
                    "source_information": factor.get("source", "unknown")
                }
            )
            components.append(component)
        
        return components
    
    def _generate_counterfactual_scenarios(
        self,
        decision_tracker: DecisionTracker,
        confidence_metrics: ConfidenceMetrics
    ) -> List[CounterfactualScenario]:
        """生成反事实场景"""
        
        scenarios = []
        
        # 基于关键置信度因子生成反事实场景
        key_factors = sorted(
            decision_tracker.confidence_factors,
            key=lambda x: x["weight"] * x["impact"],
            reverse=True
        )[:3]  # 取前3个最重要的因子
        
        for factor in key_factors:
            scenario = CounterfactualScenario(
                scenario_name=f"{factor['factor_name']}变化场景",
                changed_factors={
                    factor['factor_name']: self._generate_alternative_value(factor)
                },
                predicted_outcome=self._predict_alternative_outcome(factor),
                probability=0.7,  # 默认概率
                impact_difference=self._estimate_confidence_change(factor, confidence_metrics),
                explanation=f"如果{factor['factor_name']}的值改变为{self._generate_alternative_value(factor)}，预计{self._predict_alternative_outcome(factor)}"
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _calculate_explanation_confidence(
        self,
        decision_tracker: DecisionTracker,
        custom_context: Optional[Dict[str, Any]] = None
    ) -> ConfidenceMetrics:
        """计算解释的置信度指标"""
        
        # 构建模型预测数据
        model_prediction = {
            "probability": 0.8,  # 基于决策最终状态
            "confidence": decision_tracker.final_decision is not None
        }
        
        # 转换置信度因子为证据数据
        evidence_data = []
        for factor in decision_tracker.confidence_factors:
            evidence_data.append({
                "factor_name": factor["factor_name"],
                "weight": factor["weight"],
                "reliability_score": 0.8,  # 默认可靠性
                "relevance_score": factor["impact"],
                "freshness_score": 0.9,  # 默认新鲜度
                "source": factor.get("source", "unknown")
            })
        
        return self.confidence_calculator.calculate_confidence_metrics(
            model_prediction,
            evidence_data,
            context_factors=custom_context
        )
    
    def _prepare_explanation_context(
        self,
        decision_tracker: DecisionTracker,
        confidence_metrics: ConfidenceMetrics,
        custom_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """准备解释上下文数据"""
        
        return {
            "decision_id": decision_tracker.decision_id,
            "final_decision": decision_tracker.final_decision or "待定",
            "decision_context": decision_tracker.decision_context,
            "overall_confidence": confidence_metrics.overall_confidence,
            "confidence_factors": decision_tracker.confidence_factors,
            "decision_path": decision_tracker.get_decision_path(),
            "confidence_metrics": confidence_metrics,
            "custom_context": custom_context or {}
        }
    
    def _format_decision_context(self, decision_tracker: DecisionTracker) -> str:
        """格式化决策上下文"""
        
        context_parts = []
        
        if decision_tracker.decision_context:
            context_parts.append(f"决策背景：{decision_tracker.decision_context}")
        
        if decision_tracker.data_sources:
            sources = ", ".join(decision_tracker.data_sources)
            context_parts.append(f"数据来源：{sources}")
        
        if decision_tracker.processing_steps:
            steps_count = len(decision_tracker.processing_steps)
            context_parts.append(f"处理步骤：{steps_count}个")
        
        return " | ".join(context_parts)
    
    def _format_key_factors(self, factors: List[Dict[str, Any]]) -> str:
        """格式化关键因素"""
        
        if not factors:
            return "无关键因素"
        
        factor_lines = []
        for i, factor in enumerate(factors[:5], 1):  # 最多显示5个
            impact = factor.get("impact", 0)
            weight = factor.get("weight", 0)
            factor_lines.append(
                f"{i}. {factor['factor_name']}: {factor['factor_value']} "
                f"(权重: {weight:.2f}, 影响: {impact:.2f})"
            )
        
        return "\n".join(factor_lines)
    
    def _format_decision_path(self, path: List[Dict[str, Any]]) -> str:
        """格式化决策路径"""
        
        if not path:
            return "无决策路径记录"
        
        path_lines = []
        for i, step in enumerate(path, 1):
            step_type = step.get("type", "unknown")
            description = step.get("description", "无描述")
            path_lines.append(f"{i}. {step_type}: {description}")
        
        return "\n".join(path_lines)
    
    def _format_confidence_analysis(self, metrics: ConfidenceMetrics) -> str:
        """格式化置信度分析"""
        
        analysis_parts = [
            f"整体置信度: {metrics.overall_confidence:.2%}",
            f"不确定性: {metrics.uncertainty_score:.2%}",
        ]
        
        if metrics.confidence_interval_lower and metrics.confidence_interval_upper:
            analysis_parts.append(
                f"置信区间: {metrics.confidence_interval_lower:.2%} - "
                f"{metrics.confidence_interval_upper:.2%}"
            )
        
        return " | ".join(analysis_parts)
    
    def _determine_evidence_type(self, factor: Dict[str, Any]) -> EvidenceType:
        """确定证据类型"""
        
        source = factor.get("source", "").lower()
        
        if "user" in source or "input" in source:
            return EvidenceType.INPUT_DATA
        elif "context" in source:
            return EvidenceType.CONTEXT
        elif "memory" in source:
            return EvidenceType.MEMORY
        elif "reasoning" in source:
            return EvidenceType.REASONING_STEP
        elif "api" in source or "external" in source:
            return EvidenceType.EXTERNAL_SOURCE
        else:
            return EvidenceType.DOMAIN_KNOWLEDGE
    
    def _calculate_factor_reliability(self, factor: Dict[str, Any]) -> float:
        """计算因子可靠性"""
        
        # 基于来源和权重估算可靠性
        weight = factor.get("weight", 0.5)
        impact = factor.get("impact", 0.5)
        
        # 简单的可靠性计算
        reliability = (weight + impact) / 2
        
        # 基于来源调整
        source = factor.get("source", "").lower()
        if "database" in source or "official" in source:
            reliability *= 1.1
        elif "user" in source or "manual" in source:
            reliability *= 0.9
        
        return min(1.0, reliability)
    
    def _generate_factor_explanation(self, factor: Dict[str, Any]) -> str:
        """生成因子解释"""
        
        factor_name = factor["factor_name"]
        factor_value = factor["factor_value"]
        weight = factor["weight"]
        impact = factor["impact"]
        
        if impact > 0.7:
            impact_desc = "强烈支持"
        elif impact > 0.4:
            impact_desc = "中等支持"
        else:
            impact_desc = "轻微影响"
        
        return f"{factor_name}为{factor_value}，{impact_desc}当前决策（权重{weight:.2f}）"
    
    def _generate_alternative_value(self, factor: Dict[str, Any]) -> Any:
        """生成替代值"""
        
        current_value = factor["factor_value"]
        
        # 简单的替代值生成逻辑
        if isinstance(current_value, (int, float)):
            return current_value * 0.8  # 降低20%
        elif isinstance(current_value, bool):
            return not current_value
        elif isinstance(current_value, str):
            return f"不同的{current_value}"
        else:
            return "替代值"
    
    def _predict_alternative_outcome(self, factor: Dict[str, Any]) -> str:
        """预测替代结果"""
        
        impact = factor.get("impact", 0.5)
        
        if impact > 0.7:
            return "可能会显著改变决策结果"
        elif impact > 0.4:
            return "可能会适度影响决策"
        else:
            return "对决策结果影响较小"
    
    def _estimate_confidence_change(
        self,
        factor: Dict[str, Any],
        base_confidence: ConfidenceMetrics
    ) -> float:
        """估算置信度变化"""
        
        weight = factor.get("weight", 0.5)
        impact = factor.get("impact", 0.5)
        
        # 估算置信度变化
        change_magnitude = weight * impact * 0.3  # 最多30%的变化
        
        return -change_magnitude  # 假设改变会降低置信度
    
    def _generate_visualization_data(
        self,
        decision_tracker: DecisionTracker,
        confidence_metrics: ConfidenceMetrics,
        components: List[ExplanationComponent]
    ) -> Dict[str, Any]:
        """生成可视化数据"""
        
        return {
            "factor_importance": {
                "chart_type": "bar",
                "data": [
                    {
                        "name": comp.factor_name,
                        "value": comp.weight * comp.impact_score,
                        "weight": comp.weight,
                        "impact": comp.impact_score
                    }
                    for comp in components
                ]
            },
            "confidence_breakdown": {
                "chart_type": "pie",
                "data": [
                    {
                        "label": "模型置信度",
                        "value": confidence_metrics.model_confidence or 0.5,
                        "color": "#4CAF50"
                    },
                    {
                        "label": "证据强度",
                        "value": confidence_metrics.evidence_confidence or 0.5,
                        "color": "#2196F3"
                    },
                    {
                        "label": "不确定性",
                        "value": confidence_metrics.uncertainty_score,
                        "color": "#FF9800"
                    }
                ]
            },
            "decision_path": {
                "chart_type": "flow",
                "nodes": [
                    {
                        "id": step.get("node_id", f"step_{i}"),
                        "label": step.get("description", "决策步骤"),
                        "type": step.get("type", "process")
                    }
                    for i, step in enumerate(decision_tracker.get_decision_path())
                ]
            }
        }
    
    def _generate_fallback_explanation(
        self,
        explanation_id: uuid4,
        decision_tracker: DecisionTracker,
        explanation_type: ExplanationType,
        explanation_level: ExplanationLevel,
        error: str
    ) -> DecisionExplanation:
        """生成降级解释"""
        
        # 使用模板生成基础解释
        template_kwargs = {
            "decision": decision_tracker.final_decision or "未知",
            "confidence": "中等",
            "factors": "系统分析的多个因素"
        }
        
        summary = self.template_manager.render_template(
            explanation_type, ExplanationLevel.SUMMARY, **template_kwargs
        )
        
        return DecisionExplanation(
            id=explanation_id,
            decision_id=decision_tracker.decision_id,
            explanation_type=explanation_type,
            explanation_level=explanation_level,
            decision_description=decision_tracker.decision_context or "",
            decision_outcome=decision_tracker.final_decision or "",
            summary_explanation=summary,
            components=[],
            confidence_metrics=ConfidenceMetrics(
                overall_confidence=0.5,
                uncertainty_score=0.8,
                confidence_sources=[]
            ),
            counterfactuals=[],
            metadata={
                "generation_error": error,
                "fallback_mode": True
            }
        )
    
    def generate_cot_reasoning_explanation(
        self,
        decision_tracker: DecisionTracker,
        reasoning_mode: str = "analytical",
        explanation_level: ExplanationLevel = ExplanationLevel.DETAILED
    ) -> Tuple[Any, DecisionExplanation]:
        """生成Chain-of-Thought推理解释
        
        Args:
            decision_tracker: 决策跟踪器
            reasoning_mode: 推理模式 (analytical, deductive, inductive, abductive)
            explanation_level: 解释详细程度
            
        Returns:
            Tuple[ReasoningChain, DecisionExplanation]: 推理链和解释对象
        """
        return self.cot_explainer.generate_cot_explanation(
            decision_tracker, reasoning_mode, explanation_level
        )
    
    def enhance_explanation_with_cot(
        self,
        base_explanation: DecisionExplanation,
        decision_tracker: DecisionTracker,
        reasoning_mode: str = "analytical"
    ) -> DecisionExplanation:
        """使用CoT推理增强现有解释"""
        
        try:
            # 生成CoT推理
            reasoning_chain, cot_explanation = self.cot_explainer.generate_cot_explanation(
                decision_tracker, reasoning_mode, base_explanation.explanation_level
            )
            
            # 合并解释内容
            enhanced_explanation = base_explanation.model_copy()
            
            # 增强概要解释
            if cot_explanation.summary_explanation:
                enhanced_explanation.summary_explanation = f"{base_explanation.summary_explanation}\n\n推理过程: {cot_explanation.summary_explanation}"
            
            # 增强详细解释
            if cot_explanation.detailed_explanation:
                enhanced_summary = enhanced_explanation.detailed_explanation or ""
                enhanced_explanation.detailed_explanation = f"{enhanced_summary}\n\n详细推理: {cot_explanation.detailed_explanation}"
            
            # 合并组件
            enhanced_explanation.components.extend(cot_explanation.components)
            
            # 合并反事实场景
            enhanced_explanation.counterfactuals.extend(cot_explanation.counterfactuals)
            
            # 更新可视化数据
            if enhanced_explanation.visualization_data is None:
                enhanced_explanation.visualization_data = {}
            
            if cot_explanation.visualization_data:
                enhanced_explanation.visualization_data.update(cot_explanation.visualization_data)
            
            # 更新元数据
            if enhanced_explanation.metadata is None:
                enhanced_explanation.metadata = {}
            
            enhanced_explanation.metadata.update({
                "enhanced_with_cot": True,
                "reasoning_mode": reasoning_mode,
                "reasoning_chain_id": reasoning_chain.chain_id,
                "reasoning_steps": len(reasoning_chain.steps),
                "enhancement_timestamp": utc_now().isoformat()
            })
            
            return enhanced_explanation
            
        except Exception as e:
            # 如果增强失败，返回原始解释并添加错误信息
            base_explanation.metadata = base_explanation.metadata or {}
            base_explanation.metadata["cot_enhancement_error"] = str(e)
            return base_explanation
    
    def generate_workflow_explanation(
        self,
        workflow_execution: Any,  # WorkflowExecution from workflow_explainer
        decision_tracker: Optional[DecisionTracker] = None,
        explanation_level: ExplanationLevel = ExplanationLevel.DETAILED
    ) -> DecisionExplanation:
        """生成工作流解释
        
        Args:
            workflow_execution: 工作流执行对象
            decision_tracker: 可选的决策跟踪器
            explanation_level: 解释详细程度
            
        Returns:
            DecisionExplanation: 工作流解释对象
        """
        return self.workflow_explainer.generate_workflow_explanation(
            workflow_execution, decision_tracker, explanation_level
        )
    
    def generate_comprehensive_explanation(
        self,
        decision_tracker: DecisionTracker,
        explanation_type: ExplanationType,
        explanation_level: ExplanationLevel,
        style: str = "user_friendly",
        custom_context: Optional[Dict[str, Any]] = None,
        include_cot_reasoning: bool = True,
        workflow_execution: Optional[Any] = None
    ) -> DecisionExplanation:
        """生成综合解释，整合所有解释器的能力
        
        Args:
            decision_tracker: 决策跟踪器
            explanation_type: 解释类型
            explanation_level: 解释详细程度
            style: 解释风格
            custom_context: 自定义上下文
            include_cot_reasoning: 是否包含CoT推理
            workflow_execution: 工作流执行对象（如果适用）
            
        Returns:
            DecisionExplanation: 综合解释对象
        """
        
        try:
            # 1. 生成基础解释
            base_explanation = self.generate_explanation(
                decision_tracker=decision_tracker,
                explanation_type=explanation_type,
                explanation_level=explanation_level,
                style=style,
                custom_context=custom_context,
                use_cot_reasoning=False  # 先生成基础解释
            )
            
            # 2. 如果需要，增强CoT推理
            if include_cot_reasoning and explanation_type != ExplanationType.WORKFLOW:
                base_explanation = self.enhance_explanation_with_cot(
                    base_explanation, decision_tracker
                )
            
            # 3. 如果是工作流类型或提供了工作流执行数据，生成工作流解释
            if explanation_type == ExplanationType.WORKFLOW and workflow_execution:
                workflow_explanation = self.generate_workflow_explanation(
                    workflow_execution, decision_tracker, explanation_level
                )
                return workflow_explanation
            
            # 4. 如果提供了工作流执行数据，将其整合到解释中
            if workflow_execution and explanation_type != ExplanationType.WORKFLOW:
                base_explanation = self._integrate_workflow_insights(
                    base_explanation, workflow_execution
                )
            
            # 5. 更新综合解释的元数据
            if base_explanation.metadata is None:
                base_explanation.metadata = {}
            
            base_explanation.metadata.update({
                "comprehensive_explanation": True,
                "included_cot_reasoning": include_cot_reasoning,
                "included_workflow_analysis": workflow_execution is not None,
                "generation_timestamp": utc_now().isoformat()
            })
            
            return base_explanation
            
        except Exception as e:
            # 降级处理
            return self._generate_fallback_explanation(
                uuid4(),
                decision_tracker,
                explanation_type,
                explanation_level,
                f"综合解释生成失败: {str(e)}"
            )
    
    def _integrate_workflow_insights(
        self,
        base_explanation: DecisionExplanation,
        workflow_execution: Any
    ) -> DecisionExplanation:
        """将工作流洞察整合到基础解释中"""
        
        try:
            # 生成工作流分析
            workflow_analysis = self.workflow_explainer._analyze_workflow_execution(workflow_execution)
            
            # 增强解释内容
            enhanced_explanation = base_explanation.model_copy()
            
            # 添加工作流上下文到详细解释
            if enhanced_explanation.detailed_explanation:
                workflow_context = f"\n\n工作流执行洞察:\n- 执行了{workflow_analysis['total_nodes']}个节点\n- 总执行时间: {workflow_analysis.get('total_execution_time', 0):.2f}秒\n- 成功率: {workflow_analysis['efficiency_metrics']['success_rate']:.1%}"
                enhanced_explanation.detailed_explanation += workflow_context
            
            # 添加工作流组件
            if len(workflow_execution.nodes) > 0:
                key_node = max(workflow_execution.nodes, key=lambda x: x.execution_time)
                workflow_component = ExplanationComponent(
                    factor_name=f"关键工作流节点: {key_node.node_name}",
                    factor_value=f"执行时间: {key_node.execution_time:.3f}s",
                    weight=0.2,
                    impact_score=0.6,
                    evidence_type=EvidenceType.WORKFLOW_EXECUTION,
                    evidence_source="workflow_analysis",
                    evidence_content=f"工作流节点{key_node.node_name}的执行对整体决策产生影响",
                    causal_relationship="工作流执行性能影响决策质量",
                    metadata={
                        "workflow_id": workflow_execution.workflow_id,
                        "node_type": key_node.node_type,
                        "execution_time": key_node.execution_time
                    }
                )
                enhanced_explanation.components.append(workflow_component)
            
            # 更新元数据
            if enhanced_explanation.metadata is None:
                enhanced_explanation.metadata = {}
            
            enhanced_explanation.metadata.update({
                "workflow_integration": True,
                "workflow_id": workflow_execution.workflow_id,
                "workflow_nodes": len(workflow_execution.nodes),
                "workflow_status": workflow_execution.status
            })
            
            return enhanced_explanation
            
        except Exception as e:
            # 如果整合失败，返回原始解释
            base_explanation.metadata = base_explanation.metadata or {}
            base_explanation.metadata["workflow_integration_error"] = str(e)
            return base_explanation
