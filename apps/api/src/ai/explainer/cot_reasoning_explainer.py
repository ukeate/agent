"""Chain-of-Thought推理解释器

本模块集成思维链推理能力，为决策过程提供逐步推理解释。
"""

import json
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from src.ai.openai_client import OpenAIClient
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


class ReasoningStep:
    """推理步骤数据结构"""
    
    def __init__(
        self,
        step_id: str,
        step_type: str,
        description: str,
        input_data: Dict[str, Any],
        reasoning_process: str,
        output_data: Dict[str, Any],
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.step_id = step_id
        self.step_type = step_type
        self.description = description
        self.input_data = input_data
        self.reasoning_process = reasoning_process
        self.output_data = output_data
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = utc_now()


class ReasoningChain:
    """推理链数据结构"""
    
    def __init__(self, chain_id: str, problem_statement: str):
        self.chain_id = chain_id
        self.problem_statement = problem_statement
        self.steps: List[ReasoningStep] = []
        self.final_conclusion: Optional[str] = None
        self.overall_confidence: float = 0.0
        self.created_at = utc_now()
    
    def add_step(self, step: ReasoningStep) -> None:
        """添加推理步骤"""
        self.steps.append(step)
        # 更新整体置信度
        self._update_overall_confidence()
    
    def _update_overall_confidence(self) -> None:
        """更新整体置信度"""
        if not self.steps:
            self.overall_confidence = 0.0
            return
        
        # 使用加权平均计算整体置信度
        total_confidence = sum(step.confidence for step in self.steps)
        self.overall_confidence = total_confidence / len(self.steps)
    
    def get_reasoning_path(self) -> List[Dict[str, Any]]:
        """获取推理路径"""
        return [
            {
                "step_id": step.step_id,
                "step_type": step.step_type,
                "description": step.description,
                "reasoning": step.reasoning_process,
                "confidence": step.confidence,
                "timestamp": step.timestamp.isoformat()
            }
            for step in self.steps
        ]


class CoTReasoningExplainer:
    """Chain-of-Thought推理解释器"""
    
    def __init__(self, openai_client: Optional[OpenAIClient] = None):
        """初始化CoT推理解释器"""
        self.openai_client = openai_client or OpenAIClient()
        
        # CoT推理配置
        self.reasoning_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.2,  # 低温度确保逻辑一致性
            "max_tokens": 3000,
            "top_p": 0.9
        }
        
        # 推理模式配置
        self.reasoning_modes = {
            "analytical": {
                "description": "分析性推理",
                "approach": "逐步分解问题，系统性分析各个组成部分",
                "steps": ["问题分解", "证据收集", "因果分析", "结论推导"]
            },
            "deductive": {
                "description": "演绎推理", 
                "approach": "从一般原理出发，推导出具体结论",
                "steps": ["前提确立", "规则应用", "逻辑推导", "结论验证"]
            },
            "inductive": {
                "description": "归纳推理",
                "approach": "从具体观察归纳出一般规律",
                "steps": ["观察收集", "模式识别", "假设形成", "规律验证"]
            },
            "abductive": {
                "description": "溯因推理",
                "approach": "寻找最佳解释或原因",
                "steps": ["现象观察", "假设生成", "解释评估", "最佳推论"]
            }
        }
    
    def generate_cot_explanation(
        self,
        decision_tracker: DecisionTracker,
        reasoning_mode: str = "analytical",
        explanation_level: ExplanationLevel = ExplanationLevel.DETAILED
    ) -> Tuple[ReasoningChain, DecisionExplanation]:
        """生成CoT推理解释"""
        
        try:
            # 1. 创建推理链
            reasoning_chain = self._create_reasoning_chain(
                decision_tracker, reasoning_mode
            )
            
            # 2. 执行逐步推理
            self._execute_reasoning_steps(
                reasoning_chain, decision_tracker, reasoning_mode
            )
            
            # 3. 生成最终结论
            self._generate_final_conclusion(reasoning_chain, decision_tracker)
            
            # 4. 转换为解释对象
            explanation = self._convert_to_explanation(
                reasoning_chain, decision_tracker, explanation_level
            )
            
            return reasoning_chain, explanation
            
        except Exception as e:
            # 创建降级推理链
            fallback_chain = ReasoningChain(
                chain_id=str(uuid4()),
                problem_statement=f"推理生成失败: {str(e)}"
            )
            
            fallback_explanation = self._create_fallback_explanation(
                decision_tracker, str(e)
            )
            
            return fallback_chain, fallback_explanation
    
    def _create_reasoning_chain(
        self,
        decision_tracker: DecisionTracker,
        reasoning_mode: str
    ) -> ReasoningChain:
        """创建推理链"""
        
        chain_id = f"cot_{decision_tracker.decision_id}_{reasoning_mode}"
        
        # 构建问题陈述
        problem_statement = self._build_problem_statement(decision_tracker)
        
        return ReasoningChain(chain_id, problem_statement)
    
    def _build_problem_statement(self, decision_tracker: DecisionTracker) -> str:
        """构建问题陈述"""
        
        context = decision_tracker.decision_context or "未知背景"
        summary = decision_tracker.get_summary()
        
        return f"""决策问题: {context}
当前状态: {summary.get('current_state', '未知')}
目标: 理解并解释决策过程和结果
关键数据: {len(decision_tracker.confidence_factors)}个置信度因子
"""
    
    def _execute_reasoning_steps(
        self,
        reasoning_chain: ReasoningChain,
        decision_tracker: DecisionTracker,
        reasoning_mode: str
    ) -> None:
        """执行推理步骤"""
        
        mode_config = self.reasoning_modes.get(
            reasoning_mode, 
            self.reasoning_modes["analytical"]
        )
        
        for i, step_name in enumerate(mode_config["steps"]):
            step = self._execute_single_reasoning_step(
                step_name,
                reasoning_chain,
                decision_tracker,
                reasoning_mode,
                i + 1
            )
            reasoning_chain.add_step(step)
    
    def _execute_single_reasoning_step(
        self,
        step_name: str,
        reasoning_chain: ReasoningChain,
        decision_tracker: DecisionTracker,
        reasoning_mode: str,
        step_number: int
    ) -> ReasoningStep:
        """执行单个推理步骤"""
        
        step_id = f"step_{step_number}_{step_name.replace(' ', '_').lower()}"
        
        # 准备步骤输入数据
        input_data = self._prepare_step_input(
            step_name, decision_tracker, reasoning_chain
        )
        
        # 生成推理过程
        reasoning_process = self._generate_step_reasoning(
            step_name, input_data, reasoning_mode, step_number
        )
        
        # 生成输出数据
        output_data = self._generate_step_output(
            step_name, reasoning_process, input_data
        )
        
        # 计算步骤置信度
        confidence = self._calculate_step_confidence(
            step_name, input_data, output_data, reasoning_process
        )
        
        return ReasoningStep(
            step_id=step_id,
            step_type=step_name,
            description=f"{reasoning_mode}推理 - {step_name}",
            input_data=input_data,
            reasoning_process=reasoning_process,
            output_data=output_data,
            confidence=confidence,
            metadata={
                "reasoning_mode": reasoning_mode,
                "step_number": step_number,
                "generation_timestamp": utc_now().isoformat()
            }
        )
    
    def _prepare_step_input(
        self,
        step_name: str,
        decision_tracker: DecisionTracker,
        reasoning_chain: ReasoningChain
    ) -> Dict[str, Any]:
        """准备步骤输入数据"""
        
        base_input = {
            "decision_context": decision_tracker.decision_context,
            "confidence_factors": decision_tracker.confidence_factors,
            "decision_path": decision_tracker.get_decision_path(),
            "previous_steps": [step.output_data for step in reasoning_chain.steps]
        }
        
        # 根据步骤类型添加特定输入
        if "分解" in step_name or "问题" in step_name:
            base_input["problem_complexity"] = len(decision_tracker.confidence_factors)
            
        elif "证据" in step_name or "观察" in step_name:
            base_input["available_evidence"] = self._extract_evidence(decision_tracker)
            
        elif "分析" in step_name or "推导" in step_name:
            base_input["analytical_framework"] = self._get_analytical_framework()
            
        elif "结论" in step_name or "验证" in step_name:
            base_input["validation_criteria"] = self._get_validation_criteria()
        
        return base_input
    
    def _generate_step_reasoning(
        self,
        step_name: str,
        input_data: Dict[str, Any],
        reasoning_mode: str,
        step_number: int
    ) -> str:
        """生成步骤推理过程"""
        
        prompt = f"""作为AI推理专家，请进行{reasoning_mode}推理的第{step_number}步：{step_name}

问题背景：
{input_data.get('decision_context', '未知背景')}

可用数据：
{self._format_input_data(input_data)}

前序步骤结果：
{self._format_previous_steps(input_data.get('previous_steps', []))}

请按照{step_name}的要求进行推理，包括：
1. 当前步骤的目标和重点
2. 基于可用数据的逻辑分析
3. 推理过程中的关键洞察
4. 此步骤的结论或发现

要求：
- 逻辑清晰、层次分明
- 基于事实进行推理
- 明确指出不确定性
- 控制在300字以内
"""
        
        try:
            response = self.openai_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **self.reasoning_config
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"推理步骤生成失败: {str(e)}"
    
    def _generate_step_output(
        self,
        step_name: str,
        reasoning_process: str,
        input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成步骤输出数据"""
        
        output = {
            "step_conclusion": self._extract_conclusion_from_reasoning(reasoning_process),
            "key_insights": self._extract_insights_from_reasoning(reasoning_process),
            "identified_factors": self._extract_factors_from_reasoning(reasoning_process),
            "uncertainty_notes": self._extract_uncertainty_from_reasoning(reasoning_process)
        }
        
        # 根据步骤类型添加特定输出
        if "分解" in step_name:
            output["sub_problems"] = self._identify_sub_problems(reasoning_process)
            
        elif "证据" in step_name:
            output["evidence_quality"] = self._assess_evidence_quality(input_data)
            output["evidence_gaps"] = self._identify_evidence_gaps(reasoning_process)
            
        elif "分析" in step_name:
            output["causal_relationships"] = self._identify_causal_relationships(reasoning_process)
            output["risk_factors"] = self._identify_risk_factors(reasoning_process)
            
        elif "结论" in step_name:
            output["confidence_assessment"] = self._assess_conclusion_confidence(reasoning_process)
            output["alternative_scenarios"] = self._identify_alternatives(reasoning_process)
        
        return output
    
    def _calculate_step_confidence(
        self,
        step_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        reasoning_process: str
    ) -> float:
        """计算步骤置信度"""
        
        # 基础置信度评估
        base_confidence = 0.7
        
        # 基于输入数据质量调整
        data_quality = len(input_data.get("confidence_factors", [])) / 10.0
        data_quality = min(1.0, data_quality)
        
        # 基于推理过程长度和复杂度调整
        reasoning_complexity = min(1.0, len(reasoning_process) / 500.0)
        
        # 基于输出数据完整性调整
        output_completeness = len([v for v in output_data.values() if v]) / len(output_data)
        
        # 综合计算
        confidence = base_confidence * (0.4 * data_quality + 0.3 * reasoning_complexity + 0.3 * output_completeness)
        
        return min(1.0, max(0.1, confidence))
    
    def _generate_final_conclusion(
        self,
        reasoning_chain: ReasoningChain,
        decision_tracker: DecisionTracker
    ) -> None:
        """生成最终结论"""
        
        prompt = f"""基于以下逐步推理过程，请生成最终结论：

问题: {reasoning_chain.problem_statement}

推理步骤总结:
{self._format_reasoning_steps(reasoning_chain.steps)}

决策数据:
- 最终决策: {decision_tracker.final_decision or '待定'}
- 置信度因子数量: {len(decision_tracker.confidence_factors)}
- 整体推理置信度: {reasoning_chain.overall_confidence:.2%}

请生成包含以下内容的最终结论：
1. 基于推理过程的主要发现
2. 决策的合理性评估
3. 关键支撑证据
4. 不确定性和局限性
5. 建议和后续行动

要求简洁明了，控制在400字以内。
"""
        
        try:
            response = self.openai_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **self.reasoning_config
            )
            
            reasoning_chain.final_conclusion = response.choices[0].message.content.strip()
            
        except Exception as e:
            reasoning_chain.final_conclusion = f"结论生成失败: {str(e)}"
    
    def _convert_to_explanation(
        self,
        reasoning_chain: ReasoningChain,
        decision_tracker: DecisionTracker,
        explanation_level: ExplanationLevel
    ) -> DecisionExplanation:
        """转换为解释对象"""
        
        # 生成解释组件
        components = self._create_reasoning_components(reasoning_chain)
        
        # 计算置信度指标
        confidence_metrics = self._calculate_reasoning_confidence(reasoning_chain)
        
        # 生成反事实场景
        counterfactuals = self._generate_reasoning_counterfactuals(reasoning_chain)
        
        return DecisionExplanation(
            id=uuid4(),
            decision_id=decision_tracker.decision_id,
            explanation_type=ExplanationType.REASONING,
            explanation_level=explanation_level,
            decision_description=reasoning_chain.problem_statement,
            decision_outcome=decision_tracker.final_decision or "推理完成",
            decision_context=f"CoT推理链 ({len(reasoning_chain.steps)}步)",
            summary_explanation=self._generate_summary_explanation(reasoning_chain),
            detailed_explanation=reasoning_chain.final_conclusion,
            technical_explanation=self._generate_technical_explanation(reasoning_chain),
            components=components,
            confidence_metrics=confidence_metrics,
            counterfactuals=counterfactuals,
            visualization_data=self._generate_reasoning_visualization(reasoning_chain),
            metadata={
                "reasoning_type": "chain_of_thought",
                "reasoning_steps": len(reasoning_chain.steps),
                "overall_confidence": reasoning_chain.overall_confidence,
                "generation_timestamp": utc_now().isoformat()
            }
        )
    
    def _create_reasoning_components(
        self, 
        reasoning_chain: ReasoningChain
    ) -> List[ExplanationComponent]:
        """创建推理组件"""
        
        components = []
        
        for i, step in enumerate(reasoning_chain.steps):
            component = ExplanationComponent(
                factor_name=f"推理步骤{i+1}: {step.step_type}",
                factor_value=step.output_data.get("step_conclusion", "未知结论"),
                weight=1.0 / len(reasoning_chain.steps),
                impact_score=step.confidence,
                evidence_type=EvidenceType.REASONING_STEP,
                evidence_source="cot_reasoning",
                evidence_content=step.reasoning_process[:200] + "..." if len(step.reasoning_process) > 200 else step.reasoning_process,
                causal_relationship=f"推理步骤{i+1}支持最终结论",
                metadata={
                    "step_id": step.step_id,
                    "step_type": step.step_type,
                    "step_confidence": step.confidence
                }
            )
            components.append(component)
        
        return components
    
    def _calculate_reasoning_confidence(
        self, 
        reasoning_chain: ReasoningChain
    ) -> ConfidenceMetrics:
        """计算推理置信度指标"""
        
        from src.models.schemas.explanation import ConfidenceSource
        
        # 计算不确定性分数
        uncertainty_score = 1.0 - reasoning_chain.overall_confidence
        
        # 计算置信区间
        confidence_lower = max(0.0, reasoning_chain.overall_confidence - 0.1)
        confidence_upper = min(1.0, reasoning_chain.overall_confidence + 0.1)
        
        return ConfidenceMetrics(
            overall_confidence=reasoning_chain.overall_confidence,
            uncertainty_score=uncertainty_score,
            confidence_interval_lower=confidence_lower,
            confidence_interval_upper=confidence_upper,
            confidence_sources=[ConfidenceSource.REASONING_QUALITY],
            model_confidence=reasoning_chain.overall_confidence,
            evidence_confidence=reasoning_chain.overall_confidence,
            metadata={
                "reasoning_steps_count": len(reasoning_chain.steps),
                "average_step_confidence": reasoning_chain.overall_confidence,
                "confidence_source": "chain_of_thought_reasoning"
            }
        )
    
    def _generate_reasoning_counterfactuals(
        self, 
        reasoning_chain: ReasoningChain
    ) -> List[CounterfactualScenario]:
        """生成推理反事实场景"""
        
        scenarios = []
        
        # 为每个关键推理步骤生成反事实
        key_steps = sorted(
            reasoning_chain.steps,
            key=lambda x: x.confidence,
            reverse=True
        )[:2]  # 取置信度最高的2个步骤
        
        for step in key_steps:
            scenario = CounterfactualScenario(
                scenario_name=f"如果{step.step_type}推理不同",
                changed_factors={step.step_type: "不同的推理结果"},
                predicted_outcome="可能得出不同的结论",
                probability=1.0 - step.confidence,
                impact_difference=-(step.confidence - 0.5),
                explanation=f"如果在{step.step_type}阶段得出不同结论，整体推理结果可能发生改变"
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_summary_explanation(self, reasoning_chain: ReasoningChain) -> str:
        """生成概要解释"""
        
        step_count = len(reasoning_chain.steps)
        confidence_pct = reasoning_chain.overall_confidence * 100
        
        return f"通过{step_count}步Chain-of-Thought推理分析，整体置信度{confidence_pct:.1f}%。{reasoning_chain.final_conclusion[:100]}..."
    
    def _generate_technical_explanation(self, reasoning_chain: ReasoningChain) -> str:
        """生成技术解释"""
        
        technical_details = [
            f"推理链ID: {reasoning_chain.chain_id}",
            f"推理步骤: {len(reasoning_chain.steps)}个",
            f"平均步骤置信度: {reasoning_chain.overall_confidence:.3f}",
            f"推理模式: Chain-of-Thought",
            f"生成时间: {reasoning_chain.created_at.isoformat()}"
        ]
        
        step_details = []
        for i, step in enumerate(reasoning_chain.steps):
            step_details.append(
                f"步骤{i+1} ({step.step_type}): 置信度{step.confidence:.3f}"
            )
        
        return "\n".join(technical_details + ["", "步骤详情:"] + step_details)
    
    def _generate_reasoning_visualization(
        self, 
        reasoning_chain: ReasoningChain
    ) -> Dict[str, Any]:
        """生成推理可视化数据"""
        
        return {
            "reasoning_flow": {
                "chart_type": "flow",
                "nodes": [
                    {
                        "id": step.step_id,
                        "label": step.step_type,
                        "confidence": step.confidence,
                        "description": step.description
                    }
                    for step in reasoning_chain.steps
                ],
                "edges": [
                    {
                        "from": reasoning_chain.steps[i].step_id,
                        "to": reasoning_chain.steps[i+1].step_id,
                        "label": "推理链接"
                    }
                    for i in range(len(reasoning_chain.steps) - 1)
                ]
            },
            "confidence_progression": {
                "chart_type": "line",
                "data": [
                    {
                        "step": i+1,
                        "confidence": step.confidence,
                        "step_name": step.step_type
                    }
                    for i, step in enumerate(reasoning_chain.steps)
                ]
            }
        }
    
    def _create_fallback_explanation(
        self,
        decision_tracker: DecisionTracker,
        error: str
    ) -> DecisionExplanation:
        """创建降级解释"""
        
        return DecisionExplanation(
            id=uuid4(),
            decision_id=decision_tracker.decision_id,
            explanation_type=ExplanationType.REASONING,
            explanation_level=ExplanationLevel.SUMMARY,
            decision_description="CoT推理失败",
            decision_outcome=decision_tracker.final_decision or "未知",
            summary_explanation=f"Chain-of-Thought推理过程遇到错误: {error}",
            components=[],
            confidence_metrics=ConfidenceMetrics(
                overall_confidence=0.3,
                uncertainty_score=0.9,
                confidence_sources=[]
            ),
            counterfactuals=[],
            metadata={
                "error": error,
                "fallback_mode": True
            }
        )
    
    # 辅助方法
    def _extract_evidence(self, decision_tracker: DecisionTracker) -> List[Dict[str, Any]]:
        """提取证据数据"""
        return [
            {
                "factor_name": factor["factor_name"],
                "factor_value": factor["factor_value"],
                "source": factor.get("source", "unknown"),
                "reliability": factor.get("weight", 0.5)
            }
            for factor in decision_tracker.confidence_factors
        ]
    
    def _get_analytical_framework(self) -> Dict[str, str]:
        """获取分析框架"""
        return {
            "approach": "systematic_analysis",
            "criteria": ["relevance", "reliability", "consistency", "completeness"],
            "method": "factor_weighting_and_synthesis"
        }
    
    def _get_validation_criteria(self) -> List[str]:
        """获取验证标准"""
        return [
            "逻辑一致性",
            "证据支撑度",
            "结论合理性",
            "不确定性识别",
            "替代解释考虑"
        ]
    
    def _format_input_data(self, input_data: Dict[str, Any]) -> str:
        """格式化输入数据"""
        key_items = []
        for key, value in input_data.items():
            if key == "confidence_factors" and isinstance(value, list):
                key_items.append(f"- {key}: {len(value)}个因子")
            elif isinstance(value, (list, dict)):
                key_items.append(f"- {key}: {len(value)}项")
            else:
                key_items.append(f"- {key}: {str(value)[:50]}...")
        
        return "\n".join(key_items[:5])  # 限制显示条目
    
    def _format_previous_steps(self, previous_steps: List[Dict[str, Any]]) -> str:
        """格式化前序步骤"""
        if not previous_steps:
            return "无前序步骤"
        
        return "\n".join([
            f"步骤{i+1}: {step.get('step_conclusion', '无结论')}"
            for i, step in enumerate(previous_steps[-3:])  # 最多显示最近3步
        ])
    
    def _format_reasoning_steps(self, steps: List[ReasoningStep]) -> str:
        """格式化推理步骤"""
        return "\n".join([
            f"{i+1}. {step.step_type} (置信度: {step.confidence:.2f})\n   {step.output_data.get('step_conclusion', '无结论')}"
            for i, step in enumerate(steps)
        ])
    
    def _extract_conclusion_from_reasoning(self, reasoning: str) -> str:
        """从推理中提取结论"""
        # 简单的结论提取逻辑
        lines = reasoning.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['结论', '发现', '总结', '因此']):
                return line.strip()
        return reasoning.split('\n')[-1].strip() if lines else "无明确结论"
    
    def _extract_insights_from_reasoning(self, reasoning: str) -> List[str]:
        """从推理中提取洞察"""
        insights = []
        lines = reasoning.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['洞察', '发现', '重要', '关键']):
                insights.append(line.strip())
        return insights[:3]  # 最多3个洞察
    
    def _extract_factors_from_reasoning(self, reasoning: str) -> List[str]:
        """从推理中提取因子"""
        factors = []
        lines = reasoning.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['因素', '因子', '要素', '变量']):
                factors.append(line.strip())
        return factors[:5]  # 最多5个因子
    
    def _extract_uncertainty_from_reasoning(self, reasoning: str) -> List[str]:
        """从推理中提取不确定性"""
        uncertainties = []
        lines = reasoning.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['不确定', '可能', '风险', '限制', '假设']):
                uncertainties.append(line.strip())
        return uncertainties[:3]  # 最多3个不确定性
    
    def _identify_sub_problems(self, reasoning: str) -> List[str]:
        """识别子问题"""
        sub_problems = []
        lines = reasoning.split('\n')
        for line in lines:
            if any(keyword in line for keyword in ['问题', '需要', '分析', '考虑']):
                sub_problems.append(line.strip())
        return sub_problems[:3]
    
    def _assess_evidence_quality(self, input_data: Dict[str, Any]) -> Dict[str, float]:
        """评估证据质量"""
        factors = input_data.get("confidence_factors", [])
        if not factors:
            return {"overall": 0.5, "completeness": 0.3, "reliability": 0.5}
        
        reliability_scores = [factor.get("weight", 0.5) for factor in factors]
        avg_reliability = sum(reliability_scores) / len(reliability_scores)
        
        return {
            "overall": avg_reliability,
            "completeness": min(1.0, len(factors) / 5.0),
            "reliability": avg_reliability
        }
    
    def _identify_evidence_gaps(self, reasoning: str) -> List[str]:
        """识别证据缺口"""
        gaps = []
        if "缺乏" in reasoning or "不足" in reasoning:
            gaps.append("证据不充分")
        if "需要" in reasoning:
            gaps.append("需要额外数据")
        return gaps
    
    def _identify_causal_relationships(self, reasoning: str) -> List[Dict[str, str]]:
        """识别因果关系"""
        relationships = []
        if "导致" in reasoning:
            relationships.append({"type": "causal", "description": "因果关系已识别"})
        if "影响" in reasoning:
            relationships.append({"type": "influence", "description": "影响关系已识别"})
        return relationships
    
    def _identify_risk_factors(self, reasoning: str) -> List[str]:
        """识别风险因子"""
        risks = []
        if "风险" in reasoning:
            risks.append("已识别风险因子")
        if "不确定" in reasoning:
            risks.append("存在不确定性风险")
        return risks
    
    def _assess_conclusion_confidence(self, reasoning: str) -> float:
        """评估结论置信度"""
        confidence_keywords = ["确定", "明确", "清楚", "肯定"]
        uncertainty_keywords = ["可能", "或许", "不确定", "假设"]
        
        confidence_count = sum(1 for keyword in confidence_keywords if keyword in reasoning)
        uncertainty_count = sum(1 for keyword in uncertainty_keywords if keyword in reasoning)
        
        base_confidence = 0.7
        if confidence_count > uncertainty_count:
            return min(1.0, base_confidence + 0.2)
        elif uncertainty_count > confidence_count:
            return max(0.3, base_confidence - 0.2)
        else:
            return base_confidence
    
    def _identify_alternatives(self, reasoning: str) -> List[str]:
        """识别替代方案"""
        alternatives = []
        if "另外" in reasoning or "或者" in reasoning:
            alternatives.append("存在替代方案")
        if "其他" in reasoning:
            alternatives.append("有其他选择")
        return alternatives