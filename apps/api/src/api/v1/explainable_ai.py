"""可解释AI API端点"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.ai.explainer.explanation_generator import ExplanationGenerator
from src.ai.explainer.decision_tracker import DecisionTracker
from src.ai.explainer.workflow_explainer import WorkflowExecution, WorkflowNode
from models.schemas.explanation import (
    DecisionExplanation,
    ExplanationType,
    ExplanationLevel,
    EvidenceType,
    ConfidenceMetrics
)

router = APIRouter(prefix="/explainable-ai", tags=["Explainable AI"])

# 全局解释生成器实例
explanation_generator = ExplanationGenerator()


class ExplanationRequest(BaseModel):
    """解释请求模型"""
    decision_id: str = Field(..., description="决策ID")
    decision_context: str = Field(..., description="决策上下文")
    explanation_type: ExplanationType = Field(ExplanationType.DECISION, description="解释类型")
    explanation_level: ExplanationLevel = Field(ExplanationLevel.DETAILED, description="解释级别")
    style: str = Field("user_friendly", description="解释风格")
    factors: List[Dict[str, Any]] = Field(default_factory=list, description="置信度因子")
    use_cot_reasoning: bool = Field(False, description="是否使用CoT推理")
    reasoning_mode: str = Field("analytical", description="推理模式")


class CoTReasoningRequest(BaseModel):
    """CoT推理请求模型"""
    decision_id: str = Field(..., description="决策ID")
    decision_context: str = Field(..., description="决策上下文")
    reasoning_mode: str = Field("analytical", description="推理模式")
    explanation_level: ExplanationLevel = Field(ExplanationLevel.DETAILED, description="解释级别")
    factors: List[Dict[str, Any]] = Field(default_factory=list, description="置信度因子")


class WorkflowExecutionRequest(BaseModel):
    """工作流执行请求模型"""
    workflow_id: str = Field(..., description="工作流ID")
    workflow_name: str = Field(..., description="工作流名称")
    nodes: List[Dict[str, Any]] = Field(..., description="工作流节点")
    explanation_level: ExplanationLevel = Field(ExplanationLevel.DETAILED, description="解释级别")


class ExplanationFormatRequest(BaseModel):
    """解释格式化请求模型"""
    explanation_id: UUID = Field(..., description="解释ID")
    output_format: str = Field("html", description="输出格式")
    template_name: Optional[str] = Field(None, description="模板名称")


class DemoScenarioRequest(BaseModel):
    """演示场景请求模型"""
    scenario_type: str = Field(..., description="场景类型")
    complexity: str = Field("medium", description="复杂度")
    include_cot: bool = Field(True, description="包含CoT推理")


@router.post("/generate-explanation", response_model=DecisionExplanation)
async def generate_explanation(request: ExplanationRequest):
    """生成决策解释"""
    try:
        # 创建决策跟踪器
        tracker = DecisionTracker(request.decision_id, request.decision_context)
        
        # 添加置信度因子
        for factor in request.factors:
            tracker.add_confidence_factor(
                factor_name=factor.get("factor_name", "unknown"),
                factor_value=factor.get("factor_value", "unknown"),
                weight=float(factor.get("weight", 0.5)),
                impact=float(factor.get("impact", 0.5)),
                source=factor.get("source", "user_input")
            )
        
        # 完成决策
        tracker.finalize_decision(
            final_decision="completed",
            confidence_score=0.8,
            reasoning="决策基于提供的因子完成"
        )
        
        # 生成解释
        explanation = explanation_generator.generate_explanation(
            decision_tracker=tracker,
            explanation_type=request.explanation_type,
            explanation_level=request.explanation_level,
            style=request.style,
            use_cot_reasoning=request.use_cot_reasoning,
            reasoning_mode=request.reasoning_mode
        )
        
        return explanation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"解释生成失败: {str(e)}")


@router.post("/cot-reasoning", response_model=DecisionExplanation)
async def generate_cot_reasoning(request: CoTReasoningRequest):
    """生成CoT推理解释"""
    try:
        # 创建决策跟踪器
        tracker = DecisionTracker(request.decision_id, request.decision_context)
        
        # 添加置信度因子
        for factor in request.factors:
            tracker.add_confidence_factor(
                factor_name=factor.get("factor_name", "unknown"),
                factor_value=factor.get("factor_value", "unknown"),
                weight=float(factor.get("weight", 0.5)),
                impact=float(factor.get("impact", 0.5)),
                source=factor.get("source", "reasoning_input")
            )
        
        # 完成决策
        tracker.finalize_decision(
            final_decision="reasoning_completed",
            confidence_score=0.85,
            reasoning="CoT推理过程完成"
        )
        
        # 生成CoT推理解释
        reasoning_chain, explanation = explanation_generator.generate_cot_reasoning_explanation(
            decision_tracker=tracker,
            reasoning_mode=request.reasoning_mode,
            explanation_level=request.explanation_level
        )
        
        return explanation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CoT推理生成失败: {str(e)}")


@router.post("/workflow-explanation", response_model=DecisionExplanation)
async def generate_workflow_explanation(request: WorkflowExecutionRequest):
    """生成工作流解释"""
    try:
        # 构建工作流执行对象
        from ai.explainer.workflow_explainer import WorkflowExecution, WorkflowNode
        
        workflow_execution = WorkflowExecution(
            workflow_id=request.workflow_id,
            workflow_name=request.workflow_name
        )
        
        # 添加节点
        for node_data in request.nodes:
            node = WorkflowNode(
                node_id=node_data.get("node_id", "unknown"),
                node_type=node_data.get("node_type", "processor"),
                node_name=node_data.get("node_name", "Unknown Node"),
                input_data=node_data.get("input_data", {}),
                output_data=node_data.get("output_data", {}),
                execution_time=float(node_data.get("execution_time", 0.1)),
                status=node_data.get("status", "completed"),
                metadata=node_data.get("metadata", {})
            )
            workflow_execution.add_node_execution(node)
        
        # 完成执行
        workflow_execution.complete_execution("completed")
        
        # 生成工作流解释
        explanation = explanation_generator.generate_workflow_explanation(
            workflow_execution=workflow_execution,
            explanation_level=request.explanation_level
        )
        
        return explanation
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"工作流解释生成失败: {str(e)}")


@router.post("/format-explanation")
async def format_explanation(request: ExplanationFormatRequest):
    """格式化解释输出"""
    try:
        # 这里应该从数据库获取解释，目前创建一个示例
        from models.schemas.explanation import ConfidenceSource
        
        sample_explanation = DecisionExplanation(
            decision_id="sample_001",
            explanation_type=ExplanationType.DECISION,
            explanation_level=ExplanationLevel.DETAILED,
            decision_description="示例决策",
            decision_outcome="approved",
            summary_explanation="这是一个示例解释用于格式化测试",
            components=[],
            confidence_metrics=ConfidenceMetrics(
                overall_confidence=0.8,
                uncertainty_score=0.2,
                confidence_sources=[ConfidenceSource.MODEL_PROBABILITY]
            ),
            counterfactuals=[]
        )
        
        # 格式化解释
        formatted_content = explanation_generator.workflow_explainer.workflow_explainer._format_explanation(
            sample_explanation,
            request.output_format,
            request.template_name
        )
        
        return {
            "explanation_id": request.explanation_id,
            "format": request.output_format,
            "content": formatted_content
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"格式化失败: {str(e)}")


@router.post("/demo-scenario", response_model=DecisionExplanation)
async def generate_demo_scenario(request: DemoScenarioRequest):
    """生成演示场景"""
    try:
        # 根据场景类型创建不同的演示数据
        scenarios = {
            "loan_approval": {
                "decision_id": "loan_demo_001",
                "context": "贷款审批决策",
                "factors": [
                    {"factor_name": "credit_score", "factor_value": 750, "weight": 0.35, "impact": 0.8, "source": "credit_bureau"},
                    {"factor_name": "annual_income", "factor_value": 80000, "weight": 0.25, "impact": 0.7, "source": "payroll"},
                    {"factor_name": "employment_duration", "factor_value": 5, "weight": 0.2, "impact": 0.6, "source": "hr_system"},
                    {"factor_name": "debt_ratio", "factor_value": 0.25, "weight": 0.2, "impact": 0.5, "source": "financial_calc"}
                ]
            },
            "medical_diagnosis": {
                "decision_id": "medical_demo_001", 
                "context": "医疗诊断辅助决策",
                "factors": [
                    {"factor_name": "symptom_severity", "factor_value": "moderate", "weight": 0.4, "impact": 0.7, "source": "clinical_assessment"},
                    {"factor_name": "lab_results", "factor_value": "abnormal", "weight": 0.3, "impact": 0.8, "source": "laboratory"},
                    {"factor_name": "patient_history", "factor_value": "relevant", "weight": 0.2, "impact": 0.6, "source": "medical_records"},
                    {"factor_name": "imaging_findings", "factor_value": "positive", "weight": 0.1, "impact": 0.9, "source": "radiology"}
                ]
            },
            "investment_recommendation": {
                "decision_id": "investment_demo_001",
                "context": "投资建议决策", 
                "factors": [
                    {"factor_name": "market_trend", "factor_value": "bullish", "weight": 0.3, "impact": 0.7, "source": "market_analysis"},
                    {"factor_name": "risk_tolerance", "factor_value": "moderate", "weight": 0.25, "impact": 0.6, "source": "client_profile"},
                    {"factor_name": "portfolio_balance", "factor_value": "diversified", "weight": 0.25, "impact": 0.5, "source": "portfolio_analysis"},
                    {"factor_name": "economic_indicators", "factor_value": "stable", "weight": 0.2, "impact": 0.8, "source": "economic_data"}
                ]
            }
        }
        
        scenario_data = scenarios.get(request.scenario_type, scenarios["loan_approval"])
        
        # 根据复杂度调整因子数量
        if request.complexity == "simple":
            scenario_data["factors"] = scenario_data["factors"][:2]
        elif request.complexity == "complex":
            # 添加更多因子
            additional_factors = [
                {"factor_name": "external_factor_1", "factor_value": "positive", "weight": 0.1, "impact": 0.4, "source": "external_data"},
                {"factor_name": "external_factor_2", "factor_value": "neutral", "weight": 0.05, "impact": 0.3, "source": "external_data"}
            ]
            scenario_data["factors"].extend(additional_factors)
        
        # 创建解释请求
        explanation_request = ExplanationRequest(
            decision_id=scenario_data["decision_id"],
            decision_context=scenario_data["context"],
            explanation_type=ExplanationType.DECISION,
            explanation_level=ExplanationLevel.DETAILED,
            factors=scenario_data["factors"],
            use_cot_reasoning=request.include_cot,
            reasoning_mode="analytical"
        )
        
        # 生成解释
        return await generate_explanation(explanation_request)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"演示场景生成失败: {str(e)}")


@router.get("/explanation-types")
async def get_explanation_types():
    """获取支持的解释类型"""
    return {
        "explanation_types": [
            {"value": "decision", "label": "决策解释", "description": "解释决策过程和结果"},
            {"value": "reasoning", "label": "推理解释", "description": "详细的逐步推理过程"},
            {"value": "workflow", "label": "工作流解释", "description": "工作流执行过程解释"}
        ],
        "explanation_levels": [
            {"value": "summary", "label": "概要", "description": "简洁的解释摘要"},
            {"value": "detailed", "label": "详细", "description": "详细的解释内容"},
            {"value": "technical", "label": "技术", "description": "技术性深度解释"}
        ],
        "reasoning_modes": [
            {"value": "analytical", "label": "分析性推理", "description": "逐步分解分析"},
            {"value": "deductive", "label": "演绎推理", "description": "从一般到具体"},
            {"value": "inductive", "label": "归纳推理", "description": "从具体到一般"},
            {"value": "abductive", "label": "溯因推理", "description": "寻找最佳解释"}
        ],
        "output_formats": [
            {"value": "html", "label": "HTML", "description": "网页格式"},
            {"value": "markdown", "label": "Markdown", "description": "Markdown文档"},
            {"value": "json", "label": "JSON", "description": "结构化数据"},
            {"value": "text", "label": "纯文本", "description": "纯文本格式"},
            {"value": "xml", "label": "XML", "description": "XML文档"}
        ]
    }


@router.get("/demo-scenarios")
async def get_demo_scenarios():
    """获取可用的演示场景"""
    return {
        "scenarios": [
            {
                "type": "loan_approval",
                "name": "贷款审批",
                "description": "银行贷款审批决策场景",
                "complexity_levels": ["simple", "medium", "complex"]
            },
            {
                "type": "medical_diagnosis", 
                "name": "医疗诊断",
                "description": "医疗诊断辅助决策场景",
                "complexity_levels": ["simple", "medium", "complex"]
            },
            {
                "type": "investment_recommendation",
                "name": "投资建议",
                "description": "投资决策建议场景",
                "complexity_levels": ["simple", "medium", "complex"]
            }
        ]
    }


@router.get("/health")
async def health_check():
    """健康检查"""
    return {
        "status": "healthy",
        "service": "explainable-ai",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "explanation_generator": "active",
            "cot_reasoner": "active", 
            "workflow_explainer": "active",
            "formatter": "active"
        }
    }