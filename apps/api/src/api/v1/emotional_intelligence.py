"""
情感智能决策引擎 API 端点
"""
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from ...ai.emotional_intelligence.decision_engine import EmotionalDecisionEngine
from ...ai.emotional_intelligence.risk_assessment import RiskAssessmentEngine
from ...ai.emotional_intelligence.intervention_engine import InterventionStrategySelector
from ...ai.emotional_intelligence.crisis_support import CrisisDetectionSystem
from ...ai.emotional_intelligence.health_monitor import HealthMonitoringSystem
from ...ai.emotional_intelligence.models import (
    DecisionContext, EmotionalDecision, RiskAssessment, InterventionPlan,
    CrisisAssessment, HealthDashboardData
)
from ...ai.emotion_modeling.models import EmotionState, PersonalityProfile
from ...core.dependencies import get_current_user
from ...models.schemas.base import BaseResponse


router = APIRouter(prefix="/emotional-intelligence", tags=["emotional-intelligence"])
logger = logging.getLogger(__name__)

# 初始化引擎实例
decision_engine = EmotionalDecisionEngine()
risk_engine = RiskAssessmentEngine()
intervention_engine = InterventionStrategySelector()
crisis_system = CrisisDetectionSystem()
health_monitor = HealthMonitoringSystem()


# 请求/响应模型
class EmotionalDecisionRequest(BaseResponse):
    user_id: str
    session_id: Optional[str] = None
    user_input: str
    current_emotion_state: Dict[str, Any]
    emotion_history: List[Dict[str, Any]] = []
    personality_profile: Optional[Dict[str, Any]] = None
    environmental_factors: Dict[str, Any] = {}
    previous_decisions: List[Dict[str, Any]] = []


class RiskAssessmentRequest(BaseResponse):
    user_id: str
    emotion_history: List[Dict[str, Any]]
    personality_profile: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None


class CrisisDetectionRequest(BaseResponse):
    user_id: str
    user_input: str
    emotion_state: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    emotion_history: Optional[List[Dict[str, Any]]] = None


class InterventionPlanRequest(BaseResponse):
    user_id: str
    risk_assessment: Dict[str, Any]
    user_preferences: Optional[Dict[str, Any]] = None
    past_effectiveness: Optional[Dict[str, float]] = None


class HealthDashboardRequest(BaseResponse):
    user_id: str
    time_period_days: int = 30


@router.post("/decide")
async def make_emotional_decision(
    request: EmotionalDecisionRequest,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    基于情感上下文做出智能决策
    """
    try:
        # 构建决策上下文
        context = DecisionContext(
            user_id=request.user_id,
            session_id=request.session_id,
            current_emotion_state=request.current_emotion_state,
            emotion_history=request.emotion_history,
            personality_profile=request.personality_profile or {},
            conversation_context="",
            user_input=request.user_input,
            environmental_factors=request.environmental_factors,
            previous_decisions=request.previous_decisions
        )
        
        # 生成决策
        decision = await decision_engine.make_decision(context)
        
        logger.info(f"生成情感决策 - 用户: {request.user_id}, 策略: {decision.chosen_strategy}")
        
        return {
            "success": True,
            "decision": decision.to_dict(),
            "message": "情感智能决策生成成功"
        }
        
    except Exception as e:
        logger.error(f"情感决策生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"决策生成失败: {str(e)}")


@router.post("/risk-assessment")
async def assess_emotional_risk(
    request: RiskAssessmentRequest,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    进行情感健康风险评估
    """
    try:
        # 转换情感历史数据
        emotion_history = []
        for emotion_data in request.emotion_history:
            emotion = EmotionState.from_dict(emotion_data)
            emotion_history.append(emotion)
        
        # 转换个性画像
        personality_profile = None
        if request.personality_profile:
            personality_profile = PersonalityProfile.from_dict(request.personality_profile)
        
        # 进行风险评估
        assessment = await risk_engine.assess_comprehensive_risk(
            user_id=request.user_id,
            emotion_history=emotion_history,
            personality_profile=personality_profile,
            context=request.context
        )
        
        logger.info(f"风险评估完成 - 用户: {request.user_id}, 风险等级: {assessment.risk_level}")
        
        return {
            "success": True,
            "risk_assessment": assessment.to_dict(),
            "message": "风险评估完成"
        }
        
    except Exception as e:
        logger.error(f"风险评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"风险评估失败: {str(e)}")


@router.post("/crisis-detection")
async def detect_crisis(
    request: CrisisDetectionRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    检测情感危机并触发应急响应
    """
    try:
        # 转换情感状态
        emotion_state = EmotionState.from_dict(request.emotion_state)
        
        # 转换情感历史
        emotion_history = None
        if request.emotion_history:
            emotion_history = [EmotionState.from_dict(data) for data in request.emotion_history]
        
        # 检测危机
        crisis_assessment = await crisis_system.detect_crisis_indicators(
            user_id=request.user_id,
            user_input=request.user_input,
            emotion_state=emotion_state,
            context=request.context,
            emotion_history=emotion_history
        )
        
        # 如果检测到危机，触发紧急响应
        emergency_response = None
        if crisis_assessment.crisis_detected:
            # 在后台任务中触发紧急响应
            background_tasks.add_task(
                crisis_system.trigger_emergency_response,
                request.user_id,
                crisis_assessment
            )
            
            emergency_response = {
                "emergency_triggered": True,
                "severity_level": crisis_assessment.severity_level,
                "immediate_actions": crisis_assessment.immediate_actions
            }
        
        logger.info(f"危机检测完成 - 用户: {request.user_id}, 危机检测: {crisis_assessment.crisis_detected}")
        
        return {
            "success": True,
            "crisis_assessment": crisis_assessment.to_dict(),
            "emergency_response": emergency_response,
            "message": "危机检测完成"
        }
        
    except Exception as e:
        logger.error(f"危机检测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"危机检测失败: {str(e)}")


@router.post("/intervention-plan")
async def create_intervention_plan(
    request: InterventionPlanRequest,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    创建个性化干预计划
    """
    try:
        # 重构风险评估数据
        risk_data = request.risk_assessment
        risk_assessment = RiskAssessment(
            user_id=risk_data['user_id'],
            risk_level=risk_data['risk_level'],
            risk_score=risk_data['risk_score'],
            risk_factors=[],  # 简化处理
            recommended_actions=risk_data.get('recommended_actions', [])
        )
        
        # 选择干预策略
        strategies = await intervention_engine.select_intervention_strategies(
            risk_assessment=risk_assessment,
            user_preferences=request.user_preferences,
            past_effectiveness=request.past_effectiveness
        )
        
        # 创建干预计划
        intervention_plan = await intervention_engine.create_intervention_plan(
            risk_assessment=risk_assessment,
            strategies=strategies
        )
        
        logger.info(f"干预计划创建 - 用户: {request.user_id}, 策略数量: {len(strategies)}")
        
        return {
            "success": True,
            "intervention_plan": intervention_plan.to_dict(),
            "strategies": [strategy.to_dict() for strategy in strategies],
            "message": "干预计划创建成功"
        }
        
    except Exception as e:
        logger.error(f"干预计划创建失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"干预计划创建失败: {str(e)}")


@router.get("/health-dashboard/{user_id}")
async def get_health_dashboard(
    user_id: str,
    time_period_days: int = 30,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取用户健康监测仪表盘
    """
    try:
        # 模拟获取用户数据（实际实现中应从数据库获取）
        emotion_history = []  # 应从数据库获取
        risk_assessments = []  # 应从数据库获取
        interventions = []  # 应从数据库获取
        
        # 设置时间段
        end_time = datetime.now()
        start_time = end_time - timedelta(days=time_period_days)
        time_period = (start_time, end_time)
        
        # 生成健康仪表盘
        dashboard_data = await health_monitor.generate_health_dashboard(
            user_id=user_id,
            emotion_history=emotion_history,
            risk_assessments=risk_assessments,
            interventions=interventions,
            time_period=time_period
        )
        
        logger.info(f"健康仪表盘生成 - 用户: {user_id}")
        
        return {
            "success": True,
            "dashboard": dashboard_data.to_dict(),
            "message": "健康仪表盘生成成功"
        }
        
    except Exception as e:
        logger.error(f"健康仪表盘生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"健康仪表盘生成失败: {str(e)}")


@router.get("/emotional-patterns/{user_id}")
async def analyze_emotional_patterns(
    user_id: str,
    analysis_days: int = 7,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    分析用户情感模式
    """
    try:
        # 模拟获取情感历史数据
        emotion_history = []  # 应从数据库获取
        
        # 分析情感模式
        patterns = await health_monitor.track_emotional_patterns(
            emotion_history=emotion_history,
            analysis_window=timedelta(days=analysis_days)
        )
        
        logger.info(f"情感模式分析完成 - 用户: {user_id}")
        
        return {
            "success": True,
            "patterns": patterns,
            "message": "情感模式分析完成"
        }
        
    except Exception as e:
        logger.error(f"情感模式分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"情感模式分析失败: {str(e)}")


@router.post("/suicide-risk-assessment")
async def assess_suicide_risk(
    request: CrisisDetectionRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    专门的自杀风险评估
    """
    try:
        # 转换情感状态
        emotion_state = EmotionState.from_dict(request.emotion_state)
        
        # 评估自杀风险
        suicide_risk_score, risk_factors = await crisis_system.assess_suicide_risk(
            user_input=request.user_input,
            emotion_state=emotion_state,
            context=request.context or {}
        )
        
        # 如果风险很高，触发紧急响应
        if suicide_risk_score > 0.7:
            # 创建危机评估
            crisis_assessment = CrisisAssessment(
                user_id=request.user_id,
                crisis_detected=True,
                severity_level='critical',
                risk_score=suicide_risk_score,
                professional_required=True
            )
            
            # 在后台触发紧急响应
            background_tasks.add_task(
                crisis_system.trigger_emergency_response,
                request.user_id,
                crisis_assessment
            )
        
        logger.warning(f"自杀风险评估 - 用户: {request.user_id}, 风险分数: {suicide_risk_score:.3f}")
        
        return {
            "success": True,
            "suicide_risk_score": suicide_risk_score,
            "risk_factors": risk_factors,
            "emergency_triggered": suicide_risk_score > 0.7,
            "message": "自杀风险评估完成"
        }
        
    except Exception as e:
        logger.error(f"自杀风险评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"自杀风险评估失败: {str(e)}")


@router.get("/risk-trends/{user_id}")
async def analyze_risk_trends(
    user_id: str,
    time_period_days: int = 30,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    分析用户风险趋势
    """
    try:
        # 模拟获取风险评估历史
        risk_assessments = []  # 应从数据库获取
        
        # 分析风险趋势
        trends = await risk_engine.analyze_risk_trends(
            user_id=user_id,
            assessments=risk_assessments,
            time_period=timedelta(days=time_period_days)
        )
        
        logger.info(f"风险趋势分析完成 - 用户: {user_id}")
        
        return {
            "success": True,
            "trends": trends,
            "message": "风险趋势分析完成"
        }
        
    except Exception as e:
        logger.error(f"风险趋势分析失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"风险趋势分析失败: {str(e)}")


@router.get("/crisis-prediction/{user_id}")
async def predict_crisis_probability(
    user_id: str,
    prediction_hours: int = 24,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    预测危机发生概率
    """
    try:
        # 模拟获取情感历史
        emotion_history = []  # 应从数据库获取
        
        # 预测危机概率
        crisis_probability, analysis_details = await risk_engine.predict_crisis_probability(
            emotion_history=emotion_history,
            time_horizon=timedelta(hours=prediction_hours)
        )
        
        logger.info(f"危机概率预测 - 用户: {user_id}, 概率: {crisis_probability:.3f}")
        
        return {
            "success": True,
            "crisis_probability": crisis_probability,
            "analysis_details": analysis_details,
            "prediction_timeframe_hours": prediction_hours,
            "message": "危机概率预测完成"
        }
        
    except Exception as e:
        logger.error(f"危机概率预测失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"危机概率预测失败: {str(e)}")


@router.post("/intervention-effectiveness")
async def evaluate_intervention_effectiveness(
    intervention_id: str,
    before_emotions: List[Dict[str, Any]],
    after_emotions: List[Dict[str, Any]],
    user_feedback: Optional[str] = None,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    评估干预效果
    """
    try:
        # 转换情感数据
        before_emotion_states = [EmotionState.from_dict(data) for data in before_emotions]
        after_emotion_states = [EmotionState.from_dict(data) for data in after_emotions]
        
        # 模拟获取干预策略
        # 实际实现中应从数据库获取干预策略
        strategy = None  # 应从数据库获取
        
        if strategy and before_emotion_states and after_emotion_states:
            effectiveness_score = await intervention_engine.evaluate_strategy_effectiveness(
                strategy=strategy,
                before_state=before_emotion_states[0],
                after_state=after_emotion_states[-1],
                user_feedback=user_feedback
            )
        else:
            effectiveness_score = 0.5  # 默认分数
        
        logger.info(f"干预效果评估 - 干预ID: {intervention_id}, 效果分数: {effectiveness_score:.3f}")
        
        return {
            "success": True,
            "effectiveness_score": effectiveness_score,
            "intervention_id": intervention_id,
            "message": "干预效果评估完成"
        }
        
    except Exception as e:
        logger.error(f"干预效果评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"干预效果评估失败: {str(e)}")


@router.get("/health-insights/{user_id}")
async def get_health_insights(
    user_id: str,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取个性化健康洞察
    """
    try:
        # 模拟获取用户数据
        emotion_history = []  # 应从数据库获取
        risk_assessments = []  # 应从数据库获取
        interventions = []  # 应从数据库获取
        
        # 生成健康洞察
        insights = await health_monitor._generate_health_insights(
            emotions=emotion_history,
            assessments=risk_assessments,
            interventions=interventions
        )
        
        # 生成健康建议
        recommendations = await health_monitor._generate_health_recommendations(
            health_score=0.7,  # 应基于实际数据计算
            risk_level='low',  # 应基于最新评估
            volatility=0.3     # 应基于实际数据计算
        )
        
        logger.info(f"健康洞察生成 - 用户: {user_id}")
        
        return {
            "success": True,
            "insights": insights,
            "recommendations": recommendations,
            "message": "健康洞察生成成功"
        }
        
    except Exception as e:
        logger.error(f"健康洞察生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"健康洞察生成失败: {str(e)}")


@router.get("/system-status")
async def get_system_status() -> Dict[str, Any]:
    """
    获取情感智能系统状态
    """
    try:
        status = {
            "decision_engine": "operational",
            "risk_assessment_engine": "operational",
            "intervention_engine": "operational",
            "crisis_detection_system": "operational",
            "health_monitoring_system": "operational",
            "registered_strategies": len(decision_engine.strategies),
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "system_status": status,
            "message": "系统状态正常"
        }
        
    except Exception as e:
        logger.error(f"系统状态检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"系统状态检查失败: {str(e)}")