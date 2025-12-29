"""
情感智能决策引擎 API 端点
"""

from src.core.utils.timezone_utils import utc_now
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Body
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from sqlalchemy import text
from src.ai.emotional_intelligence.decision_engine import EmotionalDecisionEngine
from src.ai.emotional_intelligence.risk_assessment import RiskAssessmentEngine
from src.ai.emotional_intelligence.intervention_engine import InterventionStrategySelector
from src.ai.emotional_intelligence.crisis_support import CrisisDetectionSystem
from src.ai.emotional_intelligence.health_monitor import HealthMonitoringSystem
from src.ai.emotional_intelligence.models import (
    DecisionContext, EmotionalDecision, RiskAssessment, InterventionPlan,
    CrisisAssessment, HealthDashboardData
)
from src.api.base_model import ApiBaseModel
from src.ai.emotion_modeling.models import EmotionState, PersonalityProfile
from src.core.dependencies import get_current_user
from src.core.database import get_db_session
from src.repositories.emotion_modeling_repository import EmotionModelingRepository

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/emotional-intelligence", tags=["emotional-intelligence"])

# 初始化引擎实例
decision_engine = EmotionalDecisionEngine()
risk_engine = RiskAssessmentEngine()
intervention_engine = InterventionStrategySelector()
crisis_system = CrisisDetectionSystem()
health_monitor = HealthMonitoringSystem()

system_start_time = utc_now()
intervention_plan_store: Dict[str, InterventionPlan] = {}
intervention_effectiveness_log: Dict[str, Dict[str, Any]] = {}
crisis_event_log: List[Dict[str, Any]] = []
emotional_intelligence_config: Dict[str, Any] = {
    "decision_confidence_threshold": 0.7,
    "risk_alert_threshold": 0.6,
    "crisis_response_delay_minutes": 1,
    "health_cache_ttl_hours": 1,
    "intervention_review_frequency_hours": 6
}

async def _load_emotion_history(
    user_id: str,
    limit: int = 200,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
) -> List[EmotionState]:
    async with get_db_session() as session:
        repo = EmotionModelingRepository(session)
        history = await repo.get_user_emotion_history(
            user_id=user_id,
            limit=limit,
            start_time=start_time,
            end_time=end_time
        )
    return sorted(history, key=lambda item: item.timestamp)

def _group_emotions_by_day(emotion_history: List[EmotionState]) -> Dict[str, List[EmotionState]]:
    grouped: Dict[str, List[EmotionState]] = defaultdict(list)
    for item in emotion_history:
        grouped[item.timestamp.date().isoformat()].append(item)
    return grouped

async def _build_daily_risk_assessments(
    user_id: str,
    emotion_history: List[EmotionState]
) -> List[RiskAssessment]:
    if not emotion_history:
        return []
    grouped = _group_emotions_by_day(emotion_history)
    assessments: List[RiskAssessment] = []
    for day, items in sorted(grouped.items()):
        assessment = await risk_engine.assess_comprehensive_risk(
            user_id=user_id,
            emotion_history=items,
            personality_profile=None,
            context={"period": day}
        )
        try:
            assessment.timestamp = datetime.fromisoformat(day)
        except ValueError:
            assessment.timestamp = utc_now()
        assessments.append(assessment)
    return assessments

def _calculate_intervention_effectiveness(
    before_emotions: List[EmotionState],
    after_emotions: List[EmotionState]
) -> Dict[str, Any]:
    before_valence = sum(item.valence for item in before_emotions) / len(before_emotions)
    after_valence = sum(item.valence for item in after_emotions) / len(after_emotions)
    before_intensity = sum(item.intensity for item in before_emotions) / len(before_emotions)
    after_intensity = sum(item.intensity for item in after_emotions) / len(after_emotions)
    valence_delta = after_valence - before_valence
    intensity_delta = before_intensity - after_intensity
    raw_score = 0.5 + 0.25 * (valence_delta / 2.0) + 0.25 * intensity_delta
    effectiveness_score = max(0.0, min(1.0, raw_score))
    return {
        "effectiveness_score": effectiveness_score,
        "valence_delta": valence_delta,
        "intensity_delta": intensity_delta,
        "before_valence": before_valence,
        "after_valence": after_valence,
        "before_intensity": before_intensity,
        "after_intensity": after_intensity
    }

async def _get_high_risk_users(
    threshold: float = 0.7,
    lookback_days: int = 30,
    limit: int = 50
) -> List[Dict[str, Any]]:
    end_time = utc_now()
    start_time = end_time - timedelta(days=lookback_days)
    high_risk_users: List[Dict[str, Any]] = []
    try:
        async with get_db_session() as session:
            result = await session.execute(
                text("SELECT DISTINCT user_id FROM emotion_states WHERE timestamp >= :start_time"),
                {"start_time": start_time}
            )
            user_ids = [row[0] for row in result.fetchall()]
            repo = EmotionModelingRepository(session)

            for user_id in user_ids:
                history = await repo.get_user_emotion_history(
                    user_id=user_id,
                    limit=200,
                    start_time=start_time,
                    end_time=end_time
                )
                if not history:
                    continue
                assessment = await risk_engine.assess_comprehensive_risk(
                    user_id=user_id,
                    emotion_history=history,
                    personality_profile=None,
                    context=None
                )
                if assessment.risk_score >= threshold:
                    high_risk_users.append({
                        "user_id": user_id,
                        "risk_score": assessment.risk_score,
                        "risk_level": assessment.risk_level
                    })
    except Exception as e:
        logger.warning(f"高风险用户查询失败，返回空列表: {e}")
        return []
    high_risk_users.sort(key=lambda item: item["risk_score"], reverse=True)
    return high_risk_users[:limit]

# 请求/响应模型
class EmotionalDecisionRequest(ApiBaseModel):
    user_id: str
    session_id: Optional[str] = None
    user_input: str
    current_emotion_state: Dict[str, Any]
    emotion_history: List[Dict[str, Any]] = []
    personality_profile: Optional[Dict[str, Any]] = None
    environmental_factors: Dict[str, Any] = {}
    previous_decisions: List[Dict[str, Any]] = []

class RiskAssessmentRequest(ApiBaseModel):
    user_id: str
    emotion_history: List[Dict[str, Any]]
    personality_profile: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None

class CrisisDetectionRequest(ApiBaseModel):
    user_id: str
    user_input: str
    emotion_state: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None
    emotion_history: Optional[List[Dict[str, Any]]] = None

class InterventionPlanRequest(ApiBaseModel):
    user_id: str
    risk_assessment: Dict[str, Any]
    user_preferences: Optional[Dict[str, Any]] = None
    past_effectiveness: Optional[Dict[str, float]] = None

class HealthDashboardRequest(ApiBaseModel):
    user_id: str
    time_period_days: int = 30

@router.get("/decisions/history")
async def get_decision_history(
    limit: int = 200,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    返回内存中的决策历史记录，真实来源于 EmotionalDecisionEngine.decision_history
    """
    try:
        history = decision_engine.decision_history
        if user_id:
            history = [item for item in history if item.user_id == user_id]
        history = history[-limit:]
        return {
            "count": len(history),
            "decisions": [d.to_dict() for d in history]
        }
    except Exception as e:
        logger.error(f"获取决策历史失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取决策历史失败: {str(e)}")

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
        emotion_history: List[EmotionState] = []
        if request.emotion_history:
            for emotion_data in request.emotion_history:
                try:
                    emotion = EmotionState.from_dict(emotion_data)
                except KeyError as e:
                    raise HTTPException(status_code=422, detail=f"emotion_history缺少字段: {e}")
                emotion_history.append(emotion)
        else:
            emotion_history = await _load_emotion_history(
                user_id=request.user_id,
                limit=200
            )
        
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
        
    except HTTPException:
        raise
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
        try:
            emotion_state = EmotionState.from_dict(request.emotion_state)
        except KeyError as e:
            raise HTTPException(status_code=422, detail=f"emotion_state缺少字段: {e}")
        
        # 转换情感历史
        emotion_history = None
        if request.emotion_history:
            emotion_history = []
            for data in request.emotion_history:
                try:
                    emotion_history.append(EmotionState.from_dict(data))
                except KeyError as e:
                    raise HTTPException(status_code=422, detail=f"emotion_history缺少字段: {e}")
        
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
            crisis_event_log.append({
                "event_id": crisis_assessment.assessment_id,
                "user_id": request.user_id,
                "timestamp": utc_now(),
                "severity_level": crisis_assessment.severity_level,
                "risk_score": crisis_assessment.risk_score,
                "acknowledged": False
            })
        
        logger.info(f"危机检测完成 - 用户: {request.user_id}, 危机检测: {crisis_assessment.crisis_detected}")
        
        return {
            "success": True,
            "crisis_assessment": crisis_assessment.to_dict(),
            "emergency_response": emergency_response,
            "message": "危机检测完成"
        }
        
    except HTTPException:
        raise
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
        try:
            risk_assessment = RiskAssessment(
                user_id=risk_data["user_id"],
                risk_level=risk_data["risk_level"],
                risk_score=risk_data["risk_score"],
                risk_factors=[],
                recommended_actions=risk_data.get("recommended_actions", []),
            )
        except KeyError as e:
            raise HTTPException(status_code=422, detail=f"risk_assessment缺少字段: {e}")
        
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
        intervention_plan_store[intervention_plan.plan_id] = intervention_plan
        
        logger.info(f"干预计划创建 - 用户: {request.user_id}, 策略数量: {len(strategies)}")
        
        return {
            "success": True,
            "intervention_plan": intervention_plan.to_dict(),
            "strategies": [strategy.to_dict() for strategy in strategies],
            "message": "干预计划创建成功"
        }
        
    except HTTPException:
        raise
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
        end_time = utc_now()
        start_time = end_time - timedelta(days=time_period_days)
        emotion_history = await _load_emotion_history(
            user_id=user_id,
            limit=1000,
            start_time=start_time,
            end_time=end_time
        )

        risk_assessments = await _build_daily_risk_assessments(user_id, emotion_history)
        if not risk_assessments and emotion_history:
            risk_assessments = [
                await risk_engine.assess_comprehensive_risk(
                    user_id=user_id,
                    emotion_history=emotion_history,
                    personality_profile=None,
                    context=None
                )
            ]

        interventions = [
            plan for plan in intervention_plan_store.values()
            if plan.user_id == user_id
        ]

        async with get_db_session() as session:
            repo = EmotionModelingRepository(session)
            personality_profile = await repo.get_personality_profile(user_id)

        dashboard = await health_monitor.generate_health_dashboard(
            user_id=user_id,
            emotion_history=emotion_history,
            risk_assessments=risk_assessments,
            interventions=interventions,
            personality_profile=personality_profile,
            time_period=(start_time, end_time)
        )

        return {
            "success": True,
            "dashboard": dashboard.to_dict(),
            "message": "健康仪表盘生成完成"
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
        end_time = utc_now()
        start_time = end_time - timedelta(days=analysis_days)
        emotion_history = await _load_emotion_history(
            user_id=user_id,
            limit=500,
            start_time=start_time,
            end_time=end_time
        )

        result = await health_monitor.track_emotional_patterns(
            emotion_history=emotion_history,
            analysis_window=timedelta(days=analysis_days)
        )
        return {
            "success": True,
            "patterns": result.get("patterns", []),
            "trends": result.get("trends", {}),
            "analysis_period": result.get("analysis_period", {}),
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
        try:
            emotion_state = EmotionState.from_dict(request.emotion_state)
        except KeyError as e:
            raise HTTPException(status_code=422, detail=f"emotion_state缺少字段: {e}")
        
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
            crisis_event_log.append({
                "event_id": crisis_assessment.assessment_id,
                "user_id": request.user_id,
                "timestamp": utc_now(),
                "severity_level": "critical",
                "risk_score": suicide_risk_score,
                "acknowledged": False
            })
        
        logger.warning(f"自杀风险评估 - 用户: {request.user_id}, 风险分数: {suicide_risk_score:.3f}")
        
        return {
            "success": True,
            "suicide_risk_score": suicide_risk_score,
            "risk_factors": risk_factors,
            "emergency_triggered": suicide_risk_score > 0.7,
            "message": "自杀风险评估完成"
        }
        
    except HTTPException:
        raise
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
        end_time = utc_now()
        start_time = end_time - timedelta(days=time_period_days)
        emotion_history = await _load_emotion_history(
            user_id=user_id,
            limit=1000,
            start_time=start_time,
            end_time=end_time
        )
        assessments = await _build_daily_risk_assessments(user_id, emotion_history)
        trends = await risk_engine.analyze_risk_trends(
            user_id=user_id,
            assessments=assessments,
            time_period=timedelta(days=time_period_days)
        )
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
        end_time = utc_now()
        start_time = end_time - timedelta(days=30)
        emotion_history = await _load_emotion_history(
            user_id=user_id,
            limit=500,
            start_time=start_time,
            end_time=end_time
        )
        crisis_probability, analysis_details = await risk_engine.predict_crisis_probability(
            emotion_history=emotion_history,
            time_horizon=timedelta(hours=prediction_hours)
        )
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
    intervention_id: str = Body(...),
    before_emotions: List[Dict[str, Any]] = Body(...),
    after_emotions: List[Dict[str, Any]] = Body(...),
    user_feedback: Optional[str] = Body(None),
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    评估干预效果
    """
    try:
        if not before_emotions or not after_emotions:
            raise HTTPException(status_code=422, detail="干预前后情感数据不能为空")

        before_states: List[EmotionState] = []
        after_states: List[EmotionState] = []

        for data in before_emotions:
            try:
                before_states.append(EmotionState.from_dict(data))
            except KeyError as e:
                raise HTTPException(status_code=422, detail=f"before_emotions缺少字段: {e}")
        for data in after_emotions:
            try:
                after_states.append(EmotionState.from_dict(data))
            except KeyError as e:
                raise HTTPException(status_code=422, detail=f"after_emotions缺少字段: {e}")

        effectiveness = _calculate_intervention_effectiveness(before_states, after_states)
        effectiveness_score = effectiveness["effectiveness_score"]

        plan = intervention_plan_store.get(intervention_id)
        if plan:
            plan.progress = max(plan.progress, effectiveness_score)
            if plan.status == "draft":
                plan.status = "active"
            if plan.progress >= 0.7:
                plan.status = "completed"
            intervention_plan_store[intervention_id] = plan

        intervention_effectiveness_log[intervention_id] = {
            "timestamp": utc_now(),
            "score": effectiveness_score,
            "user_feedback": user_feedback
        }

        return {
            "success": True,
            "effectiveness_score": effectiveness_score,
            "intervention_id": intervention_id,
            "details": effectiveness,
            "message": "干预效果评估完成"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"干预效果评估失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"干预效果评估失败: {str(e)}")

@router.put("/intervention/{plan_id}/effectiveness")
async def update_intervention_effectiveness(
    plan_id: str,
    effectiveness: float = Body(..., embed=True),
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """更新干预计划的效果评分"""
    try:
        plan = intervention_plan_store.get(plan_id)
        if not plan:
            raise HTTPException(status_code=404, detail="干预计划不存在")
        plan.progress = max(0.0, min(1.0, effectiveness))
        plan.status = "completed" if plan.progress >= 0.7 else "active"
        intervention_plan_store[plan_id] = plan
        intervention_effectiveness_log[plan_id] = {
            "timestamp": utc_now(),
            "score": plan.progress
        }
        return {
            "success": True,
            "plan": plan.to_dict(),
            "message": "干预效果更新完成"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"干预效果更新失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"干预效果更新失败: {str(e)}")

@router.get("/interventions/{user_id}")
async def get_intervention_plans(
    user_id: str,
    active: Optional[bool] = None,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """获取用户干预计划"""
    try:
        plans = [
            plan for plan in intervention_plan_store.values()
            if plan.user_id == user_id
        ]
        if active is not None:
            if active:
                plans = [plan for plan in plans if plan.status == "active"]
            else:
                plans = [plan for plan in plans if plan.status != "active"]
        return {
            "success": True,
            "plans": [plan.to_dict() for plan in plans],
            "count": len(plans),
            "message": "干预计划查询完成"
        }
    except Exception as e:
        logger.error(f"干预计划查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"干预计划查询失败: {str(e)}")

@router.get("/crisis-alerts")
async def get_crisis_alerts(
    active: Optional[bool] = None,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """获取危机告警列表"""
    try:
        alerts = crisis_event_log
        if active is not None:
            alerts = [
                alert for alert in alerts
                if (not alert.get("acknowledged", False)) == active
            ]
        return {
            "success": True,
            "alerts": [
                {
                    **alert,
                    "timestamp": alert["timestamp"].isoformat()
                }
                for alert in alerts
            ],
            "count": len(alerts),
            "message": "危机告警查询完成"
        }
    except Exception as e:
        logger.error(f"危机告警查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"危机告警查询失败: {str(e)}")

@router.post("/crisis/{assessment_id}/acknowledge")
async def acknowledge_crisis(
    assessment_id: str,
    action: str = Body(..., embed=True),
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """确认危机告警并记录处理动作"""
    try:
        target = None
        for alert in crisis_event_log:
            if alert.get("event_id") == assessment_id:
                target = alert
                break
        if not target:
            raise HTTPException(status_code=404, detail="危机告警不存在")
        target["acknowledged"] = True
        target["action"] = action
        target["acknowledged_at"] = utc_now()
        return {
            "success": True,
            "message": "危机告警已确认",
            "alert": {
                **target,
                "timestamp": target["timestamp"].isoformat(),
                "acknowledged_at": target["acknowledged_at"].isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"危机告警确认失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"危机告警确认失败: {str(e)}")

@router.get("/health-insights/{user_id}")
async def get_health_insights(
    user_id: str,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取个性化健康洞察
    """
    try:
        end_time = utc_now()
        start_time = end_time - timedelta(days=30)
        emotion_history = await _load_emotion_history(
            user_id=user_id,
            limit=500,
            start_time=start_time,
            end_time=end_time
        )
        risk_assessments = await _build_daily_risk_assessments(user_id, emotion_history)
        if not risk_assessments and emotion_history:
            risk_assessments = [
                await risk_engine.assess_comprehensive_risk(
                    user_id=user_id,
                    emotion_history=emotion_history,
                    personality_profile=None,
                    context=None
                )
            ]

        interventions = [
            plan for plan in intervention_plan_store.values()
            if plan.user_id == user_id
        ]

        async with get_db_session() as session:
            repo = EmotionModelingRepository(session)
            personality_profile = await repo.get_personality_profile(user_id)

        dashboard = await health_monitor.generate_health_dashboard(
            user_id=user_id,
            emotion_history=emotion_history,
            risk_assessments=risk_assessments,
            interventions=interventions,
            personality_profile=personality_profile,
            time_period=(start_time, end_time)
        )

        return {
            "success": True,
            "insights": dashboard.insights,
            "recommendations": dashboard.recommendations,
            "message": "健康洞察生成完成"
        }
        
    except Exception as e:
        logger.error(f"健康洞察生成失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"健康洞察生成失败: {str(e)}")

@router.get("/high-risk-users")
async def get_high_risk_users(
    threshold: float = 0.7,
    lookback_days: int = 30,
    limit: int = 50,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """获取高风险用户列表"""
    try:
        users = await _get_high_risk_users(
            threshold=threshold,
            lookback_days=lookback_days,
            limit=limit
        )
        return {
            "success": True,
            "users": users,
            "count": len(users),
            "message": "高风险用户查询完成"
        }
    except Exception as e:
        logger.error(f"高风险用户查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"高风险用户查询失败: {str(e)}")

@router.get("/stats")
async def get_system_stats(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """获取情感智能系统统计"""
    try:
        decisions = decision_engine.decision_history
        total_decisions = len(decisions)
        average_confidence = (
            sum(item.confidence_score for item in decisions) / total_decisions
            if total_decisions else 0.0
        )
        active_interventions = len([
            plan for plan in intervention_plan_store.values()
            if plan.status == "active"
        ])
        crisis_detections_24h = len([
            event for event in crisis_event_log
            if event["timestamp"] >= utc_now() - timedelta(hours=24)
        ])
        successful_interventions = len([
            record for record in intervention_effectiveness_log.values()
            if record.get("score", 0.0) >= 0.7
            and record.get("timestamp") >= utc_now() - timedelta(days=30)
        ])
        high_risk_users = await _get_high_risk_users(threshold=0.7, lookback_days=30, limit=500)
        return {
            "total_decisions": total_decisions,
            "average_confidence": average_confidence,
            "high_risk_users": len(high_risk_users),
            "active_interventions": active_interventions,
            "crisis_detections_24h": crisis_detections_24h,
            "successful_interventions": successful_interventions,
            "system_uptime": int((utc_now() - system_start_time).total_seconds()),
            "last_update": utc_now().isoformat(),
            "success": True
        }
    except Exception as e:
        logger.error(f"系统统计获取失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"系统统计获取失败: {str(e)}")

@router.get("/config")
async def get_system_config(
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """获取情感智能系统配置"""
    return {
        "success": True,
        "config": emotional_intelligence_config
    }

@router.put("/config")
async def update_system_config(
    config: Dict[str, Any] = Body(...),
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """更新情感智能系统配置"""
    try:
        emotional_intelligence_config.update(config or {})
        return {
            "success": True,
            "config": emotional_intelligence_config
        }
    except Exception as e:
        logger.error(f"系统配置更新失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"系统配置更新失败: {str(e)}")

@router.get("/system-status")
async def get_system_status() -> Dict[str, Any]:
    """
    获取情感智能系统状态
    """
    try:
        status_info = {
            "decision_engine": "operational" if decision_engine.strategies else "no_strategies",
            "risk_assessment_engine": "operational" if risk_engine.risk_weights else "uninitialized",
            "intervention_engine": "operational" if intervention_engine.strategies else "no_strategies",
            "crisis_detection_system": "operational",
            "health_monitoring_system": "operational" if health_monitor.health_metrics else "uninitialized",
            "registered_strategies": len(decision_engine.strategies),
            "decision_history_count": len(decision_engine.decision_history),
            "active_interventions": len([p for p in intervention_plan_store.values() if p.status == "active"]),
            "crisis_events_24h": len([
                event for event in crisis_event_log
                if event["timestamp"] >= utc_now() - timedelta(hours=24)
            ]),
            "system_uptime_seconds": int((utc_now() - system_start_time).total_seconds()),
            "timestamp": utc_now().isoformat()
        }
        return {
            "success": True,
            "system_status": status_info,
            "message": "系统状态更新完成"
        }
        
    except Exception as e:
        logger.error(f"系统状态检查失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"系统状态检查失败: {str(e)}")
