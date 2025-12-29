"""
共情响应生成API端点
"""

from src.core.utils.timezone_utils import utc_now
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import Field
from src.ai.empathy_response.response_engine import EmpathyResponseEngine
from src.ai.empathy_response.models import (
    EmpathyRequest, EmpathyResponse, EmpathyType,
    CulturalContext, ResponseTone
)
from src.api.base_model import ApiBaseModel
from src.ai.emotion_modeling.models import EmotionState, PersonalityProfile
from src.ai.emotion_recognition.models.emotion_models import MultiModalEmotion
from src.core.dependencies import get_current_user

from src.core.logging import get_logger
logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/empathy", tags=["empathy"])

# 全局共情响应引擎实例
empathy_engine = EmpathyResponseEngine()

# Pydantic模型用于API
class EmotionStateRequest(ApiBaseModel):
    """情感状态请求模型"""
    emotion: str = Field(..., description="情感类型")
    intensity: float = Field(..., ge=0.0, le=1.0, description="情感强度")
    valence: float = Field(..., ge=-1.0, le=1.0, description="情感效价")
    arousal: float = Field(..., ge=0.0, le=1.0, description="唤醒度")
    dominance: float = Field(..., ge=0.0, le=1.0, description="支配性")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="置信度")
    source: str = Field("api", description="来源")

class PersonalityProfileRequest(ApiBaseModel):
    """个性画像请求模型"""
    emotional_traits: Dict[str, float] = Field(..., description="情感特质")
    baseline_emotions: Dict[str, float] = Field(default_factory=dict, description="基线情感")
    emotion_volatility: float = Field(0.5, ge=0.0, le=1.0, description="情感波动性")
    recovery_rate: float = Field(0.5, ge=0.0, le=1.0, description="恢复速度")

class EmpathyFeedbackRequest(ApiBaseModel):
    """共情反馈请求模型"""
    response_id: str = Field(..., description="响应ID")
    rating: float = Field(..., ge=0.0, le=5.0, description="评分 (0-5)")
    feedback_text: Optional[str] = Field(None, description="反馈文本")
    user_id: Optional[str] = Field(None, description="用户ID")

class EmpathyGenerateRequest(ApiBaseModel):
    """共情响应生成请求"""
    user_id: str = Field(..., description="用户ID")
    message: str = Field(..., description="用户消息")
    emotion_state: Optional[EmotionStateRequest] = Field(None, description="情感状态")
    personality_profile: Optional[PersonalityProfileRequest] = Field(None, description="个性画像")
    preferred_empathy_type: Optional[EmpathyType] = Field(None, description="偏好的共情类型")
    cultural_context: Optional[CulturalContext] = Field(None, description="文化背景")
    max_response_length: int = Field(200, ge=20, le=500, description="最大响应长度")
    urgency_level: float = Field(0.5, ge=0.0, le=1.0, description="紧急程度")
    max_generation_time_ms: float = Field(300.0, gt=0, description="最大生成时间(毫秒)")

class EmpathyResponseModel(ApiBaseModel):
    """共情响应模型"""
    id: str
    response_text: str
    empathy_type: str
    emotion_addressed: str
    comfort_level: float
    personalization_score: float
    suggested_actions: List[str]
    tone: str
    confidence: float
    timestamp: str
    generation_time_ms: float
    cultural_adaptation: Optional[str]
    template_used: Optional[str]
    metadata: Dict[str, Any]

class BatchEmpathyRequest(ApiBaseModel):
    """批量共情响应请求"""
    requests: List[EmpathyGenerateRequest] = Field(..., max_items=10, description="请求列表")

@router.post("/generate", response_model=EmpathyResponseModel)
async def generate_empathy_response(
    request: EmpathyGenerateRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
) -> EmpathyResponseModel:
    """
    生成共情响应
    
    生成基于用户情感状态和个性特征的个性化共情响应。
    """
    try:
        # 转换请求数据
        empathy_request = _convert_api_request(request)
        
        # 生成响应
        response = empathy_engine.generate_response(empathy_request)
        
        # 异步记录分析数据
        background_tasks.add_task(
            _log_empathy_interaction,
            user_id=request.user_id,
            request_data=request.model_dump(),
            response_data=response.to_dict()
        )
        
        return EmpathyResponseModel(**response.to_dict())
        
    except Exception as e:
        logger.error(f"Error generating empathy response: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate empathy response: {str(e)}"
        )

@router.post("/batch-generate", response_model=List[EmpathyResponseModel])
async def batch_generate_empathy_responses(
    batch_request: BatchEmpathyRequest,
    background_tasks: BackgroundTasks,
    current_user: str = Depends(get_current_user)
) -> List[EmpathyResponseModel]:
    """
    批量生成共情响应
    
    一次性处理多个共情响应请求，适用于批处理场景。
    """
    try:
        # 转换请求
        empathy_requests = [
            _convert_api_request(req) for req in batch_request.requests
        ]
        
        # 批量生成
        responses = empathy_engine.batch_generate_responses(empathy_requests)
        
        # 异步记录
        for req, resp in zip(batch_request.requests, responses):
            background_tasks.add_task(
                _log_empathy_interaction,
                user_id=req.user_id,
                request_data=req.model_dump(),
                response_data=resp.to_dict()
            )
        
        return [EmpathyResponseModel(**resp.to_dict()) for resp in responses]
        
    except Exception as e:
        logger.error(f"Error in batch empathy generation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to batch generate empathy responses: {str(e)}"
        )

@router.get("/strategies", response_model=Dict[str, Any])
async def get_empathy_strategies(
    emotion: Optional[str] = None,
    user_id: Optional[str] = None,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取可用的共情策略信息
    
    返回系统支持的共情策略类型和适用情况。
    """
    try:
        strategies_info = {
            "available_strategies": [
                {
                    "type": EmpathyType.COGNITIVE.value,
                    "name": "认知共情",
                    "description": "理解和识别情感，提供理性的共情回应",
                    "suitable_for": ["分析性思维", "低情感强度", "理性处理"]
                },
                {
                    "type": EmpathyType.AFFECTIVE.value,
                    "name": "情感共情", 
                    "description": "分享和镜像情感，提供情感上的共鸣",
                    "suitable_for": ["高情感强度", "情感表达", "情感连接"]
                },
                {
                    "type": EmpathyType.COMPASSIONATE.value,
                    "name": "慈悲共情",
                    "description": "提供支持行动，情感安慰和建设性帮助",
                    "suitable_for": ["困难情感", "危机情况", "需要支持"]
                }
            ],
            "cultural_contexts": [ctx.value for ctx in CulturalContext],
            "response_tones": [tone.value for tone in ResponseTone]
        }
        
        # 如果指定了情感，提供策略推荐
        if emotion:
            emotion_state = EmotionState(emotion=emotion, intensity=0.5)
            rankings = empathy_engine.strategy_selector.get_strategy_rankings(emotion_state)
            
            strategies_info["recommendations_for_emotion"] = {
                "emotion": emotion,
                "strategy_rankings": [
                    {"strategy": strategy.value, "score": score}
                    for strategy, score in rankings
                ]
            }
        
        return strategies_info
        
    except Exception as e:
        logger.error(f"Error getting empathy strategies: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get empathy strategies: {str(e)}"
        )

@router.get("/analytics", response_model=Dict[str, Any])
async def get_empathy_analytics(
    user_id: Optional[str] = None,
    current_user: str = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    获取共情响应分析数据
    
    返回系统性能统计和用户个性化分析。
    """
    try:
        analytics_data = {
            "system_performance": empathy_engine.get_performance_stats(),
            "context_statistics": empathy_engine.context_manager.get_context_statistics(),
            "personalization_stats": empathy_engine.personalization_engine.get_personalization_stats()
        }
        
        # 如果指定用户，添加用户特定分析
        if user_id:
            user_summary = empathy_engine.context_manager.get_conversation_summary(user_id)
            user_patterns = empathy_engine.personalization_engine.get_user_patterns(user_id)
            
            analytics_data["user_analysis"] = {
                "user_id": user_id,
                "conversation_summary": user_summary,
                "learned_patterns": user_patterns
            }
        
        return analytics_data
        
    except Exception as e:
        logger.error(f"Error getting empathy analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get empathy analytics: {str(e)}"
        )

@router.post("/feedback")
async def submit_empathy_feedback(
    feedback_request: EmpathyFeedbackRequest,
    current_user: str = Depends(get_current_user)
):
    """
    提交共情响应反馈
    
    用户可以对生成的共情响应进行评分和反馈，用于改进系统。
    """
    try:
        feedback_data = {
            "response_id": feedback_request.response_id,
            "rating": feedback_request.rating,
            "feedback_text": feedback_request.feedback_text,
            "user_id": feedback_request.user_id or current_user,
            "timestamp": utc_now().isoformat(),
            "source": "api"
        }
        
        # 记录反馈（这里可以存储到数据库）
        logger.info(f"Received empathy feedback: {feedback_data}")
        
        # 根据反馈调整策略权重（简单示例）
        if feedback_request.rating < 2.0:  # 低评分
            logger.warning(f"Low rating received for response {feedback_request.response_id}: {feedback_request.rating}")
            # 可以触发策略调整逻辑
        
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": f"feedback_{int(utc_now().timestamp())}",
            "status": "processed"
        }
        
    except Exception as e:
        logger.error(f"Error submitting empathy feedback: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to submit feedback: {str(e)}"
        )

@router.get("/health")
async def empathy_health_check() -> Dict[str, Any]:
    """
    共情系统健康检查
    
    检查共情响应生成系统的健康状态。
    """
    try:
        health_data = empathy_engine.health_check()
        return {
            "status": "healthy" if health_data["healthy"] else "unhealthy",
            "timestamp": utc_now().isoformat(),
            "details": health_data
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "timestamp": utc_now().isoformat(),
            "error": str(e)
        }

@router.delete("/context/{user_id}")
async def clear_user_context(
    user_id: str,
    current_user: str = Depends(get_current_user)
):
    """
    清除用户对话上下文
    
    删除指定用户的对话历史和个性化学习数据。
    """
    try:
        # 清除上下文
        context_cleared = empathy_engine.context_manager.clear_user_context(user_id)
        
        # 清除个性化模式
        patterns_cleared = empathy_engine.personalization_engine.clear_user_patterns(user_id)
        
        return {
            "message": f"User context cleared for {user_id}",
            "context_cleared": context_cleared,
            "patterns_cleared": patterns_cleared,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing user context: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear user context: {str(e)}"
        )

# 辅助函数
def _convert_api_request(api_request: EmpathyGenerateRequest) -> EmpathyRequest:
    """转换API请求为内部请求格式"""
    # 转换情感状态
    emotion_state = None
    if api_request.emotion_state:
        emotion_state = EmotionState(
            user_id=api_request.user_id,
            emotion=api_request.emotion_state.emotion,
            intensity=api_request.emotion_state.intensity,
            valence=api_request.emotion_state.valence,
            arousal=api_request.emotion_state.arousal,
            dominance=api_request.emotion_state.dominance,
            confidence=api_request.emotion_state.confidence,
            source=api_request.emotion_state.source
        )
    
    # 转换个性画像
    personality_profile = None
    if api_request.personality_profile:
        personality_profile = PersonalityProfile(
            user_id=api_request.user_id,
            emotional_traits=api_request.personality_profile.emotional_traits,
            baseline_emotions=api_request.personality_profile.baseline_emotions,
            emotion_volatility=api_request.personality_profile.emotion_volatility,
            recovery_rate=api_request.personality_profile.recovery_rate
        )
    
    return EmpathyRequest(
        user_id=api_request.user_id,
        message=api_request.message,
        emotion_state=emotion_state,
        personality_profile=personality_profile,
        preferred_empathy_type=api_request.preferred_empathy_type,
        cultural_context=api_request.cultural_context,
        max_response_length=api_request.max_response_length,
        urgency_level=api_request.urgency_level,
        max_generation_time_ms=api_request.max_generation_time_ms
    )

async def _log_empathy_interaction(
    user_id: str,
    request_data: Dict[str, Any],
    response_data: Dict[str, Any]
):
    """记录共情交互数据"""
    try:
        interaction_log = {
            "user_id": user_id,
            "timestamp": utc_now().isoformat(),
            "request": request_data,
            "response": response_data,
            "type": "empathy_interaction"
        }
        
        # 这里可以存储到数据库或日志系统
        logger.info(f"Empathy interaction logged for user {user_id}")
        
    except Exception as e:
        logger.error(f"Error logging empathy interaction: {e}")
