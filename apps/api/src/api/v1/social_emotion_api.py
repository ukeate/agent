"""
社交情感理解系统API - Story 11.6 Task 8
提供完整的社交情感理解系统REST API和WebSocket接口
"""

from src.core.utils.timezone_utils import utc_now
import asyncio
import json
from datetime import timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends, BackgroundTasks
from pydantic import Field
from starlette.status import HTTP_200_OK, HTTP_400_BAD_REQUEST, HTTP_403_FORBIDDEN
from src.ai.emotion_modeling.social_emotion_system import (
    SocialEmotionSystem, SocialEmotionRequest, SocialEmotionResponse,
    SystemConfiguration, SystemMode
)
from src.api.base_model import ApiBaseModel
from src.ai.emotion_modeling.privacy_ethics_guard import PrivacyLevel, ConsentType
from src.core.dependencies import get_current_user

from src.core.logging import get_logger
logger = get_logger(__name__)

# 创建路由
router = APIRouter(prefix="/social-emotion", tags=["Social Emotion Understanding"])

# 全局系统实例
social_emotion_system: Optional[SocialEmotionSystem] = None

# Pydantic模型定义
class EmotionAnalysisRequest(ApiBaseModel):
    user_id: str = Field(..., description="用户ID")
    session_id: Optional[str] = Field(None, description="会话ID")
    emotion_data: Dict[str, Any] = Field(..., description="情感数据")
    social_context: Dict[str, Any] = Field(..., description="社交上下文")
    analysis_type: List[str] = Field(
        ["context_adaptation", "cultural_analysis"], 
        description="分析类型列表"
    )
    cultural_context: Optional[str] = Field(None, description="文化背景")
    privacy_consent: bool = Field(True, description="隐私同意")

class BatchAnalysisRequest(ApiBaseModel):
    requests: List[EmotionAnalysisRequest] = Field(..., description="批量请求列表")

class SystemConfigRequest(ApiBaseModel):
    mode: SystemMode = Field(SystemMode.FULL_INTERACTIVE, description="系统模式")
    privacy_level: PrivacyLevel = Field(PrivacyLevel.RESTRICTED, description="隐私级别")
    cultural_context: Optional[str] = Field(None, description="默认文化背景")
    enable_real_time_monitoring: bool = Field(True, description="启用实时监控")
    enable_predictive_analytics: bool = Field(True, description="启用预测分析")
    enable_emotional_coaching: bool = Field(False, description="启用情感教练")
    max_concurrent_sessions: int = Field(100, description="最大并发会话数")
    data_retention_days: int = Field(365, description="数据保留天数")
    websocket_enabled: bool = Field(True, description="启用WebSocket")

class ConsentRequest(ApiBaseModel):
    user_id: str = Field(..., description="用户ID")
    consent_type: ConsentType = Field(..., description="同意类型")
    data_categories: List[str] = Field(..., description="数据类别")
    purpose: str = Field(..., description="使用目的")
    expiry_days: Optional[int] = Field(None, description="有效期天数")

class PrivacyPolicyRequest(ApiBaseModel):
    user_id: str = Field(..., description="用户ID")
    privacy_level: PrivacyLevel = Field(..., description="隐私级别")
    data_retention_days: int = Field(365, description="数据保留天数")
    sharing_permissions: Optional[Dict[str, bool]] = Field(None, description="分享权限")
    cultural_context: Optional[str] = Field(None, description="文化背景")

class SessionRequest(ApiBaseModel):
    user_id: str = Field(..., description="用户ID")
    session_config: Dict[str, Any] = Field({}, description="会话配置")

class DataExportRequest(ApiBaseModel):
    user_id: str = Field(..., description="用户ID")
    data_types: List[str] = Field(..., description="数据类型")
    format_type: str = Field("json", description="导出格式")

def get_social_emotion_system() -> SocialEmotionSystem:
    """获取社交情感系统实例"""
    global social_emotion_system
    if social_emotion_system is None:
        # 使用默认配置初始化系统
        config = SystemConfiguration(
            mode=SystemMode.FULL_INTERACTIVE,
            privacy_level=PrivacyLevel.RESTRICTED,
            cultural_context=None,
            enable_real_time_monitoring=True,
            enable_predictive_analytics=True,
            enable_emotional_coaching=False,
            max_concurrent_sessions=100,
            data_retention_days=365,
            websocket_enabled=True
        )
        social_emotion_system = SocialEmotionSystem(config)
    return social_emotion_system

@router.post("/initialize", summary="初始化系统配置")
async def initialize_system(
    config_request: SystemConfigRequest,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """初始化或更新社交情感理解系统配置"""
    try:
        config = SystemConfiguration(
            mode=config_request.mode,
            privacy_level=config_request.privacy_level,
            cultural_context=config_request.cultural_context,
            enable_real_time_monitoring=config_request.enable_real_time_monitoring,
            enable_predictive_analytics=config_request.enable_predictive_analytics,
            enable_emotional_coaching=config_request.enable_emotional_coaching,
            max_concurrent_sessions=config_request.max_concurrent_sessions,
            data_retention_days=config_request.data_retention_days,
            websocket_enabled=config_request.websocket_enabled
        )
        
        global social_emotion_system
        social_emotion_system = SocialEmotionSystem(config)
        
        return {
            "message": "Social emotion system initialized successfully",
            "config": config_request.model_dump(),
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/analyze", summary="情感分析")
async def analyze_emotion(
    request: EmotionAnalysisRequest,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """执行社交情感分析"""
    try:
        system = get_social_emotion_system()
        
        # 构建内部请求对象
        social_request = SocialEmotionRequest(
            request_id=f"req_{utc_now().isoformat()}",
            user_id=request.user_id,
            session_id=request.session_id,
            emotion_data=request.emotion_data,
            social_context=request.social_context,
            analysis_type=request.analysis_type,
            privacy_consent=request.privacy_consent,
            cultural_context=request.cultural_context,
            timestamp=utc_now()
        )
        
        # 处理请求
        response = await system.process_social_emotion_request(social_request)
        
        return {
            "request_id": response.request_id,
            "user_id": response.user_id,
            "session_id": response.session_id,
            "results": response.results,
            "recommendations": response.recommendations,
            "cultural_adaptations": response.cultural_adaptations,
            "confidence_score": response.confidence_score,
            "processing_time": response.processing_time,
            "privacy_compliant": response.privacy_compliant,
            "timestamp": response.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Emotion analysis failed: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/analyze/batch", summary="批量情感分析")
async def batch_analyze_emotion(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """批量执行社交情感分析"""
    try:
        system = get_social_emotion_system()
        
        if len(request.requests) > 100:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Batch size cannot exceed 100 requests"
            )
        
        # 构建内部请求对象列表
        social_requests = []
        for req in request.requests:
            social_request = SocialEmotionRequest(
                request_id=f"batch_req_{utc_now().isoformat()}_{req.user_id}",
                user_id=req.user_id,
                session_id=req.session_id,
                emotion_data=req.emotion_data,
                social_context=req.social_context,
                analysis_type=req.analysis_type,
                privacy_consent=req.privacy_consent,
                cultural_context=req.cultural_context,
                timestamp=utc_now()
            )
            social_requests.append(social_request)
        
        # 批量处理
        responses = await system.batch_process(social_requests)
        
        # 格式化响应
        formatted_responses = []
        for response in responses:
            formatted_responses.append({
                "request_id": response.request_id,
                "user_id": response.user_id,
                "results": response.results,
                "recommendations": response.recommendations,
                "confidence_score": response.confidence_score,
                "privacy_compliant": response.privacy_compliant
            })
        
        return {
            "batch_id": f"batch_{utc_now().isoformat()}",
            "total_requests": len(responses),
            "successful_requests": len([r for r in responses if r.privacy_compliant]),
            "responses": formatted_responses,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch emotion analysis failed: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/session/create", summary="创建会话")
async def create_session(
    request: SessionRequest,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """创建新的社交情感分析会话"""
    try:
        system = get_social_emotion_system()
        session_id = await system.create_session(request.user_id, request.session_config)
        
        return {
            "session_id": session_id,
            "user_id": request.user_id,
            "created_at": utc_now().isoformat(),
            "config": request.session_config
        }
        
    except Exception as e:
        logger.error(f"Session creation failed: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@router.delete("/session/{session_id}", summary="关闭会话")
async def close_session(
    session_id: str,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """关闭指定的社交情感分析会话"""
    try:
        system = get_social_emotion_system()
        success = await system.close_session(session_id)
        
        if success:
            return {
                "message": f"Session {session_id} closed successfully",
                "timestamp": utc_now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session closure failed: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/privacy/consent", summary="管理用户同意")
async def manage_consent(
    request: ConsentRequest,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """管理用户数据处理同意"""
    try:
        system = get_social_emotion_system()
        
        consent_record = await system.privacy_guard.manage_consent(
            user_id=request.user_id,
            consent_type=request.consent_type,
            data_categories=request.data_categories,
            purpose=request.purpose,
            ip_address=current_user.get("ip_address", ""),
            user_agent=current_user.get("user_agent", ""),
            expiry_days=request.expiry_days
        )
        
        return {
            "message": "Consent recorded successfully",
            "consent_id": f"{request.user_id}_{consent_record.timestamp.isoformat()}",
            "user_id": consent_record.user_id,
            "consent_type": consent_record.consent_type.value,
            "data_categories": consent_record.data_categories,
            "purpose": consent_record.purpose,
            "expiry_date": consent_record.expiry_date.isoformat() if consent_record.expiry_date else None,
            "timestamp": consent_record.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Consent management failed: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@router.delete("/privacy/consent/{user_id}", summary="撤回同意")
async def withdraw_consent(
    user_id: str,
    data_categories: List[str],
    purpose: str,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """撤回用户数据处理同意"""
    try:
        system = get_social_emotion_system()
        
        success = await system.privacy_guard.withdraw_consent(
            user_id=user_id,
            data_categories=data_categories,
            purpose=purpose
        )
        
        if success:
            return {
                "message": "Consent withdrawn successfully",
                "user_id": user_id,
                "data_categories": data_categories,
                "purpose": purpose,
                "timestamp": utc_now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="No matching consent found to withdraw"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Consent withdrawal failed: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/privacy/policy", summary="创建隐私政策")
async def create_privacy_policy(
    request: PrivacyPolicyRequest,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """为用户创建隐私政策"""
    try:
        system = get_social_emotion_system()
        
        privacy_policy = await system.privacy_guard.create_privacy_policy(
            user_id=request.user_id,
            privacy_level=request.privacy_level,
            data_retention_days=request.data_retention_days,
            sharing_permissions=request.sharing_permissions,
            cultural_context=request.cultural_context
        )
        
        return {
            "message": "Privacy policy created successfully",
            "user_id": privacy_policy.user_id,
            "privacy_level": privacy_policy.privacy_level.value,
            "data_retention_days": privacy_policy.data_retention_days,
            "sharing_permissions": privacy_policy.sharing_permissions,
            "anonymization_required": privacy_policy.anonymization_required,
            "created_at": privacy_policy.created_at.isoformat(),
            "updated_at": privacy_policy.updated_at.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Privacy policy creation failed: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@router.get("/status", summary="系统状态")
async def get_system_status(
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """获取社交情感理解系统状态"""
    try:
        system = get_social_emotion_system()
        status = await system.get_system_status()
        
        return {
            "active_sessions": status.active_sessions,
            "total_users": status.total_users,
            "processing_queue_size": status.processing_queue_size,
            "average_response_time": status.average_response_time,
            "compliance_score": status.compliance_score,
            "cultural_contexts": status.cultural_contexts,
            "last_updated": status.last_updated.isoformat(),
            "system_health": "healthy" if status.compliance_score > 0.8 else "warning"
        }
        
    except Exception as e:
        logger.error(f"Status retrieval failed: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@router.get("/dashboard", summary="分析仪表板")
async def get_analytics_dashboard(
    user_id: Optional[str] = None,
    days: int = 7,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """获取分析仪表板数据"""
    try:
        system = get_social_emotion_system()
        
        time_range = (
            utc_now() - timedelta(days=days),
            utc_now()
        )
        
        dashboard_data = await system.get_analytics_dashboard(user_id, time_range)
        
        return {
            "dashboard_data": dashboard_data,
            "time_range": {
                "start": time_range[0].isoformat(),
                "end": time_range[1].isoformat(),
                "days": days
            },
            "generated_at": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@router.get("/compliance/report", summary="合规报告")
async def get_compliance_report(
    days: int = 30,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """获取系统合规报告"""
    try:
        system = get_social_emotion_system()
        
        end_date = utc_now()
        start_date = end_date - timedelta(days=days)
        
        report = await system.privacy_guard.get_compliance_report(start_date, end_date)
        
        return report
        
    except Exception as e:
        logger.error(f"Compliance report retrieval failed: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/export", summary="导出用户数据")
async def export_user_data(
    request: DataExportRequest,
    current_user: Dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """导出用户的社交情感数据"""
    try:
        system = get_social_emotion_system()
        
        # 验证用户权限
        if request.user_id != current_user.get("user_id") and not current_user.get("is_admin", False):
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Insufficient permissions to export data for this user"
            )
        
        export_data = await system.export_data(
            user_id=request.user_id,
            data_types=request.data_types,
            format_type=request.format_type
        )
        
        return export_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Data export failed: {e}")
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail=str(e))

# WebSocket端点
@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket实时情感分析连接"""
    try:
        # 这里应该验证用户令牌，简化处理
        system = get_social_emotion_system()
        
        await system.register_websocket(user_id, websocket)
        
        try:
            while True:
                # 接收客户端消息
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong",
                        "timestamp": utc_now().isoformat()
                    }))
                
                elif message.get("type") == "emotion_analysis":
                    # 处理实时情感分析请求
                    emotion_request = SocialEmotionRequest(
                        request_id=f"ws_req_{utc_now().isoformat()}",
                        user_id=user_id,
                        session_id=message.get("session_id"),
                        emotion_data=message.get("emotion_data", {}),
                        social_context=message.get("social_context", {}),
                        analysis_type=message.get("analysis_type", ["context_adaptation"]),
                        privacy_consent=message.get("privacy_consent", True),
                        cultural_context=message.get("cultural_context"),
                        timestamp=utc_now()
                    )
                    
                    # 异步处理并发送结果
                    response = await system.process_social_emotion_request(emotion_request)
                    
                    await websocket.send_text(json.dumps({
                        "type": "emotion_analysis_result",
                        "request_id": response.request_id,
                        "results": response.results,
                        "recommendations": response.recommendations,
                        "confidence_score": response.confidence_score,
                        "timestamp": response.timestamp.isoformat()
                    }))
                
        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected for user {user_id}")
        except Exception as e:
            logger.error(f"WebSocket error for user {user_id}: {e}")
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": str(e),
                "timestamp": utc_now().isoformat()
            }))
        
        finally:
            await system.unregister_websocket(user_id)
            
    except Exception as e:
        logger.error(f"WebSocket connection failed for user {user_id}: {e}")
        await websocket.close(code=1000)
