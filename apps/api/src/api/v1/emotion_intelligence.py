"""
情感智能系统主API
整合所有情感智能模块
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import logging

from ai.emotion_modeling.core_interfaces import (
    UnifiedEmotionalData, EmotionalIntelligenceResponse,
    EmotionState, EmotionType, ModalityType
)
from ai.emotion_modeling.data_flow_manager import EmotionalDataFlowManagerImpl
from ai.emotion_modeling.communication_protocol import CommunicationProtocol
from ai.emotion_modeling.quality_monitor import quality_monitor
from ai.emotion_modeling.result_formatter import result_formatter_manager, OutputFormat
from core.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/emotion-intelligence", tags=["emotion-intelligence"])

# API请求模型
class EmotionAnalysisRequest(BaseModel):
    user_id: str
    text: Optional[str] = None
    audio_data: Optional[str] = None  # Base64编码
    video_data: Optional[str] = None  # Base64编码
    image_data: Optional[str] = None  # Base64编码
    physiological_data: Optional[Dict[str, Any]] = None
    modalities: List[ModalityType]
    include_personality: bool = Field(default=True)
    include_empathy: bool = Field(default=True)
    output_format: OutputFormat = Field(default=OutputFormat.JSON)

class SystemStatusResponse(BaseModel):
    status: str
    version: str
    modules: Dict[str, bool]
    performance: Dict[str, float]
    timestamp: datetime

# 全局系统实例（在实际应用中应该使用依赖注入）
_system_components = {}

@router.on_event("startup")
async def initialize_emotion_system():
    """初始化情感智能系统"""
    try:
        # 初始化通信协议
        protocol = CommunicationProtocol()
        await protocol.start()
        
        # 初始化数据流管理器
        data_flow_manager = EmotionalDataFlowManagerImpl(protocol)
        await data_flow_manager.initialize()
        
        # 启动质量监控
        await quality_monitor.start_monitoring()
        
        _system_components.update({
            "protocol": protocol,
            "data_flow_manager": data_flow_manager,
            "quality_monitor": quality_monitor,
            "formatter_manager": result_formatter_manager
        })
        
        logger.info("Emotion intelligence system initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize emotion intelligence system: {e}")
        raise

@router.on_event("shutdown") 
async def shutdown_emotion_system():
    """关闭情感智能系统"""
    try:
        if "quality_monitor" in _system_components:
            await _system_components["quality_monitor"].stop_monitoring()
        
        if "data_flow_manager" in _system_components:
            await _system_components["data_flow_manager"].shutdown()
            
        if "protocol" in _system_components:
            await _system_components["protocol"].stop()
            
        _system_components.clear()
        logger.info("Emotion intelligence system shutdown completed")
        
    except Exception as e:
        logger.error(f"Error during emotion intelligence system shutdown: {e}")

@router.post("/analyze", response_model=EmotionalIntelligenceResponse)
async def analyze_emotion(
    request: EmotionAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    分析情感数据
    """
    try:
        if not _system_components:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # 创建模拟的情感数据（实际应用中应该调用真实的AI模型）
        emotion_state = EmotionState(
            emotion=EmotionType.HAPPINESS,  # 这里应该是实际的情感识别结果
            intensity=0.8,
            valence=0.7,
            arousal=0.6,
            dominance=0.5,
            confidence=0.9,
            timestamp=datetime.now()
        )
        
        unified_data = UnifiedEmotionalData(
            user_id=request.user_id,
            timestamp=datetime.now(),
            emotional_state=emotion_state,
            confidence=0.85,
            processing_time=0.15,
            data_quality=0.92
        )
        
        # 通过数据流管理器处理
        data_flow_manager = _system_components["data_flow_manager"]
        await data_flow_manager.route_data(unified_data)
        
        # 后台记录质量监控数据
        background_tasks.add_task(
            record_quality_metrics,
            request.user_id,
            emotion_state,
            0.15,
            0.85,
            0.92
        )
        
        # 格式化响应
        response = EmotionalIntelligenceResponse(
            success=True,
            data=unified_data,
            metadata={
                "processing_time": 0.15,
                "data_quality": 0.92,
                "modalities_processed": request.modalities,
                "output_format": request.output_format.value
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error in emotion analysis: {e}")
        return EmotionalIntelligenceResponse(
            success=False,
            error={
                "code": "ANALYSIS_ERROR",
                "message": str(e)
            },
            metadata={"timestamp": datetime.now().isoformat()}
        )

async def record_quality_metrics(
    user_id: str,
    emotion_state: EmotionState,
    processing_time: float,
    confidence: float,
    data_quality: float
):
    """记录质量监控指标"""
    try:
        await quality_monitor.record_prediction(
            user_id=user_id,
            predicted_emotion=emotion_state,
            modality=ModalityType.TEXT,
            processing_time=processing_time,
            confidence=confidence,
            data_quality=data_quality
        )
    except Exception as e:
        logger.error(f"Error recording quality metrics: {e}")

@router.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """获取系统状态"""
    try:
        modules_status = {
            "communication_protocol": "protocol" in _system_components,
            "data_flow_manager": "data_flow_manager" in _system_components,
            "quality_monitor": "quality_monitor" in _system_components,
            "formatter_manager": "formatter_manager" in _system_components
        }
        
        # 获取性能指标
        performance_metrics = {}
        if "quality_monitor" in _system_components:
            quality_report = _system_components["quality_monitor"].get_quality_report()
            performance_metrics = {
                "accuracy": quality_report.get("metrics", {}).get("accuracy", {}).get("latest", 0.0),
                "processing_rate": quality_report.get("metrics", {}).get("throughput", {}).get("latest", 0.0),
                "error_rate": quality_report.get("metrics", {}).get("error_rate", {}).get("latest", 0.0)
            }
        
        return SystemStatusResponse(
            status="healthy" if all(modules_status.values()) else "degraded",
            version="1.0.0",
            modules=modules_status,
            performance=performance_metrics,
            timestamp=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get system status")

@router.get("/quality-report")
async def get_quality_report(
    time_window_hours: Optional[int] = 24,
    format: OutputFormat = OutputFormat.JSON
):
    """获取质量监控报告"""
    try:
        if "quality_monitor" not in _system_components:
            raise HTTPException(status_code=503, detail="Quality monitor not available")
        
        time_window = timedelta(hours=time_window_hours) if time_window_hours else None
        quality_report = _system_components["quality_monitor"].get_quality_report(time_window)
        
        if format == OutputFormat.JSON:
            return quality_report
        else:
            # 使用格式化管理器转换格式
            formatter_manager = _system_components["formatter_manager"]
            # 这里需要适配格式化管理器的接口
            return {"formatted_report": "Format not yet implemented"}
            
    except Exception as e:
        logger.error(f"Error getting quality report: {e}")
        raise HTTPException(status_code=500, detail="Failed to get quality report")

@router.post("/quality/ground-truth")
async def submit_ground_truth(
    user_id: str,
    true_emotion: EmotionType,
    intensity: float = Field(ge=0.0, le=1.0),
    valence: float = Field(ge=-1.0, le=1.0),
    arousal: float = Field(ge=0.0, le=1.0),
    dominance: float = Field(ge=0.0, le=1.0),
    modality: ModalityType = ModalityType.TEXT,
    source: str = "user_feedback"
):
    """提交真实标签数据用于质量评估"""
    try:
        if "quality_monitor" not in _system_components:
            raise HTTPException(status_code=503, detail="Quality monitor not available")
        
        from ai.emotion_modeling.quality_monitor import GroundTruthData
        
        true_emotion_state = EmotionState(
            emotion=true_emotion,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            confidence=1.0,
            timestamp=datetime.now()
        )
        
        ground_truth = GroundTruthData(
            user_id=user_id,
            timestamp=datetime.now(),
            true_emotion=true_emotion_state,
            modality=modality,
            source=source,
            confidence=1.0
        )
        
        await _system_components["quality_monitor"].add_ground_truth(ground_truth)
        
        return {
            "success": True,
            "message": "Ground truth data submitted successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error submitting ground truth: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit ground truth")

@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "emotion-intelligence",
        "version": "1.0.0"
    }