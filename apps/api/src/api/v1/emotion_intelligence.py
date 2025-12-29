"""
情感智能系统主API
整合所有情感智能模块
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import base64
import time
import asyncio
import json
import uuid
import httpx
from fastapi.encoders import jsonable_encoder
from src.ai.emotion_modeling.core_interfaces import (
    UnifiedEmotionalData, EmotionalIntelligenceResponse,
    EmotionState, EmotionType, ModalityType
)
from src.api.base_model import ApiBaseModel
from src.ai.emotion_modeling.quality_monitor import quality_monitor
from src.ai.emotion_modeling.result_formatter import OutputFormat
from src.ai.emotion_recognition.analyzers.text_analyzer import TextEmotionAnalyzer
from src.ai.emotion_recognition.analyzers.audio_analyzer import AudioEmotionAnalyzer
from src.ai.emotion_recognition.analyzers.visual_analyzer import VisualEmotionAnalyzer
from src.core.utils.timezone_utils import utc_now
from src.core.redis import get_redis

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/emotion-intelligence", tags=["emotion-intelligence"])

# API请求模型
class EmotionAnalysisRequest(ApiBaseModel):
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

class SystemStatusResponse(ApiBaseModel):
    status: str
    version: str
    modules: Dict[str, bool]
    performance: Dict[str, float]
    timestamp: datetime

class GroundTruthRequest(ApiBaseModel):
    user_id: str
    true_emotion: EmotionType
    intensity: float = Field(ge=0.0, le=1.0)
    valence: float = Field(ge=-1.0, le=1.0)
    arousal: float = Field(ge=0.0, le=1.0)
    dominance: float = Field(ge=0.0, le=1.0)
    modality: ModalityType = ModalityType.TEXT
    source: str = "user_feedback"

# 全局系统实例（在实际应用中应该使用依赖注入）
_system_components = {}

_text_analyzer = TextEmotionAnalyzer()
_audio_analyzer = AudioEmotionAnalyzer()
_visual_analyzer = VisualEmotionAnalyzer()

_QUALITY_MONITOR_LOCK = asyncio.Lock()
_INTERACTIONS_KEY = "emotion_intelligence:interactions"

async def _ensure_quality_monitor():
    if "quality_monitor" in _system_components:
        return _system_components["quality_monitor"]
    async with _QUALITY_MONITOR_LOCK:
        if "quality_monitor" in _system_components:
            return _system_components["quality_monitor"]
        await quality_monitor.start_monitoring()
        _system_components["quality_monitor"] = quality_monitor
        return quality_monitor

async def _record_interaction(user_id: str, message: str, sentiment: str, valence: float):
    redis_client = get_redis()
    if not redis_client:
        return
    entry = {
        "id": uuid.uuid4().hex,
        "user_id": user_id,
        "message": message,
        "sentiment": sentiment,
        "valence": valence,
        "timestamp": utc_now().isoformat(),
    }
    await redis_client.lpush(_INTERACTIONS_KEY, json.dumps(entry, ensure_ascii=False))
    await redis_client.ltrim(_INTERACTIONS_KEY, 0, 199)

async def _record_quality_metrics(
    user_id: str,
    modality_states: Dict[ModalityType, EmotionState],
    processing_time_ms: float,
    data_quality: float,
):
    monitor = await _ensure_quality_monitor()
    for modality, state in modality_states.items():
        await monitor.record_prediction(
            user_id=user_id,
            predicted_emotion=state,
            modality=modality,
            processing_time=processing_time_ms,
            confidence=state.confidence,
            data_quality=data_quality,
        )

@router.post("/analyze", response_model=EmotionalIntelligenceResponse)
async def analyze_emotion(
    request: EmotionAnalysisRequest,
    background_tasks: BackgroundTasks
):
    """分析情感数据"""
    start = time.perf_counter()
    modalities = set(request.modalities or [])
    input_modalities: Dict[ModalityType, Any] = {}

    if ModalityType.TEXT in modalities:
        if not request.text:
            raise HTTPException(status_code=400, detail="缺少文本输入")
        input_modalities[ModalityType.TEXT] = request.text

    if ModalityType.AUDIO in modalities:
        if not request.audio_data:
            raise HTTPException(status_code=400, detail="缺少音频输入")
        try:
            input_modalities[ModalityType.AUDIO] = base64.b64decode(request.audio_data)
        except Exception:
            raise HTTPException(status_code=400, detail="音频数据Base64解码失败")

    if ModalityType.IMAGE in modalities:
        if not request.image_data:
            raise HTTPException(status_code=400, detail="缺少图像输入")
        try:
            input_modalities[ModalityType.IMAGE] = base64.b64decode(request.image_data)
        except Exception:
            raise HTTPException(status_code=400, detail="图像数据Base64解码失败")

    unsupported = modalities - {ModalityType.TEXT, ModalityType.AUDIO, ModalityType.IMAGE}
    if unsupported:
        raise HTTPException(status_code=400, detail=f"暂不支持模态: {', '.join(sorted([m.value for m in unsupported]))}")

    if not input_modalities:
        raise HTTPException(status_code=400, detail="未提供可用输入")

    modality_states: Dict[ModalityType, EmotionState] = {}

    def _to_emotion_type(label: str) -> EmotionType:
        label = (label or "").lower()
        if label in {e.value for e in EmotionType}:
            return EmotionType(label)
        mapping = {
            "joy": EmotionType.HAPPINESS,
            "excitement": EmotionType.HAPPINESS,
            "satisfaction": EmotionType.HAPPINESS,
            "contentment": EmotionType.HAPPINESS,
            "trust": EmotionType.HAPPINESS,
            "anticipation": EmotionType.HAPPINESS,
            "grief": EmotionType.SADNESS,
            "disappointment": EmotionType.SADNESS,
            "loneliness": EmotionType.SADNESS,
            "despair": EmotionType.SADNESS,
            "rage": EmotionType.ANGER,
            "irritation": EmotionType.ANGER,
            "frustration": EmotionType.ANGER,
            "annoyance": EmotionType.ANGER,
            "anxiety": EmotionType.FEAR,
            "worry": EmotionType.FEAR,
            "panic": EmotionType.FEAR,
            "nervousness": EmotionType.FEAR,
            "amazement": EmotionType.SURPRISE,
            "astonishment": EmotionType.SURPRISE,
            "wonder": EmotionType.SURPRISE,
        }
        return mapping.get(label, EmotionType.NEUTRAL)

    now = utc_now()
    if ModalityType.TEXT in input_modalities:
        r = await _text_analyzer.analyze(input_modalities[ModalityType.TEXT])
        d = r.dimension
        modality_states[ModalityType.TEXT] = EmotionState(
            emotion=_to_emotion_type(r.emotion),
            intensity=float(r.intensity),
            valence=float(d.valence) if d else 0.0,
            arousal=float(d.arousal) if d else 0.0,
            dominance=float(d.dominance) if d else 0.0,
            confidence=float(r.confidence),
            timestamp=now,
        )

    if ModalityType.AUDIO in input_modalities:
        r = await _audio_analyzer.analyze(input_modalities[ModalityType.AUDIO])
        d = r.dimension
        modality_states[ModalityType.AUDIO] = EmotionState(
            emotion=_to_emotion_type(r.emotion),
            intensity=float(r.intensity),
            valence=float(d.valence) if d else 0.0,
            arousal=float(d.arousal) if d else 0.0,
            dominance=float(d.dominance) if d else 0.0,
            confidence=float(r.confidence),
            timestamp=now,
        )

    if ModalityType.IMAGE in input_modalities:
        r = await _visual_analyzer.analyze(input_modalities[ModalityType.IMAGE])
        d = r.dimension
        modality_states[ModalityType.IMAGE] = EmotionState(
            emotion=_to_emotion_type(r.emotion),
            intensity=float(r.intensity),
            valence=float(d.valence) if d else 0.0,
            arousal=float(d.arousal) if d else 0.0,
            dominance=float(d.dominance) if d else 0.0,
            confidence=float(r.confidence),
            timestamp=now,
        )

    weights = {m: s.confidence for m, s in modality_states.items()}
    total_weight = sum(weights.values())
    if total_weight <= 0:
        total_weight = 1.0

    vote: Dict[EmotionType, float] = {}
    for m, s in modality_states.items():
        vote[s.emotion] = vote.get(s.emotion, 0.0) + weights.get(m, 0.0)
    fused_emotion = max(vote.keys(), key=lambda k: vote[k])

    fused = EmotionState(
        emotion=fused_emotion,
        intensity=sum(s.intensity * weights.get(m, 0.0) for m, s in modality_states.items()) / total_weight,
        valence=sum(s.valence * weights.get(m, 0.0) for m, s in modality_states.items()) / total_weight,
        arousal=sum(s.arousal * weights.get(m, 0.0) for m, s in modality_states.items()) / total_weight,
        dominance=sum(s.dominance * weights.get(m, 0.0) for m, s in modality_states.items()) / total_weight,
        confidence=sum(s.confidence for s in modality_states.values()) / len(modality_states),
        timestamp=now,
    )

    processing_time = time.perf_counter() - start
    data_quality = min(max(fused.confidence, 0.0), 1.0)
    processing_time_ms = processing_time * 1000

    data = UnifiedEmotionalData(
        user_id=request.user_id,
        timestamp=now,
        recognition_result={
            "emotions": modality_states,
            "fused_emotion": fused,
            "confidence": fused.confidence,
            "processing_time": processing_time,
        },
        emotional_state=fused,
        confidence=fused.confidence,
        processing_time=processing_time,
        data_quality=data_quality,
    )

    background_tasks.add_task(
        _record_quality_metrics,
        request.user_id,
        modality_states,
        processing_time_ms,
        data_quality,
    )
    if request.text:
        background_tasks.add_task(
            _record_interaction,
            request.user_id,
            request.text[:500],
            fused.emotion.value,
            fused.valence,
        )

    return EmotionalIntelligenceResponse(
        success=True,
        data=data,
        metadata={"modalities": [m.value for m in modality_states.keys()]},
    )

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
            timestamp=utc_now()
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
        monitor = await _ensure_quality_monitor()
        
        time_window = timedelta(hours=time_window_hours) if time_window_hours else None
        quality_report = monitor.get_quality_report(time_window)
        
        if format == OutputFormat.JSON:
            return quality_report
        if format == OutputFormat.YAML:
            import yaml

            return yaml.dump(quality_report, allow_unicode=True, default_flow_style=False)
        return jsonable_encoder(quality_report)
            
    except Exception as e:
        logger.error(f"Error getting quality report: {e}")
        raise HTTPException(status_code=500, detail="Failed to get quality report")

@router.post("/quality/ground-truth")
async def submit_ground_truth(
    request: GroundTruthRequest,
):
    """提交真实标签数据用于质量评估"""
    try:
        from src.ai.emotion_modeling.quality_monitor import GroundTruthData
        monitor = await _ensure_quality_monitor()
        
        true_emotion_state = EmotionState(
            emotion=request.true_emotion,
            intensity=request.intensity,
            valence=request.valence,
            arousal=request.arousal,
            dominance=request.dominance,
            confidence=1.0,
            timestamp=utc_now()
        )
        
        ground_truth = GroundTruthData(
            user_id=request.user_id,
            timestamp=utc_now(),
            true_emotion=true_emotion_state,
            modality=request.modality,
            source=request.source,
            confidence=1.0
        )
        
        await monitor.add_ground_truth(ground_truth)
        
        return {
            "success": True,
            "message": "Ground truth data submitted successfully",
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error submitting ground truth: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit ground truth")

@router.get("/health")
async def health_check():
    """健康检查端点"""
    services = []
    # Redis
    try:
        start = time.perf_counter()
        redis_client = get_redis()
        if not redis_client:
            raise RuntimeError("Redis未初始化")
        await redis_client.ping()
        services.append({"service": "redis", "status": "healthy", "latency_ms": (time.perf_counter() - start) * 1000})
    except Exception as e:
        services.append({"service": "redis", "status": "unhealthy", "error": str(e)})

    # Postgres
    try:
        start = time.perf_counter()
        from src.core.database import test_database_connection

        ok = await test_database_connection()
        services.append({"service": "postgres", "status": "healthy" if ok else "unhealthy", "latency_ms": (time.perf_counter() - start) * 1000})
    except Exception as e:
        services.append({"service": "postgres", "status": "unhealthy", "error": str(e)})

    # Qdrant
    try:
        start = time.perf_counter()
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get("http://127.0.0.1:6333/healthz")
            r.raise_for_status()
        services.append({"service": "qdrant", "status": "healthy", "latency_ms": (time.perf_counter() - start) * 1000})
    except Exception as e:
        services.append({"service": "qdrant", "status": "unhealthy", "error": str(e)})

    # 内部分析器
    services.append({"service": "text_analyzer", "status": "healthy" if _text_analyzer.is_initialized else "uninitialized"})
    services.append({"service": "audio_analyzer", "status": "healthy" if _audio_analyzer.is_initialized else "uninitialized"})
    services.append({"service": "visual_analyzer", "status": "healthy" if _visual_analyzer.is_initialized else "uninitialized"})

    return {
        "services": services,
        "timestamp": utc_now().isoformat(),
    }

@router.get("/metrics")
async def get_metrics():
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis未初始化")
    raw = await redis_client.lrange(_INTERACTIONS_KEY, 0, 199)
    interactions = []
    for item in raw:
        try:
            interactions.append(json.loads(item))
        except Exception:
            continue
    user_ids = {i.get("user_id") for i in interactions if i.get("user_id")}
    valences = [float(i.get("valence")) for i in interactions if i.get("valence") is not None]
    return {
        "active_users": len(user_ids),
        "messages": len(interactions),
        "sentiment_score": sum(valences) / len(valences) if valences else 0.0,
    }

@router.get("/interactions")
async def get_interactions(limit: int = 50):
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis未初始化")
    raw = await redis_client.lrange(_INTERACTIONS_KEY, 0, max(limit - 1, 0))
    interactions = []
    for item in raw:
        try:
            interactions.append(json.loads(item))
        except Exception:
            continue
    return {"interactions": interactions}

@router.get("/alerts")
async def get_alerts():
    monitor = await _ensure_quality_monitor()
    alerts = []
    for alert in monitor.active_alerts.values():
        if alert.resolved:
            continue
        alerts.append(
            {
                "id": alert.alert_id,
                "level": alert.severity.value,
                "message": alert.message,
                "created_at": alert.timestamp.isoformat(),
            }
        )
    return {"alerts": alerts}
