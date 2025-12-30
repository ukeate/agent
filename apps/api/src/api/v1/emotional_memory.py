"""
情感记忆管理系统 API
Story 11.4 - 长期情感记忆存储、检索和模式分析
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body, BackgroundTasks, status
from pydantic import Field
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import asyncio
import numpy as np
from enum import Enum
from src.core.database import get_db
from src.core.dependencies import get_current_user, get_redis
from src.services.emotional_memory_service import EmotionalMemoryService
from src.repositories.emotional_memory_repository import (
    EmotionalMemoryRepository,
    EmotionalEventRepository,
    UserPreferenceRepository,
    TriggerPatternRepository
)
from src.api.base_model import ApiBaseModel
from src.core.security.encryption import EncryptionService
from src.core.monitoring.metrics_collector import MetricsCollector
from src.core.config import get_settings

from src.core.logging import get_logger
logger = get_logger(__name__)

_encryption_service = EncryptionService()
_metrics_collector = MetricsCollector()

# 创建路由
router = APIRouter(prefix="/emotional-memory", tags=["Emotional Memory"])

# 数据模型定义
class StorageLayer(str, Enum):
    HOT = "hot"     # 7天内
    WARM = "warm"   # 6个月内  
    COLD = "cold"   # 长期存储

class EmotionType(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CALM = "calm"
    ANXIETY = "anxiety"
    EXCITEMENT = "excitement"

class EmotionalMemory(ApiBaseModel):
    """情感记忆数据模型"""
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    emotion_type: EmotionType
    intensity: float = Field(ge=0.0, le=1.0)
    context: str
    trigger_factors: List[str] = Field(default_factory=list)
    importance_score: float = Field(ge=0.0, le=1.0)
    related_memories: List[str] = Field(default_factory=list)
    storage_layer: StorageLayer = StorageLayer.HOT
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EmotionalEvent(ApiBaseModel):
    """情感事件模型"""
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    event_type: str  # 突变、周期、异常、转折
    start_time: datetime
    end_time: datetime
    peak_intensity: float = Field(ge=0.0, le=1.0)
    emotion_sequence: List[str]
    causal_factors: List[str]
    significance_score: float = Field(ge=0.0, le=1.0)
    recovery_time: int  # 分钟
    confidence: float = Field(ge=0.0, le=1.0)

class UserPreference(ApiBaseModel):
    """用户偏好模型"""
    category: str
    preference: str
    confidence: float = Field(ge=0.0, le=1.0)
    effectiveness: float = Field(ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.now)

class TriggerPattern(ApiBaseModel):
    """触发模式模型"""
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    pattern_type: str
    trigger_conditions: Dict[str, Any]
    frequency: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    last_occurred: datetime
    prediction_accuracy: float = Field(ge=0.0, le=1.0)

class MemorySearchQuery(ApiBaseModel):
    """记忆搜索查询"""
    query: str
    emotion_filter: Optional[List[EmotionType]] = None
    importance_min: Optional[float] = Field(None, ge=0.0, le=1.0)
    time_range: Optional[List[datetime]] = None
    storage_layer: Optional[StorageLayer] = None
    limit: int = Field(default=10, le=100)

# 依赖注入函数
async def get_emotional_memory_service(
    db: AsyncSession = Depends(get_db),
    redis = Depends(get_redis),
    current_user = Depends(get_current_user)
) -> EmotionalMemoryService:
    """获取情感记忆服务实例"""
    
    # 初始化各个repository
    memory_repo = EmotionalMemoryRepository(
        db_session=db,
        redis_client=redis,
        encryption_service=_encryption_service,
        metrics_collector=_metrics_collector
    )
    
    event_repo = EmotionalEventRepository(
        db_session=db,
        metrics_collector=_metrics_collector
    )
    
    preference_repo = UserPreferenceRepository(
        db_session=db,
        metrics_collector=_metrics_collector
    )
    
    pattern_repo = TriggerPatternRepository(
        db_session=db,
        metrics_collector=_metrics_collector
    )
    
    # 创建服务实例
    postgres_url = get_settings().DATABASE_URL
    
    service = EmotionalMemoryService(
        memory_repo=memory_repo,
        event_repo=event_repo,
        preference_repo=preference_repo,
        pattern_repo=pattern_repo,
        redis_client=redis,
        postgres_url=postgres_url
    )
    
    return service

# API端点实现

@router.post("/memories", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_memory(
    memory: EmotionalMemory,
    background_tasks: BackgroundTasks,
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    """创建新的情感记忆"""
    try:
        # 准备情感数据
        emotion_data = {
            'session_id': str(uuid.uuid4()),
            'emotion_type': memory.emotion_type.value,
            'intensity': memory.intensity,
            'content': memory.context,
            'tags': memory.tags,
            'context': memory.metadata,
            'valence': memory.metadata.get('valence', 0),
            'arousal': memory.metadata.get('arousal', 0.5)
        }
        
        # 创建记忆
        result = await service.create_memory(
            user_id=current_user,
            emotion_data=emotion_data
        )
        
        # 异步任务：检测事件和学习偏好
        background_tasks.add_task(
            service.detect_emotional_events,
            current_user
        )
        background_tasks.add_task(
            service.learn_user_preferences,
            current_user
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to create memory: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create emotional memory: {str(e)}"
        )

@router.get("/memories", response_model=List[Dict[str, Any]])
async def get_user_memories(
    storage_layer: Optional[StorageLayer] = Query(None),
    emotion_type: Optional[EmotionType] = Query(None),
    importance_min: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(50, le=100),
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    """获取用户的情感记忆"""
    try:
        # 构建筛选条件
        filters = {}
        if storage_layer:
            filters['storage_layer'] = storage_layer.value
        if emotion_type:
            filters['emotion_type'] = emotion_type.value
        if importance_min > 0:
            filters['min_intensity'] = importance_min
        
        # 获取记忆
        memories = await service.get_memories(
            user_id=current_user,
            filters=filters,
            limit=limit
        )
        
        return memories
        
    except Exception as e:
        logger.error(f"Failed to get memories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memories: {str(e)}"
        )

@router.post("/memories/search", response_model=List[Dict[str, Any]])
async def search_memories(
    search_query: MemorySearchQuery = Body(...),
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    """语义搜索情感记忆"""
    try:
        # 执行语义搜索
        results = await service.search_memories_semantic(
            user_id=current_user,
            query=search_query.query,
            limit=search_query.limit
        )
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to search memories: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search memories: {str(e)}"
        )

@router.post("/events/detect", response_model=Dict[str, Any])
async def detect_emotional_events(
    time_window: int = Query(24, description="Hours to analyze"),
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    """检测重要情感事件"""
    try:
        # 检测情感事件
        result = await service.detect_emotional_events(
            user_id=current_user,
            time_window=timedelta(hours=time_window)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to detect events: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect emotional events: {str(e)}"
        )

@router.post("/preferences/learn", response_model=Dict[str, Any])
async def learn_user_preferences(
    feedback_data: Optional[Dict[str, Any]] = Body(None),
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    """学习用户情感偏好"""
    try:
        # 学习用户偏好
        preferences = await service.learn_user_preferences(
            user_id=current_user,
            feedback_data=feedback_data
        )
        
        return preferences
        
    except Exception as e:
        logger.error(f"Failed to learn preferences: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to learn user preferences: {str(e)}"
        )

@router.get("/patterns/triggers", response_model=List[Dict[str, Any]])
async def identify_trigger_patterns(
    min_frequency: int = Query(3, description="Minimum occurrence frequency"),
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    """识别情感触发模式"""
    try:
        # 识别触发模式
        patterns = await service.identify_trigger_patterns(
            user_id=current_user,
            min_frequency=min_frequency
        )
        
        return patterns
        
    except Exception as e:
        logger.error(f"Failed to identify patterns: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to identify trigger patterns: {str(e)}"
        )

@router.post("/storage/optimize", response_model=Dict[str, str])
async def optimize_storage(
    background_tasks: BackgroundTasks,
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    """优化存储层分配"""
    try:
        # 在后台执行优化
        background_tasks.add_task(
            service.optimize_storage_tiers
        )
        
        return {
            "message": "Storage optimization initiated",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to optimize storage: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to optimize storage: {str(e)}"
        )

@router.get("/events/{user_id}", response_model=List[Dict[str, Any]])
async def get_emotional_events(
    user_id: str,
    limit: int = Query(10, le=100),
    offset: int = Query(0, ge=0),
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="无权访问其他用户数据")
    events = await service.event_repo.get_user_events(user_id=user_id, limit=limit, offset=offset)
    return [service._serialize_event(e) for e in events]

@router.get("/preferences/{user_id}", response_model=Dict[str, Any])
async def get_user_preferences(
    user_id: str,
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="无权访问其他用户数据")
    preferences = await service.preference_repo.get_or_create_preferences(user_id)
    return service._serialize_preferences(preferences)

@router.post("/patterns/identify", response_model=List[Dict[str, Any]])
async def identify_patterns(
    user_id: str = Query(...),
    min_frequency: int = Query(3, ge=1),
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="无权访问其他用户数据")
    return await service.identify_trigger_patterns(user_id=user_id, min_frequency=min_frequency)

@router.get("/patterns/{user_id}", response_model=List[Dict[str, Any]])
async def get_trigger_patterns(
    user_id: str,
    confidence_min: float = Query(0.0, ge=0.0, le=1.0),
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="无权访问其他用户数据")
    patterns = await service.pattern_repo.get_active_patterns(user_id=user_id, min_confidence=confidence_min)
    return [service._serialize_pattern(p) for p in patterns]

def _calculate_context_match(current: Dict[str, Any], conditions: Dict[str, Any]) -> float:
    if not conditions:
        return 0.0
    matches = 0
    total = len(conditions)
    for key, value in conditions.items():
        if key in current and current[key] == value:
            matches += 1
    return matches / total if total > 0 else 0.0

@router.post("/patterns/predict")
async def predict_emotional_risk(
    user_id: str = Query(...),
    current_context: Dict[str, Any] = Body(...),
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="无权访问其他用户数据")

    patterns = await service.pattern_repo.get_active_patterns(user_id=user_id, min_confidence=0.0)
    predictions = []
    for pattern in patterns:
        match_score = _calculate_context_match(current_context, pattern.trigger_conditions or {})
        if match_score > 0.6:
            predictions.append({
                "pattern_type": pattern.pattern_type,
                "risk_level": match_score * float(pattern.confidence or 0.0),
                "confidence": float(pattern.confidence or 0.0),
                "recommended_responses": pattern.recommended_responses or [],
            })

    if predictions:
        max_risk = max(p["risk_level"] for p in predictions)
        risk_level = "high" if max_risk > 0.7 else "medium" if max_risk > 0.4 else "low"
    else:
        risk_level = "low"

    return {"risk_level": risk_level, "predictions": sorted(predictions, key=lambda x: x["risk_level"], reverse=True)}

@router.delete("/memories/{user_id}/{memory_id}")
async def delete_memory(
    user_id: str,
    memory_id: str,
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="无权访问其他用户数据")
    try:
        memory_uuid = uuid.UUID(memory_id)
    except Exception:
        raise HTTPException(status_code=400, detail="无效的memory_id")
    deleted = await service.memory_repo.delete_memory(memory_id=memory_uuid, user_id=user_id, hard_delete=False)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found")
    return {"message": "Memory deleted successfully"}

@router.post("/memories/export/{user_id}")
async def export_memories(
    user_id: str,
    format: str = Query("json", enum=["json", "csv"]),
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="无权访问其他用户数据")

    memories, _ = await service.memory_repo.search_memories(user_id=user_id, filters={}, limit=10000, offset=0)
    data = [service._serialize_memory(m) for m in memories]
    if format == "json":
        return {"data": data, "format": "json", "count": len(data)}

    import csv
    import io

    buf = io.StringIO()
    fieldnames = ["id", "timestamp", "emotion_type", "intensity", "content", "storage_layer", "importance_score"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow({k: row.get(k) for k in fieldnames})
    return {"data": buf.getvalue(), "format": "csv", "count": len(data)}

@router.get("/statistics/{user_id}")
async def get_memory_statistics(
    user_id: str,
    service: EmotionalMemoryService = Depends(get_emotional_memory_service),
    current_user = Depends(get_current_user)
):
    if user_id != current_user:
        raise HTTPException(status_code=403, detail="无权访问其他用户数据")

    from src.db.emotional_memory_models import EmotionalMemory as DbMemory

    db = service.memory_repo.db
    total = (await db.execute(select(func.count()).where(DbMemory.user_id == user_id, DbMemory.deleted_at.is_(None)))).scalar() or 0
    avg_intensity = (await db.execute(select(func.avg(DbMemory.intensity)).where(DbMemory.user_id == user_id, DbMemory.deleted_at.is_(None)))).scalar() or 0

    storage_rows = (await db.execute(
        select(DbMemory.storage_layer, func.count()).where(DbMemory.user_id == user_id, DbMemory.deleted_at.is_(None)).group_by(DbMemory.storage_layer)
    )).all()
    emotion_rows = (await db.execute(
        select(DbMemory.emotion_type, func.count()).where(DbMemory.user_id == user_id, DbMemory.deleted_at.is_(None)).group_by(DbMemory.emotion_type)
    )).all()

    storage_distribution = {str(layer.value if hasattr(layer, 'value') else layer): int(count) for layer, count in storage_rows}
    emotion_distribution = {str(emotion): int(count) for emotion, count in emotion_rows}

    return {
        "total_memories": int(total),
        "storage_distribution": storage_distribution,
        "emotion_distribution": emotion_distribution,
        "avg_intensity": float(avg_intensity),
    }
