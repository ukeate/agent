"""
情感记忆管理系统 API
Story 11.4 - 长期情感记忆存储、检索和模式分析
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body, BackgroundTasks, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import asyncio
import numpy as np
from enum import Enum

from ...core.dependencies import get_db, get_current_user, get_redis
from ...services.emotional_memory_service import EmotionalMemoryService
from ...repositories.emotional_memory_repository import (
    EmotionalMemoryRepository,
    EmotionalEventRepository,
    UserPreferenceRepository,
    TriggerPatternRepository
)
from ...core.security.encryption import EncryptionService
from ...core.monitoring.metrics_collector import MetricsCollector
import structlog

logger = structlog.get_logger(__name__)

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

class EmotionalMemory(BaseModel):
    """情感记忆数据模型"""
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    emotion_type: EmotionType
    intensity: float = Field(ge=0.0, le=1.0)
    context: str
    trigger_factors: List[str] = []
    importance_score: float = Field(ge=0.0, le=1.0)
    related_memories: List[str] = []
    storage_layer: StorageLayer = StorageLayer.HOT
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

class EmotionalEvent(BaseModel):
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

class UserPreference(BaseModel):
    """用户偏好模型"""
    category: str
    preference: str
    confidence: float = Field(ge=0.0, le=1.0)
    effectiveness: float = Field(ge=0.0, le=1.0)
    last_updated: datetime = Field(default_factory=datetime.now)

class TriggerPattern(BaseModel):
    """触发模式模型"""
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str
    pattern_type: str
    trigger_conditions: Dict[str, Any]
    frequency: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    last_occurred: datetime
    prediction_accuracy: float = Field(ge=0.0, le=1.0)

class MemorySearchQuery(BaseModel):
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
    encryption_service = EncryptionService()
    metrics_collector = MetricsCollector()
    
    memory_repo = EmotionalMemoryRepository(
        db_session=db,
        redis_client=redis,
        encryption_service=encryption_service,
        metrics_collector=metrics_collector
    )
    
    event_repo = EmotionalEventRepository(
        db_session=db,
        metrics_collector=metrics_collector
    )
    
    preference_repo = UserPreferenceRepository(
        db_session=db,
        metrics_collector=metrics_collector
    )
    
    pattern_repo = TriggerPatternRepository(
        db_session=db,
        metrics_collector=metrics_collector
    )
    
    # 创建服务实例
    postgres_url = "postgresql://localhost/emotional_memory"  # 从配置读取
    
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
            user_id=current_user['id'],
            emotion_data=emotion_data
        )
        
        # 异步任务：检测事件和学习偏好
        background_tasks.add_task(
            service.detect_emotional_events,
            current_user['id']
        )
        background_tasks.add_task(
            service.learn_user_preferences,
            current_user['id']
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
            user_id=current_user['id'],
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
            user_id=current_user['id'],
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
            user_id=current_user['id'],
            time_window=timedelta(hours=time_window)
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to detect events: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to detect emotional events: {str(e)}"
        )

@router.get("/events/{user_id}", response_model=List[EmotionalEvent])
async def get_emotional_events(
    user_id: str,
    event_type: Optional[str] = Query(None),
    significance_min: float = Query(0.0, ge=0.0, le=1.0),
    limit: int = Query(10, le=100)
):
    """获取用户的情感事件"""
    if user_id not in event_store:
        return []
    
    events = event_store[user_id]
    
    # 应用筛选条件
    if event_type:
        events = [e for e in events if e.event_type == event_type]
    if significance_min > 0:
        events = [e for e in events if e.significance_score >= significance_min]
    
    # 按时间排序
    events.sort(key=lambda x: x.start_time, reverse=True)
    
    return events[:limit]

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
            user_id=current_user['id'],
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
            user_id=current_user['id'],
            min_frequency=min_frequency
        )
        
        return patterns
        
    except Exception as e:
        logger.error(f"Failed to identify patterns: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to identify trigger patterns: {str(e)}"
        )

@router.post("/storage/optimize", response_model=Dict[str, int])
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

# 保留剩余的辅助函数

def calculate_relevance(query: str, context: str) -> float:
    """计算查询和上下文的相关度"""
    # 简单的基于关键词的相似度
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    
    if not query_words:
        return 0.0
    
    intersection = query_words & context_words
    return len(intersection) / len(query_words)
    recent_memories = [
        m for m in memory_store[user_id]
        if m.timestamp >= cutoff_time
    ]
    
    # 分析偏好（简化实现）
    preferences = []
    
    # 分析情感表达偏好
    emotion_counts = {}
    for memory in recent_memories:
        emotion_counts[memory.emotion_type] = emotion_counts.get(memory.emotion_type, 0) + 1
    
    most_common_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else None
    
    if most_common_emotion:
        pref = UserPreference(
            category="情感表达",
            preference=f"倾向于{most_common_emotion}",
            confidence=0.75,
            effectiveness=0.8
        )
        preferences.append(pref)
    
    # 分析触发因素偏好
    trigger_counts = {}
    for memory in recent_memories:
        for trigger in memory.trigger_factors:
            trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
    
    if trigger_counts:
        common_trigger = max(trigger_counts, key=trigger_counts.get)
        pref = UserPreference(
            category="触发因素",
            preference=f"对{common_trigger}敏感",
            confidence=0.7,
            effectiveness=0.75
        )
        preferences.append(pref)
    
    # 存储偏好
    if user_id not in preference_store:
        preference_store[user_id] = []
    preference_store[user_id] = preferences  # 替换旧偏好
    
    return preferences

@router.get("/preferences/{user_id}", response_model=List[UserPreference])
async def get_user_preferences(user_id: str):
    """获取用户情感偏好"""
    if user_id not in preference_store:
        return []
    return preference_store[user_id]

@router.post("/patterns/identify", response_model=List[TriggerPattern])
async def identify_trigger_patterns(
    user_id: str = Query(...),
    analysis_window: int = Query(60, description="Days to analyze")
):
    """识别情感触发模式"""
    if user_id not in memory_store:
        return []
    
    # 获取分析窗口内的记忆
    cutoff_time = datetime.now() - timedelta(days=analysis_window)
    memories = [
        m for m in memory_store[user_id]
        if m.timestamp >= cutoff_time
    ]
    
    patterns = []
    
    # 分析时间模式（简化实现）
    morning_emotions = [m for m in memories if 6 <= m.timestamp.hour < 12]
    if morning_emotions:
        avg_intensity = np.mean([m.intensity for m in morning_emotions])
        pattern = TriggerPattern(
            user_id=user_id,
            pattern_type="时间触发",
            trigger_conditions={"time_of_day": "morning", "hour_range": [6, 12]},
            frequency=len(morning_emotions) / max(len(memories), 1),
            confidence=0.7,
            last_occurred=max(m.timestamp for m in morning_emotions),
            prediction_accuracy=0.75
        )
        patterns.append(pattern)
    
    # 分析情感序列模式
    if len(memories) >= 2:
        for i in range(1, len(memories)):
            if memories[i-1].emotion_type == EmotionType.ANXIETY and \
               memories[i].emotion_type == EmotionType.CALM:
                pattern = TriggerPattern(
                    user_id=user_id,
                    pattern_type="情感转换",
                    trigger_conditions={
                        "from_emotion": "anxiety",
                        "to_emotion": "calm"
                    },
                    frequency=0.3,  # 简化计算
                    confidence=0.65,
                    last_occurred=memories[i].timestamp,
                    prediction_accuracy=0.7
                )
                patterns.append(pattern)
                break  # 只添加一次
    
    # 存储模式
    if user_id not in pattern_store:
        pattern_store[user_id] = []
    pattern_store[user_id] = patterns  # 替换旧模式
    
    return patterns

@router.get("/patterns/{user_id}", response_model=List[TriggerPattern])
async def get_trigger_patterns(
    user_id: str,
    pattern_type: Optional[str] = Query(None),
    confidence_min: float = Query(0.0, ge=0.0, le=1.0)
):
    """获取用户的触发模式"""
    if user_id not in pattern_store:
        return []
    
    patterns = pattern_store[user_id]
    
    # 应用筛选条件
    if pattern_type:
        patterns = [p for p in patterns if p.pattern_type == pattern_type]
    if confidence_min > 0:
        patterns = [p for p in patterns if p.confidence >= confidence_min]
    
    return patterns

@router.post("/patterns/predict")
async def predict_emotional_risk(
    user_id: str = Query(...),
    current_context: Dict[str, Any] = Body(...)
):
    """预测情感风险"""
    if user_id not in pattern_store:
        return {"risk_level": "low", "predictions": []}
    
    patterns = pattern_store[user_id]
    predictions = []
    
    for pattern in patterns:
        # 简单的上下文匹配
        match_score = calculate_context_match(current_context, pattern.trigger_conditions)
        
        if match_score > 0.6:
            risk_level = match_score * pattern.confidence
            predictions.append({
                "pattern_type": pattern.pattern_type,
                "risk_level": risk_level,
                "confidence": pattern.confidence,
                "suggested_actions": generate_preventive_actions(pattern)
            })
    
    # 确定总体风险级别
    if predictions:
        max_risk = max(p["risk_level"] for p in predictions)
        risk_level = "high" if max_risk > 0.7 else "medium" if max_risk > 0.4 else "low"
    else:
        risk_level = "low"
    
    return {
        "risk_level": risk_level,
        "predictions": sorted(predictions, key=lambda x: x["risk_level"], reverse=True)
    }

@router.delete("/memories/{user_id}/{memory_id}")
async def delete_memory(user_id: str, memory_id: str):
    """删除特定记忆（隐私保护）"""
    if user_id not in memory_store:
        raise HTTPException(status_code=404, detail="User not found")
    
    memory_store[user_id] = [
        m for m in memory_store[user_id]
        if m.memory_id != memory_id
    ]
    
    return {"message": "Memory deleted successfully"}

@router.post("/memories/export/{user_id}")
async def export_memories(
    user_id: str,
    format: str = Query("json", enum=["json", "csv"])
):
    """导出用户情感记忆数据"""
    if user_id not in memory_store:
        return {"data": [], "format": format}
    
    memories = memory_store[user_id]
    
    if format == "json":
        return {
            "data": [m.dict() for m in memories],
            "format": "json",
            "count": len(memories)
        }
    else:
        # CSV格式（简化实现）
        return {
            "message": "CSV export not yet implemented",
            "format": "csv"
        }

@router.get("/statistics/{user_id}")
async def get_memory_statistics(user_id: str):
    """获取记忆统计信息"""
    if user_id not in memory_store:
        return {
            "total_memories": 0,
            "storage_distribution": {},
            "emotion_distribution": {},
            "avg_intensity": 0
        }
    
    memories = memory_store[user_id]
    
    # 计算统计信息
    storage_dist = {}
    emotion_dist = {}
    intensities = []
    
    for memory in memories:
        # 存储层分布
        layer = memory.storage_layer
        storage_dist[layer] = storage_dist.get(layer, 0) + 1
        
        # 情感类型分布
        emotion = memory.emotion_type
        emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1
        
        # 强度
        intensities.append(memory.intensity)
    
    return {
        "total_memories": len(memories),
        "storage_distribution": storage_dist,
        "emotion_distribution": emotion_dist,
        "avg_intensity": np.mean(intensities) if intensities else 0,
        "events_count": len(event_store.get(user_id, [])),
        "patterns_count": len(pattern_store.get(user_id, [])),
        "preferences_count": len(preference_store.get(user_id, []))
    }

# 辅助函数

def calculate_relevance(query: str, context: str) -> float:
    """计算查询与记忆的相关度（简化实现）"""
    query_words = set(query.lower().split())
    context_words = set(context.lower().split())
    
    if not query_words:
        return 0.0
    
    intersection = query_words & context_words
    return len(intersection) / len(query_words)

def calculate_context_match(current: Dict[str, Any], conditions: Dict[str, Any]) -> float:
    """计算当前上下文与触发条件的匹配度"""
    if not conditions:
        return 0.0
    
    matches = 0
    total = len(conditions)
    
    for key, value in conditions.items():
        if key in current and current[key] == value:
            matches += 1
    
    return matches / total if total > 0 else 0.0

def generate_preventive_actions(pattern: TriggerPattern) -> List[str]:
    """生成预防性建议"""
    actions = []
    
    if pattern.pattern_type == "时间触发":
        actions.append("调整日程安排")
        actions.append("设置提醒和准备")
    elif pattern.pattern_type == "情感转换":
        actions.append("练习情绪调节技巧")
        actions.append("寻求支持和陪伴")
    else:
        actions.append("保持觉察和记录")
    
    return actions

async def analyze_memory_patterns(user_id: str, new_memory: EmotionalMemory):
    """异步分析记忆模式（后台任务）"""
    # 这里可以添加复杂的模式分析逻辑
    # 例如：更新用户画像、调整推荐策略等
    await asyncio.sleep(0.1)  # 模拟异步处理
    print(f"Analyzed memory patterns for user {user_id}")