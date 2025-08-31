"""
情感状态建模系统API接口
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel

from ...core.database import get_db
from ...services.emotion_modeling_service import EmotionModelingService
from ...core.security.auth import get_current_user


router = APIRouter(prefix="/emotion", tags=["emotion-modeling"])


# Pydantic模型
class EmotionStateInput(BaseModel):
    emotion: str
    intensity: float = 0.5
    valence: Optional[float] = None
    arousal: Optional[float] = None  
    dominance: Optional[float] = None
    confidence: float = 1.0
    timestamp: Optional[str] = None
    triggers: List[str] = []
    context: Dict[str, Any] = {}
    source: str = "manual"
    session_id: Optional[str] = None


class PredictionRequest(BaseModel):
    time_horizon_hours: int = 1


class AnalyticsRequest(BaseModel):
    days_back: int = 30


@router.post("/state")
async def record_emotion_state(
    emotion_data: EmotionStateInput,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """记录情感状态"""
    service = EmotionModelingService(db)
    
    result = await service.process_emotion_state(
        user_id, emotion_data.dict()
    )
    
    if 'error' in result:
        raise HTTPException(status_code=400, detail=result['error'])
    
    return result


@router.get("/state/latest")
async def get_latest_emotion_state(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取最新情感状态"""
    service = EmotionModelingService(db)
    
    latest_state = await service.repository.get_latest_emotion_state(user_id)
    if not latest_state:
        raise HTTPException(status_code=404, detail="没有找到情感状态数据")
    
    return latest_state.to_dict()


@router.get("/state/history")
async def get_emotion_history(
    limit: int = 50,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    emotions: Optional[List[str]] = None,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取情感历史"""
    service = EmotionModelingService(db)
    
    start_time = datetime.fromisoformat(start_date) if start_date else None
    end_time = datetime.fromisoformat(end_date) if end_date else None
    
    history = await service.repository.get_user_emotion_history(
        user_id, limit, start_time, end_time, emotions
    )
    
    return [state.to_dict() for state in history]


@router.post("/predict")
async def predict_emotions(
    request: PredictionRequest,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """预测情感状态"""
    service = EmotionModelingService(db)
    
    prediction = await service.get_emotion_prediction(
        user_id, request.time_horizon_hours
    )
    
    if 'error' in prediction:
        raise HTTPException(status_code=400, detail=prediction['error'])
    
    return prediction


@router.post("/analytics")
async def get_emotion_analytics(
    request: AnalyticsRequest,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取情感分析报告"""
    service = EmotionModelingService(db)
    
    analytics = await service.get_emotion_analytics(
        user_id, request.days_back
    )
    
    if 'error' in analytics:
        raise HTTPException(status_code=400, detail=analytics['error'])
    
    return analytics


@router.get("/profile")
async def get_personality_profile(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取个性画像"""
    service = EmotionModelingService(db)
    
    profile = await service.repository.get_personality_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="个性画像不存在")
    
    return profile.to_dict()


@router.get("/patterns")
async def detect_patterns(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """检测情感模式"""
    service = EmotionModelingService(db)
    
    patterns = await service.detect_emotion_patterns(user_id)
    
    if 'error' in patterns:
        raise HTTPException(status_code=400, detail=patterns['error'])
    
    return patterns


@router.get("/clusters")
async def get_emotion_clusters(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取情感聚类分析"""
    service = EmotionModelingService(db)
    
    clusters = await service.prediction_engine.perform_emotion_clustering(user_id)
    return clusters


@router.get("/transitions")
async def get_transition_analysis(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取情感转换分析"""
    service = EmotionModelingService(db)
    
    transitions = await service.repository.get_user_transitions(user_id)
    patterns = service.transition_manager.analyze_transition_patterns(user_id)
    
    return {
        'transitions': [t.to_dict() for t in transitions],
        'patterns': patterns
    }


@router.get("/export")
async def export_data(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """导出用户情感数据"""
    service = EmotionModelingService(db)
    
    data = await service.export_user_data(user_id)
    
    if 'error' in data:
        raise HTTPException(status_code=400, detail=data['error'])
    
    return data


@router.delete("/data")
async def delete_user_data(
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """删除用户数据（右被遗忘）"""
    service = EmotionModelingService(db)
    
    success = await service.delete_user_data(user_id)
    
    if not success:
        raise HTTPException(status_code=500, detail="删除数据失败")
    
    return {"message": "用户数据已删除"}


@router.get("/status")
async def get_system_status(
    db: Session = Depends(get_db)
):
    """获取系统状态"""
    service = EmotionModelingService(db)
    return service.get_system_status()


@router.get("/statistics")
async def get_emotion_statistics(
    days: int = 30,
    user_id: str = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """获取情感统计数据"""
    service = EmotionModelingService(db)
    
    from datetime import timedelta
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    
    stats = await service.repository.get_emotion_statistics(
        user_id, start_time, end_time
    )
    
    return stats.to_dict()


# WebSocket接口（用于实时更新）
@router.websocket("/realtime/{user_id}")
async def emotion_realtime_updates(websocket, user_id: str):
    """实时情感状态更新WebSocket"""
    await websocket.accept()
    
    # 这里应该实现WebSocket的实时推送逻辑
    # 由于时间限制，暂时保持连接
    try:
        while True:
            # 等待客户端消息或发送更新
            await websocket.receive_text()
            
            # 发送状态更新（示例）
            await websocket.send_json({
                "type": "status_update",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        print(f"WebSocket连接错误: {e}")
    finally:
        await websocket.close()