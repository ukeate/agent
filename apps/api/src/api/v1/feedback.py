"""
用户反馈系统API端点实现

提供隐式和显式反馈收集、处理、分析的REST API接口
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
import asyncio

from ...core.database import get_db_session
from ...core.logging import get_logger

logger = get_logger(__name__)

# 创建路由器
router = APIRouter(prefix="/feedback", tags=["Feedback"])
security = HTTPBearer()

# 反馈类型枚举
class FeedbackType(str, Enum):
    # 隐式反馈
    CLICK = "click"
    VIEW = "view"
    DWELL_TIME = "dwell_time"
    SCROLL_DEPTH = "scroll_depth"
    HOVER = "hover"
    FOCUS = "focus"
    BLUR = "blur"
    
    # 显式反馈
    RATING = "rating"
    LIKE = "like"
    DISLIKE = "dislike"
    BOOKMARK = "bookmark"
    SHARE = "share"
    COMMENT = "comment"

# Pydantic模型定义
class FeedbackContext(BaseModel):
    """反馈上下文信息"""
    url: str
    page_title: str = Field(default="", description="页面标题")
    element_id: Optional[str] = None
    element_type: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    viewport: Optional[Dict[str, int]] = None
    timestamp: int
    user_agent: str

class FeedbackEvent(BaseModel):
    """反馈事件模型"""
    event_id: str
    user_id: str
    session_id: str
    item_id: Optional[str] = None
    feedback_type: FeedbackType
    value: Union[int, float, str, bool]
    raw_value: Optional[Any] = None
    context: FeedbackContext
    metadata: Optional[Dict[str, Any]] = None

    @validator('value')
    def validate_feedback_value(cls, v, values):
        """验证反馈值的合理性"""
        feedback_type = values.get('feedback_type')
        
        if feedback_type == FeedbackType.RATING:
            if not isinstance(v, (int, float)) or not (1 <= v <= 5):
                raise ValueError('评分必须在1-5之间')
        elif feedback_type in [FeedbackType.LIKE, FeedbackType.DISLIKE, FeedbackType.BOOKMARK]:
            if not isinstance(v, (int, bool)):
                raise ValueError('点赞/收藏反馈必须是布尔值或0/1')
        elif feedback_type == FeedbackType.SCROLL_DEPTH:
            if not isinstance(v, (int, float)) or not (0 <= v <= 100):
                raise ValueError('滚动深度必须在0-100%之间')
        elif feedback_type == FeedbackType.DWELL_TIME:
            if not isinstance(v, (int, float)) or v < 0:
                raise ValueError('停留时间必须为非负数')
        
        return v

class FeedbackBatch(BaseModel):
    """反馈事件批次"""
    batch_id: str
    user_id: str
    session_id: str
    events: List[FeedbackEvent]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processed_at: Optional[datetime] = None

class ExplicitFeedbackRequest(BaseModel):
    """显式反馈提交请求"""
    user_id: str
    session_id: str
    item_id: Optional[str] = None
    feedback_type: FeedbackType
    value: Union[int, float, str, bool]
    context: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

class FeedbackHistoryQuery(BaseModel):
    """反馈历史查询参数"""
    user_id: str
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    feedback_types: Optional[List[FeedbackType]] = None
    item_id: Optional[str] = None
    limit: int = Field(default=100, le=1000)
    offset: int = Field(default=0, ge=0)

class FeedbackAnalyticsResponse(BaseModel):
    """用户反馈分析响应"""
    user_id: str
    total_feedbacks: int
    feedback_distribution: Dict[FeedbackType, int]
    average_rating: Optional[float] = None
    engagement_score: float
    last_activity: Optional[datetime] = None
    preference_vector: List[float] = Field(default_factory=list)
    trust_score: float = Field(default=1.0)

class ItemFeedbackAnalyticsResponse(BaseModel):
    """推荐项反馈分析响应"""
    item_id: str
    total_feedbacks: int
    average_rating: Optional[float] = None
    like_ratio: float
    engagement_metrics: Dict[str, float]
    feedback_distribution: Dict[FeedbackType, int]

class FeedbackQualityScore(BaseModel):
    """反馈质量评分"""
    feedback_id: str
    quality_score: float = Field(ge=0.0, le=1.0)
    quality_factors: Dict[str, float]
    is_valid: bool
    reasons: Optional[List[str]] = None

class ApiResponse(BaseModel):
    """API统一响应格式"""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    message: Optional[str] = None

# WebSocket连接管理器
class FeedbackWebSocketManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket连接已建立，当前连接数: {len(self.active_connections)}")
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket连接已断开，当前连接数: {len(self.active_connections)}")
        
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
        
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.error(f"广播消息失败: {e}")

manager = FeedbackWebSocketManager()

# API端点实现
@router.post("/implicit", response_model=ApiResponse)
async def submit_implicit_feedback(
    batch: FeedbackBatch,
    background_tasks: BackgroundTasks
):
    """
    提交隐式反馈批次
    """
    try:
        logger.info(f"收到隐式反馈批次: {batch.batch_id}, 事件数: {len(batch.events)}")
        
        # 验证批次数据
        if not batch.events:
            raise HTTPException(status_code=400, detail="反馈批次不能为空")
        
        if len(batch.events) > 100:
            raise HTTPException(status_code=400, detail="单次批次事件数不能超过100")
        
        # 添加后台处理任务
        background_tasks.add_task(process_feedback_batch, batch)
        
        return ApiResponse(
            success=True,
            message=f"成功接收{len(batch.events)}个隐式反馈事件",
            data={"batch_id": batch.batch_id}
        )
        
    except Exception as e:
        logger.error(f"提交隐式反馈失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@router.post("/explicit", response_model=ApiResponse)
async def submit_explicit_feedback(
    feedback: ExplicitFeedbackRequest,
    background_tasks: BackgroundTasks
):
    """
    提交显式反馈
    """
    try:
        logger.info(f"收到显式反馈: 用户{feedback.user_id}, 类型{feedback.feedback_type}")
        
        # 创建反馈事件
        event = FeedbackEvent(
            event_id=f"explicit-{int(datetime.now().timestamp() * 1000)}",
            user_id=feedback.user_id,
            session_id=feedback.session_id,
            item_id=feedback.item_id,
            feedback_type=feedback.feedback_type,
            value=feedback.value,
            context=FeedbackContext(
                url=feedback.context.get('url', '') if feedback.context else '',
                page_title=feedback.context.get('page_title', '') if feedback.context else '',
                timestamp=int(datetime.now().timestamp() * 1000),
                user_agent=feedback.context.get('user_agent', '') if feedback.context else ''
            ),
            metadata=feedback.metadata
        )
        
        # 添加后台处理任务
        background_tasks.add_task(process_single_feedback, event)
        
        return ApiResponse(
            success=True,
            message="显式反馈提交成功",
            data={"event_id": event.event_id}
        )
        
    except Exception as e:
        logger.error(f"提交显式反馈失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@router.get("/user/{user_id}", response_model=ApiResponse)
async def get_user_feedback_history(
    user_id: str,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    feedback_types: Optional[str] = None,
    item_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    获取用户反馈历史
    """
    try:
        # 解析反馈类型
        parsed_types = None
        if feedback_types:
            parsed_types = [FeedbackType(t.strip()) for t in feedback_types.split(',')]
        
        # 构造查询
        query = FeedbackHistoryQuery(
            user_id=user_id,
            start_date=start_date,
            end_date=end_date,
            feedback_types=parsed_types,
            item_id=item_id,
            limit=limit,
            offset=offset
        )
        
        # 模拟查询结果（实际应从数据库查询）
        history = await get_feedback_history(query)
        
        return ApiResponse(
            success=True,
            data={
                "total": len(history),
                "items": history,
                "query": query.dict()
            }
        )
        
    except Exception as e:
        logger.error(f"获取用户反馈历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.get("/analytics/user/{user_id}", response_model=ApiResponse)
async def get_user_feedback_analytics(user_id: str):
    """
    获取用户反馈分析
    """
    try:
        analytics = await get_user_analytics(user_id)
        
        return ApiResponse(
            success=True,
            data=analytics.dict()
        )
        
    except Exception as e:
        logger.error(f"获取用户反馈分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

@router.get("/analytics/item/{item_id}", response_model=ApiResponse)
async def get_item_feedback_analytics(item_id: str):
    """
    获取推荐项反馈分析
    """
    try:
        analytics = await get_item_analytics(item_id)
        
        return ApiResponse(
            success=True,
            data=analytics.dict()
        )
        
    except Exception as e:
        logger.error(f"获取推荐项反馈分析失败: {e}")
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

@router.post("/quality/score", response_model=ApiResponse)
async def get_feedback_quality_score(feedback_ids: List[str]):
    """
    获取反馈质量评分
    """
    try:
        if not feedback_ids:
            raise HTTPException(status_code=400, detail="反馈ID列表不能为空")
        
        if len(feedback_ids) > 50:
            raise HTTPException(status_code=400, detail="单次查询ID数不能超过50")
        
        scores = await calculate_quality_scores(feedback_ids)
        
        return ApiResponse(
            success=True,
            data=scores
        )
        
    except Exception as e:
        logger.error(f"获取反馈质量评分失败: {e}")
        raise HTTPException(status_code=500, detail=f"计算失败: {str(e)}")

@router.post("/process/batch", response_model=ApiResponse)
async def process_feedback_batch_endpoint(
    batch_id: str,
    background_tasks: BackgroundTasks
):
    """
    手动触发反馈批次处理
    """
    try:
        background_tasks.add_task(reprocess_feedback_batch, batch_id)
        
        return ApiResponse(
            success=True,
            message=f"批次{batch_id}处理任务已启动"
        )
        
    except Exception as e:
        logger.error(f"处理反馈批次失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@router.post("/reward/calculate", response_model=ApiResponse)
async def calculate_reward_signal(
    user_id: str,
    item_id: str,
    time_window: int = 3600  # 默认1小时
):
    """
    计算奖励信号
    """
    try:
        reward = await compute_reward_signal(user_id, item_id, time_window)
        
        return ApiResponse(
            success=True,
            data={
                "user_id": user_id,
                "item_id": item_id,
                "reward_signal": reward,
                "time_window": time_window,
                "calculated_at": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"计算奖励信号失败: {e}")
        raise HTTPException(status_code=500, detail=f"计算失败: {str(e)}")

@router.get("/overview", response_model=ApiResponse)
async def get_feedback_overview(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
):
    """
    获取反馈系统概览
    """
    try:
        overview = await get_system_overview(start_date, end_date)
        
        return ApiResponse(
            success=True,
            data=overview
        )
        
    except Exception as e:
        logger.error(f"获取反馈系统概览失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")

@router.get("/metrics/realtime", response_model=ApiResponse)
async def get_realtime_feedback_metrics():
    """
    获取实时反馈指标
    """
    try:
        metrics = await get_realtime_metrics()
        
        return ApiResponse(
            success=True,
            data=metrics
        )
        
    except Exception as e:
        logger.error(f"获取实时反馈指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    反馈系统WebSocket端点
    用于实时传输反馈事件
    """
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get('type') == 'feedback_batch':
                # 处理实时反馈批次
                events = message.get('events', [])
                logger.info(f"WebSocket收到{len(events)}个反馈事件")
                
                # 异步处理事件
                asyncio.create_task(process_websocket_feedback(events))
                
                # 确认收到
                await manager.send_personal_message(
                    json.dumps({
                        'type': 'ack',
                        'message': f'已收到{len(events)}个事件'
                    }),
                    websocket
                )
            elif message.get('type') == 'ping':
                await manager.send_personal_message(
                    json.dumps({'type': 'pong'}),
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket错误: {e}")
        manager.disconnect(websocket)

# 辅助函数实现
async def process_feedback_batch(batch: FeedbackBatch):
    """处理反馈批次"""
    try:
        logger.info(f"开始处理反馈批次: {batch.batch_id}")
        
        # 这里应该实现实际的批次处理逻辑
        # 1. 验证事件数据
        # 2. 存储到数据库
        # 3. 更新用户画像
        # 4. 触发强化学习更新
        # 5. 计算奖励信号
        
        # 模拟处理延迟
        await asyncio.sleep(0.1)
        
        logger.info(f"反馈批次处理完成: {batch.batch_id}")
        
    except Exception as e:
        logger.error(f"处理反馈批次失败: {batch.batch_id}, 错误: {e}")

async def process_single_feedback(event: FeedbackEvent):
    """处理单个反馈事件"""
    try:
        logger.info(f"开始处理反馈事件: {event.event_id}")
        
        # 实际的事件处理逻辑
        # 1. 存储事件
        # 2. 实时更新用户画像
        # 3. 触发推荐更新
        
        await asyncio.sleep(0.05)
        
        logger.info(f"反馈事件处理完成: {event.event_id}")
        
    except Exception as e:
        logger.error(f"处理反馈事件失败: {event.event_id}, 错误: {e}")

async def process_websocket_feedback(events: List[Dict]):
    """处理WebSocket反馈事件"""
    try:
        logger.info(f"处理WebSocket反馈事件: {len(events)}")
        
        # 转换为FeedbackEvent对象并处理
        for event_data in events:
            # 验证和处理事件
            pass
            
        await asyncio.sleep(0.1)
        
    except Exception as e:
        logger.error(f"处理WebSocket反馈失败: {e}")

async def get_feedback_history(query: FeedbackHistoryQuery) -> List[Dict]:
    """获取反馈历史"""
    # 模拟数据查询
    return []

async def get_user_analytics(user_id: str) -> FeedbackAnalyticsResponse:
    """获取用户分析数据"""
    import random
    
    return FeedbackAnalyticsResponse(
        user_id=user_id,
        total_feedbacks=random.randint(50, 500),
        feedback_distribution={
            "rating": random.randint(10, 100),
            "like": random.randint(20, 150),
            "click": random.randint(50, 200),
            "comment": random.randint(5, 50),
            "bookmark": random.randint(3, 30),
            "view": random.randint(100, 300)
        },
        engagement_score=round(random.uniform(0.5, 0.95), 3),
        preferences={
            "categories": ["科技", "教育", "娱乐"],
            "activity_hours": [9, 14, 20, 22],
            "device_preference": "mobile"
        },
        quality_metrics={
            "consistency_score": round(random.uniform(0.7, 0.9), 3),
            "diversity_score": round(random.uniform(0.6, 0.8), 3),
            "authenticity_score": round(random.uniform(0.8, 0.95), 3)
        }
    )

async def get_item_analytics(item_id: str) -> ItemFeedbackAnalyticsResponse:
    """获取推荐项分析数据"""
    import random
    
    # 生成随机的反馈分布
    feedback_distribution = {}
    for feedback_type in FeedbackType:
        feedback_distribution[feedback_type.value] = random.randint(5, 150)
    
    total_feedbacks = sum(feedback_distribution.values())
    
    return ItemFeedbackAnalyticsResponse(
        item_id=item_id,
        total_feedbacks=total_feedbacks,
        average_rating=round(random.uniform(3.5, 4.8), 2) if total_feedbacks > 0 else None,
        like_ratio=round(random.uniform(0.6, 0.9), 3),
        engagement_metrics={
            "click_through_rate": round(random.uniform(0.05, 0.25), 3),
            "dwell_time_avg": round(random.uniform(120, 600), 1),
            "scroll_depth_avg": round(random.uniform(0.4, 0.8), 3),
            "interaction_rate": round(random.uniform(0.1, 0.4), 3),
            "completion_rate": round(random.uniform(0.3, 0.7), 3),
            "bounce_rate": round(random.uniform(0.2, 0.5), 3)
        },
        feedback_distribution=feedback_distribution
    )

async def calculate_quality_scores(feedback_ids: List[str]) -> List[FeedbackQualityScore]:
    """计算反馈质量评分"""
    import random
    
    quality_scores = []
    
    for feedback_id in feedback_ids:
        # 生成质量因子分数
        quality_factors = {
            "consistency": round(random.uniform(0.6, 0.95), 3),
            "timing": round(random.uniform(0.7, 0.9), 3),
            "context_relevance": round(random.uniform(0.5, 0.9), 3),
            "frequency": round(random.uniform(0.4, 0.8), 3),
            "diversity": round(random.uniform(0.6, 0.85), 3),
            "authenticity": round(random.uniform(0.8, 0.95), 3)
        }
        
        # 计算综合质量分数
        overall_score = sum(quality_factors.values()) / len(quality_factors)
        is_valid = overall_score > 0.6
        
        # 生成质量评估原因
        reasons = []
        if quality_factors["consistency"] < 0.7:
            reasons.append("反馈行为一致性偏低")
        if quality_factors["context_relevance"] < 0.6:
            reasons.append("上下文相关性不足")
        if quality_factors["frequency"] < 0.5:
            reasons.append("反馈频率异常")
        if quality_factors["authenticity"] < 0.8:
            reasons.append("真实性存疑")
        
        if not reasons:
            reasons.append("高质量反馈")
        
        quality_score = FeedbackQualityScore(
            feedback_id=feedback_id,
            quality_score=round(overall_score, 3),
            quality_factors=quality_factors,
            is_valid=is_valid,
            reasons=reasons
        )
        
        quality_scores.append(quality_score)
    
    return quality_scores

async def reprocess_feedback_batch(batch_id: str):
    """重新处理反馈批次"""
    logger.info(f"重新处理反馈批次: {batch_id}")

async def compute_reward_signal(user_id: str, item_id: str, time_window: int) -> float:
    """计算奖励信号"""
    import random
    
    # 模拟不同类型的反馈权重
    feedback_weights = {
        FeedbackType.RATING: 0.8,
        FeedbackType.LIKE: 0.6,
        FeedbackType.DISLIKE: -0.4,
        FeedbackType.BOOKMARK: 0.7,
        FeedbackType.SHARE: 0.9,
        FeedbackType.COMMENT: 0.5,
        FeedbackType.CLICK: 0.2,
        FeedbackType.VIEW: 0.1,
        FeedbackType.DWELL_TIME: 0.3,
        FeedbackType.SCROLL_DEPTH: 0.2
    }
    
    # 模拟时间衰减因子 (越近的反馈权重越高)
    time_decay_factor = random.uniform(0.8, 1.0)
    
    # 模拟用户历史行为一致性奖励
    consistency_bonus = random.uniform(0.0, 0.2)
    
    # 模拟推荐项质量分数
    item_quality_score = random.uniform(0.5, 0.9)
    
    # 计算基础奖励信号
    base_reward = 0.0
    
    # 模拟从时间窗口内的反馈数据计算奖励
    simulated_feedbacks = random.randint(3, 15)
    
    for _ in range(simulated_feedbacks):
        feedback_type = random.choice(list(feedback_weights.keys()))
        weight = feedback_weights[feedback_type]
        
        # 根据反馈类型计算价值
        if feedback_type == FeedbackType.RATING:
            value = random.uniform(1, 5)  # 评分1-5
            normalized_value = (value - 1) / 4  # 标准化到0-1
        elif feedback_type in [FeedbackType.LIKE, FeedbackType.BOOKMARK, FeedbackType.SHARE]:
            normalized_value = 1.0  # 明确的正向反馈
        elif feedback_type == FeedbackType.DISLIKE:
            normalized_value = 0.0  # 负向反馈
        else:
            normalized_value = random.uniform(0.3, 0.8)  # 隐式反馈
        
        base_reward += weight * normalized_value
    
    # 应用时间衰减
    time_weighted_reward = base_reward * time_decay_factor
    
    # 添加一致性奖励
    final_reward = time_weighted_reward + consistency_bonus
    
    # 应用推荐项质量调整
    final_reward = final_reward * item_quality_score
    
    # 标准化到 [-1, 1] 范围
    normalized_reward = max(-1.0, min(1.0, final_reward / 10.0))
    
    return round(normalized_reward, 4)

async def get_system_overview(start_date: Optional[datetime], end_date: Optional[datetime]) -> Dict:
    """获取系统概览"""
    # 返回模拟数据用于测试
    return {
        "total_feedbacks": 15420,
        "feedback_types": {
            "rating": 4520,
            "like": 3890,
            "click": 2850,
            "comment": 2120,
            "bookmark": 1580,
            "view": 460
        },
        "unique_users": 1240,
        "average_quality_score": 0.82,
        "top_items": [
            {"item_id": "item-001", "feedback_count": 245},
            {"item_id": "item-002", "feedback_count": 189},
            {"item_id": "item-003", "feedback_count": 167},
            {"item_id": "item-004", "feedback_count": 134},
            {"item_id": "item-005", "feedback_count": 98}
        ]
    }

async def get_realtime_metrics() -> Dict:
    """获取实时指标"""
    import random
    from datetime import datetime
    
    # 返回实时模拟数据
    return {
        "active_sessions": random.randint(50, 200),
        "events_per_minute": random.randint(100, 500),
        "buffer_status": {
            "pending_events": random.randint(0, 100),
            "processed_events": random.randint(1000, 5000),
            "failed_events": random.randint(0, 10),
            "buffer_utilization": round(random.uniform(0.1, 0.8), 2)
        },
        "processing_latency": round(random.uniform(20, 150), 1),
        "quality_score": round(random.uniform(0.75, 0.95), 3),
        "anomaly_count": random.randint(0, 5),
        "last_updated": datetime.now().isoformat()
    }