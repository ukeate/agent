from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Request, Body
from fastapi.responses import JSONResponse
from typing import Dict, List, Any, Optional
import asyncio
import json
import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from src.models.schemas.personalization import (
    RecommendationRequest,
    RecommendationResponse,
    UserFeedback,
    UserProfile,
    ModelConfig
)
from src.services.personalization_service import PersonalizationService, get_personalization_service
from src.core.dependencies import get_redis
from redis.asyncio import Redis

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/personalization", tags=["个性化推荐"])

@router.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(
    request: RecommendationRequest,
    background_tasks: BackgroundTasks,
    service: PersonalizationService = Depends(get_personalization_service)
) -> RecommendationResponse:
    """获取个性化推荐
    
    Args:
        request: 推荐请求
        background_tasks: 后台任务
        service: 个性化服务
        
    Returns:
        RecommendationResponse: 推荐响应
    """
    try:
        # 获取推荐
        response = await service.get_recommendations(request)
        
        # 记录请求（后台任务）
        background_tasks.add_task(
            log_recommendation_request,
            request,
            response
        )
        
        return response
        
    except Exception as e:
        logger.error(f"获取推荐失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/user/{user_id}/profile", response_model=UserProfile)
async def get_user_profile(
    user_id: str,
    service: PersonalizationService = Depends(get_personalization_service)
) -> UserProfile:
    """获取用户画像
    
    Args:
        user_id: 用户ID
        service: 个性化服务
        
    Returns:
        UserProfile: 用户画像
    """
    try:
        profile = await service.get_user_profile(user_id)
        
        if not profile:
            raise HTTPException(status_code=404, detail="用户画像不存在")
        
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取用户画像失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/user/{user_id}/profile")
async def update_user_profile(
    user_id: str,
    profile: UserProfile,
    service: PersonalizationService = Depends(get_personalization_service)
) -> JSONResponse:
    """更新用户画像
    
    Args:
        user_id: 用户ID
        profile: 用户画像
        service: 个性化服务
        
    Returns:
        JSONResponse: 更新结果
    """
    try:
        # 确保用户ID一致
        profile.user_id = user_id
        
        # 更新画像
        success = await service.update_user_profile(profile)
        
        if success:
            return JSONResponse(
                content={"status": "success", "message": "用户画像更新成功"}
            )
        else:
            raise HTTPException(status_code=500, detail="更新失败")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"更新用户画像失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/preferences")
async def get_preferences(
    request: Request,
    redis: Redis = Depends(get_redis),
    service: PersonalizationService = Depends(get_personalization_service),
) -> Dict[str, Any]:
    """获取用户偏好配置"""
    user_id = request.query_params.get("user_id") or getattr(request.state, "client_id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 缺失")

    key = f"personalization:preferences:{user_id}"
    if redis:
        cached = await redis.get(key)
        if cached:
            return json.loads(cached)

    features = await service.get_realtime_features(user_id)
    payload = _build_preferences_payload(user_id, features)

    if redis:
        await redis.set(key, json.dumps(payload, ensure_ascii=False))
    return payload

@router.put("/preferences")
async def update_preferences(
    request: Request,
    preferences: Dict[str, Any] = Body(...),
    redis: Redis = Depends(get_redis),
) -> Dict[str, Any]:
    """更新用户偏好配置"""
    user_id = preferences.get("user_id") or request.query_params.get("user_id") or getattr(request.state, "client_id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 缺失")
    if not redis:
        raise HTTPException(status_code=503, detail="Redis不可用")

    payload = {**preferences, "user_id": user_id}
    await redis.set(f"personalization:preferences:{user_id}", json.dumps(payload, ensure_ascii=False))
    return payload

@router.post("/preferences/reset")
async def reset_preferences(
    request: Request,
    redis: Redis = Depends(get_redis),
    service: PersonalizationService = Depends(get_personalization_service),
) -> Dict[str, Any]:
    """重置用户偏好配置"""
    user_id = request.query_params.get("user_id") or getattr(request.state, "client_id", None)
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id 缺失")
    if redis:
        await redis.delete(f"personalization:preferences:{user_id}")

    features = await service.get_realtime_features(user_id)
    payload = _build_preferences_payload(user_id, features)
    if redis:
        await redis.set(f"personalization:preferences:{user_id}", json.dumps(payload, ensure_ascii=False))
    return {"success": True, "message": "已重置", **payload}

@router.post("/feedback")
async def submit_feedback(
    feedback: UserFeedback,
    background_tasks: BackgroundTasks,
    service: PersonalizationService = Depends(get_personalization_service)
) -> JSONResponse:
    """提交用户反馈
    
    Args:
        feedback: 用户反馈
        background_tasks: 后台任务
        service: 个性化服务
        
    Returns:
        JSONResponse: 提交结果
    """
    try:
        # 处理反馈（异步）
        background_tasks.add_task(
            service.process_feedback,
            feedback
        )
        
        # 记录反馈
        background_tasks.add_task(
            log_user_feedback,
            feedback
        )
        
        return JSONResponse(
            content={"status": "success", "message": "反馈已接收"}
        )
        
    except Exception as e:
        logger.error(f"提交反馈失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/stream")
async def recommendation_stream(
    websocket: WebSocket,
    service: PersonalizationService = Depends(get_personalization_service)
):
    """WebSocket实时推荐流
    
    Args:
        websocket: WebSocket连接
        service: 个性化服务
    """
    await websocket.accept()
    user_id = None
    
    try:
        # 认证
        auth_message = await websocket.receive_text()
        auth_data = json.loads(auth_message)
        user_id = auth_data.get("user_id")
        
        if not user_id:
            await websocket.send_json({
                "type": "error",
                "message": "用户ID缺失"
            })
            await websocket.close()
            return
        
        # 发送欢迎消息
        await websocket.send_json({
            "type": "connected",
            "user_id": user_id,
            "timestamp": utc_now().isoformat()
        })
        
        # 实时推荐循环
        while True:
            try:
                # 接收请求
                message = await websocket.receive_text()
                data = json.loads(message)
                
                if data.get("type") == "request":
                    # 构建推荐请求
                    request = RecommendationRequest(
                        user_id=user_id,
                        context=data.get("context", {}),
                        n_recommendations=data.get("n_recommendations", 10),
                        scenario=data.get("scenario", "content")
                    )
                    
                    # 获取推荐
                    response = await service.get_recommendations(request)
                    
                    # 发送推荐
                    await websocket.send_json({
                        "type": "recommendations",
                        "data": response.model_dump(),
                        "timestamp": utc_now().isoformat()
                    })
                    
                elif data.get("type") == "feedback":
                    # 处理反馈
                    feedback = UserFeedback(
                        user_id=user_id,
                        item_id=data.get("item_id"),
                        feedback_type=data.get("feedback_type"),
                        feedback_value=data.get("feedback_value"),
                        context=data.get("context", {})
                    )
                    
                    await service.process_feedback(feedback)
                    
                    await websocket.send_json({
                        "type": "feedback_received",
                        "timestamp": utc_now().isoformat()
                    })
                    
                elif data.get("type") == "ping":
                    # 心跳
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": utc_now().isoformat()
                    })
                    
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "无效的JSON格式"
                })
            except Exception as e:
                logger.error(f"WebSocket处理错误: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket断开: user_id={user_id}")
    except Exception as e:
        logger.error(f"WebSocket错误: {e}", exc_info=True)
    finally:
        try:
            await websocket.close()
        except Exception:
            logger.exception("关闭WebSocket失败", exc_info=True)

@router.get("/features/realtime/{user_id}")
async def get_realtime_features(
    user_id: str,
    service: PersonalizationService = Depends(get_personalization_service)
) -> Dict[str, Any]:
    """获取实时特征
    
    Args:
        user_id: 用户ID
        service: 个性化服务
        
    Returns:
        Dict[str, Any]: 实时特征
    """
    try:
        features = await service.get_realtime_features(user_id)
        
        if not features:
            raise HTTPException(status_code=404, detail="特征不存在")
        
        return features.model_dump()
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取实时特征失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/features/compute")
async def compute_features_batch(
    user_ids: List[str],
    context: Optional[Dict[str, Any]] = None,
    service: PersonalizationService = Depends(get_personalization_service)
) -> Dict[str, Any]:
    """批量计算特征
    
    Args:
        user_ids: 用户ID列表
        context: 上下文
        service: 个性化服务
        
    Returns:
        Dict[str, Any]: 计算结果
    """
    try:
        results = {}
        
        # 并行计算
        tasks = [
            service.compute_features(user_id, context or {})
            for user_id in user_ids
        ]
        
        features_list = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for user_id, features in zip(user_ids, features_list):
            if isinstance(features, Exception):
                results[user_id] = {"error": str(features)}
            else:
                results[user_id] = features.model_dump() if features else None
        
        return results
        
    except Exception as e:
        logger.error(f"批量计算特征失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/status")
async def get_model_status(
    model_id: Optional[str] = None,
    service: PersonalizationService = Depends(get_personalization_service)
) -> Dict[str, Any]:
    """获取模型服务状态
    
    Args:
        model_id: 模型ID（可选）
        service: 个性化服务
        
    Returns:
        Dict[str, Any]: 模型状态
    """
    try:
        status = await service.get_model_status(model_id)
        return status
        
    except Exception as e:
        logger.error(f"获取模型状态失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/models/predict")
async def model_predict(
    model_id: str,
    features: List[float],
    service: PersonalizationService = Depends(get_personalization_service)
) -> Dict[str, Any]:
    """模型预测
    
    Args:
        model_id: 模型ID
        features: 特征向量
        service: 个性化服务
        
    Returns:
        Dict[str, Any]: 预测结果
    """
    try:
        import numpy as np
        
        # 转换为numpy数组
        feature_array = np.array(features)
        
        # 预测
        result = await service.predict(feature_array, model_id)
        
        return {
            "model_id": model_id,
            "prediction": result.tolist() if hasattr(result, 'tolist') else result,
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"模型预测失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/models/update")
async def update_model(
    model_config: ModelConfig,
    background_tasks: BackgroundTasks,
    service: PersonalizationService = Depends(get_personalization_service)
) -> JSONResponse:
    """增量更新模型
    
    Args:
        model_config: 模型配置
        background_tasks: 后台任务
        service: 个性化服务
        
    Returns:
        JSONResponse: 更新结果
    """
    try:
        # 异步更新模型
        background_tasks.add_task(
            service.update_model,
            model_config
        )
        
        return JSONResponse(
            content={
                "status": "accepted",
                "message": "模型更新任务已提交",
                "model_id": model_config.model_id
            }
        )
        
    except Exception as e:
        logger.error(f"提交模型更新失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/cache/stats")
async def get_cache_stats(
    service: PersonalizationService = Depends(get_personalization_service)
) -> Dict[str, Any]:
    """获取缓存统计信息
    
    Args:
        service: 个性化服务
        
    Returns:
        Dict[str, Any]: 缓存统计
    """
    try:
        stats = await service.get_cache_stats()
        return stats
        
    except Exception as e:
        logger.error(f"获取缓存统计失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cache/invalidate/{user_id}")
async def invalidate_user_cache(
    user_id: str,
    service: PersonalizationService = Depends(get_personalization_service)
) -> JSONResponse:
    """失效用户缓存
    
    Args:
        user_id: 用户ID
        service: 个性化服务
        
    Returns:
        JSONResponse: 失效结果
    """
    try:
        count = await service.invalidate_user_cache(user_id)
        
        return JSONResponse(
            content={
                "status": "success",
                "message": f"失效了 {count} 个缓存项"
            }
        )
        
    except Exception as e:
        logger.error(f"失效缓存失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_performance_metrics(
    service: PersonalizationService = Depends(get_personalization_service)
) -> Dict[str, Any]:
    """获取性能指标
    
    Args:
        service: 个性化服务
        
    Returns:
        Dict[str, Any]: 性能指标
    """
    try:
        metrics = await service.get_performance_metrics()
        return metrics
        
    except Exception as e:
        logger.error(f"获取性能指标失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# 辅助函数

async def log_recommendation_request(
    request: RecommendationRequest,
    response: RecommendationResponse
):
    """记录推荐请求
    
    Args:
        request: 推荐请求
        response: 推荐响应
        redis: Redis客户端
    """
    try:
        log_entry = {
            "request_id": response.request_id,
            "user_id": request.user_id,
            "scenario": request.scenario.value,
            "recommendation_count": len(response.recommendations),
            "latency_ms": response.latency_ms,
            "cache_hit": response.cache_hit,
            "timestamp": utc_now().isoformat()
        }
        
        redis = get_redis()
        # 保存到Redis（用于分析）
        log_key = f"recommendation_log:{response.request_id}"
        await redis.setex(
            log_key,
            86400,  # 保留24小时
            json.dumps(log_entry)
        )
        
        # 更新统计
        stats_key = f"recommendation_stats:{request.user_id}"
        await redis.hincrby(stats_key, "total_requests", 1)
        if response.cache_hit:
            await redis.hincrby(stats_key, "cache_hits", 1)
            
    except Exception as e:
        logger.error(f"记录推荐请求失败: {e}")

def _build_preferences_payload(user_id: str, features: Optional[Any]) -> Dict[str, Any]:
    feature_dict = features.model_dump() if hasattr(features, "model_dump") else (features or {})
    temporal = feature_dict.get("temporal") or {}
    behavioral = feature_dict.get("behavioral") or {}
    contextual = feature_dict.get("contextual") or {}
    aggregated = feature_dict.get("aggregated") or {}

    def avg(values: Dict[str, Any]) -> float:
        nums = [v for v in values.values() if isinstance(v, (int, float))]
        return sum(nums) / len(nums) if nums else 0.0

    hour = (temporal.get("hour_of_day") or 0.0) * 24
    slots = [0, 6, 12, 18, 24]

    def series(base: float) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for slot in slots:
            distance = abs(hour - slot)
            weight = 1 - min(1, distance / 24)
            result[f"{slot:02d}:00"] = max(0.0, base * weight)
        return result

    payload = {
        "user_id": user_id,
        "importance": {
            "user": series(avg(temporal)),
            "item": series(avg(aggregated)),
            "context": series(avg(contextual)),
            "interaction": series(avg(behavioral)),
        },
    }
    return payload

async def log_user_feedback(
    feedback: UserFeedback
):
    """记录用户反馈
    
    Args:
        feedback: 用户反馈
        redis: Redis客户端
    """
    try:
        log_entry = {
            "user_id": feedback.user_id,
            "item_id": feedback.item_id,
            "feedback_type": feedback.feedback_type,
            "feedback_value": str(feedback.feedback_value),
            "timestamp": feedback.timestamp.isoformat()
        }
        
        redis = get_redis()
        # 保存到Redis
        feedback_key = f"feedback_log:{feedback.user_id}:{feedback.timestamp.timestamp()}"
        await redis.setex(
            feedback_key,
            604800,  # 保留7天
            json.dumps(log_entry)
        )
        
        # 更新反馈统计
        stats_key = f"feedback_stats:{feedback.user_id}"
        await redis.hincrby(stats_key, f"total_{feedback.feedback_type}", 1)
        
    except Exception as e:
        logger.error(f"记录用户反馈失败: {e}")
