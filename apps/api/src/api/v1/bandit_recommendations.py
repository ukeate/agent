"""
多臂老虎机推荐引擎API路由

提供推荐请求、反馈处理、算法管理和性能监控的REST API接口。
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Query
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
import asyncio
import uuid
from pydantic import Field
from src.ai.reinforcement_learning.recommendation_engine import AlgorithmType
from src.ai.reinforcement_learning.evaluation import InteractionEvent, EvaluationMetrics
from src.services.bandit_recommendation_service import bandit_recommendation_service
from src.api.base_model import ApiBaseModel

router = APIRouter(prefix="/bandit", tags=["bandit-recommendations"])

class RecommendationRequestModel(ApiBaseModel):
    """推荐请求模型"""
    user_id: str = Field(..., description="用户ID")
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")
    num_recommendations: int = Field(10, description="推荐数量", ge=1, le=50)
    exclude_items: Optional[List[str]] = Field(None, description="排除的物品ID列表")
    include_explanations: bool = Field(False, description="是否包含推荐解释")
    experiment_id: Optional[str] = Field(None, description="A/B测试实验ID")

class FeedbackRequestModel(ApiBaseModel):
    """反馈请求模型"""
    user_id: str = Field(..., description="用户ID")
    item_id: str = Field(..., description="物品ID")
    feedback_type: str = Field(..., description="反馈类型：view, click, like, purchase, rating")
    feedback_value: float = Field(0.0, description="反馈值", ge=0.0, le=5.0)
    context: Optional[Dict[str, Any]] = Field(None, description="上下文信息")

class AlgorithmConfigModel(ApiBaseModel):
    """算法配置模型"""
    algorithm_type: AlgorithmType = Field(..., description="算法类型")
    config: Dict[str, Any] = Field({}, description="算法参数配置")

class ExperimentRequestModel(ApiBaseModel):
    """A/B测试实验请求模型"""
    experiment_name: str = Field(..., description="实验名称")
    algorithms: Dict[str, AlgorithmConfigModel] = Field(..., description="参与测试的算法")
    traffic_split: Dict[str, float] = Field(..., description="流量分配比例")
    duration_hours: int = Field(24, description="实验持续时间（小时）", ge=1, le=168)
    min_sample_size: int = Field(100, description="最小样本量", ge=10)

async def get_recommendation_service():
    """获取推荐服务实例"""
    if not bandit_recommendation_service.is_initialized:
        success = await bandit_recommendation_service.initialize()
        if not success:
            raise HTTPException(status_code=503, detail="推荐引擎未初始化")
    return bandit_recommendation_service

@router.post("/initialize", summary="初始化推荐引擎")
async def initialize_engine(
    n_items: int = Query(..., description="物品总数", ge=1),
    algorithm_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    enable_cold_start: bool = Query(True, description="是否启用冷启动"),
    enable_evaluation: bool = Query(True, description="是否启用评估")
):
    """初始化推荐引擎"""
    try:
        success = await bandit_recommendation_service.initialize(
            n_items=n_items,
            algorithm_configs=algorithm_configs,
            enable_cold_start=enable_cold_start,
            enable_evaluation=enable_evaluation
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="初始化失败")
        
        return {
            "status": "success",
            "message": f"推荐引擎初始化成功，支持{n_items}个物品",
            "config": {
                "n_items": n_items,
                "cold_start_enabled": enable_cold_start,
                "evaluation_enabled": enable_evaluation,
                "algorithms": list(bandit_recommendation_service.algorithm_configs.keys())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"初始化失败: {str(e)}")

@router.post("/recommend", response_model=Dict[str, Any], summary="获取推荐")
async def get_recommendations(
    request: RecommendationRequestModel,
    service = Depends(get_recommendation_service)
):
    """获取个性化推荐"""
    try:
        response = await service.get_recommendations(
            user_id=request.user_id,
            num_recommendations=request.num_recommendations,
            context=request.context,
            exclude_items=request.exclude_items,
            include_explanations=request.include_explanations,
            experiment_id=request.experiment_id
        )
        
        # 转换timestamp为ISO格式
        if response.get("timestamp"):
            response["timestamp"] = response["timestamp"].isoformat()
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推荐生成失败: {str(e)}")

@router.post("/feedback", summary="提交用户反馈")
async def submit_feedback(
    feedback: FeedbackRequestModel,
    background_tasks: BackgroundTasks,
    service = Depends(get_recommendation_service)
):
    """提交用户反馈以更新推荐模型"""
    try:
        # 异步处理反馈
        async def process_feedback_task():
            await service.process_feedback(
                user_id=feedback.user_id,
                item_id=feedback.item_id,
                feedback_type=feedback.feedback_type,
                feedback_value=feedback.feedback_value,
                context=feedback.context
            )
        
        background_tasks.add_task(process_feedback_task)
        
        return {
            "status": "success",
            "message": "反馈已提交并将异步处理",
            "feedback_id": str(uuid.uuid4()),
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"反馈处理失败: {str(e)}")

@router.get("/statistics", summary="获取引擎统计信息")
async def get_engine_statistics(
    service = Depends(get_recommendation_service)
):
    """获取推荐引擎的统计信息和性能指标"""
    try:
        stats = service.get_statistics()
        return {
            "status": "success",
            "statistics": stats,
            "timestamp": utc_now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")

@router.put("/user/{user_id}/context", summary="更新用户上下文")
async def update_user_context(
    user_id: str,
    context: Dict[str, Any],
    service = Depends(get_recommendation_service)
):
    """更新用户的上下文信息"""
    try:
        success = await service.update_user_context(user_id, context)
        if not success:
            raise HTTPException(status_code=500, detail="更新用户上下文失败")
        return {
            "status": "success",
            "message": f"用户 {user_id} 的上下文已更新",
            "timestamp": utc_now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新用户上下文失败: {str(e)}")

@router.put("/item/{item_id}/features", summary="更新物品特征")
async def update_item_features(
    item_id: str,
    features: Dict[str, Any],
    service = Depends(get_recommendation_service)
):
    """更新物品的特征信息"""
    try:
        success = await service.update_item_features(item_id, features)
        if not success:
            raise HTTPException(status_code=500, detail="更新物品特征失败")
        return {
            "status": "success",
            "message": f"物品 {item_id} 的特征已更新",
            "timestamp": utc_now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"更新物品特征失败: {str(e)}")

@router.post("/experiments", summary="创建A/B测试实验")
async def create_experiment(
    experiment: ExperimentRequestModel,
    service = Depends(get_recommendation_service)
):
    """创建新的A/B测试实验"""
    try:
        if not service.engine or not service.engine.ab_test_manager:
            raise HTTPException(status_code=400, detail="A/B测试功能未启用")
        
        # 构建算法字典
        bandits = {}
        for variant_name, algo_config in experiment.algorithms.items():
            # 简化实现，直接使用引擎中已有的算法
            if algo_config.algorithm_type.value in service.engine.algorithms:
                bandits[variant_name] = service.engine.algorithms[algo_config.algorithm_type.value]
        
        start_time = utc_now()
        end_time = start_time + timedelta(hours=experiment.duration_hours)
        
        experiment_id = service.engine.ab_test_manager.create_experiment(
            experiment_name=experiment.experiment_name,
            bandits=bandits,
            traffic_split=experiment.traffic_split,
            start_time=start_time,
            end_time=end_time,
            min_sample_size=experiment.min_sample_size
        )
        
        return {
            "status": "success",
            "experiment_id": experiment_id,
            "experiment_name": experiment.experiment_name,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "variants": list(experiment.algorithms.keys()),
            "message": "A/B测试实验创建成功"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"创建实验失败: {str(e)}")

@router.get("/experiments", summary="列出活动的A/B测试实验")
async def list_experiments(
    service = Depends(get_recommendation_service)
):
    """列出所有活动的A/B测试实验"""
    try:
        if not service.engine or not service.engine.ab_test_manager:
            raise HTTPException(status_code=400, detail="A/B测试功能未启用")
        
        experiments = service.engine.ab_test_manager.list_active_experiments()
        return {
            "status": "success",
            "experiments": experiments,
            "count": len(experiments),
            "timestamp": utc_now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取实验列表失败: {str(e)}")

@router.get("/experiments/{experiment_id}/results", summary="获取A/B测试实验结果")
async def get_experiment_results(
    experiment_id: str,
    service = Depends(get_recommendation_service)
):
    """获取指定A/B测试实验的结果"""
    try:
        if not service.engine or not service.engine.ab_test_manager:
            raise HTTPException(status_code=400, detail="A/B测试功能未启用")
        
        results = service.engine.ab_test_manager.get_experiment_results(experiment_id)
        if not results:
            raise HTTPException(status_code=404, detail="实验不存在")
        
        return {
            "status": "success",
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取实验结果失败: {str(e)}")

@router.post("/experiments/{experiment_id}/end", summary="结束A/B测试实验")
async def end_experiment(
    experiment_id: str,
    service = Depends(get_recommendation_service)
):
    """结束指定的A/B测试实验"""
    try:
        if not service.engine or not service.engine.ab_test_manager:
            raise HTTPException(status_code=400, detail="A/B测试功能未启用")
        
        success = service.engine.ab_test_manager.end_experiment(experiment_id)
        if not success:
            raise HTTPException(status_code=404, detail="实验不存在")
        
        return {
            "status": "success",
            "message": f"实验 {experiment_id} 已结束",
            "timestamp": utc_now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"结束实验失败: {str(e)}")

@router.get("/health", summary="健康检查")
async def health_check():
    """API健康检查端点"""
    health_info = bandit_recommendation_service.get_health_status()
    
    status_code = 200 if health_info["status"] == "healthy" else 503
    return health_info

@router.get("/algorithms", summary="获取可用算法列表")
async def get_available_algorithms():
    """获取所有可用的推荐算法"""
    algorithms = []
    
    for algo_type in AlgorithmType:
        algorithms.append({
            "name": algo_type.value,
            "display_name": {
                "ucb": "Upper Confidence Bound",
                "thompson_sampling": "Thompson Sampling",
                "epsilon_greedy": "Epsilon Greedy",
                "linear_contextual": "Linear Contextual Bandit"
            }.get(algo_type.value, algo_type.value),
            "description": {
                "ucb": "基于置信区间上界的算法，平衡探索与利用",
                "thompson_sampling": "贝叶斯Thompson采样算法，通过后验分布采样",
                "epsilon_greedy": "ε-贪心算法，以固定概率进行随机探索",
                "linear_contextual": "线性上下文老虎机，利用特征信息进行个性化推荐"
            }.get(algo_type.value, "多臂老虎机算法"),
            "supports_context": algo_type == AlgorithmType.LINEAR_CONTEXTUAL,
            "supports_binary_feedback": algo_type == AlgorithmType.THOMPSON_SAMPLING
        })
    
    return {
        "status": "success",
        "algorithms": algorithms,
        "count": len(algorithms)
    }
