"""
统一处理引擎API

提供流批一体化处理接口，支持智能模式切换和混合处理。
"""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import logging

from ...core.auth import get_current_user
from ...ai.unified import (
    UnifiedProcessingEngine, ProcessingMode, ProcessingRequest, ProcessingResponse,
    ProcessingItem, ModeSelector, SelectionStrategy
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/unified", tags=["unified"])

# 全局引擎实例
engine = None
mode_selector = None


def get_engine() -> UnifiedProcessingEngine:
    """获取引擎实例"""
    global engine
    if engine is None:
        engine = UnifiedProcessingEngine()
    return engine


def get_mode_selector() -> ModeSelector:
    """获取模式选择器实例"""
    global mode_selector
    if mode_selector is None:
        mode_selector = ModeSelector()
    return mode_selector


# API模型定义
class ProcessingItemRequest(BaseModel):
    """处理项目请求"""
    id: str
    data: Any
    priority: int = Field(default=5, ge=1, le=10)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class UnifiedProcessingRequest(BaseModel):
    """统一处理请求"""
    session_id: str
    items: List[ProcessingItemRequest]
    mode: Optional[ProcessingMode] = None
    
    # 流式处理配置
    requires_real_time: bool = False
    streaming_enabled: bool = True
    
    # 批处理配置
    batch_size: Optional[int] = None
    max_parallel_tasks: int = 10
    
    # 聚合配置
    requires_aggregation: bool = False
    aggregation_strategy: str = "collect"
    
    # 其他配置
    timeout: Optional[float] = None


class ProcessingStatusResponse(BaseModel):
    """处理状态响应"""
    request_id: str
    session_id: str
    mode_used: ProcessingMode
    status: str
    progress: float
    results: List[Any] = Field(default_factory=list)
    aggregated_result: Optional[Any] = None
    processing_time: Optional[float] = None
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    success_rate: float = 0.0


class SystemMetricsResponse(BaseModel):
    """系统指标响应"""
    total_requests: int
    total_items_processed: int
    active_sessions: int
    processing_history_size: int
    average_processing_time: float
    success_rate: float
    mode_usage_stats: Dict[str, int]
    default_mode: str


class ModeRecommendationResponse(BaseModel):
    """模式推荐响应"""
    mode: str
    score: float
    heuristic_score: float
    performance_score: float
    request_count: int
    success_rate: float
    avg_processing_time: float


class SelectionStatsResponse(BaseModel):
    """选择统计响应"""
    total_decisions: int
    recent_decisions: int
    mode_distribution: Dict[str, int]
    strategy_distribution: Dict[str, int]
    current_strategy: str
    average_system_load: float
    current_system_load: float
    performance_history: Dict[str, Dict[str, Any]]


@router.post("/process", response_model=ProcessingStatusResponse)
async def process_unified_request(
    request: UnifiedProcessingRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    统一处理请求
    
    支持流式、批处理、混合和流水线等多种处理模式。
    """
    try:
        engine = get_engine()
        
        # 转换请求格式
        processing_items = [
            ProcessingItem(
                id=item.id,
                data=item.data,
                priority=item.priority,
                metadata=item.metadata
            )
            for item in request.items
        ]
        
        internal_request = ProcessingRequest(
            session_id=request.session_id,
            items=processing_items,
            mode=request.mode,
            requires_real_time=request.requires_real_time,
            streaming_enabled=request.streaming_enabled,
            batch_size=request.batch_size,
            max_parallel_tasks=request.max_parallel_tasks,
            requires_aggregation=request.requires_aggregation,
            aggregation_strategy=request.aggregation_strategy,
            timeout=request.timeout
        )
        
        # 执行处理
        response = await engine.process(internal_request)
        
        # 后台更新模式选择器性能数据
        if response.processing_time is not None:
            background_tasks.add_task(
                get_mode_selector().update_mode_performance,
                response.mode_used,
                response.processing_time,
                response.status == "completed"
            )
        
        return ProcessingStatusResponse(
            request_id=response.request_id,
            session_id=response.session_id,
            mode_used=response.mode_used,
            status=response.status,
            progress=response.progress,
            results=response.results,
            aggregated_result=response.aggregated_result,
            processing_time=response.processing_time,
            errors=response.errors,
            success_rate=response.success_rate
        )
        
    except Exception as e:
        logger.error(f"统一处理请求失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@router.get("/status/{session_id}", response_model=Optional[ProcessingStatusResponse])
async def get_session_status(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """获取会话处理状态"""
    try:
        engine = get_engine()
        response = await engine.get_session_status(session_id)
        
        if not response:
            return None
        
        return ProcessingStatusResponse(
            request_id=response.request_id,
            session_id=response.session_id,
            mode_used=response.mode_used,
            status=response.status,
            progress=response.progress,
            results=response.results,
            aggregated_result=response.aggregated_result,
            processing_time=response.processing_time,
            errors=response.errors,
            success_rate=response.success_rate
        )
        
    except Exception as e:
        logger.error(f"获取会话状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取状态失败: {str(e)}")


@router.get("/history")
async def get_processing_history(
    session_id: Optional[str] = None,
    limit: int = 100,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """获取处理历史"""
    try:
        engine = get_engine()
        history = await engine.get_processing_history(session_id, limit)
        
        return [
            ProcessingStatusResponse(
                request_id=h.request_id,
                session_id=h.session_id,
                mode_used=h.mode_used,
                status=h.status,
                progress=h.progress,
                results=h.results,
                aggregated_result=h.aggregated_result,
                processing_time=h.processing_time,
                errors=h.errors,
                success_rate=h.success_rate
            )
            for h in history
        ]
        
    except Exception as e:
        logger.error(f"获取处理历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取历史失败: {str(e)}")


@router.get("/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """获取系统指标"""
    try:
        engine = get_engine()
        metrics = await engine.get_system_metrics()
        
        return SystemMetricsResponse(
            total_requests=metrics["total_requests"],
            total_items_processed=metrics["total_items_processed"],
            active_sessions=metrics["active_sessions"],
            processing_history_size=metrics["processing_history_size"],
            average_processing_time=metrics["average_processing_time"],
            success_rate=metrics["success_rate"],
            mode_usage_stats=metrics["mode_usage_stats"],
            default_mode=metrics["default_mode"]
        )
        
    except Exception as e:
        logger.error(f"获取系统指标失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取指标失败: {str(e)}")


@router.post("/mode/recommendations")
async def get_mode_recommendations(
    request: UnifiedProcessingRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> List[ModeRecommendationResponse]:
    """获取处理模式推荐"""
    try:
        selector = get_mode_selector()
        
        # 转换请求格式
        processing_items = [
            ProcessingItem(
                id=item.id,
                data=item.data,
                priority=item.priority,
                metadata=item.metadata
            )
            for item in request.items
        ]
        
        internal_request = ProcessingRequest(
            session_id=request.session_id,
            items=processing_items,
            mode=request.mode,
            requires_real_time=request.requires_real_time,
            streaming_enabled=request.streaming_enabled,
            batch_size=request.batch_size,
            max_parallel_tasks=request.max_parallel_tasks,
            requires_aggregation=request.requires_aggregation,
            aggregation_strategy=request.aggregation_strategy,
            timeout=request.timeout
        )
        
        recommendations = selector.get_mode_recommendations(internal_request)
        
        return [
            ModeRecommendationResponse(
                mode=rec["mode"],
                score=rec["score"],
                heuristic_score=rec["heuristic_score"],
                performance_score=rec["performance_score"],
                request_count=rec["request_count"],
                success_rate=rec["success_rate"],
                avg_processing_time=rec["avg_processing_time"]
            )
            for rec in recommendations
        ]
        
    except Exception as e:
        logger.error(f"获取模式推荐失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取推荐失败: {str(e)}")


@router.get("/selector/stats", response_model=SelectionStatsResponse)
async def get_selection_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """获取模式选择统计"""
    try:
        selector = get_mode_selector()
        stats = selector.get_selection_stats()
        
        return SelectionStatsResponse(
            total_decisions=stats["total_decisions"],
            recent_decisions=stats["recent_decisions"],
            mode_distribution=stats["mode_distribution"],
            strategy_distribution=stats["strategy_distribution"],
            current_strategy=stats["current_strategy"],
            average_system_load=stats["average_system_load"],
            current_system_load=stats["current_system_load"],
            performance_history=stats["performance_history"]
        )
        
    except Exception as e:
        logger.error(f"获取选择统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@router.post("/mode/set-default")
async def set_default_mode(
    mode: ProcessingMode,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """设置默认处理模式"""
    try:
        engine = get_engine()
        engine.set_default_mode(mode)
        
        return {"message": f"默认模式设置为: {mode.value}"}
        
    except Exception as e:
        logger.error(f"设置默认模式失败: {e}")
        raise HTTPException(status_code=500, detail=f"设置失败: {str(e)}")


@router.post("/selector/set-strategy")
async def set_selection_strategy(
    strategy: SelectionStrategy,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """设置模式选择策略"""
    try:
        selector = get_mode_selector()
        selector.set_strategy(strategy)
        
        return {"message": f"选择策略设置为: {strategy.value}"}
        
    except Exception as e:
        logger.error(f"设置选择策略失败: {e}")
        raise HTTPException(status_code=500, detail=f"设置失败: {str(e)}")


@router.post("/clear-history")
async def clear_processing_history(
    max_age_hours: int = 24,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """清理处理历史"""
    try:
        engine = get_engine()
        selector = get_mode_selector()
        
        await engine.clear_history(max_age_hours)
        selector.clear_history()
        
        return {"message": f"已清理超过 {max_age_hours} 小时的历史记录"}
        
    except Exception as e:
        logger.error(f"清理历史记录失败: {e}")
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")


@router.get("/modes")
async def get_available_modes():
    """获取可用的处理模式"""
    return {
        "modes": [
            {
                "value": mode.value,
                "name": mode.value.replace("_", " ").title(),
                "description": {
                    "stream": "实时流式处理，适合需要即时响应的场景",
                    "batch": "批量处理，适合大量数据的高效处理",
                    "hybrid": "混合处理，结合流式输出和批量聚合",
                    "pipeline": "流水线处理，适合多阶段数据处理",
                    "auto": "自动选择最优处理模式"
                }.get(mode.value, "")
            }
            for mode in ProcessingMode
        ]
    }


@router.get("/strategies")
async def get_available_strategies():
    """获取可用的选择策略"""
    return {
        "strategies": [
            {
                "value": strategy.value,
                "name": strategy.value.replace("_", " ").title(),
                "description": {
                    "heuristic": "基于启发式规则的模式选择",
                    "performance": "基于历史性能数据的选择",
                    "load_aware": "基于系统负载的智能选择",
                    "ml_predicted": "基于机器学习的预测选择（开发中）",
                    "hybrid": "结合多种策略的混合选择"
                }.get(strategy.value, "")
            }
            for strategy in SelectionStrategy
        ]
    }