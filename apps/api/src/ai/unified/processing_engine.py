"""
统一处理引擎

提供流式和批处理的统一接口，支持智能模式切换和混合处理。
"""

from typing import List, Dict, Any, Optional, AsyncIterator, Union, Callable
from dataclasses import dataclass, field
import asyncio
from enum import Enum
import logging
import time
import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

from ..streaming import TokenStreamer, StreamEvent, StreamType
from ..batch import BatchProcessor, BatchJob, BatchTask, BatchStatus, TaskPriority

logger = logging.getLogger(__name__)


class ProcessingMode(str, Enum):
    """处理模式"""
    STREAM = "stream"          # 纯流式处理
    BATCH = "batch"            # 纯批处理
    HYBRID = "hybrid"          # 混合处理：流式输出+批量聚合
    AUTO = "auto"              # 自动选择模式
    PIPELINE = "pipeline"      # 流水线处理


@dataclass
class ProcessingItem:
    """处理项目"""
    id: str
    data: Any
    priority: int = TaskPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_llm_response(self) -> AsyncIterator[str]:
        """获取LLM响应（需要子类实现）"""
        # 这是一个示例实现，实际应用中需要集成真实的LLM
        async def mock_response():
            text = str(self.data)
            for token in text.split():
                yield f"{token} "
                await asyncio.sleep(0.01)
        return mock_response()


@dataclass
class ProcessingRequest:
    """处理请求"""
    session_id: str
    items: List[ProcessingItem]
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
    callback: Optional[Callable] = None
    
    @property
    def item_count(self) -> int:
        return len(self.items)


@dataclass
class ProcessingResponse:
    """处理响应"""
    request_id: str
    session_id: str
    mode_used: ProcessingMode
    
    # 结果数据
    results: List[Any] = field(default_factory=list)
    aggregated_result: Optional[Any] = None
    
    # 状态信息
    status: str = "pending"
    progress: float = 0.0
    
    # 性能指标
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    processing_time: Optional[float] = None
    
    # 错误信息
    errors: List[Dict[str, Any]] = field(default_factory=list)
    
    @property
    def success_rate(self) -> float:
        if not self.results and not self.errors:
            return 0.0
        total = len(self.results) + len(self.errors)
        return len(self.results) / total if total > 0 else 0.0


class UnifiedProcessingEngine:
    """统一处理引擎"""
    
    def __init__(
        self,
        token_streamer: Optional[TokenStreamer] = None,
        batch_processor: Optional[BatchProcessor] = None,
        default_mode: ProcessingMode = ProcessingMode.AUTO
    ):
        self.token_streamer = token_streamer or TokenStreamer()
        self.batch_processor = batch_processor
        self.default_mode = default_mode
        
        # 模式选择器
        from .mode_selector import ModeSelector
        self.mode_selector = ModeSelector()
        
        # 处理会话管理
        self.active_sessions: Dict[str, ProcessingResponse] = {}
        self.processing_history: List[ProcessingResponse] = []
        
        # 性能指标
        self._total_requests = 0
        self._total_items_processed = 0
        self._mode_usage_stats = {mode: 0 for mode in ProcessingMode}
        
    async def process(self, request: ProcessingRequest) -> ProcessingResponse:
        """统一处理接口"""
        request_id = str(uuid.uuid4())
        
        # 创建响应对象
        response = ProcessingResponse(
            request_id=request_id,
            session_id=request.session_id,
            mode_used=ProcessingMode.AUTO,  # 将被实际选择的模式覆盖
            start_time=utc_now()
        )
        
        # 存储活跃会话
        self.active_sessions[request.session_id] = response
        
        try:
            # 选择处理模式
            selected_mode = await self._select_processing_mode(request)
            response.mode_used = selected_mode
            response.status = "processing"
            
            # 更新统计
            self._total_requests += 1
            self._total_items_processed += request.item_count
            self._mode_usage_stats[selected_mode] += 1
            
            logger.info(f"开始处理: {request_id} (模式: {selected_mode.value}, 项目数: {request.item_count})")
            
            # 根据选择的模式执行处理
            if selected_mode == ProcessingMode.STREAM:
                await self._process_stream(request, response)
            elif selected_mode == ProcessingMode.BATCH:
                await self._process_batch(request, response)
            elif selected_mode == ProcessingMode.HYBRID:
                await self._process_hybrid(request, response)
            elif selected_mode == ProcessingMode.PIPELINE:
                await self._process_pipeline(request, response)
            else:
                # AUTO模式的回退处理
                await self._process_auto(request, response)
            
            # 完成处理
            response.status = "completed"
            response.end_time = utc_now()
            response.processing_time = (response.end_time - response.start_time).total_seconds()
            response.progress = 1.0
            
            logger.info(f"处理完成: {request_id} (耗时: {response.processing_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"处理失败: {request_id} - {e}")
            response.status = "failed"
            response.errors.append({
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": utc_now().isoformat()
            })
            response.end_time = utc_now()
            response.processing_time = (response.end_time - response.start_time).total_seconds()
        
        finally:
            # 移动到历史记录
            self.processing_history.append(response)
            if request.session_id in self.active_sessions:
                del self.active_sessions[request.session_id]
            
            # 限制历史记录大小
            if len(self.processing_history) > 1000:
                self.processing_history = self.processing_history[-500:]
        
        return response
    
    async def _select_processing_mode(self, request: ProcessingRequest) -> ProcessingMode:
        """选择处理模式"""
        if request.mode and request.mode != ProcessingMode.AUTO:
            return request.mode
        
        # 使用模式选择器进行智能选择
        return await self.mode_selector.select_mode(request)
    
    async def _process_stream(self, request: ProcessingRequest, response: ProcessingResponse):
        """流式处理"""
        logger.debug(f"执行流式处理: {len(request.items)} 个项目")
        
        for i, item in enumerate(request.items):
            try:
                # 更新进度
                response.progress = i / len(request.items)
                
                # 流式处理单个项目
                result_parts = []
                async for event in self.token_streamer.stream_tokens(
                    item.get_llm_response(),
                    session_id=request.session_id
                ):
                    if event.type == StreamType.TOKEN:
                        result_parts.append(event.data)
                    elif event.type == StreamType.COMPLETE:
                        response.results.append(event.data)
                        break
                    elif event.type == StreamType.ERROR:
                        response.errors.append({
                            "item_id": item.id,
                            "error": event.data,
                            "timestamp": utc_now().isoformat()
                        })
                        break
                
                # 执行回调
                if request.callback:
                    await self._execute_callback(request.callback, item, response.results[-1] if response.results else None)
                
            except Exception as e:
                logger.error(f"流式处理项目失败: {item.id} - {e}")
                response.errors.append({
                    "item_id": item.id,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": utc_now().isoformat()
                })
        
        # 如果需要聚合，执行最终聚合
        if request.requires_aggregation and response.results:
            response.aggregated_result = await self._aggregate_results(
                response.results, 
                request.aggregation_strategy
            )
    
    async def _process_batch(self, request: ProcessingRequest, response: ProcessingResponse):
        """批处理"""
        logger.debug(f"执行批处理: {len(request.items)} 个项目")
        
        if not self.batch_processor:
            raise RuntimeError("批处理器未配置")
        
        # 创建批处理任务
        batch_tasks = []
        for item in request.items:
            task = BatchTask(
                id=item.id,
                type="llm_processing",
                data=item.data,
                priority=item.priority
            )
            batch_tasks.append(task)
        
        # 创建批处理作业
        job = BatchJob(
            id=str(uuid.uuid4()),
            name=f"unified_batch_{request.session_id}",
            tasks=batch_tasks,
            max_parallel_tasks=request.max_parallel_tasks
        )
        
        # 提交作业
        job_id = await self.batch_processor.submit_job(job)
        
        # 等待完成并收集结果
        while True:
            job_status = await self.batch_processor.get_job_status(job_id)
            if not job_status:
                break
            
            # 更新进度
            response.progress = job_status.progress
            
            if job_status.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]:
                # 收集结果
                for task in job_status.tasks:
                    if task.status == BatchStatus.COMPLETED:
                        response.results.append(task.result)
                    elif task.status == BatchStatus.FAILED:
                        response.errors.append({
                            "item_id": task.id,
                            "error": task.error,
                            "error_details": task.error_details,
                            "timestamp": utc_now().isoformat()
                        })
                break
            
            await asyncio.sleep(1)  # 检查间隔
        
        # 聚合结果
        if request.requires_aggregation and response.results:
            response.aggregated_result = await self._aggregate_results(
                response.results,
                request.aggregation_strategy
            )
    
    async def _process_hybrid(self, request: ProcessingRequest, response: ProcessingResponse):
        """混合处理：流式输出+批量聚合"""
        logger.debug(f"执行混合处理: {len(request.items)} 个项目")
        
        # 并发处理所有项目，但保持流式输出
        tasks = []
        for item in request.items:
            task = asyncio.create_task(self._process_item_streaming(item, request.session_id))
            tasks.append(task)
        
        # 收集结果
        completed_results = []
        for i, task in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await task
                completed_results.append(result)
                response.results.append(result)
                
                # 更新进度
                response.progress = len(completed_results) / len(request.items)
                
                # 执行回调
                if request.callback:
                    await self._execute_callback(request.callback, None, result)
                
            except Exception as e:
                response.errors.append({
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "timestamp": utc_now().isoformat()
                })
        
        # 批量聚合所有结果
        if completed_results:
            response.aggregated_result = await self._aggregate_results(
                completed_results,
                request.aggregation_strategy
            )
    
    async def _process_pipeline(self, request: ProcessingRequest, response: ProcessingResponse):
        """流水线处理"""
        logger.debug(f"执行流水线处理: {len(request.items)} 个项目")
        
        # 创建流水线阶段
        pipeline_stages = [
            self._pipeline_stage_preprocess,
            self._pipeline_stage_process,
            self._pipeline_stage_postprocess
        ]
        
        # 按阶段处理所有项目
        current_data = [(item, item.data) for item in request.items]
        
        for stage_idx, stage_func in enumerate(pipeline_stages):
            next_data = []
            
            for i, (item, data) in enumerate(current_data):
                try:
                    processed_data = await stage_func(data, item)
                    next_data.append((item, processed_data))
                    
                    # 更新进度
                    total_progress = (stage_idx * len(current_data) + i + 1) / (len(pipeline_stages) * len(current_data))
                    response.progress = total_progress
                    
                except Exception as e:
                    response.errors.append({
                        "item_id": item.id,
                        "stage": f"stage_{stage_idx}",
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "timestamp": utc_now().isoformat()
                    })
            
            current_data = next_data
        
        # 收集最终结果
        response.results = [data for _, data in current_data]
        
        # 聚合结果
        if request.requires_aggregation and response.results:
            response.aggregated_result = await self._aggregate_results(
                response.results,
                request.aggregation_strategy
            )
    
    async def _process_auto(self, request: ProcessingRequest, response: ProcessingResponse):
        """自动模式处理（回退到流式处理）"""
        logger.debug("执行自动模式处理，回退到流式处理")
        await self._process_stream(request, response)
    
    async def _process_item_streaming(self, item: ProcessingItem, session_id: str) -> Any:
        """流式处理单个项目"""
        result_parts = []
        
        async for event in self.token_streamer.stream_tokens(
            item.get_llm_response(),
            session_id=session_id
        ):
            if event.type == StreamType.TOKEN:
                result_parts.append(event.data)
            elif event.type == StreamType.COMPLETE:
                return event.data
            elif event.type == StreamType.ERROR:
                raise RuntimeError(f"处理项目失败: {event.data}")
        
        return "".join(result_parts)
    
    async def _pipeline_stage_preprocess(self, data: Any, item: ProcessingItem) -> Any:
        """流水线预处理阶段"""
        # 示例预处理：数据清洗
        if isinstance(data, str):
            return data.strip().lower()
        return data
    
    async def _pipeline_stage_process(self, data: Any, item: ProcessingItem) -> Any:
        """流水线主处理阶段"""
        # 示例处理：模拟LLM处理
        async for event in self.token_streamer.stream_tokens(
            self._mock_llm_response(str(data)),
            session_id=item.id
        ):
            if event.type == StreamType.COMPLETE:
                return event.data
        return str(data)
    
    async def _pipeline_stage_postprocess(self, data: Any, item: ProcessingItem) -> Any:
        """流水线后处理阶段"""
        # 示例后处理：格式化结果
        if isinstance(data, str):
            return {
                "processed_text": data,
                "word_count": len(data.split()),
                "item_id": item.id,
                "processing_time": utc_now().isoformat()
            }
        return data
    
    async def _mock_llm_response(self, text: str) -> AsyncIterator[str]:
        """模拟LLM响应"""
        processed_text = f"处理结果: {text}"
        tokens = processed_text.split()
        
        for token in tokens:
            yield f"{token} "
            await asyncio.sleep(0.01)
    
    async def _aggregate_results(self, results: List[Any], strategy: str) -> Any:
        """聚合结果"""
        if strategy == "collect":
            return results
        elif strategy == "merge" and all(isinstance(r, dict) for r in results):
            merged = {}
            for result in results:
                merged.update(result)
            return merged
        elif strategy == "concat" and all(isinstance(r, str) for r in results):
            return "\n".join(results)
        elif strategy == "sum" and all(isinstance(r, (int, float)) for r in results):
            return sum(results)
        else:
            return results
    
    async def _execute_callback(self, callback: Callable, item: Optional[ProcessingItem], result: Any):
        """执行回调函数"""
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(item, result)
            else:
                callback(item, result)
        except Exception as e:
            logger.error(f"回调执行失败: {e}")
    
    async def get_session_status(self, session_id: str) -> Optional[ProcessingResponse]:
        """获取会话状态"""
        return self.active_sessions.get(session_id)
    
    async def get_processing_history(
        self, 
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ProcessingResponse]:
        """获取处理历史"""
        history = self.processing_history
        
        if session_id:
            history = [h for h in history if h.session_id == session_id]
        
        return history[-limit:] if limit else history
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """获取系统指标"""
        # 计算平均处理时间
        completed_responses = [r for r in self.processing_history if r.processing_time is not None]
        avg_processing_time = 0
        if completed_responses:
            avg_processing_time = sum(r.processing_time for r in completed_responses) / len(completed_responses)
        
        # 计算成功率
        success_rate = 0
        if completed_responses:
            successful = len([r for r in completed_responses if r.status == "completed"])
            success_rate = successful / len(completed_responses)
        
        return {
            "total_requests": self._total_requests,
            "total_items_processed": self._total_items_processed,
            "active_sessions": len(self.active_sessions),
            "processing_history_size": len(self.processing_history),
            "average_processing_time": avg_processing_time,
            "success_rate": success_rate,
            "mode_usage_stats": self._mode_usage_stats.copy(),
            "default_mode": self.default_mode.value
        }
    
    def set_default_mode(self, mode: ProcessingMode):
        """设置默认处理模式"""
        self.default_mode = mode
        logger.info(f"默认处理模式设置为: {mode.value}")
    
    async def clear_history(self, max_age_hours: int = 24):
        """清理历史记录"""
        cutoff_time = utc_now().timestamp() - (max_age_hours * 3600)
        
        self.processing_history = [
            h for h in self.processing_history
            if h.start_time and h.start_time.timestamp() >= cutoff_time
        ]
        
        logger.info(f"清理了历史记录，保留最近 {max_age_hours} 小时的记录")