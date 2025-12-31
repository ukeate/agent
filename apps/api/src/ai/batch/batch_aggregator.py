"""
批处理结果聚合器

提供批处理结果的收集、聚合和后处理功能。
"""

from typing import List, Dict, Any, Optional, Callable, Union, AsyncIterator
from dataclasses import dataclass, field
import asyncio
from enum import Enum
import json
import statistics
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from collections import defaultdict
from .batch_processor import BatchTask, BatchJob, BatchStatus

from src.core.logging import get_logger
logger = get_logger(__name__)

class AggregationStrategy(str, Enum):
    """聚合策略"""
    COLLECT = "collect"              # 简单收集
    MERGE = "merge"                  # 合并结果
    REDUCE = "reduce"                # 减少计算
    STATISTICS = "statistics"        # 统计分析
    CUSTOM = "custom"                # 自定义处理

class AggregationMode(str, Enum):
    """聚合模式"""
    IMMEDIATE = "immediate"          # 立即聚合
    BATCH = "batch"                  # 批量聚合
    STREAMING = "streaming"          # 流式聚合
    ON_COMPLETE = "on_complete"      # 完成时聚合

@dataclass
class AggregationConfig:
    """聚合配置"""
    strategy: AggregationStrategy
    mode: AggregationMode
    batch_size: int = 100
    timeout: Optional[float] = None
    custom_handler: Optional[Callable] = None
    output_format: str = "json"
    include_metadata: bool = True

@dataclass
class AggregationResult:
    """聚合结果"""
    job_id: str
    aggregated_data: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    
    # 统计信息
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    processing_time: Optional[float] = None

class BatchAggregator:
    """批处理结果聚合器"""
    
    def __init__(self, default_config: Optional[AggregationConfig] = None):
        self.default_config = default_config or AggregationConfig(
            strategy=AggregationStrategy.COLLECT,
            mode=AggregationMode.ON_COMPLETE
        )
        
        # 聚合任务管理
        self.active_aggregations: Dict[str, Dict] = {}
        self.completed_aggregations: Dict[str, AggregationResult] = {}
        
        # 聚合器注册
        self.strategy_handlers: Dict[AggregationStrategy, Callable] = {
            AggregationStrategy.COLLECT: self._collect_results,
            AggregationStrategy.MERGE: self._merge_results,
            AggregationStrategy.REDUCE: self._reduce_results,
            AggregationStrategy.STATISTICS: self._compute_statistics,
        }
        
        # 流式聚合缓冲区
        self.streaming_buffers: Dict[str, List] = {}
        self.streaming_tasks: Dict[str, asyncio.Task] = {}
        
    async def start_aggregation(
        self, 
        job: BatchJob,
        config: Optional[AggregationConfig] = None
    ) -> str:
        """启动聚合任务"""
        config = config or self.default_config
        
        aggregation_info = {
            "job": job,
            "config": config,
            "results": [],
            "errors": [],
            "start_time": utc_now(),
            "last_update": utc_now()
        }
        
        self.active_aggregations[job.id] = aggregation_info
        
        # 根据模式启动相应的聚合过程
        if config.mode == AggregationMode.STREAMING:
            self.streaming_buffers[job.id] = []
            self.streaming_tasks[job.id] = asyncio.create_task(
                self._streaming_aggregation(job.id, config)
            )
        
        logger.info(f"启动聚合任务: {job.id} (策略: {config.strategy.value}, 模式: {config.mode.value})")
        return job.id
    
    async def add_task_result(self, job_id: str, task: BatchTask):
        """添加任务结果"""
        if job_id not in self.active_aggregations:
            logger.warning(f"聚合任务不存在: {job_id}")
            return
        
        aggregation_info = self.active_aggregations[job_id]
        config = aggregation_info["config"]
        
        # 添加结果或错误
        if task.status == BatchStatus.COMPLETED:
            aggregation_info["results"].append({
                "task_id": task.id,
                "result": task.result,
                "execution_time": task.execution_time,
                "completed_at": task.completed_at
            })
        elif task.status == BatchStatus.FAILED:
            aggregation_info["errors"].append({
                "task_id": task.id,
                "error": task.error,
                "error_details": task.error_details,
                "failed_at": task.completed_at
            })
        
        aggregation_info["last_update"] = utc_now()
        
        # 根据模式处理结果
        if config.mode == AggregationMode.IMMEDIATE:
            await self._process_immediate_aggregation(job_id, task)
        elif config.mode == AggregationMode.BATCH:
            await self._process_batch_aggregation(job_id)
        elif config.mode == AggregationMode.STREAMING:
            await self._add_to_streaming_buffer(job_id, task)
        
        logger.debug(f"添加任务结果到聚合: {job_id}/{task.id}")
    
    async def _process_immediate_aggregation(self, job_id: str, task: BatchTask):
        """处理立即聚合"""
        aggregation_info = self.active_aggregations[job_id]
        config = aggregation_info["config"]
        
        # 对单个结果进行处理
        if task.status == BatchStatus.COMPLETED and task.result is not None:
            handler = self.strategy_handlers.get(config.strategy, self._collect_results)
            
            # 处理单个结果
            partial_result = await handler([task.result], config)
            
            # 存储部分结果
            if "partial_results" not in aggregation_info:
                aggregation_info["partial_results"] = []
            
            aggregation_info["partial_results"].append({
                "task_id": task.id,
                "result": partial_result,
                "timestamp": utc_now()
            })
    
    async def _process_batch_aggregation(self, job_id: str):
        """处理批量聚合"""
        aggregation_info = self.active_aggregations[job_id]
        config = aggregation_info["config"]
        
        # 检查是否达到批量大小
        total_results = len(aggregation_info["results"])
        
        if total_results >= config.batch_size:
            # 执行批量聚合
            batch_results = aggregation_info["results"][-config.batch_size:]
            
            handler = self.strategy_handlers.get(config.strategy, self._collect_results)
            batch_aggregated = await handler([r["result"] for r in batch_results], config)
            
            # 存储批量结果
            if "batch_results" not in aggregation_info:
                aggregation_info["batch_results"] = []
            
            aggregation_info["batch_results"].append({
                "batch_index": len(aggregation_info["batch_results"]),
                "result": batch_aggregated,
                "task_count": len(batch_results),
                "timestamp": utc_now()
            })
            
            logger.info(f"完成批量聚合: {job_id} (批次大小: {config.batch_size})")
    
    async def _add_to_streaming_buffer(self, job_id: str, task: BatchTask):
        """添加到流式缓冲区"""
        if job_id in self.streaming_buffers and task.status == BatchStatus.COMPLETED:
            self.streaming_buffers[job_id].append({
                "task_id": task.id,
                "result": task.result,
                "timestamp": utc_now()
            })
    
    async def _streaming_aggregation(self, job_id: str, config: AggregationConfig):
        """流式聚合处理"""
        try:
            while job_id in self.active_aggregations:
                if job_id not in self.streaming_buffers:
                    await asyncio.sleep(1)
                    continue
                
                buffer = self.streaming_buffers[job_id]
                
                if len(buffer) >= config.batch_size:
                    # 处理缓冲区中的结果
                    batch_data = buffer[:config.batch_size]
                    del buffer[:config.batch_size]
                    
                    handler = self.strategy_handlers.get(config.strategy, self._collect_results)
                    result = await handler([item["result"] for item in batch_data], config)
                    
                    # 发送流式结果
                    await self._emit_streaming_result(job_id, result, len(batch_data))
                
                await asyncio.sleep(1)  # 1秒检查间隔
                
        except asyncio.CancelledError:
            logger.info(f"流式聚合任务取消: {job_id}")
        except Exception as e:
            logger.error(f"流式聚合出错: {job_id} - {e}")
    
    async def _emit_streaming_result(self, job_id: str, result: Any, batch_size: int):
        """发送流式聚合结果"""
        # 这里可以集成事件系统或回调机制
        logger.info(f"流式聚合结果: {job_id} (批次大小: {batch_size})")
        
        # 可以实现WebSocket推送或事件发布
        # await self.event_publisher.publish("streaming_aggregation", {
        #     "job_id": job_id,
        #     "result": result,
        #     "batch_size": batch_size
        # })
    
    async def complete_aggregation(self, job_id: str) -> Optional[AggregationResult]:
        """完成聚合"""
        if job_id not in self.active_aggregations:
            logger.warning(f"聚合任务不存在: {job_id}")
            return None
        
        aggregation_info = self.active_aggregations[job_id]
        config = aggregation_info["config"]
        job = aggregation_info["job"]
        
        try:
            # 收集所有成功的结果
            all_results = [r["result"] for r in aggregation_info["results"]]
            
            # 执行最终聚合
            handler = self.strategy_handlers.get(config.strategy, self._collect_results)
            
            if config.strategy == AggregationStrategy.CUSTOM and config.custom_handler:
                final_result = await config.custom_handler(all_results, aggregation_info)
            else:
                final_result = await handler(all_results, config)
            
            # 计算处理时间
            processing_time = None
            if "start_time" in aggregation_info:
                processing_time = (utc_now() - aggregation_info["start_time"]).total_seconds()
            
            # 创建聚合结果
            aggregation_result = AggregationResult(
                job_id=job_id,
                aggregated_data=final_result,
                total_tasks=len(job.tasks),
                successful_tasks=len(aggregation_info["results"]),
                failed_tasks=len(aggregation_info["errors"]),
                processing_time=processing_time
            )
            
            # 添加元数据
            if config.include_metadata:
                aggregation_result.metadata = {
                    "strategy": config.strategy.value,
                    "mode": config.mode.value,
                    "job_name": job.name,
                    "start_time": aggregation_info["start_time"].isoformat(),
                    "completion_time": utc_now().isoformat(),
                    "error_count": len(aggregation_info["errors"]),
                    "success_rate": aggregation_result.successful_tasks / aggregation_result.total_tasks
                }
                
                if aggregation_info["errors"]:
                    aggregation_result.metadata["errors"] = aggregation_info["errors"]
            
            # 存储完成的聚合
            self.completed_aggregations[job_id] = aggregation_result
            
            # 清理活跃聚合
            del self.active_aggregations[job_id]
            
            # 清理流式聚合资源
            if job_id in self.streaming_tasks:
                self.streaming_tasks[job_id].cancel()
                del self.streaming_tasks[job_id]
            
            if job_id in self.streaming_buffers:
                del self.streaming_buffers[job_id]
            
            logger.info(f"聚合完成: {job_id} (成功: {aggregation_result.successful_tasks}/"
                       f"{aggregation_result.total_tasks})")
            
            return aggregation_result
            
        except Exception as e:
            logger.error(f"聚合完成时出错: {job_id} - {e}")
            return None
    
    async def _collect_results(self, results: List[Any], config: AggregationConfig) -> List[Any]:
        """收集策略：简单收集所有结果"""
        return results
    
    async def _merge_results(self, results: List[Any], config: AggregationConfig) -> Any:
        """合并策略：合并所有结果"""
        if not results:
            return None
        
        # 尝试不同的合并方式
        if all(isinstance(r, dict) for r in results):
            # 合并字典
            merged = {}
            for result in results:
                if isinstance(result, dict):
                    merged.update(result)
            return merged
        
        elif all(isinstance(r, list) for r in results):
            # 合并列表
            merged = []
            for result in results:
                if isinstance(result, list):
                    merged.extend(result)
            return merged
        
        elif all(isinstance(r, str) for r in results):
            # 合并字符串
            return "\n".join(results)
        
        else:
            # 默认返回列表
            return results
    
    async def _reduce_results(self, results: List[Any], config: AggregationConfig) -> Any:
        """减少策略：对结果进行数学运算"""
        if not results:
            return None
        
        # 尝试数值运算
        try:
            if all(isinstance(r, (int, float)) for r in results):
                return {
                    "sum": sum(results),
                    "average": statistics.mean(results),
                    "min": min(results),
                    "max": max(results),
                    "count": len(results)
                }
            
            # 尝试对字典中的数值字段进行运算
            if all(isinstance(r, dict) for r in results):
                numeric_fields = {}
                
                # 找出所有数值字段
                for result in results:
                    for key, value in result.items():
                        if isinstance(value, (int, float)):
                            if key not in numeric_fields:
                                numeric_fields[key] = []
                            numeric_fields[key].append(value)
                
                # 对每个数值字段进行统计
                reduced = {}
                for field, values in numeric_fields.items():
                    if values:
                        reduced[f"{field}_sum"] = sum(values)
                        reduced[f"{field}_avg"] = statistics.mean(values)
                        reduced[f"{field}_min"] = min(values)
                        reduced[f"{field}_max"] = max(values)
                        reduced[f"{field}_count"] = len(values)
                
                return reduced
            
        except Exception as e:
            logger.warning(f"减少策略处理失败，回退到收集策略: {e}")
        
        # 回退到收集策略
        return results
    
    async def _compute_statistics(self, results: List[Any], config: AggregationConfig) -> Dict[str, Any]:
        """统计策略：计算结果的统计信息"""
        if not results:
            return {"count": 0}
        
        stats = {
            "count": len(results),
            "types": {},
            "sample": results[:5] if len(results) > 5 else results
        }
        
        # 统计类型分布
        for result in results:
            result_type = type(result).__name__
            stats["types"][result_type] = stats["types"].get(result_type, 0) + 1
        
        # 对数值结果进行统计
        numeric_results = [r for r in results if isinstance(r, (int, float))]
        if numeric_results:
            stats["numeric"] = {
                "count": len(numeric_results),
                "sum": sum(numeric_results),
                "mean": statistics.mean(numeric_results),
                "median": statistics.median(numeric_results),
                "min": min(numeric_results),
                "max": max(numeric_results)
            }
            
            if len(numeric_results) > 1:
                stats["numeric"]["stdev"] = statistics.stdev(numeric_results)
        
        # 对字符串结果进行统计
        string_results = [r for r in results if isinstance(r, str)]
        if string_results:
            stats["text"] = {
                "count": len(string_results),
                "total_length": sum(len(s) for s in string_results),
                "average_length": sum(len(s) for s in string_results) / len(string_results),
                "unique_count": len(set(string_results))
            }
        
        return stats
    
    async def get_aggregation_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """获取聚合进度"""
        if job_id in self.active_aggregations:
            info = self.active_aggregations[job_id]
            job = info["job"]
            
            return {
                "job_id": job_id,
                "status": "active",
                "strategy": info["config"].strategy.value,
                "mode": info["config"].mode.value,
                "results_collected": len(info["results"]),
                "errors_collected": len(info["errors"]),
                "total_tasks": len(job.tasks),
                "progress": len(info["results"]) / len(job.tasks) if job.tasks else 0,
                "start_time": info["start_time"].isoformat(),
                "last_update": info["last_update"].isoformat()
            }
        
        elif job_id in self.completed_aggregations:
            result = self.completed_aggregations[job_id]
            return {
                "job_id": job_id,
                "status": "completed",
                "total_tasks": result.total_tasks,
                "successful_tasks": result.successful_tasks,
                "failed_tasks": result.failed_tasks,
                "processing_time": result.processing_time,
                "completed_at": result.created_at.isoformat()
            }
        
        return None
    
    async def get_aggregation_result(self, job_id: str) -> Optional[AggregationResult]:
        """获取聚合结果"""
        return self.completed_aggregations.get(job_id)
    
    async def cancel_aggregation(self, job_id: str) -> bool:
        """取消聚合"""
        if job_id in self.active_aggregations:
            del self.active_aggregations[job_id]
            
            # 取消流式聚合任务
            if job_id in self.streaming_tasks:
                self.streaming_tasks[job_id].cancel()
                del self.streaming_tasks[job_id]
            
            if job_id in self.streaming_buffers:
                del self.streaming_buffers[job_id]
            
            logger.info(f"取消聚合: {job_id}")
            return True
        
        return False
    
    def register_custom_strategy(self, strategy: AggregationStrategy, handler: Callable):
        """注册自定义聚合策略"""
        self.strategy_handlers[strategy] = handler
        logger.info(f"注册自定义聚合策略: {strategy.value}")
    
    async def cleanup_completed_aggregations(self, max_age_hours: int = 24):
        """清理已完成的聚合结果"""
        cutoff_time = utc_now().timestamp() - (max_age_hours * 3600)
        
        to_remove = []
        for job_id, result in self.completed_aggregations.items():
            if result.created_at.timestamp() < cutoff_time:
                to_remove.append(job_id)
        
        for job_id in to_remove:
            del self.completed_aggregations[job_id]
        
        if to_remove:
            logger.info(f"清理了 {len(to_remove)} 个已完成的聚合结果")
