"""
任务调度器

提供智能任务调度、负载均衡和资源管理功能。
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import asyncio
from enum import Enum
import time
import heapq
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
import statistics
import psutil
import gc
from collections import deque
import threading
from .batch_types import BatchTask, BatchJob, BatchStatus, TaskPriority

from src.core.logging import get_logger
logger = get_logger(__name__)

class SchedulingStrategy(str, Enum):
    """调度策略"""
    FIFO = "fifo"              # 先进先出
    PRIORITY = "priority"       # 基于优先级
    SJF = "sjf"                # 最短作业优先
    FAIR_SHARE = "fair_share"   # 公平共享
    LOAD_BALANCED = "load_balanced"  # 负载均衡
    ADAPTIVE = "adaptive"       # 自适应调度

@dataclass
class SystemResources:
    """系统资源指标"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_io: float = 0.0
    network_io: float = 0.0
    available_memory: int = 0
    cpu_cores: int = 0
    
    def update(self):
        """更新系统资源指标"""
        self.cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        self.memory_usage = memory.percent
        self.available_memory = memory.available
        self.cpu_cores = psutil.cpu_count()
        
        # 磁盘和网络IO（可选）
        try:
            disk_io = psutil.disk_io_counters()
            self.disk_io = disk_io.read_bytes + disk_io.write_bytes if disk_io else 0
            net_io = psutil.net_io_counters()
            self.network_io = net_io.bytes_sent + net_io.bytes_recv if net_io else 0
        except Exception:
            logger.exception("获取磁盘/网络IO失败", exc_info=True)
            self.disk_io = 0
            self.network_io = 0

@dataclass
class SLARequirement:
    """SLA要求"""
    max_completion_time: float  # 最大完成时间（秒）
    max_failure_rate: float = 0.05  # 最大失败率 5%
    min_throughput: float = 0.0  # 最小吞吐量（任务/秒）
    priority_multiplier: float = 1.0  # 优先级倍数

@dataclass
class WorkerMetrics:
    """工作者指标"""
    worker_id: str
    current_load: int = 0
    total_tasks_processed: int = 0
    total_processing_time: float = 0.0
    average_task_time: float = 0.0
    success_rate: float = 1.0
    last_task_completed: Optional[datetime] = None
    
    # 性能评分
    performance_score: float = 1.0
    availability_score: float = 1.0
    
    # 资源感知指标
    preferred_task_types: List[str] = field(default_factory=list)
    task_type_performance: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Dict[str, float] = field(default_factory=dict)  # cpu, memory等
    specialization_score: float = 1.0
    
    def update_metrics(self, task_time: float, success: bool, task_type: str = None):
        """更新工作者指标"""
        self.total_tasks_processed += 1
        self.total_processing_time += task_time
        self.average_task_time = self.total_processing_time / self.total_tasks_processed
        self.last_task_completed = utc_now()
        
        # 更新成功率（使用滑动平均）
        alpha = 0.1
        self.success_rate = alpha * (1.0 if success else 0.0) + (1 - alpha) * self.success_rate
        
        # 更新任务类型性能指标
        if task_type:
            if task_type not in self.task_type_performance:
                self.task_type_performance[task_type] = task_time
            else:
                # 滑动平均更新任务类型性能
                alpha_type = 0.2
                self.task_type_performance[task_type] = (
                    alpha_type * task_time + (1 - alpha_type) * self.task_type_performance[task_type]
                )
            
            # 更新偏好任务类型
            if task_type not in self.preferred_task_types and success:
                self.preferred_task_types.append(task_type)
        
        # 计算性能评分
        self._calculate_performance_score()
    
    def _calculate_performance_score(self):
        """计算性能评分"""
        # 基于平均任务时间的倒数和成功率
        if self.average_task_time > 0:
            speed_score = min(1.0 / self.average_task_time, 1.0)
        else:
            speed_score = 1.0
            
        self.performance_score = 0.7 * speed_score + 0.3 * self.success_rate
    
    def get_estimated_completion_time(self, task_complexity: float = 1.0) -> float:
        """估算任务完成时间"""
        if self.average_task_time <= 0:
            return 60.0  # 默认1分钟
        
        # 考虑当前负载和任务复杂度
        load_factor = 1.0 + (self.current_load * 0.2)
        return self.average_task_time * task_complexity * load_factor

@dataclass
class SchedulingContext:
    """调度上下文"""
    available_workers: List[str]
    worker_metrics: Dict[str, WorkerMetrics]
    system_load: float
    queue_length: int
    current_time: datetime = field(default_factory=utc_now)

class PredictiveModel:
    """预测性调度模型"""
    
    def __init__(self):
        self.historical_patterns = deque(maxlen=200)
        self.resource_trends = deque(maxlen=100)
        self.performance_baselines = {}
        
    def predict_completion_time(self, task: BatchTask, worker_metrics: WorkerMetrics, 
                               system_resources: SystemResources) -> float:
        """预测任务完成时间"""
        # 基础预测：使用工作者历史平均时间
        base_time = worker_metrics.average_task_time if worker_metrics.average_task_time > 0 else 60.0
        
        # 任务类型调整
        if task.type in worker_metrics.task_type_performance:
            type_factor = worker_metrics.task_type_performance[task.type] / base_time
        else:
            type_factor = 1.2  # 未知任务类型的惩罚
        
        # 系统资源调整
        resource_factor = 1.0
        if system_resources.cpu_usage > 80:
            resource_factor *= 1.3
        if system_resources.memory_usage > 85:
            resource_factor *= 1.2
        
        # 当前负载调整
        load_factor = 1.0 + (worker_metrics.current_load * 0.15)
        
        predicted_time = base_time * type_factor * resource_factor * load_factor
        return max(predicted_time, 10.0)  # 最少10秒
    
    def predict_resource_demand(self, task: BatchTask) -> Dict[str, float]:
        """预测任务资源需求"""
        # 简化的资源需求预测
        base_cpu = 10.0  # 基础CPU使用百分比
        base_memory = 100 * 1024 * 1024  # 基础内存使用 100MB
        
        # 根据任务类型调整
        type_multipliers = {
            "ml_training": {"cpu": 3.0, "memory": 5.0},
            "data_processing": {"cpu": 2.0, "memory": 3.0},
            "api_call": {"cpu": 0.5, "memory": 0.5},
            "file_processing": {"cpu": 1.5, "memory": 2.0},
            "default": {"cpu": 1.0, "memory": 1.0}
        }
        
        multiplier = type_multipliers.get(task.type, type_multipliers["default"])
        
        return {
            "cpu": base_cpu * multiplier["cpu"],
            "memory": base_memory * multiplier["memory"]
        }
    
    def should_scale_up(self, queue_length: int, avg_completion_time: float, 
                       system_resources: SystemResources) -> bool:
        """判断是否需要扩容"""
        # 队列积压严重
        if queue_length > 100:
            return True
        
        # 平均完成时间过长
        if avg_completion_time > 300:  # 5分钟
            return True
        
        # 资源利用率不高但队列积压
        if (system_resources.cpu_usage < 60 and 
            system_resources.memory_usage < 70 and 
            queue_length > 20):
            return True
        
        return False
    
    def get_optimal_worker_count(self, pending_tasks: int, avg_task_time: float,
                                target_completion_time: float) -> int:
        """计算最优工作者数量"""
        if avg_task_time <= 0:
            return min(10, pending_tasks)
        
        # 并行处理时间 = 总任务时间 / 工作者数量
        # 目标：并行处理时间 <= 目标完成时间
        total_task_time = pending_tasks * avg_task_time
        optimal_workers = int(total_task_time / target_completion_time) + 1
        
        # 限制最大工作者数量
        return min(optimal_workers, 50, pending_tasks)

class TaskScheduler:
    """任务调度器"""
    
    def __init__(
        self,
        strategy: SchedulingStrategy = SchedulingStrategy.ADAPTIVE,
        max_queue_size: int = 10000,
        load_threshold: float = 0.8
    ):
        self.strategy = strategy
        self.max_queue_size = max_queue_size
        self.load_threshold = load_threshold
        
        # 任务队列 (priority, timestamp, task, job_id)
        self._task_queue: List[Tuple[int, float, BatchTask, str]] = []
        self._queue_lock = asyncio.Lock()
        
        # 线程安全的统计数据缓存（用于监控线程读取）
        self._thread_safe_stats = {
            'queue_length': 0,
            'total_scheduled': 0,
            'system_load': 0.0
        }
        self._stats_lock = threading.Lock()
        
        # 工作者管理
        self.worker_metrics: Dict[str, WorkerMetrics] = {}
        self.worker_assignments: Dict[str, List[str]] = {}  # worker_id -> [task_ids]
        
        # 调度统计
        self._total_scheduled = 0
        self._scheduling_history: List[Dict] = []
        self._performance_history: List[float] = []
        
        # 自适应调度参数
        self.adaptive_weights = {
            "priority": 0.3,
            "load_balance": 0.3, 
            "performance": 0.2,
            "fairness": 0.2
        }
        
        # 新增：资源感知和SLA保证
        self.system_resources = SystemResources()
        self.sla_requirements: Dict[str, SLARequirement] = {}  # task_type -> SLA
        self.task_predictions: Dict[str, float] = {}  # task_id -> predicted_completion_time
        
        # 预测性调度
        self._throughput_history = deque(maxlen=100)  # 吞吐量历史
        self._resource_usage_history = deque(maxlen=50)  # 资源使用历史
        self._prediction_model = PredictiveModel()
        
        # 后台监控线程
        self._monitor_thread = None
        self._monitoring_active = False
    
    def register_worker(self, worker_id: str):
        """注册工作者"""
        if worker_id not in self.worker_metrics:
            self.worker_metrics[worker_id] = WorkerMetrics(worker_id=worker_id)
            self.worker_assignments[worker_id] = []
            logger.info(f"注册工作者: {worker_id}")
    
    def unregister_worker(self, worker_id: str):
        """注销工作者"""
        if worker_id in self.worker_metrics:
            del self.worker_metrics[worker_id]
            del self.worker_assignments[worker_id]
            logger.info(f"注销工作者: {worker_id}")
    
    def register_sla_requirement(self, task_type: str, sla: SLARequirement):
        """注册SLA要求"""
        self.sla_requirements[task_type] = sla
        logger.info(f"注册SLA要求: {task_type} - 最大完成时间: {sla.max_completion_time}s")
    
    def start_monitoring(self):
        """启动系统资源监控"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_system_resources)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        logger.info("启动系统资源监控")
    
    def stop_monitoring(self):
        """停止系统资源监控"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2)
        logger.info("停止系统资源监控")
    
    def _monitor_system_resources(self):
        """监控系统资源"""
        while self._monitoring_active:
            try:
                self.system_resources.update()
                
                # 获取线程安全的统计数据
                with self._stats_lock:
                    queue_length = self._thread_safe_stats['queue_length']
                    total_scheduled = self._thread_safe_stats['total_scheduled']
                
                # 记录资源使用历史
                resource_snapshot = {
                    'timestamp': time.time(),
                    'cpu': self.system_resources.cpu_usage,
                    'memory': self.system_resources.memory_usage,
                    'queue_length': queue_length
                }
                self._resource_usage_history.append(resource_snapshot)
                
                # 记录吞吐量
                current_time = time.time()
                if hasattr(self, '_last_throughput_check'):
                    time_diff = current_time - self._last_throughput_check
                    if time_diff >= 60:  # 每分钟记录一次
                        tasks_completed = total_scheduled - queue_length
                        throughput = tasks_completed / (time_diff / 60)  # 任务/分钟
                        self._throughput_history.append(throughput)
                        self._last_throughput_check = current_time
                else:
                    self._last_throughput_check = current_time
                
                time.sleep(30)  # 每30秒更新一次
            except Exception as e:
                logger.error(f"资源监控出错: {e}")
                time.sleep(60)
    
    async def schedule_task(self, task: BatchTask, job_id: str) -> Optional[str]:
        """调度单个任务"""
        async with self._queue_lock:
            # 检查队列大小
            if len(self._task_queue) >= self.max_queue_size:
                logger.warning("任务队列已满，拒绝新任务")
                return None
            
            # 计算任务优先级
            effective_priority = self._calculate_effective_priority(task, job_id)
            
            # 加入队列
            heapq.heappush(
                self._task_queue,
                (-effective_priority, time.time(), task, job_id)  # 负号实现最大堆
            )
            
            self._total_scheduled += 1
            
            # 更新线程安全统计缓存
            with self._stats_lock:
                self._thread_safe_stats['queue_length'] = len(self._task_queue)
                self._thread_safe_stats['total_scheduled'] = self._total_scheduled
            
            logger.debug(f"任务加入调度队列: {task.id} (优先级: {effective_priority})")
            
            # 尝试立即分配
            return await self._try_assign_task()
    
    async def get_next_task(self, worker_id: str) -> Optional[Tuple[BatchTask, str]]:
        """获取下一个任务"""
        async with self._queue_lock:
            if not self._task_queue:
                return None
            
            # 根据调度策略选择任务
            if self.strategy == SchedulingStrategy.FIFO:
                return self._get_fifo_task(worker_id)
            elif self.strategy == SchedulingStrategy.PRIORITY:
                return self._get_priority_task(worker_id)
            elif self.strategy == SchedulingStrategy.SJF:
                return self._get_sjf_task(worker_id)
            elif self.strategy == SchedulingStrategy.LOAD_BALANCED:
                return self._get_load_balanced_task(worker_id)
            elif self.strategy == SchedulingStrategy.FAIR_SHARE:
                return self._get_fair_share_task(worker_id)
            elif self.strategy == SchedulingStrategy.ADAPTIVE:
                return self._get_adaptive_task(worker_id)
            else:
                return self._get_priority_task(worker_id)
    
    def _calculate_effective_priority(self, task: BatchTask, job_id: str) -> int:
        """计算任务的有效优先级"""
        base_priority = task.priority
        
        # 考虑任务年龄（防止饥饿）
        age_factor = min((time.time() - task.created_at.timestamp()) / 3600, 5)  # 最多5小时
        
        # 考虑重试次数
        retry_penalty = task.retry_count * 2
        
        # 考虑依赖关系
        dependency_boost = 0
        if task.dependencies:
            dependency_boost = len(task.dependencies) * 1
        
        # 新增：SLA要求调整
        sla_boost = 0
        if task.type in self.sla_requirements:
            sla = self.sla_requirements[task.type]
            # 预测完成时间并根据SLA调整优先级
            predicted_time = self._predict_task_completion_time(task)
            if predicted_time > sla.max_completion_time:
                sla_boost = sla.priority_multiplier * 2  # SLA风险提升优先级
        
        # 系统负载调整
        load_adjustment = 0
        if self.system_resources.cpu_usage > 90 or self.system_resources.memory_usage > 90:
            # 高负载时，优先处理轻量级任务
            predicted_resources = self._prediction_model.predict_resource_demand(task)
            if predicted_resources["cpu"] < 15:  # 轻量级CPU任务
                load_adjustment = 1
        
        effective_priority = base_priority + age_factor + dependency_boost + sla_boost + load_adjustment - retry_penalty
        return max(1, int(effective_priority))
    
    def _predict_task_completion_time(self, task: BatchTask) -> float:
        """预测任务完成时间"""
        # 使用预测模型
        best_worker = self._find_best_worker_for_task(task)
        if best_worker:
            worker_metrics = self.worker_metrics[best_worker]
            return self._prediction_model.predict_completion_time(task, worker_metrics, self.system_resources)
        else:
            # 没有合适工作者，返回默认预测
            return 120.0  # 2分钟默认
    
    def _find_best_worker_for_task(self, task: BatchTask) -> Optional[str]:
        """找到最适合执行任务的工作者"""
        best_worker = None
        best_score = -1
        
        for worker_id, metrics in self.worker_metrics.items():
            if metrics.current_load >= self.load_threshold:
                continue
            
            # 计算匹配分数
            score = self._calculate_worker_task_compatibility(metrics, task)
            
            if score > best_score:
                best_score = score
                best_worker = worker_id
        
        return best_worker
    
    def _calculate_worker_task_compatibility(self, worker_metrics: WorkerMetrics, task: BatchTask) -> float:
        """计算工作者与任务的兼容性分数"""
        score = 0.0
        
        # 任务类型匹配度
        if task.type in worker_metrics.task_type_performance:
            # 擅长此类任务
            performance_ratio = worker_metrics.average_task_time / worker_metrics.task_type_performance[task.type]
            score += min(performance_ratio, 2.0) * 0.4
        elif task.type in worker_metrics.preferred_task_types:
            score += 0.3
        else:
            score += 0.1  # 未知任务类型惩罚
        
        # 负载情况
        load_score = max(0, 1.0 - worker_metrics.current_load / self.load_threshold)
        score += load_score * 0.3
        
        # 成功率
        score += worker_metrics.success_rate * 0.2
        
        # 性能分数
        score += worker_metrics.performance_score * 0.1
        
        return score
    
    def _get_fifo_task(self, worker_id: str) -> Optional[Tuple[BatchTask, str]]:
        """FIFO调度"""
        if self._task_queue:
            _, _, task, job_id = heapq.heappop(self._task_queue)
            # 更新线程安全统计缓存
            with self._stats_lock:
                self._thread_safe_stats['queue_length'] = len(self._task_queue)
            return task, job_id
        return None
    
    def _get_priority_task(self, worker_id: str) -> Optional[Tuple[BatchTask, str]]:
        """基于优先级的调度"""
        if self._task_queue:
            _, _, task, job_id = heapq.heappop(self._task_queue)
            # 更新线程安全统计缓存
            with self._stats_lock:
                self._thread_safe_stats['queue_length'] = len(self._task_queue)
            return task, job_id
        return None
    
    def _get_sjf_task(self, worker_id: str) -> Optional[Tuple[BatchTask, str]]:
        """最短作业优先调度"""
        if not self._task_queue:
            return None
        
        # 找到估算时间最短的任务
        min_time = float('inf')
        best_index = 0
        worker_metrics = self.worker_metrics.get(worker_id)
        
        for i, (_, _, task, _) in enumerate(self._task_queue):
            # 估算任务执行时间
            estimated_time = self._estimate_task_time(task, worker_metrics)
            if estimated_time < min_time:
                min_time = estimated_time
                best_index = i
        
        # 移除并返回最短任务
        _, _, task, job_id = self._task_queue.pop(best_index)
        heapq.heapify(self._task_queue)  # 重新堆化
        # 更新线程安全统计缓存
        with self._stats_lock:
            self._thread_safe_stats['queue_length'] = len(self._task_queue)
        return task, job_id
    
    def _get_load_balanced_task(self, worker_id: str) -> Optional[Tuple[BatchTask, str]]:
        """负载均衡调度"""
        if not self._task_queue:
            return None
        
        worker_metrics = self.worker_metrics.get(worker_id)
        if not worker_metrics:
            # 新工作者，直接分配优先级最高的任务
            _, _, task, job_id = heapq.heappop(self._task_queue)
            # 更新线程安全统计缓存
            with self._stats_lock:
                self._thread_safe_stats['queue_length'] = len(self._task_queue)
            return task, job_id
        
        # 如果工作者负载过高，跳过复杂任务
        if worker_metrics.current_load > self.load_threshold:
            # 寻找简单任务
            for i, (_, _, task, job_id) in enumerate(self._task_queue):
                estimated_time = self._estimate_task_time(task, worker_metrics)
                if estimated_time < worker_metrics.average_task_time:
                    removed_task = self._task_queue.pop(i)
                    heapq.heapify(self._task_queue)
                    # 更新线程安全统计缓存
                    with self._stats_lock:
                        self._thread_safe_stats['queue_length'] = len(self._task_queue)
                    return removed_task[2], removed_task[3]
        
        # 默认返回优先级最高的任务
        _, _, task, job_id = heapq.heappop(self._task_queue)
        # 更新线程安全统计缓存
        with self._stats_lock:
            self._thread_safe_stats['queue_length'] = len(self._task_queue)
        return task, job_id
    
    def _get_fair_share_task(self, worker_id: str) -> Optional[Tuple[BatchTask, str]]:
        """公平共享调度"""
        if not self._task_queue:
            return None
        
        worker_metrics = self.worker_metrics.get(worker_id)
        if not worker_metrics:
            _, _, task, job_id = heapq.heappop(self._task_queue)
            # 更新线程安全统计缓存
            with self._stats_lock:
                self._thread_safe_stats['queue_length'] = len(self._task_queue)
            return task, job_id
        
        # 计算工作者的相对负载
        all_loads = [m.current_load for m in self.worker_metrics.values()]
        avg_load = statistics.mean(all_loads) if all_loads else 0
        
        # 如果当前工作者负载低于平均值，优先分配
        if worker_metrics.current_load <= avg_load:
            _, _, task, job_id = heapq.heappop(self._task_queue)
            # 更新线程安全统计缓存
            with self._stats_lock:
                self._thread_safe_stats['queue_length'] = len(self._task_queue)
            return task, job_id
        
        # 否则查找适合的轻量任务
        for i, (_, _, task, job_id) in enumerate(self._task_queue):
            estimated_time = self._estimate_task_time(task, worker_metrics)
            if estimated_time < worker_metrics.average_task_time * 0.8:
                removed_task = self._task_queue.pop(i)
                heapq.heapify(self._task_queue)
                # 更新线程安全统计缓存
                with self._stats_lock:
                    self._thread_safe_stats['queue_length'] = len(self._task_queue)
                return removed_task[2], removed_task[3]
        
        return None  # 暂不分配
    
    def _get_adaptive_task(self, worker_id: str) -> Optional[Tuple[BatchTask, str]]:
        """自适应调度"""
        if not self._task_queue:
            return None
        
        # 根据系统状态调整权重
        self._adapt_scheduling_weights()
        
        # 为每个任务计算综合评分
        best_score = -1
        best_index = 0
        worker_metrics = self.worker_metrics.get(worker_id)
        
        for i, (priority, timestamp, task, job_id) in enumerate(self._task_queue):
            score = self._calculate_adaptive_score(
                task, job_id, worker_id, worker_metrics, priority, timestamp
            )
            
            if score > best_score:
                best_score = score
                best_index = i
        
        # 返回评分最高的任务
        removed_task = self._task_queue.pop(best_index)
        heapq.heapify(self._task_queue)
        # 更新线程安全统计缓存
        with self._stats_lock:
            self._thread_safe_stats['queue_length'] = len(self._task_queue)
        return removed_task[2], removed_task[3]
    
    def _calculate_adaptive_score(
        self, 
        task: BatchTask, 
        job_id: str, 
        worker_id: str,
        worker_metrics: Optional[WorkerMetrics],
        priority: int,
        timestamp: float
    ) -> float:
        """计算自适应调度评分"""
        # 优先级分数
        priority_score = priority / 10.0
        
        # 负载均衡分数
        if worker_metrics:
            load_score = max(0, (1.0 - worker_metrics.current_load))
        else:
            load_score = 1.0
        
        # 性能匹配分数
        performance_score = 1.0
        if worker_metrics:
            estimated_time = self._estimate_task_time(task, worker_metrics)
            if worker_metrics.average_task_time > 0:
                # 偏好分配给擅长类似任务的工作者
                performance_score = min(1.0, worker_metrics.average_task_time / estimated_time)
        
        # 公平性分数（考虑等待时间）
        wait_time = time.time() - timestamp
        fairness_score = min(1.0, wait_time / 3600)  # 1小时内线性增长
        
        # 综合评分
        score = (
            self.adaptive_weights["priority"] * priority_score +
            self.adaptive_weights["load_balance"] * load_score +
            self.adaptive_weights["performance"] * performance_score +
            self.adaptive_weights["fairness"] * fairness_score
        )
        
        return score
    
    def _adapt_scheduling_weights(self):
        """自适应调整调度权重"""
        # 根据系统负载和性能调整权重
        system_load = self._calculate_system_load()
        recent_performance = self._calculate_recent_performance()
        
        if system_load > 0.8:
            # 高负载时优先负载均衡
            self.adaptive_weights["load_balance"] = 0.5
            self.adaptive_weights["priority"] = 0.2
        elif recent_performance < 0.7:
            # 性能不佳时优先性能匹配
            self.adaptive_weights["performance"] = 0.4
            self.adaptive_weights["priority"] = 0.3
        else:
            # 正常情况下平衡各因素
            self.adaptive_weights = {
                "priority": 0.3,
                "load_balance": 0.3,
                "performance": 0.2,
                "fairness": 0.2
            }
    
    def _estimate_task_time(self, task: BatchTask, worker_metrics: Optional[WorkerMetrics]) -> float:
        """估算任务执行时间"""
        if not worker_metrics or worker_metrics.average_task_time <= 0:
            return 60.0  # 默认1分钟
        
        # 基础时间
        base_time = worker_metrics.average_task_time
        
        # 任务复杂度因子（基于数据大小、类型等）
        complexity_factor = 1.0
        if hasattr(task.data, '__len__'):
            data_size = len(str(task.data))
            complexity_factor = min(3.0, 1.0 + data_size / 10000)
        
        # 重试惩罚
        retry_factor = 1.0 + (task.retry_count * 0.2)
        
        return base_time * complexity_factor * retry_factor
    
    def _calculate_system_load(self) -> float:
        """计算系统负载"""
        if not self.worker_metrics:
            return 0.0
        
        loads = [m.current_load for m in self.worker_metrics.values()]
        return statistics.mean(loads) if loads else 0.0
    
    def _calculate_recent_performance(self) -> float:
        """计算最近性能"""
        if not self._performance_history:
            return 1.0
        
        # 取最近10个性能记录
        recent_scores = self._performance_history[-10:]
        return statistics.mean(recent_scores)
    
    async def _try_assign_task(self) -> Optional[str]:
        """尝试立即分配任务"""
        # 寻找空闲工作者
        idle_workers = [
            worker_id for worker_id, metrics in self.worker_metrics.items()
            if metrics.current_load < self.load_threshold
        ]
        
        if idle_workers and self._task_queue:
            # 选择负载最低的工作者
            best_worker = min(idle_workers, key=lambda w: self.worker_metrics[w].current_load)
            
            task_info = await self.get_next_task(best_worker)
            if task_info:
                task, job_id = task_info
                # 更新工作者负载
                self.worker_metrics[best_worker].current_load += 1
                self.worker_assignments[best_worker].append(task.id)
                
                logger.debug(f"任务分配给工作者: {task.id} -> {best_worker}")
                return best_worker
        
        return None
    
    async def task_completed(self, worker_id: str, task: BatchTask, success: bool, execution_time: float):
        """任务完成回调"""
        if worker_id in self.worker_metrics:
            # 更新工作者指标（包含任务类型）
            self.worker_metrics[worker_id].update_metrics(execution_time, success, task.type)
            self.worker_metrics[worker_id].current_load = max(0, 
                self.worker_metrics[worker_id].current_load - 1)
            
            # 移除任务分配
            if task.id in self.worker_assignments.get(worker_id, []):
                self.worker_assignments[worker_id].remove(task.id)
            
            # 记录性能历史
            performance_score = self.worker_metrics[worker_id].performance_score
            self._performance_history.append(performance_score)
            if len(self._performance_history) > 100:
                self._performance_history = self._performance_history[-50:]
            
            # 检查SLA违反
            if task.type in self.sla_requirements:
                sla = self.sla_requirements[task.type]
                if execution_time > sla.max_completion_time:
                    logger.warning(f"SLA违反: 任务 {task.id} 类型 {task.type} 完成时间 {execution_time:.2f}s "
                                 f"超过SLA要求 {sla.max_completion_time}s")
            
            logger.debug(f"任务完成统计: {task.id} 在 {worker_id} (成功: {success}, 时间: {execution_time:.2f}s)")
    
    async def get_predictive_recommendations(self) -> Dict[str, Any]:
        """获取预测性调度建议"""
        if not self._throughput_history or not self._resource_usage_history:
            return {"recommendations": [], "confidence": 0.0}
        
        recommendations = []
        queue_length = len(self._task_queue)
        avg_completion_time = self._calculate_recent_performance()
        
        # 是否需要扩容
        if self._prediction_model.should_scale_up(queue_length, avg_completion_time, self.system_resources):
            optimal_workers = self._prediction_model.get_optimal_worker_count(
                queue_length, avg_completion_time, 300  # 5分钟目标完成时间
            )
            recommendations.append({
                "type": "scale_up",
                "reason": "队列积压或平均完成时间过长",
                "suggested_workers": optimal_workers,
                "current_workers": len(self.worker_metrics)
            })
        
        # 资源利用率分析
        if self.system_resources.cpu_usage > 85:
            recommendations.append({
                "type": "cpu_optimization",
                "reason": "CPU使用率过高",
                "suggestion": "优先调度轻量级任务或减少并发度"
            })
        
        if self.system_resources.memory_usage > 90:
            recommendations.append({
                "type": "memory_optimization", 
                "reason": "内存使用率过高",
                "suggestion": "强制垃圾回收或重启部分工作者"
            })
        
        # SLA风险预警
        sla_risks = []
        for task in [item[2] for item in self._task_queue[:10]]:  # 检查前10个任务
            if task.type in self.sla_requirements:
                predicted_time = self._predict_task_completion_time(task)
                sla_time = self.sla_requirements[task.type].max_completion_time
                if predicted_time > sla_time * 0.8:  # 80%阈值预警
                    sla_risks.append({
                        "task_id": task.id,
                        "task_type": task.type,
                        "predicted_time": predicted_time,
                        "sla_time": sla_time,
                        "risk_level": "high" if predicted_time > sla_time else "medium"
                    })
        
        if sla_risks:
            recommendations.append({
                "type": "sla_risk",
                "reason": "部分任务有SLA违反风险",
                "at_risk_tasks": sla_risks
            })
        
        # 计算建议可信度
        confidence = min(1.0, len(self._throughput_history) / 20)  # 至少20个数据点达到最高可信度
        
        return {
            "recommendations": recommendations,
            "confidence": confidence,
            "system_metrics": {
                "queue_length": queue_length,
                "avg_completion_time": avg_completion_time,
                "cpu_usage": self.system_resources.cpu_usage,
                "memory_usage": self.system_resources.memory_usage,
                "active_workers": len([w for w in self.worker_metrics.values() if w.current_load > 0])
            }
        }
    
    async def get_queue_stats(self) -> Dict[str, Any]:
        """获取队列统计信息"""
        async with self._queue_lock:
            # 按优先级分组
            priority_counts = {}
            for priority, _, task, _ in self._task_queue:
                priority = -priority  # 恢复原始优先级
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            # 按任务类型分组
            type_counts = {}
            for _, _, task, _ in self._task_queue:
                type_counts[task.type] = type_counts.get(task.type, 0) + 1
            
            return {
                "total_queued": len(self._task_queue),
                "max_queue_size": self.max_queue_size,
                "queue_utilization": len(self._task_queue) / self.max_queue_size,
                "total_scheduled": self._total_scheduled,
                "strategy": self.strategy.value,
                "priority_breakdown": priority_counts,
                "type_breakdown": type_counts,
                "adaptive_weights": self.adaptive_weights.copy(),
                "system_load": self._calculate_system_load()
            }
    
    async def get_worker_stats(self) -> Dict[str, Any]:
        """获取工作者统计信息"""
        worker_stats = {}
        
        for worker_id, metrics in self.worker_metrics.items():
            worker_stats[worker_id] = {
                "current_load": metrics.current_load,
                "total_tasks_processed": metrics.total_tasks_processed,
                "average_task_time": metrics.average_task_time,
                "success_rate": metrics.success_rate,
                "performance_score": metrics.performance_score,
                "assigned_tasks": len(self.worker_assignments.get(worker_id, [])),
                "last_task_completed": metrics.last_task_completed.isoformat() 
                                    if metrics.last_task_completed else None
            }
        
        return {
            "total_workers": len(self.worker_metrics),
            "workers": worker_stats,
            "system_load": self._calculate_system_load(),
            "recent_performance": self._calculate_recent_performance()
        }
    
    def set_strategy(self, strategy: SchedulingStrategy):
        """设置调度策略"""
        self.strategy = strategy
        logger.info(f"调度策略更改为: {strategy.value}")
    
    def clear_queue(self):
        """清空任务队列"""
        self._task_queue.clear()
        # 更新线程安全统计缓存
        with self._stats_lock:
            self._thread_safe_stats['queue_length'] = 0
        logger.info("任务队列已清空")
