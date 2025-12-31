"""
批处理共享类型定义

分离数据类型定义以解决循环依赖问题
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
import hashlib
import json

class BatchStatus(str, Enum):
    """批处理状态"""
    PENDING = "pending"
    RUNNING = "running" 
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskPriority(int, Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    URGENT = 10

@dataclass
class BatchTask:
    """批处理任务"""
    id: str
    type: str
    data: Any
    priority: int = TaskPriority.NORMAL
    retry_count: int = 0
    max_retries: int = 3
    timeout: Optional[float] = None
    dependencies: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)
    
    # 状态信息
    status: BatchStatus = BatchStatus.PENDING
    created_at: datetime = field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 结果信息
    result: Optional[Any] = None
    error: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None
    
    # 执行指标
    execution_time: Optional[float] = None
    memory_usage: Optional[int] = None
    
    # 新增：数据一致性和重试控制
    data_checksum: Optional[str] = None
    retry_strategy: str = "exponential_backoff"  # exponential_backoff, fixed_delay, linear_backoff
    retry_delays: List[float] = field(default_factory=lambda: [1, 2, 4, 8, 16])  # 自定义重试延迟
    circuit_breaker_threshold: int = 5  # 熔断器阈值
    idempotent: bool = True  # 是否幂等操作
    
    def __post_init__(self):
        # 计算数据校验和
        if self.data_checksum is None:
            self.data_checksum = self._calculate_data_checksum()
    
    def _calculate_data_checksum(self) -> str:
        """计算数据校验和"""
        try:
            data_str = json.dumps(self.data, sort_keys=True, ensure_ascii=False)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except (TypeError, ValueError):
            # 非JSON序列化数据，使用字符串表示
            return hashlib.sha256(str(self.data).encode()).hexdigest()
    
    def verify_data_integrity(self) -> bool:
        """验证数据完整性"""
        return self.data_checksum == self._calculate_data_checksum()
    
    def get_next_retry_delay(self) -> float:
        """获取下次重试延迟"""
        if self.retry_strategy == "fixed_delay":
            return self.retry_delays[0] if self.retry_delays else 2.0
        
        elif self.retry_strategy == "linear_backoff":
            return min(self.retry_count * 2, 60)
        
        elif self.retry_strategy == "exponential_backoff":
            if self.retry_count < len(self.retry_delays):
                return self.retry_delays[self.retry_count]
            else:
                # 超出预定义延迟，使用指数退避
                return min(2 ** self.retry_count, 300)  # 最大5分钟
        
        else:  # custom or default
            if self.retry_count < len(self.retry_delays):
                return self.retry_delays[self.retry_count]
            return 30.0  # 默认30秒

@dataclass
class BatchJob:
    """批处理作业"""
    id: str
    name: str
    tasks: List[BatchTask]
    priority: int = TaskPriority.NORMAL
    max_parallel_tasks: int = 10
    timeout: Optional[float] = None
    tags: Dict[str, str] = field(default_factory=dict)
    
    # 状态统计
    status: BatchStatus = BatchStatus.PENDING
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    
    # 时间信息
    created_at: datetime = field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 执行配置
    continue_on_failure: bool = True
    failure_threshold: float = 0.1  # 10%失败率阈值
    
    def __post_init__(self):
        self.total_tasks = len(self.tasks)
        
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_tasks == 0:
            return 1.0
        return self.completed_tasks / self.total_tasks
    
    @property
    def failure_rate(self) -> float:
        """失败率"""
        if self.total_tasks == 0:
            return 0.0
        return self.failed_tasks / self.total_tasks
    
    @property
    def progress(self) -> float:
        """进度百分比"""
        if self.total_tasks == 0:
            return 1.0
        processed = self.completed_tasks + self.failed_tasks + self.cancelled_tasks
        return processed / self.total_tasks
