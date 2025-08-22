"""
数据同步引擎

实现增量数据同步、断点续传和同步优先级管理
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from ..models.schemas.offline import (
    SyncOperation, SyncOperationType, VectorClock, 
    NetworkStatus, ConflictRecord, ConflictType
)
from ..offline.models import OfflineDatabase
from .vector_clock import VectorClockManager
from .delta_calculator import DeltaCalculator


class SyncPriority(int, Enum):
    """同步优先级"""
    CRITICAL = 1    # 关键数据（用户信息、权限等）
    HIGH = 2        # 高优先级（用户操作、重要业务数据）
    NORMAL = 3      # 普通优先级（一般业务数据）
    LOW = 4         # 低优先级（日志、统计数据等）
    BACKGROUND = 5  # 后台同步（缓存、临时数据等）


class SyncDirection(str, Enum):
    """同步方向"""
    UPLOAD = "upload"      # 上传到服务器
    DOWNLOAD = "download"  # 从服务器下载
    BIDIRECTIONAL = "bidirectional"  # 双向同步


class SyncStatus(str, Enum):
    """同步状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


@dataclass
class SyncTask:
    """同步任务"""
    id: str
    session_id: str
    direction: SyncDirection
    priority: SyncPriority
    operation_ids: List[str]
    status: SyncStatus = SyncStatus.PENDING
    progress: float = 0.0
    total_operations: int = 0
    completed_operations: int = 0
    failed_operations: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    
    def update_progress(self):
        """更新进度"""
        if self.total_operations > 0:
            self.progress = self.completed_operations / self.total_operations
        else:
            self.progress = 0.0


@dataclass 
class SyncResult:
    """同步结果"""
    task_id: str
    success: bool
    total_operations: int
    successful_operations: int
    failed_operations: int
    conflicts_detected: int
    duration_seconds: float
    throughput_ops_per_second: float
    error_message: Optional[str] = None
    conflicts: List[ConflictRecord] = field(default_factory=list)


class SyncEngine:
    """数据同步引擎"""
    
    def __init__(self, database: OfflineDatabase):
        self.database = database
        self.vector_clock_manager = VectorClockManager()
        self.delta_calculator = DeltaCalculator()
        
        # 同步任务管理
        self.active_tasks: Dict[str, SyncTask] = {}
        self.task_queue: List[SyncTask] = []
        
        # 同步配置
        self.max_concurrent_tasks = 3
        self.batch_size = 100
        self.sync_timeout = timedelta(minutes=30)
        self.retry_delay = timedelta(seconds=30)
        
        # 优先级权重
        self.priority_weights = {
            SyncPriority.CRITICAL: 1.0,
            SyncPriority.HIGH: 0.8,
            SyncPriority.NORMAL: 0.6,
            SyncPriority.LOW: 0.4,
            SyncPriority.BACKGROUND: 0.2
        }
        
        # 同步统计
        self.total_synced_operations = 0
        self.total_failed_operations = 0
        self.total_conflicts_resolved = 0
        self.last_sync_time = None
        
        # 断点续传支持
        self.checkpoint_interval = 50  # 每50个操作创建一个检查点
        
    async def create_sync_task(
        self,
        session_id: str,
        direction: SyncDirection,
        priority: SyncPriority = SyncPriority.NORMAL,
        operation_filter: Optional[Dict[str, Any]] = None
    ) -> str:
        """创建同步任务"""
        # 获取待同步操作
        if direction in [SyncDirection.UPLOAD, SyncDirection.BIDIRECTIONAL]:
            pending_ops = self.database.get_pending_operations(
                session_id, 
                limit=10000  # 获取所有待同步操作
            )
            
            # 应用过滤器
            if operation_filter:
                pending_ops = self._filter_operations(pending_ops, operation_filter)
            
            operation_ids = [op.id for op in pending_ops]
        else:
            # 下载任务的操作ID将在执行时确定
            operation_ids = []
        
        # 创建同步任务
        task = SyncTask(
            id=str(uuid4()),
            session_id=session_id,
            direction=direction,
            priority=priority,
            operation_ids=operation_ids,
            total_operations=len(operation_ids)
        )
        
        # 添加到任务队列
        self.task_queue.append(task)
        self._sort_task_queue()
        
        return task.id
    
    def _filter_operations(
        self, 
        operations: List[SyncOperation], 
        filter_criteria: Dict[str, Any]
    ) -> List[SyncOperation]:
        """过滤操作"""
        filtered_ops = operations
        
        # 按表名过滤
        if "table_names" in filter_criteria:
            table_names = set(filter_criteria["table_names"])
            filtered_ops = [op for op in filtered_ops if op.table_name in table_names]
        
        # 按操作类型过滤
        if "operation_types" in filter_criteria:
            op_types = set(filter_criteria["operation_types"])
            filtered_ops = [op for op in filtered_ops if op.operation_type in op_types]
        
        # 按时间范围过滤
        if "from_time" in filter_criteria:
            from_time = filter_criteria["from_time"]
            filtered_ops = [op for op in filtered_ops if op.client_timestamp >= from_time]
        
        if "to_time" in filter_criteria:
            to_time = filter_criteria["to_time"]
            filtered_ops = [op for op in filtered_ops if op.client_timestamp <= to_time]
        
        return filtered_ops
    
    def _sort_task_queue(self):
        """按优先级排序任务队列"""
        self.task_queue.sort(key=lambda task: (
            task.priority.value,  # 数值越小优先级越高
            task.created_at       # 同优先级按创建时间排序
        ))
    
    async def execute_sync_task(self, task_id: str) -> SyncResult:
        """执行同步任务"""
        if task_id not in self.active_tasks:
            # 从队列中获取任务
            task = next((t for t in self.task_queue if t.id == task_id), None)
            if not task:
                raise ValueError(f"同步任务不存在: {task_id}")
            
            self.task_queue.remove(task)
            self.active_tasks[task_id] = task
        else:
            task = self.active_tasks[task_id]
        
        # 开始执行
        task.status = SyncStatus.IN_PROGRESS
        task.started_at = datetime.utcnow()
        
        try:
            if task.direction == SyncDirection.UPLOAD:
                result = await self._execute_upload_task(task)
            elif task.direction == SyncDirection.DOWNLOAD:
                result = await self._execute_download_task(task)
            else:  # BIDIRECTIONAL
                result = await self._execute_bidirectional_task(task)
            
            task.status = SyncStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            
            return result
            
        except Exception as e:
            task.status = SyncStatus.FAILED
            task.error_message = str(e)
            task.retry_count += 1
            
            # 如果可以重试，重新加入队列
            if task.retry_count < task.max_retries:
                task.status = SyncStatus.PENDING
                await asyncio.sleep(self.retry_delay.total_seconds())
                self.task_queue.append(task)
                self._sort_task_queue()
                
                # 从活跃任务中移除
                self.active_tasks.pop(task_id, None)
                
                # 递归重试
                return await self.execute_sync_task(task_id)
            
            # 重试次数用尽，标记为失败
            return SyncResult(
                task_id=task_id,
                success=False,
                total_operations=task.total_operations,
                successful_operations=task.completed_operations,
                failed_operations=task.failed_operations,
                conflicts_detected=0,
                duration_seconds=0,
                throughput_ops_per_second=0,
                error_message=str(e)
            )
        
        finally:
            # 清理活跃任务
            self.active_tasks.pop(task_id, None)
    
    async def _execute_upload_task(self, task: SyncTask) -> SyncResult:
        """执行上传任务"""
        start_time = datetime.utcnow()
        conflicts = []
        
        # 批量处理操作
        operation_batches = self._create_operation_batches(task.operation_ids)
        
        for batch_index, batch_ids in enumerate(operation_batches):
            # 检查任务是否被取消
            if task.status == SyncStatus.CANCELLED:
                break
            
            # 获取批次操作
            batch_operations = []
            for op_id in batch_ids:
                operations = self.database.get_pending_operations(task.session_id, limit=1000)
                op = next((o for o in operations if o.id == op_id), None)
                if op:
                    batch_operations.append(op)
            
            # 上传批次
            batch_result = await self._upload_operation_batch(
                task.session_id, 
                batch_operations
            )
            
            # 更新任务进度
            task.completed_operations += batch_result["successful_operations"]
            task.failed_operations += batch_result["failed_operations"]
            conflicts.extend(batch_result.get("conflicts", []))
            
            task.update_progress()
            
            # 创建检查点
            if (batch_index + 1) % self.checkpoint_interval == 0:
                task.checkpoint_data = {
                    "completed_batches": batch_index + 1,
                    "completed_operations": task.completed_operations,
                    "last_checkpoint": datetime.utcnow().isoformat()
                }
        
        # 计算结果
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        throughput = task.completed_operations / duration if duration > 0 else 0
        
        # 更新统计
        self.total_synced_operations += task.completed_operations
        self.total_failed_operations += task.failed_operations
        self.total_conflicts_resolved += len(conflicts)
        self.last_sync_time = end_time
        
        return SyncResult(
            task_id=task.id,
            success=task.failed_operations == 0,
            total_operations=task.total_operations,
            successful_operations=task.completed_operations,
            failed_operations=task.failed_operations,
            conflicts_detected=len(conflicts),
            duration_seconds=duration,
            throughput_ops_per_second=throughput,
            conflicts=conflicts
        )
    
    async def _execute_download_task(self, task: SyncTask) -> SyncResult:
        """执行下载任务"""
        start_time = datetime.utcnow()
        
        # 获取服务器端的更新
        # 这里模拟从服务器获取数据的过程
        server_operations = await self._fetch_server_operations(task.session_id)
        
        task.total_operations = len(server_operations)
        conflicts = []
        
        # 批量处理下载的操作
        for operation in server_operations:
            try:
                # 检查冲突
                conflict = await self._detect_download_conflict(operation)
                if conflict:
                    conflicts.append(conflict)
                    task.failed_operations += 1
                else:
                    # 应用操作到本地
                    await self._apply_operation_locally(operation)
                    task.completed_operations += 1
                
                task.update_progress()
                
            except Exception as e:
                print(f"下载操作失败: {operation.id}, 错误: {e}")
                task.failed_operations += 1
        
        # 计算结果
        end_time = datetime.utcnow()
        duration = (end_time - start_time).total_seconds()
        throughput = task.completed_operations / duration if duration > 0 else 0
        
        return SyncResult(
            task_id=task.id,
            success=task.failed_operations == 0,
            total_operations=task.total_operations,
            successful_operations=task.completed_operations,
            failed_operations=task.failed_operations,
            conflicts_detected=len(conflicts),
            duration_seconds=duration,
            throughput_ops_per_second=throughput,
            conflicts=conflicts
        )
    
    async def _execute_bidirectional_task(self, task: SyncTask) -> SyncResult:
        """执行双向同步任务"""
        # 双向同步分为两个阶段：上传和下载
        upload_task = SyncTask(
            id=str(uuid4()),
            session_id=task.session_id,
            direction=SyncDirection.UPLOAD,
            priority=task.priority,
            operation_ids=task.operation_ids,
            total_operations=len(task.operation_ids)
        )
        
        download_task = SyncTask(
            id=str(uuid4()),
            session_id=task.session_id,
            direction=SyncDirection.DOWNLOAD,
            priority=task.priority,
            operation_ids=[],
            total_operations=0
        )
        
        # 执行上传
        upload_result = await self._execute_upload_task(upload_task)
        
        # 执行下载
        download_result = await self._execute_download_task(download_task)
        
        # 合并结果
        total_ops = upload_result.total_operations + download_result.total_operations
        successful_ops = upload_result.successful_operations + download_result.successful_operations
        failed_ops = upload_result.failed_operations + download_result.failed_operations
        conflicts = upload_result.conflicts + download_result.conflicts
        
        task.completed_operations = successful_ops
        task.failed_operations = failed_ops
        task.total_operations = total_ops
        task.update_progress()
        
        return SyncResult(
            task_id=task.id,
            success=failed_ops == 0,
            total_operations=total_ops,
            successful_operations=successful_ops,
            failed_operations=failed_ops,
            conflicts_detected=len(conflicts),
            duration_seconds=upload_result.duration_seconds + download_result.duration_seconds,
            throughput_ops_per_second=(successful_ops / (upload_result.duration_seconds + download_result.duration_seconds)) if (upload_result.duration_seconds + download_result.duration_seconds) > 0 else 0,
            conflicts=conflicts
        )
    
    def _create_operation_batches(self, operation_ids: List[str]) -> List[List[str]]:
        """创建操作批次"""
        batches = []
        for i in range(0, len(operation_ids), self.batch_size):
            batch = operation_ids[i:i + self.batch_size]
            batches.append(batch)
        return batches
    
    async def _upload_operation_batch(
        self, 
        session_id: str, 
        operations: List[SyncOperation]
    ) -> Dict[str, Any]:
        """上传操作批次"""
        successful_operations = 0
        failed_operations = 0
        conflicts = []
        
        for operation in operations:
            try:
                # 模拟上传到服务器
                success = await self._upload_single_operation(operation)
                
                if success:
                    # 标记为已同步
                    self.database.mark_operation_synced(
                        operation.id, 
                        datetime.utcnow()
                    )
                    successful_operations += 1
                else:
                    failed_operations += 1
                
            except Exception as e:
                print(f"上传操作失败: {operation.id}, 错误: {e}")
                failed_operations += 1
        
        return {
            "successful_operations": successful_operations,
            "failed_operations": failed_operations,
            "conflicts": conflicts
        }
    
    async def _upload_single_operation(self, operation: SyncOperation) -> bool:
        """上传单个操作（模拟）"""
        # 这里应该实现实际的网络上传逻辑
        # 模拟网络延迟和成功率
        await asyncio.sleep(0.1)  # 模拟网络延迟
        
        import random
        return random.random() > 0.05  # 95% 成功率
    
    async def _fetch_server_operations(self, session_id: str) -> List[SyncOperation]:
        """从服务器获取操作（模拟）"""
        # 这里应该实现实际的服务器数据获取逻辑
        # 模拟返回一些服务器端的操作
        await asyncio.sleep(0.5)  # 模拟网络延迟
        
        # 模拟服务器操作
        server_ops = []
        for i in range(5):  # 模拟5个服务器操作
            op = SyncOperation(
                id=str(uuid4()),
                session_id=session_id,
                operation_type=SyncOperationType.PUT,
                table_name="server_data",
                object_id=f"server_obj_{i}",
                object_type="server_data",
                data={"server_field": f"server_value_{i}"},
                client_timestamp=datetime.utcnow(),
                vector_clock=VectorClock(node_id="server")
            )
            server_ops.append(op)
        
        return server_ops
    
    async def _detect_download_conflict(self, operation: SyncOperation) -> Optional[ConflictRecord]:
        """检测下载冲突"""
        # 检查本地是否有同一对象的更新
        local_operations = self.database.get_pending_operations(
            operation.session_id, 
            limit=1000
        )
        
        # 查找同一对象的本地操作
        conflicting_op = next((
            op for op in local_operations 
            if op.table_name == operation.table_name and op.object_id == operation.object_id
        ), None)
        
        if conflicting_op:
            # 使用向量时钟判断是否冲突
            if self.vector_clock_manager.detect_conflict(
                conflicting_op.vector_clock, 
                operation.vector_clock
            ):
                return ConflictRecord(
                    id=str(uuid4()),
                    session_id=operation.session_id,
                    table_name=operation.table_name,
                    object_id=operation.object_id,
                    conflict_type=ConflictType.UPDATE_UPDATE,
                    local_data=conflicting_op.data or {},
                    remote_data=operation.data or {},
                    local_vector_clock=conflicting_op.vector_clock,
                    remote_vector_clock=operation.vector_clock,
                    detected_at=datetime.utcnow()
                )
        
        return None
    
    async def _apply_operation_locally(self, operation: SyncOperation):
        """将操作应用到本地"""
        # 这里应该实现将服务器操作应用到本地数据库的逻辑
        # 简化实现，直接添加到操作记录
        self.database.add_operation(operation)
    
    async def pause_sync_task(self, task_id: str) -> bool:
        """暂停同步任务"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if task.status == SyncStatus.IN_PROGRESS:
                task.status = SyncStatus.PAUSED
                return True
        
        # 从队列中暂停
        task = next((t for t in self.task_queue if t.id == task_id), None)
        if task:
            task.status = SyncStatus.PAUSED
            return True
        
        return False
    
    async def resume_sync_task(self, task_id: str) -> bool:
        """恢复同步任务"""
        # 检查活跃任务
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            if task.status == SyncStatus.PAUSED:
                task.status = SyncStatus.IN_PROGRESS
                return True
        
        # 检查队列中的任务
        task = next((t for t in self.task_queue if t.id == task_id), None)
        if task and task.status == SyncStatus.PAUSED:
            task.status = SyncStatus.PENDING
            self._sort_task_queue()
            return True
        
        return False
    
    async def cancel_sync_task(self, task_id: str) -> bool:
        """取消同步任务"""
        # 取消活跃任务
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = SyncStatus.CANCELLED
            return True
        
        # 从队列中移除
        task = next((t for t in self.task_queue if t.id == task_id), None)
        if task:
            task.status = SyncStatus.CANCELLED
            self.task_queue.remove(task)
            return True
        
        return False
    
    def get_sync_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取同步任务状态"""
        # 检查活跃任务
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
        else:
            # 检查队列
            task = next((t for t in self.task_queue if t.id == task_id), None)
        
        if not task:
            return None
        
        return {
            "task_id": task.id,
            "session_id": task.session_id,
            "direction": task.direction.value,
            "priority": task.priority.value,
            "status": task.status.value,
            "progress": task.progress,
            "total_operations": task.total_operations,
            "completed_operations": task.completed_operations,
            "failed_operations": task.failed_operations,
            "created_at": task.created_at.isoformat(),
            "started_at": task.started_at.isoformat() if task.started_at else None,
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "error_message": task.error_message,
            "retry_count": task.retry_count,
            "checkpoint_data": task.checkpoint_data
        }
    
    def list_sync_tasks(
        self, 
        session_id: Optional[str] = None,
        status_filter: Optional[List[SyncStatus]] = None
    ) -> List[Dict[str, Any]]:
        """列出同步任务"""
        all_tasks = list(self.active_tasks.values()) + self.task_queue
        
        # 应用过滤器
        if session_id:
            all_tasks = [t for t in all_tasks if t.session_id == session_id]
        
        if status_filter:
            all_tasks = [t for t in all_tasks if t.status in status_filter]
        
        # 转换为字典格式
        return [
            {
                "task_id": task.id,
                "session_id": task.session_id,
                "direction": task.direction.value,
                "priority": task.priority.value,
                "status": task.status.value,
                "progress": task.progress,
                "total_operations": task.total_operations,
                "created_at": task.created_at.isoformat()
            }
            for task in all_tasks
        ]
    
    def get_sync_statistics(self) -> Dict[str, Any]:
        """获取同步统计信息"""
        active_tasks_count = len(self.active_tasks)
        queued_tasks_count = len(self.task_queue)
        
        # 按状态统计任务
        status_counts = {}
        all_tasks = list(self.active_tasks.values()) + self.task_queue
        for task in all_tasks:
            status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
        
        # 按优先级统计任务
        priority_counts = {}
        for task in all_tasks:
            priority_counts[task.priority.value] = priority_counts.get(task.priority.value, 0) + 1
        
        return {
            "active_tasks": active_tasks_count,
            "queued_tasks": queued_tasks_count,
            "total_tasks": active_tasks_count + queued_tasks_count,
            "status_distribution": status_counts,
            "priority_distribution": priority_counts,
            "total_synced_operations": self.total_synced_operations,
            "total_failed_operations": self.total_failed_operations,
            "total_conflicts_resolved": self.total_conflicts_resolved,
            "last_sync_time": self.last_sync_time.isoformat() if self.last_sync_time else None,
            "sync_efficiency": (
                self.total_synced_operations / (self.total_synced_operations + self.total_failed_operations)
                if (self.total_synced_operations + self.total_failed_operations) > 0 else 0
            )
        }