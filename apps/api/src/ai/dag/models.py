"""
DAG任务编排数据模型
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import uuid

class TaskStatus(str, Enum):
    """任务状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"

@dataclass
class DAGNode:
    """DAG节点定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    dependencies: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_dependency(self, node_id: str):
        """添加依赖节点"""
        self.dependencies.add(node_id)
    
    def remove_dependency(self, node_id: str):
        """移除依赖节点"""
        self.dependencies.discard(node_id)

@dataclass  
class DAGTask:
    """DAG任务实例"""
    node: DAGNode
    status: TaskStatus = TaskStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def start(self):
        """开始任务执行"""
        self.status = TaskStatus.RUNNING
        self.start_time = utc_now()
    
    def complete(self, result: Any = None):
        """完成任务执行"""
        self.status = TaskStatus.COMPLETED
        self.end_time = utc_now()
        self.result = result
    
    def fail(self, error: str):
        """标记任务失败"""
        self.status = TaskStatus.FAILED
        self.end_time = utc_now()
        self.error = error
    
    def can_retry(self) -> bool:
        """检查是否可以重试"""
        return self.retry_count < self.max_retries
    
    def retry(self):
        """重试任务"""
        if self.can_retry():
            self.retry_count += 1
            self.status = TaskStatus.PENDING
            self.start_time = None
            self.end_time = None
            self.error = None

@dataclass
class DAGWorkflow:
    """DAG工作流定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    tasks: Dict[str, DAGTask] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_factory)
    
    def add_node(self, node: DAGNode) -> str:
        """添加节点到工作流"""
        self.nodes[node.id] = node
        self.tasks[node.id] = DAGTask(node=node)
        return node.id
    
    def add_dependency(self, from_node_id: str, to_node_id: str):
        """添加节点依赖关系"""
        if to_node_id in self.nodes:
            self.nodes[to_node_id].add_dependency(from_node_id)
    
    def get_ready_tasks(self) -> List[DAGTask]:
        """获取可以执行的任务（依赖已满足的任务）"""
        ready_tasks = []
        
        for task in self.tasks.values():
            if task.status == TaskStatus.PENDING:
                # 检查所有依赖是否已完成
                dependencies_completed = True
                for dep_id in task.node.dependencies:
                    if dep_id in self.tasks:
                        dep_task = self.tasks[dep_id]
                        if dep_task.status != TaskStatus.COMPLETED:
                            dependencies_completed = False
                            break
                
                if dependencies_completed:
                    ready_tasks.append(task)
        
        return ready_tasks
    
    def is_completed(self) -> bool:
        """检查工作流是否完成"""
        return all(
            task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
            for task in self.tasks.values()
        )
    
    def get_failed_tasks(self) -> List[DAGTask]:
        """获取失败的任务"""
        return [task for task in self.tasks.values() if task.status == TaskStatus.FAILED]
    
    def validate_dag(self) -> bool:
        """验证DAG是否有效（无循环依赖）"""
        # 使用深度优先搜索检测循环
        visited = set()
        rec_stack = set()
        
        def has_cycle(node_id: str) -> bool:
            if node_id in rec_stack:
                return True
            if node_id in visited:
                return False
            
            visited.add(node_id)
            rec_stack.add(node_id)
            
            # 检查所有依赖节点
            node = self.nodes.get(node_id)
            if node:
                for dep_id in node.dependencies:
                    if has_cycle(dep_id):
                        return True
            
            rec_stack.remove(node_id)
            return False
        
        # 检查所有节点
        for node_id in self.nodes:
            if node_id not in visited:
                if has_cycle(node_id):
                    return False
        
        return True
