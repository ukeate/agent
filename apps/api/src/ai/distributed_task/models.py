"""分布式任务协调数据模型"""

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    DECOMPOSED = "decomposed"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class TaskPriority(Enum):
    """任务优先级枚举"""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    BACKGROUND = 5


class ConsensusState(Enum):
    """Raft共识状态"""
    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


class ConflictType(Enum):
    """冲突类型枚举"""
    RESOURCE_CONFLICT = "resource_conflict"
    STATE_CONFLICT = "state_conflict"
    ASSIGNMENT_CONFLICT = "assignment_conflict"
    DEPENDENCY_CONFLICT = "dependency_conflict"


@dataclass
class Task:
    """任务数据结构"""
    task_id: str
    task_type: str
    data: Dict[str, Any]
    requirements: Dict[str, Any]
    priority: TaskPriority
    created_at: datetime
    parent_task_id: Optional[str] = None
    subtask_ids: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    assigned_to: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 3600
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    execution_context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 转换枚举和datetime类型
        data['status'] = self.status.value
        data['priority'] = self.priority.value
        data['created_at'] = self.created_at.isoformat()
        if self.started_at:
            data['started_at'] = self.started_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Task':
        """从字典创建"""
        # 转换枚举和datetime类型
        if 'status' in data:
            data['status'] = TaskStatus(data['status'])
        if 'priority' in data:
            data['priority'] = TaskPriority(data['priority'])
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'started_at' in data and data['started_at'] and isinstance(data['started_at'], str):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if 'completed_at' in data and data['completed_at'] and isinstance(data['completed_at'], str):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


@dataclass
class RaftLogEntry:
    """Raft日志条目"""
    term: int
    index: int
    timestamp: datetime
    command_type: str
    command_data: Dict[str, Any]
    client_id: str
    sequence_number: int
    
    def __post_init__(self):
        self.entry_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        """计算日志条目哈希值"""
        content = f"{self.term}-{self.index}-{self.command_type}-{json.dumps(self.command_data, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class VoteRequest:
    """投票请求"""
    term: int
    candidate_id: str
    last_log_index: int
    last_log_term: int
    election_id: str


@dataclass
class VoteResponse:
    """投票响应"""
    term: int
    vote_granted: bool
    voter_id: str
    reason: Optional[str] = None


@dataclass
class AppendEntriesRequest:
    """日志追加请求"""
    term: int
    leader_id: str
    prev_log_index: int
    prev_log_term: int
    entries: List[RaftLogEntry]
    leader_commit: int
    heartbeat: bool = False


@dataclass
class AppendEntriesResponse:
    """日志追加响应"""
    term: int
    success: bool
    match_index: int
    follower_id: str
    conflict_index: Optional[int] = None
    reason: Optional[str] = None


@dataclass
class Conflict:
    """冲突信息"""
    conflict_id: str
    conflict_type: ConflictType
    description: str
    involved_tasks: List[str]
    involved_agents: List[str]
    timestamp: datetime
    resolved: bool = False
    resolution_strategy: Optional[str] = None
    resolution_result: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['conflict_type'] = self.conflict_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data