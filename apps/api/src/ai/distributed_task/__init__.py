"""分布式任务协调引擎

基于Raft分布式共识算法实现智能体任务的分布式协调与管理。
支持任务分解、智能分配、状态同步和冲突解决。
"""

from .models import (
    Task,
    TaskStatus,
    TaskPriority,
    ConsensusState,
    ConflictType,
    Conflict,
    RaftLogEntry,
    VoteRequest,
    VoteResponse,
    AppendEntriesRequest,
    AppendEntriesResponse,
)
from .raft_consensus import RaftConsensusEngine
from .task_decomposer import TaskDecomposer
from .intelligent_assigner import IntelligentAssigner
from .state_manager import DistributedStateManager
from .conflict_resolver import ConflictResolver
from .coordination_engine import DistributedTaskCoordinationEngine

__all__ = [
    # Models
    "Task",
    "TaskStatus",
    "TaskPriority",
    "ConsensusState",
    "ConflictType",
    "Conflict",
    "RaftLogEntry",
    "VoteRequest",
    "VoteResponse",
    "AppendEntriesRequest",
    "AppendEntriesResponse",
    # Components
    "RaftConsensusEngine",
    "TaskDecomposer",
    "IntelligentAssigner",
    "DistributedStateManager",
    "ConflictResolver",
    "DistributedTaskCoordinationEngine",
]