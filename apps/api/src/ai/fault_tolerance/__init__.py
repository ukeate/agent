"""
AI Agent System - 容错和恢复系统

这个模块实现了完整的容错和恢复系统，包括：
- 智能体故障检测和隔离
- 任务重分配和恢复策略
- 分布式备份和数据一致性
- 网络分区处理和脑裂防护
"""

from .fault_detector import FaultDetector, FaultType, FaultSeverity, FaultEvent, HealthStatus
from .recovery_manager import RecoveryManager, RecoveryStrategy
from .backup_manager import BackupManager, BackupType, BackupRecord
from .consistency_manager import ConsistencyManager, ConsistencyCheckResult
from .fault_tolerance_system import FaultToleranceSystem

__all__ = [
    'FaultDetector',
    'RecoveryManager', 
    'BackupManager',
    'ConsistencyManager',
    'FaultToleranceSystem',
    'FaultType',
    'FaultSeverity',
    'RecoveryStrategy',
    'BackupType',
    'FaultEvent',
    'HealthStatus',
    'BackupRecord',
    'ConsistencyCheckResult'
]
