"""
AI智能体集群管理模块

提供智能体集群的生命周期管理、拓扑跟踪、资源监控和自动扩缩容功能。
基于Kubernetes集群管理最佳实践设计。
"""

from .topology import ClusterTopology, AgentInfo, AgentStatus, AgentGroup, AgentCapability, ResourceSpec
from .state_manager import ClusterStateManager
from .lifecycle_manager import LifecycleManager
from .metrics_collector import MetricsCollector
from .auto_scaler import AutoScaler, ScalingPolicy

__all__ = [
    'ClusterTopology',
    'AgentInfo', 
    'AgentStatus',
    'AgentGroup',
    'AgentCapability',
    'ResourceSpec',
    'ClusterStateManager',
    'LifecycleManager',
    'MetricsCollector',
    'AutoScaler',
    'ScalingPolicy'
]