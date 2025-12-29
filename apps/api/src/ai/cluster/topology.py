"""
集群拓扑跟踪模型

定义智能体集群的拓扑结构、状态跟踪和组管理模型。
基于Kubernetes节点和Pod管理模式设计。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Tuple
from enum import Enum
import time
import uuid
from datetime import datetime

class AgentStatus(Enum):
    """智能体状态枚举"""
    PENDING = "pending"              # 待启动
    RUNNING = "running"              # 运行中 
    STOPPING = "stopping"           # 停止中
    STOPPED = "stopped"             # 已停止
    FAILED = "failed"               # 失败
    UNKNOWN = "unknown"             # 未知状态
    UPGRADING = "upgrading"         # 升级中
    SCALING = "scaling"             # 扩缩容中
    MAINTENANCE = "maintenance"     # 维护中

class AgentCapability(Enum):
    """智能体能力类型"""
    COMPUTE = "compute"             # 计算能力
    STORAGE = "storage"             # 存储能力  
    NETWORK = "network"             # 网络能力
    REASONING = "reasoning"         # 推理能力
    MULTIMODAL = "multimodal"       # 多模态能力
    TOOL_USE = "tool_use"          # 工具使用能力

@dataclass
class ResourceSpec:
    """资源规格定义"""
    cpu_cores: float = 0.0          # CPU核心数
    memory_gb: float = 0.0          # 内存GB
    storage_gb: float = 0.0         # 存储GB
    gpu_count: int = 0              # GPU数量
    network_bandwidth: float = 0.0   # 网络带宽Mbps
    
    def __add__(self, other: 'ResourceSpec') -> 'ResourceSpec':
        """资源规格相加"""
        return ResourceSpec(
            cpu_cores=self.cpu_cores + other.cpu_cores,
            memory_gb=self.memory_gb + other.memory_gb,
            storage_gb=self.storage_gb + other.storage_gb,
            gpu_count=self.gpu_count + other.gpu_count,
            network_bandwidth=self.network_bandwidth + other.network_bandwidth
        )
    
    def __mul__(self, factor: float) -> 'ResourceSpec':
        """资源规格缩放"""
        return ResourceSpec(
            cpu_cores=self.cpu_cores * factor,
            memory_gb=self.memory_gb * factor,
            storage_gb=self.storage_gb * factor,
            gpu_count=int(self.gpu_count * factor),
            network_bandwidth=self.network_bandwidth * factor
        )

@dataclass
class ResourceUsage:
    """实时资源使用情况"""
    cpu_usage_percent: float = 0.0   # CPU使用率%
    memory_usage_percent: float = 0.0 # 内存使用率%
    storage_usage_percent: float = 0.0 # 存储使用率%
    gpu_usage_percent: float = 0.0    # GPU使用率%
    network_io_mbps: float = 0.0      # 网络IO Mbps
    active_tasks: int = 0             # 活跃任务数
    total_requests: int = 0           # 总请求数
    failed_requests: int = 0          # 失败请求数
    avg_response_time: float = 0.0    # 平均响应时间ms
    timestamp: float = field(default_factory=time.time)  # 时间戳
    
    @property
    def error_rate(self) -> float:
        """错误率计算"""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

@dataclass
class AgentHealthCheck:
    """智能体健康检查"""
    is_healthy: bool = True
    last_heartbeat: float = field(default_factory=time.time)
    health_check_interval: float = 30.0  # 健康检查间隔秒
    consecutive_failures: int = 0
    max_failures: int = 3
    health_details: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_responsive(self) -> bool:
        """检查是否响应"""
        current_time = time.time()
        return (current_time - self.last_heartbeat) < (self.health_check_interval * 2)
    
    @property
    def needs_restart(self) -> bool:
        """是否需要重启"""
        return self.consecutive_failures >= self.max_failures

@dataclass
class AgentInfo:
    """智能体信息模型"""
    agent_id: str = field(default_factory=lambda: f"agent-{uuid.uuid4().hex[:8]}")
    name: str = ""
    host: str = "localhost"
    port: int = 0
    endpoint: str = ""
    status: AgentStatus = AgentStatus.PENDING
    
    # 能力和版本信息
    capabilities: Set[AgentCapability] = field(default_factory=set)
    version: str = "1.0.0"
    api_version: str = "v1"
    
    # 资源信息
    resource_spec: ResourceSpec = field(default_factory=ResourceSpec)
    resource_usage: ResourceUsage = field(default_factory=ResourceUsage)
    
    # 健康状态
    health: AgentHealthCheck = field(default_factory=AgentHealthCheck)
    
    # 元数据和标签
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 组织信息
    group_id: Optional[str] = None
    cluster_id: str = "default"
    node_id: Optional[str] = None
    
    # 时间信息
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    
    # 配置信息
    config: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.name:
            self.name = self.agent_id
        if not self.endpoint and self.host and self.port:
            self.endpoint = f"http://{self.host}:{self.port}"
    
    def update_status(self, status: AgentStatus, details: Optional[str] = None):
        """更新状态"""
        self.status = status
        self.updated_at = time.time()
        if details:
            self.metadata["status_details"] = details
        
        if status == AgentStatus.RUNNING and self.started_at is None:
            self.started_at = time.time()
    
    def update_resource_usage(self, usage: ResourceUsage):
        """更新资源使用情况"""
        self.resource_usage = usage
        self.updated_at = time.time()
    
    def add_label(self, key: str, value: str):
        """添加标签"""
        self.labels[key] = value
        self.updated_at = time.time()
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """检查是否具有特定能力"""
        return capability in self.capabilities
    
    @property
    def is_healthy(self) -> bool:
        """健康状态检查"""
        return (
            self.health.is_healthy and 
            self.health.is_responsive and
            self.status in [AgentStatus.RUNNING, AgentStatus.SCALING]
        )
    
    @property
    def uptime_seconds(self) -> float:
        """运行时间（秒）"""
        if self.started_at is None:
            return 0.0
        return time.time() - self.started_at

@dataclass
class AgentGroup:
    """智能体分组模型"""
    group_id: str = field(default_factory=lambda: f"group-{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    
    # 成员管理
    agent_ids: Set[str] = field(default_factory=set)
    max_agents: Optional[int] = None
    min_agents: int = 0
    
    # 分组策略
    selection_strategy: str = "round_robin"  # round_robin, load_balanced, capability_based
    load_balancing: bool = True
    auto_scaling: bool = False
    
    # 资源限制
    resource_quota: Optional[ResourceSpec] = None
    
    # 元数据
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 时间信息
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.name:
            self.name = self.group_id
    
    def add_agent(self, agent_id: str) -> bool:
        """添加智能体到分组"""
        if self.max_agents and len(self.agent_ids) >= self.max_agents:
            return False
        
        self.agent_ids.add(agent_id)
        self.updated_at = time.time()
        return True
    
    def remove_agent(self, agent_id: str) -> bool:
        """从分组移除智能体"""
        if agent_id in self.agent_ids and len(self.agent_ids) > self.min_agents:
            self.agent_ids.remove(agent_id)
            self.updated_at = time.time()
            return True
        return False
    
    @property
    def agent_count(self) -> int:
        """智能体数量"""
        return len(self.agent_ids)
    
    @property
    def is_full(self) -> bool:
        """是否已满"""
        return self.max_agents is not None and self.agent_count >= self.max_agents
    
    @property
    def can_scale_down(self) -> bool:
        """是否可以缩减"""
        return self.agent_count > self.min_agents

@dataclass
class ClusterTopology:
    """集群拓扑结构模型"""
    cluster_id: str = "default"
    name: str = ""
    description: str = ""
    
    # 智能体管理
    agents: Dict[str, AgentInfo] = field(default_factory=dict)
    groups: Dict[str, AgentGroup] = field(default_factory=dict)
    
    # 拓扑关系
    agent_dependencies: Dict[str, Set[str]] = field(default_factory=dict)  # 智能体依赖关系
    communication_paths: Dict[str, Dict[str, float]] = field(default_factory=dict)  # 通信路径和延迟
    
    # 集群配置
    config: Dict[str, Any] = field(default_factory=dict)
    resource_limits: Optional[ResourceSpec] = None
    
    # 元数据
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 时间信息
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.name:
            self.name = self.cluster_id
    
    def add_agent(self, agent: AgentInfo) -> bool:
        """添加智能体到集群"""
        try:
            agent.cluster_id = self.cluster_id
            self.agents[agent.agent_id] = agent
            self.updated_at = time.time()
            return True
        except Exception:
            return False
    
    def remove_agent(self, agent_id: str) -> bool:
        """从集群移除智能体"""
        if agent_id in self.agents:
            # 从所有分组中移除
            for group in self.groups.values():
                group.remove_agent(agent_id)
            
            # 清理依赖关系
            self.agent_dependencies.pop(agent_id, None)
            for deps in self.agent_dependencies.values():
                deps.discard(agent_id)
            
            # 清理通信路径
            self.communication_paths.pop(agent_id, None)
            for paths in self.communication_paths.values():
                paths.pop(agent_id, None)
            
            # 移除智能体
            del self.agents[agent_id]
            self.updated_at = time.time()
            return True
        
        return False
    
    def get_agent(self, agent_id: str) -> Optional[AgentInfo]:
        """获取智能体信息"""
        return self.agents.get(agent_id)
    
    def add_group(self, group: AgentGroup) -> bool:
        """添加智能体分组"""
        try:
            self.groups[group.group_id] = group
            self.updated_at = time.time()
            return True
        except Exception:
            return False
    
    def get_agents_by_status(self, status: AgentStatus) -> List[AgentInfo]:
        """根据状态获取智能体列表"""
        return [agent for agent in self.agents.values() if agent.status == status]
    
    def get_agents_by_capability(self, capability: AgentCapability) -> List[AgentInfo]:
        """根据能力获取智能体列表"""
        return [agent for agent in self.agents.values() if agent.has_capability(capability)]
    
    def get_healthy_agents(self) -> List[AgentInfo]:
        """获取健康的智能体列表"""
        return [agent for agent in self.agents.values() if agent.is_healthy]
    
    def get_group_agents(self, group_id: str) -> List[AgentInfo]:
        """获取分组内的智能体列表"""
        group = self.groups.get(group_id)
        if not group:
            return []
        
        return [
            agent for agent_id, agent in self.agents.items()
            if agent_id in group.agent_ids
        ]
    
    def add_dependency(self, agent_id: str, depends_on: str):
        """添加智能体依赖关系"""
        if agent_id not in self.agent_dependencies:
            self.agent_dependencies[agent_id] = set()
        self.agent_dependencies[agent_id].add(depends_on)
        self.updated_at = time.time()
    
    def get_dependencies(self, agent_id: str) -> Set[str]:
        """获取智能体依赖"""
        return self.agent_dependencies.get(agent_id, set())
    
    def set_communication_latency(self, from_agent: str, to_agent: str, latency_ms: float):
        """设置智能体间通信延迟"""
        if from_agent not in self.communication_paths:
            self.communication_paths[from_agent] = {}
        self.communication_paths[from_agent][to_agent] = latency_ms
        self.updated_at = time.time()
    
    def get_communication_latency(self, from_agent: str, to_agent: str) -> Optional[float]:
        """获取智能体间通信延迟"""
        return self.communication_paths.get(from_agent, {}).get(to_agent)
    
    @property
    def total_agents(self) -> int:
        """智能体总数"""
        return len(self.agents)
    
    @property
    def running_agents(self) -> int:
        """运行中智能体数量"""
        return len(self.get_agents_by_status(AgentStatus.RUNNING))
    
    @property
    def healthy_agents(self) -> int:
        """健康智能体数量"""
        return len(self.get_healthy_agents())
    
    @property
    def cluster_resource_usage(self) -> ResourceUsage:
        """集群总资源使用情况"""
        total_usage = ResourceUsage()
        healthy_agents = self.get_healthy_agents()
        
        if not healthy_agents:
            return total_usage
        
        # 聚合所有健康智能体的资源使用情况
        for agent in healthy_agents:
            usage = agent.resource_usage
            total_usage.cpu_usage_percent += usage.cpu_usage_percent
            total_usage.memory_usage_percent += usage.memory_usage_percent
            total_usage.storage_usage_percent += usage.storage_usage_percent
            total_usage.gpu_usage_percent += usage.gpu_usage_percent
            total_usage.network_io_mbps += usage.network_io_mbps
            total_usage.active_tasks += usage.active_tasks
            total_usage.total_requests += usage.total_requests
            total_usage.failed_requests += usage.failed_requests
        
        # 计算平均值
        agent_count = len(healthy_agents)
        if agent_count > 0:
            total_usage.cpu_usage_percent /= agent_count
            total_usage.memory_usage_percent /= agent_count
            total_usage.storage_usage_percent /= agent_count
            total_usage.gpu_usage_percent /= agent_count
            
            # 计算加权平均响应时间
            total_weight = 0
            weighted_response_time = 0
            for agent in healthy_agents:
                if agent.resource_usage.total_requests > 0:
                    weight = agent.resource_usage.total_requests
                    weighted_response_time += agent.resource_usage.avg_response_time * weight
                    total_weight += weight
            
            if total_weight > 0:
                total_usage.avg_response_time = weighted_response_time / total_weight
        
        total_usage.timestamp = time.time()
        return total_usage
    
    @property
    def cluster_health_score(self) -> float:
        """集群健康评分 (0-1)"""
        if not self.agents:
            return 0.0
        
        healthy_count = self.healthy_agents
        total_count = self.total_agents
        
        # 基础健康评分
        base_score = healthy_count / total_count
        
        # 资源使用评分 (资源使用率适中得分更高)
        cluster_usage = self.cluster_resource_usage
        resource_score = 1.0
        
        # CPU使用率在30-70%之间最佳
        if cluster_usage.cpu_usage_percent > 0:
            if 30 <= cluster_usage.cpu_usage_percent <= 70:
                cpu_score = 1.0
            elif cluster_usage.cpu_usage_percent < 30:
                cpu_score = cluster_usage.cpu_usage_percent / 30
            else:  # > 70%
                cpu_score = max(0, (100 - cluster_usage.cpu_usage_percent) / 30)
            resource_score *= cpu_score
        
        # 错误率评分
        error_rate = cluster_usage.error_rate
        error_score = max(0, 1 - error_rate * 10)  # 错误率每增加10%，评分降低100%
        
        # 综合评分
        final_score = (base_score * 0.5 + resource_score * 0.3 + error_score * 0.2)
        return max(0.0, min(1.0, final_score))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "cluster_id": self.cluster_id,
            "name": self.name,
            "description": self.description,
            "total_agents": self.total_agents,
            "running_agents": self.running_agents,
            "healthy_agents": self.healthy_agents,
            "health_score": self.cluster_health_score,
            "agents": {
                agent_id: {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "status": agent.status.value,
                    "endpoint": agent.endpoint,
                    "capabilities": [cap.value for cap in agent.capabilities],
                    "is_healthy": agent.is_healthy,
                    "uptime": agent.uptime_seconds,
                    "resource_usage": {
                        "cpu_usage": agent.resource_usage.cpu_usage_percent,
                        "memory_usage": agent.resource_usage.memory_usage_percent,
                        "active_tasks": agent.resource_usage.active_tasks,
                        "error_rate": agent.resource_usage.error_rate
                    }
                }
                for agent_id, agent in self.agents.items()
            },
            "groups": {
                group_id: {
                    "group_id": group.group_id,
                    "name": group.name,
                    "agent_count": group.agent_count,
                    "agent_ids": list(group.agent_ids)
                }
                for group_id, group in self.groups.items()
            },
            "cluster_resource_usage": {
                "cpu_usage": self.cluster_resource_usage.cpu_usage_percent,
                "memory_usage": self.cluster_resource_usage.memory_usage_percent,
                "active_tasks": self.cluster_resource_usage.active_tasks,
                "total_requests": self.cluster_resource_usage.total_requests,
                "error_rate": self.cluster_resource_usage.error_rate,
                "avg_response_time": self.cluster_resource_usage.avg_response_time
            },
            "updated_at": self.updated_at
        }
