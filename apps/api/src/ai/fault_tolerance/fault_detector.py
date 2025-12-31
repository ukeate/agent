from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
import asyncio
import json
import hashlib
import time
from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
import redis.asyncio as redis
from abc import ABC, abstractmethod
import aiohttp

from src.core.logging import get_logger
class FaultType(Enum):
    """故障类型枚举"""
    AGENT_UNRESPONSIVE = "agent_unresponsive"
    AGENT_ERROR = "agent_error"
    NETWORK_PARTITION = "network_partition"
    NODE_FAILURE = "node_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_CORRUPTION = "data_corruption"
    RESOURCE_EXHAUSTION = "resource_exhaustion"

class FaultSeverity(Enum):
    """故障严重程度"""
    CRITICAL = "critical"  # 影响核心功能
    HIGH = "high"         # 影响重要功能
    MEDIUM = "medium"     # 影响一般功能
    LOW = "low"          # 轻微影响

@dataclass
class FaultEvent:
    """故障事件"""
    fault_id: str
    fault_type: FaultType
    severity: FaultSeverity
    affected_components: List[str]
    detected_at: datetime
    description: str
    context: Dict[str, Any]
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    recovery_actions: List[str] = field(default_factory=list)

@dataclass
class HealthStatus:
    """健康状态"""
    component_id: str
    status: str  # healthy, degraded, unhealthy, unknown
    last_check: datetime
    response_time: float
    error_rate: float
    resource_usage: Dict[str, float]
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

class FaultDetector:
    """故障检测器"""
    
    def __init__(self, cluster_manager, metrics_collector, config: Dict[str, Any]):
        self.cluster_manager = cluster_manager
        self.metrics_collector = metrics_collector
        self.config = config
        self.logger = get_logger(__name__)
        
        # 故障检测配置
        self.health_check_interval = config.get("health_check_interval", 10)
        self.response_timeout = config.get("response_timeout", 5.0)
        self.error_rate_threshold = config.get("error_rate_threshold", 0.1)
        self.performance_threshold = config.get("performance_threshold", 2.0)
        
        # 故障状态跟踪
        self.health_status: Dict[str, HealthStatus] = {}
        self.fault_events: List[FaultEvent] = []
        self.detection_history: Dict[str, List[datetime]] = {}
        
        # 检测器回调
        self.fault_callbacks: List[Callable[[FaultEvent], None]] = []
        
        # 运行控制
        self.running = False
    
    async def start(self):
        """启动故障检测"""
        self.running = True
        
        # 启动各种检测任务
        create_task_with_logging(self._health_check_loop())
        create_task_with_logging(self._performance_monitor_loop())
        create_task_with_logging(self._network_monitor_loop())
        
        self.logger.info("Fault detector started")
    
    async def stop(self):
        """停止故障检测"""
        self.running = False
        self.logger.info("Fault detector stopped")
    
    def register_fault_callback(self, callback: Callable[[FaultEvent], None]):
        """注册故障事件回调"""
        self.fault_callbacks.append(callback)
    
    async def check_component_health(self, component_id: str) -> HealthStatus:
        """检查单个组件健康状态"""
        
        try:
            start_time = time.time()
            
            # 获取组件信息
            component_info = await self.cluster_manager.get_agent_info(component_id)
            if not component_info:
                return HealthStatus(
                    component_id=component_id,
                    status="unknown",
                    last_check=utc_now(),
                    response_time=0.0,
                    error_rate=1.0,
                    resource_usage={}
                )
            
            # 执行健康检查
            health_result = await self._perform_health_check(component_info.endpoint)
            response_time = time.time() - start_time
            
            usage = getattr(component_info, "resource_usage", None)
            error_rate = float(getattr(usage, "error_rate", 0.0)) if usage else 0.0
            resource_usage = {
                "cpu": float(getattr(usage, "cpu_usage_percent", 0.0)),
                "memory": float(getattr(usage, "memory_usage_percent", 0.0)),
                "disk": float(getattr(usage, "storage_usage_percent", 0.0)),
            } if usage else {}
            
            # 确定健康状态
            status = self._determine_health_status(
                health_result, 
                response_time, 
                error_rate,
                resource_usage
            )
            
            health_status = HealthStatus(
                component_id=component_id,
                status=status,
                last_check=utc_now(),
                response_time=response_time,
                error_rate=error_rate,
                resource_usage=resource_usage,
                custom_metrics=health_result.get("custom_metrics", {})
            )
            
            # 更新状态记录
            self.health_status[component_id] = health_status
            
            # 检测故障
            await self._detect_faults_from_health(health_status)
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check failed for {component_id}: {e}")
            
            # 创建故障健康状态
            unhealthy_status = HealthStatus(
                component_id=component_id,
                status="unhealthy",
                last_check=utc_now(),
                response_time=float('inf'),
                error_rate=1.0,
                resource_usage={}
            )
            
            self.health_status[component_id] = unhealthy_status
            return unhealthy_status
    
    async def _perform_health_check(self, endpoint: str) -> Dict[str, Any]:
        """执行健康检查"""
        
        try:
            response_data = await self._make_http_health_request(endpoint)
            return response_data
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}
    
    async def _make_http_health_request(self, endpoint: str) -> Dict[str, Any]:
        """发送HTTP健康检查请求"""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.response_timeout)
        ) as session:
            async with session.get(f"{endpoint}/health") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"status": "unhealthy", "code": response.status}
    
    def _determine_health_status(
        self, 
        health_result: Dict[str, Any],
        response_time: float,
        error_rate: float,
        resource_usage: Dict[str, float]
    ) -> str:
        """确定健康状态"""
        
        # 检查基本健康状态
        if health_result.get("status") == "unhealthy":
            return "unhealthy"
        elif health_result.get("status") == "unreachable":
            return "unhealthy"
        
        # 检查响应时间
        if response_time > self.performance_threshold:
            return "degraded"
        
        # 检查错误率
        if error_rate > self.error_rate_threshold:
            return "degraded"
        
        # 检查资源使用情况
        cpu_usage = resource_usage.get("cpu", 0)
        memory_usage = resource_usage.get("memory", 0)
        
        if cpu_usage > 90 or memory_usage > 90:
            return "degraded"
        
        return "healthy"
    
    async def _detect_faults_from_health(self, health_status: HealthStatus):
        """从健康状态检测故障"""
        
        component_id = health_status.component_id
        
        # 检测无响应故障
        if health_status.status == "unhealthy":
            if health_status.response_time == float('inf'):
                await self._create_fault_event(
                    FaultType.AGENT_UNRESPONSIVE,
                    FaultSeverity.HIGH,
                    [component_id],
                    f"Agent {component_id} is unresponsive",
                    {"health_status": asdict(health_status)}
                )
            else:
                await self._create_fault_event(
                    FaultType.AGENT_ERROR,
                    FaultSeverity.MEDIUM,
                    [component_id],
                    f"Agent {component_id} is unhealthy",
                    {"health_status": asdict(health_status)}
                )
        
        # 检测性能降级
        elif health_status.status == "degraded":
            await self._create_fault_event(
                FaultType.PERFORMANCE_DEGRADATION,
                FaultSeverity.LOW,
                [component_id],
                f"Agent {component_id} performance degraded",
                {"health_status": asdict(health_status)}
            )
        
        # 检测资源耗尽
        cpu_usage = health_status.resource_usage.get("cpu", 0)
        memory_usage = health_status.resource_usage.get("memory", 0)
        
        if cpu_usage > 95 or memory_usage > 95:
            await self._create_fault_event(
                FaultType.RESOURCE_EXHAUSTION,
                FaultSeverity.HIGH,
                [component_id],
                f"Agent {component_id} resource exhaustion",
                {"health_status": asdict(health_status)}
            )
    
    async def _create_fault_event(
        self,
        fault_type: FaultType,
        severity: FaultSeverity,
        affected_components: List[str],
        description: str,
        context: Dict[str, Any]
    ):
        """创建故障事件"""
        
        fault_event = FaultEvent(
            fault_id=f"{fault_type.value}_{int(time.time())}_{hash(tuple(affected_components)) % 10000}",
            fault_type=fault_type,
            severity=severity,
            affected_components=affected_components,
            detected_at=utc_now(),
            description=description,
            context=context
        )
        
        # 记录故障事件
        self.fault_events.append(fault_event)
        
        # 限制故障事件历史大小
        if len(self.fault_events) > 10000:
            self.fault_events = self.fault_events[-5000:]
        
        # 通知回调函数
        for callback in self.fault_callbacks:
            try:
                await callback(fault_event)
            except Exception as e:
                self.logger.error(f"Fault callback error: {e}")
        
        self.logger.warning(f"Fault detected: {fault_event.fault_id} - {description}")
    
    async def _health_check_loop(self):
        """健康检查循环"""
        
        while self.running:
            try:
                # 获取所有智能体
                topology = await self.cluster_manager.get_cluster_topology()
                
                if topology and hasattr(topology, 'agents'):
                    # 并发检查所有智能体健康状态
                    check_tasks = []
                    for agent_id in topology.agents.keys():
                        task = self.check_component_health(agent_id)
                        check_tasks.append(task)
                    
                    if check_tasks:
                        await asyncio.gather(*check_tasks, return_exceptions=True)
                
                await asyncio.sleep(self.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(self.health_check_interval)
    
    async def _performance_monitor_loop(self):
        """性能监控循环"""
        
        monitor_interval = 30  # 30秒间隔
        
        while self.running:
            try:
                # 检查系统整体性能
                await self._check_system_performance()
                await asyncio.sleep(monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor loop: {e}")
                await asyncio.sleep(monitor_interval)
    
    async def _network_monitor_loop(self):
        """网络监控循环"""
        
        monitor_interval = 20  # 20秒间隔
        
        while self.running:
            try:
                # 检查网络分区
                await self._check_network_partitions()
                await asyncio.sleep(monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Error in network monitor loop: {e}")
                await asyncio.sleep(monitor_interval)
    
    async def _check_system_performance(self):
        """检查系统整体性能"""
        
        # 获取所有健康状态
        healthy_agents = 0
        total_response_time = 0.0
        total_error_rate = 0.0
        
        for health_status in self.health_status.values():
            if health_status.status == "healthy":
                healthy_agents += 1
            total_response_time += health_status.response_time
            total_error_rate += health_status.error_rate
        
        total_agents = len(self.health_status)
        
        if total_agents > 0:
            avg_response_time = total_response_time / total_agents
            avg_error_rate = total_error_rate / total_agents
            healthy_ratio = healthy_agents / total_agents
            
            # 检查系统级性能问题
            if healthy_ratio < 0.8:  # 健康智能体比例低于80%
                await self._create_fault_event(
                    FaultType.NODE_FAILURE,
                    FaultSeverity.CRITICAL,
                    ["system"],
                    f"System health degraded: only {healthy_ratio:.1%} agents healthy",
                    {
                        "healthy_agents": healthy_agents,
                        "total_agents": total_agents,
                        "avg_response_time": avg_response_time,
                        "avg_error_rate": avg_error_rate
                    }
                )
    
    async def _check_network_partitions(self):
        """检查网络分区"""
        
        # 简化实现：检查智能体间的连通性
        topology = await self.cluster_manager.get_cluster_topology()
        
        if not topology or len(topology.agents) < 2:
            return
        
        # 随机选择一些智能体进行连通性测试
        agent_ids = list(topology.agents.keys())
        sample_size = min(5, len(agent_ids))
        sample_agents = agent_ids[:sample_size]
        
        connectivity_matrix = {}
        
        for agent_id in sample_agents:
            connectivity_matrix[agent_id] = {}
            
            agent_info = topology.agents[agent_id]
            
            # 测试到其他智能体的连通性
            for other_agent_id in sample_agents:
                if agent_id != other_agent_id:
                    other_agent_info = topology.agents[other_agent_id]
                    
                    # 简化的连通性测试
                    is_connected = await self._test_connectivity(
                        agent_info.endpoint,
                        other_agent_info.endpoint
                    )
                    
                    connectivity_matrix[agent_id][other_agent_id] = is_connected
        
        # 分析网络分区
        partitions = self._analyze_network_partitions(connectivity_matrix)
        
        if len(partitions) > 1:
            await self._create_fault_event(
                FaultType.NETWORK_PARTITION,
                FaultSeverity.CRITICAL,
                ["network"],
                f"Network partition detected: {len(partitions)} partitions found",
                {"partitions": partitions}
            )
    
    async def _test_connectivity(self, endpoint1: str, endpoint2: str) -> bool:
        """测试两个端点之间的连通性"""
        
        # 简化实现：通过HTTP请求测试连通性
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=3.0)
            ) as session:
                # 让endpoint1测试到endpoint2的连通性
                async with session.get(f"{endpoint1}/test_connectivity?target={endpoint2}") as response:
                    return response.status == 200
        except Exception:
            return False
    
    def _analyze_network_partitions(self, connectivity_matrix: Dict[str, Dict[str, bool]]) -> List[List[str]]:
        """分析网络分区"""
        
        agents = list(connectivity_matrix.keys())
        visited = set()
        partitions = []
        
        def dfs(agent, current_partition):
            if agent in visited:
                return
            visited.add(agent)
            current_partition.append(agent)
            
            # 访问所有连通的智能体
            for other_agent, is_connected in connectivity_matrix.get(agent, {}).items():
                if is_connected and other_agent not in visited:
                    dfs(other_agent, current_partition)
        
        for agent in agents:
            if agent not in visited:
                partition = []
                dfs(agent, partition)
                if partition:
                    partitions.append(partition)
        
        return partitions
    
    def get_fault_events(
        self, 
        fault_type: Optional[FaultType] = None,
        severity: Optional[FaultSeverity] = None,
        resolved: Optional[bool] = None,
        limit: int = 100
    ) -> List[FaultEvent]:
        """获取故障事件"""
        
        filtered_events = self.fault_events
        
        if fault_type:
            filtered_events = [e for e in filtered_events if e.fault_type == fault_type]
        
        if severity:
            filtered_events = [e for e in filtered_events if e.severity == severity]
        
        if resolved is not None:
            filtered_events = [e for e in filtered_events if e.resolved == resolved]
        
        # 按时间倒序排列，返回最近的事件
        filtered_events.sort(key=lambda e: e.detected_at, reverse=True)
        
        return filtered_events[:limit]
    
    def get_system_health_summary(self) -> Dict[str, Any]:
        """获取系统健康状态摘要"""
        
        status_counts = {"healthy": 0, "degraded": 0, "unhealthy": 0, "unknown": 0}
        total_response_time = 0.0
        total_error_rate = 0.0
        
        for health_status in self.health_status.values():
            status_counts[health_status.status] = status_counts.get(health_status.status, 0) + 1
            total_response_time += health_status.response_time
            total_error_rate += health_status.error_rate
        
        total_components = len(self.health_status)
        
        return {
            "total_components": total_components,
            "status_counts": status_counts,
            "avg_response_time": total_response_time / max(total_components, 1),
            "avg_error_rate": total_error_rate / max(total_components, 1),
            "health_ratio": status_counts["healthy"] / max(total_components, 1),
            "active_faults": len([e for e in self.fault_events if not e.resolved]),
            "last_update": utc_now().isoformat()
        }
