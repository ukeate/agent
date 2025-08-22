"""
AutoGen企业级架构管理器
基于现有AsyncAgentManager实现企业级异步事件驱动架构增强
包含负载均衡、池化管理、分布式事件处理、安全框架和监控系统
"""
import asyncio
import json
import time
import uuid
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import structlog
import redis.asyncio as redis

from .events import Event, EventType, EventPriority, EventHandler, EventBus
from .async_manager import AsyncAgentManager, AgentTask, AgentInfo, AgentStatus, TaskStatus
from .config import AgentConfig, AgentRole
from .distributed_events import DistributedEventBus, DistributedEvent
from .monitoring import EnterpriseMonitoringManager as EnterpriseMonitoring
from .error_recovery import ErrorRecoveryService as ErrorRecoveryManager
from .enterprise_config import get_config_manager, ConfigCategory, ConfigLevel
from .flow_control import FlowController, BackpressureStrategy, get_flow_controller
# from .backpressure_task_processor import BackpressureTaskProcessor  # 避免循环导入
from .structured_errors import (
    ErrorFactory, ErrorContext, StructuredException, handle_structured_error,
    ErrorCodes, ErrorCategory, ErrorSeverity
)
from .monitoring_dashboard import get_metric_collector, MetricType

logger = structlog.get_logger(__name__)


class LoadBalancingStrategy(str, Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    WEIGHTED = "weighted"
    RESOURCE_BASED = "resource_based"


class PoolingStrategy(str, Enum):
    """池化策略"""
    FIXED_SIZE = "fixed_size"
    DYNAMIC = "dynamic"
    AUTO_SCALING = "auto_scaling"


@dataclass
class AgentPoolConfig:
    """智能体池配置"""
    min_size: int = field(default_factory=lambda: get_config_manager().get_int('AGENT_POOL_MIN_SIZE', 1))
    max_size: int = field(default_factory=lambda: get_config_manager().get_int('AGENT_POOL_MAX_SIZE', 10))
    initial_size: int = field(default_factory=lambda: get_config_manager().get_int('AGENT_POOL_INITIAL_SIZE', 3))
    idle_timeout: int = field(default_factory=lambda: get_config_manager().get_int('AGENT_POOL_IDLE_TIMEOUT', 300))
    scaling_threshold: float = field(default_factory=lambda: get_config_manager().get_float('AGENT_POOL_SCALING_THRESHOLD', 0.8))
    scaling_factor: float = 1.5
    pooling_strategy: PoolingStrategy = PoolingStrategy.DYNAMIC
    
    @classmethod
    def from_config(cls) -> 'AgentPoolConfig':
        """从配置管理器创建配置实例"""
        config_manager = get_config_manager()
        return cls(
            min_size=config_manager.get_int('AGENT_POOL_MIN_SIZE', 1),
            max_size=config_manager.get_int('AGENT_POOL_MAX_SIZE', 10), 
            initial_size=config_manager.get_int('AGENT_POOL_INITIAL_SIZE', 3),
            idle_timeout=config_manager.get_int('AGENT_POOL_IDLE_TIMEOUT', 300),
            scaling_threshold=config_manager.get_float('AGENT_POOL_SCALING_THRESHOLD', 0.8),
            scaling_factor=config_manager.get_float('AGENT_POOL_SCALING_FACTOR', 1.5),
            pooling_strategy=PoolingStrategy(config_manager.get('AGENT_POOL_STRATEGY', 'dynamic'))
        )


@dataclass
class EnterpriseAgentInfo(AgentInfo):
    """企业级智能体信息"""
    pool_id: Optional[str] = None
    resource_usage: Dict[str, float] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    health_score: float = 1.0
    last_health_check: Optional[datetime] = None
    failover_count: int = 0
    security_clearance: str = "standard"
    
    def calculate_load_score(self) -> float:
        """计算负载评分"""
        if self.status == AgentStatus.IDLE:
            return 0.0
        elif self.status == AgentStatus.BUSY:
            return 1.0
        elif self.status == AgentStatus.ERROR:
            return float('inf')
        return 0.5


@dataclass
class AgentPool:
    """智能体池"""
    id: str
    name: str
    role: AgentRole
    config: AgentPoolConfig
    agents: Dict[str, EnterpriseAgentInfo] = field(default_factory=dict)
    load_balancer_index: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_scaled: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_current_load(self) -> float:
        """获取当前池负载"""
        if not self.agents:
            return 0.0
        
        total_load = sum(agent.calculate_load_score() for agent in self.agents.values())
        return total_load / len(self.agents)
    
    def get_available_agents(self) -> List[EnterpriseAgentInfo]:
        """获取可用智能体"""
        return [
            agent for agent in self.agents.values()
            if agent.status == AgentStatus.IDLE and agent.health_score > 0.5
        ]
    
    def needs_scaling_up(self) -> bool:
        """是否需要扩容"""
        if len(self.agents) >= self.config.max_size:
            return False
        
        load = self.get_current_load()
        return load > self.config.scaling_threshold
    
    def needs_scaling_down(self) -> bool:
        """是否需要缩容"""
        if len(self.agents) <= self.config.min_size:
            return False
        
        # 检查空闲智能体数量
        idle_agents = len(self.get_available_agents())
        return idle_agents > (len(self.agents) * 0.5)


class EnterpriseAgentManager(AsyncAgentManager):
    """企业级智能体管理器"""
    
    def __init__(
        self,
        event_bus: EventBus,
        message_queue,
        state_manager,
        redis_client: redis.Redis,
        max_concurrent_tasks: int = 50,
        node_id: Optional[str] = None
    ):
        super().__init__(event_bus, message_queue, state_manager, max_concurrent_tasks)
        
        self.redis_client = redis_client
        self.node_id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        
        # 企业级组件
        self.distributed_event_bus = DistributedEventBus(redis_client, self.node_id)
        self.monitoring = EnterpriseMonitoring()
        self.error_recovery = ErrorRecoveryManager(self)
        self.flow_controller = FlowController(strategy=BackpressureStrategy.ADAPTIVE)
        self.task_processor = BackpressureTaskProcessor(self.flow_controller, self)
        self.metric_collector = get_metric_collector()
        
        # 智能体池管理
        self.agent_pools: Dict[str, AgentPool] = {}
        self.pool_configs: Dict[AgentRole, AgentPoolConfig] = {}
        
        # 负载均衡
        self.load_balancing_strategy = LoadBalancingStrategy.LEAST_LOADED
        self.load_balancer_lock = threading.Lock()
        
        # 线程池用于CPU密集型任务
        self.thread_pool = ThreadPoolExecutor(max_workers=10)
        
        # 健康检查
        self._health_check_task: Optional[asyncio.Task] = None
        self._pool_scaling_task: Optional[asyncio.Task] = None
        
        # 故障转移
        self.failover_nodes: Set[str] = set()
        self.primary_node = True
        
        logger.info(
            "企业级智能体管理器初始化完成",
            node_id=self.node_id,
            max_concurrent_tasks=max_concurrent_tasks
        )
    
    async def start(self) -> None:
        """启动企业级管理器"""
        await super().start()
        
        # 启动分布式事件总线
        await self.distributed_event_bus.start()
        
        # 启动企业级监控
        await self.monitoring.start()
        
        # 启动错误恢复管理器
        await self.error_recovery.start()
        
        # 启动流控器
        await self.flow_controller.start()
        
        # 启动背压任务处理器
        await self.task_processor.start()
        
        # 启动指标收集任务
        self._metrics_collection_task = asyncio.create_task(self._collect_enterprise_metrics())
        
        # 启动健康检查
        self._health_check_task = asyncio.create_task(self._enterprise_health_monitor())
        
        # 启动池扩缩容管理
        self._pool_scaling_task = asyncio.create_task(self._pool_scaling_monitor())
        
        # 注册分布式事件处理器
        await self._register_distributed_handlers()
        
        logger.info("企业级智能体管理器启动完成")
    
    async def stop(self) -> None:
        """停止企业级管理器"""
        # 停止企业级任务
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._pool_scaling_task:
            self._pool_scaling_task.cancel()
        
        # 停止企业级组件
        await self.error_recovery.stop()
        await self.monitoring.stop()
        await self.distributed_event_bus.stop()
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        
        await super().stop()
        
        logger.info("企业级智能体管理器停止完成")
    
    async def create_agent_pool(
        self,
        role: AgentRole,
        pool_config: AgentPoolConfig,
        pool_name: Optional[str] = None
    ) -> str:
        """创建智能体池"""
        pool_id = f"pool_{role.value}_{uuid.uuid4().hex[:8]}"
        pool_name = pool_name or f"{role.value}_pool"
        
        pool = AgentPool(
            id=pool_id,
            name=pool_name,
            role=role,
            config=pool_config
        )
        
        self.agent_pools[pool_id] = pool
        self.pool_configs[role] = pool_config
        
        # 创建初始智能体
        for i in range(pool_config.initial_size):
            agent_config = AgentConfig(
                name=f"{pool_name}_agent_{i}",
                role=role,
                capabilities=["standard"]
            )
            await self._create_pooled_agent(pool_id, agent_config)
        
        logger.info(
            "智能体池创建成功",
            pool_id=pool_id,
            role=role,
            initial_size=pool_config.initial_size
        )
        
        return pool_id
    
    async def _create_pooled_agent(
        self,
        pool_id: str,
        config: AgentConfig
    ) -> str:
        """创建池化智能体"""
        agent_id = await self.create_agent(config)
        
        # 转换为企业级智能体信息
        base_info = self.agents[agent_id]
        enterprise_info = EnterpriseAgentInfo(
            id=base_info.id,
            name=base_info.name,
            role=base_info.role,
            status=base_info.status,
            agent=base_info.agent,
            created_at=base_info.created_at,
            last_activity=base_info.last_activity,
            pool_id=pool_id,
            last_health_check=datetime.now(timezone.utc)
        )
        
        # 替换智能体信息
        self.agents[agent_id] = enterprise_info
        
        # 添加到池
        pool = self.agent_pools[pool_id]
        pool.agents[agent_id] = enterprise_info
        
        return agent_id
    
    async def submit_task_to_pool(
        self,
        pool_id: str,
        task_type: str,
        description: str,
        input_data: Dict[str, Any],
        priority: int = 0,
        timeout_seconds: int = 300
    ) -> str:
        """提交任务到智能体池"""
        if pool_id not in self.agent_pools:
            context = ErrorContext(
                node_id=self.node_id,
                operation="submit_task_to_pool",
                component="EnterpriseAgentManager"
            )
            raise ErrorFactory.create_resource_not_found_error(
                "AgentPool", pool_id, context
            )
        
        # 流控检查 - 先提交到流控队列
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        task_data = {
            'pool_id': pool_id,
            'task_type': task_type,
            'description': description,
            'input_data': input_data,
            'timeout_seconds': timeout_seconds
        }
        
        # 通过流控器提交任务
        flow_accepted = await self.flow_controller.submit_task(
            task_id=task_id,
            task_data=task_data,
            priority=priority,
            deadline=datetime.now() + timedelta(seconds=timeout_seconds) if timeout_seconds > 0 else None
        )
        
        if not flow_accepted:
            context = ErrorContext(
                node_id=self.node_id,
                task_id=task_id,
                operation="submit_task_to_pool",
                component="EnterpriseAgentManager"
            )
            raise ErrorFactory.create_rate_limit_error(
                current_rate=self.flow_controller.current_metrics.throughput,
                rate_limit=self.flow_controller.max_queue_size,
                reset_time="系统负载降低后",
                context=context
            )
        
        # 选择智能体
        agent_id = await self._select_agent_from_pool(pool_id)
        if not agent_id:
            # 任务被拒绝，需要从流控队列中移除
            await self.flow_controller.complete_task(task_id, success=False)
            context = ErrorContext(
                node_id=self.node_id,
                task_id=task_id,
                operation="submit_task_to_pool",
                component="EnterpriseAgentManager",
                additional_data={"pool_id": pool_id}
            )
            raise ErrorFactory.create_system_error(
                f"池 {pool_id} 中没有可用的智能体",
                details={"pool_id": pool_id, "available_agents": len(self.agent_pools[pool_id].get_available_agents())},
                context=context
            )
        
        # 提交任务
        task_id = await self.submit_task(
            agent_id=agent_id,
            task_type=task_type,
            description=description,
            input_data=input_data,
            priority=priority,
            timeout_seconds=timeout_seconds
        )
        
        # 记录监控指标
        self.monitoring.record_task_submission(pool_id, task_type)
        
        return task_id
    
    async def _select_agent_from_pool(self, pool_id: str) -> Optional[str]:
        """从池中选择智能体"""
        pool = self.agent_pools[pool_id]
        available_agents = pool.get_available_agents()
        
        if not available_agents:
            # 尝试扩容
            if pool.needs_scaling_up():
                await self._scale_pool_up(pool_id)
                available_agents = pool.get_available_agents()
        
        if not available_agents:
            return None
        
        # 根据负载均衡策略选择
        if self.load_balancing_strategy == LoadBalancingStrategy.ROUND_ROBIN:
            with self.load_balancer_lock:
                index = pool.load_balancer_index % len(available_agents)
                pool.load_balancer_index += 1
                return available_agents[index].id
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.LEAST_LOADED:
            # 选择负载最低的智能体
            best_agent = min(available_agents, key=lambda a: a.calculate_load_score())
            return best_agent.id
        
        elif self.load_balancing_strategy == LoadBalancingStrategy.RESOURCE_BASED:
            # 基于资源使用选择
            best_agent = min(
                available_agents,
                key=lambda a: a.resource_usage.get('cpu', 0) + a.resource_usage.get('memory', 0)
            )
            return best_agent.id
        
        # 默认返回第一个
        return available_agents[0].id
    
    async def _scale_pool_up(self, pool_id: str) -> bool:
        """扩容智能体池"""
        pool = self.agent_pools[pool_id]
        
        if len(pool.agents) >= pool.config.max_size:
            return False
        
        # 计算扩容数量
        current_size = len(pool.agents)
        target_size = min(
            pool.config.max_size,
            int(current_size * pool.config.scaling_factor)
        )
        
        agents_to_add = target_size - current_size
        
        try:
            for i in range(agents_to_add):
                agent_config = AgentConfig(
                    name=f"{pool.name}_agent_{current_size + i}",
                    role=pool.role,
                    capabilities=["standard"]
                )
                await self._create_pooled_agent(pool_id, agent_config)
            
            pool.last_scaled = datetime.now(timezone.utc)
            
            logger.info(
                "智能体池扩容成功",
                pool_id=pool_id,
                from_size=current_size,
                to_size=len(pool.agents)
            )
            
            return True
            
        except Exception as e:
            logger.error("智能体池扩容失败", pool_id=pool_id, error=str(e))
            return False
    
    async def _scale_pool_down(self, pool_id: str) -> bool:
        """缩容智能体池"""
        pool = self.agent_pools[pool_id]
        
        if len(pool.agents) <= pool.config.min_size:
            return False
        
        # 找到空闲时间最长的智能体
        idle_agents = [
            agent for agent in pool.agents.values()
            if agent.status == AgentStatus.IDLE
        ]
        
        if not idle_agents:
            return False
        
        # 按空闲时间排序
        idle_agents.sort(
            key=lambda a: a.last_activity,
            reverse=False
        )
        
        # 移除一个智能体
        agent_to_remove = idle_agents[0]
        
        try:
            await self.destroy_agent(agent_to_remove.id)
            del pool.agents[agent_to_remove.id]
            
            pool.last_scaled = datetime.now(timezone.utc)
            
            logger.info(
                "智能体池缩容成功",
                pool_id=pool_id,
                removed_agent=agent_to_remove.id,
                new_size=len(pool.agents)
            )
            
            return True
            
        except Exception as e:
            logger.error("智能体池缩容失败", pool_id=pool_id, error=str(e))
            return False
    
    async def _register_distributed_handlers(self):
        """注册分布式事件处理器"""
        await self.distributed_event_bus.subscribe(
            "node_health_check",
            self._handle_node_health_check
        )
        
        await self.distributed_event_bus.subscribe(
            "failover_request",
            self._handle_failover_request
        )
        
        await self.distributed_event_bus.subscribe(
            "load_balancing_update",
            self._handle_load_balancing_update
        )
    
    async def _handle_node_health_check(self, event: DistributedEvent):
        """处理节点健康检查"""
        if event.source_node != self.node_id:
            # 响应健康检查
            response_event = DistributedEvent(
                event_id=str(uuid.uuid4()),
                event_type="node_health_response",
                source_node=self.node_id,
                target_nodes=[event.source_node],
                payload={
                    "status": "healthy",
                    "stats": self.get_manager_stats(),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                timestamp=datetime.now(timezone.utc)
            )
            await self.distributed_event_bus.publish(response_event)
    
    async def _handle_failover_request(self, event: DistributedEvent):
        """处理故障转移请求"""
        if not self.primary_node:
            return
        
        failed_node = event.payload.get("failed_node")
        if failed_node:
            self.failover_nodes.add(failed_node)
            
            # 重新分配失败节点的任务
            await self._reassign_failed_node_tasks(failed_node)
            
            logger.info("处理故障转移", failed_node=failed_node)
    
    async def _handle_load_balancing_update(self, event: DistributedEvent):
        """处理负载均衡更新"""
        strategy = event.payload.get("strategy")
        if strategy and strategy in LoadBalancingStrategy:
            self.load_balancing_strategy = LoadBalancingStrategy(strategy)
            logger.info("负载均衡策略更新", strategy=strategy)
    
    async def _reassign_failed_node_tasks(self, failed_node: str):
        """重新分配失败节点的任务"""
        # 获取失败节点的任务
        failed_tasks = await self._get_failed_node_tasks(failed_node)
        
        for task_info in failed_tasks:
            try:
                # 重新提交任务
                await self.submit_task(
                    agent_id=task_info["agent_id"],
                    task_type=task_info["task_type"],
                    description=task_info["description"],
                    input_data=task_info["input_data"],
                    priority=task_info["priority"]
                )
                
                logger.info("任务重新分配成功", task_id=task_info["id"])
                
            except Exception as e:
                logger.error(
                    "任务重新分配失败",
                    task_id=task_info["id"],
                    error=str(e)
                )
    
    async def _get_failed_node_tasks(self, failed_node: str) -> List[Dict[str, Any]]:
        """获取失败节点的任务"""
        # 从Redis或数据库获取失败节点的任务信息
        try:
            key = f"node_tasks:{failed_node}"
            tasks_data = await self.redis_client.get(key)
            
            if tasks_data:
                return json.loads(tasks_data)
            
        except Exception as e:
            logger.error("获取失败节点任务失败", failed_node=failed_node, error=str(e))
        
        return []
    
    async def _enterprise_health_monitor(self):
        """企业级健康监控"""
        logger.info("企业级健康监控启动")
        
        while self.running:
            try:
                # 检查所有智能体健康状态
                for agent_id, agent_info in self.agents.items():
                    if isinstance(agent_info, EnterpriseAgentInfo):
                        await self._check_agent_health(agent_info)
                
                # 检查池状态
                for pool_id, pool in self.agent_pools.items():
                    await self._check_pool_health(pool)
                
                # 发布健康状态
                await self._publish_health_status()
                
                # 等待下次检查
                await asyncio.sleep(60)  # 1分钟检查一次
                
            except Exception as e:
                logger.error("企业级健康监控异常", error=str(e))
                await asyncio.sleep(30)
        
        logger.info("企业级健康监控停止")
    
    async def _check_agent_health(self, agent_info: EnterpriseAgentInfo):
        """检查智能体健康状态"""
        current_time = datetime.now(timezone.utc)
        
        # 更新最后健康检查时间
        agent_info.last_health_check = current_time
        
        # 计算健康评分
        health_score = 1.0
        
        # 基于任务成功率
        if agent_info.total_tasks > 0:
            success_rate = agent_info.completed_tasks / agent_info.total_tasks
            health_score *= success_rate
        
        # 基于响应时间
        if agent_info.average_task_time > 0:
            response_factor = min(1.0, 300 / agent_info.average_task_time)  # 5分钟为基准
            health_score *= response_factor
        
        # 基于错误计数
        if agent_info.failover_count > 0:
            error_factor = max(0.1, 1.0 - (agent_info.failover_count * 0.1))
            health_score *= error_factor
        
        agent_info.health_score = max(0.0, min(1.0, health_score))
        
        # 如果健康评分太低，标记为错误状态
        if agent_info.health_score < 0.3 and agent_info.status != AgentStatus.ERROR:
            await self._update_agent_status(agent_info.id, AgentStatus.ERROR)
            await self.error_recovery.handle_agent_failure(agent_info.id)
    
    async def _check_pool_health(self, pool: AgentPool):
        """检查池健康状态"""
        healthy_agents = sum(
            1 for agent in pool.agents.values()
            if agent.health_score > 0.5
        )
        
        if healthy_agents < pool.config.min_size:
            logger.warning(
                "智能体池健康智能体不足",
                pool_id=pool.id,
                healthy_count=healthy_agents,
                min_required=pool.config.min_size
            )
            
            # 尝试恢复
            await self._recover_unhealthy_agents(pool)
    
    async def _recover_unhealthy_agents(self, pool: AgentPool):
        """恢复不健康的智能体"""
        unhealthy_agents = [
            agent for agent in pool.agents.values()
            if agent.health_score < 0.3
        ]
        
        for agent in unhealthy_agents:
            try:
                # 尝试重启智能体
                await self.destroy_agent(agent.id)
                
                # 创建新的智能体替换
                agent_config = AgentConfig(
                    name=f"{pool.name}_agent_{len(pool.agents)}",
                    role=pool.role,
                    capabilities=["standard"]
                )
                await self._create_pooled_agent(pool.id, agent_config)
                
                logger.info("不健康智能体恢复成功", agent_id=agent.id)
                
            except Exception as e:
                logger.error("不健康智能体恢复失败", agent_id=agent.id, error=str(e))
    
    async def _publish_health_status(self):
        """发布健康状态"""
        health_data = {
            "node_id": self.node_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stats": self.get_enterprise_stats(),
            "status": "healthy" if self.running else "stopped"
        }
        
        # 发布到分布式事件总线
        event = DistributedEvent(
            event_id=str(uuid.uuid4()),
            event_type="node_health_status",
            source_node=self.node_id,
            target_nodes=[],  # 广播
            payload=health_data,
            timestamp=datetime.now(timezone.utc)
        )
        
        await self.distributed_event_bus.publish(event)
    
    async def _pool_scaling_monitor(self):
        """池扩缩容监控"""
        logger.info("池扩缩容监控启动")
        
        while self.running:
            try:
                for pool_id, pool in self.agent_pools.items():
                    current_time = datetime.now(timezone.utc)
                    
                    # 检查是否需要扩容
                    if pool.needs_scaling_up():
                        # 限制扩容频率（至少间隔5分钟）
                        if (current_time - pool.last_scaled).total_seconds() > 300:
                            await self._scale_pool_up(pool_id)
                    
                    # 检查是否需要缩容
                    elif pool.needs_scaling_down():
                        # 限制缩容频率（至少间隔10分钟）
                        if (current_time - pool.last_scaled).total_seconds() > 600:
                            await self._scale_pool_down(pool_id)
                
                # 等待下次检查
                await asyncio.sleep(120)  # 2分钟检查一次
                
            except Exception as e:
                logger.error("池扩缩容监控异常", error=str(e))
                await asyncio.sleep(60)
        
        logger.info("池扩缩容监控停止")
    
    def get_enterprise_stats(self) -> Dict[str, Any]:
        """获取企业级统计信息"""
        base_stats = self.get_manager_stats()
        
        # 池统计
        pool_stats = {}
        for pool_id, pool in self.agent_pools.items():
            pool_stats[pool_id] = {
                "name": pool.name,
                "role": pool.role.value if hasattr(pool.role, 'value') else str(pool.role),
                "size": len(pool.agents),
                "min_size": pool.config.min_size,
                "max_size": pool.config.max_size,
                "load": pool.get_current_load(),
                "available_agents": len(pool.get_available_agents()),
                "health_scores": [
                    agent.health_score
                    for agent in pool.agents.values()
                    if isinstance(agent, EnterpriseAgentInfo)
                ]
            }
        
        # 性能指标
        performance_metrics = {
            "average_response_time": self._calculate_average_response_time(),
            "throughput_per_minute": self._calculate_throughput(),
            "error_rate": self._calculate_error_rate(),
            "resource_utilization": self._get_resource_utilization()
        }
        
        return {
            **base_stats,
            "node_id": self.node_id,
            "pools": pool_stats,
            "performance": performance_metrics,
            "failover_nodes": list(self.failover_nodes),
            "load_balancing_strategy": self.load_balancing_strategy.value
        }
    
    def _calculate_average_response_time(self) -> float:
        """计算平均响应时间"""
        response_times = [
            agent.average_task_time
            for agent in self.agents.values()
            if isinstance(agent, EnterpriseAgentInfo) and agent.average_task_time > 0
        ]
        
        return sum(response_times) / len(response_times) if response_times else 0.0
    
    def _calculate_throughput(self) -> float:
        """计算吞吐量（每分钟完成任务数）"""
        current_time = datetime.now(timezone.utc)
        one_minute_ago = current_time - timedelta(minutes=1)
        
        recent_tasks = [
            task for task in self.tasks.values()
            if (task.completed_at and task.completed_at > one_minute_ago and 
                task.status == TaskStatus.COMPLETED)
        ]
        
        return len(recent_tasks)
    
    def _calculate_error_rate(self) -> float:
        """计算错误率"""
        if not self.tasks:
            return 0.0
        
        failed_tasks = sum(1 for task in self.tasks.values() if task.status == TaskStatus.FAILED)
        return failed_tasks / len(self.tasks)
    
    def _get_resource_utilization(self) -> Dict[str, float]:
        """获取资源利用率"""
        cpu_usage = []
        memory_usage = []
        
        for agent in self.agents.values():
            if isinstance(agent, EnterpriseAgentInfo):
                cpu_usage.append(agent.resource_usage.get('cpu', 0))
                memory_usage.append(agent.resource_usage.get('memory', 0))
        
        return {
            "cpu": sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0.0,
            "memory": sum(memory_usage) / len(memory_usage) if memory_usage else 0.0,
            "concurrent_tasks": len(self.running_tasks),
            "max_concurrent": self.max_concurrent_tasks
        }


class SecurityLevel(str, Enum):
    """安全级别"""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class AlertSeverity(str, Enum):
    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SecurityContext:
    """安全上下文"""
    user_id: str
    session_id: str
    permissions: List[str] = field(default_factory=list)
    security_level: SecurityLevel = SecurityLevel.PUBLIC
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class AuditLogEntry:
    """审计日志条目"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    user_id: str = ""
    session_id: str = ""
    action: str = ""
    resource: str = ""
    result: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    security_context: Optional[SecurityContext] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "user_id": self.user_id,
            "session_id": self.session_id,
            "action": self.action,
            "resource": self.resource,
            "result": self.result,
            "details": self.details,
            "security_context": self.security_context.__dict__ if self.security_context else None
        }


@dataclass
class Alert:
    """告警"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    severity: AlertSeverity = AlertSeverity.INFO
    title: str = ""
    message: str = ""
    source: str = ""
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "tags": self.tags,
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class EnterpriseErrorHandler:
    """企业级错误处理器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.error_patterns = {}
        self.recovery_strategies = {}
        self.error_stats = {
            "total_errors": 0,
            "errors_by_type": {},
            "errors_by_agent": {},
            "recovery_attempts": 0,
            "successful_recoveries": 0
        }
    
    async def handle_agent_error(
        self,
        agent_id: str,
        error: Exception,
        context: Dict[str, Any],
        security_context: Optional[SecurityContext] = None
    ) -> bool:
        """处理智能体错误"""
        try:
            error_type = type(error).__name__
            error_message = str(error)
            
            # 更新错误统计
            self.error_stats["total_errors"] += 1
            if error_type not in self.error_stats["errors_by_type"]:
                self.error_stats["errors_by_type"][error_type] = 0
            self.error_stats["errors_by_type"][error_type] += 1
            
            if agent_id not in self.error_stats["errors_by_agent"]:
                self.error_stats["errors_by_agent"][agent_id] = 0
            self.error_stats["errors_by_agent"][agent_id] += 1
            
            # 记录错误事件
            await self.event_bus.publish(Event(
                type=EventType.ERROR_OCCURRED,
                source=agent_id,
                priority=EventPriority.HIGH,
                data={
                    "error_type": error_type,
                    "error_message": error_message,
                    "context": context,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            ))
            
            # 尝试错误恢复
            recovery_success = await self._attempt_error_recovery(
                agent_id, error_type, error_message, context
            )
            
            # 根据错误严重程度创建告警
            severity = self._determine_error_severity(error_type, context)
            if severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
                await self._create_error_alert(
                    agent_id, error_type, error_message, severity, context
                )
            
            logger.error(
                "智能体错误处理完成",
                agent_id=agent_id,
                error_type=error_type,
                recovery_success=recovery_success,
                severity=severity.value
            )
            
            return recovery_success
            
        except Exception as e:
            logger.error("错误处理器异常", error=str(e))
            return False
    
    async def _attempt_error_recovery(
        self,
        agent_id: str,
        error_type: str,
        error_message: str,
        context: Dict[str, Any]
    ) -> bool:
        """尝试错误恢复"""
        try:
            self.error_stats["recovery_attempts"] += 1
            
            # 根据错误类型选择恢复策略
            recovery_strategy = self.recovery_strategies.get(error_type)
            if recovery_strategy:
                success = await recovery_strategy(agent_id, error_message, context)
                if success:
                    self.error_stats["successful_recoveries"] += 1
                    logger.info("错误恢复成功", agent_id=agent_id, error_type=error_type)
                return success
            
            # 默认恢复策略：重启智能体
            logger.info("使用默认恢复策略", agent_id=agent_id, error_type=error_type)
            return await self._default_recovery_strategy(agent_id, context)
            
        except Exception as e:
            logger.error("错误恢复失败", agent_id=agent_id, error=str(e))
            return False
    
    async def _default_recovery_strategy(
        self,
        agent_id: str,
        context: Dict[str, Any]
    ) -> bool:
        """默认恢复策略"""
        try:
            # 这里可以实现智能体重启逻辑
            logger.info("执行默认恢复策略", agent_id=agent_id)
            return True
        except Exception as e:
            logger.error("默认恢复策略失败", agent_id=agent_id, error=str(e))
            return False
    
    def _determine_error_severity(
        self,
        error_type: str,
        context: Dict[str, Any]
    ) -> AlertSeverity:
        """确定错误严重程度"""
        critical_errors = [
            "SystemExit", "KeyboardInterrupt", "MemoryError",
            "RecursionError", "OSError"
        ]
        
        error_errors = [
            "ValueError", "TypeError", "AttributeError",
            "KeyError", "IndexError", "TimeoutError"
        ]
        
        if error_type in critical_errors:
            return AlertSeverity.CRITICAL
        elif error_type in error_errors:
            return AlertSeverity.ERROR
        elif "timeout" in error_type.lower():
            return AlertSeverity.WARNING
        else:
            return AlertSeverity.INFO
    
    async def _create_error_alert(
        self,
        agent_id: str,
        error_type: str,
        error_message: str,
        severity: AlertSeverity,
        context: Dict[str, Any]
    ) -> None:
        """创建错误告警"""
        alert = Alert(
            severity=severity,
            title=f"智能体错误: {error_type}",
            message=f"智能体 {agent_id} 发生错误: {error_message}",
            source=agent_id,
            tags=["agent_error", error_type],
            metadata={
                "error_type": error_type,
                "error_message": error_message,
                "context": context
            }
        )
        
        # 发布告警事件
        await self.event_bus.publish(Event(
            type=EventType.ERROR_OCCURRED,
            source="error_handler",
            priority=EventPriority.CRITICAL if severity == AlertSeverity.CRITICAL else EventPriority.HIGH,
            data=alert.to_dict()
        ))
    
    def register_recovery_strategy(
        self,
        error_type: str,
        strategy: Callable[[str, str, Dict[str, Any]], bool]
    ) -> None:
        """注册错误恢复策略"""
        self.recovery_strategies[error_type] = strategy
        logger.info("注册错误恢复策略", error_type=error_type)
    
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        return self.error_stats.copy()


class SecurityManager:
    """安全管理器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.access_rules = {}
        self.rate_limits = {}
        self.security_policies = {}
        self.failed_attempts = {}
        
    async def check_permissions(
        self,
        security_context: SecurityContext,
        resource: str,
        action: str
    ) -> bool:
        """检查权限"""
        try:
            # 检查用户权限
            required_permission = f"{resource}:{action}"
            if required_permission not in security_context.permissions:
                await self._log_security_violation(
                    security_context, resource, action, "权限不足"
                )
                return False
            
            # 检查安全级别
            resource_security_level = self.security_policies.get(resource, SecurityLevel.PUBLIC)
            if security_context.security_level.value < resource_security_level.value:
                await self._log_security_violation(
                    security_context, resource, action, "安全级别不足"
                )
                return False
            
            # 检查频率限制
            if not await self._check_rate_limit(security_context, action):
                await self._log_security_violation(
                    security_context, resource, action, "频率限制"
                )
                return False
            
            return True
            
        except Exception as e:
            logger.error("权限检查失败", error=str(e))
            return False
    
    async def _check_rate_limit(
        self,
        security_context: SecurityContext,
        action: str
    ) -> bool:
        """检查频率限制"""
        key = f"{security_context.user_id}:{action}"
        current_time = time.time()
        
        if key not in self.rate_limits:
            self.rate_limits[key] = []
        
        # 清理过期记录
        self.rate_limits[key] = [
            t for t in self.rate_limits[key]
            if current_time - t < 3600  # 1小时窗口
        ]
        
        # 检查限制
        limit = get_config_manager().get_int('SECURITY_MAX_VIOLATIONS_PER_HOUR', 100)
        if len(self.rate_limits[key]) >= limit:
            return False
        
        # 记录本次访问
        self.rate_limits[key].append(current_time)
        return True
    
    async def _log_security_violation(
        self,
        security_context: SecurityContext,
        resource: str,
        action: str,
        reason: str
    ) -> None:
        """记录安全违规"""
        await self.event_bus.publish(Event(
            type=EventType.ERROR_OCCURRED,
            source="security_manager",
            priority=EventPriority.HIGH,
            data={
                "type": "security_violation",
                "user_id": security_context.user_id,
                "session_id": security_context.session_id,
                "resource": resource,
                "action": action,
                "reason": reason,
                "ip_address": security_context.ip_address,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        ))
        
        # 记录失败次数
        key = security_context.user_id
        if key not in self.failed_attempts:
            self.failed_attempts[key] = 0
        self.failed_attempts[key] += 1
        
        # 如果失败次数过多，创建告警
        if self.failed_attempts[key] > 5:
            alert = Alert(
                severity=AlertSeverity.WARNING,
                title="频繁安全违规",
                message=f"用户 {security_context.user_id} 频繁违反安全策略",
                source="security_manager",
                tags=["security", "violation"],
                metadata={
                    "user_id": security_context.user_id,
                    "attempts": self.failed_attempts[key]
                }
            )
            
            await self.event_bus.publish(Event(
                type=EventType.ERROR_OCCURRED,
                source="security_manager",
                priority=EventPriority.HIGH,
                data=alert.to_dict()
            ))
    
    def set_security_policy(self, resource: str, security_level: SecurityLevel) -> None:
        """设置安全策略"""
        self.security_policies[resource] = security_level
        logger.info("设置安全策略", resource=resource, level=security_level.value)


class AuditLogger:
    """审计日志器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.audit_logs: List[AuditLogEntry] = []
        self.max_logs = get_config_manager().get_int('MONITORING_MAX_LOGS', 10000)
        
    async def log_action(
        self,
        user_id: str,
        session_id: str,
        action: str,
        resource: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> None:
        """记录操作日志"""
        try:
            log_entry = AuditLogEntry(
                user_id=user_id,
                session_id=session_id,
                action=action,
                resource=resource,
                result=result,
                details=details or {},
                security_context=security_context
            )
            
            # 添加到内存日志
            self.audit_logs.append(log_entry)
            
            # 保持日志数量在限制内
            if len(self.audit_logs) > self.max_logs:
                self.audit_logs = self.audit_logs[-self.max_logs:]
            
            # 发布审计事件
            await self.event_bus.publish(Event(
                type=EventType.MESSAGE_SENT,  # 使用适当的事件类型
                source="audit_logger",
                data={
                    "type": "audit_log",
                    "entry": log_entry.to_dict()
                }
            ))
            
            logger.debug(
                "审计日志记录",
                user_id=user_id,
                action=action,
                resource=resource,
                result=result
            )
            
        except Exception as e:
            logger.error("审计日志记录失败", error=str(e))
    
    async def log_agent_operation(
        self,
        agent_id: str,
        operation: str,
        result: str,
        details: Optional[Dict[str, Any]] = None,
        security_context: Optional[SecurityContext] = None
    ) -> None:
        """记录智能体操作日志"""
        await self.log_action(
            user_id=security_context.user_id if security_context else "system",
            session_id=security_context.session_id if security_context else "system",
            action=f"agent.{operation}",
            resource=agent_id,
            result=result,
            details=details,
            security_context=security_context
        )
    
    def search_logs(
        self,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100
    ) -> List[AuditLogEntry]:
        """搜索审计日志"""
        results = []
        
        for log in self.audit_logs:
            # 应用过滤条件
            if user_id and log.user_id != user_id:
                continue
            if action and action not in log.action:
                continue
            if resource and resource not in log.resource:
                continue
            if start_time and log.timestamp < start_time:
                continue
            if end_time and log.timestamp > end_time:
                continue
            
            results.append(log)
            
            # 限制结果数量
            if len(results) >= limit:
                break
        
        return results
    
    def get_audit_stats(self) -> Dict[str, Any]:
        """获取审计统计"""
        total_logs = len(self.audit_logs)
        actions_count = {}
        users_count = {}
        
        for log in self.audit_logs:
            # 统计操作类型
            if log.action not in actions_count:
                actions_count[log.action] = 0
            actions_count[log.action] += 1
            
            # 统计用户活动
            if log.user_id not in users_count:
                users_count[log.user_id] = 0
            users_count[log.user_id] += 1
        
        return {
            "total_logs": total_logs,
            "actions_count": actions_count,
            "users_count": users_count,
            "oldest_log": self.audit_logs[0].timestamp.isoformat() if self.audit_logs else None,
            "newest_log": self.audit_logs[-1].timestamp.isoformat() if self.audit_logs else None
        }


class MonitoringSystem:
    """监控系统"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.metrics = {}
        self.alerts: List[Alert] = []
        self.thresholds = {
            "error_rate": 0.1,  # 10% 错误率
            "response_time": 5.0,  # 5秒响应时间
            "memory_usage": 0.8,  # 80% 内存使用率
            "cpu_usage": 0.9  # 90% CPU使用率
        }
        
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """记录指标"""
        timestamp = datetime.now(timezone.utc)
        
        if name not in self.metrics:
            self.metrics[name] = []
        
        metric_entry = {
            "timestamp": timestamp,
            "value": value,
            "tags": tags or {}
        }
        
        self.metrics[name].append(metric_entry)
        
        # 保持最近1000个数据点
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
        
        # 检查阈值
        asyncio.create_task(self._check_thresholds(name, value, tags))
    
    async def _check_thresholds(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]]
    ) -> None:
        """检查阈值"""
        try:
            threshold = self.thresholds.get(metric_name)
            if threshold and value > threshold:
                alert = Alert(
                    severity=AlertSeverity.WARNING,
                    title=f"指标阈值超限: {metric_name}",
                    message=f"指标 {metric_name} 的值 {value} 超过阈值 {threshold}",
                    source="monitoring_system",
                    tags=["threshold", "monitoring", metric_name],
                    metadata={
                        "metric_name": metric_name,
                        "value": value,
                        "threshold": threshold,
                        "tags": tags
                    }
                )
                
                self.alerts.append(alert)
                
                # 发布告警事件
                await self.event_bus.publish(Event(
                    type=EventType.ERROR_OCCURRED,
                    source="monitoring_system",
                    priority=EventPriority.HIGH,
                    data=alert.to_dict()
                ))
                
        except Exception as e:
            logger.error("阈值检查失败", error=str(e))
    
    def get_metric_summary(self, name: str, duration_minutes: int = 60) -> Dict[str, Any]:
        """获取指标摘要"""
        if name not in self.metrics:
            return {}
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=duration_minutes)
        recent_metrics = [
            m for m in self.metrics[name]
            if m["timestamp"] > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        values = [m["value"] for m in recent_metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "avg": sum(values) / len(values),
            "latest": values[-1] if values else None,
            "duration_minutes": duration_minutes
        }
    
    def get_all_alerts(self, resolved: Optional[bool] = None) -> List[Alert]:
        """获取所有告警"""
        if resolved is None:
            return self.alerts.copy()
        
        return [alert for alert in self.alerts if alert.resolved == resolved]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.now(timezone.utc)
                return True
        return False
    
    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """设置阈值"""
        self.thresholds[metric_name] = threshold
        logger.info("设置监控阈值", metric=metric_name, threshold=threshold)


class EnterpriseIntegrationService:
    """企业级集成服务"""
    
    def __init__(self, event_bus: EventBus, agent_manager: AsyncAgentManager):
        self.event_bus = event_bus
        self.agent_manager = agent_manager
        
        # 初始化企业级组件
        self.error_handler = EnterpriseErrorHandler(event_bus)
        self.security_manager = SecurityManager(event_bus)
        self.audit_logger = AuditLogger(event_bus)
        self.monitoring_system = MonitoringSystem(event_bus)
        
        # 注册事件处理器
        self._register_event_handlers()
        
        logger.info("企业级集成服务初始化完成")
    
    def _register_event_handlers(self):
        """注册事件处理器"""
        # 注册企业级事件处理器
        enterprise_handler = EnterpriseEventHandler(
            self.error_handler,
            self.security_manager,
            self.audit_logger,
            self.monitoring_system
        )
        
        self.event_bus.subscribe_all(enterprise_handler)
    
    async def handle_agent_operation(
        self,
        operation: str,
        agent_id: str,
        security_context: SecurityContext,
        operation_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """处理智能体操作"""
        try:
            # 权限检查
            if not await self.security_manager.check_permissions(
                security_context, f"agent:{agent_id}", operation
            ):
                result = {"success": False, "error": "权限不足"}
                await self.audit_logger.log_agent_operation(
                    agent_id, operation, "denied", {"error": "权限不足"}, security_context
                )
                return result
            
            # 记录操作开始
            start_time = time.time()
            
            # 执行操作（这里需要根据具体操作类型实现）
            operation_result = await self._execute_agent_operation(
                operation, agent_id, operation_data or {}
            )
            
            # 记录执行时间
            execution_time = time.time() - start_time
            self.monitoring_system.record_metric(
                "agent_operation_time",
                execution_time,
                {"operation": operation, "agent_id": agent_id}
            )
            
            # 记录审计日志
            await self.audit_logger.log_agent_operation(
                agent_id,
                operation,
                "success" if operation_result.get("success") else "failed",
                {
                    "execution_time": execution_time,
                    "result": operation_result
                },
                security_context
            )
            
            return operation_result
            
        except Exception as e:
            # 错误处理
            await self.error_handler.handle_agent_error(
                agent_id, e, {"operation": operation}, security_context
            )
            
            await self.audit_logger.log_agent_operation(
                agent_id, operation, "error", {"error": str(e)}, security_context
            )
            
            return {"success": False, "error": str(e)}
    
    async def _execute_agent_operation(
        self,
        operation: str,
        agent_id: str,
        operation_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行智能体操作"""
        if operation == "create":
            # 创建智能体
            return {"success": True, "agent_id": agent_id}
        elif operation == "destroy":
            # 销毁智能体
            success = await self.agent_manager.destroy_agent(agent_id)
            return {"success": success}
        elif operation == "submit_task":
            # 提交任务
            task_id = await self.agent_manager.submit_task(
                agent_id=agent_id,
                task_type=operation_data.get("task_type", "general"),
                description=operation_data.get("description", ""),
                input_data=operation_data.get("input_data", {}),
                priority=operation_data.get("priority", 0)
            )
            return {"success": True, "task_id": task_id}
        else:
            return {"success": False, "error": f"未知操作: {operation}"}
    
    def get_enterprise_status(self) -> Dict[str, Any]:
        """获取企业级状态"""
        return {
            "error_stats": self.error_handler.get_error_stats(),
            "audit_stats": self.audit_logger.get_audit_stats(),
            "active_alerts": len(self.monitoring_system.get_all_alerts(resolved=False)),
            "total_alerts": len(self.monitoring_system.get_all_alerts()),
            "metrics_count": len(self.monitoring_system.metrics),
            "thresholds": self.monitoring_system.thresholds
        }


class EnterpriseEventHandler(EventHandler):
    """企业级事件处理器"""
    
    def __init__(
        self,
        error_handler: EnterpriseErrorHandler,
        security_manager: SecurityManager,
        audit_logger: AuditLogger,
        monitoring_system: MonitoringSystem
    ):
        self.error_handler = error_handler
        self.security_manager = security_manager
        self.audit_logger = audit_logger
        self.monitoring_system = monitoring_system
    
    @property
    def supported_events(self) -> List[EventType]:
        return list(EventType)  # 支持所有事件类型
    
    async def handle(self, event: Event) -> None:
        """处理事件"""
        try:
            # 记录事件指标
            self.monitoring_system.record_metric(
                "event_count",
                1,
                {"event_type": event.type.value, "source": event.source}
            )
            
            # 特殊事件处理
            if event.type == EventType.ERROR_OCCURRED:
                await self._handle_error_event(event)
            elif event.type == EventType.AGENT_CREATED:
                await self._handle_agent_created_event(event)
            elif event.type == EventType.TASK_COMPLETED:
                await self._handle_task_completed_event(event)
                
        except Exception as e:
            logger.error("企业级事件处理失败", event_type=event.type, error=str(e))
    
    async def _handle_error_event(self, event: Event) -> None:
        """处理错误事件"""
        error_data = event.data
        
        # 记录错误指标
        self.monitoring_system.record_metric(
            "error_rate",
            1,
            {"source": event.source, "error_type": error_data.get("error_type")}
        )
    
    async def _handle_agent_created_event(self, event: Event) -> None:
        """处理智能体创建事件"""
        # 记录智能体创建指标
        self.monitoring_system.record_metric(
            "agent_created",
            1,
            {"agent_id": event.target}
        )
    
    async def _handle_task_completed_event(self, event: Event) -> None:
        """处理任务完成事件"""
        task_data = event.data
        execution_time = task_data.get("execution_time", 0)
        
        # 记录任务执行时间
        self.monitoring_system.record_metric(
            "task_execution_time",
            execution_time,
            {"agent_id": event.source, "task_type": task_data.get("task_type")}
        )
    
    async def _collect_enterprise_metrics(self):
        """收集企业级指标"""
        while self.running:
            try:
                # 收集智能体池指标
                self._collect_agent_pool_metrics()
                
                # 收集任务队列指标  
                self._collect_task_queue_metrics()
                
                # 收集流控指标
                self._collect_flow_control_metrics()
                
                # 收集分布式事件指标
                self._collect_distributed_event_metrics()
                
                # 收集错误统计指标
                self._collect_error_metrics()
                
                await asyncio.sleep(30)  # 每30秒收集一次
                
            except Exception as e:
                logger.error(f"收集企业级指标失败: {e}")
                await asyncio.sleep(30)
    
    def _collect_agent_pool_metrics(self):
        """收集智能体池指标"""
        total_agents = 0
        active_agents = 0
        idle_agents = 0
        
        for pool in self.agent_pools.values():
            pool_size = len(pool.agents)
            pool_active = len([a for a in pool.agents.values() if a.status == AgentStatus.BUSY])
            pool_idle = len([a for a in pool.agents.values() if a.status == AgentStatus.IDLE])
            
            total_agents += pool_size
            active_agents += pool_active
            idle_agents += pool_idle
            
            # 记录池级指标
            self.metric_collector.record_metric("agent_pool_size", pool_size, {"pool_id": pool.pool_id})
            self.metric_collector.record_metric("agent_pool_active", pool_active, {"pool_id": pool.pool_id})
            self.metric_collector.record_metric("agent_pool_idle", pool_idle, {"pool_id": pool.pool_id})
            self.metric_collector.record_metric("agent_pool_load", pool.get_current_load(), {"pool_id": pool.pool_id})
        
        # 记录全局指标
        self.metric_collector.record_metric("total_agents", total_agents)
        self.metric_collector.record_metric("total_active_agents", active_agents)  
        self.metric_collector.record_metric("total_idle_agents", idle_agents)
    
    def _collect_task_queue_metrics(self):
        """收集任务队列指标"""
        if hasattr(self, 'task_queue') and self.task_queue:
            queue_size = self.task_queue.qsize() if hasattr(self.task_queue, 'qsize') else 0
            self.metric_collector.record_metric("task_queue_size", queue_size)
    
    def _collect_flow_control_metrics(self):
        """收集流控指标"""
        if self.flow_controller:
            metrics = self.flow_controller.get_current_metrics()
            
            self.metric_collector.record_metric("flow_control_throughput", metrics.throughput)
            self.metric_collector.record_metric("flow_control_queue_size", metrics.queue_size)
            self.metric_collector.record_metric("flow_control_avg_latency", metrics.avg_latency)
            self.metric_collector.record_metric("flow_control_drop_rate", 
                                              (metrics.dropped_tasks / max(metrics.dropped_tasks + 1, 1)) * 100)
            self.metric_collector.record_metric("flow_control_backpressure", 
                                              1 if metrics.backpressure_triggered else 0)
    
    def _collect_distributed_event_metrics(self):
        """收集分布式事件指标"""
        if self.distributed_event_bus and hasattr(self.distributed_event_bus, 'stats'):
            stats = self.distributed_event_bus.stats
            
            self.metric_collector.record_metric("event_bus_messages_sent", 
                                              stats.get("events_sent", 0))
            self.metric_collector.record_metric("event_bus_messages_received", 
                                              stats.get("events_received", 0))
            self.metric_collector.record_metric("event_bus_processing_errors", 
                                              stats.get("processing_errors", 0))
    
    def _collect_error_metrics(self):
        """收集错误统计指标"""
        if hasattr(self, '_error_stats'):
            for category, count in self._error_stats.items():
                self.metric_collector.record_metric("error_count_by_category", count, 
                                                  {"category": category})
        
        # 收集结构化错误统计
        # 这里可以添加从错误记录中统计的逻辑