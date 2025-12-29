"""
智能体服务发现核心实现（基于Redis）

目标：
- 真实注册/发现：数据落到Redis，使用TTL自动过期
- 负载均衡：基于注册的指标做选择（非静态假数据）
"""

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
import redis.asyncio as redis

from src.core.logging import get_logger
class AgentStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"

@dataclass
class AgentCapability:
    name: str
    description: str
    version: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.performance_metrics:
            self.performance_metrics = {
                "avg_latency": 1.0,
                "throughput": 10.0,
                "accuracy": 0.95,
                "success_rate": 0.98,
            }

@dataclass
class AgentMetadata:
    agent_id: str
    agent_type: str
    name: str
    version: str
    capabilities: List[AgentCapability]
    host: str
    port: int
    endpoint: str
    health_endpoint: str
    resources: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    group: str = "default"
    region: str = "default"
    status: AgentStatus = AgentStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_heartbeat: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    request_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0

class HealthCheckStrategy:
    def __init__(
        self,
        interval: float = 30.0,
        timeout: float = 5.0,
        retries: int = 3,
        failure_threshold: int = 3,
        success_threshold: int = 1,
    ):
        self.interval = interval
        self.timeout = timeout
        self.retries = retries
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold

class ServiceRegistry:
    def __init__(
        self,
        redis_client: redis.Redis,
        prefix: str = "service_discovery:agents:",
        ttl_seconds: int = 90,
    ):
        self.redis = redis_client
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self.logger = get_logger(__name__)
        self.metrics = {
            "registered_agents": 0,
            "active_agents": 0,
            "health_checks": 0,
            "failed_health_checks": 0,
            "discovery_requests": 0,
        }

    async def initialize(self):
        if not self.redis:
            raise RuntimeError("Redis未初始化，无法启动服务发现")
        await self.redis.ping()
        self.logger.info("服务发现注册表已初始化（Redis）")

    def _agent_key(self, agent_id: str) -> str:
        return f"{self.prefix}{agent_id}"

    def _serialize(self, metadata: AgentMetadata) -> str:
        data = asdict(metadata)
        data["status"] = metadata.status.value
        data["created_at"] = metadata.created_at.isoformat()
        data["last_heartbeat"] = metadata.last_heartbeat.isoformat()
        return json.dumps(data, ensure_ascii=False)

    def _deserialize(self, raw: str) -> AgentMetadata:
        data = json.loads(raw)
        capabilities = [AgentCapability(**cap) for cap in data.get("capabilities", [])]
        created_at = datetime.fromisoformat(data["created_at"])
        last_heartbeat = datetime.fromisoformat(data["last_heartbeat"])
        return AgentMetadata(
            agent_id=data["agent_id"],
            agent_type=data["agent_type"],
            name=data["name"],
            version=data["version"],
            capabilities=capabilities,
            host=data["host"],
            port=int(data["port"]),
            endpoint=data["endpoint"],
            health_endpoint=data["health_endpoint"],
            resources=data.get("resources") or {},
            tags=data.get("tags") or [],
            group=data.get("group") or "default",
            region=data.get("region") or "default",
            status=AgentStatus(data.get("status", AgentStatus.ACTIVE.value)),
            created_at=created_at,
            last_heartbeat=last_heartbeat,
            request_count=int(data.get("request_count", 0)),
            error_count=int(data.get("error_count", 0)),
            avg_response_time=float(data.get("avg_response_time", 0.0)),
        )

    async def register_agent(self, metadata: AgentMetadata) -> bool:
        now = datetime.now(timezone.utc)
        metadata.last_heartbeat = now
        key = self._agent_key(metadata.agent_id)
        await self.redis.set(key, self._serialize(metadata), ex=self.ttl_seconds)
        self.metrics["registered_agents"] += 1
        return True

    async def deregister_agent(self, agent_id: str) -> bool:
        deleted = await self.redis.delete(self._agent_key(agent_id))
        return deleted > 0

    async def get_agent(self, agent_id: str) -> Optional[AgentMetadata]:
        raw = await self.redis.get(self._agent_key(agent_id))
        if not raw:
            return None
        return self._deserialize(raw)

    async def discover_agents(
        self,
        capability: Optional[str] = None,
        tags: Optional[List[str]] = None,
        status: Optional[AgentStatus] = None,
        group: Optional[str] = None,
        region: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AgentMetadata]:
        self.metrics["discovery_requests"] += 1
        agents: List[AgentMetadata] = []
        pattern = f"{self.prefix}*"
        async for key in self.redis.scan_iter(match=pattern):
            raw = await self.redis.get(key)
            if not raw:
                continue
            agent = self._deserialize(raw)
            if capability and not any(c.name == capability for c in agent.capabilities):
                continue
            if tags and not set(tags).issubset(set(agent.tags)):
                continue
            if status and agent.status != status:
                continue
            if group and agent.group != group:
                continue
            if region and agent.region != region:
                continue
            agents.append(agent)
            if limit and len(agents) >= limit:
                break
        return agents

    async def update_agent_status(self, agent_id: str, status: AgentStatus) -> bool:
        agent = await self.get_agent(agent_id)
        if not agent:
            return False
        agent.status = status
        agent.last_heartbeat = datetime.now(timezone.utc)
        await self.redis.set(self._agent_key(agent_id), self._serialize(agent), ex=self.ttl_seconds)
        return True

    async def update_agent_metrics(
        self,
        agent_id: str,
        request_count: Optional[int] = None,
        error_count: Optional[int] = None,
        avg_response_time: Optional[float] = None,
    ) -> bool:
        agent = await self.get_agent(agent_id)
        if not agent:
            return False
        if request_count is not None:
            agent.request_count = request_count
        if error_count is not None:
            agent.error_count = error_count
        if avg_response_time is not None:
            agent.avg_response_time = avg_response_time
        agent.last_heartbeat = datetime.now(timezone.utc)
        await self.redis.set(self._agent_key(agent_id), self._serialize(agent), ex=self.ttl_seconds)
        return True

    async def list_all_agents(self) -> List[AgentMetadata]:
        return await self.discover_agents()

    async def get_registry_stats(self) -> Dict[str, Any]:
        agents = await self.list_all_agents()
        active = sum(1 for a in agents if a.status == AgentStatus.ACTIVE)
        return {
            "registered_agents": len(agents),
            "active_agents": active,
            "discovery_requests": self.metrics["discovery_requests"],
        }

    async def cleanup(self):
        return

class LoadBalancer:
    def __init__(self, registry: ServiceRegistry):
        self.registry = registry
        self._rr_index: Dict[str, int] = {}

    def _rr_key(self, capability: str, tags: Optional[List[str]]) -> str:
        tag_part = ",".join(sorted(tags or []))
        return f"{capability}|{tag_part}"

    def _score_agent(self, agent: AgentMetadata, capability: str) -> float:
        cap = next((c for c in agent.capabilities if c.name == capability), None)
        m = cap.performance_metrics if cap else {}
        avg_latency = float(m.get("avg_latency", 1.0))
        throughput = float(m.get("throughput", 0.0))
        accuracy = float(m.get("accuracy", 0.0))
        success_rate = float(m.get("success_rate", 0.0))
        reliability = 1.0
        if agent.request_count > 0:
            reliability = max(0.0, 1.0 - agent.error_count / agent.request_count)
        cpu_usage = float(agent.resources.get("cpu_usage", 0.0) or 0.0)
        mem_usage = float(agent.resources.get("memory_usage", 0.0) or 0.0)
        score = throughput * 0.4 + accuracy * 10.0 * 0.3 + success_rate * 10.0 * 0.3
        score = score * reliability
        score -= avg_latency
        score -= (cpu_usage + mem_usage) * 0.5
        return score

    async def select_agent(
        self,
        capability: str,
        strategy: str = "capability_based",
        tags: Optional[List[str]] = None,
        requirements: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentMetadata]:
        agents = await self.registry.discover_agents(
            capability=capability,
            tags=tags,
            status=AgentStatus.ACTIVE,
        )
        if not agents:
            return None

        if strategy == "round_robin":
            key = self._rr_key(capability, tags)
            idx = self._rr_index.get(key, 0) % len(agents)
            self._rr_index[key] = idx + 1
            return agents[idx]

        if strategy == "least_connections":
            return min(agents, key=lambda a: a.request_count)

        if strategy == "least_response_time":
            return min(agents, key=lambda a: a.avg_response_time if a.avg_response_time > 0 else 1e9)

        if strategy == "weighted_random":
            weights = []
            for a in agents:
                w = max(0.1, self._score_agent(a, capability))
                weights.append(w)
            return random.choices(agents, weights=weights, k=1)[0]

        if strategy == "resource_aware":
            cpu_th = float((requirements or {}).get("cpu_threshold", 0.8))
            mem_th = float((requirements or {}).get("memory_threshold", 0.8))
            candidates = [
                a
                for a in agents
                if float(a.resources.get("cpu_usage", 0.0) or 0.0) <= cpu_th
                and float(a.resources.get("memory_usage", 0.0) or 0.0) <= mem_th
            ]
            if candidates:
                return max(candidates, key=lambda a: self._score_agent(a, capability))
            return max(agents, key=lambda a: self._score_agent(a, capability))

        return max(agents, key=lambda a: self._score_agent(a, capability))

    async def get_load_balancer_stats(self) -> Dict[str, Any]:
        return {"strategies": ["round_robin", "weighted_random", "least_connections", "least_response_time", "capability_based", "resource_aware"]}

class AgentServiceDiscoverySystem:
    def __init__(self, redis_client: redis.Redis, prefix: str = "service_discovery:agents:", ttl_seconds: int = 90):
        self.registry = ServiceRegistry(redis_client, prefix=prefix, ttl_seconds=ttl_seconds)
        self.load_balancer = LoadBalancer(self.registry)
        self.logger = get_logger(__name__)

    async def initialize(self):
        await self.registry.initialize()
        self.logger.info("服务发现系统已初始化")

    async def register_agent(
        self,
        agent_id: str,
        agent_type: str,
        name: str,
        version: str,
        capabilities: List[Dict[str, Any]],
        host: str,
        port: int,
        endpoint: str,
        health_endpoint: str,
        **kwargs,
    ) -> bool:
        capability_objects = [AgentCapability(**cap) for cap in capabilities]
        metadata = AgentMetadata(
            agent_id=agent_id,
            agent_type=agent_type,
            name=name,
            version=version,
            capabilities=capability_objects,
            host=host,
            port=port,
            endpoint=endpoint,
            health_endpoint=health_endpoint,
            **kwargs,
        )
        return await self.registry.register_agent(metadata)

    async def discover_and_select_agent(
        self,
        capability: str,
        strategy: str = "capability_based",
        **kwargs,
    ) -> Optional[AgentMetadata]:
        return await self.load_balancer.select_agent(capability=capability, strategy=strategy, **kwargs)

    async def get_system_stats(self) -> Dict[str, Any]:
        registry_stats = await self.registry.get_registry_stats()
        lb_stats = await self.load_balancer.get_load_balancer_stats()
        return {
            "registry": registry_stats,
            "load_balancer": lb_stats,
            "system_status": "healthy" if registry_stats["active_agents"] > 0 else "no_agents",
        }

    async def cleanup(self):
        await self.registry.cleanup()
        self.logger.info("服务发现系统已清理")
