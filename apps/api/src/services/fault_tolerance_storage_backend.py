import json
from typing import Any, Dict, List, Optional
import redis.asyncio as redis
from src.ai.cluster.topology import AgentStatus
from src.ai.distributed_task.models import Task, TaskStatus

from src.core.logging import get_logger
logger = get_logger(__name__)

class FaultToleranceStorageBackend:
    def __init__(self, redis_client: redis.Redis, cluster_manager: Any = None, task_coordinator: Any = None):
        self.redis = redis_client
        self.cluster_manager = cluster_manager
        self.task_coordinator = task_coordinator
        self.prefix = "fault_tolerance:storage:"

    async def _get_json(self, key: str) -> Optional[Any]:
        raw = await self.redis.get(key)
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    async def _set_json(self, key: str, value: Any) -> None:
        await self.redis.set(key, json.dumps(value, ensure_ascii=False))

    async def get_component_state(self, component_id: str) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        if self.cluster_manager and hasattr(self.cluster_manager, "get_agent_info"):
            agent = await self.cluster_manager.get_agent_info(component_id)
            if agent:
                state = {
                    "agent_id": agent.agent_id,
                    "name": agent.name,
                    "endpoint": agent.endpoint,
                    "status": agent.status.value if getattr(agent, "status", None) else None,
                    "updated_at": getattr(agent, "updated_at", None),
                }
        stored = await self._get_json(f"{self.prefix}state:{component_id}")
        if isinstance(stored, dict):
            state = {**stored, **state}
        return state

    async def set_component_state(self, component_id: str, state: Dict[str, Any]) -> bool:
        await self._set_json(f"{self.prefix}state:{component_id}", state)
        if self.cluster_manager and hasattr(self.cluster_manager, "update_agent_status") and "status" in state:
            try:
                await self.cluster_manager.update_agent_status(component_id, AgentStatus(state["status"]))
            except Exception:
                logger.exception("更新集群组件状态失败", exc_info=True)
        return True

    async def get_component_config(self, component_id: str) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        if self.cluster_manager and hasattr(self.cluster_manager, "get_agent_info"):
            agent = await self.cluster_manager.get_agent_info(component_id)
            if agent and getattr(agent, "config", None):
                config = dict(agent.config)
        stored = await self._get_json(f"{self.prefix}config:{component_id}")
        if isinstance(stored, dict):
            config = {**stored, **config}
        return config

    async def set_component_config(self, component_id: str, config: Dict[str, Any]) -> bool:
        await self._set_json(f"{self.prefix}config:{component_id}", config)
        updater = getattr(self.cluster_manager, "update_agent_config", None) if self.cluster_manager else None
        if updater:
            try:
                await updater(component_id, config)
            except Exception:
                logger.exception("更新集群组件配置失败", exc_info=True)
        return True

    async def get_component_tasks(self, component_id: str) -> List[Dict[str, Any]]:
        tasks: List[Dict[str, Any]] = []
        if self.task_coordinator and hasattr(self.task_coordinator, "get_agent_tasks"):
            try:
                for task in await self.task_coordinator.get_agent_tasks(component_id):
                    if hasattr(task, "to_dict"):
                        tasks.append(task.to_dict())
                    else:
                        tasks.append(dict(getattr(task, "__dict__", {})))
            except Exception:
                logger.exception("获取组件任务失败", exc_info=True)
        stored = await self._get_json(f"{self.prefix}tasks:{component_id}")
        if isinstance(stored, list) and not tasks:
            return stored
        return tasks

    async def set_component_tasks(self, component_id: str, tasks: List[Dict[str, Any]]) -> bool:
        await self._set_json(f"{self.prefix}tasks:{component_id}", tasks)
        if self.task_coordinator and hasattr(self.task_coordinator, "active_tasks"):
            try:
                for item in tasks:
                    task = Task.from_dict(item)
                    if task.status in {TaskStatus.PENDING, TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.RETRY}:
                        self.task_coordinator.active_tasks[task.task_id] = task
                    if hasattr(self.task_coordinator, "state_manager"):
                        await self.task_coordinator.state_manager.set_global_state(
                            f"task_{task.task_id}",
                            task.to_dict()
                        )
            except Exception:
                logger.exception("同步组件任务失败", exc_info=True)
        return True

    async def get_component_metrics(self, component_id: str) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {}
        if self.cluster_manager and hasattr(self.cluster_manager, "get_agent_info"):
            agent = await self.cluster_manager.get_agent_info(component_id)
            usage = getattr(agent, "resource_usage", None) if agent else None
            if usage:
                metrics = {
                    "cpu": float(getattr(usage, "cpu_usage_percent", 0.0)),
                    "memory": float(getattr(usage, "memory_usage_percent", 0.0)),
                    "disk": float(getattr(usage, "storage_usage_percent", 0.0)),
                    "error_rate": float(getattr(usage, "error_rate", 0.0)),
                    "active_tasks": int(getattr(usage, "active_tasks", 0)),
                }
        stored = await self._get_json(f"{self.prefix}metrics:{component_id}")
        if isinstance(stored, dict):
            metrics = {**stored, **metrics}
        return metrics

    async def get_cluster_state(self, component_id: str) -> Dict[str, Any]:
        if not self.cluster_manager or not hasattr(self.cluster_manager, "get_cluster_topology"):
            return {}
        topology = await self.cluster_manager.get_cluster_topology()
        return topology.to_dict() if topology else {}

    async def get_task_assignments(self, component_id: str) -> Dict[str, Any]:
        if not self.task_coordinator:
            return {}
        active_tasks = list(getattr(self.task_coordinator, "active_tasks", {}).values())
        return {
            "assigned": [t.task_id for t in active_tasks if t.assigned_to == component_id],
            "active_total": len(active_tasks),
        }

    async def get_agent_config(self, component_id: str) -> Dict[str, Any]:
        return await self.get_component_config(component_id)

    async def get_component_data(self, component_id: str, data_key: str) -> Any:
        return await self._get_json(f"{self.prefix}data:{component_id}:{data_key}")
