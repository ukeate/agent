"""
智能体生命周期管理器

负责智能体的生命周期管理，包括注册、启动、停止、重启、升级等操作。
实现操作历史记录、审计功能和批量操作支持。
"""

import asyncio
import time
import uuid
import httpx
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from .topology import AgentInfo, AgentStatus, AgentHealthCheck
from .state_manager import ClusterStateManager

from src.core.logging import get_logger

from src.core.utils.async_utils import create_task_with_logging
class AgentOperation(Enum):
    """智能体操作类型"""
    START = "start"
    STOP = "stop"
    RESTART = "restart"
    UPGRADE = "upgrade"
    REGISTER = "register"
    UNREGISTER = "unregister"
    HEALTH_CHECK = "health_check"

@dataclass
class OperationResult:
    """操作结果"""
    success: bool
    operation: AgentOperation
    agent_id: str
    message: str
    details: Dict[str, Any]
    timestamp: float
    duration_ms: float
    operation_id: str

@dataclass
class BatchOperationResult:
    """批量操作结果"""
    total_count: int
    success_count: int
    failed_count: int
    results: Dict[str, OperationResult]
    operation_type: AgentOperation
    started_at: float
    completed_at: float
    batch_id: str
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_count == 0:
            return 0.0
        return self.success_count / self.total_count
    
    @property
    def duration_seconds(self) -> float:
        """操作时长"""
        return self.completed_at - self.started_at

class LifecycleManager:
    """智能体生命周期管理器
    
    提供智能体生命周期的完整管理功能，包括：
    - 智能体注册和注销
    - 启动、停止、重启操作
    - 版本升级和配置更新
    - 批量操作支持
    - 操作历史和审计
    """
    
    def __init__(self, cluster_manager: ClusterStateManager):
        self.cluster_manager = cluster_manager
        self.logger = get_logger(__name__)
        
        # 操作锁，防止并发操作冲突
        self.operation_locks: Dict[str, asyncio.Lock] = {}
        
        # 操作历史
        self.operation_history: List[OperationResult] = []
        self.max_history_size = 10000
        
        # 批量操作历史
        self.batch_history: List[BatchOperationResult] = []
        self.max_batch_history = 1000
        
        # HTTP客户端用于与智能体通信
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # 操作监听器
        self.operation_listeners: List[Callable[[OperationResult], None]] = []
        
        # 性能指标
        self.metrics = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "batch_operations": 0,
            "avg_operation_time": 0.0
        }
        
        self.logger.info("LifecycleManager initialized")
    
    async def shutdown(self):
        """关闭生命周期管理器"""
        try:
            await self.http_client.aclose()
            self.logger.info("LifecycleManager shutdown completed")
        except Exception as e:
            self.logger.error(f"Error during LifecycleManager shutdown: {e}")
    
    # 智能体注册管理
    async def register_agent(
        self, 
        agent_info: AgentInfo,
        auto_start: bool = True
    ) -> OperationResult:
        """注册智能体"""
        
        operation_id = f"register-{agent_info.agent_id}-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # 验证智能体信息
            if not await self._validate_agent_info(agent_info):
                return self._create_operation_result(
                    False, AgentOperation.REGISTER, agent_info.agent_id,
                    "Invalid agent information", {}, start_time, operation_id
                )
            
            # 检查是否已存在
            existing = await self.cluster_manager.get_agent_info(agent_info.agent_id)
            if existing:
                return self._create_operation_result(
                    False, AgentOperation.REGISTER, agent_info.agent_id,
                    "Agent already exists", {"existing_agent": True}, start_time, operation_id
                )
            
            # 注册到集群管理器
            success = await self.cluster_manager.register_agent(agent_info)
            
            if success:
                result = self._create_operation_result(
                    True, AgentOperation.REGISTER, agent_info.agent_id,
                    "Agent registered successfully", 
                    {"agent_info": asdict(agent_info)}, start_time, operation_id
                )
                
                # 如果需要自动启动
                if auto_start:
                    start_result = await self.start_agent(agent_info.agent_id)
                    result.details["auto_start_result"] = asdict(start_result)
                
                return result
            else:
                return self._create_operation_result(
                    False, AgentOperation.REGISTER, agent_info.agent_id,
                    "Failed to register agent", {}, start_time, operation_id
                )
                
        except Exception as e:
            self.logger.error(f"Error registering agent {agent_info.agent_id}: {e}")
            return self._create_operation_result(
                False, AgentOperation.REGISTER, agent_info.agent_id,
                f"Registration error: {str(e)}", {"error": str(e)}, start_time, operation_id
            )
    
    async def unregister_agent(
        self, 
        agent_id: str,
        force: bool = False
    ) -> OperationResult:
        """注销智能体"""
        
        operation_id = f"unregister-{agent_id}-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # 获取智能体信息
            agent = await self.cluster_manager.get_agent_info(agent_id)
            if not agent:
                return self._create_operation_result(
                    False, AgentOperation.UNREGISTER, agent_id,
                    "Agent not found", {}, start_time, operation_id
                )
            
            # 如果智能体正在运行，先停止
            if agent.status == AgentStatus.RUNNING and not force:
                stop_result = await self.stop_agent(agent_id, graceful=True)
                if not stop_result.success:
                    return self._create_operation_result(
                        False, AgentOperation.UNREGISTER, agent_id,
                        "Failed to stop agent before unregistration",
                        {"stop_result": asdict(stop_result)}, start_time, operation_id
                    )
            
            # 从集群管理器注销
            success = await self.cluster_manager.unregister_agent(agent_id)
            
            if success:
                return self._create_operation_result(
                    True, AgentOperation.UNREGISTER, agent_id,
                    "Agent unregistered successfully", 
                    {"force": force}, start_time, operation_id
                )
            else:
                return self._create_operation_result(
                    False, AgentOperation.UNREGISTER, agent_id,
                    "Failed to unregister agent", {}, start_time, operation_id
                )
                
        except Exception as e:
            self.logger.error(f"Error unregistering agent {agent_id}: {e}")
            return self._create_operation_result(
                False, AgentOperation.UNREGISTER, agent_id,
                f"Unregistration error: {str(e)}", {"error": str(e)}, start_time, operation_id
            )
    
    # 智能体操作管理
    async def start_agent(self, agent_id: str) -> OperationResult:
        """启动智能体"""
        
        return await self._perform_agent_operation(
            agent_id, 
            AgentOperation.START,
            self._execute_start_operation
        )
    
    async def stop_agent(self, agent_id: str, graceful: bool = True) -> OperationResult:
        """停止智能体"""
        
        return await self._perform_agent_operation(
            agent_id,
            AgentOperation.STOP,
            lambda agent: self._execute_stop_operation(agent, graceful)
        )
    
    async def restart_agent(self, agent_id: str) -> OperationResult:
        """重启智能体"""
        
        return await self._perform_agent_operation(
            agent_id,
            AgentOperation.RESTART,
            self._execute_restart_operation
        )
    
    async def upgrade_agent(
        self, 
        agent_id: str, 
        target_version: str,
        config_updates: Optional[Dict[str, Any]] = None
    ) -> OperationResult:
        """升级智能体"""
        
        async def upgrade_operation(agent_info: AgentInfo):
            return await self._execute_upgrade_operation(
                agent_info, 
                target_version, 
                config_updates
            )
        
        return await self._perform_agent_operation(
            agent_id,
            AgentOperation.UPGRADE,
            upgrade_operation
        )
    
    async def health_check_agent(self, agent_id: str) -> OperationResult:
        """对智能体进行健康检查"""
        
        return await self._perform_agent_operation(
            agent_id,
            AgentOperation.HEALTH_CHECK,
            self._execute_health_check_operation
        )
    
    # 批量操作
    async def batch_operation(
        self, 
        agent_ids: List[str], 
        operation: AgentOperation,
        params: Optional[Dict[str, Any]] = None,
        max_concurrent: int = 10
    ) -> BatchOperationResult:
        """批量操作智能体"""
        
        batch_id = f"batch-{operation.value}-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        self.logger.info(f"Starting batch operation {batch_id}: {operation.value} on {len(agent_ids)} agents")
        
        # 限制并发数量
        semaphore = asyncio.Semaphore(max_concurrent)
        results: Dict[str, OperationResult] = {}
        
        async def perform_single_operation(agent_id: str):
            async with semaphore:
                try:
                    if operation == AgentOperation.START:
                        result = await self.start_agent(agent_id)
                    elif operation == AgentOperation.STOP:
                        graceful = params.get("graceful", True) if params else True
                        result = await self.stop_agent(agent_id, graceful)
                    elif operation == AgentOperation.RESTART:
                        result = await self.restart_agent(agent_id)
                    elif operation == AgentOperation.UPGRADE:
                        version = params.get("version") if params else None
                        config = params.get("config") if params else None
                        if version:
                            result = await self.upgrade_agent(agent_id, version, config)
                        else:
                            result = OperationResult(
                                success=False,
                                operation=operation,
                                agent_id=agent_id,
                                message="Missing version parameter",
                                details={},
                                timestamp=time.time(),
                                duration_ms=0,
                                operation_id=f"batch-{agent_id}"
                            )
                    elif operation == AgentOperation.HEALTH_CHECK:
                        result = await self.health_check_agent(agent_id)
                    else:
                        result = OperationResult(
                            success=False,
                            operation=operation,
                            agent_id=agent_id,
                            message=f"Unsupported operation: {operation.value}",
                            details={},
                            timestamp=time.time(),
                            duration_ms=0,
                            operation_id=f"batch-{agent_id}"
                        )
                    
                    results[agent_id] = result
                    
                except Exception as e:
                    self.logger.error(f"Batch operation failed for agent {agent_id}: {e}")
                    results[agent_id] = OperationResult(
                        success=False,
                        operation=operation,
                        agent_id=agent_id,
                        message=f"Operation error: {str(e)}",
                        details={"error": str(e)},
                        timestamp=time.time(),
                        duration_ms=0,
                        operation_id=f"batch-{agent_id}"
                    )
        
        # 创建任务列表并执行
        tasks = [perform_single_operation(agent_id) for agent_id in agent_ids]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # 计算结果统计
        completed_time = time.time()
        success_count = sum(1 for result in results.values() if result.success)
        failed_count = len(results) - success_count
        
        batch_result = BatchOperationResult(
            total_count=len(agent_ids),
            success_count=success_count,
            failed_count=failed_count,
            results=results,
            operation_type=operation,
            started_at=start_time,
            completed_at=completed_time,
            batch_id=batch_id
        )
        
        # 记录批量操作历史
        self.batch_history.append(batch_result)
        if len(self.batch_history) > self.max_batch_history:
            self.batch_history = self.batch_history[-self.max_batch_history:]
        
        # 更新指标
        self.metrics["batch_operations"] += 1
        
        self.logger.info(
            f"Batch operation {batch_id} completed: "
            f"{success_count}/{len(agent_ids)} successful, "
            f"took {batch_result.duration_seconds:.2f}s"
        )
        
        return batch_result
    
    # 操作历史和审计
    async def get_operation_history(
        self, 
        agent_id: Optional[str] = None,
        operation_type: Optional[AgentOperation] = None,
        limit: int = 100
    ) -> List[OperationResult]:
        """获取操作历史"""
        
        filtered_history = self.operation_history
        
        # 按智能体ID过滤
        if agent_id:
            filtered_history = [
                op for op in filtered_history 
                if op.agent_id == agent_id
            ]
        
        # 按操作类型过滤
        if operation_type:
            filtered_history = [
                op for op in filtered_history 
                if op.operation == operation_type
            ]
        
        # 按时间降序排序并限制数量
        filtered_history.sort(key=lambda x: x.timestamp, reverse=True)
        return filtered_history[:limit]
    
    async def get_batch_history(self, limit: int = 50) -> List[BatchOperationResult]:
        """获取批量操作历史"""
        sorted_history = sorted(self.batch_history, key=lambda x: x.started_at, reverse=True)
        return sorted_history[:limit]
    
    async def get_operation_metrics(self) -> Dict[str, Any]:
        """获取操作指标"""
        return {
            **self.metrics,
            "history_size": len(self.operation_history),
            "batch_history_size": len(self.batch_history),
            "success_rate": (
                self.metrics["successful_operations"] / self.metrics["total_operations"]
                if self.metrics["total_operations"] > 0 else 0
            )
        }
    
    # 内部方法
    async def _perform_agent_operation(
        self, 
        agent_id: str, 
        operation: AgentOperation,
        operation_func: Callable
    ) -> OperationResult:
        """执行智能体操作的通用方法"""
        
        operation_id = f"{operation.value}-{agent_id}-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        # 获取操作锁
        if agent_id not in self.operation_locks:
            self.operation_locks[agent_id] = asyncio.Lock()
        
        async with self.operation_locks[agent_id]:
            try:
                # 获取智能体信息
                agent_info = await self.cluster_manager.get_agent_info(agent_id)
                if not agent_info:
                    return self._create_operation_result(
                        False, operation, agent_id,
                        "Agent not found", {}, start_time, operation_id
                    )
                
                # 执行操作
                result = await operation_func(agent_info)
                
                # 记录操作历史
                self._record_operation_result(result)
                
                return result
                
            except Exception as e:
                self.logger.error(f"Error in {operation.value} operation for agent {agent_id}: {e}")
                result = self._create_operation_result(
                    False, operation, agent_id,
                    f"Operation error: {str(e)}", {"error": str(e)}, start_time, operation_id
                )
                self._record_operation_result(result)
                return result
    
    async def _execute_start_operation(self, agent_info: AgentInfo) -> OperationResult:
        """执行启动操作"""
        
        operation_id = f"start-{agent_info.agent_id}-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            if agent_info.agent_id == "api":
                try:
                    resp = await self.http_client.get(f"{agent_info.endpoint}/health", timeout=10.0)
                    if resp.status_code == 200:
                        await self.cluster_manager.update_agent_status(
                            agent_info.agent_id, AgentStatus.RUNNING, "本地API健康"
                        )
                        return self._create_operation_result(
                            True, AgentOperation.START, agent_info.agent_id,
                            "本地API已运行",
                            {"endpoint": agent_info.endpoint}, start_time, operation_id
                        )
                    await self.cluster_manager.update_agent_status(
                        agent_info.agent_id, AgentStatus.FAILED, f"本地API健康检查失败: HTTP {resp.status_code}"
                    )
                    return self._create_operation_result(
                        False, AgentOperation.START, agent_info.agent_id,
                        f"本地API健康检查失败: HTTP {resp.status_code}",
                        {"endpoint": agent_info.endpoint, "status_code": resp.status_code}, start_time, operation_id
                    )
                except Exception as e:
                    await self.cluster_manager.update_agent_status(
                        agent_info.agent_id, AgentStatus.FAILED, f"本地API健康检查异常: {str(e)}"
                    )
                    return self._create_operation_result(
                        False, AgentOperation.START, agent_info.agent_id,
                        f"本地API健康检查异常: {str(e)}",
                        {"endpoint": agent_info.endpoint}, start_time, operation_id
                    )

            # 检查当前状态
            if agent_info.status == AgentStatus.RUNNING:
                return self._create_operation_result(
                    True, AgentOperation.START, agent_info.agent_id,
                    "Agent already running", {"status": agent_info.status.value}, 
                    start_time, operation_id
                )
            
            # 更新状态为启动中
            await self.cluster_manager.update_agent_status(
                agent_info.agent_id, AgentStatus.PENDING, "Starting agent"
            )
            
            # 发送启动请求到智能体
            try:
                response = await self.http_client.post(
                    f"{agent_info.endpoint}/start",
                    json={"config": agent_info.config},
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    # 启动成功，更新状态
                    await self.cluster_manager.update_agent_status(
                        agent_info.agent_id, AgentStatus.RUNNING, "Agent started successfully"
                    )
                    
                    return self._create_operation_result(
                        True, AgentOperation.START, agent_info.agent_id,
                        "Agent started successfully", 
                        {"endpoint": agent_info.endpoint, "response": response.json()},
                        start_time, operation_id
                    )
                else:
                    # 启动失败
                    await self.cluster_manager.update_agent_status(
                        agent_info.agent_id, AgentStatus.FAILED, f"Start failed: HTTP {response.status_code}"
                    )
                    
                    return self._create_operation_result(
                        False, AgentOperation.START, agent_info.agent_id,
                        f"Start failed with HTTP {response.status_code}",
                        {"endpoint": agent_info.endpoint, "status_code": response.status_code},
                        start_time, operation_id
                    )
                    
            except httpx.ConnectError:
                # 连接失败，可能智能体未启动或网络问题
                await self.cluster_manager.update_agent_status(
                    agent_info.agent_id, AgentStatus.FAILED, "Connection failed"
                )
                
                return self._create_operation_result(
                    False, AgentOperation.START, agent_info.agent_id,
                    "Failed to connect to agent",
                    {"endpoint": agent_info.endpoint}, start_time, operation_id
                )
                
            except httpx.TimeoutException:
                # 超时
                await self.cluster_manager.update_agent_status(
                    agent_info.agent_id, AgentStatus.FAILED, "Start operation timeout"
                )
                
                return self._create_operation_result(
                    False, AgentOperation.START, agent_info.agent_id,
                    "Start operation timeout",
                    {"endpoint": agent_info.endpoint}, start_time, operation_id
                )
                
        except Exception as e:
            await self.cluster_manager.update_agent_status(
                agent_info.agent_id, AgentStatus.FAILED, f"Start error: {str(e)}"
            )
            
            return self._create_operation_result(
                False, AgentOperation.START, agent_info.agent_id,
                f"Start operation error: {str(e)}",
                {"error": str(e)}, start_time, operation_id
            )
    
    async def _execute_stop_operation(self, agent_info: AgentInfo, graceful: bool = True) -> OperationResult:
        """执行停止操作"""
        
        operation_id = f"stop-{agent_info.agent_id}-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            if agent_info.agent_id == "api":
                return self._create_operation_result(
                    False, AgentOperation.STOP, agent_info.agent_id,
                    "本地API不支持停止",
                    {"endpoint": agent_info.endpoint}, start_time, operation_id
                )

            # 检查当前状态
            if agent_info.status in [AgentStatus.STOPPED, AgentStatus.STOPPING]:
                return self._create_operation_result(
                    True, AgentOperation.STOP, agent_info.agent_id,
                    "Agent already stopped or stopping", 
                    {"status": agent_info.status.value}, start_time, operation_id
                )
            
            # 更新状态为停止中
            await self.cluster_manager.update_agent_status(
                agent_info.agent_id, AgentStatus.STOPPING, "Stopping agent"
            )
            
            # 发送停止请求到智能体
            try:
                endpoint = f"{agent_info.endpoint}/stop"
                if not graceful:
                    endpoint += "?force=true"
                
                response = await self.http_client.post(endpoint, timeout=30.0)
                
                if response.status_code == 200:
                    # 停止成功，更新状态
                    await self.cluster_manager.update_agent_status(
                        agent_info.agent_id, AgentStatus.STOPPED, "Agent stopped successfully"
                    )
                    
                    return self._create_operation_result(
                        True, AgentOperation.STOP, agent_info.agent_id,
                        "Agent stopped successfully",
                        {"graceful": graceful, "endpoint": agent_info.endpoint},
                        start_time, operation_id
                    )
                else:
                    # 停止失败，但可能智能体已经停止
                    await self.cluster_manager.update_agent_status(
                        agent_info.agent_id, AgentStatus.FAILED, f"Stop failed: HTTP {response.status_code}"
                    )
                    
                    return self._create_operation_result(
                        False, AgentOperation.STOP, agent_info.agent_id,
                        f"Stop failed with HTTP {response.status_code}",
                        {"graceful": graceful, "status_code": response.status_code},
                        start_time, operation_id
                    )
                    
            except (httpx.ConnectError, httpx.TimeoutException):
                # 连接失败或超时，可能智能体已经停止
                await self.cluster_manager.update_agent_status(
                    agent_info.agent_id, AgentStatus.STOPPED, "Agent appears to be stopped"
                )
                
                return self._create_operation_result(
                    True, AgentOperation.STOP, agent_info.agent_id,
                    "Agent appears to be stopped (connection failed)",
                    {"graceful": graceful}, start_time, operation_id
                )
                
        except Exception as e:
            return self._create_operation_result(
                False, AgentOperation.STOP, agent_info.agent_id,
                f"Stop operation error: {str(e)}",
                {"error": str(e), "graceful": graceful}, start_time, operation_id
            )
    
    async def _execute_restart_operation(self, agent_info: AgentInfo) -> OperationResult:
        """执行重启操作"""
        
        operation_id = f"restart-{agent_info.agent_id}-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            if agent_info.agent_id == "api":
                try:
                    resp = await self.http_client.get(f"{agent_info.endpoint}/health", timeout=10.0)
                    if resp.status_code == 200:
                        await self.cluster_manager.update_agent_status(
                            agent_info.agent_id, AgentStatus.RUNNING, "本地API健康"
                        )
                        return self._create_operation_result(
                            True, AgentOperation.RESTART, agent_info.agent_id,
                            "本地API无需重启",
                            {"endpoint": agent_info.endpoint}, start_time, operation_id
                        )
                    await self.cluster_manager.update_agent_status(
                        agent_info.agent_id, AgentStatus.FAILED, f"本地API健康检查失败: HTTP {resp.status_code}"
                    )
                    return self._create_operation_result(
                        False, AgentOperation.RESTART, agent_info.agent_id,
                        f"本地API健康检查失败: HTTP {resp.status_code}",
                        {"endpoint": agent_info.endpoint, "status_code": resp.status_code}, start_time, operation_id
                    )
                except Exception as e:
                    await self.cluster_manager.update_agent_status(
                        agent_info.agent_id, AgentStatus.FAILED, f"本地API健康检查异常: {str(e)}"
                    )
                    return self._create_operation_result(
                        False, AgentOperation.RESTART, agent_info.agent_id,
                        f"本地API健康检查异常: {str(e)}",
                        {"endpoint": agent_info.endpoint}, start_time, operation_id
                    )

            # 先停止智能体
            stop_result = await self._execute_stop_operation(agent_info, graceful=True)
            
            if not stop_result.success:
                return self._create_operation_result(
                    False, AgentOperation.RESTART, agent_info.agent_id,
                    "Restart failed: could not stop agent",
                    {"stop_result": asdict(stop_result)}, start_time, operation_id
                )
            
            # 等待一小段时间
            await asyncio.sleep(2.0)
            
            # 重新获取智能体信息（状态可能已更新）
            updated_agent = await self.cluster_manager.get_agent_info(agent_info.agent_id)
            if not updated_agent:
                return self._create_operation_result(
                    False, AgentOperation.RESTART, agent_info.agent_id,
                    "Restart failed: agent not found after stop",
                    {}, start_time, operation_id
                )
            
            # 再启动智能体
            start_result = await self._execute_start_operation(updated_agent)
            
            if start_result.success:
                return self._create_operation_result(
                    True, AgentOperation.RESTART, agent_info.agent_id,
                    "Agent restarted successfully",
                    {
                        "stop_result": asdict(stop_result),
                        "start_result": asdict(start_result)
                    }, start_time, operation_id
                )
            else:
                return self._create_operation_result(
                    False, AgentOperation.RESTART, agent_info.agent_id,
                    "Restart failed: could not start agent",
                    {
                        "stop_result": asdict(stop_result),
                        "start_result": asdict(start_result)
                    }, start_time, operation_id
                )
                
        except Exception as e:
            return self._create_operation_result(
                False, AgentOperation.RESTART, agent_info.agent_id,
                f"Restart operation error: {str(e)}",
                {"error": str(e)}, start_time, operation_id
            )
    
    async def _execute_upgrade_operation(
        self, 
        agent_info: AgentInfo, 
        target_version: str,
        config_updates: Optional[Dict[str, Any]] = None
    ) -> OperationResult:
        """执行升级操作"""
        
        operation_id = f"upgrade-{agent_info.agent_id}-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # 检查版本
            if agent_info.version == target_version:
                return self._create_operation_result(
                    True, AgentOperation.UPGRADE, agent_info.agent_id,
                    "Agent already at target version",
                    {"current_version": agent_info.version, "target_version": target_version},
                    start_time, operation_id
                )
            
            # 更新状态为升级中
            await self.cluster_manager.update_agent_status(
                agent_info.agent_id, AgentStatus.UPGRADING, f"Upgrading to version {target_version}"
            )
            
            # 发送升级请求
            try:
                upgrade_data = {
                    "target_version": target_version,
                    "config_updates": config_updates or {}
                }
                
                response = await self.http_client.post(
                    f"{agent_info.endpoint}/upgrade",
                    json=upgrade_data,
                    timeout=120.0  # 升级可能需要更长时间
                )
                
                if response.status_code == 200:
                    # 升级成功，更新版本信息
                    agent_info.version = target_version
                    if config_updates:
                        agent_info.config.update(config_updates)
                    
                    await self.cluster_manager.update_agent_status(
                        agent_info.agent_id, AgentStatus.RUNNING, f"Upgraded to version {target_version}"
                    )
                    
                    return self._create_operation_result(
                        True, AgentOperation.UPGRADE, agent_info.agent_id,
                        f"Agent upgraded to version {target_version}",
                        {
                            "old_version": agent_info.version,
                            "new_version": target_version,
                            "config_updates": config_updates
                        }, start_time, operation_id
                    )
                else:
                    # 升级失败
                    await self.cluster_manager.update_agent_status(
                        agent_info.agent_id, AgentStatus.FAILED, f"Upgrade failed: HTTP {response.status_code}"
                    )
                    
                    return self._create_operation_result(
                        False, AgentOperation.UPGRADE, agent_info.agent_id,
                        f"Upgrade failed with HTTP {response.status_code}",
                        {
                            "target_version": target_version,
                            "status_code": response.status_code,
                            "response": response.text
                        }, start_time, operation_id
                    )
                    
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                await self.cluster_manager.update_agent_status(
                    agent_info.agent_id, AgentStatus.FAILED, f"Upgrade connection error: {str(e)}"
                )
                
                return self._create_operation_result(
                    False, AgentOperation.UPGRADE, agent_info.agent_id,
                    f"Upgrade connection error: {str(e)}",
                    {"target_version": target_version}, start_time, operation_id
                )
                
        except Exception as e:
            return self._create_operation_result(
                False, AgentOperation.UPGRADE, agent_info.agent_id,
                f"Upgrade operation error: {str(e)}",
                {"error": str(e), "target_version": target_version}, start_time, operation_id
            )
    
    async def _execute_health_check_operation(self, agent_info: AgentInfo) -> OperationResult:
        """执行健康检查操作"""
        
        operation_id = f"health-{agent_info.agent_id}-{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        try:
            # 发送健康检查请求
            response = await self.http_client.get(
                f"{agent_info.endpoint}/health",
                timeout=10.0
            )
            
            if response.status_code == 200:
                health_data = response.json()
                
                # 更新健康状态
                health_check = AgentHealthCheck(
                    is_healthy=health_data.get("healthy", True),
                    last_heartbeat=time.time(),
                    consecutive_failures=0,
                    health_details=health_data
                )
                
                await self.cluster_manager.update_agent_health(agent_info.agent_id, health_check)
                
                return self._create_operation_result(
                    True, AgentOperation.HEALTH_CHECK, agent_info.agent_id,
                    "Health check successful",
                    {"health_data": health_data}, start_time, operation_id
                )
            else:
                # 健康检查失败
                health_check = AgentHealthCheck(
                    is_healthy=False,
                    last_heartbeat=time.time(),
                    consecutive_failures=agent_info.health.consecutive_failures + 1
                )
                
                await self.cluster_manager.update_agent_health(agent_info.agent_id, health_check)
                
                return self._create_operation_result(
                    False, AgentOperation.HEALTH_CHECK, agent_info.agent_id,
                    f"Health check failed with HTTP {response.status_code}",
                    {"status_code": response.status_code}, start_time, operation_id
                )
                
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            # 连接失败或超时
            health_check = AgentHealthCheck(
                is_healthy=False,
                last_heartbeat=agent_info.health.last_heartbeat,  # 保持上次心跳时间
                consecutive_failures=agent_info.health.consecutive_failures + 1
            )
            
            await self.cluster_manager.update_agent_health(agent_info.agent_id, health_check)
            
            return self._create_operation_result(
                False, AgentOperation.HEALTH_CHECK, agent_info.agent_id,
                f"Health check connection error: {str(e)}",
                {"error": str(e)}, start_time, operation_id
            )
        
        except Exception as e:
            return self._create_operation_result(
                False, AgentOperation.HEALTH_CHECK, agent_info.agent_id,
                f"Health check error: {str(e)}",
                {"error": str(e)}, start_time, operation_id
            )
    
    async def _validate_agent_info(self, agent_info: AgentInfo) -> bool:
        """验证智能体信息"""
        try:
            # 基本信息验证
            if not agent_info.agent_id or not agent_info.endpoint:
                return False
            
            # 端点验证
            if not (agent_info.endpoint.startswith("http://") or 
                   agent_info.endpoint.startswith("https://")):
                return False
            
            # 端口验证
            if agent_info.port and not (1 <= agent_info.port <= 65535):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _create_operation_result(
        self, 
        success: bool, 
        operation: AgentOperation, 
        agent_id: str,
        message: str, 
        details: Dict[str, Any], 
        start_time: float, 
        operation_id: str
    ) -> OperationResult:
        """创建操作结果"""
        
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        return OperationResult(
            success=success,
            operation=operation,
            agent_id=agent_id,
            message=message,
            details=details,
            timestamp=end_time,
            duration_ms=duration_ms,
            operation_id=operation_id
        )
    
    def _record_operation_result(self, result: OperationResult):
        """记录操作结果"""
        # 记录到历史
        self.operation_history.append(result)
        
        # 保持历史大小限制
        if len(self.operation_history) > self.max_history_size:
            self.operation_history = self.operation_history[-self.max_history_size:]
        
        # 更新指标
        self.metrics["total_operations"] += 1
        if result.success:
            self.metrics["successful_operations"] += 1
        else:
            self.metrics["failed_operations"] += 1
        
        # 更新平均操作时间
        if self.metrics["total_operations"] > 0:
            old_avg = self.metrics["avg_operation_time"]
            new_avg = (
                (old_avg * (self.metrics["total_operations"] - 1) + result.duration_ms) / 
                self.metrics["total_operations"]
            )
            self.metrics["avg_operation_time"] = new_avg
        
        # 通知监听器
        for listener in self.operation_listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    create_task_with_logging(listener(result))
                else:
                    listener(result)
            except Exception as e:
                self.logger.error(f"Error in operation listener: {e}")
    
    def add_operation_listener(self, listener: Callable[[OperationResult], None]):
        """添加操作监听器"""
        self.operation_listeners.append(listener)
    
    def remove_operation_listener(self, listener: Callable[[OperationResult], None]):
        """移除操作监听器"""
        if listener in self.operation_listeners:
            self.operation_listeners.remove(listener)
