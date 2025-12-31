"""平台集成器核心实现"""

import asyncio
import json
import time
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
import aiohttp
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
from .models import (
    ComponentType,
    ComponentStatus,
    ComponentInfo,
    ComponentRegistration,
    WorkflowRequest,
    WorkflowStatus,
    WorkflowState,
    WorkflowStep,
    PlatformHealthStatus
)

class PlatformIntegrator:
    """平台集成器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis_client = redis.Redis(
            host=config.get('redis_host', 'localhost'),
            port=config.get('redis_port', 6379),
            db=config.get('redis_db', 0),
            decode_responses=True
        )
        
        # 组件注册表
        self.components: Dict[str, ComponentInfo] = {}
        self.component_dependencies: Dict[str, List[str]] = {}
        
        self.logger = get_logger(__name__)
        
        # 健康监控任务
        self._health_monitor_task = None
        self._load_components_from_redis()

    @staticmethod
    def _normalize_api_endpoint(api_endpoint: str) -> str:
        base = (api_endpoint or "").rstrip("/")
        if base.endswith("/api/v1"):
            base = base[: -len("/api/v1")]
        return base

    def _load_components_from_redis(self) -> None:
        for key in self.redis_client.scan_iter("component:*", count=1000):
            raw = self.redis_client.get(key)
            if not raw:
                continue
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            try:
                registered_at = data.get("registered_at")
                last_heartbeat = data.get("last_heartbeat")
                component_info = ComponentInfo(
                    component_id=str(data.get("component_id") or ""),
                    component_type=ComponentType(str(data.get("component_type") or ComponentType.CUSTOM.value)),
                    name=str(data.get("name") or ""),
                    version=str(data.get("version") or ""),
                    status=ComponentStatus(str(data.get("status") or ComponentStatus.UNHEALTHY.value)),
                    health_endpoint=str(data.get("health_endpoint") or ""),
                    api_endpoint=self._normalize_api_endpoint(str(data.get("api_endpoint") or "")),
                    metadata=data.get("metadata") or {},
                    registered_at=datetime.fromisoformat(registered_at) if registered_at else utc_now(),
                    last_heartbeat=datetime.fromisoformat(last_heartbeat) if last_heartbeat else utc_now(),
                )
                if component_info.component_id:
                    self.components[component_info.component_id] = component_info
            except Exception:
                continue
    
    def create_app(self) -> FastAPI:
        """创建FastAPI应用"""
        @asynccontextmanager
        async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
            await self.start_health_monitor()
            yield
            await self.stop_health_monitor()

        app = FastAPI(
            title="Model Fine-tuning Platform",
            description="Integrated platform for model fine-tuning and optimization",
            version="1.0.0",
            lifespan=lifespan
        )
        
        self._setup_routes(app)
        
        return app
    
    def _setup_routes(self, app: FastAPI):
        """设置API路由"""
        
        @app.post("/platform/components/register")
        async def register_component(component: ComponentRegistration):
            """注册组件"""
            try:
                component_info = ComponentInfo(
                    component_id=component.component_id,
                    component_type=component.component_type,
                    name=component.name,
                    version=component.version,
                    status=ComponentStatus.STARTING,
                    health_endpoint=component.health_endpoint,
                    api_endpoint=component.api_endpoint,
                    metadata=component.metadata,
                    registered_at=utc_now(),
                    last_heartbeat=utc_now()
                )
                
                await self._register_component(component_info)
                
                return {
                    "status": "success",
                    "component_id": component.component_id,
                    "message": "Component registered successfully"
                }
                
            except Exception as e:
                self.logger.error(f"Error registering component: {e}")
                raise HTTPException(status_code=500, detail="Failed to register component")
        
        @app.delete("/platform/components/{component_id}")
        async def unregister_component(component_id: str):
            """注销组件"""
            try:
                await self._unregister_component(component_id)
                return {"status": "success", "message": "Component unregistered"}
            except Exception as e:
                self.logger.error(f"Error unregistering component: {e}")
                raise HTTPException(status_code=500, detail="Failed to unregister component")
        
        @app.get("/platform/components")
        async def list_components():
            """列出所有组件"""
            components = {}
            for comp_id, comp_info in self.components.items():
                components[comp_id] = comp_info.to_dict()
            return {"components": components}
        
        @app.get("/platform/health", response_model=PlatformHealthStatus)
        async def platform_health():
            """平台健康检查"""
            health_status = await self._check_platform_health()
            return health_status
        
        @app.post("/platform/workflows/run")
        async def run_workflow(workflow: WorkflowRequest, background_tasks: BackgroundTasks):
            """执行端到端工作流"""
            try:
                workflow_id = f"workflow_{int(utc_now().timestamp())}"
                
                # 在后台任务中执行工作流
                background_tasks.add_task(
                    self._execute_workflow_background,
                    workflow_id,
                    workflow
                )
                
                return {
                    "workflow_id": workflow_id,
                    "status": "started",
                    "message": "Workflow execution started"
                }
            except Exception as e:
                self.logger.error(f"Workflow execution error: {e}")
                raise HTTPException(status_code=500, detail="Failed to execute workflow")
        
        @app.get("/platform/workflows/{workflow_id}/status")
        async def get_workflow_status(workflow_id: str):
            """获取工作流状态"""
            try:
                status = await self._get_workflow_status(workflow_id)
                return status
            except Exception as e:
                raise HTTPException(status_code=404, detail="Workflow not found")
    
    async def _register_component(self, component_info: ComponentInfo):
        """注册组件内部方法"""
        
        # 检查组件健康状态
        is_healthy = await self._check_component_health(component_info)
        
        if is_healthy:
            component_info.status = ComponentStatus.HEALTHY
        else:
            component_info.status = ComponentStatus.UNHEALTHY
        
        self.components[component_info.component_id] = component_info
        
        # 保存到Redis
        await self._save_component_to_redis(component_info)
        
        self.logger.info(f"Component {component_info.component_id} registered with status {component_info.status}")
    
    async def _register_component_from_registration(self, component: ComponentRegistration) -> ComponentInfo:
        """从ComponentRegistration创建ComponentInfo并注册"""
        component_info = ComponentInfo(
            component_id=component.component_id,
            component_type=component.component_type,
            name=component.name,
            version=component.version,
            status=ComponentStatus.STARTING,
            health_endpoint=component.health_endpoint,
            api_endpoint=self._normalize_api_endpoint(component.api_endpoint),
            metadata=component.metadata,
            registered_at=utc_now(),
            last_heartbeat=utc_now()
        )
        
        await self._register_component(component_info)
        return component_info
    
    async def _unregister_component(self, component_id: str):
        """注销组件"""
        if component_id in self.components:
            # 从内存中删除
            del self.components[component_id]
            
            # 从Redis中删除
            self.redis_client.delete(f"component:{component_id}")
            
            self.logger.info(f"Component {component_id} unregistered")
        else:
            raise ValueError(f"Component {component_id} not found")
    
    async def _check_component_health(self, component_info: ComponentInfo) -> bool:
        """检查组件健康状态"""
        start = time.perf_counter()
        ok = False
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    component_info.health_endpoint,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    ok = response.status == 200
        except Exception as e:
            error = str(e)
            self.logger.warning(f"Health check failed for {component_info.component_id}: {e}")

        elapsed_ms = (time.perf_counter() - start) * 1000
        meta = component_info.metadata or {}
        meta["last_health_check_ms"] = round(elapsed_ms, 2)
        meta["last_health_check_at"] = utc_now().isoformat()
        meta["health_check_total"] = int(meta.get("health_check_total") or 0) + 1
        meta["health_check_failures"] = int(meta.get("health_check_failures") or 0) + (0 if ok else 1)
        if error:
            meta["last_health_check_error"] = error
        else:
            meta.pop("last_health_check_error", None)
        component_info.metadata = meta
        return ok
    
    async def _save_component_to_redis(self, component_info: ComponentInfo):
        """保存组件信息到Redis"""
        key = f"component:{component_info.component_id}"
        value = json.dumps(component_info.to_dict())
        self.redis_client.setex(key, 3600, value)  # 1小时过期
    
    async def start_health_monitor(self):
        """启动健康监控"""
        if self._health_monitor_task is None:
            self._health_monitor_task = create_task_with_logging(self._health_monitor_loop())
            self.logger.info("Health monitor started")
    
    async def stop_health_monitor(self):
        """停止健康监控"""
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                raise
            self._health_monitor_task = None
            self.logger.info("Health monitor stopped")
    
    async def _health_monitor_loop(self):
        """健康监控循环"""
        while True:
            try:
                for component_id, component_info in list(self.components.items()):
                    is_healthy = await self._check_component_health(component_info)
                    
                    if is_healthy:
                        component_info.status = ComponentStatus.HEALTHY
                        component_info.last_heartbeat = utc_now()
                    else:
                        component_info.status = ComponentStatus.UNHEALTHY
                    
                    await self._save_component_to_redis(component_info)
                
                await asyncio.sleep(30)  # 每30秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(30)
    
    async def _check_platform_health(self) -> PlatformHealthStatus:
        """检查平台整体健康状态"""
        
        healthy_count = 0
        total_count = len(self.components)
        
        component_status = {}
        
        for comp_id, comp_info in self.components.items():
            component_status[comp_id] = {
                "status": comp_info.status.value,
                "last_heartbeat": comp_info.last_heartbeat.isoformat(),
                "component_type": comp_info.component_type.value
            }
            
            if comp_info.status == ComponentStatus.HEALTHY:
                healthy_count += 1
        
        overall_health = "healthy" if healthy_count == total_count else "degraded"
        
        if healthy_count == 0:
            overall_health = "critical"
        elif healthy_count < total_count * 0.3:  # 小于30%健康才是unhealthy
            overall_health = "unhealthy"
        
        return PlatformHealthStatus(
            overall_status=overall_health,
            healthy_components=healthy_count,
            total_components=total_count,
            components=component_status,
            timestamp=utc_now()
        )
    
    async def _execute_workflow_background(self, workflow_id: str, workflow: WorkflowRequest):
        """后台执行工作流"""
        try:
            await self._execute_workflow(workflow_id, workflow)
        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
    
    async def _execute_workflow(self, workflow_id: str, workflow: WorkflowRequest) -> Dict[str, Any]:
        """执行端到端工作流"""
        
        self.logger.info(f"Starting workflow {workflow_id}: {workflow.workflow_type}")
        
        # 工作流状态追踪
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            workflow_type=workflow.workflow_type,
            status=WorkflowStatus.RUNNING,
            steps=[],
            parameters=workflow.parameters,
            started_at=utc_now(),
            current_step=None
        )
        await self._save_workflow_state(workflow_id, workflow_state)
        
        try:
            workflow_steps: Dict[str, List[str]] = {
                "full_fine_tuning": [
                    "hyperparameter_optimization",
                    "fine_tuning",
                    "evaluation",
                    "compression",
                    "model_deployment",
                ],
                "model_optimization": ["compression", "evaluation"],
                "evaluation_only": ["evaluation"],
                "data_processing": ["data_processing"],
            }

            steps = workflow_steps.get(workflow.workflow_type, [])
            if not steps:
                raise ValueError("无效工作流类型")

            for step in steps:
                if self._is_workflow_cancelled(workflow_id):
                    workflow_state.status = WorkflowStatus.CANCELLED
                    workflow_state.completed_at = utc_now()
                    workflow_state.error = "cancelled"
                    workflow_state.current_step = None
                    await self._save_workflow_state(workflow_id, workflow_state)
                    return {"workflow_id": workflow_id, "status": workflow_state.status.value}

                workflow_state.current_step = step
                await self._save_workflow_state(workflow_id, workflow_state)

                started_at = utc_now()
                step_result = await self._execute_workflow_step(step, workflow.parameters)
                completed_at = utc_now()

                workflow_step = WorkflowStep(
                    step_name=step,
                    status=WorkflowStatus.COMPLETED if step_result.get("success") else WorkflowStatus.FAILED,
                    started_at=started_at,
                    completed_at=completed_at,
                    result=step_result.get("result"),
                    error=step_result.get("error"),
                )

                workflow_state.steps.append(workflow_step)
                await self._save_workflow_state(workflow_id, workflow_state)

                if not step_result.get("success"):
                    workflow_state.status = WorkflowStatus.FAILED
                    workflow_state.error = step_result.get("error") or "step_failed"
                    break

            if workflow_state.status == WorkflowStatus.RUNNING:
                workflow_state.status = WorkflowStatus.COMPLETED

            workflow_state.completed_at = utc_now()
            workflow_state.current_step = None
            
            # 保存工作流状态到Redis
            await self._save_workflow_state(workflow_id, workflow_state)
            
            return {
                "workflow_id": workflow_id,
                "status": workflow_state.status.value,
                "steps_completed": len([s for s in workflow_state.steps if s.status == WorkflowStatus.COMPLETED]),
                "total_steps": len(workflow_state.steps)
            }
            
        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} failed: {e}")
            
            workflow_state.status = WorkflowStatus.ERROR
            workflow_state.error = str(e)
            
            await self._save_workflow_state(workflow_id, workflow_state)
            
            raise
    
    async def _execute_workflow_step(
        self, 
        step: str, 
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行工作流步骤"""
        step_map: Dict[str, Dict[str, Any]] = {
            "hyperparameter_optimization": {
                "component_type": ComponentType.HYPERPARAMETER,
                "method": "GET",
                "endpoint": "/api/v1/hyperparameter-optimization/health",
            },
            "fine_tuning": {
                "component_type": ComponentType.FINE_TUNING,
                "method": "GET",
                "endpoint": "/api/v1/fine-tuning/jobs",
            },
            "evaluation": {
                "component_type": ComponentType.EVALUATION,
                "method": "GET",
                "endpoint": "/api/v1/model-evaluation/performance/system",
            },
            "compression": {
                "component_type": ComponentType.COMPRESSION,
                "method": "GET",
                "endpoint": "/api/v1/model-compression/health",
            },
            "model_deployment": {
                "component_type": ComponentType.MODEL_SERVICE,
                "method": "GET",
                "endpoint": "/api/v1/model-service/health",
            },
            "data_processing": {
                "component_type": ComponentType.DATA_MANAGEMENT,
                "method": "GET",
                "endpoint": "/api/v1/rag/health",
            },
        }

        cfg = step_map.get(step)
        if not cfg:
            return {"success": False, "error": f"未知步骤: {step}"}

        component = self._get_component_by_type(cfg["component_type"])
        if not component:
            return {"success": False, "error": f"无可用组件: {cfg['component_type'].value}"}

        try:
            result = await self._call_component_api(
                component,
                cfg["endpoint"],
                data=parameters,
                method=cfg["method"],
            )
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _is_workflow_cancelled(self, workflow_id: str) -> bool:
        return bool(self.redis_client.get(f"workflow_cancel:{workflow_id}"))

    async def _cancel_workflow(self, workflow_id: str):
        self.redis_client.setex(f"workflow_cancel:{workflow_id}", 86400, "1")

        key = f"workflow:{workflow_id}"
        value = self.redis_client.get(key)
        if not value:
            return
        try:
            state = json.loads(value)
        except Exception:
            return

        if state.get("status") in {WorkflowStatus.RUNNING.value, WorkflowStatus.PENDING.value}:
            state["status"] = WorkflowStatus.CANCELLED.value
            state["completed_at"] = utc_now().isoformat()
            state["error"] = "cancelled"
            self.redis_client.setex(key, 86400, json.dumps(state, ensure_ascii=False, default=str))
    
    def _get_component_by_type(self, component_type: ComponentType) -> Optional[ComponentInfo]:
        """根据类型获取组件"""
        for component_info in self.components.values():
            if (component_info.component_type == component_type and 
                component_info.status == ComponentStatus.HEALTHY):
                return component_info
        
        return None
    
    async def _call_component_api(
        self,
        component: ComponentInfo,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        method: str = "POST",
    ) -> Dict[str, Any]:
        """调用组件API"""
        
        url = f"{component.api_endpoint.rstrip('/')}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            timeout = aiohttp.ClientTimeout(total=300)
            req_method = method.upper()
            if req_method == "GET":
                async with session.get(url, timeout=timeout) as response:
                    if response.status != 200:
                        raise HTTPException(status_code=response.status, detail=f"Component API call failed: {await response.text()}")
                    return await response.json()

            async with session.request(url=url, method=req_method, json=data or {}, timeout=timeout) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail=f"Component API call failed: {await response.text()}")
                return await response.json()
    
    async def _save_workflow_state(self, workflow_id: str, state: WorkflowState):
        """保存工作流状态"""
        key = f"workflow:{workflow_id}"
        value = state.model_dump_json()
        self.redis_client.setex(key, 86400, value)  # 24小时过期
    
    async def _get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流状态"""
        key = f"workflow:{workflow_id}"
        value = self.redis_client.get(key)
        
        if value:
            return json.loads(value)
        else:
            raise ValueError(f"Workflow {workflow_id} not found")
from src.core.logging import get_logger
