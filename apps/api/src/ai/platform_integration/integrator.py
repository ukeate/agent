"""平台集成器核心实现"""

import asyncio
import json
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import redis
import aiohttp

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
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 健康监控任务
        self._health_monitor_task = None
    
    def create_app(self) -> FastAPI:
        """创建FastAPI应用"""
        app = FastAPI(
            title="Model Fine-tuning Platform",
            description="Integrated platform for model fine-tuning and optimization",
            version="1.0.0"
        )
        
        self._setup_routes(app)
        
        @app.on_event("startup")
        async def startup_event():
            """启动事件"""
            await self.start_health_monitor()
        
        @app.on_event("shutdown")
        async def shutdown_event():
            """关闭事件"""
            await self.stop_health_monitor()
        
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
                    registered_at=datetime.now(),
                    last_heartbeat=datetime.now()
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
                workflow_id = f"workflow_{int(datetime.now().timestamp())}"
                
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
            api_endpoint=component.api_endpoint,
            metadata=component.metadata,
            registered_at=datetime.now(),
            last_heartbeat=datetime.now()
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
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    component_info.health_endpoint, 
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.warning(f"Health check failed for {component_info.component_id}: {e}")
            return False
    
    async def _save_component_to_redis(self, component_info: ComponentInfo):
        """保存组件信息到Redis"""
        key = f"component:{component_info.component_id}"
        value = json.dumps(component_info.to_dict())
        self.redis_client.setex(key, 3600, value)  # 1小时过期
    
    async def start_health_monitor(self):
        """启动健康监控"""
        if self._health_monitor_task is None:
            self._health_monitor_task = asyncio.create_task(self._health_monitor_loop())
            self.logger.info("Health monitor started")
    
    async def stop_health_monitor(self):
        """停止健康监控"""
        if self._health_monitor_task:
            self._health_monitor_task.cancel()
            try:
                await self._health_monitor_task
            except asyncio.CancelledError:
                pass
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
                        component_info.last_heartbeat = datetime.now()
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
            timestamp=datetime.now()
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
            started_at=datetime.now(),
            current_step=None
        )
        
        try:
            if workflow.workflow_type == "full_fine_tuning":
                # 完整的微调工作流
                steps = [
                    "data_preparation",
                    "hyperparameter_optimization", 
                    "fine_tuning",
                    "evaluation",
                    "compression",
                    "model_deployment"
                ]
                
                for step in steps:
                    workflow_state.current_step = step
                    
                    step_result = await self._execute_workflow_step(
                        step, 
                        workflow.parameters
                    )
                    
                    workflow_step = WorkflowStep(
                        step_name=step,
                        status=WorkflowStatus.COMPLETED if step_result["success"] else WorkflowStatus.FAILED,
                        started_at=datetime.now(),
                        completed_at=datetime.now(),
                        result=step_result.get("result"),
                        error=step_result.get("error")
                    )
                    
                    workflow_state.steps.append(workflow_step)
                    
                    if not step_result["success"]:
                        workflow_state.status = WorkflowStatus.FAILED
                        break
                
                if workflow_state.status != WorkflowStatus.FAILED:
                    workflow_state.status = WorkflowStatus.COMPLETED
            
            elif workflow.workflow_type == "model_optimization":
                # 模型优化工作流
                steps = ["compression", "quantization", "evaluation"]
                
                for step in steps:
                    workflow_state.current_step = step
                    
                    step_result = await self._execute_workflow_step(
                        step,
                        workflow.parameters
                    )
                    
                    workflow_step = WorkflowStep(
                        step_name=step,
                        status=WorkflowStatus.COMPLETED if step_result["success"] else WorkflowStatus.FAILED,
                        started_at=datetime.now(),
                        completed_at=datetime.now(),
                        result=step_result.get("result"),
                        error=step_result.get("error")
                    )
                    
                    workflow_state.steps.append(workflow_step)
            
            workflow_state.completed_at = datetime.now()
            
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
        
        self.logger.info(f"Executing workflow step: {step}")
        
        try:
            # 这里模拟调用各个组件服务
            # 实际实现中需要根据组件类型调用对应的API
            
            if step == "data_preparation":
                # 调用数据管理服务
                await asyncio.sleep(1)  # 模拟处理时间
                result = {"dataset_id": "dataset_001", "status": "prepared"}
                
            elif step == "hyperparameter_optimization":
                # 调用超参数优化服务
                await asyncio.sleep(2)  # 模拟处理时间
                result = {"best_params": {"learning_rate": 0.001, "batch_size": 32}}
                
            elif step == "fine_tuning":
                # 调用微调服务
                await asyncio.sleep(3)  # 模拟处理时间
                result = {"model_id": "model_001", "metrics": {"accuracy": 0.95}}
                
            elif step == "evaluation":
                # 调用评估服务
                await asyncio.sleep(1)  # 模拟处理时间
                result = {"evaluation_score": 0.92, "benchmark_results": {}}
                
            elif step == "compression":
                # 调用压缩服务
                await asyncio.sleep(2)  # 模拟处理时间
                result = {"compressed_model_id": "model_001_compressed", "size_reduction": "75%"}
                
            elif step == "model_deployment":
                # 调用模型服务
                await asyncio.sleep(1)  # 模拟处理时间
                result = {"deployment_id": "deploy_001", "endpoint": "http://api.example.com/model"}
                
            elif step == "quantization":
                # 量化处理
                await asyncio.sleep(2)  # 模拟处理时间
                result = {"quantized_model_id": "model_001_quantized", "precision": "int8"}
                
            else:
                raise ValueError(f"Unknown workflow step: {step}")
            
            return {
                "success": True,
                "result": result,
                "step": step
            }
            
        except Exception as e:
            self.logger.error(f"Step {step} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "step": step
            }
    
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
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """调用组件API"""
        
        url = f"{component.api_endpoint.rstrip('/')}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url, 
                json=data, 
                timeout=aiohttp.ClientTimeout(total=300)
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Component API call failed: {error_text}"
                    )
    
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