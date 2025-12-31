"""
工作流服务层
统一的工作流业务逻辑管理
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
import uuid
from src.ai.langgraph.state import MessagesState, create_initial_state
from src.ai.langgraph.state_graph import LangGraphWorkflowBuilder, create_simple_workflow, create_conditional_workflow
from langgraph.graph import START, END
from src.ai.langgraph.checkpoints import checkpoint_manager
from src.repositories.workflow_repository import WorkflowRepository
from src.models.schemas.workflow import WorkflowCreate, WorkflowUpdate, WorkflowResponse

from src.core.logging import get_logger
logger = get_logger(__name__)

class WorkflowService:
    """工作流服务"""
    
    def __init__(self, repository: WorkflowRepository = None):
        self.repository = repository or WorkflowRepository()
        self.active_workflows: Dict[str, LangGraphWorkflowBuilder] = {}

    def _definition_to_dict(self, definition: Any) -> Optional[Dict[str, Any]]:
        if not definition:
            return None
        if isinstance(definition, dict):
            return definition
        if hasattr(definition, "model_dump"):
            return definition.model_dump()
        return None

    def _default_definition(self, workflow_type: str) -> Dict[str, Any]:
        if workflow_type == "simple":
            return {
                "nodes": [
                    {"id": "start", "name": "开始", "type": "start", "position": {"x": 100, "y": 100}},
                    {"id": "process", "name": "数据处理", "type": "process", "position": {"x": 350, "y": 100}},
                    {"id": "end", "name": "结束", "type": "end", "position": {"x": 600, "y": 100}},
                ],
                "edges": [
                    {"from": "start", "to": "process"},
                    {"from": "process", "to": "end"},
                ],
            }
        if workflow_type == "conditional":
            return {
                "nodes": [
                    {"id": "start", "name": "开始", "type": "start", "position": {"x": 100, "y": 100}},
                    {"id": "process", "name": "数据处理", "type": "process", "position": {"x": 300, "y": 100}},
                    {"id": "decision", "name": "条件判断", "type": "decision", "position": {"x": 500, "y": 100}},
                    {"id": "path_a", "name": "路径A", "type": "process", "position": {"x": 700, "y": 50}},
                    {"id": "path_b", "name": "路径B", "type": "process", "position": {"x": 700, "y": 150}},
                    {"id": "end", "name": "结束", "type": "end", "position": {"x": 900, "y": 100}},
                ],
                "edges": [
                    {"from": "start", "to": "process"},
                    {"from": "process", "to": "decision"},
                    {"from": "decision", "to": "path_a", "label": "高质量"},
                    {"from": "decision", "to": "path_b", "label": "低质量"},
                    {"from": "path_a", "to": "end"},
                    {"from": "path_b", "to": "end"},
                ],
            }
        return {}

    def _apply_definition(self, builder: LangGraphWorkflowBuilder, definition: Dict[str, Any]) -> None:
        if not definition:
            return
        nodes = definition.get("nodes")
        edges = definition.get("edges")
        steps = definition.get("steps")
        if isinstance(nodes, list) and isinstance(edges, list):
            node_ids = set()
            for node_def in nodes:
                if isinstance(node_def, dict):
                    node_id = node_def.get("id") or node_def.get("node_id")
                    if node_id:
                        node_ids.add(str(node_id))
                        builder.add_node(str(node_id), lambda state: state)
            graph = builder.build()
            for edge_def in edges:
                if isinstance(edge_def, dict):
                    from_node = edge_def.get("from") or edge_def.get("source")
                    to_node = edge_def.get("to") or edge_def.get("target")
                    if from_node and to_node:
                        graph.add_edge(str(from_node), str(to_node))
            if "start" in node_ids:
                graph.add_edge(START, "start")
            if "end" in node_ids:
                graph.add_edge("end", END)
            return
        if isinstance(steps, list):
            node_ids = set()
            for step in steps:
                if isinstance(step, dict):
                    step_id = step.get("id")
                    if step_id:
                        node_ids.add(str(step_id))
                        builder.add_node(str(step_id), lambda state: state)
            graph = builder.build()
            for step in steps:
                if isinstance(step, dict):
                    step_id = step.get("id")
                    deps = step.get("dependencies") or []
                    if step_id and isinstance(deps, list):
                        for dep in deps:
                            graph.add_edge(str(dep), str(step_id))
            if "start" in node_ids:
                graph.add_edge(START, "start")
            if "end" in node_ids:
                graph.add_edge("end", END)
    
    async def create_workflow(self, workflow_data: WorkflowCreate) -> WorkflowResponse:
        """创建新工作流"""
        try:
            workflow_id = str(uuid.uuid4())
            definition_payload = self._definition_to_dict(workflow_data.definition)
            if not definition_payload:
                definition_payload = self._default_definition(workflow_data.workflow_type)
            if workflow_data.workflow_type not in ["simple", "conditional"] and not definition_payload:
                raise ValueError("自定义工作流必须提供definition")
            
            # 创建初始状态
            initial_state = create_initial_state(workflow_id)
            initial_state["metadata"]["name"] = workflow_data.name
            initial_state["metadata"]["description"] = workflow_data.description
            initial_state["metadata"]["type"] = workflow_data.workflow_type
            
            # 根据类型创建工作流构建器
            if workflow_data.workflow_type == "simple":
                builder = create_simple_workflow()
            elif workflow_data.workflow_type == "conditional":
                builder = create_conditional_workflow()
            else:
                # 自定义工作流
                builder = LangGraphWorkflowBuilder()
                self._apply_definition(builder, definition_payload)
            
            # 保存到活跃工作流
            self.active_workflows[workflow_id] = builder
            
            # 保存到数据库
            await self.repository.create_workflow({
                "id": workflow_id,
                "name": workflow_data.name,
                "description": workflow_data.description,
                "workflow_type": workflow_data.workflow_type,
                "definition": definition_payload or {},
                "status": "created",
                "created_at": utc_now()
            })
            
            # 创建初始检查点
            await checkpoint_manager.create_checkpoint(
                workflow_id=workflow_id,
                state=initial_state,
                metadata={"type": "initial_checkpoint"}
            )
            
            return WorkflowResponse(
                id=workflow_id,
                name=workflow_data.name,
                description=workflow_data.description,
                workflow_type=workflow_data.workflow_type,
                definition=definition_payload or {},
                status="created",
                created_at=utc_now()
            )
            
        except Exception as e:
            raise RuntimeError(f"创建工作流失败: {str(e)}")
    
    async def start_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None) -> WorkflowResponse:
        """启动工作流执行"""
        try:
            # 获取工作流构建器
            builder = self.active_workflows.get(workflow_id)
            workflow_data = None
            if not builder:
                # 从数据库恢复
                workflow_data = await self.repository.get_workflow(workflow_id)
                if not workflow_data:
                    raise ValueError(f"工作流不存在: {workflow_id}")
                
                # 重新创建构建器
                if workflow_data["workflow_type"] == "simple":
                    builder = create_simple_workflow()
                elif workflow_data["workflow_type"] == "conditional":
                    builder = create_conditional_workflow()
                else:
                    builder = LangGraphWorkflowBuilder()
                    self._apply_definition(builder, workflow_data.get("definition") or {})
                
                self.active_workflows[workflow_id] = builder
            if workflow_data is None:
                workflow_data = await self.repository.get_workflow(workflow_id)
                if not workflow_data:
                    raise ValueError(f"工作流不存在: {workflow_id}")
            definition_payload = workflow_data.get("definition") or {}
            if not definition_payload:
                definition_payload = self._default_definition(workflow_data.get("workflow_type") or "")
            
            # 获取初始状态
            latest_checkpoint = await checkpoint_manager.get_latest_checkpoint(workflow_id)
            initial_state = latest_checkpoint.state if latest_checkpoint else None
            if not initial_state:
                initial_state = create_initial_state(workflow_id)
            
            # 添加输入数据
            if input_data:
                initial_state["context"]["input"] = input_data
                initial_state["messages"].append({
                    "role": "user",
                    "content": f"工作流输入: {input_data}",
                    "timestamp": utc_now().isoformat()
                })
            
            # 更新状态为运行中
            initial_state["metadata"]["status"] = "running"
            initial_state["metadata"]["started_at"] = utc_now().isoformat()
            
            # 异步执行工作流
            create_task_with_logging(self._execute_workflow(workflow_id, builder, initial_state))
            
            # 更新数据库状态
            await self.repository.update_workflow(workflow_id, {
                "status": "running",
                "started_at": utc_now()
            })
            
            return WorkflowResponse(
                id=workflow_id,
                status="running",
                definition=definition_payload,
                current_state=initial_state
            )
            
        except Exception as e:
            await self.repository.update_workflow(workflow_id, {
                "status": "failed",
                "error_message": str(e)
            })
            raise RuntimeError(f"启动工作流失败: {str(e)}")
    
    async def _execute_workflow(self, workflow_id: str, builder: LangGraphWorkflowBuilder, initial_state: MessagesState):
        """内部工作流执行方法"""
        try:
            result = await builder.execute(initial_state)
            
            # 更新数据库状态
            await self.repository.update_workflow(workflow_id, {
                "status": "completed",
                "completed_at": utc_now()
            })
            
            # 创建完成检查点
            await checkpoint_manager.create_checkpoint(
                workflow_id=workflow_id,
                state=result,
                metadata={"type": "completion_checkpoint"}
            )
            
        except Exception as e:
            # 更新错误状态
            await self.repository.update_workflow(workflow_id, {
                "status": "failed",
                "error_message": str(e),
                "failed_at": utc_now()
            })
    
    async def get_workflow_status(self, workflow_id: str) -> WorkflowResponse:
        """获取工作流状态"""
        try:
            # 从数据库获取基本信息
            workflow_data = await self.repository.get_workflow(workflow_id)
            if not workflow_data:
                raise ValueError(f"工作流不存在: {workflow_id}")
            definition_payload = workflow_data.get("definition") or {}
            if not definition_payload:
                definition_payload = self._default_definition(workflow_data.get("workflow_type") or "")
            
            # 获取最新检查点
            latest_checkpoint = await checkpoint_manager.get_latest_checkpoint(workflow_id)
            current_state = latest_checkpoint.state if latest_checkpoint else None
            
            return WorkflowResponse(
                id=workflow_id,
                name=workflow_data["name"],
                description=workflow_data["description"],
                workflow_type=workflow_data["workflow_type"],
                definition=definition_payload,
                status=workflow_data["status"],
                current_state=current_state,
                created_at=workflow_data["created_at"],
                started_at=workflow_data.get("started_at"),
                completed_at=workflow_data.get("completed_at"),
                error_message=workflow_data.get("error_message")
            )
            
        except Exception as e:
            raise RuntimeError(f"获取工作流状态失败: {str(e)}")
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """暂停工作流"""
        try:
            builder = self.active_workflows.get(workflow_id)
            if not builder:
                return False
            
            success = await builder.pause_workflow(workflow_id)
            if success:
                await self.repository.update_workflow(workflow_id, {
                    "status": "paused",
                    "paused_at": utc_now()
                })
            
            return success
            
        except Exception as e:
            logger.error("暂停工作流失败", error=str(e), workflow_id=workflow_id, exc_info=True)
            return False
    
    async def resume_workflow(self, workflow_id: str) -> bool:
        """恢复工作流"""
        try:
            builder = self.active_workflows.get(workflow_id)
            if not builder:
                # 重新创建构建器
                workflow_data = await self.repository.get_workflow(workflow_id)
                if not workflow_data:
                    return False
                
                if workflow_data["workflow_type"] == "simple":
                    builder = create_simple_workflow()
                elif workflow_data["workflow_type"] == "conditional":
                    builder = create_conditional_workflow()
                else:
                    builder = LangGraphWorkflowBuilder()
                
                self.active_workflows[workflow_id] = builder
            
            result = await builder.resume_workflow(workflow_id)
            if result:
                await self.repository.update_workflow(workflow_id, {
                    "status": "running",
                    "resumed_at": utc_now()
                })
                return True
            
            return False
            
        except Exception as e:
            logger.error("恢复工作流失败", error=str(e), workflow_id=workflow_id, exc_info=True)
            return False
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流"""
        try:
            builder = self.active_workflows.get(workflow_id)
            if builder:
                success = await builder.cancel_workflow(workflow_id)
            else:
                success = True  # 如果不在活跃列表中，直接标记为取消
            
            if success:
                await self.repository.update_workflow(workflow_id, {
                    "status": "cancelled",
                    "cancelled_at": utc_now()
                })
                
                # 从活跃工作流中移除
                if workflow_id in self.active_workflows:
                    del self.active_workflows[workflow_id]
            
            return success
            
        except Exception as e:
            logger.error("取消工作流失败", error=str(e), workflow_id=workflow_id, exc_info=True)
            return False
    
    async def list_workflows(self, status: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[WorkflowResponse]:
        """列出工作流"""
        try:
            workflows = await self.repository.list_workflows(status, limit, offset)
            return [
                WorkflowResponse(
                    id=wf["id"],
                    name=wf["name"],
                    description=wf["description"],
                    workflow_type=wf["workflow_type"],
                    definition=wf.get("definition") or self._default_definition(wf.get("workflow_type") or ""),
                    status=wf["status"],
                    created_at=wf["created_at"],
                    started_at=wf.get("started_at"),
                    completed_at=wf.get("completed_at"),
                    error_message=wf.get("error_message")
                )
                for wf in workflows
            ]
            
        except Exception as e:
            raise RuntimeError(f"列出工作流失败: {str(e)}")
    
    async def get_workflow_checkpoints(self, workflow_id: str) -> List[Dict[str, Any]]:
        """获取工作流检查点列表"""
        try:
            checkpoints = await checkpoint_manager.storage.list_checkpoints(workflow_id)
            return [
                {
                    "id": cp.id,
                    "workflow_id": cp.workflow_id,
                    "created_at": cp.created_at.isoformat(),
                    "version": cp.version,
                    "metadata": cp.metadata
                }
                for cp in checkpoints
            ]
            
        except Exception as e:
            raise RuntimeError(f"获取工作流检查点失败: {str(e)}")
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """删除工作流（软删除）"""
        try:
            # 确保工作流已停止
            if workflow_id in self.active_workflows:
                await self.cancel_workflow(workflow_id)
            
            # 执行软删除
            result = await self.repository.delete_workflow(workflow_id)
            
            # 清理检查点数据
            await checkpoint_manager.storage.cleanup_old_checkpoints(workflow_id, keep_count=0)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"删除工作流失败: {str(e)}")

# 全局工作流服务实例
workflow_service = WorkflowService()
