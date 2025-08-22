"""
工作流服务层
统一的工作流业务逻辑管理
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid
import asyncio

from src.ai.langgraph.state import MessagesState, create_initial_state
from src.ai.langgraph.state_graph import LangGraphWorkflowBuilder, create_simple_workflow, create_conditional_workflow
from src.ai.langgraph.checkpoints import checkpoint_manager
from src.repositories.workflow_repository import WorkflowRepository
from src.models.schemas.workflow import WorkflowCreate, WorkflowUpdate, WorkflowResponse


class WorkflowService:
    """工作流服务"""
    
    def __init__(self, repository: WorkflowRepository = None):
        self.repository = repository or WorkflowRepository()
        self.active_workflows: Dict[str, LangGraphWorkflowBuilder] = {}
    
    async def create_workflow(self, workflow_data: WorkflowCreate) -> WorkflowResponse:
        """创建新工作流"""
        try:
            workflow_id = str(uuid.uuid4())
            
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
                
                # 从定义中构建自定义工作流
                if workflow_data.definition and isinstance(workflow_data.definition, dict):
                    nodes = workflow_data.definition.get("nodes", [])
                    edges = workflow_data.definition.get("edges", [])
                    
                    # 添加节点
                    for node_def in nodes:
                        if isinstance(node_def, dict) and "id" in node_def:
                            builder.add_node(node_def["id"], lambda state: state)
                    
                    # 添加边
                    for edge_def in edges:
                        if isinstance(edge_def, dict) and "from" in edge_def and "to" in edge_def:
                            builder.add_edge(edge_def["from"], edge_def["to"])
            
            # 保存到活跃工作流
            self.active_workflows[workflow_id] = builder
            
            # 保存到数据库
            await self.repository.create_workflow({
                "id": workflow_id,
                "name": workflow_data.name,
                "description": workflow_data.description,
                "workflow_type": workflow_data.workflow_type,
                "definition": workflow_data.definition or {},
                "status": "created",
                "created_at": datetime.now()
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
                status="created",
                created_at=datetime.now()
            )
            
        except Exception as e:
            raise RuntimeError(f"创建工作流失败: {str(e)}")
    
    async def start_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None) -> WorkflowResponse:
        """启动工作流执行"""
        try:
            # 获取工作流构建器
            builder = self.active_workflows.get(workflow_id)
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
                
                self.active_workflows[workflow_id] = builder
            
            # 获取初始状态
            initial_state = await checkpoint_manager.restore_from_checkpoint(workflow_id, "latest")
            if not initial_state:
                initial_state = create_initial_state(workflow_id)
            
            # 添加输入数据
            if input_data:
                initial_state["context"]["input"] = input_data
                initial_state["messages"].append({
                    "role": "user",
                    "content": f"工作流输入: {input_data}",
                    "timestamp": datetime.now().isoformat()
                })
            
            # 更新状态为运行中
            initial_state["metadata"]["status"] = "running"
            initial_state["metadata"]["started_at"] = datetime.now().isoformat()
            
            # 异步执行工作流
            asyncio.create_task(self._execute_workflow(workflow_id, builder, initial_state))
            
            # 更新数据库状态
            await self.repository.update_workflow(workflow_id, {
                "status": "running",
                "started_at": datetime.now()
            })
            
            return WorkflowResponse(
                id=workflow_id,
                status="running",
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
                "completed_at": datetime.now()
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
                "failed_at": datetime.now()
            })
    
    async def get_workflow_status(self, workflow_id: str) -> WorkflowResponse:
        """获取工作流状态"""
        try:
            # 从数据库获取基本信息
            workflow_data = await self.repository.get_workflow(workflow_id)
            if not workflow_data:
                raise ValueError(f"工作流不存在: {workflow_id}")
            
            # 获取最新检查点
            latest_checkpoint = await checkpoint_manager.get_latest_checkpoint(workflow_id)
            current_state = latest_checkpoint.state if latest_checkpoint else None
            
            return WorkflowResponse(
                id=workflow_id,
                name=workflow_data["name"],
                description=workflow_data["description"],
                workflow_type=workflow_data["workflow_type"],
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
                    "paused_at": datetime.now()
                })
            
            return success
            
        except Exception as e:
            print(f"暂停工作流失败: {e}")
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
                    "resumed_at": datetime.now()
                })
                return True
            
            return False
            
        except Exception as e:
            print(f"恢复工作流失败: {e}")
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
                    "cancelled_at": datetime.now()
                })
                
                # 从活跃工作流中移除
                if workflow_id in self.active_workflows:
                    del self.active_workflows[workflow_id]
            
            return success
            
        except Exception as e:
            print(f"取消工作流失败: {e}")
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
            await checkpoint_manager.storage.delete_checkpoints(workflow_id)
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"删除工作流失败: {str(e)}")


# 全局工作流服务实例
workflow_service = WorkflowService()