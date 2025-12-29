"""
工作流数据访问层
"""

from typing import Any, Dict, List, Optional
from sqlalchemy import and_, desc, select
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from src.models.database.workflow import WorkflowModel, Task, DAGExecution as DAGExecutionModel
from src.core.database import get_db_session

class WorkflowRepository:
    """工作流数据仓库"""
    
    async def create_workflow(self, workflow_data: Dict[str, Any]) -> str:
        """创建工作流"""
        async with get_db_session() as session:
            workflow = WorkflowModel(**workflow_data)
            session.add(workflow)
            await session.commit()
            await session.refresh(workflow)
            return workflow.id
    
    async def get_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """获取工作流"""
        async with get_db_session() as session:
            stmt = select(WorkflowModel).filter(
                and_(
                    WorkflowModel.id == workflow_id,
                    WorkflowModel.is_deleted.is_(False)
                )
            )
            result = await session.execute(stmt)
            workflow = result.scalar_one_or_none()
            
            if not workflow:
                return None
            
            return {
                "id": str(workflow.id),
                "name": workflow.name,
                "description": workflow.description,
                "workflow_type": workflow.workflow_type,
                "definition": workflow.definition,
                "status": workflow.status,
                "created_at": workflow.created_at,
                "started_at": workflow.started_at,
                "paused_at": workflow.paused_at,
                "resumed_at": workflow.resumed_at,
                "completed_at": workflow.completed_at,
                "failed_at": workflow.failed_at,
                "cancelled_at": workflow.cancelled_at,
                "error_message": workflow.error_message
            }
    
    async def update_workflow(self, workflow_id: str, update_data: Dict[str, Any]) -> bool:
        """更新工作流"""
        async with get_db_session() as session:
            stmt = select(WorkflowModel).filter(
                and_(
                    WorkflowModel.id == workflow_id,
                    WorkflowModel.is_deleted.is_(False)
                )
            )
            result = await session.execute(stmt)
            workflow = result.scalar_one_or_none()
            
            if not workflow:
                return False
            
            for key, value in update_data.items():
                if hasattr(workflow, key):
                    setattr(workflow, key, value)
            
            await session.commit()
            return True
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """软删除工作流"""
        async with get_db_session() as session:
            stmt = select(WorkflowModel).filter(
                and_(
                    WorkflowModel.id == workflow_id,
                    WorkflowModel.is_deleted.is_(False)
                )
            )
            result = await session.execute(stmt)
            workflow = result.scalar_one_or_none()
            
            if not workflow:
                return False
            
            workflow.is_deleted = True
            workflow.deleted_at = utc_now()
            await session.commit()
            return True
    
    async def list_workflows(self, status: Optional[str] = None, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """列出工作流"""
        async with get_db_session() as session:
            stmt = select(WorkflowModel).filter(
                WorkflowModel.is_deleted.is_(False)
            )
            
            if status:
                stmt = stmt.filter(WorkflowModel.status == status)
            
            stmt = stmt.order_by(desc(WorkflowModel.created_at)).offset(offset).limit(limit)
            result = await session.execute(stmt)
            workflows = result.scalars().all()
            
            return [
                {
                    "id": str(wf.id),
                    "name": wf.name,
                    "description": wf.description,
                    "workflow_type": wf.workflow_type,
                    "definition": wf.definition,
                    "status": wf.status,
                    "created_at": wf.created_at,
                    "started_at": wf.started_at,
                    "completed_at": wf.completed_at,
                    "error_message": wf.error_message
                }
                for wf in workflows
            ]
    
    async def create_task(self, task_data: Dict[str, Any]) -> str:
        """创建任务"""
        async with get_db_session() as session:
            task = Task(**task_data)
            session.add(task)
            await session.commit()
            await session.refresh(task)
            return task.id
    
    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务"""
        async with get_db_session() as session:
            stmt = select(Task).filter(
                and_(
                    Task.id == task_id,
                    Task.is_deleted.is_(False)
                )
            )
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            
            if not task:
                return None
            
            return {
                "id": task.id,
                "workflow_id": task.workflow_id,
                "name": task.name,
                "task_type": task.task_type,
                "status": task.status,
                "input_data": task.input_data,
                "output_data": task.output_data,
                "error_message": task.error_message,
                "dependencies": task.dependencies,
                "agent_id": task.agent_id,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at
            }
    
    async def update_task(self, task_id: str, update_data: Dict[str, Any]) -> bool:
        """更新任务"""
        async with get_db_session() as session:
            stmt = select(Task).filter(
                and_(
                    Task.id == task_id,
                    Task.is_deleted.is_(False)
                )
            )
            result = await session.execute(stmt)
            task = result.scalar_one_or_none()
            
            if not task:
                return False
            
            for key, value in update_data.items():
                if hasattr(task, key):
                    setattr(task, key, value)
            
            await session.commit()
            return True
    
    async def list_workflow_tasks(self, workflow_id: str) -> List[Dict[str, Any]]:
        """列出工作流的所有任务"""
        async with get_db_session() as session:
            stmt = select(Task).filter(
                and_(
                    Task.workflow_id == workflow_id,
                    Task.is_deleted.is_(False)
                )
            ).order_by(Task.created_at)
            result = await session.execute(stmt)
            tasks = result.scalars().all()
            
            return [
                {
                    "id": task.id,
                    "workflow_id": task.workflow_id,
                    "name": task.name,
                    "task_type": task.task_type,
                    "status": task.status,
                    "input_data": task.input_data,
                    "output_data": task.output_data,
                    "error_message": task.error_message,
                    "dependencies": task.dependencies,
                    "agent_id": task.agent_id,
                    "created_at": task.created_at,
                    "started_at": task.started_at,
                    "completed_at": task.completed_at
                }
                for task in tasks
            ]
    
    async def create_dag_execution(self, dag_data: Dict[str, Any]) -> str:
        """创建DAG执行"""
        async with get_db_session() as session:
            dag = DAGExecutionModel(**dag_data)
            session.add(dag)
            await session.commit()
            await session.refresh(dag)
            return dag.id
    
    async def get_dag_execution(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """获取DAG执行"""
        async with get_db_session() as session:
            stmt = select(DAGExecutionModel).filter(
                and_(
                    DAGExecutionModel.id == execution_id,
                    DAGExecutionModel.is_deleted.is_(False)
                )
            )
            result = await session.execute(stmt)
            dag = result.scalar_one_or_none()
            
            if not dag:
                return None
            
            return {
                "id": dag.id,
                "conversation_id": dag.conversation_id,
                "graph_definition": dag.graph_definition,
                "status": dag.status,
                "current_task_id": dag.current_task_id,
                "context": dag.context,
                "checkpoints": dag.checkpoints,
                "progress": dag.progress,
                "created_at": dag.created_at,
                "started_at": dag.started_at,
                "completed_at": dag.completed_at
            }
    
    async def update_dag_execution(self, execution_id: str, update_data: Dict[str, Any]) -> bool:
        """更新DAG执行"""
        async with get_db_session() as session:
            stmt = select(DAGExecutionModel).filter(
                and_(
                    DAGExecutionModel.id == execution_id,
                    DAGExecutionModel.is_deleted.is_(False)
                )
            )
            result = await session.execute(stmt)
            dag = result.scalar_one_or_none()
            
            if not dag:
                return False
            
            for key, value in update_data.items():
                if hasattr(dag, key):
                    setattr(dag, key, value)
            
            dag.updated_at = utc_now()
            await session.commit()
            return True
