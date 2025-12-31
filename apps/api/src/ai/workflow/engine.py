"""
多步推理工作流执行引擎
支持DAG解析、验证和调度执行
"""

from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import uuid4
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from enum import Enum
import networkx as nx
from pydantic import BaseModel
from src.models.schemas.workflow import (
    WorkflowDefinition, WorkflowExecution, WorkflowStepExecution,
    WorkflowStep, WorkflowStepType, WorkflowStepStatus, WorkflowExecutionMode,
    TaskDependencyType
)
from src.ai.workflow.executor import WorkflowExecutor, ParallelExecutor, SequentialExecutor

from src.core.logging import get_logger
logger = get_logger(__name__)

class WorkflowValidationError(Exception):
    """工作流验证错误"""
    ...

class WorkflowExecutionError(Exception):
    """工作流执行错误"""
    ...

class ExecutionStatus(str, Enum):
    """执行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class WorkflowEngine:
    """工作流执行引擎核心类"""
    
    def __init__(self):
        self.executors: Dict[WorkflowExecutionMode, WorkflowExecutor] = {
            WorkflowExecutionMode.SEQUENTIAL: SequentialExecutor(),
            WorkflowExecutionMode.PARALLEL: ParallelExecutor(),
            WorkflowExecutionMode.HYBRID: ParallelExecutor()  # 混合模式使用并行执行器
        }
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_graphs: Dict[str, nx.DiGraph] = {}
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
    
    async def validate_workflow(self, definition: WorkflowDefinition) -> List[str]:
        """
        验证工作流定义
        
        Args:
            definition: 工作流定义
            
        Returns:
            验证错误列表，空列表表示验证通过
        """
        errors = []
        
        try:
            # 1. 基本验证
            if not definition.steps:
                errors.append("工作流必须包含至少一个步骤")
                return errors
            
            # 2. 步骤ID唯一性验证
            step_ids = [step.id for step in definition.steps]
            if len(step_ids) != len(set(step_ids)):
                errors.append("步骤ID必须唯一")
            
            # 3. 构建依赖图并验证
            graph = self._build_dependency_graph(definition.steps)
            
            # 4. 检查循环依赖
            if not nx.is_directed_acyclic_graph(graph):
                cycles = list(nx.simple_cycles(graph))
                errors.append(f"存在循环依赖: {cycles}")
            
            # 5. 检查依赖引用有效性
            for step in definition.steps:
                for dep_id in step.dependencies:
                    if dep_id not in step_ids:
                        errors.append(f"步骤 '{step.id}' 引用了不存在的依赖 '{dep_id}'")
            
            # 6. 验证并行组配置
            if definition.execution_mode == WorkflowExecutionMode.PARALLEL:
                parallel_groups = self._analyze_parallel_groups(definition.steps, graph)
                if definition.max_parallel_steps < len(max(parallel_groups, key=len, default=[])):
                    errors.append("最大并行步骤数小于最大并行组大小")
            
            # 7. 验证条件表达式
            for step in definition.steps:
                if step.condition:
                    # 简单的条件表达式验证
                    if not self._validate_condition_expression(step.condition):
                        errors.append(f"步骤 '{step.id}' 的条件表达式无效: {step.condition}")
            
            logger.info(f"工作流验证完成: {len(errors)} 个错误")
            
        except Exception as e:
            logger.error(f"工作流验证异常: {e}")
            errors.append(f"验证过程中发生异常: {str(e)}")
        
        return errors
    
    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> nx.DiGraph:
        """构建步骤依赖图"""
        graph = nx.DiGraph()
        
        # 添加节点
        for step in steps:
            graph.add_node(step.id, step=step)
        
        # 添加边（依赖关系）
        for step in steps:
            for dep_id in step.dependencies:
                # 依赖关系：dep_id -> step.id
                graph.add_edge(dep_id, step.id, dependency_type=step.dependency_type)
        
        return graph
    
    def _analyze_parallel_groups(self, steps: List[WorkflowStep], graph: nx.DiGraph) -> List[List[str]]:
        """分析并行执行组"""
        try:
            # 使用拓扑排序的生成器来确定并行组
            parallel_groups = []
            for generation in nx.topological_generations(graph):
                parallel_groups.append(list(generation))
            
            return parallel_groups
        except Exception as e:
            logger.warning(f"分析并行组失败: {e}")
            return [[step.id] for step in steps]  # 降级为串行执行
    
    def _validate_condition_expression(self, condition: str) -> bool:
        """验证条件表达式"""
        try:
            # 简单的语法检查
            if not condition or not isinstance(condition, str):
                return False
            
            # 检查基本的条件表达式格式
            allowed_operators = ['>', '<', '>=', '<=', '==', '!=', 'and', 'or', 'not', 'in']
            allowed_keywords = ['input', 'context', 'previous', 'step']
            
            # 这里只做简单验证，实际应该使用更严格的表达式解析器
            words = condition.replace('(', ' ').replace(')', ' ').split()
            for word in words:
                if word.isdigit() or word.replace('.', '').isdigit():
                    continue  # 数字
                if word.startswith('"') and word.endswith('"'):
                    continue  # 字符串字面量
                if word.startswith("'") and word.endswith("'"):
                    continue  # 字符串字面量
                if word in allowed_operators or any(word.startswith(kw) for kw in allowed_keywords):
                    continue  # 允许的操作符或关键字
                if word.replace('_', '').isalnum():
                    continue  # 变量名
                # 如果都不满足，可能是无效表达式
                logger.warning(f"条件表达式中的可疑词汇: {word}")
            
            return True
        except Exception:
            return False
    
    async def create_execution(
        self,
        definition: WorkflowDefinition,
        input_data: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        execution_config: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """
        创建工作流执行实例
        
        Args:
            definition: 工作流定义
            input_data: 输入数据
            session_id: 会话ID
            execution_config: 执行配置
            
        Returns:
            工作流执行实例
        """
        # 验证工作流定义
        validation_errors = await self.validate_workflow(definition)
        if validation_errors:
            raise WorkflowValidationError(f"工作流验证失败: {'; '.join(validation_errors)}")
        
        # 创建执行实例
        execution_id = str(uuid4())
        execution = WorkflowExecution(
            id=execution_id,
            workflow_definition_id=definition.id,
            session_id=session_id,
            status=ExecutionStatus.PENDING,
            created_at=utc_now(),
            execution_context=input_data or {},
            total_steps=len(definition.steps)
        )
        
        # 构建执行图
        graph = self._build_dependency_graph(definition.steps)
        self.execution_graphs[execution_id] = graph
        self.workflow_definitions[definition.id] = definition
        
        # 初始化步骤执行状态
        step_executions = []
        for step in definition.steps:
            step_execution = WorkflowStepExecution(
                step_id=step.id,
                status=WorkflowStepStatus.PENDING
            )
            step_executions.append(step_execution)
        
        execution.step_executions = step_executions
        self.active_executions[execution_id] = execution
        
        logger.info(f"创建工作流执行: {execution_id}, 定义: {definition.id}")
        return execution
    
    async def execute_workflow(
        self,
        execution_id: str,
        stream_callback: Optional[callable] = None
    ) -> WorkflowExecution:
        """
        执行工作流
        
        Args:
            execution_id: 执行ID
            stream_callback: 流式回调函数
            
        Returns:
            更新后的执行实例
        """
        if execution_id not in self.active_executions:
            raise WorkflowExecutionError(f"执行实例不存在: {execution_id}")
        
        execution = self.active_executions[execution_id]
        graph = self.execution_graphs[execution_id]
        
        try:
            # 更新执行状态
            execution.status = ExecutionStatus.RUNNING
            execution.started_at = utc_now()
            
            # 获取工作流定义
            definition = await self._get_workflow_definition(execution.workflow_definition_id)
            if not definition:
                raise WorkflowExecutionError(f"工作流定义不存在: {execution.workflow_definition_id}")
            
            # 选择执行器
            executor = self.executors[definition.execution_mode]
            
            # 执行工作流
            await executor.execute(
                execution=execution,
                definition=definition,
                graph=graph,
                stream_callback=stream_callback
            )
            
            # 更新最终状态
            if execution.failed_steps > 0:
                execution.status = ExecutionStatus.FAILED
            else:
                execution.status = ExecutionStatus.COMPLETED
                execution.completed_at = utc_now()
            
            logger.info(f"工作流执行完成: {execution_id}, 状态: {execution.status}")
            
        except Exception as e:
            logger.error(f"工作流执行失败: {execution_id}, 错误: {e}")
            execution.status = ExecutionStatus.FAILED
            # 记录错误信息到执行上下文
            if 'errors' not in execution.execution_context:
                execution.execution_context['errors'] = []
            execution.execution_context['errors'].append({
                'timestamp': utc_now().isoformat(),
                'error': str(e),
                'type': type(e).__name__
            })
            raise WorkflowExecutionError(f"工作流执行失败: {e}")
        
        return execution
    
    async def pause_execution(self, execution_id: str) -> bool:
        """暂停工作流执行"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        if execution.status == ExecutionStatus.RUNNING:
            execution.status = ExecutionStatus.PAUSED
            # 暂停具体的执行器
            definition = await self._get_workflow_definition(execution.workflow_definition_id)
            if definition:
                executor = self.executors[definition.execution_mode]
                await executor.pause(execution_id)
            
            logger.info(f"工作流执行已暂停: {execution_id}")
            return True
        
        return False
    
    async def resume_execution(self, execution_id: str) -> bool:
        """恢复工作流执行"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        if execution.status == ExecutionStatus.PAUSED:
            execution.status = ExecutionStatus.RUNNING
            # 恢复具体的执行器
            definition = await self._get_workflow_definition(execution.workflow_definition_id)
            if definition:
                executor = self.executors[definition.execution_mode]
                await executor.resume(execution_id)
            
            logger.info(f"工作流执行已恢复: {execution_id}")
            return True
        
        return False
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """取消工作流执行"""
        if execution_id not in self.active_executions:
            return False
        
        execution = self.active_executions[execution_id]
        if execution.status in [ExecutionStatus.RUNNING, ExecutionStatus.PAUSED]:
            execution.status = ExecutionStatus.CANCELLED
            # 取消具体的执行器
            definition = await self._get_workflow_definition(execution.workflow_definition_id)
            if definition:
                executor = self.executors[definition.execution_mode]
                await executor.cancel(execution_id)
            
            logger.info(f"工作流执行已取消: {execution_id}")
            return True
        
        return False
    
    async def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """获取执行状态"""
        return self.active_executions.get(execution_id)
    
    def get_execution_graph(self, execution_id: str) -> Optional[nx.DiGraph]:
        """获取执行图"""
        return self.execution_graphs.get(execution_id)
    
    async def cleanup_execution(self, execution_id: str) -> bool:
        """清理执行实例"""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            if execution.status in [ExecutionStatus.COMPLETED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED]:
                del self.active_executions[execution_id]
                if execution_id in self.execution_graphs:
                    del self.execution_graphs[execution_id]
                logger.info(f"清理工作流执行实例: {execution_id}")
                return True
        return False
    
    async def _get_workflow_definition(self, definition_id: str) -> Optional[WorkflowDefinition]:
        """获取工作流定义"""
        return self.workflow_definitions.get(definition_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        active_count = len(self.active_executions)
        status_counts = {}
        
        for execution in self.active_executions.values():
            status = execution.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "active_executions": active_count,
            "status_distribution": status_counts,
            "available_execution_modes": list(self.executors.keys())
        }

class WorkflowEngineBuilder:
    """工作流引擎构建器"""
    
    def __init__(self):
        self.engine = WorkflowEngine()
    
    def with_custom_executor(self, mode: WorkflowExecutionMode, executor: WorkflowExecutor) -> 'WorkflowEngineBuilder':
        """添加自定义执行器"""
        self.engine.executors[mode] = executor
        return self
    
    def build(self) -> WorkflowEngine:
        """构建引擎实例"""
        return self.engine

# 全局引擎实例
_engine_instance = None

def get_workflow_engine() -> WorkflowEngine:
    """获取全局工作流引擎实例"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = WorkflowEngine()
    return _engine_instance
