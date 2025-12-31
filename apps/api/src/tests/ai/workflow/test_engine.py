"""
工作流执行引擎单元测试
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from uuid import uuid4
import networkx as nx
from src.ai.workflow.engine import WorkflowEngine, WorkflowValidationError, WorkflowExecutionError
from src.ai.workflow.executor import SequentialExecutor, ParallelExecutor, BaseStepExecutor
from models.schemas.workflow import (
    WorkflowDefinition, WorkflowStep, WorkflowStepType, WorkflowExecutionMode,
    WorkflowExecution, WorkflowStepExecution, WorkflowStepStatus,
    TaskDependencyType
)

class TestWorkflowEngine:
    """工作流引擎测试"""
    
    @pytest.fixture
    def engine(self):
        """创建工作流引擎实例"""
        return WorkflowEngine()
    
    @pytest.fixture
    def simple_workflow_definition(self):
        """简单工作流定义"""
        steps = [
            WorkflowStep(
                id="step1",
                name="第一步",
                step_type=WorkflowStepType.REASONING,
                config={"strategy": "zero_shot"}
            ),
            WorkflowStep(
                id="step2",
                name="第二步",
                step_type=WorkflowStepType.TOOL_CALL,
                dependencies=["step1"],
                config={"tool": "calculator"}
            )
        ]
        
        return WorkflowDefinition(
            id="simple-workflow",
            name="简单工作流",
            steps=steps,
            execution_mode=WorkflowExecutionMode.SEQUENTIAL
        )
    
    @pytest.fixture
    def parallel_workflow_definition(self):
        """并行工作流定义"""
        steps = [
            WorkflowStep(
                id="start",
                name="开始",
                step_type=WorkflowStepType.REASONING,
                config={"strategy": "zero_shot"}
            ),
            WorkflowStep(
                id="parallel1",
                name="并行任务1",
                step_type=WorkflowStepType.TOOL_CALL,
                dependencies=["start"],
                config={"tool": "search"}
            ),
            WorkflowStep(
                id="parallel2",
                name="并行任务2",
                step_type=WorkflowStepType.TOOL_CALL,
                dependencies=["start"],
                config={"tool": "analyze"}
            ),
            WorkflowStep(
                id="end",
                name="结束",
                step_type=WorkflowStepType.AGGREGATION,
                dependencies=["parallel1", "parallel2"],
                config={"method": "merge"}
            )
        ]
        
        return WorkflowDefinition(
            id="parallel-workflow",
            name="并行工作流",
            steps=steps,
            execution_mode=WorkflowExecutionMode.PARALLEL,
            max_parallel_steps=2
        )
    
    @pytest.fixture
    def invalid_workflow_definition(self):
        """无效工作流定义（循环依赖）"""
        steps = [
            WorkflowStep(
                id="step1",
                name="步骤1",
                step_type=WorkflowStepType.REASONING,
                dependencies=["step2"]
            ),
            WorkflowStep(
                id="step2",
                name="步骤2",
                step_type=WorkflowStepType.TOOL_CALL,
                dependencies=["step1"]
            )
        ]
        
        return WorkflowDefinition(
            id="invalid-workflow",
            name="无效工作流",
            steps=steps
        )

class TestWorkflowValidation:
    """工作流验证测试"""
    
    @pytest.mark.asyncio
    async def test_validate_simple_workflow(self, engine, simple_workflow_definition):
        """测试验证简单工作流"""
        errors = await engine.validate_workflow(simple_workflow_definition)
        assert errors == []
    
    @pytest.mark.asyncio
    async def test_validate_parallel_workflow(self, engine, parallel_workflow_definition):
        """测试验证并行工作流"""
        errors = await engine.validate_workflow(parallel_workflow_definition)
        assert errors == []
    
    @pytest.mark.asyncio
    async def test_validate_empty_workflow(self, engine):
        """测试验证空工作流"""
        empty_definition = WorkflowDefinition(
            id="empty",
            name="空工作流",
            steps=[]
        )
        
        errors = await engine.validate_workflow(empty_definition)
        assert len(errors) == 1
        assert "至少一个步骤" in errors[0]
    
    @pytest.mark.asyncio
    async def test_validate_cyclic_dependency(self, engine, invalid_workflow_definition):
        """测试验证循环依赖"""
        errors = await engine.validate_workflow(invalid_workflow_definition)
        assert len(errors) > 0
        assert any("循环依赖" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_validate_duplicate_step_ids(self, engine):
        """测试验证重复步骤ID"""
        steps = [
            WorkflowStep(id="step1", name="步骤1", step_type=WorkflowStepType.REASONING),
            WorkflowStep(id="step1", name="步骤2", step_type=WorkflowStepType.TOOL_CALL)
        ]
        
        definition = WorkflowDefinition(
            id="duplicate-ids",
            name="重复ID工作流",
            steps=steps
        )
        
        errors = await engine.validate_workflow(definition)
        assert len(errors) > 0
        assert any("步骤ID必须唯一" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_validate_invalid_dependency_reference(self, engine):
        """测试验证无效依赖引用"""
        steps = [
            WorkflowStep(
                id="step1",
                name="步骤1",
                step_type=WorkflowStepType.REASONING,
                dependencies=["nonexistent"]
            )
        ]
        
        definition = WorkflowDefinition(
            id="invalid-ref",
            name="无效引用工作流",
            steps=steps
        )
        
        errors = await engine.validate_workflow(definition)
        assert len(errors) > 0
        assert any("不存在的依赖" in error for error in errors)

class TestWorkflowExecution:
    """工作流执行测试"""
    
    @pytest.mark.asyncio
    async def test_create_execution(self, engine, simple_workflow_definition):
        """测试创建执行实例"""
        input_data = {"problem": "测试问题"}
        execution = await engine.create_execution(
            definition=simple_workflow_definition,
            input_data=input_data,
            session_id="test-session"
        )
        
        assert execution.id is not None
        assert execution.workflow_definition_id == simple_workflow_definition.id
        assert execution.session_id == "test-session"
        assert execution.status == "pending"
        assert execution.execution_context == input_data
        assert execution.total_steps == 2
        assert len(execution.step_executions) == 2
    
    @pytest.mark.asyncio
    async def test_create_execution_with_invalid_workflow(self, engine, invalid_workflow_definition):
        """测试创建无效工作流执行"""
        with pytest.raises(WorkflowValidationError):
            await engine.create_execution(definition=invalid_workflow_definition)
    
    @pytest.mark.asyncio
    async def test_get_execution_status(self, engine, simple_workflow_definition):
        """测试获取执行状态"""
        execution = await engine.create_execution(definition=simple_workflow_definition)
        
        retrieved_execution = await engine.get_execution_status(execution.id)
        assert retrieved_execution is not None
        assert retrieved_execution.id == execution.id
        assert retrieved_execution.status == execution.status
    
    @pytest.mark.asyncio
    async def test_get_execution_graph(self, engine, simple_workflow_definition):
        """测试获取执行图"""
        execution = await engine.create_execution(definition=simple_workflow_definition)
        
        graph = engine.get_execution_graph(execution.id)
        assert graph is not None
        assert isinstance(graph, nx.DiGraph)
        assert "step1" in graph.nodes
        assert "step2" in graph.nodes
        assert graph.has_edge("step1", "step2")
    
    @pytest.mark.asyncio
    async def test_cleanup_execution(self, engine, simple_workflow_definition):
        """测试清理执行实例"""
        execution = await engine.create_execution(definition=simple_workflow_definition)
        
        # 模拟完成状态
        execution.status = "completed"
        
        result = await engine.cleanup_execution(execution.id)
        assert result is True
        
        # 验证已清理
        retrieved_execution = await engine.get_execution_status(execution.id)
        assert retrieved_execution is None

class TestDependencyGraph:
    """依赖图测试"""
    
    def test_build_simple_dependency_graph(self, engine):
        """测试构建简单依赖图"""
        steps = [
            WorkflowStep(id="A", name="A", step_type=WorkflowStepType.REASONING),
            WorkflowStep(id="B", name="B", step_type=WorkflowStepType.TOOL_CALL, dependencies=["A"]),
            WorkflowStep(id="C", name="C", step_type=WorkflowStepType.VALIDATION, dependencies=["B"])
        ]
        
        graph = engine._build_dependency_graph(steps)
        
        assert "A" in graph.nodes
        assert "B" in graph.nodes
        assert "C" in graph.nodes
        assert graph.has_edge("A", "B")
        assert graph.has_edge("B", "C")
        assert nx.is_directed_acyclic_graph(graph)
    
    def test_build_parallel_dependency_graph(self, engine):
        """测试构建并行依赖图"""
        steps = [
            WorkflowStep(id="start", name="开始", step_type=WorkflowStepType.REASONING),
            WorkflowStep(id="task1", name="任务1", step_type=WorkflowStepType.TOOL_CALL, dependencies=["start"]),
            WorkflowStep(id="task2", name="任务2", step_type=WorkflowStepType.TOOL_CALL, dependencies=["start"]),
            WorkflowStep(id="end", name="结束", step_type=WorkflowStepType.AGGREGATION, dependencies=["task1", "task2"])
        ]
        
        graph = engine._build_dependency_graph(steps)
        
        assert len(graph.nodes) == 4
        assert graph.has_edge("start", "task1")
        assert graph.has_edge("start", "task2")
        assert graph.has_edge("task1", "end")
        assert graph.has_edge("task2", "end")
        assert nx.is_directed_acyclic_graph(graph)
    
    def test_analyze_parallel_groups(self, engine):
        """测试分析并行组"""
        steps = [
            WorkflowStep(id="start", name="开始", step_type=WorkflowStepType.REASONING),
            WorkflowStep(id="task1", name="任务1", step_type=WorkflowStepType.TOOL_CALL, dependencies=["start"]),
            WorkflowStep(id="task2", name="任务2", step_type=WorkflowStepType.TOOL_CALL, dependencies=["start"]),
            WorkflowStep(id="end", name="结束", step_type=WorkflowStepType.AGGREGATION, dependencies=["task1", "task2"])
        ]
        
        graph = engine._build_dependency_graph(steps)
        parallel_groups = engine._analyze_parallel_groups(steps, graph)
        
        assert len(parallel_groups) == 3
        assert ["start"] == parallel_groups[0]
        assert set(parallel_groups[1]) == {"task1", "task2"}  # 并行组
        assert ["end"] == parallel_groups[2]

class TestConditionValidation:
    """条件验证测试"""
    
    def test_validate_simple_conditions(self, engine):
        """测试验证简单条件"""
        valid_conditions = [
            "input.score > 0.5",
            "context.user_type == 'admin'",
            "previous.confidence >= 0.8",
            "input.count > 10 and input.status == 'active'"
        ]
        
        for condition in valid_conditions:
            result = engine._validate_condition_expression(condition)
            assert result is True, f"条件应该有效: {condition}"
    
    def test_validate_invalid_conditions(self, engine):
        """测试验证无效条件"""
        invalid_conditions = [
            "",  # 空条件
            None,  # None值
        ]
        
        for condition in invalid_conditions:
            result = engine._validate_condition_expression(condition)
            assert result is False, f"条件应该无效: {condition}"

class TestEngineStatistics:
    """引擎统计测试"""
    
    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, engine):
        """测试获取空引擎统计"""
        stats = engine.get_statistics()
        
        assert stats["active_executions"] == 0
        assert stats["status_distribution"] == {}
        assert len(stats["available_execution_modes"]) > 0
    
    @pytest.mark.asyncio
    async def test_get_statistics_with_executions(self, engine, simple_workflow_definition):
        """测试获取有执行实例的统计"""
        # 创建几个执行实例
        execution1 = await engine.create_execution(definition=simple_workflow_definition)
        execution2 = await engine.create_execution(definition=simple_workflow_definition)
        execution2.status = "running"
        
        stats = engine.get_statistics()
        
        assert stats["active_executions"] == 2
        assert stats["status_distribution"]["pending"] == 1
        assert stats["status_distribution"]["running"] == 1

class TestEngineBuilder:
    """引擎构建器测试"""
    
    def test_build_default_engine(self):
        """测试构建默认引擎"""
        from ai.workflow.engine import WorkflowEngineBuilder
        
        engine = WorkflowEngineBuilder().build()
        
        assert isinstance(engine, WorkflowEngine)
        assert WorkflowExecutionMode.SEQUENTIAL in engine.executors
        assert WorkflowExecutionMode.PARALLEL in engine.executors
        assert WorkflowExecutionMode.HYBRID in engine.executors
    
    def test_build_custom_engine(self):
        """测试构建自定义引擎"""
        from ai.workflow.engine import WorkflowEngineBuilder
        
        custom_executor = Mock(spec=SequentialExecutor)
        engine = (WorkflowEngineBuilder()
                 .with_custom_executor(WorkflowExecutionMode.SEQUENTIAL, custom_executor)
                 .build())
        
        assert engine.executors[WorkflowExecutionMode.SEQUENTIAL] == custom_executor

class TestErrorHandling:
    """错误处理测试"""
    
    @pytest.mark.asyncio
    async def test_validation_error_handling(self, engine):
        """测试验证错误处理"""
        invalid_definition = WorkflowDefinition(
            id="invalid",
            name="无效工作流",
            steps=[]  # 空步骤列表
        )
        
        with pytest.raises(WorkflowValidationError) as exc_info:
            await engine.create_execution(definition=invalid_definition)
        
        assert "工作流验证失败" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_execution_not_found_error(self, engine):
        """测试执行实例未找到错误"""
        nonexistent_id = str(uuid4())
        
        with pytest.raises(WorkflowExecutionError) as exc_info:
            await engine.execute_workflow(nonexistent_id)
        
        assert "执行实例不存在" in str(exc_info.value)

class TestWorkflowControl:
    """工作流控制测试"""
    
    @pytest.mark.asyncio
    async def test_pause_execution(self, engine, simple_workflow_definition):
        """测试暂停执行"""
        execution = await engine.create_execution(definition=simple_workflow_definition)
        execution.status = "running"  # 模拟运行状态
        
        result = await engine.pause_execution(execution.id)
        assert result is True
        assert execution.status == "paused"
    
    @pytest.mark.asyncio
    async def test_resume_execution(self, engine, simple_workflow_definition):
        """测试恢复执行"""
        execution = await engine.create_execution(definition=simple_workflow_definition)
        execution.status = "paused"  # 模拟暂停状态
        
        result = await engine.resume_execution(execution.id)
        assert result is True
        assert execution.status == "running"
    
    @pytest.mark.asyncio
    async def test_cancel_execution(self, engine, simple_workflow_definition):
        """测试取消执行"""
        execution = await engine.create_execution(definition=simple_workflow_definition)
        execution.status = "running"  # 模拟运行状态
        
        result = await engine.cancel_execution(execution.id)
        assert result is True
        assert execution.status == "cancelled"
    
    @pytest.mark.asyncio
    async def test_control_nonexistent_execution(self, engine):
        """测试控制不存在的执行"""
        nonexistent_id = str(uuid4())
        
        pause_result = await engine.pause_execution(nonexistent_id)
        assert pause_result is False
        
        resume_result = await engine.resume_execution(nonexistent_id)
        assert resume_result is False
        
        cancel_result = await engine.cancel_execution(nonexistent_id)
        assert cancel_result is False

@pytest.mark.integration
class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_complete_workflow_lifecycle(self, engine, simple_workflow_definition):
        """测试完整工作流生命周期"""
        # 1. 创建执行
        execution = await engine.create_execution(
            definition=simple_workflow_definition,
            input_data={"problem": "测试问题"}
        )
        
        assert execution.status == "pending"
        assert execution.total_steps == 2
        
        # 2. 获取执行图
        graph = engine.get_execution_graph(execution.id)
        assert graph is not None
        assert len(graph.nodes) == 2
        
        # 3. 检查统计信息
        stats = engine.get_statistics()
        assert stats["active_executions"] == 1
        
        # 4. 控制操作
        execution.status = "running"
        pause_result = await engine.pause_execution(execution.id)
        assert pause_result is True
        
        resume_result = await engine.resume_execution(execution.id)
        assert resume_result is True
        
        # 5. 完成并清理
        execution.status = "completed"
        cleanup_result = await engine.cleanup_execution(execution.id)
        assert cleanup_result is True
        
        # 6. 验证清理
        final_stats = engine.get_statistics()
        assert final_stats["active_executions"] == 0
