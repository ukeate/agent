"""
SupervisorAgent测试
测试Supervisor智能体的核心功能
"""
import pytest
import asyncio
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from ai.autogen.supervisor_agent import (
    SupervisorAgent, TaskComplexityAnalyzer, AgentCapabilityMatcher,
    TaskType, TaskPriority, AgentStatus, TaskComplexity, AgentCapabilityMatch
)
from ai.autogen.agents import BaseAutoGenAgent
from ai.autogen.config import AgentConfig, AgentRole


@pytest.fixture
def mock_agent():
    """创建模拟智能体"""
    config = AgentConfig(
        name="test_code_expert",
        role=AgentRole.CODE_EXPERT,
        system_prompt="Test code expert",
        capabilities=["代码生成", "编程语言", "调试"]
    )
    agent = Mock(spec=BaseAutoGenAgent)
    agent.config = config
    return agent


@pytest.fixture
def available_agents(mock_agent):
    """创建可用智能体字典"""
    return {"test_code_expert": mock_agent}


@pytest.fixture
def supervisor_agent(available_agents):
    """创建SupervisorAgent实例"""
    return SupervisorAgent(available_agents)


class TestTaskComplexityAnalyzer:
    """任务复杂度分析器测试"""
    
    @pytest.fixture
    def analyzer(self):
        return TaskComplexityAnalyzer()
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_simple_task(self, analyzer):
        """测试简单任务复杂度分析"""
        task_description = "创建一个简单的Hello World程序"
        task_type = TaskType.CODE_GENERATION
        
        complexity = await analyzer.analyze_complexity(task_description, task_type)
        
        assert isinstance(complexity, TaskComplexity)
        assert 0.0 <= complexity.score <= 1.0
        assert complexity.estimated_time > 0
        assert len(complexity.required_capabilities) > 0
        assert "factors" in complexity.to_dict()
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_complex_task(self, analyzer):
        """测试复杂任务复杂度分析"""
        task_description = """
        需要设计并实现一个微服务架构的用户认证系统，包括：
        1. JWT token管理
        2. OAuth2集成
        3. 数据库设计
        4. API安全性
        5. 性能优化
        6. 监控和日志
        """
        task_type = TaskType.ARCHITECTURE
        
        complexity = await analyzer.analyze_complexity(task_description, task_type)
        
        assert complexity.score > 0.5  # 复杂任务应该有较高分数
        assert complexity.estimated_time > 600  # 预估时间较长
        assert "高级分析" in complexity.required_capabilities
    
    @pytest.mark.asyncio
    async def test_analyze_complexity_error_handling(self, analyzer):
        """测试复杂度分析错误处理"""
        # 测试异常情况下的默认复杂度返回
        with patch.object(analyzer, '_determine_required_capabilities', side_effect=Exception("Test error")):
            complexity = await analyzer.analyze_complexity("test task", TaskType.CODE_GENERATION)
            
            assert complexity.score == 0.5  # 默认复杂度
            assert complexity.estimated_time == 600  # 默认时间
            assert "error" in complexity.factors


class TestAgentCapabilityMatcher:
    """智能体能力匹配器测试"""
    
    @pytest.fixture
    def matcher(self, available_agents):
        return AgentCapabilityMatcher(available_agents)
    
    @pytest.mark.asyncio
    async def test_find_best_matches(self, matcher):
        """测试寻找最佳匹配智能体"""
        complexity = TaskComplexity(
            score=0.5,
            factors={"test": 0.5},
            estimated_time=300,
            required_capabilities=["代码生成", "编程语言"]
        )
        task_type = TaskType.CODE_GENERATION
        
        matches = await matcher.find_best_matches(complexity, task_type, top_n=3)
        
        assert len(matches) > 0
        assert all(isinstance(match, AgentCapabilityMatch) for match in matches)
        assert matches[0].match_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_update_agent_load(self, matcher):
        """测试更新智能体负载"""
        agent_name = "test_code_expert"
        initial_load = matcher._agent_loads[agent_name]
        
        matcher.update_agent_load(agent_name, 0.3)
        
        assert matcher._agent_loads[agent_name] == initial_load + 0.3
        assert 0.0 <= matcher._agent_loads[agent_name] <= 1.0
    
    def test_get_agent_loads(self, matcher):
        """测试获取智能体负载"""
        loads = matcher.get_agent_loads()
        
        assert isinstance(loads, dict)
        assert "test_code_expert" in loads
        assert all(0.0 <= load <= 1.0 for load in loads.values())


class TestSupervisorAgent:
    """SupervisorAgent核心功能测试"""
    
    @pytest.mark.asyncio
    async def test_analyze_and_assign_task(self, supervisor_agent):
        """测试任务分析和分配"""
        task_description = "实现一个用户登录功能"
        task_type = TaskType.CODE_GENERATION
        priority = TaskPriority.MEDIUM
        
        assignment = await supervisor_agent.analyze_and_assign_task(
            task_description=task_description,
            task_type=task_type,
            priority=priority
        )
        
        assert assignment.task_id.startswith("task_")
        assert assignment.assigned_agent in supervisor_agent.available_agents
        assert assignment.confidence_level >= 0.0
        assert assignment.estimated_completion_time > datetime.now(timezone.utc)
        assert len(assignment.alternative_agents) >= 0
        assert "complexity" in assignment.decision_metadata
    
    @pytest.mark.asyncio
    async def test_analyze_and_assign_task_no_agents(self):
        """测试没有可用智能体时的错误处理"""
        supervisor = SupervisorAgent({})  # 空智能体列表
        
        with pytest.raises(ValueError, match="未找到合适的智能体"):
            await supervisor.analyze_and_assign_task(
                task_description="test task",
                task_type=TaskType.CODE_GENERATION,
                priority=TaskPriority.MEDIUM
            )
    
    @pytest.mark.asyncio
    async def test_get_supervisor_status(self, supervisor_agent):
        """测试获取Supervisor状态"""
        status = await supervisor_agent.get_supervisor_status()
        
        assert "supervisor_name" in status
        assert "status" in status
        assert "available_agents" in status
        assert "agent_loads" in status
        assert "decision_history_count" in status
        assert "task_queue_length" in status
        assert "performance_metrics" in status
        
        assert isinstance(status["available_agents"], list)
        assert isinstance(status["agent_loads"], dict)
        assert isinstance(status["performance_metrics"], dict)
    
    @pytest.mark.asyncio
    async def test_get_decision_history(self, supervisor_agent):
        """测试获取决策历史"""
        # 先创建一些决策记录
        await supervisor_agent.analyze_and_assign_task(
            task_description="test task 1",
            task_type=TaskType.CODE_GENERATION,
            priority=TaskPriority.MEDIUM
        )
        await supervisor_agent.analyze_and_assign_task(
            task_description="test task 2",
            task_type=TaskType.CODE_REVIEW,
            priority=TaskPriority.HIGH
        )
        
        history = await supervisor_agent.get_decision_history(limit=5)
        
        assert len(history) >= 2
        assert all("decision_id" in decision for decision in history)
        assert all("assignment" in decision for decision in history)
        assert all("reasoning" in decision for decision in history)
    
    def test_add_agent(self, supervisor_agent):
        """测试添加智能体"""
        config = AgentConfig(
            name="new_agent",
            role=AgentRole.DOC_EXPERT,
            system_prompt="New doc expert",
            capabilities=["文档撰写"]
        )
        new_agent = Mock(spec=BaseAutoGenAgent)
        new_agent.config = config
        
        initial_count = len(supervisor_agent.available_agents)
        supervisor_agent.add_agent("new_agent", new_agent)
        
        assert len(supervisor_agent.available_agents) == initial_count + 1
        assert "new_agent" in supervisor_agent.available_agents
    
    def test_remove_agent(self, supervisor_agent):
        """测试移除智能体"""
        agent_name = "test_code_expert"
        assert agent_name in supervisor_agent.available_agents
        
        supervisor_agent.remove_agent(agent_name)
        
        assert agent_name not in supervisor_agent.available_agents
    
    @pytest.mark.asyncio
    async def test_multiple_task_assignments(self, supervisor_agent):
        """测试多任务分配的负载平衡"""
        tasks = [
            ("任务1", TaskType.CODE_GENERATION, TaskPriority.HIGH),
            ("任务2", TaskType.CODE_REVIEW, TaskPriority.MEDIUM),
            ("任务3", TaskType.DOCUMENTATION, TaskPriority.LOW),
        ]
        
        assignments = []
        for description, task_type, priority in tasks:
            assignment = await supervisor_agent.analyze_and_assign_task(
                task_description=description,
                task_type=task_type,
                priority=priority
            )
            assignments.append(assignment)
        
        assert len(assignments) == 3
        assert all(a.assigned_agent for a in assignments)
        
        # 验证负载更新
        loads = supervisor_agent.capability_matcher.get_agent_loads()
        assert any(load > 0 for load in loads.values())  # 应该有一些负载
    
    @pytest.mark.asyncio
    async def test_task_priority_handling(self, supervisor_agent):
        """测试任务优先级处理"""
        urgent_assignment = await supervisor_agent.analyze_and_assign_task(
            task_description="紧急任务",
            task_type=TaskType.CODE_GENERATION,
            priority=TaskPriority.URGENT
        )
        
        low_assignment = await supervisor_agent.analyze_and_assign_task(
            task_description="低优先级任务",
            task_type=TaskType.CODE_GENERATION,
            priority=TaskPriority.LOW
        )
        
        # 紧急任务应该有更高的置信度或更好的智能体分配
        assert urgent_assignment.confidence_level >= 0.0
        assert low_assignment.confidence_level >= 0.0


class TestSupervisorIntegration:
    """Supervisor集成测试"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_task_flow(self):
        """测试端到端任务流程"""
        # 创建多个模拟智能体
        code_expert = Mock(spec=BaseAutoGenAgent)
        code_expert.config = AgentConfig(
            name="code_expert",
            role=AgentRole.CODE_EXPERT,
            system_prompt="Code expert",
            capabilities=["代码生成", "代码审查", "调试"]
        )
        
        architect = Mock(spec=BaseAutoGenAgent)
        architect.config = AgentConfig(
            name="architect",
            role=AgentRole.ARCHITECT,
            system_prompt="System architect",
            capabilities=["系统设计", "架构模式", "技术选型"]
        )
        
        agents = {
            "code_expert": code_expert,
            "architect": architect
        }
        
        supervisor = SupervisorAgent(agents)
        
        # 测试不同类型的任务分配
        code_task = await supervisor.analyze_and_assign_task(
            task_description="实现用户认证API",
            task_type=TaskType.CODE_GENERATION,
            priority=TaskPriority.HIGH
        )
        
        architecture_task = await supervisor.analyze_and_assign_task(
            task_description="设计微服务架构",
            task_type=TaskType.ARCHITECTURE,
            priority=TaskPriority.MEDIUM
        )
        
        # 验证任务分配的合理性
        assert code_task.assigned_agent in ["code_expert", "architect"]
        assert architecture_task.assigned_agent in ["code_expert", "architect"]
        
        # 验证状态更新
        status = await supervisor.get_supervisor_status()
        assert status["decision_history_count"] >= 2
        
        # 验证决策历史
        decisions = await supervisor.get_decision_history()
        assert len(decisions) >= 2
        assert all(d["confidence"] > 0 for d in decisions)


@pytest.mark.asyncio
async def test_performance_with_many_agents():
    """测试大量智能体时的性能"""
    # 创建多个智能体
    agents = {}
    for i in range(10):
        agent = Mock(spec=BaseAutoGenAgent)
        agent.config = AgentConfig(
            name=f"agent_{i}",
            role=AgentRole.CODE_EXPERT,
            system_prompt=f"Agent {i}",
            capabilities=[f"能力_{i}"]
        )
        agents[f"agent_{i}"] = agent
    
    supervisor = SupervisorAgent(agents)
    
    # 测试分配性能
    start_time = datetime.now()
    
    assignment = await supervisor.analyze_and_assign_task(
        task_description="性能测试任务",
        task_type=TaskType.CODE_GENERATION,
        priority=TaskPriority.MEDIUM
    )
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    assert assignment.assigned_agent.startswith("agent_")
    assert duration < 1.0  # 分配应该在1秒内完成