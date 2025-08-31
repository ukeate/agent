"""
Supervisor智能体决策引擎测试
测试任务分析、智能体匹配和任务分配决策流程
"""
import pytest
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import structlog

from src.ai.autogen.supervisor_agent import (
    SupervisorAgent,
    TaskType,
    TaskPriority,
    TaskComplexityAnalyzer,
    AgentCapabilityMatcher,
    TaskComplexity,
    AgentCapabilityMatch,
    TaskAssignment,
    SupervisorDecision,
    AgentStatus
)
from src.ai.autogen.config import AgentConfig, AgentRole, AGENT_CONFIGS
from src.ai.autogen.agents import BaseAutoGenAgent


@pytest.fixture
def mock_agents():
    """创建模拟智能体"""
    agents = {}
    
    # 创建代码专家智能体
    code_agent = Mock(spec=BaseAutoGenAgent)
    code_agent.config = AgentConfig(
        name="code_expert",
        role=AgentRole.CODE_EXPERT,
        capabilities=["代码生成", "编程语言", "调试", "代码审查", "质量评估"],
        system_prompt="代码专家智能体",
        description=""
    )
    agents["code_expert"] = code_agent
    
    # 创建架构师智能体
    architect_agent = Mock(spec=BaseAutoGenAgent)
    architect_agent.config = AgentConfig(
        name="architect",
        role=AgentRole.ARCHITECT,
        capabilities=["系统设计", "架构模式", "技术选型", "高级分析", "系统集成"],
        system_prompt="架构师智能体",
        description=""
    )
    agents["architect"] = architect_agent
    
    # 创建文档专家智能体
    doc_agent = Mock(spec=BaseAutoGenAgent)
    doc_agent.config = AgentConfig(
        name="doc_expert",
        role=AgentRole.DOC_EXPERT,
        capabilities=["文档撰写", "技术写作", "信息架构"],
        system_prompt="文档专家智能体",
        description=""
    )
    agents["doc_expert"] = doc_agent
    
    # 创建知识检索智能体
    knowledge_agent = Mock(spec=BaseAutoGenAgent)
    knowledge_agent.config = AgentConfig(
        name="knowledge_retrieval",
        role=AgentRole.KNOWLEDGE_RETRIEVAL,
        capabilities=["信息检索", "语义搜索", "知识整合"],
        system_prompt="知识检索智能体",
        description=""
    )
    agents["knowledge_retrieval"] = knowledge_agent
    
    return agents


@pytest.fixture
def supervisor_agent(mock_agents):
    """创建Supervisor智能体实例"""
    return SupervisorAgent(available_agents=mock_agents)


@pytest.fixture
def complexity_analyzer():
    """创建任务复杂度分析器实例"""
    return TaskComplexityAnalyzer()


@pytest.fixture
def capability_matcher(mock_agents):
    """创建智能体能力匹配器实例"""
    return AgentCapabilityMatcher(mock_agents)


class TestTaskComplexityAnalyzer:
    """任务复杂度分析器测试"""
    
    @pytest.mark.asyncio
    async def test_analyze_simple_task(self, complexity_analyzer):
        """测试简单任务的复杂度分析"""
        task_description = "创建一个简单的Hello World函数"
        task_type = TaskType.CODE_GENERATION
        
        complexity = await complexity_analyzer.analyze_complexity(task_description, task_type)
        
        assert isinstance(complexity, TaskComplexity)
        assert 0.0 <= complexity.score <= 1.0
        assert complexity.score < 0.5  # 简单任务应该有较低的复杂度分数
        assert len(complexity.factors) > 0
        assert complexity.estimated_time > 0
        assert len(complexity.required_capabilities) > 0
        assert "代码生成" in complexity.required_capabilities
    
    @pytest.mark.asyncio
    async def test_analyze_complex_task(self, complexity_analyzer):
        """测试复杂任务的复杂度分析"""
        task_description = """
        设计并实现一个高性能的微服务架构系统，包括API网关、身份认证服务、
        数据库集成、消息队列、负载均衡、缓存策略、监控和日志系统。
        需要支持水平扩展、故障恢复、数据一致性保证，并提供完整的性能优化方案。
        """
        task_type = TaskType.ARCHITECTURE
        
        complexity = await complexity_analyzer.analyze_complexity(task_description, task_type)
        
        assert complexity.score > 0.5  # 复杂任务应该有较高的复杂度分数
        assert "integration_complexity" in complexity.factors
        assert complexity.factors["integration_complexity"] == 0.9  # 架构任务的集成复杂度高
        assert complexity.estimated_time > 600  # 复杂任务需要更多时间
        assert "系统设计" in complexity.required_capabilities
        assert "高级分析" in complexity.required_capabilities or "系统集成" in complexity.required_capabilities
    
    @pytest.mark.asyncio
    async def test_technical_terms_density(self, complexity_analyzer):
        """测试技术术语密度计算"""
        task_with_technical_terms = """
        实现RESTful API与database的integration，包括authentication和authorization。
        需要考虑microservice架构下的performance和scalability问题。
        """
        task_type = TaskType.CODE_GENERATION
        
        complexity = await complexity_analyzer.analyze_complexity(task_with_technical_terms, task_type)
        
        assert complexity.factors["technical_terms"] > 0.3  # 高技术术语密度
        assert complexity.score > 0.4  # 应该有中等以上的复杂度
    
    @pytest.mark.asyncio
    async def test_different_task_types(self, complexity_analyzer):
        """测试不同任务类型的复杂度差异"""
        task_description = "分析系统性能瓶颈并提供优化建议"
        
        # 测试不同任务类型的集成复杂度
        task_types_and_expected_complexity = [
            (TaskType.PLANNING, 0.8),
            (TaskType.ARCHITECTURE, 0.9),
            (TaskType.DOCUMENTATION, 0.3),
            (TaskType.KNOWLEDGE_RETRIEVAL, 0.4),
        ]
        
        for task_type, expected_integration in task_types_and_expected_complexity:
            complexity = await complexity_analyzer.analyze_complexity(task_description, task_type)
            assert complexity.factors["integration_complexity"] == expected_integration
    
    @pytest.mark.asyncio
    async def test_error_handling(self, complexity_analyzer):
        """测试错误处理"""
        # 测试空任务描述
        complexity = await complexity_analyzer.analyze_complexity("", TaskType.CODE_GENERATION)
        assert complexity.score >= 0.0
        assert complexity.estimated_time >= 300  # 至少基础时间
        
        # 测试异常情况
        with patch.object(complexity_analyzer, '_determine_required_capabilities', side_effect=Exception("Test error")):
            complexity = await complexity_analyzer.analyze_complexity("test task", TaskType.CODE_GENERATION)
            assert complexity.score == 0.5  # 默认复杂度
            assert complexity.factors.get("error") == 1.0
            assert complexity.estimated_time == 600


class TestAgentCapabilityMatcher:
    """智能体能力匹配器测试"""
    
    @pytest.mark.asyncio
    async def test_find_best_matches_for_code_generation(self, capability_matcher):
        """测试代码生成任务的智能体匹配"""
        complexity = TaskComplexity(
            score=0.6,
            factors={"technical": 0.7},
            estimated_time=900,
            required_capabilities=["代码生成", "编程语言", "调试"]
        )
        
        matches = await capability_matcher.find_best_matches(
            complexity, TaskType.CODE_GENERATION, top_n=3
        )
        
        assert len(matches) <= 3
        assert all(isinstance(match, AgentCapabilityMatch) for match in matches)
        
        # 代码专家应该排名第一
        if matches:
            assert matches[0].agent_name == "code_expert"
            assert matches[0].agent_role == AgentRole.CODE_EXPERT
            assert matches[0].match_score > 0.5
    
    @pytest.mark.asyncio
    async def test_find_best_matches_for_architecture(self, capability_matcher):
        """测试架构设计任务的智能体匹配"""
        complexity = TaskComplexity(
            score=0.8,
            factors={"integration": 0.9},
            estimated_time=1800,
            required_capabilities=["系统设计", "架构模式", "技术选型"]
        )
        
        matches = await capability_matcher.find_best_matches(
            complexity, TaskType.ARCHITECTURE, top_n=2
        )
        
        assert len(matches) <= 2
        if matches:
            # 架构师应该是最佳匹配
            assert matches[0].agent_name == "architect"
            assert matches[0].agent_role == AgentRole.ARCHITECT
    
    @pytest.mark.asyncio
    async def test_capability_alignment(self, capability_matcher, mock_agents):
        """测试能力对齐评分"""
        complexity = TaskComplexity(
            score=0.5,
            factors={},
            estimated_time=600,
            required_capabilities=["文档撰写", "技术写作", "信息架构"]
        )
        
        matches = await capability_matcher.find_best_matches(
            complexity, TaskType.DOCUMENTATION, top_n=4
        )
        
        # 找到文档专家的匹配
        doc_match = next((m for m in matches if m.agent_name == "doc_expert"), None)
        assert doc_match is not None
        assert doc_match.capability_alignment["文档撰写"] == 1.0
        assert doc_match.capability_alignment["技术写作"] == 1.0
    
    @pytest.mark.asyncio
    async def test_load_factor_management(self, capability_matcher):
        """测试负载因子管理"""
        # 初始负载应该为0
        loads = capability_matcher.get_agent_loads()
        assert all(load == 0.0 for load in loads.values())
        
        # 更新负载
        capability_matcher.update_agent_load("code_expert", 0.5)
        loads = capability_matcher.get_agent_loads()
        assert loads["code_expert"] == 0.5
        
        # 负载应该影响可用性
        complexity = TaskComplexity(
            score=0.5,
            factors={},
            estimated_time=600,
            required_capabilities=["代码生成"]
        )
        
        # 设置高负载
        capability_matcher.update_agent_load("code_expert", 0.4)  # 总负载0.9
        matches = await capability_matcher.find_best_matches(
            complexity, TaskType.CODE_GENERATION, top_n=3
        )
        
        code_expert_match = next((m for m in matches if m.agent_name == "code_expert"), None)
        if code_expert_match:
            assert code_expert_match.load_factor == 0.9
            assert code_expert_match.availability == False  # 0.9 > 0.9 阈值，应该不可用
    
    @pytest.mark.asyncio
    async def test_no_available_agents(self, capability_matcher):
        """测试无可用智能体的情况"""
        # 设置所有智能体高负载
        for agent_name in capability_matcher.available_agents.keys():
            capability_matcher.update_agent_load(agent_name, 0.95)
        
        complexity = TaskComplexity(
            score=0.5,
            factors={},
            estimated_time=600,
            required_capabilities=["未知能力"]
        )
        
        matches = await capability_matcher.find_best_matches(
            complexity, TaskType.CODE_GENERATION, top_n=3
        )
        
        # 即使没有完美匹配，也应该返回一些候选
        assert len(matches) > 0


class TestSupervisorAgent:
    """Supervisor智能体决策引擎测试"""
    
    @pytest.mark.asyncio
    async def test_analyze_and_assign_simple_task(self, supervisor_agent):
        """测试简单任务的分析和分配"""
        task_description = "编写一个计算两数之和的函数"
        task_type = TaskType.CODE_GENERATION
        priority = TaskPriority.MEDIUM
        
        assignment = await supervisor_agent.analyze_and_assign_task(
            task_description, task_type, priority
        )
        
        assert isinstance(assignment, TaskAssignment)
        assert assignment.task_id.startswith("task_decision_")
        assert assignment.assigned_agent in ["code_expert", "architect"]
        assert assignment.confidence_level > 0.0
        assert assignment.assignment_reason != ""
        assert isinstance(assignment.estimated_completion_time, datetime)
        assert len(assignment.alternative_agents) <= 2
    
    @pytest.mark.asyncio
    async def test_analyze_and_assign_complex_architecture_task(self, supervisor_agent):
        """测试复杂架构任务的分析和分配"""
        task_description = """
        设计一个分布式微服务架构，包含服务发现、负载均衡、
        熔断器、API网关、分布式追踪和监控系统
        """
        task_type = TaskType.ARCHITECTURE
        priority = TaskPriority.HIGH
        
        assignment = await supervisor_agent.analyze_and_assign_task(
            task_description, task_type, priority
        )
        
        # 架构任务应该分配给架构师
        assert assignment.assigned_agent == "architect"
        assert assignment.confidence_level > 0.5
        assert "架构" in assignment.assignment_reason or "系统设计" in assignment.assignment_reason
        
        # 检查决策元数据
        assert "complexity" in assignment.decision_metadata
        assert assignment.decision_metadata["complexity"]["score"] > 0.5
        assert "match_details" in assignment.decision_metadata
    
    @pytest.mark.asyncio
    async def test_decision_history_recording(self, supervisor_agent):
        """测试决策历史记录"""
        # 执行多个任务分配
        tasks = [
            ("任务1", TaskType.CODE_GENERATION),
            ("任务2", TaskType.DOCUMENTATION),
            ("任务3", TaskType.ANALYSIS)
        ]
        
        for desc, task_type in tasks:
            await supervisor_agent.analyze_and_assign_task(desc, task_type)
        
        # 检查决策历史
        assert len(supervisor_agent.decision_history) == 3
        
        history = await supervisor_agent.get_decision_history(limit=2)
        assert len(history) == 2
        
        # 检查决策记录结构
        for decision_dict in history:
            assert "decision_id" in decision_dict
            assert "timestamp" in decision_dict
            assert "task_description" in decision_dict
            assert "assignment" in decision_dict
            assert "confidence" in decision_dict
    
    @pytest.mark.asyncio
    async def test_load_balancing(self, supervisor_agent):
        """测试负载均衡功能"""
        # 分配第一个任务
        await supervisor_agent.analyze_and_assign_task(
            "高复杂度代码任务", TaskType.CODE_GENERATION
        )
        
        # 检查负载已更新
        loads = supervisor_agent.capability_matcher.get_agent_loads()
        assert loads["code_expert"] > 0.0
        
        # 分配更多任务，应该考虑负载
        for i in range(3):
            await supervisor_agent.analyze_and_assign_task(
                f"代码任务{i}", TaskType.CODE_GENERATION
            )
        
        # 负载应该持续增加
        final_loads = supervisor_agent.capability_matcher.get_agent_loads()
        assert final_loads["code_expert"] > loads["code_expert"]
    
    @pytest.mark.asyncio
    async def test_supervisor_status(self, supervisor_agent):
        """测试Supervisor状态获取"""
        # 执行一些任务
        await supervisor_agent.analyze_and_assign_task(
            "测试任务", TaskType.CODE_GENERATION
        )
        
        status = await supervisor_agent.get_supervisor_status()
        
        assert status["supervisor_name"] == "任务调度器"
        assert status["status"] == "active"
        assert "code_expert" in status["available_agents"]
        assert "agent_loads" in status
        assert status["decision_history_count"] == 1
        assert status["performance_metrics"]["total_decisions"] == 1
    
    @pytest.mark.asyncio
    async def test_no_suitable_agent_error(self, supervisor_agent):
        """测试无合适智能体时的错误处理"""
        # 清空可用智能体
        supervisor_agent.available_agents = {}
        supervisor_agent.capability_matcher = AgentCapabilityMatcher({})
        
        with pytest.raises(ValueError, match="未找到合适的智能体"):
            await supervisor_agent.analyze_and_assign_task(
                "任务", TaskType.CODE_GENERATION
            )
    
    @pytest.mark.asyncio
    async def test_add_remove_agents(self, supervisor_agent):
        """测试动态添加和移除智能体"""
        initial_count = len(supervisor_agent.available_agents)
        
        # 添加新智能体
        new_agent = Mock(spec=BaseAutoGenAgent)
        new_agent.config = AgentConfig(
            name="new_agent",
            role=AgentRole.CODE_EXPERT,
            capabilities=["新能力"],
            system_prompt="新智能体",
            description=""
        )
        
        supervisor_agent.add_agent("new_agent", new_agent)
        assert len(supervisor_agent.available_agents) == initial_count + 1
        assert "new_agent" in supervisor_agent.available_agents
        
        # 移除智能体
        supervisor_agent.remove_agent("new_agent")
        assert len(supervisor_agent.available_agents) == initial_count
        assert "new_agent" not in supervisor_agent.available_agents
    
    @pytest.mark.asyncio
    async def test_assignment_reason_generation(self, supervisor_agent):
        """测试分配理由生成的准确性"""
        # 低复杂度任务
        simple_assignment = await supervisor_agent.analyze_and_assign_task(
            "简单任务", TaskType.DOCUMENTATION
        )
        # 验证分配理由包含关键信息
        assert simple_assignment.assignment_reason != ""
        assert "匹配度" in simple_assignment.assignment_reason
        
        # 高复杂度任务
        complex_assignment = await supervisor_agent.analyze_and_assign_task(
            "设计复杂的分布式系统架构，包括多个微服务、消息队列、缓存层、数据库分片、负载均衡、服务发现、配置中心、监控告警、日志系统等关键组件",
            TaskType.ARCHITECTURE
        )
        # 验证分配理由包含复杂度相关信息
        assert complex_assignment.assignment_reason != ""
        assert "匹配度" in complex_assignment.assignment_reason
    
    @pytest.mark.asyncio
    async def test_constraints_handling(self, supervisor_agent):
        """测试约束条件处理"""
        constraints = {
            "max_time": 300,
            "preferred_agent": "code_expert",
            "require_expertise": ["代码生成", "性能优化"]
        }
        
        assignment = await supervisor_agent.analyze_and_assign_task(
            "优化算法性能",
            TaskType.CODE_GENERATION,
            TaskPriority.HIGH,
            constraints=constraints
        )
        
        assert assignment is not None
        # 约束应该被记录在元数据中（即使当前实现可能未完全使用）
        assert assignment.decision_metadata is not None


class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_task_flow(self, supervisor_agent):
        """测试端到端的任务流程"""
        # 创建不同类型的任务
        tasks = [
            ("实现用户认证系统", TaskType.CODE_GENERATION, TaskPriority.HIGH),
            ("编写API文档", TaskType.DOCUMENTATION, TaskPriority.MEDIUM),
            ("分析系统性能", TaskType.ANALYSIS, TaskPriority.LOW),
            ("设计数据库架构", TaskType.ARCHITECTURE, TaskPriority.HIGH),
        ]
        
        assignments = []
        for desc, task_type, priority in tasks:
            assignment = await supervisor_agent.analyze_and_assign_task(
                desc, task_type, priority
            )
            assignments.append(assignment)
        
        # 验证所有任务都被分配
        assert len(assignments) == len(tasks)
        
        # 验证不同任务类型的分配逻辑
        assert assignments[0].assigned_agent in ["code_expert", "architect"]  # 代码生成
        assert assignments[1].assigned_agent in ["doc_expert", "architect"]  # 文档
        assert assignments[2].assigned_agent in ["architect", "knowledge_retrieval", "code_expert"]  # 分析
        assert assignments[3].assigned_agent == "architect"  # 架构设计
        
        # 检查决策历史
        history = await supervisor_agent.get_decision_history(limit=10)
        assert len(history) == 4
        
        # 检查状态
        status = await supervisor_agent.get_supervisor_status()
        assert status["decision_history_count"] == 4
        assert status["performance_metrics"]["average_confidence"] > 0.0
    
    @pytest.mark.asyncio
    async def test_concurrent_task_assignment(self, supervisor_agent):
        """测试并发任务分配"""
        tasks = [
            supervisor_agent.analyze_and_assign_task(f"任务{i}", TaskType.CODE_GENERATION)
            for i in range(5)
        ]
        
        # 并发执行
        assignments = await asyncio.gather(*tasks)
        
        # 验证所有任务都成功分配
        assert len(assignments) == 5
        assert all(isinstance(a, TaskAssignment) for a in assignments)
        
        # 验证决策ID的唯一性
        decision_ids = [a.task_id for a in assignments]
        assert len(decision_ids) == len(set(decision_ids))
        
        # 检查负载分布
        loads = supervisor_agent.capability_matcher.get_agent_loads()
        assert loads["code_expert"] > 0.0  # 应该有负载增加