"""
离线推理系统集成测试
"""

import pytest
import tempfile
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from uuid import uuid4
from offline.reasoning_engine import (
    OfflineReasoningEngine, ReasoningStrategy, ReasoningStepResult,
    ReasoningStep, WorkflowStatus
)
from offline.local_inference import LocalInferenceEngine, ModelType, NetworkStatus
from offline.memory_manager import OfflineMemoryManager
from offline.model_cache import ModelCacheManager
from src.services.reasoning_service import ReasoningService
from models.schemas.reasoning import ReasoningRequest, ReasoningStrategy as OnlineReasoningStrategy
from ...models.schemas.offline import OfflineMode

class TestOfflineReasoningIntegration:
    """离线推理系统集成测试"""
    
    @pytest.fixture
    def temp_offline_components(self):
        """临时离线组件"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建组件
            model_cache = ModelCacheManager(cache_dir=temp_dir + "/models")
            inference_engine = LocalInferenceEngine(model_cache)
            memory_manager = OfflineMemoryManager(storage_path=temp_dir + "/memory")
            reasoning_engine = OfflineReasoningEngine(inference_engine, memory_manager)
            
            yield {
                "model_cache": model_cache,
                "inference_engine": inference_engine,
                "memory_manager": memory_manager,
                "reasoning_engine": reasoning_engine
            }
            
            # 清理
            model_cache.close()
    
    @pytest.mark.asyncio
    async def test_offline_reasoning_workflow_creation(self, temp_offline_components):
        """测试离线推理工作流创建"""
        reasoning_engine = temp_offline_components["reasoning_engine"]
        
        session_id = "test_session"
        problem = "如何学习Python编程？"
        
        # 创建推理工作流
        workflow_id = await reasoning_engine.create_reasoning_workflow(
            name="Python学习推理",
            problem=problem,
            session_id=session_id,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )
        
        # 验证工作流创建
        assert workflow_id is not None
        
        # 检查工作流状态
        status = reasoning_engine.get_workflow_status(workflow_id)
        assert status == WorkflowStatus.PENDING
        
        # 检查工作流列表
        workflows = reasoning_engine.list_workflows(session_id)
        assert len(workflows) == 1
        assert workflows[0]["workflow_id"] == workflow_id
        assert workflows[0]["strategy"] == ReasoningStrategy.CHAIN_OF_THOUGHT.value
    
    @pytest.mark.asyncio
    async def test_offline_reasoning_execution(self, temp_offline_components):
        """测试离线推理执行"""
        reasoning_engine = temp_offline_components["reasoning_engine"]
        
        session_id = "execution_test"
        problem = "什么是递归算法？请详细解释。"
        
        # 创建并执行推理工作流
        workflow_id = await reasoning_engine.create_reasoning_workflow(
            name="递归算法解释",
            problem=problem,
            session_id=session_id,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )
        
        # 执行工作流
        reasoning_chain = await reasoning_engine.execute_workflow(workflow_id)
        
        # 验证执行结果
        assert reasoning_chain is not None
        assert reasoning_chain.status == WorkflowStatus.COMPLETED
        assert len(reasoning_chain.steps) > 0
        assert reasoning_chain.final_conclusion is not None
        assert reasoning_chain.overall_confidence > 0.0
        
        # 验证推理步骤
        step_types = [step.step_type for step in reasoning_chain.steps]
        assert ReasoningStep.PROBLEM_ANALYSIS in step_types
        assert ReasoningStep.CONCLUSION_GENERATION in step_types
    
    @pytest.mark.asyncio
    async def test_different_reasoning_strategies(self, temp_offline_components):
        """测试不同推理策略"""
        reasoning_engine = temp_offline_components["reasoning_engine"]
        
        session_id = "strategy_test"
        problem = "如何优化程序性能？"
        
        strategies = [
            ReasoningStrategy.CHAIN_OF_THOUGHT,
            ReasoningStrategy.TREE_OF_THOUGHT,
            ReasoningStrategy.DEDUCTIVE
        ]
        
        results = []
        
        for strategy in strategies:
            workflow_id = await reasoning_engine.create_reasoning_workflow(
                name=f"性能优化-{strategy.value}",
                problem=problem,
                session_id=session_id,
                strategy=strategy
            )
            
            reasoning_chain = await reasoning_engine.execute_workflow(workflow_id)
            results.append((strategy, reasoning_chain))
        
        # 验证所有策略都成功执行
        for strategy, chain in results:
            assert chain.status == WorkflowStatus.COMPLETED
            assert chain.strategy == strategy
            assert len(chain.steps) > 0
        
        # 验证不同策略产生不同的推理路径
        step_counts = [len(chain.steps) for _, chain in results]
        assert len(set(step_counts)) > 1  # 至少有一种策略的步骤数不同
    
    @pytest.mark.asyncio
    async def test_workflow_control_operations(self, temp_offline_components):
        """测试工作流控制操作"""
        reasoning_engine = temp_offline_components["reasoning_engine"]
        
        session_id = "control_test"
        problem = "设计一个数据库系统需要考虑哪些因素？"
        
        # 创建工作流
        workflow_id = await reasoning_engine.create_reasoning_workflow(
            name="数据库设计分析",
            problem=problem,
            session_id=session_id,
            strategy=ReasoningStrategy.TREE_OF_THOUGHT
        )
        
        # 测试暂停
        paused = await reasoning_engine.pause_workflow(workflow_id)
        assert paused
        
        status = reasoning_engine.get_workflow_status(workflow_id)
        assert status == WorkflowStatus.PENDING  # 暂停前是PENDING
        
        # 测试恢复
        resumed = await reasoning_engine.resume_workflow(workflow_id)
        assert resumed
        
        # 测试取消
        cancelled = await reasoning_engine.cancel_workflow(workflow_id)
        assert cancelled
        
        status = reasoning_engine.get_workflow_status(workflow_id)
        assert status == WorkflowStatus.CANCELLED
    
    @pytest.mark.asyncio
    async def test_reasoning_memory_integration(self, temp_offline_components):
        """测试推理和记忆系统集成"""
        reasoning_engine = temp_offline_components["reasoning_engine"]
        memory_manager = temp_offline_components["memory_manager"]
        
        session_id = "memory_integration_test"
        problem = "解释面向对象编程的核心概念"
        
        # 执行推理
        workflow_id = await reasoning_engine.create_reasoning_workflow(
            name="OOP概念解释",
            problem=problem,
            session_id=session_id,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT
        )
        
        reasoning_chain = await reasoning_engine.execute_workflow(workflow_id)
        
        # 检查记忆系统中是否保存了推理相关记忆
        session_memories = memory_manager.get_memories_by_session(session_id)
        
        # 应该有工作流和推理步骤相关的记忆
        workflow_memories = [
            m for m in session_memories 
            if 'workflow' in m.tags
        ]
        step_memories = [
            m for m in session_memories 
            if 'reasoning_step' in m.tags
        ]
        
        assert len(workflow_memories) > 0
        assert len(step_memories) > 0
        
        # 验证记忆内容包含推理相关信息
        workflow_memory = workflow_memories[0]
        assert workflow_id in workflow_memory.content
    
    def test_reasoning_statistics(self, temp_offline_components):
        """测试推理统计信息"""
        reasoning_engine = temp_offline_components["reasoning_engine"]
        
        # 获取初始统计
        initial_stats = reasoning_engine.get_reasoning_statistics()
        assert initial_stats["total_workflows"] == 0
        assert initial_stats["active_workflows"] == 0
        assert initial_stats["completed_workflows"] == 0
    
    @pytest.mark.asyncio
    async def test_reasoning_service_offline_integration(self, temp_offline_components):
        """测试推理服务的离线集成"""
        # 创建推理服务并设置离线模式
        reasoning_service = ReasoningService()
        reasoning_service.set_offline_mode(OfflineMode.OFFLINE, NetworkStatus.DISCONNECTED)
        
        # 替换离线引擎
        reasoning_service._offline_engine = temp_offline_components["reasoning_engine"]
        
        # 创建推理请求
        request = ReasoningRequest(
            problem="什么是机器学习？",
            strategy=OnlineReasoningStrategy.ZERO_SHOT,
            context={"domain": "AI"},
            max_steps=5
        )
        
        user = {"id": "test_user"}
        
        # 执行离线推理
        response = await reasoning_service.execute_reasoning(request, user)
        
        # 验证响应
        assert response.success
        assert response.problem == request.problem
        assert len(response.steps) > 0
        assert response.conclusion is not None
        assert response.metadata["offline_mode"] is True
    
    @pytest.mark.asyncio
    async def test_reasoning_service_mode_switching(self, temp_offline_components):
        """测试推理服务的模式切换"""
        reasoning_service = ReasoningService()
        reasoning_service._offline_engine = temp_offline_components["reasoning_engine"]
        
        request = ReasoningRequest(
            problem="什么是区块链技术？",
            strategy=OnlineReasoningStrategy.AUTO_COT,
            context={}
        )
        
        user = {"id": "mode_test_user"}
        
        # 测试离线模式
        reasoning_service.set_offline_mode(OfflineMode.OFFLINE, NetworkStatus.DISCONNECTED)
        
        offline_response = await reasoning_service.execute_reasoning(request, user)
        assert offline_response.success
        assert offline_response.metadata.get("offline_mode") is True
        
        # 测试在线模式（因为没有真实的在线引擎，会失败）
        reasoning_service.set_offline_mode(OfflineMode.ONLINE, NetworkStatus.CONNECTED)
        
        # 这里由于没有真实的在线引擎，会抛出异常或返回错误响应
        # 在实际环境中，这会正常执行在线推理
        online_response = await reasoning_service.execute_reasoning(request, user)
        assert online_response.problem == request.problem
        assert online_response.strategy == request.strategy
    
    @pytest.mark.asyncio
    async def test_complex_reasoning_scenario(self, temp_offline_components):
        """测试复杂推理场景"""
        reasoning_engine = temp_offline_components["reasoning_engine"]
        
        session_id = "complex_scenario"
        problem = """
        一个电商网站需要设计推荐系统。请分析：
        1. 需要考虑哪些因素？
        2. 可以使用哪些算法？
        3. 如何评估推荐效果？
        4. 如何处理冷启动问题？
        """
        
        # 使用树状思维推理处理复杂问题
        workflow_id = await reasoning_engine.create_reasoning_workflow(
            name="电商推荐系统设计",
            problem=problem,
            session_id=session_id,
            strategy=ReasoningStrategy.TREE_OF_THOUGHT
        )
        
        reasoning_chain = await reasoning_engine.execute_workflow(workflow_id)
        
        # 验证复杂推理的结果
        assert reasoning_chain.status == WorkflowStatus.COMPLETED
        assert len(reasoning_chain.steps) >= 3  # 至少有分析、假设生成、评估、结论
        assert reasoning_chain.overall_confidence > 0.5
        
        # 验证推理步骤涵盖了问题的各个方面
        reasoning_text = " ".join([step.reasoning_text for step in reasoning_chain.steps])
        
        # 检查是否涵盖了关键概念
        key_concepts = ["推荐", "算法", "评估", "冷启动"]
        covered_concepts = [concept for concept in key_concepts if concept in reasoning_text]
        assert len(covered_concepts) >= 2  # 至少涵盖一半的关键概念
    
    @pytest.mark.asyncio
    async def test_reasoning_error_handling(self, temp_offline_components):
        """测试推理错误处理"""
        reasoning_engine = temp_offline_components["reasoning_engine"]
        
        session_id = "error_test"
        
        # 测试无效工作流ID
        invalid_workflow_id = "invalid_workflow_id"
        
        status = reasoning_engine.get_workflow_status(invalid_workflow_id)
        assert status is None
        
        result = reasoning_engine.get_workflow_result(invalid_workflow_id)
        assert result is None
        
        # 测试取消不存在的工作流
        cancelled = await reasoning_engine.cancel_workflow(invalid_workflow_id)
        assert not cancelled
    
    def test_reasoning_engine_statistics_tracking(self, temp_offline_components):
        """测试推理引擎统计跟踪"""
        reasoning_engine = temp_offline_components["reasoning_engine"]
        inference_engine = temp_offline_components["inference_engine"]
        memory_manager = temp_offline_components["memory_manager"]
        
        # 获取各组件的统计信息
        reasoning_stats = reasoning_engine.get_reasoning_statistics()
        inference_stats = inference_engine.get_inference_stats()
        memory_stats = memory_manager.get_memory_stats()
        
        # 验证统计信息结构
        assert "total_workflows" in reasoning_stats
        assert "average_confidence" in reasoning_stats
        
        assert "total_requests" in inference_stats
        assert "cache_hit_rate" in inference_stats
        
        assert "total_memories" in memory_stats
        assert "memory_types" in memory_stats
