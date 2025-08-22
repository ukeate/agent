"""
流控机制测试
测试背压机制、流量控制和任务处理器
"""

import pytest
import asyncio
import time
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

from src.ai.autogen.flow_control import (
    FlowController, BackpressureStrategy, DropPolicy,
    QueueBasedBackpressure, ThroughputBasedBackpressure, AdaptiveBackpressure,
    FlowControlMetrics, TaskInfo, CircuitBreaker
)
from src.ai.autogen.backpressure_task_processor import BackpressureTaskProcessor


@pytest.fixture
def flow_metrics():
    """创建流控指标实例"""
    return FlowControlMetrics(
        queue_size=500,
        throughput=50.0,
        avg_latency=1000.0,
        cpu_usage=0.7,
        memory_usage=0.6,
        success_rate=0.95
    )


@pytest.fixture
def high_pressure_metrics():
    """创建高压力指标实例"""
    return FlowControlMetrics(
        queue_size=1500,
        throughput=150.0,
        avg_latency=8000.0,
        cpu_usage=0.95,
        memory_usage=0.9,
        success_rate=0.6
    )


@pytest.fixture
def task_info():
    """创建任务信息实例"""
    return TaskInfo(
        task_id="test_task_001",
        priority=1,
        created_at=datetime.now(),
        deadline=datetime.now() + timedelta(seconds=300),
        metadata={'data': {'task_type': 'test'}}
    )


class TestBackpressureControllers:
    """背压控制器测试"""
    
    @pytest.mark.asyncio
    async def test_queue_based_backpressure_normal(self, flow_metrics):
        """测试基于队列的背压控制 - 正常情况"""
        controller = QueueBasedBackpressure(threshold=1000)
        
        should_apply = await controller.should_apply_backpressure(flow_metrics)
        assert should_apply is False
        
        throttle_rate = await controller.calculate_throttle_rate(flow_metrics)
        assert throttle_rate == 0.0
    
    @pytest.mark.asyncio
    async def test_queue_based_backpressure_overflow(self, high_pressure_metrics):
        """测试基于队列的背压控制 - 队列溢出"""
        controller = QueueBasedBackpressure(threshold=1000)
        
        should_apply = await controller.should_apply_backpressure(high_pressure_metrics)
        assert should_apply is True
        
        throttle_rate = await controller.calculate_throttle_rate(high_pressure_metrics)
        assert throttle_rate > 0.0
        assert throttle_rate <= 1.0
    
    @pytest.mark.asyncio
    async def test_throughput_based_backpressure(self, high_pressure_metrics):
        """测试基于吞吐量的背压控制"""
        controller = ThroughputBasedBackpressure(max_throughput=100.0)
        
        should_apply = await controller.should_apply_backpressure(high_pressure_metrics)
        assert should_apply is True
        
        throttle_rate = await controller.calculate_throttle_rate(high_pressure_metrics)
        assert throttle_rate > 0.0
        assert throttle_rate <= 0.8
    
    @pytest.mark.asyncio
    async def test_adaptive_backpressure_multiple_factors(self, high_pressure_metrics):
        """测试自适应背压控制 - 多因素综合"""
        controller = AdaptiveBackpressure()
        
        should_apply = await controller.should_apply_backpressure(high_pressure_metrics)
        assert should_apply is True
        
        throttle_rate = await controller.calculate_throttle_rate(high_pressure_metrics)
        assert throttle_rate > 0.0


class TestFlowController:
    """流控器测试"""
    
    @pytest.fixture
    def flow_controller(self):
        """创建流控器实例"""
        return FlowController(
            strategy=BackpressureStrategy.ADAPTIVE,
            drop_policy=DropPolicy.OLDEST,
            max_queue_size=100
        )
    
    @pytest.mark.asyncio
    async def test_flow_controller_startup_shutdown(self, flow_controller):
        """测试流控器启动和关闭"""
        await flow_controller.start()
        assert flow_controller.running is True
        
        await flow_controller.stop()
        assert flow_controller.running is False
    
    @pytest.mark.asyncio
    async def test_submit_task_normal(self, flow_controller):
        """测试正常任务提交"""
        await flow_controller.start()
        
        try:
            # 正常提交任务
            result = await flow_controller.submit_task(
                task_id="test_task_1",
                task_data={"type": "test"},
                priority=1
            )
            assert result is True
            assert flow_controller.task_queue.qsize() == 1
        finally:
            await flow_controller.stop()
    
    @pytest.mark.asyncio
    async def test_submit_task_with_backpressure(self, flow_controller):
        """测试背压情况下的任务提交"""
        await flow_controller.start()
        
        try:
            # 模拟高压力指标
            flow_controller.current_metrics = FlowControlMetrics(
                queue_size=1500,
                throughput=200.0,
                cpu_usage=0.95,
                memory_usage=0.9,
                success_rate=0.5
            )
            
            # 尝试提交任务，可能被背压机制拒绝
            results = []
            for i in range(10):
                result = await flow_controller.submit_task(
                    task_id=f"test_task_{i}",
                    task_data={"type": "test"},
                    priority=1
                )
                results.append(result)
            
            # 应该有部分任务被拒绝
            rejected_count = results.count(False)
            assert rejected_count > 0
            
        finally:
            await flow_controller.stop()
    
    @pytest.mark.asyncio
    async def test_queue_overflow_handling(self, flow_controller):
        """测试队列溢出处理"""
        flow_controller.max_queue_size = 5  # 设置较小的队列大小
        await flow_controller.start()
        
        try:
            # 填满队列
            for i in range(10):
                result = await flow_controller.submit_task(
                    task_id=f"overflow_task_{i}",
                    task_data={"type": "test"},
                    priority=1
                )
                # 前5个应该成功，后面的可能被拒绝或替换
            
            # 队列不应该超过最大大小
            assert flow_controller.task_queue.qsize() <= flow_controller.max_queue_size
            # 应该有任务被丢弃
            assert flow_controller.dropped_tasks > 0
            
        finally:
            await flow_controller.stop()
    
    @pytest.mark.asyncio
    async def test_get_task_and_complete(self, flow_controller, task_info):
        """测试获取任务和完成任务"""
        await flow_controller.start()
        
        try:
            # 提交任务
            await flow_controller.submit_task(
                task_info.task_id,
                task_info.metadata['data'],
                task_info.priority
            )
            
            # 获取任务
            retrieved_task = await flow_controller.get_task()
            assert retrieved_task is not None
            assert retrieved_task.task_id == task_info.task_id
            
            # 完成任务
            await flow_controller.complete_task(
                task_info.task_id, 
                success=True, 
                execution_time=150.0
            )
            
        finally:
            await flow_controller.stop()
    
    def test_get_statistics(self, flow_controller):
        """测试获取统计信息"""
        stats = flow_controller.get_statistics()
        
        assert 'total_tasks' in stats
        assert 'dropped_tasks' in stats
        assert 'throttled_tasks' in stats
        assert 'drop_rate' in stats
        assert 'throttle_rate' in stats
        assert 'current_queue_size' in stats
        assert 'strategy' in stats
        assert 'drop_policy' in stats
    
    def test_get_metrics(self, flow_controller):
        """测试获取指标"""
        metrics = flow_controller.get_current_metrics()
        assert isinstance(metrics, FlowControlMetrics)
        
        history = flow_controller.get_metrics_history()
        assert isinstance(history, list)


class TestCircuitBreaker:
    """断路器测试"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """创建断路器实例"""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30,
            expected_exception=RuntimeError
        )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_normal_operation(self, circuit_breaker):
        """测试断路器正常运行"""
        async def success_func():
            return "success"
        
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_failure_threshold(self, circuit_breaker):
        """测试断路器失败阈值"""
        async def failing_func():
            raise RuntimeError("Test failure")
        
        # 触发失败直到达到阈值
        for i in range(3):
            with pytest.raises(RuntimeError):
                await circuit_breaker.call(failing_func)
        
        # 断路器应该打开
        assert circuit_breaker.state == "OPEN"
        
        # 后续调用应该直接被拒绝
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await circuit_breaker.call(failing_func)
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_recovery(self, circuit_breaker):
        """测试断路器恢复"""
        async def failing_func():
            raise RuntimeError("Test failure")
        
        async def success_func():
            return "recovered"
        
        # 触发失败打开断路器
        for i in range(3):
            with pytest.raises(RuntimeError):
                await circuit_breaker.call(failing_func)
        
        assert circuit_breaker.state == "OPEN"
        
        # 模拟时间过去，断路器尝试半开
        circuit_breaker.last_failure_time = time.time() - 35  # 35秒前
        
        # 成功调用应该重置断路器
        result = await circuit_breaker.call(success_func)
        assert result == "recovered"
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0


class TestBackpressureTaskProcessor:
    """背压任务处理器测试"""
    
    @pytest.fixture
    def mock_enterprise_manager(self):
        """创建模拟企业级管理器"""
        manager = AsyncMock()
        manager._select_agent_from_pool = AsyncMock(return_value="test_agent_001")
        manager.submit_task = AsyncMock(return_value="actual_task_001")
        return manager
    
    @pytest.fixture
    def mock_flow_controller(self):
        """创建模拟流控器"""
        controller = AsyncMock()
        controller.get_task = AsyncMock()
        controller.complete_task = AsyncMock()
        return controller
    
    @pytest.fixture
    def task_processor(self, mock_flow_controller, mock_enterprise_manager):
        """创建任务处理器实例"""
        return BackpressureTaskProcessor(mock_flow_controller, mock_enterprise_manager)
    
    @pytest.mark.asyncio
    async def test_task_processor_startup_shutdown(self, task_processor):
        """测试任务处理器启动关闭"""
        await task_processor.start()
        assert task_processor.running is True
        assert len(task_processor.worker_tasks) > 0
        
        await task_processor.stop()
        assert task_processor.running is False
    
    @pytest.mark.asyncio
    async def test_process_task_success(self, task_processor, task_info, 
                                       mock_flow_controller, mock_enterprise_manager):
        """测试成功处理任务"""
        # 配置任务数据
        task_info.metadata = {
            'data': {
                'pool_id': 'test_pool',
                'task_type': 'test_task',
                'description': 'Test task',
                'input_data': {'key': 'value'},
                'timeout_seconds': 300
            }
        }
        
        # 处理任务
        await task_processor._process_task("worker-0", task_info)
        
        # 验证调用
        mock_enterprise_manager._select_agent_from_pool.assert_called_once_with('test_pool')
        mock_enterprise_manager.submit_task.assert_called_once()
        mock_flow_controller.complete_task.assert_called_once_with(
            task_info.task_id, success=True, execution_time=pytest.approx(0, abs=1000)
        )
    
    @pytest.mark.asyncio
    async def test_process_task_no_agent_available(self, task_processor, task_info,
                                                  mock_flow_controller, mock_enterprise_manager):
        """测试无可用智能体的情况"""
        # 模拟没有可用智能体
        mock_enterprise_manager._select_agent_from_pool.return_value = None
        
        task_info.metadata = {
            'data': {
                'pool_id': 'test_pool',
                'task_type': 'test_task',
                'description': 'Test task',
                'input_data': {},
                'timeout_seconds': 300
            }
        }
        
        await task_processor._process_task("worker-0", task_info)
        
        # 应该标记为失败
        mock_flow_controller.complete_task.assert_called_once_with(
            task_info.task_id, success=False, execution_time=pytest.approx(0, abs=1000)
        )
    
    @pytest.mark.asyncio
    async def test_process_expired_task(self, task_processor, mock_flow_controller):
        """测试处理过期任务"""
        expired_task = TaskInfo(
            task_id="expired_task",
            deadline=datetime.now() - timedelta(seconds=10),  # 已过期
            metadata={'data': {'pool_id': 'test_pool'}}
        )
        
        await task_processor._process_task("worker-0", expired_task)
        
        # 过期任务应该被跳过，标记为失败
        mock_flow_controller.complete_task.assert_called_once_with(
            expired_task.task_id, success=False, execution_time=pytest.approx(0, abs=1000)
        )


@pytest.mark.integration
class TestFlowControlIntegration:
    """流控集成测试"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_flow_control(self):
        """测试端到端流控机制"""
        # 创建流控器
        flow_controller = FlowController(
            strategy=BackpressureStrategy.QUEUE_SIZE,
            max_queue_size=10
        )
        
        await flow_controller.start()
        
        try:
            # 提交一批任务
            submitted_tasks = []
            for i in range(15):  # 超过队列容量
                result = await flow_controller.submit_task(
                    task_id=f"integration_task_{i}",
                    task_data={"batch_id": "batch_001"},
                    priority=i % 3  # 不同优先级
                )
                submitted_tasks.append(result)
            
            # 验证结果
            accepted_tasks = sum(submitted_tasks)
            rejected_tasks = len(submitted_tasks) - accepted_tasks
            
            assert accepted_tasks <= 10  # 不超过队列容量
            assert rejected_tasks > 0    # 有任务被拒绝
            assert flow_controller.dropped_tasks > 0  # 统计正确
            
            # 处理任务
            processed_count = 0
            while processed_count < accepted_tasks:
                task_info = await flow_controller.get_task()
                if task_info:
                    await flow_controller.complete_task(
                        task_info.task_id, 
                        success=True, 
                        execution_time=100.0
                    )
                    processed_count += 1
                else:
                    break
            
            assert processed_count == accepted_tasks
            
        finally:
            await flow_controller.stop()


if __name__ == "__main__":
    pytest.main([__file__])