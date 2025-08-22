"""
背压管理器测试
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

from src.ai.streaming.backpressure import (
    BackpressureManager, RateLimiter, CircuitBreaker,
    ThrottleLevel, PressureSource, PressureMetrics, ThrottleAction
)


class TestBackpressureManager:
    """背压管理器测试"""
    
    @pytest.fixture
    def backpressure_manager(self):
        """创建背压管理器实例"""
        return BackpressureManager(
            max_buffer_size=1000,
            high_watermark=0.8,
            critical_watermark=0.95,
            check_interval=0.1
        )
    
    def test_init(self, backpressure_manager):
        """测试初始化"""
        assert backpressure_manager.max_buffer_size == 1000
        assert backpressure_manager.high_watermark == 0.8
        assert backpressure_manager.critical_watermark == 0.95
        assert backpressure_manager.current_throttle_level == ThrottleLevel.NONE
        assert not backpressure_manager.is_monitoring
    
    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, backpressure_manager):
        """测试启动和停止监控"""
        # 启动监控
        await backpressure_manager.start_monitoring()
        assert backpressure_manager.is_monitoring
        assert backpressure_manager._monitor_task is not None
        
        # 停止监控
        await backpressure_manager.stop_monitoring()
        assert not backpressure_manager.is_monitoring
    
    def test_update_buffer_usage(self, backpressure_manager):
        """测试更新缓冲区使用情况"""
        backpressure_manager.update_buffer_usage(500)
        assert backpressure_manager.buffer_usage == 500
        
        backpressure_manager.update_buffer_usage(900)
        assert backpressure_manager.buffer_usage == 900
    
    def test_pressure_metrics_creation(self, backpressure_manager):
        """测试压力指标创建"""
        # 设置缓冲区使用率超过阈值
        backpressure_manager.buffer_usage = 850  # 85% utilization
        
        # 手动触发指标收集
        asyncio.run(backpressure_manager._collect_pressure_metrics())
        
        # 检查缓冲区压力指标
        buffer_metric = backpressure_manager.pressure_metrics.get(PressureSource.BUFFER_OVERFLOW)
        assert buffer_metric is not None
        assert buffer_metric.current_value == 0.85
        assert buffer_metric.threshold == 0.8
        assert buffer_metric.is_over_threshold
    
    def test_pressure_analysis(self, backpressure_manager):
        """测试压力分析"""
        # 无压力情况
        level = backpressure_manager._analyze_pressure()
        assert level == ThrottleLevel.NONE
        
        # 添加轻度压力
        backpressure_manager.pressure_metrics[PressureSource.BUFFER_OVERFLOW] = PressureMetrics(
            source=PressureSource.BUFFER_OVERFLOW,
            current_value=0.85,
            threshold=0.8,
            severity=0.2
        )
        
        level = backpressure_manager._analyze_pressure()
        assert level == ThrottleLevel.LIGHT
        
        # 添加重度压力
        backpressure_manager.pressure_metrics[PressureSource.CPU_HIGH] = PressureMetrics(
            source=PressureSource.CPU_HIGH,
            current_value=0.95,
            threshold=0.85,
            severity=0.9
        )
        
        level = backpressure_manager._analyze_pressure()
        assert level == ThrottleLevel.SEVERE
    
    @pytest.mark.asyncio
    async def test_throttle_application(self, backpressure_manager):
        """测试限流应用"""
        # 添加回调函数
        throttle_called = []
        release_called = []
        
        async def throttle_callback(level):
            throttle_called.append(level)
        
        async def release_callback():
            release_called.append(True)
        
        backpressure_manager.add_throttle_callback(throttle_callback)
        backpressure_manager.add_release_callback(release_callback)
        
        # 应用轻度限流
        await backpressure_manager._apply_throttling(ThrottleLevel.LIGHT)
        # Note: current_throttle_level is only updated by _apply_throttle_if_needed, not directly by _apply_throttling
        assert len(backpressure_manager.active_throttles) > 0
        assert throttle_called[-1] == ThrottleLevel.LIGHT
        
        # 释放限流
        await backpressure_manager._release_throttling()
        assert len(backpressure_manager.active_throttles) == 0
        assert len(release_called) > 0
    
    def test_get_current_status(self, backpressure_manager):
        """测试获取当前状态"""
        backpressure_manager.buffer_usage = 600
        
        status = backpressure_manager.get_current_status()
        
        assert "throttle_level" in status
        assert "buffer_usage" in status
        assert "buffer_usage_ratio" in status
        assert "is_monitoring" in status
        assert "pressure_metrics" in status
        assert "active_throttles" in status
        
        assert status["buffer_usage"] == 600
        assert status["buffer_usage_ratio"] == 0.6
    
    def test_configure_threshold(self, backpressure_manager):
        """测试配置阈值"""
        backpressure_manager.configure_threshold(PressureSource.CPU_HIGH, 0.9)
        assert backpressure_manager.thresholds[PressureSource.CPU_HIGH] == 0.9
        
        thresholds = backpressure_manager.get_thresholds()
        assert "cpu_high" in thresholds
        assert thresholds["cpu_high"] == 0.9


class TestRateLimiter:
    """速率限制器测试"""
    
    @pytest.fixture
    def rate_limiter(self):
        """创建速率限制器实例"""
        return RateLimiter(rate=10, per=1.0, burst=15)
    
    def test_init(self, rate_limiter):
        """测试初始化"""
        assert rate_limiter.rate == 10
        assert rate_limiter.per == 1.0
        assert rate_limiter.burst == 15
        assert rate_limiter.allowance == 10.0
    
    @pytest.mark.asyncio
    async def test_acquire_tokens(self, rate_limiter):
        """测试获取令牌"""
        # 初始状态应该有足够的令牌
        assert await rate_limiter.acquire(5)
        assert abs(rate_limiter.allowance - 5.0) < 0.1  # 允许时间误差
        
        # 继续获取令牌
        assert await rate_limiter.acquire(3)
        assert abs(rate_limiter.allowance - 2.0) < 0.1  # 允许时间误差
        
        # 尝试获取超过剩余的令牌
        assert not await rate_limiter.acquire(5)
        assert rate_limiter.allowance == 2.0
    
    @pytest.mark.asyncio
    async def test_token_replenishment(self, rate_limiter):
        """测试令牌补充"""
        # 消耗所有令牌
        await rate_limiter.acquire(10)
        assert rate_limiter.allowance < 0.1  # 几乎为0，允许时间误差
        
        # 等待一段时间让令牌补充
        await asyncio.sleep(0.5)
        
        # 应该能获取一些令牌（但不是全部，因为时间不够）
        assert await rate_limiter.acquire(1)
    
    @pytest.mark.asyncio
    async def test_wait_for_token(self, rate_limiter):
        """测试等待令牌"""
        # 消耗所有令牌
        await rate_limiter.acquire(10)
        
        # 等待令牌可用
        start_time = time.time()
        result = await rate_limiter.wait_for_token(1, timeout=2.0)
        end_time = time.time()
        
        assert result
        assert end_time - start_time < 2.0  # 应该在超时前获得令牌
    
    def test_get_stats(self, rate_limiter):
        """测试获取统计信息"""
        stats = rate_limiter.get_stats()
        
        assert "rate" in stats
        assert "per" in stats
        assert "burst" in stats
        assert "current_allowance" in stats
        assert "total_requests" in stats
        assert "total_allowed" in stats
        assert "total_rejected" in stats
        assert "rejection_rate" in stats
        
        assert stats["rate"] == 10
        assert stats["per"] == 1.0
        assert stats["burst"] == 15
    
    def test_reset(self, rate_limiter):
        """测试重置"""
        # 使用一些令牌
        asyncio.run(rate_limiter.acquire(5))
        
        # 重置
        rate_limiter.reset()
        
        assert rate_limiter.allowance == 10.0
        assert rate_limiter.total_requests == 0
        assert rate_limiter.total_allowed == 0
        assert rate_limiter.total_rejected == 0


class TestCircuitBreaker:
    """熔断器测试"""
    
    @pytest.fixture
    def circuit_breaker(self):
        """创建熔断器实例"""
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0
        )
    
    def test_init(self, circuit_breaker):
        """测试初始化"""
        assert circuit_breaker.failure_threshold == 3
        assert circuit_breaker.recovery_timeout == 1.0
        assert circuit_breaker.failure_count == 0
        assert circuit_breaker.state == "CLOSED"
    
    @pytest.mark.asyncio
    async def test_successful_call(self, circuit_breaker):
        """测试成功调用"""
        async def success_func():
            return "success"
        
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0
    
    @pytest.mark.asyncio
    async def test_failure_handling(self, circuit_breaker):
        """测试失败处理"""
        async def failure_func():
            raise Exception("test error")
        
        # 连续失败直到熔断
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failure_func)
        
        # 应该进入开启状态
        assert circuit_breaker.state == "OPEN"
        assert circuit_breaker.failure_count == 3
        
        # 现在调用应该立即失败
        with pytest.raises(Exception, match="Circuit breaker is OPEN"):
            await circuit_breaker.call(failure_func)
    
    @pytest.mark.asyncio
    async def test_recovery(self, circuit_breaker):
        """测试恢复"""
        async def failure_func():
            raise Exception("test error")
        
        async def success_func():
            return "success"
        
        # 触发熔断
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failure_func)
        
        assert circuit_breaker.state == "OPEN"
        
        # 等待恢复时间
        await asyncio.sleep(1.1)
        
        # 成功调用应该重置熔断器
        result = await circuit_breaker.call(success_func)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"
        assert circuit_breaker.failure_count == 0
    
    def test_get_state(self, circuit_breaker):
        """测试获取状态"""
        state = circuit_breaker.get_state()
        
        assert "state" in state
        assert "failure_count" in state
        assert "failure_threshold" in state
        assert "last_failure_time" in state
        assert "recovery_timeout" in state
        
        assert state["state"] == "CLOSED"
        assert state["failure_count"] == 0
        assert state["failure_threshold"] == 3
        assert state["recovery_timeout"] == 1.0


@pytest.mark.asyncio
async def test_integration_backpressure_with_monitoring():
    """测试背压管理器与监控的集成"""
    manager = BackpressureManager(
        max_buffer_size=100,
        high_watermark=0.8,
        check_interval=0.1
    )
    
    # 记录回调触发
    throttle_events = []
    release_events = []
    
    async def on_throttle(level):
        throttle_events.append(level)
    
    async def on_release():
        release_events.append(time.time())
    
    manager.add_throttle_callback(on_throttle)
    manager.add_release_callback(on_release)
    
    # 启动监控
    await manager.start_monitoring()
    
    try:
        # 模拟缓冲区使用率增加
        manager.update_buffer_usage(85)  # 85% utilization
        
        # 等待监控循环处理
        await asyncio.sleep(0.3)
        
        # 检查是否触发了限流
        assert len(throttle_events) > 0
        assert manager.current_throttle_level != ThrottleLevel.NONE
        
        # 降低缓冲区使用率
        manager.update_buffer_usage(30)  # 30% utilization
        
        # 等待监控循环处理
        await asyncio.sleep(0.3)
        
        # 检查是否释放了限流
        assert len(release_events) > 0
        assert manager.current_throttle_level == ThrottleLevel.NONE
        
    finally:
        await manager.stop_monitoring()


if __name__ == "__main__":
    pytest.main([__file__])