"""
错误处理和重试策略测试
"""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from ai.langgraph.state import MessagesState, create_initial_state
from ai.langgraph.error_handling import (
    ErrorType, ErrorSeverity, WorkflowError, ErrorHandler, 
    RetryStrategy, WorkflowErrorRecovery, workflow_retry
)


class TestWorkflowError:
    """WorkflowError测试"""
    
    def test_workflow_error_creation(self):
        """测试工作流错误创建"""
        error = WorkflowError(
            error_type=ErrorType.NETWORK_ERROR,
            severity=ErrorSeverity.HIGH,
            message="网络连接失败",
            node_name="api_call_node",
            workflow_id="test-workflow"
        )
        
        assert error.error_type == ErrorType.NETWORK_ERROR
        assert error.severity == ErrorSeverity.HIGH
        assert error.message == "网络连接失败"
        assert error.node_name == "api_call_node"
        assert error.workflow_id == "test-workflow"
        assert error.retry_count == 0
        assert error.max_retries == 3
        assert error.is_recoverable == True
        assert isinstance(error.timestamp, datetime)


class TestErrorHandler:
    """ErrorHandler测试"""
    
    def setUp(self):
        self.handler = ErrorHandler()
        self.state = create_initial_state("test-workflow")
    
    @pytest.mark.asyncio
    async def test_error_classification(self):
        """测试错误分类"""
        # 测试不同类型的错误
        test_cases = [
            (asyncio.TimeoutError("超时"), ErrorType.TIMEOUT_ERROR, ErrorSeverity.HIGH),
            (ValueError("无效参数"), ErrorType.VALIDATION_ERROR, ErrorSeverity.LOW),
            (ConnectionError("连接失败"), ErrorType.NETWORK_ERROR, ErrorSeverity.MEDIUM),
            (RuntimeError("运行时错误"), ErrorType.SYSTEM_ERROR, ErrorSeverity.HIGH),
            (Exception("未知错误"), ErrorType.UNKNOWN_ERROR, ErrorSeverity.MEDIUM),
        ]
        
        for exception, expected_type, expected_severity in test_cases:
            workflow_error = await self.handler.handle_error(exception, self.state)
            assert workflow_error.error_type == expected_type
            assert workflow_error.severity == expected_severity
    
    @pytest.mark.asyncio
    async def test_error_state_update(self):
        """测试错误状态更新"""
        exception = ValueError("测试错误")
        
        workflow_error = await self.handler.handle_error(exception, self.state)
        
        # 检查状态更新
        assert self.state["metadata"]["status"] == "error"
        assert self.state["metadata"]["last_error"] == "测试错误"
        assert "error_timestamp" in self.state["metadata"]
        
        # 检查错误日志
        assert "errors" in self.state["context"]
        assert len(self.state["context"]["errors"]) == 1
        error_log = self.state["context"]["errors"][0]
        assert error_log["message"] == "测试错误"
        assert error_log["type"] == ErrorType.VALIDATION_ERROR.value
    
    @pytest.mark.asyncio
    async def test_custom_error_handler(self):
        """测试自定义错误处理器"""
        custom_handler_called = False
        
        async def custom_network_handler(error: WorkflowError, state: MessagesState):
            nonlocal custom_handler_called
            custom_handler_called = True
            state["context"]["custom_handled"] = True
        
        self.handler.register_handler(ErrorType.NETWORK_ERROR, custom_network_handler)
        
        exception = ConnectionError("网络错误")
        await self.handler.handle_error(exception, self.state)
        
        assert custom_handler_called
        assert self.state["context"]["custom_handled"] == True
    
    @pytest.mark.asyncio
    async def test_global_error_callback(self):
        """测试全局错误回调"""
        callback_called = False
        
        async def global_callback(error: WorkflowError, state: MessagesState):
            nonlocal callback_called
            callback_called = True
            state["context"]["global_callback_executed"] = True
        
        self.handler.add_global_callback(global_callback)
        
        exception = Exception("测试错误")
        await self.handler.handle_error(exception, self.state)
        
        assert callback_called
        assert self.state["context"]["global_callback_executed"] == True


class TestRetryStrategy:
    """RetryStrategy测试"""
    
    def setUp(self):
        self.strategy = RetryStrategy(max_attempts=3, base_delay=1.0, max_delay=10.0)
    
    def test_should_retry_recoverable_error(self):
        """测试可恢复错误的重试判断"""
        error = WorkflowError(
            error_type=ErrorType.NETWORK_ERROR,
            severity=ErrorSeverity.MEDIUM,
            is_recoverable=True,
            retry_count=1,
            max_retries=3
        )
        
        assert self.strategy.should_retry(error) == True
    
    def test_should_not_retry_unrecoverable_error(self):
        """测试不可恢复错误的重试判断"""
        error = WorkflowError(
            error_type=ErrorType.VALIDATION_ERROR,
            severity=ErrorSeverity.LOW,
            is_recoverable=False,
            retry_count=0,
            max_retries=3
        )
        
        assert self.strategy.should_retry(error) == False
    
    def test_should_not_retry_max_attempts_reached(self):
        """测试达到最大重试次数"""
        error = WorkflowError(
            error_type=ErrorType.NETWORK_ERROR,
            severity=ErrorSeverity.MEDIUM,
            is_recoverable=True,
            retry_count=3,
            max_retries=3
        )
        
        assert self.strategy.should_retry(error) == False
    
    def test_should_not_retry_validation_error(self):
        """测试验证错误不重试"""
        error = WorkflowError(
            error_type=ErrorType.VALIDATION_ERROR,
            retry_count=0,
            max_retries=3
        )
        
        assert self.strategy.should_retry(error) == False
    
    def test_should_not_retry_business_logic_error(self):
        """测试业务逻辑错误不重试"""
        error = WorkflowError(
            error_type=ErrorType.BUSINESS_LOGIC_ERROR,
            retry_count=0,
            max_retries=3
        )
        
        assert self.strategy.should_retry(error) == False
    
    def test_critical_error_limited_retries(self):
        """测试严重错误限制重试次数"""
        error = WorkflowError(
            error_type=ErrorType.SYSTEM_ERROR,
            severity=ErrorSeverity.CRITICAL,
            is_recoverable=True,
            retry_count=1,
            max_retries=3
        )
        
        assert self.strategy.should_retry(error) == False  # 严重错误只重试一次
    
    def test_retry_delay_calculation(self):
        """测试重试延迟计算"""
        # 指数退避测试
        assert self.strategy.get_retry_delay(0) == 1.0  # 2^0 * 1.0
        assert self.strategy.get_retry_delay(1) == 2.0  # 2^1 * 1.0
        assert self.strategy.get_retry_delay(2) == 4.0  # 2^2 * 1.0
        assert self.strategy.get_retry_delay(3) == 8.0  # 2^3 * 1.0
        
        # 最大延迟限制
        assert self.strategy.get_retry_delay(10) == 10.0  # 受max_delay限制


class TestWorkflowErrorRecovery:
    """WorkflowErrorRecovery测试"""
    
    def setUp(self):
        self.recovery = WorkflowErrorRecovery()
        self.state = create_initial_state("test-workflow")
    
    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """测试成功执行"""
        def successful_func(state: MessagesState) -> str:
            return "执行成功"
        
        result = await self.recovery.execute_with_recovery(
            successful_func, 
            self.state
        )
        
        assert result == "执行成功"
        assert self.state["context"]["retry_info"]["success"] == True
    
    @pytest.mark.asyncio
    async def test_retry_on_recoverable_error(self):
        """测试可恢复错误的重试"""
        call_count = 0
        
        def failing_then_succeeding_func(state: MessagesState) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("网络错误")
            return "最终成功"
        
        result = await self.recovery.execute_with_recovery(
            failing_then_succeeding_func,
            self.state
        )
        
        assert result == "最终成功"
        assert call_count == 3
        assert self.state["context"]["retry_info"]["success"] == True
    
    @pytest.mark.asyncio
    async def test_failure_after_max_retries(self):
        """测试达到最大重试次数后失败"""
        def always_failing_func(state: MessagesState) -> str:
            raise ConnectionError("持续网络错误")
        
        with pytest.raises(RuntimeError) as exc_info:
            await self.recovery.execute_with_recovery(
                always_failing_func,
                self.state
            )
        
        assert "已重试3次" in str(exc_info.value)
        assert self.state["metadata"]["status"] == "failed"
        assert "final_error" in self.state["context"]
    
    @pytest.mark.asyncio
    async def test_no_retry_on_validation_error(self):
        """测试验证错误不重试"""
        call_count = 0
        
        def validation_error_func(state: MessagesState) -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("参数验证失败")
        
        with pytest.raises(RuntimeError):
            await self.recovery.execute_with_recovery(
                validation_error_func,
                self.state
            )
        
        assert call_count == 1  # 只调用一次，不重试
        assert self.state["metadata"]["status"] == "failed"
    
    @pytest.mark.asyncio
    async def test_async_function_execution(self):
        """测试异步函数执行"""
        async def async_func(state: MessagesState) -> str:
            await asyncio.sleep(0.1)
            return "异步执行成功"
        
        result = await self.recovery.execute_with_recovery(
            async_func,
            self.state
        )
        
        assert result == "异步执行成功"
    
    @pytest.mark.asyncio
    async def test_retry_delay_applied(self):
        """测试重试延迟"""
        call_times = []
        
        def timing_func(state: MessagesState) -> str:
            call_times.append(datetime.now())
            if len(call_times) < 2:
                raise ConnectionError("网络错误")
            return "延迟重试成功"
        
        # 使用较短的延迟进行测试
        recovery = WorkflowErrorRecovery(
            retry_strategy=RetryStrategy(max_attempts=3, base_delay=0.1, max_delay=1.0)
        )
        
        start_time = datetime.now()
        result = await recovery.execute_with_recovery(timing_func, self.state)
        end_time = datetime.now()
        
        assert result == "延迟重试成功"
        assert len(call_times) == 2
        
        # 检查总执行时间大于重试延迟
        total_time = (end_time - start_time).total_seconds()
        assert total_time >= 0.1  # 至少有一次延迟


class TestWorkflowRetryDecorator:
    """workflow_retry装饰器测试"""
    
    @pytest.mark.asyncio
    async def test_retry_decorator_success(self):
        """测试重试装饰器成功情况"""
        call_count = 0
        
        @workflow_retry(max_attempts=3, delay=0.1)
        def decorated_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("网络错误")
            return "装饰器重试成功"
        
        result = decorated_func()
        assert result == "装饰器重试成功"
        assert call_count == 2
    
    @pytest.mark.asyncio
    async def test_retry_decorator_max_attempts(self):
        """测试重试装饰器达到最大尝试次数"""
        call_count = 0
        
        @workflow_retry(max_attempts=2, delay=0.1)
        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("持续网络错误")
        
        with pytest.raises(ConnectionError):
            always_failing_func()
        
        assert call_count == 2  # 尝试了2次


class TestErrorHandlingIntegration:
    """错误处理集成测试"""
    
    @pytest.mark.asyncio
    async def test_error_recovery_with_state_logging(self):
        """测试错误恢复与状态日志记录"""
        recovery = WorkflowErrorRecovery()
        state = create_initial_state("integration-test")
        
        call_count = 0
        
        def intermittent_failure_func(state: MessagesState) -> MessagesState:
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                raise ConnectionError("第一次网络错误")
            elif call_count == 2:
                raise asyncio.TimeoutError("第二次超时错误")
            else:
                state["messages"].append({
                    "role": "system",
                    "content": "最终执行成功"
                })
                return state
        
        result = await recovery.execute_with_recovery(
            intermittent_failure_func,
            state
        )
        
        # 验证最终结果
        assert len(result["messages"]) == 1
        assert result["messages"][0]["content"] == "最终执行成功"
        
        # 验证错误日志记录
        assert "errors" in state["context"]
        assert len(state["context"]["errors"]) == 2  # 记录了两次错误
        
        # 验证重试信息
        assert state["context"]["retry_info"]["success"] == True
        assert state["context"]["retry_info"]["max_attempts"] == 3
        
        # 验证错误处理日志
        error_types = [error["type"] for error in state["context"]["errors"]]
        assert ErrorType.NETWORK_ERROR.value in error_types
        assert ErrorType.TIMEOUT_ERROR.value in error_types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])