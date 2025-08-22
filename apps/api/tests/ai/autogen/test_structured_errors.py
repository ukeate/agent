"""
结构化错误处理系统测试
测试错误代码、消息模板、错误构建器和异常处理
"""

import pytest
import json
from datetime import datetime, timezone
from unittest.mock import patch

from src.ai.autogen.structured_errors import (
    ErrorCategory, ErrorSeverity, ErrorContext, StructuredError,
    ErrorCodes, ErrorMessages, ErrorBuilder, StructuredException,
    ErrorFactory, handle_structured_error
)


class TestErrorContext:
    """错误上下文测试"""
    
    def test_error_context_creation(self):
        """测试错误上下文创建"""
        context = ErrorContext(
            node_id="test_node",
            session_id="session_123",
            user_id="user_456",
            agent_id="agent_789",
            task_id="task_abc",
            operation="create_agent",
            component="AgentManager"
        )
        
        assert context.node_id == "test_node"
        assert context.session_id == "session_123"
        assert context.user_id == "user_456"
        assert context.agent_id == "agent_789"
        assert context.task_id == "task_abc"
        assert context.operation == "create_agent"
        assert context.component == "AgentManager"
        assert context.version == "1.0.0"
        assert context.environment == "development"
    
    def test_error_context_to_dict(self):
        """测试错误上下文转换为字典"""
        context = ErrorContext(
            node_id="test_node",
            user_id="user_123",
            additional_data={"request_method": "POST", "endpoint": "/api/agents"}
        )
        
        context_dict = context.to_dict()
        
        assert context_dict["node_id"] == "test_node"
        assert context_dict["user_id"] == "user_123"
        assert "timestamp" in context_dict
        assert context_dict["additional_data"]["request_method"] == "POST"
        assert context_dict["additional_data"]["endpoint"] == "/api/agents"
    
    def test_error_context_default_values(self):
        """测试错误上下文默认值"""
        context = ErrorContext()
        
        assert isinstance(context.timestamp, datetime)
        assert context.node_id is None
        assert context.session_id is None
        assert context.version == "1.0.0"
        assert context.environment == "development"
        assert isinstance(context.additional_data, dict)


class TestStructuredError:
    """结构化错误测试"""
    
    @pytest.fixture
    def sample_context(self):
        """创建样例上下文"""
        return ErrorContext(
            node_id="test_node",
            session_id="session_123",
            operation="test_operation"
        )
    
    @pytest.fixture
    def sample_error(self, sample_context):
        """创建样例结构化错误"""
        return StructuredError(
            code=ErrorCodes.VALIDATION_INPUT_INVALID,
            message="输入数据无效：name字段不能为空",
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=sample_context,
            details={"field_name": "name", "value": ""},
            suggestions=["请提供有效的name值", "检查输入格式是否正确"],
            related_errors=[ErrorCodes.VALIDATION_REQUIRED_FIELD_MISSING]
        )
    
    def test_structured_error_creation(self, sample_error, sample_context):
        """测试结构化错误创建"""
        assert sample_error.code == ErrorCodes.VALIDATION_INPUT_INVALID
        assert sample_error.message == "输入数据无效：name字段不能为空"
        assert sample_error.category == ErrorCategory.VALIDATION
        assert sample_error.severity == ErrorSeverity.MEDIUM
        assert sample_error.context == sample_context
        assert sample_error.details["field_name"] == "name"
        assert len(sample_error.suggestions) == 2
        assert len(sample_error.related_errors) == 1
    
    def test_structured_error_to_dict(self, sample_error):
        """测试结构化错误转换为字典"""
        error_dict = sample_error.to_dict()
        
        assert error_dict["code"] == ErrorCodes.VALIDATION_INPUT_INVALID
        assert error_dict["message"] == "输入数据无效：name字段不能为空"
        assert error_dict["category"] == ErrorCategory.VALIDATION.value
        assert error_dict["severity"] == ErrorSeverity.MEDIUM.value
        assert "context" in error_dict
        assert "details" in error_dict
        assert "suggestions" in error_dict
        assert "related_errors" in error_dict
    
    def test_structured_error_to_json(self, sample_error):
        """测试结构化错误转换为JSON"""
        error_json = sample_error.to_json()
        
        # 验证能够正确解析JSON
        parsed = json.loads(error_json)
        assert parsed["code"] == ErrorCodes.VALIDATION_INPUT_INVALID
        assert parsed["category"] == ErrorCategory.VALIDATION.value
        assert "context" in parsed


class TestErrorMessages:
    """错误消息测试"""
    
    def test_get_message_with_parameters(self):
        """测试获取带参数的错误消息"""
        message = ErrorMessages.get_message(
            ErrorCodes.RESOURCE_NOT_FOUND,
            resource_type="Agent",
            resource_id="agent_123"
        )
        
        assert "Agent agent_123" in message
        assert "资源未找到" in message
    
    def test_get_message_missing_parameters(self):
        """测试缺少参数的错误消息"""
        message = ErrorMessages.get_message(
            ErrorCodes.CONFIG_VALUE_OUT_OF_RANGE,
            config_key="max_size"
            # 故意缺少其他参数
        )
        
        # 应该包含错误信息，指示缺少参数
        assert "缺少参数" in message or "KeyError" in message
    
    def test_get_message_unknown_error_code(self):
        """测试未知错误代码"""
        message = ErrorMessages.get_message("UNKNOWN-9999")
        
        assert "未知错误" in message
        assert "UNKNOWN-9999" in message
    
    def test_get_message_no_parameters(self):
        """测试不需要参数的错误消息"""
        # 使用一个不需要参数的错误代码进行测试
        message = ErrorMessages.get_message(ErrorCodes.SYSTEM_STARTUP_FAILED, reason="测试原因")
        
        assert "系统启动失败" in message
        assert "测试原因" in message


class TestErrorBuilder:
    """错误构建器测试"""
    
    @pytest.fixture
    def sample_context(self):
        """创建样例上下文"""
        return ErrorContext(node_id="builder_test_node")
    
    def test_error_builder_basic_usage(self, sample_context):
        """测试错误构建器基本用法"""
        error = (ErrorBuilder()
                .code(ErrorCodes.VALIDATION_INPUT_INVALID)
                .message("测试验证错误")
                .category(ErrorCategory.VALIDATION)
                .severity(ErrorSeverity.HIGH)
                .context(sample_context)
                .detail("field", "test_field")
                .suggestion("请检查输入")
                .build())
        
        assert error.code == ErrorCodes.VALIDATION_INPUT_INVALID
        assert error.message == "测试验证错误"
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.HIGH
        assert error.context == sample_context
        assert error.details["field"] == "test_field"
        assert "请检查输入" in error.suggestions
    
    def test_error_builder_auto_message_generation(self):
        """测试错误构建器自动消息生成"""
        error = (ErrorBuilder()
                .code(ErrorCodes.RESOURCE_NOT_FOUND)
                .detail("resource_type", "Agent")
                .detail("resource_id", "agent_123")
                .build())
        
        # 消息应该根据错误代码和详细信息自动生成
        assert "Agent agent_123" in error.message
        assert "资源未找到" in error.message
    
    def test_error_builder_category_inference(self):
        """测试错误构建器分类推断"""
        error = (ErrorBuilder()
                .code(ErrorCodes.NETWORK_CONNECTION_FAILED)
                .build())
        
        # 分类应该根据错误代码自动推断
        assert error.category == ErrorCategory.NETWORK
    
    def test_error_builder_severity_inference(self):
        """测试错误构建器严重程度推断"""
        error = (ErrorBuilder()
                .code(ErrorCodes.AUTH_TOKEN_INVALID)
                .build())
        
        # 严重程度应该根据错误代码自动推断
        assert error.severity == ErrorSeverity.HIGH
    
    def test_error_builder_chaining(self):
        """测试错误构建器链式调用"""
        error = (ErrorBuilder()
                .code(ErrorCodes.RATE_LIMIT_EXCEEDED)
                .details({"current_rate": 100, "rate_limit": 50})
                .suggestions(["降低请求频率", "等待重置时间"])
                .related_error(ErrorCodes.RATE_LIMIT_QUOTA_EXCEEDED)
                .with_stacktrace(True)
                .build())
        
        assert error.code == ErrorCodes.RATE_LIMIT_EXCEEDED
        assert error.details["current_rate"] == 100
        assert len(error.suggestions) == 2
        assert len(error.related_errors) == 1
        assert error.stacktrace is not None  # 应该包含堆栈跟踪
    
    def test_error_builder_inner_exception(self):
        """测试错误构建器内部异常"""
        original_exception = ValueError("原始错误")
        
        error = (ErrorBuilder()
                .code(ErrorCodes.SYSTEM_INTERNAL_ERROR)
                .inner_exception(original_exception)
                .build())
        
        assert error.inner_exception == str(original_exception)
    
    def test_error_builder_minimal_usage(self):
        """测试错误构建器最少参数用法"""
        error = ErrorBuilder().build()
        
        # 即使没有提供任何参数，也应该能构建有效的错误
        assert error.code == "UNKNOWN"
        assert error.message == "未知错误"
        assert error.category == ErrorCategory.SYSTEM
        assert error.severity == ErrorSeverity.MEDIUM
        assert isinstance(error.context, ErrorContext)


class TestStructuredException:
    """结构化异常测试"""
    
    @pytest.fixture
    def sample_structured_error(self):
        """创建样例结构化错误"""
        return StructuredError(
            code=ErrorCodes.BUSINESS_AGENT_NOT_AVAILABLE,
            message="智能体不可用",
            category=ErrorCategory.BUSINESS_LOGIC,
            severity=ErrorSeverity.MEDIUM,
            context=ErrorContext(agent_id="agent_123")
        )
    
    def test_structured_exception_creation(self, sample_structured_error):
        """测试结构化异常创建"""
        exception = StructuredException(sample_structured_error)
        
        assert exception.structured_error == sample_structured_error
        assert str(exception) == "智能体不可用"
    
    def test_structured_exception_to_dict(self, sample_structured_error):
        """测试结构化异常转换为字典"""
        exception = StructuredException(sample_structured_error)
        
        exception_dict = exception.to_dict()
        
        assert exception_dict["code"] == ErrorCodes.BUSINESS_AGENT_NOT_AVAILABLE
        assert exception_dict["message"] == "智能体不可用"
        assert exception_dict["category"] == ErrorCategory.BUSINESS_LOGIC.value
    
    def test_structured_exception_to_json(self, sample_structured_error):
        """测试结构化异常转换为JSON"""
        exception = StructuredException(sample_structured_error)
        
        exception_json = exception.to_json()
        
        # 验证JSON格式正确
        parsed = json.loads(exception_json)
        assert parsed["code"] == ErrorCodes.BUSINESS_AGENT_NOT_AVAILABLE
        assert parsed["message"] == "智能体不可用"


class TestErrorFactory:
    """错误工厂测试"""
    
    def test_create_system_error(self):
        """测试创建系统错误"""
        context = ErrorContext(component="TestComponent")
        details = {"operation": "test_operation", "input": "test_input"}
        
        exception = ErrorFactory.create_system_error(
            "测试系统错误", details, context
        )
        
        assert isinstance(exception, StructuredException)
        assert exception.structured_error.code == ErrorCodes.SYSTEM_INTERNAL_ERROR
        assert exception.structured_error.message == "测试系统错误"
        assert exception.structured_error.category == ErrorCategory.SYSTEM
        assert exception.structured_error.severity == ErrorSeverity.HIGH
        assert exception.structured_error.context == context
        assert exception.structured_error.details == details
    
    def test_create_validation_error(self):
        """测试创建验证错误"""
        context = ErrorContext(operation="validate_input")
        
        exception = ErrorFactory.create_validation_error(
            "name", "", "non-empty string", context
        )
        
        assert isinstance(exception, StructuredException)
        assert exception.structured_error.code == ErrorCodes.VALIDATION_INPUT_INVALID
        assert exception.structured_error.category == ErrorCategory.VALIDATION
        assert exception.structured_error.severity == ErrorSeverity.MEDIUM
        assert exception.structured_error.details["field_name"] == "name"
        assert exception.structured_error.details["value"] == ""
        assert exception.structured_error.details["expected_format"] == "non-empty string"
        assert len(exception.structured_error.suggestions) > 0
    
    def test_create_resource_not_found_error(self):
        """测试创建资源未找到错误"""
        context = ErrorContext(operation="find_agent")
        
        exception = ErrorFactory.create_resource_not_found_error(
            "Agent", "agent_123", context
        )
        
        assert isinstance(exception, StructuredException)
        assert exception.structured_error.code == ErrorCodes.RESOURCE_NOT_FOUND
        assert exception.structured_error.category == ErrorCategory.RESOURCE
        assert exception.structured_error.severity == ErrorSeverity.MEDIUM
        assert exception.structured_error.details["resource_type"] == "Agent"
        assert exception.structured_error.details["resource_id"] == "agent_123"
        assert len(exception.structured_error.suggestions) > 0
    
    def test_create_rate_limit_error(self):
        """测试创建限流错误"""
        context = ErrorContext(operation="api_request")
        
        exception = ErrorFactory.create_rate_limit_error(
            100, 50, "2023-01-01T10:00:00Z", context
        )
        
        assert isinstance(exception, StructuredException)
        assert exception.structured_error.code == ErrorCodes.RATE_LIMIT_EXCEEDED
        assert exception.structured_error.category == ErrorCategory.RATE_LIMIT
        assert exception.structured_error.severity == ErrorSeverity.LOW
        assert exception.structured_error.details["current_rate"] == 100
        assert exception.structured_error.details["rate_limit"] == 50
        assert exception.structured_error.details["reset_time"] == "2023-01-01T10:00:00Z"
        assert len(exception.structured_error.suggestions) > 0


class TestErrorHandlingDecorator:
    """错误处理装饰器测试"""
    
    def test_handle_structured_error_passthrough(self):
        """测试结构化异常直接传递"""
        @handle_structured_error
        def test_function():
            raise ErrorFactory.create_system_error("测试错误")
        
        with pytest.raises(StructuredException) as exc_info:
            test_function()
        
        assert exc_info.value.structured_error.message == "测试错误"
    
    def test_handle_structured_error_conversion(self):
        """测试普通异常转换为结构化异常"""
        @handle_structured_error
        def test_function():
            raise ValueError("普通异常")
        
        with pytest.raises(StructuredException) as exc_info:
            test_function()
        
        assert exc_info.value.structured_error.code == ErrorCodes.SYSTEM_INTERNAL_ERROR
        assert "普通异常" in exc_info.value.structured_error.message
        assert exc_info.value.structured_error.inner_exception == "普通异常"
    
    def test_handle_structured_error_success(self):
        """测试正常执行不被影响"""
        @handle_structured_error
        def test_function():
            return "success"
        
        result = test_function()
        assert result == "success"


class TestErrorCodeCategories:
    """错误代码分类测试"""
    
    def test_system_error_codes(self):
        """测试系统错误代码"""
        system_codes = [
            ErrorCodes.SYSTEM_STARTUP_FAILED,
            ErrorCodes.SYSTEM_SHUTDOWN_FAILED,
            ErrorCodes.SYSTEM_RESOURCE_EXHAUSTED,
            ErrorCodes.SYSTEM_INTERNAL_ERROR,
            ErrorCodes.SYSTEM_DEPENDENCY_UNAVAILABLE
        ]
        
        for code in system_codes:
            assert code.startswith("SYS-")
            # 验证消息模板存在
            assert code in ErrorMessages.MESSAGES
    
    def test_configuration_error_codes(self):
        """测试配置错误代码"""
        config_codes = [
            ErrorCodes.CONFIG_FILE_NOT_FOUND,
            ErrorCodes.CONFIG_INVALID_FORMAT,
            ErrorCodes.CONFIG_MISSING_REQUIRED,
            ErrorCodes.CONFIG_VALUE_OUT_OF_RANGE,
            ErrorCodes.CONFIG_VALIDATION_FAILED
        ]
        
        for code in config_codes:
            assert code.startswith("CFG-")
            assert code in ErrorMessages.MESSAGES
    
    def test_validation_error_codes(self):
        """测试验证错误代码"""
        validation_codes = [
            ErrorCodes.VALIDATION_INPUT_INVALID,
            ErrorCodes.VALIDATION_REQUIRED_FIELD_MISSING,
            ErrorCodes.VALIDATION_FORMAT_INVALID,
            ErrorCodes.VALIDATION_LENGTH_INVALID,
            ErrorCodes.VALIDATION_CONSTRAINT_VIOLATED
        ]
        
        for code in validation_codes:
            assert code.startswith("VAL-")
            assert code in ErrorMessages.MESSAGES
    
    def test_business_logic_error_codes(self):
        """测试业务逻辑错误代码"""
        business_codes = [
            ErrorCodes.BUSINESS_AGENT_NOT_AVAILABLE,
            ErrorCodes.BUSINESS_TASK_FAILED,
            ErrorCodes.BUSINESS_INVALID_STATE,
            ErrorCodes.BUSINESS_WORKFLOW_ERROR,
            ErrorCodes.BUSINESS_POOL_EXHAUSTED
        ]
        
        for code in business_codes:
            assert code.startswith("BIZ-")
            assert code in ErrorMessages.MESSAGES


@pytest.mark.integration
class TestStructuredErrorIntegration:
    """结构化错误集成测试"""
    
    def test_complete_error_flow(self):
        """测试完整的错误处理流程"""
        # 1. 创建错误上下文
        context = ErrorContext(
            node_id="integration_node",
            session_id="session_integration",
            user_id="user_integration",
            operation="integration_test"
        )
        
        # 2. 使用错误构建器创建结构化错误
        error = (ErrorBuilder()
                .code(ErrorCodes.BUSINESS_AGENT_NOT_AVAILABLE)
                .context(context)
                .detail("agent_id", "agent_integration")
                .detail("status", "busy")
                .suggestion("请稍后重试")
                .suggestion("检查智能体状态")
                .related_error(ErrorCodes.RESOURCE_IN_USE)
                .build())
        
        # 3. 创建结构化异常
        exception = StructuredException(error)
        
        # 4. 验证完整的错误信息
        error_dict = exception.to_dict()
        
        assert error_dict["code"] == ErrorCodes.BUSINESS_AGENT_NOT_AVAILABLE
        assert "智能体不可用" in error_dict["message"]
        assert error_dict["category"] == ErrorCategory.BUSINESS_LOGIC.value
        assert error_dict["severity"] == ErrorSeverity.MEDIUM.value
        assert error_dict["context"]["node_id"] == "integration_node"
        assert error_dict["details"]["agent_id"] == "agent_integration"
        assert len(error_dict["suggestions"]) == 2
        assert len(error_dict["related_errors"]) == 1
        
        # 5. 验证JSON序列化
        error_json = exception.to_json()
        parsed_error = json.loads(error_json)
        assert parsed_error["code"] == ErrorCodes.BUSINESS_AGENT_NOT_AVAILABLE
        
        # 6. 验证异常字符串表示
        assert str(exception) in error_dict["message"]


if __name__ == "__main__":
    pytest.main([__file__])