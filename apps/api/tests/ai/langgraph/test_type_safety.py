"""
类型安全系统测试
测试增强的AgentContext类型系统、验证器和类型工具
"""
import pytest
from typing import Dict, Any, List, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
import uuid
from pydantic import ValidationError, BaseModel, Field

from src.ai.langgraph.context import (
    AgentContext,
    UserPreferences,
    SessionContext,
    WorkflowMetadata,
    ContextVersion,
    create_context,
    validate_context
)
from src.ai.langgraph.validators import (
    TypeValidator,
    ContextTypeGuard
)
from src.ai.langgraph.type_utils import (
    TypeSafeSerializer,
    TypeRegistry,
    create_typed_context,
    cast_context
)


class CustomData(BaseModel):
    """测试用自定义数据类型"""
    name: str = Field(..., description="名称")
    value: int = Field(..., ge=0, description="值")
    tags: List[str] = Field(default_factory=list, description="标签")


class TestAgentContext:
    """测试AgentContext类型系统"""
    
    def test_create_basic_context(self):
        """测试创建基础上下文"""
        session_id = str(uuid.uuid4())
        context = create_context(
            user_id="test_user",
            session_id=session_id
        )
        
        assert context.user_id == "test_user"
        assert context.session_id == session_id
        assert context.version == ContextVersion.CURRENT
        assert isinstance(context.user_preferences, UserPreferences)
        assert isinstance(context.session_context, SessionContext)
        assert isinstance(context.workflow_metadata, WorkflowMetadata)
    
    def test_nested_types(self):
        """测试复杂嵌套类型"""
        session_id = str(uuid.uuid4())
        context = AgentContext(
            user_id="test_user",
            session_id=session_id,
            session_context=SessionContext(session_id=session_id),
            user_preferences=UserPreferences(
                language="en-US",
                timezone="UTC",
                theme="dark",
                custom_settings={"key": "value"}
            )
        )
        
        assert context.user_preferences.language == "en-US"
        assert context.user_preferences.theme == "dark"
        assert context.user_preferences.custom_settings["key"] == "value"
        assert context.session_context.session_id == session_id
    
    def test_generic_type_support(self):
        """测试泛型类型支持"""
        session_id = str(uuid.uuid4())
        custom_data = CustomData(name="test", value=42, tags=["tag1", "tag2"])
        
        context = create_typed_context(
            user_id="test_user",
            session_id=session_id,
            custom_type=CustomData,
            custom_data=custom_data
        )
        
        assert context.custom_data == custom_data
        assert context.custom_data.name == "test"
        assert context.custom_data.value == 42
        assert len(context.custom_data.tags) == 2
    
    def test_field_validation(self):
        """测试字段验证"""
        session_id = str(uuid.uuid4())
        
        # 测试有效值
        context = AgentContext(
            user_id="test_user",
            session_id=session_id,
            session_context=SessionContext(session_id=session_id),
            max_iterations=100,
            timeout_seconds=600
        )
        assert context.max_iterations == 100
        
        # 测试无效值
        with pytest.raises(ValidationError):
            AgentContext(
                user_id="test_user",
                session_id=session_id,
                session_context=SessionContext(session_id=session_id),
                max_iterations=0  # 必须 >= 1
            )
        
        with pytest.raises(ValidationError):
            AgentContext(
                user_id="test_user",
                session_id=session_id,
                session_context=SessionContext(session_id=session_id),
                timeout_seconds=5000  # 必须 <= 3600
            )
    
    def test_session_id_consistency(self):
        """测试session_id一致性验证"""
        session_id1 = str(uuid.uuid4())
        session_id2 = str(uuid.uuid4())
        
        # 测试不一致的session_id
        with pytest.raises(ValidationError) as exc_info:
            AgentContext(
                user_id="test_user",
                session_id=session_id1,
                session_context=SessionContext(session_id=session_id2)
            )
        assert "不匹配" in str(exc_info.value)
    
    def test_status_validation(self):
        """测试状态值验证"""
        session_id = str(uuid.uuid4())
        
        # 测试有效状态
        for status in ["running", "paused", "completed", "failed", "cancelled"]:
            context = AgentContext(
                user_id="test_user",
                session_id=session_id,
                session_context=SessionContext(session_id=session_id),
                status=status
            )
            assert context.status == status
        
        # 测试无效状态
        with pytest.raises(ValidationError):
            AgentContext(
                user_id="test_user",
                session_id=session_id,
                session_context=SessionContext(session_id=session_id),
                status="invalid_status"
            )
    
    def test_update_step(self):
        """测试步骤更新功能"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id)
        
        assert context.step_count == 0
        assert context.current_node is None
        
        context.update_step("node1")
        assert context.step_count == 1
        assert context.current_node == "node1"
        assert context.last_updated is not None
        assert len(context.workflow_metadata.execution_path) == 1
        
        context.update_step("node2")
        assert context.step_count == 2
        assert context.current_node == "node2"
        assert len(context.workflow_metadata.execution_path) == 2
    
    def test_checkpoint_management(self):
        """测试检查点管理"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id)
        
        # 添加检查点
        context.add_checkpoint({"data": "checkpoint1"})
        assert len(context.workflow_metadata.checkpoints) == 1
        
        context.update_step("node1")
        context.add_checkpoint({"data": "checkpoint2"})
        assert len(context.workflow_metadata.checkpoints) == 2
        
        # 验证检查点内容
        checkpoint = context.workflow_metadata.checkpoints[0]
        assert checkpoint["data"]["data"] == "checkpoint1"
        assert "timestamp" in checkpoint
        assert checkpoint["step_count"] == 0
    
    def test_timeout_check(self):
        """测试超时检查"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id, timeout_seconds=1)
        
        # 初始状态不应超时
        assert not context.is_timeout()
        
        # 设置过去的时间
        context.last_updated = utc_now()
        import time
        time.sleep(1.1)
        assert context.is_timeout()
    
    def test_max_iterations_check(self):
        """测试最大迭代次数检查"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id, max_iterations=3)
        
        assert not context.is_max_iterations_reached()
        
        context.update_step("node1")
        context.update_step("node2")
        assert not context.is_max_iterations_reached()
        
        context.update_step("node3")
        assert context.is_max_iterations_reached()


class TestTypeValidator:
    """测试类型验证器"""
    
    def test_validate_context_type(self):
        """测试上下文类型验证"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id)
        
        # 测试基础类型验证
        assert TypeValidator.validate_context_type(context, AgentContext)
        
        # 测试非上下文对象
        assert not TypeValidator.validate_context_type({"key": "value"}, AgentContext)
        assert not TypeValidator.validate_context_type(None, AgentContext)
    
    def test_type_compatibility_check(self):
        """测试类型兼容性检查"""
        # 测试基础类型
        assert TypeValidator._check_type_compatibility("string", str)
        assert TypeValidator._check_type_compatibility(42, int)
        assert not TypeValidator._check_type_compatibility("string", int)
        
        # 测试Optional类型
        from typing import Optional
        assert TypeValidator._check_type_compatibility(None, Optional[str])
        assert TypeValidator._check_type_compatibility("value", Optional[str])
        
        # 测试List类型
        assert TypeValidator._check_type_compatibility([1, 2, 3], List[int])
        assert not TypeValidator._check_type_compatibility([1, "2", 3], List[int])
        
        # 测试Dict类型
        assert TypeValidator._check_type_compatibility(
            {"key": "value"}, 
            Dict[str, str]
        )
        assert not TypeValidator._check_type_compatibility(
            {"key": 123}, 
            Dict[str, str]
        )
    
    def test_get_type_schema(self):
        """测试获取类型模式"""
        schema = TypeValidator.get_type_schema(AgentContext)
        
        assert "title" in schema
        assert "properties" in schema
        assert "user_id" in schema["properties"]
        assert "session_id" in schema["properties"]
    
    def test_diagnose_type_error(self):
        """测试类型错误诊断"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id)
        
        # 测试无错误情况
        diagnosis = TypeValidator.diagnose_type_error(context, AgentContext)
        assert diagnosis == "没有检测到类型错误"
        
        # 测试类型不匹配
        diagnosis = TypeValidator.diagnose_type_error("not_a_context", AgentContext)
        assert "期望 AgentContext" in diagnosis
    
    def test_validate_field_constraints(self):
        """测试字段约束验证"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id, max_iterations=10)
        
        # 测试正常情况
        warnings = TypeValidator.validate_field_constraints(context)
        assert len(warnings) == 0
        
        # 测试接近迭代限制
        context.step_count = 9
        warnings = TypeValidator.validate_field_constraints(context)
        assert any("接近最大迭代次数" in w for w in warnings)
        
        # 测试过多性能标签
        context.performance_tags = ["tag"] * 101
        warnings = TypeValidator.validate_field_constraints(context)
        assert any("性能标签过多" in w for w in warnings)


class TestContextTypeGuard:
    """测试上下文类型守卫"""
    
    def test_is_valid_context(self):
        """测试上下文有效性检查"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id)
        
        assert ContextTypeGuard.is_valid_context(context)
        assert not ContextTypeGuard.is_valid_context(None)
        assert not ContextTypeGuard.is_valid_context({"key": "value"})
    
    def test_assert_context_type(self):
        """测试上下文类型断言"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id)
        
        # 正常情况不应抛出异常
        ContextTypeGuard.assert_context_type(context)
        
        # 类型不匹配应抛出异常
        with pytest.raises(TypeError) as exc_info:
            ContextTypeGuard.assert_context_type("not_a_context")
        assert "上下文类型验证失败" in str(exc_info.value)
    
    def test_narrow_type(self):
        """测试类型缩窄"""
        session_id = str(uuid.uuid4())
        custom_data = CustomData(name="test", value=42)
        
        context = create_typed_context(
            user_id="test_user",
            session_id=session_id,
            custom_type=CustomData,
            custom_data=custom_data
        )
        
        # 测试成功的类型缩窄
        narrowed = ContextTypeGuard.narrow_type(context, CustomData)
        assert narrowed is not None
        assert narrowed.custom_data.name == "test"
        
        # 测试失败的类型缩窄
        narrowed = ContextTypeGuard.narrow_type(context, str)
        assert narrowed is None


class TestTypeSafeSerializer:
    """测试类型安全序列化器"""
    
    def test_serialize_context(self):
        """测试上下文序列化"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id)
        
        serialized = TypeSafeSerializer.serialize_context(context)
        
        assert "data" in serialized
        assert "type_info" in serialized
        assert "checksum" in serialized
        
        assert serialized["type_info"]["class"] == "AgentContext"
        assert serialized["type_info"]["version"] == ContextVersion.CURRENT.value
        assert serialized["data"]["user_id"] == "test_user"
    
    def test_deserialize_context(self):
        """测试上下文反序列化"""
        session_id = str(uuid.uuid4())
        original = create_context("test_user", session_id)
        
        # 序列化
        serialized = TypeSafeSerializer.serialize_context(original)
        
        # 反序列化
        restored = TypeSafeSerializer.deserialize_context(serialized)
        
        assert restored.user_id == original.user_id
        assert restored.session_id == original.session_id
        assert restored.version == original.version
    
    def test_serialize_with_custom_data(self):
        """测试带自定义数据的序列化"""
        session_id = str(uuid.uuid4())
        custom_data = CustomData(name="test", value=42, tags=["tag1"])
        
        context = create_typed_context(
            user_id="test_user",
            session_id=session_id,
            custom_type=CustomData,
            custom_data=custom_data
        )
        
        serialized = TypeSafeSerializer.serialize_context(context)
        
        assert "generic_type" in serialized["type_info"]
        assert serialized["type_info"]["generic_type"]["type"] == "CustomData"
        
        # 反序列化
        restored = TypeSafeSerializer.deserialize_context(serialized)
        assert restored.custom_data is not None
    
    def test_checksum_validation(self):
        """测试校验和验证"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id)
        
        serialized = TypeSafeSerializer.serialize_context(context)
        
        # 篡改数据
        serialized["data"]["user_id"] = "hacked_user"
        
        # 反序列化应该失败
        with pytest.raises(ValueError) as exc_info:
            TypeSafeSerializer.deserialize_context(serialized)
        assert "校验和不匹配" in str(exc_info.value)


class TestTypeRegistry:
    """测试类型注册表"""
    
    def test_register_and_get_type(self):
        """测试类型注册和获取"""
        TypeRegistry.register("CustomData", CustomData)
        
        retrieved = TypeRegistry.get("CustomData")
        assert retrieved == CustomData
        
        # 测试未注册的类型
        assert TypeRegistry.get("UnknownType") is None
    
    def test_list_registered(self):
        """测试列出注册的类型"""
        TypeRegistry.register("Type1", str)
        TypeRegistry.register("Type2", int)
        
        registered = TypeRegistry.list_registered()
        assert "Type1" in registered
        assert "Type2" in registered
    
    def test_resolve_type(self):
        """测试类型解析"""
        type_info = {
            "type": "CustomData",
            "module": "test_module",
            "class": "CustomData"
        }
        
        # 注册类型
        TypeRegistry.register("CustomData", CustomData)
        
        resolved = TypeRegistry.resolve_type(type_info)
        assert resolved == CustomData


class TestHelperFunctions:
    """测试辅助函数"""
    
    def test_create_typed_context(self):
        """测试创建类型化上下文"""
        session_id = str(uuid.uuid4())
        custom_data = CustomData(name="test", value=100)
        
        context = create_typed_context(
            user_id="test_user",
            session_id=session_id,
            custom_type=CustomData,
            custom_data=custom_data
        )
        
        assert context.custom_data == custom_data
        
        # 测试类型不匹配
        with pytest.raises(TypeError):
            create_typed_context(
                user_id="test_user",
                session_id=session_id,
                custom_type=str,
                custom_data=custom_data  # CustomData而不是str
            )
    
    def test_cast_context(self):
        """测试上下文类型转换"""
        session_id = str(uuid.uuid4())
        custom_data = CustomData(name="test", value=50)
        
        context = create_context("test_user", session_id)
        context.custom_data = custom_data
        
        # 成功的类型转换
        casted = cast_context(context, CustomData)
        assert casted.custom_data == custom_data
        
        # 失败的类型转换
        with pytest.raises(TypeError):
            cast_context(context, str)
    
    def test_extend_with_data(self):
        """测试使用数据扩展上下文"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id)
        
        custom_data = CustomData(name="extended", value=75)
        extended = context.extend_with_data(custom_data)
        
        assert extended.custom_data == custom_data
        assert extended is context  # 应该是同一个对象
    
    def test_merge_metadata(self):
        """测试合并元数据"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id)
        
        context.metadata = {"key1": "value1"}
        context.merge_metadata({"key2": "value2", "key3": "value3"})
        
        assert context.metadata["key1"] == "value1"
        assert context.metadata["key2"] == "value2"
        assert context.metadata["key3"] == "value3"
    
    def test_get_type_info(self):
        """测试获取类型信息"""
        session_id = str(uuid.uuid4())
        context = create_context("test_user", session_id)
        
        type_info = context.get_type_info()
        
        assert type_info["version"] == ContextVersion.CURRENT.value
        assert type_info["type"] == "AgentContext"
        assert "fields" in type_info
        assert "properties" in type_info["fields"]