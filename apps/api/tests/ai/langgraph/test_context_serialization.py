"""
上下文序列化测试
测试类型安全的序列化和反序列化功能
"""
import pytest
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

from src.ai.langgraph.context import (
    AgentContext,
    UserPreferences,
    SessionContext,
    ContextVersion,
    create_context
)
from src.ai.langgraph.type_utils import (
    TypeSafeSerializer,
    TypeSafeCachedNode,
    create_typed_context
)


class ComplexCustomData(BaseModel):
    """复杂的自定义数据类型用于测试"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    values: List[int]
    metadata: Dict[str, Any]
    nested: Optional['NestedData'] = None


class NestedData(BaseModel):
    """嵌套数据类型"""
    level: int
    description: str
    tags: List[str] = Field(default_factory=list)


# 更新forward references
ComplexCustomData.model_rebuild()


class TestBasicSerialization:
    """基础序列化测试"""
    
    def test_serialize_simple_context(self):
        """测试简单上下文序列化"""
        session_id = str(uuid.uuid4())
        context = create_context("user123", session_id)
        
        serialized = TypeSafeSerializer.serialize_context(context)
        
        # 验证序列化结构
        assert isinstance(serialized, dict)
        assert "data" in serialized
        assert "type_info" in serialized
        assert "checksum" in serialized
        
        # 验证数据内容
        data = serialized["data"]
        assert data["user_id"] == "user123"
        assert data["session_id"] == session_id
        assert data["version"] == ContextVersion.CURRENT.value
    
    def test_deserialize_simple_context(self):
        """测试简单上下文反序列化"""
        session_id = str(uuid.uuid4())
        original = create_context("user456", session_id)
        
        # 序列化然后反序列化
        serialized = TypeSafeSerializer.serialize_context(original)
        restored = TypeSafeSerializer.deserialize_context(serialized)
        
        # 验证恢复的数据
        assert restored.user_id == original.user_id
        assert restored.session_id == original.session_id
        assert restored.version == original.version
        assert restored.status == original.status
    
    def test_serialize_with_preferences(self):
        """测试带用户偏好的序列化"""
        session_id = str(uuid.uuid4())
        context = create_context("user789", session_id)
        context.user_preferences.language = "en-US"
        context.user_preferences.theme = "dark"
        context.user_preferences.custom_settings = {"font_size": 14}
        
        serialized = TypeSafeSerializer.serialize_context(context)
        restored = TypeSafeSerializer.deserialize_context(serialized)
        
        assert restored.user_preferences.language == "en-US"
        assert restored.user_preferences.theme == "dark"
        assert restored.user_preferences.custom_settings["font_size"] == 14
    
    def test_serialize_with_metadata(self):
        """测试带元数据的序列化"""
        session_id = str(uuid.uuid4())
        context = create_context("user_meta", session_id)
        context.metadata = {
            "request_id": "req_123",
            "client_version": "1.0.0",
            "features": ["feature1", "feature2"],
            "config": {"timeout": 30, "retry": 3}
        }
        
        serialized = TypeSafeSerializer.serialize_context(context)
        restored = TypeSafeSerializer.deserialize_context(serialized)
        
        assert restored.metadata["request_id"] == "req_123"
        assert restored.metadata["client_version"] == "1.0.0"
        assert len(restored.metadata["features"]) == 2
        assert restored.metadata["config"]["timeout"] == 30


class TestComplexSerialization:
    """复杂序列化场景测试"""
    
    def test_serialize_with_complex_custom_data(self):
        """测试复杂自定义数据的序列化"""
        session_id = str(uuid.uuid4())
        
        # 创建复杂的自定义数据
        nested = NestedData(
            level=2,
            description="Nested data for testing",
            tags=["test", "nested"]
        )
        custom_data = ComplexCustomData(
            name="Complex Test",
            values=[1, 2, 3, 4, 5],
            metadata={"key": "value", "count": 100},
            nested=nested
        )
        
        context = create_typed_context(
            user_id="complex_user",
            session_id=session_id,
            custom_type=ComplexCustomData,
            custom_data=custom_data
        )
        
        # 序列化
        serialized = TypeSafeSerializer.serialize_context(context)
        
        # 验证类型信息
        assert "generic_type" in serialized["type_info"]
        generic_info = serialized["type_info"]["generic_type"]
        assert generic_info["type"] == "ComplexCustomData"
        
        # 反序列化
        restored = TypeSafeSerializer.deserialize_context(serialized)
        
        # 验证恢复的自定义数据
        assert restored.custom_data is not None
        if isinstance(restored.custom_data, ComplexCustomData):
            # 正确重构为Pydantic模型
            assert restored.custom_data.name == "Complex Test"
            assert len(restored.custom_data.values) == 5
            assert restored.custom_data.nested.level == 2
        else:
            # 字典格式
            assert restored.custom_data["name"] == "Complex Test"
            assert len(restored.custom_data["values"]) == 5
            assert restored.custom_data["nested"]["level"] == 2
    
    def test_serialize_with_workflow_data(self):
        """测试带工作流数据的序列化"""
        session_id = str(uuid.uuid4())
        context = create_context("workflow_user", session_id)
        
        # 添加工作流数据
        context.workflow_metadata.workflow_id = "wf_123"
        context.workflow_metadata.workflow_version = "2.0"
        context.workflow_metadata.execution_path = ["start", "process", "validate"]
        context.workflow_metadata.checkpoints = [
            {"step": 1, "data": {"result": "success"}},
            {"step": 2, "data": {"result": "pending"}}
        ]
        
        serialized = TypeSafeSerializer.serialize_context(context)
        restored = TypeSafeSerializer.deserialize_context(serialized)
        
        assert restored.workflow_metadata.workflow_id == "wf_123"
        assert restored.workflow_metadata.workflow_version == "2.0"
        assert len(restored.workflow_metadata.execution_path) == 3
        assert len(restored.workflow_metadata.checkpoints) == 2
    
    def test_serialize_with_performance_tags(self):
        """测试带性能标签的序列化"""
        session_id = str(uuid.uuid4())
        context = create_context("perf_user", session_id)
        
        context.cache_namespace = "cache_ns_1"
        context.performance_tags = ["fast", "cached", "optimized"]
        
        serialized = TypeSafeSerializer.serialize_context(context)
        restored = TypeSafeSerializer.deserialize_context(serialized)
        
        assert restored.cache_namespace == "cache_ns_1"
        assert len(restored.performance_tags) == 3
        assert "cached" in restored.performance_tags


class TestSerializationEdgeCases:
    """序列化边界情况测试"""
    
    def test_serialize_empty_context(self):
        """测试空上下文序列化"""
        session_id = str(uuid.uuid4())
        context = create_context("empty_user", session_id)
        
        serialized = TypeSafeSerializer.serialize_context(context)
        restored = TypeSafeSerializer.deserialize_context(serialized)
        
        assert restored.user_id == "empty_user"
        assert restored.metadata == {}
        assert restored.performance_tags == []
        assert restored.custom_data is None
    
    def test_serialize_with_none_values(self):
        """测试包含None值的序列化"""
        session_id = str(uuid.uuid4())
        context = create_context("none_user", session_id)
        
        context.conversation_id = None
        context.agent_id = None
        context.thread_id = None
        context.current_node = None
        context.last_updated = None
        
        serialized = TypeSafeSerializer.serialize_context(context)
        restored = TypeSafeSerializer.deserialize_context(serialized)
        
        assert restored.conversation_id is None
        assert restored.agent_id is None
        assert restored.thread_id is None
        assert restored.current_node is None
        assert restored.last_updated is None
    
    def test_serialize_with_special_characters(self):
        """测试特殊字符的序列化"""
        session_id = str(uuid.uuid4())
        context = create_context("special_user", session_id)
        
        context.metadata = {
            "unicode": "你好世界 🌍",
            "escape": "line1\nline2\ttab",
            "quotes": 'He said "Hello"',
            "path": "/usr/local/bin\\test"
        }
        
        serialized = TypeSafeSerializer.serialize_context(context)
        restored = TypeSafeSerializer.deserialize_context(serialized)
        
        assert restored.metadata["unicode"] == "你好世界 🌍"
        assert restored.metadata["escape"] == "line1\nline2\ttab"
        assert restored.metadata["quotes"] == 'He said "Hello"'
        assert restored.metadata["path"] == "/usr/local/bin\\test"
    
    def test_serialize_large_context(self):
        """测试大型上下文的序列化"""
        session_id = str(uuid.uuid4())
        context = create_context("large_user", session_id)
        
        # 添加大量数据
        context.metadata = {
            f"key_{i}": f"value_{i}" for i in range(1000)
        }
        context.performance_tags = [f"tag_{i}" for i in range(100)]
        
        serialized = TypeSafeSerializer.serialize_context(context)
        restored = TypeSafeSerializer.deserialize_context(serialized)
        
        assert len(restored.metadata) == 1000
        assert len(restored.performance_tags) == 100
        assert restored.metadata["key_500"] == "value_500"


class TestSerializationValidation:
    """序列化验证测试"""
    
    def test_checksum_integrity(self):
        """测试校验和完整性"""
        session_id = str(uuid.uuid4())
        context = create_context("checksum_user", session_id)
        
        serialized = TypeSafeSerializer.serialize_context(context)
        original_checksum = serialized["checksum"]
        
        # 修改数据
        serialized["data"]["user_id"] = "modified_user"
        
        # 验证校验和不匹配
        with pytest.raises(ValueError) as exc_info:
            TypeSafeSerializer.deserialize_context(serialized)
        assert "校验和不匹配" in str(exc_info.value)
        
        # 恢复原始数据
        serialized["data"]["user_id"] = "checksum_user"
        serialized["checksum"] = original_checksum
        
        # 应该能正常反序列化
        restored = TypeSafeSerializer.deserialize_context(serialized)
        assert restored.user_id == "checksum_user"
    
    def test_invalid_serialized_format(self):
        """测试无效的序列化格式"""
        # 缺少必要字段
        with pytest.raises(ValueError) as exc_info:
            TypeSafeSerializer.deserialize_context({"data": {}})
        assert "序列化数据格式无效" in str(exc_info.value)
        
        with pytest.raises(ValueError) as exc_info:
            TypeSafeSerializer.deserialize_context({"type_info": {}})
        assert "序列化数据格式无效" in str(exc_info.value)
    
    def test_type_info_preservation(self):
        """测试类型信息保留"""
        session_id = str(uuid.uuid4())
        context = create_context("type_user", session_id)
        
        serialized = TypeSafeSerializer.serialize_context(context)
        
        # 验证类型信息
        type_info = serialized["type_info"]
        assert type_info["class"] == "AgentContext"
        assert type_info["module"] == "src.ai.langgraph.context"
        assert type_info["version"] == ContextVersion.CURRENT.value
        assert "timestamp" in type_info


class TestJsonCompatibility:
    """JSON兼容性测试"""
    
    def test_json_round_trip(self):
        """测试JSON往返转换"""
        session_id = str(uuid.uuid4())
        context = create_context("json_user", session_id)
        context.metadata = {"test": "data"}
        
        # 序列化为JSON字符串
        serialized = TypeSafeSerializer.serialize_context(context)
        json_str = json.dumps(serialized, default=str)
        
        # 从JSON字符串恢复
        loaded = json.loads(json_str)
        restored = TypeSafeSerializer.deserialize_context(loaded)
        
        assert restored.user_id == "json_user"
        assert restored.metadata["test"] == "data"
    
    def test_datetime_serialization(self):
        """测试datetime序列化"""
        session_id = str(uuid.uuid4())
        context = create_context("datetime_user", session_id)
        
        # 更新时间戳
        context.update_step("test_node")
        original_time = context.last_updated
        
        serialized = TypeSafeSerializer.serialize_context(context)
        restored = TypeSafeSerializer.deserialize_context(serialized)
        
        # 验证时间恢复（可能是字符串格式）
        assert restored.last_updated is not None
        if isinstance(restored.last_updated, str):
            # 尝试解析
            parsed = datetime.fromisoformat(restored.last_updated)
            assert parsed is not None
    
    def test_uuid_serialization(self):
        """测试UUID序列化"""
        session_id = str(uuid.uuid4())
        agent_id = str(uuid.uuid4())
        context = create_context("uuid_user", session_id)
        context.agent_id = agent_id
        
        serialized = TypeSafeSerializer.serialize_context(context)
        json_str = json.dumps(serialized, default=str)
        loaded = json.loads(json_str)
        restored = TypeSafeSerializer.deserialize_context(loaded)
        
        assert restored.session_id == session_id
        assert restored.agent_id == agent_id


class TestExtractTypeInfo:
    """测试类型信息提取"""
    
    def test_extract_basic_types(self):
        """测试基础类型信息提取"""
        # 字符串
        info = TypeSafeSerializer._extract_type_info("test")
        assert info["type"] == "str"
        assert info["value"] == "test"
        
        # 整数
        info = TypeSafeSerializer._extract_type_info(42)
        assert info["type"] == "int"
        assert info["value"] == 42
        
        # 布尔值
        info = TypeSafeSerializer._extract_type_info(True)
        assert info["type"] == "bool"
        assert info["value"] is True
        
        # None
        info = TypeSafeSerializer._extract_type_info(None)
        assert info["type"] == "None"
    
    def test_extract_collection_types(self):
        """测试集合类型信息提取"""
        # 列表
        info = TypeSafeSerializer._extract_type_info([1, 2, 3])
        assert info["type"] == "list"
        assert "items" in info
        assert len(info["items"]) == 3
        
        # 字典
        info = TypeSafeSerializer._extract_type_info({"key": "value", "count": 10})
        assert info["type"] == "dict"
        assert "sample" in info
        assert len(info["sample"]) == 2
    
    def test_extract_pydantic_model(self):
        """测试Pydantic模型信息提取"""
        preferences = UserPreferences(language="fr-FR", theme="auto")
        info = TypeSafeSerializer._extract_type_info(preferences)
        
        assert info["type"] == "UserPreferences"
        assert "schema" in info
        assert "data" in info
        assert info["data"]["language"] == "fr-FR"