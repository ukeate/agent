"""
LangGraph Context API类型定义
基于LangGraph v0.6.5的新Context API实现类型安全的上下文传递
增强版本：支持复杂嵌套类型、泛型和完整的类型验证
"""

from typing import Optional, Any, Dict, TypeVar, Generic, List, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
import uuid

T = TypeVar('T')

class ContextVersion(str, Enum):
    """上下文版本枚举"""
    V1_0 = "1.0"
    V1_1 = "1.1"
    V1_2 = "1.2"
    CURRENT = V1_2

class UserPreferences(BaseModel):
    """用户偏好设置类型"""
    language: str = Field(default="zh-CN", description="用户语言偏好")
    timezone: str = Field(default="Asia/Shanghai", description="时区设置")
    theme: str = Field(default="light", description="主题设置")
    notification_enabled: bool = Field(default=True, description="通知开关")
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="自定义设置")

class SessionContext(BaseModel):
    """会话上下文类型"""
    session_id: str = Field(..., description="会话唯一标识")
    created_at: datetime = Field(default_factory=lambda: utc_now())
    last_active: datetime = Field(default_factory=lambda: utc_now())
    message_count: int = Field(default=0, ge=0)
    interaction_mode: str = Field(default="chat", description="交互模式")
    
    @field_validator('session_id')
    @classmethod
    def validate_session_id(cls, v):
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            # 允许非UUID格式的session_id，但要求长度至少3个字符
            if len(str(v)) < 3:
                raise ValueError('session_id长度至少需要3个字符')
            return v

class WorkflowMetadata(BaseModel):
    """工作流元数据"""
    workflow_id: Optional[str] = Field(None, description="工作流ID")
    workflow_version: str = Field(default="1.0", description="工作流版本")
    parent_workflow_id: Optional[str] = Field(None, description="父工作流ID")
    execution_path: List[str] = Field(default_factory=list, description="执行路径")
    checkpoints: List[Dict[str, Any]] = Field(default_factory=list, description="检查点列表")

class AgentContext(BaseModel, Generic[T]):
    """Agent上下文类型系统"""
    
    # 基础字段
    user_id: str = Field(..., description="用户唯一标识")
    session_id: str = Field(..., description="会话标识")
    
    # 扩展字段
    version: ContextVersion = Field(default=ContextVersion.CURRENT, description="上下文版本")
    conversation_id: Optional[str] = Field(None, description="对话标识")
    agent_id: Optional[str] = Field(None, description="智能体标识")
    workflow_id: Optional[str] = Field(None, description="工作流标识")
    thread_id: Optional[str] = Field(None, description="线程标识")
    
    # 复杂嵌套类型
    user_preferences: UserPreferences = Field(default_factory=UserPreferences)
    session_context: SessionContext = Field(...)
    workflow_metadata: WorkflowMetadata = Field(default_factory=WorkflowMetadata)
    
    # 泛型扩展数据
    custom_data: Optional[T] = Field(None, description="自定义类型数据")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")
    
    # 执行状态
    status: str = Field(default="running", description="执行状态")
    step_count: int = Field(default=0, ge=0, description="步骤计数")
    current_node: Optional[str] = Field(None, description="当前节点")
    last_updated: Optional[datetime] = Field(None, description="最后更新时间")
    
    # 性能和缓存相关
    cache_namespace: Optional[str] = Field(None, description="缓存命名空间")
    performance_tags: List[str] = Field(default_factory=list, description="性能标签")
    enable_checkpoints: bool = Field(default=True, description="启用检查点")
    
    # 配置选项
    max_iterations: int = Field(default=10, ge=1, le=1000, description="最大迭代次数")
    timeout_seconds: int = Field(default=300, ge=1, le=3600, description="超时时间(秒)")
    
    model_config = ConfigDict(
        # 启用类型验证
        validate_assignment=True,
        # 序列化时使用枚举值
        use_enum_values=True,
        # 允许任意类型（为了泛型支持）
        arbitrary_types_allowed=True,
        # 模式示例
        json_schema_extra={
            "example": {
                "user_id": "user_123",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "session_context": {
                    "session_id": "550e8400-e29b-41d4-a716-446655440000",
                    "message_count": 0
                }
            }
        }
    )
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """验证上下文数据的一致性"""
        # 验证session_id一致性
        if self.session_context and hasattr(self.session_context, 'session_id'):
            if self.session_context.session_id != self.session_id:
                raise ValueError('session_id在context和session_context中不匹配')
        
        # 验证状态值
        valid_statuses = ["running", "paused", "completed", "failed", "cancelled"]
        if self.status and self.status not in valid_statuses:
            raise ValueError(f'无效的状态值: {self.status}')
        
        return self
    
    @field_validator('agent_id', 'conversation_id', 'workflow_id', 'thread_id')
    @classmethod
    def validate_optional_ids(cls, v):
        """验证可选的ID字段 - 允许UUID格式或自定义ID格式"""
        if v is not None:
            # 尝试UUID验证，如果失败则允许自定义格式（长度至少3个字符）
            try:
                uuid.UUID(v)
            except ValueError:
                if len(str(v)) < 3:
                    raise ValueError(f'ID长度至少需要3个字符: {v}')
                # 允许自定义格式，只要不是空字符串且长度合理
        return v

    @field_validator('user_id', 'session_id')
    @classmethod
    def validate_required_ids(cls, v):
        """验证必填ID字段不能为空"""
        if v is None or not str(v).strip():
            raise ValueError("user_id/session_id不能为空")
        return v
    
    def update_step(self, node_name: str) -> None:
        """更新步骤信息"""
        self.step_count += 1
        self.current_node = node_name
        self.last_updated = utc_now()
        if self.workflow_metadata:
            self.workflow_metadata.execution_path.append(node_name)
    
    def is_timeout(self) -> bool:
        """检查是否超时"""
        if not self.last_updated:
            return False
        
        elapsed = (utc_now() - self.last_updated).total_seconds()
        return elapsed > self.timeout_seconds
    
    def is_max_iterations_reached(self) -> bool:
        """检查是否达到最大迭代次数"""
        return self.step_count >= self.max_iterations
    
    def add_checkpoint(self, checkpoint_data: Dict[str, Any]) -> None:
        """添加检查点"""
        if self.enable_checkpoints and self.workflow_metadata:
            checkpoint = {
                "timestamp": utc_now().isoformat(),
                "step_count": self.step_count,
                "node": self.current_node,
                "data": checkpoint_data
            }
            self.workflow_metadata.checkpoints.append(checkpoint)
    
    def extend_with_data(self, data: T) -> 'EnhancedAgentContext[T]':
        """使用自定义数据扩展上下文"""
        self.custom_data = data
        return self
    
    def merge_metadata(self, new_metadata: Dict[str, Any]) -> None:
        """合并新的元数据"""
        self.metadata.update(new_metadata)
    
    def get_type_info(self) -> Dict[str, Any]:
        """获取类型信息"""
        return {
            "version": self.version.value,
            "type": self.__class__.__name__,
            "module": self.__class__.__module__,
            "fields": self.model_json_schema()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return self.model_dump()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentContext':
        """从字典创建AgentContext实例"""
        # 确保session_context存在
        session_id = data.get('session_id', '550e8400-e29b-41d4-a716-446655440000')
        session_context = SessionContext(session_id=session_id)
        
        context = cls(
            user_id=data.get('user_id', 'unknown'),
            session_id=session_id,
            conversation_id=data.get('conversation_id'),
            agent_id=data.get('agent_id'),
            workflow_id=data.get('workflow_id'),
            thread_id=data.get('thread_id'),
            session_context=session_context,
            max_iterations=data.get('max_iterations', 10),
            timeout_seconds=data.get('timeout_seconds', 300),
            cache_namespace=data.get('cache_namespace'),
            enable_checkpoints=data.get('enable_checkpoints', True),
            status=data.get('status', 'running'),
            step_count=data.get('step_count', 0)
        )
        
        # 设置其他字段
        if 'current_node' in data:
            context.current_node = data['current_node']
        if 'metadata' in data:
            context.metadata.update(data['metadata'])
            
        return context

def create_context(
    user_id: str,
    session_id: str,
    **kwargs
) -> AgentContext:
    """创建上下文实例"""
    # 确保session_context存在
    if 'session_context' not in kwargs:
        kwargs['session_context'] = SessionContext(session_id=session_id)
    
    return AgentContext(
        user_id=user_id,
        session_id=session_id,
        **kwargs
    )

def create_default_context(
    user_id: str = "default_user",
    session_id: str = "550e8400-e29b-41d4-a716-446655440000",  # 使用有效的UUID
    **kwargs
) -> AgentContext:
    """创建默认上下文"""
    return create_context(user_id, session_id, **kwargs)

def validate_context(context: AgentContext) -> bool:
    """验证上下文是否有效"""
    try:
        context.model_dump()
        return True
    except Exception:
        return False

# LangGraph 0.6.5 新Context API兼容的dataclass结构
@dataclass
class LangGraphContextSchema:
    """LangGraph 0.6.5新Context API的dataclass结构"""
    user_id: str = "unknown"
    session_id: str = "550e8400-e29b-41d4-a716-446655440000"
    conversation_id: Optional[str] = None
    agent_id: Optional[str] = None
    workflow_id: Optional[str] = None
    thread_id: Optional[str] = None
    
    # 执行控制参数
    max_iterations: int = 10
    timeout_seconds: int = 300
    
    # 缓存和性能相关
    cache_namespace: Optional[str] = None
    enable_checkpoints: bool = True
    
    def to_agent_context(self) -> AgentContext:
        """转换为完整的AgentContext对象"""
        session_context = SessionContext(session_id=self.session_id)
        return AgentContext(
            user_id=self.user_id,
            session_id=self.session_id,
            conversation_id=self.conversation_id,
            agent_id=self.agent_id,
            thread_id=self.thread_id,
            session_context=session_context,
            max_iterations=self.max_iterations,
            timeout_seconds=self.timeout_seconds,
            cache_namespace=self.cache_namespace,
            enable_checkpoints=self.enable_checkpoints
        )
    
    @classmethod
    def from_agent_context(cls, context: AgentContext) -> 'LangGraphContextSchema':
        """从AgentContext创建dataclass实例"""
        return cls(
            user_id=context.user_id,
            session_id=context.session_id,
            conversation_id=context.conversation_id,
            agent_id=context.agent_id,
            workflow_id=context.workflow_id,
            thread_id=context.thread_id,
            max_iterations=context.max_iterations,
            timeout_seconds=context.timeout_seconds,
            cache_namespace=context.cache_namespace,
            enable_checkpoints=context.enable_checkpoints
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "conversation_id": self.conversation_id,
            "agent_id": self.agent_id,
            "workflow_id": self.workflow_id,
            "thread_id": self.thread_id,
            "max_iterations": self.max_iterations,
            "timeout_seconds": self.timeout_seconds,
            "cache_namespace": self.cache_namespace,
            "enable_checkpoints": self.enable_checkpoints
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LangGraphContextSchema':
        """从字典创建dataclass实例"""
        return cls(
            user_id=data.get('user_id', 'unknown'),
            session_id=data.get('session_id', '550e8400-e29b-41d4-a716-446655440000'),
            conversation_id=data.get('conversation_id'),
            agent_id=data.get('agent_id'),
            workflow_id=data.get('workflow_id'),
            thread_id=data.get('thread_id'),
            max_iterations=data.get('max_iterations', 10),
            timeout_seconds=data.get('timeout_seconds', 300),
            cache_namespace=data.get('cache_namespace'),
            enable_checkpoints=data.get('enable_checkpoints', True)
        )
