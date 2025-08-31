"""
LangGraph状态管理
统一的工作流状态管理，支持检查点和状态持久化
"""
from typing import Any, Dict, List, Optional, TypedDict, Annotated
from typing_extensions import TypedDict as TypedDictExt
from dataclasses import dataclass, field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
import uuid
import json


class MessagesState(TypedDict):
    """LangGraph统一状态结构 - 符合LangGraph标准的消息状态定义"""
    messages: Annotated[List[Dict[str, Any]], "工作流消息列表"] 
    metadata: Annotated[Dict[str, Any], "状态元数据信息"]
    context: Annotated[Dict[str, Any], "工作流执行上下文"]
    workflow_id: Annotated[str, "工作流执行ID"]


def create_initial_state(workflow_id: Optional[str] = None) -> MessagesState:
    """创建初始工作流状态"""
    return MessagesState(
        messages=[],
        metadata={
            "created_at": utc_now().isoformat(),
            "step_count": 0,
            "status": "pending",
            "version": "1.0"
        },
        context={},
        workflow_id=workflow_id or str(uuid.uuid4())
    )


def validate_state(state: MessagesState) -> bool:
    """验证状态结构完整性"""
    required_keys = {"messages", "metadata", "context", "workflow_id"}
    
    # 检查必要字段是否存在
    if not all(key in state for key in required_keys):
        return False
    
    # 检查字段类型
    if not isinstance(state.get("messages"), list):
        return False
    if not isinstance(state.get("metadata"), dict):
        return False
    if not isinstance(state.get("context"), dict):
        return False
    if not isinstance(state.get("workflow_id"), str):
        return False
    
    return True


def serialize_state(state: MessagesState) -> str:
    """序列化状态为JSON字符串"""
    return json.dumps(state, default=str, ensure_ascii=False)


def deserialize_state(state_json: str) -> MessagesState:
    """从JSON字符串反序列化状态"""
    data = json.loads(state_json)
    return MessagesState(
        messages=data.get("messages", []),
        metadata=data.get("metadata", {}),
        context=data.get("context", {}),
        workflow_id=data.get("workflow_id", str(uuid.uuid4()))
    )

