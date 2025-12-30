"""
LangGraph 0.6.5 Pre/Post Model Hooks Implementation
实现模型调用前后的钩子函数，用于请求预处理、响应后处理和guardrails
"""

from typing import Any, Dict, List, Optional, Callable, Union, Literal
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
import asyncio
import json
import re
from .state import MessagesState
from .context import AgentContext, LangGraphContextSchema

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class HookConfig:
    """钩子配置"""
    enabled: bool = True
    priority: int = 0  # 优先级，数字越小优先级越高
    name: str = ""
    description: str = ""

class BaseHook(ABC):
    """钩子基类"""
    
    def __init__(self, config: Optional[HookConfig] = None):
        self.config = config or HookConfig()
    
    @abstractmethod
    async def execute(self, state: MessagesState, context: Optional[AgentContext] = None) -> MessagesState:
        """执行钩子逻辑"""
        ...

class PreModelHook(BaseHook):
    """模型调用前钩子基类"""
    ...

class PostModelHook(BaseHook):
    """模型调用后钩子基类"""
    ...

# 预处理钩子实现

class MessageCompressionHook(PreModelHook):
    """消息压缩钩子 - 压缩历史消息以节省token"""
    
    def __init__(self, config: Optional[HookConfig] = None, max_messages: int = 10, compression_ratio: float = 0.5):
        super().__init__(config)
        self.max_messages = max_messages
        self.compression_ratio = compression_ratio
    
    async def execute(self, state: MessagesState, context: Optional[AgentContext] = None) -> MessagesState:
        """压缩历史消息"""
        if not self.config.enabled:
            return state
        
        messages = state.get("messages", [])
        
        # 如果消息数量超过限制，进行压缩
        if len(messages) > self.max_messages:
            # 保留最新的消息
            recent_messages = messages[-self.max_messages:]
            
            # 压缩旧消息
            old_messages = messages[:-self.max_messages]
            compressed_content = self._compress_messages(old_messages)
            
            # 创建压缩消息
            compressed_message = {
                "role": "system",
                "content": f"[历史消息摘要] {compressed_content}",
                "timestamp": utc_now().isoformat(),
                "metadata": {
                    "type": "compressed_history",
                    "original_count": len(old_messages),
                    "compression_hook": self.config.name or "MessageCompressionHook"
                }
            }
            
            # 更新状态
            state["messages"] = [compressed_message] + recent_messages
            
            # 记录压缩操作
            if "hook_logs" not in state["context"]:
                state["context"]["hook_logs"] = []
            
            state["context"]["hook_logs"].append({
                "hook": "MessageCompressionHook",
                "timestamp": utc_now().isoformat(),
                "action": "compressed_messages",
                "details": f"压缩了 {len(old_messages)} 条消息"
            })
        
        return state
    
    def _compress_messages(self, messages: List[Dict]) -> str:
        """压缩消息内容"""
        if not messages:
            return ""
        
        # 简单的压缩策略：提取关键信息
        summary_parts = []
        for msg in messages[-3:]:  # 只取最后3条进行摘要
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            # 截断长内容
            if len(content) > 100:
                content = content[:100] + "..."
            
            summary_parts.append(f"{role}: {content}")
        
        return " | ".join(summary_parts)

class InputSanitizationHook(PreModelHook):
    """输入清理钩子 - 清理和验证用户输入"""
    
    def __init__(self, config: Optional[HookConfig] = None, max_message_length: int = 2000):
        super().__init__(config)
        self.max_message_length = max_message_length
        self.forbidden_patterns = [
            r"<script.*?>.*?</script>",  # XSS脚本
            r"javascript:",              # JavaScript协议
            r"data:.*base64",           # Base64数据URL
        ]
    
    async def execute(self, state: MessagesState, context: Optional[AgentContext] = None) -> MessagesState:
        """清理输入内容"""
        if not self.config.enabled:
            return state
        
        messages = state.get("messages", [])
        cleaned_messages = []
        
        for message in messages:
            cleaned_message = message.copy()
            content = message.get("content", "")
            
            # 长度限制
            if len(content) > self.max_message_length:
                content = content[:self.max_message_length] + "[内容被截断]"
                cleaned_message["metadata"] = cleaned_message.get("metadata", {})
                cleaned_message["metadata"]["truncated"] = True
            
            # 移除危险模式
            for pattern in self.forbidden_patterns:
                content = re.sub(pattern, "[已过滤]", content, flags=re.IGNORECASE)
            
            cleaned_message["content"] = content
            cleaned_messages.append(cleaned_message)
        
        state["messages"] = cleaned_messages
        return state

class ContextEnrichmentHook(PreModelHook):
    """上下文增强钩子 - 添加系统上下文信息"""
    
    async def execute(self, state: MessagesState, context: Optional[AgentContext] = None) -> MessagesState:
        """增强上下文信息"""
        if not self.config.enabled:
            return state
        
        # 添加系统提示消息
        if context:
            system_context = {
                "role": "system",
                "content": f"当前用户: {context.user_id}, 会话: {context.session_id}, 时间: {utc_now().strftime('%Y-%m-%d %H:%M:%S')}",
                "timestamp": utc_now().isoformat(),
                "metadata": {
                    "type": "system_context",
                    "hook": "ContextEnrichmentHook"
                }
            }
            
            # 插入到消息开头（如果还没有系统消息）
            messages = state.get("messages", [])
            if not messages or messages[0].get("role") != "system":
                state["messages"] = [system_context] + messages
        
        return state

# 后处理钩子实现

class ResponseFilterHook(PostModelHook):
    """响应过滤钩子 - 过滤不当内容"""
    
    def __init__(self, config: Optional[HookConfig] = None):
        super().__init__(config)
        self.blocked_words = [
            "禁词1", "禁词2"  # 可配置的敏感词列表
        ]
        self.replacement_text = "[内容已被过滤]"
    
    async def execute(self, state: MessagesState, context: Optional[AgentContext] = None) -> MessagesState:
        """过滤响应内容"""
        if not self.config.enabled:
            return state
        
        messages = state.get("messages", [])
        filtered_messages = []
        
        for message in messages:
            filtered_message = message.copy()
            content = message.get("content", "")
            
            # 检查并替换敏感词
            original_content = content
            for word in self.blocked_words:
                content = content.replace(word, self.replacement_text)
            
            # 如果内容被修改，记录日志
            if content != original_content:
                filtered_message["metadata"] = filtered_message.get("metadata", {})
                filtered_message["metadata"]["filtered"] = True
                filtered_message["metadata"]["filter_hook"] = "ResponseFilterHook"
            
            filtered_message["content"] = content
            filtered_messages.append(filtered_message)
        
        state["messages"] = filtered_messages
        return state

class QualityCheckHook(PostModelHook):
    """质量检查钩子 - 验证AI输出质量"""
    
    def __init__(self, config: Optional[HookConfig] = None, min_length: int = 10, max_length: int = 5000):
        super().__init__(config)
        self.min_length = min_length
        self.max_length = max_length
    
    async def execute(self, state: MessagesState, context: Optional[AgentContext] = None) -> MessagesState:
        """检查输出质量"""
        if not self.config.enabled:
            return state
        
        messages = state.get("messages", [])
        quality_logs = []
        
        for i, message in enumerate(messages):
            if message.get("role") == "assistant":
                content = message.get("content", "")
                quality_issues = []
                
                # 长度检查
                if len(content) < self.min_length:
                    quality_issues.append("内容过短")
                elif len(content) > self.max_length:
                    quality_issues.append("内容过长")
                
                # 基本完整性检查
                if not content.strip():
                    quality_issues.append("内容为空")
                
                # 如果有质量问题，记录并可能修复
                if quality_issues:
                    quality_logs.append({
                        "message_index": i,
                        "issues": quality_issues,
                        "content_length": len(content)
                    })
                    
                    # 为消息添加质量标记
                    message["metadata"] = message.get("metadata", {})
                    message["metadata"]["quality_issues"] = quality_issues
                    message["metadata"]["quality_check_hook"] = "QualityCheckHook"
        
        # 记录质量检查结果
        if quality_logs:
            if "hook_logs" not in state["context"]:
                state["context"]["hook_logs"] = []
            
            state["context"]["hook_logs"].append({
                "hook": "QualityCheckHook",
                "timestamp": utc_now().isoformat(),
                "quality_issues_found": len(quality_logs),
                "details": quality_logs
            })
        
        return state

class ResponseEnhancementHook(PostModelHook):
    """响应增强钩子 - 增强AI输出"""
    
    async def execute(self, state: MessagesState, context: Optional[AgentContext] = None) -> MessagesState:
        """增强响应内容"""
        if not self.config.enabled:
            return state
        
        messages = state.get("messages", [])
        
        for message in messages:
            if message.get("role") == "assistant":
                # 添加时间戳和元数据
                message["timestamp"] = message.get("timestamp", utc_now().isoformat())
                message["metadata"] = message.get("metadata", {})
                message["metadata"]["enhanced"] = True
                message["metadata"]["enhancement_hook"] = "ResponseEnhancementHook"
                
                # 可以在这里添加更多增强逻辑，如格式化、添加引用等
        
        return state

# 钩子管理器

class HookManager:
    """钩子管理器"""
    
    def __init__(self):
        self.pre_hooks: List[PreModelHook] = []
        self.post_hooks: List[PostModelHook] = []
    
    def add_pre_hook(self, hook: PreModelHook) -> None:
        """添加预处理钩子"""
        self.pre_hooks.append(hook)
        # 按优先级排序
        self.pre_hooks.sort(key=lambda h: h.config.priority)
    
    def add_post_hook(self, hook: PostModelHook) -> None:
        """添加后处理钩子"""
        self.post_hooks.append(hook)
        # 按优先级排序
        self.post_hooks.sort(key=lambda h: h.config.priority)
    
    def remove_hook(self, hook_name: str, hook_type: Literal["pre", "post"] = "pre") -> bool:
        """移除钩子"""
        if hook_type == "pre":
            hooks = self.pre_hooks
        else:
            hooks = self.post_hooks
        
        for i, hook in enumerate(hooks):
            if hook.config.name == hook_name:
                hooks.pop(i)
                return True
        return False
    
    async def execute_pre_hooks(self, state: MessagesState, context: Optional[AgentContext] = None) -> MessagesState:
        """执行所有预处理钩子"""
        for hook in self.pre_hooks:
            if hook.config.enabled:
                try:
                    state = await hook.execute(state, context)
                except Exception as e:
                    logger.error(
                        "预处理钩子执行失败",
                        hook_name=hook.config.name,
                        error=str(e),
                        exc_info=True,
                    )
        return state
    
    async def execute_post_hooks(self, state: MessagesState, context: Optional[AgentContext] = None) -> MessagesState:
        """执行所有后处理钩子"""
        for hook in self.post_hooks:
            if hook.config.enabled:
                try:
                    state = await hook.execute(state, context)
                except Exception as e:
                    logger.error(
                        "后处理钩子执行失败",
                        hook_name=hook.config.name,
                        error=str(e),
                        exc_info=True,
                    )
        return state
    
    def get_hook_status(self) -> Dict[str, Any]:
        """获取钩子状态信息"""
        return {
            "pre_hooks": [
                {
                    "name": hook.config.name,
                    "enabled": hook.config.enabled,
                    "priority": hook.config.priority,
                    "description": hook.config.description
                }
                for hook in self.pre_hooks
            ],
            "post_hooks": [
                {
                    "name": hook.config.name,
                    "enabled": hook.config.enabled,
                    "priority": hook.config.priority,
                    "description": hook.config.description
                }
                for hook in self.post_hooks
            ]
        }

# 全局钩子管理器实例
_hook_manager: Optional[HookManager] = None

def get_hook_manager() -> HookManager:
    """获取全局钩子管理器"""
    global _hook_manager
    if _hook_manager is None:
        _hook_manager = HookManager()
        
        # 添加默认钩子
        _hook_manager.add_pre_hook(MessageCompressionHook(
            HookConfig(name="MessageCompression", description="压缩历史消息", priority=1)
        ))
        _hook_manager.add_pre_hook(InputSanitizationHook(
            HookConfig(name="InputSanitization", description="输入内容清理", priority=2)
        ))
        _hook_manager.add_pre_hook(ContextEnrichmentHook(
            HookConfig(name="ContextEnrichment", description="上下文增强", priority=3)
        ))
        
        _hook_manager.add_post_hook(ResponseFilterHook(
            HookConfig(name="ResponseFilter", description="响应内容过滤", priority=1)
        ))
        _hook_manager.add_post_hook(QualityCheckHook(
            HookConfig(name="QualityCheck", description="输出质量检查", priority=2)
        ))
        _hook_manager.add_post_hook(ResponseEnhancementHook(
            HookConfig(name="ResponseEnhancement", description="响应内容增强", priority=3)
        ))
    
    return _hook_manager

def set_hook_manager(manager: HookManager) -> None:
    """设置全局钩子管理器"""
    global _hook_manager
    _hook_manager = manager

# 钩子装饰器工具

def with_hooks(pre_hooks: Optional[List[str]] = None, post_hooks: Optional[List[str]] = None):
    """为函数添加钩子的装饰器"""
    def decorator(func):
        async def wrapper(state: MessagesState, *args, **kwargs):
            hook_manager = get_hook_manager()
            context = kwargs.get('context')
            
            # 执行预处理钩子
            state = await hook_manager.execute_pre_hooks(state, context)
            
            # 执行原始函数
            if asyncio.iscoroutinefunction(func):
                result = await func(state, *args, **kwargs)
            else:
                result = func(state, *args, **kwargs)
            
            # 执行后处理钩子
            result = await hook_manager.execute_post_hooks(result, context)
            
            return result
        return wrapper
    return decorator
