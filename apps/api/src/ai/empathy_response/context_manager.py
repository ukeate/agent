"""
对话上下文管理器

管理多轮对话的情感上下文和共情响应历史
"""
import logging
from typing import Dict, Optional, List, Any
from datetime import datetime, timedelta
from collections import defaultdict
import threading

from .models import DialogueContext, EmpathyResponse, CulturalContext
from ..emotion_modeling.models import EmotionState, PersonalityProfile

logger = logging.getLogger(__name__)


class ContextManager:
    """对话上下文管理器"""
    
    def __init__(self):
        """初始化上下文管理器"""
        # 内存中的上下文存储
        self._contexts: Dict[str, DialogueContext] = {}
        
        # 用户上下文映射 (user_id -> context_id)
        self._user_contexts: Dict[str, str] = {}
        
        # 上下文访问锁
        self._lock = threading.RLock()
        
        # 配置参数
        self.config = {
            "max_contexts": 1000,           # 最大上下文数量
            "context_ttl_hours": 24,        # 上下文生存时间（小时）
            "max_emotion_history": 50,      # 最大情感历史记录数
            "max_response_history": 30,     # 最大回应历史记录数
            "cleanup_interval_minutes": 60  # 清理间隔（分钟）
        }
        
        # 统计信息
        self.stats = {
            "total_contexts": 0,
            "active_contexts": 0,
            "contexts_created": 0,
            "contexts_updated": 0,
            "contexts_cleaned": 0
        }
        
        # 定期清理任务
        self._last_cleanup = datetime.now()
    
    def create_context(
        self,
        user_id: str,
        conversation_id: str,
        session_id: Optional[str] = None,
        cultural_context: Optional[CulturalContext] = None
    ) -> DialogueContext:
        """
        创建新的对话上下文
        
        Args:
            user_id: 用户ID
            conversation_id: 对话ID
            session_id: 会话ID
            cultural_context: 文化背景
            
        Returns:
            DialogueContext: 新创建的对话上下文
        """
        with self._lock:
            # 检查是否需要清理
            self._maybe_cleanup()
            
            # 如果上下文数量超限，清理旧的上下文
            if len(self._contexts) >= self.config["max_contexts"]:
                self._force_cleanup()
            
            # 创建新上下文
            context = DialogueContext(
                user_id=user_id,
                conversation_id=conversation_id,
                session_id=session_id,
                cultural_context=cultural_context,
                start_time=datetime.now(),
                last_update=datetime.now()
            )
            
            # 存储上下文
            self._contexts[conversation_id] = context
            self._user_contexts[user_id] = conversation_id
            
            # 更新统计
            self.stats["total_contexts"] += 1
            self.stats["active_contexts"] += 1
            self.stats["contexts_created"] += 1
            
            logger.info(f"Created new context for user {user_id}, conversation {conversation_id}")
            
            return context
    
    def get_context(self, user_id: str) -> Optional[DialogueContext]:
        """
        获取用户的当前上下文
        
        Args:
            user_id: 用户ID
            
        Returns:
            Optional[DialogueContext]: 用户的对话上下文
        """
        with self._lock:
            conversation_id = self._user_contexts.get(user_id)
            if conversation_id:
                context = self._contexts.get(conversation_id)
                if context and not self._is_context_expired(context):
                    return context
                else:
                    # 清理过期上下文
                    self._remove_context(conversation_id)
            
            return None
    
    def get_context_by_conversation(self, conversation_id: str) -> Optional[DialogueContext]:
        """
        通过对话ID获取上下文
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            Optional[DialogueContext]: 对话上下文
        """
        with self._lock:
            context = self._contexts.get(conversation_id)
            if context and not self._is_context_expired(context):
                return context
            elif context:
                # 清理过期上下文
                self._remove_context(conversation_id)
            
            return None
    
    def get_or_create_context(
        self,
        user_id: str,
        existing_context: Optional[DialogueContext] = None
    ) -> DialogueContext:
        """
        获取或创建上下文
        
        Args:
            user_id: 用户ID
            existing_context: 现有上下文（如果有的话）
            
        Returns:
            DialogueContext: 对话上下文
        """
        if existing_context:
            # 更新现有上下文
            self.update_context(existing_context)
            return existing_context
        
        # 尝试获取现有上下文
        context = self.get_context(user_id)
        if context:
            return context
        
        # 创建新上下文
        conversation_id = f"conv_{user_id}_{int(datetime.now().timestamp())}"
        return self.create_context(user_id, conversation_id)
    
    def update_context(self, context: DialogueContext):
        """
        更新对话上下文
        
        Args:
            context: 要更新的上下文
        """
        with self._lock:
            context.last_update = datetime.now()
            
            # 限制历史记录长度
            self._trim_context_history(context)
            
            # 更新存储
            self._contexts[context.conversation_id] = context
            
            # 更新统计
            self.stats["contexts_updated"] += 1
            
            logger.debug(f"Updated context for conversation {context.conversation_id}")
    
    def add_emotion_to_context(
        self,
        user_id: str,
        emotion_state: EmotionState
    ):
        """
        向上下文添加情感状态
        
        Args:
            user_id: 用户ID
            emotion_state: 情感状态
        """
        context = self.get_context(user_id)
        if context:
            context.add_emotion(emotion_state)
            self.update_context(context)
        else:
            logger.warning(f"No context found for user {user_id} when adding emotion")
    
    def add_response_to_context(
        self,
        user_id: str,
        response: EmpathyResponse
    ):
        """
        向上下文添加共情响应
        
        Args:
            user_id: 用户ID
            response: 共情响应
        """
        context = self.get_context(user_id)
        if context:
            context.add_response(response)
            self.update_context(context)
        else:
            logger.warning(f"No context found for user {user_id} when adding response")
    
    def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """
        获取对话摘要
        
        Args:
            user_id: 用户ID
            
        Returns:
            Dict[str, Any]: 对话摘要信息
        """
        context = self.get_context(user_id)
        if not context:
            return {"error": "No context found"}
        
        # 计算情感统计
        emotion_patterns = context.get_emotional_pattern()
        recent_emotions = context.get_recent_emotions(10)
        
        # 计算响应统计
        recent_responses = context.get_recent_responses(10)
        strategy_usage = {}
        for response in recent_responses:
            strategy = response.empathy_type.value
            strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
        
        # 计算平均指标
        avg_comfort_level = (
            sum(r.comfort_level for r in recent_responses) / len(recent_responses)
            if recent_responses else 0.0
        )
        
        avg_personalization = (
            sum(r.personalization_score for r in recent_responses) / len(recent_responses)
            if recent_responses else 0.0
        )
        
        return {
            "conversation_id": context.conversation_id,
            "start_time": context.start_time.isoformat(),
            "last_update": context.last_update.isoformat(),
            "conversation_length": context.conversation_length,
            "emotional_patterns": emotion_patterns,
            "recent_emotional_trend": self._analyze_emotional_trend(recent_emotions),
            "strategy_usage": strategy_usage,
            "avg_comfort_level": round(avg_comfort_level, 2),
            "avg_personalization": round(avg_personalization, 2),
            "avg_response_time": round(context.average_response_time, 2)
        }
    
    def clear_user_context(self, user_id: str) -> bool:
        """
        清理用户的上下文
        
        Args:
            user_id: 用户ID
            
        Returns:
            bool: 是否成功清理
        """
        with self._lock:
            conversation_id = self._user_contexts.get(user_id)
            if conversation_id:
                return self._remove_context(conversation_id)
            return False
    
    def _trim_context_history(self, context: DialogueContext):
        """修剪上下文历史记录"""
        # 限制情感历史长度
        if len(context.emotion_history) > self.config["max_emotion_history"]:
            context.emotion_history = context.emotion_history[-self.config["max_emotion_history"]:]
        
        # 限制回应历史长度
        if len(context.response_history) > self.config["max_response_history"]:
            context.response_history = context.response_history[-self.config["max_response_history"]:]
        
        # 更新情感弧线
        context.emotional_arc = [e.emotion for e in context.emotion_history[-10:]]
    
    def _is_context_expired(self, context: DialogueContext) -> bool:
        """检查上下文是否过期"""
        expiry_time = context.last_update + timedelta(hours=self.config["context_ttl_hours"])
        return datetime.now() > expiry_time
    
    def _remove_context(self, conversation_id: str) -> bool:
        """移除指定的上下文"""
        context = self._contexts.pop(conversation_id, None)
        if context:
            # 移除用户映射
            user_id = context.user_id
            if self._user_contexts.get(user_id) == conversation_id:
                del self._user_contexts[user_id]
            
            self.stats["active_contexts"] -= 1
            logger.info(f"Removed context for conversation {conversation_id}")
            return True
        
        return False
    
    def _maybe_cleanup(self):
        """检查是否需要清理"""
        now = datetime.now()
        if (now - self._last_cleanup).total_seconds() > self.config["cleanup_interval_minutes"] * 60:
            self._cleanup_expired_contexts()
            self._last_cleanup = now
    
    def _cleanup_expired_contexts(self):
        """清理过期的上下文"""
        expired_contexts = []
        
        for conversation_id, context in self._contexts.items():
            if self._is_context_expired(context):
                expired_contexts.append(conversation_id)
        
        for conversation_id in expired_contexts:
            self._remove_context(conversation_id)
        
        if expired_contexts:
            self.stats["contexts_cleaned"] += len(expired_contexts)
            logger.info(f"Cleaned up {len(expired_contexts)} expired contexts")
    
    def _force_cleanup(self):
        """强制清理最旧的上下文"""
        # 找到最旧的上下文
        oldest_context = None
        oldest_time = datetime.now()
        
        for conversation_id, context in self._contexts.items():
            if context.last_update < oldest_time:
                oldest_time = context.last_update
                oldest_context = conversation_id
        
        if oldest_context:
            self._remove_context(oldest_context)
            logger.info(f"Force removed oldest context: {oldest_context}")
    
    def _analyze_emotional_trend(self, emotions: List[EmotionState]) -> str:
        """分析情感趋势"""
        if len(emotions) < 2:
            return "insufficient_data"
        
        # 计算效价变化
        valence_changes = []
        for i in range(1, len(emotions)):
            valence_changes.append(emotions[i].valence - emotions[i-1].valence)
        
        avg_change = sum(valence_changes) / len(valence_changes)
        
        if avg_change > 0.1:
            return "improving"
        elif avg_change < -0.1:
            return "deteriorating"
        else:
            return "stable"
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """获取上下文统计信息"""
        with self._lock:
            # 计算活跃度分布
            activity_distribution = defaultdict(int)
            for context in self._contexts.values():
                hours_since_update = (datetime.now() - context.last_update).total_seconds() / 3600
                if hours_since_update < 1:
                    activity_distribution["very_recent"] += 1
                elif hours_since_update < 6:
                    activity_distribution["recent"] += 1
                elif hours_since_update < 24:
                    activity_distribution["today"] += 1
                else:
                    activity_distribution["older"] += 1
            
            return {
                **self.stats,
                "activity_distribution": dict(activity_distribution),
                "memory_usage": len(self._contexts),
                "user_mapping_size": len(self._user_contexts)
            }
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(new_config)
        logger.info(f"Updated context manager configuration: {new_config}")
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        with self._lock:
            total_contexts = len(self._contexts)
            expired_count = sum(1 for ctx in self._contexts.values() if self._is_context_expired(ctx))
            
            # 内存使用估算（简单）
            memory_estimate = total_contexts * 1024  # 假设每个上下文占用1KB
            
            return {
                "healthy": True,
                "total_contexts": total_contexts,
                "expired_contexts": expired_count,
                "memory_estimate_bytes": memory_estimate,
                "last_cleanup": self._last_cleanup.isoformat(),
                "stats": self.stats
            }