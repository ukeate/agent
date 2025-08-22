"""
冲突解决器

实现自动冲突解决、支持多种解决策略和创建用户交互接口
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from ..models.schemas.offline import (
    SyncOperation, SyncOperationType, VectorClock,
    ConflictRecord, ConflictType, ConflictResolutionStrategy
)
from .conflict_detector import ConflictContext, ConflictSeverity, ConflictCategory
from .merge_strategies import MergeStrategies


class ResolutionMethod(str, Enum):
    """解决方法"""
    AUTOMATIC = "automatic"      # 自动解决
    INTERACTIVE = "interactive"  # 交互式解决
    MANUAL = "manual"           # 手动解决
    POLICY_BASED = "policy_based"  # 基于策略解决


class ResolutionStatus(str, Enum):
    """解决状态"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    FAILED = "failed"
    REQUIRES_MANUAL_INTERVENTION = "requires_manual_intervention"


@dataclass
class ResolutionResult:
    """解决结果"""
    conflict_id: str
    resolution_method: ResolutionMethod
    resolution_strategy: ConflictResolutionStrategy
    status: ResolutionStatus
    resolved_data: Optional[Dict[str, Any]] = None
    confidence_score: float = 0.0
    resolution_time: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


@dataclass
class InteractionRequest:
    """用户交互请求"""
    request_id: str
    conflict_context: ConflictContext
    suggested_resolutions: List[Tuple[ConflictResolutionStrategy, Dict[str, Any]]]
    deadline: Optional[datetime] = None
    priority: int = 1  # 1-5, 5最高
    context_info: Dict[str, Any] = field(default_factory=dict)


class ConflictResolver:
    """冲突解决器"""
    
    def __init__(self):
        self.merge_strategies = MergeStrategies()
        
        # 解决策略映射
        self.strategy_handlers = {
            ConflictResolutionStrategy.LAST_WRITER_WINS: self._resolve_last_writer_wins,
            ConflictResolutionStrategy.FIRST_WRITER_WINS: self._resolve_first_writer_wins,
            ConflictResolutionStrategy.CLIENT_WINS: self._resolve_client_wins,
            ConflictResolutionStrategy.SERVER_WINS: self._resolve_server_wins,
            ConflictResolutionStrategy.MERGE: self._resolve_merge,
            ConflictResolutionStrategy.MANUAL: self._resolve_manual
        }
        
        # 自动解决策略选择器
        self.auto_strategy_selectors = {
            ConflictType.UPDATE_UPDATE: self._select_update_update_strategy,
            ConflictType.CREATE_CREATE: self._select_create_create_strategy,
            ConflictType.UPDATE_DELETE: self._select_update_delete_strategy,
            ConflictType.DELETE_UPDATE: self._select_delete_update_strategy,
        }
        
        # 解决策略配置
        self.resolution_config = {
            "auto_resolve_enabled": True,
            "auto_resolve_confidence_threshold": 0.8,
            "prefer_client_for_ties": True,
            "max_merge_attempts": 3,
            "interaction_timeout_minutes": 30
        }
        
        # 用户交互管理
        self.pending_interactions: Dict[str, InteractionRequest] = {}
        self.interaction_callbacks: List[Callable[[InteractionRequest], None]] = []
        
        # 统计信息
        self.resolution_stats = {
            "total_resolutions": 0,
            "automatic_resolutions": 0,
            "manual_resolutions": 0,
            "interactive_resolutions": 0,
            "failed_resolutions": 0,
            "total_resolution_time_ms": 0.0
        }
    
    async def resolve_conflict(
        self,
        conflict_context: ConflictContext,
        preferred_strategy: Optional[ConflictResolutionStrategy] = None,
        user_context: Optional[Dict[str, Any]] = None
    ) -> ResolutionResult:
        """解决冲突"""
        start_time = datetime.utcnow()
        conflict_id = str(uuid4())
        
        try:
            # 确定解决策略
            if preferred_strategy:
                strategy = preferred_strategy
                method = ResolutionMethod.MANUAL
            elif conflict_context.auto_resolvable and self.resolution_config["auto_resolve_enabled"]:
                strategy = self._select_auto_strategy(conflict_context)
                method = ResolutionMethod.AUTOMATIC
            else:
                # 需要用户交互
                return await self._request_user_interaction(conflict_context, conflict_id)
            
            # 执行解决策略
            resolution_result = await self._execute_resolution_strategy(
                conflict_context, strategy, method, conflict_id, user_context
            )
            
            # 验证解决结果
            if resolution_result.status == ResolutionStatus.RESOLVED:
                validation_result = self._validate_resolution(
                    conflict_context, resolution_result
                )
                if not validation_result:
                    resolution_result.status = ResolutionStatus.FAILED
                    resolution_result.error_message = "解决结果验证失败"
            
            return resolution_result
            
        except Exception as e:
            return ResolutionResult(
                conflict_id=conflict_id,
                resolution_method=ResolutionMethod.AUTOMATIC,
                resolution_strategy=ConflictResolutionStrategy.MANUAL,
                status=ResolutionStatus.FAILED,
                error_message=str(e)
            )
        
        finally:
            # 更新统计
            end_time = datetime.utcnow()
            resolution_time = (end_time - start_time).total_seconds() * 1000
            self._update_resolution_stats(method, resolution_time)
    
    def _select_auto_strategy(self, conflict_context: ConflictContext) -> ConflictResolutionStrategy:
        """选择自动解决策略"""
        conflict_type = conflict_context.conflict_type
        
        if conflict_type in self.auto_strategy_selectors:
            return self.auto_strategy_selectors[conflict_type](conflict_context)
        
        # 默认策略
        return ConflictResolutionStrategy.MERGE
    
    def _select_update_update_strategy(self, conflict_context: ConflictContext) -> ConflictResolutionStrategy:
        """选择更新-更新冲突的策略"""
        local_op = conflict_context.local_operation
        remote_op = conflict_context.remote_operation
        
        # 基于时间戳选择
        if local_op.client_timestamp > remote_op.client_timestamp:
            return ConflictResolutionStrategy.LAST_WRITER_WINS
        elif local_op.client_timestamp < remote_op.client_timestamp:
            return ConflictResolutionStrategy.LAST_WRITER_WINS
        else:
            # 时间戳相同，尝试合并
            return ConflictResolutionStrategy.MERGE
    
    def _select_create_create_strategy(self, conflict_context: ConflictContext) -> ConflictResolutionStrategy:
        """选择创建-创建冲突的策略"""
        # 创建冲突通常需要手动处理，但可以尝试基于时间戳
        return ConflictResolutionStrategy.FIRST_WRITER_WINS
    
    def _select_update_delete_strategy(self, conflict_context: ConflictContext) -> ConflictResolutionStrategy:
        """选择更新-删除冲突的策略"""
        # 更新-删除冲突通常需要手动干预
        return ConflictResolutionStrategy.MANUAL
    
    def _select_delete_update_strategy(self, conflict_context: ConflictContext) -> ConflictResolutionStrategy:
        """选择删除-更新冲突的策略"""
        # 删除-更新冲突通常需要手动干预
        return ConflictResolutionStrategy.MANUAL
    
    async def _execute_resolution_strategy(
        self,
        conflict_context: ConflictContext,
        strategy: ConflictResolutionStrategy,
        method: ResolutionMethod,
        conflict_id: str,
        user_context: Optional[Dict[str, Any]] = None
    ) -> ResolutionResult:
        """执行解决策略"""
        if strategy not in self.strategy_handlers:
            return ResolutionResult(
                conflict_id=conflict_id,
                resolution_method=method,
                resolution_strategy=strategy,
                status=ResolutionStatus.FAILED,
                error_message=f"不支持的解决策略: {strategy}"
            )
        
        handler = self.strategy_handlers[strategy]
        
        try:
            resolved_data, confidence = await handler(conflict_context, user_context)
            
            return ResolutionResult(
                conflict_id=conflict_id,
                resolution_method=method,
                resolution_strategy=strategy,
                status=ResolutionStatus.RESOLVED,
                resolved_data=resolved_data,
                confidence_score=confidence,
                metadata={
                    "conflict_type": conflict_context.conflict_type.value,
                    "conflict_category": conflict_context.conflict_category.value,
                    "user_context": user_context or {}
                }
            )
            
        except Exception as e:
            return ResolutionResult(
                conflict_id=conflict_id,
                resolution_method=method,
                resolution_strategy=strategy,
                status=ResolutionStatus.FAILED,
                error_message=str(e)
            )
    
    async def _resolve_last_writer_wins(
        self,
        conflict_context: ConflictContext,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """最后写入者获胜策略"""
        local_op = conflict_context.local_operation
        remote_op = conflict_context.remote_operation
        
        # 比较时间戳
        if local_op.client_timestamp >= remote_op.client_timestamp:
            winner_data = local_op.data or {}
            confidence = 0.9
        else:
            winner_data = remote_op.data or {}
            confidence = 0.9
        
        return winner_data, confidence
    
    async def _resolve_first_writer_wins(
        self,
        conflict_context: ConflictContext,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """第一写入者获胜策略"""
        local_op = conflict_context.local_operation
        remote_op = conflict_context.remote_operation
        
        # 比较时间戳
        if local_op.client_timestamp <= remote_op.client_timestamp:
            winner_data = local_op.data or {}
            confidence = 0.9
        else:
            winner_data = remote_op.data or {}
            confidence = 0.9
        
        return winner_data, confidence
    
    async def _resolve_client_wins(
        self,
        conflict_context: ConflictContext,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """客户端获胜策略"""
        local_op = conflict_context.local_operation
        return local_op.data or {}, 0.8
    
    async def _resolve_server_wins(
        self,
        conflict_context: ConflictContext,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """服务器获胜策略"""
        remote_op = conflict_context.remote_operation
        return remote_op.data or {}, 0.8
    
    async def _resolve_merge(
        self,
        conflict_context: ConflictContext,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """合并策略"""
        local_data = conflict_context.local_operation.data or {}
        remote_data = conflict_context.remote_operation.data or {}
        
        # 使用合并策略
        merged_data, confidence = self.merge_strategies.three_way_merge(
            base_data={},  # 简化：假设空的基础数据
            local_data=local_data,
            remote_data=remote_data
        )
        
        return merged_data, confidence
    
    async def _resolve_manual(
        self,
        conflict_context: ConflictContext,
        user_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], float]:
        """手动解决策略"""
        # 手动策略需要用户提供解决数据
        if user_context and "resolved_data" in user_context:
            return user_context["resolved_data"], 1.0
        else:
            raise ValueError("手动解决策略需要用户提供解决数据")
    
    async def _request_user_interaction(
        self,
        conflict_context: ConflictContext,
        conflict_id: str
    ) -> ResolutionResult:
        """请求用户交互"""
        # 生成建议的解决方案
        suggested_resolutions = self._generate_resolution_suggestions(conflict_context)
        
        # 创建交互请求
        interaction_request = InteractionRequest(
            request_id=str(uuid4()),
            conflict_context=conflict_context,
            suggested_resolutions=suggested_resolutions,
            deadline=datetime.utcnow() + timedelta(
                minutes=self.resolution_config["interaction_timeout_minutes"]
            ),
            context_info={
                "conflict_id": conflict_id,
                "local_operation": {
                    "id": conflict_context.local_operation.id,
                    "timestamp": conflict_context.local_operation.client_timestamp.isoformat(),
                    "data": conflict_context.local_operation.data
                },
                "remote_operation": {
                    "id": conflict_context.remote_operation.id,
                    "timestamp": conflict_context.remote_operation.client_timestamp.isoformat(),
                    "data": conflict_context.remote_operation.data
                }
            }
        )
        
        # 保存待处理的交互
        self.pending_interactions[interaction_request.request_id] = interaction_request
        
        # 通知交互回调
        for callback in self.interaction_callbacks:
            try:
                callback(interaction_request)
            except Exception as e:
                print(f"交互回调错误: {e}")
        
        return ResolutionResult(
            conflict_id=conflict_id,
            resolution_method=ResolutionMethod.INTERACTIVE,
            resolution_strategy=ConflictResolutionStrategy.MANUAL,
            status=ResolutionStatus.REQUIRES_MANUAL_INTERVENTION,
            metadata={
                "interaction_request_id": interaction_request.request_id,
                "suggested_resolutions": len(suggested_resolutions)
            }
        )
    
    def _generate_resolution_suggestions(
        self,
        conflict_context: ConflictContext
    ) -> List[Tuple[ConflictResolutionStrategy, Dict[str, Any]]]:
        """生成解决建议"""
        suggestions = []
        
        # 基于冲突类型生成建议
        conflict_type = conflict_context.conflict_type
        local_data = conflict_context.local_operation.data or {}
        remote_data = conflict_context.remote_operation.data or {}
        
        if conflict_type == ConflictType.UPDATE_UPDATE:
            # 建议最后写入者获胜
            suggestions.append((
                ConflictResolutionStrategy.LAST_WRITER_WINS,
                {"description": "使用最新的更改", "preview": remote_data if conflict_context.remote_operation.client_timestamp > conflict_context.local_operation.client_timestamp else local_data}
            ))
            
            # 建议合并
            try:
                merged_data, _ = self.merge_strategies.three_way_merge({}, local_data, remote_data)
                suggestions.append((
                    ConflictResolutionStrategy.MERGE,
                    {"description": "合并两个版本的更改", "preview": merged_data}
                ))
            except Exception:
                pass
            
            # 建议客户端获胜
            suggestions.append((
                ConflictResolutionStrategy.CLIENT_WINS,
                {"description": "保留本地更改", "preview": local_data}
            ))
            
            # 建议服务器获胜
            suggestions.append((
                ConflictResolutionStrategy.SERVER_WINS,
                {"description": "使用服务器版本", "preview": remote_data}
            ))
        
        elif conflict_type == ConflictType.CREATE_CREATE:
            # 建议第一写入者获胜
            suggestions.append((
                ConflictResolutionStrategy.FIRST_WRITER_WINS,
                {"description": "保留第一个创建的版本", "preview": local_data if conflict_context.local_operation.client_timestamp <= conflict_context.remote_operation.client_timestamp else remote_data}
            ))
        
        return suggestions
    
    async def handle_user_response(
        self,
        request_id: str,
        chosen_strategy: ConflictResolutionStrategy,
        custom_data: Optional[Dict[str, Any]] = None
    ) -> ResolutionResult:
        """处理用户响应"""
        if request_id not in self.pending_interactions:
            raise ValueError(f"交互请求不存在: {request_id}")
        
        interaction_request = self.pending_interactions[request_id]
        conflict_context = interaction_request.conflict_context
        
        # 构建用户上下文
        user_context = {}
        if custom_data:
            user_context["resolved_data"] = custom_data
        
        # 执行用户选择的策略
        result = await self._execute_resolution_strategy(
            conflict_context,
            chosen_strategy,
            ResolutionMethod.INTERACTIVE,
            interaction_request.context_info["conflict_id"],
            user_context
        )
        
        # 清理待处理的交互
        del self.pending_interactions[request_id]
        
        return result
    
    def _validate_resolution(
        self,
        conflict_context: ConflictContext,
        resolution_result: ResolutionResult
    ) -> bool:
        """验证解决结果"""
        if not resolution_result.resolved_data:
            return False
        
        # 基本数据完整性检查
        try:
            json.dumps(resolution_result.resolved_data)
        except (TypeError, ValueError):
            return False
        
        # 检查解决结果是否包含必要字段
        local_data = conflict_context.local_operation.data or {}
        remote_data = conflict_context.remote_operation.data or {}
        resolved_data = resolution_result.resolved_data
        
        # 检查是否丢失了重要字段
        important_fields = {"id", "created_at", "user_id"}
        for field in important_fields:
            if (field in local_data or field in remote_data) and field not in resolved_data:
                return False
        
        return True
    
    def _update_resolution_stats(self, method: ResolutionMethod, resolution_time_ms: float):
        """更新解决统计"""
        self.resolution_stats["total_resolutions"] += 1
        self.resolution_stats["total_resolution_time_ms"] += resolution_time_ms
        
        if method == ResolutionMethod.AUTOMATIC:
            self.resolution_stats["automatic_resolutions"] += 1
        elif method == ResolutionMethod.INTERACTIVE:
            self.resolution_stats["interactive_resolutions"] += 1
        elif method == ResolutionMethod.MANUAL:
            self.resolution_stats["manual_resolutions"] += 1
    
    def add_interaction_callback(self, callback: Callable[[InteractionRequest], None]):
        """添加交互回调"""
        self.interaction_callbacks.append(callback)
    
    def remove_interaction_callback(self, callback: Callable[[InteractionRequest], None]):
        """移除交互回调"""
        if callback in self.interaction_callbacks:
            self.interaction_callbacks.remove(callback)
    
    def get_pending_interactions(self) -> List[InteractionRequest]:
        """获取待处理的交互"""
        return list(self.pending_interactions.values())
    
    def get_resolution_statistics(self) -> Dict[str, Any]:
        """获取解决统计信息"""
        stats = self.resolution_stats.copy()
        
        if stats["total_resolutions"] > 0:
            stats["average_resolution_time_ms"] = stats["total_resolution_time_ms"] / stats["total_resolutions"]
            stats["automatic_resolution_rate"] = stats["automatic_resolutions"] / stats["total_resolutions"]
            stats["interactive_resolution_rate"] = stats["interactive_resolutions"] / stats["total_resolutions"]
            stats["manual_resolution_rate"] = stats["manual_resolutions"] / stats["total_resolutions"]
        else:
            stats["average_resolution_time_ms"] = 0
            stats["automatic_resolution_rate"] = 0
            stats["interactive_resolution_rate"] = 0
            stats["manual_resolution_rate"] = 0
        
        return stats
    
    def configure_resolution(self, new_config: Dict[str, Any]):
        """配置解决设置"""
        self.resolution_config.update(new_config)
    
    def reset_statistics(self):
        """重置统计信息"""
        self.resolution_stats = {
            "total_resolutions": 0,
            "automatic_resolutions": 0,
            "manual_resolutions": 0,
            "interactive_resolutions": 0,
            "failed_resolutions": 0,
            "total_resolution_time_ms": 0.0
        }