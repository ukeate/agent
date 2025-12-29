"""
向量时钟管理器

实现版本向量时钟、因果关系检测和时钟同步机制
"""

from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
from ..models.schemas.offline import VectorClock

from src.core.logging import get_logger
class CausalRelation(str, Enum):
    """因果关系"""
    BEFORE = "before"           # A 发生在 B 之前
    AFTER = "after"             # A 发生在 B 之后
    CONCURRENT = "concurrent"   # A 和 B 并发
    EQUAL = "equal"             # A 和 B 相等

@dataclass
class ClockSyncResult:
    """时钟同步结果"""
    local_clock: VectorClock
    remote_clock: VectorClock
    merged_clock: VectorClock
    conflicts_detected: bool
    sync_timestamp: datetime = field(default_factory=utc_now)

class VectorClockManager:
    """向量时钟管理器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        # 节点时钟缓存
        self.node_clocks: Dict[str, VectorClock] = {}
        
        # 同步历史
        self.sync_history: List[ClockSyncResult] = []
        self.max_history_size = 1000
        
        # 时钟配置
        self.max_clock_drift = 1000  # 最大时钟偏移
        self.cleanup_threshold = 10000  # 清理阈值
    
    def get_or_create_clock(self, node_id: str) -> VectorClock:
        """获取或创建节点时钟"""
        if node_id not in self.node_clocks:
            self.node_clocks[node_id] = VectorClock(node_id=node_id)
        return self.node_clocks[node_id]
    
    def increment_clock(self, node_id: str) -> VectorClock:
        """递增节点时钟"""
        clock = self.get_or_create_clock(node_id)
        clock.increment(node_id)
        return clock
    
    def update_clock(self, node_id: str, other_clock: VectorClock) -> VectorClock:
        """更新本地时钟"""
        local_clock = self.get_or_create_clock(node_id)
        
        # 创建新的合并时钟
        merged_clock = VectorClock(node_id=node_id)
        
        # 合并时钟值
        all_nodes = set(local_clock.clock.keys()) | set(other_clock.clock.keys())
        
        for node in all_nodes:
            local_value = local_clock.clock.get(node, 0)
            other_value = other_clock.clock.get(node, 0)
            merged_clock.clock[node] = max(local_value, other_value)
        
        # 递增本地节点时钟
        merged_clock.increment(node_id)
        
        # 更新缓存
        self.node_clocks[node_id] = merged_clock
        
        return merged_clock
    
    def compare_clocks(self, clock1: VectorClock, clock2: VectorClock) -> CausalRelation:
        """比较两个向量时钟"""
        # 获取所有节点
        all_nodes = set(clock1.clock.keys()) | set(clock2.clock.keys())
        
        clock1_greater = False
        clock2_greater = False
        
        for node in all_nodes:
            value1 = clock1.clock.get(node, 0)
            value2 = clock2.clock.get(node, 0)
            
            if value1 > value2:
                clock1_greater = True
            elif value1 < value2:
                clock2_greater = True
        
        # 判断关系
        if clock1_greater and not clock2_greater:
            return CausalRelation.AFTER  # clock1 在 clock2 之后
        elif clock2_greater and not clock1_greater:
            return CausalRelation.BEFORE  # clock1 在 clock2 之前
        elif not clock1_greater and not clock2_greater:
            return CausalRelation.EQUAL  # 时钟相等
        else:
            return CausalRelation.CONCURRENT  # 并发事件
    
    def detect_conflict(self, local_clock: VectorClock, remote_clock: VectorClock) -> bool:
        """检测时钟冲突"""
        relation = self.compare_clocks(local_clock, remote_clock)
        return relation == CausalRelation.CONCURRENT
    
    def can_merge_safely(self, local_clock: VectorClock, remote_clock: VectorClock) -> bool:
        """检查是否可以安全合并"""
        relation = self.compare_clocks(local_clock, remote_clock)
        
        # 如果一个时钟完全在另一个之前/之后，可以安全合并
        return relation in [CausalRelation.BEFORE, CausalRelation.AFTER, CausalRelation.EQUAL]
    
    def merge_clocks(
        self, 
        local_clock: VectorClock, 
        remote_clock: VectorClock,
        resolve_conflicts: bool = True
    ) -> ClockSyncResult:
        """合并两个时钟"""
        # 检测冲突
        conflicts_detected = self.detect_conflict(local_clock, remote_clock)
        
        if conflicts_detected and not resolve_conflicts:
            # 如果不解决冲突，返回原始时钟
            return ClockSyncResult(
                local_clock=local_clock,
                remote_clock=remote_clock,
                merged_clock=local_clock,
                conflicts_detected=True
            )
        
        # 创建合并时钟
        merged_clock = VectorClock(node_id=local_clock.node_id)
        
        # 获取所有节点
        all_nodes = set(local_clock.clock.keys()) | set(remote_clock.clock.keys())
        
        for node in all_nodes:
            local_value = local_clock.clock.get(node, 0)
            remote_value = remote_clock.clock.get(node, 0)
            
            if conflicts_detected:
                # 冲突解决：取较大值
                merged_clock.clock[node] = max(local_value, remote_value)
            else:
                # 无冲突：取较大值
                merged_clock.clock[node] = max(local_value, remote_value)
        
        # 递增本地节点时钟
        merged_clock.increment(local_clock.node_id)
        
        result = ClockSyncResult(
            local_clock=local_clock,
            remote_clock=remote_clock,
            merged_clock=merged_clock,
            conflicts_detected=conflicts_detected
        )
        
        # 记录同步历史
        self._add_sync_history(result)
        
        return result
    
    def sync_with_remote(
        self, 
        local_node_id: str, 
        remote_clocks: List[VectorClock]
    ) -> Dict[str, ClockSyncResult]:
        """与多个远程时钟同步"""
        local_clock = self.get_or_create_clock(local_node_id)
        sync_results = {}
        
        for remote_clock in remote_clocks:
            result = self.merge_clocks(local_clock, remote_clock)
            sync_results[remote_clock.node_id] = result
            
            # 更新本地时钟
            local_clock = result.merged_clock
            self.node_clocks[local_node_id] = local_clock
        
        return sync_results
    
    def get_clock_dependencies(self, clock: VectorClock) -> Set[str]:
        """获取时钟依赖的节点"""
        return set(node for node, value in clock.clock.items() if value > 0)
    
    def is_causally_ready(
        self, 
        event_clock: VectorClock, 
        local_clock: VectorClock
    ) -> bool:
        """检查事件是否因果就绪"""
        # 事件因果就绪的条件：
        # 1. 对于事件源节点：event_clock[source] = local_clock[source] + 1
        # 2. 对于其他节点：event_clock[node] <= local_clock[node]
        
        source_node = event_clock.node_id
        
        # 检查源节点
        event_source_value = event_clock.clock.get(source_node, 0)
        local_source_value = local_clock.clock.get(source_node, 0)
        
        if event_source_value != local_source_value + 1:
            return False
        
        # 检查其他节点
        for node, event_value in event_clock.clock.items():
            if node != source_node:
                local_value = local_clock.clock.get(node, 0)
                if event_value > local_value:
                    return False
        
        return True
    
    def calculate_clock_distance(
        self, 
        clock1: VectorClock, 
        clock2: VectorClock
    ) -> int:
        """计算时钟距离"""
        all_nodes = set(clock1.clock.keys()) | set(clock2.clock.keys())
        total_distance = 0
        
        for node in all_nodes:
            value1 = clock1.clock.get(node, 0)
            value2 = clock2.clock.get(node, 0)
            total_distance += abs(value1 - value2)
        
        return total_distance
    
    def find_common_ancestor(
        self, 
        clock1: VectorClock, 
        clock2: VectorClock
    ) -> VectorClock:
        """找到两个时钟的公共祖先"""
        ancestor_clock = VectorClock(node_id="common_ancestor")
        
        all_nodes = set(clock1.clock.keys()) | set(clock2.clock.keys())
        
        for node in all_nodes:
            value1 = clock1.clock.get(node, 0)
            value2 = clock2.clock.get(node, 0)
            # 公共祖先取较小值
            ancestor_clock.clock[node] = min(value1, value2)
        
        return ancestor_clock
    
    def validate_clock_consistency(self, clocks: List[VectorClock]) -> bool:
        """验证时钟一致性"""
        if len(clocks) < 2:
            return True
        
        # 检查所有时钟对是否存在循环依赖
        for i in range(len(clocks)):
            for j in range(i + 1, len(clocks)):
                relation1 = self.compare_clocks(clocks[i], clocks[j])
                relation2 = self.compare_clocks(clocks[j], clocks[i])
                
                # 检查一致性
                if relation1 == CausalRelation.BEFORE and relation2 != CausalRelation.AFTER:
                    return False
                elif relation1 == CausalRelation.AFTER and relation2 != CausalRelation.BEFORE:
                    return False
                elif relation1 == CausalRelation.EQUAL and relation2 != CausalRelation.EQUAL:
                    return False
        
        return True
    
    def compress_clock(self, clock: VectorClock, active_nodes: Set[str]) -> VectorClock:
        """压缩时钟（移除非活跃节点）"""
        compressed_clock = VectorClock(node_id=clock.node_id)
        
        for node, value in clock.clock.items():
            if node in active_nodes or node == clock.node_id:
                compressed_clock.clock[node] = value
        
        return compressed_clock
    
    def _add_sync_history(self, result: ClockSyncResult):
        """添加同步历史"""
        self.sync_history.append(result)
        
        # 清理过多的历史记录
        if len(self.sync_history) > self.max_history_size:
            self.sync_history = self.sync_history[-self.max_history_size:]
    
    def get_sync_statistics(self) -> Dict[str, any]:
        """获取同步统计信息"""
        if not self.sync_history:
            return {
                "total_syncs": 0,
                "conflicts_detected": 0,
                "conflict_rate": 0.0,
                "active_nodes": 0
            }
        
        total_syncs = len(self.sync_history)
        conflicts_detected = sum(1 for result in self.sync_history if result.conflicts_detected)
        conflict_rate = conflicts_detected / total_syncs if total_syncs > 0 else 0.0
        
        return {
            "total_syncs": total_syncs,
            "conflicts_detected": conflicts_detected,
            "conflict_rate": conflict_rate,
            "active_nodes": len(self.node_clocks),
            "recent_sync_time": self.sync_history[-1].sync_timestamp.isoformat() if self.sync_history else None
        }
    
    def cleanup_old_clocks(self, active_nodes: Set[str]):
        """清理非活跃节点的时钟"""
        nodes_to_remove = []
        
        for node_id in self.node_clocks:
            if node_id not in active_nodes:
                # 检查时钟是否过于陈旧
                clock = self.node_clocks[node_id]
                max_value = max(clock.clock.values()) if clock.clock else 0
                
                if max_value > self.cleanup_threshold:
                    nodes_to_remove.append(node_id)
        
        # 移除陈旧时钟
        for node_id in nodes_to_remove:
            del self.node_clocks[node_id]
    
    def export_clock_state(self) -> Dict[str, any]:
        """导出时钟状态"""
        return {
            "node_clocks": {
                node_id: {
                    "node_id": clock.node_id,
                    "clock": clock.clock.copy()
                }
                for node_id, clock in self.node_clocks.items()
            },
            "sync_history_count": len(self.sync_history),
            "export_timestamp": utc_now().isoformat()
        }
    
    def import_clock_state(self, state_data: Dict[str, any]) -> bool:
        """导入时钟状态"""
        try:
            node_clocks_data = state_data.get("node_clocks", {})
            
            for node_id, clock_data in node_clocks_data.items():
                clock = VectorClock(
                    node_id=clock_data["node_id"],
                    clock=clock_data["clock"].copy()
                )
                self.node_clocks[node_id] = clock
            
            return True
            
        except Exception as e:
            self.logger.error("导入时钟状态失败", error=str(e))
            return False
