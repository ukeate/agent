"""分布式状态管理器实现"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from .models import TaskStatus
from src.core.utils.timezone_utils import utc_now

class DistributedStateManager:
    """分布式状态管理器"""
    
    def __init__(self, node_id: str, raft_consensus=None):
        self.node_id = node_id
        self.raft_consensus = raft_consensus
        self.logger = get_logger(__name__)
        
        # 状态存储
        self.global_state: Dict[str, Any] = {}
        self.local_state: Dict[str, Any] = {}
        self.state_locks: Dict[str, asyncio.Lock] = {}
        
        # 状态同步配置
        self.sync_interval = 5.0  # 秒
        self.consistency_check_interval = 30.0  # 秒
        
        # 状态变更监听器
        self.state_listeners: Dict[str, List[Callable]] = {}
        
        # 检查点
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        
        # 启动状态管理循环
        asyncio.create_task(self._start_state_sync_loop())
        asyncio.create_task(self._start_consistency_check_loop())
    
    async def set_global_state(
        self, 
        key: str, 
        value: Any, 
        atomic: bool = True
    ) -> bool:
        """设置全局状态"""
        
        try:
            if atomic and self.raft_consensus:
                # 通过Raft共识设置状态
                command = {
                    "action": "set_state",
                    "key": key,
                    "value": value,
                    "timestamp": utc_now().isoformat(),
                    "node_id": self.node_id
                }
                
                success = await self.raft_consensus.append_entry(command)
                
                if success:
                    # 在本地也更新状态
                    await self._update_local_state(key, value)
                
                return success
            else:
                # 直接设置本地状态
                await self._update_local_state(key, value)
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to set global state {key}: {e}")
            return False
    
    async def get_global_state(self, key: str) -> Optional[Any]:
        """获取全局状态"""
        
        # 优先从本地状态获取
        if key in self.global_state:
            return self.global_state[key]
        
        return None
    
    async def delete_global_state(self, key: str) -> bool:
        """删除全局状态"""
        
        try:
            if self.raft_consensus:
                command = {
                    "action": "delete_state",
                    "key": key,
                    "timestamp": utc_now().isoformat(),
                    "node_id": self.node_id
                }
                
                success = await self.raft_consensus.append_entry(command)
                
                if success and key in self.global_state:
                    del self.global_state[key]
                
                return success
            else:
                if key in self.global_state:
                    del self.global_state[key]
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to delete global state {key}: {e}")
            return False
    
    async def acquire_lock(self, lock_name: str, timeout: float = 30.0) -> bool:
        """获取分布式锁"""
        
        try:
            # 通过Raft共识获取锁
            if self.raft_consensus:
                command = {
                    "action": "acquire_lock",
                    "lock_name": lock_name,
                    "node_id": self.node_id,
                    "timestamp": utc_now().isoformat(),
                    "timeout": timeout
                }
                
                success = await self.raft_consensus.append_entry(command)
                
                if success:
                    # 在本地创建锁对象
                    if lock_name not in self.state_locks:
                        self.state_locks[lock_name] = asyncio.Lock()
                    
                    # 尝试获取本地锁
                    try:
                        await asyncio.wait_for(
                            self.state_locks[lock_name].acquire(), 
                            timeout=timeout
                        )
                        return True
                    except asyncio.TimeoutError:
                        # 释放分布式锁
                        await self.release_lock(lock_name)
                        return False
            else:
                # 仅使用本地锁
                if lock_name not in self.state_locks:
                    self.state_locks[lock_name] = asyncio.Lock()
                
                try:
                    await asyncio.wait_for(
                        self.state_locks[lock_name].acquire(),
                        timeout=timeout
                    )
                    return True
                except asyncio.TimeoutError:
                    return False
            
        except Exception as e:
            self.logger.error(f"Failed to acquire lock {lock_name}: {e}")
            return False
    
    async def release_lock(self, lock_name: str) -> bool:
        """释放分布式锁"""
        
        try:
            if self.raft_consensus:
                command = {
                    "action": "release_lock",
                    "lock_name": lock_name,
                    "node_id": self.node_id,
                    "timestamp": utc_now().isoformat()
                }
                
                success = await self.raft_consensus.append_entry(command)
                
                if success and lock_name in self.state_locks:
                    if self.state_locks[lock_name].locked():
                        self.state_locks[lock_name].release()
                
                return success
            else:
                if lock_name in self.state_locks and self.state_locks[lock_name].locked():
                    self.state_locks[lock_name].release()
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to release lock {lock_name}: {e}")
            return False
    
    async def atomic_update(
        self, 
        updates: Dict[str, Any], 
        conditions: Optional[Dict[str, Any]] = None
    ) -> bool:
        """原子性更新多个状态"""
        
        try:
            # 检查条件
            if conditions:
                for key, expected_value in conditions.items():
                    current_value = await self.get_global_state(key)
                    if current_value != expected_value:
                        self.logger.warning(f"Condition failed for {key}: {current_value} != {expected_value}")
                        return False
            
            if self.raft_consensus:
                command = {
                    "action": "atomic_update",
                    "updates": updates,
                    "conditions": conditions,
                    "timestamp": utc_now().isoformat(),
                    "node_id": self.node_id
                }
                
                success = await self.raft_consensus.append_entry(command)
                
                if success:
                    # 在本地应用更新
                    for key, value in updates.items():
                        await self._update_local_state(key, value)
                
                return success
            else:
                # 本地原子更新
                for key, value in updates.items():
                    await self._update_local_state(key, value)
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to perform atomic update: {e}")
            return False
    
    async def create_checkpoint(self, name: str) -> bool:
        """创建状态检查点"""
        
        try:
            checkpoint = {
                "state_snapshot": self.global_state.copy(),
                "timestamp": utc_now().isoformat(),
                "node_id": self.node_id
            }
            
            self.checkpoints[name] = checkpoint
            
            if self.raft_consensus:
                command = {
                    "action": "create_checkpoint",
                    "name": name,
                    "checkpoint": checkpoint
                }
                
                return await self.raft_consensus.append_entry(command)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint {name}: {e}")
            return False
    
    async def rollback_state(self, checkpoint_name: str) -> bool:
        """回滚状态到指定检查点"""
        
        try:
            if checkpoint_name not in self.checkpoints:
                self.logger.error(f"Checkpoint {checkpoint_name} not found")
                return False
            
            checkpoint = self.checkpoints[checkpoint_name]
            self.global_state = checkpoint["state_snapshot"].copy()
            
            if self.raft_consensus:
                command = {
                    "action": "rollback_state",
                    "checkpoint": checkpoint_name,
                    "timestamp": utc_now().isoformat(),
                    "node_id": self.node_id
                }
                
                return await self.raft_consensus.append_entry(command)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback state to {checkpoint_name}: {e}")
            return False
    
    def add_state_listener(self, pattern: str, listener: Callable):
        """添加状态变更监听器"""
        
        if pattern not in self.state_listeners:
            self.state_listeners[pattern] = []
        
        self.state_listeners[pattern].append(listener)
    
    async def _update_local_state(self, key: str, value: Any):
        """更新本地状态"""
        
        old_value = self.global_state.get(key)
        self.global_state[key] = value
        
        # 通知监听器
        await self._notify_state_listeners(key, old_value, value)
    
    async def _notify_state_listeners(self, key: str, old_value: Any, new_value: Any):
        """通知状态监听器"""
        
        for pattern, listeners in self.state_listeners.items():
            if self._match_pattern(key, pattern):
                for listener in listeners:
                    try:
                        if asyncio.iscoroutinefunction(listener):
                            await listener(key, old_value, new_value)
                        else:
                            listener(key, old_value, new_value)
                    except Exception as e:
                        self.logger.error(f"Error in state listener: {e}")
    
    def _match_pattern(self, key: str, pattern: str) -> bool:
        """匹配模式"""
        
        # 简单的通配符匹配
        if pattern == "*":
            return True
        
        if pattern.endswith("*"):
            return key.startswith(pattern[:-1])
        
        return key == pattern
    
    async def _start_state_sync_loop(self):
        """启动状态同步循环"""
        
        while True:
            try:
                await self._sync_state_with_peers()
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                self.logger.error(f"Error in state sync loop: {e}")
                await asyncio.sleep(self.sync_interval)
    
    async def _sync_state_with_peers(self):
        """与peers同步状态"""
        
        # 这里简化实现，实际应该通过消息总线同步
        # 在Raft共识层已经处理了状态同步
        self.logger.debug("状态同步由共识层处理，跳过直接同步")
    
    async def _start_consistency_check_loop(self):
        """启动一致性检查循环"""
        
        while True:
            try:
                await self._check_state_consistency()
                await asyncio.sleep(self.consistency_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in consistency check loop: {e}")
                await asyncio.sleep(self.consistency_check_interval)
    
    async def _check_state_consistency(self):
        """检查状态一致性"""
        
        # 检查关键状态的一致性
        inconsistencies = []
        
        # 检查任务状态一致性
        task_states = {k: v for k, v in self.global_state.items() if k.startswith("task_")}
        
        for task_key, task_data in task_states.items():
            if not self._validate_task_state(task_data):
                inconsistencies.append(f"Invalid task state: {task_key}")
        
        if inconsistencies:
            self.logger.warning(f"Found state inconsistencies: {inconsistencies}")
            # 触发修复机制
            await self._repair_inconsistencies(inconsistencies)
    
    def _validate_task_state(self, task_data: Dict[str, Any]) -> bool:
        """验证任务状态"""
        
        if not isinstance(task_data, dict):
            return False
        
        required_fields = ["task_id", "status", "created_at"]
        
        for field in required_fields:
            if field not in task_data:
                return False
        
        # 检查状态值是否有效
        valid_statuses = [status.value for status in TaskStatus]
        if task_data.get("status") not in valid_statuses:
            return False
        
        return True
    
    async def _repair_inconsistencies(self, inconsistencies: List[str]):
        """修复不一致性"""
        
        for inconsistency in inconsistencies:
            self.logger.info(f"Attempting to repair: {inconsistency}")
            
            # 这里可以实现具体的修复逻辑
            # 例如：重新从可信源同步状态、删除无效状态等
            if "Invalid task state" in inconsistency:
                key = inconsistency.split(": ")[1]
                # 可以选择删除无效状态或尝试修复
                await self.delete_global_state(key)
    
    async def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要"""
        
        return {
            "node_id": self.node_id,
            "total_keys": len(self.global_state),
            "total_locks": len(self.state_locks),
            "active_locks": sum(1 for lock in self.state_locks.values() if lock.locked()),
            "checkpoints": list(self.checkpoints.keys()),
            "listeners": len(self.state_listeners)
        }
from src.core.logging import get_logger
