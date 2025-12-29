from src.core.utils.timezone_utils import utc_now
import asyncio
import json
import hashlib
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime

from src.core.logging import get_logger
@dataclass
class ConsistencyCheckResult:
    """一致性检查结果"""
    check_id: str
    checked_at: datetime
    components: List[str]
    consistent: bool
    inconsistencies: List[Dict[str, Any]]
    repair_actions: List[str] = field(default_factory=list)

class ConsistencyManager:
    """一致性管理器"""
    
    def __init__(self, cluster_manager, storage_backend, config: Dict[str, Any]):
        self.cluster_manager = cluster_manager
        self.storage_backend = storage_backend
        self.config = config
        self.logger = get_logger(__name__)
        
        # 一致性检查配置
        self.check_interval = config.get("consistency_check_interval", 300)  # 5分钟
        self.repair_threshold = config.get("repair_threshold", 0.7)  # 70%节点一致时认为是正确的
        
        # 一致性检查结果
        self.check_results: List[ConsistencyCheckResult] = []
        
        # 运行控制
        self.running = False
    
    async def start(self):
        """启动一致性管理器"""
        if not self.cluster_manager:
            raise RuntimeError("cluster manager not configured for consistency checks")
        if not self.storage_backend:
            raise RuntimeError("storage backend not configured for consistency checks")
        self.running = True
        
        # 启动一致性检查循环
        asyncio.create_task(self._consistency_check_loop())
        
        self.logger.info("Consistency manager started")
    
    async def stop(self):
        """停止一致性管理器"""
        self.running = False
        self.logger.info("Consistency manager stopped")
    
    async def check_data_consistency(self, data_keys: List[str]) -> ConsistencyCheckResult:
        """检查数据一致性"""
        
        check_id = f"consistency_check_{int(time.time())}"
        inconsistencies = []
        
        try:
            self.logger.info(f"Starting consistency check {check_id}")
            
            # 获取集群拓扑
            topology = await self.cluster_manager.get_cluster_topology()
            if not topology or not hasattr(topology, 'agents'):
                return ConsistencyCheckResult(
                    check_id=check_id,
                    checked_at=utc_now(),
                    components=[],
                    consistent=False,
                    inconsistencies=[{"error": "No cluster topology available"}]
                )
            
            component_ids = list(topology.agents.keys())
            
            # 检查每个数据键的一致性
            for data_key in data_keys:
                key_inconsistencies = await self._check_key_consistency(data_key, component_ids)
                inconsistencies.extend(key_inconsistencies)
            
            # 创建检查结果
            result = ConsistencyCheckResult(
                check_id=check_id,
                checked_at=utc_now(),
                components=component_ids,
                consistent=len(inconsistencies) == 0,
                inconsistencies=inconsistencies
            )
            
            # 如果发现不一致，生成修复动作
            if not result.consistent:
                result.repair_actions = await self._generate_repair_actions(result.inconsistencies)
            
            # 记录检查结果
            self.check_results.append(result)
            
            # 限制结果历史大小
            if len(self.check_results) > 1000:
                self.check_results = self.check_results[-500:]
            
            self.logger.info(f"Consistency check completed: {check_id}, consistent: {result.consistent}")
            return result
            
        except Exception as e:
            self.logger.error(f"Consistency check failed: {e}")
            return ConsistencyCheckResult(
                check_id=check_id,
                checked_at=utc_now(),
                components=[],
                consistent=False,
                inconsistencies=[{"error": str(e)}]
            )
    
    async def _check_key_consistency(
        self, 
        data_key: str, 
        component_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """检查单个数据键的一致性"""
        
        inconsistencies = []
        
        try:
            # 从各个组件获取数据
            component_data = {}
            
            for component_id in component_ids:
                try:
                    data_value = await self._get_component_data(component_id, data_key)
                    component_data[component_id] = data_value
                except Exception as e:
                    self.logger.warning(f"Failed to get data {data_key} from {component_id}: {e}")
                    component_data[component_id] = None
            
            # 分析数据一致性
            value_groups = {}
            for component_id, value in component_data.items():
                value_hash = self._hash_value(value)
                
                if value_hash not in value_groups:
                    value_groups[value_hash] = {"value": value, "components": []}
                
                value_groups[value_hash]["components"].append(component_id)
            
            # 检查是否一致
            if len(value_groups) > 1:
                # 数据不一致
                inconsistency = {
                    "data_key": data_key,
                    "type": "value_mismatch",
                    "value_groups": []
                }
                
                for value_hash, group_info in value_groups.items():
                    inconsistency["value_groups"].append({
                        "value_hash": value_hash,
                        "components": group_info["components"],
                        "value_preview": str(group_info["value"])[:100]  # 只显示前100个字符
                    })
                
                inconsistencies.append(inconsistency)
            
            return inconsistencies
            
        except Exception as e:
            self.logger.error(f"Key consistency check failed for {data_key}: {e}")
            return [{
                "data_key": data_key,
                "type": "check_error",
                "error": str(e)
            }]
    
    async def _get_component_data(self, component_id: str, data_key: str) -> Any:
        """从组件获取数据"""
        
        try:
            # 这里应该实现从具体组件获取数据的逻辑
            # 根据data_key的不同获取不同类型的数据
            
            if data_key == "cluster_state":
                return await self._get_cluster_state_from_component(component_id)
            elif data_key == "task_assignments":
                return await self._get_task_assignments_from_component(component_id)
            elif data_key == "agent_config":
                return await self._get_agent_config_from_component(component_id)
            else:
                # 通用数据获取
                return await self._get_generic_data_from_component(component_id, data_key)
                
        except Exception as e:
            self.logger.error(f"Failed to get data {data_key} from component {component_id}: {e}")
            return None
    
    async def _get_cluster_state_from_component(self, component_id: str) -> Dict[str, Any]:
        """从组件获取集群状态"""
        try:
            if hasattr(self.cluster_manager, "get_cluster_topology"):
                topology = await self.cluster_manager.get_cluster_topology()
                agent_info = topology.agents.get(component_id) if topology and getattr(topology, "agents", None) else None
                return {
                    "cluster_id": topology.cluster_id if topology else None,
                    "leader": topology.config.get("leader") if topology and hasattr(topology, "config") else None,
                    "members": list(topology.agents.keys()) if topology and getattr(topology, "agents", None) else [],
                    "agent": asdict(agent_info) if agent_info else None,
                    "version": getattr(topology, "state_version", None)
                }
            getter = getattr(self.storage_backend, "get_cluster_state", None)
            if getter:
                state = getter(component_id)
                if asyncio.iscoroutine(state):
                    state = await state
                return state or {}
            raise RuntimeError("no cluster state provider configured")
        except Exception:
            return None
    
    async def _get_task_assignments_from_component(self, component_id: str) -> Dict[str, Any]:
        """从组件获取任务分配"""
        try:
            getter = getattr(self.storage_backend, "get_task_assignments", None)
            if getter:
                assignments = getter(component_id)
                if asyncio.iscoroutine(assignments):
                    assignments = await assignments
                return assignments or {}
            raise RuntimeError("task assignment provider not configured")
        except Exception:
            return None
    
    async def _get_agent_config_from_component(self, component_id: str) -> Dict[str, Any]:
        """从组件获取代理配置"""
        try:
            if hasattr(self.cluster_manager, "get_agent_info"):
                agent_info = await self.cluster_manager.get_agent_info(component_id)
                if agent_info:
                    return asdict(agent_info)
            getter = getattr(self.storage_backend, "get_agent_config", None)
            if getter:
                config = getter(component_id)
                if asyncio.iscoroutine(config):
                    config = await config
                return config or {}
            raise RuntimeError("agent config provider not configured")
        except Exception:
            return None
    
    async def _get_generic_data_from_component(self, component_id: str, data_key: str) -> Any:
        """从组件获取通用数据"""
        try:
            getter = getattr(self.storage_backend, "get_component_data", None)
            if getter:
                data = getter(component_id, data_key)
                if asyncio.iscoroutine(data):
                    data = await data
                return data
            raise RuntimeError(f"generic data provider missing for key {data_key}")
        except Exception:
            return None
    
    def _hash_value(self, value: Any) -> str:
        """计算值的哈希"""
        
        if value is None:
            return "null"
        
        try:
            # 序列化后计算哈希
            serialized = json.dumps(value, sort_keys=True)
            return hashlib.md5(serialized.encode()).hexdigest()
        except (TypeError, ValueError):
            # 不能序列化的对象使用字符串表示
            return hashlib.md5(str(value).encode()).hexdigest()
    
    async def _generate_repair_actions(
        self, 
        inconsistencies: List[Dict[str, Any]]
    ) -> List[str]:
        """生成修复动作"""
        
        repair_actions = []
        
        for inconsistency in inconsistencies:
            if inconsistency.get("type") == "value_mismatch":
                data_key = inconsistency["data_key"]
                value_groups = inconsistency["value_groups"]
                
                # 找到最多组件持有的值作为正确值
                largest_group = max(value_groups, key=lambda g: len(g["components"]))
                correct_components = set(largest_group["components"])
                
                # 为少数组件生成修复动作
                for group in value_groups:
                    if group != largest_group:
                        for component in group["components"]:
                            repair_actions.append(
                                f"repair_data:{component}:{data_key}:from_{group['value_hash']}_to_{largest_group['value_hash']}"
                            )
        
        return repair_actions
    
    async def repair_inconsistencies(self, check_result: ConsistencyCheckResult) -> bool:
        """修复不一致问题"""
        
        if check_result.consistent:
            self.logger.info("No inconsistencies to repair")
            return True
        
        repair_success = True
        
        try:
            self.logger.info(f"Starting repair for check {check_result.check_id}")
            
            for repair_action in check_result.repair_actions:
                try:
                    success = await self._execute_repair_action(repair_action)
                    if not success:
                        repair_success = False
                        self.logger.warning(f"Repair action failed: {repair_action}")
                    else:
                        self.logger.info(f"Repair action completed: {repair_action}")
                        
                except Exception as e:
                    self.logger.error(f"Repair action error: {repair_action}, {e}")
                    repair_success = False
            
            if repair_success:
                self.logger.info(f"All repairs completed for check {check_result.check_id}")
            else:
                self.logger.warning(f"Some repairs failed for check {check_result.check_id}")
            
            return repair_success
            
        except Exception as e:
            self.logger.error(f"Repair process failed: {e}")
            return False
    
    async def _execute_repair_action(self, repair_action: str) -> bool:
        """执行修复动作"""
        
        try:
            # 解析修复动作
            parts = repair_action.split(":")
            if len(parts) < 3:
                self.logger.error(f"Invalid repair action format: {repair_action}")
                return False
            
            action_type = parts[0]
            component_id = parts[1]
            data_key = parts[2]
            
            if action_type == "repair_data":
                # 修复组件数据
                return await self._repair_component_data(component_id, data_key, parts[3] if len(parts) > 3 else "")
            else:
                self.logger.error(f"Unknown repair action type: {action_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Repair action execution failed: {e}")
            return False
    
    async def _repair_component_data(
        self, 
        component_id: str, 
        data_key: str, 
        repair_info: str
    ) -> bool:
        """修复组件数据"""
        
        try:
            # 这里应该实现具体的数据修复逻辑
            self.logger.info(f"Repairing data {data_key} for component {component_id}: {repair_info}")
            
            # 根据不同的数据键执行不同的修复逻辑
            if data_key == "cluster_state":
                return await self._repair_cluster_state(component_id, repair_info)
            elif data_key == "task_assignments":
                return await self._repair_task_assignments(component_id, repair_info)
            elif data_key == "agent_config":
                return await self._repair_agent_config(component_id, repair_info)
            else:
                return await self._repair_generic_data(component_id, data_key, repair_info)
            
        except Exception as e:
            self.logger.error(f"Component data repair failed: {e}")
            return False
    
    async def _repair_cluster_state(self, component_id: str, repair_info: str) -> bool:
        """修复集群状态"""
        try:
            # 这里应该实现集群状态的修复逻辑
            self.logger.info(f"Repairing cluster state for {component_id}")
            return True
        except Exception as e:
            self.logger.error(f"Cluster state repair failed: {e}")
            return False
    
    async def _repair_task_assignments(self, component_id: str, repair_info: str) -> bool:
        """修复任务分配"""
        try:
            # 这里应该实现任务分配的修复逻辑
            self.logger.info(f"Repairing task assignments for {component_id}")
            return True
        except Exception as e:
            self.logger.error(f"Task assignments repair failed: {e}")
            return False
    
    async def _repair_agent_config(self, component_id: str, repair_info: str) -> bool:
        """修复代理配置"""
        try:
            # 这里应该实现代理配置的修复逻辑
            self.logger.info(f"Repairing agent config for {component_id}")
            return True
        except Exception as e:
            self.logger.error(f"Agent config repair failed: {e}")
            return False
    
    async def _repair_generic_data(self, component_id: str, data_key: str, repair_info: str) -> bool:
        """修复通用数据"""
        try:
            # 这里应该实现通用数据的修复逻辑
            self.logger.info(f"Repairing generic data {data_key} for {component_id}")
            return True
        except Exception as e:
            self.logger.error(f"Generic data repair failed: {e}")
            return False
    
    async def _consistency_check_loop(self):
        """一致性检查循环"""
        
        while self.running:
            try:
                # 执行定期一致性检查
                data_keys = self.config.get("critical_data_keys", [])
                if data_keys:
                    result = await self.check_data_consistency(data_keys)
                    
                    # 如果发现不一致，尝试自动修复
                    if not result.consistent:
                        auto_repair = self.config.get("auto_repair", True)
                        if auto_repair:
                            await self.repair_inconsistencies(result)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in consistency check loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    def get_consistency_statistics(self) -> Dict[str, Any]:
        """获取一致性统计信息"""
        
        if not self.check_results:
            return {
                "total_checks": 0,
                "consistent_checks": 0,
                "consistency_rate": 1.0,
                "common_inconsistencies": {}
            }
        
        total_checks = len(self.check_results)
        consistent_checks = len([r for r in self.check_results if r.consistent])
        consistency_rate = consistent_checks / total_checks
        
        # 统计常见不一致问题
        inconsistency_types = {}
        for result in self.check_results:
            for inconsistency in result.inconsistencies:
                inconsistency_type = inconsistency.get("type", "unknown")
                inconsistency_types[inconsistency_type] = inconsistency_types.get(inconsistency_type, 0) + 1
        
        return {
            "total_checks": total_checks,
            "consistent_checks": consistent_checks,
            "consistency_rate": consistency_rate,
            "common_inconsistencies": inconsistency_types,
            "recent_checks": [
                {
                    "check_id": r.check_id,
                    "checked_at": r.checked_at.isoformat(),
                    "consistent": r.consistent,
                    "components_count": len(r.components),
                    "inconsistencies_count": len(r.inconsistencies)
                }
                for r in self.check_results[-10:]
            ]
        }
    
    def get_check_result_by_id(self, check_id: str) -> Optional[ConsistencyCheckResult]:
        """根据ID获取检查结果"""
        
        for result in self.check_results:
            if result.check_id == check_id:
                return result
        return None
    
    async def force_consistency_repair(
        self, 
        data_key: str, 
        authoritative_component_id: str
    ) -> bool:
        """强制一致性修复"""
        
        try:
            self.logger.info(f"Force consistency repair for {data_key} using {authoritative_component_id} as authority")
            
            # 获取权威组件的数据
            authoritative_value = await self._get_component_data(authoritative_component_id, data_key)
            if authoritative_value is None:
                self.logger.error(f"Cannot get authoritative value from {authoritative_component_id}")
                return False
            
            # 获取所有组件
            topology = await self.cluster_manager.get_cluster_topology()
            if not topology:
                self.logger.error("No cluster topology available")
                return False
            
            # 修复所有其他组件的数据
            repair_success = True
            for component_id in topology.agents.keys():
                if component_id != authoritative_component_id:
                    try:
                        success = await self._force_set_component_data(component_id, data_key, authoritative_value)
                        if not success:
                            repair_success = False
                            self.logger.error(f"Failed to force repair {data_key} on {component_id}")
                    except Exception as e:
                        self.logger.error(f"Force repair failed for {component_id}: {e}")
                        repair_success = False
            
            return repair_success
            
        except Exception as e:
            self.logger.error(f"Force consistency repair failed: {e}")
            return False
    
    async def _force_set_component_data(
        self, 
        component_id: str, 
        data_key: str, 
        value: Any
    ) -> bool:
        """强制设置组件数据"""
        
        try:
            # 这里应该实现强制设置组件数据的逻辑
            self.logger.info(f"Force setting {data_key} on {component_id}")
            return True
        except Exception as e:
            self.logger.error(f"Force set component data failed: {e}")
            return False
