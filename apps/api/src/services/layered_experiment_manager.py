"""
分层实验管理器 - 实现多实验互斥组管理和分层流量分配
"""
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from enum import Enum
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from dataclasses import dataclass, field
from collections import defaultdict

from models.schemas.experiment import TrafficAllocation
from services.traffic_splitter import TrafficSplitter
from repositories.experiment_repository import ExperimentRepository
from core.logging import logger


class LayerType(Enum):
    """实验层类型"""
    MUTUALLY_EXCLUSIVE = "mutually_exclusive"  # 互斥层
    ORTHOGONAL = "orthogonal"  # 正交层
    HOLDBACK = "holdback"  # 保留层


class ConflictResolution(Enum):
    """冲突解决策略"""
    PRIORITY_BASED = "priority_based"  # 基于优先级
    FIRST_COME_FIRST_SERVE = "first_come_first_serve"  # 先到先得
    ROUND_ROBIN = "round_robin"  # 轮转分配
    RANDOM = "random"  # 随机选择


@dataclass
class ExperimentLayer:
    """实验层定义"""
    layer_id: str
    name: str
    description: str
    layer_type: LayerType
    traffic_percentage: float  # 该层占总流量的百分比
    priority: int = 0  # 层优先级，数值越高优先级越高
    is_active: bool = True
    max_experiments: Optional[int] = None  # 最大实验数量
    conflict_resolution: ConflictResolution = ConflictResolution.PRIORITY_BASED
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: utc_now())


@dataclass
class ExperimentGroup:
    """实验组（互斥组）"""
    group_id: str
    name: str
    layer_id: str
    experiments: List[str] = field(default_factory=list)  # 实验ID列表
    traffic_split: Dict[str, float] = field(default_factory=dict)  # 实验间流量分配
    is_active: bool = True
    priority: int = 0
    max_group_size: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UserLayerAssignment:
    """用户层分配记录"""
    user_id: str
    layer_id: str
    assigned_experiment: Optional[str]
    assignment_reason: str
    assigned_at: datetime
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class LayeredExperimentManager:
    """分层实验管理器"""
    
    def __init__(self, experiment_repository: ExperimentRepository):
        """
        初始化分层实验管理器
        Args:
            experiment_repository: 实验仓库
        """
        self.experiment_repository = experiment_repository
        self.traffic_splitter = TrafficSplitter()
        
        # 内存存储（实际应该使用数据库）
        self._layers: Dict[str, ExperimentLayer] = {}
        self._groups: Dict[str, ExperimentGroup] = {}
        self._user_assignments: Dict[str, Dict[str, UserLayerAssignment]] = {}  # {user_id: {layer_id: assignment}}
        self._experiment_layer_mapping: Dict[str, str] = {}  # {experiment_id: layer_id}
        
        # 统计信息
        self._assignment_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    def create_layer(self, layer: ExperimentLayer) -> ExperimentLayer:
        """
        创建实验层
        
        Args:
            layer: 层定义
            
        Returns:
            创建的层
        """
        try:
            # 验证层配置
            if layer.traffic_percentage <= 0 or layer.traffic_percentage > 100:
                raise ValueError(f"Invalid traffic percentage: {layer.traffic_percentage}")
            
            # 检查层ID唯一性
            if layer.layer_id in self._layers:
                raise ValueError(f"Layer {layer.layer_id} already exists")
            
            # 检查总流量不超过100%
            total_traffic = sum(l.traffic_percentage for l in self._layers.values()) + layer.traffic_percentage
            if total_traffic > 100:
                raise ValueError(f"Total traffic allocation exceeds 100%: {total_traffic}%")
            
            self._layers[layer.layer_id] = layer
            logger.info(f"Created layer {layer.layer_id} with {layer.traffic_percentage}% traffic")
            
            return layer
            
        except Exception as e:
            logger.error(f"Error creating layer {layer.layer_id}: {str(e)}")
            raise
    
    def create_experiment_group(self, group: ExperimentGroup) -> ExperimentGroup:
        """
        创建实验组
        
        Args:
            group: 组定义
            
        Returns:
            创建的组
        """
        try:
            # 验证层存在
            if group.layer_id not in self._layers:
                raise ValueError(f"Layer {group.layer_id} not found")
            
            # 验证组ID唯一性
            if group.group_id in self._groups:
                raise ValueError(f"Group {group.group_id} already exists")
            
            # 验证流量分配
            if group.traffic_split:
                total_split = sum(group.traffic_split.values())
                if abs(total_split - 100.0) > 0.01:
                    raise ValueError(f"Traffic split doesn't sum to 100%: {total_split}%")
                
                # 验证实验存在
                for exp_id in group.traffic_split.keys():
                    if exp_id not in group.experiments:
                        raise ValueError(f"Experiment {exp_id} not in group experiments")
            
            self._groups[group.group_id] = group
            
            # 更新实验到层的映射
            for exp_id in group.experiments:
                self._experiment_layer_mapping[exp_id] = group.layer_id
            
            logger.info(f"Created experiment group {group.group_id} in layer {group.layer_id}")
            
            return group
            
        except Exception as e:
            logger.error(f"Error creating experiment group {group.group_id}: {str(e)}")
            raise
    
    def assign_user_to_layers(self, user_id: str, user_attributes: Dict[str, Any] = None) -> Dict[str, UserLayerAssignment]:
        """
        为用户分配实验层
        
        Args:
            user_id: 用户ID
            user_attributes: 用户属性
            
        Returns:
            用户的层分配结果
        """
        try:
            user_attributes = user_attributes or {}
            current_time = utc_now()
            
            # 检查现有分配
            if user_id in self._user_assignments:
                existing_assignments = self._user_assignments[user_id]
                valid_assignments = {}
                
                for layer_id, assignment in existing_assignments.items():
                    # 检查分配是否过期
                    if assignment.expires_at and assignment.expires_at <= current_time:
                        logger.debug(f"Assignment expired for user {user_id} in layer {layer_id}")
                        continue
                    
                    # 检查层是否仍然活跃
                    if layer_id not in self._layers or not self._layers[layer_id].is_active:
                        logger.debug(f"Layer {layer_id} no longer active for user {user_id}")
                        continue
                    
                    valid_assignments[layer_id] = assignment
                
                if valid_assignments:
                    logger.debug(f"Using existing assignments for user {user_id}")
                    return valid_assignments
            
            # 为用户分配新的层
            new_assignments = {}
            
            # 按优先级排序层
            sorted_layers = sorted(
                [l for l in self._layers.values() if l.is_active],
                key=lambda x: x.priority,
                reverse=True
            )
            
            for layer in sorted_layers:
                assignment = self._assign_user_to_layer(user_id, layer, user_attributes)
                if assignment:
                    new_assignments[layer.layer_id] = assignment
            
            # 缓存分配结果
            self._user_assignments[user_id] = new_assignments
            
            return new_assignments
            
        except Exception as e:
            logger.error(f"Error assigning user {user_id} to layers: {str(e)}")
            return {}
    
    def _assign_user_to_layer(self, user_id: str, layer: ExperimentLayer, 
                            user_attributes: Dict[str, Any]) -> Optional[UserLayerAssignment]:
        """
        为用户分配单个层
        
        Args:
            user_id: 用户ID
            layer: 层定义
            user_attributes: 用户属性
            
        Returns:
            层分配结果
        """
        try:
            # 检查用户是否在该层的流量范围内
            if not self.traffic_splitter.is_user_in_percentage(
                user_id, 
                f"layer_{layer.layer_id}", 
                layer.traffic_percentage
            ):
                logger.debug(f"User {user_id} not in traffic range for layer {layer.layer_id}")
                return None
            
            # 获取该层的实验组
            layer_groups = [g for g in self._groups.values() if g.layer_id == layer.layer_id and g.is_active]
            
            if not layer_groups:
                # 层中没有活跃的组，分配到层但不分配实验
                assignment = UserLayerAssignment(
                    user_id=user_id,
                    layer_id=layer.layer_id,
                    assigned_experiment=None,
                    assignment_reason="Layer assigned but no active groups",
                    assigned_at=utc_now()
                )
                return assignment
            
            # 根据层类型处理分配
            if layer.layer_type == LayerType.MUTUALLY_EXCLUSIVE:
                return self._assign_mutually_exclusive_layer(user_id, layer, layer_groups, user_attributes)
            elif layer.layer_type == LayerType.ORTHOGONAL:
                return self._assign_orthogonal_layer(user_id, layer, layer_groups, user_attributes)
            elif layer.layer_type == LayerType.HOLDBACK:
                return self._assign_holdback_layer(user_id, layer, layer_groups, user_attributes)
            else:
                logger.warning(f"Unknown layer type: {layer.layer_type}")
                return None
                
        except Exception as e:
            logger.error(f"Error assigning user {user_id} to layer {layer.layer_id}: {str(e)}")
            return None
    
    def _assign_mutually_exclusive_layer(self, user_id: str, layer: ExperimentLayer,
                                       groups: List[ExperimentGroup], 
                                       user_attributes: Dict[str, Any]) -> Optional[UserLayerAssignment]:
        """分配互斥层"""
        try:
            # 在互斥层中，用户只能参与一个实验组中的一个实验
            
            # 按优先级排序组
            sorted_groups = sorted(groups, key=lambda x: x.priority, reverse=True)
            
            for group in sorted_groups:
                # 检查组是否有容量
                if group.max_group_size:
                    current_group_size = self._get_group_assignment_count(group.group_id)
                    if current_group_size >= group.max_group_size:
                        logger.debug(f"Group {group.group_id} at capacity")
                        continue
                
                # 在组内分配实验
                if group.traffic_split:
                    # 使用配置的流量分割
                    traffic_allocations = [
                        TrafficAllocation(variant_id=exp_id, percentage=percentage)
                        for exp_id, percentage in group.traffic_split.items()
                    ]
                else:
                    # 均匀分配
                    if group.experiments:
                        percentage_per_exp = 100.0 / len(group.experiments)
                        traffic_allocations = [
                            TrafficAllocation(variant_id=exp_id, percentage=percentage_per_exp)
                            for exp_id in group.experiments
                        ]
                    else:
                        continue
                
                assigned_experiment = self.traffic_splitter.get_variant(
                    user_id, 
                    f"group_{group.group_id}", 
                    traffic_allocations
                )
                
                if assigned_experiment:
                    assignment = UserLayerAssignment(
                        user_id=user_id,
                        layer_id=layer.layer_id,
                        assigned_experiment=assigned_experiment,
                        assignment_reason=f"Mutually exclusive assignment in group {group.group_id}",
                        assigned_at=utc_now(),
                        metadata={"group_id": group.group_id, "layer_type": layer.layer_type.value}
                    )
                    
                    # 更新统计
                    self._assignment_stats[layer.layer_id][assigned_experiment] += 1
                    
                    return assignment
            
            # 如果没有分配到任何实验，仍然分配到层
            return UserLayerAssignment(
                user_id=user_id,
                layer_id=layer.layer_id,
                assigned_experiment=None,
                assignment_reason="In layer but no experiment assigned",
                assigned_at=utc_now(),
                metadata={"layer_type": layer.layer_type.value}
            )
            
        except Exception as e:
            logger.error(f"Error in mutually exclusive assignment: {str(e)}")
            return None
    
    def _assign_orthogonal_layer(self, user_id: str, layer: ExperimentLayer,
                               groups: List[ExperimentGroup], 
                               user_attributes: Dict[str, Any]) -> Optional[UserLayerAssignment]:
        """分配正交层"""
        try:
            # 在正交层中，用户可以参与多个实验组
            assigned_experiments = []
            group_assignments = []
            
            for group in groups:
                # 检查用户是否符合该组的条件
                if group.max_group_size:
                    current_group_size = self._get_group_assignment_count(group.group_id)
                    if current_group_size >= group.max_group_size:
                        continue
                
                # 为该组分配实验
                if group.traffic_split:
                    traffic_allocations = [
                        TrafficAllocation(variant_id=exp_id, percentage=percentage)
                        for exp_id, percentage in group.traffic_split.items()
                    ]
                else:
                    if group.experiments:
                        percentage_per_exp = 100.0 / len(group.experiments)
                        traffic_allocations = [
                            TrafficAllocation(variant_id=exp_id, percentage=percentage_per_exp)
                            for exp_id in group.experiments
                        ]
                    else:
                        continue
                
                assigned_experiment = self.traffic_splitter.get_variant(
                    user_id, 
                    f"orthogonal_group_{group.group_id}", 
                    traffic_allocations
                )
                
                if assigned_experiment:
                    assigned_experiments.append(assigned_experiment)
                    group_assignments.append(group.group_id)
                    self._assignment_stats[layer.layer_id][assigned_experiment] += 1
            
            # 创建分配记录（正交层可能包含多个实验）
            assignment = UserLayerAssignment(
                user_id=user_id,
                layer_id=layer.layer_id,
                assigned_experiment=assigned_experiments[0] if assigned_experiments else None,
                assignment_reason=f"Orthogonal assignment in {len(group_assignments)} groups",
                assigned_at=utc_now(),
                metadata={
                    "layer_type": layer.layer_type.value,
                    "assigned_experiments": assigned_experiments,
                    "group_assignments": group_assignments
                }
            )
            
            return assignment
            
        except Exception as e:
            logger.error(f"Error in orthogonal assignment: {str(e)}")
            return None
    
    def _assign_holdback_layer(self, user_id: str, layer: ExperimentLayer,
                             groups: List[ExperimentGroup], 
                             user_attributes: Dict[str, Any]) -> Optional[UserLayerAssignment]:
        """分配保留层"""
        try:
            # 保留层用于对照组，用户通常不参与实验
            # 但可以用于收集基线数据
            
            assignment = UserLayerAssignment(
                user_id=user_id,
                layer_id=layer.layer_id,
                assigned_experiment=None,
                assignment_reason="Holdback layer - control group",
                assigned_at=utc_now(),
                metadata={
                    "layer_type": layer.layer_type.value,
                    "is_holdback": True
                }
            )
            
            # 更新保留层统计
            self._assignment_stats[layer.layer_id]["holdback"] += 1
            
            return assignment
            
        except Exception as e:
            logger.error(f"Error in holdback assignment: {str(e)}")
            return None
    
    def _get_group_assignment_count(self, group_id: str) -> int:
        """获取组的当前分配数量"""
        try:
            # 统计该组中所有实验的分配数量
            group = self._groups.get(group_id)
            if not group:
                return 0
            
            layer_id = group.layer_id
            if layer_id not in self._assignment_stats:
                return 0
            
            total_count = 0
            for exp_id in group.experiments:
                total_count += self._assignment_stats[layer_id].get(exp_id, 0)
            
            return total_count
            
        except Exception as e:
            logger.error(f"Error getting group assignment count: {str(e)}")
            return 0
    
    def get_user_experiments(self, user_id: str) -> List[Dict[str, Any]]:
        """
        获取用户参与的所有实验
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户实验列表
        """
        try:
            assignments = self.assign_user_to_layers(user_id)
            user_experiments = []
            
            for layer_id, assignment in assignments.items():
                layer = self._layers.get(layer_id)
                if not layer:
                    continue
                
                if assignment.assigned_experiment:
                    # 单个实验分配
                    user_experiments.append({
                        "experiment_id": assignment.assigned_experiment,
                        "layer_id": layer_id,
                        "layer_name": layer.name,
                        "layer_type": layer.layer_type.value,
                        "assignment_reason": assignment.assignment_reason,
                        "assigned_at": assignment.assigned_at
                    })
                
                # 检查正交层的多实验分配
                if assignment.metadata.get("assigned_experiments"):
                    for exp_id in assignment.metadata["assigned_experiments"]:
                        if exp_id != assignment.assigned_experiment:  # 避免重复
                            user_experiments.append({
                                "experiment_id": exp_id,
                                "layer_id": layer_id,
                                "layer_name": layer.name,
                                "layer_type": layer.layer_type.value,
                                "assignment_reason": "Orthogonal layer assignment",
                                "assigned_at": assignment.assigned_at
                            })
            
            return user_experiments
            
        except Exception as e:
            logger.error(f"Error getting user experiments for {user_id}: {str(e)}")
            return []
    
    def check_experiment_conflicts(self, user_id: str, new_experiment_id: str) -> Dict[str, Any]:
        """
        检查新实验与用户现有实验的冲突
        
        Args:
            user_id: 用户ID
            new_experiment_id: 新实验ID
            
        Returns:
            冲突检查结果
        """
        try:
            # 获取新实验所在的层
            new_experiment_layer = self._experiment_layer_mapping.get(new_experiment_id)
            if not new_experiment_layer:
                return {
                    "has_conflict": False,
                    "reason": "Experiment not assigned to any layer"
                }
            
            # 获取用户现有分配
            user_assignments = self._user_assignments.get(user_id, {})
            
            conflicts = []
            for layer_id, assignment in user_assignments.items():
                layer = self._layers.get(layer_id)
                if not layer:
                    continue
                
                # 检查互斥层冲突
                if (layer.layer_type == LayerType.MUTUALLY_EXCLUSIVE and 
                    layer_id == new_experiment_layer and 
                    assignment.assigned_experiment):
                    conflicts.append({
                        "type": "mutually_exclusive",
                        "layer_id": layer_id,
                        "existing_experiment": assignment.assigned_experiment,
                        "message": f"User already assigned to experiment in exclusive layer {layer_id}"
                    })
                
                # 检查流量重叠冲突
                if layer_id == new_experiment_layer and assignment.assigned_experiment == new_experiment_id:
                    conflicts.append({
                        "type": "duplicate_assignment",
                        "layer_id": layer_id,
                        "message": f"User already assigned to this experiment"
                    })
            
            return {
                "has_conflict": len(conflicts) > 0,
                "conflicts": conflicts,
                "can_assign": len(conflicts) == 0
            }
            
        except Exception as e:
            logger.error(f"Error checking experiment conflicts: {str(e)}")
            return {
                "has_conflict": True,
                "reason": f"Error during conflict check: {str(e)}"
            }
    
    def get_layer_statistics(self, layer_id: str) -> Dict[str, Any]:
        """
        获取层的统计信息
        
        Args:
            layer_id: 层ID
            
        Returns:
            层统计信息
        """
        try:
            layer = self._layers.get(layer_id)
            if not layer:
                return {"error": "Layer not found"}
            
            # 获取层中的组
            layer_groups = [g for g in self._groups.values() if g.layer_id == layer_id]
            
            # 统计分配数据
            assignment_stats = self._assignment_stats.get(layer_id, {})
            total_assignments = sum(assignment_stats.values())
            
            # 计算流量利用率
            total_users_in_layer = len([
                assignment for user_assignments in self._user_assignments.values()
                for assignment in user_assignments.values()
                if assignment.layer_id == layer_id
            ])
            
            statistics = {
                "layer_id": layer_id,
                "layer_name": layer.name,
                "layer_type": layer.layer_type.value,
                "traffic_percentage": layer.traffic_percentage,
                "is_active": layer.is_active,
                "total_groups": len(layer_groups),
                "total_assignments": total_assignments,
                "total_users_in_layer": total_users_in_layer,
                "experiment_distribution": assignment_stats,
                "groups": [
                    {
                        "group_id": g.group_id,
                        "name": g.name,
                        "experiment_count": len(g.experiments),
                        "assignments": sum(assignment_stats.get(exp_id, 0) for exp_id in g.experiments)
                    }
                    for g in layer_groups
                ]
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"Error getting layer statistics: {str(e)}")
            return {"error": str(e)}
    
    def get_all_layer_overview(self) -> Dict[str, Any]:
        """
        获取所有层的概览
        
        Returns:
            所有层的概览信息
        """
        try:
            overview = {
                "total_layers": len(self._layers),
                "total_groups": len(self._groups),
                "total_assigned_users": len(self._user_assignments),
                "layers": []
            }
            
            for layer_id, layer in self._layers.items():
                layer_stats = self.get_layer_statistics(layer_id)
                overview["layers"].append(layer_stats)
            
            # 计算总流量分配
            total_traffic = sum(l.traffic_percentage for l in self._layers.values() if l.is_active)
            overview["total_traffic_allocated"] = total_traffic
            overview["remaining_traffic"] = 100.0 - total_traffic
            
            return overview
            
        except Exception as e:
            logger.error(f"Error getting layer overview: {str(e)}")
            return {"error": str(e)}
    
    def clear_user_assignments(self, user_id: Optional[str] = None, layer_id: Optional[str] = None):
        """
        清理用户分配
        
        Args:
            user_id: 用户ID，为None则清理所有用户
            layer_id: 层ID，为None则清理所有层
        """
        try:
            if user_id and layer_id:
                # 清理特定用户在特定层的分配
                if user_id in self._user_assignments and layer_id in self._user_assignments[user_id]:
                    del self._user_assignments[user_id][layer_id]
                    logger.info(f"Cleared assignment for user {user_id} in layer {layer_id}")
            elif user_id:
                # 清理特定用户的所有分配
                if user_id in self._user_assignments:
                    del self._user_assignments[user_id]
                    logger.info(f"Cleared all assignments for user {user_id}")
            elif layer_id:
                # 清理特定层的所有用户分配
                for uid in list(self._user_assignments.keys()):
                    if layer_id in self._user_assignments[uid]:
                        del self._user_assignments[uid][layer_id]
                        if not self._user_assignments[uid]:
                            del self._user_assignments[uid]
                logger.info(f"Cleared all assignments for layer {layer_id}")
            else:
                # 清理所有分配
                self._user_assignments.clear()
                self._assignment_stats.clear()
                logger.info("Cleared all user assignments")
                
        except Exception as e:
            logger.error(f"Error clearing user assignments: {str(e)}")
    
    def validate_layer_configuration(self) -> Dict[str, Any]:
        """
        验证层配置的完整性
        
        Returns:
            验证结果
        """
        try:
            validation_result = {
                "is_valid": True,
                "errors": [],
                "warnings": []
            }
            
            # 检查总流量分配
            total_traffic = sum(l.traffic_percentage for l in self._layers.values() if l.is_active)
            if total_traffic > 100:
                validation_result["is_valid"] = False
                validation_result["errors"].append(f"Total traffic allocation exceeds 100%: {total_traffic}%")
            elif total_traffic < 95:
                validation_result["warnings"].append(f"Traffic allocation is low: {total_traffic}%")
            
            # 检查每个层的配置
            for layer_id, layer in self._layers.items():
                layer_groups = [g for g in self._groups.values() if g.layer_id == layer_id]
                
                if layer.is_active and not layer_groups:
                    validation_result["warnings"].append(f"Active layer {layer_id} has no experiment groups")
                
                # 检查互斥层是否有重叠实验
                if layer.layer_type == LayerType.MUTUALLY_EXCLUSIVE:
                    all_experiments = []
                    for group in layer_groups:
                        all_experiments.extend(group.experiments)
                    
                    if len(all_experiments) != len(set(all_experiments)):
                        validation_result["is_valid"] = False
                        validation_result["errors"].append(f"Mutually exclusive layer {layer_id} has overlapping experiments")
                
                # 检查组的流量分配
                for group in layer_groups:
                    if group.traffic_split:
                        total_split = sum(group.traffic_split.values())
                        if abs(total_split - 100.0) > 0.01:
                            validation_result["is_valid"] = False
                            validation_result["errors"].append(f"Group {group.group_id} traffic split doesn't sum to 100%: {total_split}%")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating layer configuration: {str(e)}")
            return {
                "is_valid": False,
                "errors": [f"Validation error: {str(e)}"]
            }