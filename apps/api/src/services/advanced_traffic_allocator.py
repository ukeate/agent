"""
高级流量分配服务 - 支持多变体、分层实验、条件分配的流量分配算法
"""

import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from enum import Enum
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from dataclasses import dataclass
from src.models.schemas.experiment import TrafficAllocation
from src.services.traffic_splitter import TrafficSplitter
from src.core.security.expression import safe_eval_bool

from src.core.logging import get_logger
logger = get_logger(__name__)

class AllocationStrategy(Enum):
    """流量分配策略"""
    UNIFORM = "uniform"  # 均匀分配
    WEIGHTED = "weighted"  # 按权重分配
    CONDITIONAL = "conditional"  # 条件分配
    STAGED = "staged"  # 阶段性分配
    ADAPTIVE = "adaptive"  # 自适应分配

class AllocationPhase(Enum):
    """分配阶段"""
    RAMP_UP = "ramp_up"  # 爬坡阶段
    FULL_TRAFFIC = "full_traffic"  # 全流量阶段
    RAMP_DOWN = "ramp_down"  # 缩量阶段

@dataclass
class UserContext:
    """用户上下文信息"""
    user_id: str
    user_attributes: Dict[str, Any]
    session_attributes: Dict[str, Any]
    timestamp: datetime
    geo_location: Optional[str] = None
    device_type: Optional[str] = None
    platform: Optional[str] = None

@dataclass
class AllocationRule:
    """分配规则"""
    condition: str  # 条件表达式
    target_variants: List[str]  # 目标变体列表
    allocation_percentages: List[float]  # 对应的分配百分比
    priority: int = 0  # 优先级，数值越高优先级越高
    is_active: bool = True

@dataclass
class StageConfig:
    """阶段配置"""
    stage_id: str
    start_time: datetime
    end_time: datetime
    allocation_percentages: Dict[str, float]  # 变体ID到百分比的映射
    max_users_per_variant: Optional[int] = None
    phase: AllocationPhase = AllocationPhase.FULL_TRAFFIC

class AdvancedTrafficAllocator:
    """高级流量分配器"""
    
    def __init__(self, base_splitter: Optional[TrafficSplitter] = None):
        """
        初始化高级流量分配器
        Args:
            base_splitter: 基础流量分配器实例
        """
        self.base_splitter = base_splitter or TrafficSplitter()
        self._allocation_cache: Dict[str, Dict[str, str]] = {}
        self._stage_configs: Dict[str, List[StageConfig]] = {}
        self._allocation_rules: Dict[str, List[AllocationRule]] = {}
        
    def allocate_with_strategy(self, user_context: UserContext, experiment_id: str,
                             allocations: List[TrafficAllocation],
                             strategy: AllocationStrategy = AllocationStrategy.WEIGHTED,
                             rules: Optional[List[AllocationRule]] = None) -> Optional[str]:
        """
        使用指定策略进行流量分配
        
        Args:
            user_context: 用户上下文
            experiment_id: 实验ID
            allocations: 基础流量分配配置
            strategy: 分配策略
            rules: 分配规则（可选）
            
        Returns:
            分配的变体ID
        """
        try:
            user_id = user_context.user_id
            
            # 检查缓存
            cache_key = f"{experiment_id}:{user_id}"
            if cache_key in self._allocation_cache:
                cached_result = self._allocation_cache[cache_key].get('variant_id')
                if cached_result:
                    logger.debug(f"Using cached allocation for user {user_id}: {cached_result}")
                    return cached_result
            
            # 根据策略选择分配方法
            if strategy == AllocationStrategy.UNIFORM:
                variant_id = self._uniform_allocation(user_context, experiment_id, allocations)
            elif strategy == AllocationStrategy.WEIGHTED:
                variant_id = self._weighted_allocation(user_context, experiment_id, allocations)
            elif strategy == AllocationStrategy.CONDITIONAL:
                variant_id = self._conditional_allocation(user_context, experiment_id, allocations, rules or [])
            elif strategy == AllocationStrategy.STAGED:
                variant_id = self._staged_allocation(user_context, experiment_id, allocations)
            elif strategy == AllocationStrategy.ADAPTIVE:
                variant_id = self._adaptive_allocation(user_context, experiment_id, allocations)
            else:
                # 默认使用基础分配器
                variant_id = self.base_splitter.get_variant(user_id, experiment_id, allocations)
            
            # 缓存结果
            if variant_id:
                self._cache_allocation(cache_key, variant_id, user_context.timestamp)
            
            logger.info(f"Allocated user {user_id} to variant {variant_id} using strategy {strategy.value}")
            return variant_id
            
        except Exception as e:
            logger.error(f"Error in advanced traffic allocation: {str(e)}")
            # 降级到基础分配器
            return self.base_splitter.get_variant(user_context.user_id, experiment_id, allocations)
    
    def _uniform_allocation(self, user_context: UserContext, experiment_id: str,
                          allocations: List[TrafficAllocation]) -> Optional[str]:
        """均匀分配策略"""
        try:
            if not allocations:
                return None
            
            # 确保所有变体均匀分配
            num_variants = len(allocations)
            uniform_percentage = 100.0 / num_variants
            
            # 创建均匀分配配置
            uniform_allocations = []
            for i, allocation in enumerate(allocations):
                uniform_alloc = TrafficAllocation(
                    variant_id=allocation.variant_id,
                    percentage=uniform_percentage
                )
                uniform_allocations.append(uniform_alloc)
            
            return self.base_splitter.get_variant(user_context.user_id, experiment_id, uniform_allocations)
            
        except Exception as e:
            logger.error(f"Error in uniform allocation: {str(e)}")
            return None
    
    def _weighted_allocation(self, user_context: UserContext, experiment_id: str,
                           allocations: List[TrafficAllocation]) -> Optional[str]:
        """加权分配策略（考虑用户属性权重）"""
        try:
            user_attributes = user_context.user_attributes
            modified_allocations = []
            
            for allocation in allocations:
                # 基于用户属性调整权重
                weight_modifier = self._calculate_weight_modifier(user_attributes, allocation.variant_id)
                adjusted_percentage = allocation.percentage * weight_modifier
                
                modified_alloc = TrafficAllocation(
                    variant_id=allocation.variant_id,
                    percentage=adjusted_percentage
                )
                modified_allocations.append(modified_alloc)
            
            # 归一化百分比确保总和为100%
            total_percentage = sum(alloc.percentage for alloc in modified_allocations)
            if total_percentage > 0:
                for alloc in modified_allocations:
                    alloc.percentage = (alloc.percentage / total_percentage) * 100.0
            
            return self.base_splitter.get_variant(user_context.user_id, experiment_id, modified_allocations)
            
        except Exception as e:
            logger.error(f"Error in weighted allocation: {str(e)}")
            return self.base_splitter.get_variant(user_context.user_id, experiment_id, allocations)
    
    def _conditional_allocation(self, user_context: UserContext, experiment_id: str,
                              allocations: List[TrafficAllocation],
                              rules: List[AllocationRule]) -> Optional[str]:
        """条件分配策略"""
        try:
            # 获取实验的分配规则
            experiment_rules = self._allocation_rules.get(experiment_id, []) + rules
            
            # 按优先级排序规则
            sorted_rules = sorted(experiment_rules, key=lambda x: x.priority, reverse=True)
            
            for rule in sorted_rules:
                if not rule.is_active:
                    continue
                
                # 评估条件
                if self._evaluate_condition(rule.condition, user_context):
                    # 创建符合条件的分配配置
                    conditional_allocations = []
                    for i, variant_id in enumerate(rule.target_variants):
                        if i < len(rule.allocation_percentages):
                            conditional_alloc = TrafficAllocation(
                                variant_id=variant_id,
                                percentage=rule.allocation_percentages[i]
                            )
                            conditional_allocations.append(conditional_alloc)
                    
                    if conditional_allocations:
                        result = self.base_splitter.get_variant(
                            user_context.user_id, experiment_id, conditional_allocations
                        )
                        if result:
                            logger.debug(f"Applied conditional rule for user {user_context.user_id}: {rule.condition}")
                            return result
            
            # 如果没有匹配的条件规则，使用默认分配
            return self.base_splitter.get_variant(user_context.user_id, experiment_id, allocations)
            
        except Exception as e:
            logger.error(f"Error in conditional allocation: {str(e)}")
            return self.base_splitter.get_variant(user_context.user_id, experiment_id, allocations)
    
    def _staged_allocation(self, user_context: UserContext, experiment_id: str,
                         allocations: List[TrafficAllocation]) -> Optional[str]:
        """阶段性分配策略"""
        try:
            current_time = user_context.timestamp
            stages = self._stage_configs.get(experiment_id, [])
            
            # 找到当前活跃的阶段
            active_stage = None
            for stage in stages:
                if stage.start_time <= current_time <= stage.end_time:
                    active_stage = stage
                    break
            
            if not active_stage:
                # 没有活跃阶段，使用默认分配
                return self.base_splitter.get_variant(user_context.user_id, experiment_id, allocations)
            
            # 使用阶段配置进行分配
            stage_allocations = []
            for variant_id, percentage in active_stage.allocation_percentages.items():
                stage_alloc = TrafficAllocation(
                    variant_id=variant_id,
                    percentage=percentage
                )
                stage_allocations.append(stage_alloc)
            
            # 检查用户数量限制
            if active_stage.max_users_per_variant:
                if self._check_variant_capacity(experiment_id, stage_allocations, active_stage.max_users_per_variant):
                    return self.base_splitter.get_variant(user_context.user_id, experiment_id, stage_allocations)
                else:
                    logger.warning(f"Variant capacity exceeded in stage {active_stage.stage_id}")
                    return None
            
            return self.base_splitter.get_variant(user_context.user_id, experiment_id, stage_allocations)
            
        except Exception as e:
            logger.error(f"Error in staged allocation: {str(e)}")
            return self.base_splitter.get_variant(user_context.user_id, experiment_id, allocations)
    
    def _adaptive_allocation(self, user_context: UserContext, experiment_id: str,
                           allocations: List[TrafficAllocation]) -> Optional[str]:
        """自适应分配策略（基于实时性能调整）"""
        try:
            # 获取各变体的实时性能指标
            variant_performance = self._get_variant_performance(experiment_id)
            
            if not variant_performance:
                # 没有性能数据，使用默认分配
                return self.base_splitter.get_variant(user_context.user_id, experiment_id, allocations)
            
            # 基于性能调整分配权重
            adaptive_allocations = []
            for allocation in allocations:
                variant_id = allocation.variant_id
                performance_score = variant_performance.get(variant_id, 1.0)
                
                # 根据性能分数调整分配比例
                adjusted_percentage = allocation.percentage * performance_score
                
                adaptive_alloc = TrafficAllocation(
                    variant_id=variant_id,
                    percentage=adjusted_percentage
                )
                adaptive_allocations.append(adaptive_alloc)
            
            # 归一化
            total_percentage = sum(alloc.percentage for alloc in adaptive_allocations)
            if total_percentage > 0:
                for alloc in adaptive_allocations:
                    alloc.percentage = (alloc.percentage / total_percentage) * 100.0
            
            return self.base_splitter.get_variant(user_context.user_id, experiment_id, adaptive_allocations)
            
        except Exception as e:
            logger.error(f"Error in adaptive allocation: {str(e)}")
            return self.base_splitter.get_variant(user_context.user_id, experiment_id, allocations)
    
    def _calculate_weight_modifier(self, user_attributes: Dict[str, Any], variant_id: str) -> float:
        """计算基于用户属性的权重调整系数"""
        try:
            # 基础权重
            weight_modifier = 1.0
            
            # 基于用户类型调整权重
            user_type = user_attributes.get('user_type', 'regular')
            if user_type == 'premium':
                weight_modifier *= 1.2  # 付费用户增加20%权重
            elif user_type == 'new':
                weight_modifier *= 0.8  # 新用户减少20%权重
            
            # 基于地理位置调整权重
            geo_location = user_attributes.get('geo_location')
            if geo_location in ['US', 'EU']:
                weight_modifier *= 1.1  # 主要市场增加权重
            
            # 基于设备类型调整权重
            device_type = user_attributes.get('device_type', 'desktop')
            if device_type == 'mobile':
                weight_modifier *= 1.05  # 移动用户略增权重
            
            return max(0.1, min(2.0, weight_modifier))  # 限制权重范围在0.1-2.0之间
            
        except Exception as e:
            logger.error(f"Error calculating weight modifier: {str(e)}")
            return 1.0
    
    def _evaluate_condition(self, condition: str, user_context: UserContext) -> bool:
        """评估条件表达式"""
        try:
            # 构建评估上下文
            eval_context = {
                'user_id': user_context.user_id,
                'user_attributes': user_context.user_attributes,
                'session_attributes': user_context.session_attributes,
                'geo_location': user_context.geo_location,
                'device_type': user_context.device_type,
                'platform': user_context.platform,
                'timestamp': user_context.timestamp
            }
            
            # 简单的条件评估（生产环境中应使用更安全的表达式引擎）
            # 支持的条件格式例如：user_attributes.get('country') == 'US'
            safe_globals = {
                '__builtins__': {},
                'user_attributes': user_context.user_attributes,
                'session_attributes': user_context.session_attributes,
                'geo_location': user_context.geo_location,
                'device_type': user_context.device_type,
                'platform': user_context.platform,
            }
            return safe_eval_bool(condition, safe_globals)
            
        except Exception as e:
            logger.error(f"Error evaluating condition '{condition}': {str(e)}")
            return False
    
    def _check_variant_capacity(self, experiment_id: str, allocations: List[TrafficAllocation],
                              max_users_per_variant: int) -> bool:
        """检查变体容量限制"""
        try:
            # 这里应该查询数据库获取实际分配计数
            # 为了演示，假设所有变体都未达到上限
            # 实际实现需要查询experiment_assignments表
            return True
            
        except Exception as e:
            logger.error(f"Error checking variant capacity: {str(e)}")
            return False
    
    def _get_variant_performance(self, experiment_id: str) -> Dict[str, float]:
        """获取各变体的实时性能指标"""
        try:
            # 这里应该从监控系统或数据库获取实时性能指标
            # 返回格式：{variant_id: performance_score}
            # performance_score > 1.0 表示性能好，< 1.0 表示性能差
            
            # 示例数据（实际实现需要从监控系统获取）
            performance_data = {
                # 'variant_a': 1.1,  # 性能好，增加分配
                # 'variant_b': 0.9,  # 性能差，减少分配
            }
            
            return performance_data
            
        except Exception as e:
            logger.error(f"Error getting variant performance: {str(e)}")
            return {}
    
    def _cache_allocation(self, cache_key: str, variant_id: str, timestamp: datetime):
        """缓存分配结果"""
        try:
            self._allocation_cache[cache_key] = {
                'variant_id': variant_id,
                'timestamp': timestamp,
                'ttl': timestamp.timestamp() + 3600  # 1小时TTL
            }
            
            # 清理过期缓存
            current_time = utc_now().timestamp()
            expired_keys = [
                key for key, value in self._allocation_cache.items()
                if value.get('ttl', 0) < current_time
            ]
            for key in expired_keys:
                del self._allocation_cache[key]
                
        except Exception as e:
            logger.error(f"Error caching allocation: {str(e)}")
    
    def set_stage_config(self, experiment_id: str, stages: List[StageConfig]):
        """设置实验的阶段配置"""
        try:
            # 验证阶段配置
            for stage in stages:
                if stage.start_time >= stage.end_time:
                    raise ValueError(f"Invalid stage time range: {stage.stage_id}")
                
                total_percentage = sum(stage.allocation_percentages.values())
                if abs(total_percentage - 100.0) > 0.01:
                    raise ValueError(f"Stage {stage.stage_id} allocation doesn't sum to 100%: {total_percentage}%")
            
            self._stage_configs[experiment_id] = stages
            logger.info(f"Set {len(stages)} stage configs for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Error setting stage config: {str(e)}")
            raise
    
    def set_allocation_rules(self, experiment_id: str, rules: List[AllocationRule]):
        """设置实验的分配规则"""
        try:
            # 验证规则
            for rule in rules:
                if len(rule.target_variants) != len(rule.allocation_percentages):
                    raise ValueError(f"Mismatch between variants and percentages in rule")
                
                total_percentage = sum(rule.allocation_percentages)
                if abs(total_percentage - 100.0) > 0.01:
                    raise ValueError(f"Rule allocation doesn't sum to 100%: {total_percentage}%")
            
            self._allocation_rules[experiment_id] = rules
            logger.info(f"Set {len(rules)} allocation rules for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Error setting allocation rules: {str(e)}")
            raise
    
    def get_allocation_analytics(self, experiment_id: str) -> Dict[str, Any]:
        """获取分配分析数据"""
        try:
            analytics = {
                'experiment_id': experiment_id,
                'cache_size': len([k for k in self._allocation_cache.keys() if k.startswith(experiment_id)]),
                'stage_configs': len(self._stage_configs.get(experiment_id, [])),
                'allocation_rules': len(self._allocation_rules.get(experiment_id, [])),
                'cache_hit_rate': 0.0,  # 实际实现需要统计缓存命中率
                'allocation_distribution': {},  # 实际分配分布
                'rule_matching_stats': {}  # 规则匹配统计
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting allocation analytics: {str(e)}")
            return {}
    
    def clear_cache(self, experiment_id: Optional[str] = None):
        """清理缓存"""
        try:
            if experiment_id:
                # 清理特定实验的缓存
                keys_to_remove = [k for k in self._allocation_cache.keys() if k.startswith(experiment_id)]
                for key in keys_to_remove:
                    del self._allocation_cache[key]
                logger.info(f"Cleared cache for experiment {experiment_id}")
            else:
                # 清理所有缓存
                self._allocation_cache.clear()
                logger.info("Cleared all allocation cache")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
