"""
多变体分配服务 - 集成高级流量分配功能到实验管理系统
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from dataclasses import dataclass, asdict
from src.models.schemas.experiment import TrafficAllocation, ExperimentConfig
from src.models.database.experiment import Experiment, ExperimentVariant, ExperimentAssignment
from src.services.advanced_traffic_allocator import (
    AdvancedTrafficAllocator, 
    UserContext, 
    AllocationStrategy,
    AllocationRule,
    StageConfig,
    AllocationPhase
)
from src.services.traffic_splitter import TrafficSplitter
from src.repositories.experiment_repository import ExperimentRepository

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class MultiVariantAllocationRequest:
    """多变体分配请求"""
    user_id: str
    experiment_id: str
    user_attributes: Dict[str, Any]
    session_attributes: Dict[str, Any] = None
    allocation_strategy: AllocationStrategy = AllocationStrategy.WEIGHTED
    force_assignment: bool = False  # 强制重新分配
    respect_previous_assignment: bool = True  # 尊重之前的分配

@dataclass
class MultiVariantAllocationResult:
    """多变体分配结果"""
    user_id: str
    experiment_id: str
    variant_id: Optional[str]
    allocation_strategy: AllocationStrategy
    assignment_timestamp: datetime
    is_cached: bool = False
    allocation_reason: str = ""
    allocation_metadata: Dict[str, Any] = None

class MultiVariantAllocator:
    """多变体分配服务"""
    
    def __init__(self, experiment_repository: ExperimentRepository):
        """
        初始化多变体分配服务
        Args:
            experiment_repository: 实验仓库
        """
        self.experiment_repository = experiment_repository
        self.base_splitter = TrafficSplitter()
        self.advanced_allocator = AdvancedTrafficAllocator(self.base_splitter)
        
        # 分配统计
        self._allocation_stats: Dict[str, Dict[str, int]] = {}
        
    async def allocate_user_to_variants(
        self, 
        request: MultiVariantAllocationRequest
    ) -> MultiVariantAllocationResult:
        """
        为用户分配到实验变体
        
        Args:
            request: 分配请求
            
        Returns:
            分配结果
        """
        try:
            user_id = request.user_id
            experiment_id = request.experiment_id
            
            # 获取实验配置
            experiment = await self.experiment_repository.get_experiment(experiment_id)
            if not experiment:
                logger.error(f"Experiment not found: {experiment_id}")
                return self._create_error_result(request, "Experiment not found")
            
            # 检查实验状态
            if experiment.status != "running":
                logger.info(f"Experiment {experiment_id} is not running (status: {experiment.status})")
                return self._create_error_result(request, f"Experiment not running: {experiment.status}")
            
            # 检查是否已有分配（如果尊重之前的分配）
            if request.respect_previous_assignment and not request.force_assignment:
                existing_assignment = await self.experiment_repository.get_user_assignment(
                    user_id, experiment_id
                )
                if existing_assignment:
                    logger.debug(f"Using existing assignment for user {user_id}: {existing_assignment.variant_id}")
                    return MultiVariantAllocationResult(
                        user_id=user_id,
                        experiment_id=experiment_id,
                        variant_id=existing_assignment.variant_id,
                        allocation_strategy=request.allocation_strategy,
                        assignment_timestamp=existing_assignment.assigned_at,
                        is_cached=True,
                        allocation_reason="Previous assignment",
                        allocation_metadata={"assignment_id": existing_assignment.id}
                    )
            
            # 获取实验变体和流量配置
            variants = await self.experiment_repository.get_experiment_variants(experiment_id)
            if not variants:
                logger.error(f"No variants found for experiment {experiment_id}")
                return self._create_error_result(request, "No variants configured")
            
            # 构建流量分配配置
            traffic_allocations = []
            for variant in variants:
                allocation = TrafficAllocation(
                    variant_id=variant.id,
                    percentage=variant.traffic_percentage or 0.0
                )
                traffic_allocations.append(allocation)
            
            # 验证流量分配
            validation_result = self.base_splitter.validate_traffic_allocation(traffic_allocations)
            if not validation_result['is_valid']:
                logger.error(f"Invalid traffic allocation for experiment {experiment_id}: {validation_result['errors']}")
                return self._create_error_result(request, f"Invalid traffic allocation: {validation_result['errors']}")
            
            # 构建用户上下文
            user_context = UserContext(
                user_id=user_id,
                user_attributes=request.user_attributes,
                session_attributes=request.session_attributes or {},
                timestamp=utc_now(),
                geo_location=request.user_attributes.get('geo_location'),
                device_type=request.user_attributes.get('device_type'),
                platform=request.user_attributes.get('platform')
            )
            
            # 执行高级分配
            variant_id = self.advanced_allocator.allocate_with_strategy(
                user_context=user_context,
                experiment_id=experiment_id,
                allocations=traffic_allocations,
                strategy=request.allocation_strategy
            )
            
            # 记录分配结果
            if variant_id:
                await self._record_assignment(user_id, experiment_id, variant_id, user_context)
                self._update_allocation_stats(experiment_id, variant_id)
                
                allocation_result = MultiVariantAllocationResult(
                    user_id=user_id,
                    experiment_id=experiment_id,
                    variant_id=variant_id,
                    allocation_strategy=request.allocation_strategy,
                    assignment_timestamp=user_context.timestamp,
                    is_cached=False,
                    allocation_reason=f"Allocated using {request.allocation_strategy.value} strategy",
                    allocation_metadata={
                        "user_attributes": request.user_attributes,
                        "traffic_allocations": [asdict(alloc) for alloc in traffic_allocations]
                    }
                )
            else:
                logger.warning(f"Failed to allocate user {user_id} to any variant in experiment {experiment_id}")
                allocation_result = self._create_error_result(request, "Allocation failed")
            
            return allocation_result
            
        except Exception as e:
            logger.error(f"Error in multi-variant allocation: {str(e)}")
            return self._create_error_result(request, f"Internal error: {str(e)}")
    
    async def batch_allocate_users(
        self, 
        requests: List[MultiVariantAllocationRequest]
    ) -> List[MultiVariantAllocationResult]:
        """
        批量分配用户到实验变体
        
        Args:
            requests: 分配请求列表
            
        Returns:
            分配结果列表
        """
        results = []
        
        try:
            # 按实验分组请求以优化处理
            requests_by_experiment = {}
            for request in requests:
                exp_id = request.experiment_id
                if exp_id not in requests_by_experiment:
                    requests_by_experiment[exp_id] = []
                requests_by_experiment[exp_id].append(request)
            
            # 并发处理每个实验的分配
            for experiment_id, exp_requests in requests_by_experiment.items():
                logger.info(f"Processing {len(exp_requests)} allocation requests for experiment {experiment_id}")
                
                for request in exp_requests:
                    result = await self.allocate_user_to_variants(request)
                    results.append(result)
                    
                    # 添加延迟防止过载
                    if len(exp_requests) > 100:
                        await self._batch_processing_delay()
            
            logger.info(f"Completed batch allocation for {len(requests)} requests")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch allocation: {str(e)}")
            # 为失败的请求创建错误结果
            for request in requests:
                if not any(r.user_id == request.user_id and r.experiment_id == request.experiment_id for r in results):
                    results.append(self._create_error_result(request, f"Batch processing error: {str(e)}"))
            return results
    
    async def configure_experiment_allocation_rules(
        self, 
        experiment_id: str, 
        rules: List[AllocationRule]
    ):
        """
        配置实验的分配规则
        
        Args:
            experiment_id: 实验ID
            rules: 分配规则列表
        """
        try:
            self.advanced_allocator.set_allocation_rules(experiment_id, rules)
            
            # 可选：持久化规则到数据库
            # await self.experiment_repository.save_allocation_rules(experiment_id, rules)
            
            logger.info(f"Configured {len(rules)} allocation rules for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Error configuring allocation rules: {str(e)}")
            raise
    
    async def configure_experiment_stages(
        self, 
        experiment_id: str, 
        stages: List[StageConfig]
    ):
        """
        配置实验的阶段设置
        
        Args:
            experiment_id: 实验ID
            stages: 阶段配置列表
        """
        try:
            self.advanced_allocator.set_stage_config(experiment_id, stages)
            
            # 可选：持久化配置到数据库
            # await self.experiment_repository.save_stage_configs(experiment_id, stages)
            
            logger.info(f"Configured {len(stages)} stages for experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Error configuring experiment stages: {str(e)}")
            raise
    
    async def get_allocation_distribution(self, experiment_id: str) -> Dict[str, Any]:
        """
        获取实验的分配分布统计
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            分配分布统计
        """
        try:
            # 从数据库获取实际分配统计
            assignment_stats = await self.experiment_repository.get_assignment_statistics(experiment_id)
            
            # 从分配器获取分析数据
            allocator_analytics = self.advanced_allocator.get_allocation_analytics(experiment_id)
            
            # 合并统计数据
            distribution = {
                'experiment_id': experiment_id,
                'total_assignments': sum(assignment_stats.values()),
                'variant_distribution': assignment_stats,
                'allocator_analytics': allocator_analytics,
                'allocation_stats': self._allocation_stats.get(experiment_id, {}),
                'timestamp': utc_now().isoformat()
            }
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error getting allocation distribution: {str(e)}")
            return {'error': str(e)}
    
    async def simulate_allocation_distribution(
        self, 
        experiment_id: str, 
        num_simulated_users: int = 10000,
        user_attributes_samples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        模拟分配分布
        
        Args:
            experiment_id: 实验ID
            num_simulated_users: 模拟用户数量
            user_attributes_samples: 用户属性样本
            
        Returns:
            模拟结果
        """
        return {
            "experiment_id": experiment_id,
            "simulation_config": {
                "num_simulated_users": num_simulated_users,
                "actual_simulated": 0
            },
            "variant_distribution": {},
            "expected_distribution": {},
            "simulation_timestamp": utc_now().isoformat()
        }
    
    async def _record_assignment(self, user_id: str, experiment_id: str, 
                               variant_id: str, user_context: UserContext):
        """记录用户分配"""
        try:
            assignment = ExperimentAssignment(
                user_id=user_id,
                experiment_id=experiment_id,
                variant_id=variant_id,
                assigned_at=user_context.timestamp,
                assignment_context={
                    'user_attributes': user_context.user_attributes,
                    'session_attributes': user_context.session_attributes,
                    'geo_location': user_context.geo_location,
                    'device_type': user_context.device_type,
                    'platform': user_context.platform
                }
            )
            
            await self.experiment_repository.create_assignment(assignment)
            
        except Exception as e:
            logger.error(f"Error recording assignment: {str(e)}")
            # 不抛出异常，记录失败不应该影响分配结果
    
    def _update_allocation_stats(self, experiment_id: str, variant_id: str):
        """更新分配统计"""
        try:
            if experiment_id not in self._allocation_stats:
                self._allocation_stats[experiment_id] = {}
            
            if variant_id not in self._allocation_stats[experiment_id]:
                self._allocation_stats[experiment_id][variant_id] = 0
            
            self._allocation_stats[experiment_id][variant_id] += 1
            
        except Exception as e:
            logger.error(f"Error updating allocation stats: {str(e)}")
    
    def _create_error_result(self, request: MultiVariantAllocationRequest, 
                           error_reason: str) -> MultiVariantAllocationResult:
        """创建错误结果"""
        return MultiVariantAllocationResult(
            user_id=request.user_id,
            experiment_id=request.experiment_id,
            variant_id=None,
            allocation_strategy=request.allocation_strategy,
            assignment_timestamp=utc_now(),
            is_cached=False,
            allocation_reason=error_reason,
            allocation_metadata={'error': True}
        )
    
    def _generate_sample_user_attributes(self, user_index: int) -> Dict[str, Any]:
        """生成示例用户属性（用于模拟）"""
        import random
        
        user_types = ['new', 'regular', 'premium']
        device_types = ['desktop', 'mobile', 'tablet']
        geo_locations = ['US', 'EU', 'APAC', 'OTHER']
        platforms = ['web', 'ios', 'android']
        
        return {
            'user_type': random.choice(user_types),
            'device_type': random.choice(device_types),
            'geo_location': random.choice(geo_locations),
            'platform': random.choice(platforms),
            'signup_date': (utc_now() - 
                          timedelta(days=random.randint(1, 365))).isoformat(),
            'session_count': random.randint(1, 100),
            'is_premium': random.choice([True, False])
        }
    
    async def _batch_processing_delay(self):
        """批处理延迟"""
        import asyncio
        await asyncio.sleep(0.01)  # 10ms延迟
    
    def get_allocation_stats(self) -> Dict[str, Any]:
        """获取分配统计信息"""
        return {
            'total_experiments': len(self._allocation_stats),
            'allocation_stats': self._allocation_stats,
            'cache_stats': self.advanced_allocator.get_allocation_analytics(''),
            'timestamp': utc_now().isoformat()
        }
    
    def clear_allocation_cache(self, experiment_id: Optional[str] = None):
        """清理分配缓存"""
        try:
            self.advanced_allocator.clear_cache(experiment_id)
            
            if experiment_id and experiment_id in self._allocation_stats:
                del self._allocation_stats[experiment_id]
            elif not experiment_id:
                self._allocation_stats.clear()
                
            logger.info(f"Cleared allocation cache for experiment: {experiment_id or 'all'}")
            
        except Exception as e:
            logger.error(f"Error clearing allocation cache: {str(e)}")
