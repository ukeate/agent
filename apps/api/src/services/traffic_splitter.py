"""
A/B测试流量分配服务 - 实现确定性哈希分组算法和流量分配逻辑
"""
import hashlib
import struct
from typing import List, Optional, Dict, Any
import mmh3  # pip install mmh3

from models.schemas.experiment import TrafficAllocation
from core.logging import logger


class TrafficSplitter:
    """流量分配器 - 基于Murmur3哈希算法的确定性用户分组"""
    
    def __init__(self, seed: int = 42):
        """
        初始化流量分配器
        Args:
            seed: 哈希种子，确保可重现性
        """
        self.seed = seed
        self.max_hash_value = 2**32 - 1  # 32位无符号整数最大值
    
    def get_variant(self, user_id: str, experiment_id: str, 
                   allocations: List[TrafficAllocation]) -> Optional[str]:
        """
        为用户分配实验变体
        
        Args:
            user_id: 用户ID
            experiment_id: 实验ID  
            allocations: 流量分配配置
            
        Returns:
            分配的变体ID，如果分配失败返回None
        """
        try:
            # 参数验证
            if not user_id or not experiment_id or not allocations:
                logger.warning("Invalid parameters for variant assignment")
                return None
            
            # 验证流量分配总和
            total_percentage = sum(alloc.percentage for alloc in allocations)
            if abs(total_percentage - 100.0) > 0.01:
                logger.error(f"Traffic allocation doesn't sum to 100%: {total_percentage}")
                return None
            
            # 生成用户的哈希值
            hash_value = self._hash_user_experiment(user_id, experiment_id)
            
            # 将哈希值映射到0-100的范围
            percentage_value = self._hash_to_percentage(hash_value)
            
            # 根据百分比分配变体
            variant_id = self._assign_variant_by_percentage(percentage_value, allocations)
            
            logger.debug(f"User {user_id} in experiment {experiment_id} assigned to variant {variant_id} "
                        f"(hash: {hash_value}, percentage: {percentage_value:.4f})")
            
            return variant_id
            
        except Exception as e:
            logger.error(f"Error assigning variant for user {user_id}: {str(e)}")
            return None
    
    def get_user_bucket(self, user_id: str, experiment_id: str, num_buckets: int = 100) -> int:
        """
        获取用户在实验中的分桶编号
        
        Args:
            user_id: 用户ID
            experiment_id: 实验ID
            num_buckets: 分桶数量
            
        Returns:
            分桶编号 (0 到 num_buckets-1)
        """
        try:
            hash_value = self._hash_user_experiment(user_id, experiment_id)
            bucket = hash_value % num_buckets
            return bucket
            
        except Exception as e:
            logger.error(f"Error getting bucket for user {user_id}: {str(e)}")
            return 0
    
    def is_user_in_percentage(self, user_id: str, experiment_id: str, percentage: float) -> bool:
        """
        检查用户是否在指定的流量百分比内
        
        Args:
            user_id: 用户ID
            experiment_id: 实验ID
            percentage: 流量百分比 (0-100)
            
        Returns:
            True if用户在流量范围内
        """
        try:
            if not 0 <= percentage <= 100:
                logger.warning(f"Invalid percentage: {percentage}")
                return False
            
            hash_value = self._hash_user_experiment(user_id, experiment_id)
            user_percentage = self._hash_to_percentage(hash_value)
            
            return user_percentage < percentage
            
        except Exception as e:
            logger.error(f"Error checking user percentage for {user_id}: {str(e)}")
            return False
    
    def batch_assign_variants(self, user_ids: List[str], experiment_id: str,
                            allocations: List[TrafficAllocation]) -> Dict[str, str]:
        """
        批量分配变体
        
        Args:
            user_ids: 用户ID列表
            experiment_id: 实验ID
            allocations: 流量分配配置
            
        Returns:
            用户ID到变体ID的映射字典
        """
        assignments = {}
        
        try:
            for user_id in user_ids:
                variant_id = self.get_variant(user_id, experiment_id, allocations)
                if variant_id:
                    assignments[user_id] = variant_id
            
            logger.info(f"Batch assigned {len(assignments)} users to variants in experiment {experiment_id}")
            
        except Exception as e:
            logger.error(f"Error in batch assignment: {str(e)}")
        
        return assignments
    
    def analyze_distribution(self, user_ids: List[str], experiment_id: str,
                           allocations: List[TrafficAllocation]) -> Dict[str, Any]:
        """
        分析用户分配的分布情况
        
        Args:
            user_ids: 用户ID列表
            experiment_id: 实验ID
            allocations: 流量分配配置
            
        Returns:
            分布分析结果
        """
        try:
            assignments = self.batch_assign_variants(user_ids, experiment_id, allocations)
            
            # 统计每个变体的分配数量
            variant_counts = {}
            for variant_id in assignments.values():
                variant_counts[variant_id] = variant_counts.get(variant_id, 0) + 1
            
            total_users = len(user_ids)
            
            # 计算实际分配比例
            actual_distribution = {}
            expected_distribution = {}
            
            for allocation in allocations:
                variant_id = allocation.variant_id
                actual_count = variant_counts.get(variant_id, 0)
                actual_percentage = (actual_count / total_users * 100) if total_users > 0 else 0
                
                actual_distribution[variant_id] = {
                    'count': actual_count,
                    'percentage': actual_percentage
                }
                expected_distribution[variant_id] = allocation.percentage
            
            # 计算偏差
            deviations = {}
            for variant_id in expected_distribution:
                expected = expected_distribution[variant_id]
                actual = actual_distribution.get(variant_id, {}).get('percentage', 0)
                deviations[variant_id] = actual - expected
            
            return {
                'total_users': total_users,
                'expected_distribution': expected_distribution,
                'actual_distribution': actual_distribution,
                'deviations': deviations,
                'max_deviation': max(abs(d) for d in deviations.values()) if deviations else 0
            }
            
        except Exception as e:
            logger.error(f"Error analyzing distribution: {str(e)}")
            return {}
    
    def _hash_user_experiment(self, user_id: str, experiment_id: str) -> int:
        """
        使用Murmur3算法为用户和实验组合生成哈希值
        
        Args:
            user_id: 用户ID
            experiment_id: 实验ID
            
        Returns:
            32位无符号整数哈希值
        """
        try:
            # 组合用户ID和实验ID作为输入
            input_string = f"{user_id}#{experiment_id}"
            input_bytes = input_string.encode('utf-8')
            
            # 使用Murmur3_32算法生成哈希值
            hash_value = mmh3.hash(input_bytes, self.seed, signed=False)
            
            return hash_value
            
        except Exception as e:
            logger.error(f"Error generating hash: {str(e)}")
            # 降级到SHA256作为备用
            return self._fallback_hash(user_id, experiment_id)
    
    def _fallback_hash(self, user_id: str, experiment_id: str) -> int:
        """
        备用哈希函数（使用SHA256）
        
        Args:
            user_id: 用户ID
            experiment_id: 实验ID
            
        Returns:
            32位无符号整数哈希值
        """
        input_string = f"{user_id}#{experiment_id}#{self.seed}"
        hash_bytes = hashlib.sha256(input_string.encode('utf-8')).digest()
        # 取前4个字节转换为32位无符号整数
        hash_value = struct.unpack('>I', hash_bytes[:4])[0]
        return hash_value
    
    def _hash_to_percentage(self, hash_value: int) -> float:
        """
        将哈希值映射到0-100的百分比
        
        Args:
            hash_value: 32位无符号整数哈希值
            
        Returns:
            0-100之间的浮点数百分比
        """
        # 将哈希值映射到[0, 1)区间，然后乘以100
        percentage = (hash_value / (self.max_hash_value + 1)) * 100
        return percentage
    
    def _assign_variant_by_percentage(self, percentage: float, 
                                    allocations: List[TrafficAllocation]) -> Optional[str]:
        """
        根据百分比分配变体
        
        Args:
            percentage: 用户的百分比值 (0-100)
            allocations: 流量分配配置
            
        Returns:
            分配的变体ID
        """
        try:
            # 按变体ID排序确保一致性
            sorted_allocations = sorted(allocations, key=lambda x: x.variant_id)
            
            cumulative_percentage = 0.0
            
            for allocation in sorted_allocations:
                cumulative_percentage += allocation.percentage
                
                if percentage < cumulative_percentage:
                    return allocation.variant_id
            
            # 如果由于浮点精度问题没有匹配到任何变体，返回最后一个
            if sorted_allocations:
                logger.warning(f"No variant matched for percentage {percentage}, assigning to last variant")
                return sorted_allocations[-1].variant_id
            
            return None
            
        except Exception as e:
            logger.error(f"Error assigning variant by percentage: {str(e)}")
            return None
    
    def validate_traffic_allocation(self, allocations: List[TrafficAllocation]) -> Dict[str, Any]:
        """
        验证流量分配配置
        
        Args:
            allocations: 流量分配配置
            
        Returns:
            验证结果
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            if not allocations:
                validation_result['is_valid'] = False
                validation_result['errors'].append("No traffic allocations provided")
                return validation_result
            
            # 检查总百分比
            total_percentage = sum(alloc.percentage for alloc in allocations)
            if abs(total_percentage - 100.0) > 0.01:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f"Total percentage is {total_percentage}%, must be 100%")
            
            # 检查重复的变体ID
            variant_ids = [alloc.variant_id for alloc in allocations]
            if len(variant_ids) != len(set(variant_ids)):
                validation_result['is_valid'] = False
                validation_result['errors'].append("Duplicate variant IDs found")
            
            # 检查百分比范围
            for allocation in allocations:
                if allocation.percentage < 0 or allocation.percentage > 100:
                    validation_result['is_valid'] = False
                    validation_result['errors'].append(
                        f"Invalid percentage {allocation.percentage}% for variant {allocation.variant_id}"
                    )
                elif allocation.percentage == 0:
                    validation_result['warnings'].append(
                        f"Zero percentage allocation for variant {allocation.variant_id}"
                    )
            
            # 检查是否有过小的分配（可能导致样本量不足）
            for allocation in allocations:
                if 0 < allocation.percentage < 5:
                    validation_result['warnings'].append(
                        f"Small allocation {allocation.percentage}% for variant {allocation.variant_id} "
                        "may result in insufficient sample size"
                    )
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Validation error: {str(e)}")
        
        return validation_result
    
    def simulate_distribution(self, num_users: int, allocations: List[TrafficAllocation],
                            experiment_id: str = "test_experiment") -> Dict[str, Any]:
        """
        模拟用户分配分布
        
        Args:
            num_users: 模拟用户数量
            allocations: 流量分配配置
            experiment_id: 实验ID
            
        Returns:
            模拟结果
        """
        try:
            # 生成模拟用户ID
            user_ids = [f"user_{i:06d}" for i in range(num_users)]
            
            # 分析分布
            distribution_analysis = self.analyze_distribution(user_ids, experiment_id, allocations)
            
            # 添加统计信息
            distribution_analysis['simulation_info'] = {
                'num_users': num_users,
                'experiment_id': experiment_id,
                'hash_seed': self.seed
            }
            
            return distribution_analysis
            
        except Exception as e:
            logger.error(f"Error in distribution simulation: {str(e)}")
            return {}
    
    def get_hash_uniformity_stats(self, user_ids: List[str], experiment_id: str,
                                num_buckets: int = 100) -> Dict[str, Any]:
        """
        分析哈希函数的均匀性统计
        
        Args:
            user_ids: 用户ID列表
            experiment_id: 实验ID
            num_buckets: 分桶数量
            
        Returns:
            均匀性统计结果
        """
        try:
            bucket_counts = [0] * num_buckets
            
            for user_id in user_ids:
                bucket = self.get_user_bucket(user_id, experiment_id, num_buckets)
                bucket_counts[bucket] += 1
            
            total_users = len(user_ids)
            expected_per_bucket = total_users / num_buckets
            
            # 计算均匀性指标
            variance = sum((count - expected_per_bucket) ** 2 for count in bucket_counts) / num_buckets
            std_dev = variance ** 0.5
            coefficient_of_variation = (std_dev / expected_per_bucket) if expected_per_bucket > 0 else float('inf')
            
            # 卡方检验统计量
            chi_square = sum((count - expected_per_bucket) ** 2 / expected_per_bucket 
                           for count in bucket_counts if expected_per_bucket > 0)
            
            return {
                'total_users': total_users,
                'num_buckets': num_buckets,
                'bucket_counts': bucket_counts,
                'expected_per_bucket': expected_per_bucket,
                'variance': variance,
                'std_dev': std_dev,
                'coefficient_of_variation': coefficient_of_variation,
                'chi_square_statistic': chi_square,
                'min_bucket_size': min(bucket_counts),
                'max_bucket_size': max(bucket_counts)
            }
            
        except Exception as e:
            logger.error(f"Error calculating uniformity stats: {str(e)}")
            return {}