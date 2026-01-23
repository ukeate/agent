"""
统计分析服务 - A/B测试实验数据统计计算
"""

from typing import List, Dict, Optional, Union, Any, Tuple
import math
import statistics
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from dataclasses import dataclass
from enum import Enum
from src.core.logging import get_logger

logger = get_logger(__name__)

class MetricType(str, Enum):
    """指标类型"""
    CONVERSION = "conversion"  # 转化率（二元指标）
    CONTINUOUS = "continuous"  # 连续指标（收入、时长等）
    COUNT = "count"  # 计数指标（页面浏览量等）
    RATIO = "ratio"  # 比率指标（CTR等）

class DistributionType(str, Enum):
    """分布类型"""
    NORMAL = "normal"  # 正态分布
    BINOMIAL = "binomial"  # 二项分布
    POISSON = "poisson"  # 泊松分布
    UNKNOWN = "unknown"  # 未知分布

@dataclass
class DescriptiveStats:
    """描述性统计结果"""
    count: int  # 样本数量
    mean: float  # 均值
    variance: float  # 方差
    std_dev: float  # 标准差
    min_value: float  # 最小值
    max_value: float  # 最大值
    median: float  # 中位数
    q25: float  # 25分位数
    q75: float  # 75分位数
    skewness: Optional[float] = None  # 偏度
    kurtosis: Optional[float] = None  # 峰度
    sum_value: Optional[float] = None  # 总和
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "count": self.count,
            "mean": self.mean,
            "variance": self.variance,
            "std_dev": self.std_dev,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "median": self.median,
            "q25": self.q25,
            "q75": self.q75,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "sum_value": self.sum_value
        }

@dataclass
class GroupStats:
    """分组统计结果"""
    group_id: str  # 分组ID
    group_name: str  # 分组名称
    stats: DescriptiveStats  # 描述性统计
    metric_type: MetricType  # 指标类型
    distribution_type: DistributionType = DistributionType.UNKNOWN
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "stats": self.stats.to_dict(),
            "metric_type": self.metric_type.value,
            "distribution_type": self.distribution_type.value
        }

class BasicStatisticsCalculator:
    """基础统计计算器"""
    
    def __init__(self):
        self.logger = logger
        self._epsilon = 1e-10  # 用于避免除零错误的小值
    
    def calculate_mean(self, values: List[Union[int, float]]) -> float:
        """计算均值"""
        if not values:
            raise ValueError("Values list cannot be empty")
        
        try:
            return sum(values) / len(values)
        except Exception as e:
            self.logger.error(f"Failed to calculate mean: {e}")
            raise
    
    def calculate_variance(self, values: List[Union[int, float]], 
                         sample: bool = True) -> float:
        """
        计算方差
        
        Args:
            values: 数值列表
            sample: 是否为样本方差（使用n-1作为分母）
        
        Returns:
            方差值
        """
        if not values:
            raise ValueError("Values list cannot be empty")
        
        if len(values) == 1:
            return 0.0
        
        try:
            mean = self.calculate_mean(values)
            squared_diffs = [(x - mean) ** 2 for x in values]
            divisor = len(values) - 1 if sample else len(values)
            
            if divisor <= 0:
                return 0.0
                
            return sum(squared_diffs) / divisor
        except Exception as e:
            self.logger.error(f"Failed to calculate variance: {e}")
            raise
    
    def calculate_std_deviation(self, values: List[Union[int, float]], 
                              sample: bool = True) -> float:
        """
        计算标准差
        
        Args:
            values: 数值列表
            sample: 是否为样本标准差
        
        Returns:
            标准差值
        """
        try:
            variance = self.calculate_variance(values, sample)
            return math.sqrt(variance)
        except Exception as e:
            self.logger.error(f"Failed to calculate standard deviation: {e}")
            raise
    
    def calculate_percentiles(self, values: List[Union[int, float]], 
                            percentiles: List[float]) -> List[float]:
        """
        计算分位数
        
        Args:
            values: 数值列表
            percentiles: 分位数列表（0-100）
        
        Returns:
            对应的分位数值列表
        """
        if not values:
            raise ValueError("Values list cannot be empty")
        
        try:
            sorted_values = sorted(values)
            results = []
            
            for p in percentiles:
                if not (0 <= p <= 100):
                    raise ValueError(f"Percentile must be between 0 and 100, got {p}")
                
                # 使用线性插值方法计算分位数
                n = len(sorted_values)
                if p == 100:
                    results.append(sorted_values[-1])
                elif p == 0:
                    results.append(sorted_values[0])
                else:
                    index = (p / 100) * (n - 1)
                    lower_index = int(index)
                    upper_index = min(lower_index + 1, n - 1)
                    weight = index - lower_index
                    
                    if lower_index == upper_index:
                        results.append(sorted_values[lower_index])
                    else:
                        interpolated = (sorted_values[lower_index] * (1 - weight) + 
                                      sorted_values[upper_index] * weight)
                        results.append(interpolated)
            
            return results
        except Exception as e:
            self.logger.error(f"Failed to calculate percentiles: {e}")
            raise
    
    def calculate_skewness(self, values: List[Union[int, float]]) -> Optional[float]:
        """
        计算偏度（分布的不对称性）
        
        Args:
            values: 数值列表
        
        Returns:
            偏度值，如果计算失败返回None
        """
        try:
            if len(values) < 3:
                return None
            
            mean = self.calculate_mean(values)
            std_dev = self.calculate_std_deviation(values)
            
            if std_dev < self._epsilon:
                return None
            
            n = len(values)
            skewness = sum(((x - mean) / std_dev) ** 3 for x in values) / n
            
            # 应用样本偏度的偏差校正
            if n > 2:
                skewness = skewness * math.sqrt(n * (n - 1)) / (n - 2)
            
            return skewness
        except Exception as e:
            self.logger.warning(f"Failed to calculate skewness: {e}")
            return None
    
    def calculate_kurtosis(self, values: List[Union[int, float]]) -> Optional[float]:
        """
        计算峰度（分布的尖锐程度）
        
        Args:
            values: 数值列表
        
        Returns:
            峰度值（超额峰度），如果计算失败返回None
        """
        try:
            if len(values) < 4:
                return None
            
            mean = self.calculate_mean(values)
            std_dev = self.calculate_std_deviation(values)
            
            if std_dev < self._epsilon:
                return None
            
            n = len(values)
            kurtosis = sum(((x - mean) / std_dev) ** 4 for x in values) / n
            
            # 应用样本峰度的偏差校正并转换为超额峰度
            if n > 3:
                kurtosis = ((n - 1) * ((n + 1) * kurtosis - 3 * (n - 1)) / 
                           ((n - 2) * (n - 3)))
            else:
                kurtosis = kurtosis - 3  # 转换为超额峰度
            
            return kurtosis
        except Exception as e:
            self.logger.warning(f"Failed to calculate kurtosis: {e}")
            return None
    
    def calculate_descriptive_stats(self, values: List[Union[int, float]],
                                  calculate_advanced: bool = True) -> DescriptiveStats:
        """
        计算完整的描述性统计
        
        Args:
            values: 数值列表
            calculate_advanced: 是否计算高级统计指标（偏度、峰度）
        
        Returns:
            描述性统计结果
        """
        if not values:
            raise ValueError("Values list cannot be empty")
        
        try:
            # 基础统计
            count = len(values)
            mean = self.calculate_mean(values)
            variance = self.calculate_variance(values)
            std_dev = self.calculate_std_deviation(values)
            min_value = min(values)
            max_value = max(values)
            sum_value = sum(values)
            
            # 分位数
            percentiles = self.calculate_percentiles(values, [25, 50, 75])
            q25, median, q75 = percentiles[0], percentiles[1], percentiles[2]
            
            # 高级统计（可选）
            skewness = None
            kurtosis = None
            if calculate_advanced:
                skewness = self.calculate_skewness(values)
                kurtosis = self.calculate_kurtosis(values)
            
            return DescriptiveStats(
                count=count,
                mean=mean,
                variance=variance,
                std_dev=std_dev,
                min_value=min_value,
                max_value=max_value,
                median=median,
                q25=q25,
                q75=q75,
                skewness=skewness,
                kurtosis=kurtosis,
                sum_value=sum_value
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate descriptive statistics: {e}")
            raise
    
    def calculate_conversion_rate_stats(self, conversions: int, 
                                      total_users: int) -> DescriptiveStats:
        """
        计算转化率统计（针对二元指标）
        
        Args:
            conversions: 转化用户数
            total_users: 总用户数
        
        Returns:
            转化率描述性统计
        """
        if total_users <= 0:
            raise ValueError("Total users must be greater than 0")
        
        if conversions < 0 or conversions > total_users:
            raise ValueError("Conversions must be between 0 and total_users")
        
        try:
            conversion_rate = conversions / total_users
            
            # 对于二项分布，方差 = p * (1 - p) / n
            variance = conversion_rate * (1 - conversion_rate) / total_users
            std_dev = math.sqrt(variance)
            
            return DescriptiveStats(
                count=total_users,
                mean=conversion_rate,
                variance=variance,
                std_dev=std_dev,
                min_value=0.0,
                max_value=1.0,
                median=conversion_rate,  # 对于转化率，中位数近似等于均值
                q25=max(0.0, conversion_rate - 0.674 * std_dev),
                q75=min(1.0, conversion_rate + 0.674 * std_dev),
                sum_value=conversions
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate conversion rate stats: {e}")
            raise

class ExperimentStatsCalculator:
    """实验统计计算器 - 整合实验数据的统计分析"""
    
    def __init__(self):
        self.basic_calculator = BasicStatisticsCalculator()
        self.logger = logger
    
    def calculate_group_stats(self, group_id: str, group_name: str,
                            values: List[Union[int, float]],
                            metric_type: MetricType) -> GroupStats:
        """
        计算分组统计
        
        Args:
            group_id: 分组ID
            group_name: 分组名称
            values: 指标值列表
            metric_type: 指标类型
        
        Returns:
            分组统计结果
        """
        try:
            stats = self.basic_calculator.calculate_descriptive_stats(values)
            
            # 根据指标类型推断分布类型
            distribution_type = self._infer_distribution_type(values, metric_type)
            
            return GroupStats(
                group_id=group_id,
                group_name=group_name,
                stats=stats,
                metric_type=metric_type,
                distribution_type=distribution_type
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate group stats for {group_id}: {e}")
            raise
    
    def calculate_conversion_group_stats(self, group_id: str, group_name: str,
                                       conversions: int, 
                                       total_users: int) -> GroupStats:
        """
        计算转化率分组统计
        
        Args:
            group_id: 分组ID
            group_name: 分组名称
            conversions: 转化用户数
            total_users: 总用户数
        
        Returns:
            转化率分组统计结果
        """
        try:
            stats = self.basic_calculator.calculate_conversion_rate_stats(
                conversions, total_users
            )
            
            return GroupStats(
                group_id=group_id,
                group_name=group_name,
                stats=stats,
                metric_type=MetricType.CONVERSION,
                distribution_type=DistributionType.BINOMIAL
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate conversion group stats for {group_id}: {e}")
            raise
    
    def calculate_multiple_groups_stats(self, 
                                      groups_data: Dict[str, Dict[str, Any]],
                                      metric_type: MetricType) -> Dict[str, GroupStats]:
        """
        计算多个分组的统计
        
        Args:
            groups_data: 分组数据，格式：{group_id: {"name": str, "values": List}}
            metric_type: 指标类型
        
        Returns:
            各分组统计结果字典
        """
        results = {}
        
        for group_id, group_info in groups_data.items():
            try:
                if metric_type == MetricType.CONVERSION:
                    # 对于转化率指标，需要conversions和total_users
                    if "conversions" in group_info and "total_users" in group_info:
                        stats = self.calculate_conversion_group_stats(
                            group_id, 
                            group_info["name"],
                            group_info["conversions"],
                            group_info["total_users"]
                        )
                    else:
                        raise ValueError(f"Conversion metric requires 'conversions' and 'total_users' for group {group_id}")
                else:
                    # 对于其他指标类型，使用values列表
                    stats = self.calculate_group_stats(
                        group_id,
                        group_info["name"], 
                        group_info["values"],
                        metric_type
                    )
                
                results[group_id] = stats
                
            except Exception as e:
                self.logger.error(f"Failed to calculate stats for group {group_id}: {e}")
                # 继续处理其他分组，不因单个分组失败而停止
                continue
        
        return results
    
    def _infer_distribution_type(self, values: List[Union[int, float]], 
                               metric_type: MetricType) -> DistributionType:
        """
        根据数据特征推断分布类型
        
        Args:
            values: 数值列表
            metric_type: 指标类型
        
        Returns:
            推断的分布类型
        """
        try:
            # 基于指标类型的简单推断
            if metric_type == MetricType.CONVERSION:
                return DistributionType.BINOMIAL
            elif metric_type == MetricType.COUNT:
                # 检查是否为非负整数（泊松分布特征）
                if all(isinstance(v, int) and v >= 0 for v in values):
                    mean = sum(values) / len(values)
                    variance = self.basic_calculator.calculate_variance(values)
                    # 泊松分布的均值和方差相等
                    if abs(mean - variance) / max(mean, 1) < 0.3:
                        return DistributionType.POISSON
            
            # 对于连续指标，检查是否接近正态分布
            if metric_type in [MetricType.CONTINUOUS, MetricType.RATIO]:
                if len(values) >= 30:  # 有足够样本进行分布检验
                    skewness = self.basic_calculator.calculate_skewness(values)
                    kurtosis = self.basic_calculator.calculate_kurtosis(values)
                    
                    # 简单的正态分布判断（偏度接近0，峰度接近0）
                    if (skewness is not None and abs(skewness) < 1.0 and
                        kurtosis is not None and abs(kurtosis) < 3.0):
                        return DistributionType.NORMAL
            
            return DistributionType.UNKNOWN
            
        except Exception as e:
            self.logger.warning(f"Failed to infer distribution type: {e}")
            return DistributionType.UNKNOWN

# 统计分析服务类
class StatisticalAnalysisService:
    """统计分析服务 - 提供完整的统计分析功能"""
    
    def __init__(self):
        self.basic_calculator = BasicStatisticsCalculator()
        self.experiment_calculator = ExperimentStatsCalculator()
        self.logger = logger
    
    def analyze_experiment_data(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析实验数据"""
        try:
            groups_data = experiment_data.get('groups', {})
            metric_type = MetricType(experiment_data.get('metric_type', MetricType.CONTINUOUS.value))
            
            # 计算各组统计数据
            group_stats = self.experiment_calculator.calculate_multiple_groups_stats(
                groups_data, metric_type
            )
            
            return {
                'group_statistics': {k: v.to_dict() for k, v in group_stats.items()},
                'analysis_timestamp': utc_now().isoformat(),
                'metric_type': metric_type.value
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze experiment data: {e}")
            raise
    
    def calculate_descriptive_stats(self, values: List[Union[int, float]]) -> Dict[str, Any]:
        """计算描述性统计"""
        try:
            stats = self.basic_calculator.calculate_descriptive_stats(values)
            return stats.to_dict()
            
        except Exception as e:
            self.logger.error(f"Failed to calculate descriptive stats: {e}")
            raise

# 全局实例
_stats_calculator = None
_stats_service = None

def get_stats_calculator() -> ExperimentStatsCalculator:
    """获取统计计算器实例（单例模式）"""
    global _stats_calculator
    if _stats_calculator is None:
        _stats_calculator = ExperimentStatsCalculator()
    return _stats_calculator

def get_statistical_analysis_service() -> StatisticalAnalysisService:
    """获取统计分析服务实例（单例模式）"""
    global _stats_service
    if _stats_service is None:
        _stats_service = StatisticalAnalysisService()
    return _stats_service
