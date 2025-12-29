"""
置信区间计算服务 - 实现各种参数的置信区间估计
"""

import math
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import numpy as np
from src.services.statistical_analysis_service import DescriptiveStats, MetricType

class ConfidenceIntervalMethod(str, Enum):
    """置信区间计算方法"""
    NORMAL = "normal"  # 正态分布（z区间）
    T_DISTRIBUTION = "t_distribution"  # t分布
    BOOTSTRAP = "bootstrap"  # 自助法
    WILSON = "wilson"  # Wilson方法（比例）
    EXACT_BINOMIAL = "exact_binomial"  # 精确二项分布
    CLOPPER_PEARSON = "clopper_pearson"  # Clopper-Pearson方法

class ParameterType(str, Enum):
    """参数类型"""
    MEAN = "mean"  # 均值
    PROPORTION = "proportion"  # 比例
    VARIANCE = "variance"  # 方差
    DIFFERENCE_MEANS = "difference_means"  # 均值差
    DIFFERENCE_PROPORTIONS = "difference_proportions"  # 比例差
    RATIO_VARIANCES = "ratio_variances"  # 方差比

@dataclass
class ConfidenceInterval:
    """置信区间结果"""
    parameter_type: ParameterType  # 参数类型
    method: ConfidenceIntervalMethod  # 计算方法
    confidence_level: float  # 置信水平
    point_estimate: float  # 点估计
    lower_bound: float  # 下界
    upper_bound: float  # 上界
    margin_of_error: float  # 误差边界
    standard_error: Optional[float] = None  # 标准误
    sample_size: Optional[int] = None  # 样本量
    degrees_of_freedom: Optional[int] = None  # 自由度
    
    @property
    def width(self) -> float:
        """置信区间宽度"""
        return self.upper_bound - self.lower_bound
    
    @property
    def contains_zero(self) -> bool:
        """是否包含零"""
        return self.lower_bound <= 0 <= self.upper_bound
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "parameter_type": self.parameter_type.value,
            "method": self.method.value,
            "confidence_level": self.confidence_level,
            "point_estimate": self.point_estimate,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "margin_of_error": self.margin_of_error,
            "standard_error": self.standard_error,
            "sample_size": self.sample_size,
            "degrees_of_freedom": self.degrees_of_freedom,
            "width": self.width,
            "contains_zero": self.contains_zero
        }

class MeanConfidenceIntervalCalculator:
    """均值置信区间计算器"""
    
    def __init__(self):
        self.logger = logger
    
    def one_sample_mean_ci(self, sample: List[float], 
                          confidence_level: float = 0.95,
                          method: ConfidenceIntervalMethod = ConfidenceIntervalMethod.T_DISTRIBUTION,
                          population_std: Optional[float] = None) -> ConfidenceInterval:
        """
        单样本均值置信区间
        
        Args:
            sample: 样本数据
            confidence_level: 置信水平
            method: 计算方法
            population_std: 总体标准差（已知时使用）
        
        Returns:
            置信区间结果
        """
        if not sample:
            raise ValueError("Sample cannot be empty")
        
        try:
            n = len(sample)
            sample_mean = sum(sample) / n
            alpha = 1 - confidence_level
            
            if method == ConfidenceIntervalMethod.NORMAL and population_std is not None:
                # 使用正态分布（总体标准差已知）
                z_critical = stats.norm.ppf(1 - alpha / 2)
                standard_error = population_std / math.sqrt(n)
                margin_of_error = z_critical * standard_error
                
                return ConfidenceInterval(
                    parameter_type=ParameterType.MEAN,
                    method=ConfidenceIntervalMethod.NORMAL,
                    confidence_level=confidence_level,
                    point_estimate=sample_mean,
                    lower_bound=sample_mean - margin_of_error,
                    upper_bound=sample_mean + margin_of_error,
                    margin_of_error=margin_of_error,
                    standard_error=standard_error,
                    sample_size=n
                )
            
            elif method == ConfidenceIntervalMethod.T_DISTRIBUTION or population_std is None:
                # 使用t分布（总体标准差未知）
                if n < 2:
                    raise ValueError("Sample size must be at least 2 for t-distribution")
                
                sample_std = math.sqrt(sum((x - sample_mean) ** 2 for x in sample) / (n - 1))
                df = n - 1
                t_critical = stats.t.ppf(1 - alpha / 2, df)
                standard_error = sample_std / math.sqrt(n)
                margin_of_error = t_critical * standard_error
                
                return ConfidenceInterval(
                    parameter_type=ParameterType.MEAN,
                    method=ConfidenceIntervalMethod.T_DISTRIBUTION,
                    confidence_level=confidence_level,
                    point_estimate=sample_mean,
                    lower_bound=sample_mean - margin_of_error,
                    upper_bound=sample_mean + margin_of_error,
                    margin_of_error=margin_of_error,
                    standard_error=standard_error,
                    sample_size=n,
                    degrees_of_freedom=df
                )
            
            elif method == ConfidenceIntervalMethod.BOOTSTRAP:
                # 自助法
                return self._bootstrap_mean_ci(sample, confidence_level)
            
            else:
                raise ValueError(f"Unsupported method: {method}")
                
        except Exception as e:
            self.logger.error(f"Single sample mean CI calculation failed: {e}")
            raise
    
    def two_sample_mean_difference_ci(self, sample1: List[float], sample2: List[float],
                                    confidence_level: float = 0.95,
                                    equal_variances: bool = True) -> ConfidenceInterval:
        """
        两样本均值差置信区间
        
        Args:
            sample1: 样本1
            sample2: 样本2
            confidence_level: 置信水平
            equal_variances: 是否假定等方差
        
        Returns:
            置信区间结果
        """
        if len(sample1) < 2 or len(sample2) < 2:
            raise ValueError("Each sample must have at least 2 observations")
        
        try:
            n1, n2 = len(sample1), len(sample2)
            mean1 = sum(sample1) / n1
            mean2 = sum(sample2) / n2
            mean_diff = mean1 - mean2
            
            var1 = sum((x - mean1) ** 2 for x in sample1) / (n1 - 1)
            var2 = sum((x - mean2) ** 2 for x in sample2) / (n2 - 1)
            
            alpha = 1 - confidence_level
            
            if equal_variances:
                # 等方差假设：合并方差估计
                pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
                standard_error = math.sqrt(pooled_var * (1/n1 + 1/n2))
                df = n1 + n2 - 2
            else:
                # 不等方差：Welch-Satterthwaite方程
                standard_error = math.sqrt(var1/n1 + var2/n2)
                numerator = (var1/n1 + var2/n2) ** 2
                denominator = (var1/n1) ** 2 / (n1 - 1) + (var2/n2) ** 2 / (n2 - 1)
                df = int(numerator / denominator) if denominator > 0 else n1 + n2 - 2
            
            t_critical = stats.t.ppf(1 - alpha / 2, df)
            margin_of_error = t_critical * standard_error
            
            return ConfidenceInterval(
                parameter_type=ParameterType.DIFFERENCE_MEANS,
                method=ConfidenceIntervalMethod.T_DISTRIBUTION,
                confidence_level=confidence_level,
                point_estimate=mean_diff,
                lower_bound=mean_diff - margin_of_error,
                upper_bound=mean_diff + margin_of_error,
                margin_of_error=margin_of_error,
                standard_error=standard_error,
                sample_size=n1 + n2,
                degrees_of_freedom=df
            )
            
        except Exception as e:
            self.logger.error(f"Two sample mean difference CI calculation failed: {e}")
            raise
    
    def _bootstrap_mean_ci(self, sample: List[float], 
                          confidence_level: float = 0.95,
                          n_bootstrap: int = 1000) -> ConfidenceInterval:
        """
        自助法计算均值置信区间
        
        Args:
            sample: 原始样本
            confidence_level: 置信水平
            n_bootstrap: 自助样本数量
        
        Returns:
            置信区间结果
        """
        try:
            import random
            
            n = len(sample)
            sample_mean = sum(sample) / n
            bootstrap_means = []
            
            # 生成自助样本
            for _ in range(n_bootstrap):
                bootstrap_sample = [random.choice(sample) for _ in range(n)]
                bootstrap_mean = sum(bootstrap_sample) / n
                bootstrap_means.append(bootstrap_mean)
            
            # 计算分位数
            bootstrap_means.sort()
            alpha = 1 - confidence_level
            lower_index = int((alpha / 2) * n_bootstrap)
            upper_index = int((1 - alpha / 2) * n_bootstrap) - 1
            
            lower_bound = bootstrap_means[lower_index]
            upper_bound = bootstrap_means[upper_index]
            margin_of_error = max(sample_mean - lower_bound, upper_bound - sample_mean)
            
            return ConfidenceInterval(
                parameter_type=ParameterType.MEAN,
                method=ConfidenceIntervalMethod.BOOTSTRAP,
                confidence_level=confidence_level,
                point_estimate=sample_mean,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                margin_of_error=margin_of_error,
                sample_size=n
            )
            
        except Exception as e:
            self.logger.error(f"Bootstrap mean CI calculation failed: {e}")
            raise

class ProportionConfidenceIntervalCalculator:
    """比例置信区间计算器"""
    
    def __init__(self):
        self.logger = logger
    
    def single_proportion_ci(self, successes: int, total: int,
                           confidence_level: float = 0.95,
                           method: ConfidenceIntervalMethod = ConfidenceIntervalMethod.WILSON) -> ConfidenceInterval:
        """
        单个比例置信区间
        
        Args:
            successes: 成功次数
            total: 总次数
            confidence_level: 置信水平
            method: 计算方法
        
        Returns:
            置信区间结果
        """
        if total <= 0:
            raise ValueError("Total must be positive")
        
        if successes < 0 or successes > total:
            raise ValueError("Successes must be between 0 and total")
        
        try:
            p_hat = successes / total
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha / 2)
            
            if method == ConfidenceIntervalMethod.NORMAL:
                # 正态近似方法（Wald区间）
                standard_error = math.sqrt(p_hat * (1 - p_hat) / total)
                margin_of_error = z_critical * standard_error
                
                lower_bound = max(0, p_hat - margin_of_error)
                upper_bound = min(1, p_hat + margin_of_error)
                
                return ConfidenceInterval(
                    parameter_type=ParameterType.PROPORTION,
                    method=ConfidenceIntervalMethod.NORMAL,
                    confidence_level=confidence_level,
                    point_estimate=p_hat,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    margin_of_error=margin_of_error,
                    standard_error=standard_error,
                    sample_size=total
                )
            
            elif method == ConfidenceIntervalMethod.WILSON:
                # Wilson方法（Wilson区间）
                n = total
                z_squared = z_critical ** 2
                
                center = (p_hat + z_squared / (2 * n)) / (1 + z_squared / n)
                half_width = z_critical * math.sqrt(p_hat * (1 - p_hat) / n + z_squared / (4 * n ** 2)) / (1 + z_squared / n)
                
                lower_bound = max(0, center - half_width)
                upper_bound = min(1, center + half_width)
                margin_of_error = half_width
                
                return ConfidenceInterval(
                    parameter_type=ParameterType.PROPORTION,
                    method=ConfidenceIntervalMethod.WILSON,
                    confidence_level=confidence_level,
                    point_estimate=p_hat,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    margin_of_error=margin_of_error,
                    sample_size=total
                )
            
            elif method == ConfidenceIntervalMethod.CLOPPER_PEARSON:
                # Clopper-Pearson方法（精确方法）
                if successes == 0:
                    lower_bound = 0
                else:
                    lower_bound = stats.beta.ppf(alpha / 2, successes, total - successes + 1)
                
                if successes == total:
                    upper_bound = 1
                else:
                    upper_bound = stats.beta.ppf(1 - alpha / 2, successes + 1, total - successes)
                
                margin_of_error = max(p_hat - lower_bound, upper_bound - p_hat)
                
                return ConfidenceInterval(
                    parameter_type=ParameterType.PROPORTION,
                    method=ConfidenceIntervalMethod.CLOPPER_PEARSON,
                    confidence_level=confidence_level,
                    point_estimate=p_hat,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    margin_of_error=margin_of_error,
                    sample_size=total
                )
            
            else:
                raise ValueError(f"Unsupported method: {method}")
                
        except Exception as e:
            self.logger.error(f"Single proportion CI calculation failed: {e}")
            raise
    
    def two_proportion_difference_ci(self, successes1: int, total1: int,
                                   successes2: int, total2: int,
                                   confidence_level: float = 0.95) -> ConfidenceInterval:
        """
        两比例差置信区间
        
        Args:
            successes1: 样本1成功次数
            total1: 样本1总次数
            successes2: 样本2成功次数
            total2: 样本2总次数
            confidence_level: 置信水平
        
        Returns:
            置信区间结果
        """
        if total1 <= 0 or total2 <= 0:
            raise ValueError("Totals must be positive")
        
        if (successes1 < 0 or successes1 > total1 or 
            successes2 < 0 or successes2 > total2):
            raise ValueError("Successes must be between 0 and totals")
        
        try:
            p1 = successes1 / total1
            p2 = successes2 / total2
            p_diff = p1 - p2
            
            # 标准误计算
            se1 = p1 * (1 - p1) / total1
            se2 = p2 * (1 - p2) / total2
            standard_error = math.sqrt(se1 + se2)
            
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha / 2)
            margin_of_error = z_critical * standard_error
            
            lower_bound = p_diff - margin_of_error
            upper_bound = p_diff + margin_of_error
            
            return ConfidenceInterval(
                parameter_type=ParameterType.DIFFERENCE_PROPORTIONS,
                method=ConfidenceIntervalMethod.NORMAL,
                confidence_level=confidence_level,
                point_estimate=p_diff,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                margin_of_error=margin_of_error,
                standard_error=standard_error,
                sample_size=total1 + total2
            )
            
        except Exception as e:
            self.logger.error(f"Two proportion difference CI calculation failed: {e}")
            raise

class VarianceConfidenceIntervalCalculator:
    """方差置信区间计算器"""
    
    def __init__(self):
        self.logger = logger
    
    def single_variance_ci(self, sample: List[float],
                          confidence_level: float = 0.95) -> ConfidenceInterval:
        """
        单样本方差置信区间
        
        Args:
            sample: 样本数据
            confidence_level: 置信水平
        
        Returns:
            置信区间结果
        """
        if len(sample) < 2:
            raise ValueError("Sample size must be at least 2")
        
        try:
            n = len(sample)
            sample_mean = sum(sample) / n
            sample_variance = sum((x - sample_mean) ** 2 for x in sample) / (n - 1)
            
            df = n - 1
            alpha = 1 - confidence_level
            
            # 卡方分布的临界值
            chi2_lower = stats.chi2.ppf(alpha / 2, df)
            chi2_upper = stats.chi2.ppf(1 - alpha / 2, df)
            
            # 方差置信区间
            lower_bound = (df * sample_variance) / chi2_upper
            upper_bound = (df * sample_variance) / chi2_lower
            
            margin_of_error = max(sample_variance - lower_bound, 
                                upper_bound - sample_variance)
            
            return ConfidenceInterval(
                parameter_type=ParameterType.VARIANCE,
                method=ConfidenceIntervalMethod.T_DISTRIBUTION,  # 基于卡方分布
                confidence_level=confidence_level,
                point_estimate=sample_variance,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                margin_of_error=margin_of_error,
                sample_size=n,
                degrees_of_freedom=df
            )
            
        except Exception as e:
            self.logger.error(f"Single variance CI calculation failed: {e}")
            raise

class ConfidenceIntervalService:
    """置信区间服务 - 统一接口"""
    
    def __init__(self):
        self.mean_calculator = MeanConfidenceIntervalCalculator()
        self.proportion_calculator = ProportionConfidenceIntervalCalculator()
        self.variance_calculator = VarianceConfidenceIntervalCalculator()
        self.logger = logger
    
    def calculate_confidence_interval(self, parameter_type: ParameterType,
                                    confidence_level: float = 0.95,
                                    **kwargs) -> ConfidenceInterval:
        """
        计算置信区间（统一接口）
        
        Args:
            parameter_type: 参数类型
            confidence_level: 置信水平
            **kwargs: 计算参数
        
        Returns:
            置信区间结果
        """
        try:
            if parameter_type == ParameterType.MEAN:
                if "sample" in kwargs:
                    return self.mean_calculator.one_sample_mean_ci(
                        confidence_level=confidence_level, **kwargs
                    )
                else:
                    raise ValueError("Missing 'sample' parameter for mean CI")
            
            elif parameter_type == ParameterType.DIFFERENCE_MEANS:
                if "sample1" in kwargs and "sample2" in kwargs:
                    return self.mean_calculator.two_sample_mean_difference_ci(
                        confidence_level=confidence_level, **kwargs
                    )
                else:
                    raise ValueError("Missing 'sample1' and 'sample2' parameters for mean difference CI")
            
            elif parameter_type == ParameterType.PROPORTION:
                if "successes" in kwargs and "total" in kwargs:
                    return self.proportion_calculator.single_proportion_ci(
                        confidence_level=confidence_level, **kwargs
                    )
                else:
                    raise ValueError("Missing 'successes' and 'total' parameters for proportion CI")
            
            elif parameter_type == ParameterType.DIFFERENCE_PROPORTIONS:
                required_params = ["successes1", "total1", "successes2", "total2"]
                if all(param in kwargs for param in required_params):
                    return self.proportion_calculator.two_proportion_difference_ci(
                        confidence_level=confidence_level, **kwargs
                    )
                else:
                    raise ValueError(f"Missing parameters for proportion difference CI: {required_params}")
            
            elif parameter_type == ParameterType.VARIANCE:
                if "sample" in kwargs:
                    return self.variance_calculator.single_variance_ci(
                        confidence_level=confidence_level, **kwargs
                    )
                else:
                    raise ValueError("Missing 'sample' parameter for variance CI")
            
            else:
                raise ValueError(f"Unsupported parameter type: {parameter_type}")
                
        except Exception as e:
            self.logger.error(f"Confidence interval calculation failed: {e}")
            raise
    
    def calculate_ab_test_confidence_intervals(self, control_group: Dict[str, Any],
                                             treatment_group: Dict[str, Any],
                                             metric_type: MetricType,
                                             confidence_level: float = 0.95) -> Dict[str, ConfidenceInterval]:
        """
        计算A/B测试的置信区间
        
        Args:
            control_group: 对照组数据
            treatment_group: 实验组数据
            metric_type: 指标类型
            confidence_level: 置信水平
        
        Returns:
            置信区间结果字典
        """
        try:
            results = {}
            
            if metric_type == MetricType.CONVERSION:
                # 转化率置信区间
                control_ci = self.calculate_confidence_interval(
                    parameter_type=ParameterType.PROPORTION,
                    successes=control_group["conversions"],
                    total=control_group["total_users"],
                    confidence_level=confidence_level,
                    method=ConfidenceIntervalMethod.WILSON
                )
                
                treatment_ci = self.calculate_confidence_interval(
                    parameter_type=ParameterType.PROPORTION,
                    successes=treatment_group["conversions"],
                    total=treatment_group["total_users"],
                    confidence_level=confidence_level,
                    method=ConfidenceIntervalMethod.WILSON
                )
                
                # 转化率差异置信区间
                difference_ci = self.calculate_confidence_interval(
                    parameter_type=ParameterType.DIFFERENCE_PROPORTIONS,
                    successes1=control_group["conversions"],
                    total1=control_group["total_users"],
                    successes2=treatment_group["conversions"],
                    total2=treatment_group["total_users"],
                    confidence_level=confidence_level
                )
                
                results = {
                    "control": control_ci,
                    "treatment": treatment_ci,
                    "difference": difference_ci
                }
                
            else:
                # 连续指标置信区间
                control_ci = self.calculate_confidence_interval(
                    parameter_type=ParameterType.MEAN,
                    sample=control_group["values"],
                    confidence_level=confidence_level
                )
                
                treatment_ci = self.calculate_confidence_interval(
                    parameter_type=ParameterType.MEAN,
                    sample=treatment_group["values"],
                    confidence_level=confidence_level
                )
                
                # 均值差异置信区间
                difference_ci = self.calculate_confidence_interval(
                    parameter_type=ParameterType.DIFFERENCE_MEANS,
                    sample1=control_group["values"],
                    sample2=treatment_group["values"],
                    confidence_level=confidence_level
                )
                
                results = {
                    "control": control_ci,
                    "treatment": treatment_ci,
                    "difference": difference_ci
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"A/B test confidence intervals calculation failed: {e}")
            raise

# 全局实例
_confidence_interval_service = None

def get_confidence_interval_service() -> ConfidenceIntervalService:
    """获取置信区间服务实例（单例模式）"""
    global _confidence_interval_service
    if _confidence_interval_service is None:
        _confidence_interval_service = ConfidenceIntervalService()
    return _confidence_interval_service
