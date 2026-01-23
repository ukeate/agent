"""
假设检验服务 - 实现t检验、卡方检验等统计推断算法
"""

import math
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import numpy as np
from src.services.statistical_analysis_service import DescriptiveStats, MetricType
from src.core.logging import get_logger

logger = get_logger(__name__)

class HypothesisType(str, Enum):
    """假设检验类型"""
    TWO_SIDED = "two-sided"  # 双边检验
    LESS = "less"  # 左边检验（小于）
    GREATER = "greater"  # 右边检验（大于）

class TestStatistic(str, Enum):
    """检验统计量类型"""
    T_STATISTIC = "t_statistic"  # t统计量
    CHI_SQUARE = "chi_square"  # 卡方统计量
    Z_STATISTIC = "z_statistic"  # z统计量
    F_STATISTIC = "f_statistic"  # F统计量

@dataclass
class HypothesisTestResult:
    """假设检验结果"""
    test_type: str  # 检验类型
    statistic: float  # 检验统计量
    p_value: float  # p值
    critical_value: Optional[float] = None  # 临界值
    degrees_of_freedom: Optional[Union[int, Tuple[int, int]]] = None  # 自由度
    confidence_interval: Optional[Tuple[float, float]] = None  # 置信区间
    effect_size: Optional[float] = None  # 效应量
    power: Optional[float] = None  # 统计功效
    alpha: float = 0.05  # 显著性水平
    hypothesis_type: HypothesisType = HypothesisType.TWO_SIDED  # 假设类型
    
    @property
    def is_significant(self) -> bool:
        """是否达到统计显著性"""
        return bool(self.p_value < self.alpha)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        degrees_of_freedom = self.degrees_of_freedom
        if isinstance(degrees_of_freedom, tuple):
            degrees_of_freedom = tuple(int(x) for x in degrees_of_freedom)
        elif degrees_of_freedom is not None:
            degrees_of_freedom = int(degrees_of_freedom)

        confidence_interval = self.confidence_interval
        if confidence_interval is not None:
            confidence_interval = (float(confidence_interval[0]), float(confidence_interval[1]))

        return {
            "test_type": self.test_type,
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "critical_value": float(self.critical_value) if self.critical_value is not None else None,
            "degrees_of_freedom": degrees_of_freedom,
            "confidence_interval": confidence_interval,
            "effect_size": float(self.effect_size) if self.effect_size is not None else None,
            "power": float(self.power) if self.power is not None else None,
            "alpha": float(self.alpha),
            "hypothesis_type": self.hypothesis_type.value,
            "is_significant": self.is_significant
        }

class TTestCalculator:
    """t检验计算器"""
    
    def __init__(self):
        self.logger = logger
    
    def one_sample_t_test(self, sample: List[float], population_mean: float,
                         hypothesis_type: HypothesisType = HypothesisType.TWO_SIDED,
                         alpha: float = 0.05) -> HypothesisTestResult:
        """
        单样本t检验
        
        Args:
            sample: 样本数据
            population_mean: 总体均值（零假设值）
            hypothesis_type: 假设检验类型
            alpha: 显著性水平
        
        Returns:
            检验结果
        """
        if len(sample) < 2:
            raise ValueError("Sample size must be at least 2")
        
        try:
            n = len(sample)
            sample_mean = sum(sample) / n
            sample_std = math.sqrt(sum((x - sample_mean) ** 2 for x in sample) / (n - 1))
            
            if sample_std == 0:
                raise ValueError("Sample standard deviation cannot be zero")
            
            # 计算t统计量
            t_statistic = (sample_mean - population_mean) / (sample_std / math.sqrt(n))
            df = n - 1
            
            # 计算p值
            if hypothesis_type == HypothesisType.TWO_SIDED:
                p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
            elif hypothesis_type == HypothesisType.GREATER:
                p_value = 1 - stats.t.cdf(t_statistic, df)
            else:  # LESS
                p_value = stats.t.cdf(t_statistic, df)
            
            # 计算临界值
            if hypothesis_type == HypothesisType.TWO_SIDED:
                critical_value = stats.t.ppf(1 - alpha / 2, df)
            else:
                critical_value = stats.t.ppf(1 - alpha, df)
            
            # 计算置信区间
            margin_error = critical_value * (sample_std / math.sqrt(n))
            if hypothesis_type == HypothesisType.TWO_SIDED:
                confidence_interval = (sample_mean - margin_error, sample_mean + margin_error)
            else:
                confidence_interval = None
            
            # 计算效应量（Cohen's d）
            effect_size = abs(sample_mean - population_mean) / sample_std
            
            return HypothesisTestResult(
                test_type="one_sample_t_test",
                statistic=t_statistic,
                p_value=p_value,
                critical_value=critical_value,
                degrees_of_freedom=df,
                confidence_interval=confidence_interval,
                effect_size=effect_size,
                alpha=alpha,
                hypothesis_type=hypothesis_type
            )
            
        except Exception as e:
            self.logger.error(f"One sample t-test failed: {e}")
            raise
    
    def independent_two_sample_t_test(self, sample1: List[float], sample2: List[float],
                                    equal_variances: bool = True,
                                    hypothesis_type: HypothesisType = HypothesisType.TWO_SIDED,
                                    alpha: float = 0.05) -> HypothesisTestResult:
        """
        独立双样本t检验
        
        Args:
            sample1: 样本1数据
            sample2: 样本2数据
            equal_variances: 是否假定方差相等（True: Student t-test, False: Welch t-test）
            hypothesis_type: 假设检验类型
            alpha: 显著性水平
        
        Returns:
            检验结果
        """
        if len(sample1) < 2 or len(sample2) < 2:
            raise ValueError("Each sample size must be at least 2")
        
        try:
            n1, n2 = len(sample1), len(sample2)
            mean1 = sum(sample1) / n1
            mean2 = sum(sample2) / n2
            
            var1 = sum((x - mean1) ** 2 for x in sample1) / (n1 - 1)
            var2 = sum((x - mean2) ** 2 for x in sample2) / (n2 - 1)
            std1 = math.sqrt(var1)
            std2 = math.sqrt(var2)
            
            mean_diff = mean1 - mean2
            
            if equal_variances:
                # Student's t-test (等方差假设)
                pooled_std = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
                standard_error = pooled_std * math.sqrt(1/n1 + 1/n2)
                df = n1 + n2 - 2
                test_name = "independent_two_sample_t_test"
            else:
                # Welch's t-test (不等方差假设)
                standard_error = math.sqrt(var1/n1 + var2/n2)
                
                # Welch-Satterthwaite方程计算自由度
                numerator = (var1/n1 + var2/n2) ** 2
                denominator = (var1/n1) ** 2 / (n1 - 1) + (var2/n2) ** 2 / (n2 - 1)
                df = int(numerator / denominator) if denominator > 0 else n1 + n2 - 2
                test_name = "welch_t_test"
            
            if standard_error == 0:
                raise ValueError("Standard error cannot be zero")
            
            # 计算t统计量
            t_statistic = mean_diff / standard_error
            
            # 计算p值
            if hypothesis_type == HypothesisType.TWO_SIDED:
                p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
            elif hypothesis_type == HypothesisType.GREATER:
                p_value = 1 - stats.t.cdf(t_statistic, df)
            else:  # LESS
                p_value = stats.t.cdf(t_statistic, df)
            
            # 计算临界值
            if hypothesis_type == HypothesisType.TWO_SIDED:
                critical_value = stats.t.ppf(1 - alpha / 2, df)
            else:
                critical_value = stats.t.ppf(1 - alpha, df)
            
            # 计算置信区间
            margin_error = critical_value * standard_error
            if hypothesis_type == HypothesisType.TWO_SIDED:
                confidence_interval = (mean_diff - margin_error, mean_diff + margin_error)
            else:
                confidence_interval = None
            
            # 计算效应量（Cohen's d）
            if equal_variances:
                pooled_std_for_effect = pooled_std
            else:
                pooled_std_for_effect = math.sqrt((var1 + var2) / 2)
            
            effect_size = abs(mean_diff) / pooled_std_for_effect if pooled_std_for_effect > 0 else 0
            
            return HypothesisTestResult(
                test_type=test_name,
                statistic=t_statistic,
                p_value=p_value,
                critical_value=critical_value,
                degrees_of_freedom=df,
                confidence_interval=confidence_interval,
                effect_size=effect_size,
                alpha=alpha,
                hypothesis_type=hypothesis_type
            )
            
        except Exception as e:
            self.logger.error(f"Independent two-sample t-test failed: {e}")
            raise
    
    def paired_t_test(self, sample1: List[float], sample2: List[float],
                     hypothesis_type: HypothesisType = HypothesisType.TWO_SIDED,
                     alpha: float = 0.05) -> HypothesisTestResult:
        """
        配对t检验
        
        Args:
            sample1: 配对样本1
            sample2: 配对样本2
            hypothesis_type: 假设检验类型
            alpha: 显著性水平
        
        Returns:
            检验结果
        """
        if len(sample1) != len(sample2):
            raise ValueError("Paired samples must have the same length")
        
        if len(sample1) < 2:
            raise ValueError("Sample size must be at least 2")
        
        try:
            # 计算差值
            differences = [x1 - x2 for x1, x2 in zip(sample1, sample2)]
            
            # 对差值进行单样本t检验（零假设：差值均值 = 0）
            return self.one_sample_t_test(differences, 0, hypothesis_type, alpha)
            
        except Exception as e:
            self.logger.error(f"Paired t-test failed: {e}")
            raise

class ChiSquareTestCalculator:
    """卡方检验计算器"""
    
    def __init__(self):
        self.logger = logger
    
    def goodness_of_fit_test(self, observed: List[int], expected: List[float],
                            alpha: float = 0.05) -> HypothesisTestResult:
        """
        卡方拟合优度检验
        
        Args:
            observed: 观测频数
            expected: 期望频数
            alpha: 显著性水平
        
        Returns:
            检验结果
        """
        if len(observed) != len(expected):
            raise ValueError("Observed and expected frequencies must have the same length")
        
        if len(observed) < 2:
            raise ValueError("At least 2 categories are required")
        
        if any(e <= 0 for e in expected):
            raise ValueError("All expected frequencies must be positive")
        
        if any(e < 5 for e in expected):
            self.logger.warning("Some expected frequencies are less than 5, results may be unreliable")
        
        try:
            # 计算卡方统计量
            chi_square = sum((o - e) ** 2 / e for o, e in zip(observed, expected))
            
            # 自由度
            df = len(observed) - 1
            
            # 计算p值
            p_value = 1 - stats.chi2.cdf(chi_square, df)
            
            # 计算临界值
            critical_value = stats.chi2.ppf(1 - alpha, df)
            
            return HypothesisTestResult(
                test_type="chi_square_goodness_of_fit",
                statistic=chi_square,
                p_value=p_value,
                critical_value=critical_value,
                degrees_of_freedom=df,
                alpha=alpha,
                hypothesis_type=HypothesisType.GREATER  # 卡方检验总是右尾检验
            )
            
        except Exception as e:
            self.logger.error(f"Chi-square goodness of fit test failed: {e}")
            raise
    
    def independence_test(self, contingency_table: List[List[int]],
                         alpha: float = 0.05) -> HypothesisTestResult:
        """
        卡方独立性检验
        
        Args:
            contingency_table: 列联表（二维列表）
            alpha: 显著性水平
        
        Returns:
            检验结果
        """
        if len(contingency_table) < 2 or len(contingency_table[0]) < 2:
            raise ValueError("Contingency table must be at least 2x2")
        
        try:
            rows = len(contingency_table)
            cols = len(contingency_table[0])
            
            # 验证表格维度一致性
            if not all(len(row) == cols for row in contingency_table):
                raise ValueError("All rows must have the same number of columns")
            
            # 计算行和、列和、总和
            row_sums = [sum(row) for row in contingency_table]
            col_sums = [sum(contingency_table[i][j] for i in range(rows)) for j in range(cols)]
            total = sum(row_sums)
            
            if total == 0:
                raise ValueError("Contingency table cannot be empty")
            
            # 计算期望频数
            expected = []
            for i in range(rows):
                expected_row = []
                for j in range(cols):
                    expected_freq = (row_sums[i] * col_sums[j]) / total
                    expected_row.append(expected_freq)
                expected.append(expected_row)
            
            # 检查期望频数
            if any(any(e < 5 for e in row) for row in expected):
                self.logger.warning("Some expected frequencies are less than 5, results may be unreliable")
            
            # 计算卡方统计量
            chi_square = 0
            for i in range(rows):
                for j in range(cols):
                    observed_freq = contingency_table[i][j]
                    expected_freq = expected[i][j]
                    if expected_freq > 0:
                        chi_square += (observed_freq - expected_freq) ** 2 / expected_freq
            
            # 自由度
            df = (rows - 1) * (cols - 1)
            
            # 计算p值
            p_value = 1 - stats.chi2.cdf(chi_square, df)
            
            # 计算临界值
            critical_value = stats.chi2.ppf(1 - alpha, df)
            
            # 计算Cramér's V（效应量）
            n = total
            effect_size = math.sqrt(chi_square / (n * min(rows - 1, cols - 1)))
            
            return HypothesisTestResult(
                test_type="chi_square_independence",
                statistic=chi_square,
                p_value=p_value,
                critical_value=critical_value,
                degrees_of_freedom=df,
                effect_size=effect_size,
                alpha=alpha,
                hypothesis_type=HypothesisType.GREATER
            )
            
        except Exception as e:
            self.logger.error(f"Chi-square independence test failed: {e}")
            raise
    
    def proportion_test(self, successes1: int, total1: int, successes2: int, total2: int,
                       hypothesis_type: HypothesisType = HypothesisType.TWO_SIDED,
                       alpha: float = 0.05) -> HypothesisTestResult:
        """
        两比例卡方检验（用于A/B测试转化率比较）
        
        Args:
            successes1: 组1成功数
            total1: 组1总数
            successes2: 组2成功数
            total2: 组2总数
            hypothesis_type: 假设检验类型
            alpha: 显著性水平
        
        Returns:
            检验结果
        """
        if total1 <= 0 or total2 <= 0:
            raise ValueError("Total counts must be positive")
        
        if successes1 < 0 or successes1 > total1 or successes2 < 0 or successes2 > total2:
            raise ValueError("Success counts must be between 0 and total counts")
        
        try:
            # 构建2x2列联表
            failures1 = total1 - successes1
            failures2 = total2 - successes2
            
            contingency_table = [
                [successes1, failures1],
                [successes2, failures2]
            ]
            
            # 使用独立性检验
            result = self.independence_test(contingency_table, alpha)
            
            # 更新测试类型和计算额外信息
            result.test_type = "two_proportion_chi_square"
            
            # 计算比例差异
            p1 = successes1 / total1
            p2 = successes2 / total2
            proportion_diff = p1 - p2
            
            # 对于双边检验，p值已经正确计算
            # 对于单边检验，需要调整
            if hypothesis_type != HypothesisType.TWO_SIDED:
                # 使用z检验方法重新计算单边p值
                pooled_proportion = (successes1 + successes2) / (total1 + total2)
                standard_error = math.sqrt(pooled_proportion * (1 - pooled_proportion) * (1/total1 + 1/total2))
                
                if standard_error > 0:
                    z_statistic = proportion_diff / standard_error
                    
                    if hypothesis_type == HypothesisType.GREATER:
                        result.p_value = 1 - stats.norm.cdf(z_statistic)
                    else:  # LESS
                        result.p_value = stats.norm.cdf(z_statistic)
            
            result.hypothesis_type = hypothesis_type
            
            return result
            
        except Exception as e:
            self.logger.error(f"Two proportion chi-square test failed: {e}")
            raise

class HypothesisTestingService:
    """假设检验服务 - 统一接口"""
    
    def __init__(self):
        self.t_test_calculator = TTestCalculator()
        self.chi_square_calculator = ChiSquareTestCalculator()
        self.logger = logger
    
    def compare_two_groups(self, group1_data: Dict[str, Any], group2_data: Dict[str, Any],
                          metric_type: MetricType, 
                          hypothesis_type: HypothesisType = HypothesisType.TWO_SIDED,
                          alpha: float = 0.05,
                          equal_variances: bool = True) -> HypothesisTestResult:
        """
        比较两组数据（自动选择合适的检验方法）
        
        Args:
            group1_data: 组1数据
            group2_data: 组2数据
            metric_type: 指标类型
            hypothesis_type: 假设检验类型
            alpha: 显著性水平
            equal_variances: 是否假定等方差（仅对t检验有效）
        
        Returns:
            检验结果
        """
        try:
            if metric_type == MetricType.CONVERSION:
                # 转化率比较：使用卡方检验
                if "conversions" not in group1_data or "total_users" not in group1_data:
                    raise ValueError("Conversion data requires 'conversions' and 'total_users' fields")
                if "conversions" not in group2_data or "total_users" not in group2_data:
                    raise ValueError("Conversion data requires 'conversions' and 'total_users' fields")
                
                return self.chi_square_calculator.proportion_test(
                    successes1=group1_data["conversions"],
                    total1=group1_data["total_users"],
                    successes2=group2_data["conversions"],
                    total2=group2_data["total_users"],
                    hypothesis_type=hypothesis_type,
                    alpha=alpha
                )
            
            else:
                # 连续数据比较：使用t检验
                if "values" not in group1_data or "values" not in group2_data:
                    raise ValueError("Continuous data requires 'values' field")
                
                return self.t_test_calculator.independent_two_sample_t_test(
                    sample1=group1_data["values"],
                    sample2=group2_data["values"],
                    equal_variances=equal_variances,
                    hypothesis_type=hypothesis_type,
                    alpha=alpha
                )
                
        except Exception as e:
            self.logger.error(f"Two group comparison failed: {e}")
            raise
    
    def run_t_test(self, test_type: str, **kwargs) -> HypothesisTestResult:
        """
        运行t检验
        
        Args:
            test_type: 检验类型（"one_sample", "independent_two_sample", "paired"）
            **kwargs: 检验参数
        
        Returns:
            检验结果
        """
        try:
            if test_type == "one_sample":
                return self.t_test_calculator.one_sample_t_test(**kwargs)
            elif test_type == "independent_two_sample":
                return self.t_test_calculator.independent_two_sample_t_test(**kwargs)
            elif test_type == "paired":
                return self.t_test_calculator.paired_t_test(**kwargs)
            else:
                raise ValueError(f"Unknown t-test type: {test_type}")
        
        except Exception as e:
            self.logger.error(f"T-test failed: {e}")
            raise
    
    def run_chi_square_test(self, test_type: str, **kwargs) -> HypothesisTestResult:
        """
        运行卡方检验
        
        Args:
            test_type: 检验类型（"goodness_of_fit", "independence", "proportion"）
            **kwargs: 检验参数
        
        Returns:
            检验结果
        """
        try:
            if test_type == "goodness_of_fit":
                return self.chi_square_calculator.goodness_of_fit_test(**kwargs)
            elif test_type == "independence":
                return self.chi_square_calculator.independence_test(**kwargs)
            elif test_type == "proportion":
                return self.chi_square_calculator.proportion_test(**kwargs)
            else:
                raise ValueError(f"Unknown chi-square test type: {test_type}")
        
        except Exception as e:
            self.logger.error(f"Chi-square test failed: {e}")
            raise

# 全局实例
_hypothesis_testing_service = None

def get_hypothesis_testing_service() -> HypothesisTestingService:
    """获取假设检验服务实例（单例模式）"""
    global _hypothesis_testing_service
    if _hypothesis_testing_service is None:
        _hypothesis_testing_service = HypothesisTestingService()
    return _hypothesis_testing_service
