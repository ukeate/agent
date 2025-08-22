"""
统计功效和样本量计算服务 - 实验设计的核心工具
"""
import math
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import numpy as np

from core.logging import get_logger
from services.statistical_analysis_service import MetricType

logger = get_logger(__name__)


class PowerAnalysisType(str, Enum):
    """功效分析类型"""
    SAMPLE_SIZE = "sample_size"  # 计算样本量
    POWER = "power"  # 计算统计功效
    EFFECT_SIZE = "effect_size"  # 计算可检测效应量
    ALPHA = "alpha"  # 计算显著性水平


class TestType(str, Enum):
    """检验类型"""
    ONE_SAMPLE_T = "one_sample_t"  # 单样本t检验
    TWO_SAMPLE_T = "two_sample_t"  # 双样本t检验
    PAIRED_T = "paired_t"  # 配对t检验
    ONE_PROPORTION = "one_proportion"  # 单比例检验
    TWO_PROPORTIONS = "two_proportions"  # 双比例检验
    CHI_SQUARE = "chi_square"  # 卡方检验


class AlternativeHypothesis(str, Enum):
    """备择假设类型"""
    TWO_SIDED = "two-sided"  # 双边
    GREATER = "greater"  # 大于
    LESS = "less"  # 小于


@dataclass
class PowerAnalysisResult:
    """功效分析结果"""
    analysis_type: PowerAnalysisType  # 分析类型
    test_type: TestType  # 检验类型
    effect_size: float  # 效应量
    alpha: float  # 显著性水平
    power: float  # 统计功效
    sample_size: Optional[Union[int, Tuple[int, int]]] = None  # 样本量
    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "analysis_type": self.analysis_type.value,
            "test_type": self.test_type.value,
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "power": self.power,
            "sample_size": self.sample_size,
            "alternative": self.alternative.value
        }


class TTestPowerCalculator:
    """t检验功效计算器"""
    
    def __init__(self):
        self.logger = logger
    
    def calculate_power(self, effect_size: float, sample_size: Union[int, Tuple[int, int]],
                       alpha: float = 0.05, test_type: TestType = TestType.TWO_SAMPLE_T,
                       alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED) -> float:
        """
        计算统计功效
        
        Args:
            effect_size: 效应量（Cohen's d）
            sample_size: 样本量
            alpha: 显著性水平
            test_type: 检验类型
            alternative: 备择假设类型
        
        Returns:
            统计功效
        """
        try:
            if test_type == TestType.ONE_SAMPLE_T:
                n = sample_size if isinstance(sample_size, int) else sample_size[0]
                df = n - 1
                ncp = effect_size * math.sqrt(n)  # 非中心参数
                
            elif test_type == TestType.TWO_SAMPLE_T:
                if isinstance(sample_size, tuple):
                    n1, n2 = sample_size
                    harmonic_mean = 2 / (1/n1 + 1/n2)
                    df = n1 + n2 - 2
                else:
                    n1 = n2 = sample_size
                    harmonic_mean = sample_size
                    df = 2 * sample_size - 2
                
                ncp = effect_size * math.sqrt(harmonic_mean / 2)
                
            elif test_type == TestType.PAIRED_T:
                n = sample_size if isinstance(sample_size, int) else sample_size[0]
                df = n - 1
                ncp = effect_size * math.sqrt(n)
                
            else:
                raise ValueError(f"Unsupported test type for t-test: {test_type}")
            
            # 计算临界值
            if alternative == AlternativeHypothesis.TWO_SIDED:
                t_critical = stats.t.ppf(1 - alpha / 2, df)
                power = 1 - stats.nct.cdf(t_critical, df, ncp) + stats.nct.cdf(-t_critical, df, ncp)
            elif alternative == AlternativeHypothesis.GREATER:
                t_critical = stats.t.ppf(1 - alpha, df)
                power = 1 - stats.nct.cdf(t_critical, df, ncp)
            else:  # LESS
                t_critical = stats.t.ppf(alpha, df)
                power = stats.nct.cdf(t_critical, df, ncp)
            
            return min(1.0, max(0.0, power))
            
        except Exception as e:
            self.logger.error(f"T-test power calculation failed: {e}")
            raise
    
    def calculate_sample_size(self, effect_size: float, power: float = 0.8,
                            alpha: float = 0.05, test_type: TestType = TestType.TWO_SAMPLE_T,
                            alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
                            ratio: float = 1.0) -> Union[int, Tuple[int, int]]:
        """
        计算所需样本量
        
        Args:
            effect_size: 效应量
            power: 期望统计功效
            alpha: 显著性水平
            test_type: 检验类型
            alternative: 备择假设类型
            ratio: 样本量比例（n2/n1，仅用于双样本检验）
        
        Returns:
            样本量
        """
        try:
            if not (0 < power < 1):
                raise ValueError("Power must be between 0 and 1")
            
            if effect_size <= 0:
                raise ValueError("Effect size must be positive")
            
            # 使用二分搜索找到合适的样本量
            if test_type in [TestType.ONE_SAMPLE_T, TestType.PAIRED_T]:
                min_n, max_n = 2, 10000
                
                while max_n - min_n > 1:
                    mid_n = (min_n + max_n) // 2
                    calculated_power = self.calculate_power(
                        effect_size, mid_n, alpha, test_type, alternative
                    )
                    
                    if calculated_power < power:
                        min_n = mid_n
                    else:
                        max_n = mid_n
                
                return max_n
                
            elif test_type == TestType.TWO_SAMPLE_T:
                min_n, max_n = 2, 10000
                
                while max_n - min_n > 1:
                    mid_n = (min_n + max_n) // 2
                    n1 = mid_n
                    n2 = int(mid_n * ratio)
                    
                    calculated_power = self.calculate_power(
                        effect_size, (n1, n2), alpha, test_type, alternative
                    )
                    
                    if calculated_power < power:
                        min_n = mid_n
                    else:
                        max_n = mid_n
                
                n1 = max_n
                n2 = int(max_n * ratio)
                return (n1, n2)
            
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
                
        except Exception as e:
            self.logger.error(f"T-test sample size calculation failed: {e}")
            raise
    
    def calculate_detectable_effect_size(self, sample_size: Union[int, Tuple[int, int]],
                                       power: float = 0.8, alpha: float = 0.05,
                                       test_type: TestType = TestType.TWO_SAMPLE_T,
                                       alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED) -> float:
        """
        计算可检测的最小效应量
        
        Args:
            sample_size: 样本量
            power: 期望统计功效
            alpha: 显著性水平
            test_type: 检验类型
            alternative: 备择假设类型
        
        Returns:
            最小可检测效应量
        """
        try:
            # 使用二分搜索找到最小效应量
            min_effect, max_effect = 0.01, 5.0
            
            while max_effect - min_effect > 0.001:
                mid_effect = (min_effect + max_effect) / 2
                calculated_power = self.calculate_power(
                    mid_effect, sample_size, alpha, test_type, alternative
                )
                
                if calculated_power < power:
                    min_effect = mid_effect
                else:
                    max_effect = mid_effect
            
            return max_effect
            
        except Exception as e:
            self.logger.error(f"Detectable effect size calculation failed: {e}")
            raise


class ProportionPowerCalculator:
    """比例检验功效计算器"""
    
    def __init__(self):
        self.logger = logger
    
    def calculate_power(self, p1: float, p2: float, sample_size: Union[int, Tuple[int, int]],
                       alpha: float = 0.05, test_type: TestType = TestType.TWO_PROPORTIONS,
                       alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED) -> float:
        """
        计算比例检验的统计功效
        
        Args:
            p1: 第一组比例
            p2: 第二组比例（或假设值）
            sample_size: 样本量
            alpha: 显著性水平
            test_type: 检验类型
            alternative: 备择假设类型
        
        Returns:
            统计功效
        """
        try:
            if not (0 <= p1 <= 1) or not (0 <= p2 <= 1):
                raise ValueError("Proportions must be between 0 and 1")
            
            if test_type == TestType.ONE_PROPORTION:
                n = sample_size if isinstance(sample_size, int) else sample_size[0]
                p0 = p2  # 假设值
                p_alt = p1  # 备择值
                
                # 零假设下的标准误
                se_null = math.sqrt(p0 * (1 - p0) / n)
                
                # 备择假设下的标准误
                se_alt = math.sqrt(p_alt * (1 - p_alt) / n)
                
                if alternative == AlternativeHypothesis.TWO_SIDED:
                    z_critical = stats.norm.ppf(1 - alpha / 2)
                    z_beta_upper = (z_critical * se_null - (p_alt - p0)) / se_alt
                    z_beta_lower = (-z_critical * se_null - (p_alt - p0)) / se_alt
                    power = 1 - (stats.norm.cdf(z_beta_upper) - stats.norm.cdf(z_beta_lower))
                elif alternative == AlternativeHypothesis.GREATER:
                    z_critical = stats.norm.ppf(1 - alpha)
                    z_beta = (z_critical * se_null - (p_alt - p0)) / se_alt
                    power = 1 - stats.norm.cdf(z_beta)
                else:  # LESS
                    z_critical = stats.norm.ppf(alpha)
                    z_beta = (z_critical * se_null - (p_alt - p0)) / se_alt
                    power = stats.norm.cdf(z_beta)
                
            elif test_type == TestType.TWO_PROPORTIONS:
                if isinstance(sample_size, tuple):
                    n1, n2 = sample_size
                else:
                    n1 = n2 = sample_size
                
                # 合并比例估计
                p_pooled = (p1 + p2) / 2
                
                # 零假设下的标准误（假设p1 = p2）
                se_null = math.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                
                # 备择假设下的标准误
                se_alt = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
                
                if alternative == AlternativeHypothesis.TWO_SIDED:
                    z_critical = stats.norm.ppf(1 - alpha / 2)
                    z_beta_upper = (z_critical * se_null - abs(p1 - p2)) / se_alt
                    z_beta_lower = (-z_critical * se_null - abs(p1 - p2)) / se_alt
                    power = 1 - (stats.norm.cdf(z_beta_upper) - stats.norm.cdf(z_beta_lower))
                elif alternative == AlternativeHypothesis.GREATER:
                    z_critical = stats.norm.ppf(1 - alpha)
                    z_beta = (z_critical * se_null - (p1 - p2)) / se_alt
                    power = 1 - stats.norm.cdf(z_beta)
                else:  # LESS
                    z_critical = stats.norm.ppf(alpha)
                    z_beta = (z_critical * se_null - (p1 - p2)) / se_alt
                    power = stats.norm.cdf(z_beta)
            
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
            
            return min(1.0, max(0.0, power))
            
        except Exception as e:
            self.logger.error(f"Proportion power calculation failed: {e}")
            raise
    
    def calculate_sample_size(self, p1: float, p2: float, power: float = 0.8,
                            alpha: float = 0.05, test_type: TestType = TestType.TWO_PROPORTIONS,
                            alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED,
                            ratio: float = 1.0) -> Union[int, Tuple[int, int]]:
        """
        计算比例检验所需样本量
        
        Args:
            p1: 第一组比例
            p2: 第二组比例
            power: 期望统计功效
            alpha: 显著性水平
            test_type: 检验类型
            alternative: 备择假设类型
            ratio: 样本量比例
        
        Returns:
            样本量
        """
        try:
            if not (0 <= p1 <= 1) or not (0 <= p2 <= 1):
                raise ValueError("Proportions must be between 0 and 1")
            
            if not (0 < power < 1):
                raise ValueError("Power must be between 0 and 1")
            
            # 使用二分搜索
            min_n, max_n = 2, 100000
            
            while max_n - min_n > 1:
                mid_n = (min_n + max_n) // 2
                
                if test_type == TestType.ONE_PROPORTION:
                    calculated_power = self.calculate_power(
                        p1, p2, mid_n, alpha, test_type, alternative
                    )
                elif test_type == TestType.TWO_PROPORTIONS:
                    n1 = mid_n
                    n2 = int(mid_n * ratio)
                    calculated_power = self.calculate_power(
                        p1, p2, (n1, n2), alpha, test_type, alternative
                    )
                
                if calculated_power < power:
                    min_n = mid_n
                else:
                    max_n = mid_n
            
            if test_type == TestType.ONE_PROPORTION:
                return max_n
            else:
                n1 = max_n
                n2 = int(max_n * ratio)
                return (n1, n2)
                
        except Exception as e:
            self.logger.error(f"Proportion sample size calculation failed: {e}")
            raise
    
    def calculate_detectable_proportion_difference(self, baseline_proportion: float,
                                                 sample_size: Union[int, Tuple[int, int]],
                                                 power: float = 0.8, alpha: float = 0.05,
                                                 test_type: TestType = TestType.TWO_PROPORTIONS,
                                                 alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED) -> float:
        """
        计算可检测的最小比例差异
        
        Args:
            baseline_proportion: 基准比例
            sample_size: 样本量
            power: 期望统计功效
            alpha: 显著性水平
            test_type: 检验类型
            alternative: 备择假设类型
        
        Returns:
            最小可检测比例差异
        """
        try:
            # 使用二分搜索找到最小差异
            min_diff, max_diff = 0.001, 1 - baseline_proportion
            
            while max_diff - min_diff > 0.0001:
                mid_diff = (min_diff + max_diff) / 2
                p2 = baseline_proportion + mid_diff
                
                if p2 > 1:
                    max_diff = mid_diff
                    continue
                
                calculated_power = self.calculate_power(
                    baseline_proportion, p2, sample_size, alpha, test_type, alternative
                )
                
                if calculated_power < power:
                    min_diff = mid_diff
                else:
                    max_diff = mid_diff
            
            return max_diff
            
        except Exception as e:
            self.logger.error(f"Detectable proportion difference calculation failed: {e}")
            raise


class PowerAnalysisService:
    """统计功效分析服务 - 统一接口"""
    
    def __init__(self):
        self.t_test_calculator = TTestPowerCalculator()
        self.proportion_calculator = ProportionPowerCalculator()
        self.logger = logger
    
    def run_power_analysis(self, analysis_type: PowerAnalysisType,
                          test_type: TestType, **kwargs) -> PowerAnalysisResult:
        """
        运行功效分析
        
        Args:
            analysis_type: 分析类型
            test_type: 检验类型
            **kwargs: 分析参数
        
        Returns:
            功效分析结果
        """
        try:
            if test_type in [TestType.ONE_SAMPLE_T, TestType.TWO_SAMPLE_T, TestType.PAIRED_T]:
                return self._run_t_test_analysis(analysis_type, test_type, **kwargs)
            elif test_type in [TestType.ONE_PROPORTION, TestType.TWO_PROPORTIONS]:
                return self._run_proportion_analysis(analysis_type, test_type, **kwargs)
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
                
        except Exception as e:
            self.logger.error(f"Power analysis failed: {e}")
            raise
    
    def _run_t_test_analysis(self, analysis_type: PowerAnalysisType,
                           test_type: TestType, **kwargs) -> PowerAnalysisResult:
        """运行t检验功效分析"""
        try:
            if analysis_type == PowerAnalysisType.POWER:
                power = self.t_test_calculator.calculate_power(
                    effect_size=kwargs["effect_size"],
                    sample_size=kwargs["sample_size"],
                    alpha=kwargs.get("alpha", 0.05),
                    test_type=test_type,
                    alternative=kwargs.get("alternative", AlternativeHypothesis.TWO_SIDED)
                )
                
                return PowerAnalysisResult(
                    analysis_type=analysis_type,
                    test_type=test_type,
                    effect_size=kwargs["effect_size"],
                    alpha=kwargs.get("alpha", 0.05),
                    power=power,
                    sample_size=kwargs["sample_size"],
                    alternative=kwargs.get("alternative", AlternativeHypothesis.TWO_SIDED)
                )
            
            elif analysis_type == PowerAnalysisType.SAMPLE_SIZE:
                sample_size = self.t_test_calculator.calculate_sample_size(
                    effect_size=kwargs["effect_size"],
                    power=kwargs.get("power", 0.8),
                    alpha=kwargs.get("alpha", 0.05),
                    test_type=test_type,
                    alternative=kwargs.get("alternative", AlternativeHypothesis.TWO_SIDED),
                    ratio=kwargs.get("ratio", 1.0)
                )
                
                return PowerAnalysisResult(
                    analysis_type=analysis_type,
                    test_type=test_type,
                    effect_size=kwargs["effect_size"],
                    alpha=kwargs.get("alpha", 0.05),
                    power=kwargs.get("power", 0.8),
                    sample_size=sample_size,
                    alternative=kwargs.get("alternative", AlternativeHypothesis.TWO_SIDED)
                )
            
            elif analysis_type == PowerAnalysisType.EFFECT_SIZE:
                effect_size = self.t_test_calculator.calculate_detectable_effect_size(
                    sample_size=kwargs["sample_size"],
                    power=kwargs.get("power", 0.8),
                    alpha=kwargs.get("alpha", 0.05),
                    test_type=test_type,
                    alternative=kwargs.get("alternative", AlternativeHypothesis.TWO_SIDED)
                )
                
                return PowerAnalysisResult(
                    analysis_type=analysis_type,
                    test_type=test_type,
                    effect_size=effect_size,
                    alpha=kwargs.get("alpha", 0.05),
                    power=kwargs.get("power", 0.8),
                    sample_size=kwargs["sample_size"],
                    alternative=kwargs.get("alternative", AlternativeHypothesis.TWO_SIDED)
                )
            
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
        except Exception as e:
            self.logger.error(f"T-test power analysis failed: {e}")
            raise
    
    def _run_proportion_analysis(self, analysis_type: PowerAnalysisType,
                               test_type: TestType, **kwargs) -> PowerAnalysisResult:
        """运行比例检验功效分析"""
        try:
            if analysis_type == PowerAnalysisType.POWER:
                power = self.proportion_calculator.calculate_power(
                    p1=kwargs["p1"],
                    p2=kwargs["p2"],
                    sample_size=kwargs["sample_size"],
                    alpha=kwargs.get("alpha", 0.05),
                    test_type=test_type,
                    alternative=kwargs.get("alternative", AlternativeHypothesis.TWO_SIDED)
                )
                
                # 计算效应量（Cohen's h for proportions）
                effect_size = 2 * (math.asin(math.sqrt(kwargs["p1"])) - 
                                  math.asin(math.sqrt(kwargs["p2"])))
                
                return PowerAnalysisResult(
                    analysis_type=analysis_type,
                    test_type=test_type,
                    effect_size=abs(effect_size),
                    alpha=kwargs.get("alpha", 0.05),
                    power=power,
                    sample_size=kwargs["sample_size"],
                    alternative=kwargs.get("alternative", AlternativeHypothesis.TWO_SIDED)
                )
            
            elif analysis_type == PowerAnalysisType.SAMPLE_SIZE:
                sample_size = self.proportion_calculator.calculate_sample_size(
                    p1=kwargs["p1"],
                    p2=kwargs["p2"],
                    power=kwargs.get("power", 0.8),
                    alpha=kwargs.get("alpha", 0.05),
                    test_type=test_type,
                    alternative=kwargs.get("alternative", AlternativeHypothesis.TWO_SIDED),
                    ratio=kwargs.get("ratio", 1.0)
                )
                
                # 计算效应量
                effect_size = 2 * (math.asin(math.sqrt(kwargs["p1"])) - 
                                  math.asin(math.sqrt(kwargs["p2"])))
                
                return PowerAnalysisResult(
                    analysis_type=analysis_type,
                    test_type=test_type,
                    effect_size=abs(effect_size),
                    alpha=kwargs.get("alpha", 0.05),
                    power=kwargs.get("power", 0.8),
                    sample_size=sample_size,
                    alternative=kwargs.get("alternative", AlternativeHypothesis.TWO_SIDED)
                )
            
            else:
                raise ValueError(f"Unsupported analysis type: {analysis_type}")
                
        except Exception as e:
            self.logger.error(f"Proportion power analysis failed: {e}")
            raise
    
    def calculate_ab_test_sample_size(self, baseline_conversion_rate: float,
                                    minimum_detectable_effect: float,
                                    power: float = 0.8, alpha: float = 0.05,
                                    alternative: AlternativeHypothesis = AlternativeHypothesis.TWO_SIDED) -> Dict[str, Any]:
        """
        计算A/B测试所需样本量
        
        Args:
            baseline_conversion_rate: 基准转化率
            minimum_detectable_effect: 最小可检测效应（相对提升）
            power: 期望统计功效
            alpha: 显著性水平
            alternative: 备择假设类型
        
        Returns:
            样本量计算结果
        """
        try:
            # 计算绝对效应
            absolute_effect = baseline_conversion_rate * minimum_detectable_effect
            treatment_conversion_rate = baseline_conversion_rate + absolute_effect
            
            if treatment_conversion_rate > 1:
                raise ValueError("Treatment conversion rate cannot exceed 1")
            
            # 计算样本量
            result = self.run_power_analysis(
                analysis_type=PowerAnalysisType.SAMPLE_SIZE,
                test_type=TestType.TWO_PROPORTIONS,
                p1=baseline_conversion_rate,
                p2=treatment_conversion_rate,
                power=power,
                alpha=alpha,
                alternative=alternative
            )
            
            control_size, treatment_size = result.sample_size
            total_size = control_size + treatment_size
            
            # 计算额外信息
            duration_days_estimates = {}
            for daily_visitors in [100, 500, 1000, 5000, 10000]:
                days = math.ceil(total_size / daily_visitors)
                duration_days_estimates[f"{daily_visitors}_visitors_per_day"] = days
            
            return {
                "power_analysis_result": result.to_dict(),
                "control_group_size": control_size,
                "treatment_group_size": treatment_size,
                "total_sample_size": total_size,
                "baseline_conversion_rate": baseline_conversion_rate,
                "treatment_conversion_rate": treatment_conversion_rate,
                "relative_effect": minimum_detectable_effect,
                "absolute_effect": absolute_effect,
                "estimated_duration_days": duration_days_estimates,
                "recommendations": self._generate_sample_size_recommendations(
                    total_size, baseline_conversion_rate, minimum_detectable_effect
                )
            }
            
        except Exception as e:
            self.logger.error(f"A/B test sample size calculation failed: {e}")
            raise
    
    def _generate_sample_size_recommendations(self, total_size: int, 
                                            baseline_rate: float,
                                            effect_size: float) -> List[str]:
        """生成样本量计算建议"""
        recommendations = []
        
        if total_size > 50000:
            recommendations.append("样本量较大，建议考虑分阶段实验或提高最小可检测效应")
        elif total_size < 1000:
            recommendations.append("样本量较小，实验可以快速完成")
        
        if baseline_rate < 0.05:
            recommendations.append("基准转化率较低，建议延长实验时间以获得足够样本")
        
        if effect_size < 0.1:
            recommendations.append("检测的效应较小，需要大样本量，考虑是否具有实际意义")
        elif effect_size > 0.5:
            recommendations.append("检测的效应较大，相对容易验证")
        
        recommendations.append("建议在实验期间监控指标变化，必要时调整样本量")
        
        return recommendations


# 全局实例
_power_analysis_service = None

def get_power_analysis_service() -> PowerAnalysisService:
    """获取功效分析服务实例（单例模式）"""
    global _power_analysis_service
    if _power_analysis_service is None:
        _power_analysis_service = PowerAnalysisService()
    return _power_analysis_service