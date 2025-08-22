"""
多重检验校正服务 - 控制多重比较的错误率
"""
import math
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import stats

from core.logging import get_logger

logger = get_logger(__name__)


class CorrectionMethod(str, Enum):
    """多重检验校正方法"""
    BONFERRONI = "bonferroni"  # Bonferroni校正
    HOLM = "holm"  # Holm-Bonferroni校正
    HOCHBERG = "hochberg"  # Hochberg校正
    HOMMEL = "hommel"  # Hommel校正
    FDR_BH = "fdr_bh"  # Benjamini-Hochberg FDR
    FDR_BY = "fdr_by"  # Benjamini-Yekutieli FDR
    SIDAK = "sidak"  # Šidák校正
    HOLM_SIDAK = "holm_sidak"  # Holm-Šidák校正
    NONE = "none"  # 不进行校正


class ErrorRateType(str, Enum):
    """错误率类型"""
    FWER = "fwer"  # 家族错误率（Family-Wise Error Rate）
    FDR = "fdr"  # 错误发现率（False Discovery Rate）
    PER_COMPARISON = "per_comparison"  # 每次比较错误率


@dataclass
class MultipleTestingResult:
    """多重检验校正结果"""
    original_pvalues: List[float]  # 原始p值
    corrected_pvalues: List[float]  # 校正后p值
    rejected: List[bool]  # 是否拒绝零假设
    correction_method: CorrectionMethod  # 校正方法
    alpha: float  # 显著性水平
    adjusted_alpha: Optional[float] = None  # 调整后的显著性水平
    error_rate_type: ErrorRateType = ErrorRateType.FWER
    
    @property
    def num_rejected(self) -> int:
        """拒绝的假设数量"""
        return sum(self.rejected)
    
    @property
    def num_tests(self) -> int:
        """总检验数量"""
        return len(self.original_pvalues)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "original_pvalues": self.original_pvalues,
            "corrected_pvalues": self.corrected_pvalues,
            "rejected": self.rejected,
            "correction_method": self.correction_method.value,
            "alpha": self.alpha,
            "adjusted_alpha": self.adjusted_alpha,
            "error_rate_type": self.error_rate_type.value,
            "num_rejected": self.num_rejected,
            "num_tests": self.num_tests
        }


class FWERCorrection:
    """家族错误率（FWER）校正方法"""
    
    def __init__(self):
        self.logger = logger
    
    def bonferroni_correction(self, pvalues: List[float], alpha: float = 0.05) -> MultipleTestingResult:
        """
        Bonferroni校正
        最保守的方法，将显著性水平除以检验次数
        
        Args:
            pvalues: p值列表
            alpha: 显著性水平
        
        Returns:
            校正结果
        """
        try:
            n = len(pvalues)
            if n == 0:
                raise ValueError("P-values list cannot be empty")
            
            # 调整后的显著性水平
            adjusted_alpha = alpha / n
            
            # 校正后的p值（简单地乘以n，但不超过1）
            corrected_pvalues = [min(1.0, p * n) for p in pvalues]
            
            # 判断是否拒绝
            rejected = [p <= adjusted_alpha for p in pvalues]
            
            return MultipleTestingResult(
                original_pvalues=pvalues,
                corrected_pvalues=corrected_pvalues,
                rejected=rejected,
                correction_method=CorrectionMethod.BONFERRONI,
                alpha=alpha,
                adjusted_alpha=adjusted_alpha,
                error_rate_type=ErrorRateType.FWER
            )
            
        except Exception as e:
            self.logger.error(f"Bonferroni correction failed: {e}")
            raise
    
    def holm_bonferroni_correction(self, pvalues: List[float], alpha: float = 0.05) -> MultipleTestingResult:
        """
        Holm-Bonferroni校正
        逐步调整的Bonferroni方法，比原始Bonferroni更有功效
        
        Args:
            pvalues: p值列表
            alpha: 显著性水平
        
        Returns:
            校正结果
        """
        try:
            n = len(pvalues)
            if n == 0:
                raise ValueError("P-values list cannot be empty")
            
            # 创建索引-p值对并排序
            indexed_pvalues = [(i, p) for i, p in enumerate(pvalues)]
            indexed_pvalues.sort(key=lambda x: x[1])
            
            # 逐步调整
            corrected_pvalues = [0.0] * n
            rejected = [False] * n
            
            for rank, (original_index, p) in enumerate(indexed_pvalues):
                # Holm调整：alpha / (n - rank)
                adjusted_alpha = alpha / (n - rank)
                
                # 校正后的p值
                corrected_p = min(1.0, p * (n - rank))
                
                # 确保单调性
                if rank > 0:
                    prev_index = indexed_pvalues[rank - 1][0]
                    corrected_p = max(corrected_p, corrected_pvalues[prev_index])
                
                corrected_pvalues[original_index] = corrected_p
                
                # 判断是否拒绝（一旦有一个不拒绝，后续都不拒绝）
                if p <= adjusted_alpha and (rank == 0 or rejected[indexed_pvalues[rank - 1][0]]):
                    rejected[original_index] = True
            
            return MultipleTestingResult(
                original_pvalues=pvalues,
                corrected_pvalues=corrected_pvalues,
                rejected=rejected,
                correction_method=CorrectionMethod.HOLM,
                alpha=alpha,
                error_rate_type=ErrorRateType.FWER
            )
            
        except Exception as e:
            self.logger.error(f"Holm-Bonferroni correction failed: {e}")
            raise
    
    def sidak_correction(self, pvalues: List[float], alpha: float = 0.05) -> MultipleTestingResult:
        """
        Šidák校正
        基于独立性假设，比Bonferroni略微不保守
        
        Args:
            pvalues: p值列表
            alpha: 显著性水平
        
        Returns:
            校正结果
        """
        try:
            n = len(pvalues)
            if n == 0:
                raise ValueError("P-values list cannot be empty")
            
            # Šidák调整：1 - (1 - alpha)^(1/n)
            adjusted_alpha = 1 - (1 - alpha) ** (1 / n)
            
            # 校正后的p值：1 - (1 - p)^n
            corrected_pvalues = [min(1.0, 1 - (1 - p) ** n) for p in pvalues]
            
            # 判断是否拒绝
            rejected = [p <= adjusted_alpha for p in pvalues]
            
            return MultipleTestingResult(
                original_pvalues=pvalues,
                corrected_pvalues=corrected_pvalues,
                rejected=rejected,
                correction_method=CorrectionMethod.SIDAK,
                alpha=alpha,
                adjusted_alpha=adjusted_alpha,
                error_rate_type=ErrorRateType.FWER
            )
            
        except Exception as e:
            self.logger.error(f"Šidák correction failed: {e}")
            raise
    
    def hochberg_correction(self, pvalues: List[float], alpha: float = 0.05) -> MultipleTestingResult:
        """
        Hochberg校正
        逐步向上的方法，在某些条件下比Holm更有功效
        
        Args:
            pvalues: p值列表
            alpha: 显著性水平
        
        Returns:
            校正结果
        """
        try:
            n = len(pvalues)
            if n == 0:
                raise ValueError("P-values list cannot be empty")
            
            # 创建索引-p值对并排序（降序）
            indexed_pvalues = [(i, p) for i, p in enumerate(pvalues)]
            indexed_pvalues.sort(key=lambda x: x[1], reverse=True)
            
            corrected_pvalues = [0.0] * n
            rejected = [False] * n
            
            for rank, (original_index, p) in enumerate(indexed_pvalues):
                # Hochberg调整：alpha / (rank + 1)
                adjusted_alpha = alpha / (rank + 1)
                
                # 校正后的p值
                corrected_p = min(1.0, p * (rank + 1))
                
                # 确保单调性
                if rank > 0:
                    prev_index = indexed_pvalues[rank - 1][0]
                    corrected_p = min(corrected_p, corrected_pvalues[prev_index])
                
                corrected_pvalues[original_index] = corrected_p
                
                # 判断是否拒绝
                if p <= adjusted_alpha:
                    # 找到第一个满足条件的，则拒绝所有更小的p值
                    for j in range(rank, n):
                        rejected[indexed_pvalues[j][0]] = True
                    break
            
            return MultipleTestingResult(
                original_pvalues=pvalues,
                corrected_pvalues=corrected_pvalues,
                rejected=rejected,
                correction_method=CorrectionMethod.HOCHBERG,
                alpha=alpha,
                error_rate_type=ErrorRateType.FWER
            )
            
        except Exception as e:
            self.logger.error(f"Hochberg correction failed: {e}")
            raise


class FDRCorrection:
    """错误发现率（FDR）校正方法"""
    
    def __init__(self):
        self.logger = logger
    
    def benjamini_hochberg_correction(self, pvalues: List[float], alpha: float = 0.05) -> MultipleTestingResult:
        """
        Benjamini-Hochberg FDR校正
        控制错误发现率，比FWER方法更有功效
        
        Args:
            pvalues: p值列表
            alpha: 显著性水平（FDR水平）
        
        Returns:
            校正结果
        """
        try:
            n = len(pvalues)
            if n == 0:
                raise ValueError("P-values list cannot be empty")
            
            # 创建索引-p值对并排序
            indexed_pvalues = [(i, p) for i, p in enumerate(pvalues)]
            indexed_pvalues.sort(key=lambda x: x[1])
            
            corrected_pvalues = [0.0] * n
            rejected = [False] * n
            
            # BH校正
            for rank_minus_1, (original_index, p) in enumerate(indexed_pvalues):
                rank = rank_minus_1 + 1
                
                # BH调整：p * n / rank
                corrected_p = min(1.0, p * n / rank)
                
                # 确保单调性（从大到小）
                if rank < n:
                    corrected_p = min(corrected_p, 1.0)
                
                corrected_pvalues[original_index] = corrected_p
            
            # 确保单调性（从后向前）
            for rank_minus_1 in range(n - 2, -1, -1):
                original_index = indexed_pvalues[rank_minus_1][0]
                next_index = indexed_pvalues[rank_minus_1 + 1][0]
                corrected_pvalues[original_index] = min(
                    corrected_pvalues[original_index],
                    corrected_pvalues[next_index]
                )
            
            # 判断是否拒绝
            rejected = [p <= alpha for p in corrected_pvalues]
            
            return MultipleTestingResult(
                original_pvalues=pvalues,
                corrected_pvalues=corrected_pvalues,
                rejected=rejected,
                correction_method=CorrectionMethod.FDR_BH,
                alpha=alpha,
                error_rate_type=ErrorRateType.FDR
            )
            
        except Exception as e:
            self.logger.error(f"Benjamini-Hochberg correction failed: {e}")
            raise
    
    def benjamini_yekutieli_correction(self, pvalues: List[float], alpha: float = 0.05) -> MultipleTestingResult:
        """
        Benjamini-Yekutieli FDR校正
        更保守的FDR方法，不需要独立性假设
        
        Args:
            pvalues: p值列表
            alpha: 显著性水平（FDR水平）
        
        Returns:
            校正结果
        """
        try:
            n = len(pvalues)
            if n == 0:
                raise ValueError("P-values list cannot be empty")
            
            # 计算调和级数和
            harmonic_sum = sum(1 / i for i in range(1, n + 1))
            
            # 创建索引-p值对并排序
            indexed_pvalues = [(i, p) for i, p in enumerate(pvalues)]
            indexed_pvalues.sort(key=lambda x: x[1])
            
            corrected_pvalues = [0.0] * n
            rejected = [False] * n
            
            # BY校正
            for rank_minus_1, (original_index, p) in enumerate(indexed_pvalues):
                rank = rank_minus_1 + 1
                
                # BY调整：p * n * C(n) / rank，其中C(n)是调和级数和
                corrected_p = min(1.0, p * n * harmonic_sum / rank)
                
                corrected_pvalues[original_index] = corrected_p
            
            # 确保单调性
            for rank_minus_1 in range(n - 2, -1, -1):
                original_index = indexed_pvalues[rank_minus_1][0]
                next_index = indexed_pvalues[rank_minus_1 + 1][0]
                corrected_pvalues[original_index] = min(
                    corrected_pvalues[original_index],
                    corrected_pvalues[next_index]
                )
            
            # 判断是否拒绝
            rejected = [p <= alpha for p in corrected_pvalues]
            
            return MultipleTestingResult(
                original_pvalues=pvalues,
                corrected_pvalues=corrected_pvalues,
                rejected=rejected,
                correction_method=CorrectionMethod.FDR_BY,
                alpha=alpha,
                error_rate_type=ErrorRateType.FDR
            )
            
        except Exception as e:
            self.logger.error(f"Benjamini-Yekutieli correction failed: {e}")
            raise


class MultipleTestingCorrectionService:
    """多重检验校正服务 - 统一接口"""
    
    def __init__(self):
        self.fwer_corrector = FWERCorrection()
        self.fdr_corrector = FDRCorrection()
        self.logger = logger
    
    def apply_correction(self, pvalues: List[float], method: CorrectionMethod,
                        alpha: float = 0.05) -> MultipleTestingResult:
        """
        应用多重检验校正
        
        Args:
            pvalues: p值列表
            method: 校正方法
            alpha: 显著性水平
        
        Returns:
            校正结果
        """
        try:
            # 验证输入
            if not pvalues:
                raise ValueError("P-values list cannot be empty")
            
            if any(p < 0 or p > 1 for p in pvalues):
                raise ValueError("All p-values must be between 0 and 1")
            
            if not (0 < alpha < 1):
                raise ValueError("Alpha must be between 0 and 1")
            
            # 根据方法选择校正器
            if method == CorrectionMethod.BONFERRONI:
                return self.fwer_corrector.bonferroni_correction(pvalues, alpha)
            elif method == CorrectionMethod.HOLM:
                return self.fwer_corrector.holm_bonferroni_correction(pvalues, alpha)
            elif method == CorrectionMethod.SIDAK:
                return self.fwer_corrector.sidak_correction(pvalues, alpha)
            elif method == CorrectionMethod.HOCHBERG:
                return self.fwer_corrector.hochberg_correction(pvalues, alpha)
            elif method == CorrectionMethod.FDR_BH:
                return self.fdr_corrector.benjamini_hochberg_correction(pvalues, alpha)
            elif method == CorrectionMethod.FDR_BY:
                return self.fdr_corrector.benjamini_yekutieli_correction(pvalues, alpha)
            elif method == CorrectionMethod.NONE:
                # 不进行校正
                return MultipleTestingResult(
                    original_pvalues=pvalues,
                    corrected_pvalues=pvalues,
                    rejected=[p <= alpha for p in pvalues],
                    correction_method=CorrectionMethod.NONE,
                    alpha=alpha,
                    error_rate_type=ErrorRateType.PER_COMPARISON
                )
            else:
                raise ValueError(f"Unsupported correction method: {method}")
                
        except Exception as e:
            self.logger.error(f"Multiple testing correction failed: {e}")
            raise
    
    def compare_methods(self, pvalues: List[float], alpha: float = 0.05) -> Dict[str, MultipleTestingResult]:
        """
        比较不同校正方法的结果
        
        Args:
            pvalues: p值列表
            alpha: 显著性水平
        
        Returns:
            各方法的校正结果字典
        """
        try:
            results = {}
            
            # FWER方法
            fwer_methods = [
                CorrectionMethod.BONFERRONI,
                CorrectionMethod.HOLM,
                CorrectionMethod.SIDAK,
                CorrectionMethod.HOCHBERG
            ]
            
            # FDR方法
            fdr_methods = [
                CorrectionMethod.FDR_BH,
                CorrectionMethod.FDR_BY
            ]
            
            # 应用各种方法
            for method in fwer_methods + fdr_methods:
                try:
                    results[method.value] = self.apply_correction(pvalues, method, alpha)
                except Exception as e:
                    self.logger.warning(f"Method {method} failed: {e}")
                    continue
            
            # 添加无校正结果作为对比
            results[CorrectionMethod.NONE.value] = self.apply_correction(
                pvalues, CorrectionMethod.NONE, alpha
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Method comparison failed: {e}")
            raise
    
    def recommend_method(self, num_tests: int, study_type: str = "exploratory",
                        independence: bool = True) -> CorrectionMethod:
        """
        根据研究特点推荐校正方法
        
        Args:
            num_tests: 检验数量
            study_type: 研究类型（exploratory/confirmatory）
            independence: 检验是否独立
        
        Returns:
            推荐的校正方法
        """
        try:
            if study_type == "confirmatory":
                # 验证性研究：控制FWER
                if num_tests <= 5:
                    return CorrectionMethod.BONFERRONI
                elif independence:
                    return CorrectionMethod.HOLM
                else:
                    return CorrectionMethod.HOLM
            else:
                # 探索性研究：控制FDR
                if independence:
                    return CorrectionMethod.FDR_BH
                else:
                    return CorrectionMethod.FDR_BY
                    
        except Exception as e:
            self.logger.error(f"Method recommendation failed: {e}")
            return CorrectionMethod.HOLM  # 默认返回Holm方法
    
    def calculate_adjusted_power(self, original_power: float, num_tests: int,
                               method: CorrectionMethod) -> float:
        """
        计算多重检验校正后的统计功效
        
        Args:
            original_power: 原始统计功效
            num_tests: 检验数量
            method: 校正方法
        
        Returns:
            调整后的统计功效
        """
        try:
            if method == CorrectionMethod.BONFERRONI:
                # Bonferroni校正后的功效（近似）
                adjusted_power = 1 - (1 - original_power) ** (1 / num_tests)
            elif method == CorrectionMethod.SIDAK:
                # Šidák校正后的功效
                adjusted_power = 1 - (1 - original_power) ** (1 / num_tests)
            elif method in [CorrectionMethod.HOLM, CorrectionMethod.HOCHBERG]:
                # Holm/Hochberg的功效介于Bonferroni和无校正之间
                bonferroni_power = 1 - (1 - original_power) ** (1 / num_tests)
                adjusted_power = (original_power + bonferroni_power) / 2
            elif method in [CorrectionMethod.FDR_BH, CorrectionMethod.FDR_BY]:
                # FDR方法通常有更高的功效
                adjusted_power = original_power * 0.9  # 近似值
            else:
                adjusted_power = original_power
            
            return min(1.0, max(0.0, adjusted_power))
            
        except Exception as e:
            self.logger.error(f"Adjusted power calculation failed: {e}")
            return original_power


# 全局实例
_correction_service = None

def get_multiple_testing_correction_service() -> MultipleTestingCorrectionService:
    """获取多重检验校正服务实例（单例模式）"""
    global _correction_service
    if _correction_service is None:
        _correction_service = MultipleTestingCorrectionService()
    return _correction_service