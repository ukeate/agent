"""
统计分析服务单元测试
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
from scipy import stats
from services.statistical_analysis_service import (
    StatisticalAnalysisService,
    TestType,
    CorrectionMethod
)

@pytest.fixture
def stats_service():
    """创建统计分析服务实例"""
    return StatisticalAnalysisService()

@pytest.fixture
def sample_data():
    """生成示例数据"""
    np.random.seed(42)
    return {
        "control": np.random.normal(100, 15, 1000),
        "treatment": np.random.normal(105, 15, 1000),
        "small_control": np.random.normal(100, 15, 30),
        "small_treatment": np.random.normal(110, 15, 30)
    }

class TestBasicStatistics:
    """基础统计测试"""
    
    def test_calculate_mean(self, stats_service, sample_data):
        """测试均值计算"""
        mean = stats_service.calculate_mean(sample_data["control"])
        assert abs(mean - 100) < 1  # 接近100
        assert isinstance(mean, float)
    
    def test_calculate_variance(self, stats_service, sample_data):
        """测试方差计算"""
        variance = stats_service.calculate_variance(sample_data["control"])
        assert abs(variance - 225) < 20  # 接近15^2
        assert variance > 0
    
    def test_calculate_std(self, stats_service, sample_data):
        """测试标准差计算"""
        std = stats_service.calculate_std(sample_data["control"])
        assert abs(std - 15) < 1  # 接近15
        assert std > 0
    
    def test_calculate_median(self, stats_service, sample_data):
        """测试中位数计算"""
        median = stats_service.calculate_median(sample_data["control"])
        assert abs(median - 100) < 2
    
    def test_calculate_percentiles(self, stats_service, sample_data):
        """测试百分位数计算"""
        p25, p50, p75 = stats_service.calculate_percentiles(
            sample_data["control"],
            [25, 50, 75]
        )
        
        assert p25 < p50 < p75
        assert abs(p50 - 100) < 2  # 中位数接近100

class TestHypothesisTesting:
    """假设检验测试"""
    
    @pytest.mark.asyncio
    async def test_t_test_independent(self, stats_service, sample_data):
        """测试独立样本t检验"""
        result = await stats_service.t_test(
            sample_data["control"],
            sample_data["treatment"],
            test_type=TestType.TWO_SIDED
        )
        
        assert "t_statistic" in result
        assert "p_value" in result
        assert "degrees_of_freedom" in result
        assert result["p_value"] < 0.05  # 应该显著
    
    @pytest.mark.asyncio
    async def test_t_test_one_sided(self, stats_service, sample_data):
        """测试单侧t检验"""
        # 右尾检验
        result = await stats_service.t_test(
            sample_data["control"],
            sample_data["treatment"],
            test_type=TestType.GREATER
        )
        assert result["p_value"] < 0.05
        
        # 左尾检验
        result = await stats_service.t_test(
            sample_data["treatment"],
            sample_data["control"],
            test_type=TestType.LESS
        )
        assert result["p_value"] < 0.05
    
    @pytest.mark.asyncio
    async def test_welch_t_test(self, stats_service):
        """测试Welch's t检验（不等方差）"""
        # 创建不等方差数据
        np.random.seed(42)
        control = np.random.normal(100, 10, 1000)
        treatment = np.random.normal(105, 20, 1000)  # 更大的方差
        
        result = await stats_service.t_test(
            control, treatment,
            equal_variance=False
        )
        
        assert result["test_type"] == "Welch's t-test"
        assert result["p_value"] < 0.05
    
    @pytest.mark.asyncio
    async def test_chi_square_test(self, stats_service):
        """测试卡方检验"""
        # 创建列联表
        observed = np.array([[100, 200], [150, 250]])
        
        result = await stats_service.chi_square_test(observed)
        
        assert "chi2_statistic" in result
        assert "p_value" in result
        assert "degrees_of_freedom" in result
        assert result["degrees_of_freedom"] == 1
    
    @pytest.mark.asyncio
    async def test_mann_whitney_u_test(self, stats_service, sample_data):
        """测试Mann-Whitney U检验（非参数）"""
        result = await stats_service.mann_whitney_u_test(
            sample_data["control"],
            sample_data["treatment"]
        )
        
        assert "u_statistic" in result
        assert "p_value" in result
        assert result["p_value"] < 0.05
    
    @pytest.mark.asyncio
    async def test_anova(self, stats_service):
        """测试方差分析"""
        np.random.seed(42)
        group1 = np.random.normal(100, 15, 100)
        group2 = np.random.normal(105, 15, 100)
        group3 = np.random.normal(110, 15, 100)
        
        result = await stats_service.anova([group1, group2, group3])
        
        assert "f_statistic" in result
        assert "p_value" in result
        assert result["p_value"] < 0.05  # 组间差异显著

class TestConfidenceIntervals:
    """置信区间测试"""
    
    @pytest.mark.asyncio
    async def test_confidence_interval_mean(self, stats_service, sample_data):
        """测试均值置信区间"""
        ci = await stats_service.confidence_interval_mean(
            sample_data["control"],
            confidence_level=0.95
        )
        
        assert len(ci) == 2
        assert ci[0] < 100 < ci[1]  # 真实均值应在区间内
        assert ci[1] - ci[0] < 2  # 大样本区间应较窄
    
    @pytest.mark.asyncio
    async def test_confidence_interval_difference(self, stats_service, sample_data):
        """测试均值差异置信区间"""
        ci = await stats_service.confidence_interval_difference(
            sample_data["control"],
            sample_data["treatment"],
            confidence_level=0.95
        )
        
        assert len(ci) == 2
        assert ci[0] < 0  # treatment均值更高，差异为负
        assert not (ci[0] <= 0 <= ci[1])  # 0不在区间内表示显著
    
    @pytest.mark.asyncio
    async def test_confidence_interval_proportion(self, stats_service):
        """测试比例置信区间"""
        successes = 450
        trials = 1000
        
        ci = await stats_service.confidence_interval_proportion(
            successes, trials,
            confidence_level=0.95
        )
        
        assert len(ci) == 2
        assert ci[0] < 0.45 < ci[1]
        assert ci[1] - ci[0] < 0.1  # 合理的区间宽度
    
    @pytest.mark.asyncio
    async def test_bootstrap_confidence_interval(self, stats_service, sample_data):
        """测试Bootstrap置信区间"""
        ci = await stats_service.bootstrap_confidence_interval(
            sample_data["small_control"],
            statistic="mean",
            confidence_level=0.95,
            n_bootstrap=1000
        )
        
        assert len(ci) == 2
        assert ci[0] < np.mean(sample_data["small_control"]) < ci[1]

class TestPowerAnalysis:
    """功效分析测试"""
    
    @pytest.mark.asyncio
    async def test_calculate_power(self, stats_service):
        """测试统计功效计算"""
        power = await stats_service.calculate_power(
            effect_size=0.5,
            sample_size=100,
            alpha=0.05
        )
        
        assert 0 < power < 1
        assert power > 0.8  # 中等效应量应有较高功效
    
    @pytest.mark.asyncio
    async def test_calculate_sample_size(self, stats_service):
        """测试样本量计算"""
        n = await stats_service.calculate_sample_size(
            effect_size=0.5,
            power=0.8,
            alpha=0.05
        )
        
        assert n > 0
        assert isinstance(n, int)
        assert n < 200  # 中等效应量不需要太大样本
    
    @pytest.mark.asyncio
    async def test_calculate_effect_size(self, stats_service, sample_data):
        """测试效应量计算"""
        # Cohen's d
        effect_size = await stats_service.calculate_effect_size(
            sample_data["control"],
            sample_data["treatment"],
            effect_type="cohens_d"
        )
        
        assert effect_size > 0
        assert 0.2 < effect_size < 0.5  # 小到中等效应
    
    @pytest.mark.asyncio
    async def test_minimum_detectable_effect(self, stats_service):
        """测试最小可检测效应"""
        mde = await stats_service.calculate_mde(
            baseline_rate=0.1,
            sample_size=1000,
            power=0.8,
            alpha=0.05
        )
        
        assert mde > 0
        assert mde < 0.05  # 合理的MDE

class TestMultipleTestingCorrection:
    """多重检验校正测试"""
    
    @pytest.mark.asyncio
    async def test_bonferroni_correction(self, stats_service):
        """测试Bonferroni校正"""
        p_values = [0.01, 0.04, 0.03, 0.20]
        
        corrected = await stats_service.multiple_testing_correction(
            p_values,
            method=CorrectionMethod.BONFERRONI,
            alpha=0.05
        )
        
        assert len(corrected["adjusted_p_values"]) == 4
        assert all(adj >= orig for adj, orig in 
                  zip(corrected["adjusted_p_values"], p_values))
        assert corrected["rejected"][0] is False  # 0.01 * 4 = 0.04 < 0.05
    
    @pytest.mark.asyncio
    async def test_benjamini_hochberg_correction(self, stats_service):
        """测试Benjamini-Hochberg校正"""
        p_values = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205]
        
        corrected = await stats_service.multiple_testing_correction(
            p_values,
            method=CorrectionMethod.BENJAMINI_HOCHBERG,
            alpha=0.05
        )
        
        assert len(corrected["adjusted_p_values"]) == len(p_values)
        # BH方法应该比Bonferroni更宽松
        assert sum(corrected["rejected"]) >= 3
    
    @pytest.mark.asyncio
    async def test_holm_bonferroni_correction(self, stats_service):
        """测试Holm-Bonferroni校正"""
        p_values = [0.01, 0.04, 0.03, 0.20]
        
        corrected = await stats_service.multiple_testing_correction(
            p_values,
            method=CorrectionMethod.HOLM_BONFERRONI,
            alpha=0.05
        )
        
        assert len(corrected["adjusted_p_values"]) == 4
        # Holm方法应该比标准Bonferroni更强大
        assert sum(corrected["rejected"]) >= sum(
            p < 0.05/4 for p in p_values
        )

class TestVariantComparison:
    """变体比较测试"""
    
    @pytest.mark.asyncio
    async def test_compare_two_variants(self, stats_service, sample_data):
        """测试两个变体比较"""
        result = await stats_service.compare_variants(
            control_data=sample_data["control"],
            treatment_data=sample_data["treatment"],
            metric_type="continuous"
        )
        
        assert "difference" in result
        assert "relative_difference" in result
        assert "p_value" in result
        assert "confidence_interval" in result
        assert "significant" in result
        assert result["significant"] is True
    
    @pytest.mark.asyncio
    async def test_compare_multiple_variants(self, stats_service):
        """测试多个变体比较"""
        np.random.seed(42)
        variants = {
            "control": np.random.normal(100, 15, 1000),
            "variant_a": np.random.normal(105, 15, 1000),
            "variant_b": np.random.normal(103, 15, 1000)
        }
        
        result = await stats_service.compare_multiple_variants(
            variants,
            baseline="control"
        )
        
        assert len(result["comparisons"]) == 2
        assert "variant_a" in result["comparisons"]
        assert "variant_b" in result["comparisons"]
        assert result["overall_p_value"] < 0.05
    
    @pytest.mark.asyncio
    async def test_compare_proportions(self, stats_service):
        """测试比例比较"""
        result = await stats_service.compare_proportions(
            control_successes=450,
            control_trials=1000,
            treatment_successes=500,
            treatment_trials=1000
        )
        
        assert "control_rate" in result
        assert "treatment_rate" in result
        assert "absolute_difference" in result
        assert "relative_difference" in result
        assert "p_value" in result
        assert result["p_value"] < 0.05

class TestNormality:
    """正态性检验测试"""
    
    @pytest.mark.asyncio
    async def test_shapiro_wilk_test(self, stats_service):
        """测试Shapiro-Wilk正态性检验"""
        # 正态分布数据
        normal_data = np.random.normal(0, 1, 100)
        result = await stats_service.test_normality(
            normal_data,
            method="shapiro"
        )
        assert result["p_value"] > 0.05  # 不拒绝正态性
        
        # 非正态分布数据
        exponential_data = np.random.exponential(1, 100)
        result = await stats_service.test_normality(
            exponential_data,
            method="shapiro"
        )
        assert result["p_value"] < 0.05  # 拒绝正态性
    
    @pytest.mark.asyncio
    async def test_anderson_darling_test(self, stats_service):
        """测试Anderson-Darling正态性检验"""
        normal_data = np.random.normal(0, 1, 100)
        result = await stats_service.test_normality(
            normal_data,
            method="anderson"
        )
        
        assert "statistic" in result
        assert "critical_values" in result
        assert "significance_levels" in result

class TestCorrelation:
    """相关性分析测试"""
    
    @pytest.mark.asyncio
    async def test_pearson_correlation(self, stats_service):
        """测试Pearson相关系数"""
        x = np.random.normal(0, 1, 100)
        y = 2 * x + np.random.normal(0, 0.5, 100)  # 强正相关
        
        result = await stats_service.calculate_correlation(
            x, y,
            method="pearson"
        )
        
        assert -1 <= result["correlation"] <= 1
        assert result["correlation"] > 0.8  # 强相关
        assert result["p_value"] < 0.05
    
    @pytest.mark.asyncio
    async def test_spearman_correlation(self, stats_service):
        """测试Spearman秩相关系数"""
        x = np.random.normal(0, 1, 100)
        y = x ** 2  # 非线性关系
        
        result = await stats_service.calculate_correlation(
            x, y,
            method="spearman"
        )
        
        assert -1 <= result["correlation"] <= 1
        assert "p_value" in result

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
