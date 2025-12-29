"""
假设检验服务单元测试
"""

import pytest
import math
from services.hypothesis_testing_service import (
    HypothesisTestingService,
    TTestCalculator,
    ChiSquareTestCalculator,
    HypothesisType,
    get_hypothesis_testing_service
)
from services.statistical_analysis_service import MetricType

class TestTTestCalculator:
    """t检验计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        return TTestCalculator()
    
    def test_one_sample_t_test(self, calculator):
        """测试单样本t检验"""
        # 测试数据：样本均值明显不等于总体均值
        sample = [5.1, 5.3, 4.9, 5.2, 5.0, 5.1, 4.8, 5.2, 5.0, 5.1]
        population_mean = 4.0
        
        result = calculator.one_sample_t_test(
            sample=sample,
            population_mean=population_mean,
            hypothesis_type=HypothesisType.TWO_SIDED,
            alpha=0.05
        )
        
        assert result.test_type == "one_sample_t_test"
        assert result.degrees_of_freedom == 9
        assert result.statistic > 0  # 样本均值大于总体均值
        assert result.p_value < 0.05  # 应该显著
        assert result.is_significant
        assert result.effect_size > 0
    
    def test_independent_two_sample_t_test_equal_variances(self, calculator):
        """测试独立双样本t检验（等方差）"""
        sample1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample2 = [3.0, 4.0, 5.0, 6.0, 7.0]
        
        result = calculator.independent_two_sample_t_test(
            sample1=sample1,
            sample2=sample2,
            equal_variances=True,
            hypothesis_type=HypothesisType.TWO_SIDED,
            alpha=0.05
        )
        
        assert result.test_type == "independent_two_sample_t_test"
        assert result.degrees_of_freedom == 8  # n1 + n2 - 2
        assert result.statistic < 0  # sample1 < sample2
        assert result.is_significant  # 差异应该显著
        assert result.confidence_interval is not None
    
    def test_independent_two_sample_t_test_unequal_variances(self, calculator):
        """测试独立双样本t检验（不等方差，Welch检验）"""
        sample1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        sample2 = [10.0, 20.0, 30.0, 40.0, 50.0]  # 更大的方差
        
        result = calculator.independent_two_sample_t_test(
            sample1=sample1,
            sample2=sample2,
            equal_variances=False,
            hypothesis_type=HypothesisType.TWO_SIDED,
            alpha=0.05
        )
        
        assert result.test_type == "welch_t_test"
        assert result.statistic < 0  # sample1 < sample2
        assert result.is_significant  # 差异应该显著
    
    def test_paired_t_test(self, calculator):
        """测试配对t检验"""
        before = [10, 12, 11, 13, 15, 14, 16, 12, 11, 13]
        after = [12, 14, 13, 15, 17, 16, 18, 14, 13, 15]
        
        result = calculator.paired_t_test(
            sample1=before,
            sample2=after,
            hypothesis_type=HypothesisType.TWO_SIDED,
            alpha=0.05
        )
        
        assert result.test_type == "one_sample_t_test"  # 配对t检验实际上是差值的单样本t检验
        assert result.statistic < 0  # before < after
        assert result.is_significant  # 差异应该显著
    
    def test_edge_cases(self, calculator):
        """测试边界情况"""
        # 样本太小
        with pytest.raises(ValueError):
            calculator.one_sample_t_test([1], 0)
        
        # 配对样本长度不一致
        with pytest.raises(ValueError):
            calculator.paired_t_test([1, 2], [1, 2, 3])
        
        # 标准差为零
        with pytest.raises(ValueError):
            calculator.one_sample_t_test([5, 5, 5, 5], 0)

class TestChiSquareTestCalculator:
    """卡方检验计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        return ChiSquareTestCalculator()
    
    def test_goodness_of_fit_test(self, calculator):
        """测试卡方拟合优度检验"""
        observed = [16, 18, 16, 14, 12, 12]  # 骰子投掷结果
        expected = [15, 15, 15, 15, 15, 15]  # 均匀分布期望
        
        result = calculator.goodness_of_fit_test(
            observed=observed,
            expected=expected,
            alpha=0.05
        )
        
        assert result.test_type == "chi_square_goodness_of_fit"
        assert result.degrees_of_freedom == 5
        assert result.statistic >= 0
        assert result.p_value >= 0 and result.p_value <= 1
        assert result.hypothesis_type == HypothesisType.GREATER
    
    def test_independence_test(self, calculator):
        """测试卡方独立性检验"""
        # 2x2列联表：性别 vs 产品偏好
        contingency_table = [
            [20, 30],  # 男性：喜欢产品A=20，喜欢产品B=30
            [25, 25]   # 女性：喜欢产品A=25，喜欢产品B=25
        ]
        
        result = calculator.independence_test(
            contingency_table=contingency_table,
            alpha=0.05
        )
        
        assert result.test_type == "chi_square_independence"
        assert result.degrees_of_freedom == 1  # (2-1)*(2-1)
        assert result.statistic >= 0
        assert result.effect_size is not None  # Cramér's V
        assert result.hypothesis_type == HypothesisType.GREATER
    
    def test_proportion_test(self, calculator):
        """测试两比例卡方检验"""
        successes1 = 85  # 对照组转化
        total1 = 500     # 对照组总数
        successes2 = 95  # 实验组转化
        total2 = 500     # 实验组总数
        
        result = calculator.proportion_test(
            successes1=successes1,
            total1=total1,
            successes2=successes2,
            total2=total2,
            hypothesis_type=HypothesisType.TWO_SIDED,
            alpha=0.05
        )
        
        assert result.test_type == "two_proportion_chi_square"
        assert result.statistic >= 0
        # 计算期望的比例差异
        p1 = successes1 / total1  # 0.17
        p2 = successes2 / total2  # 0.19
        assert abs(p2 - p1) == 0.02  # 2%的差异
    
    def test_edge_cases(self, calculator):
        """测试边界情况"""
        # 期望频数不匹配
        with pytest.raises(ValueError):
            calculator.goodness_of_fit_test([1, 2], [1])
        
        # 期望频数为负或零
        with pytest.raises(ValueError):
            calculator.goodness_of_fit_test([1, 2], [1, -1])
        
        # 列联表太小
        with pytest.raises(ValueError):
            calculator.independence_test([[1]])
        
        # 成功数超过总数
        with pytest.raises(ValueError):
            calculator.proportion_test(101, 100, 50, 100)

class TestHypothesisTestingService:
    """假设检验服务集成测试"""
    
    @pytest.fixture
    def service(self):
        return HypothesisTestingService()
    
    def test_compare_two_groups_conversion(self, service):
        """测试转化率两组比较"""
        control_group = {
            "conversions": 75,
            "total_users": 500
        }
        treatment_group = {
            "conversions": 95,
            "total_users": 500
        }
        
        result = service.compare_two_groups(
            group1_data=control_group,
            group2_data=treatment_group,
            metric_type=MetricType.CONVERSION,
            hypothesis_type=HypothesisType.TWO_SIDED,
            alpha=0.05
        )
        
        assert result.test_type == "two_proportion_chi_square"
        assert result.statistic >= 0
        # 检查比例差异：19% vs 15% = 4%的差异应该不显著（样本不够大）
    
    def test_compare_two_groups_continuous(self, service):
        """测试连续指标两组比较"""
        control_group = {
            "values": [1.0, 2.0, 3.0, 4.0, 5.0, 2.5, 3.5, 4.5, 3.0, 4.0]
        }
        treatment_group = {
            "values": [3.0, 4.0, 5.0, 6.0, 7.0, 4.5, 5.5, 6.5, 5.0, 6.0]
        }
        
        result = service.compare_two_groups(
            group1_data=control_group,
            group2_data=treatment_group,
            metric_type=MetricType.CONTINUOUS,
            hypothesis_type=HypothesisType.TWO_SIDED,
            alpha=0.05
        )
        
        assert result.test_type in ["independent_two_sample_t_test", "welch_t_test"]
        assert result.statistic < 0  # control < treatment
        assert result.is_significant  # 差异应该显著
    
    def test_run_t_test_methods(self, service):
        """测试不同类型的t检验"""
        sample1 = [1, 2, 3, 4, 5]
        sample2 = [3, 4, 5, 6, 7]
        
        # 单样本t检验
        result1 = service.run_t_test(
            test_type="one_sample",
            sample=sample1,
            population_mean=0,
            hypothesis_type=HypothesisType.TWO_SIDED,
            alpha=0.05
        )
        assert result1.test_type == "one_sample_t_test"
        assert result1.is_significant
        
        # 独立双样本t检验
        result2 = service.run_t_test(
            test_type="independent_two_sample",
            sample1=sample1,
            sample2=sample2,
            hypothesis_type=HypothesisType.TWO_SIDED,
            alpha=0.05
        )
        assert result2.test_type == "independent_two_sample_t_test"
        
        # 配对t检验
        result3 = service.run_t_test(
            test_type="paired",
            sample1=sample1,
            sample2=sample2,
            hypothesis_type=HypothesisType.TWO_SIDED,
            alpha=0.05
        )
        assert result3.test_type == "one_sample_t_test"
    
    def test_run_chi_square_test_methods(self, service):
        """测试不同类型的卡方检验"""
        # 拟合优度检验
        result1 = service.run_chi_square_test(
            test_type="goodness_of_fit",
            observed=[10, 15, 20, 25],
            expected=[17.5, 17.5, 17.5, 17.5],
            alpha=0.05
        )
        assert result1.test_type == "chi_square_goodness_of_fit"
        
        # 独立性检验
        result2 = service.run_chi_square_test(
            test_type="independence",
            contingency_table=[[10, 20], [15, 25]],
            alpha=0.05
        )
        assert result2.test_type == "chi_square_independence"
        
        # 比例检验
        result3 = service.run_chi_square_test(
            test_type="proportion",
            successes1=50,
            total1=200,
            successes2=75,
            total2=200,
            hypothesis_type=HypothesisType.TWO_SIDED,
            alpha=0.05
        )
        assert result3.test_type == "two_proportion_chi_square"
    
    def test_invalid_test_types(self, service):
        """测试无效的检验类型"""
        with pytest.raises(ValueError):
            service.run_t_test(test_type="invalid", sample=[1, 2, 3])
        
        with pytest.raises(ValueError):
            service.run_chi_square_test(test_type="invalid", observed=[1, 2])

# 集成测试
def test_hypothesis_testing_service_integration():
    """假设检验服务集成测试"""
    service = get_hypothesis_testing_service()
    assert service is not None
    
    # 测试单例模式
    service2 = get_hypothesis_testing_service()
    assert service is service2
    
    # 基础功能测试
    test_data1 = [1, 2, 3, 4, 5]
    test_data2 = [3, 4, 5, 6, 7]
    
    result = service.run_t_test(
        test_type="independent_two_sample",
        sample1=test_data1,
        sample2=test_data2,
        hypothesis_type=HypothesisType.TWO_SIDED,
        alpha=0.05
    )
    
    assert result.is_significant
    assert result.effect_size > 0

# 实际A/B测试场景测试
def test_ab_test_scenarios():
    """测试真实A/B测试场景"""
    service = get_hypothesis_testing_service()
    
    # 场景1：转化率提升显著
    significant_improvement = service.compare_two_groups(
        group1_data={"conversions": 100, "total_users": 1000},  # 10%
        group2_data={"conversions": 130, "total_users": 1000},  # 13%
        metric_type=MetricType.CONVERSION,
        alpha=0.05
    )
    
    # 场景2：转化率提升不显著
    insignificant_improvement = service.compare_two_groups(
        group1_data={"conversions": 100, "total_users": 1000},  # 10%
        group2_data={"conversions": 105, "total_users": 1000},  # 10.5%
        metric_type=MetricType.CONVERSION,
        alpha=0.05
    )
    
    # 验证结果符合预期
    # 注意：具体的显著性取决于样本量和效应大小
    assert significant_improvement.p_value <= insignificant_improvement.p_value

if __name__ == "__main__":
    pytest.main([__file__])
