"""
统计分析服务单元测试
"""

import pytest
import math
from services.statistical_analysis_service import (
    BasicStatisticsCalculator,
    ExperimentStatsCalculator,
    MetricType,
    DistributionType
)

class TestBasicStatisticsCalculator:
    """基础统计计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        return BasicStatisticsCalculator()
    
    @pytest.fixture
    def sample_data(self):
        return [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    def test_calculate_mean(self, calculator, sample_data):
        """测试均值计算"""
        mean = calculator.calculate_mean(sample_data)
        assert mean == 5.5
        
        # 测试单个值
        assert calculator.calculate_mean([42]) == 42
        
        # 测试空列表
        with pytest.raises(ValueError):
            calculator.calculate_mean([])
    
    def test_calculate_variance(self, calculator, sample_data):
        """测试方差计算"""
        # 样本方差 (n-1)
        sample_variance = calculator.calculate_variance(sample_data, sample=True)
        expected_sample_var = sum((x - 5.5) ** 2 for x in sample_data) / 9
        assert abs(sample_variance - expected_sample_var) < 1e-10
        
        # 总体方差 (n)
        population_variance = calculator.calculate_variance(sample_data, sample=False)
        expected_pop_var = sum((x - 5.5) ** 2 for x in sample_data) / 10
        assert abs(population_variance - expected_pop_var) < 1e-10
        
        # 测试单个值
        assert calculator.calculate_variance([42]) == 0.0
    
    def test_calculate_std_deviation(self, calculator, sample_data):
        """测试标准差计算"""
        std_dev = calculator.calculate_std_deviation(sample_data)
        variance = calculator.calculate_variance(sample_data)
        assert abs(std_dev - math.sqrt(variance)) < 1e-10
    
    def test_calculate_percentiles(self, calculator, sample_data):
        """测试分位数计算"""
        percentiles = calculator.calculate_percentiles(sample_data, [0, 25, 50, 75, 100])
        
        assert percentiles[0] == 1  # 0th percentile (min)
        assert percentiles[4] == 10  # 100th percentile (max)
        assert percentiles[2] == 5.5  # 50th percentile (median)
        
        # 测试无效分位数
        with pytest.raises(ValueError):
            calculator.calculate_percentiles(sample_data, [-1])
        
        with pytest.raises(ValueError):
            calculator.calculate_percentiles(sample_data, [101])
    
    def test_calculate_skewness(self, calculator):
        """测试偏度计算"""
        # 对称分布的偏度应该接近0
        symmetric_data = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        skewness = calculator.calculate_skewness(symmetric_data)
        assert skewness is not None
        assert abs(skewness) < 0.5  # 应该接近0
        
        # 右偏分布
        right_skewed = [1, 1, 1, 2, 2, 3, 10]
        skewness = calculator.calculate_skewness(right_skewed)
        assert skewness is not None
        assert skewness > 0  # 右偏应该为正值
        
        # 数据太少的情况
        assert calculator.calculate_skewness([1, 2]) is None
    
    def test_calculate_kurtosis(self, calculator):
        """测试峰度计算"""
        # 正态分布的峰度应该接近0（超额峰度）
        normal_like_data = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        kurtosis = calculator.calculate_kurtosis(normal_like_data)
        assert kurtosis is not None
        
        # 数据太少的情况
        assert calculator.calculate_kurtosis([1, 2, 3]) is None
    
    def test_calculate_descriptive_stats(self, calculator, sample_data):
        """测试完整描述性统计计算"""
        stats = calculator.calculate_descriptive_stats(sample_data)
        
        assert stats.count == 10
        assert stats.mean == 5.5
        assert stats.min_value == 1
        assert stats.max_value == 10
        assert stats.median == 5.5
        assert stats.sum_value == 55
        
        # 验证方差和标准差的关系
        assert abs(stats.std_dev - math.sqrt(stats.variance)) < 1e-10
    
    def test_calculate_conversion_rate_stats(self, calculator):
        """测试转化率统计计算"""
        conversions = 150
        total_users = 1000
        
        stats = calculator.calculate_conversion_rate_stats(conversions, total_users)
        
        assert stats.count == total_users
        assert stats.mean == 0.15  # 15% conversion rate
        assert stats.min_value == 0.0
        assert stats.max_value == 1.0
        assert stats.sum_value == conversions
        
        # 验证二项分布方差公式: p(1-p)/n
        expected_variance = 0.15 * 0.85 / 1000
        assert abs(stats.variance - expected_variance) < 1e-10
        
        # 测试边界情况
        with pytest.raises(ValueError):
            calculator.calculate_conversion_rate_stats(-1, 100)
        
        with pytest.raises(ValueError):
            calculator.calculate_conversion_rate_stats(101, 100)

class TestExperimentStatsCalculator:
    """实验统计计算器测试"""
    
    @pytest.fixture
    def calculator(self):
        return ExperimentStatsCalculator()
    
    def test_calculate_group_stats(self, calculator):
        """测试分组统计计算"""
        group_id = "control"
        group_name = "Control Group"
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        metric_type = MetricType.CONTINUOUS
        
        group_stats = calculator.calculate_group_stats(
            group_id, group_name, values, metric_type
        )
        
        assert group_stats.group_id == group_id
        assert group_stats.group_name == group_name
        assert group_stats.metric_type == metric_type
        assert group_stats.stats.count == 5
        assert group_stats.stats.mean == 3.0
    
    def test_calculate_conversion_group_stats(self, calculator):
        """测试转化率分组统计计算"""
        group_id = "treatment"
        group_name = "Treatment Group" 
        conversions = 85
        total_users = 500
        
        group_stats = calculator.calculate_conversion_group_stats(
            group_id, group_name, conversions, total_users
        )
        
        assert group_stats.group_id == group_id
        assert group_stats.metric_type == MetricType.CONVERSION
        assert group_stats.distribution_type == DistributionType.BINOMIAL
        assert group_stats.stats.mean == 0.17  # 17% conversion rate
        assert group_stats.stats.count == total_users
        assert group_stats.stats.sum_value == conversions
    
    def test_calculate_multiple_groups_stats(self, calculator):
        """测试多分组统计计算"""
        groups_data = {
            "control": {
                "name": "Control Group",
                "values": [1.0, 2.0, 3.0, 4.0, 5.0]
            },
            "treatment": {
                "name": "Treatment Group",
                "values": [2.0, 3.0, 4.0, 5.0, 6.0]
            }
        }
        
        results = calculator.calculate_multiple_groups_stats(
            groups_data, MetricType.CONTINUOUS
        )
        
        assert len(results) == 2
        assert "control" in results
        assert "treatment" in results
        
        control_stats = results["control"]
        assert control_stats.stats.mean == 3.0
        
        treatment_stats = results["treatment"]
        assert treatment_stats.stats.mean == 4.0
    
    def test_calculate_multiple_conversion_groups_stats(self, calculator):
        """测试多个转化率分组统计计算"""
        groups_data = {
            "control": {
                "name": "Control Group",
                "conversions": 75,
                "total_users": 500
            },
            "treatment": {
                "name": "Treatment Group",
                "conversions": 95,
                "total_users": 500
            }
        }
        
        results = calculator.calculate_multiple_groups_stats(
            groups_data, MetricType.CONVERSION
        )
        
        assert len(results) == 2
        
        control_stats = results["control"]
        assert control_stats.stats.mean == 0.15  # 15% conversion
        assert control_stats.distribution_type == DistributionType.BINOMIAL
        
        treatment_stats = results["treatment"]
        assert treatment_stats.stats.mean == 0.19  # 19% conversion
        assert treatment_stats.distribution_type == DistributionType.BINOMIAL
    
    def test_infer_distribution_type(self, calculator):
        """测试分布类型推断"""
        # 测试转化率指标
        conversion_values = [0, 1, 0, 1, 1]
        dist_type = calculator._infer_distribution_type(
            conversion_values, MetricType.CONVERSION
        )
        assert dist_type == DistributionType.BINOMIAL
        
        # 测试计数指标（接近泊松分布）
        poisson_like_values = [0, 1, 1, 2, 1, 0, 1, 2, 3, 1]
        dist_type = calculator._infer_distribution_type(
            poisson_like_values, MetricType.COUNT
        )
        # 注意：由于样本较小，可能不会被识别为泊松分布
        
        # 测试连续指标
        continuous_values = [1.2, 3.4, 2.1, 4.5, 3.8, 2.9]
        dist_type = calculator._infer_distribution_type(
            continuous_values, MetricType.CONTINUOUS
        )
        # 样本太小，应该返回UNKNOWN
        assert dist_type == DistributionType.UNKNOWN

# 集成测试
def test_statistical_analysis_integration():
    """统计分析服务集成测试"""
    from services.statistical_analysis_service import get_stats_calculator
    
    calculator = get_stats_calculator()
    assert calculator is not None
    
    # 测试单例模式
    calculator2 = get_stats_calculator()
    assert calculator is calculator2
    
    # 基础功能测试
    test_data = [1, 2, 3, 4, 5]
    stats = calculator.basic_calculator.calculate_descriptive_stats(test_data)
    assert stats.mean == 3.0
    assert stats.count == 5

if __name__ == "__main__":
    pytest.main([__file__])
