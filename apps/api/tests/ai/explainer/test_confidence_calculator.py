"""置信度计算器单元测试"""

import pytest
from unittest.mock import patch
from datetime import datetime, timezone

from src.ai.explainer.confidence_calculator import ConfidenceCalculator
from src.ai.explainer.uncertainty_quantifier import UncertaintyQuantifier
from src.models.schemas.explanation import ConfidenceSource


class TestConfidenceCalculator:
    """测试置信度计算器"""
    
    @pytest.fixture
    def calculator(self):
        """创建测试用的置信度计算器"""
        return ConfidenceCalculator()
    
    @pytest.fixture
    def sample_model_prediction(self):
        """样本模型预测数据"""
        return {
            "probability": 0.85,
            "logits": [2.1, -0.5, 1.2],
            "softmax": [0.7, 0.1, 0.2],
            "feature_importance": {
                "age": 0.4,
                "income": 0.3,
                "credit_score": 0.2,
                "employment": 0.1
            },
            "uncertainty": 0.15
        }
    
    @pytest.fixture
    def sample_evidence_data(self):
        """样本证据数据"""
        return [
            {
                "factor_name": "age",
                "weight": 0.8,
                "reliability_score": 0.9,
                "relevance_score": 0.8,
                "freshness_score": 0.95,
                "source": "user_profile",
                "supporting_evidence_count": 2,
                "contradictory_evidence_count": 0
            },
            {
                "factor_name": "income",
                "weight": 0.7,
                "reliability_score": 0.85,
                "relevance_score": 0.75,
                "freshness_score": 0.9,
                "source": "financial_data",
                "supporting_evidence_count": 1,
                "contradictory_evidence_count": 1
            },
            {
                "factor_name": "credit_score",
                "weight": 0.9,
                "reliability_score": 0.95,
                "relevance_score": 0.9,
                "freshness_score": 0.8,
                "source": "credit_bureau",
                "supporting_evidence_count": 3,
                "contradictory_evidence_count": 0
            }
        ]
    
    @pytest.fixture
    def sample_historical_performance(self):
        """样本历史性能数据"""
        # 使用当前时间避免时间衰减影响
        from datetime import datetime, timezone
        current_time = datetime.now(timezone.utc).isoformat()
        return {
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.85,
            "f1_score": 0.86,
            "sample_count": 1500,  # 增加样本数以提高置信度
            "last_update": current_time
        }
    
    def test_create_confidence_calculator(self, calculator):
        """测试创建置信度计算器"""
        assert calculator is not None
        assert len(calculator.calculation_history) == 0
        assert len(calculator.calibration_data) == 0
        assert ConfidenceSource.MODEL_PROBABILITY in calculator.confidence_weights
        assert calculator.confidence_weights[ConfidenceSource.MODEL_PROBABILITY] == 0.3
    
    def test_calculate_model_confidence(self, calculator, sample_model_prediction):
        """测试计算模型置信度"""
        confidence = calculator._calculate_model_confidence(sample_model_prediction)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # 高概率预测应该有较高置信度
    
    def test_calculate_model_confidence_empty(self, calculator):
        """测试空模型预测的置信度计算"""
        confidence = calculator._calculate_model_confidence({})
        assert confidence == 0.5  # 默认置信度
    
    def test_calculate_evidence_confidence(self, calculator, sample_evidence_data):
        """测试计算证据置信度"""
        confidence = calculator._calculate_evidence_confidence(sample_evidence_data)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # 高质量证据应该有较高置信度
    
    def test_calculate_evidence_confidence_empty(self, calculator):
        """测试空证据数据的置信度计算"""
        confidence = calculator._calculate_evidence_confidence([])
        assert confidence == 0.3  # 低置信度，因为没有证据
    
    def test_calculate_feature_importance_confidence(self, calculator, sample_model_prediction, sample_evidence_data):
        """测试计算特征重要性置信度"""
        confidence = calculator._calculate_feature_importance_confidence(
            sample_model_prediction, sample_evidence_data
        )
        
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_historical_confidence(self, calculator, sample_historical_performance):
        """测试计算历史置信度"""
        confidence = calculator._calculate_historical_confidence(sample_historical_performance)
        
        assert 0.0 <= confidence <= 1.0
        assert confidence > 0.5  # 良好的历史性能应该有较高置信度
    
    def test_calculate_historical_confidence_none(self, calculator):
        """测试无历史数据的置信度计算"""
        confidence = calculator._calculate_historical_confidence(None)
        assert confidence == 0.5  # 默认置信度
    
    def test_calculate_consensus_confidence(self, calculator, sample_evidence_data):
        """测试计算一致性置信度"""
        confidence = calculator._calculate_consensus_confidence(sample_evidence_data, None)
        
        assert 0.0 <= confidence <= 1.0
    
    def test_calculate_consensus_confidence_insufficient_evidence(self, calculator):
        """测试证据不足时的一致性置信度"""
        single_evidence = [{"source": "test", "supporting_evidence_count": 1}]
        confidence = calculator._calculate_consensus_confidence(single_evidence, None)
        assert confidence == 0.5  # 默认值
    
    def test_calculate_uncertainty_score(self, calculator, sample_model_prediction, sample_evidence_data):
        """测试计算不确定性分数"""
        uncertainty = calculator._calculate_uncertainty_score(
            sample_model_prediction, sample_evidence_data, None
        )
        
        assert 0.0 <= uncertainty <= 1.0
    
    def test_calculate_confidence_interval(self, calculator):
        """测试计算置信区间"""
        confidence = 0.8
        uncertainty = 0.2
        
        lower, upper = calculator._calculate_confidence_interval(confidence, uncertainty)
        
        assert 0.0 <= lower <= confidence <= upper <= 1.0
        assert upper - lower > 0  # 区间应该有宽度
    
    def test_calculate_variance(self, calculator, sample_evidence_data):
        """测试计算方差"""
        variance = calculator._calculate_variance(sample_evidence_data)
        
        assert variance >= 0  # 方差非负
    
    def test_calculate_variance_empty(self, calculator):
        """测试空数据的方差计算"""
        variance = calculator._calculate_variance([])
        assert variance == 0.0
    
    def test_calculate_calibration_score(self, calculator, sample_historical_performance):
        """测试计算校准分数"""
        predicted_confidence = 0.85
        calibration_score = calculator._calculate_calibration_score(
            predicted_confidence, sample_historical_performance
        )
        
        assert 0.0 <= calibration_score <= 1.0
    
    def test_calculate_gini_coefficient(self, calculator):
        """测试计算基尼系数"""
        # 完全平等分布
        equal_values = [1.0, 1.0, 1.0, 1.0]
        gini_equal = calculator._calculate_gini_coefficient(equal_values)
        assert gini_equal == 0.0
        
        # 不平等分布
        unequal_values = [1.0, 2.0, 3.0, 4.0]
        gini_unequal = calculator._calculate_gini_coefficient(unequal_values)
        assert gini_unequal > 0.0
    
    def test_calculate_confidence_metrics_full(
        self, calculator, sample_model_prediction, sample_evidence_data, sample_historical_performance
    ):
        """测试完整的置信度指标计算"""
        context_factors = {
            "temporal_distance_days": 30,
            "context_similarity": 0.8
        }
        
        metrics = calculator.calculate_confidence_metrics(
            sample_model_prediction,
            sample_evidence_data,
            sample_historical_performance,
            context_factors
        )
        
        # 验证返回的指标结构
        assert metrics.overall_confidence is not None
        assert 0.0 <= metrics.overall_confidence <= 1.0
        assert 0.0 <= metrics.uncertainty_score <= 1.0
        assert metrics.confidence_interval_lower <= metrics.overall_confidence <= metrics.confidence_interval_upper
        assert len(metrics.confidence_sources) > 0
        
        # 验证计算历史被记录
        assert len(calculator.calculation_history) == 1
        
        history_record = calculator.calculation_history[0]
        assert "calculation_id" in history_record
        assert "result_metrics" in history_record
        assert history_record["input_summary"]["evidence_count"] == 3
    
    def test_calculate_confidence_metrics_minimal(self, calculator):
        """测试最小输入的置信度指标计算"""
        minimal_prediction = {"probability": 0.6}
        minimal_evidence = []
        
        metrics = calculator.calculate_confidence_metrics(
            minimal_prediction, minimal_evidence
        )
        
        assert metrics.overall_confidence is not None
        assert 0.0 <= metrics.overall_confidence <= 1.0
        assert metrics.uncertainty_score is not None
    
    def test_calculate_confidence_metrics_error_handling(self, calculator):
        """测试错误处理"""
        # 模拟计算过程中的异常
        with patch.object(calculator, '_calculate_model_confidence', side_effect=Exception("Test error")):
            metrics = calculator.calculate_confidence_metrics({}, [])
            
            # 应该返回保守的默认值
            assert metrics.overall_confidence == 0.5
            assert metrics.uncertainty_score == 0.5
    
    def test_calibrate_confidence(self, calculator):
        """测试置信度校准"""
        predicted_confidence = 0.85
        actual_outcome = True
        context = {"scenario": "test"}
        
        calculator.calibrate_confidence(predicted_confidence, actual_outcome, context)
        
        assert len(calculator.calibration_data) == 1
        calibration_record = calculator.calibration_data[0]
        assert calibration_record["predicted_confidence"] == predicted_confidence
        assert calibration_record["actual_outcome"] is True
        assert calibration_record["context"]["scenario"] == "test"
    
    def test_get_calibration_statistics_no_data(self, calculator):
        """测试无校准数据时的统计信息"""
        stats = calculator.get_calibration_statistics()
        assert stats["status"] == "no_data"
    
    def test_get_calibration_statistics_with_data(self, calculator):
        """测试有校准数据时的统计信息"""
        # 添加一些校准数据
        for i in range(10):
            calculator.calibrate_confidence(0.5 + i * 0.05, i % 2 == 0)
        
        stats = calculator.get_calibration_statistics()
        
        assert "overall_calibration_error" in stats
        assert "bin_calibration_errors" in stats
        assert stats["total_samples"] == 10
        assert 0.0 <= stats["overall_calibration_error"] <= 1.0
    
    def test_confidence_weights_sum(self, calculator):
        """测试置信度权重配置"""
        total_weight = sum(calculator.confidence_weights.values())
        assert abs(total_weight - 1.0) < 0.01  # 权重总和应该接近1
    
    def test_uncertainty_factors_sum(self, calculator):
        """测试不确定性因子配置"""
        total_factor = sum(calculator.uncertainty_factors.values())
        assert abs(total_factor - 1.0) < 0.01  # 因子总和应该接近1


class TestUncertaintyQuantifier:
    """测试不确定性量化器"""
    
    @pytest.fixture
    def quantifier(self):
        """创建测试用的不确定性量化器"""
        return UncertaintyQuantifier()
    
    def test_create_uncertainty_quantifier(self, quantifier):
        """测试创建不确定性量化器"""
        assert quantifier is not None
        assert len(quantifier.uncertainty_history) == 0
        assert quantifier.monte_carlo_samples == 1000
        assert 0.95 in quantifier.confidence_levels
    
    def test_quantify_aleatoric_uncertainty(self, quantifier):
        """测试量化偶然不确定性"""
        predictions = [0.8, 0.85, 0.75, 0.9, 0.82]
        labels = [0.8, 0.9, 0.7, 0.85, 0.8]
        
        result = quantifier.quantify_aleatoric_uncertainty(predictions, labels)
        
        assert "aleatoric_uncertainty" in result
        assert "variance" in result
        assert "residual_variance" in result
        assert 0.0 <= result["aleatoric_uncertainty"] <= 1.0
        assert result["source"] == "data_noise"
    
    def test_quantify_aleatoric_uncertainty_no_labels(self, quantifier):
        """测试无标签时的偶然不确定性量化"""
        predictions = [0.8, 0.85, 0.75, 0.9]
        
        result = quantifier.quantify_aleatoric_uncertainty(predictions)
        
        assert "aleatoric_uncertainty" in result
        assert result["residual_variance"] == 0.0
    
    def test_quantify_epistemic_uncertainty(self, quantifier):
        """测试量化认知不确定性"""
        ensemble_predictions = [
            [0.8, 0.7, 0.9],
            [0.85, 0.75, 0.85],
            [0.75, 0.8, 0.95]
        ]
        
        feature_importance_variations = [
            {"feature1": 0.5, "feature2": 0.3},
            {"feature1": 0.6, "feature2": 0.25},
            {"feature1": 0.45, "feature2": 0.35}
        ]
        
        result = quantifier.quantify_epistemic_uncertainty(
            ensemble_predictions, feature_importance_variations
        )
        
        assert "epistemic_uncertainty" in result
        assert "model_disagreement" in result
        assert "feature_uncertainty" in result
        assert 0.0 <= result["epistemic_uncertainty"] <= 1.0
        assert result["source"] == "model_knowledge"
    
    def test_monte_carlo_dropout_uncertainty(self, quantifier):
        """测试Monte Carlo Dropout不确定性"""
        forward_passes = [
            {"prediction": 0.8},
            {"prediction": 0.85},
            {"prediction": 0.75},
            {"prediction": 0.9}
        ]
        
        result = quantifier.monte_carlo_dropout_uncertainty(forward_passes, 0.1)
        
        assert "mc_dropout_uncertainty" in result
        assert "prediction_mean" in result
        assert "prediction_variance" in result
        assert 0.0 <= result["mc_dropout_uncertainty"] <= 1.0
        assert result["num_samples"] == 4
    
    def test_bayesian_uncertainty_estimation(self, quantifier):
        """测试贝叶斯不确定性估计"""
        prior_beliefs = {"uncertainty": 0.6, "confidence": 0.4}
        likelihood_data = [
            {"strength": 0.8},
            {"strength": 0.7},
            {"strength": 0.9}
        ]
        
        result = quantifier.bayesian_uncertainty_estimation(prior_beliefs, likelihood_data)
        
        assert "bayesian_uncertainty" in result
        assert "prior_uncertainty" in result
        assert "evidence_uncertainty" in result
        assert "information_gain" in result
        assert 0.0 <= result["bayesian_uncertainty"] <= 1.0
    
    def test_calculate_prediction_intervals(self, quantifier):
        """测试计算预测区间"""
        predictions = [0.1, 0.3, 0.5, 0.7, 0.9, 0.2, 0.4, 0.6, 0.8]
        
        intervals = quantifier.calculate_prediction_intervals(predictions)
        
        assert "95%" in intervals
        assert "68%" in intervals
        
        interval_95 = intervals["95%"]
        assert "lower_bound" in interval_95
        assert "upper_bound" in interval_95
        assert interval_95["lower_bound"] <= interval_95["upper_bound"]
        assert interval_95["confidence_level"] == 0.95
    
    def test_uncertainty_decomposition(self, quantifier):
        """测试不确定性分解"""
        result = quantifier.uncertainty_decomposition(
            total_uncertainty=0.5,
            aleatoric_uncertainty=0.3,
            epistemic_uncertainty=0.2
        )
        
        assert "total_uncertainty" in result
        assert "aleatoric_ratio" in result
        assert "epistemic_ratio" in result
        assert "reducible_uncertainty" in result
        assert "irreducible_uncertainty" in result
        assert "recommendations" in result
        assert isinstance(result["recommendations"], list)
    
    def test_generate_visualization_data(self, quantifier):
        """测试生成可视化数据"""
        uncertainty_metrics = {
            "aleatoric_uncertainty": 0.3,
            "epistemic_uncertainty": 0.4,
            "total_uncertainty": 0.5
        }
        
        prediction_intervals = {
            "95%": {
                "lower_bound": 0.2,
                "upper_bound": 0.8,
                "interval_center": 0.5,
                "interval_width": 0.6
            }
        }
        
        viz_data = quantifier.generate_visualization_data(
            uncertainty_metrics, prediction_intervals
        )
        
        assert "uncertainty_decomposition" in viz_data
        assert "prediction_intervals" in viz_data
        assert "metadata" in viz_data
        
        # 检查不确定性分解图表数据
        decomp_data = viz_data["uncertainty_decomposition"]["data"]
        assert len(decomp_data) == 2  # 偶然和认知不确定性
        assert decomp_data[0]["label"] == "偶然不确定性（数据噪声）"
        assert decomp_data[1]["label"] == "认知不确定性（模型知识）"
    
    def test_record_uncertainty_calculation(self, quantifier):
        """测试记录不确定性计算"""
        input_data = {"data_points": 100, "method": "bayesian"}
        result = {"uncertainty": 0.3, "calculation_time_ms": 150}
        
        quantifier.record_uncertainty_calculation("bayesian", input_data, result)
        
        assert len(quantifier.uncertainty_history) == 1
        record = quantifier.uncertainty_history[0]
        assert record["calculation_type"] == "bayesian"
        assert record["input_summary"]["data_points"] == 100
        assert record["result"]["uncertainty"] == 0.3
    
    def test_get_uncertainty_summary_no_data(self, quantifier):
        """测试无数据时的不确定性摘要"""
        summary = quantifier.get_uncertainty_summary()
        assert summary["status"] == "no_data"
    
    def test_get_uncertainty_summary_with_data(self, quantifier):
        """测试有数据时的不确定性摘要"""
        # 添加一些计算记录
        for i in range(5):
            quantifier.record_uncertainty_calculation(
                "test_type",
                {"data_points": i * 10},
                {"uncertainty": 0.1 * i, "calculation_time_ms": 100 + i * 10}
            )
        
        summary = quantifier.get_uncertainty_summary()
        
        assert summary["total_calculations"] == 5
        assert summary["recent_calculations"] == 5
        assert "test_type" in summary["calculation_types"]
        assert summary["average_duration_ms"] > 0
        assert summary["status"] == "active"


if __name__ == "__main__":
    pytest.main([__file__])