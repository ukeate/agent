"""置信度计算器

本模块实现多维度置信度计算、不确定性量化和置信度校准功能。
"""

import math
import statistics
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from models.schemas.explanation import (
    ConfidenceMetrics,
    ConfidenceSource,
    ExplanationComponent
)


class ConfidenceCalculator:
    """置信度计算器"""
    
    def __init__(self):
        """初始化置信度计算器"""
        self.calculation_history: List[Dict[str, Any]] = []
        self.calibration_data: List[Dict[str, Any]] = []
        
        # 置信度计算权重配置
        self.confidence_weights = {
            ConfidenceSource.MODEL_PROBABILITY: 0.3,
            ConfidenceSource.FEATURE_IMPORTANCE: 0.25,
            ConfidenceSource.EVIDENCE_STRENGTH: 0.25,
            ConfidenceSource.HISTORICAL_ACCURACY: 0.15,
            ConfidenceSource.CONSENSUS_AGREEMENT: 0.05
        }
        
        # 不确定性因子
        self.uncertainty_factors = {
            "data_quality": 0.2,
            "model_complexity": 0.15,
            "feature_reliability": 0.2,
            "temporal_distance": 0.15,
            "context_similarity": 0.15,
            "sample_size": 0.15
        }
    
    def calculate_confidence_metrics(
        self,
        model_prediction: Dict[str, Any],
        evidence_data: List[Dict[str, Any]],
        historical_performance: Optional[Dict[str, Any]] = None,
        context_factors: Optional[Dict[str, Any]] = None
    ) -> ConfidenceMetrics:
        """计算综合置信度指标"""
        
        calculation_id = str(uuid4())
        calculation_start = datetime.now(timezone.utc)
        
        try:
            # 1. 模型概率置信度
            model_confidence = self._calculate_model_confidence(model_prediction)
            
            # 2. 证据强度置信度
            evidence_confidence = self._calculate_evidence_confidence(evidence_data)
            
            # 3. 特征重要性置信度
            feature_confidence = self._calculate_feature_importance_confidence(
                model_prediction, evidence_data
            )
            
            # 4. 历史准确性置信度
            historical_confidence = self._calculate_historical_confidence(
                historical_performance
            )
            
            # 5. 一致性协议置信度
            consensus_confidence = self._calculate_consensus_confidence(
                evidence_data, context_factors
            )
            
            # 6. 计算总体置信度
            overall_confidence = self._calculate_weighted_confidence({
                ConfidenceSource.MODEL_PROBABILITY: model_confidence,
                ConfidenceSource.FEATURE_IMPORTANCE: feature_confidence,
                ConfidenceSource.EVIDENCE_STRENGTH: evidence_confidence,
                ConfidenceSource.HISTORICAL_ACCURACY: historical_confidence,
                ConfidenceSource.CONSENSUS_AGREEMENT: consensus_confidence
            })
            
            # 7. 计算不确定性分数
            uncertainty_score = self._calculate_uncertainty_score(
                model_prediction, evidence_data, context_factors
            )
            
            # 8. 计算置信区间
            confidence_interval = self._calculate_confidence_interval(
                overall_confidence, uncertainty_score
            )
            
            # 9. 计算校准分数
            calibration_score = self._calculate_calibration_score(
                overall_confidence, historical_performance
            )
            
            # 创建置信度指标
            metrics = ConfidenceMetrics(
                overall_confidence=overall_confidence,
                prediction_confidence=model_confidence,
                evidence_confidence=evidence_confidence,
                model_confidence=feature_confidence,
                uncertainty_score=uncertainty_score,
                variance=self._calculate_variance(evidence_data),
                confidence_interval_lower=confidence_interval[0],
                confidence_interval_upper=confidence_interval[1],
                confidence_sources=[
                    source for source, weight in self.confidence_weights.items()
                    if weight > 0
                ],
                calibration_score=calibration_score
            )
            
            # 记录计算历史
            self._record_calculation(
                calculation_id,
                calculation_start,
                model_prediction,
                evidence_data,
                metrics,
                context_factors
            )
            
            return metrics
            
        except Exception as e:
            # 降级处理：返回保守的置信度
            return ConfidenceMetrics(
                overall_confidence=0.5,
                uncertainty_score=0.5,
                confidence_sources=[ConfidenceSource.MODEL_PROBABILITY]
            )
    
    def _calculate_model_confidence(self, model_prediction: Dict[str, Any]) -> float:
        """计算模型预测置信度"""
        if not model_prediction:
            return 0.5
        
        # 从模型预测中提取概率信息
        probability = model_prediction.get("probability", 0.5)
        logits = model_prediction.get("logits", [])
        softmax_scores = model_prediction.get("softmax", [])
        
        # 基于概率的置信度
        prob_confidence = abs(probability - 0.5) * 2  # 将[0.5,1]映射到[0,1]
        
        # 基于logits的置信度（如果可用）
        logit_confidence = 0.5
        if logits:
            max_logit = max(logits)
            logit_range = max(logits) - min(logits) if len(logits) > 1 else 1
            logit_confidence = min(1.0, abs(max_logit) / max(1.0, logit_range))
        
        # 基于softmax分布的置信度
        softmax_confidence = 0.5
        if softmax_scores:
            max_score = max(softmax_scores)
            entropy = -sum(p * math.log(p + 1e-10) for p in softmax_scores if p > 0)
            max_entropy = math.log(len(softmax_scores))
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            softmax_confidence = max_score * (1 - normalized_entropy)
        
        # 综合置信度
        confidence = (prob_confidence * 0.5 + logit_confidence * 0.3 + softmax_confidence * 0.2)
        return min(1.0, max(0.0, confidence))
    
    def _calculate_evidence_confidence(self, evidence_data: List[Dict[str, Any]]) -> float:
        """计算证据强度置信度"""
        if not evidence_data:
            return 0.3
        
        total_strength = 0
        total_weight = 0
        
        for evidence in evidence_data:
            weight = evidence.get("weight", 1.0)
            reliability = evidence.get("reliability_score", 0.5)
            relevance = evidence.get("relevance_score", 0.5)
            freshness = evidence.get("freshness_score", 0.5)
            
            # 证据强度 = 可靠性 * 相关性 * 新鲜度
            strength = reliability * relevance * freshness
            
            total_strength += strength * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.3
        
        average_strength = total_strength / total_weight
        
        # 考虑证据数量的影响
        evidence_count_factor = min(1.0, len(evidence_data) / 5.0)  # 5个证据达到满分
        
        return average_strength * (0.7 + 0.3 * evidence_count_factor)
    
    def _calculate_feature_importance_confidence(
        self,
        model_prediction: Dict[str, Any],
        evidence_data: List[Dict[str, Any]]
    ) -> float:
        """计算特征重要性置信度"""
        feature_importance = model_prediction.get("feature_importance", {})
        if not feature_importance:
            return 0.5
        
        # 计算重要特征的覆盖率
        important_features = {
            k: v for k, v in feature_importance.items()
            if v > 0.1  # 重要性阈值
        }
        
        if not important_features:
            return 0.3
        
        # 检查重要特征在证据中的覆盖情况
        evidence_features = set()
        for evidence in evidence_data:
            if "factor_name" in evidence:
                evidence_features.add(evidence["factor_name"])
        
        covered_importance = sum(
            importance for feature, importance in important_features.items()
            if feature in evidence_features
        )
        
        total_importance = sum(important_features.values())
        coverage_ratio = covered_importance / total_importance if total_importance > 0 else 0
        
        # 计算特征重要性分布的集中度
        importance_values = list(feature_importance.values())
        if len(importance_values) > 1:
            gini_coefficient = self._calculate_gini_coefficient(importance_values)
            concentration_factor = gini_coefficient  # 更集中的分布更可信
        else:
            concentration_factor = 0.5
        
        return coverage_ratio * 0.7 + concentration_factor * 0.3
    
    def _calculate_historical_confidence(
        self,
        historical_performance: Optional[Dict[str, Any]]
    ) -> float:
        """计算历史准确性置信度"""
        if not historical_performance:
            return 0.5
        
        accuracy = historical_performance.get("accuracy", 0.5)
        precision = historical_performance.get("precision", 0.5)
        recall = historical_performance.get("recall", 0.5)
        f1_score = historical_performance.get("f1_score", 0.5)
        
        # 计算综合性能分数
        performance_score = (accuracy * 0.4 + precision * 0.2 + recall * 0.2 + f1_score * 0.2)
        
        # 考虑样本数量的影响
        sample_count = historical_performance.get("sample_count", 100)
        sample_factor = min(1.0, sample_count / 1000.0)  # 1000个样本达到满分
        
        # 考虑时间衰减
        last_update = historical_performance.get("last_update")
        time_factor = 1.0
        if last_update:
            try:
                last_update_dt = datetime.fromisoformat(last_update.replace("Z", "+00:00"))
                days_ago = (datetime.now(timezone.utc) - last_update_dt).days
                time_factor = max(0.5, 1.0 - (days_ago / 365.0))  # 一年后衰减到0.5
            except:
                time_factor = 0.8
        
        return performance_score * sample_factor * time_factor
    
    def _calculate_consensus_confidence(
        self,
        evidence_data: List[Dict[str, Any]],
        context_factors: Optional[Dict[str, Any]]
    ) -> float:
        """计算一致性协议置信度"""
        if len(evidence_data) < 2:
            return 0.5
        
        # 检查证据间的一致性
        supporting_count = 0
        contradicting_count = 0
        
        for evidence in evidence_data:
            support_evidence = evidence.get("supporting_evidence_count", 0)
            contradict_evidence = evidence.get("contradictory_evidence_count", 0)
            
            supporting_count += support_evidence
            contradicting_count += contradict_evidence
        
        total_relationships = supporting_count + contradicting_count
        if total_relationships == 0:
            return 0.5
        
        # 一致性比例
        consistency_ratio = supporting_count / total_relationships
        
        # 考虑多源验证
        unique_sources = len(set(evidence.get("source", "unknown") for evidence in evidence_data))
        source_diversity = min(1.0, unique_sources / 3.0)  # 3个不同来源达到满分
        
        return consistency_ratio * 0.7 + source_diversity * 0.3
    
    def _calculate_weighted_confidence(
        self,
        confidence_scores: Dict[ConfidenceSource, float]
    ) -> float:
        """计算加权置信度"""
        total_weighted_score = 0
        total_weight = 0
        
        for source, score in confidence_scores.items():
            weight = self.confidence_weights.get(source, 0.2)
            total_weighted_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.5
        
        return total_weighted_score / total_weight
    
    def _calculate_uncertainty_score(
        self,
        model_prediction: Dict[str, Any],
        evidence_data: List[Dict[str, Any]],
        context_factors: Optional[Dict[str, Any]]
    ) -> float:
        """计算不确定性分数"""
        uncertainty_components = {}
        
        # 1. 数据质量不确定性
        if evidence_data:
            quality_scores = [evidence.get("reliability_score", 0.5) for evidence in evidence_data]
            avg_quality = statistics.mean(quality_scores)
            uncertainty_components["data_quality"] = 1 - avg_quality
        else:
            uncertainty_components["data_quality"] = 0.8
        
        # 2. 模型复杂性不确定性
        model_uncertainty = model_prediction.get("uncertainty", 0.5)
        uncertainty_components["model_complexity"] = model_uncertainty
        
        # 3. 特征可靠性不确定性
        feature_importance = model_prediction.get("feature_importance", {})
        if feature_importance:
            # 特征重要性的方差表示不确定性
            importance_variance = statistics.variance(feature_importance.values())
            uncertainty_components["feature_reliability"] = min(1.0, importance_variance * 2)
        else:
            uncertainty_components["feature_reliability"] = 0.6
        
        # 4. 时间距离不确定性
        if context_factors:
            temporal_distance = context_factors.get("temporal_distance_days", 0)
            uncertainty_components["temporal_distance"] = min(1.0, temporal_distance / 365.0)
        else:
            uncertainty_components["temporal_distance"] = 0.3
        
        # 5. 上下文相似性不确定性
        if context_factors:
            context_similarity = context_factors.get("context_similarity", 0.7)
            uncertainty_components["context_similarity"] = 1 - context_similarity
        else:
            uncertainty_components["context_similarity"] = 0.4
        
        # 6. 样本大小不确定性
        sample_size = len(evidence_data)
        if sample_size < 3:
            uncertainty_components["sample_size"] = 0.8
        elif sample_size < 10:
            uncertainty_components["sample_size"] = 0.5
        else:
            uncertainty_components["sample_size"] = 0.2
        
        # 计算加权不确定性
        total_uncertainty = 0
        total_weight = 0
        
        for factor, uncertainty in uncertainty_components.items():
            weight = self.uncertainty_factors.get(factor, 0.1)
            total_uncertainty += uncertainty * weight
            total_weight += weight
        
        return total_uncertainty / total_weight if total_weight > 0 else 0.5
    
    def _calculate_confidence_interval(
        self,
        confidence: float,
        uncertainty: float,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """计算置信区间"""
        # 基于不确定性计算误差边界
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% vs 99%
        margin_of_error = z_score * uncertainty * 0.5  # 调整因子
        
        lower_bound = max(0.0, confidence - margin_of_error)
        upper_bound = min(1.0, confidence + margin_of_error)
        
        return (lower_bound, upper_bound)
    
    def _calculate_variance(self, evidence_data: List[Dict[str, Any]]) -> float:
        """计算证据权重方差"""
        if not evidence_data:
            return 0.0
        
        weights = [evidence.get("weight", 1.0) for evidence in evidence_data]
        if len(weights) < 2:
            return 0.0
        
        return statistics.variance(weights)
    
    def _calculate_calibration_score(
        self,
        predicted_confidence: float,
        historical_performance: Optional[Dict[str, Any]]
    ) -> float:
        """计算置信度校准分数"""
        if not historical_performance:
            return 0.5
        
        # 校准分数衡量预测置信度与实际准确性的匹配程度
        actual_accuracy = historical_performance.get("accuracy", 0.5)
        
        # 计算校准误差（越小越好）
        calibration_error = abs(predicted_confidence - actual_accuracy)
        
        # 转换为校准分数（越大越好）
        calibration_score = 1 - calibration_error
        
        return max(0.0, calibration_score)
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """计算基尼系数（衡量分布的不平等程度）"""
        if not values or len(values) < 2:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        sum_of_differences = sum(
            abs(sorted_values[i] - sorted_values[j])
            for i in range(n) for j in range(n)
        )
        
        mean_value = statistics.mean(sorted_values)
        gini = sum_of_differences / (2 * n * n * mean_value) if mean_value > 0 else 0
        
        return min(1.0, gini)
    
    def calibrate_confidence(
        self,
        predicted_confidence: float,
        actual_outcome: bool,
        context: Optional[Dict[str, Any]] = None
    ):
        """校准置信度（用于模型改进）"""
        calibration_record = {
            "predicted_confidence": predicted_confidence,
            "actual_outcome": actual_outcome,
            "context": context or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "calibration_id": str(uuid4())
        }
        
        self.calibration_data.append(calibration_record)
        
        # 保持校准数据在合理范围内
        if len(self.calibration_data) > 1000:
            self.calibration_data = self.calibration_data[-1000:]
    
    def get_calibration_statistics(self) -> Dict[str, Any]:
        """获取校准统计信息"""
        if not self.calibration_data:
            return {"status": "no_data"}
        
        # 按置信度区间分组计算校准误差
        confidence_bins = {}
        bin_size = 0.1
        
        for record in self.calibration_data:
            confidence = record["predicted_confidence"]
            bin_index = int(confidence / bin_size)
            bin_key = f"{bin_index * bin_size:.1f}-{(bin_index + 1) * bin_size:.1f}"
            
            if bin_key not in confidence_bins:
                confidence_bins[bin_key] = {"predictions": [], "outcomes": []}
            
            confidence_bins[bin_key]["predictions"].append(confidence)
            confidence_bins[bin_key]["outcomes"].append(1 if record["actual_outcome"] else 0)
        
        # 计算每个区间的校准误差
        calibration_errors = {}
        for bin_key, data in confidence_bins.items():
            if data["predictions"]:
                avg_prediction = statistics.mean(data["predictions"])
                avg_outcome = statistics.mean(data["outcomes"])
                calibration_error = abs(avg_prediction - avg_outcome)
                calibration_errors[bin_key] = {
                    "avg_prediction": avg_prediction,
                    "avg_outcome": avg_outcome,
                    "calibration_error": calibration_error,
                    "sample_count": len(data["predictions"])
                }
        
        # 计算总体校准误差
        all_predictions = [record["predicted_confidence"] for record in self.calibration_data]
        all_outcomes = [1 if record["actual_outcome"] else 0 for record in self.calibration_data]
        
        overall_calibration_error = abs(
            statistics.mean(all_predictions) - statistics.mean(all_outcomes)
        )
        
        return {
            "overall_calibration_error": overall_calibration_error,
            "bin_calibration_errors": calibration_errors,
            "total_samples": len(self.calibration_data),
            "last_updated": datetime.now(timezone.utc).isoformat()
        }
    
    def _record_calculation(
        self,
        calculation_id: str,
        start_time: datetime,
        model_prediction: Dict[str, Any],
        evidence_data: List[Dict[str, Any]],
        result_metrics: ConfidenceMetrics,
        context_factors: Optional[Dict[str, Any]]
    ):
        """记录置信度计算历史"""
        calculation_record = {
            "calculation_id": calculation_id,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "input_summary": {
                "model_prediction_keys": list(model_prediction.keys()),
                "evidence_count": len(evidence_data),
                "context_available": context_factors is not None
            },
            "result_metrics": result_metrics.model_dump(),
            "calculation_duration_ms": int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )
        }
        
        self.calculation_history.append(calculation_record)
        
        # 保持历史记录在合理范围内
        if len(self.calculation_history) > 500:
            self.calculation_history = self.calculation_history[-500:]