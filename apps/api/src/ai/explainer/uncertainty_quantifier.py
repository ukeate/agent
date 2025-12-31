"""不确定性量化器

本模块实现贝叶斯不确定性评估、模型预测区间计算和置信度可视化数据生成功能。
"""

import math
import numpy as np
import statistics
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from src.models.schemas.explanation import ConfidenceMetrics

class UncertaintyQuantifier:
    """不确定性量化器"""
    
    def __init__(self):
        """初始化不确定性量化器"""
        self.uncertainty_history: List[Dict[str, Any]] = []
        self.monte_carlo_samples = 1000  # Monte Carlo采样数量
        self.confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ置信水平
    
    def quantify_aleatoric_uncertainty(
        self,
        predictions: List[float],
        labels: Optional[List[float]] = None
    ) -> Dict[str, float]:
        """量化偶然不确定性（数据内在的噪声）"""
        if not predictions:
            return {"aleatoric_uncertainty": 0.5, "variance": 0.0}
        
        # 计算预测的方差作为偶然不确定性的估计
        prediction_variance = statistics.variance(predictions) if len(predictions) > 1 else 0.0
        
        # 如果有真实标签，计算残差的方差
        residual_variance = 0.0
        if labels and len(labels) == len(predictions):
            residuals = [abs(pred - label) for pred, label in zip(predictions, labels)]
            residual_variance = statistics.variance(residuals) if len(residuals) > 1 else 0.0
        
        # 偶然不确定性是数据固有的，不能通过更多数据减少
        aleatoric_uncertainty = max(prediction_variance, residual_variance)
        
        # 归一化到[0,1]范围
        normalized_uncertainty = min(1.0, aleatoric_uncertainty / (aleatoric_uncertainty + 1.0))
        
        return {
            "aleatoric_uncertainty": normalized_uncertainty,
            "variance": prediction_variance,
            "residual_variance": residual_variance,
            "source": "data_noise"
        }
    
    def quantify_epistemic_uncertainty(
        self,
        model_ensemble_predictions: List[List[float]],
        feature_importance_variations: Optional[List[Dict[str, float]]] = None
    ) -> Dict[str, float]:
        """量化认知不确定性（模型知识的不确定性）"""
        if not model_ensemble_predictions:
            return {"epistemic_uncertainty": 0.5, "model_disagreement": 0.0}
        
        # 计算模型间的分歧作为认知不确定性
        if len(model_ensemble_predictions) > 1:
            # 每个位置的预测方差
            position_variances = []
            for i in range(len(model_ensemble_predictions[0])):
                position_predictions = [pred[i] for pred in model_ensemble_predictions if i < len(pred)]
                if len(position_predictions) > 1:
                    position_variances.append(statistics.variance(position_predictions))
            
            model_disagreement = statistics.mean(position_variances) if position_variances else 0.0
        else:
            model_disagreement = 0.0
        
        # 特征重要性的变化也反映认知不确定性
        feature_uncertainty = 0.0
        if feature_importance_variations:
            all_features = set()
            for importance_dict in feature_importance_variations:
                all_features.update(importance_dict.keys())
            
            feature_variances = []
            for feature in all_features:
                feature_values = [
                    importance_dict.get(feature, 0.0)
                    for importance_dict in feature_importance_variations
                ]
                if len(feature_values) > 1:
                    feature_variances.append(statistics.variance(feature_values))
            
            feature_uncertainty = statistics.mean(feature_variances) if feature_variances else 0.0
        
        # 认知不确定性可以通过更多数据和更好的模型减少
        epistemic_uncertainty = (model_disagreement + feature_uncertainty) / 2
        
        # 归一化到[0,1]范围
        normalized_uncertainty = min(1.0, epistemic_uncertainty / (epistemic_uncertainty + 0.1))
        
        return {
            "epistemic_uncertainty": normalized_uncertainty,
            "model_disagreement": model_disagreement,
            "feature_uncertainty": feature_uncertainty,
            "source": "model_knowledge"
        }
    
    def monte_carlo_dropout_uncertainty(
        self,
        forward_passes: List[Dict[str, Any]],
        dropout_rate: float = 0.1
    ) -> Dict[str, float]:
        """通过Monte Carlo Dropout估计不确定性"""
        if not forward_passes:
            return {"mc_dropout_uncertainty": 0.5}
        
        # 提取每次前向传播的预测结果
        predictions = []
        for pass_result in forward_passes:
            if "prediction" in pass_result:
                predictions.append(pass_result["prediction"])
        
        if len(predictions) < 2:
            return {"mc_dropout_uncertainty": 0.5}
        
        # 计算预测的统计信息
        mean_prediction = statistics.mean(predictions)
        variance_prediction = statistics.variance(predictions)
        
        # MC Dropout不确定性
        mc_uncertainty = math.sqrt(variance_prediction)
        
        # 考虑dropout rate的影响
        adjusted_uncertainty = mc_uncertainty * (1 + dropout_rate)
        
        # 归一化
        normalized_uncertainty = min(1.0, adjusted_uncertainty)
        
        return {
            "mc_dropout_uncertainty": normalized_uncertainty,
            "prediction_mean": mean_prediction,
            "prediction_variance": variance_prediction,
            "dropout_rate": dropout_rate,
            "num_samples": len(predictions)
        }
    
    def bayesian_uncertainty_estimation(
        self,
        prior_beliefs: Dict[str, Any],
        likelihood_data: List[Dict[str, Any]],
        posterior_samples: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """贝叶斯不确定性估计"""
        
        # 先验不确定性
        prior_uncertainty = prior_beliefs.get("uncertainty", 0.5)
        prior_confidence = prior_beliefs.get("confidence", 0.5)
        
        # 似然数据的不确定性
        if likelihood_data:
            evidence_strengths = [data.get("strength", 0.5) for data in likelihood_data]
            evidence_uncertainties = [1 - strength for strength in evidence_strengths]
            avg_evidence_uncertainty = statistics.mean(evidence_uncertainties)
        else:
            avg_evidence_uncertainty = 0.8
        
        # 后验不确定性（结合先验和似然）
        if posterior_samples:
            # 从后验样本计算不确定性
            posterior_values = []
            for sample in posterior_samples:
                if "value" in sample:
                    posterior_values.append(sample["value"])
            
            if len(posterior_values) > 1:
                posterior_variance = statistics.variance(posterior_values)
                posterior_uncertainty = min(1.0, math.sqrt(posterior_variance))
            else:
                posterior_uncertainty = (prior_uncertainty + avg_evidence_uncertainty) / 2
        else:
            # 简化的贝叶斯更新
            # 不确定性 = 先验不确定性 * 证据不确定性 / (先验不确定性 + 证据不确定性)
            posterior_uncertainty = (prior_uncertainty * avg_evidence_uncertainty) / \
                                  (prior_uncertainty + avg_evidence_uncertainty + 1e-10)
        
        # 计算信息增益（不确定性的减少）
        information_gain = prior_uncertainty - posterior_uncertainty
        
        return {
            "bayesian_uncertainty": posterior_uncertainty,
            "prior_uncertainty": prior_uncertainty,
            "evidence_uncertainty": avg_evidence_uncertainty,
            "information_gain": information_gain,
            "uncertainty_reduction_ratio": information_gain / (prior_uncertainty + 1e-10)
        }
    
    def calculate_prediction_intervals(
        self,
        predictions: List[float],
        confidence_levels: Optional[List[float]] = None
    ) -> Dict[str, Dict[str, float]]:
        """计算预测区间"""
        if not predictions:
            return {}
        
        if confidence_levels is None:
            confidence_levels = self.confidence_levels
        
        prediction_intervals = {}
        
        # 排序预测值
        sorted_predictions = sorted(predictions)
        n = len(sorted_predictions)
        
        for confidence_level in confidence_levels:
            # 计算分位数
            alpha = 1 - confidence_level
            lower_quantile = alpha / 2
            upper_quantile = 1 - alpha / 2
            
            # 计算对应的索引
            lower_idx = max(0, int(lower_quantile * n))
            upper_idx = min(n - 1, int(upper_quantile * n))
            
            lower_bound = sorted_predictions[lower_idx]
            upper_bound = sorted_predictions[upper_idx]
            
            interval_width = upper_bound - lower_bound
            interval_center = (upper_bound + lower_bound) / 2
            
            prediction_intervals[f"{confidence_level:.0%}"] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "interval_width": interval_width,
                "interval_center": interval_center,
                "confidence_level": confidence_level
            }
        
        return prediction_intervals
    
    def uncertainty_decomposition(
        self,
        total_uncertainty: float,
        aleatoric_uncertainty: float,
        epistemic_uncertainty: float
    ) -> Dict[str, Any]:
        """不确定性分解分析"""
        
        # 确保总不确定性不小于各分量的最大值
        total_uncertainty = max(total_uncertainty, aleatoric_uncertainty, epistemic_uncertainty)
        
        # 计算各分量的贡献比例
        if total_uncertainty > 0:
            aleatoric_ratio = aleatoric_uncertainty / total_uncertainty
            epistemic_ratio = epistemic_uncertainty / total_uncertainty
            
            # 处理重叠部分
            overlap = max(0, aleatoric_ratio + epistemic_ratio - 1.0)
            if overlap > 0:
                aleatoric_ratio = aleatoric_ratio * (1 - overlap / 2)
                epistemic_ratio = epistemic_ratio * (1 - overlap / 2)
        else:
            aleatoric_ratio = 0.5
            epistemic_ratio = 0.5
            overlap = 0.0
        
        # 计算可减少的不确定性（主要是认知不确定性）
        reducible_uncertainty = epistemic_uncertainty
        irreducible_uncertainty = aleatoric_uncertainty
        
        return {
            "total_uncertainty": total_uncertainty,
            "aleatoric_uncertainty": aleatoric_uncertainty,
            "epistemic_uncertainty": epistemic_uncertainty,
            "aleatoric_ratio": aleatoric_ratio,
            "epistemic_ratio": epistemic_ratio,
            "reducible_uncertainty": reducible_uncertainty,
            "irreducible_uncertainty": irreducible_uncertainty,
            "uncertainty_overlap": overlap,
            "recommendations": self._generate_uncertainty_recommendations(
                aleatoric_ratio, epistemic_ratio
            )
        }
    
    def generate_visualization_data(
        self,
        uncertainty_metrics: Dict[str, Any],
        prediction_intervals: Dict[str, Dict[str, float]],
        confidence_history: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """生成不确定性可视化数据"""
        
        visualization_data = {
            "uncertainty_decomposition": {
                "chart_type": "pie",
                "data": [
                    {
                        "label": "偶然不确定性（数据噪声）",
                        "value": uncertainty_metrics.get("aleatoric_uncertainty", 0.3),
                        "color": "#FF6B6B",
                        "description": "数据固有的不确定性，无法通过更多数据消除"
                    },
                    {
                        "label": "认知不确定性（模型知识）",
                        "value": uncertainty_metrics.get("epistemic_uncertainty", 0.3),
                        "color": "#4ECDC4",
                        "description": "模型知识不足导致的不确定性，可通过更多数据和更好模型改善"
                    }
                ]
            },
            
            "prediction_intervals": {
                "chart_type": "interval_plot",
                "data": []
            },
            
            "uncertainty_distribution": {
                "chart_type": "histogram",
                "data": {
                    "bins": [],
                    "frequencies": [],
                    "total_samples": 0
                }
            },
            
            "confidence_trend": {
                "chart_type": "line",
                "data": {
                    "timestamps": [],
                    "confidence_values": [],
                    "uncertainty_values": []
                }
            }
        }
        
        # 预测区间数据
        for level, interval in prediction_intervals.items():
            visualization_data["prediction_intervals"]["data"].append({
                "confidence_level": level,
                "lower_bound": interval["lower_bound"],
                "upper_bound": interval["upper_bound"],
                "center": interval["interval_center"],
                "width": interval["interval_width"]
            })
        
        # 置信度历史趋势
        if confidence_history:
            timestamps = []
            confidence_values = []
            uncertainty_values = []
            
            for record in confidence_history[-50:]:  # 最近50个记录
                if "timestamp" in record:
                    timestamps.append(record["timestamp"])
                    confidence_values.append(record.get("confidence", 0.5))
                    uncertainty_values.append(record.get("uncertainty", 0.5))
            
            visualization_data["confidence_trend"]["data"] = {
                "timestamps": timestamps,
                "confidence_values": confidence_values,
                "uncertainty_values": uncertainty_values
            }
        
        # 不确定性分布直方图
        if "uncertainty_samples" in uncertainty_metrics:
            samples = uncertainty_metrics["uncertainty_samples"]
            if samples:
                # 创建直方图数据
                bin_count = min(20, len(samples) // 5)  # 适应性bin数量
                bin_edges, frequencies = self._create_histogram(samples, bin_count)
                
                visualization_data["uncertainty_distribution"]["data"] = {
                    "bins": bin_edges,
                    "frequencies": frequencies,
                    "total_samples": len(samples)
                }
        
        # 添加元数据
        visualization_data["metadata"] = {
            "generated_at": utc_now().isoformat(),
            "total_uncertainty": uncertainty_metrics.get("total_uncertainty", 0.5),
            "confidence_level": uncertainty_metrics.get("confidence_level", 0.95),
            "visualization_id": str(uuid4())
        }
        
        return visualization_data
    
    def _generate_uncertainty_recommendations(
        self,
        aleatoric_ratio: float,
        epistemic_ratio: float
    ) -> List[str]:
        """生成不确定性改进建议"""
        recommendations = []
        
        if epistemic_ratio > 0.6:
            recommendations.extend([
                "认知不确定性较高，建议收集更多训练数据",
                "考虑使用集成模型或贝叶斯方法",
                "增加模型复杂度或改进特征工程"
            ])
        
        if aleatoric_ratio > 0.6:
            recommendations.extend([
                "偶然不确定性较高，可能需要改善数据质量",
                "考虑去除噪声数据或异常值",
                "检查数据收集过程是否存在系统性误差"
            ])
        
        if aleatoric_ratio < 0.3 and epistemic_ratio < 0.3:
            recommendations.append("不确定性较低，模型预测相对可靠")
        
        if abs(aleatoric_ratio - epistemic_ratio) < 0.2:
            recommendations.append("两种不确定性相当，需要平衡数据收集和模型改进")
        
        return recommendations
    
    def _create_histogram(self, data: List[float], bin_count: int) -> Tuple[List[float], List[int]]:
        """创建直方图数据"""
        if not data:
            return [], []
        
        min_val = min(data)
        max_val = max(data)
        
        if min_val == max_val:
            return [min_val, max_val], [len(data)]
        
        bin_width = (max_val - min_val) / bin_count
        bin_edges = [min_val + i * bin_width for i in range(bin_count + 1)]
        
        frequencies = [0] * bin_count
        for value in data:
            bin_index = min(bin_count - 1, int((value - min_val) / bin_width))
            frequencies[bin_index] += 1
        
        return bin_edges, frequencies
    
    def record_uncertainty_calculation(
        self,
        calculation_type: str,
        input_data: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """记录不确定性计算历史"""
        record = {
            "calculation_id": str(uuid4()),
            "calculation_type": calculation_type,
            "timestamp": utc_now().isoformat(),
            "input_summary": {
                "data_points": input_data.get("data_points", 0),
                "method": input_data.get("method", "unknown"),
                "parameters": input_data.get("parameters", {})
            },
            "result": result,
            "calculation_duration_ms": result.get("calculation_time_ms", 0)
        }
        
        self.uncertainty_history.append(record)
        
        # 保持历史记录在合理范围内
        if len(self.uncertainty_history) > 200:
            self.uncertainty_history = self.uncertainty_history[-200:]
    
    def get_uncertainty_summary(self) -> Dict[str, Any]:
        """获取不确定性量化摘要"""
        if not self.uncertainty_history:
            return {"status": "no_data"}
        
        recent_calculations = self.uncertainty_history[-10:]
        
        # 统计不同类型的计算
        calculation_types = {}
        total_duration = 0
        
        for record in recent_calculations:
            calc_type = record["calculation_type"]
            calculation_types[calc_type] = calculation_types.get(calc_type, 0) + 1
            total_duration += record.get("calculation_duration_ms", 0)
        
        return {
            "total_calculations": len(self.uncertainty_history),
            "recent_calculations": len(recent_calculations),
            "calculation_types": calculation_types,
            "average_duration_ms": total_duration / len(recent_calculations) if recent_calculations else 0,
            "last_calculation": recent_calculations[-1]["timestamp"] if recent_calculations else None,
            "status": "active"
        }
