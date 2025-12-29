"""
情感识别准确率和质量监控系统
提供实时质量评估、准确率监控和性能优化建议
"""

from src.core.utils.timezone_utils import utc_now
import asyncio
import time
import statistics
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
from abc import ABC, abstractmethod
from .core_interfaces import (
    EmotionType, ModalityType, EmotionState, MultiModalEmotion,
    UnifiedEmotionalData, EmotionalIntelligenceResponse
)
from .result_formatter import FormattingConfig, OutputFormat, result_formatter_manager

from src.core.logging import get_logger
logger = get_logger(__name__)

class QualityMetricType(str, Enum):
    """质量指标类型"""
    ACCURACY = "accuracy"              # 准确率
    PRECISION = "precision"            # 精确率
    RECALL = "recall"                  # 召回率
    F1_SCORE = "f1_score"             # F1分数
    CONFIDENCE = "confidence"          # 置信度
    CONSISTENCY = "consistency"        # 一致性
    LATENCY = "latency"               # 延迟
    THROUGHPUT = "throughput"         # 吞吐量
    DATA_QUALITY = "data_quality"     # 数据质量
    DRIFT_DETECTION = "drift_detection" # 数据漂移

class AlertSeverity(str, Enum):
    """告警严重级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class QualityThreshold:
    """质量阈值"""
    metric_type: QualityMetricType
    warning_threshold: float
    critical_threshold: float
    comparison_operator: str = "less_than"  # "less_than", "greater_than"
    enabled: bool = True

@dataclass
class QualityMetric:
    """质量指标"""
    metric_type: QualityMetricType
    value: float
    timestamp: datetime
    modality: Optional[ModalityType] = None
    user_id: Optional[str] = None
    confidence: Optional[float] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QualityAlert:
    """质量告警"""
    alert_id: str
    metric_type: QualityMetricType
    severity: AlertSeverity
    message: str
    current_value: float
    threshold_value: float
    modality: Optional[ModalityType] = None
    user_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    resolved: bool = False
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GroundTruthData:
    """真实标签数据"""
    user_id: str
    timestamp: datetime
    true_emotion: EmotionState
    modality: ModalityType
    source: str  # "expert_annotation", "user_feedback", "self_report"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

class AccuracyCalculator:
    """准确率计算器"""
    
    def __init__(self):
        self.emotion_similarity_threshold = 0.3
        self.vad_tolerance = {
            'valence': 0.2,
            'arousal': 0.2, 
            'dominance': 0.2
        }
    
    def calculate_emotion_accuracy(
        self, 
        predicted: EmotionState, 
        ground_truth: EmotionState
    ) -> float:
        """计算情感准确率"""
        # 基础情感类型匹配
        emotion_match = 1.0 if predicted.emotion == ground_truth.emotion else 0.0
        
        # VAD维度相似性
        valence_diff = abs(predicted.valence - ground_truth.valence)
        arousal_diff = abs(predicted.arousal - ground_truth.arousal)
        dominance_diff = abs(predicted.dominance - ground_truth.dominance)
        
        valence_sim = max(0, 1 - valence_diff / 2.0)
        arousal_sim = max(0, 1 - arousal_diff)
        dominance_sim = max(0, 1 - dominance_diff)
        
        vad_similarity = (valence_sim + arousal_sim + dominance_sim) / 3
        
        # 强度相似性
        intensity_diff = abs(predicted.intensity - ground_truth.intensity)
        intensity_sim = max(0, 1 - intensity_diff)
        
        # 综合准确率
        total_accuracy = (emotion_match * 0.5 + vad_similarity * 0.3 + intensity_sim * 0.2)
        
        return min(1.0, max(0.0, total_accuracy))
    
    def calculate_classification_metrics(
        self,
        predictions: List[EmotionState],
        ground_truths: List[EmotionState]
    ) -> Dict[str, float]:
        """计算分类指标"""
        if len(predictions) != len(ground_truths):
            raise ValueError("Predictions and ground truths must have the same length")
        
        if not predictions:
            return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0}
        
        # 构建混淆矩阵
        emotion_types = list(EmotionType)
        confusion_matrix = np.zeros((len(emotion_types), len(emotion_types)))
        
        type_to_index = {emotion: i for i, emotion in enumerate(emotion_types)}
        
        for pred, true in zip(predictions, ground_truths):
            pred_idx = type_to_index[pred.emotion]
            true_idx = type_to_index[true.emotion]
            confusion_matrix[true_idx, pred_idx] += 1
        
        # 计算指标
        total_samples = len(predictions)
        correct = np.trace(confusion_matrix)
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        # 计算每类的精确率和召回率
        precisions = []
        recalls = []
        
        for i in range(len(emotion_types)):
            tp = confusion_matrix[i, i]
            fp = confusion_matrix[:, i].sum() - tp
            fn = confusion_matrix[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # 宏平均
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = (2 * macro_precision * macro_recall) / (macro_precision + macro_recall) \
                   if (macro_precision + macro_recall) > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "precision": macro_precision,
            "recall": macro_recall,
            "f1_score": macro_f1,
            "per_class_precision": dict(zip([e.value for e in emotion_types], precisions)),
            "per_class_recall": dict(zip([e.value for e in emotion_types], recalls))
        }

class DataDriftDetector:
    """数据漂移检测器"""
    
    def __init__(self, window_size: int = 1000, drift_threshold: float = 0.1):
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        self.reference_distributions: Dict[ModalityType, Dict[str, Any]] = {}
        self.current_windows: Dict[ModalityType, deque] = {}
        
        for modality in ModalityType:
            self.current_windows[modality] = deque(maxlen=window_size)
    
    def add_sample(self, modality: ModalityType, emotion_state: EmotionState):
        """添加样本"""
        sample = {
            'emotion': emotion_state.emotion.value,
            'valence': emotion_state.valence,
            'arousal': emotion_state.arousal,
            'dominance': emotion_state.dominance,
            'intensity': emotion_state.intensity,
            'confidence': emotion_state.confidence
        }
        
        self.current_windows[modality].append(sample)
    
    def set_reference_distribution(self, modality: ModalityType, samples: List[Dict[str, Any]]):
        """设置参考分布"""
        if not samples:
            return
        
        # 计算分布统计量
        emotions = [s['emotion'] for s in samples]
        valences = [s['valence'] for s in samples]
        arousals = [s['arousal'] for s in samples]
        dominances = [s['dominance'] for s in samples]
        intensities = [s['intensity'] for s in samples]
        
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        emotion_probs = {k: v / len(emotions) for k, v in emotion_counts.items()}
        
        self.reference_distributions[modality] = {
            'emotion_distribution': emotion_probs,
            'valence_stats': {'mean': np.mean(valences), 'std': np.std(valences)},
            'arousal_stats': {'mean': np.mean(arousals), 'std': np.std(arousals)},
            'dominance_stats': {'mean': np.mean(dominances), 'std': np.std(dominances)},
            'intensity_stats': {'mean': np.mean(intensities), 'std': np.std(intensities)}
        }
    
    def detect_drift(self, modality: ModalityType) -> Tuple[bool, float, Dict[str, Any]]:
        """检测数据漂移"""
        if modality not in self.reference_distributions:
            return False, 0.0, {"error": "No reference distribution set"}
        
        current_window = list(self.current_windows[modality])
        if len(current_window) < self.window_size // 2:
            return False, 0.0, {"error": "Insufficient current data"}
        
        reference = self.reference_distributions[modality]
        
        # 计算当前窗口的分布
        current_emotions = [s['emotion'] for s in current_window]
        current_valences = [s['valence'] for s in current_window]
        current_arousals = [s['arousal'] for s in current_window]
        current_dominances = [s['dominance'] for s in current_window]
        current_intensities = [s['intensity'] for s in current_window]
        
        # 情感分布漂移（KL散度近似）
        current_emotion_counts = {}
        for emotion in current_emotions:
            current_emotion_counts[emotion] = current_emotion_counts.get(emotion, 0) + 1
        
        current_emotion_probs = {k: v / len(current_emotions) for k, v in current_emotion_counts.items()}
        
        kl_divergence = 0.0
        for emotion in reference['emotion_distribution']:
            ref_prob = reference['emotion_distribution'][emotion]
            curr_prob = current_emotion_probs.get(emotion, 1e-8)
            if ref_prob > 0:
                kl_divergence += ref_prob * np.log(ref_prob / curr_prob)
        
        # VAD统计漂移
        def calculate_stat_drift(current_values, ref_stats):
            current_mean = np.mean(current_values)
            current_std = np.std(current_values)
            
            mean_drift = abs(current_mean - ref_stats['mean']) / (ref_stats['std'] + 1e-8)
            std_drift = abs(current_std - ref_stats['std']) / (ref_stats['std'] + 1e-8)
            
            return max(mean_drift, std_drift)
        
        valence_drift = calculate_stat_drift(current_valences, reference['valence_stats'])
        arousal_drift = calculate_stat_drift(current_arousals, reference['arousal_stats'])
        dominance_drift = calculate_stat_drift(current_dominances, reference['dominance_stats'])
        intensity_drift = calculate_stat_drift(current_intensities, reference['intensity_stats'])
        
        # 综合漂移分数
        drift_score = (kl_divergence + valence_drift + arousal_drift + dominance_drift + intensity_drift) / 5
        
        is_drift = drift_score > self.drift_threshold
        
        drift_details = {
            "kl_divergence": kl_divergence,
            "valence_drift": valence_drift,
            "arousal_drift": arousal_drift,
            "dominance_drift": dominance_drift,
            "intensity_drift": intensity_drift,
            "emotion_distribution_change": current_emotion_probs
        }
        
        return is_drift, drift_score, drift_details

class QualityMonitor:
    """质量监控器"""
    
    def __init__(self):
        self.accuracy_calculator = AccuracyCalculator()
        self.drift_detector = DataDriftDetector()
        
        # 监控配置
        self.thresholds: List[QualityThreshold] = self._init_default_thresholds()
        self.monitoring_window = timedelta(hours=1)
        self.metric_retention_period = timedelta(days=30)
        
        # 数据存储
        self.metrics_history: Dict[QualityMetricType, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self.ground_truth_data: Dict[str, List[GroundTruthData]] = defaultdict(list)
        self.active_alerts: Dict[str, QualityAlert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        
        # 性能统计
        self.processing_times: Dict[ModalityType, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.prediction_cache: deque = deque(maxlen=1000)
        
        # 异步任务
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # 回调函数
        self._alert_callbacks: List[callable] = []
        self._metric_callbacks: List[callable] = []
    
    def _init_default_thresholds(self) -> List[QualityThreshold]:
        """初始化默认阈值"""
        return [
            QualityThreshold(QualityMetricType.ACCURACY, 0.7, 0.5),
            QualityThreshold(QualityMetricType.PRECISION, 0.7, 0.5),
            QualityThreshold(QualityMetricType.RECALL, 0.7, 0.5),
            QualityThreshold(QualityMetricType.F1_SCORE, 0.7, 0.5),
            QualityThreshold(QualityMetricType.CONFIDENCE, 0.6, 0.4),
            QualityThreshold(QualityMetricType.LATENCY, 1000, 2000, "greater_than"),
            QualityThreshold(QualityMetricType.THROUGHPUT, 10, 5, "less_than"),
            QualityThreshold(QualityMetricType.DATA_QUALITY, 0.8, 0.6),
            QualityThreshold(QualityMetricType.DRIFT_DETECTION, 0.2, 0.4, "greater_than")
        ]
    
    async def start_monitoring(self):
        """启动监控"""
        logger.info("Starting quality monitoring")
        self._shutdown_event.clear()
        
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Quality monitoring started")
    
    async def stop_monitoring(self):
        """停止监控"""
        logger.info("Stopping quality monitoring")
        self._shutdown_event.set()
        
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        
        await asyncio.gather(
            self._monitoring_task,
            self._cleanup_task,
            return_exceptions=True
        )
        
        logger.info("Quality monitoring stopped")
    
    async def record_prediction(
        self,
        user_id: str,
        predicted_emotion: EmotionState,
        modality: ModalityType,
        processing_time: float,
        confidence: float,
        data_quality: float
    ):
        """记录预测结果"""
        timestamp = utc_now()
        
        # 记录预测
        self.prediction_cache.append({
            'user_id': user_id,
            'predicted_emotion': predicted_emotion,
            'modality': modality,
            'processing_time': processing_time,
            'confidence': confidence,
            'data_quality': data_quality,
            'timestamp': timestamp
        })
        
        # 记录性能指标
        self.processing_times[modality].append(processing_time)
        
        # 添加到漂移检测
        self.drift_detector.add_sample(modality, predicted_emotion)
        
        # 记录基础质量指标
        await self._record_metric(
            QualityMetricType.CONFIDENCE, confidence, timestamp, modality, user_id
        )
        await self._record_metric(
            QualityMetricType.DATA_QUALITY, data_quality, timestamp, modality, user_id
        )
        await self._record_metric(
            QualityMetricType.LATENCY, processing_time, timestamp, modality, user_id
        )
    
    async def add_ground_truth(self, ground_truth: GroundTruthData):
        """添加真实标签"""
        self.ground_truth_data[ground_truth.user_id].append(ground_truth)
        
        # 如果有匹配的预测，立即计算准确率
        await self._evaluate_against_ground_truth(ground_truth)
    
    async def _evaluate_against_ground_truth(self, ground_truth: GroundTruthData):
        """根据真实标签评估"""
        # 查找时间窗口内的预测
        time_window = timedelta(minutes=5)  # 5分钟内的预测
        
        matching_predictions = []
        for pred in self.prediction_cache:
            if (pred['user_id'] == ground_truth.user_id and 
                pred['modality'] == ground_truth.modality and
                abs((pred['timestamp'] - ground_truth.timestamp).total_seconds()) <= time_window.total_seconds()):
                matching_predictions.append(pred)
        
        if not matching_predictions:
            return
        
        # 选择最接近的预测
        closest_pred = min(
            matching_predictions,
            key=lambda p: abs((p['timestamp'] - ground_truth.timestamp).total_seconds())
        )
        
        # 计算准确率
        accuracy = self.accuracy_calculator.calculate_emotion_accuracy(
            closest_pred['predicted_emotion'],
            ground_truth.true_emotion
        )
        
        await self._record_metric(
            QualityMetricType.ACCURACY,
            accuracy,
            utc_now(),
            ground_truth.modality,
            ground_truth.user_id,
            confidence=ground_truth.confidence
        )
    
    async def _record_metric(
        self,
        metric_type: QualityMetricType,
        value: float,
        timestamp: datetime,
        modality: Optional[ModalityType] = None,
        user_id: Optional[str] = None,
        confidence: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        """记录质量指标"""
        metric = QualityMetric(
            metric_type=metric_type,
            value=value,
            timestamp=timestamp,
            modality=modality,
            user_id=user_id,
            confidence=confidence,
            context=context or {}
        )
        
        self.metrics_history[metric_type].append(metric)
        
        # 检查阈值
        await self._check_thresholds(metric)
        
        # 调用回调
        for callback in self._metric_callbacks:
            try:
                await callback(metric)
            except Exception as e:
                logger.error(f"Error in metric callback: {e}")
    
    async def _check_thresholds(self, metric: QualityMetric):
        """检查阈值"""
        for threshold in self.thresholds:
            if threshold.metric_type != metric.metric_type or not threshold.enabled:
                continue
            
            is_violation = False
            violated_threshold = None
            severity = None
            
            if threshold.comparison_operator == "less_than":
                if metric.value < threshold.critical_threshold:
                    is_violation = True
                    violated_threshold = threshold.critical_threshold
                    severity = AlertSeverity.CRITICAL
                elif metric.value < threshold.warning_threshold:
                    is_violation = True
                    violated_threshold = threshold.warning_threshold
                    severity = AlertSeverity.WARNING
            else:  # "greater_than"
                if metric.value > threshold.critical_threshold:
                    is_violation = True
                    violated_threshold = threshold.critical_threshold
                    severity = AlertSeverity.CRITICAL
                elif metric.value > threshold.warning_threshold:
                    is_violation = True
                    violated_threshold = threshold.warning_threshold
                    severity = AlertSeverity.WARNING
            
            if is_violation:
                await self._create_alert(metric, threshold, violated_threshold, severity)
    
    async def _create_alert(
        self,
        metric: QualityMetric,
        threshold: QualityThreshold,
        violated_threshold: float,
        severity: AlertSeverity
    ):
        """创建告警"""
        alert_id = f"{metric.metric_type.value}_{metric.modality}_{int(time.time())}"
        
        message = (
            f"Quality threshold violation: {metric.metric_type.value} "
            f"({metric.value:.3f}) {threshold.comparison_operator} {violated_threshold:.3f}"
        )
        
        if metric.modality:
            message += f" for {metric.modality.value}"
        
        alert = QualityAlert(
            alert_id=alert_id,
            metric_type=metric.metric_type,
            severity=severity,
            message=message,
            current_value=metric.value,
            threshold_value=violated_threshold,
            modality=metric.modality,
            user_id=metric.user_id,
            context={
                'threshold_config': {
                    'warning': threshold.warning_threshold,
                    'critical': threshold.critical_threshold,
                    'operator': threshold.comparison_operator
                },
                'metric_context': metric.context
            }
        )
        
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)
        
        # 调用告警回调
        for callback in self._alert_callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.warning(f"Quality alert created: {message}")
    
    async def _monitoring_loop(self):
        """监控循环"""
        while not self._shutdown_event.is_set():
            try:
                await self._run_periodic_checks()
                await asyncio.sleep(60)  # 每分钟检查一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)
    
    async def _run_periodic_checks(self):
        """运行周期性检查"""
        current_time = utc_now()
        
        # 计算吞吐量
        recent_predictions = [
            p for p in self.prediction_cache
            if (current_time - p['timestamp']).total_seconds() <= 60
        ]
        throughput = len(recent_predictions) / 60.0  # predictions per second
        
        await self._record_metric(
            QualityMetricType.THROUGHPUT,
            throughput,
            current_time
        )
        
        # 检查数据漂移
        for modality in ModalityType:
            is_drift, drift_score, _ = self.drift_detector.detect_drift(modality)
            
            await self._record_metric(
                QualityMetricType.DRIFT_DETECTION,
                drift_score,
                current_time,
                modality
            )
        
        # 计算聚合分类指标
        await self._calculate_aggregate_metrics()
    
    async def _calculate_aggregate_metrics(self):
        """计算聚合分类指标"""
        current_time = utc_now()
        window_start = current_time - self.monitoring_window
        
        # 收集窗口内的预测和真实标签
        predictions = []
        ground_truths = []
        
        for user_id, gt_list in self.ground_truth_data.items():
            for gt in gt_list:
                if window_start <= gt.timestamp <= current_time:
                    # 查找匹配的预测
                    matching_pred = None
                    for pred in self.prediction_cache:
                        if (pred['user_id'] == user_id and
                            pred['modality'] == gt.modality and
                            abs((pred['timestamp'] - gt.timestamp).total_seconds()) <= 300):
                            matching_pred = pred
                            break
                    
                    if matching_pred:
                        predictions.append(matching_pred['predicted_emotion'])
                        ground_truths.append(gt.true_emotion)
        
        if len(predictions) >= 10:  # 至少需要10个样本
            metrics = self.accuracy_calculator.calculate_classification_metrics(
                predictions, ground_truths
            )
            
            for metric_name, value in metrics.items():
                if metric_name.startswith('per_class_'):
                    continue  # 跳过每类指标
                
                metric_type = QualityMetricType(metric_name)
                await self._record_metric(metric_type, value, current_time)
    
    async def _cleanup_loop(self):
        """清理循环"""
        while not self._shutdown_event.is_set():
            try:
                await self._cleanup_old_data()
                await asyncio.sleep(3600)  # 每小时清理一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(3600)
    
    async def _cleanup_old_data(self):
        """清理过期数据"""
        cutoff_time = utc_now() - self.metric_retention_period
        
        # 清理指标历史
        for metric_type, history in self.metrics_history.items():
            while history and history[0].timestamp < cutoff_time:
                history.popleft()
        
        # 清理真实标签
        for user_id, gt_list in self.ground_truth_data.items():
            self.ground_truth_data[user_id] = [
                gt for gt in gt_list if gt.timestamp >= cutoff_time
            ]
        
        # 清理已解决的旧告警
        resolved_alerts = [
            alert_id for alert_id, alert in self.active_alerts.items()
            if alert.resolved and (utc_now() - alert.timestamp).days > 7
        ]
        
        for alert_id in resolved_alerts:
            del self.active_alerts[alert_id]
    
    def add_alert_callback(self, callback: callable):
        """添加告警回调"""
        self._alert_callbacks.append(callback)
    
    def add_metric_callback(self, callback: callable):
        """添加指标回调"""
        self._metric_callbacks.append(callback)
    
    def resolve_alert(self, alert_id: str, resolution_note: str = ""):
        """解决告警"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            self.active_alerts[alert_id].context['resolution_note'] = resolution_note
            logger.info(f"Alert resolved: {alert_id}")
    
    def get_quality_report(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """获取质量报告"""
        window = time_window or self.monitoring_window
        cutoff_time = utc_now() - window
        
        report = {
            "report_time": utc_now().isoformat(),
            "time_window_hours": window.total_seconds() / 3600,
            "metrics": {},
            "alerts": {
                "active": len([a for a in self.active_alerts.values() if not a.resolved]),
                "resolved": len([a for a in self.active_alerts.values() if a.resolved]),
                "by_severity": defaultdict(int)
            },
            "performance": {},
            "recommendations": []
        }
        
        # 收集指标统计
        for metric_type, history in self.metrics_history.items():
            recent_metrics = [m for m in history if m.timestamp >= cutoff_time]
            
            if recent_metrics:
                values = [m.value for m in recent_metrics]
                report["metrics"][metric_type.value] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "min": min(values),
                    "max": max(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "latest": values[-1]
                }
        
        # 告警统计
        for alert in self.active_alerts.values():
            report["alerts"]["by_severity"][alert.severity.value] += 1
        
        # 性能统计
        for modality, times in self.processing_times.items():
            if times:
                report["performance"][modality.value] = {
                    "avg_latency_ms": statistics.mean(times),
                    "p95_latency_ms": np.percentile(list(times), 95),
                    "p99_latency_ms": np.percentile(list(times), 99)
                }
        
        # 生成建议
        report["recommendations"] = self._generate_recommendations(report)
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 准确率建议
        if "accuracy" in report["metrics"]:
            accuracy = report["metrics"]["accuracy"]["latest"]
            if accuracy < 0.7:
                recommendations.append(
                    f"Accuracy is low ({accuracy:.3f}). Consider retraining models or collecting more ground truth data."
                )
        
        # 延迟建议
        if "latency" in report["metrics"]:
            latency = report["metrics"]["latency"]["latest"]
            if latency > 1000:
                recommendations.append(
                    f"Processing latency is high ({latency:.1f}ms). Consider model optimization or hardware upgrades."
                )
        
        # 数据质量建议
        if "data_quality" in report["metrics"]:
            quality = report["metrics"]["data_quality"]["latest"]
            if quality < 0.8:
                recommendations.append(
                    f"Data quality is suboptimal ({quality:.3f}). Review data preprocessing and validation."
                )
        
        # 数据漂移建议
        if "drift_detection" in report["metrics"]:
            drift = report["metrics"]["drift_detection"]["latest"]
            if drift > 0.2:
                recommendations.append(
                    f"Data drift detected ({drift:.3f}). Consider model retraining or adaptation."
                )
        
        # 告警建议
        critical_alerts = report["alerts"]["by_severity"].get("critical", 0)
        if critical_alerts > 0:
            recommendations.append(
                f"{critical_alerts} critical alerts active. Immediate attention required."
            )
        
        return recommendations
    
    def export_metrics(
        self,
        format_type: OutputFormat = OutputFormat.JSON,
        time_window: Optional[timedelta] = None
    ) -> str:
        """导出指标数据"""
        window = time_window or self.monitoring_window
        cutoff_time = utc_now() - window
        
        export_data = []
        
        for metric_type, history in self.metrics_history.items():
            recent_metrics = [m for m in history if m.timestamp >= cutoff_time]
            
            for metric in recent_metrics:
                export_data.append({
                    "metric_type": metric.metric_type.value,
                    "value": metric.value,
                    "timestamp": metric.timestamp.isoformat(),
                    "modality": metric.modality.value if metric.modality else None,
                    "user_id": metric.user_id,
                    "confidence": metric.confidence,
                    "context": metric.context
                })
        
        # 使用结果格式化器
        if format_type == OutputFormat.JSON:
            import json
            return json.dumps(export_data, ensure_ascii=False, indent=2)
        elif format_type == OutputFormat.CSV:
            import csv
            import io
            
            output = io.StringIO()
            if export_data:
                writer = csv.DictWriter(output, fieldnames=export_data[0].keys())
                writer.writeheader()
                writer.writerows(export_data)
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format_type}")

# 全局质量监控器实例
quality_monitor = QualityMonitor()
