"""
异常检测算法实现

实现统计异常检测、机器学习异常检测和实时异常评分系统。
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import json
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
import pandas as pd
from scipy import stats
from ..models import BehaviorEvent, AnomalyDetection, AnomalyType, SeverityLevel
from ..storage.event_store import EventStore

from src.core.logging import get_logger
logger = get_logger(__name__)

# 机器学习相关导入

class DetectionMethod(str, Enum):
    """异常检测方法枚举"""
    Z_SCORE = "z_score"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    STATISTICAL_COMBINED = "statistical_combined"

@dataclass
class AnomalyScore:
    """异常分数"""
    score: float
    method: DetectionMethod
    threshold: float
    is_anomaly: bool
    confidence: float
    details: Dict[str, Any]

@dataclass
class DetectionResult:
    """检测结果"""
    anomalies: List[AnomalyDetection]
    total_samples: int
    anomaly_rate: float
    processing_time_seconds: float
    method_stats: Dict[str, Any]

class StatisticalAnomalyDetector:
    """统计异常检测器"""
    
    def __init__(
        self,
        z_score_threshold: float = 3.0,
        iqr_multiplier: float = 1.5,
        min_samples: int = 10
    ):
        self.z_score_threshold = z_score_threshold
        self.iqr_multiplier = iqr_multiplier
        self.min_samples = min_samples
        
        # 统计缓存
        self.feature_stats = {}
        self.last_update = None
    
    async def detect_anomalies(
        self,
        events: List[BehaviorEvent],
        features: List[str] = None
    ) -> List[AnomalyScore]:
        """检测统计异常"""
        if len(events) < self.min_samples:
            return []
        
        anomaly_scores = []
        
        try:
            # 提取特征
            feature_data = await self._extract_features(events, features)
            
            if not feature_data:
                return []
            
            # 对每个特征进行异常检测
            for feature_name, values in feature_data.items():
                if len(values) < self.min_samples:
                    continue
                
                # Z-Score检测
                z_scores = await self._z_score_detection(values, feature_name)
                anomaly_scores.extend(z_scores)
                
                # IQR检测
                iqr_scores = await self._iqr_detection(values, feature_name)
                anomaly_scores.extend(iqr_scores)
            
            return anomaly_scores
            
        except Exception as e:
            logger.error(f"统计异常检测失败: {e}")
            return []
    
    async def _extract_features(
        self,
        events: List[BehaviorEvent],
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, List[float]]:
        """提取数值特征"""
        feature_data = defaultdict(list)
        
        for event in events:
            # 时间相关特征
            if not feature_names or 'hour_of_day' in feature_names:
                hour = event.timestamp.hour
                feature_data['hour_of_day'].append(float(hour))
            
            if not feature_names or 'day_of_week' in feature_names:
                day = event.timestamp.weekday()
                feature_data['day_of_week'].append(float(day))
            
            # 持续时间特征
            if event.duration_ms is not None:
                if not feature_names or 'duration_ms' in feature_names:
                    feature_data['duration_ms'].append(float(event.duration_ms))
            
            # 事件数据中的数值特征
            for key, value in event.event_data.items():
                if isinstance(value, (int, float)):
                    feature_name = f"event_data_{key}"
                    if not feature_names or feature_name in feature_names:
                        feature_data[feature_name].append(float(value))
            
            # 上下文数据中的数值特征
            for key, value in event.context.items():
                if isinstance(value, (int, float)):
                    feature_name = f"context_{key}"
                    if not feature_names or feature_name in feature_names:
                        feature_data[feature_name].append(float(value))
        
        return dict(feature_data)
    
    async def _z_score_detection(
        self,
        values: List[float],
        feature_name: str
    ) -> List[AnomalyScore]:
        """Z-Score异常检测"""
        if len(values) < self.min_samples:
            return []
        
        try:
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if std_val == 0:
                return []
            
            z_scores = np.abs((np.array(values) - mean_val) / std_val)
            
            anomaly_scores = []
            for i, z_score in enumerate(z_scores):
                is_anomaly = z_score > self.z_score_threshold
                confidence = min(z_score / self.z_score_threshold, 1.0) if is_anomaly else 0.0
                
                score = AnomalyScore(
                    score=float(z_score),
                    method=DetectionMethod.Z_SCORE,
                    threshold=self.z_score_threshold,
                    is_anomaly=is_anomaly,
                    confidence=confidence,
                    details={
                        'feature': feature_name,
                        'value': values[i],
                        'mean': mean_val,
                        'std': std_val,
                        'sample_index': i
                    }
                )
                
                if is_anomaly:
                    anomaly_scores.append(score)
            
            return anomaly_scores
            
        except Exception as e:
            logger.error(f"Z-Score检测失败: {e}")
            return []
    
    async def _iqr_detection(
        self,
        values: List[float],
        feature_name: str
    ) -> List[AnomalyScore]:
        """IQR异常检测"""
        if len(values) < self.min_samples:
            return []
        
        try:
            q1 = np.percentile(values, 25)
            q3 = np.percentile(values, 75)
            iqr = q3 - q1
            
            if iqr == 0:
                return []
            
            lower_bound = q1 - self.iqr_multiplier * iqr
            upper_bound = q3 + self.iqr_multiplier * iqr
            
            anomaly_scores = []
            for i, value in enumerate(values):
                is_anomaly = value < lower_bound or value > upper_bound
                
                if is_anomaly:
                    # 计算异常分数
                    if value < lower_bound:
                        score = (lower_bound - value) / iqr
                    else:
                        score = (value - upper_bound) / iqr
                    
                    confidence = min(score, 1.0)
                    
                    anomaly_score = AnomalyScore(
                        score=float(score),
                        method=DetectionMethod.IQR,
                        threshold=self.iqr_multiplier,
                        is_anomaly=True,
                        confidence=confidence,
                        details={
                            'feature': feature_name,
                            'value': value,
                            'q1': q1,
                            'q3': q3,
                            'iqr': iqr,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'sample_index': i
                        }
                    )
                    
                    anomaly_scores.append(anomaly_score)
            
            return anomaly_scores
            
        except Exception as e:
            logger.error(f"IQR检测失败: {e}")
            return []

class MLAnomalyDetector:
    """机器学习异常检测器"""
    
    def __init__(
        self,
        contamination: float = 0.1,
        n_estimators: int = 100
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        
        # 检测器实例
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=contamination
        )
        
        self.elliptic_envelope = EllipticEnvelope(
            contamination=contamination,
            random_state=42
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    async def fit_and_detect(
        self,
        events: List[BehaviorEvent],
        method: DetectionMethod = DetectionMethod.ISOLATION_FOREST
    ) -> List[AnomalyScore]:
        """拟合模型并检测异常"""
        if len(events) < 10:
            return []
        
        try:
            # 构建特征矩阵
            feature_matrix = await self._build_feature_matrix(events)
            
            if feature_matrix.size == 0:
                return []
            
            # 标准化特征
            normalized_features = self.scaler.fit_transform(feature_matrix)
            
            # 根据方法选择检测器
            if method == DetectionMethod.ISOLATION_FOREST:
                return await self._isolation_forest_detection(normalized_features, events)
            elif method == DetectionMethod.LOCAL_OUTLIER_FACTOR:
                return await self._lof_detection(normalized_features, events)
            elif method == DetectionMethod.ELLIPTIC_ENVELOPE:
                return await self._elliptic_envelope_detection(normalized_features, events)
            else:
                raise ValueError(f"不支持的检测方法: {method}")
            
        except Exception as e:
            logger.error(f"机器学习异常检测失败: {e}")
            return []
    
    async def _build_feature_matrix(self, events: List[BehaviorEvent]) -> np.ndarray:
        """构建特征矩阵"""
        features = []
        
        for event in events:
            feature_vector = []
            
            # 时间特征
            feature_vector.append(float(event.timestamp.hour))
            feature_vector.append(float(event.timestamp.weekday()))
            feature_vector.append(float(event.timestamp.minute))
            
            # 持续时间特征
            feature_vector.append(float(event.duration_ms or 0))
            
            # 事件类型特征(one-hot编码)
            event_types = ['user_action', 'agent_response', 'system_event', 'error_event', 'feedback_event']
            for et in event_types:
                feature_vector.append(float(event.event_type.value == et))
            
            # 会话特征(基于session_id的哈希)
            session_hash = abs(hash(event.session_id)) % 1000
            feature_vector.append(float(session_hash))
            
            # 事件数据大小特征
            event_data_size = len(json.dumps(event.event_data, default=str))
            feature_vector.append(float(event_data_size))
            
            context_size = len(json.dumps(event.context, default=str))
            feature_vector.append(float(context_size))
            
            features.append(feature_vector)
        
        return np.array(features)
    
    async def _isolation_forest_detection(
        self,
        features: np.ndarray,
        events: List[BehaviorEvent]
    ) -> List[AnomalyScore]:
        """Isolation Forest异常检测"""
        try:
            # 拟合模型
            self.isolation_forest.fit(features)
            
            # 预测异常
            predictions = self.isolation_forest.predict(features)
            decision_scores = self.isolation_forest.decision_function(features)
            
            anomaly_scores = []
            for i, (pred, score) in enumerate(zip(predictions, decision_scores)):
                is_anomaly = pred == -1
                
                if is_anomaly:
                    # 归一化分数到0-1范围
                    normalized_score = (0.5 - score)  # decision_function返回负值表示异常
                    confidence = min(max(normalized_score, 0.0), 1.0)
                    
                    anomaly_score = AnomalyScore(
                        score=float(normalized_score),
                        method=DetectionMethod.ISOLATION_FOREST,
                        threshold=0.0,
                        is_anomaly=True,
                        confidence=confidence,
                        details={
                            'decision_score': float(score),
                            'event_id': events[i].event_id,
                            'sample_index': i
                        }
                    )
                    
                    anomaly_scores.append(anomaly_score)
            
            return anomaly_scores
            
        except Exception as e:
            logger.error(f"Isolation Forest检测失败: {e}")
            return []
    
    async def _lof_detection(
        self,
        features: np.ndarray,
        events: List[BehaviorEvent]
    ) -> List[AnomalyScore]:
        """Local Outlier Factor异常检测"""
        try:
            # LOF需要使用fit_predict
            predictions = self.lof.fit_predict(features)
            negative_outlier_factors = self.lof.negative_outlier_factor_
            
            anomaly_scores = []
            for i, (pred, lof_score) in enumerate(zip(predictions, negative_outlier_factors)):
                is_anomaly = pred == -1
                
                if is_anomaly:
                    # LOF分数越负表示越异常
                    normalized_score = min(abs(lof_score - 1.0), 5.0) / 5.0
                    confidence = min(normalized_score, 1.0)
                    
                    anomaly_score = AnomalyScore(
                        score=float(normalized_score),
                        method=DetectionMethod.LOCAL_OUTLIER_FACTOR,
                        threshold=1.0,
                        is_anomaly=True,
                        confidence=confidence,
                        details={
                            'lof_score': float(lof_score),
                            'event_id': events[i].event_id,
                            'sample_index': i
                        }
                    )
                    
                    anomaly_scores.append(anomaly_score)
            
            return anomaly_scores
            
        except Exception as e:
            logger.error(f"LOF检测失败: {e}")
            return []
    
    async def _elliptic_envelope_detection(
        self,
        features: np.ndarray,
        events: List[BehaviorEvent]
    ) -> List[AnomalyScore]:
        """Elliptic Envelope异常检测"""
        try:
            # 拟合模型
            self.elliptic_envelope.fit(features)
            
            # 预测异常
            predictions = self.elliptic_envelope.predict(features)
            decision_scores = self.elliptic_envelope.decision_function(features)
            
            anomaly_scores = []
            for i, (pred, score) in enumerate(zip(predictions, decision_scores)):
                is_anomaly = pred == -1
                
                if is_anomaly:
                    # 归一化分数
                    normalized_score = max(-score / 10.0, 0.0)  # decision_function返回负值表示异常
                    confidence = min(normalized_score, 1.0)
                    
                    anomaly_score = AnomalyScore(
                        score=float(normalized_score),
                        method=DetectionMethod.ELLIPTIC_ENVELOPE,
                        threshold=0.0,
                        is_anomaly=True,
                        confidence=confidence,
                        details={
                            'decision_score': float(score),
                            'event_id': events[i].event_id,
                            'sample_index': i
                        }
                    )
                    
                    anomaly_scores.append(anomaly_score)
            
            return anomaly_scores
            
        except Exception as e:
            logger.error(f"Elliptic Envelope检测失败: {e}")
            return []

class RealTimeAnomalyScorer:
    """实时异常评分系统"""
    
    def __init__(
        self,
        window_size: int = 1000,
        update_interval: int = 100
    ):
        self.window_size = window_size
        self.update_interval = update_interval
        
        # 滑动窗口存储历史事件
        self.event_window = deque(maxlen=window_size)
        self.feature_stats = defaultdict(lambda: {'mean': 0.0, 'std': 1.0, 'count': 0})
        
        # 实时统计
        self.events_processed = 0
        self.anomalies_detected = 0
        
        # 检测器实例
        self.statistical_detector = StatisticalAnomalyDetector()
        self.ml_detector = MLAnomalyDetector()
        
        # 上次模型更新时间
        self.last_model_update = utc_now()
        self.model_update_interval = timedelta(hours=1)
    
    async def score_event(self, event: BehaviorEvent) -> AnomalyScore:
        """为单个事件评分"""
        self.events_processed += 1
        
        try:
            # 添加到滑动窗口
            self.event_window.append(event)
            
            # 更新特征统计
            await self._update_feature_stats(event)
            
            # 计算实时异常分数
            scores = []
            
            # 快速统计检测
            stat_score = await self._quick_statistical_check(event)
            if stat_score:
                scores.append(stat_score)
            
            # 如果有足够的数据且需要更新模型
            if (len(self.event_window) >= 100 and 
                utc_now() - self.last_model_update > self.model_update_interval):
                await self._update_ml_models()
                self.last_model_update = utc_now()
            
            # 机器学习检测(如果模型已拟合)
            if self.ml_detector.is_fitted:
                ml_score = await self._ml_score_single_event(event)
                if ml_score:
                    scores.append(ml_score)
            
            # 合并分数
            if scores:
                combined_score = await self._combine_scores(scores)
                
                if combined_score.is_anomaly:
                    self.anomalies_detected += 1
                
                return combined_score
            else:
                # 返回正常分数
                return AnomalyScore(
                    score=0.0,
                    method=DetectionMethod.STATISTICAL_COMBINED,
                    threshold=0.5,
                    is_anomaly=False,
                    confidence=0.0,
                    details={'status': 'normal'}
                )
                
        except Exception as e:
            logger.error(f"实时评分失败: {e}")
            return AnomalyScore(
                score=0.0,
                method=DetectionMethod.STATISTICAL_COMBINED,
                threshold=0.5,
                is_anomaly=False,
                confidence=0.0,
                details={'error': str(e)}
            )
    
    async def _update_feature_stats(self, event: BehaviorEvent):
        """更新特征统计信息"""
        # 时间特征
        hour = float(event.timestamp.hour)
        self._update_running_stats('hour_of_day', hour)
        
        day = float(event.timestamp.weekday())
        self._update_running_stats('day_of_week', day)
        
        # 持续时间特征
        if event.duration_ms is not None:
            self._update_running_stats('duration_ms', float(event.duration_ms))
        
        # 事件数据大小
        event_data_size = len(json.dumps(event.event_data, default=str))
        self._update_running_stats('event_data_size', float(event_data_size))
    
    def _update_running_stats(self, feature_name: str, value: float):
        """更新运行统计信息(增量计算均值和标准差)"""
        stats = self.feature_stats[feature_name]
        stats['count'] += 1
        
        # 增量计算均值
        delta = value - stats['mean']
        stats['mean'] += delta / stats['count']
        
        # 增量计算方差(Welford算法)
        if stats['count'] > 1:
            delta2 = value - stats['mean']
            if 'variance_sum' not in stats:
                stats['variance_sum'] = 0.0
            stats['variance_sum'] += delta * delta2
            stats['std'] = np.sqrt(stats['variance_sum'] / (stats['count'] - 1))
    
    async def _quick_statistical_check(self, event: BehaviorEvent) -> Optional[AnomalyScore]:
        """快速统计检查"""
        max_z_score = 0.0
        max_feature = None
        
        # 检查各个特征的Z-score
        features_to_check = [
            ('hour_of_day', float(event.timestamp.hour)),
            ('day_of_week', float(event.timestamp.weekday())),
        ]
        
        if event.duration_ms is not None:
            features_to_check.append(('duration_ms', float(event.duration_ms)))
        
        event_data_size = len(json.dumps(event.event_data, default=str))
        features_to_check.append(('event_data_size', float(event_data_size)))
        
        for feature_name, value in features_to_check:
            stats = self.feature_stats[feature_name]
            
            if stats['count'] > 10 and stats['std'] > 0:
                z_score = abs(value - stats['mean']) / stats['std']
                
                if z_score > max_z_score:
                    max_z_score = z_score
                    max_feature = feature_name
        
        # 如果Z-score超过阈值，返回异常分数
        if max_z_score > 3.0:
            return AnomalyScore(
                score=float(max_z_score),
                method=DetectionMethod.Z_SCORE,
                threshold=3.0,
                is_anomaly=True,
                confidence=min(max_z_score / 5.0, 1.0),
                details={
                    'feature': max_feature,
                    'z_score': max_z_score,
                    'value': features_to_check[0][1] if max_feature == features_to_check[0][0] else None
                }
            )
        
        return None
    
    async def _update_ml_models(self):
        """更新机器学习模型"""
        if len(self.event_window) < 100:
            return
        
        try:
            # 使用滑动窗口中的事件更新模型
            window_events = list(self.event_window)
            
            # 重新拟合Isolation Forest
            feature_matrix = await self.ml_detector._build_feature_matrix(window_events)
            if feature_matrix.size > 0:
                normalized_features = self.ml_detector.scaler.fit_transform(feature_matrix)
                self.ml_detector.isolation_forest.fit(normalized_features)
                self.ml_detector.is_fitted = True
                
                logger.info("实时异常检测模型已更新")
                
        except Exception as e:
            logger.error(f"更新ML模型失败: {e}")
    
    async def _ml_score_single_event(self, event: BehaviorEvent) -> Optional[AnomalyScore]:
        """使用ML模型为单个事件评分"""
        try:
            # 构建特征
            feature_matrix = await self.ml_detector._build_feature_matrix([event])
            if feature_matrix.size == 0:
                return None
            
            # 标准化
            normalized_features = self.ml_detector.scaler.transform(feature_matrix)
            
            # 预测
            prediction = self.ml_detector.isolation_forest.predict(normalized_features)[0]
            decision_score = self.ml_detector.isolation_forest.decision_function(normalized_features)[0]
            
            is_anomaly = prediction == -1
            
            if is_anomaly:
                normalized_score = (0.5 - decision_score)
                confidence = min(max(normalized_score, 0.0), 1.0)
                
                return AnomalyScore(
                    score=float(normalized_score),
                    method=DetectionMethod.ISOLATION_FOREST,
                    threshold=0.0,
                    is_anomaly=True,
                    confidence=confidence,
                    details={
                        'decision_score': float(decision_score),
                        'event_id': event.event_id
                    }
                )
            
        except Exception as e:
            logger.error(f"ML单事件评分失败: {e}")
        
        return None
    
    async def _combine_scores(self, scores: List[AnomalyScore]) -> AnomalyScore:
        """合并多个异常分数"""
        if not scores:
            return AnomalyScore(0.0, DetectionMethod.STATISTICAL_COMBINED, 0.5, False, 0.0, {})
        
        # 加权平均分数
        weights = {
            DetectionMethod.Z_SCORE: 0.4,
            DetectionMethod.IQR: 0.3,
            DetectionMethod.ISOLATION_FOREST: 0.6,
            DetectionMethod.LOCAL_OUTLIER_FACTOR: 0.5,
            DetectionMethod.ELLIPTIC_ENVELOPE: 0.4
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        max_confidence = 0.0
        
        for score in scores:
            weight = weights.get(score.method, 0.3)
            weighted_score += score.score * weight
            total_weight += weight
            max_confidence = max(max_confidence, score.confidence)
        
        if total_weight > 0:
            combined_score_value = weighted_score / total_weight
        else:
            combined_score_value = 0.0
        
        # 判断是否为异常
        is_anomaly = combined_score_value > 0.5 or any(s.is_anomaly for s in scores)
        
        return AnomalyScore(
            score=combined_score_value,
            method=DetectionMethod.STATISTICAL_COMBINED,
            threshold=0.5,
            is_anomaly=is_anomaly,
            confidence=max_confidence,
            details={
                'component_scores': [
                    {
                        'method': s.method.value,
                        'score': s.score,
                        'confidence': s.confidence
                    }
                    for s in scores
                ],
                'num_components': len(scores)
            }
        )
    
    def get_scorer_stats(self) -> Dict[str, Any]:
        """获取评分器统计信息"""
        return {
            'events_processed': self.events_processed,
            'anomalies_detected': self.anomalies_detected,
            'anomaly_rate': self.anomalies_detected / max(self.events_processed, 1),
            'window_size': len(self.event_window),
            'max_window_size': self.window_size,
            'model_fitted': self.ml_detector.is_fitted,
            'last_model_update': self.last_model_update.isoformat(),
            'feature_stats': {
                name: {
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'count': stats['count']
                }
                for name, stats in self.feature_stats.items()
            }
        }

class AnomalyDetectionEngine:
    """异常检测引擎"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.statistical_detector = StatisticalAnomalyDetector()
        self.ml_detector = MLAnomalyDetector()
        self.realtime_scorer = RealTimeAnomalyScorer()
        
        # 检测历史和缓存
        self.detection_history = []
        self.anomaly_cache = {}
    
    async def detect_batch_anomalies(
        self,
        user_ids: Optional[List[str]] = None,
        time_range_days: int = 7,
        methods: List[DetectionMethod] = None
    ) -> DetectionResult:
        """批量异常检测"""
        if methods is None:
            methods = [DetectionMethod.ISOLATION_FOREST, DetectionMethod.Z_SCORE]
        
        start_time = asyncio.get_running_loop().time()
        
        try:
            # 获取事件数据
            events = await self._fetch_events_for_detection(user_ids, time_range_days)
            
            if not events:
                return DetectionResult([], 0, 0.0, 0.0, {})
            
            all_anomalies = []
            method_stats = {}
            
            # 执行各种检测方法
            for method in methods:
                if method in [DetectionMethod.Z_SCORE, DetectionMethod.IQR]:
                    scores = await self.statistical_detector.detect_anomalies(events)
                    method_anomalies = await self._scores_to_anomalies(scores, events)
                elif method in [DetectionMethod.ISOLATION_FOREST, DetectionMethod.LOCAL_OUTLIER_FACTOR, DetectionMethod.ELLIPTIC_ENVELOPE]:
                    scores = await self.ml_detector.fit_and_detect(events, method)
                    method_anomalies = await self._scores_to_anomalies(scores, events)
                else:
                    continue
                
                all_anomalies.extend(method_anomalies)
                method_stats[method.value] = {
                    'anomalies_found': len(method_anomalies),
                    'samples_processed': len(events)
                }
            
            # 去重异常(基于event_id)
            unique_anomalies = {}
            for anomaly in all_anomalies:
                event_id = anomaly.metadata.get('event_id')
                if event_id:
                    if event_id not in unique_anomalies or anomaly.anomaly_score > unique_anomalies[event_id].anomaly_score:
                        unique_anomalies[event_id] = anomaly
            
            final_anomalies = list(unique_anomalies.values())
            processing_time = asyncio.get_running_loop().time() - start_time
            
            return DetectionResult(
                anomalies=final_anomalies,
                total_samples=len(events),
                anomaly_rate=len(final_anomalies) / len(events),
                processing_time_seconds=processing_time,
                method_stats=method_stats
            )
            
        except Exception as e:
            logger.error(f"批量异常检测失败: {e}")
            raise
    
    async def _fetch_events_for_detection(
        self,
        user_ids: Optional[List[str]],
        time_range_days: int
    ) -> List[BehaviorEvent]:
        """获取用于检测的事件数据"""
        from ..models import EventQueryFilter
        
        end_time = utc_now()
        start_time = end_time - timedelta(days=time_range_days)
        
        all_events = []
        
        if user_ids:
            for user_id in user_ids:
                filter_params = EventQueryFilter(
                    user_id=user_id,
                    start_time=start_time,
                    end_time=end_time,
                    limit=5000
                )
                events_data, _ = await self.event_store.query_events(filter_params)
                
                # 转换为BehaviorEvent对象
                for event_data in events_data:
                    event = BehaviorEvent(
                        event_id=event_data['event_id'],
                        session_id=event_data['session_id'],
                        user_id=event_data['user_id'],
                        event_type=event_data['event_type'],
                        event_name=event_data['event_name'],
                        event_data=event_data['event_data'],
                        context=event_data['context'],
                        timestamp=datetime.fromisoformat(event_data['timestamp']),
                        duration_ms=event_data.get('duration_ms')
                    )
                    all_events.append(event)
        else:
            filter_params = EventQueryFilter(
                start_time=start_time,
                end_time=end_time,
                limit=10000
            )
            events_data, _ = await self.event_store.query_events(filter_params)
            
            for event_data in events_data:
                event = BehaviorEvent(
                    event_id=event_data['event_id'],
                    session_id=event_data['session_id'],
                    user_id=event_data['user_id'],
                    event_type=event_data['event_type'],
                    event_name=event_data['event_name'],
                    event_data=event_data['event_data'],
                    context=event_data['context'],
                    timestamp=datetime.fromisoformat(event_data['timestamp']),
                    duration_ms=event_data.get('duration_ms')
                )
                all_events.append(event)
        
        return all_events
    
    async def _scores_to_anomalies(
        self,
        scores: List[AnomalyScore],
        events: List[BehaviorEvent]
    ) -> List[AnomalyDetection]:
        """将异常分数转换为异常检测对象"""
        anomalies = []
        
        for score in scores:
            if not score.is_anomaly:
                continue
            
            # 获取对应的事件
            sample_index = score.details.get('sample_index')
            event_id = score.details.get('event_id')
            
            event = None
            if sample_index is not None and 0 <= sample_index < len(events):
                event = events[sample_index]
            elif event_id:
                event = next((e for e in events if e.event_id == event_id), None)
            
            if not event:
                continue
            
            # 确定严重程度
            severity = SeverityLevel.LOW
            if score.confidence >= 0.8:
                severity = SeverityLevel.CRITICAL
            elif score.confidence >= 0.6:
                severity = SeverityLevel.HIGH
            elif score.confidence >= 0.4:
                severity = SeverityLevel.MEDIUM
            
            # 确定异常类型
            anomaly_type = AnomalyType.STATISTICAL
            if score.method in [DetectionMethod.ISOLATION_FOREST, DetectionMethod.LOCAL_OUTLIER_FACTOR, DetectionMethod.ELLIPTIC_ENVELOPE]:
                anomaly_type = AnomalyType.BEHAVIORAL
            
            anomaly = AnomalyDetection(
                session_id=event.session_id,
                user_id=event.user_id,
                anomaly_type=anomaly_type,
                severity=severity,
                anomaly_score=score.score,
                detection_method=score.method.value,
                details={
                    'confidence': score.confidence,
                    'threshold': score.threshold,
                    'event_id': event.event_id,
                    'event_name': event.event_name,
                    'event_type': event.event_type.value,
                    **score.details
                }
            )
            
            anomalies.append(anomaly)
        
        return anomalies
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            'statistical_detector': {
                'z_score_threshold': self.statistical_detector.z_score_threshold,
                'iqr_multiplier': self.statistical_detector.iqr_multiplier,
                'min_samples': self.statistical_detector.min_samples
            },
            'ml_detector': {
                'contamination': self.ml_detector.contamination,
                'n_estimators': self.ml_detector.n_estimators,
                'is_fitted': self.ml_detector.is_fitted
            },
            'realtime_scorer': self.realtime_scorer.get_scorer_stats(),
            'detection_history_count': len(self.detection_history),
            'cached_anomalies': len(self.anomaly_cache)
        }
