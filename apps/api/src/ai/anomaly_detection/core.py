"""
智能异常检测核心算法模块
实现多种异常检测算法，从演示版升级为真正的数据科学实现
"""

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats

from src.core.logging import get_logger
logger = get_logger(__name__)

@dataclass
class AnomalyResult:
    """异常检测结果"""
    anomaly_id: str
    user_id: str
    event_type: str
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    description: str
    anomaly_type: str
    detected_by: List[str]
    context: Dict[str, Any]
    score: float
    resolved: bool = False

class UserBehaviorFeatureExtractor:
    """用户行为特征提取器"""
    
    def extract_features(self, events: List[Dict[str, Any]], user_id: str, time_window: int = 3600) -> Dict[str, float]:
        """
        提取用户行为特征
        
        Args:
            events: 用户事件列表
            user_id: 用户ID
            time_window: 时间窗口(秒)
            
        Returns:
            用户行为特征字典
        """
        if not events:
            return self._get_empty_features()
            
        # 过滤指定用户的事件
        user_events = [e for e in events if e.get('user_id') == user_id]
        if not user_events:
            return self._get_empty_features()
            
        # 时间特征
        timestamps = [e.get('timestamp', utc_now()) for e in user_events]
        if isinstance(timestamps[0], str):
            timestamps = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
            
        # 基础统计特征
        features = {
            'event_count': len(user_events),
            'unique_event_types': len(set(e.get('event_type', '') for e in user_events)),
            'avg_events_per_minute': len(user_events) / max(1, time_window / 60),
        }
        
        # 时间间隔特征
        if len(timestamps) > 1:
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                        for i in range(len(timestamps)-1)]
            features.update({
                'avg_interval_seconds': np.mean(intervals),
                'std_interval_seconds': np.std(intervals),
                'min_interval_seconds': np.min(intervals),
                'max_interval_seconds': np.max(intervals),
            })
        else:
            features.update({
                'avg_interval_seconds': 0,
                'std_interval_seconds': 0,
                'min_interval_seconds': 0,
                'max_interval_seconds': 0,
            })
        
        # 事件类型分布特征
        event_types = [e.get('event_type', '') for e in user_events]
        type_counts = pd.Series(event_types).value_counts()
        
        features.update({
            'most_common_event_ratio': type_counts.iloc[0] / len(user_events) if len(type_counts) > 0 else 0,
            'event_type_entropy': self._calculate_entropy(type_counts.values),
        })
        
        # 会话特征(基于30分钟无活动分割会话)
        sessions = self._extract_sessions(timestamps, session_gap=1800)
        if sessions:
            session_lengths = [len(session) for session in sessions]
            features.update({
                'session_count': len(sessions),
                'avg_session_length': np.mean(session_lengths),
                'max_session_length': np.max(session_lengths),
                'session_duration_variance': np.var(session_lengths),
            })
        else:
            features.update({
                'session_count': 1,
                'avg_session_length': len(user_events),
                'max_session_length': len(user_events),
                'session_duration_variance': 0,
            })
            
        # 时间模式特征
        hours = [ts.hour for ts in timestamps]
        features.update({
            'activity_hour_mean': np.mean(hours),
            'activity_hour_std': np.std(hours),
            'is_night_active': sum(1 for h in hours if h < 6 or h > 22) / len(hours),
            'is_work_hours_active': sum(1 for h in hours if 9 <= h <= 17) / len(hours),
        })
        
        return features
    
    def _get_empty_features(self) -> Dict[str, float]:
        """返回空特征向量"""
        return {
            'event_count': 0,
            'unique_event_types': 0,
            'avg_events_per_minute': 0,
            'avg_interval_seconds': 0,
            'std_interval_seconds': 0,
            'min_interval_seconds': 0,
            'max_interval_seconds': 0,
            'most_common_event_ratio': 0,
            'event_type_entropy': 0,
            'session_count': 0,
            'avg_session_length': 0,
            'max_session_length': 0,
            'session_duration_variance': 0,
            'activity_hour_mean': 12,
            'activity_hour_std': 0,
            'is_night_active': 0,
            'is_work_hours_active': 0,
        }
    
    def _calculate_entropy(self, values: np.ndarray) -> float:
        """计算熵值"""
        if len(values) == 0:
            return 0
        probabilities = values / np.sum(values)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _extract_sessions(self, timestamps: List[datetime], session_gap: int = 1800) -> List[List[datetime]]:
        """根据时间间隔提取用户会话"""
        if not timestamps:
            return []
            
        sorted_timestamps = sorted(timestamps)
        sessions = []
        current_session = [sorted_timestamps[0]]
        
        for i in range(1, len(sorted_timestamps)):
            time_diff = (sorted_timestamps[i] - sorted_timestamps[i-1]).total_seconds()
            if time_diff <= session_gap:
                current_session.append(sorted_timestamps[i])
            else:
                sessions.append(current_session)
                current_session = [sorted_timestamps[i]]
        
        sessions.append(current_session)
        return sessions

class StatisticalAnomalyDetector:
    """统计方法异常检测器"""
    
    def __init__(self, z_threshold: float = 2.5, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        
    def detect_z_score_anomalies(self, features: Dict[str, float]) -> List[Tuple[str, float, str]]:
        """Z-score方法检测异常"""
        anomalies = []
        
        # 为了计算Z-score，我们需要历史数据的均值和标准差
        # 这里使用经验值，实际应用中应该从历史数据计算
        expected_ranges = {
            'event_count': (10, 50, 15),  # (min, max, std)
            'avg_events_per_minute': (0.1, 2.0, 0.5),
            'unique_event_types': (2, 8, 2),
            'avg_interval_seconds': (30, 300, 100),
            'most_common_event_ratio': (0.3, 0.8, 0.2),
            'session_count': (1, 10, 3),
        }
        
        for feature, value in features.items():
            if feature in expected_ranges:
                min_val, max_val, std_val = expected_ranges[feature]
                mean_val = (min_val + max_val) / 2
                z_score = abs((value - mean_val) / std_val) if std_val > 0 else 0
                
                if z_score > self.z_threshold:
                    severity = 'high' if z_score > 3 else 'medium'
                    anomalies.append((feature, z_score, severity))
        
        return anomalies
    
    def detect_iqr_anomalies(self, feature_series: List[float]) -> List[int]:
        """IQR方法检测异常值的索引"""
        if len(feature_series) < 4:
            return []
            
        q1 = np.percentile(feature_series, 25)
        q3 = np.percentile(feature_series, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - self.iqr_multiplier * iqr
        upper_bound = q3 + self.iqr_multiplier * iqr
        
        anomalies = []
        for i, value in enumerate(feature_series):
            if value < lower_bound or value > upper_bound:
                anomalies.append(i)
                
        return anomalies

class MachineLearningAnomalyDetector:
    """机器学习异常检测器"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=0.1
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def fit_and_detect(self, feature_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        训练并检测异常
        
        Returns:
            (isolation_forest_predictions, lof_predictions, anomaly_scores)
        """
        if len(feature_matrix) < 10:
            # 数据太少，无法进行机器学习检测
            return np.zeros(len(feature_matrix)), np.zeros(len(feature_matrix)), np.zeros(len(feature_matrix))
            
        # 数据标准化
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # Isolation Forest检测
        if_predictions = self.isolation_forest.fit_predict(scaled_features)
        if_scores = self.isolation_forest.decision_function(scaled_features)
        
        # LOF检测
        lof_predictions = self.lof.fit_predict(scaled_features)
        
        self.is_fitted = True
        
        return if_predictions, lof_predictions, if_scores

class IntelligentAnomalyDetector:
    """智能异常检测主类 - 整合多种检测方法"""
    
    def __init__(self):
        self.feature_extractor = UserBehaviorFeatureExtractor()
        self.statistical_detector = StatisticalAnomalyDetector()
        self.ml_detector = MachineLearningAnomalyDetector()
        
    def detect_anomalies(self, events: List[Dict[str, Any]], 
                        time_window: int = 3600) -> List[AnomalyResult]:
        """
        综合异常检测
        
        Args:
            events: 事件列表
            time_window: 分析时间窗口(秒)
            
        Returns:
            检测到的异常列表
        """
        if not events:
            return []
            
        # 按用户分组
        users = {}
        for event in events:
            user_id = event.get('user_id', 'unknown')
            if user_id not in users:
                users[user_id] = []
            users[user_id].append(event)
        
        anomalies = []
        all_features = []
        user_feature_map = {}
        
        # 1. 特征提取
        for user_id, user_events in users.items():
            features = self.feature_extractor.extract_features(user_events, user_id, time_window)
            all_features.append(list(features.values()))
            user_feature_map[user_id] = (features, user_events)
        
        if len(all_features) < 2:
            logger.warning("用户数量太少，无法进行异常检测")
            return []
            
        # 2. 机器学习检测
        feature_matrix = np.array(all_features)
        if_preds, lof_preds, if_scores = self.ml_detector.fit_and_detect(feature_matrix)
        
        # 3. 对每个用户进行综合分析
        user_ids = list(users.keys())
        
        for i, user_id in enumerate(user_ids):
            features, user_events = user_feature_map[user_id]
            
            # 统计方法检测
            stat_anomalies = self.statistical_detector.detect_z_score_anomalies(features)
            
            # 机器学习结果
            is_if_anomaly = if_preds[i] == -1 if i < len(if_preds) else False
            is_lof_anomaly = lof_preds[i] == -1 if i < len(lof_preds) else False
            anomaly_score = if_scores[i] if i < len(if_scores) else 0
            
            # 综合判断
            detected_methods = []
            severity_scores = []
            
            if stat_anomalies:
                detected_methods.append('statistical')
                severity_scores.extend([2 if sev == 'high' else 1 for _, _, sev in stat_anomalies])
                
            if is_if_anomaly:
                detected_methods.append('isolation_forest')
                severity_scores.append(abs(anomaly_score) * 2)
                
            if is_lof_anomaly:
                detected_methods.append('local_outlier_factor')
                severity_scores.append(2)
            
            # 如果检测到异常
            if detected_methods:
                # 计算综合严重程度
                avg_severity = np.mean(severity_scores) if severity_scores else 1
                confidence = len(detected_methods) / 3  # 最多3种方法
                
                severity = self._calculate_severity(avg_severity, confidence)
                
                # 生成异常描述
                description = self._generate_anomaly_description(features, stat_anomalies, user_events)
                
                # 选择最严重的事件作为代表
                representative_event = max(user_events, 
                                         key=lambda e: self._event_importance_score(e.get('event_type', '')))
                
                anomaly = AnomalyResult(
                    anomaly_id=f"anomaly_{user_id}_{int(utc_now().timestamp())}",
                    user_id=user_id,
                    event_type=representative_event.get('event_type', 'unknown'),
                    timestamp=representative_event.get('timestamp', utc_now()),
                    severity=severity,
                    confidence=confidence,
                    description=description,
                    anomaly_type=self._determine_anomaly_type(stat_anomalies),
                    detected_by=detected_methods,
                    context=self._build_context(features, user_events),
                    score=anomaly_score,
                )
                
                anomalies.append(anomaly)
        
        return anomalies
    
    def _calculate_severity(self, avg_severity: float, confidence: float) -> str:
        """计算异常严重程度"""
        combined_score = avg_severity * confidence
        
        if combined_score >= 2.0:
            return 'critical'
        elif combined_score >= 1.5:
            return 'high'
        elif combined_score >= 1.0:
            return 'medium'
        else:
            return 'low'
    
    def _generate_anomaly_description(self, features: Dict[str, float], 
                                    stat_anomalies: List[Tuple[str, float, str]], 
                                    user_events: List[Dict]) -> str:
        """生成异常描述"""
        descriptions = []
        
        if stat_anomalies:
            for feature, score, severity in stat_anomalies:
                if feature == 'event_count':
                    if features[feature] > 100:
                        descriptions.append("事件频率异常高")
                    elif features[feature] < 5:
                        descriptions.append("活动水平异常低")
                elif feature == 'avg_events_per_minute':
                    if features[feature] > 5:
                        descriptions.append("单位时间事件频率过高，可能存在异常行为")
                elif feature == 'unique_event_types':
                    if features[feature] < 2:
                        descriptions.append("行为模式过于单一")
                elif feature == 'avg_interval_seconds':
                    if features[feature] < 1:
                        descriptions.append("事件间隔异常短，可能为机器行为")
        
        if not descriptions:
            descriptions.append("检测到异常的用户行为模式")
            
        return "; ".join(descriptions[:3])  # 最多显示3个描述
    
    def _determine_anomaly_type(self, stat_anomalies: List[Tuple[str, float, str]]) -> str:
        """确定异常类型"""
        if not stat_anomalies:
            return "behavioral_outlier"
            
        feature_types = [feature for feature, _, _ in stat_anomalies]
        
        if 'avg_events_per_minute' in feature_types or 'event_count' in feature_types:
            return "frequency_anomaly"
        elif 'avg_interval_seconds' in feature_types:
            return "timing_anomaly"
        elif 'unique_event_types' in feature_types:
            return "pattern_anomaly"
        else:
            return "behavioral_outlier"
    
    def _build_context(self, features: Dict[str, float], user_events: List[Dict]) -> Dict[str, Any]:
        """构建异常上下文信息"""
        return {
            'total_events': len(user_events),
            'time_span_minutes': (max(e.get('timestamp', utc_now()) for e in user_events) - 
                                min(e.get('timestamp', utc_now()) for e in user_events)).total_seconds() / 60,
            'most_common_event': max(set(e.get('event_type', '') for e in user_events), 
                                   key=lambda x: sum(1 for e in user_events if e.get('event_type') == x)),
            'activity_score': features.get('event_count', 0) * features.get('unique_event_types', 1),
        }
    
    def _event_importance_score(self, event_type: str) -> int:
        """事件重要性评分"""
        importance_map = {
            'purchase': 5,
            'login': 4,
            'form_submit': 3,
            'search': 3,
            'click': 2,
            'page_view': 2,
            'scroll': 1,
            'hover': 1,
        }
        return importance_map.get(event_type, 1)

def create_sample_events(num_users: int = 50, num_events: int = 1000) -> List[Dict[str, Any]]:
    """创建示例事件数据用于测试"""
    import random
    
    events = []
    event_types = ['click', 'page_view', 'search', 'login', 'purchase', 'form_submit', 'scroll', 'hover']
    
    # 正常用户
    for user_id in range(1, num_users - 5):
        user_events = random.randint(10, 50)
        base_time = utc_now() - timedelta(hours=2)
        
        for i in range(user_events):
            events.append({
                'user_id': f'user_{user_id}',
                'event_type': random.choice(event_types),
                'timestamp': base_time + timedelta(
                    seconds=i * random.randint(30, 300)
                ),
                'session_id': f'session_{user_id}',
            })
    
    # 异常用户1: 高频用户
    user_id = num_users - 4
    base_time = utc_now() - timedelta(hours=1)
    for i in range(200):  # 异常高的事件数
        events.append({
            'user_id': f'user_{user_id}',
            'event_type': random.choice(['click', 'scroll']),
            'timestamp': base_time + timedelta(seconds=i * 2),  # 异常短的间隔
            'session_id': f'session_{user_id}',
        })
    
    # 异常用户2: 单一行为用户
    user_id = num_users - 3
    base_time = utc_now() - timedelta(hours=1.5)
    for i in range(30):
        events.append({
            'user_id': f'user_{user_id}',
            'event_type': 'click',  # 只有一种事件类型
            'timestamp': base_time + timedelta(seconds=i * 60),
            'session_id': f'session_{user_id}',
        })
    
    return events

# 使用示例
if __name__ == "__main__":
    # 创建检测器
    detector = IntelligentAnomalyDetector()
    
    # 创建测试数据
    events = create_sample_events()
    
    # 检测异常
    anomalies = detector.detect_anomalies(events)
    
    logger.info("检测到异常", anomaly_count=len(anomalies))
    for anomaly in anomalies:
        logger.info(
            "异常详情",
            user_id=anomaly.user_id,
            description=anomaly.description,
            severity=anomaly.severity,
            confidence=f"{anomaly.confidence:.2f}",
        )
