"""
简化版异常检测系统，用于快速测试和部署
避免复杂的机器学习模型初始化导致的启动延迟
"""

import numpy as np
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import random

@dataclass
class SimpleAnomalyResult:
    """简化的异常检测结果"""
    anomaly_id: str
    user_id: str
    event_type: str
    timestamp: datetime
    severity: str  # 'low', 'medium', 'high', 'critical'
    confidence: float
    description: str
    anomaly_type: str
    score: float
    resolved: bool = False

class SimpleAnomalyDetector:
    """简化版异常检测器 - 基于统计方法"""
    
    def __init__(self):
        self.normal_thresholds = {
            'events_per_minute': (0.5, 10.0),  # 正常事件频率范围
            'session_length': (1, 100),         # 正常会话长度
            'event_types_per_session': (1, 15), # 正常事件类型数量
        }
    
    def detect_anomalies(self, events: List[Dict[str, Any]], time_window: int = 3600) -> List[SimpleAnomalyResult]:
        """
        检测异常行为
        
        Args:
            events: 事件列表
            time_window: 时间窗口(秒)
            
        Returns:
            异常检测结果列表
        """
        if not events:
            return []
        
        anomalies = []
        now = utc_now()
        
        # 按用户分组事件
        user_events = {}
        for event in events:
            user_id = event.get('user_id', 'unknown')
            if user_id not in user_events:
                user_events[user_id] = []
            user_events[user_id].append(event)
        
        # 为每个用户检测异常
        for user_id, user_event_list in user_events.items():
            user_anomalies = self._detect_user_anomalies(user_id, user_event_list, time_window)
            anomalies.extend(user_anomalies)
        
        return anomalies
    
    def _detect_user_anomalies(self, user_id: str, events: List[Dict[str, Any]], time_window: int) -> List[SimpleAnomalyResult]:
        """为单个用户检测异常"""
        anomalies = []
        
        if len(events) < 2:
            return anomalies
        
        # 计算事件频率
        events_per_minute = len(events) / (time_window / 60)
        
        # 检测高频异常
        if events_per_minute > self.normal_thresholds['events_per_minute'][1]:
            anomaly = SimpleAnomalyResult(
                anomaly_id=f"freq_{user_id}_{int(utc_now().timestamp())}",
                user_id=user_id,
                event_type="high_frequency",
                timestamp=utc_now(),
                severity="high",
                confidence=min(0.9, events_per_minute / 20),
                description=f"用户 {user_id} 事件频率异常高: {events_per_minute:.2f}/分钟",
                anomaly_type="frequency",
                score=events_per_minute
            )
            anomalies.append(anomaly)
        
        # 检测低频异常
        elif events_per_minute < self.normal_thresholds['events_per_minute'][0]:
            anomaly = SimpleAnomalyResult(
                anomaly_id=f"low_freq_{user_id}_{int(utc_now().timestamp())}",
                user_id=user_id,
                event_type="low_frequency",
                timestamp=utc_now(),
                severity="medium",
                confidence=0.7,
                description=f"用户 {user_id} 事件频率异常低: {events_per_minute:.2f}/分钟",
                anomaly_type="frequency",
                score=1.0 / max(0.1, events_per_minute)
            )
            anomalies.append(anomaly)
        
        # 检测事件类型异常
        event_types = set(event.get('event_type', 'unknown') for event in events)
        unique_types = len(event_types)
        
        if unique_types > self.normal_thresholds['event_types_per_session'][1]:
            anomaly = SimpleAnomalyResult(
                anomaly_id=f"types_{user_id}_{int(utc_now().timestamp())}",
                user_id=user_id,
                event_type="diverse_behavior",
                timestamp=utc_now(),
                severity="medium",
                confidence=min(0.8, unique_types / 20),
                description=f"用户 {user_id} 行为模式过于多样化: {unique_types}种事件类型",
                anomaly_type="pattern",
                score=unique_types
            )
            anomalies.append(anomaly)
        
        # 随机生成一些其他类型的异常用于演示
        if random.random() < 0.1:  # 10%概率
            anomaly_types = [
                ("timing", "时间模式异常", "pattern"),
                ("session", "会话行为异常", "session"),
                ("sequence", "事件序列异常", "sequence")
            ]
            
            anomaly_type, description, category = random.choice(anomaly_types)
            
            anomaly = SimpleAnomalyResult(
                anomaly_id=f"{anomaly_type}_{user_id}_{int(utc_now().timestamp())}",
                user_id=user_id,
                event_type=anomaly_type,
                timestamp=utc_now(),
                severity=random.choice(["low", "medium", "high"]),
                confidence=random.uniform(0.6, 0.9),
                description=f"用户 {user_id} {description}",
                anomaly_type=category,
                score=random.uniform(1.0, 5.0)
            )
            anomalies.append(anomaly)
        
        return anomalies

def create_sample_events(num_users: int = 50, num_events: int = 1000) -> List[Dict[str, Any]]:
    """创建示例事件数据"""
    event_types = [
        'page_view', 'click', 'form_submit', 'scroll', 'hover', 
        'download', 'search', 'login', 'logout', 'purchase'
    ]
    
    events = []
    now = utc_now()
    
    for i in range(num_events):
        # 为前10%的事件创建一些异常模式
        if i < num_events * 0.1:
            # 高频用户
            user_id = f"user_{random.randint(1, 5)}"
        else:
            user_id = f"user_{random.randint(1, num_users)}"
        
        event = {
            'event_id': f"event_{i}",
            'user_id': user_id,
            'event_type': random.choice(event_types),
            'timestamp': now - timedelta(seconds=random.randint(0, 3600)),
            'properties': {
                'page': f"/page{random.randint(1, 10)}",
                'duration': random.randint(100, 5000),
                'source': random.choice(['direct', 'search', 'social', 'referral'])
            }
        }
        events.append(event)
    
    return events
