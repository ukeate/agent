"""
数据采集质量监控器

负责监控数据采集过程中的质量指标，包括完整性、准确性、及时性等。
"""

import asyncio
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import statistics
from ..models import BehaviorEvent, EventType

from src.core.logging import get_logger
logger = get_logger(__name__)

class QualityIssueType(str, Enum):
    """数据质量问题类型"""
    DUPLICATE_EVENT = "duplicate_event"
    MISSING_FIELD = "missing_field"
    INVALID_FORMAT = "invalid_format"
    LATE_ARRIVAL = "late_arrival"
    OUT_OF_ORDER = "out_of_order"
    SUSPICIOUS_PATTERN = "suspicious_pattern"
    HIGH_FREQUENCY = "high_frequency"
    SCHEMA_VIOLATION = "schema_violation"

class QualitySeverity(str, Enum):
    """质量问题严重程度"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class QualityIssue:
    """数据质量问题"""
    issue_id: str
    issue_type: QualityIssueType
    severity: QualitySeverity
    event_id: str
    user_id: str
    session_id: str
    description: str
    detected_at: datetime
    metadata: Dict[str, Any]

@dataclass
class QualityMetrics:
    """数据质量指标"""
    total_events: int = 0
    valid_events: int = 0
    invalid_events: int = 0
    duplicate_events: int = 0
    late_events: int = 0
    out_of_order_events: int = 0
    
    # 完整性指标
    completeness_score: float = 1.0  # 0-1
    
    # 准确性指标
    accuracy_score: float = 1.0  # 0-1
    
    # 及时性指标
    timeliness_score: float = 1.0  # 0-1
    avg_latency_seconds: float = 0.0
    
    # 一致性指标
    consistency_score: float = 1.0  # 0-1
    
    # 唯一性指标
    uniqueness_score: float = 1.0  # 0-1
    
    # 总体质量分数
    overall_quality_score: float = 1.0  # 0-1

class QualityMonitor:
    """数据质量监控器"""
    
    def __init__(
        self,
        max_latency_seconds: int = 300,  # 5分钟
        duplicate_detection_window: int = 3600,  # 1小时
        max_event_frequency_per_second: int = 100,
        quality_history_size: int = 1000
    ):
        self.max_latency_seconds = max_latency_seconds
        self.duplicate_detection_window = duplicate_detection_window
        self.max_event_frequency_per_second = max_event_frequency_per_second
        self.quality_history_size = quality_history_size
        
        # 质量监控状态
        self.metrics = QualityMetrics()
        self.issues: List[QualityIssue] = []
        self.quality_history: deque[QualityMetrics] = deque(maxlen=quality_history_size)
        
        # 重复检测缓存
        self.event_hashes: Dict[str, datetime] = {}
        self.user_event_sequences: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        
        # 频率监控
        self.user_event_counts: Dict[str, List[datetime]] = defaultdict(list)
        
        # 统计信息
        self.stats = {
            'monitoring_start_time': utc_now(),
            'total_checks': 0,
            'issues_detected': 0,
            'false_positives': 0
        }
    
    async def check_event_quality(self, event: BehaviorEvent) -> Tuple[bool, List[QualityIssue]]:
        """检查单个事件的质量"""
        issues = []
        is_valid = True
        
        self.stats['total_checks'] += 1
        
        try:
            # 1. 完整性检查
            completeness_issues = await self._check_completeness(event)
            issues.extend(completeness_issues)
            
            # 2. 准确性检查
            accuracy_issues = await self._check_accuracy(event)
            issues.extend(accuracy_issues)
            
            # 3. 及时性检查
            timeliness_issues = await self._check_timeliness(event)
            issues.extend(timeliness_issues)
            
            # 4. 唯一性检查
            uniqueness_issues = await self._check_uniqueness(event)
            issues.extend(uniqueness_issues)
            
            # 5. 一致性检查
            consistency_issues = await self._check_consistency(event)
            issues.extend(consistency_issues)
            
            # 6. 频率检查
            frequency_issues = await self._check_frequency(event)
            issues.extend(frequency_issues)
            
            # 判断事件是否有效
            critical_issues = [issue for issue in issues if issue.severity == QualitySeverity.CRITICAL]
            if critical_issues:
                is_valid = False
            
            # 更新指标
            await self._update_metrics(event, issues)
            
            # 记录问题
            if issues:
                self.issues.extend(issues)
                self.stats['issues_detected'] += len(issues)
                
                # 保持问题列表大小
                if len(self.issues) > 10000:
                    self.issues = self.issues[-5000:]
            
            return is_valid, issues
            
        except Exception as e:
            logger.error(f"质量检查失败: {e}")
            return False, []
    
    async def _check_completeness(self, event: BehaviorEvent) -> List[QualityIssue]:
        """检查数据完整性"""
        issues = []
        
        # 检查必填字段
        required_fields = ['user_id', 'session_id', 'event_type', 'event_name', 'timestamp']
        
        for field in required_fields:
            value = getattr(event, field, None)
            if not value:
                issue = QualityIssue(
                    issue_id=f"missing_{field}_{event.event_id}",
                    issue_type=QualityIssueType.MISSING_FIELD,
                    severity=QualitySeverity.HIGH,
                    event_id=event.event_id,
                    user_id=event.user_id or "unknown",
                    session_id=event.session_id or "unknown",
                    description=f"缺少必填字段: {field}",
                    detected_at=utc_now(),
                    metadata={'field': field}
                )
                issues.append(issue)
        
        # 检查关键业务字段
        if event.event_type == EventType.USER_ACTION and not event.event_data.get('action_target'):
            issue = QualityIssue(
                issue_id=f"missing_action_target_{event.event_id}",
                issue_type=QualityIssueType.MISSING_FIELD,
                severity=QualitySeverity.MEDIUM,
                event_id=event.event_id,
                user_id=event.user_id,
                session_id=event.session_id,
                description="用户行为事件缺少目标对象信息",
                detected_at=utc_now(),
                metadata={'expected_field': 'action_target'}
            )
            issues.append(issue)
        
        return issues
    
    async def _check_accuracy(self, event: BehaviorEvent) -> List[QualityIssue]:
        """检查数据准确性"""
        issues = []
        
        # 检查时间戳格式和合理性
        now = utc_now()
        
        # 时间戳不能太久远(超过24小时)
        if event.timestamp < now - timedelta(hours=24):
            issue = QualityIssue(
                issue_id=f"old_timestamp_{event.event_id}",
                issue_type=QualityIssueType.INVALID_FORMAT,
                severity=QualitySeverity.LOW,
                event_id=event.event_id,
                user_id=event.user_id,
                session_id=event.session_id,
                description=f"事件时间戳过于久远: {event.timestamp}",
                detected_at=utc_now(),
                metadata={'timestamp': event.timestamp.isoformat()}
            )
            issues.append(issue)
        
        # 时间戳不能在未来(超过1小时)
        if event.timestamp > now + timedelta(hours=1):
            issue = QualityIssue(
                issue_id=f"future_timestamp_{event.event_id}",
                issue_type=QualityIssueType.INVALID_FORMAT,
                severity=QualitySeverity.MEDIUM,
                event_id=event.event_id,
                user_id=event.user_id,
                session_id=event.session_id,
                description=f"事件时间戳在未来: {event.timestamp}",
                detected_at=utc_now(),
                metadata={'timestamp': event.timestamp.isoformat()}
            )
            issues.append(issue)
        
        # 检查用户ID格式
        if event.user_id and (len(event.user_id) < 3 or len(event.user_id) > 100):
            issue = QualityIssue(
                issue_id=f"invalid_user_id_{event.event_id}",
                issue_type=QualityIssueType.INVALID_FORMAT,
                severity=QualitySeverity.MEDIUM,
                event_id=event.event_id,
                user_id=event.user_id,
                session_id=event.session_id,
                description=f"用户ID格式异常: {event.user_id}",
                detected_at=utc_now(),
                metadata={'user_id': event.user_id}
            )
            issues.append(issue)
        
        # 检查持续时间合理性
        if event.duration_ms is not None:
            if event.duration_ms < 0:
                issue = QualityIssue(
                    issue_id=f"negative_duration_{event.event_id}",
                    issue_type=QualityIssueType.INVALID_FORMAT,
                    severity=QualitySeverity.MEDIUM,
                    event_id=event.event_id,
                    user_id=event.user_id,
                    session_id=event.session_id,
                    description=f"事件持续时间为负数: {event.duration_ms}ms",
                    detected_at=utc_now(),
                    metadata={'duration_ms': event.duration_ms}
                )
                issues.append(issue)
            elif event.duration_ms > 3600000:  # 超过1小时
                issue = QualityIssue(
                    issue_id=f"excessive_duration_{event.event_id}",
                    issue_type=QualityIssueType.SUSPICIOUS_PATTERN,
                    severity=QualitySeverity.LOW,
                    event_id=event.event_id,
                    user_id=event.user_id,
                    session_id=event.session_id,
                    description=f"事件持续时间异常长: {event.duration_ms}ms",
                    detected_at=utc_now(),
                    metadata={'duration_ms': event.duration_ms}
                )
                issues.append(issue)
        
        return issues
    
    async def _check_timeliness(self, event: BehaviorEvent) -> List[QualityIssue]:
        """检查数据及时性"""
        issues = []
        
        # 计算事件延迟
        now = utc_now()
        latency_seconds = (now - event.timestamp).total_seconds()
        
        if latency_seconds > self.max_latency_seconds:
            severity = QualitySeverity.HIGH if latency_seconds > self.max_latency_seconds * 2 else QualitySeverity.MEDIUM
            
            issue = QualityIssue(
                issue_id=f"late_arrival_{event.event_id}",
                issue_type=QualityIssueType.LATE_ARRIVAL,
                severity=severity,
                event_id=event.event_id,
                user_id=event.user_id,
                session_id=event.session_id,
                description=f"事件延迟到达: {latency_seconds:.1f}秒",
                detected_at=utc_now(),
                metadata={'latency_seconds': latency_seconds}
            )
            issues.append(issue)
        
        return issues
    
    async def _check_uniqueness(self, event: BehaviorEvent) -> List[QualityIssue]:
        """检查数据唯一性"""
        issues = []
        
        # 生成事件指纹
        event_fingerprint = f"{event.user_id}_{event.session_id}_{event.event_type}_{event.event_name}_{event.timestamp.isoformat()}"
        
        # 检查重复事件
        now = utc_now()
        if event_fingerprint in self.event_hashes:
            last_seen = self.event_hashes[event_fingerprint]
            if (now - last_seen).total_seconds() < self.duplicate_detection_window:
                issue = QualityIssue(
                    issue_id=f"duplicate_{event.event_id}",
                    issue_type=QualityIssueType.DUPLICATE_EVENT,
                    severity=QualitySeverity.MEDIUM,
                    event_id=event.event_id,
                    user_id=event.user_id,
                    session_id=event.session_id,
                    description="检测到重复事件",
                    detected_at=utc_now(),
                    metadata={'fingerprint': event_fingerprint, 'last_seen': last_seen.isoformat()}
                )
                issues.append(issue)
        
        # 更新事件哈希缓存
        self.event_hashes[event_fingerprint] = now
        
        # 清理过期的哈希记录
        cutoff_time = now - timedelta(seconds=self.duplicate_detection_window)
        expired_keys = [k for k, v in self.event_hashes.items() if v < cutoff_time]
        for key in expired_keys:
            del self.event_hashes[key]
        
        return issues
    
    async def _check_consistency(self, event: BehaviorEvent) -> List[QualityIssue]:
        """检查数据一致性"""
        issues = []
        
        # 检查同一用户会话内的事件顺序
        user_session_key = f"{event.user_id}_{event.session_id}"
        user_events = self.user_event_sequences[user_session_key]
        
        # 检查时间顺序
        if user_events:
            last_timestamp, _ = user_events[-1]
            if event.timestamp < last_timestamp:
                issue = QualityIssue(
                    issue_id=f"out_of_order_{event.event_id}",
                    issue_type=QualityIssueType.OUT_OF_ORDER,
                    severity=QualitySeverity.LOW,
                    event_id=event.event_id,
                    user_id=event.user_id,
                    session_id=event.session_id,
                    description=f"事件时间乱序: 当前{event.timestamp}, 上一个{last_timestamp}",
                    detected_at=utc_now(),
                    metadata={
                        'current_timestamp': event.timestamp.isoformat(),
                        'last_timestamp': last_timestamp.isoformat()
                    }
                )
                issues.append(issue)
        
        # 更新用户事件序列
        user_events.append((event.timestamp, event.event_name))
        
        # 保持序列长度
        if len(user_events) > 100:
            user_events[:] = user_events[-50:]
        
        return issues
    
    async def _check_frequency(self, event: BehaviorEvent) -> List[QualityIssue]:
        """检查事件频率"""
        issues = []
        
        now = utc_now()
        user_events = self.user_event_counts[event.user_id]
        
        # 添加当前事件时间
        user_events.append(now)
        
        # 清理1分钟前的记录
        cutoff_time = now - timedelta(minutes=1)
        user_events[:] = [ts for ts in user_events if ts >= cutoff_time]
        
        # 检查频率
        events_per_minute = len(user_events)
        if events_per_minute > self.max_event_frequency_per_second * 60:  # 转换为每分钟
            issue = QualityIssue(
                issue_id=f"high_frequency_{event.event_id}",
                issue_type=QualityIssueType.HIGH_FREQUENCY,
                severity=QualitySeverity.HIGH,
                event_id=event.event_id,
                user_id=event.user_id,
                session_id=event.session_id,
                description=f"用户事件频率过高: {events_per_minute}事件/分钟",
                detected_at=utc_now(),
                metadata={'events_per_minute': events_per_minute}
            )
            issues.append(issue)
        
        return issues
    
    async def _update_metrics(self, event: BehaviorEvent, issues: List[QualityIssue]):
        """更新质量指标"""
        self.metrics.total_events += 1
        
        # 统计问题类型
        critical_issues = sum(1 for issue in issues if issue.severity == QualitySeverity.CRITICAL)
        high_issues = sum(1 for issue in issues if issue.severity == QualitySeverity.HIGH)
        
        if critical_issues == 0 and high_issues == 0:
            self.metrics.valid_events += 1
        else:
            self.metrics.invalid_events += 1
        
        # 更新具体指标
        for issue in issues:
            if issue.issue_type == QualityIssueType.DUPLICATE_EVENT:
                self.metrics.duplicate_events += 1
            elif issue.issue_type == QualityIssueType.LATE_ARRIVAL:
                self.metrics.late_events += 1
            elif issue.issue_type == QualityIssueType.OUT_OF_ORDER:
                self.metrics.out_of_order_events += 1
        
        # 计算质量分数
        await self._calculate_quality_scores()
        
        # 记录质量历史
        if len(self.quality_history) == 0 or self.metrics.total_events % 100 == 0:
            self.quality_history.append(QualityMetrics(**asdict(self.metrics)))
    
    async def _calculate_quality_scores(self):
        """计算质量分数"""
        if self.metrics.total_events == 0:
            return
        
        # 完整性分数
        self.metrics.completeness_score = self.metrics.valid_events / self.metrics.total_events
        
        # 准确性分数
        accuracy_penalty = (self.metrics.invalid_events + self.metrics.duplicate_events) / self.metrics.total_events
        self.metrics.accuracy_score = max(0.0, 1.0 - accuracy_penalty)
        
        # 及时性分数
        timeliness_penalty = self.metrics.late_events / self.metrics.total_events
        self.metrics.timeliness_score = max(0.0, 1.0 - timeliness_penalty)
        
        # 一致性分数
        consistency_penalty = self.metrics.out_of_order_events / self.metrics.total_events
        self.metrics.consistency_score = max(0.0, 1.0 - consistency_penalty)
        
        # 唯一性分数
        uniqueness_penalty = self.metrics.duplicate_events / self.metrics.total_events
        self.metrics.uniqueness_score = max(0.0, 1.0 - uniqueness_penalty)
        
        # 总体质量分数(加权平均)
        weights = {
            'completeness': 0.3,
            'accuracy': 0.25,
            'timeliness': 0.2,
            'consistency': 0.15,
            'uniqueness': 0.1
        }
        
        self.metrics.overall_quality_score = (
            weights['completeness'] * self.metrics.completeness_score +
            weights['accuracy'] * self.metrics.accuracy_score +
            weights['timeliness'] * self.metrics.timeliness_score +
            weights['consistency'] * self.metrics.consistency_score +
            weights['uniqueness'] * self.metrics.uniqueness_score
        )
    
    def get_quality_report(self) -> Dict[str, Any]:
        """获取质量报告"""
        recent_issues = [issue for issue in self.issues[-100:]]  # 最近100个问题
        
        # 按类型统计问题
        issue_type_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        
        for issue in recent_issues:
            issue_type_counts[issue.issue_type.value] += 1
            severity_counts[issue.severity.value] += 1
        
        return {
            'metrics': asdict(self.metrics),
            'statistics': {
                **self.stats,
                'monitoring_duration_hours': (utc_now() - self.stats['monitoring_start_time']).total_seconds() / 3600,
                'avg_quality_score': self.metrics.overall_quality_score,
                'recent_issues_count': len(recent_issues)
            },
            'issue_breakdown': {
                'by_type': dict(issue_type_counts),
                'by_severity': dict(severity_counts)
            },
            'quality_trends': [asdict(metrics) for metrics in list(self.quality_history)[-10:]],  # 最近10个快照
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if self.metrics.completeness_score < 0.9:
            recommendations.append("提高数据完整性：确保所有必填字段都有值，特别关注用户行为事件的目标对象信息")
        
        if self.metrics.accuracy_score < 0.9:
            recommendations.append("改善数据准确性：检查时间戳格式和用户ID格式的验证逻辑")
        
        if self.metrics.timeliness_score < 0.9:
            recommendations.append("优化数据及时性：减少事件传输延迟，考虑增加缓冲区或优化网络配置")
        
        if self.metrics.uniqueness_score < 0.95:
            recommendations.append("减少重复事件：检查客户端事件发送逻辑，避免重复提交")
        
        if self.metrics.consistency_score < 0.95:
            recommendations.append("改善数据一致性：确保客户端时钟同步，优化事件发送顺序")
        
        return recommendations
    
    def reset_monitoring(self):
        """重置监控状态"""
        self.metrics = QualityMetrics()
        self.issues = []
        self.quality_history.clear()
        self.event_hashes.clear()
        self.user_event_sequences.clear()
        self.user_event_counts.clear()
        
        self.stats = {
            'monitoring_start_time': utc_now(),
            'total_checks': 0,
            'issues_detected': 0,
            'false_positives': 0
        }
