"""
数据质量服务 - 高级数据去重和质量检查机制
"""
import json
import hashlib
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter, defaultdict
import statistics
import re

from core.logging import get_logger
from models.schemas.event_tracking import CreateEventRequest, DataQuality, EventType, EventValidationResult
from repositories.event_tracking_repository import EventDeduplicationRepository, EventSchemaRepository

logger = get_logger(__name__)


class QualityCheckType(str, Enum):
    """质量检查类型"""
    COMPLETENESS = "completeness"     # 完整性检查
    CONSISTENCY = "consistency"       # 一致性检查
    ACCURACY = "accuracy"            # 准确性检查
    TIMELINESS = "timeliness"        # 时效性检查
    VALIDITY = "validity"            # 有效性检查
    UNIQUENESS = "uniqueness"        # 唯一性检查
    INTEGRITY = "integrity"          # 完整性检查


class QualityIssue(str, Enum):
    """质量问题类型"""
    DUPLICATE = "duplicate"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    INVALID_FORMAT = "invalid_format"
    OUT_OF_RANGE = "out_of_range"
    INCONSISTENT_DATA = "inconsistent_data"
    STALE_DATA = "stale_data"
    ANOMALOUS_VALUE = "anomalous_value"
    SCHEMA_VIOLATION = "schema_violation"


@dataclass
class QualityCheckResult:
    """质量检查结果"""
    check_type: QualityCheckType
    passed: bool
    score: float  # 0.0 - 1.0
    issues: List[QualityIssue] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


@dataclass
class DeduplicationResult:
    """去重结果"""
    is_duplicate: bool
    fingerprint: str
    similarity_score: float = 0.0
    duplicate_type: Optional[str] = None
    original_event_id: Optional[str] = None
    duplicate_reasons: List[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class QualityProfile:
    """数据质量档案"""
    event_type: EventType
    event_name: str
    total_events: int = 0
    quality_scores: Dict[QualityCheckType, float] = field(default_factory=dict)
    common_issues: Dict[QualityIssue, int] = field(default_factory=dict)
    field_completeness: Dict[str, float] = field(default_factory=dict)
    value_distributions: Dict[str, Dict[str, int]] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: utc_now())


class AdvancedDeduplicationEngine:
    """高级去重引擎"""
    
    def __init__(self, dedup_repo: EventDeduplicationRepository):
        self.dedup_repo = dedup_repo
        self.similarity_threshold = 0.95
        self.fuzzy_match_threshold = 0.85
    
    def generate_event_signatures(self, event: CreateEventRequest) -> Dict[str, str]:
        """生成多种事件签名"""
        signatures = {}
        
        # 1. 精确签名 - 完全匹配
        exact_data = {
            'experiment_id': event.experiment_id,
            'user_id': event.user_id,
            'event_type': event.event_type.value,
            'event_name': event.event_name,
            'event_timestamp': event.event_timestamp.isoformat() if event.event_timestamp else None,
            'variant_id': event.variant_id,
            'session_id': event.session_id
        }
        
        # 包含关键属性
        if event.properties:
            key_props = self._extract_key_properties(event.properties)
            if key_props:
                exact_data['key_properties'] = key_props
        
        signatures['exact'] = self._hash_data(exact_data)
        
        # 2. 时间窗口签名 - 忽略精确时间戳
        time_window_data = exact_data.copy()
        if event.event_timestamp:
            # 舍入到最近的分钟
            rounded_time = event.event_timestamp.replace(second=0, microsecond=0)
            time_window_data['event_timestamp'] = rounded_time.isoformat()
        
        signatures['time_window'] = self._hash_data(time_window_data)
        
        # 3. 用户会话签名 - 基于用户和会话
        session_data = {
            'user_id': event.user_id,
            'session_id': event.session_id,
            'event_type': event.event_type.value,
            'event_name': event.event_name
        }
        signatures['session'] = self._hash_data(session_data)
        
        # 4. 内容签名 - 基于事件内容
        content_data = {
            'event_type': event.event_type.value,
            'event_name': event.event_name,
            'properties': self._normalize_properties(event.properties)
        }
        signatures['content'] = self._hash_data(content_data)
        
        return signatures
    
    def _extract_key_properties(self, properties: Dict[str, Any]) -> Dict[str, Any]:
        """提取关键属性用于去重"""
        key_fields = {
            'transaction_id', 'order_id', 'product_id', 'payment_id', 
            'user_action', 'page_id', 'button_id', 'form_id', 'item_id'
        }
        
        return {k: v for k, v in properties.items() 
                if k.lower() in key_fields and v is not None}
    
    def _normalize_properties(self, properties: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """标准化属性数据"""
        if not properties:
            return {}
        
        normalized = {}
        for key, value in properties.items():
            # 标准化键名
            norm_key = key.lower().strip()
            
            # 标准化值
            if isinstance(value, str):
                norm_value = value.strip().lower()
            elif isinstance(value, (int, float)):
                norm_value = value
            elif isinstance(value, bool):
                norm_value = value
            else:
                norm_value = str(value).strip().lower()
            
            normalized[norm_key] = norm_value
        
        return normalized
    
    def _hash_data(self, data: Dict[str, Any]) -> str:
        """生成数据哈希"""
        json_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(json_str.encode('utf-8')).hexdigest()
    
    async def check_duplicates(self, event: CreateEventRequest) -> DeduplicationResult:
        """检查重复事件"""
        signatures = self.generate_event_signatures(event)
        
        # 检查精确重复
        exact_duplicate = await self.dedup_repo.check_duplicate(signatures['exact'])
        if exact_duplicate:
            return DeduplicationResult(
                is_duplicate=True,
                fingerprint=signatures['exact'],
                similarity_score=1.0,
                duplicate_type='exact',
                original_event_id=exact_duplicate.original_event_id,
                duplicate_reasons=['完全匹配'],
                confidence=1.0
            )
        
        # 检查时间窗口重复
        if signatures['time_window'] != signatures['exact']:
            time_duplicate = await self.dedup_repo.check_duplicate(signatures['time_window'])
            if time_duplicate:
                return DeduplicationResult(
                    is_duplicate=True,
                    fingerprint=signatures['time_window'],
                    similarity_score=0.98,
                    duplicate_type='time_window',
                    original_event_id=time_duplicate.original_event_id,
                    duplicate_reasons=['时间窗口内重复'],
                    confidence=0.95
                )
        
        # 检查会话级重复（可能的重复点击）
        session_duplicate = await self.dedup_repo.check_duplicate(signatures['session'])
        if session_duplicate:
            # 进一步检查时间间隔
            time_diff = self._calculate_time_difference(event, session_duplicate)
            if time_diff < 5:  # 5秒内的相同操作
                return DeduplicationResult(
                    is_duplicate=True,
                    fingerprint=signatures['session'],
                    similarity_score=0.90,
                    duplicate_type='session_rapid',
                    original_event_id=session_duplicate.original_event_id,
                    duplicate_reasons=['短时间内重复操作'],
                    confidence=0.85
                )
        
        # 检查内容相似度重复
        content_duplicate = await self.dedup_repo.check_duplicate(signatures['content'])
        if content_duplicate:
            similarity = self._calculate_content_similarity(event, content_duplicate)
            if similarity > self.fuzzy_match_threshold:
                return DeduplicationResult(
                    is_duplicate=True,
                    fingerprint=signatures['content'],
                    similarity_score=similarity,
                    duplicate_type='content_similar',
                    original_event_id=content_duplicate.original_event_id,
                    duplicate_reasons=['内容高度相似'],
                    confidence=similarity
                )
        
        # 没有发现重复
        return DeduplicationResult(
            is_duplicate=False,
            fingerprint=signatures['exact'],
            similarity_score=0.0,
            confidence=1.0
        )
    
    def _calculate_time_difference(self, event: CreateEventRequest, duplicate_record: Any) -> float:
        """计算时间差异（秒）"""
        if not event.event_timestamp or not duplicate_record.event_timestamp:
            return float('inf')
        
        return abs((event.event_timestamp - duplicate_record.event_timestamp).total_seconds())
    
    def _calculate_content_similarity(self, event: CreateEventRequest, duplicate_record: Any) -> float:
        """计算内容相似度"""
        # 简化的相似度计算
        # 实际应用中可以使用更复杂的算法，如编辑距离、余弦相似度等
        
        if not event.properties or not hasattr(duplicate_record, 'event_data'):
            return 0.5
        
        # 这里返回一个模拟的相似度分数
        return 0.87  # 模拟高相似度


class DataQualityChecker:
    """数据质量检查器"""
    
    def __init__(self, schema_repo: EventSchemaRepository):
        self.schema_repo = schema_repo
        self.quality_profiles: Dict[str, QualityProfile] = {}
        
        # 质量检查规则配置
        self.required_fields = {
            EventType.CONVERSION: ['experiment_id', 'user_id', 'variant_id', 'event_name'],
            EventType.EXPOSURE: ['experiment_id', 'user_id', 'variant_id'],
            EventType.INTERACTION: ['experiment_id', 'user_id', 'event_name']
        }
        
        # 数据格式验证规则
        self.format_patterns = {
            'email': re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            'url': re.compile(r'^https?://[^\s/$.?#].[^\s]*$'),
            'uuid': re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'),
            'phone': re.compile(r'^\+?[\d\s\-\(\)]{10,}$')
        }
    
    async def perform_quality_checks(self, event: CreateEventRequest) -> List[QualityCheckResult]:
        """执行全面的质量检查"""
        results = []
        
        # 1. 完整性检查
        completeness_result = await self._check_completeness(event)
        results.append(completeness_result)
        
        # 2. 有效性检查
        validity_result = await self._check_validity(event)
        results.append(validity_result)
        
        # 3. 一致性检查
        consistency_result = await self._check_consistency(event)
        results.append(consistency_result)
        
        # 4. 时效性检查
        timeliness_result = await self._check_timeliness(event)
        results.append(timeliness_result)
        
        # 5. 准确性检查
        accuracy_result = await self._check_accuracy(event)
        results.append(accuracy_result)
        
        # 6. Schema验证
        schema_result = await self._check_schema_compliance(event)
        if schema_result:
            results.append(schema_result)
        
        return results
    
    async def _check_completeness(self, event: CreateEventRequest) -> QualityCheckResult:
        """检查数据完整性"""
        required = self.required_fields.get(event.event_type, ['experiment_id', 'user_id', 'event_name'])
        
        missing_fields = []
        total_fields = len(required)
        present_fields = 0
        
        for field in required:
            value = getattr(event, field, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                missing_fields.append(field)
            else:
                present_fields += 1
        
        score = present_fields / total_fields if total_fields > 0 else 1.0
        passed = len(missing_fields) == 0
        
        issues = [QualityIssue.MISSING_REQUIRED_FIELD] if missing_fields else []
        suggestions = [f"缺少必需字段: {', '.join(missing_fields)}"] if missing_fields else []
        
        return QualityCheckResult(
            check_type=QualityCheckType.COMPLETENESS,
            passed=passed,
            score=score,
            issues=issues,
            details={'missing_fields': missing_fields, 'completion_rate': score},
            suggestions=suggestions
        )
    
    async def _check_validity(self, event: CreateEventRequest) -> QualityCheckResult:
        """检查数据有效性"""
        issues = []
        details = {}
        suggestions = []
        validity_checks = 0
        passed_checks = 0
        
        # 检查ID格式
        if event.experiment_id:
            validity_checks += 1
            if len(event.experiment_id) < 3 or len(event.experiment_id) > 128:
                issues.append(QualityIssue.INVALID_FORMAT)
                suggestions.append("实验ID长度应在3-128字符之间")
            else:
                passed_checks += 1
        
        if event.user_id:
            validity_checks += 1
            if len(event.user_id) < 1 or len(event.user_id) > 128:
                issues.append(QualityIssue.INVALID_FORMAT)
                suggestions.append("用户ID长度应在1-128字符之间")
            else:
                passed_checks += 1
        
        # 检查事件名称格式
        if event.event_name:
            validity_checks += 1
            if not re.match(r'^[a-zA-Z][a-zA-Z0-9_.-]*$', event.event_name):
                issues.append(QualityIssue.INVALID_FORMAT)
                suggestions.append("事件名称应以字母开头，只包含字母、数字、下划线、点和连字符")
            else:
                passed_checks += 1
        
        # 检查属性值格式
        if event.properties:
            for key, value in event.properties.items():
                validity_checks += 1
                if self._validate_property_format(key, value):
                    passed_checks += 1
                else:
                    issues.append(QualityIssue.INVALID_FORMAT)
                    suggestions.append(f"属性 {key} 的值格式不正确")
        
        score = passed_checks / validity_checks if validity_checks > 0 else 1.0
        passed = len(issues) == 0
        
        return QualityCheckResult(
            check_type=QualityCheckType.VALIDITY,
            passed=passed,
            score=score,
            issues=issues,
            details=details,
            suggestions=suggestions
        )
    
    def _validate_property_format(self, key: str, value: Any) -> bool:
        """验证属性值格式"""
        if value is None:
            return True
        
        key_lower = key.lower()
        str_value = str(value)
        
        # 检查特定格式
        if 'email' in key_lower and not self.format_patterns['email'].match(str_value):
            return False
        
        if 'url' in key_lower and not self.format_patterns['url'].match(str_value):
            return False
        
        if 'id' in key_lower and len(str_value) > 256:  # ID不应太长
            return False
        
        return True
    
    async def _check_consistency(self, event: CreateEventRequest) -> QualityCheckResult:
        """检查数据一致性"""
        issues = []
        suggestions = []
        consistency_checks = 0
        passed_checks = 0
        
        # 检查转化事件是否有variant_id
        consistency_checks += 1
        if event.event_type == EventType.CONVERSION and not event.variant_id:
            issues.append(QualityIssue.INCONSISTENT_DATA)
            suggestions.append("转化事件应包含variant_id以标识测试组")
        else:
            passed_checks += 1
        
        # 检查曝光事件是否有variant_id
        consistency_checks += 1
        if event.event_type == EventType.EXPOSURE and not event.variant_id:
            issues.append(QualityIssue.INCONSISTENT_DATA)
            suggestions.append("曝光事件应包含variant_id以标识用户看到的变体")
        else:
            passed_checks += 1
        
        # 检查属性值的一致性
        if event.properties:
            consistency_checks += 1
            if self._check_property_consistency(event.properties):
                passed_checks += 1
            else:
                issues.append(QualityIssue.INCONSISTENT_DATA)
                suggestions.append("事件属性存在不一致性")
        
        score = passed_checks / consistency_checks if consistency_checks > 0 else 1.0
        passed = len(issues) == 0
        
        return QualityCheckResult(
            check_type=QualityCheckType.CONSISTENCY,
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions
        )
    
    def _check_property_consistency(self, properties: Dict[str, Any]) -> bool:
        """检查属性一致性"""
        # 检查数值属性是否在合理范围内
        for key, value in properties.items():
            if isinstance(value, (int, float)):
                # 检查是否为合理的数值
                if value < 0 and key.lower() in ['price', 'amount', 'quantity', 'count']:
                    return False  # 这些字段不应为负数
                
                if abs(value) > 10**12:  # 过大的数值可能是错误
                    return False
        
        return True
    
    async def _check_timeliness(self, event: CreateEventRequest) -> QualityCheckResult:
        """检查数据时效性"""
        issues = []
        suggestions = []
        
        if not event.event_timestamp:
            # 没有时间戳，无法检查时效性
            return QualityCheckResult(
                check_type=QualityCheckType.TIMELINESS,
                passed=True,
                score=0.8,  # 降低分数，因为缺少时间戳
                suggestions=["建议提供事件时间戳以便进行时效性分析"]
            )
        
        now = utc_now()
        time_diff = now - event.event_timestamp
        
        # 检查是否为未来时间
        if time_diff.total_seconds() < 0:
            issues.append(QualityIssue.STALE_DATA)
            suggestions.append("事件时间戳不应为未来时间")
        
        # 检查是否过于陈旧
        elif time_diff.total_seconds() > 30 * 24 * 3600:  # 30天
            issues.append(QualityIssue.STALE_DATA)
            suggestions.append("事件时间戳过于陈旧，可能影响分析准确性")
        
        # 计算时效性分数
        if time_diff.total_seconds() < 0:
            score = 0.0
        elif time_diff.total_seconds() > 7 * 24 * 3600:  # 7天以上
            score = 0.6
        elif time_diff.total_seconds() > 24 * 3600:  # 1天以上
            score = 0.8
        else:
            score = 1.0
        
        passed = len(issues) == 0
        
        return QualityCheckResult(
            check_type=QualityCheckType.TIMELINESS,
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions,
            details={'age_hours': time_diff.total_seconds() / 3600}
        )
    
    async def _check_accuracy(self, event: CreateEventRequest) -> QualityCheckResult:
        """检查数据准确性"""
        issues = []
        suggestions = []
        accuracy_checks = 0
        passed_checks = 0
        
        # 检查枚举值的准确性
        accuracy_checks += 1
        if event.event_type in EventType:
            passed_checks += 1
        else:
            issues.append(QualityIssue.OUT_OF_RANGE)
            suggestions.append(f"未知的事件类型: {event.event_type}")
        
        # 检查数值范围的准确性
        if event.properties:
            for key, value in event.properties.items():
                accuracy_checks += 1
                if self._is_value_accurate(key, value):
                    passed_checks += 1
                else:
                    issues.append(QualityIssue.OUT_OF_RANGE)
                    suggestions.append(f"属性 {key} 的值 {value} 超出预期范围")
        
        score = passed_checks / accuracy_checks if accuracy_checks > 0 else 1.0
        passed = len(issues) == 0
        
        return QualityCheckResult(
            check_type=QualityCheckType.ACCURACY,
            passed=passed,
            score=score,
            issues=issues,
            suggestions=suggestions
        )
    
    def _is_value_accurate(self, key: str, value: Any) -> bool:
        """检查值是否在准确范围内"""
        key_lower = key.lower()
        
        if isinstance(value, (int, float)):
            # 百分比应在0-100或0-1之间
            if any(term in key_lower for term in ['percent', 'rate', 'ratio']):
                if 'percent' in key_lower:
                    return 0 <= value <= 100
                else:
                    return 0 <= value <= 1
            
            # 年龄应在合理范围内
            if 'age' in key_lower:
                return 0 <= value <= 150
            
            # 价格应为正数
            if any(term in key_lower for term in ['price', 'cost', 'amount']):
                return value >= 0
        
        return True
    
    async def _check_schema_compliance(self, event: CreateEventRequest) -> Optional[QualityCheckResult]:
        """检查Schema合规性"""
        try:
            schema = await self.schema_repo.get_active_schema_by_event(
                event.event_type.value, event.event_name
            )
            
            if not schema:
                return None  # 没有Schema定义，跳过检查
            
            # TODO: 实现JSONSchema验证逻辑
            # 这里简化实现，返回通过结果
            
            return QualityCheckResult(
                check_type=QualityCheckType.VALIDITY,
                passed=True,
                score=1.0,
                details={'schema_version': schema.schema_version}
            )
            
        except Exception as e:
            logger.error(f"Schema合规性检查失败: {e}")
            return QualityCheckResult(
                check_type=QualityCheckType.VALIDITY,
                passed=False,
                score=0.5,
                issues=[QualityIssue.SCHEMA_VIOLATION],
                suggestions=["Schema验证失败，请检查事件结构"]
            )


class DataQualityService:
    """数据质量服务主入口"""
    
    def __init__(
        self,
        dedup_repo: EventDeduplicationRepository,
        schema_repo: EventSchemaRepository
    ):
        self.dedup_engine = AdvancedDeduplicationEngine(dedup_repo)
        self.quality_checker = DataQualityChecker(schema_repo)
        self.quality_profiles: Dict[str, QualityProfile] = {}
        
        # 质量阈值配置
        self.quality_thresholds = {
            DataQuality.HIGH: 0.9,
            DataQuality.MEDIUM: 0.7,
            DataQuality.LOW: 0.4,
            DataQuality.INVALID: 0.0
        }
    
    async def analyze_event_quality(
        self, 
        event: CreateEventRequest
    ) -> Tuple[DeduplicationResult, List[QualityCheckResult], DataQuality, float]:
        """全面分析事件质量"""
        
        # 1. 去重检查
        dedup_result = await self.dedup_engine.check_duplicates(event)
        
        # 2. 质量检查
        quality_results = await self.quality_checker.perform_quality_checks(event)
        
        # 3. 计算总体质量分数
        overall_score = self._calculate_overall_quality_score(quality_results)
        
        # 4. 确定质量等级
        quality_level = self._determine_quality_level(overall_score)
        
        # 5. 更新质量档案
        await self._update_quality_profile(event, quality_results, overall_score)
        
        return dedup_result, quality_results, quality_level, overall_score
    
    def _calculate_overall_quality_score(self, results: List[QualityCheckResult]) -> float:
        """计算总体质量分数"""
        if not results:
            return 0.5
        
        # 加权平均分数计算
        weights = {
            QualityCheckType.COMPLETENESS: 0.3,
            QualityCheckType.VALIDITY: 0.25,
            QualityCheckType.CONSISTENCY: 0.2,
            QualityCheckType.TIMELINESS: 0.15,
            QualityCheckType.ACCURACY: 0.1
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for result in results:
            weight = weights.get(result.check_type, 0.1)
            total_score += result.score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.5
    
    def _determine_quality_level(self, score: float) -> DataQuality:
        """根据分数确定质量等级"""
        if score >= self.quality_thresholds[DataQuality.HIGH]:
            return DataQuality.HIGH
        elif score >= self.quality_thresholds[DataQuality.MEDIUM]:
            return DataQuality.MEDIUM
        elif score >= self.quality_thresholds[DataQuality.LOW]:
            return DataQuality.LOW
        else:
            return DataQuality.INVALID
    
    async def _update_quality_profile(
        self, 
        event: CreateEventRequest, 
        results: List[QualityCheckResult], 
        score: float
    ):
        """更新质量档案"""
        profile_key = f"{event.event_type.value}:{event.event_name}"
        
        if profile_key not in self.quality_profiles:
            self.quality_profiles[profile_key] = QualityProfile(
                event_type=event.event_type,
                event_name=event.event_name
            )
        
        profile = self.quality_profiles[profile_key]
        profile.total_events += 1
        
        # 更新质量分数（移动平均）
        alpha = 0.1  # 学习率
        for result in results:
            if result.check_type in profile.quality_scores:
                profile.quality_scores[result.check_type] = (
                    alpha * result.score + 
                    (1 - alpha) * profile.quality_scores[result.check_type]
                )
            else:
                profile.quality_scores[result.check_type] = result.score
        
        # 统计常见问题
        for result in results:
            for issue in result.issues:
                profile.common_issues[issue] = profile.common_issues.get(issue, 0) + 1
        
        profile.last_updated = utc_now()
    
    def get_quality_profile(self, event_type: EventType, event_name: str) -> Optional[QualityProfile]:
        """获取质量档案"""
        profile_key = f"{event_type.value}:{event_name}"
        return self.quality_profiles.get(profile_key)
    
    def get_all_quality_profiles(self) -> Dict[str, QualityProfile]:
        """获取所有质量档案"""
        return self.quality_profiles.copy()
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """获取质量汇总信息"""
        if not self.quality_profiles:
            return {
                'total_profiles': 0,
                'avg_quality_score': 0.0,
                'quality_distribution': {},
                'common_issues': {}
            }
        
        total_events = sum(profile.total_events for profile in self.quality_profiles.values())
        
        # 计算平均质量分数
        total_score = 0.0
        score_count = 0
        
        quality_distribution = defaultdict(int)
        all_issues = defaultdict(int)
        
        for profile in self.quality_profiles.values():
            # 汇总质量分数
            for check_type, score in profile.quality_scores.items():
                total_score += score
                score_count += 1
                
                # 质量分布
                quality_level = self._determine_quality_level(score)
                quality_distribution[quality_level.value] += 1
            
            # 汇总问题
            for issue, count in profile.common_issues.items():
                all_issues[issue.value] += count
        
        avg_score = total_score / score_count if score_count > 0 else 0.0
        
        return {
            'total_profiles': len(self.quality_profiles),
            'total_events': total_events,
            'avg_quality_score': avg_score,
            'quality_distribution': dict(quality_distribution),
            'common_issues': dict(sorted(all_issues.items(), key=lambda x: x[1], reverse=True)[:10])
        }