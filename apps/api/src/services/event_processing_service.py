"""
事件处理服务 - 负责事件的验证、处理和流转
"""

from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from typing import List, Optional, Dict, Any, Tuple
import asyncio
import json
import hashlib
from enum import Enum
from dataclasses import dataclass
from jsonschema import Draft7Validator
from src.models.schemas.event_tracking import (

    CreateEventRequest, EventStatus, DataQuality, EventType,
    EventValidationResult, EventDeduplicationInfo
)
from src.models.database.event_tracking import EventStream
from src.repositories.event_tracking_repository import (
    EventStreamRepository, EventDeduplicationRepository, 
    EventSchemaRepository, EventErrorRepository

)

from src.core.logging import get_logger
logger = get_logger(__name__)

class ProcessingStage(str, Enum):
    """处理阶段"""
    VALIDATION = "validation"
    DEDUPLICATION = "deduplication"
    TRANSFORMATION = "transformation"
    ENRICHMENT = "enrichment"
    STORAGE = "storage"
    AGGREGATION = "aggregation"

@dataclass
class EventProcessingResult:
    """事件处理结果"""
    success: bool
    event_id: str
    status: EventStatus
    message: str
    errors: List[str] = None
    warnings: List[str] = None
    processing_time_ms: int = None
    stage: ProcessingStage = None

class EventValidationService:
    """事件验证服务"""
    
    def __init__(self, schema_repo: EventSchemaRepository):
        self.schema_repo = schema_repo
    
    async def validate_event(self, event: CreateEventRequest) -> EventValidationResult:
        """全面验证事件"""
        start_time = utc_now()
        errors = []
        warnings = []
        quality = DataQuality.HIGH
        quality_score = 1.0
        
        try:
            # 1. 基础字段验证
            errors.extend(await self._validate_basic_fields(event))
            
            # 2. 业务逻辑验证
            business_errors, business_warnings = await self._validate_business_logic(event)
            errors.extend(business_errors)
            warnings.extend(business_warnings)
            
            # 3. 时间戳验证
            timestamp_errors, timestamp_warnings = await self._validate_timestamp(event)
            errors.extend(timestamp_errors)
            warnings.extend(timestamp_warnings)
            
            # 4. JSON字段验证
            json_errors = await self._validate_json_fields(event)
            errors.extend(json_errors)
            
            # 5. Schema验证（如果有定义）
            schema_errors = await self._validate_schema(event)
            errors.extend(schema_errors)
            
            # 6. 数据质量评分
            quality, quality_score = self._calculate_quality_score(errors, warnings)
            
            # 计算处理时间
            duration = utc_now() - start_time
            duration_ms = int(duration.total_seconds() * 1000)
            
            return EventValidationResult(
                is_valid=len(errors) == 0,
                event_id=event.event_id,
                validation_errors=errors,
                validation_warnings=warnings,
                data_quality=quality,
                quality_score=quality_score,
                validation_duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"事件验证异常: {e}", exc_info=True)
            duration = utc_now() - start_time
            duration_ms = int(duration.total_seconds() * 1000)
            
            return EventValidationResult(
                is_valid=False,
                event_id=event.event_id,
                validation_errors=[f"验证过程异常: {str(e)}"],
                data_quality=DataQuality.INVALID,
                quality_score=0.0,
                validation_duration_ms=duration_ms
            )
    
    async def _validate_basic_fields(self, event: CreateEventRequest) -> List[str]:
        """验证基础字段"""
        errors = []
        
        # 必需字段检查
        if not event.experiment_id or len(event.experiment_id.strip()) == 0:
            errors.append("experiment_id是必需的且不能为空")
        
        if not event.user_id or len(event.user_id.strip()) == 0:
            errors.append("user_id是必需的且不能为空")
        
        if not event.event_name or len(event.event_name.strip()) == 0:
            errors.append("event_name是必需的且不能为空")
        
        # 字段长度检查
        if event.experiment_id and len(event.experiment_id) > 128:
            errors.append("experiment_id长度不能超过128字符")
        
        if event.user_id and len(event.user_id) > 128:
            errors.append("user_id长度不能超过128字符")
        
        if event.event_name and len(event.event_name) > 128:
            errors.append("event_name长度不能超过128字符")
        
        if event.event_category and len(event.event_category) > 64:
            errors.append("event_category长度不能超过64字符")
        
        # 枚举值验证
        if event.event_type not in EventType:
            errors.append(f"无效的event_type: {event.event_type}")
        
        return errors
    
    async def _validate_business_logic(self, event: CreateEventRequest) -> Tuple[List[str], List[str]]:
        """验证业务逻辑"""
        errors = []
        warnings = []
        
        # 转化事件必须有variant_id
        if event.event_type == EventType.CONVERSION and not event.variant_id:
            errors.append("转化事件必须包含variant_id")
        
        # 曝光事件必须有variant_id
        if event.event_type == EventType.EXPOSURE and not event.variant_id:
            errors.append("曝光事件必须包含variant_id")
        
        # 交互事件建议有session_id
        if event.event_type == EventType.INTERACTION and not event.session_id:
            warnings.append("交互事件建议包含session_id以便追踪用户行为")
        
        return errors, warnings
    
    async def _validate_timestamp(self, event: CreateEventRequest) -> Tuple[List[str], List[str]]:
        """验证时间戳"""
        errors = []
        warnings = []
        
        if event.event_timestamp:
            now = utc_now()
            
            # 未来时间检查
            if event.event_timestamp > now + timedelta(minutes=5):
                errors.append("事件时间戳不能超过当前时间5分钟")
            
            # 过期时间检查
            if event.event_timestamp < now - timedelta(days=365):
                errors.append("事件时间戳不能早于一年前")
            elif event.event_timestamp < now - timedelta(days=30):
                warnings.append("事件时间戳超过30天，可能影响分析准确性")
        
        return errors, warnings
    
    async def _validate_json_fields(self, event: CreateEventRequest) -> List[str]:
        """验证JSON字段"""
        errors = []
        
        # 检查JSON字段大小
        max_size = 65536  # 64KB
        json_fields = [
            ("properties", event.properties),
            ("user_properties", event.user_properties), 
            ("experiment_context", event.experiment_context)
        ]
        
        for field_name, field_value in json_fields:
            if field_value is not None:
                try:
                    json_str = json.dumps(field_value, ensure_ascii=False)
                    size_bytes = len(json_str.encode('utf-8'))
                    
                    if size_bytes > max_size:
                        errors.append(f"{field_name}数据过大：{size_bytes}字节，最大允许{max_size}字节")
                    
                    # 检查嵌套深度
                    max_depth = self._get_json_depth(field_value)
                    if max_depth > 10:
                        errors.append(f"{field_name}嵌套层级过深：{max_depth}层，最大允许10层")
                        
                except (TypeError, ValueError) as e:
                    errors.append(f"{field_name}包含不可序列化的数据: {e}")
        
        return errors
    
    def _get_json_depth(self, obj: Any, depth: int = 0) -> int:
        """计算JSON对象嵌套深度"""
        if isinstance(obj, dict):
            if not obj:
                return depth
            return max(self._get_json_depth(v, depth + 1) for v in obj.values())
        elif isinstance(obj, list):
            if not obj:
                return depth
            return max(self._get_json_depth(item, depth + 1) for item in obj)
        else:
            return depth
    
    async def _validate_schema(self, event: CreateEventRequest) -> List[str]:
        """根据预定义Schema验证事件"""
        schema = await self.schema_repo.get_active_schema_by_event(event.event_type.value, event.event_name)
        if not schema:
            return []

        try:
            validator = Draft7Validator(schema.schema_definition)
        except Exception as e:
            return [f"Schema无效: {e}"]

        instance = event.model_dump(mode="json")
        return [
            f"{'.'.join(str(p) for p in err.path)}: {err.message}" if err.path else err.message
            for err in validator.iter_errors(instance)
        ]
    
    def _calculate_quality_score(self, errors: List[str], warnings: List[str]) -> Tuple[DataQuality, float]:
        """计算数据质量等级和分数"""
        if errors:
            return DataQuality.INVALID, 0.0
        
        if len(warnings) >= 3:
            return DataQuality.LOW, 0.4
        elif len(warnings) >= 1:
            return DataQuality.MEDIUM, 0.7
        else:
            return DataQuality.HIGH, 1.0

class EventDeduplicationService:
    """事件去重服务"""
    
    def __init__(self, dedup_repo: EventDeduplicationRepository):
        self.dedup_repo = dedup_repo
    
    def generate_event_fingerprint(self, event: CreateEventRequest) -> str:
        """生成事件指纹用于去重"""
        # 构建用于指纹计算的核心数据
        fingerprint_data = {
            "experiment_id": event.experiment_id,
            "user_id": event.user_id,
            "event_type": event.event_type.value,
            "event_name": event.event_name,
            "variant_id": event.variant_id,
            "session_id": event.session_id
        }
        
        # 时间戳精确到秒，避免毫秒差异导致的重复
        if event.event_timestamp:
            fingerprint_data["event_timestamp"] = event.event_timestamp.replace(microsecond=0).isoformat()
        
        # 如果有关键属性，也加入指纹
        if event.properties:
            # 只包含稳定的属性，排除时间戳等变化字段
            stable_props = {}
            exclude_keys = {'timestamp', 'server_time', 'client_time', 'request_id'}
            
            for key, value in event.properties.items():
                if key.lower() not in exclude_keys:
                    stable_props[key] = value
            
            if stable_props:
                fingerprint_data["key_properties"] = stable_props
        
        # 生成SHA-256指纹
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(fingerprint_str.encode('utf-8')).hexdigest()
    
    async def check_duplicate(self, event: CreateEventRequest) -> EventDeduplicationInfo:
        """检查事件是否重复"""
        fingerprint = self.generate_event_fingerprint(event)
        
        try:
            # 查询去重记录
            duplicate_info = await self.dedup_repo.check_duplicate(fingerprint)
            
            if duplicate_info:
                # 更新重复计数
                await self.dedup_repo.update_duplicate_count(fingerprint)
                
                return EventDeduplicationInfo(
                    event_fingerprint=fingerprint,
                    is_duplicate=True,
                    original_event_id=duplicate_info.original_event_id,
                    duplicate_count=duplicate_info.duplicate_count + 1,
                    first_seen_at=duplicate_info.first_seen_at,
                    last_duplicate_at=utc_now()
                )
            else:
                # 记录新事件指纹
                await self.dedup_repo.record_event_fingerprint(
                    fingerprint=fingerprint,
                    original_event_id=event.event_id,
                    experiment_id=event.experiment_id,
                    user_id=event.user_id,
                    event_timestamp=event.event_timestamp or utc_now()
                )
                
                return EventDeduplicationInfo(
                    event_fingerprint=fingerprint,
                    is_duplicate=False,
                    duplicate_count=0
                )
        
        except Exception as e:
            logger.error(f"去重检查失败: {e}")
            # 失败时返回非重复，避免误删除事件
            return EventDeduplicationInfo(
                event_fingerprint=fingerprint,
                is_duplicate=False,
                duplicate_count=0
            )

class EventEnrichmentService:
    """事件增强服务 - 丰富事件数据"""
    
    async def enrich_event(self, event: CreateEventRequest, client_ip: str = None) -> CreateEventRequest:
        """增强事件数据"""
        try:
            # 1. 地理位置信息增强
            if client_ip and not event.geo_info:
                geo_info = await self._enrich_geo_info(client_ip)
                if geo_info:
                    event.geo_info = geo_info
            
            # 2. 设备信息增强
            if event.client_info and event.client_info.user_agent and not event.device_info:
                device_info = await self._parse_user_agent(event.client_info.user_agent)
                if device_info:
                    event.device_info = device_info
            
            # 3. 会话信息增强
            if not event.session_id:
                event.session_id = self._generate_session_id(event.user_id, event.client_info)
            
            # 4. 实验上下文增强
            event.experiment_context = await self._enrich_experiment_context(event)
            
            return event
            
        except Exception as e:
            logger.warning(f"事件增强失败，使用原始事件: {e}")
            return event
    
    async def _enrich_geo_info(self, ip_address: str) -> Optional[Dict[str, Any]]:
        """通过IP地址获取地理位置信息"""
        import ipaddress

        try:
            ip = ipaddress.ip_address(ip_address)
        except Exception:
            return None

        return {
            "ip": ip_address,
            "version": int(getattr(ip, "version", 0) or 0),
            "is_private": bool(getattr(ip, "is_private", False)),
            "is_global": bool(getattr(ip, "is_global", False)),
            "is_loopback": bool(getattr(ip, "is_loopback", False)),
            "is_multicast": bool(getattr(ip, "is_multicast", False)),
            "is_reserved": bool(getattr(ip, "is_reserved", False)),
        }
    
    async def _parse_user_agent(self, user_agent: str) -> Optional[Dict[str, Any]]:
        """解析User Agent获取设备信息"""
        import re

        ua = (user_agent or "").strip()
        if not ua:
            return None

        ua_lower = ua.lower()
        is_bot = any(k in ua_lower for k in ["bot", "crawler", "spider", "slurp"])

        device_type = "desktop"
        if any(k in ua_lower for k in ["ipad", "tablet"]):
            device_type = "tablet"
        elif any(k in ua_lower for k in ["mobile", "android", "iphone"]):
            device_type = "mobile"

        os_name = "unknown"
        os_version = None
        if "windows nt" in ua_lower:
            os_name = "windows"
            m = re.search(r"windows nt ([0-9.]+)", ua_lower)
            os_version = m.group(1) if m else None
        elif "mac os x" in ua_lower and "iphone" not in ua_lower and "ipad" not in ua_lower:
            os_name = "macos"
            m = re.search(r"mac os x ([0-9_]+)", ua_lower)
            os_version = m.group(1).replace("_", ".") if m else None
        elif "android" in ua_lower:
            os_name = "android"
            m = re.search(r"android ([0-9.]+)", ua_lower)
            os_version = m.group(1) if m else None
        elif "iphone os" in ua_lower or "cpu iphone os" in ua_lower:
            os_name = "ios"
            m = re.search(r"(?:iphone os|cpu iphone os) ([0-9_]+)", ua_lower)
            os_version = m.group(1).replace("_", ".") if m else None
        elif "linux" in ua_lower:
            os_name = "linux"

        browser_name = "unknown"
        browser_version = None
        if "edg/" in ua_lower:
            browser_name = "edge"
            m = re.search(r"edg/([0-9.]+)", ua_lower)
            browser_version = m.group(1) if m else None
        elif "chrome/" in ua_lower and "chromium" not in ua_lower:
            browser_name = "chrome"
            m = re.search(r"chrome/([0-9.]+)", ua_lower)
            browser_version = m.group(1) if m else None
        elif "firefox/" in ua_lower:
            browser_name = "firefox"
            m = re.search(r"firefox/([0-9.]+)", ua_lower)
            browser_version = m.group(1) if m else None
        elif "safari/" in ua_lower and "chrome/" not in ua_lower:
            browser_name = "safari"
            m = re.search(r"version/([0-9.]+)", ua_lower)
            browser_version = m.group(1) if m else None

        return {
            "user_agent": ua,
            "is_bot": is_bot,
            "device": {"type": device_type},
            "os": {"name": os_name, "version": os_version},
            "browser": {"name": browser_name, "version": browser_version},
        }
    
    def _generate_session_id(self, user_id: str, client_info: Any) -> str:
        """生成会话ID"""
        # 简单的会话ID生成策略
        import uuid
        return str(uuid.uuid4())
    
    async def _enrich_experiment_context(self, event: CreateEventRequest) -> Dict[str, Any]:
        """增强实验上下文"""
        context = event.experiment_context or {}
        
        # 添加处理时间戳
        context["enriched_at"] = utc_now().isoformat()
        context["processing_version"] = "1.0"
        
        return context

class EventProcessingService:
    """事件处理主服务"""
    
    def __init__(
        self,
        event_repo: EventStreamRepository,
        dedup_repo: EventDeduplicationRepository,
        schema_repo: EventSchemaRepository,
        error_repo: EventErrorRepository
    ):
        self.event_repo = event_repo
        self.validation_service = EventValidationService(schema_repo)
        self.deduplication_service = EventDeduplicationService(dedup_repo)
        self.enrichment_service = EventEnrichmentService()
        self.error_repo = error_repo
    
    async def process_event(self, event: CreateEventRequest, client_ip: str = None) -> EventProcessingResult:
        """处理单个事件的完整流程"""
        start_time = utc_now()
        
        try:
            # 阶段1: 事件验证
            validation_result = await self.validation_service.validate_event(event)
            
            if not validation_result.is_valid:
                await self._record_processing_error(
                    event, ProcessingStage.VALIDATION,
                    "事件验证失败", {"validation_errors": validation_result.validation_errors}
                )
                
                return EventProcessingResult(
                    success=False,
                    event_id=event.event_id,
                    status=EventStatus.FAILED,
                    message="事件验证失败",
                    errors=validation_result.validation_errors,
                    stage=ProcessingStage.VALIDATION
                )
            
            # 阶段2: 去重检查
            dedup_info = await self.deduplication_service.check_duplicate(event)
            
            if dedup_info.is_duplicate:
                return EventProcessingResult(
                    success=True,
                    event_id=event.event_id,
                    status=EventStatus.DUPLICATE,
                    message=f"重复事件，原始事件: {dedup_info.original_event_id}"
                )
            
            # 阶段3: 事件增强
            enriched_event = await self.enrichment_service.enrich_event(event, client_ip)
            
            # 阶段4: 存储事件
            db_event = await self.event_repo.create_event(enriched_event, validation_result.data_quality)
            
            # 计算处理时间
            duration = utc_now() - start_time
            duration_ms = int(duration.total_seconds() * 1000)
            
            return EventProcessingResult(
                success=True,
                event_id=event.event_id,
                status=EventStatus.PENDING,
                message="事件处理完成",
                warnings=validation_result.validation_warnings,
                processing_time_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"事件处理异常: {e}", exc_info=True)
            
            await self._record_processing_error(
                event, ProcessingStage.STORAGE,
                f"处理异常: {str(e)}", {"exception": str(e)}
            )
            
            duration = utc_now() - start_time
            duration_ms = int(duration.total_seconds() * 1000)
            
            return EventProcessingResult(
                success=False,
                event_id=event.event_id,
                status=EventStatus.FAILED,
                message=f"处理异常: {str(e)}",
                errors=[str(e)],
                processing_time_ms=duration_ms
            )
    
    async def process_events_batch(
        self, 
        events: List[CreateEventRequest], 
        client_ip: str = None
    ) -> List[EventProcessingResult]:
        """批量处理事件"""
        # 并发处理事件，但限制并发数量
        semaphore = asyncio.Semaphore(20)
        
        async def process_single_event(event: CreateEventRequest) -> EventProcessingResult:
            async with semaphore:
                return await self.process_event(event, client_ip)
        
        # 并发执行
        tasks = [process_single_event(event) for event in events]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        return results
    
    async def _record_processing_error(
        self,
        event: CreateEventRequest,
        stage: ProcessingStage,
        error_message: str,
        error_details: Dict[str, Any] = None
    ):
        """记录处理错误"""
        try:
            await self.error_repo.create_error(
                failed_event_id=event.event_id,
                raw_event_data=event.model_dump(mode="json"),
                error_type="processing_error",
                error_message=error_message,
                error_details=error_details or {},
                processing_stage=stage.value
            )
        except Exception as e:
            logger.error(f"记录错误失败: {e}")  # 避免递归错误
