"""
企业级配置管理服务
集中管理所有企业级组件的阈值、参数和策略配置
"""

from typing import Dict, Any, Optional, Union, List, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import yaml
import os
from pathlib import Path
import asyncio
import redis.asyncio as redis
import structlog
from pydantic import Field, ConfigDict
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory

logger = structlog.get_logger(__name__)


class ConfigLevel(str, Enum):
    """配置级别"""
    SYSTEM = "system"      # 系统级配置
    ENTERPRISE = "enterprise"  # 企业级配置
    SERVICE = "service"    # 服务级配置
    AGENT = "agent"        # 智能体级配置


class ConfigCategory(str, Enum):
    """配置分类"""
    PERFORMANCE = "performance"  # 性能相关
    SECURITY = "security"       # 安全相关
    MONITORING = "monitoring"   # 监控相关
    SCALING = "scaling"         # 扩缩容相关
    RECOVERY = "recovery"       # 错误恢复相关
    NETWORKING = "networking"   # 网络相关
    STORAGE = "storage"         # 存储相关


@dataclass
class ConfigItem:
    """配置项数据结构"""
    key: str
    value: Union[int, float, str, bool, List, Dict]
    category: ConfigCategory
    level: ConfigLevel
    description: str = ""
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    valid_values: Optional[List] = None
    created_at: datetime = field(default_factory=utc_factory)
    updated_at: datetime = field(default_factory=utc_factory)
    version: str = "1.0.0"


class EnterpriseConfigSettings(BaseSettings):
    """企业级配置设置"""
    
    # === 智能体池配置 ===
    # Agent Pool Configuration
    AGENT_POOL_MIN_SIZE: int = Field(default=1, description="智能体池最小大小")
    AGENT_POOL_MAX_SIZE: int = Field(default=10, description="智能体池最大大小") 
    AGENT_POOL_INITIAL_SIZE: int = Field(default=3, description="智能体池初始大小")
    AGENT_POOL_IDLE_TIMEOUT: int = Field(default=300, description="智能体空闲超时时间(秒)")
    AGENT_POOL_SCALING_THRESHOLD: float = Field(default=0.8, description="智能体池扩容负载阈值")
    
    # === 负载均衡配置 ===
    # Load Balancing Configuration
    LOAD_BALANCER_THRESHOLD: float = Field(default=0.8, description="负载均衡阈值")
    LOAD_BALANCER_CAPABILITY_WEIGHT: float = Field(default=0.5, description="能力权重")
    LOAD_BALANCER_LOAD_WEIGHT: float = Field(default=0.3, description="负载权重")
    LOAD_BALANCER_AVAILABILITY_WEIGHT: float = Field(default=0.2, description="可用性权重")
    LOAD_BALANCER_MAX_RETRIES: int = Field(default=3, description="负载均衡最大重试次数")
    
    # === 分布式事件配置 ===
    # Distributed Events Configuration
    DISTRIBUTED_EVENT_ELECTION_TIMEOUT: int = Field(default=5, description="选举超时时间(秒)")
    DISTRIBUTED_EVENT_HEARTBEAT_INTERVAL: int = Field(default=2, description="心跳间隔(秒)")
    DISTRIBUTED_EVENT_LEADER_LEASE_TIME: int = Field(default=10, description="领导者租约时间(秒)")
    DISTRIBUTED_EVENT_LOAD_BALANCE_THRESHOLD: float = Field(default=1.2, description="负载均衡阈值倍数")
    DISTRIBUTED_EVENT_UNDERLOAD_THRESHOLD: float = Field(default=0.8, description="负载不足阈值倍数")
    
    # === 性能优化配置 ===
    # Performance Optimization Configuration  
    PERFORMANCE_SMOOTHING_ALPHA: float = Field(default=0.1, description="性能平滑因子")
    PERFORMANCE_LOAD_NORMALIZATION_FACTOR: float = Field(default=100.0, description="负载归一化因子")
    PERFORMANCE_MAX_CONCURRENT_TASKS: int = Field(default=50, description="最大并发任务数")
    PERFORMANCE_TASK_TIMEOUT: int = Field(default=30000, description="任务超时时间(毫秒)")
    
    # === 安全配置 ===
    # Security Configuration
    SECURITY_RATE_LIMIT_DEFAULT: int = Field(default=10, description="默认速率限制(次/分钟)")
    SECURITY_RATE_LIMIT_WINDOW: int = Field(default=60, description="速率限制窗口(秒)")
    SECURITY_HIGH_SEVERITY_WINDOW: int = Field(default=300, description="高严重性事件窗口(秒)")
    SECURITY_CRITICAL_SEVERITY_WINDOW: int = Field(default=180, description="关键严重性事件窗口(秒)")
    SECURITY_QUARANTINE_TIME: int = Field(default=3600, description="隔离时间(秒)")
    SECURITY_MAX_VIOLATIONS_PER_HOUR: int = Field(default=100, description="每小时最大违规次数")
    
    # === 错误恢复配置 ===
    # Error Recovery Configuration
    ERROR_RECOVERY_TIMEOUT: int = Field(default=60000, description="错误恢复超时时间(毫秒)")
    ERROR_RECOVERY_MAX_RETRIES: int = Field(default=3, description="错误恢复最大重试次数")
    ERROR_RECOVERY_BACKOFF_FACTOR: float = Field(default=2.0, description="错误恢复退避因子")
    ERROR_RECOVERY_CIRCUIT_BREAKER_THRESHOLD: int = Field(default=5, description="断路器阈值")
    ERROR_RECOVERY_CIRCUIT_BREAKER_TIMEOUT: int = Field(default=30, description="断路器超时(秒)")
    
    # === 监控配置 ===
    # Monitoring Configuration  
    MONITORING_MAX_LOGS: int = Field(default=10000, description="最大日志条数")
    MONITORING_LOG_RETENTION_DAYS: int = Field(default=30, description="日志保留天数")
    MONITORING_METRIC_COLLECTION_INTERVAL: int = Field(default=30, description="指标收集间隔(秒)")
    MONITORING_ALERT_CPU_THRESHOLD: float = Field(default=0.9, description="CPU告警阈值")
    MONITORING_ALERT_MEMORY_THRESHOLD: float = Field(default=0.85, description="内存告警阈值")
    MONITORING_ALERT_DISK_THRESHOLD: float = Field(default=0.9, description="磁盘告警阈值")
    
    # === 缓存配置 ===
    # Cache Configuration
    CACHE_DEFAULT_TTL: int = Field(default=3600, description="默认缓存TTL(秒)")
    CACHE_MAX_SIZE: int = Field(default=10000, description="缓存最大大小")
    CACHE_CLEANUP_INTERVAL: int = Field(default=300, description="缓存清理间隔(秒)")
    CACHE_COMPRESSION_THRESHOLD: int = Field(default=1024, description="压缩阈值(字节)")
    
    # === 资源管理配置 ===
    # Resource Management Configuration
    RESOURCE_CPU_LIMIT: float = Field(default=0.8, description="CPU使用限制")
    RESOURCE_MEMORY_LIMIT: float = Field(default=0.85, description="内存使用限制")
    RESOURCE_CONNECTION_POOL_SIZE: int = Field(default=20, description="连接池大小")
    RESOURCE_THREAD_POOL_SIZE: int = Field(default=10, description="线程池大小")
    
    # === 流控配置 ===
    # Flow Control Configuration
    FLOW_CONTROL_ENABLE_BACKPRESSURE: bool = Field(default=True, description="启用背压机制")
    FLOW_CONTROL_QUEUE_SIZE_THRESHOLD: int = Field(default=1000, description="队列大小阈值")
    FLOW_CONTROL_DROP_POLICY: str = Field(default="oldest", description="丢弃策略(oldest/newest/random)")
    FLOW_CONTROL_THROTTLE_THRESHOLD: float = Field(default=0.8, description="限流阈值")
    
    # === 高可用配置 ===
    # High Availability Configuration
    HA_ENABLE_CLUSTERING: bool = Field(default=True, description="启用集群模式")
    HA_CLUSTER_MIN_NODES: int = Field(default=3, description="集群最少节点数")
    HA_FAILOVER_TIMEOUT: int = Field(default=30, description="故障转移超时(秒)")
    HA_HEALTH_CHECK_INTERVAL: int = Field(default=10, description="健康检查间隔(秒)")
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8", 
        case_sensitive=True,
        env_prefix="ENTERPRISE_"
    )


class EnterpriseConfigManager:
    """企业级配置管理器"""
    
    def __init__(self, 
                 redis_client: Optional[redis.Redis] = None,
                 config_file: Optional[str] = None):
        self.redis_client = redis_client
        self.config_file = config_file or "enterprise_config.yaml"
        self.settings = EnterpriseConfigSettings()
        self._config_cache: Dict[str, ConfigItem] = {}
        self._watch_tasks: List[asyncio.Task] = []
        self._subscribers: Dict[str, List[Callable]] = {}
        self._load_local_config()
    
    def _load_local_config(self):
        """加载本地配置文件"""
        try:
            config_path = Path(self.config_file)
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
                        local_config = yaml.safe_load(f)
                    else:
                        local_config = json.load(f)
                
                # 将本地配置转换为ConfigItem
                for key, value in local_config.items():
                    if isinstance(value, dict) and 'value' in value:
                        config_item = ConfigItem(
                            key=key,
                            value=value['value'],
                            category=ConfigCategory(value.get('category', 'performance')),
                            level=ConfigLevel(value.get('level', 'system')),
                            description=value.get('description', ''),
                            min_value=value.get('min_value'),
                            max_value=value.get('max_value'),
                            valid_values=value.get('valid_values')
                        )
                    else:
                        config_item = ConfigItem(
                            key=key,
                            value=value,
                            category=ConfigCategory.PERFORMANCE,
                            level=ConfigLevel.SYSTEM,
                            description=f"Local config: {key}"
                        )
                    self._config_cache[key] = config_item
                    
                logger.info(f"Loaded {len(local_config)} configuration items from {self.config_file}")
        except Exception as e:
            logger.warning(f"Failed to load local config file {self.config_file}: {e}")
    
    async def start(self):
        """启动配置管理器"""
        if self.redis_client:
            # 启动Redis配置同步
            sync_task = asyncio.create_task(self._sync_from_redis())
            watch_task = asyncio.create_task(self._watch_redis_changes())
            self._watch_tasks.extend([sync_task, watch_task])
        
        logger.info("Enterprise config manager started")
    
    async def stop(self):
        """停止配置管理器"""
        for task in self._watch_tasks:
            task.cancel()
        
        await asyncio.gather(*self._watch_tasks, return_exceptions=True)
        logger.info("Enterprise config manager stopped")
    
    def get(self, key: str, default: Any = None, category: Optional[ConfigCategory] = None) -> Any:
        """获取配置值"""
        # 先从缓存查找
        if key in self._config_cache:
            config_item = self._config_cache[key]
            if category is None or config_item.category == category:
                return config_item.value
        
        # 从settings查找
        if hasattr(self.settings, key):
            return getattr(self.settings, key)
            
        return default
    
    def get_int(self, key: str, default: int = 0) -> int:
        """获取整数配置值"""
        value = self.get(key, default)
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """获取浮点配置值"""
        value = self.get(key, default)
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """获取布尔配置值"""
        value = self.get(key, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', '1', 'yes', 'on')
        return bool(value) if value else default
    
    def get_list(self, key: str, default: List = None) -> List:
        """获取列表配置值"""
        value = self.get(key, default or [])
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            # 尝试解析逗号分隔的字符串
            return [item.strip() for item in value.split(',') if item.strip()]
        return default or []
    
    async def set(self, key: str, value: Any, 
                  category: ConfigCategory = ConfigCategory.PERFORMANCE,
                  level: ConfigLevel = ConfigLevel.SYSTEM,
                  description: str = "",
                  persist: bool = True):
        """设置配置值"""
        config_item = ConfigItem(
            key=key,
            value=value,
            category=category,
            level=level,
            description=description,
            updated_at=utc_now()
        )
        
        # 验证配置值
        if not self._validate_config(config_item):
            raise ValueError(f"Invalid configuration value for {key}: {value}")
        
        # 更新本地缓存
        self._config_cache[key] = config_item
        
        # 持久化到Redis
        if persist and self.redis_client:
            await self._persist_to_redis(config_item)
        
        # 通知订阅者
        await self._notify_subscribers(key, value)
        
        logger.info(f"Configuration updated: {key} = {value}")
    
    def _validate_config(self, config_item: ConfigItem) -> bool:
        """验证配置项"""
        value = config_item.value
        
        # 检查数值范围
        if config_item.min_value is not None and isinstance(value, (int, float)):
            if value < config_item.min_value:
                return False
        
        if config_item.max_value is not None and isinstance(value, (int, float)):
            if value > config_item.max_value:
                return False
        
        # 检查有效值列表
        if config_item.valid_values and value not in config_item.valid_values:
            return False
        
        return True
    
    async def _persist_to_redis(self, config_item: ConfigItem):
        """持久化配置到Redis"""
        try:
            config_data = {
                'value': config_item.value,
                'category': config_item.category.value,
                'level': config_item.level.value,
                'description': config_item.description,
                'updated_at': config_item.updated_at.isoformat(),
                'version': config_item.version
            }
            
            await self.redis_client.hset(
                "enterprise_config",
                config_item.key,
                json.dumps(config_data)
            )
            
            # 发布配置变更事件
            await self.redis_client.publish(
                "config_changes",
                json.dumps({
                    'key': config_item.key,
                    'value': config_item.value,
                    'timestamp': utc_now().isoformat()
                })
            )
        except Exception as e:
            logger.error(f"Failed to persist config to Redis: {e}")
    
    async def _sync_from_redis(self):
        """从Redis同步配置"""
        try:
            if not self.redis_client:
                return
            
            config_data = await self.redis_client.hgetall("enterprise_config")
            
            for key, value_json in config_data.items():
                try:
                    config_dict = json.loads(value_json)
                    config_item = ConfigItem(
                        key=key.decode() if isinstance(key, bytes) else key,
                        value=config_dict['value'],
                        category=ConfigCategory(config_dict.get('category', 'performance')),
                        level=ConfigLevel(config_dict.get('level', 'system')),
                        description=config_dict.get('description', ''),
                        updated_at=datetime.fromisoformat(config_dict.get('updated_at', utc_now().isoformat())),
                        version=config_dict.get('version', '1.0.0')
                    )
                    self._config_cache[config_item.key] = config_item
                except Exception as e:
                    logger.warning(f"Failed to parse config item {key}: {e}")
            
            logger.info(f"Synced {len(config_data)} configuration items from Redis")
            
        except Exception as e:
            logger.error(f"Failed to sync config from Redis: {e}")
    
    async def _watch_redis_changes(self):
        """监控Redis配置变更"""
        try:
            if not self.redis_client:
                return
            
            pubsub = self.redis_client.pubsub()
            await pubsub.subscribe("config_changes")
            
            async for message in pubsub.listen():
                if message['type'] == 'message':
                    try:
                        change_data = json.loads(message['data'])
                        key = change_data['key']
                        value = change_data['value']
                        
                        # 更新本地缓存
                        if key in self._config_cache:
                            self._config_cache[key].value = value
                            self._config_cache[key].updated_at = utc_now()
                        
                        # 通知订阅者
                        await self._notify_subscribers(key, value)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process config change: {e}")
                        
        except Exception as e:
            logger.error(f"Config watch task failed: {e}")
    
    def subscribe(self, key: str, callback: Callable[[str, Any], None]):
        """订阅配置变更"""
        if key not in self._subscribers:
            self._subscribers[key] = []
        self._subscribers[key].append(callback)
    
    def unsubscribe(self, key: str, callback: Callable[[str, Any], None]):
        """取消配置变更订阅"""
        if key in self._subscribers and callback in self._subscribers[key]:
            self._subscribers[key].remove(callback)
    
    async def _notify_subscribers(self, key: str, value: Any):
        """通知配置变更订阅者"""
        if key in self._subscribers:
            for callback in self._subscribers[key]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(key, value)
                    else:
                        callback(key, value)
                except Exception as e:
                    logger.warning(f"Config subscriber callback failed: {e}")
    
    def export_config(self, file_path: str, format: str = "yaml"):
        """导出配置到文件"""
        config_dict = {}
        
        for key, config_item in self._config_cache.items():
            config_dict[key] = {
                'value': config_item.value,
                'category': config_item.category.value,
                'level': config_item.level.value,
                'description': config_item.description,
                'updated_at': config_item.updated_at.isoformat(),
                'version': config_item.version
            }
        
        # 添加settings中的配置
        for field_name, field_info in self.settings.__fields__.items():
            if field_name not in config_dict:
                config_dict[field_name] = {
                    'value': getattr(self.settings, field_name),
                    'category': 'performance',
                    'level': 'system',
                    'description': field_info.description or f"Setting: {field_name}",
                    'updated_at': utc_now().isoformat(),
                    'version': '1.0.0'
                }
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if format.lower() in ('yaml', 'yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Configuration exported to {file_path}")
        except Exception as e:
            logger.error(f"Failed to export configuration: {e}")
    
    def get_all_configs(self, category: Optional[ConfigCategory] = None,
                       level: Optional[ConfigLevel] = None) -> Dict[str, ConfigItem]:
        """获取所有配置项"""
        result = {}
        
        for key, config_item in self._config_cache.items():
            if category and config_item.category != category:
                continue
            if level and config_item.level != level:
                continue
            result[key] = config_item
        
        return result
    
    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能相关配置"""
        return {
            'agent_pool': {
                'min_size': self.get_int('AGENT_POOL_MIN_SIZE'),
                'max_size': self.get_int('AGENT_POOL_MAX_SIZE'),
                'initial_size': self.get_int('AGENT_POOL_INITIAL_SIZE'),
                'idle_timeout': self.get_int('AGENT_POOL_IDLE_TIMEOUT'),
                'scaling_threshold': self.get_float('AGENT_POOL_SCALING_THRESHOLD'),
            },
            'load_balancer': {
                'threshold': self.get_float('LOAD_BALANCER_THRESHOLD'),
                'capability_weight': self.get_float('LOAD_BALANCER_CAPABILITY_WEIGHT'),
                'load_weight': self.get_float('LOAD_BALANCER_LOAD_WEIGHT'),
                'availability_weight': self.get_float('LOAD_BALANCER_AVAILABILITY_WEIGHT'),
                'max_retries': self.get_int('LOAD_BALANCER_MAX_RETRIES'),
            },
            'performance': {
                'max_concurrent_tasks': self.get_int('PERFORMANCE_MAX_CONCURRENT_TASKS'),
                'task_timeout': self.get_int('PERFORMANCE_TASK_TIMEOUT'),
                'smoothing_alpha': self.get_float('PERFORMANCE_SMOOTHING_ALPHA'),
                'load_normalization_factor': self.get_float('PERFORMANCE_LOAD_NORMALIZATION_FACTOR'),
            }
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """获取安全相关配置"""
        return {
            'rate_limiting': {
                'default_limit': self.get_int('SECURITY_RATE_LIMIT_DEFAULT'),
                'window': self.get_int('SECURITY_RATE_LIMIT_WINDOW'),
                'high_severity_window': self.get_int('SECURITY_HIGH_SEVERITY_WINDOW'),
                'critical_severity_window': self.get_int('SECURITY_CRITICAL_SEVERITY_WINDOW'),
            },
            'quarantine': {
                'time': self.get_int('SECURITY_QUARANTINE_TIME'),
                'max_violations_per_hour': self.get_int('SECURITY_MAX_VIOLATIONS_PER_HOUR'),
            }
        }
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """获取监控相关配置"""
        return {
            'logging': {
                'max_logs': self.get_int('MONITORING_MAX_LOGS'),
                'retention_days': self.get_int('MONITORING_LOG_RETENTION_DAYS'),
            },
            'metrics': {
                'collection_interval': self.get_int('MONITORING_METRIC_COLLECTION_INTERVAL'),
            },
            'alerts': {
                'cpu_threshold': self.get_float('MONITORING_ALERT_CPU_THRESHOLD'),
                'memory_threshold': self.get_float('MONITORING_ALERT_MEMORY_THRESHOLD'),
                'disk_threshold': self.get_float('MONITORING_ALERT_DISK_THRESHOLD'),
            }
        }


# 全局配置管理器实例
_config_manager: Optional[EnterpriseConfigManager] = None

def get_config_manager() -> EnterpriseConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = EnterpriseConfigManager()
    return _config_manager

async def init_config_manager(redis_client: Optional[redis.Redis] = None,
                             config_file: Optional[str] = None):
    """初始化全局配置管理器"""
    global _config_manager
    _config_manager = EnterpriseConfigManager(redis_client, config_file)
    await _config_manager.start()

async def shutdown_config_manager():
    """关闭全局配置管理器"""
    global _config_manager
    if _config_manager:
        await _config_manager.stop()