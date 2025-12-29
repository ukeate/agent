"""
缓存工厂和集成模块
提供统一的缓存创建和管理接口
"""

from typing import Optional
from functools import lru_cache
from src.core.config import get_settings
from .caching import CacheConfig, NodeCache, create_node_cache

from src.core.logging import get_logger
logger = get_logger(__name__)

class CacheFactory:
    """缓存工厂类"""
    
    _instance: Optional['CacheFactory'] = None
    _cache_instance: Optional[NodeCache] = None
    
    def __new__(cls) -> 'CacheFactory':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_cache_config(self) -> CacheConfig:
        """从应用配置创建缓存配置"""
        settings = get_settings()
        
        return CacheConfig(
            enabled=settings.CACHE_ENABLED,
            backend=settings.CACHE_BACKEND,
            ttl_default=settings.CACHE_TTL_DEFAULT,
            max_entries=settings.CACHE_MAX_ENTRIES,
            key_prefix=settings.CACHE_KEY_PREFIX,
            redis_url=settings.CACHE_REDIS_URL,
            compression=settings.CACHE_COMPRESSION,
            monitoring=settings.CACHE_MONITORING,
            cleanup_interval=settings.CACHE_CLEANUP_INTERVAL
        )
    
    def get_cache(self) -> NodeCache:
        """获取缓存实例（单例）"""
        if self._cache_instance is None:
            config = self.get_cache_config()
            self._cache_instance = create_node_cache(config)
            logger.info(f"创建缓存实例: backend={config.backend}, enabled={config.enabled}")
        
        return self._cache_instance
    
    async def close_cache(self):
        """关闭缓存实例"""
        if self._cache_instance:
            if hasattr(self._cache_instance, 'close'):
                await self._cache_instance.close()
            self._cache_instance = None
            logger.info("缓存实例已关闭")
    
    def reset_cache(self):
        """重置缓存实例（用于测试）"""
        self._cache_instance = None

# 全局缓存工厂实例
cache_factory = CacheFactory()

@lru_cache(maxsize=1)
def get_cache_factory() -> CacheFactory:
    """获取缓存工厂实例"""
    return cache_factory

def get_node_cache() -> NodeCache:
    """获取Node缓存实例（快捷函数）"""
    return cache_factory.get_cache()

async def initialize_cache():
    """初始化缓存系统"""
    try:
        cache = get_node_cache()
        logger.info(f"缓存系统初始化成功: {type(cache).__name__}")
        return True
    except Exception as e:
        logger.error(f"缓存系统初始化失败: {e}")
        return False

async def shutdown_cache():
    """关闭缓存系统"""
    try:
        await cache_factory.close_cache()
        logger.info("缓存系统已关闭")
    except Exception as e:
        logger.error(f"缓存系统关闭失败: {e}")
