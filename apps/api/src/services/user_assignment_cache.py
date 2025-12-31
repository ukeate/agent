"""
用户分组缓存和持久化服务 - 管理用户实验分配的缓存和数据库存储
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from contextlib import asynccontextmanager
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, and_
from src.models.database.experiment import ExperimentAssignment
from src.core.database import get_db_session

from src.core.logging import get_logger
logger = get_logger(__name__)

class CacheStrategy(Enum):
    """缓存策略"""
    WRITE_THROUGH = "write_through"  # 写入缓存同时写入数据库
    WRITE_BEHIND = "write_behind"   # 先写入缓存，异步写入数据库
    CACHE_ASIDE = "cache_aside"     # 应用程序管理缓存

class CacheStatus(Enum):
    """缓存状态"""
    HIT = "hit"
    MISS = "miss"
    EXPIRED = "expired"
    ERROR = "error"

@dataclass
class CachedAssignment:
    """缓存的分配记录"""
    user_id: str
    experiment_id: str
    variant_id: str
    assigned_at: datetime
    expires_at: Optional[datetime] = None
    assignment_context: Dict[str, Any] = None
    cache_metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 序列化时间戳
        if self.assigned_at:
            data['assigned_at'] = self.assigned_at.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedAssignment':
        """从字典创建实例"""
        if 'assigned_at' in data and isinstance(data['assigned_at'], str):
            data['assigned_at'] = datetime.fromisoformat(data['assigned_at'])
        if 'expires_at' in data and isinstance(data['expires_at'], str):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)

@dataclass
class CacheMetrics:
    """缓存指标"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_errors: int = 0
    avg_response_time_ms: float = 0.0
    cache_size: int = 0
    hit_rate: float = 0.0
    
    def calculate_hit_rate(self):
        """计算命中率"""
        total_requests = self.cache_hits + self.cache_misses
        self.hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0.0

class UserAssignmentCache:
    """用户分配缓存管理器"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0", 
                 cache_strategy: CacheStrategy = CacheStrategy.WRITE_THROUGH):
        """
        初始化缓存管理器
        
        Args:
            redis_url: Redis连接URL
            cache_strategy: 缓存策略
        """
        self.redis_url = redis_url
        self.cache_strategy = cache_strategy
        self._redis_pool: Optional[redis.ConnectionPool] = None
        self._redis: Optional[redis.Redis] = None
        
        # 缓存配置
        self.default_ttl = 3600  # 1小时默认TTL
        self.max_cache_size = 100000  # 最大缓存条目数
        self.key_prefix = "ab_test:assignment:"
        
        # 批处理配置
        self.batch_size = 100
        self.batch_timeout = 5.0  # 5秒批处理超时
        self._pending_writes: List[CachedAssignment] = []
        self._write_lock = asyncio.Lock()
        
        # 指标统计
        self.metrics = CacheMetrics()
        
    async def initialize(self):
        """初始化Redis连接"""
        try:
            self._redis_pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=20,
                retry_on_timeout=True
            )
            self._redis = redis.Redis(connection_pool=self._redis_pool)
            
            # 测试连接
            await self._redis.ping()
            logger.info("Redis connection initialized successfully")
            
            # 启动后台写入任务（用于write_behind策略）
            if self.cache_strategy == CacheStrategy.WRITE_BEHIND:
                create_task_with_logging(self._background_writer())
                
        except Exception as e:
            logger.error(f"Failed to initialize Redis connection: {str(e)}")
            raise
    
    async def close(self):
        """关闭Redis连接"""
        try:
            if self._redis:
                await self._redis.aclose()
            if self._redis_pool:
                await self._redis_pool.aclose()
            logger.info("Redis connection closed")
        except Exception as e:
            logger.error(f"Error closing Redis connection: {str(e)}")
    
    def _generate_cache_key(self, user_id: str, experiment_id: str) -> str:
        """生成缓存键"""
        return f"{self.key_prefix}{user_id}:{experiment_id}"
    
    def _generate_user_key(self, user_id: str) -> str:
        """生成用户所有分配的键模式"""
        return f"{self.key_prefix}{user_id}:*"
    
    async def get_assignment(self, user_id: str, experiment_id: str) -> Tuple[Optional[CachedAssignment], CacheStatus]:
        """
        获取用户分配
        
        Args:
            user_id: 用户ID
            experiment_id: 实验ID
            
        Returns:
            (分配记录, 缓存状态)
        """
        start_time = utc_now()
        self.metrics.total_requests += 1
        
        try:
            cache_key = self._generate_cache_key(user_id, experiment_id)
            
            # 尝试从缓存获取
            cached_data = await self._redis.get(cache_key)
            
            if cached_data:
                try:
                    assignment_data = json.loads(cached_data)
                    assignment = CachedAssignment.from_dict(assignment_data)
                    
                    # 检查是否过期
                    if assignment.expires_at and assignment.expires_at <= utc_now():
                        await self._redis.delete(cache_key)
                        self.metrics.cache_misses += 1
                        return None, CacheStatus.EXPIRED
                    
                    self.metrics.cache_hits += 1
                    return assignment, CacheStatus.HIT
                    
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Error deserializing cached assignment: {str(e)}")
                    await self._redis.delete(cache_key)
                    self.metrics.cache_errors += 1
                    return None, CacheStatus.ERROR
            
            # 缓存未命中，尝试从数据库获取
            assignment = await self._get_from_database(user_id, experiment_id)
            
            if assignment:
                # 将数据库记录缓存
                await self._cache_assignment(assignment, cache_key)
                return assignment, CacheStatus.MISS
            
            self.metrics.cache_misses += 1
            return None, CacheStatus.MISS
            
        except Exception as e:
            logger.error(f"Error getting assignment from cache: {str(e)}")
            self.metrics.cache_errors += 1
            return None, CacheStatus.ERROR
            
        finally:
            # 更新响应时间统计
            response_time = (utc_now() - start_time).total_seconds() * 1000
            self._update_avg_response_time(response_time)
    
    async def set_assignment(self, assignment: CachedAssignment, ttl: Optional[int] = None) -> bool:
        """
        设置用户分配
        
        Args:
            assignment: 分配记录
            ttl: 过期时间（秒）
            
        Returns:
            是否成功
        """
        try:
            cache_key = self._generate_cache_key(assignment.user_id, assignment.experiment_id)
            ttl = ttl or self.default_ttl
            
            # 根据缓存策略处理
            if self.cache_strategy == CacheStrategy.WRITE_THROUGH:
                # 同时写入缓存和数据库
                success = await self._write_to_database(assignment)
                if success:
                    await self._cache_assignment(assignment, cache_key, ttl)
                return success
                
            elif self.cache_strategy == CacheStrategy.WRITE_BEHIND:
                # 先写入缓存，异步写入数据库
                await self._cache_assignment(assignment, cache_key, ttl)
                async with self._write_lock:
                    self._pending_writes.append(assignment)
                return True
                
            elif self.cache_strategy == CacheStrategy.CACHE_ASIDE:
                # 仅写入缓存，由应用程序处理数据库
                await self._cache_assignment(assignment, cache_key, ttl)
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error setting assignment in cache: {str(e)}")
            return False
    
    async def _cache_assignment(self, assignment: CachedAssignment, cache_key: str, ttl: int = None):
        """将分配记录写入缓存"""
        try:
            ttl = ttl or self.default_ttl
            assignment_json = json.dumps(assignment.to_dict(), ensure_ascii=False)
            
            await self._redis.setex(cache_key, ttl, assignment_json)
            logger.debug(f"Cached assignment for key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Error caching assignment: {str(e)}")
            raise
    
    async def get_user_assignments(self, user_id: str) -> List[CachedAssignment]:
        """
        获取用户的所有分配
        
        Args:
            user_id: 用户ID
            
        Returns:
            分配列表
        """
        try:
            pattern = self._generate_user_key(user_id)
            keys = await self._redis.keys(pattern)
            
            assignments = []
            
            if keys:
                cached_data_list = await self._redis.mget(keys)
                
                for cached_data in cached_data_list:
                    if cached_data:
                        try:
                            assignment_data = json.loads(cached_data)
                            assignment = CachedAssignment.from_dict(assignment_data)
                            
                            # 检查是否过期
                            if not assignment.expires_at or assignment.expires_at > utc_now():
                                assignments.append(assignment)
                                
                        except (json.JSONDecodeError, TypeError) as e:
                            logger.error(f"Error deserializing cached assignment: {str(e)}")
                            continue
            
            # 如果缓存中没有找到，尝试从数据库获取
            if not assignments:
                assignments = await self._get_user_assignments_from_database(user_id)
                
                # 缓存数据库结果
                for assignment in assignments:
                    cache_key = self._generate_cache_key(assignment.user_id, assignment.experiment_id)
                    await self._cache_assignment(assignment, cache_key)
            
            return assignments
            
        except Exception as e:
            logger.error(f"Error getting user assignments: {str(e)}")
            return []
    
    async def delete_assignment(self, user_id: str, experiment_id: str) -> bool:
        """
        删除用户分配
        
        Args:
            user_id: 用户ID
            experiment_id: 实验ID
            
        Returns:
            是否成功
        """
        try:
            cache_key = self._generate_cache_key(user_id, experiment_id)
            
            # 从缓存删除
            await self._redis.delete(cache_key)
            
            # 从数据库删除
            success = await self._delete_from_database(user_id, experiment_id)
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting assignment: {str(e)}")
            return False
    
    async def clear_user_assignments(self, user_id: str) -> int:
        """
        清除用户的所有分配
        
        Args:
            user_id: 用户ID
            
        Returns:
            清除的数量
        """
        try:
            pattern = self._generate_user_key(user_id)
            keys = await self._redis.keys(pattern)
            
            deleted_count = 0
            if keys:
                deleted_count = await self._redis.delete(*keys)
            
            # 从数据库删除
            db_deleted = await self._clear_user_assignments_from_database(user_id)
            
            logger.info(f"Cleared {deleted_count} cached assignments and {db_deleted} database records for user {user_id}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error clearing user assignments: {str(e)}")
            return 0
    
    async def batch_get_assignments(self, user_experiment_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Optional[CachedAssignment]]:
        """
        批量获取分配
        
        Args:
            user_experiment_pairs: (用户ID, 实验ID) 元组列表
            
        Returns:
            分配字典
        """
        try:
            cache_keys = [
                self._generate_cache_key(user_id, experiment_id)
                for user_id, experiment_id in user_experiment_pairs
            ]
            
            # 批量从缓存获取
            cached_data_list = await self._redis.mget(cache_keys)
            
            result = {}
            missing_pairs = []
            
            for i, cached_data in enumerate(cached_data_list):
                pair = user_experiment_pairs[i]
                
                if cached_data:
                    try:
                        assignment_data = json.loads(cached_data)
                        assignment = CachedAssignment.from_dict(assignment_data)
                        
                        # 检查是否过期
                        if not assignment.expires_at or assignment.expires_at > utc_now():
                            result[pair] = assignment
                            self.metrics.cache_hits += 1
                        else:
                            result[pair] = None
                            missing_pairs.append(pair)
                            self.metrics.cache_misses += 1
                            
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.error(f"Error deserializing cached assignment: {str(e)}")
                        result[pair] = None
                        missing_pairs.append(pair)
                        self.metrics.cache_errors += 1
                else:
                    result[pair] = None
                    missing_pairs.append(pair)
                    self.metrics.cache_misses += 1
            
            # 从数据库获取缺失的记录
            if missing_pairs:
                db_assignments = await self._batch_get_from_database(missing_pairs)
                
                for pair, assignment in db_assignments.items():
                    result[pair] = assignment
                    if assignment:
                        # 缓存数据库结果
                        cache_key = self._generate_cache_key(pair[0], pair[1])
                        await self._cache_assignment(assignment, cache_key)
            
            self.metrics.total_requests += len(user_experiment_pairs)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in batch get assignments: {str(e)}")
            self.metrics.cache_errors += len(user_experiment_pairs)
            return {pair: None for pair in user_experiment_pairs}
    
    async def _get_from_database(self, user_id: str, experiment_id: str) -> Optional[CachedAssignment]:
        """从数据库获取分配"""
        try:
            async with get_db_session() as db:
                query = select(ExperimentAssignment).where(
                    and_(
                        ExperimentAssignment.user_id == user_id,
                        ExperimentAssignment.experiment_id == experiment_id
                    )
                )
                result = await db.execute(query)
                assignment = result.scalar_one_or_none()
                
                if assignment:
                    return CachedAssignment(
                        user_id=assignment.user_id,
                        experiment_id=assignment.experiment_id,
                        variant_id=assignment.variant_id,
                        assigned_at=assignment.timestamp,
                        assignment_context=assignment.context or {}
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting assignment from database: {str(e)}")
            return None
    
    async def _get_user_assignments_from_database(self, user_id: str) -> List[CachedAssignment]:
        """从数据库获取用户的所有分配"""
        try:
            async with get_db_session() as db:
                query = select(ExperimentAssignment).where(
                    ExperimentAssignment.user_id == user_id
                )
                result = await db.execute(query)
                assignments = result.scalars().all()
                
                return [
                    CachedAssignment(
                        user_id=assignment.user_id,
                        experiment_id=assignment.experiment_id,
                        variant_id=assignment.variant_id,
                        assigned_at=assignment.timestamp,
                        assignment_context=assignment.context or {}
                    )
                    for assignment in assignments
                ]
                
        except Exception as e:
            logger.error(f"Error getting user assignments from database: {str(e)}")
            return []
    
    async def _batch_get_from_database(self, user_experiment_pairs: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Optional[CachedAssignment]]:
        """从数据库批量获取分配"""
        try:
            result = {}
            
            async with get_db_session() as db:
                for user_id, experiment_id in user_experiment_pairs:
                    query = select(ExperimentAssignment).where(
                        and_(
                            ExperimentAssignment.user_id == user_id,
                            ExperimentAssignment.experiment_id == experiment_id
                        )
                    )
                    db_result = await db.execute(query)
                    assignment = db_result.scalar_one_or_none()
                    
                    if assignment:
                        result[(user_id, experiment_id)] = CachedAssignment(
                            user_id=assignment.user_id,
                            experiment_id=assignment.experiment_id,
                            variant_id=assignment.variant_id,
                            assigned_at=assignment.timestamp,
                            assignment_context=assignment.context or {}
                        )
                    else:
                        result[(user_id, experiment_id)] = None
                
                return result
                
        except Exception as e:
            logger.error(f"Error batch getting from database: {str(e)}")
            return {pair: None for pair in user_experiment_pairs}
    
    async def _write_to_database(self, assignment: CachedAssignment) -> bool:
        """写入数据库"""
        try:
            async with get_db_session() as db:
                db_assignment = ExperimentAssignment(
                    user_id=assignment.user_id,
                    experiment_id=assignment.experiment_id,
                    variant_id=assignment.variant_id,
                    timestamp=assignment.assigned_at,
                    context=assignment.assignment_context or {}
                )
                
                # 使用upsert逻辑
                existing_query = select(ExperimentAssignment).where(
                    and_(
                        ExperimentAssignment.user_id == assignment.user_id,
                        ExperimentAssignment.experiment_id == assignment.experiment_id
                    )
                )
                result = await db.execute(existing_query)
                existing = result.scalar_one_or_none()
                
                if existing:
                    # 更新现有记录
                    update_query = update(ExperimentAssignment).where(
                        and_(
                            ExperimentAssignment.user_id == assignment.user_id,
                            ExperimentAssignment.experiment_id == assignment.experiment_id
                        )
                    ).values(
                        variant_id=assignment.variant_id,
                        timestamp=assignment.assigned_at,
                        context=assignment.assignment_context or {}
                    )
                    await db.execute(update_query)
                else:
                    # 插入新记录
                    db.add(db_assignment)
                
                await db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error writing to database: {str(e)}")
            return False
    
    async def _delete_from_database(self, user_id: str, experiment_id: str) -> bool:
        """从数据库删除"""
        try:
            async with get_db_session() as db:
                delete_query = delete(ExperimentAssignment).where(
                    and_(
                        ExperimentAssignment.user_id == user_id,
                        ExperimentAssignment.experiment_id == experiment_id
                    )
                )
                await db.execute(delete_query)
                await db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error deleting from database: {str(e)}")
            return False
    
    async def _clear_user_assignments_from_database(self, user_id: str) -> int:
        """从数据库清除用户的所有分配"""
        try:
            async with get_db_session() as db:
                delete_query = delete(ExperimentAssignment).where(
                    ExperimentAssignment.user_id == user_id
                )
                result = await db.execute(delete_query)
                await db.commit()
                return result.rowcount
                
        except Exception as e:
            logger.error(f"Error clearing user assignments from database: {str(e)}")
            return 0
    
    async def _background_writer(self):
        """后台写入任务（用于write_behind策略）"""
        logger.info("Started background writer task")
        
        while True:
            try:
                await asyncio.sleep(self.batch_timeout)
                
                async with self._write_lock:
                    if not self._pending_writes:
                        continue
                    
                    # 获取待写入的记录
                    batch = self._pending_writes[:self.batch_size]
                    self._pending_writes = self._pending_writes[self.batch_size:]
                
                # 批量写入数据库
                if batch:
                    await self._batch_write_to_database(batch)
                    logger.debug(f"Background writer processed {len(batch)} assignments")
                    
            except Exception as e:
                logger.error(f"Error in background writer: {str(e)}")
                await asyncio.sleep(1)  # 出错时短暂休息
    
    async def _batch_write_to_database(self, assignments: List[CachedAssignment]):
        """批量写入数据库"""
        try:
            async with get_db_session() as db:
                for assignment in assignments:
                    db_assignment = ExperimentAssignment(
                        user_id=assignment.user_id,
                        experiment_id=assignment.experiment_id,
                        variant_id=assignment.variant_id,
                        timestamp=assignment.assigned_at,
                        context=assignment.assignment_context or {}
                    )
                    
                    # 简单插入，忽略冲突（在生产环境中可能需要更复杂的upsert逻辑）
                    try:
                        db.add(db_assignment)
                        await db.commit()
                    except Exception as e:
                        await db.rollback()
                        logger.warning(f"Failed to write assignment to database: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error in batch write to database: {str(e)}")
    
    def _update_avg_response_time(self, response_time_ms: float):
        """更新平均响应时间"""
        if self.metrics.avg_response_time_ms == 0:
            self.metrics.avg_response_time_ms = response_time_ms
        else:
            # 使用指数移动平均
            alpha = 0.1
            self.metrics.avg_response_time_ms = (
                alpha * response_time_ms + 
                (1 - alpha) * self.metrics.avg_response_time_ms
            )
    
    async def get_cache_metrics(self) -> Dict[str, Any]:
        """获取缓存指标"""
        try:
            # 更新缓存大小
            info = await self._redis.info()
            self.metrics.cache_size = info.get('db0', {}).get('keys', 0) if 'db0' in info else 0
            
            # 计算命中率
            self.metrics.calculate_hit_rate()
            
            return {
                "total_requests": self.metrics.total_requests,
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "cache_errors": self.metrics.cache_errors,
                "hit_rate_percentage": round(self.metrics.hit_rate, 2),
                "avg_response_time_ms": round(self.metrics.avg_response_time_ms, 2),
                "cache_size": self.metrics.cache_size,
                "pending_writes": len(self._pending_writes),
                "cache_strategy": self.cache_strategy.value,
                "redis_info": {
                    "connected_clients": info.get('connected_clients', 0),
                    "used_memory_human": info.get('used_memory_human', '0B'),
                    "keyspace_hits": info.get('keyspace_hits', 0),
                    "keyspace_misses": info.get('keyspace_misses', 0)
                } if info else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting cache metrics: {str(e)}")
            return {"error": str(e)}
    
    async def clear_cache(self) -> bool:
        """清空所有缓存"""
        try:
            pattern = f"{self.key_prefix}*"
            keys = await self._redis.keys(pattern)
            
            if keys:
                deleted = await self._redis.delete(*keys)
                logger.info(f"Cleared {deleted} cache entries")
                return True
            
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            start_time = utc_now()
            
            # 测试Redis连接
            await self._redis.ping()
            
            # 测试读写
            test_key = f"{self.key_prefix}health_check"
            await self._redis.setex(test_key, 10, "test")
            test_value = await self._redis.get(test_key)
            await self._redis.delete(test_key)
            
            response_time = (utc_now() - start_time).total_seconds() * 1000
            
            return {
                "status": "healthy",
                "redis_connected": True,
                "read_write_test": test_value == b"test",
                "response_time_ms": round(response_time, 2),
                "cache_strategy": self.cache_strategy.value,
                "timestamp": utc_now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": utc_now().isoformat()
            }
