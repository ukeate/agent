"""
Neo4j图数据库抽象层
提供统一的图数据库操作接口，支持连接池管理、事务处理、监控和故障恢复
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

import neo4j
from neo4j import AsyncGraphDatabase, AsyncDriver, AsyncSession, AsyncManagedTransaction
from neo4j.exceptions import (
    ServiceUnavailable, 
    TransientError, 
    DatabaseError,
    DriverError,
    SessionExpired
)

from core.config import get_settings

logger = logging.getLogger(__name__)


class GraphDatabaseConfig:
    """图数据库配置类"""
    
    def __init__(self):
        self.settings = get_settings()
        
    @property
    def connection_config(self) -> Dict[str, Any]:
        """获取连接配置"""
        return {
            "max_connection_pool_size": self.settings.NEO4J_MAX_POOL_SIZE,
            "max_transaction_retry_time": self.settings.NEO4J_MAX_RETRY_TIME,
            "connection_timeout": self.settings.NEO4J_CONNECTION_TIMEOUT,
            "encrypted": self.settings.NEO4J_ENCRYPTED,
            "trust": neo4j.TRUST_SYSTEM_CA_SIGNED_CERTIFICATES if self.settings.NEO4J_TRUST_SYSTEM_CA else neo4j.TRUST_ALL_CERTIFICATES,
            "max_connection_lifetime": self.settings.NEO4J_POOL_MAX_LIFETIME,
            "keep_alive": True
        }


class GraphDatabaseError(Exception):
    """图数据库异常基类"""
    pass


class GraphConnectionError(GraphDatabaseError):
    """图数据库连接异常"""
    pass


class GraphQueryError(GraphDatabaseError):
    """图数据库查询异常"""
    pass


class GraphTransactionError(GraphDatabaseError):
    """图数据库事务异常"""
    pass


class Neo4jGraphDatabase:
    """Neo4j图数据库管理器"""
    
    def __init__(self):
        self.config = GraphDatabaseConfig()
        self.settings = get_settings()
        self.driver: Optional[AsyncDriver] = None
        self._connection_pool_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "failed_connections": 0,
            "last_health_check": None
        }
    
    async def initialize(self) -> None:
        """初始化数据库连接"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                self.settings.NEO4J_URI,
                auth=(self.settings.NEO4J_USERNAME, self.settings.NEO4J_PASSWORD),
                **self.config.connection_config
            )
            
            # 验证连接
            await self._verify_connectivity()
            logger.info(f"Neo4j图数据库连接成功: {self.settings.NEO4J_URI}")
            
        except Exception as e:
            logger.error(f"Neo4j图数据库初始化失败: {str(e)}")
            raise GraphConnectionError(f"无法连接到Neo4j数据库: {str(e)}")
    
    async def close(self) -> None:
        """关闭数据库连接"""
        if self.driver:
            await self.driver.close()
            self.driver = None
            logger.info("Neo4j图数据库连接已关闭")
    
    async def _verify_connectivity(self) -> None:
        """验证数据库连接"""
        if not self.driver:
            raise GraphConnectionError("数据库驱动未初始化")
        
        try:
            await self.driver.verify_connectivity()
            self._connection_pool_stats["last_health_check"] = utc_now()
            
        except ServiceUnavailable as e:
            logger.error(f"Neo4j服务不可用: {str(e)}")
            raise GraphConnectionError("Neo4j服务不可用")
        
        except Exception as e:
            logger.error(f"连接验证失败: {str(e)}")
            raise GraphConnectionError(f"连接验证失败: {str(e)}")
    
    @asynccontextmanager
    async def session(self, 
                     database: Optional[str] = None,
                     access_mode: neo4j.AccessMode = neo4j.WRITE_ACCESS,
                     bookmarks: Optional[neo4j.Bookmarks] = None) -> AsyncSession:
        """创建数据库会话上下文管理器"""
        if not self.driver:
            raise GraphConnectionError("数据库驱动未初始化")
        
        session = self.driver.session(
            database=database or self.settings.NEO4J_DATABASE,
            default_access_mode=access_mode,
            bookmarks=bookmarks
        )
        
        try:
            self._connection_pool_stats["active_connections"] += 1
            yield session
            
        except asyncio.CancelledError:
            session.cancel()
            raise
            
        except Exception as e:
            self._connection_pool_stats["failed_connections"] += 1
            logger.error(f"会话执行错误: {str(e)}")
            raise GraphQueryError(f"会话执行失败: {str(e)}")
            
        finally:
            await session.close()
            self._connection_pool_stats["active_connections"] -= 1
    
    async def execute_query(self, 
                           query: str, 
                           parameters: Optional[Dict[str, Any]] = None,
                           database: Optional[str] = None,
                           access_mode: neo4j.AccessMode = neo4j.WRITE_ACCESS) -> List[Dict[str, Any]]:
        """执行单个查询"""
        if not self.driver:
            raise GraphConnectionError("数据库驱动未初始化")
        
        try:
            records, summary, keys = await self.driver.execute_query(
                query,
                parameters or {},
                database_=database or self.settings.NEO4J_DATABASE,
                routing_=neo4j.RoutingControl.WRITE if access_mode == neo4j.WRITE_ACCESS else neo4j.RoutingControl.READ
            )
            
            # 转换记录为字典列表
            result = []
            for record in records:
                record_dict = {}
                for key in keys:
                    record_dict[key] = record[key]
                result.append(record_dict)
            
            # 记录查询统计
            if self.settings.NEO4J_MONITORING_ENABLED:
                logger.info(f"执行查询: {query[:100]}... 返回 {len(result)} 条记录, "
                          f"执行时间: {summary.result_available_after + summary.result_consumed_after}ms")
            
            return result
            
        except TransientError as e:
            logger.warning(f"查询暂时失败，将重试: {str(e)}")
            raise GraphQueryError(f"查询暂时失败: {str(e)}")
            
        except DatabaseError as e:
            logger.error(f"数据库查询错误: {str(e)}")
            raise GraphQueryError(f"数据库查询错误: {str(e)}")
        
        except Exception as e:
            logger.error(f"查询执行失败: {str(e)}")
            raise GraphQueryError(f"查询执行失败: {str(e)}")
    
    async def execute_read_query(self, 
                                query: str, 
                                parameters: Optional[Dict[str, Any]] = None,
                                database: Optional[str] = None) -> List[Dict[str, Any]]:
        """执行只读查询"""
        return await self.execute_query(
            query, 
            parameters, 
            database, 
            access_mode=neo4j.READ_ACCESS
        )
    
    async def execute_write_query(self, 
                                 query: str, 
                                 parameters: Optional[Dict[str, Any]] = None,
                                 database: Optional[str] = None) -> List[Dict[str, Any]]:
        """执行写入查询"""
        return await self.execute_query(
            query, 
            parameters, 
            database, 
            access_mode=neo4j.WRITE_ACCESS
        )
    
    async def execute_transaction(self, work_func, *args, **kwargs) -> Any:
        """执行管理事务"""
        if not self.driver:
            raise GraphConnectionError("数据库驱动未初始化")
        
        async with self.session() as session:
            try:
                result = await session.execute_write(work_func, *args, **kwargs)
                return result
                
            except TransientError as e:
                logger.warning(f"事务暂时失败，将重试: {str(e)}")
                raise GraphTransactionError(f"事务暂时失败: {str(e)}")
                
            except Exception as e:
                logger.error(f"事务执行失败: {str(e)}")
                raise GraphTransactionError(f"事务执行失败: {str(e)}")
    
    async def execute_read_transaction(self, work_func, *args, **kwargs) -> Any:
        """执行只读管理事务"""
        if not self.driver:
            raise GraphConnectionError("数据库驱动未初始化")
        
        async with self.session(access_mode=neo4j.READ_ACCESS) as session:
            try:
                result = await session.execute_read(work_func, *args, **kwargs)
                return result
                
            except Exception as e:
                logger.error(f"只读事务执行失败: {str(e)}")
                raise GraphTransactionError(f"只读事务执行失败: {str(e)}")
    
    async def batch_execute(self, 
                           queries: List[Dict[str, Any]], 
                           batch_size: Optional[int] = None) -> List[Any]:
        """批量执行查询"""
        batch_size = batch_size or self.settings.NEO4J_BATCH_SIZE
        results = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_results = []
            
            async with self.session() as session:
                async with session.begin_transaction() as tx:
                    try:
                        for query_dict in batch:
                            query = query_dict.get("query", "")
                            parameters = query_dict.get("parameters", {})
                            
                            result = await tx.run(query, parameters)
                            records = [record.data() async for record in result]
                            batch_results.append(records)
                        
                        await tx.commit()
                        
                    except Exception as e:
                        await tx.rollback()
                        logger.error(f"批量执行失败: {str(e)}")
                        raise GraphQueryError(f"批量执行失败: {str(e)}")
            
            results.extend(batch_results)
        
        return results
    
    async def check_health(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "status": "unhealthy",
            "database": "neo4j",
            "timestamp": utc_now().isoformat(),
            "connection_pool": self._connection_pool_stats.copy(),
            "error": None
        }
        
        try:
            await self._verify_connectivity()
            
            # 执行简单查询测试
            result = await self.execute_read_query("RETURN 1 as test")
            if result and result[0]["test"] == 1:
                health_status["status"] = "healthy"
            
        except Exception as e:
            health_status["error"] = str(e)
            logger.error(f"健康检查失败: {str(e)}")
        
        return health_status
    
    async def get_server_info(self) -> Dict[str, Any]:
        """获取服务器信息"""
        if not self.driver:
            raise GraphConnectionError("数据库驱动未初始化")
        
        try:
            server_info = await self.driver.get_server_info()
            return {
                "address": server_info.address,
                "protocol_version": server_info.protocol_version,
                "agent": server_info.agent
            }
            
        except Exception as e:
            logger.error(f"获取服务器信息失败: {str(e)}")
            raise GraphConnectionError(f"获取服务器信息失败: {str(e)}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        return self._connection_pool_stats.copy()


# 全局图数据库实例
_graph_db_instance: Optional[Neo4jGraphDatabase] = None

async def get_graph_database() -> Neo4jGraphDatabase:
    """获取图数据库实例（单例模式）"""
    global _graph_db_instance
    
    if _graph_db_instance is None:
        _graph_db_instance = Neo4jGraphDatabase()
        await _graph_db_instance.initialize()
    
    return _graph_db_instance

async def close_graph_database() -> None:
    """关闭图数据库连接"""
    global _graph_db_instance
    
    if _graph_db_instance:
        await _graph_db_instance.close()
        _graph_db_instance = None