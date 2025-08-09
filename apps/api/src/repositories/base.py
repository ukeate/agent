"""
基础仓储模式实现
提供通用的数据访问抽象层
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Generic, TypeVar, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import select, update, delete, and_, or_
from sqlalchemy.sql import Select
import logging

logger = logging.getLogger(__name__)

# 泛型类型变量
T = TypeVar('T', bound=DeclarativeBase)
ID = TypeVar('ID', bound=Union[str, int])


class BaseRepository(Generic[T, ID], ABC):
    """基础仓储抽象类"""
    
    def __init__(self, session: AsyncSession, model_class: type[T]):
        self.session = session
        self.model_class = model_class
    
    async def create(self, entity: T) -> T:
        """创建实体"""
        try:
            self.session.add(entity)
            await self.session.commit()
            await self.session.refresh(entity)
            logger.info(f"创建 {self.model_class.__name__}: {entity}")
            return entity
        except Exception as e:
            await self.session.rollback()
            logger.error(f"创建 {self.model_class.__name__} 失败: {str(e)}")
            raise
    
    async def get_by_id(self, entity_id: ID) -> Optional[T]:
        """根据ID获取实体"""
        try:
            result = await self.session.execute(
                select(self.model_class).where(self.model_class.id == entity_id)
            )
            entity = result.scalar_one_or_none()
            if entity:
                logger.debug(f"获取 {self.model_class.__name__} ID={entity_id}: 成功")
            else:
                logger.debug(f"获取 {self.model_class.__name__} ID={entity_id}: 未找到")
            return entity
        except Exception as e:
            logger.error(f"获取 {self.model_class.__name__} ID={entity_id} 失败: {str(e)}")
            raise
    
    async def get_all(
        self, 
        limit: Optional[int] = None, 
        offset: Optional[int] = None,
        order_by: Optional[str] = None
    ) -> List[T]:
        """获取所有实体"""
        try:
            query = select(self.model_class)
            
            if order_by:
                order_field = getattr(self.model_class, order_by, None)
                if order_field:
                    query = query.order_by(order_field)
            
            if offset:
                query = query.offset(offset)
            
            if limit:
                query = query.limit(limit)
            
            result = await self.session.execute(query)
            entities = result.scalars().all()
            logger.debug(f"获取 {self.model_class.__name__} 列表: {len(entities)} 条记录")
            return list(entities)
        except Exception as e:
            logger.error(f"获取 {self.model_class.__name__} 列表失败: {str(e)}")
            raise
    
    async def update(self, entity_id: ID, updates: Dict[str, Any]) -> Optional[T]:
        """更新实体"""
        try:
            await self.session.execute(
                update(self.model_class)
                .where(self.model_class.id == entity_id)
                .values(**updates)
            )
            await self.session.commit()
            
            # 获取更新后的实体
            updated_entity = await self.get_by_id(entity_id)
            if updated_entity:
                logger.info(f"更新 {self.model_class.__name__} ID={entity_id}: 成功")
            else:
                logger.warning(f"更新 {self.model_class.__name__} ID={entity_id}: 实体不存在")
            
            return updated_entity
        except Exception as e:
            await self.session.rollback()
            logger.error(f"更新 {self.model_class.__name__} ID={entity_id} 失败: {str(e)}")
            raise
    
    async def delete(self, entity_id: ID) -> bool:
        """删除实体"""
        try:
            result = await self.session.execute(
                delete(self.model_class).where(self.model_class.id == entity_id)
            )
            await self.session.commit()
            
            deleted_count = result.rowcount
            if deleted_count > 0:
                logger.info(f"删除 {self.model_class.__name__} ID={entity_id}: 成功")
                return True
            else:
                logger.warning(f"删除 {self.model_class.__name__} ID={entity_id}: 实体不存在")
                return False
        except Exception as e:
            await self.session.rollback()
            logger.error(f"删除 {self.model_class.__name__} ID={entity_id} 失败: {str(e)}")
            raise
    
    async def exists(self, entity_id: ID) -> bool:
        """检查实体是否存在"""
        try:
            result = await self.session.execute(
                select(self.model_class.id).where(self.model_class.id == entity_id)
            )
            exists = result.scalar_one_or_none() is not None
            logger.debug(f"检查 {self.model_class.__name__} ID={entity_id} 是否存在: {exists}")
            return exists
        except Exception as e:
            logger.error(f"检查 {self.model_class.__name__} ID={entity_id} 存在性失败: {str(e)}")
            raise
    
    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        """统计实体数量"""
        try:
            query = select(self.model_class)
            
            if filters:
                conditions = []
                for field, value in filters.items():
                    if hasattr(self.model_class, field):
                        conditions.append(getattr(self.model_class, field) == value)
                
                if conditions:
                    query = query.where(and_(*conditions))
            
            result = await self.session.execute(query)
            count = len(result.scalars().all())
            logger.debug(f"统计 {self.model_class.__name__} 数量: {count}")
            return count
        except Exception as e:
            logger.error(f"统计 {self.model_class.__name__} 数量失败: {str(e)}")
            raise
    
    async def find_by_fields(self, **filters) -> List[T]:
        """根据字段条件查找实体"""
        try:
            conditions = []
            for field, value in filters.items():
                if hasattr(self.model_class, field):
                    conditions.append(getattr(self.model_class, field) == value)
            
            if not conditions:
                return []
            
            query = select(self.model_class).where(and_(*conditions))
            result = await self.session.execute(query)
            entities = result.scalars().all()
            
            logger.debug(f"条件查找 {self.model_class.__name__}: {len(entities)} 条记录")
            return list(entities)
        except Exception as e:
            logger.error(f"条件查找 {self.model_class.__name__} 失败: {str(e)}")
            raise
    
    async def find_one_by_fields(self, **filters) -> Optional[T]:
        """根据字段条件查找单个实体"""
        entities = await self.find_by_fields(**filters)
        return entities[0] if entities else None
    
    def build_query(self) -> Select:
        """构建基础查询"""
        return select(self.model_class)
    
    async def execute_query(self, query: Select) -> List[T]:
        """执行自定义查询"""
        try:
            result = await self.session.execute(query)
            return list(result.scalars().all())
        except Exception as e:
            logger.error(f"执行自定义查询失败: {str(e)}")
            raise


class UnitOfWork:
    """工作单元模式"""
    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def commit(self):
        """提交事务"""
        await self.session.commit()
    
    async def rollback(self):
        """回滚事务"""
        await self.session.rollback()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()
        else:
            await self.commit()


# 仓储工厂
class RepositoryFactory:
    """仓储工厂"""
    
    @staticmethod
    def create_repository(
        session: AsyncSession,
        model_class: type[T],
        repository_class: Optional[type[BaseRepository]] = None
    ) -> BaseRepository[T, Any]:
        """创建仓储实例"""
        if repository_class:
            return repository_class(session, model_class)
        return BaseRepository(session, model_class)