"""
LangGraph检查点系统
支持PostgreSQL持久化的检查点存储和恢复
"""
from typing import Any, Dict, List, Optional, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import json
import uuid
from sqlalchemy import Column, String, DateTime, Text, Integer, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from .state import MessagesState, serialize_state, deserialize_state
from ...core.database import get_db_session

Base = declarative_base()


class CheckpointModel(Base):
    """检查点数据库模型"""
    __tablename__ = "workflow_checkpoints"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id = Column(String, nullable=False, index=True)
    checkpoint_id = Column(String, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    state_data = Column(JSON, nullable=False)
    checkpoint_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
    is_deleted = Column(Boolean, default=False)


@dataclass
class Checkpoint:
    """检查点数据结构"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = ""
    state: MessagesState = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    version: int = 1


class CheckpointStorage(Protocol):
    """检查点存储接口"""
    
    async def save_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """保存检查点"""
        ...
    
    async def load_checkpoint(self, workflow_id: str, checkpoint_id: str) -> Optional[Checkpoint]:
        """加载检查点"""
        ...
    
    async def list_checkpoints(self, workflow_id: str) -> List[Checkpoint]:
        """列出工作流的所有检查点"""
        ...
    
    async def delete_checkpoint(self, workflow_id: str, checkpoint_id: str) -> bool:
        """删除检查点"""
        ...
    
    async def cleanup_old_checkpoints(self, workflow_id: str, keep_count: int = 10) -> int:
        """清理旧检查点，保留最新的指定数量"""
        ...


class PostgreSQLCheckpointStorage:
    """PostgreSQL检查点存储实现"""
    
    def __init__(self):
        self.session_factory = sessionmaker()
    
    async def save_checkpoint(self, checkpoint: Checkpoint) -> bool:
        """保存检查点到PostgreSQL"""
        try:
            async with get_db_session() as session:
                # 检查是否已存在相同检查点
                existing = session.query(CheckpointModel).filter(
                    CheckpointModel.workflow_id == checkpoint.workflow_id,
                    CheckpointModel.checkpoint_id == checkpoint.id,
                    CheckpointModel.is_deleted == False
                ).first()
                
                if existing:
                    # 更新现有检查点
                    existing.state_data = checkpoint.state
                    existing.checkpoint_metadata = checkpoint.metadata
                    existing.version += 1
                else:
                    # 创建新检查点
                    model = CheckpointModel(
                        id=str(uuid.uuid4()),
                        workflow_id=checkpoint.workflow_id,
                        checkpoint_id=checkpoint.id,
                        state_data=checkpoint.state,
                        checkpoint_metadata=checkpoint.metadata,
                        version=checkpoint.version
                    )
                    session.add(model)
                
                await session.commit()
                return True
                
        except SQLAlchemyError as e:
            print(f"保存检查点失败: {e}")
            return False
    
    async def load_checkpoint(self, workflow_id: str, checkpoint_id: str) -> Optional[Checkpoint]:
        """从PostgreSQL加载检查点"""
        try:
            async with get_db_session() as session:
                model = session.query(CheckpointModel).filter(
                    CheckpointModel.workflow_id == workflow_id,
                    CheckpointModel.checkpoint_id == checkpoint_id,
                    CheckpointModel.is_deleted == False
                ).first()
                
                if not model:
                    return None
                
                return Checkpoint(
                    id=model.checkpoint_id,
                    workflow_id=model.workflow_id,
                    state=model.state_data,
                    metadata=model.checkpoint_metadata or {},
                    created_at=model.created_at,
                    version=model.version
                )
                
        except SQLAlchemyError as e:
            print(f"加载检查点失败: {e}")
            return None
    
    async def list_checkpoints(self, workflow_id: str) -> List[Checkpoint]:
        """列出工作流的所有检查点"""
        try:
            async with get_db_session() as session:
                models = session.query(CheckpointModel).filter(
                    CheckpointModel.workflow_id == workflow_id,
                    CheckpointModel.is_deleted == False
                ).order_by(CheckpointModel.created_at.desc()).all()
                
                return [
                    Checkpoint(
                        id=model.checkpoint_id,
                        workflow_id=model.workflow_id,
                        state=model.state_data,
                        metadata=model.checkpoint_metadata or {},
                        created_at=model.created_at,
                        version=model.version
                    )
                    for model in models
                ]
                
        except SQLAlchemyError as e:
            print(f"列出检查点失败: {e}")
            return []
    
    async def delete_checkpoint(self, workflow_id: str, checkpoint_id: str) -> bool:
        """软删除检查点"""
        try:
            async with get_db_session() as session:
                model = session.query(CheckpointModel).filter(
                    CheckpointModel.workflow_id == workflow_id,
                    CheckpointModel.checkpoint_id == checkpoint_id,
                    CheckpointModel.is_deleted == False
                ).first()
                
                if model:
                    model.is_deleted = True
                    await session.commit()
                    return True
                return False
                
        except SQLAlchemyError as e:
            print(f"删除检查点失败: {e}")
            return False
    
    async def cleanup_old_checkpoints(self, workflow_id: str, keep_count: int = 10) -> int:
        """清理旧检查点"""
        try:
            async with get_db_session() as session:
                # 获取所有检查点，按创建时间降序
                all_checkpoints = session.query(CheckpointModel).filter(
                    CheckpointModel.workflow_id == workflow_id,
                    CheckpointModel.is_deleted == False
                ).order_by(CheckpointModel.created_at.desc()).all()
                
                if len(all_checkpoints) <= keep_count:
                    return 0
                
                # 标记需要删除的检查点
                to_delete = all_checkpoints[keep_count:]
                deleted_count = 0
                
                for checkpoint in to_delete:
                    checkpoint.is_deleted = True
                    deleted_count += 1
                
                await session.commit()
                return deleted_count
                
        except SQLAlchemyError as e:
            print(f"清理检查点失败: {e}")
            return 0


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, storage: CheckpointStorage = None):
        self.storage = storage or PostgreSQLCheckpointStorage()
    
    async def create_checkpoint(self, workflow_id: str, state: MessagesState, metadata: Optional[Dict[str, Any]] = None) -> Checkpoint:
        """创建新检查点"""
        checkpoint = Checkpoint(
            workflow_id=workflow_id,
            state=state,
            metadata=metadata or {}
        )
        
        success = await self.storage.save_checkpoint(checkpoint)
        if not success:
            raise RuntimeError(f"创建检查点失败: {checkpoint.id}")
        
        return checkpoint
    
    async def restore_from_checkpoint(self, workflow_id: str, checkpoint_id: str) -> Optional[MessagesState]:
        """从检查点恢复状态"""
        checkpoint = await self.storage.load_checkpoint(workflow_id, checkpoint_id)
        return checkpoint.state if checkpoint else None
    
    async def get_latest_checkpoint(self, workflow_id: str) -> Optional[Checkpoint]:
        """获取最新检查点"""
        checkpoints = await self.storage.list_checkpoints(workflow_id)
        return checkpoints[0] if checkpoints else None
    
    async def cleanup_workflow_checkpoints(self, workflow_id: str, keep_count: int = 10) -> int:
        """清理工作流检查点"""
        return await self.storage.cleanup_old_checkpoints(workflow_id, keep_count)


# 全局检查点管理器实例
checkpoint_manager = CheckpointManager()