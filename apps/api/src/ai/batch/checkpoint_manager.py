"""
批处理检查点管理器

提供批处理任务的检查点保存、恢复和断点续传功能。
"""

import os
import json
import pickle
import asyncio
import hashlib
import time
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from pathlib import Path
import sqlite3
from contextlib import asynccontextmanager

# Import from shared types to avoid circular dependency
from .batch_types import BatchJob, BatchTask, BatchStatus

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """检查点元数据"""
    checkpoint_id: str
    job_id: str
    created_at: datetime
    checkpoint_type: str  # 'full', 'incremental', 'emergency'
    task_count: int
    completed_tasks: int
    failed_tasks: int
    file_path: str
    file_size: int
    checksum: str
    version: str = "1.0"
    dependencies: List[str] = field(default_factory=list)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    """检查点配置"""
    enabled: bool = True
    storage_path: str = "./checkpoints"
    auto_save_interval: float = 300.0  # 5分钟
    max_checkpoints_per_job: int = 10
    compression_enabled: bool = True
    encryption_enabled: bool = False
    cleanup_older_than_days: int = 7
    incremental_threshold: float = 0.1  # 10%任务变化触发增量检查点


class CheckpointStorage:
    """检查点存储引擎"""
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化SQLite数据库存储元数据
        self.db_path = self.storage_path / "checkpoints.db"
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    job_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    checkpoint_type TEXT NOT NULL,
                    task_count INTEGER NOT NULL,
                    completed_tasks INTEGER NOT NULL,
                    failed_tasks INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    version TEXT NOT NULL,
                    dependencies TEXT,
                    tags TEXT,
                    FOREIGN KEY (job_id) REFERENCES jobs (id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_job_id ON checkpoints (job_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON checkpoints (created_at)
            """)
    
    async def save_checkpoint(self, metadata: CheckpointMetadata, data: bytes) -> bool:
        """保存检查点"""
        try:
            # 写入文件
            file_path = self.storage_path / f"{metadata.checkpoint_id}.ckpt"
            with open(file_path, 'wb') as f:
                f.write(data)
            
            # 更新元数据
            metadata.file_path = str(file_path)
            metadata.file_size = len(data)
            metadata.checksum = hashlib.sha256(data).hexdigest()
            
            # 保存到数据库
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO checkpoints 
                    (checkpoint_id, job_id, created_at, checkpoint_type, task_count, 
                     completed_tasks, failed_tasks, file_path, file_size, checksum, 
                     version, dependencies, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    metadata.checkpoint_id,
                    metadata.job_id,
                    metadata.created_at.isoformat(),
                    metadata.checkpoint_type,
                    metadata.task_count,
                    metadata.completed_tasks,
                    metadata.failed_tasks,
                    metadata.file_path,
                    metadata.file_size,
                    metadata.checksum,
                    metadata.version,
                    json.dumps(metadata.dependencies),
                    json.dumps(metadata.tags)
                ))
            
            logger.info(f"检查点保存成功: {metadata.checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"保存检查点失败: {metadata.checkpoint_id} - {e}")
            return False
    
    async def load_checkpoint(self, checkpoint_id: str) -> Optional[tuple[CheckpointMetadata, bytes]]:
        """加载检查点"""
        try:
            # 从数据库获取元数据
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM checkpoints WHERE checkpoint_id = ?
                """, (checkpoint_id,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                # 构建元数据对象
                metadata = CheckpointMetadata(
                    checkpoint_id=row[0],
                    job_id=row[1],
                    created_at=datetime.fromisoformat(row[2]),
                    checkpoint_type=row[3],
                    task_count=row[4],
                    completed_tasks=row[5],
                    failed_tasks=row[6],
                    file_path=row[7],
                    file_size=row[8],
                    checksum=row[9],
                    version=row[10],
                    dependencies=json.loads(row[11] or "[]"),
                    tags=json.loads(row[12] or "{}")
                )
            
            # 读取检查点文件
            with open(metadata.file_path, 'rb') as f:
                data = f.read()
            
            # 验证校验和
            if hashlib.sha256(data).hexdigest() != metadata.checksum:
                raise ValueError("检查点文件校验和不匹配")
            
            logger.info(f"检查点加载成功: {checkpoint_id}")
            return metadata, data
            
        except Exception as e:
            logger.error(f"加载检查点失败: {checkpoint_id} - {e}")
            return None
    
    async def list_checkpoints(self, job_id: Optional[str] = None) -> List[CheckpointMetadata]:
        """列出检查点"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                if job_id:
                    cursor = conn.execute("""
                        SELECT * FROM checkpoints WHERE job_id = ? 
                        ORDER BY created_at DESC
                    """, (job_id,))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM checkpoints ORDER BY created_at DESC
                    """)
                
                checkpoints = []
                for row in cursor.fetchall():
                    metadata = CheckpointMetadata(
                        checkpoint_id=row[0],
                        job_id=row[1],
                        created_at=datetime.fromisoformat(row[2]),
                        checkpoint_type=row[3],
                        task_count=row[4],
                        completed_tasks=row[5],
                        failed_tasks=row[6],
                        file_path=row[7],
                        file_size=row[8],
                        checksum=row[9],
                        version=row[10],
                        dependencies=json.loads(row[11] or "[]"),
                        tags=json.loads(row[12] or "{}")
                    )
                    checkpoints.append(metadata)
                
                return checkpoints
                
        except Exception as e:
            logger.error(f"列出检查点失败: {e}")
            return []
    
    async def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """删除检查点"""
        try:
            # 获取文件路径
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT file_path FROM checkpoints WHERE checkpoint_id = ?
                """, (checkpoint_id,))
                
                row = cursor.fetchone()
                if row:
                    file_path = row[0]
                    
                    # 删除文件
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    
                    # 从数据库删除
                    conn.execute("""
                        DELETE FROM checkpoints WHERE checkpoint_id = ?
                    """, (checkpoint_id,))
                    
                    logger.info(f"检查点删除成功: {checkpoint_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"删除检查点失败: {checkpoint_id} - {e}")
            return False


class CheckpointManager:
    """检查点管理器"""
    
    def __init__(self, config: CheckpointConfig = None):
        self.config = config or CheckpointConfig()
        self.storage = CheckpointStorage(self.config.storage_path)
        
        # 自动保存任务
        self._auto_save_task: Optional[asyncio.Task] = None
        self._auto_save_enabled = False
        
        # 作业状态缓存
        self._job_states: Dict[str, Dict] = {}
        self._last_checkpoint_time: Dict[str, float] = {}
    
    async def start_auto_save(self):
        """启动自动保存"""
        if self._auto_save_enabled or not self.config.enabled:
            return
        
        self._auto_save_enabled = True
        self._auto_save_task = asyncio.create_task(self._auto_save_loop())
        logger.info(f"启动自动检查点保存 (间隔: {self.config.auto_save_interval}s)")
    
    async def stop_auto_save(self):
        """停止自动保存"""
        self._auto_save_enabled = False
        if self._auto_save_task:
            self._auto_save_task.cancel()
            try:
                await self._auto_save_task
            except asyncio.CancelledError:
                pass
        logger.info("停止自动检查点保存")
    
    async def _auto_save_loop(self):
        """自动保存循环"""
        while self._auto_save_enabled:
            try:
                await asyncio.sleep(self.config.auto_save_interval)
                
                # 检查需要保存检查点的作业
                for job_id in list(self._job_states.keys()):
                    if await self._should_create_checkpoint(job_id):
                        job_data = self._job_states[job_id]
                        await self.create_checkpoint(job_data['job'], 'auto')
                
            except Exception as e:
                logger.error(f"自动保存检查点出错: {e}")
                await asyncio.sleep(60)  # 出错后等待1分钟再重试
    
    async def _should_create_checkpoint(self, job_id: str) -> bool:
        """判断是否应该创建检查点"""
        if job_id not in self._job_states:
            return False
        
        job_data = self._job_states[job_id]
        job = job_data['job']
        
        # 检查作业状态
        if job.status not in [BatchStatus.RUNNING]:
            return False
        
        # 检查时间间隔
        last_checkpoint = self._last_checkpoint_time.get(job_id, 0)
        if time.time() - last_checkpoint < self.config.auto_save_interval:
            return False
        
        # 检查任务进度变化
        last_completed = job_data.get('last_completed', 0)
        progress_change = (job.completed_tasks - last_completed) / job.total_tasks
        
        return progress_change >= self.config.incremental_threshold
    
    async def register_job(self, job):
        """注册作业以进行检查点管理"""
        self._job_states[job.id] = {
            'job': job,
            'last_completed': job.completed_tasks,
            'registered_at': time.time()
        }
        logger.debug(f"注册作业检查点管理: {job.id}")
    
    async def unregister_job(self, job_id: str):
        """注销作业检查点管理"""
        if job_id in self._job_states:
            del self._job_states[job_id]
            if job_id in self._last_checkpoint_time:
                del self._last_checkpoint_time[job_id]
        logger.debug(f"注销作业检查点管理: {job_id}")
    
    async def create_checkpoint(self, job, checkpoint_type: str = 'manual') -> Optional[str]:
        """创建检查点"""
        if not self.config.enabled:
            return None
        
        try:
            # 生成检查点ID
            checkpoint_id = f"{job.id}_{checkpoint_type}_{int(time.time())}"
            
            # 序列化作业数据
            job_data = {
                'job': asdict(job),
                'tasks': [asdict(task) for task in job.tasks],
                'checkpoint_metadata': {
                    'created_by': checkpoint_type,
                    'progress': job.progress,
                    'success_rate': job.success_rate
                }
            }
            
            # 压缩和序列化
            serialized_data = pickle.dumps(job_data)
            if self.config.compression_enabled:
                import gzip
                serialized_data = gzip.compress(serialized_data)
            
            # 创建元数据
            metadata = CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                job_id=job.id,
                created_at=utc_now(),
                checkpoint_type=checkpoint_type,
                task_count=job.total_tasks,
                completed_tasks=job.completed_tasks,
                failed_tasks=job.failed_tasks,
                file_path="",  # 将由存储引擎填充
                file_size=0,   # 将由存储引擎填充
                checksum="",   # 将由存储引擎填充
                tags={
                    'job_name': job.name,
                    'job_priority': str(job.priority)
                }
            )
            
            # 保存检查点
            success = await self.storage.save_checkpoint(metadata, serialized_data)
            
            if success:
                # 更新状态
                self._last_checkpoint_time[job.id] = time.time()
                if job.id in self._job_states:
                    self._job_states[job.id]['last_completed'] = job.completed_tasks
                
                # 清理旧检查点
                await self._cleanup_old_checkpoints(job.id)
                
                logger.info(f"创建检查点成功: {checkpoint_id} (类型: {checkpoint_type})")
                return checkpoint_id
            
            return None
            
        except Exception as e:
            logger.error(f"创建检查点失败: {job.id} - {e}")
            return None
    
    async def restore_job(self, checkpoint_id: str):
        """从检查点恢复作业"""
        try:
            # 加载检查点
            result = await self.storage.load_checkpoint(checkpoint_id)
            if not result:
                return None
            
            metadata, data = result
            
            # 解压缩和反序列化
            if self.config.compression_enabled:
                import gzip
                data = gzip.decompress(data)
            
            job_data = pickle.loads(data)
            
            # 重建作业对象
            job_dict = job_data['job']
            tasks_dict = job_data['tasks']
            
            # 使用已导入的类型
            
            # 重建任务对象
            tasks = []
            for task_dict in tasks_dict:
                task = BatchTask(**task_dict)
                # 修复日期时间字段
                if isinstance(task.created_at, str):
                    task.created_at = datetime.fromisoformat(task.created_at)
                if isinstance(task.started_at, str):
                    task.started_at = datetime.fromisoformat(task.started_at) if task.started_at else None
                if isinstance(task.completed_at, str):
                    task.completed_at = datetime.fromisoformat(task.completed_at) if task.completed_at else None
                tasks.append(task)
            
            # 重建作业对象
            job = BatchJob(**job_dict)
            job.tasks = tasks
            
            # 修复日期时间字段
            if isinstance(job.created_at, str):
                job.created_at = datetime.fromisoformat(job.created_at)
            if isinstance(job.started_at, str):
                job.started_at = datetime.fromisoformat(job.started_at) if job.started_at else None
            if isinstance(job.completed_at, str):
                job.completed_at = datetime.fromisoformat(job.completed_at) if job.completed_at else None
            
            logger.info(f"作业恢复成功: {job.id} (检查点: {checkpoint_id})")
            return job
            
        except Exception as e:
            logger.error(f"恢复作业失败: {checkpoint_id} - {e}")
            return None
    
    async def get_latest_checkpoint(self, job_id: str) -> Optional[CheckpointMetadata]:
        """获取作业的最新检查点"""
        checkpoints = await self.storage.list_checkpoints(job_id)
        return checkpoints[0] if checkpoints else None
    
    async def list_job_checkpoints(self, job_id: str) -> List[CheckpointMetadata]:
        """列出作业的所有检查点"""
        return await self.storage.list_checkpoints(job_id)
    
    async def delete_job_checkpoints(self, job_id: str) -> int:
        """删除作业的所有检查点"""
        checkpoints = await self.storage.list_checkpoints(job_id)
        deleted_count = 0
        
        for checkpoint in checkpoints:
            if await self.storage.delete_checkpoint(checkpoint.checkpoint_id):
                deleted_count += 1
        
        logger.info(f"删除作业检查点: {job_id} ({deleted_count}个)")
        return deleted_count
    
    async def _cleanup_old_checkpoints(self, job_id: str):
        """清理旧检查点"""
        checkpoints = await self.storage.list_checkpoints(job_id)
        
        # 按创建时间排序，保留最新的N个
        if len(checkpoints) > self.config.max_checkpoints_per_job:
            to_delete = checkpoints[self.config.max_checkpoints_per_job:]
            for checkpoint in to_delete:
                await self.storage.delete_checkpoint(checkpoint.checkpoint_id)
    
    async def cleanup_expired_checkpoints(self):
        """清理过期检查点"""
        cutoff_date = utc_now() - timedelta(days=self.config.cleanup_older_than_days)
        all_checkpoints = await self.storage.list_checkpoints()
        
        deleted_count = 0
        for checkpoint in all_checkpoints:
            if checkpoint.created_at < cutoff_date:
                if await self.storage.delete_checkpoint(checkpoint.checkpoint_id):
                    deleted_count += 1
        
        if deleted_count > 0:
            logger.info(f"清理过期检查点: {deleted_count}个")
        
        return deleted_count
    
    async def get_checkpoint_stats(self) -> Dict[str, Any]:
        """获取检查点统计信息"""
        all_checkpoints = await self.storage.list_checkpoints()
        
        stats = {
            'total_checkpoints': len(all_checkpoints),
            'total_size_bytes': sum(cp.file_size for cp in all_checkpoints),
            'jobs_with_checkpoints': len(set(cp.job_id for cp in all_checkpoints)),
            'checkpoint_types': {},
            'oldest_checkpoint': None,
            'newest_checkpoint': None
        }
        
        if all_checkpoints:
            # 按类型分组
            for checkpoint in all_checkpoints:
                cp_type = checkpoint.checkpoint_type
                if cp_type not in stats['checkpoint_types']:
                    stats['checkpoint_types'][cp_type] = 0
                stats['checkpoint_types'][cp_type] += 1
            
            # 最新和最旧
            sorted_checkpoints = sorted(all_checkpoints, key=lambda x: x.created_at)
            stats['oldest_checkpoint'] = sorted_checkpoints[0].created_at.isoformat()
            stats['newest_checkpoint'] = sorted_checkpoints[-1].created_at.isoformat()
        
        return stats


# 全局检查点管理器实例
checkpoint_manager = CheckpointManager()