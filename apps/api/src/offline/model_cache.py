"""
本地模型缓存管理器

实现压缩模型的本地缓存，支持：
- 模型版本管理
- 预加载策略
- 缓存优化
- 模型量化支持
"""

import os
import json
import hashlib
from src.core.utils import secure_pickle as pickle
import gzip
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ..core.config import get_settings
from ..core.logging import get_logger
from ..core.utils.timezone_utils import utc_now

from src.core.logging import get_logger
@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    version: str
    size_bytes: int
    checksum: str
    created_at: datetime
    last_used: datetime
    use_count: int
    compression_type: str
    quantization_level: Optional[str] = None
    tags: List[str] = None

@dataclass
class CacheEntry:
    """缓存条目"""
    model_id: str
    data: Any
    metadata: ModelMetadata
    is_loaded: bool = False
    load_time: Optional[datetime] = None

class ModelCompressionManager:
    """模型压缩管理器"""
    
    @staticmethod
    def compress_model(model_data: bytes, compression_type: str = "gzip") -> Tuple[bytes, float]:
        """压缩模型数据"""
        original_size = len(model_data)
        
        if compression_type == "gzip":
            compressed_data = gzip.compress(model_data, compresslevel=9)
        else:
            # 可以添加其他压缩算法
            compressed_data = model_data
        
        compression_ratio = len(compressed_data) / original_size
        return compressed_data, compression_ratio
    
    @staticmethod
    def decompress_model(compressed_data: bytes, compression_type: str = "gzip") -> bytes:
        """解压模型数据"""
        if compression_type == "gzip":
            return gzip.decompress(compressed_data)
        else:
            return compressed_data
    
    @staticmethod
    def calculate_checksum(data: bytes) -> str:
        """计算数据校验和"""
        return hashlib.sha256(data).hexdigest()

class ModelCacheManager:
    """本地模型缓存管理器"""
    
    def __init__(self, cache_dir: Optional[str] = None, max_cache_size_gb: float = 5.0):
        self.logger = get_logger(__name__)
        settings = get_settings()
        self.cache_dir = Path(cache_dir or Path(settings.OFFLINE_STORAGE_PATH) / "model_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.compression_manager = ModelCompressionManager()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # 内存缓存
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """加载缓存元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    self._cache_metadata = {
                        model_id: ModelMetadata(
                            model_id=meta['model_id'],
                            version=meta['version'],
                            size_bytes=meta['size_bytes'],
                            checksum=meta['checksum'],
                            created_at=datetime.fromisoformat(meta['created_at']),
                            last_used=datetime.fromisoformat(meta['last_used']),
                            use_count=meta['use_count'],
                            compression_type=meta['compression_type'],
                            quantization_level=meta.get('quantization_level'),
                            tags=meta.get('tags', [])
                        )
                        for model_id, meta in data.items()
                    }
            except Exception as e:
                self.logger.error("加载模型缓存元数据失败", error=str(e))
                self._cache_metadata = {}
        else:
            self._cache_metadata = {}
    
    def _save_metadata(self):
        """保存缓存元数据"""
        try:
            data = {
                model_id: {
                    'model_id': meta.model_id,
                    'version': meta.version,
                    'size_bytes': meta.size_bytes,
                    'checksum': meta.checksum,
                    'created_at': meta.created_at.isoformat(),
                    'last_used': meta.last_used.isoformat(),
                    'use_count': meta.use_count,
                    'compression_type': meta.compression_type,
                    'quantization_level': meta.quantization_level,
                    'tags': meta.tags or []
                }
                for model_id, meta in self._cache_metadata.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.logger.error("保存模型缓存元数据失败", error=str(e))
    
    def _get_model_path(self, model_id: str, version: str) -> Path:
        """获取模型文件路径"""
        return self.cache_dir / f"{model_id}_{version}.model"
    
    def _cleanup_cache(self):
        """清理缓存空间"""
        total_size = sum(meta.size_bytes for meta in self._cache_metadata.values())
        
        if total_size <= self.max_cache_size_bytes:
            return
        
        # 按最后使用时间和使用频率排序
        sorted_models = sorted(
            self._cache_metadata.values(),
            key=lambda m: (m.last_used, m.use_count)
        )
        
        # 删除最旧的模型直到空间足够
        for model_meta in sorted_models:
            if total_size <= self.max_cache_size_bytes * 0.8:  # 保留20%空间
                break
            
            model_path = self._get_model_path(model_meta.model_id, model_meta.version)
            if model_path.exists():
                model_path.unlink()
                total_size -= model_meta.size_bytes
                
                # 从内存缓存中移除
                if model_meta.model_id in self._memory_cache:
                    del self._memory_cache[model_meta.model_id]
                
                # 从元数据中移除
                del self._cache_metadata[model_meta.model_id]
        
        self._save_metadata()
    
    async def cache_model(
        self, 
        model_id: str, 
        version: str, 
        model_data: Any,
        compression_type: str = "gzip",
        quantization_level: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """缓存模型到本地"""
        try:
            # 序列化模型数据
            serialized_data = pickle.dumps(model_data)
            
            # 压缩模型数据
            compressed_data, compression_ratio = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                self.compression_manager.compress_model,
                serialized_data,
                compression_type
            )
            
            # 计算校验和
            checksum = self.compression_manager.calculate_checksum(compressed_data)
            
            # 检查缓存空间
            self._cleanup_cache()
            
            # 保存到文件
            model_path = self._get_model_path(model_id, version)
            with open(model_path, 'wb') as f:
                f.write(compressed_data)
            
            # 更新元数据
            metadata = ModelMetadata(
                model_id=model_id,
                version=version,
                size_bytes=len(compressed_data),
                checksum=checksum,
                created_at=utc_now(),
                last_used=utc_now(),
                use_count=0,
                compression_type=compression_type,
                quantization_level=quantization_level,
                tags=tags or []
            )
            
            self._cache_metadata[model_id] = metadata
            self._save_metadata()
            
            # 添加到内存缓存
            cache_entry = CacheEntry(
                model_id=model_id,
                data=model_data,
                metadata=metadata,
                is_loaded=True,
                load_time=utc_now()
            )
            self._memory_cache[model_id] = cache_entry
            
            self.logger.info(
                "模型已缓存",
                model_id=model_id,
                version=version,
                compression_ratio=round(compression_ratio, 4),
            )
            
            return True
            
        except Exception as e:
            self.logger.error("缓存模型失败", model_id=model_id, error=str(e))
            return False
    
    async def load_model(self, model_id: str, preload_to_memory: bool = True) -> Optional[Any]:
        """加载缓存的模型"""
        # 首先检查内存缓存
        if model_id in self._memory_cache and self._memory_cache[model_id].is_loaded:
            cache_entry = self._memory_cache[model_id]
            # 更新使用统计
            cache_entry.metadata.last_used = utc_now()
            cache_entry.metadata.use_count += 1
            self._save_metadata()
            return cache_entry.data
        
        # 检查磁盘缓存
        if model_id not in self._cache_metadata:
            return None
        
        metadata = self._cache_metadata[model_id]
        model_path = self._get_model_path(model_id, metadata.version)
        
        if not model_path.exists():
            # 文件不存在，清理元数据
            del self._cache_metadata[model_id]
            self._save_metadata()
            return None
        
        try:
            # 从磁盘加载
            with open(model_path, 'rb') as f:
                compressed_data = f.read()
            
            # 验证校验和
            checksum = self.compression_manager.calculate_checksum(compressed_data)
            if checksum != metadata.checksum:
                self.logger.error("模型校验和不匹配", model_id=model_id)
                return None
            
            # 解压缩
            serialized_data = await asyncio.get_running_loop().run_in_executor(
                self.executor,
                self.compression_manager.decompress_model,
                compressed_data,
                metadata.compression_type
            )
            
            # 反序列化
            model_data = pickle.loads(serialized_data)
            
            # 更新使用统计
            metadata.last_used = utc_now()
            metadata.use_count += 1
            self._save_metadata()
            
            # 添加到内存缓存
            if preload_to_memory:
                cache_entry = CacheEntry(
                    model_id=model_id,
                    data=model_data,
                    metadata=metadata,
                    is_loaded=True,
                    load_time=utc_now()
                )
                self._memory_cache[model_id] = cache_entry
            
            return model_data
            
        except Exception as e:
            self.logger.error("加载模型失败", model_id=model_id, error=str(e))
            return None
    
    def get_model_info(self, model_id: str) -> Optional[ModelMetadata]:
        """获取模型信息"""
        return self._cache_metadata.get(model_id)
    
    def list_cached_models(self, tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """列出缓存的模型"""
        models = list(self._cache_metadata.values())
        
        if tags:
            models = [
                model for model in models
                if any(tag in (model.tags or []) for tag in tags)
            ]
        
        # 按最后使用时间排序
        models.sort(key=lambda m: m.last_used, reverse=True)
        return models
    
    def remove_model(self, model_id: str) -> bool:
        """移除缓存的模型"""
        if model_id not in self._cache_metadata:
            return False
        
        try:
            metadata = self._cache_metadata[model_id]
            model_path = self._get_model_path(model_id, metadata.version)
            
            # 删除文件
            if model_path.exists():
                model_path.unlink()
            
            # 从内存缓存移除
            if model_id in self._memory_cache:
                del self._memory_cache[model_id]
            
            # 从元数据移除
            del self._cache_metadata[model_id]
            self._save_metadata()
            
            return True
            
        except Exception as e:
            self.logger.error("移除模型失败", model_id=model_id, error=str(e))
            return False
    
    async def preload_models(self, model_ids: List[str], max_concurrent: int = 2) -> Dict[str, bool]:
        """预加载多个模型"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def load_single_model(model_id: str) -> Tuple[str, bool]:
            async with semaphore:
                model = await self.load_model(model_id, preload_to_memory=True)
                return model_id, model is not None
        
        tasks = [load_single_model(model_id) for model_id in model_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            model_id: success for model_id, success in results
            if not isinstance(success, Exception)
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_models = len(self._cache_metadata)
        total_size = sum(meta.size_bytes for meta in self._cache_metadata.values())
        memory_loaded = len(self._memory_cache)
        
        if total_models > 0:
            avg_size = total_size / total_models
            most_used = max(self._cache_metadata.values(), key=lambda m: m.use_count)
            least_used = min(self._cache_metadata.values(), key=lambda m: m.use_count)
        else:
            avg_size = 0
            most_used = None
            least_used = None
        
        return {
            'total_models': total_models,
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'max_cache_size_gb': self.max_cache_size_bytes / (1024 * 1024 * 1024),
            'cache_usage_percent': (total_size / self.max_cache_size_bytes) * 100,
            'memory_loaded_models': memory_loaded,
            'average_model_size_mb': avg_size / (1024 * 1024),
            'most_used_model': most_used.model_id if most_used else None,
            'least_used_model': least_used.model_id if least_used else None
        }
    
    def cleanup_old_models(self, days: int = 30) -> int:
        """清理旧模型"""
        cutoff_date = utc_now() - timedelta(days=days)
        removed_count = 0
        
        models_to_remove = [
            model_id for model_id, meta in self._cache_metadata.items()
            if meta.last_used < cutoff_date
        ]
        
        for model_id in models_to_remove:
            if self.remove_model(model_id):
                removed_count += 1
        
        return removed_count
    
    def close(self):
        """关闭缓存管理器"""
        self._save_metadata()
        self.executor.shutdown(wait=True)
