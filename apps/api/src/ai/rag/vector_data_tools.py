"""
向量数据导入导出工具

支持多种格式的向量数据导入导出、迁移和备份恢复
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, BinaryIO, TextIO
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
import json
import csv
from src.core.utils import secure_pickle as pickle
import h5py
import struct
import gzip
import shutil
from pathlib import Path
import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
import aiofiles
import io

from src.core.logging import get_logger
logger = get_logger(__name__)

class DataFormat(str, Enum):
    """数据格式"""
    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    HDF5 = "hdf5"
    NUMPY = "numpy"
    PICKLE = "pickle"
    FAISS = "faiss"
    ONNX = "onnx"
    BINARY = "binary"

class CompressionType(str, Enum):
    """压缩类型"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZ4 = "lz4"
    ZSTD = "zstd"

@dataclass
class ImportConfig:
    """导入配置"""
    format: DataFormat = DataFormat.CSV
    batch_size: int = 1000
    validate: bool = True
    normalize: bool = False
    dimension: Optional[int] = None
    skip_errors: bool = False
    compression: CompressionType = CompressionType.NONE
    encoding: str = "utf-8"

@dataclass
class ExportConfig:
    """导出配置"""
    format: DataFormat = DataFormat.JSON
    batch_size: int = 1000
    include_metadata: bool = True
    compression: CompressionType = CompressionType.NONE
    split_files: bool = False
    max_file_size_mb: int = 100

@dataclass
class MigrationConfig:
    """迁移配置"""
    source_type: str = "pgvector"
    target_type: str = "pgvector"
    batch_size: int = 1000
    parallel_workers: int = 4
    verify_after: bool = True

class VectorDataImporter:
    """向量数据导入器"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.import_stats = {
            "total_vectors": 0,
            "successful": 0,
            "failed": 0,
            "skipped": 0
        }
        
    async def import_from_file(
        self,
        file_path: str,
        table_name: str,
        config: ImportConfig
    ) -> Dict[str, Any]:
        """从文件导入向量数据"""
        try:
            file_path = Path(file_path)
            
            # 解压文件（如需要）
            if config.compression != CompressionType.NONE:
                file_path = await self._decompress_file(file_path, config.compression)
            
            # 根据格式选择导入方法
            if config.format == DataFormat.CSV:
                result = await self._import_csv(file_path, table_name, config)
            elif config.format == DataFormat.JSON:
                result = await self._import_json(file_path, table_name, config)
            elif config.format == DataFormat.JSONL:
                result = await self._import_jsonl(file_path, table_name, config)
            elif config.format == DataFormat.PARQUET:
                result = await self._import_parquet(file_path, table_name, config)
            elif config.format == DataFormat.HDF5:
                result = await self._import_hdf5(file_path, table_name, config)
            elif config.format == DataFormat.NUMPY:
                result = await self._import_numpy(file_path, table_name, config)
            elif config.format == DataFormat.PICKLE:
                result = await self._import_pickle(file_path, table_name, config)
            elif config.format == DataFormat.BINARY:
                result = await self._import_binary(file_path, table_name, config)
            else:
                raise ValueError(f"不支持的导入格式: {config.format}")
            
            logger.info(f"导入完成: {result}")
            return result
            
        except Exception as e:
            logger.error(f"导入失败: {e}")
            raise
    
    async def _import_csv(
        self,
        file_path: Path,
        table_name: str,
        config: ImportConfig
    ) -> Dict[str, Any]:
        """导入CSV格式数据"""
        vectors_imported = 0
        
        async with aiofiles.open(file_path, 'r', encoding=config.encoding) as f:
            content = await f.read()
            reader = csv.DictReader(io.StringIO(content))
            
            batch = []
            for row in reader:
                try:
                    # 解析向量
                    vector_str = row.get('embedding', row.get('vector', ''))
                    vector = self._parse_vector_string(vector_str)
                    
                    if config.validate and not self._validate_vector(vector, config.dimension):
                        self.import_stats["skipped"] += 1
                        continue
                    
                    if config.normalize:
                        vector = vector / np.linalg.norm(vector)
                    
                    # 准备数据
                    data = {
                        'id': row.get('id'),
                        'embedding': vector.tolist(),
                        'metadata': json.loads(row.get('metadata', '{}') if row.get('metadata') else '{}')
                    }
                    
                    batch.append(data)
                    
                    # 批量插入
                    if len(batch) >= config.batch_size:
                        await self._insert_batch(table_name, batch)
                        vectors_imported += len(batch)
                        batch = []
                        
                except Exception as e:
                    self.import_stats["failed"] += 1
                    if not config.skip_errors:
                        raise
                    logger.warning(f"跳过错误行: {e}")
            
            # 插入剩余数据
            if batch:
                await self._insert_batch(table_name, batch)
                vectors_imported += len(batch)
        
        self.import_stats["successful"] = vectors_imported
        return self.import_stats
    
    async def _import_json(
        self,
        file_path: Path,
        table_name: str,
        config: ImportConfig
    ) -> Dict[str, Any]:
        """导入JSON格式数据"""
        async with aiofiles.open(file_path, 'r', encoding=config.encoding) as f:
            content = await f.read()
            data = json.loads(content)
        
        if isinstance(data, dict) and 'vectors' in data:
            vectors_data = data['vectors']
        elif isinstance(data, list):
            vectors_data = data
        else:
            raise ValueError("无效的JSON格式")
        
        vectors_imported = 0
        batch = []
        
        for item in vectors_data:
            try:
                vector = np.array(item['embedding'])
                
                if config.validate and not self._validate_vector(vector, config.dimension):
                    self.import_stats["skipped"] += 1
                    continue
                
                if config.normalize:
                    vector = vector / np.linalg.norm(vector)
                
                batch.append({
                    'id': item.get('id'),
                    'embedding': vector.tolist(),
                    'metadata': item.get('metadata', {})
                })
                
                if len(batch) >= config.batch_size:
                    await self._insert_batch(table_name, batch)
                    vectors_imported += len(batch)
                    batch = []
                    
            except Exception as e:
                self.import_stats["failed"] += 1
                if not config.skip_errors:
                    raise
                logger.warning(f"跳过错误项: {e}")
        
        if batch:
            await self._insert_batch(table_name, batch)
            vectors_imported += len(batch)
        
        self.import_stats["successful"] = vectors_imported
        return self.import_stats
    
    async def _import_jsonl(
        self,
        file_path: Path,
        table_name: str,
        config: ImportConfig
    ) -> Dict[str, Any]:
        """导入JSONL格式数据（每行一个JSON对象）"""
        vectors_imported = 0
        batch = []
        
        async with aiofiles.open(file_path, 'r', encoding=config.encoding) as f:
            async for line in f:
                try:
                    item = json.loads(line.strip())
                    vector = np.array(item['embedding'])
                    
                    if config.validate and not self._validate_vector(vector, config.dimension):
                        self.import_stats["skipped"] += 1
                        continue
                    
                    if config.normalize:
                        vector = vector / np.linalg.norm(vector)
                    
                    batch.append({
                        'id': item.get('id'),
                        'embedding': vector.tolist(),
                        'metadata': item.get('metadata', {})
                    })
                    
                    if len(batch) >= config.batch_size:
                        await self._insert_batch(table_name, batch)
                        vectors_imported += len(batch)
                        batch = []
                        
                except Exception as e:
                    self.import_stats["failed"] += 1
                    if not config.skip_errors:
                        raise
                    logger.warning(f"跳过错误行: {e}")
        
        if batch:
            await self._insert_batch(table_name, batch)
            vectors_imported += len(batch)
        
        self.import_stats["successful"] = vectors_imported
        return self.import_stats
    
    async def _import_parquet(
        self,
        file_path: Path,
        table_name: str,
        config: ImportConfig
    ) -> Dict[str, Any]:
        """导入Parquet格式数据"""
        df = pd.read_parquet(file_path)
        
        vectors_imported = 0
        batch = []
        
        for _, row in df.iterrows():
            try:
                vector = np.array(row['embedding'])
                
                if config.validate and not self._validate_vector(vector, config.dimension):
                    self.import_stats["skipped"] += 1
                    continue
                
                if config.normalize:
                    vector = vector / np.linalg.norm(vector)
                
                batch.append({
                    'id': row.get('id'),
                    'embedding': vector.tolist(),
                    'metadata': row.get('metadata', {})
                })
                
                if len(batch) >= config.batch_size:
                    await self._insert_batch(table_name, batch)
                    vectors_imported += len(batch)
                    batch = []
                    
            except Exception as e:
                self.import_stats["failed"] += 1
                if not config.skip_errors:
                    raise
        
        if batch:
            await self._insert_batch(table_name, batch)
            vectors_imported += len(batch)
        
        self.import_stats["successful"] = vectors_imported
        return self.import_stats
    
    async def _import_hdf5(
        self,
        file_path: Path,
        table_name: str,
        config: ImportConfig
    ) -> Dict[str, Any]:
        """导入HDF5格式数据"""
        vectors_imported = 0
        
        with h5py.File(file_path, 'r') as f:
            vectors = f['vectors'][:]
            ids = f.get('ids', range(len(vectors)))[:]
            metadata = f.get('metadata', [{}] * len(vectors))[:]
            
            batch = []
            for i in range(len(vectors)):
                try:
                    vector = vectors[i]
                    
                    if config.validate and not self._validate_vector(vector, config.dimension):
                        self.import_stats["skipped"] += 1
                        continue
                    
                    if config.normalize:
                        vector = vector / np.linalg.norm(vector)
                    
                    batch.append({
                        'id': str(ids[i]),
                        'embedding': vector.tolist(),
                        'metadata': metadata[i] if isinstance(metadata[i], dict) else {}
                    })
                    
                    if len(batch) >= config.batch_size:
                        await self._insert_batch(table_name, batch)
                        vectors_imported += len(batch)
                        batch = []
                        
                except Exception as e:
                    self.import_stats["failed"] += 1
                    if not config.skip_errors:
                        raise
            
            if batch:
                await self._insert_batch(table_name, batch)
                vectors_imported += len(batch)
        
        self.import_stats["successful"] = vectors_imported
        return self.import_stats
    
    async def _import_numpy(
        self,
        file_path: Path,
        table_name: str,
        config: ImportConfig
    ) -> Dict[str, Any]:
        """导入NumPy格式数据"""
        vectors = np.load(file_path)
        
        if len(vectors.shape) != 2:
            raise ValueError("NumPy数组必须是2维的")
        
        vectors_imported = 0
        batch = []
        
        for i, vector in enumerate(vectors):
            try:
                if config.validate and not self._validate_vector(vector, config.dimension):
                    self.import_stats["skipped"] += 1
                    continue
                
                if config.normalize:
                    vector = vector / np.linalg.norm(vector)
                
                batch.append({
                    'id': str(i),
                    'embedding': vector.tolist(),
                    'metadata': {}
                })
                
                if len(batch) >= config.batch_size:
                    await self._insert_batch(table_name, batch)
                    vectors_imported += len(batch)
                    batch = []
                    
            except Exception as e:
                self.import_stats["failed"] += 1
                if not config.skip_errors:
                    raise
        
        if batch:
            await self._insert_batch(table_name, batch)
            vectors_imported += len(batch)
        
        self.import_stats["successful"] = vectors_imported
        return self.import_stats
    
    async def _import_pickle(
        self,
        file_path: Path,
        table_name: str,
        config: ImportConfig
    ) -> Dict[str, Any]:
        """导入Pickle格式数据"""
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, np.ndarray):
            # 纯向量数组
            return await self._import_numpy_array(data, table_name, config)
        elif isinstance(data, list):
            # 向量列表
            return await self._import_vector_list(data, table_name, config)
        elif isinstance(data, dict):
            # 包含向量和元数据的字典
            return await self._import_vector_dict(data, table_name, config)
        else:
            raise ValueError("不支持的Pickle数据格式")
    
    async def _import_binary(
        self,
        file_path: Path,
        table_name: str,
        config: ImportConfig
    ) -> Dict[str, Any]:
        """导入二进制格式数据"""
        vectors_imported = 0
        
        with open(file_path, 'rb') as f:
            # 读取头部信息
            n_vectors = struct.unpack('I', f.read(4))[0]
            dimension = struct.unpack('I', f.read(4))[0]
            
            if config.dimension and dimension != config.dimension:
                raise ValueError(f"维度不匹配: 期望{config.dimension}, 实际{dimension}")
            
            batch = []
            for i in range(n_vectors):
                try:
                    # 读取向量
                    vector_bytes = f.read(dimension * 4)  # float32
                    vector = np.frombuffer(vector_bytes, dtype=np.float32)
                    
                    if config.validate and not self._validate_vector(vector, dimension):
                        self.import_stats["skipped"] += 1
                        continue
                    
                    if config.normalize:
                        vector = vector / np.linalg.norm(vector)
                    
                    batch.append({
                        'id': str(i),
                        'embedding': vector.tolist(),
                        'metadata': {}
                    })
                    
                    if len(batch) >= config.batch_size:
                        await self._insert_batch(table_name, batch)
                        vectors_imported += len(batch)
                        batch = []
                        
                except Exception as e:
                    self.import_stats["failed"] += 1
                    if not config.skip_errors:
                        raise
            
            if batch:
                await self._insert_batch(table_name, batch)
                vectors_imported += len(batch)
        
        self.import_stats["successful"] = vectors_imported
        return self.import_stats
    
    def _parse_vector_string(self, vector_str: str) -> np.ndarray:
        """解析向量字符串"""
        # 支持多种格式: [1,2,3], (1,2,3), 1,2,3, "1 2 3"
        vector_str = vector_str.strip()
        
        if vector_str.startswith('[') and vector_str.endswith(']'):
            vector_str = vector_str[1:-1]
        elif vector_str.startswith('(') and vector_str.endswith(')'):
            vector_str = vector_str[1:-1]
        
        # 尝试不同的分隔符
        for delimiter in [',', ' ', '\t', ';']:
            try:
                values = [float(x.strip()) for x in vector_str.split(delimiter) if x.strip()]
                if values:
                    return np.array(values)
            except:
                continue
        
        raise ValueError(f"无法解析向量字符串: {vector_str}")
    
    def _validate_vector(self, vector: np.ndarray, expected_dim: Optional[int]) -> bool:
        """验证向量"""
        if expected_dim and len(vector) != expected_dim:
            return False
        
        if not np.isfinite(vector).all():
            return False
        
        return True
    
    async def _insert_batch(self, table_name: str, batch: List[Dict]) -> None:
        """批量插入数据"""
        if not batch:
            return
        
        # 构建插入语句
        insert_sql = f"""
        INSERT INTO {table_name} (id, embedding, metadata)
        VALUES (:id, :embedding, :metadata)
        ON CONFLICT (id) DO UPDATE
        SET embedding = EXCLUDED.embedding,
            metadata = EXCLUDED.metadata
        """
        
        await self.db.execute(text(insert_sql), batch)
        await self.db.commit()
    
    async def _decompress_file(
        self,
        file_path: Path,
        compression: CompressionType
    ) -> Path:
        """解压文件"""
        decompressed_path = file_path.with_suffix('')
        
        if compression == CompressionType.GZIP:
            with gzip.open(file_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif compression == CompressionType.BZIP2:
            import bz2
            with bz2.open(file_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif compression == CompressionType.LZ4:
            import lz4.frame
            with lz4.frame.open(file_path, 'rb') as f_in:
                with open(decompressed_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif compression == CompressionType.ZSTD:
            import zstandard
            with open(file_path, 'rb') as f_in:
                dctx = zstandard.ZstdDecompressor()
                with open(decompressed_path, 'wb') as f_out:
                    dctx.copy_stream(f_in, f_out)
        
        return decompressed_path

class VectorDataExporter:
    """向量数据导出器"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.export_stats = {
            "total_vectors": 0,
            "exported": 0,
            "failed": 0
        }
        
    async def export_to_file(
        self,
        table_name: str,
        output_path: str,
        config: ExportConfig,
        filter_condition: Optional[str] = None
    ) -> Dict[str, Any]:
        """导出向量数据到文件"""
        try:
            output_path = Path(output_path)
            
            # 根据格式选择导出方法
            if config.format == DataFormat.CSV:
                files = await self._export_csv(table_name, output_path, config, filter_condition)
            elif config.format == DataFormat.JSON:
                files = await self._export_json(table_name, output_path, config, filter_condition)
            elif config.format == DataFormat.JSONL:
                files = await self._export_jsonl(table_name, output_path, config, filter_condition)
            elif config.format == DataFormat.PARQUET:
                files = await self._export_parquet(table_name, output_path, config, filter_condition)
            elif config.format == DataFormat.HDF5:
                files = await self._export_hdf5(table_name, output_path, config, filter_condition)
            elif config.format == DataFormat.NUMPY:
                files = await self._export_numpy(table_name, output_path, config, filter_condition)
            elif config.format == DataFormat.BINARY:
                files = await self._export_binary(table_name, output_path, config, filter_condition)
            else:
                raise ValueError(f"不支持的导出格式: {config.format}")
            
            # 压缩文件（如需要）
            if config.compression != CompressionType.NONE:
                files = await self._compress_files(files, config.compression)
            
            self.export_stats["files"] = files
            logger.info(f"导出完成: {self.export_stats}")
            return self.export_stats
            
        except Exception as e:
            logger.error(f"导出失败: {e}")
            raise
    
    async def _export_csv(
        self,
        table_name: str,
        output_path: Path,
        config: ExportConfig,
        filter_condition: Optional[str]
    ) -> List[Path]:
        """导出为CSV格式"""
        files = []
        file_index = 0
        current_file = None
        writer = None
        current_size = 0
        
        async for batch in self._fetch_batches(table_name, config.batch_size, filter_condition):
            for row in batch:
                try:
                    # 检查是否需要新文件
                    if current_file is None or (
                        config.split_files and 
                        current_size >= config.max_file_size_mb * 1024 * 1024
                    ):
                        if current_file:
                            current_file.close()
                        
                        file_name = f"{output_path.stem}_{file_index}.csv" if config.split_files else f"{output_path.stem}.csv"
                        current_file = open(output_path.parent / file_name, 'w', newline='')
                        writer = csv.DictWriter(current_file, fieldnames=['id', 'embedding', 'metadata'])
                        writer.writeheader()
                        files.append(Path(current_file.name))
                        file_index += 1
                        current_size = 0
                    
                    # 写入数据
                    writer.writerow({
                        'id': row['id'],
                        'embedding': json.dumps(row['embedding']),
                        'metadata': json.dumps(row.get('metadata', {}))
                    })
                    
                    current_size = current_file.tell()
                    self.export_stats["exported"] += 1
                    
                except Exception as e:
                    self.export_stats["failed"] += 1
                    logger.warning(f"导出行失败: {e}")
        
        if current_file:
            current_file.close()
        
        return files
    
    async def _export_json(
        self,
        table_name: str,
        output_path: Path,
        config: ExportConfig,
        filter_condition: Optional[str]
    ) -> List[Path]:
        """导出为JSON格式"""
        all_data = []
        
        async for batch in self._fetch_batches(table_name, config.batch_size, filter_condition):
            for row in batch:
                try:
                    item = {
                        'id': row['id'],
                        'embedding': row['embedding']
                    }
                    
                    if config.include_metadata:
                        item['metadata'] = row.get('metadata', {})
                    
                    all_data.append(item)
                    self.export_stats["exported"] += 1
                    
                except Exception as e:
                    self.export_stats["failed"] += 1
                    logger.warning(f"导出行失败: {e}")
        
        # 写入文件
        with open(output_path, 'w') as f:
            json.dump({'vectors': all_data}, f, indent=2)
        
        return [output_path]
    
    async def _export_hdf5(
        self,
        table_name: str,
        output_path: Path,
        config: ExportConfig,
        filter_condition: Optional[str]
    ) -> List[Path]:
        """导出为HDF5格式"""
        vectors = []
        ids = []
        metadata = []
        
        async for batch in self._fetch_batches(table_name, config.batch_size, filter_condition):
            for row in batch:
                try:
                    vectors.append(row['embedding'])
                    ids.append(row['id'])
                    
                    if config.include_metadata:
                        metadata.append(row.get('metadata', {}))
                    
                    self.export_stats["exported"] += 1
                    
                except Exception as e:
                    self.export_stats["failed"] += 1
                    logger.warning(f"导出行失败: {e}")
        
        # 写入HDF5文件
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('vectors', data=np.array(vectors))
            # 转换ids为字节字符串以兼容HDF5
            ids_bytes = [str(id).encode('utf-8') for id in ids]
            f.create_dataset('ids', data=np.array(ids_bytes, dtype='S'))
            
            if config.include_metadata:
                # HDF5不直接支持字典，需要序列化
                metadata_str = [json.dumps(m) for m in metadata]
                f.create_dataset('metadata', data=metadata_str)
        
        return [output_path]
    
    async def _fetch_batches(
        self,
        table_name: str,
        batch_size: int,
        filter_condition: Optional[str]
    ):
        """批量获取数据"""
        offset = 0
        
        while True:
            query = f"""
            SELECT id, embedding, metadata
            FROM {table_name}
            {f"WHERE {filter_condition}" if filter_condition else ""}
            ORDER BY id
            LIMIT :limit OFFSET :offset
            """
            
            result = await self.db.execute(
                text(query),
                {"limit": batch_size, "offset": offset}
            )
            
            batch = result.fetchall()
            if not batch:
                break
            
            yield [dict(row._mapping) for row in batch]
            offset += batch_size
    
    async def _compress_files(
        self,
        files: List[Path],
        compression: CompressionType
    ) -> List[Path]:
        """压缩文件"""
        compressed_files = []
        
        for file_path in files:
            if compression == CompressionType.GZIP:
                compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
                with open(file_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                compressed_path = file_path
            
            compressed_files.append(compressed_path)
            
            # 删除原文件
            if compressed_path != file_path:
                file_path.unlink()
        
        return compressed_files

class VectorDataMigrator:
    """向量数据迁移器"""
    
    def __init__(self, source_db: AsyncSession, target_db: AsyncSession):
        self.source_db = source_db
        self.target_db = target_db
        self.migration_stats = {
            "total_vectors": 0,
            "migrated": 0,
            "failed": 0,
            "verified": 0
        }
        
    async def migrate_table(
        self,
        source_table: str,
        target_table: str,
        config: MigrationConfig
    ) -> Dict[str, Any]:
        """迁移表数据"""
        try:
            # 获取源表记录数
            count_query = f"SELECT COUNT(*) FROM {source_table}"
            result = await self.source_db.execute(text(count_query))
            total_count = result.scalar()
            self.migration_stats["total_vectors"] = total_count
            
            # 创建目标表（如果不存在）
            await self._create_target_table(target_table)
            
            # 暂时只支持顺序迁移
            await self._sequential_migrate(
                source_table, target_table, config
            )
            
            # 验证迁移
            if config.verify_after:
                await self._verify_migration(source_table, target_table)
            
            logger.info(f"迁移完成: {self.migration_stats}")
            return self.migration_stats
            
        except Exception as e:
            logger.error(f"迁移失败: {e}")
            raise
    
    async def _create_target_table(self, table_name: str) -> None:
        """创建目标表"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id VARCHAR(255) PRIMARY KEY,
            embedding vector(1024),
            metadata JSONB DEFAULT '{{}}',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        )
        """
        
        await self.target_db.execute(text(create_table_sql))
        await self.target_db.commit()
    
    async def _sequential_migrate(
        self,
        source_table: str,
        target_table: str,
        config: MigrationConfig
    ) -> None:
        """顺序迁移"""
        offset = 0
        
        while True:
            # 读取批次
            query = f"""
            SELECT id, embedding, metadata
            FROM {source_table}
            ORDER BY id
            LIMIT :limit OFFSET :offset
            """
            
            result = await self.source_db.execute(
                text(query),
                {"limit": config.batch_size, "offset": offset}
            )
            
            batch = result.fetchall()
            if not batch:
                break
            
            # 写入目标
            for row in batch:
                try:
                    insert_sql = f"""
                    INSERT INTO {target_table} (id, embedding, metadata)
                    VALUES (:id, :embedding, :metadata)
                    ON CONFLICT (id) DO UPDATE
                    SET embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """
                    
                    await self.target_db.execute(
                        text(insert_sql),
                        {
                            "id": row.id,
                            "embedding": row.embedding,
                            "metadata": row.metadata or {}
                        }
                    )
                    
                    self.migration_stats["migrated"] += 1
                    
                except Exception as e:
                    self.migration_stats["failed"] += 1
                    logger.warning(f"迁移记录失败: {e}")
            
            await self.target_db.commit()
            offset += config.batch_size
    
    async def _verify_migration(
        self,
        source_table: str,
        target_table: str,
        sample_size: int = 100
    ) -> None:
        """验证迁移结果"""
        # 随机抽样验证
        sample_query = f"""
        SELECT id, embedding
        FROM {source_table}
        ORDER BY RANDOM()
        LIMIT :limit
        """
        
        source_result = await self.source_db.execute(
            text(sample_query),
            {"limit": sample_size}
        )
        
        for row in source_result:
            # 在目标表中查找
            target_query = f"""
            SELECT embedding
            FROM {target_table}
            WHERE id = :id
            """
            
            target_result = await self.target_db.execute(
                text(target_query),
                {"id": row.id}
            )
            
            target_row = target_result.fetchone()
            
            if target_row and np.array_equal(row.embedding, target_row.embedding):
                self.migration_stats["verified"] += 1

class VectorBackupRestore:
    """向量备份恢复工具"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        
    async def create_backup(
        self,
        table_name: str,
        backup_path: str,
        include_indexes: bool = True
    ) -> Dict[str, Any]:
        """创建备份"""
        backup_path = Path(backup_path)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # 导出数据
        exporter = VectorDataExporter(self.db)
        data_file = backup_path / f"{table_name}_data.hdf5"
        
        await exporter.export_to_file(
            table_name,
            str(data_file),
            ExportConfig(format=DataFormat.HDF5)
        )
        
        # 备份索引信息
        if include_indexes:
            index_info = await self._get_index_info(table_name)
            index_file = backup_path / f"{table_name}_indexes.json"
            
            with open(index_file, 'w') as f:
                json.dump(index_info, f, indent=2)
        
        # 创建元信息文件
        meta_file = backup_path / "backup_meta.json"
        meta_info = {
            "table_name": table_name,
            "backup_time": utc_now().isoformat(),
            "data_file": data_file.name,
            "index_file": f"{table_name}_indexes.json" if include_indexes else None,
            "version": "1.0"
        }
        
        with open(meta_file, 'w') as f:
            json.dump(meta_info, f, indent=2)
        
        return {
            "status": "success",
            "backup_path": str(backup_path),
            "files": [str(f) for f in backup_path.glob("*")]
        }
    
    async def restore_backup(
        self,
        backup_path: str,
        target_table: Optional[str] = None
    ) -> Dict[str, Any]:
        """恢复备份"""
        backup_path = Path(backup_path)
        
        # 读取元信息
        meta_file = backup_path / "backup_meta.json"
        with open(meta_file, 'r') as f:
            meta_info = json.load(f)
        
        table_name = target_table or meta_info["table_name"]
        
        # 恢复数据
        importer = VectorDataImporter(self.db)
        data_file = backup_path / meta_info["data_file"]
        
        await importer.import_from_file(
            str(data_file),
            table_name,
            ImportConfig(format=DataFormat.HDF5)
        )
        
        # 恢复索引
        if meta_info.get("index_file"):
            index_file = backup_path / meta_info["index_file"]
            await self._restore_indexes(table_name, index_file)
        
        return {
            "status": "success",
            "table_name": table_name,
            "restored_from": str(backup_path)
        }
    
    async def _get_index_info(self, table_name: str) -> List[Dict]:
        """获取索引信息"""
        query = """
        SELECT 
            indexname,
            indexdef
        FROM pg_indexes
        WHERE tablename = :table_name
        """
        
        result = await self.db.execute(
            text(query),
            {"table_name": table_name}
        )
        
        indexes = []
        for row in result:
            indexes.append({
                "name": row.indexname,
                "definition": row.indexdef
            })
        
        return indexes
    
    async def _restore_indexes(self, table_name: str, index_file: Path) -> None:
        """恢复索引"""
        with open(index_file, 'r') as f:
            indexes = json.load(f)
        
        for index in indexes:
            try:
                # 修改索引定义中的表名
                definition = index["definition"].replace(
                    f"ON {index['name'].split('_')[0]}",
                    f"ON {table_name}"
                )
                
                await self.db.execute(text(definition))
                await self.db.commit()
                
                logger.info(f"索引 {index['name']} 恢复成功")
                
            except Exception as e:
                logger.warning(f"索引恢复失败: {e}")
