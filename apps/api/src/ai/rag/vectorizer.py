"""文件内容向量化处理器"""

import hashlib
import os
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from pathlib import Path
from typing import Dict, List, Optional
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue
from src.ai.rag.embeddings import embedding_service, text_chunker
from functools import partial
from src.core.qdrant import get_qdrant_client
from src.core.utils.async_utils import run_sync_io

logger = get_logger(__name__)

class FileVectorizer:
    """文件向量化处理器"""

    SUPPORTED_EXTENSIONS = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".md": "markdown",
        ".txt": "text",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
    }

    def __init__(self):
        self.embedding_service = embedding_service
        self.chunker = text_chunker
        self.client = None

    def _get_client(self):
        """获取有效的Qdrant客户端"""
        if self.client is None:
            self.client = get_qdrant_client()
        return self.client

    def _get_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        with open(file_path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_file_type(self, file_path: str) -> str:
        """获取文件类型"""
        ext = Path(file_path).suffix.lower()
        return self.SUPPORTED_EXTENSIONS.get(ext, "text")

    async def _check_file_indexed(self, file_path: str, file_hash: str) -> bool:
        """检查文件是否已索引"""
        try:
            # 搜索具有相同文件路径和哈希的点
            collection_name = "code" if self._get_file_type(file_path) in [
                "python", "javascript", "typescript", "java", "cpp", "c", 
                "csharp", "go", "rust"
            ] else "documents"
            
            client = self._get_client()
            result = await run_sync_io(
                partial(
                    client.scroll,
                    collection_name=collection_name,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="file_path",
                                match=MatchValue(value=file_path),
                            ),
                            FieldCondition(
                                key="file_hash",
                                match=MatchValue(value=file_hash),
                            ),
                        ]
                    ),
                    limit=1,
                )
            )
            
            return len(result[0]) > 0
        except Exception as e:
            logger.warning(f"Failed to check if file is indexed: {e}")
            return False

    async def vectorize_file(self, file_path: str, force: bool = False) -> Dict:
        """向量化单个文件"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_type = self._get_file_type(file_path)
        if Path(file_path).suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path}")

        # 计算文件哈希
        file_hash = self._get_file_hash(file_path)
        
        # 检查是否已索引
        if not force and await self._check_file_indexed(file_path, file_hash):
            logger.info(f"File already indexed: {file_path}")
            return {"status": "skipped", "file": file_path}

        # 读取文件内容
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 检查内容是否为空或只包含空白字符
        if not content.strip():
            logger.info(f"Skipping empty file: {file_path}")
            return {"status": "skipped", "file": file_path, "reason": "empty_content"}

        # 分块处理
        if file_type in ["python", "javascript", "typescript", "java", "cpp", 
                        "c", "csharp", "go", "rust"]:
            chunks = self.chunker.chunk_code(content, file_type)
            collection_name = "code"
        else:
            chunks = self.chunker.chunk_text(content)
            collection_name = "documents"

        # 检查是否有有效的文本块
        if not chunks:
            logger.info(f"No valid chunks found in file: {file_path}")
            return {"status": "skipped", "file": file_path, "reason": "no_chunks"}

        # 生成嵌入向量
        texts = [chunk["content"] for chunk in chunks if chunk["content"].strip()]
        if not texts:
            logger.info(f"No valid content in chunks for file: {file_path}")
            return {"status": "skipped", "file": file_path, "reason": "no_valid_content"}
            
        embeddings = await self.embedding_service.embed_batch(texts)

        # 创建点数据
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = hashlib.md5(
                f"{file_path}:{i}:{file_hash}".encode()
            ).hexdigest()
            
            payload = {
                "file_path": file_path,
                "file_type": file_type,
                "file_hash": file_hash,
                "chunk_index": i,
                "content": chunk["content"],
                "created_at": utc_now().isoformat(),
            }
            
            # 添加额外的元数据
            if "start_line" in chunk:
                payload["start_line"] = chunk["start_line"]
                payload["end_line"] = chunk["end_line"]
            if "start" in chunk:
                payload["start"] = chunk["start"]
                payload["end"] = chunk["end"]
            if "type" in chunk:
                payload["chunk_type"] = chunk["type"]
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

        # 删除旧的索引（如果存在）
        if force:
            try:
                client = self._get_client()
                await run_sync_io(
                    partial(
                        client.delete,
                        collection_name=collection_name,
                        points_selector=Filter(
                            must=[
                                FieldCondition(
                                    key="file_path",
                                    match=MatchValue(value=file_path),
                                )
                            ]
                        ),
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to delete old index: {e}")

        # 上传到Qdrant
        client = self._get_client()
        await run_sync_io(
            partial(
                client.upsert,
                collection_name=collection_name,
                points=points,
            )
        )

        logger.info(
            f"Vectorized file: {file_path} ({len(chunks)} chunks, {collection_name})"
        )
        
        return {
            "status": "indexed",
            "file": file_path,
            "chunks": len(chunks),
            "collection": collection_name,
        }

    async def vectorize_directory(
        self, 
        directory: str, 
        recursive: bool = True,
        force: bool = False,
        extensions: Optional[List[str]] = None
    ) -> List[Dict]:
        """向量化目录中的所有文件"""
        results = []
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # 获取所有文件
        if recursive:
            pattern = "**/*"
        else:
            pattern = "*"
        
        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue
            
            # 检查扩展名
            if extensions:
                if file_path.suffix.lower() not in extensions:
                    continue
            elif file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            
            # 跳过隐藏文件和特殊目录
            if any(part.startswith(".") for part in file_path.parts):
                continue
            if any(part in ["__pycache__", "node_modules", "dist", "build"] 
                  for part in file_path.parts):
                continue
            
            try:
                result = await self.vectorize_file(str(file_path), force=force)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to vectorize {file_path}: {e}")
                results.append({
                    "status": "error",
                    "file": str(file_path),
                    "error": str(e),
                })
        
        return results

    async def update_index(self, file_paths: List[str]) -> List[Dict]:
        """增量更新索引"""
        results = []
        
        for file_path in file_paths:
            try:
                # 检查文件是否存在
                if not os.path.exists(file_path):
                    # 文件被删除，从索引中移除
                    await self.remove_from_index(file_path)
                    results.append({
                        "status": "removed",
                        "file": file_path,
                    })
                else:
                    # 文件存在，检查是否需要更新
                    file_hash = self._get_file_hash(file_path)
                    if not await self._check_file_indexed(file_path, file_hash):
                        result = await self.vectorize_file(file_path, force=True)
                        results.append(result)
                    else:
                        results.append({
                            "status": "unchanged",
                            "file": file_path,
                        })
            except Exception as e:
                logger.error(f"Failed to update index for {file_path}: {e}")
                results.append({
                    "status": "error",
                    "file": file_path,
                    "error": str(e),
                })
        
        return results

    async def remove_from_index(self, file_path: str):
        """从索引中移除文件"""
        # 尝试从两个集合中删除
        for collection_name in ["documents", "code"]:
            try:
                self.client.delete(
                    collection_name=collection_name,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="file_path",
                                match=MatchValue(value=file_path),
                            )
                        ]
                    ),
                )
                logger.info(f"Removed {file_path} from {collection_name}")
            except Exception as e:
                logger.warning(f"Failed to remove {file_path} from {collection_name}: {e}")

    async def get_index_stats(self) -> Dict:
        """获取索引统计信息"""
        stats = {}
        
        for collection_name in ["documents", "code"]:
            try:
                client = self._get_client()
                info = await run_sync_io(partial(client.get_collection, collection_name))
                
                # 估算存储大小：1536维 float32向量 + 元数据开销
                # 每个向量: 1536 * 4 bytes = 6,144 bytes
                # 加上元数据、索引等开销，约 7KB per point
                estimated_size = (info.points_count or 0) * 7 * 1024  # 7KB per point
                
                logger.info(f"Collection {collection_name}: points={info.points_count}, estimated_size={estimated_size}")
                
                stats[collection_name] = {
                    "vectors_count": info.vectors_count or 0,
                    "points_count": info.points_count or 0,
                    "segments_count": info.segments_count or 0,
                    "status": str(info.status) if info.status else "unknown",
                    "estimated_disk_size": estimated_size,
                }
            except Exception as e:
                logger.error(f"Failed to get stats for {collection_name}: {e}")
                stats[collection_name] = {
                    "error": str(e),
                    "vectors_count": 0,
                    "points_count": 0,
                    "segments_count": 0,
                    "status": "error",
                    "estimated_disk_size": 0,
                }
        
        return stats

# 全局向量化器实例
file_vectorizer = FileVectorizer()
from src.core.logging import get_logger
