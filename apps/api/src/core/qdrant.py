"""Qdrant向量数据库连接管理"""

import logging
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from .config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class QdrantManager:
    """Qdrant向量数据库管理器"""

    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.collections = {
            "documents": {
                "size": 1536,  # OpenAI embedding维度
                "distance": Distance.COSINE,
            },
            "code": {
                "size": 1536,
                "distance": Distance.COSINE,
            },
        }

    def get_client(self) -> QdrantClient:
        """获取Qdrant客户端"""
        if not self.client:
            # 使用主机和端口直接连接，绕过HTTP代理
            self.client = QdrantClient(
                host="127.0.0.1",
                port=settings.QDRANT_PORT,
                timeout=30.0,
                # 禁用grpc以强制使用直接HTTP连接
                prefer_grpc=False,
                # 设置连接参数绕过代理
                https=False,
                # 跳过版本兼容性检查
                check_compatibility=False,
            )
            logger.info(f"Connected to Qdrant at 127.0.0.1:{settings.QDRANT_PORT}")
                    
        return self.client

    async def initialize_collections(self):
        """初始化向量集合"""
        client = self.get_client()
        
        for collection_name, config in self.collections.items():
            try:
                # 检查集合是否存在
                collections = client.get_collections().collections
                exists = any(c.name == collection_name for c in collections)
                
                if not exists:
                    # 创建新集合
                    client.create_collection(
                        collection_name=collection_name,
                        vectors_config=VectorParams(
                            size=config["size"],
                            distance=config["distance"],
                        ),
                    )
                    logger.info(f"Created collection: {collection_name}")
                else:
                    logger.info(f"Collection already exists: {collection_name}")
            except Exception as e:
                logger.error(f"Failed to initialize collection {collection_name}: {e}")
                raise

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            client = self.get_client()
            # 尝试获取集合信息
            client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant health check failed: {e}")
            return False

    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            self.client = None
            logger.info("Qdrant connection closed")


# 全局Qdrant管理器实例
qdrant_manager = QdrantManager()


def get_qdrant_client() -> QdrantClient:
    """获取Qdrant客户端依赖"""
    return qdrant_manager.get_client()