"""Mock嵌入服务 - 用于测试和网络受限环境"""

import hashlib
import logging
import random
from typing import List
import numpy as np

logger = logging.getLogger(__name__)

class MockEmbeddingService:
    """Mock嵌入服务 - 用于测试环境"""

    def __init__(self):
        self.model = "mock-embedding-model"
        self.dimension = 1536
        random.seed(42)  # 确保可重现的结果

    def _generate_deterministic_embedding(self, text: str) -> List[float]:
        """生成基于文本内容的确定性嵌入向量"""
        # 使用文本哈希作为种子，确保相同文本产生相同向量
        text_hash = hashlib.md5(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        np.random.seed(seed)
        
        # 生成随机向量并归一化
        vector = np.random.normal(0, 1, self.dimension)
        vector = vector / np.linalg.norm(vector)
        
        return vector.tolist()

    async def embed_text(self, text: str) -> List[float]:
        """生成单个文本的Mock嵌入向量"""
        logger.debug(f"Generating mock embedding for text of length {len(text)}")
        return self._generate_deterministic_embedding(text)

    async def embed_batch(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """批量生成文本Mock嵌入向量"""
        if not texts:
            return []
        
        embeddings = []
        for text in texts:
            embeddings.append(self._generate_deterministic_embedding(text))
        
        logger.info(f"Generated mock embeddings for {len(texts)} texts")
        return embeddings

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(dot_product / (norm1 * norm2))


# Mock嵌入服务实例
mock_embedding_service = MockEmbeddingService()