"""OpenAI Embeddings服务实现"""

import hashlib
from typing import Dict, List, Optional
import numpy as np
from openai import AsyncOpenAI
from src.core.config import get_settings
from src.core.redis import get_redis
from src.core.logging import get_logger

logger = get_logger(__name__)

settings = get_settings()

class EmbeddingService:
    """OpenAI嵌入服务"""

    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = "text-embedding-ada-002"
        self.dimension = 1536
        self.cache_ttl = 86400  # 24小时缓存

    def _get_cache_key(self, text: str) -> str:
        """生成缓存键"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{self.model}:{text_hash}"

    async def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """从缓存获取嵌入向量"""
        try:
            # 检查是否为测试模式
            if settings.TESTING:
                return None
            redis = get_redis()  # get_redis()不是异步函数
            cache_key = self._get_cache_key(text)
            cached = await redis.get(cache_key)
            if cached:
                import json
                return json.loads(cached)
        except Exception as e:
            logger.warning(f"Failed to get cached embedding: {e}")
        return None

    async def _cache_embedding(self, text: str, embedding: List[float]):
        """缓存嵌入向量"""
        try:
            # 检查是否为测试模式
            if settings.TESTING:
                return
            redis = get_redis()  # get_redis()不是异步函数
            cache_key = self._get_cache_key(text)
            import json
            await redis.setex(
                cache_key,
                self.cache_ttl,
                json.dumps(embedding)
            )
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")

    async def embed_text(self, text: str) -> List[float]:
        """生成单个文本的嵌入向量"""
        # 检查缓存
        cached = await self._get_cached_embedding(text)
        if cached:
            logger.debug(f"Using cached embedding for text of length {len(text)}")
            return cached

        try:
            # 调用OpenAI API
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            embedding = response.data[0].embedding
            
            # 缓存结果
            await self._cache_embedding(text, embedding)
            
            return embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise

    async def embed_batch(self, texts: List[str], batch_size: int = 20) -> List[List[float]]:
        """批量生成文本嵌入向量"""
        embeddings = []
        
        # 检查输入列表是否为空
        if not texts:
            return embeddings
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []
            
            # 先检查缓存
            uncached_texts = []
            uncached_indices = []
            
            for j, text in enumerate(batch):
                cached = await self._get_cached_embedding(text)
                if cached:
                    batch_embeddings.append((j, cached))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(j)
            
            # 处理未缓存的文本
            if uncached_texts:
                try:
                    response = await self.client.embeddings.create(
                        model=self.model,
                        input=uncached_texts,
                    )
                    
                    for idx, data in zip(uncached_indices, response.data):
                        embedding = data.embedding
                        batch_embeddings.append((idx, embedding))
                        # 缓存新生成的嵌入
                        await self._cache_embedding(batch[idx], embedding)
                        
                except Exception as e:
                    logger.error(f"Failed to generate batch embeddings: {e}")
                    raise
            
            # 按原顺序排列
            batch_embeddings.sort(key=lambda x: x[0])
            embeddings.extend([emb for _, emb in batch_embeddings])
            
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

class TextChunker:
    """文本分块器"""

    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str) -> List[Dict[str, any]]:
        """将文本分块"""
        chunks = []
        
        # 检查文本是否为空
        if not text.strip():
            return chunks
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        current_chunk = ""
        current_size = 0
        chunk_start = 0
        
        for para in paragraphs:
            para_size = len(para)
            
            # 如果当前块加上新段落超过大小限制
            if current_size + para_size > self.chunk_size and current_chunk:
                chunks.append({
                    "content": current_chunk.strip(),
                    "start": chunk_start,
                    "end": chunk_start + len(current_chunk),
                })
                
                # 重叠处理
                if self.overlap > 0 and len(current_chunk) > self.overlap:
                    overlap_text = current_chunk[-self.overlap:]
                    current_chunk = overlap_text + "\n\n" + para
                    chunk_start = chunk_start + len(current_chunk) - len(overlap_text) - len(para) - 2
                else:
                    current_chunk = para
                    chunk_start = chunk_start + current_size
                
                current_size = len(current_chunk)
            else:
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_size = len(current_chunk)
        
        # 添加最后一个块
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "start": chunk_start,
                "end": chunk_start + len(current_chunk),
            })
        
        return chunks

    def chunk_code(self, code: str, language: str = "python") -> List[Dict[str, any]]:
        """将代码按函数/类分块"""
        chunks = []
        
        # 检查代码是否为空
        if not code.strip():
            return chunks
        
        if language == "python":
            lines = code.split('\n')
            current_chunk = []
            current_indent = 0
            chunk_start_line = 0
            
            for i, line in enumerate(lines):
                # 检测顶级定义
                if line.startswith(('def ', 'class ', 'async def ')):
                    # 保存之前的块
                    if current_chunk:
                        chunks.append({
                            "content": '\n'.join(current_chunk),
                            "start_line": chunk_start_line,
                            "end_line": i - 1,
                            "type": "function" if current_chunk[0].startswith(('def ', 'async def ')) else "class"
                        })
                    
                    # 开始新块
                    current_chunk = [line]
                    chunk_start_line = i
                    current_indent = len(line) - len(line.lstrip())
                elif current_chunk:
                    # 继续当前块
                    if line.strip() and not line.startswith(' ' * current_indent) and current_indent == 0:
                        # 顶级定义结束
                        chunks.append({
                            "content": '\n'.join(current_chunk),
                            "start_line": chunk_start_line,
                            "end_line": i - 1,
                            "type": "function" if current_chunk[0].startswith(('def ', 'async def ')) else "class"
                        })
                        current_chunk = [line] if line.strip() else []
                        chunk_start_line = i
                    else:
                        current_chunk.append(line)
                else:
                    current_chunk.append(line)
            
            # 添加最后一个块
            if current_chunk and any(line.strip() for line in current_chunk):
                chunks.append({
                    "content": '\n'.join(current_chunk),
                    "start_line": chunk_start_line,
                    "end_line": len(lines) - 1,
                    "type": "module"
                })
        else:
            # 其他语言使用简单分块
            return self.chunk_text(code)
        
        return chunks

# 全局嵌入服务实例
embedding_service = EmbeddingService()
text_chunker = TextChunker()
