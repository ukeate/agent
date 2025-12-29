"""多模态RAG问答链"""

import time
from typing import Optional, Dict, Any, AsyncGenerator
from src.core.utils.timezone_utils import utc_now
from .multimodal_config import (
    MultimodalConfig,
    QAResponse,
    RetrievalResults
)
from .multimodal_vectorstore import MultimodalVectorStore
from .multimodal_query_analyzer import MultimodalQueryAnalyzer
from .retrieval_strategy import SmartRetrievalStrategy

logger = get_logger(__name__)

class MultimodalQAChain:
    """多模态RAG问答链"""
    
    def __init__(
        self,
        config: MultimodalConfig,
        retriever: Optional[SmartRetrievalStrategy] = None,
        vector_store: Optional[MultimodalVectorStore] = None,
    ):
        """初始化问答链
        
        Args:
            config: 多模态配置
            retriever: 检索器（可选）
        """
        self.config = config
        
        # 初始化组件
        self.vector_store = vector_store or MultimodalVectorStore(config)
        self.query_analyzer = MultimodalQueryAnalyzer()
        self.retriever = retriever or SmartRetrievalStrategy(
            config, self.vector_store, self.query_analyzer
        )
        
        # 缓存
        self._query_cache = {}
        self._cache_max_size = 100
        self._cache_hits = 0
        self._cache_misses = 0
    
    async def arun(
        self,
        query: str,
        stream: bool = False,
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> QAResponse:
        """异步运行问答链
        
        Args:
            query: 用户查询
            stream: 是否流式响应
            max_tokens: 最大token数
            temperature: 生成温度
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            问答响应
        """
        start_time = time.time()
        
        # 检查缓存
        cache_key = self._get_cache_key(query, kwargs)
        if self.config.cache_enabled and cache_key in self._query_cache:
            cached_response = self._query_cache[cache_key]
            if self._is_cache_valid(cached_response):
                logger.info(f"Cache hit for query: {query[:50]}...")
                self._cache_hits += 1
                return QAResponse(**cached_response)
            self._cache_misses += 1
        elif self.config.cache_enabled:
            self._cache_misses += 1

        query_context = self.query_analyzer.analyze_query(
            query=query,
            files=kwargs.get("context_files", [])
        )

        retrieval_results = await self.retriever.retrieve(query_context)
        answer, confidence = self._generate_answer(query, retrieval_results)

        processing_time = time.time() - start_time
        qa_response = QAResponse(
            answer=answer,
            sources=retrieval_results.sources,
            confidence=confidence,
            processing_time=processing_time,
            context_used={
                "text_chunks": len(retrieval_results.texts),
                "images": len(retrieval_results.images),
                "tables": len(retrieval_results.tables),
            },
        )

        if self.config.cache_enabled:
            self._update_cache(cache_key, qa_response.model_dump())

        return qa_response

    def _generate_answer(self, query: str, retrieval_results: RetrievalResults) -> tuple[str, float]:
        items: list[tuple[str, float]] = []
        seen = set()

        for it in (retrieval_results.texts or [])[:3]:
            content = str(it.get("content") or "").strip()
            score = it.get("score")
            try:
                score_f = float(score) if score is not None else 0.0
            except Exception:
                score_f = 0.0
            if content and content not in seen:
                seen.add(content)
                items.append((content, score_f))

        if not items:
            return "未找到与问题相关的内容。", 0.0

        answer = "\n\n".join([c for c, _ in items])
        confidence = max(0.0, min(1.0, sum(s for _, s in items) / len(items)))
        return answer, round(confidence, 4)
    
    async def stream_response(
        self,
        query: str,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """流式生成响应
        
        Args:
            query: 查询
            **kwargs: 其他参数
            
        Yields:
            响应文本块
        """
        # 分析查询
        query_context = self.query_analyzer.analyze_query(
            query=query,
            files=kwargs.get("context_files", [])
        )
        
        # 检索相关内容
        retrieval_results = await self.retriever.retrieve(query_context)

        answer, _ = self._generate_answer(query, retrieval_results)
        for i in range(0, len(answer), 64):
            yield answer[i : i + 64]
    
    def _get_cache_key(self, query: str, kwargs: Dict) -> str:
        """生成缓存键
        
        Args:
            query: 查询
            kwargs: 其他参数
            
        Returns:
            缓存键
        """
        import hashlib
        
        # 包含查询和关键参数
        key_parts = [
            query,
            str(kwargs.get("include_images", True)),
            str(kwargs.get("include_tables", True)),
            str(kwargs.get("context_files", []))
        ]
        
        key_str = "|".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_cache_valid(self, cached_response: Dict) -> bool:
        """检查缓存是否有效
        
        Args:
            cached_response: 缓存的响应
            
        Returns:
            是否有效
        """
        if "timestamp" not in cached_response:
            return False
        
        # 检查缓存时间
        cache_time = cached_response["timestamp"]
        current_time = utc_now().timestamp()
        
        # 缓存有效期
        if current_time - cache_time > self.config.cache_ttl_seconds:
            return False
        
        return True
    
    def _update_cache(self, key: str, response: Dict):
        """更新缓存
        
        Args:
            key: 缓存键
            response: 响应数据
        """
        # 添加时间戳
        response["timestamp"] = utc_now().timestamp()
        
        # 限制缓存大小
        if len(self._query_cache) >= self._cache_max_size:
            # 移除最旧的缓存项
            oldest_key = min(
                self._query_cache.keys(),
                key=lambda k: self._query_cache[k].get("timestamp", 0)
            )
            del self._query_cache[oldest_key]
        
        self._query_cache[key] = response
    
    def clear_cache(self):
        """清空缓存"""
        self._query_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Query cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        total = self._cache_hits + self._cache_misses
        return {
            "enabled": bool(self.config.cache_enabled),
            "size": len(self._query_cache),
            "hits": int(self._cache_hits),
            "misses": int(self._cache_misses),
            "hit_rate": float(self._cache_hits / total) if total else 0.0,
        }
from src.core.logging import get_logger
