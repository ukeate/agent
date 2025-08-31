"""
混合搜索引擎

结合语义向量搜索和BM25关键词搜索，实现更精准的检索
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass
import asyncio
import logging
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
import json
import re
from sqlalchemy import text, select, func
from sqlalchemy.ext.asyncio import AsyncSession
import math

logger = logging.getLogger(__name__)


class SearchMode(str, Enum):
    """搜索模式"""
    SEMANTIC = "semantic"              # 纯语义搜索
    KEYWORD = "keyword"                # 纯关键词搜索
    HYBRID = "hybrid"                  # 混合搜索
    IMAGE = "image"                    # 图像搜索
    CROSS_MODAL = "cross_modal"        # 跨模态搜索
    TEMPORAL = "temporal"              # 时序搜索


class FusionStrategy(str, Enum):
    """融合策略"""
    RRF = "rrf"                        # Reciprocal Rank Fusion
    LINEAR = "linear"                  # 线性加权
    CROSS_ENCODER = "cross_encoder"    # 交叉编码器重排序
    LEARNED = "learned"                # 学习的融合权重


@dataclass
class SearchConfig:
    """搜索配置"""
    search_mode: SearchMode = SearchMode.HYBRID
    fusion_strategy: FusionStrategy = FusionStrategy.RRF
    semantic_weight: float = 0.7       # 语义搜索权重
    keyword_weight: float = 0.3        # 关键词搜索权重
    top_k: int = 10
    rerank_top_n: int = 50            # 重排序候选数量
    enable_query_expansion: bool = True
    enable_synonyms: bool = True
    min_relevance_score: float = 0.5


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    content: str
    metadata: Dict[str, Any]
    semantic_score: float
    keyword_score: float
    final_score: float
    distance: float
    highlights: List[str] = None


class HybridSearchEngine:
    """混合搜索引擎"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.search_stats = {
            "total_searches": 0,
            "semantic_searches": 0,
            "keyword_searches": 0,
            "hybrid_searches": 0,
            "avg_latency_ms": 0.0
        }
        
    async def hybrid_search(
        self,
        query: str,
        query_vector: Optional[np.ndarray] = None,
        config: Optional[SearchConfig] = None
    ) -> List[SearchResult]:
        """执行混合搜索"""
        start_time = asyncio.get_event_loop().time()
        
        if config is None:
            config = SearchConfig()
        
        try:
            # 根据搜索模式执行
            if config.search_mode == SearchMode.SEMANTIC:
                if query_vector is None:
                    raise ValueError("语义搜索需要提供查询向量")
                results = await self._semantic_search(query_vector, config)
                
            elif config.search_mode == SearchMode.KEYWORD:
                results = await self._keyword_search(query, config)
                
            elif config.search_mode == SearchMode.HYBRID:
                results = await self._hybrid_search_internal(
                    query, query_vector, config
                )
            else:
                raise ValueError(f"不支持的搜索模式: {config.search_mode}")
            
            # 更新统计
            self._update_stats(config.search_mode, start_time)
            
            return results
            
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []
    
    async def _semantic_search(
        self,
        query_vector: np.ndarray,
        config: SearchConfig
    ) -> List[SearchResult]:
        """纯语义向量搜索"""
        try:
            # 使用pgvector的向量搜索
            search_sql = """
            SELECT 
                id,
                content,
                metadata,
                embedding <=> %s::vector AS distance
            FROM knowledge_items
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
            
            vector_list = query_vector.tolist()
            result = await self.db.execute(
                text(search_sql),
                (vector_list, vector_list, config.rerank_top_n)
            )
            
            results = []
            for row in result.fetchall():
                # 计算语义相似度分数（1 - cosine_distance）
                semantic_score = 1.0 - float(row.distance)
                
                results.append(SearchResult(
                    id=str(row.id),
                    content=row.content,
                    metadata=row.metadata or {},
                    semantic_score=semantic_score,
                    keyword_score=0.0,
                    final_score=semantic_score,
                    distance=float(row.distance)
                ))
            
            return results[:config.top_k]
            
        except Exception as e:
            logger.error(f"语义搜索失败: {e}")
            return []
    
    async def _keyword_search(
        self,
        query: str,
        config: SearchConfig
    ) -> List[SearchResult]:
        """BM25关键词搜索"""
        try:
            # 预处理查询
            processed_query = await self._preprocess_query(query, config)
            
            # 使用PostgreSQL全文搜索
            search_sql = """
            WITH query_terms AS (
                SELECT plainto_tsquery('english', %s) AS query
            )
            SELECT 
                k.id,
                k.content,
                k.metadata,
                ts_rank_cd(
                    to_tsvector('english', k.content),
                    q.query
                ) AS rank
            FROM knowledge_items k, query_terms q
            WHERE to_tsvector('english', k.content) @@ q.query
            ORDER BY rank DESC
            LIMIT %s
            """
            
            result = await self.db.execute(
                text(search_sql),
                (processed_query, config.rerank_top_n)
            )
            
            results = []
            max_rank = 0.0
            
            rows = result.fetchall()
            if rows:
                max_rank = max(float(row.rank) for row in rows)
            
            for row in rows:
                # 归一化BM25分数
                keyword_score = float(row.rank) / max_rank if max_rank > 0 else 0
                
                # 提取高亮片段
                highlights = await self._extract_highlights(
                    row.content, processed_query
                )
                
                results.append(SearchResult(
                    id=str(row.id),
                    content=row.content,
                    metadata=row.metadata or {},
                    semantic_score=0.0,
                    keyword_score=keyword_score,
                    final_score=keyword_score,
                    distance=1.0,
                    highlights=highlights
                ))
            
            return results[:config.top_k]
            
        except Exception as e:
            logger.error(f"关键词搜索失败: {e}")
            return []
    
    async def _hybrid_search_internal(
        self,
        query: str,
        query_vector: Optional[np.ndarray],
        config: SearchConfig
    ) -> List[SearchResult]:
        """内部混合搜索实现"""
        # 并行执行语义搜索和关键词搜索
        tasks = []
        
        if query_vector is not None:
            tasks.append(self._semantic_search(query_vector, config))
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))  # 占位
            
        tasks.append(self._keyword_search(query, config))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        semantic_results = results[0] if not isinstance(results[0], Exception) and query_vector is not None else []
        keyword_results = results[1] if not isinstance(results[1], Exception) else []
        
        # 融合结果
        fused_results = await self._fuse_results(
            semantic_results,
            keyword_results,
            config
        )
        
        # 重排序（如果配置了）
        if config.fusion_strategy == FusionStrategy.CROSS_ENCODER:
            fused_results = await self._rerank_with_cross_encoder(
                query, fused_results, config
            )
        
        return fused_results[:config.top_k]
    
    async def _fuse_results(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        config: SearchConfig
    ) -> List[SearchResult]:
        """融合搜索结果"""
        if config.fusion_strategy == FusionStrategy.RRF:
            return await self._rrf_fusion(semantic_results, keyword_results, config)
        elif config.fusion_strategy == FusionStrategy.LINEAR:
            return await self._linear_fusion(semantic_results, keyword_results, config)
        else:
            # 默认使用RRF
            return await self._rrf_fusion(semantic_results, keyword_results, config)
    
    async def _rrf_fusion(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        config: SearchConfig,
        k: int = 60
    ) -> List[SearchResult]:
        """Reciprocal Rank Fusion融合"""
        fusion_scores = {}
        result_map = {}
        
        # 计算语义搜索的RRF分数
        for i, result in enumerate(semantic_results):
            rrf_score = 1.0 / (k + i + 1)
            fusion_scores[result.id] = config.semantic_weight * rrf_score
            result_map[result.id] = result
        
        # 计算关键词搜索的RRF分数
        for i, result in enumerate(keyword_results):
            rrf_score = 1.0 / (k + i + 1)
            if result.id in fusion_scores:
                fusion_scores[result.id] += config.keyword_weight * rrf_score
                # 更新结果的关键词相关信息
                result_map[result.id].keyword_score = result.keyword_score
                result_map[result.id].highlights = result.highlights
            else:
                fusion_scores[result.id] = config.keyword_weight * rrf_score
                result_map[result.id] = result
        
        # 更新最终分数并排序
        fused_results = []
        for doc_id, final_score in fusion_scores.items():
            result = result_map[doc_id]
            result.final_score = final_score
            fused_results.append(result)
        
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        return fused_results
    
    async def _linear_fusion(
        self,
        semantic_results: List[SearchResult],
        keyword_results: List[SearchResult],
        config: SearchConfig
    ) -> List[SearchResult]:
        """线性加权融合"""
        fusion_scores = {}
        result_map = {}
        
        # 收集所有结果
        for result in semantic_results:
            result_map[result.id] = result
            fusion_scores[result.id] = config.semantic_weight * result.semantic_score
        
        for result in keyword_results:
            if result.id in fusion_scores:
                fusion_scores[result.id] += config.keyword_weight * result.keyword_score
                result_map[result.id].keyword_score = result.keyword_score
                result_map[result.id].highlights = result.highlights
            else:
                fusion_scores[result.id] = config.keyword_weight * result.keyword_score
                result_map[result.id] = result
        
        # 更新最终分数并排序
        fused_results = []
        for doc_id, final_score in fusion_scores.items():
            result = result_map[doc_id]
            result.final_score = final_score
            if final_score >= config.min_relevance_score:
                fused_results.append(result)
        
        fused_results.sort(key=lambda x: x.final_score, reverse=True)
        return fused_results
    
    async def _rerank_with_cross_encoder(
        self,
        query: str,
        results: List[SearchResult],
        config: SearchConfig
    ) -> List[SearchResult]:
        """使用交叉编码器重排序"""
        # 这里是一个简化的实现
        # 实际应用中应该使用真正的交叉编码器模型
        for result in results:
            # 模拟交叉编码器评分
            query_terms = set(query.lower().split())
            content_terms = set(result.content.lower().split())
            overlap = len(query_terms & content_terms)
            
            # 结合原始分数和交叉编码器分数
            cross_encoder_score = overlap / (len(query_terms) + 1)
            result.final_score = 0.5 * result.final_score + 0.5 * cross_encoder_score
        
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results
    
    async def _preprocess_query(
        self,
        query: str,
        config: SearchConfig
    ) -> str:
        """预处理查询"""
        processed = query.lower().strip()
        
        # 查询扩展
        if config.enable_query_expansion:
            processed = await self._expand_query(processed)
        
        # 同义词处理
        if config.enable_synonyms:
            processed = await self._apply_synonyms(processed)
        
        return processed
    
    async def _expand_query(self, query: str) -> str:
        """查询扩展"""
        # 简单的查询扩展示例
        expansions = {
            "ml": "machine learning",
            "ai": "artificial intelligence",
            "nlp": "natural language processing",
            "db": "database",
            "api": "application programming interface"
        }
        
        words = query.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            if word in expansions:
                expanded_words.append(expansions[word])
        
        return " ".join(expanded_words)
    
    async def _apply_synonyms(self, query: str) -> str:
        """应用同义词"""
        # 简单的同义词替换示例
        synonyms = {
            "search": "find retrieve query",
            "vector": "embedding representation",
            "similar": "related relevant like",
            "fast": "quick rapid speedy"
        }
        
        words = query.split()
        expanded_words = []
        
        for word in words:
            expanded_words.append(word)
            if word in synonyms:
                expanded_words.extend(synonyms[word].split())
        
        return " ".join(expanded_words)
    
    async def _extract_highlights(
        self,
        content: str,
        query: str,
        max_highlights: int = 3,
        context_length: int = 50
    ) -> List[str]:
        """提取高亮片段"""
        highlights = []
        query_terms = query.lower().split()
        content_lower = content.lower()
        
        for term in query_terms:
            pos = content_lower.find(term)
            if pos != -1:
                # 提取上下文
                start = max(0, pos - context_length)
                end = min(len(content), pos + len(term) + context_length)
                
                highlight = content[start:end]
                if start > 0:
                    highlight = "..." + highlight
                if end < len(content):
                    highlight = highlight + "..."
                
                # 高亮关键词
                highlight = highlight.replace(
                    term,
                    f"<mark>{term}</mark>"
                )
                
                highlights.append(highlight)
                
                if len(highlights) >= max_highlights:
                    break
        
        return highlights
    
    def _update_stats(self, mode: SearchMode, start_time: float) -> None:
        """更新搜索统计"""
        self.search_stats["total_searches"] += 1
        
        if mode == SearchMode.SEMANTIC:
            self.search_stats["semantic_searches"] += 1
        elif mode == SearchMode.KEYWORD:
            self.search_stats["keyword_searches"] += 1
        elif mode == SearchMode.HYBRID:
            self.search_stats["hybrid_searches"] += 1
        
        # 更新平均延迟
        end_time = asyncio.get_event_loop().time()
        latency_ms = (end_time - start_time) * 1000
        
        n = self.search_stats["total_searches"]
        current_avg = self.search_stats["avg_latency_ms"]
        self.search_stats["avg_latency_ms"] = (
            (current_avg * (n - 1) + latency_ms) / n
        )
    
    async def get_search_stats(self) -> Dict[str, Any]:
        """获取搜索统计"""
        return self.search_stats.copy()
    
    async def create_full_text_index(self, table_name: str, text_column: str) -> bool:
        """创建全文搜索索引"""
        try:
            # 添加tsvector列（如果不存在）
            add_column_sql = f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_name = '{table_name}' 
                    AND column_name = 'textsearch'
                ) THEN
                    ALTER TABLE {table_name} 
                    ADD COLUMN textsearch tsvector
                    GENERATED ALWAYS AS (to_tsvector('english', {text_column})) STORED;
                END IF;
            END $$;
            """
            await self.db.execute(text(add_column_sql))
            
            # 创建GIN索引
            index_sql = f"""
            CREATE INDEX IF NOT EXISTS idx_{table_name}_textsearch
            ON {table_name} USING GIN (textsearch);
            """
            await self.db.execute(text(index_sql))
            
            await self.db.commit()
            logger.info(f"全文搜索索引创建成功: {table_name}.{text_column}")
            return True
            
        except Exception as e:
            logger.error(f"创建全文搜索索引失败: {e}")
            await self.db.rollback()
            return False