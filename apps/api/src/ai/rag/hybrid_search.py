"""Qdrant BM42混合搜索引擎实现"""

import asyncio
import hashlib
import logging
import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchText,
    ScoredPoint,
    SearchRequest,
    VectorParams,
    Distance,
    CollectionInfo,
)

from src.ai.rag.embeddings import embedding_service
from src.core.config import get_settings
from src.core.qdrant import get_qdrant_client

logger = logging.getLogger(__name__)


class SearchStrategy(str, Enum):
    """搜索策略枚举"""
    VECTOR_ONLY = "vector_only"
    BM25_ONLY = "bm25_only"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_WEIGHTED = "hybrid_weighted"
    ADAPTIVE = "adaptive"


@dataclass
class HybridSearchConfig:
    """混合搜索配置"""
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    top_k: int = 20
    rerank_size: int = 100
    strategy: SearchStrategy = SearchStrategy.HYBRID_RRF
    language: str = "auto"
    enable_cache: bool = True
    cache_ttl: int = 3600
    rrf_k: int = 60


@dataclass
class ProcessedQuery:
    """预处理后的查询"""
    original: str
    text: str
    language: str
    keywords: List[str]
    expanded_terms: List[str]


@dataclass
class SearchResult:
    """搜索结果"""
    id: str
    score: float
    content: str
    file_path: str
    file_type: str
    chunk_index: int
    metadata: Dict[str, Any]
    semantic_score: float = 0.0
    keyword_score: float = 0.0
    final_score: float = 0.0
    collection: str = ""


class LanguageDetector:
    """语言检测器"""
    
    def __init__(self):
        self.chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
        self.english_pattern = re.compile(r'[a-zA-Z]')
    
    def detect(self, text: str) -> str:
        """检测文本语言"""
        chinese_chars = len(self.chinese_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        
        if chinese_chars > english_chars:
            return "zh"
        elif english_chars > 0:
            return "en"
        else:
            return "auto"


class ChineseTextAnalyzer:
    """中文文本分析器"""
    
    def segment(self, text: str) -> List[str]:
        """中文分词（简化实现）"""
        # 简化的中文分词，实际应使用jieba等专业库
        # 移除标点符号并按空格和标点分割
        text = re.sub(r'[^\u4e00-\u9fff\w\s]', ' ', text)
        segments = text.split()
        return [s for s in segments if len(s) > 1]
    
    def extract_keywords(self, segments: List[str]) -> List[str]:
        """提取关键词"""
        # 简化的关键词提取
        counter = Counter(segments)
        return [word for word, count in counter.most_common(10) if count > 1]
    
    def expand_synonyms(self, keywords: List[str]) -> List[str]:
        """扩展同义词（简化实现）"""
        # 实际应使用词典或词向量模型
        expanded = keywords.copy()
        # 简单的同义词映射
        synonym_map = {
            "实现": ["实现", "完成", "执行"],
            "函数": ["函数", "方法", "功能"],
            "问题": ["问题", "错误", "bug"],
        }
        
        for keyword in keywords:
            if keyword in synonym_map:
                expanded.extend(synonym_map[keyword])
        
        return list(set(expanded))


class EnglishTextAnalyzer:
    """英文文本分析器"""
    
    def __init__(self):
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'shall',
            'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we',
            'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """英文分词"""
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        words = text.split()
        return [w for w in words if w not in self.stop_words and len(w) > 2]
    
    def stem(self, words: List[str]) -> List[str]:
        """词干提取（简化实现）"""
        # 简化的词干提取，实际应使用NLTK或其他专业库
        stemmed = []
        for word in words:
            if word.endswith('ing'):
                stemmed.append(word[:-3])
            elif word.endswith('ed'):
                stemmed.append(word[:-2])
            elif word.endswith('s'):
                stemmed.append(word[:-1])
            else:
                stemmed.append(word)
        return stemmed


class QueryPreprocessor:
    """查询预处理器"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.chinese_analyzer = ChineseTextAnalyzer()
        self.english_analyzer = EnglishTextAnalyzer()
    
    async def process(self, query: str) -> ProcessedQuery:
        """预处理查询"""
        language = self.language_detector.detect(query)
        
        if language == "zh":
            return self._process_chinese(query)
        elif language == "en":
            return self._process_english(query)
        else:
            return self._process_mixed(query)
    
    def _process_chinese(self, query: str) -> ProcessedQuery:
        """处理中文查询"""
        segments = self.chinese_analyzer.segment(query)
        keywords = self.chinese_analyzer.extract_keywords(segments)
        expanded_terms = self.chinese_analyzer.expand_synonyms(keywords)
        
        return ProcessedQuery(
            original=query,
            text=query,
            language="zh",
            keywords=keywords,
            expanded_terms=expanded_terms
        )
    
    def _process_english(self, query: str) -> ProcessedQuery:
        """处理英文查询"""
        tokens = self.english_analyzer.tokenize(query)
        stemmed = self.english_analyzer.stem(tokens)
        
        return ProcessedQuery(
            original=query,
            text=query,
            language="en",
            keywords=tokens,
            expanded_terms=stemmed
        )
    
    def _process_mixed(self, query: str) -> ProcessedQuery:
        """处理混合语言查询"""
        # 简化处理：同时应用中英文分析
        chinese_result = self._process_chinese(query)
        english_result = self._process_english(query)
        
        return ProcessedQuery(
            original=query,
            text=query,
            language="mixed",
            keywords=chinese_result.keywords + english_result.keywords,
            expanded_terms=chinese_result.expanded_terms + english_result.expanded_terms
        )


class QdrantBM42Client:
    """增强的Qdrant客户端，支持BM42混合搜索"""
    
    def __init__(self, client: QdrantClient, config: HybridSearchConfig):
        self.client = client
        self.config = config
        self.settings = get_settings()
    
    async def hybrid_search(
        self,
        query: str,
        vector: List[float],
        collection_name: str = "documents",
        filters: Optional[Filter] = None
    ) -> List[ScoredPoint]:
        """执行混合搜索"""
        try:
            # 并行执行向量搜索和BM25搜索
            vector_results, bm25_results = await self._parallel_search(
                query, vector, collection_name, filters
            )
            
            # 根据策略融合结果
            if self.config.strategy == SearchStrategy.HYBRID_RRF:
                return self._rrf_fusion(vector_results, bm25_results)
            elif self.config.strategy == SearchStrategy.HYBRID_WEIGHTED:
                return self._weighted_fusion(vector_results, bm25_results)
            elif self.config.strategy == SearchStrategy.ADAPTIVE:
                return self._adaptive_fusion(query, vector_results, bm25_results)
            elif self.config.strategy == SearchStrategy.VECTOR_ONLY:
                return vector_results
            elif self.config.strategy == SearchStrategy.BM25_ONLY:
                return bm25_results
            else:
                return self._rrf_fusion(vector_results, bm25_results)
                
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # 降级到向量搜索
            return await self._vector_search(vector, collection_name, filters)
    
    async def _parallel_search(
        self,
        query: str,
        vector: List[float],
        collection_name: str,
        filters: Optional[Filter]
    ) -> Tuple[List[ScoredPoint], List[ScoredPoint]]:
        """并行执行向量搜索和BM25搜索"""
        try:
            vector_task = self._vector_search(vector, collection_name, filters)
            bm25_task = self._bm25_search(query, collection_name, filters)
            
            vector_results, bm25_results = await asyncio.gather(
                vector_task, bm25_task, return_exceptions=True
            )
            
            # 处理异常
            if isinstance(vector_results, Exception):
                logger.error(f"Vector search failed: {vector_results}")
                vector_results = []
            if isinstance(bm25_results, Exception):
                logger.error(f"BM25 search failed: {bm25_results}")
                bm25_results = []
            
            return vector_results, bm25_results
            
        except Exception as e:
            logger.error(f"Parallel search failed: {e}")
            # 降级到向量搜索
            vector_results = await self._vector_search(vector, collection_name, filters)
            return vector_results, []
    
    async def _vector_search(
        self,
        vector: List[float],
        collection_name: str,
        filters: Optional[Filter]
    ) -> List[ScoredPoint]:
        """执行向量搜索"""
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=vector,
                limit=self.config.rerank_size,
                score_threshold=0.5,
                query_filter=filters,
                with_payload=True,
                with_vectors=False,
            )
            return results
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _bm25_search(
        self,
        query: str,
        collection_name: str,
        filters: Optional[Filter]
    ) -> List[ScoredPoint]:
        """执行BM25搜索"""
        try:
            # 提取查询关键词
            processor = QueryPreprocessor()
            processed_query = await processor.process(query)
            
            if not processed_query.keywords:
                return []
            
            # 构建文本搜索过滤器
            text_filters = []
            for keyword in processed_query.keywords[:5]:  # 限制关键词数量
                text_filters.append(
                    FieldCondition(
                        key="content",
                        match=MatchText(text=keyword)
                    )
                )
            
            if filters:
                combined_filter = Filter(
                    must=[filters],
                    should=text_filters
                )
            else:
                combined_filter = Filter(should=text_filters)
            
            # 执行搜索
            scroll_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=combined_filter,
                limit=self.config.rerank_size,
                with_payload=True,
                with_vectors=False,
            )
            
            # 计算BM25分数
            results = []
            for point in scroll_result[0]:
                bm25_score = self._calculate_bm25_score(
                    processed_query.keywords,
                    point.payload.get("content", "")
                )
                
                # 创建ScoredPoint对象
                scored_point = ScoredPoint(
                    id=point.id,
                    score=bm25_score,
                    payload=point.payload,
                    vector=None,
                    version=getattr(point, 'version', 1)  # 添加版本字段
                )
                results.append(scored_point)
            
            # 按分数排序
            results.sort(key=lambda x: x.score, reverse=True)
            return results
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []
    
    def _calculate_bm25_score(
        self,
        query_keywords: List[str],
        doc_content: str,
        avg_doc_length: Optional[float] = None,
        k1: Optional[float] = None,
        b: Optional[float] = None
    ) -> float:
        """计算BM25分数"""
        if not doc_content or not query_keywords:
            return 0.0
        
        # 使用配置参数或默认值
        if avg_doc_length is None:
            avg_doc_length = self.settings.BM25_AVG_DOC_LENGTH
        if k1 is None:
            k1 = self.settings.BM25_K1
        if b is None:
            b = self.settings.BM25_B
        
        # 文档预处理
        doc_words = re.sub(r'[^\w\s]', ' ', doc_content.lower()).split()
        doc_length = len(doc_words)
        
        if doc_length == 0:
            return 0.0
        
        score = 0.0
        for keyword in query_keywords:
            # 计算词频
            tf = doc_words.count(keyword.lower())
            if tf == 0:
                continue
            
            # 简化的IDF（实际应基于文档集合计算）
            idf = 1.0
            
            # BM25公式
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            score += idf * (numerator / denominator)
        
        return score
    
    def _rrf_fusion(
        self,
        vector_results: List[ScoredPoint],
        bm25_results: List[ScoredPoint]
    ) -> List[ScoredPoint]:
        """RRF (Reciprocal Rank Fusion) 算法融合结果"""
        rrf_scores = {}
        
        # 处理向量搜索结果
        for i, result in enumerate(vector_results):
            rrf_scores[result.id] = {
                "score": 1.0 / (self.config.rrf_k + i + 1),
                "result": result,
                "vector_rank": i + 1,
                "bm25_rank": 0
            }
        
        # 处理BM25搜索结果
        for i, result in enumerate(bm25_results):
            if result.id in rrf_scores:
                rrf_scores[result.id]["score"] += 1.0 / (self.config.rrf_k + i + 1)
                rrf_scores[result.id]["bm25_rank"] = i + 1
            else:
                rrf_scores[result.id] = {
                    "score": 1.0 / (self.config.rrf_k + i + 1),
                    "result": result,
                    "vector_rank": 0,
                    "bm25_rank": i + 1
                }
        
        # 排序并返回
        sorted_results = sorted(
            rrf_scores.values(),
            key=lambda x: x["score"],
            reverse=True
        )
        
        # 创建融合后的结果
        final_results = []
        for item in sorted_results[:self.config.top_k]:
            result = item["result"]
            result.score = item["score"]
            final_results.append(result)
        
        return final_results
    
    def _weighted_fusion(
        self,
        vector_results: List[ScoredPoint],
        bm25_results: List[ScoredPoint]
    ) -> List[ScoredPoint]:
        """加权分数融合算法"""
        # 归一化分数
        vector_scores = self._normalize_scores([r.score for r in vector_results])
        bm25_scores = self._normalize_scores([r.score for r in bm25_results])
        
        combined_results = {}
        
        # 添加向量搜索结果
        for i, result in enumerate(vector_results):
            combined_results[result.id] = {
                "result": result,
                "vector_score": vector_scores[i] if i < len(vector_scores) else 0.0,
                "bm25_score": 0.0
            }
        
        # 融合BM25搜索结果
        for i, result in enumerate(bm25_results):
            bm25_score = bm25_scores[i] if i < len(bm25_scores) else 0.0
            
            if result.id in combined_results:
                combined_results[result.id]["bm25_score"] = bm25_score
            else:
                combined_results[result.id] = {
                    "result": result,
                    "vector_score": 0.0,
                    "bm25_score": bm25_score
                }
        
        # 计算最终分数
        final_results = []
        for item in combined_results.values():
            final_score = (
                item["vector_score"] * self.config.vector_weight +
                item["bm25_score"] * self.config.bm25_weight
            )
            result = item["result"]
            result.score = final_score
            final_results.append(result)
        
        # 排序并返回
        final_results.sort(key=lambda x: x.score, reverse=True)
        return final_results[:self.config.top_k]
    
    def _adaptive_fusion(
        self,
        query: str,
        vector_results: List[ScoredPoint],
        bm25_results: List[ScoredPoint]
    ) -> List[ScoredPoint]:
        """自适应融合算法"""
        # 分析查询特征来调整权重
        vector_weight = self.config.vector_weight
        bm25_weight = self.config.bm25_weight
        
        # 如果查询包含具体术语，提高BM25权重
        if len(query.split()) > 5:
            bm25_weight *= 1.2
            vector_weight *= 0.8
        
        # 临时调整配置
        old_vector_weight = self.config.vector_weight
        old_bm25_weight = self.config.bm25_weight
        
        self.config.vector_weight = vector_weight
        self.config.bm25_weight = bm25_weight
        
        try:
            result = self._weighted_fusion(vector_results, bm25_results)
        finally:
            # 恢复原配置
            self.config.vector_weight = old_vector_weight
            self.config.bm25_weight = old_bm25_weight
        
        return result
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """归一化分数"""
        if not scores:
            return []
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            return [1.0] * len(scores)
        
        return [(score - min_score) / (max_score - min_score) for score in scores]


# 全局实例
def get_hybrid_search_config() -> HybridSearchConfig:
    """获取混合搜索配置"""
    settings = get_settings()
    
    # 从配置文件映射搜索策略
    strategy_map = {
        "vector_only": SearchStrategy.VECTOR_ONLY,
        "bm25_only": SearchStrategy.BM25_ONLY,
        "hybrid_rrf": SearchStrategy.HYBRID_RRF,
        "hybrid_weighted": SearchStrategy.HYBRID_WEIGHTED,
        "adaptive": SearchStrategy.ADAPTIVE,
    }
    
    return HybridSearchConfig(
        vector_weight=settings.HYBRID_SEARCH_VECTOR_WEIGHT,
        bm25_weight=settings.HYBRID_SEARCH_BM25_WEIGHT,
        top_k=settings.HYBRID_SEARCH_TOP_K,
        rerank_size=settings.HYBRID_SEARCH_RERANK_SIZE,
        strategy=strategy_map.get(settings.HYBRID_SEARCH_STRATEGY, SearchStrategy.HYBRID_RRF),
        language="auto",
        enable_cache=settings.HYBRID_SEARCH_ENABLE_CACHE,
        cache_ttl=settings.CACHE_TTL_DEFAULT,
        rrf_k=settings.HYBRID_SEARCH_RRF_K
    )


class HybridSearchEngine:
    """混合搜索引擎"""
    
    def __init__(self):
        self.bm42_client = get_bm42_client()
        self.preprocessor = QueryPreprocessor()
        self.cache = SearchCache() if get_settings().HYBRID_SEARCH_ENABLE_CACHE else None
        self.settings = get_settings()
    
    async def search(
        self,
        query: str,
        collection: str = "documents",
        limit: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        strategy: Optional[SearchStrategy] = None
    ) -> List[SearchResult]:
        """主搜索接口"""
        try:
            # 使用指定的限制或配置默认值
            if limit is None:
                limit = self.bm42_client.config.top_k
            
            # 1. 查询预处理
            processed_query = await self.preprocessor.process(query)
            
            # 2. 检查缓存
            if self.cache:
                cache_key = self._generate_cache_key(processed_query, collection, filters, strategy)
                cached_result = await self.cache.get(cache_key)
                if cached_result:
                    return cached_result[:limit]
            
            # 3. 生成查询向量
            query_vector = await embedding_service.embed_text(processed_query.text)
            
            # 4. 构建过滤条件
            qdrant_filters = self._build_filters(filters) if filters else None
            
            # 5. 临时调整搜索策略
            original_strategy = self.bm42_client.config.strategy
            if strategy:
                self.bm42_client.config.strategy = strategy
            
            try:
                # 6. 执行混合搜索
                search_results = await self.bm42_client.hybrid_search(
                    query=processed_query.text,
                    vector=query_vector,
                    collection_name=collection,
                    filters=qdrant_filters
                )
                
                # 7. 转换为SearchResult格式
                formatted_results = self._format_results(search_results, collection)
                
                # 8. 后处理和重排序
                final_results = await self._post_process(formatted_results, processed_query)
                
                # 9. 缓存结果
                if self.cache:
                    await self.cache.set(cache_key, final_results)
                
                return final_results[:limit]
                
            finally:
                # 恢复原策略
                self.bm42_client.config.strategy = original_strategy
                
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            # 降级到语义搜索
            return await self._fallback_search(query, collection, limit, filters)
    
    async def multi_collection_search(
        self,
        query: str,
        collections: List[str] = ["documents", "code"],
        limit: int = 20,
        strategy: Optional[SearchStrategy] = None
    ) -> List[SearchResult]:
        """跨多个集合搜索"""
        all_results = []
        
        for collection in collections:
            try:
                results = await self.search(
                    query=query,
                    collection=collection,
                    limit=limit,
                    strategy=strategy
                )
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Failed to search in {collection}: {e}")
        
        # 按分数排序并去重
        unique_results = {}
        for result in all_results:
            key = f"{result.collection}:{result.id}"
            if key not in unique_results or result.final_score > unique_results[key].final_score:
                unique_results[key] = result
        
        final_results = list(unique_results.values())
        final_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return final_results[:limit]
    
    def _generate_cache_key(
        self,
        processed_query: ProcessedQuery,
        collection: str,
        filters: Optional[Dict[str, Any]],
        strategy: Optional[SearchStrategy]
    ) -> str:
        """生成缓存键"""
        key_data = {
            "query": processed_query.text,
            "language": processed_query.language,
            "collection": collection,
            "filters": filters or {},
            "strategy": strategy.value if strategy else self.bm42_client.config.strategy.value,
            "config": {
                "vector_weight": self.bm42_client.config.vector_weight,
                "bm25_weight": self.bm42_client.config.bm25_weight,
                "top_k": self.bm42_client.config.top_k
            }
        }
        
        key_str = str(key_data)
        return f"hybrid_search:{hashlib.md5(key_str.encode()).hexdigest()}"
    
    def _build_filters(self, filter_dict: Dict[str, Any]) -> Optional[Filter]:
        """构建Qdrant过滤条件"""
        if not filter_dict:
            return None
        
        must_conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, str):
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            elif isinstance(value, list):
                must_conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        
        return Filter(must=must_conditions) if must_conditions else None
    
    def _format_results(self, scored_points: List[ScoredPoint], collection: str) -> List[SearchResult]:
        """格式化搜索结果"""
        results = []
        
        for point in scored_points:
            result = SearchResult(
                id=point.id,
                score=point.score,
                content=point.payload.get("content", ""),
                file_path=point.payload.get("file_path", ""),
                file_type=point.payload.get("file_type", ""),
                chunk_index=point.payload.get("chunk_index", 0),
                metadata=point.payload,
                final_score=point.score,
                collection=collection
            )
            results.append(result)
        
        return results
    
    async def _post_process(
        self,
        results: List[SearchResult],
        processed_query: ProcessedQuery
    ) -> List[SearchResult]:
        """后处理和重排序"""
        if not results:
            return results
        
        # 多样性重排序
        reranked_results = self._diversity_rerank(results)
        
        return reranked_results
    
    def _diversity_rerank(self, results: List[SearchResult]) -> List[SearchResult]:
        """多样性重排序"""
        if len(results) <= 1:
            return results
        
        # 按文件分组
        file_groups = {}
        for result in results:
            file_path = result.file_path
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(result)
        
        # 重排序策略：确保不同文件的结果交替出现
        reranked = []
        used_files = set()
        
        # 第一轮：每个文件取最佳结果
        for result in results:
            if result.file_path not in used_files:
                reranked.append(result)
                used_files.add(result.file_path)
                if len(reranked) >= len(results) // 2:
                    break
        
        # 第二轮：添加剩余结果
        for result in results:
            if result not in reranked:
                reranked.append(result)
        
        return reranked
    
    async def _fallback_search(
        self,
        query: str,
        collection: str,
        limit: int,
        filters: Optional[Dict[str, Any]]
    ) -> List[SearchResult]:
        """降级搜索（纯向量搜索）"""
        try:
            from src.ai.rag.retriever import semantic_retriever
            
            # 使用现有的语义检索器作为降级
            results = await semantic_retriever.search(
                query=query,
                collection=collection,
                limit=limit,
                filter_dict=filters
            )
            
            # 转换格式
            search_results = []
            for result in results:
                search_result = SearchResult(
                    id=result["id"],
                    score=result["score"],
                    content=result["content"],
                    file_path=result["file_path"],
                    file_type=result["file_type"],
                    chunk_index=result["chunk_index"],
                    metadata=result["metadata"],
                    final_score=result["score"],
                    collection=collection
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Fallback search failed: {e}")
            return []


class SearchCache:
    """搜索结果缓存"""
    
    def __init__(self):
        self.settings = get_settings()
        self.hit_count = 0
        self.miss_count = 0
    
    async def get(self, key: str) -> Optional[List[SearchResult]]:
        """获取缓存结果"""
        try:
            # 简化实现：在实际项目中应使用Redis
            # 这里只做接口定义
            self.miss_count += 1
            return None
        except Exception as e:
            logger.warning(f"Cache get error: {e}")
            return None
    
    async def set(self, key: str, results: List[SearchResult]) -> None:
        """设置缓存结果"""
        try:
            # 简化实现：在实际项目中应使用Redis
            pass
        except Exception as e:
            logger.warning(f"Cache set error: {e}")
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0


def get_bm42_client() -> QdrantBM42Client:
    """获取BM42客户端实例"""
    client = get_qdrant_client()
    config = get_hybrid_search_config()
    return QdrantBM42Client(client, config)


def get_hybrid_search_engine() -> HybridSearchEngine:
    """获取混合搜索引擎实例"""
    return HybridSearchEngine()