"""
语义检索和混合检索实现
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Tuple
import numpy as np
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchText
from src.ai.rag.embeddings import embedding_service
from functools import partial
from src.core.qdrant import get_qdrant_client
from src.core.utils.async_utils import run_sync_io

from src.core.logging import get_logger
logger = get_logger(__name__)

class SemanticRetriever:
    """语义检索器"""

    def __init__(self):
        self.client = None
        self.embedding_service = embedding_service

    def _get_client(self):
        if self.client is None:
            self.client = get_qdrant_client()
        return self.client

    async def search(
        self,
        query: str,
        collection: str = "documents",
        limit: int = 10,
        score_threshold: float = 0.7,
        filter_dict: Optional[Dict] = None,
    ) -> List[Dict]:
        """执行语义搜索"""
        # 生成查询向量
        query_vector = await self.embedding_service.embed_text(query)
        
        # 构建过滤条件
        filters = None
        if filter_dict:
            must_conditions = []
            for key, value in filter_dict.items():
                if isinstance(value, str):
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
                elif isinstance(value, list):
                    # 支持多值匹配
                    must_conditions.append(
                        FieldCondition(key=key, match=MatchValue(value=value))
                    )
            if must_conditions:
                filters = Filter(must=must_conditions)
        
        # 执行搜索
        client = self._get_client()

        results = await run_sync_io(
            partial(
                client.search,
                collection_name=collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filters,
                with_payload=True,
                with_vectors=False,
            )
        )
        
        # 格式化结果
        formatted_results = []
        for hit in results:
            result = {
                "id": hit.id,
                "score": hit.score,
                "content": hit.payload.get("content", ""),
                "file_path": hit.payload.get("file_path", ""),
                "file_type": hit.payload.get("file_type", ""),
                "chunk_index": hit.payload.get("chunk_index", 0),
                "metadata": hit.payload,
            }
            formatted_results.append(result)
        
        return formatted_results

    async def multi_collection_search(
        self,
        query: str,
        collections: List[str] = ["documents", "code"],
        limit: int = 10,
        score_threshold: float = 0.7,
    ) -> List[Dict]:
        """跨多个集合搜索"""
        all_results = []
        
        for collection in collections:
            try:
                results = await self.search(
                    query=query,
                    collection=collection,
                    limit=limit,
                    score_threshold=score_threshold,
                )
                for result in results:
                    result["collection"] = collection
                all_results.extend(results)
            except Exception as e:
                logger.error(f"Failed to search in {collection}: {e}")
        
        # 按分数排序
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # 取前n个结果
        return all_results[:limit]

class HybridRetriever:
    """混合检索器（语义搜索 + 关键词匹配）"""

    def __init__(self):
        self.semantic_retriever = SemanticRetriever()
        self.client = None

    def _get_client(self):
        if self.client is None:
            self.client = get_qdrant_client()
        return self.client

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 移除特殊字符
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # 分词
        words = text.split()
        
        # 移除停用词
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',
            'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'should', 'could', 'may', 'might', 'must', 'can', 'shall',
            'it', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we',
            'they', 'what', 'which', 'who', 'when', 'where', 'why', 'how'
        }
        
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        # 返回最常见的关键词
        counter = Counter(keywords)
        return [word for word, _ in counter.most_common(10)]

    def _calculate_bm25_score(
        self, 
        query_keywords: List[str], 
        doc_content: str,
        avg_doc_length: float = 1000,
        k1: float = 1.2,
        b: float = 0.75
    ) -> float:
        """计算BM25分数"""
        doc_keywords = self._extract_keywords(doc_content)
        doc_length = len(doc_content.split())
        
        score = 0.0
        for keyword in query_keywords:
            # 词频
            tf = doc_keywords.count(keyword)
            if tf == 0:
                continue
            
            # BM25公式
            idf = 1.0  # 简化的IDF，实际应该基于文档集合计算
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            score += idf * (numerator / denominator)
        
        return score

    async def hybrid_search(
        self,
        query: str,
        collection: str = "documents",
        limit: int = 10,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        score_threshold: float = 0.5,
    ) -> List[Dict]:
        """执行混合检索"""
        # 提取关键词
        query_keywords = self._extract_keywords(query)
        
        # 语义搜索
        semantic_results = await self.semantic_retriever.search(
            query=query,
            collection=collection,
            limit=limit * 2,  # 获取更多结果以便融合
            score_threshold=score_threshold,
        )
        
        # 关键词搜索（使用Qdrant的文本搜索）
        keyword_results = []
        if query_keywords:
            try:
                client = self._get_client()
                # 构建文本搜索过滤器
                text_filter = Filter(
                    should=[
                        FieldCondition(
                            key="content",
                            match=MatchText(text=keyword)
                        )
                        for keyword in query_keywords
                    ]
                )
                
                # 执行搜索
                scroll_result = await run_sync_io(
                    partial(
                        client.scroll,
                        collection_name=collection,
                        scroll_filter=text_filter,
                        limit=limit * 2,
                        with_payload=True,
                        with_vectors=False,
                    )
                )
                
                for point in scroll_result[0]:
                    bm25_score = self._calculate_bm25_score(
                        query_keywords,
                        point.payload.get("content", "")
                    )
                    keyword_results.append({
                        "id": point.id,
                        "score": bm25_score,
                        "content": point.payload.get("content", ""),
                        "file_path": point.payload.get("file_path", ""),
                        "file_type": point.payload.get("file_type", ""),
                        "chunk_index": point.payload.get("chunk_index", 0),
                        "metadata": point.payload,
                    })
            except Exception as e:
                logger.error(f"Keyword search failed: {e}")
        
        # 融合结果
        combined_results = {}
        
        # 添加语义搜索结果
        for result in semantic_results:
            combined_results[result["id"]] = {
                **result,
                "final_score": result["score"] * semantic_weight,
                "semantic_score": result["score"],
                "keyword_score": 0.0,
            }
        
        # 融合关键词搜索结果
        for result in keyword_results:
            if result["id"] in combined_results:
                # 已存在，更新分数
                combined_results[result["id"]]["keyword_score"] = result["score"]
                combined_results[result["id"]]["final_score"] += (
                    result["score"] * keyword_weight
                )
            else:
                # 新结果
                combined_results[result["id"]] = {
                    **result,
                    "final_score": result["score"] * keyword_weight,
                    "semantic_score": 0.0,
                    "keyword_score": result["score"],
                }
        
        # 排序并返回
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return final_results[:limit]

    async def rerank_results(
        self,
        query: str,
        results: List[Dict],
        diversity_weight: float = 0.2,
    ) -> List[Dict]:
        """重新排序结果，考虑多样性"""
        if not results:
            return results
        
        # 计算文件多样性
        file_groups = {}
        for result in results:
            file_path = result.get("file_path", "")
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(result)
        
        # 重新排序
        reranked = []
        used_files = set()
        
        # 第一轮：每个文件取最佳结果
        for result in results:
            file_path = result.get("file_path", "")
            if file_path not in used_files:
                reranked.append(result)
                used_files.add(file_path)
                if len(reranked) >= len(results) // 2:
                    break
        
        # 第二轮：添加剩余的高分结果
        for result in results:
            if result not in reranked:
                reranked.append(result)
        
        return reranked

class QueryIntentClassifier:
    """查询意图分类器"""

    def classify(self, query: str) -> Dict[str, any]:
        """分类查询意图"""
        intent = {
            "type": "general",  # general, code, documentation, error
            "language": None,
            "framework": None,
            "keywords": [],
        }
        
        # 检测代码相关查询
        code_keywords = [
            "function", "class", "method", "implement", "code", "algorithm",
            "bug", "error", "exception", "debug", "fix", "compile"
        ]
        if any(keyword in query.lower() for keyword in code_keywords):
            intent["type"] = "code"
        
        # 检测文档相关查询
        doc_keywords = [
            "document", "guide", "tutorial", "example", "how to", "what is",
            "explain", "description", "reference"
        ]
        if any(keyword in query.lower() for keyword in doc_keywords):
            intent["type"] = "documentation"
        
        # 检测编程语言
        languages = {
            "python": ["python", "py", "django", "flask"],
            "javascript": ["javascript", "js", "node", "react", "vue"],
            "typescript": ["typescript", "ts"],
            "java": ["java", "spring", "maven"],
            "cpp": ["c++", "cpp"],
            "go": ["golang", "go"],
            "rust": ["rust", "cargo"],
        }
        
        for lang, keywords in languages.items():
            if any(keyword in query.lower() for keyword in keywords):
                intent["language"] = lang
                break
        
        return intent

# 全局检索器实例
semantic_retriever = SemanticRetriever()
hybrid_retriever = HybridRetriever()
query_classifier = QueryIntentClassifier()
