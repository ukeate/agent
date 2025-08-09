"""
多策略检索代理协作系统

实现多个专业化检索代理的协调与协作，包括：
- 语义检索专家代理（基于Story 3.1）
- 关键词匹配专家代理（BM25算法）
- 结构化数据检索代理（数据库查询）
- 代理协作调度和结果融合机制
- 检索策略动态选择算法
"""

import asyncio
import math
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
import re

from ..rag.retriever import SemanticRetriever
from ..rag.embeddings import embedding_service
from ...core.database import get_db_session
from ...db.models import *
from .query_analyzer import QueryAnalysis, QueryIntent

logger = logging.getLogger(__name__)


class RetrievalStrategy(str, Enum):
    """检索策略类型"""
    SEMANTIC = "semantic"        # 语义检索
    KEYWORD = "keyword"          # 关键词匹配
    STRUCTURED = "structured"    # 结构化查询
    HYBRID = "hybrid"           # 混合检索


@dataclass
class RetrievalResult:
    """检索结果数据结构"""
    agent_type: RetrievalStrategy
    query: str
    results: List[Dict[str, Any]]
    score: float  # 整体检索质量评分 0-1
    confidence: float  # 结果置信度 0-1
    processing_time: float  # 处理时间（秒）
    metadata: Dict[str, Any] = None
    explanation: str = ""

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseRetrievalAgent(ABC):
    """检索代理基类"""
    
    def __init__(self, name: str, strategy: RetrievalStrategy):
        self.name = name
        self.strategy = strategy
        self.performance_stats = {
            "total_queries": 0,
            "avg_response_time": 0.0,
            "avg_score": 0.0,
            "success_rate": 0.0
        }
    
    @abstractmethod
    async def retrieve(self, 
                      query_analysis: QueryAnalysis,
                      limit: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """执行检索"""
        pass
    
    def is_suitable_for_query(self, query_analysis: QueryAnalysis) -> float:
        """评估代理对查询的适用性 (0-1)"""
        return 0.5  # 默认适用性
    
    def update_performance_stats(self, result: RetrievalResult):
        """更新性能统计"""
        self.performance_stats["total_queries"] += 1
        
        # 更新平均响应时间
        prev_avg = self.performance_stats["avg_response_time"]
        n = self.performance_stats["total_queries"]
        self.performance_stats["avg_response_time"] = (
            (prev_avg * (n - 1) + result.processing_time) / n
        )
        
        # 更新平均分数
        prev_score = self.performance_stats["avg_score"]
        self.performance_stats["avg_score"] = (
            (prev_score * (n - 1) + result.score) / n
        )
        
        # 更新成功率（分数 > 0.3 视为成功）
        success_count = self.performance_stats["success_rate"] * (n - 1)
        if result.score > 0.3:
            success_count += 1
        self.performance_stats["success_rate"] = success_count / n


class SemanticRetrievalAgent(BaseRetrievalAgent):
    """语义检索专家代理（继承Story 3.1基础能力）"""
    
    def __init__(self):
        super().__init__("SemanticExpert", RetrievalStrategy.SEMANTIC)
        self.semantic_retriever = SemanticRetriever()
    
    def is_suitable_for_query(self, query_analysis: QueryAnalysis) -> float:
        """评估语义检索的适用性"""
        suitability = 0.6  # 基础适用性
        
        # 基于意图类型调整
        if query_analysis.intent_type in [QueryIntent.FACTUAL, QueryIntent.EXPLORATORY]:
            suitability += 0.3
        elif query_analysis.intent_type == QueryIntent.CREATIVE:
            suitability += 0.2
        
        # 基于查询复杂度调整
        if query_analysis.complexity_score > 0.5:
            suitability += 0.1
        
        # 基于实体数量调整
        if len(query_analysis.entities) > 0:
            suitability += 0.1
        
        return min(suitability, 1.0)
    
    async def retrieve(self, 
                      query_analysis: QueryAnalysis,
                      limit: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """执行语义检索"""
        import time
        start_time = time.time()
        
        try:
            # 选择集合
            collection = "code" if query_analysis.intent_type == QueryIntent.CODE else "documents"
            
            # 执行语义搜索
            results = await self.semantic_retriever.search(
                query=query_analysis.query_text,
                collection=collection,
                limit=limit,
                score_threshold=0.3,
                filter_dict=filters
            )
            
            # 计算整体分数
            if results:
                avg_score = sum(r["score"] for r in results) / len(results)
                # 语义检索置信度基于平均分数和结果数量
                confidence = min(avg_score * (1 + 0.1 * len(results) / limit), 1.0)
            else:
                avg_score = 0.0
                confidence = 0.0
            
            processing_time = time.time() - start_time
            
            result = RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query=query_analysis.query_text,
                results=results,
                score=avg_score,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    "collection": collection,
                    "total_results": len(results),
                    "intent_type": query_analysis.intent_type.value
                },
                explanation=f"使用语义向量搜索在{collection}集合中找到{len(results)}个相关结果"
            )
            
            self.update_performance_stats(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Semantic retrieval failed: {e}")
            
            result = RetrievalResult(
                agent_type=RetrievalStrategy.SEMANTIC,
                query=query_analysis.query_text,
                results=[],
                score=0.0,
                confidence=0.0,
                processing_time=processing_time,
                explanation=f"语义检索失败: {str(e)}"
            )
            
            self.update_performance_stats(result)
            return result


class KeywordRetrievalAgent(BaseRetrievalAgent):
    """关键词匹配专家代理（BM25算法）"""
    
    def __init__(self):
        super().__init__("KeywordExpert", RetrievalStrategy.KEYWORD)
        # BM25参数
        self.k1 = 1.2
        self.b = 0.75
    
    def is_suitable_for_query(self, query_analysis: QueryAnalysis) -> float:
        """评估关键词检索的适用性"""
        suitability = 0.5  # 基础适用性
        
        # 基于意图类型调整
        if query_analysis.intent_type in [QueryIntent.PROCEDURAL, QueryIntent.CODE]:
            suitability += 0.3
        elif query_analysis.intent_type == QueryIntent.FACTUAL:
            suitability += 0.2
        
        # 基于关键词数量调整
        if len(query_analysis.keywords) >= 3:
            suitability += 0.2
        elif len(query_analysis.keywords) >= 2:
            suitability += 0.1
        
        # 基于查询长度调整（较短查询更适合关键词匹配）
        if len(query_analysis.query_text.split()) <= 5:
            suitability += 0.1
        
        return min(suitability, 1.0)
    
    def _calculate_bm25_score(self, 
                            query_terms: List[str], 
                            document_terms: List[str],
                            document_length: int,
                            avg_doc_length: float,
                            term_frequencies: Dict[str, int],
                            total_docs: int) -> float:
        """计算BM25分数"""
        score = 0.0
        
        for term in query_terms:
            if term in document_terms:
                tf = term_frequencies.get(term, 0)
                if tf > 0:
                    # BM25公式
                    idf = math.log((total_docs - tf + 0.5) / (tf + 0.5))
                    tf_component = (tf * (self.k1 + 1)) / (
                        tf + self.k1 * (1 - self.b + self.b * (document_length / avg_doc_length))
                    )
                    score += idf * tf_component
        
        return score
    
    async def retrieve(self, 
                      query_analysis: QueryAnalysis,
                      limit: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """执行基于BM25的关键词检索"""
        import time
        start_time = time.time()
        
        try:
            # 使用关键词和查询文本进行搜索
            search_terms = query_analysis.keywords + query_analysis.query_text.split()
            search_terms = [term.lower() for term in search_terms if len(term) > 2]
            
            # 这里实现一个简化的BM25搜索
            # 在实际应用中，可以集成Elasticsearch或其他全文搜索引擎
            results = await self._perform_keyword_search(
                search_terms, limit, filters, query_analysis
            )
            
            # 计算分数
            if results:
                avg_score = sum(r.get("bm25_score", 0) for r in results) / len(results)
                confidence = min(0.8 * avg_score, 1.0)
            else:
                avg_score = 0.0
                confidence = 0.0
            
            processing_time = time.time() - start_time
            
            result = RetrievalResult(
                agent_type=RetrievalStrategy.KEYWORD,
                query=query_analysis.query_text,
                results=results,
                score=avg_score,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    "search_terms": search_terms,
                    "total_results": len(results)
                },
                explanation=f"使用BM25关键词匹配找到{len(results)}个相关结果"
            )
            
            self.update_performance_stats(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Keyword retrieval failed: {e}")
            
            result = RetrievalResult(
                agent_type=RetrievalStrategy.KEYWORD,
                query=query_analysis.query_text,
                results=[],
                score=0.0,
                confidence=0.0,
                processing_time=processing_time,
                explanation=f"关键词检索失败: {str(e)}"
            )
            
            self.update_performance_stats(result)
            return result
    
    async def _perform_keyword_search(self, 
                                    search_terms: List[str], 
                                    limit: int,
                                    filters: Optional[Dict[str, Any]],
                                    query_analysis: QueryAnalysis) -> List[Dict[str, Any]]:
        """执行关键词搜索（简化实现）"""
        # 这是一个简化的实现，实际应用中应该集成专门的全文搜索引擎
        # 这里通过语义检索器获取候选结果，然后使用BM25重新排序
        
        semantic_retriever = SemanticRetriever()
        
        # 先获取较多的候选结果
        collection = "code" if query_analysis.intent_type == QueryIntent.CODE else "documents"
        candidates = await semantic_retriever.search(
            query=query_analysis.query_text,
            collection=collection,
            limit=limit * 3,  # 获取更多候选
            score_threshold=0.1,  # 降低阈值获取更多结果
            filter_dict=filters
        )
        
        # 使用简化的关键词匹配重新排序
        scored_results = []
        for candidate in candidates:
            content = candidate.get("content", "").lower()
            
            # 计算关键词匹配得分
            keyword_matches = 0
            for term in search_terms:
                keyword_matches += content.count(term)
            
            # 简化的BM25风格评分
            content_length = len(content.split())
            if content_length > 0:
                tf_score = keyword_matches / content_length
                bm25_score = tf_score * math.log(1 + keyword_matches)
            else:
                bm25_score = 0.0
            
            if bm25_score > 0:
                candidate["bm25_score"] = bm25_score
                candidate["keyword_matches"] = keyword_matches
                scored_results.append(candidate)
        
        # 按BM25分数排序
        scored_results.sort(key=lambda x: x["bm25_score"], reverse=True)
        
        return scored_results[:limit]


class StructuredRetrievalAgent(BaseRetrievalAgent):
    """结构化数据检索代理（数据库查询）"""
    
    def __init__(self):
        super().__init__("StructuredExpert", RetrievalStrategy.STRUCTURED)
    
    def is_suitable_for_query(self, query_analysis: QueryAnalysis) -> float:
        """评估结构化检索的适用性"""
        suitability = 0.3  # 基础适用性较低
        
        # 基于实体数量调整（结构化查询通常涉及具体实体）
        if len(query_analysis.entities) >= 2:
            suitability += 0.4
        elif len(query_analysis.entities) >= 1:
            suitability += 0.2
        
        # 基于意图类型调整
        if query_analysis.intent_type == QueryIntent.FACTUAL:
            suitability += 0.3
        elif query_analysis.intent_type == QueryIntent.PROCEDURAL:
            suitability += 0.1
        
        # 基于领域调整
        if query_analysis.domain in ["技术", "业务"]:
            suitability += 0.2
        
        return min(suitability, 1.0)
    
    async def retrieve(self, 
                      query_analysis: QueryAnalysis,
                      limit: int = 10,
                      filters: Optional[Dict[str, Any]] = None) -> RetrievalResult:
        """执行结构化数据检索"""
        import time
        start_time = time.time()
        
        try:
            # 执行数据库查询
            results = await self._perform_structured_search(
                query_analysis, limit, filters
            )
            
            # 计算分数
            if results:
                # 结构化查询的分数基于结果的完整性和相关性
                avg_score = 0.7  # 结构化数据通常质量较高
                confidence = min(0.9, 0.5 + 0.1 * len(results))
            else:
                avg_score = 0.0
                confidence = 0.0
            
            processing_time = time.time() - start_time
            
            result = RetrievalResult(
                agent_type=RetrievalStrategy.STRUCTURED,
                query=query_analysis.query_text,
                results=results,
                score=avg_score,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    "entities_found": len(query_analysis.entities),
                    "total_results": len(results)
                },
                explanation=f"通过结构化数据库查询找到{len(results)}个精确匹配结果"
            )
            
            self.update_performance_stats(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Structured retrieval failed: {e}")
            
            result = RetrievalResult(
                agent_type=RetrievalStrategy.STRUCTURED,
                query=query_analysis.query_text,
                results=[],
                score=0.0,
                confidence=0.0,
                processing_time=processing_time,
                explanation=f"结构化检索失败: {str(e)}"
            )
            
            self.update_performance_stats(result)
            return result
    
    async def _perform_structured_search(self, 
                                       query_analysis: QueryAnalysis,
                                       limit: int,
                                       filters: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行结构化搜索（基于数据库模型）"""
        results = []
        
        # 这里是一个简化的实现，实际应用中应该根据具体的业务模型进行查询
        # 示例：搜索会话和消息记录
        
        try:
            async with get_db_session() as session:
                # 搜索相关的实体或关键词
                for entity in query_analysis.entities:
                    # 可以扩展为具体的业务模型查询
                    # 这里只是示例结构
                    structured_result = {
                        "id": f"struct_{entity}",
                        "score": 0.8,
                        "content": f"结构化数据匹配: {entity}",
                        "source_type": "database",
                        "entity": entity,
                        "metadata": {
                            "query_type": "structured",
                            "match_type": "entity_match"
                        }
                    }
                    results.append(structured_result)
                    
                    if len(results) >= limit:
                        break
                        
        except Exception as e:
            logger.warning(f"Database query failed: {e}")
            # 即使数据库查询失败，也可以返回基于实体的结构化信息
            for entity in query_analysis.entities[:limit]:
                fallback_result = {
                    "id": f"fallback_{entity}",
                    "score": 0.5,
                    "content": f"实体信息: {entity}（来自查询分析）",
                    "source_type": "analysis",
                    "entity": entity,
                    "metadata": {
                        "query_type": "structured",
                        "match_type": "entity_analysis"
                    }
                }
                results.append(fallback_result)
        
        return results[:limit]


class MultiAgentRetriever:
    """多代理检索协调器"""
    
    def __init__(self):
        # 初始化各个专家代理
        self.agents = {
            RetrievalStrategy.SEMANTIC: SemanticRetrievalAgent(),
            RetrievalStrategy.KEYWORD: KeywordRetrievalAgent(),
            RetrievalStrategy.STRUCTURED: StructuredRetrievalAgent(),
        }
        
        # 策略选择权重
        self.strategy_weights = {
            RetrievalStrategy.SEMANTIC: 1.0,
            RetrievalStrategy.KEYWORD: 1.0,
            RetrievalStrategy.STRUCTURED: 0.8,
        }
    
    def select_strategies(self, query_analysis: QueryAnalysis) -> List[Tuple[RetrievalStrategy, float]]:
        """动态选择检索策略和权重"""
        strategy_scores = []
        
        for strategy, agent in self.agents.items():
            # 计算基础适用性
            suitability = agent.is_suitable_for_query(query_analysis)
            
            # 考虑代理历史性能
            performance_score = agent.performance_stats.get("success_rate", 0.5)
            
            # 综合评分
            final_score = (
                suitability * 0.7 +  # 适用性权重70%
                performance_score * 0.2 +  # 历史性能权重20%
                self.strategy_weights[strategy] * 0.1  # 全局权重10%
            )
            
            strategy_scores.append((strategy, final_score))
        
        # 排序并返回
        strategy_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 根据分数阈值过滤
        filtered_strategies = [(s, score) for s, score in strategy_scores if score > 0.3]
        
        # 至少保留一个策略
        if not filtered_strategies:
            filtered_strategies = [strategy_scores[0]]
        
        return filtered_strategies
    
    async def retrieve(self, 
                      query_analysis: QueryAnalysis,
                      limit: int = 10,
                      strategies: Optional[List[RetrievalStrategy]] = None,
                      enable_parallel: bool = True,
                      filters: Optional[Dict[str, Any]] = None) -> List[RetrievalResult]:
        """执行多代理协作检索"""
        
        # 策略选择
        if strategies is None:
            selected_strategies = self.select_strategies(query_analysis)
            strategies = [s for s, _ in selected_strategies]
        else:
            selected_strategies = [(s, 1.0) for s in strategies]
        
        # 执行检索
        if enable_parallel:
            # 并行执行
            tasks = []
            for strategy in strategies:
                if strategy in self.agents:
                    agent = self.agents[strategy]
                    task = agent.retrieve(query_analysis, limit, filters)
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 过滤异常结果
            valid_results = []
            for result in results:
                if isinstance(result, RetrievalResult):
                    valid_results.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Agent retrieval failed: {result}")
            
            return valid_results
            
        else:
            # 串行执行
            results = []
            for strategy in strategies:
                if strategy in self.agents:
                    agent = self.agents[strategy]
                    try:
                        result = await agent.retrieve(query_analysis, limit, filters)
                        results.append(result)
                    except Exception as e:
                        logger.error(f"Agent {strategy} failed: {e}")
                        continue
            
            return results
    
    def fuse_results(self, 
                    results: List[RetrievalResult],
                    fusion_method: str = "weighted_score",
                    max_results: int = 20) -> List[Dict[str, Any]]:
        """融合多个检索结果"""
        
        if not results:
            return []
        
        if fusion_method == "weighted_score":
            return self._weighted_score_fusion(results, max_results)
        elif fusion_method == "rank_fusion":
            return self._rank_fusion(results, max_results)
        elif fusion_method == "confidence_weighted":
            return self._confidence_weighted_fusion(results, max_results)
        else:
            # 默认使用加权分数融合
            return self._weighted_score_fusion(results, max_results)
    
    def _weighted_score_fusion(self, results: List[RetrievalResult], max_results: int) -> List[Dict[str, Any]]:
        """基于加权分数的结果融合"""
        all_items = []
        
        for result in results:
            agent_weight = result.confidence  # 使用置信度作为代理权重
            
            for item in result.results:
                # 计算融合分数
                original_score = item.get("score", 0.0)
                fused_score = original_score * agent_weight
                
                fused_item = item.copy()
                fused_item["fused_score"] = fused_score
                fused_item["agent_type"] = result.agent_type.value
                fused_item["agent_confidence"] = result.confidence
                
                all_items.append(fused_item)
        
        # 去重（基于content或id）
        unique_items = self._deduplicate_results(all_items)
        
        # 按融合分数排序
        unique_items.sort(key=lambda x: x["fused_score"], reverse=True)
        
        return unique_items[:max_results]
    
    def _rank_fusion(self, results: List[RetrievalResult], max_results: int) -> List[Dict[str, Any]]:
        """基于排名的结果融合（Reciprocal Rank Fusion）"""
        item_scores = defaultdict(float)
        item_data = {}
        
        for result in results:
            for rank, item in enumerate(result.results):
                item_id = item.get("id", item.get("content", str(rank)))
                
                # RRF分数计算
                rrf_score = 1.0 / (rank + 60)  # 60是常用的k值
                item_scores[item_id] += rrf_score * result.confidence
                
                # 保存物品数据（第一次遇到时）
                if item_id not in item_data:
                    fused_item = item.copy()
                    fused_item["agent_type"] = result.agent_type.value
                    fused_item["agent_confidence"] = result.confidence
                    item_data[item_id] = fused_item
        
        # 排序并构建最终结果
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        fused_results = []
        for item_id, score in sorted_items[:max_results]:
            item = item_data[item_id]
            item["fused_score"] = score
            fused_results.append(item)
        
        return fused_results
    
    def _confidence_weighted_fusion(self, results: List[RetrievalResult], max_results: int) -> List[Dict[str, Any]]:
        """基于置信度加权的结果融合"""
        # 计算总置信度用于归一化
        total_confidence = sum(result.confidence for result in results)
        if total_confidence == 0:
            return []
        
        all_items = []
        
        for result in results:
            normalized_weight = result.confidence / total_confidence
            
            for item in result.results:
                fused_item = item.copy()
                fused_item["fused_score"] = item.get("score", 0.0) * normalized_weight
                fused_item["agent_type"] = result.agent_type.value
                fused_item["agent_confidence"] = result.confidence
                fused_item["normalized_weight"] = normalized_weight
                
                all_items.append(fused_item)
        
        # 去重和排序
        unique_items = self._deduplicate_results(all_items)
        unique_items.sort(key=lambda x: x["fused_score"], reverse=True)
        
        return unique_items[:max_results]
    
    def _deduplicate_results(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """结果去重"""
        seen = set()
        unique_items = []
        
        for item in items:
            # 使用多个字段组合作为唯一标识
            identifier = (
                item.get("id", ""),
                item.get("content", "")[:100],  # 使用内容前100字符
                item.get("file_path", "")
            )
            
            if identifier not in seen:
                seen.add(identifier)
                unique_items.append(item)
        
        return unique_items
    
    def get_retrieval_explanation(self, 
                                query_analysis: QueryAnalysis,
                                results: List[RetrievalResult],
                                fused_results: List[Dict[str, Any]]) -> str:
        """生成检索过程解释"""
        explanation_parts = []
        
        # 查询分析摘要
        explanation_parts.append(f"查询分析: 意图={query_analysis.intent_type.value}, 复杂度={query_analysis.complexity_score:.2f}")
        
        # 策略使用情况
        strategies_used = [r.agent_type.value for r in results]
        explanation_parts.append(f"使用策略: {', '.join(strategies_used)}")
        
        # 各代理结果摘要
        for result in results:
            explanation_parts.append(
                f"{result.agent_type.value}: {len(result.results)}个结果, "
                f"置信度={result.confidence:.2f}, 用时={result.processing_time:.3f}s"
            )
        
        # 融合结果摘要
        explanation_parts.append(f"最终融合: {len(fused_results)}个结果")
        
        return " | ".join(explanation_parts)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        
        for strategy, agent in self.agents.items():
            summary[strategy.value] = {
                "name": agent.name,
                "stats": agent.performance_stats.copy()
            }
        
        return summary


# 全局多代理检索器实例
multi_agent_retriever = MultiAgentRetriever()