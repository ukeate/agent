"""智能检索策略"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from .multimodal_config import (
    QueryContext,
    QueryType,
    RetrievalResults,
    MultimodalConfig
)
from .multimodal_vectorstore import MultimodalVectorStore
from .multimodal_query_analyzer import MultimodalQueryAnalyzer

logger = get_logger(__name__)

@dataclass
class RetrievalWeight:
    """检索权重配置"""
    text_weight: float = 0.5
    image_weight: float = 0.3
    table_weight: float = 0.2
    
    def normalize(self):
        """归一化权重"""
        total = self.text_weight + self.image_weight + self.table_weight
        if total > 0:
            self.text_weight /= total
            self.image_weight /= total
            self.table_weight /= total

class SmartRetrievalStrategy:
    """基于内容类型的智能检索"""
    
    def __init__(
        self,
        config: MultimodalConfig,
        vector_store: MultimodalVectorStore,
        query_analyzer: MultimodalQueryAnalyzer
    ):
        """初始化检索策略
        
        Args:
            config: 多模态配置
            vector_store: 向量存储
            query_analyzer: 查询分析器
        """
        self.config = config
        self.vector_store = vector_store
        self.query_analyzer = query_analyzer
        
        # 默认检索权重
        self.default_weights = {
            QueryType.TEXT: RetrievalWeight(0.8, 0.1, 0.1),
            QueryType.VISUAL: RetrievalWeight(0.2, 0.7, 0.1),
            QueryType.DOCUMENT: RetrievalWeight(0.5, 0.2, 0.3),
            QueryType.MIXED: RetrievalWeight(0.4, 0.3, 0.3)
        }
    
    async def retrieve(
        self,
        query_context: QueryContext
    ) -> RetrievalResults:
        """执行智能检索
        
        Args:
            query_context: 查询上下文
            
        Returns:
            检索结果
        """
        # 根据查询类型选择检索策略
        if query_context.query_type == QueryType.VISUAL:
            return await self._visual_retrieval(query_context)
        elif query_context.query_type == QueryType.MIXED:
            return await self._hybrid_retrieval(query_context)
        elif query_context.query_type == QueryType.DOCUMENT:
            return await self._document_retrieval(query_context)
        else:
            return await self._text_retrieval(query_context)
    
    async def _text_retrieval(
        self,
        query_context: QueryContext
    ) -> RetrievalResults:
        """纯文本检索
        
        Args:
            query_context: 查询上下文
            
        Returns:
            检索结果
        """
        logger.info("Performing text retrieval")
        
        # 执行文本搜索
        results = await self.vector_store.search(
            query=query_context.query,
            search_type="text",
            top_k=query_context.top_k or self.config.retrieval_top_k,
            filters=query_context.filters
        )
        
        # 如果启用重排序
        if self.config.rerank_enabled:
            results = await self._rerank_results(results, query_context)
        
        return results
    
    async def _visual_retrieval(
        self,
        query_context: QueryContext
    ) -> RetrievalResults:
        """视觉内容检索
        
        Args:
            query_context: 查询上下文
            
        Returns:
            检索结果
        """
        logger.info("Performing visual retrieval")
        
        # 主要搜索图像
        results = await self.vector_store.search(
            query=query_context.query,
            search_type="image",
            top_k=query_context.top_k or self.config.retrieval_top_k,
            filters=query_context.filters
        )
        
        # 补充相关文本
        if results.total_results < (query_context.top_k or self.config.retrieval_top_k):
            text_results = await self.vector_store.search(
                query=query_context.query,
                search_type="text",
                top_k=5,  # 少量补充
                filters=query_context.filters
            )
            results.texts.extend(text_results.texts)
            results.total_results = len(results.texts) + len(results.images)
        
        return results
    
    async def _document_retrieval(
        self,
        query_context: QueryContext
    ) -> RetrievalResults:
        """文档检索（包括表格）
        
        Args:
            query_context: 查询上下文
            
        Returns:
            检索结果
        """
        logger.info("Performing document retrieval")
        
        # 并行搜索文本和表格
        text_task = self.vector_store.search(
            query=query_context.query,
            search_type="text",
            top_k=query_context.top_k or self.config.retrieval_top_k,
            filters=query_context.filters
        )
        
        table_task = self.vector_store.search(
            query=query_context.query,
            search_type="table",
            top_k=(query_context.top_k or self.config.retrieval_top_k) // 2,
            filters=query_context.filters
        )
        
        text_results, table_results = await asyncio.gather(text_task, table_task)
        
        # 合并结果
        combined_results = RetrievalResults()
        combined_results.texts = text_results.texts
        combined_results.tables = table_results.tables
        combined_results.sources = list(set(text_results.sources + table_results.sources))
        combined_results.total_results = len(combined_results.texts) + len(combined_results.tables)
        combined_results.retrieval_time_ms = max(
            text_results.retrieval_time_ms,
            table_results.retrieval_time_ms
        )
        
        return combined_results
    
    async def _hybrid_retrieval(
        self,
        query_context: QueryContext
    ) -> RetrievalResults:
        """混合检索（多模态）
        
        Args:
            query_context: 查询上下文
            
        Returns:
            检索结果
        """
        logger.info("Performing hybrid retrieval")
        
        # 获取动态权重
        weights = self._calculate_dynamic_weights(query_context)
        
        # 计算每种类型的检索数量
        total_k = query_context.top_k or self.config.retrieval_top_k
        text_k = int(total_k * weights.text_weight)
        image_k = int(total_k * weights.image_weight)
        table_k = int(total_k * weights.table_weight)
        
        # 确保至少检索一个
        text_k = max(text_k, 1)
        image_k = max(image_k, 1) if query_context.requires_image_search else 0
        table_k = max(table_k, 1) if query_context.requires_table_search else 0
        
        # 并行执行多种检索
        tasks = []
        
        tasks.append(self.vector_store.search(
            query=query_context.query,
            search_type="text",
            top_k=text_k,
            filters=query_context.filters
        ))
        
        if image_k > 0:
            tasks.append(self.vector_store.search(
                query=query_context.query,
                search_type="image",
                top_k=image_k,
                filters=query_context.filters
            ))
        
        if table_k > 0:
            tasks.append(self.vector_store.search(
                query=query_context.query,
                search_type="table",
                top_k=table_k,
                filters=query_context.filters
            ))
        
        # 等待所有检索完成
        results_list = await asyncio.gather(*tasks)
        
        # 融合结果
        return self._fuse_results(results_list, weights)
    
    def _calculate_dynamic_weights(
        self,
        query_context: QueryContext
    ) -> RetrievalWeight:
        """计算动态检索权重
        
        Args:
            query_context: 查询上下文
            
        Returns:
            检索权重
        """
        # 基础权重
        weights = RetrievalWeight()
        base_weights = self.default_weights.get(
            query_context.query_type,
            RetrievalWeight(0.4, 0.3, 0.3)
        )
        
        weights.text_weight = base_weights.text_weight
        weights.image_weight = base_weights.image_weight
        weights.table_weight = base_weights.table_weight
        
        # 根据查询需求调整权重
        if query_context.requires_image_search:
            weights.image_weight *= 1.5
        
        if query_context.requires_table_search:
            weights.table_weight *= 1.5
        
        # 根据输入文件调整权重
        if query_context.input_files:
            for file_path in query_context.input_files:
                if file_path.endswith(('.png', '.jpg', '.jpeg')):
                    weights.image_weight *= 1.2
                elif file_path.endswith(('.csv', '.xlsx', '.xls')):
                    weights.table_weight *= 1.2
                elif file_path.endswith(('.pdf', '.docx', '.txt')):
                    weights.text_weight *= 1.2
        
        # 归一化权重
        weights.normalize()
        
        return weights
    
    def _fuse_results(
        self,
        results_list: List[RetrievalResults],
        weights: RetrievalWeight
    ) -> RetrievalResults:
        """融合多个检索结果
        
        Args:
            results_list: 检索结果列表
            weights: 各类型权重
            
        Returns:
            融合后的结果
        """
        fused = RetrievalResults()
        
        # 收集所有结果
        all_texts = []
        all_images = []
        all_tables = []
        all_sources = set()
        max_time = 0
        
        for results in results_list:
            all_texts.extend(results.texts)
            all_images.extend(results.images)
            all_tables.extend(results.tables)
            all_sources.update(results.sources)
            max_time = max(max_time, results.retrieval_time_ms)
        
        # 重新评分和排序
        if all_texts:
            all_texts = self._rescore_items(all_texts, weights.text_weight)
            all_texts.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        
        if all_images:
            all_images = self._rescore_items(all_images, weights.image_weight)
            all_images.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        
        if all_tables:
            all_tables = self._rescore_items(all_tables, weights.table_weight)
            all_tables.sort(key=lambda x: x.get("weighted_score", 0), reverse=True)
        
        # 填充融合结果
        fused.texts = all_texts
        fused.images = all_images
        fused.tables = all_tables
        fused.sources = list(all_sources)
        fused.total_results = len(all_texts) + len(all_images) + len(all_tables)
        fused.retrieval_time_ms = max_time
        
        return fused
    
    def _rescore_items(
        self,
        items: List[Dict[str, Any]],
        weight: float
    ) -> List[Dict[str, Any]]:
        """重新评分项目
        
        Args:
            items: 项目列表
            weight: 权重
            
        Returns:
            重新评分的项目列表
        """
        for item in items:
            original_score = item.get("score", 0.5)
            item["weighted_score"] = original_score * weight
        
        return items
    
    async def _rerank_results(
        self,
        results: RetrievalResults,
        query_context: QueryContext
    ) -> RetrievalResults:
        """重排序结果
        
        Args:
            results: 原始检索结果
            query_context: 查询上下文
            
        Returns:
            重排序后的结果
        """
        # 这里可以使用更复杂的重排序模型
        # 目前使用简单的基于查询相关性的重排序
        
        if results.texts:
            results.texts = await self._rerank_text_results(
                results.texts,
                query_context.query
            )
        
        if results.images:
            results.images = await self._rerank_image_results(
                results.images,
                query_context.query
            )
        
        if results.tables:
            results.tables = await self._rerank_table_results(
                results.tables,
                query_context.query
            )
        
        return results
    
    async def _rerank_text_results(
        self,
        texts: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """重排序文本结果
        
        Args:
            texts: 文本结果列表
            query: 查询文本
            
        Returns:
            重排序后的文本列表
        """
        # 计算额外的相关性特征
        for text in texts:
            content = text.get("content", "").lower()
            query_lower = query.lower()
            
            # 关键词匹配度
            query_words = set(query_lower.split())
            content_words = set(content.split())
            word_overlap = len(query_words & content_words) / max(len(query_words), 1)
            
            # 位置权重（查询词在文本中的位置）
            position_score = 0
            for word in query_words:
                if word in content:
                    pos = content.find(word)
                    # 越靠前权重越高
                    position_score += (1.0 - pos / max(len(content), 1))
            position_score /= max(len(query_words), 1)
            
            # 长度惩罚（过长或过短的文本降低权重）
            optimal_length = 200
            length_penalty = 1.0 - abs(len(content) - optimal_length) / 1000
            length_penalty = max(length_penalty, 0.5)
            
            # 综合评分
            original_score = text.get("score", 0.5)
            text["rerank_score"] = (
                original_score * 0.5 +
                word_overlap * 0.2 +
                position_score * 0.2 +
                length_penalty * 0.1
            )
        
        # 按新分数排序
        texts.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return texts
    
    async def _rerank_image_results(
        self,
        images: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """重排序图像结果
        
        Args:
            images: 图像结果列表
            query: 查询文本
            
        Returns:
            重排序后的图像列表
        """
        # 图像重排序可以基于描述文本的相关性
        for image in images:
            description = image.get("content", "").lower()
            query_lower = query.lower()
            
            # 描述与查询的相关性
            query_words = set(query_lower.split())
            desc_words = set(description.split())
            word_overlap = len(query_words & desc_words) / max(len(query_words), 1)
            
            # 综合评分
            original_score = image.get("score", 0.5)
            image["rerank_score"] = original_score * 0.7 + word_overlap * 0.3
        
        # 按新分数排序
        images.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return images
    
    async def _rerank_table_results(
        self,
        tables: List[Dict[str, Any]],
        query: str
    ) -> List[Dict[str, Any]]:
        """重排序表格结果
        
        Args:
            tables: 表格结果列表
            query: 查询文本
            
        Returns:
            重排序后的表格列表
        """
        # 表格重排序可以基于表头和内容的相关性
        for table in tables:
            description = table.get("content", "").lower()
            
            # 检查是否包含数字（如果查询中有数字）
            import re
            query_numbers = re.findall(r'\d+', query)
            table_numbers = re.findall(r'\d+', description)
            
            number_match = 0
            if query_numbers and table_numbers:
                number_match = len(set(query_numbers) & set(table_numbers)) / len(query_numbers)
            
            # 综合评分
            original_score = table.get("score", 0.5)
            table["rerank_score"] = original_score * 0.8 + number_match * 0.2
        
        # 按新分数排序
        tables.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        
        return tables
    
    def get_retrieval_explanation(
        self,
        results: RetrievalResults,
        query_context: QueryContext
    ) -> Dict[str, Any]:
        """获取检索结果的解释
        
        Args:
            results: 检索结果
            query_context: 查询上下文
            
        Returns:
            检索解释
        """
        explanation = {
            "query_type": query_context.query_type.value,
            "retrieval_strategy": self._get_strategy_name(query_context.query_type),
            "total_results": results.total_results,
            "result_distribution": {
                "texts": len(results.texts),
                "images": len(results.images),
                "tables": len(results.tables)
            },
            "retrieval_time_ms": results.retrieval_time_ms,
            "sources_count": len(results.sources),
            "filters_applied": bool(query_context.filters),
            "reranking_applied": self.config.rerank_enabled
        }
        
        # 添加置信度信息
        if results.texts:
            text_scores = [t.get("score", 0) for t in results.texts]
            explanation["text_confidence"] = {
                "mean": np.mean(text_scores),
                "max": max(text_scores),
                "min": min(text_scores)
            }
        
        if results.images:
            image_scores = [i.get("score", 0) for i in results.images]
            explanation["image_confidence"] = {
                "mean": np.mean(image_scores),
                "max": max(image_scores),
                "min": min(image_scores)
            }
        
        return explanation
    
    def _get_strategy_name(self, query_type: QueryType) -> str:
        """获取策略名称
        
        Args:
            query_type: 查询类型
            
        Returns:
            策略名称
        """
        strategy_names = {
            QueryType.TEXT: "Text-only Retrieval",
            QueryType.VISUAL: "Visual-focused Retrieval",
            QueryType.DOCUMENT: "Document & Table Retrieval",
            QueryType.MIXED: "Hybrid Multimodal Retrieval"
        }
        return strategy_names.get(query_type, "Default Retrieval")
from src.core.logging import get_logger
