"""
GraphRAG核心引擎

GraphRAG系统的主引擎，提供：
- 混合检索策略(向量+图谱)
- 实体感知的查询处理流程
- 上下文扩展和关系追踪算法
- 检索结果融合和重排序机制
"""

import time
import uuid
import asyncio
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from .data_models import (
    GraphRAGRequest,
    GraphRAGResponse,
    GraphRAGConfig,
    QueryType,
    RetrievalMode,
    GraphContext,
    ReasoningPath,
    KnowledgeSource,
    create_empty_graph_context,
    validate_graph_rag_request
)
from .query_analyzer import QueryAnalyzer
from .reasoning_engine import ReasoningEngine
from .knowledge_fusion import KnowledgeFusion
from .cache_manager import CacheManager, get_cache_manager
from ..rag.vector_store import get_vector_store
from ..knowledge_graph.graph_operations import GraphOperations
from ..knowledge_graph.graph_database import get_graph_database
from ..knowledge_graph.schema import get_schema_manager
from ..openai_client import get_openai_client

from src.core.logging import get_logger
logger = get_logger(__name__)

class GraphRAGEngine:
    """GraphRAG核心引擎"""
    
    def __init__(self, config: Optional[GraphRAGConfig] = None):
        self.config = config or GraphRAGConfig()
        
        # 核心组件（延迟初始化）
        self.vector_store = None
        self.knowledge_graph = None
        self.graph_ops = None
        self.openai_client = None
        
        # GraphRAG特定组件
        self.query_analyzer = None
        self.reasoning_engine = None
        self.fusion_engine = None
        self.cache_manager = None
        
        # 初始化标志
        self._initialized = False

    async def initialize(self):
        """初始化GraphRAG引擎"""
        if self._initialized:
            return
        
        try:
            logger.info("开始初始化GraphRAG引擎...")
            
            # 1. 初始化基础组件
            self.vector_store = await get_vector_store()
            self.knowledge_graph = await get_graph_database()
            self.openai_client = get_openai_client()
            
            # 2. 初始化图谱操作
            schema_manager = await get_schema_manager()
            self.graph_ops = GraphOperations(self.knowledge_graph, schema_manager)
            
            # 3. 初始化GraphRAG组件
            self.query_analyzer = QueryAnalyzer(self.graph_ops, self.config)
            self.reasoning_engine = ReasoningEngine(self.graph_ops, self.config)
            self.fusion_engine = KnowledgeFusion(self.config)
            
            # 4. 初始化缓存管理器
            if self.config.enable_caching:
                self.cache_manager = await get_cache_manager()
            
            self._initialized = True
            logger.info("GraphRAG引擎初始化完成")
            
        except Exception as e:
            logger.error(f"GraphRAG引擎初始化失败: {e}")
            raise

    async def enhanced_query(self, request: GraphRAGRequest) -> GraphRAGResponse:
        """GraphRAG增强查询主流程"""
        if not self._initialized:
            await self.initialize()
        
        query_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"开始GraphRAG查询: {request['query'][:100]}...")
            
            # 1. 验证请求
            validation_errors = validate_graph_rag_request(request)
            if validation_errors:
                raise ValueError(f"请求验证失败: {'; '.join(validation_errors)}")
            
            # 2. 查询分析和分解
            decomposition = await self.query_analyzer.analyze_query(
                request['query'], 
                request.get('query_type')
            )
            
            logger.info(f"查询分解完成，策略: {decomposition.decomposition_strategy}")
            
            # 3. 检查缓存
            cached_result = None
            if self.config.enable_caching and self.cache_manager:
                cached_result = await self.cache_manager.get_cached_result(
                    request['query'], 
                    request['retrieval_mode'].value,
                    {"expansion_depth": request['expansion_depth']}
                )
            
            if cached_result and request['retrieval_mode'] != RetrievalMode.ADAPTIVE:
                logger.info("返回缓存的查询结果")
                return cached_result
            
            # 4. 多模式检索
            retrieval_results = await self._multi_modal_retrieve(
                request, decomposition
            )
            
            logger.info(f"检索完成，获得{len(retrieval_results)}类结果")
            
            # 5. 图谱上下文扩展
            graph_context = await self._expand_graph_context(
                retrieval_results, 
                request['expansion_depth']
            )
            
            logger.info(f"图谱上下文扩展完成，实体数: {len(graph_context.entities)}, 关系数: {len(graph_context.relations)}")
            
            # 6. 推理路径生成
            reasoning_time = 0.0
            reasoning_results = []
            if request['include_reasoning'] and self.config.enable_reasoning:
                reasoning_start = time.time()
                reasoning_results = await self.reasoning_engine.generate_reasoning_paths(
                    decomposition,
                    graph_context,
                    max_paths=self.config.max_reasoning_paths
                )
                reasoning_time = time.time() - reasoning_start
                
                logger.info(f"推理路径生成完成，路径数: {len(reasoning_results)}")
            
            # 7. 知识融合
            fusion_start = time.time()
            fusion_results = await self.fusion_engine.fuse_knowledge_sources(
                retrieval_results,
                graph_context,
                reasoning_results,
                confidence_threshold=request['confidence_threshold']
            )
            fusion_time = time.time() - fusion_start
            
            logger.info(f"知识融合完成，最终文档数: {len(fusion_results['final_ranking'])}")
            
            # 8. 构建响应
            total_time = time.time() - start_time
            
            response = GraphRAGResponse(
                query_id=query_id,
                query=request['query'],
                documents=fusion_results['final_ranking'],
                graph_context=graph_context.to_dict(),
                reasoning_results=[path.to_dict() for path in reasoning_results],
                knowledge_sources=[source.to_dict() for source in fusion_results.get('knowledge_sources', [])],
                fusion_results={
                    'conflicts_detected': fusion_results.get('conflicts_detected', []),
                    'resolution_strategy': fusion_results.get('resolution_strategy', ''),
                    'consistency_score': fusion_results.get('consistency_score', 0.0)
                },
                performance_metrics={
                    'total_time': total_time,
                    'retrieval_time': retrieval_results.get('retrieval_time', 0.0),
                    'reasoning_time': reasoning_time,
                    'fusion_time': fusion_time,
                    'cache_hit': cached_result is not None
                },
                timestamp=utc_now().isoformat()
            )
            
            # 9. 缓存结果
            if self.config.enable_caching and self.cache_manager and not cached_result:
                await self.cache_manager.cache_result(
                    request['query'], 
                    response,
                    {"expansion_depth": request['expansion_depth']}
                )
            
            logger.info(f"GraphRAG查询完成，耗时: {total_time:.2f}秒")
            return response
            
        except Exception as e:
            logger.error(f"GraphRAG查询失败: {e}")
            # 降级到传统RAG
            return await self._fallback_to_traditional_rag(request, query_id)

    async def _multi_modal_retrieve(
        self, 
        request: GraphRAGRequest, 
        decomposition
    ) -> Dict[str, Any]:
        """多模式检索"""
        start_time = time.time()
        results = {}
        
        try:
            retrieval_mode = request['retrieval_mode']
            max_docs = request['max_docs']
            
            # 1. 向量检索
            if retrieval_mode in [RetrievalMode.VECTOR_ONLY, RetrievalMode.HYBRID, RetrievalMode.ADAPTIVE]:
                vector_results = await self._vector_retrieve(
                    request['query'], 
                    min(max_docs, self.config.vector_search_limit)
                )
                results['vector'] = vector_results
                logger.info(f"向量检索完成，获得{len(vector_results)}个结果")
            
            # 2. 图谱检索
            if retrieval_mode in [RetrievalMode.GRAPH_ONLY, RetrievalMode.HYBRID, RetrievalMode.ADAPTIVE]:
                graph_results = await self._graph_retrieve(
                    decomposition.entity_queries,
                    decomposition.relation_queries,
                    max_docs
                )
                results['graph'] = graph_results
                logger.info(f"图谱检索完成，获得实体数: {len(graph_results.get('entities', []))}, 关系数: {len(graph_results.get('relations', []))}")
            
            # 3. 自适应检索策略
            if retrieval_mode == RetrievalMode.ADAPTIVE:
                results = await self._adaptive_retrieve(request, decomposition, results)
            
            results['retrieval_time'] = time.time() - start_time
            return results
            
        except Exception as e:
            logger.error(f"多模式检索失败: {e}")
            return {'retrieval_time': time.time() - start_time}

    async def _vector_retrieve(self, query: str, limit: int) -> List[Dict[str, Any]]:
        """向量检索"""
        try:
            # 使用默认集合名称，如果没有配置的话
            collection_name = getattr(self.vector_store.settings, 'DEFAULT_COLLECTION', 'documents')
            
            # 执行相似性搜索
            results = await self.vector_store.similarity_search(
                collection_name=collection_name,
                query_vector=await self._get_query_embedding(query),
                limit=limit,
                include_distances=True
            )
            
            # 转换结果格式
            formatted_results = []
            for result in results:
                formatted_results.append({
                    'id': result.get('id', ''),
                    'content': result.get('content', ''),
                    'score': 1.0 - result.get('distance', 0.0),  # 转换距离为相似度分数
                    'metadata': result.get('metadata', {}),
                    'source': 'vector_store'
                })
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    async def _get_query_embedding(self, query: str) -> List[float]:
        """获取查询的嵌入向量"""
        try:
            # 使用OpenAI嵌入模型
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=query
            )
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"获取查询嵌入失败: {e}")
            # 返回零向量作为后备
            return [0.0] * 1536  # OpenAI text-embedding-3-small 的维度

    async def _graph_retrieve(
        self,
        entity_queries: List[Dict[str, Any]],
        relation_queries: List[Dict[str, Any]],
        max_results: int
    ) -> Dict[str, Any]:
        """图谱检索"""
        try:
            entities = []
            relations = []
            
            # 1. 实体查询
            for entity_query in entity_queries[:5]:  # 限制查询数量
                entity_name = entity_query.get('entity', '')
                if entity_name:
                    # 查找实体
                    result = await self.graph_ops.find_entities({
                        'canonical_form_contains': entity_name
                    }, limit=10)
                    
                    if result.success:
                        entities.extend(result.data)
                    
                    # 获取实体关系
                    if entities:
                        for entity_data in result.data[:3]:  # 限制每个实体的关系数量
                            entity_id = entity_data.get('id')
                            if entity_id:
                                relations_result = await self.graph_ops.get_entity_relationships(
                                    entity_id, limit=10
                                )
                                if relations_result.success:
                                    relations.extend(relations_result.data)
            
            # 2. 关系查询
            for relation_query in relation_queries[:5]:  # 限制查询数量
                entity1 = relation_query.get('entity1', '')
                entity2 = relation_query.get('entity2', '')
                max_hops = relation_query.get('max_hops', 3)
                
                if entity1 and entity2:
                    # 查找最短路径
                    path_result = await self.graph_ops.find_shortest_path(
                        entity1, entity2, max_depth=max_hops
                    )
                    
                    if path_result.success:
                        for path_data in path_result.data:
                            if 'nodes' in path_data:
                                entities.extend(path_data['nodes'])
                            if 'relationships' in path_data:
                                relations.extend(path_data['relationships'])
            
            # 限制结果数量
            entities = entities[:max_results//2]
            relations = relations[:max_results//2]
            
            return {
                'entities': entities,
                'relations': relations,
                'source': 'knowledge_graph'
            }
            
        except Exception as e:
            logger.error(f"图谱检索失败: {e}")
            return {'entities': [], 'relations': []}

    async def _adaptive_retrieve(
        self,
        request: GraphRAGRequest,
        decomposition,
        existing_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """自适应检索策略"""
        try:
            # 基于查询复杂度选择策略
            complexity = decomposition.complexity_score
            
            if complexity < 0.3:
                # 简单查询，优先使用向量检索
                if 'vector' not in existing_results:
                    existing_results['vector'] = await self._vector_retrieve(
                        request['query'], 
                        request['max_docs']
                    )
            elif complexity > 0.7:
                # 复杂查询，加强图谱检索
                if 'graph' not in existing_results:
                    existing_results['graph'] = await self._graph_retrieve(
                        decomposition.entity_queries,
                        decomposition.relation_queries,
                        request['max_docs']
                    )
                
                # 增加子图检索
                subgraph_results = await self._retrieve_subgraphs(decomposition)
                existing_results['subgraph'] = subgraph_results
            
            return existing_results
            
        except Exception as e:
            logger.error(f"自适应检索失败: {e}")
            return existing_results

    async def _retrieve_subgraphs(self, decomposition) -> List[Dict[str, Any]]:
        """检索相关子图"""
        try:
            subgraphs = []
            
            # 为每个实体检索子图
            for entity_query in decomposition.entity_queries[:3]:
                entity_name = entity_query.get('entity', '')
                if entity_name:
                    # 寻找实体ID
                    entity_result = await self.graph_ops.find_entities({
                        'canonical_form_contains': entity_name
                    }, limit=1)
                    
                    if entity_result.success and entity_result.data:
                        entity_id = entity_result.data[0].get('id')
                        if entity_id:
                            # 获取子图
                            subgraph_result = await self.graph_ops.get_subgraph(
                                entity_id, 
                                depth=2,
                                max_nodes=50
                            )
                            
                            if subgraph_result.success:
                                subgraphs.extend(subgraph_result.data)
            
            return subgraphs
            
        except Exception as e:
            logger.error(f"子图检索失败: {e}")
            return []

    async def _expand_graph_context(
        self, 
        retrieval_results: Dict[str, Any], 
        max_depth: int
    ) -> GraphContext:
        """扩展图谱上下文"""
        try:
            entities = []
            relations = []
            
            # 1. 从检索结果中提取实体
            # 向量检索结果
            if 'vector' in retrieval_results:
                for doc in retrieval_results['vector']:
                    # 使用简单的实体识别从文档内容中提取实体
                    doc_entities = await self._extract_entities_from_text(doc.get('content', ''))
                    entities.extend(doc_entities)
            
            # 图谱检索结果
            if 'graph' in retrieval_results:
                graph_data = retrieval_results['graph']
                entities.extend(graph_data.get('entities', []))
                relations.extend(graph_data.get('relations', []))
            
            # 子图检索结果
            if 'subgraph' in retrieval_results:
                for subgraph in retrieval_results['subgraph']:
                    if 'nodes' in subgraph:
                        entities.extend(subgraph['nodes'])
                    if 'relationships' in subgraph:
                        relations.extend(subgraph['relationships'])
            
            # 2. 执行上下文扩展
            expanded_entities = []
            expanded_relations = []
            
            for depth in range(1, max_depth + 1):
                current_expansion = await self._expand_entities_at_depth(
                    entities[-10:], depth  # 只处理最近的10个实体
                )
                expanded_entities.extend(current_expansion['entities'])
                expanded_relations.extend(current_expansion['relations'])
                
                if len(expanded_entities) > self.config.graph_traversal_limit:
                    break
            
            # 3. 构建子图
            all_entities = entities + expanded_entities
            all_relations = relations + expanded_relations
            
            subgraph = await self._build_subgraph(all_entities, all_relations)
            
            # 4. 计算置信度
            confidence_score = self._calculate_context_confidence(
                entities, relations, expanded_entities, expanded_relations
            )
            
            return GraphContext(
                entities=all_entities[:100],  # 限制实体数量
                relations=all_relations[:200], # 限制关系数量
                subgraph=subgraph,
                reasoning_paths=[],
                expansion_depth=max_depth,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            logger.error(f"图谱上下文扩展失败: {e}")
            return create_empty_graph_context()

    async def _extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """从文本中提取实体"""
        try:
            # 简单的实体提取逻辑（可以替换为更复杂的NER模型）
            import re
            
            entities = []
            
            # 提取专有名词
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
            
            for noun in set(proper_nouns):  # 去重
                entities.append({
                    'canonical_form': noun,
                    'text': noun,
                    'type': 'CONCEPT',
                    'confidence': 0.6,
                    'source': 'text_extraction'
                })
            
            return entities[:10]  # 限制数量
            
        except Exception as e:
            logger.error(f"文本实体提取失败: {e}")
            return []

    async def _expand_entities_at_depth(
        self, 
        entities: List[Dict[str, Any]], 
        depth: int
    ) -> Dict[str, Any]:
        """在指定深度扩展实体"""
        try:
            expanded_entities = []
            expanded_relations = []
            
            for entity in entities:
                entity_id = entity.get('id')
                if not entity_id:
                    # 尝试通过canonical_form查找实体ID
                    canonical_form = entity.get('canonical_form', entity.get('text', ''))
                    if canonical_form:
                        search_result = await self.graph_ops.find_entities({
                            'canonical_form': canonical_form
                        }, limit=1)
                        
                        if search_result.success and search_result.data:
                            entity_id = search_result.data[0].get('id')
                
                if entity_id:
                    # 获取相关实体和关系
                    relations_result = await self.graph_ops.get_entity_relationships(
                        entity_id, limit=5  # 限制每个实体的关系数量
                    )
                    
                    if relations_result.success:
                        expanded_relations.extend(relations_result.data)
            
            return {
                'entities': expanded_entities,
                'relations': expanded_relations
            }
            
        except Exception as e:
            logger.error(f"实体深度扩展失败: {e}")
            return {'entities': [], 'relations': []}

    async def _build_subgraph(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """构建子图"""
        try:
            # 构建简化的子图表示
            subgraph = {
                'nodes': entities[:50],  # 限制节点数量
                'edges': relations[:100], # 限制边数量
                'statistics': {
                    'node_count': len(entities),
                    'edge_count': len(relations),
                    'density': len(relations) / max(1, len(entities)) if entities else 0
                }
            }
            
            return subgraph
            
        except Exception as e:
            logger.error(f"子图构建失败: {e}")
            return {'nodes': [], 'edges': [], 'statistics': {}}

    def _calculate_context_confidence(
        self,
        original_entities: List[Dict[str, Any]],
        original_relations: List[Dict[str, Any]], 
        expanded_entities: List[Dict[str, Any]],
        expanded_relations: List[Dict[str, Any]]
    ) -> float:
        """计算上下文置信度"""
        try:
            # 基础置信度
            base_confidence = 0.5
            
            # 原始实体和关系的置信度加成
            if original_entities:
                entity_confidences = [
                    e.get('confidence', 0.5) for e in original_entities
                ]
                base_confidence += sum(entity_confidences) / len(entity_confidences) * 0.3
            
            if original_relations:
                relation_confidences = [
                    r.get('confidence', 0.5) for r in original_relations
                ]
                base_confidence += sum(relation_confidences) / len(relation_confidences) * 0.2
            
            # 扩展成功率加成
            expansion_success = len(expanded_entities) / max(1, len(original_entities))
            base_confidence += min(0.3, expansion_success * 0.1)
            
            return min(1.0, base_confidence)
            
        except Exception as e:
            logger.error(f"置信度计算失败: {e}")
            return 0.5

    async def _fallback_to_traditional_rag(
        self, 
        request: GraphRAGRequest, 
        query_id: str
    ) -> GraphRAGResponse:
        """降级到传统RAG"""
        try:
            logger.info("降级到传统RAG模式")
            
            # 执行简单的向量检索
            vector_results = await self._vector_retrieve(
                request['query'], 
                request['max_docs']
            )
            
            return GraphRAGResponse(
                query_id=query_id,
                query=request['query'],
                documents=vector_results,
                graph_context=create_empty_graph_context().to_dict(),
                reasoning_results=[],
                knowledge_sources=[],
                fusion_results={
                    'conflicts_detected': [],
                    'resolution_strategy': 'fallback',
                    'consistency_score': 0.0
                },
                performance_metrics={
                    'total_time': 0.0,
                    'retrieval_time': 0.0,
                    'reasoning_time': 0.0,
                    'fusion_time': 0.0,
                    'fallback': True
                },
                timestamp=utc_now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"传统RAG降级也失败: {e}")
            # 返回空结果
            return GraphRAGResponse(
                query_id=query_id,
                query=request['query'],
                documents=[],
                graph_context=create_empty_graph_context().to_dict(),
                reasoning_results=[],
                knowledge_sources=[],
                fusion_results={
                    'conflicts_detected': [],
                    'resolution_strategy': 'empty_fallback',
                    'consistency_score': 0.0
                },
                performance_metrics={
                    'total_time': 0.0,
                    'retrieval_time': 0.0,
                    'reasoning_time': 0.0,
                    'fusion_time': 0.0,
                    'error': True
                },
                timestamp=utc_now().isoformat()
            )

    async def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        try:
            stats = {
                'engine_status': 'initialized' if self._initialized else 'not_initialized',
                'config': self.config.to_dict(),
                'components': {
                    'vector_store': self.vector_store is not None,
                    'knowledge_graph': self.knowledge_graph is not None,
                    'query_analyzer': self.query_analyzer is not None,
                    'reasoning_engine': self.reasoning_engine is not None,
                    'fusion_engine': self.fusion_engine is not None,
                    'cache_manager': self.cache_manager is not None
                }
            }
            
            # 缓存统计
            if self.cache_manager:
                stats['cache_stats'] = self.cache_manager.get_cache_stats()
            
            # 图谱统计
            if self.graph_ops:
                graph_stats_result = await self.graph_ops.get_graph_statistics()
                if graph_stats_result.success:
                    stats['graph_stats'] = graph_stats_result.data
            
            return stats
            
        except Exception as e:
            logger.error(f"获取性能统计失败: {e}")
            return {'error': str(e)}

    async def close(self):
        """关闭引擎"""
        try:
            if self.cache_manager:
                await self.cache_manager.close()
            
            if self.knowledge_graph:
                await self.knowledge_graph.close()
            
            if self.vector_store:
                await self.vector_store.close_pool()
            
            logger.info("GraphRAG引擎已关闭")
            
        except Exception as e:
            logger.error(f"关闭GraphRAG引擎失败: {e}")

# 全局GraphRAG引擎实例
_graphrag_engine_instance: Optional[GraphRAGEngine] = None

async def get_graphrag_engine(config: Optional[GraphRAGConfig] = None) -> GraphRAGEngine:
    """获取GraphRAG引擎实例（单例模式）"""
    global _graphrag_engine_instance
    
    if _graphrag_engine_instance is None:
        _graphrag_engine_instance = GraphRAGEngine(config)
        await _graphrag_engine_instance.initialize()
    
    return _graphrag_engine_instance

async def close_graphrag_engine():
    """关闭GraphRAG引擎"""
    global _graphrag_engine_instance
    
    if _graphrag_engine_instance:
        await _graphrag_engine_instance.close()
        _graphrag_engine_instance = None
