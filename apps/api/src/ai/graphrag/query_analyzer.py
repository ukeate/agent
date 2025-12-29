"""
GraphRAG查询分析和分解器

提供基于图谱的查询分析能力：
- 自动检测查询类型
- 实体识别和规范化
- 基于图谱的问题分解
- 多跳查询计划生成
- 查询复杂度评估
"""

import re
from typing import List, Dict, Any, Optional, Pattern, Tuple
from dataclasses import asdict
from .data_models import (
    QueryType, 
    QueryDecomposition, 
    EntityRecognitionResult,
    GraphRAGConfig
)
from ..openai_client import get_openai_client

from src.core.logging import get_logger
logger = get_logger(__name__)

# 可选导入知识图谱组件
try:
    from ..knowledge_graph.graph_operations import GraphOperations
    from ..knowledge_graph.data_models import Entity, EntityType
    KNOWLEDGE_GRAPH_AVAILABLE = True
except ImportError:
    logger.warning("Knowledge graph components not available, using fallback implementations")
    GraphOperations = None
    Entity = None
    EntityType = None
    KNOWLEDGE_GRAPH_AVAILABLE = False

class QueryAnalyzer:
    """查询分析器"""
    
    def __init__(self, graph_operations: Optional[GraphOperations] = None, config: Optional[GraphRAGConfig] = None):
        self.graph_ops = graph_operations if KNOWLEDGE_GRAPH_AVAILABLE else None
        self.config = config or GraphRAGConfig()
        self.openai_client = get_openai_client()
        
        # 查询模式正则表达式
        self.query_patterns = {
            'simple_entity': re.compile(r'\b(?:who|what|where|when)\s+(?:is|are|was|were)\s+(\w+)', re.IGNORECASE),
            'relational': re.compile(r'\b(\w+(?:\s+\w+)*)\s+(?:and|with|related to|relationship between)\s+(\w+(?:\s+\w+)*)', re.IGNORECASE),
            'multi_hop': re.compile(r'\b(\w+)\s+(?:through|via)\s+(\w+)\s+(?:to|with)\s+(\w+)', re.IGNORECASE),
            'complex': re.compile(r'\b(?:how|why|explain|reason|cause|effect|impact)', re.IGNORECASE),
            'comparison': re.compile(r'\b(?:compare|difference|similar|versus|vs)\b', re.IGNORECASE),
            'temporal': re.compile(r'\b(?:when|before|after|during|since|until|timeline)\b', re.IGNORECASE),
            'causal': re.compile(r'\b(?:because|due to|leads to|results in|caused by|affects)\b', re.IGNORECASE)
        }
        
        # 实体类型提示词 - 使用字符串而不是EntityType枚举作为fallback
        if KNOWLEDGE_GRAPH_AVAILABLE and EntityType:
            self.entity_type_keywords = {
                EntityType.PERSON: ['person', 'people', 'individual', 'human', 'character'],
                EntityType.ORGANIZATION: ['company', 'organization', 'institution', 'agency', 'corp'],
                EntityType.LOCATION: ['place', 'location', 'city', 'country', 'region', 'area'],
                EntityType.CONCEPT: ['concept', 'idea', 'theory', 'principle', 'notion'],
                EntityType.EVENT: ['event', 'happening', 'occurrence', 'incident', 'situation'],
                EntityType.OBJECT: ['object', 'thing', 'item', 'product', 'device', 'tool'],
                EntityType.TIME: ['time', 'date', 'period', 'era', 'moment', 'duration']
            }
        else:
            # Fallback实体类型关键词（使用字符串）
            self.entity_type_keywords = {
                'PERSON': ['person', 'people', 'individual', 'human', 'character'],
                'ORGANIZATION': ['company', 'organization', 'institution', 'agency', 'corp'],
                'LOCATION': ['place', 'location', 'city', 'country', 'region', 'area'],
                'CONCEPT': ['concept', 'idea', 'theory', 'principle', 'notion'],
                'EVENT': ['event', 'happening', 'occurrence', 'incident', 'situation'],
                'OBJECT': ['object', 'thing', 'item', 'product', 'device', 'tool'],
                'TIME': ['time', 'date', 'period', 'era', 'moment', 'duration']
            }

    async def analyze_query(
        self, 
        query: str, 
        query_type: Optional[QueryType] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> QueryDecomposition:
        """分析和分解查询"""
        try:
            logger.info(f"开始分析查询: {query[:100]}...")
            
            # 1. 自动检测查询类型
            if not query_type:
                query_type = self._detect_query_type(query)
            
            # 2. 识别查询中的实体
            entities = await self._extract_entities(query)
            
            # 3. 基于查询类型进行分解
            if query_type == QueryType.SIMPLE:
                decomposition = await self._decompose_simple_query(query, entities)
            elif query_type == QueryType.MULTI_ENTITY:
                decomposition = await self._decompose_multi_entity_query(query, entities)
            elif query_type == QueryType.RELATIONAL:
                decomposition = await self._decompose_relational_query(query, entities)
            elif query_type == QueryType.COMPLEX_REASONING:
                decomposition = await self._decompose_complex_query(query, entities)
            else:  # COMPOSITIONAL
                decomposition = await self._decompose_compositional_query(query, entities)
            
            # 4. 计算查询复杂度
            complexity_score = self._calculate_complexity_score(decomposition)
            decomposition.complexity_score = complexity_score
            
            logger.info(f"查询分析完成，类型: {query_type}, 复杂度: {complexity_score:.2f}")
            return decomposition
            
        except Exception as e:
            logger.error(f"查询分析失败: {e}")
            # 返回基础的分解结果
            return QueryDecomposition(
                original_query=query,
                sub_queries=[query],
                entity_queries=[],
                relation_queries=[],
                decomposition_strategy="fallback",
                complexity_score=0.5
            )

    def _detect_query_type(self, query: str) -> QueryType:
        """检测查询类型"""
        query_lower = query.lower().strip()
        
        # 检查复杂推理关键词
        if any(keyword in query_lower for keyword in [
            'how', 'why', 'explain', 'reason', 'cause', 'effect', 'impact',
            'analyze', 'evaluate', 'assess', 'compare and contrast'
        ]):
            return QueryType.COMPLEX_REASONING
        
        # 检查关系型查询
        if any(keyword in query_lower for keyword in [
            'relationship', 'related', 'connection', 'between', 'link',
            'associate', 'correlate', 'interact'
        ]):
            return QueryType.RELATIONAL
        
        # 检查多跳查询模式
        if self.query_patterns['multi_hop'].search(query):
            return QueryType.RELATIONAL
        
        # 检查组合查询
        if any(keyword in query_lower for keyword in [
            'and', 'or', 'both', 'either', 'compare', 'contrast', 
            'versus', 'vs', 'as well as', 'along with'
        ]):
            return QueryType.COMPOSITIONAL
        
        # 简单实体计数检查
        potential_entities = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query))
        if potential_entities > 2:
            return QueryType.MULTI_ENTITY
        
        return QueryType.SIMPLE

    async def _extract_entities(self, query: str) -> List[EntityRecognitionResult]:
        """从查询中提取实体"""
        try:
            # 使用简单的规则基础实体识别（可以扩展为ML模型）
            entities = []
            
            # 1. 识别专有名词（首字母大写的词组）
            proper_noun_pattern = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
            for match in proper_noun_pattern.finditer(query):
                entity_text = match.group()
                entities.append(EntityRecognitionResult(
                    text=entity_text,
                    canonical_form=entity_text,  # 这里可以加入实体链接逻辑
                    entity_type=self._infer_entity_type(entity_text, query),
                    confidence=0.8,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    metadata={"extraction_method": "proper_noun"}
                ))
            
            # 2. 识别引号中的实体
            quoted_pattern = re.compile(r'"([^"]+)"')
            for match in quoted_pattern.finditer(query):
                entity_text = match.group(1)
                entities.append(EntityRecognitionResult(
                    text=entity_text,
                    canonical_form=entity_text,
                    entity_type="CONCEPT",
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    metadata={"extraction_method": "quoted"}
                ))
            
            # 3. 在知识图谱中查找已知实体
            await self._enrich_entities_from_graph(entities, query)
            
            # 4. 去重和排序
            entities = self._deduplicate_entities(entities)
            
            logger.info(f"提取到{len(entities)}个实体")
            return entities
            
        except Exception as e:
            logger.error(f"实体提取失败: {e}")
            return []

    def _infer_entity_type(self, entity_text: str, context: str) -> Optional[str]:
        """推断实体类型"""
        entity_lower = entity_text.lower()
        context_lower = context.lower()
        
        # 基于关键词匹配推断类型
        for entity_type, keywords in self.entity_type_keywords.items():
            if any(keyword in context_lower for keyword in keywords):
                return entity_type.value if hasattr(entity_type, 'value') else entity_type
        
        # 基于实体文本特征推断
        if re.match(r'\b\d{4}\b', entity_text):  # 年份
            return "TIME" if not KNOWLEDGE_GRAPH_AVAILABLE else EntityType.TIME.value
        elif re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', entity_text):  # 人名模式
            return "PERSON" if not KNOWLEDGE_GRAPH_AVAILABLE else EntityType.PERSON.value
        elif entity_text.endswith(('Inc', 'Corp', 'LLC', 'Ltd')):
            return "ORGANIZATION" if not KNOWLEDGE_GRAPH_AVAILABLE else EntityType.ORGANIZATION.value
        
        return "CONCEPT" if not KNOWLEDGE_GRAPH_AVAILABLE else EntityType.CONCEPT.value  # 默认为概念

    async def _enrich_entities_from_graph(self, entities: List[EntityRecognitionResult], query: str):
        """从知识图谱中丰富实体信息"""
        try:
            # 如果知识图谱不可用，跳过实体丰富
            if not self.graph_ops:
                logger.info("Knowledge graph not available, skipping entity enrichment")
                return
                
            for entity in entities:
                # 在图谱中查找匹配的实体
                search_result = await self.graph_ops.find_entities({
                    "canonical_form_contains": entity.text
                }, limit=5)
                
                if search_result.success and search_result.data:
                    # 找到匹配的实体，更新信息
                    best_match = search_result.data[0]  # 取第一个匹配
                    entity.canonical_form = best_match.get("canonical_form", entity.text)
                    entity.entity_type = best_match.get("type", entity.entity_type)
                    entity.confidence = min(1.0, entity.confidence + 0.1)  # 提高置信度
                    entity.metadata["graph_id"] = best_match.get("id")
                    
        except Exception as e:
            logger.warning(f"从图谱丰富实体信息失败: {e}")

    def _deduplicate_entities(self, entities: List[EntityRecognitionResult]) -> List[EntityRecognitionResult]:
        """实体去重"""
        seen = set()
        deduplicated = []
        
        for entity in entities:
            key = entity.canonical_form.lower() if entity.canonical_form else entity.text.lower()
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
        
        # 按置信度排序
        return sorted(deduplicated, key=lambda x: x.confidence, reverse=True)

    async def _decompose_simple_query(
        self, 
        query: str, 
        entities: List[EntityRecognitionResult]
    ) -> QueryDecomposition:
        """分解简单查询"""
        sub_queries = [query]  # 简单查询通常不需要分解
        entity_queries = []
        relation_queries = []
        
        # 为每个实体生成详细查询
        for entity in entities[:3]:  # 限制前3个实体
            entity_query = {
                "entity": entity.canonical_form or entity.text,
                "properties": ["all"],
                "expand_relations": True,
                "confidence_threshold": 0.5
            }
            entity_queries.append(entity_query)
        
        return QueryDecomposition(
            original_query=query,
            sub_queries=sub_queries,
            entity_queries=entity_queries,
            relation_queries=relation_queries,
            decomposition_strategy="simple",
            complexity_score=0.0
        )

    async def _decompose_multi_entity_query(
        self, 
        query: str, 
        entities: List[EntityRecognitionResult]
    ) -> QueryDecomposition:
        """分解多实体查询"""
        sub_queries = []
        entity_queries = []
        relation_queries = []
        
        # 为每个实体生成子查询
        for entity in entities[:5]:  # 限制前5个实体
            sub_query = f"What information is available about {entity.text}?"
            sub_queries.append(sub_query)
            
            entity_query = {
                "entity": entity.canonical_form or entity.text,
                "properties": ["all"],
                "expand_relations": True,
                "confidence_threshold": 0.4
            }
            entity_queries.append(entity_query)
        
        # 生成实体间关系查询
        for i, entity1 in enumerate(entities[:3]):
            for entity2 in entities[i+1:4]:  # 避免过多组合
                relation_query = {
                    "entity1": entity1.canonical_form or entity1.text,
                    "entity2": entity2.canonical_form or entity2.text,
                    "relation_types": ["all"],
                    "max_hops": 2
                }
                relation_queries.append(relation_query)
        
        return QueryDecomposition(
            original_query=query,
            sub_queries=sub_queries,
            entity_queries=entity_queries,
            relation_queries=relation_queries,
            decomposition_strategy="multi_entity",
            complexity_score=0.0
        )

    async def _decompose_relational_query(
        self, 
        query: str, 
        entities: List[EntityRecognitionResult]
    ) -> QueryDecomposition:
        """分解关系型查询"""
        sub_queries = []
        entity_queries = []
        relation_queries = []
        
        # 为每个实体对生成关系查询
        for i, entity1 in enumerate(entities[:4]):
            for entity2 in entities[i+1:5]:
                # 查找实体间的关系
                relation_query = {
                    "entity1": entity1.canonical_form or entity1.text,
                    "entity2": entity2.canonical_form or entity2.text,
                    "relation_types": ["all"],
                    "max_hops": 3,
                    "bidirectional": True
                }
                relation_queries.append(relation_query)
                
                # 生成子查询
                sub_query = f"What is the relationship between {entity1.text} and {entity2.text}?"
                sub_queries.append(sub_query)
        
        # 为每个实体生成详细查询
        for entity in entities[:4]:
            entity_query = {
                "entity": entity.canonical_form or entity.text,
                "properties": ["all"],
                "expand_relations": True,
                "relation_depth": 2
            }
            entity_queries.append(entity_query)
        
        return QueryDecomposition(
            original_query=query,
            sub_queries=sub_queries,
            entity_queries=entity_queries,
            relation_queries=relation_queries,
            decomposition_strategy="relational",
            complexity_score=0.0
        )

    async def _decompose_complex_query(
        self, 
        query: str, 
        entities: List[EntityRecognitionResult]
    ) -> QueryDecomposition:
        """分解复杂推理查询"""
        try:
            # 使用LLM来分解复杂查询
            sub_queries = await self._llm_decompose_query(query, entities)
            
            entity_queries = []
            relation_queries = []
            
            # 为每个实体生成深度查询
            for entity in entities[:3]:
                entity_query = {
                    "entity": entity.canonical_form or entity.text,
                    "properties": ["all"],
                    "expand_relations": True,
                    "relation_depth": 3,
                    "include_reasoning": True
                }
                entity_queries.append(entity_query)
            
            # 生成多跳关系查询
            for i, entity1 in enumerate(entities[:3]):
                for entity2 in entities[i+1:4]:
                    relation_query = {
                        "entity1": entity1.canonical_form or entity1.text,
                        "entity2": entity2.canonical_form or entity2.text,
                        "relation_types": ["all"],
                        "max_hops": 4,
                        "include_reasoning_paths": True
                    }
                    relation_queries.append(relation_query)
            
            return QueryDecomposition(
                original_query=query,
                sub_queries=sub_queries,
                entity_queries=entity_queries,
                relation_queries=relation_queries,
                decomposition_strategy="complex_reasoning",
                complexity_score=0.0
            )
            
        except Exception as e:
            logger.warning(f"复杂查询分解失败，使用基础策略: {e}")
            return await self._decompose_relational_query(query, entities)

    async def _decompose_compositional_query(
        self, 
        query: str, 
        entities: List[EntityRecognitionResult]
    ) -> QueryDecomposition:
        """分解组合查询"""
        # 分析查询中的逻辑连接词
        logical_connectors = ['and', 'or', 'but', 'however', 'although', 'while']
        
        sub_queries = []
        
        # 尝试根据逻辑连接词分割查询
        for connector in logical_connectors:
            if connector in query.lower():
                parts = query.lower().split(connector)
                if len(parts) > 1:
                    for part in parts:
                        part = part.strip()
                        if part:
                            sub_queries.append(part.capitalize())
                break
        
        # 如果没有找到明显的分割点，使用句号或分号分割
        if not sub_queries:
            parts = re.split(r'[.;]', query)
            sub_queries = [part.strip() for part in parts if part.strip()]
        
        # 如果还是没有分割，将原查询作为单个子查询
        if not sub_queries:
            sub_queries = [query]
        
        entity_queries = []
        relation_queries = []
        
        # 为实体生成查询
        for entity in entities[:4]:
            entity_query = {
                "entity": entity.canonical_form or entity.text,
                "properties": ["all"],
                "expand_relations": True
            }
            entity_queries.append(entity_query)
        
        return QueryDecomposition(
            original_query=query,
            sub_queries=sub_queries,
            entity_queries=entity_queries,
            relation_queries=relation_queries,
            decomposition_strategy="compositional",
            complexity_score=0.0
        )

    async def _llm_decompose_query(
        self, 
        query: str, 
        entities: List[EntityRecognitionResult]
    ) -> List[str]:
        """使用LLM分解查询"""
        try:
            entity_names = [e.text for e in entities[:5]]
            
            prompt = f"""
请将以下复杂查询分解为更简单的子查询。

原始查询: {query}
识别到的实体: {', '.join(entity_names)}

请按照以下格式分解查询：
1. 每个子查询应该专注于一个特定的方面
2. 子查询应该能够独立回答
3. 返回2-5个子查询

请直接返回子查询列表，每行一个，格式如下：
- 子查询1
- 子查询2
- 子查询3
"""
            
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个查询分解专家，能够将复杂查询分解为简单的子查询。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # 解析响应
            sub_queries = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('-'):
                    sub_query = line[1:].strip()
                    if sub_query:
                        sub_queries.append(sub_query)
            
            return sub_queries if sub_queries else [query]
            
        except Exception as e:
            logger.warning(f"LLM查询分解失败: {e}")
            return [query]

    def _calculate_complexity_score(self, decomposition: QueryDecomposition) -> float:
        """计算查询复杂度评分"""
        score = 0.0
        
        # 基于子查询数量
        score += min(len(decomposition.sub_queries) * 0.1, 0.3)
        
        # 基于实体查询数量
        score += min(len(decomposition.entity_queries) * 0.05, 0.2)
        
        # 基于关系查询数量
        score += min(len(decomposition.relation_queries) * 0.1, 0.3)
        
        # 基于分解策略
        strategy_scores = {
            "simple": 0.1,
            "multi_entity": 0.3,
            "relational": 0.5,
            "complex_reasoning": 0.8,
            "compositional": 0.6,
            "fallback": 0.2
        }
        score += strategy_scores.get(decomposition.decomposition_strategy, 0.2)
        
        # 基于原始查询的语法复杂度
        query = decomposition.original_query
        if any(word in query.lower() for word in ['how', 'why', 'explain']):
            score += 0.2
        if len(query.split()) > 10:
            score += 0.1
        if '?' in query:
            score += 0.05
        
        return min(score, 1.0)

    async def optimize_query_plan(self, decomposition: QueryDecomposition) -> QueryDecomposition:
        """优化查询计划"""
        try:
            # 根据图谱统计信息优化查询顺序
            # 优先执行选择性高的查询
            
            # 重排实体查询 - 高置信度实体优先
            if decomposition.entity_queries:
                # 这里可以添加基于图谱统计的排序逻辑
                decomposition.entity_queries.sort(
                    key=lambda x: x.get("confidence", 0.0),
                    reverse=True
                )
            
            # 重排关系查询 - 短路径优先
            if decomposition.relation_queries:
                # 按照期望的跳数排序
                decomposition.relation_queries.sort(
                    key=lambda x: x.get('max_hops', 3)
                )
            
            return decomposition
            
        except Exception as e:
            logger.warning(f"查询计划优化失败: {e}")
            return decomposition
