# Epic 8: 动态知识图谱系统

**Epic ID**: EPIC-008-DYNAMIC-KNOWLEDGE-GRAPH  
**优先级**: 高 (P1)  
**预估工期**: 10-12周  
**负责团队**: AI团队 + 后端团队  
**创建日期**: 2025-08-19

## 📋 Epic概述

构建动态知识图谱系统，实现实体识别与关系抽取、动态图谱构建与更新、图谱推理与查询，以及与RAG系统的深度融合(GraphRAG)，让AI Agent具备结构化知识表示、推理和动态学习能力。

### 🎯 业务价值
- **结构化知识**: 将非结构化文本转换为可推理的知识图谱
- **动态更新**: 实时学习和更新知识，保持知识库的时效性
- **深度推理**: 基于图结构的多跳推理和关联分析
- **技术竞争力**: 掌握知识图谱和GraphRAG的先进技术

## 🚀 核心功能清单

### 1. **实体识别与关系抽取(NER+RE)**
- 命名实体识别(人物、地点、组织、时间等)
- 关系三元组抽取(主语-谓语-宾语)
- 实体链接和消歧
- 多语言实体抽取支持

### 2. **动态知识图谱构建**
- 增量式图谱构建
- 实体和关系的动态更新
- 知识冲突检测和解决
- 图谱质量评估和优化

### 3. **图谱推理引擎**
- 基于规则的推理(SWRL)
- 基于嵌入的推理(TransE、RotatE)
- 多跳关系推理
- 不确定性推理和置信度计算

### 4. **GraphRAG系统集成**
- 图谱增强的文档检索
- 实体和关系的上下文扩展
- 图谱引导的问题分解
- 多源知识融合

### 5. **可视化和查询接口**
- 交互式知识图谱可视化
- 自然语言到图查询转换
- SPARQL查询接口
- 知识探索和发现工具

### 6. **知识图谱管理**
- 图谱版本管理和回溯
- 知识来源追踪
- 图谱统计和分析
- 数据导入导出工具

## 🏗️ 用户故事分解

### Story 8.1: 实体识别与关系抽取引擎
**优先级**: P1 | **工期**: 3周
- 集成spaCy、Stanza等NER模型
- 实现关系抽取模型(BERT-based)
- 构建实体链接和消歧算法
- 支持中英文等多语言处理

### Story 8.2: 动态知识图谱存储系统
**优先级**: P1 | **工期**: 2-3周
- 选择和集成图数据库(Neo4j/ArangoDB)
- 设计知识图谱数据模型
- 实现增量更新和冲突解决
- 构建图谱质量评估框架

### Story 8.3: 图谱推理引擎
**优先级**: P1 | **工期**: 3-4周
- 实现基于规则的推理引擎
- 集成知识图谱嵌入模型
- 构建多跳推理算法
- 实现置信度计算和不确定性处理

### Story 8.4: GraphRAG系统集成
**优先级**: P1 | **工期**: 2-3周
- 扩展现有RAG系统支持图谱
- 实现实体和关系的上下文扩展
- 构建图谱引导的问题分解
- 集成多源知识融合算法

### Story 8.5: 知识图谱可视化界面
**优先级**: P2 | **工期**: 2周
- 实现交互式图谱可视化(D3.js/Cytoscape)
- 构建自然语言查询界面
- 实现知识探索和发现工具
- 创建图谱统计仪表板

### Story 8.6: 知识管理和API接口
**优先级**: P2 | **工期**: 1-2周
- 实现SPARQL查询接口
- 构建知识图谱管理API
- 实现数据导入导出功能
- 创建版本管理和追踪系统

### Story 8.7: 系统优化和部署
**优先级**: P1 | **工期**: 1-2周
- 性能调优和扩容准备
- 集成测试和质量保证
- 监控告警系统集成
- 生产环境部署

## 🎯 成功标准 (Definition of Done)

### 技术指标
- ✅ **实体识别准确率**: >90% (标准数据集)
- ✅ **关系抽取F1值**: >85% (标准数据集)
- ✅ **图谱查询延迟**: <500ms (单跳查询)
- ✅ **多跳推理准确率**: >80% (3跳内推理)
- ✅ **GraphRAG提升**: 相比传统RAG准确率提升25%+

### 功能指标
- ✅ **实体类型覆盖**: 支持20种以上实体类型
- ✅ **关系类型覆盖**: 支持50种以上关系类型
- ✅ **语言支持**: 中文、英文双语支持
- ✅ **并发查询**: 支持1000+并发图谱查询
- ✅ **数据规模**: 支持百万级实体和千万级关系

### 质量标准
- ✅ **测试覆盖率≥85%**: 单元测试 + 集成测试 + E2E测试
- ✅ **图谱质量分数**: >8.0/10.0 (完整性、一致性、准确性综合评分)
- ✅ **系统稳定性**: 99.5%可用性，MTTR<15分钟
- ✅ **知识时效性**: 90%知识在24小时内更新

## 🔧 技术实现亮点

### 实体识别与关系抽取引擎
```python
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    confidence: float
    canonical_form: Optional[str] = None
    linked_entity: Optional[str] = None

@dataclass
class Relation:
    subject: Entity
    predicate: str
    object: Entity
    confidence: float
    context: str

class EntityRecognizer:
    """命名实体识别器"""
    
    def __init__(self, model_name: str = "dbmdz/bert-large-cased-finetuned-conll03-english"):
        # 加载预训练的NER模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.ner_pipeline = pipeline(
            "ner", 
            model=self.model, 
            tokenizer=self.tokenizer,
            aggregation_strategy="simple"
        )
        
        # 加载spaCy模型用于额外处理
        self.nlp = spacy.load("en_core_web_lg")
        
        # 实体类型映射
        self.entity_types = {
            'PERSON': 'Person',
            'ORG': 'Organization', 
            'GPE': 'Location',
            'DATE': 'Date',
            'MONEY': 'Money',
            'PERCENT': 'Percentage',
            'FACILITY': 'Facility',
            'PRODUCT': 'Product',
            'EVENT': 'Event',
            'WORK_OF_ART': 'WorkOfArt',
            'LAW': 'Law',
            'LANGUAGE': 'Language'
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """提取命名实体"""
        entities = []
        
        # 使用BERT-based NER
        ner_results = self.ner_pipeline(text)
        
        for result in ner_results:
            entity = Entity(
                text=result['word'],
                label=self.entity_types.get(result['entity_group'], result['entity_group']),
                start=result['start'],
                end=result['end'],
                confidence=result['score']
            )
            entities.append(entity)
        
        # 使用spaCy进行补充和验证
        doc = self.nlp(text)
        
        for ent in doc.ents:
            # 检查是否已经被BERT识别
            overlap = False
            for existing_entity in entities:
                if (ent.start_char >= existing_entity.start and 
                    ent.start_char <= existing_entity.end):
                    overlap = True
                    break
            
            if not overlap:
                entity = Entity(
                    text=ent.text,
                    label=self.entity_types.get(ent.label_, ent.label_),
                    start=ent.start_char,
                    end=ent.end_char,
                    confidence=0.8  # spaCy默认置信度
                )
                entities.append(entity)
        
        # 实体链接和规范化
        entities = self._link_entities(entities, text)
        
        return sorted(entities, key=lambda x: x.start)
    
    def _link_entities(self, entities: List[Entity], context: str) -> List[Entity]:
        """实体链接和规范化"""
        for entity in entities:
            # 简单的规范化处理
            canonical_form = entity.text.strip().title()
            
            # 实体消歧 - 这里可以集成更复杂的实体链接算法
            if entity.label == 'Person':
                canonical_form = self._normalize_person_name(entity.text)
            elif entity.label == 'Organization':
                canonical_form = self._normalize_organization_name(entity.text)
            
            entity.canonical_form = canonical_form
            entity.linked_entity = self._find_linked_entity(canonical_form, entity.label)
        
        return entities
    
    def _normalize_person_name(self, name: str) -> str:
        """人名规范化"""
        # 简单的人名处理逻辑
        name_parts = name.strip().split()
        if len(name_parts) >= 2:
            return f"{name_parts[-1]}, {' '.join(name_parts[:-1])}"
        return name.strip().title()
    
    def _normalize_organization_name(self, org: str) -> str:
        """组织名规范化"""
        # 移除常见后缀
        suffixes = ['Inc.', 'Corp.', 'LLC', 'Ltd.', 'Co.']
        normalized = org.strip()
        
        for suffix in suffixes:
            if normalized.endswith(suffix):
                normalized = normalized[:-len(suffix)].strip()
                break
        
        return normalized.title()
    
    def _find_linked_entity(self, canonical_form: str, entity_type: str) -> Optional[str]:
        """查找链接实体 - 这里可以连接到知识库"""
        # 简化实现，实际应该查询知识库
        return f"KB:{entity_type}:{canonical_form.replace(' ', '_')}"

class RelationExtractor:
    """关系抽取器"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        # 使用预训练的关系抽取模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 关系类型定义
        self.relation_types = [
            'works_for', 'located_in', 'born_in', 'died_in',
            'founded_by', 'owned_by', 'part_of', 'member_of',
            'spouse', 'child_of', 'parent_of', 'sibling_of',
            'educated_at', 'nationality', 'occupation',
            'headquartered_in', 'subsidiary_of', 'competitor_of'
        ]
        
        # 简单的模式匹配规则
        self.relation_patterns = {
            'works_for': [
                r'\b(?P<subject>\w+(?:\s+\w+)*)\s+works?\s+(?:for|at)\s+(?P<object>\w+(?:\s+\w+)*)',
                r'\b(?P<subject>\w+(?:\s+\w+)*)\s+(?:is|was)\s+employed\s+(?:by|at)\s+(?P<object>\w+(?:\s+\w+)*)'
            ],
            'located_in': [
                r'\b(?P<subject>\w+(?:\s+\w+)*)\s+(?:is|was)\s+located\s+in\s+(?P<object>\w+(?:\s+\w+)*)',
                r'\b(?P<subject>\w+(?:\s+\w+)*)\s+in\s+(?P<object>\w+(?:\s+\w+)*)'
            ],
            'founded_by': [
                r'\b(?P<object>\w+(?:\s+\w+)*)\s+founded\s+(?P<subject>\w+(?:\s+\w+)*)',
                r'\b(?P<subject>\w+(?:\s+\w+)*)\s+(?:was|is)\s+founded\s+by\s+(?P<object>\w+(?:\s+\w+)*)'
            ]
        }
    
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """提取实体间关系"""
        relations = []
        
        # 基于模式的关系抽取
        pattern_relations = self._extract_by_patterns(text, entities)
        relations.extend(pattern_relations)
        
        # 基于依存句法的关系抽取
        dependency_relations = self._extract_by_dependency(text, entities)
        relations.extend(dependency_relations)
        
        # 去重和置信度计算
        relations = self._deduplicate_relations(relations)
        
        return relations
    
    def _extract_by_patterns(self, text: str, entities: List[Entity]) -> List[Relation]:
        """基于模式的关系抽取"""
        import re
        relations = []
        
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                
                for match in matches:
                    subject_text = match.group('subject').strip()
                    object_text = match.group('object').strip()
                    
                    # 查找对应的实体
                    subject_entity = self._find_matching_entity(subject_text, entities)
                    object_entity = self._find_matching_entity(object_text, entities)
                    
                    if subject_entity and object_entity:
                        relation = Relation(
                            subject=subject_entity,
                            predicate=relation_type,
                            object=object_entity,
                            confidence=0.8,  # 模式匹配的基础置信度
                            context=match.group(0)
                        )
                        relations.append(relation)
        
        return relations
    
    def _extract_by_dependency(self, text: str, entities: List[Entity]) -> List[Relation]:
        """基于依存句法的关系抽取"""
        import spacy
        
        nlp = spacy.load("en_core_web_lg")
        doc = nlp(text)
        relations = []
        
        for sent in doc.sents:
            # 查找句子中的实体
            sent_entities = []
            for entity in entities:
                if entity.start >= sent.start_char and entity.end <= sent.end_char:
                    sent_entities.append(entity)
            
            # 如果句子中有多个实体，尝试提取关系
            if len(sent_entities) >= 2:
                for i, entity1 in enumerate(sent_entities):
                    for entity2 in sent_entities[i+1:]:
                        # 基于依存路径提取关系
                        relation_type = self._infer_relation_from_dependency(
                            sent, entity1, entity2
                        )
                        
                        if relation_type:
                            relation = Relation(
                                subject=entity1,
                                predicate=relation_type,
                                object=entity2,
                                confidence=0.6,  # 依存分析的置信度
                                context=sent.text
                            )
                            relations.append(relation)
        
        return relations
    
    def _find_matching_entity(self, text: str, entities: List[Entity]) -> Optional[Entity]:
        """查找匹配的实体"""
        text_lower = text.lower()
        
        for entity in entities:
            if (entity.text.lower() == text_lower or 
                text_lower in entity.text.lower() or 
                entity.text.lower() in text_lower):
                return entity
        
        return None
    
    def _infer_relation_from_dependency(self, sent, entity1: Entity, entity2: Entity) -> Optional[str]:
        """从依存关系推断关系类型"""
        # 简化的依存关系推断
        # 实际实现需要更复杂的依存路径分析
        
        sent_text = sent.text.lower()
        
        if 'work' in sent_text and ('for' in sent_text or 'at' in sent_text):
            return 'works_for'
        elif 'located' in sent_text or 'in' in sent_text:
            return 'located_in'
        elif 'founded' in sent_text:
            return 'founded_by'
        elif 'married' in sent_text or 'spouse' in sent_text:
            return 'spouse'
        
        return None
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """关系去重"""
        unique_relations = {}
        
        for relation in relations:
            # 创建关系的唯一键
            key = (
                relation.subject.canonical_form or relation.subject.text,
                relation.predicate,
                relation.object.canonical_form or relation.object.text
            )
            
            # 保留置信度更高的关系
            if key not in unique_relations or relation.confidence > unique_relations[key].confidence:
                unique_relations[key] = relation
        
        return list(unique_relations.values())

class KnowledgeGraphBuilder:
    """知识图谱构建器"""
    
    def __init__(self, graph_db_uri: str = "bolt://localhost:7687"):
        from neo4j import GraphDatabase
        
        self.driver = GraphDatabase.driver(graph_db_uri, auth=("neo4j", "password"))
        self.entity_recognizer = EntityRecognizer()
        self.relation_extractor = RelationExtractor()
        
        # 图谱统计
        self.stats = {
            'entities': 0,
            'relations': 0,
            'entity_types': set(),
            'relation_types': set()
        }
    
    def process_document(self, text: str, document_id: str) -> Dict[str, int]:
        """处理文档并构建知识图谱"""
        
        # 提取实体和关系
        entities = self.entity_recognizer.extract_entities(text)
        relations = self.relation_extractor.extract_relations(text, entities)
        
        # 存储到图数据库
        with self.driver.session() as session:
            # 创建实体节点
            for entity in entities:
                self._create_entity_node(session, entity, document_id)
            
            # 创建关系边
            for relation in relations:
                self._create_relation_edge(session, relation, document_id)
        
        # 更新统计信息
        self._update_stats(entities, relations)
        
        return {
            'entities_extracted': len(entities),
            'relations_extracted': len(relations)
        }
    
    def _create_entity_node(self, session, entity: Entity, document_id: str):
        """创建实体节点"""
        query = """
        MERGE (e:Entity {canonical_form: $canonical_form})
        SET e.text = $text,
            e.label = $label,
            e.confidence = $confidence,
            e.linked_entity = $linked_entity,
            e.last_updated = datetime()
        WITH e
        MERGE (d:Document {id: $document_id})
        MERGE (e)-[:MENTIONED_IN]->(d)
        """
        
        session.run(query, 
            canonical_form=entity.canonical_form or entity.text,
            text=entity.text,
            label=entity.label,
            confidence=entity.confidence,
            linked_entity=entity.linked_entity,
            document_id=document_id
        )
    
    def _create_relation_edge(self, session, relation: Relation, document_id: str):
        """创建关系边"""
        query = f"""
        MATCH (s:Entity {{canonical_form: $subject}})
        MATCH (o:Entity {{canonical_form: $object}})
        MERGE (s)-[r:{relation.predicate.upper()}]->(o)
        SET r.confidence = $confidence,
            r.context = $context,
            r.document_id = $document_id,
            r.last_updated = datetime()
        """
        
        session.run(query,
            subject=relation.subject.canonical_form or relation.subject.text,
            object=relation.object.canonical_form or relation.object.text,
            confidence=relation.confidence,
            context=relation.context,
            document_id=document_id
        )
    
    def _update_stats(self, entities: List[Entity], relations: List[Relation]):
        """更新图谱统计信息"""
        self.stats['entities'] += len(entities)
        self.stats['relations'] += len(relations)
        
        for entity in entities:
            self.stats['entity_types'].add(entity.label)
        
        for relation in relations:
            self.stats['relation_types'].add(relation.predicate)
    
    def query_graph(self, cypher_query: str) -> List[Dict]:
        """执行Cypher查询"""
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]
    
    def find_entity(self, entity_name: str) -> Optional[Dict]:
        """查找实体"""
        query = """
        MATCH (e:Entity)
        WHERE e.canonical_form CONTAINS $name OR e.text CONTAINS $name
        RETURN e
        LIMIT 1
        """
        
        results = self.query_graph(query)
        return results[0]['e'] if results else None
    
    def find_relations(self, entity1: str, entity2: str = None) -> List[Dict]:
        """查找关系"""
        if entity2:
            query = """
            MATCH (e1:Entity)-[r]-(e2:Entity)
            WHERE (e1.canonical_form CONTAINS $entity1 OR e1.text CONTAINS $entity1)
              AND (e2.canonical_form CONTAINS $entity2 OR e2.text CONTAINS $entity2)
            RETURN e1, r, e2
            """
            return self.query_graph(query)
        else:
            query = """
            MATCH (e1:Entity)-[r]-(e2:Entity)
            WHERE e1.canonical_form CONTAINS $entity1 OR e1.text CONTAINS $entity1
            RETURN e1, r, e2
            """
            return self.query_graph(query)
    
    def get_graph_stats(self) -> Dict:
        """获取图谱统计信息"""
        with self.driver.session() as session:
            # 获取实时统计
            entity_count = session.run("MATCH (e:Entity) RETURN count(e) as count").single()["count"]
            
            relation_query = """
            MATCH ()-[r]->()
            RETURN type(r) as rel_type, count(r) as count
            """
            relation_stats = session.run(relation_query).data()
            
            return {
                'total_entities': entity_count,
                'total_relations': sum(r['count'] for r in relation_stats),
                'entity_types': list(self.stats['entity_types']),
                'relation_types': list(self.stats['relation_types']),
                'relation_distribution': relation_stats
            }
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
```

### GraphRAG系统集成
```python
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass

@dataclass
class GraphContext:
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    subgraph: Dict[str, Any]
    reasoning_path: List[str]

class GraphRAG:
    """图谱增强的检索生成系统"""
    
    def __init__(self, knowledge_graph: KnowledgeGraphBuilder, vector_store, llm_client):
        self.kg = knowledge_graph
        self.vector_store = vector_store
        self.llm = llm_client
        
        # 实体嵌入缓存
        self.entity_embeddings = {}
        
    async def enhanced_retrieve(
        self, 
        query: str, 
        top_k: int = 10,
        include_reasoning: bool = True
    ) -> Dict[str, Any]:
        """图谱增强检索"""
        
        # 1. 传统向量检索
        vector_results = await self._vector_retrieve(query, top_k)
        
        # 2. 识别查询中的实体
        query_entities = self.kg.entity_recognizer.extract_entities(query)
        
        # 3. 图谱检索
        graph_context = await self._graph_retrieve(query_entities, query)
        
        # 4. 融合检索结果
        enhanced_results = await self._fuse_results(
            vector_results, 
            graph_context, 
            query
        )
        
        # 5. 多跳推理(如果需要)
        if include_reasoning:
            reasoning_results = await self._multi_hop_reasoning(
                query_entities, 
                query, 
                max_hops=3
            )
            enhanced_results['reasoning'] = reasoning_results
        
        return enhanced_results
    
    async def _vector_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """传统向量检索"""
        # 调用现有的向量检索系统
        results = await self.vector_store.similarity_search(query, k=top_k)
        return [{'content': r.page_content, 'metadata': r.metadata} for r in results]
    
    async def _graph_retrieve(self, entities: List[Entity], query: str) -> GraphContext:
        """图谱检索"""
        graph_entities = []
        graph_relations = []
        
        # 为每个识别的实体查找图谱信息
        for entity in entities:
            # 查找实体详细信息
            entity_info = self.kg.find_entity(entity.canonical_form or entity.text)
            if entity_info:
                graph_entities.append(entity_info)
                
                # 查找相关关系
                relations = self.kg.find_relations(entity.canonical_form or entity.text)
                graph_relations.extend(relations)
        
        # 构建查询相关的子图
        subgraph = self._build_subgraph(graph_entities, graph_relations)
        
        return GraphContext(
            entities=graph_entities,
            relations=graph_relations,
            subgraph=subgraph,
            reasoning_path=[]
        )
    
    def _build_subgraph(self, entities: List[Dict], relations: List[Dict]) -> Dict[str, Any]:
        """构建子图"""
        nodes = {}
        edges = []
        
        # 添加实体节点
        for entity in entities:
            node_id = entity.get('canonical_form', entity.get('text'))
            nodes[node_id] = {
                'id': node_id,
                'label': entity.get('label'),
                'type': 'entity',
                'properties': entity
            }
        
        # 添加关系边
        for rel in relations:
            if 'r' in rel and 'e1' in rel and 'e2' in rel:
                relation_info = rel['r']
                source = rel['e1'].get('canonical_form', rel['e1'].get('text'))
                target = rel['e2'].get('canonical_form', rel['e2'].get('text'))
                
                edges.append({
                    'source': source,
                    'target': target,
                    'relation': relation_info.get('type'),
                    'properties': relation_info
                })
        
        return {'nodes': nodes, 'edges': edges}
    
    async def _fuse_results(
        self, 
        vector_results: List[Dict], 
        graph_context: GraphContext, 
        query: str
    ) -> Dict[str, Any]:
        """融合向量检索和图谱检索结果"""
        
        # 基于图谱上下文重新排序向量结果
        enhanced_vector_results = []
        
        for result in vector_results:
            # 计算文档与图谱上下文的相关性
            graph_relevance = self._calculate_graph_relevance(
                result, 
                graph_context
            )
            
            result['graph_relevance'] = graph_relevance
            result['enhanced_score'] = result.get('score', 0.5) * 0.7 + graph_relevance * 0.3
            enhanced_vector_results.append(result)
        
        # 重新排序
        enhanced_vector_results.sort(key=lambda x: x['enhanced_score'], reverse=True)
        
        # 添加图谱特有的上下文
        graph_facts = self._extract_graph_facts(graph_context)
        
        return {
            'documents': enhanced_vector_results,
            'graph_context': graph_context,
            'graph_facts': graph_facts,
            'entities': graph_context.entities,
            'relations': graph_context.relations[:10]  # 限制关系数量
        }
    
    def _calculate_graph_relevance(
        self, 
        document: Dict, 
        graph_context: GraphContext
    ) -> float:
        """计算文档与图谱上下文的相关性"""
        content = document.get('content', '')
        relevance_score = 0.0
        
        # 实体匹配度
        entity_matches = 0
        for entity in graph_context.entities:
            entity_text = entity.get('canonical_form', entity.get('text', ''))
            if entity_text.lower() in content.lower():
                entity_matches += 1
        
        entity_relevance = entity_matches / max(1, len(graph_context.entities))
        
        # 关系相关度
        relation_matches = 0
        for rel in graph_context.relations[:5]:  # 检查前5个关系
            if 'r' in rel:
                relation_type = rel['r'].get('type', '').replace('_', ' ')
                if relation_type.lower() in content.lower():
                    relation_matches += 1
        
        relation_relevance = relation_matches / max(1, min(5, len(graph_context.relations)))
        
        # 组合相关性分数
        relevance_score = entity_relevance * 0.6 + relation_relevance * 0.4
        
        return min(1.0, relevance_score)
    
    def _extract_graph_facts(self, graph_context: GraphContext) -> List[str]:
        """从图谱上下文提取事实"""
        facts = []
        
        for rel in graph_context.relations[:10]:
            if 'r' in rel and 'e1' in rel and 'e2' in rel:
                subject = rel['e1'].get('canonical_form', rel['e1'].get('text'))
                predicate = rel['r'].get('type', '').replace('_', ' ')
                obj = rel['e2'].get('canonical_form', rel['e2'].get('text'))
                
                fact = f"{subject} {predicate} {obj}"
                facts.append(fact)
        
        return facts
    
    async def _multi_hop_reasoning(
        self, 
        entities: List[Entity], 
        query: str, 
        max_hops: int = 3
    ) -> Dict[str, Any]:
        """多跳推理"""
        reasoning_paths = []
        
        for entity in entities:
            entity_name = entity.canonical_form or entity.text
            paths = self._find_reasoning_paths(entity_name, query, max_hops)
            reasoning_paths.extend(paths)
        
        # 评估推理路径的相关性
        scored_paths = []
        for path in reasoning_paths:
            score = await self._score_reasoning_path(path, query)
            scored_paths.append({
                'path': path,
                'score': score,
                'explanation': self._explain_reasoning_path(path)
            })
        
        # 排序并返回最佳推理路径
        scored_paths.sort(key=lambda x: x['score'], reverse=True)
        
        return {
            'reasoning_paths': scored_paths[:5],  # 返回前5个最佳路径
            'total_paths_found': len(reasoning_paths)
        }
    
    def _find_reasoning_paths(
        self, 
        start_entity: str, 
        query: str, 
        max_hops: int
    ) -> List[List[str]]:
        """查找推理路径"""
        
        # 使用BFS查找路径
        from collections import deque
        
        queue = deque([(start_entity, [start_entity])])
        paths = []
        visited = set()
        
        while queue and len(paths) < 100:  # 限制路径数量
            current_entity, path = queue.popleft()
            
            if len(path) > max_hops:
                continue
            
            if current_entity in visited:
                continue
                
            visited.add(current_entity)
            
            # 获取相关关系
            relations = self.kg.find_relations(current_entity)
            
            for rel in relations:
                if 'e2' in rel:
                    next_entity = rel['e2'].get('canonical_form', rel['e2'].get('text'))
                    if next_entity not in path:  # 避免循环
                        new_path = path + [rel['r'].get('type', ''), next_entity]
                        paths.append(new_path)
                        
                        if len(new_path) < max_hops * 2:  # path包含实体和关系
                            queue.append((next_entity, new_path))
        
        return paths
    
    async def _score_reasoning_path(self, path: List[str], query: str) -> float:
        """评估推理路径的相关性"""
        # 简化评分：基于路径中关键词与查询的匹配度
        query_words = set(query.lower().split())
        path_text = ' '.join(path).lower()
        path_words = set(path_text.split())
        
        # 计算交集
        common_words = query_words.intersection(path_words)
        relevance_score = len(common_words) / max(1, len(query_words))
        
        # 路径长度惩罚
        length_penalty = 1.0 / (1 + len(path) / 10)
        
        return relevance_score * length_penalty
    
    def _explain_reasoning_path(self, path: List[str]) -> str:
        """解释推理路径"""
        if len(path) < 3:
            return "路径太短，无法解释"
        
        explanation_parts = []
        
        for i in range(0, len(path) - 1, 2):
            if i + 2 < len(path):
                entity1 = path[i]
                relation = path[i + 1].replace('_', ' ')
                entity2 = path[i + 2]
                
                explanation_parts.append(f"{entity1} {relation} {entity2}")
        
        return " → ".join(explanation_parts)
    
    async def generate_response(
        self, 
        query: str, 
        enhanced_results: Dict[str, Any]
    ) -> str:
        """生成图谱增强的回答"""
        
        # 构建增强的上下文
        context_parts = []
        
        # 添加文档上下文
        for doc in enhanced_results['documents'][:5]:
            context_parts.append(f"文档: {doc['content']}")
        
        # 添加图谱事实
        if enhanced_results.get('graph_facts'):
            context_parts.append("相关事实:")
            for fact in enhanced_results['graph_facts'][:5]:
                context_parts.append(f"- {fact}")
        
        # 添加推理路径
        if enhanced_results.get('reasoning', {}).get('reasoning_paths'):
            context_parts.append("推理路径:")
            for path_info in enhanced_results['reasoning']['reasoning_paths'][:2]:
                context_parts.append(f"- {path_info['explanation']}")
        
        context = "\n".join(context_parts)
        
        # 构建提示词
        prompt = f"""
        基于以下上下文信息回答问题。请综合考虑文档信息、知识图谱事实和推理路径。

        上下文:
        {context}

        问题: {query}

        请提供准确、完整的回答，并在必要时引用具体的事实和推理过程。
        """
        
        # 调用LLM生成回答
        response = await self.llm.generate_response(prompt)
        
        return response
```

## 🚦 风险评估与缓解

### 高风险项
1. **实体识别和关系抽取准确率**
   - 缓解: 使用多个模型组合，人工标注验证集
   - 验证: 在标准数据集上达到目标准确率

2. **知识图谱规模和性能**
   - 缓解: 图数据库优化，分层存储，查询缓存
   - 验证: 百万级实体的查询性能测试

3. **GraphRAG系统复杂性**
   - 缓解: 逐步集成，充分测试，降级方案
   - 验证: A/B测试验证效果提升

### 中风险项
1. **多语言支持复杂度**
   - 缓解: 先支持中英文，逐步扩展
   - 验证: 各语言的抽取质量测试

2. **图谱质量维护**
   - 缓解: 自动质量检测，人工审核机制
   - 验证: 图谱一致性和完整性评估

## 📅 实施路线图

### Phase 1: 基础抽取能力 (Week 1-4)
- 实体识别与关系抽取引擎
- 动态知识图谱存储系统
- 基础图谱构建流程

### Phase 2: 推理和查询 (Week 5-8)
- 图谱推理引擎
- 知识管理和API接口
- 查询优化和性能调优

### Phase 3: GraphRAG集成 (Week 9-10)
- GraphRAG系统集成
- 多跳推理算法
- 效果评估和优化

### Phase 4: 可视化和部署 (Week 11-12)
- 知识图谱可视化界面
- 系统集成测试
- 生产环境部署

---

**文档状态**: ✅ 完成  
**下一步**: 开始Story 8.1的实体识别与关系抽取引擎实施  
**依赖Epic**: 可与现有RAG系统并行开发，最后阶段集成