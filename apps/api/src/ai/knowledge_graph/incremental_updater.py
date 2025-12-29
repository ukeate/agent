"""
增量更新器
智能实体合并、关系去重、冲突解决和数据一致性维护
"""

from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
import hashlib
from difflib import SequenceMatcher
from .graph_database import Neo4jGraphDatabase
from .graph_operations import GraphOperations, QueryResult
from .schema import SchemaManager
from .data_models import Entity, Relation, EntityType, RelationType

from src.core.logging import get_logger
logger = get_logger(__name__)

class ConflictResolutionStrategy(str, Enum):
    """冲突解决策略"""
    MERGE_HIGHEST_CONFIDENCE = "merge_highest_confidence"
    MERGE_LATEST_TIMESTAMP = "merge_latest_timestamp"
    MERGE_ALL_PROPERTIES = "merge_all_properties"
    MANUAL_REVIEW = "manual_review"
    REJECT_DUPLICATES = "reject_duplicates"

class UpdateOperation(str, Enum):
    """更新操作类型"""
    CREATE = "create"
    UPDATE = "update"
    MERGE = "merge"
    DELETE = "delete"
    LINK = "link"

@dataclass
class EntitySimilarity:
    """实体相似度"""
    entity_id: str
    canonical_form: str
    similarity_score: float
    matching_attributes: List[str]
    confidence_diff: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "canonical_form": self.canonical_form,
            "similarity_score": self.similarity_score,
            "matching_attributes": self.matching_attributes,
            "confidence_diff": self.confidence_diff
        }

@dataclass
class ConflictReport:
    """冲突报告"""
    conflict_id: str
    conflict_type: str
    description: str
    entities_involved: List[str]
    recommended_action: str
    confidence: float
    created_at: datetime = field(default_factory=utc_factory)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type,
            "description": self.description,
            "entities_involved": self.entities_involved,
            "recommended_action": self.recommended_action,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class UpdateResult:
    """更新结果"""
    operation: UpdateOperation
    success: bool
    entity_id: Optional[str] = None
    merged_entities: List[str] = field(default_factory=list)
    conflicts_detected: List[ConflictReport] = field(default_factory=list)
    changes_made: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "operation": self.operation.value,
            "success": self.success,
            "entity_id": self.entity_id,
            "merged_entities": self.merged_entities,
            "conflicts_detected": [c.to_dict() for c in self.conflicts_detected],
            "changes_made": self.changes_made,
            "execution_time_ms": self.execution_time_ms,
            "error_message": self.error_message
        }

class IncrementalUpdater:
    """增量更新器"""
    
    def __init__(self, 
                 graph_db: Neo4jGraphDatabase,
                 graph_ops: GraphOperations,
                 schema_manager: SchemaManager):
        self.graph_db = graph_db
        self.graph_ops = graph_ops
        self.schema_manager = schema_manager
        self.similarity_threshold = 0.8
        self.confidence_threshold = 0.1
        
    async def upsert_entity(self,
                           entity: Entity,
                           conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MERGE_HIGHEST_CONFIDENCE) -> UpdateResult:
        """插入或更新实体（智能合并）"""
        start_time = utc_now()
        
        try:
            # 1. 查找潜在重复实体
            similar_entities = await self._find_similar_entities(entity)
            
            if not similar_entities:
                # 没有重复，直接创建
                result = await self._create_new_entity(entity)
                return UpdateResult(
                    operation=UpdateOperation.CREATE,
                    success=result.success,
                    entity_id=result.data[0]["entity_id"] if result.data else None,
                    execution_time_ms=(utc_now() - start_time).total_seconds() * 1000,
                    error_message=result.error_message
                )
            
            # 2. 找到最佳匹配
            best_match = max(similar_entities, key=lambda x: x.similarity_score)
            
            if best_match.similarity_score >= self.similarity_threshold:
                # 3. 执行合并
                merge_result = await self._merge_entities(entity, best_match, conflict_strategy)
                return merge_result
            else:
                # 4. 相似度不够，创建新实体但记录潜在冲突
                conflicts = []
                if best_match.similarity_score > 0.6:  # 中等相似度
                    conflict = ConflictReport(
                        conflict_id=str(uuid.uuid4()),
                        conflict_type="potential_duplicate",
                        description=f"新实体与现有实体 {best_match.entity_id} 相似度为 {best_match.similarity_score:.2f}",
                        entities_involved=[best_match.entity_id],
                        recommended_action="manual_review",
                        confidence=best_match.similarity_score
                    )
                    conflicts.append(conflict)
                
                result = await self._create_new_entity(entity)
                return UpdateResult(
                    operation=UpdateOperation.CREATE,
                    success=result.success,
                    entity_id=result.data[0]["entity_id"] if result.data else None,
                    conflicts_detected=conflicts,
                    execution_time_ms=(utc_now() - start_time).total_seconds() * 1000,
                    error_message=result.error_message
                )
                
        except Exception as e:
            logger.error(f"实体upsert失败: {str(e)}")
            return UpdateResult(
                operation=UpdateOperation.CREATE,
                success=False,
                execution_time_ms=(utc_now() - start_time).total_seconds() * 1000,
                error_message=str(e)
            )
    
    async def _find_similar_entities(self, entity: Entity) -> List[EntitySimilarity]:
        """查找相似实体"""
        # 1. 基于canonical_form的精确匹配
        exact_matches = await self._find_exact_matches(entity.canonical_form, entity.label.value)
        
        # 2. 基于文本相似度的模糊匹配
        fuzzy_matches = await self._find_fuzzy_matches(entity)
        
        # 3. 基于embedding的语义匹配（如果有）
        semantic_matches = []
        if hasattr(entity, 'embedding') and entity.metadata.get('embedding'):
            semantic_matches = await self._find_semantic_matches(entity)
        
        # 合并并去重
        all_matches = {}
        
        for match in exact_matches:
            all_matches[match.entity_id] = match
        
        for match in fuzzy_matches:
            if match.entity_id not in all_matches or match.similarity_score > all_matches[match.entity_id].similarity_score:
                all_matches[match.entity_id] = match
        
        for match in semantic_matches:
            if match.entity_id not in all_matches or match.similarity_score > all_matches[match.entity_id].similarity_score:
                all_matches[match.entity_id] = match
        
        return list(all_matches.values())
    
    async def _find_exact_matches(self, canonical_form: str, entity_type: str) -> List[EntitySimilarity]:
        """查找精确匹配"""
        query = """
        MATCH (e:Entity)
        WHERE e.canonical_form = $canonical_form AND e.type = $entity_type
        RETURN e.id as entity_id, e.canonical_form as canonical_form, e.confidence as confidence
        """
        
        result = await self.graph_db.execute_read_query(
            query, 
            {"canonical_form": canonical_form, "entity_type": entity_type}
        )
        
        matches = []
        for record in result:
            matches.append(EntitySimilarity(
                entity_id=record["entity_id"],
                canonical_form=record["canonical_form"],
                similarity_score=1.0,  # 精确匹配
                matching_attributes=["canonical_form", "type"],
                confidence_diff=0.0
            ))
        
        return matches
    
    async def _find_fuzzy_matches(self, entity: Entity) -> List[EntitySimilarity]:
        """查找模糊匹配"""
        # 使用文本相似度算法
        query = """
        MATCH (e:Entity)
        WHERE e.type = $entity_type 
        AND e.canonical_form <> $canonical_form
        RETURN e.id as entity_id, e.canonical_form as canonical_form, 
               e.text as text, e.confidence as confidence
        LIMIT 50
        """
        
        result = await self.graph_db.execute_read_query(
            query,
            {
                "entity_type": entity.label.value,
                "canonical_form": entity.canonical_form
            }
        )
        
        matches = []
        for record in result:
            # 计算文本相似度
            canonical_sim = SequenceMatcher(None, entity.canonical_form, record["canonical_form"]).ratio()
            text_sim = SequenceMatcher(None, entity.text, record.get("text", "")).ratio() if record.get("text") else 0
            
            # 综合相似度
            similarity_score = max(canonical_sim, text_sim)
            
            if similarity_score > 0.6:  # 只考虑相似度较高的
                matching_attributes = []
                if canonical_sim > 0.8:
                    matching_attributes.append("canonical_form")
                if text_sim > 0.8:
                    matching_attributes.append("text")
                
                matches.append(EntitySimilarity(
                    entity_id=record["entity_id"],
                    canonical_form=record["canonical_form"],
                    similarity_score=similarity_score,
                    matching_attributes=matching_attributes,
                    confidence_diff=abs(entity.confidence - record["confidence"])
                ))
        
        return matches
    
    async def _find_semantic_matches(self, entity: Entity) -> List[EntitySimilarity]:
        """查找语义匹配（基于向量相似度）"""
        # 如果支持向量索引，使用向量相似度查询
        embedding = entity.metadata.get('embedding', [])
        if not embedding:
            return []
        
        try:
            # Neo4j 5.x向量相似度查询
            query = """
            MATCH (e:Entity)
            WHERE e.type = $entity_type 
            AND e.embedding IS NOT NULL
            AND e.id <> $exclude_id
            WITH e, gds.similarity.cosine(e.embedding, $embedding) as similarity
            WHERE similarity > 0.7
            RETURN e.id as entity_id, e.canonical_form as canonical_form, 
                   e.confidence as confidence, similarity
            ORDER BY similarity DESC
            LIMIT 10
            """
            
            result = await self.graph_db.execute_read_query(
                query,
                {
                    "entity_type": entity.label.value,
                    "embedding": embedding,
                    "exclude_id": entity.entity_id
                }
            )
            
            matches = []
            for record in result:
                matches.append(EntitySimilarity(
                    entity_id=record["entity_id"],
                    canonical_form=record["canonical_form"],
                    similarity_score=record["similarity"],
                    matching_attributes=["semantic_embedding"],
                    confidence_diff=abs(entity.confidence - record["confidence"])
                ))
            
            return matches
            
        except Exception as e:
            logger.warning(f"语义匹配查询失败: {str(e)}")
            return []
    
    async def _create_new_entity(self, entity: Entity) -> QueryResult:
        """创建新实体"""
        entity_props = {
            "canonical_form": entity.canonical_form,
            "text": entity.text,
            "type": entity.label.value,
            "confidence": entity.confidence,
            "language": entity.language,
            "linked_entity": entity.linked_entity,
            **entity.metadata
        }
        
        return await self.graph_ops.create_entity(
            entity.entity_id,
            entity.label.value,
            entity.canonical_form,
            entity_props
        )
    
    async def _merge_entities(self,
                             new_entity: Entity,
                             existing_match: EntitySimilarity,
                             strategy: ConflictResolutionStrategy) -> UpdateResult:
        """合并实体"""
        start_time = utc_now()
        
        try:
            # 获取现有实体的详细信息
            existing_result = await self.graph_ops.get_entity(existing_match.entity_id)
            if not existing_result.success or not existing_result.data:
                raise ValueError(f"无法获取现有实体: {existing_match.entity_id}")
            
            existing_entity = existing_result.data[0]
            
            # 根据策略决定合并方式
            if strategy == ConflictResolutionStrategy.MERGE_HIGHEST_CONFIDENCE:
                merged_props = await self._merge_by_highest_confidence(new_entity, existing_entity)
            elif strategy == ConflictResolutionStrategy.MERGE_LATEST_TIMESTAMP:
                merged_props = await self._merge_by_latest_timestamp(new_entity, existing_entity)
            elif strategy == ConflictResolutionStrategy.MERGE_ALL_PROPERTIES:
                merged_props = await self._merge_all_properties(new_entity, existing_entity)
            else:
                raise ValueError(f"不支持的合并策略: {strategy}")
            
            # 更新现有实体
            update_result = await self.graph_ops.update_entity(existing_match.entity_id, merged_props)
            
            # 记录合并操作
            changes_made = {
                "original_entity": existing_entity,
                "new_entity_data": {
                    "canonical_form": new_entity.canonical_form,
                    "text": new_entity.text,
                    "confidence": new_entity.confidence
                },
                "merged_properties": merged_props,
                "merge_strategy": strategy.value
            }
            
            return UpdateResult(
                operation=UpdateOperation.MERGE,
                success=update_result.success,
                entity_id=existing_match.entity_id,
                merged_entities=[new_entity.entity_id],
                changes_made=changes_made,
                execution_time_ms=(utc_now() - start_time).total_seconds() * 1000,
                error_message=update_result.error_message
            )
            
        except Exception as e:
            logger.error(f"实体合并失败: {str(e)}")
            return UpdateResult(
                operation=UpdateOperation.MERGE,
                success=False,
                execution_time_ms=(utc_now() - start_time).total_seconds() * 1000,
                error_message=str(e)
            )
    
    async def _merge_by_highest_confidence(self, 
                                         new_entity: Entity, 
                                         existing_entity: Dict[str, Any]) -> Dict[str, Any]:
        """基于置信度合并"""
        merged_props = {}
        
        # 选择置信度更高的值
        if new_entity.confidence > existing_entity["confidence"]:
            merged_props["canonical_form"] = new_entity.canonical_form
            merged_props["text"] = new_entity.text
            merged_props["confidence"] = new_entity.confidence
            if new_entity.language:
                merged_props["language"] = new_entity.language
        else:
            merged_props["confidence"] = existing_entity["confidence"]
        
        # 合并其他属性
        merged_props.update(self._merge_metadata(new_entity.metadata, existing_entity.get("properties", {})))
        
        return merged_props
    
    async def _merge_by_latest_timestamp(self,
                                       new_entity: Entity,
                                       existing_entity: Dict[str, Any]) -> Dict[str, Any]:
        """基于时间戳合并（新的优先）"""
        merged_props = {
            "canonical_form": new_entity.canonical_form,
            "text": new_entity.text,
            "confidence": max(new_entity.confidence, existing_entity["confidence"]),  # 取最高置信度
        }
        
        if new_entity.language:
            merged_props["language"] = new_entity.language
        
        # 合并元数据
        merged_props.update(self._merge_metadata(new_entity.metadata, existing_entity.get("properties", {})))
        
        return merged_props
    
    async def _merge_all_properties(self,
                                  new_entity: Entity,
                                  existing_entity: Dict[str, Any]) -> Dict[str, Any]:
        """合并所有属性"""
        merged_props = {}
        
        # 数值属性：取平均值或最大值
        merged_props["confidence"] = max(new_entity.confidence, existing_entity["confidence"])
        
        # 文本属性：如果不同，创建alternatives列表
        if new_entity.canonical_form != existing_entity["canonical_form"]:
            merged_props["canonical_form"] = existing_entity["canonical_form"]  # 保持现有
            merged_props["alternative_forms"] = existing_entity.get("alternative_forms", []) + [new_entity.canonical_form]
        else:
            merged_props["canonical_form"] = new_entity.canonical_form
        
        if new_entity.text != existing_entity.get("text"):
            merged_props["text"] = existing_entity.get("text", new_entity.text)
            merged_props["alternative_texts"] = existing_entity.get("alternative_texts", []) + [new_entity.text]
        else:
            merged_props["text"] = new_entity.text
        
        # 合并元数据
        merged_props.update(self._merge_metadata(new_entity.metadata, existing_entity.get("properties", {})))
        
        return merged_props
    
    def _merge_metadata(self, 
                       new_metadata: Dict[str, Any], 
                       existing_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """合并元数据"""
        merged = existing_metadata.copy()
        
        for key, value in new_metadata.items():
            if key not in merged:
                merged[key] = value
            elif isinstance(value, (int, float)) and isinstance(merged[key], (int, float)):
                # 数值：取最大值
                merged[key] = max(value, merged[key])
            elif isinstance(value, list) and isinstance(merged[key], list):
                # 列表：合并去重
                merged[key] = list(set(merged[key] + value))
            elif key.endswith('_count') or key.endswith('_score'):
                # 计数或分数：累加或取最大值
                if isinstance(value, (int, float)) and isinstance(merged[key], (int, float)):
                    merged[key] = max(value, merged[key])
        
        return merged
    
    async def upsert_relation(self,
                             relation: Relation,
                             allow_duplicates: bool = False) -> UpdateResult:
        """插入或更新关系"""
        start_time = utc_now()
        
        try:
            # 1. 检查实体是否存在
            subject_exists = await self.graph_ops.get_entity(relation.subject.entity_id)
            object_exists = await self.graph_ops.get_entity(relation.object.entity_id)
            
            if not subject_exists.success or not object_exists.success:
                return UpdateResult(
                    operation=UpdateOperation.CREATE,
                    success=False,
                    error_message="关系涉及的实体不存在",
                    execution_time_ms=(utc_now() - start_time).total_seconds() * 1000
                )
            
            if not allow_duplicates:
                # 2. 检查关系是否已存在
                existing_relations = await self._find_existing_relations(relation)
                
                if existing_relations:
                    # 更新现有关系
                    best_match = existing_relations[0]  # 假设第一个是最匹配的
                    
                    update_props = {
                        "confidence": max(relation.confidence, best_match.get("confidence", 0)),
                        "context": relation.context,
                        "source_sentence": relation.source_sentence,
                        "evidence": relation.evidence,
                        **relation.metadata
                    }
                    
                    # 这里需要实现关系更新的逻辑
                    # Neo4j中更新关系相对复杂，需要先匹配再更新
                    query = """
                    MATCH (s:Entity {id: $subject_id})-[r:RELATION]->(o:Entity {id: $object_id})
                    WHERE r.type = $relation_type
                    SET r += $updates, r.updated_at = $updated_at
                    RETURN r.id as relation_id
                    """
                    
                    result = await self.graph_db.execute_write_query(
                        query,
                        {
                            "subject_id": relation.subject.entity_id,
                            "object_id": relation.object.entity_id,
                            "relation_type": relation.predicate.value,
                            "updates": update_props,
                            "updated_at": utc_now().isoformat()
                        }
                    )
                    
                    return UpdateResult(
                        operation=UpdateOperation.UPDATE,
                        success=len(result) > 0,
                        entity_id=result[0]["relation_id"] if result else None,
                        changes_made={"updated_properties": update_props},
                        execution_time_ms=(utc_now() - start_time).total_seconds() * 1000
                    )
            
            # 3. 创建新关系
            rel_props = {
                "context": relation.context,
                "source_sentence": relation.source_sentence,
                "evidence": relation.evidence,
                **relation.metadata
            }
            
            result = await self.graph_ops.create_relationship(
                relation.subject.entity_id,
                relation.object.entity_id,
                relation.predicate.value,
                rel_props
            )
            
            return UpdateResult(
                operation=UpdateOperation.CREATE,
                success=result.success,
                entity_id=result.data[0]["relation_id"] if result.data else None,
                execution_time_ms=(utc_now() - start_time).total_seconds() * 1000,
                error_message=result.error_message
            )
            
        except Exception as e:
            logger.error(f"关系upsert失败: {str(e)}")
            return UpdateResult(
                operation=UpdateOperation.CREATE,
                success=False,
                execution_time_ms=(utc_now() - start_time).total_seconds() * 1000,
                error_message=str(e)
            )
    
    async def _find_existing_relations(self, relation: Relation) -> List[Dict[str, Any]]:
        """查找现有关系"""
        query = """
        MATCH (s:Entity {id: $subject_id})-[r:RELATION]->(o:Entity {id: $object_id})
        WHERE r.type = $relation_type
        RETURN r.id as id, r.confidence as confidence, r.context as context,
               properties(r) as properties
        """
        
        result = await self.graph_db.execute_read_query(
            query,
            {
                "subject_id": relation.subject.entity_id,
                "object_id": relation.object.entity_id,
                "relation_type": relation.predicate.value
            }
        )
        
        return result
    
    async def batch_upsert_entities(self,
                                  entities: List[Entity],
                                  conflict_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.MERGE_HIGHEST_CONFIDENCE) -> List[UpdateResult]:
        """批量更新实体"""
        results = []
        
        # 并行处理（控制并发数）
        semaphore = asyncio.Semaphore(10)  # 最多同时处理10个
        
        async def process_entity(entity: Entity) -> UpdateResult:
            async with semaphore:
                return await self.upsert_entity(entity, conflict_strategy)
        
        # 创建所有任务
        tasks = [process_entity(entity) for entity in entities]
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(UpdateResult(
                    operation=UpdateOperation.CREATE,
                    success=False,
                    error_message=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def detect_and_resolve_conflicts(self) -> List[ConflictReport]:
        """检测和解决数据冲突"""
        conflicts = []
        
        # 1. 检测重复实体
        duplicate_conflicts = await self._detect_duplicate_entities()
        conflicts.extend(duplicate_conflicts)
        
        # 2. 检测矛盾关系
        contradiction_conflicts = await self._detect_contradictory_relations()
        conflicts.extend(contradiction_conflicts)
        
        # 3. 检测数据一致性问题
        consistency_conflicts = await self._detect_consistency_issues()
        conflicts.extend(consistency_conflicts)
        
        return conflicts
    
    async def _detect_duplicate_entities(self) -> List[ConflictReport]:
        """检测重复实体"""
        conflicts = []
        
        # 查找可能重复的实体对
        query = """
        MATCH (e1:Entity), (e2:Entity)
        WHERE e1.id < e2.id 
        AND e1.type = e2.type
        AND e1.canonical_form = e2.canonical_form
        RETURN e1.id as entity1_id, e2.id as entity2_id, 
               e1.canonical_form as canonical_form,
               e1.confidence as conf1, e2.confidence as conf2
        """
        
        result = await self.graph_db.execute_read_query(query)
        
        for record in result:
            conflict = ConflictReport(
                conflict_id=str(uuid.uuid4()),
                conflict_type="exact_duplicate",
                description=f"实体重复: {record['canonical_form']}",
                entities_involved=[record["entity1_id"], record["entity2_id"]],
                recommended_action="merge_entities",
                confidence=1.0
            )
            conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_contradictory_relations(self) -> List[ConflictReport]:
        """检测矛盾关系"""
        conflicts = []
        
        # 检测互相矛盾的关系类型
        contradiction_pairs = [
            ("born_in", "died_in"),  # 出生地和死亡地可能矛盾
            ("works_for", "competitor_of"),  # 工作关系和竞争关系矛盾
            ("spouse", "divorced_from")  # 配偶和离婚关系矛盾
        ]
        
        for rel1, rel2 in contradiction_pairs:
            query = """
            MATCH (a:Entity)-[r1:RELATION]->(b:Entity),
                  (a)-[r2:RELATION]->(c:Entity)
            WHERE r1.type = $rel1 AND r2.type = $rel2
            AND b.id = c.id
            RETURN a.id as entity_id, b.id as target_id,
                   r1.confidence as conf1, r2.confidence as conf2
            """
            
            result = await self.graph_db.execute_read_query(
                query, 
                {"rel1": rel1, "rel2": rel2}
            )
            
            for record in result:
                conflict = ConflictReport(
                    conflict_id=str(uuid.uuid4()),
                    conflict_type="contradictory_relations",
                    description=f"矛盾关系: {rel1} vs {rel2}",
                    entities_involved=[record["entity_id"], record["target_id"]],
                    recommended_action="manual_review",
                    confidence=abs(record["conf1"] - record["conf2"])
                )
                conflicts.append(conflict)
        
        return conflicts
    
    async def _detect_consistency_issues(self) -> List[ConflictReport]:
        """检测一致性问题"""
        conflicts = []
        
        # 检测置信度异常的实体
        query = """
        MATCH (e:Entity)
        WHERE e.confidence < 0.1 OR e.confidence > 1.0
        RETURN e.id as entity_id, e.canonical_form as canonical_form, 
               e.confidence as confidence
        """
        
        result = await self.graph_db.execute_read_query(query)
        
        for record in result:
            conflict = ConflictReport(
                conflict_id=str(uuid.uuid4()),
                conflict_type="invalid_confidence",
                description=f"置信度异常: {record['confidence']}",
                entities_involved=[record["entity_id"]],
                recommended_action="fix_confidence",
                confidence=1.0
            )
            conflicts.append(conflict)
        
        return conflicts
    
    async def get_update_statistics(self) -> Dict[str, Any]:
        """获取更新统计信息"""
        stats = {
            "total_entities": 0,
            "total_relations": 0,
            "avg_entity_confidence": 0.0,
            "avg_relation_confidence": 0.0,
            "entity_types_distribution": {},
            "relation_types_distribution": {},
            "recent_updates": 0,
            "potential_conflicts": 0
        }
        
        try:
            # 基础统计
            basic_stats = await self.graph_ops.get_graph_statistics()
            if basic_stats.success and basic_stats.data:
                graph_stats = basic_stats.data[0]["statistics"]
                stats.update({
                    "total_entities": graph_stats.get("entity_count", [{"count": 0}])[0]["count"],
                    "total_relations": graph_stats.get("relation_count", [{"count": 0}])[0]["count"],
                    "avg_entity_confidence": graph_stats.get("avg_confidence", [{"avg_confidence": 0.0}])[0]["avg_confidence"] or 0.0
                })
                
                # 类型分布
                if graph_stats.get("entity_types"):
                    stats["entity_types_distribution"] = {
                        item["type"]: item["count"] 
                        for item in graph_stats["entity_types"]
                    }
                
                if graph_stats.get("relation_types"):
                    stats["relation_types_distribution"] = {
                        item["type"]: item["count"] 
                        for item in graph_stats["relation_types"]
                    }
            
            # 最近更新统计
            recent_query = """
            MATCH (e:Entity)
            WHERE e.updated_at > datetime() - duration('P7D')
            RETURN count(e) as recent_count
            """
            
            recent_result = await self.graph_db.execute_read_query(recent_query)
            if recent_result:
                stats["recent_updates"] = recent_result[0]["recent_count"]
            
            # 潜在冲突统计
            conflicts = await self.detect_and_resolve_conflicts()
            stats["potential_conflicts"] = len(conflicts)
            
        except Exception as e:
            logger.error(f"获取更新统计失败: {str(e)}")
        
        return stats
