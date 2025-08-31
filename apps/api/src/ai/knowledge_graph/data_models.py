"""
知识图谱核心数据模型

定义实体、关系、知识图谱等核心数据结构
支持序列化、验证和类型安全
"""

from typing import List, Dict, Optional, Any, TypedDict, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from enum import Enum
import json
import uuid
from pydantic import BaseModel, field_validator, Field


class EntityType(str, Enum):
    """实体类型枚举 - 支持20种以上实体类型"""
    PERSON = "PERSON"
    ORGANIZATION = "ORGANIZATION"
    LOCATION = "LOCATION" 
    GPE = "GPE"  # Geopolitical entity
    DATE = "DATE"
    TIME = "TIME"
    MONEY = "MONEY"
    PERCENTAGE = "PERCENTAGE"
    FACILITY = "FACILITY"
    PRODUCT = "PRODUCT"
    EVENT = "EVENT"
    WORK_OF_ART = "WORK_OF_ART"
    LAW = "LAW"
    LANGUAGE = "LANGUAGE"
    NATIONALITY = "NATIONALITY"
    RELIGION = "RELIGION"
    CARDINAL = "CARDINAL"  # 数字
    ORDINAL = "ORDINAL"    # 序数
    QUANTITY = "QUANTITY"
    MISC = "MISC"
    
    # 中文特有实体类型
    COUNTRY = "COUNTRY"
    CITY = "CITY"
    PROVINCE = "PROVINCE"
    UNIVERSITY = "UNIVERSITY"
    COMPANY = "COMPANY"


class RelationType(str, Enum):
    """关系类型枚举 - 支持50种以上关系类型"""
    # 基础关系
    WORKS_FOR = "works_for"
    LOCATED_IN = "located_in"
    BORN_IN = "born_in"
    FOUNDED_BY = "founded_by"
    OWNED_BY = "owned_by"
    PART_OF = "part_of"
    MEMBER_OF = "member_of"
    SPOUSE = "spouse"
    CHILD_OF = "child_of"
    PARENT_OF = "parent_of"
    EDUCATED_AT = "educated_at"
    
    # 组织关系
    CEO_OF = "ceo_of"
    CHAIRMAN_OF = "chairman_of"
    EMPLOYEE_OF = "employee_of"
    SUBSIDIARY_OF = "subsidiary_of"
    COMPETITOR_OF = "competitor_of"
    PARTNER_OF = "partner_of"
    INVESTOR_IN = "investor_in"
    
    # 地理关系
    CAPITAL_OF = "capital_of"
    NEIGHBOR_OF = "neighbor_of"
    CONTAINS = "contains"
    NEAR = "near"
    BORDER_WITH = "border_with"
    
    # 时间关系
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    FOUNDED_IN = "founded_in"
    OCCURRED_IN = "occurred_in"
    
    # 产品关系
    MANUFACTURED_BY = "manufactured_by"
    DEVELOPED_BY = "developed_by"
    USED_BY = "used_by"
    COMPETES_WITH = "competes_with"
    
    # 其他关系
    AUTHOR_OF = "author_of"
    DIRECTED_BY = "directed_by"
    STARRED_IN = "starred_in"
    WON = "won"
    NOMINATED_FOR = "nominated_for"
    COLLABORATED_WITH = "collaborated_with"
    
    # 学术关系
    ADVISOR_OF = "advisor_of"
    STUDIED_UNDER = "studied_under"
    PUBLISHED_IN = "published_in"
    CITED_BY = "cited_by"
    
    # 语义关系
    SIMILAR_TO = "similar_to"
    OPPOSITE_TO = "opposite_to"
    CAUSE_OF = "cause_of"
    EFFECT_OF = "effect_of"
    
    # 技术关系
    IMPLEMENTS = "implements"
    DEPENDS_ON = "depends_on"
    
    # 其他
    UNRELATED = "unrelated"
    OTHER = "other"


@dataclass
class Entity:
    """实体数据类"""
    text: str
    label: EntityType
    start: int
    end: int
    confidence: float
    canonical_form: Optional[str] = None
    linked_entity: Optional[str] = None  # Wikidata/DBpedia URI
    language: Optional[str] = None
    entity_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后处理验证"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.start >= self.end:
            raise ValueError("Start position must be less than end position")
        if self.canonical_form is None:
            self.canonical_form = self.text.lower().strip()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "entity_id": self.entity_id,
            "text": self.text,
            "label": self.label.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "canonical_form": self.canonical_form,
            "linked_entity": self.linked_entity,
            "language": self.language,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        """从字典创建实体"""
        return cls(
            text=data["text"],
            label=EntityType(data["label"]),
            start=data["start"],
            end=data["end"], 
            confidence=data["confidence"],
            canonical_form=data.get("canonical_form"),
            linked_entity=data.get("linked_entity"),
            language=data.get("language"),
            entity_id=data.get("entity_id", str(uuid.uuid4())),
            metadata=data.get("metadata", {})
        )


@dataclass
class Relation:
    """关系数据类"""
    subject: Entity
    predicate: RelationType
    object: Entity
    confidence: float
    context: str
    source_sentence: str
    relation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """后处理验证"""
        if not 0 <= self.confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
        if self.subject == self.object:
            raise ValueError("Subject and object cannot be the same entity")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "relation_id": self.relation_id,
            "subject": self.subject.to_dict(),
            "predicate": self.predicate.value,
            "object": self.object.to_dict(),
            "confidence": self.confidence,
            "context": self.context,
            "source_sentence": self.source_sentence,
            "evidence": self.evidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Relation":
        """从字典创建关系"""
        return cls(
            subject=Entity.from_dict(data["subject"]),
            predicate=RelationType(data["predicate"]),
            object=Entity.from_dict(data["object"]),
            confidence=data["confidence"],
            context=data["context"],
            source_sentence=data["source_sentence"],
            relation_id=data.get("relation_id", str(uuid.uuid4())),
            evidence=data.get("evidence", []),
            metadata=data.get("metadata", {})
        )
    
    def to_triple(self) -> Tuple[str, str, str]:
        """转换为三元组格式 (主语, 谓语, 宾语)"""
        return (
            self.subject.canonical_form or self.subject.text,
            self.predicate.value,
            self.object.canonical_form or self.object.text
        )


class ExtractionResult(TypedDict):
    """知识抽取结果"""
    document_id: str
    text: str
    language: str
    entities: List[Entity]
    relations: List[Relation]
    processing_time: float
    model_versions: Dict[str, str]
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class TripleStore:
    """三元组存储"""
    triples: Set[Tuple[str, str, str]] = field(default_factory=set)
    triple_metadata: Dict[Tuple[str, str, str], Dict[str, Any]] = field(
        default_factory=dict
    )
    
    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """添加三元组"""
        triple = (subject, predicate, obj)
        self.triples.add(triple)
        if metadata:
            self.triple_metadata[triple] = metadata
    
    def add_relation(self, relation: Relation):
        """从关系添加三元组"""
        triple = relation.to_triple()
        self.triples.add(triple)
        self.triple_metadata[triple] = {
            "confidence": relation.confidence,
            "context": relation.context,
            "source_sentence": relation.source_sentence,
            "relation_id": relation.relation_id,
            "evidence": relation.evidence
        }
    
    def query_subject(self, subject: str) -> List[Tuple[str, str, str]]:
        """查询主语相关的三元组"""
        return [t for t in self.triples if t[0] == subject]
    
    def query_predicate(self, predicate: str) -> List[Tuple[str, str, str]]:
        """查询谓语相关的三元组"""  
        return [t for t in self.triples if t[1] == predicate]
    
    def query_object(self, obj: str) -> List[Tuple[str, str, str]]:
        """查询宾语相关的三元组"""
        return [t for t in self.triples if t[2] == obj]
    
    def size(self) -> int:
        """获取三元组数量"""
        return len(self.triples)


@dataclass
class KnowledgeGraph:
    """知识图谱"""
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: Dict[str, Relation] = field(default_factory=dict) 
    triple_store: TripleStore = field(default_factory=TripleStore)
    graph_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_entity(self, entity: Entity):
        """添加实体"""
        self.entities[entity.entity_id] = entity
    
    def add_relation(self, relation: Relation):
        """添加关系"""
        self.relations[relation.relation_id] = relation
        self.triple_store.add_relation(relation)
        
        # 确保实体也被添加到图中
        self.add_entity(relation.subject)
        self.add_entity(relation.object)
    
    def get_entity_relations(self, entity_id: str) -> List[Relation]:
        """获取实体相关的所有关系"""
        if entity_id not in self.entities:
            return []
        
        entity = self.entities[entity_id]
        canonical_form = entity.canonical_form or entity.text
        
        relations = []
        for relation in self.relations.values():
            subject_canonical = (
                relation.subject.canonical_form or relation.subject.text
            )
            object_canonical = (
                relation.object.canonical_form or relation.object.text
            )
            
            if canonical_form in (subject_canonical, object_canonical):
                relations.append(relation)
        
        return relations
    
    def merge(self, other: "KnowledgeGraph"):
        """合并另一个知识图谱"""
        # 合并实体
        for entity in other.entities.values():
            self.add_entity(entity)
        
        # 合并关系
        for relation in other.relations.values():
            self.add_relation(relation)
        
        # 合并元数据
        self.metadata.update(other.metadata)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "graph_id": self.graph_id,
            "entities": {k: v.to_dict() for k, v in self.entities.items()},
            "relations": {k: v.to_dict() for k, v in self.relations.items()},
            "triple_count": self.triple_store.size(),
            "metadata": self.metadata
        }
    
    @classmethod
    def from_extraction_result(cls, result: ExtractionResult) -> "KnowledgeGraph":
        """从抽取结果创建知识图谱"""
        kg = cls()
        
        # 添加实体
        for entity in result["entities"]:
            kg.add_entity(entity)
        
        # 添加关系
        for relation in result["relations"]:
            kg.add_relation(relation)
        
        # 添加元数据
        kg.metadata = {
            "document_id": result["document_id"],
            "language": result["language"],
            "processing_time": result["processing_time"],
            "model_versions": result["model_versions"],
            "timestamp": result["timestamp"].isoformat(),
            "source_metadata": result.get("metadata", {})
        }
        
        return kg
    
    def export_triples(self) -> List[Dict[str, Any]]:
        """导出所有三元组"""
        triples = []
        for triple in self.triple_store.triples:
            triple_dict = {
                "subject": triple[0],
                "predicate": triple[1], 
                "object": triple[2]
            }
            
            # 添加元数据
            if triple in self.triple_store.triple_metadata:
                triple_dict["metadata"] = self.triple_store.triple_metadata[triple]
            
            triples.append(triple_dict)
        
        return triples
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取知识图谱统计信息"""
        entity_types = {}
        relation_types = {}
        
        # 统计实体类型
        for entity in self.entities.values():
            entity_type = entity.label.value
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        
        # 统计关系类型
        for relation in self.relations.values():
            relation_type = relation.predicate.value
            relation_types[relation_type] = (
                relation_types.get(relation_type, 0) + 1
            )
        
        return {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_triples": self.triple_store.size(),
            "entity_types": entity_types,
            "relation_types": relation_types,
            "graph_id": self.graph_id
        }


# Pydantic 模型用于API验证
class EntityModel(BaseModel):
    """实体API模型"""
    text: str = Field(..., min_length=1, max_length=1000)
    label: str
    start: int = Field(..., ge=0)
    end: int = Field(..., gt=0)
    confidence: float = Field(..., ge=0, le=1)
    canonical_form: Optional[str] = None
    linked_entity: Optional[str] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('end')
    @classmethod
    def end_must_be_greater_than_start(cls, v, info):
        if info.data and 'start' in info.data and v <= info.data['start']:
            raise ValueError('End must be greater than start')
        return v
    
    @field_validator('label')
    @classmethod
    def label_must_be_valid(cls, v):
        try:
            EntityType(v)
        except ValueError:
            raise ValueError(f'Invalid entity type: {v}')
        return v
    
    def to_entity(self) -> Entity:
        """转换为Entity对象"""
        return Entity(
            text=self.text,
            label=EntityType(self.label),
            start=self.start,
            end=self.end,
            confidence=self.confidence,
            canonical_form=self.canonical_form,
            linked_entity=self.linked_entity,
            language=self.language,
            metadata=self.metadata
        )


class RelationModel(BaseModel):
    """关系API模型"""
    subject: EntityModel
    predicate: str
    object: EntityModel
    confidence: float = Field(..., ge=0, le=1)
    context: str = Field(..., min_length=1)
    source_sentence: str = Field(..., min_length=1)
    evidence: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('predicate')
    @classmethod
    def predicate_must_be_valid(cls, v):
        try:
            RelationType(v)
        except ValueError:
            raise ValueError(f'Invalid relation type: {v}')
        return v
    
    def to_relation(self) -> Relation:
        """转换为Relation对象"""
        return Relation(
            subject=self.subject.to_entity(),
            predicate=RelationType(self.predicate),
            object=self.object.to_entity(),
            confidence=self.confidence,
            context=self.context,
            source_sentence=self.source_sentence,
            evidence=self.evidence,
            metadata=self.metadata
        )


class ExtractionRequest(BaseModel):
    """知识抽取请求模型"""
    text: str = Field(..., min_length=1, max_length=50000)
    language: Optional[str] = "auto"
    extract_entities: bool = True
    extract_relations: bool = True
    link_entities: bool = True
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    extraction_config: Dict[str, Any] = Field(default_factory=dict)


class ExtractionResponse(BaseModel):
    """知识抽取响应模型"""
    document_id: str
    text: str
    language: str
    entities: List[EntityModel] = Field(default_factory=list)
    relations: List[RelationModel] = Field(default_factory=list)
    processing_time: float
    model_versions: Dict[str, str] = Field(default_factory=dict)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class BatchProcessingResult:
    """批处理结果数据结构"""
    batch_id: str
    total_documents: int
    successful_documents: int
    failed_documents: int
    results: List[Dict[str, Any]]
    errors: List[Dict[str, str]]
    processing_time: float
    metrics: Dict[str, Any]
    created_at: datetime = field(default_factory=utc_factory)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "batch_id": self.batch_id,
            "total_documents": self.total_documents,
            "successful_documents": self.successful_documents,
            "failed_documents": self.failed_documents,
            "success_rate": self.successful_documents / self.total_documents if self.total_documents > 0 else 0,
            "results": self.results,
            "errors": self.errors,
            "processing_time": self.processing_time,
            "metrics": self.metrics,
            "created_at": self.created_at.isoformat()
        }


class BatchProcessingRequest(BaseModel):
    """批处理请求模型"""
    documents: List[Dict[str, Any]] = Field(..., min_length=1, max_length=1000)
    priority: int = Field(default=0, ge=0, le=10)
    language: Optional[str] = "auto"
    extract_entities: bool = True
    extract_relations: bool = True
    link_entities: bool = True
    confidence_threshold: float = Field(default=0.5, ge=0, le=1)
    batch_settings: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('documents')
    @classmethod
    def validate_documents(cls, v):
        for doc in v:
            if "text" not in doc:
                raise ValueError("每个文档必须包含 'text' 字段")
            if not isinstance(doc["text"], str) or len(doc["text"]) == 0:
                raise ValueError("文档 'text' 字段必须是非空字符串")
        return v


class BatchProcessingResponse(BaseModel):
    """批处理响应模型"""
    batch_id: str
    status: str  # pending, processing, completed, failed
    total_documents: int
    processed_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    success_rate: float = 0.0
    estimated_completion_time: Optional[datetime] = None
    results: List[ExtractionResponse] = Field(default_factory=list)
    errors: List[Dict[str, str]] = Field(default_factory=list)
    processing_time: float = 0.0
    metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)