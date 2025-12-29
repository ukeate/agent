"""
知识图谱数据模型和序列化 - 统一的数据结构定义和序列化工具
"""

import json
import uuid
from src.core.utils import secure_pickle as pickle
import gzip
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from typing import Dict, List, Any, Optional, Union, Set, Type, Generic, TypeVar
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

try:
    import pydantic
    from pydantic import BaseModel, Field, field_validator
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object

try:
    from rdflib import URIRef, Literal, BNode
    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False
    URIRef = Literal = BNode = None

T = TypeVar('T')

class DataType(Enum):
    """数据类型枚举"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    URI = "uri"
    LITERAL = "literal"
    BLANK_NODE = "blank_node"

class SerializationFormat(Enum):
    """序列化格式"""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    PROTOBUF = "protobuf"

@dataclass
class BaseEntity:
    """基础实体类"""
    id: str
    created_at: datetime = field(default_factory=lambda: utc_now())
    updated_at: datetime = field(default_factory=lambda: utc_now())
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
    
    def update_timestamp(self):
        """更新修改时间"""
        self.updated_at = utc_now()

@dataclass
class Triple(BaseEntity):
    """三元组模型"""
    subject: str
    predicate: str
    object: str
    object_type: DataType = DataType.LITERAL
    object_datatype: Optional[str] = None
    object_language: Optional[str] = None
    graph_uri: Optional[str] = None
    confidence: float = 1.0
    source: Optional[str] = None
    
    def to_rdf_triple(self):
        """转换为RDF三元组"""
        if not HAS_RDFLIB:
            return (self.subject, self.predicate, self.object)
        
        # 主语处理
        if self.subject.startswith('_:'):
            s = BNode(self.subject[2:])
        else:
            s = URIRef(self.subject)
        
        # 谓语处理
        p = URIRef(self.predicate)
        
        # 宾语处理
        if self.object_type == DataType.URI:
            o = URIRef(self.object)
        elif self.object_type == DataType.BLANK_NODE:
            o = BNode(self.object[2:] if self.object.startswith('_:') else self.object)
        else:
            # 字面量
            if self.object_language:
                o = Literal(self.object, lang=self.object_language)
            elif self.object_datatype:
                o = Literal(self.object, datatype=URIRef(self.object_datatype))
            else:
                o = Literal(self.object)
        
        return (s, p, o)
    
    @classmethod
    def from_rdf_triple(cls, triple_tuple, graph_uri: str = None) -> 'Triple':
        """从RDF三元组创建"""
        if not HAS_RDFLIB:
            s, p, o = triple_tuple
            return cls(
                id=str(uuid.uuid4()),
                subject=str(s),
                predicate=str(p),
                object=str(o),
                graph_uri=graph_uri
            )
        
        s, p, o = triple_tuple
        
        # 处理主语
        subject = str(s)
        if isinstance(s, BNode):
            subject = f"_:{s}"
        
        # 处理谓语
        predicate = str(p)
        
        # 处理宾语
        object_val = str(o)
        object_type = DataType.LITERAL
        object_datatype = None
        object_language = None
        
        if isinstance(o, URIRef):
            object_type = DataType.URI
        elif isinstance(o, BNode):
            object_type = DataType.BLANK_NODE
            object_val = f"_:{o}"
        elif isinstance(o, Literal):
            if o.language:
                object_language = str(o.language)
            if o.datatype:
                object_datatype = str(o.datatype)
        
        return cls(
            id=str(uuid.uuid4()),
            subject=subject,
            predicate=predicate,
            object=object_val,
            object_type=object_type,
            object_datatype=object_datatype,
            object_language=object_language,
            graph_uri=graph_uri
        )

@dataclass
class Entity(BaseEntity):
    """实体模型"""
    uri: str
    label: Optional[str] = None
    description: Optional[str] = None
    type_uris: List[str] = field(default_factory=list)
    properties: Dict[str, List[Any]] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def add_property(self, predicate: str, value: Any, datatype: DataType = DataType.LITERAL):
        """添加属性"""
        if predicate not in self.properties:
            self.properties[predicate] = []
        
        property_value = {
            "value": value,
            "datatype": datatype.value,
            "added_at": utc_now().isoformat()
        }
        
        self.properties[predicate].append(property_value)
        self.update_timestamp()
    
    def remove_property(self, predicate: str, value: Any = None):
        """移除属性"""
        if predicate in self.properties:
            if value is None:
                del self.properties[predicate]
            else:
                self.properties[predicate] = [
                    p for p in self.properties[predicate] 
                    if p.get("value") != value
                ]
                if not self.properties[predicate]:
                    del self.properties[predicate]
            self.update_timestamp()
    
    def get_property_values(self, predicate: str) -> List[Any]:
        """获取属性值"""
        return [p.get("value") for p in self.properties.get(predicate, [])]
    
    def to_triples(self) -> List[Triple]:
        """转换为三元组列表"""
        triples = []
        
        # 类型三元组
        for type_uri in self.type_uris:
            triple = Triple(
                id=str(uuid.uuid4()),
                subject=self.uri,
                predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
                object=type_uri,
                object_type=DataType.URI
            )
            triples.append(triple)
        
        # 标签三元组
        if self.label:
            triple = Triple(
                id=str(uuid.uuid4()),
                subject=self.uri,
                predicate="http://www.w3.org/2000/01/rdf-schema#label",
                object=self.label,
                object_type=DataType.STRING
            )
            triples.append(triple)
        
        # 描述三元组
        if self.description:
            triple = Triple(
                id=str(uuid.uuid4()),
                subject=self.uri,
                predicate="http://www.w3.org/2000/01/rdf-schema#comment",
                object=self.description,
                object_type=DataType.STRING
            )
            triples.append(triple)
        
        # 属性三元组
        for predicate, values in self.properties.items():
            for prop in values:
                triple = Triple(
                    id=str(uuid.uuid4()),
                    subject=self.uri,
                    predicate=predicate,
                    object=prop["value"],
                    object_type=DataType(prop.get("datatype", DataType.LITERAL.value))
                )
                triples.append(triple)
        
        return triples

@dataclass
class Relation(BaseEntity):
    """关系模型"""
    uri: str
    label: Optional[str] = None
    description: Optional[str] = None
    domain_uris: List[str] = field(default_factory=list)
    range_uris: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    is_symmetric: bool = False
    is_transitive: bool = False
    is_functional: bool = False
    inverse_of: Optional[str] = None
    
    def to_triples(self) -> List[Triple]:
        """转换为三元组列表"""
        triples = []
        
        # 类型声明
        triple = Triple(
            id=str(uuid.uuid4()),
            subject=self.uri,
            predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
            object="http://www.w3.org/1999/02/22-rdf-syntax-ns#Property",
            object_type=DataType.URI
        )
        triples.append(triple)
        
        # 标签
        if self.label:
            triple = Triple(
                id=str(uuid.uuid4()),
                subject=self.uri,
                predicate="http://www.w3.org/2000/01/rdf-schema#label",
                object=self.label,
                object_type=DataType.STRING
            )
            triples.append(triple)
        
        # 描述
        if self.description:
            triple = Triple(
                id=str(uuid.uuid4()),
                subject=self.uri,
                predicate="http://www.w3.org/2000/01/rdf-schema#comment",
                object=self.description,
                object_type=DataType.STRING
            )
            triples.append(triple)
        
        # 定义域
        for domain_uri in self.domain_uris:
            triple = Triple(
                id=str(uuid.uuid4()),
                subject=self.uri,
                predicate="http://www.w3.org/2000/01/rdf-schema#domain",
                object=domain_uri,
                object_type=DataType.URI
            )
            triples.append(triple)
        
        # 值域
        for range_uri in self.range_uris:
            triple = Triple(
                id=str(uuid.uuid4()),
                subject=self.uri,
                predicate="http://www.w3.org/2000/01/rdf-schema#range",
                object=range_uri,
                object_type=DataType.URI
            )
            triples.append(triple)
        
        return triples

@dataclass
class KnowledgeGraph(BaseEntity):
    """知识图谱模型"""
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    namespace: Optional[str] = None
    entities: Dict[str, Entity] = field(default_factory=dict)
    relations: Dict[str, Relation] = field(default_factory=dict)
    triples: Dict[str, Triple] = field(default_factory=dict)
    statistics: Optional[Dict[str, int]] = None
    
    def add_entity(self, entity: Entity):
        """添加实体"""
        self.entities[entity.uri] = entity
        self.update_timestamp()
    
    def add_relation(self, relation: Relation):
        """添加关系"""
        self.relations[relation.uri] = relation
        self.update_timestamp()
    
    def add_triple(self, triple: Triple):
        """添加三元组"""
        self.triples[triple.id] = triple
        self.update_timestamp()
    
    def remove_entity(self, uri: str):
        """移除实体"""
        if uri in self.entities:
            del self.entities[uri]
            # 移除相关三元组
            self.triples = {
                tid: t for tid, t in self.triples.items()
                if t.subject != uri and t.object != uri
            }
            self.update_timestamp()
    
    def remove_relation(self, uri: str):
        """移除关系"""
        if uri in self.relations:
            del self.relations[uri]
            # 移除相关三元组
            self.triples = {
                tid: t for tid, t in self.triples.items()
                if t.predicate != uri
            }
            self.update_timestamp()
    
    def remove_triple(self, triple_id: str):
        """移除三元组"""
        if triple_id in self.triples:
            del self.triples[triple_id]
            self.update_timestamp()
    
    def get_entity_triples(self, uri: str) -> List[Triple]:
        """获取实体相关的三元组"""
        return [
            t for t in self.triples.values()
            if t.subject == uri or t.object == uri
        ]
    
    def get_relation_triples(self, predicate: str) -> List[Triple]:
        """获取关系的所有三元组"""
        return [
            t for t in self.triples.values()
            if t.predicate == predicate
        ]
    
    def calculate_statistics(self):
        """计算统计信息"""
        self.statistics = {
            "total_entities": len(self.entities),
            "total_relations": len(self.relations),
            "total_triples": len(self.triples),
            "unique_subjects": len(set(t.subject for t in self.triples.values())),
            "unique_predicates": len(set(t.predicate for t in self.triples.values())),
            "unique_objects": len(set(t.object for t in self.triples.values())),
            "updated_at": self.updated_at.isoformat()
        }
        
        return self.statistics
    
    def merge_with(self, other_graph: 'KnowledgeGraph'):
        """合并另一个知识图谱"""
        # 合并实体
        for uri, entity in other_graph.entities.items():
            if uri in self.entities:
                # 合并属性
                existing = self.entities[uri]
                for pred, values in entity.properties.items():
                    if pred not in existing.properties:
                        existing.properties[pred] = values
                    else:
                        existing.properties[pred].extend(values)
            else:
                self.entities[uri] = entity
        
        # 合并关系
        for uri, relation in other_graph.relations.items():
            if uri not in self.relations:
                self.relations[uri] = relation
        
        # 合并三元组
        for triple_id, triple in other_graph.triples.items():
            if triple_id not in self.triples:
                self.triples[triple_id] = triple
        
        self.update_timestamp()

class SerializerInterface(ABC, Generic[T]):
    """序列化器接口"""
    
    @abstractmethod
    def serialize(self, obj: T) -> bytes:
        """序列化对象"""
        raise NotImplementedError
    
    @abstractmethod
    def deserialize(self, data: bytes) -> T:
        """反序列化对象"""
        raise NotImplementedError
    
    @abstractmethod
    def get_format(self) -> SerializationFormat:
        """获取序列化格式"""
        raise NotImplementedError

class JSONSerializer(SerializerInterface[T]):
    """JSON序列化器"""
    
    def __init__(self, indent: int = None):
        self.indent = indent
    
    def serialize(self, obj: T) -> bytes:
        """序列化为JSON"""
        try:
            if isinstance(obj, BaseEntity):
                data = self._serialize_entity(obj)
            else:
                data = obj
            
            json_str = json.dumps(data, default=self._json_default, 
                                indent=self.indent, ensure_ascii=False)
            return json_str.encode('utf-8')
        except Exception as e:
            logger.error(f"JSON序列化失败: {e}")
            raise
    
    def deserialize(self, data: bytes) -> T:
        """从JSON反序列化"""
        try:
            json_str = data.decode('utf-8')
            obj_data = json.loads(json_str)
            
            # 根据类型信息重建对象
            obj_type = obj_data.get('__type__')
            if obj_type == 'Triple':
                return self._deserialize_triple(obj_data)
            elif obj_type == 'Entity':
                return self._deserialize_entity(obj_data)
            elif obj_type == 'Relation':
                return self._deserialize_relation(obj_data)
            elif obj_type == 'KnowledgeGraph':
                return self._deserialize_knowledge_graph(obj_data)
            else:
                return obj_data
                
        except Exception as e:
            logger.error(f"JSON反序列化失败: {e}")
            raise
    
    def get_format(self) -> SerializationFormat:
        return SerializationFormat.JSON
    
    def _serialize_entity(self, obj):
        """序列化实体"""
        data = asdict(obj)
        data['__type__'] = obj.__class__.__name__
        
        # 处理日期时间
        if hasattr(obj, 'created_at') and obj.created_at:
            data['created_at'] = obj.created_at.isoformat()
        if hasattr(obj, 'updated_at') and obj.updated_at:
            data['updated_at'] = obj.updated_at.isoformat()
        
        # 处理枚举
        for key, value in data.items():
            if isinstance(value, Enum):
                data[key] = value.value
            elif isinstance(value, set):
                data[key] = list(value)
        
        return data
    
    def _deserialize_triple(self, data) -> Triple:
        """反序列化三元组"""
        # 处理日期时间
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # 处理枚举
        if 'object_type' in data and isinstance(data['object_type'], str):
            data['object_type'] = DataType(data['object_type'])
        
        data.pop('__type__', None)
        return Triple(**data)
    
    def _deserialize_entity(self, data) -> Entity:
        """反序列化实体"""
        # 处理日期时间
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        data.pop('__type__', None)
        return Entity(**data)
    
    def _deserialize_relation(self, data) -> Relation:
        """反序列化关系"""
        # 处理日期时间
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        data.pop('__type__', None)
        return Relation(**data)
    
    def _deserialize_knowledge_graph(self, data) -> KnowledgeGraph:
        """反序列化知识图谱"""
        # 处理日期时间
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # 重建实体字典
        if 'entities' in data:
            entities = {}
            for uri, entity_data in data['entities'].items():
                entities[uri] = self._deserialize_entity(entity_data)
            data['entities'] = entities
        
        # 重建关系字典
        if 'relations' in data:
            relations = {}
            for uri, relation_data in data['relations'].items():
                relations[uri] = self._deserialize_relation(relation_data)
            data['relations'] = relations
        
        # 重建三元组字典
        if 'triples' in data:
            triples = {}
            for triple_id, triple_data in data['triples'].items():
                triples[triple_id] = self._deserialize_triple(triple_data)
            data['triples'] = triples
        
        data.pop('__type__', None)
        return KnowledgeGraph(**data)
    
    def _json_default(self, obj):
        """JSON默认序列化处理"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, set):
            return list(obj)
        else:
            return str(obj)

class PickleSerializer(SerializerInterface[T]):
    """Pickle序列化器"""
    
    def __init__(self, use_compression: bool = True):
        self.use_compression = use_compression
    
    def serialize(self, obj: T) -> bytes:
        """序列化为Pickle"""
        try:
            data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
            
            if self.use_compression:
                data = gzip.compress(data)
            
            return data
        except Exception as e:
            logger.error(f"Pickle序列化失败: {e}")
            raise
    
    def deserialize(self, data: bytes) -> T:
        """从Pickle反序列化"""
        try:
            if self.use_compression:
                data = gzip.decompress(data)
            
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Pickle反序列化失败: {e}")
            raise
    
    def get_format(self) -> SerializationFormat:
        return SerializationFormat.PICKLE

class ModelRegistry:
    """模型注册表"""
    
    def __init__(self):
        self.serializers: Dict[SerializationFormat, SerializerInterface] = {
            SerializationFormat.JSON: JSONSerializer(indent=2),
            SerializationFormat.PICKLE: PickleSerializer(use_compression=True)
        }
        self.default_format = SerializationFormat.JSON
    
    def register_serializer(self, format: SerializationFormat, serializer: SerializerInterface):
        """注册序列化器"""
        self.serializers[format] = serializer
    
    def get_serializer(self, format: SerializationFormat = None) -> SerializerInterface:
        """获取序列化器"""
        format = format or self.default_format
        if format not in self.serializers:
            raise ValueError(f"不支持的序列化格式: {format}")
        return self.serializers[format]
    
    def serialize(self, obj: Any, format: SerializationFormat = None) -> bytes:
        """序列化对象"""
        serializer = self.get_serializer(format)
        return serializer.serialize(obj)
    
    def deserialize(self, data: bytes, format: SerializationFormat = None) -> Any:
        """反序列化对象"""
        serializer = self.get_serializer(format)
        return serializer.deserialize(data)

# 全局模型注册表实例
model_registry = ModelRegistry()

# Pydantic模型（如果可用）
if HAS_PYDANTIC:
    
    class TripleModel(BaseModel):
        """三元组Pydantic模型"""
        id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        subject: str
        predicate: str
        object: str
        object_type: DataType = DataType.LITERAL
        object_datatype: Optional[str] = None
        object_language: Optional[str] = None
        graph_uri: Optional[str] = None
        confidence: float = Field(default=1.0, ge=0.0, le=1.0)
        source: Optional[str] = None
        created_at: datetime = Field(default_factory=lambda: utc_now())
        updated_at: datetime = Field(default_factory=lambda: utc_now())
        metadata: Optional[Dict[str, Any]] = None
        
        @field_validator('confidence')
        def validate_confidence(cls, v):
            if not 0.0 <= v <= 1.0:
                raise ValueError('置信度必须在0.0到1.0之间')
            return v
    
    class EntityModel(BaseModel):
        """实体Pydantic模型"""
        id: str = Field(default_factory=lambda: str(uuid.uuid4()))
        uri: str
        label: Optional[str] = None
        description: Optional[str] = None
        type_uris: List[str] = Field(default_factory=list)
        properties: Dict[str, List[Any]] = Field(default_factory=dict)
        aliases: List[str] = Field(default_factory=list)
        tags: List[str] = Field(default_factory=list)
        created_at: datetime = Field(default_factory=lambda: utc_now())
        updated_at: datetime = Field(default_factory=lambda: utc_now())
        metadata: Optional[Dict[str, Any]] = None

# 便捷函数
def create_triple(subject: str, predicate: str, object: str, **kwargs) -> Triple:
    """创建三元组的便捷函数"""
    return Triple(
        id=str(uuid.uuid4()),
        subject=subject,
        predicate=predicate,
        object=object,
        **kwargs
    )

def create_entity(uri: str, label: str = None, **kwargs) -> Entity:
    """创建实体的便捷函数"""
    return Entity(
        id=str(uuid.uuid4()),
        uri=uri,
        label=label,
        **kwargs
    )

def create_relation(uri: str, label: str = None, **kwargs) -> Relation:
    """创建关系的便捷函数"""
    return Relation(
        id=str(uuid.uuid4()),
        uri=uri,
        label=label,
        **kwargs
    )

def create_knowledge_graph(name: str, description: str = None, **kwargs) -> KnowledgeGraph:
    """创建知识图谱的便捷函数"""
    return KnowledgeGraph(
        id=str(uuid.uuid4()),
        name=name,
        description=description,
        **kwargs
    )

if __name__ == "__main__":
    # 测试数据模型
    setup_logging()
    logger.info("测试知识图谱数据模型")
    
    # 创建知识图谱
    kg = create_knowledge_graph(
        name="测试知识图谱",
        description="用于测试的知识图谱",
        namespace="http://example.org/"
    )
    
    # 创建实体
    person_entity = create_entity(
        uri="http://example.org/Person",
        label="人",
        description="人类实体",
        type_uris=["http://www.w3.org/2002/07/owl#Class"]
    )
    
    john_entity = create_entity(
        uri="http://example.org/John",
        label="约翰",
        description="一个人的实例",
        type_uris=["http://example.org/Person"]
    )
    
    # 添加属性
    john_entity.add_property("http://example.org/age", 30, DataType.INTEGER)
    john_entity.add_property("http://example.org/name", "John Doe", DataType.STRING)
    
    # 创建关系
    age_relation = create_relation(
        uri="http://example.org/age",
        label="年龄",
        description="人的年龄属性",
        domain_uris=["http://example.org/Person"],
        range_uris=["http://www.w3.org/2001/XMLSchema#integer"]
    )
    
    # 创建三元组
    type_triple = create_triple(
        subject="http://example.org/John",
        predicate="http://www.w3.org/1999/02/22-rdf-syntax-ns#type",
        object="http://example.org/Person",
        object_type=DataType.URI
    )
    
    age_triple = create_triple(
        subject="http://example.org/John",
        predicate="http://example.org/age",
        object="30",
        object_type=DataType.INTEGER
    )
    
    # 添加到知识图谱
    kg.add_entity(person_entity)
    kg.add_entity(john_entity)
    kg.add_relation(age_relation)
    kg.add_triple(type_triple)
    kg.add_triple(age_triple)
    
    # 计算统计信息
    stats = kg.calculate_statistics()
    logger.info("知识图谱统计", stats=stats)
    
    # 测试序列化
    logger.info("测试序列化")
    
    # JSON序列化
    json_data = model_registry.serialize(kg, SerializationFormat.JSON)
    logger.info("JSON序列化大小", bytes=len(json_data))
    
    # 反序列化
    kg_restored = model_registry.deserialize(json_data, SerializationFormat.JSON)
    logger.info(
        "恢复的知识图谱",
        name=kg_restored.name,
        entity_count=len(kg_restored.entities),
    )
    
    # Pickle序列化
    pickle_data = model_registry.serialize(kg, SerializationFormat.PICKLE)
    logger.info("Pickle序列化大小", bytes=len(pickle_data))
    
    # 测试实体三元组转换
    entity_triples = john_entity.to_triples()
    logger.info("实体生成的三元组数", total=len(entity_triples))
    logger.info("数据模型测试完成")
