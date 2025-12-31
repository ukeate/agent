"""
知识图谱模式定义和管理
提供图谱结构的标准化定义、索引策略、约束管理和版本控制
"""

from typing import Any, Dict, List, Optional, Set, Union
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from .graph_database import Neo4jGraphDatabase, get_graph_database
from .data_models import EntityType, RelationType

from src.core.logging import get_logger
logger = get_logger(__name__)

class GraphNodeType(str, Enum):
    """图节点类型"""
    ENTITY = "Entity"
    CONCEPT = "Concept"
    DOCUMENT = "Document" 
    SOURCE = "Source"
    CLUSTER = "Cluster"

class GraphEdgeType(str, Enum):
    """图边类型"""
    RELATION = "RELATION"
    MENTIONED_IN = "MENTIONED_IN"
    DERIVED_FROM = "DERIVED_FROM"
    SIMILAR_TO = "SIMILAR_TO"
    CLUSTER_MEMBER = "CLUSTER_MEMBER"
    TEMPORAL_BEFORE = "TEMPORAL_BEFORE"
    TEMPORAL_AFTER = "TEMPORAL_AFTER"

@dataclass
class GraphNode:
    """图节点数据结构"""
    id: str
    type: GraphNodeType
    properties: Dict[str, Any]
    labels: List[str]
    created_at: datetime
    updated_at: datetime
    confidence: float
    source_count: int = 0
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "properties": self.properties,
            "labels": self.labels,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confidence": self.confidence,
            "source_count": self.source_count,
            "version": self.version
        }

@dataclass
class GraphEdge:
    """图边数据结构"""
    id: str
    source_id: str
    target_id: str
    type: GraphEdgeType
    relation_type: str
    properties: Dict[str, Any]
    confidence: float
    created_at: datetime
    updated_at: datetime
    source_documents: List[str]
    version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "type": self.type.value,
            "relation_type": self.relation_type,
            "properties": self.properties,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source_documents": self.source_documents,
            "version": self.version
        }

@dataclass
class IndexDefinition:
    """索引定义"""
    name: str
    index_type: str  # btree, fulltext, vector, composite
    node_labels: List[str]
    properties: List[str]
    options: Dict[str, Any] = field(default_factory=dict)
    
    def to_cypher(self) -> str:
        """转换为Cypher创建索引语句"""
        if self.index_type == "fulltext":
            return f"""
            CREATE FULLTEXT INDEX {self.name} IF NOT EXISTS
            FOR (n:{":".join(self.node_labels)})
            ON EACH [{", ".join(f"n.{prop}" for prop in self.properties)}]
            """
        elif self.index_type == "vector":
            # Neo4j 5.x向量索引
            vector_options = self.options.get("vector", {})
            dimensions = vector_options.get("dimensions", 1536)
            similarity_function = vector_options.get("similarity", "cosine")
            return f"""
            CREATE VECTOR INDEX {self.name} IF NOT EXISTS
            FOR (n:{self.node_labels[0]})
            ON (n.{self.properties[0]})
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {dimensions},
                    `vector.similarity_function`: '{similarity_function}'
                }}
            }}
            """
        else:
            # 标准索引
            if not self.node_labels:
                return f"""
                CREATE INDEX {self.name} IF NOT EXISTS
                FOR ()-[r:RELATION]-()
                ON ({", ".join(f"r.{prop}" for prop in self.properties)})
                """
            return f"""
            CREATE INDEX {self.name} IF NOT EXISTS 
            FOR (n:{self.node_labels[0]})
            ON ({", ".join(f"n.{prop}" for prop in self.properties)})
            """

@dataclass
class ConstraintDefinition:
    """约束定义"""
    name: str
    constraint_type: str  # unique
    node_labels: List[str]
    properties: List[str]
    
    def to_cypher(self) -> str:
        """转换为Cypher创建约束语句"""
        label = self.node_labels[0]
        
        if self.constraint_type == "unique":
            if len(self.properties) == 1:
                return f"""
                CREATE CONSTRAINT {self.name} IF NOT EXISTS 
                FOR (n:{label}) REQUIRE n.{self.properties[0]} IS UNIQUE
                """
            else:
                props = ", ".join(f"n.{prop}" for prop in self.properties)
                return f"""
                CREATE CONSTRAINT {self.name} IF NOT EXISTS 
                FOR (n:{label}) REQUIRE ({props}) IS UNIQUE
                """
        
class GraphSchema:
    """图谱模式定义和管理"""
    
    def __init__(self):
        self.version = "1.0.0"
        self.created_at = utc_now()
        self.updated_at = utc_now()
        
        # 节点类型定义
        self.node_types: Dict[str, Dict[str, Any]] = {
            "Entity": {
                "required_properties": ["id", "canonical_form", "type", "confidence", "created_at", "updated_at"],
                "optional_properties": ["text", "language", "linked_entity", "source_count", "embedding"],
                "allowed_entity_types": [t.value for t in EntityType]
            },
            "Document": {
                "required_properties": ["id", "title", "content", "created_at", "updated_at"],
                "optional_properties": ["language", "author", "source_url", "metadata"]
            },
            "Source": {
                "required_properties": ["id", "name", "type", "created_at"],
                "optional_properties": ["url", "description", "reliability_score"]
            },
            "Concept": {
                "required_properties": ["id", "name", "definition", "created_at"],
                "optional_properties": ["domain", "synonyms", "related_concepts"]
            }
        }
        
        # 关系类型定义
        self.relation_types: Dict[str, Dict[str, Any]] = {
            "RELATION": {
                "required_properties": ["id", "type", "confidence", "created_at", "updated_at"],
                "optional_properties": ["context", "evidence", "source_documents"],
                "allowed_relation_types": [t.value for t in RelationType]
            },
            "MENTIONED_IN": {
                "required_properties": ["id", "position", "created_at"],
                "optional_properties": ["context", "sentence"]
            },
            "DERIVED_FROM": {
                "required_properties": ["id", "extraction_method", "created_at"],
                "optional_properties": ["confidence", "model_version"]
            },
            "SIMILAR_TO": {
                "required_properties": ["id", "similarity_score", "created_at"],
                "optional_properties": ["method", "threshold"]
            }
        }
        
        # 索引定义
        self.indexes = self._define_indexes()
        
        # 约束定义
        self.constraints = self._define_constraints()
    
    def _define_indexes(self) -> List[IndexDefinition]:
        """定义索引策略"""
        return [
            # 实体索引
            IndexDefinition(
                name="entity_id_idx",
                index_type="btree",
                node_labels=["Entity"],
                properties=["id"]
            ),
            IndexDefinition(
                name="entity_canonical_idx",
                index_type="btree", 
                node_labels=["Entity"],
                properties=["canonical_form"]
            ),
            IndexDefinition(
                name="entity_type_idx",
                index_type="btree",
                node_labels=["Entity"], 
                properties=["type"]
            ),
            IndexDefinition(
                name="entity_confidence_idx",
                index_type="btree",
                node_labels=["Entity"],
                properties=["confidence"]
            ),
            IndexDefinition(
                name="entity_text_fulltext_idx",
                index_type="fulltext",
                node_labels=["Entity"],
                properties=["text", "canonical_form"]
            ),
            
            # 向量索引（用于语义搜索）
            IndexDefinition(
                name="entity_embedding_vector_idx",
                index_type="vector",
                node_labels=["Entity"],
                properties=["embedding"],
                options={
                    "vector": {
                        "dimensions": 1536,
                        "similarity": "cosine"
                    }
                }
            ),
            
            # 关系索引
            IndexDefinition(
                name="relation_type_idx",
                index_type="btree",
                node_labels=[],  # 用于关系
                properties=["type"]
            ),
            IndexDefinition(
                name="relation_confidence_idx",
                index_type="btree",
                node_labels=[],
                properties=["confidence"]
            ),
            
            # 复合索引
            IndexDefinition(
                name="entity_type_confidence_idx",
                index_type="btree",
                node_labels=["Entity"],
                properties=["type", "confidence"]
            ),
            
            # 文档索引
            IndexDefinition(
                name="document_title_fulltext_idx",
                index_type="fulltext", 
                node_labels=["Document"],
                properties=["title", "content"]
            ),
            
            # 时间索引
            IndexDefinition(
                name="entity_created_at_idx",
                index_type="btree",
                node_labels=["Entity", "Document", "Source"],
                properties=["created_at"]
            ),
            IndexDefinition(
                name="entity_updated_at_idx", 
                index_type="btree",
                node_labels=["Entity", "Document", "Source"],
                properties=["updated_at"]
            )
        ]
    
    def _define_constraints(self) -> List[ConstraintDefinition]:
        """定义约束条件"""
        return [
            # 唯一性约束
            ConstraintDefinition(
                name="entity_id_unique",
                constraint_type="unique",
                node_labels=["Entity"],
                properties=["id"]
            ),
            ConstraintDefinition(
                name="document_id_unique",
                constraint_type="unique", 
                node_labels=["Document"],
                properties=["id"]
            ),
            ConstraintDefinition(
                name="source_id_unique",
                constraint_type="unique",
                node_labels=["Source"], 
                properties=["id"]
            ),
        ]
    
    def get_cypher_statements(self) -> List[str]:
        """获取所有模式定义的Cypher语句"""
        statements = []
        
        # 约束语句（先创建约束）
        for constraint in self.constraints:
            statements.append(constraint.to_cypher().strip())
        
        # 索引语句
        for index in self.indexes:
            statements.append(index.to_cypher().strip())
        
        return statements
    
    def validate_node(self, node_type: str, properties: Dict[str, Any]) -> List[str]:
        """验证节点属性"""
        errors = []
        
        if node_type not in self.node_types:
            errors.append(f"未知的节点类型: {node_type}")
            return errors
        
        schema = self.node_types[node_type]
        
        # 检查必需属性
        for prop in schema["required_properties"]:
            if prop not in properties:
                errors.append(f"缺少必需属性: {prop}")
        
        # 检查实体类型有效性
        if node_type == "Entity" and "type" in properties:
            entity_type = properties["type"]
            if entity_type not in schema["allowed_entity_types"]:
                errors.append(f"无效的实体类型: {entity_type}")
        
        # 检查置信度范围
        if "confidence" in properties:
            confidence = properties["confidence"]
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                errors.append("置信度必须在0-1之间")
        
        return errors
    
    def validate_relationship(self, rel_type: str, properties: Dict[str, Any]) -> List[str]:
        """验证关系属性"""
        errors = []
        
        if rel_type not in self.relation_types:
            errors.append(f"未知的关系类型: {rel_type}")
            return errors
        
        schema = self.relation_types[rel_type]
        
        # 检查必需属性
        for prop in schema["required_properties"]:
            if prop not in properties:
                errors.append(f"缺少必需属性: {prop}")
        
        # 检查关系类型有效性
        if rel_type == "RELATION" and "type" in properties:
            relation_type = properties["type"]
            if relation_type not in schema["allowed_relation_types"]:
                errors.append(f"无效的关系类型: {relation_type}")
        
        # 检查置信度范围
        if "confidence" in properties:
            confidence = properties["confidence"]
            if not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                errors.append("置信度必须在0-1之间")
        
        return errors
    
    def get_schema_info(self) -> Dict[str, Any]:
        """获取模式信息"""
        return {
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "node_types": list(self.node_types.keys()),
            "relation_types": list(self.relation_types.keys()),
            "indexes_count": len(self.indexes),
            "constraints_count": len(self.constraints),
            "supported_entity_types": len([t.value for t in EntityType]),
            "supported_relation_types": len([t.value for t in RelationType])
        }

class SchemaManager:
    """图谱模式管理器"""
    
    def __init__(self, graph_db: Neo4jGraphDatabase):
        self.graph_db = graph_db
        self.schema = GraphSchema()
        self._schema_applied = False
    
    async def initialize_schema(self) -> Dict[str, Any]:
        """初始化图谱模式"""
        try:
            result = {
                "success": True,
                "created_constraints": 0,
                "created_indexes": 0,
                "errors": []
            }
            
            statements = self.schema.get_cypher_statements()
            
            for statement in statements:
                try:
                    await self.graph_db.execute_write_query(statement)
                    
                    if "CONSTRAINT" in statement:
                        result["created_constraints"] += 1
                    elif "INDEX" in statement:
                        result["created_indexes"] += 1
                        
                except Exception as e:
                    error_msg = f"执行语句失败: {statement[:100]}... 错误: {str(e)}"
                    result["errors"].append(error_msg)
                    logger.warning(error_msg)
            
            self._schema_applied = True
            result["total_statements"] = len(statements)
            
            logger.info(f"模式初始化完成: 创建了 {result['created_constraints']} 个约束和 {result['created_indexes']} 个索引")
            
            return result
            
        except Exception as e:
            logger.error(f"模式初始化失败: {str(e)}")
            raise
    
    async def validate_schema(self) -> Dict[str, Any]:
        """验证图谱模式状态"""
        try:
            # 检查约束
            constraints_query = """
            CALL db.constraints() YIELD name, type, entityType, properties, options
            RETURN name, type, entityType, properties, options
            """
            
            # 检查索引
            indexes_query = """
            CALL db.indexes() YIELD name, type, entityType, labelsOrTypes, properties, options, state
            RETURN name, type, entityType, labelsOrTypes, properties, options, state
            """
            
            constraints_result = await self.graph_db.execute_read_query(constraints_query)
            indexes_result = await self.graph_db.execute_read_query(indexes_query)
            
            return {
                "constraints": constraints_result,
                "indexes": indexes_result,
                "schema_applied": self._schema_applied,
                "schema_version": self.schema.version
            }
            
        except Exception as e:
            logger.error(f"模式验证失败: {str(e)}")
            raise
    
    async def drop_schema(self) -> Dict[str, Any]:
        """删除图谱模式（谨慎使用）"""
        try:
            result = {
                "success": True,
                "dropped_constraints": 0,
                "dropped_indexes": 0,
                "errors": []
            }
            
            # 获取现有约束和索引
            validation = await self.validate_schema()
            
            # 删除约束
            for constraint in validation["constraints"]:
                try:
                    drop_statement = f"DROP CONSTRAINT {constraint['name']}"
                    await self.graph_db.execute_write_query(drop_statement)
                    result["dropped_constraints"] += 1
                except Exception as e:
                    result["errors"].append(f"删除约束失败: {constraint['name']}: {str(e)}")
            
            # 删除索引
            for index in validation["indexes"]:
                try:
                    drop_statement = f"DROP INDEX {index['name']}"
                    await self.graph_db.execute_write_query(drop_statement)
                    result["dropped_indexes"] += 1
                except Exception as e:
                    result["errors"].append(f"删除索引失败: {index['name']}: {str(e)}")
            
            self._schema_applied = False
            logger.warning(f"模式已删除: 删除了 {result['dropped_constraints']} 个约束和 {result['dropped_indexes']} 个索引")
            
            return result
            
        except Exception as e:
            logger.error(f"模式删除失败: {str(e)}")
            raise
    
    def get_schema(self) -> GraphSchema:
        """获取当前模式"""
        return self.schema
    
    async def update_schema_version(self, new_version: str) -> None:
        """更新模式版本"""
        self.schema.version = new_version
        self.schema.updated_at = utc_now()
        
        # 记录版本更新
        version_query = """
        MERGE (v:SchemaVersion {version: $version})
        SET v.updated_at = $updated_at, v.applied = $applied
        """
        
        await self.graph_db.execute_write_query(
            version_query,
            {
                "version": new_version,
                "updated_at": self.schema.updated_at.isoformat(),
                "applied": self._schema_applied
            }
        )
        
        logger.info(f"模式版本已更新到: {new_version}")
    
    async def get_schema_statistics(self) -> Dict[str, Any]:
        """获取模式统计信息"""
        try:
            # 统计各类型节点数量
            node_stats_query = """
            CALL db.labels() YIELD label
            CALL {
                WITH label
                CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) as count', {})
                YIELD value
                RETURN label, value.count as count
            }
            RETURN label, count
            """
            
            # 统计关系类型数量
            rel_stats_query = """
            CALL db.relationshipTypes() YIELD relationshipType
            CALL {
                WITH relationshipType  
                CALL apoc.cypher.run('MATCH ()-[r:`' + relationshipType + '`]->() RETURN count(r) as count', {})
                YIELD value
                RETURN relationshipType, value.count as count
            }
            RETURN relationshipType, count
            """
            
            try:
                node_stats = await self.graph_db.execute_read_query(node_stats_query)
            except Exception:
                # 如果APOC不可用，使用基本统计
                node_stats = await self.graph_db.execute_read_query(
                    "MATCH (n) RETURN labels(n) as labels, count(n) as count"
                )
            
            try:
                rel_stats = await self.graph_db.execute_read_query(rel_stats_query)
            except Exception:
                # 如果APOC不可用，使用基本统计
                rel_stats = await self.graph_db.execute_read_query(
                    "MATCH ()-[r]->() RETURN type(r) as relationshipType, count(r) as count"
                )
            
            return {
                "schema_info": self.schema.get_schema_info(),
                "node_statistics": node_stats,
                "relationship_statistics": rel_stats,
                "schema_applied": self._schema_applied
            }
        
        except Exception as e:
            logger.error(f"获取模式统计失败: {str(e)}")
            raise

_schema_manager_instance: Optional['SchemaManager'] = None

async def get_schema_manager() -> 'SchemaManager':
    """获取模式管理器单例"""
    global _schema_manager_instance
    if _schema_manager_instance is None:
        graph_db = await get_graph_database()
        manager = SchemaManager(graph_db)
        try:
            await manager.initialize_schema()
        except Exception as e:
            logger.warning(f"模式初始化失败，继续使用未应用状态: {e}")
        _schema_manager_instance = manager
    return _schema_manager_instance
