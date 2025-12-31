"""
知识图谱操作引擎
提供高级的CRUD操作、查询构建器、事务处理和批量操作
"""

from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from dataclasses import dataclass
from enum import Enum
import uuid
import json
from .graph_database import Neo4jGraphDatabase
from .schema import GraphNode, GraphEdge, GraphNodeType, GraphEdgeType, SchemaManager
from .data_models import Entity, Relation, EntityType, RelationType

from src.core.logging import get_logger
logger = get_logger(__name__)

class QueryType(str, Enum):
    """查询类型"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    MATCH = "match"
    MERGE = "merge"

@dataclass
class QueryResult:
    """查询结果"""
    success: bool
    data: List[Dict[str, Any]]
    affected_count: int
    execution_time_ms: float
    query_type: QueryType
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "affected_count": self.affected_count,
            "execution_time_ms": self.execution_time_ms,
            "query_type": self.query_type.value,
            "error_message": self.error_message
        }

class CypherQueryBuilder:
    """Cypher查询构建器"""
    
    def __init__(self):
        self.query_parts = {
            "match": [],
            "where": [],
            "create": [],
            "merge": [],
            "set": [],
            "remove": [],
            "return": [],
            "order_by": [],
            "limit": None,
            "skip": None
        }
        self.parameters = {}
    
    def match(self, pattern: str) -> "CypherQueryBuilder":
        """添加MATCH子句"""
        self.query_parts["match"].append(pattern)
        return self
    
    def where(self, condition: str) -> "CypherQueryBuilder":
        """添加WHERE条件"""
        self.query_parts["where"].append(condition)
        return self
    
    def create(self, pattern: str) -> "CypherQueryBuilder":
        """添加CREATE子句"""
        self.query_parts["create"].append(pattern)
        return self
    
    def merge(self, pattern: str) -> "CypherQueryBuilder":
        """添加MERGE子句"""
        self.query_parts["merge"].append(pattern)
        return self
    
    def set(self, assignment: str) -> "CypherQueryBuilder":
        """添加SET子句"""
        self.query_parts["set"].append(assignment)
        return self
    
    def remove(self, property_path: str) -> "CypherQueryBuilder":
        """添加REMOVE子句"""
        self.query_parts["remove"].append(property_path)
        return self
    
    def return_(self, fields: str) -> "CypherQueryBuilder":
        """添加RETURN子句"""
        self.query_parts["return"].append(fields)
        return self
    
    def order_by(self, field: str, direction: str = "ASC") -> "CypherQueryBuilder":
        """添加ORDER BY子句"""
        self.query_parts["order_by"].append(f"{field} {direction}")
        return self
    
    def limit(self, count: int) -> "CypherQueryBuilder":
        """添加LIMIT子句"""
        self.query_parts["limit"] = count
        return self
    
    def skip(self, count: int) -> "CypherQueryBuilder":
        """添加SKIP子句"""
        self.query_parts["skip"] = count
        return self
    
    def with_parameter(self, name: str, value: Any) -> "CypherQueryBuilder":
        """添加参数"""
        self.parameters[name] = value
        return self
    
    def build(self) -> Tuple[str, Dict[str, Any]]:
        """构建查询"""
        query_parts = []
        
        # MATCH
        if self.query_parts["match"]:
            query_parts.append("MATCH " + ", ".join(self.query_parts["match"]))
        
        # CREATE
        if self.query_parts["create"]:
            query_parts.append("CREATE " + ", ".join(self.query_parts["create"]))
        
        # MERGE
        if self.query_parts["merge"]:
            query_parts.append("MERGE " + ", ".join(self.query_parts["merge"]))
        
        # WHERE
        if self.query_parts["where"]:
            query_parts.append("WHERE " + " AND ".join(self.query_parts["where"]))
        
        # SET
        if self.query_parts["set"]:
            query_parts.append("SET " + ", ".join(self.query_parts["set"]))
        
        # REMOVE
        if self.query_parts["remove"]:
            query_parts.append("REMOVE " + ", ".join(self.query_parts["remove"]))
        
        # RETURN
        if self.query_parts["return"]:
            query_parts.append("RETURN " + ", ".join(self.query_parts["return"]))
        
        # ORDER BY
        if self.query_parts["order_by"]:
            query_parts.append("ORDER BY " + ", ".join(self.query_parts["order_by"]))
        
        # SKIP
        if self.query_parts["skip"] is not None:
            query_parts.append(f"SKIP {self.query_parts['skip']}")
        
        # LIMIT
        if self.query_parts["limit"] is not None:
            query_parts.append(f"LIMIT {self.query_parts['limit']}")
        
        return "\n".join(query_parts), self.parameters

class GraphOperations:
    """图谱操作引擎"""
    
    def __init__(self, graph_db: Neo4jGraphDatabase, schema_manager: SchemaManager):
        self.graph_db = graph_db
        self.schema_manager = schema_manager
    
    async def create_entity(self, 
                           entity_id: str,
                           entity_type: str,
                           canonical_form: str,
                           properties: Dict[str, Any]) -> QueryResult:
        """创建实体"""
        try:
            start_time = utc_now()
            
            # 验证实体类型
            if entity_type not in [t.value for t in EntityType]:
                raise ValueError(f"无效的实体类型: {entity_type}")
            
            # 准备属性
            node_props = {
                "id": entity_id,
                "canonical_form": canonical_form,
                "type": entity_type,
                "created_at": utc_now().isoformat(),
                "updated_at": utc_now().isoformat(),
                "confidence": properties.get("confidence", 1.0),
                **properties
            }
            
            # 验证节点属性
            validation_errors = self.schema_manager.get_schema().validate_node("Entity", node_props)
            if validation_errors:
                raise ValueError(f"节点验证失败: {'; '.join(validation_errors)}")
            
            # 构建查询
            query = """
            CREATE (e:Entity $props)
            RETURN e.id as entity_id, e.canonical_form as canonical_form
            """
            
            result = await self.graph_db.execute_write_query(query, {"props": node_props})
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return QueryResult(
                success=True,
                data=result,
                affected_count=len(result),
                execution_time_ms=execution_time,
                query_type=QueryType.CREATE
            )
            
        except Exception as e:
            logger.error(f"创建实体失败: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                affected_count=0,
                execution_time_ms=0,
                query_type=QueryType.CREATE,
                error_message=str(e)
            )
    
    async def update_entity(self,
                           entity_id: str,
                           updates: Dict[str, Any]) -> QueryResult:
        """更新实体"""
        try:
            start_time = utc_now()
            
            # 添加更新时间戳
            updates["updated_at"] = utc_now().isoformat()
            
            # 构建SET子句
            set_clauses = []
            parameters = {"entity_id": entity_id}
            
            for key, value in updates.items():
                param_name = f"update_{key}"
                set_clauses.append(f"e.{key} = ${param_name}")
                parameters[param_name] = value
            
            query = f"""
            MATCH (e:Entity {{id: $entity_id}})
            SET {", ".join(set_clauses)}
            RETURN e.id as entity_id, e.canonical_form as canonical_form, e.updated_at as updated_at
            """
            
            result = await self.graph_db.execute_write_query(query, parameters)
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return QueryResult(
                success=True,
                data=result,
                affected_count=len(result),
                execution_time_ms=execution_time,
                query_type=QueryType.UPDATE
            )
            
        except Exception as e:
            logger.error(f"更新实体失败: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                affected_count=0,
                execution_time_ms=0,
                query_type=QueryType.UPDATE,
                error_message=str(e)
            )
    
    async def get_entity(self, entity_id: str) -> QueryResult:
        """获取实体"""
        try:
            start_time = utc_now()
            
            query = """
            MATCH (e:Entity {id: $entity_id})
            RETURN e.id as id, e.canonical_form as canonical_form, e.type as type,
                   e.confidence as confidence, e.created_at as created_at,
                   e.updated_at as updated_at, properties(e) as properties
            """
            
            result = await self.graph_db.execute_read_query(query, {"entity_id": entity_id})
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return QueryResult(
                success=True,
                data=result,
                affected_count=len(result),
                execution_time_ms=execution_time,
                query_type=QueryType.READ
            )
            
        except Exception as e:
            logger.error(f"获取实体失败: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                affected_count=0,
                execution_time_ms=0,
                query_type=QueryType.READ,
                error_message=str(e)
            )
    
    async def delete_entity(self, entity_id: str, cascade: bool = False) -> QueryResult:
        """删除实体"""
        try:
            start_time = utc_now()
            
            if cascade:
                # 级联删除：删除实体及其所有关系
                query = """
                MATCH (e:Entity {id: $entity_id})
                DETACH DELETE e
                RETURN count(e) as deleted_count
                """
            else:
                # 仅删除实体（如果有关系会失败）
                query = """
                MATCH (e:Entity {id: $entity_id})
                DELETE e
                RETURN count(e) as deleted_count
                """
            
            result = await self.graph_db.execute_write_query(query, {"entity_id": entity_id})
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            deleted_count = result[0]["deleted_count"] if result else 0
            
            return QueryResult(
                success=True,
                data=result,
                affected_count=deleted_count,
                execution_time_ms=execution_time,
                query_type=QueryType.DELETE
            )
            
        except Exception as e:
            logger.error(f"删除实体失败: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                affected_count=0,
                execution_time_ms=0,
                query_type=QueryType.DELETE,
                error_message=str(e)
            )
    
    async def create_relationship(self,
                                 source_entity_id: str,
                                 target_entity_id: str,
                                 relation_type: str,
                                 properties: Dict[str, Any]) -> QueryResult:
        """创建关系"""
        try:
            start_time = utc_now()
            
            # 验证关系类型
            if relation_type not in [t.value for t in RelationType]:
                raise ValueError(f"无效的关系类型: {relation_type}")
            
            # 准备关系属性
            rel_props = {
                "id": str(uuid.uuid4()),
                "type": relation_type,
                "created_at": utc_now().isoformat(),
                "updated_at": utc_now().isoformat(),
                "confidence": properties.get("confidence", 1.0),
                **properties
            }
            
            # 验证关系属性
            validation_errors = self.schema_manager.get_schema().validate_relationship("RELATION", rel_props)
            if validation_errors:
                raise ValueError(f"关系验证失败: {'; '.join(validation_errors)}")
            
            query = """
            MATCH (source:Entity {id: $source_id}), (target:Entity {id: $target_id})
            CREATE (source)-[r:RELATION $props]->(target)
            RETURN r.id as relation_id, r.type as relation_type, r.confidence as confidence
            """
            
            result = await self.graph_db.execute_write_query(
                query, 
                {
                    "source_id": source_entity_id,
                    "target_id": target_entity_id,
                    "props": rel_props
                }
            )
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return QueryResult(
                success=True,
                data=result,
                affected_count=len(result),
                execution_time_ms=execution_time,
                query_type=QueryType.CREATE
            )
            
        except Exception as e:
            logger.error(f"创建关系失败: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                affected_count=0,
                execution_time_ms=0,
                query_type=QueryType.CREATE,
                error_message=str(e)
            )
    
    async def get_entity_relationships(self,
                                     entity_id: str,
                                     direction: str = "both",
                                     relation_types: Optional[List[str]] = None,
                                     limit: Optional[int] = None) -> QueryResult:
        """获取实体的关系"""
        try:
            start_time = utc_now()
            
            # 构建方向模式
            if direction == "outgoing":
                pattern = "(e)-[r:RELATION]->(target)"
                return_clause = "target.id as target_id, target.canonical_form as target_canonical_form"
            elif direction == "incoming":
                pattern = "(source)-[r:RELATION]->(e)"
                return_clause = "source.id as source_id, source.canonical_form as source_canonical_form"
            else:  # both
                pattern = "(e)-[r:RELATION]-(other)"
                return_clause = "other.id as other_id, other.canonical_form as other_canonical_form"
            
            # 构建查询
            query_builder = (CypherQueryBuilder()
                           .match(f"(e:Entity {{id: $entity_id}})")
                           .match(pattern))
            
            # 添加关系类型过滤
            if relation_types:
                type_conditions = " OR ".join([f"r.type = '{rt}'" for rt in relation_types])
                query_builder.where(f"({type_conditions})")
            
            query_builder.return_(f"r.id as relation_id, r.type as relation_type, r.confidence as confidence, {return_clause}")
            
            if limit:
                query_builder.limit(limit)
            
            query, parameters = query_builder.build()
            parameters["entity_id"] = entity_id
            
            result = await self.graph_db.execute_read_query(query, parameters)
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return QueryResult(
                success=True,
                data=result,
                affected_count=len(result),
                execution_time_ms=execution_time,
                query_type=QueryType.READ
            )
            
        except Exception as e:
            logger.error(f"获取实体关系失败: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                affected_count=0,
                execution_time_ms=0,
                query_type=QueryType.READ,
                error_message=str(e)
            )
    
    async def find_entities(self,
                           filters: Dict[str, Any],
                           limit: int = 100,
                           skip: int = 0) -> QueryResult:
        """查找实体"""
        try:
            start_time = utc_now()
            
            query_builder = CypherQueryBuilder().match("(e:Entity)")
            
            # 构建过滤条件
            parameters = {}
            for key, value in filters.items():
                param_name = f"filter_{key}"
                if key == "canonical_form_contains":
                    query_builder.where(f"e.canonical_form CONTAINS ${param_name}")
                elif key == "type":
                    query_builder.where(f"e.type = ${param_name}")
                elif key == "confidence_gte":
                    query_builder.where(f"e.confidence >= ${param_name}")
                elif key == "confidence_lte":
                    query_builder.where(f"e.confidence <= ${param_name}")
                elif key == "created_after":
                    query_builder.where(f"e.created_at > ${param_name}")
                elif key == "created_before":
                    query_builder.where(f"e.created_at < ${param_name}")
                else:
                    query_builder.where(f"e.{key} = ${param_name}")
                
                parameters[param_name] = value
            
            query_builder.return_(
                "e.id as id, e.canonical_form as canonical_form, e.type as type, "
                "e.confidence as confidence, e.created_at as created_at, properties(e) as properties"
            ).skip(skip).limit(limit)
            
            query, query_params = query_builder.build()
            parameters.update(query_params)
            
            result = await self.graph_db.execute_read_query(query, parameters)
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return QueryResult(
                success=True,
                data=result,
                affected_count=len(result),
                execution_time_ms=execution_time,
                query_type=QueryType.READ
            )
            
        except Exception as e:
            logger.error(f"查找实体失败: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                affected_count=0,
                execution_time_ms=0,
                query_type=QueryType.READ,
                error_message=str(e)
            )
    
    async def find_shortest_path(self,
                                source_entity_id: str,
                                target_entity_id: str,
                                max_depth: int = 5) -> QueryResult:
        """查找最短路径"""
        try:
            start_time = utc_now()
            
            query = """
            MATCH path = shortestPath((source:Entity {id: $source_id})-[*..%d]-(target:Entity {id: $target_id}))
            RETURN 
                path,
                length(path) as path_length,
                [node in nodes(path) | {id: node.id, canonical_form: node.canonical_form, type: node.type}] as nodes,
                [rel in relationships(path) | {id: rel.id, type: rel.type, confidence: rel.confidence}] as relationships
            """ % max_depth
            
            result = await self.graph_db.execute_read_query(
                query,
                {"source_id": source_entity_id, "target_id": target_entity_id}
            )
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return QueryResult(
                success=True,
                data=result,
                affected_count=len(result),
                execution_time_ms=execution_time,
                query_type=QueryType.READ
            )
            
        except Exception as e:
            logger.error(f"查找最短路径失败: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                affected_count=0,
                execution_time_ms=0,
                query_type=QueryType.READ,
                error_message=str(e)
            )
    
    async def get_subgraph(self,
                          center_entity_id: str,
                          depth: int = 2,
                          max_nodes: int = 100) -> QueryResult:
        """获取子图"""
        try:
            start_time = utc_now()
            
            query = f"""
            MATCH path = (center:Entity {{id: $center_id}})-[*..{depth}]-(node)
            WITH collect(DISTINCT node) + collect(DISTINCT center) as nodes, 
                 collect(DISTINCT path) as paths
            UNWIND nodes as n
            WITH collect(DISTINCT n)[..{max_nodes}] as limited_nodes, paths
            UNWIND limited_nodes as node
            MATCH (node)-[rel]-(connected)
            WHERE connected in limited_nodes
            RETURN 
                collect(DISTINCT {{
                    id: node.id, 
                    canonical_form: node.canonical_form, 
                    type: node.type,
                    properties: properties(node)
                }}) as nodes,
                collect(DISTINCT {{
                    id: rel.id,
                    type: rel.type,
                    source: startNode(rel).id,
                    target: endNode(rel).id,
                    properties: properties(rel)
                }}) as relationships
            """
            
            result = await self.graph_db.execute_read_query(query, {"center_id": center_entity_id})
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return QueryResult(
                success=True,
                data=result,
                affected_count=len(result),
                execution_time_ms=execution_time,
                query_type=QueryType.READ
            )
            
        except Exception as e:
            logger.error(f"获取子图失败: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                affected_count=0,
                execution_time_ms=0,
                query_type=QueryType.READ,
                error_message=str(e)
            )
    
    async def batch_create_entities(self, entities: List[Dict[str, Any]]) -> QueryResult:
        """批量创建实体"""
        try:
            start_time = utc_now()
            
            # 准备批量数据
            batch_data = []
            for entity_data in entities:
                entity_props = {
                    "id": entity_data.get("id", str(uuid.uuid4())),
                    "canonical_form": entity_data["canonical_form"],
                    "type": entity_data["type"],
                    "created_at": utc_now().isoformat(),
                    "updated_at": utc_now().isoformat(),
                    "confidence": entity_data.get("confidence", 1.0),
                    **{k: v for k, v in entity_data.items() if k not in ["id", "canonical_form", "type", "confidence"]}
                }
                batch_data.append(entity_props)
            
            query = """
            UNWIND $batch as entity_data
            CREATE (e:Entity)
            SET e = entity_data
            RETURN e.id as entity_id
            """
            
            result = await self.graph_db.execute_write_query(query, {"batch": batch_data})
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return QueryResult(
                success=True,
                data=result,
                affected_count=len(result),
                execution_time_ms=execution_time,
                query_type=QueryType.CREATE
            )
            
        except Exception as e:
            logger.error(f"批量创建实体失败: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                affected_count=0,
                execution_time_ms=0,
                query_type=QueryType.CREATE,
                error_message=str(e)
            )
    
    async def execute_custom_query(self,
                                  query: str,
                                  parameters: Dict[str, Any],
                                  read_only: bool = True) -> QueryResult:
        """执行自定义查询"""
        try:
            start_time = utc_now()
            
            if read_only:
                result = await self.graph_db.execute_read_query(query, parameters)
            else:
                result = await self.graph_db.execute_write_query(query, parameters)
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return QueryResult(
                success=True,
                data=result,
                affected_count=len(result),
                execution_time_ms=execution_time,
                query_type=QueryType.READ if read_only else QueryType.CREATE
            )
            
        except Exception as e:
            logger.error(f"执行自定义查询失败: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                affected_count=0,
                execution_time_ms=0,
                query_type=QueryType.READ if read_only else QueryType.CREATE,
                error_message=str(e)
            )
    
    async def get_graph_statistics(self) -> QueryResult:
        """获取图统计信息"""
        try:
            start_time = utc_now()
            
            stats_queries = {
                "entity_count": "MATCH (e:Entity) RETURN count(e) as count",
                "relation_count": "MATCH ()-[r:RELATION]->() RETURN count(r) as count",
                "entity_types": "MATCH (e:Entity) RETURN e.type as type, count(e) as count ORDER BY count DESC",
                "relation_types": "MATCH ()-[r:RELATION]->() RETURN r.type as type, count(r) as count ORDER BY count DESC",
                "avg_confidence": "MATCH (e:Entity) RETURN avg(e.confidence) as avg_confidence",
                "recent_entities": "MATCH (e:Entity) WHERE e.created_at >= datetime() - duration('P7D') RETURN count(e) as count"
            }
            
            stats_result = {}
            
            for stat_name, query in stats_queries.items():
                try:
                    result = await self.graph_db.execute_read_query(query)
                    stats_result[stat_name] = result
                except Exception as e:
                    logger.warning(f"统计查询失败 {stat_name}: {str(e)}")
                    stats_result[stat_name] = None
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            return QueryResult(
                success=True,
                data=[{"statistics": stats_result}],
                affected_count=1,
                execution_time_ms=execution_time,
                query_type=QueryType.READ
            )
            
        except Exception as e:
            logger.error(f"获取图统计信息失败: {str(e)}")
            return QueryResult(
                success=False,
                data=[],
                affected_count=0,
                execution_time_ms=0,
                query_type=QueryType.READ,
                error_message=str(e)
            )
