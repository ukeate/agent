"""
知识图谱数据迁移和升级工具
支持模式升级、数据迁移、版本控制和回滚操作
"""

from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import json
from .graph_database import Neo4jGraphDatabase
from .schema import SchemaManager

from src.core.logging import get_logger
logger = get_logger(__name__)

class MigrationType(str, Enum):
    """迁移类型"""
    SCHEMA_UPDATE = "schema_update"
    DATA_TRANSFORMATION = "data_transformation"
    INDEX_REBUILD = "index_rebuild"
    CONSTRAINT_UPDATE = "constraint_update"
    DATA_CLEANUP = "data_cleanup"

class MigrationStatus(str, Enum):
    """迁移状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"

@dataclass
class Migration:
    """迁移定义"""
    id: str
    name: str
    description: str
    version: str
    migration_type: MigrationType
    up_statements: List[str]
    down_statements: List[str]  # 回滚语句
    dependencies: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=utc_factory)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "migration_type": self.migration_type.value,
            "up_statements": self.up_statements,
            "down_statements": self.down_statements,
            "dependencies": self.dependencies,
            "created_at": self.created_at.isoformat()
        }

@dataclass
class MigrationRecord:
    """迁移执行记录"""
    migration_id: str
    status: MigrationStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    rollback_at: Optional[datetime] = None
    execution_time_ms: Optional[float] = None
    affected_nodes: int = 0
    affected_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "migration_id": self.migration_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "rollback_at": self.rollback_at.isoformat() if self.rollback_at else None,
            "execution_time_ms": self.execution_time_ms,
            "affected_nodes": self.affected_nodes,
            "affected_relationships": self.affected_relationships
        }

class MigrationManager:
    """数据迁移管理器"""
    
    def __init__(self, graph_db: Neo4jGraphDatabase, schema_manager: SchemaManager):
        self.graph_db = graph_db
        self.schema_manager = schema_manager
        self.migrations: Dict[str, Migration] = {}
        self.migration_records: List[MigrationRecord] = []
        
        # 预定义迁移
        self._register_default_migrations()
    
    def _register_default_migrations(self):
        """注册默认迁移"""
        
        # 初始化模式迁移
        init_schema = Migration(
            id="001_init_schema",
            name="Initialize Graph Schema",
            description="创建基础图谱模式，包括节点类型、关系类型、索引和约束",
            version="1.0.0",
            migration_type=MigrationType.SCHEMA_UPDATE,
            up_statements=self.schema_manager.get_schema().get_cypher_statements(),
            down_statements=[
                "CALL db.constraints() YIELD name CALL { WITH name CALL apoc.cypher.run('DROP CONSTRAINT ' + name, {}) YIELD value RETURN value } RETURN count(*)",
                "CALL db.indexes() YIELD name CALL { WITH name CALL apoc.cypher.run('DROP INDEX ' + name, {}) YIELD value RETURN value } RETURN count(*)"
            ]
        )
        self.register_migration(init_schema)
        
        # 添加实体嵌入向量迁移
        add_embeddings = Migration(
            id="002_add_entity_embeddings",
            name="Add Entity Embeddings",
            description="为现有实体添加嵌入向量属性",
            version="1.1.0",
            migration_type=MigrationType.DATA_TRANSFORMATION,
            up_statements=[
                """
                MATCH (e:Entity)
                WHERE e.embedding IS NULL
                SET e.embedding = []
                RETURN count(e) as updated_count
                """
            ],
            down_statements=[
                """
                MATCH (e:Entity)
                WHERE e.embedding IS NOT NULL
                REMOVE e.embedding
                RETURN count(e) as updated_count
                """
            ],
            dependencies=["001_init_schema"]
        )
        self.register_migration(add_embeddings)
        
        # 添加时间戳索引迁移
        add_timestamp_indexes = Migration(
            id="003_add_timestamp_indexes",
            name="Add Timestamp Indexes",
            description="为created_at和updated_at字段创建索引以提高查询性能",
            version="1.2.0",
            migration_type=MigrationType.INDEX_REBUILD,
            up_statements=[
                """
                CREATE INDEX entity_created_at_idx IF NOT EXISTS
                FOR (e:Entity) ON (e.created_at)
                """,
                """
                CREATE INDEX entity_updated_at_idx IF NOT EXISTS  
                FOR (e:Entity) ON (e.updated_at)
                """,
                """
                CREATE INDEX relation_created_at_idx IF NOT EXISTS
                FOR ()-[r:RELATION]-() ON (r.created_at)
                """
            ],
            down_statements=[
                "DROP INDEX entity_created_at_idx IF EXISTS",
                "DROP INDEX entity_updated_at_idx IF EXISTS",
                "DROP INDEX relation_created_at_idx IF EXISTS"
            ],
            dependencies=["001_init_schema"]
        )
        self.register_migration(add_timestamp_indexes)
        
        # 数据质量清理迁移
        data_quality_cleanup = Migration(
            id="004_data_quality_cleanup",
            name="Data Quality Cleanup",
            description="清理低质量数据，删除置信度过低的实体和关系",
            version="1.3.0",
            migration_type=MigrationType.DATA_CLEANUP,
            up_statements=[
                """
                MATCH (e:Entity)
                WHERE e.confidence < 0.1
                DETACH DELETE e
                """,
                """
                MATCH ()-[r:RELATION]->()
                WHERE r.confidence < 0.1
                DELETE r
                """
            ],
            down_statements=[
                # 无法回滚数据删除，只能记录
                "RETURN 'Data cleanup cannot be rolled back' as warning"
            ],
            dependencies=["002_add_entity_embeddings"]
        )
        self.register_migration(data_quality_cleanup)
    
    def register_migration(self, migration: Migration):
        """注册迁移"""
        self.migrations[migration.id] = migration
        logger.info(f"注册迁移: {migration.id} - {migration.name}")
    
    async def get_applied_migrations(self) -> List[str]:
        """获取已应用的迁移列表"""
        try:
            query = """
            MATCH (m:Migration)
            WHERE m.status = 'completed'
            RETURN m.migration_id as migration_id
            ORDER BY m.applied_at
            """
            
            result = await self.graph_db.execute_read_query(query)
            return [record["migration_id"] for record in result]
            
        except Exception as e:
            logger.error(f"获取已应用迁移失败: {str(e)}")
            # 如果Migration节点不存在，返回空列表
            return []
    
    async def get_pending_migrations(self) -> List[Migration]:
        """获取待应用的迁移"""
        applied_migrations = await self.get_applied_migrations()
        
        pending = []
        for migration_id, migration in self.migrations.items():
            if migration_id not in applied_migrations:
                # 检查依赖是否满足
                dependencies_met = all(
                    dep in applied_migrations for dep in migration.dependencies
                )
                if dependencies_met:
                    pending.append(migration)
        
        # 按版本排序
        pending.sort(key=lambda m: m.version)
        return pending
    
    async def apply_migration(self, migration: Migration) -> MigrationRecord:
        """应用单个迁移"""
        record = MigrationRecord(
            migration_id=migration.id,
            status=MigrationStatus.RUNNING,
            started_at=utc_now()
        )
        
        try:
            logger.info(f"开始应用迁移: {migration.id} - {migration.name}")
            
            start_time = utc_now()
            
            # 执行迁移语句
            for statement in migration.up_statements:
                if statement.strip():
                    result = await self.graph_db.execute_write_query(statement)
                    logger.debug(f"执行语句: {statement[:100]}...")
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            # 记录迁移成功
            record.status = MigrationStatus.COMPLETED
            record.completed_at = end_time
            record.execution_time_ms = execution_time
            
            # 保存迁移记录到数据库
            await self._save_migration_record(migration, record)
            
            logger.info(f"迁移应用成功: {migration.id}, 耗时: {execution_time:.2f}ms")
            
        except Exception as e:
            record.status = MigrationStatus.FAILED
            record.error_message = str(e)
            record.completed_at = utc_now()
            
            logger.error(f"迁移应用失败: {migration.id}: {str(e)}")
            
            # 尝试保存失败记录
            try:
                await self._save_migration_record(migration, record)
            except Exception:
                logger.error(f"保存迁移记录失败: {migration.id}")
        
        self.migration_records.append(record)
        return record
    
    async def rollback_migration(self, migration_id: str) -> MigrationRecord:
        """回滚迁移"""
        if migration_id not in self.migrations:
            raise ValueError(f"未知的迁移ID: {migration_id}")
        
        migration = self.migrations[migration_id]
        
        # 检查是否已应用
        applied_migrations = await self.get_applied_migrations()
        if migration_id not in applied_migrations:
            raise ValueError(f"迁移未应用，无法回滚: {migration_id}")
        
        record = MigrationRecord(
            migration_id=migration_id,
            status=MigrationStatus.RUNNING,
            started_at=utc_now()
        )
        
        try:
            logger.info(f"开始回滚迁移: {migration_id} - {migration.name}")
            
            start_time = utc_now()
            
            # 执行回滚语句
            for statement in migration.down_statements:
                if statement.strip():
                    await self.graph_db.execute_write_query(statement)
                    logger.debug(f"执行回滚语句: {statement[:100]}...")
            
            end_time = utc_now()
            execution_time = (end_time - start_time).total_seconds() * 1000
            
            # 更新迁移状态
            record.status = MigrationStatus.ROLLED_BACK
            record.rollback_at = end_time
            record.execution_time_ms = execution_time
            
            # 从数据库中删除迁移记录
            await self._remove_migration_record(migration_id)
            
            logger.info(f"迁移回滚成功: {migration_id}, 耗时: {execution_time:.2f}ms")
            
        except Exception as e:
            record.status = MigrationStatus.FAILED
            record.error_message = f"回滚失败: {str(e)}"
            record.completed_at = utc_now()
            
            logger.error(f"迁移回滚失败: {migration_id}: {str(e)}")
        
        self.migration_records.append(record)
        return record
    
    async def apply_all_pending_migrations(self) -> List[MigrationRecord]:
        """应用所有待处理的迁移"""
        pending_migrations = await self.get_pending_migrations()
        
        if not pending_migrations:
            logger.info("没有待应用的迁移")
            return []
        
        results = []
        for migration in pending_migrations:
            record = await self.apply_migration(migration)
            results.append(record)
            
            # 如果迁移失败，停止后续迁移
            if record.status == MigrationStatus.FAILED:
                logger.error(f"迁移失败，停止后续迁移: {migration.id}")
                break
        
        return results
    
    async def _save_migration_record(self, migration: Migration, record: MigrationRecord):
        """保存迁移记录到数据库"""
        query = """
        MERGE (m:Migration {migration_id: $migration_id})
        SET 
            m.name = $name,
            m.description = $description,
            m.version = $version,
            m.migration_type = $migration_type,
            m.status = $status,
            m.applied_at = $applied_at,
            m.execution_time_ms = $execution_time_ms,
            m.error_message = $error_message
        """
        
        parameters = {
            "migration_id": record.migration_id,
            "name": migration.name,
            "description": migration.description,
            "version": migration.version,
            "migration_type": migration.migration_type.value,
            "status": record.status.value,
            "applied_at": record.completed_at.isoformat() if record.completed_at else None,
            "execution_time_ms": record.execution_time_ms,
            "error_message": record.error_message
        }
        
        await self.graph_db.execute_write_query(query, parameters)
    
    async def _remove_migration_record(self, migration_id: str):
        """从数据库中删除迁移记录"""
        query = """
        MATCH (m:Migration {migration_id: $migration_id})
        DELETE m
        """
        
        await self.graph_db.execute_write_query(query, {"migration_id": migration_id})
    
    async def get_migration_status(self) -> Dict[str, Any]:
        """获取迁移状态概览"""
        applied_migrations = await self.get_applied_migrations()
        pending_migrations = await self.get_pending_migrations()
        
        # 获取最近的迁移记录
        recent_records = sorted(
            self.migration_records, 
            key=lambda r: r.started_at, 
            reverse=True
        )[:10]
        
        return {
            "total_migrations": len(self.migrations),
            "applied_migrations": len(applied_migrations),
            "pending_migrations": len(pending_migrations),
            "applied_migration_ids": applied_migrations,
            "pending_migration_ids": [m.id for m in pending_migrations],
            "recent_migration_records": [r.to_dict() for r in recent_records],
            "last_migration_time": recent_records[0].started_at.isoformat() if recent_records else None
        }
    
    def get_migration_by_id(self, migration_id: str) -> Optional[Migration]:
        """根据ID获取迁移"""
        return self.migrations.get(migration_id)
    
    def list_migrations(self) -> List[Dict[str, Any]]:
        """列出所有迁移"""
        return [migration.to_dict() for migration in self.migrations.values()]

class DataExportImportTool:
    """数据导出导入工具"""
    
    def __init__(self, graph_db: Neo4jGraphDatabase):
        self.graph_db = graph_db
    
    async def export_graph_data(self, 
                               export_format: str = "cypher",
                               include_metadata: bool = True) -> Dict[str, Any]:
        """导出图数据"""
        try:
            logger.info(f"开始导出图数据，格式: {export_format}")
            
            if export_format == "cypher":
                return await self._export_cypher_statements(include_metadata)
            elif export_format == "json":
                return await self._export_json_format(include_metadata)
            else:
                raise ValueError(f"不支持的导出格式: {export_format}")
                
        except Exception as e:
            logger.error(f"导出图数据失败: {str(e)}")
            raise
    
    async def _export_cypher_statements(self, include_metadata: bool) -> Dict[str, Any]:
        """导出为Cypher语句"""
        statements = []
        
        # 导出节点
        nodes_query = """
        MATCH (n)
        RETURN labels(n) as labels, properties(n) as properties
        """
        
        nodes = await self.graph_db.execute_read_query(nodes_query)
        
        for node in nodes:
            labels = ":".join(node["labels"])
            props = node["properties"]
            
            # 构造CREATE语句
            prop_strings = []
            for key, value in props.items():
                if isinstance(value, str):
                    prop_strings.append(f"{key}: '{value}'")
                else:
                    prop_strings.append(f"{key}: {value}")
            
            prop_string = "{" + ", ".join(prop_strings) + "}" if prop_strings else ""
            statement = f"CREATE (:{labels} {prop_string});"
            statements.append(statement)
        
        # 导出关系
        rels_query = """
        MATCH (a)-[r]->(b)
        RETURN id(a) as start_id, id(b) as end_id, type(r) as type, properties(r) as properties
        """
        
        rels = await self.graph_db.execute_read_query(rels_query)
        
        for rel in rels:
            rel_type = rel["type"]
            props = rel["properties"]
            
            prop_strings = []
            for key, value in props.items():
                if isinstance(value, str):
                    prop_strings.append(f"{key}: '{value}'")
                else:
                    prop_strings.append(f"{key}: {value}")
            
            prop_string = "{" + ", ".join(prop_strings) + "}" if prop_strings else ""
            # 注意：这里使用id()的方式在实际导入时需要特殊处理
            statement = f"MATCH (a), (b) WHERE id(a) = {rel['start_id']} AND id(b) = {rel['end_id']} CREATE (a)-[:{rel_type} {prop_string}]->(b);"
            statements.append(statement)
        
        result = {
            "format": "cypher",
            "statements": statements,
            "export_time": utc_now().isoformat(),
            "total_statements": len(statements)
        }
        
        if include_metadata:
            result["metadata"] = await self._get_export_metadata()
        
        return result
    
    async def _export_json_format(self, include_metadata: bool) -> Dict[str, Any]:
        """导出为JSON格式"""
        # 导出所有节点
        nodes_query = """
        MATCH (n)
        RETURN id(n) as id, labels(n) as labels, properties(n) as properties
        """
        
        # 导出所有关系
        rels_query = """
        MATCH (a)-[r]->(b)
        RETURN id(r) as id, id(a) as start_node, id(b) as end_node, 
               type(r) as type, properties(r) as properties
        """
        
        nodes = await self.graph_db.execute_read_query(nodes_query)
        relationships = await self.graph_db.execute_read_query(rels_query)
        
        result = {
            "format": "json",
            "nodes": nodes,
            "relationships": relationships,
            "export_time": utc_now().isoformat(),
            "node_count": len(nodes),
            "relationship_count": len(relationships)
        }
        
        if include_metadata:
            result["metadata"] = await self._get_export_metadata()
        
        return result
    
    async def _get_export_metadata(self) -> Dict[str, Any]:
        """获取导出元数据"""
        stats_query = """
        CALL db.stats.retrieve('GRAPH COUNTS') YIELD section, data
        RETURN section, data
        """
        
        try:
            stats = await self.graph_db.execute_read_query(stats_query)
            return {
                "database_stats": stats,
                "schema_info": "Available via schema manager"
            }
        except Exception:
            # 如果统计查询失败，返回基本信息
            return {
                "export_note": "Metadata collection failed, basic export completed"
            }
    
    async def import_graph_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """导入图数据"""
        try:
            logger.info("开始导入图数据")
            
            format_type = data.get("format", "json")
            
            if format_type == "cypher":
                return await self._import_cypher_statements(data)
            elif format_type == "json":
                return await self._import_json_format(data)
            else:
                raise ValueError(f"不支持的导入格式: {format_type}")
                
        except Exception as e:
            logger.error(f"导入图数据失败: {str(e)}")
            raise
    
    async def _import_cypher_statements(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """导入Cypher语句"""
        statements = data.get("statements", [])
        
        imported_count = 0
        failed_count = 0
        errors = []
        
        for statement in statements:
            try:
                await self.graph_db.execute_write_query(statement)
                imported_count += 1
            except Exception as e:
                failed_count += 1
                errors.append(f"语句失败: {statement[:100]}... 错误: {str(e)}")
                logger.warning(f"导入语句失败: {str(e)}")
        
        return {
            "imported_statements": imported_count,
            "failed_statements": failed_count,
            "total_statements": len(statements),
            "errors": errors[:10],  # 只返回前10个错误
            "success_rate": imported_count / len(statements) if statements else 0
        }
    
    async def _import_json_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """导入JSON格式数据"""
        nodes = data.get("nodes", [])
        relationships = data.get("relationships", [])
        
        # 先导入节点
        node_results = await self._import_nodes(nodes)
        
        # 再导入关系
        rel_results = await self._import_relationships(relationships)
        
        return {
            "nodes": node_results,
            "relationships": rel_results,
            "total_imported": node_results["imported"] + rel_results["imported"],
            "total_failed": node_results["failed"] + rel_results["failed"]
        }
    
    async def _import_nodes(self, nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """导入节点"""
        imported = 0
        failed = 0
        
        for node in nodes:
            try:
                labels = ":".join(node.get("labels", []))
                properties = node.get("properties", {})
                
                query = f"CREATE (n:{labels} $props) RETURN id(n) as node_id"
                await self.graph_db.execute_write_query(query, {"props": properties})
                imported += 1
                
            except Exception as e:
                failed += 1
                logger.warning(f"导入节点失败: {str(e)}")
        
        return {"imported": imported, "failed": failed, "total": len(nodes)}
    
    async def _import_relationships(self, relationships: List[Dict[str, Any]]) -> Dict[str, Any]:
        """导入关系"""
        imported = 0
        failed = 0
        
        for rel in relationships:
            try:
                rel_type = rel.get("type", "RELATES_TO")
                properties = rel.get("properties", {})
                start_node = rel.get("start_node")
                end_node = rel.get("end_node")
                
                query = f"""
                MATCH (a), (b) 
                WHERE id(a) = $start_id AND id(b) = $end_id
                CREATE (a)-[r:{rel_type} $props]->(b)
                RETURN id(r) as rel_id
                """
                
                await self.graph_db.execute_write_query(
                    query, 
                    {
                        "start_id": start_node,
                        "end_id": end_node,
                        "props": properties
                    }
                )
                imported += 1
                
            except Exception as e:
                failed += 1
                logger.warning(f"导入关系失败: {str(e)}")
        
        return {"imported": imported, "failed": failed, "total": len(relationships)}
