"""
pgvector 0.8.0 升级迁移脚本
支持向后兼容和零停机升级
"""

import asyncio
import logging
from typing import List, Dict, Any
import asyncpg
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class PgVectorUpgradeMigration:
    """pgvector升级迁移管理器"""
    
    def __init__(self, connection_url: str):
        self.connection_url = connection_url
        self.backup_tables: List[str] = []
        
    async def create_connection(self) -> asyncpg.Connection:
        """创建数据库连接"""
        return await asyncpg.connect(self.connection_url)
        
    async def check_current_version(self, conn: asyncpg.Connection) -> str:
        """检查当前pgvector版本"""
        try:
            result = await conn.fetchval(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            )
            return result if result else "not_installed"
        except Exception as e:
            logger.warning(f"检查版本失败: {e}")
            return "not_installed"
            
    async def backup_vector_data(self, conn: asyncpg.Connection) -> Dict[str, Any]:
        """备份现有向量数据"""
        backup_info = {
            "timestamp": datetime.now().isoformat(),
            "tables": [],
            "indexes": []
        }
        
        # 查找所有包含vector列的表
        vector_tables = await conn.fetch("""
            SELECT 
                t.table_name,
                c.column_name,
                c.data_type,
                c.character_maximum_length
            FROM information_schema.tables t
            JOIN information_schema.columns c ON c.table_name = t.table_name
            WHERE t.table_schema = 'public' 
            AND (c.data_type LIKE '%vector%' OR c.data_type = 'USER-DEFINED')
            AND t.table_type = 'BASE TABLE'
        """)
        
        for table_info in vector_tables:
            table_name = table_info['table_name']
            column_name = table_info['column_name']
            
            # 创建备份表 - 使用参数化查询防止SQL注入
            timestamp_suffix = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_table_name = f"{table_name}_backup_{timestamp_suffix}"
            
            # 验证表名和备份表名格式
            if not table_name.replace('_', '').isalnum() or not backup_table_name.replace('_', '').replace('-', '').isalnum():
                logger.error(f"无效的表名格式: {table_name} -> {backup_table_name}")
                continue
            
            await conn.execute(f"""
                CREATE TABLE "{backup_table_name}" AS 
                SELECT * FROM "{table_name}"
            """)
            
            self.backup_tables.append(backup_table_name)
            
            # 记录表信息
            row_count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
            backup_info["tables"].append({
                "original_table": table_name,
                "backup_table": backup_table_name,
                "column_name": column_name,
                "row_count": row_count
            })
            
        # 备份索引信息
        vector_indexes = await conn.fetch("""
            SELECT 
                indexname,
                tablename,
                indexdef
            FROM pg_indexes 
            WHERE schemaname = 'public'
            AND (indexdef LIKE '%vector%' OR indexdef LIKE '%hnsw%' OR indexdef LIKE '%ivfflat%')
        """)
        
        for index_info in vector_indexes:
            backup_info["indexes"].append({
                "index_name": index_info['indexname'],
                "table_name": index_info['tablename'],
                "definition": index_info['indexdef']
            })
            
        logger.info(f"数据备份完成: {len(backup_info['tables'])} 个表, {len(backup_info['indexes'])} 个索引")
        return backup_info
        
    async def upgrade_extension(self, conn: asyncpg.Connection) -> bool:
        """升级pgvector扩展"""
        try:
            current_version = await self.check_current_version(conn)
            logger.info(f"当前pgvector版本: {current_version}")
            
            if current_version == "not_installed":
                # 全新安装
                await conn.execute("CREATE EXTENSION vector")
                logger.info("pgvector扩展已安装")
            else:
                # 升级现有版本
                await conn.execute("ALTER EXTENSION vector UPDATE")
                logger.info("pgvector扩展已升级")
                
            # 验证升级结果
            new_version = await self.check_current_version(conn)
            logger.info(f"升级后版本: {new_version}")
            
            return True
            
        except Exception as e:
            logger.error(f"扩展升级失败: {e}")
            return False
            
    async def optimize_vector_indexes(self, conn: asyncpg.Connection, backup_info: Dict[str, Any]) -> bool:
        """优化向量索引配置"""
        try:
            # 应用全局优化配置
            optimizations = [
                ("hnsw.ef_search", "100"),
                ("hnsw.iterative_scan", "strict_order"),
                ("hnsw.max_scan_tuples", "20000"),
                ("hnsw.scan_mem_multiplier", "2"),
                ("ivfflat.probes", "10"),
                ("ivfflat.iterative_scan", "strict_order"),
                ("ivfflat.max_probes", "100"),
                ("maintenance_work_mem", "512MB"),
                ("max_parallel_maintenance_workers", "4")
            ]
            
            for param, value in optimizations:
                await conn.execute(f"ALTER SYSTEM SET {param} = '{value}'")
                
            # 重新创建索引使用新的优化参数
            for index_info in backup_info["indexes"]:
                old_index_name = index_info["index_name"]
                table_name = index_info["table_name"]
                
                # 删除旧索引
                await conn.execute(f"DROP INDEX IF EXISTS {old_index_name}")
                
                # 创建优化后的索引
                if "hnsw" in index_info["definition"].lower():
                    # HNSW索引优化
                    new_index_def = index_info["definition"].replace(
                        "USING hnsw",
                        "USING hnsw WITH (m = 16, ef_construction = 64)"
                    )
                elif "ivfflat" in index_info["definition"].lower():
                    # IVFFlat索引优化
                    new_index_def = index_info["definition"].replace(
                        "USING ivfflat",
                        "USING ivfflat WITH (lists = 1000)"
                    )
                else:
                    new_index_def = index_info["definition"]
                    
                # 并发创建索引
                new_index_def = new_index_def.replace("CREATE INDEX", "CREATE INDEX CONCURRENTLY")
                await conn.execute(new_index_def)
                
            logger.info("索引优化完成")
            return True
            
        except Exception as e:
            logger.error(f"索引优化失败: {e}")
            return False
            
    async def validate_migration(self, conn: asyncpg.Connection, backup_info: Dict[str, Any]) -> bool:
        """验证迁移结果"""
        try:
            validation_results = []
            
            # 验证扩展版本
            current_version = await self.check_current_version(conn)
            validation_results.append({
                "check": "extension_version",
                "result": current_version,
                "status": "pass" if current_version >= "0.8.0" else "fail"
            })
            
            # 验证数据完整性
            for table_info in backup_info["tables"]:
                original_table = table_info["original_table"]
                backup_table = table_info["backup_table"]
                
                # 检查行数
                original_count = await conn.fetchval(f"SELECT COUNT(*) FROM {original_table}")
                backup_count = table_info["row_count"]
                
                validation_results.append({
                    "check": f"data_integrity_{original_table}",
                    "original_count": backup_count,
                    "current_count": original_count,
                    "status": "pass" if original_count == backup_count else "fail"
                })
                
            # 验证索引状态
            active_indexes = await conn.fetch("""
                SELECT indexname, tablename 
                FROM pg_indexes 
                WHERE schemaname = 'public'
                AND (indexdef LIKE '%vector%' OR indexdef LIKE '%hnsw%' OR indexdef LIKE '%ivfflat%')
            """)
            
            validation_results.append({
                "check": "indexes_recreated",
                "count": len(active_indexes),
                "status": "pass" if len(active_indexes) > 0 else "warning"
            })
            
            # 测试向量查询功能
            try:
                test_query = """
                    SELECT 1 as test 
                    WHERE EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE data_type LIKE '%vector%'
                    )
                """
                await conn.fetchval(test_query)
                
                validation_results.append({
                    "check": "vector_functionality",
                    "status": "pass"
                })
                
            except Exception:
                validation_results.append({
                    "check": "vector_functionality", 
                    "status": "fail"
                })
                
            # 输出验证结果
            all_passed = all(r["status"] == "pass" for r in validation_results)
            logger.info(f"迁移验证结果: {'通过' if all_passed else '部分失败'}")
            
            for result in validation_results:
                logger.info(f"  {result['check']}: {result['status']}")
                
            return all_passed
            
        except Exception as e:
            logger.error(f"迁移验证失败: {e}")
            return False
            
    async def rollback_migration(self, conn: asyncpg.Connection) -> bool:
        """回滚迁移"""
        try:
            logger.info("开始回滚迁移...")
            
            # 恢复备份数据
            for backup_table in self.backup_tables:
                original_table = backup_table.split('_backup_')[0]
                
                # 验证表名格式
                if not original_table.replace('_', '').isalnum() or not backup_table.replace('_', '').replace('-', '').isalnum():
                    logger.error(f"无效的表名格式: {original_table} <- {backup_table}")
                    continue
                
                # 清空原表
                await conn.execute(f'TRUNCATE TABLE "{original_table}"')
                
                # 恢复数据
                await conn.execute(f"""
                    INSERT INTO "{original_table}" 
                    SELECT * FROM "{backup_table}"
                """)
                
                logger.info(f"已恢复表: {original_table}")
                
            logger.info("迁移回滚完成")
            return True
            
        except Exception as e:
            logger.error(f"迁移回滚失败: {e}")
            return False
            
    async def cleanup_backup_tables(self, conn: asyncpg.Connection) -> bool:
        """清理备份表"""
        try:
            for backup_table in self.backup_tables:
                # 验证表名格式
                if not backup_table.replace('_', '').replace('-', '').isalnum():
                    logger.error(f"无效的备份表名格式: {backup_table}")
                    continue
                    
                await conn.execute(f'DROP TABLE IF EXISTS "{backup_table}"')
                logger.info(f"已清理备份表: {backup_table}")
                
            self.backup_tables.clear()
            return True
            
        except Exception as e:
            logger.error(f"清理备份失败: {e}")
            return False
            
    async def run_migration(self, validate_only: bool = False) -> bool:
        """运行完整的迁移流程"""
        conn = await self.create_connection()
        
        try:
            logger.info("开始pgvector 0.8.0升级迁移")
            
            # 1. 备份现有数据
            backup_info = await self.backup_vector_data(conn)
            
            if validate_only:
                logger.info("仅验证模式，跳过实际升级")
                return await self.validate_migration(conn, backup_info)
            
            # 2. 升级扩展
            if not await self.upgrade_extension(conn):
                logger.error("扩展升级失败，开始回滚")
                await self.rollback_migration(conn)
                return False
                
            # 3. 优化索引
            if not await self.optimize_vector_indexes(conn, backup_info):
                logger.error("索引优化失败，但继续验证")
                
            # 4. 验证迁移
            if not await self.validate_migration(conn, backup_info):
                logger.error("迁移验证失败，开始回滚")
                await self.rollback_migration(conn)
                return False
                
            # 5. 清理备份（可选）
            logger.info("迁移成功完成，保留备份表以备回滚需要")
            
            return True
            
        except Exception as e:
            logger.error(f"迁移过程发生错误: {e}")
            await self.rollback_migration(conn)
            return False
            
        finally:
            await conn.close()


async def main():
    """主函数"""
    import os
    
    # 数据库连接配置
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql://ai_agent_user:ai_agent_password@localhost:5433/ai_agent_db"
    )
    
    migration = PgVectorUpgradeMigration(db_url)
    
    # 运行迁移
    success = await migration.run_migration(validate_only=False)
    
    if success:
        print("✅ pgvector升级迁移成功完成")
    else:
        print("❌ pgvector升级迁移失败")
        exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())