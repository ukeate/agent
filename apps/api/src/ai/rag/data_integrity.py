"""
向量数据完整性验证模块

验证现有向量数据的完整性和质量
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class VectorDataIntegrityValidator:
    """向量数据完整性验证器"""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
        self.validation_stats = {
            "total_validations": 0,
            "integrity_checks": 0,
            "repair_operations": 0
        }
    
    async def validate_vector_data_integrity(
        self,
        table_name: str = "knowledge_items",
        vector_column: str = "embedding",
        batch_size: int = 1000
    ) -> Dict[str, Any]:
        """验证向量数据完整性"""
        logger.info(f"Starting vector data integrity validation for {table_name}.{vector_column}")
        
        integrity_report = {
            "table_name": table_name,
            "vector_column": vector_column,
            "total_records": 0,
            "valid_vectors": 0,
            "null_vectors": 0,
            "invalid_vectors": 0,
            "dimension_mismatches": 0,
            "zero_vectors": 0,
            "integrity_rate": 0.0,
            "issues": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        try:
            # 获取总记录数
            count_sql = f"SELECT COUNT(*) as total FROM {table_name}"
            result = await self.db.execute(text(count_sql))
            total_count = result.fetchone().total
            integrity_report["total_records"] = total_count
            
            if total_count == 0:
                logger.warning(f"No records found in {table_name}")
                return integrity_report
            
            # 分批处理数据
            offset = 0
            expected_dimension = None
            
            while offset < total_count:
                batch_sql = f"""
                SELECT id, {vector_column} 
                FROM {table_name} 
                ORDER BY id 
                LIMIT {batch_size} OFFSET {offset}
                """
                
                result = await self.db.execute(text(batch_sql))
                batch_records = result.fetchall()
                
                for record in batch_records:
                    vector_status = await self._validate_single_vector(
                        record.id, 
                        record.__getattribute__(vector_column),
                        expected_dimension
                    )
                    
                    # 更新统计
                    if vector_status["status"] == "valid":
                        integrity_report["valid_vectors"] += 1
                        if expected_dimension is None:
                            expected_dimension = vector_status["dimension"]
                    elif vector_status["status"] == "null":
                        integrity_report["null_vectors"] += 1
                    elif vector_status["status"] == "invalid":
                        integrity_report["invalid_vectors"] += 1
                    elif vector_status["status"] == "dimension_mismatch":
                        integrity_report["dimension_mismatches"] += 1
                    elif vector_status["status"] == "zero_vector":
                        integrity_report["zero_vectors"] += 1
                    
                    # 记录问题
                    if vector_status["status"] != "valid":
                        integrity_report["issues"].append({
                            "record_id": str(record.id),
                            "issue": vector_status["status"],
                            "details": vector_status.get("details", "")
                        })
                
                offset += batch_size
                
                # 进度日志
                if offset % 10000 == 0:
                    logger.info(f"Processed {offset}/{total_count} records")
            
            # 计算完整性率
            if total_count > 0:
                integrity_report["integrity_rate"] = integrity_report["valid_vectors"] / total_count
            
            self.validation_stats["total_validations"] += 1
            self.validation_stats["integrity_checks"] += total_count
            
            logger.info(f"Validation completed: {integrity_report['integrity_rate']:.2%} integrity rate")
            
        except Exception as e:
            logger.error(f"Vector data integrity validation failed: {e}")
            integrity_report["error"] = str(e)
        
        return integrity_report
    
    async def _validate_single_vector(
        self,
        record_id: Any,
        vector_data: Any,
        expected_dimension: Optional[int] = None
    ) -> Dict[str, Any]:
        """验证单个向量"""
        if vector_data is None:
            return {"status": "null", "dimension": 0}
        
        try:
            # 尝试转换为numpy数组
            if isinstance(vector_data, str):
                # 处理文本格式的向量
                vector = np.fromstring(vector_data.strip('[]'), sep=',')
            elif isinstance(vector_data, list):
                vector = np.array(vector_data)
            elif isinstance(vector_data, np.ndarray):
                vector = vector_data
            else:
                return {"status": "invalid", "details": f"Unsupported vector type: {type(vector_data)}"}
            
            # 检查维度
            if expected_dimension is not None and vector.shape[0] != expected_dimension:
                return {
                    "status": "dimension_mismatch",
                    "dimension": vector.shape[0],
                    "details": f"Expected {expected_dimension}, got {vector.shape[0]}"
                }
            
            # 检查是否为零向量
            if np.allclose(vector, 0):
                return {"status": "zero_vector", "dimension": vector.shape[0]}
            
            # 检查是否包含无效值
            if np.any(np.isnan(vector)) or np.any(np.isinf(vector)):
                return {"status": "invalid", "details": "Contains NaN or Inf values"}
            
            return {"status": "valid", "dimension": vector.shape[0]}
            
        except Exception as e:
            return {"status": "invalid", "details": f"Parsing error: {str(e)}"}
    
    async def repair_vector_data(
        self,
        integrity_report: Dict[str, Any],
        repair_strategy: str = "remove_invalid"
    ) -> Dict[str, Any]:
        """修复向量数据"""
        logger.info(f"Starting vector data repair with strategy: {repair_strategy}")
        
        repair_report = {
            "strategy": repair_strategy,
            "processed_issues": 0,
            "successful_repairs": 0,
            "failed_repairs": 0,
            "removed_records": 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if not integrity_report.get("issues"):
            logger.info("No issues to repair")
            return repair_report
        
        try:
            for issue in integrity_report["issues"]:
                record_id = issue["record_id"]
                issue_type = issue["issue"]
                
                repair_report["processed_issues"] += 1
                
                try:
                    if repair_strategy == "remove_invalid":
                        if issue_type in ["invalid", "null", "zero_vector"]:
                            # 删除有问题的记录
                            delete_sql = f"""
                            DELETE FROM {integrity_report['table_name']} 
                            WHERE id = :record_id
                            """
                            await self.db.execute(text(delete_sql), {"record_id": record_id})
                            repair_report["removed_records"] += 1
                            repair_report["successful_repairs"] += 1
                    
                    elif repair_strategy == "set_null":
                        # 将有问题的向量设为NULL
                        update_sql = f"""
                        UPDATE {integrity_report['table_name']} 
                        SET {integrity_report['vector_column']} = NULL
                        WHERE id = :record_id
                        """
                        await self.db.execute(text(update_sql), {"record_id": record_id})
                        repair_report["successful_repairs"] += 1
                    
                    else:
                        logger.warning(f"Unknown repair strategy: {repair_strategy}")
                        repair_report["failed_repairs"] += 1
                
                except Exception as e:
                    logger.error(f"Failed to repair record {record_id}: {e}")
                    repair_report["failed_repairs"] += 1
            
            # 提交修复
            await self.db.commit()
            self.validation_stats["repair_operations"] += repair_report["successful_repairs"]
            
            logger.info(f"Repair completed: {repair_report['successful_repairs']} successful, "
                       f"{repair_report['failed_repairs']} failed")
        
        except Exception as e:
            logger.error(f"Vector data repair failed: {e}")
            await self.db.rollback()
            repair_report["error"] = str(e)
        
        return repair_report
    
    async def generate_integrity_summary(
        self,
        table_name: str = "knowledge_items"
    ) -> Dict[str, Any]:
        """生成完整性摘要报告"""
        logger.info(f"Generating integrity summary for {table_name}")
        
        try:
            # 基本统计
            stats_sql = f"""
            SELECT 
                COUNT(*) as total_records,
                COUNT(embedding) as non_null_embeddings,
                COUNT(*) - COUNT(embedding) as null_embeddings
            FROM {table_name}
            """
            
            result = await self.db.execute(text(stats_sql))
            stats = result.fetchone()
            
            # 索引统计
            index_sql = f"""
            SELECT 
                indexname,
                tablename,
                indexdef
            FROM pg_indexes 
            WHERE tablename = '{table_name}' 
            AND indexname LIKE '%embedding%'
            """
            
            result = await self.db.execute(text(index_sql))
            indexes = result.fetchall()
            
            summary = {
                "table_name": table_name,
                "statistics": {
                    "total_records": stats.total_records,
                    "non_null_embeddings": stats.non_null_embeddings,
                    "null_embeddings": stats.null_embeddings,
                    "null_rate": stats.null_embeddings / stats.total_records if stats.total_records > 0 else 0
                },
                "indexes": [
                    {
                        "name": idx.indexname,
                        "definition": idx.indexdef
                    }
                    for idx in indexes
                ],
                "validation_stats": self.validation_stats.copy(),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate integrity summary: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}
    
    async def validate_quantization_integrity(
        self,
        quantization_table: str = "vector_quantization_params"
    ) -> Dict[str, Any]:
        """验证量化参数的完整性"""
        logger.info(f"Validating quantization integrity in {quantization_table}")
        
        try:
            # 检查量化参数表是否存在
            table_exists_sql = f"""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = '{quantization_table}'
            )
            """
            
            result = await self.db.execute(text(table_exists_sql))
            table_exists = result.fetchone()[0]
            
            if not table_exists:
                return {
                    "table_exists": False,
                    "message": f"Quantization table {quantization_table} does not exist",
                    "timestamp": datetime.utcnow().isoformat()
                }
            
            # 检查量化参数的完整性
            integrity_sql = f"""
            SELECT 
                quantization_mode,
                COUNT(*) as count,
                AVG(compression_ratio) as avg_compression,
                AVG(precision_loss) as avg_precision_loss
            FROM {quantization_table}
            GROUP BY quantization_mode
            """
            
            result = await self.db.execute(text(integrity_sql))
            quantization_stats = result.fetchall()
            
            # 检查孤立的量化参数
            orphaned_params_sql = f"""
            SELECT COUNT(*) as orphaned_count
            FROM {quantization_table} qp
            LEFT JOIN knowledge_items ki ON qp.id = ki.quantization_params_id
            WHERE ki.id IS NULL
            """
            
            result = await self.db.execute(text(orphaned_params_sql))
            orphaned_count = result.fetchone().orphaned_count
            
            integrity_report = {
                "table_exists": True,
                "quantization_modes": [
                    {
                        "mode": stat.quantization_mode,
                        "count": stat.count,
                        "avg_compression": float(stat.avg_compression),
                        "avg_precision_loss": float(stat.avg_precision_loss)
                    }
                    for stat in quantization_stats
                ],
                "orphaned_params": orphaned_count,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return integrity_report
            
        except Exception as e:
            logger.error(f"Quantization integrity validation failed: {e}")
            return {"error": str(e), "timestamp": datetime.utcnow().isoformat()}