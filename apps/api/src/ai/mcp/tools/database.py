"""数据库MCP工具实现"""

import asyncio
import re
from typing import Any, Dict, List, Optional, Union
from sqlalchemy import text, inspect
from sqlalchemy.exc import SQLAlchemyError
from src.core.database import get_db_session
from ..client import get_mcp_client_manager
from ..exceptions import MCPConnectionError

logger = get_logger(__name__)

class DatabaseSecurityError(Exception):
    """数据库安全异常"""
    ...

class DatabaseTool:
    """数据库MCP工具实现"""
    
    def __init__(self):
        self.dangerous_keywords = [
            'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT', 'UPDATE',
            'GRANT', 'REVOKE', 'EXEC', 'EXECUTE', 'xp_', 'sp_'
        ]
        self.allowed_read_patterns = [
            r'^SELECT\s+',
            r'^DESCRIBE\s+',
            r'^SHOW\s+',
            r'^EXPLAIN\s+'
        ]
    
    def _validate_query(self, query: str, read_only: bool = True) -> tuple[bool, str]:
        """验证SQL查询安全性
        
        Args:
            query: SQL查询语句
            read_only: 是否只允许读取操作
            
        Returns:
            (is_valid, error_message)
        """
        try:
            # 去除注释和多余空格
            cleaned_query = re.sub(r'/\*.*?\*/', '', query, flags=re.DOTALL)
            cleaned_query = re.sub(r'--.*', '', cleaned_query)
            cleaned_query = ' '.join(cleaned_query.split())
            
            if not cleaned_query.strip():
                return False, "Empty query"
            
            # 只读模式下的安全检查
            if read_only:
                # 检查是否包含危险关键词
                query_upper = cleaned_query.upper()
                for keyword in self.dangerous_keywords:
                    if keyword in query_upper:
                        return False, f"Dangerous keyword '{keyword}' not allowed in read-only mode"
                
                # 检查是否匹配允许的模式
                valid_pattern = False
                for pattern in self.allowed_read_patterns:
                    if re.match(pattern, query_upper):
                        valid_pattern = True
                        break
                
                if not valid_pattern:
                    return False, "Query does not match allowed read-only patterns"
            
            # 检查SQL注入风险
            if self._detect_sql_injection(cleaned_query):
                return False, "Potential SQL injection detected"
            
            return True, "Valid query"
            
        except Exception as e:
            return False, f"Query validation error: {str(e)}"
    
    def _detect_sql_injection(self, query: str) -> bool:
        """检测潜在的SQL注入攻击
        
        Args:
            query: SQL查询语句
            
        Returns:
            是否检测到SQL注入风险
        """
        # 简单的SQL注入检测模式
        injection_patterns = [
            r';\s*(DROP|DELETE|UPDATE|INSERT)',
            r'UNION\s+SELECT',
            r'OR\s+1\s*=\s*1',
            r'AND\s+1\s*=\s*1',
            r"'\s*OR\s*'.*'\s*=\s*'",
            r'--\s*$',
            r'/\*.*\*/',
            r'xp_cmdshell',
            r'sp_executesql'
        ]
        
        query_upper = query.upper()
        for pattern in injection_patterns:
            if re.search(pattern, query_upper):
                return True
        
        return False
    
    async def execute_query(self, query: str, parameters: Optional[List] = None, 
                          read_only: bool = True) -> Dict[str, Any]:
        """执行SQL查询
        
        Args:
            query: SQL查询语句
            parameters: 查询参数
            read_only: 是否只读查询
            
        Returns:
            查询结果字典
        """
        try:
            # 验证查询安全性
            is_valid, error_msg = self._validate_query(query, read_only)
            if not is_valid:
                logger.warning(f"Invalid query rejected: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": "SecurityError"
                }
            
            # 获取数据库会话
            async with get_db_session() as session:
                try:
                    # 准备参数
                    params = parameters or []
                    
                    # 执行查询
                    result = await session.execute(text(query), params)
                    
                    # 处理结果
                    if result.returns_rows:
                        # SELECT查询
                        rows = result.fetchall()
                        columns = list(result.keys())
                        
                        # 转换为字典列表
                        data = []
                        for row in rows:
                            row_dict = {}
                            for i, col in enumerate(columns):
                                value = row[i]
                                # 处理特殊类型
                                if hasattr(value, 'isoformat'):
                                    value = value.isoformat()
                                row_dict[col] = value
                            data.append(row_dict)
                        
                        logger.info(f"Query executed successfully, returned {len(data)} rows")
                        return {
                            "success": True,
                            "data": data,
                            "columns": columns,
                            "row_count": len(data),
                            "query": query
                        }
                    else:
                        # 非SELECT查询（如果允许）
                        if not read_only:
                            affected_rows = result.rowcount
                            await session.commit()
                            
                            logger.info(f"Query executed successfully, affected {affected_rows} rows")
                            return {
                                "success": True,
                                "affected_rows": affected_rows,
                                "query": query
                            }
                        else:
                            return {
                                "success": False,
                                "error": "Non-SELECT query not allowed in read-only mode",
                                "error_type": "SecurityError"
                            }
                            
                except SQLAlchemyError as e:
                    await session.rollback()
                    logger.error(f"Database error executing query: {str(e)}")
                    return {
                        "success": False,
                        "error": f"Database error: {str(e)}",
                        "error_type": "DatabaseError"
                    }
                    
        except RuntimeError as e:
            logger.error(f"Database runtime error: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "DatabaseNotInitialized"
            }
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "UnknownError"
            }
    
    async def describe_tables(self, table_name: Optional[str] = None, 
                            schema: Optional[str] = None) -> Dict[str, Any]:
        """描述数据库表结构
        
        Args:
            table_name: 表名（可选，为空则列出所有表）
            schema: 模式名（可选）
            
        Returns:
            表结构信息字典
        """
        try:
            async with get_db_session() as session:
                inspector = inspect(session.bind)
                
                if table_name:
                    # 描述特定表
                    try:
                        # 检查表是否存在
                        tables = inspector.get_table_names(schema=schema)
                        if table_name not in tables:
                            return {
                                "success": False,
                                "error": f"Table '{table_name}' not found",
                                "error_type": "TableNotFound"
                            }
                        
                        # 获取表结构
                        columns = inspector.get_columns(table_name, schema=schema)
                        primary_keys = inspector.get_pk_constraint(table_name, schema=schema)
                        foreign_keys = inspector.get_foreign_keys(table_name, schema=schema)
                        indexes = inspector.get_indexes(table_name, schema=schema)
                        
                        # 格式化列信息
                        column_info = []
                        for col in columns:
                            col_info = {
                                "name": col["name"],
                                "type": str(col["type"]),
                                "nullable": col["nullable"],
                                "default": col.get("default"),
                                "autoincrement": col.get("autoincrement", False)
                            }
                            column_info.append(col_info)
                        
                        logger.info(f"Successfully described table: {table_name}")
                        return {
                            "success": True,
                            "table_name": table_name,
                            "schema": schema,
                            "columns": column_info,
                            "primary_keys": primary_keys.get("constrained_columns", []),
                            "foreign_keys": foreign_keys,
                            "indexes": indexes
                        }
                        
                    except SQLAlchemyError as e:
                        logger.error(f"Error describing table {table_name}: {str(e)}")
                        return {
                            "success": False,
                            "error": f"Database error: {str(e)}",
                            "error_type": "DatabaseError"
                        }
                else:
                    # 列出所有表
                    try:
                        tables = inspector.get_table_names(schema=schema)
                        views = inspector.get_view_names(schema=schema)
                        
                        table_info = []
                        for table in tables:
                            try:
                                columns = inspector.get_columns(table, schema=schema)
                                table_info.append({
                                    "name": table,
                                    "type": "table",
                                    "column_count": len(columns),
                                    "schema": schema
                                })
                            except Exception as e:
                                logger.warning(f"Error getting info for table {table}: {str(e)}")
                                table_info.append({
                                    "name": table,
                                    "type": "table",
                                    "column_count": None,
                                    "schema": schema,
                                    "error": str(e)
                                })
                        
                        # 添加视图信息
                        for view in views:
                            table_info.append({
                                "name": view,
                                "type": "view",
                                "column_count": None,
                                "schema": schema
                            })
                        
                        logger.info(f"Successfully listed {len(tables)} tables and {len(views)} views")
                        return {
                            "success": True,
                            "schema": schema,
                            "tables": table_info,
                            "total_count": len(table_info)
                        }
                        
                    except SQLAlchemyError as e:
                        logger.error(f"Error listing tables: {str(e)}")
                        return {
                            "success": False,
                            "error": f"Database error: {str(e)}",
                            "error_type": "DatabaseError"
                        }
                        
        except Exception as e:
            logger.error(f"Error describing tables: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "UnknownError"
            }
    
    async def execute_transaction(self, queries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """执行事务（多个查询的原子操作）
        
        Args:
            queries: 查询列表，每个查询包含query和parameters
            
        Returns:
            事务执行结果
        """
        try:
            if not queries:
                return {
                    "success": False,
                    "error": "No queries provided",
                    "error_type": "InvalidInput"
                }
            
            # 验证所有查询
            for i, query_info in enumerate(queries):
                query = query_info.get("query", "")
                if not query:
                    return {
                        "success": False,
                        "error": f"Empty query at index {i}",
                        "error_type": "InvalidInput"
                    }
                
                # 事务中不允许只读模式
                is_valid, error_msg = self._validate_query(query, read_only=False)
                if not is_valid:
                    return {
                        "success": False,
                        "error": f"Query {i} validation failed: {error_msg}",
                        "error_type": "SecurityError"
                    }
            
            results = []
            async with get_db_session() as session:
                try:
                    # 执行所有查询
                    for i, query_info in enumerate(queries):
                        query = query_info["query"]
                        parameters = query_info.get("parameters", [])
                        
                        result = await session.execute(text(query), parameters)
                        
                        if result.returns_rows:
                            rows = result.fetchall()
                            columns = list(result.keys())
                            data = []
                            for row in rows:
                                row_dict = {}
                                for j, col in enumerate(columns):
                                    value = row[j]
                                    if hasattr(value, 'isoformat'):
                                        value = value.isoformat()
                                    row_dict[col] = value
                                data.append(row_dict)
                            
                            results.append({
                                "query_index": i,
                                "type": "select",
                                "data": data,
                                "row_count": len(data)
                            })
                        else:
                            results.append({
                                "query_index": i,
                                "type": "modification",
                                "affected_rows": result.rowcount
                            })
                    
                    # 提交事务
                    await session.commit()
                    
                    logger.info(f"Transaction executed successfully with {len(queries)} queries")
                    return {
                        "success": True,
                        "results": results,
                        "query_count": len(queries)
                    }
                    
                except SQLAlchemyError as e:
                    await session.rollback()
                    logger.error(f"Transaction failed: {str(e)}")
                    return {
                        "success": False,
                        "error": f"Transaction failed: {str(e)}",
                        "error_type": "DatabaseError"
                    }
                    
        except Exception as e:
            logger.error(f"Error executing transaction: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "UnknownError"
            }

# 全局数据库工具实例
database_tool = DatabaseTool()

async def call_database_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """调用数据库工具的统一接口"""
    try:
        if tool_name == "execute_query":
            return await database_tool.execute_query(
                query=arguments["query"],
                parameters=arguments.get("parameters"),
                read_only=arguments.get("read_only", True)
            )
        elif tool_name == "describe_tables":
            return await database_tool.describe_tables(
                table_name=arguments.get("table_name"),
                schema=arguments.get("schema")
            )
        elif tool_name == "execute_transaction":
            return await database_tool.execute_transaction(
                queries=arguments["queries"]
            )
        else:
            return {
                "success": False,
                "error": f"Unknown database tool: {tool_name}",
                "error_type": "UnknownTool"
            }
    except Exception as e:
        logger.error(f"Error calling database tool {tool_name}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "ToolError"
        }
from src.core.logging import get_logger
