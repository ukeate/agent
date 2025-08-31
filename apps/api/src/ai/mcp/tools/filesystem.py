"""文件系统MCP工具实现"""

import asyncio
import aiofiles
import os
import stat
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

from ..client import get_mcp_client_manager
from ..exceptions import MCPSecurityError, MCPResourceError, MCPValidationError, handle_mcp_exception
from ..monitoring import monitor_operation

logger = logging.getLogger(__name__)


class FileSystemTool:
    """文件系统MCP工具实现"""
    
    def __init__(self):
        import tempfile
        self.allowed_paths = [
            "/tmp",
            "/var/tmp",
            tempfile.gettempdir(),  # 系统临时目录
            os.path.expanduser("~/Documents"),
            os.path.expanduser("~/Desktop"),
            # 开发环境允许访问项目目录
            os.getcwd()
        ]
        self.blocked_patterns = [
            "/etc/passwd",
            "/etc/shadow", 
            "/root",
            "/sys",
            "/proc",
            "/.ssh",
            "/home/*/.ssh"
        ]
    
    def _validate_path(self, path: str) -> str:
        """验证和规范化文件路径"""
        try:
            # 规范化路径
            normalized_path = os.path.abspath(os.path.expanduser(path))
            
            # 检查是否在允许的路径内
            path_allowed = False
            for allowed in self.allowed_paths:
                allowed_abs = os.path.abspath(os.path.expanduser(allowed))
                if normalized_path.startswith(allowed_abs):
                    path_allowed = True
                    break
            
            if not path_allowed:
                raise MCPSecurityError(
                    f"Access denied: {path} not in allowed paths",
                    violation_type="path_access",
                    attempted_action=f"access_path:{path}"
                )
            
            # 检查是否匹配阻止的模式
            for pattern in self.blocked_patterns:
                if pattern in normalized_path:
                    raise MCPSecurityError(
                        f"Access denied: {path} matches blocked pattern",
                        violation_type="blocked_pattern",
                        attempted_action=f"access_path:{path}"
                    )
            
            return normalized_path
            
        except Exception as e:
            if isinstance(e, MCPSecurityError):
                raise
            raise MCPValidationError(f"Invalid path: {str(e)}")
    
    @monitor_operation("filesystem_read_file", server_type="filesystem", tool_name="read_file")
    async def read_file(self, path: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """读取文件内容
        
        Args:
            path: 文件路径
            encoding: 文件编码，默认utf-8
            
        Returns:
            包含文件内容的字典
        """
        try:
            validated_path = self._validate_path(path)
            
            if not os.path.exists(validated_path):
                raise MCPResourceError(
                    f"File not found: {path}",
                    resource_type="file",
                    resource_path=path
                )
            
            if not os.path.isfile(validated_path):
                return {
                    "success": False,
                    "error": f"Path is not a file: {path}",
                    "error_type": "NotAFile"
                }
            
            # 检查文件大小（限制为10MB）
            file_size = os.path.getsize(validated_path)
            if file_size > 10 * 1024 * 1024:
                return {
                    "success": False,
                    "error": f"File too large: {file_size} bytes (max 10MB)",
                    "error_type": "FileTooLarge"
                }
            
            async with aiofiles.open(validated_path, mode='r', encoding=encoding) as file:
                content = await file.read()
                
            logger.info(f"Successfully read file: {path}")
            return {
                "success": True,
                "content": content,
                "size": file_size,
                "encoding": encoding,
                "path": validated_path
            }
            
        except MCPSecurityError as e:
            logger.warning(f"Security violation reading file {path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "SecurityError"
            }
        except MCPResourceError as e:
            logger.error(f"Resource error reading file {path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "FileNotFound"
            }
        except FileNotFoundError as e:
            logger.error(f"File not found: {path}")
            return {
                "success": False,
                "error": f"File not found: {path}",
                "error_type": "FileNotFound"
            }
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading file {path}: {str(e)}")
            return {
                "success": False,
                "error": f"Cannot decode file with encoding {encoding}: {str(e)}",
                "error_type": "EncodingError"
            }
        except Exception as e:
            logger.error(f"Error reading file {path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "UnknownError"
            }
    
    async def write_file(self, path: str, content: str, encoding: str = "utf-8") -> Dict[str, Any]:
        """写入文件内容
        
        Args:
            path: 文件路径
            content: 文件内容
            encoding: 文件编码，默认utf-8
            
        Returns:
            操作结果字典
        """
        try:
            validated_path = self._validate_path(path)
            
            # 确保目录存在
            directory = os.path.dirname(validated_path)
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            
            # 检查内容大小（限制为10MB）
            content_size = len(content.encode(encoding))
            if content_size > 10 * 1024 * 1024:
                return {
                    "success": False,
                    "error": f"Content too large: {content_size} bytes (max 10MB)",
                    "error_type": "ContentTooLarge"
                }
            
            async with aiofiles.open(validated_path, mode='w', encoding=encoding) as file:
                await file.write(content)
            
            # 获取写入后的文件信息
            file_info = os.stat(validated_path)
            
            logger.info(f"Successfully wrote file: {path}")
            return {
                "success": True,
                "path": validated_path,
                "size": file_info.st_size,
                "encoding": encoding,
                "modified_time": datetime.fromtimestamp(file_info.st_mtime).isoformat()
            }
            
        except MCPSecurityError as e:
            logger.warning(f"Security violation writing file {path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "SecurityError"
            }
        except Exception as e:
            logger.error(f"Error writing file {path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "UnknownError"
            }
    
    async def list_directory(self, path: str, include_hidden: bool = False) -> Dict[str, Any]:
        """列出目录内容
        
        Args:
            path: 目录路径
            include_hidden: 是否包含隐藏文件
            
        Returns:
            目录内容列表
        """
        try:
            validated_path = self._validate_path(path)
            
            if not os.path.exists(validated_path):
                return {
                    "success": False,
                    "error": f"Directory not found: {path}",
                    "error_type": "DirectoryNotFound"
                }
            
            if not os.path.isdir(validated_path):
                return {
                    "success": False,
                    "error": f"Path is not a directory: {path}",
                    "error_type": "NotADirectory"
                }
            
            entries = []
            for entry in os.listdir(validated_path):
                # 跳过隐藏文件（除非明确要求）
                if not include_hidden and entry.startswith('.'):
                    continue
                
                entry_path = os.path.join(validated_path, entry)
                try:
                    stat_info = os.stat(entry_path)
                    
                    entry_info = {
                        "name": entry,
                        "path": entry_path,
                        "type": "directory" if os.path.isdir(entry_path) else "file",
                        "size": stat_info.st_size,
                        "modified_time": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                        "permissions": oct(stat_info.st_mode)[-3:]
                    }
                    
                    # 添加文件特定信息
                    if os.path.isfile(entry_path):
                        entry_info["extension"] = os.path.splitext(entry)[1]
                    
                    entries.append(entry_info)
                    
                except OSError as e:
                    # 跳过无法访问的文件/目录
                    logger.warning(f"Cannot access {entry_path}: {str(e)}")
                    continue
            
            # 按名称排序
            entries.sort(key=lambda x: x["name"].lower())
            
            logger.info(f"Successfully listed directory: {path}")
            return {
                "success": True,
                "path": validated_path,
                "entries": entries,
                "total_count": len(entries)
            }
            
        except MCPSecurityError as e:
            logger.warning(f"Security violation listing directory {path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "SecurityError"
            }
        except Exception as e:
            logger.error(f"Error listing directory {path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "UnknownError"
            }
    
    async def file_info(self, path: str) -> Dict[str, Any]:
        """获取文件或目录信息
        
        Args:
            path: 文件或目录路径
            
        Returns:
            文件信息字典
        """
        try:
            validated_path = self._validate_path(path)
            
            if not os.path.exists(validated_path):
                return {
                    "success": False,
                    "error": f"Path not found: {path}",
                    "error_type": "PathNotFound"
                }
            
            stat_info = os.stat(validated_path)
            
            info = {
                "success": True,
                "path": validated_path,
                "name": os.path.basename(validated_path),
                "type": "directory" if os.path.isdir(validated_path) else "file",
                "size": stat_info.st_size,
                "created_time": datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                "accessed_time": datetime.fromtimestamp(stat_info.st_atime).isoformat(),
                "permissions": oct(stat_info.st_mode)[-3:],
                "owner_uid": stat_info.st_uid,
                "group_gid": stat_info.st_gid
            }
            
            # 添加文件特定信息
            if os.path.isfile(validated_path):
                info["extension"] = os.path.splitext(validated_path)[1]
                info["is_executable"] = os.access(validated_path, os.X_OK)
            
            # 添加目录特定信息
            if os.path.isdir(validated_path):
                try:
                    entries = os.listdir(validated_path)
                    info["entry_count"] = len(entries)
                except PermissionError:
                    info["entry_count"] = None
            
            logger.info(f"Successfully got file info: {path}")
            return info
            
        except MCPSecurityError as e:
            logger.warning(f"Security violation getting file info {path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "SecurityError"
            }
        except Exception as e:
            logger.error(f"Error getting file info {path}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "UnknownError"
            }


# 全局文件系统工具实例
filesystem_tool = FileSystemTool()


async def call_filesystem_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """调用文件系统工具的统一接口"""
    try:
        # 参数名映射 - 支持多种参数名格式
        def get_path_param(args: Dict[str, Any]) -> str:
            for key in ["path", "file_path", "directory_path", "filepath", "dir_path"]:
                if key in args:
                    return args[key]
            raise KeyError("No path parameter found in arguments")
        
        if tool_name == "read_file":
            return await filesystem_tool.read_file(
                path=get_path_param(arguments),
                encoding=arguments.get("encoding", "utf-8")
            )
        elif tool_name == "write_file":
            return await filesystem_tool.write_file(
                path=get_path_param(arguments),
                content=arguments["content"],
                encoding=arguments.get("encoding", "utf-8")
            )
        elif tool_name == "list_directory":
            return await filesystem_tool.list_directory(
                path=get_path_param(arguments),
                include_hidden=arguments.get("include_hidden", False)
            )
        elif tool_name == "file_info":
            return await filesystem_tool.file_info(
                path=get_path_param(arguments)
            )
        else:
            return {
                "success": False,
                "error": f"Unknown filesystem tool: {tool_name}",
                "error_type": "UnknownTool"
            }
    except Exception as e:
        logger.error(f"Error calling filesystem tool {tool_name}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "ToolError"
        }