"""系统命令MCP工具实现"""

import asyncio
import os
import signal
import psutil
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from ..client import get_mcp_client_manager, MCPConnectionError

logger = logging.getLogger(__name__)


class SystemSecurityError(Exception):
    """系统安全异常"""
    pass


class SystemTool:
    """系统命令MCP工具实现"""
    
    def __init__(self):
        # 允许的安全命令白名单
        self.allowed_commands = [
            'ls', 'dir', 'pwd', 'echo', 'cat', 'head', 'tail', 'grep', 'find',
            'ps', 'top', 'htop', 'free', 'df', 'du', 'who', 'whoami', 'id',
            'date', 'uptime', 'uname', 'hostname', 'which', 'whereis',
            'git', 'python', 'python3', 'pip', 'pip3', 'node', 'npm', 'yarn',
            'docker', 'docker-compose', 'curl', 'wget', 'ping', 'nslookup',
            'mkdir', 'touch', 'cp', 'mv', 'ln', 'chmod', 'chown'
        ]
        
        # 危险命令黑名单
        self.blocked_commands = [
            'rm', 'rmdir', 'del', 'format', 'fdisk', 'mkfs', 'dd',
            'sudo', 'su', 'passwd', 'useradd', 'userdel', 'usermod',
            'shutdown', 'reboot', 'halt', 'poweroff', 'init',
            'kill', 'killall', 'pkill', 'killall9',
            'crontab', 'at', 'batch',
            'mount', 'umount', 'fsck',
            'iptables', 'ufw', 'firewall-cmd',
            'nc', 'netcat', 'telnet', 'ssh', 'scp', 'rsync'
        ]
        
        # 允许的环境变量前缀
        self.allowed_env_prefixes = [
            'PATH', 'HOME', 'USER', 'SHELL', 'TERM', 'LANG', 'LC_',
            'PWD', 'OLDPWD', 'TMPDIR', 'TMP', 'TEMP',
            'PYTHON', 'NODE', 'NPM', 'DOCKER', 'GIT',
            'DATABASE_URL', 'REDIS_URL', 'API_KEY', 'SECRET'
        ]
    
    def _validate_command(self, command: str) -> tuple[bool, str]:
        """验证命令安全性
        
        Args:
            command: 要执行的命令
            
        Returns:
            (is_valid, error_message)
        """
        try:
            # 提取命令的第一个词（实际的可执行文件）
            cmd_parts = command.strip().split()
            if not cmd_parts:
                return False, "Empty command"
            
            base_command = cmd_parts[0].split('/')[-1]  # 去除路径，只保留命令名
            
            # 检查是否在黑名单中
            if base_command in self.blocked_commands:
                return False, f"Command '{base_command}' is blocked for security reasons"
            
            # 检查是否在白名单中
            if base_command not in self.allowed_commands:
                return False, f"Command '{base_command}' is not in allowed list"
            
            # 检查危险的参数组合
            command_lower = command.lower()
            dangerous_patterns = [
                '&& rm', '&& del', '|| rm', '|| del',
                '; rm', '; del', '| rm', '| del',
                '--force', '--recursive', '-rf', '-r ',
                '> /dev/', '< /dev/', '2>/dev/',
                '/etc/', '/sys/', '/proc/', '/root',
                '$(' , '`', '${', 'eval ', 'exec '
            ]
            
            for pattern in dangerous_patterns:
                if pattern in command_lower:
                    return False, f"Dangerous pattern '{pattern}' detected in command"
            
            return True, "Command is safe"
            
        except Exception as e:
            return False, f"Command validation error: {str(e)}"
    
    def _validate_env_var(self, var_name: str) -> bool:
        """验证环境变量访问权限
        
        Args:
            var_name: 环境变量名
            
        Returns:
            是否允许访问
        """
        # 检查是否以允许的前缀开头
        for prefix in self.allowed_env_prefixes:
            if var_name.startswith(prefix):
                return True
        
        # 特殊情况：允许查看当前用户的基本环境变量
        safe_vars = ['USER', 'HOME', 'SHELL', 'PWD', 'TERM']
        if var_name in safe_vars:
            return True
        
        return False
    
    async def run_command(self, command: str, working_dir: Optional[str] = None, 
                         timeout: int = 30) -> Dict[str, Any]:
        """执行系统命令
        
        Args:
            command: 要执行的命令
            working_dir: 工作目录
            timeout: 超时时间（秒）
            
        Returns:
            命令执行结果
        """
        try:
            # 验证命令安全性
            is_valid, error_msg = self._validate_command(command)
            if not is_valid:
                logger.warning(f"Unsafe command rejected: {error_msg}")
                return {
                    "success": False,
                    "error": error_msg,
                    "error_type": "SecurityError"
                }
            
            # 验证工作目录
            if working_dir:
                if not os.path.exists(working_dir):
                    return {
                        "success": False,
                        "error": f"Working directory does not exist: {working_dir}",
                        "error_type": "DirectoryNotFound"
                    }
                if not os.path.isdir(working_dir):
                    return {
                        "success": False,
                        "error": f"Working directory is not a directory: {working_dir}",
                        "error_type": "NotADirectory"
                    }
            
            # 限制超时时间
            timeout = min(timeout, 60)  # 最大60秒
            
            start_time = datetime.now()
            
            try:
                # 执行命令
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=working_dir,
                    limit=1024 * 1024  # 1MB输出限制
                )
                
                # 等待命令完成或超时
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                
                end_time = datetime.now()
                execution_time = (end_time - start_time).total_seconds()
                
                # 解码输出
                stdout_str = stdout.decode('utf-8', errors='replace')
                stderr_str = stderr.decode('utf-8', errors='replace')
                
                logger.info(f"Command executed successfully: {command}")
                return {
                    "success": True,
                    "command": command,
                    "return_code": process.returncode,
                    "stdout": stdout_str,
                    "stderr": stderr_str,
                    "execution_time": execution_time,
                    "working_dir": working_dir or os.getcwd()
                }
                
            except asyncio.TimeoutError:
                # 杀死超时的进程
                try:
                    process.kill()
                    await process.wait()
                except:
                    pass
                
                logger.warning(f"Command timeout: {command}")
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout} seconds",
                    "error_type": "TimeoutError",
                    "command": command
                }
                
            except Exception as e:
                logger.error(f"Error executing command {command}: {str(e)}")
                return {
                    "success": False,
                    "error": f"Execution error: {str(e)}",
                    "error_type": "ExecutionError",
                    "command": command
                }
                
        except Exception as e:
            logger.error(f"Error in run_command: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "UnknownError"
            }
    
    async def check_process(self, process_name: Optional[str] = None, 
                          pid: Optional[int] = None) -> Dict[str, Any]:
        """检查进程状态
        
        Args:
            process_name: 进程名称
            pid: 进程ID
            
        Returns:
            进程状态信息
        """
        try:
            if not process_name and not pid:
                return {
                    "success": False,
                    "error": "Either process_name or pid must be provided",
                    "error_type": "InvalidInput"
                }
            
            processes = []
            
            if pid:
                # 检查特定PID
                try:
                    proc = psutil.Process(pid)
                    if proc.is_running():
                        process_info = {
                            "pid": proc.pid,
                            "name": proc.name(),
                            "status": proc.status(),
                            "cpu_percent": proc.cpu_percent(),
                            "memory_percent": proc.memory_percent(),
                            "create_time": datetime.fromtimestamp(proc.create_time()).isoformat(),
                            "cmdline": proc.cmdline()[:5]  # 只显示前5个参数
                        }
                        processes.append(process_info)
                except psutil.NoSuchProcess:
                    return {
                        "success": False,
                        "error": f"Process with PID {pid} not found",
                        "error_type": "ProcessNotFound"
                    }
                except psutil.AccessDenied:
                    return {
                        "success": False,
                        "error": f"Access denied to process {pid}",
                        "error_type": "AccessDenied"
                    }
            
            if process_name:
                # 按名称搜索进程
                found_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent', 'create_time']):
                    try:
                        if process_name.lower() in proc.info['name'].lower():
                            process_info = {
                                "pid": proc.info['pid'],
                                "name": proc.info['name'],
                                "status": proc.info['status'],
                                "cpu_percent": proc.info['cpu_percent'] or 0.0,
                                "memory_percent": proc.info['memory_percent'] or 0.0,
                                "create_time": datetime.fromtimestamp(proc.info['create_time']).isoformat()
                            }
                            found_processes.append(process_info)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                processes.extend(found_processes)
            
            logger.info(f"Process check completed, found {len(processes)} processes")
            return {
                "success": True,
                "processes": processes,
                "count": len(processes),
                "search_criteria": {
                    "process_name": process_name,
                    "pid": pid
                }
            }
            
        except Exception as e:
            logger.error(f"Error checking process: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "UnknownError"
            }
    
    async def get_env(self, var_name: str) -> Dict[str, Any]:
        """获取环境变量
        
        Args:
            var_name: 环境变量名称
            
        Returns:
            环境变量信息
        """
        try:
            # 验证环境变量访问权限
            if not self._validate_env_var(var_name):
                logger.warning(f"Access denied to environment variable: {var_name}")
                return {
                    "success": False,
                    "error": f"Access denied to environment variable '{var_name}'",
                    "error_type": "SecurityError"
                }
            
            # 获取环境变量值
            value = os.environ.get(var_name)
            
            if value is None:
                return {
                    "success": False,
                    "error": f"Environment variable '{var_name}' not found",
                    "error_type": "VariableNotFound"
                }
            
            # 对敏感信息进行脱敏处理
            if any(sensitive in var_name.upper() for sensitive in ['PASSWORD', 'SECRET', 'KEY', 'TOKEN']):
                if len(value) > 8:
                    masked_value = value[:4] + '*' * (len(value) - 8) + value[-4:]
                else:
                    masked_value = '*' * len(value)
            else:
                masked_value = value
            
            logger.info(f"Successfully retrieved environment variable: {var_name}")
            return {
                "success": True,
                "variable_name": var_name,
                "value": masked_value,
                "is_masked": 'PASSWORD' in var_name.upper() or 'SECRET' in var_name.upper() or 'KEY' in var_name.upper()
            }
            
        except Exception as e:
            logger.error(f"Error getting environment variable {var_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "UnknownError"
            }
    
    async def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        try:
            # 获取基础系统信息
            info = {
                "success": True,
                "system": {
                    "platform": os.name,
                    "hostname": os.uname().nodename,
                    "current_user": os.environ.get('USER', 'unknown'),
                    "current_directory": os.getcwd(),
                    "python_version": os.sys.version.split()[0]
                },
                "resources": {
                    "cpu_count": psutil.cpu_count(),
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory": {
                        "total": psutil.virtual_memory().total,
                        "available": psutil.virtual_memory().available,
                        "percent": psutil.virtual_memory().percent
                    },
                    "disk": {
                        "total": psutil.disk_usage('/').total,
                        "free": psutil.disk_usage('/').free,
                        "percent": psutil.disk_usage('/').percent
                    }
                },
                "network": {
                    "interfaces": list(psutil.net_if_addrs().keys())
                }
            }
            
            logger.info("Successfully retrieved system information")
            return info
            
        except Exception as e:
            logger.error(f"Error getting system info: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "error_type": "UnknownError"
            }


# 全局系统工具实例
system_tool = SystemTool()


async def call_system_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """调用系统工具的统一接口"""
    try:
        if tool_name == "run_command":
            return await system_tool.run_command(
                command=arguments["command"],
                working_dir=arguments.get("working_dir"),
                timeout=arguments.get("timeout", 30)
            )
        elif tool_name == "check_process":
            return await system_tool.check_process(
                process_name=arguments.get("process_name"),
                pid=arguments.get("pid")
            )
        elif tool_name == "get_env":
            return await system_tool.get_env(
                var_name=arguments["var_name"]
            )
        elif tool_name == "get_system_info":
            return await system_tool.get_system_info()
        else:
            return {
                "success": False,
                "error": f"Unknown system tool: {tool_name}",
                "error_type": "UnknownTool"
            }
    except Exception as e:
        logger.error(f"Error calling system tool {tool_name}: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "error_type": "ToolError"
        }