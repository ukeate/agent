"""
任务执行器服务
负责执行supervisor分配的pending任务
"""

import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db_session
from src.repositories.supervisor_repository import (
    SupervisorTaskRepository, SupervisorRepository, 
    SupervisorDecisionRepository
)
from src.models.schemas.supervisor import TaskStatus
from src.ai.autogen.agents import BaseAutoGenAgent, create_default_agents

from src.core.logging import get_logger
logger = get_logger(__name__)

class TaskExecutor:
    """任务执行器"""
    
    def __init__(self):
        self._agents: Dict[str, BaseAutoGenAgent] = {}
        self._running_tasks: Dict[str, str] = {}  # task_id -> agent_name
        self._cached_supervisors = None  # 缓存活跃的supervisors
        self._supervisors_cache_time = None  # 缓存时间
        self._cache_ttl = 300  # 缓存5分钟
        self._initialize_agents()
    
    def _initialize_agents(self):
        """初始化智能体池"""
        try:
            agents_list = create_default_agents()
            for agent in agents_list:
                self._agents[agent.config.name] = agent
                logger.info("智能体已加载", agent_name=agent.config.name)
            
            logger.info("任务执行器智能体池初始化完成", total_agents=len(self._agents))
        except Exception as e:
            logger.error("智能体池初始化失败", error=str(e))
    
    async def _get_cached_active_supervisors(self):
        """获取缓存的活跃supervisors，减少数据库查询"""
        now = utc_now()
        
        # 检查缓存是否过期
        if (self._cached_supervisors is None or 
            self._supervisors_cache_time is None or 
            (now - self._supervisors_cache_time).total_seconds() > self._cache_ttl):
            
            # 缓存过期，重新获取
            try:
                async with get_db_session() as db:
                    supervisor_repo = SupervisorRepository(db)
                    self._cached_supervisors = await supervisor_repo.get_active_supervisors()
                    self._supervisors_cache_time = now
                    
                    logger.debug("已更新活跃supervisors缓存", 
                               supervisor_count=len(self._cached_supervisors))
            except Exception as e:
                logger.error("获取活跃supervisors失败", error=str(e))
                return []
        
        return self._cached_supervisors or []
    
    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """执行单个任务"""
        if task_id in self._running_tasks:
            return {"success": False, "error": "任务正在执行中"}
        
        try:
            async with get_db_session() as db:
                task_repo = SupervisorTaskRepository(db)
                decision_repo = SupervisorDecisionRepository(db)
                
                # 获取任务详情
                task = await task_repo.get_by_id(task_id)
                if not task:
                    return {"success": False, "error": "任务不存在"}
                
                if task.status != TaskStatus.PENDING.value:
                    return {"success": False, "error": f"任务状态不是pending: {task.status}"}
                
                # 获取分配的智能体
                agent_name = task.assigned_agent_name
                if not agent_name or agent_name == "未分配":
                    # 如果任务没有分配智能体，根据任务类型自动分配一个合适的智能体
                    task_type_mapping = {
                        "code_generation": "代码专家",
                        "code_review": "代码专家", 
                        "documentation": "文档专家",
                        "analysis": "架构师",
                        "planning": "任务调度器",
                        "architecture": "架构师",
                        "knowledge_retrieval": "知识检索专家"
                    }
                    agent_name = task_type_mapping.get(task.task_type, "架构师")  # 默认使用架构师
                    logger.info("任务未分配智能体，自动分配", task_id=task_id, task_type=task.task_type, assigned_agent=agent_name)
                
                if agent_name not in self._agents:
                    return {"success": False, "error": f"分配的智能体不可用: {agent_name}"}
                
                agent = self._agents[agent_name]
                self._running_tasks[task_id] = agent_name
                
                logger.info("开始执行任务", task_id=task_id, agent_name=agent_name, task_name=task.name)
                
                # 更新任务状态为running
                await task_repo.update_task_status(task_id, TaskStatus.RUNNING)
                
                # 确保状态更新后有足够时间让前端获取到running状态
                await asyncio.sleep(3)  # 增加到3秒，让running状态更容易被观察到  
                
                # 执行任务
                start_time = utc_now()
                
                execution_result = await agent.execute_task(
                    task_description=task.description or task.name,
                    task_type=task.task_type,
                    input_data=task.input_data or {}
                )
                
                end_time = utc_now()
                execution_duration = int((end_time - start_time).total_seconds())
                
                # 更新任务状态和结果
                if execution_result.get("success", False):
                    await task_repo.update_task_status(
                        task_id=task_id,
                        status=TaskStatus.COMPLETED,
                        output_data=execution_result,
                        actual_time_seconds=execution_duration
                    )
                    task_success = True
                else:
                    await task_repo.update_task_status(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        output_data=execution_result,
                        actual_time_seconds=execution_duration
                    )
                    task_success = False
                
                # 更新对应的决策记录
                decision = await decision_repo.get_by_task_id(task_id)
                if decision:
                    await decision_repo.update_decision_outcome(
                        decision_id=decision.decision_id,
                        task_success=task_success,
                        actual_completion_time=end_time,
                        quality_score=execution_result.get("quality_score")
                    )
                
                logger.info("任务执行完成", 
                          task_id=task_id, 
                          success=task_success, 
                          duration=execution_duration)
                
                return {
                    "success": True,
                    "task_id": task_id,
                    "agent_name": agent_name,
                    "execution_result": execution_result,
                    "duration_seconds": execution_duration
                }
                
        except Exception as e:
            logger.error("任务执行异常", task_id=task_id, error=str(e))
            # 更新任务状态为失败
            try:
                async with get_db_session() as db:
                    task_repo = SupervisorTaskRepository(db)
                    await task_repo.update_task_status(
                        task_id=task_id,
                        status=TaskStatus.FAILED,
                        output_data={"error": str(e)}
                    )
            except Exception as update_error:
                logger.error("更新任务失败状态异常", task_id=task_id, error=str(update_error))
            
            return {"success": False, "error": str(e)}
        
        finally:
            # 清理运行状态
            self._running_tasks.pop(task_id, None)
    
    async def get_pending_tasks(self, supervisor_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取待执行任务"""
        try:
            async with get_db_session() as db:
                task_repo = SupervisorTaskRepository(db)
                
                if supervisor_id:
                    # 获取特定supervisor的pending任务
                    tasks = await task_repo.get_tasks_by_status(supervisor_id, TaskStatus.PENDING)
                else:
                    tasks = await task_repo.get_all_tasks_by_status(TaskStatus.PENDING)
                
                return [
                    {
                        "task_id": task.id,
                        "name": task.name,
                        "description": task.description,
                        "task_type": task.task_type,
                        "assigned_agent": task.assigned_agent_name,
                        "supervisor_id": task.supervisor_id,
                        "created_at": task.created_at.isoformat() if task.created_at else None
                    }
                    for task in tasks
                ]
                
        except Exception as e:
            logger.error("获取待执行任务失败", error=str(e))
            return []
    
    async def execute_pending_tasks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """批量执行pending任务"""
        results = []
        
        # 获取所有活跃supervisor的pending任务（使用缓存）
        try:
            active_supervisors = await self._get_cached_active_supervisors()
            
            if not active_supervisors:
                logger.debug("没有找到活跃的supervisors")
                return results
                
            for supervisor in active_supervisors:
                    pending_tasks = await self.get_pending_tasks(supervisor.id)
                    
                    # 限制同时执行的任务数量
                    tasks_to_execute = pending_tasks[:limit]
                    
                    for task_info in tasks_to_execute:
                        if len(self._running_tasks) >= 10:  # 最大并发任务数
                            break
                        
                        result = await self.execute_task(task_info["task_id"])
                        results.append(result)
                
        except Exception as e:
            logger.error("批量执行任务失败", error=str(e))
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """获取执行器状态"""
        return {
            "available_agents": list(self._agents.keys()),
            "running_tasks": len(self._running_tasks),
            "running_task_details": dict(self._running_tasks),
            "max_concurrent_tasks": 10
        }

# 全局任务执行器实例
task_executor = TaskExecutor()
