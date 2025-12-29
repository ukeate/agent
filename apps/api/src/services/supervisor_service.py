"""
Supervisor业务逻辑层实现
提供Supervisor相关的业务操作和协调功能
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
import uuid
import inspect
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_db_session
from src.ai.autogen.supervisor_agent import (
    SupervisorAgent, TaskType, TaskPriority, TaskAssignment, 
    SupervisorDecision, TaskComplexity
)
from src.ai.autogen.agents import BaseAutoGenAgent
from src.ai.autogen.config import AGENT_CONFIGS, AgentRole
from src.repositories.supervisor_repository import (
    SupervisorRepository, SupervisorTaskRepository, 
    SupervisorDecisionRepository, AgentLoadMetricsRepository,
    SupervisorConfigRepository
)
from src.models.database.supervisor import (
    SupervisorAgent as DBSupervisorAgent,
    SupervisorTask as DBSupervisorTask,
    SupervisorDecision as DBSupervisorDecision,
    SupervisorConfig as DBSupervisorConfig
)
from src.models.schemas.supervisor import (
    TaskSubmissionRequest, TaskSubmissionResponse, SupervisorStatusResponse,
    SupervisorConfigUpdateRequest, TaskStatus, AgentStatus

)

from src.core.logging import get_logger
logger = get_logger(__name__)

class SupervisorService:
    """Supervisor业务服务"""
    
    def __init__(self):
        self._supervisor_agents: Dict[str, SupervisorAgent] = {}
        self._available_agents: Dict[str, BaseAutoGenAgent] = {}
    
    def _create_default_agent_pool(self) -> Dict[str, BaseAutoGenAgent]:
        """创建默认的智能体池"""
        try:
            from src.ai.autogen.agents import (
                CodeExpertAgent, ArchitectAgent, DocExpertAgent, 
                KnowledgeRetrievalExpertAgent, create_default_agents
            )
            
            logger.info("开始创建默认智能体池")
            
            # 使用create_default_agents函数创建所有默认智能体
            agents_list = create_default_agents()
            
            # 转换为字典格式
            agent_pool = {}
            for agent in agents_list:
                agent_pool[agent.config.name] = agent
                logger.info("智能体已加入池", agent_name=agent.config.name, role=agent.config.role)
            
            logger.info("默认智能体池创建完成", total_agents=len(agent_pool))
            return agent_pool
            
        except Exception as e:
            logger.error("创建默认智能体池失败", error=str(e))
            # 返回空字典作为fallback
            return {}
        
    async def initialize_supervisor(
        self, 
        supervisor_name: str,
        available_agents: Optional[Dict[str, BaseAutoGenAgent]] = None
    ) -> str:
        """初始化Supervisor智能体"""
        try:
            async with get_db_session() as db:
                supervisor_repo = SupervisorRepository(db)
                config_repo = SupervisorConfigRepository(db)
                
                # 检查是否已存在
                existing_supervisor = await supervisor_repo.get_by_name(supervisor_name)
                if existing_supervisor:
                    logger.info("Supervisor已存在", name=supervisor_name)
                    # 如果Supervisor已存在但内存中没有实例，创建实例
                    if existing_supervisor.id not in self._supervisor_agents:
                        # 创建默认智能体集合
                        if available_agents is None:
                            available_agents = self._create_default_agent_pool()
                        
                        supervisor_agent = SupervisorAgent(available_agents)
                        self._supervisor_agents[existing_supervisor.id] = supervisor_agent
                        self._supervisor_agents[existing_supervisor.name] = supervisor_agent
                        self._available_agents.update(available_agents)
                        
                        logger.info("现有Supervisor实例化完成", 
                                  supervisor_id=existing_supervisor.id,
                                  available_agents=list(available_agents.keys()))
                    
                    return existing_supervisor.id
                
                # 创建默认智能体集合（如果没有提供）
                if available_agents is None:
                    available_agents = self._create_default_agent_pool()
                
                # 创建新Supervisor
                supervisor_id = str(uuid.uuid4())
                db_supervisor = DBSupervisorAgent(
                    id=supervisor_id,
                    name=supervisor_name,
                    role="supervisor",
                    status=AgentStatus.ACTIVE.value,
                    capabilities=[
                        "任务分析", "智能体路由", "负载平衡", 
                        "决策制定", "质量评估", "学习优化"
                    ],
                    configuration={
                        "model": "claude-3-5-sonnet-20241022",
                        "temperature": 0.1,
                        "max_tokens": 4000
                    }
                )
                
                await supervisor_repo.create(db_supervisor)
                
                # 创建默认配置
                default_config = DBSupervisorConfig(
                    id=str(uuid.uuid4()),
                    supervisor_id=supervisor_id,
                    config_name="default",
                    config_version="1.0",
                    routing_strategy="hybrid",
                    load_threshold=0.8,
                    capability_weight=0.5,
                    load_weight=0.3,
                    availability_weight=0.2,
                    is_active=True
                )
                
                await config_repo.create(default_config)
                
                # 初始化Supervisor智能体实例
                supervisor_agent = SupervisorAgent(available_agents)
                self._supervisor_agents[supervisor_id] = supervisor_agent
                self._supervisor_agents[supervisor_name] = supervisor_agent
                self._available_agents.update(available_agents)
                
                logger.info("Supervisor初始化完成", 
                          supervisor_id=supervisor_id, 
                          name=supervisor_name,
                          available_agents=list(available_agents.keys()))
                
                return supervisor_id
                
        except Exception as e:
            logger.error("Supervisor初始化失败", name=supervisor_name, error=str(e))
            raise
    
    async def submit_task(
        self, 
        supervisor_id: str, 
        request: TaskSubmissionRequest
    ) -> TaskAssignment:
        """提交任务给Supervisor"""
        try:
            async with get_db_session() as db:
                task_repo = SupervisorTaskRepository(db)
                decision_repo = SupervisorDecisionRepository(db)
                
                # 获取Supervisor实例
                supervisor_agent = self._supervisor_agents.get(supervisor_id)
                if not supervisor_agent:
                    # 尝试从数据库加载Supervisor
                    supervisor_repo = SupervisorRepository(db)
                    db_supervisor = await supervisor_repo.get_by_id(supervisor_id)
                    if not db_supervisor:
                        # 尝试按名称查找
                        db_supervisor = await supervisor_repo.get_by_name(supervisor_id)
                    
                    if db_supervisor:
                        # 创建默认智能体池（如果没有可用智能体）
                        available_agents = self._available_agents
                        if not available_agents:
                            available_agents = self._create_default_agent_pool()
                            self._available_agents.update(available_agents)
                        
                        # 创建内存中的Supervisor实例
                        supervisor_agent = SupervisorAgent(available_agents)
                        self._supervisor_agents[db_supervisor.id] = supervisor_agent
                        # 同时也按名称索引
                        self._supervisor_agents[db_supervisor.name] = supervisor_agent
                        # 更新supervisor_id为实际的数据库ID
                        actual_supervisor_id = db_supervisor.id
                    else:
                        raise ValueError(f"Supervisor {supervisor_id} 未找到")
                else:
                    # 如果内存中已存在，也需要获取正确的supervisor_id
                    supervisor_repo = SupervisorRepository(db)
                    db_supervisor = await supervisor_repo.get_by_name(supervisor_id)
                    if not db_supervisor:
                        db_supervisor = await supervisor_repo.get_by_id(supervisor_id)
                    if db_supervisor:
                        actual_supervisor_id = db_supervisor.id
                    else:
                        actual_supervisor_id = supervisor_id  # 假设传入的就是正确的ID
                
                # 创建任务记录
                task_id = str(uuid.uuid4())
                db_task = DBSupervisorTask(
                    id=task_id,
                    name=request.name,
                    description=request.description,
                    task_type=request.task_type.value,
                    priority=request.priority.value,
                    status=TaskStatus.PENDING.value,
                    supervisor_id=actual_supervisor_id,
                    input_data=request.input_data or {}
                )
                
                await task_repo.create(db_task)
                
                # 使用Supervisor分析和分配任务
                assignment = await supervisor_agent.analyze_and_assign_task(
                    task_description=request.description,
                    task_type=request.task_type,
                    priority=request.priority,
                    constraints=request.constraints
                )
                
                # 更新任务分配信息
                logger.info("准备分配任务", 
                          task_id=task_id, 
                          assigned_agent=assignment.assigned_agent,
                          assignment_reason=assignment.assignment_reason)
                
                success = await task_repo.assign_task_to_agent(
                    task_id=task_id,
                    agent_id=assignment.assigned_agent,  # 这里暂时使用名称作为ID
                    agent_name=assignment.assigned_agent
                )
                
                if not success:
                    logger.error("任务分配失败", task_id=task_id, assigned_agent=assignment.assigned_agent)
                    raise ValueError(f"任务分配失败: {task_id}")
                
                logger.info("任务分配成功", 
                          task_id=task_id, 
                          assigned_agent=assignment.assigned_agent)
                
                # 更新任务复杂度和时间估算
                complexity_metadata = assignment.decision_metadata.get("complexity", {})
                if complexity_metadata:
                    await self._update_task_complexity(
                        db, task_id, complexity_metadata
                    )
                
                # 记录决策到数据库
                await self._save_decision_to_db(
                    db, actual_supervisor_id, task_id, assignment
                )
                
                logger.info("任务提交完成", 
                          task_id=task_id, 
                          assigned_agent=assignment.assigned_agent)
                
                return assignment
                
        except Exception as e:
            logger.error("任务提交失败", supervisor_id=supervisor_id, error=str(e))
            raise
    
    async def get_supervisor_status(self, supervisor_id: str) -> SupervisorStatusResponse:
        """获取Supervisor状态"""
        try:
            async with get_db_session() as db:
                supervisor_repo = SupervisorRepository(db)
                task_repo = SupervisorTaskRepository(db)
                load_repo = AgentLoadMetricsRepository(db)
                config_repo = SupervisorConfigRepository(db)
                
                # 获取Supervisor信息 - 支持按ID或名称查询
                db_supervisor = await supervisor_repo.get_by_name(supervisor_id)
                if not db_supervisor:
                    db_supervisor = await supervisor_repo.get_by_id(supervisor_id)
                if not db_supervisor:
                    raise ValueError(f"Supervisor {supervisor_id} 未找到")
                actual_supervisor_id = db_supervisor.id
                
                # 获取智能体负载
                agent_loads = await load_repo.get_current_loads(actual_supervisor_id)
                
                # 获取任务队列长度
                pending_tasks = await task_repo.get_pending_tasks(actual_supervisor_id)
                
                # 获取决策历史数量
                decision_repo = SupervisorDecisionRepository(db)
                decision_stats = await decision_repo.get_decision_statistics(actual_supervisor_id)
                
                # 获取当前配置
                active_config = await config_repo.get_active_config(actual_supervisor_id)
                config_data = None
                if active_config:
                    config_data = {
                        "routing_strategy": active_config.routing_strategy,
                        "load_threshold": active_config.load_threshold,
                        "capability_weight": active_config.capability_weight,
                        "load_weight": active_config.load_weight,
                        "availability_weight": active_config.availability_weight
                    }
                
                # 获取Supervisor实例状态
                supervisor_agent = self._supervisor_agents.get(actual_supervisor_id) or self._supervisor_agents.get(db_supervisor.name)
                available_agents = list(supervisor_agent.available_agents.keys()) if supervisor_agent else []
                
                return SupervisorStatusResponse(
                    supervisor_name=db_supervisor.name,
                    status=AgentStatus(db_supervisor.status),
                    available_agents=available_agents,
                    agent_loads=agent_loads,
                    decision_history_count=decision_stats.get("total_decisions", 0),
                    task_queue_length=len(pending_tasks),
                    performance_metrics=decision_stats,
                    current_config=config_data
                )
                
        except Exception as e:
            logger.error("获取Supervisor状态失败", supervisor_id=supervisor_id, error=str(e))
            raise
    
    async def get_decision_history(
        self, 
        supervisor_id: str, 
        limit: int = 10,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """获取决策历史"""
        try:
            async with get_db_session() as db:
                decision_repo = SupervisorDecisionRepository(db)
                
                decisions = await decision_repo.get_by_supervisor_id(
                    supervisor_id, limit, offset
                )
                
                return [
                    {
                        "id": d.id,
                        "decision_id": d.decision_id,
                        "task_id": d.task_id,
                        "task_description": d.task_description,
                        "assigned_agent": d.assigned_agent,
                        "assignment_reason": d.assignment_reason,
                        "confidence_level": d.confidence_level,
                        "match_score": d.match_score if d.match_score is not None else 0.0,
                        "routing_strategy": d.routing_strategy,
                        "alternative_agents": d.alternative_agents,
                        "alternatives_considered": d.alternatives_considered,  # 添加替代方案
                        "task_success": d.task_success,
                        "quality_score": d.quality_score,
                        "timestamp": d.timestamp.isoformat() if d.timestamp else None,
                        "estimated_completion_time": d.estimated_completion_time.isoformat() if d.estimated_completion_time else None,
                        "actual_completion_time": d.actual_completion_time.isoformat() if d.actual_completion_time else None,
                        "decision_metadata": d.decision_metadata,
                        "routing_metadata": d.routing_metadata  # 添加路由元数据
                    }
                    for d in decisions
                ]
                
        except Exception as e:
            logger.error("获取决策历史失败", supervisor_id=supervisor_id, error=str(e))
            # 不抛出异常，返回空列表
            return []

    async def get_decision_history_total(self, supervisor_id: str) -> int:
        """获取Supervisor决策总数"""
        try:
            async with get_db_session() as db:
                decision_repo = SupervisorDecisionRepository(db)
                return await decision_repo.count_by_supervisor_id(supervisor_id)
        except Exception as e:
            logger.error("获取决策总数失败", supervisor_id=supervisor_id, error=str(e))
            return 0
    
    async def update_supervisor_config(
        self, 
        supervisor_id: str, 
        request: SupervisorConfigUpdateRequest
    ) -> DBSupervisorConfig:
        """更新Supervisor配置"""
        try:
            async with get_db_session() as db:
                config_repo = SupervisorConfigRepository(db)
                supervisor_repo = SupervisorRepository(db)
                
                # 验证Supervisor存在
                supervisor = await supervisor_repo.get_by_id(supervisor_id)
                if not supervisor:
                    raise ValueError(f"Supervisor {supervisor_id} 未找到")
                
                # 获取当前活跃配置
                current_config = await config_repo.get_active_config(supervisor_id)
                
                if current_config:
                    # 停用当前配置
                    await config_repo.deactivate_all_configs(supervisor_id)
                
                # 创建新配置
                config_data = {}
                if request.config_name:
                    config_data["config_name"] = request.config_name
                if request.routing_strategy:
                    config_data["routing_strategy"] = request.routing_strategy.value
                if request.load_threshold is not None:
                    config_data["load_threshold"] = request.load_threshold
                if request.capability_weight is not None:
                    config_data["capability_weight"] = request.capability_weight
                if request.load_weight is not None:
                    config_data["load_weight"] = request.load_weight
                if request.availability_weight is not None:
                    config_data["availability_weight"] = request.availability_weight
                if request.enable_quality_assessment is not None:
                    config_data["enable_quality_assessment"] = request.enable_quality_assessment
                if request.min_confidence_threshold is not None:
                    config_data["min_confidence_threshold"] = request.min_confidence_threshold
                if request.enable_learning is not None:
                    config_data["enable_learning"] = request.enable_learning
                if request.learning_rate is not None:
                    config_data["learning_rate"] = request.learning_rate
                if request.max_concurrent_tasks is not None:
                    config_data["max_concurrent_tasks"] = request.max_concurrent_tasks
                if request.task_timeout_minutes is not None:
                    config_data["task_timeout_minutes"] = request.task_timeout_minutes
                if request.enable_fallback is not None:
                    config_data["enable_fallback"] = request.enable_fallback
                
                # 继承当前配置的未更新字段
                if current_config:
                    for field in [
                        "config_name", "routing_strategy", "load_threshold",
                        "capability_weight", "load_weight", "availability_weight",
                        "enable_quality_assessment", "min_confidence_threshold",
                        "enable_learning", "learning_rate", "max_concurrent_tasks",
                        "task_timeout_minutes", "enable_fallback"
                    ]:
                        if field not in config_data:
                            config_data[field] = getattr(current_config, field)
                
                new_config = DBSupervisorConfig(
                    id=str(uuid.uuid4()),
                    supervisor_id=supervisor_id,
                    config_version="1.1",
                    is_active=True,
                    **config_data
                )
                
                await config_repo.create(new_config)
                
                # 更新内存中的Supervisor配置
                supervisor_agent = self._supervisor_agents.get(supervisor_id)
                if supervisor_agent:
                    if hasattr(supervisor_agent, "update_config"):
                        result = supervisor_agent.update_config(config_data)
                        if inspect.isawaitable(result):
                            await result
                    else:
                        supervisor_agent.supervisor_config = config_data
                
                logger.info("Supervisor配置已更新", supervisor_id=supervisor_id)
                
                return new_config
                
        except Exception as e:
            logger.error("更新Supervisor配置失败", supervisor_id=supervisor_id, error=str(e))
            raise
    
    async def add_agent_to_supervisor(
        self, 
        supervisor_id: str, 
        agent_name: str, 
        agent: BaseAutoGenAgent
    ):
        """添加智能体到Supervisor"""
        try:
            supervisor_agent = self._supervisor_agents.get(supervisor_id)
            if supervisor_agent:
                supervisor_agent.add_agent(agent_name, agent)
                self._available_agents[agent_name] = agent
                
                logger.info("智能体已添加到Supervisor", 
                          supervisor_id=supervisor_id, 
                          agent_name=agent_name)
            else:
                raise ValueError(f"Supervisor {supervisor_id} 未找到")
                
        except Exception as e:
            logger.error("添加智能体失败", 
                        supervisor_id=supervisor_id, 
                        agent_name=agent_name, 
                        error=str(e))
            raise
    
    async def remove_agent_from_supervisor(
        self, 
        supervisor_id: str, 
        agent_name: str
    ):
        """从Supervisor移除智能体"""
        try:
            supervisor_agent = self._supervisor_agents.get(supervisor_id)
            if supervisor_agent:
                supervisor_agent.remove_agent(agent_name)
                if agent_name in self._available_agents:
                    del self._available_agents[agent_name]
                
                logger.info("智能体已从Supervisor移除", 
                          supervisor_id=supervisor_id, 
                          agent_name=agent_name)
            else:
                raise ValueError(f"Supervisor {supervisor_id} 未找到")
                
        except Exception as e:
            logger.error("移除智能体失败", 
                        supervisor_id=supervisor_id, 
                        agent_name=agent_name, 
                        error=str(e))
            raise
    
    async def update_task_completion(
        self, 
        task_id: str, 
        success: bool,
        output_data: Optional[Dict[str, Any]] = None,
        quality_score: Optional[float] = None
    ):
        """更新任务完成情况"""
        try:
            async with get_db_session() as db:
                task_repo = SupervisorTaskRepository(db)
                decision_repo = SupervisorDecisionRepository(db)
                
                # 更新任务状态
                new_status = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                await task_repo.update_task_status(
                    task_id=task_id,
                    status=new_status,
                    output_data=output_data
                )
                
                # 更新相关决策结果
                decision = await decision_repo.get_by_task_id(task_id)
                if decision:
                    await decision_repo.update_decision_outcome(
                        decision_id=decision.decision_id,
                        task_success=success,
                        actual_completion_time=utc_now(),
                        quality_score=quality_score
                    )
                
                logger.info("任务完成情况已更新", 
                          task_id=task_id, 
                          success=success)
                
        except Exception as e:
            logger.error("更新任务完成情况失败", task_id=task_id, error=str(e))
            raise
    
    async def _save_decision_to_db(
        self, 
        db: AsyncSession, 
        supervisor_id: str, 
        task_id: str, 
        assignment: TaskAssignment
    ):
        """保存决策到数据库"""
        try:
            decision_repo = SupervisorDecisionRepository(db)
            
            # 转换alternatives格式
            alternatives_considered = []
            if assignment.decision_metadata and "alternatives" in assignment.decision_metadata:
                for alt in assignment.decision_metadata["alternatives"]:
                    alternatives_considered.append({
                        "agent": alt.get("agent_name", ""),
                        "score": alt.get("match_score", 0.0),
                        "reason": f"匹配度: {alt.get('match_score', 0.0)*100:.1f}%, 负载: {alt.get('load_factor', 0.0)*100:.1f}%"
                    })
            
            db_decision = DBSupervisorDecision(
                id=str(uuid.uuid4()),
                decision_id=assignment.task_id.replace("task_", "decision_"),
                supervisor_id=supervisor_id,
                task_id=task_id,
                task_description=assignment.decision_metadata.get("task_description", ""),
                assigned_agent=assignment.assigned_agent,
                assignment_reason=assignment.assignment_reason,
                confidence_level=assignment.confidence_level,
                match_score=assignment.decision_metadata.get("match_details", {}).get("match_score", 0.0),
                routing_strategy=assignment.decision_metadata.get("routing_strategy", "capability_based"),  # 添加路由策略
                alternative_agents=assignment.alternative_agents,
                alternatives_considered=alternatives_considered,  # 添加转换后的替代方案
                decision_metadata=assignment.decision_metadata,
                estimated_completion_time=assignment.estimated_completion_time
            )
            
            await decision_repo.create(db_decision)
            
        except Exception as e:
            logger.error("保存决策记录失败", error=str(e))
            raise
    
    async def _update_task_complexity(
        self, 
        db: AsyncSession, 
        task_id: str, 
        complexity_data: Dict[str, Any]
    ):
        """更新任务复杂度信息"""
        try:
            from sqlalchemy import update
            
            # 使用异步SQLAlchemy更新
            stmt = update(DBSupervisorTask).where(
                DBSupervisorTask.id == task_id
            ).values(
                complexity_score=complexity_data.get("score", 0.0),
                estimated_time_seconds=complexity_data.get("estimated_time", 0)
            )
            
            await db.execute(stmt)
            await db.commit()
            
        except Exception as e:
            logger.error("更新任务复杂度失败", task_id=task_id, error=str(e))
            raise

# 单例服务实例
supervisor_service = SupervisorService()
