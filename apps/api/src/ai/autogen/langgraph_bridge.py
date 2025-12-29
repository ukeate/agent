"""
AutoGen与LangGraph集成桥接器
实现AutoGen和LangGraph的深度集成和统一上下文管理
"""

import asyncio
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Dict, List, Optional, Any, Union, Callable
from .agents import BaseAutoGenAgent
from .config import AgentConfig
from .async_manager import AsyncAgentManager, AgentTask
from .events import Event, EventType, EventBus, EventHandler
from src.ai.langgraph.context import AgentContext
from src.ai.langgraph.state_graph import StateGraph

from src.core.logging import get_logger
logger = get_logger(__name__)

class AutoGenLangGraphBridge:
    """AutoGen和LangGraph的桥接器"""
    
    def __init__(
        self,
        agent_manager: AsyncAgentManager,
        event_bus: EventBus,
        state_graph: Optional[StateGraph] = None
    ):
        self.agent_manager = agent_manager
        self.event_bus = event_bus
        self.state_graph = state_graph
        
        # 上下文缓存
        self.context_cache: Dict[str, AgentContext] = {}
        
        # 注册事件处理器
        self._register_event_handlers()
        
        logger.info("AutoGen-LangGraph桥接器初始化完成")
    
    def _register_event_handlers(self):
        """注册事件处理器"""
        bridge = self

        class _BridgeEventHandler(EventHandler):
            @property
            def supported_events(self) -> List[EventType]:
                return [EventType.AGENT_DESTROYED]

            async def handle(self, event: Event) -> None:
                if event.target and event.target in bridge.context_cache:
                    bridge.context_cache.pop(event.target, None)

        self.event_bus.subscribe(EventType.AGENT_DESTROYED, _BridgeEventHandler())
    
    async def create_contextual_agent(
        self,
        config: AgentConfig,
        context: AgentContext,
        agent_id: Optional[str] = None
    ) -> str:
        """创建带有上下文的智能体"""
        try:
            # 增强配置，注入LangGraph上下文信息
            enhanced_config = self._enhance_config_with_context(config, context)
            
            # 创建智能体
            agent_id = await self.agent_manager.create_agent(enhanced_config, agent_id)
            
            # 缓存上下文
            self.context_cache[agent_id] = context
            
            # 发布集成事件
            await self.event_bus.publish(Event(
                type=EventType.AGENT_CREATED,
                source="langgraph_bridge",
                target=agent_id,
                data={
                    "agent_id": agent_id,
                    "context_integration": True,
                    "user_id": context.user_id,
                    "session_id": context.session_id,
                    "conversation_id": context.conversation_id
                }
            ))
            
            logger.info(
                "上下文智能体创建成功",
                agent_id=agent_id,
                user_id=context.user_id,
                session_id=context.session_id
            )
            
            return agent_id
            
        except Exception as e:
            logger.error("创建上下文智能体失败", error=str(e))
            raise
    
    def _enhance_config_with_context(
        self,
        config: AgentConfig,
        context: AgentContext
    ) -> AgentConfig:
        """使用上下文增强智能体配置"""
        # 构建上下文信息字符串
        context_info = f"""
上下文信息:
- 用户ID: {context.user_id}
- 会话ID: {context.session_id}
- 对话ID: {context.conversation_id or '新对话'}
- 时间戳: {utc_now().isoformat()}
"""
        
        # 如果有额外的上下文数据，添加到提示中
        if hasattr(context, 'additional_context') and context.additional_context:
            context_info += f"- 额外上下文: {context.additional_context}\n"
        
        # 增强系统提示词
        enhanced_prompt = config.system_prompt + "\n\n" + context_info + """
请在回复时考虑上述上下文信息，确保响应的连贯性和相关性。
"""
        
        # 创建增强的配置
        enhanced_config = AgentConfig(
            name=config.name,
            role=config.role,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            system_prompt=enhanced_prompt,
            tools=config.tools,
            capabilities=config.capabilities
        )
        
        return enhanced_config
    
    async def execute_contextual_task(
        self,
        agent_id: str,
        task_type: str,
        description: str,
        input_data: Dict[str, Any],
        priority: int = 0
    ) -> str:
        """执行带上下文的任务"""
        try:
            # 检查是否有缓存的上下文
            context = self.context_cache.get(agent_id)
            if context:
                # 将上下文信息注入到输入数据中
                enhanced_input_data = {
                    **input_data,
                    "_context": {
                        "user_id": context.user_id,
                        "session_id": context.session_id,
                        "conversation_id": context.conversation_id,
                        "timestamp": utc_now().isoformat()
                    }
                }
            else:
                enhanced_input_data = input_data
            
            # 提交任务
            task_id = await self.agent_manager.submit_task(
                agent_id=agent_id,
                task_type=task_type,
                description=description,
                input_data=enhanced_input_data,
                priority=priority
            )
            
            logger.info(
                "上下文任务提交成功",
                task_id=task_id,
                agent_id=agent_id,
                has_context=context is not None
            )
            
            return task_id
            
        except Exception as e:
            logger.error("执行上下文任务失败", agent_id=agent_id, error=str(e))
            raise
    
    async def create_agent_workflow(
        self,
        workflow_config: Dict[str, Any],
        context: AgentContext
    ) -> str:
        """创建智能体工作流"""
        try:
            workflow_id = f"workflow_{utc_now().strftime('%Y%m%d_%H%M%S')}"
            
            # 解析工作流配置
            agents_config = workflow_config.get("agents", [])
            tasks_config = workflow_config.get("tasks", [])
            dependencies = workflow_config.get("dependencies", {})
            
            # 创建工作流中的所有智能体
            created_agents = {}
            for agent_config_data in agents_config:
                agent_config = AgentConfig(**agent_config_data)
                role_key = agent_config.role.value if hasattr(agent_config.role, "value") else str(agent_config.role)
                agent_id = await self.create_contextual_agent(
                    agent_config,
                    context,
                    agent_id=f"{workflow_id}_{role_key}"
                )
                created_agents[role_key] = agent_id
            
            # 提交工作流任务
            workflow_tasks = []
            for task_config in tasks_config:
                agent_role = task_config.get("agent_role")
                agent_role_key = agent_role.value if hasattr(agent_role, "value") else str(agent_role)
                if agent_role_key in created_agents:
                    task_id = await self.execute_contextual_task(
                        agent_id=created_agents[agent_role_key],
                        task_type=task_config.get("task_type", "general"),
                        description=task_config.get("description", ""),
                        input_data=task_config.get("input_data", {}),
                        priority=task_config.get("priority", 0)
                    )
                    workflow_tasks.append(task_id)
            
            # 发布工作流创建事件
            await self.event_bus.publish(Event(
                type=EventType.CONVERSATION_STARTED,
                source="langgraph_bridge",
                data={
                    "workflow_id": workflow_id,
                    "agents": list(created_agents.values()),
                    "tasks": workflow_tasks,
                    "context": {
                        "user_id": context.user_id,
                        "session_id": context.session_id
                    }
                }
            ))
            
            logger.info(
                "智能体工作流创建成功",
                workflow_id=workflow_id,
                agent_count=len(created_agents),
                task_count=len(workflow_tasks)
            )
            
            return workflow_id
            
        except Exception as e:
            logger.error("创建智能体工作流失败", error=str(e))
            raise
    
    async def sync_with_langgraph_state(
        self,
        agent_id: str,
        langgraph_state: Dict[str, Any]
    ) -> bool:
        """与LangGraph状态同步"""
        try:
            if not self.state_graph:
                logger.warning("StateGraph未配置，跳过状态同步")
                return False
            
            # 获取智能体信息
            agent_info = await self.agent_manager.get_agent_info(agent_id)
            if not agent_info:
                logger.error("智能体不存在", agent_id=agent_id)
                return False
            
            # 更新智能体状态管理器中的状态
            await self.agent_manager.state_manager.update_agent_state(agent_id, {
                "langgraph_state": langgraph_state,
                "last_sync": utc_now().isoformat()
            })
            
            # 发布状态同步事件
            await self.event_bus.publish(Event(
                type=EventType.AGENT_STATUS_CHANGED,
                source="langgraph_bridge",
                target=agent_id,
                data={
                    "agent_id": agent_id,
                    "sync_type": "langgraph_state",
                    "state_keys": list(langgraph_state.keys())
                }
            ))
            
            logger.debug("LangGraph状态同步成功", agent_id=agent_id)
            return True
            
        except Exception as e:
            logger.error("LangGraph状态同步失败", agent_id=agent_id, error=str(e))
            return False
    
    async def get_context_for_agent(self, agent_id: str) -> Optional[AgentContext]:
        """获取智能体的上下文"""
        return self.context_cache.get(agent_id)
    
    async def update_agent_context(
        self,
        agent_id: str,
        context_updates: Dict[str, Any]
    ) -> bool:
        """更新智能体上下文"""
        try:
            context = self.context_cache.get(agent_id)
            if not context:
                logger.warning("智能体上下文不存在", agent_id=agent_id)
                return False
            
            # 更新上下文（这里假设AgentContext支持更新）
            for key, value in context_updates.items():
                if hasattr(context, key):
                    setattr(context, key, value)
            
            # 更新缓存
            self.context_cache[agent_id] = context
            
            # 发布上下文更新事件
            await self.event_bus.publish(Event(
                type=EventType.AGENT_STATUS_CHANGED,
                source="langgraph_bridge",
                target=agent_id,
                data={
                    "agent_id": agent_id,
                    "context_updated": True,
                    "updated_fields": list(context_updates.keys())
                }
            ))
            
            logger.info("智能体上下文更新成功", agent_id=agent_id)
            return True
            
        except Exception as e:
            logger.error("更新智能体上下文失败", agent_id=agent_id, error=str(e))
            return False
    
    async def execute_collaborative_task(
        self,
        agent_ids: List[str],
        task_description: str,
        coordination_strategy: str = "sequential"
    ) -> List[str]:
        """执行协作任务"""
        try:
            collaboration_id = f"collab_{utc_now().strftime('%Y%m%d_%H%M%S')}"
            task_ids = []
            
            if coordination_strategy == "sequential":
                # 顺序执行
                current_input = {"description": task_description}
                for i, agent_id in enumerate(agent_ids):
                    task_id = await self.execute_contextual_task(
                        agent_id=agent_id,
                        task_type="collaborative",
                        description=f"协作任务 {i+1}/{len(agent_ids)}: {task_description}",
                        input_data={
                            **current_input,
                            "collaboration_id": collaboration_id,
                            "sequence_number": i + 1,
                            "total_agents": len(agent_ids)
                        },
                        priority=1
                    )
                    task_ids.append(task_id)
                    
                    # 等待当前任务完成，获取结果作为下一个智能体的输入
                    # 这里可以添加等待和结果传递逻辑
                    
            elif coordination_strategy == "parallel":
                # 并行执行
                for i, agent_id in enumerate(agent_ids):
                    task_id = await self.execute_contextual_task(
                        agent_id=agent_id,
                        task_type="collaborative",
                        description=f"并行协作任务: {task_description}",
                        input_data={
                            "description": task_description,
                            "collaboration_id": collaboration_id,
                            "agent_index": i,
                            "total_agents": len(agent_ids)
                        },
                        priority=1
                    )
                    task_ids.append(task_id)
            
            # 发布协作开始事件
            await self.event_bus.publish(Event(
                type=EventType.CONVERSATION_STARTED,
                source="langgraph_bridge",
                data={
                    "collaboration_id": collaboration_id,
                    "strategy": coordination_strategy,
                    "participating_agents": agent_ids,
                    "tasks": task_ids,
                    "description": task_description
                }
            ))
            
            logger.info(
                "协作任务启动成功",
                collaboration_id=collaboration_id,
                strategy=coordination_strategy,
                agent_count=len(agent_ids),
                task_count=len(task_ids)
            )
            
            return task_ids
            
        except Exception as e:
            logger.error("执行协作任务失败", error=str(e))
            raise
    
    async def cleanup_agent_context(self, agent_id: str) -> bool:
        """清理智能体上下文"""
        try:
            if agent_id in self.context_cache:
                del self.context_cache[agent_id]
                logger.info("智能体上下文清理成功", agent_id=agent_id)
                return True
            return False
        except Exception as e:
            logger.error("清理智能体上下文失败", agent_id=agent_id, error=str(e))
            return False
    
    def get_bridge_stats(self) -> Dict[str, Any]:
        """获取桥接器统计信息"""
        return {
            "cached_contexts": len(self.context_cache),
            "state_graph_available": self.state_graph is not None,
            "agent_manager_stats": self.agent_manager.get_manager_stats(),
            "event_bus_stats": self.event_bus.get_stats()
        }
