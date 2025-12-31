"""
AutoGen 0.7.x 智能体实现
使用新版本的autogen-agentchat API
"""

from typing import Dict, List, Optional, Any, Union
from autogen_agentchat.agents import AssistantAgent, BaseChatAgent
from autogen_agentchat.messages import TextMessage
from autogen_core.models import ChatCompletionClient
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from src.ai.openai_client import get_openai_client
from src.core.config import get_settings
from .config import AgentConfig, AgentRole, AGENT_CONFIGS

from src.core.logging import get_logger
logger = get_logger(__name__)

class BaseAutoGenAgent:
    """AutoGen智能体基类封装"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self._agent: Optional[AssistantAgent] = None
        self._settings = get_settings()
        self._initialize_agent()
    
    def _initialize_agent(self):
        """初始化AutoGen智能体"""
        try:
            # 创建OpenAI客户端
            model_client = OpenAIChatCompletionClient(
                model=self.config.model,
                api_key=self._settings.OPENAI_API_KEY,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            # 创建AssistantAgent
            self._agent = AssistantAgent(
                name=self.config.name,
                model_client=model_client,
                system_message=self.config.system_prompt,
            )
            
            logger.info(
                "AutoGen智能体初始化成功",
                agent_name=self.config.name,
                role=self.config.role,
                model=self.config.model
            )
            
        except Exception as e:
            logger.error(
                "AutoGen智能体初始化失败",
                agent_name=self.config.name,
                error=str(e)
            )
            raise
    
    @property
    def agent(self) -> AssistantAgent:
        """获取AutoGen智能体实例"""
        if self._agent is None:
            raise ValueError("智能体未初始化")
        return self._agent
    
    async def generate_response(
        self, 
        message: str,
        cancellation_token: Optional[CancellationToken] = None
    ) -> str:
        """生成响应"""
        try:
            logger.info(f"开始生成响应 - 智能体: {self.config.name}, 输入消息长度: {len(message)}")
            
            # 直接使用模型客户端，包含系统提示词
            from autogen_core.models import UserMessage, SystemMessage
            
            # 创建包含系统提示词的消息列表
            messages = [
                SystemMessage(content=self.config.system_prompt, source="system"),
                UserMessage(content=message, source="user")
            ]
            logger.info(f"创建消息完成 - 智能体: {self.config.name}, 包含系统提示词: {len(self.config.system_prompt)} 字符")
            
            # 检查模型客户端是否存在
            if not self._agent or not self._agent._model_client:
                logger.error(f"模型客户端未初始化 - 智能体: {self.config.name}")
                return f"我是{self.config.name}，模型客户端未初始化。"
            
            logger.info(f"准备调用模型客户端 - 智能体: {self.config.name}")
            
            # 直接使用模型客户端生成响应
            response = await self._agent._model_client.create(
                messages=messages,
                cancellation_token=cancellation_token or CancellationToken(),
            )
            
            logger.info(f"模型客户端调用完成 - 智能体: {self.config.name}, 响应类型: {type(response)}")
            
            # 提取响应内容
            if hasattr(response, 'content'):
                content = response.content
                logger.info(f"使用response.content提取内容 - 智能体: {self.config.name}")
            elif hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                logger.info(f"使用response.choices[0].message.content提取内容 - 智能体: {self.config.name}")
            else:
                content = str(response)
                logger.info(f"使用str(response)提取内容 - 智能体: {self.config.name}")
            
            logger.info(
                "智能体响应生成成功",
                agent_name=self.config.name,
                message_length=len(message),
                response_length=len(content),
                content_preview=content[:100] + "..." if len(content) > 100 else content
            )
            
            return content
            
        except Exception as e:
            import traceback
            logger.error(
                "智能体响应生成失败",
                agent_name=self.config.name,
                error=str(e),
                traceback=traceback.format_exc()
            )
            # 返回一个简单的错误响应而不是抛出异常
            return f"我是{self.config.name}，很抱歉遇到了技术问题: {str(e)}"
    
    async def generate_streaming_response(
        self, 
        message: str,
        stream_callback=None,
        cancellation_token: Optional[CancellationToken] = None
    ) -> str:
        """生成流式响应"""
        try:
            logger.info(f"开始生成流式响应 - 智能体: {self.config.name}, 输入消息长度: {len(message)}")
            
            # 检查取消令牌
            if cancellation_token and cancellation_token.is_cancelled():
                logger.info(f"流式响应被取消 - 智能体: {self.config.name}")
                return f"我是{self.config.name}，响应已被暂停。"
            
            # 使用OpenAI客户端直接进行流式调用
            from src.ai.openai_client import OpenAIClient
            
            openai_client = OpenAIClient()
            
            # 构建消息
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": message}
            ]
            
            logger.info(f"准备进行流式调用 - 智能体: {self.config.name}")
            
            full_content = ""
            chunk_count = 0
            
            # 进行流式调用
            async for chunk in openai_client.create_streaming_completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            ):
                # 关键修复：每个chunk都检查取消令牌
                if cancellation_token and cancellation_token.is_cancelled():
                    logger.info(f"流式响应在第{chunk_count}个chunk时被取消 - 智能体: {self.config.name}")
                    
                    # 发送取消回调
                    if stream_callback:
                        await stream_callback({
                            "type": "cancelled",
                            "content": "",
                            "full_content": full_content + f" [响应被暂停]",
                            "agent_name": self.config.name,
                            "chunk_count": chunk_count,
                            "is_complete": True
                        })
                    
                    return full_content + f" [响应被暂停]" if full_content else f"我是{self.config.name}，响应已被暂停。"
                
                chunk_count += 1
                token_content = chunk.get("content", "")
                
                if token_content:
                    full_content += token_content
                    
                    # 调用流式回调
                    if stream_callback:
                        await stream_callback({
                            "type": "token",
                            "content": token_content,
                            "full_content": full_content,
                            "agent_name": self.config.name,
                            "chunk_count": chunk_count,
                            "is_complete": False
                        })
                
                # 检查是否完成
                if chunk.get("finish_reason"):
                    logger.info(f"流式响应完成 - 智能体: {self.config.name}, 总块数: {chunk_count}, 总长度: {len(full_content)}")
                    
                    # 发送完成回调
                    if stream_callback:
                        await stream_callback({
                            "type": "complete",
                            "content": "",
                            "full_content": full_content,
                            "agent_name": self.config.name,
                            "chunk_count": chunk_count,
                            "is_complete": True
                        })
                    break
            
            logger.info(
                "流式响应生成成功",
                agent_name=self.config.name,
                message_length=len(message),
                response_length=len(full_content),
                chunk_count=chunk_count,
                content_preview=full_content[:100] + "..." if len(full_content) > 100 else full_content
            )
            
            return full_content
            
        except Exception as e:
            import traceback
            logger.error(
                "流式响应生成失败",
                agent_name=self.config.name,
                error=str(e),
                traceback=traceback.format_exc()
            )
            
            # 发送错误回调
            if stream_callback:
                try:
                    await stream_callback({
                        "type": "error",
                        "content": f"遇到了技术问题: {str(e)}",
                        "full_content": f"我是{self.config.name}，很抱歉遇到了技术问题: {str(e)}",
                        "agent_name": self.config.name,
                        "error": str(e),
                        "is_complete": True
                    })
                except Exception as callback_error:
                    logger.error(f"流式回调发送错误消息失败: {callback_error}")
            
            return f"我是{self.config.name}，很抱歉遇到了技术问题: {str(e)}"
    
    def get_status(self) -> Dict[str, Any]:
        """获取智能体状态"""
        return {
            "name": self.config.name,
            "role": self.config.role,
            "status": "active" if self._agent else "inactive",
            "model": self.config.model,
            "capabilities": self.config.capabilities,
        }
    
    async def execute_task(self, task_description: str, task_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务的统一接口 - 基类默认实现"""
        import time
        try:
            start = time.perf_counter()
            # 构造执行提示
            prompt = f"""
任务类型: {task_type}
任务描述: {task_description}
输入数据: {input_data if input_data else '无'}

请根据你的角色和专业能力执行这个任务，并提供详细的结果。
"""
            
            # 调用智能体生成响应
            response = await self.generate_response(prompt)
            
            return {
                "success": True,
                "result": response,
                "agent_name": self.config.name,
                "task_type": task_type,
                "execution_time": round(time.perf_counter() - start, 6)
            }
            
        except Exception as e:
            logger.error("任务执行失败", agent_name=self.config.name, task_type=task_type, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "agent_name": self.config.name,
                "task_type": task_type
            }

class CodeExpertAgent(BaseAutoGenAgent):
    """代码专家智能体"""
    
    def __init__(self, custom_config: Optional[AgentConfig] = None):
        config = custom_config or AGENT_CONFIGS[AgentRole.CODE_EXPERT]
        super().__init__(config)
    
    async def execute_task(self, task_description: str, task_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """代码专家的任务执行"""
        try:
            if task_type in ["code_generation", "CODE_GENERATION"]:
                # 代码生成任务
                result = await self.generate_code(task_description, input_data.get("requirements", ""))
            elif task_type in ["code_review", "CODE_REVIEW"]:
                # 代码审查任务
                code = input_data.get("code", task_description)
                result = await self.analyze_code(code)
            else:
                # 通用代码相关任务
                result = await self.analyze_code(task_description)
            
            return {
                "success": True,
                "result": result,
                "agent_name": self.config.name,
                "task_type": task_type
            }
        except Exception as e:
            logger.error("代码专家任务执行失败", task_type=task_type, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "agent_name": self.config.name,
                "task_type": task_type
            }
    
    async def generate_code(self, description: str, requirements: str = "") -> Dict[str, Any]:
        """生成代码"""
        prompt = f"""作为代码专家，请根据以下需求生成高质量的代码：

需求描述: {description}
详细要求: {requirements}

请提供：
1. 完整的代码实现
2. 代码说明和注释
3. 使用示例
4. 潜在的改进建议
"""
        response = await self.generate_response(prompt)
        return {"code": response, "type": "code_generation"}

    async def analyze_code(self, code: str) -> Dict[str, Any]:
        """分析代码质量和问题"""
        analysis_prompt = f"""请分析以下代码的质量、潜在问题和改进建议：

```
{code}
```

请从以下角度分析：
1. 代码质量和可读性
2. 性能优化机会  
3. 安全问题
4. 最佳实践建议
"""
        response = await self.generate_response(analysis_prompt)
        return {
            "analysis_type": "code_quality",
            "code_length": len(code),
            "analysis": response
        }

class ArchitectAgent(BaseAutoGenAgent):
    """架构师智能体"""
    
    def __init__(self, custom_config: Optional[AgentConfig] = None):
        config = custom_config or AGENT_CONFIGS[AgentRole.ARCHITECT]
        super().__init__(config)
    
    async def execute_task(self, task_description: str, task_type: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """架构师的任务执行"""
        try:
            if task_type in ["architecture", "ARCHITECTURE", "planning", "PLANNING"]:
                result = await self.design_architecture(task_description)
            else:
                # 通用架构设计任务
                result = await self.design_architecture(task_description)
            
            return {
                "success": True,
                "result": result,
                "agent_name": self.config.name,
                "task_type": task_type
            }
        except Exception as e:
            logger.error("架构师任务执行失败", task_type=task_type, error=str(e))
            return {
                "success": False,
                "error": str(e),
                "agent_name": self.config.name,
                "task_type": task_type
            }
    
    async def design_architecture(self, requirements: str) -> Dict[str, Any]:
        """设计系统架构"""
        design_prompt = f"""基于以下需求设计系统架构：

需求：
{requirements}

请提供：
1. 整体架构设计
2. 技术栈选择
3. 关键组件设计
4. 可扩展性考虑
"""
        response = await self.generate_response(design_prompt)
        return {
            "design_type": "system_architecture", 
            "requirements_length": len(requirements),
            "design": response
        }

class DocExpertAgent(BaseAutoGenAgent):
    """文档专家智能体"""
    
    def __init__(self, custom_config: Optional[AgentConfig] = None):
        config = custom_config or AGENT_CONFIGS[AgentRole.DOC_EXPERT]
        super().__init__(config)
    
    async def generate_documentation(self, content: str, doc_type: str) -> Dict[str, Any]:
        """生成技术文档"""
        documentation_prompt = f"""为以下内容生成{doc_type}文档：

内容：
{content}

请生成：
1. 清晰的文档结构
2. 详细的说明和示例
3. 最佳实践建议
4. 常见问题解答
"""
        response = await self.generate_response(documentation_prompt)
        return {
            "doc_type": doc_type,
            "content_length": len(content),
            "documentation": response
        }

class SupervisorAgent(BaseAutoGenAgent):
    """任务调度器智能体"""
    
    def __init__(self, custom_config: Optional[AgentConfig] = None):
        config = custom_config or AGENT_CONFIGS[AgentRole.SUPERVISOR]
        super().__init__(config)
    
    async def coordinate_task(self, task: str, agents: List[str]) -> Dict[str, Any]:
        """协调任务分配"""
        coordination_prompt = f"""作为任务调度器，请为以下任务制定执行计划：

任务：
{task}

可用智能体：
{', '.join(agents)}

请提供：
1. 任务分解
2. 智能体分工
3. 执行顺序
4. 质量控制要点
"""
        response = await self.generate_response(coordination_prompt)
        return {
            "task_type": "coordination",
            "task_length": len(task),
            "plan": response
        }

class KnowledgeRetrievalExpertAgent(BaseAutoGenAgent):
    """知识检索专家智能体"""
    
    def __init__(self, custom_config: Optional[AgentConfig] = None):
        config = custom_config or AGENT_CONFIGS[AgentRole.KNOWLEDGE_RETRIEVAL]
        super().__init__(config)
    
    async def semantic_search(self, query: str, knowledge_base: str = "general") -> Dict[str, Any]:
        """语义搜索"""
        search_prompt = f"""作为知识检索专家，请针对以下查询进行语义搜索和信息整合：

查询：
{query}

知识库：{knowledge_base}

请提供：
1. 最相关的信息和概念
2. 相关上下文和背景
3. 可能的相关问题和扩展
4. 信息来源的可信度评估
"""
        response = await self.generate_response(search_prompt)
        return {
            "query_type": "semantic_search",
            "query": query,
            "knowledge_base": knowledge_base,
            "results": response
        }
    
    async def integrate_knowledge(self, sources: List[str], topic: str) -> Dict[str, Any]:
        """知识整合"""
        integration_prompt = f"""请整合以下信息源，为主题提供全面的知识整合：

主题：
{topic}

信息源：
{chr(10).join(f"{i+1}. {source}" for i, source in enumerate(sources))}

请提供：
1. 核心概念和原理
2. 不同来源的观点对比
3. 综合结论和建议
4. 知识体系的完整性评估
"""
        response = await self.generate_response(integration_prompt)
        return {
            "integration_type": "knowledge_synthesis",
            "topic": topic,
            "source_count": len(sources),
            "synthesis": response
        }
    
    async def generate_answer(self, question: str, context: str = "") -> Dict[str, Any]:
        """基于上下文生成答案"""
        answer_prompt = f"""基于提供的上下文信息，为问题生成准确、相关的答案：

问题：
{question}

上下文：
{context}

请提供：
1. 直接回答问题
2. 支持答案的证据
3. 答案的置信度评估
4. 可能的限制和注意事项
"""
        response = await self.generate_response(answer_prompt)
        return {
            "answer_type": "contextual_response",
            "question": question,
            "context_length": len(context),
            "answer": response
        }

def create_agent_from_config(config: AgentConfig) -> BaseAutoGenAgent:
    """根据配置创建智能体实例"""
    agent_classes = {
        AgentRole.CODE_EXPERT: CodeExpertAgent,
        AgentRole.ARCHITECT: ArchitectAgent,
        AgentRole.DOC_EXPERT: DocExpertAgent,
        AgentRole.SUPERVISOR: SupervisorAgent,
        AgentRole.KNOWLEDGE_RETRIEVAL: KnowledgeRetrievalExpertAgent,
        AgentRole.ASSISTANT: BaseAutoGenAgent,
        AgentRole.CRITIC: BaseAutoGenAgent,
        AgentRole.CODER: BaseAutoGenAgent,
        AgentRole.PLANNER: BaseAutoGenAgent,
        AgentRole.EXECUTOR: BaseAutoGenAgent,
    }
    
    agent_class = agent_classes.get(config.role)
    if not agent_class:
        raise ValueError(f"不支持的智能体角色: {config.role}")
    
    return agent_class(config)

def create_default_agents() -> List[BaseAutoGenAgent]:
    """创建默认的智能体集合"""
    agents = []
    for role, config in AGENT_CONFIGS.items():
        try:
            agent = create_agent_from_config(config)
            agents.append(agent)
        except Exception as e:
            logger.error(
                "创建智能体失败",
                role=role,
                error=str(e)
            )
    
    return agents
