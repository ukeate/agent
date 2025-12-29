"""
AutoGen智能体配置管理
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from src.core.constants import ConversationConstants

class AgentRole(str, Enum):
    """智能体角色枚举"""
    CODE_EXPERT = "code_expert"
    ARCHITECT = "architect"
    DOC_EXPERT = "doc_expert"
    SUPERVISOR = "supervisor"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"
    ASSISTANT = "assistant"
    CRITIC = "critic"
    CODER = "coder"
    PLANNER = "planner"
    EXECUTOR = "executor"

class AgentConfig(BaseModel):
    """智能体配置模型"""
    name: str = Field(..., description="智能体名称")
    role: AgentRole = Field(..., description="智能体角色")
    model: str = Field(default="gpt-4o-mini", description="使用的模型")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="温度参数")
    max_tokens: int = Field(default=2000, gt=0, description="最大token数")
    system_prompt: str = Field(..., description="系统提示词")
    tools: List[str] = Field(default_factory=list, description="可用工具列表")
    capabilities: List[str] = Field(default_factory=list, description="智能体能力")
    
    model_config = ConfigDict(use_enum_values=True)

class ConversationConfig(BaseModel):
    """对话配置模型"""
    max_rounds: int = Field(default=ConversationConstants.DEFAULT_MAX_ROUNDS, gt=0, description="最大对话轮数")
    timeout_seconds: int = Field(default=ConversationConstants.DEFAULT_TIMEOUT_SECONDS, gt=0, description="超时时间(秒)")
    auto_reply: bool = Field(default=ConversationConstants.DEFAULT_AUTO_REPLY, description="是否自动回复")
    speaker_selection_method: str = Field(default=ConversationConstants.DEFAULT_SPEAKER_SELECTION_METHOD, description="发言者选择方法")
    allow_repeat_speaker: bool = Field(default=ConversationConstants.DEFAULT_ALLOW_REPEAT_SPEAKER, description="是否允许连续发言")

# 预定义智能体配置
AGENT_CONFIGS: Dict[AgentRole, AgentConfig] = {
    AgentRole.CODE_EXPERT: AgentConfig(
        name="代码专家",
        role=AgentRole.CODE_EXPERT,
        system_prompt="""你是一位专业的软件开发专家，擅长：
- 代码编写、重构和优化
- 代码审查和质量保证
- 调试和问题排查
- 最佳实践和设计模式应用

请用简洁、技术准确的语言回答，专注于实际的代码实现方案。""",
        capabilities=["代码生成", "代码审查", "性能优化", "问题调试"],
        tools=["code_analysis", "performance_profiler", "linter"]
    ),
    
    AgentRole.ARCHITECT: AgentConfig(
        name="架构师",
        role=AgentRole.ARCHITECT,
        system_prompt="""你是一位资深的软件架构师，专长于：
- 系统架构设计和技术选型
- 可扩展性和性能架构
- 微服务和分布式系统设计
- 架构文档和技术决策

请从架构角度分析问题，提供高层次的设计方案和技术建议。""",
        capabilities=["架构设计", "技术选型", "性能架构", "系统建模"],
        tools=["architecture_analyzer", "performance_calculator", "dependency_mapper"]
    ),
    
    AgentRole.DOC_EXPERT: AgentConfig(
        name="文档专家",
        role=AgentRole.DOC_EXPERT,
        system_prompt="""你是一位专业的技术文档专家，擅长：
- 技术文档撰写和优化
- API文档和用户手册
- 知识管理和信息组织
- 文档结构和可读性优化

请提供清晰、结构化的文档建议，确保信息准确且易于理解。""",
        capabilities=["文档撰写", "API文档", "知识管理", "信息架构"],
        tools=["doc_generator", "content_analyzer", "readability_checker"]
    ),
    
    AgentRole.SUPERVISOR: AgentConfig(
        name="任务调度器",
        role=AgentRole.SUPERVISOR,
        system_prompt="""你是一位专业的任务调度和协调专家，负责：
- 任务分解和分配
- 智能体协作调度
- 质量控制和进度管理
- 团队协调和沟通

请统筹规划任务，协调各智能体高效完成工作目标。""",
        capabilities=["任务调度", "团队协调", "质量控制", "进度管理"],
        tools=["task_scheduler", "quality_checker", "progress_tracker"]
    ),

    AgentRole.KNOWLEDGE_RETRIEVAL: AgentConfig(
        name="知识检索专家",
        role=AgentRole.KNOWLEDGE_RETRIEVAL,
        system_prompt="""你是一位专业的知识检索和整合专家，擅长：
- 语义搜索和信息检索
- 知识库构建和管理
- 信息整合和答案生成
- RAG系统优化和调试

请提供准确、相关的信息检索服务，确保答案基于可靠的知识源。""",
        capabilities=["语义搜索", "知识整合", "信息检索", "答案生成"],
        tools=["vector_search", "knowledge_graph", "embedding_generator", "relevance_scorer"]
    )
}
