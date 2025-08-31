"""
Supervisor智能体核心实现
基于AutoGen框架实现任务调度、智能体路由和质量评估功能
"""
import asyncio
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass, field
import structlog
from enum import Enum
import json

from .config import AgentConfig, AgentRole, AGENT_CONFIGS
from .agents import BaseAutoGenAgent
from src.core.constants import ConversationConstants

logger = structlog.get_logger(__name__)


class TaskType(str, Enum):
    """任务类型枚举"""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    ANALYSIS = "analysis"
    PLANNING = "planning"
    ARCHITECTURE = "architecture"
    KNOWLEDGE_RETRIEVAL = "knowledge_retrieval"


class TaskPriority(str, Enum):
    """任务优先级枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class AgentStatus(str, Enum):
    """智能体状态枚举"""
    ACTIVE = "active"
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


@dataclass
class TaskComplexity:
    """任务复杂度评估结果"""
    score: float  # 0.0-1.0，1.0最复杂
    factors: Dict[str, float]  # 复杂度因子
    estimated_time: int  # 预估执行时间（秒）
    required_capabilities: List[str]  # 所需能力
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "factors": self.factors,
            "estimated_time": self.estimated_time,
            "required_capabilities": self.required_capabilities
        }


@dataclass
class AgentCapabilityMatch:
    """智能体能力匹配结果"""
    agent_name: str
    agent_role: AgentRole
    match_score: float  # 0.0-1.0，1.0最匹配
    capability_alignment: Dict[str, float]  # 能力匹配详情
    load_factor: float  # 负载因子
    availability: bool  # 是否可用
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role.value if hasattr(self.agent_role, 'value') else str(self.agent_role),
            "match_score": self.match_score,
            "capability_alignment": self.capability_alignment,
            "load_factor": self.load_factor,
            "availability": self.availability
        }


@dataclass
class TaskAssignment:
    """任务分配结果"""
    task_id: str
    assigned_agent: str
    assignment_reason: str
    confidence_level: float
    estimated_completion_time: datetime
    alternative_agents: List[str]
    decision_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "assigned_agent": self.assigned_agent,
            "assignment_reason": self.assignment_reason,
            "confidence_level": self.confidence_level,
            "estimated_completion_time": self.estimated_completion_time.isoformat(),
            "alternative_agents": self.alternative_agents,
            "decision_metadata": self.decision_metadata
        }


@dataclass
class SupervisorDecision:
    """Supervisor决策记录"""
    decision_id: str
    timestamp: datetime
    task_description: str
    assignment: TaskAssignment
    reasoning: str
    confidence: float
    alternatives_considered: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "task_description": self.task_description,
            "assignment": self.assignment.to_dict(),
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "alternatives_considered": self.alternatives_considered
        }


class TaskComplexityAnalyzer:
    """任务复杂度分析器"""
    
    def __init__(self):
        self._complexity_weights = {
            "length": 0.2,  # 任务描述长度
            "technical_terms": 0.3,  # 技术术语密度
            "requirements_count": 0.25,  # 需求数量
            "integration_complexity": 0.25  # 集成复杂度
        }
        
        self._technical_keywords = [
            "api", "database", "authentication", "microservice", "integration",
            "architecture", "performance", "security", "scalability", "algorithm"
        ]
    
    async def analyze_complexity(self, task_description: str, task_type: TaskType) -> TaskComplexity:
        """分析任务复杂度"""
        try:
            logger.info("开始分析任务复杂度", task_type=task_type.value)
            
            factors = {}
            
            # 长度因子分析
            factors["length"] = min(len(task_description) / 1000, 1.0)
            
            # 技术术语密度分析
            words = task_description.lower().split()
            technical_count = sum(1 for word in words if any(keyword in word for keyword in self._technical_keywords))
            factors["technical_terms"] = min(technical_count / len(words) * 10, 1.0) if words else 0.0
            
            # 需求数量分析
            requirement_indicators = ["需要", "要求", "必须", "应该", "实现", "支持"]
            requirements_count = sum(task_description.lower().count(indicator) for indicator in requirement_indicators)
            factors["requirements_count"] = min(requirements_count / 10, 1.0)
            
            # 集成复杂度分析（基于任务类型）
            integration_scores = {
                TaskType.CODE_GENERATION: 0.7,
                TaskType.CODE_REVIEW: 0.5,
                TaskType.DOCUMENTATION: 0.3,
                TaskType.ANALYSIS: 0.6,
                TaskType.PLANNING: 0.8,
                TaskType.ARCHITECTURE: 0.9,
                TaskType.KNOWLEDGE_RETRIEVAL: 0.4
            }
            factors["integration_complexity"] = integration_scores.get(task_type, 0.5)
            
            # 计算总复杂度分数
            total_score = sum(
                factors[factor] * weight 
                for factor, weight in self._complexity_weights.items()
            )
            
            # 预估执行时间（基于复杂度）
            base_time = 300  # 5分钟基础时间
            estimated_time = int(base_time * (1 + total_score * 3))
            
            # 确定所需能力
            required_capabilities = self._determine_required_capabilities(task_type, total_score)
            
            complexity = TaskComplexity(
                score=total_score,
                factors=factors,
                estimated_time=estimated_time,
                required_capabilities=required_capabilities
            )
            
            logger.info(
                "任务复杂度分析完成",
                score=total_score,
                estimated_time=estimated_time,
                factors=factors
            )
            
            return complexity
            
        except Exception as e:
            logger.error("任务复杂度分析失败", error=str(e))
            # 返回默认复杂度
            return TaskComplexity(
                score=0.5,
                factors={"error": 1.0},
                estimated_time=600,
                required_capabilities=["基础处理"]
            )
    
    def _determine_required_capabilities(self, task_type: TaskType, complexity_score: float) -> List[str]:
        """确定所需能力"""
        base_capabilities = {
            TaskType.CODE_GENERATION: ["代码生成", "编程语言", "调试"],
            TaskType.CODE_REVIEW: ["代码审查", "质量评估", "最佳实践"],
            TaskType.DOCUMENTATION: ["文档撰写", "技术写作", "信息架构"],
            TaskType.ANALYSIS: ["数据分析", "逻辑推理", "模式识别"],
            TaskType.PLANNING: ["项目规划", "任务分解", "风险评估"],
            TaskType.ARCHITECTURE: ["系统设计", "架构模式", "技术选型"],
            TaskType.KNOWLEDGE_RETRIEVAL: ["信息检索", "语义搜索", "知识整合"]
        }
        
        capabilities = base_capabilities.get(task_type, ["通用处理"])
        
        # 根据复杂度添加高级能力
        if complexity_score > 0.5:  # 降低阈值
            capabilities.extend(["高级分析", "系统集成", "性能优化"])
        elif complexity_score > 0.3:
            capabilities.extend(["中级分析", "模块化设计"])
            
        return capabilities


class AgentCapabilityMatcher:
    """智能体能力匹配器"""
    
    def __init__(self, available_agents: Dict[str, BaseAutoGenAgent]):
        self.available_agents = available_agents
        self._agent_loads: Dict[str, float] = {name: 0.0 for name in available_agents.keys()}
    
    async def find_best_matches(
        self, 
        complexity: TaskComplexity, 
        task_type: TaskType,
        top_n: int = 3
    ) -> List[AgentCapabilityMatch]:
        """找到最佳匹配的智能体"""
        try:
            logger.info("开始智能体能力匹配", required_capabilities=complexity.required_capabilities)
            
            matches = []
            
            for agent_name, agent in self.available_agents.items():
                logger.info("评估智能体匹配", 
                          agent_key=agent_name, 
                          agent_config_name=agent.config.name,
                          agent_role=agent.config.role)
                
                match = await self._evaluate_agent_match(agent, complexity, task_type)
                if match:
                    logger.info("智能体匹配成功", 
                              agent_key=agent_name,
                              match_agent_name=match.agent_name,
                              match_score=match.match_score)
                    matches.append(match)
                else:
                    logger.warning("智能体匹配失败", agent_key=agent_name)
            
            # 根据匹配分数排序
            matches.sort(key=lambda x: x.match_score, reverse=True)
            
            logger.info(
                "智能体匹配完成",
                total_matches=len(matches),
                top_matches=[m.agent_name for m in matches[:top_n]]
            )
            
            return matches[:top_n]
            
        except Exception as e:
            logger.error("智能体能力匹配失败", error=str(e))
            return []
    
    async def _evaluate_agent_match(
        self, 
        agent: BaseAutoGenAgent, 
        complexity: TaskComplexity,
        task_type: TaskType
    ) -> Optional[AgentCapabilityMatch]:
        """评估单个智能体匹配度"""
        try:
            agent_config = agent.config
            
            # 基础角色匹配
            role_match_scores = {
                TaskType.CODE_GENERATION: {
                    AgentRole.CODE_EXPERT: 1.0,
                    AgentRole.ARCHITECT: 0.6,
                    AgentRole.DOC_EXPERT: 0.2,
                    AgentRole.SUPERVISOR: 0.3,
                    AgentRole.KNOWLEDGE_RETRIEVAL: 0.1
                },
                TaskType.CODE_REVIEW: {
                    AgentRole.CODE_EXPERT: 1.0,
                    AgentRole.ARCHITECT: 0.8,
                    AgentRole.DOC_EXPERT: 0.3,
                    AgentRole.SUPERVISOR: 0.4,
                    AgentRole.KNOWLEDGE_RETRIEVAL: 0.1
                },
                TaskType.DOCUMENTATION: {
                    AgentRole.DOC_EXPERT: 1.0,
                    AgentRole.ARCHITECT: 0.5,
                    AgentRole.CODE_EXPERT: 0.4,
                    AgentRole.SUPERVISOR: 0.3,
                    AgentRole.KNOWLEDGE_RETRIEVAL: 0.6
                },
                TaskType.ANALYSIS: {
                    AgentRole.ARCHITECT: 0.9,
                    AgentRole.CODE_EXPERT: 0.7,
                    AgentRole.DOC_EXPERT: 0.5,
                    AgentRole.KNOWLEDGE_RETRIEVAL: 0.8,
                    AgentRole.SUPERVISOR: 0.6
                },
                TaskType.PLANNING: {
                    AgentRole.SUPERVISOR: 1.0,
                    AgentRole.ARCHITECT: 0.8,
                    AgentRole.CODE_EXPERT: 0.4,
                    AgentRole.DOC_EXPERT: 0.5,
                    AgentRole.KNOWLEDGE_RETRIEVAL: 0.3
                },
                TaskType.ARCHITECTURE: {
                    AgentRole.ARCHITECT: 1.0,
                    AgentRole.SUPERVISOR: 0.6,
                    AgentRole.CODE_EXPERT: 0.5,
                    AgentRole.DOC_EXPERT: 0.4,
                    AgentRole.KNOWLEDGE_RETRIEVAL: 0.2
                },
                TaskType.KNOWLEDGE_RETRIEVAL: {
                    AgentRole.KNOWLEDGE_RETRIEVAL: 1.0,
                    AgentRole.DOC_EXPERT: 0.7,
                    AgentRole.ARCHITECT: 0.4,
                    AgentRole.CODE_EXPERT: 0.3,
                    AgentRole.SUPERVISOR: 0.3
                }
            }
            
            base_score = role_match_scores.get(task_type, {}).get(agent_config.role, 0.0)
            
            # 能力匹配评估
            capability_alignment = {}
            capability_score = 0.0
            
            if complexity.required_capabilities:
                matched_capabilities = 0
                for required_cap in complexity.required_capabilities:
                    # 检查智能体是否具备所需能力
                    match_found = any(
                        required_cap.lower() in agent_cap.lower() 
                        for agent_cap in agent_config.capabilities
                    )
                    
                    capability_score = 1.0 if match_found else 0.0
                    capability_alignment[required_cap] = capability_score
                    
                    if match_found:
                        matched_capabilities += 1
                
                capability_score = matched_capabilities / len(complexity.required_capabilities)
            
            # 负载因子评估
            current_load = self._agent_loads.get(agent_config.name, 0.0)
            load_factor = max(0.0, 1.0 - current_load)
            
            # 可用性检查
            availability = load_factor > 0.1  # 负载超过90%认为不可用
            
            # 综合匹配分数计算
            final_score = (base_score * 0.5 + capability_score * 0.3 + load_factor * 0.2)
            
            return AgentCapabilityMatch(
                agent_name=agent_config.name,  # 使用智能体配置中的名称（如"代码专家"）
                agent_role=agent_config.role,
                match_score=final_score,
                capability_alignment=capability_alignment,
                load_factor=current_load,
                availability=availability
            )
            
        except Exception as e:
            logger.error("智能体匹配评估失败", agent_name=agent.config.name, error=str(e))
            return None
    
    def update_agent_load(self, agent_name: str, load_change: float):
        """更新智能体负载"""
        if agent_name in self._agent_loads:
            self._agent_loads[agent_name] = max(0.0, min(1.0, self._agent_loads[agent_name] + load_change))
    
    def get_agent_loads(self) -> Dict[str, float]:
        """获取所有智能体负载"""
        return self._agent_loads.copy()


class SupervisorAgent(BaseAutoGenAgent):
    """增强的Supervisor智能体实现"""
    
    def __init__(self, available_agents: Optional[Dict[str, BaseAutoGenAgent]] = None, custom_config: Optional[AgentConfig] = None):
        config = custom_config or AGENT_CONFIGS[AgentRole.SUPERVISOR]
        super().__init__(config)
        
        # 初始化组件
        self.available_agents = available_agents or {}
        self.complexity_analyzer = TaskComplexityAnalyzer()
        self.capability_matcher = AgentCapabilityMatcher(self.available_agents)
        
        # 决策历史记录
        self.decision_history: List[SupervisorDecision] = []
        self.task_queue: List[Dict[str, Any]] = []
        
        logger.info("Supervisor智能体初始化完成", available_agents=list(self.available_agents.keys()))
    
    async def analyze_and_assign_task(
        self, 
        task_description: str,
        task_type: TaskType,
        priority: TaskPriority = TaskPriority.MEDIUM,
        constraints: Optional[Dict[str, Any]] = None
    ) -> TaskAssignment:
        """分析任务并分配给最合适的智能体"""
        try:
            decision_id = f"decision_{utc_now().strftime('%Y%m%d_%H%M%S_%f')}"
            
            logger.info(
                "开始任务分析和分配",
                decision_id=decision_id,
                task_type=task_type.value,
                priority=priority.value
            )
            
            # 1. 分析任务复杂度
            complexity = await self.complexity_analyzer.analyze_complexity(task_description, task_type)
            
            # 2. 找到最佳匹配的智能体
            candidate_matches = await self.capability_matcher.find_best_matches(
                complexity, task_type, top_n=3
            )
            
            if not candidate_matches:
                raise ValueError("未找到合适的智能体来处理此任务")
            
            # 3. 选择最佳智能体
            best_match = candidate_matches[0]
            
            # 4. 生成分配决策
            assignment = TaskAssignment(
                task_id=f"task_{decision_id}",
                assigned_agent=best_match.agent_name,
                assignment_reason=await self._generate_assignment_reason(best_match, complexity),
                confidence_level=best_match.match_score,
                estimated_completion_time=utc_now().replace(
                    second=0, microsecond=0
                ) + timedelta(seconds=complexity.estimated_time),
                alternative_agents=[match.agent_name for match in candidate_matches[1:3]],
                decision_metadata={
                    "task_description": task_description,
                    "complexity": complexity.to_dict(),
                    "match_details": best_match.to_dict(),
                    "alternatives": [match.to_dict() for match in candidate_matches[1:3]]
                }
            )
            
            # 5. 记录决策
            decision = SupervisorDecision(
                decision_id=decision_id,
                timestamp=utc_now(),
                task_description=task_description,
                assignment=assignment,
                reasoning=assignment.assignment_reason,
                confidence=assignment.confidence_level,
                alternatives_considered=[match.to_dict() for match in candidate_matches[1:]]
            )
            
            self.decision_history.append(decision)
            
            # 6. 更新智能体负载
            self.capability_matcher.update_agent_load(
                best_match.agent_name, 
                complexity.score * 0.3  # 根据复杂度增加负载
            )
            
            logger.info(
                "任务分配完成",
                decision_id=decision_id,
                assigned_agent=assignment.assigned_agent,
                confidence=assignment.confidence_level
            )
            
            return assignment
            
        except Exception as e:
            logger.error("任务分析和分配失败", error=str(e))
            raise
    
    async def _generate_assignment_reason(
        self, 
        best_match: AgentCapabilityMatch,
        complexity: TaskComplexity
    ) -> str:
        """生成分配理由"""
        reason_parts = []
        
        # 基础匹配理由
        reason_parts.append(f"智能体 {best_match.agent_name} 在此任务类型上具有 {best_match.match_score:.1%} 的匹配度")
        
        # 能力匹配详情
        matched_caps = [cap for cap, score in best_match.capability_alignment.items() if score > 0.5]
        if matched_caps:
            reason_parts.append(f"具备所需能力: {', '.join(matched_caps)}")
        
        # 负载情况
        if best_match.load_factor < 0.3:
            reason_parts.append(f"当前负载较低 ({best_match.load_factor:.1%})")
        elif best_match.load_factor > 0.7:
            reason_parts.append(f"当前负载较高但仍可接受 ({best_match.load_factor:.1%})")
        
        # 复杂度适配性
        if complexity.score > 0.7:
            reason_parts.append("任务复杂度高，需要专业处理")
        elif complexity.score < 0.3:
            reason_parts.append("任务相对简单，可快速完成")
        
        return "; ".join(reason_parts)
    
    async def get_supervisor_status(self) -> Dict[str, Any]:
        """获取Supervisor状态"""
        agent_loads = self.capability_matcher.get_agent_loads()
        
        return {
            "supervisor_name": self.config.name,
            "status": "active",
            "available_agents": list(self.available_agents.keys()),
            "agent_loads": agent_loads,
            "decision_history_count": len(self.decision_history),
            "task_queue_length": len(self.task_queue),
            "performance_metrics": {
                "average_confidence": sum(d.confidence for d in self.decision_history[-10:]) / min(10, len(self.decision_history)) if self.decision_history else 0.0,
                "total_decisions": len(self.decision_history)
            }
        }
    
    async def get_decision_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取决策历史"""
        recent_decisions = self.decision_history[-limit:] if limit > 0 else self.decision_history
        return [decision.to_dict() for decision in recent_decisions]
    
    def add_agent(self, agent_name: str, agent: BaseAutoGenAgent):
        """添加可用智能体"""
        self.available_agents[agent_name] = agent
        self.capability_matcher = AgentCapabilityMatcher(self.available_agents)
        logger.info("添加智能体到Supervisor", agent_name=agent_name)
    
    def remove_agent(self, agent_name: str):
        """移除智能体"""
        if agent_name in self.available_agents:
            del self.available_agents[agent_name]
            self.capability_matcher = AgentCapabilityMatcher(self.available_agents)
            logger.info("从Supervisor移除智能体", agent_name=agent_name)