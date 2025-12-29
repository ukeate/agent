"""
智能体路由和选择系统
实现基于任务类型的路由逻辑和动态路由决策
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from dataclasses import dataclass
from enum import Enum
from .supervisor_agent import (
    TaskType, TaskPriority, AgentStatus, 
    TaskComplexity, AgentCapabilityMatch, 
    SupervisorAgent
)
from .agents import BaseAutoGenAgent
from .config import AgentRole

from src.core.logging import get_logger
logger = get_logger(__name__)

class RoutingStrategy(str, Enum):
    """路由策略枚举"""
    ROUND_ROBIN = "round_robin"
    CAPABILITY_BASED = "capability_based"
    LOAD_BALANCED = "load_balanced"
    PRIORITY_FIRST = "priority_first"
    HYBRID = "hybrid"

@dataclass
class RoutingConfig:
    """路由配置"""
    strategy: RoutingStrategy = RoutingStrategy.HYBRID
    load_threshold: float = 0.8  # 负载阈值
    capability_weight: float = 0.5  # 能力权重
    load_weight: float = 0.3  # 负载权重
    availability_weight: float = 0.2  # 可用性权重
    enable_fallback: bool = True  # 启用回退机制
    max_retries: int = 3  # 最大重试次数

@dataclass
class RoutingResult:
    """路由结果"""
    selected_agent: str
    strategy_used: RoutingStrategy
    selection_confidence: float
    routing_time_ms: int
    alternatives: List[str]
    routing_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_agent": self.selected_agent,
            "strategy_used": self.strategy_used.value,
            "selection_confidence": self.selection_confidence,
            "routing_time_ms": self.routing_time_ms,
            "alternatives": self.alternatives,
            "routing_metadata": self.routing_metadata
        }

class AgentRouter:
    """智能体路由器核心实现"""
    
    def __init__(self, config: Optional[RoutingConfig] = None):
        self.config = config or RoutingConfig()
        self.routing_history: List[RoutingResult] = []
        self._round_robin_index = 0
        
        logger.info("智能体路由器初始化完成", strategy=self.config.strategy.value)
    
    async def route_task(
        self,
        task_description: str,
        task_type: TaskType,
        priority: TaskPriority,
        available_matches: List[AgentCapabilityMatch],
        constraints: Optional[Dict[str, Any]] = None
    ) -> RoutingResult:
        """根据配置的策略路由任务到最合适的智能体"""
        start_time = utc_now()
        
        try:
            logger.info(
                "开始任务路由",
                task_type=task_type.value,
                priority=priority.value,
                strategy=self.config.strategy.value,
                candidates=len(available_matches)
            )
            
            if not available_matches:
                raise ValueError("没有可用的智能体候选")
            
            # 根据策略选择智能体
            selected_agent, confidence, metadata = await self._execute_routing_strategy(
                available_matches, task_type, priority, constraints or {}
            )
            
            # 计算路由时间
            routing_time = int((utc_now() - start_time).total_seconds() * 1000)
            
            # 准备替代选项
            alternatives = [
                match.agent_name for match in available_matches 
                if match.agent_name != selected_agent
            ][:3]  # 最多3个替代选项
            
            # 创建路由结果
            result = RoutingResult(
                selected_agent=selected_agent,
                strategy_used=self.config.strategy,
                selection_confidence=confidence,
                routing_time_ms=routing_time,
                alternatives=alternatives,
                routing_metadata=metadata
            )
            
            # 记录路由历史
            self.routing_history.append(result)
            
            logger.info(
                "任务路由完成",
                selected_agent=selected_agent,
                confidence=confidence,
                routing_time_ms=routing_time
            )
            
            return result
            
        except Exception as e:
            logger.error("任务路由失败", error=str(e))
            raise
    
    async def _execute_routing_strategy(
        self,
        matches: List[AgentCapabilityMatch],
        task_type: TaskType,
        priority: TaskPriority,
        constraints: Dict[str, Any]
    ) -> Tuple[str, float, Dict[str, Any]]:
        """执行具体的路由策略"""
        
        if self.config.strategy == RoutingStrategy.ROUND_ROBIN:
            return await self._round_robin_routing(matches)
        
        elif self.config.strategy == RoutingStrategy.CAPABILITY_BASED:
            return await self._capability_based_routing(matches)
        
        elif self.config.strategy == RoutingStrategy.LOAD_BALANCED:
            return await self._load_balanced_routing(matches)
        
        elif self.config.strategy == RoutingStrategy.PRIORITY_FIRST:
            return await self._priority_first_routing(matches, priority)
        
        elif self.config.strategy == RoutingStrategy.HYBRID:
            return await self._hybrid_routing(matches, task_type, priority, constraints)
        
        else:
            # 默认使用能力匹配
            return await self._capability_based_routing(matches)
    
    async def _round_robin_routing(
        self, 
        matches: List[AgentCapabilityMatch]
    ) -> Tuple[str, float, Dict[str, Any]]:
        """轮询路由策略"""
        if not matches:
            raise ValueError("没有可用的匹配智能体")
        
        # 过滤可用的智能体
        available_matches = [match for match in matches if match.availability]
        if not available_matches:
            available_matches = matches  # 如果没有可用的，使用全部
        
        # 轮询选择
        selected_match = available_matches[self._round_robin_index % len(available_matches)]
        self._round_robin_index += 1
        
        return (
            selected_match.agent_name,
            0.5,  # 轮询策略置信度固定为中等
            {
                "strategy_details": "round_robin_selection",
                "selected_index": (self._round_robin_index - 1) % len(available_matches),
                "total_candidates": len(available_matches)
            }
        )
    
    async def _capability_based_routing(
        self, 
        matches: List[AgentCapabilityMatch]
    ) -> Tuple[str, float, Dict[str, Any]]:
        """基于能力的路由策略"""
        # 按匹配分数排序（已经排序过了）
        best_match = matches[0]
        
        return (
            best_match.agent_name,
            best_match.match_score,
            {
                "strategy_details": "capability_match_selection",
                "match_score": best_match.match_score,
                "capability_alignment": best_match.capability_alignment
            }
        )
    
    async def _load_balanced_routing(
        self, 
        matches: List[AgentCapabilityMatch]
    ) -> Tuple[str, float, Dict[str, Any]]:
        """负载平衡路由策略"""
        # 按负载因子排序（负载越低越优先）
        sorted_matches = sorted(matches, key=lambda x: x.load_factor)
        best_match = sorted_matches[0]
        
        # 置信度基于负载情况
        confidence = max(0.1, 1.0 - best_match.load_factor)
        
        return (
            best_match.agent_name,
            confidence,
            {
                "strategy_details": "load_balanced_selection",
                "selected_load": best_match.load_factor,
                "all_loads": {match.agent_name: match.load_factor for match in matches}
            }
        )
    
    async def _priority_first_routing(
        self, 
        matches: List[AgentCapabilityMatch],
        priority: TaskPriority
    ) -> Tuple[str, float, Dict[str, Any]]:
        """优先级优先路由策略"""
        # 对于高优先级任务，选择最佳匹配且负载不超过阈值的智能体
        if priority in [TaskPriority.HIGH, TaskPriority.URGENT]:
            suitable_matches = [
                match for match in matches 
                if match.load_factor < self.config.load_threshold and match.availability
            ]
            
            if suitable_matches:
                best_match = suitable_matches[0]  # 已按匹配分数排序
            else:
                best_match = matches[0]  # 如果没有合适的，选择最佳匹配
        else:
            # 低优先级任务可以分配给负载较高的智能体
            best_match = matches[0]
        
        confidence = best_match.match_score * (0.8 if priority in [TaskPriority.HIGH, TaskPriority.URGENT] else 0.6)
        
        return (
            best_match.agent_name,
            confidence,
            {
                "strategy_details": "priority_first_selection",
                "task_priority": priority.value,
                "load_threshold_applied": priority in [TaskPriority.HIGH, TaskPriority.URGENT]
            }
        )
    
    async def _hybrid_routing(
        self,
        matches: List[AgentCapabilityMatch],
        task_type: TaskType,
        priority: TaskPriority,
        constraints: Dict[str, Any]
    ) -> Tuple[str, float, Dict[str, Any]]:
        """混合路由策略（综合考虑多个因素）"""
        
        # 计算每个智能体的综合分数
        scored_matches = []
        
        for match in matches:
            # 能力分数
            capability_score = match.match_score * self.config.capability_weight
            
            # 负载分数（负载越低分数越高）
            load_score = (1.0 - match.load_factor) * self.config.load_weight
            
            # 可用性分数
            availability_score = 1.0 if match.availability else 0.0
            availability_score *= self.config.availability_weight
            
            # 优先级调整
            priority_multiplier = self._get_priority_multiplier(priority)
            
            # 综合分数
            total_score = (capability_score + load_score + availability_score) * priority_multiplier
            
            scored_matches.append((match, total_score))
        
        # 按综合分数排序
        scored_matches.sort(key=lambda x: x[1], reverse=True)
        best_match, best_score = scored_matches[0]
        
        # 置信度基于综合分数
        max_possible_score = (
            self.config.capability_weight + 
            self.config.load_weight + 
            self.config.availability_weight
        ) * self._get_priority_multiplier(priority)
        
        confidence = min(1.0, best_score / max_possible_score)
        
        return (
            best_match.agent_name,
            confidence,
            {
                "strategy_details": "hybrid_selection",
                "capability_score": best_match.match_score * self.config.capability_weight,
                "load_score": (1.0 - best_match.load_factor) * self.config.load_weight,
                "availability_score": (1.0 if best_match.availability else 0.0) * self.config.availability_weight,
                "total_score": best_score,
                "priority_multiplier": self._get_priority_multiplier(priority),
                "weights": {
                    "capability": self.config.capability_weight,
                    "load": self.config.load_weight,
                    "availability": self.config.availability_weight
                }
            }
        )
    
    def _get_priority_multiplier(self, priority: TaskPriority) -> float:
        """获取优先级调整因子"""
        multipliers = {
            TaskPriority.LOW: 0.8,
            TaskPriority.MEDIUM: 1.0,
            TaskPriority.HIGH: 1.2,
            TaskPriority.URGENT: 1.5
        }
        return multipliers.get(priority, 1.0)
    
    async def get_routing_statistics(self) -> Dict[str, Any]:
        """获取路由统计信息"""
        if not self.routing_history:
            return {
                "total_routings": 0,
                "average_confidence": 0.0,
                "average_routing_time": 0.0,
                "strategy_distribution": {},
                "agent_selection_frequency": {}
            }
        
        # 统计数据计算
        total_routings = len(self.routing_history)
        average_confidence = sum(r.selection_confidence for r in self.routing_history) / total_routings
        average_routing_time = sum(r.routing_time_ms for r in self.routing_history) / total_routings
        
        # 策略分布统计
        strategy_counts = {}
        for result in self.routing_history:
            strategy = result.strategy_used.value
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        strategy_distribution = {
            strategy: count / total_routings 
            for strategy, count in strategy_counts.items()
        }
        
        # 智能体选择频率
        agent_counts = {}
        for result in self.routing_history:
            agent = result.selected_agent
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        agent_selection_frequency = {
            agent: count / total_routings 
            for agent, count in agent_counts.items()
        }
        
        return {
            "total_routings": total_routings,
            "average_confidence": average_confidence,
            "average_routing_time": average_routing_time,
            "strategy_distribution": strategy_distribution,
            "agent_selection_frequency": agent_selection_frequency,
            "recent_routing_times": [r.routing_time_ms for r in self.routing_history[-10:]]
        }
    
    def update_routing_config(self, new_config: RoutingConfig):
        """更新路由配置"""
        old_strategy = self.config.strategy
        self.config = new_config
        
        logger.info(
            "路由配置已更新",
            old_strategy=old_strategy.value,
            new_strategy=new_config.strategy.value
        )
    
    def clear_history(self):
        """清空路由历史"""
        self.routing_history.clear()
        self._round_robin_index = 0
        logger.info("路由历史已清空")

class LoadBalancer:
    """负载平衡器"""
    
    def __init__(self, agents: Dict[str, BaseAutoGenAgent]):
        self.agents = agents
        self.agent_loads: Dict[str, float] = {name: 0.0 for name in agents.keys()}
        self.task_counts: Dict[str, int] = {name: 0 for name in agents.keys()}
        self.last_update = utc_now()
    
    def get_agent_loads(self) -> Dict[str, float]:
        """获取智能体负载情况"""
        return self.agent_loads.copy()
    
    def update_agent_load(self, agent_name: str, load_change: float):
        """更新智能体负载"""
        if agent_name in self.agent_loads:
            new_load = self.agent_loads[agent_name] + load_change
            self.agent_loads[agent_name] = max(0.0, min(1.0, new_load))
            
            if load_change > 0:
                self.task_counts[agent_name] = self.task_counts.get(agent_name, 0) + 1
            
            self.last_update = utc_now()
            
            logger.debug(
                "智能体负载已更新",
                agent_name=agent_name,
                new_load=self.agent_loads[agent_name],
                load_change=load_change
            )
    
    def get_load_balanced_ranking(self, matches: List[AgentCapabilityMatch]) -> List[AgentCapabilityMatch]:
        """基于负载平衡重新排序智能体匹配结果"""
        # 结合匹配分数和负载因子重新计算排序
        def calculate_balanced_score(match: AgentCapabilityMatch) -> float:
            # 负载越低，分数越高
            load_factor = 1.0 - self.agent_loads.get(match.agent_name, 0.0)
            # 综合匹配分数和负载因子
            return match.match_score * 0.7 + load_factor * 0.3
        
        return sorted(matches, key=calculate_balanced_score, reverse=True)
    
    async def get_load_statistics(self) -> Dict[str, Any]:
        """获取负载统计信息"""
        total_tasks = sum(self.task_counts.values())
        average_load = sum(self.agent_loads.values()) / len(self.agent_loads) if self.agent_loads else 0.0
        
        # 找出最忙和最闲的智能体
        busiest_agent = max(self.agent_loads.items(), key=lambda x: x[1]) if self.agent_loads else ("", 0.0)
        least_busy_agent = min(self.agent_loads.items(), key=lambda x: x[1]) if self.agent_loads else ("", 0.0)
        
        return {
            "total_tasks_assigned": total_tasks,
            "average_load": average_load,
            "agent_loads": self.agent_loads,
            "task_counts": self.task_counts,
            "busiest_agent": {"name": busiest_agent[0], "load": busiest_agent[1]},
            "least_busy_agent": {"name": least_busy_agent[0], "load": least_busy_agent[1]},
            "last_update": self.last_update.isoformat(),
            "load_distribution": {
                "low_load": sum(1 for load in self.agent_loads.values() if load < 0.3),
                "medium_load": sum(1 for load in self.agent_loads.values() if 0.3 <= load < 0.7),
                "high_load": sum(1 for load in self.agent_loads.values() if load >= 0.7)
            }
        }
