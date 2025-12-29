"""智能任务分配器实现"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from .models import Task, TaskStatus, TaskPriority
from src.ai.service_discovery.core import AgentStatus
from src.core.utils.timezone_utils import utc_now

class IntelligentAssigner:
    """智能分配器"""
    
    def __init__(self, service_registry=None, load_balancer=None):
        self.service_registry = service_registry
        self.load_balancer = load_balancer
        self.logger = get_logger(__name__)
        
        # 分配策略
        self.assignment_strategies = {
            "capability_based": self._capability_based_assignment,
            "load_balanced": self._load_balanced_assignment,
            "resource_optimized": self._resource_optimized_assignment,
            "deadline_aware": self._deadline_aware_assignment,
            "cost_optimized": self._cost_optimized_assignment,
            "locality_aware": self._locality_aware_assignment,
            "affinity_based": self._affinity_based_assignment,
            "priority_weighted": self._priority_weighted_assignment
        }
        
        # 分配历史和统计
        self.assignment_history: Dict[str, List[str]] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        self.agent_load: Dict[str, int] = {}
        self.agent_failures: Dict[str, List[datetime]] = {}
        self.task_execution_times: Dict[str, List[float]] = {}
    
    async def assign_task(
        self, 
        task: Task, 
        strategy: str = "capability_based",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """分配任务给智能体"""
        
        try:
            if strategy not in self.assignment_strategies:
                self.logger.warning(f"Unknown assignment strategy: {strategy}, using capability_based")
                strategy = "capability_based"
            
            # 查找候选智能体
            candidates = await self._find_candidate_agents(task, constraints)
            
            if not candidates:
                self.logger.warning(f"No suitable agents found for task {task.task_id}")
                return None
            
            # 执行分配策略
            selected_agent = await self.assignment_strategies[strategy](task, candidates)
            
            if selected_agent:
                agent_id = selected_agent.get("agent_id") if isinstance(selected_agent, dict) else selected_agent.agent_id
                
                # 记录分配历史
                if task.task_id not in self.assignment_history:
                    self.assignment_history[task.task_id] = []
                self.assignment_history[task.task_id].append(agent_id)
                
                # 更新智能体负载
                self.agent_load[agent_id] = self.agent_load.get(agent_id, 0) + 1
                
                # 更新任务状态
                task.assigned_to = agent_id
                task.status = TaskStatus.ASSIGNED
                
                self.logger.info(f"Task {task.task_id} assigned to agent {agent_id}")
                return agent_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to assign task {task.task_id}: {e}")
            return None
    
    async def reassign_task(self, task: Task, reason: str = "failure") -> Optional[str]:
        """重新分配任务"""
        
        try:
            # 获取之前分配的智能体
            previous_agents = self.assignment_history.get(task.task_id, [])
            
            # 增加重试计数
            task.retry_count += 1
            
            if task.retry_count > task.max_retries:
                self.logger.error(f"Task {task.task_id} exceeded max retries")
                task.status = TaskStatus.FAILED
                return None
            
            # 设置约束，排除之前失败的智能体
            constraints = {"excluded_agents": previous_agents}
            
            # 使用更可靠的策略重新分配
            if reason == "failure":
                # 记录失败
                if previous_agents:
                    failed_agent = previous_agents[-1]
                    if failed_agent not in self.agent_failures:
                        self.agent_failures[failed_agent] = []
                    self.agent_failures[failed_agent].append(utc_now())
                
                # 使用load_balanced策略避免过载的智能体
                return await self.assign_task(task, "load_balanced", constraints)
            elif reason == "timeout":
                # 使用deadline_aware策略找更快的智能体
                return await self.assign_task(task, "deadline_aware", constraints)
            else:
                # 默认使用capability_based策略
                return await self.assign_task(task, "capability_based", constraints)
            
        except Exception as e:
            self.logger.error(f"Failed to reassign task {task.task_id}: {e}")
            return None
    
    async def _find_candidate_agents(
        self, 
        task: Task, 
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[Any]:
        """查找候选智能体"""
        
        if not self.service_registry:
            raise RuntimeError("未接入服务注册中心，无法发现可用智能体")
        
        # 基本能力匹配
        capable_agents = await self.service_registry.discover_agents(
            capability=task.task_type,
            status=AgentStatus.ACTIVE
        )
        
        if not capable_agents:
            return []
        
        # 应用约束条件
        if constraints:
            capable_agents = await self._apply_constraints(capable_agents, constraints)
        
        # 资源需求过滤
        if task.resource_requirements:
            capable_agents = await self._filter_by_resources(capable_agents, task.resource_requirements)
        
        # 地理位置约束（如果有）
        if "location" in task.requirements:
            capable_agents = await self._filter_by_location(capable_agents, task.requirements["location"])
        
        # 过滤掉最近失败过多的智能体
        capable_agents = await self._filter_unreliable_agents(capable_agents)
        
        return capable_agents
    
    async def _apply_constraints(
        self, 
        agents: List[Any], 
        constraints: Dict[str, Any]
    ) -> List[Any]:
        """应用约束条件"""
        
        filtered_agents = []
        
        for agent in agents:
            meets_constraints = True
            
            # 获取智能体属性（兼容字典和对象）
            agent_dict = agent if isinstance(agent, dict) else agent.__dict__
            
            # 检查标签约束
            if "required_tags" in constraints:
                required_tags = set(constraints["required_tags"])
                agent_tags = set(agent_dict.get("tags", []))
                if not required_tags.issubset(agent_tags):
                    meets_constraints = False
            
            # 检查排除约束
            if "excluded_agents" in constraints:
                agent_id = agent_dict.get("agent_id")
                if agent_id in constraints["excluded_agents"]:
                    meets_constraints = False
            
            # 检查版本约束
            if "min_version" in constraints:
                agent_version = agent_dict.get("version", "0.0.0")
                if self._compare_versions(agent_version, constraints["min_version"]) < 0:
                    meets_constraints = False
            
            # 检查性能约束
            if "min_accuracy" in constraints:
                capabilities = agent_dict.get("capabilities", [])
                max_accuracy = 0
                for cap in capabilities:
                    metrics = cap.get("performance_metrics", {})
                    accuracy = metrics.get("accuracy", 0)
                    max_accuracy = max(max_accuracy, accuracy)
                if max_accuracy < constraints["min_accuracy"]:
                    meets_constraints = False
            
            if meets_constraints:
                filtered_agents.append(agent)
        
        return filtered_agents
    
    async def _filter_by_resources(
        self, 
        agents: List[Any], 
        requirements: Dict[str, Any]
    ) -> List[Any]:
        """根据资源需求过滤智能体"""
        
        suitable_agents = []
        
        for agent in agents:
            # 获取智能体资源（兼容字典和对象）
            if isinstance(agent, dict):
                resources = agent.get("resources", {})
            else:
                resources = getattr(agent, "resources", {})
            
            meets_requirements = True
            
            # 检查CPU需求
            if "cpu" in requirements:
                available_cpu = 1.0 - resources.get("cpu_usage", 0.0)
                if available_cpu < requirements["cpu"]:
                    meets_requirements = False
            
            # 检查内存需求
            if "memory" in requirements:
                available_memory = resources.get("memory_total", 0) - resources.get("memory_used", 0)
                if available_memory < requirements["memory"]:
                    meets_requirements = False
            
            # 检查磁盘需求
            if "disk_space" in requirements:
                available_disk = resources.get("disk_total", 0) - resources.get("disk_used", 0)
                if available_disk < requirements["disk_space"]:
                    meets_requirements = False
            
            # 检查GPU需求
            if "gpu" in requirements:
                if requirements["gpu"] and not resources.get("gpu_available", False):
                    meets_requirements = False
            
            if meets_requirements:
                suitable_agents.append(agent)
        
        return suitable_agents
    
    async def _filter_by_location(
        self, 
        agents: List[Any], 
        location_requirements: Dict[str, Any]
    ) -> List[Any]:
        """根据地理位置过滤智能体"""
        
        suitable_agents = []
        
        for agent in agents:
            # 获取智能体位置（兼容字典和对象）
            if isinstance(agent, dict):
                agent_location = agent.get("resources", {}).get("location", {})
                if not agent_location:
                    agent_location = agent.get("location", {})
            else:
                resources = getattr(agent, "resources", {})
                agent_location = resources.get("location", {})
            
            meets_requirements = True
            
            # 检查区域约束
            if "region" in location_requirements:
                if agent_location.get("region") != location_requirements["region"]:
                    meets_requirements = False
            
            # 检查数据中心约束
            if "datacenter" in location_requirements:
                if agent_location.get("datacenter") != location_requirements["datacenter"]:
                    meets_requirements = False
            
            if meets_requirements:
                suitable_agents.append(agent)
        
        return suitable_agents
    
    async def _filter_unreliable_agents(self, agents: List[Any]) -> List[Any]:
        """过滤不可靠的智能体"""
        
        reliable_agents = []
        failure_threshold = 3  # 最近3次失败
        time_window = timedelta(minutes=30)  # 30分钟内
        
        for agent in agents:
            # 获取智能体ID（兼容字典和对象）
            agent_id = agent.get("agent_id") if isinstance(agent, dict) else agent.agent_id
            
            # 检查最近的失败记录
            if agent_id in self.agent_failures:
                recent_failures = [
                    f for f in self.agent_failures[agent_id]
                    if utc_now() - f < time_window
                ]
                
                if len(recent_failures) >= failure_threshold:
                    self.logger.warning(f"Agent {agent_id} has too many recent failures, excluding")
                    continue
            
            reliable_agents.append(agent)
        
        return reliable_agents
    
    async def _capability_based_assignment(
        self, 
        task: Task, 
        candidates: List[Any]
    ) -> Optional[Any]:
        """基于能力的分配策略"""
        
        # 为每个候选智能体计算能力匹配分数
        scored_candidates = []
        
        for agent in candidates:
            score = await self._calculate_capability_score(agent, task)
            scored_candidates.append((agent, score))
        
        # 选择分数最高的智能体
        if scored_candidates:
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            return scored_candidates[0][0]
        
        return None
    
    async def _load_balanced_assignment(
        self, 
        task: Task, 
        candidates: List[Any]
    ) -> Optional[Any]:
        """负载均衡分配策略"""
        
        if self.load_balancer:
            # 使用负载均衡器选择智能体
            return await self.load_balancer.select_agent(
                capability=task.task_type,
                strategy="least_connections",
                requirements=task.resource_requirements
            )
        
        # 如果没有负载均衡器，选择负载最低的智能体
        best_agent = None
        min_load = float('inf')
        
        for agent in candidates:
            agent_id = agent.get("agent_id") if isinstance(agent, dict) else agent.agent_id
            current_load = self.agent_load.get(agent_id, 0)
            
            if current_load < min_load:
                min_load = current_load
                best_agent = agent
        
        return best_agent
    
    async def _resource_optimized_assignment(
        self, 
        task: Task, 
        candidates: List[Any]
    ) -> Optional[Any]:
        """资源优化分配策略"""
        
        best_agent = None
        best_efficiency = 0.0
        
        for agent in candidates:
            # 计算资源效率
            efficiency = await self._calculate_resource_efficiency(agent, task)
            
            if efficiency > best_efficiency:
                best_efficiency = efficiency
                best_agent = agent
        
        return best_agent
    
    async def _deadline_aware_assignment(
        self, 
        task: Task, 
        candidates: List[Any]
    ) -> Optional[Any]:
        """截止时间感知分配策略"""
        
        deadline = task.requirements.get("deadline")
        if not deadline:
            # 没有截止时间，使用默认策略
            return await self._capability_based_assignment(task, candidates)
        
        deadline_dt = datetime.fromisoformat(deadline) if isinstance(deadline, str) else deadline
        time_remaining = (deadline_dt - utc_now()).total_seconds()
        
        best_agent = None
        best_completion_time = float('inf')
        
        for agent in candidates:
            # 估算完成时间
            estimated_time = await self._estimate_completion_time(agent, task)
            
            # 考虑当前负载
            agent_id = agent.get("agent_id") if isinstance(agent, dict) else agent.agent_id
            current_load = self.agent_load.get(agent_id, 0)
            adjusted_time = estimated_time * (1 + current_load * 0.2)
            
            if adjusted_time < best_completion_time and adjusted_time <= time_remaining:
                best_completion_time = adjusted_time
                best_agent = agent
        
        return best_agent
    
    async def _cost_optimized_assignment(
        self, 
        task: Task, 
        candidates: List[Any]
    ) -> Optional[Any]:
        """成本优化分配策略"""
        
        best_agent = None
        best_cost_efficiency = 0.0
        
        for agent in candidates:
            # 计算成本效率
            cost_efficiency = await self._calculate_cost_efficiency(agent, task)
            
            if cost_efficiency > best_cost_efficiency:
                best_cost_efficiency = cost_efficiency
                best_agent = agent
        
        return best_agent
    
    async def _locality_aware_assignment(
        self, 
        task: Task, 
        candidates: List[Any]
    ) -> Optional[Any]:
        """本地化感知分配策略"""
        
        # 如果任务有数据位置要求
        data_location = task.requirements.get("data_location")
        if not data_location:
            return await self._capability_based_assignment(task, candidates)
        
        best_agent = None
        min_distance = float('inf')
        
        for agent in candidates:
            # 计算数据传输成本/距离
            distance = await self._calculate_data_distance(agent, data_location)
            
            if distance < min_distance:
                min_distance = distance
                best_agent = agent
        
        return best_agent
    
    async def _affinity_based_assignment(
        self, 
        task: Task, 
        candidates: List[Any]
    ) -> Optional[Any]:
        """亲和性分配策略"""
        
        # 查找与任务类型有良好历史表现的智能体
        best_agent = None
        best_affinity = 0.0
        
        for agent in candidates:
            agent_id = agent.get("agent_id") if isinstance(agent, dict) else agent.agent_id
            
            # 计算亲和性分数
            affinity = 0.0
            
            # 历史性能
            if agent_id in self.agent_performance:
                performance = self.agent_performance[agent_id].get(task.task_type, 0.5)
                affinity += performance * 0.5
            
            # 历史成功率
            history = self.assignment_history.get(task.task_id, [])
            if agent_id in history:
                affinity += 0.3  # 之前处理过相关任务
            
            # 任务类型专长
            if isinstance(agent, dict):
                capabilities = agent.get("capabilities", [])
            else:
                capabilities = getattr(agent, "capabilities", [])
            
            for cap in capabilities:
                if cap.get("name") == task.task_type:
                    affinity += 0.2
                    break
            
            if affinity > best_affinity:
                best_affinity = affinity
                best_agent = agent
        
        return best_agent
    
    async def _priority_weighted_assignment(
        self, 
        task: Task, 
        candidates: List[Any]
    ) -> Optional[Any]:
        """优先级权重分配策略"""
        
        # 根据任务优先级选择合适的智能体
        if task.priority == TaskPriority.CRITICAL:
            # 选择最可靠的智能体
            return await self._select_most_reliable(candidates)
        elif task.priority == TaskPriority.HIGH:
            # 选择高性能智能体
            return await self._select_high_performance(candidates)
        elif task.priority == TaskPriority.MEDIUM:
            # 使用负载均衡
            return await self._load_balanced_assignment(task, candidates)
        else:
            # LOW或BACKGROUND，选择空闲智能体
            return await self._select_least_busy(candidates)
    
    async def _calculate_capability_score(self, agent: Any, task: Task) -> float:
        """计算能力匹配分数"""
        
        score = 0.0
        
        # 获取智能体能力（兼容字典和对象）
        if isinstance(agent, dict):
            capabilities = agent.get("capabilities", [])
            agent_id = agent.get("agent_id")
        else:
            capabilities = getattr(agent, "capabilities", [])
            agent_id = agent.agent_id
        
        # 能力匹配分数
        for capability in capabilities:
            if capability.get("name") == task.task_type:
                # 基于性能指标计算分数
                metrics = capability.get("performance_metrics", {})
                
                accuracy = metrics.get("accuracy", 0.8)
                throughput = metrics.get("throughput", 1.0)
                avg_latency = metrics.get("avg_latency", 1.0)
                
                # 综合分数计算
                capability_score = (
                    accuracy * 0.4 + 
                    min(throughput / 10.0, 1.0) * 0.3 + 
                    max(0, 1.0 - avg_latency / 10.0) * 0.3
                )
                
                score = max(score, capability_score)
        
        # 历史性能加权
        if agent_id in self.agent_performance:
            historical_score = self.agent_performance[agent_id].get(task.task_type, 0.5)
            score = score * 0.7 + historical_score * 0.3
        
        return score
    
    async def _calculate_resource_efficiency(self, agent: Any, task: Task) -> float:
        """计算资源效率"""
        
        # 获取智能体资源（兼容字典和对象）
        if isinstance(agent, dict):
            resources = agent.get("resources", {})
        else:
            resources = getattr(agent, "resources", {})
        
        cpu_efficiency = 1.0 - resources.get("cpu_usage", 0.5)
        memory_efficiency = 1.0 - resources.get("memory_usage", 0.5)
        
        # 综合效率
        efficiency = (cpu_efficiency + memory_efficiency) / 2
        
        # 考虑任务资源需求匹配度
        if task.resource_requirements:
            match_score = self._calculate_resource_match(resources, task.resource_requirements)
            efficiency *= match_score
        
        return efficiency
    
    async def _calculate_cost_efficiency(self, agent: Any, task: Task) -> float:
        """计算成本效率"""
        
        # 获取智能体信息（兼容字典和对象）
        if isinstance(agent, dict):
            resources = agent.get("resources", {})
            capabilities = agent.get("capabilities", [])
        else:
            resources = getattr(agent, "resources", {})
            capabilities = getattr(agent, "capabilities", [])
        
        # 计算成本
        cost_per_hour = resources.get("cost_per_hour", 1.0)
        estimated_hours = await self._estimate_completion_time(agent, task) / 3600
        estimated_cost = cost_per_hour * estimated_hours
        
        # 能力分数
        capability_score = 0.0
        for capability in capabilities:
            if capability.get("name") == task.task_type:
                capability_score = capability.get("performance_metrics", {}).get("accuracy", 0.8)
                break
        
        # 成本效率 = 能力分数 / 预估成本
        cost_efficiency = capability_score / max(estimated_cost, 0.01)
        
        return cost_efficiency
    
    async def _calculate_data_distance(self, agent: Any, data_location: Dict[str, Any]) -> float:
        """计算数据距离（简化版）"""
        
        # 获取智能体位置（兼容字典和对象）
        if isinstance(agent, dict):
            agent_location = agent.get("location", {})
        else:
            agent_location = getattr(agent, "location", {})
        
        # 简化计算：同区域0，同数据中心1，不同数据中心10
        if agent_location.get("datacenter") == data_location.get("datacenter"):
            return 0
        elif agent_location.get("region") == data_location.get("region"):
            return 1
        else:
            return 10
    
    async def _select_most_reliable(self, candidates: List[Any]) -> Optional[Any]:
        """选择最可靠的智能体"""
        
        best_agent = None
        min_failures = float('inf')
        
        for agent in candidates:
            agent_id = agent.get("agent_id") if isinstance(agent, dict) else agent.agent_id
            failures = len(self.agent_failures.get(agent_id, []))
            
            if failures < min_failures:
                min_failures = failures
                best_agent = agent
        
        return best_agent
    
    async def _select_high_performance(self, candidates: List[Any]) -> Optional[Any]:
        """选择高性能智能体"""
        
        best_agent = None
        best_throughput = 0.0
        
        for agent in candidates:
            # 获取智能体能力（兼容字典和对象）
            if isinstance(agent, dict):
                capabilities = agent.get("capabilities", [])
            else:
                capabilities = getattr(agent, "capabilities", [])
            
            for cap in capabilities:
                metrics = cap.get("performance_metrics", {})
                throughput = metrics.get("throughput", 0)
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_agent = agent
        
        return best_agent
    
    async def _select_least_busy(self, candidates: List[Any]) -> Optional[Any]:
        """选择最空闲的智能体"""
        
        best_agent = None
        min_load = float('inf')
        
        for agent in candidates:
            agent_id = agent.get("agent_id") if isinstance(agent, dict) else agent.agent_id
            load = self.agent_load.get(agent_id, 0)
            
            if load < min_load:
                min_load = load
                best_agent = agent
        
        return best_agent
    
    async def _estimate_completion_time(self, agent: Any, task: Task) -> float:
        """估算任务完成时间（秒）"""
        
        # 获取智能体信息（兼容字典和对象）
        if isinstance(agent, dict):
            capabilities = agent.get("capabilities", [])
            agent_id = agent.get("agent_id")
        else:
            capabilities = getattr(agent, "capabilities", [])
            agent_id = agent.agent_id
        
        # 基础时间
        base_time = 300  # 默认5分钟
        
        for capability in capabilities:
            if capability.get("name") == task.task_type:
                throughput = capability.get("performance_metrics", {}).get("throughput", 1.0)
                avg_latency = capability.get("performance_metrics", {}).get("avg_latency", 1.0)
                
                # 基于吞吐量和延迟估算
                estimated_time = max(avg_latency, base_time / max(throughput, 0.1))
                
                # 基于历史数据调整
                if agent_id in self.task_execution_times:
                    historical_times = self.task_execution_times[agent_id]
                    if historical_times:
                        avg_historical = sum(historical_times) / len(historical_times)
                        estimated_time = estimated_time * 0.5 + avg_historical * 0.5
                
                return estimated_time
        
        return base_time
    
    def _calculate_resource_match(
        self, 
        agent_resources: Dict[str, Any], 
        task_requirements: Dict[str, Any]
    ) -> float:
        """计算资源匹配分数"""
        
        match_scores = []
        
        for resource, required_amount in task_requirements.items():
            if resource in ["cpu", "memory", "disk_space"]:
                if resource == "cpu":
                    available_amount = 1.0 - agent_resources.get("cpu_usage", 0.5)
                elif resource == "memory":
                    total = agent_resources.get("memory_total", 0)
                    used = agent_resources.get("memory_used", 0)
                    available_amount = total - used
                else:
                    total = agent_resources.get("disk_total", 0)
                    used = agent_resources.get("disk_used", 0)
                    available_amount = total - used
                
                if required_amount <= available_amount:
                    # 满足需求，计算过剩程度（过剩太多也不好）
                    excess_ratio = available_amount / max(required_amount, 0.01)
                    if excess_ratio <= 2.0:
                        score = 1.0
                    else:
                        score = 2.0 / excess_ratio  # 过剩惩罚
                else:
                    # 不满足需求
                    score = available_amount / required_amount
                
                match_scores.append(score)
        
        return sum(match_scores) / len(match_scores) if match_scores else 0.5
    
    def _compare_versions(self, version1: str, version2: str) -> int:
        """比较版本号"""
        
        def version_tuple(v):
            return tuple(map(int, (v.split("."))))
        
        try:
            v1_tuple = version_tuple(version1)
            v2_tuple = version_tuple(version2)
            
            if v1_tuple < v2_tuple:
                return -1
            elif v1_tuple > v2_tuple:
                return 1
            else:
                return 0
        except:
            return 0
    
    async def update_agent_performance(
        self, 
        agent_id: str, 
        task_type: str, 
        performance_score: float,
        execution_time: Optional[float] = None
    ):
        """更新智能体性能记录"""
        
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {}
        
        # 使用指数移动平均更新性能分数
        current_score = self.agent_performance[agent_id].get(task_type, 0.5)
        alpha = 0.3  # 学习率
        
        new_score = alpha * performance_score + (1 - alpha) * current_score
        self.agent_performance[agent_id][task_type] = new_score
        
        # 更新执行时间记录
        if execution_time:
            if agent_id not in self.task_execution_times:
                self.task_execution_times[agent_id] = []
            
            self.task_execution_times[agent_id].append(execution_time)
            
            # 保留最近的100条记录
            if len(self.task_execution_times[agent_id]) > 100:
                self.task_execution_times[agent_id] = self.task_execution_times[agent_id][-100:]
    
    async def release_agent(self, agent_id: str):
        """释放智能体（减少负载）"""
        
        if agent_id in self.agent_load:
            self.agent_load[agent_id] = max(0, self.agent_load[agent_id] - 1)
from src.core.logging import get_logger
