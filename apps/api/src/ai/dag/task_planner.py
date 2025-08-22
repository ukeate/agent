"""
DAG任务规划器
使用NetworkX构建和优化任务依赖图
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass
from enum import Enum

import networkx as nx

from models.schemas.workflow import TaskNode, TaskDAG, TaskDependencyType, WorkflowStepType
from src.core.logging import get_logger

logger = get_logger(__name__)


class PlanningError(Exception):
    """规划错误"""
    pass


class SchedulingStrategy(Enum):
    """调度策略"""
    EARLIEST_FIRST = "earliest_first"      # 最早优先
    CRITICAL_PATH = "critical_path"        # 关键路径优先
    RESOURCE_BALANCED = "resource_balanced" # 资源平衡
    DEADLINE_AWARE = "deadline_aware"      # 截止时间感知


@dataclass
class ExecutionPlan:
    """执行计划"""
    stages: List[List[str]]  # 执行阶段，每个阶段包含可并行执行的任务ID
    critical_path: List[str]  # 关键路径
    total_duration: int       # 总预估时间(分钟)
    max_parallelism: int      # 最大并行度
    resource_requirements: Dict[str, Any]  # 资源需求


class TaskPlanner:
    """DAG任务规划器"""
    
    def __init__(self):
        self.graph_cache: Dict[str, nx.DiGraph] = {}
        self.planning_strategies = {
            SchedulingStrategy.EARLIEST_FIRST: self._plan_earliest_first,
            SchedulingStrategy.CRITICAL_PATH: self._plan_critical_path,
            SchedulingStrategy.RESOURCE_BALANCED: self._plan_resource_balanced,
            SchedulingStrategy.DEADLINE_AWARE: self._plan_deadline_aware
        }
    
    def build_graph(self, dag: TaskDAG) -> nx.DiGraph:
        """
        构建NetworkX图
        
        Args:
            dag: 任务依赖图
            
        Returns:
            NetworkX有向图
        """
        graph = nx.DiGraph()
        
        # 添加节点
        for task in dag.nodes:
            graph.add_node(
                task.id,
                task=task,
                name=task.name,
                description=task.description,
                task_type=task.task_type,
                complexity=task.complexity_score,
                duration=task.estimated_duration_minutes or 1,
                priority=task.priority,
                required_tools=task.required_tools
            )
        
        # 添加边（依赖关系）
        for task in dag.nodes:
            for dep_id in task.dependencies:
                if dep_id in [t.id for t in dag.nodes]:
                    graph.add_edge(
                        dep_id, 
                        task.id,
                        dependency_type=task.dependency_type,
                        weight=1  # 可以根据依赖强度调整
                    )
        
        # 缓存图
        self.graph_cache[dag.id] = graph
        
        logger.info(f"构建图完成: {len(graph.nodes)} 节点, {len(graph.edges)} 边")
        return graph
    
    def validate_dag(self, graph: nx.DiGraph) -> List[str]:
        """
        验证DAG有效性
        
        Args:
            graph: NetworkX图
            
        Returns:
            验证错误列表
        """
        errors = []
        
        # 检查是否为有向无环图
        if not nx.is_directed_acyclic_graph(graph):
            cycles = list(nx.simple_cycles(graph))
            errors.append(f"存在循环依赖: {cycles}")
        
        # 检查孤立节点
        isolated_nodes = list(nx.isolates(graph))
        if isolated_nodes:
            logger.warning(f"存在孤立节点: {isolated_nodes}")
        
        # 检查连通性
        if not nx.is_weakly_connected(graph):
            components = list(nx.weakly_connected_components(graph))
            if len(components) > 1:
                logger.warning(f"图不连通，有{len(components)}个组件")
        
        # 检查节点属性完整性
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]
            if 'task' not in node_data:
                errors.append(f"节点{node_id}缺少任务数据")
            if 'duration' not in node_data or node_data['duration'] <= 0:
                errors.append(f"节点{node_id}持续时间无效")
        
        return errors
    
    def compute_topological_order(self, graph: nx.DiGraph) -> List[str]:
        """
        计算拓扑排序
        
        Args:
            graph: NetworkX图
            
        Returns:
            拓扑排序的节点ID列表
        """
        try:
            return list(nx.topological_sort(graph))
        except nx.NetworkXError as e:
            logger.error(f"拓扑排序失败: {e}")
            # 返回任意顺序
            return list(graph.nodes())
    
    def compute_critical_path(self, graph: nx.DiGraph) -> Tuple[List[str], int]:
        """
        计算关键路径
        
        Args:
            graph: NetworkX图
            
        Returns:
            (关键路径节点列表, 关键路径长度)
        """
        try:
            # 找到所有起始节点（入度为0）
            start_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
            # 找到所有结束节点（出度为0）
            end_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]
            
            if not start_nodes or not end_nodes:
                return list(graph.nodes()), 0
            
            # 计算所有路径中的最长路径
            longest_path = []
            max_duration = 0
            
            for start in start_nodes:
                for end in end_nodes:
                    if nx.has_path(graph, start, end):
                        # 使用Dijkstra算法找最长路径（通过取负权重）
                        try:
                            # 创建权重为负的图来找最长路径
                            weighted_graph = graph.copy()
                            for u, v in weighted_graph.edges():
                                duration = graph.nodes[v].get('duration', 1)
                                weighted_graph[u][v]['weight'] = -duration
                            
                            path = nx.shortest_path(
                                weighted_graph, start, end, weight='weight'
                            )
                            
                            # 计算路径总时长
                            path_duration = sum(
                                graph.nodes[node].get('duration', 1) for node in path
                            )
                            
                            if path_duration > max_duration:
                                max_duration = path_duration
                                longest_path = path
                                
                        except nx.NetworkXNoPath:
                            continue
            
            return longest_path, max_duration
            
        except Exception as e:
            logger.error(f"计算关键路径失败: {e}")
            return list(graph.nodes())[:1], 0
    
    def compute_parallel_stages(self, graph: nx.DiGraph) -> List[List[str]]:
        """
        计算并行执行阶段
        
        Args:
            graph: NetworkX图
            
        Returns:
            并行执行阶段列表
        """
        try:
            # 使用拓扑生成器获取并行组
            generations = list(nx.topological_generations(graph))
            return [list(gen) for gen in generations]
        except Exception as e:
            logger.error(f"计算并行阶段失败: {e}")
            # 降级为串行执行
            return [[node] for node in self.compute_topological_order(graph)]
    
    def analyze_bottlenecks(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        分析执行瓶颈
        
        Args:
            graph: NetworkX图
            
        Returns:
            瓶颈分析结果
        """
        analysis = {
            "high_degree_nodes": [],      # 高度数节点
            "long_duration_nodes": [],    # 长时间节点
            "critical_path_nodes": [],    # 关键路径节点
            "resource_conflicts": [],     # 资源冲突
            "suggestions": []             # 优化建议
        }
        
        try:
            # 找到高出度节点（可能的瓶颈）
            for node in graph.nodes():
                out_degree = graph.out_degree(node)
                if out_degree > 3:  # 阈值可配置
                    analysis["high_degree_nodes"].append({
                        "node": node,
                        "out_degree": out_degree,
                        "name": graph.nodes[node].get('name', node)
                    })
            
            # 找到长时间任务
            durations = [graph.nodes[node].get('duration', 1) for node in graph.nodes()]
            if durations:
                avg_duration = sum(durations) / len(durations)
                for node in graph.nodes():
                    duration = graph.nodes[node].get('duration', 1)
                    if duration > avg_duration * 2:  # 超过平均时间2倍
                        analysis["long_duration_nodes"].append({
                            "node": node,
                            "duration": duration,
                            "name": graph.nodes[node].get('name', node)
                        })
            
            # 分析关键路径
            critical_path, _ = self.compute_critical_path(graph)
            analysis["critical_path_nodes"] = critical_path
            
            # 分析工具资源冲突
            tool_usage = {}
            for node in graph.nodes():
                tools = graph.nodes[node].get('required_tools', [])
                for tool in tools:
                    if tool not in tool_usage:
                        tool_usage[tool] = []
                    tool_usage[tool].append(node)
            
            # 找到工具使用冲突
            parallel_stages = self.compute_parallel_stages(graph)
            for stage in parallel_stages:
                stage_tools = {}
                for node in stage:
                    tools = graph.nodes[node].get('required_tools', [])
                    for tool in tools:
                        if tool not in stage_tools:
                            stage_tools[tool] = []
                        stage_tools[tool].append(node)
                
                for tool, nodes in stage_tools.items():
                    if len(nodes) > 1:
                        analysis["resource_conflicts"].append({
                            "tool": tool,
                            "conflicting_nodes": nodes,
                            "stage": stage.index(stage) if stage in parallel_stages else -1
                        })
            
            # 生成优化建议
            if analysis["high_degree_nodes"]:
                analysis["suggestions"].append("考虑重构高出度节点以减少依赖复杂度")
            
            if analysis["long_duration_nodes"]:
                analysis["suggestions"].append("考虑拆分长时间任务或并行化处理")
            
            if analysis["resource_conflicts"]:
                analysis["suggestions"].append("调整任务调度以避免工具资源冲突")
            
        except Exception as e:
            logger.error(f"瓶颈分析失败: {e}")
        
        return analysis
    
    def create_execution_plan(
        self, 
        graph: nx.DiGraph, 
        strategy: SchedulingStrategy = SchedulingStrategy.CRITICAL_PATH,
        constraints: Optional[Dict[str, Any]] = None
    ) -> ExecutionPlan:
        """
        创建执行计划
        
        Args:
            graph: NetworkX图
            strategy: 调度策略
            constraints: 约束条件
            
        Returns:
            执行计划
        """
        constraints = constraints or {}
        
        try:
            # 根据策略选择规划方法
            planner_func = self.planning_strategies.get(strategy)
            if not planner_func:
                logger.warning(f"未知调度策略: {strategy}, 使用默认策略")
                planner_func = self._plan_critical_path
            
            # 执行规划
            plan = planner_func(graph, constraints)
            
            logger.info(f"执行计划创建完成: {len(plan.stages)} 阶段, "
                       f"关键路径长度: {len(plan.critical_path)}")
            
            return plan
            
        except Exception as e:
            logger.error(f"创建执行计划失败: {e}")
            raise PlanningError(f"规划失败: {e}")
    
    def _plan_earliest_first(self, graph: nx.DiGraph, constraints: Dict[str, Any]) -> ExecutionPlan:
        """最早优先调度策略"""
        stages = self.compute_parallel_stages(graph)
        critical_path, total_duration = self.compute_critical_path(graph)
        
        return ExecutionPlan(
            stages=stages,
            critical_path=critical_path,
            total_duration=total_duration,
            max_parallelism=max(len(stage) for stage in stages) if stages else 0,
            resource_requirements=self._estimate_resources(graph)
        )
    
    def _plan_critical_path(self, graph: nx.DiGraph, constraints: Dict[str, Any]) -> ExecutionPlan:
        """关键路径优先调度策略"""
        # 计算关键路径
        critical_path, total_duration = self.compute_critical_path(graph)
        
        # 优先调度关键路径上的任务
        stages = []
        scheduled = set()
        
        # 按拓扑顺序处理
        topo_order = self.compute_topological_order(graph)
        
        while scheduled != set(topo_order):
            current_stage = []
            
            for node in topo_order:
                if node in scheduled:
                    continue
                
                # 检查依赖是否都已调度
                deps_satisfied = all(
                    dep in scheduled for dep in graph.predecessors(node)
                )
                
                if deps_satisfied:
                    # 关键路径任务优先
                    if node in critical_path:
                        current_stage.insert(0, node)
                    else:
                        current_stage.append(node)
            
            if current_stage:
                stages.append(current_stage)
                scheduled.update(current_stage)
            else:
                # 避免无限循环
                remaining = set(topo_order) - scheduled
                if remaining:
                    stages.append(list(remaining))
                    scheduled.update(remaining)
        
        return ExecutionPlan(
            stages=stages,
            critical_path=critical_path,
            total_duration=total_duration,
            max_parallelism=max(len(stage) for stage in stages) if stages else 0,
            resource_requirements=self._estimate_resources(graph)
        )
    
    def _plan_resource_balanced(self, graph: nx.DiGraph, constraints: Dict[str, Any]) -> ExecutionPlan:
        """资源平衡调度策略"""
        max_parallel = constraints.get('max_parallel_tasks', 5)
        
        # 获取基本的并行阶段
        base_stages = self.compute_parallel_stages(graph)
        
        # 重新分配以平衡资源使用
        balanced_stages = []
        for stage in base_stages:
            if len(stage) <= max_parallel:
                balanced_stages.append(stage)
            else:
                # 拆分大阶段
                for i in range(0, len(stage), max_parallel):
                    balanced_stages.append(stage[i:i+max_parallel])
        
        critical_path, total_duration = self.compute_critical_path(graph)
        
        return ExecutionPlan(
            stages=balanced_stages,
            critical_path=critical_path,
            total_duration=total_duration,
            max_parallelism=min(max_parallel, max(len(stage) for stage in balanced_stages) if balanced_stages else 0),
            resource_requirements=self._estimate_resources(graph)
        )
    
    def _plan_deadline_aware(self, graph: nx.DiGraph, constraints: Dict[str, Any]) -> ExecutionPlan:
        """截止时间感知调度策略"""
        deadline_minutes = constraints.get('deadline_minutes')
        
        # 计算基本计划
        base_plan = self._plan_critical_path(graph, constraints)
        
        if deadline_minutes and base_plan.total_duration > deadline_minutes:
            logger.warning(f"预估时间({base_plan.total_duration}分钟)超过截止时间({deadline_minutes}分钟)")
            
            # 尝试增加并行度
            max_parallel = constraints.get('max_parallel_tasks', 5)
            new_max_parallel = min(max_parallel * 2, len(graph.nodes()))
            
            new_constraints = constraints.copy()
            new_constraints['max_parallel_tasks'] = new_max_parallel
            
            return self._plan_resource_balanced(graph, new_constraints)
        
        return base_plan
    
    def _estimate_resources(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """估计资源需求"""
        resources = {
            "total_tasks": len(graph.nodes()),
            "total_estimated_time": sum(
                graph.nodes[node].get('duration', 1) for node in graph.nodes()
            ),
            "required_tools": set(),
            "complexity_distribution": {"simple": 0, "medium": 0, "complex": 0}
        }
        
        for node in graph.nodes():
            # 收集工具需求
            tools = graph.nodes[node].get('required_tools', [])
            resources["required_tools"].update(tools)
            
            # 分析复杂度分布
            complexity = graph.nodes[node].get('complexity', 1.0)
            if complexity < 3.0:
                resources["complexity_distribution"]["simple"] += 1
            elif complexity < 7.0:
                resources["complexity_distribution"]["medium"] += 1
            else:
                resources["complexity_distribution"]["complex"] += 1
        
        resources["required_tools"] = list(resources["required_tools"])
        
        return resources
    
    def optimize_dag(self, dag: TaskDAG, objectives: List[str] = None) -> TaskDAG:
        """
        优化DAG结构
        
        Args:
            dag: 原始DAG
            objectives: 优化目标列表
            
        Returns:
            优化后的DAG
        """
        objectives = objectives or ["minimize_duration", "maximize_parallelism"]
        
        try:
            graph = self.build_graph(dag)
            
            # 执行各种优化
            if "minimize_duration" in objectives:
                graph = self._optimize_for_duration(graph)
            
            if "maximize_parallelism" in objectives:
                graph = self._optimize_for_parallelism(graph)
            
            if "balance_resources" in objectives:
                graph = self._optimize_for_resources(graph)
            
            # 更新DAG
            optimized_dag = self._graph_to_dag(graph, dag)
            
            logger.info("DAG优化完成")
            return optimized_dag
            
        except Exception as e:
            logger.error(f"DAG优化失败: {e}")
            return dag  # 返回原始DAG
    
    def _optimize_for_duration(self, graph: nx.DiGraph) -> nx.DiGraph:
        """优化总执行时间"""
        # 识别可以并行化的串行任务
        for node in list(graph.nodes()):
            successors = list(graph.successors(node))
            if len(successors) == 1:
                successor = successors[0]
                # 如果后继任务只依赖于当前任务，可以考虑合并
                if len(list(graph.predecessors(successor))) == 1:
                    # 这里可以实现任务合并逻辑
                    pass
        
        return graph
    
    def _optimize_for_parallelism(self, graph: nx.DiGraph) -> nx.DiGraph:
        """优化并行度"""
        # 分析可以拆分的任务
        for node in list(graph.nodes()):
            complexity = graph.nodes[node].get('complexity', 1.0)
            duration = graph.nodes[node].get('duration', 1)
            
            # 如果任务复杂度高且时间长，考虑拆分
            if complexity > 7.0 and duration > 10:
                # 这里可以实现任务拆分逻辑
                logger.info(f"任务{node}可以考虑拆分以提高并行度")
        
        return graph
    
    def _optimize_for_resources(self, graph: nx.DiGraph) -> nx.DiGraph:
        """优化资源使用"""
        # 分析工具使用冲突
        tool_conflicts = self.analyze_bottlenecks(graph)["resource_conflicts"]
        
        for conflict in tool_conflicts:
            # 这里可以实现冲突解决逻辑
            logger.info(f"工具{conflict['tool']}存在使用冲突")
        
        return graph
    
    def _graph_to_dag(self, graph: nx.DiGraph, original_dag: TaskDAG) -> TaskDAG:
        """将NetworkX图转换回TaskDAG"""
        # 更新节点信息
        updated_nodes = []
        for node_id in graph.nodes():
            # 找到原始任务节点
            original_task = next((task for task in original_dag.nodes if task.id == node_id), None)
            if original_task:
                # 更新依赖关系
                original_task.dependencies = list(graph.predecessors(node_id))
                updated_nodes.append(original_task)
        
        # 重新计算DAG属性
        stages = self.compute_parallel_stages(graph)
        critical_path, _ = self.compute_critical_path(graph)
        topo_order = self.compute_topological_order(graph)
        
        # 创建新的DAG
        optimized_dag = TaskDAG(
            id=original_dag.id,
            name=original_dag.name,
            description=original_dag.description,
            nodes=updated_nodes,
            is_acyclic=nx.is_directed_acyclic_graph(graph),
            total_nodes=len(updated_nodes),
            max_depth=len(stages),
            topological_order=topo_order,
            parallel_groups=stages,
            critical_path=critical_path,
            created_from_problem=original_dag.created_from_problem,
            decomposition_strategy=original_dag.decomposition_strategy,
            metadata=original_dag.metadata
        )
        
        return optimized_dag