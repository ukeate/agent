"""
任务分解器
使用CoT推理自动将复杂问题分解为任务依赖图(DAG)
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set
from uuid import uuid4
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory

import networkx as nx

from models.schemas.workflow import (
    TaskDecompositionRequest, TaskNode, TaskDAG, WorkflowStepType,
    TaskDependencyType, WorkflowDefinition, WorkflowStep
)
from models.schemas.reasoning import ReasoningRequest, ReasoningStrategy
from src.ai.reasoning.cot_engine import BaseCoTEngine
from src.core.logging import get_logger

logger = get_logger(__name__)


class TaskDecompositionError(Exception):
    """任务分解错误"""
    pass


class TaskComplexityAnalyzer:
    """任务复杂度分析器"""
    
    def __init__(self):
        self.complexity_keywords = {
            "simple": ["get", "show", "display", "list", "view", "read"],
            "medium": ["analyze", "process", "calculate", "transform", "filter"],
            "complex": ["optimize", "predict", "classify", "recommend", "design", "integrate"]
        }
        self.domain_weights = {
            "data": 1.2,
            "machine_learning": 1.5,
            "optimization": 1.8,
            "integration": 1.3,
            "ui": 0.8
        }
    
    def analyze_complexity(self, task_description: str, context: Optional[str] = None) -> float:
        """
        分析任务复杂度
        
        Args:
            task_description: 任务描述
            context: 上下文信息
            
        Returns:
            复杂度评分 (1.0-10.0)
        """
        description_lower = task_description.lower()
        base_score = 1.0
        
        # 基于关键词的复杂度评估
        for complexity, keywords in self.complexity_keywords.items():
            for keyword in keywords:
                if keyword in description_lower:
                    if complexity == "simple":
                        base_score = max(base_score, 2.0)
                    elif complexity == "medium":
                        base_score = max(base_score, 5.0)
                    elif complexity == "complex":
                        base_score = max(base_score, 8.0)
        
        # 基于描述长度的调整
        length_factor = min(len(task_description) / 100.0, 2.0)
        base_score *= (1.0 + length_factor * 0.3)
        
        # 基于领域的调整
        if context:
            context_lower = context.lower()
            for domain, weight in self.domain_weights.items():
                if domain in context_lower:
                    base_score *= weight
                    break
        
        # 检查是否包含多个动作
        action_count = sum(1 for keywords in self.complexity_keywords.values() 
                          for keyword in keywords if keyword in description_lower)
        if action_count > 2:
            base_score *= 1.5
        
        return min(base_score, 10.0)
    
    def estimate_duration(self, complexity_score: float, task_type: WorkflowStepType) -> int:
        """
        估计任务执行时间
        
        Args:
            complexity_score: 复杂度评分
            task_type: 任务类型
            
        Returns:
            预估时间(分钟)
        """
        base_time = {
            WorkflowStepType.REASONING: 3,
            WorkflowStepType.TOOL_CALL: 1,
            WorkflowStepType.VALIDATION: 1,
            WorkflowStepType.AGGREGATION: 2,
            WorkflowStepType.DECISION: 1
        }.get(task_type, 2)
        
        # 基于复杂度调整
        time_multiplier = 1.0 + (complexity_score - 1.0) * 0.5
        
        return max(1, int(base_time * time_multiplier))


class TaskDecomposer:
    """任务分解器核心类"""
    
    def __init__(self, cot_engine: BaseCoTEngine):
        self.cot_engine = cot_engine
        self.complexity_analyzer = TaskComplexityAnalyzer()
        self.decomposition_templates = {
            "analysis": self._get_analysis_template(),
            "development": self._get_development_template(),
            "research": self._get_research_template(),
            "optimization": self._get_optimization_template()
        }
    
    async def decompose_problem(self, request: TaskDecompositionRequest) -> TaskDAG:
        """
        分解问题为任务DAG
        
        Args:
            request: 分解请求
            
        Returns:
            任务依赖图
        """
        try:
            logger.info(f"开始分解问题: {request.problem_statement}")
            
            # 1. 分析问题类型和选择模板
            problem_type = await self._classify_problem_type(request.problem_statement, request.context)
            template = self.decomposition_templates.get(problem_type, self.decomposition_templates["analysis"])
            
            # 2. 使用CoT推理进行任务分解
            decomposition_reasoning = await self._perform_decomposition_reasoning(request, template)
            
            # 3. 解析推理结果为任务节点
            task_nodes = await self._parse_reasoning_to_tasks(decomposition_reasoning, request)
            
            # 4. 构建DAG并验证
            dag = await self._build_task_dag(task_nodes, request)
            
            # 5. 优化DAG结构
            optimized_dag = await self._optimize_dag(dag, request)
            
            logger.info(f"问题分解完成: {len(optimized_dag.nodes)} 个任务")
            return optimized_dag
            
        except Exception as e:
            logger.error(f"任务分解失败: {e}")
            raise TaskDecompositionError(f"分解失败: {e}")
    
    async def _classify_problem_type(self, problem: str, context: Optional[str] = None) -> str:
        """分类问题类型"""
        problem_lower = problem.lower()
        context_lower = (context or "").lower()
        
        # 基于关键词的简单分类
        if any(word in problem_lower for word in ["analyze", "analysis", "understand", "explore"]):
            return "analysis"
        elif any(word in problem_lower for word in ["develop", "build", "create", "implement"]):
            return "development" 
        elif any(word in problem_lower for word in ["research", "investigate", "find", "discover"]):
            return "research"
        elif any(word in problem_lower for word in ["optimize", "improve", "enhance", "maximize"]):
            return "optimization"
        
        # 默认分析类型
        return "analysis"
    
    async def _perform_decomposition_reasoning(
        self, 
        request: TaskDecompositionRequest, 
        template: str
    ) -> str:
        """执行分解推理"""
        # 构建推理提示
        reasoning_prompt = template.format(
            problem=request.problem_statement,
            context=request.context or "无特定上下文",
            max_depth=request.max_depth,
            target_complexity=request.target_complexity,
            time_limit=request.time_limit_minutes or "无限制"
        )
        
        # 创建推理请求
        reasoning_request = ReasoningRequest(
            problem=reasoning_prompt,
            context=request.context,
            strategy=ReasoningStrategy(request.reasoning_strategy),
            max_steps=min(request.max_depth * 3, 15),  # 限制推理步骤
            enable_branching=request.enable_branching
        )
        
        # 执行推理
        chain = await self.cot_engine.execute_chain(reasoning_request)
        
        if not chain.conclusion:
            raise TaskDecompositionError("推理未产生有效结论")
        
        return chain.conclusion
    
    async def _parse_reasoning_to_tasks(
        self, 
        reasoning_result: str, 
        request: TaskDecompositionRequest
    ) -> List[TaskNode]:
        """解析推理结果为任务节点"""
        tasks = []
        
        try:
            # 尝试解析JSON格式的结果
            if reasoning_result.strip().startswith("{") or reasoning_result.strip().startswith("["):
                parsed_data = json.loads(reasoning_result)
                tasks = await self._parse_json_tasks(parsed_data, request)
            else:
                # 解析文本格式的结果
                tasks = await self._parse_text_tasks(reasoning_result, request)
        
        except json.JSONDecodeError:
            # 如果JSON解析失败，使用文本解析
            tasks = await self._parse_text_tasks(reasoning_result, request)
        
        if not tasks:
            # 如果解析失败，创建基本任务结构
            tasks = await self._create_fallback_tasks(request.problem_statement, request)
        
        return tasks
    
    async def _parse_json_tasks(self, data: Dict[str, Any], request: TaskDecompositionRequest) -> List[TaskNode]:
        """解析JSON格式的任务数据"""
        tasks = []
        
        if isinstance(data, dict) and "tasks" in data:
            task_list = data["tasks"]
        elif isinstance(data, list):
            task_list = data
        else:
            return []
        
        for i, task_data in enumerate(task_list):
            if isinstance(task_data, dict):
                task_id = task_data.get("id", f"task_{i+1}")
                name = task_data.get("name", f"任务 {i+1}")
                description = task_data.get("description", "")
                task_type = self._determine_task_type(name, description)
                dependencies = task_data.get("dependencies", [])
                
                # 分析复杂度
                complexity = self.complexity_analyzer.analyze_complexity(description, request.context)
                duration = self.complexity_analyzer.estimate_duration(complexity, task_type)
                
                task = TaskNode(
                    id=task_id,
                    name=name,
                    description=description,
                    task_type=task_type,
                    dependencies=dependencies,
                    complexity_score=complexity,
                    estimated_duration_minutes=duration,
                    priority=task_data.get("priority", 5)
                )
                tasks.append(task)
        
        return tasks
    
    async def _parse_text_tasks(self, text: str, request: TaskDecompositionRequest) -> List[TaskNode]:
        """解析文本格式的任务描述"""
        tasks = []
        lines = text.split('\n')
        current_task = None
        task_counter = 1
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 检查是否是任务标题行
            if (line.startswith(f"{task_counter}.") or 
                line.startswith(f"Task {task_counter}") or
                line.startswith(f"步骤{task_counter}") or
                line.startswith(f"任务{task_counter}")):
                
                # 保存之前的任务
                if current_task:
                    tasks.append(current_task)
                
                # 提取任务名称
                task_name = line.split(".", 1)[-1].strip() if "." in line else line
                task_type = self._determine_task_type(task_name, "")
                complexity = self.complexity_analyzer.analyze_complexity(task_name, request.context)
                duration = self.complexity_analyzer.estimate_duration(complexity, task_type)
                
                current_task = TaskNode(
                    id=f"task_{task_counter}",
                    name=task_name,
                    description=task_name,
                    task_type=task_type,
                    complexity_score=complexity,
                    estimated_duration_minutes=duration,
                    priority=5
                )
                task_counter += 1
            
            # 检查依赖关系
            elif current_task and ("依赖" in line or "depends" in line.lower()):
                # 简单的依赖解析
                if "task" in line.lower():
                    deps = [word for word in line.split() if word.startswith("task")]
                    current_task.dependencies.extend(deps)
        
        # 添加最后一个任务
        if current_task:
            tasks.append(current_task)
        
        # 如果没有解析到任务，创建基于关键词的任务
        if not tasks:
            tasks = await self._extract_tasks_from_keywords(text, request)
        
        return tasks
    
    async def _extract_tasks_from_keywords(self, text: str, request: TaskDecompositionRequest) -> List[TaskNode]:
        """基于关键词提取任务"""
        tasks = []
        
        # 查找动作关键词
        action_keywords = [
            "analyze", "分析", "process", "处理", "collect", "收集",
            "validate", "验证", "transform", "转换", "generate", "生成",
            "evaluate", "评估", "optimize", "优化", "test", "测试"
        ]
        
        found_actions = []
        text_lower = text.lower()
        
        for keyword in action_keywords:
            if keyword in text_lower:
                found_actions.append(keyword)
        
        # 为每个找到的动作创建任务
        for i, action in enumerate(found_actions[:request.max_depth]):
            task_name = f"{action.title()} {request.problem_statement.split()[-1] if request.problem_statement.split() else 'Data'}"
            task_type = self._determine_task_type(action, "")
            complexity = self.complexity_analyzer.analyze_complexity(action, request.context)
            duration = self.complexity_analyzer.estimate_duration(complexity, task_type)
            
            task = TaskNode(
                id=f"task_{i+1}",
                name=task_name,
                description=f"执行{action}操作",
                task_type=task_type,
                complexity_score=complexity,
                estimated_duration_minutes=duration,
                priority=5,
                dependencies=[f"task_{i}"] if i > 0 else []
            )
            tasks.append(task)
        
        return tasks
    
    async def _create_fallback_tasks(self, problem: str, request: TaskDecompositionRequest) -> List[TaskNode]:
        """创建回退任务结构"""
        logger.warning("使用回退任务结构")
        
        # 创建基本的3步任务流程
        tasks = [
            TaskNode(
                id="task_1",
                name="分析问题",
                description=f"分析问题: {problem}",
                task_type=WorkflowStepType.REASONING,
                complexity_score=3.0,
                estimated_duration_minutes=5,
                priority=10
            ),
            TaskNode(
                id="task_2", 
                name="执行处理",
                description=f"处理问题: {problem}",
                task_type=WorkflowStepType.TOOL_CALL,
                dependencies=["task_1"],
                complexity_score=5.0,
                estimated_duration_minutes=10,
                priority=8
            ),
            TaskNode(
                id="task_3",
                name="验证结果",
                description=f"验证处理结果",
                task_type=WorkflowStepType.VALIDATION,
                dependencies=["task_2"],
                complexity_score=2.0,
                estimated_duration_minutes=3,
                priority=6
            )
        ]
        
        return tasks
    
    def _determine_task_type(self, name: str, description: str) -> WorkflowStepType:
        """确定任务类型"""
        text = (name + " " + description).lower()
        
        if any(word in text for word in ["analyze", "think", "reason", "分析", "思考", "推理"]):
            return WorkflowStepType.REASONING
        elif any(word in text for word in ["call", "invoke", "execute", "run", "调用", "执行", "运行"]):
            return WorkflowStepType.TOOL_CALL
        elif any(word in text for word in ["validate", "verify", "check", "test", "验证", "检查", "测试"]):
            return WorkflowStepType.VALIDATION
        elif any(word in text for word in ["aggregate", "merge", "combine", "sum", "聚合", "合并", "汇总"]):
            return WorkflowStepType.AGGREGATION
        elif any(word in text for word in ["decide", "choose", "select", "判断", "决策", "选择"]):
            return WorkflowStepType.DECISION
        else:
            return WorkflowStepType.REASONING  # 默认推理类型
    
    async def _build_task_dag(self, tasks: List[TaskNode], request: TaskDecompositionRequest) -> TaskDAG:
        """构建任务DAG"""
        # 创建NetworkX图进行验证
        graph = nx.DiGraph()
        
        # 添加节点
        for task in tasks:
            graph.add_node(task.id, task=task)
        
        # 添加边（依赖关系）
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in [t.id for t in tasks]:
                    graph.add_edge(dep_id, task.id)
        
        # 检查循环依赖
        if not nx.is_directed_acyclic_graph(graph):
            logger.warning("检测到循环依赖，正在修复...")
            graph = self._fix_cycles(graph, tasks)
        
        # 计算拓扑排序
        try:
            topological_order = list(nx.topological_sort(graph))
        except nx.NetworkXError:
            logger.error("无法计算拓扑排序")
            topological_order = [task.id for task in tasks]
        
        # 分析并行执行组
        parallel_groups = self._analyze_parallel_groups(graph)
        
        # 计算关键路径
        critical_path = self._calculate_critical_path(graph, tasks)
        
        # 创建DAG对象
        dag = TaskDAG(
            id=str(uuid4()),
            name=f"分解任务: {request.problem_statement[:50]}...",
            description=f"自动分解的任务依赖图",
            nodes=tasks,
            is_acyclic=nx.is_directed_acyclic_graph(graph),
            total_nodes=len(tasks),
            max_depth=self._calculate_max_depth(graph),
            topological_order=topological_order,
            parallel_groups=parallel_groups,
            critical_path=critical_path,
            created_from_problem=request.problem_statement,
            decomposition_strategy=request.reasoning_strategy,
            metadata={
                "complexity_target": request.target_complexity,
                "max_depth_requested": request.max_depth,
                "time_limit": request.time_limit_minutes,
                "created_at": utc_now().isoformat()
            }
        )
        
        return dag
    
    def _fix_cycles(self, graph: nx.DiGraph, tasks: List[TaskNode]) -> nx.DiGraph:
        """修复循环依赖"""
        # 找到所有循环
        cycles = list(nx.simple_cycles(graph))
        
        for cycle in cycles:
            # 移除循环中的最后一条边
            if len(cycle) >= 2:
                graph.remove_edge(cycle[-1], cycle[0])
                logger.info(f"移除循环边: {cycle[-1]} -> {cycle[0]}")
        
        return graph
    
    def _analyze_parallel_groups(self, graph: nx.DiGraph) -> List[List[str]]:
        """分析并行执行组"""
        try:
            # 使用拓扑生成器找到并行组
            generations = list(nx.topological_generations(graph))
            return [list(gen) for gen in generations]
        except:
            # 如果失败，返回串行结构
            return [[node] for node in graph.nodes()]
    
    def _calculate_critical_path(self, graph: nx.DiGraph, tasks: List[TaskNode]) -> List[str]:
        """计算关键路径"""
        task_durations = {task.id: task.estimated_duration_minutes or 1 for task in tasks}
        
        try:
            # 简单的关键路径算法
            if not graph.nodes():
                return []
            
            # 找到没有前驱的节点作为起点
            start_nodes = [n for n in graph.nodes() if graph.in_degree(n) == 0]
            if not start_nodes:
                return list(graph.nodes())[:1]
            
            # 找到没有后继的节点作为终点
            end_nodes = [n for n in graph.nodes() if graph.out_degree(n) == 0]
            if not end_nodes:
                return list(graph.nodes())[-1:]
            
            # 计算最长路径
            longest_path = []
            max_length = 0
            
            for start in start_nodes:
                for end in end_nodes:
                    try:
                        if nx.has_path(graph, start, end):
                            path = nx.shortest_path(graph, start, end)
                            path_length = sum(task_durations.get(node, 1) for node in path)
                            if path_length > max_length:
                                max_length = path_length
                                longest_path = path
                    except nx.NetworkXNoPath:
                        continue
            
            return longest_path or [start_nodes[0]]
            
        except Exception as e:
            logger.warning(f"计算关键路径失败: {e}")
            return list(graph.nodes())[:1]
    
    def _calculate_max_depth(self, graph: nx.DiGraph) -> int:
        """计算图的最大深度"""
        try:
            if not graph.nodes():
                return 0
            
            # 计算最长路径长度
            try:
                return nx.dag_longest_path_length(graph)
            except:
                # 如果不是DAG，返回节点数
                return len(graph.nodes())
        except:
            return 0
    
    async def _optimize_dag(self, dag: TaskDAG, request: TaskDecompositionRequest) -> TaskDAG:
        """优化DAG结构"""
        # 优化并行度
        if len(dag.parallel_groups) > 0:
            max_parallel_size = max(len(group) for group in dag.parallel_groups)
            if max_parallel_size > 5:  # 限制最大并行度
                logger.info(f"并行度过高({max_parallel_size})，建议调整")
        
        # 优化任务粒度
        await self._optimize_task_granularity(dag, request)
        
        # 设置任务优先级
        self._set_task_priorities(dag)
        
        return dag
    
    async def _optimize_task_granularity(self, dag: TaskDAG, request: TaskDecompositionRequest):
        """优化任务粒度"""
        # 合并过于细粒度的任务
        simple_tasks = [task for task in dag.nodes if task.complexity_score < 2.0]
        
        if len(simple_tasks) > request.max_depth:
            logger.info(f"检测到{len(simple_tasks)}个简单任务，建议合并")
            # 这里可以实现任务合并逻辑
    
    def _set_task_priorities(self, dag: TaskDAG):
        """设置任务优先级"""
        # 基于关键路径设置优先级
        critical_tasks = set(dag.critical_path)
        
        for task in dag.nodes:
            if task.id in critical_tasks:
                task.priority = min(task.priority + 3, 10)  # 提高关键路径任务优先级
            
            # 基于依赖数量调整优先级
            dependency_count = len(task.dependencies)
            if dependency_count == 0:
                task.priority = min(task.priority + 2, 10)  # 提高起始任务优先级
    
    def _get_analysis_template(self) -> str:
        """获取分析类问题的分解模板"""
        return """
作为一个专业的任务分解专家，请将以下问题分解为具体的可执行任务：

问题: {problem}
上下文: {context}
目标复杂度: {target_complexity}
最大分解深度: {max_depth}
时间限制: {time_limit}分钟

请按照以下格式输出JSON结果：
{{
    "tasks": [
        {{
            "id": "task_1",
            "name": "任务名称",
            "description": "详细描述",
            "dependencies": [],
            "priority": 5
        }}
    ]
}}

分解原则：
1. 每个任务应该是具体可执行的
2. 任务之间的依赖关系要清晰
3. 考虑并行执行的可能性
4. 确保任务粒度适中
"""
    
    def _get_development_template(self) -> str:
        """获取开发类问题的分解模板"""
        return """
作为一个软件开发专家，请将以下开发需求分解为具体的开发任务：

需求: {problem}
上下文: {context}
目标复杂度: {target_complexity}
最大分解深度: {max_depth}
时间限制: {time_limit}分钟

请按照软件开发生命周期分解任务，输出JSON格式：
{{
    "tasks": [
        {{
            "id": "task_1",
            "name": "需求分析",
            "description": "分析具体需求",
            "dependencies": [],
            "priority": 10
        }},
        {{
            "id": "task_2", 
            "name": "设计",
            "description": "系统设计",
            "dependencies": ["task_1"],
            "priority": 8
        }}
    ]
}}
"""
    
    def _get_research_template(self) -> str:
        """获取研究类问题的分解模板"""
        return """
作为一个研究专家，请将以下研究问题分解为系统的研究任务：

研究问题: {problem}
研究背景: {context}
目标复杂度: {target_complexity}
最大分解深度: {max_depth}
时间限制: {time_limit}分钟

请按照研究方法论分解任务，输出JSON格式：
{{
    "tasks": [
        {{
            "id": "task_1",
            "name": "文献调研",
            "description": "收集相关文献",
            "dependencies": [],
            "priority": 9
        }}
    ]
}}
"""
    
    def _get_optimization_template(self) -> str:
        """获取优化类问题的分解模板"""
        return """
作为一个优化专家，请将以下优化问题分解为系统的优化任务：

优化目标: {problem}
当前状况: {context}
目标复杂度: {target_complexity}
最大分解深度: {max_depth}
时间限制: {time_limit}分钟

请按照优化流程分解任务，输出JSON格式：
{{
    "tasks": [
        {{
            "id": "task_1",
            "name": "现状分析",
            "description": "分析当前状况",
            "dependencies": [],
            "priority": 10
        }}
    ]
}}
"""


async def create_workflow_from_dag(dag: TaskDAG) -> WorkflowDefinition:
    """
    从TaskDAG创建WorkflowDefinition
    
    Args:
        dag: 任务依赖图
        
    Returns:
        工作流定义
    """
    # 转换TaskNode为WorkflowStep
    steps = []
    for task_node in dag.nodes:
        step = WorkflowStep(
            id=task_node.id,
            name=task_node.name,
            step_type=task_node.task_type,
            description=task_node.description,
            dependencies=task_node.dependencies,
            dependency_type=task_node.dependency_type,
            config=task_node.config,
            timeout_seconds=task_node.estimated_duration_minutes * 60 if task_node.estimated_duration_minutes else None
        )
        steps.append(step)
    
    # 创建工作流定义
    workflow_definition = WorkflowDefinition(
        id=dag.id,
        name=dag.name,
        description=dag.description,
        steps=steps,
        metadata=dag.metadata
    )
    
    return workflow_definition