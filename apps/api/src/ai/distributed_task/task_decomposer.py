"""任务分解器实现"""

import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from .models import Task, TaskStatus, TaskPriority


class TaskDecomposer:
    """任务分解器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.decomposition_strategies = {
            "parallel": self._parallel_decomposition,
            "sequential": self._sequential_decomposition,
            "hierarchical": self._hierarchical_decomposition,
            "pipeline": self._pipeline_decomposition
        }
    
    async def decompose_task(self, task: Task) -> List[Task]:
        """分解任务"""
        
        try:
            # 确定分解策略
            strategy = task.requirements.get("decomposition_strategy", "parallel")
            
            if strategy not in self.decomposition_strategies:
                self.logger.warning(f"Unknown decomposition strategy: {strategy}, using parallel")
                strategy = "parallel"
            
            # 执行分解
            subtasks = await self.decomposition_strategies[strategy](task)
            
            # 设置子任务关系
            for subtask in subtasks:
                subtask.parent_task_id = task.task_id
                task.subtask_ids.append(subtask.task_id)
            
            # 更新父任务状态
            task.status = TaskStatus.DECOMPOSED
            
            self.logger.info(f"Task {task.task_id} decomposed into {len(subtasks)} subtasks")
            return subtasks
            
        except ValueError as ve:
            self.logger.error(f"Invalid task data for decomposition {task.task_id}: {ve}")
            # Mark task as failed due to invalid data
            task.status = TaskStatus.FAILED
            task.error = f"Decomposition failed: {str(ve)}"
            return []
        except Exception as e:
            self.logger.error(f"Unexpected error decomposing task {task.task_id}: {e}")
            # Mark task as failed 
            task.status = TaskStatus.FAILED
            task.error = f"Decomposition error: {str(e)}"
            return []
    
    async def _parallel_decomposition(self, task: Task) -> List[Task]:
        """并行分解策略"""
        
        subtasks = []
        data_chunks = task.data.get("chunks", [])
        
        if not data_chunks:
            # 如果没有预定义chunks，尝试自动分解
            data_chunks = await self._auto_chunk_data(task.data)
        
        for i, chunk in enumerate(data_chunks):
            subtask = Task(
                task_id=f"{task.task_id}_subtask_{i}",
                task_type=task.task_type,
                data=chunk,
                requirements=task.requirements.copy(),
                priority=task.priority,
                created_at=datetime.now(),
                parent_task_id=task.task_id,
                resource_requirements=self._scale_requirements(
                    task.resource_requirements, 
                    len(data_chunks)
                )
            )
            subtasks.append(subtask)
        
        return subtasks
    
    async def _sequential_decomposition(self, task: Task) -> List[Task]:
        """序列分解策略"""
        
        subtasks = []
        steps = task.data.get("steps", [])
        
        if not steps:
            # 如果没有预定义步骤，创建默认步骤
            steps = await self._generate_default_steps(task)
        
        for i, step in enumerate(steps):
            subtask = Task(
                task_id=f"{task.task_id}_step_{i}",
                task_type=step.get("type", task.task_type),
                data=step.get("data", {}),
                requirements=step.get("requirements", task.requirements.copy()),
                priority=task.priority,
                created_at=datetime.now(),
                parent_task_id=task.task_id,
                timeout_seconds=step.get("timeout", task.timeout_seconds // len(steps))
            )
            
            # 设置依赖关系
            if i > 0:
                subtask.dependencies.append(f"{task.task_id}_step_{i-1}")
            
            subtasks.append(subtask)
        
        return subtasks
    
    async def _hierarchical_decomposition(self, task: Task) -> List[Task]:
        """分层分解策略"""
        
        subtasks = []
        hierarchy = task.data.get("hierarchy", {})
        
        if not hierarchy:
            # 如果没有层级结构，生成默认层级
            hierarchy = await self._generate_default_hierarchy(task)
        
        # 递归处理层级结构
        await self._process_hierarchy_level(
            hierarchy, 
            task.task_id, 
            task, 
            subtasks
        )
        
        return subtasks
    
    async def _pipeline_decomposition(self, task: Task) -> List[Task]:
        """管道分解策略"""
        
        subtasks = []
        pipeline_stages = task.data.get("pipeline", [])
        
        if not pipeline_stages:
            # 如果没有管道定义，生成默认管道
            pipeline_stages = await self._generate_default_pipeline(task)
        
        for i, stage in enumerate(pipeline_stages):
            subtask = Task(
                task_id=f"{task.task_id}_stage_{i}",
                task_type=stage.get("type", "pipeline_stage"),
                data=stage.get("data", {}),
                requirements=stage.get("requirements", {}),
                priority=task.priority,
                created_at=datetime.now(),
                parent_task_id=task.task_id
            )
            
            # 管道依赖关系
            if i > 0:
                subtask.dependencies.append(f"{task.task_id}_stage_{i-1}")
            
            # 设置管道特定属性
            subtask.execution_context["pipeline_stage"] = i
            subtask.execution_context["total_stages"] = len(pipeline_stages)
            
            subtasks.append(subtask)
        
        return subtasks
    
    async def _auto_chunk_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """自动数据分块"""
        
        chunk_size = 1000  # 可配置
        chunks = []
        
        if "items" in data and isinstance(data["items"], list):
            items = data["items"]
            for i in range(0, len(items), chunk_size):
                chunk = data.copy()
                chunk["items"] = items[i:i+chunk_size]
                chunk["chunk_index"] = i // chunk_size
                chunk["total_chunks"] = (len(items) + chunk_size - 1) // chunk_size
                chunks.append(chunk)
        elif "text" in data and isinstance(data["text"], str):
            # 文本分块
            text = data["text"]
            for i in range(0, len(text), chunk_size):
                chunk = data.copy()
                chunk["text"] = text[i:i+chunk_size]
                chunk["chunk_index"] = i // chunk_size
                chunk["total_chunks"] = (len(text) + chunk_size - 1) // chunk_size
                chunks.append(chunk)
        elif "data_points" in data and isinstance(data["data_points"], list):
            # 数据点分块
            data_points = data["data_points"]
            for i in range(0, len(data_points), chunk_size):
                chunk = data.copy()
                chunk["data_points"] = data_points[i:i+chunk_size]
                chunk["chunk_index"] = i // chunk_size
                chunk["total_chunks"] = (len(data_points) + chunk_size - 1) // chunk_size
                chunks.append(chunk)
        else:
            # 如果无法分块，返回原数据
            chunks.append(data)
        
        return chunks
    
    async def _process_hierarchy_level(
        self, 
        level: Dict[str, Any], 
        parent_id: str, 
        parent_task: Task, 
        subtasks: List[Task]
    ):
        """处理层级结构"""
        
        for key, value in level.items():
            subtask_id = f"{parent_id}_{key}"
            
            if isinstance(value, dict) and "subtasks" in value:
                # 这是一个包含子任务的节点
                subtask = Task(
                    task_id=subtask_id,
                    task_type=value.get("type", parent_task.task_type),
                    data=value.get("data", {}),
                    requirements=value.get("requirements", {}),
                    priority=value.get("priority", parent_task.priority),
                    created_at=datetime.now(),
                    parent_task_id=parent_id,
                    timeout_seconds=value.get("timeout", parent_task.timeout_seconds)
                )
                subtasks.append(subtask)
                
                # 递归处理子任务
                await self._process_hierarchy_level(
                    value["subtasks"], 
                    subtask_id, 
                    subtask, 
                    subtasks
                )
            else:
                # 这是一个叶子节点任务
                subtask = Task(
                    task_id=subtask_id,
                    task_type=parent_task.task_type,
                    data=value if isinstance(value, dict) else {"content": value},
                    requirements=parent_task.requirements.copy(),
                    priority=parent_task.priority,
                    created_at=datetime.now(),
                    parent_task_id=parent_id
                )
                subtasks.append(subtask)
    
    def _scale_requirements(
        self, 
        requirements: Dict[str, Any], 
        scale_factor: int
    ) -> Dict[str, Any]:
        """按比例缩放资源需求"""
        
        scaled = requirements.copy()
        
        scalable_fields = ["memory", "cpu", "disk_space", "bandwidth"]
        
        for field in scalable_fields:
            if field in scaled:
                # 按比例缩放，但保证最小值
                scaled[field] = max(
                    scaled[field] / scale_factor,
                    self._get_min_requirement(field)
                )
        
        return scaled
    
    def _get_min_requirement(self, field: str) -> float:
        """获取资源最小需求"""
        
        min_requirements = {
            "memory": 128,  # MB
            "cpu": 0.1,     # cores
            "disk_space": 10,  # MB
            "bandwidth": 1   # Mbps
        }
        
        return min_requirements.get(field, 1)
    
    async def _generate_default_steps(self, task: Task) -> List[Dict[str, Any]]:
        """生成默认步骤"""
        
        # 根据任务类型生成默认步骤
        if task.task_type == "data_processing":
            return [
                {"type": "data_validation", "data": {"validate": True}},
                {"type": "data_transformation", "data": {"transform": True}},
                {"type": "data_aggregation", "data": {"aggregate": True}},
                {"type": "data_output", "data": {"output": True}}
            ]
        elif task.task_type == "model_training":
            return [
                {"type": "data_preparation", "data": {"prepare": True}},
                {"type": "feature_engineering", "data": {"engineer": True}},
                {"type": "model_training", "data": {"train": True}},
                {"type": "model_evaluation", "data": {"evaluate": True}},
                {"type": "model_deployment", "data": {"deploy": True}}
            ]
        else:
            # 通用步骤
            return [
                {"type": "initialization", "data": {"init": True}},
                {"type": "processing", "data": {"process": True}},
                {"type": "finalization", "data": {"finalize": True}}
            ]
    
    async def _generate_default_hierarchy(self, task: Task) -> Dict[str, Any]:
        """生成默认层级结构"""
        
        return {
            "preparation": {
                "type": "preparation",
                "data": {"phase": "prepare"},
                "subtasks": {
                    "validate": {"type": "validation", "data": {"validate": True}},
                    "setup": {"type": "setup", "data": {"setup": True}}
                }
            },
            "execution": {
                "type": "execution",
                "data": {"phase": "execute"},
                "subtasks": {
                    "main": {"type": "main_processing", "data": {"process": True}},
                    "secondary": {"type": "secondary_processing", "data": {"process": True}}
                }
            },
            "completion": {
                "type": "completion",
                "data": {"phase": "complete"},
                "subtasks": {
                    "verification": {"type": "verification", "data": {"verify": True}},
                    "cleanup": {"type": "cleanup", "data": {"cleanup": True}}
                }
            }
        }
    
    async def _generate_default_pipeline(self, task: Task) -> List[Dict[str, Any]]:
        """生成默认管道"""
        
        if task.task_type == "etl":
            return [
                {"type": "extract", "data": {"source": "default"}},
                {"type": "transform", "data": {"rules": []}},
                {"type": "load", "data": {"destination": "default"}}
            ]
        elif task.task_type == "analysis":
            return [
                {"type": "data_ingestion", "data": {"ingest": True}},
                {"type": "data_cleaning", "data": {"clean": True}},
                {"type": "analysis", "data": {"analyze": True}},
                {"type": "visualization", "data": {"visualize": True}},
                {"type": "reporting", "data": {"report": True}}
            ]
        else:
            # 通用管道
            return [
                {"type": "input", "data": {"stage": "input"}},
                {"type": "processing", "data": {"stage": "process"}},
                {"type": "output", "data": {"stage": "output"}}
            ]
    
    async def validate_decomposition(self, parent_task: Task, subtasks: List[Task]) -> bool:
        """验证任务分解的有效性"""
        
        try:
            # 验证所有子任务都有父任务ID
            for subtask in subtasks:
                if subtask.parent_task_id != parent_task.task_id:
                    self.logger.error(f"Subtask {subtask.task_id} has incorrect parent ID")
                    return False
            
            # 验证依赖关系没有循环
            if not self._validate_dependencies(subtasks):
                self.logger.error("Circular dependency detected in subtasks")
                return False
            
            # 验证资源需求总和不超过父任务
            total_requirements = self._sum_requirements(subtasks)
            if not self._validate_requirements(parent_task.resource_requirements, total_requirements):
                self.logger.warning("Total resource requirements exceed parent task limits")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to validate decomposition: {e}")
            return False
    
    def _validate_dependencies(self, tasks: List[Task]) -> bool:
        """验证依赖关系（检查循环依赖）"""
        
        task_map = {task.task_id: task for task in tasks}
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = task_map.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dep_id in task_map:
                        if dep_id not in visited:
                            if has_cycle(dep_id):
                                return True
                        elif dep_id in rec_stack:
                            return True
            
            rec_stack.remove(task_id)
            return False
        
        for task in tasks:
            if task.task_id not in visited:
                if has_cycle(task.task_id):
                    return False
        
        return True
    
    def _sum_requirements(self, tasks: List[Task]) -> Dict[str, Any]:
        """汇总资源需求"""
        
        total = {}
        
        for task in tasks:
            for key, value in task.resource_requirements.items():
                if isinstance(value, (int, float)):
                    total[key] = total.get(key, 0) + value
                elif isinstance(value, bool):
                    total[key] = total.get(key, False) or value
        
        return total
    
    def _validate_requirements(
        self, 
        parent_requirements: Dict[str, Any], 
        total_requirements: Dict[str, Any]
    ) -> bool:
        """验证资源需求"""
        
        for key, parent_value in parent_requirements.items():
            if isinstance(parent_value, (int, float)):
                total_value = total_requirements.get(key, 0)
                if total_value > parent_value * 1.1:  # 允许10%的超出
                    return False
        
        return True