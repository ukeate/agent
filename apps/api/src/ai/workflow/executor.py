"""
工作流执行器实现
支持串行、并行和条件执行
"""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from uuid import uuid4
import networkx as nx
from src.models.schemas.workflow import (
    WorkflowDefinition, WorkflowExecution, WorkflowStepExecution,
    WorkflowStep, WorkflowStepType, WorkflowStepStatus, WorkflowExecutionMode,
    TaskDependencyType
)
from src.ai.reasoning.cot_engine import BaseCoTEngine
from src.ai.mcp.client import MCPClientManager
from src.core.security.expression import safe_eval_bool

from src.core.logging import get_logger
logger = get_logger(__name__)

class StepExecutionError(Exception):
    """步骤执行错误"""
    ...

class WorkflowExecutor(ABC):
    """工作流执行器抽象基类"""
    
    @abstractmethod
    async def execute(
        self,
        execution: WorkflowExecution,
        definition: WorkflowDefinition,
        graph: nx.DiGraph,
        stream_callback: Optional[callable] = None
    ) -> None:
        """执行工作流"""
        raise NotImplementedError
    
    @abstractmethod
    async def pause(self, execution_id: str) -> None:
        """暂停执行"""
        raise NotImplementedError
    
    @abstractmethod
    async def resume(self, execution_id: str) -> None:
        """恢复执行"""
        raise NotImplementedError
    
    @abstractmethod
    async def cancel(self, execution_id: str) -> None:
        """取消执行"""
        raise NotImplementedError

class BaseStepExecutor:
    """基础步骤执行器"""
    
    def __init__(self):
        self.cot_engine: Optional[BaseCoTEngine] = None
        self.mcp_client: Optional[MCPClientManager] = None
        self.tool_registry: Dict[str, Any] = {}
    
    def set_cot_engine(self, engine: BaseCoTEngine):
        """设置CoT推理引擎"""
        self.cot_engine = engine
    
    def set_mcp_client(self, client: MCPClientManager):
        """设置MCP客户端"""
        self.mcp_client = client
    
    def register_tool(self, name: str, tool: Any):
        """注册工具"""
        self.tool_registry[name] = tool
    
    async def execute_step(
        self,
        step: WorkflowStep,
        step_execution: WorkflowStepExecution,
        context: Dict[str, Any],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行单个步骤
        
        Args:
            step: 步骤定义
            step_execution: 步骤执行状态
            context: 执行上下文
            previous_results: 前序步骤结果
            
        Returns:
            步骤执行结果
        """
        start_time = time.time()
        step_execution.started_at = utc_now()
        step_execution.status = WorkflowStepStatus.RUNNING
        
        try:
            # 准备输入数据
            input_data = self._prepare_input_data(step, context, previous_results)
            step_execution.input_data = input_data
            
            # 检查执行条件
            if step.condition:
                if not await self._evaluate_condition(step.condition, input_data, context):
                    step_execution.status = WorkflowStepStatus.SKIPPED
                    step_execution.completed_at = utc_now()
                    step_execution.duration_ms = int((time.time() - start_time) * 1000)
                    return {"status": "skipped", "reason": "condition_not_met"}
            
            # 根据步骤类型执行
            result = await self._execute_by_type(step, input_data, context)
            
            # 更新执行状态
            step_execution.output_data = result
            step_execution.status = WorkflowStepStatus.COMPLETED
            step_execution.completed_at = utc_now()
            step_execution.duration_ms = int((time.time() - start_time) * 1000)
            
            # 计算置信度（如果是推理步骤）
            if step.step_type == WorkflowStepType.REASONING and isinstance(result, dict):
                step_execution.confidence_score = result.get("confidence", 1.0)
                step_execution.reasoning_chain_id = result.get("chain_id")
            
            logger.info(f"步骤执行完成: {step.id}, 耗时: {step_execution.duration_ms}ms")
            return result
            
        except Exception as e:
            # 处理执行错误
            step_execution.status = WorkflowStepStatus.FAILED
            step_execution.error_message = str(e)
            step_execution.error_code = type(e).__name__
            step_execution.completed_at = utc_now()
            step_execution.duration_ms = int((time.time() - start_time) * 1000)
            
            logger.error(f"步骤执行失败: {step.id}, 错误: {e}")
            
            # 检查是否需要重试
            if step_execution.retry_count < step.retry_count:
                step_execution.retry_count += 1
                logger.info(f"重试步骤: {step.id}, 第{step_execution.retry_count}次")
                return await self.execute_step(step, step_execution, context, previous_results)
            
            raise StepExecutionError(f"步骤执行失败: {step.id}, {e}")
    
    def _prepare_input_data(
        self,
        step: WorkflowStep,
        context: Dict[str, Any],
        previous_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """准备步骤输入数据"""
        input_data = {
            "context": context,
            "previous_results": previous_results,
            "step_config": step.config
        }
        
        # 根据依赖获取特定的前序结果
        if step.dependencies:
            dependency_results = {}
            for dep_id in step.dependencies:
                if dep_id in previous_results:
                    dependency_results[dep_id] = previous_results[dep_id]
            input_data["dependencies"] = dependency_results
        
        return input_data
    
    async def _evaluate_condition(
        self,
        condition: str,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> bool:
        """评估执行条件"""
        try:
            evaluation_context = {
                "input": input_data,
                "context": context,
            }
            return safe_eval_bool(condition, evaluation_context)
            
        except Exception as e:
            logger.warning(f"条件评估失败: {condition}, 错误: {e}")
            return False
    
    async def _execute_by_type(
        self,
        step: WorkflowStep,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """根据步骤类型执行"""
        if step.step_type == WorkflowStepType.REASONING:
            return await self._execute_reasoning_step(step, input_data, context)
        elif step.step_type == WorkflowStepType.TOOL_CALL:
            return await self._execute_tool_call_step(step, input_data, context)
        elif step.step_type == WorkflowStepType.VALIDATION:
            return await self._execute_validation_step(step, input_data, context)
        elif step.step_type == WorkflowStepType.AGGREGATION:
            return await self._execute_aggregation_step(step, input_data, context)
        elif step.step_type == WorkflowStepType.DECISION:
            return await self._execute_decision_step(step, input_data, context)
        else:
            return await self._execute_generic_step(step, input_data, context)
    
    async def _execute_reasoning_step(
        self,
        step: WorkflowStep,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行推理步骤"""
        if not self.cot_engine:
            raise StepExecutionError("CoT推理引擎未配置")
        
        # 从配置中获取推理参数
        config = step.config
        problem = config.get("problem") or input_data.get("problem", "")
        strategy = config.get("strategy", "zero_shot")
        max_steps = config.get("max_steps", 5)
        
        # 构建推理请求
        from models.schemas.reasoning import ReasoningRequest, ReasoningStrategy
        
        request = ReasoningRequest(
            problem=problem,
            context=str(context),
            strategy=ReasoningStrategy(strategy),
            max_steps=max_steps,
            enable_branching=config.get("enable_branching", False)
        )
        
        # 执行推理
        chain = await self.cot_engine.execute_chain(request)
        
        return {
            "chain_id": chain.id,
            "conclusion": chain.conclusion,
            "confidence": chain.confidence_score or 0.8,
            "steps": len(chain.steps),
            "reasoning_type": strategy
        }
    
    async def _execute_tool_call_step(
        self,
        step: WorkflowStep,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行工具调用步骤"""
        config = step.config
        tool_name = config.get("tool")
        
        if not tool_name:
            raise StepExecutionError("工具调用步骤缺少工具名称配置")
        
        # 优先使用MCP客户端
        if self.mcp_client:
            try:
                # 准备工具调用参数
                tool_params = config.get("parameters", {})
                # 可以从input_data中获取动态参数
                for key, value in input_data.items():
                    if key.startswith("param_"):
                        tool_params[key[6:]] = value
                
                result = await self.mcp_client.call_tool("system", tool_name, tool_params)
                return {
                    "tool": tool_name,
                    "result": result,
                    "source": "mcp"
                }
            except Exception as e:
                logger.warning(f"MCP工具调用失败: {tool_name}, 错误: {e}")
        
        # 使用本地工具注册表
        if tool_name in self.tool_registry:
            tool = self.tool_registry[tool_name]
            try:
                if asyncio.iscoroutinefunction(tool):
                    result = await tool(input_data, config)
                else:
                    result = tool(input_data, config)
                
                return {
                    "tool": tool_name,
                    "result": result,
                    "source": "local"
                }
            except Exception as e:
                raise StepExecutionError(f"本地工具调用失败: {tool_name}, {e}")
        
        raise StepExecutionError(f"工具未找到: {tool_name}")
    
    async def _execute_validation_step(
        self,
        step: WorkflowStep,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行验证步骤"""
        config = step.config
        validation_rules = config.get("rules", [])
        data_to_validate = input_data.get("dependencies", {})
        
        validation_results = []
        all_passed = True
        
        for rule in validation_rules:
            try:
                # 简单的验证规则实现
                if rule == "not_empty":
                    passed = bool(data_to_validate)
                elif rule == "has_confidence":
                    passed = any("confidence" in str(result) for result in data_to_validate.values())
                elif rule.startswith("confidence_gt_"):
                    threshold = float(rule.split("_")[-1])
                    confidences = []
                    for result in data_to_validate.values():
                        if isinstance(result, dict) and "confidence" in result:
                            confidences.append(result["confidence"])
                    passed = all(c > threshold for c in confidences) if confidences else False
                else:
                    # 自定义验证规则
                    passed = await self._evaluate_condition(rule, input_data, context)
                
                validation_results.append({
                    "rule": rule,
                    "passed": passed
                })
                
                if not passed:
                    all_passed = False
                    
            except Exception as e:
                logger.warning(f"验证规则执行失败: {rule}, 错误: {e}")
                validation_results.append({
                    "rule": rule,
                    "passed": False,
                    "error": str(e)
                })
                all_passed = False
        
        return {
            "validation_passed": all_passed,
            "results": validation_results,
            "total_rules": len(validation_rules),
            "passed_count": sum(1 for r in validation_results if r["passed"])
        }
    
    async def _execute_aggregation_step(
        self,
        step: WorkflowStep,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行聚合步骤"""
        config = step.config
        aggregation_method = config.get("method", "merge")
        dependency_results = input_data.get("dependencies", {})
        
        if aggregation_method == "merge":
            # 简单合并
            merged_result = {}
            for dep_id, result in dependency_results.items():
                if isinstance(result, dict):
                    merged_result.update(result)
                else:
                    merged_result[dep_id] = result
            
            return {
                "aggregation_method": "merge",
                "result": merged_result,
                "source_count": len(dependency_results)
            }
        
        elif aggregation_method == "consensus":
            # 共识聚合
            if not dependency_results:
                return {"aggregation_method": "consensus", "result": None, "confidence": 0.0}
            
            # 简单的共识实现：选择置信度最高的结果
            best_result = None
            best_confidence = 0.0
            
            for dep_id, result in dependency_results.items():
                if isinstance(result, dict) and "confidence" in result:
                    confidence = result["confidence"]
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = result
            
            return {
                "aggregation_method": "consensus",
                "result": best_result,
                "confidence": best_confidence,
                "source_count": len(dependency_results)
            }
        
        elif aggregation_method == "weighted_average":
            # 加权平均
            weights = config.get("weights", {})
            weighted_sum = 0.0
            total_weight = 0.0
            
            for dep_id, result in dependency_results.items():
                weight = weights.get(dep_id, 1.0)
                if isinstance(result, dict) and "confidence" in result:
                    weighted_sum += result["confidence"] * weight
                    total_weight += weight
            
            average_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            return {
                "aggregation_method": "weighted_average",
                "result": average_confidence,
                "confidence": average_confidence,
                "source_count": len(dependency_results)
            }
        
        else:
            raise StepExecutionError(f"不支持的聚合方法: {aggregation_method}")
    
    async def _execute_decision_step(
        self,
        step: WorkflowStep,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行决策步骤"""
        config = step.config
        decision_rules = config.get("rules", [])
        dependency_results = input_data.get("dependencies", {})
        
        decision_result = None
        for rule in decision_rules:
            condition = rule.get("condition")
            action = rule.get("action")
            
            if await self._evaluate_condition(condition, input_data, context):
                decision_result = action
                break
        
        return {
            "decision": decision_result,
            "rules_evaluated": len(decision_rules),
            "input_sources": list(dependency_results.keys())
        }
    
    async def _execute_generic_step(
        self,
        step: WorkflowStep,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """执行通用步骤"""
        # 通用步骤处理
        return {
            "step_type": step.step_type.value,
            "step_id": step.id,
            "executed_at": utc_now().isoformat(),
            "input_data": input_data,
            "config": step.config
        }

class SequentialExecutor(WorkflowExecutor):
    """串行执行器"""
    
    def __init__(self):
        self.step_executor = BaseStepExecutor()
        self.paused_executions: Set[str] = set()
        self.cancelled_executions: Set[str] = set()
    
    async def execute(
        self,
        execution: WorkflowExecution,
        definition: WorkflowDefinition,
        graph: nx.DiGraph,
        stream_callback: Optional[callable] = None
    ) -> None:
        """串行执行工作流"""
        try:
            # 获取拓扑排序的执行顺序
            topological_order = list(nx.topological_sort(graph))
            step_map = {step.id: step for step in definition.steps}
            step_execution_map = {se.step_id: se for se in execution.step_executions}
            
            results = {}
            
            for step_id in topological_order:
                # 检查是否暂停或取消
                if execution.id in self.paused_executions:
                    logger.info(f"执行已暂停: {execution.id}")
                    return
                
                if execution.id in self.cancelled_executions:
                    logger.info(f"执行已取消: {execution.id}")
                    return
                
                step = step_map.get(step_id)
                step_execution = step_execution_map.get(step_id)
                
                if not step or not step_execution:
                    continue
                
                # 更新当前步骤
                execution.current_step_id = step_id
                
                try:
                    # 执行步骤
                    result = await self.step_executor.execute_step(
                        step, step_execution, execution.execution_context, results
                    )
                    results[step_id] = result
                    execution.completed_steps += 1
                    
                    # 流式回调
                    if stream_callback:
                        await stream_callback({
                            "execution_id": execution.id,
                            "step_id": step_id,
                            "status": step_execution.status,
                            "result": result
                        })
                
                except StepExecutionError as e:
                    execution.failed_steps += 1
                    logger.error(f"步骤执行失败: {step_id}, 错误: {e}")
                    
                    # 检查是否继续执行（根据错误处理配置）
                    error_handling = definition.error_handling
                    if not error_handling.get("continue_on_error", False):
                        raise
            
            # 设置最终结果
            execution.final_result = results
            
        except Exception as e:
            logger.error(f"串行执行失败: {execution.id}, 错误: {e}")
            raise
    
    async def pause(self, execution_id: str) -> None:
        """暂停执行"""
        self.paused_executions.add(execution_id)
    
    async def resume(self, execution_id: str) -> None:
        """恢复执行"""
        self.paused_executions.discard(execution_id)
    
    async def cancel(self, execution_id: str) -> None:
        """取消执行"""
        self.cancelled_executions.add(execution_id)

class ParallelExecutor(WorkflowExecutor):
    """并行执行器"""
    
    def __init__(self):
        self.step_executor = BaseStepExecutor()
        self.paused_executions: Set[str] = set()
        self.cancelled_executions: Set[str] = set()
        self.execution_tasks: Dict[str, List[asyncio.Task]] = {}
    
    async def execute(
        self,
        execution: WorkflowExecution,
        definition: WorkflowDefinition,
        graph: nx.DiGraph,
        stream_callback: Optional[callable] = None
    ) -> None:
        """并行执行工作流"""
        try:
            # 获取并行执行组
            parallel_groups = list(nx.topological_generations(graph))
            step_map = {step.id: step for step in definition.steps}
            step_execution_map = {se.step_id: se for se in execution.step_executions}
            
            results = {}
            execution_tasks = []
            
            for group in parallel_groups:
                # 检查是否暂停或取消
                if execution.id in self.paused_executions:
                    logger.info(f"执行已暂停: {execution.id}")
                    break
                
                if execution.id in self.cancelled_executions:
                    logger.info(f"执行已取消: {execution.id}")
                    break
                
                # 并行执行组内的步骤
                group_tasks = []
                for step_id in group:
                    step = step_map.get(step_id)
                    step_execution = step_execution_map.get(step_id)
                    
                    if step and step_execution:
                        task = asyncio.create_task(
                            self._execute_step_with_callback(
                                step, step_execution, execution, results, stream_callback
                            )
                        )
                        group_tasks.append(task)
                        execution_tasks.append(task)
                
                # 等待组内所有步骤完成
                if group_tasks:
                    await asyncio.gather(*group_tasks, return_exceptions=True)
            
            # 保存任务引用以便取消
            self.execution_tasks[execution.id] = execution_tasks
            
            # 设置最终结果
            execution.final_result = results
            
        except Exception as e:
            logger.error(f"并行执行失败: {execution.id}, 错误: {e}")
            raise
        finally:
            # 清理任务引用
            self.execution_tasks.pop(execution.id, None)
    
    async def _execute_step_with_callback(
        self,
        step: WorkflowStep,
        step_execution: WorkflowStepExecution,
        execution: WorkflowExecution,
        results: Dict[str, Any],
        stream_callback: Optional[callable]
    ) -> None:
        """执行步骤并处理回调"""
        try:
            # 更新当前步骤
            execution.current_step_id = step.id
            
            result = await self.step_executor.execute_step(
                step, step_execution, execution.execution_context, results
            )
            results[step.id] = result
            execution.completed_steps += 1
            
            # 流式回调
            if stream_callback:
                await stream_callback({
                    "execution_id": execution.id,
                    "step_id": step.id,
                    "status": step_execution.status,
                    "result": result
                })
        
        except StepExecutionError as e:
            execution.failed_steps += 1
            logger.error(f"步骤执行失败: {step.id}, 错误: {e}")
    
    async def pause(self, execution_id: str) -> None:
        """暂停执行"""
        self.paused_executions.add(execution_id)
        # 暂停所有相关任务
        tasks = self.execution_tasks.get(execution_id, [])
        for task in tasks:
            if not task.done():
                task.cancel()
    
    async def resume(self, execution_id: str) -> None:
        """恢复执行"""
        self.paused_executions.discard(execution_id)
        # 注意：恢复需要重新创建任务，这里只是移除暂停标记
    
    async def cancel(self, execution_id: str) -> None:
        """取消执行"""
        self.cancelled_executions.add(execution_id)
        # 取消所有相关任务
        tasks = self.execution_tasks.get(execution_id, [])
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # 等待任务取消完成
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
