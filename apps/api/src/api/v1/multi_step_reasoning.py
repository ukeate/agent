"""
多步推理工作流API接口
对应技术架构的API端点
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from src.core.utils.timezone_utils import utc_now
from src.core.utils.async_utils import create_task_with_logging
import asyncio
from uuid import uuid4
import psutil
import networkx as nx
from src.ai.workflow.engine import get_workflow_engine
from src.models.schemas.workflow import (
    WorkflowDefinition,
    WorkflowStep,
    WorkflowExecutionMode,
    WorkflowStepType,
    WorkflowStepStatus,
)

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/multi-step-reasoning", tags=["多步推理工作流"])

# 请求模型
class DecompositionRequest(ApiBaseModel):
    problem_statement: str
    context: Optional[str] = None
    strategy: str = "analysis"  # analysis, development, research, optimization
    max_depth: int = 5
    target_complexity: float = 5.0
    enable_branching: bool = False
    time_limit_minutes: Optional[int] = None

class ExecutionRequest(ApiBaseModel):
    workflow_definition_id: str
    execution_mode: str = "parallel"  # sequential, parallel, hybrid
    max_parallel_steps: int = 3
    scheduling_strategy: str = "critical_path"  # critical_path, priority, resource_aware, fifo
    input_data: Optional[Dict[str, Any]] = None

class ExecutionControlRequest(ApiBaseModel):
    execution_id: str
    action: str  # pause, resume, cancel

# 响应模型
class DecompositionResponse(ApiBaseModel):
    task_dag: Dict[str, Any]
    workflow_definition: Dict[str, Any]
    decomposition_metadata: Dict[str, Any]

class ExecutionResponse(ApiBaseModel):
    execution_id: str
    status: str
    workflow_definition_id: str
    progress: float
    current_step: Optional[str]
    start_time: datetime
    estimated_completion: Optional[datetime]

class SystemMetricsResponse(ApiBaseModel):
    active_workers: int
    queue_depth: int
    average_wait_time: float
    success_rate: float
    throughput: float
    resource_utilization: Dict[str, float]

_engine = get_workflow_engine()

@router.post("/decompose", response_model=DecompositionResponse)
async def decompose_problem(request: DecompositionRequest):
    """问题分解接口"""
    try:
        problem = request.problem_statement.strip()
        if not problem:
            raise HTTPException(status_code=400, detail="问题陈述不能为空")

        base = [
            ("问题分析", WorkflowStepType.REASONING, "分析问题、明确目标与约束"),
            ("数据收集", WorkflowStepType.TOOL_CALL, "收集必要信息、资料与上下文"),
            ("深度分析", WorkflowStepType.REASONING, "对关键点进行深入推理与方案对比"),
        ]
        extra = {
            "analysis": [
                ("方案设计", WorkflowStepType.REASONING, "提出可行方案并进行权衡"),
                ("风险评估", WorkflowStepType.VALIDATION, "识别风险并给出应对策略"),
                ("结论输出", WorkflowStepType.AGGREGATION, "汇总结果并输出结论"),
            ],
            "research": [
                ("研究计划", WorkflowStepType.DECISION, "制定研究路径与验证方式"),
                ("证据评估", WorkflowStepType.VALIDATION, "评估证据质量与一致性"),
                ("研究结论", WorkflowStepType.AGGREGATION, "汇总研究发现并给出结论"),
            ],
            "optimization": [
                ("瓶颈定位", WorkflowStepType.REASONING, "定位性能瓶颈与关键路径"),
                ("优化方案", WorkflowStepType.REASONING, "设计优化方案并评估收益"),
                ("验证回归", WorkflowStepType.VALIDATION, "验证优化效果并回归检查"),
            ],
            "development": [
                ("架构设计", WorkflowStepType.REASONING, "设计系统架构与模块边界"),
                ("接口定义", WorkflowStepType.TOOL_CALL, "定义接口与数据契约"),
                ("实现计划", WorkflowStepType.SEQUENTIAL, "制定实现步骤与里程碑"),
                ("测试验证", WorkflowStepType.VALIDATION, "制定测试策略并验证结果"),
            ],
        }

        max_steps = max(3, min(int(request.max_depth or 5), 20))
        template = base + extra.get(request.strategy, extra["analysis"])
        while len(template) < max_steps:
            template.append(("结果整合", WorkflowStepType.AGGREGATION, "整合各步骤输出并形成最终结果"))
        template = template[:max_steps]

        definition_id = str(uuid4())
        steps: List[WorkflowStep] = []

        prev_id: Optional[str] = None
        for idx, (name, step_type, desc) in enumerate(template, start=1):
            step_id = f"step_{idx}"
            deps = [prev_id] if prev_id else []
            step = WorkflowStep(
                id=step_id,
                name=name,
                step_type=step_type,
                description=desc,
                dependencies=deps,
                config={"problem_statement": problem, "step": name, "index": idx, "strategy": request.strategy},
                timeout_seconds=(request.time_limit_minutes or 0) * 60 or None,
            )
            steps.append(step)
            prev_id = step_id

        definition = WorkflowDefinition(
            id=definition_id,
            name=problem[:60],
            description=request.context,
            version="1.0",
            steps=steps,
            execution_mode=WorkflowExecutionMode.SEQUENTIAL,
            max_parallel_steps=1,
            total_timeout_seconds=(request.time_limit_minutes or 0) * 60 or None,
            metadata={
                "strategy": request.strategy,
                "target_complexity": request.target_complexity,
                "enable_branching": request.enable_branching,
                "created_at": utc_now().isoformat(),
            },
            tags=["multi_step_reasoning"],
        )

        _engine.workflow_definitions[definition.id] = definition

        graph = nx.DiGraph()
        for s in steps:
            graph.add_node(s.id)
            for dep in s.dependencies:
                if dep:
                    graph.add_edge(dep, s.id)

        levels: Dict[str, int] = {}
        for node_id in nx.topological_sort(graph):
            preds = list(graph.predecessors(node_id))
            levels[node_id] = (max(levels[p] for p in preds) + 1) if preds else 0
        groups: Dict[int, List[str]] = {}
        for node_id, level in levels.items():
            groups.setdefault(level, []).append(node_id)
        parallel_groups = [groups[level] for level in sorted(groups)]
        critical_path = nx.algorithms.dag.dag_longest_path(graph)

        total_estimated = 0.0
        nodes = []
        for idx, s in enumerate(steps, start=1):
            complexity = max(1.0, float(request.target_complexity or 5.0)) * (idx / max(1, len(steps)))
            duration = max(1.0, complexity * 3.0)
            total_estimated += duration
            nodes.append(
                {
                    "id": s.id,
                    "name": s.name,
                    "description": s.description,
                    "task_type": s.step_type.value,
                    "dependencies": s.dependencies,
                    "complexity_score": round(complexity, 2),
                    "estimated_duration_minutes": round(duration, 1),
                    "priority": max(1, len(steps) - idx + 1),
                    "status": WorkflowStepStatus.PENDING.value,
                }
            )

        edges = [{"from": a, "to": b} for a, b in graph.edges()]
        task_dag = {
            "id": definition.id,
            "name": definition.name,
            "description": definition.description,
            "nodes": nodes,
            "edges": edges,
            "parallel_groups": parallel_groups,
            "critical_path": critical_path,
            "is_acyclic": nx.is_directed_acyclic_graph(graph),
            "total_nodes": len(nodes),
            "max_depth": max(levels.values(), default=0) + 1,
        }

        return DecompositionResponse(
            task_dag=task_dag,
            workflow_definition=definition.model_dump(),
            decomposition_metadata={
                "strategy_used": request.strategy,
                "complexity_achieved": float(request.target_complexity or 5.0),
                "total_estimated_time": round(total_estimated, 1),
                "parallelization_factor": round(len(nodes) / max(1, len(critical_path)), 2),
                "critical_path_length": len(critical_path),
                "created_at": utc_now().isoformat(),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"问题分解失败: {e}")
        raise HTTPException(status_code=500, detail=f"问题分解失败: {str(e)}")

@router.post("/execute", response_model=ExecutionResponse) 
async def start_execution(request: ExecutionRequest):
    """启动工作流执行"""
    try:
        definition = _engine.workflow_definitions.get(request.workflow_definition_id)
        if not definition:
            raise HTTPException(status_code=404, detail="工作流定义不存在")

        mode = WorkflowExecutionMode(request.execution_mode)
        updated_def = definition.model_copy(
            update={
                "execution_mode": mode,
                "max_parallel_steps": int(request.max_parallel_steps or definition.max_parallel_steps),
            }
        )
        _engine.workflow_definitions[updated_def.id] = updated_def

        execution = await _engine.create_execution(
            definition=updated_def,
            input_data=request.input_data or {},
        )

        create_task_with_logging(_engine.execute_workflow(execution.id), logger=logger)
        execution.status = "running"
        execution.started_at = utc_now()

        return ExecutionResponse(
            execution_id=execution.id,
            status=execution.status,
            workflow_definition_id=execution.workflow_definition_id,
            progress=0.0,
            current_step=None,
            start_time=execution.started_at or execution.created_at,
            estimated_completion=None,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"启动执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动执行失败: {str(e)}")

@router.get("/executions/{execution_id}", response_model=ExecutionResponse)
async def get_execution_status(execution_id: str):
    """获取执行状态"""
    execution = await _engine.get_execution_status(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="执行不存在")

    total = max(1, int(execution.total_steps or len(execution.step_executions or [])))
    completed = int(execution.completed_steps or 0)
    progress = float(completed) / float(total)

    start_time = execution.started_at or execution.created_at
    estimated_completion = None
    definition = _engine.workflow_definitions.get(execution.workflow_definition_id)
    if definition and start_time:
        if definition.total_timeout_seconds:
            estimated_completion = start_time + timedelta(seconds=int(definition.total_timeout_seconds))
        else:
            step_timeouts = [s.timeout_seconds for s in definition.steps if s.timeout_seconds]
            if step_timeouts:
                estimated_completion = start_time + timedelta(seconds=int(sum(step_timeouts)))

    return ExecutionResponse(
        execution_id=execution.id,
        status=execution.status,
        workflow_definition_id=execution.workflow_definition_id,
        progress=progress,
        current_step=execution.current_step_id,
        start_time=start_time,
        estimated_completion=estimated_completion,
    )

@router.post("/executions/control")
async def control_execution(request: ExecutionControlRequest):
    """控制执行：pause/resume/cancel"""
    if request.action == "pause":
        ok = await _engine.pause_execution(request.execution_id)
    elif request.action == "resume":
        ok = await _engine.resume_execution(request.execution_id)
    elif request.action == "cancel":
        ok = await _engine.cancel_execution(request.execution_id)
    else:
        raise HTTPException(status_code=400, detail="不支持的控制动作")

    if not ok:
        raise HTTPException(status_code=400, detail="控制执行失败")
    return {"message": "ok", "status": request.action}

@router.get("/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """获取系统指标"""
    executions = list(_engine.active_executions.values())
    total_steps = 0
    completed_steps = 0
    failed_steps = 0
    running_steps = 0
    pending_steps = 0
    total_duration = 0.0

    now = utc_now()
    for ex in executions:
        step_execs = ex.step_executions or []
        total_steps += len(step_execs)
        completed_steps += int(ex.completed_steps or 0)
        failed_steps += int(ex.failed_steps or 0)
        for se in step_execs:
            if se.status == WorkflowStepStatus.RUNNING:
                running_steps += 1
            elif se.status == WorkflowStepStatus.PENDING:
                pending_steps += 1
        if ex.started_at:
            end = ex.completed_at or now
            total_duration += max(0.0, (end - ex.started_at).total_seconds())

    processed = completed_steps + failed_steps
    success_rate = float(completed_steps) / float(processed) if processed > 0 else 0.0
    throughput = (float(completed_steps) / total_duration) * 60.0 if total_duration > 0 else 0.0

    try:
        cpu = psutil.cpu_percent(interval=None) / 100.0
        mem = psutil.virtual_memory().percent / 100.0
        resource_utilization = {"cpu": float(cpu), "memory": float(mem)}
    except Exception:
        resource_utilization = {}

    return SystemMetricsResponse(
        active_workers=running_steps,
        queue_depth=pending_steps,
        average_wait_time=total_duration / max(1, len(executions)) if executions else 0.0,
        success_rate=success_rate,
        throughput=throughput,
        resource_utilization=resource_utilization,
    )

@router.get("/workflows")
async def list_workflows():
    """列出已创建的工作流定义"""
    return {
        "count": len(_engine.workflow_definitions),
        "workflows": [d.model_dump() for d in _engine.workflow_definitions.values()],
    }

@router.get("/executions")
async def list_executions():
    """列出当前活跃的执行"""
    items = []
    for ex in _engine.active_executions.values():
        total = max(1, int(ex.total_steps or len(ex.step_executions or [])))
        progress = float(int(ex.completed_steps or 0)) / float(total)
        items.append({
            "execution_id": ex.id,
            "status": ex.status,
            "workflow_definition_id": ex.workflow_definition_id,
            "progress": progress,
            "current_step": ex.current_step_id,
            "start_time": (ex.started_at or ex.created_at).isoformat() if (ex.started_at or ex.created_at) else None,
        })
    return {"count": len(items), "executions": items}

@router.delete("/executions/{execution_id}")
async def delete_execution(execution_id: str):
    """删除执行记录（仅非运行态）"""
    execution = await _engine.get_execution_status(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="执行不存在")
    if execution.status == "running":
        raise HTTPException(status_code=400, detail="运行中的执行不可删除")
    _engine.active_executions.pop(execution_id, None)
    _engine.execution_graphs.pop(execution_id, None)
    return {"success": True, "deleted": execution_id}

@router.get("/executions/{execution_id}/results")
async def get_execution_results(execution_id: str):
    """获取执行结果"""
    execution = await _engine.get_execution_status(execution_id)
    if not execution:
        raise HTTPException(status_code=404, detail="执行不存在")
    return {
        "execution_id": execution.id,
        "status": execution.status,
        "final_result": execution.final_result,
        "errors": execution.execution_context.get("errors", []) if execution.execution_context else [],
    }
