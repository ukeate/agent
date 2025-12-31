"""
离线推理引擎

集成离线CoT推理，支持：
- 离线工作流执行
- 推理状态管理
- 决策解释
- 推理链构建
"""

import json
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4
from .local_inference import LocalInferenceEngine, InferenceRequest, InferenceResult, ModelType
from .memory_manager import OfflineMemoryManager, MemoryEntry, MemoryType, MemoryPriority
from ..models.schemas.offline import OfflineMode, NetworkStatus, VectorClock

class ReasoningStep(str, Enum):
    """推理步骤类型"""
    PROBLEM_ANALYSIS = "problem_analysis"
    INFORMATION_GATHERING = "information_gathering"
    HYPOTHESIS_GENERATION = "hypothesis_generation"
    EVIDENCE_EVALUATION = "evidence_evaluation"
    LOGICAL_REASONING = "logical_reasoning"
    CONCLUSION_GENERATION = "conclusion_generation"
    VALIDATION = "validation"

class ReasoningStrategy(str, Enum):
    """推理策略"""
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"
    BACKWARD_CHAINING = "backward_chaining"
    FORWARD_CHAINING = "forward_chaining"
    ANALOGICAL_REASONING = "analogical_reasoning"
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"

class WorkflowStatus(str, Enum):
    """工作流状态"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ReasoningStepResult:
    """推理步骤结果"""
    step_id: str
    step_type: ReasoningStep
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    reasoning_text: str
    confidence_score: float
    execution_time_ms: float
    tokens_used: int
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=utc_now)

@dataclass
class ReasoningChain:
    """推理链"""
    chain_id: str
    session_id: str
    strategy: ReasoningStrategy
    steps: List[ReasoningStepResult] = field(default_factory=list)
    final_conclusion: Optional[str] = None
    overall_confidence: float = 0.0
    total_execution_time_ms: float = 0.0
    total_tokens_used: int = 0
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=utc_now)
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningWorkflow:
    """推理工作流"""
    workflow_id: str
    name: str
    description: str
    session_id: str
    strategy: ReasoningStrategy
    steps_definition: List[Dict[str, Any]]
    current_step_index: int = 0
    status: WorkflowStatus = WorkflowStatus.PENDING
    reasoning_chain: Optional[ReasoningChain] = None
    created_at: datetime = field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class OfflineReasoningEngine:
    """离线推理引擎"""
    
    def __init__(
        self, 
        inference_engine: LocalInferenceEngine,
        memory_manager: OfflineMemoryManager
    ):
        self.inference_engine = inference_engine
        self.memory_manager = memory_manager
        
        # 活跃的推理工作流
        self.active_workflows: Dict[str, ReasoningWorkflow] = {}
        self.completed_workflows: Dict[str, ReasoningWorkflow] = {}
        
        # 推理模板
        self.reasoning_templates = self._init_reasoning_templates()
        
        # 配置
        self.max_concurrent_workflows = 5
        self.max_reasoning_steps = 20
        self.default_confidence_threshold = 0.7
    
    def _init_reasoning_templates(self) -> Dict[ReasoningStrategy, Dict[str, Any]]:
        """初始化推理模板"""
        return {
            ReasoningStrategy.CHAIN_OF_THOUGHT: {
                "name": "链式思维推理",
                "description": "逐步分析问题，形成推理链",
                "steps": [
                    {
                        "type": ReasoningStep.PROBLEM_ANALYSIS,
                        "prompt_template": "首先，让我分析这个问题：{problem}\n\n问题的关键要素是什么？",
                        "required": True
                    },
                    {
                        "type": ReasoningStep.INFORMATION_GATHERING,
                        "prompt_template": "基于问题分析，我需要收集以下信息：\n{previous_analysis}\n\n相关的背景知识和事实是什么？",
                        "required": True
                    },
                    {
                        "type": ReasoningStep.LOGICAL_REASONING,
                        "prompt_template": "现在让我进行逻辑推理：\n问题：{problem}\n信息：{gathered_info}\n\n步骤：",
                        "required": True
                    },
                    {
                        "type": ReasoningStep.CONCLUSION_GENERATION,
                        "prompt_template": "基于以上分析和推理：{reasoning_steps}\n\n我的结论是：",
                        "required": True
                    },
                    {
                        "type": ReasoningStep.VALIDATION,
                        "prompt_template": "让我验证这个结论：{conclusion}\n\n这个结论是否合理？是否有遗漏或错误？",
                        "required": False
                    }
                ]
            },
            ReasoningStrategy.TREE_OF_THOUGHT: {
                "name": "树状思维推理",
                "description": "探索多个推理分支，选择最佳路径",
                "steps": [
                    {
                        "type": ReasoningStep.PROBLEM_ANALYSIS,
                        "prompt_template": "分析问题：{problem}\n\n可能的解决方案有哪些？",
                        "required": True
                    },
                    {
                        "type": ReasoningStep.HYPOTHESIS_GENERATION,
                        "prompt_template": "生成多个假设：\n问题：{problem}\n\n假设1：\n假设2：\n假设3：",
                        "required": True
                    },
                    {
                        "type": ReasoningStep.EVIDENCE_EVALUATION,
                        "prompt_template": "评估每个假设的证据：\n{hypotheses}\n\n证据分析：",
                        "required": True
                    },
                    {
                        "type": ReasoningStep.CONCLUSION_GENERATION,
                        "prompt_template": "选择最佳方案：\n{evidence_analysis}\n\n最终选择：",
                        "required": True
                    }
                ]
            },
            ReasoningStrategy.DEDUCTIVE: {
                "name": "演绎推理",
                "description": "从一般原理推导具体结论",
                "steps": [
                    {
                        "type": ReasoningStep.PROBLEM_ANALYSIS,
                        "prompt_template": "确定问题：{problem}\n\n适用的一般原理是什么？",
                        "required": True
                    },
                    {
                        "type": ReasoningStep.LOGICAL_REASONING,
                        "prompt_template": "应用演绎推理：\n大前提：{general_principle}\n小前提：{specific_case}\n\n结论：",
                        "required": True
                    },
                    {
                        "type": ReasoningStep.VALIDATION,
                        "prompt_template": "验证推理的有效性：{reasoning}\n\n逻辑是否成立？",
                        "required": True
                    }
                ]
            }
        }
    
    async def create_reasoning_workflow(
        self,
        name: str,
        problem: str,
        session_id: str,
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
        custom_steps: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """创建推理工作流"""
        
        workflow_id = str(uuid4())
        
        # 获取推理模板
        template = self.reasoning_templates.get(strategy)
        if not template:
            raise ValueError(f"Unsupported reasoning strategy: {strategy}")
        
        # 使用自定义步骤或默认模板
        steps_definition = custom_steps or template["steps"]
        
        # 创建推理链
        reasoning_chain = ReasoningChain(
            chain_id=str(uuid4()),
            session_id=session_id,
            strategy=strategy
        )
        
        # 创建工作流
        workflow = ReasoningWorkflow(
            workflow_id=workflow_id,
            name=name,
            description=f"基于{strategy.value}策略的推理工作流",
            session_id=session_id,
            strategy=strategy,
            steps_definition=steps_definition,
            reasoning_chain=reasoning_chain,
            metadata={
                "problem": problem,
                "template_name": template["name"]
            }
        )
        
        # 添加到活跃工作流
        self.active_workflows[workflow_id] = workflow
        
        # 存储到记忆系统
        await self._store_workflow_memory(workflow, "workflow_created")
        
        return workflow_id
    
    async def execute_workflow(self, workflow_id: str) -> ReasoningChain:
        """执行推理工作流"""
        
        if workflow_id not in self.active_workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.active_workflows[workflow_id]
        
        if workflow.status not in [WorkflowStatus.PENDING, WorkflowStatus.PAUSED]:
            raise ValueError(f"Workflow {workflow_id} is not in executable state")
        
        # 更新状态
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = utc_now()
        workflow.reasoning_chain.status = WorkflowStatus.RUNNING
        
        try:
            # 执行推理步骤
            await self._execute_reasoning_steps(workflow)
            
            # 计算总体置信度
            self._calculate_overall_confidence(workflow.reasoning_chain)
            
            # 更新状态
            workflow.status = WorkflowStatus.COMPLETED
            workflow.completed_at = utc_now()
            workflow.reasoning_chain.status = WorkflowStatus.COMPLETED
            workflow.reasoning_chain.completed_at = utc_now()
            
            # 移动到已完成工作流
            self.completed_workflows[workflow_id] = workflow
            del self.active_workflows[workflow_id]
            
            # 存储最终结果
            await self._store_workflow_memory(workflow, "workflow_completed")
            
            return workflow.reasoning_chain
            
        except Exception as e:
            # 更新失败状态
            workflow.status = WorkflowStatus.FAILED
            workflow.reasoning_chain.status = WorkflowStatus.FAILED
            workflow.metadata["error"] = str(e)
            
            # 存储错误信息
            await self._store_workflow_memory(workflow, "workflow_failed")
            
            raise e
    
    async def _execute_reasoning_steps(self, workflow: ReasoningWorkflow):
        """执行推理步骤"""
        
        problem = workflow.metadata["problem"]
        context = {"problem": problem}
        
        for i, step_def in enumerate(workflow.steps_definition):
            if i < workflow.current_step_index:
                continue  # 跳过已执行的步骤
            
            step_type = ReasoningStep(step_def["type"])
            prompt_template = step_def["prompt_template"]
            
            # 构建提示
            prompt = self._build_step_prompt(prompt_template, context, workflow.reasoning_chain)
            
            # 执行推理步骤
            step_result = await self._execute_reasoning_step(
                step_type, prompt, workflow.session_id
            )
            
            # 添加到推理链
            workflow.reasoning_chain.steps.append(step_result)
            workflow.reasoning_chain.total_execution_time_ms += step_result.execution_time_ms
            workflow.reasoning_chain.total_tokens_used += step_result.tokens_used
            
            # 更新上下文
            context.update({
                f"step_{len(workflow.reasoning_chain.steps)}": step_result.output_data,
                f"{step_type.value}_result": step_result.reasoning_text
            })
            
            # 更新当前步骤索引
            workflow.current_step_index = i + 1
            
            # 存储中间结果
            await self._store_step_memory(step_result, workflow.session_id)
            
            # 检查是否需要暂停
            if step_result.confidence_score < self.default_confidence_threshold:
                await self._handle_low_confidence(workflow, step_result)
        
        # 生成最终结论
        if workflow.reasoning_chain.steps:
            final_step = workflow.reasoning_chain.steps[-1]
            if final_step.step_type == ReasoningStep.CONCLUSION_GENERATION:
                workflow.reasoning_chain.final_conclusion = final_step.reasoning_text
    
    def _build_step_prompt(
        self, 
        template: str, 
        context: Dict[str, Any], 
        reasoning_chain: ReasoningChain
    ) -> str:
        """构建步骤提示"""
        
        # 收集之前的推理步骤
        previous_steps = []
        for step in reasoning_chain.steps:
            previous_steps.append(f"{step.step_type.value}: {step.reasoning_text}")
        
        # 扩展上下文
        enhanced_context = {
            **context,
            "previous_steps": "\n".join(previous_steps),
            "reasoning_history": reasoning_chain.steps,
            "step_count": len(reasoning_chain.steps)
        }
        
        # 处理可能的上下文引用
        enhanced_context.update({
            "previous_analysis": enhanced_context.get("step_1", {}).get("reasoning_text", ""),
            "gathered_info": enhanced_context.get("step_2", {}).get("reasoning_text", ""),
            "reasoning_steps": "\n".join(previous_steps),
            "conclusion": enhanced_context.get("step_4", {}).get("reasoning_text", ""),
            "hypotheses": enhanced_context.get("step_2", {}).get("reasoning_text", ""),
            "evidence_analysis": enhanced_context.get("step_3", {}).get("reasoning_text", ""),
            "general_principle": enhanced_context.get("step_1", {}).get("reasoning_text", ""),
            "specific_case": context.get("problem", ""),
            "reasoning": enhanced_context.get("step_2", {}).get("reasoning_text", "")
        })
        
        # 格式化模板
        try:
            return template.format(**enhanced_context)
        except KeyError as e:
            # 如果某些变量不存在，使用默认值
            safe_context = {k: v for k, v in enhanced_context.items() if isinstance(v, str)}
            return template.format_map(safe_context)
    
    async def _execute_reasoning_step(
        self, 
        step_type: ReasoningStep, 
        prompt: str, 
        session_id: str
    ) -> ReasoningStepResult:
        """执行单个推理步骤"""
        
        start_time = utc_now()
        
        # 创建推理请求
        request = InferenceRequest(
            request_id=str(uuid4()),
            model_type=ModelType.REASONING_MODEL,
            prompt=prompt,
            parameters={
                "step_type": step_type.value,
                "reasoning": True,
                "temperature": 0.3,  # 推理需要更稳定的输出
                "max_tokens": 512
            }
        )
        
        # 执行推理
        result = await self.inference_engine.infer(request)
        
        if not result:
            raise Exception(f"Failed to execute reasoning step: {step_type.value}")
        
        execution_time = (utc_now() - start_time).total_seconds() * 1000
        
        # 分析推理结果中的证据
        evidence = self._extract_evidence(result.response, result.reasoning_steps)
        
        return ReasoningStepResult(
            step_id=str(uuid4()),
            step_type=step_type,
            input_data={"prompt": prompt},
            output_data={
                "response": result.response,
                "reasoning_steps": result.reasoning_steps
            },
            reasoning_text=result.response,
            confidence_score=result.confidence_score or 0.8,
            execution_time_ms=execution_time,
            tokens_used=result.token_count,
            evidence=evidence,
            metadata={
                "model_used": result.model_used,
                "inference_time": result.inference_time_ms
            }
        )
    
    def _extract_evidence(
        self, 
        response: str, 
        reasoning_steps: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """从推理结果中提取证据"""
        
        evidence = []
        
        # 从推理步骤中提取证据
        if reasoning_steps:
            for step in reasoning_steps:
                if "evidence" in step.get("content", "").lower():
                    evidence.append({
                        "type": "reasoning_step",
                        "content": step.get("content", ""),
                        "confidence": 0.8
                    })
        
        # 从响应文本中提取关键论据
        import re
        
        # 查找"因为"、"由于"等关键词后的内容
        evidence_patterns = [
            r"因为([^。]+)",
            r"由于([^。]+)",
            r"根据([^。]+)",
            r"基于([^。]+)",
            r"鉴于([^。]+)"
        ]
        
        for pattern in evidence_patterns:
            matches = re.findall(pattern, response)
            for match in matches:
                evidence.append({
                    "type": "extracted_reasoning",
                    "content": match.strip(),
                    "confidence": 0.6
                })
        
        return evidence
    
    async def _handle_low_confidence(
        self, 
        workflow: ReasoningWorkflow, 
        step_result: ReasoningStepResult
    ):
        """处理低置信度步骤"""
        
        # 记录低置信度警告
        await self._store_workflow_memory(
            workflow, 
            "low_confidence_warning",
            {
                "step_type": step_result.step_type.value,
                "confidence": step_result.confidence_score,
                "threshold": self.default_confidence_threshold
            }
        )
        
        # 可以选择暂停工作流或继续执行
        # 这里我们选择继续，但记录警告
        warnings = workflow.metadata.setdefault("low_confidence_steps", [])
        warnings.append(
            {
                "step_id": step_result.step_id,
                "step_type": step_result.step_type.value,
                "confidence": step_result.confidence_score,
                "threshold": self.default_confidence_threshold,
                "timestamp": step_result.timestamp.isoformat(),
            }
        )
        step_result.metadata["low_confidence"] = True
    
    def _calculate_overall_confidence(self, reasoning_chain: ReasoningChain):
        """计算总体置信度"""
        
        if not reasoning_chain.steps:
            reasoning_chain.overall_confidence = 0.0
            return
        
        # 加权平均，最后的步骤权重更高
        total_weight = 0
        weighted_sum = 0
        
        for i, step in enumerate(reasoning_chain.steps):
            weight = (i + 1) / len(reasoning_chain.steps)  # 线性递增权重
            weighted_sum += step.confidence_score * weight
            total_weight += weight
        
        reasoning_chain.overall_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
    
    async def _store_workflow_memory(
        self, 
        workflow: ReasoningWorkflow, 
        event_type: str,
        extra_data: Optional[Dict[str, Any]] = None
    ):
        """存储工作流记忆"""
        
        memory_content = {
            "workflow_id": workflow.workflow_id,
            "event_type": event_type,
            "status": workflow.status.value,
            "strategy": workflow.strategy.value,
            "timestamp": utc_now().isoformat()
        }
        
        if extra_data:
            memory_content.update(extra_data)
        
        memory = MemoryEntry(
            id=str(uuid4()),
            session_id=workflow.session_id,
            memory_type=MemoryType.PROCEDURAL,
            content=json.dumps(memory_content, ensure_ascii=False),
            context={
                "workflow_id": workflow.workflow_id,
                "event_type": event_type
            },
            priority=MemoryPriority.MEDIUM,
            tags=["workflow", "reasoning", workflow.strategy.value]
        )
        
        self.memory_manager.store_memory(memory)
    
    async def _store_step_memory(
        self, 
        step_result: ReasoningStepResult, 
        session_id: str
    ):
        """存储步骤记忆"""
        
        memory = MemoryEntry(
            id=str(uuid4()),
            session_id=session_id,
            memory_type=MemoryType.FACTUAL,
            content=step_result.reasoning_text,
            context={
                "step_id": step_result.step_id,
                "step_type": step_result.step_type.value,
                "confidence": step_result.confidence_score
            },
            priority=MemoryPriority.HIGH if step_result.confidence_score > 0.8 else MemoryPriority.MEDIUM,
            tags=["reasoning_step", step_result.step_type.value]
        )
        
        self.memory_manager.store_memory(memory)
    
    # 公共API方法
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowStatus]:
        """获取工作流状态"""
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id].status
        elif workflow_id in self.completed_workflows:
            return self.completed_workflows[workflow_id].status
        return None
    
    def get_workflow_result(self, workflow_id: str) -> Optional[ReasoningChain]:
        """获取工作流结果"""
        if workflow_id in self.completed_workflows:
            return self.completed_workflows[workflow_id].reasoning_chain
        elif workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id].reasoning_chain
        return None
    
    def list_workflows(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出工作流"""
        workflows = []
        
        # 活跃工作流
        for workflow in self.active_workflows.values():
            if session_id is None or workflow.session_id == session_id:
                workflows.append({
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "status": workflow.status.value,
                    "strategy": workflow.strategy.value,
                    "created_at": workflow.created_at,
                    "current_step": workflow.current_step_index,
                    "total_steps": len(workflow.steps_definition)
                })
        
        # 已完成工作流
        for workflow in self.completed_workflows.values():
            if session_id is None or workflow.session_id == session_id:
                workflows.append({
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "status": workflow.status.value,
                    "strategy": workflow.strategy.value,
                    "created_at": workflow.created_at,
                    "completed_at": workflow.completed_at,
                    "final_confidence": workflow.reasoning_chain.overall_confidence if workflow.reasoning_chain else 0.0
                })
        
        return sorted(workflows, key=lambda x: x["created_at"], reverse=True)
    
    async def pause_workflow(self, workflow_id: str) -> bool:
        """暂停工作流"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            if workflow.status == WorkflowStatus.RUNNING:
                workflow.status = WorkflowStatus.PAUSED
                await self._store_workflow_memory(workflow, "workflow_paused")
                return True
        return False
    
    async def resume_workflow(self, workflow_id: str) -> bool:
        """恢复工作流"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            if workflow.status == WorkflowStatus.PAUSED:
                workflow.status = WorkflowStatus.PENDING
                await self._store_workflow_memory(workflow, "workflow_resumed")
                return True
        return False
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流"""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = WorkflowStatus.CANCELLED
            workflow.completed_at = utc_now()
            
            # 移动到已完成工作流
            self.completed_workflows[workflow_id] = workflow
            del self.active_workflows[workflow_id]
            
            await self._store_workflow_memory(workflow, "workflow_cancelled")
            return True
        return False
    
    def get_reasoning_statistics(self) -> Dict[str, Any]:
        """获取推理统计信息"""
        total_workflows = len(self.active_workflows) + len(self.completed_workflows)
        completed_count = len(self.completed_workflows)
        
        if completed_count > 0:
            avg_confidence = sum(
                w.reasoning_chain.overall_confidence 
                for w in self.completed_workflows.values()
                if w.reasoning_chain
            ) / completed_count
            
            avg_steps = sum(
                len(w.reasoning_chain.steps)
                for w in self.completed_workflows.values()
                if w.reasoning_chain
            ) / completed_count
        else:
            avg_confidence = 0.0
            avg_steps = 0.0
        
        return {
            "total_workflows": total_workflows,
            "active_workflows": len(self.active_workflows),
            "completed_workflows": completed_count,
            "average_confidence": avg_confidence,
            "average_steps_per_workflow": avg_steps,
            "strategies_used": list(set(
                w.strategy.value for w in 
                list(self.active_workflows.values()) + list(self.completed_workflows.values())
            ))
        }
