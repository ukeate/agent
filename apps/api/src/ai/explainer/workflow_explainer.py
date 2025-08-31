"""工作流决策解释器

本模块集成LangGraph工作流系统，为工作流执行过程提供详细的解释功能。
"""

import json
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from src.ai.openai_client import OpenAIClient
from src.ai.explainer.decision_tracker import DecisionTracker
from src.models.schemas.explanation import (
    DecisionExplanation,
    ExplanationComponent,
    ExplanationLevel,
    ExplanationType,
    EvidenceType,
    ConfidenceMetrics,
    CounterfactualScenario
)


class WorkflowNode:
    """工作流节点数据结构"""
    
    def __init__(
        self,
        node_id: str,
        node_type: str,
        node_name: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time: float,
        status: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.node_id = node_id
        self.node_type = node_type
        self.node_name = node_name
        self.input_data = input_data
        self.output_data = output_data
        self.execution_time = execution_time
        self.status = status
        self.metadata = metadata or {}
        self.timestamp = utc_now()


class WorkflowExecution:
    """工作流执行数据结构"""
    
    def __init__(self, workflow_id: str, workflow_name: str):
        self.workflow_id = workflow_id
        self.workflow_name = workflow_name
        self.nodes: List[WorkflowNode] = []
        self.edges: List[Dict[str, str]] = []
        self.execution_path: List[str] = []
        self.global_state: Dict[str, Any] = {}
        self.start_time = utc_now()
        self.end_time: Optional[datetime] = None
        self.status = "running"
        self.error_messages: List[str] = []
    
    def add_node_execution(self, node: WorkflowNode) -> None:
        """添加节点执行记录"""
        self.nodes.append(node)
        self.execution_path.append(node.node_id)
    
    def add_edge(self, from_node: str, to_node: str, condition: str = "") -> None:
        """添加边信息"""
        self.edges.append({
            "from": from_node,
            "to": to_node,
            "condition": condition
        })
    
    def update_global_state(self, updates: Dict[str, Any]) -> None:
        """更新全局状态"""
        self.global_state.update(updates)
    
    def complete_execution(self, status: str = "completed") -> None:
        """完成执行"""
        self.status = status
        self.end_time = utc_now()
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """获取执行摘要"""
        total_time = 0.0
        if self.end_time:
            total_time = (self.end_time - self.start_time).total_seconds()
        
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "status": self.status,
            "total_nodes": len(self.nodes),
            "execution_path_length": len(self.execution_path),
            "total_execution_time": total_time,
            "average_node_time": sum(node.execution_time for node in self.nodes) / len(self.nodes) if self.nodes else 0,
            "error_count": len(self.error_messages),
            "state_variables": len(self.global_state)
        }


class WorkflowExplainer:
    """工作流解释器"""
    
    def __init__(self, openai_client: Optional[OpenAIClient] = None):
        """初始化工作流解释器"""
        self.openai_client = openai_client or OpenAIClient()
        
        # 工作流解释配置
        self.explanation_config = {
            "model": "gpt-4o-mini",
            "temperature": 0.2,
            "max_tokens": 3000,
            "top_p": 0.9
        }
        
        # 节点类型解释映射
        self.node_type_explanations = {
            "agent": "智能体执行节点",
            "tool": "工具调用节点",
            "condition": "条件判断节点",
            "router": "路由选择节点",
            "aggregator": "数据聚合节点",
            "processor": "数据处理节点",
            "validator": "数据验证节点",
            "transformer": "数据转换节点",
            "checkpoint": "检查点节点",
            "human": "人工干预节点"
        }
        
        # 工作流模式解释
        self.workflow_patterns = {
            "sequential": "顺序执行模式",
            "parallel": "并行执行模式",
            "conditional": "条件分支模式",
            "loop": "循环执行模式",
            "graph": "图状执行模式",
            "pipeline": "管道处理模式"
        }
    
    def generate_workflow_explanation(
        self,
        workflow_execution: WorkflowExecution,
        decision_tracker: Optional[DecisionTracker] = None,
        explanation_level: ExplanationLevel = ExplanationLevel.DETAILED
    ) -> DecisionExplanation:
        """生成工作流解释"""
        
        try:
            # 1. 分析工作流执行
            workflow_analysis = self._analyze_workflow_execution(workflow_execution)
            
            # 2. 生成节点解释
            node_explanations = self._generate_node_explanations(
                workflow_execution.nodes, explanation_level
            )
            
            # 3. 分析执行路径
            path_analysis = self._analyze_execution_path(
                workflow_execution.execution_path, workflow_execution.nodes
            )
            
            # 4. 生成状态变化解释
            state_analysis = self._analyze_state_changes(
                workflow_execution.global_state, workflow_execution.nodes
            )
            
            # 5. 计算工作流置信度
            confidence_metrics = self._calculate_workflow_confidence(
                workflow_execution, workflow_analysis
            )
            
            # 6. 生成解释文本
            explanation_texts = self._generate_workflow_explanation_texts(
                workflow_execution, workflow_analysis, explanation_level
            )
            
            # 7. 创建解释组件
            components = self._create_workflow_components(
                workflow_execution, node_explanations, path_analysis
            )
            
            # 8. 生成反事实场景
            counterfactuals = self._generate_workflow_counterfactuals(
                workflow_execution, workflow_analysis
            )
            
            # 9. 生成可视化数据
            visualization_data = self._generate_workflow_visualization(
                workflow_execution, workflow_analysis
            )
            
            # 10. 创建解释对象
            explanation = DecisionExplanation(
                id=uuid4(),
                decision_id=workflow_execution.workflow_id,
                explanation_type=ExplanationType.WORKFLOW,
                explanation_level=explanation_level,
                decision_description=f"工作流执行: {workflow_execution.workflow_name}",
                decision_outcome=f"执行{workflow_execution.status}",
                decision_context=self._format_workflow_context(workflow_execution),
                summary_explanation=explanation_texts["summary"],
                detailed_explanation=explanation_texts.get("detailed"),
                technical_explanation=explanation_texts.get("technical"),
                components=components,
                confidence_metrics=confidence_metrics,
                counterfactuals=counterfactuals,
                visualization_data=visualization_data,
                metadata={
                    "workflow_type": "langgraph_workflow",
                    "total_nodes": len(workflow_execution.nodes),
                    "execution_time": workflow_analysis.get("total_execution_time", 0),
                    "workflow_pattern": self._identify_workflow_pattern(workflow_execution),
                    "generation_timestamp": utc_now().isoformat()
                }
            )
            
            return explanation
            
        except Exception as e:
            # 降级处理
            return self._create_fallback_workflow_explanation(workflow_execution, str(e))
    
    def _analyze_workflow_execution(self, workflow_execution: WorkflowExecution) -> Dict[str, Any]:
        """分析工作流执行"""
        
        summary = workflow_execution.get_execution_summary()
        
        # 分析节点性能
        node_performance = {}
        if workflow_execution.nodes:
            execution_times = [node.execution_time for node in workflow_execution.nodes]
            node_performance = {
                "fastest_node": min(execution_times),
                "slowest_node": max(execution_times),
                "average_time": sum(execution_times) / len(execution_times),
                "total_time": sum(execution_times)
            }
        
        # 分析状态变化
        state_changes = len(workflow_execution.global_state)
        
        # 分析错误情况
        error_analysis = {
            "error_count": len(workflow_execution.error_messages),
            "error_rate": len(workflow_execution.error_messages) / len(workflow_execution.nodes) if workflow_execution.nodes else 0,
            "has_errors": len(workflow_execution.error_messages) > 0
        }
        
        # 分析执行效率
        efficiency_metrics = {
            "node_density": len(workflow_execution.nodes) / max(1, len(workflow_execution.execution_path)),
            "state_efficiency": state_changes / max(1, len(workflow_execution.nodes)),
            "success_rate": 1.0 - error_analysis["error_rate"]
        }
        
        return {
            **summary,
            "node_performance": node_performance,
            "state_changes": state_changes,
            "error_analysis": error_analysis,
            "efficiency_metrics": efficiency_metrics
        }
    
    def _generate_node_explanations(
        self, 
        nodes: List[WorkflowNode], 
        explanation_level: ExplanationLevel
    ) -> List[Dict[str, Any]]:
        """生成节点解释"""
        
        explanations = []
        
        for node in nodes:
            node_explanation = {
                "node_id": node.node_id,
                "node_type": node.node_type,
                "node_name": node.node_name,
                "type_description": self.node_type_explanations.get(
                    node.node_type, "未知节点类型"
                ),
                "execution_time": node.execution_time,
                "status": node.status,
                "input_summary": self._summarize_data(node.input_data),
                "output_summary": self._summarize_data(node.output_data),
                "performance_rating": self._rate_node_performance(node)
            }
            
            # 根据解释级别添加详细信息
            if explanation_level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]:
                node_explanation.update({
                    "detailed_description": self._generate_node_detailed_description(node),
                    "input_analysis": self._analyze_node_input(node.input_data),
                    "output_analysis": self._analyze_node_output(node.output_data)
                })
            
            if explanation_level == ExplanationLevel.TECHNICAL:
                node_explanation.update({
                    "technical_details": node.metadata,
                    "execution_metrics": {
                        "execution_time": node.execution_time,
                        "timestamp": node.timestamp.isoformat(),
                        "status_code": node.status
                    }
                })
            
            explanations.append(node_explanation)
        
        return explanations
    
    def _analyze_execution_path(
        self, 
        execution_path: List[str], 
        nodes: List[WorkflowNode]
    ) -> Dict[str, Any]:
        """分析执行路径"""
        
        # 构建节点映射
        node_map = {node.node_id: node for node in nodes}
        
        # 分析路径特征
        path_analysis = {
            "path_length": len(execution_path),
            "unique_nodes": len(set(execution_path)),
            "revisited_nodes": len(execution_path) - len(set(execution_path)),
            "has_loops": len(execution_path) != len(set(execution_path))
        }
        
        # 分析节点类型分布
        node_types = [node_map[node_id].node_type for node_id in execution_path if node_id in node_map]
        type_distribution = {}
        for node_type in node_types:
            type_distribution[node_type] = type_distribution.get(node_type, 0) + 1
        
        path_analysis["type_distribution"] = type_distribution
        
        # 分析关键路径节点
        if nodes:
            execution_times = {node.node_id: node.execution_time for node in nodes}
            bottleneck_nodes = sorted(
                execution_path, 
                key=lambda x: execution_times.get(x, 0), 
                reverse=True
            )[:3]
            path_analysis["bottleneck_nodes"] = bottleneck_nodes
        
        return path_analysis
    
    def _analyze_state_changes(
        self, 
        global_state: Dict[str, Any], 
        nodes: List[WorkflowNode]
    ) -> Dict[str, Any]:
        """分析状态变化"""
        
        state_analysis = {
            "total_state_variables": len(global_state),
            "state_complexity": self._calculate_state_complexity(global_state),
            "state_size": len(str(global_state))
        }
        
        # 分析状态变化模式
        state_changes_per_node = []
        for node in nodes:
            if "state_changes" in node.metadata:
                state_changes_per_node.append(len(node.metadata["state_changes"]))
            else:
                state_changes_per_node.append(0)
        
        if state_changes_per_node:
            state_analysis.update({
                "average_changes_per_node": sum(state_changes_per_node) / len(state_changes_per_node),
                "max_changes_in_single_node": max(state_changes_per_node),
                "nodes_with_state_changes": sum(1 for x in state_changes_per_node if x > 0)
            })
        
        return state_analysis
    
    def _calculate_workflow_confidence(
        self, 
        workflow_execution: WorkflowExecution, 
        workflow_analysis: Dict[str, Any]
    ) -> ConfidenceMetrics:
        """计算工作流置信度"""
        
        from src.models.schemas.explanation import ConfidenceSource
        
        # 基于执行成功率计算基础置信度
        success_rate = workflow_analysis["efficiency_metrics"]["success_rate"]
        base_confidence = success_rate
        
        # 基于性能指标调整
        if workflow_analysis.get("node_performance"):
            perf = workflow_analysis["node_performance"]
            if perf["average_time"] < 1.0:  # 快速执行加分
                base_confidence = min(1.0, base_confidence + 0.1)
            elif perf["average_time"] > 5.0:  # 慢速执行减分
                base_confidence = max(0.1, base_confidence - 0.1)
        
        # 基于错误率调整
        error_rate = workflow_analysis["error_analysis"]["error_rate"]
        base_confidence = base_confidence * (1.0 - error_rate * 0.5)
        
        # 计算不确定性
        uncertainty_score = 1.0 - base_confidence
        
        # 计算置信区间
        confidence_lower = max(0.0, base_confidence - 0.15)
        confidence_upper = min(1.0, base_confidence + 0.15)
        
        return ConfidenceMetrics(
            overall_confidence=base_confidence,
            uncertainty_score=uncertainty_score,
            confidence_interval_lower=confidence_lower,
            confidence_interval_upper=confidence_upper,
            confidence_sources=[ConfidenceSource.EXECUTION_SUCCESS],
            model_confidence=base_confidence,
            evidence_confidence=success_rate,
            metadata={
                "workflow_success_rate": success_rate,
                "error_rate": error_rate,
                "performance_factor": workflow_analysis.get("node_performance", {}),
                "confidence_source": "workflow_execution_analysis"
            }
        )
    
    def _generate_workflow_explanation_texts(
        self,
        workflow_execution: WorkflowExecution,
        workflow_analysis: Dict[str, Any],
        explanation_level: ExplanationLevel
    ) -> Dict[str, str]:
        """生成工作流解释文本"""
        
        texts = {}
        
        # 生成概要解释
        summary_prompt = self._build_workflow_explanation_prompt(
            workflow_execution, workflow_analysis, ExplanationLevel.SUMMARY
        )
        texts["summary"] = self._call_openai_for_explanation(summary_prompt)
        
        # 生成详细解释
        if explanation_level in [ExplanationLevel.DETAILED, ExplanationLevel.TECHNICAL]:
            detailed_prompt = self._build_workflow_explanation_prompt(
                workflow_execution, workflow_analysis, ExplanationLevel.DETAILED
            )
            texts["detailed"] = self._call_openai_for_explanation(detailed_prompt)
        
        # 生成技术解释
        if explanation_level == ExplanationLevel.TECHNICAL:
            technical_prompt = self._build_workflow_explanation_prompt(
                workflow_execution, workflow_analysis, ExplanationLevel.TECHNICAL
            )
            texts["technical"] = self._call_openai_for_explanation(technical_prompt)
        
        return texts
    
    def _build_workflow_explanation_prompt(
        self,
        workflow_execution: WorkflowExecution,
        workflow_analysis: Dict[str, Any],
        level: ExplanationLevel
    ) -> str:
        """构建工作流解释提示"""
        
        prompt = f"""作为工作流分析专家，请对以下工作流执行进行{level.value}级别的解释。

工作流信息：
- 工作流名称：{workflow_execution.workflow_name}
- 执行状态：{workflow_execution.status}
- 节点数量：{len(workflow_execution.nodes)}
- 执行路径长度：{len(workflow_execution.execution_path)}
- 总执行时间：{workflow_analysis.get('total_execution_time', 0):.2f}秒

节点执行摘要：
{self._format_nodes_summary(workflow_execution.nodes)}

执行路径：
{' -> '.join(workflow_execution.execution_path)}

性能分析：
{self._format_performance_analysis(workflow_analysis.get('node_performance', {}))}

错误分析：
{self._format_error_analysis(workflow_analysis.get('error_analysis', {}))}

请生成解释，包括：
1. 工作流执行概况
2. 关键节点和路径分析
3. 性能和效率评估
4. 可能的优化建议
"""
        
        # 根据解释层次添加特定要求
        if level == ExplanationLevel.SUMMARY:
            prompt += "\n要求：简洁明了，控制在150字以内，突出执行结果和关键指标。"
        elif level == ExplanationLevel.DETAILED:
            prompt += "\n要求：详细分析执行过程，包含节点分析和路径解释，控制在500字以内。"
        elif level == ExplanationLevel.TECHNICAL:
            prompt += "\n要求：技术性分析，包含性能指标、技术细节和优化建议，控制在800字以内。"
        
        return prompt
    
    def _call_openai_for_explanation(self, prompt: str) -> str:
        """调用OpenAI API生成解释"""
        try:
            response = self.openai_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                **self.explanation_config
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"工作流解释生成失败，请稍后重试。错误：{str(e)}"
    
    def _create_workflow_components(
        self,
        workflow_execution: WorkflowExecution,
        node_explanations: List[Dict[str, Any]],
        path_analysis: Dict[str, Any]
    ) -> List[ExplanationComponent]:
        """创建工作流组件"""
        
        components = []
        
        # 为关键节点创建组件
        key_nodes = sorted(
            workflow_execution.nodes,
            key=lambda x: x.execution_time,
            reverse=True
        )[:5]  # 取执行时间最长的5个节点
        
        for i, node in enumerate(key_nodes):
            node_explanation = next(
                (exp for exp in node_explanations if exp["node_id"] == node.node_id),
                {}
            )
            
            component = ExplanationComponent(
                factor_name=f"节点: {node.node_name}",
                factor_value=f"执行时间: {node.execution_time:.3f}s",
                weight=node.execution_time / sum(n.execution_time for n in workflow_execution.nodes) if workflow_execution.nodes else 0,
                impact_score=node_explanation.get("performance_rating", 0.5),
                evidence_type=EvidenceType.WORKFLOW_EXECUTION,
                evidence_source="workflow_node",
                evidence_content=node_explanation.get("detailed_description", f"节点{node.node_name}执行"),
                causal_relationship=f"节点{node.node_name}影响工作流整体性能",
                metadata={
                    "node_id": node.node_id,
                    "node_type": node.node_type,
                    "execution_time": node.execution_time,
                    "status": node.status
                }
            )
            components.append(component)
        
        # 添加执行路径组件
        if path_analysis.get("has_loops"):
            component = ExplanationComponent(
                factor_name="执行路径特征",
                factor_value=f"包含循环，重访节点{path_analysis['revisited_nodes']}次",
                weight=0.3,
                impact_score=0.7 if path_analysis["revisited_nodes"] < 3 else 0.4,
                evidence_type=EvidenceType.WORKFLOW_EXECUTION,
                evidence_source="execution_path",
                evidence_content="工作流执行路径包含循环，可能影响执行效率",
                causal_relationship="执行路径复杂度影响工作流性能",
                metadata=path_analysis
            )
            components.append(component)
        
        return components
    
    def _generate_workflow_counterfactuals(
        self,
        workflow_execution: WorkflowExecution,
        workflow_analysis: Dict[str, Any]
    ) -> List[CounterfactualScenario]:
        """生成工作流反事实场景"""
        
        scenarios = []
        
        # 基于性能瓶颈生成反事实
        if workflow_analysis.get("node_performance"):
            bottleneck_nodes = workflow_analysis.get("efficiency_metrics", {}).get("bottleneck_nodes", [])
            
            for node_id in bottleneck_nodes[:2]:  # 取前2个瓶颈节点
                node = next((n for n in workflow_execution.nodes if n.node_id == node_id), None)
                if node:
                    scenario = CounterfactualScenario(
                        scenario_name=f"优化节点{node.node_name}性能",
                        changed_factors={node.node_name: "执行时间减半"},
                        predicted_outcome="工作流整体执行时间显著缩短",
                        probability=0.8,
                        impact_difference=0.3,
                        explanation=f"如果{node.node_name}执行时间减半，预计工作流整体性能提升30%"
                    )
                    scenarios.append(scenario)
        
        # 基于错误情况生成反事实
        if workflow_analysis["error_analysis"]["has_errors"]:
            scenario = CounterfactualScenario(
                scenario_name="消除执行错误",
                changed_factors={"error_handling": "完善错误处理机制"},
                predicted_outcome="工作流执行成功率提升至100%",
                probability=0.9,
                impact_difference=0.2,
                explanation="完善错误处理机制可显著提升工作流可靠性"
            )
            scenarios.append(scenario)
        
        return scenarios
    
    def _generate_workflow_visualization(
        self,
        workflow_execution: WorkflowExecution,
        workflow_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生成工作流可视化数据"""
        
        return {
            "workflow_graph": {
                "chart_type": "graph",
                "nodes": [
                    {
                        "id": node.node_id,
                        "label": node.node_name,
                        "type": node.node_type,
                        "execution_time": node.execution_time,
                        "status": node.status
                    }
                    for node in workflow_execution.nodes
                ],
                "edges": workflow_execution.edges
            },
            "execution_timeline": {
                "chart_type": "timeline",
                "data": [
                    {
                        "node_id": node.node_id,
                        "node_name": node.node_name,
                        "start_time": node.timestamp.isoformat(),
                        "duration": node.execution_time,
                        "status": node.status
                    }
                    for node in workflow_execution.nodes
                ]
            },
            "performance_metrics": {
                "chart_type": "bar",
                "data": [
                    {
                        "node_name": node.node_name,
                        "execution_time": node.execution_time,
                        "performance_score": self._rate_node_performance(node)
                    }
                    for node in workflow_execution.nodes
                ]
            },
            "state_evolution": {
                "chart_type": "area",
                "data": {
                    "state_variables": len(workflow_execution.global_state),
                    "complexity_score": self._calculate_state_complexity(workflow_execution.global_state)
                }
            }
        }
    
    def _create_fallback_workflow_explanation(
        self,
        workflow_execution: WorkflowExecution,
        error: str
    ) -> DecisionExplanation:
        """创建降级工作流解释"""
        
        return DecisionExplanation(
            id=uuid4(),
            decision_id=workflow_execution.workflow_id,
            explanation_type=ExplanationType.WORKFLOW,
            explanation_level=ExplanationLevel.SUMMARY,
            decision_description=f"工作流执行: {workflow_execution.workflow_name}",
            decision_outcome=f"执行{workflow_execution.status}",
            summary_explanation=f"工作流解释生成失败: {error}",
            components=[],
            confidence_metrics=ConfidenceMetrics(
                overall_confidence=0.3,
                uncertainty_score=0.9,
                confidence_sources=[]
            ),
            counterfactuals=[],
            metadata={
                "error": error,
                "fallback_mode": True
            }
        )
    
    # 辅助方法
    def _format_workflow_context(self, workflow_execution: WorkflowExecution) -> str:
        """格式化工作流上下文"""
        
        context_parts = [
            f"工作流: {workflow_execution.workflow_name}",
            f"节点数: {len(workflow_execution.nodes)}",
            f"状态: {workflow_execution.status}"
        ]
        
        if workflow_execution.end_time:
            total_time = (workflow_execution.end_time - workflow_execution.start_time).total_seconds()
            context_parts.append(f"执行时间: {total_time:.2f}s")
        
        return " | ".join(context_parts)
    
    def _identify_workflow_pattern(self, workflow_execution: WorkflowExecution) -> str:
        """识别工作流模式"""
        
        # 简单的模式识别逻辑
        if len(set(workflow_execution.execution_path)) != len(workflow_execution.execution_path):
            return "loop"
        elif len(workflow_execution.edges) > len(workflow_execution.nodes):
            return "graph" 
        elif all(len([e for e in workflow_execution.edges if e["from"] == node.node_id]) <= 1 
                 for node in workflow_execution.nodes):
            return "sequential"
        else:
            return "conditional"
    
    def _summarize_data(self, data: Dict[str, Any]) -> str:
        """总结数据内容"""
        if not data:
            return "无数据"
        
        summary_parts = []
        for key, value in list(data.items())[:3]:  # 最多显示3个键
            if isinstance(value, (list, dict)):
                summary_parts.append(f"{key}: {len(value)}项")
            else:
                summary_parts.append(f"{key}: {str(value)[:20]}...")
        
        if len(data) > 3:
            summary_parts.append(f"...等{len(data)}个字段")
        
        return " | ".join(summary_parts)
    
    def _rate_node_performance(self, node: WorkflowNode) -> float:
        """评估节点性能"""
        
        # 基于执行时间和状态评估性能
        base_score = 0.8
        
        # 根据执行时间调整
        if node.execution_time < 0.1:
            base_score += 0.2
        elif node.execution_time > 2.0:
            base_score -= 0.3
        
        # 根据状态调整
        if node.status == "completed":
            base_score += 0.1
        elif node.status == "failed":
            base_score -= 0.5
        elif node.status == "timeout":
            base_score -= 0.4
        
        return max(0.0, min(1.0, base_score))
    
    def _generate_node_detailed_description(self, node: WorkflowNode) -> str:
        """生成节点详细描述"""
        
        type_desc = self.node_type_explanations.get(node.node_type, "未知类型")
        status_desc = "成功完成" if node.status == "completed" else f"状态: {node.status}"
        
        return f"{type_desc}节点'{node.node_name}'{status_desc}，执行时间{node.execution_time:.3f}秒"
    
    def _analyze_node_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析节点输入"""
        return {
            "field_count": len(input_data),
            "data_size": len(str(input_data)),
            "has_complex_data": any(isinstance(v, (list, dict)) for v in input_data.values())
        }
    
    def _analyze_node_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析节点输出"""
        return {
            "field_count": len(output_data),
            "data_size": len(str(output_data)),
            "has_complex_data": any(isinstance(v, (list, dict)) for v in output_data.values())
        }
    
    def _calculate_state_complexity(self, state: Dict[str, Any]) -> float:
        """计算状态复杂度"""
        
        complexity = 0.0
        
        for value in state.values():
            if isinstance(value, dict):
                complexity += 1.0 + len(value) * 0.1
            elif isinstance(value, list):
                complexity += 0.5 + len(value) * 0.05
            else:
                complexity += 0.1
        
        return complexity
    
    def _format_nodes_summary(self, nodes: List[WorkflowNode]) -> str:
        """格式化节点摘要"""
        if not nodes:
            return "无节点执行"
        
        summary_lines = []
        for node in nodes[:5]:  # 最多显示5个节点
            summary_lines.append(
                f"- {node.node_name} ({node.node_type}): {node.execution_time:.3f}s [{node.status}]"
            )
        
        if len(nodes) > 5:
            summary_lines.append(f"...等{len(nodes)}个节点")
        
        return "\n".join(summary_lines)
    
    def _format_performance_analysis(self, performance: Dict[str, Any]) -> str:
        """格式化性能分析"""
        if not performance:
            return "无性能数据"
        
        return f"""- 最快节点: {performance.get('fastest_node', 0):.3f}s
- 最慢节点: {performance.get('slowest_node', 0):.3f}s
- 平均执行时间: {performance.get('average_time', 0):.3f}s
- 总执行时间: {performance.get('total_time', 0):.3f}s"""
    
    def _format_error_analysis(self, error_analysis: Dict[str, Any]) -> str:
        """格式化错误分析"""
        if not error_analysis:
            return "无错误数据"
        
        return f"""- 错误数量: {error_analysis.get('error_count', 0)}
- 错误率: {error_analysis.get('error_rate', 0):.1%}
- 是否有错误: {'是' if error_analysis.get('has_errors', False) else '否'}"""