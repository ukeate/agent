"""决策过程记录器

本模块实现决策过程的结构化记录功能，包括：
- 决策路径跟踪
- 关键决策点记录
- 数据流程监控
- 决策分支管理
"""

import uuid
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, timezone
from typing import Any, Dict, List, Optional, Set, Union
from uuid import UUID
from src.models.schemas.explanation import (
    ExplanationType,
    EvidenceType,
    ExplanationComponent,
    DecisionExplanation
)

class DecisionNode:
    """决策节点 - 表示决策树中的一个节点"""
    
    def __init__(
        self,
        node_id: str,
        node_type: str,
        description: str,
        input_data: Dict[str, Any],
        parent_id: Optional[str] = None
    ):
        self.node_id = node_id
        self.node_type = node_type  # "condition", "action", "decision", "data_processing"
        self.description = description
        self.input_data = input_data
        self.output_data: Dict[str, Any] = {}
        self.parent_id = parent_id
        self.children: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.timestamp = utc_now()
        self.processing_time_ms: Optional[int] = None
        self.status = "pending"  # pending, processing, completed, failed
        
    def complete(self, output_data: Dict[str, Any], processing_time_ms: int = 0):
        """完成节点处理"""
        self.output_data = output_data
        self.processing_time_ms = processing_time_ms
        self.status = "completed"
        
    def add_child(self, child_id: str):
        """添加子节点"""
        if child_id not in self.children:
            self.children.append(child_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "description": self.description,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "parent_id": self.parent_id,
            "children": self.children,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "processing_time_ms": self.processing_time_ms,
            "status": self.status
        }

class DecisionTracker:
    """决策跟踪器 - 记录和管理决策过程"""
    
    def __init__(self, decision_id: str, decision_context: str = ""):
        self.decision_id = decision_id
        self.decision_context = decision_context
        self.nodes: Dict[str, DecisionNode] = {}
        self.root_node_id: Optional[str] = None
        self.current_node_id: Optional[str] = None
        self.decision_path: List[str] = []  # 记录决策路径
        self.data_sources: Set[str] = set()  # 记录数据来源
        self.processing_steps: List[Dict[str, Any]] = []  # 记录处理步骤
        self.created_at = utc_now()
        self.completed_at: Optional[datetime] = None
        self.final_decision: Optional[str] = None
        self.confidence_factors: List[Dict[str, Any]] = []
        
    def create_node(
        self,
        node_type: str,
        description: str,
        input_data: Dict[str, Any],
        parent_id: Optional[str] = None
    ) -> str:
        """创建新的决策节点"""
        node_id = f"{self.decision_id}_{len(self.nodes) + 1}_{uuid.uuid4().hex[:8]}"
        
        node = DecisionNode(
            node_id=node_id,
            node_type=node_type,
            description=description,
            input_data=input_data,
            parent_id=parent_id
        )
        
        self.nodes[node_id] = node
        
        # 设置根节点
        if not self.root_node_id:
            self.root_node_id = node_id
        
        # 建立父子关系
        if parent_id and parent_id in self.nodes:
            self.nodes[parent_id].add_child(node_id)
            
        # 更新当前节点和路径
        self.current_node_id = node_id
        self.decision_path.append(node_id)
        
        # 记录数据来源
        self._extract_data_sources(input_data)
        
        return node_id
    
    def complete_node(
        self,
        node_id: str,
        output_data: Dict[str, Any],
        processing_time_ms: int = 0,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """完成节点处理"""
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
            
        node = self.nodes[node_id]
        node.complete(output_data, processing_time_ms)
        
        if metadata:
            node.metadata.update(metadata)
            
        # 记录处理步骤
        self.processing_steps.append({
            "node_id": node_id,
            "step_type": node.node_type,
            "description": node.description,
            "processing_time_ms": processing_time_ms,
            "timestamp": utc_now().isoformat()
        })
        
        # 提取数据来源
        self._extract_data_sources(output_data)
    
    def add_decision_branch(
        self,
        condition: str,
        condition_result: bool,
        branch_description: str,
        parent_id: Optional[str] = None
    ) -> str:
        """添加决策分支"""
        parent_id = parent_id or self.current_node_id
        
        branch_node_id = self.create_node(
            node_type="decision_branch",
            description=f"Branch: {branch_description} (Condition: {condition} = {condition_result})",
            input_data={
                "condition": condition,
                "condition_result": condition_result,
                "branch_description": branch_description
            },
            parent_id=parent_id
        )
        
        return branch_node_id
    
    def record_data_processing(
        self,
        operation: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        processing_time_ms: int = 0,
        parent_id: Optional[str] = None
    ) -> str:
        """记录数据处理步骤"""
        parent_id = parent_id or self.current_node_id
        
        node_id = self.create_node(
            node_type="data_processing",
            description=f"Data Processing: {operation}",
            input_data=input_data,
            parent_id=parent_id
        )
        
        self.complete_node(node_id, output_data, processing_time_ms)
        return node_id
    
    def record_condition_evaluation(
        self,
        condition: str,
        evaluation_result: bool,
        evaluation_data: Dict[str, Any],
        parent_id: Optional[str] = None
    ) -> str:
        """记录条件评估"""
        parent_id = parent_id or self.current_node_id
        
        node_id = self.create_node(
            node_type="condition",
            description=f"Condition: {condition}",
            input_data=evaluation_data,
            parent_id=parent_id
        )
        
        output_data = {
            "condition": condition,
            "result": evaluation_result,
            "evaluation_details": evaluation_data
        }
        
        self.complete_node(node_id, output_data)
        return node_id
    
    def finalize_decision(
        self,
        final_decision: str,
        confidence_score: float,
        reasoning: str
    ):
        """最终化决策"""
        self.final_decision = final_decision
        self.completed_at = utc_now()
        
        # 添加最终决策节点
        final_node_id = self.create_node(
            node_type="final_decision",
            description="Final Decision",
            input_data={
                "decision": final_decision,
                "confidence": confidence_score,
                "reasoning": reasoning
            },
            parent_id=self.current_node_id
        )
        
        self.complete_node(final_node_id, {
            "decision": final_decision,
            "confidence": confidence_score,
            "reasoning": reasoning
        })
    
    def add_confidence_factor(
        self,
        factor_name: str,
        factor_value: Any,
        weight: float,
        impact: float,
        source: str
    ):
        """添加置信度因子"""
        self.confidence_factors.append({
            "factor_name": factor_name,
            "factor_value": factor_value,
            "weight": weight,
            "impact": impact,
            "source": source,
            "timestamp": utc_now().isoformat()
        })
    
    def get_decision_path(self) -> List[Dict[str, Any]]:
        """获取决策路径"""
        path_details = []
        for node_id in self.decision_path:
            if node_id in self.nodes:
                node = self.nodes[node_id]
                path_details.append({
                    "node_id": node_id,
                    "type": node.node_type,
                    "description": node.description,
                    "timestamp": node.timestamp.isoformat(),
                    "status": node.status,
                    "processing_time_ms": node.processing_time_ms
                })
        return path_details
    
    def get_decision_tree(self) -> Dict[str, Any]:
        """获取完整的决策树结构"""
        if not self.root_node_id:
            return {}
            
        def build_tree(node_id: str) -> Dict[str, Any]:
            node = self.nodes[node_id]
            tree_node = node.to_dict()
            tree_node["children"] = [
                build_tree(child_id) for child_id in node.children
            ]
            return tree_node
        
        return build_tree(self.root_node_id)
    
    def get_data_flow(self) -> Dict[str, Any]:
        """获取数据流信息"""
        return {
            "data_sources": list(self.data_sources),
            "processing_steps": self.processing_steps,
            "total_steps": len(self.processing_steps),
            "total_processing_time": sum(
                step.get("processing_time_ms", 0) for step in self.processing_steps
            )
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """获取决策过程摘要"""
        total_time = 0
        if self.completed_at and self.created_at:
            total_time = int((self.completed_at - self.created_at).total_seconds() * 1000)
        
        return {
            "decision_id": self.decision_id,
            "decision_context": self.decision_context,
            "final_decision": self.final_decision,
            "total_nodes": len(self.nodes),
            "decision_path_length": len(self.decision_path),
            "data_sources_count": len(self.data_sources),
            "processing_steps_count": len(self.processing_steps),
            "confidence_factors_count": len(self.confidence_factors),
            "total_time_ms": total_time,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": "completed" if self.completed_at else "in_progress"
        }
    
    def generate_explanation_components(self) -> List[ExplanationComponent]:
        """生成解释组件"""
        components = []
        
        # 从置信度因子生成组件
        for factor in self.confidence_factors:
            component = ExplanationComponent(
                factor_name=factor["factor_name"],
                factor_value=factor["factor_value"],
                weight=factor["weight"],
                impact_score=factor["impact"],
                evidence_type=EvidenceType.INPUT_DATA,
                evidence_source=factor["source"],
                evidence_content=f"因子 {factor['factor_name']} 的值为 {factor['factor_value']}",
                causal_relationship=f"该因子对决策的影响权重为 {factor['weight']}",
                metadata={
                    "timestamp": factor["timestamp"],
                    "decision_node": "confidence_factor"
                }
            )
            components.append(component)
        
        # 从关键决策节点生成组件  
        for node_id, node in self.nodes.items():
            if node.node_type in ["condition", "decision_branch", "final_decision"]:
                component = ExplanationComponent(
                    factor_name=f"decision_node_{node_id}",
                    factor_value=node.output_data.get("result", node.description),
                    weight=0.5,  # 默认权重
                    impact_score=0.0,  # 需要进一步计算
                    evidence_type=EvidenceType.REASONING_STEP,
                    evidence_source=f"decision_node_{node_id}",
                    evidence_content=node.description,
                    causal_relationship=f"决策节点：{node.description}",
                    metadata={
                        "node_type": node.node_type,
                        "processing_time_ms": node.processing_time_ms,
                        "timestamp": node.timestamp.isoformat()
                    }
                )
                components.append(component)
        
        return components
    
    def _extract_data_sources(self, data: Dict[str, Any]):
        """提取数据来源"""
        for key, value in data.items():
            if isinstance(value, str) and key.endswith("_source"):
                self.data_sources.add(value)
            elif key == "source":
                self.data_sources.add(str(value))
            elif isinstance(value, dict):
                self._extract_data_sources(value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "decision_id": self.decision_id,
            "decision_context": self.decision_context,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "root_node_id": self.root_node_id,
            "current_node_id": self.current_node_id,
            "decision_path": self.decision_path,
            "data_sources": list(self.data_sources),
            "processing_steps": self.processing_steps,
            "confidence_factors": self.confidence_factors,
            "final_decision": self.final_decision,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DecisionTracker':
        """从字典恢复DecisionTracker对象"""
        tracker = cls(data["decision_id"], data["decision_context"])
        
        # 恢复节点
        for node_id, node_data in data["nodes"].items():
            node = DecisionNode(
                node_id=node_data["node_id"],
                node_type=node_data["node_type"],
                description=node_data["description"],
                input_data=node_data["input_data"],
                parent_id=node_data["parent_id"]
            )
            node.output_data = node_data["output_data"]
            node.children = node_data["children"]
            node.metadata = node_data["metadata"]
            node.timestamp = datetime.fromisoformat(node_data["timestamp"])
            node.processing_time_ms = node_data["processing_time_ms"]
            node.status = node_data["status"]
            tracker.nodes[node_id] = node
        
        # 恢复其他属性
        tracker.root_node_id = data["root_node_id"]
        tracker.current_node_id = data["current_node_id"]
        tracker.decision_path = data["decision_path"]
        tracker.data_sources = set(data["data_sources"])
        tracker.processing_steps = data["processing_steps"]
        tracker.confidence_factors = data["confidence_factors"]
        tracker.final_decision = data["final_decision"]
        tracker.created_at = datetime.fromisoformat(data["created_at"])
        if data["completed_at"]:
            tracker.completed_at = datetime.fromisoformat(data["completed_at"])
        
        return tracker
