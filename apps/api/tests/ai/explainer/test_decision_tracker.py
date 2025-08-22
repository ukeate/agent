"""决策跟踪器单元测试"""

import pytest
from datetime import datetime, timezone
from unittest.mock import patch

from src.ai.explainer.decision_tracker import DecisionTracker, DecisionNode
from src.models.schemas.explanation import EvidenceType


class TestDecisionNode:
    """测试决策节点"""
    
    def test_create_decision_node(self):
        """测试创建决策节点"""
        node = DecisionNode(
            node_id="test_node_1",
            node_type="condition",
            description="Test condition",
            input_data={"test_input": "value"}
        )
        
        assert node.node_id == "test_node_1"
        assert node.node_type == "condition"
        assert node.description == "Test condition"
        assert node.input_data == {"test_input": "value"}
        assert node.status == "pending"
        assert len(node.children) == 0
    
    def test_complete_node(self):
        """测试完成节点"""
        node = DecisionNode(
            node_id="test_node_1",
            node_type="action",
            description="Test action",
            input_data={}
        )
        
        output_data = {"result": "success"}
        processing_time = 100
        
        node.complete(output_data, processing_time)
        
        assert node.output_data == output_data
        assert node.processing_time_ms == processing_time
        assert node.status == "completed"
    
    def test_add_child(self):
        """测试添加子节点"""
        node = DecisionNode(
            node_id="parent",
            node_type="decision",
            description="Parent node",
            input_data={}
        )
        
        node.add_child("child_1")
        node.add_child("child_2")
        node.add_child("child_1")  # 重复添加
        
        assert len(node.children) == 2
        assert "child_1" in node.children
        assert "child_2" in node.children
    
    def test_to_dict(self):
        """测试转换为字典"""
        node = DecisionNode(
            node_id="test_node",
            node_type="data_processing",
            description="Test processing",
            input_data={"input": "test"}
        )
        
        node.complete({"output": "processed"}, 50)
        node.add_child("child_1")
        
        node_dict = node.to_dict()
        
        assert node_dict["node_id"] == "test_node"
        assert node_dict["node_type"] == "data_processing"
        assert node_dict["description"] == "Test processing"
        assert node_dict["input_data"] == {"input": "test"}
        assert node_dict["output_data"] == {"output": "processed"}
        assert node_dict["processing_time_ms"] == 50
        assert node_dict["status"] == "completed"
        assert "child_1" in node_dict["children"]


class TestDecisionTracker:
    """测试决策跟踪器"""
    
    @pytest.fixture
    def tracker(self):
        """创建测试用的决策跟踪器"""
        return DecisionTracker("test_decision_001", "测试决策上下文")
    
    def test_create_decision_tracker(self, tracker):
        """测试创建决策跟踪器"""
        assert tracker.decision_id == "test_decision_001"
        assert tracker.decision_context == "测试决策上下文"
        assert len(tracker.nodes) == 0
        assert tracker.root_node_id is None
        assert tracker.current_node_id is None
        assert len(tracker.decision_path) == 0
        assert len(tracker.data_sources) == 0
    
    def test_create_node(self, tracker):
        """测试创建节点"""
        node_id = tracker.create_node(
            node_type="condition",
            description="Test condition",
            input_data={"test": "value"}
        )
        
        assert node_id in tracker.nodes
        assert tracker.root_node_id == node_id
        assert tracker.current_node_id == node_id
        assert len(tracker.decision_path) == 1
        assert tracker.decision_path[0] == node_id
        
        node = tracker.nodes[node_id]
        assert node.node_type == "condition"
        assert node.description == "Test condition"
    
    def test_create_child_node(self, tracker):
        """测试创建子节点"""
        parent_id = tracker.create_node(
            node_type="decision",
            description="Parent decision",
            input_data={}
        )
        
        child_id = tracker.create_node(
            node_type="action",
            description="Child action",
            input_data={"parent_result": "yes"},
            parent_id=parent_id
        )
        
        assert len(tracker.nodes) == 2
        assert child_id in tracker.nodes[parent_id].children
        assert tracker.nodes[child_id].parent_id == parent_id
    
    def test_complete_node(self, tracker):
        """测试完成节点"""
        node_id = tracker.create_node(
            node_type="data_processing",
            description="Process data",
            input_data={"raw_data": [1, 2, 3]}
        )
        
        output_data = {"processed_data": [2, 4, 6]}
        processing_time = 150
        metadata = {"algorithm": "multiply_by_2"}
        
        tracker.complete_node(node_id, output_data, processing_time, metadata)
        
        node = tracker.nodes[node_id]
        assert node.output_data == output_data
        assert node.processing_time_ms == processing_time
        assert node.metadata["algorithm"] == "multiply_by_2"
        assert node.status == "completed"
        
        # 检查处理步骤记录
        assert len(tracker.processing_steps) == 1
        step = tracker.processing_steps[0]
        assert step["node_id"] == node_id
        assert step["step_type"] == "data_processing"
        assert step["processing_time_ms"] == processing_time
    
    def test_add_decision_branch(self, tracker):
        """测试添加决策分支"""
        parent_id = tracker.create_node(
            node_type="decision",
            description="Main decision",
            input_data={}
        )
        
        branch_id = tracker.add_decision_branch(
            condition="age > 18",
            condition_result=True,
            branch_description="Adult branch",
            parent_id=parent_id
        )
        
        assert branch_id in tracker.nodes
        assert branch_id in tracker.nodes[parent_id].children
        
        branch_node = tracker.nodes[branch_id]
        assert branch_node.node_type == "decision_branch"
        assert "age > 18" in branch_node.description
        assert branch_node.input_data["condition"] == "age > 18"
        assert branch_node.input_data["condition_result"] is True
    
    def test_record_data_processing(self, tracker):
        """测试记录数据处理"""
        input_data = {"numbers": [1, 2, 3, 4, 5]}
        output_data = {"sum": 15, "average": 3.0}
        
        node_id = tracker.record_data_processing(
            operation="calculate_statistics",
            input_data=input_data,
            output_data=output_data,
            processing_time_ms=50
        )
        
        assert node_id in tracker.nodes
        node = tracker.nodes[node_id]
        assert node.node_type == "data_processing"
        assert "calculate_statistics" in node.description
        assert node.output_data == output_data
        assert node.processing_time_ms == 50
        assert node.status == "completed"
    
    def test_record_condition_evaluation(self, tracker):
        """测试记录条件评估"""
        evaluation_data = {"user_age": 25, "threshold": 18}
        
        node_id = tracker.record_condition_evaluation(
            condition="user_age > threshold",
            evaluation_result=True,
            evaluation_data=evaluation_data
        )
        
        assert node_id in tracker.nodes
        node = tracker.nodes[node_id]
        assert node.node_type == "condition"
        assert "user_age > threshold" in node.description
        assert node.output_data["result"] is True
        assert node.output_data["condition"] == "user_age > threshold"
    
    def test_finalize_decision(self, tracker):
        """测试最终化决策"""
        # 创建一些节点
        tracker.create_node("decision", "Initial decision", {})
        
        tracker.finalize_decision(
            final_decision="approved",
            confidence_score=0.85,
            reasoning="用户满足所有条件"
        )
        
        assert tracker.final_decision == "approved"
        assert tracker.completed_at is not None
        
        # 检查是否创建了最终决策节点
        final_nodes = [
            node for node in tracker.nodes.values()
            if node.node_type == "final_decision"
        ]
        assert len(final_nodes) == 1
        
        final_node = final_nodes[0]
        assert final_node.input_data["decision"] == "approved"
        assert final_node.input_data["confidence"] == 0.85
        assert final_node.input_data["reasoning"] == "用户满足所有条件"
    
    def test_add_confidence_factor(self, tracker):
        """测试添加置信度因子"""
        tracker.add_confidence_factor(
            factor_name="user_age",
            factor_value=25,
            weight=0.8,
            impact=0.6,
            source="user_profile"
        )
        
        tracker.add_confidence_factor(
            factor_name="credit_score",
            factor_value=750,
            weight=0.9,
            impact=0.8,
            source="credit_bureau"
        )
        
        assert len(tracker.confidence_factors) == 2
        
        age_factor = tracker.confidence_factors[0]
        assert age_factor["factor_name"] == "user_age"
        assert age_factor["factor_value"] == 25
        assert age_factor["weight"] == 0.8
        assert age_factor["impact"] == 0.6
        assert age_factor["source"] == "user_profile"
    
    def test_get_decision_path(self, tracker):
        """测试获取决策路径"""
        node1_id = tracker.create_node("start", "Start node", {})
        node2_id = tracker.create_node("condition", "Check condition", {}, node1_id)
        node3_id = tracker.create_node("action", "Execute action", {}, node2_id)
        
        tracker.complete_node(node1_id, {}, 10)
        tracker.complete_node(node2_id, {"result": True}, 20)
        tracker.complete_node(node3_id, {"result": "success"}, 30)
        
        path = tracker.get_decision_path()
        
        assert len(path) == 3
        assert path[0]["node_id"] == node1_id
        assert path[1]["node_id"] == node2_id
        assert path[2]["node_id"] == node3_id
        
        assert path[0]["type"] == "start"
        assert path[1]["type"] == "condition"
        assert path[2]["type"] == "action"
    
    def test_get_decision_tree(self, tracker):
        """测试获取决策树"""
        root_id = tracker.create_node("root", "Root decision", {})
        child1_id = tracker.create_node("child1", "Child 1", {}, root_id)
        child2_id = tracker.create_node("child2", "Child 2", {}, root_id)
        grandchild_id = tracker.create_node("grandchild", "Grandchild", {}, child1_id)
        
        tree = tracker.get_decision_tree()
        
        assert tree["node_id"] == root_id
        assert len(tree["children"]) == 2
        
        # 找到child1分支
        child1_tree = None
        for child in tree["children"]:
            if child["node_id"] == child1_id:
                child1_tree = child
                break
        
        assert child1_tree is not None
        assert len(child1_tree["children"]) == 1
        assert child1_tree["children"][0]["node_id"] == grandchild_id
    
    def test_get_data_flow(self, tracker):
        """测试获取数据流"""
        # 添加一些数据源
        tracker.create_node(
            "data_input",
            "Input data",
            {"source": "user_input", "data_source": "form"}
        )
        
        tracker.record_data_processing(
            "validate_input",
            {"input": "test"},
            {"validated": True},
            50
        )
        
        data_flow = tracker.get_data_flow()
        
        assert "user_input" in data_flow["data_sources"]
        assert "form" in data_flow["data_sources"]
        assert data_flow["total_steps"] == 1
        assert data_flow["total_processing_time"] == 50
        assert len(data_flow["processing_steps"]) == 1
    
    def test_get_summary(self, tracker):
        """测试获取摘要"""
        # 创建一些节点和数据
        tracker.create_node("start", "Start", {})
        tracker.add_confidence_factor("factor1", "value1", 0.5, 0.3, "source1")
        tracker.record_data_processing("process", {}, {}, 100)
        tracker.finalize_decision("approved", 0.8, "All good")
        
        summary = tracker.get_summary()
        
        assert summary["decision_id"] == "test_decision_001"
        assert summary["final_decision"] == "approved"
        assert summary["total_nodes"] == 3  # start + process + final
        assert summary["confidence_factors_count"] == 1
        assert summary["processing_steps_count"] == 2  # process + final
        assert summary["status"] == "completed"
        assert summary["created_at"] is not None
        assert summary["completed_at"] is not None
    
    def test_generate_explanation_components(self, tracker):
        """测试生成解释组件"""
        # 添加置信度因子
        tracker.add_confidence_factor(
            factor_name="user_age",
            factor_value=25,
            weight=0.8,
            impact=0.6,
            source="user_profile"
        )
        
        # 创建一些决策节点
        condition_id = tracker.record_condition_evaluation(
            condition="age >= 18",
            evaluation_result=True,
            evaluation_data={"age": 25, "threshold": 18}
        )
        
        tracker.finalize_decision("approved", 0.9, "User is eligible")
        
        components = tracker.generate_explanation_components()
        
        # 应该有置信度因子组件和决策节点组件
        assert len(components) > 0
        
        # 检查置信度因子组件
        factor_components = [
            comp for comp in components
            if comp.factor_name == "user_age"
        ]
        assert len(factor_components) == 1
        
        factor_comp = factor_components[0]
        assert factor_comp.factor_value == 25
        assert factor_comp.weight == 0.8
        assert factor_comp.impact_score == 0.6
        assert factor_comp.evidence_type == EvidenceType.INPUT_DATA
    
    def test_to_dict_and_from_dict(self, tracker):
        """测试序列化和反序列化"""
        # 创建一些数据
        node_id = tracker.create_node("test", "Test node", {"input": "value"})
        tracker.complete_node(node_id, {"output": "result"}, 100)
        tracker.add_confidence_factor("factor1", "value1", 0.5, 0.3, "source1")
        tracker.finalize_decision("approved", 0.8, "Good to go")
        
        # 序列化
        tracker_dict = tracker.to_dict()
        
        # 反序列化
        restored_tracker = DecisionTracker.from_dict(tracker_dict)
        
        # 验证恢复的数据
        assert restored_tracker.decision_id == tracker.decision_id
        assert restored_tracker.decision_context == tracker.decision_context
        assert restored_tracker.final_decision == tracker.final_decision
        assert len(restored_tracker.nodes) == len(tracker.nodes)
        assert len(restored_tracker.confidence_factors) == len(tracker.confidence_factors)
        assert restored_tracker.root_node_id == tracker.root_node_id
        assert restored_tracker.decision_path == tracker.decision_path
        
        # 验证节点数据
        for node_id, original_node in tracker.nodes.items():
            restored_node = restored_tracker.nodes[node_id]
            assert restored_node.node_id == original_node.node_id
            assert restored_node.node_type == original_node.node_type
            assert restored_node.description == original_node.description
            assert restored_node.status == original_node.status
    
    def test_data_source_extraction(self, tracker):
        """测试数据源提取"""
        tracker.create_node(
            "input",
            "Input node",
            {
                "user_data": {"name": "John", "age": 25},
                "source": "user_form",
                "api_source": "external_api",
                "nested": {
                    "db_source": "user_database",
                    "data": "some_data"
                }
            }
        )
        
        # 数据源应该被自动提取
        expected_sources = {"user_form", "external_api", "user_database"}
        assert tracker.data_sources == expected_sources


if __name__ == "__main__":
    pytest.main([__file__])