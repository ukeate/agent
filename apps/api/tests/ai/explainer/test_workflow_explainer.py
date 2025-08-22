"""工作流解释器单元测试"""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from src.ai.explainer.workflow_explainer import (
    WorkflowExplainer,
    WorkflowNode,
    WorkflowExecution
)
from src.ai.explainer.decision_tracker import DecisionTracker
from src.models.schemas.explanation import (
    ExplanationType,
    ExplanationLevel,
    EvidenceType
)


class TestWorkflowNode:
    """测试工作流节点"""
    
    def test_create_workflow_node(self):
        """测试创建工作流节点"""
        node = WorkflowNode(
            node_id="node_001",
            node_type="processor",
            node_name="数据处理节点",
            input_data={"input": "test_data"},
            output_data={"output": "processed_data"},
            execution_time=0.5,
            status="completed",
            metadata={"info": "test_metadata"}
        )
        
        assert node.node_id == "node_001"
        assert node.node_type == "processor"
        assert node.node_name == "数据处理节点"
        assert node.input_data["input"] == "test_data"
        assert node.output_data["output"] == "processed_data"
        assert node.execution_time == 0.5
        assert node.status == "completed"
        assert node.metadata["info"] == "test_metadata"
        assert node.timestamp is not None
    
    def test_workflow_node_default_metadata(self):
        """测试工作流节点默认元数据"""
        node = WorkflowNode(
            node_id="node_002",
            node_type="validator",
            node_name="验证节点",
            input_data={},
            output_data={},
            execution_time=0.1,
            status="completed"
        )
        
        assert node.metadata == {}
        assert isinstance(node.timestamp, datetime)


class TestWorkflowExecution:
    """测试工作流执行"""
    
    def test_create_workflow_execution(self):
        """测试创建工作流执行"""
        execution = WorkflowExecution("workflow_001", "测试工作流")
        
        assert execution.workflow_id == "workflow_001"
        assert execution.workflow_name == "测试工作流"
        assert len(execution.nodes) == 0
        assert len(execution.edges) == 0
        assert len(execution.execution_path) == 0
        assert execution.global_state == {}
        assert execution.start_time is not None
        assert execution.end_time is None
        assert execution.status == "running"
        assert len(execution.error_messages) == 0
    
    def test_add_node_execution(self):
        """测试添加节点执行"""
        execution = WorkflowExecution("workflow_001", "测试工作流")
        
        node1 = WorkflowNode(
            node_id="node_1",
            node_type="processor",
            node_name="节点1",
            input_data={},
            output_data={},
            execution_time=0.3,
            status="completed"
        )
        
        node2 = WorkflowNode(
            node_id="node_2",
            node_type="validator",
            node_name="节点2",
            input_data={},
            output_data={},
            execution_time=0.2,
            status="completed"
        )
        
        execution.add_node_execution(node1)
        execution.add_node_execution(node2)
        
        assert len(execution.nodes) == 2
        assert len(execution.execution_path) == 2
        assert execution.execution_path == ["node_1", "node_2"]
    
    def test_add_edge(self):
        """测试添加边"""
        execution = WorkflowExecution("workflow_001", "测试工作流")
        
        execution.add_edge("node_1", "node_2", "success")
        execution.add_edge("node_2", "node_3")
        
        assert len(execution.edges) == 2
        assert execution.edges[0]["from"] == "node_1"
        assert execution.edges[0]["to"] == "node_2"
        assert execution.edges[0]["condition"] == "success"
        assert execution.edges[1]["from"] == "node_2"
        assert execution.edges[1]["to"] == "node_3"
        assert execution.edges[1]["condition"] == ""
    
    def test_update_global_state(self):
        """测试更新全局状态"""
        execution = WorkflowExecution("workflow_001", "测试工作流")
        
        execution.update_global_state({"variable1": "value1"})
        execution.update_global_state({"variable2": "value2", "variable1": "updated_value1"})
        
        assert execution.global_state["variable1"] == "updated_value1"
        assert execution.global_state["variable2"] == "value2"
    
    def test_complete_execution(self):
        """测试完成执行"""
        execution = WorkflowExecution("workflow_001", "测试工作流")
        
        assert execution.status == "running"
        assert execution.end_time is None
        
        execution.complete_execution("completed")
        
        assert execution.status == "completed"
        assert execution.end_time is not None
    
    def test_get_execution_summary(self):
        """测试获取执行摘要"""
        execution = WorkflowExecution("workflow_001", "测试工作流")
        
        # 添加一些节点
        node1 = WorkflowNode("node_1", "processor", "节点1", {}, {}, 0.3, "completed")
        node2 = WorkflowNode("node_2", "validator", "节点2", {}, {}, 0.2, "completed")
        
        execution.add_node_execution(node1)
        execution.add_node_execution(node2)
        execution.update_global_state({"var1": "value1", "var2": "value2"})
        execution.error_messages.append("测试错误")
        execution.complete_execution("completed")
        
        summary = execution.get_execution_summary()
        
        assert summary["workflow_id"] == "workflow_001"
        assert summary["workflow_name"] == "测试工作流"
        assert summary["status"] == "completed"
        assert summary["total_nodes"] == 2
        assert summary["execution_path_length"] == 2
        assert summary["total_execution_time"] > 0
        assert summary["average_node_time"] == 0.25  # (0.3 + 0.2) / 2
        assert summary["error_count"] == 1
        assert summary["state_variables"] == 2


class TestWorkflowExplainer:
    """测试工作流解释器"""
    
    @pytest.fixture
    def mock_openai_client(self):
        """模拟OpenAI客户端"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "这是生成的工作流解释。"
        mock_client.chat_completion.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def explainer(self, mock_openai_client):
        """创建测试用的工作流解释器"""
        return WorkflowExplainer(mock_openai_client)
    
    @pytest.fixture
    def sample_workflow_execution(self):
        """创建样本工作流执行"""
        execution = WorkflowExecution("workflow_test_001", "测试工作流")
        
        # 添加节点
        node1 = WorkflowNode(
            node_id="node_1",
            node_type="processor",
            node_name="数据处理节点",
            input_data={"raw_data": "test_input"},
            output_data={"processed_data": "test_output"},
            execution_time=0.5,
            status="completed",
            metadata={"processor_type": "data_transformer"}
        )
        
        node2 = WorkflowNode(
            node_id="node_2",
            node_type="validator",
            node_name="数据验证节点",
            input_data={"processed_data": "test_output"},
            output_data={"valid": True, "errors": []},
            execution_time=0.2,
            status="completed",
            metadata={"validation_rules": ["not_empty", "format_check"]}
        )
        
        node3 = WorkflowNode(
            node_id="node_3",
            node_type="aggregator",
            node_name="结果聚合节点",
            input_data={"valid_data": "test_output"},
            output_data={"final_result": "aggregated_data"},
            execution_time=0.3,
            status="completed"
        )
        
        execution.add_node_execution(node1)
        execution.add_node_execution(node2)
        execution.add_node_execution(node3)
        
        # 添加边
        execution.add_edge("node_1", "node_2", "success")
        execution.add_edge("node_2", "node_3", "valid")
        
        # 更新状态
        execution.update_global_state({
            "total_records": 100,
            "processed_records": 95,
            "validation_status": "passed"
        })
        
        execution.complete_execution("completed")
        
        return execution
    
    def test_create_workflow_explainer(self, explainer):
        """测试创建工作流解释器"""
        assert explainer is not None
        assert explainer.openai_client is not None
        assert "explanation_config" in explainer.__dict__
        assert "node_type_explanations" in explainer.__dict__
        assert "workflow_patterns" in explainer.__dict__
    
    def test_node_type_explanations(self, explainer):
        """测试节点类型解释配置"""
        assert "agent" in explainer.node_type_explanations
        assert "tool" in explainer.node_type_explanations
        assert "processor" in explainer.node_type_explanations
        assert "validator" in explainer.node_type_explanations
        
        assert explainer.node_type_explanations["agent"] == "智能体执行节点"
        assert explainer.node_type_explanations["processor"] == "数据处理节点"
    
    def test_workflow_patterns(self, explainer):
        """测试工作流模式"""
        assert "sequential" in explainer.workflow_patterns
        assert "parallel" in explainer.workflow_patterns
        assert "conditional" in explainer.workflow_patterns
        assert "loop" in explainer.workflow_patterns
        
        assert explainer.workflow_patterns["sequential"] == "顺序执行模式"
        assert explainer.workflow_patterns["parallel"] == "并行执行模式"
    
    def test_generate_workflow_explanation_basic(self, explainer, sample_workflow_execution):
        """测试基本工作流解释生成"""
        explanation = explainer.generate_workflow_explanation(
            sample_workflow_execution,
            explanation_level=ExplanationLevel.DETAILED
        )
        
        assert explanation is not None
        assert explanation.explanation_type == ExplanationType.WORKFLOW
        assert explanation.decision_id == "workflow_test_001"
        assert explanation.explanation_level == ExplanationLevel.DETAILED
        assert "测试工作流" in explanation.decision_description
        assert explanation.decision_outcome == "执行completed"
        assert explanation.summary_explanation is not None
        assert len(explanation.components) > 0
        assert explanation.confidence_metrics is not None
        assert explanation.visualization_data is not None
    
    def test_generate_workflow_explanation_with_decision_tracker(self, explainer, sample_workflow_execution):
        """测试带决策跟踪器的工作流解释生成"""
        decision_tracker = DecisionTracker("decision_001", "工作流决策跟踪")
        decision_tracker.add_confidence_factor(
            factor_name="execution_success",
            factor_value=True,
            weight=0.9,
            impact=0.8,
            source="workflow_execution"
        )
        
        explanation = explainer.generate_workflow_explanation(
            sample_workflow_execution,
            decision_tracker=decision_tracker,
            explanation_level=ExplanationLevel.TECHNICAL
        )
        
        assert explanation is not None
        assert explanation.explanation_level == ExplanationLevel.TECHNICAL
        assert explanation.technical_explanation is not None
    
    def test_analyze_workflow_execution(self, explainer, sample_workflow_execution):
        """测试分析工作流执行"""
        analysis = explainer._analyze_workflow_execution(sample_workflow_execution)
        
        assert "workflow_id" in analysis
        assert "workflow_name" in analysis
        assert "status" in analysis
        assert "total_nodes" in analysis
        assert "node_performance" in analysis
        assert "error_analysis" in analysis
        assert "efficiency_metrics" in analysis
        
        assert analysis["workflow_id"] == "workflow_test_001"
        assert analysis["workflow_name"] == "测试工作流"
        assert analysis["status"] == "completed"
        assert analysis["total_nodes"] == 3
        
        # 检查性能分析
        node_performance = analysis["node_performance"]
        assert "fastest_node" in node_performance
        assert "slowest_node" in node_performance
        assert "average_time" in node_performance
        assert node_performance["fastest_node"] == 0.2
        assert node_performance["slowest_node"] == 0.5
        
        # 检查效率指标
        efficiency = analysis["efficiency_metrics"]
        assert "success_rate" in efficiency
        assert efficiency["success_rate"] == 1.0  # 没有错误
    
    def test_generate_node_explanations(self, explainer, sample_workflow_execution):
        """测试生成节点解释"""
        node_explanations = explainer._generate_node_explanations(
            sample_workflow_execution.nodes,
            ExplanationLevel.DETAILED
        )
        
        assert len(node_explanations) == 3
        
        # 检查第一个节点解释
        first_node_exp = node_explanations[0]
        assert first_node_exp["node_id"] == "node_1"
        assert first_node_exp["node_type"] == "processor"
        assert first_node_exp["node_name"] == "数据处理节点"
        assert first_node_exp["type_description"] == "数据处理节点"
        assert first_node_exp["execution_time"] == 0.5
        assert first_node_exp["status"] == "completed"
        assert "input_summary" in first_node_exp
        assert "output_summary" in first_node_exp
        assert "performance_rating" in first_node_exp
        assert "detailed_description" in first_node_exp
    
    def test_generate_node_explanations_technical(self, explainer, sample_workflow_execution):
        """测试生成技术级节点解释"""
        node_explanations = explainer._generate_node_explanations(
            sample_workflow_execution.nodes,
            ExplanationLevel.TECHNICAL
        )
        
        first_node_exp = node_explanations[0]
        assert "technical_details" in first_node_exp
        assert "execution_metrics" in first_node_exp
        
        technical_details = first_node_exp["technical_details"]
        assert technical_details["processor_type"] == "data_transformer"
        
        execution_metrics = first_node_exp["execution_metrics"]
        assert execution_metrics["execution_time"] == 0.5
        assert "timestamp" in execution_metrics
        assert execution_metrics["status_code"] == "completed"
    
    def test_analyze_execution_path(self, explainer, sample_workflow_execution):
        """测试分析执行路径"""
        path_analysis = explainer._analyze_execution_path(
            sample_workflow_execution.execution_path,
            sample_workflow_execution.nodes
        )
        
        assert "path_length" in path_analysis
        assert "unique_nodes" in path_analysis
        assert "revisited_nodes" in path_analysis
        assert "has_loops" in path_analysis
        assert "type_distribution" in path_analysis
        assert "bottleneck_nodes" in path_analysis
        
        assert path_analysis["path_length"] == 3
        assert path_analysis["unique_nodes"] == 3
        assert path_analysis["revisited_nodes"] == 0
        assert path_analysis["has_loops"] is False
        
        # 检查类型分布
        type_dist = path_analysis["type_distribution"]
        assert type_dist["processor"] == 1
        assert type_dist["validator"] == 1
        assert type_dist["aggregator"] == 1
        
        # 检查瓶颈节点（按执行时间排序）
        bottleneck = path_analysis["bottleneck_nodes"]
        assert bottleneck[0] == "node_1"  # 执行时间最长(0.5s)
    
    def test_analyze_state_changes(self, explainer, sample_workflow_execution):
        """测试分析状态变化"""
        state_analysis = explainer._analyze_state_changes(
            sample_workflow_execution.global_state,
            sample_workflow_execution.nodes
        )
        
        assert "total_state_variables" in state_analysis
        assert "state_complexity" in state_analysis
        assert "state_size" in state_analysis
        
        assert state_analysis["total_state_variables"] == 3
        assert state_analysis["state_complexity"] > 0
        assert state_analysis["state_size"] > 0
    
    def test_calculate_workflow_confidence(self, explainer, sample_workflow_execution):
        """测试计算工作流置信度"""
        analysis = explainer._analyze_workflow_execution(sample_workflow_execution)
        confidence_metrics = explainer._calculate_workflow_confidence(
            sample_workflow_execution, analysis
        )
        
        assert confidence_metrics is not None
        assert 0.0 <= confidence_metrics.overall_confidence <= 1.0
        assert 0.0 <= confidence_metrics.uncertainty_score <= 1.0
        assert confidence_metrics.confidence_interval_lower is not None
        assert confidence_metrics.confidence_interval_upper is not None
        assert confidence_metrics.confidence_sources is not None
        
        # 没有错误的情况下，置信度应该较高
        assert confidence_metrics.overall_confidence > 0.8
    
    def test_create_workflow_components(self, explainer, sample_workflow_execution):
        """测试创建工作流组件"""
        node_explanations = explainer._generate_node_explanations(
            sample_workflow_execution.nodes,
            ExplanationLevel.DETAILED
        )
        path_analysis = explainer._analyze_execution_path(
            sample_workflow_execution.execution_path,
            sample_workflow_execution.nodes
        )
        
        components = explainer._create_workflow_components(
            sample_workflow_execution,
            node_explanations,
            path_analysis
        )
        
        assert len(components) >= 3  # 至少3个节点组件
        
        # 检查节点组件
        first_component = components[0]
        assert first_component.evidence_type == EvidenceType.WORKFLOW_EXECUTION
        assert first_component.evidence_source == "workflow_node"
        assert "节点:" in first_component.factor_name
        assert "执行时间:" in first_component.factor_value
        assert 0.0 <= first_component.weight <= 1.0
        assert 0.0 <= first_component.impact_score <= 1.0
    
    def test_generate_workflow_counterfactuals(self, explainer, sample_workflow_execution):
        """测试生成工作流反事实场景"""
        analysis = explainer._analyze_workflow_execution(sample_workflow_execution)
        counterfactuals = explainer._generate_workflow_counterfactuals(
            sample_workflow_execution, analysis
        )
        
        # 由于没有错误，主要生成性能优化相关的反事实
        assert len(counterfactuals) >= 0
        
        if counterfactuals:
            first_scenario = counterfactuals[0]
            assert first_scenario.scenario_name is not None
            assert first_scenario.predicted_outcome is not None
            assert 0.0 <= first_scenario.probability <= 1.0
            assert first_scenario.explanation is not None
    
    def test_generate_workflow_visualization(self, explainer, sample_workflow_execution):
        """测试生成工作流可视化数据"""
        analysis = explainer._analyze_workflow_execution(sample_workflow_execution)
        viz_data = explainer._generate_workflow_visualization(
            sample_workflow_execution, analysis
        )
        
        assert "workflow_graph" in viz_data
        assert "execution_timeline" in viz_data
        assert "performance_metrics" in viz_data
        assert "state_evolution" in viz_data
        
        # 检查工作流图
        workflow_graph = viz_data["workflow_graph"]
        assert workflow_graph["chart_type"] == "graph"
        assert "nodes" in workflow_graph
        assert "edges" in workflow_graph
        assert len(workflow_graph["nodes"]) == 3
        assert len(workflow_graph["edges"]) == 2
        
        # 检查执行时间线
        timeline = viz_data["execution_timeline"]
        assert timeline["chart_type"] == "timeline"
        assert len(timeline["data"]) == 3
        
        # 检查性能指标
        performance = viz_data["performance_metrics"]
        assert performance["chart_type"] == "bar"
        assert len(performance["data"]) == 3
    
    def test_identify_workflow_pattern(self, explainer, sample_workflow_execution):
        """测试识别工作流模式"""
        pattern = explainer._identify_workflow_pattern(sample_workflow_execution)
        
        # 当前的示例是顺序执行模式
        assert pattern == "sequential"
    
    def test_identify_workflow_pattern_with_loops(self, explainer):
        """测试识别带循环的工作流模式"""
        execution = WorkflowExecution("loop_workflow", "循环工作流")
        
        # 创建有循环的执行路径
        node1 = WorkflowNode("node_1", "processor", "节点1", {}, {}, 0.1, "completed")
        node2 = WorkflowNode("node_2", "condition", "节点2", {}, {}, 0.1, "completed")
        
        execution.add_node_execution(node1)
        execution.add_node_execution(node2)
        execution.add_node_execution(node1)  # 重复执行node_1
        
        pattern = explainer._identify_workflow_pattern(execution)
        assert pattern == "loop"
    
    def test_rate_node_performance(self, explainer):
        """测试评估节点性能"""
        # 快速完成的节点
        fast_node = WorkflowNode("fast", "processor", "快速节点", {}, {}, 0.05, "completed")
        fast_rating = explainer._rate_node_performance(fast_node)
        assert fast_rating > 0.8
        
        # 慢速节点
        slow_node = WorkflowNode("slow", "processor", "慢速节点", {}, {}, 3.0, "completed")
        slow_rating = explainer._rate_node_performance(slow_node)
        assert slow_rating < 0.8
        
        # 失败节点
        failed_node = WorkflowNode("failed", "processor", "失败节点", {}, {}, 1.0, "failed")
        failed_rating = explainer._rate_node_performance(failed_node)
        assert failed_rating < 0.5
    
    def test_calculate_state_complexity(self, explainer):
        """测试计算状态复杂度"""
        # 简单状态
        simple_state = {"var1": "value1", "var2": 123}
        simple_complexity = explainer._calculate_state_complexity(simple_state)
        
        # 复杂状态
        complex_state = {
            "dict_var": {"nested": {"deep": "value"}},
            "list_var": [1, 2, 3, 4, 5],
            "simple_var": "value"
        }
        complex_complexity = explainer._calculate_state_complexity(complex_state)
        
        assert complex_complexity > simple_complexity
        assert simple_complexity > 0
        assert complex_complexity > 0
    
    def test_format_nodes_summary(self, explainer, sample_workflow_execution):
        """测试格式化节点摘要"""
        summary = explainer._format_nodes_summary(sample_workflow_execution.nodes)
        
        assert "数据处理节点" in summary
        assert "数据验证节点" in summary
        assert "结果聚合节点" in summary
        assert "processor" in summary
        assert "validator" in summary
        assert "aggregator" in summary
        assert "completed" in summary
    
    def test_format_performance_analysis(self, explainer):
        """测试格式化性能分析"""
        performance_data = {
            "fastest_node": 0.1,
            "slowest_node": 0.5,
            "average_time": 0.3,
            "total_time": 1.0
        }
        
        formatted = explainer._format_performance_analysis(performance_data)
        
        assert "最快节点: 0.100s" in formatted
        assert "最慢节点: 0.500s" in formatted
        assert "平均执行时间: 0.300s" in formatted
        assert "总执行时间: 1.000s" in formatted
    
    def test_format_error_analysis(self, explainer):
        """测试格式化错误分析"""
        error_data = {
            "error_count": 2,
            "error_rate": 0.25,
            "has_errors": True
        }
        
        formatted = explainer._format_error_analysis(error_data)
        
        assert "错误数量: 2" in formatted
        assert "错误率: 25.0%" in formatted
        assert "是否有错误: 是" in formatted
    
    def test_openai_api_failure_fallback(self, sample_workflow_execution):
        """测试OpenAI API失败时的降级处理"""
        # 创建一个会失败的mock客户端
        mock_client = Mock()
        mock_client.chat_completion.side_effect = Exception("API Error")
        
        explainer = WorkflowExplainer(mock_client)
        
        explanation = explainer.generate_workflow_explanation(
            sample_workflow_execution,
            explanation_level=ExplanationLevel.SUMMARY
        )
        
        assert explanation is not None
        assert explanation.explanation_type == ExplanationType.WORKFLOW
        assert "工作流解释生成失败" in explanation.summary_explanation
        assert explanation.metadata.get("error") is not None
        assert explanation.metadata.get("fallback_mode") is True
    
    def test_summarize_data(self, explainer):
        """测试数据摘要"""
        # 空数据
        empty_summary = explainer._summarize_data({})
        assert empty_summary == "无数据"
        
        # 简单数据
        simple_data = {"key1": "value1", "key2": "value2"}
        simple_summary = explainer._summarize_data(simple_data)
        assert "key1: value1..." in simple_summary
        assert "key2: value2..." in simple_summary
        
        # 复杂数据
        complex_data = {"list_key": [1, 2, 3], "dict_key": {"nested": "value"}}
        complex_summary = explainer._summarize_data(complex_data)
        assert "list_key: 3项" in complex_summary
        assert "dict_key: 1项" in complex_summary
        
        # 超过3个键的数据
        large_data = {f"key_{i}": f"value_{i}" for i in range(5)}
        large_summary = explainer._summarize_data(large_data)
        assert "等5个字段" in large_summary
    
    def test_generate_node_detailed_description(self, explainer):
        """测试生成节点详细描述"""
        node = WorkflowNode(
            node_id="test_node",
            node_type="processor",
            node_name="测试节点",
            input_data={},
            output_data={},
            execution_time=0.5,
            status="completed"
        )
        
        description = explainer._generate_node_detailed_description(node)
        
        assert "数据处理节点" in description
        assert "测试节点" in description
        assert "成功完成" in description
        assert "0.500秒" in description
    
    def test_analyze_node_input_output(self, explainer):
        """测试分析节点输入输出"""
        # 测试输入分析
        input_data = {
            "simple_field": "value",
            "complex_field": {"nested": "data"},
            "list_field": [1, 2, 3]
        }
        
        input_analysis = explainer._analyze_node_input(input_data)
        
        assert input_analysis["field_count"] == 3
        assert input_analysis["data_size"] > 0
        assert input_analysis["has_complex_data"] is True
        
        # 测试输出分析
        output_data = {"result": "success"}
        output_analysis = explainer._analyze_node_output(output_data)
        
        assert output_analysis["field_count"] == 1
        assert output_analysis["has_complex_data"] is False


if __name__ == "__main__":
    pytest.main([__file__])