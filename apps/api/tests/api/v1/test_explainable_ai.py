"""可解释AI API端点集成测试"""

import pytest
import json
from uuid import uuid4
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch

from src.main import app
from src.models.schemas.explanation import (
    ExplanationType,
    ExplanationLevel,
    EvidenceType,
    ConfidenceSource
)


class TestExplainableAiAPI:
    """可解释AI API测试"""
    
    @pytest.fixture
    def client(self):
        """测试客户端"""
        return TestClient(app)
    
    @pytest.fixture
    def sample_explanation_request(self):
        """示例解释请求"""
        return {
            "decision_id": "test_decision_001",
            "decision_context": "贷款审批决策测试",
            "explanation_type": "decision",
            "explanation_level": "detailed",
            "style": "user_friendly",
            "factors": [
                {
                    "factor_name": "credit_score",
                    "factor_value": 750,
                    "weight": 0.8,
                    "impact": 0.7,
                    "source": "credit_bureau"
                },
                {
                    "factor_name": "annual_income",
                    "factor_value": 80000,
                    "weight": 0.7,
                    "impact": 0.6,
                    "source": "financial_statement"
                }
            ],
            "use_cot_reasoning": False,
            "reasoning_mode": "analytical"
        }
    
    @pytest.fixture
    def sample_cot_request(self):
        """示例CoT推理请求"""
        return {
            "decision_id": "cot_test_001",
            "decision_context": "复杂决策推理测试",
            "reasoning_mode": "analytical",
            "explanation_level": "detailed",
            "factors": [
                {
                    "factor_name": "complexity_factor",
                    "factor_value": "high",
                    "weight": 0.9,
                    "impact": 0.8,
                    "source": "system_analysis"
                }
            ]
        }
    
    @pytest.fixture
    def sample_workflow_request(self):
        """示例工作流请求"""
        return {
            "workflow_id": "workflow_test_001",
            "workflow_name": "测试工作流",
            "nodes": [
                {
                    "node_id": "node_1",
                    "node_type": "processor",
                    "node_name": "数据处理节点",
                    "input_data": {"input": "test_data"},
                    "output_data": {"output": "processed_data"},
                    "execution_time": 0.5,
                    "status": "completed",
                    "metadata": {"info": "test_metadata"}
                },
                {
                    "node_id": "node_2",
                    "node_type": "validator",
                    "node_name": "数据验证节点",
                    "input_data": {"input": "processed_data"},
                    "output_data": {"valid": True},
                    "execution_time": 0.2,
                    "status": "completed",
                    "metadata": {"validation": "passed"}
                }
            ],
            "explanation_level": "detailed"
        }
    
    def test_health_check(self, client):
        """测试健康检查端点"""
        response = client.get("/api/v1/explainable-ai/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "healthy"
        assert data["service"] == "explainable-ai"
        assert "timestamp" in data
        assert "version" in data
        assert "components" in data
        
        components = data["components"]
        assert components["explanation_generator"] == "active"
        assert components["cot_reasoner"] == "active"
        assert components["workflow_explainer"] == "active"
        assert components["formatter"] == "active"
    
    @patch('src.ai.explainer.explanation_generator.ExplanationGenerator.generate_explanation')
    def test_generate_explanation_success(self, mock_generate, client, sample_explanation_request):
        """测试生成解释成功"""
        # 模拟生成解释的返回值
        mock_explanation = Mock()
        mock_explanation.id = uuid4()
        mock_explanation.decision_id = "test_decision_001"
        mock_explanation.explanation_type = ExplanationType.DECISION
        mock_explanation.explanation_level = ExplanationLevel.DETAILED
        mock_explanation.decision_description = "贷款审批决策测试"
        mock_explanation.decision_outcome = "approved"
        mock_explanation.summary_explanation = "基于信用评分和收入水平，建议批准贷款"
        mock_explanation.components = []
        mock_explanation.confidence_metrics = Mock()
        mock_explanation.confidence_metrics.overall_confidence = 0.8
        mock_explanation.confidence_metrics.uncertainty_score = 0.2
        mock_explanation.confidence_metrics.confidence_sources = [ConfidenceSource.MODEL_PROBABILITY]
        mock_explanation.counterfactuals = []
        mock_explanation.metadata = {"test": "data"}
        
        mock_generate.return_value = mock_explanation
        
        response = client.post(
            "/api/v1/explainable-ai/generate-explanation",
            json=sample_explanation_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["decision_id"] == "test_decision_001"
        assert data["explanation_type"] == "decision"
        assert data["explanation_level"] == "detailed"
        assert data["decision_description"] == "贷款审批决策测试"
        assert data["decision_outcome"] == "approved"
        assert data["summary_explanation"] == "基于信用评分和收入水平，建议批准贷款"
    
    def test_generate_explanation_validation_error(self, client):
        """测试解释请求验证错误"""
        invalid_request = {
            "decision_id": "",  # 空的decision_id应该失败
            "decision_context": "测试上下文"
        }
        
        response = client.post(
            "/api/v1/explainable-ai/generate-explanation",
            json=invalid_request
        )
        
        assert response.status_code == 422  # 验证错误
    
    @patch('src.ai.explainer.explanation_generator.ExplanationGenerator.generate_cot_reasoning_explanation')
    def test_generate_cot_reasoning_success(self, mock_generate_cot, client, sample_cot_request):
        """测试生成CoT推理成功"""
        # 模拟CoT推理返回值
        mock_chain = Mock()
        mock_chain.chain_id = "chain_001"
        
        mock_explanation = Mock()
        mock_explanation.id = uuid4()
        mock_explanation.decision_id = "cot_test_001"
        mock_explanation.explanation_type = ExplanationType.REASONING
        mock_explanation.explanation_level = ExplanationLevel.DETAILED
        mock_explanation.decision_description = "复杂决策推理测试"
        mock_explanation.decision_outcome = "reasoning_completed"
        mock_explanation.summary_explanation = "通过分析性推理得出结论"
        mock_explanation.components = []
        mock_explanation.confidence_metrics = Mock()
        mock_explanation.confidence_metrics.overall_confidence = 0.85
        mock_explanation.confidence_metrics.uncertainty_score = 0.15
        mock_explanation.confidence_metrics.confidence_sources = [ConfidenceSource.REASONING_QUALITY]
        mock_explanation.counterfactuals = []
        mock_explanation.metadata = {"reasoning_type": "analytical"}
        
        mock_generate_cot.return_value = (mock_chain, mock_explanation)
        
        response = client.post(
            "/api/v1/explainable-ai/cot-reasoning",
            json=sample_cot_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["decision_id"] == "cot_test_001"
        assert data["explanation_type"] == "reasoning"
        assert data["decision_description"] == "复杂决策推理测试"
        assert data["decision_outcome"] == "reasoning_completed"
    
    @patch('src.ai.explainer.explanation_generator.ExplanationGenerator.generate_workflow_explanation')
    def test_generate_workflow_explanation_success(self, mock_generate_workflow, client, sample_workflow_request):
        """测试生成工作流解释成功"""
        # 模拟工作流解释返回值
        mock_explanation = Mock()
        mock_explanation.id = uuid4()
        mock_explanation.decision_id = "workflow_test_001"
        mock_explanation.explanation_type = ExplanationType.WORKFLOW
        mock_explanation.explanation_level = ExplanationLevel.DETAILED
        mock_explanation.decision_description = "工作流执行: 测试工作流"
        mock_explanation.decision_outcome = "执行completed"
        mock_explanation.summary_explanation = "工作流成功执行，包含2个节点"
        mock_explanation.components = []
        mock_explanation.confidence_metrics = Mock()
        mock_explanation.confidence_metrics.overall_confidence = 0.9
        mock_explanation.confidence_metrics.uncertainty_score = 0.1
        mock_explanation.confidence_metrics.confidence_sources = [ConfidenceSource.EXECUTION_SUCCESS]
        mock_explanation.counterfactuals = []
        mock_explanation.metadata = {"workflow_type": "langgraph_workflow"}
        
        mock_generate_workflow.return_value = mock_explanation
        
        response = client.post(
            "/api/v1/explainable-ai/workflow-explanation",
            json=sample_workflow_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["decision_id"] == "workflow_test_001"
        assert data["explanation_type"] == "workflow"
        assert data["decision_description"] == "工作流执行: 测试工作流"
        assert data["decision_outcome"] == "执行completed"
    
    def test_get_explanation_types(self, client):
        """测试获取解释类型"""
        response = client.get("/api/v1/explainable-ai/explanation-types")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "explanation_types" in data
        assert "explanation_levels" in data
        assert "reasoning_modes" in data
        assert "output_formats" in data
        
        # 检查解释类型
        explanation_types = data["explanation_types"]
        assert len(explanation_types) == 3
        assert any(et["value"] == "decision" for et in explanation_types)
        assert any(et["value"] == "reasoning" for et in explanation_types)
        assert any(et["value"] == "workflow" for et in explanation_types)
        
        # 检查解释级别
        explanation_levels = data["explanation_levels"]
        assert len(explanation_levels) == 3
        assert any(el["value"] == "summary" for el in explanation_levels)
        assert any(el["value"] == "detailed" for el in explanation_levels)
        assert any(el["value"] == "technical" for el in explanation_levels)
        
        # 检查推理模式
        reasoning_modes = data["reasoning_modes"]
        assert len(reasoning_modes) == 4
        assert any(rm["value"] == "analytical" for rm in reasoning_modes)
        assert any(rm["value"] == "deductive" for rm in reasoning_modes)
        assert any(rm["value"] == "inductive" for rm in reasoning_modes)
        assert any(rm["value"] == "abductive" for rm in reasoning_modes)
    
    def test_get_demo_scenarios(self, client):
        """测试获取演示场景"""
        response = client.get("/api/v1/explainable-ai/demo-scenarios")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "scenarios" in data
        scenarios = data["scenarios"]
        
        assert len(scenarios) == 3
        
        # 检查贷款审批场景
        loan_scenario = next(s for s in scenarios if s["type"] == "loan_approval")
        assert loan_scenario["name"] == "贷款审批"
        assert loan_scenario["description"] == "银行贷款审批决策场景"
        assert "complexity_levels" in loan_scenario
        assert "simple" in loan_scenario["complexity_levels"]
        assert "medium" in loan_scenario["complexity_levels"]
        assert "complex" in loan_scenario["complexity_levels"]
        
        # 检查医疗诊断场景
        medical_scenario = next(s for s in scenarios if s["type"] == "medical_diagnosis")
        assert medical_scenario["name"] == "医疗诊断"
        
        # 检查投资建议场景
        investment_scenario = next(s for s in scenarios if s["type"] == "investment_recommendation")
        assert investment_scenario["name"] == "投资建议"
    
    @patch('src.ai.explainer.explanation_generator.ExplanationGenerator.generate_explanation')
    def test_generate_demo_scenario_loan_approval(self, mock_generate, client):
        """测试生成贷款审批演示场景"""
        # 模拟生成解释的返回值
        mock_explanation = Mock()
        mock_explanation.id = uuid4()
        mock_explanation.decision_id = "loan_demo_001"
        mock_explanation.explanation_type = ExplanationType.DECISION
        mock_explanation.explanation_level = ExplanationLevel.DETAILED
        mock_explanation.decision_description = "贷款审批决策"
        mock_explanation.decision_outcome = "approved"
        mock_explanation.summary_explanation = "基于多项因素分析，建议批准贷款"
        mock_explanation.components = []
        mock_explanation.confidence_metrics = Mock()
        mock_explanation.confidence_metrics.overall_confidence = 0.8
        mock_explanation.confidence_metrics.uncertainty_score = 0.2
        mock_explanation.confidence_metrics.confidence_sources = [ConfidenceSource.MODEL_PROBABILITY]
        mock_explanation.counterfactuals = []
        mock_explanation.metadata = {"demo_scenario": True}
        
        mock_generate.return_value = mock_explanation
        
        demo_request = {
            "scenario_type": "loan_approval",
            "complexity": "medium",
            "include_cot": True
        }
        
        response = client.post(
            "/api/v1/explainable-ai/demo-scenario",
            json=demo_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["decision_id"] == "loan_demo_001"
        assert data["explanation_type"] == "decision"
        assert data["decision_description"] == "贷款审批决策"
    
    def test_generate_demo_scenario_different_complexity(self, client):
        """测试不同复杂度的演示场景"""
        complexities = ["simple", "medium", "complex"]
        
        for complexity in complexities:
            with patch('src.ai.explainer.explanation_generator.ExplanationGenerator.generate_explanation') as mock_generate:
                mock_explanation = Mock()
                mock_explanation.id = uuid4()
                mock_explanation.decision_id = f"demo_{complexity}"
                mock_explanation.explanation_type = ExplanationType.DECISION
                mock_explanation.explanation_level = ExplanationLevel.DETAILED
                mock_explanation.decision_description = f"演示决策 - {complexity}"
                mock_explanation.decision_outcome = "completed"
                mock_explanation.summary_explanation = f"{complexity}复杂度演示"
                mock_explanation.components = []
                mock_explanation.confidence_metrics = Mock()
                mock_explanation.confidence_metrics.overall_confidence = 0.8
                mock_explanation.confidence_metrics.uncertainty_score = 0.2
                mock_explanation.confidence_metrics.confidence_sources = []
                mock_explanation.counterfactuals = []
                mock_explanation.metadata = {}
                
                mock_generate.return_value = mock_explanation
                
                demo_request = {
                    "scenario_type": "loan_approval",
                    "complexity": complexity,
                    "include_cot": False
                }
                
                response = client.post(
                    "/api/v1/explainable-ai/demo-scenario",
                    json=demo_request
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["decision_id"] == f"demo_{complexity}"
    
    def test_format_explanation(self, client):
        """测试格式化解释"""
        format_request = {
            "explanation_id": str(uuid4()),
            "output_format": "html",
            "template_name": "default"
        }
        
        response = client.post(
            "/api/v1/explainable-ai/format-explanation",
            json=format_request
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "explanation_id" in data
        assert "format" in data
        assert "content" in data
        assert data["format"] == "html"
    
    def test_api_error_handling(self, client):
        """测试API错误处理"""
        # 测试无效的JSON
        response = client.post(
            "/api/v1/explainable-ai/generate-explanation",
            data="invalid json"
        )
        assert response.status_code == 422
        
        # 测试缺少必需字段
        incomplete_request = {
            "decision_context": "测试上下文"
            # 缺少 decision_id
        }
        
        response = client.post(
            "/api/v1/explainable-ai/generate-explanation",
            json=incomplete_request
        )
        assert response.status_code == 422
    
    @patch('src.ai.explainer.explanation_generator.ExplanationGenerator.generate_explanation')
    def test_api_internal_error_handling(self, mock_generate, client, sample_explanation_request):
        """测试API内部错误处理"""
        # 模拟内部错误
        mock_generate.side_effect = Exception("内部服务器错误")
        
        response = client.post(
            "/api/v1/explainable-ai/generate-explanation",
            json=sample_explanation_request
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "解释生成失败" in data["detail"]
    
    def test_concurrent_requests(self, client, sample_explanation_request):
        """测试并发请求处理"""
        import threading
        import time
        
        results = []
        
        def make_request():
            with patch('src.ai.explainer.explanation_generator.ExplanationGenerator.generate_explanation') as mock_generate:
                mock_explanation = Mock()
                mock_explanation.id = uuid4()
                mock_explanation.decision_id = "concurrent_test"
                mock_explanation.explanation_type = ExplanationType.DECISION
                mock_explanation.explanation_level = ExplanationLevel.DETAILED
                mock_explanation.decision_description = "并发测试"
                mock_explanation.decision_outcome = "completed"
                mock_explanation.summary_explanation = "并发处理测试"
                mock_explanation.components = []
                mock_explanation.confidence_metrics = Mock()
                mock_explanation.confidence_metrics.overall_confidence = 0.8
                mock_explanation.confidence_metrics.uncertainty_score = 0.2
                mock_explanation.confidence_metrics.confidence_sources = []
                mock_explanation.counterfactuals = []
                mock_explanation.metadata = {}
                
                mock_generate.return_value = mock_explanation
                
                response = client.post(
                    "/api/v1/explainable-ai/generate-explanation",
                    json=sample_explanation_request
                )
                results.append(response.status_code)
        
        # 创建5个并发请求
        threads = []
        for i in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # 启动所有线程
        for thread in threads:
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 验证所有请求都成功
        assert len(results) == 5
        assert all(status_code == 200 for status_code in results)


class TestExplainableAiAPIIntegration:
    """可解释AI API集成测试"""
    
    @pytest.fixture
    def client(self):
        """测试客户端"""
        return TestClient(app)
    
    def test_full_explanation_workflow(self, client):
        """测试完整的解释工作流"""
        # 1. 获取可用的解释类型
        types_response = client.get("/api/v1/explainable-ai/explanation-types")
        assert types_response.status_code == 200
        
        # 2. 获取演示场景
        scenarios_response = client.get("/api/v1/explainable-ai/demo-scenarios")
        assert scenarios_response.status_code == 200
        
        # 3. 生成演示场景解释
        with patch('src.ai.explainer.explanation_generator.ExplanationGenerator.generate_explanation') as mock_generate:
            mock_explanation = Mock()
            mock_explanation.id = uuid4()
            mock_explanation.decision_id = "workflow_test"
            mock_explanation.explanation_type = ExplanationType.DECISION
            mock_explanation.explanation_level = ExplanationLevel.DETAILED
            mock_explanation.decision_description = "工作流测试"
            mock_explanation.decision_outcome = "completed"
            mock_explanation.summary_explanation = "工作流测试完成"
            mock_explanation.components = []
            mock_explanation.confidence_metrics = Mock()
            mock_explanation.confidence_metrics.overall_confidence = 0.8
            mock_explanation.confidence_metrics.uncertainty_score = 0.2
            mock_explanation.confidence_metrics.confidence_sources = []
            mock_explanation.counterfactuals = []
            mock_explanation.metadata = {}
            
            mock_generate.return_value = mock_explanation
            
            demo_request = {
                "scenario_type": "loan_approval",
                "complexity": "medium",
                "include_cot": False
            }
            
            demo_response = client.post(
                "/api/v1/explainable-ai/demo-scenario",
                json=demo_request
            )
            assert demo_response.status_code == 200
            
            explanation_data = demo_response.json()
            explanation_id = explanation_data["id"]
            
            # 4. 格式化解释
            format_request = {
                "explanation_id": explanation_id,
                "output_format": "html"
            }
            
            format_response = client.post(
                "/api/v1/explainable-ai/format-explanation",
                json=format_request
            )
            assert format_response.status_code == 200
        
        # 5. 检查健康状态
        health_response = client.get("/api/v1/explainable-ai/health")
        assert health_response.status_code == 200
    
    def test_cross_explanation_type_consistency(self, client):
        """测试不同解释类型的一致性"""
        base_request = {
            "decision_id": "consistency_test",
            "decision_context": "一致性测试",
            "factors": [
                {
                    "factor_name": "test_factor",
                    "factor_value": "test_value",
                    "weight": 0.8,
                    "impact": 0.7,
                    "source": "test_source"
                }
            ]
        }
        
        explanation_types = ["decision", "reasoning"]
        results = []
        
        for exp_type in explanation_types:
            request = base_request.copy()
            request["explanation_type"] = exp_type
            request["explanation_level"] = "detailed"
            
            if exp_type == "reasoning":
                request["reasoning_mode"] = "analytical"
                
                with patch('src.ai.explainer.explanation_generator.ExplanationGenerator.generate_cot_reasoning_explanation') as mock_generate:
                    mock_chain = Mock()
                    mock_explanation = Mock()
                    mock_explanation.id = uuid4()
                    mock_explanation.decision_id = "consistency_test"
                    mock_explanation.explanation_type = ExplanationType.REASONING
                    mock_explanation.explanation_level = ExplanationLevel.DETAILED
                    mock_explanation.decision_description = "一致性测试"
                    mock_explanation.decision_outcome = "completed"
                    mock_explanation.summary_explanation = f"{exp_type}解释测试"
                    mock_explanation.components = []
                    mock_explanation.confidence_metrics = Mock()
                    mock_explanation.confidence_metrics.overall_confidence = 0.8
                    mock_explanation.confidence_metrics.uncertainty_score = 0.2
                    mock_explanation.confidence_metrics.confidence_sources = []
                    mock_explanation.counterfactuals = []
                    mock_explanation.metadata = {}
                    
                    mock_generate.return_value = (mock_chain, mock_explanation)
                    
                    response = client.post(
                        "/api/v1/explainable-ai/cot-reasoning",
                        json=request
                    )
            else:
                with patch('src.ai.explainer.explanation_generator.ExplanationGenerator.generate_explanation') as mock_generate:
                    mock_explanation = Mock()
                    mock_explanation.id = uuid4()
                    mock_explanation.decision_id = "consistency_test"
                    mock_explanation.explanation_type = ExplanationType.DECISION
                    mock_explanation.explanation_level = ExplanationLevel.DETAILED
                    mock_explanation.decision_description = "一致性测试"
                    mock_explanation.decision_outcome = "completed"
                    mock_explanation.summary_explanation = f"{exp_type}解释测试"
                    mock_explanation.components = []
                    mock_explanation.confidence_metrics = Mock()
                    mock_explanation.confidence_metrics.overall_confidence = 0.8
                    mock_explanation.confidence_metrics.uncertainty_score = 0.2
                    mock_explanation.confidence_metrics.confidence_sources = []
                    mock_explanation.counterfactuals = []
                    mock_explanation.metadata = {}
                    
                    mock_generate.return_value = mock_explanation
                    
                    response = client.post(
                        "/api/v1/explainable-ai/generate-explanation",
                        json=request
                    )
            
            assert response.status_code == 200
            results.append(response.json())
        
        # 验证一致性
        assert len(results) == 2
        assert results[0]["decision_id"] == results[1]["decision_id"]
        assert results[0]["decision_description"] == results[1]["decision_description"]


if __name__ == "__main__":
    pytest.main([__file__])