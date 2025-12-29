"""推理API端点测试"""

import pytest
import json
from unittest.mock import Mock, AsyncMock, patch
from uuid import uuid4
from fastapi.testclient import TestClient
from httpx import AsyncClient
from main import app
from models.schemas.reasoning import (
    ReasoningRequest,
    ReasoningResponse,
    ReasoningChain,
    ReasoningStrategy,
    ThoughtStep,
    ThoughtStepType
)

class TestReasoningAPI:
    """测试推理API"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def async_client(self):
        return AsyncClient(app=app, base_url="http://test")

    @pytest.fixture
    def auth_headers(self):
        """模拟认证头"""
        return {"Authorization": "Bearer test_token"}

    @pytest.fixture
    def mock_user(self):
        return {"id": "test_user_id", "email": "test@example.com"}

    def test_create_reasoning_chain(self, client, auth_headers):
        """测试创建推理链"""
        with patch('api.v1.reasoning.get_current_user', return_value={"id": "test_user"}):
            with patch('api.v1.reasoning.reasoning_service.execute_reasoning') as mock_execute:
                # 模拟响应
                mock_response = ReasoningResponse(
                    chain_id=uuid4(),
                    problem="计算 2+2",
                    strategy=ReasoningStrategy.ZERO_SHOT,
                    steps=[
                        ThoughtStep(
                            step_number=1,
                            step_type=ThoughtStepType.CONCLUSION,
                            content="4",
                            reasoning="2+2=4",
                            confidence=1.0
                        )
                    ],
                    conclusion="4",
                    confidence=1.0,
                    success=True
                )
                mock_execute.return_value = mock_response
                
                request_data = {
                    "problem": "计算 2+2",
                    "strategy": "zero_shot",
                    "max_steps": 5,
                    "stream": False
                }
                
                response = client.post(
                    "/api/v1/reasoning/chain",
                    json=request_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["problem"] == "计算 2+2"
                assert data["success"] is True
                assert data["conclusion"] == "4"

    @pytest.mark.asyncio
    async def test_stream_reasoning(self, async_client, auth_headers):
        """测试流式推理"""
        with patch('api.v1.reasoning.get_current_user', return_value={"id": "test_user"}):
            with patch('api.v1.reasoning.reasoning_service.stream_reasoning') as mock_stream:
                # 模拟流式响应
                async def mock_generator():
                    for i in range(3):
                        yield {
                            "chain_id": str(uuid4()),
                            "step_number": i+1,
                            "step_type": "analysis",
                            "content": f"步骤{i+1}",
                            "reasoning": "推理",
                            "confidence": 0.8,
                            "is_final": i == 2
                        }
                
                mock_stream.return_value = mock_generator()
                
                request_data = {
                    "problem": "测试问题",
                    "strategy": "zero_shot",
                    "max_steps": 5,
                    "stream": True
                }
                
                response = await async_client.post(
                    "/api/v1/reasoning/stream",
                    json=request_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream"

    def test_get_reasoning_chain(self, client, auth_headers):
        """测试获取推理链"""
        chain_id = uuid4()
        
        with patch('api.v1.reasoning.get_current_user', return_value={"id": "test_user"}):
            with patch('api.v1.reasoning.reasoning_service.get_chain') as mock_get:
                # 模拟推理链
                mock_chain = ReasoningChain(
                    id=chain_id,
                    strategy=ReasoningStrategy.ZERO_SHOT,
                    problem="测试问题",
                    conclusion="测试结论",
                    confidence_score=0.9
                )
                mock_get.return_value = mock_chain
                
                response = client.get(
                    f"/api/v1/reasoning/chain/{chain_id}",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["id"] == str(chain_id)
                assert data["problem"] == "测试问题"

    def test_get_reasoning_history(self, client, auth_headers):
        """测试获取推理历史"""
        with patch('api.v1.reasoning.get_current_user', return_value={"id": "test_user"}):
            with patch('api.v1.reasoning.reasoning_service.get_user_history') as mock_history:
                # 模拟历史记录
                mock_chains = [
                    ReasoningChain(
                        id=uuid4(),
                        strategy=ReasoningStrategy.ZERO_SHOT,
                        problem=f"问题{i}",
                        conclusion=f"结论{i}",
                        confidence_score=0.8 + i*0.05
                    )
                    for i in range(3)
                ]
                mock_history.return_value = mock_chains
                
                response = client.get(
                    "/api/v1/reasoning/history?limit=10&offset=0",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert len(data) == 3
                assert data[0]["problem"] == "问题0"

    def test_validate_reasoning_chain(self, client, auth_headers):
        """测试验证推理链"""
        chain_id = uuid4()
        
        with patch('api.v1.reasoning.get_current_user', return_value={"id": "test_user"}):
            with patch('api.v1.reasoning.reasoning_service.validate_chain') as mock_validate:
                # 模拟验证结果
                mock_validation = {
                    "step_id": str(uuid4()),
                    "is_valid": True,
                    "consistency_score": 0.85,
                    "issues": [],
                    "suggestions": []
                }
                mock_validate.return_value = mock_validation
                
                response = client.post(
                    f"/api/v1/reasoning/chain/{chain_id}/validate",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["is_valid"] is True
                assert data["consistency_score"] == 0.85

    def test_create_reasoning_branch(self, client, auth_headers):
        """测试创建推理分支"""
        chain_id = uuid4()
        
        with patch('api.v1.reasoning.get_current_user', return_value={"id": "test_user"}):
            with patch('api.v1.reasoning.reasoning_service.create_branch') as mock_branch:
                branch_id = uuid4()
                mock_branch.return_value = branch_id
                
                response = client.post(
                    f"/api/v1/reasoning/chain/{chain_id}/branch",
                    params={
                        "parent_step_number": 2,
                        "reason": "探索替代方案"
                    },
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["branch_id"] == str(branch_id)

    def test_recover_reasoning_chain(self, client, auth_headers):
        """测试恢复推理链"""
        chain_id = uuid4()
        
        with patch('api.v1.reasoning.get_current_user', return_value={"id": "test_user"}):
            with patch('api.v1.reasoning.reasoning_service.recover_chain') as mock_recover:
                mock_recover.return_value = True
                
                response = client.post(
                    f"/api/v1/reasoning/chain/{chain_id}/recover",
                    params={"strategy": "backtrack"},
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["message"] == "推理链已恢复"

    def test_get_reasoning_stats(self, client, auth_headers):
        """测试获取统计信息"""
        with patch('api.v1.reasoning.get_current_user', return_value={"id": "test_user"}):
            with patch('api.v1.reasoning.reasoning_service.get_user_stats') as mock_stats:
                # 模拟统计信息
                mock_stats.return_value = {
                    "total_chains": 10,
                    "completed_chains": 8,
                    "completion_rate": 0.8,
                    "average_confidence": 0.85,
                    "recovery_stats": {
                        "total_failures": 3,
                        "recovery_attempts": 3,
                        "success_rate": 0.67
                    }
                }
                
                response = client.get(
                    "/api/v1/reasoning/stats",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["total_chains"] == 10
                assert data["completion_rate"] == 0.8

    def test_delete_reasoning_chain(self, client, auth_headers):
        """测试删除推理链"""
        chain_id = uuid4()
        
        with patch('api.v1.reasoning.get_current_user', return_value={"id": "test_user"}):
            with patch('api.v1.reasoning.reasoning_service.delete_chain') as mock_delete:
                mock_delete.return_value = True
                
                response = client.delete(
                    f"/api/v1/reasoning/chain/{chain_id}",
                    headers=auth_headers
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                assert data["message"] == "推理链已删除"

    def test_error_handling(self, client, auth_headers):
        """测试错误处理"""
        with patch('api.v1.reasoning.get_current_user', return_value={"id": "test_user"}):
            with patch('api.v1.reasoning.reasoning_service.execute_reasoning') as mock_execute:
                # 模拟错误
                mock_execute.side_effect = Exception("推理引擎错误")
                
                request_data = {
                    "problem": "测试问题",
                    "strategy": "zero_shot"
                }
                
                response = client.post(
                    "/api/v1/reasoning/chain",
                    json=request_data,
                    headers=auth_headers
                )
                
                assert response.status_code == 500
                data = response.json()
                assert "推理引擎错误" in data["detail"]
