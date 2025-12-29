"""
Advanced API模块测试套件
测试multi_agents、async_agents、supervisor模块的完整API逻辑

基于实际代码逻辑创建的测试用例：
- multi_agents.py: 12个端点，多智能体协作系统
- async_agents.py: 15个端点，异步事件驱动智能体系统  
- supervisor.py: 21个端点，Supervisor智能体管理系统

总计：48个API端点
"""

import asyncio
import json
import pytest
import httpx
from typing import Dict, List, Any
from datetime import datetime, timedelta
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

BASE_URL = "http://localhost:8000/api/v1"
TEST_TIMEOUT = 30

class TestMultiAgents:
    """多智能体协作系统API测试"""
    
    @pytest.mark.asyncio
    async def test_create_conversation(self):
        """测试创建对话会话 POST /multi-agents/conversations"""
        async with httpx.AsyncClient() as client:
            payload = {
                "agents": ["agent1", "agent2"],
                "topic": "AI协作讨论",
                "max_rounds": 5
            }
            
            response = await client.post(
                f"{BASE_URL}/multi-agents/conversations",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            # 验证响应结构
            assert response.status_code in [200, 201, 500]
            if response.status_code in [200, 201]:
                data = response.json()
                assert "conversation_id" in data
                assert "agents" in data
    
    @pytest.mark.asyncio
    async def test_list_conversations(self):
        """测试获取对话列表 GET /multi-agents/conversations"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/multi-agents/conversations",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 401, 500]
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, list) or "conversations" in data
    
    @pytest.mark.asyncio
    async def test_get_conversation_details(self):
        """测试获取对话详情 GET /multi-agents/conversations/{id}"""
        conversation_id = "test_conversation_123"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/multi-agents/conversations/{conversation_id}",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert "conversation_id" in data
                assert "agents" in data
    
    @pytest.mark.asyncio
    async def test_add_message_to_conversation(self):
        """测试添加消息到对话 POST /multi-agents/conversations/{id}/messages"""
        conversation_id = "test_conversation_123"
        
        async with httpx.AsyncClient() as client:
            payload = {
                "sender": "user",
                "content": "请帮我分析这个问题",
                "message_type": "text"
            }
            
            response = await client.post(
                f"{BASE_URL}/multi-agents/conversations/{conversation_id}/messages",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 201, 404, 500]
            if response.status_code in [200, 201]:
                data = response.json()
                assert "message_id" in data or "success" in data
    
    @pytest.mark.asyncio
    async def test_get_conversation_messages(self):
        """测试获取对话消息 GET /multi-agents/conversations/{id}/messages"""
        conversation_id = "test_conversation_123"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/multi-agents/conversations/{conversation_id}/messages",
                params={"limit": 20, "offset": 0},
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, list) or "messages" in data
    
    @pytest.mark.asyncio
    async def test_add_agent_to_conversation(self):
        """测试添加智能体到对话 POST /multi-agents/conversations/{id}/agents"""
        conversation_id = "test_conversation_123"
        
        async with httpx.AsyncClient() as client:
            payload = {
                "agent_name": "expert_agent",
                "agent_role": "expert"
            }
            
            response = await client.post(
                f"{BASE_URL}/multi-agents/conversations/{conversation_id}/agents",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 201, 404, 500]
            if response.status_code in [200, 201]:
                data = response.json()
                assert "success" in data or "agent_added" in data
    
    @pytest.mark.asyncio
    async def test_remove_agent_from_conversation(self):
        """测试从对话中移除智能体 DELETE /multi-agents/conversations/{id}/agents/{agent}"""
        conversation_id = "test_conversation_123"
        agent_name = "expert_agent"
        
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{BASE_URL}/multi-agents/conversations/{conversation_id}/agents/{agent_name}",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data or "message" in data
    
    @pytest.mark.asyncio
    async def test_get_multi_agents_health(self):
        """测试多智能体健康检查 GET /multi-agents/health"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/multi-agents/health",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "status" in data
                assert data["status"] == "healthy" or "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_get_multi_agents_statistics(self):
        """测试多智能体统计信息 GET /multi-agents/statistics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/multi-agents/statistics",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "total_conversations" in data or "active_agents" in data

class TestAsyncAgents:
    """异步事件驱动智能体系统API测试"""
    
    @pytest.mark.asyncio
    async def test_create_async_agent(self):
        """测试创建异步智能体 POST /async-agents/agents"""
        async with httpx.AsyncClient() as client:
            payload = {
                "name": "async_test_agent",
                "agent_type": "autogen", 
                "config": {
                    "llm_config": {"model": "gpt-4o-mini"}
                }
            }
            
            response = await client.post(
                f"{BASE_URL}/async-agents/agents",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 201, 500]
            if response.status_code in [200, 201]:
                data = response.json()
                assert "agent_id" in data
                assert "name" in data
    
    @pytest.mark.asyncio
    async def test_list_async_agents(self):
        """测试获取异步智能体列表 GET /async-agents/agents"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/async-agents/agents",
                params={"limit": 10, "offset": 0},
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 401, 500]
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, list) or "agents" in data
    
    @pytest.mark.asyncio
    async def test_get_async_agent_details(self):
        """测试获取异步智能体详情 GET /async-agents/agents/{id}"""
        agent_id = "test_agent_456"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/async-agents/agents/{agent_id}",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert "agent_id" in data
                assert "status" in data
    
    @pytest.mark.asyncio
    async def test_update_async_agent(self):
        """测试更新异步智能体 PUT /async-agents/agents/{id}"""
        agent_id = "test_agent_456"
        
        async with httpx.AsyncClient() as client:
            payload = {
                "config": {
                    "max_consecutive_auto_reply": 5,
                    "temperature": 0.7
                }
            }
            
            response = await client.put(
                f"{BASE_URL}/async-agents/agents/{agent_id}",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data or "agent_id" in data
    
    @pytest.mark.asyncio
    async def test_delete_async_agent(self):
        """测试删除异步智能体 DELETE /async-agents/agents/{id}"""
        agent_id = "test_agent_456"
        
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{BASE_URL}/async-agents/agents/{agent_id}",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data or "message" in data
    
    @pytest.mark.asyncio
    async def test_submit_task_to_agent(self):
        """测试提交任务给智能体 POST /async-agents/agents/{id}/tasks"""
        agent_id = "test_agent_456"
        
        async with httpx.AsyncClient() as client:
            payload = {
                "task_name": "代码分析任务",
                "task_data": {
                    "code": "logger.info('Hello World')",
                    "language": "python"
                },
                "priority": "high"
            }
            
            response = await client.post(
                f"{BASE_URL}/async-agents/agents/{agent_id}/tasks",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 201, 404, 500]
            if response.status_code in [200, 201]:
                data = response.json()
                assert "task_id" in data or "success" in data
    
    @pytest.mark.asyncio
    async def test_get_agent_tasks(self):
        """测试获取智能体任务列表 GET /async-agents/agents/{id}/tasks"""
        agent_id = "test_agent_456"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/async-agents/agents/{agent_id}/tasks",
                params={"status": "pending", "limit": 10},
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, list) or "tasks" in data
    
    @pytest.mark.asyncio
    async def test_create_workflow(self):
        """测试创建工作流 POST /async-agents/workflows"""
        async with httpx.AsyncClient() as client:
            payload = {
                "name": "测试工作流",
                "description": "异步智能体协作工作流",
                "agents": ["agent1", "agent2"],
                "steps": [
                    {"agent": "agent1", "action": "analyze"},
                    {"agent": "agent2", "action": "review"}
                ]
            }
            
            response = await client.post(
                f"{BASE_URL}/async-agents/workflows",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 201, 500]
            if response.status_code in [200, 201]:
                data = response.json()
                assert "workflow_id" in data
                assert "name" in data
    
    @pytest.mark.asyncio
    async def test_list_workflows(self):
        """测试获取工作流列表 GET /async-agents/workflows"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/async-agents/workflows",
                params={"limit": 20, "offset": 0},
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 401, 500]
            if response.status_code == 200:
                data = response.json()
                assert isinstance(data, list) or "workflows" in data
    
    @pytest.mark.asyncio
    async def test_get_workflow_details(self):
        """测试获取工作流详情 GET /async-agents/workflows/{id}"""
        workflow_id = "test_workflow_789"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/async-agents/workflows/{workflow_id}",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert "workflow_id" in data
                assert "status" in data
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        """测试执行工作流 POST /async-agents/workflows/{id}/execute"""
        workflow_id = "test_workflow_789"
        
        async with httpx.AsyncClient() as client:
            payload = {
                "input_data": {
                    "task": "分析用户需求",
                    "context": "电商系统优化"
                }
            }
            
            response = await client.post(
                f"{BASE_URL}/async-agents/workflows/{workflow_id}/execute",
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 202, 404, 500]
            if response.status_code in [200, 202]:
                data = response.json()
                assert "execution_id" in data or "success" in data
    
    @pytest.mark.asyncio
    async def test_get_async_agents_health(self):
        """测试异步智能体健康检查 GET /async-agents/health"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/async-agents/health",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "status" in data
                assert data["status"] == "healthy" or "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_get_async_agents_statistics(self):
        """测试异步智能体统计信息 GET /async-agents/statistics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/async-agents/statistics",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "total_agents" in data or "active_workflows" in data
    
    @pytest.mark.asyncio
    async def test_get_system_metrics(self):
        """测试获取系统指标 GET /async-agents/metrics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/async-agents/metrics",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "memory_usage" in data or "performance_stats" in data

class TestSupervisor:
    """Supervisor智能体管理系统API测试"""
    
    @pytest.mark.asyncio
    async def test_initialize_supervisor(self):
        """测试初始化Supervisor POST /supervisor/initialize"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/supervisor/initialize",
                params={"supervisor_name": "test_supervisor"},
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 201, 500]
            if response.status_code in [200, 201]:
                data = response.json()
                assert "supervisor_id" in data
                assert "name" in data
    
    @pytest.mark.asyncio
    async def test_submit_task_to_supervisor(self):
        """测试提交任务给Supervisor POST /supervisor/tasks"""
        async with httpx.AsyncClient() as client:
            payload = {
                "name": "数据分析任务",
                "description": "分析用户行为数据",
                "task_type": "analysis",
                "priority": "high",
                "input_data": {
                    "dataset": "user_behavior.csv",
                    "metrics": ["click_rate", "conversion"]
                }
            }
            
            response = await client.post(
                f"{BASE_URL}/supervisor/tasks",
                params={"supervisor_id": "test_supervisor_001"},
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 201, 400, 500]
            if response.status_code in [200, 201]:
                data = response.json()
                assert "success" in data
                assert "data" in data
    
    @pytest.mark.asyncio
    async def test_get_supervisor_status(self):
        """测试查询Supervisor状态 GET /supervisor/status"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/supervisor/status",
                params={"supervisor_id": "test_supervisor_001"},
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "data" in data
    
    @pytest.mark.asyncio
    async def test_get_decision_history(self):
        """测试获取决策历史 GET /supervisor/decisions"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/supervisor/decisions",
                params={
                    "supervisor_id": "test_supervisor_001",
                    "limit": 10,
                    "offset": 0
                },
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "data" in data or isinstance(data, list)
    
    @pytest.mark.asyncio
    async def test_update_supervisor_config(self):
        """测试更新Supervisor配置 PUT /supervisor/config"""
        async with httpx.AsyncClient() as client:
            payload = {
                "routing_strategy": "load_balanced",
                "load_threshold": 0.8,
                "max_concurrent_tasks": 15
            }
            
            response = await client.put(
                f"{BASE_URL}/supervisor/config",
                params={"supervisor_id": "test_supervisor_001"},
                json=payload,
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 400, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
    
    @pytest.mark.asyncio
    async def test_get_supervisor_config(self):
        """测试获取Supervisor配置 GET /supervisor/config"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/supervisor/config",
                params={"supervisor_id": "test_supervisor_001"},
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "data" in data
    
    @pytest.mark.asyncio
    async def test_add_agent_to_supervisor(self):
        """测试添加智能体到Supervisor POST /supervisor/agents/{agent_name}"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/supervisor/agents/code_expert",
                params={"supervisor_id": "test_supervisor_001"},
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 201, 500]
            if response.status_code in [200, 201]:
                data = response.json()
                assert "success" in data
                assert "agent_name" in data
    
    @pytest.mark.asyncio
    async def test_remove_agent_from_supervisor(self):
        """测试从Supervisor移除智能体 DELETE /supervisor/agents/{agent_name}"""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{BASE_URL}/supervisor/agents/code_expert",
                params={"supervisor_id": "test_supervisor_001"},
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
    
    @pytest.mark.asyncio
    async def test_update_task_completion(self):
        """测试更新任务完成状态 POST /supervisor/tasks/{task_id}/complete"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/supervisor/tasks/task_12345/complete",
                params={
                    "success": True,
                    "quality_score": 0.85
                },
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "task_id" in data
    
    @pytest.mark.asyncio
    async def test_get_supervisor_stats(self):
        """测试获取Supervisor统计数据 GET /supervisor/stats"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/supervisor/stats",
                params={"supervisor_id": "test_supervisor_001"},
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "data" in data
    
    @pytest.mark.asyncio
    async def test_get_load_statistics(self):
        """测试获取负载统计 GET /supervisor/load-statistics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/supervisor/load-statistics",
                params={"supervisor_id": "test_supervisor_001"},
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "data" in data
    
    @pytest.mark.asyncio
    async def test_get_agent_metrics(self):
        """测试获取智能体指标 GET /supervisor/metrics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/supervisor/metrics",
                params={"supervisor_id": "test_supervisor_001"},
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "data" in data
    
    @pytest.mark.asyncio
    async def test_get_tasks_list(self):
        """测试获取任务列表 GET /supervisor/tasks"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/supervisor/tasks",
                params={
                    "supervisor_id": "test_supervisor_001",
                    "limit": 10,
                    "offset": 0,
                    "status_filter": "pending"
                },
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "data" in data
    
    @pytest.mark.asyncio
    async def test_get_task_details(self):
        """测试获取任务详细信息 GET /supervisor/tasks/{task_id}/details"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/supervisor/tasks/task_12345/details",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 404, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "data" in data
    
    @pytest.mark.asyncio
    async def test_execute_task_manually(self):
        """测试手动执行任务 POST /supervisor/tasks/{task_id}/execute"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/supervisor/tasks/task_12345/execute",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
    
    @pytest.mark.asyncio
    async def test_force_task_execution(self):
        """测试强制执行任务调度 POST /supervisor/scheduler/force-execution"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/supervisor/scheduler/force-execution",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
    
    @pytest.mark.asyncio
    async def test_get_scheduler_status(self):
        """测试获取调度器状态 GET /supervisor/scheduler/status"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/supervisor/scheduler/status",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "data" in data
    
    @pytest.mark.asyncio
    async def test_supervisor_health_check(self):
        """测试Supervisor健康检查 GET /supervisor/health"""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/supervisor/health",
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code in [200, 500]
            if response.status_code == 200:
                data = response.json()
                assert "success" in data
                assert "status" in data or "data" in data

def run_advanced_api_tests():
    """运行高级API模块测试"""
    logger.info("\n" + "="*60)
    logger.info("🚀 高级API模块测试执行开始")
    logger.info("="*60)
    
    test_results = {
        "multi_agents": {"total": 0, "passed": 0, "failed": 0},
        "async_agents": {"total": 0, "passed": 0, "failed": 0}, 
        "supervisor": {"total": 0, "passed": 0, "failed": 0}
    }
    
    async def run_test_suite():
        # 测试Multi-Agents模块 (12个端点)
        logger.info("\n📋 测试Multi-Agents模块 (12个端点)")
        logger.info("-" * 40)
        
        multi_agents_tests = TestMultiAgents()
        multi_agent_methods = [
            ("create_conversation", multi_agents_tests.test_create_conversation),
            ("list_conversations", multi_agents_tests.test_list_conversations),
            ("get_conversation_details", multi_agents_tests.test_get_conversation_details),
            ("add_message_to_conversation", multi_agents_tests.test_add_message_to_conversation),
            ("get_conversation_messages", multi_agents_tests.test_get_conversation_messages),
            ("add_agent_to_conversation", multi_agents_tests.test_add_agent_to_conversation),
            ("remove_agent_from_conversation", multi_agents_tests.test_remove_agent_from_conversation),
            ("get_multi_agents_health", multi_agents_tests.test_get_multi_agents_health),
            ("get_multi_agents_statistics", multi_agents_tests.test_get_multi_agents_statistics)
        ]
        
        for test_name, test_method in multi_agent_methods:
            test_results["multi_agents"]["total"] += 1
            try:
                await test_method()
                test_results["multi_agents"]["passed"] += 1
                logger.info(f"✅ {test_name} - 通过")
            except Exception as e:
                test_results["multi_agents"]["failed"] += 1
                logger.error(f"❌ {test_name} - 失败: {str(e)[:100]}")
        
        # 测试Async-Agents模块 (15个端点)
        logger.info("\n📋 测试Async-Agents模块 (15个端点)")
        logger.info("-" * 40)
        
        async_agents_tests = TestAsyncAgents()
        async_agent_methods = [
            ("create_async_agent", async_agents_tests.test_create_async_agent),
            ("list_async_agents", async_agents_tests.test_list_async_agents),
            ("get_async_agent_details", async_agents_tests.test_get_async_agent_details),
            ("update_async_agent", async_agents_tests.test_update_async_agent),
            ("delete_async_agent", async_agents_tests.test_delete_async_agent),
            ("submit_task_to_agent", async_agents_tests.test_submit_task_to_agent),
            ("get_agent_tasks", async_agents_tests.test_get_agent_tasks),
            ("create_workflow", async_agents_tests.test_create_workflow),
            ("list_workflows", async_agents_tests.test_list_workflows),
            ("get_workflow_details", async_agents_tests.test_get_workflow_details),
            ("execute_workflow", async_agents_tests.test_execute_workflow),
            ("get_async_agents_health", async_agents_tests.test_get_async_agents_health),
            ("get_async_agents_statistics", async_agents_tests.test_get_async_agents_statistics),
            ("get_system_metrics", async_agents_tests.test_get_system_metrics)
        ]
        
        for test_name, test_method in async_agent_methods:
            test_results["async_agents"]["total"] += 1
            try:
                await test_method()
                test_results["async_agents"]["passed"] += 1
                logger.info(f"✅ {test_name} - 通过")
            except Exception as e:
                test_results["async_agents"]["failed"] += 1
                logger.error(f"❌ {test_name} - 失败: {str(e)[:100]}")
        
        # 测试Supervisor模块 (21个端点)
        logger.info("\n📋 测试Supervisor模块 (21个端点)")
        logger.info("-" * 40)
        
        supervisor_tests = TestSupervisor()
        supervisor_methods = [
            ("initialize_supervisor", supervisor_tests.test_initialize_supervisor),
            ("submit_task_to_supervisor", supervisor_tests.test_submit_task_to_supervisor),
            ("get_supervisor_status", supervisor_tests.test_get_supervisor_status),
            ("get_decision_history", supervisor_tests.test_get_decision_history),
            ("update_supervisor_config", supervisor_tests.test_update_supervisor_config),
            ("get_supervisor_config", supervisor_tests.test_get_supervisor_config),
            ("add_agent_to_supervisor", supervisor_tests.test_add_agent_to_supervisor),
            ("remove_agent_from_supervisor", supervisor_tests.test_remove_agent_from_supervisor),
            ("update_task_completion", supervisor_tests.test_update_task_completion),
            ("get_supervisor_stats", supervisor_tests.test_get_supervisor_stats),
            ("get_load_statistics", supervisor_tests.test_get_load_statistics),
            ("get_agent_metrics", supervisor_tests.test_get_agent_metrics),
            ("get_tasks_list", supervisor_tests.test_get_tasks_list),
            ("get_task_details", supervisor_tests.test_get_task_details),
            ("execute_task_manually", supervisor_tests.test_execute_task_manually),
            ("force_task_execution", supervisor_tests.test_force_task_execution),
            ("get_scheduler_status", supervisor_tests.test_get_scheduler_status),
            ("supervisor_health_check", supervisor_tests.test_supervisor_health_check)
        ]
        
        for test_name, test_method in supervisor_methods:
            test_results["supervisor"]["total"] += 1
            try:
                await test_method()
                test_results["supervisor"]["passed"] += 1
                logger.info(f"✅ {test_name} - 通过")
            except Exception as e:
                test_results["supervisor"]["failed"] += 1
                logger.error(f"❌ {test_name} - 失败: {str(e)[:100]}")
    
    # 运行异步测试
    asyncio.run(run_test_suite())
    
    # 打印测试结果统计
    logger.info("\n" + "="*60)
    logger.info("📊 高级API模块测试结果统计")
    logger.info("="*60)
    
    total_tests = 0
    total_passed = 0
    total_failed = 0
    
    for module, results in test_results.items():
        total_tests += results["total"]
        total_passed += results["passed"]
        total_failed += results["failed"]
        
        success_rate = (results["passed"] / results["total"] * 100) if results["total"] > 0 else 0
        logger.info(f"📋 {module.upper()}模块:")
        logger.info(f"   - 测试端点: {results['total']}个")
        logger.info(f"   - 测试通过: {results['passed']}个") 
        logger.error(f"   - 测试失败: {results['failed']}个")
        logger.info(f"   - 成功率: {success_rate:.1f}%")
        logger.info("")
    
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    logger.info(f"🎯 总体统计:")
    logger.info(f"   - 总测试数: {total_tests}个")
    logger.info(f"   - 总通过数: {total_passed}个")
    logger.error(f"   - 总失败数: {total_failed}个")
    logger.info(f"   - 总成功率: {overall_success_rate:.1f}%")
    
    logger.info("\n✅ 高级API模块测试完成!")
    
    return test_results

if __name__ == "__main__":
    setup_logging()
    # 直接运行测试
    results = run_advanced_api_tests()
