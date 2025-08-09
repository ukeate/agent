"""
简化的集成测试
测试ReAct智能体系统的基本功能
"""

import pytest
import asyncio
import json
from unittest.mock import AsyncMock, Mock, patch
import sys
import os

# 添加src目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))

class TestReActIntegration:
    """ReAct智能体集成测试"""

    @pytest.mark.asyncio
    async def test_conversation_service_basic_flow(self):
        """测试对话服务基本流程"""
        # 导入需要的模块
        from services.conversation_service import ConversationService
        
        # 创建服务实例
        service = ConversationService()
        
        # 创建对话
        conversation_id = await service.create_conversation(
            user_id="test_user",
            title="测试对话",
            agent_type="react"
        )
        
        assert conversation_id is not None
        assert len(conversation_id) > 0
        
        # 添加消息
        message_id = await service.add_message(
            conversation_id=conversation_id,
            content="Hello, world!",
            sender_type="user"
        )
        
        assert message_id is not None
        
        # 获取对话历史
        history = await service.get_conversation_history(conversation_id)
        assert len(history) == 1
        assert history[0]["content"] == "Hello, world!"
        assert history[0]["sender_type"] == "user"
        
        # 获取对话上下文
        context = await service.get_conversation_context(conversation_id)
        assert context["conversation_id"] == conversation_id
        assert len(context["messages"]) == 1
        
        # 获取对话摘要
        summary = await service.get_conversation_summary(conversation_id)
        assert summary["conversation_id"] == conversation_id
        assert summary["message_stats"]["total"] == 1
        assert summary["message_stats"]["user"] == 1

    @pytest.mark.asyncio
    async def test_agent_service_basic_flow(self):
        """测试智能体服务基本流程"""
        # Mock外部依赖
        with patch('ai.openai_client.get_openai_client') as mock_openai, \
             patch('ai.mcp.client.get_mcp_client_manager') as mock_mcp:
            
            # 配置Mock
            mock_openai_client = AsyncMock()
            mock_openai_client.create_completion.return_value = {
                "content": "Final Answer: 这是一个测试回答"
            }
            mock_openai.return_value = mock_openai_client
            
            mock_mcp_client = AsyncMock()
            mock_mcp_client.get_available_tools.return_value = {
                "filesystem": [
                    {"name": "read_file", "description": "读取文件"}
                ]
            }
            mock_mcp.return_value = mock_mcp_client
            
            # 导入服务
            from services.agent_service import AgentService
            
            # 创建服务实例
            service = AgentService()
            await service.initialize()
            
            # 创建智能体会话
            session_result = await service.create_agent_session(
                user_id="test_user",
                agent_type="react",
                conversation_title="测试智能体对话"
            )
            
            assert session_result["agent_type"] == "react"
            assert session_result["conversation_id"] is not None
            
            conversation_id = session_result["conversation_id"]
            
            # 与智能体对话
            chat_result = await service.chat_with_agent(
                conversation_id=conversation_id,
                user_input="你好，请帮我分析一个问题",
                user_id="test_user",
                stream=False
            )
            
            assert chat_result["conversation_id"] == conversation_id
            assert "response" in chat_result
            assert chat_result["completed"] is True
            
            # 获取对话历史
            history_result = await service.get_conversation_history(conversation_id)
            assert history_result["conversation_id"] == conversation_id
            assert len(history_result["messages"]) >= 1
            
            # 获取智能体状态
            status_result = await service.get_agent_status(conversation_id)
            assert status_result["conversation_id"] == conversation_id
            assert status_result["agent_type"] == "react"

    @pytest.mark.asyncio 
    async def test_react_agent_step_parsing(self):
        """测试ReAct智能体步骤解析"""
        with patch('ai.openai_client.get_openai_client') as mock_openai, \
             patch('ai.mcp.client.get_mcp_client_manager') as mock_mcp:
            
            # 配置Mock
            mock_openai_client = AsyncMock()
            mock_openai.return_value = mock_openai_client
            
            mock_mcp_client = AsyncMock()
            mock_mcp_client.get_available_tools.return_value = {}
            mock_mcp.return_value = mock_mcp_client
            
            # 导入智能体
            from ai.agents.react_agent import ReActAgent, ReActStepType
            
            # 创建智能体
            agent = ReActAgent()
            await agent.initialize()
            
            # 测试思考解析
            thought_response = "Thought: 我需要分析这个问题"
            step = agent._parse_response(thought_response)
            assert step.step_type == ReActStepType.THOUGHT
            assert step.content == "我需要分析这个问题"
            
            # 测试行动解析
            action_response = '''Action: read_file
Action Input: {"file_path": "test.txt"}'''
            step = agent._parse_response(action_response)
            assert step.step_type == ReActStepType.ACTION
            assert step.tool_name == "read_file"
            assert step.tool_args == {"file_path": "test.txt"}
            
            # 测试最终答案解析
            final_response = "Final Answer: 根据分析，结论是..."
            step = agent._parse_response(final_response)
            assert step.step_type == ReActStepType.FINAL_ANSWER
            assert step.content == "根据分析，结论是..."
            
            # 测试无效JSON处理
            invalid_response = '''Action: read_file
Action Input: {invalid json}'''
            step = agent._parse_response(invalid_response)
            assert step.step_type == ReActStepType.THOUGHT
            assert "格式错误" in step.content

    def test_api_request_models(self):
        """测试API请求模型"""
        from api.v1.agents import CreateAgentSessionRequest, ChatRequest, TaskRequest
        
        # 测试创建会话请求
        session_request = CreateAgentSessionRequest(
            agent_type="react",
            conversation_title="测试对话",
            agent_config={"model": "gpt-4o-mini"}
        )
        assert session_request.agent_type == "react"
        assert session_request.conversation_title == "测试对话"
        assert session_request.agent_config["model"] == "gpt-4o-mini"
        
        # 测试对话请求
        chat_request = ChatRequest(
            message="测试消息",
            stream=True
        )
        assert chat_request.message == "测试消息"
        assert chat_request.stream is True
        
        # 测试任务请求
        task_request = TaskRequest(
            task_description="执行文件分析任务",
            task_type="analysis",
            context={"file_path": "test.txt"}
        )
        assert task_request.task_description == "执行文件分析任务"
        assert task_request.task_type == "analysis"
        assert task_request.context["file_path"] == "test.txt"

    def test_configuration_loading(self):
        """测试配置加载"""
        from core.config import get_settings
        
        settings = get_settings()
        
        # 验证基本配置
        assert hasattr(settings, 'DEBUG')
        assert hasattr(settings, 'HOST')
        assert hasattr(settings, 'PORT')
        
        # 验证AI相关配置
        assert hasattr(settings, 'OPENAI_API_KEY')
        assert hasattr(settings, 'MAX_CONTEXT_LENGTH')
        assert hasattr(settings, 'SESSION_TIMEOUT_MINUTES')
        
        # 验证默认值
        assert settings.MAX_CONTEXT_LENGTH == 100000
        assert settings.SESSION_TIMEOUT_MINUTES == 60

    @pytest.mark.asyncio
    async def test_context_management(self):
        """测试上下文管理"""
        from services.conversation_service import ConversationService
        
        service = ConversationService()
        
        # 创建对话
        conversation_id = await service.create_conversation(
            user_id="test_user",
            title="上下文测试"
        )
        
        # 添加多条消息模拟长对话
        messages = [
            "第一条消息",
            "第二条消息",
            "第三条消息"
        ]
        
        for msg in messages:
            await service.add_message(
                conversation_id=conversation_id,
                content=msg,
                sender_type="user"
            )
        
        # 更新上下文
        await service.update_conversation_context(
            conversation_id=conversation_id,
            context_updates={
                "user_preference": "简洁回答",
                "task_type": "问答"
            }
        )
        
        # 获取上下文验证
        context = await service.get_conversation_context(conversation_id)
        assert len(context["messages"]) == 3
        assert context["session_context"]["user_preference"] == "简洁回答"
        assert context["session_context"]["task_type"] == "问答"

    def test_error_handling(self):
        """测试错误处理"""
        from services.conversation_service import ConversationService
        
        service = ConversationService()
        
        # 测试获取不存在的对话
        with pytest.raises(ValueError):
            asyncio.run(service.get_conversation_context("non_existent_id"))
        
        # 测试向不存在的对话添加消息
        with pytest.raises(ValueError):
            asyncio.run(service.add_message(
                conversation_id="non_existent_id",
                content="test message",
                sender_type="user"
            ))