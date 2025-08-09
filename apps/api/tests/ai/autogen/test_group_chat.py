"""
AutoGen 群组对话测试
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from src.ai.autogen.group_chat import (
    ConversationSession,
    ConversationStatus,
    GroupChatManager,
)
from src.ai.autogen.config import ConversationConfig
from src.ai.autogen.agents import BaseAutoGenAgent


class TestConversationSession:
    """对话会话测试"""
    
    @pytest.fixture
    def mock_agents(self):
        """模拟智能体列表"""
        agents = []
        for i in range(3):
            agent = Mock(spec=BaseAutoGenAgent)
            agent.config = Mock()
            agent.config.name = f"智能体{i+1}"
            agent.config.role = f"role_{i+1}"
            agent.agent = Mock()  # AutoGen智能体实例
            agent.generate_response = AsyncMock(return_value=f"响应{i+1}")
            agent.get_status.return_value = {"status": "active"}
            agents.append(agent)
        return agents
    
    @pytest.fixture
    def sample_config(self):
        """示例对话配置"""
        return ConversationConfig(
            max_rounds=5,
            timeout_seconds=300,
            auto_reply=True,
        )
    
    @patch('src.ai.autogen.group_chat.RoundRobinGroupChat')
    def test_session_initialization(self, mock_group_chat, mock_agents, sample_config):
        """测试会话初始化"""
        session = ConversationSession(
            session_id="test-session",
            participants=mock_agents,
            config=sample_config,
            initial_topic="测试话题",
        )
        
        # 验证基本属性
        assert session.session_id == "test-session"
        assert session.participants == mock_agents
        assert session.config == sample_config
        assert session.initial_topic == "测试话题"
        assert session.status == ConversationStatus.CREATED
        assert session.round_count == 0
        assert len(session.messages) == 0
        
        # 验证GroupChat创建
        mock_group_chat.assert_called_once()
        call_args = mock_group_chat.call_args[0][0]
        assert len(call_args) == len(mock_agents)
    
    @patch('src.ai.autogen.group_chat.RoundRobinGroupChat')
    def test_session_with_defaults(self, mock_group_chat):
        """测试使用默认参数的会话创建"""
        with patch('src.ai.autogen.group_chat.create_default_agents') as mock_create:
            mock_create.return_value = []
            
            session = ConversationSession()
            
            # 验证使用了默认值
            assert session.session_id is not None
            assert len(session.session_id) > 0
            assert session.participants == []
            assert isinstance(session.config, ConversationConfig)
            assert session.initial_topic is None
    
    @patch('src.ai.autogen.group_chat.RoundRobinGroupChat')
    @pytest.mark.asyncio
    async def test_start_conversation_success(self, mock_group_chat, mock_agents, sample_config):
        """测试成功启动对话"""
        session = ConversationSession(
            participants=mock_agents,
            config=sample_config,
        )
        
        initial_message = "开始讨论"
        result = await session.start_conversation(initial_message)
        
        # 验证状态变更 - 对话完成后状态为COMPLETED
        assert session.status in [ConversationStatus.ACTIVE, ConversationStatus.COMPLETED]
        
        # 验证返回结果
        assert result["session_id"] == session.session_id
        assert result["status"] in ["active", "completed"]
        assert result["message_count"] >= 1  # 至少有初始消息
        assert result["current_round"] >= 0
        
        # 验证消息记录
        assert len(session.messages) >= 1  # 至少有初始消息
        first_message = session.messages[0]
        assert first_message["content"] == initial_message
        assert first_message["sender"] == "系统"
        assert first_message["role"] == "user"
    
    @patch('src.ai.autogen.group_chat.RoundRobinGroupChat')
    @pytest.mark.asyncio
    async def test_start_conversation_wrong_status(self, mock_group_chat, mock_agents, sample_config):
        """测试错误状态下启动对话"""
        session = ConversationSession(
            participants=mock_agents,
            config=sample_config,
        )
        session.status = ConversationStatus.ACTIVE  # 设置为已启动状态
        
        with pytest.raises(ValueError, match="会话状态不正确"):
            await session.start_conversation("测试消息")
    
    @patch('src.ai.autogen.group_chat.RoundRobinGroupChat')
    @pytest.mark.asyncio
    async def test_run_group_chat(self, mock_group_chat, mock_agents, sample_config):
        """测试运行群组对话"""
        # 设置较小的最大轮数用于测试
        sample_config.max_rounds = 2
        
        session = ConversationSession(
            participants=mock_agents,
            config=sample_config,
        )
        
        await session.start_conversation("测试消息")
        
        # 验证智能体响应被调用
        for agent in mock_agents:
            agent.generate_response.assert_called()
        
        # 验证消息记录
        assert len(session.messages) > len(mock_agents)  # 至少每个智能体一条消息
        
        # 验证轮数计数
        assert session.round_count > 0
        assert session.round_count <= sample_config.max_rounds
    
    @patch('src.ai.autogen.group_chat.RoundRobinGroupChat')
    @pytest.mark.asyncio
    async def test_termination_conditions(self, mock_group_chat, mock_agents, sample_config):
        """测试终止条件检测"""
        session = ConversationSession(
            participants=mock_agents,
            config=sample_config,
        )
        
        # 测试轮数限制
        session.round_count = sample_config.max_rounds
        assert session._should_terminate() == True
        
        # 测试关键词终止 - 需要至少2条消息才会检查关键词
        session.round_count = 1
        session._add_message("assistant", "智能体1", "开始讨论")  # 第一条消息
        session._add_message("assistant", "智能体1", "讨论完成，会话结束")  # 第二条消息包含终止关键词
        assert session._should_terminate() == True
        
        # 测试正常情况
        session.messages = []
        session.round_count = 1
        assert session._should_terminate() == False
    
    @patch('src.ai.autogen.group_chat.RoundRobinGroupChat')
    @pytest.mark.asyncio
    async def test_pause_conversation(self, mock_group_chat, mock_agents, sample_config):
        """测试暂停对话"""
        session = ConversationSession(
            participants=mock_agents,
            config=sample_config,
        )
        session.status = ConversationStatus.ACTIVE
        
        result = await session.pause_conversation()
        
        # 验证状态变更
        assert session.status == ConversationStatus.PAUSED
        assert result["status"] == "paused"
        
        # 验证取消令牌被调用
        assert session._cancellation_token.is_cancelled()
    
    @patch('src.ai.autogen.group_chat.RoundRobinGroupChat')
    @pytest.mark.asyncio
    async def test_pause_conversation_wrong_status(self, mock_group_chat, mock_agents, sample_config):
        """测试错误状态下暂停对话"""
        session = ConversationSession(
            participants=mock_agents,
            config=sample_config,
        )
        # 状态为CREATED，不是ACTIVE
        
        with pytest.raises(ValueError, match="无法暂停非活跃状态的对话"):
            await session.pause_conversation()
    
    @patch('src.ai.autogen.group_chat.RoundRobinGroupChat')
    @pytest.mark.asyncio
    async def test_resume_conversation(self, mock_group_chat, mock_agents, sample_config):
        """测试恢复对话"""
        session = ConversationSession(
            participants=mock_agents,
            config=sample_config,
        )
        session.status = ConversationStatus.PAUSED
        
        result = await session.resume_conversation()
        
        # 验证状态变更
        assert session.status == ConversationStatus.ACTIVE
        assert result["status"] == "active"
    
    @patch('src.ai.autogen.group_chat.RoundRobinGroupChat')
    @pytest.mark.asyncio
    async def test_terminate_conversation(self, mock_group_chat, mock_agents, sample_config):
        """测试终止对话"""
        session = ConversationSession(
            participants=mock_agents,
            config=sample_config,
        )
        session.status = ConversationStatus.ACTIVE
        
        reason = "测试终止"
        result = await session.terminate_conversation(reason)
        
        # 验证状态变更
        assert session.status == ConversationStatus.TERMINATED
        assert result["status"] == "terminated"
        
        # 验证总结信息
        assert "summary" in result
        assert result["summary"]["termination_reason"] == reason
        assert "duration_minutes" in result["summary"]
        assert "total_messages" in result["summary"]
        
        # 验证取消令牌被调用
        assert session._cancellation_token.is_cancelled()
    
    @patch('src.ai.autogen.group_chat.RoundRobinGroupChat')
    def test_get_status(self, mock_group_chat, mock_agents, sample_config):
        """测试获取会话状态"""
        session = ConversationSession(
            session_id="test-session",
            participants=mock_agents,
            config=sample_config,
        )
        
        status = session.get_status()
        
        # 验证状态信息
        assert status["session_id"] == "test-session"
        assert status["status"] == "created"
        assert "created_at" in status
        assert "updated_at" in status
        assert status["message_count"] == 0
        assert status["round_count"] == 0
        assert len(status["participants"]) == len(mock_agents)
        assert "config" in status
        
        # 验证参与者信息
        for i, participant in enumerate(status["participants"]):
            assert participant["name"] == f"智能体{i+1}"
            assert participant["role"] == f"role_{i+1}"
            assert participant["status"] == "active"
    
    @patch('src.ai.autogen.group_chat.RoundRobinGroupChat')
    def test_add_message(self, mock_group_chat, mock_agents, sample_config):
        """测试添加消息"""
        session = ConversationSession(
            participants=mock_agents,
            config=sample_config,
        )
        
        session._add_message("assistant", "测试智能体", "测试内容")
        
        # 验证消息添加
        assert len(session.messages) == 1
        message = session.messages[0]
        assert message["role"] == "assistant"
        assert message["sender"] == "测试智能体"
        assert message["content"] == "测试内容"
        assert message["round"] == 0
        assert "id" in message
        assert "timestamp" in message


class TestGroupChatManager:
    """群组对话管理器测试"""
    
    @pytest.fixture
    def manager(self):
        """群组对话管理器实例"""
        with patch('src.ai.autogen.group_chat.get_settings'):
            return GroupChatManager()
    
    @pytest.fixture
    def mock_agents(self):
        """模拟智能体列表"""
        agents = []
        for i in range(2):
            agent = Mock(spec=BaseAutoGenAgent)
            agent.config = Mock()
            agent.config.name = f"智能体{i+1}"
            agent.agent = Mock()
            agents.append(agent)
        return agents
    
    @patch('src.ai.autogen.group_chat.ConversationSession')
    @pytest.mark.asyncio
    async def test_create_session(self, mock_session_class, manager, mock_agents):
        """测试创建会话"""
        # 配置mock
        mock_session = Mock()
        mock_session.session_id = "test-session-id"
        mock_session.participants = mock_agents  # 设置participants属性以支持len()
        mock_session_class.return_value = mock_session
        
        config = ConversationConfig()
        topic = "测试话题"
        
        session = await manager.create_session(mock_agents, config, topic)
        
        # 验证ConversationSession创建
        mock_session_class.assert_called_once_with(
            participants=mock_agents,
            config=config,
            initial_topic=topic,
        )
        
        # 验证会话存储
        assert session == mock_session
        assert manager.sessions["test-session-id"] == mock_session
    
    @pytest.mark.asyncio
    async def test_get_session_exists(self, manager):
        """测试获取存在的会话"""
        # 添加一个模拟会话
        mock_session = Mock()
        manager.sessions["existing-session"] = mock_session
        
        session = await manager.get_session("existing-session")
        
        assert session == mock_session
    
    @pytest.mark.asyncio
    async def test_get_session_not_exists(self, manager):
        """测试获取不存在的会话"""
        session = await manager.get_session("non-existing-session")
        
        assert session is None
    
    @pytest.mark.asyncio
    async def test_list_sessions(self, manager):
        """测试列出所有会话"""
        # 添加模拟会话
        mock_sessions = []
        for i in range(3):
            mock_session = Mock()
            mock_session.get_status.return_value = {"session_id": f"session-{i}"}
            manager.sessions[f"session-{i}"] = mock_session
            mock_sessions.append(mock_session)
        
        session_list = await manager.list_sessions()
        
        # 验证返回的会话状态列表
        assert len(session_list) == 3
        for i, status in enumerate(session_list):
            assert status["session_id"] == f"session-{i}"
    
    @pytest.mark.asyncio
    async def test_cleanup_completed_sessions(self, manager):
        """测试清理已完成的会话"""
        # 添加不同状态的会话
        active_session = Mock()
        active_session.status = ConversationStatus.ACTIVE
        
        completed_session = Mock()
        completed_session.status = ConversationStatus.COMPLETED
        
        terminated_session = Mock()
        terminated_session.status = ConversationStatus.TERMINATED
        
        manager.sessions = {
            "active": active_session,
            "completed": completed_session,
            "terminated": terminated_session,
        }
        
        cleaned_count = await manager.cleanup_completed_sessions()
        
        # 验证清理结果
        assert cleaned_count == 2  # completed 和 terminated
        assert len(manager.sessions) == 1
        assert "active" in manager.sessions
        assert "completed" not in manager.sessions
        assert "terminated" not in manager.sessions