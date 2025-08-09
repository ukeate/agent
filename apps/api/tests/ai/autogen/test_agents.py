"""
AutoGen智能体集成测试
"""
import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

from src.ai.autogen.agents import (
    BaseAutoGenAgent,
    CodeExpertAgent,
    ArchitectAgent,
    DocExpertAgent,
    create_agent_from_config,
    create_default_agents,
)
from src.ai.autogen.config import AgentConfig, AgentRole, AGENT_CONFIGS


class TestBaseAutoGenAgent:
    """BaseAutoGenAgent基础测试"""
    
    @pytest.fixture
    def mock_settings(self):
        """模拟设置"""
        with patch('src.ai.autogen.agents.get_settings') as mock:
            mock.return_value.OPENAI_API_KEY = 'test-api-key'
            yield mock
    
    @pytest.fixture
    def sample_config(self):
        """示例配置"""
        return AgentConfig(
            name="测试智能体",
            role=AgentRole.CODE_EXPERT,
            system_prompt="你是一个测试智能体",
            model="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000,
            capabilities=["测试能力"],
        )
    
    @patch('src.ai.autogen.agents.OpenAIChatCompletionClient')
    @patch('src.ai.autogen.agents.AssistantAgent')
    def test_agent_initialization(self, mock_assistant, mock_client, mock_settings, sample_config):
        """测试智能体初始化"""
        # 创建智能体
        agent = BaseAutoGenAgent(sample_config)
        
        # 验证模型客户端创建
        mock_client.assert_called_once_with(
            model=sample_config.model,
            api_key='test-api-key',
            temperature=sample_config.temperature,
            max_tokens=sample_config.max_tokens,
        )
        
        # 验证AssistantAgent创建
        mock_assistant.assert_called_once_with(
            name=sample_config.name,
            model_client=mock_client.return_value,
            system_message=sample_config.system_prompt,
        )
        
        # 验证属性设置
        assert agent.config == sample_config
        assert agent._agent == mock_assistant.return_value
    
    @patch('src.ai.autogen.agents.ChatCompletionClient')
    @patch('src.ai.autogen.agents.AssistantAgent')
    def test_agent_property_access(self, mock_assistant, mock_client, mock_settings, sample_config):
        """测试智能体属性访问"""
        agent = BaseAutoGenAgent(sample_config)
        
        # 测试agent属性
        assert agent.agent == mock_assistant.return_value
        
        # 测试状态获取
        status = agent.get_status()
        expected_status = {
            "name": sample_config.name,
            "role": sample_config.role,
            "status": "active",
            "model": sample_config.model,
            "capabilities": sample_config.capabilities,
        }
        assert status == expected_status
    
    @patch('src.ai.autogen.agents.OpenAIChatCompletionClient')
    @patch('src.ai.autogen.agents.AssistantAgent')
    @pytest.mark.asyncio
    async def test_generate_response(self, mock_assistant, mock_client, mock_settings, sample_config):
        """测试响应生成"""
        # 设置mock响应
        mock_response = Mock()
        mock_response.content = "测试响应内容"
        
        # 设置model_client的mock
        mock_model_client = AsyncMock()
        mock_model_client.create.return_value = mock_response
        mock_assistant.return_value._model_client = mock_model_client
        
        agent = BaseAutoGenAgent(sample_config)
        
        # 测试响应生成
        response = await agent.generate_response("测试消息")
        
        # 验证调用
        mock_model_client.create.assert_called_once()
        call_args = mock_model_client.create.call_args
        messages = call_args[1]['messages']
        assert len(messages) == 2  # 系统消息 + 用户消息
        assert messages[1].content == "测试消息"
        
        # 验证响应
        assert response == "测试响应内容"
    
    def test_initialization_failure(self, mock_settings, sample_config):
        """测试初始化失败情况"""
        # 模拟OpenAIChatCompletionClient初始化失败
        with patch('src.ai.autogen.agents.OpenAIChatCompletionClient', side_effect=Exception("API错误")):
            with pytest.raises(Exception, match="API错误"):
                BaseAutoGenAgent(sample_config)


class TestSpecializedAgents:
    """专业化智能体测试"""
    
    @pytest.fixture
    def mock_base_init(self):
        """模拟基类初始化"""
        with patch.object(BaseAutoGenAgent, '__init__', return_value=None) as mock:
            yield mock
    
    def test_code_expert_agent_creation(self, mock_base_init):
        """测试代码专家智能体创建"""
        agent = CodeExpertAgent()
        
        # 验证使用了正确的配置
        mock_base_init.assert_called_once_with(AGENT_CONFIGS[AgentRole.CODE_EXPERT])
    
    def test_architect_agent_creation(self, mock_base_init):
        """测试架构师智能体创建"""
        agent = ArchitectAgent()
        
        # 验证使用了正确的配置
        mock_base_init.assert_called_once_with(AGENT_CONFIGS[AgentRole.ARCHITECT])
    
    def test_doc_expert_agent_creation(self, mock_base_init):
        """测试文档专家智能体创建"""
        agent = DocExpertAgent()
        
        # 验证使用了正确的配置
        mock_base_init.assert_called_once_with(AGENT_CONFIGS[AgentRole.DOC_EXPERT])
    
    def test_custom_config_override(self, mock_base_init):
        """测试自定义配置覆盖"""
        custom_config = AgentConfig(
            name="自定义代码专家",
            role=AgentRole.CODE_EXPERT,
            system_prompt="自定义提示",
            model="gpt-4",
            temperature=0.2,
            max_tokens=2000,
            capabilities=["自定义能力"],
        )
        
        agent = CodeExpertAgent(custom_config)
        
        # 验证使用了自定义配置
        mock_base_init.assert_called_once_with(custom_config)
    
    @patch.object(BaseAutoGenAgent, 'generate_response')
    @pytest.mark.asyncio
    async def test_code_expert_analyze_code(self, mock_generate):
        """测试代码专家代码分析功能"""
        mock_generate.return_value = "代码分析结果"
        
        with patch.object(BaseAutoGenAgent, '__init__', return_value=None):
            agent = CodeExpertAgent()
            agent.config = AGENT_CONFIGS[AgentRole.CODE_EXPERT]
            
            code = "def hello(): print('Hello, World!')"
            result = await agent.analyze_code(code)
            
            # 验证调用了generate_response
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[0][0]
            assert "分析以下代码" in call_args
            assert code in call_args
            
            # 验证返回结果
            assert result["analysis_type"] == "code_quality"
            assert result["code_length"] == len(code)
            assert result["analysis"] == "代码分析结果"
    
    @patch.object(BaseAutoGenAgent, 'generate_response')
    @pytest.mark.asyncio
    async def test_architect_design_architecture(self, mock_generate):
        """测试架构师架构设计功能"""
        mock_generate.return_value = "架构设计结果"
        
        with patch.object(BaseAutoGenAgent, '__init__', return_value=None):
            agent = ArchitectAgent()
            agent.config = AGENT_CONFIGS[AgentRole.ARCHITECT]
            
            requirements = "构建一个高可用的Web应用"
            result = await agent.design_architecture(requirements)
            
            # 验证调用了generate_response
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[0][0]
            assert "设计系统架构" in call_args
            assert requirements in call_args
            
            # 验证返回结果
            assert result["design_type"] == "system_architecture"
            assert result["requirements_length"] == len(requirements)
            assert result["design"] == "架构设计结果"
    
    @patch.object(BaseAutoGenAgent, 'generate_response')
    @pytest.mark.asyncio
    async def test_doc_expert_generate_documentation(self, mock_generate):
        """测试文档专家文档生成功能"""
        mock_generate.return_value = "文档生成结果"
        
        with patch.object(BaseAutoGenAgent, '__init__', return_value=None):
            agent = DocExpertAgent()
            agent.config = AGENT_CONFIGS[AgentRole.DOC_EXPERT]
            
            content = "API接口说明"
            doc_type = "API文档"
            result = await agent.generate_documentation(content, doc_type)
            
            # 验证调用了generate_response
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args[0][0]
            assert f"生成{doc_type}文档" in call_args
            assert content in call_args
            
            # 验证返回结果
            assert result["doc_type"] == doc_type
            assert result["content_length"] == len(content)
            assert result["documentation"] == "文档生成结果"


class TestAgentFactory:
    """智能体工厂函数测试"""
    
    @patch.object(BaseAutoGenAgent, '__init__', return_value=None)
    def test_create_agent_from_config_code_expert(self, mock_init):
        """测试从配置创建代码专家"""
        config = AGENT_CONFIGS[AgentRole.CODE_EXPERT]
        agent = create_agent_from_config(config)
        
        assert isinstance(agent, CodeExpertAgent)
        mock_init.assert_called_once_with(config)
    
    @patch.object(BaseAutoGenAgent, '__init__', return_value=None)
    def test_create_agent_from_config_architect(self, mock_init):
        """测试从配置创建架构师"""
        config = AGENT_CONFIGS[AgentRole.ARCHITECT]
        agent = create_agent_from_config(config)
        
        assert isinstance(agent, ArchitectAgent)
        mock_init.assert_called_once_with(config)
    
    @patch.object(BaseAutoGenAgent, '__init__', return_value=None)
    def test_create_agent_from_config_doc_expert(self, mock_init):
        """测试从配置创建文档专家"""
        config = AGENT_CONFIGS[AgentRole.DOC_EXPERT]
        agent = create_agent_from_config(config)
        
        assert isinstance(agent, DocExpertAgent)
        mock_init.assert_called_once_with(config)
    
    def test_create_agent_from_config_invalid_role(self):
        """测试创建不支持的智能体角色"""
        # 使用Mock来绕过Pydantic验证
        invalid_config = Mock()
        invalid_config.role = "invalid_role"
        
        with pytest.raises(ValueError, match="不支持的智能体角色"):
            create_agent_from_config(invalid_config)
    
    @patch('src.ai.autogen.agents.create_agent_from_config')
    def test_create_default_agents_success(self, mock_create):
        """测试创建默认智能体集合成功"""
        # 模拟成功创建智能体
        mock_agents = [Mock(), Mock(), Mock()]
        mock_create.side_effect = mock_agents
        
        agents = create_default_agents()
        
        # 验证创建了正确数量的智能体
        assert len(agents) == len(mock_agents)
        assert agents == mock_agents
        
        # 验证每个角色都被调用
        assert mock_create.call_count == len(AGENT_CONFIGS)
    
    @patch('src.ai.autogen.agents.create_agent_from_config')
    def test_create_default_agents_with_failure(self, mock_create):
        """测试创建默认智能体时部分失败"""
        # 模拟第二个智能体创建失败
        mock_agent1 = Mock()
        mock_agent3 = Mock()
        mock_create.side_effect = [mock_agent1, Exception("创建失败"), mock_agent3]
        
        agents = create_default_agents()
        
        # 验证只返回成功创建的智能体
        assert len(agents) == 2
        assert agents == [mock_agent1, mock_agent3]