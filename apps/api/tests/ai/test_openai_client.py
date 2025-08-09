"""
OpenAI客户端测试
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from openai import APIError, RateLimitError, AuthenticationError, APITimeoutError

from src.ai.openai_client import OpenAIClient, get_openai_client


class TestOpenAIClient:
    """OpenAI客户端测试类"""

    @pytest.fixture
    def client(self):
        """创建测试客户端"""
        with patch("src.ai.openai_client.get_settings") as mock_settings:
            mock_settings.return_value.OPENAI_API_KEY = "test-api-key"
            return OpenAIClient()

    @pytest.fixture
    def mock_openai_response(self):
        """模拟OpenAI API响应"""
        mock_choice = Mock()
        mock_choice.message.content = "Test response"
        mock_choice.message.tool_calls = None
        mock_choice.finish_reason = "stop"
        
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        return mock_response

    @pytest.mark.asyncio
    async def test_client_initialization(self):
        """测试客户端初始化"""
        with patch("src.ai.openai_client.get_settings") as mock_settings:
            mock_settings.return_value.OPENAI_API_KEY = "test-api-key"
            client = OpenAIClient()
            
            assert client.api_key == "test-api-key"
            assert client.model == "gpt-4o-mini"
            assert client.max_retries == 3
            assert client.base_delay == 1.0

    def test_client_initialization_without_api_key(self):
        """测试无API密钥时的初始化"""
        with patch("src.ai.openai_client.get_settings") as mock_settings:
            mock_settings.return_value.OPENAI_API_KEY = ""
            
            with pytest.raises(ValueError, match="OpenAI API密钥未配置"):
                OpenAIClient()

    @pytest.mark.asyncio
    async def test_create_completion_success(self, client, mock_openai_response):
        """测试成功创建聊天完成"""
        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_openai_response
            
            messages = [{"role": "user", "content": "Hello"}]
            result = await client.create_completion(messages)
            
            assert result["content"] == "Test response"
            assert result["tool_calls"] is None
            assert result["finish_reason"] == "stop"
            assert result["usage"]["total_tokens"] == 15
            assert "duration" in result
            
            mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_completion_with_tools(self, client, mock_openai_response):
        """测试带工具的聊天完成"""
        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_openai_response
            
            messages = [{"role": "user", "content": "Hello"}]
            tools = [{"name": "test_tool", "description": "Test tool"}]
            
            result = await client.create_completion(messages, tools=tools, tool_choice="auto")
            
            assert result["content"] == "Test response"
            mock_create.assert_called_once()
            
            call_kwargs = mock_create.call_args[1]
            assert call_kwargs["tools"] == tools
            assert call_kwargs["tool_choice"] == "auto"

    @pytest.mark.asyncio
    async def test_create_completion_rate_limit_error(self, client):
        """测试速率限制错误"""
        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            from tenacity import RetryError
            import httpx
            mock_response = Mock(spec=httpx.Response)
            mock_response.status_code = 429
            mock_response.headers = {"x-request-id": "test-request-id"}
            mock_create.side_effect = RateLimitError("Rate limit exceeded", response=mock_response, body=None)
            
            messages = [{"role": "user", "content": "Hello"}]
            
            with pytest.raises(RetryError):
                await client.create_completion(messages)

    @pytest.mark.asyncio
    async def test_create_completion_authentication_error(self, client):
        """测试认证错误"""
        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.side_effect = AuthenticationError("Invalid API key", response=Mock(), body=None)
            
            messages = [{"role": "user", "content": "Hello"}]
            
            with pytest.raises(AuthenticationError):
                await client.create_completion(messages)

    @pytest.mark.asyncio
    async def test_create_completion_timeout_error(self, client):
        """测试超时错误"""
        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            from tenacity import RetryError
            mock_create.side_effect = APITimeoutError("Request timeout")
            
            messages = [{"role": "user", "content": "Hello"}]
            
            with pytest.raises(RetryError):
                await client.create_completion(messages)

    @pytest.mark.asyncio
    async def test_create_streaming_completion(self, client):
        """测试流式聊天完成"""
        async def mock_stream():
            for content in ["Hello", " world", "!"]:
                mock_chunk = Mock()
                mock_chunk.choices = [Mock()]
                mock_chunk.choices[0].delta.content = content
                mock_chunk.choices[0].finish_reason = None if content != "!" else "stop"
                yield mock_chunk
        
        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_stream()
            
            messages = [{"role": "user", "content": "Hello"}]
            chunks = []
            
            async for chunk in client.create_streaming_completion(messages):
                chunks.append(chunk)
            
            assert len(chunks) == 3
            assert chunks[0]["content"] == "Hello"
            assert chunks[1]["content"] == " world"
            assert chunks[2]["content"] == "!"
            assert chunks[2]["finish_reason"] == "stop"

    @pytest.mark.asyncio
    async def test_create_embeddings(self, client):
        """测试创建嵌入向量"""
        mock_embedding = Mock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        
        mock_response = Mock()
        mock_response.data = [mock_embedding]
        mock_response.usage.total_tokens = 10
        
        with patch.object(client.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            result = await client.create_embeddings("test text")
            
            assert result == [[0.1, 0.2, 0.3]]
            mock_create.assert_called_once_with(
                model="text-embedding-3-small",
                input=["test text"]
            )

    @pytest.mark.asyncio
    async def test_create_embeddings_multiple_texts(self, client):
        """测试创建多个文本的嵌入向量"""
        mock_embeddings = [
            Mock(embedding=[0.1, 0.2, 0.3]),
            Mock(embedding=[0.4, 0.5, 0.6])
        ]
        
        mock_response = Mock()
        mock_response.data = mock_embeddings
        mock_response.usage.total_tokens = 20
        
        with patch.object(client.client.embeddings, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_response
            
            texts = ["text1", "text2"]
            result = await client.create_embeddings(texts)
            
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
            mock_create.assert_called_once_with(
                model="text-embedding-3-small",
                input=texts
            )

    def test_format_messages_for_openai(self, client):
        """测试消息格式化"""
        system_prompt = "You are a helpful assistant"
        user_message = "Hello"
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"}
        ]
        
        result = client.format_messages_for_openai(system_prompt, user_message, history)
        
        expected = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello there!"},
            {"role": "user", "content": user_message}
        ]
        
        assert result == expected

    def test_format_tools_for_openai(self, client):
        """测试工具格式转换"""
        mcp_tools = [
            {
                "name": "test_tool",
                "description": "A test tool",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "param": {"type": "string"}
                    }
                }
            }
        ]
        
        result = client.format_tools_for_openai(mcp_tools)
        
        expected = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "param": {"type": "string"}
                        }
                    }
                }
            }
        ]
        
        assert result == expected

    @pytest.mark.asyncio
    async def test_health_check_success(self, client, mock_openai_response):
        """测试健康检查成功"""
        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            mock_create.return_value = mock_openai_response
            
            result = await client.health_check()
            
            assert result["healthy"] is True
            assert result["model"] == "gpt-4o-mini"
            assert result["response_received"] is True
            assert result["tokens_used"] == 15
            assert "duration" in result

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        """测试健康检查失败"""
        with patch.object(client.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
            import httpx
            mock_request = Mock(spec=httpx.Request)
            mock_create.side_effect = APIError("API error", request=mock_request, body=None)
            
            result = await client.health_check()
            
            assert result["healthy"] is False
            assert result["error"] == "API error"
            assert result["error_type"] == "APIError"
            assert "duration" in result

    @pytest.mark.asyncio
    async def test_get_openai_client_singleton(self):
        """测试全局客户端单例"""
        with patch("src.ai.openai_client.get_settings") as mock_settings:
            mock_settings.return_value.OPENAI_API_KEY = "test-api-key"
            
            client1 = await get_openai_client()
            client2 = await get_openai_client()
            
            assert client1 is client2