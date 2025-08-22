"""
OpenAI多模态客户端测试
"""

import pytest
import base64
from unittest.mock import AsyncMock, MagicMock, patch
from src.ai.multimodal.client import OpenAIMultimodalClient, ModelSelector
from src.ai.multimodal.types import ModelPriority, ModelComplexity


class TestModelSelector:
    """模型选择器测试"""
    
    def test_select_model_for_cost_priority(self):
        """测试成本优先的模型选择"""
        # 简单任务，成本优先
        model = ModelSelector.select_model(
            "image",
            priority=ModelPriority.COST,
            complexity=ModelComplexity.SIMPLE
        )
        assert model == "gpt-5-nano"
        
        # 复杂任务，成本优先
        model = ModelSelector.select_model(
            "document",
            priority=ModelPriority.COST,
            complexity=ModelComplexity.COMPLEX
        )
        assert model == "gpt-4o-mini"
    
    def test_select_model_for_quality_priority(self):
        """测试质量优先的模型选择"""
        # 复杂任务，质量优先
        model = ModelSelector.select_model(
            "image",
            priority=ModelPriority.QUALITY,
            complexity=ModelComplexity.COMPLEX
        )
        assert model in ["gpt-5", "gpt-4o"]  # 取决于是否有GPT-5
        
        # 简单任务，质量优先
        model = ModelSelector.select_model(
            "document",
            priority=ModelPriority.QUALITY,
            complexity=ModelComplexity.SIMPLE
        )
        assert model == "gpt-4o"
    
    def test_select_model_for_speed_priority(self):
        """测试速度优先的模型选择"""
        model = ModelSelector.select_model(
            "image",
            priority=ModelPriority.SPEED,
            complexity=ModelComplexity.MEDIUM
        )
        assert model in ["gpt-5-nano", "gpt-4o-mini"]
    
    def test_select_model_for_pdf_content(self):
        """测试PDF内容的模型选择"""
        model = ModelSelector.select_model(
            "pdf",
            priority=ModelPriority.BALANCED,
            complexity=ModelComplexity.MEDIUM
        )
        assert model == "gpt-4o"  # PDF优先使用GPT-4o
    
    def test_select_model_with_file_upload_requirement(self):
        """测试需要文件上传功能的模型选择"""
        model = ModelSelector.select_model(
            "document",
            priority=ModelPriority.COST,
            complexity=ModelComplexity.SIMPLE,
            requires_file_upload=True
        )
        # gpt-5-nano不支持文件上传，应该选择gpt-4o-mini
        assert model == "gpt-4o-mini"
    
    def test_get_model_cost(self):
        """测试模型成本计算"""
        cost = ModelSelector.get_model_cost("gpt-4o-mini", 1000, 500)
        expected_input_cost = (1000 / 1000) * 0.00015
        expected_output_cost = (500 / 1000) * 0.0006
        assert cost == pytest.approx(expected_input_cost + expected_output_cost)


@pytest.mark.asyncio
class TestOpenAIMultimodalClient:
    """OpenAI多模态客户端测试"""
    
    async def test_client_initialization(self):
        """测试客户端初始化"""
        client = OpenAIMultimodalClient(api_key="test_key")
        assert client.api_key == "test_key"
        assert client.model_selector is not None
        assert client.base_url == "https://api.openai.com/v1"
    
    @patch('src.ai.multimodal.client.AsyncOpenAI')
    async def test_process_image(self, mock_openai):
        """测试图像处理"""
        # 模拟OpenAI响应
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"description": "测试图像", "confidence": 0.9}'
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # 创建客户端
        client = OpenAIMultimodalClient(api_key="test_key")
        client.client = mock_client
        
        # 测试图像数据
        test_image = b"fake_image_data"
        
        # 处理图像
        result = await client.process_image(
            test_image,
            "描述这张图像",
            max_tokens=100,
            priority=ModelPriority.BALANCED,
            complexity=ModelComplexity.MEDIUM
        )
        
        # 验证结果
        assert result is not None
        assert "content" in result
        assert "model" in result
        assert "usage" in result
        assert result["usage"]["total_tokens"] == 150
    
    @patch('src.ai.multimodal.client.AsyncOpenAI')
    async def test_process_document(self, mock_openai):
        """测试文档处理"""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"summary": "测试文档摘要", "confidence": 0.85}'
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage.prompt_tokens = 200
        mock_response.usage.completion_tokens = 100
        mock_response.usage.total_tokens = 300
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        client = OpenAIMultimodalClient(api_key="test_key")
        client.client = mock_client
        
        # 处理文档
        result = await client.process_document(
            document_text="这是测试文档内容",
            prompt="总结这份文档",
            max_tokens=200
        )
        
        assert result is not None
        assert "content" in result
        assert result["usage"]["total_tokens"] == 300
    
    @patch('src.ai.multimodal.client.AsyncOpenAI')
    async def test_process_document_with_file_id(self, mock_openai):
        """测试使用文件ID处理文档"""
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"summary": "PDF文档摘要"}'
        mock_response.choices[0].finish_reason = "stop"
        mock_response.usage = MagicMock()
        
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        client = OpenAIMultimodalClient(api_key="test_key")
        client.client = mock_client
        
        # 使用文件ID处理
        result = await client.process_document(
            file_id="file-abc123",
            prompt="分析PDF文档"
        )
        
        assert result is not None
        assert "content" in result
    
    @patch('aiohttp.ClientSession')
    async def test_upload_file(self, mock_session_class):
        """测试文件上传"""
        # 创建模拟session
        mock_session = AsyncMock()
        mock_session_class.return_value = mock_session
        
        # 模拟上传响应
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "id": "file-xyz789",
            "object": "file",
            "purpose": "assistants"
        })
        
        mock_session.post = AsyncMock(return_value=mock_response)
        mock_session.__aenter__ = AsyncMock(return_value=mock_response)
        mock_session.__aexit__ = AsyncMock()
        
        client = OpenAIMultimodalClient(api_key="test_key")
        client.upload_session = mock_session
        
        # 创建临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            f.write(b"fake pdf content")
            temp_path = f.name
        
        try:
            # 上传文件
            result = await client.upload_file(temp_path, "assistants")
            
            assert result is not None
            assert result["id"] == "file-xyz789"
            assert result["purpose"] == "assistants"
        finally:
            import os
            os.remove(temp_path)
    
    async def test_process_video_frame(self):
        """测试视频帧处理"""
        with patch.object(OpenAIMultimodalClient, 'process_image') as mock_process:
            mock_process.return_value = {
                "content": "帧描述",
                "model": "gpt-4o",
                "usage": {"total_tokens": 50}
            }
            
            client = OpenAIMultimodalClient(api_key="test_key")
            
            frame_data = b"fake_frame_data"
            result = await client.process_video_frame(
                frame_data,
                0,
                "描述这一帧"
            )
            
            assert result is not None
            mock_process.assert_called_once()
    
    async def test_health_check_multimodal(self):
        """测试多模态健康检查"""
        with patch.object(OpenAIMultimodalClient, 'process_image') as mock_process:
            mock_process.return_value = {
                "content": "一个像素",
                "model": "gpt-4o-mini",
                "duration": 0.5
            }
            
            client = OpenAIMultimodalClient(api_key="test_key")
            result = await client.health_check_multimodal()
            
            assert result["healthy"] is True
            assert result["multimodal_enabled"] is True
            assert result["model_used"] == "gpt-4o-mini"
    
    async def test_error_handling(self):
        """测试错误处理"""
        client = OpenAIMultimodalClient(api_key="test_key")
        
        # 测试无效输入
        with pytest.raises(ValueError):
            await client.process_document()  # 没有提供文本或文件ID