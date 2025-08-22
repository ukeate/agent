"""
多模态处理器测试
"""

import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path
from datetime import datetime, timezone

from src.ai.multimodal.processor import MultimodalProcessor
from src.ai.multimodal.types import (
    MultimodalContent, ContentType, ProcessingOptions,
    ProcessingStatus, ModelPriority, ModelComplexity
)


@pytest.mark.asyncio
class TestMultimodalProcessor:
    """多模态处理器测试"""
    
    @pytest.fixture
    async def processor(self):
        """创建处理器实例"""
        mock_client = AsyncMock()
        processor = MultimodalProcessor(mock_client, "/tmp/test_multimodal")
        return processor
    
    async def test_process_image_content(self, processor):
        """测试图像内容处理"""
        # 模拟OpenAI客户端响应
        processor.openai.process_image = AsyncMock(return_value={
            "content": json.dumps({
                "description": "测试图像描述",
                "objects": ["对象1", "对象2"],
                "text_content": "图像中的文本",
                "sentiment": "positive",
                "confidence": 0.9
            }),
            "model": "gpt-4o",
            "usage": {"total_tokens": 100}
        })
        
        # 创建测试内容
        content = MultimodalContent(
            content_id="test_image_001",
            content_type=ContentType.IMAGE,
            file_path="/tmp/test.jpg",
            file_size=1024,
            mime_type="image/jpeg"
        )
        
        # 模拟文件读取
        with patch('aiofiles.open', new_callable=mock_open, read_data=b'fake_image_data'):
            # 处理内容
            result = await processor.process_content(content)
        
        # 验证结果
        assert result.status == ProcessingStatus.COMPLETED
        assert result.content_id == "test_image_001"
        assert result.confidence_score == 0.9
        assert "description" in result.extracted_data
        assert result.extracted_data["objects"] == ["对象1", "对象2"]
        assert result.model_used == "gpt-4o"
    
    async def test_process_document_content(self, processor):
        """测试文档内容处理"""
        # 模拟文档文本提取
        processor._extract_document_text = AsyncMock(return_value="测试文档内容")
        
        # 模拟OpenAI响应
        processor.openai.process_document = AsyncMock(return_value={
            "content": json.dumps({
                "summary": "文档摘要",
                "key_points": ["要点1", "要点2"],
                "document_type": "report",
                "confidence": 0.85
            }),
            "model": "gpt-4o",
            "usage": {"total_tokens": 200}
        })
        
        content = MultimodalContent(
            content_id="test_doc_001",
            content_type=ContentType.DOCUMENT,
            file_path="/tmp/test.pdf",
            file_size=2048
        )
        
        result = await processor.process_content(content)
        
        assert result.status == ProcessingStatus.COMPLETED
        assert result.extracted_data["summary"] == "文档摘要"
        assert len(result.extracted_data["key_points"]) == 2
        assert result.confidence_score == 0.85
    
    async def test_process_video_content(self, processor):
        """测试视频内容处理"""
        # 模拟视频帧提取
        processor._extract_video_frames = AsyncMock(return_value=[
            b'frame1', b'frame2', b'frame3'
        ])
        
        # 模拟帧处理
        processor.openai.process_video_frame = AsyncMock(side_effect=[
            {"content": "第1帧: 开场画面", "usage": {"total_tokens": 50}},
            {"content": "第2帧: 主要内容", "usage": {"total_tokens": 50}},
            {"content": "第3帧: 结束画面", "usage": {"total_tokens": 50}}
        ])
        
        content = MultimodalContent(
            content_id="test_video_001",
            content_type=ContentType.VIDEO,
            file_path="/tmp/test.mp4",
            file_size=10240
        )
        
        result = await processor.process_content(content)
        
        assert result.status == ProcessingStatus.COMPLETED
        assert result.extracted_data["frame_count"] == 3
        assert len(result.extracted_data["key_frames"]) == 3
        assert result.extracted_data["key_frames"][0]["analysis"] == "第1帧: 开场画面"
    
    async def test_process_with_cache(self, processor):
        """测试缓存功能"""
        # 模拟缓存命中
        cached_result = {
            'content_id': 'test_cached',
            'extracted_data': {'cached': True},
            'confidence_score': 0.95,
            'processing_time': 0.1,
            'model_used': 'gpt-4o',
            'tokens_used': {'total_tokens': 100}
        }
        
        processor._check_cache = AsyncMock(return_value=MagicMock(
            content_id='test_cached',
            status=ProcessingStatus.CACHED,
            extracted_data={'cached': True},
            confidence_score=0.95,
            processing_time=0.1
        ))
        
        content = MultimodalContent(
            content_id="test_cached",
            content_type=ContentType.IMAGE,
            file_path="/tmp/cached.jpg"
        )
        
        options = ProcessingOptions(enable_cache=True)
        result = await processor.process_content(content, options)
        
        assert result.status == ProcessingStatus.CACHED
        assert result.extracted_data['cached'] is True
    
    async def test_process_with_options(self, processor):
        """测试处理选项"""
        processor.openai.process_image = AsyncMock(return_value={
            "content": json.dumps({
                "description": "详细描述",
                "confidence": 0.88
            }),
            "model": "gpt-5",
            "usage": {"total_tokens": 150}
        })
        
        content = MultimodalContent(
            content_id="test_options",
            content_type=ContentType.IMAGE,
            file_path="/tmp/options.jpg"
        )
        
        options = ProcessingOptions(
            priority=ModelPriority.QUALITY,
            complexity=ModelComplexity.COMPLEX,
            max_tokens=2000,
            temperature=0.2,
            extract_sentiment=True
        )
        
        with patch('aiofiles.open', new_callable=mock_open, read_data=b'image'):
            result = await processor.process_content(content, options)
        
        assert result.status == ProcessingStatus.COMPLETED
        assert result.model_used == "gpt-5"
    
    async def test_error_handling(self, processor):
        """测试错误处理"""
        # 模拟处理错误
        processor.openai.process_image = AsyncMock(side_effect=Exception("API错误"))
        
        content = MultimodalContent(
            content_id="test_error",
            content_type=ContentType.IMAGE,
            file_path="/tmp/error.jpg"
        )
        
        with patch('aiofiles.open', new_callable=mock_open, read_data=b'image'):
            result = await processor.process_content(content)
        
        assert result.status == ProcessingStatus.FAILED
        assert result.error_message == "API错误"
        assert result.confidence_score == 0.0
    
    async def test_extract_document_text_pdf(self, processor):
        """测试PDF文本提取"""
        with patch('PyPDF2.PdfReader') as mock_pdf:
            mock_reader = MagicMock()
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "PDF内容"
            mock_reader.pages = [mock_page]
            mock_pdf.return_value = mock_reader
            
            with patch('builtins.open', mock_open()):
                text = await processor._extract_document_text("/tmp/test.pdf")
            
            assert "PDF内容" in text
    
    async def test_extract_document_text_docx(self, processor):
        """测试DOCX文本提取"""
        with patch('docx.Document') as mock_doc:
            mock_document = MagicMock()
            mock_paragraph = MagicMock()
            mock_paragraph.text = "Word文档内容"
            mock_document.paragraphs = [mock_paragraph]
            mock_doc.return_value = mock_document
            
            text = await processor._extract_document_text("/tmp/test.docx")
            
            assert "Word文档内容" in text
    
    async def test_extract_document_text_txt(self, processor):
        """测试纯文本提取"""
        with patch('aiofiles.open', new_callable=mock_open, read_data="纯文本内容"):
            text = await processor._extract_document_text("/tmp/test.txt")
        
        assert text == "纯文本内容"
    
    @patch('cv2.VideoCapture')
    async def test_extract_video_frames(self, mock_capture, processor):
        """测试视频帧提取"""
        # 模拟视频捕获
        mock_cap = MagicMock()
        mock_capture.return_value = mock_cap
        
        # 模拟视频属性
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_COUNT: 100
        }.get(prop, 0)
        
        mock_cap.isOpened.return_value = True
        
        # 模拟帧读取
        fake_frame = MagicMock()
        mock_cap.read.side_effect = [
            (True, fake_frame),
            (True, fake_frame),
            (True, fake_frame),
            (False, None)
        ]
        
        # 模拟帧编码
        with patch('cv2.imencode') as mock_encode:
            mock_encode.return_value = (True, MagicMock(tobytes=lambda: b'frame_data'))
            
            frames = await processor._extract_video_frames("/tmp/test.mp4", max_frames=3)
        
        assert len(frames) == 3
        assert all(frame == b'frame_data' for frame in frames)
        mock_cap.release.assert_called_once()