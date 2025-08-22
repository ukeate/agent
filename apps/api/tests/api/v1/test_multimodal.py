"""
多模态API路由测试
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
import base64
import json

from src.ai.multimodal.types import (
    MultimodalContent, ContentType, ProcessingStatus,
    ProcessingResult, ProcessingOptions
)


@pytest.mark.asyncio
class TestMultimodalAPI:
    """多模态API测试"""
    
    @pytest.fixture
    def client(self, test_app):
        """创建测试客户端"""
        return TestClient(test_app)
    
    async def test_upload_file(self, client):
        """测试文件上传"""
        with patch('src.api.v1.multimodal.get_file_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.save_uploaded_file = AsyncMock(return_value=MultimodalContent(
                content_id="file_123",
                content_type=ContentType.IMAGE,
                file_size=1024,
                mime_type="image/jpeg",
                metadata={"original_filename": "test.jpg"}
            ))
            mock_get_service.return_value = mock_service
            
            # 创建测试文件
            test_file = ("test.jpg", b"fake_image_data", "image/jpeg")
            
            response = client.post(
                "/api/v1/multimodal/upload",
                files={"file": test_file}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["content_id"] == "file_123"
            assert data["content_type"] == "image"
            assert data["file_size"] == 1024
    
    async def test_upload_file_too_large(self, client):
        """测试上传过大文件"""
        # 创建超过100MB的文件
        large_file = ("large.jpg", b"x" * (101 * 1024 * 1024), "image/jpeg")
        
        response = client.post(
            "/api/v1/multimodal/upload",
            files={"file": large_file}
        )
        
        assert response.status_code == 413
        assert "文件大小超过100MB限制" in response.json()["detail"]
    
    async def test_process_content(self, client):
        """测试内容处理"""
        with patch('src.api.v1.multimodal.get_processor') as mock_get_processor:
            with patch('src.api.v1.multimodal.get_file_service') as mock_get_service:
                # 模拟文件服务
                mock_service = AsyncMock()
                mock_service.get_file_info = AsyncMock(return_value={
                    "file_path": "/tmp/test.jpg",
                    "file_size": 1024
                })
                mock_get_service.return_value = mock_service
                
                # 模拟处理器
                mock_processor = AsyncMock()
                mock_processor.process_content = AsyncMock(return_value=ProcessingResult(
                    content_id="file_123",
                    status=ProcessingStatus.COMPLETED,
                    extracted_data={"description": "测试图像"},
                    confidence_score=0.9,
                    processing_time=1.5,
                    model_used="gpt-4o",
                    tokens_used={"total_tokens": 100}
                ))
                mock_get_processor.return_value = mock_processor
                
                # 发送处理请求
                response = client.post(
                    "/api/v1/multimodal/process",
                    json={
                        "content_id": "file_123",
                        "content_type": "image",
                        "priority": "balanced",
                        "complexity": "medium"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["content_id"] == "file_123"
                assert data["status"] == "completed"
                assert data["confidence_score"] == 0.9
                assert data["model_used"] == "gpt-4o"
    
    async def test_process_content_file_not_found(self, client):
        """测试处理不存在的文件"""
        with patch('src.api.v1.multimodal.get_file_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.get_file_info = AsyncMock(return_value=None)
            mock_get_service.return_value = mock_service
            
            response = client.post(
                "/api/v1/multimodal/process",
                json={
                    "content_id": "nonexistent",
                    "content_type": "image"
                }
            )
            
            assert response.status_code == 404
            assert "文件未找到" in response.json()["detail"]
    
    async def test_batch_processing(self, client):
        """测试批量处理"""
        with patch('src.api.v1.multimodal.get_pipeline') as mock_get_pipeline:
            with patch('src.api.v1.multimodal.get_file_service') as mock_get_service:
                # 模拟文件服务
                mock_service = AsyncMock()
                mock_service.get_file_info = AsyncMock(side_effect=[
                    {"file_path": "/tmp/file1.jpg", "file_size": 1024},
                    {"file_path": "/tmp/file2.jpg", "file_size": 2048}
                ])
                mock_get_service.return_value = mock_service
                
                # 模拟处理管道
                mock_pipeline = AsyncMock()
                mock_pipeline.submit_batch = AsyncMock(return_value=[
                    "file_001", "file_002"
                ])
                mock_get_pipeline.return_value = mock_pipeline
                
                # 发送批量处理请求
                response = client.post(
                    "/api/v1/multimodal/process/batch",
                    json={
                        "content_ids": ["file_001", "file_002"],
                        "priority": "balanced"
                    }
                )
                
                assert response.status_code == 200
                data = response.json()
                assert data["batch_id"].startswith("batch_")
                assert len(data["content_ids"]) == 2
                assert data["status"] == "processing"
                assert data["total_items"] == 2
    
    async def test_get_processing_status(self, client):
        """测试获取处理状态"""
        with patch('src.api.v1.multimodal.get_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.get_processing_status = AsyncMock(return_value=ProcessingResult(
                content_id="file_123",
                status=ProcessingStatus.COMPLETED,
                extracted_data={},
                confidence_score=0.85,
                processing_time=2.0,
                model_used="gpt-4o-mini"
            ))
            mock_get_pipeline.return_value = mock_pipeline
            
            response = client.get("/api/v1/multimodal/status/file_123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["content_id"] == "file_123"
            assert data["status"] == "completed"
            assert data["confidence_score"] == 0.85
    
    async def test_get_processing_status_not_found(self, client):
        """测试获取不存在的处理状态"""
        with patch('src.api.v1.multimodal.get_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.get_processing_status = AsyncMock(return_value=None)
            mock_get_pipeline.return_value = mock_pipeline
            
            response = client.get("/api/v1/multimodal/status/nonexistent")
            
            assert response.status_code == 404
            assert "处理任务未找到" in response.json()["detail"]
    
    async def test_get_queue_status(self, client):
        """测试获取队列状态"""
        with patch('src.api.v1.multimodal.get_pipeline') as mock_get_pipeline:
            mock_pipeline = AsyncMock()
            mock_pipeline.get_queue_status = MagicMock(return_value={
                "is_running": True,
                "active_tasks": 2,
                "queued_tasks": 5,
                "completed_tasks": 10,
                "failed_tasks": 1
            })
            mock_get_pipeline.return_value = mock_pipeline
            
            response = client.get("/api/v1/multimodal/queue/status")
            
            assert response.status_code == 200
            data = response.json()
            assert data["is_running"] is True
            assert data["active_tasks"] == 2
            assert data["queued_tasks"] == 5
    
    async def test_analyze_image_direct(self, client):
        """测试直接分析图像"""
        with patch('src.api.v1.multimodal.get_multimodal_client') as mock_get_client:
            mock_client = AsyncMock()
            mock_client.process_image = AsyncMock(return_value={
                "content": json.dumps({
                    "description": "直接分析的图像",
                    "objects": ["对象1"],
                    "confidence": 0.92
                }),
                "model": "gpt-4o",
                "usage": {"total_tokens": 80},
                "cost": 0.002,
                "duration": 1.2
            })
            mock_get_client.return_value = mock_client
            
            # 创建测试图像
            test_image = ("test.jpg", b"fake_image", "image/jpeg")
            
            response = client.post(
                "/api/v1/multimodal/analyze/image",
                files={"file": test_image},
                data={
                    "prompt": "分析这张图像",
                    "extract_text": "true",
                    "priority": "balanced"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "extracted_data" in data
            assert data["extracted_data"]["description"] == "直接分析的图像"
            assert data["model_used"] == "gpt-4o"
            assert data["cost"] == 0.002
    
    async def test_delete_file(self, client):
        """测试删除文件"""
        with patch('src.api.v1.multimodal.get_file_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.delete_file = AsyncMock(return_value=True)
            mock_get_service.return_value = mock_service
            
            response = client.delete("/api/v1/multimodal/file/file_123")
            
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "文件删除成功"
            assert data["content_id"] == "file_123"
    
    async def test_delete_file_not_found(self, client):
        """测试删除不存在的文件"""
        with patch('src.api.v1.multimodal.get_file_service') as mock_get_service:
            mock_service = AsyncMock()
            mock_service.delete_file = AsyncMock(return_value=False)
            mock_get_service.return_value = mock_service
            
            response = client.delete("/api/v1/multimodal/file/nonexistent")
            
            assert response.status_code == 404
            assert "文件未找到" in response.json()["detail"]