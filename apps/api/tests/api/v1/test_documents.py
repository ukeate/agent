"""文档处理API测试"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
import json
from unittest.mock import Mock, patch, AsyncMock
import base64

# 假设的main应用导入
from src.main import app


@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)


@pytest.fixture
def sample_pdf_file():
    """创建示例PDF文件"""
    content = b"%PDF-1.4\n%Sample PDF content"
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    yield tmp_path
    Path(tmp_path).unlink()


@pytest.fixture
def sample_docx_file():
    """创建示例Word文件"""
    # 简化的DOCX文件内容
    content = b"PK\x03\x04"  # ZIP文件头
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    yield tmp_path
    Path(tmp_path).unlink()


@pytest.fixture
def sample_txt_file():
    """创建示例文本文件"""
    content = b"This is a sample text document for testing."
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    yield tmp_path
    Path(tmp_path).unlink()


class TestDocumentUpload:
    """文档上传API测试"""
    
    @patch('src.api.v1.documents.DocumentProcessor')
    def test_upload_single_document(self, mock_processor, client, sample_pdf_file):
        """测试单文档上传"""
        # 配置mock
        mock_instance = mock_processor.return_value
        mock_instance.process_document = AsyncMock(return_value=Mock(
            doc_id="doc123",
            title="Test Document",
            file_type="pdf",
            content="Sample content",
            metadata={"pages": 10},
            processing_info={},
            to_dict=lambda: {
                "doc_id": "doc123",
                "title": "Test Document",
                "file_type": "pdf",
                "content": "Sample content",
                "metadata": {"pages": 10},
                "processing_info": {}
            }
        ))
        
        # 上传文件
        with open(sample_pdf_file, "rb") as f:
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("test.pdf", f, "application/pdf")},
                params={
                    "enable_ocr": False,
                    "extract_images": True,
                    "auto_tag": True,
                    "chunk_strategy": "semantic"
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc123"
        assert data["file_type"] == "pdf"
        assert "version" in data
    
    def test_upload_invalid_file_type(self, client):
        """测试上传不支持的文件类型"""
        content = b"Invalid file content"
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.exe", content, "application/x-msdownload")}
        )
        
        # 应该返回错误
        assert response.status_code in [400, 500]
    
    @patch('src.api.v1.documents.DocumentProcessor')
    def test_upload_with_ocr_enabled(self, mock_processor, client, sample_pdf_file):
        """测试启用OCR的文档上传"""
        mock_instance = mock_processor.return_value
        mock_instance.process_document = AsyncMock(return_value=Mock(
            doc_id="doc456",
            title="OCR Document",
            file_type="pdf",
            content="OCR extracted text",
            metadata={"ocr_enabled": True},
            processing_info={"ocr_pages": [1, 2, 3]},
            to_dict=lambda: {
                "doc_id": "doc456",
                "title": "OCR Document",
                "content": "OCR extracted text",
                "metadata": {"ocr_enabled": True}
            }
        ))
        
        with open(sample_pdf_file, "rb") as f:
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("scan.pdf", f, "application/pdf")},
                params={"enable_ocr": True}
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["metadata"]["ocr_enabled"] is True


class TestBatchUpload:
    """批量上传API测试"""
    
    @patch('src.api.v1.documents.document_processor')
    def test_batch_upload_multiple_files(
        self, mock_processor, client, sample_pdf_file, sample_txt_file
    ):
        """测试批量文件上传"""
        # 配置mock
        mock_processor.process_batch = AsyncMock(return_value=[
            Mock(doc_id="doc1", title="File 1", file_type="pdf"),
            Mock(doc_id="doc2", title="File 2", file_type="txt")
        ])
        
        # 批量上传
        files = [
            ("files", ("file1.pdf", open(sample_pdf_file, "rb"), "application/pdf")),
            ("files", ("file2.txt", open(sample_txt_file, "rb"), "text/plain"))
        ]
        
        response = client.post(
            "/api/v1/documents/batch-upload",
            files=files,
            params={
                "concurrent_limit": 5,
                "continue_on_error": True
            }
        )
        
        # 关闭文件
        for _, file_tuple in files:
            file_tuple[1].close()
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert data["success"] == 2
        assert data["failed"] == 0
    
    @patch('src.api.v1.documents.document_processor')
    def test_batch_upload_with_errors(
        self, mock_processor, client, sample_pdf_file
    ):
        """测试批量上传错误处理"""
        # 模拟部分失败
        mock_processor.process_batch = AsyncMock(return_value=[
            Mock(doc_id="doc1", title="Success", file_type="pdf")
        ])
        
        files = [
            ("files", ("good.pdf", open(sample_pdf_file, "rb"), "application/pdf")),
            ("files", ("bad.pdf", b"corrupted", "application/pdf"))
        ]
        
        response = client.post(
            "/api/v1/documents/batch-upload",
            files=files,
            params={"continue_on_error": True}
        )
        
        files[0][1][1].close()
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] >= 1


class TestDocumentRelationships:
    """文档关系分析API测试"""
    
    @patch('src.api.v1.documents.relationship_analyzer')
    def test_analyze_relationships(self, mock_analyzer, client):
        """测试文档关系分析"""
        # 配置mock
        mock_analyzer.analyze_relationships = AsyncMock(return_value={
            "relationships": [
                Mock(
                    source_doc_id="doc1",
                    target_doc_id="doc2",
                    relationship_type=Mock(value="reference"),
                    confidence=0.95
                )
            ],
            "clusters": [
                Mock(
                    cluster_id="cluster1",
                    documents=["doc1", "doc2"],
                    topic="Machine Learning"
                )
            ],
            "summary": {"total_relationships": 1}
        })
        
        response = client.post(
            "/api/v1/documents/doc1/analyze-relationships",
            params={"related_doc_ids": ["doc2", "doc3"]}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc1"
        assert len(data["relationships"]) > 0
        assert data["relationships"][0]["type"] == "reference"
        assert len(data["clusters"]) > 0


class TestDocumentTags:
    """文档标签API测试"""
    
    @patch('src.api.v1.documents.auto_tagger')
    def test_generate_tags(self, mock_tagger, client):
        """测试标签生成"""
        # 配置mock
        mock_tagger.generate_tags = AsyncMock(return_value=[
            Mock(
                tag="Python",
                category=Mock(value="language"),
                confidence=0.9,
                source="content"
            ),
            Mock(
                tag="Machine Learning",
                category=Mock(value="topic"),
                confidence=0.85,
                source="keywords"
            )
        ])
        
        response = client.post(
            "/api/v1/documents/doc123/generate-tags",
            params={
                "content": "Python machine learning tutorial with TensorFlow",
                "existing_tags": ["tutorial"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc123"
        assert len(data["tags"]) == 2
        assert data["tags"][0]["tag"] == "Python"
        assert data["tags"][0]["category"] == "language"


class TestDocumentVersions:
    """文档版本管理API测试"""
    
    @patch('src.api.v1.documents.version_manager')
    def test_get_version_history(self, mock_version_manager, client):
        """测试获取版本历史"""
        from datetime import datetime
        
        # 配置mock
        mock_version_manager.get_version_history = AsyncMock(return_value=[
            Mock(
                version_id="v1",
                version_number=1,
                created_at=datetime(2024, 1, 1, 12, 0),
                change_summary="Initial version",
                is_current=False
            ),
            Mock(
                version_id="v2",
                version_number=2,
                created_at=datetime(2024, 1, 2, 12, 0),
                change_summary="Updated content",
                is_current=True
            )
        ])
        
        response = client.get(
            "/api/v1/documents/doc123/versions",
            params={"limit": 10}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc123"
        assert len(data["versions"]) == 2
        assert data["versions"][0]["version_number"] == 1
        assert data["versions"][1]["is_current"] is True
    
    @patch('src.api.v1.documents.version_manager')
    def test_rollback_version(self, mock_version_manager, client):
        """测试版本回滚"""
        from datetime import datetime
        
        # 配置mock
        mock_version_manager.rollback_version = AsyncMock(return_value=Mock(
            version_id="v3",
            version_number=3,
            created_at=datetime(2024, 1, 3, 12, 0),
            change_summary="Rollback to v1"
        ))
        
        response = client.post(
            "/api/v1/documents/doc123/rollback",
            params={"target_version_id": "v1"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["doc_id"] == "doc123"
        assert data["new_version"]["version_number"] == 3
        assert "Rollback" in data["new_version"]["change_summary"]


class TestSupportedFormats:
    """支持格式API测试"""
    
    def test_get_supported_formats(self, client):
        """测试获取支持的文档格式"""
        response = client.get("/api/v1/documents/supported-formats")
        
        assert response.status_code == 200
        data = response.json()
        assert "formats" in data
        assert "categories" in data
        assert "documents" in data["categories"]
        assert ".pdf" in data["categories"]["documents"]
        assert ".docx" in data["categories"]["documents"]


class TestChunkingStrategies:
    """分块策略测试"""
    
    @patch('src.api.v1.documents.DocumentProcessor')
    @patch('src.api.v1.documents.chunker')
    def test_semantic_chunking(self, mock_chunker, mock_processor, client, sample_txt_file):
        """测试语义分块"""
        # 配置mock
        mock_processor.return_value.process_document = AsyncMock(return_value=Mock(
            doc_id="doc789",
            content="Long document content that needs chunking",
            file_type="txt",
            metadata={},
            processing_info={},
            to_dict=lambda: {"doc_id": "doc789", "content": "Long content"}
        ))
        
        mock_chunker.chunk_document = AsyncMock(return_value=[
            Mock(
                chunk_id="chunk1",
                content="First semantic chunk",
                chunk_type="paragraph",
                chunk_index=0
            ),
            Mock(
                chunk_id="chunk2",
                content="Second semantic chunk",
                chunk_type="paragraph",
                chunk_index=1
            )
        ])
        
        with open(sample_txt_file, "rb") as f:
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("long.txt", f, "text/plain")},
                params={"chunk_strategy": "semantic"}
            )
        
        assert response.status_code == 200
        mock_chunker.chunk_document.assert_called_once()


class TestErrorHandling:
    """错误处理测试"""
    
    def test_upload_empty_file(self, client):
        """测试上传空文件"""
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.txt", b"", "text/plain")}
        )
        
        # 应该处理空文件
        assert response.status_code in [200, 400, 500]
    
    def test_invalid_doc_id(self, client):
        """测试无效的文档ID"""
        response = client.get("/api/v1/documents/invalid_id/versions")
        
        # 应该返回适当的错误
        assert response.status_code in [200, 404, 500]
    
    @patch('src.api.v1.documents.document_processor')
    def test_processing_timeout(self, mock_processor, client, sample_pdf_file):
        """测试处理超时"""
        # 模拟超时错误
        mock_processor.process_document = AsyncMock(
            side_effect=TimeoutError("Processing timeout")
        )
        
        with open(sample_pdf_file, "rb") as f:
            response = client.post(
                "/api/v1/documents/upload",
                files={"file": ("slow.pdf", f, "application/pdf")}
            )
        
        assert response.status_code == 500


if __name__ == "__main__":
    pytest.main([__file__, "-v"])