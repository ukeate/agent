"""
知识抽取API测试

测试FastAPI端点和请求/响应处理
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI

from ai.knowledge_graph.knowledge_extraction import router
from ai.knowledge_graph.data_models import (
    Entity, Relation, EntityType, RelationType,
    ExtractionRequest, ExtractionResponse,
    BatchProcessingRequest, BatchProcessingResponse
)


# 创建测试应用
app = FastAPI()
app.include_router(router)

client = TestClient(app)


class TestHealthEndpoint:
    """健康检查端点测试"""
    
    def test_health_check_healthy(self):
        """测试健康状态检查"""
        with patch('ai.knowledge_graph.knowledge_extraction.entity_recognizer', Mock()):
            with patch('ai.knowledge_graph.knowledge_extraction.relation_extractor', Mock()):
                with patch('ai.knowledge_graph.knowledge_extraction.entity_linker', Mock()):
                    with patch('ai.knowledge_graph.knowledge_extraction.multilingual_processor', Mock()):
                        with patch('ai.knowledge_graph.knowledge_extraction.batch_processor', Mock()):
                            response = client.get("/knowledge/health")
                            
                            assert response.status_code == 200
                            data = response.json()
                            
                            assert "status" in data
                            assert "version" in data
                            assert "components" in data
                            assert "uptime_seconds" in data
                            assert "timestamp" in data
    
    def test_health_check_unhealthy(self):
        """测试不健康状态检查"""
        # 模拟组件未初始化
        with patch('ai.knowledge_graph.knowledge_extraction.entity_recognizer', None):
            response = client.get("/knowledge/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert "entity_recognizer" in data["components"]
            assert data["components"]["entity_recognizer"] == "not_initialized"


class TestMetricsEndpoint:
    """系统指标端点测试"""
    
    def test_get_system_metrics(self):
        """测试获取系统指标"""
        with patch('ai.knowledge_graph.knowledge_extraction.metrics_collector') as mock_collector:
            mock_metrics = Mock()
            mock_metrics.total_requests = 100
            mock_metrics.successful_requests = 95
            mock_metrics.failed_requests = 5
            mock_metrics.average_response_time = 1.5
            mock_metrics.entities_extracted = 500
            mock_metrics.relations_extracted = 200
            mock_collector.get_metrics.return_value = mock_metrics
            
            response = client.get("/knowledge/metrics")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["total_requests"] == 100
            assert data["successful_requests"] == 95
            assert data["failed_requests"] == 5
            assert data["average_response_time"] == 1.5


class TestExtractionEndpoint:
    """知识抽取端点测试"""
    
    @pytest.fixture
    def mock_components(self):
        """Mock组件依赖"""
        mock_entities = [
            Entity("张三", EntityType.PERSON, 0, 2, 0.9),
            Entity("苹果公司", EntityType.COMPANY, 3, 7, 0.95)
        ]
        
        mock_relations = [
            Relation(
                subject=mock_entities[0],
                predicate=RelationType.WORKS_FOR,
                object=mock_entities[1],
                confidence=0.85,
                context="张三在苹果公司工作",
                source_sentence="张三在苹果公司工作"
            )
        ]
        
        mock_multilingual_result = Mock()
        mock_multilingual_result.entities = mock_entities
        mock_multilingual_result.relations = mock_relations
        mock_multilingual_result.detected_language.value = "zh"
        
        components = {
            "entity_recognizer": Mock(),
            "relation_extractor": Mock(),
            "entity_linker": Mock(),
            "multilingual_processor": Mock(),
            "batch_processor": Mock()
        }
        
        components["multilingual_processor"].process_multilingual_text = AsyncMock(
            return_value=mock_multilingual_result
        )
        components["entity_recognizer"].extract_entities = AsyncMock(
            return_value=mock_entities
        )
        components["relation_extractor"].extract_relations = AsyncMock(
            return_value=mock_relations
        )
        components["entity_linker"].link_entities = AsyncMock(
            return_value=mock_entities
        )
        
        return components
    
    def test_extract_knowledge_auto_language(self, mock_components):
        """测试自动语言检测的知识抽取"""
        request_data = {
            "text": "张三在苹果公司工作。",
            "language": "auto",
            "extract_entities": True,
            "extract_relations": True,
            "link_entities": True,
            "confidence_threshold": 0.5
        }
        
        with patch('ai.knowledge_graph.knowledge_extraction.get_components', return_value=mock_components):
            response = client.post("/knowledge/extract", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["document_id"] is not None
            assert data["text"] == "张三在苹果公司工作。"
            assert data["language"] == "zh"
            assert len(data["entities"]) == 2
            assert len(data["relations"]) == 1
            assert data["processing_time"] > 0
            
            # 检查实体数据
            entities = data["entities"]
            person_entity = next(e for e in entities if e["label"] == "PERSON")
            assert person_entity["text"] == "张三"
            assert person_entity["confidence"] == 0.9
            
            company_entity = next(e for e in entities if e["label"] == "COMPANY")
            assert company_entity["text"] == "苹果公司"
            assert company_entity["confidence"] == 0.95
            
            # 检查关系数据
            relations = data["relations"]
            work_relation = relations[0]
            assert work_relation["predicate"] == "WORKS_FOR"
            assert work_relation["subject"]["text"] == "张三"
            assert work_relation["object"]["text"] == "苹果公司"
            assert work_relation["confidence"] == 0.85
    
    def test_extract_knowledge_specific_language(self, mock_components):
        """测试指定语言的知识抽取"""
        request_data = {
            "text": "John Smith works for Apple Inc.",
            "language": "en",
            "extract_entities": True,
            "extract_relations": True,
            "link_entities": False,
            "confidence_threshold": 0.7
        }
        
        with patch('ai.knowledge_graph.knowledge_extraction.get_components', return_value=mock_components):
            response = client.post("/knowledge/extract", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["language"] == "en"
            assert data["text"] == "John Smith works for Apple Inc."
    
    def test_extract_knowledge_entities_only(self, mock_components):
        """测试仅提取实体"""
        request_data = {
            "text": "张三是一个人。",
            "language": "zh",
            "extract_entities": True,
            "extract_relations": False,
            "link_entities": False
        }
        
        # Mock仅返回实体
        mock_components["entity_recognizer"].extract_entities = AsyncMock(
            return_value=[Entity("张三", EntityType.PERSON, 0, 2, 0.9)]
        )
        
        with patch('ai.knowledge_graph.knowledge_extraction.get_components', return_value=mock_components):
            response = client.post("/knowledge/extract", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert len(data["entities"]) == 1
            assert len(data["relations"]) == 0  # 未提取关系
    
    def test_extract_knowledge_text_too_long(self):
        """测试文本长度超限"""
        request_data = {
            "text": "x" * 50001,  # 超过50000字符限制
            "language": "zh"
        }
        
        response = client.post("/knowledge/extract", json=request_data)
        
        assert response.status_code == 400
        assert "文本长度超过限制" in response.json()["detail"]
    
    def test_extract_knowledge_invalid_request(self):
        """测试无效请求"""
        # 缺少必需字段
        request_data = {}
        
        response = client.post("/knowledge/extract", json=request_data)
        
        assert response.status_code == 422  # Validation error
    
    def test_extract_knowledge_processing_error(self, mock_components):
        """测试处理错误"""
        request_data = {
            "text": "测试文本",
            "language": "zh"
        }
        
        # Mock处理异常
        mock_components["multilingual_processor"].process_multilingual_text = AsyncMock(
            side_effect=Exception("处理失败")
        )
        
        with patch('ai.knowledge_graph.knowledge_extraction.get_components', return_value=mock_components):
            response = client.post("/knowledge/extract", json=request_data)
            
            assert response.status_code == 500
            assert "知识抽取处理失败" in response.json()["detail"]


class TestBatchProcessingEndpoints:
    """批处理端点测试"""
    
    @pytest.fixture
    def mock_batch_processor(self):
        """Mock批处理器"""
        processor = Mock()
        processor.process_batch = AsyncMock(return_value="batch_123456")
        processor.get_processing_status = Mock(return_value={
            "is_running": True,
            "queue_size": 5,
            "active_tasks": 2,
            "worker_count": 4,
            "metrics": {
                "total_requests": 100,
                "successful_requests": 95,
                "average_processing_time": 2.5
            },
            "cache_stats": {
                "hit_rate": 0.85,
                "cache_size": 1000
            },
            "memory_usage_mb": 512.5
        })
        
        return processor
    
    def test_process_batch_success(self, mock_batch_processor):
        """测试成功的批处理"""
        request_data = {
            "documents": [
                {"id": "doc1", "text": "张三在苹果公司工作。"},
                {"id": "doc2", "text": "李四在谷歌工作。"},
                {"id": "doc3", "text": "王五在微软工作。"}
            ],
            "priority": 5,
            "language": "zh",
            "confidence_threshold": 0.7
        }
        
        mock_components = {"batch_processor": mock_batch_processor}
        
        with patch('ai.knowledge_graph.knowledge_extraction.get_components', return_value=mock_components):
            response = client.post("/knowledge/batch", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["batch_id"] == "batch_123456"
            assert data["status"] == "processing"
            assert data["total_documents"] == 3
            assert data["processed_documents"] == 0
            assert data["successful_documents"] == 0
            assert data["failed_documents"] == 0
    
    def test_process_batch_too_many_documents(self):
        """测试文档数量超限"""
        request_data = {
            "documents": [{"text": f"文档{i}"} for i in range(1001)],  # 超过1000限制
            "priority": 1
        }
        
        response = client.post("/knowledge/batch", json=request_data)
        
        assert response.status_code == 400
        assert "单次批处理文档数量不能超过1000" in response.json()["detail"]
    
    def test_process_batch_invalid_documents(self):
        """测试无效文档格式"""
        request_data = {
            "documents": [
                {"id": "doc1"},  # 缺少text字段
                {"id": "doc2", "text": "正常文档"}
            ],
            "priority": 1
        }
        
        response = client.post("/knowledge/batch", json=request_data)
        
        assert response.status_code == 400
        assert "缺少 'text' 字段" in response.json()["detail"]
    
    def test_process_batch_text_too_long(self):
        """测试文档文本过长"""
        request_data = {
            "documents": [
                {"id": "doc1", "text": "x" * 50001}  # 超过50000字符
            ],
            "priority": 1
        }
        
        response = client.post("/knowledge/batch", json=request_data)
        
        assert response.status_code == 400
        assert "文本长度超过限制" in response.json()["detail"]
    
    def test_get_batch_status_found(self, mock_batch_processor):
        """测试获取已存在的批处理状态"""
        # Mock任务调度器和任务
        mock_completed_task = Mock()
        mock_completed_task.task_id = "batch_123_task_0"
        mock_completed_task.status.value = "completed"
        mock_completed_task.result = {
            "document_id": "doc1",
            "entities": [],
            "relations": [],
            "processing_time": 1.5
        }
        
        mock_failed_task = Mock()
        mock_failed_task.task_id = "batch_123_task_1"
        mock_failed_task.status.value = "failed"
        mock_failed_task.error = "处理失败"
        mock_failed_task.document_id = "doc2"
        
        mock_scheduler = Mock()
        mock_scheduler.completed_tasks.values.return_value = [mock_completed_task]
        mock_scheduler.failed_tasks.values.return_value = [mock_failed_task]
        mock_scheduler.active_tasks.values.return_value = []
        
        mock_batch_processor.task_scheduler = mock_scheduler
        mock_components = {"batch_processor": mock_batch_processor}
        
        with patch('ai.knowledge_graph.knowledge_extraction.get_components', return_value=mock_components):
            response = client.get("/knowledge/batch/batch_123")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["batch_id"] == "batch_123"
            assert data["status"] == "completed"
            assert data["total_documents"] == 2
            assert data["successful_documents"] == 1
            assert data["failed_documents"] == 1
            assert data["success_rate"] == 0.5
            assert len(data["results"]) == 1
            assert len(data["errors"]) == 1
    
    def test_get_batch_status_not_found(self, mock_batch_processor):
        """测试获取不存在的批处理状态"""
        mock_scheduler = Mock()
        mock_scheduler.completed_tasks.values.return_value = []
        mock_scheduler.failed_tasks.values.return_value = []
        mock_scheduler.active_tasks.values.return_value = []
        
        mock_batch_processor.task_scheduler = mock_scheduler
        mock_components = {"batch_processor": mock_batch_processor}
        
        with patch('ai.knowledge_graph.knowledge_extraction.get_components', return_value=mock_components):
            response = client.get("/knowledge/batch/nonexistent_batch")
            
            assert response.status_code == 404
            assert "未找到批处理任务" in response.json()["detail"]
    
    def test_get_processing_status(self, mock_batch_processor):
        """测试获取整体处理状态"""
        mock_components = {"batch_processor": mock_batch_processor}
        
        with patch('ai.knowledge_graph.knowledge_extraction.get_components', return_value=mock_components):
            response = client.get("/knowledge/processing-status")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["is_running"] is True
            assert data["queue_size"] == 5
            assert data["active_tasks"] == 2
            assert data["worker_count"] == 4
            assert "metrics" in data
            assert "cache_stats" in data
            assert data["memory_usage_mb"] == 512.5


class TestSearchEndpoints:
    """搜索端点测试"""
    
    def test_search_entities(self):
        """测试实体搜索"""
        with patch('ai.knowledge_graph.knowledge_extraction.get_components', return_value={}):
            response = client.post(
                "/knowledge/entities/search?query=张三&entity_type=PERSON&limit=50"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["query"] == "张三"
            assert data["entity_type"] == "PERSON"
            assert data["limit"] == 50
            assert data["results"] == []  # 当前实现返回空结果
            assert data["total"] == 0
    
    def test_search_relations(self):
        """测试关系搜索"""
        with patch('ai.knowledge_graph.knowledge_extraction.get_components', return_value={}):
            response = client.post(
                "/knowledge/relations/search?subject=张三&predicate=WORKS_FOR&object=苹果公司&limit=100"
            )
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["subject"] == "张三"
            assert data["predicate"] == "WORKS_FOR"
            assert data["object"] == "苹果公司"
            assert data["limit"] == 100
            assert data["results"] == []  # 当前实现返回空结果
            assert data["total"] == 0


class TestCacheManagement:
    """缓存管理测试"""
    
    def test_clear_cache(self):
        """测试清空缓存"""
        mock_cache = Mock()
        mock_batch_processor = Mock()
        mock_batch_processor.result_cache = mock_cache
        
        mock_components = {"batch_processor": mock_batch_processor}
        
        with patch('ai.knowledge_graph.knowledge_extraction.get_components', return_value=mock_components):
            response = client.delete("/knowledge/cache")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["message"] == "缓存已清空"
            mock_cache.clear.assert_called_once()
    
    def test_clear_cache_error(self):
        """测试清空缓存失败"""
        mock_cache = Mock()
        mock_cache.clear.side_effect = Exception("清空失败")
        
        mock_batch_processor = Mock()
        mock_batch_processor.result_cache = mock_cache
        
        mock_components = {"batch_processor": mock_batch_processor}
        
        with patch('ai.knowledge_graph.knowledge_extraction.get_components', return_value=mock_components):
            response = client.delete("/knowledge/cache")
            
            assert response.status_code == 500
            assert "清空缓存失败" in response.json()["detail"]


class TestComponentDependency:
    """组件依赖测试"""
    
    def test_get_components_not_initialized(self):
        """测试组件未初始化的情况"""
        with patch('ai.knowledge_graph.knowledge_extraction.entity_recognizer', None):
            response = client.post("/knowledge/extract", json={
                "text": "测试",
                "language": "zh"
            })
            
            assert response.status_code == 503
            assert "知识抽取服务尚未初始化完成" in response.json()["detail"]


class TestRequestValidation:
    """请求验证测试"""
    
    def test_extraction_request_validation(self):
        """测试抽取请求验证"""
        # 测试空文本
        response = client.post("/knowledge/extract", json={
            "text": "",
            "language": "zh"
        })
        assert response.status_code == 422
        
        # 测试无效置信度阈值
        response = client.post("/knowledge/extract", json={
            "text": "测试文本",
            "language": "zh",
            "confidence_threshold": 1.5  # 超过1.0
        })
        assert response.status_code == 422
    
    def test_batch_request_validation(self):
        """测试批处理请求验证"""
        # 测试空文档列表
        response = client.post("/knowledge/batch", json={
            "documents": []
        })
        assert response.status_code == 422
        
        # 测试无效优先级
        response = client.post("/knowledge/batch", json={
            "documents": [{"text": "测试"}],
            "priority": 15  # 超过10
        })
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])