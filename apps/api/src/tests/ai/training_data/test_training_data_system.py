"""
训练数据管理系统综合测试

测试覆盖：
- 数据收集功能
- 数据预处理管道
- 标注系统
- 版本管理
- API端点
- 后台任务
"""

import pytest
import asyncio
import json
import tempfile
from datetime import datetime
from src.core.utils.timezone_utils import utc_now
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from ....main import app
from ....ai.training_data.models import Base, SourceType, DataStatus, AnnotationTaskType, AnnotationStatus
from ....ai.training_data.core import DataSource, DataRecord, AnnotationTask, Annotation, DataFilter, ExportFormat
from ....ai.training_data.collectors import CollectorFactory, APIDataCollector, FileDataCollector, CollectionStats
from ....ai.training_data.preprocessing import DataPreprocessor
from ....ai.training_data.annotation import AnnotationManager, QualityController
from ....ai.training_data.version_manager import DataVersionManager
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

TEST_DATABASE_URL = "sqlite+aiosqlite:///./test_training_data.db"

@pytest.fixture
async def async_engine():
    """创建测试数据库引擎"""
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()

@pytest.fixture
async def async_session(async_engine):
    """创建测试数据库会话"""
    async_session_factory = async_sessionmaker(
        async_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session_factory() as session:
        yield session

@pytest.fixture
def client():
    """创建测试客户端"""
    return TestClient(app)

@pytest.fixture
def sample_data_source():
    """示例数据源"""
    return DataSource(
        source_id="test_source_001",
        source_type=SourceType.API,
        name="Test API Source",
        description="Test API data source",
        config={
            "url": "https://api.example.com/data",
            "headers": {"Authorization": "Bearer test_token"},
            "method": "GET",
            "params": {"limit": 100}
        }
    )

@pytest.fixture
def sample_records():
    """示例数据记录"""
    return [
        DataRecord(
            record_id="rec_001",
            source_id="test_source_001",
            raw_data={"text": "This is a test document", "category": "test"},
            metadata={"source": "api"},
            quality_score=0.8,
            status=DataStatus.RAW
        ),
        DataRecord(
            record_id="rec_002", 
            source_id="test_source_001",
            raw_data={"text": "Another test document", "category": "sample"},
            metadata={"source": "api"},
            quality_score=0.9,
            status=DataStatus.RAW
        )
    ]

class TestDataCollectors:
    """数据收集器测试"""
    
    def test_collector_factory(self, sample_data_source):
        """测试收集器工厂"""
        collector = CollectorFactory.create_collector(sample_data_source)
        assert isinstance(collector, APIDataCollector)
        assert collector.source.source_id == sample_data_source.source_id
    
    @pytest.mark.asyncio
    async def test_api_collector_mock(self, sample_data_source):
        """测试API收集器（模拟）"""
        with patch('aiohttp.ClientSession.get') as mock_get:
            # 模拟API响应
            mock_response = Mock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "data": [
                    {"id": 1, "text": "Sample text 1"},
                    {"id": 2, "text": "Sample text 2"}
                ]
            })
            mock_get.return_value.__aenter__.return_value = mock_response
            
            collector = APIDataCollector(sample_data_source)
            records = []
            
            async for record in collector.collect_data():
                records.append(record)
                if len(records) >= 2:  # 限制测试数据量
                    break
            
            assert len(records) == 2
            assert records[0].source_id == sample_data_source.source_id
            assert "text" in records[0].raw_data
    
    @pytest.mark.asyncio
    async def test_file_collector(self):
        """测试文件收集器"""
        # 创建临时JSON文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [
                {"id": 1, "content": "Test content 1"},
                {"id": 2, "content": "Test content 2"}
            ]
            json.dump(test_data, f)
            temp_file_path = f.name
        
        try:
            # 创建文件数据源
            file_source = DataSource(
                source_id="file_test",
                source_type=SourceType.FILE,
                name="Test File Source",
                description="Test file data source",
                config={
                    "file_path": temp_file_path,
                    "file_type": "json"
                }
            )
            
            collector = FileDataCollector(file_source)
            records = []
            
            async for record in collector.collect_data():
                records.append(record)
            
            assert len(records) == 2
            assert records[0].raw_data["id"] == 1
            assert records[1].raw_data["content"] == "Test content 2"
        
        finally:
            Path(temp_file_path).unlink()

class TestDataPreprocessing:
    """数据预处理测试"""
    
    @pytest.mark.asyncio
    async def test_preprocessing_pipeline(self, async_session, sample_records):
        """测试预处理管道"""
        preprocessor = DataPreprocessor(async_session)
        
        # 测试文本清理
        processed_records = await preprocessor.preprocess_records(
            sample_records, ["text_cleaning"]
        )
        
        assert len(processed_records) == 2
        for record in processed_records:
            assert record.processed_data is not None
            assert "cleaned_text" in record.processed_data
    
    @pytest.mark.asyncio
    async def test_quality_assessment(self, async_session, sample_records):
        """测试质量评估"""
        preprocessor = DataPreprocessor(async_session)
        
        processed_records = await preprocessor.preprocess_records(
            sample_records, ["quality_filtering"]
        )
        
        for record in processed_records:
            assert record.quality_score is not None
            assert 0.0 <= record.quality_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_deduplication(self, async_session):
        """测试去重功能"""
        # 创建重复记录
        duplicate_records = [
            DataRecord(
                record_id="dup_001",
                source_id="test_source",
                raw_data={"text": "Duplicate content"},
                status=DataStatus.RAW
            ),
            DataRecord(
                record_id="dup_002", 
                source_id="test_source",
                raw_data={"text": "Duplicate content"},  # 相同内容
                status=DataStatus.RAW
            ),
            DataRecord(
                record_id="dup_003",
                source_id="test_source", 
                raw_data={"text": "Unique content"},
                status=DataStatus.RAW
            )
        ]
        
        preprocessor = DataPreprocessor(async_session)
        processed_records = await preprocessor.preprocess_records(
            duplicate_records, ["deduplication"]
        )
        
        # 应该去除一个重复记录
        assert len(processed_records) == 2

class TestAnnotationSystem:
    """标注系统测试"""
    
    @pytest.mark.asyncio
    async def test_annotation_manager_create_task(self, async_session):
        """测试创建标注任务"""
        manager = AnnotationManager(async_session)
        
        task = AnnotationTask(
            task_id="test_task_001",
            name="Test Annotation Task",
            description="Test task description",
            task_type=AnnotationTaskType.TEXT_CLASSIFICATION,
            data_records=["rec_001", "rec_002"],
            annotation_schema={"labels": ["positive", "negative"]},
            guidelines="Please classify the sentiment",
            assignees=["annotator_001"],
            created_by="test_user"
        )
        
        task_id = await manager.create_task(task)
        assert task_id == "test_task_001"
    
    @pytest.mark.asyncio
    async def test_annotation_submission(self, async_session):
        """测试提交标注结果"""
        manager = AnnotationManager(async_session)
        
        # 先创建任务
        task = AnnotationTask(
            task_id="test_task_002",
            name="Test Task",
            description="Test",
            task_type=AnnotationTaskType.TEXT_CLASSIFICATION,
            data_records=["rec_001"],
            annotation_schema={"labels": ["A", "B"]},
            guidelines="Test guidelines",
            assignees=["annotator_001"],
            created_by="test_user"
        )
        await manager.create_task(task)
        
        # 提交标注
        annotation = Annotation(
            annotation_id="ann_001",
            task_id="test_task_002",
            record_id="rec_001",
            annotator_id="annotator_001",
            annotation_data={"label": "A", "confidence": 0.9},
            confidence=0.9,
            time_spent=120
        )
        
        annotation_id = await manager.submit_annotation(annotation)
        assert annotation_id == "ann_001"
    
    @pytest.mark.asyncio
    async def test_quality_controller(self, async_session):
        """测试质量控制器"""
        controller = QualityController(async_session)
        
        # 创建测试任务和标注数据
        manager = AnnotationManager(async_session)
        task = AnnotationTask(
            task_id="quality_test_task",
            name="Quality Test",
            description="Test quality assessment",
            task_type=AnnotationTaskType.TEXT_CLASSIFICATION,
            data_records=["rec_001", "rec_002"],
            annotation_schema={"labels": ["positive", "negative"]},
            guidelines="Classify sentiment",
            assignees=["ann_001", "ann_002"],
            created_by="test_user"
        )
        await manager.create_task(task)
        
        # 添加标注结果
        annotations = [
            Annotation(
                annotation_id="ann_001_rec_001",
                task_id="quality_test_task",
                record_id="rec_001",
                annotator_id="ann_001",
                annotation_data={"label": "positive"},
                confidence=0.8
            ),
            Annotation(
                annotation_id="ann_002_rec_001",
                task_id="quality_test_task", 
                record_id="rec_001",
                annotator_id="ann_002",
                annotation_data={"label": "positive"},  # 一致
                confidence=0.9
            )
        ]
        
        for annotation in annotations:
            await manager.submit_annotation(annotation)
        
        # 生成质量报告
        report = await controller.generate_quality_report("quality_test_task")
        
        assert report.task_id == "quality_test_task"
        assert 0.0 <= report.overall_score <= 1.0
        assert "percentage_agreement" in report.agreement_metrics

class TestVersionManagement:
    """版本管理测试"""
    
    @pytest.mark.asyncio 
    async def test_version_creation(self, async_session, sample_records):
        """测试版本创建"""
        # 先保存一些数据记录到数据库
        from ....ai.training_data.models import DataRecordModel
        
        for record in sample_records:
            record_model = DataRecordModel(
                record_id=record.record_id,
                source_id=record.source_id,
                raw_data=record.raw_data,
                processed_data=record.processed_data,
                metadata=record.metadata,
                quality_score=record.quality_score,
                status=record.status
            )
            async_session.add(record_model)
        await async_session.commit()
        
        # 创建版本管理器
        with tempfile.TemporaryDirectory() as temp_dir:
            version_manager = DataVersionManager(async_session, temp_dir)
            
            # 创建版本
            version_id = await version_manager.create_version(
                dataset_name="test_dataset",
                version_number="v1.0",
                description="Test version",
                created_by="test_user"
            )
            
            assert version_id.startswith("test_dataset_v1.0")
    
    @pytest.mark.asyncio
    async def test_version_comparison(self, async_session, sample_records):
        """测试版本比较"""
        from ....ai.training_data.models import DataRecordModel
        
        # 保存数据记录
        for record in sample_records:
            record_model = DataRecordModel(
                record_id=record.record_id,
                source_id=record.source_id,
                raw_data=record.raw_data,
                processed_data=record.processed_data,
                metadata=record.metadata,
                quality_score=record.quality_score,
                status=record.status
            )
            async_session.add(record_model)
        await async_session.commit()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            version_manager = DataVersionManager(async_session, temp_dir)
            
            # 创建两个版本
            version1_id = await version_manager.create_version(
                dataset_name="test_dataset",
                version_number="v1.0", 
                description="First version",
                created_by="test_user"
            )
            
            # 添加一条新记录
            new_record = DataRecord(
                record_id="rec_003",
                source_id="test_source_001",
                raw_data={"text": "New document", "category": "new"},
                metadata={"source": "api"},
                quality_score=0.7,
                status=DataStatus.RAW
            )
            
            record_model = DataRecordModel(
                record_id=new_record.record_id,
                source_id=new_record.source_id,
                raw_data=new_record.raw_data,
                processed_data=new_record.processed_data,
                metadata=new_record.metadata,
                quality_score=new_record.quality_score,
                status=new_record.status
            )
            async_session.add(record_model)
            await async_session.commit()
            
            version2_id = await version_manager.create_version(
                dataset_name="test_dataset",
                version_number="v2.0",
                description="Second version",
                created_by="test_user"
            )
            
            # 比较版本
            comparison = await version_manager.compare_versions(version1_id, version2_id)
            
            assert comparison.version1_id == version1_id
            assert comparison.version2_id == version2_id
            assert comparison.summary["added_count"] == 1
    
    @pytest.mark.asyncio
    async def test_version_export(self, async_session, sample_records):
        """测试版本导出"""
        from ....ai.training_data.models import DataRecordModel
        
        # 保存数据记录
        for record in sample_records:
            record_model = DataRecordModel(
                record_id=record.record_id,
                source_id=record.source_id,
                raw_data=record.raw_data,
                processed_data=record.processed_data,
                metadata=record.metadata,
                quality_score=record.quality_score,
                status=record.status
            )
            async_session.add(record_model)
        await async_session.commit()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            version_manager = DataVersionManager(async_session, temp_dir)
            
            # 创建版本
            version_id = await version_manager.create_version(
                dataset_name="test_dataset",
                version_number="v1.0",
                description="Export test version",
                created_by="test_user"
            )
            
            # 导出为JSON
            output_path = await version_manager.export_version(
                version_id=version_id,
                export_format=ExportFormat.JSON,
                output_path=f"{temp_dir}/export_test.json"
            )
            
            # 验证导出文件
            assert Path(output_path).exists()
            
            with open(output_path, 'r') as f:
                exported_data = json.load(f)
                assert len(exported_data) == 2
                assert exported_data[0]['record_id'] in ['rec_001', 'rec_002']

class TestAPIEndpoints:
    """API端点测试"""
    
    def test_create_data_source(self, client):
        """测试创建数据源API"""
        source_data = {
            "source_type": "api",
            "name": "Test API Source",
            "description": "Test description",
            "config": {
                "url": "https://api.example.com/data",
                "method": "GET"
            }
        }
        
        response = client.post("/training-data/sources", json=source_data)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Test API Source"
        assert data["source_type"] == "api"
    
    def test_list_data_sources(self, client):
        """测试获取数据源列表API"""
        response = client.get("/training-data/sources")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_health_check(self, client):
        """测试健康检查端点"""
        response = client.get("/training-data/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_stats_overview(self, client):
        """测试统计概览端点"""
        response = client.get("/training-data/stats/overview")
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        assert "records" in data
        assert "tasks" in data
        assert "annotations" in data

class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, async_session):
        """测试完整工作流程"""
        # 1. 创建数据源
        data_source = DataSource(
            source_id="integration_test_source",
            source_type=SourceType.FILE,
            name="Integration Test Source",
            description="Full workflow test",
            config={"file_path": "test.json", "file_type": "json"}
        )
        
        # 2. 模拟数据收集（直接创建记录）
        records = [
            DataRecord(
                record_id="int_rec_001",
                source_id="integration_test_source",
                raw_data={"text": "Integration test document 1"},
                status=DataStatus.RAW
            ),
            DataRecord(
                record_id="int_rec_002", 
                source_id="integration_test_source",
                raw_data={"text": "Integration test document 2"},
                status=DataStatus.RAW
            )
        ]
        
        # 3. 数据预处理
        preprocessor = DataPreprocessor(async_session)
        processed_records = await preprocessor.preprocess_records(
            records, ["text_cleaning", "quality_filtering"]
        )
        
        # 4. 创建标注任务
        annotation_manager = AnnotationManager(async_session)
        task = AnnotationTask(
            task_id="integration_task",
            name="Integration Test Task",
            description="Full workflow annotation task",
            task_type=AnnotationTaskType.TEXT_CLASSIFICATION,
            data_records=["int_rec_001", "int_rec_002"],
            annotation_schema={"labels": ["category_a", "category_b"]},
            guidelines="Classify the documents",
            assignees=["test_annotator"],
            created_by="integration_test"
        )
        
        task_id = await annotation_manager.create_task(task)
        
        # 5. 提交标注
        annotations = [
            Annotation(
                annotation_id="int_ann_001",
                task_id=task_id,
                record_id="int_rec_001",
                annotator_id="test_annotator",
                annotation_data={"label": "category_a"},
                confidence=0.85
            ),
            Annotation(
                annotation_id="int_ann_002",
                task_id=task_id,
                record_id="int_rec_002", 
                annotator_id="test_annotator",
                annotation_data={"label": "category_b"},
                confidence=0.92
            )
        ]
        
        for annotation in annotations:
            await annotation_manager.submit_annotation(annotation)
        
        # 6. 创建数据版本
        with tempfile.TemporaryDirectory() as temp_dir:
            version_manager = DataVersionManager(async_session, temp_dir)
            
            # 保存处理后的记录到数据库
            from ....ai.training_data.models import DataRecordModel
            for record in processed_records:
                record_model = DataRecordModel(
                    record_id=record.record_id,
                    source_id=record.source_id,
                    raw_data=record.raw_data,
                    processed_data=record.processed_data,
                    metadata=record.metadata,
                    quality_score=record.quality_score,
                    status=record.status,
                    processed_at=record.processed_at
                )
                async_session.add(record_model)
            await async_session.commit()
            
            version_id = await version_manager.create_version(
                dataset_name="integration_test_dataset",
                version_number="v1.0",
                description="Integration test version",
                created_by="integration_test"
            )
            
            # 7. 验证结果
            assert task_id == "integration_task"
            assert version_id.startswith("integration_test_dataset_v1.0")
            
            # 8. 检查任务进度
            progress = await annotation_manager.get_task_progress(task_id)
            assert progress.total_records == 2
            assert progress.annotated_records == 2
            assert progress.progress_percentage == 100.0
        
        logger.info("Integration test completed successfully!")

# 性能测试
class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_large_dataset_processing(self, async_session):
        """测试大数据集处理性能"""
        # 创建大量测试记录
        large_records = []
        for i in range(100):  # 测试环境使用较小数据集
            record = DataRecord(
                record_id=f"perf_rec_{i:03d}",
                source_id="performance_test_source",
                raw_data={"text": f"Performance test document {i}", "index": i},
                status=DataStatus.RAW
            )
            large_records.append(record)
        
        # 测试批量预处理性能
        import time
        start_time = time.time()
        
        preprocessor = DataPreprocessor(async_session)
        processed_records = await preprocessor.preprocess_records(
            large_records, ["text_cleaning"]
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert len(processed_records) == 100
        assert processing_time < 30.0  # 应在30秒内完成
        
        # 计算处理速度
        records_per_second = len(processed_records) / processing_time
        logger.info(f"Processing speed: {records_per_second:.2f} records/second")
        
        # 基本性能要求
        assert records_per_second > 1.0  # 至少每秒处理1条记录

if __name__ == "__main__":
    setup_logging()
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
