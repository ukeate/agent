"""
多模态处理管道集成测试
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.ai.multimodal import (
    MultimodalProcessor,
    ProcessingPipeline,
    BatchProcessor,
    ContentType,
    MultimodalContent,
    ProcessingOptions,
    ModelPriority,
    ProcessingStatus
)


@pytest.mark.asyncio
class TestProcessingPipeline:
    """处理管道集成测试"""
    
    @pytest.fixture
    async def pipeline(self):
        """创建处理管道"""
        mock_processor = AsyncMock()
        pipeline = ProcessingPipeline(mock_processor, max_concurrent_tasks=3)
        return pipeline
    
    async def test_pipeline_start_stop(self, pipeline):
        """测试管道启动和停止"""
        # 启动管道
        await pipeline.start()
        assert pipeline.is_running is True
        assert len(pipeline._workers) == 3
        
        # 停止管道
        await pipeline.stop()
        assert pipeline.is_running is False
        assert len(pipeline._workers) == 0
    
    async def test_submit_single_task(self, pipeline):
        """测试提交单个任务"""
        content = MultimodalContent(
            content_id="test_001",
            content_type=ContentType.IMAGE,
            file_path="/tmp/test.jpg"
        )
        
        content_id = await pipeline.submit_for_processing(content)
        
        assert content_id == "test_001"
        assert len(pipeline.priority_queue) == 1
        assert pipeline.priority_queue[0].content.content_id == "test_001"
    
    async def test_submit_batch_tasks(self, pipeline):
        """测试批量提交任务"""
        contents = [
            MultimodalContent(
                content_id=f"test_{i:03d}",
                content_type=ContentType.IMAGE,
                file_path=f"/tmp/test_{i}.jpg"
            )
            for i in range(5)
        ]
        
        content_ids = await pipeline.submit_batch(contents)
        
        assert len(content_ids) == 5
        assert len(pipeline.priority_queue) == 5
        assert all(cid.startswith("test_") for cid in content_ids)
    
    async def test_priority_queue_ordering(self, pipeline):
        """测试优先级队列排序"""
        # 提交不同优先级的任务
        content1 = MultimodalContent(content_id="low", content_type=ContentType.IMAGE)
        content2 = MultimodalContent(content_id="high", content_type=ContentType.IMAGE)
        content3 = MultimodalContent(content_id="medium", content_type=ContentType.IMAGE)
        
        await pipeline.submit_for_processing(content1, priority=1)
        await pipeline.submit_for_processing(content2, priority=3)
        await pipeline.submit_for_processing(content3, priority=2)
        
        # 验证队列顺序（高优先级在前）
        assert pipeline.priority_queue[0].content.content_id == "high"
        assert pipeline.priority_queue[1].content.content_id == "medium"
        assert pipeline.priority_queue[2].content.content_id == "low"
    
    async def test_processing_with_mock_processor(self, pipeline):
        """测试使用模拟处理器的处理流程"""
        # 配置模拟处理器
        pipeline.processor.process_content = AsyncMock(return_value=MagicMock(
            content_id="test_001",
            status=ProcessingStatus.COMPLETED,
            extracted_data={"test": "data"},
            confidence_score=0.9,
            processing_time=1.0
        ))
        
        # 启动管道
        await pipeline.start()
        
        # 提交任务
        content = MultimodalContent(
            content_id="test_001",
            content_type=ContentType.IMAGE,
            file_path="/tmp/test.jpg"
        )
        await pipeline.submit_for_processing(content)
        
        # 等待处理完成
        result = await pipeline.wait_for_completion("test_001", timeout=5)
        
        # 验证结果
        assert result is not None
        assert result.status == ProcessingStatus.COMPLETED
        assert result.confidence_score == 0.9
        
        # 停止管道
        await pipeline.stop()
    
    async def test_get_processing_status(self, pipeline):
        """测试获取处理状态"""
        # 添加一个结果
        from src.ai.multimodal.types import ProcessingResult
        test_result = ProcessingResult(
            content_id="test_status",
            status=ProcessingStatus.COMPLETED,
            extracted_data={"status": "test"},
            confidence_score=0.85,
            processing_time=2.0
        )
        pipeline.processing_results["test_status"] = test_result
        
        # 获取状态
        status = await pipeline.get_processing_status("test_status")
        
        assert status is not None
        assert status.content_id == "test_status"
        assert status.status == ProcessingStatus.COMPLETED
        
        # 获取不存在的状态
        none_status = await pipeline.get_processing_status("nonexistent")
        assert none_status is None
    
    async def test_queue_status(self, pipeline):
        """测试队列状态"""
        # 添加一些任务和结果
        for i in range(3):
            content = MultimodalContent(
                content_id=f"queued_{i}",
                content_type=ContentType.IMAGE
            )
            await pipeline.submit_for_processing(content)
        
        # 添加完成的结果
        from src.ai.multimodal.types import ProcessingResult
        for i in range(2):
            pipeline.processing_results[f"completed_{i}"] = ProcessingResult(
                content_id=f"completed_{i}",
                status=ProcessingStatus.COMPLETED,
                extracted_data={},
                confidence_score=0.9,
                processing_time=1.0
            )
        
        # 添加失败的结果
        pipeline.processing_results["failed_1"] = ProcessingResult(
            content_id="failed_1",
            status=ProcessingStatus.FAILED,
            extracted_data={},
            confidence_score=0.0,
            processing_time=0.5,
            error_message="测试错误"
        )
        
        # 获取队列状态
        status = pipeline.get_queue_status()
        
        assert status["queued_tasks"] == 3
        assert status["completed_tasks"] == 2
        assert status["failed_tasks"] == 1
    
    async def test_error_retry_mechanism(self, pipeline):
        """测试错误重试机制"""
        # 配置处理器先失败后成功
        call_count = 0
        
        async def mock_process(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("第一次失败")
            return MagicMock(
                content_id="retry_test",
                status=ProcessingStatus.COMPLETED,
                extracted_data={},
                confidence_score=0.8,
                processing_time=1.0
            )
        
        pipeline.processor.process_content = mock_process
        
        # 启动管道
        await pipeline.start()
        
        # 提交任务
        content = MultimodalContent(
            content_id="retry_test",
            content_type=ContentType.IMAGE
        )
        await pipeline.submit_for_processing(content)
        
        # 等待处理
        await asyncio.sleep(2)
        
        # 验证重试
        assert call_count >= 1  # 至少调用了一次
        
        # 停止管道
        await pipeline.stop()


@pytest.mark.asyncio
class TestBatchProcessor:
    """批量处理器测试"""
    
    async def test_batch_processing_wait(self):
        """测试批量处理等待完成"""
        # 创建模拟管道
        mock_pipeline = AsyncMock()
        mock_pipeline.submit_batch = AsyncMock(return_value=[
            "batch_1", "batch_2", "batch_3"
        ])
        
        # 模拟等待完成
        from src.ai.multimodal.types import ProcessingResult
        mock_results = [
            ProcessingResult(
                content_id=f"batch_{i}",
                status=ProcessingStatus.COMPLETED,
                extracted_data={},
                confidence_score=0.9,
                processing_time=1.0
            )
            for i in range(1, 4)
        ]
        
        mock_pipeline.wait_for_completion = AsyncMock(side_effect=mock_results)
        
        # 创建批处理器
        batch_processor = BatchProcessor(mock_pipeline)
        
        # 创建测试内容
        contents = [
            MultimodalContent(
                content_id=f"batch_{i}",
                content_type=ContentType.DOCUMENT
            )
            for i in range(1, 4)
        ]
        
        # 执行批处理
        results = await batch_processor.process_batch(
            contents,
            wait_for_completion=True,
            timeout=10
        )
        
        # 验证结果
        assert len(results) == 3
        assert all(r.status == ProcessingStatus.COMPLETED for r in results)
    
    async def test_batch_processing_no_wait(self):
        """测试批量处理不等待完成"""
        mock_pipeline = AsyncMock()
        mock_pipeline.submit_batch = AsyncMock(return_value=["b1", "b2"])
        
        batch_processor = BatchProcessor(mock_pipeline)
        
        contents = [
            MultimodalContent(content_id=f"b{i}", content_type=ContentType.IMAGE)
            for i in range(1, 3)
        ]
        
        results = await batch_processor.process_batch(
            contents,
            wait_for_completion=False
        )
        
        assert len(results) == 0  # 不等待，返回空结果
        mock_pipeline.submit_batch.assert_called_once()
    
    async def test_batch_processing_timeout(self):
        """测试批量处理超时"""
        mock_pipeline = AsyncMock()
        mock_pipeline.submit_batch = AsyncMock(return_value=["timeout_1"])
        mock_pipeline.wait_for_completion = AsyncMock(return_value=None)  # 模拟超时
        
        batch_processor = BatchProcessor(mock_pipeline)
        
        contents = [
            MultimodalContent(content_id="timeout_1", content_type=ContentType.VIDEO)
        ]
        
        results = await batch_processor.process_batch(
            contents,
            wait_for_completion=True,
            timeout=0.1  # 短超时
        )
        
        assert len(results) == 1
        assert results[0].status == ProcessingStatus.FAILED
        assert results[0].error_message == "处理超时"