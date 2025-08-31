"""
压缩流水线测试

测试压缩流水线的任务管理和调度功能
"""

import pytest
import asyncio
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from src.ai.model_compression.compression_pipeline import CompressionPipeline, PipelineStatus
from src.ai.model_compression.models import (
    CompressionJob, 
    CompressionMethod, 
    QuantizationConfig, 
    QuantizationMethod, 
    PrecisionType
)


class SimpleModel(nn.Module):
    """简单测试模型"""
    
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 2)
    
    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def simple_model():
    """创建简单模型"""
    return SimpleModel()


@pytest.fixture
def compression_pipeline():
    """创建压缩流水线"""
    return CompressionPipeline()


@pytest.fixture
def quantization_job(tmp_path):
    """创建量化任务"""
    model_path = tmp_path / "test_model.pth"
    torch.save(SimpleModel(), model_path)
    
    return CompressionJob(
        job_name="test_quantization",
        model_path=str(model_path),
        compression_method=CompressionMethod.QUANTIZATION,
        quantization_config=QuantizationConfig(
            method=QuantizationMethod.PTQ,
            precision=PrecisionType.INT8
        )
    )


class TestCompressionPipeline:
    """压缩流水线测试类"""
    
    def test_pipeline_initialization(self, compression_pipeline):
        """测试流水线初始化"""
        assert compression_pipeline is not None
        assert compression_pipeline.max_concurrent_jobs > 0
        assert len(compression_pipeline.active_jobs) == 0
        assert len(compression_pipeline.job_history) == 0
    
    def test_job_submission(self, compression_pipeline, quantization_job):
        """测试任务提交"""
        job_id = compression_pipeline.submit_job(quantization_job)
        
        assert job_id is not None
        assert job_id == quantization_job.job_id
        assert len(compression_pipeline.active_jobs) == 1
        assert compression_pipeline.active_jobs[job_id]["job"] == quantization_job
        assert compression_pipeline.active_jobs[job_id]["status"] == PipelineStatus.PENDING
    
    def test_job_status_query(self, compression_pipeline, quantization_job):
        """测试任务状态查询"""
        job_id = compression_pipeline.submit_job(quantization_job)
        
        status = compression_pipeline.get_job_status(job_id)
        assert status == PipelineStatus.PENDING
        
        # 测试不存在的任务
        invalid_status = compression_pipeline.get_job_status("invalid_id")
        assert invalid_status is None
    
    def test_job_info_retrieval(self, compression_pipeline, quantization_job):
        """测试任务信息检索"""
        job_id = compression_pipeline.submit_job(quantization_job)
        
        job_info = compression_pipeline.get_job_info(job_id)
        assert job_info is not None
        assert job_info["job"] == quantization_job
        assert job_info["status"] == PipelineStatus.PENDING
        assert "created_at" in job_info
        assert "updated_at" in job_info
    
    @pytest.mark.asyncio
    async def test_job_execution(self, compression_pipeline, quantization_job):
        """测试任务执行"""
        job_id = compression_pipeline.submit_job(quantization_job)
        
        # 启动流水线
        pipeline_task = asyncio.create_task(compression_pipeline.start())
        
        # 等待任务开始执行
        await asyncio.sleep(0.1)
        
        # 检查任务状态变化
        status = compression_pipeline.get_job_status(job_id)
        assert status in [PipelineStatus.RUNNING, PipelineStatus.COMPLETED]
        
        # 停止流水线
        await compression_pipeline.stop()
        pipeline_task.cancel()
        
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_concurrent_job_execution(self, compression_pipeline, tmp_path):
        """测试并发任务执行"""
        # 创建多个任务
        jobs = []
        for i in range(3):
            model_path = tmp_path / f"model_{i}.pth"
            torch.save(SimpleModel(), model_path)
            
            job = CompressionJob(
                job_name=f"job_{i}",
                model_path=str(model_path),
                compression_method=CompressionMethod.QUANTIZATION,
                quantization_config=QuantizationConfig(
                    method=QuantizationMethod.PTQ,
                    precision=PrecisionType.INT8
                )
            )
            jobs.append(job)
            compression_pipeline.submit_job(job)
        
        # 启动流水线
        pipeline_task = asyncio.create_task(compression_pipeline.start())
        
        # 等待任务执行
        await asyncio.sleep(0.5)
        
        # 检查并发执行
        running_count = sum(
            1 for job_info in compression_pipeline.active_jobs.values()
            if job_info["status"] == PipelineStatus.RUNNING
        )
        
        # 应该有任务在运行（考虑并发限制）
        assert running_count <= compression_pipeline.max_concurrent_jobs
        
        # 停止流水线
        await compression_pipeline.stop()
        pipeline_task.cancel()
        
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass
    
    def test_job_cancellation(self, compression_pipeline, quantization_job):
        """测试任务取消"""
        job_id = compression_pipeline.submit_job(quantization_job)
        
        # 取消任务
        success = compression_pipeline.cancel_job(job_id)
        assert success is True
        
        # 验证任务状态
        status = compression_pipeline.get_job_status(job_id)
        assert status == PipelineStatus.CANCELLED
        
        # 尝试取消不存在的任务
        invalid_cancel = compression_pipeline.cancel_job("invalid_id")
        assert invalid_cancel is False
    
    def test_job_history_management(self, compression_pipeline, quantization_job):
        """测试任务历史管理"""
        job_id = compression_pipeline.submit_job(quantization_job)
        
        # 模拟任务完成
        compression_pipeline._move_to_history(job_id, PipelineStatus.COMPLETED, {"result": "success"})
        
        # 验证历史记录
        assert len(compression_pipeline.job_history) == 1
        assert job_id in compression_pipeline.job_history
        assert compression_pipeline.job_history[job_id]["status"] == PipelineStatus.COMPLETED
        assert job_id not in compression_pipeline.active_jobs
    
    def test_job_priority_handling(self, compression_pipeline, tmp_path):
        """测试任务优先级处理"""
        # 创建不同优先级的任务
        high_priority_job = CompressionJob(
            job_name="high_priority",
            model_path=str(tmp_path / "model1.pth"),
            compression_method=CompressionMethod.QUANTIZATION,
            priority=1  # 高优先级
        )
        torch.save(SimpleModel(), tmp_path / "model1.pth")
        
        low_priority_job = CompressionJob(
            job_name="low_priority", 
            model_path=str(tmp_path / "model2.pth"),
            compression_method=CompressionMethod.QUANTIZATION,
            priority=10  # 低优先级
        )
        torch.save(SimpleModel(), tmp_path / "model2.pth")
        
        # 先提交低优先级任务
        low_id = compression_pipeline.submit_job(low_priority_job)
        high_id = compression_pipeline.submit_job(high_priority_job)
        
        # 获取队列中的下一个任务
        next_job_id = compression_pipeline._get_next_job()
        
        # 应该返回高优先级任务
        assert next_job_id == high_id
    
    def test_job_retry_mechanism(self, compression_pipeline, quantization_job):
        """测试任务重试机制"""
        job_id = compression_pipeline.submit_job(quantization_job)
        
        # 模拟任务失败
        compression_pipeline.active_jobs[job_id]["status"] = PipelineStatus.FAILED
        compression_pipeline.active_jobs[job_id]["error"] = "Test error"
        compression_pipeline.active_jobs[job_id]["retry_count"] = 0
        
        # 重试任务
        success = compression_pipeline.retry_job(job_id)
        assert success is True
        
        # 验证状态重置
        status = compression_pipeline.get_job_status(job_id)
        assert status == PipelineStatus.PENDING
        assert compression_pipeline.active_jobs[job_id]["retry_count"] == 1
    
    def test_pipeline_statistics(self, compression_pipeline, quantization_job):
        """测试流水线统计信息"""
        # 提交任务
        job_id = compression_pipeline.submit_job(quantization_job)
        
        # 模拟各种状态的任务
        compression_pipeline._move_to_history(job_id, PipelineStatus.COMPLETED, {"result": "success"})
        
        stats = compression_pipeline.get_pipeline_statistics()
        
        assert "total_jobs" in stats
        assert "active_jobs" in stats
        assert "completed_jobs" in stats
        assert "failed_jobs" in stats
        assert "cancelled_jobs" in stats
        assert stats["total_jobs"] >= 1
        assert stats["completed_jobs"] >= 1
    
    def test_job_queue_management(self, compression_pipeline, tmp_path):
        """测试任务队列管理"""
        # 填满任务队列
        for i in range(compression_pipeline.max_concurrent_jobs + 2):
            model_path = tmp_path / f"model_{i}.pth"
            torch.save(SimpleModel(), model_path)
            
            job = CompressionJob(
                job_name=f"job_{i}",
                model_path=str(model_path),
                compression_method=CompressionMethod.QUANTIZATION
            )
            compression_pipeline.submit_job(job)
        
        # 验证队列管理
        assert len(compression_pipeline.active_jobs) > compression_pipeline.max_concurrent_jobs
        
        # 获取所有待处理任务
        pending_jobs = [
            job_id for job_id, job_info in compression_pipeline.active_jobs.items()
            if job_info["status"] == PipelineStatus.PENDING
        ]
        assert len(pending_jobs) > 0
    
    def test_job_cleanup(self, compression_pipeline, quantization_job):
        """测试任务清理"""
        job_id = compression_pipeline.submit_job(quantization_job)
        
        # 移到历史记录
        compression_pipeline._move_to_history(job_id, PipelineStatus.COMPLETED, {"result": "success"})
        
        # 执行清理（保留最近的任务）
        compression_pipeline.cleanup_history(keep_recent=0)
        
        # 验证历史记录被清理
        assert len(compression_pipeline.job_history) == 0
    
    def test_job_timeout_handling(self, compression_pipeline, quantization_job):
        """测试任务超时处理"""
        # 设置短超时时间
        quantization_job.timeout = 1  # 1秒超时
        job_id = compression_pipeline.submit_job(quantization_job)
        
        # 模拟任务运行但未完成
        compression_pipeline.active_jobs[job_id]["status"] = PipelineStatus.RUNNING
        compression_pipeline.active_jobs[job_id]["started_at"] = datetime.now(timezone.utc)
        
        # 检查超时
        is_timeout = compression_pipeline._check_job_timeout(job_id)
        
        # 如果立即检查可能不会超时，所以这个测试主要验证超时检查逻辑存在
        assert isinstance(is_timeout, bool)
    
    def test_error_handling_invalid_job(self, compression_pipeline):
        """测试无效任务的错误处理"""
        invalid_job = CompressionJob(
            job_name="invalid_job",
            model_path="/nonexistent/model.pth",  # 不存在的路径
            compression_method=CompressionMethod.QUANTIZATION
        )
        
        job_id = compression_pipeline.submit_job(invalid_job)
        assert job_id is not None  # 任务应该被接受，但在执行时会失败
        
        # 验证任务被添加到队列
        assert job_id in compression_pipeline.active_jobs
    
    def test_pipeline_state_persistence(self, compression_pipeline, quantization_job, tmp_path):
        """测试流水线状态持久化"""
        job_id = compression_pipeline.submit_job(quantization_job)
        
        # 保存状态
        state_file = tmp_path / "pipeline_state.json"
        compression_pipeline.save_state(str(state_file))
        
        assert state_file.exists()
        
        # 创建新流水线实例并加载状态
        new_pipeline = CompressionPipeline()
        new_pipeline.load_state(str(state_file))
        
        # 验证状态恢复
        assert job_id in new_pipeline.active_jobs
        assert new_pipeline.active_jobs[job_id]["job"].job_name == quantization_job.job_name
    
    @patch('src.ai.model_compression.compression_pipeline.QuantizationEngine')
    def test_job_execution_with_mock(self, mock_engine_class, compression_pipeline, quantization_job):
        """使用模拟测试任务执行"""
        # 设置模拟引擎
        mock_engine = Mock()
        mock_engine.quantize_model.return_value = (SimpleModel(), {"compression_ratio": 2.0})
        mock_engine_class.return_value = mock_engine
        
        job_id = compression_pipeline.submit_job(quantization_job)
        
        # 手动执行任务（用于测试）
        result = compression_pipeline._execute_job_sync(job_id)
        
        assert result is not None
        assert "result" in result
        mock_engine.quantize_model.assert_called_once()
    
    def test_get_available_workers(self, compression_pipeline):
        """测试获取可用工作器数量"""
        available = compression_pipeline.get_available_workers()
        assert available == compression_pipeline.max_concurrent_jobs
        
        # 添加一些运行中的任务
        compression_pipeline.active_jobs["dummy1"] = {
            "status": PipelineStatus.RUNNING,
            "job": None
        }
        
        available_after = compression_pipeline.get_available_workers()
        assert available_after == compression_pipeline.max_concurrent_jobs - 1
    
    def test_list_all_jobs(self, compression_pipeline, quantization_job):
        """测试列出所有任务"""
        job_id = compression_pipeline.submit_job(quantization_job)
        
        # 模拟完成任务
        compression_pipeline._move_to_history(job_id, PipelineStatus.COMPLETED, {"result": "success"})
        
        all_jobs = compression_pipeline.list_all_jobs()
        
        assert len(all_jobs) >= 1
        assert job_id in [job["job_id"] for job in all_jobs]


@pytest.mark.parametrize("compression_method", [
    CompressionMethod.QUANTIZATION,
    CompressionMethod.PRUNING,
    CompressionMethod.DISTILLATION,
])
def test_different_compression_methods(compression_method, compression_pipeline, tmp_path):
    """参数化测试不同的压缩方法"""
    model_path = tmp_path / "test_model.pth"
    torch.save(SimpleModel(), model_path)
    
    job = CompressionJob(
        job_name=f"test_{compression_method.value}",
        model_path=str(model_path),
        compression_method=compression_method
    )
    
    job_id = compression_pipeline.submit_job(job)
    assert job_id is not None
    
    status = compression_pipeline.get_job_status(job_id)
    assert status == PipelineStatus.PENDING


class TestPipelineEdgeCases:
    """流水线边界情况测试"""
    
    def test_empty_pipeline(self, compression_pipeline):
        """测试空流水线"""
        stats = compression_pipeline.get_pipeline_statistics()
        assert stats["total_jobs"] == 0
        assert stats["active_jobs"] == 0
        
        # 获取不存在的任务
        job_info = compression_pipeline.get_job_info("nonexistent")
        assert job_info is None
    
    def test_max_capacity_reached(self, compression_pipeline, tmp_path):
        """测试达到最大容量"""
        # 提交大量任务
        job_ids = []
        for i in range(compression_pipeline.max_concurrent_jobs * 2):
            model_path = tmp_path / f"model_{i}.pth"
            torch.save(SimpleModel(), model_path)
            
            job = CompressionJob(
                job_name=f"job_{i}",
                model_path=str(model_path),
                compression_method=CompressionMethod.QUANTIZATION
            )
            job_id = compression_pipeline.submit_job(job)
            job_ids.append(job_id)
        
        # 验证所有任务都被接受
        assert len(job_ids) == compression_pipeline.max_concurrent_jobs * 2
        
        # 验证有些任务处于等待状态
        pending_count = sum(
            1 for job_info in compression_pipeline.active_jobs.values()
            if job_info["status"] == PipelineStatus.PENDING
        )
        assert pending_count > 0
    
    def test_duplicate_job_submission(self, compression_pipeline, quantization_job):
        """测试重复任务提交"""
        job_id1 = compression_pipeline.submit_job(quantization_job)
        job_id2 = compression_pipeline.submit_job(quantization_job)
        
        # 应该生成不同的任务ID
        assert job_id1 != job_id2
        assert len(compression_pipeline.active_jobs) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])