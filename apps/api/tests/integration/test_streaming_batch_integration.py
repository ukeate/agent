"""流批处理集成测试"""

import pytest
import asyncio
from typing import List, Dict, Any, AsyncIterator
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
import json


class MockStreamProcessor:
    """模拟流处理器"""
    def __init__(self):
        self.processed_tokens = []
        self.stream_metrics = {
            "total_tokens": 0,
            "processing_time": 0,
            "throughput": 0
        }
    
    async def process_stream(self, stream: AsyncIterator[str]) -> AsyncIterator[Dict[str, Any]]:
        """处理token流"""
        start_time = time.time()
        
        async for token in stream:
            self.processed_tokens.append(token)
            self.stream_metrics["total_tokens"] += 1
            
            # 返回处理后的事件
            yield {
                "type": "token",
                "data": token,
                "timestamp": time.time()
            }
        
        self.stream_metrics["processing_time"] = time.time() - start_time
        if self.stream_metrics["processing_time"] > 0:
            self.stream_metrics["throughput"] = (
                self.stream_metrics["total_tokens"] / 
                self.stream_metrics["processing_time"]
            )


class MockBatchProcessor:
    """模拟批处理器"""
    def __init__(self):
        self.jobs = {}
        self.job_counter = 0
    
    async def submit_batch(self, tasks: List[Dict[str, Any]]) -> str:
        """提交批处理任务"""
        job_id = f"job_{self.job_counter}"
        self.job_counter += 1
        
        self.jobs[job_id] = {
            "id": job_id,
            "status": "processing",
            "tasks": tasks,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "start_time": time.time()
        }
        
        # 异步处理任务
        asyncio.create_task(self._process_job(job_id))
        
        return job_id
    
    async def _process_job(self, job_id: str):
        """处理批任务"""
        job = self.jobs[job_id]
        
        for task in job["tasks"]:
            await asyncio.sleep(0.01)  # 模拟处理时间
            
            # 模拟偶尔失败
            if task.get("will_fail"):
                job["failed_tasks"] += 1
            else:
                job["completed_tasks"] += 1
        
        job["status"] = "completed"
        job["end_time"] = time.time()
    
    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        return self.jobs.get(job_id, {"status": "not_found"})


class TestStreamBatchIntegration:
    """流批处理集成测试"""
    
    @pytest.fixture
    def stream_processor(self):
        """创建流处理器"""
        return MockStreamProcessor()
    
    @pytest.fixture
    def batch_processor(self):
        """创建批处理器"""
        return MockBatchProcessor()
    
    @pytest.mark.asyncio
    async def test_basic_stream_processing(self, stream_processor):
        """测试基础流处理"""
        async def mock_token_stream():
            """模拟token流"""
            tokens = ["Hello", " ", "world", "!", " ", "How", " ", "are", " ", "you", "?"]
            for token in tokens:
                yield token
                await asyncio.sleep(0.01)
        
        # 处理流
        events = []
        async for event in stream_processor.process_stream(mock_token_stream()):
            events.append(event)
        
        # 验证处理结果
        assert len(events) == 11
        assert all(e["type"] == "token" for e in events)
        assert "".join(e["data"] for e in events) == "Hello world! How are you?"
        
        # 验证指标
        assert stream_processor.stream_metrics["total_tokens"] == 11
        assert stream_processor.stream_metrics["throughput"] > 0
    
    @pytest.mark.asyncio
    async def test_stream_backpressure_handling(self, stream_processor):
        """测试流背压处理"""
        async def fast_token_stream():
            """快速token流"""
            for i in range(1000):
                yield f"token_{i}"
                # 不等待，模拟快速生产
        
        processed_count = 0
        start_time = time.time()
        
        async for event in stream_processor.process_stream(fast_token_stream()):
            processed_count += 1
            # 模拟慢消费
            if processed_count % 100 == 0:
                await asyncio.sleep(0.1)
        
        processing_time = time.time() - start_time
        
        # 验证背压处理
        assert processed_count == 1000
        assert processing_time > 1.0  # 应该有背压控制
    
    @pytest.mark.asyncio
    async def test_stream_error_recovery(self, stream_processor):
        """测试流错误恢复"""
        async def faulty_stream():
            """带错误的流"""
            for i in range(10):
                if i == 5:
                    raise Exception("Stream error")
                yield f"token_{i}"
        
        events = []
        errors = []
        
        try:
            async for event in stream_processor.process_stream(faulty_stream()):
                events.append(event)
        except Exception as e:
            errors.append(str(e))
        
        # 验证错误处理
        assert len(events) == 5  # 错误前的5个token
        assert len(errors) == 1
        assert "Stream error" in errors[0]
    
    @pytest.mark.asyncio
    async def test_batch_processing_basic(self, batch_processor):
        """测试基础批处理"""
        # 创建批任务
        tasks = []
        for i in range(50):
            tasks.append({
                "id": i,
                "type": "process",
                "data": f"Task {i}"
            })
        
        # 提交批处理
        job_id = await batch_processor.submit_batch(tasks)
        
        # 等待处理完成
        max_wait = 2.0
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            status = await batch_processor.get_job_status(job_id)
            if status["status"] == "completed":
                break
            await asyncio.sleep(0.1)
        
        # 验证结果
        final_status = await batch_processor.get_job_status(job_id)
        assert final_status["status"] == "completed"
        assert final_status["completed_tasks"] == 50
        assert final_status["failed_tasks"] == 0
    
    @pytest.mark.asyncio
    async def test_batch_processing_with_failures(self, batch_processor):
        """测试带失败的批处理"""
        # 创建混合任务
        tasks = []
        for i in range(20):
            tasks.append({
                "id": i,
                "type": "process",
                "will_fail": i % 5 == 0  # 每5个任务失败一个
            })
        
        job_id = await batch_processor.submit_batch(tasks)
        
        # 等待完成
        await asyncio.sleep(0.5)
        
        # 验证结果
        status = await batch_processor.get_job_status(job_id)
        assert status["completed_tasks"] == 16  # 20 - 4个失败
        assert status["failed_tasks"] == 4
    
    @pytest.mark.asyncio
    async def test_stream_batch_hybrid_processing(self, stream_processor, batch_processor):
        """测试流批混合处理"""
        # 流处理部分
        async def generate_stream():
            for i in range(100):
                yield f"item_{i}"
                await asyncio.sleep(0.001)
        
        # 收集流数据到批
        batch_buffer = []
        batch_size = 10
        
        async for event in stream_processor.process_stream(generate_stream()):
            batch_buffer.append(event["data"])
            
            # 达到批大小时提交批处理
            if len(batch_buffer) >= batch_size:
                tasks = [{"id": i, "data": item} for i, item in enumerate(batch_buffer)]
                job_id = await batch_processor.submit_batch(tasks)
                batch_buffer = []
        
        # 处理剩余数据
        if batch_buffer:
            tasks = [{"id": i, "data": item} for i, item in enumerate(batch_buffer)]
            await batch_processor.submit_batch(tasks)
        
        # 验证混合处理
        assert stream_processor.stream_metrics["total_tokens"] == 100
        assert len(batch_processor.jobs) == 10  # 100个项目，批大小10
    
    @pytest.mark.asyncio
    async def test_concurrent_stream_processing(self, stream_processor):
        """测试并发流处理"""
        async def create_stream(stream_id: int):
            """创建独立的流"""
            for i in range(10):
                yield f"stream_{stream_id}_token_{i}"
                await asyncio.sleep(0.01)
        
        # 并发处理多个流
        tasks = []
        processors = []
        
        for i in range(5):
            processor = MockStreamProcessor()
            processors.append(processor)
            
            async def process_stream(p, s):
                events = []
                async for event in p.process_stream(s):
                    events.append(event)
                return events
            
            task = asyncio.create_task(
                process_stream(processor, create_stream(i))
            )
            tasks.append(task)
        
        # 等待所有流处理完成
        results = await asyncio.gather(*tasks)
        
        # 验证并发处理
        assert len(results) == 5
        for i, result in enumerate(results):
            assert len(result) == 10
            assert all(f"stream_{i}" in e["data"] for e in result)
    
    @pytest.mark.asyncio
    async def test_batch_job_scheduling(self, batch_processor):
        """测试批任务调度"""
        job_ids = []
        
        # 提交多个批任务
        for i in range(5):
            tasks = [{"id": j, "priority": i} for j in range(10)]
            job_id = await batch_processor.submit_batch(tasks)
            job_ids.append(job_id)
        
        # 等待所有任务完成
        await asyncio.sleep(1.0)
        
        # 验证所有任务完成
        for job_id in job_ids:
            status = await batch_processor.get_job_status(job_id)
            assert status["status"] == "completed"
            assert status["completed_tasks"] == 10
    
    @pytest.mark.asyncio
    async def test_stream_windowing(self, stream_processor):
        """测试流窗口处理"""
        class WindowedProcessor:
            def __init__(self, window_size=5):
                self.window_size = window_size
                self.windows = []
                self.current_window = []
            
            async def process_windowed(self, stream):
                async for token in stream:
                    self.current_window.append(token)
                    
                    if len(self.current_window) >= self.window_size:
                        self.windows.append(list(self.current_window))
                        self.current_window = []
                
                # 处理最后的窗口
                if self.current_window:
                    self.windows.append(list(self.current_window))
        
        # 创建流
        async def token_stream():
            for i in range(23):
                yield f"token_{i}"
        
        # 窗口处理
        processor = WindowedProcessor(window_size=5)
        await processor.process_windowed(token_stream())
        
        # 验证窗口
        assert len(processor.windows) == 5  # 23个token，窗口大小5
        assert len(processor.windows[0]) == 5
        assert len(processor.windows[-1]) == 3  # 最后窗口只有3个
    
    @pytest.mark.asyncio
    async def test_batch_retry_mechanism(self, batch_processor):
        """测试批处理重试机制"""
        class RetryableBatchProcessor:
            def __init__(self, processor):
                self.processor = processor
                self.retry_count = {}
                self.max_retries = 3
            
            async def submit_with_retry(self, tasks):
                job_id = await self.processor.submit_batch(tasks)
                self.retry_count[job_id] = 0
                
                while self.retry_count[job_id] < self.max_retries:
                    await asyncio.sleep(0.5)
                    status = await self.processor.get_job_status(job_id)
                    
                    if status["status"] == "completed":
                        if status["failed_tasks"] == 0:
                            return {"job_id": job_id, "status": "success"}
                        else:
                            # 重试失败的任务
                            self.retry_count[job_id] += 1
                            failed_tasks = [
                                t for t in tasks 
                                if t.get("will_fail")
                            ]
                            if failed_tasks and self.retry_count[job_id] < self.max_retries:
                                # 移除失败标记进行重试
                                retry_tasks = [
                                    {**t, "will_fail": False} 
                                    for t in failed_tasks
                                ]
                                new_job_id = await self.processor.submit_batch(retry_tasks)
                                self.retry_count[new_job_id] = self.retry_count[job_id]
                                job_id = new_job_id
                
                return {"job_id": job_id, "status": "partial_success"}
        
        # 测试重试
        retryable = RetryableBatchProcessor(batch_processor)
        
        tasks = [
            {"id": i, "will_fail": i < 3}  # 前3个会失败
            for i in range(10)
        ]
        
        result = await retryable.submit_with_retry(tasks)
        
        # 验证重试机制
        assert result["status"] in ["success", "partial_success"]
        assert len(batch_processor.jobs) >= 1  # 至少有一次提交