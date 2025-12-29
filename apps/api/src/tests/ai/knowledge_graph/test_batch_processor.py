"""
批处理器测试

测试高性能批量文档处理和缓存机制
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from ai.knowledge_graph.batch_processor import (
    BatchProcessor, BatchConfig, BatchTask,
    ProcessingStatus, CacheStrategy,
    TaskScheduler, MemoryManager, ResultCache,
    ProcessingMetrics
)
from ai.knowledge_graph.data_models import (
    Entity, Relation, EntityType, RelationType,
    BatchProcessingResult

)

class TestBatchConfig:
    """批处理配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = BatchConfig()
        
        assert config.max_concurrent_tasks == 50
        assert config.max_concurrent_per_model == 10
        assert config.worker_pool_size > 0
        assert config.chunk_size == 1000
        assert config.memory_limit_mb == 2048
        assert config.cache_strategy == CacheStrategy.HYBRID
        assert config.task_timeout_seconds == 300
        assert config.max_retries == 3
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = BatchConfig(
            max_concurrent_tasks=20,
            memory_limit_mb=1024,
            cache_strategy=CacheStrategy.MEMORY,
            enable_distributed=True
        )
        
        assert config.max_concurrent_tasks == 20
        assert config.memory_limit_mb == 1024
        assert config.cache_strategy == CacheStrategy.MEMORY
        assert config.enable_distributed is True

class TestBatchTask:
    """批处理任务测试"""
    
    def test_task_creation(self):
        """测试任务创建"""
        task = BatchTask(
            task_id="test_001",
            document_id="doc_123",
            text="这是一个测试文档",
            language="zh",
            priority=5
        )
        
        assert task.task_id == "test_001"
        assert task.document_id == "doc_123"
        assert task.text == "这是一个测试文档"
        assert task.language == "zh"
        assert task.priority == 5
        assert task.status == ProcessingStatus.PENDING
        assert task.retry_count == 0
        assert isinstance(task.created_at, datetime)
    
    def test_task_with_metadata(self):
        """测试带元数据的任务"""
        metadata = {"source": "test", "category": "demo"}
        
        task = BatchTask(
            task_id="test_002",
            document_id="doc_456",
            text="测试文档",
            metadata=metadata
        )
        
        assert task.metadata == metadata
        assert task.metadata["source"] == "test"

class TestMemoryManager:
    """内存管理器测试"""
    
    @pytest.fixture
    def memory_manager(self):
        """创建内存管理器实例"""
        return MemoryManager(limit_mb=100, gc_threshold=10)
    
    def test_memory_manager_creation(self, memory_manager):
        """测试内存管理器创建"""
        assert memory_manager.limit_mb == 100
        assert memory_manager.gc_threshold == 10
        assert memory_manager.processed_count == 0
    
    @patch('psutil.Process')
    def test_check_memory_usage(self, mock_process, memory_manager):
        """测试内存使用检查"""
        # Mock内存信息
        mock_memory_info = Mock()
        mock_memory_info.rss = 50 * 1024 * 1024  # 50MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        memory_mb = memory_manager.check_memory_usage()
        assert memory_mb == 50.0
    
    @patch('psutil.Process')
    def test_should_gc_by_count(self, mock_process, memory_manager):
        """测试基于计数的垃圾回收触发"""
        # 设置较小的内存使用
        mock_memory_info = Mock()
        mock_memory_info.rss = 10 * 1024 * 1024  # 10MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        # 处理次数未达到阈值
        memory_manager.processed_count = 5
        assert not memory_manager.should_gc()
        
        # 处理次数达到阈值
        memory_manager.processed_count = 10
        assert memory_manager.should_gc()
    
    @patch('psutil.Process')
    def test_should_gc_by_memory(self, mock_process, memory_manager):
        """测试基于内存的垃圾回收触发"""
        # 设置高内存使用（超过80%阈值）
        mock_memory_info = Mock()
        mock_memory_info.rss = 90 * 1024 * 1024  # 90MB，超过100MB的80%
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        memory_manager.processed_count = 1  # 计数未达到阈值
        assert memory_manager.should_gc()
    
    @patch('gc.collect')
    def test_perform_gc(self, mock_gc_collect, memory_manager):
        """测试执行垃圾回收"""
        mock_gc_collect.return_value = 42
        memory_manager.processed_count = 10
        
        memory_manager.perform_gc()
        
        mock_gc_collect.assert_called_once()
        assert memory_manager.processed_count == 0

class TestResultCache:
    """结果缓存测试"""
    
    @pytest.fixture
    def cache(self):
        """创建结果缓存实例"""
        return ResultCache(
            strategy=CacheStrategy.MEMORY,
            size_limit=10,
            ttl_seconds=60
        )
    
    def test_cache_creation(self, cache):
        """测试缓存创建"""
        assert cache.strategy == CacheStrategy.MEMORY
        assert cache.size_limit == 10
        assert cache.ttl_seconds == 60
        assert len(cache.cache) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0
    
    def test_cache_put_and_get(self, cache):
        """测试缓存存储和获取"""
        text = "这是测试文本"
        result = {"entities": [], "relations": []}
        
        # 存储
        cache.put(text, result, "zh")
        
        # 获取
        cached_result = cache.get(text, "zh")
        assert cached_result == result
        assert cache.hit_count == 1
        assert cache.miss_count == 0
    
    def test_cache_miss(self, cache):
        """测试缓存未命中"""
        result = cache.get("不存在的文本", "zh")
        assert result is None
        assert cache.hit_count == 0
        assert cache.miss_count == 1
    
    def test_cache_ttl_expiration(self, cache):
        """测试缓存TTL过期"""
        cache.ttl_seconds = 0.1  # 0.1秒过期
        
        text = "测试文本"
        result = {"entities": [], "relations": []}
        
        cache.put(text, result)
        
        # 立即获取应该成功
        cached_result = cache.get(text)
        assert cached_result == result
        
        # 等待过期后获取应该失败
        time.sleep(0.2)
        cached_result = cache.get(text)
        assert cached_result is None
    
    def test_cache_size_limit_lru_eviction(self, cache):
        """测试缓存大小限制和LRU淘汰"""
        cache.size_limit = 3
        
        # 添加3个缓存项
        for i in range(3):
            cache.put(f"text_{i}", {"data": i})
        
        assert len(cache.cache) == 3
        
        # 访问第一个，使其成为最近使用
        cache.get("text_0")
        
        # 添加第4个项，应该淘汰text_1（最久未使用）
        cache.put("text_3", {"data": 3})
        
        assert len(cache.cache) == 3
        assert cache.get("text_0") is not None  # 应该还在
        assert cache.get("text_1") is None     # 应该被淘汰
        assert cache.get("text_2") is not None  # 应该还在
        assert cache.get("text_3") is not None  # 新添加的
    
    def test_cache_hit_rate(self, cache):
        """测试缓存命中率计算"""
        text = "测试文本"
        result = {"entities": [], "relations": []}
        
        # 初始命中率应该为0
        assert cache.get_hit_rate() == 0.0
        
        # 未命中
        cache.get("不存在")
        assert cache.get_hit_rate() == 0.0
        
        # 存储并命中
        cache.put(text, result)
        cache.get(text)
        
        # 命中率应该为0.5 (1命中/2总数)
        assert cache.get_hit_rate() == 0.5
    
    def test_cache_clear(self, cache):
        """测试缓存清空"""
        cache.put("text1", {"data": 1})
        cache.put("text2", {"data": 2})
        cache.get("text1")  # 增加命中次数
        
        assert len(cache.cache) == 2
        assert cache.hit_count > 0
        
        cache.clear()
        
        assert len(cache.cache) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0
    
    def test_none_strategy_bypass(self):
        """测试NONE策略绕过缓存"""
        cache = ResultCache(strategy=CacheStrategy.NONE)
        
        text = "测试文本"
        result = {"entities": [], "relations": []}
        
        # 存储应该被忽略
        cache.put(text, result)
        assert len(cache.cache) == 0
        
        # 获取应该返回None
        cached_result = cache.get(text)
        assert cached_result is None

class TestTaskScheduler:
    """任务调度器测试"""
    
    @pytest.fixture
    def scheduler(self):
        """创建任务调度器实例"""
        config = BatchConfig(max_concurrent_tasks=5)
        return TaskScheduler(config)
    
    @pytest.mark.asyncio
    async def test_add_and_get_task(self, scheduler):
        """测试添加和获取任务"""
        task = BatchTask(
            task_id="test_001",
            document_id="doc_123",
            text="测试文档"
        )
        
        await scheduler.add_task(task)
        
        retrieved_task = await scheduler.get_next_task()
        assert retrieved_task is not None
        assert retrieved_task.task_id == "test_001"
    
    @pytest.mark.asyncio
    async def test_priority_queue(self, scheduler):
        """测试优先级队列"""
        # 添加不同优先级的任务
        low_priority_task = BatchTask(
            task_id="low",
            document_id="doc1",
            text="低优先级",
            priority=1
        )
        
        high_priority_task = BatchTask(
            task_id="high", 
            document_id="doc2",
            text="高优先级",
            priority=10
        )
        
        # 先添加低优先级，后添加高优先级
        await scheduler.add_task(low_priority_task)
        await scheduler.add_task(high_priority_task)
        
        # 应该先获取高优先级任务
        first_task = await scheduler.get_next_task()
        assert first_task.task_id == "high"
        
        second_task = await scheduler.get_next_task()
        assert second_task.task_id == "low"
    
    @pytest.mark.asyncio
    async def test_task_status_management(self, scheduler):
        """测试任务状态管理"""
        task = BatchTask(
            task_id="test_status",
            document_id="doc_status",
            text="状态测试"
        )
        
        # 标记任务开始
        scheduler.mark_task_started(task)
        assert task.status == ProcessingStatus.PROCESSING
        assert task.started_at is not None
        assert task.task_id in scheduler.active_tasks
        
        # 标记任务完成
        result = {"entities": [], "relations": []}
        scheduler.mark_task_completed(task, result)
        assert task.status == ProcessingStatus.COMPLETED
        assert task.completed_at is not None
        assert task.result == result
        assert task.task_id not in scheduler.active_tasks
        assert task.task_id in scheduler.completed_tasks
        
        # 测试任务失败
        failed_task = BatchTask(
            task_id="test_failed",
            document_id="doc_failed", 
            text="失败测试"
        )
        
        scheduler.mark_task_started(failed_task)
        scheduler.mark_task_failed(failed_task, "测试错误")
        
        assert failed_task.status == ProcessingStatus.FAILED
        assert failed_task.error == "测试错误"
        assert failed_task.retry_count == 1
    
    @pytest.mark.asyncio
    async def test_task_retry_mechanism(self, scheduler):
        """测试任务重试机制"""
        task = BatchTask(
            task_id="test_retry",
            document_id="doc_retry",
            text="重试测试",
            max_retries=2
        )
        
        # 第一次失败
        scheduler.mark_task_started(task)
        scheduler.mark_task_failed(task, "第一次失败")
        
        assert task.retry_count == 1
        assert task.task_id not in scheduler.failed_tasks  # 应该重新排队
        
        # 第二次失败
        scheduler.mark_task_started(task)
        scheduler.mark_task_failed(task, "第二次失败")
        
        assert task.retry_count == 2
        assert task.task_id not in scheduler.failed_tasks  # 还能重试一次
        
        # 第三次失败（达到最大重试次数）
        scheduler.mark_task_started(task)
        scheduler.mark_task_failed(task, "第三次失败")
        
        assert task.retry_count == 3
        assert task.task_id in scheduler.failed_tasks  # 最终失败
    
    def test_queue_and_active_count(self, scheduler):
        """测试队列大小和活动任务统计"""
        assert scheduler.get_queue_size() == 0
        assert scheduler.get_active_count() == 0
        
        # 添加任务到队列
        task1 = BatchTask("task1", "doc1", "text1")
        task2 = BatchTask("task2", "doc2", "text2")
        
        # 由于是异步方法，这里只测试计数逻辑
        scheduler.active_tasks["task1"] = task1
        scheduler.active_tasks["task2"] = task2
        
        assert scheduler.get_active_count() == 2

class TestBatchProcessor:
    """批处理器测试"""
    
    @pytest.fixture
    def processor(self):
        """创建批处理器实例"""
        config = BatchConfig(
            max_concurrent_tasks=5,
            worker_pool_size=2,
            memory_limit_mb=512,
            cache_size_limit=100
        )
        return BatchProcessor(config)
    
    def test_processor_creation(self, processor):
        """测试批处理器创建"""
        assert processor.config.max_concurrent_tasks == 5
        assert processor.config.worker_pool_size == 2
        assert not processor.is_running
        assert len(processor.worker_tasks) == 0
        assert isinstance(processor.metrics, ProcessingMetrics)
    
    @pytest.mark.asyncio
    async def test_initialization(self, processor):
        """测试批处理器初始化"""
        with patch.object(processor.entity_recognizer, 'initialize') as mock_entity_init, \
             patch.object(processor.relation_extractor, 'initialize') as mock_relation_init, \
             patch.object(processor.entity_linker, 'initialize') as mock_linker_init, \
             patch.object(processor.multilingual_processor, 'initialize') as mock_multi_init:
            
            await processor.initialize()
            
            mock_entity_init.assert_called_once()
            mock_relation_init.assert_called_once()
            mock_linker_init.assert_called_once()
            mock_multi_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_worker_lifecycle(self, processor):
        """测试工作协程生命周期"""
        # 启动工作协程
        await processor.start_workers(2)
        
        assert processor.is_running
        assert len(processor.worker_tasks) == 2
        
        # 停止工作协程
        await processor.stop_workers()
        
        assert not processor.is_running
        assert len(processor.worker_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_process_batch(self, processor):
        """测试批量处理"""
        documents = [
            {"id": "doc1", "text": "张三在苹果公司工作。"},
            {"id": "doc2", "text": "李四在谷歌工作。"},
            {"id": "doc3", "text": "王五在微软工作。"}
        ]
        
        batch_id = await processor.process_batch(documents, priority=5)
        
        assert batch_id is not None
        assert batch_id.startswith("batch_")
        assert processor.metrics.total_tasks == 3
    
    @pytest.mark.asyncio 
    async def test_single_task_processing_with_cache(self, processor):
        """测试单任务处理和缓存"""
        # Mock组件以避免实际NLP处理
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
        
        processor.entity_recognizer.extract_entities = AsyncMock(return_value=mock_entities)
        processor.relation_extractor.extract_relations = AsyncMock(return_value=mock_relations)
        processor.entity_linker.link_entities = AsyncMock(return_value=mock_entities)
        
        task = BatchTask(
            task_id="test_single",
            document_id="doc_single",
            text="张三在苹果公司工作",
            language="zh"
        )
        
        # 第一次处理（无缓存）
        await processor._process_single_task(task)
        
        assert task.status == ProcessingStatus.COMPLETED
        assert task.result is not None
        assert "entities" in task.result
        assert "relations" in task.result
        
        # 验证结果被缓存
        cached_result = processor.result_cache.get(task.text, task.language)
        assert cached_result is not None
        
        # 第二次处理相同文本（使用缓存）
        task2 = BatchTask(
            task_id="test_cache",
            document_id="doc_cache",
            text="张三在苹果公司工作",
            language="zh"
        )
        
        await processor._process_single_task(task2)
        
        assert task2.status == ProcessingStatus.COMPLETED
        assert task2.result == cached_result
    
    @pytest.mark.asyncio
    async def test_task_error_handling(self, processor):
        """测试任务错误处理"""
        # Mock抛出异常的组件
        processor.entity_recognizer.extract_entities = AsyncMock(
            side_effect=Exception("实体识别失败")
        )
        
        task = BatchTask(
            task_id="test_error",
            document_id="doc_error",
            text="错误测试文本"
        )
        
        await processor._process_single_task(task)
        
        assert task.status == ProcessingStatus.FAILED
        assert "实体识别失败" in task.error
    
    def test_metrics_update(self, processor):
        """测试指标更新"""
        task = BatchTask(
            task_id="metrics_test",
            document_id="doc_metrics",
            text="指标测试"
        )
        
        task.started_at = utc_now()
        task.completed_at = utc_now()
        
        # 更新成功指标
        processor._update_metrics(task, cache_hit=False, failed=False)
        
        assert processor.metrics.completed_tasks == 1
        assert processor.metrics.failed_tasks == 0
        
        # 更新失败指标
        failed_task = BatchTask(
            task_id="failed_test",
            document_id="doc_failed",
            text="失败测试"
        )
        failed_task.started_at = utc_now()
        failed_task.completed_at = utc_now()
        
        processor._update_metrics(failed_task, cache_hit=False, failed=True)
        
        assert processor.metrics.completed_tasks == 1
        assert processor.metrics.failed_tasks == 1
    
    def test_processing_status(self, processor):
        """测试处理状态获取"""
        status = processor.get_processing_status()
        
        assert "is_running" in status
        assert "queue_size" in status
        assert "active_tasks" in status
        assert "worker_count" in status
        assert "metrics" in status
        assert "cache_stats" in status
        assert "memory_usage_mb" in status
        
        assert status["is_running"] == processor.is_running
        assert status["worker_count"] == len(processor.worker_tasks)
    
    @pytest.mark.asyncio
    async def test_shutdown(self, processor):
        """测试批处理器关闭"""
        # 启动后关闭
        await processor.start_workers(1)
        assert processor.is_running
        
        await processor.shutdown()
        
        assert not processor.is_running
        assert len(processor.worker_tasks) == 0

class TestProcessingMetrics:
    """处理指标测试"""
    
    def test_metrics_initialization(self):
        """测试指标初始化"""
        metrics = ProcessingMetrics()
        
        assert metrics.total_tasks == 0
        assert metrics.completed_tasks == 0
        assert metrics.failed_tasks == 0
        assert metrics.average_processing_time == 0.0
        assert metrics.throughput_docs_per_minute == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.cache_hit_rate == 0.0
        assert metrics.error_rate == 0.0
        assert isinstance(metrics.start_time, datetime)
        assert isinstance(metrics.last_update, datetime)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
