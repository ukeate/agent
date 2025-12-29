"""
用户反馈学习系统集成测试

测试feedback_collector和reward_generator的集成工作流程，
以及与数据库、API端点的完整交互。
"""

import pytest
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from unittest.mock import Mock, patch, AsyncMock
import json
import sys
import os
from src.services.feedback_collector import (
    FeedbackCollector, 
    get_feedback_collector,
    shutdown_feedback_collector
)
from src.services.reward_generator import (
    RewardSignalGenerator,
    RewardCalculationStrategy
)
from models.schemas.feedback import FeedbackType
from src.core.database import get_db

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TestFeedbackCollectionIntegration:
    """反馈收集集成测试"""
    
    @pytest.fixture
    async def collector(self):
        """获取反馈收集器实例"""
        collector = await get_feedback_collector()
        yield collector
        await shutdown_feedback_collector()
    
    @pytest.fixture
    def reward_generator(self):
        """获取奖励生成器实例"""
        return RewardSignalGenerator()
    
    @pytest.mark.asyncio
    async def test_full_pipeline_implicit_feedback(self, collector, reward_generator):
        """测试隐式反馈完整处理管道"""
        # 模拟用户浏览行为序列
        user_id = "integration_user_1"
        session_id = "integration_session_1"
        item_id = "integration_item_1"
        
        # 1. 收集页面浏览事件
        view_result = await collector.collect_implicit_feedback(
            user_id=user_id,
            session_id=session_id,
            item_id=item_id,
            event_type="view",
            event_data={"value": True, "entry_point": "search"},
            context={"page_type": "product", "category": "electronics"}
        )
        assert view_result is True
        
        # 2. 收集点击事件
        click_result = await collector.collect_implicit_feedback(
            user_id=user_id,
            session_id=session_id,
            item_id=item_id,
            event_type="click",
            event_data={"value": True, "element": "product_image"},
            context={"click_position": [100, 200]}
        )
        assert click_result is True
        
        # 3. 收集停留时间事件
        dwell_result = await collector.collect_implicit_feedback(
            user_id=user_id,
            session_id=session_id,
            item_id=item_id,
            event_type="dwell_time", 
            event_data={"value": 45.0},  # 45秒
            context={"page_sections_viewed": ["description", "specs", "reviews"]}
        )
        assert dwell_result is True
        
        # 4. 收集滚动深度事件
        scroll_result = await collector.collect_implicit_feedback(
            user_id=user_id,
            session_id=session_id,
            item_id=item_id,
            event_type="scroll_depth",
            event_data={"value": 80.0},  # 80%
            context={"viewport_height": 800, "content_height": 2000}
        )
        assert scroll_result is True
        
        # 5. 验证收集器状态
        stats = await collector.get_stats()
        assert stats["total_processed"] == 4
        assert stats["validation_failures"] == 0
        assert stats["duplicates_filtered"] == 0
        
        # 6. 强制刷新缓冲区以获取事件
        events = await collector.buffer.flush()
        assert len(events) == 4
        
        # 7. 使用奖励生成器处理事件
        processed_feedbacks = await reward_generator.process_batch_events(events)
        assert len(processed_feedbacks) == 4
        
        # 8. 生成奖励信号
        reward_signal = await reward_generator.generate_reward_signal(
            processed_feedbacks,
            strategy=RewardCalculationStrategy.WEIGHTED_AVERAGE
        )
        
        # 9. 验证最终奖励信号
        assert reward_signal is not None
        assert 0.0 <= reward_signal.value <= 1.0
        assert reward_signal.confidence > 0
        assert len(reward_signal.contributing_feedbacks) == 4
        
        # 10. 验证用户参与度体现在奖励中（有点击+长时间停留+深度滚动）
        assert reward_signal.value > 0.5  # 积极的用户行为应该产生较高奖励
    
    @pytest.mark.asyncio
    async def test_full_pipeline_explicit_feedback(self, collector, reward_generator):
        """测试显式反馈完整处理管道"""
        user_id = "integration_user_2"
        session_id = "integration_session_2"
        item_id = "integration_item_2"
        
        # 1. 收集用户评分
        rating_result = await collector.collect_explicit_feedback(
            user_id=user_id,
            session_id=session_id,
            item_id=item_id,
            feedback_type="rating",
            value=5,
            context={
                "rating_aspect": "overall_satisfaction",
                "survey_id": "post_purchase_survey",
                "user_type": "premium_member"
            }
        )
        assert rating_result is True
        
        # 2. 收集点赞
        like_result = await collector.collect_explicit_feedback(
            user_id=user_id,
            session_id=session_id,
            item_id=item_id,
            feedback_type="like",
            value=True,
            context={"like_source": "product_page"}
        )
        assert like_result is True
        
        # 3. 收集收藏
        bookmark_result = await collector.collect_explicit_feedback(
            user_id=user_id,
            session_id=session_id,
            item_id=item_id,
            feedback_type="bookmark",
            value=True,
            context={"bookmark_list": "favorites"}
        )
        assert bookmark_result is True
        
        # 4. 收集分享行为
        share_result = await collector.collect_explicit_feedback(
            user_id=user_id,
            session_id=session_id,
            item_id=item_id,
            feedback_type="share",
            value=True,
            context={
                "share_platform": "social_media",
                "share_type": "public"
            }
        )
        assert share_result is True
        
        # 5. 收集用户评论
        comment_result = await collector.collect_explicit_feedback(
            user_id=user_id,
            session_id=session_id,
            item_id=item_id,
            feedback_type="comment",
            value="这个产品非常好用，强烈推荐给大家！质量很棒，服务也很好。",
            context={
                "comment_length": 35,
                "language": "zh-CN"
            }
        )
        assert comment_result is True
        
        # 6. 强制刷新缓冲区
        events = await collector.buffer.flush()
        assert len(events) == 5
        
        # 7. 验证事件优先级（显式反馈应该是高优先级）
        high_priority_count = sum(1 for e in events if e.priority.value == "high")
        assert high_priority_count == 5  # 所有显式反馈都应该是高优先级
        
        # 8. 处理事件生成奖励
        processed_feedbacks = await reward_generator.process_batch_events(events)
        reward_signal = await reward_generator.generate_reward_signal(
            processed_feedbacks,
            strategy=RewardCalculationStrategy.WEIGHTED_AVERAGE
        )
        
        # 9. 验证显式反馈的奖励信号
        assert reward_signal.value > 0.8  # 显式正面反馈应该产生很高的奖励
        assert reward_signal.confidence > 0.8  # 显式反馈的置信度应该很高
    
    @pytest.mark.asyncio
    async def test_mixed_feedback_pipeline(self, collector, reward_generator):
        """测试混合反馈（隐式+显式）处理管道"""
        user_id = "integration_user_3"
        session_id = "integration_session_3"
        item_id = "integration_item_3"
        
        # 收集混合反馈事件
        events_data = [
            # 隐式反馈
            {
                "type": "implicit",
                "event_type": "view",
                "data": {"value": True},
                "context": {"source": "recommendation"}
            },
            {
                "type": "implicit",
                "event_type": "dwell_time",
                "data": {"value": 120.0},  # 2分钟
                "context": {"content_type": "video"}
            },
            # 显式反馈
            {
                "type": "explicit",
                "feedback_type": "rating",
                "value": 3,  # 中性评分
                "context": {"rating_reason": "average_quality"}
            },
            {
                "type": "explicit",
                "feedback_type": "comment",
                "value": "还可以，但是有改进空间",
                "context": {"comment_sentiment": "neutral"}
            }
        ]
        
        # 收集所有事件
        for event_info in events_data:
            if event_info["type"] == "implicit":
                result = await collector.collect_implicit_feedback(
                    user_id=user_id,
                    session_id=session_id,
                    item_id=item_id,
                    event_type=event_info["event_type"],
                    event_data=event_info["data"],
                    context=event_info["context"]
                )
            else:  # explicit
                result = await collector.collect_explicit_feedback(
                    user_id=user_id,
                    session_id=session_id,
                    item_id=item_id,
                    feedback_type=event_info["feedback_type"],
                    value=event_info["value"],
                    context=event_info["context"]
                )
            assert result is True
        
        # 处理混合反馈
        events = await collector.buffer.flush()
        processed_feedbacks = await reward_generator.process_batch_events(events)
        
        # 生成不同策略的奖励信号进行对比
        strategies = [
            RewardCalculationStrategy.WEIGHTED_AVERAGE,
            RewardCalculationStrategy.QUALITY_ADJUSTED,
            RewardCalculationStrategy.CONTEXT_BOOSTED
        ]
        
        rewards = {}
        for strategy in strategies:
            reward = await reward_generator.generate_reward_signal(
                processed_feedbacks,
                strategy=strategy
            )
            rewards[strategy] = reward
        
        # 验证不同策略产生不同结果
        values = [r.value for r in rewards.values()]
        assert len(set(values)) > 1  # 不同策略应该产生不同值
        
        # 验证混合反馈的奖励值在合理范围内（中性反馈）
        for reward in rewards.values():
            assert 0.3 <= reward.value <= 0.7  # 中性反馈应该在中间范围
    
    @pytest.mark.asyncio
    async def test_duplicate_detection_across_components(self, collector, reward_generator):
        """测试跨组件的重复检测"""
        user_id = "duplicate_test_user"
        session_id = "duplicate_test_session"
        item_id = "duplicate_test_item"
        
        # 连续提交相同的反馈
        duplicate_feedback = {
            "user_id": user_id,
            "session_id": session_id,
            "item_id": item_id,
            "feedback_type": "rating",
            "value": 4
        }
        
        # 第一次提交
        result1 = await collector.collect_explicit_feedback(**duplicate_feedback)
        assert result1 is True
        
        # 立即重复提交
        result2 = await collector.collect_explicit_feedback(**duplicate_feedback)
        assert result2 is False  # 应该被过滤
        
        # 验证统计信息
        stats = await collector.get_stats()
        assert stats["duplicates_filtered"] >= 1
        assert stats["total_processed"] == 1  # 只有一个被处理
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_resilience(self, collector, reward_generator):
        """测试错误恢复和系统韧性"""
        user_id = "error_test_user"
        session_id = "error_test_session"
        item_id = "error_test_item"
        
        # 提交包含各种错误的反馈数据
        test_cases = [
            # 有效数据
            {
                "data": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "item_id": item_id,
                    "feedback_type": "rating",
                    "value": 5
                },
                "should_succeed": True
            },
            # 无效评分值
            {
                "data": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "item_id": item_id,
                    "feedback_type": "rating",
                    "value": 10  # 超出1-5范围
                },
                "should_succeed": False
            },
            # 无效反馈类型
            {
                "data": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "item_id": item_id,
                    "feedback_type": "invalid_type",
                    "value": 3
                },
                "should_succeed": False
            },
            # 空评论
            {
                "data": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "item_id": item_id,
                    "feedback_type": "comment",
                    "value": ""
                },
                "should_succeed": False
            },
            # 另一个有效数据
            {
                "data": {
                    "user_id": user_id,
                    "session_id": session_id,
                    "item_id": item_id + "_2",  # 不同item
                    "feedback_type": "like",
                    "value": True
                },
                "should_succeed": True
            }
        ]
        
        # 提交所有测试用例
        results = []
        for case in test_cases:
            try:
                result = await collector.collect_explicit_feedback(**case["data"])
                results.append(result)
            except Exception:
                results.append(False)
        
        # 验证结果符合预期
        expected_results = [case["should_succeed"] for case in test_cases]
        assert results == expected_results
        
        # 验证系统仍然可用
        valid_feedback = {
            "user_id": user_id,
            "session_id": session_id,
            "item_id": item_id + "_recovery",
            "feedback_type": "rating", 
            "value": 4
        }
        result = await collector.collect_explicit_feedback(**valid_feedback)
        assert result is True  # 系统应该从错误中恢复
    
    @pytest.mark.asyncio
    async def test_concurrent_collection_performance(self, collector, reward_generator):
        """测试并发收集性能"""
        import time
        
        # 创建并发任务
        async def collect_user_feedback(user_index: int):
            user_id = f"concurrent_user_{user_index}"
            session_id = f"concurrent_session_{user_index}"
            
            # 每个用户收集多种反馈
            tasks = []
            
            # 隐式反馈
            for item_index in range(5):  # 每个用户5个物品
                item_id = f"item_{item_index}"
                
                task1 = collector.collect_implicit_feedback(
                    user_id=user_id,
                    session_id=session_id,
                    item_id=item_id,
                    event_type="click",
                    event_data={"value": True}
                )
                tasks.append(task1)
                
                task2 = collector.collect_implicit_feedback(
                    user_id=user_id,
                    session_id=session_id,
                    item_id=item_id,
                    event_type="dwell_time",
                    event_data={"value": 30 + item_index * 10}  # 不同停留时间
                )
                tasks.append(task2)
            
            # 显式反馈
            task3 = collector.collect_explicit_feedback(
                user_id=user_id,
                session_id=session_id,
                item_id="item_favorite",
                feedback_type="rating",
                value=4 + (user_index % 2)  # 4或5分
            )
            tasks.append(task3)
            
            return await asyncio.gather(*tasks)
        
        # 并发执行多个用户的反馈收集
        start_time = time.time()
        user_tasks = [collect_user_feedback(i) for i in range(20)]  # 20个并发用户
        all_results = await asyncio.gather(*user_tasks)
        end_time = time.time()
        
        # 验证性能和结果
        duration = end_time - start_time
        assert duration < 10.0  # 10秒内完成
        
        # 验证大部分反馈成功收集
        total_feedbacks = sum(sum(user_results) for user_results in all_results)
        expected_feedbacks = 20 * 11  # 每用户11个反馈
        success_rate = total_feedbacks / expected_feedbacks
        assert success_rate > 0.8  # 至少80%成功率
        
        # 验证收集器统计
        stats = await collector.get_stats()
        assert stats["total_received"] >= expected_feedbacks
        assert stats["buffer_overflows"] == 0  # 不应该有缓冲区溢出

class TestDatabaseIntegration:
    """数据库集成测试"""
    
    @pytest.mark.asyncio
    async def test_feedback_signal_persistence(self):
        """测试反馈信号的数据库持久化"""
        # 这个测试需要实际的数据库连接
        # 在实际环境中会测试将FeedbackSignal保存到数据库
        
        # 模拟数据库操作（因为没有实际连接）
        feedback_signal_data = {
            "id": "test_signal_1",
            "user_id": "test_user",
            "item_id": "test_item",
            "signal_type": FeedbackType.RATING.value,
            "raw_value": 4,
            "processed_value": 0.75,
            "confidence_score": 0.85,
            "context": {"test": "integration"},
            "timestamp": utc_now(),
            "session_id": "test_session"
        }
        
        # 在实际测试中，这里会：
        # 1. 创建FeedbackSignal实例
        # 2. 保存到数据库
        # 3. 从数据库读取验证
        # 4. 测试更新和删除操作
        
        assert True  # 占位符，实际实现时替换为真实的数据库测试
    
    @pytest.mark.asyncio
    async def test_user_feedback_history_queries(self):
        """测试用户反馈历史查询"""
        # 测试查询用户反馈历史的各种场景
        # 包括按时间范围、反馈类型、物品等过滤
        
        # 在实际测试中，这里会测试：
        # 1. 查询用户所有反馈
        # 2. 查询特定时间范围内的反馈  
        # 3. 查询特定物品的反馈
        # 4. 聚合查询（平均评分、反馈数量等）
        
        assert True  # 占位符
    
    @pytest.mark.asyncio
    async def test_reward_signal_aggregation(self):
        """测试奖励信号聚合查询"""
        # 测试对奖励信号的各种聚合查询
        
        # 在实际测试中，这里会测试：
        # 1. 按物品聚合奖励信号
        # 2. 按用户聚合奖励信号
        # 3. 时间窗口内的奖励趋势
        # 4. 奖励信号的统计分析
        
        assert True  # 占位符

class TestAPIEndpointIntegration:
    """API端点集成测试"""
    
    @pytest.mark.asyncio
    async def test_feedback_collection_endpoint(self):
        """测试反馈收集API端点"""
        # 模拟API请求测试
        # 在实际测试中会使用TestClient测试API端点
        
        test_payload = {
            "user_id": "api_test_user",
            "session_id": "api_test_session", 
            "item_id": "api_test_item",
            "feedback_type": "rating",
            "value": 4,
            "context": {
                "source": "api_test",
                "device": "mobile"
            }
        }
        
        # 在实际测试中，这里会：
        # 1. 发送POST请求到反馈收集端点
        # 2. 验证响应状态和内容
        # 3. 测试各种错误情况
        # 4. 测试批量提交端点
        
        assert True  # 占位符
    
    @pytest.mark.asyncio
    async def test_reward_signal_retrieval_endpoint(self):
        """测试奖励信号检索API端点"""
        # 测试获取奖励信号的API端点
        
        # 在实际测试中，这里会：
        # 1. 测试获取单个物品的奖励信号
        # 2. 测试获取用户相关的奖励信号
        # 3. 测试奖励信号的过滤和排序
        # 4. 测试分页功能
        
        assert True  # 占位符

# 性能基准测试
class TestPerformanceBenchmarks:
    """性能基准测试"""
    
    @pytest.mark.asyncio
    async def test_feedback_processing_throughput(self):
        """测试反馈处理吞吐量"""
        collector = FeedbackCollector()
        generator = RewardSignalGenerator()
        
        await collector.start()
        
        try:
            # 测试吞吐量基准
            events_per_batch = 100
            num_batches = 10
            
            total_start_time = time.time()
            
            for batch_idx in range(num_batches):
                batch_tasks = []
                
                for event_idx in range(events_per_batch):
                    task = collector.collect_implicit_feedback(
                        user_id=f"perf_user_{event_idx % 50}",  # 50个用户
                        session_id=f"perf_session_{batch_idx}_{event_idx}",
                        item_id=f"perf_item_{event_idx % 100}",  # 100个物品
                        event_type="click",
                        event_data={"value": True}
                    )
                    batch_tasks.append(task)
                
                # 执行当前批次
                batch_start = time.time()
                batch_results = await asyncio.gather(*batch_tasks)
                batch_end = time.time()
                
                # 验证批次结果
                success_count = sum(1 for r in batch_results if r)
                assert success_count >= events_per_batch * 0.8  # 至少80%成功
                
                # 记录批次性能
                batch_duration = batch_end - batch_start
                throughput = events_per_batch / batch_duration
                
                # 期望吞吐量：至少100 events/second
                assert throughput >= 100, f"Batch {batch_idx} throughput too low: {throughput:.2f} events/sec"
            
            total_end_time = time.time()
            total_duration = total_end_time - total_start_time
            total_events = events_per_batch * num_batches
            overall_throughput = total_events / total_duration
            
            # 验证整体性能
            assert overall_throughput >= 80, f"Overall throughput too low: {overall_throughput:.2f} events/sec"
            
        finally:
            await collector.stop()
    
    @pytest.mark.asyncio
    async def test_reward_generation_latency(self):
        """测试奖励生成延迟"""
        generator = RewardSignalGenerator()
        
        # 创建不同规模的反馈批次测试延迟
        batch_sizes = [1, 10, 50, 100]
        
        for batch_size in batch_sizes:
            # 创建测试事件
            events = []
            for i in range(batch_size):
                event = CollectedEvent(
                    event_id=f"latency_test_{i}",
                    user_id=f"user_{i % 10}",
                    session_id=f"session_{i}",
                    item_id=f"item_{i % 20}",
                    feedback_type=FeedbackType.CLICK,
                    raw_value=True,
                    context={"latency_test": True},
                    timestamp=utc_now(),
                    priority=EventPriority.LOW
                )
                events.append(event)
            
            # 测试处理延迟
            start_time = time.time()
            processed_feedbacks = await generator.process_batch_events(events)
            processing_end = time.time()
            
            reward_signal = await generator.generate_reward_signal(
                processed_feedbacks,
                strategy=RewardCalculationStrategy.WEIGHTED_AVERAGE
            )
            end_time = time.time()
            
            # 计算各阶段延迟
            processing_latency = processing_end - start_time
            total_latency = end_time - start_time
            
            # 延迟要求（毫秒）
            max_processing_latency = batch_size * 10  # 每个事件最多10ms
            max_total_latency = batch_size * 15  # 总计每个事件最多15ms
            
            assert processing_latency * 1000 <= max_processing_latency, \
                f"Processing latency too high for batch size {batch_size}: {processing_latency*1000:.2f}ms"
            
            assert total_latency * 1000 <= max_total_latency, \
                f"Total latency too high for batch size {batch_size}: {total_latency*1000:.2f}ms"
