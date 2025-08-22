"""
Token流式处理器测试
"""

import pytest
import asyncio
from typing import AsyncIterator
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ai.streaming.token_streamer import TokenStreamer, StreamType, StreamEvent


class TestTokenStreamer:
    """Token流式处理器测试"""
    
    @pytest.fixture
    def streamer(self):
        """创建测试用的Token流处理器"""
        return TokenStreamer(buffer_size=50)
    
    async def mock_llm_response(self, text: str, delay: float = 0.01) -> AsyncIterator[str]:
        """模拟LLM响应"""
        tokens = text.split()
        for token in tokens:
            yield f"{token} "
            await asyncio.sleep(delay)
    
    @pytest.mark.asyncio
    async def test_basic_token_streaming(self, streamer):
        """测试基本Token流式处理"""
        test_text = "这是一个测试消息"
        events = []
        
        async for event in streamer.stream_tokens(
            self.mock_llm_response(test_text),
            session_id="test_session"
        ):
            events.append(event)
        
        # 验证事件结构
        assert len(events) > 0
        
        # 检查开始事件
        start_events = [e for e in events if e.type == StreamType.METADATA]
        assert len(start_events) == 1
        assert start_events[0].data["status"] == "started"
        
        # 检查Token事件
        token_events = [e for e in events if e.type == StreamType.TOKEN]
        assert len(token_events) == len(test_text.split())
        
        # 检查完成事件
        complete_events = [e for e in events if e.type == StreamType.COMPLETE]
        assert len(complete_events) == 1
        
        # 验证最终响应
        final_response = complete_events[0].data
        assert test_text in final_response
    
    @pytest.mark.asyncio
    async def test_event_sequence_numbers(self, streamer):
        """测试事件序列号"""
        test_text = "测试 序列号"
        events = []
        
        async for event in streamer.stream_tokens(
            self.mock_llm_response(test_text),
            session_id="seq_test"
        ):
            events.append(event)
        
        # 验证序列号递增
        for i in range(1, len(events)):
            assert events[i].sequence > events[i-1].sequence
    
    @pytest.mark.asyncio
    async def test_session_metrics(self, streamer):
        """测试会话指标收集"""
        session_id = "metrics_test"
        test_text = "测试 指标 收集"
        
        # 处理流式响应
        async for event in streamer.stream_tokens(
            self.mock_llm_response(test_text),
            session_id=session_id
        ):
            pass
        
        # 验证指标
        metrics = streamer.get_session_metrics(session_id)
        assert metrics is not None
        assert metrics["token_count"] == len(test_text.split())
        assert metrics["event_count"] > 0
        assert "start_time" in metrics
    
    @pytest.mark.asyncio
    async def test_subscriber_broadcasting(self, streamer):
        """测试订阅者广播机制"""
        # 创建订阅队列
        queue1 = await streamer.subscribe()
        queue2 = await streamer.subscribe()
        
        # 启动流式处理
        task = asyncio.create_task(
            self._process_stream(streamer, "广播 测试")
        )
        
        # 从两个队列接收事件
        events1 = []
        events2 = []
        
        # 接收事件
        for _ in range(5):  # 预期至少有开始、token、完成事件
            try:
                event1 = await asyncio.wait_for(queue1.get(), timeout=1.0)
                events1.append(event1)
                
                event2 = await asyncio.wait_for(queue2.get(), timeout=1.0)
                events2.append(event2)
            except asyncio.TimeoutError:
                break
        
        await task
        
        # 验证两个订阅者都收到了事件
        assert len(events1) > 0
        assert len(events2) > 0
        assert len(events1) == len(events2)
        
        # 验证事件内容一致
        for e1, e2 in zip(events1, events2):
            assert e1.type == e2.type
            assert e1.data == e2.data
            assert e1.sequence == e2.sequence
    
    async def _process_stream(self, streamer: TokenStreamer, text: str):
        """处理流式响应的辅助方法"""
        async for _ in streamer.stream_tokens(
            self.mock_llm_response(text),
            session_id="broadcast_test"
        ):
            pass
    
    @pytest.mark.asyncio
    async def test_error_handling(self, streamer):
        """测试错误处理"""
        async def error_response():
            yield "正常 "
            yield "token "
            raise ValueError("模拟错误")
        
        events = []
        with pytest.raises(ValueError):
            async for event in streamer.stream_tokens(
                error_response(),
                session_id="error_test"
            ):
                events.append(event)
        
        # 验证错误事件
        error_events = [e for e in events if e.type == StreamType.ERROR]
        assert len(error_events) == 1
        assert "模拟错误" in error_events[0].data
        assert error_events[0].metadata["error_type"] == "ValueError"
    
    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self, streamer):
        """测试心跳机制"""
        # 设置短心跳间隔用于测试
        streamer.heartbeat_interval = 0.1
        
        # 创建订阅队列
        queue = await streamer.subscribe()
        
        # 启动长时间的流式处理
        async def long_response():
            for i in range(3):
                yield f"token{i} "
                await asyncio.sleep(0.15)  # 比心跳间隔长
        
        # 处理流式响应
        task = asyncio.create_task(
            self._process_stream_with_queue(streamer, long_response(), queue)
        )
        
        # 收集事件
        events = []
        start_time = time.time()
        
        while time.time() - start_time < 1.0:  # 最多等待1秒
            try:
                event = await asyncio.wait_for(queue.get(), timeout=0.2)
                events.append(event)
                
                # 如果收到完成事件，退出
                if event.type == StreamType.COMPLETE:
                    break
            except asyncio.TimeoutError:
                continue
        
        await task
        
        # 验证心跳事件
        heartbeat_events = [e for e in events if e.type == StreamType.HEARTBEAT]
        assert len(heartbeat_events) > 0  # 应该有心跳事件
    
    async def _process_stream_with_queue(
        self, 
        streamer: TokenStreamer, 
        llm_response: AsyncIterator[str],
        queue: asyncio.Queue
    ):
        """带队列的流式处理辅助方法"""
        async for _ in streamer.stream_tokens(
            llm_response,
            session_id="heartbeat_test"
        ):
            pass
    
    @pytest.mark.asyncio
    async def test_cleanup_and_unsubscribe(self, streamer):
        """测试清理和取消订阅"""
        # 创建多个订阅者
        queues = []
        for i in range(3):
            queue = await streamer.subscribe()
            queues.append(queue)
        
        assert streamer.get_active_subscribers_count() == 3
        
        # 取消订阅
        streamer.unsubscribe(queues[1])
        assert streamer.get_active_subscribers_count() == 2
        
        # 清理会话指标
        session_id = "cleanup_test"
        streamer._session_metrics[session_id] = {"test": "data"}
        
        streamer.clear_session_metrics(session_id)
        assert session_id not in streamer._session_metrics
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, streamer):
        """测试并发会话处理"""
        sessions = ["session1", "session2", "session3"]
        tasks = []
        
        # 启动多个并发会话
        for session_id in sessions:
            task = asyncio.create_task(
                self._process_concurrent_session(streamer, session_id)
            )
            tasks.append(task)
        
        # 等待所有会话完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证所有会话都成功完成
        for result in results:
            assert not isinstance(result, Exception)
        
        # 验证每个会话的指标
        for session_id in sessions:
            metrics = streamer.get_session_metrics(session_id)
            assert metrics is not None
            assert metrics["token_count"] > 0
    
    async def _process_concurrent_session(self, streamer: TokenStreamer, session_id: str):
        """并发会话处理辅助方法"""
        text = f"并发会话 {session_id} 测试"
        event_count = 0
        
        async for event in streamer.stream_tokens(
            self.mock_llm_response(text, delay=0.01),
            session_id=session_id
        ):
            event_count += 1
        
        return event_count