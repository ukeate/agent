"""
消息总线监控集成测试
测试MessageBus与MonitoringManager的完整集成
"""

import pytest
import asyncio
import uuid
from unittest.mock import Mock, AsyncMock, patch

from src.ai.distributed_message.message_bus import DistributedMessageBus
from src.ai.distributed_message.models import MessageType, MessagePriority


class TestMessageBusMonitoringIntegration:
    """消息总线监控集成测试"""
    
    @pytest.fixture
    def message_bus(self):
        """创建消息总线实例"""
        with patch('src.ai.distributed_message.message_bus.NATSClient') as mock_client_class:
            # 模拟NATSClient
            mock_client = Mock()
            mock_client.connect = AsyncMock(return_value=True)
            mock_client.disconnect = AsyncMock(return_value=True)
            
            # 创建带有 unsubscribe 异步方法的 mock subscription
            mock_subscription = Mock()
            mock_subscription.unsubscribe = AsyncMock()
            mock_client.subscribe = AsyncMock(return_value=mock_subscription)
            
            mock_client.publish = AsyncMock(return_value=True)
            mock_client.js_publish = AsyncMock(return_value=Mock(sequence=1))
            mock_client.js = Mock()
            mock_client.is_connected = Mock(return_value=True)
            mock_client.metrics = Mock(
                messages_sent=0,
                messages_received=0,
                bytes_sent=0,
                bytes_received=0,
                messages_failed=0
            )
            mock_client_class.return_value = mock_client
            
            bus = DistributedMessageBus(
                nats_servers=["nats://localhost:4222"],
                agent_id="test-agent",
                cluster_name="test-cluster"
            )
            
            # 设置回调以避免实际NATS操作
            bus.client = mock_client
            
            yield bus
    
    @pytest.mark.asyncio
    async def test_monitoring_manager_lifecycle(self, message_bus):
        """测试监控管理器的生命周期"""
        # 验证初始状态
        assert not message_bus.monitoring_manager.is_running
        
        # 连接应启动监控管理器
        await message_bus.connect()
        assert message_bus.monitoring_manager.is_running
        
        # 断开连接应停止监控管理器
        await message_bus.disconnect()
        assert not message_bus.monitoring_manager.is_running
    
    @pytest.mark.asyncio
    async def test_message_metrics_recording(self, message_bus):
        """测试消息指标记录"""
        await message_bus.connect()
        
        try:
            # 发送消息
            success = await message_bus.send_message(
                receiver_id="target-agent",
                message_type=MessageType.TASK_REQUEST,
                payload={"task": "test_task", "data": "test_data"}
            )
            
            assert success is True
            
            # 验证指标记录
            dashboard = message_bus.get_monitoring_dashboard()
            assert "metrics" in dashboard
            
            # 验证发送消息指标
            metrics = dashboard["metrics"]
            sent_metrics = [m for m in metrics if m.name == "messages_sent"]
            assert len(sent_metrics) > 0
            
            sent_metric = sent_metrics[0]
            assert sent_metric.labels.get("message_type") == MessageType.TASK_REQUEST.value
            assert sent_metric.value == 1
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_compression_integration(self, message_bus):
        """测试压缩集成"""
        await message_bus.connect()
        
        try:
            # 启用压缩
            message_bus.enable_compression(threshold=100)
            
            # 验证压缩已启用
            dashboard = message_bus.get_monitoring_dashboard()
            assert dashboard["performance"]["compression_enabled"] is True
            
            # 发送大消息
            large_payload = {"data": "x" * 500}  # 超过压缩阈值
            success = await message_bus.send_message(
                receiver_id="target-agent",
                message_type=MessageType.DATA_CHUNK,
                payload=large_payload
            )
            
            assert success is True
            
            # 禁用压缩
            message_bus.disable_compression()
            dashboard = message_bus.get_monitoring_dashboard()
            assert dashboard["performance"]["compression_enabled"] is False
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_batching_integration(self, message_bus):
        """测试批处理集成"""
        await message_bus.connect()
        
        try:
            # 启用批处理
            message_bus.enable_batching(batch_size=5, timeout=1.0)
            
            # 验证批处理已启用
            dashboard = message_bus.get_monitoring_dashboard()
            assert dashboard["performance"]["batching_enabled"] is True
            
            # 禁用批处理
            message_bus.disable_batching()
            dashboard = message_bus.get_monitoring_dashboard()
            assert dashboard["performance"]["batching_enabled"] is False
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_alert_integration(self, message_bus):
        """测试告警集成"""
        await message_bus.connect()
        
        try:
            # 注册告警处理器
            alerts_received = []
            
            def alert_handler(alert):
                alerts_received.append(alert)
            
            message_bus.register_alert_handler(alert_handler)
            
            # 手动创建一个告警来测试
            message_bus.monitoring_manager.alert_manager.create_alert(
                "test_alert", 
                message_bus.monitoring_manager.alert_manager.AlertLevel.WARNING,
                "测试告警"
            )
            
            # 等待告警处理
            await asyncio.sleep(0.1)
            
            # 验证告警已处理
            dashboard = message_bus.get_monitoring_dashboard()
            assert dashboard["alerts"]["total_alerts"] == 1
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_health_check_integration(self, message_bus):
        """测试健康检查集成"""
        await message_bus.connect()
        
        try:
            # 获取健康状态
            dashboard = message_bus.get_monitoring_dashboard()
            health_status = dashboard["health"]
            
            # 验证健康检查结果
            assert "overall_health" in health_status
            assert "checks" in health_status
            
            # 基本健康检查应该存在
            check_names = [check["name"] for check in health_status["checks"]]
            assert "system_cpu" in check_names
            assert "system_memory" in check_names
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, message_bus):
        """测试性能指标收集"""
        await message_bus.connect()
        
        try:
            # 等待一些时间让监控收集系统指标
            await asyncio.sleep(1.1)
            
            # 获取仪表板信息
            dashboard = message_bus.get_monitoring_dashboard()
            
            # 验证性能指标
            performance = dashboard["performance"]
            assert "cpu_usage" in performance
            assert "memory_usage" in performance
            assert "disk_usage" in performance
            
            # 验证指标有效性
            assert isinstance(performance["cpu_usage"], (int, float))
            assert isinstance(performance["memory_usage"], (int, float))
            assert isinstance(performance["disk_usage"], (int, float))
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_monitoring_with_failed_messages(self, message_bus):
        """测试失败消息的监控"""
        # 模拟发送失败
        message_bus.client.publish = AsyncMock(side_effect=Exception("发送失败"))
        message_bus.client.js_publish = AsyncMock(side_effect=Exception("发送失败"))
        
        await message_bus.connect()
        
        try:
            # 尝试发送消息（应该失败）
            success = await message_bus.send_message(
                receiver_id="target-agent",
                message_type=MessageType.PING,
                payload={"timestamp": "2025-08-26T12:00:00"}
            )
            
            assert success is False
            
            # 验证失败指标记录
            dashboard = message_bus.get_monitoring_dashboard()
            metrics = dashboard["metrics"]
            failed_metrics = [m for m in metrics if m.name == "messages_failed"]
            assert len(failed_metrics) > 0
            
            failed_metric = failed_metrics[0]
            assert failed_metric.labels.get("message_type") == MessageType.PING.value
            assert failed_metric.value == 1
            
        finally:
            await message_bus.disconnect()
    
    @pytest.mark.asyncio
    async def test_comprehensive_monitoring_dashboard(self, message_bus):
        """测试综合监控仪表板"""
        await message_bus.connect()
        
        try:
            # 发送一些消息
            await message_bus.send_message(
                receiver_id="target-1",
                message_type=MessageType.PING,
                payload={"test": "data1"}
            )
            
            await message_bus.send_message(
                receiver_id="target-2",
                message_type=MessageType.TASK_REQUEST,
                payload={"test": "data2"}
            )
            
            # 启用性能优化
            message_bus.enable_compression(1024)
            message_bus.enable_batching(10, 2.0)
            
            # 等待指标收集
            await asyncio.sleep(0.1)
            
            # 获取完整仪表板
            dashboard = message_bus.get_monitoring_dashboard()
            
            # 验证仪表板结构
            expected_keys = ["metrics", "health", "alerts", "performance"]
            for key in expected_keys:
                assert key in dashboard
            
            # 验证指标存在
            assert len(dashboard["metrics"]) > 0
            
            # 验证健康检查
            assert "overall_health" in dashboard["health"]
            
            # 验证告警信息
            assert "total_alerts" in dashboard["alerts"]
            
            # 验证性能信息
            performance = dashboard["performance"]
            assert "compression_enabled" in performance
            assert "batching_enabled" in performance
            assert performance["compression_enabled"] is True
            assert performance["batching_enabled"] is True
            
        finally:
            await message_bus.disconnect()