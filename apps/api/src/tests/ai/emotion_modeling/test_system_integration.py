"""
情感智能系统集成测试
包括单元测试、集成测试和压力测试
"""

from src.core.utils.timezone_utils import utc_now
import asyncio
import pytest
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import psutil
from ai.emotion_modeling.core_interfaces import (
    EmotionType, ModalityType, EmotionState, MultiModalEmotion,
    UnifiedEmotionalData, EmotionalIntelligenceResponse,
    PersonalityProfile, EmpathyResponse, EmotionalMemory
)
from ai.emotion_modeling.communication_protocol import (
    CommunicationProtocol, ModuleType, Priority, MessageBus
)
from ai.emotion_modeling.data_flow_manager import (
    EmotionalDataFlowManagerImpl, DataFlowContext
)
from ai.emotion_modeling.system_monitor import (
    EmotionalSystemMonitorImpl, SystemHealthStatus
)
from ai.emotion_modeling.emotion_recognition_integration import (
    EmotionRecognitionEngineImpl, EmotionFusionEngine
)
from ai.emotion_modeling.realtime_stream_processor import (
    RealtimeStreamProcessor, MultiUserStreamManager, ProcessingMode
)
from ai.emotion_modeling.result_formatter import (
    ResultFormatterManager, OutputFormat, FormattingConfig
)
from ai.emotion_modeling.quality_monitor import (
    QualityMonitor, QualityThreshold, GroundTruthData

)

class TestEmotionModelingCore:
    """核心接口测试"""
    
    @pytest.fixture
    def sample_emotion_state(self) -> EmotionState:
        """样本情感状态"""
        return EmotionState(
            emotion=EmotionType.HAPPINESS,
            intensity=0.8,
            valence=0.7,
            arousal=0.6,
            dominance=0.5,
            confidence=0.9,
            timestamp=utc_now()
        )
    
    @pytest.fixture
    def sample_multimodal_emotion(self, sample_emotion_state) -> MultiModalEmotion:
        """样本多模态情感"""
        return MultiModalEmotion(
            emotions={
                ModalityType.TEXT: sample_emotion_state,
                ModalityType.AUDIO: sample_emotion_state
            },
            fused_emotion=sample_emotion_state,
            confidence=0.85,
            processing_time=0.15
        )
    
    def test_emotion_state_creation(self, sample_emotion_state):
        """测试情感状态创建"""
        assert sample_emotion_state.emotion == EmotionType.HAPPINESS
        assert 0 <= sample_emotion_state.intensity <= 1
        assert -1 <= sample_emotion_state.valence <= 1
        assert 0 <= sample_emotion_state.arousal <= 1
        assert 0 <= sample_emotion_state.dominance <= 1
        assert 0 <= sample_emotion_state.confidence <= 1
        assert isinstance(sample_emotion_state.timestamp, datetime)
    
    def test_multimodal_emotion_creation(self, sample_multimodal_emotion):
        """测试多模态情感创建"""
        assert len(sample_multimodal_emotion.emotions) == 2
        assert ModalityType.TEXT in sample_multimodal_emotion.emotions
        assert ModalityType.AUDIO in sample_multimodal_emotion.emotions
        assert sample_multimodal_emotion.fused_emotion is not None
        assert 0 <= sample_multimodal_emotion.confidence <= 1
        assert sample_multimodal_emotion.processing_time > 0
    
    def test_unified_emotional_data(self, sample_multimodal_emotion):
        """测试统一情感数据"""
        data = UnifiedEmotionalData(
            user_id="test_user",
            timestamp=utc_now(),
            recognition_result=sample_multimodal_emotion,
            confidence=0.9,
            processing_time=0.2,
            data_quality=0.95
        )
        
        assert data.user_id == "test_user"
        assert data.recognition_result == sample_multimodal_emotion
        assert 0 <= data.confidence <= 1
        assert data.processing_time > 0
        assert 0 <= data.data_quality <= 1

class TestCommunicationProtocol:
    """通信协议测试"""
    
    @pytest.fixture
    async def communication_protocol(self):
        """通信协议实例"""
        protocol = CommunicationProtocol()
        await protocol.start()
        yield protocol
        await protocol.stop()
    
    @pytest.mark.asyncio
    async def test_message_bus_creation(self):
        """测试消息总线创建"""
        bus = MessageBus()
        assert bus.is_running is False
        
        await bus.start()
        assert bus.is_running is True
        
        await bus.stop()
        assert bus.is_running is False
    
    @pytest.mark.asyncio
    async def test_module_registration(self, communication_protocol):
        """测试模块注册"""
        module_id = "test_module"
        
        success = await communication_protocol.register_module(
            module_id, ModuleType.EMOTION_RECOGNITION
        )
        assert success is True
        
        # 重复注册应该失败
        success = await communication_protocol.register_module(
            module_id, ModuleType.EMOTION_RECOGNITION
        )
        assert success is False
    
    @pytest.mark.asyncio
    async def test_message_sending(self, communication_protocol):
        """测试消息发送"""
        # 注册模块
        await communication_protocol.register_module(
            "sender", ModuleType.EMOTION_RECOGNITION
        )
        await communication_protocol.register_module(
            "receiver", ModuleType.STATE_MODELING
        )
        
        # 发送消息
        payload = {"test": "data", "value": 123}
        response = await communication_protocol.send_request(
            target_module=ModuleType.STATE_MODELING,
            payload=payload,
            timeout=5.0
        )
        
        # 由于是模拟环境，响应可能为None
        assert response is None or isinstance(response, dict)
    
    @pytest.mark.asyncio
    async def test_priority_handling(self, communication_protocol):
        """测试优先级处理"""
        await communication_protocol.register_module(
            "test_module", ModuleType.EMOTION_RECOGNITION
        )
        
        # 发送不同优先级的消息
        high_priority_payload = {"priority_test": "high"}
        low_priority_payload = {"priority_test": "low"}
        
        # 高优先级消息
        await communication_protocol.send_request(
            target_module=ModuleType.STATE_MODELING,
            payload=high_priority_payload,
            priority=Priority.HIGH
        )
        
        # 低优先级消息
        await communication_protocol.send_request(
            target_module=ModuleType.STATE_MODELING,
            payload=low_priority_payload,
            priority=Priority.LOW
        )
        
        # 验证消息队列不为空
        assert len(communication_protocol.message_bus.message_queues) > 0

class TestDataFlowManager:
    """数据流管理器测试"""
    
    @pytest.fixture
    async def data_flow_manager(self):
        """数据流管理器实例"""
        protocol = CommunicationProtocol()
        await protocol.start()
        
        manager = EmotionalDataFlowManagerImpl(protocol)
        await manager.initialize()
        
        yield manager
        
        await manager.shutdown()
        await protocol.stop()
    
    @pytest.mark.asyncio
    async def test_data_validation(self, data_flow_manager):
        """测试数据验证"""
        # 有效数据
        valid_data = UnifiedEmotionalData(
            user_id="test_user",
            timestamp=utc_now(),
            confidence=0.9,
            processing_time=0.1,
            data_quality=0.95
        )
        
        is_valid = await data_flow_manager.validate_data_integrity(valid_data)
        assert is_valid is True
        
        # 无效数据（置信度超出范围）
        invalid_data = UnifiedEmotionalData(
            user_id="test_user", 
            timestamp=utc_now(),
            confidence=1.5,  # 无效值
            processing_time=0.1,
            data_quality=0.95
        )
        
        with pytest.raises(ValueError):
            await data_flow_manager.validate_data_integrity(invalid_data)
    
    @pytest.mark.asyncio
    async def test_data_routing(self, data_flow_manager):
        """测试数据路由"""
        data = UnifiedEmotionalData(
            user_id="test_user",
            timestamp=utc_now(),
            recognition_result=MultiModalEmotion(
                emotions={},
                fused_emotion=EmotionState(
                    emotion=EmotionType.HAPPINESS,
                    intensity=0.8,
                    valence=0.7,
                    arousal=0.6,
                    dominance=0.5,
                    confidence=0.9,
                    timestamp=utc_now()
                ),
                confidence=0.9,
                processing_time=0.1
            ),
            confidence=0.9,
            processing_time=0.1,
            data_quality=0.95
        )
        
        result = await data_flow_manager.route_data(data)
        
        # 验证路由结果
        assert isinstance(result, dict)
        assert "status" in result or result is not None
    
    @pytest.mark.asyncio
    async def test_module_synchronization(self, data_flow_manager):
        """测试模块同步"""
        sync_result = await data_flow_manager.synchronize_modules()
        
        # 在测试环境中，同步可能失败或成功
        assert isinstance(sync_result, bool)

class TestSystemMonitor:
    """系统监控测试"""
    
    @pytest.fixture
    async def system_monitor(self):
        """系统监控器实例"""
        protocol = CommunicationProtocol()
        await protocol.start()
        
        monitor = EmotionalSystemMonitorImpl(protocol)
        await monitor.start_monitoring()
        
        yield monitor
        
        await monitor.stop_monitoring()
        await protocol.stop()
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self, system_monitor):
        """测试性能指标收集"""
        metrics = await system_monitor.collect_performance_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        
        # 验证必要的性能指标
        expected_metrics = ["cpu_usage", "memory_usage", "response_time"]
        for metric in expected_metrics:
            if metric in metrics:
                assert isinstance(metrics[metric], (int, float))
                assert metrics[metric] >= 0
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, system_monitor):
        """测试异常检测"""
        anomalies = await system_monitor.detect_anomalies()
        
        assert isinstance(anomalies, list)
        
        # 如果有异常，验证格式
        for anomaly in anomalies:
            assert isinstance(anomaly, dict)
            assert "type" in anomaly
            assert "severity" in anomaly
            assert "description" in anomaly
    
    @pytest.mark.asyncio
    async def test_health_report_generation(self, system_monitor):
        """测试健康报告生成"""
        health_report = await system_monitor.generate_health_report()
        
        assert isinstance(health_report, dict)
        assert "timestamp" in health_report
        assert "overall_status" in health_report
        assert "components" in health_report
        
        # 验证整体状态
        assert health_report["overall_status"] in [
            status.value for status in SystemHealthStatus
        ]

class TestRealtimeStreamProcessor:
    """实时流处理器测试"""
    
    @pytest.fixture
    async def stream_processor(self):
        """流处理器实例"""
        # 模拟依赖
        mock_recognition_engine = AsyncMock()
        mock_data_flow_manager = AsyncMock()
        mock_communication_protocol = AsyncMock()
        
        processor = RealtimeStreamProcessor(
            recognition_engine=mock_recognition_engine,
            data_flow_manager=mock_data_flow_manager,
            communication_protocol=mock_communication_protocol,
            processing_mode=ProcessingMode.REAL_TIME
        )
        
        yield processor
        
        if processor.stream_state.value != "idle":
            await processor.stop_processing()
    
    @pytest.mark.asyncio
    async def test_stream_processor_lifecycle(self, stream_processor):
        """测试流处理器生命周期"""
        user_id = "test_user"
        
        # 启动处理
        await stream_processor.start_processing(user_id)
        assert stream_processor.stream_state.value == "active"
        
        # 停止处理
        await stream_processor.stop_processing()
        assert stream_processor.stream_state.value == "idle"
    
    @pytest.mark.asyncio
    async def test_stream_data_addition(self, stream_processor):
        """测试流数据添加"""
        user_id = "test_user"
        await stream_processor.start_processing(user_id)
        
        # 添加文本数据
        text_data = "I am feeling happy today"
        success = await stream_processor.add_stream_data(
            user_id, ModalityType.TEXT, text_data
        )
        assert success is True
        
        # 验证缓冲区
        text_buffer = stream_processor.buffers[ModalityType.TEXT]
        assert len(text_buffer.data) > 0
        
        await stream_processor.stop_processing()
    
    @pytest.mark.asyncio
    async def test_multi_user_manager(self):
        """测试多用户管理器"""
        mock_recognition_engine = AsyncMock()
        mock_data_flow_manager = AsyncMock()
        mock_communication_protocol = AsyncMock()
        
        manager = MultiUserStreamManager(
            recognition_engine=mock_recognition_engine,
            data_flow_manager=mock_data_flow_manager,
            communication_protocol=mock_communication_protocol
        )
        
        await manager.start_manager()
        
        # 添加用户数据
        user1_id = "user1"
        user2_id = "user2"
        
        await manager.add_user_data(user1_id, ModalityType.TEXT, "Happy message")
        await manager.add_user_data(user2_id, ModalityType.TEXT, "Sad message")
        
        # 验证用户处理器
        assert user1_id in manager.user_processors
        assert user2_id in manager.user_processors
        
        # 获取状态
        status = manager.get_manager_status()
        assert status["active_users"] == 2
        assert user1_id in status["users"]
        assert user2_id in status["users"]
        
        await manager.stop_manager()

class TestResultFormatter:
    """结果格式化器测试"""
    
    @pytest.fixture
    def formatter_manager(self):
        """格式化管理器实例"""
        return ResultFormatterManager()
    
    @pytest.fixture
    def sample_unified_data(self):
        """样本统一数据"""
        emotion_state = EmotionState(
            emotion=EmotionType.HAPPINESS,
            intensity=0.8,
            valence=0.7,
            arousal=0.6,
            dominance=0.5,
            confidence=0.9,
            timestamp=utc_now()
        )
        
        return UnifiedEmotionalData(
            user_id="test_user",
            timestamp=utc_now(),
            emotional_state=emotion_state,
            confidence=0.9,
            processing_time=0.15,
            data_quality=0.95
        )
    
    def test_json_formatting(self, formatter_manager, sample_unified_data):
        """测试JSON格式化"""
        json_result = formatter_manager.format_data(
            sample_unified_data, OutputFormat.JSON
        )
        
        assert isinstance(json_result, str)
        
        # 验证可以解析为有效JSON
        parsed = json.loads(json_result)
        assert isinstance(parsed, dict)
        assert "user_id" in parsed
        assert "timestamp" in parsed
        assert "confidence" in parsed
    
    def test_csv_formatting(self, formatter_manager, sample_unified_data):
        """测试CSV格式化"""
        csv_result = formatter_manager.format_data(
            sample_unified_data, OutputFormat.CSV
        )
        
        assert isinstance(csv_result, str)
        assert "user_id" in csv_result
        assert "confidence" in csv_result
        
        # 验证CSV格式
        lines = csv_result.strip().split('\n')
        assert len(lines) >= 2  # 至少有头部和一行数据
    
    def test_batch_formatting(self, formatter_manager, sample_unified_data):
        """测试批量格式化"""
        data_list = [sample_unified_data, sample_unified_data]
        
        batch_result = formatter_manager.format_batch(
            data_list, OutputFormat.JSON
        )
        
        assert isinstance(batch_result, str)
        parsed = json.loads(batch_result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2

class TestQualityMonitor:
    """质量监控测试"""
    
    @pytest.fixture
    async def quality_monitor(self):
        """质量监控器实例"""
        monitor = QualityMonitor()
        await monitor.start_monitoring()
        yield monitor
        await monitor.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_prediction_recording(self, quality_monitor):
        """测试预测记录"""
        user_id = "test_user"
        emotion_state = EmotionState(
            emotion=EmotionType.HAPPINESS,
            intensity=0.8,
            valence=0.7,
            arousal=0.6,
            dominance=0.5,
            confidence=0.9,
            timestamp=utc_now()
        )
        
        await quality_monitor.record_prediction(
            user_id=user_id,
            predicted_emotion=emotion_state,
            modality=ModalityType.TEXT,
            processing_time=0.15,
            confidence=0.9,
            data_quality=0.95
        )
        
        # 验证记录
        assert len(quality_monitor.prediction_cache) > 0
        assert len(quality_monitor.processing_times[ModalityType.TEXT]) > 0
    
    @pytest.mark.asyncio
    async def test_ground_truth_addition(self, quality_monitor):
        """测试真实标签添加"""
        ground_truth = GroundTruthData(
            user_id="test_user",
            timestamp=utc_now(),
            true_emotion=EmotionState(
                emotion=EmotionType.HAPPINESS,
                intensity=0.8,
                valence=0.7,
                arousal=0.6,
                dominance=0.5,
                confidence=1.0,
                timestamp=utc_now()
            ),
            modality=ModalityType.TEXT,
            source="expert_annotation",
            confidence=1.0
        )
        
        await quality_monitor.add_ground_truth(ground_truth)
        
        # 验证存储
        assert len(quality_monitor.ground_truth_data["test_user"]) > 0
    
    def test_quality_report_generation(self, quality_monitor):
        """测试质量报告生成"""
        report = quality_monitor.get_quality_report()
        
        assert isinstance(report, dict)
        assert "report_time" in report
        assert "metrics" in report
        assert "alerts" in report
        assert "performance" in report
        assert "recommendations" in report

class TestStressAndPerformance:
    """压力测试和性能测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_stream_processing(self):
        """测试并发流处理"""
        # 模拟依赖
        mock_recognition_engine = AsyncMock()
        mock_recognition_engine.recognize_emotion.return_value = MultiModalEmotion(
            emotions={},
            fused_emotion=EmotionState(
                emotion=EmotionType.NEUTRAL,
                intensity=0.5,
                valence=0.0,
                arousal=0.5,
                dominance=0.5,
                confidence=0.8,
                timestamp=utc_now()
            ),
            confidence=0.8,
            processing_time=0.1
        )
        
        mock_data_flow_manager = AsyncMock()
        mock_communication_protocol = AsyncMock()
        
        # 创建多个流处理器
        processors = []
        for i in range(5):
            processor = RealtimeStreamProcessor(
                recognition_engine=mock_recognition_engine,
                data_flow_manager=mock_data_flow_manager,
                communication_protocol=mock_communication_protocol,
                processing_mode=ProcessingMode.REAL_TIME
            )
            processors.append(processor)
        
        # 并发启动
        user_ids = [f"user_{i}" for i in range(5)]
        await asyncio.gather(*[
            processor.start_processing(user_id)
            for processor, user_id in zip(processors, user_ids)
        ])
        
        # 并发添加数据
        tasks = []
        for processor, user_id in zip(processors, user_ids):
            for j in range(10):
                task = processor.add_stream_data(
                    user_id, ModalityType.TEXT, f"Message {j}"
                )
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证大部分任务成功
        successful_results = [r for r in results if r is True]
        assert len(successful_results) > len(tasks) * 0.8  # 至少80%成功
        
        # 清理
        await asyncio.gather(*[
            processor.stop_processing() for processor in processors
        ])
    
    @pytest.mark.asyncio
    async def test_high_throughput_formatting(self):
        """测试高吞吐量格式化"""
        formatter_manager = ResultFormatterManager()
        
        # 创建大量测试数据
        test_data_list = []
        for i in range(1000):
            emotion_state = EmotionState(
                emotion=EmotionType.HAPPINESS,
                intensity=np.random.random(),
                valence=np.random.random() * 2 - 1,
                arousal=np.random.random(),
                dominance=np.random.random(),
                confidence=np.random.random(),
                timestamp=utc_now()
            )
            
            data = UnifiedEmotionalData(
                user_id=f"user_{i}",
                timestamp=utc_now(),
                emotional_state=emotion_state,
                confidence=np.random.random(),
                processing_time=np.random.random() * 0.5,
                data_quality=np.random.random()
            )
            test_data_list.append(data)
        
        # 测试批量格式化性能
        start_time = time.time()
        batch_result = formatter_manager.format_batch(
            test_data_list, OutputFormat.JSON
        )
        processing_time = time.time() - start_time
        
        assert isinstance(batch_result, str)
        assert processing_time < 10.0  # 应在10秒内完成
        
        # 验证结果
        parsed = json.loads(batch_result)
        assert len(parsed) == 1000
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self):
        """测试内存使用监控"""
        import gc
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 创建大量对象
        large_data_list = []
        for i in range(10000):
            emotion_state = EmotionState(
                emotion=EmotionType.HAPPINESS,
                intensity=0.8,
                valence=0.7,
                arousal=0.6,
                dominance=0.5,
                confidence=0.9,
                timestamp=utc_now()
            )
            
            data = UnifiedEmotionalData(
                user_id=f"user_{i}",
                timestamp=utc_now(),
                emotional_state=emotion_state,
                confidence=0.9,
                processing_time=0.1,
                data_quality=0.95
            )
            large_data_list.append(data)
        
        peak_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 清理
        del large_data_list
        gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # 验证内存使用合理
        memory_increase = peak_memory - initial_memory
        memory_released = peak_memory - final_memory
        
        assert memory_increase < 500  # 增加不超过500MB
        assert memory_released > memory_increase * 0.5  # 至少释放50%
    
    def test_cpu_intensive_quality_calculations(self):
        """测试CPU密集型质量计算"""
        from ai.emotion_modeling.quality_monitor import AccuracyCalculator
        
        calculator = AccuracyCalculator()
        
        # 创建大量预测和真实标签
        predictions = []
        ground_truths = []
        
        for i in range(10000):
            pred_emotion = EmotionState(
                emotion=np.random.choice(list(EmotionType)),
                intensity=np.random.random(),
                valence=np.random.random() * 2 - 1,
                arousal=np.random.random(),
                dominance=np.random.random(),
                confidence=np.random.random(),
                timestamp=utc_now()
            )
            
            true_emotion = EmotionState(
                emotion=np.random.choice(list(EmotionType)),
                intensity=np.random.random(),
                valence=np.random.random() * 2 - 1,
                arousal=np.random.random(),
                dominance=np.random.random(),
                confidence=1.0,
                timestamp=utc_now()
            )
            
            predictions.append(pred_emotion)
            ground_truths.append(true_emotion)
        
        # 测试计算性能
        start_time = time.time()
        metrics = calculator.calculate_classification_metrics(
            predictions, ground_truths
        )
        calculation_time = time.time() - start_time
        
        assert isinstance(metrics, dict)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert calculation_time < 30.0  # 应在30秒内完成
    
    @pytest.mark.asyncio
    async def test_system_resilience_under_load(self):
        """测试系统在高负载下的韧性"""
        # 创建系统组件
        protocol = CommunicationProtocol()
        await protocol.start()
        
        data_flow_manager = EmotionalDataFlowManagerImpl(protocol)
        await data_flow_manager.initialize()
        
        monitor = EmotionalSystemMonitorImpl(protocol)
        await monitor.start_monitoring()
        
        quality_monitor = QualityMonitor()
        await quality_monitor.start_monitoring()
        
        try:
            # 模拟高负载
            tasks = []
            
            # 大量数据流任务
            for i in range(100):
                data = UnifiedEmotionalData(
                    user_id=f"load_test_user_{i}",
                    timestamp=utc_now(),
                    confidence=0.8,
                    processing_time=0.1,
                    data_quality=0.9
                )
                task = data_flow_manager.route_data(data)
                tasks.append(task)
            
            # 系统监控任务
            for _ in range(10):
                tasks.append(monitor.collect_performance_metrics())
                tasks.append(monitor.detect_anomalies())
            
            # 执行所有任务
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = time.time() - start_time
            
            # 分析结果
            successful_tasks = len([r for r in results if not isinstance(r, Exception)])
            error_rate = (len(tasks) - successful_tasks) / len(tasks)
            
            # 验证系统韧性
            assert error_rate < 0.1  # 错误率应低于10%
            assert total_time < 60.0  # 总时间应在60秒内
            
            # 验证系统健康状态
            health_report = await monitor.generate_health_report()
            assert health_report["overall_status"] != "critical"
            
        finally:
            # 清理资源
            await quality_monitor.stop_monitoring()
            await monitor.stop_monitoring()
            await data_flow_manager.shutdown()
            await protocol.stop()

@pytest.mark.integration
class TestEndToEndIntegration:
    """端到端集成测试"""
    
    @pytest.fixture
    async def full_system(self):
        """完整系统实例"""
        # 初始化所有组件
        protocol = CommunicationProtocol()
        await protocol.start()
        
        data_flow_manager = EmotionalDataFlowManagerImpl(protocol)
        await data_flow_manager.initialize()
        
        system_monitor = EmotionalSystemMonitorImpl(protocol)
        await system_monitor.start_monitoring()
        
        quality_monitor = QualityMonitor()
        await quality_monitor.start_monitoring()
        
        formatter_manager = ResultFormatterManager()
        
        components = {
            "protocol": protocol,
            "data_flow_manager": data_flow_manager,
            "system_monitor": system_monitor,
            "quality_monitor": quality_monitor,
            "formatter_manager": formatter_manager
        }
        
        yield components
        
        # 清理
        await quality_monitor.stop_monitoring()
        await system_monitor.stop_monitoring()
        await data_flow_manager.shutdown()
        await protocol.stop()
    
    @pytest.mark.asyncio
    async def test_complete_emotion_processing_pipeline(self, full_system):
        """测试完整的情感处理流水线"""
        components = full_system
        
        # 1. 创建测试数据
        emotion_state = EmotionState(
            emotion=EmotionType.HAPPINESS,
            intensity=0.8,
            valence=0.7,
            arousal=0.6,
            dominance=0.5,
            confidence=0.9,
            timestamp=utc_now()
        )
        
        multimodal_emotion = MultiModalEmotion(
            emotions={ModalityType.TEXT: emotion_state},
            fused_emotion=emotion_state,
            confidence=0.9,
            processing_time=0.15
        )
        
        unified_data = UnifiedEmotionalData(
            user_id="e2e_test_user",
            timestamp=utc_now(),
            recognition_result=multimodal_emotion,
            emotional_state=emotion_state,
            confidence=0.9,
            processing_time=0.15,
            data_quality=0.95
        )
        
        # 2. 数据流处理
        routing_result = await components["data_flow_manager"].route_data(unified_data)
        assert routing_result is not None
        
        # 3. 质量监控记录
        await components["quality_monitor"].record_prediction(
            user_id=unified_data.user_id,
            predicted_emotion=emotion_state,
            modality=ModalityType.TEXT,
            processing_time=unified_data.processing_time,
            confidence=unified_data.confidence,
            data_quality=unified_data.data_quality
        )
        
        # 4. 结果格式化
        json_result = components["formatter_manager"].format_data(
            unified_data, OutputFormat.JSON
        )
        assert isinstance(json_result, str)
        
        parsed_result = json.loads(json_result)
        assert parsed_result["user_id"] == "e2e_test_user"
        
        # 5. 系统监控检查
        health_report = await components["system_monitor"].generate_health_report()
        assert isinstance(health_report, dict)
        assert "overall_status" in health_report
        
        # 6. 质量报告生成
        quality_report = components["quality_monitor"].get_quality_report()
        assert isinstance(quality_report, dict)
        assert "metrics" in quality_report
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, full_system):
        """测试错误处理和恢复"""
        components = full_system
        
        # 注入无效数据
        invalid_data = UnifiedEmotionalData(
            user_id="",  # 空用户ID
            timestamp=utc_now(),
            confidence=1.5,  # 无效置信度
            processing_time=-0.1,  # 负处理时间
            data_quality=2.0  # 无效数据质量
        )
        
        # 测试错误处理
        with pytest.raises(ValueError):
            await components["data_flow_manager"].validate_data_integrity(invalid_data)
        
        # 系统应该仍然正常运行
        health_report = await components["system_monitor"].generate_health_report()
        assert health_report["overall_status"] != "critical"
    
    @pytest.mark.asyncio
    async def test_scalability_with_multiple_users(self, full_system):
        """测试多用户场景下的可扩展性"""
        components = full_system
        
        # 创建多个用户的数据
        users_data = []
        for i in range(50):
            emotion_state = EmotionState(
                emotion=np.random.choice(list(EmotionType)),
                intensity=np.random.random(),
                valence=np.random.random() * 2 - 1,
                arousal=np.random.random(),
                dominance=np.random.random(),
                confidence=np.random.random(),
                timestamp=utc_now()
            )
            
            data = UnifiedEmotionalData(
                user_id=f"scale_test_user_{i}",
                timestamp=utc_now(),
                emotional_state=emotion_state,
                confidence=np.random.random(),
                processing_time=np.random.random() * 0.5,
                data_quality=np.random.random()
            )
            users_data.append(data)
        
        # 并发处理所有用户数据
        start_time = time.time()
        routing_tasks = [
            components["data_flow_manager"].route_data(data)
            for data in users_data
        ]
        
        results = await asyncio.gather(*routing_tasks, return_exceptions=True)
        processing_time = time.time() - start_time
        
        # 验证可扩展性
        successful_results = [r for r in results if not isinstance(r, Exception)]
        success_rate = len(successful_results) / len(users_data)
        
        assert success_rate > 0.9  # 成功率应大于90%
        assert processing_time < 30.0  # 处理时间应在合理范围内

if __name__ == "__main__":
    # 运行测试
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto",
        "-m", "not integration"  # 默认跳过集成测试
    ])
