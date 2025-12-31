"""
实时多模态情感数据流处理器
支持文本、音频、视频、生理信号的实时流式处理
"""

from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
import asyncio
import json
import time
from typing import Dict, List, Optional, Any, AsyncIterator, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
import numpy as np
from contextlib import asynccontextmanager
from .core_interfaces import (
    ModalityType, EmotionState, MultiModalEmotion, UnifiedEmotionalData,
    EmotionRecognitionEngine, EmotionalDataFlowManager
)
from .communication_protocol import CommunicationProtocol, ModuleType, Priority

from src.core.logging import get_logger
logger = get_logger(__name__)

class StreamState(str, Enum):
    """数据流状态"""
    IDLE = "idle"
    ACTIVE = "active"
    BUFFERING = "buffering"
    PROCESSING = "processing"
    ERROR = "error"
    STOPPED = "stopped"

class ProcessingMode(str, Enum):
    """处理模式"""
    REAL_TIME = "real_time"        # 实时处理
    BATCH = "batch"                # 批处理
    HYBRID = "hybrid"              # 混合模式

@dataclass
class StreamMetrics:
    """流处理指标"""
    total_processed: int = 0
    processing_rate: float = 0.0  # items/second
    average_latency: float = 0.0  # milliseconds
    error_rate: float = 0.0       # percentage
    buffer_usage: float = 0.0     # percentage
    last_updated: datetime = field(default_factory=datetime.now)

@dataclass
class StreamBuffer:
    """流缓冲区"""
    modality: ModalityType
    data: deque = field(default_factory=deque)
    max_size: int = 1000
    window_size: int = 100  # 滑动窗口大小
    overlap_size: int = 20  # 重叠大小
    
    def add_data(self, item: Any) -> bool:
        """添加数据到缓冲区"""
        if len(self.data) >= self.max_size:
            self.data.popleft()  # 移除最旧数据
        
        self.data.append({
            'data': item,
            'timestamp': utc_now(),
            'processed': False
        })
        return True
    
    def get_processing_window(self) -> List[Any]:
        """获取处理窗口数据"""
        if len(self.data) < self.window_size:
            return list(self.data)
        
        # 返回最新的window_size个数据
        return list(self.data)[-self.window_size:]
    
    def mark_processed(self, count: int):
        """标记已处理的数据"""
        for i, item in enumerate(self.data):
            if i >= count:
                break
            item['processed'] = True

class RealtimeStreamProcessor:
    """实时流处理器"""
    
    def __init__(
        self,
        recognition_engine: EmotionRecognitionEngine,
        data_flow_manager: EmotionalDataFlowManager,
        communication_protocol: CommunicationProtocol,
        processing_mode: ProcessingMode = ProcessingMode.REAL_TIME
    ):
        self.recognition_engine = recognition_engine
        self.data_flow_manager = data_flow_manager
        self.communication_protocol = communication_protocol
        self.processing_mode = processing_mode
        
        # 流状态管理
        self.stream_state = StreamState.IDLE
        self.buffers: Dict[ModalityType, StreamBuffer] = {}
        self.metrics: Dict[ModalityType, StreamMetrics] = {}
        
        # 处理配置
        self.batch_size = 10
        self.processing_interval = 0.1  # seconds
        self.max_latency = 500.0  # milliseconds
        
        # 异步任务管理
        self._processing_tasks: Dict[ModalityType, asyncio.Task] = {}
        self._metrics_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # 回调函数
        self._result_callbacks: List[Callable[[str, UnifiedEmotionalData], None]] = []
        self._error_callbacks: List[Callable[[str, Exception], None]] = []
        
        # 性能监控
        self._latency_samples: Dict[ModalityType, deque] = {}
        self._processing_history: deque = deque(maxlen=1000)
        
        self._initialize_buffers()
    
    def _initialize_buffers(self):
        """初始化缓冲区"""
        for modality in ModalityType:
            self.buffers[modality] = StreamBuffer(
                modality=modality,
                max_size=1000,
                window_size=self._get_window_size(modality),
                overlap_size=self._get_overlap_size(modality)
            )
            self.metrics[modality] = StreamMetrics()
            self._latency_samples[modality] = deque(maxlen=100)
    
    def _get_window_size(self, modality: ModalityType) -> int:
        """获取模态特定的窗口大小"""
        window_sizes = {
            ModalityType.TEXT: 50,
            ModalityType.AUDIO: 200,  # 更大窗口用于音频特征
            ModalityType.VIDEO: 30,   # 视频帧
            ModalityType.IMAGE: 10,
            ModalityType.PHYSIOLOGICAL: 100
        }
        return window_sizes.get(modality, 50)
    
    def _get_overlap_size(self, modality: ModalityType) -> int:
        """获取模态特定的重叠大小"""
        overlap_sizes = {
            ModalityType.TEXT: 10,
            ModalityType.AUDIO: 40,
            ModalityType.VIDEO: 5,
            ModalityType.IMAGE: 2,
            ModalityType.PHYSIOLOGICAL: 20
        }
        return overlap_sizes.get(modality, 10)
    
    async def start_processing(self, user_id: str):
        """启动流处理"""
        if self.stream_state != StreamState.IDLE:
            raise RuntimeError(f"Stream processor already in state: {self.stream_state}")
        
        logger.info(f"Starting realtime stream processor for user {user_id}")
        self.stream_state = StreamState.ACTIVE
        self._shutdown_event.clear()
        
        # 启动各模态的处理任务
        for modality in ModalityType:
            self._processing_tasks[modality] = create_task_with_logging(
                self._process_modality_stream(user_id, modality)
            )
        
        # 启动指标收集任务
        self._metrics_task = create_task_with_logging(
            self._collect_metrics_loop()
        )
        
        logger.info("Realtime stream processor started successfully")
    
    async def stop_processing(self):
        """停止流处理"""
        logger.info("Stopping realtime stream processor")
        self.stream_state = StreamState.STOPPED
        self._shutdown_event.set()
        
        # 取消所有处理任务
        for task in self._processing_tasks.values():
            if not task.done():
                task.cancel()
        
        if self._metrics_task and not self._metrics_task.done():
            self._metrics_task.cancel()
        
        # 等待任务完成
        await asyncio.gather(
            *self._processing_tasks.values(),
            self._metrics_task,
            return_exceptions=True
        )
        
        self._processing_tasks.clear()
        self._metrics_task = None
        self.stream_state = StreamState.IDLE
        
        logger.info("Realtime stream processor stopped")
    
    async def add_stream_data(
        self, 
        user_id: str,
        modality: ModalityType, 
        data: Any
    ) -> bool:
        """添加流数据"""
        if self.stream_state not in [StreamState.ACTIVE, StreamState.BUFFERING]:
            logger.warning(f"Cannot add data in state: {self.stream_state}")
            return False
        
        try:
            # 添加用户ID和时间戳
            stream_data = {
                'user_id': user_id,
                'data': data,
                'timestamp': utc_now(),
                'modality': modality
            }
            
            success = self.buffers[modality].add_data(stream_data)
            if success:
                self.metrics[modality].total_processed += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding stream data: {e}")
            await self._handle_processing_error(user_id, modality, e)
            return False
    
    async def _process_modality_stream(self, user_id: str, modality: ModalityType):
        """处理特定模态的数据流"""
        logger.info(f"Starting {modality.value} stream processing for user {user_id}")
        
        buffer = self.buffers[modality]
        
        while not self._shutdown_event.is_set():
            try:
                # 获取处理窗口
                window_data = buffer.get_processing_window()
                
                if not window_data:
                    await asyncio.sleep(self.processing_interval)
                    continue
                
                # 根据处理模式执行处理
                if self.processing_mode == ProcessingMode.REAL_TIME:
                    await self._process_realtime_window(user_id, modality, window_data)
                elif self.processing_mode == ProcessingMode.BATCH:
                    await self._process_batch_window(user_id, modality, window_data)
                else:  # HYBRID
                    await self._process_hybrid_window(user_id, modality, window_data)
                
                # 标记已处理
                buffer.mark_processed(len(window_data) - buffer.overlap_size)
                
                # 更新指标
                await self._update_processing_metrics(modality, len(window_data))
                
            except asyncio.CancelledError:
                logger.info(f"Stream processing cancelled for {modality.value}")
                break
            except Exception as e:
                logger.error(f"Error in {modality.value} stream processing: {e}")
                await self._handle_processing_error(user_id, modality, e)
                await asyncio.sleep(1.0)  # 错误后短暂暂停
        
        logger.info(f"Stopped {modality.value} stream processing for user {user_id}")
    
    async def _process_realtime_window(
        self, 
        user_id: str, 
        modality: ModalityType, 
        window_data: List[Dict[str, Any]]
    ):
        """实时处理窗口数据"""
        start_time = time.time()
        
        try:
            # 提取最新数据进行处理
            latest_data = window_data[-1]['data']
            
            # 构建输入数据
            input_data = {modality: latest_data}
            
            # 调用情感识别引擎
            recognition_result = await self.recognition_engine.recognize_emotion(input_data)
            
            # 创建统一数据格式
            unified_data = UnifiedEmotionalData(
                user_id=user_id,
                timestamp=utc_now(),
                recognition_result=recognition_result,
                confidence=recognition_result.confidence,
                processing_time=time.time() - start_time,
                data_quality=self._calculate_data_quality(window_data)
            )
            
            # 通过数据流管理器处理
            await self.data_flow_manager.route_data(unified_data)
            
            # 调用结果回调
            for callback in self._result_callbacks:
                try:
                    callback(user_id, unified_data)
                except Exception as e:
                    logger.error(f"Error in result callback: {e}")
            
            # 记录延迟
            latency = (time.time() - start_time) * 1000  # milliseconds
            self._latency_samples[modality].append(latency)
            
        except Exception as e:
            logger.error(f"Error in realtime processing: {e}")
            raise
    
    async def _process_batch_window(
        self, 
        user_id: str, 
        modality: ModalityType, 
        window_data: List[Dict[str, Any]]
    ):
        """批处理窗口数据"""
        if len(window_data) < self.batch_size:
            return  # 等待更多数据
        
        start_time = time.time()
        
        try:
            # 批量处理数据
            batch_results = []
            for item in window_data[-self.batch_size:]:
                input_data = {modality: item['data']}
                result = await self.recognition_engine.recognize_emotion(input_data)
                batch_results.append(result)
            
            # 融合批次结果
            fused_result = self._fuse_batch_results(batch_results)
            
            unified_data = UnifiedEmotionalData(
                user_id=user_id,
                timestamp=utc_now(),
                recognition_result=fused_result,
                confidence=fused_result.confidence,
                processing_time=time.time() - start_time,
                data_quality=self._calculate_data_quality(window_data)
            )
            
            await self.data_flow_manager.route_data(unified_data)
            
            for callback in self._result_callbacks:
                try:
                    callback(user_id, unified_data)
                except Exception as e:
                    logger.error(f"Error in result callback: {e}")
            
            latency = (time.time() - start_time) * 1000
            self._latency_samples[modality].append(latency)
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise
    
    async def _process_hybrid_window(
        self, 
        user_id: str, 
        modality: ModalityType, 
        window_data: List[Dict[str, Any]]
    ):
        """混合模式处理"""
        # 根据数据特性选择处理方式
        if len(window_data) >= self.batch_size:
            await self._process_batch_window(user_id, modality, window_data)
        else:
            await self._process_realtime_window(user_id, modality, window_data)
    
    def _fuse_batch_results(self, results: List[MultiModalEmotion]) -> MultiModalEmotion:
        """融合批次结果"""
        if not results:
            raise ValueError("Empty results list")
        
        if len(results) == 1:
            return results[0]
        
        # 计算平均情感状态
        emotions = [r.fused_emotion for r in results]
        
        avg_valence = sum(e.valence for e in emotions) / len(emotions)
        avg_arousal = sum(e.arousal for e in emotions) / len(emotions)
        avg_dominance = sum(e.dominance for e in emotions) / len(emotions)
        avg_intensity = sum(e.intensity for e in emotions) / len(emotions)
        avg_confidence = sum(e.confidence for e in emotions) / len(emotions)
        
        # 确定主导情感
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion.emotion] = emotion_counts.get(emotion.emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts.keys(), key=lambda k: emotion_counts[k])
        
        fused_emotion = EmotionState(
            emotion=dominant_emotion,
            intensity=avg_intensity,
            valence=avg_valence,
            arousal=avg_arousal,
            dominance=avg_dominance,
            confidence=avg_confidence,
            timestamp=utc_now()
        )
        
        return MultiModalEmotion(
            emotions=results[0].emotions,  # 使用最新的单模态结果
            fused_emotion=fused_emotion,
            confidence=avg_confidence,
            processing_time=max(r.processing_time for r in results)
        )
    
    def _calculate_data_quality(self, window_data: List[Dict[str, Any]]) -> float:
        """计算数据质量分数"""
        if not window_data:
            return 0.0
        
        quality_factors = []
        
        # 数据完整性
        completeness = len([d for d in window_data if d.get('data') is not None]) / len(window_data)
        quality_factors.append(completeness)
        
        # 时间一致性 - 检查时间戳间隔的一致性
        timestamps = [d['timestamp'] for d in window_data if 'timestamp' in d]
        if len(timestamps) > 1:
            intervals = [(timestamps[i] - timestamps[i-1]).total_seconds() 
                        for i in range(1, len(timestamps))]
            interval_variance = np.var(intervals) if intervals else 0
            time_consistency = max(0, 1 - min(interval_variance / 0.1, 1))  # 标准化
            quality_factors.append(time_consistency)
        
        # 数据新鲜度
        if window_data:
            latest_timestamp = max(d['timestamp'] for d in window_data)
            age = (utc_now() - latest_timestamp).total_seconds()
            freshness = max(0, 1 - min(age / 10.0, 1))  # 10秒内为新鲜数据
            quality_factors.append(freshness)
        
        return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
    
    async def _collect_metrics_loop(self):
        """指标收集循环"""
        while not self._shutdown_event.is_set():
            try:
                await self._update_stream_metrics()
                await asyncio.sleep(1.0)  # 每秒更新一次指标
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_stream_metrics(self):
        """更新流指标"""
        current_time = utc_now()
        
        for modality, metrics in self.metrics.items():
            # 计算处理速率
            time_diff = (current_time - metrics.last_updated).total_seconds()
            if time_diff > 0:
                recent_count = len([h for h in self._processing_history 
                                  if (current_time - h['timestamp']).total_seconds() <= 1.0
                                  and h['modality'] == modality])
                metrics.processing_rate = recent_count / max(time_diff, 1.0)
            
            # 计算平均延迟
            if self._latency_samples[modality]:
                metrics.average_latency = sum(self._latency_samples[modality]) / len(self._latency_samples[modality])
            
            # 计算缓冲区使用率
            buffer = self.buffers[modality]
            metrics.buffer_usage = len(buffer.data) / buffer.max_size
            
            metrics.last_updated = current_time
    
    async def _update_processing_metrics(self, modality: ModalityType, processed_count: int):
        """更新处理指标"""
        self._processing_history.append({
            'timestamp': utc_now(),
            'modality': modality,
            'count': processed_count
        })
    
    async def _handle_processing_error(self, user_id: str, modality: ModalityType, error: Exception):
        """处理错误"""
        logger.error(f"Processing error for user {user_id}, modality {modality.value}: {error}")
        
        # 更新错误率
        metrics = self.metrics[modality]
        metrics.error_rate = min(metrics.error_rate + 0.01, 1.0)
        
        # 调用错误回调
        for callback in self._error_callbacks:
            try:
                callback(user_id, error)
            except Exception as e:
                logger.error(f"Error in error callback: {e}")
        
        # 如果错误率过高，暂时停止该模态的处理
        if metrics.error_rate > 0.5:
            logger.warning(f"High error rate for {modality.value}, temporarily stopping")
            if modality in self._processing_tasks:
                self._processing_tasks[modality].cancel()
    
    def add_result_callback(self, callback: Callable[[str, UnifiedEmotionalData], None]):
        """添加结果回调"""
        self._result_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[str, Exception], None]):
        """添加错误回调"""
        self._error_callbacks.append(callback)
    
    def remove_result_callback(self, callback: Callable[[str, UnifiedEmotionalData], None]):
        """移除结果回调"""
        if callback in self._result_callbacks:
            self._result_callbacks.remove(callback)
    
    def remove_error_callback(self, callback: Callable[[str, Exception], None]):
        """移除错误回调"""
        if callback in self._error_callbacks:
            self._error_callbacks.remove(callback)
    
    def get_stream_status(self) -> Dict[str, Any]:
        """获取流状态"""
        return {
            'state': self.stream_state.value,
            'processing_mode': self.processing_mode.value,
            'active_modalities': list(self._processing_tasks.keys()),
            'buffer_states': {
                modality.value: {
                    'size': len(buffer.data),
                    'max_size': buffer.max_size,
                    'usage': len(buffer.data) / buffer.max_size
                }
                for modality, buffer in self.buffers.items()
            },
            'metrics': {
                modality.value: {
                    'total_processed': metrics.total_processed,
                    'processing_rate': metrics.processing_rate,
                    'average_latency': metrics.average_latency,
                    'error_rate': metrics.error_rate,
                    'buffer_usage': metrics.buffer_usage
                }
                for modality, metrics in self.metrics.items()
            }
        }
    
    @asynccontextmanager
    async def stream_session(self, user_id: str):
        """流处理会话上下文管理器"""
        await self.start_processing(user_id)
        try:
            yield self
        finally:
            await self.stop_processing()

class MultiUserStreamManager:
    """多用户流管理器"""
    
    def __init__(
        self,
        recognition_engine: EmotionRecognitionEngine,
        data_flow_manager: EmotionalDataFlowManager,
        communication_protocol: CommunicationProtocol
    ):
        self.recognition_engine = recognition_engine
        self.data_flow_manager = data_flow_manager
        self.communication_protocol = communication_protocol
        
        # 用户流处理器映射
        self.user_processors: Dict[str, RealtimeStreamProcessor] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start_manager(self):
        """启动管理器"""
        logger.info("Starting multi-user stream manager")
        self._shutdown_event.clear()
        self._cleanup_task = create_task_with_logging(self._cleanup_inactive_users())
    
    async def stop_manager(self):
        """停止管理器"""
        logger.info("Stopping multi-user stream manager")
        self._shutdown_event.set()
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
        
        # 停止所有用户的流处理器
        tasks = []
        for processor in self.user_processors.values():
            tasks.append(processor.stop_processing())
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        self.user_processors.clear()
        logger.info("Multi-user stream manager stopped")
    
    async def get_user_processor(self, user_id: str) -> RealtimeStreamProcessor:
        """获取或创建用户流处理器"""
        if user_id not in self.user_processors:
            processor = RealtimeStreamProcessor(
                recognition_engine=self.recognition_engine,
                data_flow_manager=self.data_flow_manager,
                communication_protocol=self.communication_protocol,
                processing_mode=ProcessingMode.HYBRID
            )
            self.user_processors[user_id] = processor
            await processor.start_processing(user_id)
        
        return self.user_processors[user_id]
    
    async def add_user_data(
        self, 
        user_id: str, 
        modality: ModalityType, 
        data: Any
    ) -> bool:
        """为用户添加流数据"""
        processor = await self.get_user_processor(user_id)
        return await processor.add_stream_data(user_id, modality, data)
    
    async def remove_user(self, user_id: str):
        """移除用户流处理器"""
        if user_id in self.user_processors:
            await self.user_processors[user_id].stop_processing()
            del self.user_processors[user_id]
            logger.info(f"Removed stream processor for user {user_id}")
    
    async def _cleanup_inactive_users(self):
        """清理不活跃的用户"""
        while not self._shutdown_event.is_set():
            try:
                current_time = utc_now()
                inactive_users = []
                
                for user_id, processor in self.user_processors.items():
                    # 检查最后活动时间
                    last_activity = max(
                        metrics.last_updated 
                        for metrics in processor.metrics.values()
                    )
                    
                    if (current_time - last_activity) > timedelta(minutes=30):
                        inactive_users.append(user_id)
                
                # 移除不活跃用户
                for user_id in inactive_users:
                    await self.remove_user(user_id)
                
                await asyncio.sleep(300)  # 每5分钟清理一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)
    
    def get_manager_status(self) -> Dict[str, Any]:
        """获取管理器状态"""
        return {
            'active_users': len(self.user_processors),
            'users': list(self.user_processors.keys()),
            'total_streams': sum(
                len(processor._processing_tasks) 
                for processor in self.user_processors.values()
            )
        }
