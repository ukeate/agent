"""
数据压缩和传输优化处理器

负责事件数据的压缩、解压缩和传输优化。
"""

import gzip
import lz4.frame
import json
import pickle
import asyncio
from typing import List, Dict, Any, Union, Optional, Tuple
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from dataclasses import dataclass
from enum import Enum
import logging

from ..models import BehaviorEvent

logger = logging.getLogger(__name__)


class CompressionMethod(str, Enum):
    """压缩方法枚举"""
    GZIP = "gzip"
    LZ4 = "lz4"
    NONE = "none"


class SerializationMethod(str, Enum):
    """序列化方法枚举"""
    JSON = "json"
    PICKLE = "pickle"


@dataclass
class CompressionMetrics:
    """压缩度量指标"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    compression_time_ms: float
    decompression_time_ms: Optional[float] = None
    method: CompressionMethod = CompressionMethod.GZIP
    serialization: SerializationMethod = SerializationMethod.JSON


class CompressionHandler:
    """数据压缩处理器"""
    
    def __init__(
        self,
        default_method: CompressionMethod = CompressionMethod.GZIP,
        compression_threshold: int = 1024,  # 1KB
        auto_select_method: bool = True
    ):
        self.default_method = default_method
        self.compression_threshold = compression_threshold
        self.auto_select_method = auto_select_method
        
        # 压缩统计
        self.stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'total_bytes_saved': 0,
            'method_usage': {method.value: 0 for method in CompressionMethod},
            'avg_compression_ratio': 0.0
        }
    
    async def compress_events(
        self,
        events: List[BehaviorEvent],
        method: Optional[CompressionMethod] = None,
        serialization: SerializationMethod = SerializationMethod.JSON
    ) -> Tuple[bytes, CompressionMetrics]:
        """压缩事件列表"""
        if not events:
            return b'', CompressionMetrics(0, 0, 0.0, 0.0)
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 序列化数据
            serialized_data = await self._serialize_events(events, serialization)
            original_size = len(serialized_data)
            
            # 判断是否需要压缩
            if original_size < self.compression_threshold:
                method = CompressionMethod.NONE
            elif method is None:
                method = await self._select_best_method(serialized_data) if self.auto_select_method else self.default_method
            
            # 执行压缩
            if method == CompressionMethod.NONE:
                compressed_data = serialized_data
                compressed_size = original_size
            else:
                compressed_data = await self._compress_data(serialized_data, method)
                compressed_size = len(compressed_data)
            
            compression_time = (asyncio.get_event_loop().time() - start_time) * 1000
            compression_ratio = (original_size - compressed_size) / original_size if original_size > 0 else 0.0
            
            # 更新统计
            self._update_compression_stats(method, original_size, compressed_size, compression_ratio)
            
            metrics = CompressionMetrics(
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                compression_time_ms=compression_time,
                method=method,
                serialization=serialization
            )
            
            logger.debug(f"压缩完成: {original_size} -> {compressed_size} bytes, 压缩率: {compression_ratio:.2%}")
            return compressed_data, metrics
            
        except Exception as e:
            logger.error(f"事件压缩失败: {e}")
            raise
    
    async def decompress_events(
        self,
        compressed_data: bytes,
        method: CompressionMethod,
        serialization: SerializationMethod = SerializationMethod.JSON
    ) -> Tuple[List[BehaviorEvent], CompressionMetrics]:
        """解压缩事件列表"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # 解压缩数据
            if method == CompressionMethod.NONE:
                decompressed_data = compressed_data
            else:
                decompressed_data = await self._decompress_data(compressed_data, method)
            
            # 反序列化事件
            events = await self._deserialize_events(decompressed_data, serialization)
            
            decompression_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            # 更新统计
            self.stats['total_decompressions'] += 1
            
            metrics = CompressionMetrics(
                original_size=len(decompressed_data),
                compressed_size=len(compressed_data),
                compression_ratio=0.0,  # 解压时不计算压缩率
                compression_time_ms=0.0,
                decompression_time_ms=decompression_time,
                method=method,
                serialization=serialization
            )
            
            logger.debug(f"解压缩完成: {len(compressed_data)} -> {len(decompressed_data)} bytes")
            return events, metrics
            
        except Exception as e:
            logger.error(f"事件解压缩失败: {e}")
            raise
    
    async def _serialize_events(
        self,
        events: List[BehaviorEvent],
        method: SerializationMethod
    ) -> bytes:
        """序列化事件列表"""
        if method == SerializationMethod.JSON:
            data = json.dumps([event.dict() for event in events], default=str)
            return data.encode('utf-8')
        elif method == SerializationMethod.PICKLE:
            return pickle.dumps([event.dict() for event in events])
        else:
            raise ValueError(f"不支持的序列化方法: {method}")
    
    async def _deserialize_events(
        self,
        data: bytes,
        method: SerializationMethod
    ) -> List[BehaviorEvent]:
        """反序列化事件列表"""
        if method == SerializationMethod.JSON:
            events_data = json.loads(data.decode('utf-8'))
        elif method == SerializationMethod.PICKLE:
            events_data = pickle.loads(data)
        else:
            raise ValueError(f"不支持的反序列化方法: {method}")
        
        return [BehaviorEvent(**event_data) for event_data in events_data]
    
    async def _compress_data(self, data: bytes, method: CompressionMethod) -> bytes:
        """压缩数据"""
        if method == CompressionMethod.GZIP:
            return gzip.compress(data, compresslevel=6)
        elif method == CompressionMethod.LZ4:
            return lz4.frame.compress(data)
        else:
            raise ValueError(f"不支持的压缩方法: {method}")
    
    async def _decompress_data(self, data: bytes, method: CompressionMethod) -> bytes:
        """解压缩数据"""
        if method == CompressionMethod.GZIP:
            return gzip.decompress(data)
        elif method == CompressionMethod.LZ4:
            return lz4.frame.decompress(data)
        else:
            raise ValueError(f"不支持的解压缩方法: {method}")
    
    async def _select_best_method(self, data: bytes) -> CompressionMethod:
        """自动选择最佳压缩方法"""
        if len(data) < self.compression_threshold:
            return CompressionMethod.NONE
        
        # 对于较小数据，使用LZ4(速度快)
        if len(data) < 10240:  # 10KB
            return CompressionMethod.LZ4
        
        # 对于较大数据，使用GZIP(压缩率高)
        return CompressionMethod.GZIP
    
    def _update_compression_stats(
        self,
        method: CompressionMethod,
        original_size: int,
        compressed_size: int,
        compression_ratio: float
    ):
        """更新压缩统计"""
        self.stats['total_compressions'] += 1
        self.stats['method_usage'][method.value] += 1
        
        if original_size > compressed_size:
            self.stats['total_bytes_saved'] += (original_size - compressed_size)
        
        # 更新平均压缩率(简单移动平均)
        total_compressions = self.stats['total_compressions']
        current_avg = self.stats['avg_compression_ratio']
        self.stats['avg_compression_ratio'] = (current_avg * (total_compressions - 1) + compression_ratio) / total_compressions
    
    def get_stats(self) -> Dict[str, Any]:
        """获取压缩统计信息"""
        return {
            **self.stats,
            'compression_threshold': self.compression_threshold,
            'default_method': self.default_method.value,
            'auto_select_method': self.auto_select_method
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_compressions': 0,
            'total_decompressions': 0,
            'total_bytes_saved': 0,
            'method_usage': {method.value: 0 for method in CompressionMethod},
            'avg_compression_ratio': 0.0
        }


class StreamingCompressor:
    """流式压缩器
    
    用于处理大量数据的流式压缩，避免内存占用过大。
    """
    
    def __init__(
        self,
        method: CompressionMethod = CompressionMethod.GZIP,
        chunk_size: int = 8192
    ):
        self.method = method
        self.chunk_size = chunk_size
    
    async def compress_stream(
        self,
        data_stream: asyncio.StreamReader
    ) -> Tuple[asyncio.StreamReader, CompressionMetrics]:
        """流式压缩数据"""
        if self.method == CompressionMethod.NONE:
            return data_stream, CompressionMetrics(0, 0, 0.0, 0.0)
        
        # 创建压缩器
        if self.method == CompressionMethod.GZIP:
            compressor = gzip.GzipFile(mode='wb', fileobj=None)
        elif self.method == CompressionMethod.LZ4:
            # LZ4流式压缩需要特殊处理
            raise NotImplementedError("LZ4流式压缩尚未实现")
        else:
            raise ValueError(f"不支持的流式压缩方法: {self.method}")
        
        # 实现流式压缩逻辑
        # 这里需要更复杂的实现来处理流式数据
        raise NotImplementedError("流式压缩功能尚未完全实现")


class AdaptiveCompressionManager:
    """自适应压缩管理器
    
    根据数据特征和网络条件动态调整压缩策略。
    """
    
    def __init__(self):
        self.compression_handler = CompressionHandler()
        self.performance_history = []
        self.max_history_size = 100
    
    async def compress_with_adaptation(
        self,
        events: List[BehaviorEvent],
        network_speed_mbps: Optional[float] = None,
        cpu_usage_percent: Optional[float] = None
    ) -> Tuple[bytes, CompressionMetrics]:
        """自适应压缩"""
        # 根据网络速度和CPU使用率选择压缩策略
        method = await self._select_adaptive_method(
            len(events),
            network_speed_mbps,
            cpu_usage_percent
        )
        
        compressed_data, metrics = await self.compression_handler.compress_events(
            events, method
        )
        
        # 记录性能历史
        self._record_performance(metrics, network_speed_mbps, cpu_usage_percent)
        
        return compressed_data, metrics
    
    async def _select_adaptive_method(
        self,
        events_count: int,
        network_speed_mbps: Optional[float],
        cpu_usage_percent: Optional[float]
    ) -> CompressionMethod:
        """自适应选择压缩方法"""
        # 如果网络很快且CPU使用率低，使用GZIP获得更好压缩率
        if network_speed_mbps and network_speed_mbps < 10 and (not cpu_usage_percent or cpu_usage_percent < 50):
            return CompressionMethod.GZIP
        
        # 如果CPU使用率高，优先选择速度快的LZ4
        if cpu_usage_percent and cpu_usage_percent > 80:
            return CompressionMethod.LZ4
        
        # 对于小数据量，不压缩或使用LZ4
        if events_count < 10:
            return CompressionMethod.NONE
        
        # 默认使用LZ4平衡压缩率和速度
        return CompressionMethod.LZ4
    
    def _record_performance(
        self,
        metrics: CompressionMetrics,
        network_speed_mbps: Optional[float],
        cpu_usage_percent: Optional[float]
    ):
        """记录性能数据"""
        performance_record = {
            'timestamp': utc_now(),
            'compression_ratio': metrics.compression_ratio,
            'compression_time_ms': metrics.compression_time_ms,
            'method': metrics.method,
            'network_speed_mbps': network_speed_mbps,
            'cpu_usage_percent': cpu_usage_percent
        }
        
        self.performance_history.append(performance_record)
        
        # 保持历史记录大小
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size:]
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """获取性能洞察"""
        if not self.performance_history:
            return {}
        
        # 分析各种压缩方法的性能
        method_performance = {}
        for record in self.performance_history:
            method = record['method'].value
            if method not in method_performance:
                method_performance[method] = {
                    'count': 0,
                    'avg_ratio': 0.0,
                    'avg_time': 0.0
                }
            
            perf = method_performance[method]
            perf['count'] += 1
            perf['avg_ratio'] += record['compression_ratio']
            perf['avg_time'] += record['compression_time_ms']
        
        # 计算平均值
        for method_perf in method_performance.values():
            if method_perf['count'] > 0:
                method_perf['avg_ratio'] /= method_perf['count']
                method_perf['avg_time'] /= method_perf['count']
        
        return {
            'total_compressions': len(self.performance_history),
            'method_performance': method_performance,
            'recent_history': self.performance_history[-10:] if len(self.performance_history) >= 10 else self.performance_history
        }