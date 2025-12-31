"""
网络感知监控器

实现网络连接检测，支持：
- 连接质量评估
- 网络状态事件
- 自动降级策略
"""

import asyncio
import time
import socket
import httpx
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now

from src.core.utils.async_utils import create_task_with_logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from ..models.schemas.offline import NetworkStatus
from ..core.logging import get_logger

from src.core.logging import get_logger
class ConnectionTest(str, Enum):
    """连接测试类型"""
    PING = "ping"
    HTTP = "http"
    DNS = "dns"
    WEBSOCKET = "websocket"

@dataclass
class NetworkMetrics:
    """网络指标"""
    latency_ms: float
    packet_loss_rate: float
    bandwidth_kbps: Optional[float] = None
    jitter_ms: Optional[float] = None
    timestamp: datetime = field(default_factory=utc_now)

@dataclass
class ConnectionResult:
    """连接测试结果"""
    test_type: ConnectionTest
    success: bool
    latency_ms: float
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=utc_now)

class NetworkMonitor:
    """网络监控器"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.current_status = NetworkStatus.UNKNOWN
        self.current_metrics = NetworkMetrics(latency_ms=0, packet_loss_rate=0)
        
        # 监控配置
        self.test_endpoints = [
            "https://www.google.com",
            "https://www.baidu.com",
            "https://httpbin.org/get"
        ]
        self.dns_servers = ["8.8.8.8", "114.114.114.114"]
        
        # 阈值配置
        self.latency_thresholds = {
            "excellent": 50,
            "good": 100,
            "fair": 200,
            "poor": 500
        }
        self.packet_loss_thresholds = {
            "excellent": 0.01,
            "good": 0.05,
            "fair": 0.1,
            "poor": 0.2
        }
        
        # 历史记录
        self.metrics_history: List[NetworkMetrics] = []
        self.max_history_size = 100
        
        # 状态变化回调
        self.status_change_callbacks: List[Callable[[NetworkStatus, NetworkMetrics], None]] = []
        
        # 监控任务
        self._monitoring_task = None
        self._is_monitoring = False
        self.monitor_interval = 30  # 30秒检测一次
    
    async def start_monitoring(self):
        """开始网络监控"""
        if self._is_monitoring:
            return
        
        self._is_monitoring = True
        self._monitoring_task = create_task_with_logging(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """停止网络监控"""
        self._is_monitoring = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                raise
    
    async def _monitoring_loop(self):
        """监控循环"""
        while self._is_monitoring:
            try:
                # 执行网络检测
                await self._perform_network_check()
                
                # 等待下一次检测
                await asyncio.sleep(self.monitor_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("网络监控错误", error=str(e))
                await asyncio.sleep(self.monitor_interval)
    
    async def _perform_network_check(self):
        """执行网络检测"""
        # 并行执行多种连接测试
        test_tasks = [
            self._test_http_connectivity(),
            self._test_dns_resolution(),
            self._measure_latency()
        ]
        
        results = await asyncio.gather(*test_tasks, return_exceptions=True)
        
        # 分析结果并更新状态
        connection_results = [r for r in results if isinstance(r, list)]
        
        if connection_results:
            # 合并所有测试结果
            all_results = []
            for result_list in connection_results:
                all_results.extend(result_list)
            
            # 计算网络指标
            metrics = self._calculate_metrics(all_results)
            
            # 评估网络状态
            new_status = self._evaluate_network_status(metrics)
            
            # 更新状态和指标
            old_status = self.current_status
            self.current_status = new_status
            self.current_metrics = metrics
            
            # 添加到历史记录
            self._add_to_history(metrics)
            
            # 触发状态变化回调
            if old_status != new_status:
                for callback in self.status_change_callbacks:
                    try:
                        await callback(new_status, metrics)
                    except Exception as e:
                        self.logger.error("状态变化回调错误", error=str(e))
    
    async def _test_http_connectivity(self) -> List[ConnectionResult]:
        """测试HTTP连接"""
        results = []
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint in self.test_endpoints:
                start_time = time.time()
                try:
                    response = await client.get(endpoint)
                    latency = (time.time() - start_time) * 1000
                    
                    success = response.status_code == 200
                    error_msg = None if success else f"HTTP {response.status_code}"
                    
                    results.append(ConnectionResult(
                        test_type=ConnectionTest.HTTP,
                        success=success,
                        latency_ms=latency,
                        error_message=error_msg
                    ))
                    
                except Exception as e:
                    latency = (time.time() - start_time) * 1000
                    results.append(ConnectionResult(
                        test_type=ConnectionTest.HTTP,
                        success=False,
                        latency_ms=latency,
                        error_message=str(e)
                    ))
        
        return results
    
    async def _test_dns_resolution(self) -> List[ConnectionResult]:
        """测试DNS解析"""
        results = []
        
        for dns_server in self.dns_servers:
            start_time = time.time()
            try:
                # 测试DNS解析
                loop = asyncio.get_running_loop()
                await loop.getaddrinfo("www.google.com", 80, family=socket.AF_INET)
                
                latency = (time.time() - start_time) * 1000
                results.append(ConnectionResult(
                    test_type=ConnectionTest.DNS,
                    success=True,
                    latency_ms=latency
                ))
                
            except Exception as e:
                latency = (time.time() - start_time) * 1000
                results.append(ConnectionResult(
                    test_type=ConnectionTest.DNS,
                    success=False,
                    latency_ms=latency,
                    error_message=str(e)
                ))
        
        return results
    
    async def _measure_latency(self) -> List[ConnectionResult]:
        """测量网络延迟"""
        results = []
        
        # 使用TCP连接测试延迟
        test_hosts = [
            ("www.google.com", 80),
            ("www.baidu.com", 80),
            ("httpbin.org", 80)
        ]
        
        for host, port in test_hosts:
            start_time = time.time()
            try:
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(host, port),
                    timeout=5.0
                )
                latency = (time.time() - start_time) * 1000
                
                writer.close()
                await writer.wait_closed()
                
                results.append(ConnectionResult(
                    test_type=ConnectionTest.PING,
                    success=True,
                    latency_ms=latency
                ))
                
            except Exception as e:
                latency = (time.time() - start_time) * 1000
                results.append(ConnectionResult(
                    test_type=ConnectionTest.PING,
                    success=False,
                    latency_ms=latency,
                    error_message=str(e)
                ))
        
        return results
    
    def _calculate_metrics(self, results: List[ConnectionResult]) -> NetworkMetrics:
        """计算网络指标"""
        if not results:
            return NetworkMetrics(latency_ms=float('inf'), packet_loss_rate=1.0)
        
        # 分离成功和失败的结果
        successful_results = [r for r in results if r.success]
        total_tests = len(results)
        successful_tests = len(successful_results)
        
        # 计算丢包率
        packet_loss_rate = (total_tests - successful_tests) / total_tests
        
        # 计算平均延迟
        if successful_results:
            latencies = [r.latency_ms for r in successful_results]
            avg_latency = sum(latencies) / len(latencies)
            
            # 计算抖动（延迟标准差）
            if len(latencies) > 1:
                variance = sum((l - avg_latency) ** 2 for l in latencies) / len(latencies)
                jitter = variance ** 0.5
            else:
                jitter = 0.0
        else:
            avg_latency = float('inf')
            jitter = 0.0
        
        return NetworkMetrics(
            latency_ms=avg_latency,
            packet_loss_rate=packet_loss_rate,
            jitter_ms=jitter
        )
    
    def _evaluate_network_status(self, metrics: NetworkMetrics) -> NetworkStatus:
        """评估网络状态"""
        # 如果完全无法连接
        if metrics.packet_loss_rate >= 1.0:
            return NetworkStatus.DISCONNECTED
        
        # 基于延迟和丢包率评估
        if (metrics.latency_ms <= self.latency_thresholds["excellent"] and 
            metrics.packet_loss_rate <= self.packet_loss_thresholds["excellent"]):
            return NetworkStatus.CONNECTED
        
        elif (metrics.latency_ms <= self.latency_thresholds["good"] and 
              metrics.packet_loss_rate <= self.packet_loss_thresholds["good"]):
            return NetworkStatus.CONNECTED
        
        elif (metrics.latency_ms <= self.latency_thresholds["fair"] and 
              metrics.packet_loss_rate <= self.packet_loss_thresholds["fair"]):
            return NetworkStatus.WEAK
        
        else:
            return NetworkStatus.WEAK
    
    def _add_to_history(self, metrics: NetworkMetrics):
        """添加指标到历史记录"""
        self.metrics_history.append(metrics)
        
        # 保持历史记录大小限制
        if len(self.metrics_history) > self.max_history_size:
            self.metrics_history = self.metrics_history[-self.max_history_size:]
    
    def add_status_change_callback(self, callback: Callable[[NetworkStatus, NetworkMetrics], None]):
        """添加状态变化回调"""
        self.status_change_callbacks.append(callback)
    
    def remove_status_change_callback(self, callback: Callable[[NetworkStatus, NetworkMetrics], None]):
        """移除状态变化回调"""
        if callback in self.status_change_callbacks:
            self.status_change_callbacks.remove(callback)
    
    async def force_check(self) -> NetworkStatus:
        """强制执行网络检测"""
        await self._perform_network_check()
        return self.current_status
    
    def get_current_status(self) -> NetworkStatus:
        """获取当前网络状态"""
        return self.current_status
    
    def get_current_metrics(self) -> NetworkMetrics:
        """获取当前网络指标"""
        return self.current_metrics
    
    def get_connection_quality_score(self) -> float:
        """获取连接质量评分 (0-1)"""
        if self.current_status == NetworkStatus.DISCONNECTED:
            return 0.0
        
        # 基于延迟和丢包率计算质量评分
        latency_score = max(0, 1 - self.current_metrics.latency_ms / 1000)  # 1秒以上延迟为0分
        loss_score = max(0, 1 - self.current_metrics.packet_loss_rate * 10)  # 10%丢包率以上为0分
        
        return (latency_score + loss_score) / 2
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        if not self.metrics_history:
            return {
                "current_status": self.current_status.value,
                "current_latency_ms": 0,
                "current_packet_loss": 0,
                "connection_quality": 0.0,
                "uptime_percentage": 0.0,
                "average_latency_ms": 0,
                "history_size": 0
            }
        
        # 计算历史统计
        recent_history = self.metrics_history[-20:]  # 最近20次记录
        
        connected_count = sum(
            1 for m in recent_history 
            if m.packet_loss_rate < 1.0
        )
        uptime_percentage = connected_count / len(recent_history) * 100
        
        valid_latencies = [
            m.latency_ms for m in recent_history 
            if m.latency_ms != float('inf')
        ]
        avg_latency = sum(valid_latencies) / len(valid_latencies) if valid_latencies else 0
        
        return {
            "current_status": self.current_status.value,
            "current_latency_ms": self.current_metrics.latency_ms,
            "current_packet_loss": self.current_metrics.packet_loss_rate,
            "current_jitter_ms": self.current_metrics.jitter_ms,
            "connection_quality": self.get_connection_quality_score(),
            "uptime_percentage": uptime_percentage,
            "average_latency_ms": avg_latency,
            "history_size": len(self.metrics_history),
            "last_check": self.current_metrics.timestamp.isoformat()
        }

class ModeSwitcher:
    """自动模式切换器"""
    
    def __init__(self, network_monitor: NetworkMonitor):
        self.logger = get_logger(__name__)
        self.network_monitor = network_monitor
        self.mode_change_callbacks: List[Callable[[NetworkStatus], None]] = []
        
        # 切换配置
        self.offline_threshold_seconds = 30  # 30秒无连接后切换到离线
        self.online_threshold_seconds = 10   # 10秒连接稳定后切换到在线
        
        # 状态跟踪
        self.last_online_time = utc_now()
        self.last_offline_time = None
        self.current_mode = "auto"
        
        # 注册网络状态变化回调
        self.network_monitor.add_status_change_callback(self._handle_network_change)
    
    async def _handle_network_change(self, status: NetworkStatus, metrics: NetworkMetrics):
        """处理网络状态变化"""
        now = utc_now()
        
        if status == NetworkStatus.DISCONNECTED:
            if self.last_offline_time is None:
                self.last_offline_time = now
            
            # 检查是否应该切换到离线模式
            if self.last_offline_time:
                offline_duration = (now - self.last_offline_time).total_seconds()
                if offline_duration >= self.offline_threshold_seconds:
                    await self._trigger_mode_change("offline")
        
        elif status in [NetworkStatus.CONNECTED, NetworkStatus.WEAK]:
            self.last_online_time = now
            self.last_offline_time = None
            
            # 检查是否应该切换到在线模式
            if status == NetworkStatus.CONNECTED:
                # 连接稳定，立即切换到在线模式
                await self._trigger_mode_change("online")
            else:
                # 弱连接，等待一段时间再决定
                await asyncio.sleep(self.online_threshold_seconds)
                current_status = self.network_monitor.get_current_status()
                if current_status == NetworkStatus.CONNECTED:
                    await self._trigger_mode_change("online")
    
    async def _trigger_mode_change(self, new_mode: str):
        """触发模式变化"""
        if self.current_mode != new_mode:
            old_mode = self.current_mode
            self.current_mode = new_mode
            
            self.logger.info("网络模式切换", old_mode=old_mode, new_mode=new_mode)
            
            # 通知所有回调
            for callback in self.mode_change_callbacks:
                try:
                    await callback(new_mode)
                except Exception as e:
                    self.logger.error("模式切换回调错误", error=str(e))
    
    def add_mode_change_callback(self, callback: Callable[[str], None]):
        """添加模式变化回调"""
        self.mode_change_callbacks.append(callback)
    
    def remove_mode_change_callback(self, callback: Callable[[str], None]):
        """移除模式变化回调"""
        if callback in self.mode_change_callbacks:
            self.mode_change_callbacks.remove(callback)
    
    def get_current_mode(self) -> str:
        """获取当前模式"""
        return self.current_mode
    
    def force_mode(self, mode: str):
        """强制设置模式"""
        self.current_mode = mode
    
    def get_mode_statistics(self) -> Dict[str, Any]:
        """获取模式统计信息"""
        now = utc_now()
        
        return {
            "current_mode": self.current_mode,
            "last_online_time": self.last_online_time.isoformat(),
            "last_offline_time": self.last_offline_time.isoformat() if self.last_offline_time else None,
            "offline_threshold_seconds": self.offline_threshold_seconds,
            "online_threshold_seconds": self.online_threshold_seconds,
            "network_status": self.network_monitor.get_current_status().value,
            "connection_quality": self.network_monitor.get_connection_quality_score()
        }
