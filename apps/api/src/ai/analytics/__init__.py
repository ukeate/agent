"""
智能用户行为分析系统

这是一个完整的用户行为分析系统，提供数据收集、模式识别、异常检测、
趋势分析、实时监控和性能优化等功能。

主要组件：
- 事件收集和存储
- 行为模式识别
- 异常检测引擎
- 趋势分析和预测
- 实时数据推送
- 性能监控和优化

使用示例：
    from ai.analytics import AnalyticsSystem
    
    # 初始化系统
    analytics = AnalyticsSystem()
    await analytics.start()
    
    # 收集事件
    event = BehaviorEvent(...)
    await analytics.collect_event(event)
    
    # 分析行为
    results = await analytics.analyze_behavior(user_id="user-123")
    
    # 停止系统
    await analytics.stop()
"""

from .models import (
    BehaviorEvent, UserSession, BehaviorPattern, AnomalyDetection,
    EventFilter, SessionFilter, AnalyticsRequest, CrossAnalysisResult
)
from .behavior.event_collector import EventCollector
from .behavior.session_manager import SessionManager  
from .behavior.pattern_recognition import PatternRecognitionEngine
from .behavior.anomaly_detection import AnomalyDetectionEngine
from .behavior.cross_analysis import MultiDimensionInsightEngine
from .behavior.report_generator import ReportGenerator
from .storage.event_store import EventStore
from .realtime.websocket_manager import realtime_manager
from .monitoring.performance_monitor import performance_monitor
from .config.performance import performance_config, performance_optimizer
import asyncio
from typing import Dict, Any, List, Optional

from src.core.logging import get_logger
logger = get_logger(__name__)

class AnalyticsSystem:
    """智能行为分析系统主类"""
    
    def __init__(self):
        # 核心组件
        self.event_collector = EventCollector()
        self.session_manager = SessionManager()
        self.pattern_engine = PatternRecognitionEngine()
        self.anomaly_engine = AnomalyDetectionEngine()
        self.insight_engine = MultiDimensionInsightEngine()
        self.report_generator = ReportGenerator()
        self.event_store = EventStore()
        
        # 实时和监控
        self.realtime_manager = realtime_manager
        self.performance_monitor = performance_monitor
        
        # 状态标记
        self.running = False
        
        logger.info("智能行为分析系统已初始化")
    
    async def start(self):
        """启动分析系统"""
        if self.running:
            logger.warning("系统已经在运行")
            return
        
        try:
            # 启动核心组件
            await self.event_collector.start()
            await self.session_manager.start()
            
            # 启动实时管理器
            await self.realtime_manager.start()
            
            # 启动性能监控
            await self.performance_monitor.start()
            
            # 初始化存储
            await self.event_store.initialize()
            
            self.running = True
            logger.info("智能行为分析系统启动成功")
            
        except Exception as e:
            logger.error(f"系统启动失败: {e}")
            await self.stop()
            raise
    
    async def stop(self):
        """停止分析系统"""
        if not self.running:
            return
        
        try:
            # 停止监控
            await self.performance_monitor.stop()
            
            # 停止实时管理器
            await self.realtime_manager.stop()
            
            # 停止核心组件
            await self.session_manager.stop()
            await self.event_collector.stop()
            
            self.running = False
            logger.info("智能行为分析系统已停止")
            
        except Exception as e:
            logger.error(f"系统停止时出错: {e}")
    
    async def collect_event(self, event: BehaviorEvent) -> bool:
        """收集用户行为事件"""
        try:
            await self.event_collector.collect_event(event)
            
            # 实时处理
            await self.realtime_manager.handle_new_event(event)
            
            return True
            
        except Exception as e:
            logger.error(f"事件收集失败: {e}")
            return False
    
    async def analyze_behavior(self, user_id: Optional[str] = None, 
                             session_id: Optional[str] = None,
                             analysis_types: List[str] = None) -> Dict[str, Any]:
        """分析用户行为"""
        if analysis_types is None:
            analysis_types = ["patterns", "anomalies", "insights"]
        
        try:
            # 获取相关事件
            filter_params = EventFilter(user_id=user_id, session_id=session_id)
            events = await self.event_store.get_events(filter_params)
            
            results = {}
            
            # 模式识别
            if "patterns" in analysis_types:
                results["patterns"] = await self.pattern_engine.analyze_patterns(events)
            
            # 异常检测
            if "anomalies" in analysis_types:
                results["anomalies"] = await self.anomaly_engine.detect_anomalies(events)
            
            # 多维度洞察
            if "insights" in analysis_types:
                results["insights"] = await self.insight_engine.generate_comprehensive_insights(events)
            
            return {
                "status": "success",
                "event_count": len(events),
                "analysis_types": analysis_types,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"行为分析失败: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def generate_report(self, report_type: str = "comprehensive") -> Dict[str, Any]:
        """生成分析报告"""
        try:
            # 获取所有事件用于报告
            events = await self.event_store.get_events(EventFilter(limit=100000))
            
            if report_type == "comprehensive":
                report = await self.report_generator.generate_comprehensive_report(events)
            elif report_type == "summary":
                report = await self.report_generator.generate_summary_report(events)
            else:
                report = await self.report_generator.generate_custom_report(events, {})
            
            return report
            
        except Exception as e:
            logger.error(f"报告生成失败: {e}")
            return {"error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "running": self.running,
            "components": {
                "event_collector": self.event_collector.is_running() if hasattr(self.event_collector, 'is_running') else True,
                "session_manager": self.session_manager.is_running() if hasattr(self.session_manager, 'is_running') else True,
                "realtime_manager": self.realtime_manager.running,
                "performance_monitor": self.performance_monitor.collector.running
            },
            "performance": self.performance_monitor.get_current_metrics() if self.running else {},
            "websocket_stats": self.realtime_manager.websocket_manager.get_stats() if self.running else {}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            status = self.get_system_status()
            
            # 检查关键组件是否正常
            all_healthy = all(status["components"].values())
            
            return {
                "status": "healthy" if all_healthy else "degraded",
                "timestamp": asyncio.get_running_loop().time(),
                "details": status
            }
            
        except Exception as e:
            logger.error(f"健康检查失败: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": asyncio.get_running_loop().time()
            }

# 全局系统实例
analytics_system = AnalyticsSystem()

# 便捷函数
async def initialize_analytics():
    """初始化分析系统"""
    await analytics_system.start()

async def shutdown_analytics():
    """关闭分析系统"""
    await analytics_system.stop()

async def collect_user_event(event: BehaviorEvent) -> bool:
    """收集用户事件的便捷函数"""
    return await analytics_system.collect_event(event)

async def analyze_user_behavior(user_id: str, analysis_types: List[str] = None) -> Dict[str, Any]:
    """分析用户行为的便捷函数"""
    return await analytics_system.analyze_behavior(user_id=user_id, analysis_types=analysis_types)

# 导出主要类和函数
__all__ = [
    # 数据模型
    'BehaviorEvent', 'UserSession', 'BehaviorPattern', 'AnomalyDetection',
    'EventFilter', 'SessionFilter', 'AnalyticsRequest', 'CrossAnalysisResult',
    
    # 核心组件
    'EventCollector', 'SessionManager', 'PatternRecognitionEngine',
    'AnomalyDetectionEngine', 'MultiDimensionInsightEngine', 'ReportGenerator',
    'EventStore',
    
    # 系统管理
    'AnalyticsSystem', 'analytics_system',
    
    # 全局管理器
    'realtime_manager', 'performance_monitor', 'performance_config',
    
    # 便捷函数
    'initialize_analytics', 'shutdown_analytics',
    'collect_user_event', 'analyze_user_behavior'
]
