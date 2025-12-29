"""
监控模块初始化
"""

from .vector_db_metrics import (
    VectorMetricsCollector,
    VectorQueryMetrics,
    VectorIndexMetrics,
    VectorSystemMetrics,
    get_metrics_collector
)
from .service import monitoring_service

from src.core.logging import get_logger
logger = get_logger(__name__)

class _Monitor:
    def log_info(self, message: str, **kwargs):
        logger.info(message, **kwargs)

    def log_warning(self, message: str, **kwargs):
        logger.warning(message, **kwargs)

    def log_error(self, message: str, **kwargs):
        logger.error(message, **kwargs)

monitor = _Monitor()

def get_monitoring_service():
    """获取监控服务实例"""
    return monitoring_service

__all__ = [
    "VectorMetricsCollector",
    "VectorQueryMetrics", 
    "VectorIndexMetrics",
    "VectorSystemMetrics",
    "get_metrics_collector",
    "monitor",
    "monitoring_service",
    "get_monitoring_service"
]
