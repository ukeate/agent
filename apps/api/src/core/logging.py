"""
日志配置
"""

from typing import Any

import structlog
from structlog.stdlib import LoggerFactory


def setup_logging() -> None:
    """设置结构化日志"""

    # 配置structlog
    structlog.configure(
        processors=[
            # 添加日志级别
            structlog.stdlib.add_log_level,
            # 添加时间戳
            structlog.processors.TimeStamper(fmt="iso"),
            # 添加调用者信息
            structlog.processors.add_log_level,
            # JSON格式化
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> Any:
    """获取结构化日志器"""
    return structlog.get_logger(name)
