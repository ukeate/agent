"""
日志配置
"""

import logging
import os
import sys
from typing import Any

import structlog
from structlog.stdlib import LoggerFactory

_LOGGING_CONFIGURED = False

def setup_logging() -> None:
    """设置结构化日志"""
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.getLevelName(level_name),
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _LOGGING_CONFIGURED = True

def get_logger(name: str = __name__) -> Any:
    """获取结构化日志器"""
    if not _LOGGING_CONFIGURED:
        setup_logging()
    return structlog.get_logger(name)
