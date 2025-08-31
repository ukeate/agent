"""
统一的时区处理工具模块
用于解决项目中时区处理不一致的问题
"""

from datetime import datetime, timezone
from typing import Optional


def utc_now() -> datetime:
    """
    获取当前UTC时间
    统一使用 datetime.now(timezone.utc) 替代 datetime.utcnow()
    """
    return datetime.now(timezone.utc)


def to_utc(dt: datetime) -> datetime:
    """
    将datetime对象转换为UTC时区
    """
    if dt.tzinfo is None:
        # 如果没有时区信息，假设为本地时区
        return dt.replace(tzinfo=timezone.utc)
    else:
        # 转换到UTC时区
        return dt.astimezone(timezone.utc)


def from_timestamp(timestamp: float) -> datetime:
    """
    从时间戳创建UTC时间的datetime对象
    """
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def to_timestamp(dt: datetime) -> float:
    """
    将datetime对象转换为时间戳
    """
    return dt.timestamp()


def utc_factory() -> datetime:
    """
    工厂函数，用于dataclass的default_factory
    """
    return utc_now()


def parse_iso_string(iso_string: str) -> Optional[datetime]:
    """
    解析ISO格式的时间字符串为UTC datetime对象
    """
    try:
        dt = datetime.fromisoformat(iso_string.replace('Z', '+00:00'))
        return to_utc(dt)
    except (ValueError, AttributeError):
        return None


def format_iso_string(dt: datetime) -> str:
    """
    将datetime对象格式化为ISO字符串
    """
    return to_utc(dt).isoformat()