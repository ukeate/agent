"""
安全序列化工具
使用HMAC签名保护pickle数据完整性
"""

from __future__ import annotations

import hashlib
import hmac
import pickle as _pickle
from typing import Any, BinaryIO, Optional
from src.core.config import get_settings

MAGIC = b"SPK1"
_SIGNATURE_SIZE = 32
HIGHEST_PROTOCOL = _pickle.HIGHEST_PROTOCOL

def _get_key() -> bytes:
    settings = get_settings()
    if not settings.SECRET_KEY:
        raise ValueError("SECRET_KEY未配置，无法进行安全序列化")
    return settings.SECRET_KEY.encode("utf-8")

def dumps(value: Any, protocol: Optional[int] = None) -> bytes:
    payload = _pickle.dumps(value, protocol=protocol or HIGHEST_PROTOCOL)
    signature = hmac.new(_get_key(), payload, hashlib.sha256).digest()
    return MAGIC + signature + payload

def loads(data: bytes) -> Any:
    if not data or len(data) <= len(MAGIC) + _SIGNATURE_SIZE:
        raise ValueError("序列化数据长度不合法")
    if not data.startswith(MAGIC):
        raise ValueError("不支持的序列化格式")
    signature = data[len(MAGIC):len(MAGIC) + _SIGNATURE_SIZE]
    payload = data[len(MAGIC) + _SIGNATURE_SIZE:]
    expected = hmac.new(_get_key(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(signature, expected):
        raise ValueError("序列化数据签名验证失败")
    return _pickle.loads(payload)

def dump(value: Any, fp: BinaryIO, protocol: Optional[int] = None) -> None:
    fp.write(dumps(value, protocol=protocol))

def load(fp: BinaryIO) -> Any:
    return loads(fp.read())
