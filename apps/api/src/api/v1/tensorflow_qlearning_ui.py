"""
TensorFlow Q-Learning UI接口

说明：此模块仅作为前端展示接口，不依赖 TensorFlow。
"""

import json
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from src.core.redis import get_redis

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/tensorflow-qlearning", tags=["TensorFlow Q-Learning"])

_SESSION_PREFIX = "tensorflow_qlearning:sessions:"

async def _list_sessions() -> list[dict]:
    redis_client = get_redis()
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis未初始化，无法读取会话数据")

    sessions: list[dict] = []
    async for key in redis_client.scan_iter(match=f"{_SESSION_PREFIX}*"):
        raw = await redis_client.get(key)
        if raw is None:
            continue
        sessions.append(json.loads(raw))
    return sessions

@router.get("/overview")
async def get_overview():
    """
    返回当前 Q-Learning 会话概览。
    不依赖 TensorFlow，如无会话则返回空数据。
    """
    sessions = await _list_sessions()

    perf_values = []
    total_episodes = 0
    for s in sessions:
        try:
            total_episodes += int(s.get("total_episodes", 0) or 0)
        except (TypeError, ValueError):
            logger.debug("总回合数转换失败", value=s.get("total_episodes"), exc_info=True)
        try:
            avg_perf = s.get("average_performance")
            if avg_perf is not None:
                perf_values.append(float(avg_perf))
        except (TypeError, ValueError):
            logger.debug("平均性能转换失败", value=s.get("average_performance"), exc_info=True)

    summary = {
        "running": sum(1 for s in sessions if s.get("is_training")),
        "training": sum(1 for s in sessions if s.get("is_training")),
        "average_performance": (sum(perf_values) / len(perf_values)) if perf_values else 0.0,
        "total_episodes": total_episodes,
        "tensorflow_available": False,
    }

    return {
        "summary": summary,
        "models": sessions,
        "trend": [],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
