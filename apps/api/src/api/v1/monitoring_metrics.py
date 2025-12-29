"""
监控指标汇总API（面向仪表盘）
"""

from fastapi import APIRouter, Query
from typing import Any, Dict, List, Optional
import json
import time
from collections import deque
import psutil
from src.core.monitoring import get_monitoring_service
from src.core.redis import get_redis
from src.core.utils.timezone_utils import utc_now

router = APIRouter(tags=["monitoring"])

_NET_STATE_KEY = "monitoring:net:latest"
_PERF_HISTORY_KEY = "monitoring:metrics:history"
_PERF_HISTORY_MAX = 300

_local_net_state = {"ts": 0.0, "in": 0.0, "out": 0.0}
_local_perf_history: deque = deque(maxlen=_PERF_HISTORY_MAX)

async def _get_net_throughput_mbps() -> float:
    net = psutil.net_io_counters()
    now = time.time()
    redis = get_redis()
    state: Optional[Dict[str, float]] = None
    if redis:
        raw = await redis.get(_NET_STATE_KEY)
        if raw:
            try:
                state = json.loads(raw)
            except Exception:
                state = None

    if not state:
        state = _local_net_state.copy()

    dt = now - float(state.get("ts") or 0)
    last_in = float(state.get("in") or 0)
    last_out = float(state.get("out") or 0)
    in_rate = (max(0.0, net.bytes_recv - last_in) / dt / (1024 * 1024)) if dt > 0 else 0.0
    out_rate = (max(0.0, net.bytes_sent - last_out) / dt / (1024 * 1024)) if dt > 0 else 0.0
    throughput = in_rate + out_rate

    new_state = {"ts": now, "in": float(net.bytes_recv), "out": float(net.bytes_sent)}
    if redis:
        await redis.set(_NET_STATE_KEY, json.dumps(new_state))
    else:
        _local_net_state.update(new_state)

    return throughput

def _collect_perf_sample() -> Dict[str, Any]:
    return {
        "timestamp": utc_now().isoformat(),
        "cpu": psutil.cpu_percent(interval=0.1),
        "memory": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
    }

async def _append_perf_sample(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    redis = get_redis()
    if redis:
        await redis.lpush(_PERF_HISTORY_KEY, json.dumps(sample, ensure_ascii=False))
        await redis.ltrim(_PERF_HISTORY_KEY, 0, _PERF_HISTORY_MAX - 1)
        raw_list = await redis.lrange(_PERF_HISTORY_KEY, 0, _PERF_HISTORY_MAX - 1)
        items = []
        for raw in raw_list:
            try:
                items.append(json.loads(raw))
            except Exception:
                continue
        items.reverse()
        return items

    _local_perf_history.append(sample)
    return list(_local_perf_history)

@router.get("/metrics")
async def get_metrics_summary() -> Dict[str, Any]:
    monitoring_service = get_monitoring_service()
    perf_stats = await monitoring_service.performance_monitor.get_stats()

    connections = None
    try:
        connections = len(psutil.net_connections())
    except Exception:
        connections = None

    throughput = await _get_net_throughput_mbps()

    return {
        "cpu": {"usage": psutil.cpu_percent(interval=0.1)},
        "memory": {"usage": psutil.virtual_memory().percent},
        "disk": {"usage": psutil.disk_usage("/").percent},
        "network": {"throughput": throughput},
        "connections": connections or 0,
        "requests": {"rate": float(perf_stats.get("requests_per_minute") or 0) / 60},
        "timestamp": utc_now().isoformat(),
    }

@router.get("/metrics/performance")
async def get_metrics_performance(limit: int = Query(60, ge=1, le=300)) -> List[Dict[str, Any]]:
    sample = _collect_perf_sample()
    history = await _append_perf_sample(sample)
    if len(history) > limit:
        return history[-limit:]
    return history

@router.get("/metrics/request-distribution")
async def get_request_distribution(limit: int = Query(10, ge=1, le=50)) -> Dict[str, Any]:
    monitoring_service = get_monitoring_service()
    requests = list(monitoring_service.performance_monitor.request_times)
    counts: Dict[str, int] = {}

    for req in requests:
        endpoint = str(req.get("endpoint") or "").strip("/")
        parts = endpoint.split("/") if endpoint else []
        if len(parts) >= 2 and parts[0] == "api" and parts[1] == "v1":
            key = parts[2] if len(parts) > 2 else "root"
        else:
            key = parts[0] if parts else "root"
        counts[key] = counts.get(key, 0) + 1

    items = [
        {"name": name, "value": value}
        for name, value in sorted(counts.items(), key=lambda item: item[1], reverse=True)
    ]
    return {"distribution": items[:limit]}

@router.get("/alerts")
async def get_alerts() -> Dict[str, Any]:
    monitoring_service = get_monitoring_service()
    alerts = await monitoring_service.alert_manager.get_active_alerts()
    mapped = []
    for alert in alerts:
        ts = alert.get("timestamp") or utc_now().isoformat()
        name = alert.get("name") or "system"
        mapped.append(
            {
                "id": f"{name}:{ts}",
                "level": alert.get("severity") or "warning",
                "message": alert.get("message") or "",
                "timestamp": ts,
                "service": name,
                "resolved": False,
            }
        )
    return {"alerts": mapped}
