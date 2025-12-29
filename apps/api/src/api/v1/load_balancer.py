"""
负载均衡配置与指标API（Redis持久化）
"""

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import Field
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import time
from src.core.redis import get_redis
from src.core.utils.timezone_utils import utc_now
from src.ai.cluster import ClusterStateManager, MetricsCollector
from src.api.v1.service_discovery import get_service_discovery_system
from src.api.base_model import ApiBaseModel

router = APIRouter(prefix="/load-balancer", tags=["load-balancer"])

_LB_CONFIG_KEY = "load_balancer:config"

class LoadBalancerConfig(ApiBaseModel):
    globalStrategy: str = "round_robin"
    healthCheckInterval: int = 30
    connectionTimeout: int = 5
    retryAttempts: int = 3
    circuitBreakerEnabled: bool = True
    circuitBreakerThreshold: int = 5
    circuitBreakerTimeout: int = 30
    stickySession: bool = False
    sessionTimeout: int = 300
    adaptiveLoadBalancing: bool = True
    geographicRouting: bool = False
    capabilityWeighting: bool = True
    responseTimeWeighting: float = 0.4
    connectionWeighting: float = 0.2
    cpuWeighting: float = 0.2
    memoryWeighting: float = 0.2

class StrategyTestRequest(ApiBaseModel):
    strategy: str

async def _require_redis():
    redis = get_redis()
    if not redis:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Redis未初始化，负载均衡不可用",
        )
    return redis

async def get_cluster_manager(request: Request) -> ClusterStateManager:
    return request.app.state.cluster_manager

async def get_metrics_collector(request: Request) -> MetricsCollector:
    return request.app.state.metrics_collector

def _default_config() -> Dict[str, Any]:
    return LoadBalancerConfig().model_dump()

async def _get_config(redis) -> Dict[str, Any]:
    raw = await redis.get(_LB_CONFIG_KEY)
    if not raw:
        config = _default_config()
        await redis.set(_LB_CONFIG_KEY, json.dumps(config, ensure_ascii=False))
        return config
    try:
        return json.loads(raw)
    except Exception:
        config = _default_config()
        await redis.set(_LB_CONFIG_KEY, json.dumps(config, ensure_ascii=False))
        return config

async def _save_config(redis, config: Dict[str, Any]) -> Dict[str, Any]:
    await redis.set(_LB_CONFIG_KEY, json.dumps(config, ensure_ascii=False))
    return config

async def _compute_throughput(metrics_collector: MetricsCollector) -> float:
    data = await metrics_collector.get_cluster_metrics(metric_names=["cluster_total_requests"], duration_seconds=3600)
    series = data.get("cluster_total_requests") or []
    if len(series) < 2:
        return 0.0
    last = series[-1]
    prev = series[-2]
    dt = max(1e-6, float(last.timestamp - prev.timestamp))
    return max(0.0, (float(last.value) - float(prev.value)) / dt)

def _aggregate_performance(series_map: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[int, Dict[str, Any]] = {}
    mapping = {
        "cluster_avg_response_time": "response_time",
        "cluster_error_rate": "error_rate",
        "cluster_total_requests": "requests",
    }
    for metric_name, points in series_map.items():
        key = mapping.get(metric_name)
        if not key:
            continue
        for point in points:
            ts = int(point.timestamp)
            entry = bucket.setdefault(ts, {"timestamp": ts})
            entry[key] = float(point.value)

    if not bucket:
        return []

    ordered = [bucket[k] for k in sorted(bucket.keys())]
    trimmed = ordered[-30:]
    results: List[Dict[str, Any]] = []
    for item in trimmed:
        ts = item["timestamp"]
        results.append({
            "timestamp": datetime.fromtimestamp(ts).isoformat(),
            "response_time": item.get("response_time", 0),
            "error_rate": item.get("error_rate", 0),
            "requests": item.get("requests", 0),
        })
    return results

@router.get("/config")
async def get_load_balancer_config():
    redis = await _require_redis()
    config = await _get_config(redis)
    return {"success": True, "config": config}

@router.post("/config")
async def update_load_balancer_config(config: LoadBalancerConfig):
    redis = await _require_redis()
    saved = await _save_config(redis, config.model_dump())
    return {"success": True, "config": saved}

@router.get("/metrics")
async def get_load_balancer_metrics(
    cluster_manager: ClusterStateManager = Depends(get_cluster_manager),
    metrics_collector: MetricsCollector = Depends(get_metrics_collector),
):
    stats = await cluster_manager.get_cluster_stats()
    usage = stats.get("resource_usage") or {}
    total_requests = int(usage.get("total_requests") or 0)
    error_rate = float(usage.get("error_rate") or 0)
    success_rate = (1 - error_rate) * 100 if total_requests > 0 else 0
    avg_response = float(usage.get("avg_response_time") or 0)
    active_connections = int(usage.get("active_tasks") or 0)
    throughput = await _compute_throughput(metrics_collector)

    redis = await _require_redis()
    config = await _get_config(redis)
    strategy = config.get("globalStrategy", "round_robin")

    metrics = [
        {
            "strategy": strategy,
            "requestsHandled": total_requests,
            "averageResponseTime": avg_response,
            "successRate": round(success_rate, 2),
            "errorRate": round(error_rate * 100, 2),
            "activeConnections": active_connections,
            "throughputPerSecond": round(throughput, 2),
        }
    ]

    series = await metrics_collector.get_cluster_metrics(
        metric_names=["cluster_avg_response_time", "cluster_error_rate", "cluster_total_requests"],
        duration_seconds=3600,
    )
    performance = _aggregate_performance(series)

    return {"success": True, "metrics": metrics, "performance": performance}

@router.post("/strategy/test")
async def test_strategy(request: StrategyTestRequest):
    system = await get_service_discovery_system()
    agent = await system.discover_and_select_agent(capability="text_processing", strategy=request.strategy)
    stats = await system.get_system_stats()
    registry_stats = stats.get("registry", {})
    total = int(registry_stats.get("total_requests", 0) or 0)
    error_count = int(registry_stats.get("error_count", 0) or 0)
    avg_response = float(registry_stats.get("avg_response_time", 0) or 0)
    success_rate = ((total - error_count) / total * 100) if total > 0 else 0
    throughput = float(registry_stats.get("requests_per_minute", 0) or 0) / 60

    return {
        "strategy": request.strategy,
        "timestamp": utc_now().isoformat(),
        "averageResponseTime": round(avg_response, 2),
        "successRate": round(success_rate, 2),
        "throughput": round(throughput, 2),
        "requests": total,
        "selectedAgent": agent.agent_id if agent else "",
    }
