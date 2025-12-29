"""
用户行为分析API端点

基于事件存储提供事件采集、会话聚合、简单模式/异常检测、报告生成与导出能力。
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response, StreamingResponse
from pydantic import Field
from src.ai.autogen.events import Event, EventPriority, EventType
from src.ai.autogen.event_store import EventStore
from src.core.utils.timezone_utils import utc_now
from src.api.base_model import ApiBaseModel

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/analytics", tags=["用户行为分析"])

event_store: Optional[EventStore] = None
export_tasks: List[Dict[str, Any]] = []
reports_data: Dict[str, Dict[str, Any]] = {}

def init_services(store: EventStore) -> None:
    global event_store
    event_store = store

def _require_event_store() -> EventStore:
    if event_store is None:
        raise HTTPException(status_code=503, detail="事件存储未初始化")
    return event_store

class BehaviorEvent(ApiBaseModel):
    """用户行为事件"""

    event_id: str
    user_id: str
    session_id: Optional[str] = None
    event_type: str
    timestamp: datetime
    properties: Optional[Dict[str, Any]] = None
    context: Optional[Dict[str, Any]] = None
    duration: Optional[int] = None

class EventSubmissionRequest(ApiBaseModel):
    """事件提交请求"""

    events: List[BehaviorEvent]
    batch_id: Optional[str] = None

class AnalysisRequest(ApiBaseModel):
    """分析请求"""

    user_id: Optional[str] = None
    session_id: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    event_types: Optional[List[str]] = None
    analysis_types: List[str] = ["patterns", "anomalies", "insights"]

class ReportRequest(ApiBaseModel):
    """报告生成请求"""

    report_type: str = "comprehensive"
    format: str = "json"
    filters: Optional[Dict[str, Any]] = None
    include_visualizations: bool = True

class RealtimeMessage(ApiBaseModel):
    type: str
    data: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: utc_now())

@dataclass
class _WsConn:
    websocket: WebSocket
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    subscriptions: set[str] = field(default_factory=set)
    connected_at: datetime = field(default_factory=utc_now)
    last_activity: datetime = field(default_factory=utc_now)
    messages_sent: int = 0
    messages_received: int = 0
    client_host: Optional[str] = None
    client_port: Optional[int] = None
    user_agent: Optional[str] = None

class _WsManager:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._connections: Dict[str, _WsConn] = {}
        self._sent = 0
        self._failed = 0

    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        await websocket.accept()
        client = websocket.client
        user_agent = websocket.headers.get("user-agent")
        async with self._lock:
            self._connections[connection_id] = _WsConn(
                websocket=websocket,
                user_id=user_id,
                session_id=session_id,
                connected_at=utc_now(),
                last_activity=utc_now(),
                client_host=client.host if client else None,
                client_port=client.port if client else None,
                user_agent=user_agent,
            )

    async def disconnect(self, connection_id: str) -> None:
        async with self._lock:
            conn = self._connections.pop(connection_id, None)
        if not conn:
            return
        try:
            await conn.websocket.close()
        except Exception:
            logger.exception("关闭WebSocket连接失败", exc_info=True)

    async def subscribe(self, connection_id: str, subscription_type: str) -> None:
        async with self._lock:
            conn = self._connections.get(connection_id)
            if not conn:
                raise ValueError("连接不存在")
            conn.subscriptions.add(subscription_type)

    async def unsubscribe(self, connection_id: str, subscription_type: str) -> None:
        async with self._lock:
            conn = self._connections.get(connection_id)
            if not conn:
                return
            conn.subscriptions.discard(subscription_type)

    async def broadcast_message(self, message: RealtimeMessage) -> None:
        data = message.model_dump(mode="json")
        async with self._lock:
            conns = list(self._connections.items())

        tasks = []
        for connection_id, conn in conns:
            if message.user_id and conn.user_id != message.user_id:
                continue
            if message.session_id and conn.session_id != message.session_id:
                continue
            if conn.subscriptions and message.type not in conn.subscriptions:
                continue
            tasks.append(self._send(connection_id, conn.websocket, data))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _send(self, connection_id: str, websocket: WebSocket, data: Dict[str, Any]) -> None:
        try:
            await websocket.send_text(json.dumps(data, ensure_ascii=False, default=str))
            self._sent += 1
            async with self._lock:
                conn = self._connections.get(connection_id)
                if conn:
                    conn.messages_sent += 1
                    conn.last_activity = utc_now()
        except Exception:
            self._failed += 1
            await self.disconnect(connection_id)

    async def record_receive(self, connection_id: str) -> None:
        async with self._lock:
            conn = self._connections.get(connection_id)
            if conn:
                conn.messages_received += 1
                conn.last_activity = utc_now()

    async def list_connection_details(self) -> List[Dict[str, Any]]:
        async with self._lock:
            items = list(self._connections.items())
        details = []
        for idx, (connection_id, conn) in enumerate(sorted(items, key=lambda x: x[0])):
            details.append({
                "index": idx,
                "state": "connected",
                "connected_at": conn.connected_at.isoformat(),
                "last_activity": conn.last_activity.isoformat(),
                "messages_sent": conn.messages_sent,
                "messages_received": conn.messages_received,
                "client_info": {
                    "user_agent": conn.user_agent,
                    "ip_address": conn.client_host,
                    "session_id": conn.session_id,
                },
                "connection_id": connection_id,
            })
        return details

    async def resolve_connection_id(self, index: int) -> Optional[str]:
        async with self._lock:
            ids = sorted(self._connections.keys())
        if index < 0 or index >= len(ids):
            return None
        return ids[index]

    async def send_to_connection(self, connection_id: str, payload: Dict[str, Any]) -> None:
        async with self._lock:
            conn = self._connections.get(connection_id)
        if not conn:
            raise ValueError("连接不存在")
        await self._send(connection_id, conn.websocket, payload)

    async def broadcast_payload(self, payload: Dict[str, Any]) -> int:
        async with self._lock:
            conns = list(self._connections.items())
        tasks = [self._send(connection_id, conn.websocket, payload) for connection_id, conn in conns]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(conns)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "active_connections": len(self._connections),
            "messages_sent": self._sent,
            "messages_failed": self._failed,
        }

ws_manager = _WsManager()

def _to_event(behavior: BehaviorEvent) -> Event:
    event_id: Optional[str] = None
    try:
        uuid.UUID(behavior.event_id)
        event_id = behavior.event_id
    except Exception:
        event_id = str(uuid.uuid4())

    session_uuid: Optional[str] = None
    if behavior.session_id:
        try:
            uuid.UUID(behavior.session_id)
            session_uuid = behavior.session_id
        except Exception:
            session_uuid = None

    return Event(
        id=event_id,
        type=EventType.MESSAGE_SENT,
        timestamp=behavior.timestamp,
        source=behavior.user_id,
        session_id=session_uuid,
        priority=EventPriority.NORMAL,
        data={
            "client_event_id": behavior.event_id,
            "client_session_id": behavior.session_id,
            "event_type": behavior.event_type,
            "properties": behavior.properties or {},
            "context": behavior.context or {},
            "duration": behavior.duration,
        },
    )

async def _filter_events(
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    event_type: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    store = _require_event_store()
    start = start_time or utc_now() - timedelta(days=7)
    end = end_time or utc_now()

    filters: Dict[str, Any] = {}
    if user_id:
        filters["source"] = user_id
    session_uuid: Optional[str] = None
    if session_id:
        try:
            uuid.UUID(session_id)
            session_uuid = session_id
        except Exception:
            session_uuid = None
    if session_uuid:
        filters["session_id"] = session_uuid

    events = await store.replay_events(
        start_time=start,
        end_time=end,
        event_types=None,
        filters=filters or None,
    )

    results: List[Dict[str, Any]] = []
    for ev in events:
        data = getattr(ev, "data", None) or {}
        current_session_id = data.get("client_session_id") or ev.session_id
        if session_id and current_session_id != session_id:
            continue
        current_type = data.get("event_type") or (
            ev.type.value if hasattr(ev.type, "value") else str(ev.type)
        )
        if event_type and current_type != event_type:
            continue

        results.append(
            {
                "id": ev.id,
                "user_id": ev.source,
                "session_id": current_session_id,
                "event_type": current_type,
                "timestamp": ev.timestamp,
                "properties": data.get("properties") or {},
                "context": data.get("context") or {},
                "client_event_id": data.get("client_event_id"),
            }
        )

    return results

@router.post("/events", summary="提交用户行为事件")
async def submit_events(request: EventSubmissionRequest):
    """批量提交用户行为事件"""
    try:
        store = _require_event_store()
        if not request.events:
            raise HTTPException(status_code=400, detail="事件列表不能为空")
        if len(request.events) > 1000:
            raise HTTPException(status_code=400, detail="单次提交事件数量不能超过1000")

        for behavior_event in request.events:
            await store.append_event(_to_event(behavior_event))

        return {
            "status": "accepted",
            "event_count": len(request.events),
            "batch_id": request.batch_id or str(uuid.uuid4()),
            "message": "事件已接收",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("事件处理失败", error=str(e))
        raise HTTPException(status_code=500, detail=f"事件处理失败: {str(e)}")

@router.get("/events", summary="查询行为事件")
async def get_events(
    user_id: Optional[str] = Query(None, description="用户ID"),
    session_id: Optional[str] = Query(None, description="会话ID"),
    event_type: Optional[str] = Query(None, description="事件类型"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    limit: int = Query(50, ge=1, le=1000, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
):
    """查询行为事件数据"""
    events = await _filter_events(user_id, session_id, event_type, start_time, end_time)
    total = len(events)
    result_events = events[offset : offset + limit]
    return {
        "events": result_events,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total,
    }

@router.get("/sessions", summary="查询用户会话")
async def get_sessions(
    user_id: Optional[str] = Query(None, description="用户ID"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    status: Optional[str] = Query(None, description="会话状态: active, inactive, expired"),
    min_duration: Optional[int] = Query(None, description="最小持续时间(秒)"),
    max_duration: Optional[int] = Query(None, description="最大持续时间(秒)"),
    min_events: Optional[int] = Query(None, description="最小事件数"),
    max_events: Optional[int] = Query(None, description="最大事件数"),
    limit: int = Query(50, ge=1, le=1000, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
):
    """查询用户会话数据"""
    events = await _filter_events(user_id=user_id, start_time=start_time, end_time=end_time)
    sessions: Dict[str, Dict[str, Any]] = {}

    for ev in events:
        sid = ev.get("session_id") or "unknown"
        ts: datetime = ev.get("timestamp") or utc_now()
        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "user_id": ev.get("user_id"),
                "start": ts,
                "end": ts,
                "events": 0,
                "event_types": set(),
            }
        sessions[sid]["events"] += 1
        sessions[sid]["event_types"].add(ev.get("event_type"))
        if ts < sessions[sid]["start"]:
            sessions[sid]["start"] = ts
        if ts > sessions[sid]["end"]:
            sessions[sid]["end"] = ts

    session_list: List[Dict[str, Any]] = []
    now = utc_now()
    for s in sessions.values():
        duration = (s["end"] - s["start"]).total_seconds()
        if max_duration is not None and duration > max_duration:
            continue
        if min_duration is not None and duration < min_duration:
            continue
        if min_events is not None and s["events"] < min_events:
            continue
        if max_events is not None and s["events"] > max_events:
            continue

        inactive_seconds = (now - s["end"]).total_seconds()
        s_status = "expired"
        if inactive_seconds <= 300:
            s_status = "active"
        elif inactive_seconds <= 3600:
            s_status = "inactive"
        if status and s_status != status:
            continue

        event_types = sorted(t for t in s.get("event_types") if t)
        session_list.append(
            {
                "session_id": s["session_id"],
                "user_id": s["user_id"],
                "start_time": s["start"].isoformat(),
                "end_time": s["end"].isoformat(),
                "duration_seconds": duration,
                "event_count": s["events"],
                "unique_event_types": len(event_types),
                "last_activity": s["end"].isoformat(),
                "status": s_status,
                "metadata": {"event_types": event_types},
            }
        )

    total = len(session_list)
    session_list = session_list[offset : offset + limit]
    return {
        "sessions": session_list,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total,
    }

@router.get("/sessions/stats", summary="会话统计")
async def get_session_stats(
    user_id: Optional[str] = Query(None, description="用户ID"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
):
    events = await _filter_events(user_id=user_id, start_time=start_time, end_time=end_time)
    sessions: Dict[str, Dict[str, Any]] = {}
    for ev in events:
        sid = ev.get("session_id") or "unknown"
        ts: datetime = ev.get("timestamp") or utc_now()
        if sid not in sessions:
            sessions[sid] = {"user_id": ev.get("user_id"), "start": ts, "end": ts, "events": 0}
        sessions[sid]["events"] += 1
        if ts < sessions[sid]["start"]:
            sessions[sid]["start"] = ts
        if ts > sessions[sid]["end"]:
            sessions[sid]["end"] = ts

    now = utc_now()
    durations = []
    total_events = 0
    active_sessions = 0
    unique_users = set()
    for s in sessions.values():
        durations.append((s["end"] - s["start"]).total_seconds())
        total_events += int(s["events"])
        if s.get("user_id"):
            unique_users.add(s["user_id"])
        if (now - s["end"]).total_seconds() <= 300:
            active_sessions += 1

    avg_duration = sum(durations) / len(durations) if durations else 0.0
    return {
        "active_sessions": active_sessions,
        "avg_duration": avg_duration,
        "total_events": total_events,
        "unique_users": len(unique_users),
    }

@router.post("/analyze", summary="执行行为分析")
async def analyze_behavior(request: AnalysisRequest):
    """执行用户行为分析"""
    events = await _filter_events(
        user_id=request.user_id,
        session_id=request.session_id,
        start_time=request.start_time,
        end_time=request.end_time,
    )
    if request.event_types:
        events = [e for e in events if e.get("event_type") in set(request.event_types)]

    if not events:
        return {"status": "no_data", "message": "未找到匹配的事件数据"}

    results: Dict[str, Any] = {}
    sorted_freq: List[tuple] = []

    if "patterns" in request.analysis_types:
        freq: Dict[str, int] = {}
        for ev in events:
            et = ev.get("event_type")
            freq[et] = freq.get(et, 0) + 1
        sorted_freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        results["patterns"] = [{"event_type": et, "count": cnt} for et, cnt in sorted_freq]

    if "anomalies" in request.analysis_types:
        total = len(events)
        anomalies = []
        for et, cnt in sorted_freq:
            ratio = cnt / total
            if ratio > 0.8:
                anomalies.append({"event_type": et, "ratio": ratio, "severity": "high"})
        results["anomalies"] = anomalies

    if "insights" in request.analysis_types:
        results["insights"] = {
            "total_events": len(events),
            "unique_event_types": len({ev.get("event_type") for ev in events}),
            "active_sessions": len({ev.get("session_id") for ev in events if ev.get("session_id")}),
            "active_users": len({ev.get("user_id") for ev in events if ev.get("user_id")}),
        }

    return {
        "status": "success",
        "event_count": len(events),
        "analysis_timestamp": utc_now().isoformat(),
        "results": results,
    }

@router.get("/patterns", summary="获取行为模式")
async def get_behavior_patterns(
    user_id: Optional[str] = Query(None, description="用户ID"),
    min_support: float = Query(0.1, ge=0.01, le=1.0, description="最小支持度"),
    limit: int = Query(50, ge=1, le=200, description="返回数量限制"),
):
    events = await _filter_events(user_id=user_id)
    freq: Dict[str, int] = {}
    for ev in events:
        et = ev.get("event_type")
        freq[et] = freq.get(et, 0) + 1
    patterns = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:limit]
    return {
        "patterns": [{"event_type": et, "count": cnt} for et, cnt in patterns],
        "total_patterns": len(patterns),
        "analysis_parameters": {"min_support": min_support, "event_count": len(events)},
    }

@router.get("/anomalies", summary="获取异常检测结果")
async def get_anomalies(
    user_id: Optional[str] = Query(None, description="用户ID"),
    severity: Optional[str] = Query(None, description="严重程度"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    limit: int = Query(100, ge=1, le=500, description="返回数量限制"),
):
    events = await _filter_events(user_id=user_id, start_time=start_time, end_time=end_time)
    freq: Dict[str, int] = {}
    for ev in events:
        et = ev.get("event_type")
        freq[et] = freq.get(et, 0) + 1
    total = len(events) or 1
    anomalies = []
    for et, cnt in freq.items():
        ratio = cnt / total
        if ratio > 0.8 or ratio < 0.05:
            anomalies.append(
                {"event_type": et, "ratio": ratio, "severity": "high" if ratio > 0.8 else "low"}
            )
    if severity:
        anomalies = [a for a in anomalies if a["severity"] == severity]
    anomalies = anomalies[:limit]
    return {"anomalies": anomalies, "total_anomalies": len(anomalies), "detection_methods": ["frequency_deviation"]}

def _parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None
    return None

@router.post("/reports/generate", summary="生成分析报告")
async def generate_report(request: ReportRequest):
    valid_types = ["comprehensive", "summary", "custom"]
    if request.report_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"无效的报告类型，支持: {valid_types}")

    valid_formats = ["json", "html"]
    if request.format not in valid_formats:
        raise HTTPException(status_code=400, detail=f"无效的格式，支持: {valid_formats}")

    filters = request.filters or {}
    events = await _filter_events(
        user_id=filters.get("user_id"),
        session_id=filters.get("session_id"),
        event_type=filters.get("event_type"),
        start_time=_parse_datetime(filters.get("start_time")),
        end_time=_parse_datetime(filters.get("end_time")),
    )
    if not events:
        raise HTTPException(status_code=404, detail="没有找到匹配的数据")

    freq: Dict[str, int] = {}
    for ev in events:
        et = ev.get("event_type")
        freq[et] = freq.get(et, 0) + 1

    report_id = f"report_{utc_now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    reports_data[report_id] = {
        "report_id": report_id,
        "status": "completed",
        "format": request.format,
        "report_type": request.report_type,
        "filters": filters,
        "generated_at": utc_now().isoformat(),
        "summary": {
            "total_events": len(events),
            "unique_users": len({e.get("user_id") for e in events if e.get("user_id")}),
            "unique_sessions": len({e.get("session_id") for e in events if e.get("session_id")}),
            "event_types": freq,
        },
    }

    return {"status": "accepted", "report_id": report_id, "message": "报告已生成"}

@router.get("/reports/{report_id}", summary="获取分析报告")
async def get_report(report_id: str):
    report = reports_data.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="报告不存在")
    return report

@router.get("/reports", summary="报告列表")
async def list_reports(
    limit: int = Query(50, ge=1, le=500, description="返回数量限制"),
    offset: int = Query(0, ge=0, description="偏移量"),
):
    reports = list(reports_data.values())
    reports.sort(key=lambda r: r.get("generated_at") or "", reverse=True)
    total = len(reports)
    return {"reports": reports[offset : offset + limit], "total": total, "limit": limit, "offset": offset}

@router.delete("/reports/{report_id}", summary="删除分析报告")
async def delete_report(report_id: str):
    report = reports_data.pop(report_id, None)
    if not report:
        raise HTTPException(status_code=404, detail="报告不存在")
    return {"status": "success", "report_id": report_id}

@router.get("/reports/{report_id}/download", summary="下载分析报告")
async def download_report(report_id: str, format: str = Query("json", description="下载格式")):
    report = reports_data.get(report_id)
    if not report:
        raise HTTPException(status_code=404, detail="报告不存在")

    if format == "json":
        content = json.dumps(report, ensure_ascii=False, default=str)
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=report_{report_id}.json"},
        )

    if format == "html":
        summary = report.get("summary") or {}
        content = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>分析报告 {report_id}</title></head>
<body>
<h1>行为分析报告</h1>
<p>报告ID: {report_id}</p>
<p>生成时间: {report.get('generated_at')}</p>
<h2>摘要</h2>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
</body>
</html>
"""
        return Response(
            content=content,
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=report_{report_id}.html"},
        )

    raise HTTPException(status_code=400, detail="不支持的下载格式")

@router.get("/dashboard/stats", summary="仪表板统计数据")
async def get_dashboard_stats(
    time_range: str = Query("24h", description="时间范围: 1h, 24h, 7d, 30d"),
    user_id: Optional[str] = Query(None, description="用户ID"),
):
    now = utc_now()
    if time_range == "1h":
        start_time = now - timedelta(hours=1)
    elif time_range == "24h":
        start_time = now - timedelta(hours=24)
    elif time_range == "7d":
        start_time = now - timedelta(days=7)
    elif time_range == "30d":
        start_time = now - timedelta(days=30)
    else:
        raise HTTPException(status_code=400, detail="无效的时间范围")

    events = await _filter_events(start_time=start_time, end_time=now, user_id=user_id)
    stats: Dict[str, int] = {}
    for ev in events:
        et = ev.get("event_type")
        stats[et] = stats.get(et, 0) + 1

    return {
        "time_range": time_range,
        "period": {"start": start_time.isoformat(), "end": now.isoformat()},
        "stats": stats,
    }

@router.get("/realtime/events", summary="实时事件流")
async def stream_events():
    _require_event_store()

    async def event_stream():
        while True:
            recent_events = await _filter_events(
                start_time=utc_now() - timedelta(minutes=5),
                end_time=utc_now(),
            )
            for event in recent_events:
                yield f"data: {json.dumps(event, default=str, ensure_ascii=False)}\n\n"
            await asyncio.sleep(5)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket, user_id: Optional[str] = None, session_id: Optional[str] = None
):
    connection_id = f"ws_{uuid.uuid4().hex}"
    await ws_manager.connect(websocket, connection_id, user_id, session_id)

    try:
        while True:
            data = await websocket.receive_text()
            await ws_manager.record_receive(connection_id)
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"type": "error", "message": "无效的JSON格式"}))
                continue

            action = message.get("action")
            if action == "subscribe":
                sub_type = message.get("type")
                if sub_type:
                    await ws_manager.subscribe(connection_id, sub_type)
            elif action == "unsubscribe":
                sub_type = message.get("type")
                if sub_type:
                    await ws_manager.unsubscribe(connection_id, sub_type)
            elif action == "ping":
                await websocket.send_text(json.dumps({"type": "pong", "timestamp": utc_now().isoformat()}))
    except WebSocketDisconnect:
        logger.info("WebSocket连接断开", user_id=user_id, session_id=session_id)
    except Exception as e:
        logger.error("WebSocket处理失败", error=str(e))
    finally:
        await ws_manager.disconnect(connection_id)

@router.get("/ws/stats", summary="WebSocket连接统计")
async def get_websocket_stats():
    return {"status": "success", "stats": ws_manager.get_stats(), "timestamp": utc_now().isoformat()}

@router.post("/realtime/broadcast", summary="广播实时消息")
async def broadcast_realtime_message(
    message_type: str,
    data: Dict[str, Any],
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
):
    message = RealtimeMessage(type=message_type, data=data, user_id=user_id, session_id=session_id)
    await ws_manager.broadcast_message(message)
    return {"status": "success", "message": "消息已广播", "timestamp": utc_now().isoformat()}

@router.get("/export/events", summary="导出事件数据")
async def export_events(
    format: str = Query("csv", description="导出格式: csv, json, xlsx"),
    user_id: Optional[str] = Query(None, description="用户ID"),
    start_time: Optional[datetime] = Query(None, description="开始时间"),
    end_time: Optional[datetime] = Query(None, description="结束时间"),
    limit: int = Query(10000, ge=1, le=50000, description="导出数量限制"),
):
    if format not in ["csv", "json", "xlsx"]:
        raise HTTPException(status_code=400, detail="不支持的导出格式")

    events = await _filter_events(user_id=user_id, start_time=start_time, end_time=end_time)
    events = events[:limit]
    if not events:
        raise HTTPException(status_code=404, detail="没有找到匹配的数据")

    if format == "json":
        content = json.dumps(events, default=str, ensure_ascii=False)
        media_type = "application/json"
        filename = f"events_{utc_now().strftime('%Y%m%d_%H%M%S')}.json"
        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    if format == "csv":
        output = io.StringIO()
        fieldnames = sorted({k for e in events for k in e.keys()})
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for e in events:
            writer.writerow({k: (json.dumps(e[k], ensure_ascii=False, default=str) if isinstance(e.get(k), (dict, list)) else e.get(k)) for k in fieldnames})
        filename = f"events_{utc_now().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    import pandas as pd

    df = pd.DataFrame(events)
    output = io.BytesIO()
    df.to_excel(output, index=False)
    filename = f"events_{utc_now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return Response(
        content=output.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )

@router.get("/health", summary="健康检查")
async def health_check():
    return {
        "status": "healthy" if event_store else "unhealthy",
        "timestamp": utc_now().isoformat(),
        "components": {
            "event_store": "healthy" if event_store else "uninitialized",
            "websocket_manager": "healthy",
        },
        "ws_stats": ws_manager.get_stats(),
    }

class ExportCreateRequest(ApiBaseModel):
    title: str
    data_type: str
    format: str
    filters: Optional[Dict[str, Any]] = None

@router.get("/exports", summary="导出任务列表")
async def list_exports() -> Dict[str, Any]:
    return {"tasks": export_tasks, "total": len(export_tasks)}

@router.post("/exports", summary="创建导出任务")
async def create_export(request: ExportCreateRequest) -> Dict[str, Any]:
    task_id = f"export-{uuid.uuid4()}"
    export_tasks.append(
        {
            "task_id": task_id,
            "title": request.title,
            "data_type": request.data_type,
            "format": request.format,
            "status": "pending",
            "progress": 0,
            "created_at": utc_now().isoformat(),
            "filters": request.filters or {},
        }
    )
    return {"task_id": task_id, "status": "accepted"}
