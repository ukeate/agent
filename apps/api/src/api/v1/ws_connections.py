"""
WebSocket 连接管理 API
提供连接列表、断开连接与消息发送等能力
"""

import asyncio
import json
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from src.api.v1.analytics import ws_manager
from src.core.utils.timezone_utils import utc_now

router = APIRouter(prefix="/ws", tags=["websocket-connections"])

def _normalize_message(message: Any) -> Dict[str, Any]:
    if isinstance(message, dict):
        return message
    return {"type": "custom", "data": message}

def _format_sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

@router.get("/connections")
async def list_connections():
    details = await ws_manager.list_connection_details()
    return {
        "active_connections": len(details),
        "connection_details": details,
    }

@router.delete("/connections/{connection_index}")
async def disconnect_connection(connection_index: int):
    connection_id = await ws_manager.resolve_connection_id(connection_index)
    if not connection_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="连接不存在")
    await ws_manager.disconnect(connection_id)
    return {"success": True, "message": "连接已断开"}

@router.delete("/connections/all")
async def disconnect_all_connections():
    details = await ws_manager.list_connection_details()
    connection_ids = [item.get("connection_id") for item in details if item.get("connection_id")]
    for connection_id in connection_ids:
        await ws_manager.disconnect(connection_id)
    return {
        "success": True,
        "disconnected_count": len(connection_ids),
        "message": "已断开所有连接",
    }

@router.post("/connections/{connection_index}/send")
async def send_message_to_connection(connection_index: int, payload: Dict[str, Any]):
    message = payload.get("message")
    if message is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="message不能为空")
    connection_id = await ws_manager.resolve_connection_id(connection_index)
    if not connection_id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="连接不存在")
    await ws_manager.send_to_connection(connection_id, _normalize_message(message))
    return {"success": True, "message": "消息已发送"}

@router.post("/connections/broadcast")
async def broadcast_message(payload: Dict[str, Any]):
    message = payload.get("message")
    if message is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="message不能为空")
    sent_count = await ws_manager.broadcast_payload(_normalize_message(message))
    return {"success": True, "sent_count": sent_count, "message": "消息已广播"}

@router.get("/connections/events")
async def stream_connection_events(poll_interval: float = 2.0):
    async def event_stream():
        previous: Dict[str, Dict[str, Any]] = {}
        while True:
            try:
                details = await ws_manager.list_connection_details()
                current = {
                    detail.get("connection_id"): detail
                    for detail in details
                    if detail.get("connection_id")
                }

                for connection_id, detail in current.items():
                    if connection_id not in previous:
                        yield _format_sse({
                            "type": "connect",
                            "connection_index": detail.get("index", -1),
                            "connection_id": connection_id,
                            "timestamp": utc_now().isoformat(),
                            "data": detail,
                        })

                for connection_id, detail in previous.items():
                    if connection_id not in current:
                        yield _format_sse({
                            "type": "disconnect",
                            "connection_index": detail.get("index", -1),
                            "connection_id": connection_id,
                            "timestamp": utc_now().isoformat(),
                            "data": detail,
                        })

                for connection_id, detail in current.items():
                    previous_detail = previous.get(connection_id)
                    if not previous_detail:
                        continue
                    if (
                        detail.get("messages_sent") != previous_detail.get("messages_sent")
                        or detail.get("messages_received") != previous_detail.get("messages_received")
                    ):
                        yield _format_sse({
                            "type": "message",
                            "connection_index": detail.get("index", -1),
                            "connection_id": connection_id,
                            "timestamp": utc_now().isoformat(),
                            "data": {
                                "messages_sent": detail.get("messages_sent"),
                                "messages_received": detail.get("messages_received"),
                                "last_activity": detail.get("last_activity"),
                            },
                        })

                previous = current
                yield ": keep-alive\n\n"
                await asyncio.sleep(poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                yield _format_sse({
                    "type": "error",
                    "connection_index": -1,
                    "timestamp": utc_now().isoformat(),
                    "data": {"error": str(exc)},
                })
                await asyncio.sleep(poll_interval)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)
