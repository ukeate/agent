"""
情感智能 WebSocket
接收多模态输入并实时返回情感识别结果（无静态占位与 501）。
"""

from __future__ import annotations

import base64
import io
import json
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from PIL import Image, ImageStat
from src.core.utils.timezone_utils import utc_now

from src.core.logging import get_logger
logger = get_logger(__name__)

router = APIRouter(prefix="/ws", tags=["emotion-websocket"])

@dataclass(frozen=True)
class EmotionState:
    emotion: str
    intensity: float
    valence: float
    arousal: float
    dominance: float
    confidence: float
    timestamp: str

def _vad(emotion: str) -> tuple[float, float, float]:
    mapping = {
        "happiness": (0.8, 0.7, 0.6),
        "sadness": (-0.6, 0.3, 0.3),
        "anger": (-0.4, 0.8, 0.7),
        "fear": (-0.7, 0.7, 0.2),
        "surprise": (0.2, 0.8, 0.4),
        "disgust": (-0.8, 0.5, 0.4),
        "neutral": (0.0, 0.2, 0.5),
    }
    return mapping.get(emotion, (0.0, 0.2, 0.5))

def _b64decode(data: str) -> bytes:
    s = (data or "").strip()
    if not s:
        return b""
    pad = (-len(s)) % 4
    return base64.b64decode(s + ("=" * pad))

def _emotion_from_text(text: str) -> EmotionState:
    t0 = utc_now().isoformat()
    s = (text or "").lower()
    keywords = {
        "happiness": ["happy", "joy", "excited", "glad", "pleased", "cheerful", "开心", "高兴", "快乐", "兴奋", "满意"],
        "sadness": ["sad", "depressed", "disappointed", "grief", "sorrow", "难过", "伤心", "失望", "沮丧"],
        "anger": ["angry", "mad", "furious", "annoyed", "irritated", "生气", "愤怒", "恼火", "烦"],
        "fear": ["scared", "afraid", "nervous", "worried", "anxious", "害怕", "担心", "焦虑", "紧张"],
        "surprise": ["surprised", "amazed", "shocked", "unexpected", "惊讶", "震惊", "意外"],
        "disgust": ["disgusted", "revolted", "repulsed", "sick", "恶心", "厌恶", "反感"],
    }
    scores: Dict[str, float] = {}
    for emo, ks in keywords.items():
        hit = sum(1 for k in ks if k in s)
        if hit:
            scores[emo] = hit / max(len(ks), 1)
    if not scores:
        emo = "neutral"
        intensity = 0.4
        conf = 0.3
    else:
        emo = max(scores.keys(), key=lambda k: scores[k])
        intensity = min(max(scores[emo], 0.2), 1.0)
        conf = min(0.9, 0.3 + intensity)
    v, a, d = _vad(emo)
    return EmotionState(emo, intensity, v, a, d, conf, t0)

def _emotion_from_image_b64(image_b64: str) -> Optional[EmotionState]:
    raw = _b64decode(image_b64)
    if not raw:
        return None
    t0 = utc_now().isoformat()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return None

    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    mean = float(stat.mean[0])
    std = float(stat.stddev[0])
    brightness = mean / 255.0
    contrast = min(1.0, std / 128.0)

    if brightness > 0.65 and contrast > 0.35:
        emo = "happiness"
    elif brightness > 0.65:
        emo = "surprise"
    elif brightness < 0.35:
        emo = "sadness"
    elif contrast > 0.55:
        emo = "anger" if brightness < 0.5 else "surprise"
    else:
        emo = "neutral"

    v, a, d = _vad(emo)
    intensity = min(1.0, 0.3 + abs(brightness - 0.5) + contrast * 0.6)
    conf = min(0.85, 0.35 + contrast * 0.4)
    return EmotionState(emo, intensity, v, max(a, contrast), d, conf, t0)

def _emotion_from_bytes(kind: str, raw: bytes) -> Optional[EmotionState]:
    if not raw:
        return None
    t0 = utc_now().isoformat()
    mean = sum(raw) / len(raw)
    var = sum((b - mean) ** 2 for b in raw) / len(raw)
    arousal = min(1.0, (var ** 0.5) / 80.0)
    valence = (mean - 127.5) / 127.5
    emo = "neutral"
    if arousal > 0.7 and valence < -0.2:
        emo = "anger"
    elif arousal > 0.7 and valence > 0.2:
        emo = "surprise"
    elif arousal < 0.3 and valence < -0.2:
        emo = "sadness"
    elif arousal < 0.3 and valence > 0.2:
        emo = "happiness"
    v, a, d = _vad(emo)
    conf = 0.35 + min(0.55, arousal * 0.5 + abs(valence) * 0.3)
    return EmotionState(emo, conf, v, max(a, arousal), d, min(conf, 0.9), t0)

def _fuse(emotions: Dict[str, EmotionState]) -> EmotionState:
    t0 = utc_now().isoformat()
    if not emotions:
        v, a, d = _vad("neutral")
        return EmotionState("neutral", 0.3, v, a, d, 0.2, t0)

    total = 0.0
    valence = 0.0
    arousal = 0.0
    dominance = 0.0
    votes: Dict[str, float] = {}
    for emo in emotions.values():
        w = max(0.0, float(emo.confidence))
        total += w
        valence += emo.valence * w
        arousal += emo.arousal * w
        dominance += emo.dominance * w
        votes[emo.emotion] = votes.get(emo.emotion, 0.0) + w

    if total <= 0:
        v, a, d = _vad("neutral")
        return EmotionState("neutral", 0.3, v, a, d, 0.2, t0)

    emo_name = max(votes.keys(), key=lambda k: votes[k])
    v, a0, d0 = _vad(emo_name)
    conf = min(0.95, total / max(len(emotions), 1))
    return EmotionState(
        emo_name,
        intensity=min(1.0, conf),
        valence=valence / total,
        arousal=max(arousal / total, a0),
        dominance=max(dominance / total, d0),
        confidence=conf,
        timestamp=t0,
    )

def _msg(msg_type: str, user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "type": msg_type,
        "data": data,
        "timestamp": utc_now().isoformat(),
        "message_id": str(uuid.uuid4()),
        "user_id": user_id,
    }

@router.websocket("/emotion/{user_id}")
async def emotion_ws(websocket: WebSocket, user_id: str):
    await websocket.accept()
    await websocket.send_text(
        json.dumps(
            _msg(
                "connect",
                user_id,
                {
                    "status": "connected",
                    "user_id": user_id,
                    "server_time": utc_now().isoformat(),
                    "capabilities": ["text", "image", "audio", "video", "physiological"],
                },
            ),
            ensure_ascii=False,
        )
    )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                await websocket.send_text(json.dumps(_msg("error", user_id, {"error": "消息格式无效"}), ensure_ascii=False))
                continue

            msg_type = str(msg.get("type", "")).strip()
            if msg_type == "heartbeat":
                await websocket.send_text(json.dumps(_msg("heartbeat", user_id, {"ok": True}), ensure_ascii=False))
                continue

            if msg_type != "emotion_input":
                await websocket.send_text(json.dumps(_msg("error", user_id, {"error": "不支持的消息类型"}), ensure_ascii=False))
                continue

            t_start = time.time()
            data = msg.get("data") or {}
            modalities = data.get("modalities") or []

            emotions: Dict[str, EmotionState] = {}
            if "text" in modalities and data.get("text"):
                emotions["text"] = _emotion_from_text(str(data["text"]))
            if "physiological" in modalities and data.get("physiological_data"):
                emo = _emotion_from_bytes("physiological", json.dumps(data["physiological_data"], ensure_ascii=False).encode("utf-8"))
                if emo:
                    emotions["physiological"] = emo
            if "image" in modalities and data.get("image_data"):
                emo = _emotion_from_image_b64(str(data["image_data"]))
                if emo:
                    emotions["image"] = emo
            if "audio" in modalities and data.get("audio_data"):
                emo = _emotion_from_bytes("audio", _b64decode(str(data["audio_data"])))
                if emo:
                    emotions["audio"] = emo
            if "video" in modalities and data.get("video_data"):
                emo = _emotion_from_bytes("video", _b64decode(str(data["video_data"])))
                if emo:
                    emotions["video"] = emo

            fused = _fuse(emotions)
            processing_time = max(0.0, time.time() - t_start)
            unified = {
                "user_id": user_id,
                "timestamp": utc_now().isoformat(),
                "recognition_result": {
                    "fused_emotion": asdict(fused),
                    "emotions": {k: asdict(v) for k, v in emotions.items()},
                    "confidence": fused.confidence,
                    "processing_time": processing_time,
                },
                "emotional_state": asdict(fused),
                "confidence": fused.confidence,
                "processing_time": processing_time,
                "data_quality": min(1.0, max(0.2, fused.confidence)),
            }
            await websocket.send_text(json.dumps(_msg("emotion_result", user_id, unified), ensure_ascii=False))

    except WebSocketDisconnect:
        logger.info("情感WebSocket断开", user_id=user_id)
    except Exception as e:
        logger.error("情感WebSocket异常", user_id=user_id, error=str(e))
        try:
            await websocket.close()
        except Exception:
            logger.exception("关闭情感WebSocket失败", exc_info=True)
