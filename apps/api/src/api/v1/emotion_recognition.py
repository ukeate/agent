"""
情感识别API端点
Story 11.1: 多模态情感识别引擎
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, List, Dict, Any
import httpx
from PIL import Image
from src.api.base_model import ApiBaseModel
from src.ai.emotion_recognition.models.emotion_models import (
    EmotionResult,
)
from src.ai.emotion_recognition.analyzers.text_analyzer import TextEmotionAnalyzer
from src.ai.emotion_recognition.analyzers.audio_analyzer import AudioEmotionAnalyzer
from src.ai.emotion_recognition.analyzers.visual_analyzer import VisualEmotionAnalyzer
from src.ai.emotion_recognition.fusion.engine import MultiModalEmotionFusion
from src.core.redis import get_redis
from src.core.utils.timezone_utils import utc_now

# 导入情感识别组件

router = APIRouter(prefix="/emotion-recognition", tags=["emotion_recognition"])

# 初始化分析器
text_analyzer = TextEmotionAnalyzer()
audio_analyzer = AudioEmotionAnalyzer()
visual_analyzer = VisualEmotionAnalyzer()
fusion_engine = MultiModalEmotionFusion()

# 请求模型
class TextEmotionRequest(ApiBaseModel):
    """文本情感分析请求"""
    text: str
    language: Optional[str] = "auto"
    include_context: bool = True
    confidence_threshold: float = 0.5

class MultiModalEmotionRequest(ApiBaseModel):
    """多模态情感分析请求"""
    text: Optional[str] = None
    audio_url: Optional[str] = None
    image_url: Optional[str] = None
    fusion_strategy: str = "dynamic_adaptive"
    weights: Optional[Dict[str, float]] = None

_STATS_KEY = "emotion_recognition:stats"
_EMOTION_COUNTS_KEY = "emotion_recognition:emotion_counts"
_MODALITY_COUNTS_KEY = "emotion_recognition:modality_counts"
_MODALITY_LATENCY_KEY = "emotion_recognition:modality_latency_ms"

async def _record_stats(modality: str, emotion: str, processing_time_ms: float) -> None:
    redis_client = get_redis()
    if not redis_client:
        return
    await redis_client.hincrby(_STATS_KEY, "total_requests", 1)
    await redis_client.hincrbyfloat(_STATS_KEY, "total_latency_ms", float(processing_time_ms))
    await redis_client.hincrby(_EMOTION_COUNTS_KEY, emotion, 1)
    await redis_client.hincrby(_MODALITY_COUNTS_KEY, modality, 1)
    await redis_client.hincrbyfloat(_MODALITY_LATENCY_KEY, modality, float(processing_time_ms))
    day_key = f"emotion_recognition:processed:{utc_now().strftime('%Y%m%d')}"
    await redis_client.incr(day_key)

def _emotion_result_payload(result: EmotionResult) -> Dict[str, Any]:
    dimension = result.dimension
    emotions = [{"label": result.emotion, "score": result.confidence}]
    if result.sub_emotions:
        for label, score in result.sub_emotions:
            emotions.append({"label": label, "score": score})
    return {
        "primaryEmotion": result.emotion,
        "confidence": result.confidence,
        "intensity": result.intensity,
        "emotions": emotions,
        "valence": dimension.valence if dimension else 0.0,
        "arousal": dimension.arousal if dimension else 0.0,
        "dominance": dimension.dominance if dimension else 0.0,
        "details": result.details,
    }

@router.post("/analyze/text")
async def analyze_text_emotion(request: TextEmotionRequest):
    """分析文本情感"""
    try:
        start_time = utc_now()
        
        # 执行文本情感分析
        result = await text_analyzer.analyze(request.text)
        
        # 计算处理时间
        processing_time = (utc_now() - start_time).total_seconds() * 1000
        payload = _emotion_result_payload(result)
        payload.update(
            {
                "processingTime": processing_time,
                "timestamp": utc_now().isoformat(),
                "language": request.language,
                "features": {
                    "textLength": len(request.text),
                    "wordCount": len(request.text.split()),
                    "sentenceCount": len([s for s in request.text.replace("\n", " ").split(".") if s.strip()]),
                    "exclamationCount": request.text.count("!"),
                    "questionCount": request.text.count("?"),
                },
            }
        )
        await _record_stats("text", payload["primaryEmotion"], processing_time)
        return payload
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/audio")
async def analyze_audio_emotion(
    audio_file: UploadFile = File(...),
    sample_rate: int = Form(16000),
    chunk_duration: float = Form(2.0)
):
    """分析音频情感"""
    try:
        start_time = utc_now()
        if audio_file.filename and not audio_file.filename.lower().endswith(".wav"):
            raise HTTPException(status_code=400, detail="当前仅支持wav格式音频")
        audio_bytes = await audio_file.read()
        result = await audio_analyzer.analyze(audio_bytes)
        processing_time = (utc_now() - start_time).total_seconds() * 1000
        payload = _emotion_result_payload(result)
        payload.update(
            {
                "processingTime": processing_time,
                "timestamp": utc_now().isoformat(),
                "filename": audio_file.filename,
                "sampleRate": sample_rate,
                "chunkDuration": chunk_duration,
            }
        )
        await _record_stats("audio", payload["primaryEmotion"], processing_time)
        return payload
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/visual")
async def analyze_visual_emotion(
    image_file: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None),
    detect_faces: bool = Form(True),
    aggregate_method: str = Form("weighted_average")
):
    """分析图像情感"""
    try:
        start_time = utc_now()
        if image_file:
            image_bytes = await image_file.read()
        elif image_url:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(image_url)
                resp.raise_for_status()
                image_bytes = resp.content
        else:
            raise HTTPException(status_code=400, detail="缺少图像输入")

        if not visual_analyzer.is_initialized:
            await visual_analyzer.initialize()

        preprocessed = await visual_analyzer.preprocess(image_bytes)
        faces = preprocessed.get("faces") or []
        face_images = preprocessed.get("face_images") or []

        face_results = []
        for idx, ((x, y, w, h), face_img) in enumerate(zip(faces, face_images)):
            if visual_analyzer.pipeline:
                preds = visual_analyzer.pipeline(Image.fromarray(face_img))
                primary = preds[0] if isinstance(preds, list) and preds else {"label": "neutral", "score": 0.0}
                emotion = visual_analyzer._standardize_visual_emotion_label(primary["label"])
                confidence = float(primary["score"])
            else:
                single = {
                    "faces": [(x, y, w, h)],
                    "features": await visual_analyzer._extract_visual_features(preprocessed["original_image"], [(x, y, w, h)]),
                }
                r = await visual_analyzer._rule_based_analysis(single)
                emotion = r.emotion
                confidence = float(r.confidence)

            face_results.append(
                {
                    "id": idx,
                    "emotion": emotion,
                    "confidence": confidence,
                    "intensity": confidence,
                    "boundingBox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
                }
            )

        if not face_results and not detect_faces:
            result = await visual_analyzer.analyze(image_bytes)
            face_results = [
                {
                    "id": 0,
                    "emotion": result.emotion,
                    "confidence": float(result.confidence),
                    "intensity": float(result.intensity),
                    "boundingBox": {"x": 0, "y": 0, "w": 0, "h": 0},
                }
            ]

        counts: Dict[str, List[float]] = {}
        for f in face_results:
            counts.setdefault(f["emotion"], []).append(f["confidence"])
        emotions_summary = [
            {
                "label": label,
                "count": len(scores),
                "avgConfidence": sum(scores) / len(scores) if scores else 0.0,
            }
            for label, scores in counts.items()
        ]
        emotions_summary.sort(key=lambda x: x["count"], reverse=True)

        aggregated = {"emotion": "neutral", "confidence": 0.0}
        if emotions_summary:
            top = max(emotions_summary, key=lambda x: x["avgConfidence"])
            aggregated = {"emotion": top["label"], "confidence": top["avgConfidence"]}

        processing_time = (utc_now() - start_time).total_seconds() * 1000
        await _record_stats("visual", aggregated["emotion"], processing_time)

        return {
            "numFaces": len(face_results),
            "aggregatedEmotion": aggregated,
            "faces": face_results,
            "emotions": emotions_summary,
            "processingTime": processing_time,
            "timestamp": utc_now().isoformat(),
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze/multimodal")
async def analyze_multimodal_emotion(request: MultiModalEmotionRequest):
    """多模态情感融合分析"""
    try:
        start_time = utc_now()
        audio_bytes = None
        image_bytes = None
        if request.audio_url:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(request.audio_url)
                resp.raise_for_status()
                audio_bytes = resp.content
        if request.image_url:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(request.image_url)
                resp.raise_for_status()
                image_bytes = resp.content

        fused = await fusion_engine.analyze(text=request.text, audio=audio_bytes, image=image_bytes)
        processing_time = (utc_now() - start_time).total_seconds() * 1000

        return {
            "primaryEmotion": fused.primary_emotion,
            "confidence": fused.overall_confidence,
            "intensity": fused.intensity_level,
            "valence": fused.valence,
            "arousal": fused.arousal,
            "dominance": fused.dominance,
            "modalityWeights": fused.modality_weights,
            "processingTime": processing_time,
            "timestamp": utc_now().isoformat(),
            "fusionStrategy": request.fusion_strategy,
        }
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_emotion_stats():
    """获取情感识别系统统计信息"""
    redis_client = get_redis()
    if not redis_client:
        return {"total_requests": 0, "average_latency_ms": 0.0, "processed_today": 0, "emotion_distribution": {}}
    stats = await redis_client.hgetall(_STATS_KEY)
    total = int(stats.get("total_requests") or 0)
    total_latency = float(stats.get("total_latency_ms") or 0.0)
    avg_latency = total_latency / total if total else 0.0
    processed_today = int(await redis_client.get(f"emotion_recognition:processed:{utc_now().strftime('%Y%m%d')}") or 0)
    emotion_distribution = await redis_client.hgetall(_EMOTION_COUNTS_KEY)
    modality_counts = await redis_client.hgetall(_MODALITY_COUNTS_KEY)
    modality_latency = await redis_client.hgetall(_MODALITY_LATENCY_KEY)
    modality_performance = {}
    for modality, count_str in modality_counts.items():
        count = int(count_str or 0)
        latency_sum = float(modality_latency.get(modality) or 0.0)
        modality_performance[modality] = {
            "requests": count,
            "avg_latency_ms": latency_sum / count if count else 0.0,
        }
    return {
        "total_requests": total,
        "average_latency_ms": avg_latency,
        "accuracy_rate": None,
        "active_models": 3,
        "processed_today": processed_today,
        "emotion_distribution": {k: int(v) for k, v in emotion_distribution.items()},
        "modality_performance": modality_performance,
        "system_status": "healthy",
        "last_update": utc_now().isoformat(),
    }

@router.get("/models")
async def get_available_models():
    """获取可用的情感识别模型"""
    return {
        "text": text_analyzer.get_model_info(),
        "audio": audio_analyzer.get_model_info(),
        "visual": visual_analyzer.get_model_info(),
        "fusion": {"config": fusion_engine.config if hasattr(fusion_engine, "config") else {}},
    }

@router.websocket("/stream")
async def emotion_stream_endpoint(websocket):
    """WebSocket端点for实时情感流分析"""
    await websocket.accept()
    try:
        while True:
            # 接收数据
            data = await websocket.receive_json()
            
            # 处理数据并返回结果
            if data.get("type") == "text":
                result = await text_analyzer.analyze(data.get("content", ""))
                await websocket.send_json({
                    "type": "result",
                    "emotion": result.emotion,
                    "confidence": result.confidence
                })
            elif data.get("type") == "close":
                break
                
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()

@router.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "analyzers": {
            "text": "ready" if text_analyzer.is_initialized else "not_initialized",
            "audio": "ready" if audio_analyzer.is_initialized else "not_initialized",
            "visual": "ready" if visual_analyzer.is_initialized else "not_initialized",
            "fusion": "ready"
        },
        "timestamp": utc_now().isoformat()
    }
