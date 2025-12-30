"""
多模态情感识别引擎集成实现
整合Story 11.1的情感识别系统到统一架构中
"""

from src.core.utils.timezone_utils import utc_now
import asyncio
import json
import numpy as np
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import timedelta
from abc import ABC, abstractmethod
from dataclasses import asdict
import base64
import io
from concurrent.futures import ThreadPoolExecutor
from .core_interfaces import (
    EmotionRecognitionEngine, MultiModalEmotion, EmotionState,
    EmotionType, ModalityType, UnifiedEmotionalData
)
from .communication_protocol import (
    CommunicationProtocol, ModuleType, MessageHandler, Message
)

from src.core.logging import get_logger
class ModalityProcessor(ABC):
    """模态处理器抽象基类"""
    
    @abstractmethod
    async def process(self, data: Any) -> EmotionState:
        """处理特定模态数据"""
        ...
    
    @abstractmethod
    def validate_input(self, data: Any) -> bool:
        """验证输入数据格式"""
        ...
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """获取支持的数据格式"""
        ...

class TextEmotionProcessor(ModalityProcessor):
    """文本情感处理器"""
    
    def __init__(self):
        self._logger = get_logger(__name__)
        # 这里可以集成具体的文本情感分析模型
        self._emotion_keywords = {
            EmotionType.HAPPINESS: ["happy", "joy", "excited", "glad", "pleased", "cheerful"],
            EmotionType.SADNESS: ["sad", "depressed", "disappointed", "grief", "sorrow"],
            EmotionType.ANGER: ["angry", "mad", "furious", "annoyed", "irritated"],
            EmotionType.FEAR: ["scared", "afraid", "nervous", "worried", "anxious"],
            EmotionType.SURPRISE: ["surprised", "amazed", "shocked", "unexpected"],
            EmotionType.DISGUST: ["disgusted", "revolted", "repulsed", "sick"]
        }
    
    async def process(self, data: Any) -> EmotionState:
        """处理文本数据"""
        if not self.validate_input(data):
            raise ValueError("Invalid text input")
        
        text = str(data).lower()
        
        # 简化的关键词匹配算法
        emotion_scores = {}
        for emotion, keywords in self._emotion_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                emotion_scores[emotion] = score / len(keywords)
        
        if not emotion_scores:
            return EmotionState(
                emotion=EmotionType.NEUTRAL,
                intensity=0.5,
                valence=0.0,
                arousal=0.0,
                dominance=0.5,
                confidence=0.3,
                timestamp=utc_now()
            )
        
        # 选择得分最高的情感
        primary_emotion = max(emotion_scores.keys(), key=lambda k: emotion_scores[k])
        intensity = min(max(emotion_scores[primary_emotion], 0.0), 1.0)
        
        # 根据情感类型设置VAD值
        valence, arousal, dominance = self._get_vad_values(primary_emotion)
        
        return EmotionState(
            emotion=primary_emotion,
            intensity=intensity,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            confidence=intensity,
            timestamp=utc_now()
        )
    
    def validate_input(self, data: Any) -> bool:
        """验证文本输入"""
        return isinstance(data, (str, bytes)) and len(str(data).strip()) > 0
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式"""
        return ["text/plain", "string"]
    
    def _get_vad_values(self, emotion: EmotionType) -> Tuple[float, float, float]:
        """获取情感的VAD值"""
        vad_mapping = {
            EmotionType.HAPPINESS: (0.8, 0.7, 0.6),
            EmotionType.SADNESS: (-0.6, -0.4, -0.3),
            EmotionType.ANGER: (-0.4, 0.8, 0.7),
            EmotionType.FEAR: (-0.7, 0.6, -0.5),
            EmotionType.SURPRISE: (0.2, 0.8, 0.0),
            EmotionType.DISGUST: (-0.8, 0.4, 0.2),
            EmotionType.NEUTRAL: (0.0, 0.0, 0.0)
        }
        return vad_mapping.get(emotion, (0.0, 0.0, 0.0))

class AudioEmotionProcessor(ModalityProcessor):
    """音频情感处理器"""
    
    def __init__(self):
        self._logger = get_logger(__name__)
        # 这里可以集成具体的音频情感分析模型
        self._supported_formats = ["audio/wav", "audio/mp3", "audio/flac"]
    
    async def process(self, data: Any) -> EmotionState:
        """处理音频数据"""
        if not self.validate_input(data):
            raise ValueError("Invalid audio input")
        raise RuntimeError("audio emotion model not integrated; please configure real processor")
    
    def validate_input(self, data: Any) -> bool:
        """验证音频输入"""
        if isinstance(data, dict):
            return "audio_data" in data or "file_path" in data
        return isinstance(data, (bytes, str))
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式"""
        return self._supported_formats
    
    def _extract_audio_features(self, data: Any) -> Dict[str, float]:
        """提取音频特征"""
        raise RuntimeError("audio feature extraction not implemented")
    
    def _predict_emotion_from_features(self, features: Dict[str, float]) -> Tuple[EmotionType, float]:
        """基于特征预测情感"""
        raise RuntimeError("audio emotion prediction not implemented")
    
    def _get_vad_from_emotion(self, emotion: EmotionType) -> Tuple[float, float, float]:
        """从情感获取VAD值"""
        vad_mapping = {
            EmotionType.HAPPINESS: (0.8, 0.7, 0.6),
            EmotionType.SADNESS: (-0.6, -0.4, -0.3),
            EmotionType.ANGER: (-0.4, 0.8, 0.7),
            EmotionType.FEAR: (-0.7, 0.6, -0.5),
            EmotionType.SURPRISE: (0.2, 0.8, 0.0),
            EmotionType.DISGUST: (-0.8, 0.4, 0.2),
            EmotionType.NEUTRAL: (0.0, 0.0, 0.0)
        }
        return vad_mapping.get(emotion, (0.0, 0.0, 0.0))

class VideoEmotionProcessor(ModalityProcessor):
    """视频情感处理器（主要处理面部表情）"""
    
    def __init__(self):
        self._logger = get_logger(__name__)
        self._supported_formats = ["video/mp4", "video/avi", "image/jpeg", "image/png"]
    
    async def process(self, data: Any) -> EmotionState:
        """处理视频/图像数据"""
        if not self.validate_input(data):
            raise ValueError("Invalid video/image input")
        raise RuntimeError("video/face emotion model not integrated; configure real processor")
    
    def validate_input(self, data: Any) -> bool:
        """验证视频/图像输入"""
        if isinstance(data, dict):
            return "video_data" in data or "image_data" in data or "file_path" in data
        return isinstance(data, (bytes, str))
    
    def get_supported_formats(self) -> List[str]:
        """获取支持的格式"""
        return self._supported_formats
    
    def _extract_facial_features(self, data: Any) -> Dict[str, float]:
        """提取面部特征"""
        raise RuntimeError("facial feature extraction not implemented")
    
    def _predict_emotion_from_facial_features(
        self, 
        features: Dict[str, float]
    ) -> Tuple[EmotionType, float]:
        """基于面部特征预测情感"""
        raise RuntimeError("facial emotion prediction not implemented")
    
    def _get_vad_from_emotion(self, emotion: EmotionType) -> Tuple[float, float, float]:
        """从情感获取VAD值"""
        vad_mapping = {
            EmotionType.HAPPINESS: (0.8, 0.7, 0.6),
            EmotionType.SADNESS: (-0.6, -0.4, -0.3),
            EmotionType.ANGER: (-0.4, 0.8, 0.7),
            EmotionType.FEAR: (-0.7, 0.6, -0.5),
            EmotionType.SURPRISE: (0.2, 0.8, 0.0),
            EmotionType.DISGUST: (-0.8, 0.4, 0.2),
            EmotionType.NEUTRAL: (0.0, 0.0, 0.0)
        }
        return vad_mapping.get(emotion, (0.0, 0.0, 0.0))

class EmotionFusionEngine:
    """情感融合引擎"""
    
    def __init__(self):
        self._logger = get_logger(__name__)
        # 模态权重配置
        self._modality_weights = {
            ModalityType.TEXT: 0.4,
            ModalityType.AUDIO: 0.3,
            ModalityType.VIDEO: 0.2,
            ModalityType.IMAGE: 0.1
        }
    
    def fuse_emotions(
        self, 
        modality_emotions: Dict[ModalityType, EmotionState]
    ) -> EmotionState:
        """融合多模态情感"""
        if not modality_emotions:
            return EmotionState(
                emotion=EmotionType.NEUTRAL,
                intensity=0.0,
                valence=0.0,
                arousal=0.0,
                dominance=0.0,
                confidence=0.0,
                timestamp=utc_now()
            )
        
        # 计算加权平均
        total_weight = 0.0
        weighted_valence = 0.0
        weighted_arousal = 0.0
        weighted_dominance = 0.0
        weighted_intensity = 0.0
        confidence_sum = 0.0
        
        emotion_votes = {}
        
        for modality, emotion_state in modality_emotions.items():
            weight = self._modality_weights.get(modality, 0.1)
            confidence_weight = weight * emotion_state.confidence
            
            total_weight += confidence_weight
            weighted_valence += emotion_state.valence * confidence_weight
            weighted_arousal += emotion_state.arousal * confidence_weight
            weighted_dominance += emotion_state.dominance * confidence_weight
            weighted_intensity += emotion_state.intensity * confidence_weight
            confidence_sum += emotion_state.confidence * weight
            
            # 收集情感投票
            emotion_key = emotion_state.emotion
            if emotion_key not in emotion_votes:
                emotion_votes[emotion_key] = 0.0
            emotion_votes[emotion_key] += confidence_weight
        
        if total_weight == 0:
            return EmotionState(
                emotion=EmotionType.NEUTRAL,
                intensity=0.0,
                valence=0.0,
                arousal=0.0,
                dominance=0.0,
                confidence=0.0,
                timestamp=utc_now()
            )
        
        # 计算最终结果
        final_valence = weighted_valence / total_weight
        final_arousal = weighted_arousal / total_weight
        final_dominance = weighted_dominance / total_weight
        final_intensity = weighted_intensity / total_weight
        
        # 选择得票最高的情感
        final_emotion = max(emotion_votes.keys(), key=lambda k: emotion_votes[k])
        
        # 计算整体置信度
        final_confidence = confidence_sum / len(modality_emotions)
        
        return EmotionState(
            emotion=final_emotion,
            intensity=final_intensity,
            valence=final_valence,
            arousal=final_arousal,
            dominance=final_dominance,
            confidence=final_confidence,
            timestamp=utc_now()
        )
    
    def set_modality_weights(self, weights: Dict[ModalityType, float]):
        """设置模态权重"""
        # 归一化权重
        total = sum(weights.values())
        if total > 0:
            self._modality_weights = {
                modality: weight / total 
                for modality, weight in weights.items()
            }

class MultiModalEmotionRecognitionEngine(EmotionRecognitionEngine):
    """多模态情感识别引擎实现"""
    
    def __init__(self):
        self._logger = get_logger(__name__)
        self._protocol = CommunicationProtocol(ModuleType.EMOTION_RECOGNITION)
        
        # 模态处理器
        self._processors = {
            ModalityType.TEXT: TextEmotionProcessor(),
            ModalityType.AUDIO: AudioEmotionProcessor(),
            ModalityType.VIDEO: VideoEmotionProcessor(),
            ModalityType.IMAGE: VideoEmotionProcessor()  # 复用视频处理器
        }
        
        # 融合引擎
        self._fusion_engine = EmotionFusionEngine()
        
        # 质量监控
        self._quality_metrics = {
            "total_recognitions": 0,
            "successful_recognitions": 0,
            "failed_recognitions": 0,
            "average_confidence": 0.0,
            "average_processing_time": 0.0,
            "modality_usage": {modality.value: 0 for modality in ModalityType}
        }
        
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._is_running = False
    
    async def start(self):
        """启动情感识别引擎"""
        self._is_running = True
        self._logger.info("Starting multi-modal emotion recognition engine")
        
        # 添加消息处理器
        from .communication_protocol import EmotionRecognitionHandler
        handler = EmotionRecognitionHandler(self)
        self._protocol.add_message_handler(handler)
        
        # 启动协议服务
        await self._protocol.start()
    
    async def stop(self):
        """停止情感识别引擎"""
        self._is_running = False
        await self._protocol.stop()
        self._executor.shutdown(wait=True)
        self._logger.info("Multi-modal emotion recognition engine stopped")
    
    async def recognize_emotion(
        self, 
        input_data: Dict[ModalityType, Any]
    ) -> MultiModalEmotion:
        """识别多模态情感"""
        start_time = utc_now()
        self._quality_metrics["total_recognitions"] += 1
        
        try:
            # 并行处理各个模态
            modality_tasks = {}
            for modality, data in input_data.items():
                if modality in self._processors:
                    processor = self._processors[modality]
                    if processor.validate_input(data):
                        task = processor.process(data)
                        modality_tasks[modality] = task
                        self._quality_metrics["modality_usage"][modality.value] += 1
                    else:
                        self._logger.warning(f"Invalid input for modality {modality.value}")
            
            if not modality_tasks:
                raise ValueError("No valid input data for emotion recognition")
            
            # 等待所有模态处理完成
            completed_tasks = await asyncio.gather(
                *modality_tasks.values(), 
                return_exceptions=True
            )
            
            # 收集成功的结果
            modality_emotions = {}
            for (modality, _), result in zip(modality_tasks.items(), completed_tasks):
                if isinstance(result, Exception):
                    self._logger.error(f"Error processing {modality.value}: {result}")
                else:
                    modality_emotions[modality] = result
            
            if not modality_emotions:
                raise ValueError("All modality processing failed")
            
            # 融合情感结果
            fused_emotion = self._fusion_engine.fuse_emotions(modality_emotions)
            
            # 计算处理时间
            processing_time = (utc_now() - start_time).total_seconds()
            
            # 创建多模态情感结果
            result = MultiModalEmotion(
                emotions=modality_emotions,
                fused_emotion=fused_emotion,
                confidence=fused_emotion.confidence,
                processing_time=processing_time
            )
            
            # 更新质量指标
            self._quality_metrics["successful_recognitions"] += 1
            self._update_quality_metrics(result)
            
            self._logger.debug(f"Emotion recognition completed: {fused_emotion.emotion.value}")
            
            return result
            
        except Exception as e:
            self._logger.error(f"Emotion recognition failed: {e}")
            self._quality_metrics["failed_recognitions"] += 1
            
            # 返回默认结果
            processing_time = (utc_now() - start_time).total_seconds()
            default_emotion = EmotionState(
                emotion=EmotionType.NEUTRAL,
                intensity=0.0,
                valence=0.0,
                arousal=0.0,
                dominance=0.0,
                confidence=0.0,
                timestamp=utc_now()
            )
            
            return MultiModalEmotion(
                emotions={},
                fused_emotion=default_emotion,
                confidence=0.0,
                processing_time=processing_time
            )
    
    async def get_recognition_quality(self) -> Dict[str, float]:
        """获取识别质量指标"""
        total = self._quality_metrics["total_recognitions"]
        if total == 0:
            return {
                "accuracy": 0.0,
                "success_rate": 0.0,
                "average_confidence": 0.0,
                "average_processing_time": 0.0,
                "total_recognitions": 0
            }
        
        return {
            "accuracy": self._quality_metrics["successful_recognitions"] / total,
            "success_rate": self._quality_metrics["successful_recognitions"] / total,
            "average_confidence": self._quality_metrics["average_confidence"],
            "average_processing_time": self._quality_metrics["average_processing_time"],
            "total_recognitions": total,
            "modality_usage": self._quality_metrics["modality_usage"].copy()
        }
    
    def _update_quality_metrics(self, result: MultiModalEmotion):
        """更新质量指标"""
        # 更新平均置信度
        total = self._quality_metrics["successful_recognitions"]
        current_avg = self._quality_metrics["average_confidence"]
        self._quality_metrics["average_confidence"] = (
            (current_avg * (total - 1) + result.confidence) / total
        )
        
        # 更新平均处理时间
        current_time_avg = self._quality_metrics["average_processing_time"]
        self._quality_metrics["average_processing_time"] = (
            (current_time_avg * (total - 1) + result.processing_time) / total
        )
    
    def get_supported_modalities(self) -> List[ModalityType]:
        """获取支持的模态"""
        return list(self._processors.keys())
    
    def configure_fusion_weights(self, weights: Dict[ModalityType, float]):
        """配置融合权重"""
        self._fusion_engine.set_modality_weights(weights)
        self._logger.info(f"Updated fusion weights: {weights}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取引擎指标"""
        return {
            **self._quality_metrics,
            "is_running": self._is_running,
            "supported_modalities": [m.value for m in self.get_supported_modalities()],
            "protocol_metrics": self._protocol.get_metrics()
        }
