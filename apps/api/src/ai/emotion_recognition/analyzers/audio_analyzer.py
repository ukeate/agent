"""音频情感分析器"""

import asyncio
from typing import Any, Dict, Optional, List, Tuple, Union
from datetime import datetime
import numpy as np
import torch
import io
import wave
from .base_analyzer import BaseEmotionAnalyzer
from ..models.emotion_models import (
    EmotionResult, EmotionDimension, Modality,
    EmotionCategory, EMOTION_DIMENSIONS
)
from src.core.utils.timezone_utils import utc_now

logger = get_logger(__name__)

class AudioEmotionAnalyzer(BaseEmotionAnalyzer):
    """基于Wav2Vec2的音频情感分析器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化音频情感分析器
        
        Args:
            config: 配置参数
        """
        default_config = {
            "model_name": "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "sampling_rate": 16000,
            "max_duration": 30,  # 最大音频长度(秒)
            "chunk_size": 5,  # 分块大小(秒)
            "feature_extractor": "mfcc"  # 特征提取方法
        }
        
        if config:
            default_config.update(config)
            
        super().__init__(Modality.AUDIO, default_config)
        self.sampling_rate = self.config["sampling_rate"]
        
    async def _load_model(self):
        """加载预训练模型和处理器"""
        try:
            from transformers import (
                Wav2Vec2ForSequenceClassification,
                Wav2Vec2Processor,
                pipeline
            )
            
            # 加载处理器和模型
            self.processor = Wav2Vec2Processor.from_pretrained(
                self.config["model_name"]
            )
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                self.config["model_name"]
            )
            
            # 将模型移到指定设备
            if self.config["device"] == "cuda":
                self.model = self.model.cuda()
            
            # 创建音频分类pipeline
            self.pipeline = pipeline(
                "audio-classification",
                model=self.model,
                feature_extractor=self.processor,
                device=0 if self.config["device"] == "cuda" else -1
            )
            
            logger.info(f"音频情感模型加载成功: {self.config['model_name']}")
            
        except Exception as e:
            logger.error(f"加载音频情感模型失败: {e}")
            self._load_fallback_model()
            
    def _load_fallback_model(self):
        """加载备用模型"""
        try:
            from transformers import pipeline
            
            # 使用更轻量的模型
            self.pipeline = pipeline(
                "audio-classification",
                model="superb/wav2vec2-base-superb-er",
                device=-1  # CPU
            )
            logger.info("使用备用音频情感分析模型")
        except Exception as e:
            logger.error(f"备用模型加载失败: {e}")
            self.pipeline = None
            
    async def preprocess(self, audio_data: Union[np.ndarray, bytes, str]) -> Dict[str, Any]:
        """
        预处理音频数据
        
        Args:
            audio_data: 音频数据(numpy数组、字节或文件路径)
            
        Returns:
            预处理后的数据
        """
        # 加载音频
        audio_array, sample_rate = await self._load_audio(audio_data)
        
        # 重采样到目标采样率
        if sample_rate != self.sampling_rate:
            audio_array = await self._resample_audio(audio_array, sample_rate)
            
        # 提取音频特征
        features = await self._extract_audio_features(audio_array)
        
        return {
            "audio": audio_array,
            "sampling_rate": self.sampling_rate,
            "duration": len(audio_array) / self.sampling_rate,
            "features": features
        }
        
    async def _load_audio(self, audio_data: Union[np.ndarray, bytes, str]) -> Tuple[np.ndarray, int]:
        """加载音频数据"""
        try:
            if isinstance(audio_data, np.ndarray):
                # 已经是numpy数组
                return audio_data, self.sampling_rate
                
            elif isinstance(audio_data, bytes):
                # 字节数据
                with wave.open(io.BytesIO(audio_data), 'rb') as wf:
                    sample_rate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    frames = wf.readframes(wf.getnframes())
                audio_array = self._pcm_to_float(frames, sampwidth)
                if n_channels > 1:
                    audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)
                return audio_array, sample_rate
                
            elif isinstance(audio_data, str):
                # 文件路径
                with wave.open(audio_data, 'rb') as wf:
                    sample_rate = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sampwidth = wf.getsampwidth()
                    frames = wf.readframes(wf.getnframes())
                audio_array = self._pcm_to_float(frames, sampwidth)
                if n_channels > 1:
                    audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)
                return audio_array, sample_rate
                
            else:
                raise ValueError(f"不支持的音频数据类型: {type(audio_data)}")
                
        except Exception as e:
            logger.error(f"加载音频失败: {e}")
            raise

    def _pcm_to_float(self, frames: bytes, sampwidth: int) -> np.ndarray:
        if sampwidth == 1:
            pcm = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
            return (pcm - 128.0) / 128.0
        if sampwidth == 2:
            pcm = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            return pcm / 32768.0
        if sampwidth == 4:
            pcm = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
            return pcm / 2147483648.0
        raise ValueError(f"不支持的wav采样宽度: {sampwidth}")
            
    async def _resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """重采样音频"""
        try:
            if orig_sr == self.sampling_rate:
                return audio
            target_len = int(len(audio) * (self.sampling_rate / orig_sr))
            if target_len <= 0:
                return audio
            x_old = np.arange(len(audio), dtype=np.float32)
            x_new = np.linspace(0, len(audio) - 1, target_len, dtype=np.float32)
            return np.interp(x_new, x_old, audio).astype(np.float32)
        except Exception as e:
            logger.error(f"重采样失败: {e}")
            return audio
            
    async def _extract_audio_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """提取音频特征"""
        try:
            features = {}
            audio = audio.astype(np.float32)

            frame = max(int(self.sampling_rate * 0.05), 1)
            hop = max(int(self.sampling_rate * 0.025), 1)
            frames = [audio[i:i + frame] for i in range(0, len(audio) - frame + 1, hop)]
            if not frames:
                frames = [audio]

            rms_vals = [float(np.sqrt(np.mean(f * f))) for f in frames]
            features["energy_mean"] = float(np.mean(rms_vals))
            features["energy_std"] = float(np.std(rms_vals))

            zcr_vals = []
            for f in frames:
                s = np.sign(f)
                zcr_vals.append(float(np.mean(s[1:] != s[:-1])))
            features["zcr_mean"] = float(np.mean(zcr_vals))
            features["zcr_std"] = float(np.std(zcr_vals))

            spec = np.abs(np.fft.rfft(audio))
            if spec.size > 0 and np.sum(spec) > 0:
                freqs = np.fft.rfftfreq(len(audio), d=1.0 / self.sampling_rate)
                centroid = float(np.sum(freqs * spec) / np.sum(spec))
            else:
                centroid = 0.0
            features["spectral_centroid_mean"] = centroid
            features["spectral_centroid_std"] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return {}
            
    async def analyze(self, audio_data: Union[np.ndarray, bytes, str]) -> EmotionResult:
        """
        分析音频情感
        
        Args:
            audio_data: 音频数据
            
        Returns:
            EmotionResult: 情感分析结果
        """
        if not self.is_initialized:
            await self.initialize()
            
        if not self.validate_input(audio_data):
            raise ValueError("输入音频无效")
            
        # 预处理音频
        preprocessed = await self.preprocess(audio_data)
        
        if self.pipeline:
            try:
                # 如果音频过长，分块处理
                if preprocessed["duration"] > self.config["max_duration"]:
                    result = await self._analyze_long_audio(preprocessed)
                else:
                    # 使用模型进行预测
                    predictions = self.pipeline(preprocessed["audio"])
                    
                    # 后处理结果
                    result = await self.postprocess({
                        "predictions": predictions,
                        "features": preprocessed["features"],
                        "duration": preprocessed["duration"]
                    })
                    
                return result
                
            except Exception as e:
                logger.error(f"音频情感分析出错: {e}")
                return self._create_neutral_result(preprocessed["features"])
        else:
            return self._create_neutral_result(preprocessed["features"])
            
    async def _analyze_long_audio(self, preprocessed: Dict[str, Any]) -> EmotionResult:
        """分析长音频(分块处理)"""
        audio = preprocessed["audio"]
        chunk_size = int(self.config["chunk_size"] * self.sampling_rate)
        chunks = [audio[i:i+chunk_size] for i in range(0, len(audio), chunk_size)]
        
        chunk_results = []
        for chunk in chunks:
            if len(chunk) < self.sampling_rate:  # 跳过太短的片段
                continue
                
            predictions = self.pipeline(chunk)
            chunk_results.append(predictions)
            
        # 聚合所有块的结果
        aggregated = await self._aggregate_chunk_results(chunk_results)
        
        return await self.postprocess({
            "predictions": aggregated,
            "features": preprocessed["features"],
            "duration": preprocessed["duration"]
        })
        
    async def _aggregate_chunk_results(self, chunk_results: List[Any]) -> List[Dict]:
        """聚合分块结果"""
        if not chunk_results:
            return []
            
        # 统计各情感出现频率和平均分数
        emotion_scores = {}
        
        for chunk in chunk_results:
            for pred in chunk:
                label = pred["label"]
                score = pred["score"]
                
                if label not in emotion_scores:
                    emotion_scores[label] = []
                emotion_scores[label].append(score)
                
        # 计算平均分数并排序
        aggregated = []
        for label, scores in emotion_scores.items():
            aggregated.append({
                "label": label,
                "score": np.mean(scores)
            })
            
        return sorted(aggregated, key=lambda x: x["score"], reverse=True)
        
    async def postprocess(self, raw_output: Dict[str, Any]) -> EmotionResult:
        """
        后处理模型输出
        
        Args:
            raw_output: 模型原始输出
            
        Returns:
            EmotionResult: 格式化的情感结果
        """
        predictions = raw_output.get("predictions", [])
        features = raw_output.get("features", {})
        duration = raw_output.get("duration", 0)
        
        if not predictions:
            return self._create_neutral_result(features)
            
        # 主要情感
        primary = predictions[0] if isinstance(predictions, list) else predictions
        emotion_label = self._standardize_audio_emotion_label(primary["label"])
        confidence = primary["score"]
        
        # 次要情感
        sub_emotions = []
        if isinstance(predictions, list):
            for pred in predictions[1:5]:
                if pred["score"] > 0.1:
                    sub_emotions.append((
                        self._standardize_audio_emotion_label(pred["label"]),
                        pred["score"]
                    ))
                    
        # 基于音频特征计算强度
        intensity = self._calculate_audio_intensity(confidence, features)
        
        # 映射到VAD维度
        dimension = self.map_to_dimension(emotion_label, intensity)
        
        return EmotionResult(
            emotion=emotion_label,
            confidence=confidence,
            intensity=intensity,
            timestamp=utc_now(),
            modality=str(self.modality.value),
            details={
                "duration": duration,
                "pitch_mean": features.get("pitch_mean", 0),
                "energy_mean": features.get("energy_mean", 0),
                "spectral_centroid_mean": features.get("spectral_centroid_mean", 0),
                "raw_predictions": predictions[:3] if isinstance(predictions, list) else [predictions]
            },
            sub_emotions=sub_emotions,
            dimension=dimension
        )
        
    def _standardize_audio_emotion_label(self, label: str) -> str:
        """标准化音频情感标签"""
        label = label.lower().strip()
        
        # 映射到标准情感类别
        audio_emotion_mapping = {
            "happy": EmotionCategory.HAPPINESS,
            "happiness": EmotionCategory.HAPPINESS,
            "joy": EmotionCategory.JOY,
            "excited": EmotionCategory.EXCITEMENT,
            "sad": EmotionCategory.SADNESS,
            "sadness": EmotionCategory.SADNESS,
            "angry": EmotionCategory.ANGER,
            "anger": EmotionCategory.ANGER,
            "fear": EmotionCategory.FEAR,
            "fearful": EmotionCategory.FEAR,
            "disgust": EmotionCategory.DISGUST,
            "disgusted": EmotionCategory.DISGUST,
            "surprise": EmotionCategory.SURPRISE,
            "surprised": EmotionCategory.SURPRISE,
            "neutral": EmotionCategory.NEUTRAL,
            "calm": EmotionCategory.CONTENTMENT
        }
        
        for key, value in audio_emotion_mapping.items():
            if key in label:
                return value.value
                
        return label
        
    def _calculate_audio_intensity(self, confidence: float, features: Dict[str, Any]) -> float:
        """基于音频特征计算情感强度"""
        intensity = confidence
        
        # 能量影响强度
        if "energy_mean" in features:
            energy_factor = min(features["energy_mean"] * 2, 1.0)
            intensity = (intensity + energy_factor) / 2
            
        # 音高变化影响强度
        if "pitch_std" in features and features["pitch_std"] > 0:
            pitch_factor = min(features["pitch_std"] / 100, 1.0)
            intensity = (intensity + pitch_factor) / 2
            
        return min(max(intensity, 0.0), 1.0)
        
    def _create_neutral_result(self, features: Dict[str, Any]) -> EmotionResult:
        """创建中性情感结果"""
        return EmotionResult(
            emotion=EmotionCategory.NEUTRAL.value,
            confidence=0.5,
            intensity=0.3,
            timestamp=utc_now(),
            modality=str(self.modality.value),
            details=features,
            dimension=EMOTION_DIMENSIONS[EmotionCategory.NEUTRAL]
        )
        
    def validate_input(self, audio_data: Any) -> bool:
        """验证输入音频"""
        if audio_data is None:
            return False
            
        # 检查类型
        if not isinstance(audio_data, (np.ndarray, bytes, str)):
            return False
            
        return True
        
    async def analyze_realtime_stream(
        self,
        audio_stream: asyncio.Queue,
        callback: callable
    ):
        """
        实时音频流分析
        
        Args:
            audio_stream: 音频流队列
            callback: 结果回调函数
        """
        buffer = []
        buffer_duration = 3.0  # 3秒缓冲
        buffer_size = int(buffer_duration * self.sampling_rate)
        
        while True:
            try:
                # 从队列获取音频块
                chunk = await audio_stream.get()
                
                if chunk is None:  # 结束信号
                    break
                    
                buffer.extend(chunk)
                
                # 当缓冲区满时进行分析
                if len(buffer) >= buffer_size:
                    audio_array = np.array(buffer[:buffer_size])
                    
                    # 异步分析
                    result = await self.analyze(audio_array)
                    
                    # 调用回调
                    await callback(result)
                    
                    # 滑动窗口
                    buffer = buffer[buffer_size // 2:]
                    
            except Exception as e:
                logger.error(f"实时分析出错: {e}")
from src.core.logging import get_logger
