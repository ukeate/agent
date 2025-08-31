"""音频情感分析器"""

import asyncio
from typing import Any, Dict, Optional, List, Tuple, Union
from datetime import datetime
import logging
import numpy as np
import torch
import io

from .base_analyzer import BaseEmotionAnalyzer
from ..models.emotion_models import (
    EmotionResult, EmotionDimension, Modality,
    EmotionCategory, EMOTION_DIMENSIONS
)

logger = logging.getLogger(__name__)


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
            import librosa
            
            if isinstance(audio_data, np.ndarray):
                # 已经是numpy数组
                return audio_data, self.sampling_rate
                
            elif isinstance(audio_data, bytes):
                # 字节数据
                audio_array, sample_rate = librosa.load(
                    io.BytesIO(audio_data),
                    sr=None
                )
                return audio_array, sample_rate
                
            elif isinstance(audio_data, str):
                # 文件路径
                audio_array, sample_rate = librosa.load(audio_data, sr=None)
                return audio_array, sample_rate
                
            else:
                raise ValueError(f"不支持的音频数据类型: {type(audio_data)}")
                
        except ImportError:
            logger.error("需要安装librosa库: pip install librosa")
            raise
        except Exception as e:
            logger.error(f"加载音频失败: {e}")
            raise
            
    async def _resample_audio(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """重采样音频"""
        try:
            import librosa
            
            resampled = librosa.resample(
                audio,
                orig_sr=orig_sr,
                target_sr=self.sampling_rate
            )
            return resampled
        except Exception as e:
            logger.error(f"重采样失败: {e}")
            return audio
            
    async def _extract_audio_features(self, audio: np.ndarray) -> Dict[str, Any]:
        """提取音频特征"""
        try:
            import librosa
            
            features = {}
            
            # MFCC特征
            if "mfcc" in self.config["feature_extractor"]:
                mfccs = librosa.feature.mfcc(
                    y=audio,
                    sr=self.sampling_rate,
                    n_mfcc=13
                )
                features["mfcc_mean"] = np.mean(mfccs, axis=1).tolist()
                features["mfcc_std"] = np.std(mfccs, axis=1).tolist()
                
            # 音高特征
            pitches, magnitudes = librosa.piptrack(
                y=audio,
                sr=self.sampling_rate
            )
            features["pitch_mean"] = float(np.mean(pitches[pitches > 0]))
            features["pitch_std"] = float(np.std(pitches[pitches > 0]))
            
            # 能量特征
            rms = librosa.feature.rms(y=audio)[0]
            features["energy_mean"] = float(np.mean(rms))
            features["energy_std"] = float(np.std(rms))
            
            # 语速特征(通过零交叉率估计)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            features["zcr_mean"] = float(np.mean(zcr))
            features["zcr_std"] = float(np.std(zcr))
            
            # 频谱特征
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio,
                sr=self.sampling_rate
            )[0]
            features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
            features["spectral_centroid_std"] = float(np.std(spectral_centroids))
            
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
            timestamp=datetime.now(),
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
            timestamp=datetime.now(),
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