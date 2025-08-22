# Epic 11: 高级情感智能系统

**Epic ID**: EPIC-011-EMOTIONAL-INTELLIGENCE  
**优先级**: 高 (P1)  
**预估工期**: 8-10周  
**负责团队**: AI团队 + 前端团队  
**创建日期**: 2025-08-19

## 📋 Epic概述

构建全方位的情感智能系统，实现AI Agent的情感理解、情感生成、情感记忆和情感适应能力，包括多模态情感识别、情感状态建模、共情响应生成和长期情感关系管理，让AI具备接近人类的情感交互体验。

### 🎯 业务价值
- **情感共鸣**: 让AI能够理解和响应用户的情感需求
- **个性化体验**: 基于情感状态提供更贴合的交互体验  
- **长期关系**: 建立和维护持久的用户情感关系
- **技术前沿**: 掌握情感计算和多模态AI的最新技术

## 🚀 核心功能清单

### 1. **多模态情感识别引擎**
- 文本情感分析和细粒度情感分类
- 语音语调情感识别和韵律分析
- 面部表情和肢体语言识别
- 生理信号情感状态推断

### 2. **情感状态建模系统**
- 多维度情感状态空间建模
- 情感强度和时间动态跟踪
- 个性化情感画像构建
- 情感状态转换模型

### 3. **共情响应生成器**
- 情感感知的回复生成
- 多样化情感表达策略
- 情感调节和安慰机制
- 适应性情感镜像

### 4. **情感记忆管理**
- 长期情感交互历史记录
- 情感事件关联分析
- 个人情感偏好学习
- 情感触发模式识别

### 5. **情感智能决策引擎**
- 基于情感状态的行为选择
- 情感风险评估和预警
- 情感干预策略制定
- 情感健康监测

### 6. **社交情感理解**
- 群体情感氛围感知
- 人际关系情感动态
- 社交场景情感适应
- 文化背景情感差异

## 🏗️ 用户故事分解

### Story 11.1: 多模态情感识别引擎
**优先级**: P1 | **工期**: 2-3周
- 集成文本、语音、图像的情感识别模型
- 实现实时多模态情感融合算法
- 构建细粒度情感分类体系
- 创建情感识别准确性评估框架

### Story 11.2: 情感状态建模系统
**优先级**: P1 | **工期**: 2周
- 设计多维情感状态空间模型
- 实现情感状态动态跟踪算法
- 构建个性化情感画像系统
- 创建情感状态可视化界面

### Story 11.3: 共情响应生成器
**优先级**: P1 | **工期**: 3周
- 实现情感感知的回复生成模型
- 构建多样化情感表达策略库
- 集成情感调节和安慰机制
- 实现适应性情感镜像算法

### Story 11.4: 情感记忆管理系统
**优先级**: P2 | **工期**: 2周
- 构建长期情感交互存储系统
- 实现情感事件关联分析
- 建立个人情感偏好学习算法
- 创建情感触发模式识别

### Story 11.5: 情感智能决策引擎
**优先级**: P1 | **工期**: 2-3周
- 实现基于情感的行为选择算法
- 构建情感风险评估系统
- 建立情感干预策略库
- 集成情感健康监测功能

### Story 11.6: 社交情感理解系统
**优先级**: P2 | **工期**: 1-2周
- 实现群体情感氛围感知
- 构建人际关系情感分析
- 建立社交场景适应机制
- 集成文化背景情感处理

### Story 11.7: 系统集成和用户界面
**优先级**: P1 | **工期**: 2周
- 端到端情感智能系统集成
- 创建情感交互用户界面
- 实现情感状态可视化
- 构建情感分析调试工具

## 🎯 成功标准 (Definition of Done)

### 技术指标
- ✅ **情感识别准确率**: 文本>92%, 语音>88%, 图像>85%
- ✅ **多模态融合精度**: 综合情感识别准确率>95%
- ✅ **响应时间**: 情感分析和响应生成<500ms
- ✅ **情感状态跟踪**: 情感变化检测精度>90%
- ✅ **共情质量**: 用户情感满意度评分>4.2/5.0

### 功能指标
- ✅ **情感类别覆盖**: 支持20+基础情感和100+细粒度情感
- ✅ **情感表达多样性**: 每种情感支持10+不同表达方式
- ✅ **记忆容量**: 支持1年+长期情感交互历史
- ✅ **个性化程度**: 3次交互内建立基础情感画像
- ✅ **多语言支持**: 支持5种主要语言的情感处理

### 用户体验指标
- ✅ **情感共鸣度**: 用户感受到AI理解其情感>85%
- ✅ **交互自然度**: 情感对话自然度评分>4.0/5.0
- ✅ **情感支持效果**: 负面情感缓解有效率>75%
- ✅ **长期满意度**: 情感关系建立满意度>4.1/5.0

## 🔧 技术实现亮点

### 多模态情感识别引擎
```python
import asyncio
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModel, pipeline
import cv2
import librosa
from scipy.signal import find_peaks
import logging

@dataclass
class EmotionResult:
    emotion: str
    confidence: float
    intensity: float
    timestamp: datetime
    modality: str
    details: Dict[str, Any]

@dataclass
class MultiModalEmotion:
    primary_emotion: str
    secondary_emotions: List[Tuple[str, float]]
    overall_confidence: float
    intensity_level: float
    valence: float  # 正负情感倾向 (-1 to 1)
    arousal: float  # 情感激活度 (0 to 1)
    dominance: float  # 情感主导性 (0 to 1)
    modality_weights: Dict[str, float]
    timestamp: datetime

class TextEmotionAnalyzer:
    """文本情感分析器"""
    
    def __init__(self, model_name: str = "j-hartmann/emotion-english-distilroberta-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # 情感分类器
        self.classifier = pipeline(
            "text-classification",
            model=model_name,
            return_all_scores=True
        )
        
        # 情感维度分析器
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            return_all_scores=True
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze_emotion(self, text: str) -> EmotionResult:
        """分析文本情感"""
        
        try:
            # 基础情感分类
            emotions = await asyncio.get_event_loop().run_in_executor(
                None, self.classifier, text
            )
            
            # 情感维度分析
            sentiment_scores = await asyncio.get_event_loop().run_in_executor(
                None, self.sentiment_analyzer, text
            )
            
            # 获取主要情感
            primary_emotion = max(emotions[0], key=lambda x: x['score'])
            
            # 计算情感强度
            intensity = self._calculate_text_intensity(text, emotions[0])
            
            # 计算情感维度
            valence, arousal, dominance = self._calculate_emotion_dimensions(
                emotions[0], sentiment_scores[0]
            )
            
            return EmotionResult(
                emotion=primary_emotion['label'].lower(),
                confidence=primary_emotion['score'],
                intensity=intensity,
                timestamp=datetime.now(),
                modality="text",
                details={
                    "all_emotions": emotions[0],
                    "sentiment_scores": sentiment_scores[0],
                    "valence": valence,
                    "arousal": arousal,
                    "dominance": dominance,
                    "text_length": len(text),
                    "word_count": len(text.split())
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in text emotion analysis: {e}")
            return EmotionResult(
                emotion="neutral",
                confidence=0.5,
                intensity=0.5,
                timestamp=datetime.now(),
                modality="text",
                details={"error": str(e)}
            )
    
    def _calculate_text_intensity(self, text: str, emotions: List[Dict]) -> float:
        """计算文本情感强度"""
        
        # 基于多个因子计算强度
        factors = []
        
        # 情感词强度
        emotion_words = [
            "extremely", "incredibly", "absolutely", "totally", "completely",
            "very", "really", "quite", "rather", "somewhat", "slightly"
        ]
        
        intensity_multiplier = 1.0
        for word in emotion_words:
            if word in text.lower():
                if word in ["extremely", "incredibly", "absolutely"]:
                    intensity_multiplier *= 1.5
                elif word in ["very", "really"]:
                    intensity_multiplier *= 1.3
                elif word in ["quite", "rather"]:
                    intensity_multiplier *= 1.1
                else:
                    intensity_multiplier *= 1.05
        
        # 标点符号强度
        exclamation_count = text.count('!')
        question_count = text.count('?')
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        
        punctuation_intensity = min(1.0 + exclamation_count * 0.1 + caps_ratio * 0.5, 2.0)
        
        # 情感分数强度
        emotion_confidence = max(emotion['score'] for emotion in emotions)
        
        # 综合强度计算
        final_intensity = min(
            (emotion_confidence * intensity_multiplier * punctuation_intensity) / 2,
            1.0
        )
        
        return final_intensity
    
    def _calculate_emotion_dimensions(
        self, 
        emotions: List[Dict], 
        sentiments: List[Dict]
    ) -> Tuple[float, float, float]:
        """计算情感维度：效价、唤醒度、支配性"""
        
        # 情感到维度的映射
        emotion_dimensions = {
            'joy': (0.8, 0.7, 0.6),
            'happiness': (0.9, 0.6, 0.7),
            'sadness': (-0.7, 0.4, 0.3),
            'anger': (-0.6, 0.9, 0.8),
            'fear': (-0.8, 0.8, 0.2),
            'surprise': (0.2, 0.8, 0.5),
            'disgust': (-0.7, 0.5, 0.6),
            'neutral': (0.0, 0.3, 0.5),
            'love': (0.9, 0.5, 0.6),
            'excitement': (0.8, 0.9, 0.7)
        }
        
        # 加权计算维度值
        valence = arousal = dominance = 0.0
        total_weight = 0.0
        
        for emotion in emotions:
            emotion_name = emotion['label'].lower()
            confidence = emotion['score']
            
            if emotion_name in emotion_dimensions:
                dims = emotion_dimensions[emotion_name]
                valence += dims[0] * confidence
                arousal += dims[1] * confidence
                dominance += dims[2] * confidence
                total_weight += confidence
        
        if total_weight > 0:
            valence /= total_weight
            arousal /= total_weight
            dominance /= total_weight
        
        # 结合情感极性调整效价
        for sentiment in sentiments:
            if sentiment['label'] == 'LABEL_2':  # Positive
                valence = max(valence, sentiment['score'] * 0.5)
            elif sentiment['label'] == 'LABEL_0':  # Negative  
                valence = min(valence, -sentiment['score'] * 0.5)
        
        return valence, arousal, dominance

class AudioEmotionAnalyzer:
    """语音情感分析器"""
    
    def __init__(self):
        # 使用预训练的语音情感识别模型
        self.emotion_classifier = pipeline(
            "audio-classification",
            model="superb/wav2vec2-base-superb-er"
        )
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze_emotion(self, audio_data: np.ndarray, sample_rate: int = 16000) -> EmotionResult:
        """分析语音情感"""
        
        try:
            # 提取语音特征
            features = await self._extract_audio_features(audio_data, sample_rate)
            
            # 情感分类
            emotions = await asyncio.get_event_loop().run_in_executor(
                None, self.emotion_classifier, audio_data
            )
            
            # 计算情感强度
            intensity = self._calculate_audio_intensity(features)
            
            # 获取主要情感
            primary_emotion = max(emotions, key=lambda x: x['score'])
            
            return EmotionResult(
                emotion=primary_emotion['label'].lower(),
                confidence=primary_emotion['score'],
                intensity=intensity,
                timestamp=datetime.now(),
                modality="audio",
                details={
                    "all_emotions": emotions,
                    "features": features,
                    "sample_rate": sample_rate,
                    "duration": len(audio_data) / sample_rate
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in audio emotion analysis: {e}")
            return EmotionResult(
                emotion="neutral",
                confidence=0.5,
                intensity=0.5,
                timestamp=datetime.now(),
                modality="audio",
                details={"error": str(e)}
            )
    
    async def _extract_audio_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """提取音频特征"""
        
        features = {}
        
        # MFCC特征
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs)
        features['mfcc_std'] = np.std(mfccs)
        
        # 频谱质心
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # 过零率
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
        features['zcr_mean'] = np.mean(zero_crossing_rate)
        features['zcr_std'] = np.std(zero_crossing_rate)
        
        # 音高特征
        pitches, magnitudes = librosa.piptrack(y=audio_data, sr=sample_rate)
        pitch_values = []
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
            features['pitch_range'] = max(pitch_values) - min(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0
            features['pitch_range'] = 0
        
        # 能量特征
        rms_energy = librosa.feature.rms(y=audio_data)[0]
        features['energy_mean'] = np.mean(rms_energy)
        features['energy_std'] = np.std(rms_energy)
        
        # 语音速率估计
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
        features['tempo'] = tempo
        
        return features
    
    def _calculate_audio_intensity(self, features: Dict[str, float]) -> float:
        """计算语音情感强度"""
        
        # 基于多个语音特征计算情感强度
        intensity_factors = []
        
        # 能量强度
        energy_intensity = min(features['energy_mean'] * 2, 1.0)
        intensity_factors.append(energy_intensity)
        
        # 音高变化强度
        pitch_intensity = min(features['pitch_std'] / 50.0, 1.0)  # 归一化
        intensity_factors.append(pitch_intensity)
        
        # 语音速率强度
        tempo_intensity = min(abs(features['tempo'] - 120) / 100.0, 1.0)  # 与正常语速的偏差
        intensity_factors.append(tempo_intensity)
        
        # 频谱变化强度
        spectral_intensity = min(features['spectral_centroid_std'] / 1000.0, 1.0)
        intensity_factors.append(spectral_intensity)
        
        # 综合强度
        overall_intensity = np.mean(intensity_factors)
        
        return max(0.1, min(1.0, overall_intensity))

class VisualEmotionAnalyzer:
    """视觉情感分析器"""
    
    def __init__(self):
        # 使用预训练的面部表情识别模型
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 表情分类器(这里使用placeholder，实际应该加载真实模型)
        self.expression_classifier = None  # 实际项目中应该加载FER模型
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze_emotion(self, image: np.ndarray) -> EmotionResult:
        """分析图像情感"""
        
        try:
            # 检测人脸
            faces = self._detect_faces(image)
            
            if len(faces) == 0:
                return EmotionResult(
                    emotion="neutral",
                    confidence=0.3,
                    intensity=0.3,
                    timestamp=datetime.now(),
                    modality="visual",
                    details={"faces_detected": 0, "error": "No faces detected"}
                )
            
            # 分析主要人脸的表情
            main_face = faces[0]  # 使用最大的人脸
            face_roi = self._extract_face_roi(image, main_face)
            
            # 表情识别 (简化实现)
            emotion_scores = await self._classify_expression(face_roi)
            
            # 计算情感强度
            intensity = self._calculate_visual_intensity(face_roi, main_face)
            
            # 获取主要情感
            primary_emotion = max(emotion_scores, key=lambda x: x['score'])
            
            return EmotionResult(
                emotion=primary_emotion['label'].lower(),
                confidence=primary_emotion['score'],
                intensity=intensity,
                timestamp=datetime.now(),
                modality="visual",
                details={
                    "all_emotions": emotion_scores,
                    "faces_detected": len(faces),
                    "face_size": main_face[2] * main_face[3],  # width * height
                    "face_position": main_face[:2]
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in visual emotion analysis: {e}")
            return EmotionResult(
                emotion="neutral",
                confidence=0.5,
                intensity=0.5,
                timestamp=datetime.now(),
                modality="visual",
                details={"error": str(e)}
            )
    
    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """检测人脸"""
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # 按面积排序，返回最大的几个
        faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
        
        return faces[:3]  # 最多返回3个人脸
    
    def _extract_face_roi(self, image: np.ndarray, face_rect: Tuple[int, int, int, int]) -> np.ndarray:
        """提取人脸区域"""
        
        x, y, w, h = face_rect
        face_roi = image[y:y+h, x:x+w]
        
        # 调整大小到标准尺寸
        face_roi = cv2.resize(face_roi, (224, 224))
        
        return face_roi
    
    async def _classify_expression(self, face_roi: np.ndarray) -> List[Dict[str, Any]]:
        """表情分类 (简化实现)"""
        
        # 这里是简化的实现，实际应该使用训练好的深度学习模型
        # 如 FER2013, AffectNet 等数据集训练的模型
        
        # 模拟表情分类结果
        emotion_labels = ['happy', 'sad', 'angry', 'surprised', 'fearful', 'disgusted', 'neutral']
        
        # 基于简单的像素特征做粗略分类 (实际项目中应该使用深度学习模型)
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # 计算简单的统计特征
        mean_intensity = np.mean(gray_roi)
        std_intensity = np.std(gray_roi)
        
        # 模拟分类结果
        scores = np.random.dirichlet(np.ones(len(emotion_labels)))
        
        # 基于强度调整某些情感的概率
        if std_intensity > 30:  # 高对比度，可能是强烈表情
            scores[0] *= 1.2  # happy
            scores[2] *= 1.2  # angry
        
        emotion_scores = [
            {'label': label, 'score': score}
            for label, score in zip(emotion_labels, scores)
        ]
        
        return sorted(emotion_scores, key=lambda x: x['score'], reverse=True)
    
    def _calculate_visual_intensity(self, face_roi: np.ndarray, face_rect: Tuple[int, int, int, int]) -> float:
        """计算视觉情感强度"""
        
        # 基于多个视觉特征计算强度
        intensity_factors = []
        
        # 对比度强度 (表情变化通常伴随对比度变化)
        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray_roi) / 255.0
        intensity_factors.append(min(contrast * 2, 1.0))
        
        # 边缘强度 (表情变化会产生更多边缘)
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        intensity_factors.append(min(edge_density * 10, 1.0))
        
        # 人脸大小 (更大的人脸通常表达更明显)
        face_area = face_rect[2] * face_rect[3]
        size_intensity = min(face_area / (200 * 200), 1.0)  # 相对于标准大小
        intensity_factors.append(size_intensity)
        
        # 综合强度
        overall_intensity = np.mean(intensity_factors)
        
        return max(0.2, min(1.0, overall_intensity))

class MultiModalEmotionFusion:
    """多模态情感融合"""
    
    def __init__(self):
        self.text_analyzer = TextEmotionAnalyzer()
        self.audio_analyzer = AudioEmotionAnalyzer()
        self.visual_analyzer = VisualEmotionAnalyzer()
        
        # 模态权重配置 (可动态调整)
        self.default_weights = {
            "text": 0.4,
            "audio": 0.35,
            "visual": 0.25
        }
        
        # 情感标签标准化映射
        self.emotion_mapping = {
            'joy': 'happiness',
            'happy': 'happiness',
            'pleased': 'happiness',
            'sad': 'sadness',
            'unhappy': 'sadness',
            'angry': 'anger',
            'mad': 'anger',
            'fearful': 'fear',
            'afraid': 'fear',
            'surprised': 'surprise',
            'amazed': 'surprise',
            'disgusted': 'disgust',
            'revolted': 'disgust',
            'neutral': 'neutral'
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def analyze_multimodal_emotion(
        self,
        text: Optional[str] = None,
        audio_data: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        sample_rate: int = 16000,
        custom_weights: Optional[Dict[str, float]] = None
    ) -> MultiModalEmotion:
        """多模态情感分析"""
        
        emotion_results = {}
        available_modalities = []
        
        # 分析各个模态
        if text:
            emotion_results["text"] = await self.text_analyzer.analyze_emotion(text)
            available_modalities.append("text")
        
        if audio_data is not None:
            emotion_results["audio"] = await self.audio_analyzer.analyze_emotion(audio_data, sample_rate)
            available_modalities.append("audio")
        
        if image is not None:
            emotion_results["visual"] = await self.visual_analyzer.analyze_emotion(image)
            available_modalities.append("visual")
        
        if not emotion_results:
            # 没有有效输入
            return MultiModalEmotion(
                primary_emotion="neutral",
                secondary_emotions=[],
                overall_confidence=0.3,
                intensity_level=0.3,
                valence=0.0,
                arousal=0.3,
                dominance=0.5,
                modality_weights={},
                timestamp=datetime.now()
            )
        
        # 融合情感分析结果
        fused_emotion = await self._fuse_emotions(
            emotion_results, 
            available_modalities,
            custom_weights or self.default_weights
        )
        
        return fused_emotion
    
    async def _fuse_emotions(
        self,
        emotion_results: Dict[str, EmotionResult],
        available_modalities: List[str],
        weights: Dict[str, float]
    ) -> MultiModalEmotion:
        """融合多模态情感结果"""
        
        # 规范化权重
        total_weight = sum(weights[modality] for modality in available_modalities if modality in weights)
        if total_weight == 0:
            total_weight = 1.0
        
        normalized_weights = {
            modality: weights.get(modality, 0) / total_weight
            for modality in available_modalities
        }
        
        # 收集所有情感
        all_emotions = {}
        total_intensity = 0.0
        total_confidence = 0.0
        
        # 计算情感维度
        total_valence = 0.0
        total_arousal = 0.0
        total_dominance = 0.0
        
        for modality, result in emotion_results.items():
            weight = normalized_weights.get(modality, 0)
            
            # 标准化情感标签
            normalized_emotion = self._normalize_emotion_label(result.emotion)
            
            # 累加情感分数
            if normalized_emotion not in all_emotions:
                all_emotions[normalized_emotion] = 0.0
            
            all_emotions[normalized_emotion] += result.confidence * weight
            
            # 累加强度和置信度
            total_intensity += result.intensity * weight
            total_confidence += result.confidence * weight
            
            # 累加情感维度 (如果有的话)
            if modality == "text" and "valence" in result.details:
                total_valence += result.details["valence"] * weight
                total_arousal += result.details["arousal"] * weight
                total_dominance += result.details["dominance"] * weight
            else:
                # 从情感映射获取维度值
                dims = self._get_emotion_dimensions(normalized_emotion)
                total_valence += dims[0] * weight
                total_arousal += dims[1] * weight
                total_dominance += dims[2] * weight
        
        # 确定主要和次要情感
        sorted_emotions = sorted(all_emotions.items(), key=lambda x: x[1], reverse=True)
        
        primary_emotion = sorted_emotions[0][0] if sorted_emotions else "neutral"
        primary_confidence = sorted_emotions[0][1] if sorted_emotions else 0.5
        
        secondary_emotions = [
            (emotion, confidence) for emotion, confidence in sorted_emotions[1:5]
            if confidence > 0.1  # 只保留置信度较高的次要情感
        ]
        
        return MultiModalEmotion(
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            overall_confidence=total_confidence,
            intensity_level=total_intensity,
            valence=max(-1.0, min(1.0, total_valence)),
            arousal=max(0.0, min(1.0, total_arousal)),
            dominance=max(0.0, min(1.0, total_dominance)),
            modality_weights=normalized_weights,
            timestamp=datetime.now()
        )
    
    def _normalize_emotion_label(self, emotion: str) -> str:
        """标准化情感标签"""
        return self.emotion_mapping.get(emotion.lower(), emotion.lower())
    
    def _get_emotion_dimensions(self, emotion: str) -> Tuple[float, float, float]:
        """获取情感的维度值 (效价, 唤醒度, 支配性)"""
        
        dimension_map = {
            'happiness': (0.8, 0.6, 0.7),
            'sadness': (-0.7, 0.4, 0.3),
            'anger': (-0.6, 0.9, 0.8),
            'fear': (-0.8, 0.8, 0.2),
            'surprise': (0.2, 0.8, 0.5),
            'disgust': (-0.7, 0.5, 0.6),
            'neutral': (0.0, 0.3, 0.5)
        }
        
        return dimension_map.get(emotion, (0.0, 0.3, 0.5))
    
    def update_modality_weights(self, new_weights: Dict[str, float]):
        """更新模态权重"""
        
        # 验证权重
        total = sum(new_weights.values())
        if total > 0:
            self.default_weights.update({
                k: v / total for k, v in new_weights.items()
            })
            self.logger.info(f"Updated modality weights: {self.default_weights}")
```

### 情感状态建模系统
```python
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import json
import asyncio
from scipy.spatial.distance import euclidean, cosine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging

class EmotionIntensity(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 1.0

@dataclass
class EmotionState:
    emotion: str
    intensity: float
    valence: float
    arousal: float
    dominance: float
    confidence: float
    timestamp: datetime
    duration: Optional[timedelta] = None
    triggers: List[str] = None
    context: Dict[str, Any] = None

@dataclass
class PersonalityProfile:
    user_id: str
    emotional_traits: Dict[str, float]  # Big Five + 其他情感特质
    baseline_emotions: Dict[str, float]  # 基线情感状态
    emotion_volatility: float  # 情感波动性
    recovery_rate: float  # 情感恢复速度
    dominant_emotions: List[str]  # 主导情感
    trigger_patterns: Dict[str, List[str]]  # 情感触发模式
    created_at: datetime
    updated_at: datetime

class EmotionStateModel:
    """情感状态建模系统"""
    
    def __init__(self):
        # 情感状态空间维度
        self.dimensions = ['valence', 'arousal', 'dominance']
        
        # 情感状态历史
        self.emotion_history: Dict[str, List[EmotionState]] = {}
        
        # 个性化情感画像
        self.personality_profiles: Dict[str, PersonalityProfile] = {}
        
        # 情感转换模型
        self.transition_matrices: Dict[str, np.ndarray] = {}
        
        # 情感聚类模型
        self.emotion_clusters: Dict[str, Any] = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def update_emotion_state(
        self,
        user_id: str,
        emotion: MultiModalEmotion,
        context: Dict[str, Any] = None
    ) -> EmotionState:
        """更新用户情感状态"""
        
        # 创建新的情感状态
        emotion_state = EmotionState(
            emotion=emotion.primary_emotion,
            intensity=emotion.intensity_level,
            valence=emotion.valence,
            arousal=emotion.arousal,
            dominance=emotion.dominance,
            confidence=emotion.overall_confidence,
            timestamp=emotion.timestamp,
            triggers=context.get('triggers', []) if context else [],
            context=context or {}
        )
        
        # 添加到历史记录
        if user_id not in self.emotion_history:
            self.emotion_history[user_id] = []
        
        self.emotion_history[user_id].append(emotion_state)
        
        # 保持历史记录限制
        max_history = 1000
        if len(self.emotion_history[user_id]) > max_history:
            self.emotion_history[user_id] = self.emotion_history[user_id][-max_history:]
        
        # 更新个性化画像
        await self._update_personality_profile(user_id, emotion_state)
        
        # 更新转换模型
        await self._update_transition_model(user_id, emotion_state)
        
        self.logger.info(f"Updated emotion state for user {user_id}: {emotion.primary_emotion}")
        
        return emotion_state
    
    async def get_current_emotion_state(self, user_id: str) -> Optional[EmotionState]:
        """获取当前情感状态"""
        
        if user_id not in self.emotion_history or not self.emotion_history[user_id]:
            return None
        
        return self.emotion_history[user_id][-1]
    
    async def predict_emotion_trajectory(
        self,
        user_id: str,
        time_horizon_minutes: int = 60
    ) -> List[Tuple[datetime, str, float]]:
        """预测情感轨迹"""
        
        current_state = await self.get_current_emotion_state(user_id)
        if not current_state:
            return []
        
        # 获取转换概率矩阵
        transition_matrix = self.transition_matrices.get(user_id)
        if transition_matrix is None:
            return []
        
        # 情感标签到索引的映射
        emotions = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
        emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}
        idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}
        
        # 当前情感索引
        current_idx = emotion_to_idx.get(current_state.emotion, emotion_to_idx['neutral'])
        
        # 预测未来情感状态
        predictions = []
        current_time = datetime.now()
        time_step = timedelta(minutes=5)  # 5分钟间隔预测
        
        current_prob_vector = np.zeros(len(emotions))
        current_prob_vector[current_idx] = 1.0
        
        for step in range(time_horizon_minutes // 5):
            # 应用转换矩阵
            current_prob_vector = current_prob_vector @ transition_matrix
            
            # 找到最可能的情感
            predicted_idx = np.argmax(current_prob_vector)
            predicted_emotion = idx_to_emotion[predicted_idx]
            confidence = current_prob_vector[predicted_idx]
            
            prediction_time = current_time + time_step * (step + 1)
            predictions.append((prediction_time, predicted_emotion, confidence))
        
        return predictions
    
    async def analyze_emotion_patterns(self, user_id: str) -> Dict[str, Any]:
        """分析用户情感模式"""
        
        if user_id not in self.emotion_history:
            return {}
        
        history = self.emotion_history[user_id]
        if len(history) < 10:  # 需要足够的数据
            return {"error": "Insufficient data for pattern analysis"}
        
        analysis = {}
        
        # 情感分布
        emotion_counts = {}
        for state in history:
            emotion_counts[state.emotion] = emotion_counts.get(state.emotion, 0) + 1
        
        total_states = len(history)
        emotion_distribution = {
            emotion: count / total_states
            for emotion, count in emotion_counts.items()
        }
        
        analysis['emotion_distribution'] = emotion_distribution
        
        # 情感强度分析
        intensities = [state.intensity for state in history]
        analysis['intensity_stats'] = {
            'mean': np.mean(intensities),
            'std': np.std(intensities),
            'min': np.min(intensities),
            'max': np.max(intensities)
        }
        
        # 情感维度分析
        valences = [state.valence for state in history]
        arousals = [state.arousal for state in history]
        dominances = [state.dominance for state in history]
        
        analysis['dimension_stats'] = {
            'valence': {'mean': np.mean(valences), 'std': np.std(valences)},
            'arousal': {'mean': np.mean(arousals), 'std': np.std(arousals)},
            'dominance': {'mean': np.mean(dominances), 'std': np.std(dominances)}
        }
        
        # 时间模式分析
        hourly_emotions = {}
        for state in history:
            hour = state.timestamp.hour
            if hour not in hourly_emotions:
                hourly_emotions[hour] = {}
            
            emotion = state.emotion
            hourly_emotions[hour][emotion] = hourly_emotions[hour].get(emotion, 0) + 1
        
        analysis['hourly_patterns'] = hourly_emotions
        
        # 情感变化频率
        transitions = []
        for i in range(1, len(history)):
            if history[i].emotion != history[i-1].emotion:
                transitions.append({
                    'from': history[i-1].emotion,
                    'to': history[i].emotion,
                    'time_diff': (history[i].timestamp - history[i-1].timestamp).total_seconds() / 60
                })
        
        analysis['transition_count'] = len(transitions)
        analysis['avg_transition_time'] = np.mean([t['time_diff'] for t in transitions]) if transitions else 0
        
        # 情感稳定性
        emotion_changes = sum(1 for i in range(1, len(history)) if history[i].emotion != history[i-1].emotion)
        analysis['stability'] = 1.0 - (emotion_changes / max(len(history) - 1, 1))
        
        return analysis
    
    async def _update_personality_profile(self, user_id: str, emotion_state: EmotionState):
        """更新个性化情感画像"""
        
        if user_id not in self.personality_profiles:
            # 创建新的个性画像
            self.personality_profiles[user_id] = PersonalityProfile(
                user_id=user_id,
                emotional_traits={
                    'extraversion': 0.5,
                    'agreeableness': 0.5,
                    'conscientiousness': 0.5,
                    'neuroticism': 0.5,
                    'openness': 0.5
                },
                baseline_emotions={
                    'happiness': 0.4,
                    'sadness': 0.2,
                    'anger': 0.1,
                    'fear': 0.1,
                    'surprise': 0.1,
                    'disgust': 0.05,
                    'neutral': 0.05
                },
                emotion_volatility=0.5,
                recovery_rate=0.5,
                dominant_emotions=[],
                trigger_patterns={},
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
        
        profile = self.personality_profiles[user_id]
        
        # 更新情感特质 (基于最近的情感状态)
        recent_history = self.emotion_history[user_id][-50:]  # 最近50个状态
        
        if len(recent_history) >= 10:
            # 计算外向性 (基于高唤醒度正面情感)
            positive_high_arousal = sum(
                1 for state in recent_history
                if state.valence > 0.5 and state.arousal > 0.5
            )
            profile.emotional_traits['extraversion'] = min(
                1.0, positive_high_arousal / len(recent_history) * 2
            )
            
            # 计算神经质倾向 (基于情感波动性)
            intensities = [state.intensity for state in recent_history]
            volatility = np.std(intensities)
            profile.emotion_volatility = min(1.0, volatility * 2)
            profile.emotional_traits['neuroticism'] = profile.emotion_volatility
            
            # 更新基线情感
            emotion_counts = {}
            for state in recent_history:
                emotion_counts[state.emotion] = emotion_counts.get(state.emotion, 0) + 1
            
            total = len(recent_history)
            for emotion in profile.baseline_emotions:
                profile.baseline_emotions[emotion] = emotion_counts.get(emotion, 0) / total
        
        # 更新触发模式
        if emotion_state.triggers:
            for trigger in emotion_state.triggers:
                if trigger not in profile.trigger_patterns:
                    profile.trigger_patterns[trigger] = []
                profile.trigger_patterns[trigger].append(emotion_state.emotion)
        
        profile.updated_at = datetime.now()
    
    async def _update_transition_model(self, user_id: str, current_state: EmotionState):
        """更新情感转换模型"""
        
        history = self.emotion_history[user_id]
        
        if len(history) < 20:  # 需要足够的历史数据
            return
        
        # 情感标签
        emotions = ['happiness', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
        n_emotions = len(emotions)
        emotion_to_idx = {emotion: idx for idx, emotion in enumerate(emotions)}
        
        # 初始化转换矩阵
        if user_id not in self.transition_matrices:
            self.transition_matrices[user_id] = np.eye(n_emotions) * 0.1 + 0.9 / n_emotions
        
        transition_matrix = self.transition_matrices[user_id].copy()
        
        # 统计转换
        transitions = {}
        for i in range(1, len(history)):
            prev_emotion = history[i-1].emotion
            curr_emotion = history[i].emotion
            
            prev_idx = emotion_to_idx.get(prev_emotion, emotion_to_idx['neutral'])
            curr_idx = emotion_to_idx.get(curr_emotion, emotion_to_idx['neutral'])
            
            if prev_idx not in transitions:
                transitions[prev_idx] = {}
            
            transitions[prev_idx][curr_idx] = transitions[prev_idx].get(curr_idx, 0) + 1
        
        # 更新转换矩阵
        for from_idx, to_transitions in transitions.items():
            total_transitions = sum(to_transitions.values())
            
            if total_transitions > 0:
                for to_idx, count in to_transitions.items():
                    # 使用指数移动平均更新
                    alpha = 0.1  # 学习率
                    new_prob = count / total_transitions
                    old_prob = transition_matrix[from_idx][to_idx]
                    transition_matrix[from_idx][to_idx] = old_prob * (1 - alpha) + new_prob * alpha
                
                # 重新归一化
                row_sum = np.sum(transition_matrix[from_idx])
                if row_sum > 0:
                    transition_matrix[from_idx] /= row_sum
        
        self.transition_matrices[user_id] = transition_matrix
    
    async def cluster_emotion_states(self, user_id: str, n_clusters: int = 5) -> Dict[str, Any]:
        """情感状态聚类分析"""
        
        if user_id not in self.emotion_history:
            return {}
        
        history = self.emotion_history[user_id]
        
        if len(history) < n_clusters * 3:  # 需要足够的数据点
            return {"error": "Insufficient data for clustering"}
        
        # 构造特征矩阵
        features = []
        for state in history:
            feature_vector = [
                state.intensity,
                state.valence,
                state.arousal,
                state.dominance,
                state.confidence
            ]
            features.append(feature_vector)
        
        features = np.array(features)
        
        # 标准化特征
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        # 分析聚类结果
        cluster_analysis = {}
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_states = [history[idx] for idx in cluster_indices]
            
            if cluster_states:
                # 聚类中心特征
                cluster_emotions = [state.emotion for state in cluster_states]
                dominant_emotion = max(set(cluster_emotions), key=cluster_emotions.count)
                
                cluster_analysis[f'cluster_{cluster_id}'] = {
                    'dominant_emotion': dominant_emotion,
                    'size': len(cluster_states),
                    'avg_intensity': np.mean([state.intensity for state in cluster_states]),
                    'avg_valence': np.mean([state.valence for state in cluster_states]),
                    'avg_arousal': np.mean([state.arousal for state in cluster_states]),
                    'avg_dominance': np.mean([state.dominance for state in cluster_states]),
                    'emotion_distribution': {
                        emotion: cluster_emotions.count(emotion) / len(cluster_emotions)
                        for emotion in set(cluster_emotions)
                    }
                }
        
        # 保存聚类模型
        self.emotion_clusters[user_id] = {
            'kmeans': kmeans,
            'scaler': scaler,
            'analysis': cluster_analysis,
            'created_at': datetime.now()
        }
        
        return cluster_analysis
    
    def get_personality_profile(self, user_id: str) -> Optional[PersonalityProfile]:
        """获取用户个性画像"""
        return self.personality_profiles.get(user_id)
    
    def export_emotion_data(self, user_id: str, format: str = 'json') -> str:
        """导出用户情感数据"""
        
        if user_id not in self.emotion_history:
            return ""
        
        data = {
            'user_id': user_id,
            'emotion_history': [asdict(state) for state in self.emotion_history[user_id]],
            'personality_profile': asdict(self.personality_profiles.get(user_id)) if user_id in self.personality_profiles else None,
            'export_timestamp': datetime.now().isoformat()
        }
        
        if format == 'json':
            return json.dumps(data, default=str, indent=2)
        
        # 可以添加其他格式支持
        return str(data)
```

### 共情响应生成器
```python
import random
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
import logging

@dataclass
class EmpathyResponse:
    response_text: str
    empathy_type: str  # 'cognitive', 'affective', 'compassionate'
    emotion_addressed: str
    comfort_level: float
    personalization_score: float
    suggested_actions: List[str]
    tone: str
    timestamp: datetime

class EmpathyStrategy:
    """共情策略基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    async def generate_response(
        self,
        user_emotion: EmotionState,
        context: Dict[str, Any],
        personality: Optional[PersonalityProfile] = None
    ) -> EmpathyResponse:
        raise NotImplementedError

class CognitiveEmpathyStrategy(EmpathyStrategy):
    """认知共情策略"""
    
    def __init__(self):
        super().__init__("cognitive_empathy", "理解和识别他人情感")
        
        self.recognition_templates = {
            'happiness': [
                "我能感受到你现在很开心，{context}真的值得庆祝！",
                "看得出来你心情很好，这种快乐的感觉真棒！",
                "你的喜悦感染了我，{context}一定让你很满足。"
            ],
            'sadness': [
                "我理解你现在的难过，{context}确实让人感到沉重。",
                "我能感受到你的痛苦，经历{context}一定不容易。",
                "我看得出你现在很伤心，这种感受是完全可以理解的。"
            ],
            'anger': [
                "我能理解你的愤怒，{context}确实令人沮丧。",
                "我看得出你很生气，这种感受是可以理解的。",
                "我知道{context}让你感到愤怒，这是很自然的反应。"
            ],
            'fear': [
                "我能感受到你的担忧，{context}确实让人感到不安。",
                "我理解你现在的恐惧，面对{context}会让人紧张。",
                "我看得出你很担心，这种不安感是可以理解的。"
            ],
            'surprise': [
                "我能感受到你的惊讶，{context}确实出人意料！",
                "看得出来你很惊讶，{context}真的很意外呢。",
                "我理解你的震惊，这种情况确实令人意外。"
            ]
        }
    
    async def generate_response(
        self,
        user_emotion: EmotionState,
        context: Dict[str, Any],
        personality: Optional[PersonalityProfile] = None
    ) -> EmpathyResponse:
        
        emotion = user_emotion.emotion
        templates = self.recognition_templates.get(emotion, self.recognition_templates['sadness'])
        
        # 选择合适的模板
        template = random.choice(templates)
        
        # 填充上下文
        context_text = context.get('situation', '这种情况')
        response_text = template.format(context=context_text)
        
        # 根据个性调整语调
        if personality:
            if personality.emotional_traits.get('extraversion', 0.5) > 0.7:
                response_text = self._adjust_for_extraversion(response_text)
            if personality.emotional_traits.get('neuroticism', 0.5) > 0.7:
                response_text = self._adjust_for_sensitivity(response_text)
        
        return EmpathyResponse(
            response_text=response_text,
            empathy_type="cognitive",
            emotion_addressed=emotion,
            comfort_level=0.7,
            personalization_score=0.6 if personality else 0.3,
            suggested_actions=["继续倾听", "询问更多细节"],
            tone="understanding",
            timestamp=datetime.now()
        )
    
    def _adjust_for_extraversion(self, text: str) -> str:
        """为外向性格调整语调"""
        # 添加更多互动性的表达
        if "我能感受到" in text:
            text = text.replace("我能感受到", "我完全能感受到")
        return text + " 想和我多分享一些吗？"
    
    def _adjust_for_sensitivity(self, text: str) -> str:
        """为敏感性格调整语调"""
        # 使用更温和的表达
        text = text.replace("确实", "可能")
        text = text.replace("一定", "或许")
        return text

class AffectiveEmpathyStrategy(EmpathyStrategy):
    """情感共情策略"""
    
    def __init__(self):
        super().__init__("affective_empathy", "感受和分享他人的情感")
        
        self.emotion_sharing_templates = {
            'happiness': [
                "你的快乐也感染了我！我也为{context}感到高兴。",
                "看到你这么开心，我的心情也变好了！",
                "你的喜悦让我也感到温暖，{context}真是太棒了！"
            ],
            'sadness': [
                "看到你难过，我的心也很沉重。我和你一起承受这份痛苦。",
                "你的伤心让我也感到心痛，我陪着你度过这个难关。",
                "我能感受到你内心的痛苦，让我陪伴你一起面对。"
            ],
            'anger': [
                "你的愤怒我也感受到了，这确实令人气愤！",
                "我也为这种不公感到愤怒，你有权利生气。",
                "看到你这样生气，我也感到很愤怒，这真的不应该发生。"
            ],
            'fear': [
                "你的恐惧我也感受到了，这确实很可怕。",
                "我也为你感到担心，这种不确定性确实令人不安。",
                "我和你一样感到担忧，我们一起面对这个挑战。"
            ]
        }
    
    async def generate_response(
        self,
        user_emotion: EmotionState,
        context: Dict[str, Any],
        personality: Optional[PersonalityProfile] = None
    ) -> EmpathyResponse:
        
        emotion = user_emotion.emotion
        templates = self.emotion_sharing_templates.get(emotion, self.emotion_sharing_templates['sadness'])
        
        template = random.choice(templates)
        context_text = context.get('situation', '这种情况')
        response_text = template.format(context=context_text)
        
        # 根据情感强度调整响应强度
        if user_emotion.intensity > 0.8:
            response_text = self._intensify_response(response_text)
        elif user_emotion.intensity < 0.4:
            response_text = self._soften_response(response_text)
        
        return EmpathyResponse(
            response_text=response_text,
            empathy_type="affective",
            emotion_addressed=emotion,
            comfort_level=0.8,
            personalization_score=0.7,
            suggested_actions=["提供陪伴", "共同体验情感"],
            tone="sharing",
            timestamp=datetime.now()
        )
    
    def _intensify_response(self, text: str) -> str:
        """增强响应强度"""
        text = text.replace("感受到了", "深深地感受到了")
        text = text.replace("我也", "我也强烈地")
        return text
    
    def _soften_response(self, text: str) -> str:
        """缓和响应强度"""
        text = text.replace("深深", "")
        text = text.replace("强烈", "")
        return text

class CompassionateEmpathyStrategy(EmpathyStrategy):
    """慈悲共情策略"""
    
    def __init__(self):
        super().__init__("compassionate_empathy", "理解情感并提供支持行动")
        
        self.support_templates = {
            'sadness': [
                "我理解你的痛苦。虽然现在很难熬，但请记住这种感受是暂时的。我们可以一起找到走出阴霾的方法。",
                "看到你这么难过，我很心疼。让我陪你一起度过这个艰难的时期，我相信你有力量克服这些困难。",
                "我能感受到你内心的痛苦。记住，寻求帮助是勇敢的表现，你不需要独自承受这一切。"
            ],
            'anger': [
                "我理解你的愤怒，这些感受完全可以理解。让我们一起找到建设性的方式来处理这种情绪。",
                "你有权利感到愤怒。现在最重要的是找到健康的方式来表达和处理这些情绪。",
                "我看得出你很生气，这是人之常情。我们可以一起探讨如何将这种能量转化为积极的行动。"
            ],
            'fear': [
                "我理解你的恐惧。恐惧往往源于不确定性，让我们一起面对这些挑战，找到应对的方法。",
                "感到害怕是完全正常的。我会在这里支持你，我们可以一步步地克服这些困难。",
                "你的担忧我完全理解。让我们一起制定一个计划，帮你获得更多的控制感和安全感。"
            ],
            'happiness': [
                "你的快乐真的很感染人！享受这美好的时刻，你值得拥有这种幸福。",
                "看到你这么开心，我也很高兴！记住这种感觉，它会在困难时给你力量。",
                "你的喜悦让我也感到温暖。珍惜这些美好的时刻，它们是生活的珍贵礼物。"
            ]
        }
        
        self.action_suggestions = {
            'sadness': [
                "我们可以一起制定一个小目标，帮你重新找到方向",
                "不如我们聊聊一些让你感到温暖的回忆？",
                "你想要我陪你静静坐一会儿，还是想要分散一下注意力？"
            ],
            'anger': [
                "我们可以一起练习一些深呼吸技巧来缓解愤怒",
                "写日记或运动可能会帮你更好地处理这些情绪",
                "你想要讨论一下具体的解决方案吗？"
            ],
            'fear': [
                "我们可以一起制定一个应对计划，让你感到更有准备",
                "分步骤地面对恐惧往往比一次性解决更有效",
                "你想要了解一些放松技巧来缓解焦虑吗？"
            ],
            'happiness': [
                "你想要分享更多关于这件开心事的细节吗？",
                "这种快乐的感觉值得庆祝！你想怎么纪念这个时刻？",
                "把这种正能量传递给其他人也是很棒的选择"
            ]
        }
    
    async def generate_response(
        self,
        user_emotion: EmotionState,
        context: Dict[str, Any],
        personality: Optional[PersonalityProfile] = None
    ) -> EmpathyResponse:
        
        emotion = user_emotion.emotion
        
        # 选择支持性回应模板
        templates = self.support_templates.get(emotion, self.support_templates['sadness'])
        response_text = random.choice(templates)
        
        # 选择行动建议
        actions = self.action_suggestions.get(emotion, ["我会在这里陪伴你"])
        suggested_actions = random.sample(actions, min(2, len(actions)))
        
        # 根据个性调整建议
        if personality:
            suggested_actions = self._personalize_actions(suggested_actions, personality)
        
        return EmpathyResponse(
            response_text=response_text,
            empathy_type="compassionate",
            emotion_addressed=emotion,
            comfort_level=0.9,
            personalization_score=0.8 if personality else 0.6,
            suggested_actions=suggested_actions,
            tone="supportive",
            timestamp=datetime.now()
        )
    
    def _personalize_actions(self, actions: List[str], personality: PersonalityProfile) -> List[str]:
        """根据个性化建议调整行动方案"""
        
        personalized_actions = []
        
        for action in actions:
            # 根据外向性调整
            if personality.emotional_traits.get('extraversion', 0.5) > 0.7:
                if "静静坐" in action:
                    action = action.replace("静静坐一会儿", "和朋友聊聊天")
                elif "写日记" in action:
                    action = action.replace("写日记", "和朋友分享你的感受")
            
            # 根据开放性调整
            if personality.emotional_traits.get('openness', 0.5) > 0.7:
                if "放松技巧" in action:
                    action = action.replace("放松技巧", "创新的放松方法，比如艺术疗法或音乐冥想")
            
            # 根据尽责性调整
            if personality.emotional_traits.get('conscientiousness', 0.5) > 0.7:
                if "应对计划" in action:
                    action = action.replace("应对计划", "详细的步骤规划和时间表")
            
            personalized_actions.append(action)
        
        return personalized_actions

class EmpathyResponseGenerator:
    """共情响应生成器"""
    
    def __init__(self):
        self.strategies = [
            CognitiveEmpathyStrategy(),
            AffectiveEmpathyStrategy(),
            CompassionateEmpathyStrategy()
        ]
        
        self.response_history: Dict[str, List[EmpathyResponse]] = {}
        self.logger = logging.getLogger(__name__)
    
    async def generate_empathy_response(
        self,
        user_id: str,
        user_emotion: EmotionState,
        context: Dict[str, Any],
        personality: Optional[PersonalityProfile] = None,
        preferred_strategy: Optional[str] = None
    ) -> EmpathyResponse:
        """生成共情响应"""
        
        try:
            # 选择策略
            if preferred_strategy:
                strategy = next((s for s in self.strategies if s.name == preferred_strategy), None)
                if not strategy:
                    strategy = self._select_best_strategy(user_emotion, context, personality)
            else:
                strategy = self._select_best_strategy(user_emotion, context, personality)
            
            # 生成响应
            response = await strategy.generate_response(user_emotion, context, personality)
            
            # 后处理和个性化
            response = await self._post_process_response(response, user_id, user_emotion, personality)
            
            # 记录响应历史
            if user_id not in self.response_history:
                self.response_history[user_id] = []
            
            self.response_history[user_id].append(response)
            
            # 保持历史记录限制
            if len(self.response_history[user_id]) > 100:
                self.response_history[user_id] = self.response_history[user_id][-100:]
            
            self.logger.info(f"Generated empathy response for user {user_id}: {strategy.name}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating empathy response: {e}")
            
            # 返回默认响应
            return EmpathyResponse(
                response_text="我理解你现在的感受，我在这里陪伴你。",
                empathy_type="default",
                emotion_addressed=user_emotion.emotion,
                comfort_level=0.5,
                personalization_score=0.3,
                suggested_actions=["继续交流"],
                tone="neutral",
                timestamp=datetime.now()
            )
    
    def _select_best_strategy(
        self,
        user_emotion: EmotionState,
        context: Dict[str, Any],
        personality: Optional[PersonalityProfile] = None
    ) -> EmpathyStrategy:
        """选择最佳共情策略"""
        
        # 根据情感类型和强度选择策略
        emotion = user_emotion.emotion
        intensity = user_emotion.intensity
        
        # 高强度负面情感优先使用慈悲共情
        if emotion in ['sadness', 'anger', 'fear'] and intensity > 0.7:
            return self.strategies[2]  # CompassionateEmpathyStrategy
        
        # 积极情感优先使用情感共情
        if emotion in ['happiness', 'surprise'] and user_emotion.valence > 0.5:
            return self.strategies[1]  # AffectiveEmpathyStrategy
        
        # 中等强度情感使用认知共情
        if intensity < 0.6:
            return self.strategies[0]  # CognitiveEmpathyStrategy
        
        # 根据个性选择
        if personality:
            # 高神经质倾向使用慈悲共情
            if personality.emotional_traits.get('neuroticism', 0.5) > 0.7:
                return self.strategies[2]  # CompassionateEmpathyStrategy
            
            # 高外向性使用情感共情
            if personality.emotional_traits.get('extraversion', 0.5) > 0.7:
                return self.strategies[1]  # AffectiveEmpathyStrategy
        
        # 默认使用认知共情
        return self.strategies[0]
    
    async def _post_process_response(
        self,
        response: EmpathyResponse,
        user_id: str,
        user_emotion: EmotionState,
        personality: Optional[PersonalityProfile] = None
    ) -> EmpathyResponse:
        """后处理和个性化响应"""
        
        # 避免重复响应
        if user_id in self.response_history:
            recent_responses = self.response_history[user_id][-5:]  # 最近5次响应
            similar_responses = [
                r for r in recent_responses
                if self._calculate_similarity(response.response_text, r.response_text) > 0.8
            ]
            
            if similar_responses:
                # 生成变体
                response.response_text = self._generate_response_variant(response.response_text)
        
        # 根据时间调整语调
        current_hour = datetime.now().hour
        if 0 <= current_hour < 6:
            response.response_text = self._adjust_for_night_time(response.response_text)
        elif 6 <= current_hour < 12:
            response.response_text = self._adjust_for_morning(response.response_text)
        
        # 根据文化背景调整（简化版本）
        if personality and 'culture' in personality.emotional_traits:
            culture = personality.emotional_traits['culture']
            if culture == 'collectivist':
                response.response_text = self._adjust_for_collectivist_culture(response.response_text)
        
        return response
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        
        # 简单的词汇重叠相似度计算
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _generate_response_variant(self, original_text: str) -> str:
        """生成响应变体"""
        
        # 同义词替换
        replacements = {
            '我理解': ['我明白', '我知道', '我能体会'],
            '感受到': ['体会到', '察觉到', '意识到'],
            '陪伴你': ['支持你', '在你身边', '和你一起'],
            '困难': ['挑战', '难题', '问题'],
            '美好': ['棒', 'wonderful', '精彩']
        }
        
        modified_text = original_text
        for original, variants in replacements.items():
            if original in modified_text:
                replacement = random.choice(variants)
                modified_text = modified_text.replace(original, replacement, 1)
                break  # 只替换一个，避免过度修改
        
        return modified_text
    
    def _adjust_for_night_time(self, text: str) -> str:
        """为夜晚时间调整语调"""
        return "这么晚了，" + text.lower()
    
    def _adjust_for_morning(self, text: str) -> str:
        """为早晨时间调整语调"""
        if "困难" in text:
            return text + " 新的一天，也许会带来新的希望。"
        return text
    
    def _adjust_for_collectivist_culture(self, text: str) -> str:
        """为集体主义文化调整表达"""
        text = text.replace("我", "我们")
        text = text.replace("你", "大家")
        return text
    
    async def get_response_statistics(self, user_id: str) -> Dict[str, Any]:
        """获取响应统计信息"""
        
        if user_id not in self.response_history:
            return {}
        
        responses = self.response_history[user_id]
        
        if not responses:
            return {}
        
        stats = {
            'total_responses': len(responses),
            'empathy_type_distribution': {},
            'emotion_addressed_distribution': {},
            'avg_comfort_level': 0.0,
            'avg_personalization_score': 0.0,
            'tone_distribution': {},
            'response_time_analysis': []
        }
        
        # 统计共情类型分布
        empathy_types = [r.empathy_type for r in responses]
        for emp_type in set(empathy_types):
            stats['empathy_type_distribution'][emp_type] = empathy_types.count(emp_type)
        
        # 统计情感处理分布
        emotions = [r.emotion_addressed for r in responses]
        for emotion in set(emotions):
            stats['emotion_addressed_distribution'][emotion] = emotions.count(emotion)
        
        # 统计平均分数
        stats['avg_comfort_level'] = sum(r.comfort_level for r in responses) / len(responses)
        stats['avg_personalization_score'] = sum(r.personalization_score for r in responses) / len(responses)
        
        # 统计语调分布
        tones = [r.tone for r in responses]
        for tone in set(tones):
            stats['tone_distribution'][tone] = tones.count(tone)
        
        return stats
```

## 🚦 风险评估与缓解

### 高风险项
1. **情感识别准确性不足**
   - 缓解: 多模态融合，持续学习优化，人工标注数据增强
   - 验证: A/B测试情感识别准确率，用户反馈验证

2. **文化和个体差异处理**
   - 缓解: 多元化训练数据，个性化适配算法，文化敏感性设计
   - 验证: 跨文化用户测试，个性化效果评估

3. **隐私和伦理问题**
   - 缓解: 端到端加密，用户控制数据，透明的隐私政策
   - 验证: 隐私合规审计，伦理委员会评估

### 中风险项
1. **情感操纵风险**
   - 缓解: 设置情感影响边界，避免过度干预，用户自主选择
   - 验证: 情感影响评估，用户反馈监控

2. **技术依赖性**
   - 缓解: 多模型备份，降级策略，本地化部署选项
   - 验证: 故障恢复测试，离线功能验证

## 📅 实施路线图

### Phase 1: 基础情感能力 (Week 1-3)
- 多模态情感识别引擎
- 情感状态建模系统
- 基础共情响应生成

### Phase 2: 高级情感功能 (Week 4-6)
- 情感记忆管理系统
- 情感智能决策引擎
- 个性化适配算法

### Phase 3: 社交和应用 (Week 7-8)
- 社交情感理解系统
- 用户界面和可视化
- 情感分析调试工具

### Phase 4: 优化和部署 (Week 9-10)
- 系统性能优化
- 隐私保护增强
- 生产环境部署

---

**文档状态**: ✅ 完成  
**下一步**: 开始Story 11.1的多模态情感识别引擎实施  
**依赖Epic**: 建议与Epic 7 (语音交互) 协同开发，充分利用多模态数据