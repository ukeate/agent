"""多模态情感融合引擎"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
import numpy as np
from enum import Enum

from ..models.emotion_models import (
    EmotionResult, MultiModalEmotion, EmotionDimension,
    Modality, EmotionCategory, EMOTION_DIMENSIONS
)
from ..analyzers import (
    TextEmotionAnalyzer,
    AudioEmotionAnalyzer,
    VisualEmotionAnalyzer
)

logger = logging.getLogger(__name__)


class FusionStrategy(str, Enum):
    """融合策略"""
    WEIGHTED_AVERAGE = "weighted_average"      # 加权平均
    CONFIDENCE_BASED = "confidence_based"       # 基于置信度
    DYNAMIC_ADAPTIVE = "dynamic_adaptive"       # 动态自适应
    HIERARCHICAL = "hierarchical"               # 层次融合
    VOTING = "voting"                           # 投票机制


class MultiModalEmotionFusion:
    """多模态情感融合引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化融合引擎
        
        Args:
            config: 配置参数
        """
        default_config = {
            "fusion_strategy": FusionStrategy.DYNAMIC_ADAPTIVE,
            "default_weights": {
                Modality.TEXT.value: 0.4,
                Modality.AUDIO.value: 0.35,
                Modality.VISUAL.value: 0.25
            },
            "confidence_threshold": 0.5,
            "conflict_resolution": "confidence",  # confidence, voting, or average
            "enable_temporal_fusion": True,
            "temporal_window": 3  # 时间窗口(秒)
        }
        
        if config:
            default_config.update(config)
            
        self.config = default_config
        self.fusion_strategy = self.config["fusion_strategy"]
        
        # 初始化各模态分析器
        self.analyzers = {
            Modality.TEXT: TextEmotionAnalyzer(),
            Modality.AUDIO: AudioEmotionAnalyzer(),
            Modality.VISUAL: VisualEmotionAnalyzer()
        }
        
        # 历史记录(用于时间融合)
        self.history: List[MultiModalEmotion] = []
        self.max_history_size = 100
        
        # 动态权重学习
        self.dynamic_weights = self.config["default_weights"].copy()
        self.weight_history: List[Dict[str, float]] = []
        
        self.is_initialized = False
        
    async def initialize(self):
        """初始化所有分析器"""
        if not self.is_initialized:
            logger.info("正在初始化多模态融合引擎...")
            
            # 并行初始化所有分析器
            init_tasks = [
                analyzer.initialize() 
                for analyzer in self.analyzers.values()
            ]
            await asyncio.gather(*init_tasks)
            
            self.is_initialized = True
            logger.info("多模态融合引擎初始化完成")
            
    async def analyze(
        self,
        text: Optional[str] = None,
        audio: Optional[Any] = None,
        image: Optional[Any] = None,
        user_profile: Optional[Any] = None
    ) -> MultiModalEmotion:
        """
        多模态情感分析
        
        Args:
            text: 文本输入
            audio: 音频输入
            image: 图像输入
            user_profile: 用户画像(用于个性化)
            
        Returns:
            MultiModalEmotion: 融合后的情感结果
        """
        if not self.is_initialized:
            await self.initialize()
            
        # 收集可用的模态
        inputs = {}
        if text is not None:
            inputs[Modality.TEXT] = text
        if audio is not None:
            inputs[Modality.AUDIO] = audio
        if image is not None:
            inputs[Modality.VISUAL] = image
            
        if not inputs:
            raise ValueError("至少需要一种模态的输入")
            
        # 并行分析各模态
        modality_results = await self._analyze_modalities(inputs)
        
        # 计算模态权重
        weights = await self._calculate_weights(modality_results, user_profile)
        
        # 融合结果
        if self.fusion_strategy == FusionStrategy.WEIGHTED_AVERAGE:
            fusion_result = await self._weighted_average_fusion(modality_results, weights)
        elif self.fusion_strategy == FusionStrategy.CONFIDENCE_BASED:
            fusion_result = await self._confidence_based_fusion(modality_results)
        elif self.fusion_strategy == FusionStrategy.DYNAMIC_ADAPTIVE:
            fusion_result = await self._dynamic_adaptive_fusion(modality_results, weights)
        elif self.fusion_strategy == FusionStrategy.HIERARCHICAL:
            fusion_result = await self._hierarchical_fusion(modality_results, weights)
        elif self.fusion_strategy == FusionStrategy.VOTING:
            fusion_result = await self._voting_fusion(modality_results)
        else:
            fusion_result = await self._weighted_average_fusion(modality_results, weights)
            
        # 时间融合(如果启用)
        if self.config["enable_temporal_fusion"] and len(self.history) > 0:
            fusion_result = await self._temporal_fusion(fusion_result)
            
        # 更新历史
        self._update_history(fusion_result)
        
        return fusion_result
        
    async def _analyze_modalities(self, inputs: Dict[Modality, Any]) -> Dict[str, EmotionResult]:
        """并行分析各模态"""
        tasks = []
        modality_keys = []
        
        for modality, data in inputs.items():
            analyzer = self.analyzers[modality]
            tasks.append(analyzer.analyze(data))
            modality_keys.append(modality.value)
            
        # 并行执行
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        modality_results = {}
        for i, (key, result) in enumerate(zip(modality_keys, results)):
            if isinstance(result, Exception):
                logger.error(f"{key}模态分析失败: {result}")
            else:
                modality_results[key] = result
                
        return modality_results
        
    async def _calculate_weights(
        self,
        modality_results: Dict[str, EmotionResult],
        user_profile: Optional[Any] = None
    ) -> Dict[str, float]:
        """计算模态权重"""
        weights = {}
        
        # 基础权重
        base_weights = self.dynamic_weights.copy()
        
        # 根据置信度调整权重
        total_confidence = sum(r.confidence for r in modality_results.values())
        
        for modality, result in modality_results.items():
            # 置信度权重
            confidence_weight = result.confidence / total_confidence if total_confidence > 0 else 0.33
            
            # 可靠性评估
            reliability = await self._assess_reliability(modality, result)
            
            # 个性化权重(如果有用户画像)
            personal_weight = 1.0
            if user_profile and hasattr(user_profile, 'personalized_weights'):
                personal_weight = user_profile.personalized_weights.get(modality, 1.0)
                
            # 综合权重
            if modality in base_weights:
                weights[modality] = (
                    base_weights[modality] * 0.4 +
                    confidence_weight * 0.3 +
                    reliability * 0.2 +
                    personal_weight * 0.1
                )
            else:
                weights[modality] = confidence_weight * reliability * personal_weight
                
        # 归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
            
        return weights
        
    async def _assess_reliability(self, modality: str, result: EmotionResult) -> float:
        """评估模态可靠性"""
        reliability = 1.0
        
        # 基于置信度的可靠性
        if result.confidence < self.config["confidence_threshold"]:
            reliability *= 0.7
            
        # 基于详细信息的可靠性评估
        details = result.details
        
        if modality == Modality.TEXT.value:
            # 文本长度影响可靠性
            if details.get("word_count", 0) < 3:
                reliability *= 0.8
                
        elif modality == Modality.AUDIO.value:
            # 音频时长影响可靠性
            if details.get("duration", 0) < 1.0:
                reliability *= 0.8
                
        elif modality == Modality.VISUAL.value:
            # 面部数量影响可靠性
            if details.get("num_faces", 0) == 0:
                reliability *= 0.5
                
        return min(max(reliability, 0.1), 1.0)
        
    async def _weighted_average_fusion(
        self,
        modality_results: Dict[str, EmotionResult],
        weights: Dict[str, float]
    ) -> MultiModalEmotion:
        """加权平均融合"""
        # 收集所有情感及其加权分数
        emotion_scores: Dict[str, float] = {}
        
        for modality, result in modality_results.items():
            weight = weights.get(modality, 0.33)
            
            # 主要情感
            emotion = result.emotion
            score = result.confidence * weight
            
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0
            emotion_scores[emotion] += score
            
            # 次要情感
            if result.sub_emotions:
                for sub_emotion, sub_confidence in result.sub_emotions[:3]:
                    sub_score = sub_confidence * weight * 0.5  # 次要情感权重降低
                    if sub_emotion not in emotion_scores:
                        emotion_scores[sub_emotion] = 0
                    emotion_scores[sub_emotion] += sub_score
                    
        # 排序并选择主要情感
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_emotions:
            # 返回中性结果
            return self._create_neutral_fusion_result(modality_results, weights)
            
        primary_emotion = sorted_emotions[0][0]
        overall_confidence = min(sorted_emotions[0][1], 1.0)
        
        # 次要情感
        secondary_emotions = [(e, s) for e, s in sorted_emotions[1:6] if s > 0.1]
        
        # 计算综合维度
        valence, arousal, dominance = await self._calculate_dimensions(modality_results, weights)
        
        # 计算强度
        intensity = await self._calculate_intensity(modality_results, weights)
        
        return MultiModalEmotion(
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            overall_confidence=overall_confidence,
            intensity_level=intensity,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            modality_weights=weights,
            modality_results=modality_results,
            timestamp=datetime.now(),
            fusion_strategy=self.fusion_strategy.value
        )
        
    async def _confidence_based_fusion(
        self,
        modality_results: Dict[str, EmotionResult]
    ) -> MultiModalEmotion:
        """基于置信度的融合"""
        # 选择置信度最高的模态作为主导
        if not modality_results:
            return self._create_neutral_fusion_result({}, {})
            
        best_modality = max(modality_results.items(), key=lambda x: x[1].confidence)
        primary_result = best_modality[1]
        
        # 使用最高置信度的结果作为主要情感
        primary_emotion = primary_result.emotion
        overall_confidence = primary_result.confidence
        
        # 收集其他模态的支持
        secondary_emotions = []
        for modality, result in modality_results.items():
            if modality != best_modality[0]:
                if result.emotion != primary_emotion:
                    secondary_emotions.append((result.emotion, result.confidence))
                else:
                    # 相同情感，增强置信度
                    overall_confidence = min(overall_confidence + result.confidence * 0.2, 1.0)
                    
        # 排序次要情感
        secondary_emotions = sorted(secondary_emotions, key=lambda x: x[1], reverse=True)[:5]
        
        # 使用主导模态的维度，但考虑其他模态的影响
        valence = primary_result.dimension.valence if primary_result.dimension else 0
        arousal = primary_result.dimension.arousal if primary_result.dimension else 0.3
        dominance = primary_result.dimension.dominance if primary_result.dimension else 0.5
        
        # 动态权重(基于置信度)
        total_confidence = sum(r.confidence for r in modality_results.values())
        weights = {
            m: r.confidence / total_confidence 
            for m, r in modality_results.items()
        }
        
        return MultiModalEmotion(
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            overall_confidence=overall_confidence,
            intensity_level=primary_result.intensity,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            modality_weights=weights,
            modality_results=modality_results,
            timestamp=datetime.now(),
            fusion_strategy=self.fusion_strategy.value
        )
        
    async def _dynamic_adaptive_fusion(
        self,
        modality_results: Dict[str, EmotionResult],
        weights: Dict[str, float]
    ) -> MultiModalEmotion:
        """动态自适应融合"""
        # 检测模态间的一致性
        consistency = await self._check_consistency(modality_results)
        
        if consistency > 0.7:
            # 高一致性，使用加权平均
            result = await self._weighted_average_fusion(modality_results, weights)
        elif consistency < 0.3:
            # 低一致性，使用冲突解决
            result = await self._resolve_conflicts(modality_results, weights)
        else:
            # 中等一致性，使用混合策略
            weighted_result = await self._weighted_average_fusion(modality_results, weights)
            confidence_result = await self._confidence_based_fusion(modality_results)
            
            # 混合两种结果
            result = await self._blend_results(weighted_result, confidence_result, consistency)
            
        # 更新动态权重
        await self._update_dynamic_weights(modality_results, result)
        
        return result
        
    async def _hierarchical_fusion(
        self,
        modality_results: Dict[str, EmotionResult],
        weights: Dict[str, float]
    ) -> MultiModalEmotion:
        """层次融合"""
        # 第一层：模态内融合(如果有多个相同模态的输入)
        # 这里简化处理，直接使用单个模态结果
        
        # 第二层：跨模态融合
        # 优先级：视觉 > 音频 > 文本(可配置)
        hierarchy = [Modality.VISUAL.value, Modality.AUDIO.value, Modality.TEXT.value]
        
        primary_emotion = None
        overall_confidence = 0
        
        for modality in hierarchy:
            if modality in modality_results:
                result = modality_results[modality]
                if result.confidence > self.config["confidence_threshold"]:
                    if primary_emotion is None:
                        primary_emotion = result.emotion
                        overall_confidence = result.confidence
                    elif result.emotion == primary_emotion:
                        # 增强置信度
                        overall_confidence = min(overall_confidence + result.confidence * 0.3, 1.0)
                        
        if primary_emotion is None:
            # 没有高置信度结果，使用加权平均
            return await self._weighted_average_fusion(modality_results, weights)
            
        # 收集次要情感
        secondary_emotions = []
        for modality, result in modality_results.items():
            if result.emotion != primary_emotion:
                secondary_emotions.append((result.emotion, result.confidence))
                
        secondary_emotions = sorted(secondary_emotions, key=lambda x: x[1], reverse=True)[:5]
        
        # 计算维度
        valence, arousal, dominance = await self._calculate_dimensions(modality_results, weights)
        intensity = await self._calculate_intensity(modality_results, weights)
        
        return MultiModalEmotion(
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            overall_confidence=overall_confidence,
            intensity_level=intensity,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            modality_weights=weights,
            modality_results=modality_results,
            timestamp=datetime.now(),
            fusion_strategy=self.fusion_strategy.value
        )
        
    async def _voting_fusion(
        self,
        modality_results: Dict[str, EmotionResult]
    ) -> MultiModalEmotion:
        """投票机制融合"""
        # 统计投票
        votes: Dict[str, List[float]] = {}
        
        for modality, result in modality_results.items():
            emotion = result.emotion
            if emotion not in votes:
                votes[emotion] = []
            votes[emotion].append(result.confidence)
            
            # 次要情感也参与投票(权重降低)
            if result.sub_emotions:
                for sub_emotion, sub_confidence in result.sub_emotions[:2]:
                    if sub_emotion not in votes:
                        votes[sub_emotion] = []
                    votes[sub_emotion].append(sub_confidence * 0.5)
                    
        # 计算每个情感的得分
        emotion_scores = {}
        for emotion, confidences in votes.items():
            # 得分 = 投票数 * 平均置信度
            score = len(confidences) * np.mean(confidences)
            emotion_scores[emotion] = score
            
        # 排序选择
        sorted_emotions = sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_emotions:
            return self._create_neutral_fusion_result(modality_results, {})
            
        primary_emotion = sorted_emotions[0][0]
        
        # 计算综合置信度
        total_votes = sum(len(v) for v in votes.values())
        primary_votes = len(votes[primary_emotion])
        overall_confidence = primary_votes / total_votes if total_votes > 0 else 0.5
        
        # 次要情感
        secondary_emotions = [(e, s/sorted_emotions[0][1]) for e, s in sorted_emotions[1:6]]
        
        # 计算维度(使用所有模态的平均)
        equal_weights = {m: 1.0/len(modality_results) for m in modality_results.keys()}
        valence, arousal, dominance = await self._calculate_dimensions(modality_results, equal_weights)
        intensity = await self._calculate_intensity(modality_results, equal_weights)
        
        return MultiModalEmotion(
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            overall_confidence=overall_confidence,
            intensity_level=intensity,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            modality_weights=equal_weights,
            modality_results=modality_results,
            timestamp=datetime.now(),
            fusion_strategy=self.fusion_strategy.value
        )
        
    async def _check_consistency(self, modality_results: Dict[str, EmotionResult]) -> float:
        """检查模态间一致性"""
        if len(modality_results) < 2:
            return 1.0
            
        emotions = [r.emotion for r in modality_results.values()]
        
        # 计算相同情感的比例
        unique_emotions = set(emotions)
        consistency = 1.0 - (len(unique_emotions) - 1) / len(emotions)
        
        return max(0.0, min(1.0, consistency))
        
    async def _resolve_conflicts(
        self,
        modality_results: Dict[str, EmotionResult],
        weights: Dict[str, float]
    ) -> MultiModalEmotion:
        """解决模态冲突"""
        conflict_resolution = self.config["conflict_resolution"]
        
        if conflict_resolution == "confidence":
            # 选择置信度最高的
            return await self._confidence_based_fusion(modality_results)
        elif conflict_resolution == "voting":
            # 使用投票机制
            return await self._voting_fusion(modality_results)
        else:
            # 使用加权平均
            return await self._weighted_average_fusion(modality_results, weights)
            
    async def _blend_results(
        self,
        result1: MultiModalEmotion,
        result2: MultiModalEmotion,
        blend_ratio: float
    ) -> MultiModalEmotion:
        """混合两个融合结果"""
        # 主要情感选择置信度更高的
        if result1.overall_confidence > result2.overall_confidence:
            primary_emotion = result1.primary_emotion
            overall_confidence = (
                result1.overall_confidence * blend_ratio +
                result2.overall_confidence * (1 - blend_ratio)
            )
        else:
            primary_emotion = result2.primary_emotion
            overall_confidence = (
                result2.overall_confidence * blend_ratio +
                result1.overall_confidence * (1 - blend_ratio)
            )
            
        # 混合次要情感
        all_secondary = list(result1.secondary_emotions) + list(result2.secondary_emotions)
        secondary_dict = {}
        for emotion, confidence in all_secondary:
            if emotion not in secondary_dict:
                secondary_dict[emotion] = 0
            secondary_dict[emotion] += confidence * 0.5
            
        secondary_emotions = sorted(secondary_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # 混合维度
        valence = result1.valence * blend_ratio + result2.valence * (1 - blend_ratio)
        arousal = result1.arousal * blend_ratio + result2.arousal * (1 - blend_ratio)
        dominance = result1.dominance * blend_ratio + result2.dominance * (1 - blend_ratio)
        intensity = result1.intensity_level * blend_ratio + result2.intensity_level * (1 - blend_ratio)
        
        # 混合权重
        blended_weights = {}
        for modality in set(list(result1.modality_weights.keys()) + list(result2.modality_weights.keys())):
            w1 = result1.modality_weights.get(modality, 0)
            w2 = result2.modality_weights.get(modality, 0)
            blended_weights[modality] = w1 * blend_ratio + w2 * (1 - blend_ratio)
            
        return MultiModalEmotion(
            primary_emotion=primary_emotion,
            secondary_emotions=secondary_emotions,
            overall_confidence=overall_confidence,
            intensity_level=intensity,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            modality_weights=blended_weights,
            modality_results=result1.modality_results,  # 使用原始结果
            timestamp=datetime.now(),
            fusion_strategy="blended"
        )
        
    async def _temporal_fusion(self, current: MultiModalEmotion) -> MultiModalEmotion:
        """时间融合"""
        # 获取时间窗口内的历史
        window = self.config["temporal_window"]
        recent_history = []
        
        now = datetime.now()
        for hist in reversed(self.history):
            time_diff = (now - hist.timestamp).total_seconds()
            if time_diff <= window:
                recent_history.append(hist)
            else:
                break
                
        if not recent_history:
            return current
            
        # 计算时间权重(越近权重越高)
        time_weights = []
        for hist in recent_history:
            time_diff = (now - hist.timestamp).total_seconds()
            weight = 1.0 - (time_diff / window) * 0.5  # 最旧的权重为0.5
            time_weights.append(weight)
            
        # 归一化
        total_weight = sum(time_weights) + 1.0  # 当前结果权重为1
        time_weights = [w / total_weight for w in time_weights]
        current_weight = 1.0 / total_weight
        
        # 融合情感
        emotion_scores = {current.primary_emotion: current_weight * current.overall_confidence}
        
        for hist, weight in zip(recent_history, time_weights):
            emotion = hist.primary_emotion
            if emotion not in emotion_scores:
                emotion_scores[emotion] = 0
            emotion_scores[emotion] += weight * hist.overall_confidence
            
        # 选择主要情感
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0]
        
        # 平滑维度值
        smoothed_valence = current.valence * current_weight
        smoothed_arousal = current.arousal * current_weight
        smoothed_dominance = current.dominance * current_weight
        
        for hist, weight in zip(recent_history, time_weights):
            smoothed_valence += hist.valence * weight
            smoothed_arousal += hist.arousal * weight
            smoothed_dominance += hist.dominance * weight
            
        # 更新结果
        current.primary_emotion = primary_emotion
        current.valence = smoothed_valence
        current.arousal = smoothed_arousal
        current.dominance = smoothed_dominance
        
        return current
        
    async def _calculate_dimensions(
        self,
        modality_results: Dict[str, EmotionResult],
        weights: Dict[str, float]
    ) -> Tuple[float, float, float]:
        """计算综合维度值"""
        valence = 0
        arousal = 0
        dominance = 0
        
        for modality, result in modality_results.items():
            weight = weights.get(modality, 0.33)
            
            if result.dimension:
                valence += result.dimension.valence * weight
                arousal += result.dimension.arousal * weight
                dominance += result.dimension.dominance * weight
            else:
                # 使用默认维度
                emotion_enum = EmotionCategory(result.emotion.lower()) if result.emotion.lower() in [e.value for e in EmotionCategory] else EmotionCategory.NEUTRAL
                default_dim = EMOTION_DIMENSIONS.get(emotion_enum, EMOTION_DIMENSIONS[EmotionCategory.NEUTRAL])
                valence += default_dim.valence * weight
                arousal += default_dim.arousal * weight
                dominance += default_dim.dominance * weight
                
        return valence, arousal, dominance
        
    async def _calculate_intensity(
        self,
        modality_results: Dict[str, EmotionResult],
        weights: Dict[str, float]
    ) -> float:
        """计算综合强度"""
        intensity = 0
        
        for modality, result in modality_results.items():
            weight = weights.get(modality, 0.33)
            intensity += result.intensity * weight
            
        return min(max(intensity, 0.0), 1.0)
        
    async def _update_dynamic_weights(
        self,
        modality_results: Dict[str, EmotionResult],
        fusion_result: MultiModalEmotion
    ):
        """更新动态权重"""
        # 基于融合结果的置信度调整权重
        if fusion_result.overall_confidence > 0.7:
            # 高置信度，增强当前权重配置
            for modality in modality_results.keys():
                if modality in self.dynamic_weights:
                    current = self.dynamic_weights[modality]
                    used = fusion_result.modality_weights.get(modality, current)
                    # 缓慢调整
                    self.dynamic_weights[modality] = current * 0.9 + used * 0.1
                    
        # 记录权重历史
        self.weight_history.append(self.dynamic_weights.copy())
        if len(self.weight_history) > 100:
            self.weight_history.pop(0)
            
    def _update_history(self, result: MultiModalEmotion):
        """更新历史记录"""
        self.history.append(result)
        
        # 限制历史大小
        if len(self.history) > self.max_history_size:
            self.history.pop(0)
            
    def _create_neutral_fusion_result(
        self,
        modality_results: Dict[str, EmotionResult],
        weights: Dict[str, float]
    ) -> MultiModalEmotion:
        """创建中性融合结果"""
        return MultiModalEmotion(
            primary_emotion=EmotionCategory.NEUTRAL.value,
            secondary_emotions=[],
            overall_confidence=0.5,
            intensity_level=0.3,
            valence=0.0,
            arousal=0.3,
            dominance=0.5,
            modality_weights=weights,
            modality_results=modality_results,
            timestamp=datetime.now(),
            fusion_strategy=self.fusion_strategy.value
        )
        
    def get_fusion_statistics(self) -> Dict[str, Any]:
        """获取融合统计信息"""
        if not self.history:
            return {}
            
        # 情感分布
        emotion_distribution = {}
        for result in self.history:
            emotion = result.primary_emotion
            if emotion not in emotion_distribution:
                emotion_distribution[emotion] = 0
            emotion_distribution[emotion] += 1
            
        # 平均维度值
        avg_valence = np.mean([r.valence for r in self.history])
        avg_arousal = np.mean([r.arousal for r in self.history])
        avg_dominance = np.mean([r.dominance for r in self.history])
        
        # 平均置信度和强度
        avg_confidence = np.mean([r.overall_confidence for r in self.history])
        avg_intensity = np.mean([r.intensity_level for r in self.history])
        
        return {
            "history_size": len(self.history),
            "emotion_distribution": emotion_distribution,
            "average_dimensions": {
                "valence": float(avg_valence),
                "arousal": float(avg_arousal),
                "dominance": float(avg_dominance)
            },
            "average_confidence": float(avg_confidence),
            "average_intensity": float(avg_intensity),
            "current_weights": self.dynamic_weights,
            "fusion_strategy": self.fusion_strategy.value
        }
        
    async def cleanup(self):
        """清理资源"""
        logger.info("清理多模态融合引擎...")
        
        # 清理各分析器
        cleanup_tasks = [
            analyzer.cleanup()
            for analyzer in self.analyzers.values()
        ]
        await asyncio.gather(*cleanup_tasks)
        
        # 清理历史
        self.history.clear()
        self.weight_history.clear()
        
        self.is_initialized = False