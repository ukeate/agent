"""
Group Emotion Analyzer
群体情感分析引擎

This module provides comprehensive group emotion analysis capabilities including:
- Collective emotion trend identification
- Emotion contagion detection
- Group polarization analysis
- Emotional leadership identification
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
from collections import defaultdict, Counter
import statistics

from .group_emotion_models import (
    EmotionState,
    GroupEmotionalState,
    EmotionContagionEvent,
    ContagionPattern,
    EmotionalLeader,
    GroupEmotionAnalysisConfig,
    EmotionContagionType,
    GroupCohesionLevel,
    generate_group_id,
    generate_event_id
)


class GroupEmotionAnalyzer:
    """群体情感分析器"""
    
    def __init__(self, config: Optional[GroupEmotionAnalysisConfig] = None):
        self.config = config or GroupEmotionAnalysisConfig()
        
        # 情感维度权重映射
        self.emotion_weights = {
            'intensity_factor': self.config.intensity_weight,
            'frequency_factor': self.config.frequency_weight,
            'influence_factor': self.config.influence_weight
        }
        
        # 情感极性映射 (用于计算共识和极化)
        self.emotion_polarity = {
            'joy': 1.0, 'happiness': 1.0, 'excitement': 1.0, 'satisfaction': 0.8,
            'anger': -1.0, 'frustration': -0.8, 'irritation': -0.6,
            'sadness': -0.9, 'disappointment': -0.7, 'melancholy': -0.6,
            'fear': -0.8, 'anxiety': -0.7, 'worry': -0.5,
            'surprise': 0.0, 'neutral': 0.0, 'calm': 0.2,
            'disgust': -0.9, 'contempt': -0.8,
            'love': 1.0, 'affection': 0.9, 'empathy': 0.8,
            'shame': -0.6, 'guilt': -0.7, 'embarrassment': -0.5
        }
        
    async def analyze_group_emotion(
        self,
        participants_emotions: Dict[str, EmotionState],
        interaction_history: Optional[List[Dict[str, Any]]] = None,
        group_id: Optional[str] = None
    ) -> GroupEmotionalState:
        """分析群体情感状态"""
        
        if len(participants_emotions) < self.config.min_participants:
            raise ValueError(f"需要至少 {self.config.min_participants} 个参与者")
            
        if len(participants_emotions) > self.config.max_participants:
            raise ValueError(f"参与者数量不能超过 {self.config.max_participants}")
        
        # 计算参与者影响权重
        influence_weights = await self._calculate_influence_weights(
            list(participants_emotions.keys()),
            interaction_history or []
        )
        
        # 计算加权情感分布
        emotion_distribution = self._calculate_weighted_emotion_distribution(
            participants_emotions, influence_weights
        )
        
        # 确定主导情感
        dominant_emotion = max(emotion_distribution.items(), key=lambda x: x[1])[0]
        
        # 计算群体动态指标
        consensus_level = self._calculate_consensus_level(participants_emotions)
        polarization_index = self._calculate_polarization_index(participants_emotions)
        emotional_volatility = self._calculate_emotional_volatility(participants_emotions)
        group_cohesion = self._determine_group_cohesion(consensus_level, polarization_index)
        
        # 识别情感领导者
        emotional_leaders = await self._identify_emotional_leaders(
            participants_emotions, interaction_history or [], influence_weights
        )
        
        # 构建影响网络
        influence_network = await self._build_influence_network(
            participants_emotions, interaction_history or []
        )
        
        # 分析传染模式
        contagion_patterns = await self._analyze_contagion_patterns(
            participants_emotions, interaction_history or []
        )
        
        # 计算传染速度
        contagion_velocity = self._calculate_contagion_velocity(contagion_patterns)
        
        # 趋势预测
        trend_prediction = self._predict_group_trend(
            participants_emotions, interaction_history or []
        )
        
        # 计算稳定性分数
        stability_score = self._calculate_stability_score(
            consensus_level, polarization_index, emotional_volatility
        )
        
        return GroupEmotionalState(
            group_id=group_id or generate_group_id(),
            timestamp=datetime.now(),
            participants=list(participants_emotions.keys()),
            dominant_emotion=dominant_emotion,
            emotion_distribution=emotion_distribution,
            consensus_level=consensus_level,
            polarization_index=polarization_index,
            emotional_volatility=emotional_volatility,
            group_cohesion=group_cohesion,
            emotional_leaders=emotional_leaders,
            influence_network=influence_network,
            contagion_patterns=contagion_patterns,
            contagion_velocity=contagion_velocity,
            trend_prediction=trend_prediction,
            stability_score=stability_score,
            analysis_confidence=self._calculate_analysis_confidence(
                participants_emotions, interaction_history or []
            ),
            data_completeness=self._calculate_data_completeness(participants_emotions)
        )
    
    def _calculate_weighted_emotion_distribution(
        self,
        participants_emotions: Dict[str, EmotionState],
        influence_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """计算加权情感分布"""
        
        weighted_emotions = defaultdict(float)
        
        for participant_id, emotion_state in participants_emotions.items():
            weight = influence_weights.get(participant_id, 1.0)
            emotion = emotion_state.emotion
            intensity = emotion_state.intensity
            
            weighted_emotions[emotion] += intensity * weight
        
        # 归一化
        total_weight = sum(weighted_emotions.values())
        if total_weight > 0:
            return {
                emotion: weight / total_weight
                for emotion, weight in weighted_emotions.items()
            }
        else:
            return {'neutral': 1.0}
    
    async def _calculate_influence_weights(
        self,
        participant_ids: List[str],
        interaction_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """计算参与者影响权重"""
        
        weights = {pid: 1.0 for pid in participant_ids}  # 默认权重
        
        if not interaction_history:
            return weights
        
        # 基于交互频率计算权重
        interaction_counts = Counter()
        response_counts = defaultdict(int)
        influence_scores = defaultdict(float)
        
        for interaction in interaction_history:
            sender = interaction.get('sender_id')
            if sender in participant_ids:
                interaction_counts[sender] += 1
                
                # 如果有回应，增加影响力分数
                if interaction.get('has_responses', False):
                    influence_scores[sender] += 1
                
                # 情感强度影响权重
                emotion_intensity = interaction.get('emotion_intensity', 0.5)
                influence_scores[sender] += emotion_intensity * 0.5
        
        # 归一化权重
        max_interactions = max(interaction_counts.values()) if interaction_counts else 1
        max_influence = max(influence_scores.values()) if influence_scores else 1
        
        for participant_id in participant_ids:
            freq_weight = interaction_counts.get(participant_id, 0) / max_interactions
            influence_weight = influence_scores.get(participant_id, 0) / max_influence
            
            # 组合权重
            weights[participant_id] = (
                0.4 * freq_weight + 
                0.6 * influence_weight + 
                0.3  # 基础权重
            )
        
        # 确保权重在合理范围内
        min_weight, max_weight = 0.1, 2.0
        for participant_id in weights:
            weights[participant_id] = max(min_weight, min(max_weight, weights[participant_id]))
        
        return weights
    
    def _calculate_consensus_level(
        self,
        participants_emotions: Dict[str, EmotionState]
    ) -> float:
        """计算群体情感共识水平"""
        
        if len(participants_emotions) <= 1:
            return 1.0
        
        emotions_list = list(participants_emotions.values())
        similarities = []
        
        # 计算两两之间的情感相似度
        for i in range(len(emotions_list)):
            for j in range(i + 1, len(emotions_list)):
                emotion1, emotion2 = emotions_list[i], emotions_list[j]
                
                # 基于VAD维度计算相似度
                valence_sim = 1 - abs(emotion1.valence - emotion2.valence) / 2
                arousal_sim = 1 - abs(emotion1.arousal - emotion2.arousal) / 1
                dominance_sim = 1 - abs(emotion1.dominance - emotion2.dominance) / 1
                
                # 情感类型一致性
                emotion_sim = 1.0 if emotion1.emotion == emotion2.emotion else 0.5
                
                # 综合相似度
                overall_similarity = (
                    0.3 * valence_sim +
                    0.25 * arousal_sim +
                    0.25 * dominance_sim +
                    0.2 * emotion_sim
                )
                
                similarities.append(overall_similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_polarization_index(
        self,
        participants_emotions: Dict[str, EmotionState]
    ) -> float:
        """计算群体极化指数"""
        
        if len(participants_emotions) <= 1:
            return 0.0
        
        # 获取情感极性值
        polarities = []
        for emotion_state in participants_emotions.values():
            polarity = self.emotion_polarity.get(emotion_state.emotion, 0.0)
            # 考虑强度影响
            weighted_polarity = polarity * emotion_state.intensity
            polarities.append(weighted_polarity)
        
        if not polarities:
            return 0.0
        
        # 计算极化指数：方差越大，极化越严重
        mean_polarity = np.mean(polarities)
        variance = np.var(polarities)
        
        # 归一化极化指数 [0, 1]
        max_possible_variance = 1.0  # 理论最大方差
        polarization_index = min(variance / max_possible_variance, 1.0)
        
        return polarization_index
    
    def _calculate_emotional_volatility(
        self,
        participants_emotions: Dict[str, EmotionState]
    ) -> float:
        """计算情感波动性"""
        
        if len(participants_emotions) <= 1:
            return 0.0
        
        # 基于情感强度的标准差计算波动性
        intensities = [es.intensity for es in participants_emotions.values()]
        arousals = [es.arousal for es in participants_emotions.values()]
        
        intensity_volatility = np.std(intensities) if len(intensities) > 1 else 0.0
        arousal_volatility = np.std(arousals) if len(arousals) > 1 else 0.0
        
        # 综合波动性
        volatility = (intensity_volatility + arousal_volatility) / 2
        
        return min(volatility, 1.0)
    
    def _determine_group_cohesion(
        self,
        consensus_level: float,
        polarization_index: float
    ) -> GroupCohesionLevel:
        """确定群体凝聚力水平"""
        
        # 基于共识水平和极化指数判断凝聚力
        cohesion_score = consensus_level * 0.7 + (1 - polarization_index) * 0.3
        
        if cohesion_score >= 0.8:
            return GroupCohesionLevel.HIGH
        elif cohesion_score >= 0.6:
            return GroupCohesionLevel.MEDIUM
        elif cohesion_score >= 0.4:
            return GroupCohesionLevel.LOW
        else:
            return GroupCohesionLevel.FRAGMENTED
    
    async def _identify_emotional_leaders(
        self,
        participants_emotions: Dict[str, EmotionState],
        interaction_history: List[Dict[str, Any]],
        influence_weights: Dict[str, float]
    ) -> List[EmotionalLeader]:
        """识别情感领导者"""
        
        leaders = []
        threshold = self.config.leader_influence_threshold
        
        for participant_id, emotion_state in participants_emotions.items():
            influence_score = influence_weights.get(participant_id, 0.0)
            
            if influence_score >= threshold:
                # 分析领导类型
                leadership_type = self._analyze_leadership_type(
                    emotion_state, interaction_history, participant_id
                )
                
                # 找到被影响的参与者
                influenced_participants = self._find_influenced_participants(
                    participant_id, participants_emotions, interaction_history
                )
                
                # 计算一致性分数
                consistency_score = self._calculate_leadership_consistency(
                    participant_id, interaction_history
                )
                
                leader = EmotionalLeader(
                    participant_id=participant_id,
                    influence_score=influence_score,
                    leadership_type=leadership_type,
                    influenced_participants=influenced_participants,
                    dominant_emotions=[emotion_state.emotion],
                    consistency_score=consistency_score
                )
                
                leaders.append(leader)
        
        # 按影响力排序
        leaders.sort(key=lambda x: x.influence_score, reverse=True)
        
        return leaders[:5]  # 最多返回5个领导者
    
    def _analyze_leadership_type(
        self,
        emotion_state: EmotionState,
        interaction_history: List[Dict[str, Any]],
        participant_id: str
    ) -> str:
        """分析领导类型"""
        
        polarity = self.emotion_polarity.get(emotion_state.emotion, 0.0)
        
        # 统计该参与者的历史情感倾向
        positive_count = 0
        negative_count = 0
        
        for interaction in interaction_history:
            if interaction.get('sender_id') == participant_id:
                emotion = interaction.get('detected_emotion')
                if emotion:
                    emotion_polarity = self.emotion_polarity.get(emotion, 0.0)
                    if emotion_polarity > 0.3:
                        positive_count += 1
                    elif emotion_polarity < -0.3:
                        negative_count += 1
        
        total_interactions = positive_count + negative_count
        if total_interactions == 0:
            return "neutral"
        
        positive_ratio = positive_count / total_interactions
        
        if positive_ratio > 0.7:
            return "positive"
        elif positive_ratio < 0.3:
            return "negative"
        else:
            return "neutral"
    
    def _find_influenced_participants(
        self,
        leader_id: str,
        participants_emotions: Dict[str, EmotionState],
        interaction_history: List[Dict[str, Any]]
    ) -> List[str]:
        """找到被影响的参与者"""
        
        influenced = []
        leader_emotion = participants_emotions[leader_id].emotion
        
        # 简化版本：寻找情感相似的其他参与者
        for participant_id, emotion_state in participants_emotions.items():
            if participant_id != leader_id:
                if emotion_state.emotion == leader_emotion:
                    influenced.append(participant_id)
        
        return influenced
    
    def _calculate_leadership_consistency(
        self,
        participant_id: str,
        interaction_history: List[Dict[str, Any]]
    ) -> float:
        """计算领导一致性分数"""
        
        if not interaction_history:
            return 0.5
        
        # 统计该参与者的情感稳定性
        emotions = []
        for interaction in interaction_history:
            if interaction.get('sender_id') == participant_id:
                emotion = interaction.get('detected_emotion')
                if emotion:
                    emotions.append(emotion)
        
        if len(emotions) <= 1:
            return 0.5
        
        # 计算情感一致性
        emotion_counts = Counter(emotions)
        most_common_emotion = emotion_counts.most_common(1)[0]
        consistency = most_common_emotion[1] / len(emotions)
        
        return consistency
    
    async def _build_influence_network(
        self,
        participants_emotions: Dict[str, EmotionState],
        interaction_history: List[Dict[str, Any]]
    ) -> Dict[str, List[str]]:
        """构建影响网络"""
        
        network = defaultdict(list)
        
        # 基于交互历史构建影响关系
        for interaction in interaction_history:
            sender = interaction.get('sender_id')
            # 简化：假设回复者受到发送者影响
            if interaction.get('has_responses'):
                responders = interaction.get('responders', [])
                for responder in responders:
                    if responder in participants_emotions:
                        network[sender].append(responder)
        
        # 去重
        for sender in network:
            network[sender] = list(set(network[sender]))
        
        return dict(network)
    
    async def _analyze_contagion_patterns(
        self,
        participants_emotions: Dict[str, EmotionState],
        interaction_history: List[Dict[str, Any]]
    ) -> List[ContagionPattern]:
        """分析情感传染模式"""
        
        patterns = []
        
        if not interaction_history:
            return patterns
        
        # 按时间排序
        sorted_history = sorted(
            interaction_history,
            key=lambda x: x.get('timestamp', datetime.now())
        )
        
        # 检测情感传染事件
        emotion_sequences = defaultdict(list)
        
        for interaction in sorted_history:
            sender = interaction.get('sender_id')
            emotion = interaction.get('detected_emotion')
            timestamp = interaction.get('timestamp', datetime.now())
            
            if sender and emotion:
                emotion_sequences[emotion].append({
                    'participant': sender,
                    'timestamp': timestamp,
                    'intensity': interaction.get('emotion_intensity', 0.5)
                })
        
        # 分析每种情感的传播模式
        for emotion, sequence in emotion_sequences.items():
            if len(sequence) >= 2:
                pattern = self._analyze_single_emotion_contagion(emotion, sequence)
                if pattern:
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_single_emotion_contagion(
        self,
        emotion: str,
        sequence: List[Dict[str, Any]]
    ) -> Optional[ContagionPattern]:
        """分析单一情感的传染模式"""
        
        if len(sequence) < 2:
            return None
        
        # 按时间排序
        sequence = sorted(sequence, key=lambda x: x['timestamp'])
        
        source_participant = sequence[0]['participant']
        target_participants = [item['participant'] for item in sequence[1:]]
        
        # 计算传染强度
        source_intensity = sequence[0]['intensity']
        target_intensities = [item['intensity'] for item in sequence[1:]]
        avg_target_intensity = np.mean(target_intensities)
        
        contagion_strength = min(avg_target_intensity / max(source_intensity, 0.1), 1.0)
        
        # 计算传播速度
        time_span = (sequence[-1]['timestamp'] - sequence[0]['timestamp']).total_seconds()
        propagation_speed = len(target_participants) / max(time_span / 60, 1)  # participants/minute
        
        # 确定传染类型
        contagion_type = self._determine_contagion_type(
            source_intensity, target_intensities, time_span
        )
        
        return ContagionPattern(
            source_participant=source_participant,
            target_participants=target_participants,
            emotion=emotion,
            contagion_type=contagion_type,
            strength=contagion_strength,
            propagation_speed=propagation_speed,
            timestamp=sequence[0]['timestamp'],
            duration_seconds=int(time_span)
        )
    
    def _determine_contagion_type(
        self,
        source_intensity: float,
        target_intensities: List[float],
        time_span: float
    ) -> EmotionContagionType:
        """确定传染类型"""
        
        if not target_intensities:
            return EmotionContagionType.DAMPENING
        
        avg_target_intensity = np.mean(target_intensities)
        intensity_ratio = avg_target_intensity / max(source_intensity, 0.1)
        speed = len(target_intensities) / max(time_span / 60, 1)
        
        if intensity_ratio > 1.2:
            return EmotionContagionType.AMPLIFICATION
        elif speed > 2.0 and intensity_ratio > 0.8:
            return EmotionContagionType.VIRAL
        elif intensity_ratio > 0.6:
            return EmotionContagionType.CASCADE
        else:
            return EmotionContagionType.DAMPENING
    
    def _calculate_contagion_velocity(
        self,
        contagion_patterns: List[ContagionPattern]
    ) -> float:
        """计算传染速度"""
        
        if not contagion_patterns:
            return 0.0
        
        velocities = [pattern.propagation_speed for pattern in contagion_patterns]
        return np.mean(velocities)
    
    def _predict_group_trend(
        self,
        participants_emotions: Dict[str, EmotionState],
        interaction_history: List[Dict[str, Any]]
    ) -> str:
        """预测群体趋势"""
        
        if not interaction_history:
            return "stable"
        
        # 分析最近的情感变化趋势
        recent_history = [
            interaction for interaction in interaction_history
            if (datetime.now() - interaction.get('timestamp', datetime.now())).seconds <= 300  # 5分钟内
        ]
        
        if len(recent_history) < 3:
            return "stable"
        
        # 计算情感强度趋势
        intensities = [
            interaction.get('emotion_intensity', 0.5)
            for interaction in recent_history[-5:]  # 最近5条
        ]
        
        if len(intensities) >= 3:
            # 简单线性趋势分析
            x = np.arange(len(intensities))
            slope = np.polyfit(x, intensities, 1)[0]
            
            if slope > 0.05:
                return "escalating"
            elif slope < -0.05:
                return "declining"
            else:
                return "stable"
        
        return "stable"
    
    def _calculate_stability_score(
        self,
        consensus_level: float,
        polarization_index: float,
        emotional_volatility: float
    ) -> float:
        """计算稳定性分数"""
        
        # 高共识、低极化、低波动 = 高稳定性
        stability_score = (
            consensus_level * 0.4 +
            (1 - polarization_index) * 0.35 +
            (1 - emotional_volatility) * 0.25
        )
        
        return min(max(stability_score, 0.0), 1.0)
    
    def _calculate_analysis_confidence(
        self,
        participants_emotions: Dict[str, EmotionState],
        interaction_history: List[Dict[str, Any]]
    ) -> float:
        """计算分析置信度"""
        
        # 基于数据质量计算置信度
        participant_count = len(participants_emotions)
        history_length = len(interaction_history)
        
        # 参与者数量因子
        participant_factor = min(participant_count / 5, 1.0)
        
        # 历史数据因子
        history_factor = min(history_length / 10, 1.0)
        
        # 情感识别置信度
        emotion_confidences = [es.confidence for es in participants_emotions.values()]
        avg_emotion_confidence = np.mean(emotion_confidences) if emotion_confidences else 0.5
        
        # 综合置信度
        overall_confidence = (
            participant_factor * 0.3 +
            history_factor * 0.3 +
            avg_emotion_confidence * 0.4
        )
        
        return min(max(overall_confidence, 0.1), 1.0)
    
    def _calculate_data_completeness(
        self,
        participants_emotions: Dict[str, EmotionState]
    ) -> float:
        """计算数据完整性"""
        
        if not participants_emotions:
            return 0.0
        
        # 检查必要字段的完整性
        complete_count = 0
        total_count = len(participants_emotions)
        
        for emotion_state in participants_emotions.values():
            # 检查关键字段是否完整
            if (emotion_state.emotion and 
                emotion_state.intensity is not None and
                emotion_state.valence is not None and
                emotion_state.arousal is not None):
                complete_count += 1
        
        return complete_count / total_count if total_count > 0 else 0.0