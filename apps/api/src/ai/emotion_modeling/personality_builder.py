"""
个性化情感画像构建系统

基于Big Five人格理论和情感历史数据构建个性化情感画像
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import logging

from .models import EmotionState, PersonalityProfile, PersonalityTrait, EmotionType
from .space_mapper import EmotionSpaceMapper

logger = logging.getLogger(__name__)


class PersonalityProfileBuilder:
    """个性化画像构建器"""
    
    def __init__(self):
        self.space_mapper = EmotionSpaceMapper()
        
        # Big Five特质计算权重
        self.trait_weights = {
            PersonalityTrait.EXTRAVERSION: {
                'positive_high_arousal': 0.4,
                'social_emotions': 0.3,
                'intensity_variance': 0.3
            },
            PersonalityTrait.NEUROTICISM: {
                'volatility': 0.5,
                'negative_emotions': 0.3,
                'stress_response': 0.2
            },
            PersonalityTrait.AGREEABLENESS: {
                'positive_emotions': 0.4,
                'empathy_emotions': 0.3,
                'conflict_avoidance': 0.3
            },
            PersonalityTrait.CONSCIENTIOUSNESS: {
                'stability': 0.4,
                'consistency': 0.3,
                'goal_oriented': 0.3
            },
            PersonalityTrait.OPENNESS: {
                'emotion_diversity': 0.4,
                'surprise_response': 0.3,
                'adaptability': 0.3
            }
        }
        
        # 情感类别分组
        self.emotion_categories = {
            'positive': ['happiness', 'joy', 'gratitude', 'love', 'pride', 'hope'],
            'negative': ['sadness', 'anger', 'fear', 'disgust', 'anxiety', 'depression', 'shame', 'guilt'],
            'social': ['love', 'gratitude', 'trust', 'envy', 'contempt'],
            'empathetic': ['gratitude', 'trust', 'shame', 'guilt'],
            'activating': ['anger', 'fear', 'surprise', 'anticipation', 'joy'],
            'high_arousal': ['anger', 'fear', 'surprise', 'anxiety', 'joy']
        }
    
    async def build_personality_profile(
        self, 
        user_id: str, 
        emotion_history: List[EmotionState],
        min_samples: int = 50
    ) -> PersonalityProfile:
        """
        构建完整的个性化情感画像
        
        Args:
            user_id: 用户ID
            emotion_history: 情感历史数据
            min_samples: 最少样本数量
            
        Returns:
            个性化画像
        """
        if len(emotion_history) < min_samples:
            logger.warning(f"用户 {user_id} 情感历史样本不足 ({len(emotion_history)} < {min_samples})")
        
        # 按时间排序
        history_sorted = sorted(emotion_history, key=lambda x: x.timestamp)
        
        # 计算Big Five人格特质
        big_five_traits = await self.compute_big_five_traits(history_sorted)
        
        # 计算基线情感分布
        baseline_emotions = self.compute_baseline_emotions(history_sorted)
        
        # 计算情感波动性
        volatility = self.compute_emotion_volatility(history_sorted)
        
        # 计算恢复速度
        recovery_rate = self.compute_recovery_rate(history_sorted)
        
        # 识别主导情感
        dominant_emotions = self.identify_dominant_emotions(history_sorted)
        
        # 识别触发模式
        trigger_patterns = self.identify_trigger_patterns(history_sorted)
        
        # 计算置信度
        confidence_score = self.calculate_profile_confidence(history_sorted)
        
        # 构建画像
        profile = PersonalityProfile(
            user_id=user_id,
            emotional_traits=big_five_traits,
            baseline_emotions=baseline_emotions,
            emotion_volatility=volatility,
            recovery_rate=recovery_rate,
            dominant_emotions=dominant_emotions,
            trigger_patterns=trigger_patterns,
            sample_count=len(history_sorted),
            confidence_score=confidence_score,
            updated_at=datetime.now()
        )
        
        logger.info(f"已构建用户 {user_id} 的个性画像 (样本数: {len(history_sorted)})")
        return profile
    
    async def compute_big_five_traits(self, emotion_history: List[EmotionState]) -> Dict[str, float]:
        """
        计算Big Five人格特质
        
        Args:
            emotion_history: 情感历史数据
            
        Returns:
            Big Five特质分数 [0, 1]
        """
        traits = {}
        
        if not emotion_history:
            return {trait.value: 0.5 for trait in PersonalityTrait}
        
        # 提取基础统计信息
        emotions = [state.emotion for state in emotion_history]
        intensities = [state.intensity for state in emotion_history]
        valences = [self.space_mapper.map_state_to_space(state)[0] for state in emotion_history]
        arousals = [self.space_mapper.map_state_to_space(state)[1] for state in emotion_history]
        
        emotion_counter = Counter(emotions)
        total_count = len(emotion_history)
        
        # 1. 外向性 (Extraversion)
        extraversion_score = self._compute_extraversion(
            emotion_counter, total_count, valences, arousals, intensities
        )
        traits[PersonalityTrait.EXTRAVERSION.value] = extraversion_score
        
        # 2. 神经质 (Neuroticism)
        neuroticism_score = self._compute_neuroticism(
            emotion_counter, total_count, intensities, emotion_history
        )
        traits[PersonalityTrait.NEUROTICISM.value] = neuroticism_score
        
        # 3. 宜人性 (Agreeableness)
        agreeableness_score = self._compute_agreeableness(
            emotion_counter, total_count, valences
        )
        traits[PersonalityTrait.AGREEABLENESS.value] = agreeableness_score
        
        # 4. 尽责性 (Conscientiousness)
        conscientiousness_score = self._compute_conscientiousness(
            emotion_history, intensities
        )
        traits[PersonalityTrait.CONSCIENTIOUSNESS.value] = conscientiousness_score
        
        # 5. 开放性 (Openness)
        openness_score = self._compute_openness(
            emotion_counter, total_count, emotion_history
        )
        traits[PersonalityTrait.OPENNESS.value] = openness_score
        
        return traits
    
    def _compute_extraversion(
        self, 
        emotion_counter: Counter, 
        total_count: int, 
        valences: List[float], 
        arousals: List[float],
        intensities: List[float]
    ) -> float:
        """计算外向性特质"""
        score = 0.0
        
        # 积极高唤醒情感比例
        positive_high_arousal = 0
        for i, (valence, arousal) in enumerate(zip(valences, arousals)):
            if valence > 0.3 and arousal > 0.6:
                positive_high_arousal += 1
        
        if total_count > 0:
            score += (positive_high_arousal / total_count) * self.trait_weights[PersonalityTrait.EXTRAVERSION]['positive_high_arousal']
        
        # 社交情感比例
        social_emotions_count = sum(
            emotion_counter.get(emotion, 0) 
            for emotion in self.emotion_categories['social']
        )
        
        if total_count > 0:
            social_ratio = social_emotions_count / total_count
            score += social_ratio * self.trait_weights[PersonalityTrait.EXTRAVERSION]['social_emotions']
        
        # 情感强度方差（外向者情感更强烈）
        if intensities:
            intensity_mean = np.mean(intensities)
            score += min(1.0, intensity_mean) * self.trait_weights[PersonalityTrait.EXTRAVERSION]['intensity_variance']
        
        return min(1.0, score)
    
    def _compute_neuroticism(
        self, 
        emotion_counter: Counter, 
        total_count: int, 
        intensities: List[float],
        emotion_history: List[EmotionState]
    ) -> float:
        """计算神经质特质"""
        score = 0.0
        
        # 情感波动性
        if len(intensities) > 1:
            volatility = np.std(intensities)
            score += min(1.0, volatility * 2) * self.trait_weights[PersonalityTrait.NEUROTICISM]['volatility']
        
        # 负面情感比例
        negative_emotions_count = sum(
            emotion_counter.get(emotion, 0) 
            for emotion in self.emotion_categories['negative']
        )
        
        if total_count > 0:
            negative_ratio = negative_emotions_count / total_count
            score += negative_ratio * self.trait_weights[PersonalityTrait.NEUROTICISM]['negative_emotions']
        
        # 压力响应（快速情感变化）
        rapid_changes = self._count_rapid_emotion_changes(emotion_history)
        if total_count > 1:
            change_ratio = rapid_changes / (total_count - 1)
            score += min(1.0, change_ratio) * self.trait_weights[PersonalityTrait.NEUROTICISM]['stress_response']
        
        return min(1.0, score)
    
    def _compute_agreeableness(
        self, 
        emotion_counter: Counter, 
        total_count: int, 
        valences: List[float]
    ) -> float:
        """计算宜人性特质"""
        score = 0.0
        
        # 积极情感比例
        positive_emotions_count = sum(
            emotion_counter.get(emotion, 0) 
            for emotion in self.emotion_categories['positive']
        )
        
        if total_count > 0:
            positive_ratio = positive_emotions_count / total_count
            score += positive_ratio * self.trait_weights[PersonalityTrait.AGREEABLENESS]['positive_emotions']
        
        # 共情情感比例
        empathetic_emotions_count = sum(
            emotion_counter.get(emotion, 0) 
            for emotion in self.emotion_categories['empathetic']
        )
        
        if total_count > 0:
            empathy_ratio = empathetic_emotions_count / total_count
            score += empathy_ratio * self.trait_weights[PersonalityTrait.AGREEABLENESS]['empathy_emotions']
        
        # 整体效价水平
        if valences:
            mean_valence = np.mean(valences)
            # 将效价从[-1,1]映射到[0,1]
            valence_score = (mean_valence + 1) / 2
            score += valence_score * self.trait_weights[PersonalityTrait.AGREEABLENESS]['conflict_avoidance']
        
        return min(1.0, score)
    
    def _compute_conscientiousness(
        self, 
        emotion_history: List[EmotionState], 
        intensities: List[float]
    ) -> float:
        """计算尽责性特质"""
        score = 0.0
        
        if not emotion_history:
            return 0.5
        
        # 情感稳定性（少变化）
        emotion_changes = 0
        for i in range(1, len(emotion_history)):
            if emotion_history[i].emotion != emotion_history[i-1].emotion:
                emotion_changes += 1
        
        if len(emotion_history) > 1:
            stability = 1.0 - (emotion_changes / (len(emotion_history) - 1))
            score += stability * self.trait_weights[PersonalityTrait.CONSCIENTIOUSNESS]['stability']
        
        # 强度一致性（低方差）
        if len(intensities) > 1:
            intensity_consistency = 1.0 - min(1.0, np.std(intensities))
            score += intensity_consistency * self.trait_weights[PersonalityTrait.CONSCIENTIOUSNESS]['consistency']
        
        # 目标导向性（基于上下文分析）
        goal_oriented_score = self._analyze_goal_orientation(emotion_history)
        score += goal_oriented_score * self.trait_weights[PersonalityTrait.CONSCIENTIOUSNESS]['goal_oriented']
        
        return min(1.0, score)
    
    def _compute_openness(
        self, 
        emotion_counter: Counter, 
        total_count: int, 
        emotion_history: List[EmotionState]
    ) -> float:
        """计算开放性特质"""
        score = 0.0
        
        # 情感多样性
        unique_emotions = len(emotion_counter)
        max_possible_emotions = len(EmotionType)
        emotion_diversity = unique_emotions / max_possible_emotions
        score += emotion_diversity * self.trait_weights[PersonalityTrait.OPENNESS]['emotion_diversity']
        
        # 对惊讶的响应
        surprise_count = emotion_counter.get('surprise', 0)
        if total_count > 0:
            surprise_ratio = surprise_count / total_count
            score += surprise_ratio * self.trait_weights[PersonalityTrait.OPENNESS]['surprise_response']
        
        # 适应性（情感转换多样性）
        adaptability_score = self._measure_adaptability(emotion_history)
        score += adaptability_score * self.trait_weights[PersonalityTrait.OPENNESS]['adaptability']
        
        return min(1.0, score)
    
    def _count_rapid_emotion_changes(
        self, 
        emotion_history: List[EmotionState],
        time_threshold: timedelta = timedelta(hours=1)
    ) -> int:
        """计算快速情感变化次数"""
        rapid_changes = 0
        
        for i in range(1, len(emotion_history)):
            time_diff = emotion_history[i].timestamp - emotion_history[i-1].timestamp
            if (time_diff <= time_threshold and 
                emotion_history[i].emotion != emotion_history[i-1].emotion):
                rapid_changes += 1
        
        return rapid_changes
    
    def _analyze_goal_orientation(self, emotion_history: List[EmotionState]) -> float:
        """分析目标导向性（基于上下文）"""
        goal_score = 0.5  # 默认中等水平
        
        # 分析触发因素中的目标导向词汇
        goal_keywords = ['achievement', 'goal', 'success', 'completion', 'progress']
        total_triggers = 0
        goal_triggers = 0
        
        for state in emotion_history:
            for trigger in state.triggers:
                total_triggers += 1
                if any(keyword in trigger.lower() for keyword in goal_keywords):
                    goal_triggers += 1
        
        if total_triggers > 0:
            goal_score = goal_triggers / total_triggers
        
        return goal_score
    
    def _measure_adaptability(self, emotion_history: List[EmotionState]) -> float:
        """测量情感适应性"""
        if len(emotion_history) < 3:
            return 0.5
        
        # 计算情感转换的多样性
        transitions = set()
        for i in range(1, len(emotion_history)):
            from_emotion = emotion_history[i-1].emotion
            to_emotion = emotion_history[i].emotion
            if from_emotion != to_emotion:
                transitions.add((from_emotion, to_emotion))
        
        # 理论最大转换数
        unique_emotions = len(set(state.emotion for state in emotion_history))
        max_transitions = unique_emotions * (unique_emotions - 1)
        
        if max_transitions > 0:
            adaptability = len(transitions) / max_transitions
        else:
            adaptability = 0.5
        
        return min(1.0, adaptability)
    
    def compute_baseline_emotions(self, emotion_history: List[EmotionState]) -> Dict[str, float]:
        """计算个人基线情感分布"""
        if not emotion_history:
            return {}
        
        # 计算每种情感的归一化频率
        emotion_counter = Counter(state.emotion for state in emotion_history)
        total_count = len(emotion_history)
        
        baseline = {
            emotion: count / total_count 
            for emotion, count in emotion_counter.items()
        }
        
        return baseline
    
    def compute_emotion_volatility(self, emotion_history: List[EmotionState]) -> float:
        """计算情感波动性"""
        if len(emotion_history) < 2:
            return 0.5
        
        # 计算强度方差
        intensities = [state.intensity for state in emotion_history]
        intensity_var = np.var(intensities)
        
        # 计算VAD空间中的位置方差
        vad_points = [self.space_mapper.map_state_to_space(state) for state in emotion_history]
        vad_array = np.array(vad_points)
        
        spatial_var = np.mean(np.var(vad_array, axis=0))
        
        # 组合两种波动性度量
        volatility = 0.6 * intensity_var + 0.4 * spatial_var
        
        return min(1.0, volatility)
    
    def compute_recovery_rate(self, emotion_history: List[EmotionState]) -> float:
        """计算情感恢复速度"""
        if len(emotion_history) < 3:
            return 0.5
        
        recovery_times = []
        
        # 寻找负面情感到中性/积极情感的转换
        for i in range(1, len(emotion_history)):
            current_state = emotion_history[i]
            prev_state = emotion_history[i-1]
            
            # 检查是否从负面情感恢复
            if (prev_state.emotion in self.emotion_categories['negative'] and
                current_state.emotion not in self.emotion_categories['negative']):
                
                # 计算恢复时间
                recovery_time = current_state.timestamp - prev_state.timestamp
                recovery_times.append(recovery_time.total_seconds() / 3600)  # 小时为单位
        
        if not recovery_times:
            return 0.5
        
        # 计算平均恢复时间，转换为恢复速度分数
        avg_recovery_time = np.mean(recovery_times)
        
        # 1小时内恢复得满分，24小时以上得0分
        recovery_rate = max(0.0, 1.0 - (avg_recovery_time / 24.0))
        
        return recovery_rate
    
    def identify_dominant_emotions(
        self, 
        emotion_history: List[EmotionState],
        top_k: int = 3
    ) -> List[str]:
        """识别主导情感"""
        if not emotion_history:
            return []
        
        # 考虑强度加权的情感频率
        emotion_weights = defaultdict(float)
        
        for state in emotion_history:
            # 使用强度和置信度加权
            weight = state.intensity * state.confidence
            emotion_weights[state.emotion] += weight
        
        # 排序并返回前k个
        sorted_emotions = sorted(
            emotion_weights.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [emotion for emotion, _ in sorted_emotions[:top_k]]
    
    def identify_trigger_patterns(self, emotion_history: List[EmotionState]) -> Dict[str, List[str]]:
        """识别情感触发模式"""
        trigger_patterns = defaultdict(list)
        
        for state in emotion_history:
            for trigger in state.triggers:
                if trigger not in trigger_patterns[state.emotion]:
                    trigger_patterns[state.emotion].append(trigger)
        
        # 转换为普通字典
        return dict(trigger_patterns)
    
    def calculate_profile_confidence(self, emotion_history: List[EmotionState]) -> float:
        """计算画像置信度"""
        if not emotion_history:
            return 0.0
        
        # 基于样本数量的置信度
        sample_confidence = min(1.0, len(emotion_history) / 100.0)  # 100个样本达到满分
        
        # 基于时间跨度的置信度
        if len(emotion_history) > 1:
            time_span = emotion_history[-1].timestamp - emotion_history[0].timestamp
            days_span = time_span.days
            time_confidence = min(1.0, days_span / 30.0)  # 30天跨度达到满分
        else:
            time_confidence = 0.1
        
        # 基于数据质量的置信度
        avg_confidence = np.mean([state.confidence for state in emotion_history])
        
        # 综合置信度
        overall_confidence = 0.4 * sample_confidence + 0.3 * time_confidence + 0.3 * avg_confidence
        
        return overall_confidence