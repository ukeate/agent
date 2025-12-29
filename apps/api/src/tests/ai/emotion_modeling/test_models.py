"""
测试情感状态建模系统的数据模型
"""

from src.core.utils.timezone_utils import utc_now
import pytest
from datetime import timedelta
from src.ai.emotion_modeling.models import (
    EmotionState, PersonalityProfile, EmotionTransition, EmotionPrediction,
    EmotionStatistics, EmotionType, PersonalityTrait
)

class TestEmotionState:
    """测试EmotionState数据类"""
    
    def test_create_emotion_state(self):
        """测试创建情感状态"""
        state = EmotionState(
            user_id="test_user",
            emotion="happiness",
            intensity=0.8,
            valence=0.7,
            arousal=0.6,
            dominance=0.5
        )
        
        assert state.user_id == "test_user"
        assert state.emotion == "happiness"
        assert state.intensity == 0.8
        assert state.valence == 0.7
        assert state.arousal == 0.6
        assert state.dominance == 0.5
        assert state.confidence == 1.0  # 默认值
        assert state.source == "manual"  # 默认值
    
    def test_emotion_state_to_dict(self):
        """测试情感状态转字典"""
        state = EmotionState(
            user_id="test_user",
            emotion="sadness",
            intensity=0.6,
            valence=-0.5,
            arousal=0.3,
            dominance=0.2,
            triggers=["work_stress", "family_issue"],
            context={"location": "office", "time_of_day": "evening"}
        )
        
        data = state.to_dict()
        assert data['user_id'] == "test_user"
        assert data['emotion'] == "sadness"
        assert data['triggers'] == ["work_stress", "family_issue"]
        assert data['context'] == {"location": "office", "time_of_day": "evening"}
        assert 'timestamp' in data
        assert 'id' in data
    
    def test_emotion_state_from_dict(self):
        """测试从字典创建情感状态"""
        data = {
            'user_id': "test_user",
            'emotion': "anger",
            'intensity': 0.9,
            'valence': -0.8,
            'arousal': 0.9,
            'dominance': 0.8,
            'confidence': 0.85,
            'timestamp': utc_now().isoformat(),
            'triggers': ["injustice"],
            'context': {"event": "meeting"},
            'source': 'voice'
        }
        
        state = EmotionState.from_dict(data)
        assert state.user_id == "test_user"
        assert state.emotion == "anger"
        assert state.intensity == 0.9
        assert state.source == "voice"
        assert state.triggers == ["injustice"]
    
    def test_get_vad_coordinates(self):
        """测试获取VAD坐标"""
        state = EmotionState(
            valence=0.5, arousal=0.7, dominance=0.3
        )
        vad = state.get_vad_coordinates()
        assert vad == (0.5, 0.7, 0.3)
    
    def test_is_positive(self):
        """测试是否为积极情感"""
        positive_state = EmotionState(valence=0.5)
        negative_state = EmotionState(valence=-0.3)
        neutral_state = EmotionState(valence=0.0)
        
        assert positive_state.is_positive() == True
        assert negative_state.is_positive() == False
        assert neutral_state.is_positive() == False
    
    def test_is_high_arousal(self):
        """测试是否为高唤醒情感"""
        high_arousal = EmotionState(arousal=0.8)
        low_arousal = EmotionState(arousal=0.4)
        
        assert high_arousal.is_high_arousal() == True
        assert low_arousal.is_high_arousal() == False

class TestPersonalityProfile:
    """测试PersonalityProfile数据类"""
    
    def test_create_personality_profile(self):
        """测试创建个性画像"""
        profile = PersonalityProfile(
            user_id="test_user",
            emotional_traits={"extraversion": 0.7, "neuroticism": 0.3},
            baseline_emotions={"happiness": 0.6, "sadness": 0.2},
            emotion_volatility=0.4,
            recovery_rate=0.8
        )
        
        assert profile.user_id == "test_user"
        assert profile.emotional_traits["extraversion"] == 0.7
        assert profile.baseline_emotions["happiness"] == 0.6
        assert profile.emotion_volatility == 0.4
        assert profile.sample_count == 0  # 默认值
    
    def test_personality_profile_to_dict(self):
        """测试个性画像转字典"""
        profile = PersonalityProfile(
            user_id="test_user",
            emotional_traits={"openness": 0.8},
            baseline_emotions={"joy": 0.7}
        )
        
        data = profile.to_dict()
        assert data['user_id'] == "test_user"
        assert data['emotional_traits'] == {"openness": 0.8}
        assert data['baseline_emotions'] == {"joy": 0.7}
        assert 'created_at' in data
        assert 'updated_at' in data
    
    def test_get_set_trait(self):
        """测试获取和设置人格特质"""
        profile = PersonalityProfile(user_id="test_user")
        
        # 设置特质
        profile.set_trait(PersonalityTrait.EXTRAVERSION, 0.8)
        assert profile.get_trait(PersonalityTrait.EXTRAVERSION) == 0.8
        
        # 测试边界值
        profile.set_trait(PersonalityTrait.NEUROTICISM, 1.5)  # 超出范围
        assert profile.get_trait(PersonalityTrait.NEUROTICISM) == 1.0
        
        profile.set_trait(PersonalityTrait.AGREEABLENESS, -0.5)  # 低于范围
        assert profile.get_trait(PersonalityTrait.AGREEABLENESS) == 0.0
    
    def test_is_high_volatility(self):
        """测试高波动性判断"""
        high_vol = PersonalityProfile(emotion_volatility=0.8)
        low_vol = PersonalityProfile(emotion_volatility=0.5)
        
        assert high_vol.is_high_volatility() == True
        assert low_vol.is_high_volatility() == False
    
    def test_is_fast_recovery(self):
        """测试快速恢复判断"""
        fast = PersonalityProfile(recovery_rate=0.8)
        slow = PersonalityProfile(recovery_rate=0.5)
        
        assert fast.is_fast_recovery() == True
        assert slow.is_fast_recovery() == False

class TestEmotionTransition:
    """测试EmotionTransition数据类"""
    
    def test_create_emotion_transition(self):
        """测试创建情感转换"""
        transition = EmotionTransition(
            user_id="test_user",
            from_emotion="sadness",
            to_emotion="happiness",
            transition_probability=0.3,
            occurrence_count=5
        )
        
        assert transition.user_id == "test_user"
        assert transition.from_emotion == "sadness"
        assert transition.to_emotion == "happiness"
        assert transition.transition_probability == 0.3
        assert transition.occurrence_count == 5
    
    def test_transition_to_dict(self):
        """测试转换记录转字典"""
        transition = EmotionTransition(
            user_id="test_user",
            from_emotion="anger",
            to_emotion="neutral",
            transition_probability=0.4,
            occurrence_count=3,
            avg_duration=timedelta(minutes=15)
        )
        
        data = transition.to_dict()
        assert data['from_emotion'] == "anger"
        assert data['to_emotion'] == "neutral"
        assert data['avg_duration'] == 900.0  # 15分钟的秒数
    
    def test_transition_from_dict(self):
        """测试从字典创建转换记录"""
        data = {
            'user_id': "test_user",
            'from_emotion': "fear",
            'to_emotion': "relief", 
            'transition_probability': 0.6,
            'occurrence_count': 2,
            'avg_duration': 600,  # 10分钟
            'updated_at': utc_now().isoformat(),
            'context_factors': ["support", "resolution"]
        }
        
        transition = EmotionTransition.from_dict(data)
        assert transition.from_emotion == "fear"
        assert transition.to_emotion == "relief"
        assert transition.avg_duration == timedelta(seconds=600)
        assert transition.context_factors == ["support", "resolution"]

class TestEmotionPrediction:
    """测试EmotionPrediction数据类"""
    
    def test_create_prediction(self):
        """测试创建情感预测"""
        prediction = EmotionPrediction(
            user_id="test_user",
            current_emotion="neutral",
            predicted_emotions=[("happiness", 0.6), ("sadness", 0.3)],
            confidence=0.8,
            time_horizon=timedelta(hours=2)
        )
        
        assert prediction.user_id == "test_user"
        assert prediction.current_emotion == "neutral"
        assert len(prediction.predicted_emotions) == 2
        assert prediction.confidence == 0.8
    
    def test_get_most_likely_emotion(self):
        """测试获取最可能的情感"""
        prediction = EmotionPrediction(
            predicted_emotions=[("happiness", 0.3), ("sadness", 0.7), ("anger", 0.2)]
        )
        
        most_likely = prediction.get_most_likely_emotion()
        assert most_likely == ("sadness", 0.7)
        
        # 测试空预测
        empty_prediction = EmotionPrediction()
        assert empty_prediction.get_most_likely_emotion() is None

class TestEmotionStatistics:
    """测试EmotionStatistics数据类"""
    
    def test_create_statistics(self):
        """测试创建情感统计"""
        start_time = utc_now() - timedelta(days=7)
        end_time = utc_now()
        
        stats = EmotionStatistics(
            user_id="test_user",
            time_period=(start_time, end_time),
            emotion_distribution={"happiness": 0.6, "sadness": 0.4},
            total_samples=100
        )
        
        assert stats.user_id == "test_user"
        assert stats.time_period == (start_time, end_time)
        assert stats.emotion_distribution["happiness"] == 0.6
        assert stats.total_samples == 100
    
    def test_statistics_to_dict(self):
        """测试统计信息转字典"""
        start_time = utc_now() - timedelta(days=1)
        end_time = utc_now()
        
        stats = EmotionStatistics(
            user_id="test_user",
            time_period=(start_time, end_time),
            intensity_stats={"mean": 0.65, "std": 0.2}
        )
        
        data = stats.to_dict()
        assert data['user_id'] == "test_user"
        assert len(data['time_period']) == 2
        assert data['intensity_stats'] == {"mean": 0.65, "std": 0.2}

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
