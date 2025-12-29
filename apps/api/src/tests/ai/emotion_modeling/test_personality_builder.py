"""
测试个性化画像构建器
"""

from src.core.utils.timezone_utils import utc_now
import pytest
from datetime import timedelta
import numpy as np
from src.ai.emotion_modeling.personality_builder import PersonalityProfileBuilder
from src.ai.emotion_modeling.models import EmotionState, PersonalityTrait

class TestPersonalityProfileBuilder:
    """测试PersonalityProfileBuilder类"""
    
    def setup_method(self):
        """测试前设置"""
        self.builder = PersonalityProfileBuilder()
    
    def test_init(self):
        """测试初始化"""
        assert self.builder.space_mapper is not None
        assert len(self.builder.trait_weights) == 5  # Big Five特质
        assert len(self.builder.emotion_categories) > 0
    
    @pytest.mark.asyncio
    async def test_build_personality_profile_empty(self):
        """测试空历史数据的画像构建"""
        profile = await self.builder.build_personality_profile("user1", [])
        
        assert profile.user_id == "user1"
        assert profile.sample_count == 0
        assert profile.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_build_personality_profile_normal(self):
        """测试正常画像构建"""
        # 创建测试数据
        states = []
        base_time = utc_now()
        
        for i in range(20):
            state = EmotionState(
                user_id="user1",
                emotion="happiness" if i % 2 == 0 else "sadness",
                intensity=0.5 + i * 0.02,
                valence=0.5 if i % 2 == 0 else -0.3,
                arousal=0.6,
                dominance=0.5,
                confidence=0.8,
                timestamp=base_time + timedelta(hours=i),
                triggers=["test_trigger"]
            )
            states.append(state)
        
        profile = await self.builder.build_personality_profile("user1", states)
        
        assert profile.user_id == "user1"
        assert profile.sample_count == 20
        assert profile.confidence_score > 0
        assert len(profile.emotional_traits) == 5  # Big Five
        assert len(profile.baseline_emotions) > 0
        assert profile.emotion_volatility >= 0
        assert profile.recovery_rate >= 0
    
    @pytest.mark.asyncio
    async def test_compute_big_five_traits_empty(self):
        """测试空数据的特质计算"""
        traits = await self.builder.compute_big_five_traits([])
        
        assert len(traits) == 5
        for trait in PersonalityTrait:
            assert trait.value in traits
            assert traits[trait.value] == 0.5  # 默认中等水平
    
    @pytest.mark.asyncio
    async def test_compute_big_five_traits_normal(self):
        """测试正常特质计算"""
        # 创建带有不同特质倾向的测试数据
        states = []
        base_time = utc_now()
        
        # 创建高外向性数据（积极高唤醒）
        for i in range(10):
            state = EmotionState(
                emotion="joy",
                intensity=0.8,
                valence=0.8,
                arousal=0.9,
                dominance=0.7,
                timestamp=base_time + timedelta(hours=i),
                confidence=0.9
            )
            states.append(state)
        
        traits = await self.builder.compute_big_five_traits(states)
        
        # 验证结果
        assert len(traits) == 5
        for trait_name, score in traits.items():
            assert 0 <= score <= 1
        
        # 外向性应该较高（因为都是积极高唤醒情感）
        assert traits[PersonalityTrait.EXTRAVERSION.value] > 0.3
    
    def test_compute_baseline_emotions_empty(self):
        """测试空数据的基线情感计算"""
        baseline = self.builder.compute_baseline_emotions([])
        assert baseline == {}
    
    def test_compute_baseline_emotions_normal(self):
        """测试正常基线情感计算"""
        states = [
            EmotionState(emotion="happiness", intensity=0.8),
            EmotionState(emotion="happiness", intensity=0.7),
            EmotionState(emotion="sadness", intensity=0.6),
        ]
        
        baseline = self.builder.compute_baseline_emotions(states)
        
        assert "happiness" in baseline
        assert "sadness" in baseline
        assert abs(baseline["happiness"] - 2/3) < 0.01  # 2个happiness，总共3个
        assert abs(baseline["sadness"] - 1/3) < 0.01    # 1个sadness，总共3个
        
        # 频率总和应该为1
        assert abs(sum(baseline.values()) - 1.0) < 0.001
    
    def test_compute_emotion_volatility_empty(self):
        """测试空数据的波动性计算"""
        volatility = self.builder.compute_emotion_volatility([])
        assert volatility == 0.5  # 默认值
    
    def test_compute_emotion_volatility_normal(self):
        """测试正常波动性计算"""
        # 创建低波动性数据
        low_volatility_states = [
            EmotionState(emotion="neutral", intensity=0.5, valence=0, arousal=0.3, dominance=0.5),
            EmotionState(emotion="neutral", intensity=0.5, valence=0, arousal=0.3, dominance=0.5)
        ]
        
        # 创建高波动性数据
        high_volatility_states = [
            EmotionState(emotion="happiness", intensity=1.0, valence=0.8, arousal=0.8, dominance=0.8),
            EmotionState(emotion="sadness", intensity=0.1, valence=-0.8, arousal=0.2, dominance=0.2)
        ]
        
        low_vol = self.builder.compute_emotion_volatility(low_volatility_states)
        high_vol = self.builder.compute_emotion_volatility(high_volatility_states)
        
        assert 0 <= low_vol <= 1
        assert 0 <= high_vol <= 1
        assert high_vol > low_vol  # 高波动数据应该有更高的波动性分数
    
    def test_identify_dominant_emotions_empty(self):
        """测试空数据的主导情感识别"""
        dominant = self.builder.identify_dominant_emotions([])
        assert dominant == []
    
    def test_identify_dominant_emotions_normal(self):
        """测试正常主导情感识别"""
        states = [
            EmotionState(emotion="happiness", intensity=0.8, confidence=0.9),
            EmotionState(emotion="happiness", intensity=0.7, confidence=0.8),
            EmotionState(emotion="sadness", intensity=0.6, confidence=0.7),
            EmotionState(emotion="joy", intensity=0.9, confidence=0.9),
        ]
        
        dominant = self.builder.identify_dominant_emotions(states, top_k=2)
        
        assert len(dominant) <= 2
        assert "happiness" in dominant  # 应该包含happiness（出现2次且强度高）
    
    def test_identify_trigger_patterns_empty(self):
        """测试空数据的触发模式识别"""
        patterns = self.builder.identify_trigger_patterns([])
        assert patterns == {}
    
    def test_identify_trigger_patterns_normal(self):
        """测试正常触发模式识别"""
        states = [
            EmotionState(emotion="happiness", triggers=["success", "achievement"]),
            EmotionState(emotion="happiness", triggers=["success", "praise"]),
            EmotionState(emotion="sadness", triggers=["failure", "criticism"]),
        ]
        
        patterns = self.builder.identify_trigger_patterns(states)
        
        assert "happiness" in patterns
        assert "sadness" in patterns
        assert "success" in patterns["happiness"]
        assert "failure" in patterns["sadness"]
    
    def test_calculate_profile_confidence_empty(self):
        """测试空数据的置信度计算"""
        confidence = self.builder.calculate_profile_confidence([])
        assert confidence == 0.0
    
    def test_calculate_profile_confidence_normal(self):
        """测试正常置信度计算"""
        base_time = utc_now() - timedelta(days=15)  # 15天前开始
        
        states = []
        for i in range(50):  # 50个样本
            state = EmotionState(
                emotion="happiness",
                intensity=0.8,
                confidence=0.9,
                timestamp=base_time + timedelta(hours=i*6)  # 每6小时一个
            )
            states.append(state)
        
        confidence = self.builder.calculate_profile_confidence(states)
        
        assert 0 <= confidence <= 1
        assert confidence > 0.3  # 应该有较高置信度（样本多且时间跨度长）

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
