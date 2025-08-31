"""
Unit tests for Group Emotion Analysis System
群体情感分析系统单元测试
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np

from ai.emotion_modeling.group_emotion_models import (
    EmotionState,
    GroupEmotionalState,
    GroupEmotionAnalysisConfig,
    EmotionContagionType,
    GroupCohesionLevel
)
from ai.emotion_modeling.group_emotion_analyzer import GroupEmotionAnalyzer
from ai.emotion_modeling.contagion_detector import EmotionContagionDetector


class TestGroupEmotionModels:
    """测试群体情感数据模型"""
    
    def test_emotion_state_creation(self):
        """测试情感状态创建"""
        emotion_state = EmotionState(
            participant_id="user_1",
            emotion="joy",
            intensity=0.8,
            valence=0.9,
            arousal=0.7,
            dominance=0.6,
            timestamp=datetime.now(),
            confidence=0.85
        )
        
        assert emotion_state.participant_id == "user_1"
        assert emotion_state.emotion == "joy"
        assert emotion_state.intensity == 0.8
        assert 0 <= emotion_state.intensity <= 1
        assert -1 <= emotion_state.valence <= 1
        assert 0 <= emotion_state.arousal <= 1
        assert 0 <= emotion_state.dominance <= 1
    
    def test_group_emotional_state_creation(self):
        """测试群体情感状态创建"""
        participants = ["user_1", "user_2", "user_3"]
        emotion_distribution = {"joy": 0.6, "excitement": 0.3, "neutral": 0.1}
        
        group_state = GroupEmotionalState(
            group_id="test_group",
            timestamp=datetime.now(),
            participants=participants,
            dominant_emotion="joy",
            emotion_distribution=emotion_distribution,
            consensus_level=0.8,
            polarization_index=0.2,
            emotional_volatility=0.3,
            group_cohesion=GroupCohesionLevel.HIGH,
            emotional_leaders=[],
            influence_network={},
            contagion_patterns=[],
            contagion_velocity=0.5,
            trend_prediction="stable",
            stability_score=0.85
        )
        
        assert group_state.group_id == "test_group"
        assert len(group_state.participants) == 3
        assert group_state.dominant_emotion == "joy"
        assert abs(sum(group_state.emotion_distribution.values()) - 1.0) < 0.001
        assert group_state.consensus_level == 0.8
        assert group_state.group_cohesion == GroupCohesionLevel.HIGH


class TestGroupEmotionAnalyzer:
    """测试群体情感分析器"""
    
    @pytest.fixture
    def analyzer(self):
        """创建分析器实例"""
        config = GroupEmotionAnalysisConfig(
            intensity_weight=0.4,
            frequency_weight=0.3,
            influence_weight=0.3,
            consensus_threshold=0.7,
            polarization_threshold=0.6
        )
        return GroupEmotionAnalyzer(config)
    
    @pytest.fixture
    def sample_emotions(self):
        """创建示例情感数据"""
        return {
            "user_1": EmotionState(
                participant_id="user_1",
                emotion="joy",
                intensity=0.8,
                valence=0.9,
                arousal=0.7,
                dominance=0.8,
                timestamp=datetime.now()
            ),
            "user_2": EmotionState(
                participant_id="user_2",
                emotion="joy",
                intensity=0.7,
                valence=0.8,
                arousal=0.6,
                dominance=0.7,
                timestamp=datetime.now()
            ),
            "user_3": EmotionState(
                participant_id="user_3",
                emotion="excitement",
                intensity=0.9,
                valence=0.9,
                arousal=0.9,
                dominance=0.8,
                timestamp=datetime.now()
            )
        }
    
    @pytest.fixture
    def sample_interaction_history(self):
        """创建示例交互历史"""
        base_time = datetime.now() - timedelta(minutes=10)
        return [
            {
                "sender_id": "user_1",
                "detected_emotion": "joy",
                "emotion_intensity": 0.8,
                "timestamp": base_time,
                "has_responses": True,
                "responders": ["user_2", "user_3"]
            },
            {
                "sender_id": "user_2",
                "detected_emotion": "joy",
                "emotion_intensity": 0.7,
                "timestamp": base_time + timedelta(minutes=2),
                "has_responses": True,
                "responders": ["user_3"]
            },
            {
                "sender_id": "user_3",
                "detected_emotion": "excitement",
                "emotion_intensity": 0.9,
                "timestamp": base_time + timedelta(minutes=4),
                "has_responses": False
            }
        ]
    
    @pytest.mark.asyncio
    async def test_analyze_group_emotion_basic(self, analyzer, sample_emotions):
        """测试基本群体情感分析"""
        result = await analyzer.analyze_group_emotion(sample_emotions)
        
        assert isinstance(result, GroupEmotionalState)
        assert len(result.participants) == 3
        assert result.dominant_emotion in ["joy", "excitement"]
        assert 0 <= result.consensus_level <= 1
        assert 0 <= result.polarization_index <= 1
        assert 0 <= result.emotional_volatility <= 1
        assert 0 <= result.stability_score <= 1
    
    @pytest.mark.asyncio
    async def test_analyze_group_emotion_with_history(
        self, analyzer, sample_emotions, sample_interaction_history
    ):
        """测试带历史记录的群体情感分析"""
        result = await analyzer.analyze_group_emotion(
            sample_emotions, sample_interaction_history
        )
        
        assert isinstance(result, GroupEmotionalState)
        assert len(result.emotional_leaders) >= 0
        assert len(result.contagion_patterns) >= 0
        assert result.analysis_confidence > 0
    
    @pytest.mark.asyncio
    async def test_insufficient_participants(self, analyzer):
        """测试参与者不足的情况"""
        single_emotion = {
            "user_1": EmotionState(
                participant_id="user_1",
                emotion="joy",
                intensity=0.8,
                valence=0.9,
                arousal=0.7,
                dominance=0.8,
                timestamp=datetime.now()
            )
        }
        
        with pytest.raises(ValueError):
            await analyzer.analyze_group_emotion(single_emotion)
    
    def test_consensus_level_calculation(self, analyzer, sample_emotions):
        """测试共识水平计算"""
        consensus = analyzer._calculate_consensus_level(sample_emotions)
        
        assert 0 <= consensus <= 1
        # 由于测试数据中有两个joy和一个excitement，共识应该较高
        assert consensus > 0.5
    
    def test_polarization_index_calculation(self, analyzer):
        """测试极化指数计算"""
        # 创建高极化场景
        polarized_emotions = {
            "user_1": EmotionState(
                participant_id="user_1",
                emotion="joy",
                intensity=0.9,
                valence=1.0,
                arousal=0.8,
                dominance=0.9,
                timestamp=datetime.now()
            ),
            "user_2": EmotionState(
                participant_id="user_2",
                emotion="anger",
                intensity=0.9,
                valence=-1.0,
                arousal=0.9,
                dominance=0.8,
                timestamp=datetime.now()
            )
        }
        
        polarization = analyzer._calculate_polarization_index(polarized_emotions)
        assert 0 <= polarization <= 1
        # 高度对立的情感应该产生较高的极化指数
        assert polarization > 0.3
    
    def test_group_cohesion_determination(self, analyzer):
        """测试群体凝聚力判定"""
        # 高共识，低极化 -> 高凝聚力
        cohesion = analyzer._determine_group_cohesion(0.9, 0.1)
        assert cohesion == GroupCohesionLevel.HIGH
        
        # 低共识，高极化 -> 分化状态
        cohesion = analyzer._determine_group_cohesion(0.2, 0.8)
        assert cohesion == GroupCohesionLevel.FRAGMENTED
        
        # 中等水平 -> 中等凝聚力
        cohesion = analyzer._determine_group_cohesion(0.6, 0.4)
        assert cohesion == GroupCohesionLevel.MEDIUM


class TestEmotionContagionDetector:
    """测试情感传染检测器"""
    
    @pytest.fixture
    def detector(self):
        """创建检测器实例"""
        return EmotionContagionDetector(
            detection_window_seconds=300,
            min_contagion_threshold=0.3,
            max_propagation_delay=120.0
        )
    
    @pytest.fixture
    def contagion_scenario_emotions(self):
        """创建传染场景的情感数据"""
        return {
            "user_1": EmotionState(
                participant_id="user_1",
                emotion="excitement",
                intensity=0.9,
                valence=0.9,
                arousal=0.9,
                dominance=0.8,
                timestamp=datetime.now()
            ),
            "user_2": EmotionState(
                participant_id="user_2",
                emotion="excitement",
                intensity=0.8,
                valence=0.8,
                arousal=0.8,
                dominance=0.7,
                timestamp=datetime.now()
            ),
            "user_3": EmotionState(
                participant_id="user_3",
                emotion="excitement",
                intensity=0.7,
                valence=0.7,
                arousal=0.7,
                dominance=0.6,
                timestamp=datetime.now()
            )
        }
    
    @pytest.fixture
    def contagion_interaction_history(self):
        """创建传染交互历史"""
        base_time = datetime.now() - timedelta(minutes=5)
        return [
            {
                "sender_id": "user_1",
                "detected_emotion": "excitement",
                "emotion_intensity": 0.9,
                "timestamp": base_time,
                "type": "interaction"
            },
            {
                "sender_id": "user_2",
                "detected_emotion": "excitement",
                "emotion_intensity": 0.8,
                "timestamp": base_time + timedelta(seconds=30),
                "type": "interaction"
            },
            {
                "sender_id": "user_3",
                "detected_emotion": "excitement",
                "emotion_intensity": 0.7,
                "timestamp": base_time + timedelta(minutes=1),
                "type": "interaction"
            }
        ]
    
    @pytest.mark.asyncio
    async def test_detect_contagion_events(
        self, detector, contagion_scenario_emotions, contagion_interaction_history
    ):
        """测试情感传染事件检测"""
        events = await detector.detect_contagion_events(
            contagion_scenario_emotions,
            contagion_interaction_history,
            time_window_minutes=5
        )
        
        assert isinstance(events, list)
        # 应该能检测到至少一个传染事件
        if events:
            event = events[0]
            assert event.source_participant == "user_1"
            assert event.source_emotion == "excitement"
            assert len(event.affected_participants) >= 1
            assert 0 <= event.effectiveness_score <= 1
    
    def test_group_by_emotion(self, detector, contagion_scenario_emotions):
        """测试按情感分组"""
        groups = detector._group_by_emotion(contagion_scenario_emotions)
        
        assert "excitement" in groups
        assert len(groups["excitement"]) == 3
        assert all(user_id in groups["excitement"] for user_id in contagion_scenario_emotions.keys())
    
    def test_build_emotion_timeline(self, detector):
        """测试构建情感时间线"""
        # 模拟事件缓存
        test_events = [
            {
                "participant_id": "user_1",
                "emotion": "joy",
                "intensity": 0.8,
                "timestamp": datetime.now() - timedelta(minutes=2),
                "type": "emotion_state"
            },
            {
                "participant_id": "user_2",
                "emotion": "joy",
                "intensity": 0.7,
                "timestamp": datetime.now() - timedelta(minutes=1),
                "type": "emotion_state"
            }
        ]
        
        detector.event_buffer.extend(test_events)
        
        timeline = detector._build_emotion_timeline(
            "joy", ["user_1", "user_2"], time_window_minutes=5
        )
        
        assert len(timeline) == 2
        assert timeline[0]["timestamp"] < timeline[1]["timestamp"]  # 时间排序
    
    def test_contagion_type_determination(self, detector):
        """测试传染类型判定"""
        # 测试不同的强度变化模式
        
        # 放大型传染
        chain_amplification = [
            {"intensity": 0.5, "timestamp": datetime.now()},
            {"intensity": 0.7, "timestamp": datetime.now() + timedelta(seconds=30)},
            {"intensity": 0.8, "timestamp": datetime.now() + timedelta(seconds=60)}
        ]
        contagion_type = detector._determine_contagion_type_from_chain(chain_amplification)
        assert contagion_type in [EmotionContagionType.AMPLIFICATION, EmotionContagionType.CASCADE, EmotionContagionType.VIRAL]
        
        # 衰减型传染
        chain_dampening = [
            {"intensity": 0.8, "timestamp": datetime.now()},
            {"intensity": 0.6, "timestamp": datetime.now() + timedelta(seconds=30)},
            {"intensity": 0.4, "timestamp": datetime.now() + timedelta(seconds=60)}
        ]
        contagion_type = detector._determine_contagion_type_from_chain(chain_dampening)
        assert contagion_type == EmotionContagionType.DAMPENING
    
    def test_validate_contagion_chain(self, detector):
        """测试传染链条验证"""
        base_time = datetime.now()
        
        # 有效链条
        valid_chain = [
            {
                "participant_id": "user_1",
                "intensity": 0.8,
                "timestamp": base_time
            },
            {
                "participant_id": "user_2",
                "intensity": 0.7,
                "timestamp": base_time + timedelta(seconds=30)
            }
        ]
        assert detector._validate_contagion_chain(valid_chain) is True
        
        # 无效链条：时间间隔过长
        invalid_chain_time = [
            {
                "participant_id": "user_1",
                "intensity": 0.8,
                "timestamp": base_time
            },
            {
                "participant_id": "user_2",
                "intensity": 0.7,
                "timestamp": base_time + timedelta(minutes=10)  # 超过max_delay
            }
        ]
        assert detector._validate_contagion_chain(invalid_chain_time) is False
        
        # 无效链条：强度过低
        invalid_chain_intensity = [
            {
                "participant_id": "user_1",
                "intensity": 0.1,  # 低于阈值
                "timestamp": base_time
            },
            {
                "participant_id": "user_2",
                "intensity": 0.1,
                "timestamp": base_time + timedelta(seconds=30)
            }
        ]
        assert detector._validate_contagion_chain(invalid_chain_intensity) is False
    
    def test_network_statistics(self, detector):
        """测试网络统计"""
        stats = detector.get_network_statistics()
        
        assert "total_networks" in stats
        assert "total_affected_participants" in stats
        assert "average_coverage_rate" in stats
        assert "most_contagious_emotion" in stats
        assert "network_efficiency" in stats
        
        # 空网络的情况
        assert stats["total_networks"] == 0
        assert stats["total_affected_participants"] == 0
        assert stats["average_coverage_rate"] == 0.0


class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_group_analysis_with_contagion_detection(self):
        """测试群体分析与传染检测的集成"""
        # 创建分析器和检测器
        analyzer = GroupEmotionAnalyzer()
        detector = EmotionContagionDetector()
        
        # 创建测试数据
        emotions = {
            "user_1": EmotionState(
                participant_id="user_1",
                emotion="joy",
                intensity=0.9,
                valence=0.9,
                arousal=0.8,
                dominance=0.8,
                timestamp=datetime.now()
            ),
            "user_2": EmotionState(
                participant_id="user_2",
                emotion="joy",
                intensity=0.8,
                valence=0.8,
                arousal=0.7,
                dominance=0.7,
                timestamp=datetime.now()
            ),
            "user_3": EmotionState(
                participant_id="user_3",
                emotion="joy",
                intensity=0.7,
                valence=0.7,
                arousal=0.6,
                dominance=0.6,
                timestamp=datetime.now()
            )
        }
        
        interaction_history = [
            {
                "sender_id": "user_1",
                "detected_emotion": "joy",
                "emotion_intensity": 0.9,
                "timestamp": datetime.now() - timedelta(minutes=2),
                "has_responses": True,
                "responders": ["user_2", "user_3"]
            }
        ]
        
        # 执行群体分析
        group_state = await analyzer.analyze_group_emotion(emotions, interaction_history)
        
        # 执行传染检测
        contagion_events = await detector.detect_contagion_events(
            emotions, interaction_history
        )
        
        # 验证结果
        assert isinstance(group_state, GroupEmotionalState)
        assert group_state.dominant_emotion == "joy"
        assert group_state.consensus_level > 0.7  # 高共识
        assert group_state.polarization_index < 0.3  # 低极化
        
        assert isinstance(contagion_events, list)
        # 可能检测到传染事件
        if contagion_events:
            assert contagion_events[0].source_emotion == "joy"
    
    @pytest.mark.asyncio
    async def test_performance_requirements(self):
        """测试性能要求"""
        import time
        
        analyzer = GroupEmotionAnalyzer()
        
        # 创建大量测试数据
        emotions = {}
        for i in range(5):  # 5个参与者
            emotions[f"user_{i}"] = EmotionState(
                participant_id=f"user_{i}",
                emotion="joy",
                intensity=0.5 + (i * 0.1),
                valence=0.5,
                arousal=0.5,
                dominance=0.5,
                timestamp=datetime.now()
            )
        
        # 测量分析时间
        start_time = time.time()
        result = await analyzer.analyze_group_emotion(emotions)
        end_time = time.time()
        
        analysis_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 验证性能要求：5人以内群体分析 < 300ms
        assert analysis_time < 300, f"分析时间 {analysis_time}ms 超过300ms要求"
        assert isinstance(result, GroupEmotionalState)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])