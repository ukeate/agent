"""
测试情感空间映射器
"""
import pytest
import numpy as np
from datetime import datetime
from src.ai.emotion_modeling.space_mapper import EmotionSpaceMapper
from src.ai.emotion_modeling.models import EmotionState, EmotionType


class TestEmotionSpaceMapper:
    """测试EmotionSpaceMapper类"""
    
    def setup_method(self):
        """测试前设置"""
        self.mapper = EmotionSpaceMapper()
    
    def test_init(self):
        """测试初始化"""
        assert len(self.mapper.emotion_dimensions) > 0
        assert EmotionType.HAPPINESS.value in self.mapper.emotion_dimensions
        assert EmotionType.SADNESS.value in self.mapper.emotion_dimensions
    
    def test_map_emotion_to_space(self):
        """测试情感到空间映射"""
        # 测试基础情感映射
        vad = self.mapper.map_emotion_to_space("happiness", 1.0)
        assert len(vad) == 3
        assert -1.0 <= vad[0] <= 1.0  # valence
        assert 0.0 <= vad[1] <= 1.0   # arousal
        assert 0.0 <= vad[2] <= 1.0   # dominance
        
        # 测试强度对映射的影响
        vad_full = self.mapper.map_emotion_to_space("happiness", 1.0)
        vad_half = self.mapper.map_emotion_to_space("happiness", 0.5)
        
        # 强度减半应该影响各维度
        assert abs(vad_half[0]) < abs(vad_full[0])
        assert vad_half[1] < vad_full[1]
        assert vad_half[2] < vad_full[2]
    
    def test_map_unknown_emotion(self):
        """测试未知情感的映射"""
        vad = self.mapper.map_emotion_to_space("unknown_emotion", 1.0)
        # 应该返回中性情感的坐标
        expected = self.mapper.emotion_dimensions[EmotionType.NEUTRAL.value]
        np.testing.assert_array_almost_equal(vad, expected, decimal=2)
    
    def test_map_state_to_space(self):
        """测试情感状态到空间映射"""
        state = EmotionState(
            emotion="happiness",
            intensity=0.8
        )
        vad = self.mapper.map_state_to_space(state)
        expected = self.mapper.map_emotion_to_space("happiness", 0.8)
        assert vad == expected
    
    def test_calculate_emotion_distance(self):
        """测试情感距离计算"""
        state1 = EmotionState(emotion="happiness", intensity=1.0)
        state2 = EmotionState(emotion="sadness", intensity=1.0)
        state3 = EmotionState(emotion="happiness", intensity=1.0)
        
        # 不同情感的距离应该大于相同情感的距离
        dist_different = self.mapper.calculate_emotion_distance(state1, state2)
        dist_same = self.mapper.calculate_emotion_distance(state1, state3)
        
        assert dist_different > dist_same
        assert dist_same == 0.0  # 相同状态距离为0
    
    def test_calculate_emotion_distance_metrics(self):
        """测试不同距离度量"""
        state1 = EmotionState(emotion="happiness", intensity=1.0)
        state2 = EmotionState(emotion="anger", intensity=1.0)
        
        euclidean_dist = self.mapper.calculate_emotion_distance(state1, state2, 'euclidean')
        cosine_dist = self.mapper.calculate_emotion_distance(state1, state2, 'cosine')
        manhattan_dist = self.mapper.calculate_emotion_distance(state1, state2, 'manhattan')
        
        # 所有距离都应该为正数
        assert euclidean_dist > 0
        assert cosine_dist >= 0
        assert manhattan_dist > 0
        
        # 测试无效度量
        with pytest.raises(ValueError):
            self.mapper.calculate_emotion_distance(state1, state2, 'invalid_metric')
    
    def test_find_similar_emotions(self):
        """测试查找相似情感"""
        similar = self.mapper.find_similar_emotions("happiness", 1.0, threshold=0.3, top_k=3)
        
        assert isinstance(similar, list)
        assert len(similar) <= 3
        
        if similar:
            for emotion, similarity in similar:
                assert isinstance(emotion, str)
                assert 0.0 <= similarity <= 1.0
                assert emotion != "happiness"  # 不应包含原始情感
            
            # 应该按相似度降序排列
            similarities = [sim for _, sim in similar]
            assert similarities == sorted(similarities, reverse=True)
    
    def test_get_emotion_quadrant(self):
        """测试情感象限判断"""
        # 测试不同象限的情感
        happy_state = EmotionState(emotion="happiness", intensity=1.0)
        sad_state = EmotionState(emotion="sadness", intensity=1.0)
        angry_state = EmotionState(emotion="anger", intensity=1.0)
        
        happy_quadrant = self.mapper.get_emotion_quadrant(happy_state)
        sad_quadrant = self.mapper.get_emotion_quadrant(sad_state)
        angry_quadrant = self.mapper.get_emotion_quadrant(angry_state)
        
        assert isinstance(happy_quadrant, str)
        assert isinstance(sad_quadrant, str)
        assert isinstance(angry_quadrant, str)
        
        # 不同情感应该在不同象限
        assert happy_quadrant != sad_quadrant
        assert happy_quadrant != angry_quadrant
    
    def test_calculate_emotional_vector(self):
        """测试情感向量计算"""
        states = [
            EmotionState(emotion="happiness", intensity=0.8, confidence=0.9),
            EmotionState(emotion="joy", intensity=0.7, confidence=0.8),
            EmotionState(emotion="neutral", intensity=0.5, confidence=0.7)
        ]
        
        vector = self.mapper.calculate_emotional_vector(states)
        
        assert isinstance(vector, np.ndarray)
        assert len(vector) == 3  # VAD维度
        
        # 测试空列表
        empty_vector = self.mapper.calculate_emotional_vector([])
        np.testing.assert_array_equal(empty_vector, np.zeros(3))
    
    def test_get_space_boundaries(self):
        """测试空间边界获取"""
        boundaries = self.mapper.get_space_boundaries()
        
        assert 'valence' in boundaries
        assert 'arousal' in boundaries
        assert 'dominance' in boundaries
        
        assert boundaries['valence'] == (-1.0, 1.0)
        assert boundaries['arousal'] == (0.0, 1.0)
        assert boundaries['dominance'] == (0.0, 1.0)
    
    def test_normalize_coordinates(self):
        """测试坐标标准化"""
        # 测试边界内的坐标
        normalized = self.mapper.normalize_coordinates(0.5, 0.7, 0.3)
        assert normalized == (0.5, 0.7, 0.3)
        
        # 测试超出边界的坐标
        normalized = self.mapper.normalize_coordinates(2.0, -0.5, 1.5)
        assert normalized == (1.0, 0.0, 1.0)  # 应该被截断到边界
        
        normalized = self.mapper.normalize_coordinates(-2.0, 0.3, 0.5)
        assert normalized == (-1.0, 0.3, 0.5)  # valence应该被截断到-1.0
    
    def test_detect_emotion_clusters_insufficient_data(self):
        """测试数据不足时的聚类"""
        states = [
            EmotionState(emotion="happiness", intensity=0.8),
            EmotionState(emotion="sadness", intensity=0.6)
        ]
        
        clusters = self.mapper.detect_emotion_clusters(states, min_cluster_size=3)
        assert clusters == []  # 数据不足应该返回空列表
    
    @pytest.mark.skipif(True, reason="需要sklearn支持")
    def test_detect_emotion_clusters_with_sklearn(self):
        """测试带sklearn的聚类分析（如果可用）"""
        # 创建足够的测试数据
        states = []
        emotions = ["happiness", "sadness", "anger", "fear", "joy"]
        
        for i in range(15):  # 创建15个状态
            emotion = emotions[i % len(emotions)]
            states.append(EmotionState(
                emotion=emotion,
                intensity=0.5 + (i % 5) * 0.1,
                confidence=0.8
            ))
        
        clusters = self.mapper.detect_emotion_clusters(states, n_clusters=3, min_cluster_size=2)
        
        # 如果sklearn可用且有足够数据，应该返回聚类结果
        if clusters:
            assert isinstance(clusters, list)
            for cluster in clusters:
                assert 'cluster_id' in cluster
                assert 'size' in cluster
                assert 'dominant_emotions' in cluster
                assert cluster['size'] >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])