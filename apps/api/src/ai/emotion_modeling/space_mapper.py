"""
三维情感空间建模引擎 (VAD空间映射)

基于Valence-Arousal-Dominance三维情感空间理论进行情感建模
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import euclidean, cosine
from sklearn.metrics.pairwise import pairwise_distances
from .models import EmotionState, EmotionType

from src.core.logging import get_logger
logger = get_logger(__name__)

class EmotionSpaceMapper:
    """情感空间映射器"""
    
    def __init__(self):
        """初始化情感空间映射"""
        # 基础情感到VAD空间的映射 (valence, arousal, dominance)
        # 基于心理学研究的标准化VAD数值
        self.emotion_dimensions = {
            # 基础情感
            EmotionType.HAPPINESS.value: (0.8, 0.6, 0.7),
            EmotionType.SADNESS.value: (-0.7, 0.4, 0.3),
            EmotionType.ANGER.value: (-0.6, 0.9, 0.8),
            EmotionType.FEAR.value: (-0.8, 0.8, 0.2),
            EmotionType.SURPRISE.value: (0.2, 0.8, 0.5),
            EmotionType.DISGUST.value: (-0.7, 0.5, 0.6),
            EmotionType.NEUTRAL.value: (0.0, 0.3, 0.5),
            
            # 扩展情感
            EmotionType.JOY.value: (0.9, 0.7, 0.8),
            EmotionType.TRUST.value: (0.6, 0.4, 0.6),
            EmotionType.ANTICIPATION.value: (0.4, 0.7, 0.6),
            EmotionType.CONTEMPT.value: (-0.5, 0.4, 0.8),
            EmotionType.SHAME.value: (-0.6, 0.5, 0.2),
            EmotionType.GUILT.value: (-0.5, 0.6, 0.3),
            EmotionType.PRIDE.value: (0.7, 0.6, 0.9),
            EmotionType.ENVY.value: (-0.4, 0.7, 0.4),
            EmotionType.LOVE.value: (0.9, 0.5, 0.6),
            EmotionType.GRATITUDE.value: (0.8, 0.4, 0.5),
            EmotionType.HOPE.value: (0.6, 0.6, 0.4),
            EmotionType.ANXIETY.value: (-0.6, 0.8, 0.3),
            EmotionType.DEPRESSION.value: (-0.8, 0.2, 0.2)
        }
        
        # 情感强度影响因子
        self.intensity_impact = {
            'valence': 1.0,    # 强度对效价的影响
            'arousal': 0.8,    # 强度对唤醒度的影响  
            'dominance': 0.6   # 强度对支配性的影响
        }
    
    def map_emotion_to_space(
        self, 
        emotion: str, 
        intensity: float = 1.0
    ) -> Tuple[float, float, float]:
        """
        将情感映射到VAD三维空间
        
        Args:
            emotion: 情感类型
            intensity: 情感强度 [0,1]
            
        Returns:
            (valence, arousal, dominance) 三元组
        """
        base_dims = self.emotion_dimensions.get(emotion, (0.0, 0.3, 0.5))
        
        # 根据强度调整各维度
        valence = base_dims[0] * intensity * self.intensity_impact['valence']
        arousal = base_dims[1] * intensity * self.intensity_impact['arousal']
        dominance = base_dims[2] * intensity * self.intensity_impact['dominance']
        
        # 确保值在合理范围内
        valence = max(-1.0, min(1.0, valence))
        arousal = max(0.0, min(1.0, arousal))
        dominance = max(0.0, min(1.0, dominance))
        
        return (valence, arousal, dominance)
    
    def map_state_to_space(self, state: EmotionState) -> Tuple[float, float, float]:
        """将情感状态映射到VAD空间"""
        return self.map_emotion_to_space(state.emotion, state.intensity)
    
    def calculate_emotion_distance(
        self, 
        state1: EmotionState, 
        state2: EmotionState,
        metric: str = 'euclidean'
    ) -> float:
        """
        计算两个情感状态在VAD空间中的距离
        
        Args:
            state1: 第一个情感状态
            state2: 第二个情感状态
            metric: 距离度量方式 ('euclidean', 'cosine', 'manhattan')
            
        Returns:
            距离值
        """
        point1 = np.array(self.map_state_to_space(state1))
        point2 = np.array(self.map_state_to_space(state2))
        
        if metric == 'euclidean':
            return euclidean(point1, point2)
        elif metric == 'cosine':
            return cosine(point1, point2)
        elif metric == 'manhattan':
            return np.sum(np.abs(point1 - point2))
        else:
            raise ValueError(f"不支持的距离度量: {metric}")
    
    def find_similar_emotions(
        self, 
        target_emotion: str,
        target_intensity: float = 1.0,
        threshold: float = 0.5,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        找到在VAD空间中相似的情感
        
        Args:
            target_emotion: 目标情感
            target_intensity: 目标强度
            threshold: 距离阈值
            top_k: 返回最相似的k个情感
            
        Returns:
            [(情感, 相似度分数)] 列表
        """
        target_point = np.array(self.map_emotion_to_space(target_emotion, target_intensity))
        similarities = []
        
        for emotion, base_dims in self.emotion_dimensions.items():
            if emotion == target_emotion:
                continue
                
            emotion_point = np.array(self.map_emotion_to_space(emotion, target_intensity))
            distance = euclidean(target_point, emotion_point)
            
            # 转换距离为相似度分数 (0-1, 越高越相似)
            max_distance = np.sqrt(3)  # VAD空间最大距离
            similarity = 1 - (distance / max_distance)
            
            if similarity >= threshold:
                similarities.append((emotion, similarity))
        
        # 按相似度排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_emotion_quadrant(self, state: EmotionState) -> str:
        """
        根据VAD空间确定情感象限
        
        Args:
            state: 情感状态
            
        Returns:
            象限名称
        """
        valence, arousal, dominance = self.map_state_to_space(state)
        
        if valence > 0 and arousal > 0.5:
            return "高兴-激活" if dominance > 0.5 else "高兴-被动"
        elif valence > 0 and arousal <= 0.5:
            return "平静-满足" if dominance > 0.5 else "放松-满足"
        elif valence <= 0 and arousal > 0.5:
            return "愤怒-激动" if dominance > 0.5 else "恐惧-焦虑"
        else:
            return "沮丧-控制" if dominance > 0.5 else "悲伤-无助"
    
    def calculate_emotional_vector(self, states: List[EmotionState]) -> np.ndarray:
        """
        计算一系列情感状态的向量表示
        
        Args:
            states: 情感状态列表
            
        Returns:
            情感向量
        """
        if not states:
            return np.zeros(3)
        
        points = []
        weights = []
        
        for state in states:
            point = self.map_state_to_space(state)
            points.append(point)
            # 使用置信度作为权重
            weights.append(state.confidence)
        
        points = np.array(points)
        weights = np.array(weights)
        
        # 计算加权平均
        if np.sum(weights) > 0:
            weighted_vector = np.average(points, axis=0, weights=weights)
        else:
            weighted_vector = np.mean(points, axis=0)
        
        return weighted_vector
    
    def detect_emotion_clusters(
        self, 
        states: List[EmotionState],
        n_clusters: int = 5,
        min_cluster_size: int = 3
    ) -> List[Dict]:
        """
        在VAD空间中检测情感聚类
        
        Args:
            states: 情感状态列表
            n_clusters: 聚类数量
            min_cluster_size: 最小聚类大小
            
        Returns:
            聚类结果列表
        """
        if len(states) < min_cluster_size:
            return []
        
        try:
            from sklearn.cluster import KMeans
            from sklearn.preprocessing import StandardScaler
            
            # 提取VAD特征
            features = []
            for state in states:
                point = self.map_state_to_space(state)
                features.append([point[0], point[1], point[2], state.intensity])
            
            features = np.array(features)
            
            # 标准化特征
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # K-means聚类
            kmeans = KMeans(n_clusters=min(n_clusters, len(states)), random_state=42)
            cluster_labels = kmeans.fit_predict(features_scaled)
            
            # 分析聚类结果
            clusters = []
            for i in range(n_clusters):
                cluster_indices = np.where(cluster_labels == i)[0]
                
                if len(cluster_indices) < min_cluster_size:
                    continue
                
                cluster_states = [states[idx] for idx in cluster_indices]
                
                # 计算聚类中心
                centroid = kmeans.cluster_centers_[i]
                centroid_original = scaler.inverse_transform([centroid])[0]
                
                # 分析聚类特征
                emotions_in_cluster = [state.emotion for state in cluster_states]
                dominant_emotions = list(set(emotions_in_cluster))
                
                clusters.append({
                    'cluster_id': i,
                    'size': len(cluster_states),
                    'centroid_valence': centroid_original[0],
                    'centroid_arousal': centroid_original[1], 
                    'centroid_dominance': centroid_original[2],
                    'avg_intensity': centroid_original[3],
                    'dominant_emotions': dominant_emotions,
                    'states': cluster_states
                })
            
            return clusters
            
        except ImportError:
            logger.warning("sklearn未安装，无法进行聚类分析")
            return []
        except Exception as e:
            logger.error(f"情感聚类分析失败: {e}")
            return []
    
    def get_space_boundaries(self) -> Dict[str, Tuple[float, float]]:
        """获取VAD空间的边界值"""
        return {
            'valence': (-1.0, 1.0),
            'arousal': (0.0, 1.0),
            'dominance': (0.0, 1.0)
        }
    
    def normalize_coordinates(
        self, 
        valence: float, 
        arousal: float, 
        dominance: float
    ) -> Tuple[float, float, float]:
        """标准化VAD坐标到有效范围"""
        boundaries = self.get_space_boundaries()
        
        valence = max(boundaries['valence'][0], 
                     min(boundaries['valence'][1], valence))
        arousal = max(boundaries['arousal'][0], 
                     min(boundaries['arousal'][1], arousal))
        dominance = max(boundaries['dominance'][0], 
                       min(boundaries['dominance'][1], dominance))
        
        return (valence, arousal, dominance)
