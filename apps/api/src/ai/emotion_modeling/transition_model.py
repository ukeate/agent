"""
情感状态转换建模系统

基于马尔可夫链的情感状态转换概率建模和预测
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

from .models import EmotionState, EmotionTransition, EmotionType

logger = logging.getLogger(__name__)


class TransitionModelManager:
    """情感状态转换模型管理器"""
    
    def __init__(self, smoothing_alpha: float = 0.01):
        """
        初始化转换模型管理器
        
        Args:
            smoothing_alpha: 拉普拉斯平滑参数，避免零概率
        """
        self.transition_matrices: Dict[str, np.ndarray] = {}
        self.smoothing_alpha = smoothing_alpha
        
        # 情感类型列表 (确保顺序一致)
        self.emotions = [
            EmotionType.HAPPINESS.value,
            EmotionType.SADNESS.value,
            EmotionType.ANGER.value,
            EmotionType.FEAR.value,
            EmotionType.SURPRISE.value,
            EmotionType.DISGUST.value,
            EmotionType.NEUTRAL.value,
            EmotionType.JOY.value,
            EmotionType.TRUST.value,
            EmotionType.ANTICIPATION.value,
            EmotionType.CONTEMPT.value,
            EmotionType.SHAME.value,
            EmotionType.GUILT.value,
            EmotionType.PRIDE.value,
            EmotionType.ENVY.value,
            EmotionType.LOVE.value,
            EmotionType.GRATITUDE.value,
            EmotionType.HOPE.value,
            EmotionType.ANXIETY.value,
            EmotionType.DEPRESSION.value
        ]
        
        self.emotion_to_index = {emotion: i for i, emotion in enumerate(self.emotions)}
        self.n_emotions = len(self.emotions)
    
    def _count_transitions(self, history: List[EmotionState]) -> np.ndarray:
        """
        统计情感状态转换次数
        
        Args:
            history: 按时间排序的情感状态历史
            
        Returns:
            转换计数矩阵 [n_emotions x n_emotions]
        """
        transition_counts = np.zeros((self.n_emotions, self.n_emotions))
        
        for i in range(len(history) - 1):
            from_emotion = history[i].emotion
            to_emotion = history[i + 1].emotion
            
            if from_emotion in self.emotion_to_index and to_emotion in self.emotion_to_index:
                from_idx = self.emotion_to_index[from_emotion]
                to_idx = self.emotion_to_index[to_emotion]
                
                # 考虑置信度加权
                weight = min(history[i].confidence, history[i + 1].confidence)
                transition_counts[from_idx, to_idx] += weight
        
        return transition_counts
    
    def _normalize_transitions(self, transition_counts: np.ndarray) -> np.ndarray:
        """
        将转换计数矩阵归一化为概率矩阵
        
        Args:
            transition_counts: 转换计数矩阵
            
        Returns:
            转换概率矩阵
        """
        # 拉普拉斯平滑
        smoothed_counts = transition_counts + self.smoothing_alpha
        
        # 归一化每一行（从某个情感出发的转换概率和为1）
        row_sums = smoothed_counts.sum(axis=1)
        # 避免除以零
        row_sums[row_sums == 0] = 1
        
        probability_matrix = smoothed_counts / row_sums[:, np.newaxis]
        
        return probability_matrix
    
    def _apply_time_decay(
        self, 
        history: List[EmotionState],
        decay_factor: float = 0.95
    ) -> List[float]:
        """
        应用时间衰减权重，最近的转换有更高权重
        
        Args:
            history: 情感状态历史
            decay_factor: 衰减因子 [0, 1]
            
        Returns:
            时间权重列表
        """
        if not history:
            return []
        
        # 计算时间权重
        now = datetime.now()
        weights = []
        
        for state in history:
            time_diff = (now - state.timestamp).total_seconds()
            # 时间越近，权重越高
            weight = decay_factor ** (time_diff / 3600)  # 小时为单位衰减
            weights.append(weight)
        
        return weights
    
    async def update_transition_model(
        self, 
        user_id: str, 
        history: List[EmotionState],
        use_time_decay: bool = True
    ) -> bool:
        """
        更新用户的情感转换模型
        
        Args:
            user_id: 用户ID
            history: 情感状态历史（按时间排序）
            use_time_decay: 是否使用时间衰减
            
        Returns:
            是否更新成功
        """
        try:
            if len(history) < 2:
                logger.warning(f"用户 {user_id} 的情感历史不足，无法构建转换模型")
                return False
            
            # 按时间排序
            history_sorted = sorted(history, key=lambda x: x.timestamp)
            
            # 统计转换次数
            if use_time_decay:
                # 使用时间衰减权重
                weights = self._apply_time_decay(history_sorted)
                transition_counts = self._count_weighted_transitions(history_sorted, weights)
            else:
                transition_counts = self._count_transitions(history_sorted)
            
            # 归一化为概率
            transition_matrix = self._normalize_transitions(transition_counts)
            
            # 存储用户的转换矩阵
            self.transition_matrices[user_id] = transition_matrix
            
            logger.info(f"已更新用户 {user_id} 的情感转换模型")
            return True
            
        except Exception as e:
            logger.error(f"更新转换模型失败: {e}")
            return False
    
    def _count_weighted_transitions(
        self, 
        history: List[EmotionState], 
        weights: List[float]
    ) -> np.ndarray:
        """使用权重统计转换次数"""
        transition_counts = np.zeros((self.n_emotions, self.n_emotions))
        
        for i in range(len(history) - 1):
            from_emotion = history[i].emotion
            to_emotion = history[i + 1].emotion
            
            if from_emotion in self.emotion_to_index and to_emotion in self.emotion_to_index:
                from_idx = self.emotion_to_index[from_emotion]
                to_idx = self.emotion_to_index[to_emotion]
                
                # 使用时间权重和置信度权重
                weight = weights[i] * min(history[i].confidence, history[i + 1].confidence)
                transition_counts[from_idx, to_idx] += weight
        
        return transition_counts
    
    def predict_next_emotion(
        self, 
        user_id: str, 
        current_emotion: str,
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        预测下一个最可能的情感状态
        
        Args:
            user_id: 用户ID
            current_emotion: 当前情感状态
            top_k: 返回前k个最可能的情感
            
        Returns:
            [(情感, 概率)] 列表，按概率降序排列
        """
        if user_id not in self.transition_matrices:
            logger.warning(f"用户 {user_id} 没有转换模型")
            return []
        
        if current_emotion not in self.emotion_to_index:
            logger.warning(f"未知情感类型: {current_emotion}")
            return []
        
        try:
            matrix = self.transition_matrices[user_id]
            current_idx = self.emotion_to_index[current_emotion]
            probabilities = matrix[current_idx]
            
            # 获取概率最高的情感
            emotion_probs = [
                (self.emotions[i], float(probabilities[i])) 
                for i in range(len(probabilities))
            ]
            
            # 按概率降序排序
            emotion_probs.sort(key=lambda x: x[1], reverse=True)
            
            return emotion_probs[:top_k]
            
        except Exception as e:
            logger.error(f"预测下一情感失败: {e}")
            return []
    
    def predict_emotion_sequence(
        self, 
        user_id: str, 
        current_emotion: str,
        sequence_length: int = 5
    ) -> List[List[Tuple[str, float]]]:
        """
        预测情感状态序列
        
        Args:
            user_id: 用户ID
            current_emotion: 当前情感
            sequence_length: 预测序列长度
            
        Returns:
            每步预测结果的列表
        """
        if user_id not in self.transition_matrices:
            return []
        
        sequence_predictions = []
        current = current_emotion
        
        for step in range(sequence_length):
            predictions = self.predict_next_emotion(user_id, current, top_k=3)
            if not predictions:
                break
            
            sequence_predictions.append(predictions)
            # 选择概率最高的作为下一步的当前状态
            current = predictions[0][0]
        
        return sequence_predictions
    
    def get_transition_probability(
        self, 
        user_id: str, 
        from_emotion: str, 
        to_emotion: str
    ) -> float:
        """
        获取特定转换的概率
        
        Args:
            user_id: 用户ID
            from_emotion: 源情感
            to_emotion: 目标情感
            
        Returns:
            转换概率
        """
        if user_id not in self.transition_matrices:
            return 0.0
        
        if (from_emotion not in self.emotion_to_index or 
            to_emotion not in self.emotion_to_index):
            return 0.0
        
        try:
            matrix = self.transition_matrices[user_id]
            from_idx = self.emotion_to_index[from_emotion]
            to_idx = self.emotion_to_index[to_emotion]
            
            return float(matrix[from_idx, to_idx])
            
        except Exception as e:
            logger.error(f"获取转换概率失败: {e}")
            return 0.0
    
    def get_steady_state_distribution(self, user_id: str) -> Dict[str, float]:
        """
        计算稳态分布（长期情感分布）
        
        Args:
            user_id: 用户ID
            
        Returns:
            稳态情感分布
        """
        if user_id not in self.transition_matrices:
            return {}
        
        try:
            matrix = self.transition_matrices[user_id]
            
            # 计算特征向量和特征值
            eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
            
            # 找到特征值为1的特征向量（稳态向量）
            steady_state_idx = np.argmin(np.abs(eigenvalues - 1))
            steady_state = np.real(eigenvectors[:, steady_state_idx])
            
            # 归一化
            steady_state = np.abs(steady_state) / np.sum(np.abs(steady_state))
            
            # 转换为字典
            distribution = {
                self.emotions[i]: float(steady_state[i]) 
                for i in range(len(steady_state))
            }
            
            return distribution
            
        except Exception as e:
            logger.error(f"计算稳态分布失败: {e}")
            return {}
    
    def analyze_transition_patterns(self, user_id: str) -> Dict[str, any]:
        """
        分析用户的情感转换模式
        
        Args:
            user_id: 用户ID
            
        Returns:
            转换模式分析结果
        """
        if user_id not in self.transition_matrices:
            return {}
        
        try:
            matrix = self.transition_matrices[user_id]
            
            analysis = {
                'most_stable_emotions': [],  # 最稳定的情感（自转换概率高）
                'most_volatile_emotions': [],  # 最不稳定的情感
                'common_transitions': [],  # 常见转换
                'rare_transitions': [],  # 罕见转换
                'transition_entropy': 0.0  # 转换熵（衡量转换的随机性）
            }
            
            # 分析每种情感的自转换概率（稳定性）
            stability_scores = []
            for i, emotion in enumerate(self.emotions):
                self_transition_prob = matrix[i, i]
                stability_scores.append((emotion, self_transition_prob))
            
            # 按稳定性排序
            stability_scores.sort(key=lambda x: x[1], reverse=True)
            analysis['most_stable_emotions'] = stability_scores[:5]
            analysis['most_volatile_emotions'] = stability_scores[-5:]
            
            # 找出最常见和最罕见的转换
            all_transitions = []
            for i in range(self.n_emotions):
                for j in range(self.n_emotions):
                    if i != j:  # 排除自转换
                        prob = matrix[i, j]
                        all_transitions.append((
                            self.emotions[i], 
                            self.emotions[j], 
                            prob
                        ))
            
            all_transitions.sort(key=lambda x: x[2], reverse=True)
            analysis['common_transitions'] = all_transitions[:10]
            analysis['rare_transitions'] = all_transitions[-10:]
            
            # 计算转换熵
            entropy = 0.0
            for i in range(self.n_emotions):
                row_probs = matrix[i]
                for prob in row_probs:
                    if prob > 0:
                        entropy -= prob * np.log2(prob)
            analysis['transition_entropy'] = float(entropy)
            
            return analysis
            
        except Exception as e:
            logger.error(f"分析转换模式失败: {e}")
            return {}
    
    def detect_anomalous_transitions(
        self, 
        user_id: str, 
        recent_history: List[EmotionState],
        anomaly_threshold: float = 0.1
    ) -> List[Tuple[str, str, float]]:
        """
        检测异常的情感转换
        
        Args:
            user_id: 用户ID
            recent_history: 最近的情感历史
            anomaly_threshold: 异常阈值（概率低于此值视为异常）
            
        Returns:
            异常转换列表 [(from_emotion, to_emotion, probability)]
        """
        if user_id not in self.transition_matrices or len(recent_history) < 2:
            return []
        
        anomalies = []
        history_sorted = sorted(recent_history, key=lambda x: x.timestamp)
        
        try:
            for i in range(len(history_sorted) - 1):
                from_emotion = history_sorted[i].emotion
                to_emotion = history_sorted[i + 1].emotion
                
                prob = self.get_transition_probability(user_id, from_emotion, to_emotion)
                
                if prob < anomaly_threshold:
                    anomalies.append((from_emotion, to_emotion, prob))
            
            return anomalies
            
        except Exception as e:
            logger.error(f"检测异常转换失败: {e}")
            return []
    
    def get_model_summary(self, user_id: str) -> Dict[str, any]:
        """获取转换模型摘要信息"""
        if user_id not in self.transition_matrices:
            return {}
        
        matrix = self.transition_matrices[user_id]
        
        return {
            'model_size': matrix.shape,
            'total_transitions': float(np.sum(matrix)),
            'non_zero_transitions': int(np.count_nonzero(matrix)),
            'sparsity': float(1.0 - np.count_nonzero(matrix) / matrix.size),
            'emotions_covered': self.emotions
        }