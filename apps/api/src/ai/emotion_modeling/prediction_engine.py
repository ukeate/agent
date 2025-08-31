"""
情感状态动态跟踪和预测引擎

实现实时情感状态跟踪、轨迹预测和聚类分析
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging

from .models import EmotionState, EmotionPrediction, PersonalityProfile
from .transition_model import TransitionModelManager
from .space_mapper import EmotionSpaceMapper
from .personality_builder import PersonalityProfileBuilder

logger = logging.getLogger(__name__)


class EmotionPredictionEngine:
    """情感预测引擎"""
    
    def __init__(self):
        self.transition_manager = TransitionModelManager()
        self.space_mapper = EmotionSpaceMapper()
        self.personality_builder = PersonalityProfileBuilder()
        
        # 实时状态缓存 {user_id: deque of recent states}
        self.user_state_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 预测参数
        self.prediction_config = {
            'max_horizon_hours': 24,
            'min_confidence_threshold': 0.3,
            'context_window_size': 20,
            'anomaly_threshold': 0.1
        }
    
    async def track_emotion_state(
        self, 
        user_id: str, 
        new_state: EmotionState
    ) -> Dict[str, Any]:
        """
        实时跟踪情感状态变化
        
        Args:
            user_id: 用户ID
            new_state: 新的情感状态
            
        Returns:
            跟踪分析结果
        """
        try:
            # 添加到状态缓存
            self.user_state_cache[user_id].append(new_state)
            
            # 获取最近的状态历史
            recent_states = list(self.user_state_cache[user_id])
            
            tracking_result = {
                'state_id': new_state.id,
                'user_id': user_id,
                'timestamp': new_state.timestamp.isoformat(),
                'emotion': new_state.emotion,
                'intensity': new_state.intensity,
                'vad_coordinates': new_state.get_vad_coordinates(),
                'quadrant': self.space_mapper.get_emotion_quadrant(new_state),
                'is_anomaly': False,
                'change_detected': False,
                'trend_analysis': {},
                'similarity_to_baseline': 0.0
            }
            
            # 检测状态变化
            if len(recent_states) >= 2:
                prev_state = recent_states[-2]
                tracking_result['change_detected'] = (
                    new_state.emotion != prev_state.emotion
                )
                
                # 计算状态距离
                distance = self.space_mapper.calculate_emotion_distance(new_state, prev_state)
                tracking_result['change_magnitude'] = float(distance)
            
            # 异常检测
            if len(recent_states) >= 5:
                is_anomaly = await self._detect_anomalous_state(user_id, new_state, recent_states[:-1])
                tracking_result['is_anomaly'] = is_anomaly
            
            # 趋势分析
            if len(recent_states) >= 10:
                trend = self._analyze_recent_trend(recent_states)
                tracking_result['trend_analysis'] = trend
            
            logger.info(f"用户 {user_id} 情感状态已更新: {new_state.emotion}")
            return tracking_result
            
        except Exception as e:
            logger.error(f"跟踪情感状态失败: {e}")
            return {'error': str(e)}
    
    async def predict_emotion_trajectory(
        self, 
        user_id: str, 
        current_state: EmotionState,
        time_horizon: timedelta = timedelta(hours=1),
        personality_profile: Optional[PersonalityProfile] = None
    ) -> EmotionPrediction:
        """
        预测情感轨迹
        
        Args:
            user_id: 用户ID
            current_state: 当前情感状态
            time_horizon: 预测时间范围
            personality_profile: 个性画像（可选）
            
        Returns:
            情感预测结果
        """
        try:
            # 获取历史状态
            recent_states = list(self.user_state_cache[user_id])
            
            if not recent_states:
                recent_states = [current_state]
            
            # 基于转换矩阵的预测
            transition_predictions = self.transition_manager.predict_next_emotion(
                user_id, current_state.emotion, top_k=5
            )
            
            # 基于个性画像的调整
            if personality_profile:
                adjusted_predictions = self._adjust_predictions_by_personality(
                    transition_predictions, personality_profile
                )
            else:
                adjusted_predictions = transition_predictions
            
            # 基于时间模式的调整
            time_adjusted_predictions = self._adjust_predictions_by_time_pattern(
                adjusted_predictions, current_state.timestamp
            )
            
            # 计算预测置信度
            confidence = self._calculate_prediction_confidence(
                user_id, current_state, recent_states
            )
            
            # 分析影响因素
            factors = self._analyze_prediction_factors(
                current_state, recent_states, personality_profile
            )
            
            prediction = EmotionPrediction(
                user_id=user_id,
                current_emotion=current_state.emotion,
                predicted_emotions=time_adjusted_predictions,
                confidence=confidence,
                time_horizon=time_horizon,
                prediction_time=datetime.now(),
                factors=factors
            )
            
            logger.info(f"已生成用户 {user_id} 的情感预测")
            return prediction
            
        except Exception as e:
            logger.error(f"预测情感轨迹失败: {e}")
            return EmotionPrediction(
                user_id=user_id,
                current_emotion=current_state.emotion,
                confidence=0.0,
                time_horizon=time_horizon
            )
    
    async def perform_emotion_clustering(
        self, 
        user_id: str,
        min_samples: int = 20
    ) -> List[Dict[str, Any]]:
        """
        对用户情感状态进行聚类分析
        
        Args:
            user_id: 用户ID
            min_samples: 最小样本数
            
        Returns:
            聚类分析结果
        """
        try:
            recent_states = list(self.user_state_cache[user_id])
            
            if len(recent_states) < min_samples:
                logger.warning(f"用户 {user_id} 数据不足，无法进行聚类分析")
                return []
            
            # 使用空间映射器进行聚类
            clusters = self.space_mapper.detect_emotion_clusters(
                recent_states,
                n_clusters=min(5, len(recent_states) // 5),
                min_cluster_size=3
            )
            
            # 增强聚类结果
            enhanced_clusters = []
            for cluster in clusters:
                enhanced_cluster = cluster.copy()
                
                # 分析聚类的时间模式
                cluster_states = cluster['states']
                time_pattern = self._analyze_cluster_time_pattern(cluster_states)
                enhanced_cluster['temporal_pattern'] = time_pattern
                
                # 分析聚类的触发因素
                triggers = self._analyze_cluster_triggers(cluster_states)
                enhanced_cluster['common_triggers'] = triggers
                
                # 计算聚类稳定性
                stability = self._calculate_cluster_stability(cluster_states)
                enhanced_cluster['stability_score'] = stability
                
                # 移除原始状态数据（减少输出大小）
                enhanced_cluster.pop('states', None)
                
                enhanced_clusters.append(enhanced_cluster)
            
            logger.info(f"用户 {user_id} 聚类分析完成，发现 {len(enhanced_clusters)} 个聚类")
            return enhanced_clusters
            
        except Exception as e:
            logger.error(f"情感聚类分析失败: {e}")
            return []
    
    async def _detect_anomalous_state(
        self, 
        user_id: str, 
        new_state: EmotionState, 
        history: List[EmotionState]
    ) -> bool:
        """检测异常情感状态"""
        try:
            if not history:
                return False
            
            # 计算与历史状态的平均距离
            distances = [
                self.space_mapper.calculate_emotion_distance(new_state, hist_state)
                for hist_state in history[-10:]  # 使用最近10个状态
            ]
            
            avg_distance = np.mean(distances)
            std_distance = np.std(distances)
            
            # 使用3-sigma规则检测异常
            threshold = avg_distance + 2 * std_distance
            current_distance = np.mean(distances[-3:]) if len(distances) >= 3 else avg_distance
            
            return current_distance > threshold
            
        except Exception as e:
            logger.error(f"异常检测失败: {e}")
            return False
    
    def _analyze_recent_trend(self, states: List[EmotionState]) -> Dict[str, Any]:
        """分析最近的情感趋势"""
        if len(states) < 5:
            return {}
        
        # 提取VAD时间序列
        vad_series = [self.space_mapper.map_state_to_space(state) for state in states[-10:]]
        vad_array = np.array(vad_series)
        
        trend_analysis = {
            'valence_trend': 'stable',
            'arousal_trend': 'stable',
            'dominance_trend': 'stable',
            'overall_direction': 'stable'
        }
        
        try:
            # 计算线性趋势
            x = np.arange(len(vad_array))
            
            for i, dimension in enumerate(['valence', 'arousal', 'dominance']):
                y = vad_array[:, i]
                slope = np.polyfit(x, y, 1)[0]
                
                if slope > 0.05:
                    trend_analysis[f'{dimension}_trend'] = 'increasing'
                elif slope < -0.05:
                    trend_analysis[f'{dimension}_trend'] = 'decreasing'
            
            # 综合趋势判断
            valence_slope = np.polyfit(x, vad_array[:, 0], 1)[0]
            if valence_slope > 0.05:
                trend_analysis['overall_direction'] = 'improving'
            elif valence_slope < -0.05:
                trend_analysis['overall_direction'] = 'declining'
                
        except Exception as e:
            logger.error(f"趋势分析失败: {e}")
        
        return trend_analysis
    
    def _adjust_predictions_by_personality(
        self, 
        predictions: List[Tuple[str, float]], 
        personality: PersonalityProfile
    ) -> List[Tuple[str, float]]:
        """根据个性画像调整预测"""
        adjusted_predictions = []
        
        for emotion, probability in predictions:
            adjusted_prob = probability
            
            # 根据个性特质调整
            if emotion in personality.dominant_emotions:
                adjusted_prob *= 1.2  # 增加主导情感的概率
            
            # 基于情感波动性调整
            if personality.emotion_volatility > 0.7:
                # 高波动性用户更容易变化
                if emotion != predictions[0][0]:  # 非最可能情感
                    adjusted_prob *= 1.1
            
            adjusted_predictions.append((emotion, min(1.0, adjusted_prob)))
        
        # 重新归一化
        total_prob = sum(prob for _, prob in adjusted_predictions)
        if total_prob > 0:
            adjusted_predictions = [
                (emotion, prob / total_prob) 
                for emotion, prob in adjusted_predictions
            ]
        
        return adjusted_predictions
    
    def _adjust_predictions_by_time_pattern(
        self, 
        predictions: List[Tuple[str, float]], 
        current_time: datetime
    ) -> List[Tuple[str, float]]:
        """根据时间模式调整预测"""
        # 简单的时间模式调整
        hour = current_time.hour
        
        # 晚上时间增加平静情感的概率
        if 22 <= hour or hour <= 6:
            adjusted_predictions = []
            for emotion, prob in predictions:
                if emotion in ['neutral', 'sadness', 'depression']:
                    prob *= 1.1
                adjusted_predictions.append((emotion, prob))
            predictions = adjusted_predictions
        
        # 早上时间增加积极情感的概率  
        elif 6 <= hour <= 10:
            adjusted_predictions = []
            for emotion, prob in predictions:
                if emotion in ['happiness', 'joy', 'anticipation']:
                    prob *= 1.1
                adjusted_predictions.append((emotion, prob))
            predictions = adjusted_predictions
        
        return predictions
    
    def _calculate_prediction_confidence(
        self, 
        user_id: str, 
        current_state: EmotionState,
        history: List[EmotionState]
    ) -> float:
        """计算预测置信度"""
        confidence_factors = []
        
        # 基于历史数据量
        data_confidence = min(1.0, len(history) / 50.0)
        confidence_factors.append(data_confidence)
        
        # 基于转换模型质量
        if user_id in self.transition_manager.transition_matrices:
            model_summary = self.transition_manager.get_model_summary(user_id)
            model_confidence = 1.0 - model_summary.get('sparsity', 0.5)
            confidence_factors.append(model_confidence)
        
        # 基于当前状态置信度
        confidence_factors.append(current_state.confidence)
        
        # 基于最近趋势的稳定性
        if len(history) >= 5:
            recent_emotions = [s.emotion for s in history[-5:]]
            stability = 1.0 - (len(set(recent_emotions)) / len(recent_emotions))
            confidence_factors.append(stability)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _analyze_prediction_factors(
        self, 
        current_state: EmotionState,
        history: List[EmotionState],
        personality: Optional[PersonalityProfile]
    ) -> Dict[str, Any]:
        """分析影响预测的因素"""
        factors = {
            'current_intensity': current_state.intensity,
            'current_confidence': current_state.confidence,
            'recent_volatility': 0.0,
            'personality_influence': {},
            'temporal_context': {
                'hour_of_day': current_state.timestamp.hour,
                'day_of_week': current_state.timestamp.weekday()
            }
        }
        
        # 计算最近波动性
        if len(history) >= 5:
            recent_intensities = [s.intensity for s in history[-5:]]
            factors['recent_volatility'] = float(np.std(recent_intensities))
        
        # 个性影响因素
        if personality:
            factors['personality_influence'] = {
                'volatility': personality.emotion_volatility,
                'recovery_rate': personality.recovery_rate,
                'dominant_emotions': personality.dominant_emotions[:2]
            }
        
        return factors
    
    def _analyze_cluster_time_pattern(self, cluster_states: List[EmotionState]) -> Dict[str, Any]:
        """分析聚类的时间模式"""
        if not cluster_states:
            return {}
        
        hours = [state.timestamp.hour for state in cluster_states]
        weekdays = [state.timestamp.weekday() for state in cluster_states]
        
        # 最常见的小时和星期
        hour_counter = Counter(hours)
        weekday_counter = Counter(weekdays)
        
        return {
            'most_common_hours': hour_counter.most_common(3),
            'most_common_weekdays': weekday_counter.most_common(3),
            'time_span': {
                'earliest': min(state.timestamp for state in cluster_states).isoformat(),
                'latest': max(state.timestamp for state in cluster_states).isoformat()
            }
        }
    
    def _analyze_cluster_triggers(self, cluster_states: List[EmotionState]) -> List[str]:
        """分析聚类的常见触发因素"""
        all_triggers = []
        for state in cluster_states:
            all_triggers.extend(state.triggers)
        
        if not all_triggers:
            return []
        
        trigger_counter = Counter(all_triggers)
        return [trigger for trigger, _ in trigger_counter.most_common(5)]
    
    def _calculate_cluster_stability(self, cluster_states: List[EmotionState]) -> float:
        """计算聚类稳定性"""
        if len(cluster_states) < 2:
            return 1.0
        
        # 计算状态间的平均距离
        distances = []
        for i in range(len(cluster_states)):
            for j in range(i + 1, len(cluster_states)):
                distance = self.space_mapper.calculate_emotion_distance(
                    cluster_states[i], cluster_states[j]
                )
                distances.append(distance)
        
        if not distances:
            return 1.0
        
        # 距离越小，稳定性越高
        avg_distance = np.mean(distances)
        stability = max(0.0, 1.0 - avg_distance)
        
        return stability
    
    def get_user_state_summary(self, user_id: str) -> Dict[str, Any]:
        """获取用户状态摘要"""
        states = list(self.user_state_cache[user_id])
        
        if not states:
            return {'error': '没有状态数据'}
        
        recent_emotions = [s.emotion for s in states[-10:]]
        
        return {
            'user_id': user_id,
            'total_states': len(states),
            'latest_emotion': states[-1].emotion,
            'latest_timestamp': states[-1].timestamp.isoformat(),
            'recent_emotions': recent_emotions,
            'cache_size': len(states),
            'time_span': {
                'start': states[0].timestamp.isoformat(),
                'end': states[-1].timestamp.isoformat()
            }
        }