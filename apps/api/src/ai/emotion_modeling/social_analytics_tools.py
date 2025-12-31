"""
情感社交分析工具 - Story 11.6 Task 6
提供对话情感流分析、社交网络情感地图和社交情感统计
"""

from src.core.utils.timezone_utils import utc_now
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from datetime import datetime, timedelta
import json
import numpy as np
from collections import defaultdict, Counter
from .models import EmotionVector, SocialContext
from .core_interfaces import EmotionModelingInterface
from .group_emotion_analyzer import GroupEmotionAnalyzer
from .relationship_analyzer import RelationshipDynamicsAnalyzer

from src.core.logging import get_logger
logger = get_logger(__name__)

class AnalysisType(Enum):
    """分析类型"""
    EMOTION_FLOW = "emotion_flow"
    NETWORK_MAP = "network_map"
    STATISTICS = "statistics"
    TREND_ANALYSIS = "trend_analysis"

@dataclass
class EmotionFlowPoint:
    """情感流时间点"""
    timestamp: datetime
    user_id: str
    emotion: str
    intensity: float
    valence: float
    arousal: float
    context: Optional[Dict[str, Any]] = None

@dataclass
class EmotionFlow:
    """对话情感流"""
    session_id: str
    participants: List[str]
    start_time: datetime
    end_time: datetime
    flow_points: List[EmotionFlowPoint]
    emotional_peaks: List[Dict[str, Any]]
    emotional_valleys: List[Dict[str, Any]]
    turning_points: List[Dict[str, Any]]
    overall_trend: str
    dominant_emotions: Dict[str, float]

@dataclass
class NetworkNode:
    """网络节点"""
    user_id: str
    emotional_influence: float  # 情感影响力 0.0-1.0
    connection_strength: Dict[str, float]  # 与其他节点的连接强度
    emotion_state: EmotionVector
    role: str  # 网络中的角色：influencer, supporter, neutral, etc.

@dataclass
class EmotionNetwork:
    """情感网络"""
    network_id: str
    nodes: Dict[str, NetworkNode]
    edges: Dict[Tuple[str, str], float]  # 连接边和权重
    clusters: List[List[str]]  # 情感聚类
    central_nodes: List[str]  # 中心节点
    influence_paths: Dict[str, List[str]]  # 影响传播路径
    network_cohesion: float  # 网络凝聚力
    polarization_level: float  # 极化程度

@dataclass
class SocialEmotionStats:
    """社交情感统计"""
    user_id: str
    analysis_period: Tuple[datetime, datetime]
    total_interactions: int
    emotional_diversity: float  # 情感多样性
    average_valence: float  # 平均效价
    average_arousal: float  # 平均激活度
    emotional_stability: float  # 情感稳定性
    social_adaptability: float  # 社会适应性
    influence_score: float  # 影响力分数
    support_given: int  # 提供支持次数
    support_received: int  # 接受支持次数
    conflict_involvement: int  # 冲突参与次数
    scenario_performance: Dict[str, float]  # 不同场景下的表现

class SocialAnalyticsTools(EmotionModelingInterface):
    """社交情感分析工具"""
    
    def __init__(self):
        self.group_analyzer = GroupEmotionAnalyzer()
        self.relationship_analyzer = RelationshipDynamicsAnalyzer()
        self.analysis_cache: Dict[str, Any] = {}
        self.flow_history: Dict[str, EmotionFlow] = {}
        self.network_history: Dict[str, EmotionNetwork] = {}
        
    async def analyze_emotion_flow(
        self,
        session_id: str,
        conversation_data: List[Dict[str, Any]],
        time_window: int = 300  # 秒
    ) -> EmotionFlow:
        """分析对话情感流"""
        try:
            # 提取参与者
            participants = list(set(
                msg.get('user_id') for msg in conversation_data 
                if msg.get('user_id')
            ))
            
            if len(conversation_data) < 2:
                logger.warning("Insufficient conversation data for flow analysis")
                return self._create_empty_flow(session_id, participants)
            
            # 构建情感流点
            flow_points = []
            for msg in conversation_data:
                if not msg.get('emotion_data'):
                    continue
                    
                emotion_data = msg['emotion_data']
                flow_point = EmotionFlowPoint(
                    timestamp=datetime.fromisoformat(msg['timestamp']),
                    user_id=msg['user_id'],
                    emotion=emotion_data.get('dominant_emotion', 'neutral'),
                    intensity=emotion_data.get('intensity', 0.5),
                    valence=emotion_data.get('valence', 0.0),
                    arousal=emotion_data.get('arousal', 0.0),
                    context=msg.get('context')
                )
                flow_points.append(flow_point)
            
            # 排序时间点
            flow_points.sort(key=lambda x: x.timestamp)
            
            # 识别情感高峰和低谷
            emotional_peaks = self._identify_emotional_peaks(flow_points)
            emotional_valleys = self._identify_emotional_valleys(flow_points)
            
            # 识别转折点
            turning_points = self._identify_turning_points(flow_points)
            
            # 计算整体趋势
            overall_trend = self._calculate_overall_trend(flow_points)
            
            # 统计主导情感
            dominant_emotions = self._calculate_dominant_emotions(flow_points)
            
            emotion_flow = EmotionFlow(
                session_id=session_id,
                participants=participants,
                start_time=flow_points[0].timestamp if flow_points else utc_now(),
                end_time=flow_points[-1].timestamp if flow_points else utc_now(),
                flow_points=flow_points,
                emotional_peaks=emotional_peaks,
                emotional_valleys=emotional_valleys,
                turning_points=turning_points,
                overall_trend=overall_trend,
                dominant_emotions=dominant_emotions
            )
            
            # 缓存结果
            self.flow_history[session_id] = emotion_flow
            
            return emotion_flow
            
        except Exception as e:
            logger.error(f"Emotion flow analysis failed: {e}")
            return self._create_empty_flow(session_id, [])
    
    def _identify_emotional_peaks(
        self, 
        flow_points: List[EmotionFlowPoint]
    ) -> List[Dict[str, Any]]:
        """识别情感高潮时刻"""
        peaks = []
        
        if len(flow_points) < 3:
            return peaks
        
        # 使用滑动窗口识别局部最大值
        window_size = 5
        for i in range(window_size, len(flow_points) - window_size):
            current_point = flow_points[i]
            
            # 检查是否为局部最大值
            is_peak = True
            for j in range(i - window_size // 2, i + window_size // 2 + 1):
                if j != i and flow_points[j].intensity > current_point.intensity:
                    is_peak = False
                    break
            
            if is_peak and current_point.intensity > 0.7:  # 强度阈值
                peaks.append({
                    'timestamp': current_point.timestamp,
                    'user_id': current_point.user_id,
                    'emotion': current_point.emotion,
                    'intensity': current_point.intensity,
                    'context': current_point.context,
                    'type': 'peak'
                })
        
        return peaks
    
    def _identify_emotional_valleys(
        self, 
        flow_points: List[EmotionFlowPoint]
    ) -> List[Dict[str, Any]]:
        """识别情感低谷时刻"""
        valleys = []
        
        if len(flow_points) < 3:
            return valleys
        
        window_size = 5
        for i in range(window_size, len(flow_points) - window_size):
            current_point = flow_points[i]
            
            # 检查是否为局部最小值
            is_valley = True
            for j in range(i - window_size // 2, i + window_size // 2 + 1):
                if j != i and flow_points[j].intensity < current_point.intensity:
                    is_valley = False
                    break
            
            if is_valley and current_point.intensity < 0.3:  # 强度阈值
                valleys.append({
                    'timestamp': current_point.timestamp,
                    'user_id': current_point.user_id,
                    'emotion': current_point.emotion,
                    'intensity': current_point.intensity,
                    'context': current_point.context,
                    'type': 'valley'
                })
        
        return valleys
    
    def _identify_turning_points(
        self, 
        flow_points: List[EmotionFlowPoint]
    ) -> List[Dict[str, Any]]:
        """识别情感转折点"""
        turning_points = []
        
        if len(flow_points) < 5:
            return turning_points
        
        # 计算情感变化率
        for i in range(2, len(flow_points) - 2):
            prev_trend = self._calculate_trend(flow_points[i-2:i+1])
            next_trend = self._calculate_trend(flow_points[i:i+3])
            
            # 检测趋势变化
            if abs(prev_trend - next_trend) > 0.5:  # 变化阈值
                turning_points.append({
                    'timestamp': flow_points[i].timestamp,
                    'user_id': flow_points[i].user_id,
                    'emotion': flow_points[i].emotion,
                    'intensity': flow_points[i].intensity,
                    'trend_change': next_trend - prev_trend,
                    'influence_factor': self._identify_influence_factor(flow_points, i),
                    'type': 'turning_point'
                })
        
        return turning_points
    
    def _calculate_trend(self, points: List[EmotionFlowPoint]) -> float:
        """计算趋势斜率"""
        if len(points) < 2:
            return 0.0
        
        x = list(range(len(points)))
        y = [p.intensity for p in points]
        
        # 简单线性回归
        n = len(points)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_xx = sum(x[i] * x[i] for i in range(n))
        
        if n * sum_xx - sum_x * sum_x == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        return slope
    
    def _identify_influence_factor(
        self, 
        flow_points: List[EmotionFlowPoint], 
        index: int
    ) -> str:
        """识别影响因素"""
        if index < 1 or index >= len(flow_points):
            return "unknown"
        
        current_point = flow_points[index]
        prev_point = flow_points[index - 1]
        
        # 检查是否有新参与者加入
        if current_point.user_id != prev_point.user_id:
            return "participant_change"
        
        # 检查情感类型变化
        if current_point.emotion != prev_point.emotion:
            return "emotion_shift"
        
        # 检查上下文变化
        if (current_point.context and prev_point.context and 
            current_point.context.get('topic') != prev_point.context.get('topic')):
            return "topic_change"
        
        return "internal_dynamic"
    
    def _calculate_overall_trend(self, flow_points: List[EmotionFlowPoint]) -> str:
        """计算整体趋势"""
        if not flow_points:
            return "stable"
        
        # 计算整体斜率
        overall_slope = self._calculate_trend(flow_points)
        
        if overall_slope > 0.1:
            return "improving"
        elif overall_slope < -0.1:
            return "declining"
        else:
            return "stable"
    
    def _calculate_dominant_emotions(
        self, 
        flow_points: List[EmotionFlowPoint]
    ) -> Dict[str, float]:
        """计算主导情感分布"""
        if not flow_points:
            return {}
        
        emotion_counter = Counter()
        total_intensity = 0.0
        
        for point in flow_points:
            emotion_counter[point.emotion] += point.intensity
            total_intensity += point.intensity
        
        if total_intensity == 0:
            return {}
        
        # 归一化
        return {
            emotion: count / total_intensity
            for emotion, count in emotion_counter.items()
        }
    
    async def build_emotion_network(
        self,
        session_data: List[Dict[str, Any]],
        interaction_history: List[Dict[str, Any]]
    ) -> EmotionNetwork:
        """构建社交网络情感地图"""
        try:
            network_id = f"network_{utc_now().isoformat()}"
            
            # 提取参与者
            participants = list(set(
                msg.get('user_id') for msg in session_data 
                if msg.get('user_id')
            ))
            
            if len(participants) < 2:
                return self._create_empty_network(network_id)
            
            # 构建网络节点
            nodes = {}
            for user_id in participants:
                # 计算情感影响力
                influence = await self._calculate_emotional_influence(
                    user_id, session_data
                )
                
                # 计算连接强度
                connections = await self._calculate_connection_strengths(
                    user_id, participants, interaction_history
                )
                
                # 获取当前情感状态
                emotion_state = self._get_current_emotion_state(user_id, session_data)
                
                # 确定网络角色
                role = self._determine_network_role(user_id, influence, connections)
                
                nodes[user_id] = NetworkNode(
                    user_id=user_id,
                    emotional_influence=influence,
                    connection_strength=connections,
                    emotion_state=emotion_state,
                    role=role
                )
            
            # 构建连接边
            edges = {}
            for user1 in participants:
                for user2 in participants:
                    if user1 != user2:
                        strength = nodes[user1].connection_strength.get(user2, 0.0)
                        if strength > 0.1:  # 最小连接阈值
                            edges[(user1, user2)] = strength
            
            # 识别情感聚类
            clusters = self._identify_emotion_clusters(nodes, edges)
            
            # 识别中心节点
            central_nodes = self._identify_central_nodes(nodes, edges)
            
            # 分析影响传播路径
            influence_paths = self._analyze_influence_paths(nodes, edges)
            
            # 计算网络指标
            network_cohesion = self._calculate_network_cohesion(nodes, edges)
            polarization_level = self._calculate_polarization_level(nodes)
            
            emotion_network = EmotionNetwork(
                network_id=network_id,
                nodes=nodes,
                edges=edges,
                clusters=clusters,
                central_nodes=central_nodes,
                influence_paths=influence_paths,
                network_cohesion=network_cohesion,
                polarization_level=polarization_level
            )
            
            # 缓存结果
            self.network_history[network_id] = emotion_network
            
            return emotion_network
            
        except Exception as e:
            logger.error(f"Network building failed: {e}")
            return self._create_empty_network("error_network")
    
    async def _calculate_emotional_influence(
        self, 
        user_id: str, 
        session_data: List[Dict[str, Any]]
    ) -> float:
        """计算用户的情感影响力"""
        user_messages = [msg for msg in session_data if msg.get('user_id') == user_id]
        
        if not user_messages:
            return 0.0
        
        # 消息数量权重
        message_count_weight = min(len(user_messages) / 10.0, 1.0)
        
        # 情感强度权重
        avg_intensity = np.mean([
            msg.get('emotion_data', {}).get('intensity', 0.0)
            for msg in user_messages
        ])
        
        # 响应触发权重（其他人对该用户消息的响应数量）
        response_trigger_weight = 0.5  # 简化计算，实际应该分析响应关系
        
        influence = (message_count_weight * 0.4 + 
                    avg_intensity * 0.4 + 
                    response_trigger_weight * 0.2)
        
        return min(influence, 1.0)
    
    async def _calculate_connection_strengths(
        self,
        user_id: str,
        all_participants: List[str],
        interaction_history: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """计算与其他参与者的连接强度"""
        connections = {}
        
        for other_user in all_participants:
            if other_user == user_id:
                continue
            
            # 使用关系分析器计算连接强度
            try:
                relationship_data = await self.relationship_analyzer.analyze_relationship_dynamics(
                    user_id, other_user, interaction_history
                )
                
                # 基于关系健康度和亲密度计算连接强度
                connection_strength = (
                    relationship_data.relationship_health * 0.6 +
                    relationship_data.intimacy_level * 0.4
                )
                
                connections[other_user] = connection_strength
                
            except Exception as e:
                logger.warning(f"Failed to calculate connection strength between {user_id} and {other_user}: {e}")
                connections[other_user] = 0.2  # 默认弱连接
        
        return connections
    
    def _get_current_emotion_state(
        self, 
        user_id: str, 
        session_data: List[Dict[str, Any]]
    ) -> EmotionVector:
        """获取用户当前情感状态"""
        user_messages = [msg for msg in session_data if msg.get('user_id') == user_id]
        
        if not user_messages:
            return EmotionVector(
                emotions={'neutral': 1.0},
                intensity=0.5,
                confidence=0.5,
                context={}
            )
        
        # 获取最近的情感数据
        latest_msg = max(user_messages, key=lambda x: x.get('timestamp', ''))
        emotion_data = latest_msg.get('emotion_data', {})
        
        return EmotionVector(
            emotions=emotion_data.get('emotions', {'neutral': 1.0}),
            intensity=emotion_data.get('intensity', 0.5),
            confidence=emotion_data.get('confidence', 0.5),
            context=latest_msg.get('context', {})
        )
    
    def _determine_network_role(
        self,
        user_id: str,
        influence: float,
        connections: Dict[str, float]
    ) -> str:
        """确定用户在网络中的角色"""
        avg_connection = np.mean(list(connections.values())) if connections else 0.0
        
        if influence > 0.7 and avg_connection > 0.6:
            return "influencer"
        elif influence < 0.3 and avg_connection > 0.7:
            return "supporter"
        elif influence > 0.5 and avg_connection < 0.4:
            return "independent"
        elif avg_connection > 0.8:
            return "connector"
        else:
            return "neutral"
    
    def _identify_emotion_clusters(
        self,
        nodes: Dict[str, NetworkNode],
        edges: Dict[Tuple[str, str], float]
    ) -> List[List[str]]:
        """识别情感聚类"""
        # 简化的聚类算法，基于情感相似性
        clusters = []
        visited = set()
        
        for user_id, node in nodes.items():
            if user_id in visited:
                continue
            
            # 创建新聚类
            cluster = [user_id]
            visited.add(user_id)
            
            # 寻找相似的情感状态
            for other_id, other_node in nodes.items():
                if other_id in visited:
                    continue
                
                # 计算情感相似性
                similarity = self._calculate_emotion_similarity(
                    node.emotion_state, other_node.emotion_state
                )
                
                # 检查连接强度
                connection_strength = edges.get((user_id, other_id), 0.0)
                
                if similarity > 0.6 and connection_strength > 0.3:
                    cluster.append(other_id)
                    visited.add(other_id)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _calculate_emotion_similarity(
        self, 
        emotion1: EmotionVector, 
        emotion2: EmotionVector
    ) -> float:
        """计算情感相似性"""
        # 基于余弦相似性
        emotions1 = emotion1.emotions
        emotions2 = emotion2.emotions
        
        common_emotions = set(emotions1.keys()) & set(emotions2.keys())
        
        if not common_emotions:
            return 0.0
        
        dot_product = sum(emotions1[e] * emotions2[e] for e in common_emotions)
        norm1 = np.sqrt(sum(v**2 for v in emotions1.values()))
        norm2 = np.sqrt(sum(v**2 for v in emotions2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _identify_central_nodes(
        self,
        nodes: Dict[str, NetworkNode],
        edges: Dict[Tuple[str, str], float]
    ) -> List[str]:
        """识别中心节点"""
        centrality_scores = {}
        
        for user_id in nodes.keys():
            # 计算度中心性（连接数量）
            degree = len([edge for edge in edges.keys() if edge[0] == user_id or edge[1] == user_id])
            
            # 计算强度中心性（连接权重和）
            strength = sum(
                weight for edge, weight in edges.items()
                if edge[0] == user_id or edge[1] == user_id
            )
            
            # 结合影响力
            influence = nodes[user_id].emotional_influence
            
            centrality_scores[user_id] = degree * 0.3 + strength * 0.4 + influence * 0.3
        
        # 返回top 20%的中心节点
        sorted_nodes = sorted(centrality_scores.items(), key=lambda x: x[1], reverse=True)
        top_count = max(1, len(sorted_nodes) // 5)
        
        return [node_id for node_id, _ in sorted_nodes[:top_count]]
    
    def _analyze_influence_paths(
        self,
        nodes: Dict[str, NetworkNode],
        edges: Dict[Tuple[str, str], float]
    ) -> Dict[str, List[str]]:
        """分析影响传播路径"""
        influence_paths = {}
        
        # 找到高影响力节点
        high_influence_nodes = [
            user_id for user_id, node in nodes.items()
            if node.emotional_influence > 0.6
        ]
        
        for influencer in high_influence_nodes:
            paths = []
            
            # 简化的路径查找：直接连接和二级连接
            direct_connections = [
                target for (source, target), weight in edges.items()
                if source == influencer and weight > 0.4
            ]
            
            for direct_target in direct_connections:
                # 查找二级连接
                indirect_connections = [
                    final_target for (source, final_target), weight in edges.items()
                    if source == direct_target and weight > 0.3 and final_target != influencer
                ]
                
                for indirect_target in indirect_connections:
                    paths.append([influencer, direct_target, indirect_target])
            
            if paths:
                influence_paths[influencer] = paths
        
        return influence_paths
    
    def _calculate_network_cohesion(
        self,
        nodes: Dict[str, NetworkNode],
        edges: Dict[Tuple[str, str], float]
    ) -> float:
        """计算网络凝聚力"""
        if len(nodes) < 2:
            return 1.0
        
        # 计算平均连接强度
        if not edges:
            return 0.0
        
        avg_edge_weight = np.mean(list(edges.values()))
        
        # 计算连接密度
        possible_edges = len(nodes) * (len(nodes) - 1)
        actual_edges = len(edges)
        connection_density = actual_edges / possible_edges if possible_edges > 0 else 0
        
        # 综合计算凝聚力
        cohesion = avg_edge_weight * 0.6 + connection_density * 0.4
        
        return min(cohesion, 1.0)
    
    def _calculate_polarization_level(self, nodes: Dict[str, NetworkNode]) -> float:
        """计算网络极化程度"""
        if len(nodes) < 2:
            return 0.0
        
        # 计算情感状态的方差
        emotions_data = []
        for node in nodes.values():
            dominant_emotion = max(node.emotion_state.emotions.items(), key=lambda x: x[1])
            emotions_data.append(dominant_emotion[1])  # 使用情感强度
        
        emotion_variance = np.var(emotions_data)
        
        # 归一化到0-1范围
        polarization = min(emotion_variance * 2, 1.0)
        
        return polarization
    
    async def generate_social_emotion_stats(
        self,
        user_id: str,
        analysis_period: Tuple[datetime, datetime],
        interaction_data: List[Dict[str, Any]]
    ) -> SocialEmotionStats:
        """生成社交情感统计"""
        try:
            # 筛选时间范围内的数据
            period_data = [
                data for data in interaction_data
                if (analysis_period[0] <= 
                    datetime.fromisoformat(data.get('timestamp', '1970-01-01')) <= 
                    analysis_period[1])
                and data.get('user_id') == user_id
            ]
            
            total_interactions = len(period_data)
            
            if total_interactions == 0:
                return self._create_empty_stats(user_id, analysis_period)
            
            # 计算情感多样性
            emotions_used = set()
            for data in period_data:
                emotion_data = data.get('emotion_data', {})
                dominant_emotion = max(emotion_data.get('emotions', {'neutral': 1.0}).items(), 
                                     key=lambda x: x[1])[0]
                emotions_used.add(dominant_emotion)
            
            emotional_diversity = len(emotions_used) / 10.0  # 假设有10种基础情感
            
            # 计算平均效价和激活度
            valences = [data.get('emotion_data', {}).get('valence', 0.0) for data in period_data]
            arousals = [data.get('emotion_data', {}).get('arousal', 0.0) for data in period_data]
            
            average_valence = np.mean(valences) if valences else 0.0
            average_arousal = np.mean(arousals) if arousals else 0.0
            
            # 计算情感稳定性（方差的倒数）
            valence_stability = 1.0 / (np.var(valences) + 0.1) if valences else 0.5
            arousal_stability = 1.0 / (np.var(arousals) + 0.1) if arousals else 0.5
            emotional_stability = (valence_stability + arousal_stability) / 2
            emotional_stability = min(emotional_stability, 1.0)
            
            # 计算社会适应性（不同场景下的表现一致性）
            social_adaptability = await self._calculate_social_adaptability(
                user_id, period_data
            )
            
            # 计算影响力分数
            influence_score = await self._calculate_user_influence_score(
                user_id, period_data
            )
            
            # 统计支持行为
            support_given = len([
                data for data in period_data
                if data.get('behavior_type') == 'support_given'
            ])
            
            support_received = len([
                data for data in period_data
                if data.get('behavior_type') == 'support_received'
            ])
            
            # 统计冲突参与
            conflict_involvement = len([
                data for data in period_data
                if data.get('behavior_type') == 'conflict_participation'
            ])
            
            # 计算场景表现
            scenario_performance = await self._calculate_scenario_performance(
                user_id, period_data
            )
            
            return SocialEmotionStats(
                user_id=user_id,
                analysis_period=analysis_period,
                total_interactions=total_interactions,
                emotional_diversity=emotional_diversity,
                average_valence=average_valence,
                average_arousal=average_arousal,
                emotional_stability=emotional_stability,
                social_adaptability=social_adaptability,
                influence_score=influence_score,
                support_given=support_given,
                support_received=support_received,
                conflict_involvement=conflict_involvement,
                scenario_performance=scenario_performance
            )
            
        except Exception as e:
            logger.error(f"Stats generation failed for user {user_id}: {e}")
            return self._create_empty_stats(user_id, analysis_period)
    
    async def _calculate_social_adaptability(
        self,
        user_id: str,
        period_data: List[Dict[str, Any]]
    ) -> float:
        """计算社会适应性"""
        # 按场景分组数据
        scenario_groups = defaultdict(list)
        for data in period_data:
            scenario = data.get('context', {}).get('scenario', 'unknown')
            scenario_groups[scenario].append(data)
        
        if len(scenario_groups) < 2:
            return 0.5  # 场景数据不足
        
        # 计算不同场景下的表现一致性
        scenario_scores = {}
        for scenario, scenario_data in scenario_groups.items():
            # 计算该场景下的平均表现
            avg_valence = np.mean([
                d.get('emotion_data', {}).get('valence', 0.0)
                for d in scenario_data
            ])
            scenario_scores[scenario] = abs(avg_valence)  # 使用绝对值作为表现指标
        
        # 计算表现的一致性（方差的倒数）
        if len(scenario_scores.values()) > 1:
            consistency = 1.0 / (np.var(list(scenario_scores.values())) + 0.1)
            return min(consistency, 1.0)
        else:
            return 0.5
    
    async def _calculate_user_influence_score(
        self,
        user_id: str,
        period_data: List[Dict[str, Any]]
    ) -> float:
        """计算用户影响力分数"""
        # 简化计算，基于消息频率和情感强度
        message_count = len(period_data)
        avg_intensity = np.mean([
            d.get('emotion_data', {}).get('intensity', 0.0)
            for d in period_data
        ]) if period_data else 0.0
        
        # 归一化影响力分数
        frequency_score = min(message_count / 50.0, 1.0)  # 假设50条消息为满分
        intensity_score = avg_intensity
        
        return (frequency_score + intensity_score) / 2
    
    async def _calculate_scenario_performance(
        self,
        user_id: str,
        period_data: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """计算不同场景下的表现"""
        scenario_performance = {}
        
        # 按场景分组
        scenario_groups = defaultdict(list)
        for data in period_data:
            scenario = data.get('context', {}).get('scenario', 'general')
            scenario_groups[scenario].append(data)
        
        for scenario, scenario_data in scenario_groups.items():
            # 计算该场景的表现分数
            valences = [d.get('emotion_data', {}).get('valence', 0.0) for d in scenario_data]
            confidences = [d.get('emotion_data', {}).get('confidence', 0.5) for d in scenario_data]
            
            if valences:
                # 表现分数 = 平均效价（正向性） + 置信度
                performance = (np.mean(valences) + 1) / 2 + np.mean(confidences) * 0.5
                scenario_performance[scenario] = min(performance, 1.0)
        
        return scenario_performance
    
    def _create_empty_flow(self, session_id: str, participants: List[str]) -> EmotionFlow:
        """创建空的情感流"""
        return EmotionFlow(
            session_id=session_id,
            participants=participants,
            start_time=utc_now(),
            end_time=utc_now(),
            flow_points=[],
            emotional_peaks=[],
            emotional_valleys=[],
            turning_points=[],
            overall_trend="stable",
            dominant_emotions={}
        )
    
    def _create_empty_network(self, network_id: str) -> EmotionNetwork:
        """创建空的情感网络"""
        return EmotionNetwork(
            network_id=network_id,
            nodes={},
            edges={},
            clusters=[],
            central_nodes=[],
            influence_paths={},
            network_cohesion=0.0,
            polarization_level=0.0
        )
    
    def _create_empty_stats(
        self, 
        user_id: str, 
        analysis_period: Tuple[datetime, datetime]
    ) -> SocialEmotionStats:
        """创建空的统计数据"""
        return SocialEmotionStats(
            user_id=user_id,
            analysis_period=analysis_period,
            total_interactions=0,
            emotional_diversity=0.0,
            average_valence=0.0,
            average_arousal=0.0,
            emotional_stability=0.5,
            social_adaptability=0.5,
            influence_score=0.0,
            support_given=0,
            support_received=0,
            conflict_involvement=0,
            scenario_performance={}
        )
    
    async def get_analysis_report(
        self,
        analysis_type: AnalysisType,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        time_range: Optional[Tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """获取分析报告"""
        try:
            if analysis_type == AnalysisType.EMOTION_FLOW:
                if not session_id or session_id not in self.flow_history:
                    return {"error": "Session not found or flow not analyzed"}
                
                flow = self.flow_history[session_id]
                return {
                    "type": "emotion_flow",
                    "session_id": session_id,
                    "participants": flow.participants,
                    "duration": (flow.end_time - flow.start_time).total_seconds(),
                    "total_points": len(flow.flow_points),
                    "peaks_count": len(flow.emotional_peaks),
                    "valleys_count": len(flow.emotional_valleys),
                    "turning_points_count": len(flow.turning_points),
                    "overall_trend": flow.overall_trend,
                    "dominant_emotions": flow.dominant_emotions
                }
            
            elif analysis_type == AnalysisType.NETWORK_MAP:
                # 返回最新的网络分析
                if not self.network_history:
                    return {"error": "No network data available"}
                
                latest_network = list(self.network_history.values())[-1]
                return {
                    "type": "network_map",
                    "network_id": latest_network.network_id,
                    "nodes_count": len(latest_network.nodes),
                    "edges_count": len(latest_network.edges),
                    "clusters_count": len(latest_network.clusters),
                    "central_nodes": latest_network.central_nodes,
                    "network_cohesion": latest_network.network_cohesion,
                    "polarization_level": latest_network.polarization_level
                }
            
            else:
                return {"error": f"Unsupported analysis type: {analysis_type}"}
                
        except Exception as e:
            logger.error(f"Failed to generate analysis report: {e}")
            return {"error": str(e)}
    
    async def cleanup_old_data(self, max_age_hours: int = 24) -> None:
        """清理旧数据"""
        cutoff_time = utc_now() - timedelta(hours=max_age_hours)
        
        # 清理流历史
        expired_flows = [
            session_id for session_id, flow in self.flow_history.items()
            if flow.end_time < cutoff_time
        ]
        
        for session_id in expired_flows:
            del self.flow_history[session_id]
        
        logger.info(f"Cleaned up {len(expired_flows)} expired emotion flows")
