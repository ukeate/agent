"""
Emotion Contagion Detection Module
情感传染检测模块

This module provides advanced emotion contagion detection and analysis capabilities:
- Real-time contagion detection
- Contagion pattern analysis
- Propagation network mapping
- Contagion prediction and prevention
"""

from src.core.utils.timezone_utils import utc_now
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
import networkx as nx
from .group_emotion_models import (
    EmotionState,
    EmotionContagionEvent,
    ContagionPattern,
    EmotionContagionType,
    generate_event_id
)

@dataclass
class ContagionNode:
    """传染网络节点"""
    participant_id: str
    emotion: str
    intensity: float
    timestamp: datetime
    is_source: bool = False
    propagation_delay: float = 0.0  # 从源传播的延迟(秒)

@dataclass
class ContagionPath:
    """传染路径"""
    path_id: str
    nodes: List[ContagionNode]
    total_delay: float
    intensity_change: float  # 强度变化率
    path_length: int
    effectiveness: float  # 路径有效性 [0,1]

@dataclass
class ContagionNetwork:
    """传染网络"""
    network_id: str
    emotion: str
    source_node: ContagionNode
    affected_nodes: List[ContagionNode]
    propagation_paths: List[ContagionPath]
    network_graph: Optional[nx.DiGraph] = None
    coverage_rate: float = 0.0  # 覆盖率
    average_delay: float = 0.0
    peak_intensity: float = 0.0
    decay_rate: float = 0.0

class EmotionContagionDetector:
    """情感传染检测器"""
    
    def __init__(
        self,
        detection_window_seconds: int = 300,  # 5分钟检测窗口
        min_contagion_threshold: float = 0.3,  # 最小传染阈值
        max_propagation_delay: float = 120.0   # 最大传播延迟(秒)
    ):
        self.detection_window = detection_window_seconds
        self.contagion_threshold = min_contagion_threshold
        self.max_delay = max_propagation_delay
        
        # 历史事件缓存
        self.event_buffer = deque(maxlen=1000)
        self.active_networks = {}  # 活跃的传染网络
        
        # 情感相似性映射
        self.emotion_similarity = {
            'joy': {'happiness': 0.9, 'excitement': 0.8, 'satisfaction': 0.7},
            'anger': {'frustration': 0.9, 'irritation': 0.8, 'rage': 0.9},
            'sadness': {'disappointment': 0.8, 'melancholy': 0.7, 'grief': 0.9},
            'fear': {'anxiety': 0.9, 'worry': 0.7, 'panic': 0.8},
            'surprise': {'shock': 0.8, 'amazement': 0.7},
            'disgust': {'contempt': 0.8, 'revulsion': 0.9}
        }
    
    async def detect_contagion_events(
        self,
        current_emotions: Dict[str, EmotionState],
        interaction_history: List[Dict[str, Any]],
        time_window_minutes: int = 5
    ) -> List[EmotionContagionEvent]:
        """检测情感传染事件"""
        
        # 更新事件缓存
        self._update_event_buffer(current_emotions, interaction_history)
        
        # 检测新的传染事件
        new_events = []
        
        # 分析每种情感的传染模式
        emotion_groups = self._group_by_emotion(current_emotions)
        
        for emotion, participants in emotion_groups.items():
            if len(participants) >= 2:  # 至少2人才可能传染
                events = await self._detect_emotion_specific_contagion(
                    emotion, participants, interaction_history, time_window_minutes
                )
                new_events.extend(events)
        
        # 更新活跃网络
        await self._update_active_networks(new_events)
        
        return new_events
    
    def _update_event_buffer(
        self,
        current_emotions: Dict[str, EmotionState],
        interaction_history: List[Dict[str, Any]]
    ):
        """更新事件缓存"""
        
        current_time = utc_now()
        
        # 添加当前情感状态到缓存
        for participant_id, emotion_state in current_emotions.items():
            event_data = {
                'participant_id': participant_id,
                'emotion': emotion_state.emotion,
                'intensity': emotion_state.intensity,
                'timestamp': current_time,
                'type': 'emotion_state'
            }
            self.event_buffer.append(event_data)
        
        # 添加交互历史
        for interaction in interaction_history:
            if interaction.get('timestamp'):
                # 只保留时间窗口内的事件
                if (current_time - interaction['timestamp']).seconds <= self.detection_window:
                    self.event_buffer.append({
                        **interaction,
                        'type': 'interaction'
                    })
        
        # 清理过期事件
        cutoff_time = current_time - timedelta(seconds=self.detection_window)
        self.event_buffer = deque([
            event for event in self.event_buffer 
            if event.get('timestamp', current_time) > cutoff_time
        ], maxlen=1000)
    
    def _group_by_emotion(
        self,
        current_emotions: Dict[str, EmotionState]
    ) -> Dict[str, List[str]]:
        """按情感类型分组参与者"""
        
        emotion_groups = defaultdict(list)
        
        for participant_id, emotion_state in current_emotions.items():
            emotion = emotion_state.emotion
            emotion_groups[emotion].append(participant_id)
            
            # 也添加到相似情感组
            if emotion in self.emotion_similarity:
                for similar_emotion, similarity in self.emotion_similarity[emotion].items():
                    if similarity >= 0.7:  # 高相似度
                        emotion_groups[similar_emotion].append(participant_id)
        
        return dict(emotion_groups)
    
    async def _detect_emotion_specific_contagion(
        self,
        emotion: str,
        participants: List[str],
        interaction_history: List[Dict[str, Any]],
        time_window_minutes: int
    ) -> List[EmotionContagionEvent]:
        """检测特定情感的传染"""
        
        events = []
        
        # 获取该情感的时间序列
        emotion_timeline = self._build_emotion_timeline(
            emotion, participants, time_window_minutes
        )
        
        if len(emotion_timeline) < 2:
            return events
        
        # 检测传染模式
        contagion_chains = self._identify_contagion_chains(emotion_timeline)
        
        for chain in contagion_chains:
            # 验证传染有效性
            if self._validate_contagion_chain(chain):
                event = self._create_contagion_event(emotion, chain, interaction_history)
                events.append(event)
        
        return events
    
    def _build_emotion_timeline(
        self,
        emotion: str,
        participants: List[str],
        time_window_minutes: int
    ) -> List[Dict[str, Any]]:
        """构建情感时间线"""
        
        timeline = []
        cutoff_time = utc_now() - timedelta(minutes=time_window_minutes)
        
        for event in self.event_buffer:
            if (event.get('emotion') == emotion and 
                event.get('participant_id') in participants and
                event.get('timestamp', utc_now()) > cutoff_time):
                timeline.append(event)
        
        # 按时间排序
        timeline.sort(key=lambda x: x.get('timestamp', utc_now()))
        
        return timeline
    
    def _identify_contagion_chains(
        self,
        emotion_timeline: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """识别传染链条"""
        
        chains = []
        
        if len(emotion_timeline) < 2:
            return chains
        
        # 滑动窗口检测传染序列
        for i in range(len(emotion_timeline)):
            current_event = emotion_timeline[i]
            chain = [current_event]
            
            # 寻找可能的传染目标
            for j in range(i + 1, len(emotion_timeline)):
                next_event = emotion_timeline[j]
                
                # 检查时间间隔
                time_diff = (next_event['timestamp'] - current_event['timestamp']).total_seconds()
                if time_diff > self.max_delay:
                    break
                
                # 检查不同参与者
                if next_event['participant_id'] != current_event['participant_id']:
                    # 检查强度变化
                    if self._is_valid_intensity_progression(current_event, next_event):
                        chain.append(next_event)
                        current_event = next_event  # 继续链式传播
            
            # 保存有效的链条
            if len(chain) >= 2:
                chains.append(chain)
        
        # 去重和合并重叠链条
        chains = self._merge_overlapping_chains(chains)
        
        return chains
    
    def _is_valid_intensity_progression(
        self,
        source_event: Dict[str, Any],
        target_event: Dict[str, Any]
    ) -> bool:
        """检查强度变化是否有效"""
        
        source_intensity = source_event.get('intensity', 0.5)
        target_intensity = target_event.get('intensity', 0.5)
        
        # 传染通常会保持或略微降低强度
        # 但也可能因为个人特征而放大
        intensity_ratio = target_intensity / max(source_intensity, 0.1)
        
        # 有效范围：0.3 到 1.5
        return 0.3 <= intensity_ratio <= 1.5
    
    def _merge_overlapping_chains(
        self,
        chains: List[List[Dict[str, Any]]]
    ) -> List[List[Dict[str, Any]]]:
        """合并重叠的链条"""
        
        if not chains:
            return chains
        
        # 按长度排序，优先保留较长的链条
        chains.sort(key=len, reverse=True)
        
        merged = []
        used_events = set()
        
        for chain in chains:
            # 检查是否与已使用的事件重叠
            chain_event_ids = set(
                (event['participant_id'], event['timestamp']) 
                for event in chain
            )
            
            if not chain_event_ids.intersection(used_events):
                merged.append(chain)
                used_events.update(chain_event_ids)
        
        return merged
    
    def _validate_contagion_chain(
        self,
        chain: List[Dict[str, Any]]
    ) -> bool:
        """验证传染链条的有效性"""
        
        if len(chain) < 2:
            return False
        
        # 检查时间连续性
        for i in range(1, len(chain)):
            time_diff = (chain[i]['timestamp'] - chain[i-1]['timestamp']).total_seconds()
            if time_diff > self.max_delay or time_diff < 0:
                return False
        
        # 检查参与者唯一性
        participants = [event['participant_id'] for event in chain]
        if len(set(participants)) != len(participants):
            return False
        
        # 检查强度阈值
        min_intensity = min(event.get('intensity', 0) for event in chain)
        if min_intensity < self.contagion_threshold:
            return False
        
        return True
    
    def _create_contagion_event(
        self,
        emotion: str,
        chain: List[Dict[str, Any]],
        interaction_history: List[Dict[str, Any]]
    ) -> EmotionContagionEvent:
        """创建传染事件"""
        
        source_event = chain[0]
        target_events = chain[1:]
        
        # 计算传播路径
        propagation_path = [event['participant_id'] for event in chain]
        affected_participants = [event['participant_id'] for event in target_events]
        
        # 计算强度放大系数
        source_intensity = source_event.get('intensity', 0.5)
        target_intensities = [event.get('intensity', 0.5) for event in target_events]
        avg_target_intensity = np.mean(target_intensities)
        intensity_amplification = avg_target_intensity / max(source_intensity, 0.1)
        
        # 计算到达率和转换率
        total_participants = len(set(
            interaction.get('sender_id') for interaction in interaction_history
            if interaction.get('sender_id')
        ))
        reach_percentage = len(affected_participants) / max(total_participants, 1)
        conversion_rate = len(target_events) / max(len(propagation_path) - 1, 1)
        
        # 计算传播时间
        propagation_time = (chain[-1]['timestamp'] - chain[0]['timestamp']).total_seconds()
        
        # 确定传染类型
        contagion_type = self._determine_contagion_type_from_chain(chain)
        
        # 计算有效性分数
        effectiveness_score = self._calculate_contagion_effectiveness(
            intensity_amplification, reach_percentage, conversion_rate, propagation_time
        )
        
        return EmotionContagionEvent(
            event_id=generate_event_id(),
            group_id="",  # 将由调用者设置
            timestamp=source_event['timestamp'],
            source_participant=source_event['participant_id'],
            source_emotion=emotion,
            source_intensity=source_intensity,
            propagation_path=propagation_path,
            affected_participants=affected_participants,
            intensity_amplification=intensity_amplification,
            reach_percentage=reach_percentage,
            conversion_rate=conversion_rate,
            propagation_time_seconds=int(propagation_time),
            peak_intensity_time=self._find_peak_intensity_time(chain),
            decay_time_seconds=self._estimate_decay_time(chain),
            contagion_type=contagion_type,
            effectiveness_score=effectiveness_score
        )
    
    def _determine_contagion_type_from_chain(
        self,
        chain: List[Dict[str, Any]]
    ) -> EmotionContagionType:
        """从链条确定传染类型"""
        
        if len(chain) < 2:
            return EmotionContagionType.DAMPENING
        
        # 分析强度变化模式
        intensities = [event.get('intensity', 0.5) for event in chain]
        
        # 计算强度趋势
        intensity_changes = [
            intensities[i] - intensities[i-1] 
            for i in range(1, len(intensities))
        ]
        
        avg_change = np.mean(intensity_changes)
        total_time = (chain[-1]['timestamp'] - chain[0]['timestamp']).total_seconds()
        speed = len(chain) / max(total_time / 60, 1)  # chains per minute
        
        if avg_change > 0.1 and speed > 1:
            return EmotionContagionType.VIRAL
        elif avg_change > 0.05:
            return EmotionContagionType.AMPLIFICATION
        elif avg_change > -0.05:
            return EmotionContagionType.CASCADE
        else:
            return EmotionContagionType.DAMPENING
    
    def _calculate_contagion_effectiveness(
        self,
        intensity_amplification: float,
        reach_percentage: float,
        conversion_rate: float,
        propagation_time: float
    ) -> float:
        """计算传染有效性"""
        
        # 综合考虑各项指标
        intensity_score = min(intensity_amplification, 2.0) / 2.0  # 归一化到[0,1]
        reach_score = min(reach_percentage, 1.0)
        conversion_score = conversion_rate
        
        # 时间惩罚：传播越快越有效
        time_penalty = max(0, 1 - propagation_time / 300)  # 5分钟内最优
        
        effectiveness = (
            intensity_score * 0.3 +
            reach_score * 0.3 +
            conversion_score * 0.2 +
            time_penalty * 0.2
        )
        
        return min(max(effectiveness, 0.0), 1.0)
    
    def _find_peak_intensity_time(
        self,
        chain: List[Dict[str, Any]]
    ) -> datetime:
        """找到峰值强度时间"""
        
        if not chain:
            return utc_now()
        
        max_intensity = 0
        peak_time = chain[0]['timestamp']
        
        for event in chain:
            intensity = event.get('intensity', 0)
            if intensity > max_intensity:
                max_intensity = intensity
                peak_time = event['timestamp']
        
        return peak_time
    
    def _estimate_decay_time(
        self,
        chain: List[Dict[str, Any]]
    ) -> int:
        """估算衰减时间"""
        
        if len(chain) < 2:
            return 60  # 默认1分钟
        
        # 基于传播时间估算衰减时间
        propagation_time = (chain[-1]['timestamp'] - chain[0]['timestamp']).total_seconds()
        
        # 通常衰减时间是传播时间的2-3倍
        decay_time = propagation_time * 2.5
        
        return int(min(max(decay_time, 30), 600))  # 30秒到10分钟之间
    
    async def _update_active_networks(
        self,
        new_events: List[EmotionContagionEvent]
    ):
        """更新活跃传染网络"""
        
        current_time = utc_now()
        
        # 清理过期网络
        expired_networks = [
            net_id for net_id, network in self.active_networks.items()
            if (current_time - network.source_node.timestamp).seconds > 600  # 10分钟
        ]
        
        for net_id in expired_networks:
            del self.active_networks[net_id]
        
        # 添加新网络
        for event in new_events:
            network_id = f"{event.source_emotion}_{event.source_participant}_{event.timestamp.timestamp()}"
            
            # 构建网络结构
            network = self._build_contagion_network(event)
            self.active_networks[network_id] = network
    
    def _build_contagion_network(
        self,
        event: EmotionContagionEvent
    ) -> ContagionNetwork:
        """构建传染网络"""
        
        # 创建源节点
        source_node = ContagionNode(
            participant_id=event.source_participant,
            emotion=event.source_emotion,
            intensity=event.source_intensity,
            timestamp=event.timestamp,
            is_source=True
        )
        
        # 创建受影响节点
        affected_nodes = []
        for i, participant_id in enumerate(event.affected_participants):
            node = ContagionNode(
                participant_id=participant_id,
                emotion=event.source_emotion,
                intensity=event.source_intensity * event.intensity_amplification,
                timestamp=event.timestamp + timedelta(
                    seconds=event.propagation_time_seconds * (i + 1) / len(event.affected_participants)
                ),
                propagation_delay=event.propagation_time_seconds * (i + 1) / len(event.affected_participants)
            )
            affected_nodes.append(node)
        
        # 创建传播路径
        paths = []
        all_nodes = [source_node] + affected_nodes
        
        for i in range(1, len(all_nodes)):
            path = ContagionPath(
                path_id=f"path_{i}",
                nodes=all_nodes[:i+1],
                total_delay=all_nodes[i].propagation_delay,
                intensity_change=(all_nodes[i].intensity - source_node.intensity) / source_node.intensity,
                path_length=i + 1,
                effectiveness=event.effectiveness_score
            )
            paths.append(path)
        
        # 构建网络图
        graph = nx.DiGraph()
        for i in range(len(all_nodes) - 1):
            graph.add_edge(
                all_nodes[i].participant_id,
                all_nodes[i + 1].participant_id,
                weight=all_nodes[i + 1].intensity,
                delay=all_nodes[i + 1].propagation_delay
            )
        
        return ContagionNetwork(
            network_id=f"{event.source_emotion}_{event.source_participant}",
            emotion=event.source_emotion,
            source_node=source_node,
            affected_nodes=affected_nodes,
            propagation_paths=paths,
            network_graph=graph,
            coverage_rate=event.reach_percentage,
            average_delay=np.mean([node.propagation_delay for node in affected_nodes]),
            peak_intensity=max(node.intensity for node in all_nodes),
            decay_rate=1.0 / event.decay_time_seconds if event.decay_time_seconds > 0 else 0.0
        )
    
    def get_active_networks(self) -> Dict[str, ContagionNetwork]:
        """获取活跃的传染网络"""
        return self.active_networks.copy()
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        
        if not self.active_networks:
            return {
                'total_networks': 0,
                'total_affected_participants': 0,
                'average_coverage_rate': 0.0,
                'most_contagious_emotion': None,
                'network_efficiency': 0.0
            }
        
        total_affected = sum(
            len(network.affected_nodes) 
            for network in self.active_networks.values()
        )
        
        average_coverage = np.mean([
            network.coverage_rate 
            for network in self.active_networks.values()
        ])
        
        # 统计最传染性的情感
        emotion_counts = Counter(
            network.emotion 
            for network in self.active_networks.values()
        )
        most_contagious = emotion_counts.most_common(1)[0][0] if emotion_counts else None
        
        # 计算网络效率
        network_efficiency = np.mean([
            len(network.affected_nodes) / max(network.average_delay, 1)
            for network in self.active_networks.values()
        ])
        
        return {
            'total_networks': len(self.active_networks),
            'total_affected_participants': total_affected,
            'average_coverage_rate': average_coverage,
            'most_contagious_emotion': most_contagious,
            'network_efficiency': network_efficiency,
            'emotions_distribution': dict(emotion_counts)
        }
