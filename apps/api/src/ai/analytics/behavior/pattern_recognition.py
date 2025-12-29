"""
行为模式识别引擎

实现序列模式挖掘、用户行为聚类和模式匹配功能。
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from collections import defaultdict, Counter
from dataclasses import dataclass
import json
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from ..models import BehaviorEvent, UserSession, BehaviorPattern
from ..storage.event_store import EventStore

from src.core.logging import get_logger
logger = get_logger(__name__)

# 机器学习相关导入

@dataclass
class PatternMiningResult:
    """模式挖掘结果"""
    patterns: List[BehaviorPattern]
    total_sequences: int
    processing_time_seconds: float
    algorithm_params: Dict[str, Any]

class SequencePatternMiner:
    """序列模式挖掘器
    
    实现类似PrefixSpan的序列模式挖掘算法
    """
    
    def __init__(
        self,
        min_support: float = 0.05,
        max_pattern_length: int = 10,
        min_pattern_length: int = 2
    ):
        self.min_support = min_support
        self.max_pattern_length = max_pattern_length
        self.min_pattern_length = min_pattern_length
    
    async def mine_patterns(
        self, 
        user_sessions: List[UserSession],
        event_sequences: List[List[Dict[str, Any]]]
    ) -> PatternMiningResult:
        """挖掘序列模式"""
        start_time = asyncio.get_running_loop().time()
        
        try:
            # 预处理序列数据
            processed_sequences = await self._preprocess_sequences(event_sequences)
            
            # 挖掘频繁模式
            frequent_patterns = await self._mine_frequent_patterns(processed_sequences)
            
            # 转换为BehaviorPattern对象
            behavior_patterns = []
            for pattern, support in frequent_patterns.items():
                confidence = await self._calculate_confidence(pattern, processed_sequences)
                
                behavior_pattern = BehaviorPattern(
                    pattern_name=f"序列模式_{len(behavior_patterns) + 1}",
                    pattern_type="sequence",
                    pattern_definition={
                        "sequence": pattern,
                        "algorithm": "sequence_mining"
                    },
                    support=support,
                    confidence=confidence,
                    users_count=len([seq for seq in processed_sequences if self._contains_pattern(seq, pattern)])
                )
                behavior_patterns.append(behavior_pattern)
            
            processing_time = asyncio.get_running_loop().time() - start_time
            
            return PatternMiningResult(
                patterns=behavior_patterns,
                total_sequences=len(processed_sequences),
                processing_time_seconds=processing_time,
                algorithm_params={
                    "min_support": self.min_support,
                    "max_pattern_length": self.max_pattern_length,
                    "min_pattern_length": self.min_pattern_length
                }
            )
            
        except Exception as e:
            logger.error(f"序列模式挖掘失败: {e}")
            raise
    
    async def _preprocess_sequences(
        self, 
        event_sequences: List[List[Dict[str, Any]]]
    ) -> List[List[str]]:
        """预处理事件序列"""
        processed = []
        
        for sequence in event_sequences:
            # 提取事件名称序列
            event_names = []
            for event in sorted(sequence, key=lambda e: e['timestamp']):
                event_names.append(event['event_name'])
            
            if len(event_names) >= self.min_pattern_length:
                processed.append(event_names)
        
        return processed
    
    async def _mine_frequent_patterns(
        self, 
        sequences: List[List[str]]
    ) -> Dict[Tuple[str, ...], float]:
        """挖掘频繁模式"""
        pattern_counts = defaultdict(int)
        total_sequences = len(sequences)
        
        # 生成候选模式
        for sequence in sequences:
            # 生成所有可能的子序列
            subsequences = self._generate_subsequences(sequence)
            for subseq in subsequences:
                if self.min_pattern_length <= len(subseq) <= self.max_pattern_length:
                    pattern_counts[tuple(subseq)] += 1
        
        # 筛选频繁模式
        frequent_patterns = {}
        min_count = int(total_sequences * self.min_support)
        
        for pattern, count in pattern_counts.items():
            if count >= min_count:
                support = count / total_sequences
                frequent_patterns[pattern] = support
        
        return frequent_patterns
    
    def _generate_subsequences(self, sequence: List[str]) -> List[List[str]]:
        """生成序列的所有子序列"""
        subsequences = []
        n = len(sequence)
        
        for length in range(self.min_pattern_length, min(self.max_pattern_length + 1, n + 1)):
            for start in range(n - length + 1):
                subseq = sequence[start:start + length]
                subsequences.append(subseq)
        
        return subsequences
    
    async def _calculate_confidence(
        self, 
        pattern: Tuple[str, ...], 
        sequences: List[List[str]]
    ) -> float:
        """计算模式置信度"""
        if len(pattern) < 2:
            return 1.0
        
        # 前缀出现次数
        prefix = pattern[:-1]
        prefix_count = sum(1 for seq in sequences if self._contains_pattern(seq, prefix))
        
        # 完整模式出现次数
        pattern_count = sum(1 for seq in sequences if self._contains_pattern(seq, pattern))
        
        if prefix_count == 0:
            return 0.0
        
        return pattern_count / prefix_count
    
    def _contains_pattern(self, sequence: List[str], pattern: Tuple[str, ...]) -> bool:
        """检查序列是否包含模式"""
        pattern_list = list(pattern)
        pattern_len = len(pattern_list)
        
        for i in range(len(sequence) - pattern_len + 1):
            if sequence[i:i + pattern_len] == pattern_list:
                return True
        
        return False

class BehaviorClustering:
    """用户行为聚类分析器"""
    
    def __init__(
        self,
        n_clusters: Optional[int] = None,
        algorithm: str = 'kmeans',  # 'kmeans', 'dbscan'
        auto_tune_clusters: bool = True
    ):
        self.n_clusters = n_clusters
        self.algorithm = algorithm
        self.auto_tune_clusters = auto_tune_clusters
        self.scaler = StandardScaler()
        self.clusterer = None
        self.feature_names = []
    
    async def cluster_users(
        self,
        user_sessions: List[UserSession],
        user_event_features: Dict[str, Dict[str, Any]]
    ) -> PatternMiningResult:
        """对用户行为进行聚类"""
        start_time = asyncio.get_running_loop().time()
        
        try:
            # 构建特征矩阵
            feature_matrix, user_ids = await self._build_feature_matrix(user_event_features)
            
            if len(feature_matrix) < 2:
                logger.warning("用户数量太少，无法进行聚类")
                return PatternMiningResult([], len(user_sessions), 0, {})
            
            # 标准化特征
            normalized_features = self.scaler.fit_transform(feature_matrix)
            
            # 执行聚类
            if self.algorithm == 'kmeans':
                cluster_labels = await self._kmeans_clustering(normalized_features)
            elif self.algorithm == 'dbscan':
                cluster_labels = await self._dbscan_clustering(normalized_features)
            else:
                raise ValueError(f"不支持的聚类算法: {self.algorithm}")
            
            # 生成聚类模式
            patterns = await self._generate_cluster_patterns(
                user_ids, cluster_labels, user_event_features
            )
            
            processing_time = asyncio.get_running_loop().time() - start_time
            
            return PatternMiningResult(
                patterns=patterns,
                total_sequences=len(user_sessions),
                processing_time_seconds=processing_time,
                algorithm_params={
                    "algorithm": self.algorithm,
                    "n_clusters": self.n_clusters or len(set(cluster_labels)),
                    "n_features": len(self.feature_names)
                }
            )
            
        except Exception as e:
            logger.error(f"用户聚类失败: {e}")
            raise
    
    async def _build_feature_matrix(
        self,
        user_event_features: Dict[str, Dict[str, Any]]
    ) -> Tuple[np.ndarray, List[str]]:
        """构建特征矩阵"""
        if not user_event_features:
            return np.array([]), []
        
        # 确定特征名称
        all_features = set()
        for features in user_event_features.values():
            all_features.update(features.keys())
        
        self.feature_names = sorted(all_features)
        user_ids = list(user_event_features.keys())
        
        # 构建特征矩阵
        feature_matrix = []
        for user_id in user_ids:
            user_features = user_event_features[user_id]
            feature_vector = []
            
            for feature_name in self.feature_names:
                value = user_features.get(feature_name, 0)
                # 确保数值类型
                if isinstance(value, (int, float)):
                    feature_vector.append(float(value))
                else:
                    feature_vector.append(0.0)
            
            feature_matrix.append(feature_vector)
        
        return np.array(feature_matrix), user_ids
    
    async def _kmeans_clustering(self, features: np.ndarray) -> np.ndarray:
        """K-means聚类"""
        # 自动确定最佳聚类数
        if self.auto_tune_clusters and self.n_clusters is None:
            self.n_clusters = await self._find_optimal_clusters(features)
        elif self.n_clusters is None:
            self.n_clusters = min(8, len(features) // 2)  # 默认策略
        
        # 执行K-means
        self.clusterer = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        
        cluster_labels = self.clusterer.fit_predict(features)
        return cluster_labels
    
    async def _dbscan_clustering(self, features: np.ndarray) -> np.ndarray:
        """DBSCAN聚类"""
        # 使用DBSCAN进行基于密度的聚类
        self.clusterer = DBSCAN(
            eps=0.5,
            min_samples=3
        )
        
        cluster_labels = self.clusterer.fit_predict(features)
        return cluster_labels
    
    async def _find_optimal_clusters(self, features: np.ndarray) -> int:
        """使用轮廓系数找到最佳聚类数"""
        max_clusters = min(10, len(features) - 1)
        best_score = -1
        best_k = 2
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features)
            
            # 计算轮廓系数
            score = silhouette_score(features, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        logger.info(f"最佳聚类数: {best_k}, 轮廓系数: {best_score:.3f}")
        return best_k
    
    async def _generate_cluster_patterns(
        self,
        user_ids: List[str],
        cluster_labels: np.ndarray,
        user_event_features: Dict[str, Dict[str, Any]]
    ) -> List[BehaviorPattern]:
        """生成聚类模式"""
        patterns = []
        unique_labels = set(cluster_labels)
        
        for label in unique_labels:
            if label == -1:  # DBSCAN的噪声点
                continue
            
            # 获取该聚类的用户
            cluster_user_indices = np.where(cluster_labels == label)[0]
            cluster_users = [user_ids[i] for i in cluster_user_indices]
            
            # 分析聚类特征
            cluster_features = self._analyze_cluster_features(
                cluster_users, user_event_features
            )
            
            pattern = BehaviorPattern(
                pattern_name=f"用户群体_{int(label)}",
                pattern_type="clustering",
                pattern_definition={
                    "cluster_id": int(label),
                    "algorithm": self.algorithm,
                    "dominant_features": cluster_features["dominant_features"],
                    "average_features": cluster_features["average_features"]
                },
                support=len(cluster_users) / len(user_ids),
                confidence=1.0,  # 聚类的置信度为1
                users_count=len(cluster_users),
                examples=cluster_users[:10]  # 存储前10个示例用户
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _analyze_cluster_features(
        self,
        cluster_users: List[str],
        user_event_features: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """分析聚类特征"""
        if not cluster_users:
            return {"dominant_features": [], "average_features": {}}
        
        # 收集聚类用户的所有特征
        cluster_feature_values = defaultdict(list)
        
        for user_id in cluster_users:
            user_features = user_event_features.get(user_id, {})
            for feature_name, value in user_features.items():
                if isinstance(value, (int, float)):
                    cluster_feature_values[feature_name].append(value)
        
        # 计算平均特征值
        average_features = {}
        for feature_name, values in cluster_feature_values.items():
            if values:
                average_features[feature_name] = sum(values) / len(values)
        
        # 找出主导特征(高于平均水平)
        all_feature_averages = {}
        for user_features in user_event_features.values():
            for feature_name, value in user_features.items():
                if isinstance(value, (int, float)):
                    if feature_name not in all_feature_averages:
                        all_feature_averages[feature_name] = []
                    all_feature_averages[feature_name].append(value)
        
        # 计算全局平均值
        global_averages = {}
        for feature_name, values in all_feature_averages.items():
            if values:
                global_averages[feature_name] = sum(values) / len(values)
        
        # 识别主导特征
        dominant_features = []
        for feature_name, cluster_avg in average_features.items():
            global_avg = global_averages.get(feature_name, 0)
            if global_avg > 0 and cluster_avg > global_avg * 1.2:  # 高于全局平均20%
                dominant_features.append({
                    "feature": feature_name,
                    "cluster_average": cluster_avg,
                    "global_average": global_avg,
                    "ratio": cluster_avg / global_avg
                })
        
        # 按比例排序
        dominant_features.sort(key=lambda x: x["ratio"], reverse=True)
        
        return {
            "dominant_features": dominant_features[:5],  # 前5个主导特征
            "average_features": average_features
        }

class PatternRecognitionEngine:
    """行为模式识别引擎"""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
        self.sequence_miner = SequencePatternMiner()
        self.behavior_clusterer = BehaviorClustering()
        
        # 缓存已识别的模式
        self.pattern_cache = {}
        self.last_update = None
    
    async def analyze_behavior_patterns(
        self,
        user_ids: Optional[List[str]] = None,
        time_range_days: int = 30,
        pattern_types: List[str] = ["sequence", "clustering"]
    ) -> Dict[str, PatternMiningResult]:
        """分析行为模式"""
        results = {}
        
        try:
            # 获取用户会话和事件数据
            user_sessions, event_sequences, user_features = await self._prepare_analysis_data(
                user_ids, time_range_days
            )
            
            if not user_sessions:
                logger.warning("没有找到用户会话数据")
                return results
            
            # 序列模式挖掘
            if "sequence" in pattern_types:
                logger.info("开始序列模式挖掘...")
                results["sequence"] = await self.sequence_miner.mine_patterns(
                    user_sessions, event_sequences
                )
            
            # 用户行为聚类
            if "clustering" in pattern_types:
                logger.info("开始用户行为聚类...")
                results["clustering"] = await self.behavior_clusterer.cluster_users(
                    user_sessions, user_features
                )
            
            # 更新缓存
            self.last_update = utc_now()
            self.pattern_cache = results
            
            return results
            
        except Exception as e:
            logger.error(f"行为模式分析失败: {e}")
            raise
    
    async def _prepare_analysis_data(
        self,
        user_ids: Optional[List[str]],
        time_range_days: int
    ) -> Tuple[List[UserSession], List[List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
        """准备分析数据"""
        # 时间范围
        end_time = utc_now()
        start_time = end_time - timedelta(days=time_range_days)
        
        # 获取事件数据
        from ..models import EventQueryFilter
        filter_params = EventQueryFilter(
            start_time=start_time,
            end_time=end_time,
            limit=50000  # 限制数据量
        )
        
        if user_ids:
            # 如果指定了用户，分别获取每个用户的数据
            all_events = []
            for user_id in user_ids:
                filter_params.user_id = user_id
                events, _ = await self.event_store.query_events(filter_params)
                all_events.extend(events)
        else:
            all_events, _ = await self.event_store.query_events(filter_params)
        
        if not all_events:
            return [], [], {}
        
        # 按用户和会话分组事件
        user_session_events = defaultdict(lambda: defaultdict(list))
        for event in all_events:
            user_id = event['user_id']
            session_id = event['session_id']
            user_session_events[user_id][session_id].append(event)
        
        # 构建用户会话和事件序列
        user_sessions = []
        event_sequences = []
        
        for user_id, sessions in user_session_events.items():
            for session_id, events in sessions.items():
                if len(events) < 2:  # 跳过事件太少的会话
                    continue
                
                # 创建会话对象
                sorted_events = sorted(events, key=lambda e: e['timestamp'])
                start_time = datetime.fromisoformat(sorted_events[0]['timestamp'])
                end_time = datetime.fromisoformat(sorted_events[-1]['timestamp'])
                
                session = UserSession(
                    session_id=session_id,
                    user_id=user_id,
                    start_time=start_time,
                    end_time=end_time,
                    events_count=len(events)
                )
                
                user_sessions.append(session)
                event_sequences.append(events)
        
        # 构建用户特征
        user_features = await self._extract_user_features(user_session_events)
        
        return user_sessions, event_sequences, user_features
    
    async def _extract_user_features(
        self,
        user_session_events: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ) -> Dict[str, Dict[str, Any]]:
        """提取用户特征"""
        user_features = {}
        
        for user_id, sessions in user_session_events.items():
            # 聚合用户所有事件
            all_user_events = []
            for events in sessions.values():
                all_user_events.extend(events)
            
            if not all_user_events:
                continue
            
            # 计算用户特征
            features = {
                # 基础统计特征
                'total_events': len(all_user_events),
                'total_sessions': len(sessions),
                'avg_events_per_session': len(all_user_events) / len(sessions),
                
                # 时间特征
                'total_active_days': len(set(
                    datetime.fromisoformat(event['timestamp']).date()
                    for event in all_user_events
                )),
                
                # 事件类型分布
                'user_action_ratio': len([e for e in all_user_events if e['event_type'] == 'user_action']) / len(all_user_events),
                'feedback_event_ratio': len([e for e in all_user_events if e['event_type'] == 'feedback_event']) / len(all_user_events),
                'error_event_ratio': len([e for e in all_user_events if e['event_type'] == 'error_event']) / len(all_user_events),
            }
            
            # 计算平均会话时长
            session_durations = []
            for events in sessions.values():
                if len(events) >= 2:
                    sorted_events = sorted(events, key=lambda e: e['timestamp'])
                    start = datetime.fromisoformat(sorted_events[0]['timestamp'])
                    end = datetime.fromisoformat(sorted_events[-1]['timestamp'])
                    duration = (end - start).total_seconds() / 60  # 分钟
                    session_durations.append(duration)
            
            if session_durations:
                features['avg_session_duration_minutes'] = sum(session_durations) / len(session_durations)
                features['max_session_duration_minutes'] = max(session_durations)
            else:
                features['avg_session_duration_minutes'] = 0
                features['max_session_duration_minutes'] = 0
            
            # 计算事件响应时间特征
            durations = [
                event.get('duration_ms', 0) for event in all_user_events
                if event.get('duration_ms') is not None and event.get('duration_ms') > 0
            ]
            
            if durations:
                features['avg_response_time_ms'] = sum(durations) / len(durations)
                features['max_response_time_ms'] = max(durations)
            else:
                features['avg_response_time_ms'] = 0
                features['max_response_time_ms'] = 0
            
            # 最频繁的事件名称
            event_name_counts = Counter(event['event_name'] for event in all_user_events)
            most_common_event = event_name_counts.most_common(1)
            if most_common_event:
                features['most_common_event'] = most_common_event[0][0]
                features['most_common_event_ratio'] = most_common_event[0][1] / len(all_user_events)
            else:
                features['most_common_event'] = 'unknown'
                features['most_common_event_ratio'] = 0
            
            user_features[user_id] = features
        
        return user_features
    
    async def find_similar_patterns(
        self,
        pattern: BehaviorPattern,
        similarity_threshold: float = 0.8
    ) -> List[BehaviorPattern]:
        """查找相似的模式"""
        similar_patterns = []
        
        # 如果缓存为空，先进行一次分析
        if not self.pattern_cache:
            await self.analyze_behavior_patterns()
        
        # 在所有已识别的模式中查找相似的
        for pattern_type, result in self.pattern_cache.items():
            for cached_pattern in result.patterns:
                if await self._calculate_pattern_similarity(pattern, cached_pattern) >= similarity_threshold:
                    similar_patterns.append(cached_pattern)
        
        return similar_patterns
    
    async def _calculate_pattern_similarity(
        self,
        pattern1: BehaviorPattern,
        pattern2: BehaviorPattern
    ) -> float:
        """计算模式相似度"""
        if pattern1.pattern_type != pattern2.pattern_type:
            return 0.0
        
        # 基于支持度和置信度的相似度
        support_sim = 1 - abs(pattern1.support - pattern2.support)
        confidence_sim = 1 - abs(pattern1.confidence - pattern2.confidence)
        
        # 基于模式定义的相似度
        def_sim = 0.0
        if pattern1.pattern_type == "sequence":
            seq1 = pattern1.pattern_definition.get("sequence", [])
            seq2 = pattern2.pattern_definition.get("sequence", [])
            def_sim = self._sequence_similarity(seq1, seq2)
        elif pattern1.pattern_type == "clustering":
            # 聚类模式的相似度基于用户重叠率
            users1 = set(pattern1.examples)
            users2 = set(pattern2.examples)
            if users1 and users2:
                def_sim = len(users1.intersection(users2)) / len(users1.union(users2))
        
        # 加权平均
        return (support_sim * 0.3 + confidence_sim * 0.3 + def_sim * 0.4)
    
    def _sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """计算序列相似度"""
        if not seq1 or not seq2:
            return 0.0
        
        # 使用最长公共子序列计算相似度
        def lcs_length(s1, s2):
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        
        return lcs_len / max_len if max_len > 0 else 0.0
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        stats = {
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'cached_patterns': {
                pattern_type: len(result.patterns)
                for pattern_type, result in self.pattern_cache.items()
            },
            'sequence_miner_params': {
                'min_support': self.sequence_miner.min_support,
                'max_pattern_length': self.sequence_miner.max_pattern_length,
                'min_pattern_length': self.sequence_miner.min_pattern_length
            },
            'clustering_params': {
                'algorithm': self.behavior_clusterer.algorithm,
                'n_clusters': self.behavior_clusterer.n_clusters,
                'auto_tune_clusters': self.behavior_clusterer.auto_tune_clusters
            }
        }
        
        return stats
