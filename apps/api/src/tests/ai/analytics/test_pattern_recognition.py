"""
行为模式识别测试
"""

import pytest
import numpy as np
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from unittest.mock import MagicMock, patch
from src.ai.analytics.models import BehaviorEvent, BehaviorPattern
from src.ai.analytics.behavior.pattern_recognition import (
    SequencePatternMiner, BehaviorClustering, PatternRecognitionEngine
)

class TestSequencePatternMiner:
    """序列模式挖掘测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.miner = SequencePatternMiner()
        
        # 创建测试事件序列
        base_time = utc_now()
        self.events = [
            BehaviorEvent(
                event_id="e1", user_id="user1", event_type="login",
                timestamp=base_time
            ),
            BehaviorEvent(
                event_id="e2", user_id="user1", event_type="view_dashboard",
                timestamp=base_time + timedelta(minutes=1)
            ),
            BehaviorEvent(
                event_id="e3", user_id="user1", event_type="click_button",
                timestamp=base_time + timedelta(minutes=2)
            ),
            BehaviorEvent(
                event_id="e4", user_id="user2", event_type="login",
                timestamp=base_time + timedelta(minutes=3)
            ),
            BehaviorEvent(
                event_id="e5", user_id="user2", event_type="view_dashboard",
                timestamp=base_time + timedelta(minutes=4)
            ),
        ]
    
    @pytest.mark.asyncio
    async def test_sequence_extraction(self):
        """测试序列提取"""
        sequences = await self.miner.extract_user_sequences(self.events)
        
        # 验证序列数量
        assert len(sequences) == 2  # 两个用户
        
        # 验证序列内容
        user1_sequence = sequences.get("user1", [])
        assert len(user1_sequence) == 3
        assert user1_sequence[0] == "login"
        assert user1_sequence[1] == "view_dashboard"
        assert user1_sequence[2] == "click_button"
    
    @pytest.mark.asyncio
    async def test_frequent_patterns(self):
        """测试频繁模式挖掘"""
        # 添加更多重复模式
        repeated_events = self.events.copy()
        base_time = utc_now() + timedelta(hours=1)
        
        # 添加相似的用户行为
        for i in range(3, 6):
            repeated_events.extend([
                BehaviorEvent(
                    event_id=f"e{i}_1", user_id=f"user{i}",
                    event_type="login", timestamp=base_time
                ),
                BehaviorEvent(
                    event_id=f"e{i}_2", user_id=f"user{i}",
                    event_type="view_dashboard",
                    timestamp=base_time + timedelta(minutes=1)
                ),
            ])
        
        patterns = await self.miner.find_frequent_patterns(
            repeated_events, min_support=0.6
        )
        
        # 验证找到频繁模式
        assert len(patterns) > 0
        
        # 验证模式结构
        for pattern in patterns:
            assert isinstance(pattern, BehaviorPattern)
            assert pattern.support >= 0.6
            assert len(pattern.sequence) >= 2
    
    @pytest.mark.asyncio
    async def test_pattern_confidence(self):
        """测试模式置信度计算"""
        patterns = await self.miner.find_frequent_patterns(self.events, min_support=0.3)
        
        for pattern in patterns:
            # 置信度应在0-1之间
            assert 0 <= pattern.confidence <= 1
            
            # 支持度应满足最小支持度要求
            assert pattern.support >= 0.3
    
    def test_sequence_similarity(self):
        """测试序列相似度计算"""
        seq1 = ["login", "view_dashboard", "click_button"]
        seq2 = ["login", "view_dashboard", "logout"]
        seq3 = ["register", "setup_profile"]
        
        # 相似序列应有高相似度
        similarity1 = self.miner.calculate_sequence_similarity(seq1, seq2)
        assert similarity1 > 0.5
        
        # 不相似序列应有低相似度
        similarity2 = self.miner.calculate_sequence_similarity(seq1, seq3)
        assert similarity2 < 0.5

class TestBehaviorClustering:
    """行为聚类测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.clusterer = BehaviorClustering()
        
        # 创建测试事件
        self.events = []
        base_time = utc_now()
        
        # 创建两组不同的行为模式
        for i in range(10):
            # 组1: 频繁的页面浏览用户
            self.events.append(BehaviorEvent(
                event_id=f"group1_{i}",
                user_id=f"user_{i}",
                event_type="page_view",
                timestamp=base_time + timedelta(minutes=i),
                properties={"session_duration": 300 + i * 10}
            ))
            
            # 组2: 频繁点击用户
            self.events.append(BehaviorEvent(
                event_id=f"group2_{i}",
                user_id=f"user_{i + 10}",
                event_type="click",
                timestamp=base_time + timedelta(minutes=i),
                properties={"session_duration": 100 + i * 5}
            ))
    
    @pytest.mark.asyncio
    async def test_kmeans_clustering(self):
        """测试K-means聚类"""
        clusters = await self.clusterer.kmeans_clustering(self.events, n_clusters=2)
        
        # 验证聚类结果
        assert len(clusters) == 2
        
        # 验证每个聚类的结构
        for cluster_id, events in clusters.items():
            assert len(events) > 0
            assert all(isinstance(event, BehaviorEvent) for event in events)
    
    @pytest.mark.asyncio
    async def test_dbscan_clustering(self):
        """测试DBSCAN聚类"""
        clusters = await self.clusterer.dbscan_clustering(self.events)
        
        # DBSCAN可能产生不同数量的聚类（包括噪声点）
        assert isinstance(clusters, dict)
        
        # 验证聚类质量
        total_events = sum(len(events) for events in clusters.values())
        assert total_events <= len(self.events)  # 可能有噪声点被排除
    
    @pytest.mark.asyncio
    async def test_feature_extraction(self):
        """测试特征提取"""
        features = await self.clusterer.extract_behavioral_features(self.events)
        
        # 验证特征矩阵
        assert features.shape[0] > 0  # 至少有一些样本
        assert features.shape[1] > 0  # 至少有一些特征
        
        # 验证特征值的合理性
        assert not np.any(np.isnan(features))  # 没有NaN值
    
    @pytest.mark.asyncio 
    async def test_cluster_analysis(self):
        """测试聚类分析"""
        analysis = await self.clusterer.analyze_clusters(self.events, n_clusters=2)
        
        # 验证分析结果结构
        assert "cluster_centers" in analysis
        assert "cluster_sizes" in analysis
        assert "silhouette_score" in analysis
        
        # 验证轮廓系数
        silhouette_score = analysis["silhouette_score"]
        assert -1 <= silhouette_score <= 1

class TestPatternRecognitionEngine:
    """模式识别引擎测试"""
    
    def setup_method(self):
        """测试前设置"""
        self.engine = PatternRecognitionEngine()
        
        # 创建复杂的测试数据
        self.complex_events = []
        base_time = utc_now()
        
        # 模拟多个用户的复杂行为序列
        user_behaviors = [
            ["login", "view_dashboard", "create_task", "logout"],
            ["login", "view_dashboard", "edit_profile", "logout"],
            ["register", "setup_profile", "view_tutorial", "create_task"],
            ["login", "view_dashboard", "create_task", "share_task", "logout"]
        ]
        
        for user_id, behavior in enumerate(user_behaviors):
            for step, event_type in enumerate(behavior):
                self.complex_events.append(BehaviorEvent(
                    event_id=f"complex_{user_id}_{step}",
                    user_id=f"user_{user_id}",
                    event_type=event_type,
                    timestamp=base_time + timedelta(minutes=step),
                    properties={"step": step, "total_steps": len(behavior)}
                ))
    
    @pytest.mark.asyncio
    async def test_comprehensive_analysis(self):
        """测试综合分析"""
        results = await self.engine.analyze_patterns(self.complex_events)
        
        # 验证结果结构
        assert "sequential_patterns" in results
        assert "behavioral_clusters" in results
        assert "pattern_insights" in results
        
        # 验证序列模式
        patterns = results["sequential_patterns"]
        assert len(patterns) > 0
        
        # 验证聚类结果
        clusters = results["behavioral_clusters"]
        assert len(clusters) > 0
    
    @pytest.mark.asyncio
    async def test_real_time_pattern_detection(self):
        """测试实时模式检测"""
        # 先训练模式
        await self.engine.train_patterns(self.complex_events[:15])
        
        # 测试新事件的模式匹配
        new_event = BehaviorEvent(
            event_id="new_test",
            user_id="new_user",
            event_type="login",
            timestamp=utc_now(),
            properties={}
        )
        
        match_results = await self.engine.match_real_time_patterns([new_event])
        
        # 验证匹配结果
        assert isinstance(match_results, list)
        
        # 如果有匹配，验证匹配质量
        for match in match_results:
            assert "pattern_id" in match
            assert "confidence" in match
            assert 0 <= match["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_pattern_evolution_tracking(self):
        """测试模式演化跟踪"""
        # 分时间段的数据
        old_events = self.complex_events[:10]
        new_events = self.complex_events[10:]
        
        # 分析模式演化
        evolution = await self.engine.track_pattern_evolution(old_events, new_events)
        
        # 验证演化结果
        assert "new_patterns" in evolution
        assert "disappeared_patterns" in evolution
        assert "evolved_patterns" in evolution
    
    @pytest.mark.asyncio
    async def test_anomaly_integration(self):
        """测试与异常检测的集成"""
        # 添加一个明显异常的事件
        anomaly_event = BehaviorEvent(
            event_id="anomaly_test",
            user_id="anomaly_user",
            event_type="unusual_action",
            timestamp=utc_now(),
            properties={"duration": 10000}  # 异常长的持续时间
        )
        
        all_events = self.complex_events + [anomaly_event]
        
        # 运行分析
        results = await self.engine.analyze_patterns(all_events)
        
        # 验证异常事件是否被正确处理
        assert "anomalous_events" in results or len(results["sequential_patterns"]) > 0
    
    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self):
        """测试大数据集性能"""
        # 创建大量测试数据
        large_dataset = []
        base_time = utc_now()
        
        for i in range(1000):
            large_dataset.append(BehaviorEvent(
                event_id=f"perf_test_{i}",
                user_id=f"user_{i % 100}",  # 100个用户
                event_type=f"action_{i % 10}",  # 10种动作类型
                timestamp=base_time + timedelta(seconds=i),
                properties={"index": i}
            ))
        
        import time
        start_time = time.time()
        
        # 执行分析
        results = await self.engine.analyze_patterns(large_dataset)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 验证性能（应在合理时间内完成）
        assert processing_time < 30  # 30秒内完成
        assert len(results["sequential_patterns"]) > 0
