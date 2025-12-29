"""
Task 6情感社交分析工具完整单元测试套件
测试SocialAnalyticsTools的所有分析功能和数据处理能力
"""

from src.core.utils.timezone_utils import utc_now
import pytest
import asyncio
from datetime import timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, patch, AsyncMock
import json
from ai.emotion_modeling.social_analytics_tools import (
    SocialAnalyticsTools,
    EmotionFlow,
    SocialNetworkAnalysis,
    InfluencePattern,
    GroupCohesionMetrics,
    ConversationQuality,
    SocialDynamicsInsight
)
from ai.emotion_modeling.models import EmotionVector, SocialContext

@pytest.fixture
def analytics_tools():
    """创建社交分析工具实例"""
    return SocialAnalyticsTools()

@pytest.fixture
def sample_conversation_data():
    """创建测试对话数据"""
    base_time = utc_now()
    return [
        {
            "timestamp": base_time,
            "user_id": "user1",
            "message": "我觉得这个方案很好",
            "emotions": {"happiness": 0.7, "confidence": 0.6},
            "response_to": None
        },
        {
            "timestamp": base_time + timedelta(seconds=30),
            "user_id": "user2", 
            "message": "我不太确定这样是否可行",
            "emotions": {"uncertainty": 0.8, "concern": 0.5},
            "response_to": "user1"
        },
        {
            "timestamp": base_time + timedelta(seconds=60),
            "user_id": "user3",
            "message": "让我们仔细考虑一下",
            "emotions": {"thoughtfulness": 0.7, "neutral": 0.4},
            "response_to": "user2"
        },
        {
            "timestamp": base_time + timedelta(seconds=90),
            "user_id": "user1",
            "message": "好的，我们可以讨论细节",
            "emotions": {"openness": 0.8, "cooperation": 0.7},
            "response_to": "user3"
        }
    ]

@pytest.fixture
def sample_network_data():
    """创建测试网络数据"""
    return {
        "nodes": [
            {"id": "user1", "name": "Alice", "role": "leader"},
            {"id": "user2", "name": "Bob", "role": "member"},
            {"id": "user3", "name": "Charlie", "role": "member"},
            {"id": "user4", "name": "Diana", "role": "observer"}
        ],
        "connections": [
            {"source": "user1", "target": "user2", "weight": 0.8, "type": "collaboration"},
            {"source": "user1", "target": "user3", "weight": 0.6, "type": "mentoring"},
            {"source": "user2", "target": "user3", "weight": 0.9, "type": "friendship"},
            {"source": "user3", "target": "user4", "weight": 0.4, "type": "acquaintance"}
        ]
    }

@pytest.fixture
def sample_group_metrics():
    """创建测试群体指标"""
    return {
        "cohesion_indicators": {
            "shared_goals": 0.8,
            "mutual_support": 0.7,
            "communication_frequency": 0.9,
            "conflict_resolution": 0.6
        },
        "participation_data": {
            "user1": {"messages": 15, "influence_score": 0.8},
            "user2": {"messages": 10, "influence_score": 0.6},
            "user3": {"messages": 12, "influence_score": 0.7},
            "user4": {"messages": 3, "influence_score": 0.2}
        },
        "interaction_patterns": {
            "response_time_avg": 45.5,
            "conversation_threads": 3,
            "active_participants": 4
        }
    }

class TestSocialAnalyticsTools:
    """社交分析工具基础功能测试"""
    
    def test_initialization(self, analytics_tools):
        """测试初始化"""
        assert analytics_tools is not None
        assert hasattr(analytics_tools, 'emotion_flow_history')
        assert hasattr(analytics_tools, 'network_analysis_cache')
        assert hasattr(analytics_tools, 'influence_patterns')
        assert analytics_tools.cache_ttl == 300  # 5分钟
    
    def test_clear_cache(self, analytics_tools):
        """测试缓存清理"""
        # 添加一些测试缓存数据
        analytics_tools.network_analysis_cache["test_key"] = {
            "data": "test_data",
            "timestamp": utc_now()
        }
        
        analytics_tools.clear_cache()
        
        assert len(analytics_tools.network_analysis_cache) == 0
        assert len(analytics_tools.influence_patterns) == 0
    
    def test_get_analytics_summary(self, analytics_tools):
        """测试分析摘要获取"""
        summary = analytics_tools.get_analytics_summary()
        
        assert isinstance(summary, dict)
        assert "emotion_flows_tracked" in summary
        assert "network_analyses_cached" in summary
        assert "influence_patterns_identified" in summary
        assert "cache_hit_rate" in summary

class TestEmotionFlowAnalysis:
    """情感流分析测试"""
    
    @pytest.mark.asyncio
    async def test_analyze_emotion_flow_basic(self, analytics_tools, sample_conversation_data):
        """测试基础情感流分析"""
        flow = await analytics_tools.analyze_emotion_flow(
            "test_session", sample_conversation_data
        )
        
        assert isinstance(flow, EmotionFlow)
        assert flow.session_id == "test_session"
        assert len(flow.timeline) > 0
        assert flow.duration_minutes > 0
        assert isinstance(flow.dominant_emotions, list)
        assert isinstance(flow.emotion_transitions, list)
        assert 0.0 <= flow.overall_sentiment <= 1.0
        assert isinstance(flow.peak_moments, list)
        assert isinstance(flow.valley_moments, list)
    
    @pytest.mark.asyncio
    async def test_analyze_emotion_flow_empty_data(self, analytics_tools):
        """测试空数据情感流分析"""
        flow = await analytics_tools.analyze_emotion_flow("empty_session", [])
        
        assert flow.session_id == "empty_session"
        assert flow.duration_minutes == 0
        assert len(flow.dominant_emotions) == 0
        assert len(flow.emotion_transitions) == 0
        assert flow.overall_sentiment == 0.5  # 中性
    
    @pytest.mark.asyncio
    async def test_analyze_emotion_flow_single_message(self, analytics_tools):
        """测试单条消息情感流分析"""
        single_message = [{
            "timestamp": utc_now(),
            "user_id": "user1",
            "message": "测试消息",
            "emotions": {"happiness": 0.8},
            "response_to": None
        }]
        
        flow = await analytics_tools.analyze_emotion_flow("single_session", single_message)
        
        assert len(flow.timeline) == 1
        assert len(flow.emotion_transitions) == 0  # 单条消息没有转换
    
    @pytest.mark.asyncio
    async def test_analyze_emotion_flow_time_window(self, analytics_tools, sample_conversation_data):
        """测试时间窗口情感流分析"""
        # 测试短时间窗口
        flow_short = await analytics_tools.analyze_emotion_flow(
            "test_session", sample_conversation_data, time_window=60
        )
        
        # 测试长时间窗口
        flow_long = await analytics_tools.analyze_emotion_flow(
            "test_session", sample_conversation_data, time_window=600
        )
        
        assert isinstance(flow_short, EmotionFlow)
        assert isinstance(flow_long, EmotionFlow)
    
    @pytest.mark.asyncio
    async def test_emotion_flow_caching(self, analytics_tools, sample_conversation_data):
        """测试情感流缓存"""
        session_id = "cache_test_session"
        
        # 第一次分析
        flow1 = await analytics_tools.analyze_emotion_flow(session_id, sample_conversation_data)
        
        # 第二次分析应该使用缓存
        flow2 = await analytics_tools.analyze_emotion_flow(session_id, sample_conversation_data)
        
        assert flow1.session_id == flow2.session_id
        assert session_id in analytics_tools.emotion_flow_history

class TestSocialNetworkAnalysis:
    """社交网络分析测试"""
    
    @pytest.mark.asyncio
    async def test_analyze_social_network_basic(self, analytics_tools, sample_network_data):
        """测试基础社交网络分析"""
        analysis = await analytics_tools.analyze_social_network(
            "test_network", sample_network_data
        )
        
        assert isinstance(analysis, SocialNetworkAnalysis)
        assert analysis.network_id == "test_network"
        assert analysis.total_nodes > 0
        assert analysis.total_connections > 0
        assert 0.0 <= analysis.network_density <= 1.0
        assert len(analysis.central_nodes) > 0
        assert len(analysis.communities) > 0
        assert isinstance(analysis.influence_scores, dict)
    
    @pytest.mark.asyncio
    async def test_analyze_social_network_empty(self, analytics_tools):
        """测试空网络分析"""
        empty_network = {"nodes": [], "connections": []}
        
        analysis = await analytics_tools.analyze_social_network("empty_network", empty_network)
        
        assert analysis.total_nodes == 0
        assert analysis.total_connections == 0
        assert analysis.network_density == 0.0
        assert len(analysis.central_nodes) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_social_network_single_node(self, analytics_tools):
        """测试单节点网络分析"""
        single_node_network = {
            "nodes": [{"id": "user1", "name": "Alice"}],
            "connections": []
        }
        
        analysis = await analytics_tools.analyze_social_network("single_node", single_node_network)
        
        assert analysis.total_nodes == 1
        assert analysis.total_connections == 0
        assert analysis.network_density == 0.0
    
    @pytest.mark.asyncio
    async def test_network_analysis_caching(self, analytics_tools, sample_network_data):
        """测试网络分析缓存"""
        network_id = "cache_test_network"
        
        # 第一次分析
        analysis1 = await analytics_tools.analyze_social_network(network_id, sample_network_data)
        
        # 验证缓存存在
        assert network_id in analytics_tools.network_analysis_cache
        
        # 第二次分析应该使用缓存
        analysis2 = await analytics_tools.analyze_social_network(network_id, sample_network_data)
        
        assert analysis1.network_id == analysis2.network_id
    
    @pytest.mark.asyncio
    async def test_network_analysis_cache_expiry(self, analytics_tools, sample_network_data):
        """测试网络分析缓存过期"""
        analytics_tools.cache_ttl = 1  # 1秒过期
        network_id = "expiry_test_network"
        
        # 第一次分析
        await analytics_tools.analyze_social_network(network_id, sample_network_data)
        
        # 等待缓存过期
        await asyncio.sleep(2)
        
        # 第二次分析应该重新计算
        analysis = await analytics_tools.analyze_social_network(network_id, sample_network_data)
        assert isinstance(analysis, SocialNetworkAnalysis)

class TestInfluencePatternAnalysis:
    """影响力模式分析测试"""
    
    @pytest.mark.asyncio
    async def test_identify_influence_patterns_basic(self, analytics_tools, sample_conversation_data):
        """测试基础影响力模式识别"""
        patterns = await analytics_tools.identify_influence_patterns(
            "test_session", sample_conversation_data
        )
        
        assert isinstance(patterns, list)
        assert len(patterns) > 0
        
        for pattern in patterns:
            assert isinstance(pattern, InfluencePattern)
            assert pattern.session_id == "test_session"
            assert pattern.influencer_id is not None
            assert 0.0 <= pattern.influence_strength <= 1.0
            assert pattern.pattern_type is not None
            assert isinstance(pattern.influenced_users, list)
            assert isinstance(pattern.evidence, list)
    
    @pytest.mark.asyncio
    async def test_identify_influence_patterns_no_interactions(self, analytics_tools):
        """测试无交互数据的影响力模式"""
        no_response_data = [
            {
                "timestamp": utc_now(),
                "user_id": "user1",
                "message": "独立消息1",
                "emotions": {"neutral": 1.0},
                "response_to": None
            },
            {
                "timestamp": utc_now(),
                "user_id": "user2",
                "message": "独立消息2", 
                "emotions": {"neutral": 1.0},
                "response_to": None
            }
        ]
        
        patterns = await analytics_tools.identify_influence_patterns(
            "no_interaction_session", no_response_data
        )
        
        # 可能没有影响力模式，或者模式较少
        assert isinstance(patterns, list)
    
    @pytest.mark.asyncio
    async def test_influence_patterns_caching(self, analytics_tools, sample_conversation_data):
        """测试影响力模式缓存"""
        session_id = "influence_cache_test"
        
        # 第一次分析
        patterns1 = await analytics_tools.identify_influence_patterns(session_id, sample_conversation_data)
        
        # 验证缓存
        assert session_id in analytics_tools.influence_patterns
        
        # 第二次分析应该使用缓存
        patterns2 = await analytics_tools.identify_influence_patterns(session_id, sample_conversation_data)
        
        assert len(patterns1) == len(patterns2)

class TestGroupCohesionAnalysis:
    """群体凝聚力分析测试"""
    
    @pytest.mark.asyncio
    async def test_calculate_group_cohesion_basic(self, analytics_tools, sample_group_metrics):
        """测试基础群体凝聚力计算"""
        cohesion = await analytics_tools.calculate_group_cohesion(
            "test_group", sample_group_metrics
        )
        
        assert isinstance(cohesion, GroupCohesionMetrics)
        assert cohesion.group_id == "test_group"
        assert 0.0 <= cohesion.overall_cohesion_score <= 1.0
        assert 0.0 <= cohesion.communication_cohesion <= 1.0
        assert 0.0 <= cohesion.emotional_cohesion <= 1.0
        assert 0.0 <= cohesion.task_cohesion <= 1.0
        assert isinstance(cohesion.cohesion_factors, dict)
        assert isinstance(cohesion.risk_factors, list)
        assert isinstance(cohesion.improvement_suggestions, list)
    
    @pytest.mark.asyncio
    async def test_calculate_group_cohesion_minimal_data(self, analytics_tools):
        """测试最小数据群体凝聚力计算"""
        minimal_metrics = {
            "cohesion_indicators": {"shared_goals": 0.5},
            "participation_data": {"user1": {"messages": 1, "influence_score": 0.5}},
            "interaction_patterns": {"response_time_avg": 60}
        }
        
        cohesion = await analytics_tools.calculate_group_cohesion("minimal_group", minimal_metrics)
        
        assert isinstance(cohesion, GroupCohesionMetrics)
        assert cohesion.overall_cohesion_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_calculate_group_cohesion_high_cohesion(self, analytics_tools):
        """测试高凝聚力群体"""
        high_cohesion_metrics = {
            "cohesion_indicators": {
                "shared_goals": 0.95,
                "mutual_support": 0.9,
                "communication_frequency": 0.95,
                "conflict_resolution": 0.85
            },
            "participation_data": {
                "user1": {"messages": 20, "influence_score": 0.8},
                "user2": {"messages": 18, "influence_score": 0.7},
                "user3": {"messages": 22, "influence_score": 0.9}
            },
            "interaction_patterns": {
                "response_time_avg": 15.0,
                "conversation_threads": 5,
                "active_participants": 3
            }
        }
        
        cohesion = await analytics_tools.calculate_group_cohesion("high_cohesion_group", high_cohesion_metrics)
        
        assert cohesion.overall_cohesion_score > 0.8
        assert cohesion.communication_cohesion > 0.8
    
    @pytest.mark.asyncio
    async def test_calculate_group_cohesion_low_cohesion(self, analytics_tools):
        """测试低凝聚力群体"""
        low_cohesion_metrics = {
            "cohesion_indicators": {
                "shared_goals": 0.2,
                "mutual_support": 0.1,
                "communication_frequency": 0.3,
                "conflict_resolution": 0.2
            },
            "participation_data": {
                "user1": {"messages": 1, "influence_score": 0.9},
                "user2": {"messages": 20, "influence_score": 0.1},
                "user3": {"messages": 2, "influence_score": 0.1}
            },
            "interaction_patterns": {
                "response_time_avg": 300.0,
                "conversation_threads": 1,
                "active_participants": 1
            }
        }
        
        cohesion = await analytics_tools.calculate_group_cohesion("low_cohesion_group", low_cohesion_metrics)
        
        assert cohesion.overall_cohesion_score < 0.5
        assert len(cohesion.risk_factors) > 0
        assert len(cohesion.improvement_suggestions) > 0

class TestConversationQualityAnalysis:
    """对话质量分析测试"""
    
    @pytest.mark.asyncio
    async def test_assess_conversation_quality_basic(self, analytics_tools, sample_conversation_data):
        """测试基础对话质量评估"""
        quality = await analytics_tools.assess_conversation_quality(
            "test_conversation", sample_conversation_data
        )
        
        assert isinstance(quality, ConversationQuality)
        assert quality.conversation_id == "test_conversation"
        assert 0.0 <= quality.overall_quality_score <= 1.0
        assert 0.0 <= quality.engagement_score <= 1.0
        assert 0.0 <= quality.coherence_score <= 1.0
        assert 0.0 <= quality.inclusivity_score <= 1.0
        assert 0.0 <= quality.constructiveness_score <= 1.0
        assert isinstance(quality.quality_indicators, dict)
        assert isinstance(quality.improvement_areas, list)
    
    @pytest.mark.asyncio
    async def test_assess_conversation_quality_monologue(self, analytics_tools):
        """测试独白式对话质量"""
        monologue_data = [
            {
                "timestamp": utc_now(),
                "user_id": "user1",
                "message": f"消息{i}",
                "emotions": {"neutral": 0.8},
                "response_to": None
            }
            for i in range(5)
        ]
        
        quality = await analytics_tools.assess_conversation_quality("monologue", monologue_data)
        
        # 独白式对话的包容性和互动性应该较低
        assert quality.inclusivity_score < 0.7
        assert "low_participant_diversity" in [area["issue"] for area in quality.improvement_areas]
    
    @pytest.mark.asyncio
    async def test_assess_conversation_quality_high_quality(self, analytics_tools):
        """测试高质量对话"""
        high_quality_data = [
            {
                "timestamp": utc_now(),
                "user_id": f"user{i%3+1}",
                "message": f"深思熟虑的回应{i}",
                "emotions": {"thoughtfulness": 0.8, "engagement": 0.7},
                "response_to": f"user{(i-1)%3+1}" if i > 0 else None
            }
            for i in range(10)
        ]
        
        quality = await analytics_tools.assess_conversation_quality("high_quality", high_quality_data)
        
        assert quality.overall_quality_score > 0.6
        assert quality.engagement_score > 0.6
        assert quality.coherence_score > 0.6

class TestSocialDynamicsInsights:
    """社交动态洞察测试"""
    
    @pytest.mark.asyncio
    async def test_generate_social_insights_comprehensive(
        self, analytics_tools, sample_conversation_data, sample_network_data, sample_group_metrics
    ):
        """测试综合社交洞察生成"""
        insights = await analytics_tools.generate_social_insights(
            "comprehensive_session",
            {
                "conversation_data": sample_conversation_data,
                "network_data": sample_network_data,
                "group_metrics": sample_group_metrics
            }
        )
        
        assert isinstance(insights, SocialDynamicsInsight)
        assert insights.session_id == "comprehensive_session"
        assert isinstance(insights.key_findings, list)
        assert len(insights.key_findings) > 0
        assert isinstance(insights.behavioral_patterns, dict)
        assert isinstance(insights.relationship_dynamics, dict)
        assert isinstance(insights.communication_trends, dict)
        assert isinstance(insights.recommendations, list)
        assert len(insights.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_generate_social_insights_partial_data(self, analytics_tools, sample_conversation_data):
        """测试部分数据社交洞察"""
        insights = await analytics_tools.generate_social_insights(
            "partial_session",
            {"conversation_data": sample_conversation_data}
        )
        
        assert isinstance(insights, SocialDynamicsInsight)
        assert len(insights.key_findings) > 0
        # 部分数据仍应提供有用洞察
    
    @pytest.mark.asyncio
    async def test_generate_social_insights_empty_data(self, analytics_tools):
        """测试空数据社交洞察"""
        insights = await analytics_tools.generate_social_insights("empty_session", {})
        
        assert isinstance(insights, SocialDynamicsInsight)
        assert insights.session_id == "empty_session"
        # 即使没有数据也应该有基础结构

class TestDataProcessingUtilities:
    """数据处理工具测试"""
    
    def test_extract_emotion_timeline(self, analytics_tools, sample_conversation_data):
        """测试情感时间线提取"""
        timeline = analytics_tools._extract_emotion_timeline(sample_conversation_data)
        
        assert isinstance(timeline, list)
        assert len(timeline) == len(sample_conversation_data)
        
        for point in timeline:
            assert "timestamp" in point
            assert "emotions" in point
            assert "user_id" in point
    
    def test_calculate_emotion_transitions(self, analytics_tools, sample_conversation_data):
        """测试情感转换计算"""
        timeline = analytics_tools._extract_emotion_timeline(sample_conversation_data)
        transitions = analytics_tools._calculate_emotion_transitions(timeline)
        
        assert isinstance(transitions, list)
        # 转换数量应该比时间点少1
        assert len(transitions) == len(timeline) - 1
        
        for transition in transitions:
            assert "from_emotion" in transition
            assert "to_emotion" in transition
            assert "transition_strength" in transition
            assert 0.0 <= transition["transition_strength"] <= 1.0
    
    def test_identify_peak_valley_moments(self, analytics_tools, sample_conversation_data):
        """测试峰谷时刻识别"""
        timeline = analytics_tools._extract_emotion_timeline(sample_conversation_data)
        peaks, valleys = analytics_tools._identify_peak_valley_moments(timeline)
        
        assert isinstance(peaks, list)
        assert isinstance(valleys, list)
        
        # 验证峰谷时刻的数据结构
        for peak in peaks:
            assert "timestamp" in peak
            assert "emotion_intensity" in peak
            assert "dominant_emotion" in peak
        
        for valley in valleys:
            assert "timestamp" in valley
            assert "emotion_intensity" in valley
            assert "dominant_emotion" in valley
    
    def test_calculate_network_centrality(self, analytics_tools, sample_network_data):
        """测试网络中心性计算"""
        centrality = analytics_tools._calculate_network_centrality(sample_network_data)
        
        assert isinstance(centrality, dict)
        assert len(centrality) == len(sample_network_data["nodes"])
        
        for node_id, centrality_score in centrality.items():
            assert 0.0 <= centrality_score <= 1.0
    
    def test_detect_communities(self, analytics_tools, sample_network_data):
        """测试社区检测"""
        communities = analytics_tools._detect_communities(sample_network_data)
        
        assert isinstance(communities, list)
        assert len(communities) > 0
        
        for community in communities:
            assert "members" in community
            assert "cohesion_score" in community
            assert isinstance(community["members"], list)
            assert 0.0 <= community["cohesion_score"] <= 1.0

class TestErrorHandlingAndEdgeCases:
    """错误处理和边界条件测试"""
    
    @pytest.mark.asyncio
    async def test_invalid_emotion_data(self, analytics_tools):
        """测试无效情感数据处理"""
        invalid_data = [
            {
                "timestamp": "invalid_timestamp",
                "user_id": None,
                "message": "",
                "emotions": {"invalid_emotion": float('nan')},
                "response_to": "nonexistent_user"
            }
        ]
        
        # 应该能够处理无效数据而不崩溃
        flow = await analytics_tools.analyze_emotion_flow("invalid_session", invalid_data)
        assert isinstance(flow, EmotionFlow)
    
    @pytest.mark.asyncio
    async def test_circular_network_references(self, analytics_tools):
        """测试循环网络引用"""
        circular_network = {
            "nodes": [
                {"id": "user1"}, {"id": "user2"}, {"id": "user3"}
            ],
            "connections": [
                {"source": "user1", "target": "user2", "weight": 1.0},
                {"source": "user2", "target": "user3", "weight": 1.0},
                {"source": "user3", "target": "user1", "weight": 1.0}
            ]
        }
        
        analysis = await analytics_tools.analyze_social_network("circular", circular_network)
        assert isinstance(analysis, SocialNetworkAnalysis)
    
    @pytest.mark.asyncio
    async def test_extremely_large_dataset(self, analytics_tools):
        """测试极大数据集处理"""
        # 创建大量数据
        large_conversation_data = []
        base_time = utc_now()
        
        for i in range(1000):
            large_conversation_data.append({
                "timestamp": base_time + timedelta(seconds=i),
                "user_id": f"user{i%10}",
                "message": f"消息{i}",
                "emotions": {"neutral": 0.5, "test": i/1000.0},
                "response_to": f"user{(i-1)%10}" if i > 0 else None
            })
        
        # 应该能够处理大数据集
        flow = await analytics_tools.analyze_emotion_flow("large_session", large_conversation_data)
        assert isinstance(flow, EmotionFlow)
        assert flow.duration_minutes > 0
    
    @pytest.mark.asyncio
    async def test_missing_required_fields(self, analytics_tools):
        """测试缺少必要字段"""
        incomplete_data = [
            {
                "user_id": "user1",
                "message": "缺少时间戳和情感"
            },
            {
                "timestamp": utc_now(),
                "emotions": {"happiness": 0.8}
                # 缺少user_id
            }
        ]
        
        flow = await analytics_tools.analyze_emotion_flow("incomplete_session", incomplete_data)
        assert isinstance(flow, EmotionFlow)

class TestPerformanceAndOptimization:
    """性能和优化测试"""
    
    @pytest.mark.asyncio
    async def test_analysis_performance(self, analytics_tools):
        """测试分析性能"""
        import time
        
        # 创建中等规模数据集
        conversation_data = []
        base_time = utc_now()
        
        for i in range(100):
            conversation_data.append({
                "timestamp": base_time + timedelta(seconds=i*10),
                "user_id": f"user{i%5}",
                "message": f"性能测试消息{i}",
                "emotions": {"test_emotion": i/100.0, "neutral": 0.5},
                "response_to": f"user{(i-1)%5}" if i > 0 else None
            })
        
        start_time = time.time()
        
        # 执行多个分析任务
        flow = await analytics_tools.analyze_emotion_flow("perf_test", conversation_data)
        patterns = await analytics_tools.identify_influence_patterns("perf_test", conversation_data)
        quality = await analytics_tools.assess_conversation_quality("perf_test", conversation_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # 验证性能
        assert processing_time < 30.0  # 30秒内完成
        assert isinstance(flow, EmotionFlow)
        assert isinstance(patterns, list)
        assert isinstance(quality, ConversationQuality)
    
    def test_memory_usage_optimization(self, analytics_tools):
        """测试内存使用优化"""
        import sys
        
        initial_size = sys.getsizeof(analytics_tools.emotion_flow_history)
        
        # 添加大量历史数据
        for i in range(100):
            analytics_tools.emotion_flow_history[f"session_{i}"] = {
                "timestamp": utc_now(),
                "data": f"test_data_{i}"
            }
        
        # 清理缓存
        analytics_tools.clear_cache()
        
        final_size = sys.getsizeof(analytics_tools.emotion_flow_history)
        
        # 验证清理有效
        assert final_size <= initial_size

class TestIntegrationScenarios:
    """集成场景测试"""
    
    @pytest.mark.asyncio
    async def test_complete_analytics_workflow(
        self, analytics_tools, sample_conversation_data, sample_network_data, sample_group_metrics
    ):
        """测试完整分析工作流"""
        session_id = "complete_workflow_test"
        
        # 1. 分析情感流
        emotion_flow = await analytics_tools.analyze_emotion_flow(session_id, sample_conversation_data)
        assert isinstance(emotion_flow, EmotionFlow)
        
        # 2. 分析社交网络
        network_analysis = await analytics_tools.analyze_social_network(session_id, sample_network_data)
        assert isinstance(network_analysis, SocialNetworkAnalysis)
        
        # 3. 识别影响力模式
        influence_patterns = await analytics_tools.identify_influence_patterns(session_id, sample_conversation_data)
        assert isinstance(influence_patterns, list)
        
        # 4. 计算群体凝聚力
        cohesion_metrics = await analytics_tools.calculate_group_cohesion(session_id, sample_group_metrics)
        assert isinstance(cohesion_metrics, GroupCohesionMetrics)
        
        # 5. 评估对话质量
        conversation_quality = await analytics_tools.assess_conversation_quality(session_id, sample_conversation_data)
        assert isinstance(conversation_quality, ConversationQuality)
        
        # 6. 生成综合洞察
        comprehensive_data = {
            "conversation_data": sample_conversation_data,
            "network_data": sample_network_data,
            "group_metrics": sample_group_metrics
        }
        insights = await analytics_tools.generate_social_insights(session_id, comprehensive_data)
        assert isinstance(insights, SocialDynamicsInsight)
        
        # 验证所有分析结果相关联
        assert emotion_flow.session_id == session_id
        assert network_analysis.network_id == session_id
        assert all(pattern.session_id == session_id for pattern in influence_patterns)
        assert cohesion_metrics.group_id == session_id
        assert conversation_quality.conversation_id == session_id
        assert insights.session_id == session_id

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
