"""
Task 8社交情感理解系统集成完整单元测试套件
测试SocialEmotionSystem的系统集成和统一接口功能
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock

from ai.emotion_modeling.social_emotion_system import (
    SocialEmotionSystem,
    SystemStatus,
    ProcessingResult,
    SystemMetrics,
    IntegrationHealth
)
from ai.emotion_modeling.models import EmotionVector, SocialContext
from ai.emotion_modeling.social_context_adapter import SocialEnvironment
from ai.emotion_modeling.cultural_context_analyzer import CulturalProfile


@pytest.fixture
def social_emotion_system():
    """创建社交情感理解系统实例"""
    return SocialEmotionSystem()


@pytest.fixture
def sample_interaction_context():
    """创建测试交互上下文"""
    return {
        "session_id": "test_session_001",
        "participants": [
            {"user_id": "user1", "role": "facilitator", "name": "Alice"},
            {"user_id": "user2", "role": "participant", "name": "Bob"},
            {"user_id": "user3", "role": "participant", "name": "Charlie"}
        ],
        "conversation_data": [
            {
                "timestamp": datetime.now() - timedelta(minutes=5),
                "user_id": "user1",
                "message": "大家好，今天我们讨论新项目计划",
                "emotions": {"confidence": 0.8, "leadership": 0.7}
            },
            {
                "timestamp": datetime.now() - timedelta(minutes=4),
                "user_id": "user2",
                "message": "我觉得这个方案很有挑战性",
                "emotions": {"concern": 0.6, "interest": 0.7}
            },
            {
                "timestamp": datetime.now() - timedelta(minutes=3),
                "user_id": "user3", 
                "message": "我同意，但我们可以尝试",
                "emotions": {"optimism": 0.7, "caution": 0.4}
            }
        ],
        "social_environment": {
            "environment_type": "business_meeting",
            "formality_level": 0.7,
            "participants_count": 3,
            "time_pressure": 0.5,
            "physical_setup": "virtual",
            "cultural_context": "mixed"
        },
        "cultural_indicators": {
            "user1": {"language": "zh", "region": "CN"},
            "user2": {"language": "en", "region": "US"},
            "user3": {"language": "en", "region": "US"}
        }
    }


@pytest.fixture
def sample_emotion_vectors():
    """创建测试情感向量"""
    return {
        "user1": EmotionVector(
            emotions={"confidence": 0.8, "leadership": 0.7, "enthusiasm": 0.6},
            intensity=0.7,
            confidence=0.9,
            context={"role": "facilitator"}
        ),
        "user2": EmotionVector(
            emotions={"concern": 0.6, "interest": 0.7, "analytical": 0.5},
            intensity=0.6,
            confidence=0.8,
            context={"role": "participant"}
        ),
        "user3": EmotionVector(
            emotions={"optimism": 0.7, "caution": 0.4, "collaborative": 0.8},
            intensity=0.6,
            confidence=0.8,
            context={"role": "participant"}
        )
    }


class TestSocialEmotionSystem:
    """社交情感理解系统基础功能测试"""
    
    def test_initialization(self, social_emotion_system):
        """测试系统初始化"""
        assert social_emotion_system is not None
        assert hasattr(social_emotion_system, 'social_context_adapter')
        assert hasattr(social_emotion_system, 'cultural_analyzer')
        assert hasattr(social_emotion_system, 'intelligence_engine')
        assert hasattr(social_emotion_system, 'analytics_tools')
        assert hasattr(social_emotion_system, 'privacy_guard')
        assert social_emotion_system.system_status == SystemStatus.INITIALIZING
    
    @pytest.mark.asyncio
    async def test_start_system(self, social_emotion_system):
        """测试系统启动"""
        await social_emotion_system.start_system()
        
        assert social_emotion_system.system_status == SystemStatus.RUNNING
        assert social_emotion_system.start_time is not None
    
    @pytest.mark.asyncio
    async def test_stop_system(self, social_emotion_system):
        """测试系统停止"""
        await social_emotion_system.start_system()
        await social_emotion_system.stop_system()
        
        assert social_emotion_system.system_status == SystemStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_restart_system(self, social_emotion_system):
        """测试系统重启"""
        await social_emotion_system.start_system()
        original_start_time = social_emotion_system.start_time
        
        await asyncio.sleep(0.1)  # 确保时间差
        await social_emotion_system.restart_system()
        
        assert social_emotion_system.system_status == SystemStatus.RUNNING
        assert social_emotion_system.start_time > original_start_time
    
    @pytest.mark.asyncio
    async def test_get_system_status(self, social_emotion_system):
        """测试获取系统状态"""
        status = await social_emotion_system.get_system_status()
        
        assert isinstance(status, dict)
        assert "status" in status
        assert "uptime" in status
        assert "components" in status
        assert "performance" in status
        assert "last_activity" in status


class TestComprehensiveProcessing:
    """综合处理功能测试"""
    
    @pytest.mark.asyncio
    async def test_process_social_interaction_basic(
        self, social_emotion_system, sample_interaction_context
    ):
        """测试基础社交交互处理"""
        await social_emotion_system.start_system()
        
        result = await social_emotion_system.process_social_interaction(sample_interaction_context)
        
        assert isinstance(result, ProcessingResult)
        assert result.session_id == sample_interaction_context["session_id"]
        assert result.processing_success is True
        assert result.social_adaptation is not None
        assert result.cultural_adaptation is not None
        assert result.intelligence_decisions is not None
        assert len(result.intelligence_decisions) > 0
        assert result.analytics_insights is not None
        assert result.privacy_assessment is not None
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_process_social_interaction_comprehensive_flow(
        self, social_emotion_system, sample_interaction_context
    ):
        """测试综合流程处理"""
        await social_emotion_system.start_system()
        
        # 添加更复杂的上下文
        complex_context = sample_interaction_context.copy()
        complex_context["group_dynamics"] = {
            "cohesion_score": 0.7,
            "energy_level": 0.6,
            "participation_balance": 0.8,
            "relationship_tensions": 0.3
        }
        complex_context["historical_context"] = [
            {"event": "previous_meeting", "outcome": "successful"},
            {"event": "team_conflict", "outcome": "resolved"}
        ]
        
        result = await social_emotion_system.process_social_interaction(complex_context)
        
        # 验证所有组件都参与了处理
        assert result.social_adaptation is not None
        assert result.cultural_adaptation is not None
        assert len(result.intelligence_decisions) > 0
        assert result.analytics_insights is not None
        assert result.privacy_assessment is not None
        
        # 验证处理质量
        assert result.processing_success is True
        assert result.confidence_score > 0.0
        assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_process_social_interaction_with_privacy_concerns(
        self, social_emotion_system, sample_interaction_context
    ):
        """测试包含隐私关注的交互处理"""
        await social_emotion_system.start_system()
        
        # 添加敏感信息
        sensitive_context = sample_interaction_context.copy()
        sensitive_context["conversation_data"][0]["message"] = "我正在处理个人健康问题，感到很焦虑"
        sensitive_context["conversation_data"][0]["emotions"] = {
            "anxiety": 0.9, "vulnerability": 0.8, "depression": 0.6
        }
        
        result = await social_emotion_system.process_social_interaction(sensitive_context)
        
        # 验证隐私保护措施
        assert result.privacy_assessment is not None
        privacy_violations = getattr(result.privacy_assessment, 'privacy_violations', [])
        ethical_violations = getattr(result.privacy_assessment, 'ethical_violations', [])
        
        # 系统应该识别并处理敏感内容
        if privacy_violations or ethical_violations:
            assert result.privacy_concerns_detected is True
        
        # 确保处理仍然成功，但有适当的保护措施
        assert result.processing_success is True
    
    @pytest.mark.asyncio
    async def test_process_social_interaction_error_handling(
        self, social_emotion_system
    ):
        """测试交互处理错误处理"""
        await social_emotion_system.start_system()
        
        # 测试无效输入
        invalid_context = {
            "session_id": None,
            "participants": [],
            "conversation_data": None
        }
        
        result = await social_emotion_system.process_social_interaction(invalid_context)
        
        # 应该优雅地处理错误
        assert isinstance(result, ProcessingResult)
        assert result.processing_success is False
        assert len(result.error_messages) > 0
    
    @pytest.mark.asyncio
    async def test_process_social_interaction_system_not_started(
        self, social_emotion_system, sample_interaction_context
    ):
        """测试系统未启动状态下的处理"""
        # 不启动系统直接处理
        result = await social_emotion_system.process_social_interaction(sample_interaction_context)
        
        assert result.processing_success is False
        assert "system not started" in " ".join(result.error_messages).lower()


class TestAnalyticsAndInsights:
    """分析和洞察功能测试"""
    
    @pytest.mark.asyncio
    async def test_get_session_analytics(self, social_emotion_system, sample_interaction_context):
        """测试会话分析获取"""
        await social_emotion_system.start_system()
        
        # 先处理一个交互
        await social_emotion_system.process_social_interaction(sample_interaction_context)
        
        # 获取分析结果
        analytics = await social_emotion_system.get_session_analytics(
            sample_interaction_context["session_id"]
        )
        
        assert isinstance(analytics, dict)
        assert "emotion_flow" in analytics
        assert "social_network" in analytics
        assert "influence_patterns" in analytics
        assert "group_cohesion" in analytics
        assert "conversation_quality" in analytics
    
    @pytest.mark.asyncio
    async def test_get_session_analytics_nonexistent_session(self, social_emotion_system):
        """测试不存在的会话分析"""
        await social_emotion_system.start_system()
        
        analytics = await social_emotion_system.get_session_analytics("nonexistent_session")
        
        # 应该返回空结果或错误信息
        assert analytics is None or len(analytics) == 0
    
    @pytest.mark.asyncio
    async def test_generate_comprehensive_insights(
        self, social_emotion_system, sample_interaction_context
    ):
        """测试综合洞察生成"""
        await social_emotion_system.start_system()
        
        # 处理多个交互以生成更丰富的洞察
        for i in range(3):
            context_copy = sample_interaction_context.copy()
            context_copy["session_id"] = f"insight_session_{i}"
            await social_emotion_system.process_social_interaction(context_copy)
        
        # 生成综合洞察
        insights = await social_emotion_system.generate_comprehensive_insights(
            session_ids=[f"insight_session_{i}" for i in range(3)]
        )
        
        assert isinstance(insights, dict)
        assert "cross_session_patterns" in insights
        assert "behavioral_trends" in insights
        assert "relationship_evolution" in insights
        assert "cultural_dynamics" in insights
        assert "system_recommendations" in insights


class TestSystemHealthAndMetrics:
    """系统健康和指标测试"""
    
    @pytest.mark.asyncio
    async def test_get_system_metrics(self, social_emotion_system):
        """测试系统指标获取"""
        await social_emotion_system.start_system()
        
        metrics = await social_emotion_system.get_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.total_sessions_processed >= 0
        assert metrics.total_interactions_processed >= 0
        assert metrics.average_processing_time >= 0.0
        assert 0.0 <= metrics.system_performance_score <= 1.0
        assert isinstance(metrics.component_health, dict)
        assert isinstance(metrics.error_rates, dict)
        assert isinstance(metrics.resource_usage, dict)
    
    @pytest.mark.asyncio
    async def test_check_integration_health(self, social_emotion_system):
        """测试集成健康检查"""
        await social_emotion_system.start_system()
        
        health = await social_emotion_system.check_integration_health()
        
        assert isinstance(health, IntegrationHealth)
        assert health.overall_health in ["HEALTHY", "WARNING", "CRITICAL", "UNKNOWN"]
        assert isinstance(health.component_status, dict)
        assert isinstance(health.connectivity_tests, dict)
        assert isinstance(health.performance_tests, dict)
        assert isinstance(health.data_integrity_tests, dict)
        
        # 验证所有关键组件都被检查
        expected_components = [
            "social_context_adapter",
            "cultural_analyzer", 
            "intelligence_engine",
            "analytics_tools",
            "privacy_guard"
        ]
        for component in expected_components:
            assert component in health.component_status
    
    @pytest.mark.asyncio
    async def test_system_health_degradation_detection(self, social_emotion_system):
        """测试系统健康退化检测"""
        await social_emotion_system.start_system()
        
        # 模拟组件故障
        with patch.object(
            social_emotion_system.analytics_tools, 
            'analyze_emotion_flow', 
            side_effect=Exception("模拟分析工具故障")
        ):
            health = await social_emotion_system.check_integration_health()
            
            # 应该检测到健康问题
            assert health.overall_health in ["WARNING", "CRITICAL"]
            assert health.component_status["analytics_tools"] == "UNHEALTHY"


class TestConcurrencyAndScaling:
    """并发和扩展性测试"""
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self, social_emotion_system, sample_interaction_context):
        """测试并发处理"""
        await social_emotion_system.start_system()
        
        # 创建多个并发任务
        tasks = []
        for i in range(5):
            context_copy = sample_interaction_context.copy()
            context_copy["session_id"] = f"concurrent_session_{i}"
            tasks.append(social_emotion_system.process_social_interaction(context_copy))
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证所有任务都成功完成
        successful_results = [r for r in results if isinstance(r, ProcessingResult) and r.processing_success]
        assert len(successful_results) == 5
    
    @pytest.mark.asyncio
    async def test_high_load_processing(self, social_emotion_system):
        """测试高负载处理"""
        await social_emotion_system.start_system()
        
        # 创建大量简单交互进行压力测试
        high_load_context = {
            "session_id": "high_load_test",
            "participants": [{"user_id": f"user_{i}"} for i in range(20)],
            "conversation_data": [
                {
                    "timestamp": datetime.now(),
                    "user_id": f"user_{i%5}",
                    "message": f"压力测试消息 {i}",
                    "emotions": {"neutral": 0.8}
                }
                for i in range(100)
            ],
            "social_environment": {
                "environment_type": "large_conference",
                "participants_count": 20
            }
        }
        
        import time
        start_time = time.time()
        result = await social_emotion_system.process_social_interaction(high_load_context)
        end_time = time.time()
        
        # 验证高负载下的性能
        processing_time = end_time - start_time
        assert processing_time < 30.0  # 30秒内完成
        assert result.processing_success is True
    
    @pytest.mark.asyncio
    async def test_memory_usage_optimization(self, social_emotion_system):
        """测试内存使用优化"""
        await social_emotion_system.start_system()
        
        import sys
        initial_memory = sys.getsizeof(social_emotion_system)
        
        # 处理大量交互
        for i in range(20):
            simple_context = {
                "session_id": f"memory_test_{i}",
                "participants": [{"user_id": "user1"}],
                "conversation_data": [
                    {
                        "timestamp": datetime.now(),
                        "user_id": "user1",
                        "message": f"内存测试消息 {i}",
                        "emotions": {"neutral": 0.5}
                    }
                ]
            }
            await social_emotion_system.process_social_interaction(simple_context)
        
        final_memory = sys.getsizeof(social_emotion_system)
        memory_growth = final_memory - initial_memory
        
        # 验证内存使用合理
        assert memory_growth < initial_memory * 2  # 内存增长不超过初始大小的2倍


class TestDataIntegrity:
    """数据完整性测试"""
    
    @pytest.mark.asyncio
    async def test_data_consistency_across_components(
        self, social_emotion_system, sample_interaction_context
    ):
        """测试组件间数据一致性"""
        await social_emotion_system.start_system()
        
        result = await social_emotion_system.process_social_interaction(sample_interaction_context)
        
        # 验证不同组件处理的数据一致性
        session_id = sample_interaction_context["session_id"]
        
        # 检查会话ID在所有结果中的一致性
        assert result.session_id == session_id
        if result.social_adaptation:
            assert hasattr(result.social_adaptation, 'session_id') or session_id in str(result.social_adaptation)
        
        if result.analytics_insights:
            assert hasattr(result.analytics_insights, 'session_id') or session_id in str(result.analytics_insights)
    
    @pytest.mark.asyncio
    async def test_temporal_data_integrity(self, social_emotion_system):
        """测试时间数据完整性"""
        await social_emotion_system.start_system()
        
        # 创建带有特定时间序列的交互
        timeline_context = {
            "session_id": "timeline_test",
            "participants": [{"user_id": "user1"}, {"user_id": "user2"}],
            "conversation_data": []
        }
        
        base_time = datetime.now() - timedelta(minutes=10)
        for i in range(5):
            timeline_context["conversation_data"].append({
                "timestamp": base_time + timedelta(minutes=i*2),
                "user_id": f"user{i%2+1}",
                "message": f"时间序列消息 {i}",
                "emotions": {"progression": i/5.0}
            })
        
        result = await social_emotion_system.process_social_interaction(timeline_context)
        
        # 验证时间序列完整性
        assert result.processing_success is True
        if result.analytics_insights and hasattr(result.analytics_insights, 'timeline'):
            timeline = result.analytics_insights.timeline
            assert len(timeline) == 5
            # 验证时间顺序
            for i in range(1, len(timeline)):
                assert timeline[i]['timestamp'] > timeline[i-1]['timestamp']


class TestErrorRecoveryAndResilience:
    """错误恢复和弹性测试"""
    
    @pytest.mark.asyncio
    async def test_component_failure_recovery(self, social_emotion_system, sample_interaction_context):
        """测试组件故障恢复"""
        await social_emotion_system.start_system()
        
        # 模拟单个组件故障
        with patch.object(
            social_emotion_system.cultural_analyzer,
            'analyze_cultural_context',
            side_effect=Exception("文化分析器故障")
        ):
            result = await social_emotion_system.process_social_interaction(sample_interaction_context)
            
            # 系统应该部分降级工作，而不是完全失败
            assert isinstance(result, ProcessingResult)
            # 可能成功或部分成功，但应该有错误信息
            if not result.processing_success:
                assert len(result.error_messages) > 0
                assert "文化分析器" in " ".join(result.error_messages) or "cultural" in " ".join(result.error_messages).lower()
    
    @pytest.mark.asyncio
    async def test_cascading_failure_prevention(self, social_emotion_system, sample_interaction_context):
        """测试级联故障预防"""
        await social_emotion_system.start_system()
        
        # 模拟多个组件故障
        with patch.object(
            social_emotion_system.social_context_adapter,
            'adapt_for_social_context',
            side_effect=Exception("社交适配器故障")
        ), patch.object(
            social_emotion_system.analytics_tools,
            'analyze_emotion_flow', 
            side_effect=Exception("分析工具故障")
        ):
            result = await social_emotion_system.process_social_interaction(sample_interaction_context)
            
            # 即使多个组件故障，系统也应该优雅降级
            assert isinstance(result, ProcessingResult)
            assert len(result.error_messages) > 0
            
            # 隐私保护等关键组件应该仍然工作
            assert result.privacy_assessment is not None
    
    @pytest.mark.asyncio
    async def test_system_recovery_after_failure(self, social_emotion_system, sample_interaction_context):
        """测试故障后系统恢复"""
        await social_emotion_system.start_system()
        
        # 模拟系统故障
        social_emotion_system.system_status = SystemStatus.ERROR
        
        # 尝试恢复
        await social_emotion_system.restart_system()
        
        # 验证系统恢复
        assert social_emotion_system.system_status == SystemStatus.RUNNING
        
        # 验证恢复后的功能
        result = await social_emotion_system.process_social_interaction(sample_interaction_context)
        assert result.processing_success is True


class TestConfigurationAndCustomization:
    """配置和定制化测试"""
    
    @pytest.mark.asyncio
    async def test_custom_configuration(self, social_emotion_system):
        """测试自定义配置"""
        # 自定义配置
        custom_config = {
            "privacy_level": "strict",
            "cultural_sensitivity": "high",
            "analytics_depth": "comprehensive",
            "processing_timeout": 60,
            "enable_learning": True
        }
        
        await social_emotion_system.configure_system(custom_config)
        
        # 验证配置应用
        config = social_emotion_system.get_current_configuration()
        assert config["privacy_level"] == "strict"
        assert config["cultural_sensitivity"] == "high"
    
    @pytest.mark.asyncio
    async def test_component_level_configuration(self, social_emotion_system):
        """测试组件级配置"""
        component_configs = {
            "privacy_guard": {
                "compliance_checking": True,
                "audit_logging": True
            },
            "analytics_tools": {
                "cache_ttl": 600,
                "detailed_analysis": True
            },
            "intelligence_engine": {
                "learning_enabled": True,
                "max_history_size": 2000
            }
        }
        
        await social_emotion_system.configure_components(component_configs)
        
        # 验证组件配置
        privacy_config = social_emotion_system.privacy_guard.get_configuration()
        assert privacy_config.get("compliance_checking") is True


class TestIntegrationScenarios:
    """集成场景测试"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, social_emotion_system, sample_interaction_context):
        """测试端到端工作流"""
        # 1. 系统启动
        await social_emotion_system.start_system()
        assert social_emotion_system.system_status == SystemStatus.RUNNING
        
        # 2. 处理交互
        result = await social_emotion_system.process_social_interaction(sample_interaction_context)
        assert result.processing_success is True
        
        # 3. 获取分析结果
        analytics = await social_emotion_system.get_session_analytics(
            sample_interaction_context["session_id"]
        )
        assert analytics is not None
        
        # 4. 生成洞察
        insights = await social_emotion_system.generate_comprehensive_insights(
            session_ids=[sample_interaction_context["session_id"]]
        )
        assert insights is not None
        
        # 5. 获取系统指标
        metrics = await social_emotion_system.get_system_metrics()
        assert metrics.total_sessions_processed >= 1
        
        # 6. 健康检查
        health = await social_emotion_system.check_integration_health()
        assert health.overall_health in ["HEALTHY", "WARNING"]
        
        # 7. 系统停止
        await social_emotion_system.stop_system()
        assert social_emotion_system.system_status == SystemStatus.STOPPED
    
    @pytest.mark.asyncio
    async def test_multi_cultural_enterprise_scenario(self, social_emotion_system):
        """测试多文化企业场景"""
        await social_emotion_system.start_system()
        
        # 创建多文化企业会议场景
        enterprise_context = {
            "session_id": "enterprise_meeting_001",
            "participants": [
                {"user_id": "ceo", "role": "leader", "culture": "western_individualistic"},
                {"user_id": "manager_asia", "role": "manager", "culture": "east_asian_collectivistic"},
                {"user_id": "manager_eu", "role": "manager", "culture": "northern_european_reserved"},
                {"user_id": "manager_latam", "role": "manager", "culture": "latin_expressive"}
            ],
            "conversation_data": [
                {
                    "timestamp": datetime.now() - timedelta(minutes=10),
                    "user_id": "ceo",
                    "message": "我们需要在下个季度实现30%的增长",
                    "emotions": {"determination": 0.9, "pressure": 0.7}
                },
                {
                    "timestamp": datetime.now() - timedelta(minutes=8),
                    "user_id": "manager_asia",
                    "message": "这个目标很有挑战性，我们需要仔细规划",
                    "emotions": {"concern": 0.6, "respect": 0.8, "caution": 0.7}
                },
                {
                    "timestamp": datetime.now() - timedelta(minutes=6),
                    "user_id": "manager_eu",
                    "message": "让我们分析一下具体的数据和资源",
                    "emotions": {"analytical": 0.8, "pragmatic": 0.9}
                },
                {
                    "timestamp": datetime.now() - timedelta(minutes=4),
                    "user_id": "manager_latam",
                    "message": "我相信我们的团队能够做到！让我们一起努力",
                    "emotions": {"enthusiasm": 0.9, "optimism": 0.8, "warmth": 0.7}
                }
            ],
            "social_environment": {
                "environment_type": "executive_meeting",
                "formality_level": 0.8,
                "participants_count": 4,
                "time_pressure": 0.8,
                "physical_setup": "hybrid",
                "cultural_context": "highly_diverse"
            },
            "cultural_indicators": {
                "ceo": {"language": "en", "region": "US"},
                "manager_asia": {"language": "zh", "region": "CN"},
                "manager_eu": {"language": "sv", "region": "SE"},
                "manager_latam": {"language": "es", "region": "MX"}
            }
        }
        
        result = await social_emotion_system.process_social_interaction(enterprise_context)
        
        # 验证多文化处理
        assert result.processing_success is True
        assert result.cultural_adaptation is not None
        
        # 验证决策建议包含文化适配策略
        cultural_decisions = [
            d for d in result.intelligence_decisions 
            if "cultural" in str(d).lower() or "adaptation" in str(d).lower()
        ]
        assert len(cultural_decisions) > 0
        
        # 验证隐私保护措施
        assert result.privacy_assessment is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])