import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock

from ...services.fault_tolerance_service import FaultToleranceService
from ...ai.fault_tolerance import FaultToleranceSystem, BackupType

@pytest.fixture
def mock_fault_tolerance_system():
    system = Mock(spec=FaultToleranceSystem)
    
    # Mock async methods
    system.get_system_status = AsyncMock()
    system.get_system_metrics = AsyncMock()
    system.get_component_health = AsyncMock()
    system.trigger_manual_backup = AsyncMock()
    system.get_fault_events = AsyncMock()
    
    # Mock recovery manager
    system.recovery_manager = Mock()
    system.recovery_manager.get_recovery_statistics = Mock()
    
    return system

@pytest.fixture
def service(mock_fault_tolerance_system):
    return FaultToleranceService(mock_fault_tolerance_system)

@pytest.fixture
def sample_system_status():
    return {
        "system_started": True,
        "health_summary": {
            "total_components": 5,
            "health_ratio": 0.9,
            "active_faults": 2,
            "status_counts": {
                "healthy": 4,
                "degraded": 1,
                "unhealthy": 0
            }
        },
        "recovery_statistics": {
            "success_rate": 0.95,
            "recent_recoveries": [
                {"fault_id": "fault-1", "recovery_time": 30.0}
            ]
        },
        "consistency_statistics": {
            "consistency_rate": 0.98
        },
        "backup_statistics": {
            "components": {
                "agent-1": {"last_backup": datetime.now().isoformat()},
                "agent-2": {"last_backup": datetime.now().isoformat()}
            }
        },
        "active_faults": [
            {
                "fault_id": "fault-1",
                "detected_at": datetime.now().isoformat()
            }
        ],
        "last_updated": datetime.now().isoformat()
    }

@pytest.fixture
def sample_system_metrics():
    return {
        "fault_detection_metrics": {
            "total_components": 5,
            "healthy_components": 4
        },
        "recovery_metrics": {
            "success_rate": 0.95
        },
        "backup_metrics": {
            "total_backups": 10
        },
        "consistency_metrics": {
            "consistency_rate": 0.98
        },
        "system_availability": 0.96,
        "last_updated": datetime.now().isoformat()
    }

@pytest.fixture
def sample_component_health():
    return {
        "component_id": "agent-1",
        "status": "healthy",
        "last_check": datetime.now().isoformat(),
        "response_time": 0.5,
        "error_rate": 0.01,
        "resource_usage": {
            "cpu": 45.0,
            "memory": 60.0
        }
    }

@pytest.mark.asyncio
class TestFaultToleranceService:
    
    async def test_get_system_overview_excellent_status(self, service, mock_fault_tolerance_system, sample_system_status, sample_system_metrics):
        """测试获取系统概览 - 优秀状态"""
        # 设置优秀状态的值
        sample_system_status["health_summary"]["health_ratio"] = 0.96
        sample_system_status["health_summary"]["active_faults"] = 1
        sample_system_status["recovery_statistics"]["success_rate"] = 0.98
        sample_system_status["consistency_statistics"]["consistency_rate"] = 0.99
        
        mock_fault_tolerance_system.get_system_status.return_value = sample_system_status
        mock_fault_tolerance_system.get_system_metrics.return_value = sample_system_metrics
        
        result = await service.get_system_overview()
        
        assert result["overall_status"] == "excellent"
        assert result["system_started"] is True
        assert result["key_metrics"]["health_ratio"] == 0.96
        assert result["key_metrics"]["system_availability"] == 0.96
        assert "component_summary" in result
        assert "recent_activity" in result
    
    async def test_get_system_overview_degraded_status(self, service, mock_fault_tolerance_system, sample_system_status, sample_system_metrics):
        """测试获取系统概览 - 降级状态"""
        # 设置降级状态的值
        sample_system_status["health_summary"]["health_ratio"] = 0.6
        sample_system_status["health_summary"]["active_faults"] = 15
        sample_system_status["recovery_statistics"]["success_rate"] = 0.75
        sample_system_status["consistency_statistics"]["consistency_rate"] = 0.85
        
        mock_fault_tolerance_system.get_system_status.return_value = sample_system_status
        mock_fault_tolerance_system.get_system_metrics.return_value = sample_system_metrics
        
        result = await service.get_system_overview()
        
        assert result["overall_status"] == "degraded"
        assert result["key_metrics"]["health_ratio"] == 0.6
        assert result["key_metrics"]["active_faults"] == 15
    
    async def test_get_system_overview_error_handling(self, service, mock_fault_tolerance_system):
        """测试获取系统概览的错误处理"""
        mock_fault_tolerance_system.get_system_status.side_effect = Exception("System error")
        
        result = await service.get_system_overview()
        
        assert result["overall_status"] == "error"
        assert "error" in result
        assert "last_updated" in result
    
    async def test_perform_health_assessment_specific_components(self, service, mock_fault_tolerance_system, sample_component_health):
        """测试对特定组件进行健康评估"""
        mock_fault_tolerance_system.get_component_health.return_value = sample_component_health
        
        result = await service.perform_health_assessment(["agent-1", "agent-2"])
        
        assert "assessment_results" in result
        assert "agent-1" in result["assessment_results"]
        assert "agent-2" in result["assessment_results"]
        assert "recommendations" in result
        assert "assessed_at" in result
        
        # 验证健康分析
        agent_analysis = result["assessment_results"]["agent-1"]
        assert agent_analysis["status"] == "healthy"
        assert "health_score" in agent_analysis
        assert "issues" in agent_analysis
        assert "recommendations" in agent_analysis
    
    async def test_perform_health_assessment_system_wide(self, service, mock_fault_tolerance_system, sample_system_status):
        """测试系统整体健康评估"""
        mock_fault_tolerance_system.get_system_status.return_value = sample_system_status
        
        result = await service.perform_health_assessment()
        
        assert "assessment_results" in result
        assert "system" in result["assessment_results"]
        
        system_analysis = result["assessment_results"]["system"]
        assert "health_ratio" in system_analysis
        assert "active_faults" in system_analysis
        assert "system_health_score" in system_analysis
        assert "status" in system_analysis
    
    async def test_perform_health_assessment_error_handling(self, service, mock_fault_tolerance_system):
        """测试健康评估的错误处理"""
        mock_fault_tolerance_system.get_component_health.side_effect = Exception("Assessment error")
        
        result = await service.perform_health_assessment(["agent-1"])
        
        assert "error" in result
        assert "assessed_at" in result
    
    async def test_create_backup_plan(self, service):
        """测试创建备份计划"""
        component_ids = ["agent-1", "agent-2", "agent-3", "agent-4"]
        
        result = await service.create_backup_plan(component_ids)
        
        assert "plan_id" in result
        assert result["components"] == component_ids
        assert "backup_strategy" in result
        assert result["backup_strategy"]["backup_type"] == BackupType.FULL_BACKUP.value
        assert result["backup_strategy"]["parallel_backups"] == 3  # min(4, 3)
        assert result["estimated_duration"] == len(component_ids) * 30
        assert "created_at" in result
    
    async def test_create_backup_plan_error_handling(self, service):
        """测试创建备份计划的错误处理"""
        # 模拟错误情况
        with pytest.raises(Exception):
            await service.create_backup_plan(None)
    
    async def test_execute_backup_plan(self, service, mock_fault_tolerance_system):
        """测试执行备份计划"""
        backup_plan = {
            "plan_id": "test_plan",
            "components": ["agent-1", "agent-2"],
            "backup_strategy": {"backup_type": "full"},
            "created_at": datetime.now().isoformat()
        }
        
        backup_results = {"agent-1": True, "agent-2": False}
        mock_fault_tolerance_system.trigger_manual_backup.return_value = backup_results
        
        result = await service.execute_backup_plan(backup_plan)
        
        assert result["plan_id"] == "test_plan"
        assert result["results"] == backup_results
        assert result["success_rate"] == 0.5  # 1 成功 / 2 总数
        assert "executed_at" in result
        assert "execution_summary" in result
        
        # 验证执行摘要
        summary = result["execution_summary"]
        assert summary["total_components"] == 2
        assert summary["successful_backups"] == 1
        assert summary["failed_backups"] == 1
        assert summary["failed_components"] == ["agent-2"]
    
    async def test_analyze_fault_patterns(self, service, mock_fault_tolerance_system):
        """测试分析故障模式"""
        # 创建测试故障数据
        now = datetime.now()
        fault_events = [
            {
                "fault_id": "fault-1",
                "fault_type": "agent_error",
                "severity": "high",
                "affected_components": ["agent-1"],
                "detected_at": now.isoformat(),
                "resolved": True,
                "resolved_at": (now + timedelta(minutes=5)).isoformat()
            },
            {
                "fault_id": "fault-2",
                "fault_type": "network_issue",
                "severity": "medium",
                "affected_components": ["agent-2"],
                "detected_at": (now - timedelta(days=1)).isoformat(),
                "resolved": False
            },
            {
                "fault_id": "fault-3",
                "fault_type": "agent_error",
                "severity": "low",
                "affected_components": ["agent-1"],
                "detected_at": (now - timedelta(hours=2)).isoformat(),
                "resolved": True,
                "resolved_at": (now - timedelta(hours=1)).isoformat()
            }
        ]
        
        mock_fault_tolerance_system.get_fault_events.return_value = fault_events
        
        result = await service.analyze_fault_patterns(days=7)
        
        assert result["analysis_period_days"] == 7
        assert result["total_faults_analyzed"] == 3
        assert "patterns" in result
        assert "insights" in result
        
        patterns = result["patterns"]
        assert "fault_frequency" in patterns
        assert "fault_types_distribution" in patterns
        assert "affected_components" in patterns
        assert "severity_distribution" in patterns
        assert "recovery_effectiveness" in patterns
        assert "temporal_patterns" in patterns
        
        # 验证故障类型分布
        assert patterns["fault_types_distribution"]["agent_error"] == 2
        assert patterns["fault_types_distribution"]["network_issue"] == 1
        
        # 验证受影响组件
        assert patterns["affected_components"]["agent-1"] == 2
        assert patterns["affected_components"]["agent-2"] == 1
        
        # 验证严重程度分布
        assert patterns["severity_distribution"]["high"] == 1
        assert patterns["severity_distribution"]["medium"] == 1
        assert patterns["severity_distribution"]["low"] == 1
    
    async def test_optimize_recovery_strategies(self, service, mock_fault_tolerance_system):
        """测试优化恢复策略"""
        recovery_stats = {
            "strategy_success_rates": {
                "immediate_restart": 0.95,
                "graceful_restart": 0.85,
                "task_migration": 0.70,
                "service_degradation": 0.60
            }
        }
        
        mock_fault_tolerance_system.recovery_manager.get_recovery_statistics.return_value = recovery_stats
        
        result = await service.optimize_recovery_strategies()
        
        assert "strategy_analysis" in result
        assert "overall_recommendations" in result
        assert "implementation_priority" in result
        assert "analyzed_at" in result
        
        # 验证策略分析
        strategy_analysis = result["strategy_analysis"]
        assert "immediate_restart" in strategy_analysis
        assert "graceful_restart" in strategy_analysis
        assert "task_migration" in strategy_analysis
        assert "service_degradation" in strategy_analysis
        
        # 验证优化潜力评估
        assert strategy_analysis["immediate_restart"]["optimization_potential"] == "low"  # success_rate >= 0.95
        assert strategy_analysis["graceful_restart"]["optimization_potential"] == "medium"  # success_rate >= 0.85
        assert strategy_analysis["task_migration"]["optimization_potential"] == "high"  # success_rate < 0.85
        
        # 验证实现优先级（按成功率从低到高排序）
        priority = result["implementation_priority"]
        assert priority[0] == "service_degradation"  # 最低成功率
        assert priority[-1] == "immediate_restart"   # 最高成功率
    
    async def test_health_score_calculation_healthy_component(self, service, sample_component_health):
        """测试健康评分计算 - 健康组件"""
        analysis = service._analyze_component_health(sample_component_health)
        
        assert analysis["status"] == "healthy"
        assert analysis["health_score"] == 100  # 健康组件，无问题
        assert len(analysis["issues"]) == 0
        assert len(analysis["recommendations"]) == 0
    
    async def test_health_score_calculation_degraded_component(self, service):
        """测试健康评分计算 - 降级组件"""
        degraded_health = {
            "component_id": "agent-1",
            "status": "degraded",
            "response_time": 3.0,  # 高响应时间
            "error_rate": 0.1,     # 高错误率
            "resource_usage": {
                "cpu": 85.0,       # 高CPU使用率
                "memory": 90.0     # 高内存使用率
            }
        }
        
        analysis = service._analyze_component_health(degraded_health)
        
        assert analysis["status"] == "degraded"
        assert analysis["health_score"] < 70  # 应该有惩罚
        assert "High response time" in analysis["issues"]
        assert "High error rate" in analysis["issues"]
        assert "High CPU usage" in analysis["issues"]
        assert "High memory usage" in analysis["issues"]
        assert len(analysis["recommendations"]) > 0
    
    def test_determine_overall_status_levels(self, service):
        """测试系统整体状态判断的不同级别"""
        # 优秀状态
        status = service._determine_overall_status(0.96, 1, 0.98, 0.99)
        assert status == "excellent"
        
        # 良好状态
        status = service._determine_overall_status(0.90, 3, 0.92, 0.96)
        assert status == "good"
        
        # 一般状态
        status = service._determine_overall_status(0.80, 8, 0.85, 0.92)
        assert status == "fair"
        
        # 降级状态
        status = service._determine_overall_status(0.60, 15, 0.70, 0.85)
        assert status == "degraded"
        
        # 严重状态
        status = service._determine_overall_status(0.30, 25, 0.50, 0.70)
        assert status == "critical"
    
    def test_time_utilities(self, service):
        """测试时间工具方法"""
        now = datetime.now()
        
        # 测试最近时间检查
        recent_time = (now - timedelta(hours=1)).isoformat()
        old_time = (now - timedelta(days=2)).isoformat()
        
        assert service._is_recent(recent_time, 24) is True
        assert service._is_recent(old_time, 24) is False
        
        # 测试天数内检查
        recent_day = (now - timedelta(days=3)).isoformat()
        old_day = (now - timedelta(days=10)).isoformat()
        
        assert service._is_within_days(recent_day, 7) is True
        assert service._is_within_days(old_day, 7) is False
        
        # 测试无效时间戳
        assert service._is_recent("invalid_timestamp") is False
        assert service._is_within_days("invalid_timestamp", 7) is False
    
    def test_backup_duration_estimation(self, service):
        """测试备份时长估算"""
        component_ids = ["agent-1", "agent-2", "agent-3"]
        duration = service._estimate_backup_duration(component_ids)
        assert duration == 90  # 3 * 30 秒
        
        # 空列表
        duration = service._estimate_backup_duration([])
        assert duration == 0
    
    def test_backup_execution_summary(self, service):
        """测试备份执行摘要"""
        backup_results = {
            "agent-1": True,
            "agent-2": False,
            "agent-3": True,
            "agent-4": False
        }
        
        summary = service._summarize_backup_execution(backup_results)
        
        assert summary["total_components"] == 4
        assert summary["successful_backups"] == 2
        assert summary["failed_backups"] == 2
        assert summary["success_rate"] == 0.5
        assert set(summary["failed_components"]) == {"agent-2", "agent-4"}
    
    def test_fault_frequency_analysis_empty(self, service):
        """测试空故障列表的频率分析"""
        result = service._analyze_fault_frequency([])
        assert result["daily_average"] == 0
        assert result["trend"] == "stable"
    
    def test_fault_types_analysis(self, service):
        """测试故障类型分析"""
        faults = [
            {"fault_type": "agent_error"},
            {"fault_type": "network_issue"},
            {"fault_type": "agent_error"},
            {"fault_type": "resource_exhaustion"}
        ]
        
        result = service._analyze_fault_types(faults)
        
        assert result["agent_error"] == 2
        assert result["network_issue"] == 1
        assert result["resource_exhaustion"] == 1
    
    def test_affected_components_analysis(self, service):
        """测试受影响组件分析"""
        faults = [
            {"affected_components": ["agent-1", "agent-2"]},
            {"affected_components": ["agent-1"]},
            {"affected_components": ["agent-3", "agent-1"]}
        ]
        
        result = service._analyze_affected_components(faults)
        
        assert result["agent-1"] == 3
        assert result["agent-2"] == 1
        assert result["agent-3"] == 1
    
    def test_recovery_effectiveness_analysis_empty(self, service):
        """测试空故障列表的恢复效果分析"""
        result = service._analyze_recovery_effectiveness([])
        assert result["recovery_rate"] == 0
        assert result["avg_resolution_time"] == 0
    
    def test_recovery_effectiveness_analysis_with_data(self, service):
        """测试有数据的恢复效果分析"""
        now = datetime.now()
        faults = [
            {
                "resolved": True,
                "detected_at": now.isoformat(),
                "resolved_at": (now + timedelta(minutes=5)).isoformat()
            },
            {
                "resolved": False,
                "detected_at": now.isoformat()
            },
            {
                "resolved": True,
                "detected_at": (now - timedelta(minutes=10)).isoformat(),
                "resolved_at": (now - timedelta(minutes=5)).isoformat()
            }
        ]
        
        result = service._analyze_recovery_effectiveness(faults)
        
        assert result["recovery_rate"] == 2/3  # 2 resolved out of 3
        assert result["total_faults"] == 3
        assert result["resolved_faults"] == 2
        assert result["avg_resolution_time"] == 300.0  # 5 minutes average
    
    def test_optimization_potential_calculation(self, service):
        """测试优化潜力计算"""
        assert service._calculate_optimization_potential(0.98) == "low"
        assert service._calculate_optimization_potential(0.90) == "medium"
        assert service._calculate_optimization_potential(0.75) == "high"
    
    def test_strategy_improvements_suggestions(self, service):
        """测试策略改进建议"""
        # 低成功率策略
        improvements = service._suggest_strategy_improvements("test_strategy", 0.75)
        assert "Review test_strategy implementation" in improvements
        assert "Add additional validation steps" in improvements
        assert "Implement retry mechanisms" in improvements
        
        # 中等成功率策略
        improvements = service._suggest_strategy_improvements("test_strategy", 0.85)
        assert "Fine-tune test_strategy parameters" in improvements
        assert "Add monitoring and alerting" in improvements
        
        # 高成功率策略
        improvements = service._suggest_strategy_improvements("test_strategy", 0.95)
        assert len(improvements) == 0
    
    def test_optimization_prioritization(self, service):
        """测试优化优先级排序"""
        strategy_analysis = {
            "strategy_a": {"current_success_rate": 0.95},
            "strategy_b": {"current_success_rate": 0.70},
            "strategy_c": {"current_success_rate": 0.85}
        }
        
        priority = service._prioritize_optimizations(strategy_analysis)
        
        # 应该按成功率从低到高排序
        assert priority == ["strategy_b", "strategy_c", "strategy_a"]
    
    async def test_pattern_insights_generation(self, service):
        """测试模式洞察生成"""
        patterns = {
            "fault_frequency": {"daily_average": 6},  # 高频故障
            "temporal_patterns": {"peak_hour": 14, "peak_count": 5},
            "affected_components": {"agent-1": 10, "agent-2": 2}
        }
        
        insights = service._generate_pattern_insights(patterns)
        
        assert "High fault frequency detected" in insights[0]
        assert "Most faults occur around 14:00" in insights[1]
        assert "Component agent-1 is most frequently affected" in insights[2]