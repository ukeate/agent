"""
实验工作流集成测试
"""
import pytest
import asyncio
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from httpx import AsyncClient
import json
import numpy as np

from main import app
from services.experiment_service import ExperimentService
from services.event_tracking_service import EventTrackingService
from services.statistical_analysis_service import StatisticalAnalysisService
from services.realtime_metrics_service import RealtimeMetricsService


@pytest.fixture
async def async_client():
    """创建异步HTTP客户端"""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
async def experiment_data():
    """创建测试实验数据"""
    return {
        "name": "集成测试实验",
        "description": "用于集成测试的实验",
        "type": "A/B",
        "variants": [
            {
                "name": "对照组",
                "traffic_percentage": 50,
                "is_control": True
            },
            {
                "name": "实验组",
                "traffic_percentage": 50,
                "is_control": False
            }
        ],
        "metrics": [
            {
                "name": "转化率",
                "type": "primary",
                "aggregation": "mean"
            },
            {
                "name": "收入",
                "type": "secondary",
                "aggregation": "sum"
            }
        ],
        "sample_size": 1000,
        "confidence_level": 95,
        "start_date": utc_now().isoformat(),
        "end_date": (utc_now() + timedelta(days=7)).isoformat()
    }


class TestExperimentLifecycle:
    """实验生命周期集成测试"""
    
    @pytest.mark.asyncio
    async def test_complete_experiment_workflow(self, async_client, experiment_data):
        """测试完整的实验工作流"""
        # 1. 创建实验
        response = await async_client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        assert response.status_code == 200
        experiment = response.json()
        experiment_id = experiment["experiment"]["id"]
        assert experiment["experiment"]["status"] == "draft"
        
        # 2. 验证配置
        response = await async_client.post(
            "/api/v1/experiments/validate",
            json=experiment_data
        )
        assert response.status_code == 200
        validation = response.json()
        assert validation["valid"] is True
        
        # 3. 计算样本量
        response = await async_client.post(
            "/api/v1/power-analysis/sample-size",
            json={
                "baseline_rate": 0.1,
                "minimum_detectable_effect": 0.02,
                "confidence_level": 0.95,
                "power": 0.8
            }
        )
        assert response.status_code == 200
        sample_size = response.json()
        assert sample_size["sample_size"] > 0
        
        # 4. 启动实验
        response = await async_client.post(
            f"/api/v1/experiments/{experiment_id}/start"
        )
        assert response.status_code == 200
        assert response.json()["experiment"]["status"] == "active"
        
        # 5. 获取用户变体分配
        user_assignments = {}
        for i in range(100):
            response = await async_client.get(
                f"/api/v1/traffic-allocation/assign",
                params={
                    "experiment_id": experiment_id,
                    "user_id": f"user_{i}"
                }
            )
            assert response.status_code == 200
            assignment = response.json()
            user_assignments[f"user_{i}"] = assignment["variant"]
        
        # 验证流量分配比例
        control_count = sum(1 for v in user_assignments.values() if v["is_control"])
        assert 40 <= control_count <= 60  # 允许一定偏差
        
        # 6. 发送事件
        events = []
        for user_id, variant in user_assignments.items():
            # 生成事件
            conversion = np.random.random() < (0.1 if variant["is_control"] else 0.12)
            revenue = np.random.exponential(50) if conversion else 0
            
            event = {
                "experiment_id": experiment_id,
                "user_id": user_id,
                "variant_id": variant["id"],
                "event_type": "conversion" if conversion else "view",
                "properties": {
                    "revenue": revenue,
                    "timestamp": utc_now().isoformat()
                }
            }
            events.append(event)
        
        # 批量发送事件
        response = await async_client.post(
            "/api/v1/event-batch/track",
            json={"events": events}
        )
        assert response.status_code == 200
        
        # 7. 获取实时指标
        response = await async_client.get(
            f"/api/v1/realtime-metrics/{experiment_id}"
        )
        assert response.status_code == 200
        metrics = response.json()
        assert "metrics" in metrics
        assert len(metrics["metrics"]) > 0
        
        # 8. 进行统计分析
        response = await async_client.post(
            "/api/v1/statistical-analysis/compare-variants",
            json={"experiment_id": experiment_id}
        )
        assert response.status_code == 200
        analysis = response.json()
        assert "p_value" in analysis
        assert "confidence_interval" in analysis
        
        # 9. 检查SRM
        response = await async_client.post(
            "/api/v1/data-quality/srm-check",
            json={"experiment_id": experiment_id}
        )
        assert response.status_code == 200
        srm = response.json()
        assert srm["passed"] is True
        
        # 10. 生成报告
        response = await async_client.post(
            "/api/v1/report-generation/generate",
            json={"experiment_id": experiment_id}
        )
        assert response.status_code == 200
        report = response.json()
        assert "summary" in report
        assert "metrics" in report
        assert "recommendations" in report
        
        # 11. 停止实验
        response = await async_client.post(
            f"/api/v1/experiments/{experiment_id}/stop"
        )
        assert response.status_code == 200
        assert response.json()["experiment"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_experiment_pause_resume(self, async_client, experiment_data):
        """测试实验暂停和恢复"""
        # 创建并启动实验
        response = await async_client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        experiment_id = response.json()["experiment"]["id"]
        
        await async_client.post(f"/api/v1/experiments/{experiment_id}/start")
        
        # 暂停实验
        response = await async_client.post(
            f"/api/v1/experiments/{experiment_id}/pause"
        )
        assert response.status_code == 200
        assert response.json()["experiment"]["status"] == "paused"
        
        # 验证暂停状态下不能分配用户
        response = await async_client.get(
            "/api/v1/traffic-allocation/assign",
            params={
                "experiment_id": experiment_id,
                "user_id": "test_user"
            }
        )
        assert response.status_code == 400
        
        # 恢复实验
        response = await async_client.post(
            f"/api/v1/experiments/{experiment_id}/resume"
        )
        assert response.status_code == 200
        assert response.json()["experiment"]["status"] == "active"
        
        # 验证恢复后可以分配用户
        response = await async_client.get(
            "/api/v1/traffic-allocation/assign",
            params={
                "experiment_id": experiment_id,
                "user_id": "test_user"
            }
        )
        assert response.status_code == 200


class TestTrafficManagement:
    """流量管理集成测试"""
    
    @pytest.mark.asyncio
    async def test_traffic_ramping(self, async_client, experiment_data):
        """测试流量渐进调整"""
        # 创建实验
        response = await async_client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        experiment_id = response.json()["experiment"]["id"]
        
        # 创建渐进计划
        response = await async_client.post(
            "/api/v1/traffic-ramp/plans",
            json={
                "experiment_id": experiment_id,
                "start_percentage": 10,
                "end_percentage": 100,
                "duration_hours": 24,
                "strategy": "linear"
            }
        )
        assert response.status_code == 200
        plan = response.json()
        
        # 执行渐进计划
        response = await async_client.post(
            f"/api/v1/traffic-ramp/execute/{plan['plan']['id']}"
        )
        assert response.status_code == 200
        
        # 获取当前流量
        response = await async_client.get(
            f"/api/v1/traffic-ramp/current/{experiment_id}"
        )
        assert response.status_code == 200
        current = response.json()
        assert 0 <= current["percentage"] <= 100
    
    @pytest.mark.asyncio
    async def test_auto_scaling(self, async_client, experiment_data):
        """测试自动扩量"""
        # 创建实验
        response = await async_client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        experiment_id = response.json()["experiment"]["id"]
        
        # 创建扩量规则
        response = await async_client.post(
            "/api/v1/auto-scaling/rules",
            json={
                "experiment_id": experiment_id,
                "name": "显著性扩量",
                "trigger_type": "statistical_significance",
                "trigger_config": {
                    "p_value_threshold": 0.05,
                    "min_sample_size": 100
                },
                "action_type": "increase_traffic",
                "action_config": {
                    "target_percentage": 100,
                    "step_size": 20
                }
            }
        )
        assert response.status_code == 200
        
        # 评估扩量决策
        response = await async_client.post(
            f"/api/v1/auto-scaling/evaluate/{experiment_id}"
        )
        assert response.status_code == 200
        decision = response.json()
        assert "should_scale" in decision
    
    @pytest.mark.asyncio
    async def test_targeting_rules(self, async_client, experiment_data):
        """测试定向规则"""
        # 添加定向规则
        experiment_data["targeting_rules"] = [
            {
                "type": "user_property",
                "field": "country",
                "operator": "in",
                "value": ["US", "CA"]
            }
        ]
        
        response = await async_client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        experiment_id = response.json()["experiment"]["id"]
        await async_client.post(f"/api/v1/experiments/{experiment_id}/start")
        
        # 测试符合规则的用户
        response = await async_client.get(
            "/api/v1/traffic-allocation/assign",
            params={
                "experiment_id": experiment_id,
                "user_id": "us_user",
                "properties": json.dumps({"country": "US"})
            }
        )
        assert response.status_code == 200
        assert response.json()["variant"] is not None
        
        # 测试不符合规则的用户
        response = await async_client.get(
            "/api/v1/traffic-allocation/assign",
            params={
                "experiment_id": experiment_id,
                "user_id": "cn_user",
                "properties": json.dumps({"country": "CN"})
            }
        )
        assert response.status_code == 200
        assert response.json()["variant"] is None


class TestMonitoringAndAlerts:
    """监控和告警集成测试"""
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, async_client, experiment_data):
        """测试异常检测"""
        # 创建实验
        response = await async_client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        experiment_id = response.json()["experiment"]["id"]
        
        # 发送正常数据
        normal_data = [
            {"timestamp": utc_now().isoformat(), "value": 100 + np.random.normal(0, 5)}
            for _ in range(50)
        ]
        
        # 添加异常点
        anomaly_data = normal_data + [
            {"timestamp": utc_now().isoformat(), "value": 150}  # 异常高值
        ]
        
        response = await async_client.post(
            "/api/v1/anomaly-detection/detect",
            json={
                "experiment_id": experiment_id,
                "metric": "conversion_rate",
                "data": anomaly_data,
                "method": "zscore"
            }
        )
        assert response.status_code == 200
        anomalies = response.json()
        assert len(anomalies["anomalies"]) > 0
    
    @pytest.mark.asyncio
    async def test_alert_rules(self, async_client, experiment_data):
        """测试告警规则"""
        # 创建实验
        response = await async_client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        experiment_id = response.json()["experiment"]["id"]
        
        # 创建告警规则
        response = await async_client.post(
            "/api/v1/alert-rules",
            json={
                "experiment_id": experiment_id,
                "name": "转化率下降",
                "metric": "conversion_rate",
                "condition": "less_than",
                "threshold": 0.08,
                "window_minutes": 60,
                "severity": "critical"
            }
        )
        assert response.status_code == 200
        rule = response.json()
        
        # 评估规则
        response = await async_client.post(
            f"/api/v1/alert-rules/evaluate/{rule['rule']['id']}"
        )
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_risk_assessment(self, async_client, experiment_data):
        """测试风险评估"""
        # 创建实验
        response = await async_client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        experiment_id = response.json()["experiment"]["id"]
        
        # 评估风险
        response = await async_client.post(
            "/api/v1/risk-assessment/assess",
            json={
                "experiment_id": experiment_id,
                "include_predictions": True
            }
        )
        assert response.status_code == 200
        assessment = response.json()
        assert "overall_risk_level" in assessment["assessment"]
        assert "risk_factors" in assessment["assessment"]
        assert "recommendations" in assessment["assessment"]


class TestReportingAndAnalysis:
    """报告和分析集成测试"""
    
    @pytest.mark.asyncio
    async def test_report_generation(self, async_client, experiment_data):
        """测试报告生成"""
        # 创建实验并生成数据
        response = await async_client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        experiment_id = response.json()["experiment"]["id"]
        
        # 生成不同格式的报告
        for format_type in ["json", "html", "pdf"]:
            response = await async_client.post(
                "/api/v1/report-generation/export",
                json={
                    "experiment_id": experiment_id,
                    "format": format_type
                }
            )
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_multiple_testing_correction(self, async_client, experiment_data):
        """测试多重检验校正"""
        # 模拟多个p值
        p_values = [0.01, 0.04, 0.03, 0.20, 0.15]
        
        response = await async_client.post(
            "/api/v1/multiple-testing-correction/correct",
            json={
                "p_values": p_values,
                "method": "benjamini_hochberg",
                "alpha": 0.05
            }
        )
        assert response.status_code == 200
        corrected = response.json()
        assert len(corrected["adjusted_p_values"]) == len(p_values)
        assert "rejected" in corrected
    
    @pytest.mark.asyncio
    async def test_segment_analysis(self, async_client, experiment_data):
        """测试分段分析"""
        # 创建实验
        response = await async_client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        experiment_id = response.json()["experiment"]["id"]
        
        # 发送分段数据
        segments = ["new_users", "returning_users", "premium_users"]
        for segment in segments:
            events = [
                {
                    "experiment_id": experiment_id,
                    "user_id": f"{segment}_{i}",
                    "segment": segment,
                    "event_type": "conversion",
                    "value": np.random.random()
                }
                for i in range(50)
            ]
            
            await async_client.post(
                "/api/v1/event-batch/track",
                json={"events": events}
            )
        
        # 分析分段结果
        response = await async_client.post(
            "/api/v1/statistical-analysis/segment-analysis",
            json={
                "experiment_id": experiment_id,
                "segments": segments
            }
        )
        assert response.status_code == 200
        analysis = response.json()
        assert len(analysis["segments"]) == len(segments)


class TestReleaseStrategy:
    """发布策略集成测试"""
    
    @pytest.mark.asyncio
    async def test_canary_release(self, async_client, experiment_data):
        """测试金丝雀发布"""
        # 创建实验
        response = await async_client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        experiment_id = response.json()["experiment"]["id"]
        
        # 创建金丝雀发布策略
        response = await async_client.post(
            "/api/v1/release-strategy/strategies",
            json={
                "experiment_id": experiment_id,
                "name": "金丝雀发布",
                "release_type": "canary",
                "stages": [
                    {
                        "name": "初始发布",
                        "environment": "production",
                        "traffic_percentage": 5,
                        "duration_hours": 24,
                        "success_criteria": {
                            "error_rate": {"max": 0.01},
                            "latency_p99": {"max": 500}
                        }
                    },
                    {
                        "name": "扩大发布",
                        "environment": "production",
                        "traffic_percentage": 50,
                        "duration_hours": 48,
                        "success_criteria": {
                            "error_rate": {"max": 0.01},
                            "latency_p99": {"max": 500}
                        }
                    }
                ]
            }
        )
        assert response.status_code == 200
        strategy = response.json()
        
        # 执行策略
        response = await async_client.post(
            f"/api/v1/release-strategy/execute/{strategy['strategy']['id']}"
        )
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_rollback_mechanism(self, async_client, experiment_data):
        """测试回滚机制"""
        # 创建实验
        response = await async_client.post(
            "/api/v1/experiments",
            json=experiment_data
        )
        experiment_id = response.json()["experiment"]["id"]
        
        # 创建回滚计划
        response = await async_client.post(
            "/api/v1/risk-assessment/rollback-plan",
            json={
                "experiment_id": experiment_id,
                "strategy": "immediate",
                "auto_execute": False
            }
        )
        assert response.status_code == 200
        plan = response.json()
        
        # 执行回滚
        response = await async_client.post(
            "/api/v1/risk-assessment/rollback/execute",
            json={
                "plan_id": plan["plan_id"],
                "force": True
            }
        )
        assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])