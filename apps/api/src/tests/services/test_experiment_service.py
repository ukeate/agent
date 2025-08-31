"""
实验服务单元测试
"""
import pytest
from datetime import datetime
from datetime import timedelta
from src.core.utils.timezone_utils import utc_now, utc_factory
from unittest.mock import Mock, patch, AsyncMock
import json

from services.experiment_service import (
    ExperimentService,
    ExperimentStatus,
    Variant,
    Metric,
    ExperimentConfig
)


@pytest.fixture
def experiment_service():
    """创建实验服务实例"""
    return ExperimentService()


@pytest.fixture
def sample_config():
    """示例实验配置"""
    return ExperimentConfig(
        name="测试实验",
        description="这是一个测试实验",
        type="A/B",
        variants=[
            Variant(
                id="control",
                name="对照组",
                description="对照组变体",
                traffic_percentage=50,
                is_control=True
            ),
            Variant(
                id="treatment",
                name="实验组",
                description="实验组变体",
                traffic_percentage=50,
                is_control=False
            )
        ],
        metrics=[
            Metric(
                name="转化率",
                type="primary",
                aggregation="mean"
            ),
            Metric(
                name="收入",
                type="secondary",
                aggregation="sum"
            )
        ],
        sample_size=10000,
        confidence_level=0.95,
        start_date=utc_now(),
        end_date=utc_now() + timedelta(days=14)
    )


class TestExperimentService:
    """实验服务测试类"""
    
    @pytest.mark.asyncio
    async def test_create_experiment(self, experiment_service, sample_config):
        """测试创建实验"""
        experiment = await experiment_service.create_experiment(sample_config)
        
        assert experiment.id is not None
        assert experiment.name == sample_config.name
        assert experiment.status == ExperimentStatus.DRAFT
        assert len(experiment.variants) == 2
        assert experiment.created_at is not None
    
    @pytest.mark.asyncio
    async def test_validate_config(self, experiment_service, sample_config):
        """测试配置验证"""
        # 有效配置
        errors = await experiment_service.validate_config(sample_config)
        assert len(errors) == 0
        
        # 无效配置 - 流量总和不为100
        invalid_config = sample_config.copy()
        invalid_config.variants[0].traffic_percentage = 30
        errors = await experiment_service.validate_config(invalid_config)
        assert len(errors) > 0
        assert any("流量" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_start_experiment(self, experiment_service, sample_config):
        """测试启动实验"""
        experiment = await experiment_service.create_experiment(sample_config)
        
        # 启动实验
        started = await experiment_service.start_experiment(experiment.id)
        assert started.status == ExperimentStatus.ACTIVE
        assert started.started_at is not None
        
        # 不能重复启动
        with pytest.raises(ValueError, match="已经在运行"):
            await experiment_service.start_experiment(experiment.id)
    
    @pytest.mark.asyncio
    async def test_pause_experiment(self, experiment_service, sample_config):
        """测试暂停实验"""
        experiment = await experiment_service.create_experiment(sample_config)
        started = await experiment_service.start_experiment(experiment.id)
        
        # 暂停实验
        paused = await experiment_service.pause_experiment(experiment.id)
        assert paused.status == ExperimentStatus.PAUSED
        
        # 不能暂停未运行的实验
        draft_exp = await experiment_service.create_experiment(sample_config)
        with pytest.raises(ValueError, match="不在运行状态"):
            await experiment_service.pause_experiment(draft_exp.id)
    
    @pytest.mark.asyncio
    async def test_stop_experiment(self, experiment_service, sample_config):
        """测试停止实验"""
        experiment = await experiment_service.create_experiment(sample_config)
        started = await experiment_service.start_experiment(experiment.id)
        
        # 停止实验
        stopped = await experiment_service.stop_experiment(experiment.id)
        assert stopped.status == ExperimentStatus.COMPLETED
        assert stopped.ended_at is not None
    
    @pytest.mark.asyncio
    async def test_get_experiment(self, experiment_service, sample_config):
        """测试获取实验"""
        created = await experiment_service.create_experiment(sample_config)
        
        # 获取存在的实验
        experiment = await experiment_service.get_experiment(created.id)
        assert experiment.id == created.id
        assert experiment.name == created.name
        
        # 获取不存在的实验
        with pytest.raises(ValueError, match="不存在"):
            await experiment_service.get_experiment("non-existent-id")
    
    @pytest.mark.asyncio
    async def test_list_experiments(self, experiment_service, sample_config):
        """测试列出实验"""
        # 创建多个实验
        exp1 = await experiment_service.create_experiment(sample_config)
        exp2 = await experiment_service.create_experiment(sample_config)
        await experiment_service.start_experiment(exp2.id)
        
        # 列出所有实验
        all_experiments = await experiment_service.list_experiments()
        assert len(all_experiments) >= 2
        
        # 按状态过滤
        active_experiments = await experiment_service.list_experiments(
            status=ExperimentStatus.ACTIVE
        )
        assert all(exp.status == ExperimentStatus.ACTIVE for exp in active_experiments)
        
        # 分页
        page1 = await experiment_service.list_experiments(page=1, page_size=1)
        assert len(page1) <= 1
    
    @pytest.mark.asyncio
    async def test_update_experiment(self, experiment_service, sample_config):
        """测试更新实验"""
        experiment = await experiment_service.create_experiment(sample_config)
        
        # 更新草稿实验
        updates = {"name": "更新后的实验", "description": "新描述"}
        updated = await experiment_service.update_experiment(experiment.id, updates)
        assert updated.name == "更新后的实验"
        assert updated.description == "新描述"
        
        # 不能更新运行中的实验
        await experiment_service.start_experiment(experiment.id)
        with pytest.raises(ValueError, match="运行中"):
            await experiment_service.update_experiment(experiment.id, updates)
    
    @pytest.mark.asyncio
    async def test_delete_experiment(self, experiment_service, sample_config):
        """测试删除实验"""
        experiment = await experiment_service.create_experiment(sample_config)
        
        # 删除草稿实验
        await experiment_service.delete_experiment(experiment.id)
        with pytest.raises(ValueError):
            await experiment_service.get_experiment(experiment.id)
        
        # 不能删除运行中的实验
        exp2 = await experiment_service.create_experiment(sample_config)
        await experiment_service.start_experiment(exp2.id)
        with pytest.raises(ValueError, match="运行中"):
            await experiment_service.delete_experiment(exp2.id)
    
    @pytest.mark.asyncio
    async def test_clone_experiment(self, experiment_service, sample_config):
        """测试克隆实验"""
        original = await experiment_service.create_experiment(sample_config)
        
        # 克隆实验
        cloned = await experiment_service.clone_experiment(original.id)
        assert cloned.id != original.id
        assert cloned.name == f"{original.name} (副本)"
        assert cloned.status == ExperimentStatus.DRAFT
        assert len(cloned.variants) == len(original.variants)
    
    @pytest.mark.asyncio
    async def test_calculate_sample_size(self, experiment_service):
        """测试样本量计算"""
        sample_size = await experiment_service.calculate_sample_size(
            baseline_rate=0.1,
            minimum_detectable_effect=0.02,
            confidence_level=0.95,
            power=0.8
        )
        
        assert sample_size > 0
        assert isinstance(sample_size, int)
    
    @pytest.mark.asyncio
    async def test_check_srm(self, experiment_service):
        """测试SRM检查"""
        # 正常分配
        result = await experiment_service.check_srm(
            control_users=5000,
            treatment_users=5100,
            expected_ratio=0.5
        )
        assert result["passed"] is True
        assert result["p_value"] > 0.05
        
        # 异常分配
        result = await experiment_service.check_srm(
            control_users=5000,
            treatment_users=6000,
            expected_ratio=0.5
        )
        assert result["passed"] is False
        assert result["p_value"] < 0.05
    
    @pytest.mark.asyncio
    async def test_experiment_lifecycle(self, experiment_service, sample_config):
        """测试完整的实验生命周期"""
        # 创建
        experiment = await experiment_service.create_experiment(sample_config)
        assert experiment.status == ExperimentStatus.DRAFT
        
        # 启动
        experiment = await experiment_service.start_experiment(experiment.id)
        assert experiment.status == ExperimentStatus.ACTIVE
        
        # 暂停
        experiment = await experiment_service.pause_experiment(experiment.id)
        assert experiment.status == ExperimentStatus.PAUSED
        
        # 恢复
        experiment = await experiment_service.resume_experiment(experiment.id)
        assert experiment.status == ExperimentStatus.ACTIVE
        
        # 停止
        experiment = await experiment_service.stop_experiment(experiment.id)
        assert experiment.status == ExperimentStatus.COMPLETED
        
        # 归档
        experiment = await experiment_service.archive_experiment(experiment.id)
        assert experiment.status == ExperimentStatus.ARCHIVED


class TestVariantAssignment:
    """变体分配测试"""
    
    @pytest.mark.asyncio
    async def test_get_variant_deterministic(self, experiment_service, sample_config):
        """测试确定性变体分配"""
        experiment = await experiment_service.create_experiment(sample_config)
        await experiment_service.start_experiment(experiment.id)
        
        # 同一用户应该总是分配到同一变体
        user_id = "test_user_123"
        variant1 = await experiment_service.get_variant(experiment.id, user_id)
        variant2 = await experiment_service.get_variant(experiment.id, user_id)
        
        assert variant1 == variant2
    
    @pytest.mark.asyncio
    async def test_traffic_distribution(self, experiment_service, sample_config):
        """测试流量分配"""
        experiment = await experiment_service.create_experiment(sample_config)
        await experiment_service.start_experiment(experiment.id)
        
        # 分配大量用户并检查分布
        assignments = {}
        for i in range(10000):
            user_id = f"user_{i}"
            variant = await experiment_service.get_variant(experiment.id, user_id)
            assignments[variant.id] = assignments.get(variant.id, 0) + 1
        
        # 检查分配比例（允许5%误差）
        for variant in experiment.variants:
            expected = variant.traffic_percentage / 100 * 10000
            actual = assignments.get(variant.id, 0)
            assert abs(actual - expected) < expected * 0.05
    
    @pytest.mark.asyncio
    async def test_whitelist_blacklist(self, experiment_service, sample_config):
        """测试黑白名单"""
        config = sample_config.copy()
        config.whitelist = ["whitelist_user"]
        config.blacklist = ["blacklist_user"]
        
        experiment = await experiment_service.create_experiment(config)
        await experiment_service.start_experiment(experiment.id)
        
        # 白名单用户应该被分配
        variant = await experiment_service.get_variant(experiment.id, "whitelist_user")
        assert variant is not None
        
        # 黑名单用户不应该被分配
        variant = await experiment_service.get_variant(experiment.id, "blacklist_user")
        assert variant is None
        
        # 普通用户应该正常分配
        variant = await experiment_service.get_variant(experiment.id, "normal_user")
        assert variant is not None


class TestMetricsCalculation:
    """指标计算测试"""
    
    @pytest.mark.asyncio
    async def test_calculate_conversion_rate(self, experiment_service):
        """测试转化率计算"""
        events = [
            {"user_id": "1", "event": "view"},
            {"user_id": "1", "event": "purchase"},
            {"user_id": "2", "event": "view"},
            {"user_id": "3", "event": "view"},
            {"user_id": "3", "event": "purchase"}
        ]
        
        conversion_rate = await experiment_service.calculate_conversion_rate(
            events,
            numerator_event="purchase",
            denominator_event="view"
        )
        
        assert conversion_rate == 2/3  # 2个购买用户 / 3个浏览用户
    
    @pytest.mark.asyncio
    async def test_calculate_average_metric(self, experiment_service):
        """测试平均值指标计算"""
        events = [
            {"user_id": "1", "value": 100},
            {"user_id": "2", "value": 200},
            {"user_id": "3", "value": 150}
        ]
        
        average = await experiment_service.calculate_average(events, "value")
        assert average == 150
    
    @pytest.mark.asyncio
    async def test_statistical_significance(self, experiment_service):
        """测试统计显著性计算"""
        # 显著差异
        result = await experiment_service.test_significance(
            control_mean=0.1,
            control_std=0.05,
            control_n=1000,
            treatment_mean=0.12,
            treatment_std=0.05,
            treatment_n=1000,
            confidence_level=0.95
        )
        
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert result["confidence_interval"][0] > 0
        
        # 无显著差异
        result = await experiment_service.test_significance(
            control_mean=0.1,
            control_std=0.05,
            control_n=100,
            treatment_mean=0.101,
            treatment_std=0.05,
            treatment_n=100,
            confidence_level=0.95
        )
        
        assert result["significant"] is False
        assert result["p_value"] > 0.05


if __name__ == "__main__":
    pytest.main([__file__, "-v"])