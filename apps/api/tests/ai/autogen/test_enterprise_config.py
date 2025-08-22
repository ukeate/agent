"""
企业级配置管理测试
测试集中式配置服务、配置同步、配置验证等功能
"""

import pytest
import asyncio
import json
import yaml
import tempfile
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch
from pathlib import Path

from src.ai.autogen.enterprise_config import (
    EnterpriseConfigManager, EnterpriseConfigSettings,
    ConfigItem, ConfigLevel, ConfigCategory,
    get_config_manager, init_config_manager
)


@pytest.fixture
def config_settings():
    """创建配置设置实例"""
    return EnterpriseConfigSettings()


@pytest.fixture
def redis_client():
    """创建模拟Redis客户端"""
    client = AsyncMock()
    client.hgetall = AsyncMock(return_value={})
    client.hset = AsyncMock(return_value=True)
    client.publish = AsyncMock(return_value=1)
    client.pubsub = AsyncMock()
    return client


@pytest.fixture
def temp_config_file():
    """创建临时配置文件"""
    config_data = {
        "TEST_SETTING_1": {
            "value": 100,
            "category": "performance",
            "level": "system",
            "description": "Test setting 1"
        },
        "TEST_SETTING_2": {
            "value": 0.75,
            "category": "security", 
            "level": "service",
            "description": "Test setting 2"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_data, f)
        temp_file = f.name
    
    yield temp_file
    
    # 清理
    os.unlink(temp_file)


@pytest.fixture
def config_manager(redis_client, temp_config_file):
    """创建配置管理器实例"""
    return EnterpriseConfigManager(
        redis_client=redis_client,
        config_file=temp_config_file
    )


class TestEnterpriseConfigSettings:
    """企业级配置设置测试"""
    
    def test_default_values(self, config_settings):
        """测试默认配置值"""
        assert config_settings.AGENT_POOL_MIN_SIZE == 1
        assert config_settings.AGENT_POOL_MAX_SIZE == 10
        assert config_settings.AGENT_POOL_INITIAL_SIZE == 3
        assert config_settings.LOAD_BALANCER_THRESHOLD == 0.8
        assert config_settings.SECURITY_RATE_LIMIT_DEFAULT == 10
        assert config_settings.MONITORING_MAX_LOGS == 10000
    
    def test_security_config_values(self, config_settings):
        """测试安全相关配置"""
        assert config_settings.SECURITY_RATE_LIMIT_WINDOW == 60
        assert config_settings.SECURITY_HIGH_SEVERITY_WINDOW == 300
        assert config_settings.SECURITY_CRITICAL_SEVERITY_WINDOW == 180
        assert config_settings.SECURITY_QUARANTINE_TIME == 3600
    
    def test_performance_config_values(self, config_settings):
        """测试性能相关配置"""
        assert config_settings.PERFORMANCE_MAX_CONCURRENT_TASKS == 50
        assert config_settings.PERFORMANCE_TASK_TIMEOUT == 30000
        assert config_settings.PERFORMANCE_SMOOTHING_ALPHA == 0.1
    
    def test_monitoring_config_values(self, config_settings):
        """测试监控相关配置"""
        assert config_settings.MONITORING_LOG_RETENTION_DAYS == 30
        assert config_settings.MONITORING_METRIC_COLLECTION_INTERVAL == 30
        assert config_settings.MONITORING_ALERT_CPU_THRESHOLD == 0.9


class TestConfigItem:
    """配置项测试"""
    
    def test_config_item_creation(self):
        """测试配置项创建"""
        item = ConfigItem(
            key="test_key",
            value=42,
            category=ConfigCategory.PERFORMANCE,
            level=ConfigLevel.SYSTEM,
            description="Test configuration item"
        )
        
        assert item.key == "test_key"
        assert item.value == 42
        assert item.category == ConfigCategory.PERFORMANCE
        assert item.level == ConfigLevel.SYSTEM
        assert item.description == "Test configuration item"
        assert isinstance(item.created_at, datetime)
    
    def test_config_item_with_constraints(self):
        """测试带约束的配置项"""
        item = ConfigItem(
            key="constrained_key",
            value=50,
            category=ConfigCategory.SECURITY,
            level=ConfigLevel.SERVICE,
            min_value=0,
            max_value=100,
            valid_values=[10, 25, 50, 75, 100]
        )
        
        assert item.min_value == 0
        assert item.max_value == 100
        assert 50 in item.valid_values


class TestEnterpriseConfigManager:
    """企业级配置管理器测试"""
    
    def test_load_local_config(self, config_manager):
        """测试加载本地配置"""
        # 配置应该从临时文件加载
        assert "TEST_SETTING_1" in config_manager._config_cache
        assert "TEST_SETTING_2" in config_manager._config_cache
        
        item1 = config_manager._config_cache["TEST_SETTING_1"]
        assert item1.value == 100
        assert item1.category == ConfigCategory.PERFORMANCE
    
    def test_get_configuration(self, config_manager):
        """测试获取配置"""
        # 测试缓存中的配置
        value1 = config_manager.get("TEST_SETTING_1")
        assert value1 == 100
        
        # 测试settings中的配置
        value2 = config_manager.get("AGENT_POOL_MIN_SIZE")
        assert value2 == 1
        
        # 测试默认值
        value3 = config_manager.get("NON_EXISTENT_KEY", "default")
        assert value3 == "default"
    
    def test_get_typed_values(self, config_manager):
        """测试获取类型化的配置值"""
        # 测试整数
        int_val = config_manager.get_int("TEST_SETTING_1")
        assert int_val == 100
        assert isinstance(int_val, int)
        
        # 测试浮点数
        float_val = config_manager.get_float("TEST_SETTING_2")
        assert float_val == 0.75
        assert isinstance(float_val, float)
        
        # 测试布尔值
        bool_val = config_manager.get_bool("FLOW_CONTROL_ENABLE_BACKPRESSURE")
        assert bool_val is True
        
        # 测试默认值转换
        default_int = config_manager.get_int("NON_EXISTENT", 42)
        assert default_int == 42
    
    def test_get_list_values(self, config_manager):
        """测试获取列表配置值"""
        # 测试逗号分隔的字符串
        with patch.object(config_manager, 'get', return_value="item1,item2,item3"):
            list_val = config_manager.get_list("TEST_LIST")
            assert list_val == ["item1", "item2", "item3"]
        
        # 测试已经是列表的值
        with patch.object(config_manager, 'get', return_value=["a", "b", "c"]):
            list_val = config_manager.get_list("TEST_LIST")
            assert list_val == ["a", "b", "c"]
        
        # 测试默认值
        default_list = config_manager.get_list("NON_EXISTENT", ["default"])
        assert default_list == ["default"]
    
    @pytest.mark.asyncio
    async def test_set_configuration(self, config_manager):
        """测试设置配置"""
        await config_manager.set(
            "NEW_SETTING",
            42,
            category=ConfigCategory.PERFORMANCE,
            level=ConfigLevel.SERVICE,
            description="New test setting"
        )
        
        assert "NEW_SETTING" in config_manager._config_cache
        item = config_manager._config_cache["NEW_SETTING"]
        assert item.value == 42
        assert item.category == ConfigCategory.PERFORMANCE
    
    @pytest.mark.asyncio
    async def test_set_invalid_configuration(self, config_manager):
        """测试设置无效配置"""
        with patch.object(config_manager, '_validate_config', return_value=False):
            with pytest.raises(ValueError):
                await config_manager.set("INVALID_SETTING", "invalid_value")
    
    def test_validate_config(self, config_manager):
        """测试配置验证"""
        # 测试数值范围验证
        valid_item = ConfigItem(
            key="test",
            value=50,
            category=ConfigCategory.PERFORMANCE,
            level=ConfigLevel.SYSTEM,
            min_value=0,
            max_value=100
        )
        assert config_manager._validate_config(valid_item) is True
        
        # 测试超出范围
        invalid_item = ConfigItem(
            key="test",
            value=150,
            category=ConfigCategory.PERFORMANCE,
            level=ConfigLevel.SYSTEM,
            min_value=0,
            max_value=100
        )
        assert config_manager._validate_config(invalid_item) is False
        
        # 测试有效值列表
        valid_enum_item = ConfigItem(
            key="test",
            value="option_a",
            category=ConfigCategory.PERFORMANCE,
            level=ConfigLevel.SYSTEM,
            valid_values=["option_a", "option_b", "option_c"]
        )
        assert config_manager._validate_config(valid_enum_item) is True
        
        invalid_enum_item = ConfigItem(
            key="test",
            value="invalid_option",
            category=ConfigCategory.PERFORMANCE,
            level=ConfigLevel.SYSTEM,
            valid_values=["option_a", "option_b", "option_c"]
        )
        assert config_manager._validate_config(invalid_enum_item) is False
    
    @pytest.mark.asyncio
    async def test_redis_persistence(self, config_manager):
        """测试Redis持久化"""
        await config_manager.set("REDIS_TEST", 123, persist=True)
        
        # 验证Redis调用
        config_manager.redis_client.hset.assert_called()
        config_manager.redis_client.publish.assert_called()
    
    @pytest.mark.asyncio
    async def test_redis_sync(self, config_manager):
        """测试Redis同步"""
        # 模拟Redis返回的配置数据
        redis_data = {
            b"REDIS_SETTING": json.dumps({
                "value": "redis_value",
                "category": "performance",
                "level": "system",
                "description": "Redis setting",
                "updated_at": datetime.now().isoformat()
            })
        }
        
        config_manager.redis_client.hgetall.return_value = redis_data
        
        await config_manager._sync_from_redis()
        
        assert "REDIS_SETTING" in config_manager._config_cache
        assert config_manager._config_cache["REDIS_SETTING"].value == "redis_value"
    
    def test_configuration_subscription(self, config_manager):
        """测试配置订阅"""
        callback_called = False
        received_value = None
        
        def test_callback(key, value):
            nonlocal callback_called, received_value
            callback_called = True
            received_value = value
        
        # 订阅配置变更
        config_manager.subscribe("TEST_KEY", test_callback)
        
        # 触发通知
        asyncio.run(config_manager._notify_subscribers("TEST_KEY", "new_value"))
        
        assert callback_called is True
        assert received_value == "new_value"
    
    def test_get_performance_config(self, config_manager):
        """测试获取性能配置"""
        perf_config = config_manager.get_performance_config()
        
        assert "agent_pool" in perf_config
        assert "load_balancer" in perf_config
        assert "performance" in perf_config
        
        assert perf_config["agent_pool"]["min_size"] == 1
        assert perf_config["load_balancer"]["threshold"] == 0.8
    
    def test_get_security_config(self, config_manager):
        """测试获取安全配置"""
        security_config = config_manager.get_security_config()
        
        assert "rate_limiting" in security_config
        assert "quarantine" in security_config
        
        assert security_config["rate_limiting"]["default_limit"] == 10
        assert security_config["quarantine"]["time"] == 3600
    
    def test_get_monitoring_config(self, config_manager):
        """测试获取监控配置"""
        monitoring_config = config_manager.get_monitoring_config()
        
        assert "logging" in monitoring_config
        assert "metrics" in monitoring_config
        assert "alerts" in monitoring_config
        
        assert monitoring_config["logging"]["max_logs"] == 10000
        assert monitoring_config["alerts"]["cpu_threshold"] == 0.9
    
    def test_export_config_yaml(self, config_manager):
        """测试导出YAML配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            export_file = f.name
        
        try:
            config_manager.export_config(export_file, "yaml")
            
            # 验证文件存在且可读
            assert os.path.exists(export_file)
            
            with open(export_file, 'r') as f:
                exported_data = yaml.safe_load(f)
            
            assert "TEST_SETTING_1" in exported_data
            assert exported_data["TEST_SETTING_1"]["value"] == 100
            
        finally:
            if os.path.exists(export_file):
                os.unlink(export_file)
    
    def test_export_config_json(self, config_manager):
        """测试导出JSON配置"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            export_file = f.name
        
        try:
            config_manager.export_config(export_file, "json")
            
            # 验证文件存在且可读
            assert os.path.exists(export_file)
            
            with open(export_file, 'r') as f:
                exported_data = json.load(f)
            
            assert "TEST_SETTING_1" in exported_data
            assert exported_data["TEST_SETTING_1"]["value"] == 100
            
        finally:
            if os.path.exists(export_file):
                os.unlink(export_file)
    
    def test_get_all_configs_filtered(self, config_manager):
        """测试获取过滤的配置"""
        # 按类别过滤
        perf_configs = config_manager.get_all_configs(category=ConfigCategory.PERFORMANCE)
        assert "TEST_SETTING_1" in perf_configs
        assert "TEST_SETTING_2" not in perf_configs
        
        # 按级别过滤
        system_configs = config_manager.get_all_configs(level=ConfigLevel.SYSTEM)
        assert "TEST_SETTING_1" in system_configs
        assert "TEST_SETTING_2" not in system_configs


class TestConfigManagerLifecycle:
    """配置管理器生命周期测试"""
    
    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self, redis_client, temp_config_file):
        """测试启动停止生命周期"""
        manager = EnterpriseConfigManager(redis_client, temp_config_file)
        
        # 启动
        await manager.start()
        assert len(manager._watch_tasks) > 0
        
        # 停止
        await manager.stop()
        
        # 验证任务被取消
        for task in manager._watch_tasks:
            assert task.cancelled() or task.done()
    
    @pytest.mark.asyncio
    async def test_redis_watch_changes(self, config_manager):
        """测试Redis配置变更监控"""
        # 模拟pubsub消息
        mock_pubsub = AsyncMock()
        mock_message = {
            'type': 'message',
            'data': json.dumps({
                'key': 'WATCHED_KEY',
                'value': 'new_value',
                'timestamp': datetime.now().isoformat()
            })
        }
        
        async def mock_listen():
            yield mock_message
        
        mock_pubsub.listen = mock_listen
        config_manager.redis_client.pubsub.return_value = mock_pubsub
        
        # 添加配置到缓存
        config_manager._config_cache['WATCHED_KEY'] = ConfigItem(
            key='WATCHED_KEY',
            value='old_value',
            category=ConfigCategory.PERFORMANCE,
            level=ConfigLevel.SYSTEM
        )
        
        # 设置订阅者
        callback_called = False
        def test_callback(key, value):
            nonlocal callback_called
            callback_called = True
        
        config_manager.subscribe('WATCHED_KEY', test_callback)
        
        # 模拟监控任务
        await config_manager._watch_redis_changes()
        
        # 验证配置被更新
        assert config_manager._config_cache['WATCHED_KEY'].value == 'new_value'


class TestGlobalConfigManager:
    """全局配置管理器测试"""
    
    def test_get_config_manager_singleton(self):
        """测试单例模式"""
        manager1 = get_config_manager()
        manager2 = get_config_manager()
        
        assert manager1 is manager2
    
    @pytest.mark.asyncio
    async def test_init_config_manager(self, redis_client, temp_config_file):
        """测试初始化全局配置管理器"""
        await init_config_manager(redis_client, temp_config_file)
        
        manager = get_config_manager()
        assert manager is not None
        assert manager.config_file == temp_config_file
        assert manager.redis_client is redis_client


@pytest.mark.integration
class TestConfigManagerIntegration:
    """配置管理器集成测试"""
    
    @pytest.mark.asyncio
    async def test_config_change_propagation(self):
        """测试配置变更传播"""
        # 创建两个管理器实例模拟分布式环境
        redis_client = AsyncMock()
        
        manager1 = EnterpriseConfigManager(redis_client)
        manager2 = EnterpriseConfigManager(redis_client)
        
        # 模拟配置变更
        await manager1.set("SHARED_CONFIG", "value1")
        
        # 模拟Redis发布配置变更
        change_message = {
            'key': 'SHARED_CONFIG',
            'value': 'value1',
            'timestamp': datetime.now().isoformat()
        }
        
        # 模拟manager2接收到变更通知
        await manager2._notify_subscribers("SHARED_CONFIG", "value1")
        
        # 验证配置在两个管理器中都存在
        assert manager1.get("SHARED_CONFIG") == "value1"
    
    @pytest.mark.asyncio
    async def test_config_validation_integration(self, config_manager):
        """测试配置验证集成"""
        # 测试设置有效的性能配置
        await config_manager.set(
            "CUSTOM_POOL_SIZE",
            5,
            category=ConfigCategory.PERFORMANCE,
            level=ConfigLevel.SERVICE,
            description="Custom pool size",
            min_value=1,
            max_value=20
        )
        
        assert config_manager.get("CUSTOM_POOL_SIZE") == 5
        
        # 测试设置无效配置应该抛出异常
        with pytest.raises(ValueError):
            await config_manager.set(
                "INVALID_POOL_SIZE",
                25,  # 超出最大值
                category=ConfigCategory.PERFORMANCE,
                level=ConfigLevel.SERVICE,
                min_value=1,
                max_value=20
            )
    
    @pytest.mark.asyncio
    async def test_config_persistence_integration(self, config_manager):
        """测试配置持久化集成"""
        # 设置配置并持久化
        await config_manager.set(
            "PERSISTENT_CONFIG",
            "persistent_value",
            category=ConfigCategory.MONITORING,
            persist=True
        )
        
        # 验证Redis持久化调用
        assert config_manager.redis_client.hset.called
        assert config_manager.redis_client.publish.called
        
        # 验证本地缓存
        assert config_manager.get("PERSISTENT_CONFIG") == "persistent_value"


if __name__ == "__main__":
    pytest.main([__file__])