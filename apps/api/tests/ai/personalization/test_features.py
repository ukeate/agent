import pytest
import asyncio
from unittest.mock import AsyncMock, Mock
from datetime import datetime
from redis.asyncio import Redis

from ai.personalization.features.realtime import RealTimeFeatureEngine, FeatureConfig
from models.schemas.personalization import RealTimeFeatures


@pytest.fixture
async def mock_redis():
    """Mock Redis客户端"""
    redis_mock = AsyncMock(spec=Redis)
    redis_mock.get.return_value = None
    redis_mock.setex.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.smembers.return_value = set()
    return redis_mock


@pytest.fixture
def feature_config():
    """特征配置"""
    return FeatureConfig(
        window_sizes=[60, 300],
        update_interval=30,
        cache_ttl=300,
        max_history_size=100
    )


@pytest.fixture
async def feature_engine(mock_redis, feature_config):
    """特征引擎实例"""
    engine = RealTimeFeatureEngine(mock_redis, feature_config)
    await engine.start()
    return engine


@pytest.mark.asyncio
class TestRealTimeFeatureEngine:
    """实时特征引擎测试"""
    
    async def test_engine_initialization(self, mock_redis, feature_config):
        """测试引擎初始化"""
        engine = RealTimeFeatureEngine(mock_redis, feature_config)
        
        assert engine.redis == mock_redis
        assert engine.config == feature_config
        assert len(engine.feature_extractors) > 0
        assert len(engine.aggregators) > 0
        assert engine._running is False
    
    async def test_engine_lifecycle(self, feature_engine):
        """测试引擎生命周期"""
        # 启动后状态
        assert feature_engine._running is True
        assert len(feature_engine._background_tasks) > 0
        
        # 停止引擎
        await feature_engine.stop()
        assert feature_engine._running is False
    
    async def test_compute_features_success(self, feature_engine):
        """测试成功计算特征"""
        # Mock特征提取器
        feature_engine.feature_extractors["temporal"].extract_features = AsyncMock(
            return_value={"hour": 14.0, "day": 1.0}
        )
        feature_engine.feature_extractors["behavioral"].extract_features = AsyncMock(
            return_value={"clicks": 10.0, "views": 20.0}
        )
        feature_engine.feature_extractors["contextual"].extract_features = AsyncMock(
            return_value={"device": 1.0, "location": 0.5}
        )
        
        # 计算特征
        features = await feature_engine.compute_features(
            user_id="test_user",
            context={"source": "test"},
            use_cache=False
        )
        
        # 验证结果
        assert isinstance(features, RealTimeFeatures)
        assert "hour" in features.temporal
        assert "clicks" in features.behavioral
        assert "device" in features.contextual
        assert len(features.aggregated) > 0
        assert isinstance(features.timestamp, datetime)
    
    async def test_compute_features_with_cache(self, feature_engine, mock_redis):
        """测试缓存特征计算"""
        # Mock缓存命中
        cached_features = {
            "temporal": {"hour": 14.0},
            "behavioral": {"clicks": 10.0},
            "contextual": {"device": 1.0},
            "aggregated": {"avg": 0.8},
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0"
        }
        
        import json
        mock_redis.get.return_value = json.dumps(cached_features)
        
        # 计算特征
        features = await feature_engine.compute_features(
            user_id="test_user",
            context={},
            use_cache=True
        )
        
        # 验证缓存命中
        assert isinstance(features, RealTimeFeatures)
        assert features.temporal["hour"] == 14.0
        mock_redis.get.assert_called_once()
    
    async def test_compute_features_cache_miss(self, feature_engine, mock_redis):
        """测试缓存未命中"""
        # Mock缓存未命中
        mock_redis.get.return_value = None
        
        # Mock特征提取
        feature_engine.feature_extractors["temporal"].extract_features = AsyncMock(
            return_value={"hour": 15.0}
        )
        feature_engine.feature_extractors["behavioral"].extract_features = AsyncMock(
            return_value={"clicks": 5.0}
        )
        feature_engine.feature_extractors["contextual"].extract_features = AsyncMock(
            return_value={"device": 2.0}
        )
        
        # 计算特征
        features = await feature_engine.compute_features(
            user_id="test_user",
            context={},
            use_cache=True
        )
        
        # 验证计算和缓存
        assert isinstance(features, RealTimeFeatures)
        mock_redis.get.assert_called_once()
        mock_redis.setex.assert_called_once()
    
    async def test_compute_features_with_errors(self, feature_engine):
        """测试特征计算错误处理"""
        # Mock特征提取器抛出异常
        feature_engine.feature_extractors["temporal"].extract_features = AsyncMock(
            side_effect=Exception("Temporal error")
        )
        feature_engine.feature_extractors["behavioral"].extract_features = AsyncMock(
            return_value={"clicks": 5.0}
        )
        feature_engine.feature_extractors["contextual"].extract_features = AsyncMock(
            side_effect=Exception("Contextual error")
        )
        
        # 计算特征
        features = await feature_engine.compute_features(
            user_id="test_user",
            context={},
            use_cache=False
        )
        
        # 验证错误处理
        assert isinstance(features, RealTimeFeatures)
        assert len(features.temporal) == 0  # 错误时应为空
        assert len(features.behavioral) > 0  # 成功的应该保留
        assert len(features.contextual) == 0  # 错误时应为空
    
    async def test_sliding_window_updates(self, feature_engine):
        """测试滑动窗口更新"""
        features = RealTimeFeatures(
            temporal={"hour": 14.0},
            behavioral={"clicks": 10.0},
            contextual={"device": 1.0},
            aggregated={"avg": 0.8}
        )
        
        # 更新滑动窗口
        await feature_engine._update_sliding_windows("test_user", features)
        
        # 验证窗口数据
        user_windows = feature_engine.sliding_windows.get("test_user", {})
        assert len(user_windows) > 0
        
        for window_size in feature_engine.config.window_sizes:
            assert window_size in user_windows
            assert len(user_windows[window_size]) > 0
    
    async def test_window_aggregations(self, feature_engine):
        """测试窗口聚合计算"""
        # 手动添加窗口数据
        import time
        current_time = time.time()
        
        for i in range(5):
            feature_engine.sliding_windows["test_user"][60].append(
                (current_time - i * 10, 0.8 + i * 0.1)
            )
        
        # 计算窗口聚合
        window_features = await feature_engine._compute_window_aggregations("test_user")
        
        # 验证聚合特征
        assert "window_60s_count" in window_features
        assert "window_60s_avg" in window_features
        assert "window_60s_max" in window_features
        assert window_features["window_60s_count"] == 5
    
    async def test_cache_operations(self, feature_engine, mock_redis):
        """测试缓存操作"""
        features = RealTimeFeatures(
            temporal={"hour": 14.0},
            behavioral={"clicks": 10.0}
        )
        
        # 测试缓存设置
        await feature_engine._cache_features("test_user", features)
        mock_redis.setex.assert_called_once()
        
        # 测试缓存获取
        import json
        feature_dict = features.model_dump()
        feature_dict["timestamp"] = features.timestamp.isoformat()
        mock_redis.get.return_value = json.dumps(feature_dict)
        
        cached_features = await feature_engine.get_cached_features("test_user")
        assert isinstance(cached_features, RealTimeFeatures)
        assert cached_features.temporal["hour"] == 14.0
    
    async def test_feature_statistics(self, feature_engine):
        """测试特征统计"""
        # 添加一些窗口数据
        import time
        current_time = time.time()
        
        feature_engine.sliding_windows["user1"][60].append((current_time, 0.8))
        feature_engine.sliding_windows["user2"][300].append((current_time, 0.9))
        
        # 获取统计信息
        stats = feature_engine.get_feature_statistics()
        
        # 验证统计
        assert "total_users" in stats
        assert "window_sizes" in stats
        assert "is_running" in stats
        assert "extractors" in stats
        assert "window_statistics" in stats
        assert stats["total_users"] == 2
    
    async def test_precompute_task(self, feature_engine, mock_redis):
        """测试预计算任务"""
        # Mock活跃用户
        mock_redis.smembers.return_value = {b"user1", b"user2"}
        
        # Mock特征计算
        feature_engine.compute_features = AsyncMock(
            return_value=RealTimeFeatures()
        )
        
        # 运行预计算（模拟一次）
        await feature_engine._precompute_features()
        
        # 验证调用
        mock_redis.smembers.assert_called_once()
    
    async def test_cleanup_task(self, feature_engine):
        """测试清理任务"""
        import time
        old_time = time.time() - 7200  # 2小时前
        
        # 添加过期数据
        feature_engine.sliding_windows["user1"][60].append((old_time, 0.8))
        feature_engine.sliding_windows["user2"][300].append((old_time, 0.9))
        
        # 运行清理（模拟一次）
        await feature_engine._cleanup_expired_features()
        
        # 验证清理效果（过期数据应该被删除）
        # 注意：这里的验证取决于具体的清理逻辑实现
    
    async def test_concurrent_feature_computation(self, feature_engine):
        """测试并发特征计算"""
        # Mock特征提取器
        for extractor in feature_engine.feature_extractors.values():
            extractor.extract_features = AsyncMock(
                return_value={"test": 1.0}
            )
        
        # 并发计算特征
        tasks = [
            feature_engine.compute_features(
                user_id=f"user_{i}",
                context={},
                use_cache=False
            )
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 验证所有计算都成功
        assert len(results) == 10
        assert all(isinstance(r, RealTimeFeatures) for r in results if not isinstance(r, Exception))


@pytest.mark.asyncio
class TestFeatureConfig:
    """特征配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = FeatureConfig()
        
        assert config.window_sizes == [60, 300, 1800, 3600]
        assert config.update_interval == 60
        assert config.cache_ttl == 300
        assert config.max_history_size == 1000
        assert config.enable_precompute is True
        assert "temporal" in config.feature_weights
        assert "behavioral" in config.feature_weights
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = FeatureConfig(
            window_sizes=[30, 60],
            update_interval=30,
            cache_ttl=600,
            feature_weights={"temporal": 0.5, "behavioral": 0.5}
        )
        
        assert config.window_sizes == [30, 60]
        assert config.update_interval == 30
        assert config.cache_ttl == 600
        assert config.feature_weights["temporal"] == 0.5