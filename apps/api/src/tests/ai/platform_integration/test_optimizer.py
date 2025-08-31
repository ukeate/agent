"""性能优化器测试"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from ai.platform_integration.optimizer import PerformanceOptimizer
from ai.platform_integration.models import PerformanceMetrics


@pytest.fixture
def optimizer_config():
    """优化器配置"""
    return {
        'cache': {
            'enabled': True,
            'redis_host': 'localhost',
            'redis_port': 6379,
            'default_ttl': 300
        }
    }


@pytest.fixture
def performance_optimizer(optimizer_config):
    """性能优化器实例"""
    with patch('redis.Redis'):
        optimizer = PerformanceOptimizer(optimizer_config)
        return optimizer


class TestPerformanceOptimizer:
    """性能优化器测试类"""

    def test_init(self, optimizer_config):
        """测试初始化"""
        with patch('redis.Redis') as mock_redis:
            optimizer = PerformanceOptimizer(optimizer_config)
            
            assert optimizer.config == optimizer_config
            assert optimizer.cache_config == optimizer_config['cache']
            assert isinstance(optimizer.metrics, dict)
            mock_redis.assert_called_once()

    def test_init_cache_disabled(self):
        """测试缓存禁用时的初始化"""
        config = {
            'cache': {
                'enabled': False
            }
        }
        
        optimizer = PerformanceOptimizer(config)
        assert not hasattr(optimizer, 'cache')

    @pytest.mark.asyncio
    async def test_optimize_system_performance(self, performance_optimizer):
        """测试系统性能优化"""
        with patch.object(performance_optimizer, '_analyze_performance_bottlenecks') as mock_analyze, \
             patch.object(performance_optimizer, '_optimize_database_connections') as mock_db, \
             patch.object(performance_optimizer, '_optimize_memory_usage') as mock_memory, \
             patch.object(performance_optimizer, '_optimize_caching') as mock_cache, \
             patch.object(performance_optimizer, '_optimize_async_processing') as mock_async:
            
            mock_analyze.return_value = {"bottlenecks": [], "recommendations": []}
            mock_db.return_value = {"status": "optimized"}
            mock_memory.return_value = {"status": "completed"}
            mock_cache.return_value = {"status": "optimized"}
            mock_async.return_value = {"status": "optimized"}
            
            result = await performance_optimizer.optimize_system_performance()
            
            assert result["status"] == "completed"
            assert len(result["optimizations"]) == 5
            assert all(opt["optimization"] in [
                "bottleneck_analysis", "database_connections", "memory_optimization",
                "caching", "async_processing"
            ] for opt in result["optimizations"])

    @pytest.mark.asyncio
    async def test_analyze_performance_bottlenecks(self, performance_optimizer):
        """测试性能瓶颈分析"""
        with patch('psutil.cpu_percent', return_value=85.0), \
             patch('psutil.cpu_count', return_value=8), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.swap_memory') as mock_swap, \
             patch('psutil.disk_io_counters') as mock_disk, \
             patch('psutil.net_io_counters') as mock_net, \
             patch('psutil.Process') as mock_process:
            
            # Mock memory info
            mock_memory.return_value = MagicMock()
            mock_memory.return_value.percent = 90.0
            mock_memory.return_value.available = 2 * 1024**3  # 2GB
            
            mock_swap.return_value = MagicMock()
            mock_swap.return_value.percent = 60.0
            
            # Mock disk I/O
            mock_disk.return_value = MagicMock()
            mock_disk.return_value.read_bytes = 1024**3  # 1GB
            mock_disk.return_value.write_bytes = 512 * 1024**2  # 512MB
            mock_disk.return_value.read_time = 15000  # 15 seconds
            mock_disk.return_value.write_time = 8000   # 8 seconds
            
            # Mock network I/O
            mock_net.return_value = MagicMock()
            mock_net.return_value.bytes_sent = 100 * 1024**2  # 100MB
            mock_net.return_value.bytes_recv = 200 * 1024**2  # 200MB
            
            # Mock process info
            mock_process_instance = MagicMock()
            mock_process_instance.cpu_percent.return_value = 25.0
            mock_process_instance.memory_percent.return_value = 15.0
            mock_process_instance.num_threads.return_value = 10
            mock_process_instance.num_fds.return_value = 50
            mock_process.return_value = mock_process_instance
            
            result = await performance_optimizer._analyze_performance_bottlenecks()
            
            assert result["cpu_percent"] == 85.0
            assert result["cpu_count"] == 8
            assert result["memory_percent"] == 90.0
            assert "high_cpu_usage" in result["bottlenecks"]
            assert "high_memory_usage" in result["bottlenecks"]
            assert "high_swap_usage" in result["bottlenecks"]
            assert "high_disk_io" in result["bottlenecks"]
            assert len(result["recommendations"]) >= 3

    @pytest.mark.asyncio
    async def test_optimize_database_connections(self, performance_optimizer):
        """测试数据库连接优化"""
        result = await performance_optimizer._optimize_database_connections()
        
        assert result["status"] == "optimized"
        assert "config" in result
        assert result["config"]["pool_size"] == 20
        assert result["config"]["max_overflow"] == 30
        assert result["config"]["pool_timeout"] == 30
        assert "query_optimizations" in result
        assert "improvements" in result

    @pytest.mark.asyncio
    async def test_optimize_memory_usage(self, performance_optimizer):
        """测试内存使用优化"""
        with patch('gc.collect', return_value=10), \
             patch('gc.get_stats', return_value=[{"collections": 5, "collected": 100}]):
            
            result = await performance_optimizer._optimize_memory_usage()
            
            assert result["status"] == "completed"
            assert result["garbage_collected"] == 10
            assert "gc_stats" in result
            assert "memory_strategies" in result
            assert result["memory_strategies"]["object_pooling"]["enabled"]

    @pytest.mark.asyncio
    async def test_optimize_caching_enabled(self, performance_optimizer):
        """测试缓存优化（启用缓存）"""
        result = await performance_optimizer._optimize_caching()
        
        assert result["status"] == "optimized"
        assert "strategies" in result
        assert "cache_warming" in result
        assert "invalidation_strategies" in result
        assert result["strategies"]["model_results"]["ttl"] == 3600

    @pytest.mark.asyncio
    async def test_optimize_caching_disabled(self):
        """测试缓存优化（禁用缓存）"""
        config = {'cache': {'enabled': False}}
        optimizer = PerformanceOptimizer(config)
        
        result = await optimizer._optimize_caching()
        
        assert result["status"] == "disabled"
        assert result["message"] == "Caching is disabled"

    @pytest.mark.asyncio
    async def test_optimize_async_processing(self, performance_optimizer):
        """测试异步处理优化"""
        result = await performance_optimizer._optimize_async_processing()
        
        assert result["status"] == "optimized"
        assert "config" in result
        assert result["config"]["max_concurrent_tasks"] == 100
        assert result["config"]["task_timeout"] == 300
        assert "retry_policy" in result["config"]
        assert "circuit_breaker" in result["config"]
        assert "strategies" in result

    @pytest.mark.asyncio
    async def test_collect_metrics(self, performance_optimizer):
        """测试收集性能指标"""
        with patch('psutil.cpu_percent', return_value=60.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_io_counters') as mock_disk, \
             patch('psutil.net_io_counters') as mock_net:
            
            mock_memory.return_value = MagicMock()
            mock_memory.return_value.percent = 70.0
            
            mock_disk.return_value = MagicMock()
            mock_disk.return_value.read_bytes = 1024**3
            mock_disk.return_value.write_bytes = 512 * 1024**2
            
            mock_net.return_value = MagicMock()
            mock_net.return_value.bytes_sent = 100 * 1024**2
            mock_net.return_value.bytes_recv = 200 * 1024**2
            
            metrics = await performance_optimizer.collect_metrics()
            
            assert isinstance(metrics, PerformanceMetrics)
            assert metrics.cpu_percent == 60.0
            assert metrics.memory_percent == 70.0
            assert metrics.disk_usage["read_bytes"] == 1024**3
            assert metrics.network_usage["bytes_sent"] == 100 * 1024**2
            assert metrics.bottlenecks == []  # No bottlenecks with these values

    @pytest.mark.asyncio
    async def test_collect_metrics_with_bottlenecks(self, performance_optimizer):
        """测试收集性能指标（有瓶颈）"""
        with patch('psutil.cpu_percent', return_value=85.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_io_counters', return_value=None), \
             patch('psutil.net_io_counters', return_value=None):
            
            mock_memory.return_value = MagicMock()
            mock_memory.return_value.percent = 90.0
            
            metrics = await performance_optimizer.collect_metrics()
            
            assert metrics.cpu_percent == 85.0
            assert metrics.memory_percent == 90.0
            assert "high_cpu" in metrics.bottlenecks
            assert "high_memory" in metrics.bottlenecks
            assert metrics.disk_usage["read_bytes"] == 0
            assert metrics.network_usage["bytes_sent"] == 0

    @pytest.mark.asyncio
    async def test_apply_optimization_profile_valid(self, performance_optimizer):
        """测试应用有效的优化配置文件"""
        result = await performance_optimizer.apply_optimization_profile("high_performance")
        
        assert result["status"] == "applied"
        assert result["profile"] == "high_performance"
        assert result["configuration"]["cache_ttl"] == 3600
        assert result["configuration"]["max_connections"] == 100
        assert performance_optimizer.config["cache_ttl"] == 3600

    @pytest.mark.asyncio
    async def test_apply_optimization_profile_invalid(self, performance_optimizer):
        """测试应用无效的优化配置文件"""
        result = await performance_optimizer.apply_optimization_profile("invalid_profile")
        
        assert result["status"] == "error"
        assert "Unknown profile" in result["message"]

    @pytest.mark.asyncio
    async def test_generate_performance_report(self, performance_optimizer):
        """测试生成性能报告"""
        mock_metrics = PerformanceMetrics(
            cpu_percent=75.0,
            memory_percent=80.0,
            disk_usage={"read_bytes": 1024**3, "write_bytes": 512 * 1024**2},
            network_usage={"bytes_sent": 100 * 1024**2, "bytes_recv": 200 * 1024**2},
            bottlenecks=["high_memory"],
            timestamp=datetime.now()
        )
        
        with patch.object(performance_optimizer, 'collect_metrics', return_value=mock_metrics), \
             patch.object(performance_optimizer, '_analyze_performance_bottlenecks') as mock_analyze:
            
            mock_analyze.return_value = {
                "bottlenecks": ["high_memory"],
                "recommendations": ["Increase memory allocation"]
            }
            
            report = await performance_optimizer.generate_performance_report()
            
            assert "timestamp" in report
            assert report["current_metrics"]["cpu_usage"] == 75.0
            assert report["current_metrics"]["memory_usage"] == 80.0
            assert report["bottlenecks"] == ["high_memory"]
            assert report["recommendations"] == ["Increase memory allocation"]
            assert "optimization_status" in report
            assert "performance_score" in report

    def test_calculate_performance_score_perfect(self, performance_optimizer):
        """测试计算完美性能评分"""
        metrics = PerformanceMetrics(
            cpu_percent=30.0,
            memory_percent=40.0,
            disk_usage={},
            network_usage={},
            bottlenecks=[],
            timestamp=datetime.now()
        )
        
        score = performance_optimizer._calculate_performance_score(metrics)
        assert score == 100.0

    def test_calculate_performance_score_with_issues(self, performance_optimizer):
        """测试计算有问题的性能评分"""
        metrics = PerformanceMetrics(
            cpu_percent=95.0,  # High CPU
            memory_percent=95.0,  # High memory
            disk_usage={},
            network_usage={},
            bottlenecks=["high_cpu", "high_memory", "high_disk_io"],  # 3 bottlenecks
            timestamp=datetime.now()
        )
        
        score = performance_optimizer._calculate_performance_score(metrics)
        # Should deduct: 30 (high CPU) + 30 (high memory) + 30 (3 bottlenecks * 10)
        expected_score = max(0, 100 - 30 - 30 - 30)
        assert score == expected_score

    def test_calculate_performance_score_medium_usage(self, performance_optimizer):
        """测试计算中等使用率的性能评分"""
        metrics = PerformanceMetrics(
            cpu_percent=60.0,  # Medium CPU
            memory_percent=60.0,  # Medium memory
            disk_usage={},
            network_usage={},
            bottlenecks=["some_bottleneck"],  # 1 bottleneck
            timestamp=datetime.now()
        )
        
        score = performance_optimizer._calculate_performance_score(metrics)
        # Should deduct: 5 (medium CPU) + 5 (medium memory) + 10 (1 bottleneck)
        expected_score = 100 - 5 - 5 - 10
        assert score == expected_score


class TestPerformanceOptimizerIntegration:
    """性能优化器集成测试"""

    @pytest.mark.asyncio
    async def test_full_optimization_cycle(self, optimizer_config):
        """测试完整的优化周期"""
        with patch('redis.Redis'), \
             patch('psutil.cpu_percent', return_value=75.0), \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.swap_memory') as mock_swap, \
             patch('psutil.disk_io_counters', return_value=None), \
             patch('psutil.net_io_counters', return_value=None), \
             patch('psutil.Process') as mock_process, \
             patch('gc.collect', return_value=5), \
             patch('gc.get_stats', return_value=[]):
            
            mock_memory.return_value = MagicMock()
            mock_memory.return_value.percent = 60.0
            mock_memory.return_value.available = 4 * 1024**3
            
            mock_swap.return_value = MagicMock()
            mock_swap.return_value.percent = 20.0
            
            mock_process_instance = MagicMock()
            mock_process_instance.cpu_percent.return_value = 15.0
            mock_process_instance.memory_percent.return_value = 10.0
            mock_process_instance.num_threads.return_value = 5
            mock_process_instance.num_fds.return_value = 25
            mock_process.return_value = mock_process_instance
            
            optimizer = PerformanceOptimizer(optimizer_config)
            
            # 运行完整优化
            result = await optimizer.optimize_system_performance()
            
            assert result["status"] == "completed"
            assert len(result["optimizations"]) == 5
            
            # 应用优化配置文件
            profile_result = await optimizer.apply_optimization_profile("balanced")
            assert profile_result["status"] == "applied"
            
            # 生成报告
            report = await optimizer.generate_performance_report()
            assert report["performance_score"] >= 0
            assert report["performance_score"] <= 100