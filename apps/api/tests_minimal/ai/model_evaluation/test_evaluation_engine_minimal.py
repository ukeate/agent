"""
模型评估引擎的轻量级测试
避免重依赖项导入，专注测试核心逻辑
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from typing import Dict, Any

# 添加src路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

# 模拟重依赖项
sys.modules['torch'] = Mock()
sys.modules['transformers'] = Mock()
sys.modules['datasets'] = Mock()
sys.modules['evaluate'] = Mock()
sys.modules['lm_eval'] = Mock()
sys.modules['lm_eval.api'] = Mock()
sys.modules['lm_eval.models'] = Mock()
sys.modules['lm_eval.tasks'] = Mock()
sys.modules['lm_eval.evaluator'] = Mock()

class TestEvaluationEngineMinimal:
    """评估引擎轻量级测试"""
    
    def setup_method(self):
        """测试前设置"""
        # 模拟导入评估引擎
        from ai.model_evaluation.evaluation_engine import EvaluationConfig
        self.config = EvaluationConfig(
            device="cpu",
            batch_size=1,
            max_concurrent_evaluations=1
        )
    
    def test_evaluation_config_creation(self):
        """测试评估配置创建"""
        assert self.config.device == "cpu"
        assert self.config.batch_size == 1
        assert self.config.max_concurrent_evaluations == 1
        assert self.config.enable_caching is True
        assert self.config.cache_dir == "cache/evaluations"
    
    def test_evaluation_config_validation(self):
        """测试评估配置验证"""
        from ai.model_evaluation.evaluation_engine import EvaluationConfig
        
        # 测试有效配置
        valid_config = EvaluationConfig(batch_size=8, max_seq_length=512)
        assert valid_config.batch_size == 8
        assert valid_config.max_seq_length == 512
        
        # 测试默认值
        default_config = EvaluationConfig()
        assert default_config.device == "auto"
        assert default_config.batch_size == 4
        assert default_config.temperature == 0.0
    
    @patch('ai.model_evaluation.evaluation_engine.ModelEvaluationEngine._initialize_model')
    def test_evaluation_engine_initialization(self, mock_init_model):
        """测试评估引擎初始化"""
        mock_init_model.return_value = None
        
        from ai.model_evaluation.evaluation_engine import ModelEvaluationEngine
        
        engine = ModelEvaluationEngine(self.config)
        assert engine.config == self.config
        assert engine.current_jobs == {}
        assert engine.job_history == []
        assert engine.is_running is False
        mock_init_model.assert_called_once()
    
    @patch('ai.model_evaluation.evaluation_engine.ModelEvaluationEngine._initialize_model')
    def test_job_id_generation(self, mock_init_model):
        """测试任务ID生成"""
        mock_init_model.return_value = None
        
        from ai.model_evaluation.evaluation_engine import ModelEvaluationEngine
        
        engine = ModelEvaluationEngine(self.config)
        job_id1 = engine._generate_job_id("test_model", "test_benchmark")
        job_id2 = engine._generate_job_id("test_model", "test_benchmark")
        
        assert job_id1 != job_id2
        assert "test_model" in job_id1
        assert "test_benchmark" in job_id1
    
    @patch('ai.model_evaluation.evaluation_engine.ModelEvaluationEngine._initialize_model')
    def test_job_status_management(self, mock_init_model):
        """测试任务状态管理"""
        mock_init_model.return_value = None
        
        from ai.model_evaluation.evaluation_engine import ModelEvaluationEngine, EvaluationStatus
        
        engine = ModelEvaluationEngine(self.config)
        job_id = "test_job_123"
        
        # 测试任务不存在
        assert engine.get_job_status(job_id) is None
        
        # 创建模拟任务
        engine.current_jobs[job_id] = {
            'status': EvaluationStatus.PENDING,
            'model_name': 'test_model',
            'benchmark_name': 'test_benchmark'
        }
        
        # 测试获取状态
        status = engine.get_job_status(job_id)
        assert status == EvaluationStatus.PENDING
        
        # 测试更新状态
        engine._update_job_status(job_id, EvaluationStatus.RUNNING)
        assert engine.get_job_status(job_id) == EvaluationStatus.RUNNING
    
    @patch('ai.model_evaluation.evaluation_engine.ModelEvaluationEngine._initialize_model')
    @pytest.mark.asyncio
    async def test_stop_evaluation_job(self, mock_init_model):
        """测试停止评估任务"""
        mock_init_model.return_value = None
        
        from ai.model_evaluation.evaluation_engine import ModelEvaluationEngine, EvaluationStatus
        
        engine = ModelEvaluationEngine(self.config)
        job_id = "test_job_456"
        
        # 创建运行中的任务
        engine.current_jobs[job_id] = {
            'status': EvaluationStatus.RUNNING,
            'task': AsyncMock(),
            'model_name': 'test_model',
            'benchmark_name': 'test_benchmark'
        }
        
        # 停止任务
        result = await engine.stop_evaluation(job_id)
        
        assert result is True
        assert engine.get_job_status(job_id) == EvaluationStatus.CANCELLED
    
    @patch('ai.model_evaluation.evaluation_engine.ModelEvaluationEngine._initialize_model')
    def test_evaluation_metrics_aggregation(self, mock_init_model):
        """测试评估指标聚合"""
        mock_init_model.return_value = None
        
        from ai.model_evaluation.evaluation_engine import ModelEvaluationEngine
        
        engine = ModelEvaluationEngine(self.config)
        
        # 模拟原始结果
        raw_results = {
            'task1': {'accuracy': 0.85, 'f1': 0.82},
            'task2': {'accuracy': 0.90, 'f1': 0.88},
            'task3': {'accuracy': 0.78, 'f1': 0.75}
        }
        
        aggregated = engine._aggregate_results(raw_results)
        
        assert 'overall_accuracy' in aggregated
        assert 'overall_f1' in aggregated
        assert 'task_results' in aggregated
        assert abs(aggregated['overall_accuracy'] - 0.843333) < 0.001
        assert abs(aggregated['overall_f1'] - 0.816667) < 0.001
    
    @patch('ai.model_evaluation.evaluation_engine.ModelEvaluationEngine._initialize_model')
    def test_resource_limits_validation(self, mock_init_model):
        """测试资源限制验证"""
        mock_init_model.return_value = None
        
        from ai.model_evaluation.evaluation_engine import ModelEvaluationEngine
        
        engine = ModelEvaluationEngine(self.config)
        
        # 测试并发限制
        assert engine._check_resource_availability() is True
        
        # 模拟达到并发限制
        for i in range(self.config.max_concurrent_evaluations):
            engine.current_jobs[f'job_{i}'] = {'status': 'running'}
        
        assert engine._check_resource_availability() is False
    
    def test_error_handling_configuration(self):
        """测试错误处理配置"""
        from ai.model_evaluation.evaluation_engine import EvaluationConfig
        
        config = EvaluationConfig(
            max_retries=3,
            retry_delay_seconds=5.0,
            timeout_seconds=3600
        )
        
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 5.0
        assert config.timeout_seconds == 3600
    
    @patch('ai.model_evaluation.evaluation_engine.ModelEvaluationEngine._initialize_model')
    def test_cleanup_completed_jobs(self, mock_init_model):
        """测试清理已完成任务"""
        mock_init_model.return_value = None
        
        from ai.model_evaluation.evaluation_engine import ModelEvaluationEngine, EvaluationStatus
        
        engine = ModelEvaluationEngine(self.config)
        
        # 添加不同状态的任务
        engine.current_jobs.update({
            'completed_job': {'status': EvaluationStatus.COMPLETED},
            'failed_job': {'status': EvaluationStatus.FAILED},
            'running_job': {'status': EvaluationStatus.RUNNING},
            'cancelled_job': {'status': EvaluationStatus.CANCELLED}
        })
        
        # 清理已完成任务
        cleaned_count = engine._cleanup_completed_jobs()
        
        assert cleaned_count == 3  # completed, failed, cancelled
        assert 'running_job' in engine.current_jobs
        assert len(engine.current_jobs) == 1

class TestEvaluationConfigValidation:
    """评估配置验证测试"""
    
    def test_device_validation(self):
        """测试设备配置验证"""
        from ai.model_evaluation.evaluation_engine import EvaluationConfig
        
        # 有效设备配置
        valid_devices = ["cpu", "cuda", "cuda:0", "auto"]
        for device in valid_devices:
            config = EvaluationConfig(device=device)
            assert config.device == device
    
    def test_batch_size_validation(self):
        """测试批次大小验证"""
        from ai.model_evaluation.evaluation_engine import EvaluationConfig
        
        # 有效批次大小
        valid_batch_sizes = [1, 2, 4, 8, 16, 32]
        for batch_size in valid_batch_sizes:
            config = EvaluationConfig(batch_size=batch_size)
            assert config.batch_size == batch_size
    
    def test_temperature_validation(self):
        """测试温度参数验证"""
        from ai.model_evaluation.evaluation_engine import EvaluationConfig
        
        # 有效温度值
        valid_temperatures = [0.0, 0.1, 0.5, 1.0, 1.5]
        for temp in valid_temperatures:
            config = EvaluationConfig(temperature=temp)
            assert config.temperature == temp
    
    def test_cache_configuration(self):
        """测试缓存配置"""
        from ai.model_evaluation.evaluation_engine import EvaluationConfig
        
        config = EvaluationConfig(
            enable_caching=True,
            cache_dir="custom_cache",
            cache_expiry_hours=48
        )
        
        assert config.enable_caching is True
        assert config.cache_dir == "custom_cache"
        assert config.cache_expiry_hours == 48

if __name__ == "__main__":
    pytest.main([__file__])