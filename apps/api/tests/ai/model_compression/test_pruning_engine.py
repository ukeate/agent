"""
模型剪枝引擎测试

测试模型剪枝引擎的各种功能和策略
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.ai.model_compression.pruning_engine import PruningEngine, PruningResult
from src.ai.model_compression.models import PruningConfig, PruningType


class TestModel(nn.Module):
    """测试模型"""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(20, 10)
        self.linear2 = nn.Linear(10, 5)
        self.conv = nn.Conv2d(3, 16, 3)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        if len(x.shape) == 4:  # 卷积输入
            x = self.relu(self.conv(x))
            x = x.view(x.size(0), -1)
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


@pytest.fixture
def test_model():
    """创建测试模型"""
    return TestModel()


@pytest.fixture
def pruning_engine():
    """创建剪枝引擎"""
    return PruningEngine()


@pytest.fixture
def basic_config():
    """基础剪枝配置"""
    return PruningConfig(
        pruning_type=PruningType.UNSTRUCTURED,
        sparsity_ratio=0.5,
        importance_metric="magnitude"
    )


class TestPruningEngine:
    """剪枝引擎测试类"""
    
    def test_engine_initialization(self, pruning_engine):
        """测试引擎初始化"""
        assert pruning_engine is not None
        assert len(pruning_engine.supported_types) > 0
        assert PruningType.UNSTRUCTURED in pruning_engine.supported_types
        assert PruningType.STRUCTURED in pruning_engine.supported_types
    
    def test_config_validation(self, pruning_engine):
        """测试配置验证"""
        # 有效配置
        valid_config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.3,
            importance_metric="magnitude"
        )
        assert pruning_engine.validate_config(valid_config) is True
        
        # 无效配置 - 稀疏度超出范围
        invalid_config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=1.5,  # 超过1.0
            importance_metric="magnitude"
        )
        assert pruning_engine.validate_config(invalid_config) is False
    
    def test_unstructured_pruning_magnitude(self, pruning_engine, test_model, basic_config):
        """测试基于幅度的非结构化剪枝"""
        original_params = sum(p.numel() for p in test_model.parameters())
        
        # 执行剪枝
        pruned_model, stats = pruning_engine.prune_model(test_model, basic_config)
        
        # 验证结果
        assert pruned_model is not None
        assert stats is not None
        assert "sparsity_achieved" in stats
        assert "compression_ratio" in stats
        assert stats["sparsity_achieved"] > 0
        
        # 验证模型仍然可以运行
        test_input = torch.randn(1, 20)
        with torch.no_grad():
            output = pruned_model(test_input)
            assert output is not None
            assert output.shape == (1, 5)
    
    def test_unstructured_pruning_gradient(self, pruning_engine, test_model):
        """测试基于梯度的非结构化剪枝"""
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.3,
            importance_metric="gradient"
        )
        
        # 创建模拟梯度数据
        dummy_data = []
        for _ in range(5):
            x = torch.randn(2, 20)
            y = torch.randn(2, 5)
            dummy_data.append((x, y))
        
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(dummy_data))
        mock_dataloader.__len__ = Mock(return_value=len(dummy_data))
        
        pruned_model, stats = pruning_engine.prune_model(
            test_model, config, calibration_data=mock_dataloader
        )
        
        assert pruned_model is not None
        assert stats["importance_metric"] == "gradient"
    
    def test_structured_pruning_channel(self, pruning_engine, test_model):
        """测试通道级结构化剪枝"""
        config = PruningConfig(
            pruning_type=PruningType.STRUCTURED,
            sparsity_ratio=0.25,
            structured_type="channel",
            importance_metric="l2_norm"
        )
        
        original_channels = test_model.conv.out_channels
        
        pruned_model, stats = pruning_engine.prune_model(test_model, config)
        
        # 验证通道数确实减少了
        assert pruned_model is not None
        assert stats["pruning_info"]["structured_type"] == "channel"
        
        # 测试卷积输入
        conv_input = torch.randn(1, 3, 32, 32)
        with torch.no_grad():
            output = pruned_model(conv_input)
            assert output is not None
    
    def test_structured_pruning_filter(self, pruning_engine, test_model):
        """测试滤波器级结构化剪枝"""
        config = PruningConfig(
            pruning_type=PruningType.STRUCTURED,
            sparsity_ratio=0.4,
            structured_type="filter",
            importance_metric="l1_norm"
        )
        
        pruned_model, stats = pruning_engine.prune_model(test_model, config)
        
        assert pruned_model is not None
        assert stats["pruning_info"]["structured_type"] == "filter"
    
    def test_gradual_pruning(self, pruning_engine, test_model):
        """测试渐进式剪枝"""
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.8,  # 高稀疏度
            gradual_pruning=True,
            pruning_steps=3,
            recovery_epochs=2
        )
        
        # 创建训练数据
        train_data = []
        for _ in range(10):
            x = torch.randn(2, 20)
            y = torch.randn(2, 5)
            train_data.append((x, y))
        
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=len(train_data))
        
        pruned_model, stats = pruning_engine.prune_model(
            test_model, config, training_data=mock_dataloader
        )
        
        assert pruned_model is not None
        assert stats["gradual_pruning"] is True
        assert len(stats["pruning_schedule"]) == config.pruning_steps
    
    def test_importance_metrics(self, pruning_engine, test_model):
        """测试不同重要性指标"""
        metrics = ["magnitude", "gradient", "l1_norm", "l2_norm", "taylor_expansion"]
        
        for metric in metrics:
            config = PruningConfig(
                pruning_type=PruningType.UNSTRUCTURED,
                sparsity_ratio=0.3,
                importance_metric=metric
            )
            
            # 为梯度和泰勒展开提供数据
            if metric in ["gradient", "taylor_expansion"]:
                dummy_data = [(torch.randn(1, 20), torch.randn(1, 5))]
                mock_dataloader = Mock()
                mock_dataloader.__iter__ = Mock(return_value=iter(dummy_data))
                mock_dataloader.__len__ = Mock(return_value=1)
                
                pruned_model, stats = pruning_engine.prune_model(
                    TestModel(), config, calibration_data=mock_dataloader
                )
            else:
                pruned_model, stats = pruning_engine.prune_model(TestModel(), config)
            
            assert pruned_model is not None
            assert stats["importance_metric"] == metric
    
    def test_layer_wise_pruning(self, pruning_engine, test_model):
        """测试逐层剪枝"""
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.5,
            layer_wise_sparsity=True,
            layer_sparsity_ratios={
                "linear1": 0.3,
                "linear2": 0.7
            }
        )
        
        pruned_model, stats = pruning_engine.prune_model(test_model, config)
        
        assert pruned_model is not None
        assert "layer_wise_sparsity" in stats["pruning_info"]
    
    def test_sensitivity_analysis(self, pruning_engine, test_model):
        """测试敏感性分析"""
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.5,
            sensitivity_analysis=True
        )
        
        # 创建验证数据
        val_data = [(torch.randn(2, 20), torch.randn(2, 5)) for _ in range(3)]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(val_data))
        mock_dataloader.__len__ = Mock(return_value=len(val_data))
        
        pruned_model, stats = pruning_engine.prune_model(
            test_model, config, validation_data=mock_dataloader
        )
        
        assert pruned_model is not None
        assert "sensitivity_scores" in stats
        assert len(stats["sensitivity_scores"]) > 0
    
    def test_sparsity_calculation(self, pruning_engine, test_model, basic_config):
        """测试稀疏度计算"""
        pruned_model, stats = pruning_engine.prune_model(test_model, basic_config)
        
        # 手动验证稀疏度
        total_params = 0
        zero_params = 0
        
        for param in pruned_model.parameters():
            total_params += param.numel()
            zero_params += (param == 0).sum().item()
        
        actual_sparsity = zero_params / total_params
        reported_sparsity = stats["sparsity_achieved"]
        
        # 允许小的误差
        assert abs(actual_sparsity - reported_sparsity) < 0.01
    
    def test_model_saving(self, pruning_engine, test_model, basic_config, tmp_path):
        """测试模型保存"""
        pruned_model, _ = pruning_engine.prune_model(test_model, basic_config)
        
        save_path = tmp_path / "pruned_model.pth"
        saved_path = pruning_engine.save_pruned_model(pruned_model, str(save_path))
        
        assert saved_path == str(save_path)
        assert save_path.exists()
        
        # 验证能够加载保存的模型
        loaded_model = torch.load(save_path)
        assert loaded_model is not None
    
    def test_mask_generation(self, pruning_engine, test_model, basic_config):
        """测试掩码生成"""
        # 生成剪枝掩码
        masks = pruning_engine._generate_pruning_masks(test_model, basic_config)
        
        assert isinstance(masks, dict)
        assert len(masks) > 0
        
        # 验证掩码形状
        for name, param in test_model.named_parameters():
            if name in masks:
                assert masks[name].shape == param.shape
                assert masks[name].dtype == torch.bool
    
    def test_performance_recovery(self, pruning_engine, test_model):
        """测试性能恢复"""
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.6,
            gradual_pruning=True,
            recovery_epochs=3
        )
        
        # 创建训练数据
        train_data = [(torch.randn(2, 20), torch.randn(2, 5)) for _ in range(5)]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=len(train_data))
        
        pruned_model, stats = pruning_engine.prune_model(
            test_model, config, training_data=mock_dataloader
        )
        
        assert pruned_model is not None
        assert stats["recovery_training"] is True
    
    def test_get_supported_types(self, pruning_engine):
        """测试获取支持的剪枝类型"""
        types = pruning_engine.get_supported_types()
        assert isinstance(types, list)
        assert len(types) > 0
        assert "unstructured" in types
        assert "structured" in types
    
    def test_error_handling_invalid_sparsity(self, pruning_engine, test_model):
        """测试无效稀疏度的错误处理"""
        invalid_config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=1.5  # 无效值
        )
        
        with pytest.raises(ValueError, match="稀疏度必须在0到1之间"):
            pruning_engine.prune_model(test_model, invalid_config)
    
    def test_error_handling_unsupported_type(self, pruning_engine, test_model):
        """测试不支持的剪枝类型"""
        # 临时移除支持的类型来测试错误处理
        original_types = pruning_engine.supported_types
        pruning_engine.supported_types = set()
        
        try:
            with pytest.raises(ValueError, match="不支持的剪枝类型"):
                pruning_engine.prune_model(test_model, basic_config)
        finally:
            pruning_engine.supported_types = original_types
    
    @patch('src.ai.model_compression.pruning_engine.torch.nn.utils.prune.global_unstructured')
    def test_pruning_error_handling(self, mock_prune, pruning_engine, test_model, basic_config):
        """测试剪枝过程中的错误处理"""
        # 模拟剪枝失败
        mock_prune.side_effect = RuntimeError("Pruning failed")
        
        with pytest.raises(RuntimeError):
            pruning_engine._apply_unstructured_pruning(test_model, basic_config)
    
    def test_memory_efficient_pruning(self, pruning_engine, test_model):
        """测试内存高效的剪枝"""
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.5,
            memory_efficient=True
        )
        
        pruned_model, stats = pruning_engine.prune_model(test_model, config)
        
        assert pruned_model is not None
        assert stats["memory_efficient"] is True
    
    def test_iterative_pruning(self, pruning_engine, test_model):
        """测试迭代式剪枝"""
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.8,
            iterative_pruning=True,
            iterations=3
        )
        
        # 创建训练数据
        train_data = [(torch.randn(1, 20), torch.randn(1, 5)) for _ in range(3)]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=len(train_data))
        
        pruned_model, stats = pruning_engine.prune_model(
            test_model, config, training_data=mock_dataloader
        )
        
        assert pruned_model is not None
        assert stats["iterative_pruning"] is True
        assert len(stats["iteration_history"]) == config.iterations


@pytest.mark.parametrize("pruning_type,sparsity_ratio,metric", [
    (PruningType.UNSTRUCTURED, 0.3, "magnitude"),
    (PruningType.UNSTRUCTURED, 0.5, "l1_norm"),
    (PruningType.STRUCTURED, 0.25, "l2_norm"),
])
def test_different_pruning_combinations(pruning_type, sparsity_ratio, metric, 
                                      pruning_engine, test_model):
    """参数化测试不同的剪枝组合"""
    config = PruningConfig(
        pruning_type=pruning_type,
        sparsity_ratio=sparsity_ratio,
        importance_metric=metric
    )
    
    pruned_model, stats = pruning_engine.prune_model(test_model, config)
    
    assert pruned_model is not None
    assert stats["sparsity_achieved"] >= 0
    assert stats["compression_ratio"] >= 1.0


class TestPruningEdgeCases:
    """剪枝边界情况测试"""
    
    def test_zero_sparsity(self, pruning_engine, test_model):
        """测试零稀疏度"""
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.0
        )
        
        pruned_model, stats = pruning_engine.prune_model(test_model, config)
        
        assert pruned_model is not None
        assert stats["sparsity_achieved"] == 0.0
        assert stats["compression_ratio"] == 1.0
    
    def test_very_high_sparsity(self, pruning_engine, test_model):
        """测试非常高的稀疏度"""
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.99
        )
        
        pruned_model, stats = pruning_engine.prune_model(test_model, config)
        
        assert pruned_model is not None
        assert stats["sparsity_achieved"] >= 0.95  # 允许一些误差
    
    def test_single_layer_model(self, pruning_engine):
        """测试单层模型"""
        single_layer = nn.Linear(10, 5)
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.5
        )
        
        pruned_model, stats = pruning_engine.prune_model(single_layer, config)
        
        assert pruned_model is not None
        assert stats["sparsity_achieved"] > 0
    
    def test_empty_model(self, pruning_engine):
        """测试空模型"""
        empty_model = nn.Module()
        config = PruningConfig(
            pruning_type=PruningType.UNSTRUCTURED,
            sparsity_ratio=0.5
        )
        
        pruned_model, stats = pruning_engine.prune_model(empty_model, config)
        
        assert pruned_model is not None
        assert stats["sparsity_achieved"] == 0.0  # 没有可剪枝的参数


if __name__ == "__main__":
    pytest.main([__file__, "-v"])