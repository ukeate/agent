"""
知识蒸馏训练器测试

测试知识蒸馏训练器的各种功能和策略
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from src.ai.model_compression.distillation_trainer import DistillationTrainer, DistillationResult
from src.ai.model_compression.models import DistillationConfig


class TeacherModel(nn.Module):
    """教师模型"""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class StudentModel(nn.Module):
    """学生模型"""
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 10)
        self.linear2 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.linear2(x)
        return x


@pytest.fixture
def teacher_model():
    """创建教师模型"""
    return TeacherModel()


@pytest.fixture
def student_model():
    """创建学生模型"""
    return StudentModel()


@pytest.fixture
def distillation_trainer():
    """创建蒸馏训练器"""
    return DistillationTrainer()


@pytest.fixture
def basic_config(teacher_model, student_model):
    """基础蒸馏配置"""
    return DistillationConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        temperature=3.0,
        alpha=0.5,
        num_epochs=2,
        batch_size=4
    )


class TestDistillationTrainer:
    """蒸馏训练器测试类"""
    
    def test_trainer_initialization(self, distillation_trainer):
        """测试训练器初始化"""
        assert distillation_trainer is not None
        assert distillation_trainer.device is not None
        assert len(distillation_trainer.supported_strategies) > 0
    
    def test_config_validation(self, distillation_trainer, teacher_model, student_model):
        """测试配置验证"""
        # 有效配置
        valid_config = DistillationConfig(
            teacher_model=teacher_model,
            student_model=student_model,
            temperature=3.0,
            alpha=0.5
        )
        assert distillation_trainer.validate_config(valid_config) is True
        
        # 无效配置 - 缺少教师模型
        invalid_config = DistillationConfig(
            teacher_model=None,
            student_model=student_model,
            temperature=3.0,
            alpha=0.5
        )
        assert distillation_trainer.validate_config(invalid_config) is False
    
    def test_response_based_distillation(self, distillation_trainer, basic_config):
        """测试响应式蒸馏"""
        # 创建训练数据
        train_data = []
        for _ in range(5):
            x = torch.randn(2, 10)
            y = torch.randint(0, 5, (2,))
            train_data.append((x, y))
        
        # 模拟数据加载器
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=len(train_data))
        
        # 执行蒸馏
        result = distillation_trainer.distill(
            basic_config, 
            train_dataloader=mock_dataloader
        )
        
        assert isinstance(result, DistillationResult)
        assert result.student_model is not None
        assert len(result.training_history) > 0
        assert result.final_accuracy >= 0
    
    def test_feature_based_distillation(self, distillation_trainer, teacher_model, student_model):
        """测试特征式蒸馏"""
        config = DistillationConfig(
            teacher_model=teacher_model,
            student_model=student_model,
            distillation_type="feature_based",
            temperature=4.0,
            alpha=0.7,
            num_epochs=1
        )
        
        # 创建训练数据
        train_data = [(torch.randn(2, 10), torch.randint(0, 5, (2,)))]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=1)
        
        result = distillation_trainer.distill(config, mock_dataloader)
        
        assert result.student_model is not None
        assert result.compression_ratio > 1.0
    
    def test_attention_based_distillation(self, distillation_trainer, teacher_model, student_model):
        """测试注意力式蒸馏"""
        config = DistillationConfig(
            teacher_model=teacher_model,
            student_model=student_model,
            distillation_type="attention_based",
            temperature=2.5,
            alpha=0.3,
            num_epochs=1
        )
        
        train_data = [(torch.randn(1, 10), torch.randint(0, 5, (1,)))]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=1)
        
        result = distillation_trainer.distill(config, mock_dataloader)
        
        assert result.student_model is not None
        assert "attention_transfer_loss" in result.training_history[-1]
    
    def test_self_distillation(self, distillation_trainer, teacher_model):
        """测试自蒸馏"""
        config = DistillationConfig(
            teacher_model=teacher_model,
            student_model=teacher_model,  # 使用相同模型
            distillation_type="self_distillation",
            temperature=3.0,
            alpha=0.5,
            num_epochs=1
        )
        
        train_data = [(torch.randn(1, 10), torch.randint(0, 5, (1,)))]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=1)
        
        result = distillation_trainer.distill(config, mock_dataloader)
        
        assert result.student_model is not None
    
    def test_knowledge_transfer_loss_calculation(self, distillation_trainer, teacher_model, student_model):
        """测试知识转移损失计算"""
        teacher_output = torch.randn(2, 5)
        student_output = torch.randn(2, 5)
        temperature = 3.0
        
        # 计算KL散度损失
        loss = distillation_trainer._calculate_kd_loss(teacher_output, student_output, temperature)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.requires_grad
        assert loss.item() >= 0
    
    def test_feature_matching_loss(self, distillation_trainer):
        """测试特征匹配损失"""
        teacher_features = torch.randn(2, 64)
        student_features = torch.randn(2, 32)
        
        # 测试特征匹配
        loss = distillation_trainer._calculate_feature_loss(teacher_features, student_features)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0
    
    def test_progressive_training(self, distillation_trainer, basic_config):
        """测试渐进式训练"""
        # 启用渐进式训练
        basic_config.progressive_distillation = True
        basic_config.num_epochs = 3
        
        train_data = [(torch.randn(1, 10), torch.randint(0, 5, (1,)))]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=1)
        
        result = distillation_trainer.distill(basic_config, mock_dataloader)
        
        # 验证训练历史记录了温度变化
        assert len(result.training_history) == basic_config.num_epochs
        assert result.student_model is not None
    
    def test_model_saving(self, distillation_trainer, basic_config, tmp_path):
        """测试模型保存"""
        train_data = [(torch.randn(1, 10), torch.randint(0, 5, (1,)))]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=1)
        
        result = distillation_trainer.distill(basic_config, mock_dataloader)
        
        # 保存学生模型
        save_path = tmp_path / "student_model.pth"
        saved_path = distillation_trainer.save_student_model(result.student_model, str(save_path))
        
        assert saved_path == str(save_path)
        assert save_path.exists()
        
        # 验证能够加载保存的模型
        loaded_model = torch.load(save_path)
        assert loaded_model is not None
    
    def test_evaluation_metrics(self, distillation_trainer, basic_config):
        """测试评估指标"""
        train_data = [(torch.randn(2, 10), torch.randint(0, 5, (2,)))]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=1)
        
        result = distillation_trainer.distill(basic_config, mock_dataloader, mock_dataloader)
        
        # 检查评估指标
        assert result.final_accuracy >= 0
        assert result.final_loss >= 0
        assert len(result.training_history) > 0
        
        # 检查训练历史中的指标
        for epoch_stats in result.training_history:
            assert "train_loss" in epoch_stats
            assert "kd_loss" in epoch_stats
    
    def test_different_optimizers(self, distillation_trainer, basic_config):
        """测试不同优化器"""
        optimizers = ["adam", "sgd", "adamw"]
        
        for opt_name in optimizers:
            config = DistillationConfig(
                teacher_model=basic_config.teacher_model,
                student_model=StudentModel(),  # 创建新实例
                optimizer=opt_name,
                num_epochs=1
            )
            
            train_data = [(torch.randn(1, 10), torch.randint(0, 5, (1,)))]
            mock_dataloader = Mock()
            mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
            mock_dataloader.__len__ = Mock(return_value=1)
            
            result = distillation_trainer.distill(config, mock_dataloader)
            assert result.student_model is not None
    
    def test_early_stopping(self, distillation_trainer, basic_config):
        """测试早停机制"""
        basic_config.early_stopping_patience = 2
        basic_config.num_epochs = 10  # 设置更多轮次
        
        train_data = [(torch.randn(2, 10), torch.randint(0, 5, (2,)))]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=1)
        
        result = distillation_trainer.distill(basic_config, mock_dataloader, mock_dataloader)
        
        # 由于是简单数据，可能会提前停止
        assert len(result.training_history) <= basic_config.num_epochs
        assert result.student_model is not None
    
    def test_error_handling_invalid_strategy(self, distillation_trainer, basic_config):
        """测试无效策略的错误处理"""
        basic_config.distillation_type = "invalid_strategy"
        
        train_data = [(torch.randn(1, 10), torch.randint(0, 5, (1,)))]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=1)
        
        # 应该降级到response_based策略
        result = distillation_trainer.distill(basic_config, mock_dataloader)
        assert result.student_model is not None
    
    @patch('src.ai.model_compression.distillation_trainer.torch.nn.functional.kl_div')
    def test_training_error_handling(self, mock_kl_div, distillation_trainer, basic_config):
        """测试训练过程中的错误处理"""
        # 模拟KL散度计算失败
        mock_kl_div.side_effect = RuntimeError("KL divergence failed")
        
        train_data = [(torch.randn(1, 10), torch.randint(0, 5, (1,)))]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=1)
        
        with pytest.raises(RuntimeError):
            distillation_trainer.distill(basic_config, mock_dataloader)
    
    def test_get_supported_strategies(self, distillation_trainer):
        """测试获取支持的策略"""
        strategies = distillation_trainer.get_supported_strategies()
        assert isinstance(strategies, list)
        assert len(strategies) > 0
        assert "response_based" in strategies
        assert "feature_based" in strategies
    
    def test_model_compression_ratio_calculation(self, distillation_trainer, teacher_model, student_model):
        """测试模型压缩比计算"""
        teacher_params = sum(p.numel() for p in teacher_model.parameters())
        student_params = sum(p.numel() for p in student_model.parameters())
        
        compression_ratio = distillation_trainer._calculate_compression_ratio(teacher_model, student_model)
        
        expected_ratio = teacher_params / student_params
        assert abs(compression_ratio - expected_ratio) < 1e-6


@pytest.mark.parametrize("distillation_type,temperature,alpha", [
    ("response_based", 3.0, 0.5),
    ("feature_based", 4.0, 0.7),
    ("attention_based", 2.5, 0.3),
])
def test_different_distillation_combinations(distillation_type, temperature, alpha, 
                                           distillation_trainer, teacher_model, student_model):
    """参数化测试不同的蒸馏组合"""
    config = DistillationConfig(
        teacher_model=teacher_model,
        student_model=student_model,
        distillation_type=distillation_type,
        temperature=temperature,
        alpha=alpha,
        num_epochs=1
    )
    
    train_data = [(torch.randn(1, 10), torch.randint(0, 5, (1,)))]
    mock_dataloader = Mock()
    mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
    mock_dataloader.__len__ = Mock(return_value=1)
    
    result = distillation_trainer.distill(config, mock_dataloader)
    
    assert result.student_model is not None
    assert result.compression_ratio >= 1.0


class TestDistillationEdgeCases:
    """蒸馏边界情况测试"""
    
    def test_identical_models(self, distillation_trainer, teacher_model):
        """测试相同的教师和学生模型"""
        config = DistillationConfig(
            teacher_model=teacher_model,
            student_model=teacher_model,
            num_epochs=1
        )
        
        train_data = [(torch.randn(1, 10), torch.randint(0, 5, (1,)))]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=1)
        
        result = distillation_trainer.distill(config, mock_dataloader)
        assert result.compression_ratio == 1.0
    
    def test_empty_dataloader(self, distillation_trainer, basic_config):
        """测试空数据加载器"""
        empty_dataloader = Mock()
        empty_dataloader.__iter__ = Mock(return_value=iter([]))
        empty_dataloader.__len__ = Mock(return_value=0)
        
        with pytest.raises(ValueError, match="训练数据为空"):
            distillation_trainer.distill(basic_config, empty_dataloader)
    
    def test_very_high_temperature(self, distillation_trainer, basic_config):
        """测试非常高的温度"""
        basic_config.temperature = 100.0
        basic_config.num_epochs = 1
        
        train_data = [(torch.randn(1, 10), torch.randint(0, 5, (1,)))]
        mock_dataloader = Mock()
        mock_dataloader.__iter__ = Mock(return_value=iter(train_data))
        mock_dataloader.__len__ = Mock(return_value=1)
        
        result = distillation_trainer.distill(basic_config, mock_dataloader)
        assert result.student_model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])