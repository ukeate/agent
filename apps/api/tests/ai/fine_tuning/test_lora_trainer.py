"""
LoRA训练器测试
"""
import pytest
import torch
import tempfile
import json
import os
from unittest.mock import Mock, patch, MagicMock
from src.ai.fine_tuning.lora_trainer import LoRATrainer
from src.ai.fine_tuning.models import (
    TrainingConfig, LoRAConfig, ModelArchitecture, 
    TrainingMode, QuantizationType
)
from src.ai.fine_tuning.training_monitor import TrainingMonitor


class TestLoRATrainer:
    """LoRA训练器测试类"""
    
    @pytest.fixture
    def training_config(self):
        """创建测试训练配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            lora_config = LoRAConfig(
                rank=8,
                alpha=16,
                dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            )
            
            config = TrainingConfig(
                model_name="facebook/opt-125m",  # 使用小型模型进行测试
                model_architecture=ModelArchitecture.LLAMA,
                training_mode=TrainingMode.LORA,
                dataset_path="test_dataset.json",
                output_dir=temp_dir,
                learning_rate=2e-4,
                num_train_epochs=1,  # 测试时使用少量epoch
                per_device_train_batch_size=2,
                gradient_accumulation_steps=2,
                warmup_steps=10,
                max_seq_length=512,  # 使用较短序列长度
                lora_config=lora_config,
                use_flash_attention=False,  # 测试环境可能不支持
                use_gradient_checkpointing=False,
                fp16=False,
                bf16=False
            )
            yield config
    
    @pytest.fixture
    def mock_monitor(self):
        """创建模拟监控器"""
        monitor = Mock(spec=TrainingMonitor)
        monitor.log_metric = Mock()
        monitor.log_event = Mock()
        return monitor
    
    @pytest.fixture
    def test_dataset_file(self):
        """创建测试数据集文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            test_data = [
                {
                    "instruction": "What is the capital of France?",
                    "output": "The capital of France is Paris."
                },
                {
                    "instruction": "Explain photosynthesis briefly.",
                    "output": "Photosynthesis is the process by which plants convert sunlight into energy."
                },
                {
                    "instruction": "What is 2 + 2?",
                    "output": "2 + 2 equals 4."
                }
            ]
            json.dump(test_data, f)
            yield f.name
        
        # 清理测试文件
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    def test_trainer_initialization(self, training_config, mock_monitor):
        """测试训练器初始化"""
        trainer = LoRATrainer(training_config, mock_monitor)
        
        assert trainer.config == training_config
        assert trainer.monitor == mock_monitor
        assert trainer.model is None
        assert trainer.tokenizer is None
        assert trainer.trainer is None
        
        # 验证监控器调用
        mock_monitor.log_event.assert_called()
    
    @patch('src.ai.fine_tuning.lora_trainer.AutoTokenizer')
    @patch('src.ai.fine_tuning.lora_trainer.AutoModelForCausalLM')
    def test_load_model_and_tokenizer(self, mock_model_cls, mock_tokenizer_cls, training_config, mock_monitor):
        """测试模型和分词器加载"""
        # 设置模拟对象
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.num_parameters.return_value = 1000000
        mock_model.dtype = torch.float32
        mock_model_cls.from_pretrained.return_value = mock_model
        
        trainer = LoRATrainer(training_config, mock_monitor)
        trainer.load_model_and_tokenizer()
        
        # 验证模型和分词器已加载
        assert trainer.model is not None
        assert trainer.tokenizer is not None
        
        # 验证分词器配置
        assert trainer.tokenizer.pad_token == "<eos>"
        
        # 验证监控器调用
        mock_monitor.log_event.assert_called_with(
            "model_loading_complete",
            {
                "model_parameters": 1000000,
                "model_dtype": "torch.float32"
            }
        )
    
    @patch('src.ai.fine_tuning.lora_trainer.get_peft_model')
    @patch('src.ai.fine_tuning.lora_trainer.LoraConfig')
    def test_setup_peft_model(self, mock_lora_config_cls, mock_get_peft_model, training_config, mock_monitor):
        """测试PEFT模型设置"""
        # 设置模拟对象
        mock_base_model = Mock()
        mock_peft_model = Mock()
        mock_peft_model.num_parameters.side_effect = [100000, 1000000]  # trainable, total
        mock_get_peft_model.return_value = mock_peft_model
        
        trainer = LoRATrainer(training_config, mock_monitor)
        trainer.model = mock_base_model
        trainer.setup_peft_model()
        
        # 验证LoRA配置创建
        mock_lora_config_cls.assert_called()
        
        # 验证PEFT模型创建
        mock_get_peft_model.assert_called_with(mock_base_model, mock_lora_config_cls.return_value)
        
        # 验证模型更新
        assert trainer.model == mock_peft_model
        
        # 验证监控器调用
        mock_monitor.log_metric.assert_any_call("trainable_parameters", 100000)
        mock_monitor.log_metric.assert_any_call("total_parameters", 1000000)
        mock_monitor.log_metric.assert_any_call("trainable_percentage", 10.0)
    
    def test_get_target_modules(self, training_config, mock_monitor):
        """测试目标模块获取"""
        trainer = LoRATrainer(training_config, mock_monitor)
        target_modules = trainer._get_target_modules()
        
        # 验证返回了配置中的目标模块
        assert target_modules == training_config.lora_config.target_modules
    
    def test_get_target_modules_with_architecture(self, training_config, mock_monitor):
        """测试根据架构获取目标模块"""
        # 清除配置中的目标模块，让它使用架构默认值
        training_config.lora_config.target_modules = None
        
        trainer = LoRATrainer(training_config, mock_monitor)
        target_modules = trainer._get_target_modules()
        
        # 验证返回了LLaMA架构的默认目标模块
        expected_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        assert target_modules == expected_modules
    
    @patch('src.ai.fine_tuning.lora_trainer.load_dataset')
    def test_prepare_dataset(self, mock_load_dataset, training_config, mock_monitor, test_dataset_file):
        """测试数据集准备"""
        # 设置模拟数据集
        mock_dataset = Mock()
        mock_dataset.__len__ = Mock(return_value=3)
        mock_dataset.column_names = ["instruction", "output"]
        mock_dataset.map.return_value = mock_dataset
        mock_load_dataset.return_value = mock_dataset
        
        # 设置模拟分词器
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token = "<eos>"
        
        trainer = LoRATrainer(training_config, mock_monitor)
        trainer.tokenizer = mock_tokenizer
        
        # 更新配置使用测试数据集文件
        training_config.dataset_path = test_dataset_file
        
        result = trainer.prepare_dataset(test_dataset_file)
        
        # 验证数据集加载
        mock_load_dataset.assert_called()
        
        # 验证监控器调用
        mock_monitor.log_event.assert_called_with(
            "dataset_processing_complete",
            {
                "tokenized_size": 3,
                "max_seq_length": training_config.max_seq_length
            }
        )
        
        assert result == mock_dataset
    
    @patch('src.ai.fine_tuning.lora_trainer.Trainer')
    @patch('src.ai.fine_tuning.lora_trainer.TrainingArguments')
    @patch('src.ai.fine_tuning.lora_trainer.DataCollatorForSeq2Seq')
    def test_create_trainer(self, mock_data_collator_cls, mock_training_args_cls, mock_trainer_cls, training_config, mock_monitor):
        """测试训练器创建"""
        # 设置模拟对象
        mock_model = Mock()
        mock_tokenizer = Mock()
        mock_dataset = Mock()
        
        trainer = LoRATrainer(training_config, mock_monitor)
        trainer.model = mock_model
        trainer.tokenizer = mock_tokenizer
        
        result = trainer.create_trainer(mock_dataset)
        
        # 验证训练参数创建
        mock_training_args_cls.assert_called()
        
        # 验证数据收集器创建
        mock_data_collator_cls.assert_called_with(
            tokenizer=mock_tokenizer,
            model=mock_model,
            label_pad_token_id=-100,
            pad_to_multiple_of=None
        )
        
        # 验证训练器创建
        mock_trainer_cls.assert_called()
        
        assert result is not None
    
    @patch('src.ai.fine_tuning.lora_trainer.LoRATrainer.load_model_and_tokenizer')
    @patch('src.ai.fine_tuning.lora_trainer.LoRATrainer.setup_peft_model')
    @patch('src.ai.fine_tuning.lora_trainer.LoRATrainer.prepare_dataset')
    @patch('src.ai.fine_tuning.lora_trainer.LoRATrainer.create_trainer')
    def test_train_success(self, mock_create_trainer, mock_prepare_dataset, 
                          mock_setup_peft, mock_load_model, training_config, mock_monitor):
        """测试成功训练流程"""
        # 设置模拟对象
        mock_dataset = Mock()
        mock_prepare_dataset.return_value = mock_dataset
        
        mock_trainer = Mock()
        mock_train_result = Mock()
        mock_train_result.metrics = {
            "train_runtime": 100.0,
            "train_samples_per_second": 10.0,
            "train_steps_per_second": 2.0,
            "train_loss": 0.5
        }
        mock_train_result.global_step = 50
        mock_trainer.train.return_value = mock_train_result
        mock_create_trainer.return_value = mock_trainer
        
        trainer = LoRATrainer(training_config, mock_monitor)
        result = trainer.train()
        
        # 验证训练流程调用
        mock_load_model.assert_called_once()
        mock_setup_peft.assert_called_once()
        mock_prepare_dataset.assert_called_once_with(training_config.dataset_path)
        mock_create_trainer.assert_called_once_with(mock_dataset)
        mock_trainer.train.assert_called_once()
        mock_trainer.save_model.assert_called_once()
        mock_trainer.save_state.assert_called_once()
        
        # 验证返回结果
        assert result["train_runtime"] == 100.0
        assert result["train_loss"] == 0.5
        assert result["total_steps"] == 50
        assert result["final_model_path"] == training_config.output_dir
        
        # 验证监控器调用
        mock_monitor.log_event.assert_any_call("training_start", {
            "epochs": training_config.num_train_epochs,
            "batch_size": training_config.per_device_train_batch_size,
            "learning_rate": training_config.learning_rate
        })
        mock_monitor.log_event.assert_any_call("training_complete", result)
    
    @patch('src.ai.fine_tuning.lora_trainer.LoRATrainer.load_model_and_tokenizer')
    def test_train_failure(self, mock_load_model, training_config, mock_monitor):
        """测试训练失败处理"""
        # 模拟加载模型失败
        mock_load_model.side_effect = Exception("模型加载失败")
        
        trainer = LoRATrainer(training_config, mock_monitor)
        
        with pytest.raises(Exception) as exc_info:
            trainer.train()
        
        assert str(exc_info.value) == "模型加载失败"
        
        # 验证监控器记录了失败事件
        mock_monitor.log_event.assert_any_call("training_failed", {"error": "模型加载失败"})
    
    def test_config_to_dict(self, training_config, mock_monitor):
        """测试配置转换为字典"""
        trainer = LoRATrainer(training_config, mock_monitor)
        config_dict = trainer._config_to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["model_name"] == training_config.model_name
        assert config_dict["training_mode"] == training_config.training_mode.value
        assert config_dict["model_architecture"] == training_config.model_architecture.value
    
    def test_wandb_available(self, training_config, mock_monitor):
        """测试wandb可用性检查"""
        trainer = LoRATrainer(training_config, mock_monitor)
        
        # 默认情况下应该返回False（测试环境没有wandb）
        assert trainer._wandb_available() is False
    
    @patch('src.ai.fine_tuning.lora_trainer.PeftModel')
    @patch('src.ai.fine_tuning.lora_trainer.AutoModelForCausalLM')
    @patch('src.ai.fine_tuning.lora_trainer.AutoTokenizer')
    def test_load_model_from_directory(self, mock_tokenizer_cls, mock_model_cls, mock_peft_model_cls, training_config, mock_monitor):
        """测试从目录加载模型"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 设置模拟对象
            mock_base_model = Mock()
            mock_model_cls.from_pretrained.return_value = mock_base_model
            
            mock_peft_model = Mock()
            mock_peft_model_cls.from_pretrained.return_value = mock_peft_model
            
            mock_tokenizer = Mock()
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            
            trainer = LoRATrainer(training_config, mock_monitor)
            trainer.load_model(temp_dir)
            
            # 验证模型加载
            mock_model_cls.from_pretrained.assert_called_with(
                training_config.model_name,
                torch_dtype=torch.float32,  # 因为配置中bf16=False, fp16=False
                device_map="auto"
            )
            mock_peft_model_cls.from_pretrained.assert_called_with(mock_base_model, temp_dir)
            mock_tokenizer_cls.from_pretrained.assert_called_with(temp_dir)
            
            # 验证属性设置
            assert trainer.peft_model == mock_peft_model
            assert trainer.model == mock_peft_model
            assert trainer.tokenizer == mock_tokenizer
    
    def test_save_model(self, training_config, mock_monitor):
        """测试模型保存"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 设置模拟PEFT模型
            mock_peft_model = Mock()
            mock_tokenizer = Mock()
            
            trainer = LoRATrainer(training_config, mock_monitor)
            trainer.peft_model = mock_peft_model
            trainer.tokenizer = mock_tokenizer
            
            trainer.save_model(temp_dir)
            
            # 验证保存调用
            mock_peft_model.save_pretrained.assert_called_with(temp_dir)
            mock_tokenizer.save_pretrained.assert_called_with(temp_dir)
    
    def test_save_model_without_peft_model(self, training_config, mock_monitor):
        """测试没有PEFT模型时的保存操作"""
        trainer = LoRATrainer(training_config, mock_monitor)
        
        with pytest.raises(ValueError, match="PEFT模型尚未初始化"):
            trainer.save_model("/tmp/test")
    
    @patch('src.ai.fine_tuning.lora_trainer.LoRATrainer.create_trainer')
    def test_evaluate(self, mock_create_trainer, training_config, mock_monitor):
        """测试模型评估"""
        # 设置模拟训练器
        mock_trainer = Mock()
        mock_eval_result = {"eval_loss": 0.3, "eval_accuracy": 0.8}
        mock_trainer.evaluate.return_value = mock_eval_result
        mock_create_trainer.return_value = mock_trainer
        
        trainer = LoRATrainer(training_config, mock_monitor)
        trainer.trainer = mock_trainer
        
        result = trainer.evaluate()
        
        # 验证评估调用
        mock_trainer.evaluate.assert_called_once()
        assert result == mock_eval_result
    
    def test_evaluate_without_trainer(self, training_config, mock_monitor):
        """测试没有训练器时的评估操作"""
        trainer = LoRATrainer(training_config, mock_monitor)
        
        with pytest.raises(ValueError, match="训练器尚未初始化"):
            trainer.evaluate()


@pytest.mark.integration
class TestLoRATrainerIntegration:
    """LoRA训练器集成测试"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="需要CUDA支持")
    def test_small_model_training(self):
        """测试小型模型训练（集成测试）"""
        # 这个测试需要实际的GPU环境和较长时间，通常在CI/CD中跳过
        pass
    
    def test_config_validation(self):
        """测试配置验证"""
        from src.ai.fine_tuning.training_config import ConfigManager
        
        config_manager = ConfigManager()
        
        # 测试无效配置
        invalid_config = TrainingConfig(
            model_name="",  # 空模型名
            model_architecture=ModelArchitecture.LLAMA,
            training_mode=TrainingMode.LORA,
            dataset_path="",  # 空数据集路径
            output_dir="",  # 空输出目录
            learning_rate=-1,  # 无效学习率
            num_train_epochs=0,  # 无效epoch数
        )
        
        errors = config_manager.validate_config(invalid_config)
        assert len(errors) > 0
        assert any("model_name 不能为空" in error for error in errors)
        assert any("dataset_path 不能为空" in error for error in errors)
        assert any("learning_rate 应该在 (0, 1] 范围内" in error for error in errors)