"""
端到端集成测试
验证完整的微调流程
"""
import pytest
import json
import tempfile
import os
from pathlib import Path
import torch
from unittest.mock import Mock, patch
from src.ai.fine_tuning import (
    LoRATrainer, QLoRATrainer, 
    TrainingConfig, LoRAConfig, QuantizationConfig,
    ConfigManager, ModelAdapterFactory,
    TrainingMonitor, ModelArchitecture, TrainingMode, QuantizationType
)


@pytest.mark.integration
class TestFineTuningE2E:
    """端到端微调测试"""
    
    @pytest.fixture
    def sample_dataset(self):
        """创建示例数据集"""
        data = [
            {
                "instruction": "什么是人工智能？",
                "output": "人工智能（AI）是一门计算机科学分支，致力于创建能够执行通常需要人类智能的任务的系统。"
            },
            {
                "instruction": "解释深度学习的概念。",
                "output": "深度学习是机器学习的一个子集，使用具有多层的神经网络来模拟人脑处理信息的方式。"
            },
            {
                "instruction": "什么是自然语言处理？",
                "output": "自然语言处理（NLP）是人工智能的一个分支，专注于计算机与人类语言之间的交互。"
            },
            {
                "instruction": "什么是机器学习？",
                "output": "机器学习是人工智能的一个分支，让计算机能够在没有明确编程的情况下学习和改进。"
            },
            {
                "instruction": "解释神经网络的概念。",
                "output": "神经网络是一种计算模型，模仿人脑神经元的工作方式，由互连的节点（神经元）层组成。"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f, ensure_ascii=False)
            yield f.name
        
        # 清理
        if os.path.exists(f.name):
            os.unlink(f.name)
    
    @pytest.fixture
    def training_config(self, sample_dataset):
        """创建训练配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            lora_config = LoRAConfig(
                rank=8,
                alpha=16,
                dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            )
            
            config = TrainingConfig(
                model_name="facebook/opt-125m",  # 使用小模型进行测试
                model_architecture=ModelArchitecture.LLAMA,
                training_mode=TrainingMode.LORA,
                dataset_path=sample_dataset,
                output_dir=temp_dir,
                learning_rate=1e-4,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=2,
                warmup_steps=5,
                max_seq_length=256,
                lora_config=lora_config,
                use_flash_attention=False,
                use_gradient_checkpointing=False,
                fp16=False,
                bf16=False
            )
            yield config
    
    def test_config_manager_workflow(self):
        """测试配置管理器工作流程"""
        config_manager = ConfigManager()
        
        # 测试模板列表
        templates = config_manager.list_templates()
        assert len(templates) > 0
        assert "lora_small" in templates
        assert "qlora_4bit" in templates
        
        # 测试获取模板
        lora_template = config_manager.get_template("lora_small")
        assert lora_template is not None
        assert lora_template["training_mode"] == TrainingMode.LORA
        
        # 测试从模板创建配置
        config = config_manager.create_config_from_template(
            "lora_small",
            model_name="test-model",
            dataset_path="test.json",
            output_dir="/tmp/test"
        )
        assert config.model_name == "test-model"
        assert config.training_mode == TrainingMode.LORA
        
        # 测试配置验证
        errors = config_manager.validate_config(config)
        # 应该有错误，因为数据集路径不存在
        assert len(errors) > 0
    
    def test_model_adapter_workflow(self):
        """测试模型适配器工作流程"""
        # 测试不同模型的适配器创建
        test_models = [
            ("meta-llama/Llama-2-7b-hf", ModelArchitecture.LLAMA),
            ("mistralai/Mistral-7B-v0.1", ModelArchitecture.MISTRAL),
            ("Qwen/Qwen-7B", ModelArchitecture.QWEN),
            ("THUDM/chatglm3-6b", ModelArchitecture.CHATGLM)
        ]
        
        for model_name, expected_arch in test_models:
            # 测试适配器创建
            adapter = ModelAdapterFactory.create_adapter(model_name)
            assert adapter.get_architecture() == expected_arch
            
            # 测试目标模块获取
            target_modules = adapter.get_target_modules()
            assert isinstance(target_modules, list)
            assert len(target_modules) > 0
            
            # 测试优化配置
            opt_config = adapter.get_optimization_config()
            assert isinstance(opt_config, dict)
            assert "use_flash_attention" in opt_config
            
            # 测试架构检测
            detected_arch = ModelAdapterFactory.detect_model_architecture(model_name)
            assert detected_arch == expected_arch
    
    def test_training_monitor_workflow(self):
        """测试训练监控器工作流程"""
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = TrainingMonitor(log_dir=temp_dir)
            
            # 测试指标记录
            monitor.log_metric("loss", 0.5, step=1)
            monitor.log_metric("loss", 0.4, step=2)
            monitor.log_metric("accuracy", 0.8, step=1)
            
            # 测试事件记录
            monitor.log_event("training_start", {"model": "test-model"})
            monitor.log_event("epoch_complete", {"epoch": 1, "loss": 0.4})
            
            # 测试告警记录
            monitor.log_alert("high_loss", "损失值异常", "warning")
            
            # 测试指标摘要
            summary = monitor.get_metrics_summary()
            assert "loss" in summary
            assert "accuracy" in summary
            assert summary["loss"]["count"] == 2
            assert summary["loss"]["latest"] == 0.4
            
            # 测试进度获取
            progress = monitor.get_training_progress()
            assert "elapsed_time" in progress
            assert "event_count" in progress
            assert progress["event_count"] == 2
            
            # 测试报告保存
            report_path = monitor.save_report()
            assert os.path.exists(report_path)
            
            # 验证报告内容
            with open(report_path, 'r', encoding='utf-8') as f:
                report = json.load(f)
                assert "metadata" in report
                assert "metrics_summary" in report
                assert "events" in report
                assert len(report["events"]) == 2
    
    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM')
    @patch('peft.get_peft_model')
    @patch('transformers.Trainer')
    def test_lora_training_workflow(self, mock_trainer_cls, mock_get_peft, mock_model_cls, 
                                   mock_tokenizer_cls, training_config, sample_dataset):
        """测试LoRA训练完整流程"""
        # 设置模拟对象
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "</s>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_base_model = Mock()
        mock_base_model.num_parameters.return_value = 1000000
        mock_base_model.dtype = torch.float32
        mock_model_cls.from_pretrained.return_value = mock_base_model
        
        mock_peft_model = Mock()
        mock_peft_model.num_parameters.side_effect = [100000, 1000000]  # trainable, total
        mock_get_peft.return_value = mock_peft_model
        
        mock_trainer = Mock()
        mock_train_result = Mock()
        mock_train_result.metrics = {
            "train_runtime": 100.0,
            "train_samples_per_second": 5.0,
            "train_steps_per_second": 1.0,
            "train_loss": 0.5
        }
        mock_train_result.global_step = 20
        mock_trainer.train.return_value = mock_train_result
        mock_trainer_cls.return_value = mock_trainer
        
        # 创建训练器并执行训练
        with tempfile.TemporaryDirectory() as temp_dir:
            monitor = TrainingMonitor(log_dir=temp_dir)
            trainer = LoRATrainer(training_config, monitor)
            
            # 模拟数据集加载
            with patch('datasets.load_dataset') as mock_load_dataset:
                mock_dataset = Mock()
                mock_dataset.__len__ = Mock(return_value=5)
                mock_dataset.column_names = ["instruction", "output"]
                mock_dataset.map.return_value = mock_dataset
                mock_load_dataset.return_value = mock_dataset
                
                # 执行训练
                result = trainer.train()
        
        # 验证训练结果
        assert result["train_loss"] == 0.5
        assert result["total_steps"] == 20
        assert result["final_model_path"] == training_config.output_dir
        
        # 验证训练流程调用
        mock_model_cls.from_pretrained.assert_called()
        mock_get_peft.assert_called()
        mock_trainer.train.assert_called()
        mock_trainer.save_model.assert_called()
    
    @patch('transformers.AutoTokenizer')
    @patch('transformers.AutoModelForCausalLM') 
    @patch('transformers.BitsAndBytesConfig')
    @patch('peft.prepare_model_for_kbit_training')
    @patch('peft.get_peft_model')
    @patch('transformers.Trainer')
    def test_qlora_training_workflow(self, mock_trainer_cls, mock_get_peft, mock_prepare_model,
                                    mock_bnb_config_cls, mock_model_cls, mock_tokenizer_cls, sample_dataset):
        """测试QLoRA训练完整流程"""
        # 创建QLoRA配置
        with tempfile.TemporaryDirectory() as temp_dir:
            quantization_config = QuantizationConfig(
                quantization_type=QuantizationType.NF4,
                bits=4,
                use_double_quant=True
            )
            
            lora_config = LoRAConfig(
                rank=16,
                alpha=32,
                dropout=0.1
            )
            
            config = TrainingConfig(
                model_name="facebook/opt-125m",
                model_architecture=ModelArchitecture.LLAMA,
                training_mode=TrainingMode.QLORA,
                dataset_path=sample_dataset,
                output_dir=temp_dir,
                learning_rate=2e-4,
                num_train_epochs=1,
                per_device_train_batch_size=1,
                lora_config=lora_config,
                quantization_config=quantization_config
            )
            
            # 设置模拟对象
            mock_tokenizer = Mock()
            mock_tokenizer.pad_token = None
            mock_tokenizer.eos_token = "</s>"
            mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
            
            mock_base_model = Mock()
            mock_base_model.num_parameters.return_value = 1000000
            mock_base_model.dtype = torch.float16
            mock_model_cls.from_pretrained.return_value = mock_base_model
            
            mock_prepared_model = Mock()
            mock_prepare_model.return_value = mock_prepared_model
            
            mock_peft_model = Mock()
            mock_peft_model.num_parameters.side_effect = [50000, 1000000]
            mock_get_peft.return_value = mock_peft_model
            
            mock_trainer = Mock()
            mock_train_result = Mock()
            mock_train_result.metrics = {
                "train_runtime": 150.0,
                "train_samples_per_second": 3.0,
                "train_steps_per_second": 0.5,
                "train_loss": 0.3
            }
            mock_train_result.global_step = 15
            mock_trainer.train.return_value = mock_train_result
            mock_trainer_cls.return_value = mock_trainer
            
            # 创建QLoRA训练器
            monitor = TrainingMonitor(log_dir=temp_dir)
            trainer = QLoRATrainer(config, monitor)
            
            # 模拟数据集加载
            with patch('datasets.load_dataset') as mock_load_dataset:
                mock_dataset = Mock()
                mock_dataset.__len__ = Mock(return_value=5)
                mock_dataset.column_names = ["instruction", "output"]
                mock_dataset.map.return_value = mock_dataset
                mock_load_dataset.return_value = mock_dataset
                
                # 执行训练
                result = trainer.train()
            
            # 验证QLoRA特定的结果
            assert result["quantization_bits"] == 4
            assert result["quantization_type"] == "nf4"
            assert result["training_mode"] == "qlora"
            assert result["train_loss"] == 0.3
            
            # 验证量化配置创建
            mock_bnb_config_cls.assert_called()
            
            # 验证模型准备用于量化训练
            mock_prepare_model.assert_called()
    
    def test_configuration_file_workflow(self, sample_dataset):
        """测试配置文件工作流程"""
        config_manager = ConfigManager()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建配置
            config = config_manager.create_config_from_template(
                "lora_medium",
                model_name="facebook/opt-125m",
                dataset_path=sample_dataset,
                output_dir=temp_dir
            )
            
            # 测试保存配置到JSON文件
            json_config_path = os.path.join(temp_dir, "config.json")
            config_manager.save_config_to_file(config, json_config_path)
            assert os.path.exists(json_config_path)
            
            # 测试从JSON文件加载配置
            loaded_config = config_manager.load_config_from_file(json_config_path)
            assert loaded_config.model_name == config.model_name
            assert loaded_config.training_mode == config.training_mode
            
            # 测试保存配置到YAML文件
            yaml_config_path = os.path.join(temp_dir, "config.yaml")
            config_manager.save_config_to_file(config, yaml_config_path)
            assert os.path.exists(yaml_config_path)
            
            # 测试从YAML文件加载配置
            loaded_yaml_config = config_manager.load_config_from_file(yaml_config_path)
            assert loaded_yaml_config.model_name == config.model_name
    
    def test_hardware_detection_workflow(self):
        """测试硬件检测工作流程"""
        config_manager = ConfigManager()
        
        # 测试硬件配置检测
        hardware_config = config_manager.detect_hardware_config()
        assert "hardware_info" in hardware_config
        assert "recommendations" in hardware_config
        
        hardware_info = hardware_config["hardware_info"]
        assert "cuda_available" in hardware_info
        assert "device_count" in hardware_info
        assert "total_memory" in hardware_info
        assert "device_names" in hardware_info
        
        recommendations = hardware_config["recommendations"]
        assert "use_bf16" in recommendations
        assert "use_fp16" in recommendations
        assert "use_gradient_checkpointing" in recommendations
        assert "use_flash_attention" in recommendations
    
    def test_model_specific_config_workflow(self):
        """测试模型特定配置工作流程"""
        config_manager = ConfigManager()
        
        test_models = [
            "meta-llama/Llama-2-7b-hf",
            "mistralai/Mistral-7B-v0.1", 
            "Qwen/Qwen-7B",
            "THUDM/chatglm3-6b"
        ]
        
        for model_name in test_models:
            model_config = config_manager.get_model_specific_config(model_name)
            assert "target_modules" in model_config
            assert "max_seq_length" in model_config
            assert isinstance(model_config["target_modules"], list)
            assert model_config["max_seq_length"] > 0
    
    def test_error_handling_workflow(self):
        """测试错误处理工作流程"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 测试无效模型名称
            invalid_config = TrainingConfig(
                model_name="",
                model_architecture=ModelArchitecture.LLAMA,
                training_mode=TrainingMode.LORA,
                dataset_path="nonexistent.json",
                output_dir=temp_dir
            )
            
            monitor = TrainingMonitor(log_dir=temp_dir)
            trainer = LoRATrainer(invalid_config, monitor)
            
            # 应该在验证阶段捕获错误
            config_manager = ConfigManager()
            errors = config_manager.validate_config(invalid_config)
            assert len(errors) > 0
            assert any("model_name 不能为空" in error for error in errors)
            assert any("数据集路径不存在" in error for error in errors)
    
    def test_memory_optimization_workflow(self):
        """测试内存优化工作流程"""
        from src.ai.fine_tuning.model_adapters import ModelOptimizer
        
        # 测试不同GPU内存大小的优化建议
        memory_sizes = [8, 16, 24, 40]  # GB
        
        for memory_gb in memory_sizes:
            adapter = ModelAdapterFactory.create_adapter("meta-llama/Llama-2-7b-hf")
            optimizer = ModelOptimizer(adapter)
            
            # 测试批次大小推荐
            batch_size, grad_acc = optimizer.get_recommended_batch_size(memory_gb, 2048)
            assert batch_size >= 1
            assert grad_acc >= 1
            assert batch_size * grad_acc <= 32  # 合理范围
            
            # 测试量化推荐
            quant_rec = optimizer.get_quantization_recommendations(memory_gb)
            assert "use_quantization" in quant_rec
            
            if memory_gb < 12:
                assert quant_rec["use_quantization"] is True
                assert quant_rec["bits"] == 4
            elif memory_gb < 24:
                assert quant_rec["use_quantization"] is True
                assert quant_rec["bits"] == 8
            else:
                assert quant_rec["use_quantization"] is False
    
    def test_complete_training_pipeline(self, sample_dataset):
        """测试完整训练流水线（模拟）"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. 配置管理
            config_manager = ConfigManager()
            
            # 2. 硬件检测
            hardware_config = config_manager.detect_hardware_config()
            
            # 3. 模型适配
            model_name = "facebook/opt-125m"
            adapter = ModelAdapterFactory.create_adapter(model_name)
            
            # 4. 配置创建和优化
            base_config = config_manager.create_config_from_template(
                "lora_small",
                model_name=model_name,
                dataset_path=sample_dataset,
                output_dir=temp_dir
            )
            
            # 5. 配置验证
            errors = config_manager.validate_config(base_config)
            assert len(errors) == 0  # 应该没有错误
            
            # 6. 监控器创建
            monitor = TrainingMonitor(log_dir=temp_dir, enable_wandb=False)
            
            # 7. 训练器创建（根据配置选择类型）
            if base_config.quantization_config:
                trainer_cls = QLoRATrainer
            else:
                trainer_cls = LoRATrainer
            
            trainer = trainer_cls(base_config, monitor)
            
            # 8. 验证训练器正确初始化
            assert trainer.config == base_config
            assert trainer.monitor == monitor
            
            # 9. 配置保存
            config_path = os.path.join(temp_dir, "training_config.json")
            config_manager.save_config_to_file(base_config, config_path)
            assert os.path.exists(config_path)
            
            # 10. 验证可以重新加载配置
            reloaded_config = config_manager.load_config_from_file(config_path)
            assert reloaded_config.model_name == base_config.model_name
            
            print(f"✅ 完整训练流水线测试通过 - 使用 {trainer_cls.__name__}")


@pytest.mark.performance
class TestFineTuningPerformance:
    """性能测试"""
    
    def test_config_manager_performance(self):
        """测试配置管理器性能"""
        import time
        
        config_manager = ConfigManager()
        
        # 测试模板加载性能
        start_time = time.time()
        for _ in range(100):
            templates = config_manager.list_templates()
        end_time = time.time()
        
        assert (end_time - start_time) < 1.0  # 应该在1秒内完成
        
        # 测试配置创建性能
        start_time = time.time()
        for _ in range(50):
            config = config_manager.create_config_from_template(
                "lora_small",
                model_name="test-model",
                dataset_path="test.json",
                output_dir="/tmp/test"
            )
        end_time = time.time()
        
        assert (end_time - start_time) < 2.0  # 应该在2秒内完成
    
    def test_model_adapter_performance(self):
        """测试模型适配器性能"""
        import time
        
        test_models = [
            "meta-llama/Llama-2-7b-hf",
            "mistralai/Mistral-7B-v0.1",
            "Qwen/Qwen-7B",
            "THUDM/chatglm3-6b"
        ] * 25  # 重复25次，共100个模型
        
        # 测试适配器创建性能
        start_time = time.time()
        for model_name in test_models:
            adapter = ModelAdapterFactory.create_adapter(model_name)
            target_modules = adapter.get_target_modules()
            arch = adapter.get_architecture()
        end_time = time.time()
        
        assert (end_time - start_time) < 5.0  # 应该在5秒内完成
        
        print(f"✅ 处理 {len(test_models)} 个模型适配器耗时: {end_time - start_time:.2f}秒")


if __name__ == "__main__":
    # 可以直接运行这个文件进行测试
    pytest.main([__file__, "-v", "-s"])