"""
QLoRA训练器测试
"""
import pytest
import torch
import tempfile
from unittest.mock import Mock, patch, MagicMock
from src.ai.fine_tuning.qlora_trainer import QLoRATrainer
from src.ai.fine_tuning.models import (
    TrainingConfig, LoRAConfig, QuantizationConfig,
    ModelArchitecture, TrainingMode, QuantizationType
)
from src.ai.fine_tuning.training_monitor import TrainingMonitor


class TestQLoRATrainer:
    """QLoRA训练器测试类"""
    
    @pytest.fixture
    def quantization_config(self):
        """创建量化配置"""
        return QuantizationConfig(
            quantization_type=QuantizationType.NF4,
            bits=4,
            use_double_quant=True,
            quant_type="nf4",
            compute_dtype="bfloat16"
        )
    
    @pytest.fixture
    def training_config(self, quantization_config):
        """创建QLoRA训练配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            lora_config = LoRAConfig(
                rank=16,
                alpha=32,
                dropout=0.1,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
            )
            
            config = TrainingConfig(
                model_name="facebook/opt-125m",
                model_architecture=ModelArchitecture.LLAMA,
                training_mode=TrainingMode.QLORA,
                dataset_path="test_dataset.json",
                output_dir=temp_dir,
                learning_rate=2e-4,
                num_train_epochs=1,
                per_device_train_batch_size=1,  # QLoRA通常使用较小批次
                gradient_accumulation_steps=8,
                warmup_steps=10,
                max_seq_length=512,
                lora_config=lora_config,
                quantization_config=quantization_config,
                use_flash_attention=False,
                use_gradient_checkpointing=True,
                fp16=False,
                bf16=True
            )
            yield config
    
    @pytest.fixture
    def mock_monitor(self):
        """创建模拟监控器"""
        monitor = Mock(spec=TrainingMonitor)
        monitor.log_metric = Mock()
        monitor.log_event = Mock()
        return monitor
    
    def test_qlora_trainer_initialization(self, training_config, mock_monitor):
        """测试QLoRA训练器初始化"""
        trainer = QLoRATrainer(training_config, mock_monitor)
        
        assert trainer.config.training_mode == TrainingMode.QLORA
        assert trainer.config.quantization_config is not None
        assert trainer.config.quantization_config.quantization_type == QuantizationType.NF4
        
        # 验证监控器调用
        mock_monitor.log_event.assert_any_call(
            "qlora_trainer_initialized",
            {
                "quantization_type": "nf4",
                "bits": 4,
                "use_double_quant": True
            }
        )
    
    def test_qlora_trainer_with_default_quantization(self, mock_monitor):
        """测试没有量化配置时的默认设置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TrainingConfig(
                model_name="facebook/opt-125m",
                model_architecture=ModelArchitecture.LLAMA,
                training_mode=TrainingMode.LORA,  # 注意这里用LORA
                dataset_path="test_dataset.json",
                output_dir=temp_dir,
                quantization_config=None  # 没有量化配置
            )
            
            trainer = QLoRATrainer(config, mock_monitor)
            
            # 验证自动设置了量化配置和训练模式
            assert trainer.config.training_mode == TrainingMode.QLORA
            assert trainer.config.quantization_config is not None
            assert trainer.config.quantization_config.quantization_type == QuantizationType.NF4
    
    @patch('transformers.BitsAndBytesConfig')
    def test_create_quantization_config_4bit(self, mock_bnb_config, training_config, mock_monitor):
        """测试4-bit量化配置创建"""
        trainer = QLoRATrainer(training_config, mock_monitor)
        
        with patch('torch.bfloat16') as mock_dtype:
            quantization_config = trainer._create_quantization_config()
            
            # 验证BitsAndBytesConfig调用
            mock_bnb_config.assert_called_with(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=mock_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
    
    @patch('transformers.BitsAndBytesConfig')
    def test_create_quantization_config_8bit(self, mock_bnb_config, training_config, mock_monitor):
        """测试8-bit量化配置创建"""
        # 修改为8-bit量化
        training_config.quantization_config.quantization_type = QuantizationType.INT8
        training_config.quantization_config.bits = 8
        
        trainer = QLoRATrainer(training_config, mock_monitor)
        
        with patch('torch.bfloat16') as mock_dtype:
            quantization_config = trainer._create_quantization_config()
            
            # 验证8-bit配置
            mock_bnb_config.assert_called_with(
                load_in_8bit=True,
                llm_int8_compute_dtype=mock_dtype,
                llm_int8_threshold=6.0
            )
    
    def test_create_quantization_config_invalid_bits(self, training_config, mock_monitor):
        """测试无效量化位数"""
        training_config.quantization_config.bits = 16  # 不支持的位数
        
        trainer = QLoRATrainer(training_config, mock_monitor)
        
        with pytest.raises(ValueError, match="不支持的量化位数: 16"):
            trainer._create_quantization_config()
    
    @patch('src.ai.fine_tuning.qlora_trainer.AutoTokenizer')
    @patch('src.ai.fine_tuning.qlora_trainer.AutoModelForCausalLM')
    @patch('src.ai.fine_tuning.qlora_trainer.prepare_model_for_kbit_training')
    def test_load_model_and_tokenizer_quantized(self, mock_prepare, mock_model_cls, mock_tokenizer_cls, training_config, mock_monitor):
        """测试量化模型加载"""
        # 设置模拟对象
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_cls.from_pretrained.return_value = mock_tokenizer
        
        mock_model = Mock()
        mock_model.num_parameters.return_value = 1000000
        mock_model.dtype = torch.float16
        mock_model_cls.from_pretrained.return_value = mock_model
        
        mock_prepared_model = Mock()
        mock_prepare.return_value = mock_prepared_model
        
        trainer = QLoRATrainer(training_config, mock_monitor)
        
        with patch.object(trainer, '_create_quantization_config') as mock_create_quant:
            mock_quant_config = Mock()
            mock_create_quant.return_value = mock_quant_config
            
            trainer.load_model_and_tokenizer()
        
        # 验证模型加载参数
        mock_model_cls.from_pretrained.assert_called_with(
            training_config.model_name,
            quantization_config=mock_quant_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,  # 因为bf16=True
            trust_remote_code=True,
            use_flash_attention_2=False
        )
        
        # 验证模型准备用于量化训练
        mock_prepare.assert_called_with(
            mock_model,
            use_gradient_checkpointing=True
        )
        
        # 验证模型更新
        assert trainer.model == mock_prepared_model
    
    @patch('src.ai.fine_tuning.qlora_trainer.get_peft_model')
    @patch('src.ai.fine_tuning.qlora_trainer.LoraConfig')
    def test_setup_peft_model_qlora(self, mock_lora_config_cls, mock_get_peft_model, training_config, mock_monitor):
        """测试QLoRA PEFT模型设置"""
        # 设置模拟对象
        mock_base_model = Mock()
        mock_peft_model = Mock()
        mock_peft_model.num_parameters.side_effect = [50000, 1000000]  # trainable, total
        mock_get_peft_model.return_value = mock_peft_model
        
        trainer = QLoRATrainer(training_config, mock_monitor)
        trainer.model = mock_base_model
        
        with patch.object(trainer, '_calculate_memory_efficiency') as mock_calc_memory:
            trainer.setup_peft_model()
        
        # 验证LoRA配置创建
        mock_lora_config_cls.assert_called()
        
        # 验证PEFT模型创建
        mock_get_peft_model.assert_called_with(mock_base_model, mock_lora_config_cls.return_value)
        
        # 验证内存效率计算
        mock_calc_memory.assert_called_with(50000, 1000000)
        
        # 验证监控器调用
        mock_monitor.log_metric.assert_any_call("qlora_trainable_parameters", 50000)
        mock_monitor.log_metric.assert_any_call("qlora_total_parameters", 1000000)
        mock_monitor.log_metric.assert_any_call("qlora_trainable_percentage", 5.0)
    
    def test_calculate_memory_efficiency(self, training_config, mock_monitor):
        """测试内存效率计算"""
        trainer = QLoRATrainer(training_config, mock_monitor)
        
        trainer._calculate_memory_efficiency(100000, 1000000)  # 10%可训练参数
        
        # 验证监控器记录了内存效率指标
        mock_monitor.log_metric.assert_any_call("quantization_memory_reduction", 70.0)  # (32-4)/32 * 100
        mock_monitor.log_metric.assert_any_call("lora_parameter_reduction", 90.0)  # (1-0.1) * 100
    
    def test_log_quantization_stats(self, training_config, mock_monitor):
        """测试量化统计信息记录"""
        # 创建模拟模型结构
        mock_layer1 = Mock()
        mock_layer1.weight = Mock()
        mock_layer1.weight.quant_type = "nf4"
        
        mock_layer2 = Mock()
        # 这层没有量化
        
        mock_model = Mock()
        mock_model.named_modules.return_value = [
            ("layer1", mock_layer1),
            ("layer2", mock_layer2)
        ]
        
        trainer = QLoRATrainer(training_config, mock_monitor)
        trainer.model = mock_model
        
        trainer._log_quantization_stats()
        
        # 验证监控器记录了量化统计
        mock_monitor.log_metric.assert_any_call("quantized_layers", 1)
        mock_monitor.log_metric.assert_any_call("total_layers", 2)
        mock_monitor.log_metric.assert_any_call("quantization_ratio", 0.5)
        mock_monitor.log_metric.assert_any_call("theoretical_memory_reduction", 70.0)
    
    @patch('src.ai.fine_tuning.qlora_trainer.QLoRATrainer.load_model_and_tokenizer')
    @patch('src.ai.fine_tuning.qlora_trainer.QLoRATrainer.setup_peft_model')  
    @patch('src.ai.fine_tuning.qlora_trainer.QLoRATrainer.prepare_dataset')
    @patch('src.ai.fine_tuning.qlora_trainer.QLoRATrainer.create_trainer')
    def test_train_qlora_success(self, mock_create_trainer, mock_prepare_dataset,
                                mock_setup_peft, mock_load_model, training_config, mock_monitor):
        """测试QLoRA训练成功流程"""
        # 设置模拟对象
        mock_dataset = Mock()
        mock_prepare_dataset.return_value = mock_dataset
        
        mock_trainer = Mock()
        mock_train_result = Mock()
        mock_train_result.metrics = {
            "train_runtime": 200.0,
            "train_samples_per_second": 5.0,
            "train_steps_per_second": 1.0,
            "train_loss": 0.3
        }
        mock_train_result.global_step = 100
        mock_trainer.train.return_value = mock_train_result
        mock_create_trainer.return_value = mock_trainer
        
        trainer = QLoRATrainer(training_config, mock_monitor)
        
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.empty_cache'), \
             patch('torch.cuda.memory_allocated', side_effect=[2000000000, 3000000000]):  # 2GB -> 3GB
            
            result = trainer.train()
        
        # 验证QLoRA特定的结果信息
        assert result["quantization_bits"] == 4
        assert result["quantization_type"] == "nf4"
        assert result["use_double_quant"] is True
        assert result["training_mode"] == "qlora"
        
        # 验证监控器记录了QLoRA训练事件
        mock_monitor.log_event.assert_any_call(
            "qlora_training_start",
            {
                "quantization_bits": 4,
                "lora_rank": 16,
                "epochs": 1
            }
        )
        mock_monitor.log_event.assert_any_call("qlora_training_complete", result)
    
    def test_optimize_for_inference(self, training_config, mock_monitor):
        """测试推理优化"""
        # 创建模拟PEFT模型
        mock_peft_model = Mock()
        mock_merged_model = Mock()
        mock_peft_model.merge_and_unload.return_value = mock_merged_model
        mock_merged_model.config = Mock()
        
        # 创建模拟参数
        mock_param = Mock()
        mock_param.requires_grad = True
        mock_merged_model.parameters.return_value = [mock_param]
        
        trainer = QLoRATrainer(training_config, mock_monitor)
        trainer.peft_model = mock_peft_model
        
        trainer.optimize_for_inference()
        
        # 验证LoRA权重合并
        mock_peft_model.merge_and_unload.assert_called_once()
        assert trainer.model == mock_merged_model
        
        # 验证模型设置为评估模式
        mock_merged_model.eval.assert_called_once()
        
        # 验证参数设置为不需要梯度
        assert mock_param.requires_grad is False
        
        # 验证缓存启用
        assert mock_merged_model.config.use_cache is True
        
        # 验证监控器记录
        mock_monitor.log_event.assert_called_with("qlora_inference_optimization_complete")
    
    def test_optimize_for_inference_without_merge(self, training_config, mock_monitor):
        """测试不支持权重合并的推理优化"""
        # 创建不支持merge_and_unload的模拟PEFT模型
        mock_peft_model = Mock(spec=[])  # 没有merge_and_unload方法
        mock_peft_model.config = Mock()
        
        mock_param = Mock()
        mock_param.requires_grad = True
        mock_peft_model.parameters.return_value = [mock_param]
        
        trainer = QLoRATrainer(training_config, mock_monitor)
        trainer.peft_model = mock_peft_model
        trainer.model = mock_peft_model
        
        trainer.optimize_for_inference()
        
        # 验证模型设置为评估模式
        mock_peft_model.eval.assert_called_once()
        
        # 验证参数设置为不需要梯度
        assert mock_param.requires_grad is False
    
    def test_optimize_for_inference_failure(self, training_config, mock_monitor):
        """测试推理优化失败"""
        trainer = QLoRATrainer(training_config, mock_monitor)
        
        with pytest.raises(ValueError, match="PEFT模型尚未初始化"):
            trainer.optimize_for_inference()
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.empty_cache')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.max_memory_allocated')
    @patch('torch.cuda.reset_peak_memory_stats')
    def test_benchmark_memory_usage(self, mock_reset_stats, mock_max_memory, mock_memory_allocated,
                                   mock_empty_cache, mock_cuda_available, training_config, mock_monitor):
        """测试内存基准测试"""
        # 设置内存使用模拟
        mock_memory_allocated.side_effect = [
            1000000000,  # baseline: 1GB
            1500000000,  # pre_inference: 1.5GB  
            2000000000   # post_inference: 2GB
        ]
        mock_max_memory.return_value = 2500000000  # max: 2.5GB
        
        # 创建模拟模型和分词器
        mock_model = Mock()
        mock_model.device = "cuda"
        mock_model.generate.return_value = torch.tensor([[1, 2, 3]])
        
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2]])}
        
        trainer = QLoRATrainer(training_config, mock_monitor)
        trainer.model = mock_model
        trainer.tokenizer = mock_tokenizer
        
        result = trainer.benchmark_memory_usage()
        
        # 验证结果
        assert result["baseline_memory_gb"] == 1.0
        assert result["pre_inference_memory_gb"] == 1.5
        assert result["post_inference_memory_gb"] == 2.0
        assert result["inference_memory_delta_gb"] == 0.5
        assert result["max_memory_gb"] == 2.5
        
        # 验证清理调用
        mock_empty_cache.assert_called()
        mock_reset_stats.assert_called()
        
        # 验证监控器记录
        for key, value in result.items():
            mock_monitor.log_metric.assert_any_call(f"memory_benchmark_{key}", value)
    
    @patch('torch.cuda.is_available', return_value=False)
    def test_benchmark_memory_usage_no_cuda(self, mock_cuda_available, training_config, mock_monitor):
        """测试无CUDA环境的内存基准测试"""
        trainer = QLoRATrainer(training_config, mock_monitor)
        
        result = trainer.benchmark_memory_usage()
        
        # 验证返回空结果
        assert result == {}
    
    def test_compare_with_full_precision(self, training_config, mock_monitor):
        """测试与全精度模型比较"""
        # 创建模拟模型和分词器
        mock_model = Mock()
        mock_model.device = "cuda"
        mock_model.eval.return_value = None
        
        mock_tokenizer = Mock()
        mock_tokenizer.eos_token_id = 2
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.to.return_value = {"input_ids": torch.tensor([[1, 2]])}
        mock_tokenizer.decode.return_value = "Generated text response"
        
        # 模拟生成和计时
        with patch('torch.cuda.Event') as mock_event_cls:
            mock_start_event = Mock()
            mock_end_event = Mock()
            mock_start_event.elapsed_time.return_value = 150.5  # 150.5ms
            mock_event_cls.return_value = Mock()
            mock_event_cls.side_effect = [mock_start_event, mock_end_event]
            
            with patch('torch.cuda.synchronize'), \
                 patch('torch.cuda.memory_allocated', return_value=2000000000), \
                 patch('torch.cuda.memory_reserved', return_value=3000000000):
                
                mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
                
                trainer = QLoRATrainer(training_config, mock_monitor)
                trainer.model = mock_model
                trainer.tokenizer = mock_tokenizer
                
                test_inputs = ["Test input 1", "Test input 2"]
                result = trainer.compare_with_full_precision(test_inputs)
        
        # 验证结果结构
        assert "quantized_outputs" in result
        assert "memory_usage" in result
        assert len(result["quantized_outputs"]) == 2
        
        # 验证输出格式
        for output in result["quantized_outputs"]:
            assert "input" in output
            assert "output" in output
            assert "inference_time_ms" in output
        
        # 验证内存使用记录
        assert result["memory_usage"]["allocated_gb"] == 2.0
        assert result["memory_usage"]["reserved_gb"] == 3.0
        
        # 验证监控器调用
        mock_monitor.log_event.assert_called_with(
            "model_comparison_complete",
            {
                "test_inputs_count": 2,
                "avg_inference_time": 150.5
            }
        )
    
    def test_compare_with_full_precision_no_model(self, training_config, mock_monitor):
        """测试没有模型时的比较操作"""
        trainer = QLoRATrainer(training_config, mock_monitor)
        
        with pytest.raises(ValueError, match="模型或分词器未初始化"):
            trainer.compare_with_full_precision(["test"])


@pytest.mark.integration
class TestQLoRATrainerIntegration:
    """QLoRA训练器集成测试"""
    
    def test_quantization_config_combinations(self):
        """测试不同量化配置组合"""
        configs = [
            (QuantizationType.NF4, 4, True),
            (QuantizationType.FP4, 4, False), 
            (QuantizationType.INT8, 8, False)
        ]
        
        for quant_type, bits, double_quant in configs:
            with tempfile.TemporaryDirectory() as temp_dir:
                quant_config = QuantizationConfig(
                    quantization_type=quant_type,
                    bits=bits,
                    use_double_quant=double_quant
                )
                
                config = TrainingConfig(
                    model_name="facebook/opt-125m",
                    model_architecture=ModelArchitecture.LLAMA,
                    training_mode=TrainingMode.QLORA,
                    dataset_path="test.json",
                    output_dir=temp_dir,
                    quantization_config=quant_config
                )
                
                monitor = Mock()
                trainer = QLoRATrainer(config, monitor)
                
                # 验证配置正确设置
                assert trainer.config.quantization_config.quantization_type == quant_type
                assert trainer.config.quantization_config.bits == bits
                assert trainer.config.quantization_config.use_double_quant == double_quant