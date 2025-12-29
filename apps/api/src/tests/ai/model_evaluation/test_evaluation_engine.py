import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from src.core.utils.timezone_utils import utc_now, utc_factory
from pathlib import Path
import asyncio
import tempfile
import shutil
from ai.model_evaluation.evaluation_engine import (
    ModelEvaluationEngine,
    EvaluationConfig,
    BenchmarkConfig,
    EvaluationMetrics,
    EvaluationResult,
    BatchEvaluationManager
)

class TestModelEvaluationEngine:
    """测试模型评估引擎"""
    
    @pytest.fixture
    def evaluation_config(self):
        """创建测试用的评估配置"""
        return EvaluationConfig(
            model_name="test_model",
            model_path="test/path",
            task_type="text_generation",
            device="cpu",
            batch_size=2,
            max_length=128,
            precision="fp32",
            enable_optimizations=False
        )
    
    @pytest.fixture
    def benchmark_config(self):
        """创建测试用的基准测试配置"""
        return BenchmarkConfig(
            name="test_benchmark",
            tasks=["test_task"],
            num_fewshot=0,
            batch_size=2,
            device="cpu"
        )
    
    @pytest.fixture
    def mock_model_engine(self, evaluation_config):
        """创建模拟的模型评估引擎"""
        with patch('transformers.AutoModelForCausalLM'), \
             patch('transformers.AutoTokenizer'), \
             patch('transformers.AutoConfig'):
            engine = ModelEvaluationEngine(evaluation_config)
            engine.model = Mock()
            engine.tokenizer = Mock()
            return engine
    
    def test_evaluation_config_creation(self):
        """测试评估配置创建"""
        config = EvaluationConfig(
            model_name="test_model",
            model_path="test/path",
            task_type="classification"
        )
        
        assert config.model_name == "test_model"
        assert config.model_path == "test/path"
        assert config.task_type == "classification"
        assert config.device == "cuda" if torch.cuda.is_available() else "cpu"
        assert config.batch_size == 8
        assert config.precision == "fp16"
    
    def test_benchmark_config_creation(self):
        """测试基准测试配置创建"""
        config = BenchmarkConfig(
            name="test_benchmark",
            tasks=["task1", "task2"]
        )
        
        assert config.name == "test_benchmark"
        assert config.tasks == ["task1", "task2"]
        assert config.num_fewshot == 0
        assert config.batch_size == 8
    
    def test_evaluation_metrics_creation(self):
        """测试评估指标创建"""
        metrics = EvaluationMetrics(
            accuracy=0.85,
            f1_score=0.82,
            inference_time=150.0,
            memory_usage=1024.0,
            throughput=6.7
        )
        
        assert metrics.accuracy == 0.85
        assert metrics.f1_score == 0.82
        assert metrics.inference_time == 150.0
        assert metrics.memory_usage == 1024.0
        assert metrics.throughput == 6.7
    
    def test_evaluation_result_creation(self, evaluation_config):
        """测试评估结果创建"""
        metrics = EvaluationMetrics(accuracy=0.85, inference_time=150.0, memory_usage=1024.0, throughput=6.7)
        
        result = EvaluationResult(
            model_name="test_model",
            benchmark_name="test_benchmark", 
            task_name="test_task",
            metrics=metrics,
            config=evaluation_config,
            timestamp=utc_now(),
            duration=300.0,
            samples_evaluated=100
        )
        
        assert result.model_name == "test_model"
        assert result.benchmark_name == "test_benchmark"
        assert result.task_name == "test_task"
        assert result.metrics.accuracy == 0.85
        assert result.duration == 300.0
        assert result.samples_evaluated == 100
    
    def test_model_engine_initialization(self, evaluation_config):
        """测试模型引擎初始化"""
        engine = ModelEvaluationEngine(evaluation_config)
        
        assert engine.config == evaluation_config
        assert engine.device.type == evaluation_config.device
        assert engine.model is None
        assert engine.tokenizer is None
        assert isinstance(engine.metrics_cache, dict)
    
    def test_torch_dtype_mapping(self, evaluation_config):
        """测试torch数据类型映射"""
        engine = ModelEvaluationEngine(evaluation_config)
        
        # 测试fp16
        evaluation_config.precision = "fp16"
        assert engine._get_torch_dtype() == torch.float16
        
        # 测试bf16
        evaluation_config.precision = "bf16"
        assert engine._get_torch_dtype() == torch.bfloat16
        
        # 测试fp32
        evaluation_config.precision = "fp32"
        assert engine._get_torch_dtype() == torch.float32
    
    def test_hardware_info_collection(self, mock_model_engine):
        """测试硬件信息收集"""
        info = mock_model_engine._get_hardware_info()
        
        assert "cpu_count" in info
        assert "memory_total" in info
        assert "memory_available" in info
        assert info["cpu_count"] > 0
        assert info["memory_total"] > 0
    
    @patch('lm_eval.evaluator.simple_evaluate')
    @patch('lm_eval.models.huggingface.HFLM')
    def test_evaluate_with_lm_eval(self, mock_hflm, mock_evaluate, mock_model_engine, benchmark_config):
        """测试使用lm-evaluation-harness进行评估"""
        # 模拟评估结果
        mock_eval_results = {
            "results": {
                "test_task": {
                    "acc": 0.85,
                    "f1": 0.82,
                    "num_samples": 100
                }
            }
        }
        mock_evaluate.return_value = mock_eval_results
        mock_hflm.return_value = Mock()
        
        results = mock_model_engine.evaluate_with_lm_eval(benchmark_config)
        
        assert len(results) == 1
        assert results[0].model_name == "test_model"
        assert results[0].benchmark_name == "test_benchmark"
        assert results[0].task_name == "test_task"
        assert results[0].metrics.accuracy == 0.85
        assert results[0].samples_evaluated == 100
        
        # 验证调用
        mock_hflm.assert_called_once()
        mock_evaluate.assert_called_once()
    
    @patch('lm_eval.evaluator.simple_evaluate')
    @patch('lm_eval.models.huggingface.HFLM')
    def test_evaluate_with_error(self, mock_hflm, mock_evaluate, mock_model_engine, benchmark_config):
        """测试评估过程中出现错误的情况"""
        # 模拟评估失败
        mock_evaluate.side_effect = Exception("Evaluation failed")
        mock_hflm.return_value = Mock()
        
        results = mock_model_engine.evaluate_with_lm_eval(benchmark_config)
        
        assert len(results) == 1
        assert results[0].error == "Evaluation failed"
        assert results[0].task_name == "failed"
    
    def test_compute_accuracy(self, mock_model_engine):
        """测试准确率计算"""
        # 模拟数据集
        dataset = [
            {"input": ["test input"], "target": ["expected output"]},
            {"input": ["test input 2"], "target": ["expected output 2"]}
        ]
        
        # 模拟预测结果
        with patch.object(mock_model_engine, '_generate_predictions') as mock_predict:
            mock_predict.side_effect = [
                ["expected output"],  # 第一批预测正确
                ["wrong output"]      # 第二批预测错误
            ]
            
            accuracy = mock_model_engine._compute_accuracy(dataset)
            
            assert accuracy == 0.5  # 2个样本中1个正确
            assert mock_predict.call_count == 2
    
    def test_compute_perplexity(self, mock_model_engine):
        """测试困惑度计算"""
        # 模拟数据集
        dataset = [{"text": ["test text"]}]
        
        # 模拟模型输出
        mock_output = Mock()
        mock_output.loss = torch.tensor(2.0)  # ln(perplexity) = 2, 所以perplexity = e^2 ≈ 7.39
        
        mock_model_engine.model.eval.return_value = None
        mock_model_engine.model.return_value = mock_output
        mock_model_engine.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]])
        }
        
        with patch.object(mock_model_engine.tokenizer, '__call__') as mock_tokenizer:
            mock_tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.tensor([[1, 1, 1, 1]])
            }
            mock_tokenizer.return_value.to = Mock(return_value=mock_tokenizer.return_value)
            
            perplexity = mock_model_engine._compute_perplexity(dataset)
            
            # 验证困惑度计算
            expected_perplexity = torch.exp(torch.tensor(2.0)).item()
            assert abs(perplexity - expected_perplexity) < 0.1
    
    @patch('evaluate.load')
    def test_compute_bleu(self, mock_evaluate_load, mock_model_engine):
        """测试BLEU分数计算"""
        # 模拟BLEU度量
        mock_bleu_metric = Mock()
        mock_bleu_metric.compute.return_value = {"bleu": 0.75}
        mock_evaluate_load.return_value = mock_bleu_metric
        
        # 模拟数据集
        dataset = [{"input": ["test input"], "target": ["expected output"]}]
        
        with patch.object(mock_model_engine, '_generate_predictions') as mock_predict:
            mock_predict.return_value = ["predicted output"]
            
            bleu_score = mock_model_engine._compute_bleu(dataset)
            
            assert bleu_score == 0.75
            mock_bleu_metric.compute.assert_called_once_with(
                predictions=["predicted output"],
                references=["expected output"]
            )
    
    @patch('evaluate.load')
    def test_compute_rouge(self, mock_evaluate_load, mock_model_engine):
        """测试ROUGE分数计算"""
        # 模拟ROUGE度量
        mock_rouge_metric = Mock()
        mock_rouge_metric.compute.return_value = {
            "rouge1": 0.75,
            "rouge2": 0.65,
            "rougeL": 0.70
        }
        mock_evaluate_load.return_value = mock_rouge_metric
        
        # 模拟数据集
        dataset = [{"input": ["test input"], "target": ["expected output"]}]
        
        with patch.object(mock_model_engine, '_generate_predictions') as mock_predict:
            mock_predict.return_value = ["predicted output"]
            
            rouge_scores = mock_model_engine._compute_rouge(dataset)
            
            assert rouge_scores["rouge1"] == 0.75
            assert rouge_scores["rouge2"] == 0.65
            assert rouge_scores["rougeL"] == 0.70
    
    def test_generate_predictions(self, mock_model_engine):
        """测试预测生成"""
        inputs = ["test input 1", "test input 2"]
        
        # 模拟tokenizer和model
        mock_model_engine.tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        }
        mock_model_engine.tokenizer.return_value.to = Mock(return_value=mock_model_engine.tokenizer.return_value)
        
        mock_model_engine.model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_model_engine.tokenizer.decode.return_value = "generated text"
        mock_model_engine.tokenizer.eos_token_id = 0
        
        predictions = mock_model_engine._generate_predictions(inputs)
        
        assert len(predictions) == 2
        assert all(pred == "generated text" for pred in predictions)
        assert mock_model_engine.model.generate.call_count == 2
    
    def test_cleanup(self, mock_model_engine):
        """测试资源清理"""
        # 设置一些资源
        mock_model_engine.model = Mock()
        mock_model_engine.tokenizer = Mock()
        
        with patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('gc.collect') as mock_gc_collect:
            
            mock_model_engine.cleanup()
            
            assert mock_model_engine.model is None
            assert mock_model_engine.tokenizer is None
            mock_gc_collect.assert_called_once()

class TestBatchEvaluationManager:
    """测试批量评估管理器"""
    
    @pytest.fixture
    def batch_manager(self):
        """创建批量评估管理器"""
        return BatchEvaluationManager(max_concurrent_evaluations=2)
    
    @pytest.fixture
    def model_configs(self):
        """创建测试用的模型配置列表"""
        return [
            EvaluationConfig(
                model_name="model_1",
                model_path="path/1",
                task_type="text_generation",
                device="cpu"
            ),
            EvaluationConfig(
                model_name="model_2", 
                model_path="path/2",
                task_type="text_generation",
                device="cpu"
            )
        ]
    
    @pytest.fixture
    def benchmark_configs(self):
        """创建测试用的基准测试配置列表"""
        return [
            BenchmarkConfig(
                name="benchmark_1",
                tasks=["task_1"],
                device="cpu"
            ),
            BenchmarkConfig(
                name="benchmark_2",
                tasks=["task_2"],
                device="cpu"
            )
        ]
    
    def test_batch_manager_initialization(self, batch_manager):
        """测试批量评估管理器初始化"""
        assert batch_manager.max_concurrent_evaluations == 2
        assert isinstance(batch_manager.active_evaluations, dict)
        assert isinstance(batch_manager.results_storage, list)
    
    @pytest.mark.asyncio
    async def test_evaluate_multiple_models_success(self, batch_manager, model_configs, benchmark_configs):
        """测试成功的多模型评估"""
        # 模拟成功的评估结果
        mock_results = [
            EvaluationResult(
                model_name="model_1",
                benchmark_name="benchmark_1",
                task_name="task_1",
                metrics=EvaluationMetrics(accuracy=0.85, inference_time=100.0, memory_usage=512.0, throughput=10.0),
                config=model_configs[0],
                timestamp=utc_now(),
                duration=60.0,
                samples_evaluated=100
            )
        ]
        
        with patch.object(batch_manager, '_evaluate_model_with_semaphore') as mock_eval:
            mock_eval.return_value = mock_results
            
            results = await batch_manager.evaluate_multiple_models(model_configs, benchmark_configs)
            
            assert len(results) == 2  # 两个模型
            assert "model_1" in results
            assert "model_2" in results
            assert mock_eval.call_count == 2
    
    @pytest.mark.asyncio 
    async def test_evaluate_multiple_models_with_error(self, batch_manager, model_configs, benchmark_configs):
        """测试包含错误的多模型评估"""
        # 模拟一个模型成功，一个模型失败
        def side_effect(*args):
            if "model_1" in str(args):
                return [
                    EvaluationResult(
                        model_name="model_1",
                        benchmark_name="benchmark_1",
                        task_name="task_1",
                        metrics=EvaluationMetrics(accuracy=0.85, inference_time=100.0, memory_usage=512.0, throughput=10.0),
                        config=model_configs[0],
                        timestamp=utc_now(),
                        duration=60.0,
                        samples_evaluated=100
                    )
                ]
            else:
                raise Exception("Model evaluation failed")
        
        with patch.object(batch_manager, '_evaluate_model_with_semaphore') as mock_eval:
            mock_eval.side_effect = side_effect
            
            results = await batch_manager.evaluate_multiple_models(model_configs, benchmark_configs)
            
            assert len(results) == 2
            assert "model_1" in results
            assert "model_2" in results
            assert len(results["model_1"]) == 1  # 成功的结果
            assert len(results["model_2"]) == 0  # 失败的结果为空列表
    
    @pytest.mark.asyncio
    async def test_semaphore_concurrency_control(self, batch_manager, model_configs, benchmark_configs):
        """测试信号量并发控制"""
        call_times = []
        
        async def mock_eval_with_delay(*args):
            call_times.append(utc_now())
            await asyncio.sleep(0.1)  # 模拟评估时间
            return []
        
        with patch.object(batch_manager, '_evaluate_model_with_semaphore', side_effect=mock_eval_with_delay):
            await batch_manager.evaluate_multiple_models(model_configs, benchmark_configs)
            
            # 验证并发控制
            assert len(call_times) == 2
            # 由于信号量限制为2，两个调用应该能同时开始
            time_diff = abs((call_times[1] - call_times[0]).total_seconds())
            assert time_diff < 0.05  # 应该几乎同时开始

class TestEvaluationIntegration:
    """集成测试"""
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_full_evaluation_pipeline(self, temp_dir):
        """测试完整的评估流程"""
        # 创建测试配置
        config = EvaluationConfig(
            model_name="test_pipeline_model",
            model_path="test/path",
            task_type="text_generation",
            device="cpu",
            batch_size=1,
            enable_optimizations=False
        )
        
        benchmark_config = BenchmarkConfig(
            name="test_pipeline_benchmark",
            tasks=["test_task"],
            device="cpu",
            batch_size=1
        )
        
        with patch('transformers.AutoModelForCausalLM'), \
             patch('transformers.AutoTokenizer'), \
             patch('transformers.AutoConfig'), \
             patch('lm_eval.evaluator.simple_evaluate') as mock_evaluate:
            
            # 模拟评估结果
            mock_evaluate.return_value = {
                "results": {
                    "test_task": {
                        "acc": 0.85,
                        "f1": 0.82,
                        "num_samples": 50
                    }
                }
            }
            
            # 执行评估
            engine = ModelEvaluationEngine(config)
            results = engine.evaluate_with_lm_eval(benchmark_config)
            
            # 验证结果
            assert len(results) == 1
            result = results[0]
            assert result.model_name == "test_pipeline_model"
            assert result.benchmark_name == "test_pipeline_benchmark"
            assert result.task_name == "test_task"
            assert result.metrics.accuracy == 0.85
            assert result.samples_evaluated == 50
            assert result.error is None
            
            # 清理
            engine.cleanup()
    
    def test_error_handling_pipeline(self):
        """测试错误处理流程"""
        config = EvaluationConfig(
            model_name="test_error_model",
            model_path="invalid/path",
            task_type="text_generation",
            device="cpu"
        )
        
        benchmark_config = BenchmarkConfig(
            name="test_error_benchmark",
            tasks=["test_task"],
            device="cpu"
        )
        
        with patch('transformers.AutoModelForCausalLM.from_pretrained', side_effect=Exception("Model load failed")):
            engine = ModelEvaluationEngine(config)
            
            # 测试模型加载失败
            with pytest.raises(Exception, match="Model load failed"):
                engine.load_model()
    
    def test_memory_cleanup(self):
        """测试内存清理"""
        config = EvaluationConfig(
            model_name="test_memory_model",
            model_path="test/path",
            task_type="text_generation",
            device="cpu"
        )
        
        with patch('transformers.AutoModelForCausalLM'), \
             patch('transformers.AutoTokenizer'), \
             patch('transformers.AutoConfig'), \
             patch('torch.cuda.empty_cache') as mock_empty_cache, \
             patch('gc.collect') as mock_gc_collect:
            
            engine = ModelEvaluationEngine(config)
            engine.model = Mock()
            engine.tokenizer = Mock()
            
            # 执行清理
            engine.cleanup()
            
            # 验证清理
            assert engine.model is None
            assert engine.tokenizer is None
            mock_gc_collect.assert_called_once()

if __name__ == "__main__":
    pytest.main([__file__])
