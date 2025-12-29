"""
模型注册表使用示例

展示如何使用ModelRegistry管理不同类型的AI模型
"""

import os
import tempfile
from pathlib import Path
from model_registry import (
    ModelRegistry, ModelFormat, ModelType, CompressionType,
    register_pytorch_model, register_onnx_model, register_huggingface_model
)
from src.core.logging import setup_logging

from src.core.logging import get_logger
logger = get_logger(__name__)

setup_logging()

def example_pytorch_model_registration():
    """PyTorch模型注册示例"""
    logger.info("PyTorch模型注册示例开始")
    
    try:
        import torch
        import torch.nn as nn
        
        # 创建一个简单的PyTorch模型
        class SimpleClassifier(nn.Module):
            def __init__(self, input_size=784, hidden_size=128, num_classes=10):
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear_relu_stack = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, num_classes)
                )
            
            def forward(self, x):
                x = self.flatten(x)
                logits = self.linear_relu_stack(x)
                return logits
        
        # 创建模型实例
        model = SimpleClassifier()
        
        # 创建临时注册表
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            
            # 注册模型
            entry = registry.register_model(
                name="simple_classifier",
                model=model,
                model_format=ModelFormat.PYTORCH,
                model_type=ModelType.CLASSIFICATION,
                version="1.0.0",
                description="简单的MNIST分类器",
                author="示例作者",
                training_dataset="MNIST",
                training_epochs=10,
                performance_metrics={"accuracy": 0.95, "loss": 0.05},
                tags=["mnist", "classification", "simple"],
                input_shape=[1, 28, 28],
                output_shape=[10]
            )
            
            logger.info(
                "成功注册PyTorch模型",
                name=entry.metadata.name,
                version=entry.metadata.version,
                parameters_count=entry.metadata.parameters_count,
                model_size_mb=f"{entry.metadata.model_size_mb:.2f}",
            )
            
            # 加载模型
            loaded_model, _ = registry.load_model("simple_classifier", "1.0.0")
            logger.info("成功加载模型", model_type=type(loaded_model).__name__)
            
            # 测试推理
            test_input = torch.randn(1, 1, 28, 28)
            with torch.no_grad():
                output = loaded_model(test_input)
                logger.info("测试输出形状", output_shape=str(output.shape))
    
    except ImportError:
        logger.warning("PyTorch未安装，跳过PyTorch示例")
    except Exception as e:
        logger.error("PyTorch示例失败", error=str(e), exc_info=True)

def example_onnx_model_registration():
    """ONNX模型注册示例"""
    logger.info("ONNX模型注册示例开始")
    
    try:
        import torch
        import torch.nn as nn
        import onnx
        import tempfile
        
        # 创建PyTorch模型
        class SimpleNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 5)
            
            def forward(self, x):
                return self.fc(x)
        
        model = SimpleNet()
        
        # 导出为ONNX
        dummy_input = torch.randn(1, 10)
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            onnx_path = f.name
        
        try:
            torch.onnx.export(
                model, dummy_input, onnx_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
            
            # 加载ONNX模型
            onnx_model = onnx.load(onnx_path)
            
            # 创建临时注册表
            with tempfile.TemporaryDirectory() as temp_dir:
                registry = ModelRegistry(temp_dir)
                
                # 注册ONNX模型
                entry = registry.register_model(
                    name="simple_onnx_net",
                    model=onnx_model,
                    model_format=ModelFormat.ONNX,
                    model_type=ModelType.CUSTOM,
                    version="1.0.0",
                    description="转换后的ONNX模型",
                    author="示例作者",
                    training_framework="PyTorch -> ONNX",
                    tags=["onnx", "converted"],
                    input_shape=[1, 10],
                    output_shape=[1, 5]
                )
                
                logger.info(
                    "成功注册ONNX模型",
                    name=entry.metadata.name,
                    version=entry.metadata.version,
                    ir_version=onnx_model.ir_version,
                    producer=onnx_model.producer_name,
                    model_size_mb=f"{entry.metadata.model_size_mb:.2f}",
                )
                
                # 加载模型
                loaded_onnx_model, _ = registry.load_model("simple_onnx_net")
                logger.info("成功加载ONNX模型")
                
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)
    
    except ImportError as e:
        logger.warning("缺少依赖，跳过ONNX示例", error=str(e))
    except Exception as e:
        logger.error("ONNX示例失败", error=str(e), exc_info=True)

def example_huggingface_model_registration():
    """HuggingFace模型注册示例"""
    logger.info("HuggingFace模型注册示例开始")
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        # 使用小型模型进行演示
        model_name = "distilbert-base-uncased"
        
        logger.info("正在加载HuggingFace模型", model_name=model_name)
        
        # 加载模型和tokenizer
        model = AutoModel.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        config = AutoConfig.from_pretrained(model_name)
        
        # 创建临时注册表
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(temp_dir)
            
            # 注册HuggingFace模型
            entry = registry.register_model(
                name="distilbert_demo",
                model=model,
                tokenizer=tokenizer,
                model_format=ModelFormat.HUGGINGFACE,
                model_type=ModelType.LANGUAGE_MODEL,
                version="1.0.0",
                description="DistilBERT演示模型",
                author="Hugging Face",
                training_dataset="English Wikipedia and BookCorpus",
                tags=["bert", "distilbert", "nlp", "transformer"],
                repository_url=f"https://huggingface.co/{model_name}",
                license="Apache-2.0"
            )
            
            logger.info(
                "成功注册HuggingFace模型",
                name=entry.metadata.name,
                version=entry.metadata.version,
                parameters_count=entry.metadata.parameters_count,
                model_size_mb=f"{entry.metadata.model_size_mb:.2f}",
                config_type=config.model_type,
            )
            
            # 加载模型
            loaded_model, loaded_tokenizer = registry.load_model("distilbert_demo")
            logger.info("成功加载HuggingFace模型和tokenizer")
            
            # 测试tokenizer
            test_text = "Hello, this is a test sentence."
            tokens = loaded_tokenizer.encode(test_text, return_tensors="pt")
            logger.info("测试文本", text=test_text)
            logger.info("Token数量", token_count=int(tokens.shape[1]))
            
            # 测试模型推理
            with torch.no_grad():
                outputs = loaded_model(tokens)
                hidden_states = outputs.last_hidden_state
                logger.info("输出形状", output_shape=str(hidden_states.shape))
    
    except ImportError:
        logger.warning("Transformers未安装，跳过HuggingFace示例")
    except Exception as e:
        logger.error("HuggingFace示例失败", error=str(e), exc_info=True)

def example_model_management_operations():
    """模型管理操作示例"""
    logger.info("模型管理操作示例开始")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(temp_dir)
        
        # 创建一些示例模型
        dummy_models = [
            {
                "name": "model_a",
                "model": {"type": "dummy", "params": 1000},
                "format": ModelFormat.PYTORCH,
                "model_type": ModelType.CLASSIFICATION,
                "description": "分类模型A"
            },
            {
                "name": "model_b", 
                "model": {"type": "dummy", "params": 2000},
                "format": ModelFormat.ONNX,
                "model_type": ModelType.GENERATION,
                "description": "生成模型B"
            },
            {
                "name": "model_c",
                "model": {"type": "dummy", "params": 500},
                "format": ModelFormat.PYTORCH,
                "model_type": ModelType.EMBEDDING,
                "description": "嵌入模型C"
            }
        ]
        
        # 注册模型（使用mock loader）
        from unittest.mock import Mock
        mock_loader = Mock()
        mock_loader.save = Mock()
        mock_loader.get_metadata.return_value = {"total_parameters": 1000, "model_size_mb": 10.0}
        
        for format_type in [ModelFormat.PYTORCH, ModelFormat.ONNX]:
            registry.loaders[format_type] = mock_loader
        
        for model_info in dummy_models:
            try:
                entry = registry.register_model(**model_info)
                logger.info("注册模型成功", name=entry.metadata.name)
            except Exception as e:
                logger.error("注册失败", error=str(e), exc_info=True)
        
        logger.info("当前注册模型数量", model_count=len(registry.models))
        
        # 列出所有模型
        logger.info("所有模型")
        all_models = registry.list_models()
        for model in all_models:
            logger.info(
                "模型条目",
                name=model.metadata.name,
                version=model.metadata.version,
                model_format=model.metadata.format.value,
                model_type=model.metadata.model_type.value,
            )
        
        # 按类型筛选
        logger.info("分类模型")
        classification_models = registry.list_models(ModelType.CLASSIFICATION)
        for model in classification_models:
            logger.info("分类模型条目", name=model.metadata.name, description=model.metadata.description)
        
        # 获取模型信息
        logger.info("模型详细信息")
        model_info = registry.get_model_info("model_a", "latest")
        if model_info:
            logger.info("模型名称", name=model_info.metadata.name)
            logger.info("模型版本", version=model_info.metadata.version)
            logger.info("模型类型", model_type=model_info.metadata.model_type.value)
            logger.info("模型格式", model_format=model_info.metadata.format.value)
            logger.info("模型描述", description=model_info.metadata.description)
        
        # 验证注册表
        logger.info("验证注册表")
        validation = registry.validate_registry()
        logger.info("总模型数", total_models=validation["total_models"])
        logger.info("有效模型数", valid_models=validation["valid_models"])
        logger.info("错误数", error_count=len(validation["errors"]))
        logger.info("警告数", warning_count=len(validation["warnings"]))
        
        if validation["errors"]:
            logger.error("注册表验证错误", errors=validation["errors"])
        
        if validation["warnings"]:
            logger.warning("注册表验证警告", warnings=validation["warnings"])

def example_model_versioning():
    """模型版本控制示例"""
    logger.info("模型版本控制示例开始")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(temp_dir)
        
        # Mock loader
        mock_loader = Mock()
        mock_loader.save = Mock()
        mock_loader.get_metadata.return_value = {"total_parameters": 1000, "model_size_mb": 10.0}
        registry.loaders[ModelFormat.PYTORCH] = mock_loader
        
        model_name = "evolving_model"
        
        # 注册多个版本
        versions = [
            {"version": "1.0.0", "accuracy": 0.80, "description": "初始版本"},
            {"version": "1.1.0", "accuracy": 0.85, "description": "改进的训练流程"},
            {"version": "2.0.0", "accuracy": 0.90, "description": "全新架构"},
            {"version": "2.1.0", "accuracy": 0.92, "description": "微调优化"}
        ]
        
        for version_info in versions:
            registry.register_model(
                name=model_name,
                model={"dummy": True},
                model_format=ModelFormat.PYTORCH,
                version=version_info["version"],
                description=version_info["description"],
                performance_metrics={"accuracy": version_info["accuracy"]}
            )
            logger.info(
                "注册版本",
                version=version_info["version"],
                description=version_info["description"],
            )
        
        # 列出所有版本
        logger.info("模型所有版本", model_name=model_name)
        all_models = registry.list_models()
        evolving_models = [m for m in all_models if m.metadata.name == model_name]
        
        # 按版本排序
        evolving_models.sort(key=lambda x: x.metadata.version, reverse=True)
        
        for model in evolving_models:
            accuracy = model.metadata.performance_metrics.get("accuracy", "N/A")
            logger.info(
                "模型版本",
                version=model.metadata.version,
                description=model.metadata.description,
                accuracy=accuracy,
            )
        
        # 获取最新版本
        latest_model = registry.get_model_info(model_name, "latest")
        logger.info("最新版本", version=latest_model.metadata.version)
        
        # 删除旧版本
        logger.info("删除旧版本", version="1.0.0")
        registry.remove_model(model_name, "1.0.0")
        
        # 确认删除
        remaining_models = [m for m in registry.list_models() if m.metadata.name == model_name]
        logger.info("剩余版本数", remaining_count=len(remaining_models))

def example_convenience_functions():
    """便捷函数使用示例"""
    logger.info("便捷函数使用示例开始")
    
    try:
        import torch
        import torch.nn as nn
        
        # 创建简单模型
        model = nn.Linear(10, 1)
        
        # 使用便捷函数注册PyTorch模型
        with tempfile.TemporaryDirectory() as temp_dir:
            # 临时设置全局注册表路径
            import model_registry
            original_registry = model_registry.model_registry
            temp_registry = ModelRegistry(temp_dir)
            model_registry.model_registry = temp_registry
            
            try:
                entry = register_pytorch_model(
                    name="linear_model",
                    model=model,
                    version="1.0.0",
                    description="线性回归模型",
                    model_type=ModelType.CUSTOM,
                    tags=["linear", "regression"]
                )
                
                logger.info(
                    "使用便捷函数注册模型",
                    name=entry.metadata.name,
                    parameters_count=entry.metadata.parameters_count,
                )
                
            finally:
                # 恢复原始注册表
                model_registry.model_registry = original_registry
    
    except ImportError:
        logger.warning("PyTorch未安装，跳过便捷函数示例")
    except Exception as e:
        logger.error("便捷函数示例失败", error=str(e), exc_info=True)

def example_error_handling():
    """错误处理示例"""
    logger.info("错误处理示例开始")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(temp_dir)
        
        # 尝试加载不存在的模型
        try:
            registry.load_model("non_existent_model")
        except ValueError as e:
            logger.info("正确捕获错误", error=str(e))
        
        # 尝试注册重复模型
        mock_loader = Mock()
        mock_loader.save = Mock()
        mock_loader.get_metadata.return_value = {}
        registry.loaders[ModelFormat.PYTORCH] = mock_loader
        
        try:
            # 第一次注册
            registry.register_model(
                name="duplicate_test",
                model={"dummy": True},
                model_format=ModelFormat.PYTORCH
            )
            
            # 第二次注册（应该失败）
            registry.register_model(
                name="duplicate_test",
                model={"dummy": True},
                model_format=ModelFormat.PYTORCH
            )
        except ValueError as e:
            logger.info("正确捕获重复注册错误", error=str(e))
        
        # 尝试使用不支持的格式
        try:
            registry.register_model(
                name="unsupported_test",
                model={"dummy": True},
                model_format=ModelFormat.SAFETENSORS  # 假设没有loader
            )
        except ValueError as e:
            logger.info("正确捕获不支持格式错误", error=str(e))

def main():
    """运行所有示例"""
    logger.info("模型注册表使用示例开始")
    
    try:
        example_pytorch_model_registration()
        example_onnx_model_registration()
        example_huggingface_model_registration()
        example_model_management_operations()
        example_model_versioning()
        example_convenience_functions()
        example_error_handling()
        
        logger.info("所有示例完成")
        
    except Exception as e:
        logger.error("运行示例时出错", error=str(e), exc_info=True)

if __name__ == "__main__":
    main()
