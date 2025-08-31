"""
模型注册表使用示例

展示如何使用ModelRegistry管理不同类型的AI模型
"""

import os
import tempfile
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from model_registry import (
    ModelRegistry, ModelFormat, ModelType, CompressionType,
    register_pytorch_model, register_onnx_model, register_huggingface_model
)


def example_pytorch_model_registration():
    """PyTorch模型注册示例"""
    print("\n=== PyTorch模型注册示例 ===")
    
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
            
            print(f"✅ 成功注册PyTorch模型: {entry.metadata.name}:{entry.metadata.version}")
            print(f"   参数数量: {entry.metadata.parameters_count:,}")
            print(f"   模型大小: {entry.metadata.model_size_mb:.2f} MB")
            
            # 加载模型
            loaded_model, _ = registry.load_model("simple_classifier", "1.0.0")
            print(f"✅ 成功加载模型: {type(loaded_model).__name__}")
            
            # 测试推理
            test_input = torch.randn(1, 1, 28, 28)
            with torch.no_grad():
                output = loaded_model(test_input)
                print(f"   测试输出形状: {output.shape}")
    
    except ImportError:
        print("❌ PyTorch未安装，跳过PyTorch示例")
    except Exception as e:
        print(f"❌ PyTorch示例失败: {e}")


def example_onnx_model_registration():
    """ONNX模型注册示例"""
    print("\n=== ONNX模型注册示例 ===")
    
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
                
                print(f"✅ 成功注册ONNX模型: {entry.metadata.name}:{entry.metadata.version}")
                print(f"   IR版本: {onnx_model.ir_version}")
                print(f"   Producer: {onnx_model.producer_name}")
                print(f"   模型大小: {entry.metadata.model_size_mb:.2f} MB")
                
                # 加载模型
                loaded_onnx_model, _ = registry.load_model("simple_onnx_net")
                print(f"✅ 成功加载ONNX模型")
                
        finally:
            if os.path.exists(onnx_path):
                os.unlink(onnx_path)
    
    except ImportError as e:
        print(f"❌ 缺少依赖，跳过ONNX示例: {e}")
    except Exception as e:
        print(f"❌ ONNX示例失败: {e}")


def example_huggingface_model_registration():
    """HuggingFace模型注册示例"""
    print("\n=== HuggingFace模型注册示例 ===")
    
    try:
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        # 使用小型模型进行演示
        model_name = "distilbert-base-uncased"
        
        print(f"正在加载HuggingFace模型: {model_name}...")
        
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
            
            print(f"✅ 成功注册HuggingFace模型: {entry.metadata.name}:{entry.metadata.version}")
            print(f"   参数数量: {entry.metadata.parameters_count:,}")
            print(f"   模型大小: {entry.metadata.model_size_mb:.2f} MB")
            print(f"   配置类型: {config.model_type}")
            
            # 加载模型
            loaded_model, loaded_tokenizer = registry.load_model("distilbert_demo")
            print(f"✅ 成功加载HuggingFace模型和tokenizer")
            
            # 测试tokenizer
            test_text = "Hello, this is a test sentence."
            tokens = loaded_tokenizer.encode(test_text, return_tensors="pt")
            print(f"   测试文本: '{test_text}'")
            print(f"   Token数量: {tokens.shape[1]}")
            
            # 测试模型推理
            with torch.no_grad():
                outputs = loaded_model(tokens)
                hidden_states = outputs.last_hidden_state
                print(f"   输出形状: {hidden_states.shape}")
    
    except ImportError:
        print("❌ Transformers未安装，跳过HuggingFace示例")
    except Exception as e:
        print(f"❌ HuggingFace示例失败: {e}")


def example_model_management_operations():
    """模型管理操作示例"""
    print("\n=== 模型管理操作示例 ===")
    
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
                print(f"✅ 注册模型: {entry.metadata.name}")
            except Exception as e:
                print(f"❌ 注册失败: {e}")
        
        print(f"\n📊 当前注册模型数量: {len(registry.models)}")
        
        # 列出所有模型
        print("\n📋 所有模型:")
        all_models = registry.list_models()
        for model in all_models:
            print(f"   - {model.metadata.name}:{model.metadata.version} "
                  f"({model.metadata.format.value}, {model.metadata.model_type.value})")
        
        # 按类型筛选
        print("\n🔍 分类模型:")
        classification_models = registry.list_models(ModelType.CLASSIFICATION)
        for model in classification_models:
            print(f"   - {model.metadata.name}: {model.metadata.description}")
        
        # 获取模型信息
        print("\n📄 模型详细信息:")
        model_info = registry.get_model_info("model_a", "latest")
        if model_info:
            print(f"   模型: {model_info.metadata.name}")
            print(f"   版本: {model_info.metadata.version}")
            print(f"   类型: {model_info.metadata.model_type.value}")
            print(f"   格式: {model_info.metadata.format.value}")
            print(f"   描述: {model_info.metadata.description}")
        
        # 验证注册表
        print("\n🔍 验证注册表:")
        validation = registry.validate_registry()
        print(f"   总模型数: {validation['total_models']}")
        print(f"   有效模型数: {validation['valid_models']}")
        print(f"   错误数: {len(validation['errors'])}")
        print(f"   警告数: {len(validation['warnings'])}")
        
        if validation['errors']:
            print("   错误:")
            for error in validation['errors']:
                print(f"     - {error}")
        
        if validation['warnings']:
            print("   警告:")
            for warning in validation['warnings']:
                print(f"     - {warning}")


def example_model_versioning():
    """模型版本控制示例"""
    print("\n=== 模型版本控制示例 ===")
    
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
            print(f"✅ 注册版本 {version_info['version']}: {version_info['description']}")
        
        # 列出所有版本
        print(f"\n📋 '{model_name}' 的所有版本:")
        all_models = registry.list_models()
        evolving_models = [m for m in all_models if m.metadata.name == model_name]
        
        # 按版本排序
        evolving_models.sort(key=lambda x: x.metadata.version, reverse=True)
        
        for model in evolving_models:
            accuracy = model.metadata.performance_metrics.get("accuracy", "N/A")
            print(f"   - v{model.metadata.version}: {model.metadata.description} "
                  f"(准确率: {accuracy})")
        
        # 获取最新版本
        latest_model = registry.get_model_info(model_name, "latest")
        print(f"\n🔥 最新版本: v{latest_model.metadata.version}")
        
        # 删除旧版本
        print(f"\n🗑️  删除旧版本 v1.0.0")
        registry.remove_model(model_name, "1.0.0")
        
        # 确认删除
        remaining_models = [m for m in registry.list_models() if m.metadata.name == model_name]
        print(f"   剩余版本数: {len(remaining_models)}")


def example_convenience_functions():
    """便捷函数使用示例"""
    print("\n=== 便捷函数使用示例 ===")
    
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
                
                print(f"✅ 使用便捷函数注册模型: {entry.metadata.name}")
                print(f"   参数数量: {entry.metadata.parameters_count:,}")
                
            finally:
                # 恢复原始注册表
                model_registry.model_registry = original_registry
    
    except ImportError:
        print("❌ PyTorch未安装，跳过便捷函数示例")
    except Exception as e:
        print(f"❌ 便捷函数示例失败: {e}")


def example_error_handling():
    """错误处理示例"""
    print("\n=== 错误处理示例 ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        registry = ModelRegistry(temp_dir)
        
        # 尝试加载不存在的模型
        try:
            registry.load_model("non_existent_model")
        except ValueError as e:
            print(f"✅ 正确捕获错误: {e}")
        
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
            print(f"✅ 正确捕获重复注册错误: {e}")
        
        # 尝试使用不支持的格式
        try:
            registry.register_model(
                name="unsupported_test",
                model={"dummy": True},
                model_format=ModelFormat.SAFETENSORS  # 假设没有loader
            )
        except ValueError as e:
            print(f"✅ 正确捕获不支持格式错误: {e}")


def main():
    """运行所有示例"""
    print("🚀 模型注册表使用示例")
    print("=" * 50)
    
    try:
        example_pytorch_model_registration()
        example_onnx_model_registration()
        example_huggingface_model_registration()
        example_model_management_operations()
        example_model_versioning()
        example_convenience_functions()
        example_error_handling()
        
        print("\n✅ 所有示例完成!")
        
    except Exception as e:
        print(f"\n❌ 运行示例时出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()