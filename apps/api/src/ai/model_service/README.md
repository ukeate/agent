# AI模型注册表系统 (ModelRegistry)

一个统一的AI模型管理系统，支持PyTorch、ONNX和HuggingFace模型的注册、加载、版本控制和元数据管理。

## 特性

### 🎯 核心功能
- **多格式支持**: PyTorch (.pth, .pt), ONNX (.onnx), HuggingFace Transformers
- **安全加载**: 遵循PyTorch `weights_only=True`等安全最佳实践
- **元数据管理**: 自动提取和管理模型元数据（参数数量、大小、性能指标等）
- **版本控制**: 支持模型多版本管理
- **完整性验证**: SHA256校验和确保模型文件完整性
- **RESTful API**: 完整的HTTP API支持

### 🔧 高级功能
- **智能加载**: 自动检测模型格式和类型
- **压缩支持**: 模型压缩和量化信息跟踪
- **序列化**: 灵活的序列化格式支持
- **批量操作**: 支持批量模型管理
- **临时加载**: 上下文管理器支持
- **导出功能**: 支持模型格式转换和导出

## 安装

```bash
# 基础依赖
pip install fastapi pydantic

# PyTorch支持
pip install torch torchvision

# ONNX支持  
pip install onnx onnxruntime

# HuggingFace支持
pip install transformers tokenizers safetensors

# 可选：用于高级功能
pip install numpy pillow
```

## 快速开始

### 1. 基础使用

```python
from model_registry import ModelRegistry, ModelFormat, ModelType

# 创建注册表
registry = ModelRegistry("./models")

# 注册PyTorch模型
import torch
import torch.nn as nn

model = nn.Linear(10, 1)
entry = registry.register_model(
    name="linear_model",
    model=model,
    model_format=ModelFormat.PYTORCH,
    model_type=ModelType.CUSTOM,
    version="1.0.0",
    description="简单线性模型"
)

# 加载模型
loaded_model, tokenizer = registry.load_model("linear_model", "1.0.0")
```

### 2. HuggingFace模型

```python
from transformers import AutoModel, AutoTokenizer

# 从Hub加载
model = AutoModel.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 注册到本地
entry = registry.register_model(
    name="distilbert",
    model=model,
    tokenizer=tokenizer,
    model_format=ModelFormat.HUGGINGFACE,
    model_type=ModelType.LANGUAGE_MODEL,
    description="DistilBERT模型"
)
```

### 3. ONNX模型

```python
import onnx

# 加载ONNX模型
onnx_model = onnx.load("model.onnx")

# 注册
entry = registry.register_model(
    name="onnx_model",
    model=onnx_model,
    model_format=ModelFormat.ONNX,
    description="ONNX格式模型"
)
```

## API文档

### REST API端点

```
GET    /model-registry/models              # 列出所有模型
GET    /model-registry/models/{name}       # 获取模型信息
POST   /model-registry/models/upload       # 上传注册模型
POST   /model-registry/models/{name}/register-from-hub  # 从Hub注册
DELETE /model-registry/models/{name}       # 删除模型
GET    /model-registry/models/{name}/download  # 下载模型
GET    /model-registry/models/{name}/export    # 导出模型
GET    /model-registry/validate             # 验证注册表
GET    /model-registry/stats               # 统计信息
```

### Python API

#### 核心类

```python
class ModelRegistry:
    def register_model(self, name: str, model: Any, model_format: ModelFormat, **kwargs) -> ModelEntry
    def load_model(self, name: str, version: str = "latest") -> Tuple[Any, Optional[Any]]
    def list_models(self, model_type: ModelType = None) -> List[ModelEntry]
    def get_model_info(self, name: str, version: str = "latest") -> Optional[ModelEntry]
    def remove_model(self, name: str, version: str = None) -> bool
    def export_model(self, name: str, version: str, export_path: str, export_format: ModelFormat = None) -> str
    def validate_registry(self) -> Dict[str, List[str]]

class ModelMetadata:
    name: str
    version: str
    format: ModelFormat
    model_type: ModelType
    description: Optional[str]
    parameters_count: Optional[int]
    model_size_mb: Optional[float]
    # ... 更多字段
```

#### 便捷函数

```python
# 快速注册函数
register_pytorch_model(name, model, version="1.0.0", **kwargs)
register_onnx_model(name, model, version="1.0.0", **kwargs)  
register_huggingface_model(name, model, tokenizer=None, version="1.0.0", **kwargs)
```

## 使用示例

### 完整工作流程

```python
import torch
import torch.nn as nn
from model_registry import ModelRegistry, ModelFormat, ModelType

# 1. 创建注册表
registry = ModelRegistry("./my_models")

# 2. 创建和训练模型
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)

model = MNISTClassifier()
# ... 训练过程 ...

# 3. 注册训练好的模型
entry = registry.register_model(
    name="mnist_classifier",
    model=model,
    model_format=ModelFormat.PYTORCH,
    model_type=ModelType.CLASSIFICATION,
    version="1.0.0",
    description="MNIST手写数字分类器",
    author="Your Name",
    training_dataset="MNIST",
    training_epochs=10,
    performance_metrics={
        "accuracy": 0.98,
        "loss": 0.05,
        "f1_score": 0.97
    },
    tags=["mnist", "classification", "cnn"],
    input_shape=[1, 28, 28],
    output_shape=[10]
)

print(f"注册成功! 模型ID: {entry.metadata.name}:{entry.metadata.version}")
print(f"参数数量: {entry.metadata.parameters_count:,}")
print(f"模型大小: {entry.metadata.model_size_mb:.2f} MB")

# 4. 稍后加载使用
loaded_model, _ = registry.load_model("mnist_classifier", "1.0.0")

# 5. 进行推理
test_input = torch.randn(1, 1, 28, 28)
with torch.no_grad():
    predictions = loaded_model(test_input)
    predicted_class = torch.argmax(predictions, dim=1)
    print(f"预测类别: {predicted_class.item()}")
```

### 版本管理

```python
# 注册多个版本
versions = [
    {"version": "1.0.0", "accuracy": 0.95},
    {"version": "1.1.0", "accuracy": 0.97}, 
    {"version": "2.0.0", "accuracy": 0.99}
]

for ver in versions:
    registry.register_model(
        name="my_model",
        model=trained_models[ver["version"]],
        model_format=ModelFormat.PYTORCH,
        version=ver["version"],
        performance_metrics={"accuracy": ver["accuracy"]}
    )

# 列出所有版本
models = registry.list_models()
my_models = [m for m in models if m.metadata.name == "my_model"]
for model in sorted(my_models, key=lambda x: x.metadata.version):
    acc = model.metadata.performance_metrics.get("accuracy", "N/A")
    print(f"版本 {model.metadata.version}: 准确率 {acc}")

# 加载最新版本
latest_model, _ = registry.load_model("my_model", "latest")
```

### 批量管理

```python
# 列出所有分类模型
classification_models = registry.list_models(ModelType.CLASSIFICATION)
print(f"分类模型数量: {len(classification_models)}")

# 验证所有模型
validation_result = registry.validate_registry()
print(f"总模型数: {validation_result['total_models']}")
print(f"有效模型数: {validation_result['valid_models']}")
if validation_result['errors']:
    print("错误:")
    for error in validation_result['errors']:
        print(f"  - {error}")

# 临时加载模型（自动清理）
with registry.temporary_model("my_model") as (model, tokenizer):
    # 使用模型
    result = model(input_data)
    # 退出时自动清理资源
```

## 最佳实践

### 1. 安全性

```python
# 使用安全加载选项
loaded_model, _ = registry.load_model(
    "my_model", 
    weights_only=True  # PyTorch安全加载
)

# 验证模型完整性
entry = registry.get_model_info("my_model")
if not entry.verify_integrity():
    print("⚠️ 模型文件可能已损坏!")
```

### 2. 性能优化

```python
# 大模型使用分片保存
registry.register_model(
    name="large_model",
    model=large_model, 
    model_format=ModelFormat.HUGGINGFACE,
    max_shard_size="2GB"  # HuggingFace模型
)

# 指定设备加载
model, _ = registry.load_model(
    "my_model",
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

### 3. 元数据管理

```python
# 详细的元数据
registry.register_model(
    name="production_model",
    model=model,
    model_format=ModelFormat.PYTORCH,
    model_type=ModelType.CLASSIFICATION,
    
    # 训练信息
    training_framework="PyTorch 2.0",
    training_dataset="Custom Dataset v2.1",
    training_epochs=50,
    
    # 性能指标
    performance_metrics={
        "accuracy": 0.987,
        "precision": 0.985,
        "recall": 0.989,
        "f1_score": 0.987,
        "inference_time_ms": 15.2
    },
    
    # 压缩信息
    compression_type=CompressionType.QUANTIZATION_INT8,
    compression_ratio=0.25,
    original_size_mb=400.0,
    
    # 依赖信息
    dependencies=["torch>=2.0", "torchvision>=0.15"],
    python_version="3.9",
    
    # 标签和文档
    tags=["production", "optimized", "quantized"],
    license="MIT",
    repository_url="https://github.com/myorg/mymodel",
    paper_url="https://arxiv.org/abs/xxxx.xxxxx"
)
```

## 错误处理

```python
try:
    # 模型注册
    entry = registry.register_model(name, model, model_format)
except ValueError as e:
    print(f"注册失败: {e}")
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
except Exception as e:
    print(f"未知错误: {e}")

try:
    # 模型加载
    model, tokenizer = registry.load_model(name, version)
except ValueError as e:
    print(f"模型未找到: {e}")
except RuntimeError as e:
    print(f"完整性验证失败: {e}")
except ImportError as e:
    print(f"缺少依赖: {e}")
```

## 配置和扩展

### 自定义加载器

```python
from model_registry import ModelLoader, ModelFormat

class CustomLoader(ModelLoader):
    def load(self, model_path: str, **kwargs):
        # 自定义加载逻辑
        pass
    
    def save(self, model, model_path: str, **kwargs):
        # 自定义保存逻辑
        pass
    
    def get_metadata(self, model, model_path: str = None):
        # 自定义元数据提取
        return {"framework": "custom"}
    
    def supported_formats(self):
        return [ModelFormat.CUSTOM]

# 注册自定义加载器
registry.loaders[ModelFormat.CUSTOM] = CustomLoader()
```

### 环境配置

```python
# 环境变量
import os
os.environ['MODEL_REGISTRY_PATH'] = '/path/to/models'
os.environ['MODEL_REGISTRY_CACHE_SIZE'] = '1000'

# 配置文件
registry_config = {
    'default_format': ModelFormat.PYTORCH,
    'auto_validate': True,
    'compression_threshold': 100,  # MB
    'backup_enabled': True
}

registry = ModelRegistry(
    registry_path="./models",
    **registry_config
)
```

## 测试

```bash
# 运行测试
cd apps/api
python -m pytest tests/ai/model_service/ -v

# 运行示例
python src/ai/model_service/examples.py

# 验证API
python -m pytest tests/ai/model_service/test_model_registry.py::TestModelRegistry -v
```

## 监控和日志

```python
import logging

# 设置日志级别
logging.getLogger('model_registry').setLevel(logging.INFO)

# 监控注册表状态
stats = registry.validate_registry()
print(f"健康状态: {stats['valid_models']}/{stats['total_models']} 模型有效")

# 获取详细统计
detailed_stats = registry.get_detailed_stats()  # 需要实现
```

## 故障排除

### 常见问题

1. **ImportError: 缺少依赖**
   ```bash
   pip install torch transformers onnx
   ```

2. **FileNotFoundError: 模型文件不存在**
   ```python
   # 验证注册表
   validation = registry.validate_registry()
   print(validation['errors'])
   ```

3. **RuntimeError: 完整性验证失败**
   ```python
   # 重新计算校验和
   entry.calculate_checksum()
   ```

4. **MemoryError: 模型过大**
   ```python
   # 使用分片加载
   model, _ = registry.load_model(name, max_memory={"0": "8GiB"})
   ```

## 贡献

欢迎提交Issue和Pull Request！

## 许可证

MIT License - 详见LICENSE文件