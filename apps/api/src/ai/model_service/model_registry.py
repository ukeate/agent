"""
AI模型注册表系统 - 支持PyTorch、ONNX和HuggingFace模型的统一管理

本模块提供了一个统一的模型注册表，支持多种AI模型格式的加载、保存、元数据管理和版本控制。
基于从Context7文档中获取的最佳实践实现。
"""

from src.core.utils.timezone_utils import utc_now
import os
import json
from src.core.utils import secure_pickle as pickle
import shutil
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Type, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import contextmanager

from src.core.logging import get_logger
logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.jit import ScriptModule
    HAS_PYTORCH = True
except ImportError:
    torch = nn = ScriptModule = None
    HAS_PYTORCH = False

try:
    import onnx
    from onnx import ModelProto, load_model, save_model
    HAS_ONNX = True
except ImportError:
    onnx = ModelProto = load_model = save_model = None
    HAS_ONNX = False

try:
    from transformers import (
        AutoModel, AutoTokenizer, AutoConfig,
        PreTrainedModel, PreTrainedTokenizer,
        AutoModelForCausalLM, AutoModelForSequenceClassification
    )
    HAS_TRANSFORMERS = True
except ImportError:
    (AutoModel, AutoTokenizer, AutoConfig, PreTrainedModel, 
     PreTrainedTokenizer, AutoModelForCausalLM, 
     AutoModelForSequenceClassification) = (None,) * 7
    HAS_TRANSFORMERS = False

class ModelFormat(Enum):
    """支持的模型格式"""
    PYTORCH = "pytorch"
    PYTORCH_SCRIPT = "torchscript"
    ONNX = "onnx"
    HUGGINGFACE = "huggingface"
    SAFETENSORS = "safetensors"

class ModelType(Enum):
    """模型类型"""
    LANGUAGE_MODEL = "language_model"
    CLASSIFICATION = "classification"
    GENERATION = "generation"
    EMBEDDING = "embedding"
    MULTIMODAL = "multimodal"
    CUSTOM = "custom"

class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"
    QUANTIZATION_INT8 = "int8"
    QUANTIZATION_INT4 = "int4"
    PRUNING = "pruning"
    DISTILLATION = "distillation"

@dataclass
class ModelMetadata:
    """模型元数据"""
    name: str
    version: str
    format: ModelFormat
    model_type: ModelType
    description: Optional[str] = None
    author: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    
    # 模型规格
    parameters_count: Optional[int] = None
    model_size_mb: Optional[float] = None
    input_shape: Optional[List[int]] = None
    output_shape: Optional[List[int]] = None
    
    # 训练信息
    training_framework: Optional[str] = None
    training_dataset: Optional[str] = None
    training_epochs: Optional[int] = None
    performance_metrics: Optional[Dict[str, float]] = None
    
    # 压缩信息
    compression_type: CompressionType = CompressionType.NONE
    compression_ratio: Optional[float] = None
    original_size_mb: Optional[float] = None
    
    # 依赖和环境
    dependencies: List[str] = field(default_factory=list)
    python_version: Optional[str] = None
    framework_versions: Optional[Dict[str, str]] = None
    
    # 附加信息
    tags: List[str] = field(default_factory=list)
    license: Optional[str] = None
    repository_url: Optional[str] = None
    paper_url: Optional[str] = None
    
    def update_timestamp(self):
        """更新修改时间"""
        self.updated_at = utc_now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        # 处理枚举类型
        data['format'] = self.format.value
        data['model_type'] = self.model_type.value
        data['compression_type'] = self.compression_type.value
        # 处理日期时间
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建"""
        # 处理枚举类型
        if 'format' in data:
            data['format'] = ModelFormat(data['format'])
        if 'model_type' in data:
            data['model_type'] = ModelType(data['model_type'])
        if 'compression_type' in data:
            data['compression_type'] = CompressionType(data['compression_type'])
        # 处理日期时间
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data and isinstance(data['updated_at'], str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)

@dataclass
class ModelEntry:
    """模型注册条目"""
    metadata: ModelMetadata
    model_path: str
    config_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    checksum: Optional[str] = None
    
    def calculate_checksum(self) -> str:
        """计算模型文件的校验和"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        
        hasher = hashlib.sha256()
        
        # 处理目录
        if os.path.isdir(self.model_path):
            for root, dirs, files in os.walk(self.model_path):
                # 排序确保一致性
                dirs.sort()
                files.sort()
                for file in files:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hasher.update(chunk)
        else:
            # 处理单个文件
            with open(self.model_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
        
        self.checksum = hasher.hexdigest()
        return self.checksum
    
    def verify_integrity(self) -> bool:
        """验证文件完整性"""
        if not self.checksum:
            return True  # 没有校验和时默认通过
        return self.calculate_checksum() == self.checksum

class ModelLoader(ABC):
    """模型加载器抽象基类"""
    
    @abstractmethod
    def load(self, model_path: str, **kwargs) -> Any:
        """加载模型"""
        raise NotImplementedError
    
    @abstractmethod
    def save(self, model: Any, model_path: str, **kwargs) -> None:
        """保存模型"""
        raise NotImplementedError
    
    @abstractmethod
    def get_metadata(self, model: Any, model_path: str = None) -> Dict[str, Any]:
        """提取模型元数据"""
        raise NotImplementedError
    
    @abstractmethod
    def supported_formats(self) -> List[ModelFormat]:
        """返回支持的格式"""
        raise NotImplementedError

class PyTorchLoader(ModelLoader):
    """PyTorch模型加载器"""
    
    def load(self, model_path: str, map_location: str = 'cpu', 
             weights_only: bool = True, **kwargs) -> Any:
        """
        加载PyTorch模型，遵循安全加载最佳实践
        
        Args:
            model_path: 模型文件路径
            map_location: 设备映射位置
            weights_only: 是否仅加载权重（安全选项）
            **kwargs: 其他参数
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch未安装")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            # 检查是否为TorchScript模型
            if model_path.endswith('.pt') and self._is_torchscript_model(model_path):
                logger.info("检测到TorchScript模型，使用torch.jit.load加载")
                return torch.jit.load(model_path, map_location=map_location)
            
            # 标准PyTorch模型加载
            logger.info("加载PyTorch模型", weights_only=weights_only)
            return torch.load(model_path, map_location=map_location, 
                            weights_only=weights_only)
            
        except Exception as e:
            logger.error("PyTorch模型加载失败", error=str(e), exc_info=True)
            raise
    
    def save(self, model: Any, model_path: str, 
             safe_serialization: bool = True, **kwargs) -> None:
        """
        保存PyTorch模型
        
        Args:
            model: 要保存的模型
            model_path: 保存路径
            safe_serialization: 是否使用安全序列化
            **kwargs: 其他参数
        """
        if not HAS_PYTORCH:
            raise ImportError("PyTorch未安装")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        try:
            # 处理TorchScript模型
            if isinstance(model, ScriptModule):
                logger.info("保存TorchScript模型")
                torch.jit.save(model, model_path)
                return
            
            # 处理nn.Module
            if isinstance(model, nn.Module):
                logger.info("保存PyTorch模型状态字典")
                torch.save(model.state_dict(), model_path)
                return
            
            # 保存其他PyTorch对象
            logger.info("保存PyTorch对象")
            torch.save(model, model_path)
            
        except Exception as e:
            logger.error("PyTorch模型保存失败", error=str(e), exc_info=True)
            raise
    
    def get_metadata(self, model: Any, model_path: str = None) -> Dict[str, Any]:
        """提取PyTorch模型元数据"""
        metadata = {
            "framework": "pytorch",
            "framework_version": torch.__version__ if HAS_PYTORCH else None
        }
        
        if isinstance(model, nn.Module):
            # 计算参数数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            metadata.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params,
                "model_class": model.__class__.__name__,
                "is_training": model.training
            })
            
            # 提取输入/输出形状信息（如果可获得）
            if hasattr(model, 'input_shape'):
                metadata["input_shape"] = model.input_shape
            if hasattr(model, 'output_shape'):
                metadata["output_shape"] = model.output_shape
        
        # 文件大小信息
        if model_path and os.path.exists(model_path):
            if os.path.isfile(model_path):
                size_bytes = os.path.getsize(model_path)
            else:
                size_bytes = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                )
            metadata["model_size_mb"] = size_bytes / (1024 * 1024)
        
        return metadata
    
    def _is_torchscript_model(self, model_path: str) -> bool:
        """检查是否为TorchScript模型"""
        try:
            # 尝试用torch.jit.load加载（不会实际加载）
            torch.jit.load(model_path, map_location='cpu')
            return True
        except:
            return False
    
    def supported_formats(self) -> List[ModelFormat]:
        return [ModelFormat.PYTORCH, ModelFormat.PYTORCH_SCRIPT]

class ONNXLoader(ModelLoader):
    """ONNX模型加载器"""
    
    def load(self, model_path: str, **kwargs) -> Any:
        """加载ONNX模型"""
        if not HAS_ONNX:
            raise ImportError("ONNX未安装")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        try:
            logger.info("加载ONNX模型", model_path=str(model_path))
            
            # 检查外部数据
            if self._has_external_data(model_path):
                logger.info("检测到外部数据，使用适当的加载方法")
                # 加载时不自动加载外部数据
                model = onnx.load(model_path, load_external_data=False)
                # 手动加载外部数据（如果需要）
                from onnx.external_data_helper import load_external_data_for_model
                load_external_data_for_model(model, os.path.dirname(model_path))
                return model
            else:
                return onnx.load(model_path)
                
        except Exception as e:
            logger.error("ONNX模型加载失败", error=str(e), exc_info=True)
            raise
    
    def save(self, model: Any, model_path: str, 
             save_as_external_data: bool = False, **kwargs) -> None:
        """保存ONNX模型"""
        if not HAS_ONNX:
            raise ImportError("ONNX未安装")
        
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        try:
            if save_as_external_data:
                logger.info("保存ONNX模型（带外部数据）")
                onnx.save_model(
                    model, model_path, 
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location=f"{os.path.basename(model_path)}.data",
                    size_threshold=1024
                )
            else:
                logger.info("保存ONNX模型")
                onnx.save(model, model_path)
                
        except Exception as e:
            logger.error("ONNX模型保存失败", error=str(e), exc_info=True)
            raise
    
    def get_metadata(self, model: Any, model_path: str = None) -> Dict[str, Any]:
        """提取ONNX模型元数据"""
        if not isinstance(model, onnx.ModelProto):
            raise ValueError("输入必须是ONNX ModelProto对象")
        
        metadata = {
            "framework": "onnx",
            "ir_version": model.ir_version,
            "producer_name": model.producer_name,
            "producer_version": model.producer_version,
            "domain": model.domain,
            "model_version": model.model_version,
            "doc_string": model.doc_string
        }
        
        # 操作集信息
        opset_imports = []
        for opset in model.opset_import:
            opset_imports.append({
                "domain": opset.domain,
                "version": opset.version
            })
        metadata["opset_imports"] = opset_imports
        
        # 输入/输出信息
        if model.graph:
            inputs = []
            for input_tensor in model.graph.input:
                input_info = {"name": input_tensor.name}
                if input_tensor.type.HasField('tensor_type'):
                    tensor_type = input_tensor.type.tensor_type
                    input_info["data_type"] = tensor_type.elem_type
                    if tensor_type.HasField('shape'):
                        shape = []
                        for dim in tensor_type.shape.dim:
                            if dim.HasField('dim_value'):
                                shape.append(dim.dim_value)
                            else:
                                shape.append(-1)  # 动态维度
                        input_info["shape"] = shape
                inputs.append(input_info)
            metadata["inputs"] = inputs
            
            outputs = []
            for output_tensor in model.graph.output:
                output_info = {"name": output_tensor.name}
                if output_tensor.type.HasField('tensor_type'):
                    tensor_type = output_tensor.type.tensor_type
                    output_info["data_type"] = tensor_type.elem_type
                    if tensor_type.HasField('shape'):
                        shape = []
                        for dim in tensor_type.shape.dim:
                            if dim.HasField('dim_value'):
                                shape.append(dim.dim_value)
                            else:
                                shape.append(-1)
                        output_info["shape"] = shape
                outputs.append(output_info)
            metadata["outputs"] = outputs
        
        # 模型大小
        if model_path and os.path.exists(model_path):
            size_bytes = os.path.getsize(model_path)
            metadata["model_size_mb"] = size_bytes / (1024 * 1024)
        
        # 元数据属性
        if model.metadata_props:
            custom_metadata = {}
            for prop in model.metadata_props:
                custom_metadata[prop.key] = prop.value
            metadata["custom_metadata"] = custom_metadata
        
        return metadata
    
    def _has_external_data(self, model_path: str) -> bool:
        """检查模型是否使用外部数据"""
        try:
            # 快速检查，不加载实际数据
            with open(model_path, 'rb') as f:
                # 读取前几KB来检查是否包含外部数据引用
                header = f.read(8192)
                return b'external_data' in header
        except:
            return False
    
    def supported_formats(self) -> List[ModelFormat]:
        return [ModelFormat.ONNX]

class HuggingFaceLoader(ModelLoader):
    """HuggingFace模型加载器"""
    
    def load(self, model_path: str, model_type: str = "auto", 
             torch_dtype: str = "auto", device_map: str = "auto", **kwargs) -> Tuple[Any, Any]:
        """
        加载HuggingFace模型和tokenizer
        
        Args:
            model_path: 模型路径或Hub标识符
            model_type: 模型类型 ("auto", "causal_lm", "sequence_classification", etc.)
            torch_dtype: 数据类型
            device_map: 设备映射
            **kwargs: 其他参数
            
        Returns:
            (model, tokenizer) 元组
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers未安装")
        
        try:
            # 选择模型类
            if model_type == "auto":
                model_class = AutoModel
            elif model_type == "causal_lm":
                model_class = AutoModelForCausalLM
            elif model_type == "sequence_classification":
                model_class = AutoModelForSequenceClassification
            else:
                model_class = AutoModel
            
            logger.info("加载HuggingFace模型", model_path=str(model_path))
            
            # 加载模型
            model = model_class.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                **kwargs
            )
            
            # 加载tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception as e:
                logger.warning("无法加载tokenizer", error=str(e))
                tokenizer = None
            
            return model, tokenizer
            
        except Exception as e:
            logger.error("HuggingFace模型加载失败", error=str(e), exc_info=True)
            raise
    
    def save(self, model: Any, model_path: str, 
             tokenizer: Any = None, safe_serialization: bool = True,
             max_shard_size: str = "5GB", **kwargs) -> None:
        """
        保存HuggingFace模型和tokenizer
        
        Args:
            model: 要保存的模型
            model_path: 保存路径
            tokenizer: tokenizer对象
            safe_serialization: 是否使用安全序列化
            max_shard_size: 最大分片大小
            **kwargs: 其他参数
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers未安装")
        
        os.makedirs(model_path, exist_ok=True)
        
        try:
            # 保存模型
            if isinstance(model, PreTrainedModel):
                logger.info("保存HuggingFace模型")
                model.save_pretrained(
                    model_path,
                    safe_serialization=safe_serialization,
                    max_shard_size=max_shard_size
                )
            else:
                raise ValueError("模型必须是PreTrainedModel实例")
            
            # 保存tokenizer
            if tokenizer and isinstance(tokenizer, PreTrainedTokenizer):
                logger.info("保存HuggingFace tokenizer")
                tokenizer.save_pretrained(model_path)
            
        except Exception as e:
            logger.error("HuggingFace模型保存失败", error=str(e), exc_info=True)
            raise
    
    def get_metadata(self, model: Any, model_path: str = None) -> Dict[str, Any]:
        """提取HuggingFace模型元数据"""
        metadata = {
            "framework": "huggingface_transformers"
        }
        
        if hasattr(model, 'config'):
            config = model.config
            metadata.update({
                "model_type": config.model_type if hasattr(config, 'model_type') else None,
                "architectures": config.architectures if hasattr(config, 'architectures') else None,
                "vocab_size": config.vocab_size if hasattr(config, 'vocab_size') else None,
                "hidden_size": config.hidden_size if hasattr(config, 'hidden_size') else None,
                "num_hidden_layers": config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else None,
                "num_attention_heads": config.num_attention_heads if hasattr(config, 'num_attention_heads') else None,
            })
            
            # 转换配置为字典
            try:
                metadata["config"] = config.to_dict()
            except Exception:
                logger.exception("模型配置转换失败", exc_info=True)
        
        # 参数统计
        if hasattr(model, 'parameters'):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            metadata.update({
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            })
        
        # 模型大小
        if model_path and os.path.exists(model_path):
            if os.path.isdir(model_path):
                size_bytes = sum(
                    os.path.getsize(os.path.join(dirpath, filename))
                    for dirpath, dirnames, filenames in os.walk(model_path)
                    for filename in filenames
                )
            else:
                size_bytes = os.path.getsize(model_path)
            metadata["model_size_mb"] = size_bytes / (1024 * 1024)
        
        return metadata
    
    def supported_formats(self) -> List[ModelFormat]:
        return [ModelFormat.HUGGINGFACE, ModelFormat.SAFETENSORS]

class ModelRegistry:
    """
    统一的AI模型注册表
    
    支持PyTorch、ONNX和HuggingFace模型的注册、加载、保存和元数据管理。
    """
    
    def __init__(self, registry_path: str = "./models"):
        """
        初始化模型注册表
        
        Args:
            registry_path: 注册表根目录
        """
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        # 注册表文件
        self.registry_file = self.registry_path / "registry.json"
        self.models: Dict[str, ModelEntry] = {}
        
        # 初始化加载器
        self.loaders: Dict[ModelFormat, ModelLoader] = {}
        if HAS_PYTORCH:
            self.loaders[ModelFormat.PYTORCH] = PyTorchLoader()
            self.loaders[ModelFormat.PYTORCH_SCRIPT] = PyTorchLoader()
        if HAS_ONNX:
            self.loaders[ModelFormat.ONNX] = ONNXLoader()
        if HAS_TRANSFORMERS:
            self.loaders[ModelFormat.HUGGINGFACE] = HuggingFaceLoader()
            self.loaders[ModelFormat.SAFETENSORS] = HuggingFaceLoader()
        
        # 加载现有注册表
        self._load_registry()
    
    def register_model(
        self,
        name: str,
        model: Any,
        model_format: ModelFormat,
        model_type: ModelType = ModelType.CUSTOM,
        version: str = "1.0.0",
        description: str = None,
        tokenizer: Any = None,
        overwrite: bool = False,
        **metadata_kwargs
    ) -> ModelEntry:
        """
        注册新模型
        
        Args:
            name: 模型名称
            model: 模型对象
            model_format: 模型格式
            model_type: 模型类型
            version: 版本号
            description: 描述
            tokenizer: tokenizer对象（HuggingFace模型）
            overwrite: 是否覆盖现有模型
            **metadata_kwargs: 其他元数据参数
            
        Returns:
            ModelEntry: 模型条目
        """
        model_id = f"{name}:{version}"
        
        # 检查是否已存在
        if model_id in self.models and not overwrite:
            raise ValueError(f"模型已存在: {model_id}，使用overwrite=True来覆盖")
        
        # 创建模型目录
        model_dir = self.registry_path / name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 获取加载器
        if model_format not in self.loaders:
            raise ValueError(f"不支持的模型格式: {model_format}")
        
        loader = self.loaders[model_format]
        
        # 保存模型
        if model_format == ModelFormat.HUGGINGFACE:
            model_path = str(model_dir)
            loader.save(model, model_path, tokenizer=tokenizer)
        else:
            model_filename = self._get_model_filename(model_format)
            model_path = str(model_dir / model_filename)
            loader.save(model, model_path)
        
        # 提取元数据
        extracted_metadata = loader.get_metadata(model, model_path)
        
        # 创建元数据
        metadata = ModelMetadata(
            name=name,
            version=version,
            format=model_format,
            model_type=model_type,
            description=description,
            **metadata_kwargs
        )
        
        # 更新元数据
        if extracted_metadata:
            if 'total_parameters' in extracted_metadata:
                metadata.parameters_count = extracted_metadata['total_parameters']
            if 'model_size_mb' in extracted_metadata:
                metadata.model_size_mb = extracted_metadata['model_size_mb']
            if 'framework_version' in extracted_metadata:
                metadata.framework_versions = {
                    extracted_metadata.get('framework', 'unknown'): 
                    extracted_metadata['framework_version']
                }
        
        # 创建模型条目
        entry = ModelEntry(
            metadata=metadata,
            model_path=model_path,
            tokenizer_path=str(model_dir) if tokenizer else None
        )
        
        # 计算校验和
        entry.calculate_checksum()
        
        # 保存元数据
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 注册模型
        self.models[model_id] = entry
        self._save_registry()
        
        logger.info("模型注册成功", model_id=model_id)
        return entry
    
    def load_model(
        self, 
        name: str, 
        version: str = "latest",
        device: str = "auto",
        **kwargs
    ) -> Tuple[Any, Optional[Any]]:
        """
        加载已注册的模型
        
        Args:
            name: 模型名称
            version: 版本号，"latest"表示最新版本
            device: 设备
            **kwargs: 其他加载参数
            
        Returns:
            (model, tokenizer) 元组，tokenizer可能为None
        """
        if version == "latest":
            version = self._get_latest_version(name)
        
        model_id = f"{name}:{version}"
        if model_id not in self.models:
            raise ValueError(f"模型未注册: {model_id}")
        
        entry = self.models[model_id]
        
        # 验证文件完整性
        if not entry.verify_integrity():
            raise RuntimeError(f"模型文件完整性验证失败: {model_id}")
        
        # 获取加载器
        loader = self.loaders[entry.metadata.format]
        
        # 加载模型
        if entry.metadata.format == ModelFormat.HUGGINGFACE:
            model, tokenizer = loader.load(entry.model_path, device_map=device, **kwargs)
            return model, tokenizer
        else:
            # 处理设备映射
            load_kwargs = kwargs.copy()
            if entry.metadata.format in [ModelFormat.PYTORCH, ModelFormat.PYTORCH_SCRIPT]:
                load_kwargs['map_location'] = device
            
            model = loader.load(entry.model_path, **load_kwargs)
            return model, None
    
    def list_models(self, model_type: ModelType = None) -> List[ModelEntry]:
        """
        列出已注册的模型
        
        Args:
            model_type: 筛选模型类型
            
        Returns:
            模型条目列表
        """
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.metadata.model_type == model_type]
        
        # 按更新时间排序
        models.sort(key=lambda x: x.metadata.updated_at, reverse=True)
        return models
    
    def get_model_info(self, name: str, version: str = "latest") -> Optional[ModelEntry]:
        """获取模型信息"""
        if version == "latest":
            try:
                version = self._get_latest_version(name)
            except ValueError:
                return None
        
        model_id = f"{name}:{version}"
        return self.models.get(model_id)
    
    def remove_model(self, name: str, version: str = None) -> bool:
        """
        移除模型
        
        Args:
            name: 模型名称
            version: 版本号，None表示移除所有版本
            
        Returns:
            是否成功移除
        """
        if version:
            model_id = f"{name}:{version}"
            if model_id not in self.models:
                return False
            
            entry = self.models[model_id]
            
            # 删除文件
            model_dir = Path(entry.model_path).parent
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # 从注册表中移除
            del self.models[model_id]
            
        else:
            # 移除所有版本
            model_ids = [mid for mid in self.models.keys() if mid.startswith(f"{name}:")]
            for model_id in model_ids:
                entry = self.models[model_id]
                model_dir = Path(entry.model_path).parent.parent  # name/version -> name
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                del self.models[model_id]
        
        self._save_registry()
        return True
    
    def export_model(
        self, 
        name: str, 
        version: str, 
        export_path: str,
        export_format: ModelFormat = None
    ) -> str:
        """
        导出模型到指定路径和格式
        
        Args:
            name: 模型名称
            version: 版本号
            export_path: 导出路径
            export_format: 导出格式，None表示保持原格式
            
        Returns:
            导出的文件路径
        """
        model_id = f"{name}:{version}"
        if model_id not in self.models:
            raise ValueError(f"模型未注册: {model_id}")
        
        entry = self.models[model_id]
        
        if export_format is None or export_format == entry.metadata.format:
            # 直接复制
            if os.path.isdir(entry.model_path):
                shutil.copytree(entry.model_path, export_path)
            else:
                shutil.copy2(entry.model_path, export_path)
            return export_path
        
        # 格式转换（需要实现具体的转换逻辑）
        raise NotImplementedError("格式转换功能尚未实现")
    
    def validate_registry(self) -> Dict[str, List[str]]:
        """
        验证注册表完整性
        
        Returns:
            验证结果字典，包含错误和警告
        """
        errors = []
        warnings = []
        
        for model_id, entry in self.models.items():
            # 检查文件存在性
            if not os.path.exists(entry.model_path):
                errors.append(f"模型文件不存在: {model_id} -> {entry.model_path}")
                continue
            
            # 检查完整性
            if not entry.verify_integrity():
                errors.append(f"模型文件完整性验证失败: {model_id}")
            
            # 检查格式支持
            if entry.metadata.format not in self.loaders:
                warnings.append(f"模型格式不支持: {model_id} -> {entry.metadata.format}")
        
        return {
            "errors": errors,
            "warnings": warnings,
            "total_models": len(self.models),
            "valid_models": len(self.models) - len(errors)
        }
    
    @contextmanager
    def temporary_model(self, name: str, version: str = "latest"):
        """临时加载模型的上下文管理器"""
        model, tokenizer = self.load_model(name, version)
        try:
            yield model, tokenizer
        finally:
            # 清理资源
            if hasattr(model, 'to'):
                model.to('cpu')  # 移到CPU释放GPU内存
            del model
            if tokenizer:
                del tokenizer
    
    def _get_model_filename(self, model_format: ModelFormat) -> str:
        """获取模型文件名"""
        filename_map = {
            ModelFormat.PYTORCH: "model.pth",
            ModelFormat.PYTORCH_SCRIPT: "model.pt",
            ModelFormat.ONNX: "model.onnx",
        }
        return filename_map.get(model_format, "model.bin")
    
    def _get_latest_version(self, name: str) -> str:
        """获取模型的最新版本"""
        versions = [
            entry.metadata.version 
            for model_id, entry in self.models.items()
            if model_id.startswith(f"{name}:")
        ]
        
        if not versions:
            raise ValueError(f"模型不存在: {name}")
        
        # 简单的版本排序，实际项目中可能需要更复杂的版本比较
        versions.sort(reverse=True)
        return versions[0]
    
    def _load_registry(self):
        """从文件加载注册表"""
        if not self.registry_file.exists():
            return
        
        try:
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                registry_data = json.load(f)
            
            for model_id, entry_data in registry_data.items():
                metadata = ModelMetadata.from_dict(entry_data['metadata'])
                entry = ModelEntry(
                    metadata=metadata,
                    model_path=entry_data['model_path'],
                    config_path=entry_data.get('config_path'),
                    tokenizer_path=entry_data.get('tokenizer_path'),
                    checksum=entry_data.get('checksum')
                )
                self.models[model_id] = entry
                
            logger.info("已加载模型注册信息", model_count=len(self.models))
            
        except Exception as e:
            logger.error("加载注册表失败", error=str(e), exc_info=True)
    
    def _save_registry(self):
        """保存注册表到文件"""
        try:
            registry_data = {}
            for model_id, entry in self.models.items():
                registry_data[model_id] = {
                    'metadata': entry.metadata.to_dict(),
                    'model_path': entry.model_path,
                    'config_path': entry.config_path,
                    'tokenizer_path': entry.tokenizer_path,
                    'checksum': entry.checksum
                }
            
            with open(self.registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error("保存注册表失败", error=str(e), exc_info=True)

# 全局模型注册表实例
model_registry = ModelRegistry()

def register_pytorch_model(
    name: str,
    model: Union[nn.Module, Any],
    version: str = "1.0.0",
    **kwargs
) -> ModelEntry:
    """注册PyTorch模型的便捷函数"""
    format_type = ModelFormat.PYTORCH_SCRIPT if isinstance(model, ScriptModule) else ModelFormat.PYTORCH
    return model_registry.register_model(
        name=name,
        model=model,
        model_format=format_type,
        version=version,
        **kwargs
    )

def register_onnx_model(
    name: str,
    model: Any,
    version: str = "1.0.0",
    **kwargs
) -> ModelEntry:
    """注册ONNX模型的便捷函数"""
    return model_registry.register_model(
        name=name,
        model=model,
        model_format=ModelFormat.ONNX,
        version=version,
        **kwargs
    )

def register_huggingface_model(
    name: str,
    model: Any,
    tokenizer: Any = None,
    version: str = "1.0.0",
    **kwargs
) -> ModelEntry:
    """注册HuggingFace模型的便捷函数"""
    return model_registry.register_model(
        name=name,
        model=model,
        model_format=ModelFormat.HUGGINGFACE,
        tokenizer=tokenizer,
        version=version,
        **kwargs
    )

if __name__ == "__main__":
    # 使用示例
    logger.info("AI模型注册表系统测试开始")
    
    # 创建测试注册表
    test_registry = ModelRegistry("./test_models")
    
    # 验证注册表
    validation_result = test_registry.validate_registry()
    logger.info("注册表验证结果", result=validation_result)
    
    # 列出所有模型
    models = test_registry.list_models()
    logger.info("已注册模型数量", model_count=len(models))
    for model in models:
        logger.info(
            "已注册模型",
            name=model.metadata.name,
            version=model.metadata.version,
            model_format=model.metadata.format.value,
        )
    
    logger.info("AI模型注册表系统测试完成")
