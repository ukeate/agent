"""模型注册和版本管理系统"""

import asyncio
import aiofiles
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid
import torch
import onnx
from transformers import AutoModel, AutoTokenizer, AutoConfig
from pydantic import BaseModel, Field

logger = get_logger(__name__)

class ModelFormat(str, Enum):
    """支持的模型格式"""
    PYTORCH = "pytorch"
    ONNX = "onnx"
    HUGGINGFACE = "huggingface"

class ModelStatus(str, Enum):
    """模型状态"""
    UPLOADING = "uploading"
    VALIDATING = "validating"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ERROR = "error"

@dataclass
class ModelMetadata:
    """模型元数据"""
    model_id: str
    name: str
    version: str
    format: ModelFormat
    framework: str
    description: Optional[str] = None
    tags: List[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    model_size_mb: Optional[float] = None
    parameter_count: Optional[int] = None
    created_at: datetime = None
    updated_at: datetime = None
    checksum: Optional[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at

class ModelRegistrationRequest(BaseModel):
    """模型注册请求"""
    name: str = Field(..., description="模型名称")
    version: str = Field(..., description="模型版本")
    format: ModelFormat = Field(..., description="模型格式")
    framework: str = Field(..., description="框架名称")
    description: Optional[str] = Field(None, description="模型描述")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="输入Schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="输出Schema")

class PyTorchLoader:
    """PyTorch模型加载器"""
    
    @staticmethod
    def load_model(model_path: str, device: str = "cpu") -> torch.nn.Module:
        """安全加载PyTorch模型"""
        try:
            # 对于模型对象，需要允许特定的类
            with torch.serialization.safe_globals([torch.nn.modules.linear.Linear, 
                                                  torch.nn.modules.conv.Conv2d,
                                                  torch.nn.modules.activation.ReLU,
                                                  torch.nn.modules.batchnorm.BatchNorm2d,
                                                  torch.nn.modules.dropout.Dropout]):
                model = torch.load(model_path, map_location=device, weights_only=True)
            return model
        except Exception as e:
            # 如果weights_only失败，尝试不安全加载但记录警告
            logger.warning(f"安全加载失败，尝试不安全加载: {e}")
            model = torch.load(model_path, map_location=device, weights_only=False)
            return model
    
    @staticmethod
    def extract_metadata(model_path: str) -> Dict[str, Any]:
        """提取PyTorch模型元数据"""
        try:
            # 先尝试获取模型状态字典信息
            try:
                with torch.serialization.safe_globals([torch.nn.modules.linear.Linear,
                                                      torch.nn.modules.conv.Conv2d,
                                                      torch.nn.modules.activation.ReLU,
                                                      torch.nn.modules.batchnorm.BatchNorm2d,
                                                      torch.nn.modules.dropout.Dropout]):
                    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            except Exception:
                # 如果安全加载失败，尝试不安全加载
                state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            
            if isinstance(state_dict, dict):
                # 计算参数数量
                param_count = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            elif hasattr(state_dict, 'parameters'):
                # 如果加载的是模型对象
                param_count = sum(p.numel() for p in state_dict.parameters())
            else:
                param_count = None
            
            # 获取文件大小
            file_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
            
            return {
                "parameter_count": param_count,
                "model_size_mb": round(file_size, 6) if file_size < 0.01 else round(file_size, 2)
            }
        except Exception as e:
            logger.error(f"提取PyTorch模型元数据失败: {e}")
            return {}

class ONNXLoader:
    """ONNX模型加载器"""
    
    @staticmethod
    def load_model(model_path: str) -> onnx.ModelProto:
        """加载ONNX模型"""
        return onnx.load(model_path)
    
    @staticmethod
    def extract_metadata(model_path: str) -> Dict[str, Any]:
        """提取ONNX模型元数据"""
        try:
            model = onnx.load(model_path)
            
            # 获取基本信息
            input_info = []
            output_info = []
            
            for input_tensor in model.graph.input:
                input_info.append({
                    "name": input_tensor.name,
                    "type": input_tensor.type.tensor_type.elem_type,
                    "shape": [d.dim_value for d in input_tensor.type.tensor_type.shape.dim]
                })
            
            for output_tensor in model.graph.output:
                output_info.append({
                    "name": output_tensor.name,
                    "type": output_tensor.type.tensor_type.elem_type,
                    "shape": [d.dim_value for d in output_tensor.type.tensor_type.shape.dim]
                })
            
            # 获取文件大小
            file_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
            
            return {
                "model_size_mb": round(file_size, 6) if file_size < 0.01 else round(file_size, 2),
                "input_schema": input_info,
                "output_schema": output_info,
                "opset_version": model.opset_import[0].version if model.opset_import else None
            }
        except Exception as e:
            logger.error(f"提取ONNX模型元数据失败: {e}")
            return {}

class HuggingFaceLoader:
    """HuggingFace模型加载器"""
    
    @staticmethod
    def load_model_and_tokenizer(model_path: str):
        """加载HuggingFace模型和tokenizer"""
        try:
            config = AutoConfig.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            return model, tokenizer, config
        except Exception as e:
            logger.error(f"加载HuggingFace模型失败: {e}")
            raise
    
    @staticmethod
    def extract_metadata(model_path: str) -> Dict[str, Any]:
        """提取HuggingFace模型元数据"""
        try:
            config = AutoConfig.from_pretrained(model_path)
            
            # 获取配置信息
            metadata = {
                "model_type": getattr(config, 'model_type', None),
                "architecture": getattr(config, 'architectures', None),
                "vocab_size": getattr(config, 'vocab_size', None),
                "hidden_size": getattr(config, 'hidden_size', None),
                "num_layers": getattr(config, 'num_hidden_layers', None) or getattr(config, 'n_layer', None),
                "num_attention_heads": getattr(config, 'num_attention_heads', None) or getattr(config, 'n_head', None)
            }
            
            # 计算参数数量（如果可能）
            if hasattr(config, 'num_parameters'):
                metadata["parameter_count"] = config.num_parameters
            
            return metadata
        except Exception as e:
            logger.error(f"提取HuggingFace模型元数据失败: {e}")
            return {}

class ModelRegistry:
    """模型注册和版本管理系统"""
    
    def __init__(self, storage_path: str = "/tmp/model_registry"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.storage_path / "registry.json"
        self.models: Dict[str, Dict[str, ModelMetadata]] = {}
        self._load_registry()
    
    def _load_registry(self):
        """加载注册表"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for model_name, versions in data.items():
                    self.models[model_name] = {}
                    for version, metadata_dict in versions.items():
                        # 恢复日期时间对象
                        metadata_dict['created_at'] = datetime.fromisoformat(metadata_dict['created_at'])
                        metadata_dict['updated_at'] = datetime.fromisoformat(metadata_dict['updated_at'])
                        self.models[model_name][version] = ModelMetadata(**metadata_dict)
                        
                logger.info(f"加载了 {len(self.models)} 个模型的注册信息")
            except Exception as e:
                logger.error(f"加载注册表失败: {e}")
                self.models = {}
    
    def _save_registry(self):
        """保存注册表"""
        try:
            data = {}
            for model_name, versions in self.models.items():
                data[model_name] = {}
                for version, metadata in versions.items():
                    metadata_dict = asdict(metadata)
                    # 转换日期时间为字符串
                    metadata_dict['created_at'] = metadata.created_at.isoformat()
                    metadata_dict['updated_at'] = metadata.updated_at.isoformat()
                    data[model_name][version] = metadata_dict
            
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info("注册表保存成功")
        except Exception as e:
            logger.error(f"保存注册表失败: {e}")
            raise
    
    def _calculate_checksum(self, file_path: str) -> str:
        """计算文件校验和"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def _get_model_path(self, name: str, version: str) -> Path:
        """获取模型存储路径"""
        return self.storage_path / name / version
    
    async def register_model(
        self, 
        request: ModelRegistrationRequest, 
        model_file_path: str
    ) -> str:
        """注册模型"""
        model_id = str(uuid.uuid4())
        
        try:
            # 创建存储目录
            model_dir = self._get_model_path(request.name, request.version)
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制模型文件
            model_file = Path(model_file_path)
            target_path = model_dir / model_file.name
            shutil.copy2(model_file_path, target_path)
            
            # 计算校验和
            checksum = self._calculate_checksum(str(target_path))
            
            # 提取元数据
            extracted_metadata = self._extract_model_metadata(str(target_path), request.format)
            
            # 创建模型元数据
            metadata = ModelMetadata(
                model_id=model_id,
                name=request.name,
                version=request.version,
                format=request.format,
                framework=request.framework,
                description=request.description,
                tags=request.tags,
                input_schema=request.input_schema,
                output_schema=request.output_schema,
                checksum=checksum,
                **extracted_metadata
            )
            
            # 保存到注册表
            if request.name not in self.models:
                self.models[request.name] = {}
            
            self.models[request.name][request.version] = metadata
            self._save_registry()
            
            logger.info(f"模型注册成功: {request.name}:{request.version} (ID: {model_id})")
            return model_id
            
        except Exception as e:
            logger.error(f"模型注册失败: {e}")
            # 清理失败的文件
            if 'model_dir' in locals() and model_dir.exists():
                shutil.rmtree(model_dir, ignore_errors=True)
            raise
    
    def _extract_model_metadata(self, model_path: str, format: ModelFormat) -> Dict[str, Any]:
        """根据格式提取模型元数据"""
        if format == ModelFormat.PYTORCH:
            return PyTorchLoader.extract_metadata(model_path)
        elif format == ModelFormat.ONNX:
            return ONNXLoader.extract_metadata(model_path)
        elif format == ModelFormat.HUGGINGFACE:
            return HuggingFaceLoader.extract_metadata(model_path)
        else:
            return {}
    
    def get_model(self, name: str, version: str = "latest") -> Optional[ModelMetadata]:
        """获取模型信息"""
        if name not in self.models:
            return None
        
        if version == "latest":
            # 返回最新版本
            versions = list(self.models[name].keys())
            if not versions:
                return None
            # 按创建时间排序，返回最新的
            latest_version = max(versions, key=lambda v: self.models[name][v].created_at)
            return self.models[name][latest_version]
        
        return self.models[name].get(version)
    
    def list_models(self, name_filter: Optional[str] = None, tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """列出模型"""
        result = []
        
        for model_name, versions in self.models.items():
            if name_filter and name_filter not in model_name:
                continue
            
            for version, metadata in versions.items():
                if tags and not any(tag in metadata.tags for tag in tags):
                    continue
                result.append(metadata)
        
        return result
    
    def delete_model(self, name: str, version: str) -> bool:
        """删除模型"""
        if name not in self.models or version not in self.models[name]:
            return False
        
        try:
            # 删除文件
            model_dir = self._get_model_path(name, version)
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # 从注册表删除
            del self.models[name][version]
            if not self.models[name]:  # 如果没有其他版本，删除整个模型
                del self.models[name]
            
            self._save_registry()
            logger.info(f"模型删除成功: {name}:{version}")
            return True
            
        except Exception as e:
            logger.error(f"删除模型失败: {e}")
            return False
    
    def get_model_path(self, name: str, version: str) -> Optional[str]:
        """获取模型文件路径"""
        if name not in self.models or version not in self.models[name]:
            return None
        
        model_dir = self._get_model_path(name, version)
        if not model_dir.exists():
            return None
        
        # 查找模型文件
        model_files = list(model_dir.glob("*"))
        if not model_files:
            return None
        
        return str(model_files[0])  # 返回第一个文件
    
    def validate_model(self, name: str, version: str) -> Dict[str, Any]:
        """验证模型完整性"""
        metadata = self.get_model(name, version)
        if not metadata:
            return {"valid": False, "error": "模型不存在"}
        
        model_path = self.get_model_path(name, version)
        if not model_path:
            return {"valid": False, "error": "模型文件不存在"}
        
        try:
            # 验证校验和
            current_checksum = self._calculate_checksum(model_path)
            if current_checksum != metadata.checksum:
                return {"valid": False, "error": "文件校验和不匹配"}
            
            # 尝试加载模型
            if metadata.format == ModelFormat.PYTORCH:
                PyTorchLoader.load_model(model_path)
            elif metadata.format == ModelFormat.ONNX:
                ONNXLoader.load_model(model_path)
            elif metadata.format == ModelFormat.HUGGINGFACE:
                HuggingFaceLoader.load_model_and_tokenizer(model_path)
            
            return {"valid": True, "message": "模型验证成功"}
            
        except Exception as e:
            return {"valid": False, "error": f"模型验证失败: {str(e)}"}
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取注册表统计信息"""
        total_models = sum(len(versions) for versions in self.models.values())
        
        format_counts = {}
        framework_counts = {}
        total_size_mb = 0
        
        for versions in self.models.values():
            for metadata in versions.values():
                format_counts[metadata.format.value] = format_counts.get(metadata.format.value, 0) + 1
                framework_counts[metadata.framework] = framework_counts.get(metadata.framework, 0) + 1
                if metadata.model_size_mb:
                    total_size_mb += metadata.model_size_mb
        
        return {
            "total_model_families": len(self.models),
            "total_model_versions": total_models,
            "formats": format_counts,
            "frameworks": framework_counts,
            "total_storage_mb": round(total_size_mb, 2)
        }
from src.core.logging import get_logger
