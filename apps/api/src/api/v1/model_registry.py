"""
模型注册表API端点

提供RESTful API接口用于管理AI模型注册表
"""

import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from datetime import datetime

from src.ai.model_service.model_registry import (
    ModelRegistry, ModelMetadata, ModelEntry, ModelFormat, ModelType, CompressionType,
    model_registry
)

logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/model-registry", tags=["Model Registry"])


# Pydantic模型定义
class ModelMetadataRequest(BaseModel):
    """模型元数据请求"""
    name: str = Field(..., description="模型名称")
    version: str = Field(default="1.0.0", description="模型版本")
    format: ModelFormat = Field(..., description="模型格式")
    model_type: ModelType = Field(default=ModelType.CUSTOM, description="模型类型")
    description: Optional[str] = Field(None, description="模型描述")
    author: Optional[str] = Field(None, description="作者")
    
    # 模型规格
    parameters_count: Optional[int] = Field(None, description="参数数量")
    model_size_mb: Optional[float] = Field(None, description="模型大小(MB)")
    input_shape: Optional[List[int]] = Field(None, description="输入形状")
    output_shape: Optional[List[int]] = Field(None, description="输出形状")
    
    # 训练信息
    training_framework: Optional[str] = Field(None, description="训练框架")
    training_dataset: Optional[str] = Field(None, description="训练数据集")
    training_epochs: Optional[int] = Field(None, description="训练轮数")
    performance_metrics: Optional[Dict[str, float]] = Field(None, description="性能指标")
    
    # 压缩信息
    compression_type: CompressionType = Field(default=CompressionType.NONE, description="压缩类型")
    compression_ratio: Optional[float] = Field(None, description="压缩比例")
    original_size_mb: Optional[float] = Field(None, description="原始大小(MB)")
    
    # 其他信息
    tags: List[str] = Field(default_factory=list, description="标签")
    license: Optional[str] = Field(None, description="许可证")
    repository_url: Optional[str] = Field(None, description="仓库URL")
    paper_url: Optional[str] = Field(None, description="论文URL")


class ModelMetadataResponse(BaseModel):
    """模型元数据响应"""
    name: str
    version: str
    format: str
    model_type: str
    description: Optional[str]
    author: Optional[str]
    created_at: datetime
    updated_at: datetime
    parameters_count: Optional[int]
    model_size_mb: Optional[float]
    input_shape: Optional[List[int]]
    output_shape: Optional[List[int]]
    training_framework: Optional[str]
    training_dataset: Optional[str]
    training_epochs: Optional[int]
    performance_metrics: Optional[Dict[str, float]]
    compression_type: str
    compression_ratio: Optional[float]
    original_size_mb: Optional[float]
    dependencies: List[str]
    python_version: Optional[str]
    framework_versions: Optional[Dict[str, str]]
    tags: List[str]
    license: Optional[str]
    repository_url: Optional[str]
    paper_url: Optional[str]


class ModelEntryResponse(BaseModel):
    """模型条目响应"""
    metadata: ModelMetadataResponse
    model_path: str
    config_path: Optional[str]
    tokenizer_path: Optional[str]
    checksum: Optional[str]


class ModelListResponse(BaseModel):
    """模型列表响应"""
    models: List[ModelEntryResponse]
    total_count: int


class ValidationResponse(BaseModel):
    """验证结果响应"""
    errors: List[str]
    warnings: List[str]
    total_models: int
    valid_models: int


class RegisterModelResponse(BaseModel):
    """注册模型响应"""
    success: bool
    message: str
    model_id: str
    entry: ModelEntryResponse


def get_model_registry() -> ModelRegistry:
    """获取模型注册表实例"""
    return model_registry


def convert_metadata_to_response(metadata: ModelMetadata) -> ModelMetadataResponse:
    """转换元数据为响应格式"""
    return ModelMetadataResponse(
        name=metadata.name,
        version=metadata.version,
        format=metadata.format.value,
        model_type=metadata.model_type.value,
        description=metadata.description,
        author=metadata.author,
        created_at=metadata.created_at,
        updated_at=metadata.updated_at,
        parameters_count=metadata.parameters_count,
        model_size_mb=metadata.model_size_mb,
        input_shape=metadata.input_shape,
        output_shape=metadata.output_shape,
        training_framework=metadata.training_framework,
        training_dataset=metadata.training_dataset,
        training_epochs=metadata.training_epochs,
        performance_metrics=metadata.performance_metrics,
        compression_type=metadata.compression_type.value,
        compression_ratio=metadata.compression_ratio,
        original_size_mb=metadata.original_size_mb,
        dependencies=metadata.dependencies,
        python_version=metadata.python_version,
        framework_versions=metadata.framework_versions,
        tags=metadata.tags,
        license=metadata.license,
        repository_url=metadata.repository_url,
        paper_url=metadata.paper_url
    )


def convert_entry_to_response(entry: ModelEntry) -> ModelEntryResponse:
    """转换模型条目为响应格式"""
    return ModelEntryResponse(
        metadata=convert_metadata_to_response(entry.metadata),
        model_path=entry.model_path,
        config_path=entry.config_path,
        tokenizer_path=entry.tokenizer_path,
        checksum=entry.checksum
    )


@router.get("/models", response_model=ModelListResponse)
async def list_models(
    model_type: Optional[ModelType] = None,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    列出所有注册的模型
    
    Args:
        model_type: 可选的模型类型筛选
        registry: 模型注册表实例
        
    Returns:
        模型列表
    """
    try:
        models = registry.list_models(model_type)
        model_responses = [convert_entry_to_response(model) for model in models]
        
        return ModelListResponse(
            models=model_responses,
            total_count=len(model_responses)
        )
    except Exception as e:
        logger.error(f"列出模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"列出模型失败: {str(e)}")


@router.get("/models/{name}", response_model=ModelEntryResponse)
async def get_model_info(
    name: str,
    version: str = "latest",
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    获取特定模型的信息
    
    Args:
        name: 模型名称
        version: 模型版本，默认为"latest"
        registry: 模型注册表实例
        
    Returns:
        模型详细信息
    """
    try:
        model_entry = registry.get_model_info(name, version)
        if not model_entry:
            raise HTTPException(status_code=404, detail=f"模型未找到: {name}:{version}")
        
        return convert_entry_to_response(model_entry)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")


@router.post("/models/upload", response_model=RegisterModelResponse)
async def upload_and_register_model(
    background_tasks: BackgroundTasks,
    model_file: UploadFile = File(..., description="模型文件"),
    metadata: str = Field(..., description="模型元数据JSON字符串"),
    tokenizer_file: Optional[UploadFile] = File(None, description="可选的tokenizer文件"),
    config_file: Optional[UploadFile] = File(None, description="可选的配置文件"),
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    上传并注册模型文件
    
    Args:
        background_tasks: 后台任务
        model_file: 模型文件
        metadata: 模型元数据JSON字符串
        tokenizer_file: 可选的tokenizer文件
        config_file: 可选的配置文件
        registry: 模型注册表实例
        
    Returns:
        注册结果
    """
    import json
    import tempfile
    
    try:
        # 解析元数据
        try:
            metadata_dict = json.loads(metadata)
            metadata_request = ModelMetadataRequest(**metadata_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"元数据格式错误: {str(e)}")
        
        # 检查文件格式
        model_filename = model_file.filename
        if not model_filename:
            raise HTTPException(status_code=400, detail="模型文件名不能为空")
        
        # 根据文件扩展名确定格式
        if model_filename.endswith(('.pth', '.pt')):
            detected_format = ModelFormat.PYTORCH
        elif model_filename.endswith('.onnx'):
            detected_format = ModelFormat.ONNX
        else:
            detected_format = metadata_request.format
        
        # 创建临时目录保存上传的文件
        temp_dir = tempfile.mkdtemp()
        temp_model_path = os.path.join(temp_dir, model_filename)
        
        # 保存模型文件
        with open(temp_model_path, 'wb') as f:
            content = await model_file.read()
            f.write(content)
        
        # 处理tokenizer文件
        temp_tokenizer_path = None
        if tokenizer_file:
            temp_tokenizer_path = os.path.join(temp_dir, tokenizer_file.filename)
            with open(temp_tokenizer_path, 'wb') as f:
                tokenizer_content = await tokenizer_file.read()
                f.write(tokenizer_content)
        
        # 处理配置文件
        temp_config_path = None
        if config_file:
            temp_config_path = os.path.join(temp_dir, config_file.filename)
            with open(temp_config_path, 'wb') as f:
                config_content = await config_file.read()
                f.write(config_content)
        
        # 加载模型进行注册
        # 注意：这里只是示例，实际实现需要根据格式加载模型
        if detected_format == ModelFormat.PYTORCH:
            import torch
            try:
                model = torch.load(temp_model_path, map_location='cpu', weights_only=True)
            except:
                # 如果weights_only失败，尝试常规加载
                model = torch.load(temp_model_path, map_location='cpu')
        elif detected_format == ModelFormat.ONNX:
            import onnx
            model = onnx.load(temp_model_path)
        else:
            # 对于其他格式，使用占位符
            model = {"file_path": temp_model_path}
        
        # 注册模型
        entry = registry.register_model(
            name=metadata_request.name,
            model=model,
            model_format=detected_format,
            model_type=metadata_request.model_type,
            version=metadata_request.version,
            description=metadata_request.description,
            author=metadata_request.author,
            training_framework=metadata_request.training_framework,
            training_dataset=metadata_request.training_dataset,
            training_epochs=metadata_request.training_epochs,
            performance_metrics=metadata_request.performance_metrics,
            compression_type=metadata_request.compression_type,
            compression_ratio=metadata_request.compression_ratio,
            original_size_mb=metadata_request.original_size_mb,
            tags=metadata_request.tags,
            license=metadata_request.license,
            repository_url=metadata_request.repository_url,
            paper_url=metadata_request.paper_url
        )
        
        # 清理临时文件
        background_tasks.add_task(cleanup_temp_dir, temp_dir)
        
        model_id = f"{metadata_request.name}:{metadata_request.version}"
        
        return RegisterModelResponse(
            success=True,
            message=f"模型注册成功: {model_id}",
            model_id=model_id,
            entry=convert_entry_to_response(entry)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"上传注册模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"上传注册模型失败: {str(e)}")


@router.post("/models/{name}/register-from-hub")
async def register_from_hub(
    name: str,
    metadata: ModelMetadataRequest,
    hub_model_id: str = Field(..., description="Hub模型ID"),
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    从Hugging Face Hub注册模型
    
    Args:
        name: 本地模型名称
        metadata: 模型元数据
        hub_model_id: Hub模型标识符
        registry: 模型注册表实例
        
    Returns:
        注册结果
    """
    try:
        if metadata.format != ModelFormat.HUGGINGFACE:
            raise HTTPException(status_code=400, detail="只支持HuggingFace格式的Hub模型")
        
        # 从Hub加载模型
        from transformers import AutoModel, AutoTokenizer
        
        try:
            model = AutoModel.from_pretrained(hub_model_id)
            tokenizer = AutoTokenizer.from_pretrained(hub_model_id)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"从Hub加载模型失败: {str(e)}")
        
        # 注册模型
        entry = registry.register_model(
            name=name,
            model=model,
            tokenizer=tokenizer,
            model_format=ModelFormat.HUGGINGFACE,
            model_type=metadata.model_type,
            version=metadata.version,
            description=metadata.description or f"从Hub加载的模型: {hub_model_id}",
            author=metadata.author,
            repository_url=f"https://huggingface.co/{hub_model_id}",
            tags=metadata.tags + ["huggingface", "hub"],
            **metadata.dict(exclude={"name", "version", "format", "description", "author", "tags"})
        )
        
        model_id = f"{name}:{metadata.version}"
        
        return RegisterModelResponse(
            success=True,
            message=f"从Hub注册模型成功: {model_id}",
            model_id=model_id,
            entry=convert_entry_to_response(entry)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"从Hub注册模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"从Hub注册模型失败: {str(e)}")


@router.delete("/models/{name}")
async def remove_model(
    name: str,
    version: Optional[str] = None,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    移除模型
    
    Args:
        name: 模型名称
        version: 模型版本，None表示移除所有版本
        registry: 模型注册表实例
        
    Returns:
        删除结果
    """
    try:
        success = registry.remove_model(name, version)
        
        if not success:
            model_id = f"{name}:{version}" if version else name
            raise HTTPException(status_code=404, detail=f"模型未找到: {model_id}")
        
        message = f"成功删除模型: {name}" + (f":{version}" if version else " (所有版本)")
        
        return {"success": True, "message": message}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")


@router.get("/models/{name}/download")
async def download_model(
    name: str,
    version: str = "latest",
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    下载模型文件
    
    Args:
        name: 模型名称
        version: 模型版本
        registry: 模型注册表实例
        
    Returns:
        模型文件
    """
    try:
        model_entry = registry.get_model_info(name, version)
        if not model_entry:
            raise HTTPException(status_code=404, detail=f"模型未找到: {name}:{version}")
        
        if not os.path.exists(model_entry.model_path):
            raise HTTPException(status_code=404, detail="模型文件不存在")
        
        filename = f"{name}_v{version}"
        if model_entry.metadata.format == ModelFormat.PYTORCH:
            filename += ".pth"
        elif model_entry.metadata.format == ModelFormat.ONNX:
            filename += ".onnx"
        else:
            filename += ".zip"  # 对于目录形式的模型
        
        # 如果是目录，需要压缩后下载
        if os.path.isdir(model_entry.model_path):
            import zipfile
            import tempfile
            
            temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
            with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(model_entry.model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, model_entry.model_path)
                        zipf.write(file_path, arcname)
            
            return FileResponse(
                temp_zip.name,
                filename=filename,
                media_type='application/zip'
            )
        else:
            return FileResponse(
                model_entry.model_path,
                filename=filename
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"下载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"下载模型失败: {str(e)}")


@router.get("/models/{name}/export")
async def export_model(
    name: str,
    version: str = "latest",
    export_format: Optional[ModelFormat] = None,
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    导出模型到指定格式
    
    Args:
        name: 模型名称
        version: 模型版本
        export_format: 导出格式
        registry: 模型注册表实例
        
    Returns:
        导出的模型文件
    """
    try:
        # 创建临时导出路径
        import tempfile
        temp_dir = tempfile.mkdtemp()
        export_path = os.path.join(temp_dir, f"{name}_exported")
        
        # 导出模型
        exported_path = registry.export_model(name, version, export_path, export_format)
        
        # 返回导出的文件
        if os.path.isfile(exported_path):
            return FileResponse(exported_path, filename=os.path.basename(exported_path))
        else:
            # 如果是目录，压缩后返回
            import zipfile
            zip_path = f"{exported_path}.zip"
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(exported_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, exported_path)
                        zipf.write(file_path, arcname)
            
            return FileResponse(zip_path, filename=f"{name}_exported.zip")
            
    except Exception as e:
        logger.error(f"导出模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"导出模型失败: {str(e)}")


@router.get("/validate", response_model=ValidationResponse)
async def validate_registry(
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    验证注册表完整性
    
    Args:
        registry: 模型注册表实例
        
    Returns:
        验证结果
    """
    try:
        validation_result = registry.validate_registry()
        
        return ValidationResponse(
            errors=validation_result["errors"],
            warnings=validation_result["warnings"],
            total_models=validation_result["total_models"],
            valid_models=validation_result["valid_models"]
        )
        
    except Exception as e:
        logger.error(f"验证注册表失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证注册表失败: {str(e)}")


@router.get("/stats")
async def get_registry_stats(
    registry: ModelRegistry = Depends(get_model_registry)
):
    """
    获取注册表统计信息
    
    Args:
        registry: 模型注册表实例
        
    Returns:
        统计信息
    """
    try:
        models = registry.list_models()
        
        # 按格式统计
        format_stats = {}
        type_stats = {}
        total_size = 0
        total_params = 0
        
        for model in models:
            # 格式统计
            format_name = model.metadata.format.value
            format_stats[format_name] = format_stats.get(format_name, 0) + 1
            
            # 类型统计
            type_name = model.metadata.model_type.value
            type_stats[type_name] = type_stats.get(type_name, 0) + 1
            
            # 大小统计
            if model.metadata.model_size_mb:
                total_size += model.metadata.model_size_mb
            
            # 参数统计
            if model.metadata.parameters_count:
                total_params += model.metadata.parameters_count
        
        return {
            "total_models": len(models),
            "formats": format_stats,
            "types": type_stats,
            "total_size_mb": round(total_size, 2),
            "total_parameters": total_params,
            "average_size_mb": round(total_size / len(models), 2) if models else 0
        }
        
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


def cleanup_temp_dir(temp_dir: str):
    """清理临时目录"""
    try:
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
    except Exception as e:
        logger.warning(f"清理临时目录失败: {e}")


# 错误处理
@router.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))


@router.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return HTTPException(status_code=404, detail=str(exc))


@router.exception_handler(PermissionError)
async def permission_error_handler(request, exc):
    return HTTPException(status_code=403, detail=str(exc))