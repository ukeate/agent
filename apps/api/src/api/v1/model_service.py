"""模型服务平台API接口"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import uuid
from pathlib import Path
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, BackgroundTasks
from src.api.base_model import ApiBaseModel
from src.ai.model_service.inference import (
    InferenceEngine, 
    InferenceRequest,
    InferenceResult,
    InferenceStatus
)
from src.ai.model_service.deployment import (
    DeploymentManager,
    DeploymentType,
    DeploymentConfig
)
from src.ai.model_service.online_learning import (
    OnlineLearningEngine,
    FeedbackData,
    ABTestConfig
)
from src.ai.model_service.monitoring import MonitoringSystem
from src.ai.model_service.registry import (
    ModelRegistry,
    ModelRegistrationRequest,
    ModelFormat,
    ModelMetadata,
)
from src.core.dependencies import get_current_user

logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(_: APIRouter) -> AsyncGenerator[None, None]:
    """模型服务路由生命周期管理"""
    try:
        await monitoring_system.start_monitoring(collection_interval=60)
        logger.info("模型服务平台API已启动")
    except Exception as e:
        logger.error(f"启动监控系统失败: {e}")
    yield
    try:
        await monitoring_system.stop_monitoring()
        logger.info("模型服务平台API已关闭")
    except Exception as e:
        logger.error(f"关闭监控系统失败: {e}")

router = APIRouter(prefix="/model-service", tags=["Model Service"], lifespan=lifespan)

# ============= Pydantic Models =============

class ModelUploadRequest(ApiBaseModel):
    """模型上传请求"""
    name: str = Field(..., description="模型名称")
    version: str = Field(..., description="模型版本")
    format: ModelFormat = Field(..., description="模型格式")
    framework: str = Field(..., description="框架名称")
    description: Optional[str] = Field(None, description="模型描述")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    input_schema: Optional[Dict[str, Any]] = Field(None, description="输入Schema")
    output_schema: Optional[Dict[str, Any]] = Field(None, description="输出Schema")

class ModelUploadResponse(ApiBaseModel):
    """模型上传响应"""
    model_id: str
    message: str

class ModelListResponse(ApiBaseModel):
    """模型列表响应"""
    models: List[Dict[str, Any]]
    total: int

class InferenceRequestModel(ApiBaseModel):
    """推理请求模型"""
    model_name: str = Field(..., description="模型名称")
    model_version: str = Field(default="latest", description="模型版本")
    inputs: Dict[str, Any] = Field(..., description="输入数据")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="推理参数")
    batch_size: int = Field(default=1, description="批处理大小")
    timeout_seconds: int = Field(default=30, description="超时时间")

class InferenceResponseModel(ApiBaseModel):
    """推理响应模型"""
    request_id: str
    status: str
    outputs: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time_ms: Optional[float] = None

class DeploymentRequestModel(ApiBaseModel):
    """部署请求模型"""
    model_name: str = Field(..., description="模型名称")
    model_version: str = Field(default="latest", description="模型版本")
    deployment_type: DeploymentType = Field(..., description="部署类型")
    replicas: int = Field(default=1, description="副本数量")
    cpu_request: str = Field(default="200m", description="CPU请求")
    cpu_limit: str = Field(default="1000m", description="CPU限制")
    memory_request: str = Field(default="512Mi", description="内存请求")
    memory_limit: str = Field(default="2Gi", description="内存限制")
    gpu_required: bool = Field(default=False, description="是否需要GPU")
    gpu_count: int = Field(default=0, description="GPU数量")
    port: int = Field(default=8080, description="服务端口")
    environment_vars: Dict[str, str] = Field(default_factory=dict, description="环境变量")

class FeedbackRequestModel(ApiBaseModel):
    """反馈请求模型"""
    prediction_id: str = Field(..., description="预测ID")
    inputs: Dict[str, Any] = Field(..., description="输入数据")
    expected_output: Any = Field(..., description="期望输出")
    actual_output: Any = Field(..., description="实际输出")
    feedback_type: str = Field(..., description="反馈类型")
    quality_score: float = Field(default=1.0, description="质量分数")
    user_id: Optional[str] = Field(None, description="用户ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="元数据")

class ABTestRequestModel(ApiBaseModel):
    """A/B测试请求模型"""
    name: str = Field(..., description="测试名称")
    description: str = Field(..., description="测试描述")
    control_model: str = Field(..., description="对照组模型")
    treatment_models: List[str] = Field(..., description="实验组模型")
    traffic_split: Dict[str, float] = Field(..., description="流量分配")
    success_metrics: List[str] = Field(..., description="成功指标")
    minimum_sample_size: int = Field(default=1000, description="最小样本量")
    confidence_level: float = Field(default=0.95, description="置信水平")
    max_duration_days: int = Field(default=30, description="最大持续时间")

# ============= Global Components =============

# 全局组件实例（在实际项目中应该通过依赖注入管理）
model_registry = ModelRegistry()
inference_engine = InferenceEngine(model_registry)
deployment_manager = DeploymentManager(model_registry)
online_learning_engine = OnlineLearningEngine(model_registry, deployment_manager)
monitoring_system = MonitoringSystem()

# ============= Model Registry APIs =============

@router.post("/models/upload", response_model=ModelUploadResponse)
async def upload_model(
    model_file: UploadFile = File(...),
    request: str = Query(..., description="JSON格式的模型注册请求"),
    current_user: dict = Depends(get_current_user)
):
    """上传并注册模型"""
    try:
        # 解析请求参数
        model_request = ModelRegistrationRequest.model_validate_json(request)
        
        # 验证文件格式
        if not model_file.filename:
            raise HTTPException(status_code=400, detail="文件名不能为空")
        
        # 保存临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(model_file.filename).suffix) as temp_file:
            content = await model_file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # 注册模型
            model_id = await model_registry.register_model(model_request, temp_file_path)
            
            # 记录监控指标
            monitoring_system.metrics_collector.record_metric(
                "model_registry.upload_count", 
                1,
                labels={"format": model_request.format.value}
            )
            
            return ModelUploadResponse(
                model_id=model_id,
                message=f"模型 {model_request.name}:{model_request.version} 注册成功"
            )
            
        finally:
            # 清理临时文件
            Path(temp_file_path).unlink(missing_ok=True)
            
    except Exception as e:
        logger.error(f"模型上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型上传失败: {str(e)}")

@router.post("/models/register-from-hub")
async def register_from_hub(
    model_name: str = Query(..., description="HuggingFace模型名称"),
    local_name: str = Query(..., description="本地模型名称"),
    version: str = Query(default="1.0", description="版本号"),
    description: Optional[str] = Query(None, description="模型描述"),
    current_user: dict = Depends(get_current_user)
):
    """从HuggingFace Hub注册模型"""
    try:
        from transformers import AutoModel, AutoTokenizer
        import tempfile
        import shutil
        
        # 下载模型到临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            model = AutoModel.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # 保存模型
            model_path = Path(temp_dir) / "model"
            model_path.mkdir()
            model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            
            # 创建注册请求
            request = ModelRegistrationRequest(
                name=local_name,
                version=version,
                format=ModelFormat.HUGGINGFACE,
                framework="transformers",
                description=description or f"从{model_name}导入的HuggingFace模型",
                tags=["huggingface", "imported"]
            )
            
            # 注册模型
            model_id = await model_registry.register_model(request, str(model_path))
            
            return {
                "model_id": model_id,
                "message": f"从HuggingFace Hub成功注册模型 {local_name}:{version}"
            }
            
    except Exception as e:
        logger.error(f"从Hub注册模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"从Hub注册模型失败: {str(e)}")

@router.get("/models", response_model=ModelListResponse)
async def list_models(
    name_filter: Optional[str] = Query(None, description="名称过滤"),
    tags: Optional[str] = Query(None, description="标签过滤，逗号分隔"),
    format_filter: Optional[ModelFormat] = Query(None, description="格式过滤"),
    limit: int = Query(default=100, description="返回数量限制"),
    offset: int = Query(default=0, description="偏移量")
):
    """获取模型列表"""
    try:
        # 解析标签过滤
        tag_list = tags.split(",") if tags else None
        
        # 获取所有模型
        all_models = model_registry.list_models(name_filter, tag_list)
        
        # 格式过滤
        if format_filter:
            all_models = [m for m in all_models if m.format == format_filter]
        
        # 分页
        total = len(all_models)
        models = all_models[offset:offset+limit]
        
        # 转换为响应格式
        model_list = []
        for model in models:
            model_dict = {
                "model_id": model.model_id,
                "name": model.name,
                "version": model.version,
                "format": model.format.value,
                "framework": model.framework,
                "description": model.description,
                "tags": model.tags,
                "model_size_mb": model.model_size_mb,
                "parameter_count": model.parameter_count,
                "created_at": model.created_at.isoformat(),
                "updated_at": model.updated_at.isoformat()
            }
            model_list.append(model_dict)
        
        return ModelListResponse(models=model_list, total=total)
        
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")

@router.get("/models/{model_name}/versions/{version}")
async def get_model(model_name: str, version: str = "latest"):
    """获取特定模型信息"""
    try:
        model = model_registry.get_model(model_name, version)
        if not model:
            raise HTTPException(status_code=404, detail="模型未找到")
        
        return {
            "model_id": model.model_id,
            "name": model.name,
            "version": model.version,
            "format": model.format.value,
            "framework": model.framework,
            "description": model.description,
            "tags": model.tags,
            "input_schema": model.input_schema,
            "output_schema": model.output_schema,
            "model_size_mb": model.model_size_mb,
            "parameter_count": model.parameter_count,
            "created_at": model.created_at.isoformat(),
            "updated_at": model.updated_at.isoformat(),
            "checksum": model.checksum
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取模型信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取模型信息失败: {str(e)}")

@router.delete("/models/{model_name}/versions/{version}")
async def delete_model(
    model_name: str, 
    version: str,
    current_user: dict = Depends(get_current_user)
):
    """删除模型"""
    try:
        success = model_registry.delete_model(model_name, version)
        if not success:
            raise HTTPException(status_code=404, detail="模型未找到或删除失败")
        
        return {"message": f"模型 {model_name}:{version} 已删除"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除模型失败: {str(e)}")

@router.post("/models/{model_name}/versions/{version}/validate")
async def validate_model(model_name: str, version: str = "latest"):
    """验证模型完整性"""
    try:
        result = model_registry.validate_model(model_name, version)
        return result
        
    except Exception as e:
        logger.error(f"验证模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"验证模型失败: {str(e)}")

# ============= Inference APIs =============

@router.post("/inference/predict", response_model=InferenceResponseModel)
async def predict(request: InferenceRequestModel):
    """执行模型推理"""
    try:
        # 创建推理请求
        inference_request = InferenceRequest(
            request_id=str(uuid.uuid4()),
            model_name=request.model_name,
            model_version=request.model_version,
            inputs=request.inputs,
            parameters=request.parameters,
            batch_size=request.batch_size,
            timeout_seconds=request.timeout_seconds
        )
        
        # 记录开始时间
        start_time = datetime.now(timezone.utc)
        
        # 执行推理
        result = await inference_engine.inference(inference_request)
        
        # 计算处理时间
        processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        # 记录监控指标
        model_key = f"{request.model_name}:{request.model_version}"
        success = result.status == InferenceStatus.COMPLETED
        monitoring_system.record_model_inference(
            model_key, 
            processing_time, 
            success,
            request.batch_size
        )
        
        return InferenceResponseModel(
            request_id=result.request_id,
            status=result.status.value,
            outputs=result.outputs,
            error=result.error,
            processing_time_ms=result.processing_time_ms
        )
        
    except Exception as e:
        logger.error(f"推理执行失败: {e}")
        raise HTTPException(status_code=500, detail=f"推理执行失败: {str(e)}")

@router.post("/inference/batch-predict")
async def batch_predict(requests: List[InferenceRequestModel]):
    """批量推理"""
    try:
        # 创建推理请求列表
        inference_requests = []
        for req in requests:
            inference_request = InferenceRequest(
                request_id=str(uuid.uuid4()),
                model_name=req.model_name,
                model_version=req.model_version,
                inputs=req.inputs,
                parameters=req.parameters,
                batch_size=req.batch_size,
                timeout_seconds=req.timeout_seconds
            )
            inference_requests.append(inference_request)
        
        # 执行批量推理
        results = await inference_engine.batch_inference(inference_requests)
        
        # 转换结果格式
        response_list = []
        for result in results:
            if isinstance(result, Exception):
                response_list.append({
                    "request_id": "unknown",
                    "status": "failed",
                    "error": str(result)
                })
            else:
                response_list.append(InferenceResponseModel(
                    request_id=result.request_id,
                    status=result.status.value,
                    outputs=result.outputs,
                    error=result.error,
                    processing_time_ms=result.processing_time_ms
                ))
        
        return {"results": response_list, "total": len(response_list)}
        
    except Exception as e:
        logger.error(f"批量推理失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量推理失败: {str(e)}")

@router.get("/inference/models/loaded")
async def get_loaded_models():
    """获取已加载的模型列表"""
    try:
        models = inference_engine.get_loaded_models()
        return {"loaded_models": models}
        
    except Exception as e:
        logger.error(f"获取已加载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取已加载模型失败: {str(e)}")

@router.post("/inference/models/{model_name}/load")
async def load_model(
    model_name: str,
    version: str = Query(default="latest", description="模型版本"),
    current_user: dict = Depends(get_current_user)
):
    """加载模型到内存"""
    try:
        success = inference_engine.load_model(model_name, version)
        if not success:
            raise HTTPException(status_code=400, detail="模型加载失败")
        
        return {"message": f"模型 {model_name}:{version} 加载成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"加载模型失败: {str(e)}")

@router.delete("/inference/models/{model_name}/unload")
async def unload_model(
    model_name: str,
    version: str = Query(default="latest", description="模型版本"),
    current_user: dict = Depends(get_current_user)
):
    """卸载模型"""
    try:
        success = inference_engine.unload_model(model_name, version)
        if not success:
            raise HTTPException(status_code=404, detail="模型未找到或卸载失败")
        
        return {"message": f"模型 {model_name}:{version} 卸载成功"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"卸载模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"卸载模型失败: {str(e)}")

# ============= Deployment APIs =============

@router.post("/deployment/deploy")
async def deploy_model(
    request: DeploymentRequestModel,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """部署模型"""
    try:
        config = {
            "replicas": request.replicas,
            "cpu_request": request.cpu_request,
            "cpu_limit": request.cpu_limit,
            "memory_request": request.memory_request,
            "memory_limit": request.memory_limit,
            "gpu_required": request.gpu_required,
            "gpu_count": request.gpu_count,
            "port": request.port,
            "environment_vars": request.environment_vars
        }
        
        # 异步执行部署
        deployment_id = await deployment_manager.deploy_model(
            request.model_name,
            request.model_version,
            request.deployment_type,
            config
        )
        
        if not deployment_id:
            raise HTTPException(status_code=400, detail="部署失败")
        
        return {
            "deployment_id": deployment_id,
            "message": f"模型 {request.model_name}:{request.model_version} 开始部署",
            "deployment_type": request.deployment_type.value
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"部署模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"部署模型失败: {str(e)}")

@router.get("/deployment/list")
async def list_deployments(model_name: Optional[str] = Query(None, description="模型名称过滤")):
    """列出所有部署"""
    try:
        deployments = deployment_manager.list_deployments(model_name)
        return {"deployments": deployments, "total": len(deployments)}
        
    except Exception as e:
        logger.error(f"获取部署列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取部署列表失败: {str(e)}")

@router.get("/deployment/{deployment_id}")
async def get_deployment_status(deployment_id: str):
    """获取部署状态"""
    try:
        status = deployment_manager.get_deployment_status(deployment_id)
        if not status:
            raise HTTPException(status_code=404, detail="部署未找到")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取部署状态失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取部署状态失败: {str(e)}")

@router.delete("/deployment/{deployment_id}")
async def stop_deployment(
    deployment_id: str,
    current_user: dict = Depends(get_current_user)
):
    """停止部署"""
    try:
        success = await deployment_manager.stop_deployment(deployment_id)
        if not success:
            raise HTTPException(status_code=404, detail="部署未找到或停止失败")
        
        return {"message": f"部署 {deployment_id} 已停止"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止部署失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止部署失败: {str(e)}")

# ============= Online Learning APIs =============

@router.post("/learning/start")
async def start_online_learning(
    model_name: str = Query(..., description="模型名称"),
    model_version: str = Query(default="latest", description="模型版本"),
    config: Dict[str, Any] = None,
    current_user: dict = Depends(get_current_user)
):
    """开始在线学习会话"""
    try:
        if config is None:
            config = {"learning_rate": 1e-4, "batch_size": 32}
        
        session_id = await online_learning_engine.start_online_learning(
            model_name, model_version, config
        )
        
        return {
            "session_id": session_id,
            "message": f"在线学习会话已启动: {model_name}:{model_version}"
        }
        
    except Exception as e:
        logger.error(f"启动在线学习失败: {e}")
        raise HTTPException(status_code=500, detail=f"启动在线学习失败: {str(e)}")

@router.post("/learning/{session_id}/feedback")
async def submit_feedback(
    session_id: str,
    feedback: FeedbackRequestModel
):
    """提交用户反馈"""
    try:
        await online_learning_engine.collect_feedback(
            session_id,
            feedback.prediction_id,
            feedback.model_dump()
        )
        
        return {"message": "反馈已提交"}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"提交反馈失败: {e}")
        raise HTTPException(status_code=500, detail=f"提交反馈失败: {str(e)}")

@router.post("/learning/{session_id}/update")
async def update_model_with_feedback(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """使用反馈更新模型"""
    try:
        result = await online_learning_engine.update_model(session_id)
        return result
        
    except Exception as e:
        logger.error(f"模型更新失败: {e}")
        raise HTTPException(status_code=500, detail=f"模型更新失败: {str(e)}")

@router.get("/learning/{session_id}/stats")
async def get_learning_stats(session_id: str):
    """获取学习统计信息"""
    try:
        stats = online_learning_engine.get_learning_stats(session_id)
        if not stats:
            raise HTTPException(status_code=404, detail="学习会话未找到")
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取学习统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取学习统计失败: {str(e)}")

@router.get("/learning/{session_id}/history")
async def get_learning_history(session_id: str):
    """获取学习性能历史"""
    try:
        stats = online_learning_engine.get_learning_stats(session_id)
        if not stats:
            raise HTTPException(status_code=404, detail="学习会话未找到")
        history = online_learning_engine.get_performance_history(session_id)
        return {"session_id": session_id, "history": history}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取学习历史失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取学习历史失败: {str(e)}")

@router.get("/learning/sessions")
async def list_learning_sessions():
    """获取所有学习会话"""
    try:
        sessions = online_learning_engine.get_all_learning_sessions()
        return {"sessions": sessions, "total": len(sessions)}
        
    except Exception as e:
        logger.error(f"获取学习会话列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取学习会话列表失败: {str(e)}")

@router.post("/learning/{session_id}/pause")
async def pause_learning_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """暂停学习会话"""
    try:
        success = await online_learning_engine.pause_learning_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="学习会话未找到")
        return {"session_id": session_id, "status": "paused"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"暂停学习会话失败: {e}")
        raise HTTPException(status_code=500, detail=f"暂停学习会话失败: {str(e)}")

@router.post("/learning/{session_id}/resume")
async def resume_learning_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """恢复学习会话"""
    try:
        success = await online_learning_engine.resume_learning_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="学习会话未找到")
        return {"session_id": session_id, "status": "active"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"恢复学习会话失败: {e}")
        raise HTTPException(status_code=500, detail=f"恢复学习会话失败: {str(e)}")

@router.post("/learning/{session_id}/stop")
async def stop_learning_session(
    session_id: str,
    current_user: dict = Depends(get_current_user)
):
    """停止学习会话"""
    try:
        success = await online_learning_engine.stop_learning_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="学习会话未找到")
        return {"session_id": session_id, "status": "completed"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"停止学习会话失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止学习会话失败: {str(e)}")

# ============= A/B Testing APIs =============

@router.post("/abtest/create")
async def create_ab_test(
    request: ABTestRequestModel,
    current_user: dict = Depends(get_current_user)
):
    """创建A/B测试"""
    try:
        test_id = online_learning_engine.create_ab_test(
            request.name,
            request.description,
            request.control_model,
            request.treatment_models,
            request.traffic_split,
            request.success_metrics,
            minimum_sample_size=request.minimum_sample_size,
            confidence_level=request.confidence_level,
            max_duration_days=request.max_duration_days
        )
        
        return {
            "test_id": test_id,
            "message": f"A/B测试 '{request.name}' 已创建"
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"创建A/B测试失败: {e}")
        raise HTTPException(status_code=500, detail=f"创建A/B测试失败: {str(e)}")

@router.get("/abtest/{test_id}/assign")
async def assign_model_for_user(
    test_id: str,
    user_id: str = Query(..., description="用户ID")
):
    """为用户分配A/B测试模型"""
    try:
        model_id = online_learning_engine.assign_model_for_user(test_id, user_id)
        if not model_id:
            raise HTTPException(status_code=404, detail="A/B测试未找到")
        
        return {"model_id": model_id, "test_id": test_id, "user_id": user_id}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"分配A/B测试模型失败: {e}")
        raise HTTPException(status_code=500, detail=f"分配A/B测试模型失败: {str(e)}")

@router.post("/abtest/{test_id}/record")
async def record_ab_test_result(
    test_id: str,
    user_id: str = Query(..., description="用户ID"),
    metrics: Dict[str, float] = None
):
    """记录A/B测试结果"""
    try:
        if metrics is None:
            metrics = {}
        
        online_learning_engine.record_ab_test_metrics(test_id, user_id, metrics)
        return {"message": "A/B测试结果已记录"}
        
    except Exception as e:
        logger.error(f"记录A/B测试结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"记录A/B测试结果失败: {str(e)}")

@router.get("/abtest/{test_id}/results")
async def get_ab_test_results(test_id: str):
    """获取A/B测试结果"""
    try:
        results = online_learning_engine.get_ab_test_results(test_id)
        return results
        
    except Exception as e:
        logger.error(f"获取A/B测试结果失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取A/B测试结果失败: {str(e)}")

@router.get("/abtest/list")
async def list_ab_tests():
    """获取所有A/B测试"""
    try:
        tests = online_learning_engine.get_ab_tests()
        return {"tests": tests, "total": len(tests)}
        
    except Exception as e:
        logger.error(f"获取A/B测试列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取A/B测试列表失败: {str(e)}")

# ============= Monitoring APIs =============

@router.get("/monitoring/overview")
async def get_monitoring_overview():
    """获取监控概览"""
    try:
        overview = monitoring_system.get_system_overview()
        model_metrics = overview.get("model_metrics", {})
        total_requests = 0
        total_latency = 0.0
        total_errors = 0.0

        for metrics in model_metrics.values():
            request_count = metrics.get("request_count", 0)
            total_requests += request_count
            total_latency += metrics.get("avg_latency_ms", 0.0) * request_count
            total_errors += metrics.get("error_rate", 0.0) * request_count

        average_latency = total_latency / total_requests if total_requests else 0.0
        error_rate = total_errors / total_requests if total_requests else 0.0
        deployment_metrics = deployment_manager.get_deployment_metrics()
        system_metrics = overview.get("system_metrics", {})
        resource_utilization = {
            "cpu_percent": system_metrics.get("system.cpu_percent", {}).get("current", 0.0),
            "memory_percent": system_metrics.get("system.memory_percent", {}).get("current", 0.0),
            "gpu_percent": system_metrics.get("system.gpu_percent", {}).get("current", 0.0),
        }

        return {
            "total_models": len(model_registry.models),
            "active_deployments": deployment_metrics.get("active_deployments", 0),
            "total_requests": total_requests,
            "average_latency": average_latency,
            "error_rate": error_rate,
            "resource_utilization": resource_utilization,
        }
        
    except Exception as e:
        logger.error(f"获取监控概览失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取监控概览失败: {str(e)}")

@router.get("/monitoring/dashboard")
async def get_dashboard_data(hours: int = Query(default=24, description="数据时间范围(小时)")):
    """获取仪表板数据"""
    try:
        dashboard = monitoring_system.get_metrics_dashboard_data(hours)
        return dashboard
        
    except Exception as e:
        logger.error(f"获取仪表板数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取仪表板数据失败: {str(e)}")

@router.get("/monitoring/alerts")
async def get_alerts(
    active_only: bool = Query(default=True, description="只返回活跃告警"),
    hours: int = Query(default=24, description="时间范围(小时)")
):
    """获取告警列表"""
    try:
        if active_only:
            alerts = monitoring_system.alert_manager.get_active_alerts()
        else:
            alerts = monitoring_system.alert_manager.get_all_alerts(hours)
        
        # 转换为可序列化格式
        alert_list = []
        for alert in alerts:
            alert_dict = {
                "alert_id": alert.alert_id,
                "name": alert.name,
                "description": alert.description,
                "severity": alert.severity.value,
                "metric_name": alert.metric_name,
                "threshold_value": alert.threshold_value,
                "actual_value": alert.actual_value,
                "timestamp": alert.timestamp.isoformat(),
                "resolved": alert.resolved,
                "resolved_at": alert.resolved_at.isoformat() if alert.resolved_at else None
            }
            alert_list.append(alert_dict)
        
        return {"alerts": alert_list, "total": len(alert_list)}
        
    except Exception as e:
        logger.error(f"获取告警列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取告警列表失败: {str(e)}")

@router.get("/monitoring/recommendations")
async def get_optimization_recommendations():
    """获取性能优化建议"""
    try:
        recommendations = monitoring_system.get_resource_recommendations()
        return {"recommendations": recommendations, "total": len(recommendations)}
        
    except Exception as e:
        logger.error(f"获取优化建议失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取优化建议失败: {str(e)}")

@router.get("/monitoring/metrics/{metric_name}")
async def get_metric_data(
    metric_name: str,
    hours: int = Query(default=1, description="时间范围(小时)")
):
    """获取特定指标数据"""
    try:
        summary = monitoring_system.metrics_collector.get_metric_summary(
            metric_name, 
            window_minutes=hours * 60
        )
        return summary
        
    except Exception as e:
        logger.error(f"获取指标数据失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取指标数据失败: {str(e)}")

# ============= System APIs =============

@router.get("/health")
async def health_check():
    """系统健康检查"""
    try:
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "model_registry": {"status": "healthy", "models_count": len(model_registry.models)},
                "inference_engine": inference_engine.health_check(),
                "deployment_manager": {
                    "status": "healthy", 
                    "deployments": deployment_manager.get_deployment_metrics()
                },
                "online_learning": online_learning_engine.health_check(),
                "monitoring": monitoring_system.health_check()
            }
        }
        
        # 检查是否有组件异常
        component_statuses = [comp.get("status") for comp in health_data["components"].values()]
        if any(status in ["degraded", "critical"] for status in component_statuses):
            health_data["status"] = "degraded"
        
        return health_data
        
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        return {
            "status": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": str(e)
        }

@router.get("/statistics")
async def get_system_statistics():
    """获取系统统计信息"""
    try:
        stats = {
            "registry": model_registry.get_statistics(),
            "inference": inference_engine.get_metrics(),
            "deployment": deployment_manager.get_deployment_metrics(),
            "learning": {
                "total_sessions": len(online_learning_engine.learning_sessions),
                "active_sessions": sum(1 for s in online_learning_engine.learning_sessions.values() 
                                     if s.status.value == "active"),
                "ab_tests": len(online_learning_engine.ab_test_engine.active_tests)
            },
            "monitoring": {
                "active_alerts": len(monitoring_system.alert_manager.get_active_alerts()),
                "total_metrics": sum(len(deque_obj) for deque_obj in monitoring_system.metrics_collector.metrics.values())
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"获取系统统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取系统统计失败: {str(e)}")
from src.core.logging import get_logger
