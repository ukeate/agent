"""模型部署管理系统"""

import asyncio
import json
import subprocess
import yaml
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import uuid
import tempfile
import shutil
from .registry import ModelRegistry, ModelMetadata
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging
import torch
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

class DeploymentType(str, Enum):
    """部署类型"""
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    EDGE = "edge"

class DeploymentStatus(str, Enum):
    """部署状态"""
    PENDING = "pending"
    BUILDING = "building"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class DeploymentConfig:
    """部署配置"""
    deployment_type: DeploymentType
    model_name: str
    model_version: str
    replicas: int = 1
    cpu_request: str = "200m"
    cpu_limit: str = "1000m"
    memory_request: str = "512Mi"
    memory_limit: str = "2Gi"
    gpu_required: bool = False
    gpu_count: int = 0
    port: int = 8080
    environment_vars: Dict[str, str] = None
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    health_check_path: str = "/health"
    readiness_check_path: str = "/ready"
    
    def __post_init__(self):
        if self.environment_vars is None:
            self.environment_vars = {}
        if self.labels is None:
            self.labels = {}
        if self.annotations is None:
            self.annotations = {}

@dataclass
class DeploymentInfo:
    """部署信息"""
    deployment_id: str
    model_name: str
    model_version: str
    deployment_type: DeploymentType
    status: DeploymentStatus
    config: DeploymentConfig
    endpoint_url: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.updated_at is None:
            self.updated_at = self.created_at

class DockerDeploymentManager:
    """Docker部署管理器"""
    
    def __init__(self, base_image: str = "python:3.11-slim"):
        self.base_image = base_image
    
    async def deploy_model(
        self, 
        deployment_id: str,
        model_info: Dict[str, Any], 
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """部署模型到Docker容器"""
        try:
            # 生成Dockerfile
            dockerfile_content = self._generate_dockerfile(model_info, config)
            
            # 创建临时构建目录
            with tempfile.TemporaryDirectory() as build_dir:
                build_path = Path(build_dir)
                
                # 写入Dockerfile
                dockerfile_path = build_path / "Dockerfile"
                with open(dockerfile_path, 'w') as f:
                    f.write(dockerfile_content)
                
                # 复制模型文件
                model_file = Path(model_info['model_path'])
                if model_file.exists():
                    shutil.copy2(model_file, build_path / "model")
                
                # 生成启动脚本
                startup_script = self._generate_startup_script(config)
                with open(build_path / "start.py", 'w') as f:
                    f.write(startup_script)
                
                # 构建Docker镜像
                image_name = f"model-service-{deployment_id}:latest"
                build_cmd = [
                    "docker", "build", 
                    "-t", image_name,
                    str(build_path)
                ]
                
                logger.info(f"构建Docker镜像: {image_name}")
                result = await self._run_command(build_cmd)
                
                if result["returncode"] != 0:
                    return {
                        "success": False,
                        "error": f"Docker构建失败: {result['stderr']}"
                    }
                
                # 运行容器
                run_cmd = [
                    "docker", "run", "-d",
                    "-p", f"{config.port}:{config.port}",
                    "--name", f"model-service-{deployment_id}",
                ]
                
                # 添加环境变量
                for key, value in config.environment_vars.items():
                    run_cmd.extend(["-e", f"{key}={value}"])
                
                # 添加GPU支持
                if config.gpu_required:
                    run_cmd.extend(["--gpus", f"{config.gpu_count}"])
                
                run_cmd.append(image_name)
                
                logger.info(f"运行Docker容器: {deployment_id}")
                result = await self._run_command(run_cmd)
                
                if result["returncode"] != 0:
                    return {
                        "success": False,
                        "error": f"Docker容器启动失败: {result['stderr']}"
                    }
                
                # 验证容器状态
                container_id = result["stdout"].strip()
                inspect_cmd = ["docker", "inspect", container_id]
                inspect_result = await self._run_command(inspect_cmd)
                
                if inspect_result["returncode"] == 0:
                    inspect_data = json.loads(inspect_result["stdout"])[0]
                    container_state = inspect_data.get("State", {})
                    
                    if container_state.get("Running"):
                        return {
                            "success": True,
                            "container_id": container_id,
                            "endpoint_url": f"http://localhost:{config.port}",
                            "status": "running"
                        }
                
                return {
                    "success": False,
                    "error": "容器启动后状态检查失败"
                }
                
        except Exception as e:
            logger.error(f"Docker部署失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_dockerfile(self, model_info: Dict[str, Any], config: DeploymentConfig) -> str:
        """生成Dockerfile"""
        requirements = self._get_requirements(model_info)
        
        dockerfile = f"""FROM {self.base_image}

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip install --no-cache-dir {' '.join(requirements)}

# 复制模型文件
COPY model /app/model

# 复制启动脚本
COPY start.py /app/start.py

# 暴露端口
EXPOSE {config.port}

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:{config.port}{config.health_check_path} || exit 1

# 启动服务
CMD ["python", "start.py"]
"""
        return dockerfile
    
    def _get_requirements(self, model_info: Dict[str, Any]) -> List[str]:
        """获取Python依赖包"""
        base_requirements = [
            "fastapi==0.104.1",
            "uvicorn[standard]==0.24.0",
            "pydantic==2.5.0",
            "numpy==1.24.3"
        ]
        
        # 根据模型格式添加特定依赖
        if model_info.get('format') == 'pytorch':
            base_requirements.extend([
                "torch==2.1.0",
                "torchvision==0.16.0"
            ])
        elif model_info.get('format') == 'onnx':
            base_requirements.append("onnxruntime==1.16.0")
        elif model_info.get('format') == 'huggingface':
            base_requirements.extend([
                "transformers==4.36.0",
                "tokenizers==0.15.0"
            ])
        
        return base_requirements
    
    def _generate_startup_script(self, config: DeploymentConfig) -> str:
        """生成启动脚本"""
        script = f"""#!/usr/bin/env python3

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global model
    try:
        model_path = "/app/model"
        if os.path.exists(model_path):
            model = torch.load(model_path, map_location="cpu")
            logger.info("模型加载成功")
        else:
            logger.error("模型文件不存在")
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
    yield

app = FastAPI(title="Model Service", version="1.0.0", lifespan=lifespan)

# 全局模型变量
model = None

class PredictionRequest(BaseModel):
    inputs: Dict[str, Any]
    parameters: Dict[str, Any] = {{}}

class PredictionResponse(BaseModel):
    outputs: Dict[str, Any]
    processing_time_ms: float

@app.get("{config.health_check_path}")
async def health_check():
    return {{"status": "healthy", "model_loaded": model is not None}}

@app.get("{config.readiness_check_path}")
async def readiness_check():
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return {{"status": "ready"}}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    
    import time
    start_time = time.time()
    
    try:
        # 执行推理
        with torch.no_grad():
            outputs = model(torch.tensor(request.inputs.get("data", [])))
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            outputs={{"prediction": outputs.tolist()}},
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {{str(e)}}")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port={config.port},
        log_level="info"
    )
"""
        return script
    
    async def stop_deployment(self, deployment_id: str) -> Dict[str, Any]:
        """停止Docker部署"""
        try:
            # 停止容器
            stop_cmd = ["docker", "stop", f"model-service-{deployment_id}"]
            result = await self._run_command(stop_cmd)
            
            if result["returncode"] != 0:
                return {"success": False, "error": f"停止容器失败: {result['stderr']}"}
            
            # 删除容器
            rm_cmd = ["docker", "rm", f"model-service-{deployment_id}"]
            await self._run_command(rm_cmd)
            
            # 删除镜像
            rmi_cmd = ["docker", "rmi", f"model-service-{deployment_id}:latest"]
            await self._run_command(rmi_cmd)
            
            return {"success": True, "message": "部署已停止"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """获取部署状态"""
        try:
            inspect_cmd = ["docker", "inspect", f"model-service-{deployment_id}"]
            result = await self._run_command(inspect_cmd)
            
            if result["returncode"] != 0:
                return {"status": "not_found"}
            
            inspect_data = json.loads(result["stdout"])[0]
            container_state = inspect_data.get("State", {})
            
            return {
                "status": "running" if container_state.get("Running") else "stopped",
                "container_id": inspect_data.get("Id"),
                "created": inspect_data.get("Created"),
                "state": container_state
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """运行系统命令"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8'),
                "stderr": stderr.decode('utf-8')
            }
            
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }

class KubernetesDeploymentManager:
    """Kubernetes部署管理器"""
    
    def __init__(self, namespace: str = "default"):
        self.namespace = namespace
    
    async def deploy_model(
        self, 
        deployment_id: str,
        model_info: Dict[str, Any], 
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """部署模型到Kubernetes"""
        try:
            # 生成K8s配置
            k8s_config = self._generate_k8s_config(
                model_info['name'], 
                deployment_id, 
                config.__dict__
            )
            
            # 创建临时配置文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                yaml.dump(k8s_config, f, default_flow_style=False)
                config_path = f.name
            
            try:
                # 应用配置
                apply_cmd = ["kubectl", "apply", "-f", config_path, "-n", self.namespace]
                result = await self._run_command(apply_cmd)
                
                if result["returncode"] != 0:
                    return {
                        "success": False,
                        "error": f"Kubernetes部署失败: {result['stderr']}"
                    }
                
                # 等待部署就绪
                await self._wait_for_deployment(deployment_id)
                
                # 获取服务端点
                endpoint = await self._get_service_endpoint(deployment_id)
                
                return {
                    "success": True,
                    "deployment_id": deployment_id,
                    "endpoint_url": endpoint,
                    "status": "deployed"
                }
                
            finally:
                # 清理临时文件
                Path(config_path).unlink(missing_ok=True)
                
        except Exception as e:
            logger.error(f"Kubernetes部署失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_k8s_config(self, model_name: str, deployment_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """生成Kubernetes配置"""
        labels = {"app": f"model-{deployment_id}", "model": model_name}
        
        # Deployment配置
        deployment_config = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"model-{deployment_id}",
                "namespace": self.namespace,
                "labels": labels
            },
            "spec": {
                "replicas": config.get("replicas", 1),
                "selector": {"matchLabels": labels},
                "template": {
                    "metadata": {"labels": labels},
                    "spec": {
                        "containers": [{
                            "name": "model-service",
                            "image": f"model-service-{deployment_id}:latest",
                            "ports": [{"containerPort": config.get("port", 8080)}],
                            "resources": {
                                "requests": {
                                    "cpu": config.get("cpu_request", "200m"),
                                    "memory": config.get("memory_request", "512Mi")
                                },
                                "limits": {
                                    "cpu": config.get("cpu_limit", "1000m"),
                                    "memory": config.get("memory_limit", "2Gi")
                                }
                            },
                            "env": [
                                {"name": k, "value": str(v)} 
                                for k, v in config.get("environment_vars", {}).items()
                            ],
                            "livenessProbe": {
                                "httpGet": {
                                    "path": config.get("health_check_path", "/health"),
                                    "port": config.get("port", 8080)
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": config.get("readiness_check_path", "/ready"),
                                    "port": config.get("port", 8080)
                                },
                                "initialDelaySeconds": 5,
                                "periodSeconds": 5
                            }
                        }]
                    }
                }
            }
        }
        
        # 如果需要GPU，添加GPU资源
        if config.get("gpu_required", False):
            gpu_count = config.get("gpu_count", 1)
            deployment_config["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = str(gpu_count)
        
        return deployment_config
    
    async def _wait_for_deployment(self, deployment_id: str, timeout: int = 300):
        """等待部署完成"""
        cmd = [
            "kubectl", "rollout", "status", 
            f"deployment/model-{deployment_id}",
            "-n", self.namespace,
            f"--timeout={timeout}s"
        ]
        await self._run_command(cmd)
    
    async def _get_service_endpoint(self, deployment_id: str) -> Optional[str]:
        """获取服务端点"""
        # 这里简化处理，实际需要根据Service类型确定端点
        return f"http://model-{deployment_id}.{self.namespace}.svc.cluster.local"
    
    async def _run_command(self, cmd: List[str]) -> Dict[str, Any]:
        """运行kubectl命令"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            return {
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8'),
                "stderr": stderr.decode('utf-8')
            }
            
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": str(e)
            }

class EdgeDeploymentManager:
    """边缘设备部署管理器"""
    
    def __init__(self):
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
    
    async def deploy_model(
        self, 
        deployment_id: str,
        model_info: Dict[str, Any], 
        config: DeploymentConfig
    ) -> Dict[str, Any]:
        """部署模型到边缘设备"""
        # 这里是边缘部署的简化实现
        # 实际场景中需要考虑设备资源限制、网络传输等
        
        try:
            logger.info(f"开始边缘部署: {deployment_id}")
            
            # 模拟边缘部署过程
            await asyncio.sleep(2)
            
            return {
                "success": True,
                "deployment_id": deployment_id,
                "endpoint_url": f"http://edge-device:8080",
                "status": "deployed"
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}

class DeploymentManager:
    """统一部署管理器"""
    
    def __init__(
        self, 
        model_registry: ModelRegistry,
        docker_base_image: str = "python:3.11-slim",
        k8s_namespace: str = "default"
    ):
        self.model_registry = model_registry
        self.deployments: Dict[str, DeploymentInfo] = {}
        
        # 初始化各类型部署管理器
        self.docker_manager = DockerDeploymentManager(docker_base_image)
        self.k8s_manager = KubernetesDeploymentManager(k8s_namespace)
        self.edge_manager = EdgeDeploymentManager()
        
        # 部署管理器映射
        self.deployment_managers = {
            DeploymentType.DOCKER: self.docker_manager,
            DeploymentType.KUBERNETES: self.k8s_manager,
            DeploymentType.EDGE: self.edge_manager
        }
    
    async def deploy_model(
        self, 
        model_name: str, 
        model_version: str,
        deployment_type: DeploymentType,
        config: Dict[str, Any]
    ) -> Optional[str]:
        """部署模型"""
        try:
            # 生成部署ID
            deployment_id = str(uuid.uuid4())[:8]
            
            # 获取模型信息
            model_metadata = self.model_registry.get_model(model_name, model_version)
            if not model_metadata:
                logger.error(f"模型不存在: {model_name}:{model_version}")
                return None
            
            model_path = self.model_registry.get_model_path(model_name, model_metadata.version)
            if not model_path:
                logger.error(f"模型文件不存在: {model_name}:{model_metadata.version}")
                return None
            
            # 创建部署配置
            deployment_config = DeploymentConfig(
                deployment_type=deployment_type,
                model_name=model_name,
                model_version=model_metadata.version,
                **config
            )
            
            # 创建部署信息
            deployment_info = DeploymentInfo(
                deployment_id=deployment_id,
                model_name=model_name,
                model_version=model_metadata.version,
                deployment_type=deployment_type,
                status=DeploymentStatus.PENDING,
                config=deployment_config
            )
            
            # 保存部署信息
            self.deployments[deployment_id] = deployment_info
            
            # 更新状态为构建中
            deployment_info.status = DeploymentStatus.BUILDING
            deployment_info.updated_at = datetime.now(timezone.utc)
            
            # 准备模型信息
            model_info = {
                "name": model_metadata.name,
                "version": model_metadata.version,
                "format": model_metadata.format.value,
                "framework": model_metadata.framework,
                "model_path": model_path
            }
            
            # 获取对应的部署管理器
            manager = self.deployment_managers.get(deployment_type)
            if not manager:
                deployment_info.status = DeploymentStatus.FAILED
                deployment_info.error_message = f"不支持的部署类型: {deployment_type}"
                return None
            
            # 执行部署
            deployment_info.status = DeploymentStatus.DEPLOYING
            deployment_info.updated_at = datetime.now(timezone.utc)
            
            result = await manager.deploy_model(deployment_id, model_info, deployment_config)
            
            if result.get("success"):
                deployment_info.status = DeploymentStatus.DEPLOYED
                deployment_info.endpoint_url = result.get("endpoint_url")
                logger.info(f"部署成功: {deployment_id}")
            else:
                deployment_info.status = DeploymentStatus.FAILED
                deployment_info.error_message = result.get("error")
                logger.error(f"部署失败: {deployment_id}, 错误: {deployment_info.error_message}")
            
            deployment_info.updated_at = datetime.now(timezone.utc)
            return deployment_id
            
        except Exception as e:
            logger.error(f"部署过程异常: {e}")
            if 'deployment_info' in locals():
                deployment_info.status = DeploymentStatus.FAILED
                deployment_info.error_message = str(e)
                deployment_info.updated_at = datetime.now(timezone.utc)
            return None
    
    async def stop_deployment(self, deployment_id: str) -> bool:
        """停止部署"""
        if deployment_id not in self.deployments:
            return False
        
        deployment_info = self.deployments[deployment_id]
        manager = self.deployment_managers.get(deployment_info.deployment_type)
        
        if not manager:
            return False
        
        try:
            if hasattr(manager, 'stop_deployment'):
                result = await manager.stop_deployment(deployment_id)
                if result.get("success"):
                    deployment_info.status = DeploymentStatus.STOPPED
                    deployment_info.updated_at = datetime.now(timezone.utc)
                    return True
            return False
        except Exception as e:
            logger.error(f"停止部署失败: {e}")
            return False
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """获取部署状态"""
        if deployment_id not in self.deployments:
            return None
        
        deployment_info = self.deployments[deployment_id]
        return {
            "deployment_id": deployment_id,
            "model_name": deployment_info.model_name,
            "model_version": deployment_info.model_version,
            "deployment_type": deployment_info.deployment_type.value,
            "status": deployment_info.status.value,
            "endpoint_url": deployment_info.endpoint_url,
            "error_message": deployment_info.error_message,
            "created_at": deployment_info.created_at.isoformat(),
            "updated_at": deployment_info.updated_at.isoformat()
        }
    
    def list_deployments(self, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """列出所有部署"""
        result = []
        for deployment_info in self.deployments.values():
            if model_name and deployment_info.model_name != model_name:
                continue
            
            result.append({
                "deployment_id": deployment_info.deployment_id,
                "model_name": deployment_info.model_name,
                "model_version": deployment_info.model_version,
                "deployment_type": deployment_info.deployment_type.value,
                "status": deployment_info.status.value,
                "endpoint_url": deployment_info.endpoint_url,
                "created_at": deployment_info.created_at.isoformat()
            })
        
        return result
    
    def get_deployment_metrics(self) -> Dict[str, Any]:
        """获取部署指标"""
        status_counts = {}
        type_counts = {}
        
        for deployment_info in self.deployments.values():
            status = deployment_info.status.value
            dep_type = deployment_info.deployment_type.value
            
            status_counts[status] = status_counts.get(status, 0) + 1
            type_counts[dep_type] = type_counts.get(dep_type, 0) + 1
        
        return {
            "total_deployments": len(self.deployments),
            "status_distribution": status_counts,
            "type_distribution": type_counts,
            "active_deployments": status_counts.get("deployed", 0)
        }
