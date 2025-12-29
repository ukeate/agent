# 模型微调和优化平台用户指南

## 快速开始

### 1. 平台访问
访问平台地址：http://localhost:8000
使用您的账户凭证登录系统。

### 2. 组件注册

#### 2.1 注册新组件
POST /platform/components/register
Content-Type: application/json

{
  "component_id": "fine_tuning_service",
  "component_type": "fine_tuning",
  "name": "Fine-tuning Service",
  "version": "1.0.0",
  "health_endpoint": "http://localhost:8001/health",
  "api_endpoint": "http://localhost:8001",
  "metadata": {"description": "LoRA fine-tuning service"}
}

#### 2.2 查看已注册组件
GET /platform/components

### 3. 执行工作流

#### 3.1 完整微调工作流
POST /platform/workflows/run
Content-Type: application/json

{
  "workflow_type": "full_fine_tuning",
  "parameters": {
    "model_name": "llama-2-7b",
    "data_config": {
      "dataset": "custom_training_data",
      "batch_size": 32,
      "max_length": 512
    },
    "training_config": {
      "learning_rate": 0.0001,
      "num_epochs": 3,
      "warmup_steps": 100
    }
  },
  "priority": 1
}

### 4. 监控工作流

#### 4.1 检查工作流状态
GET /platform/workflows/{workflow_id}/status

#### 4.2 平台健康检查
GET /platform/health

### 5. 性能监控

#### 5.1 查看Prometheus指标
访问：http://localhost:8000/metrics

#### 5.2 生成监控报告
GET /platform/monitoring/report

## 工作流类型

### 1. full_fine_tuning
完整的模型微调流程，包括：
- 数据准备
- 超参数优化
- 模型微调
- 模型评估
- 模型压缩
- 模型部署

### 2. model_optimization
模型优化流程，包括：
- 模型压缩
- 模型量化
- 性能评估

## 支持的组件类型

- fine_tuning: 微调服务
- compression: 压缩服务
- hyperparameter: 超参数优化服务
- evaluation: 评估服务
- data_management: 数据管理服务
- model_service: 模型服务

## 常见问题

### Q: 工作流执行失败怎么办？
A: 检查工作流状态API获取详细错误信息，确认相关组件服务正常运行。

### Q: 如何添加自定义组件？
A: 实现组件的健康检查和API接口，然后通过组件注册API注册到平台。

### Q: 如何监控平台性能？
A: 使用Prometheus指标端点或监控报告API获取详细性能数据。

### Q: 支持并发工作流吗？
A: 支持，平台可以同时执行多个工作流，但受到组件资源限制。
