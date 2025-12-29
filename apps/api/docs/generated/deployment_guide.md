# 部署指南

## Docker部署

### 1. 准备环境

确保系统安装了以下软件：
- Docker 24.0+
- Docker Compose 2.23+
- 至少8GB RAM
- 至少50GB磁盘空间

### 2. 构建镜像

构建API服务镜像：
cd apps/api
docker build -t model-platform-api:latest .

构建前端服务镜像：
cd apps/web
docker build -t model-platform-web:latest .

### 3. 启动服务

启动所有服务：
cd infrastructure/docker
docker-compose up -d

检查服务状态：
docker-compose ps

查看日志：
docker-compose logs -f platform-api

## Kubernetes部署

### 1. 创建命名空间

kubectl create namespace model-platform

### 2. 部署配置

应用Kubernetes配置文件部署服务

### 3. 验证部署

kubectl get pods -n model-platform

## 生产环境配置

### 1. 环境变量

设置必要的生产环境变量，包括数据库连接、Redis配置、安全密钥等。

### 2. 数据库优化

优化PostgreSQL配置以获得更好的性能。

### 3. 缓存配置

配置Redis缓存策略和持久化选项。

## 监控和日志

### 1. Prometheus配置

配置Prometheus监控指标收集。

### 2. Grafana仪表板

设置Grafana仪表板进行可视化监控。

### 3. 日志聚合

配置日志聚合和分析系统。

## 备份和恢复

### 1. 数据库备份

实现自动数据库备份策略。

### 2. 配置备份

备份重要的配置文件。

## 故障排除

### 常见问题

1. 服务启动失败
2. 数据库连接问题
3. Redis连接问题
4. 性能问题

每个问题都包含详细的诊断和解决步骤。
