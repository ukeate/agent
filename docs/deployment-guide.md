# 部署与运维指南

## 目录
- [系统要求](#系统要求)
- [环境准备](#环境准备)
- [部署架构](#部署架构)
- [安装步骤](#安装步骤)
- [配置管理](#配置管理)
- [数据库迁移](#数据库迁移)
- [监控设置](#监控设置)
- [性能优化](#性能优化)
- [故障排查](#故障排查)
- [备份恢复](#备份恢复)
- [安全加固](#安全加固)
- [扩展指南](#扩展指南)

## 系统要求

### 硬件要求

#### 最小配置
- CPU: 4核
- 内存: 8GB RAM
- 存储: 100GB SSD
- 网络: 100Mbps

#### 推荐配置
- CPU: 8核+
- 内存: 16GB+ RAM
- 存储: 500GB+ SSD
- 网络: 1Gbps

### 软件要求
- 操作系统: Ubuntu 20.04 LTS / CentOS 8 / macOS 12+
- Python: 3.11+
- Node.js: 18+
- Docker: 24.0+
- PostgreSQL: 15+
- Redis: 7.0+
- Qdrant: 1.7+

## 环境准备

### 1. 安装基础依赖

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    nodejs npm \
    docker.io docker-compose \
    git curl wget \
    build-essential libssl-dev libffi-dev

# macOS (使用Homebrew)
brew install python@3.11 node docker docker-compose
```

### 2. 安装Python包管理器 (uv)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### 3. 克隆项目

```bash
git clone https://github.com/your-org/ai-agent-system.git
cd ai-agent-system
```

## 部署架构

### 单机部署架构
```
┌─────────────────────────────────────────────┐
│                  Nginx/Traefik              │
│                   (反向代理)                 │
└─────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  Frontend   │ │   API Server │ │  WebSocket  │
│  (React)    │ │  (FastAPI)   │ │   Server    │
│  Port:3000  │ │  Port:8000   │ │  Port:8001  │
└─────────────┘ └─────────────┘ └─────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ PostgreSQL  │ │    Redis     │ │   Qdrant    │
│  Port:5432  │ │  Port:6379   │ │  Port:6333  │
└─────────────┘ └─────────────┘ └─────────────┘
```

### 分布式部署架构
```
┌──────────────────────────────────────────────────┐
│                   Load Balancer                   │
│                  (AWS ALB/NLB)                    │
└──────────────────────────────────────────────────┘
                          │
     ┌────────────────────┼────────────────────┐
     ▼                    ▼                    ▼
┌──────────┐        ┌──────────┐        ┌──────────┐
│ Frontend │        │ Frontend │        │ Frontend │
│  Node 1  │        │  Node 2  │        │  Node 3  │
└──────────┘        └──────────┘        └──────────┘
                          │
     ┌────────────────────┼────────────────────┐
     ▼                    ▼                    ▼
┌──────────┐        ┌──────────┐        ┌──────────┐
│   API    │        │   API    │        │   API    │
│ Server 1 │        │ Server 2 │        │ Server 3 │
└──────────┘        └──────────┘        └──────────┘
                          │
     ┌────────────────────┼────────────────────┐
     ▼                    ▼                    ▼
┌──────────┐        ┌──────────┐        ┌──────────┐
│PostgreSQL│        │  Redis   │        │  Qdrant  │
│ Primary  │        │ Cluster  │        │ Cluster  │
└──────────┘        └──────────┘        └──────────┘
     │
┌──────────┐
│PostgreSQL│
│ Replica  │
└──────────┘
```

## 安装步骤

### 1. 使用Docker Compose部署（推荐）

```bash
# 进入项目目录
cd ai-agent-system

# 创建环境变量文件
cp .env.example .env
# 编辑.env文件，配置必要的环境变量

# 启动所有服务
cd infrastructure/docker
docker-compose up -d

# 查看服务状态
docker-compose ps

# 查看日志
docker-compose logs -f
```

### 2. 手动部署

#### 部署PostgreSQL
```bash
# 使用Docker
docker run -d \
  --name postgres \
  -e POSTGRES_DB=ai_agent \
  -e POSTGRES_USER=admin \
  -e POSTGRES_PASSWORD=your_password \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:15

# 创建数据库
docker exec -it postgres psql -U admin -c "CREATE DATABASE ai_agent_db;"
```

#### 部署Redis
```bash
docker run -d \
  --name redis \
  -p 6379:6379 \
  -v redis_data:/data \
  redis:7-alpine \
  redis-server --appendonly yes
```

#### 部署Qdrant
```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -v qdrant_data:/qdrant/storage \
  qdrant/qdrant
```

#### 部署后端服务
```bash
# 进入后端目录
cd apps/api

# 安装依赖
uv pip install -r requirements.txt

# 运行数据库迁移
cd src
uv run alembic upgrade head

# 启动服务
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 部署前端服务
```bash
# 进入前端目录
cd apps/web

# 安装依赖
npm install

# 构建生产版本
npm run build

# 使用PM2启动
npm install -g pm2
pm2 start npm --name "ai-agent-web" -- start
```

### 3. Kubernetes部署

```bash
# 应用配置
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml

# 部署数据库
kubectl apply -f k8s/postgres/
kubectl apply -f k8s/redis/
kubectl apply -f k8s/qdrant/

# 部署应用
kubectl apply -f k8s/backend/
kubectl apply -f k8s/frontend/

# 部署Ingress
kubectl apply -f k8s/ingress.yaml
```

## 配置管理

### 环境变量配置

```bash
# 数据库配置
DATABASE_URL=postgresql://user:password@localhost:5432/ai_agent_db
REDIS_URL=redis://localhost:6379/0

# API配置
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# 认证配置
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Claude API配置
ANTHROPIC_API_KEY=your-anthropic-api-key
CLAUDE_MODEL=claude-3.5-sonnet

# 监控配置
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
LOG_FORMAT=json

# 缓存配置
CACHE_ENABLED=true
CACHE_TTL=3600
CACHE_MAX_SIZE=1000

# A/B测试配置
EXPERIMENT_CACHE_TTL=300
EVENT_BATCH_SIZE=100
EVENT_FLUSH_INTERVAL=10
```

### 配置文件管理

```yaml
# config/production.yaml
database:
  host: localhost
  port: 5432
  name: ai_agent_db
  pool_size: 20
  max_overflow: 10

redis:
  host: localhost
  port: 6379
  db: 0
  pool_size: 10

api:
  cors_origins:
    - https://app.ai-agent.com
    - https://admin.ai-agent.com
  rate_limit:
    requests_per_minute: 100
    burst_size: 200

monitoring:
  metrics_enabled: true
  logging:
    level: INFO
    format: json
    rotation: daily
    retention_days: 30
```

## 数据库迁移

### 创建迁移
```bash
cd apps/api/src
uv run alembic revision --autogenerate -m "描述变更"
```

### 执行迁移
```bash
# 升级到最新版本
uv run alembic upgrade head

# 升级到特定版本
uv run alembic upgrade +1

# 查看当前版本
uv run alembic current

# 查看历史
uv run alembic history
```

### 回滚迁移
```bash
# 回滚一个版本
uv run alembic downgrade -1

# 回滚到特定版本
uv run alembic downgrade <revision>
```

## 监控设置

### Prometheus配置
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ai-agent-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics

  - job_name: 'postgres'
    static_configs:
      - targets: ['localhost:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['localhost:9121']
```

### Grafana仪表板
```bash
# 导入仪表板
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @grafana/dashboards/ai-agent-dashboard.json
```

### 日志聚合 (ELK Stack)
```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/ai-agent/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["localhost:9200"]
  index: "ai-agent-%{+yyyy.MM.dd}"
```

## 性能优化

### 数据库优化
```sql
-- 创建索引
CREATE INDEX idx_experiments_status ON experiments(status);
CREATE INDEX idx_events_user_experiment ON events(user_id, experiment_id);
CREATE INDEX idx_assignments_user ON user_assignments(user_id);

-- 分区表
CREATE TABLE events_2024_01 PARTITION OF events
FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');

-- 查询优化
VACUUM ANALYZE experiments;
REINDEX TABLE events;
```

### Redis优化
```bash
# redis.conf
maxmemory 2gb
maxmemory-policy allkeys-lru
save 900 1
save 300 10
save 60 10000
```

### API优化
```python
# 使用连接池
from databases import Database

database = Database(
    DATABASE_URL,
    min_size=10,
    max_size=20,
    command_timeout=60
)

# 使用缓存
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_experiment(experiment_id: str):
    # 缓存实验配置
    pass

# 批量处理
async def batch_track_events(events: List[Event]):
    # 批量插入事件
    async with database.transaction():
        await database.execute_many(...)
```

## 故障排查

### 常见问题

#### 1. 数据库连接失败
```bash
# 检查PostgreSQL状态
docker ps | grep postgres
docker logs postgres

# 测试连接
psql -h localhost -U admin -d ai_agent_db

# 检查防火墙
sudo ufw status
```

#### 2. Redis连接超时
```bash
# 检查Redis状态
redis-cli ping

# 查看Redis日志
docker logs redis

# 检查内存使用
redis-cli info memory
```

#### 3. API响应慢
```bash
# 查看慢查询日志
tail -f logs/slow_queries.log

# 分析API性能
python -m cProfile -o profile.stats main.py

# 查看系统资源
htop
iostat -x 1
```

### 日志分析
```bash
# 查找错误
grep ERROR logs/app.log | tail -100

# 统计请求量
awk '{print $4}' access.log | sort | uniq -c | sort -rn

# 分析响应时间
awk '{sum+=$10; count++} END {print sum/count}' access.log
```

## 备份恢复

### 数据库备份
```bash
# 全量备份
pg_dump -h localhost -U admin -d ai_agent_db > backup_$(date +%Y%m%d).sql

# 压缩备份
pg_dump -h localhost -U admin -d ai_agent_db | gzip > backup_$(date +%Y%m%d).sql.gz

# 定时备份脚本
#!/bin/bash
BACKUP_DIR="/backup/postgres"
DB_NAME="ai_agent_db"
DATE=$(date +%Y%m%d_%H%M%S)

pg_dump -h localhost -U admin -d $DB_NAME > $BACKUP_DIR/backup_$DATE.sql
find $BACKUP_DIR -name "backup_*.sql" -mtime +7 -delete
```

### 数据恢复
```bash
# 恢复数据库
psql -h localhost -U admin -d ai_agent_db < backup.sql

# 恢复压缩备份
gunzip -c backup.sql.gz | psql -h localhost -U admin -d ai_agent_db
```

### Redis备份
```bash
# 手动触发RDB快照
redis-cli BGSAVE

# 备份AOF文件
cp /var/lib/redis/appendonly.aof backup_$(date +%Y%m%d).aof
```

## 安全加固

### SSL/TLS配置
```nginx
server {
    listen 443 ssl http2;
    server_name api.ai-agent.com;

    ssl_certificate /etc/ssl/certs/ai-agent.crt;
    ssl_certificate_key /etc/ssl/private/ai-agent.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 防火墙配置
```bash
# 配置UFW
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 安全扫描
```bash
# 依赖漏洞扫描
pip install safety
safety check

# 代码安全扫描
pip install bandit
bandit -r apps/api/src/

# Docker镜像扫描
docker scan ai-agent-api:latest
```

## 扩展指南

### 水平扩展

#### 添加API服务器节点
```bash
# 在新节点上部署API服务
docker run -d \
  --name ai-agent-api \
  -e DATABASE_URL=postgresql://... \
  -e REDIS_URL=redis://... \
  -p 8000:8000 \
  ai-agent-api:latest
```

#### 配置负载均衡
```nginx
upstream api_servers {
    least_conn;
    server api1.internal:8000 weight=1;
    server api2.internal:8000 weight=1;
    server api3.internal:8000 weight=1;
}

server {
    location /api {
        proxy_pass http://api_servers;
    }
}
```

### 数据库读写分离
```python
# 配置主从数据库
MASTER_DB_URL = "postgresql://master:5432/ai_agent"
REPLICA_DB_URL = "postgresql://replica:5432/ai_agent"

# 读写分离
async def get_db_for_read():
    return Database(REPLICA_DB_URL)

async def get_db_for_write():
    return Database(MASTER_DB_URL)
```

### 缓存层扩展
```python
# Redis集群配置
from redis.cluster import RedisCluster

startup_nodes = [
    {"host": "redis1", "port": "7000"},
    {"host": "redis2", "port": "7000"},
    {"host": "redis3", "port": "7000"}
]

redis_cluster = RedisCluster(startup_nodes=startup_nodes)
```

## 监控告警配置

### AlertManager配置
```yaml
# alertmanager.yml
route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 12h
  receiver: 'team-ops'

receivers:
- name: 'team-ops'
  email_configs:
  - to: 'ops@ai-agent.com'
    from: 'alertmanager@ai-agent.com'
  webhook_configs:
  - url: 'http://localhost:5001/webhooks/alerts'
```

### 告警规则
```yaml
# alert_rules.yml
groups:
- name: api_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.05
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: HighLatency
    expr: histogram_quantile(0.95, http_request_duration_seconds) > 1
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "95th percentile latency is {{ $value }} seconds"
```

## 升级流程

### 零停机升级
```bash
#!/bin/bash
# 蓝绿部署脚本

# 1. 部署新版本到绿色环境
docker-compose -f docker-compose.green.yml up -d

# 2. 健康检查
./health_check.sh green

# 3. 切换流量
./switch_traffic.sh green

# 4. 停止蓝色环境
docker-compose -f docker-compose.blue.yml down

# 5. 更新蓝色环境配置
cp docker-compose.green.yml docker-compose.blue.yml
```

### 数据库升级
```bash
# 1. 备份当前数据库
pg_dump -h localhost -U admin -d ai_agent_db > pre_upgrade_backup.sql

# 2. 执行迁移
cd apps/api/src
uv run alembic upgrade head

# 3. 验证升级
uv run python verify_migration.py
```

## 故障恢复流程

### 应急响应
1. **识别问题**: 通过监控告警或用户反馈发现问题
2. **评估影响**: 确定受影响的服务和用户范围
3. **临时缓解**: 实施临时解决方案（如切换到备用服务）
4. **根因分析**: 查找问题根本原因
5. **永久修复**: 实施长期解决方案
6. **事后总结**: 记录问题和解决方案，更新运维手册

### 灾难恢复
```bash
# 完整系统恢复流程
#!/bin/bash

# 1. 恢复基础设施
terraform apply -auto-approve

# 2. 恢复数据库
psql -h new-db-host -U admin -d ai_agent_db < latest_backup.sql

# 3. 恢复Redis数据
redis-cli --rdb /backup/dump.rdb

# 4. 部署应用
kubectl apply -f k8s/

# 5. 验证服务
./smoke_tests.sh
```

## 维护计划

### 日常维护
- 检查系统健康状态
- 查看错误日志
- 监控资源使用情况
- 清理临时文件

### 周期维护
- 每周：数据库备份验证
- 每月：安全更新检查
- 每季度：性能基准测试
- 每年：灾难恢复演练