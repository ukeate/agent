# AI Agent System

基于多智能体架构的企业级AI开发平台，实现从单体AI向专业化智能体网络的范式转变。

## 快速开始

### 前置要求

- Python 3.11+
- Docker 24.0+
- Docker Compose 2.23+
- Node.js 18+ (前端开发)
- PostgreSQL 15+
- Redis 7.2+

### 本地开发环境设置

1. 克隆项目
```bash
git clone <repository-url>
cd ai-agent-system
```

2. 启动开发环境
```bash
./scripts/setup-dev.sh
```

3. 启动服务
```bash
docker compose up -d
```

4. 访问应用
- API文档: http://localhost:8000/docs
- 前端应用: http://localhost:3000

## 项目架构

### 技术栈

- **后端**: Python 3.11+ + FastAPI 0.104+
- **前端**: React 18.2+ + TypeScript 5.3+
- **数据库**: PostgreSQL 15+ + Redis 7.2+
- **AI框架**: LangGraph 0.0.69+ + AutoGen 0.2.18+
- **工具协议**: MCP 1.0+
- **容器化**: Docker 24.0+ + Docker Compose 2.23+

### 项目结构

```
ai-agent-system/
├── apps/                    # 应用程序包
│   ├── api/                # FastAPI后端
│   └── web/                # React前端
├── packages/               # 共享包
│   └── shared/            # 共享类型和工具
├── infrastructure/         # 基础设施配置
│   └── docker/            # Docker配置
├── scripts/               # 构建和部署脚本
└── docs/                  # 项目文档
```

## 开发指南

### 编码标准

- **Python**: 使用snake_case命名函数，PascalCase命名类
- **TypeScript**: 组件使用PascalCase，hook使用camelCase with 'use'
- **API路由**: 使用kebab-case格式
- **错误处理**: 所有API路由必须使用标准错误处理器
- **类型共享**: 所有数据类型必须在packages/shared中定义

### 测试

```bash
# 后端测试
cd apps/api && pytest

# 前端测试
cd apps/web && npm test

# 端到端测试
npm run test:e2e
```

## 部署

### 开发环境
```bash
docker compose -f infrastructure/docker/docker-compose.yml up
```

### 生产环境
```bash
docker compose -f infrastructure/docker/docker-compose.prod.yml up
```

## 贡献

请阅读 [CONTRIBUTING.md](CONTRIBUTING.md) 了解贡献指南。

## 许可证

[MIT License](LICENSE)