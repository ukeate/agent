# AI Agent 系统开发指南

## 项目概述
个人AI智能体学习项目，通过构建完整的多智能体系统来掌握现代AI开发技术栈。主要目标是学习和实践LangGraph、AutoGen、MCP协议、RAG系统等前沿技术。

## 核心技术栈
- **框架**: FastAPI + LangGraph 0.6.5 + AutoGen
- **协议**: MCP 1.0 标准化工具集成
- **模型**: Claude-3.5-Sonnet
- **数据库**: PostgreSQL + Redis + Qdrant
- **部署**: Docker + Kubernetes

## 开发原则
- 增量优于重构
- 杜绝简化
- KISS原则(keep it simple and stupid)
- 代码决策优先级：可测试性 > 可读性 > 一致性 > 简单性
- 故事开发完成、bug修改完时，请运行单元测试和e2e测试，并用playwright mcp做一遍功能覆盖调试，修正遇到的问题
- UI中左边栏要穷尽罗列出所有页面与功能，不能减少任何条目

## 测试原则
- playwirght mcp发现上下文太长时，记得截图查看或用js查询时在js内过滤掉LLM返回的消息

## 测试方法
```bash
# 前端测试
cd apps/web
npm test                    # 单元测试
npm run test:e2e           # E2E测试

# 后端测试
cd apps/api/src
uv run pytest          # 运行测试
```

## 调试方法


## 环境启动
### 基础服务启动
```bash
# 启动PostgreSQL、Redis、Qdrant等基础服务
cd infrastructure/docker
docker-compose up -d

# 验证服务状态
docker-compose ps
```

### 后端服务启动
```bash
# 进入后端目录
cd apps/api/src
script -q /dev/null uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload | tee -a api.log
```

### 前端服务启动
```bash
# 进入前端目录
cd apps/web
# 启动开发服务器
npm run dev
# 前端运行在: http://localhost:3000
```
