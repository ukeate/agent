# AI Agent 系统开发指南

## 项目概述
个人AI智能体学习项目，通过构建完整的多智能体系统来掌握现代AI开发技术栈。主要目标是学习和实践LangGraph、AutoGen、MCP协议、RAG系统等前沿技术。

## 核心技术栈
- **框架**: FastAPI + LangGraph + AutoGen
- **协议**: MCP 1.0 标准化工具集成
- **模型**: Claude-3.5-Sonnet
- **数据库**: PostgreSQL + Redis + Qdrant
- **部署**: Docker + Kubernetes

## 开发阶段
1. **MVP阶段** (Week 1-4): 单代理系统 + MCP集成
2. **扩展阶段** (Week 5-8): Supervisor多代理协作
3. **生产阶段** (Week 9-12): RAG系统 + 完整部署

## 代理角色定义
- **代码专家**: 代码生成、审查、重构、测试
- **架构师**: 系统设计、技术选型、文档编写
- **文档专家**: 技术文档、API文档、用户手册
- **任务调度器**: 任务分解、代理协调、质量控制
- **知识检索专家**: 语义搜索、知识整合、答案生成

## 开发原则
- 优先实现功能，后续优化性能
- 代码质量高于开发速度
- 每个代理专注单一职责
- 完整错误处理和日志记录
- 渐进式复杂度递增
- 故事开发完成, 请运行单元测试和e2e测试，并用playwright mcp做一遍功能覆盖调试
- playwirght mcp发现上下文太长时，记得用js查询时在js内过滤掉LLM返回的消息
- bug修改完，要找到对应的单元测试和e2e测试执行并修改bug

## 开发环境启动指南

### 1. 基础服务启动
```bash
# 启动PostgreSQL、Redis、Qdrant等基础服务
cd infrastructure/docker
docker-compose up -d

# 验证服务状态
docker-compose ps
```

### 2. 后端服务启动
```bash
# 进入后端目录
cd apps/api/src
script -q /dev/null uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload | tee -a api.log
```

### 3. 前端服务启动
```bash
# 进入前端目录
cd apps/web
# 启动开发服务器
npm run dev
# 前端运行在: http://localhost:3000
```

### 4. 服务验证
```bash
# 验证后端API
curl http://localhost:8000/api/v1/agent/status

# 验证前端页面
curl -I http://localhost:3000

# 测试流式聊天接口
curl -X POST "http://localhost:8000/api/v1/agent/chat" \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"message": "你好", "stream": true}' \
  --no-buffer
```

### 5. 前后端联调测试
```bash
# 完整系统测试
curl -X POST "http://localhost:8000/api/v1/agent/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "前后端联调测试", "stream": false}'
```
### 8. 开发调试
```bash
# 前端测试
cd apps/web
npm test                    # 单元测试
npm run test:e2e           # E2E测试

# 后端测试
cd apps/api/src
uv run pytest          # 运行测试

# 代码格式化
npm run format             # 前端格式化
uv run black .             # 后端格式化
```
