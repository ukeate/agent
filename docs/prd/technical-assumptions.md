# ⚙️ Technical Assumptions

## Repository Structure: Monorepo

**选择理由：** 单一仓库包含所有组件，简化开发、构建和部署流程。对于学习项目而言，monorepo便于管理依赖关系和版本控制，避免多仓库的复杂性。

**仓库结构：**
```
agent-system/
├── src/agents/          # 智能体定义和实现
├── src/orchestration/   # LangGraph和AutoGen编排
├── src/rag/            # RAG系统实现
├── src/dag/            # DAG规划和执行
├── src/api/            # FastAPI接口层
├── mcp_servers/        # MCP服务器实现
├── web/               # 简单前端界面
├── docs/              # 项目文档
├── tests/             # 测试代码
└── deployment/        # 部署配置
```

## Service Architecture

**架构选择：** 在单一进程中运行的模块化应用，避免微服务的网络开销和复杂性。各模块通过内存通信和Redis队列进行协调。

**关键架构决策：**
- **进程内通信**：智能体间通过内存对象直接通信
- **异步处理**：使用asyncio处理并发任务和I/O密集操作
- **状态管理**：LangGraph提供工作流状态，Redis存储会话状态
- **扩展路径**：模块接口设计支持未来向微服务架构演进

## Testing Requirements

**测试策略：**
- **单元测试**：使用pytest测试核心组件和智能体逻辑
- **集成测试**：测试多智能体协作和外部API集成
- **模拟测试**：模拟OpenAI API调用以控制测试成本
- **手动测试**：复杂工作流场景的端到端验证

**测试覆盖目标：** 核心业务逻辑>80%，API接口100%覆盖

## Additional Technical Assumptions and Requests

**编程语言和框架：**
- **后端主语言**：Python 3.9+ （生态成熟，AI库丰富）
- **Web框架**：FastAPI 0.115+ （高性能，自动文档生成）
- **异步运行时**：asyncio + uvloop （高并发性能）

**AI和智能体技术栈：**
- **多智能体编排**：LangGraph 0.6+ + AutoGen 0.4+ （混合架构）
- **主要LLM模型**：OpenAI GPT-4o-mini （成本优化的高效推理）
- **工具集成协议**：MCP 1.0 （标准化工具接口）
- **嵌入模型**：OpenAI Embeddings（RAG系统）

**数据存储技术：**
- **主数据库**：PostgreSQL 15+ （关系数据和JSON支持）
- **缓存系统**：Redis 7+ （会话状态和任务队列）
- **向量数据库**：Qdrant 1.7+ （轻量级，本地部署）
- **图处理**：NetworkX 3.0+ （DAG任务规划）
