# Components

基于架构模式、技术栈和数据模型，我定义了以下跨全栈的逻辑组件，实现清晰的边界和接口：

## API Gateway

**Responsibility:** 作为系统统一入口，处理认证、路由、限流和跨域请求

**Key Interfaces:**
- HTTP RESTful API endpoints
- WebSocket 连接管理
- JWT 认证中间件
- CORS 处理和安全策略

**Dependencies:** FastAPI-Users (认证), Redis (限流缓存), 日志系统

**Technology Stack:** FastAPI + Uvicorn，中间件栈，JWT认证，速率限制器

## LangGraph Orchestrator

**Responsibility:** 多智能体工作流编排，状态管理，条件分支控制和执行监控

**Key Interfaces:**
- 工作流定义和执行API
- 状态检查点管理
- 智能体间消息传递
- 条件路由和分支逻辑

**Dependencies:** AutoGen Agent Pool, MCP Tool Registry, PostgreSQL (状态持久化)

**Technology Stack:** LangGraph 0.0.69+, Python asyncio, 状态管理机制

## AutoGen Agent Pool

**Responsibility:** 管理专业化AI智能体实例，提供群组对话和智能体间协作能力

**Key Interfaces:**
- 智能体创建和配置管理
- 群组对话API
- 智能体状态监控
- 角色分配和能力路由

**Dependencies:** OpenAI API, MCP Tools, LangGraph Orchestrator

**Technology Stack:** AutoGen 0.2.18+, OpenAI API集成, 智能体配置管理

## RAG Knowledge Engine

**Responsibility:** 智能知识检索系统，支持语义搜索、上下文增强和答案生成

**Key Interfaces:**
- 知识条目向量化和存储
- 语义相似度搜索API
- RAG增强查询接口
- 知识图谱关系分析

**Dependencies:** Qdrant Vector DB, OpenAI Embeddings, Knowledge Repository

**Technology Stack:** Qdrant 1.7+, sentence-transformers, 向量检索算法

## React Frontend Shell

**Responsibility:** 前端应用框架，路由管理，状态协调，组件渲染

**Key Interfaces:**
- 页面路由系统
- 全局状态管理
- API客户端集成
- 实时通信WebSocket

**Dependencies:** API Gateway, 各功能组件

**Technology Stack:** React 18.2+, React Router, Zustand, WebSocket客户端
