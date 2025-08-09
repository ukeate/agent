# Tech Stack

这是项目的权威技术选择表，是所有开发工作的唯一可信源。所有开发必须严格使用这些确切的版本。

| Category | Technology | Version | Purpose | Rationale |
|----------|------------|---------|---------|-----------|
| Frontend Language | TypeScript | 5.3+ | 静态类型检查和开发体验 | 提供类型安全，减少运行时错误，提升代码质量 |
| Frontend Framework | React | 18.2+ | 用户界面构建 | 成熟生态系统，组件化开发，优秀的AI工具集成支持 |
| UI Component Library | Ant Design | 5.12+ | 企业级UI组件库 | 丰富组件集，专业外观，减少开发时间 |
| State Management | Zustand | 4.4+ | 轻量级状态管理 | 简单API，TypeScript友好，适合中等复杂度应用 |
| Backend Language | Python | 3.11+ | 后端开发语言 | AI生态系统最佳支持，丰富的ML/AI库 |
| Backend Framework | FastAPI | 0.116.1+ | 高性能异步API框架 | 自动文档生成，异步支持，现代Python特性，CLI工具支持 |
| API Style | RESTful + WebSocket | HTTP/1.1, WS | API通信协议 | RESTful用于标准操作，WebSocket用于实时AI交互 |
| Database | PostgreSQL | 15+ | 主数据库 | 强ACID支持，JSON字段，丰富扩展生态 |
| Vector Database | Qdrant | 1.7+ | 向量存储和检索 | 高性能向量搜索，Python原生支持，易于集成 |
| Cache | Redis | 7.2+ | 缓存和会话存储 | 高性能内存存储，丰富数据结构，AI场景优化 |
| File Storage | 本地文件系统 | N/A | 文档和模型存储 | 学习阶段简化部署，后期可扩展到对象存储 |
| Authentication | FastAPI-Users | 12.1+ | 用户认证和授权 | 与FastAPI原生集成，JWT支持，灵活用户管理 |
| AI Orchestration | LangGraph | 0.2.76+ | 多智能体工作流编排 | 状态管理，条件分支，可视化调试，增强的状态检查点 |
| Multi-Agent System | AutoGen | 0.2.18+ | 智能体群组对话 | 成熟的多智能体框架，丰富的对话模式 |
| Tool Protocol | MCP | 1.0+ | 标准化工具集成 | 工具生态系统标准，支持第三方扩展 |
| Task Planning | NetworkX | 3.2+ | DAG任务规划 | 图算法库，任务依赖管理，可视化支持 |
| LLM Provider | OpenAI API | v1 | 大语言模型服务 | GPT-4o-mini高效推理，成本优化 |
| Frontend Testing | Vitest + RTL | 1.0+, 14.1+ | 单元和集成测试 | 快速测试运行，现代测试体验 |
| Backend Testing | pytest | 7.4+ | Python测试框架 | 功能强大，插件丰富，异步测试支持 |
| E2E Testing | Playwright | 1.40+ | 端到端测试 | 跨浏览器支持，AI场景测试友好 |
| Build Tool | Vite | 5.0+ | 前端构建工具 | 快速热重载，现代ES模块支持 |
| Bundler | Vite (内置) | 5.0+ | 代码打包 | 与Vite集成，优化的生产构建 |
| Package Manager | npm | 10.2+ | 依赖管理 | Monorepo workspaces支持，生态系统兼容性 |
| Containerization | Docker | 24.0+ | 应用容器化 | 环境一致性，便于部署和扩展 |
| IaC Tool | Docker Compose | 2.23+ | 基础设施即代码 | 本地开发环境管理，服务编排 |
| CI/CD | GitHub Actions | N/A | 持续集成部署 | 与GitHub集成，丰富的Action生态 |
| Monitoring | 开发阶段暂无 | N/A | 系统监控 | 后期扩展时添加APM解决方案 |
| Logging | Python logging + Pino | 内置, 8.17+ | 日志管理 | 结构化日志，JSON格式，便于分析 |
| CSS Framework | Tailwind CSS | 3.3+ | CSS工具类框架 | 快速样式开发，与Ant Design互补 |
