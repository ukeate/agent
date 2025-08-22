# Tech Stack

这是项目的权威技术选择表，是所有开发工作的唯一可信源。所有开发必须严格使用这些确切的版本。

| Category | Technology | Version | Purpose | Rationale | Upgrade Note |
|----------|------------|---------|---------|-----------|--------------|
| Frontend Language | TypeScript | 5.3+ | 静态类型检查和开发体验 | 提供类型安全，减少运行时错误，提升代码质量 | 保持最新 |
| Frontend Framework | React | 18.2+ | 用户界面构建 | 成熟生态系统，组件化开发，优秀的AI工具集成支持 | 保持最新 |
| UI Component Library | Ant Design | 5.12+ | 企业级UI组件库 | 丰富组件集，专业外观，减少开发时间 | 保持最新 |
| State Management | Zustand | 4.4+ | 轻量级状态管理 | 简单API，TypeScript友好，适合中等复杂度应用 | 保持最新 |
| Backend Language | Python | 3.11+ | 后端开发语言 | AI生态系统最佳支持，丰富的ML/AI库 | 保持最新 |
| Backend Framework | FastAPI | 0.116.1+ | 高性能异步API框架 | 自动文档生成，异步支持，现代Python特性 | 2025升级 |
| API Style | RESTful + WebSocket | HTTP/1.1, WS | API通信协议 | RESTful用于标准操作，WebSocket用于实时AI交互 | 保持现有 |
| Database | PostgreSQL | 15+ | 主数据库 | 强ACID支持，JSON字段，丰富扩展生态 | 保持现有 |
| Vector Database | Qdrant | 1.7+ | 向量存储和检索 | 高性能向量搜索，BM42混合搜索，Python原生支持 | **BM42混合搜索** |
| Vector Extension | pgvector | **0.8.0** | PostgreSQL向量扩展 | 迭代索引扫描，HNSW优化，向量量化压缩 | **🆕 关键升级** |
| Cache | Redis | 7.2+ | 缓存和会话存储 | 高性能内存存储，丰富数据结构，AI场景优化 | 保持现有 |
| File Storage | 本地文件系统 | N/A | 文档和模型存储 | 学习阶段简化部署，后期可扩展到对象存储 | 保持现有 |
| Authentication | FastAPI-Users | 12.1+ | 用户认证和授权 | 与FastAPI原生集成，JWT支持，灵活用户管理 | 保持现有 |
| AI Orchestration | LangGraph | **0.6.5** | 多智能体工作流编排 | **Context API v0.6，Durability控制，Node缓存** | **🆕 关键升级** |
| Multi-Agent System | AutoGen | **0.4.2b1** | 智能体群组对话 | **Actor Model，异步事件驱动，内置Observability** | **🆕 重大架构升级** |
| Tool Protocol | MCP | 1.0+ | 标准化工具集成 | 工具生态系统标准，支持第三方扩展 | 保持现有 |
| Task Planning | NetworkX | 3.2+ | DAG任务规划 | 图算法库，任务依赖管理，可视化支持 | 保持现有 |
| LLM Provider | OpenAI API | v1 | 大语言模型服务 | GPT-4o-mini模型，经济高效，快速响应 | 保持现有 |
| **多模态LLM** | **Claude 4 API** | **v1** | **多模态AI处理** | **图像、文档、视频理解，多模态RAG增强** | **🆕 新增组件** |
| **多模态LLM** | **GPT-4o API** | **v1** | **视觉理解能力** | **图像识别、OCR、视觉问答，补充Claude 4** | **🆕 新增组件** |
| Frontend Testing | Vitest + RTL | 1.0+, 14.1+ | 单元和集成测试 | 快速测试运行，现代测试体验 | 保持现有 |
| Backend Testing | pytest | 7.4+ | Python测试框架 | 功能强大，插件丰富，异步测试支持 | 保持现有 |
| E2E Testing | Playwright | 1.40+ | 端到端测试 | 跨浏览器支持，AI场景测试友好 | 保持现有 |
| Build Tool | Vite | 5.0+ | 前端构建工具 | 快速热重载，现代ES模块支持 | 保持现有 |
| Bundler | Vite (内置) | 5.0+ | 代码打包 | 与Vite集成，优化的生产构建 | 保持现有 |
| Package Manager | npm | 10.2+ | 依赖管理 | Monorepo workspaces支持，生态系统兼容性 | 保持现有 |
| Python Package Manager | uv | 0.4+ | Python依赖管理 | 极速Python包管理，替代pip和virtualenv | 保持现有 |
| **AI Security Framework** | **AI TRiSM** | **1.0+** | **AI安全管理** | **信任、风险、安全管理，对抗攻击防护，Prompt注入检测** | **🆕 企业级安全** |
| **Observability** | **OpenTelemetry** | **1.25+** | **AI可观测性** | **分布式追踪，AI Agent语义约定，性能监控** | **🆕 完整集成** |
| Containerization | Docker | 24.0+ | 应用容器化 | 环境一致性，便于部署和扩展 | 保持现有 |
| IaC Tool | Docker Compose | 2.23+ | 基础设施即代码 | 本地开发环境管理，服务编排 | 保持现有 |
| CI/CD | GitHub Actions | N/A | 持续集成部署 | 与GitHub集成，丰富的Action生态 | 保持现有 |
| Monitoring | OpenTelemetry + Prometheus | 1.25+ | 系统监控 | 全链路追踪，AI操作监控，企业级可观测性 | 升级集成 |
| Logging | Python logging + Pino | 内置, 8.17+ | 日志管理 | 结构化日志，JSON格式，便于分析 | 保持现有 |
| CSS Framework | Tailwind CSS | 3.3+ | CSS工具类框架 | 快速样式开发，与Ant Design互补 | 保持现有 |
| **模型量化** | **ONNX Runtime** | **1.16+** | **模型优化和压缩** | **模型量化，推理加速，边缘部署支持** | **🆕 边缘AI准备** |
| **推理框架** | **FastEmbed** | **0.3+** | **嵌入推理引擎** | **BM42混合搜索推理，高性能向量生成** | **🆕 搜索优化** |
| **强化学习框架** | **Ray/Optuna** | **2.8+/3.4+** | **RL个性化和优化** | **多臂老虎机，Q-Learning，超参数优化** | **🆕 个性化学习** |
| **语音处理引擎** | **Whisper/Azure Speech** | **v3/最新** | **实时语音交互** | **ASR，TTS，语音情感识别，VAD** | **🆕 语音AI** |
| **知识图谱数据库** | **Neo4j/ArangoDB** | **5.0+/3.10+** | **动态知识图谱** | **实体关系存储，图谱推理，GraphRAG** | **🆕 结构化知识** |
| **模型微调平台** | **LoRA/QLoRA** | **最新** | **模型定制优化** | **高效微调，模型压缩，量化技术** | **🆕 模型自主化** |
| **分布式协调** | **etcd/Consul** | **3.5+/1.17+** | **智能体网络** | **服务发现，分布式共识，集群管理** | **🆕 分布式架构** |
| **情感计算引擎** | **情感AI模型** | **定制** | **情感智能系统** | **多模态情感识别，共情响应，情感记忆** | **🆕 情感交互** |
