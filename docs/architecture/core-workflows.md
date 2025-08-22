# Core Workflows

以下是系统核心工作流的序列图，展示关键用户旅程中的组件交互，包括2025年架构升级的新特性：

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant UI as 🌐 React UI
    participant Gateway as 🚀 API Gateway
    participant Auth as 🔐 Auth Service
    participant LG as 🧠 LangGraph 0.6
    participant AG as 👥 AutoGen 0.4
    participant MCP as 🔧 MCP Tools
    participant RAG as 📚 RAG Engine
    participant MultiAI as 🎭 Multi-modal AI
    participant Security as 🔒 AI TRiSM
    participant Monitor as 📊 OpenTelemetry
    participant OpenAI as 🤖 OpenAI API
    participant Claude as 🤖 Claude 4 API
    participant DB as 🗄️ PostgreSQL
    participant Redis as ⚡ Redis
    participant Qdrant as 🔍 Qdrant BM42

    Note over User, Qdrant: 1. 用户发起多智能体协作任务 (2025升级版)

    User->>UI: 输入复杂任务请求 (支持多模态)
    UI->>Gateway: POST /conversations
    Gateway->>Auth: 验证JWT令牌
    Auth-->>Gateway: 认证成功
    
    Gateway->>Security: AI安全检查 (Prompt注入检测)
    Security-->>Gateway: 安全验证通过
    
    Gateway->>DB: 创建会话记录
    DB-->>Gateway: 返回会话ID
    
    Gateway->>Monitor: 开始分布式追踪
    Monitor->>LG: 初始化Context API工作流
    
    LG->>AG: 创建异步智能体网络 (Actor Model)
    AG->>OpenAI: 初始化主要智能体
    AG->>Claude: 初始化多模态智能体
    Claude-->>AG: 返回多模态智能体
    OpenAI-->>AG: 返回标准智能体实例
    
    LG->>Redis: 启用Node级缓存
    LG->>DB: 保存工作流检查点 (Durability控制)
    Gateway-->>UI: 返回会话创建成功
    UI-->>User: 显示增强会话界面

    Note over User, Qdrant: 2. 智能体异步协作执行任务 (事件驱动架构)

    User->>UI: 发送任务消息 (文本/图像/文档)
    UI->>Gateway: POST /conversations/{id}/messages
    Gateway->>Redis: 检查限流和缓存
    Redis-->>Gateway: 返回缓存状态
    
    Gateway->>MultiAI: 多模态内容分析
    MultiAI->>Claude: 图像/文档理解
    MultiAI->>OpenAI: 文本分析
    Claude-->>MultiAI: 多模态理解结果
    OpenAI-->>MultiAI: 文本分析结果
    MultiAI-->>Gateway: 综合分析结果
    
    Gateway->>LG: 处理增强消息 (Context API)
    LG->>RAG: BM42混合搜索知识检索
    RAG->>Qdrant: 执行稀疏+密集向量搜索
    RAG->>DB: pgvector 0.8优化查询
    Qdrant-->>RAG: 返回混合搜索结果
    DB-->>RAG: 返回向量搜索结果
    RAG-->>LG: 知识增强上下文
    
    LG->>AG: 异步事件分发任务
    
    par 并行异步智能体处理
        AG->>OpenAI: 专家智能体A处理
        and
        AG->>Claude: 专家智能体B处理 (多模态)
        and
        AG->>MCP: 工具调用智能体C
    end
    
    OpenAI-->>AG: 智能体A结果
    Claude-->>AG: 智能体B结果 (多模态)
    MCP-->>AG: 工具调用结果
    
    AG->>Monitor: 记录智能体性能指标
    AG->>Security: AI决策安全验证
    Security-->>AG: 安全检查通过
    
    AG->>LG: 聚合异步结果
    LG->>Redis: 更新Node缓存
    LG->>DB: 检查点保存 (Durability)
    
    LG->>Gateway: 返回最终增强结果
    Gateway->>Monitor: 记录完整请求追踪
    Gateway->>DB: 保存对话记录
    Gateway-->>UI: 推送实时更新
    UI-->>User: 显示多模态增强结果

    Note over User, Qdrant: 3. AI可观测性和安全监控 (持续进行)

    Monitor->>Monitor: 分析智能体性能模式
    Monitor->>Gateway: 性能异常告警
    Security->>Security: 威胁检测和防护
    Security->>Gateway: 安全事件通知
```

## 2025年架构升级的关键工作流改进:

### 1. **Context API工作流** (LangGraph 0.6.5)
- 类型安全的上下文传递，替代传统config模式
- Durability控制实现细粒度状态管理
- Node缓存优化开发迭代和运行时性能

### 2. **异步事件驱动架构** (AutoGen 0.4.2b1)
- Actor Model实现真正的异步智能体通信
- 事件驱动系统支持复杂协作模式
- 并行智能体处理，显著提升处理能力

### 3. **BM42混合搜索工作流** (Qdrant + pgvector 0.8)
- 稀疏+密集向量的混合检索策略
- FastEmbed推理引擎优化向量生成
- pgvector 0.8的迭代索引扫描优化

### 4. **多模态AI集成工作流**
- Claude 4和GPT-4o的多模态能力整合
- 文本、图像、文档的统一处理pipeline
- 多模态RAG增强的智能检索

### 5. **AI安全和监控工作流**
- AI TRiSM安全框架的实时威胁检测
- OpenTelemetry的完整分布式追踪
- 智能体行为分析和性能优化
