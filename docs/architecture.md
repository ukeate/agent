# Personal AI Agent System Fullstack Architecture Document

## Introduction

这份文档定义了个人AI智能体系统的完整全栈架构，包括后端系统、前端实现及其集成方案。它是AI驱动开发的唯一可信源，确保整个技术栈的一致性。

该统一方法结合了传统上分离的后端和前端架构文档，为现代全栈应用简化了开发流程，特别是在这些关注点日益交织的情况下。

### Starter Template or Existing Project

基于PRD文档分析，这是一个全新的greenfield项目，专注于构建AI学习平台。项目需要集成多种前沿AI技术：
- LangGraph多智能体工作流编排
- AutoGen多智能体会话系统  
- MCP协议标准化工具集成
- Agentic RAG系统（基于Qdrant）
- DAG任务规划引擎（基于NetworkX）

**决策**: 不使用现有starter模板，因为需要深度自定义AI架构集成。项目将从零开始构建，以确保对每个技术组件的完全掌控和学习价值最大化。

### Change Log
| Date | Version | Description | Author |
|------|---------|-------------|---------|
| 2025-01-01 | 1.0 | Initial fullstack architecture creation | Architect (Winston) |

## High Level Architecture

### Technical Summary

本系统adopts微服务启发的模块化单体架构，部署在Docker容器化环境中。前端使用React + TypeScript构建现代化SPA应用，后端基于FastAPI提供高性能异步API服务。核心集成点包括LangGraph工作流编排器作为多智能体协调中心，AutoGen提供群组对话能力，以及MCP协议实现标准化工具生态系统。基础设施采用PostgreSQL作为主数据库，Redis提供缓存和会话管理，Qdrant向量数据库支持RAG语义检索。该架构实现了PRD中定义的AI-First开发模式学习目标，同时保持了生产级的可扩展性和可维护性。

### Platform and Infrastructure Choice

**Platform:** Docker + 自托管（初期），AWS（扩展期）
**Key Services:** PostgreSQL, Redis, Qdrant, FastAPI, React, LangGraph, AutoGen
**Deployment Host and Regions:** 本地开发环境，后期考虑AWS us-east-1

### Repository Structure

**Structure:** Monorepo
**Monorepo Tool:** npm workspaces（轻量级，学习友好）
**Package Organization:** apps/（应用）+ packages/（共享代码）+ tools/（工具脚本）

### High Level Architecture Diagram

```mermaid
graph TB
    User[👤 User] --> Web[🌐 React Web App]
    Web --> API[🚀 FastAPI Backend]
    API --> LG[🧠 LangGraph Orchestrator]
    API --> AG[👥 AutoGen Agents]
    API --> RAG[📚 RAG System]
    
    LG --> MCP[🔧 MCP Servers]
    AG --> LG
    RAG --> Qdrant[(🔍 Qdrant Vector DB)]
    
    API --> Cache[(⚡ Redis Cache)]
    API --> DB[(🗄️ PostgreSQL)]
    
    MCP --> FS[📁 File System]
    MCP --> Git[🔄 Git Operations]
    MCP --> Search[🔍 Web Search]
    MCP --> CMD[💻 System Commands]
```

### Architectural Patterns

- **Event-Driven Architecture:** 智能体间通过事件总线进行异步通信 - _Rationale:_ 支持复杂的多智能体协作和状态管理
- **Repository Pattern:** 抽象数据访问逻辑，支持测试和数据库切换 - _Rationale:_ 提高代码可测试性和灵活性
- **Plugin Architecture:** MCP协议提供可扩展的工具生态系统 - _Rationale:_ 实现标准化工具集成，支持第三方扩展
- **Hexagonal Architecture:** 将业务逻辑与外部依赖解耦 - _Rationale:_ 提高系统的可测试性和适应性
- **CQRS Pattern:** 分离命令和查询操作，优化性能 - _Rationale:_ 支持复杂的AI推理和数据检索场景
- **Saga Pattern:** 管理跨智能体的长运行事务 - _Rationale:_ 确保多步骤AI任务的一致性和可恢复性

## Tech Stack

这是项目的权威技术选择表，是所有开发工作的唯一可信源。所有开发必须严格使用这些确切的版本。

| Category | Technology | Version | Purpose | Rationale |
|----------|------------|---------|---------|-----------|
| Frontend Language | TypeScript | 5.3+ | 静态类型检查和开发体验 | 提供类型安全，减少运行时错误，提升代码质量 |
| Frontend Framework | React | 18.2+ | 用户界面构建 | 成熟生态系统，组件化开发，优秀的AI工具集成支持 |
| UI Component Library | Ant Design | 5.12+ | 企业级UI组件库 | 丰富组件集，专业外观，减少开发时间 |
| State Management | Zustand | 4.4+ | 轻量级状态管理 | 简单API，TypeScript友好，适合中等复杂度应用 |
| Backend Language | Python | 3.11+ | 后端开发语言 | AI生态系统最佳支持，丰富的ML/AI库 |
| Backend Framework | FastAPI | 0.104+ | 高性能异步API框架 | 自动文档生成，异步支持，现代Python特性 |
| API Style | RESTful + WebSocket | HTTP/1.1, WS | API通信协议 | RESTful用于标准操作，WebSocket用于实时AI交互 |
| Database | PostgreSQL | 15+ | 主数据库 | 强ACID支持，JSON字段，丰富扩展生态 |
| Vector Database | Qdrant | 1.7+ | 向量存储和检索 | 高性能向量搜索，Python原生支持，易于集成 |
| Cache | Redis | 7.2+ | 缓存和会话存储 | 高性能内存存储，丰富数据结构，AI场景优化 |
| File Storage | 本地文件系统 | N/A | 文档和模型存储 | 学习阶段简化部署，后期可扩展到对象存储 |
| Authentication | FastAPI-Users | 12.1+ | 用户认证和授权 | 与FastAPI原生集成，JWT支持，灵活用户管理 |
| AI Orchestration | LangGraph | 0.0.69+ | 多智能体工作流编排 | 状态管理，条件分支，可视化调试 |
| Multi-Agent System | AutoGen | 0.2.18+ | 智能体群组对话 | 成熟的多智能体框架，丰富的对话模式 |
| Tool Protocol | MCP | 1.0+ | 标准化工具集成 | 工具生态系统标准，支持第三方扩展 |
| Task Planning | NetworkX | 3.2+ | DAG任务规划 | 图算法库，任务依赖管理，可视化支持 |
| LLM Provider | OpenAI API | v1 | 大语言模型服务 | GPT-4o-mini模型，经济高效，快速响应 |
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

## Data Models

基于PRD要求和AI系统特性，我定义了以下核心数据模型来支持多智能体协作、任务规划和知识管理：

### Agent

**Purpose:** 表示系统中的AI智能体实例，包括专业化配置和运行状态

**Key Attributes:**
- id: string - 唯一标识符
- name: string - 智能体显示名称
- role: AgentRole - 智能体角色类型（代码专家、架构师、文档专家等）
- status: AgentStatus - 运行状态（活跃、空闲、繁忙、离线）
- capabilities: string[] - 智能体能力列表
- configuration: AgentConfig - 模型配置和工具设置
- created_at: Date - 创建时间
- updated_at: Date - 最后更新时间

#### TypeScript Interface
```typescript
interface Agent {
  id: string;
  name: string;
  role: 'code_expert' | 'architect' | 'doc_expert' | 'supervisor' | 'rag_specialist';
  status: 'active' | 'idle' | 'busy' | 'offline';
  capabilities: string[];
  configuration: {
    model: string;
    temperature: number;
    max_tokens: number;
    tools: string[];
    system_prompt: string;
  };
  created_at: Date;
  updated_at: Date;
}
```

#### Relationships
- 一个Agent可以参与多个Conversation
- 一个Agent可以执行多个Task
- Agent之间通过Message进行交互

### Conversation

**Purpose:** 管理用户与AI系统的对话会话，支持多智能体参与的群组对话

**Key Attributes:**
- id: string - 会话唯一标识
- title: string - 会话标题
- type: ConversationType - 会话类型（单智能体、多智能体、工作流）
- participants: string[] - 参与的智能体ID列表
- status: ConversationStatus - 会话状态
- metadata: Record<string, any> - 扩展元数据
- created_at: Date - 创建时间
- updated_at: Date - 最后活动时间

#### TypeScript Interface
```typescript
interface Conversation {
  id: string;
  title: string;
  type: 'single_agent' | 'multi_agent' | 'workflow' | 'rag_enhanced';
  participants: string[]; // Agent IDs
  status: 'active' | 'paused' | 'completed' | 'archived';
  metadata: {
    user_context?: string;
    task_complexity?: number;
    workflow_type?: string;
  };
  created_at: Date;
  updated_at: Date;
}
```

#### Relationships
- 一个Conversation包含多个Message
- 一个Conversation可以关联多个Task
- 一个Conversation可以触发DAG执行

### Message

**Purpose:** 存储对话中的具体消息内容，支持多模态内容和智能体间通信

**Key Attributes:**
- id: string - 消息唯一标识
- conversation_id: string - 所属会话ID
- sender_type: SenderType - 发送者类型（用户、智能体、系统）
- sender_id: string - 发送者标识
- content: MessageContent - 消息内容（支持文本、代码、文件等）
- message_type: MessageType - 消息类型
- metadata: MessageMetadata - 消息元数据
- created_at: Date - 发送时间

#### TypeScript Interface
```typescript
interface Message {
  id: string;
  conversation_id: string;
  sender_type: 'user' | 'agent' | 'system';
  sender_id: string;
  content: {
    text?: string;
    code?: {
      language: string;
      content: string;
    };
    files?: {
      name: string;
      path: string;
      type: string;
    }[];
    tool_calls?: {
      tool: string;
      arguments: Record<string, any>;
      result?: any;
    }[];
  };
  message_type: 'chat' | 'command' | 'tool_call' | 'system_notification';
  metadata: {
    tokens_used?: number;
    processing_time?: number;
    confidence_score?: number;
  };
  created_at: Date;
}
```

#### Relationships
- 属于一个Conversation
- 可能触发Task创建
- 可能包含KnowledgeItem引用

### Task

**Purpose:** 表示系统中的可执行任务，支持DAG依赖关系和状态跟踪

**Key Attributes:**
- id: string - 任务唯一标识
- name: string - 任务名称
- description: string - 任务描述
- type: TaskType - 任务类型
- assigned_agent_id: string - 分配的智能体ID
- dependencies: string[] - 依赖任务ID列表
- status: TaskStatus - 执行状态
- priority: TaskPriority - 优先级
- input_data: Record<string, any> - 输入数据
- output_data: Record<string, any> - 输出结果
- execution_metadata: ExecutionMetadata - 执行元数据
- created_at: Date - 创建时间
- started_at: Date - 开始执行时间
- completed_at: Date - 完成时间

#### TypeScript Interface
```typescript
interface Task {
  id: string;
  name: string;
  description: string;
  type: 'code_generation' | 'code_review' | 'documentation' | 'analysis' | 'planning';
  dag_execution_id?: string;
  assigned_agent_id: string;
  dependencies: string[];
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  priority: 'low' | 'medium' | 'high' | 'urgent';
  input_data: Record<string, any>;
  output_data: Record<string, any>;
  execution_metadata: {
    start_time?: Date;
    end_time?: Date;
    error_message?: string;
    retry_count: number;
    resource_usage?: {
      tokens: number;
      api_calls: number;
    };
  };
  created_at: Date;
  started_at?: Date;
  completed_at?: Date;
}
```

#### Relationships
- 属于一个DAGExecution
- 被分配给一个Agent
- 可能由Message触发创建
- 可以生成KnowledgeItem

### DAGExecution

**Purpose:** 管理复杂任务的DAG执行实例，跟踪整个工作流的执行状态

**Key Attributes:**
- id: string - DAG执行唯一标识
- name: string - 执行名称
- conversation_id: string - 关联的会话ID
- graph_definition: DAGDefinition - DAG图结构定义
- status: DAGStatus - 整体执行状态
- current_stage: string - 当前执行阶段
- progress: DAGProgress - 执行进度信息
- metadata: DAGMetadata - 执行元数据
- created_at: Date - 创建时间
- started_at: Date - 开始执行时间
- completed_at: Date - 完成时间

#### TypeScript Interface
```typescript
interface DAGExecution {
  id: string;
  name: string;
  conversation_id: string;
  graph_definition: {
    nodes: {
      id: string;
      type: string;
      config: Record<string, any>;
    }[];
    edges: {
      source: string;
      target: string;
      condition?: string;
    }[];
  };
  status: 'created' | 'running' | 'completed' | 'failed' | 'cancelled';
  current_stage: string;
  progress: {
    total_tasks: number;
    completed_tasks: number;
    failed_tasks: number;
    success_rate: number;
  };
  metadata: {
    estimated_duration?: number;
    actual_duration?: number;
    resource_requirements?: Record<string, any>;
  };
  created_at: Date;
  started_at?: Date;
  completed_at?: Date;
}
```

#### Relationships
- 关联一个Conversation
- 包含多个Task
- 由Supervisor智能体管理

### KnowledgeItem

**Purpose:** 存储RAG系统中的知识条目，支持向量检索和语义搜索

**Key Attributes:**
- id: string - 知识条目唯一标识
- title: string - 标题
- content: string - 文本内容
- content_type: ContentType - 内容类型
- source: KnowledgeSource - 来源信息
- embedding_vector: number[] - 向量表示
- metadata: KnowledgeMetadata - 扩展元数据
- tags: string[] - 标签列表
- created_at: Date - 创建时间
- updated_at: Date - 更新时间

#### TypeScript Interface
```typescript
interface KnowledgeItem {
  id: string;
  title: string;
  content: string;
  content_type: 'code' | 'documentation' | 'conversation' | 'web_content' | 'file';
  source: {
    type: 'upload' | 'web_scrape' | 'conversation' | 'generated';
    url?: string;
    file_path?: string;
    conversation_id?: string;
  };
  embedding_vector: number[];
  metadata: {
    file_size?: number;
    language?: string;
    author?: string;
    version?: string;
    relevance_score?: number;
  };
  tags: string[];
  created_at: Date;
  updated_at: Date;
}
```

#### Relationships
- 可以被Message引用
- 用于RAG检索增强
- 可以由Task生成

## API Specification

基于选择的RESTful + WebSocket API风格，以下是完整的OpenAPI 3.0规范：

```yaml
openapi: 3.0.0
info:
  title: Personal AI Agent System API
  version: 1.0.0
  description: AI智能体系统的RESTful API，支持多智能体协作、任务规划和知识管理
  contact:
    name: API Support
    email: support@ai-agent-system.com
servers:
  - url: http://localhost:8000/api/v1
    description: 本地开发环境
  - url: ws://localhost:8000/ws
    description: WebSocket连接

paths:
  # 智能体管理
  /agents:
    get:
      summary: 获取智能体列表
      tags: [Agents]
      parameters:
        - name: status
          in: query
          schema:
            type: string
            enum: [active, idle, busy, offline]
        - name: role
          in: query
          schema:
            type: string
            enum: [code_expert, architect, doc_expert, supervisor, rag_specialist]
      responses:
        '200':
          description: 成功返回智能体列表
          content:
            application/json:
              schema:
                type: object
                properties:
                  agents:
                    type: array
                    items:
                      $ref: '#/components/schemas/Agent'
                  total:
                    type: integer
    post:
      summary: 创建新智能体
      tags: [Agents]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateAgentRequest'
      responses:
        '201':
          description: 智能体创建成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Agent'

  /agents/{agent_id}:
    get:
      summary: 获取智能体详情
      tags: [Agents]
      parameters:
        - name: agent_id
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: 成功返回智能体详情
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Agent'
    
    put:
      summary: 更新智能体配置
      tags: [Agents]
      parameters:
        - name: agent_id
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/UpdateAgentRequest'
      responses:
        '200':
          description: 智能体更新成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Agent'

  # 对话管理
  /conversations:
    get:
      summary: 获取对话列表
      tags: [Conversations]
      parameters:
        - name: type
          in: query
          schema:
            $ref: '#/components/schemas/ConversationType'
        - name: status
          in: query
          schema:
            $ref: '#/components/schemas/ConversationStatus'
        - name: limit
          in: query
          schema:
            type: integer
            default: 20
      responses:
        '200':
          description: 成功返回对话列表
          content:
            application/json:
              schema:
                type: object
                properties:
                  conversations:
                    type: array
                    items:
                      $ref: '#/components/schemas/Conversation'
    
    post:
      summary: 创建新对话
      tags: [Conversations]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CreateConversationRequest'
      responses:
        '201':
          description: 对话创建成功
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Conversation'

  # RAG查询
  /rag/query:
    post:
      summary: RAG增强查询
      tags: [RAG]
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              properties:
                query:
                  type: string
                  description: 查询文本
                context:
                  type: string
                  description: 查询上下文
                max_results:
                  type: integer
                  default: 5
              required: [query]
      responses:
        '200':
          description: 成功返回RAG查询结果
          content:
            application/json:
              schema:
                type: object
                properties:
                  answer:
                    type: string
                  sources:
                    type: array
                    items:
                      $ref: '#/components/schemas/KnowledgeItem'
                  confidence:
                    type: number
                    format: float

components:
  schemas:
    Agent:
      type: object
      properties:
        id:
          type: string
        name:
          type: string
        role:
          $ref: '#/components/schemas/AgentRole'
        status:
          $ref: '#/components/schemas/AgentStatus'
        capabilities:
          type: array
          items:
            type: string
        configuration:
          type: object
          properties:
            model:
              type: string
            temperature:
              type: number
            max_tokens:
              type: integer
            tools:
              type: array
              items:
                type: string
            system_prompt:
              type: string
        created_at:
          type: string
          format: date-time
        updated_at:
          type: string
          format: date-time

    AgentRole:
      type: string
      enum: [code_expert, architect, doc_expert, supervisor, rag_specialist]

    AgentStatus:
      type: string
      enum: [active, idle, busy, offline]

  securitySchemes:
    BearerAuth:
      type: http
      scheme: bearer
      bearerFormat: JWT

security:
  - BearerAuth: []
```

## Components

基于架构模式、技术栈和数据模型，我定义了以下跨全栈的逻辑组件，实现清晰的边界和接口：

### API Gateway

**Responsibility:** 作为系统统一入口，处理认证、路由、限流和跨域请求

**Key Interfaces:**
- HTTP RESTful API endpoints
- WebSocket 连接管理
- JWT 认证中间件
- CORS 处理和安全策略

**Dependencies:** FastAPI-Users (认证), Redis (限流缓存), 日志系统

**Technology Stack:** FastAPI + Uvicorn，中间件栈，JWT认证，速率限制器

### LangGraph Orchestrator

**Responsibility:** 多智能体工作流编排，状态管理，条件分支控制和执行监控

**Key Interfaces:**
- 工作流定义和执行API
- 状态检查点管理
- 智能体间消息传递
- 条件路由和分支逻辑

**Dependencies:** AutoGen Agent Pool, MCP Tool Registry, PostgreSQL (状态持久化)

**Technology Stack:** LangGraph 0.0.69+, Python asyncio, 状态管理机制

### AutoGen Agent Pool

**Responsibility:** 管理专业化AI智能体实例，提供群组对话和智能体间协作能力

**Key Interfaces:**
- 智能体创建和配置管理
- 群组对话API
- 智能体状态监控
- 角色分配和能力路由

**Dependencies:** OpenAI API, MCP Tools, LangGraph Orchestrator

**Technology Stack:** AutoGen 0.2.18+, OpenAI API集成, 智能体配置管理

### RAG Knowledge Engine

**Responsibility:** 智能知识检索系统，支持语义搜索、上下文增强和答案生成

**Key Interfaces:**
- 知识条目向量化和存储
- 语义相似度搜索API
- RAG增强查询接口
- 知识图谱关系分析

**Dependencies:** Qdrant Vector DB, OpenAI Embeddings, Knowledge Repository

**Technology Stack:** Qdrant 1.7+, sentence-transformers, 向量检索算法

### React Frontend Shell

**Responsibility:** 前端应用框架，路由管理，状态协调，组件渲染

**Key Interfaces:**
- 页面路由系统
- 全局状态管理
- API客户端集成
- 实时通信WebSocket

**Dependencies:** API Gateway, 各功能组件

**Technology Stack:** React 18.2+, React Router, Zustand, WebSocket客户端

## External APIs

基于PRD要求和组件设计，项目需要集成以下外部服务来实现完整的AI功能：

### OpenAI API

- **Purpose:** 提供核心语言模型推理能力，支持多智能体对话和代码生成
- **Documentation:** https://platform.openai.com/docs/api-reference
- **Base URL(s):** https://api.openai.com/v1
- **Authentication:** API Key (Bearer Token)
- **Rate Limits:** 根据订阅计划，通常为每分钟50-1000请求

**Key Endpoints Used:**
- `POST /messages` - 创建对话完成，支持工具调用和系统提示
- `POST /messages/stream` - 流式响应，实时生成内容

**Integration Notes:** 需要实现重试机制和错误处理，支持工具调用格式转换，管理上下文长度限制

### OpenAI Embeddings API

- **Purpose:** 生成文本向量表示，支持RAG系统的语义检索功能
- **Documentation:** https://platform.openai.com/docs/api-reference/embeddings
- **Base URL(s):** https://api.openai.com/v1
- **Authentication:** API Key (Bearer Token)
- **Rate Limits:** 每分钟3000请求，每分钟1M tokens

**Key Endpoints Used:**
- `POST /embeddings` - 生成文本嵌入向量，使用text-embedding-3-small模型

**Integration Notes:** 批量处理优化，缓存常用嵌入向量，处理API限制和错误重试

## Core Workflows

以下是系统核心工作流的序列图，展示关键用户旅程中的组件交互，包括外部API集成和错误处理路径：

```mermaid
sequenceDiagram
    participant User as 👤 User
    participant UI as 🌐 React UI
    participant Gateway as 🚀 API Gateway
    participant Auth as 🔐 Auth Service
    participant LG as 🧠 LangGraph
    participant AG as 👥 AutoGen
    participant MCP as 🔧 MCP Tools
    participant RAG as 📚 RAG Engine
    participant OpenAI as 🤖 OpenAI API
    participant DB as 🗄️ PostgreSQL
    participant Redis as ⚡ Redis

    Note over User, Redis: 1. 用户发起多智能体协作任务

    User->>UI: 输入复杂任务请求
    UI->>Gateway: POST /conversations
    Gateway->>Auth: 验证JWT令牌
    Auth-->>Gateway: 认证成功
    
    Gateway->>DB: 创建会话记录
    DB-->>Gateway: 返回会话ID
    
    Gateway->>LG: 初始化工作流
    LG->>AG: 创建智能体群组
    AG->>OpenAI: 初始化角色配置
    OpenAI-->>AG: 返回智能体实例
    
    LG->>DB: 保存工作流状态
    Gateway-->>UI: 返回会话创建成功
    UI-->>User: 显示会话界面

    Note over User, Redis: 2. 智能体协作执行任务

    User->>UI: 发送任务消息
    UI->>Gateway: POST /conversations/{id}/messages
    Gateway->>Redis: 检查限流
    Redis-->>Gateway: 通过检查
    
    Gateway->>LG: 处理用户消息
    LG->>AG: 分析任务复杂度
    AG->>OpenAI: 任务分解请求
    OpenAI-->>AG: 返回分解建议
    
    AG->>LG: 提出执行计划
    LG->>DB: 创建DAG执行计划
    
    loop 多智能体协作
        LG->>AG: 分配子任务给专家
        AG->>OpenAI: 执行专业任务
        OpenAI-->>AG: 返回执行结果
        AG->>MCP: 调用必要工具
        MCP-->>AG: 返回工具结果
        AG->>LG: 报告任务进度
        LG->>DB: 更新执行状态
    end
    
    LG->>Gateway: 返回最终结果
    Gateway->>DB: 保存对话记录
    Gateway-->>UI: 推送实时更新
    UI-->>User: 显示执行结果
```

## Database Schema

基于PostgreSQL数据库和已定义的数据模型，以下是完整的数据库架构定义：

```sql
-- 启用必要的扩展
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- 用户表
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 智能体表
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) NOT NULL,
    role VARCHAR(50) NOT NULL CHECK (role IN ('code_expert', 'architect', 'doc_expert', 'supervisor', 'rag_specialist')),
    status VARCHAR(20) DEFAULT 'idle' CHECK (status IN ('active', 'idle', 'busy', 'offline')),
    capabilities TEXT[] DEFAULT '{}',
    configuration JSONB NOT NULL DEFAULT '{}',
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 对话表
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(255) NOT NULL,
    type VARCHAR(30) NOT NULL CHECK (type IN ('single_agent', 'multi_agent', 'workflow', 'rag_enhanced')),
    participants UUID[] DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'paused', 'completed', 'archived')),
    metadata JSONB DEFAULT '{}',
    created_by UUID REFERENCES users(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 消息表
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    sender_type VARCHAR(10) NOT NULL CHECK (sender_type IN ('user', 'agent', 'system')),
    sender_id VARCHAR(255) NOT NULL,
    content JSONB NOT NULL DEFAULT '{}',
    message_type VARCHAR(30) DEFAULT 'chat' CHECK (message_type IN ('chat', 'command', 'tool_call', 'system_notification')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- DAG执行表
CREATE TABLE dag_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    graph_definition JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'created' CHECK (status IN ('created', 'running', 'completed', 'failed', 'cancelled')),
    current_stage VARCHAR(100),
    progress JSONB DEFAULT '{"total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0, "success_rate": 0}',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- 任务表
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    type VARCHAR(50) NOT NULL CHECK (type IN ('code_generation', 'code_review', 'documentation', 'analysis', 'planning')),
    dag_execution_id UUID REFERENCES dag_executions(id) ON DELETE CASCADE,
    assigned_agent_id UUID NOT NULL REFERENCES agents(id) ON DELETE RESTRICT,
    dependencies UUID[] DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    priority VARCHAR(10) DEFAULT 'medium' CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    input_data JSONB DEFAULT '{}',
    output_data JSONB DEFAULT '{}',
    execution_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE
);

-- 知识库条目表
CREATE TABLE knowledge_items (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    content_type VARCHAR(20) NOT NULL CHECK (content_type IN ('code', 'documentation', 'conversation', 'web_content', 'file')),
    source JSONB NOT NULL DEFAULT '{}',
    metadata JSONB DEFAULT '{}',
    tags TEXT[] DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引优化查询性能
CREATE INDEX idx_agents_role_status ON agents(role, status);
CREATE INDEX idx_conversations_created_by ON conversations(created_by);
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_tasks_dag_execution_id ON tasks(dag_execution_id);
CREATE INDEX idx_knowledge_items_content_type ON knowledge_items(content_type);
CREATE INDEX idx_knowledge_items_tags ON knowledge_items USING GIN(tags);

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为需要的表创建更新时间触发器
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_agents_updated_at BEFORE UPDATE ON agents FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_knowledge_items_updated_at BEFORE UPDATE ON knowledge_items FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 插入默认智能体
INSERT INTO agents (name, role, capabilities, configuration) VALUES
('代码专家', 'code_expert', ARRAY['代码生成', '代码审查', '调试', '重构'], '{"model": "gpt-4o-mini", "temperature": 0.3, "max_tokens": 4096, "tools": ["code_execution", "file_operations"], "system_prompt": "你是一位专业的代码专家，专注于高质量代码的生成、审查和优化。"}'),
('系统架构师', 'architect', ARRAY['系统设计', '技术选型', '架构评估', '文档编写'], '{"model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": 4096, "tools": ["documentation", "diagram_generation"], "system_prompt": "你是一位经验丰富的系统架构师，负责设计可扩展、可维护的软件架构。"}'),
('文档专家', 'doc_expert', ARRAY['技术文档', 'API文档', '用户手册', '代码注释'], '{"model": "gpt-4o-mini", "temperature": 0.4, "max_tokens": 4096, "tools": ["markdown_generation", "file_operations"], "system_prompt": "你是一位专业的技术文档专家，擅长创建清晰、准确、易懂的技术文档。"}'),
('任务调度器', 'supervisor', ARRAY['任务分解', '智能体协调', '工作流管理', '质量控制'], '{"model": "gpt-4o-mini", "temperature": 0.6, "max_tokens": 4096, "tools": ["task_management", "agent_coordination"], "system_prompt": "你是智能体团队的协调者，负责任务分解、分配和质量管控。"}'),
('知识检索专家', 'rag_specialist', ARRAY['语义搜索', '知识整合', '答案生成', '内容验证'], '{"model": "gpt-4o-mini", "temperature": 0.4, "max_tokens": 4096, "tools": ["vector_search", "knowledge_management"], "system_prompt": "你是知识检索和整合专家，擅长从大量信息中找到相关内容并生成准确答案。"}');
```

## Frontend Architecture

基于React 18.2+和选择的技术栈，以下是前端特定架构的详细设计：

### Component Architecture

#### Component Organization
```text
src/
├── components/
│   ├── ui/                     # 通用UI组件
│   │   ├── Button/
│   │   ├── Input/
│   │   ├── Modal/
│   │   └── DataTable/
│   ├── layout/                 # 布局组件
│   │   ├── Header/
│   │   ├── Sidebar/
│   │   └── MainLayout/
│   ├── agent/                  # 智能体相关组件
│   │   ├── AgentCard/
│   │   ├── AgentConfig/
│   │   └── AgentStatus/
│   ├── conversation/           # 对话相关组件
│   │   ├── MessageList/
│   │   ├── MessageInput/
│   │   └── ConversationHeader/
│   ├── task/                   # 任务相关组件
│   │   ├── TaskDashboard/
│   │   ├── DAGVisualizer/
│   │   └── TaskProgress/
│   └── knowledge/              # 知识库组件
│       ├── SearchInterface/
│       ├── KnowledgeItem/
│       └── RAGResponse/
├── pages/                      # 页面组件
├── hooks/                      # 自定义hooks
├── services/                   # API服务层
├── stores/                     # 状态管理
├── utils/                      # 工具函数
└── types/                      # TypeScript类型定义
```

### State Management Architecture

#### State Structure
```typescript
// stores/index.ts
import { create } from 'zustand';
import { devtools, persist } from 'zustand/middleware';
import { AgentSlice, createAgentSlice } from './agentSlice';
import { ConversationSlice, createConversationSlice } from './conversationSlice';
import { TaskSlice, createTaskSlice } from './taskSlice';
import { AuthSlice, createAuthSlice } from './authSlice';
import { UISlice, createUISlice } from './uiSlice';

// 全局状态类型
export interface RootState extends
  AgentSlice,
  ConversationSlice,
  TaskSlice,
  AuthSlice,
  UISlice {}

// 创建根状态存储
export const useAppStore = create<RootState>()(
  devtools(
    persist(
      (...args) => ({
        ...createAgentSlice(...args),
        ...createConversationSlice(...args),
        ...createTaskSlice(...args),
        ...createAuthSlice(...args),
        ...createUISlice(...args),
      }),
      {
        name: 'ai-agent-store',
        partialize: (state) => ({
          // 只持久化必要的状态
          auth: state.auth,
          ui: {
            theme: state.ui.theme,
            sidebarCollapsed: state.ui.sidebarCollapsed,
          },
        }),
      }
    ),
    { name: 'ai-agent-store' }
  )
);
```

#### State Management Patterns
- **分片模式**: 将状态按功能域分片，避免单一大状态对象
- **选择器模式**: 使用计算属性和记忆化选择器优化性能
- **乐观更新**: UI立即更新，API失败时回滚状态
- **错误边界**: 每个状态切片包含错误处理逻辑
- **持久化策略**: 仅持久化用户偏好和认证状态

### Routing Architecture

#### Protected Route Pattern
```typescript
import React, { Suspense } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { Spin } from 'antd';
import { useAuthStore } from '@/stores/authStore';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requiredPermissions?: string[];
  fallbackPath?: string;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({
  children,
  requiredPermissions = [],
  fallbackPath = '/login'
}) => {
  const location = useLocation();
  const { isAuthenticated, user, hasPermissions } = useAuthStore();

  // 检查认证状态
  if (!isAuthenticated) {
    return (
      <Navigate
        to={fallbackPath}
        state={{ from: location }}
        replace
      />
    );
  }

  // 检查权限
  if (requiredPermissions.length > 0 && !hasPermissions(requiredPermissions)) {
    return (
      <Navigate
        to="/unauthorized"
        state={{ from: location }}
        replace
      />
    );
  }

  return (
    <Suspense
      fallback={
        <div className="flex items-center justify-center h-64">
          <Spin size="large" tip="加载中..." />
        </div>
      }
    >
      {children}
    </Suspense>
  );
};
```

### Frontend Services Layer

#### API Client Setup
```typescript
import axios, { AxiosInstance, AxiosError } from 'axios';
import { message } from 'antd';
import { useAuthStore } from '@/stores/authStore';

// API客户端配置
class ApiClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000/api/v1',
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    this.setupInterceptors();
  }

  private setupInterceptors() {
    // 请求拦截器 - 添加认证头
    this.client.interceptors.request.use(
      (config) => {
        const { token } = useAuthStore.getState();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    // 响应拦截器 - 错误处理
    this.client.interceptors.response.use(
      (response) => response,
      (error: AxiosError) => {
        this.handleError(error);
        return Promise.reject(error);
      }
    );
  }

  private handleError(error: AxiosError) {
    if (error.response?.status === 401) {
      // 未授权，清除认证状态
      useAuthStore.getState().logout();
      window.location.href = '/login';
      return;
    }

    if (error.response?.status === 403) {
      message.error('权限不足');
      return;
    }

    if (error.response?.status >= 500) {
      message.error('服务器错误，请稍后重试');
      return;
    }

    // 显示具体错误信息
    const errorMessage = error.response?.data?.message || error.message;
    message.error(errorMessage);
  }

  // 封装常用HTTP方法
  get<T = any>(url: string, params?: any): Promise<T> {
    return this.client.get(url, { params }).then(res => res.data);
  }

  post<T = any>(url: string, data?: any): Promise<T> {
    return this.client.post(url, data).then(res => res.data);
  }

  put<T = any>(url: string, data?: any): Promise<T> {
    return this.client.put(url, data).then(res => res.data);
  }

  delete<T = any>(url: string): Promise<T> {
    return this.client.delete(url).then(res => res.data);
  }

  // WebSocket连接管理
  createWebSocket(path: string): WebSocket {
    const wsUrl = process.env.REACT_APP_WS_BASE_URL || 'ws://localhost:8000';
    const { token } = useAuthStore.getState();
    return new WebSocket(`${wsUrl}${path}?token=${token}`);
  }
}

export const apiClient = new ApiClient();
```

## Backend Architecture

基于FastAPI和选择的技术栈，以下是后端特定架构的详细设计：

### Service Architecture

#### Controller/Route Organization
```text
src/
├── api/
│   ├── v1/
│   │   ├── agents.py              # 智能体管理路由
│   │   ├── conversations.py       # 对话管理路由
│   │   ├── messages.py            # 消息处理路由
│   │   ├── tasks.py               # 任务管理路由
│   │   ├── dag_executions.py      # DAG执行路由
│   │   ├── knowledge.py           # 知识库路由
│   │   ├── rag.py                 # RAG查询路由
│   │   └── auth.py                # 认证路由
│   ├── deps.py                    # 依赖注入
│   ├── middleware.py              # 中间件
│   └── exceptions.py              # 异常处理
├── core/
│   ├── config.py                  # 配置管理
│   ├── security.py                # 安全相关
│   ├── database.py                # 数据库连接
│   └── logging.py                 # 日志配置
├── services/
│   ├── agent_service.py           # 智能体业务逻辑
│   ├── conversation_service.py    # 对话业务逻辑
│   ├── task_service.py            # 任务业务逻辑
│   └── rag_service.py             # RAG业务逻辑
├── models/
│   ├── database/                  # 数据库模型
│   └── schemas/                   # Pydantic数据模型
├── repositories/
│   ├── base.py                    # 基础仓储
│   ├── agent_repository.py       # 智能体数据访问
│   └── conversation_repository.py # 对话数据访问
├── ai/
│   ├── langgraph/                 # LangGraph集成
│   ├── autogen/                   # AutoGen集成
│   ├── mcp/                       # MCP协议实现
│   └── openai_client.py           # OpenAI API客户端
└── utils/
    ├── cache.py                   # 缓存工具
    ├── validators.py              # 验证器
    └── helpers.py                 # 辅助函数
```

### Database Architecture

#### Data Access Layer
```python
from typing import Generic, TypeVar, Type, List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete, func
from abc import ABC, abstractmethod
import uuid

ModelType = TypeVar("ModelType")
CreateSchemaType = TypeVar("CreateSchemaType")
UpdateSchemaType = TypeVar("UpdateSchemaType")

class BaseRepository(Generic[ModelType, CreateSchemaType, UpdateSchemaType], ABC):
    """基础仓储类，实现通用CRUD操作"""
    
    def __init__(self, model: Type[ModelType], db: AsyncSession):
        self.model = model
        self.db = db

    async def get(self, id: uuid.UUID) -> Optional[ModelType]:
        """根据ID获取单个实体"""
        query = select(self.model).where(self.model.id == id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()

    async def get_multi(
        self,
        *,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        order_by: Optional[str] = None
    ) -> tuple[List[ModelType], int]:
        """获取多个实体和总数"""
        query = select(self.model)
        count_query = select(func.count(self.model.id))
        
        # 应用过滤器
        if filters:
            for field, value in filters.items():
                if hasattr(self.model, field) and value is not None:
                    query = query.where(getattr(self.model, field) == value)
                    count_query = count_query.where(getattr(self.model, field) == value)
        
        # 应用排序
        if order_by and hasattr(self.model, order_by):
            query = query.order_by(getattr(self.model, order_by).desc())
        
        # 应用分页
        query = query.offset(skip).limit(limit)
        
        # 执行查询
        result = await self.db.execute(query)
        count_result = await self.db.execute(count_query)
        
        items = result.scalars().all()
        total = count_result.scalar()
        
        return items, total

    async def create(self, *, obj_in: CreateSchemaType, **kwargs) -> ModelType:
        """创建新实体"""
        obj_data = obj_in.dict() if hasattr(obj_in, 'dict') else obj_in
        obj_data.update(kwargs)
        db_obj = self.model(**obj_data)
        self.db.add(db_obj)
        await self.db.commit()
        await self.db.refresh(db_obj)
        return db_obj

    async def update(
        self, 
        *, 
        db_obj: ModelType, 
        obj_in: UpdateSchemaType
    ) -> ModelType:
        """更新实体"""
        obj_data = obj_in.dict(exclude_unset=True) if hasattr(obj_in, 'dict') else obj_in
        
        for field, value in obj_data.items():
            if hasattr(db_obj, field):
                setattr(db_obj, field, value)
        
        await self.db.commit()
        await self.db.refresh(db_obj)
        return db_obj

    async def remove(self, *, id: uuid.UUID) -> bool:
        """删除实体"""
        query = delete(self.model).where(self.model.id == id)
        result = await self.db.execute(query)
        await self.db.commit()
        return result.rowcount > 0
```

### Authentication and Authorization

#### Middleware/Guards
```python
from fastapi import Request, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional, List
import redis.asyncio as redis

from ..core.config import settings
from ..core.security import verify_password, create_access_token
from ..models.database.user import User
from ..repositories.user_repository import UserRepository

security = HTTPBearer()

class AuthService:
    """认证服务"""
    
    def __init__(self, db_session, redis_client):
        self.db = db_session
        self.redis = redis_client
        self.user_repo = UserRepository(db_session)

    async def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """验证用户凭据"""
        user = await self.user_repo.get_by_username(username)
        if not user or not user.is_active:
            return None
        
        if not verify_password(password, user.password_hash):
            return None
        
        return user

    async def create_user_session(self, user: User) -> dict:
        """创建用户会话"""
        # 生成访问令牌
        access_token = create_access_token(
            data={"sub": str(user.id), "username": user.username},
            expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        
        # 生成刷新令牌
        refresh_token = create_access_token(
            data={"sub": str(user.id), "type": "refresh"},
            expires_delta=timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        )
        
        # 存储会话到Redis
        session_key = f"session:{user.id}"
        session_data = {
            "user_id": str(user.id),
            "username": user.username,
            "is_active": user.is_active,
            "last_activity": datetime.utcnow().isoformat()
        }
        
        await self.redis.setex(
            session_key,
            timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            json.dumps(session_data)
        )
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }

    async def get_current_user(self, token: str) -> Optional[User]:
        """从令牌获取当前用户"""
        try:
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=[settings.ALGORITHM]
            )
            user_id: str = payload.get("sub")
            if user_id is None:
                return None
                
        except JWTError:
            return None
        
        # 检查会话状态
        session_key = f"session:{user_id}"
        session_data = await self.redis.get(session_key)
        if not session_data:
            return None
        
        # 获取用户信息
        user = await self.user_repo.get(uuid.UUID(user_id))
        if not user or not user.is_active:
            return None
        
        return user
```

## Unified Project Structure

基于monorepo架构和选择的技术工具，以下是完整的项目结构定义：

```plaintext
ai-agent-system/
├── .github/                           # CI/CD工作流
│   └── workflows/
│       ├── ci.yaml                    # 持续集成流水线
│       ├── deploy-staging.yaml        # 预发环境部署
│       └── deploy-production.yaml     # 生产环境部署
├── apps/                              # 应用程序包
│   ├── web/                           # 前端React应用
│   │   ├── public/                    # 静态资源
│   │   ├── src/
│   │   │   ├── components/            # React组件
│   │   │   │   ├── ui/                # 通用UI组件
│   │   │   │   ├── layout/            # 布局组件
│   │   │   │   ├── agent/             # 智能体组件
│   │   │   │   ├── conversation/      # 对话组件
│   │   │   │   ├── task/              # 任务组件
│   │   │   │   └── knowledge/         # 知识库组件
│   │   │   ├── pages/                 # 页面组件
│   │   │   ├── hooks/                 # 自定义hooks
│   │   │   ├── services/              # API服务层
│   │   │   ├── stores/                # 状态管理
│   │   │   ├── styles/                # 全局样式和主题
│   │   │   ├── utils/                 # 前端工具函数
│   │   │   ├── types/                 # TypeScript类型定义
│   │   │   ├── App.tsx                # 根组件
│   │   │   └── main.tsx               # 应用入口
│   │   ├── tests/                     # 前端测试
│   │   ├── package.json               # 前端依赖配置
│   │   ├── tailwind.config.js         # Tailwind CSS配置
│   │   ├── tsconfig.json              # TypeScript配置
│   │   └── vite.config.ts             # Vite构建配置
│   └── api/                           # 后端FastAPI应用
│       ├── src/
│       │   ├── api/                   # API路由层
│       │   │   ├── v1/
│       │   │   ├── deps.py            # 依赖注入
│       │   │   ├── middleware.py      # 中间件
│       │   │   └── exceptions.py      # 异常处理
│       │   ├── core/                  # 核心配置
│       │   │   ├── config.py          # 应用配置
│       │   │   ├── security.py        # 安全相关
│       │   │   ├── database.py        # 数据库连接
│       │   │   └── logging.py         # 日志配置
│       │   ├── services/              # 业务逻辑层
│       │   ├── models/                # 数据模型
│       │   │   ├── database/          # 数据库模型
│       │   │   ├── schemas/           # Pydantic数据模型
│       │   │   └── enums.py           # 枚举定义
│       │   ├── repositories/          # 数据访问层
│       │   ├── ai/                    # AI集成模块
│       │   │   ├── langgraph/         # LangGraph集成
│       │   │   ├── autogen/           # AutoGen集成
│       │   │   ├── mcp/               # MCP协议实现
│       │   │   ├── rag/               # RAG系统
│       │   │   ├── dag/               # DAG执行引擎
│       │   │   └── openai_client.py   # OpenAI API客户端
│       │   ├── utils/                 # 工具函数
│       │   ├── alembic/               # 数据库迁移
│       │   └── main.py                # FastAPI应用入口
│       ├── tests/                     # 后端测试
│       ├── Dockerfile                 # Docker镜像
│       ├── pyproject.toml             # Python项目配置
│       └── requirements.txt           # Python依赖
├── packages/                          # 共享包
│   ├── shared/                        # 共享类型和工具
│   │   ├── src/
│   │   │   ├── types/                 # 共享TypeScript类型
│   │   │   ├── constants/             # 共享常量
│   │   │   ├── utils/                 # 共享工具函数
│   │   │   └── index.ts               # 包导出入口
│   │   ├── package.json
│   │   └── tsconfig.json
│   ├── ui/                            # 共享UI组件库
│   │   ├── src/
│   │   │   ├── components/
│   │   │   ├── hooks/
│   │   │   ├── styles/
│   │   │   └── index.ts
│   │   ├── package.json
│   │   └── tsconfig.json
│   └── config/                        # 共享配置
│       ├── eslint/
│       ├── typescript/
│       └── jest/
├── infrastructure/                    # 基础设施即代码
│   ├── docker/                        # Docker配置
│   │   ├── Dockerfile.web
│   │   ├── Dockerfile.api
│   │   ├── docker-compose.yml         # 本地开发环境
│   │   ├── docker-compose.prod.yml    # 生产环境
│   │   └── nginx.conf                 # Nginx配置
│   ├── k8s/                          # Kubernetes部署配置
│   │   ├── namespace.yaml
│   │   ├── configmap.yaml
│   │   ├── secrets.yaml
│   │   ├── deployment.yaml
│   │   ├── service.yaml
│   │   ├── ingress.yaml
│   │   └── hpa.yaml
│   └── terraform/                     # Terraform IaC (可选)
├── scripts/                           # 构建和部署脚本
│   ├── build.sh                       # 构建脚本
│   ├── deploy.sh                      # 部署脚本
│   ├── test.sh                        # 测试脚本
│   ├── setup-dev.sh                   # 开发环境设置
│   ├── db-migrate.sh                  # 数据库迁移
│   └── seed-data.py                   # 种子数据生成
├── docs/                              # 项目文档
│   ├── brief.md                       # 项目简介
│   ├── prd.md                         # 产品需求文档
│   ├── front-end-spec.md              # 前端规格文档
│   ├── architecture.md                # 架构设计文档
│   ├── api/                           # API文档
│   ├── deployment/                    # 部署文档
│   └── development/                   # 开发文档
├── .env.example                       # 全局环境变量模板
├── .gitignore                         # Git忽略文件
├── .editorconfig                      # 编辑器配置
├── .prettierrc                        # Prettier配置
├── .eslintrc.js                       # ESLint配置
├── package.json                       # 根package.json (monorepo)
├── package-lock.json                  # 依赖锁文件
├── tsconfig.json                      # 根TypeScript配置
├── jest.config.js                     # Jest测试配置
└── README.md                         # 项目说明文档
```

## Development Workflow

基于monorepo架构和全栈应用需求，以下是完整的开发设置和工作流定义：

### Local Development Setup

#### Prerequisites
```bash
# 系统要求检查和安装
# Node.js 18+ 安装
curl -fsSL https://nodejs.org/dist/v18.19.0/node-v18.19.0-linux-x64.tar.xz | tar -xJ
export PATH=$PWD/node-v18.19.0-linux-x64/bin:$PATH

# Python 3.11+ 安装 (使用pyenv推荐)
curl https://pyenv.run | bash
pyenv install 3.11.7
pyenv global 3.11.7

# Docker和Docker Compose安装
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/download/v2.23.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 验证安装
node --version  # 应该 >= 18.0.0
python --version  # 应该 >= 3.11.0
docker --version  # 应该 >= 24.0.0
docker-compose --version  # 应该 >= 2.23.0
```

#### Initial Setup
```bash
# 1. 克隆仓库
git clone https://github.com/your-org/ai-agent-system.git
cd ai-agent-system

# 2. 安装根依赖和工作空间依赖
npm install

# 3. 设置Python虚拟环境
cd apps/api
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 4. 安装Python依赖
pip install -r requirements.txt

# 5. 复制环境变量模板
cp .env.example .env
cp apps/web/.env.example apps/web/.env.local
cp apps/api/.env.example apps/api/.env.local

# 6. 启动基础设施服务
docker-compose up -d postgres redis qdrant

# 7. 运行数据库迁移
cd apps/api
alembic upgrade head

# 8. 生成种子数据
python scripts/seed-data.py

# 9. 构建共享包
npm run build:packages

echo "开发环境设置完成！"
```

#### Development Commands
```bash
# 启动所有服务 (并行开发)
npm run dev

# 启动前端开发服务器
npm run dev:web

# 启动后端开发服务器
npm run dev:api

# 启动基础设施服务
npm run dev:infra

# 运行所有测试
npm run test

# 运行前端测试
npm run test:web

# 运行后端测试
npm run test:api

# 类型检查
npm run type-check

# 代码格式化
npm run format

# 代码检查
npm run lint

# 构建所有应用
npm run build

# 数据库操作
npm run db:migrate      # 运行迁移
npm run db:rollback     # 回滚迁移
npm run db:seed         # 生成种子数据
npm run db:reset        # 重置数据库
```

### Environment Configuration

#### Required Environment Variables
```bash
# 前端环境变量 (.env.local)
REACT_APP_API_BASE_URL=http://localhost:8000/api/v1
REACT_APP_WS_BASE_URL=ws://localhost:8000
REACT_APP_ENABLE_DEVTOOLS=true

# 后端环境变量 (.env)
APP_NAME=AI Agent System
DEBUG=true
SECRET_KEY=your-super-secret-key-change-in-production
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ai_agent_db
REDIS_URL=redis://localhost:6379/0
QDRANT_URL=http://localhost:6333
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_KEY=your_openai_api_key
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# 共享环境变量 (.env)
NODE_ENV=development
ENVIRONMENT=local
TZ=UTC
COMPOSE_PROJECT_NAME=ai-agent-system
```

## Deployment Architecture

基于Docker容器化和云原生部署的策略定义：

### Deployment Strategy

**Frontend Deployment:**
- **Platform:** Vercel / Netlify（推荐）或 Nginx + Docker
- **Build Command:** `npm run build`
- **Output Directory:** `apps/web/dist`
- **CDN/Edge:** 全球CDN加速，边缘计算优化

**Backend Deployment:**
- **Platform:** Docker容器 + Kubernetes集群
- **Build Command:** `docker build -f apps/api/Dockerfile .`
- **Deployment Method:** 滚动更新，零停机部署

### CI/CD Pipeline

```yaml
# .github/workflows/ci.yaml
name: Continuous Integration

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

env:
  NODE_VERSION: '18.19.0'
  PYTHON_VERSION: '3.11.7'

jobs:
  # 变更检测和缓存优化
  changes:
    runs-on: ubuntu-latest
    outputs:
      frontend: ${{ steps.changes.outputs.frontend }}
      backend: ${{ steps.changes.outputs.backend }}
      shared: ${{ steps.changes.outputs.shared }}
    steps:
      - uses: actions/checkout@v4
      - uses: dorny/paths-filter@v2
        id: changes
        with:
          filters: |
            frontend:
              - 'apps/web/**'
              - 'packages/ui/**'
            backend:
              - 'apps/api/**'
              - 'requirements.txt'
            shared:
              - 'packages/shared/**'
              - 'package.json'

  # 前端构建和测试
  frontend:
    runs-on: ubuntu-latest
    needs: [changes]
    if: needs.changes.outputs.frontend == 'true' || needs.changes.outputs.shared == 'true'
    steps:
      - uses: actions/checkout@v4
      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: ${{ env.NODE_VERSION }}
          cache: 'npm'
      - name: Install dependencies
        run: npm ci
      - name: Build shared packages
        run: npm run build:packages
      - name: Lint frontend
        run: npm run lint --workspace=apps/web
      - name: Run frontend tests
        run: npm run test --workspace=apps/web
      - name: Build frontend
        run: npm run build:web

  # 后端构建和测试
  backend:
    runs-on: ubuntu-latest
    needs: changes
    if: needs.changes.outputs.backend == 'true'
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test_password
          POSTGRES_USER: test_user
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'
      - name: Install Python dependencies
        run: |
          cd apps/api
          pip install -r requirements.txt
      - name: Run backend tests
        run: |
          cd apps/api
          pytest --cov=src
        env:
          DATABASE_URL: postgresql+asyncpg://test_user:test_password@localhost:5432/test_db
```

### Environments

| Environment | Frontend URL | Backend URL | Purpose |
|-------------|--------------|-------------|---------|
| Development | http://localhost:3000 | http://localhost:8000 | 本地开发环境 |
| Staging | https://staging.ai-agent-system.com | https://staging-api.ai-agent-system.com | 预发布测试环境 |
| Production | https://ai-agent-system.com | https://api.ai-agent-system.com | 生产环境 |

## Security and Performance

基于全栈AI应用的特殊需求，定义安全和性能的综合策略：

### Security Requirements

**Frontend Security:**
- CSP Headers: `default-src 'self'; script-src 'self' 'unsafe-eval'; connect-src 'self' ws: wss: https://api.openai.com;`
- XSS Prevention: DOMPurify sanitization for user-generated content, Content Security Policy enforcement
- Secure Storage: JWT tokens in httpOnly cookies, sensitive data encrypted in localStorage using Web Crypto API

**Backend Security:**
- Input Validation: Pydantic models with comprehensive validation, SQL injection prevention through parameterized queries
- Rate Limiting: `{"global": {"requests_per_minute": 1000}, "per_user": {"requests_per_minute": 100}, "ai_api": {"requests_per_minute": 50}}`
- CORS Policy: `{"allow_origins": ["https://ai-agent-system.com"], "allow_methods": ["GET", "POST", "PUT", "DELETE"], "allow_headers": ["Authorization", "Content-Type"]}`

**Authentication Security:**
- Token Storage: JWT access tokens (30min expiry) + refresh tokens (7 days) stored in secure httpOnly cookies
- Session Management: Redis-based session store with automatic cleanup, concurrent session limits (5 sessions per user)
- Password Policy: Minimum 8 characters, must include uppercase, lowercase, number, and special character; bcrypt hashing with cost factor 12

### Performance Optimization

**Frontend Performance:**
- Bundle Size Target: `{"initial": "< 500KB gzipped", "total": "< 2MB", "code_splitting": "route-based + component-based"}`
- Loading Strategy: Progressive loading with skeleton screens, image lazy loading, virtual scrolling for large lists
- Caching Strategy: `{"static_assets": "1 year", "api_responses": "5 minutes", "user_data": "session-based"}`

**Backend Performance:**
- Response Time Target: `{"p95": "< 200ms", "p99": "< 500ms", "ai_operations": "< 5s"}`
- Database Optimization: Connection pooling (min: 5, max: 20), query optimization with EXPLAIN ANALYZE, index optimization
- Caching Strategy: `{"redis": {"ttl": 300, "keys": ["user_sessions", "api_responses", "computed_results"]}, "in_memory": {"lru_cache": 1000}}`

## Testing Strategy

基于全栈AI应用的复杂性，定义分层测试策略确保系统质量：

### Testing Pyramid

```text
                  E2E Tests (10%)
                 /              \
            Integration Tests (20%)
               /                    \
          Frontend Unit (35%)    Backend Unit (35%)
```

### Test Organization

#### Frontend Tests
```text
apps/web/tests/
├── __mocks__/                     # Mock数据和服务
├── components/                    # 组件测试
├── hooks/                         # Hook测试
├── services/                      # 服务层测试
├── stores/                        # 状态管理测试
├── utils/                         # 工具函数测试
├── pages/                         # 页面集成测试
└── e2e/                          # E2E测试
```

#### Backend Tests
```text
apps/api/tests/
├── conftest.py                    # pytest配置和fixture
├── api/                          # API端点测试
├── services/                     # 业务逻辑测试
├── repositories/                 # 数据访问测试
├── ai/                          # AI模块测试
├── utils/                       # 工具函数测试
└── integration/                 # 集成测试
```

**测试覆盖率目标:**
- **单元测试覆盖率**: ≥80%
- **集成测试覆盖率**: ≥70%
- **E2E测试覆盖率**: ≥60% (关键用户流程)
- **AI模块测试覆盖率**: ≥85% (关键业务逻辑)

## Coding Standards

基于AI-First开发模式和多智能体系统特点，定义关键编码标准以防止常见错误：

### Critical Fullstack Rules

- **Type Sharing:** 所有数据类型必须在packages/shared中定义，前后端从统一位置导入，避免类型不一致导致的运行时错误
- **API Calls:** 禁止直接使用fetch或axios，必须通过service层统一调用，确保错误处理、重试逻辑和监控的一致性
- **Environment Variables:** 禁止直接访问process.env，必须通过config对象访问，确保环境变量的验证和类型安全
- **Error Handling:** 所有API路由必须使用标准错误处理器，确保错误格式一致和安全信息过滤
- **State Updates:** 禁止直接修改状态对象，必须使用不可变更新模式，避免状态同步问题
- **AI API Calls:** 所有AI服务调用必须包含超时、重试和降级机制，防止系统阻塞
- **Async Operations:** 异步操作必须正确处理Promise rejection，避免未捕获的Promise错误
- **Database Transactions:** 涉及多表操作必须使用事务，确保数据一致性
- **Cache Invalidation:** 缓存更新必须遵循write-through模式，避免数据不一致
- **WebSocket Connections:** WebSocket连接必须实现重连机制和心跳检测
- **File Operations:** 文件操作必须验证路径和权限，防止路径遍历攻击
- **Input Validation:** 所有用户输入必须在前后端都进行验证，实现纵深防御

### Naming Conventions

| Element | Frontend | Backend | Example |
|---------|----------|---------|---------|
| Components | PascalCase | - | `AgentConfigPanel.tsx` |
| Hooks | camelCase with 'use' | - | `useAgentWebSocket.ts` |
| API Routes | - | kebab-case | `/api/v1/agent-conversations` |
| Python Classes | - | PascalCase | `AgentOrchestrator` |
| Python Functions | - | snake_case | `create_multi_agent_conversation` |
| Database Tables | - | snake_case | `agent_conversations` |
| Constants | UPPER_SNAKE_CASE | UPPER_SNAKE_CASE | `MAX_CONVERSATION_LENGTH` |

## Error Handling Strategy

定义统一的错误处理机制跨前端和后端：

### Error Flow
系统实现多层错误处理，包括API重试、任务重新分配和用户友好的错误通知。

### Error Response Format
```typescript
interface ApiError {
  error: {
    code: string;
    message: string;
    details?: Record<string, any>;
    timestamp: string;
    requestId: string;
  };
}
```

### Frontend Error Handling
前端实现统一的错误处理拦截器，自动处理常见错误场景并提供用户友好的错误提示。

### Backend Error Handling
后端使用FastAPI的异常处理机制，确保所有错误都被正确捕获、记录和返回。

## Monitoring and Observability

定义系统监控策略：

### Monitoring Stack
- **Frontend Monitoring:** 客户端错误追踪和性能监控
- **Backend Monitoring:** 服务器性能指标和API监控
- **Error Tracking:** 统一的错误收集和分析
- **Performance Monitoring:** 关键性能指标跟踪

### Key Metrics
**Frontend Metrics:**
- Core Web Vitals
- JavaScript errors
- API response times
- User interactions

**Backend Metrics:**
- Request rate
- Error rate
- Response time
- Database query performance

---

本架构文档将随着项目发展不断更新和完善，确保始终反映系统的最新状态和设计决策。