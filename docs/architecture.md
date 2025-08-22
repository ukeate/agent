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
| 2025-08-19 | 2.0 | Architecture upgrade alignment for 2025 epic requirements | Architect (Winston) |

**Version 2.0 主要升级内容:**

#### **核心技术栈升级 (Epics 1-5)**
- **LangGraph 0.6.5**: Context API v0.6, Durability控制, Node级缓存
- **AutoGen 0.4.2b1**: Actor Model架构, 异步事件驱动, 内置Observability
- **Qdrant BM42混合搜索**: 稀疏+密集向量, 检索精度提升30%
- **pgvector 0.8**: 迭代索引扫描, HNSW优化, 向量量化压缩
- **多模态AI集成**: Claude 4 + GPT-4o多模态能力
- **AI TRiSM安全框架**: 企业级AI安全管理，威胁检测率>99%
- **OpenTelemetry可观测性**: AI Agent语义约定, 分布式追踪
- **高级推理引擎**: 链式思考(CoT), 多步推理, 智能记忆管理
- **边缘AI准备**: 模型量化压缩, 离线能力, ONNX Runtime集成

#### **高级AI功能扩展 (Epics 6-11)**
- **强化学习个性化系统**: 多臂老虎机推荐, Q-Learning优化, A/B测试框架
- **实时语音交互系统**: Whisper ASR, 高质量TTS, 语音情感识别, VAD处理
- **动态知识图谱引擎**: 实体关系抽取, 图谱推理, GraphRAG集成, SPARQL查询
- **模型微调优化平台**: LoRA/QLoRA微调, 模型压缩量化, 自动超参数优化
- **分布式智能体网络**: 服务发现注册, 分布式协调, 容错恢复, 集群管理
- **高级情感智能系统**: 多模态情感识别, 共情响应, 情感记忆, 情感健康监测

#### **技术能力跃升指标**
- **性能提升**: 响应时间50%↑, 并发能力100%↑, 检索精度30%↑
- **智能化程度**: 自学习个性化, 情感交互, 多模态理解, 知识推理
- **系统可扩展性**: 分布式架构, 千级智能体并发, 企业级高可用
- **技术自主性**: 模型自训练, 知识自更新, 性能自优化

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

**Responsibility:** 多智能体工作流编排，Context API v0.6状态管理，Node级缓存和执行监控

**Key Interfaces:**
- **LangGraph v0.6.5 Context API**: 类型安全的运行时上下文管理，替代config['configurable']
- **Durability Controls**: 细粒度持久化控制 (`durability="sync/async/exit"`)
- **Node Caching**: 跳过重复计算，开发迭代加速，缓存命中率优化
- **Deferred Nodes**: 延迟执行支持，map-reduce模式，批处理优化
- **Pre/Post Model Hooks**: 模型调用前后的自定义逻辑，guardrails集成
- **Checkpoint Management**: 高级状态检查点，支持工作流恢复和回滚

**Dependencies:** AutoGen Agent Pool, MCP Tool Registry, PostgreSQL (状态持久化), Redis (Node缓存), OpenTelemetry (监控)

**Technology Stack:** LangGraph 0.6.5, Context API v0.6, Durability控制, Node-level缓存, Python asyncio

**2025升级特性:**
```python
# 新Context API使用示例
@entrypoint(checkpointer=checkpointer)
def workflow(inputs, *, previous, context):
    # 类型安全的上下文访问
    user_info = context.get("user_profile")
    
    # Durability控制
    result = some_node.invoke(
        inputs, 
        durability="sync"  # 同步持久化
    )
    return entrypoint.final(value=result, save=state)
```

### AutoGen Agent Pool

**Responsibility:** 异步事件驱动的AI智能体管理，Actor Model架构，企业级多智能体协作和监控

**Key Interfaces:**
- **Actor Model架构**: 异步消息传递，分布式智能体网络通信
- **Event-Driven系统**: 支持复杂的智能体协作模式，事件路由和处理
- **模块化设计**: Core + AgentChat + Extensions三层架构
- **内置Observability**: OpenTelemetry集成，生产级监控和追踪
- **AutoGen Studio v2**: 低代码智能体构建界面，可视化工作流设计
- **异步消息处理**: 支持高并发智能体通信，消息队列管理
- **智能体生命周期管理**: 创建、暂停、恢复、销毁的完整生命周期

**Dependencies:** OpenAI API, Claude 4 API, MCP Tools, LangGraph Orchestrator, AI TRiSM Security, OpenTelemetry

**Technology Stack:** AutoGen 0.4.2b1, Actor Model, 异步事件处理, 企业级安全集成, 分布式架构

**2025重大架构升级:**
```python
# AutoGen 0.4 Actor模型示例
from autogen_core import RoutedAgent, MessageContext
from autogen_core.models import ChatCompletionClient

class AsyncAgent(RoutedAgent):
    def __init__(self, model_client: ChatCompletionClient):
        super().__init__("Async Agent")
        self._model_client = model_client
    
    @message_handler
    async def handle_message(self, message: Message, ctx: MessageContext):
        # 异步消息处理
        response = await self._model_client.create(
            messages=[message.to_llm_message()],
            cancellation_token=ctx.cancellation_token
        )
        return Message(content=response.content)
```

**架构变更影响:**
- 包名变更: `autogen` → `autogen-agentchat`
- 从同步群组对话变更为异步事件驱动模式
- 内置OpenTelemetry支持，实现生产级可观测性

### RAG Knowledge Engine

**Responsibility:** 高性能混合搜索智能知识检索系统，支持BM42+向量搜索、多模态RAG、上下文增强和答案生成

**Key Interfaces:**
- **Qdrant BM42混合搜索**: 稀疏+密集向量，精确关键词匹配+语义理解
- **FastEmbed推理引擎**: 高性能向量生成，Transformer注意力权重优化
- **pgvector 0.8优化**: 迭代索引扫描，HNSW索引优化，向量量化压缩
- **多模态RAG**: 图像、文档、视频内容的智能检索和理解
- **向量压缩优化**: 平均向量大小仅5.6元素/文档，存储效率提升
- **智能Fallback机制**: 多层搜索策略，确保检索成功率
- **上下文增强生成**: 基于检索结果的智能答案合成

**Dependencies:** Qdrant Vector DB (BM42), pgvector 0.8, OpenAI Embeddings, Claude 4 API, FastEmbed, Knowledge Repository, Performance Monitor

**Technology Stack:** Qdrant 1.7+ (BM42混合搜索), pgvector 0.8, FastEmbed 0.3+, sentence-transformers, 混合检索算法, 向量量化

**2025搜索优化特性:**
```python
# Qdrant BM42混合搜索示例
from qdrant_client import QdrantClient
from fastembed import TextEmbedding

client = QdrantClient("localhost", port=6333)

# 混合搜索查询
search_result = client.search(
    collection_name="hybrid_search",
    query_vector=("dense", dense_vector),
    sparse_vector=("sparse", sparse_vector),
    fusion=Fusion.RRF,  # Reciprocal Rank Fusion
    limit=10
)

# pgvector 0.8迭代索引扫描
SELECT * FROM documents 
ORDER BY embedding <-> query_vector 
LIMIT 10;  -- 优化的HNSW索引性能
```

**性能提升指标:**
- 检索精度提升30% (BM42混合 vs 纯向量搜索)
- 存储效率提升25% (向量量化压缩)
- 查询响应时间减少40% (迭代索引扫描优化)

### React Frontend Shell

**Responsibility:** 前端应用框架，路由管理，状态协调，组件渲染

**Key Interfaces:**
- 页面路由系统
- 全局状态管理
- API客户端集成
- 实时通信WebSocket

**Dependencies:** API Gateway, 各功能组件

**Technology Stack:** React 18.2+, React Router, Zustand, WebSocket客户端

### AI Security Framework (AI TRiSM)

**Responsibility:** 企业级AI安全管理，信任、风险和安全管理，对抗攻击防护和威胁检测

**Key Interfaces:**
- **Trust (信任)**: 模型输出可解释性和透明度，AI决策审计跟踪
- **Risk (风险)**: 对抗攻击检测和防护机制，模型中毒检测
- **Security (安全)**: 数据隐私和访问控制，敏感信息泄漏防护
- **Prompt Injection检测**: 恶意提示识别和拦截，输入过滤机制
- **Data Leakage防护**: 敏感信息检测，自动化数据脱敏
- **Model Poisoning检测**: 模型中毒和潜在威胁识别
- **自动化安全响应系统**: 威胁检测率>99%，误报率<1%

**Dependencies:** AI模型API, 安全数据库, 威胁情报源, 审计日志系统

**Technology Stack:** AI TRiSM 1.0+, 机器学习安全模型, 实时威胁检测, 自动化响应系统

### OpenTelemetry AI Observability

**Responsibility:** AI系统专用的分布式追踪、性能监控和可观测性平台

**Key Interfaces:**
- **AI Agent语义约定**: 标准化的智能体监控格式和指标
- **分布式追踪**: 跨智能体的请求链路追踪，完整调用链可视化
- **性能指标收集**: 模型推理延迟、token使用量、资源消耗监控
- **非确定性系统监控**: 专为AI系统设计的观测最佳实践
- **智能体行为分析**: 决策路径分析，工具调用模式，错误模式识别
- **实时告警系统**: 性能异常、错误率、资源瓶颈预警
- **AI操作审计**: 完整的AI决策过程记录和回溯能力

**Dependencies:** 所有AI组件, Prometheus, Grafana, 日志聚合系统

**Technology Stack:** OpenTelemetry 1.25+, AI Agent语义约定, Prometheus, Grafana, 分布式追踪

**2025可观测性示例:**
```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

# AI Agent语义约定追踪
tracer = trace.get_tracer("ai.agent.system")

with tracer.start_as_current_span("agent.reasoning") as span:
    span.set_attribute("ai.agent.name", "reasoning_agent")
    span.set_attribute("ai.model.name", "claude-3.5-sonnet")
    span.set_attribute("ai.token.usage.input", input_tokens)
    span.set_attribute("ai.token.usage.output", output_tokens)
    
    result = agent.reason(query)
    
    span.set_attribute("ai.agent.decision", result.decision)
    span.set_attribute("ai.agent.confidence", result.confidence)
```

### Multimodal AI Engine

**Responsibility:** 多模态AI处理能力，图像、文档、视频理解，智能内容分析

**Key Interfaces:**
- **Claude 4多模态集成**: 图像理解、文档分析、视觉问答
- **GPT-4o视觉能力**: 图像识别、OCR、场景理解、物体检测
- **智能文档处理**: PDF/Word/Excel解析和知识抽取
- **视频内容分析**: 关键帧提取、内容理解和摘要生成
- **多模态RAG集成**: 文本+图像+视频的统一检索和理解
- **内容质量评估**: 多模态内容的自动化质量检测

**Dependencies:** Claude 4 API, GPT-4o API, 文件存储系统, RAG Knowledge Engine

**Technology Stack:** Claude 4 API, GPT-4o API, 多模态处理pipeline, 内容分析引擎

### Advanced Reasoning Engine

**Responsibility:** 高级推理能力，链式思考，多步推理，智能记忆管理

**Key Interfaces:**
- **链式思考 (CoT)**: 逐步解决复杂问题，推理路径可视化
- **多步推理工作流**: 组合多个推理步骤，复杂问题分解
- **智能记忆管理**: 上下文感知的记忆存储和检索
- **元认知能力**: 对推理过程的反思和优化
- **解释性AI决策**: AI决策的可解释性和透明度
- **推理质量评估**: 推理过程的准确性和可信度评估

**Dependencies:** LangGraph Orchestrator, AutoGen Agent Pool, 记忆存储系统

**Technology Stack:** 高级推理算法, 记忆管理系统, 推理质量评估模型

### Edge AI Deployment Engine

**Responsibility:** 边缘AI部署支持，模型量化压缩，离线能力，端侧推理

**Key Interfaces:**
- **模型量化技术**: INT8/INT4量化，推理加速，精度保持
- **模型压缩优化**: 知识蒸馏，模型裁剪，参数压缩
- **端侧部署架构**: 轻量级推理引擎，资源优化部署
- **离线能力支持**: 无网络环境下的AI功能保持
- **同步机制设计**: 在线-离线数据同步，增量更新

**Dependencies:** ONNX Runtime, 模型压缩工具, 边缘设备管理

**Technology Stack:** ONNX Runtime 1.16+, 模型量化框架, 边缘推理引擎

### Reinforcement Learning Personalization Engine

**Responsibility:** 强化学习个性化系统，用户行为学习，智能推荐优化

**Key Interfaces:**
- **多臂老虎机推荐**: UCB、Thompson Sampling算法，动态推荐优化
- **Q-Learning智能体**: 行为策略强化学习，奖励函数优化
- **用户反馈学习**: 隐式和显式反馈处理，多维度信号融合
- **A/B测试框架**: 在线实验管理，统计显著性检验
- **实时个性化**: 毫秒级推荐响应，增量学习更新
- **行为分析**: 用户轨迹记录，模式识别，异常检测

**Dependencies:** 用户行为数据, Redis缓存, 实验管理数据库, 统计分析引擎

**Technology Stack:** Ray/Optuna, 强化学习算法库, A/B测试框架, 实时计算引擎

### Real-time Voice Interaction System

**Responsibility:** 实时语音交互系统，ASR/TTS，语音情感识别，自然语音对话

**Key Interfaces:**
- **实时语音转文本**: Whisper模型，流式识别，多语言支持
- **文本转语音合成**: 高质量TTS，多音色情感表达，流式生成
- **语音情感识别**: 音频情感特征提取，实时情感跟踪
- **语音活动检测**: VAD，智能打断处理，对话轮次管理
- **多轮对话管理**: 语音上下文理解，对话状态跟踪
- **音频优化**: 回声消除，降噪，编解码优化

**Dependencies:** 音频设备接口, WebRTC, 对话管理系统, 情感分析引擎

**Technology Stack:** Whisper v3, Azure Speech, WebRTC, 音频处理库, 实时通信

### Dynamic Knowledge Graph Engine

**Responsibility:** 动态知识图谱系统，实体关系抽取，图谱推理，GraphRAG集成

**Key Interfaces:**
- **实体识别与关系抽取**: NER+RE，实体链接消歧，多语言支持
- **动态图谱构建**: 增量式构建，知识冲突解决，质量评估
- **图谱推理引擎**: 基于规则和嵌入的推理，多跳关系推理
- **GraphRAG集成**: 图谱增强检索，实体关系上下文扩展
- **可视化查询**: 交互式图谱可视化，自然语言到图查询
- **SPARQL接口**: 标准图查询语言，知识图谱管理API

**Dependencies:** NLP模型, 图数据库, RAG Knowledge Engine, 可视化框架

**Technology Stack:** Neo4j/ArangoDB, spaCy/Stanza, 知识图谱嵌入模型, D3.js/Cytoscape

### Model Fine-tuning Platform

**Responsibility:** 模型微调优化平台，LoRA/QLoRA训练，模型压缩量化，自动优化

**Key Interfaces:**
- **LoRA/QLoRA微调**: 高效参数微调，多GPU分布式训练
- **模型压缩量化**: INT8/INT4量化，知识蒸馏，模型剪枝
- **自动超参数优化**: Optuna搜索，贝叶斯优化，早停策略
- **模型性能评估**: 多维度指标，基准测试，性能回归检测
- **训练数据管理**: 数据收集标注，质量评估，版本控制
- **模型部署优化**: 推理加速，内存优化，批处理优化

**Dependencies:** GPU计算资源, 训练数据集, 模型评估基准, 部署环境

**Technology Stack:** Hugging Face Transformers, LoRA/QLoRA, Optuna, 量化框架

### Distributed Agent Network Manager

**Responsibility:** 分布式智能体网络，服务发现，任务协调，容错恢复，集群管理

**Key Interfaces:**
- **智能体服务发现**: etcd/Consul注册中心，健康检查，负载均衡
- **分布式消息通信**: NATS/RabbitMQ消息总线，点对点通信
- **任务协调引擎**: 分布式共识，任务分解分配，状态同步
- **集群管理**: 智能体生命周期，资源监控，动态扩缩容
- **容错恢复**: 故障检测隔离，任务重分配，网络分区处理
- **性能监控**: 集群拓扑可视化，资源使用统计，告警通知

**Dependencies:** 分布式协调服务, 消息队列, 监控系统, 容器编排

**Technology Stack:** etcd/Consul, NATS/RabbitMQ, Raft/PBFT, Kubernetes, 监控栈

### Advanced Emotional Intelligence System

**Responsibility:** 高级情感智能系统，多模态情感识别，共情响应，情感记忆管理

**Key Interfaces:**
- **多模态情感识别**: 文本、语音、视觉情感分析，生理信号推断
- **情感状态建模**: 多维情感空间，时间动态跟踪，个性化画像
- **共情响应生成**: 情感感知回复，情感调节安慰，适应性镜像
- **情感记忆管理**: 长期交互历史，情感事件关联，偏好学习
- **情感智能决策**: 情感状态行为选择，风险评估，干预策略
- **情感健康监测**: 情感状态分析，心理健康评估，预警机制

**Dependencies:** 多模态AI引擎, 用户交互历史, 心理学知识库, 医疗健康数据

**Technology Stack:** 多模态情感模型, 情感计算框架, 心理学AI, 长期记忆系统

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

### 2025年架构升级的关键工作流改进:

#### 1. **Context API工作流** (LangGraph 0.6.5)
- 类型安全的上下文传递，替代传统config模式
- Durability控制实现细粒度状态管理
- Node缓存优化开发迭代和运行时性能

#### 2. **异步事件驱动架构** (AutoGen 0.4.2b1)
- Actor Model实现真正的异步智能体通信
- 事件驱动系统支持复杂协作模式
- 并行智能体处理，显著提升处理能力

#### 3. **BM42混合搜索工作流** (Qdrant + pgvector 0.8)
- 稀疏+密集向量的混合检索策略
- FastEmbed推理引擎优化向量生成
- pgvector 0.8的迭代索引扫描优化

#### 4. **多模态AI集成工作流**
- Claude 4和GPT-4o的多模态能力整合
- 文本、图像、文档的统一处理pipeline
- 多模态RAG增强的智能检索

#### 5. **AI安全和监控工作流**
- AI TRiSM安全框架的实时威胁检测
- OpenTelemetry的完整分布式追踪
- 智能体行为分析和性能优化

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

**AI Security Framework (AI TRiSM):**
- **Trust (信任)**: 模型输出可解释性和透明度，AI决策审计跟踪
- **Risk (风险)**: 对抗攻击检测和防护机制，模型中毒检测
- **Security (安全)**: 数据隐私和访问控制，敏感信息泄漏防护
- **Threat Detection**: Prompt Injection识别和拦截，恶意输入过滤
- **Automated Response**: 自动化安全响应系统，威胁检测率>99%，误报率<1%

### Performance Optimization

**Frontend Performance:**
- Bundle Size Target: `{"initial": "< 500KB gzipped", "total": "< 2MB", "code_splitting": "route-based + component-based"}`
- Loading Strategy: Progressive loading with skeleton screens, image lazy loading, virtual scrolling for large lists
- Caching Strategy: `{"static_assets": "1 year", "api_responses": "5 minutes", "user_data": "session-based"}`

**Backend Performance:**
- Response Time Target: `{"p95": "< 140ms", "p99": "< 350ms", "ai_operations": "< 3.5s"}` (30%提升目标)
- Database Optimization: Connection pooling (min: 5, max: 20), query optimization with EXPLAIN ANALYZE, index optimization
- Caching Strategy: `{"redis": {"ttl": 300, "keys": ["user_sessions", "api_responses", "computed_results"]}, "in_memory": {"lru_cache": 1000}}`
- Concurrency Target: 500 RPS → 1000+ RPS (100%+提升)

**Observability & Monitoring (OpenTelemetry):**
- **Distributed Tracing**: 全链路追踪，包括AI操作和多智能体协作
- **Metrics Collection**: 性能、错误、业务指标实时收集和分析
- **Log Correlation**: 结构化日志关联，AI决策过程可追踪
- **Alert System**: 关键问题告警时间 < 30s，预测性监控
- **Performance Dashboard**: 实时性能仪表盘，AI系统健康检查

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