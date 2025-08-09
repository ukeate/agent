# Data Models

基于PRD要求和AI系统特性，我定义了以下核心数据模型来支持多智能体协作、任务规划和知识管理：

## Agent

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

### TypeScript Interface
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

### Relationships
- 一个Agent可以参与多个Conversation
- 一个Agent可以执行多个Task
- Agent之间通过Message进行交互

## Conversation

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

### TypeScript Interface
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

### Relationships
- 一个Conversation包含多个Message
- 一个Conversation可以关联多个Task
- 一个Conversation可以触发DAG执行

## Message

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

### TypeScript Interface
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

### Relationships
- 属于一个Conversation
- 可能触发Task创建
- 可能包含KnowledgeItem引用

## Task

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

### TypeScript Interface
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

### Relationships
- 属于一个DAGExecution
- 被分配给一个Agent
- 可能由Message触发创建
- 可以生成KnowledgeItem

## DAGExecution

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

### TypeScript Interface
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

### Relationships
- 关联一个Conversation
- 包含多个Task
- 由Supervisor智能体管理

## KnowledgeItem

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

### TypeScript Interface
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

### Relationships
- 可以被Message引用
- 用于RAG检索增强
- 可以由Task生成
