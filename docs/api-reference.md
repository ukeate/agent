# API Reference - AI Agent System & A/B Testing Platform

## 目录
- [概述](#概述)
- [认证](#认证)
- [基础约定](#基础约定)
- [核心API](#核心api)
  - [智能体管理](#智能体管理)
  - [多智能体协作](#多智能体协作)
  - [工作流管理](#工作流管理)
  - [RAG系统](#rag系统)
  - [A/B测试实验](#ab测试实验)
  - [统计分析](#统计分析)
  - [发布策略](#发布策略)
- [错误处理](#错误处理)
- [限流策略](#限流策略)
- [WebSocket接口](#websocket接口)

## 概述

AI Agent System API提供了完整的智能体管理、多智能体协作、RAG系统和A/B测试实验平台功能。

**基础URL**: `https://api.ai-agent.com/api/v1`

**API版本**: v1.0.0

## 认证

大部分API需要Bearer Token认证：

```http
Authorization: Bearer <your-token>
```

获取Token:
```http
POST /api/v1/auth/login
Content-Type: application/json

{
  "username": "user@example.com",
  "password": "password"
}
```

响应:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

## 基础约定

### 请求格式
- 所有POST/PUT请求使用JSON格式
- Content-Type: application/json
- 字符编码: UTF-8

### 响应格式
成功响应:
```json
{
  "data": {...},
  "message": "Success",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

错误响应:
```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Error description",
    "details": {...}
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 分页
列表接口支持分页:
- `limit`: 每页数量 (默认: 20, 最大: 100)
- `offset`: 偏移量 (默认: 0)
- `sort`: 排序字段
- `order`: 排序方向 (asc/desc)

### 过滤
支持查询参数过滤:
- `status`: 状态过滤
- `created_after`: 创建时间起始
- `created_before`: 创建时间结束
- `search`: 关键词搜索

## 核心API

### 智能体管理

#### 创建智能体
```http
POST /api/v1/agents
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "DataAnalyst",
  "type": "react",
  "description": "数据分析智能体",
  "tools": ["calculator", "search", "database"],
  "model": "claude-3.5-sonnet",
  "temperature": 0.7,
  "max_iterations": 10
}
```

响应:
```json
{
  "data": {
    "id": "agent_123",
    "name": "DataAnalyst",
    "type": "react",
    "status": "active",
    "created_at": "2024-01-01T00:00:00Z"
  }
}
```

#### 执行智能体任务
```http
POST /api/v1/agents/{agent_id}/execute
Content-Type: application/json
Authorization: Bearer <token>

{
  "input": "分析最近一周的销售数据",
  "context": {
    "user_id": "user_123",
    "session_id": "session_456"
  }
}
```

### 多智能体协作

#### 创建对话会话
```http
POST /api/v1/multi-agents/sessions
Content-Type: application/json
Authorization: Bearer <token>

{
  "agents": ["agent_1", "agent_2", "agent_3"],
  "topic": "产品需求分析",
  "max_rounds": 10,
  "termination_condition": "consensus"
}
```

#### 发送消息到会话
```http
POST /api/v1/multi-agents/sessions/{session_id}/messages
Content-Type: application/json
Authorization: Bearer <token>

{
  "content": "请分析这个产品的市场潜力",
  "attachments": [
    {
      "type": "document",
      "url": "https://..."
    }
  ]
}
```

### 工作流管理

#### 创建工作流
```http
POST /api/v1/workflows
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "数据处理流程",
  "nodes": [
    {
      "id": "node_1",
      "type": "agent",
      "agent_id": "agent_123",
      "config": {...}
    },
    {
      "id": "node_2",
      "type": "condition",
      "condition": "result > 0.8"
    }
  ],
  "edges": [
    {
      "source": "node_1",
      "target": "node_2"
    }
  ]
}
```

#### 执行工作流
```http
POST /api/v1/workflows/{workflow_id}/execute
Content-Type: application/json
Authorization: Bearer <token>

{
  "input": {...},
  "parameters": {
    "timeout": 300,
    "retry_on_failure": true
  }
}
```

### RAG系统

#### 创建向量索引
```http
POST /api/v1/rag/indexes
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "product_docs",
  "embedding_model": "text-embedding-3-large",
  "chunk_size": 500,
  "chunk_overlap": 50
}
```

#### 添加文档
```http
POST /api/v1/rag/indexes/{index_id}/documents
Content-Type: application/json
Authorization: Bearer <token>

{
  "documents": [
    {
      "content": "文档内容...",
      "metadata": {
        "title": "产品手册",
        "category": "documentation"
      }
    }
  ]
}
```

#### 查询
```http
POST /api/v1/rag/query
Content-Type: application/json
Authorization: Bearer <token>

{
  "query": "如何使用高级功能？",
  "index_id": "index_123",
  "top_k": 5,
  "filters": {
    "category": "documentation"
  }
}
```

### A/B测试实验

#### 创建实验
```http
POST /api/v1/experiments
Content-Type: application/json
Authorization: Bearer <token>

{
  "name": "首页改版测试",
  "description": "测试新版首页的转化率",
  "status": "draft",
  "start_date": "2024-01-01",
  "end_date": "2024-01-31",
  "variants": [
    {
      "id": "control",
      "name": "原版",
      "description": "当前版本",
      "traffic_percentage": 50,
      "is_control": true
    },
    {
      "id": "treatment",
      "name": "新版",
      "description": "改版后的版本",
      "traffic_percentage": 50,
      "is_control": false
    }
  ],
  "metrics": [
    {
      "name": "conversion_rate",
      "type": "proportion",
      "numerator_event": "purchase",
      "denominator_event": "visit"
    }
  ],
  "targeting_rules": {
    "include_users": [],
    "exclude_users": [],
    "user_attributes": {
      "country": ["CN", "US"]
    }
  }
}
```

#### 获取变体分配
```http
POST /api/v1/experiments/{experiment_id}/assignments
Content-Type: application/json
Authorization: Bearer <token>

{
  "user_id": "user_123",
  "attributes": {
    "country": "CN",
    "device": "mobile"
  }
}
```

响应:
```json
{
  "data": {
    "variant_id": "treatment",
    "experiment_id": "exp_123",
    "user_id": "user_123",
    "assigned_at": "2024-01-01T00:00:00Z"
  }
}
```

#### 上报事件
```http
POST /api/v1/events/track
Content-Type: application/json
Authorization: Bearer <token>

{
  "user_id": "user_123",
  "event_type": "purchase",
  "experiment_id": "exp_123",
  "variant_id": "treatment",
  "value": 99.99,
  "timestamp": "2024-01-01T00:00:00Z",
  "properties": {
    "product_id": "prod_456",
    "category": "electronics"
  }
}
```

### 统计分析

#### 获取实验结果
```http
GET /api/v1/experiments/{experiment_id}/results
Authorization: Bearer <token>
```

响应:
```json
{
  "data": {
    "experiment_id": "exp_123",
    "status": "running",
    "results": [
      {
        "metric": "conversion_rate",
        "control": {
          "value": 0.05,
          "sample_size": 10000,
          "confidence_interval": [0.046, 0.054]
        },
        "treatment": {
          "value": 0.06,
          "sample_size": 10000,
          "confidence_interval": [0.056, 0.064]
        },
        "lift": {
          "absolute": 0.01,
          "relative": 0.20,
          "confidence_interval": [0.15, 0.25]
        },
        "p_value": 0.001,
        "statistical_significance": true,
        "practical_significance": true
      }
    ],
    "sample_ratio_mismatch": {
      "detected": false,
      "p_value": 0.95
    }
  }
}
```

#### 计算样本量
```http
POST /api/v1/analysis/sample-size
Content-Type: application/json
Authorization: Bearer <token>

{
  "baseline_rate": 0.05,
  "minimum_detectable_effect": 0.01,
  "power": 0.8,
  "significance_level": 0.05,
  "test_type": "two_sided",
  "variants_count": 2
}
```

### 发布策略

#### 创建发布策略
```http
POST /api/v1/release-strategies
Content-Type: application/json
Authorization: Bearer <token>

{
  "experiment_id": "exp_123",
  "strategy_type": "canary",
  "config": {
    "initial_percentage": 5,
    "increment_percentage": 10,
    "increment_interval_hours": 24,
    "max_percentage": 100,
    "success_criteria": {
      "error_rate_threshold": 0.01,
      "p95_latency_ms": 500
    },
    "rollback_on_failure": true
  }
}
```

## 错误处理

### 错误码

| 错误码 | HTTP状态码 | 描述 |
|--------|-----------|------|
| INVALID_REQUEST | 400 | 请求参数无效 |
| UNAUTHORIZED | 401 | 未认证 |
| FORBIDDEN | 403 | 无权限 |
| NOT_FOUND | 404 | 资源不存在 |
| CONFLICT | 409 | 资源冲突 |
| RATE_LIMITED | 429 | 请求过于频繁 |
| INTERNAL_ERROR | 500 | 服务器内部错误 |
| SERVICE_UNAVAILABLE | 503 | 服务暂时不可用 |

### 错误响应示例
```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "Invalid experiment configuration",
    "details": {
      "field": "variants",
      "reason": "Traffic percentages must sum to 100"
    }
  },
  "request_id": "req_123456",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## 限流策略

API使用令牌桶算法进行限流：

- **默认限制**: 100 requests/minute
- **突发限制**: 200 requests
- **认证用户**: 1000 requests/minute
- **企业用户**: 10000 requests/minute

限流信息在响应头中返回：
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

## WebSocket接口

### 实时事件流
```javascript
const ws = new WebSocket('wss://api.ai-agent.com/ws/events');

ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    topics: ['experiment_exp_123', 'agent_updates'],
    auth_token: 'Bearer <token>'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### 消息类型

#### 实验更新
```json
{
  "type": "experiment_update",
  "experiment_id": "exp_123",
  "data": {
    "event": "metric_updated",
    "metric": "conversion_rate",
    "values": {...}
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### 智能体状态
```json
{
  "type": "agent_status",
  "agent_id": "agent_123",
  "status": "executing",
  "progress": 0.5,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## SDK支持

### Python SDK
```python
from ai_agent_sdk import Client

client = Client(api_key="your_api_key")

# 创建实验
experiment = client.experiments.create(
    name="首页测试",
    variants=[...],
    metrics=[...]
)

# 获取变体分配
assignment = client.experiments.get_assignment(
    experiment_id=experiment.id,
    user_id="user_123"
)

# 跟踪事件
client.events.track(
    user_id="user_123",
    event_type="purchase",
    experiment_id=experiment.id,
    variant_id=assignment.variant_id
)
```

### JavaScript SDK
```javascript
import { AIAgentClient } from '@ai-agent/sdk';

const client = new AIAgentClient({ apiKey: 'your_api_key' });

// 创建智能体会话
const session = await client.agents.createSession({
  agentId: 'agent_123',
  context: { userId: 'user_123' }
});

// 发送消息
const response = await session.sendMessage('分析这份数据');
console.log(response.output);
```

## 最佳实践

1. **使用幂等键**: 对于重要操作，使用Idempotency-Key头避免重复处理
2. **批量操作**: 使用批量接口减少请求次数
3. **异步处理**: 对于长时间运行的任务，使用异步接口
4. **错误重试**: 实现指数退避的重试策略
5. **版本控制**: 在请求头中指定API版本
6. **数据压缩**: 对大量数据使用gzip压缩

## 更新日志

### v1.0.0 (2024-01-01)
- 初始版本发布
- 支持智能体管理
- 多智能体协作
- RAG系统
- A/B测试实验平台
- 统计分析功能
- 发布策略管理