# SDK 文档 - AI智能体系统

## 目录
- [Python SDK](#python-sdk)
- [JavaScript SDK](#javascript-sdk)
- [REST API客户端](#rest-api客户端)
- [WebSocket客户端](#websocket客户端)
- [示例项目](#示例项目)
- [最佳实践](#最佳实践)

## Python SDK

### 安装

```bash
pip install ai-agent-sdk
```

### 快速开始

```python
from ai_agent_sdk import AIAgentClient

# 初始化客户端
client = AIAgentClient(
    api_key="your-api-key",
    base_url="https://api.ai-agent.com"
)

# 创建智能体
agent = client.agents.create(
    name="DataAnalyst",
    type="react", 
    description="数据分析专家",
    tools=["calculator", "database"],
    model="claude-3.5-sonnet"
)

# 执行任务
result = client.agents.execute(
    agent_id=agent.id,
    input="分析最近一周的销售数据",
    context={"user_id": "user_123"}
)

print(result.output)
```

### 智能体管理

#### 创建智能体

```python
from ai_agent_sdk.types import AgentConfig, ToolConfig

# 高级配置
config = AgentConfig(
    name="CustomerService",
    type="react",
    description="智能客服助手",
    system_prompt="""你是一个专业的客服助手，具有以下特点：
    1. 友善和耐心
    2. 专业的产品知识
    3. 快速解决问题的能力""",
    tools=[
        ToolConfig(
            name="knowledge_search",
            description="搜索知识库",
            parameters={
                "query": "string",
                "category": "string"
            }
        )
    ],
    model="claude-3.5-sonnet",
    temperature=0.7,
    max_iterations=5,
    timeout=300
)

agent = client.agents.create(config)
```

#### 管理智能体

```python
# 获取智能体列表
agents = client.agents.list(
    limit=20,
    status="active",
    type="react"
)

# 获取特定智能体
agent = client.agents.get(agent_id)

# 更新智能体
updated_agent = client.agents.update(
    agent_id,
    description="更新后的描述",
    temperature=0.5
)

# 删除智能体
client.agents.delete(agent_id)
```

#### 异步执行

```python
import asyncio
from ai_agent_sdk.async_client import AsyncAIAgentClient

async def main():
    async_client = AsyncAIAgentClient(api_key="your-api-key")
    
    # 异步执行任务
    task = await async_client.agents.execute_async(
        agent_id="agent_123",
        input="处理这个复杂任务",
        callback_url="https://your-app.com/webhook"
    )
    
    # 获取任务状态
    status = await async_client.tasks.get_status(task.id)
    
    # 等待任务完成
    result = await async_client.tasks.wait_for_completion(
        task.id, 
        timeout=600
    )

asyncio.run(main())
```

### 多智能体协作

```python
from ai_agent_sdk.multi_agent import MultiAgentSession

# 创建协作会话
session = client.multi_agents.create_session(
    name="产品分析团队",
    agents=[
        {"id": "market_analyst", "role": "市场分析师"},
        {"id": "tech_expert", "role": "技术专家"},
        {"id": "product_manager", "role": "产品经理"}
    ],
    session_type="group_chat",
    max_rounds=10,
    termination_condition="consensus"
)

# 发送消息
message = session.send_message(
    content="我们需要分析新产品的市场可行性",
    attachments=[
        {
            "type": "document",
            "url": "https://example.com/market-report.pdf"
        }
    ]
)

# 监听消息
for update in session.listen():
    print(f"[{update.agent_id}]: {update.content}")
    if update.type == "consensus_reached":
        break

# 获取会话摘要
summary = session.get_summary()
```

### 工作流管理

```python
from ai_agent_sdk.workflow import WorkflowBuilder

# 构建工作流
workflow = (WorkflowBuilder()
    .add_agent_node(
        node_id="analyze",
        agent_id="data_analyst", 
        config={"task": "分析数据"}
    )
    .add_condition_node(
        node_id="check_quality",
        condition="analyze.confidence > 0.8"
    )
    .add_human_node(
        node_id="review", 
        config={"reviewers": ["manager@company.com"]}
    )
    .connect("analyze", "check_quality")
    .connect("check_quality", "review", condition="high_confidence")
    .build()
)

# 创建工作流
wf = client.workflows.create(workflow)

# 执行工作流
execution = client.workflows.execute(
    workflow_id=wf.id,
    input_data={"dataset_url": "https://data.example.com/sales.csv"},
    parameters={"quality_threshold": 0.85}
)

# 监控执行状态
for status in client.workflows.monitor_execution(execution.id):
    print(f"当前节点: {status.current_node}")
    print(f"状态: {status.status}")
    if status.status in ["completed", "failed"]:
        break
```

### A/B测试实验

```python
from ai_agent_sdk.experiments import ExperimentBuilder

# 创建实验
experiment = (ExperimentBuilder()
    .set_name("智能体提示词优化")
    .set_description("测试新的提示词模板效果") 
    .add_variant("control", "原版提示词", 50, is_control=True)
    .add_variant("treatment", "优化提示词", 50)
    .add_metric("user_satisfaction", "proportion", 
                numerator_event="positive_feedback",
                denominator_event="total_interactions")
    .set_duration(days=14)
    .set_targeting(
        include_users={"user_type": ["premium", "enterprise"]},
        exclude_users={"user_id": ["test_user_1"]}
    )
    .build()
)

exp = client.experiments.create(experiment)

# 获取用户分配
assignment = client.experiments.get_assignment(
    experiment_id=exp.id,
    user_id="user_456",
    attributes={"device": "mobile", "region": "CN"}
)

# 跟踪事件
client.events.track(
    user_id="user_456",
    event_type="positive_feedback",
    experiment_id=exp.id,
    variant_id=assignment.variant_id,
    value=1.0,
    properties={"satisfaction_score": 4.5}
)

# 获取实验结果
results = client.experiments.get_results(exp.id)
print(f"转化率提升: {results.lift.relative:.2%}")
print(f"统计显著性: {results.statistical_significance}")
```

### RAG系统

```python
from ai_agent_sdk.rag import RAGClient

# 创建向量索引
index = client.rag.create_index(
    name="product_docs",
    description="产品文档知识库",
    embedding_model="text-embedding-3-large",
    chunk_size=500,
    chunk_overlap=50
)

# 批量添加文档
documents = [
    {
        "content": "产品使用说明...",
        "metadata": {
            "title": "用户手册",
            "category": "documentation",
            "version": "1.0"
        }
    },
    # 更多文档...
]

client.rag.add_documents(index.id, documents)

# 查询
results = client.rag.query(
    query="如何配置智能体参数？",
    index_id=index.id,
    top_k=5,
    filters={"category": "documentation"},
    rerank=True
)

for result in results.documents:
    print(f"相关度: {result.score:.3f}")
    print(f"内容: {result.content[:200]}...")
    print(f"来源: {result.metadata.get('title')}")
    print("---")
```

### 监控和分析

```python
# 获取智能体性能指标
metrics = client.monitoring.get_agent_metrics(
    agent_id="agent_123",
    start_time="2024-01-01",
    end_time="2024-01-31",
    metrics=["request_count", "success_rate", "avg_response_time"]
)

# 获取用户行为分析
user_analytics = client.analytics.get_user_behavior(
    user_id="user_456",
    timeframe="last_30_days"
)

# 获取系统健康状态
health = client.monitoring.get_health()
print(f"系统状态: {health.status}")
print(f"数据库: {health.services.database}")
print(f"Redis: {health.services.redis}")
```

### 错误处理

```python
from ai_agent_sdk.exceptions import (
    AIAgentAPIError,
    RateLimitError,
    AuthenticationError
)

try:
    result = client.agents.execute(agent_id, user_input)
except RateLimitError as e:
    print(f"请求过于频繁，请等待 {e.retry_after} 秒")
    time.sleep(e.retry_after)
    result = client.agents.execute(agent_id, user_input)
    
except AuthenticationError:
    print("认证失败，请检查API密钥")
    
except AIAgentAPIError as e:
    print(f"API错误: {e.error_code} - {e.message}")
    if e.details:
        print(f"详细信息: {e.details}")
```

## JavaScript SDK

### 安装

```bash
npm install @ai-agent/sdk
```

### 基础用法

```javascript
import { AIAgentClient } from '@ai-agent/sdk';

// 初始化客户端
const client = new AIAgentClient({
  apiKey: 'your-api-key',
  baseUrl: 'https://api.ai-agent.com'
});

// 创建智能体
const agent = await client.agents.create({
  name: 'SalesAssistant',
  type: 'react',
  description: '销售助手',
  tools: ['crm_lookup', 'email_sender'],
  model: 'claude-3.5-sonnet'
});

// 执行任务
const result = await client.agents.execute({
  agentId: agent.id,
  input: '查找客户A的联系信息',
  context: { userId: 'user_123' }
});

console.log(result.output);
```

### 实时对话

```javascript
import { ConversationManager } from '@ai-agent/sdk';

class ChatInterface {
  constructor() {
    this.conversation = new ConversationManager({
      agentId: 'agent_123',
      userId: 'user_456'
    });
    
    this.setupEventHandlers();
  }

  setupEventHandlers() {
    this.conversation.on('message', (message) => {
      this.displayMessage(message);
    });

    this.conversation.on('thinking', () => {
      this.showThinkingIndicator();
    });

    this.conversation.on('toolCall', (toolCall) => {
      this.showToolExecution(toolCall);
    });

    this.conversation.on('error', (error) => {
      this.showError(error);
    });
  }

  async sendMessage(text) {
    try {
      const response = await this.conversation.sendMessage(text);
      return response;
    } catch (error) {
      console.error('发送消息失败:', error);
    }
  }

  displayMessage(message) {
    const messageEl = document.createElement('div');
    messageEl.className = `message ${message.role}`;
    messageEl.textContent = message.content;
    document.getElementById('chat-container').appendChild(messageEl);
  }
}

// 使用
const chat = new ChatInterface();
```

### 多智能体协作（前端）

```javascript
import { MultiAgentSession } from '@ai-agent/sdk';

class CollaborationInterface {
  constructor() {
    this.session = null;
  }

  async createSession() {
    this.session = await client.multiAgents.createSession({
      name: '项目规划团队',
      agents: [
        { id: 'project_manager', role: '项目经理' },
        { id: 'tech_lead', role: '技术负责人' },
        { id: 'designer', role: '设计师' }
      ],
      sessionType: 'group_chat',
      maxRounds: 15
    });

    // 监听会话更新
    this.session.onMessage((message) => {
      this.displayAgentMessage(message);
    });

    this.session.onStatusChange((status) => {
      this.updateSessionStatus(status);
    });
  }

  async sendMessage(content) {
    if (!this.session) return;

    const message = await this.session.sendMessage({
      content,
      attachments: this.getSelectedFiles()
    });

    this.displayUserMessage(message);
  }

  displayAgentMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.innerHTML = `
      <div class="agent-message">
        <div class="agent-avatar">
          <img src="/avatars/${message.agentId}.png" alt="${message.agentName}">
        </div>
        <div class="message-content">
          <div class="agent-name">${message.agentName}</div>
          <div class="message-text">${message.content}</div>
          <div class="message-time">${new Date(message.timestamp).toLocaleTimeString()}</div>
        </div>
      </div>
    `;
    document.getElementById('messages').appendChild(messageDiv);
  }
}
```

### A/B测试集成

```javascript
import { ExperimentClient } from '@ai-agent/sdk';

class ExperimentManager {
  constructor() {
    this.client = new ExperimentClient({
      apiKey: 'your-api-key'
    });
  }

  async initializeUser(userId, attributes = {}) {
    // 获取用户参与的所有实验
    const experiments = await this.client.getActiveExperiments();
    
    const assignments = {};
    for (const experiment of experiments) {
      const assignment = await this.client.getAssignment({
        experimentId: experiment.id,
        userId,
        attributes
      });
      
      assignments[experiment.name] = assignment;
    }

    return assignments;
  }

  async trackEvent(userId, eventType, properties = {}) {
    // 获取用户当前的实验分配
    const assignments = await this.getUserAssignments(userId);
    
    // 为每个实验跟踪事件
    for (const [expName, assignment] of Object.entries(assignments)) {
      await this.client.trackEvent({
        userId,
        eventType,
        experimentId: assignment.experimentId,
        variantId: assignment.variantId,
        properties: {
          ...properties,
          experiment_name: expName
        }
      });
    }
  }

  // 功能开关
  isFeatureEnabled(userId, featureName) {
    const assignment = this.assignments[featureName];
    return assignment && assignment.variantId !== 'control';
  }

  // 获取变体配置
  getVariantConfig(userId, experimentName) {
    const assignment = this.assignments[experimentName];
    if (!assignment) return null;
    
    return assignment.config;
  }
}

// 使用示例
const experimentManager = new ExperimentManager();

// 页面加载时初始化
document.addEventListener('DOMContentLoaded', async () => {
  const userId = getCurrentUserId();
  const userAttributes = {
    device: 'mobile',
    region: 'CN',
    userType: 'premium'
  };

  const assignments = await experimentManager.initializeUser(userId, userAttributes);
  
  // 根据实验分配调整界面
  if (experimentManager.isFeatureEnabled(userId, 'new_ui_design')) {
    document.body.classList.add('new-ui');
  }

  // 跟踪页面访问事件
  await experimentManager.trackEvent(userId, 'page_view', {
    page: window.location.pathname
  });
});
```

### React Hooks

```jsx
import React, { useState, useEffect } from 'react';
import { useAIAgent, useExperiment } from '@ai-agent/react';

// 智能体对话组件
function ChatComponent({ agentId }) {
  const { 
    messages, 
    sendMessage, 
    isLoading, 
    error 
  } = useAIAgent({ agentId });

  const [input, setInput] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    await sendMessage(input);
    setInput('');
  };

  return (
    <div className="chat-container">
      <div className="messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <div className="content">{msg.content}</div>
            <div className="timestamp">{msg.timestamp}</div>
          </div>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="input-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="输入消息..."
          disabled={isLoading}
        />
        <button type="submit" disabled={isLoading}>
          {isLoading ? '发送中...' : '发送'}
        </button>
      </form>

      {error && (
        <div className="error">
          错误: {error.message}
        </div>
      )}
    </div>
  );
}

// 实验功能组件
function FeatureComponent() {
  const { 
    assignment, 
    isLoading, 
    trackEvent 
  } = useExperiment('homepage_redesign');

  const handleClick = () => {
    trackEvent('button_click', {
      button_type: 'cta',
      location: 'homepage'
    });
  };

  if (isLoading) {
    return <div>加载中...</div>;
  }

  // 根据实验分配渲染不同版本
  if (assignment?.variantId === 'new_design') {
    return (
      <button 
        className="cta-button-new" 
        onClick={handleClick}
      >
        立即开始 (新版)
      </button>
    );
  }

  return (
    <button 
      className="cta-button-old" 
      onClick={handleClick}
    >
      立即开始
    </button>
  );
}
```

## REST API客户端

### 通用HTTP客户端

```python
import requests
from typing import Dict, Any, Optional

class APIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'User-Agent': 'AI-Agent-Client/1.0'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
        timeout: int = 30
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/api/v1/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                params=params,
                timeout=timeout
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            self._handle_http_error(e.response)
        except requests.exceptions.RequestException as e:
            raise APIClientError(f"请求失败: {str(e)}")

    def _handle_http_error(self, response):
        try:
            error_data = response.json()
            error_msg = error_data.get('error', {}).get('message', '未知错误')
        except:
            error_msg = f"HTTP {response.status_code} 错误"
        
        if response.status_code == 401:
            raise AuthenticationError("认证失败，请检查API密钥")
        elif response.status_code == 429:
            retry_after = response.headers.get('Retry-After', 60)
            raise RateLimitError(f"请求过于频繁，请等待{retry_after}秒", retry_after)
        else:
            raise APIClientError(error_msg)

# 使用示例
client = APIClient('https://api.ai-agent.com', 'your-api-key')

# 创建智能体
agent = client.request('POST', '/agents', {
    'name': 'TestAgent',
    'type': 'react',
    'description': '测试智能体'
})

# 执行任务
result = client.request('POST', f'/agents/{agent["id"]}/execute', {
    'input': '你好，世界！',
    'context': {'user_id': 'user_123'}
})
```

### cURL示例

```bash
# 创建智能体
curl -X POST "https://api.ai-agent.com/api/v1/agents" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "DataAnalyst",
    "type": "react",
    "description": "数据分析助手",
    "tools": ["calculator", "database"],
    "model": "claude-3.5-sonnet"
  }'

# 执行智能体任务
curl -X POST "https://api.ai-agent.com/api/v1/agents/agent_123/execute" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "分析销售数据",
    "context": {"user_id": "user_456"}
  }'

# 创建A/B测试实验
curl -X POST "https://api.ai-agent.com/api/v1/experiments" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "UI优化测试",
    "description": "测试新界面效果",
    "variants": [
      {
        "id": "control",
        "name": "原版",
        "traffic_percentage": 50,
        "is_control": true
      },
      {
        "id": "treatment", 
        "name": "新版",
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
    ]
  }'

# 获取用户分配
curl -X POST "https://api.ai-agent.com/api/v1/experiments/exp_123/assignments" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_789",
    "attributes": {
      "device": "mobile",
      "region": "CN"
    }
  }'

# 跟踪事件
curl -X POST "https://api.ai-agent.com/api/v1/events/track" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_789",
    "event_type": "purchase", 
    "experiment_id": "exp_123",
    "variant_id": "treatment",
    "value": 99.99,
    "properties": {
      "product_id": "prod_456",
      "category": "electronics"
    }
  }'
```

## WebSocket客户端

### Python WebSocket客户端

```python
import websocket
import json
import threading

class AIAgentWebSocketClient:
    def __init__(self, ws_url: str, api_key: str):
        self.ws_url = ws_url
        self.api_key = api_key
        self.ws = None
        self.is_connected = False
        self.callbacks = {}

    def connect(self):
        self.ws = websocket.WebSocketApp(
            f"{self.ws_url}/ws",
            header={"Authorization": f"Bearer {self.api_key}"},
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        
        # 在新线程中运行
        self.ws_thread = threading.Thread(target=self.ws.run_forever)
        self.ws_thread.daemon = True
        self.ws_thread.start()

    def _on_open(self, ws):
        self.is_connected = True
        print("WebSocket连接已建立")

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            message_type = data.get('type')
            
            if message_type in self.callbacks:
                self.callbacks[message_type](data)
                
        except Exception as e:
            print(f"处理消息时出错: {e}")

    def _on_error(self, ws, error):
        print(f"WebSocket错误: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        self.is_connected = False
        print("WebSocket连接已关闭")

    def subscribe(self, topics: list, callback):
        """订阅主题"""
        if not self.is_connected:
            raise Exception("WebSocket未连接")
            
        subscribe_msg = {
            "type": "subscribe",
            "topics": topics
        }
        self.ws.send(json.dumps(subscribe_msg))
        
        # 注册回调
        for topic in topics:
            self.callbacks[topic] = callback

    def send_message(self, message_type: str, data: dict):
        """发送消息"""
        if not self.is_connected:
            raise Exception("WebSocket未连接")
            
        message = {
            "type": message_type,
            "data": data
        }
        self.ws.send(json.dumps(message))

# 使用示例
def on_agent_update(data):
    print(f"智能体状态更新: {data}")

def on_experiment_update(data):
    print(f"实验数据更新: {data}")

# 连接WebSocket
ws_client = AIAgentWebSocketClient(
    ws_url="wss://api.ai-agent.com",
    api_key="your-api-key"
)
ws_client.connect()

# 订阅更新
ws_client.subscribe(["agent_status", "experiment_metrics"], on_agent_update)
ws_client.subscribe(["experiment_update"], on_experiment_update)
```

### JavaScript WebSocket客户端

```javascript
class AIAgentWebSocket {
  constructor(wsUrl, apiKey) {
    this.wsUrl = wsUrl;
    this.apiKey = apiKey;
    this.ws = null;
    this.callbacks = new Map();
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
  }

  connect() {
    const wsUrl = `${this.wsUrl}/ws?token=${this.apiKey}`;
    this.ws = new WebSocket(wsUrl);

    this.ws.onopen = (event) => {
      this.isConnected = true;
      this.reconnectAttempts = 0;
      console.log('WebSocket连接已建立');
      this.emit('connected', event);
    };

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        this.handleMessage(data);
      } catch (error) {
        console.error('解析消息失败:', error);
      }
    };

    this.ws.onclose = (event) => {
      this.isConnected = false;
      console.log('WebSocket连接已关闭');
      this.emit('disconnected', event);
      
      // 自动重连
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        setTimeout(() => {
          this.reconnectAttempts++;
          console.log(`尝试重连... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
          this.connect();
        }, 1000 * this.reconnectAttempts);
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket错误:', error);
      this.emit('error', error);
    };
  }

  handleMessage(data) {
    const { type, ...payload } = data;
    
    if (this.callbacks.has(type)) {
      const callback = this.callbacks.get(type);
      callback(payload);
    }
    
    // 触发通用消息事件
    this.emit('message', data);
  }

  subscribe(topics, callback) {
    if (!this.isConnected) {
      throw new Error('WebSocket未连接');
    }

    // 发送订阅消息
    this.send('subscribe', { topics });

    // 注册回调
    topics.forEach(topic => {
      this.callbacks.set(topic, callback);
    });
  }

  send(type, data) {
    if (!this.isConnected) {
      throw new Error('WebSocket未连接');
    }

    const message = JSON.stringify({ type, ...data });
    this.ws.send(message);
  }

  on(event, callback) {
    if (!this.eventCallbacks) {
      this.eventCallbacks = new Map();
    }
    this.eventCallbacks.set(event, callback);
  }

  emit(event, data) {
    if (this.eventCallbacks && this.eventCallbacks.has(event)) {
      const callback = this.eventCallbacks.get(event);
      callback(data);
    }
  }

  disconnect() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

// 使用示例
const wsClient = new AIAgentWebSocket('wss://api.ai-agent.com', 'your-api-key');

wsClient.on('connected', () => {
  // 订阅智能体状态更新
  wsClient.subscribe(['agent_status'], (data) => {
    console.log('智能体状态:', data);
    updateAgentStatus(data.agent_id, data.status);
  });

  // 订阅实验指标更新
  wsClient.subscribe(['experiment_metrics'], (data) => {
    console.log('实验指标:', data);
    updateExperimentDashboard(data);
  });
});

wsClient.connect();
```

## 示例项目

### 完整的聊天机器人示例

```python
"""
智能客服聊天机器人
功能：
1. 多轮对话
2. 工具调用
3. 会话记录
4. A/B测试
"""

import asyncio
from ai_agent_sdk import AIAgentClient
from ai_agent_sdk.experiments import ExperimentManager

class CustomerServiceBot:
    def __init__(self, api_key: str):
        self.client = AIAgentClient(api_key=api_key)
        self.experiment_manager = ExperimentManager(api_key)
        
        # 创建客服智能体
        self.agent = self.client.agents.create({
            'name': 'CustomerService',
            'type': 'react',
            'description': '智能客服助手',
            'tools': ['knowledge_search', 'order_lookup', 'ticket_create'],
            'system_prompt': """你是一个专业的客服助手。请遵循以下原则：
            1. 友善和耐心地回答用户问题
            2. 优先使用知识库搜索相关信息
            3. 如果无法解决问题，创建工单转人工处理
            4. 记住用户的对话上下文"""
        })
        
    async def start_conversation(self, user_id: str, user_attributes: dict = None):
        """开始新对话"""
        # 获取A/B测试分配
        assignment = await self.experiment_manager.get_assignment(
            experiment_name='customer_service_optimization',
            user_id=user_id,
            attributes=user_attributes or {}
        )
        
        # 根据实验分配调整智能体参数
        agent_config = self._get_agent_config(assignment)
        
        # 创建对话会话
        session = self.client.conversations.create_session(
            agent_id=self.agent.id,
            user_id=user_id,
            config=agent_config
        )
        
        return ChatSession(session, self.experiment_manager, assignment)
    
    def _get_agent_config(self, assignment):
        """根据实验分配调整配置"""
        if assignment and assignment.variant_id == 'proactive':
            return {
                'temperature': 0.8,
                'system_prompt_suffix': '主动询问用户需求，提供个性化建议。'
            }
        else:
            return {
                'temperature': 0.6,
                'system_prompt_suffix': '等待用户提问后再回答。'
            }

class ChatSession:
    def __init__(self, session, experiment_manager, assignment):
        self.session = session
        self.experiment_manager = experiment_manager
        self.assignment = assignment
        self.conversation_history = []
    
    async def send_message(self, user_input: str):
        """发送用户消息"""
        # 跟踪用户输入事件
        await self.experiment_manager.track_event(
            user_id=self.session.user_id,
            event_type='user_message',
            experiment_id=self.assignment.experiment_id if self.assignment else None,
            variant_id=self.assignment.variant_id if self.assignment else None,
            properties={'message_length': len(user_input)}
        )
        
        # 发送消息给智能体
        response = await self.session.send_message(user_input)
        
        # 记录对话历史
        self.conversation_history.append({
            'user': user_input,
            'assistant': response.content,
            'timestamp': response.timestamp,
            'tools_used': response.tools_used
        })
        
        return response
    
    async def rate_response(self, rating: int, feedback: str = None):
        """用户评分"""
        await self.experiment_manager.track_event(
            user_id=self.session.user_id,
            event_type='response_rating',
            experiment_id=self.assignment.experiment_id if self.assignment else None,
            variant_id=self.assignment.variant_id if self.assignment else None,
            properties={
                'rating': rating,
                'feedback': feedback,
                'conversation_length': len(self.conversation_history)
            }
        )
    
    def get_conversation_summary(self):
        """获取对话摘要"""
        return {
            'total_messages': len(self.conversation_history),
            'user_satisfaction': self._calculate_satisfaction(),
            'tools_used': self._get_tools_usage(),
            'duration': self._calculate_duration()
        }

# 使用示例
async def main():
    bot = CustomerServiceBot('your-api-key')
    
    # 开始对话
    session = await bot.start_conversation(
        user_id='user_123',
        user_attributes={'device': 'mobile', 'vip_level': 'gold'}
    )
    
    # 模拟对话
    response1 = await session.send_message("我的订单什么时候发货？")
    print(f"助手: {response1.content}")
    
    response2 = await session.send_message("订单号是ORD123456")
    print(f"助手: {response2.content}")
    
    # 用户评分
    await session.rate_response(5, "回答很有帮助")
    
    # 获取对话摘要
    summary = session.get_conversation_summary()
    print(f"对话摘要: {summary}")

if __name__ == "__main__":
    asyncio.run(main())
```

### React前端集成示例

```jsx
// CustomerServiceChat.jsx
import React, { useState, useEffect, useRef } from 'react';
import { AIAgentClient } from '@ai-agent/sdk';

const CustomerServiceChat = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [session, setSession] = useState(null);
  const messagesEndRef = useRef(null);
  
  const client = new AIAgentClient({
    apiKey: process.env.REACT_APP_AI_AGENT_API_KEY
  });

  useEffect(() => {
    initializeChat();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const initializeChat = async () => {
    try {
      const chatSession = await client.conversations.createSession({
        agentId: 'customer-service-agent',
        userId: getCurrentUserId(),
        context: {
          userAttributes: getUserAttributes()
        }
      });
      
      setSession(chatSession);
      
      // 添加欢迎消息
      setMessages([{
        role: 'assistant',
        content: '您好！我是智能客服助手，有什么可以帮助您的吗？',
        timestamp: new Date().toISOString()
      }]);
      
    } catch (error) {
      console.error('初始化聊天失败:', error);
    }
  };

  const sendMessage = async () => {
    if (!inputValue.trim() || !session) return;
    
    const userMessage = {
      role: 'user',
      content: inputValue,
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await session.sendMessage(inputValue);
      
      const assistantMessage = {
        role: 'assistant',
        content: response.content,
        timestamp: response.timestamp,
        toolsUsed: response.toolsUsed
      };
      
      setMessages(prev => [...prev, assistantMessage]);
      
    } catch (error) {
      console.error('发送消息失败:', error);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: '抱歉，我暂时无法回答您的问题，请稍后再试。',
        timestamp: new Date().toISOString(),
        isError: true
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleRating = async (messageIndex, rating) => {
    try {
      await session.rateResponse(rating);
      
      // 更新消息状态
      setMessages(prev => prev.map((msg, index) => 
        index === messageIndex 
          ? { ...msg, userRating: rating }
          : msg
      ));
      
    } catch (error) {
      console.error('评分失败:', error);
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const getCurrentUserId = () => {
    // 获取当前用户ID的逻辑
    return 'user_' + Math.random().toString(36).substr(2, 9);
  };

  const getUserAttributes = () => {
    // 获取用户属性的逻辑
    return {
      device: /Mobi|Android/i.test(navigator.userAgent) ? 'mobile' : 'desktop',
      userAgent: navigator.userAgent,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
    };
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <h3>智能客服</h3>
        <div className="status-indicator online">在线</div>
      </div>
      
      <div className="messages-container">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <div className="message-content">
              {message.content}
              
              {message.toolsUsed && message.toolsUsed.length > 0 && (
                <div className="tools-used">
                  <small>使用了工具: {message.toolsUsed.join(', ')}</small>
                </div>
              )}
              
              {message.role === 'assistant' && !message.isError && (
                <div className="message-actions">
                  <button 
                    onClick={() => handleRating(index, 1)}
                    className={message.userRating === 1 ? 'active' : ''}
                  >
                    👍
                  </button>
                  <button 
                    onClick={() => handleRating(index, -1)}
                    className={message.userRating === -1 ? 'active' : ''}
                  >
                    👎
                  </button>
                </div>
              )}
            </div>
            
            <div className="message-timestamp">
              {new Date(message.timestamp).toLocaleTimeString()}
            </div>
          </div>
        ))}
        
        {isLoading && (
          <div className="message assistant">
            <div className="typing-indicator">
              <span></span>
              <span></span>
              <span></span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>
      
      <div className="input-container">
        <input
          type="text"
          value={inputValue}
          onChange={(e) => setInputValue(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
          placeholder="输入您的问题..."
          disabled={isLoading}
        />
        <button onClick={sendMessage} disabled={isLoading || !inputValue.trim()}>
          发送
        </button>
      </div>
    </div>
  );
};

export default CustomerServiceChat;
```

## 最佳实践

### 1. 错误处理和重试

```python
import time
import random
from functools import wraps

def retry_on_failure(max_retries=3, backoff_factor=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (RateLimitError, ConnectionError) as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    # 指数退避
                    wait_time = backoff_factor ** attempt + random.uniform(0, 1)
                    time.sleep(wait_time)
                    
            return func(*args, **kwargs)
        return wrapper
    return decorator

@retry_on_failure(max_retries=3)
def execute_agent_task(client, agent_id, user_input):
    return client.agents.execute(agent_id, user_input)
```

### 2. 连接池管理

```python
import asyncio
import aiohttp
from typing import Optional

class AsyncAPIClient:
    def __init__(self, api_key: str, max_connections: int = 100):
        self.api_key = api_key
        self.base_url = "https://api.ai-agent.com/api/v1"
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=20,
            keepalive_timeout=30
        )
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            headers={'Authorization': f'Bearer {self.api_key}'},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def execute_multiple_tasks(self, tasks):
        """并发执行多个任务"""
        semaphore = asyncio.Semaphore(10)  # 限制并发数
        
        async def execute_single_task(task):
            async with semaphore:
                async with self.session.post(
                    f"{self.base_url}/agents/{task['agent_id']}/execute",
                    json={'input': task['input']}
                ) as response:
                    return await response.json()
        
        results = await asyncio.gather(
            *[execute_single_task(task) for task in tasks],
            return_exceptions=True
        )
        
        return results

# 使用示例
async def main():
    tasks = [
        {'agent_id': 'agent1', 'input': 'task 1'},
        {'agent_id': 'agent2', 'input': 'task 2'},
        # ... 更多任务
    ]
    
    async with AsyncAPIClient('your-api-key') as client:
        results = await client.execute_multiple_tasks(tasks)
        
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Task {i} failed: {result}")
        else:
            print(f"Task {i} result: {result}")
```

### 3. 缓存策略

```python
import redis
import json
import hashlib
from functools import wraps

class CacheManager:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)

    def cache_result(self, ttl: int = 3600, key_prefix: str = "ai_agent"):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                # 生成缓存键
                cache_key = self._generate_cache_key(key_prefix, func.__name__, args, kwargs)
                
                # 尝试从缓存获取
                cached_result = self.redis_client.get(cache_key)
                if cached_result:
                    return json.loads(cached_result)
                
                # 执行函数并缓存结果
                result = func(*args, **kwargs)
                self.redis_client.setex(
                    cache_key, 
                    ttl, 
                    json.dumps(result, default=str)
                )
                
                return result
            return wrapper
        return decorator

    def _generate_cache_key(self, prefix, func_name, args, kwargs):
        # 创建唯一的缓存键
        key_data = f"{prefix}:{func_name}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_data.encode()).hexdigest()

# 使用示例
cache_manager = CacheManager('redis://localhost:6379')

@cache_manager.cache_result(ttl=1800)  # 缓存30分钟
def get_agent_response(agent_id, user_input):
    return client.agents.execute(agent_id, user_input)
```

### 4. 监控和日志

```python
import logging
import time
from contextvars import ContextVar
from typing import Optional

# 请求上下文
request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)

class AIAgentLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # 创建处理器
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s - RequestID: %(request_id)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_with_context(self, level: str, message: str, **kwargs):
        extra = {'request_id': request_id.get() or 'unknown'}
        extra.update(kwargs)
        getattr(self.logger, level)(message, extra=extra)

class PerformanceMonitor:
    def __init__(self, logger: AIAgentLogger):
        self.logger = logger

    def monitor_execution(self, operation_name: str):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    
                    self.logger.log_with_context(
                        'info',
                        f"{operation_name} completed successfully",
                        duration=duration,
                        operation=operation_name
                    )
                    
                    return result
                    
                except Exception as e:
                    duration = time.time() - start_time
                    
                    self.logger.log_with_context(
                        'error',
                        f"{operation_name} failed: {str(e)}",
                        duration=duration,
                        operation=operation_name,
                        error=str(e)
                    )
                    
                    raise
                    
            return wrapper
        return decorator

# 使用示例
logger = AIAgentLogger('ai_agent_client')
monitor = PerformanceMonitor(logger)

@monitor.monitor_execution('agent_execution')
def execute_with_monitoring(agent_id, user_input):
    # 设置请求ID
    request_id.set(f"req_{int(time.time() * 1000)}")
    
    return client.agents.execute(agent_id, user_input)
```

### 5. 配置管理

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class AIAgentConfig:
    api_key: str
    base_url: str = "https://api.ai-agent.com"
    timeout: int = 30
    max_retries: int = 3
    rate_limit_per_minute: int = 100
    cache_ttl: int = 3600
    log_level: str = "INFO"
    
    @classmethod
    def from_env(cls) -> 'AIAgentConfig':
        return cls(
            api_key=os.getenv('AI_AGENT_API_KEY', ''),
            base_url=os.getenv('AI_AGENT_BASE_URL', cls.base_url),
            timeout=int(os.getenv('AI_AGENT_TIMEOUT', cls.timeout)),
            max_retries=int(os.getenv('AI_AGENT_MAX_RETRIES', cls.max_retries)),
            rate_limit_per_minute=int(os.getenv('AI_AGENT_RATE_LIMIT', cls.rate_limit_per_minute)),
            cache_ttl=int(os.getenv('AI_AGENT_CACHE_TTL', cls.cache_ttl)),
            log_level=os.getenv('AI_AGENT_LOG_LEVEL', cls.log_level)
        )
    
    def validate(self):
        if not self.api_key:
            raise ValueError("API密钥不能为空")
        if self.timeout <= 0:
            raise ValueError("超时时间必须大于0")

# 使用配置
config = AIAgentConfig.from_env()
config.validate()

client = AIAgentClient(
    api_key=config.api_key,
    base_url=config.base_url,
    timeout=config.timeout
)
```

这些文档和示例提供了完整的SDK使用指南，涵盖了从基础使用到高级特性的各个方面。开发者可以根据自己的需求选择合适的集成方式和最佳实践。