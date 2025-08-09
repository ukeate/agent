# API Specification

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
