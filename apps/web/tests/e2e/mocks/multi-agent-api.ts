import { Page } from '@playwright/test'

export async function setupMultiAgentApiMocks(page: Page) {
  // Mock agent status API
  await page.route('**/api/v1/multi-agent/agents', route => {
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify([
        {
          id: 'agent-1',
          name: '代码专家',
          role: 'code_expert',
          status: 'active',
          capabilities: ['代码生成', '代码审查', '性能优化'],
          configuration: {
            model: 'gpt-4o-mini',
            temperature: 0.1,
            max_tokens: 2000,
            tools: ['code_analyzer'],
            system_prompt: '你是一位专业的软件开发专家',
          },
          created_at: '2025-01-01T00:00:00Z',
          updated_at: '2025-01-01T00:00:00Z',
        },
        {
          id: 'agent-2',
          name: '架构师',
          role: 'architect',
          status: 'active',
          capabilities: ['系统设计', '架构规划', '技术选型'],
          configuration: {
            model: 'gpt-4o-mini',
            temperature: 0.2,
            max_tokens: 2000,
            tools: ['system_analyzer'],
            system_prompt: '你是一位资深的系统架构师',
          },
          created_at: '2025-01-01T00:00:00Z',
          updated_at: '2025-01-01T00:00:00Z',
        },
        {
          id: 'agent-3',
          name: '文档专家',
          role: 'doc_expert',
          status: 'active',
          capabilities: ['文档编写', '技术写作', '知识整理'],
          configuration: {
            model: 'gpt-4o-mini',
            temperature: 0.3,
            max_tokens: 2000,
            tools: ['doc_generator'],
            system_prompt: '你是一位专业的技术文档专家',
          },
          created_at: '2025-01-01T00:00:00Z',
          updated_at: '2025-01-01T00:00:00Z',
        },
      ]),
    })
  })

  // Mock conversation creation API
  await page.route('**/api/v1/multi-agent/conversation', route => {
    if (route.request().method() === 'POST') {
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          session_id: 'session-12345',
          status: 'active',
          participants: [
            {
              name: '代码专家',
              role: 'code_expert',
              status: 'active',
            },
            {
              name: '架构师',
              role: 'architect',
              status: 'active',
            },
            {
              name: '文档专家',
              role: 'doc_expert',
              status: 'active',
            },
          ],
          config: {
            max_rounds: 10,
            timeout_seconds: 300,
            auto_reply: true,
          },
          created_at: '2025-01-01T00:00:00Z',
          updated_at: '2025-01-01T00:00:00Z',
          message_count: 0,
          round_count: 0,
        }),
      })
    }
  })

  // Mock conversation status API
  await page.route('**/api/v1/multi-agent/conversation/*/status', route => {
    route.fulfill({
      contentType: 'application/json',
      body: JSON.stringify({
        session_id: 'session-12345',
        status: 'active',
        message_count: 3,
        round_count: 1,
        participants: [
          {
            name: '代码专家',
            role: 'code_expert',
            status: 'active',
          },
          {
            name: '架构师',
            role: 'architect',
            status: 'busy',
          },
          {
            name: '文档专家',
            role: 'doc_expert',
            status: 'idle',
          },
        ],
        current_speaker: '架构师',
        config: {
          max_rounds: 10,
          timeout_seconds: 300,
          auto_reply: true,
        },
        created_at: '2025-01-01T00:00:00Z',
        updated_at: '2025-01-01T00:00:00Z',
      }),
    })
  })

  // Mock pause conversation API
  await page.route('**/api/v1/multi-agent/conversation/*/pause', route => {
    if (route.request().method() === 'POST') {
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          session_id: 'session-12345',
          status: 'paused',
          message: '会话已暂停',
        }),
      })
    }
  })

  // Mock resume conversation API
  await page.route('**/api/v1/multi-agent/conversation/*/resume', route => {
    if (route.request().method() === 'POST') {
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          session_id: 'session-12345',
          status: 'active',
          message: '会话已恢复',
        }),
      })
    }
  })

  // Mock terminate conversation API
  await page.route('**/api/v1/multi-agent/conversation/*/terminate', route => {
    if (route.request().method() === 'POST') {
      route.fulfill({
        contentType: 'application/json',
        body: JSON.stringify({
          session_id: 'session-12345',
          status: 'terminated',
          message: '会话已终止',
          summary: {
            total_rounds: 2,
            total_messages: 6,
            duration_minutes: 5,
            termination_reason: '用户终止',
            key_decisions: [],
            participant_contributions: [],
          },
        }),
      })
    }
  })

  // Mock WebSocket connection (simulate server-sent events)
  await page.addInitScript(() => {
    // Override WebSocket constructor for testing
    const originalWebSocket = window.WebSocket

    window.WebSocket = class MockWebSocket extends EventTarget {
      url: string
      readyState: number = 1 // OPEN

      constructor(url: string) {
        super()
        this.url = url

        // Simulate connection opening
        setTimeout(() => {
          this.dispatchEvent(new Event('open'))

          // Simulate incoming messages
          setTimeout(() => {
            this.dispatchEvent(
              new MessageEvent('message', {
                data: JSON.stringify({
                  type: 'agent_message',
                  data: {
                    id: 'msg-1',
                    role: 'assistant',
                    sender: '代码专家',
                    content:
                      '我认为我们应该使用微服务架构来提高系统的可扩展性。',
                    timestamp: new Date().toISOString(),
                    round: 1,
                  },
                }),
              })
            )
          }, 1000)

          setTimeout(() => {
            this.dispatchEvent(
              new MessageEvent('message', {
                data: JSON.stringify({
                  type: 'agent_message',
                  data: {
                    id: 'msg-2',
                    role: 'assistant',
                    sender: '架构师',
                    content:
                      '同意代码专家的观点。我建议使用容器化部署和服务网格。',
                    timestamp: new Date().toISOString(),
                    round: 1,
                  },
                }),
              })
            )
          }, 2000)

          setTimeout(() => {
            this.dispatchEvent(
              new MessageEvent('message', {
                data: JSON.stringify({
                  type: 'agent_message',
                  data: {
                    id: 'msg-3',
                    role: 'assistant',
                    sender: '文档专家',
                    content: '我来整理架构文档和部署指南。',
                    timestamp: new Date().toISOString(),
                    round: 1,
                  },
                }),
              })
            )
          }, 3000)

          // Simulate status updates
          setTimeout(() => {
            this.dispatchEvent(
              new MessageEvent('message', {
                data: JSON.stringify({
                  type: 'status_update',
                  data: {
                    agent_id: 'agent-2',
                    status: 'speaking',
                    message: '架构师正在发言...',
                  },
                }),
              })
            )
          }, 1500)
        }, 100)
      }

      send(data: string) {
        // Mock send - do nothing in test
        console.log('Mock WebSocket send:', data)
      }

      close() {
        this.readyState = 3 // CLOSED
        this.dispatchEvent(new Event('close'))
      }

      // Add required properties
      static readonly CONNECTING = 0
      static readonly OPEN = 1
      static readonly CLOSING = 2
      static readonly CLOSED = 3

      readonly CONNECTING = 0
      readonly OPEN = 1
      readonly CLOSING = 2
      readonly CLOSED = 3

      binaryType: BinaryType = 'blob'
      bufferedAmount = 0
      extensions = ''
      protocol = ''

      onopen: ((this: WebSocket, ev: Event) => any) | null = null
      onclose: ((this: WebSocket, ev: CloseEvent) => any) | null = null
      onerror: ((this: WebSocket, ev: Event) => any) | null = null
      onmessage: ((this: WebSocket, ev: MessageEvent) => any) | null = null
    } as any
  })
}

export async function setupMultiAgentNetworkError(page: Page) {
  // Mock all multi-agent APIs to return network errors
  const errorResponse = {
    status: 500,
    contentType: 'application/json',
    body: JSON.stringify({
      error: '网络连接异常',
      message: '无法连接到多智能体服务',
    }),
  }

  await page.route('**/api/v1/multi-agent/**', route => {
    route.fulfill(errorResponse)
  })
}
