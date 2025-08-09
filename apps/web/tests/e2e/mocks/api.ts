import { Page } from '@playwright/test'

export async function setupApiMocks(page: Page) {
  // Mock聊天API (支持流式响应)
  await page.route('/api/v1/agent/chat', async (route) => {
    const request = route.request()
    const postData = request.postData()
    const headers = request.headers()
    
    if (postData) {
      const data = JSON.parse(postData)
      
      // 检查是否是流式请求
      if (headers['accept'] === 'text/event-stream' || data.stream) {
        // 模拟OpenAI标准格式的流式响应
        const messageId = `chatcmpl-${Date.now()}`
        const created = Math.floor(Date.now() / 1000)
        const responseText = [
          `data: {"id":"${messageId}","object":"chat.completion.chunk","created":${created},"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"我收到了你的消息："},"finish_reason":null}]}`,
          `data: {"id":"${messageId}","object":"chat.completion.chunk","created":${created},"model":"gpt-4o-mini","choices":[{"index":0,"delta":{"content":"${data.message}"},"finish_reason":null}]}`,
          `data: {"id":"${messageId}","object":"chat.completion.chunk","created":${created},"model":"gpt-4o-mini","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
          'data: [DONE]',
          ''
        ].join('\n\n')
        
        await route.fulfill({
          status: 200,
          contentType: 'text/event-stream',
          headers: {
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
          },
          body: responseText
        })
      } else {
        // 普通响应
        await route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            success: true,
            data: {
              message: {
                id: `msg-${Date.now()}`,
                content: '我收到了你的消息：' + data.message,
                role: 'agent',
                timestamp: new Date().toISOString()
              }
            }
          })
        })
      }
    } else {
      await route.fulfill({
        status: 400,
        contentType: 'application/json',
        body: JSON.stringify({
          success: false,
          error: '缺少消息内容'
        })
      })
    }
  })

  // Mock智能体状态API
  await page.route('/api/v1/agent/status', async (route) => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        success: true,
        data: {
          status: 'active',
          uptime: '1h 30m',
          version: '1.0.0'
        }
      })
    })
  })
}

export async function setupNetworkError(page: Page) {
  // Mock网络错误
  await page.route('/api/v1/agent/chat', (route) => {
    route.abort('failed')
  })
}