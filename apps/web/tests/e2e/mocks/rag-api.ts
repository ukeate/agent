import { Page } from '@playwright/test'

export async function setupRagApiMocks(page: Page) {
  // Mock基础RAG API
  await page.route('/api/v1/rag/query', async route => {
    const request = route.request()
    const postData = request.postData()

    if (postData) {
      const data = JSON.parse(postData)

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          results: [
            {
              id: 'result-1',
              content: `检索到的内容：${data.query}`,
              source: 'test-file.md',
              score: 0.95,
              metadata: {
                type: 'document',
                created_at: new Date().toISOString(),
              },
            },
            {
              id: 'result-2',
              content: `相关代码片段：${data.query}`,
              source: 'test-code.py',
              score: 0.87,
              metadata: {
                type: 'code',
                created_at: new Date().toISOString(),
              },
            },
          ],
          query_id: `query-${Date.now()}`,
          processing_time: 234,
          total_results: 2,
        }),
      })
    }
  })

  // Mock索引统计API
  await page.route('/api/v1/rag/index/stats', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        success: true,
        stats: {
          documents: {
            vectors_count: 468,
            points_count: 468,
            status: 'green',
          },
          code: {
            vectors_count: 1746,
            points_count: 1746,
            status: 'green',
          },
        },
        health: {
          qdrant: true,
          openai: true,
        },
      }),
    })
  })

  // Mock健康检查API
  await page.route('/api/v1/rag/health', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'healthy',
        details: {
          success: true,
          qdrant_connected: true,
          openai_available: true,
        },
      }),
    })
  })
}

export async function setupAgenticRagApiMocks(page: Page) {
  // Mock Agentic RAG查询API - 使用与实际后端API匹配的响应格式
  await page.route('/api/v1/rag/agentic/query', async route => {
    const request = route.request()
    const postData = request.postData()

    if (postData) {
      const data = JSON.parse(postData)

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          success: true,
          query_id: `agentic-query-${Date.now()}`,
          query_analysis: {
            intent_type: 'exploratory',
            confidence: 0.8,
            complexity_score: 0.5,
            entities: ['AI', '智能检索'],
            keywords: data.query.split(' '),
            domain: null,
            language: 'zh',
          },
          expanded_queries: [
            {
              original_query: data.query,
              expanded_queries: [`详细说明：${data.query}`],
              strategy: 'semantic',
              confidence: 0.7,
              sub_questions: null,
              language_variants: null,
              explanation: '基于语义理解的查询扩展',
            },
          ],
          retrieval_results: [
            {
              agent_type: 'semantic',
              results: [
                {
                  id: 'agentic-result-1',
                  content: `智能分析结果：${data.query}`,
                  file_path: 'intelligent-doc.md',
                  content_type: null,
                  metadata: {
                    type: 'analyzed_document',
                    agent_type: 'semantic_expert',
                    created_at: new Date().toISOString(),
                  },
                  score: 0.96,
                },
                {
                  id: 'agentic-result-2',
                  content: `多代理协作发现：${data.query}`,
                  file_path: 'multi-agent-source.py',
                  content_type: null,
                  metadata: {
                    type: 'code_analysis',
                    agent_type: 'code_expert',
                    created_at: new Date().toISOString(),
                  },
                  score: 0.89,
                },
              ],
              score: 0.92,
              confidence: 0.93,
              processing_time: 167,
              explanation: '使用语义检索策略获得优质结果',
            },
            {
              agent_type: 'keyword',
              results: [
                {
                  id: 'keyword-result-1',
                  content: `关键词匹配结果：${data.query}`,
                  file_path: 'keyword-doc.md',
                  content_type: null,
                  metadata: {
                    type: 'keyword_match',
                    agent_type: 'keyword_expert',
                    created_at: new Date().toISOString(),
                  },
                  score: 0.85,
                },
              ],
              score: 0.85,
              confidence: 0.78,
              processing_time: 45,
              explanation: '基于关键词匹配的补充检索',
            },
          ],
          validation_result: {
            quality_scores: {
              relevance: {
                dimension: 'relevance',
                score: 0.94,
                confidence: 0.9,
                explanation: '结果与查询高度相关',
              },
              accuracy: {
                dimension: 'accuracy',
                score: 0.88,
                confidence: 0.85,
                explanation: '信息准确性验证通过',
              },
            },
            conflicts: [],
            overall_quality: 0.91,
            overall_confidence: 0.87,
            recommendations: ['结果质量良好，建议继续使用当前策略'],
          },
          processing_time: 272,
          timestamp: new Date().toISOString(),
          session_id: data.session_id || null,
          error: null,
        }),
      })
    }
  })

  // Mock流式查询API
  await page.route('/api/v1/rag/agentic/query/stream', async route => {
    const request = route.request()
    const postData = request.postData()

    if (postData) {
      const data = JSON.parse(postData)

      const streamEvents = [
        {
          event: 'query_analysis_start',
          data: { step: 'analyzing', message: '正在分析查询意图...' },
        },
        {
          event: 'query_analysis_complete',
          data: { step: 'analyzed', intent_type: 'factual', complexity: 0.7 },
        },
        {
          event: 'query_expansion_start',
          data: { step: 'expanding', message: '正在扩展查询...' },
        },
        {
          event: 'query_expansion_complete',
          data: {
            step: 'expanded',
            expanded_queries: [`${data.query} 详细`, `${data.query} 相关`],
          },
        },
        {
          event: 'retrieval_start',
          data: {
            step: 'retrieving',
            agents: ['semantic', 'keyword', 'structured'],
          },
        },
        {
          event: 'retrieval_progress',
          data: {
            step: 'retrieving',
            progress: 0.5,
            current_agent: 'semantic',
          },
        },
        {
          event: 'retrieval_complete',
          data: { step: 'retrieved', results_count: 2 },
        },
        {
          event: 'validation_complete',
          data: { step: 'validated', confidence: 0.91 },
        },
        {
          event: 'complete',
          data: { step: 'done', message: '智能检索完成' },
        },
      ]

      const responseText =
        streamEvents
          .map(event => `data: ${JSON.stringify(event)}\n\n`)
          .join('') + 'data: [DONE]\n\n'

      await route.fulfill({
        status: 200,
        contentType: 'text/event-stream',
        headers: {
          'Cache-Control': 'no-cache',
          Connection: 'keep-alive',
        },
        body: responseText,
      })
    }
  })

  // Mock解释API
  await page.route('/api/v1/rag/agentic/explain*', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        success: true,
        explanation: {
          decision_process: [
            {
              step: '查询分析',
              reasoning: '识别为事实性查询，复杂度中等',
              confidence: 0.85,
            },
            {
              step: '策略选择',
              reasoning: '选择多代理协作以提高结果质量',
              confidence: 0.92,
            },
            {
              step: '结果验证',
              reasoning: '通过交叉验证确保结果准确性',
              confidence: 0.89,
            },
          ],
          visualization_data: {
            retrieval_path: [
              'query',
              'analysis',
              'expansion',
              'retrieval',
              'validation',
            ],
            agent_collaboration: {
              semantic_agent: { contribution: 0.45, confidence: 0.91 },
              keyword_agent: { contribution: 0.32, confidence: 0.87 },
              structured_agent: { contribution: 0.23, confidence: 0.83 },
            },
          },
          confidence_analysis: {
            overall: 0.91,
            factors: [
              { factor: '语义匹配度', score: 0.94, impact: 'high' },
              { factor: '多源一致性', score: 0.89, impact: 'medium' },
              { factor: '历史准确率', score: 0.92, impact: 'medium' },
            ],
          },
        },
      }),
    })
  })

  // Mock反馈API
  await page.route('/api/v1/rag/agentic/feedback', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        success: true,
        message: '反馈提交成功',
        feedback_id: `feedback-${Date.now()}`,
        learning_impact: '将用于优化检索策略',
      }),
    })
  })

  // Mock统计API
  await page.route('/api/v1/rag/agentic/stats', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        success: true,
        data: {
          total_queries: 1247,
          successful_queries: 1156,
          average_confidence: 0.87,
          average_processing_time: 245,
          top_strategies: [
            { name: 'multi_agent', usage: 0.45, success_rate: 0.94 },
            { name: 'semantic', usage: 0.32, success_rate: 0.91 },
            { name: 'hybrid', usage: 0.23, success_rate: 0.89 },
          ],
          agent_performance: {
            semantic_expert: {
              queries: 567,
              success_rate: 0.93,
              avg_time: 123,
            },
            keyword_expert: { queries: 423, success_rate: 0.89, avg_time: 98 },
            structured_expert: {
              queries: 257,
              success_rate: 0.87,
              avg_time: 156,
            },
          },
        },
      }),
    })
  })

  // Mock健康检查API
  await page.route('/api/v1/rag/agentic/health', async route => {
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'healthy',
        components: {
          query_analyzer: { status: 'healthy', response_time: 23 },
          query_expander: { status: 'healthy', response_time: 18 },
          semantic_agent: { status: 'healthy', response_time: 45 },
          keyword_agent: { status: 'healthy', response_time: 32 },
          structured_agent: { status: 'healthy', response_time: 67 },
          result_validator: { status: 'healthy', response_time: 28 },
          context_composer: { status: 'healthy', response_time: 21 },
        },
        overall_health: 'excellent',
      }),
    })
  })
}

export async function setupRagApiErrors(page: Page) {
  // Mock基础RAG查询失败
  await page.route('/api/v1/rag/query', route => {
    route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({
        success: false,
        error: '向量数据库连接失败',
      }),
    })
  })

  // Mock Agentic RAG查询失败
  await page.route('/api/v1/rag/agentic/query', route => {
    route.fulfill({
      status: 500,
      contentType: 'application/json',
      body: JSON.stringify({
        success: false,
        error: '智能分析服务暂时不可用',
        error_code: 'AGENTIC_SERVICE_DOWN',
        retry_after: 30,
      }),
    })
  })
}
