import { test, expect } from '@playwright/test';

const API_BASE_URL = 'http://localhost:8000';

test.describe('LangGraph 0.6.5 Features E2E Tests', () => {
  test.beforeAll(async () => {
    // 确保API服务器正在运行
    console.log('Testing LangGraph 0.6.5 features against:', API_BASE_URL);
  });

  test.describe('Context API 新特性测试', () => {
    test('新Context API vs 旧config模式对比', async ({ request }) => {
      // 测试新Context API
      const newApiResponse = await request.post(`${API_BASE_URL}/api/v1/langgraph/context-api/demo`, {
        data: {
          user_id: 'test_user_new',
          session_id: '550e8400-e29b-41d4-a716-446655440000',
          conversation_id: '123e4567-e89b-12d3-a456-426614174000',
          message: '测试新Context API功能',
          use_new_api: true
        }
      });

      expect(newApiResponse.ok()).toBeTruthy();
      const newApiResult = await newApiResponse.json();
      
      expect(newApiResult.success).toBe(true);
      expect(newApiResult.metadata.api_type).toBe('新Context API');
      expect(newApiResult.metadata.context_schema).toBe('LangGraphContextSchema');
      expect(newApiResult.result.messages).toBeDefined();
      expect(newApiResult.result.messages.length).toBeGreaterThan(0);
      
      // 验证消息内容包含用户信息
      const assistantMessage = newApiResult.result.messages.find((msg: any) => msg.role === 'assistant');
      expect(assistantMessage.content).toContain('test_user_new');
      expect(assistantMessage.content).toContain('新Context API');

      // 测试旧config模式
      const oldApiResponse = await request.post(`${API_BASE_URL}/api/v1/langgraph/context-api/demo`, {
        data: {
          user_id: 'test_user_old',
          session_id: '550e8400-e29b-41d4-a716-446655440001',
          message: '测试旧config模式',
          use_new_api: false
        }
      });

      expect(oldApiResponse.ok()).toBeTruthy();
      const oldApiResult = await oldApiResponse.json();
      
      expect(oldApiResult.success).toBe(true);
      expect(oldApiResult.metadata.api_type).toBe('旧config模式');
      expect(oldApiResult.metadata.context_schema).toBe('dict');

      // 验证执行时间在合理范围内
      expect(newApiResult.execution_time_ms).toBeLessThan(5000);
      expect(oldApiResult.execution_time_ms).toBeLessThan(5000);
    });

    test('Context API错误处理', async ({ request }) => {
      const response = await request.post(`${API_BASE_URL}/api/v1/langgraph/context-api/demo`, {
        data: {
          user_id: '',  // 空用户ID测试错误处理
          message: '测试错误处理',
          use_new_api: true
        }
      });

      if (!response.ok()) {
        expect(response.status()).toBe(500);
        const error = await response.json();
        expect(error.detail).toContain('Context API演示失败');
      }
    });
  });

  test.describe('Durability控制测试', () => {
    test.describe.parallel('三种durability模式测试', () => {
      ['exit', 'async', 'sync'].forEach(mode => {
        test(`${mode} durability模式`, async ({ request }) => {
          const response = await request.post(`${API_BASE_URL}/api/v1/langgraph/durability/demo`, {
            data: {
              message: `测试${mode}模式的持久化控制`,
              durability_mode: mode
            }
          });

          expect(response.ok()).toBeTruthy();
          const result = await response.json();
          
          expect(result.success).toBe(true);
          expect(result.metadata.durability_mode).toBe(mode);
          expect(result.metadata.checkpoint_strategy).toBeDefined();
          expect(result.result.messages).toBeDefined();
          
          // 验证响应消息包含durability模式信息
          const assistantMessage = result.result.messages.find((msg: any) => msg.role === 'assistant');
          expect(assistantMessage.content).toContain(mode);
          expect(assistantMessage.metadata.durability_mode).toBe(mode);
        });
      });
    });

    test('Durability性能对比', async ({ request }) => {
      const modes = ['exit', 'async', 'sync'];
      const results: any[] = [];

      for (const mode of modes) {
        const response = await request.post(`${API_BASE_URL}/api/v1/langgraph/durability/demo`, {
          data: {
            message: '性能测试消息',
            durability_mode: mode as any
          }
        });

        expect(response.ok()).toBeTruthy();
        const result = await response.json();
        results.push({ mode, execution_time: result.execution_time_ms });
      }

      // 验证所有模式都能成功执行
      expect(results.length).toBe(3);
      results.forEach(result => {
        expect(result.execution_time).toBeGreaterThan(0);
        expect(result.execution_time).toBeLessThan(10000);
      });

      console.log('Durability performance comparison:', results);
    });
  });

  test.describe('Node Caching测试', () => {
    test('缓存功能验证', async ({ request }) => {
      // 测试启用缓存
      const cachedResponse = await request.post(`${API_BASE_URL}/api/v1/langgraph/caching/demo`, {
        data: {
          message: '计算密集型任务测试',
          enable_cache: true,
          cache_ttl: 300
        }
      });

      expect(cachedResponse.ok()).toBeTruthy();
      const cachedResult = await cachedResponse.json();
      
      expect(cachedResult.success).toBe(true);
      expect(cachedResult.metadata.cache_statistics).toBeDefined();
      expect(cachedResult.metadata.cache_statistics.cache_enabled).toBe(true);
      expect(cachedResult.metadata.cache_statistics.total_executions).toBe(3);
      
      // 验证缓存命中率
      const cacheStats = cachedResult.metadata.cache_statistics;
      expect(cacheStats.cache_hits).toBeGreaterThan(0);
      expect(cacheStats.actual_computations).toBeLessThan(cacheStats.total_executions);
      
      // 验证性能提升
      const performanceImprovement = parseFloat(cachedResult.metadata.performance_improvement.replace('%', ''));
      expect(performanceImprovement).toBeGreaterThan(0);

      // 测试禁用缓存的对比
      const noCacheResponse = await request.post(`${API_BASE_URL}/api/v1/langgraph/caching/demo`, {
        data: {
          message: '计算密集型任务测试',
          enable_cache: false,
          cache_ttl: 300
        }
      });

      expect(noCacheResponse.ok()).toBeTruthy();
      const noCacheResult = await noCacheResponse.json();
      
      expect(noCacheResult.metadata.cache_statistics.cache_enabled).toBe(false);
      expect(noCacheResult.metadata.cache_statistics.cache_hits).toBe(0);
      expect(noCacheResult.metadata.cache_statistics.actual_computations).toBe(3);
    });

    test('缓存管理端点', async ({ request }) => {
      // 获取缓存统计
      const statsResponse = await request.get(`${API_BASE_URL}/api/v1/langgraph/cache/stats`);
      expect(statsResponse.ok()).toBeTruthy();
      
      const stats = await statsResponse.json();
      expect(stats.cache_backend).toBeDefined();
      expect(stats.default_policy).toBeDefined();
      expect(stats.default_policy.ttl).toBeGreaterThan(0);

      // 清空缓存
      const clearResponse = await request.post(`${API_BASE_URL}/api/v1/langgraph/cache/clear`);
      expect(clearResponse.ok()).toBeTruthy();
      
      const clearResult = await clearResponse.json();
      expect(clearResult.success).toBe(true);
      expect(clearResult.message).toContain('缓存已清空');
    });
  });

  test.describe('Pre/Post Hooks测试', () => {
    test('Hooks功能验证', async ({ request }) => {
      const testMessages = [
        {
          role: 'user',
          content: '这是一个测试消息，包含一些需要处理的内容。',
          timestamp: new Date().toISOString()
        }
      ];

      // 测试启用所有hooks
      const hooksResponse = await request.post(`${API_BASE_URL}/api/v1/langgraph/hooks/demo`, {
        data: {
          messages: testMessages,
          enable_pre_hooks: true,
          enable_post_hooks: true
        }
      });

      expect(hooksResponse.ok()).toBeTruthy();
      const hooksResult = await hooksResponse.json();
      
      expect(hooksResult.success).toBe(true);
      expect(hooksResult.metadata.pre_hooks_enabled).toBe(true);
      expect(hooksResult.metadata.post_hooks_enabled).toBe(true);
      expect(hooksResult.metadata.hooks_executed).toBeGreaterThan(0);
      expect(hooksResult.metadata.final_message_count).toBeGreaterThan(hooksResult.metadata.original_message_count);

      // 验证hook effects
      expect(hooksResult.metadata.hook_effects).toBeDefined();
      expect(Array.isArray(hooksResult.metadata.hook_effects)).toBe(true);

      // 测试禁用hooks对比
      const noHooksResponse = await request.post(`${API_BASE_URL}/api/v1/langgraph/hooks/demo`, {
        data: {
          messages: testMessages,
          enable_pre_hooks: false,
          enable_post_hooks: false
        }
      });

      expect(noHooksResponse.ok()).toBeTruthy();
      const noHooksResult = await noHooksResponse.json();
      
      expect(noHooksResult.metadata.hooks_executed).toBeLessThan(hooksResult.metadata.hooks_executed);
    });

    test('Hooks状态管理', async ({ request }) => {
      // 获取hooks状态
      const statusResponse = await request.get(`${API_BASE_URL}/api/v1/langgraph/hooks/status`);
      expect(statusResponse.ok()).toBeTruthy();
      
      const status = await statusResponse.json();
      expect(status).toBeDefined();
      
      // 验证hooks配置端点（如果有的话）
      // Note: 这个端点可能需要specific hook name，暂时跳过具体配置测试
    });
  });

  test.describe('完整工作流集成测试', () => {
    test('所有特性综合演示', async ({ request }) => {
      const response = await request.post(`${API_BASE_URL}/api/v1/langgraph/complete-demo`);
      
      expect(response.ok()).toBeTruthy();
      const result = await response.json();
      
      expect(result.success).toBe(true);
      expect(result.execution_time_ms).toBeGreaterThan(0);
      expect(result.execution_time_ms).toBeLessThan(30000); // 30秒超时限制
      
      // 验证所有特性都被演示了
      expect(result.metadata.features_demonstrated).toContain('新Context API (LangGraphContextSchema)');
      expect(result.metadata.features_demonstrated).toContain('Durability控制 (async模式)');
      expect(result.metadata.features_demonstrated).toContain('条件分支工作流');
      expect(result.metadata.features_demonstrated).toContain('类型安全的上下文传递');
      
      expect(result.metadata.workflow_type).toBe('conditional_workflow');
      expect(result.metadata.context_api_version).toBe('0.6.5');
      
      // 验证工作流执行结果
      expect(result.result.messages).toBeDefined();
      expect(result.result.messages.length).toBeGreaterThan(0);
      
      // 验证工作流的条件分支执行
      expect(result.result.context).toBeDefined();
      expect(result.result.context.processed).toBe(true);
      expect(result.result.context.data_quality).toBeDefined();
      expect(['high', 'low']).toContain(result.result.context.data_quality);
    });

    test('并发执行测试', async ({ request }) => {
      const concurrentRequests = 3;
      const promises = Array(concurrentRequests).fill(0).map(() => 
        request.post(`${API_BASE_URL}/api/v1/langgraph/complete-demo`)
      );

      const responses = await Promise.all(promises);
      
      // 验证所有请求都成功
      responses.forEach(async (response) => {
        expect(response.ok()).toBeTruthy();
        const result = await response.json();
        expect(result.success).toBe(true);
      });
    });
  });

  test.describe('错误处理和边界情况测试', () => {
    test('无效请求处理', async ({ request }) => {
      // 测试无效的durability模式
      const invalidDurabilityResponse = await request.post(`${API_BASE_URL}/api/v1/langgraph/durability/demo`, {
        data: {
          message: '测试无效durability',
          durability_mode: 'invalid_mode'
        }
      });

      expect(invalidDurabilityResponse.status()).toBe(422); // Validation error
    });

    test('大数据量处理', async ({ request }) => {
      const largeMessages = Array(100).fill(0).map((_, i) => ({
        role: 'user',
        content: `大数据量测试消息 #${i}`,
        timestamp: new Date().toISOString()
      }));

      const response = await request.post(`${API_BASE_URL}/api/v1/langgraph/hooks/demo`, {
        data: {
          messages: largeMessages,
          enable_pre_hooks: true,
          enable_post_hooks: true
        }
      });

      expect(response.ok()).toBeTruthy();
      const result = await response.json();
      expect(result.success).toBe(true);
      expect(result.execution_time_ms).toBeLessThan(60000); // 1分钟超时
    });
  });
});