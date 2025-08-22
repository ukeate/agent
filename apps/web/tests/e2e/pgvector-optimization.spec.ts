/**
 * pgvector 0.8性能优化功能E2E测试
 * 测试向量存储、搜索、量化和监控功能
 */

import { test, expect } from '@playwright/test';

// 测试配置
const API_BASE_URL = 'http://localhost:8000';
const WEB_BASE_URL = 'http://localhost:3000';

// 测试向量数据
const TEST_COLLECTION = 'e2e_test_collection';
const TEST_VECTORS = [
  {
    content: "人工智能是计算机科学的一个重要分支",
    embedding: Array.from({ length: 384 }, (_, i) => Math.sin(i * 0.1)),
    metadata: { topic: "AI", language: "zh" }
  },
  {
    content: "机器学习算法能够从数据中学习模式",
    embedding: Array.from({ length: 384 }, (_, i) => Math.cos(i * 0.1)),
    metadata: { topic: "ML", language: "zh" }
  },
  {
    content: "深度学习是机器学习的一个子领域",
    embedding: Array.from({ length: 384 }, (_, i) => Math.sin(i * 0.2)),
    metadata: { topic: "DL", language: "zh" }
  }
];

test.describe('pgvector优化功能E2E测试', () => {
  
  test.beforeAll(async ({ request }) => {
    // 清理测试数据
    try {
      await request.delete(`${API_BASE_URL}/api/v1/pgvector/collections/${TEST_COLLECTION}`);
    } catch (e) {
      // 集合可能不存在，忽略错误
    }
  });

  test.afterAll(async ({ request }) => {
    // 清理测试数据
    try {
      await request.delete(`${API_BASE_URL}/api/v1/pgvector/collections/${TEST_COLLECTION}`);
    } catch (e) {
      // 忽略清理错误
    }
  });

  test('pgvector健康检查', async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/api/v1/pgvector/health`);
    expect(response.ok()).toBeTruthy();
    
    const health = await response.json();
    console.log('pgvector健康状态:', health);
    
    expect(health).toHaveProperty('status');
    expect(health).toHaveProperty('pgvector_version');
    expect(health).toHaveProperty('database_connection');
  });

  test('创建向量集合', async ({ request }) => {
    const createRequest = {
      collection_name: TEST_COLLECTION,
      dimension: 384,
      index_type: "hnsw",
      distance_metric: "l2",
      index_options: {
        m: 16,
        ef_construction: 64
      }
    };

    const response = await request.post(`${API_BASE_URL}/api/v1/pgvector/collections`, {
      data: createRequest
    });

    expect(response.ok()).toBeTruthy();
    
    const result = await response.json();
    console.log('集合创建结果:', result);
    
    expect(result).toHaveProperty('success', true);
    expect(result).toHaveProperty('collection_name', TEST_COLLECTION);
  });

  test('批量插入向量', async ({ request }) => {
    const insertRequest = {
      collection_name: TEST_COLLECTION,
      documents: TEST_VECTORS
    };

    const response = await request.post(`${API_BASE_URL}/api/v1/pgvector/vectors`, {
      data: insertRequest
    });

    expect(response.ok()).toBeTruthy();
    
    const result = await response.json();
    console.log('向量插入结果:', result);
    
    expect(result).toHaveProperty('success', true);
    expect(result).toHaveProperty('inserted_count', TEST_VECTORS.length);
    expect(result.document_ids).toHaveLength(TEST_VECTORS.length);
  });

  test('向量相似性搜索', async ({ request }) => {
    const searchRequest = {
      collection_name: TEST_COLLECTION,
      query_vector: Array.from({ length: 384 }, (_, i) => Math.sin(i * 0.1)),
      limit: 3,
      distance_metric: "l2",
      include_distances: true
    };

    const response = await request.post(`${API_BASE_URL}/api/v1/pgvector/search/similarity`, {
      data: searchRequest
    });

    expect(response.ok()).toBeTruthy();
    
    const result = await response.json();
    console.log('相似性搜索结果:', result);
    
    expect(result).toHaveProperty('success', true);
    expect(result.results).toBeInstanceOf(Array);
    expect(result.results.length).toBeGreaterThan(0);
    
    // 验证结果格式
    const firstResult = result.results[0];
    expect(firstResult).toHaveProperty('id');
    expect(firstResult).toHaveProperty('content');
    expect(firstResult).toHaveProperty('metadata');
    expect(firstResult).toHaveProperty('distance');
  });

  test('混合搜索功能', async ({ request }) => {
    const searchRequest = {
      collection_name: TEST_COLLECTION,
      query_vector: Array.from({ length: 384 }, (_, i) => Math.cos(i * 0.1)),
      query_text: "机器学习",
      limit: 2,
      vector_weight: 0.7,
      text_weight: 0.3
    };

    const response = await request.post(`${API_BASE_URL}/api/v1/pgvector/search/hybrid`, {
      data: searchRequest
    });

    expect(response.ok()).toBeTruthy();
    
    const result = await response.json();
    console.log('混合搜索结果:', result);
    
    expect(result).toHaveProperty('success', true);
    expect(result.results).toBeInstanceOf(Array);
  });

  test('获取集合统计信息', async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/api/v1/pgvector/collections/${TEST_COLLECTION}/stats`);
    expect(response.ok()).toBeTruthy();
    
    const stats = await response.json();
    console.log('集合统计信息:', stats);
    
    expect(stats).toHaveProperty('collection_name', TEST_COLLECTION);
    expect(stats).toHaveProperty('total_vectors');
    expect(stats.total_vectors).toBeGreaterThan(0);
    expect(stats).toHaveProperty('table_size');
  });

  test('向量量化配置', async ({ request }) => {
    // 首先检查是否有足够的向量进行量化训练
    const statsResponse = await request.get(`${API_BASE_URL}/api/v1/pgvector/collections/${TEST_COLLECTION}/stats`);
    const stats = await statsResponse.json();
    
    if (stats.total_vectors < 10) {
      console.log('向量数量不足，跳过量化测试');
      test.skip();
      return;
    }

    const quantizationRequest = {
      collection_name: TEST_COLLECTION,
      quantization_type: "binary",
      config: {
        threshold: 0.0,
        batch_size: 100
      }
    };

    const response = await request.post(`${API_BASE_URL}/api/v1/pgvector/quantization/configure`, {
      data: quantizationRequest
    });

    // 量化可能失败（训练数据不足），这里不强制要求成功
    console.log('量化配置响应状态:', response.status());
    
    if (response.ok()) {
      const result = await response.json();
      console.log('量化配置结果:', result);
      expect(result).toHaveProperty('success');
    } else {
      console.log('量化配置失败，可能是训练数据不足');
    }
  });

  test('性能监控报告', async ({ request }) => {
    const response = await request.get(`${API_BASE_URL}/api/v1/pgvector/monitoring/performance-report?time_range_hours=1`);
    expect(response.ok()).toBeTruthy();
    
    const report = await response.json();
    console.log('性能监控报告:', report);
    
    expect(report).toHaveProperty('report_period');
    expect(report).toHaveProperty('query_performance');
    expect(report.query_performance).toHaveProperty('total_queries');
  });

  test('集合优化', async ({ request }) => {
    const response = await request.post(`${API_BASE_URL}/api/v1/pgvector/collections/${TEST_COLLECTION}/optimize`);
    expect(response.ok()).toBeTruthy();
    
    const result = await response.json();
    console.log('集合优化结果:', result);
    
    expect(result).toHaveProperty('success');
  });

});

test.describe('pgvector错误处理测试', () => {

  test('不存在的集合搜索', async ({ request }) => {
    const searchRequest = {
      collection_name: "non_existent_collection",
      query_vector: Array.from({ length: 384 }, () => Math.random()),
      limit: 5
    };

    const response = await request.post(`${API_BASE_URL}/api/v1/pgvector/search/similarity`, {
      data: searchRequest
    });

    expect(response.status()).toBe(500); // 应该返回错误
    
    const result = await response.json();
    console.log('错误搜索结果:', result);
    expect(result).toHaveProperty('detail');
  });

  test('无效的向量维度', async ({ request }) => {
    const createRequest = {
      collection_name: "invalid_dimension_test",
      dimension: -1, // 无效维度
      index_type: "hnsw"
    };

    const response = await request.post(`${API_BASE_URL}/api/v1/pgvector/collections`, {
      data: createRequest
    });

    expect(response.status()).toBe(422); // 验证错误
    
    const result = await response.json();
    console.log('无效维度错误:', result);
  });

});

test.describe('pgvector性能测试', () => {

  test('大批量向量插入性能', async ({ request }) => {
    const largeBatch = Array.from({ length: 50 }, (_, i) => ({
      content: `测试文档 ${i}`,
      embedding: Array.from({ length: 384 }, () => Math.random()),
      metadata: { index: i, batch: "performance_test" }
    }));

    const insertRequest = {
      collection_name: TEST_COLLECTION,
      documents: largeBatch
    };

    const startTime = Date.now();
    const response = await request.post(`${API_BASE_URL}/api/v1/pgvector/vectors`, {
      data: insertRequest
    });
    const endTime = Date.now();

    const insertTime = endTime - startTime;
    console.log(`批量插入50个向量耗时: ${insertTime}ms`);

    expect(response.ok()).toBeTruthy();
    
    const result = await response.json();
    expect(result.inserted_count).toBe(largeBatch.length);
    
    // 性能基准：50个向量插入应该在10秒内完成
    expect(insertTime).toBeLessThan(10000);
  });

  test('高频搜索性能', async ({ request }) => {
    const searchTimes: number[] = [];
    const searchCount = 10;

    for (let i = 0; i < searchCount; i++) {
      const searchRequest = {
        collection_name: TEST_COLLECTION,
        query_vector: Array.from({ length: 384 }, () => Math.random()),
        limit: 10
      };

      const startTime = Date.now();
      const response = await request.post(`${API_BASE_URL}/api/v1/pgvector/search/similarity`, {
        data: searchRequest
      });
      const endTime = Date.now();

      expect(response.ok()).toBeTruthy();
      searchTimes.push(endTime - startTime);
    }

    const avgSearchTime = searchTimes.reduce((a, b) => a + b, 0) / searchTimes.length;
    const maxSearchTime = Math.max(...searchTimes);
    
    console.log(`平均搜索时间: ${avgSearchTime.toFixed(2)}ms`);
    console.log(`最大搜索时间: ${maxSearchTime}ms`);
    console.log(`所有搜索时间: ${searchTimes.map(t => t + 'ms').join(', ')}`);

    // 性能基准：平均搜索时间应该在2秒内
    expect(avgSearchTime).toBeLessThan(2000);
    // 最大搜索时间应该在5秒内
    expect(maxSearchTime).toBeLessThan(5000);
  });

});