/**
 * pgvector 0.8.0 性能优化 E2E 测试
 * 验证向量存储、量化、索引优化和监控功能
 */

import { test, expect } from '@playwright/test'

const API_BASE_URL = 'http://localhost:8000/api/v1'

// 测试数据
const TEST_COLLECTION = 'test_pgvector_optimization'
const TEST_VECTORS = [
  {
    content: '人工智能是计算机科学的一个分支',
    embedding: Array.from({ length: 1536 }, () => Math.random() * 2 - 1),
    metadata: { category: 'ai', topic: 'introduction' },
  },
  {
    content: '机器学习是人工智能的核心技术',
    embedding: Array.from({ length: 1536 }, () => Math.random() * 2 - 1),
    metadata: { category: 'ml', topic: 'core_technology' },
  },
  {
    content: '深度学习是机器学习的一个重要分支',
    embedding: Array.from({ length: 1536 }, () => Math.random() * 2 - 1),
    metadata: { category: 'dl', topic: 'branch' },
  },
]

test.describe('pgvector 0.8.0 性能优化系统E2E测试', () => {
  test.beforeAll(async () => {
    // 等待API服务启动
    const maxRetries = 30
    let retries = 0

    while (retries < maxRetries) {
      try {
        const response = await fetch(`${API_BASE_URL}/pgvector/health`)
        if (response.ok) {
          console.log('API服务已启动')
          break
        }
      } catch (error) {
        retries++
        if (retries >= maxRetries) {
          throw new Error('API服务启动超时')
        }
        await new Promise(resolve => setTimeout(resolve, 2000))
      }
    }
  })

  test('健康检查和配置验证', async ({ request }) => {
    // 检查pgvector健康状态
    const healthResponse = await request.get(`${API_BASE_URL}/pgvector/health`)
    expect(healthResponse.ok()).toBeTruthy()

    const healthData = await healthResponse.json()
    expect(healthData.status).toBe('healthy')
    expect(healthData.pgvector_version).toBeTruthy()
    console.log(`pgvector版本: ${healthData.pgvector_version}`)

    // 验证配置信息
    const configResponse = await request.get(`${API_BASE_URL}/pgvector/config`)
    expect(configResponse.ok()).toBeTruthy()

    const configData = await configResponse.json()
    expect(configData.config.pgvector_enabled).toBe(true)
    expect(configData.config.hnsw).toBeDefined()
    expect(configData.config.ivfflat).toBeDefined()
    expect(configData.config.quantization).toBeDefined()
    console.log('配置验证通过:', JSON.stringify(configData.config, null, 2))
  })

  test('创建向量集合和索引优化', async ({ request }) => {
    // 创建HNSW索引集合
    const hnswCollectionRequest = {
      collection_name: `${TEST_COLLECTION}_hnsw`,
      dimension: 1536,
      index_type: 'hnsw',
      distance_metric: 'l2',
      index_options: {
        m: 16,
        ef_construction: 64,
      },
    }

    const hnswResponse = await request.post(
      `${API_BASE_URL}/pgvector/collections`,
      {
        data: hnswCollectionRequest,
      }
    )
    expect(hnswResponse.ok()).toBeTruthy()

    const hnswData = await hnswResponse.json()
    expect(hnswData.status).toBe('success')
    console.log('HNSW集合创建成功:', hnswData.message)

    // 创建IVFFlat索引集合
    const ivfCollectionRequest = {
      collection_name: `${TEST_COLLECTION}_ivf`,
      dimension: 1536,
      index_type: 'ivfflat',
      distance_metric: 'cosine',
      index_options: {
        lists: 100,
      },
    }

    const ivfResponse = await request.post(
      `${API_BASE_URL}/pgvector/collections`,
      {
        data: ivfCollectionRequest,
      }
    )
    expect(ivfResponse.ok()).toBeTruthy()

    const ivfData = await ivfResponse.json()
    expect(ivfData.status).toBe('success')
    console.log('IVFFlat集合创建成功:', ivfData.message)
  })

  test('向量插入和相似性搜索性能测试', async ({ request }) => {
    const collectionName = `${TEST_COLLECTION}_hnsw`

    // 批量插入测试向量
    for (const vector of TEST_VECTORS) {
      const insertResponse = await request.post(
        `${API_BASE_URL}/pgvector/vectors`,
        {
          data: {
            collection_name: collectionName,
            documents: [vector],
          },
        }
      )
      expect(insertResponse.ok()).toBeTruthy()
    }
    console.log(`${TEST_VECTORS.length}个向量插入完成`)

    // 相似性搜索测试
    const searchRequest = {
      collection_name: collectionName,
      query_vector: TEST_VECTORS[0].embedding,
      limit: 5,
      distance_metric: 'l2',
      include_distances: true,
    }

    const searchStartTime = Date.now()
    const searchResponse = await request.post(
      `${API_BASE_URL}/pgvector/search/similarity`,
      {
        data: searchRequest,
      }
    )
    const searchTime = Date.now() - searchStartTime

    expect(searchResponse.ok()).toBeTruthy()

    const searchData = await searchResponse.json()
    expect(searchData.status).toBe('success')
    expect(searchData.results).toBeDefined()
    expect(searchData.results.length).toBeGreaterThan(0)
    expect(searchData.execution_time_ms).toBeDefined()

    console.log(
      `相似性搜索完成: ${searchTime}ms, 服务器执行时间: ${searchData.execution_time_ms}ms`
    )
    console.log(`找到${searchData.results.length}个结果`)
  })

  test('向量量化配置和测试', async ({ request }) => {
    const collectionName = `${TEST_COLLECTION}_hnsw`

    // 配置二进制量化
    const binaryQuantConfig = {
      collection_name: collectionName,
      quantization_type: 'binary',
      config: {
        bits: 8,
        training_size: 1000,
      },
    }

    const binaryResponse = await request.post(
      `${API_BASE_URL}/pgvector/quantization/configure`,
      {
        data: binaryQuantConfig,
      }
    )
    expect(binaryResponse.ok()).toBeTruthy()

    const binaryData = await binaryResponse.json()
    expect(binaryData.status).toBe('success')
    console.log('二进制量化配置成功:', binaryData.message)

    // 配置半精度量化
    const halfQuantConfig = {
      collection_name: `${TEST_COLLECTION}_ivf`,
      quantization_type: 'halfprecision',
      config: {},
    }

    const halfResponse = await request.post(
      `${API_BASE_URL}/pgvector/quantization/configure`,
      {
        data: halfQuantConfig,
      }
    )
    expect(halfResponse.ok()).toBeTruthy()

    const halfData = await halfResponse.json()
    expect(halfData.status).toBe('success')
    console.log('半精度量化配置成功:', halfData.message)
  })

  test('性能监控和统计信息', async ({ request }) => {
    const collectionName = `${TEST_COLLECTION}_hnsw`

    // 获取集合统计信息
    const statsResponse = await request.get(
      `${API_BASE_URL}/pgvector/collections/${collectionName}/stats`
    )
    expect(statsResponse.ok()).toBeTruthy()

    const statsData = await statsResponse.json()
    expect(statsData.status).toBe('success')
    expect(statsData.data).toBeDefined()
    expect(statsData.data.total_vectors).toBeGreaterThan(0)

    console.log('集合统计信息:', JSON.stringify(statsData.data, null, 2))

    // 获取性能报告
    const performanceResponse = await request.get(
      `${API_BASE_URL}/pgvector/monitoring/performance-report?time_range_hours=1`
    )
    expect(performanceResponse.ok()).toBeTruthy()

    const performanceData = await performanceResponse.json()
    expect(performanceData.status).toBe('success')
    expect(performanceData.data).toBeDefined()

    console.log('性能报告生成成功')
    if (performanceData.data.query_performance) {
      console.log(
        '查询性能统计:',
        JSON.stringify(performanceData.data.query_performance, null, 2)
      )
    }
  })

  test('混合搜索功能测试', async ({ request }) => {
    const collectionName = `${TEST_COLLECTION}_hnsw`

    const hybridSearchRequest = {
      collection_name: collectionName,
      query_vector: TEST_VECTORS[0].embedding,
      query_text: '人工智能',
      limit: 3,
      vector_weight: 0.7,
      text_weight: 0.3,
    }

    const hybridResponse = await request.post(
      `${API_BASE_URL}/pgvector/search/hybrid`,
      {
        data: hybridSearchRequest,
      }
    )
    expect(hybridResponse.ok()).toBeTruthy()

    const hybridData = await hybridResponse.json()
    expect(hybridData.status).toBe('success')
    expect(hybridData.results).toBeDefined()

    console.log('混合搜索完成:', hybridData.results.length, '个结果')
    if (hybridData.results.length > 0) {
      console.log('第一个结果:', hybridData.results[0].content)
    }
  })

  test('索引优化和维护功能', async ({ request }) => {
    const collectionName = `${TEST_COLLECTION}_hnsw`

    // 触发集合优化
    const optimizeResponse = await request.post(
      `${API_BASE_URL}/pgvector/collections/${collectionName}/optimize`
    )
    expect(optimizeResponse.ok()).toBeTruthy()

    const optimizeData = await optimizeResponse.json()
    expect(optimizeData.status).toBe('success')
    console.log('集合优化完成:', optimizeData.message)

    // 验证优化后的统计信息
    const statsResponse = await request.get(
      `${API_BASE_URL}/pgvector/collections/${collectionName}/stats`
    )
    expect(statsResponse.ok()).toBeTruthy()

    const statsData = await statsResponse.json()
    console.log('优化后统计:', statsData.data.dead_tuples, '死元组')
  })

  test('错误处理和边界条件测试', async ({ request }) => {
    // 测试不存在的集合
    const nonExistentCollection = 'non_existent_collection'
    const searchResponse = await request.post(
      `${API_BASE_URL}/pgvector/search/similarity`,
      {
        data: {
          collection_name: nonExistentCollection,
          query_vector: [1, 2, 3],
          limit: 5,
        },
      }
    )
    expect(searchResponse.status()).toBe(500)

    // 测试错误的向量维度
    const wrongDimensionRequest = {
      collection_name: `${TEST_COLLECTION}_hnsw`,
      query_vector: [1, 2, 3], // 错误的维度
      limit: 5,
    }

    const wrongDimResponse = await request.post(
      `${API_BASE_URL}/pgvector/search/similarity`,
      {
        data: wrongDimensionRequest,
      }
    )
    expect(wrongDimResponse.status()).toBe(500)

    console.log('错误处理测试通过')
  })

  test.afterAll(async ({ request }) => {
    // 清理测试数据
    const collections = [`${TEST_COLLECTION}_hnsw`, `${TEST_COLLECTION}_ivf`]

    for (const collection of collections) {
      try {
        const deleteResponse = await request.delete(
          `${API_BASE_URL}/pgvector/collections/${collection}`
        )
        if (deleteResponse.ok()) {
          console.log(`清理集合: ${collection}`)
        }
      } catch (error) {
        console.warn(`清理集合失败: ${collection}`, error)
      }
    }

    console.log('E2E测试清理完成')
  })
})

test.describe('pgvector 0.8.0 性能基准测试', () => {
  const BENCHMARK_COLLECTION = 'benchmark_pgvector'
  const BENCHMARK_VECTORS_COUNT = 100

  test('大规模向量插入性能基准', async ({ request }) => {
    // 创建基准测试集合
    const collectionRequest = {
      collection_name: BENCHMARK_COLLECTION,
      dimension: 1536,
      index_type: 'hnsw',
      distance_metric: 'l2',
      index_options: {
        m: 16,
        ef_construction: 200,
      },
    }

    const createResponse = await request.post(
      `${API_BASE_URL}/pgvector/collections`,
      {
        data: collectionRequest,
      }
    )
    expect(createResponse.ok()).toBeTruthy()

    // 生成大量测试向量
    const benchmarkVectors = Array.from(
      { length: BENCHMARK_VECTORS_COUNT },
      (_, i) => ({
        content: `基准测试向量 ${i + 1}`,
        embedding: Array.from({ length: 1536 }, () => Math.random() * 2 - 1),
        metadata: { index: i, category: `benchmark_${Math.floor(i / 10)}` },
      })
    )

    // 批量插入性能测试
    const insertStartTime = Date.now()
    const batchSize = 10

    for (let i = 0; i < benchmarkVectors.length; i += batchSize) {
      const batch = benchmarkVectors.slice(i, i + batchSize)

      const insertResponse = await request.post(
        `${API_BASE_URL}/pgvector/vectors`,
        {
          data: {
            collection_name: BENCHMARK_COLLECTION,
            documents: batch,
          },
        }
      )
      expect(insertResponse.ok()).toBeTruthy()
    }

    const insertTime = Date.now() - insertStartTime
    const insertRate = (BENCHMARK_VECTORS_COUNT / insertTime) * 1000

    console.log(
      `批量插入基准: ${BENCHMARK_VECTORS_COUNT}个向量, 耗时${insertTime}ms, 速率${insertRate.toFixed(2)}向量/秒`
    )
    expect(insertRate).toBeGreaterThan(10) // 至少10向量/秒
  })

  test('大规模相似性搜索性能基准', async ({ request }) => {
    // 执行多次搜索以获得平均性能
    const searchIterations = 20
    const searchTimes = []

    for (let i = 0; i < searchIterations; i++) {
      const queryVector = Array.from(
        { length: 1536 },
        () => Math.random() * 2 - 1
      )

      const searchStartTime = Date.now()
      const searchResponse = await request.post(
        `${API_BASE_URL}/pgvector/search/similarity`,
        {
          data: {
            collection_name: BENCHMARK_COLLECTION,
            query_vector: queryVector,
            limit: 10,
            distance_metric: 'l2',
            include_distances: true,
          },
        }
      )
      const searchTime = Date.now() - searchStartTime

      expect(searchResponse.ok()).toBeTruthy()
      const searchData = await searchResponse.json()
      expect(searchData.results.length).toBeGreaterThan(0)

      searchTimes.push(searchTime)
    }

    const avgSearchTime =
      searchTimes.reduce((a, b) => a + b) / searchTimes.length
    const maxSearchTime = Math.max(...searchTimes)
    const minSearchTime = Math.min(...searchTimes)

    console.log(
      `搜索性能基准: 平均${avgSearchTime.toFixed(2)}ms, 最快${minSearchTime}ms, 最慢${maxSearchTime}ms`
    )
    expect(avgSearchTime).toBeLessThan(1000) // 平均搜索时间小于1秒
  })

  test.afterAll(async ({ request }) => {
    // 清理基准测试数据
    try {
      const deleteResponse = await request.delete(
        `${API_BASE_URL}/pgvector/collections/${BENCHMARK_COLLECTION}`
      )
      if (deleteResponse.ok()) {
        console.log('基准测试数据清理完成')
      }
    } catch (error) {
      console.warn('基准测试数据清理失败', error)
    }
  })
})
