/**
 * 多模态RAG系统页面
 *
 * 技术展示重点：
 * 1. 文档处理管道可视化
 * 2. 查询类型分析展示
 * 3. 检索策略决策过程
 * 4. 向量存储状态监控
 * 5. 多模态结果展示
 */

import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Tabs,
  message,
  Typography,
  Space,
  Statistic,
  Button,
  Input,
  Upload,
  Modal,
} from 'antd'
import { logger } from '../utils/logger'
import {
  FileImageOutlined,
  FileTextOutlined,
  TableOutlined,
  SearchOutlined,
  DatabaseOutlined,
  ThunderboltOutlined,
  UploadOutlined,
  ClearOutlined,
  DeleteOutlined,
} from '@ant-design/icons'
import DocumentUploader from '../components/multimodal/DocumentUploader'
import QueryAnalyzer from '../components/multimodal/QueryAnalyzer'
import RetrievalStrategy from '../components/multimodal/RetrievalStrategy'
import VectorStoreStatus from '../components/multimodal/VectorStoreStatus'
import MultimodalResults from '../components/multimodal/MultimodalResults'
import QueryInterface from '../components/multimodal/QueryInterface'
import {
  multimodalRagService,
  MultimodalRagStats,
} from '../services/multimodalRagService'

const { Title, Text } = Typography
const { TabPane } = Tabs

interface SystemStats {
  totalDocuments: number
  textChunks: number
  images: number
  tables: number
  embeddingDimension: number
  cacheHitRate: number
}

const MultimodalRagPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [systemStats, setSystemStats] = useState<SystemStats>({
    totalDocuments: 0,
    textChunks: 0,
    images: 0,
    tables: 0,
    embeddingDimension: 0,
    cacheHitRate: 0,
  })
  const [systemDetails, setSystemDetails] = useState<MultimodalRagStats | null>(
    null
  )

  // 查询分析结果
  const [queryAnalysis, setQueryAnalysis] = useState<any>(null)

  // 检索策略信息
  const [retrievalStrategy, setRetrievalStrategy] = useState<any>(null)

  // 检索结果
  const [retrievalResults, setRetrievalResults] = useState<any>(null)

  // 最终答案
  const [qaResponse, setQaResponse] = useState<any>(null)

  // 快速查询相关状态
  const [quickQuery, setQuickQuery] = useState<string>('')
  const [simpleQueryResult, setSimpleQueryResult] = useState<any>(null)

  useEffect(() => {
    loadSystemStatus()
  }, [])

  const loadSystemStatus = async () => {
    try {
      const status = await multimodalRagService.getSystemStatus()
      setSystemDetails(status)
      setSystemStats({
        totalDocuments: status.total_documents,
        textChunks: status.text_documents,
        images: status.image_documents,
        tables: status.table_documents,
        embeddingDimension: status.embedding_dimension,
        cacheHitRate: (status.cache?.hit_rate || 0) * 100,
      })
    } catch (error) {
      logger.error('加载系统状态失败:', error)
    }
  }

  const handleQuery = async (query: string, files?: File[]) => {
    setLoading(true)
    try {
      // 执行查询并获取完整的技术细节
      const response = await multimodalRagService.queryWithDetails(query, files)

      // 更新各个技术组件的状态
      setQueryAnalysis(response.query_analysis)
      setRetrievalStrategy(response.retrieval_strategy)
      setRetrievalResults(response.retrieval_results)
      setQaResponse(response.qa_response)

      message.success('查询执行成功')
    } catch (error) {
      message.error('查询失败: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const handleDocumentUploaded = () => {
    loadSystemStatus()
    message.success('文档已成功添加到向量存储')
  }

  // 新增：简单查询功能 - 使用未使用的API
  const handleSimpleQuery = async (query: string) => {
    setLoading(true)
    try {
      const response = await multimodalRagService.query(query, {
        topK: 5,
        includeImages: true,
        includeTables: true,
      })
      setSimpleQueryResult(response)
      message.success('简单查询完成')
    } catch (error) {
      message.error('简单查询失败: ' + error.message)
      setSimpleQueryResult(null)
    } finally {
      setLoading(false)
    }
  }

  // 新增：批量文档上传功能 - 使用未使用的API
  const handleBatchUpload = async (files: File[]) => {
    setLoading(true)
    try {
      const response = await multimodalRagService.batchUploadDocuments(files)
      message.success(
        `批量上传完成: ${response.successful || 0}/${files.length} 个文件`
      )
      loadSystemStatus() // 刷新系统状态
    } catch (error) {
      message.error('批量上传失败: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  // 新增：清空向量数据库功能 - 使用未使用的API
  const handleClearDatabase = async () => {
    setLoading(true)
    try {
      await multimodalRagService.clearVectorDatabase()
      message.success('向量数据库已清空')
      loadSystemStatus() // 刷新系统状态
    } catch (error) {
      message.error('清空向量数据库失败: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  // 新增：清空缓存功能 - 使用未使用的API
  const handleClearCache = async () => {
    setLoading(true)
    try {
      await multimodalRagService.clearCache()
      message.success('系统缓存已清空')
    } catch (error) {
      message.error('清空缓存失败: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="p-6">
      {/* 页面标题和系统概览 */}
      <Card className="mb-4">
        <Title level={2}>
          <ThunderboltOutlined className="mr-2" />
          多模态RAG系统 - 技术实现展示
        </Title>
        <Text type="secondary">
          展示LangChain多模态RAG的完整技术栈：文档处理、向量存储、智能检索、上下文组装
        </Text>

        <Row gutter={16} className="mt-4">
          <Col span={4}>
            <Statistic
              title="总文档数"
              value={systemStats.totalDocuments}
              prefix={<DatabaseOutlined />}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="文本块"
              value={systemStats.textChunks}
              prefix={<FileTextOutlined />}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="图像"
              value={systemStats.images}
              prefix={<FileImageOutlined />}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="表格"
              value={systemStats.tables}
              prefix={<TableOutlined />}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="嵌入维度"
              value={systemStats.embeddingDimension}
            />
          </Col>
          <Col span={4}>
            <Statistic
              title="缓存命中率"
              value={systemStats.cacheHitRate}
              suffix="%"
            />
          </Col>
        </Row>
      </Card>

      {/* 主要功能标签页 */}
      <Tabs defaultActiveKey="query" size="large">
        <TabPane
          tab={
            <span>
              <SearchOutlined />
              多模态查询
            </span>
          }
          key="query"
        >
          <Row gutter={16}>
            {/* 左侧：查询输入 */}
            <Col span={8}>
              <QueryInterface onQuery={handleQuery} loading={loading} />
            </Col>

            {/* 中间：技术过程展示 */}
            <Col span={8}>
              <Space
                direction="vertical"
                style={{ width: '100%' }}
                size="middle"
              >
                {/* 查询分析展示 */}
                <QueryAnalyzer analysis={queryAnalysis} />

                {/* 检索策略展示 */}
                <RetrievalStrategy strategy={retrievalStrategy} />
              </Space>
            </Col>

            {/* 右侧：结果展示 */}
            <Col span={8}>
              <MultimodalResults
                retrievalResults={retrievalResults}
                qaResponse={qaResponse}
              />
            </Col>
          </Row>
        </TabPane>

        <TabPane
          tab={
            <span>
              <DatabaseOutlined />
              文档管理
            </span>
          }
          key="documents"
        >
          <Row gutter={16}>
            <Col span={12}>
              <DocumentUploader onUploadSuccess={handleDocumentUploaded} />

              {/* 新增：批量上传功能 */}
              <Card title="批量文档上传" className="mt-4">
                <Upload
                  multiple
                  beforeUpload={(file, fileList) => {
                    handleBatchUpload(Array.from(fileList))
                    return false // 阻止默认上传行为
                  }}
                  showUploadList={false}
                >
                  <Button icon={<UploadOutlined />} loading={loading}>
                    选择多个文件批量上传
                  </Button>
                </Upload>
                <Typography.Text type="secondary" className="block mt-2">
                  支持并行处理，自动提取图像和表格
                </Typography.Text>
              </Card>
            </Col>
            <Col span={12}>
              <VectorStoreStatus stats={systemStats} />

              {/* 新增：数据库管理功能 */}
              <Card title="数据库管理" className="mt-4">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Button
                    icon={<DeleteOutlined />}
                    danger
                    loading={loading}
                    onClick={() => {
                      Modal.confirm({
                        title: '确认清空向量数据库？',
                        content: '此操作将删除所有已存储的文档向量，不可恢复。',
                        okText: '确认清空',
                        cancelText: '取消',
                        okType: 'danger',
                        onOk: handleClearDatabase,
                      })
                    }}
                  >
                    清空向量数据库
                  </Button>
                  <Button
                    icon={<ClearOutlined />}
                    loading={loading}
                    onClick={() => {
                      Modal.confirm({
                        title: '确认清空系统缓存？',
                        content: '此操作将清空查询缓存，提升查询响应速度。',
                        okText: '确认清空',
                        cancelText: '取消',
                        onOk: handleClearCache,
                      })
                    }}
                  >
                    清空系统缓存
                  </Button>
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane
          tab={
            <span>
              <SearchOutlined />
              快速查询
            </span>
          }
          key="quick-query"
        >
          <Row gutter={16}>
            <Col span={24}>
              <Card title="简单查询接口">
                <Space.Compact style={{ width: '100%' }}>
                  <Input
                    placeholder="输入查询问题..."
                    value={quickQuery}
                    onChange={e => setQuickQuery(e.target.value)}
                    onPressEnter={() => handleSimpleQuery(quickQuery)}
                  />
                  <Button
                    type="primary"
                    loading={loading}
                    onClick={() => handleSimpleQuery(quickQuery)}
                    disabled={!quickQuery.trim()}
                  >
                    查询
                  </Button>
                </Space.Compact>
                <Typography.Text type="secondary" className="block mt-2">
                  使用简化的查询接口，自动配置相似度阈值和检索参数
                </Typography.Text>

                {simpleQueryResult && (
                  <Card className="mt-4" size="small">
                    <Typography.Title level={5}>查询结果</Typography.Title>
                    <pre style={{ whiteSpace: 'pre-wrap', fontSize: '12px' }}>
                      {JSON.stringify(simpleQueryResult, null, 2)}
                    </pre>
                  </Card>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane
          tab={
            <span>
              <ThunderboltOutlined />
              系统监控
            </span>
          }
          key="monitor"
        >
          <Card title="系统监控">
            {systemDetails ? (
              <pre style={{ whiteSpace: 'pre-wrap', fontSize: '12px' }}>
                {JSON.stringify(systemDetails, null, 2)}
              </pre>
            ) : (
              <Typography.Text type="secondary">暂无监控数据。</Typography.Text>
            )}
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default MultimodalRagPage
