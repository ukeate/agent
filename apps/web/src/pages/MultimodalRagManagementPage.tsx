/**
 * 多模态RAG管理页面
 * 完整利用multimodal_rag.py的8个API端点
 */

import React, { useState, useEffect, useRef } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Row,
  Col,
  Input,
  Button,
  Upload,
  Typography,
  Divider,
  Space,
  Switch,
  Slider,
  InputNumber,
  Progress,
  Tag,
  List,
  Alert,
  Tabs,
  Spin,
  Badge,
  Modal,
  message,
  Table,
  Statistic,
} from 'antd'
import {
  UploadOutlined,
  SearchOutlined,
  DeleteOutlined,
  ClearOutlined,
  EyeOutlined,
  FileOutlined,
  PictureOutlined,
  TableOutlined,
  CloudUploadOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  QuestionCircleOutlined,
} from '@ant-design/icons'
import type { UploadProps, UploadFile } from 'antd'
import { multimodalRagService } from '../services/multimodalRagService'

const { Title, Paragraph, Text } = Typography
const { TextArea } = Input
const { TabPane } = Tabs

interface MultimodalQueryResponse {
  answer: string
  sources: string[]
  confidence: number
  processing_time: number
  context_used: Record<string, number>
}

interface DocumentUploadResponse {
  doc_id: string
  source_file: string
  content_type: string
  num_text_chunks: number
  num_images: number
  num_tables: number
  processing_time: number
}

interface VectorStoreStatus {
  text_documents: number
  image_documents: number
  table_documents: number
  total_documents: number
  embedding_dimension: number
}

const MultimodalRagManagementPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [streamLoading, setStreamLoading] = useState(false)
  const [uploadLoading, setUploadLoading] = useState(false)
  const [batchUploadLoading, setBatchUploadLoading] = useState(false)
  const [statusLoading, setStatusLoading] = useState(false)

  // 查询相关状态
  const [query, setQuery] = useState('')
  const [queryResult, setQueryResult] =
    useState<MultimodalQueryResponse | null>(null)
  const [streamResult, setStreamResult] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const streamDivRef = useRef<HTMLDivElement>(null)

  // 查询参数
  const [includeImages, setIncludeImages] = useState(true)
  const [includeTables, setIncludeTables] = useState(true)
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(1000)
  const [topK, setTopK] = useState(5)

  // 文件上传状态
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [batchFileList, setBatchFileList] = useState<UploadFile[]>([])
  const [uploadResults, setUploadResults] = useState<DocumentUploadResponse[]>(
    []
  )
  const [batchUploadResults, setBatchUploadResults] = useState<any>(null)

  // 系统状态
  const [vectorStoreStatus, setVectorStoreStatus] =
    useState<VectorStoreStatus | null>(null)

  // 查询历史
  const [queryHistory, setQueryHistory] = useState<
    Array<{
      query: string
      timestamp: Date
      result?: MultimodalQueryResponse
    }>
  >([])

  // 组件加载时获取状态
  useEffect(() => {
    handleGetStatus()
  }, [])

  // 自动滚动到最新的流式内容
  useEffect(() => {
    if (streamDivRef.current && isStreaming) {
      streamDivRef.current.scrollTop = streamDivRef.current.scrollHeight
    }
  }, [streamResult, isStreaming])

  // API端点1: 普通查询 /query
  const handleQuery = async () => {
    if (!query.trim()) {
      message.warning('请输入查询内容')
      return
    }

    setLoading(true)
    try {
      const result = await multimodalRagService.query(query, {
        includeImages,
        includeTables,
        temperature,
        maxTokens,
        topK,
      })

      setQueryResult(result)

      // 添加到查询历史
      setQueryHistory(prev => [
        {
          query,
          timestamp: new Date(),
          result,
        },
        ...prev.slice(0, 9),
      ]) // 保留最近10条

      message.success('查询完成')
    } catch (error) {
      message.error('查询失败')
      logger.error('查询错误:', error)
    } finally {
      setLoading(false)
    }
  }

  // API端点2: 带文件查询 /query-with-files
  const handleQueryWithFiles = async () => {
    if (!query.trim()) {
      message.warning('请输入查询内容')
      return
    }

    if (fileList.length === 0) {
      message.warning('请选择文件')
      return
    }

    setLoading(true)
    try {
      const files = fileList.map(file => file.originFileObj as File)
      const result = await multimodalRagService.queryWithFiles(query, files)

      setQueryResult(result)
      message.success('文件查询完成')
    } catch (error) {
      message.error('文件查询失败')
      logger.error('文件查询错误:', error)
    } finally {
      setLoading(false)
    }
  }

  // API端点3: 流式查询 /stream-query
  const handleStreamQuery = async () => {
    if (!query.trim()) {
      message.warning('请输入查询内容')
      return
    }

    setStreamLoading(true)
    setIsStreaming(true)
    setStreamResult('')

    try {
      const stream = await multimodalRagService.streamQuery(query, {
        includeImages,
        includeTables,
        temperature,
        maxTokens,
        topK,
      })

      const reader = stream.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n')
        buffer = lines.pop() || ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = line.slice(6).trim()
            if (data === '[DONE]') {
              break
            }

            try {
              const payload = JSON.parse(data)
              if (payload?.delta) {
                setStreamResult(prev => prev + String(payload.delta))
              }
              if (payload?.error) {
                setStreamResult(
                  prev => prev + '\n[错误] ' + String(payload.error)
                )
                break
              }
            } catch {
              setStreamResult(prev => prev + data)
            }
          }
        }
      }

      message.success('流式查询完成')
    } catch (error) {
      message.error('流式查询失败')
      setStreamResult(prev => prev + '\n[错误] 流式查询服务不可用')
      logger.error('流式查询错误:', error)
    } finally {
      setStreamLoading(false)
      setIsStreaming(false)
    }
  }

  // API端点4: 单文档上传 /upload-document
  const handleSingleUpload = async (file: File) => {
    setUploadLoading(true)
    try {
      const result = await multimodalRagService.uploadSingleDocument(file)
      setUploadResults(prev => [result, ...prev])
      message.success(`文档 ${file.name} 上传成功`)
      await handleGetStatus() // 更新状态
    } catch (error) {
      message.error(`文档 ${file.name} 上传失败`)
      logger.error('上传错误:', error)
    } finally {
      setUploadLoading(false)
    }
  }

  // API端点5: 批量上传 /batch-upload
  const handleBatchUpload = async () => {
    if (batchFileList.length === 0) {
      message.warning('请选择要批量上传的文件')
      return
    }

    setBatchUploadLoading(true)
    try {
      const files = batchFileList.map(file => file.originFileObj as File)
      const result = await multimodalRagService.batchUploadDocuments(files)
      setBatchUploadResults(result)
      message.success(
        `批量上传完成: ${result.successful}个成功, ${result.failed}个失败`
      )
      await handleGetStatus() // 更新状态
    } catch (error) {
      message.error('批量上传失败')
      logger.error('批量上传错误:', error)
    } finally {
      setBatchUploadLoading(false)
    }
  }

  // API端点6: 获取状态 /status
  const handleGetStatus = async () => {
    setStatusLoading(true)
    try {
      const status = await multimodalRagService.getSystemStatus()
      setVectorStoreStatus(status)
    } catch (error) {
      message.error('获取状态失败')
      logger.error('状态获取错误:', error)
    } finally {
      setStatusLoading(false)
    }
  }

  // API端点7: 清空向量存储 /clear
  const handleClearVectorStore = async () => {
    Modal.confirm({
      title: '确认清空向量存储',
      content: '这将删除所有已索引的文档数据，此操作不可撤销！',
      okText: '确认清空',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        try {
          await multimodalRagService.clearVectorDatabase()
          message.success('向量存储已清空')
          await handleGetStatus() // 更新状态
          setUploadResults([])
          setBatchUploadResults(null)
        } catch (error) {
          message.error('清空向量存储失败')
          logger.error('清空错误:', error)
        }
      },
    })
  }

  // API端点8: 清空缓存 /cache
  const handleClearCache = async () => {
    try {
      await multimodalRagService.clearCache()
      message.success('缓存已清空')
    } catch (error) {
      message.error('清空缓存失败')
      logger.error('清空缓存错误:', error)
    }
  }

  // 上传配置
  const uploadProps: UploadProps = {
    fileList,
    onChange: ({ fileList }) => setFileList(fileList),
    beforeUpload: () => false,
    multiple: true,
    accept: '.txt,.pdf,.docx,.xlsx,.jpg,.jpeg,.png,.md',
  }

  const batchUploadProps: UploadProps = {
    fileList: batchFileList,
    onChange: ({ fileList }) => setBatchFileList(fileList),
    beforeUpload: () => false,
    multiple: true,
    accept: '.txt,.pdf,.docx,.xlsx,.jpg,.jpeg,.png,.md',
  }

  const singleUploadProps: UploadProps = {
    beforeUpload: file => {
      handleSingleUpload(file)
      return false
    },
    showUploadList: false,
  }

  // 查询历史表格列
  const historyColumns = [
    {
      title: '查询内容',
      dataIndex: 'query',
      key: 'query',
      ellipsis: true,
      width: '40%',
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: '30%',
      render: (timestamp: Date) => timestamp.toLocaleString(),
    },
    {
      title: '置信度',
      dataIndex: 'result',
      key: 'confidence',
      width: '15%',
      render: (result?: MultimodalQueryResponse) =>
        result ? `${(result.confidence * 100).toFixed(1)}%` : '-',
    },
    {
      title: '处理时间',
      dataIndex: 'result',
      key: 'processing_time',
      width: '15%',
      render: (result?: MultimodalQueryResponse) =>
        result ? `${(result.processing_time * 1000).toFixed(0)}ms` : '-',
    },
  ]

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <Title level={2}>
        <ThunderboltOutlined /> 多模态RAG管理系统
      </Title>

      <Paragraph>
        完整的多模态检索增强生成系统，支持文本、图像、表格等多种内容类型的智能查询和文档管理。
        本系统完整利用了multimodal_rag.py的全部8个API端点。
      </Paragraph>

      {/* 系统状态面板 */}
      <Card style={{ marginBottom: 24 }}>
        <Row gutter={16}>
          <Col span={6}>
            <Statistic
              title="文档总数"
              value={vectorStoreStatus?.total_documents || 0}
              prefix={<FileOutlined />}
              loading={statusLoading}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="文本文档"
              value={vectorStoreStatus?.text_documents || 0}
              prefix={<FileOutlined />}
              loading={statusLoading}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="图像文档"
              value={vectorStoreStatus?.image_documents || 0}
              prefix={<PictureOutlined />}
              loading={statusLoading}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="表格文档"
              value={vectorStoreStatus?.table_documents || 0}
              prefix={<TableOutlined />}
              loading={statusLoading}
            />
          </Col>
        </Row>

        <Divider />

        <Space>
          <Button
            onClick={handleGetStatus}
            icon={<BarChartOutlined />}
            loading={statusLoading}
          >
            刷新状态
          </Button>
          <Button
            danger
            onClick={handleClearVectorStore}
            icon={<DeleteOutlined />}
          >
            清空向量存储
          </Button>
          <Button onClick={handleClearCache} icon={<ClearOutlined />}>
            清空缓存
          </Button>
        </Space>
      </Card>

      <Tabs defaultActiveKey="query" type="card">
        {/* 查询功能 */}
        <TabPane tab="智能查询" key="query">
          <Row gutter={24}>
            <Col span={12}>
              <Card title="查询配置" size="small">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <TextArea
                    placeholder="输入您的查询问题..."
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    name="multimodal-query"
                    rows={4}
                  />

                  <Row gutter={16}>
                    <Col span={12}>
                      <Text>包含图像:</Text>
                      <Switch
                        checked={includeImages}
                        onChange={setIncludeImages}
                        style={{ marginLeft: 8 }}
                      />
                    </Col>
                    <Col span={12}>
                      <Text>包含表格:</Text>
                      <Switch
                        checked={includeTables}
                        onChange={setIncludeTables}
                        style={{ marginLeft: 8 }}
                      />
                    </Col>
                  </Row>

                  <Row gutter={16}>
                    <Col span={12}>
                      <Text>温度参数:</Text>
                      <Slider
                        min={0}
                        max={1}
                        step={0.1}
                        value={temperature}
                        onChange={setTemperature}
                        tooltip={{ formatter: value => `${value}` }}
                      />
                    </Col>
                    <Col span={12}>
                      <Text>最大Token:</Text>
                      <InputNumber
                        min={100}
                        max={4000}
                        value={maxTokens}
                        onChange={value => setMaxTokens(value || 1000)}
                        name="multimodal-max-tokens"
                        style={{ width: '100%' }}
                      />
                    </Col>
                  </Row>

                  <Row gutter={16}>
                    <Col span={12}>
                      <Text>检索数量 (Top-K):</Text>
                      <InputNumber
                        min={1}
                        max={20}
                        value={topK}
                        onChange={value => setTopK(value || 5)}
                        name="multimodal-topk"
                        style={{ width: '100%' }}
                      />
                    </Col>
                  </Row>

                  <Space>
                    <Button
                      type="primary"
                      icon={<SearchOutlined />}
                      onClick={handleQuery}
                      loading={loading}
                    >
                      普通查询
                    </Button>
                    <Button
                      icon={<ThunderboltOutlined />}
                      onClick={handleStreamQuery}
                      loading={streamLoading}
                    >
                      流式查询
                    </Button>
                  </Space>
                </Space>
              </Card>

              {/* 文件查询 */}
              <Card title="文件查询" size="small" style={{ marginTop: 16 }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Upload {...uploadProps}>
                    <Button icon={<UploadOutlined />}>选择文件</Button>
                  </Upload>

                  <Button
                    type="primary"
                    icon={<SearchOutlined />}
                    onClick={handleQueryWithFiles}
                    loading={loading}
                    disabled={fileList.length === 0}
                    block
                  >
                    带文件查询
                  </Button>
                </Space>
              </Card>
            </Col>

            <Col span={12}>
              {/* 查询结果 */}
              <Card title="查询结果" size="small">
                {queryResult && (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Alert
                      message="查询成功"
                      description={
                        <div>
                          <Text>
                            置信度: {(queryResult.confidence * 100).toFixed(1)}%
                          </Text>
                          <br />
                          <Text>
                            处理时间:{' '}
                            {(queryResult.processing_time * 1000).toFixed(0)}ms
                          </Text>
                        </div>
                      }
                      type="success"
                      showIcon
                    />

                    <Card size="small" title="答案">
                      <Text>{queryResult.answer}</Text>
                    </Card>

                    {queryResult.sources.length > 0 && (
                      <Card size="small" title="引用来源">
                        <List
                          size="small"
                          dataSource={queryResult.sources}
                          renderItem={source => (
                            <List.Item>
                              <Tag icon={<FileOutlined />} color="blue">
                                {source}
                              </Tag>
                            </List.Item>
                          )}
                        />
                      </Card>
                    )}

                    {Object.keys(queryResult.context_used).length > 0 && (
                      <Card size="small" title="上下文使用统计">
                        {Object.entries(queryResult.context_used).map(
                          ([key, value]) => (
                            <div key={key}>
                              <Text strong>{key}:</Text> <Text>{value}</Text>
                            </div>
                          )
                        )}
                      </Card>
                    )}
                  </Space>
                )}
              </Card>

              {/* 流式结果 */}
              <Card title="流式查询结果" size="small" style={{ marginTop: 16 }}>
                <div
                  ref={streamDivRef}
                  style={{
                    height: '200px',
                    overflowY: 'auto',
                    border: '1px solid #d9d9d9',
                    padding: '8px',
                    backgroundColor: '#fafafa',
                    whiteSpace: 'pre-wrap',
                    fontFamily: 'monospace',
                  }}
                >
                  {isStreaming && <Spin size="small" />}
                  {streamResult || '流式查询结果将在此显示...'}
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>

        {/* 文档管理 */}
        <TabPane tab="文档管理" key="upload">
          <Row gutter={24}>
            <Col span={12}>
              <Card title="单文档上传" size="small">
                <Upload
                  {...singleUploadProps}
                  accept=".txt,.pdf,.docx,.xlsx,.jpg,.jpeg,.png,.md"
                >
                  <Button
                    icon={<CloudUploadOutlined />}
                    loading={uploadLoading}
                    block
                  >
                    选择文件上传
                  </Button>
                </Upload>

                <Divider />

                <Text type="secondary">
                  支持格式: TXT, PDF, DOCX, XLSX, JPG, JPEG, PNG, MD
                </Text>
              </Card>

              <Card title="批量上传" size="small" style={{ marginTop: 16 }}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Upload {...batchUploadProps}>
                    <Button icon={<UploadOutlined />}>选择多个文件</Button>
                  </Upload>

                  <Button
                    type="primary"
                    icon={<CloudUploadOutlined />}
                    onClick={handleBatchUpload}
                    loading={batchUploadLoading}
                    disabled={batchFileList.length === 0}
                    block
                  >
                    批量上传 ({batchFileList.length} 个文件)
                  </Button>
                </Space>
              </Card>
            </Col>

            <Col span={12}>
              <Card title="上传结果" size="small">
                <List
                  dataSource={uploadResults}
                  renderItem={result => (
                    <List.Item>
                      <List.Item.Meta
                        avatar={
                          <Badge
                            count={
                              result.num_text_chunks +
                              result.num_images +
                              result.num_tables
                            }
                          />
                        }
                        title={result.source_file}
                        description={
                          <Space size="small">
                            <Tag>文本块: {result.num_text_chunks}</Tag>
                            <Tag>图像: {result.num_images}</Tag>
                            <Tag>表格: {result.num_tables}</Tag>
                            <Tag>
                              耗时: {(result.processing_time * 1000).toFixed(0)}
                              ms
                            </Tag>
                          </Space>
                        }
                      />
                    </List.Item>
                  )}
                />
              </Card>

              {batchUploadResults && (
                <Card
                  title="批量上传结果"
                  size="small"
                  style={{ marginTop: 16 }}
                >
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Progress
                      percent={Math.round(
                        (batchUploadResults.successful /
                          batchUploadResults.total_files) *
                          100
                      )}
                      status={
                        batchUploadResults.failed > 0 ? 'exception' : 'success'
                      }
                    />
                    <div>
                      <Text>总文件: {batchUploadResults.total_files}</Text>
                      <br />
                      <Text type="success">
                        成功: {batchUploadResults.successful}
                      </Text>
                      <br />
                      <Text type="danger">
                        失败: {batchUploadResults.failed}
                      </Text>
                      <br />
                      <Text type="secondary">
                        处理时间:{' '}
                        {(
                          (batchUploadResults.processing_time || 0) * 1000
                        ).toFixed(0)}
                        ms
                      </Text>
                    </div>
                  </Space>
                </Card>
              )}
            </Col>
          </Row>
        </TabPane>

        {/* 查询历史 */}
        <TabPane tab="查询历史" key="history">
          <Card>
            <Table
              dataSource={queryHistory}
              columns={historyColumns}
              pagination={{ pageSize: 10 }}
              size="small"
              rowKey={record => `${record.query}-${record.timestamp.getTime()}`}
            />
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default MultimodalRagManagementPage
