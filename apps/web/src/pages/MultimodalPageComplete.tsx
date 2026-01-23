import React, { useState, useEffect, useRef } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Tabs,
  Upload,
  Button,
  Select,
  Radio,
  Space,
  message,
  Alert,
  Progress,
  Divider,
  Typography,
  Row,
  Col,
  Tag,
  Statistic,
  Table,
  Spin,
  Badge,
  Descriptions,
  Empty,
  Input,
  Modal,
  Form,
  Checkbox,
} from 'antd'
import {
  UploadOutlined,
  FileImageOutlined,
  FileTextOutlined,
  VideoCameraOutlined,
  AudioOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  DeleteOutlined,
  EyeOutlined,
  DownloadOutlined,
  ApiOutlined,
  DollarOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  CloudUploadOutlined,
} from '@ant-design/icons'
import {
  multimodalService,
  ContentType,
  ModelPriority,
  ModelComplexity,
  ProcessingStatus,
  type FileUploadResponse,
  type ProcessingResponse,
  type ProcessingStatusResponse,
  type QueueStatus,
  type MultimodalModelConfig,
  type TokenUsage,
} from '../services/multimodalService'

const { TabPane } = Tabs
const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TextArea } = Input

interface ProcessingTask {
  contentId: string
  fileName: string
  fileType: string
  status: ProcessingStatus
  progress: number
  model: string
  cost?: number
  processingTime?: number
  result?: ProcessingResponse
}

const MultimodalPageComplete: React.FC = () => {
  const [activeTab, setActiveTab] = useState('upload')
  const [selectedModel, setSelectedModel] = useState('gpt-4o-mini')
  const [processingPriority, setProcessingPriority] = useState<ModelPriority>(
    ModelPriority.BALANCED
  )
  const [processingComplexity, setProcessingComplexity] =
    useState<ModelComplexity>(ModelComplexity.MEDIUM)
  const [processingTasks, setProcessingTasks] = useState<
    Map<string, ProcessingTask>
  >(new Map())
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [queueStatus, setQueueStatus] = useState<QueueStatus | null>(null)
  const [quickAnalysisPrompt, setQuickAnalysisPrompt] = useState('')
  const [quickAnalysisFile, setQuickAnalysisFile] = useState<File | null>(null)
  const [costStats, setCostStats] = useState({
    daily: 0,
    totalTokens: 0,
    modelDistribution: new Map<string, number>(),
  })
  const [batchMode, setBatchMode] = useState(false)
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [processingOptions, setProcessingOptions] = useState({
    extractText: true,
    extractObjects: true,
    extractSentiment: false,
    enableCache: true,
    maxTokens: 1000,
    temperature: 0.1,
  })
  const [modelConfigs, setModelConfigs] = useState<MultimodalModelConfig[]>([])
  const [pricingLoading, setPricingLoading] = useState(false)

  const statusCheckInterval = useRef<ReturnType<typeof setTimeout>>()

  const getTokenUsage = (tokensUsed?: number | TokenUsage) => {
    if (typeof tokensUsed === 'number') {
      const inputTokens = Math.floor(tokensUsed / 3)
      const outputTokens = Math.max(0, tokensUsed - inputTokens)
      return { inputTokens, outputTokens, totalTokens: tokensUsed }
    }
    if (!tokensUsed) {
      return { inputTokens: 0, outputTokens: 0, totalTokens: 0 }
    }
    const totalTokens =
      typeof tokensUsed.total_tokens === 'number'
        ? tokensUsed.total_tokens
        : (typeof tokensUsed.prompt_tokens === 'number'
            ? tokensUsed.prompt_tokens
            : 0) +
          (typeof tokensUsed.completion_tokens === 'number'
            ? tokensUsed.completion_tokens
            : 0)
    const inputTokens =
      typeof tokensUsed.prompt_tokens === 'number'
        ? tokensUsed.prompt_tokens
        : typeof tokensUsed.completion_tokens === 'number' && totalTokens > 0
          ? 0
          : Math.floor(totalTokens / 3)
    const outputTokens =
      typeof tokensUsed.completion_tokens === 'number'
        ? tokensUsed.completion_tokens
        : Math.max(0, totalTokens - inputTokens)
    return { inputTokens, outputTokens, totalTokens }
  }

  // 加载队列状态
  const loadQueueStatus = async () => {
    try {
      const status = await multimodalService.getQueueStatus()
      setQueueStatus(status)
    } catch (error) {
      logger.error('获取队列状态失败:', error)
    }
  }

  // 定期检查处理状态
  const checkProcessingStatus = async () => {
    const tasks = Array.from(processingTasks.values())
    const processingTaskIds = tasks
      .filter(
        task =>
          task.status === ProcessingStatus.PROCESSING ||
          task.status === ProcessingStatus.PENDING
      )
      .map(task => task.contentId)

    for (const contentId of processingTaskIds) {
      try {
        const status = await multimodalService.getProcessingStatus(contentId)
        setProcessingTasks(prev => {
          const newMap = new Map(prev)
          const task = newMap.get(contentId)
          if (task) {
            task.status = status.status as ProcessingStatus
            if (status.status === 'completed') {
              task.progress = 100
              task.processingTime = status.processing_time
              task.model = status.model_used || task.model
              task.cost = calculateCostForTask(status)
            } else if (status.status === 'failed') {
              task.progress = 0
            } else if (status.status === 'processing') {
              task.progress = Math.min(90, (task.progress || 0) + 10)
            }
          }
          return newMap
        })
      } catch (error) {
        logger.error(`检查状态失败 ${contentId}:`, error)
      }
    }
  }

  useEffect(() => {
    loadQueueStatus()

    // 设置定期状态检查
    statusCheckInterval.current = setInterval(() => {
      checkProcessingStatus()
      loadQueueStatus()
    }, 3000)

    return () => {
      if (statusCheckInterval.current) {
        clearInterval(statusCheckInterval.current)
      }
    }
  }, [processingTasks])

  useEffect(() => {
    const loadModelConfigs = async () => {
      setPricingLoading(true)
      try {
        await multimodalService.ensureModelPricing()
        const configs = await multimodalService.getModelConfigs()
        setModelConfigs(configs)
        if (
          configs.length > 0 &&
          !configs.find(config => config.name === selectedModel)
        ) {
          setSelectedModel(configs[0].name)
        }
      } catch (error) {
        logger.error('获取模型配置失败:', error)
        setModelConfigs([])
      } finally {
        setPricingLoading(false)
      }
    }
    loadModelConfigs()
  }, [])

  // 计算任务成本
  const calculateCostForTask = (status: ProcessingStatusResponse): number => {
    if (!status.tokens_used || !status.model_used) return 0
    const { inputTokens, outputTokens } = getTokenUsage(status.tokens_used)
    return multimodalService.calculateCost(
      status.model_used,
      inputTokens,
      outputTokens
    )
  }

  // 处理文件上传
  const handleUpload = async (file: File): Promise<boolean> => {
    try {
      // 验证文件
      const validation = multimodalService.validateFile(file)
      if (!validation.valid) {
        message.error(validation.error || '文件验证失败')
        return false
      }

      setLoading(true)

      // 上传文件
      const uploadResult = await multimodalService.uploadFile(file)

      // 创建处理任务
      const task: ProcessingTask = {
        contentId: uploadResult.content_id,
        fileName: file.name,
        fileType: file.type,
        status: ProcessingStatus.PENDING,
        progress: 0,
        model: selectedModel,
      }

      setProcessingTasks(prev =>
        new Map(prev).set(uploadResult.content_id, task)
      )

      // 处理内容
      const processRequest = {
        content_id: uploadResult.content_id,
        content_type: getContentTypeFromMime(file.type),
        priority: processingPriority,
        complexity: processingComplexity,
        max_tokens: processingOptions.maxTokens,
        temperature: processingOptions.temperature,
        enable_cache: processingOptions.enableCache,
        extract_text: processingOptions.extractText,
        extract_objects: processingOptions.extractObjects,
        extract_sentiment: processingOptions.extractSentiment,
      }

      const processResult =
        await multimodalService.processContent(processRequest)

      // 更新任务状态
      setProcessingTasks(prev => {
        const newMap = new Map(prev)
        const updatedTask = newMap.get(uploadResult.content_id)
        if (updatedTask) {
          updatedTask.status = processResult.status as ProcessingStatus
          updatedTask.result = processResult
          updatedTask.processingTime = processResult.processing_time
          updatedTask.model = processResult.model_used
          updatedTask.cost = multimodalService.calculateCost(
            processResult.model_used,
            getTokenUsage(processResult.tokens_used).inputTokens,
            getTokenUsage(processResult.tokens_used).outputTokens
          )
        }
        return newMap
      })

      message.success('文件上传并开始处理')
      return false
    } catch (error: any) {
      message.error(error.response?.data?.detail || '文件处理失败')
      return false
    } finally {
      setLoading(false)
    }
  }

  // 获取内容类型
  const getContentTypeFromMime = (mimeType: string): string => {
    if (mimeType.startsWith('image/')) return ContentType.IMAGE
    if (mimeType.startsWith('video/')) return ContentType.VIDEO
    if (mimeType.startsWith('audio/')) return ContentType.AUDIO
    if (
      mimeType.includes('pdf') ||
      mimeType.includes('document') ||
      mimeType.includes('text')
    ) {
      return ContentType.DOCUMENT
    }
    return ContentType.DOCUMENT
  }

  // 快速分析图像
  const handleQuickAnalysis = async () => {
    if (!quickAnalysisFile) {
      message.warning('请先选择要分析的图像文件')
      return
    }

    try {
      setLoading(true)
      const result = await multimodalService.analyzeImage(
        quickAnalysisFile,
        quickAnalysisPrompt || '分析这张图像',
        processingOptions.extractText,
        processingOptions.extractObjects,
        processingPriority
      )

      // 显示结果
      Modal.info({
        title: '图像分析结果',
        width: 800,
        content: (
          <div>
            <Descriptions column={1} bordered size="small">
              <Descriptions.Item label="描述">
                {result.extracted_data.description}
              </Descriptions.Item>
              {result.extracted_data.objects && (
                <Descriptions.Item label="识别对象">
                  <Space wrap>
                    {result.extracted_data.objects.map((obj, idx) => (
                      <Tag key={idx} color="blue">
                        {obj}
                      </Tag>
                    ))}
                  </Space>
                </Descriptions.Item>
              )}
              {result.extracted_data.text && (
                <Descriptions.Item label="提取文本">
                  {result.extracted_data.text}
                </Descriptions.Item>
              )}
              <Descriptions.Item label="模型">
                {result.model_used}
              </Descriptions.Item>
              <Descriptions.Item label="Token使用">
                {getTokenUsage(result.tokens_used).totalTokens}
              </Descriptions.Item>
              <Descriptions.Item label="成本">
                ${result.cost.toFixed(4)}
              </Descriptions.Item>
              <Descriptions.Item label="处理时间">
                {result.processing_time}秒
              </Descriptions.Item>
            </Descriptions>
          </div>
        ),
      })

      // 更新成本统计
      setCostStats(prev => ({
        ...prev,
        daily: prev.daily + result.cost,
        totalTokens:
          prev.totalTokens + getTokenUsage(result.tokens_used).totalTokens,
      }))
    } catch (error: any) {
      message.error(error.response?.data?.detail || '快速分析失败')
    } finally {
      setLoading(false)
    }
  }

  // 批量处理
  const handleBatchProcess = async () => {
    if (selectedFiles.length === 0) {
      message.warning('请选择要批量处理的文件')
      return
    }

    try {
      setLoading(true)
      const batchRequest = {
        content_ids: selectedFiles,
        priority: processingPriority,
        complexity: processingComplexity,
        max_tokens: processingOptions.maxTokens,
      }

      const result = await multimodalService.processBatch(batchRequest)
      message.success(`批量处理已提交，批次ID: ${result.batch_id}`)

      // 清空选择
      setSelectedFiles([])
      setBatchMode(false)
    } catch (error: any) {
      message.error(error.response?.data?.detail || '批量处理失败')
    } finally {
      setLoading(false)
    }
  }

  // 删除文件
  const handleDeleteFile = async (contentId: string) => {
    try {
      await multimodalService.deleteFile(contentId)
      setProcessingTasks(prev => {
        const newMap = new Map(prev)
        newMap.delete(contentId)
        return newMap
      })
      message.success('文件已删除')
    } catch (error) {
      message.error('删除文件失败')
    }
  }

  // 刷新状态
  const handleRefresh = async () => {
    setRefreshing(true)
    await Promise.all([loadQueueStatus(), checkProcessingStatus()])
    setRefreshing(false)
    message.success('状态已刷新')
  }

  // 获取状态图标
  const getStatusIcon = (status: string) => {
    switch (status) {
      case ProcessingStatus.COMPLETED:
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case ProcessingStatus.FAILED:
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
      case ProcessingStatus.PROCESSING:
        return <ClockCircleOutlined style={{ color: '#1890ff' }} />
      default:
        return <ClockCircleOutlined style={{ color: '#8c8c8c' }} />
    }
  }

  // 获取状态标签
  const getStatusLabel = (status: string) => {
    switch (status) {
      case ProcessingStatus.COMPLETED:
        return '已完成'
      case ProcessingStatus.PROCESSING:
        return '处理中'
      case ProcessingStatus.FAILED:
        return '失败'
      case ProcessingStatus.PENDING:
        return '等待中'
      default:
        return status
    }
  }

  // 表格列定义
  const queueColumns = [
    {
      title: batchMode && (
        <Checkbox
          onChange={e => {
            if (e.target.checked) {
              setSelectedFiles(Array.from(processingTasks.keys()))
            } else {
              setSelectedFiles([])
            }
          }}
        />
      ),
      dataIndex: 'select',
      key: 'select',
      width: 50,
      render: (_: any, record: ProcessingTask) =>
        batchMode && (
          <Checkbox
            checked={selectedFiles.includes(record.contentId)}
            onChange={e => {
              if (e.target.checked) {
                setSelectedFiles(prev => [...prev, record.contentId])
              } else {
                setSelectedFiles(prev =>
                  prev.filter(id => id !== record.contentId)
                )
              }
            }}
          />
        ),
    },
    {
      title: '文件名',
      dataIndex: 'fileName',
      key: 'fileName',
      render: (text: string, record: ProcessingTask) => (
        <Space>
          {record.fileType.includes('image') && <FileImageOutlined />}
          {record.fileType.includes('pdf') && <FileTextOutlined />}
          {record.fileType.includes('video') && <VideoCameraOutlined />}
          {record.fileType.includes('audio') && <AudioOutlined />}
          {text}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Space>
          {getStatusIcon(status)}
          <span>{getStatusLabel(status)}</span>
        </Space>
      ),
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number) => (
        <Progress percent={progress} size="small" />
      ),
    },
    {
      title: '模型',
      dataIndex: 'model',
      key: 'model',
      render: (model: string) => <Tag color="blue">{model}</Tag>,
    },
    {
      title: '成本',
      dataIndex: 'cost',
      key: 'cost',
      render: (cost: number) => (cost ? `$${cost.toFixed(4)}` : '-'),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: ProcessingTask) => (
        <Space>
          {record.status === ProcessingStatus.COMPLETED && record.result && (
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                Modal.info({
                  title: '处理结果',
                  width: 800,
                  content: (
                    <Descriptions column={1} bordered size="small">
                      <Descriptions.Item label="提取数据">
                        <pre>
                          {JSON.stringify(
                            record.result.extracted_data,
                            null,
                            2
                          )}
                        </pre>
                      </Descriptions.Item>
                      <Descriptions.Item label="置信度">
                        {(record.result.confidence_score * 100).toFixed(1)}%
                      </Descriptions.Item>
                      <Descriptions.Item label="处理时间">
                        {record.result.processing_time}秒
                      </Descriptions.Item>
                      <Descriptions.Item label="Token使用">
                        {getTokenUsage(record.result.tokens_used).totalTokens}
                      </Descriptions.Item>
                    </Descriptions>
                  ),
                })
              }}
            >
              查看
            </Button>
          )}
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteFile(record.contentId)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ]

  // 计算统计数据
  const calculateStats = () => {
    const tasks = Array.from(processingTasks.values())
    return {
      total: tasks.length,
      completed: tasks.filter(t => t.status === ProcessingStatus.COMPLETED)
        .length,
      failed: tasks.filter(t => t.status === ProcessingStatus.FAILED).length,
      processing: tasks.filter(t => t.status === ProcessingStatus.PROCESSING)
        .length,
      pending: tasks.filter(t => t.status === ProcessingStatus.PENDING).length,
    }
  }

  const stats = calculateStats()

  return (
    <div className="p-6">
      <Title level={2}>GPT-4o 多模态API集成</Title>
      <Paragraph>
        支持图像、文档、视频等多种内容的智能分析处理，集成OpenAI最新的多模态模型。
      </Paragraph>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        {/* 文件上传标签页 */}
        <TabPane tab="文件上传" key="upload">
          <Row gutter={[24, 24]}>
            <Col span={12}>
              <Card title="处理设置" className="mb-4">
                <Form layout="vertical">
                  <Form.Item label="选择模型">
                    <Select
                      id="selectedModel"
                      name="selectedModel"
                      value={selectedModel}
                      onChange={setSelectedModel}
                      style={{ width: '100%' }}
                    >
                      {modelConfigs.map(config => (
                        <Option key={config.name} value={config.name}>
                          {config.name}
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>

                  <Form.Item label="处理优先级">
                    <Radio.Group
                      id="processingPriority"
                      name="processingPriority"
                      value={processingPriority}
                      onChange={e => setProcessingPriority(e.target.value)}
                    >
                      <Radio value={ModelPriority.SPEED}>速度优先</Radio>
                      <Radio value={ModelPriority.BALANCED}>平衡</Radio>
                      <Radio value={ModelPriority.QUALITY}>质量优先</Radio>
                    </Radio.Group>
                  </Form.Item>

                  <Form.Item label="复杂度">
                    <Radio.Group
                      id="processingComplexity"
                      name="processingComplexity"
                      value={processingComplexity}
                      onChange={e => setProcessingComplexity(e.target.value)}
                    >
                      <Radio value={ModelComplexity.LOW}>低</Radio>
                      <Radio value={ModelComplexity.MEDIUM}>中</Radio>
                      <Radio value={ModelComplexity.HIGH}>高</Radio>
                    </Radio.Group>
                  </Form.Item>

                  <Form.Item label="处理选项">
                    <Space direction="vertical">
                      <Checkbox
                        name="extractText"
                        checked={processingOptions.extractText}
                        onChange={e =>
                          setProcessingOptions(prev => ({
                            ...prev,
                            extractText: e.target.checked,
                          }))
                        }
                      >
                        提取文本
                      </Checkbox>
                      <Checkbox
                        name="extractObjects"
                        checked={processingOptions.extractObjects}
                        onChange={e =>
                          setProcessingOptions(prev => ({
                            ...prev,
                            extractObjects: e.target.checked,
                          }))
                        }
                      >
                        识别对象
                      </Checkbox>
                      <Checkbox
                        name="extractSentiment"
                        checked={processingOptions.extractSentiment}
                        onChange={e =>
                          setProcessingOptions(prev => ({
                            ...prev,
                            extractSentiment: e.target.checked,
                          }))
                        }
                      >
                        情感分析
                      </Checkbox>
                      <Checkbox
                        name="enableCache"
                        checked={processingOptions.enableCache}
                        onChange={e =>
                          setProcessingOptions(prev => ({
                            ...prev,
                            enableCache: e.target.checked,
                          }))
                        }
                      >
                        启用缓存
                      </Checkbox>
                    </Space>
                  </Form.Item>
                </Form>
              </Card>

              <Card title="文件上传">
                <Upload.Dragger
                  name="file"
                  multiple
                  beforeUpload={handleUpload}
                  showUploadList={false}
                  disabled={loading}
                >
                  <p className="ant-upload-drag-icon">
                    <CloudUploadOutlined />
                  </p>
                  <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
                  <p className="ant-upload-hint">
                    支持图像、文档、视频、音频文件，最大20MB
                  </p>
                </Upload.Dragger>

                <div className="mt-4">
                  <Alert
                    message="支持的格式"
                    description={multimodalService
                      .getSupportedFileTypes()
                      .join(', ')}
                    type="info"
                    showIcon
                  />
                </div>
              </Card>
            </Col>

            <Col span={12}>
              <Card title="快速图像分析" className="mb-4">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Upload
                    name="quickAnalysisFile"
                    beforeUpload={file => {
                      setQuickAnalysisFile(file)
                      return false
                    }}
                    showUploadList={false}
                  >
                    <Button icon={<UploadOutlined />}>选择图像文件</Button>
                  </Upload>
                  {quickAnalysisFile && (
                    <Text type="secondary">
                      已选择: {quickAnalysisFile.name}
                    </Text>
                  )}
                  <TextArea
                    name="quickAnalysisPrompt"
                    placeholder="输入分析提示（可选）..."
                    rows={4}
                    value={quickAnalysisPrompt}
                    onChange={e => setQuickAnalysisPrompt(e.target.value)}
                  />
                  <Button
                    type="primary"
                    block
                    loading={loading}
                    onClick={handleQuickAnalysis}
                    disabled={!quickAnalysisFile}
                  >
                    快速分析
                  </Button>
                </Space>
              </Card>

              <Card title="队列状态">
                {queueStatus ? (
                  <div>
                    <Row gutter={16}>
                      <Col span={12}>
                        <Statistic
                          title="总任务"
                          value={queueStatus.total_jobs}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="处理中"
                          value={queueStatus.processing_jobs}
                        />
                      </Col>
                    </Row>
                    <Row gutter={16} className="mt-4">
                      <Col span={12}>
                        <Statistic
                          title="已完成"
                          value={queueStatus.completed_jobs}
                          valueStyle={{ color: '#3f8600' }}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="失败"
                          value={queueStatus.failed_jobs}
                          valueStyle={{ color: '#cf1322' }}
                        />
                      </Col>
                    </Row>
                    <Divider />
                    <Text type="secondary">
                      平均等待时间: {queueStatus.average_wait_time.toFixed(1)}秒
                      <br />
                      平均处理时间:{' '}
                      {queueStatus.average_processing_time.toFixed(1)}秒
                    </Text>
                  </div>
                ) : (
                  <Spin />
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>

        {/* 处理队列标签页 */}
        <TabPane tab="处理队列" key="queue">
          <Card
            title="处理任务列表"
            extra={
              <Space>
                <Button
                  type={batchMode ? 'primary' : 'default'}
                  onClick={() => {
                    setBatchMode(!batchMode)
                    setSelectedFiles([])
                  }}
                >
                  {batchMode ? '退出批量' : '批量模式'}
                </Button>
                {batchMode && selectedFiles.length > 0 && (
                  <Button type="primary" onClick={handleBatchProcess}>
                    批量处理 ({selectedFiles.length})
                  </Button>
                )}
                <Button
                  icon={<ReloadOutlined />}
                  onClick={handleRefresh}
                  loading={refreshing}
                >
                  刷新
                </Button>
              </Space>
            }
          >
            <Row gutter={16} className="mb-4">
              <Col span={4}>
                <Statistic title="总计" value={stats.total} />
              </Col>
              <Col span={5}>
                <Statistic
                  title="已完成"
                  value={stats.completed}
                  valueStyle={{ color: '#3f8600' }}
                />
              </Col>
              <Col span={5}>
                <Statistic
                  title="处理中"
                  value={stats.processing}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={5}>
                <Statistic
                  title="等待中"
                  value={stats.pending}
                  valueStyle={{ color: '#faad14' }}
                />
              </Col>
              <Col span={5}>
                <Statistic
                  title="失败"
                  value={stats.failed}
                  valueStyle={{ color: '#cf1322' }}
                />
              </Col>
            </Row>

            <Table
              columns={queueColumns}
              dataSource={Array.from(processingTasks.values())}
              rowKey="contentId"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showTotal: total => `共 ${total} 条`,
              }}
            />
          </Card>
        </TabPane>

        {/* 成本监控标签页 */}
        <TabPane tab="成本监控" key="cost">
          <Row gutter={[24, 24]}>
            <Col span={12}>
              <Card title="成本统计">
                <Row gutter={16}>
                  <Col span={12}>
                    <Statistic
                      title="今日总成本"
                      value={costStats.daily}
                      precision={4}
                      prefix="$"
                      valueStyle={{ color: '#cf1322' }}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="总Token使用"
                      value={costStats.totalTokens}
                      suffix="tokens"
                    />
                  </Col>
                </Row>

                <Divider />

                <div>
                  <Text strong>成本预警</Text>
                  <Progress
                    percent={Math.min(100, (costStats.daily / 0.1) * 100)}
                    status={costStats.daily > 0.08 ? 'exception' : 'normal'}
                    format={() => `$${costStats.daily.toFixed(3)} / $0.10`}
                  />
                  <Text type="secondary" className="text-xs">
                    每日预算限制
                  </Text>
                </div>
              </Card>
            </Col>

            <Col span={12}>
              <Card title="模型使用分布">
                <Space direction="vertical" style={{ width: '100%' }}>
                  {(modelConfigs.length
                    ? modelConfigs.map(config => config.name)
                    : Array.from(costStats.modelDistribution.keys())
                  ).map(model => {
                    const usage = costStats.modelDistribution.get(model) || 0
                    const total = Array.from(
                      costStats.modelDistribution.values()
                    ).reduce((a, b) => a + b, 1)
                    const percent = (usage / total) * 100

                    return (
                      <div key={model}>
                        <div className="flex justify-between">
                          <Text>{model}</Text>
                          <Text>{percent.toFixed(1)}%</Text>
                        </div>
                        <Progress
                          percent={percent}
                          size="small"
                          showInfo={false}
                        />
                      </div>
                    )
                  })}
                </Space>
              </Card>
            </Col>
          </Row>

          <Card title="成本计算说明" className="mt-6">
            <Table
              loading={pricingLoading}
              dataSource={modelConfigs.map(config => ({
                model: config.name,
                input: config.cost_per_1k_tokens?.input ?? 0,
                output: config.cost_per_1k_tokens?.output ?? 0,
              }))}
              columns={[
                { title: '模型', dataIndex: 'model', key: 'model' },
                {
                  title: '输入价格 ($/1K tokens)',
                  dataIndex: 'input',
                  key: 'input',
                },
                {
                  title: '输出价格 ($/1K tokens)',
                  dataIndex: 'output',
                  key: 'output',
                },
              ]}
              rowKey="model"
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default MultimodalPageComplete
