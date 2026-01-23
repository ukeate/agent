import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Upload,
  Modal,
  Form,
  Input,
  Select,
  message,
  Divider,
  Statistic,
  Row,
  Col,
  Tooltip,
  Progress,
  Tabs,
  Descriptions,
  Badge,
  Timeline,
  Alert,
  Spin,
  Switch,
  InputNumber,
} from 'antd'
import {
  PlusOutlined,
  UploadOutlined,
  DeleteOutlined,
  EyeOutlined,
  DownloadOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloudUploadOutlined,
  DatabaseOutlined,
  RocketOutlined,
  ExperimentOutlined,
  FolderOpenOutlined,
  ApiOutlined,
  ClockCircleOutlined,
  BarChartOutlined,
  GlobalOutlined,
  CompressOutlined,
  ThunderboltOutlined,
  FileTextOutlined,
  BranchesOutlined,
  DeploymentUnitOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import type { UploadProps } from 'antd'
import { Line, Column, Pie } from '@ant-design/charts'
import {
  modelRegistryService,
  type ModelEntry,
  type ModelMetadata,
  type ModelVersion,
  type ModelDeployment,
  type ModelEvaluation,
  type ModelStatistics,
  type ModelFormat,
  type ModelType,
  type ModelStatus,
} from '../services/modelRegistryService'

const { Option } = Select
const { TextArea } = Input
const { TabPane } = Tabs

const ModelRegistryPage: React.FC = () => {
  // ==================== 状态管理 ====================
  const [models, setModels] = useState<ModelEntry[]>([])
  const [loading, setLoading] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [activeTab, setActiveTab] = useState('models')

  // 模态框状态
  const [uploadModalVisible, setUploadModalVisible] = useState(false)
  const [hubModalVisible, setHubModalVisible] = useState(false)
  const [detailModalVisible, setDetailModalVisible] = useState(false)
  const [deployModalVisible, setDeployModalVisible] = useState(false)
  const [evaluateModalVisible, setEvaluateModalVisible] = useState(false)

  // 选中的模型
  const [selectedModel, setSelectedModel] = useState<ModelEntry | null>(null)
  const [selectedModelVersions, setSelectedModelVersions] = useState<
    ModelVersion[]
  >([])
  const [selectedModelDeployments, setSelectedModelDeployments] = useState<
    ModelDeployment[]
  >([])
  const [selectedModelEvaluations, setSelectedModelEvaluations] = useState<
    ModelEvaluation[]
  >([])

  // 统计信息
  const [statistics, setStatistics] = useState<ModelStatistics | null>(null)

  // 过滤器
  const [filters, setFilters] = useState<{
    format?: ModelFormat
    model_type?: ModelType
    status?: ModelStatus
    search?: string
  }>({})

  const [form] = Form.useForm()
  const [hubForm] = Form.useForm()
  const [deployForm] = Form.useForm()
  const [evaluateForm] = Form.useForm()

  // ==================== 数据加载 ====================

  useEffect(() => {
    loadModels()
    loadStatistics()
  }, [filters])

  const loadModels = async () => {
    setLoading(true)
    try {
      const result = await modelRegistryService.listModels(filters)
      setModels(result.models)
    } catch (error) {
      logger.error('加载模型列表失败:', error)
      message.error('加载模型列表失败')
    }
    setLoading(false)
  }

  const loadStatistics = async () => {
    try {
      const stats = await modelRegistryService.getStatistics()
      setStatistics(stats)
    } catch (error) {
      logger.error('加载统计信息失败:', error)
    }
  }

  const loadModelDetails = async (model: ModelEntry) => {
    setSelectedModel(model)

    // 加载版本、部署和评估信息
    try {
      const [versions, deployments, evaluations] = await Promise.all([
        modelRegistryService.listVersions(model.metadata.name),
        modelRegistryService.listDeployments(model.model_id),
        modelRegistryService.listEvaluations(model.model_id),
      ])

      setSelectedModelVersions(versions)
      setSelectedModelDeployments(deployments)
      setSelectedModelEvaluations(evaluations)
    } catch (error) {
      logger.error('加载模型详情失败:', error)
    }
  }

  const handleRefresh = async () => {
    setRefreshing(true)
    await Promise.all([loadModels(), loadStatistics()])
    setRefreshing(false)
    message.success('数据已刷新')
  }

  // ==================== 操作处理 ====================

  const handleUpload = async (values: any) => {
    try {
      const metadata: ModelMetadata = {
        name: values.name,
        version: values.version || '1.0.0',
        format: values.format,
        model_type: values.model_type,
        description: values.description,
        author: values.author,
        tags: values.tags || [],
        training_framework: values.training_framework,
        parameters_count: values.parameters_count,
        model_size_mb: values.model_size_mb,
      }

      await modelRegistryService.registerModel(metadata, values.file?.file)
      message.success('模型上传成功')
      setUploadModalVisible(false)
      form.resetFields()
      loadModels()
    } catch (error) {
      message.error('模型上传失败')
    }
  }

  const handleHubImport = async (values: any) => {
    try {
      message.loading('正在从HuggingFace导入模型...', 0)
      await modelRegistryService.importFromHuggingFace(values.model_name, {
        revision: values.revision,
        use_auth_token: values.use_auth_token,
      })
      message.destroy()
      message.success('模型导入成功')
      setHubModalVisible(false)
      hubForm.resetFields()
      loadModels()
    } catch (error) {
      message.destroy()
      message.error('模型导入失败')
    }
  }

  const handleDelete = (modelId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个模型吗？此操作不可恢复。',
      onOk: async () => {
        try {
          await modelRegistryService.deleteModel(modelId)
          message.success('模型删除成功')
          loadModels()
        } catch (error) {
          message.error('删除失败')
        }
      },
    })
  }

  const handleDeploy = async (values: any) => {
    if (!selectedModel) return

    try {
      await modelRegistryService.deployModel(
        selectedModel.model_id,
        values.environment,
        {
          replicas: values.replicas,
          cpu_request: values.cpu_request,
          memory_request: values.memory_request,
          gpu_request: values.gpu_request,
          auto_scaling: values.auto_scaling,
          min_replicas: values.min_replicas,
          max_replicas: values.max_replicas,
        }
      )
      message.success('模型部署成功')
      setDeployModalVisible(false)
      deployForm.resetFields()
      loadModelDetails(selectedModel)
    } catch (error) {
      message.error('模型部署失败')
    }
  }

  const handleEvaluate = async (values: any) => {
    if (!selectedModel) return

    try {
      message.loading('正在评估模型...', 0)
      await modelRegistryService.evaluateModel(
        selectedModel.model_id,
        values.dataset_name,
        {
          batch_size: values.batch_size,
          device: values.device,
          metrics: values.metrics,
        }
      )
      message.destroy()
      message.success('模型评估完成')
      setEvaluateModalVisible(false)
      evaluateForm.resetFields()
      loadModelDetails(selectedModel)
    } catch (error) {
      message.destroy()
      message.error('模型评估失败')
    }
  }

  const handleDownload = async (modelId: string) => {
    try {
      const url = await modelRegistryService.downloadModel(modelId)
      window.open(url, '_blank')
    } catch (error) {
      message.error('获取下载链接失败')
    }
  }

  const handleConvert = async (modelId: string, targetFormat: ModelFormat) => {
    try {
      message.loading('正在转换模型格式...', 0)
      const result = await modelRegistryService.convertModel(
        modelId,
        targetFormat,
        {
          optimize: true,
          quantize: false,
        }
      )
      message.destroy()
      if (result.success) {
        message.success(
          `模型转换成功，耗时 ${result.conversion_time_seconds}秒`
        )
        loadModels()
      }
    } catch (error) {
      message.destroy()
      message.error('模型转换失败')
    }
  }

  // ==================== 工具函数 ====================

  const formatSize = (sizeInMB?: number) => {
    if (!sizeInMB) return '-'
    if (sizeInMB < 1024) {
      return `${sizeInMB.toFixed(1)} MB`
    }
    return `${(sizeInMB / 1024).toFixed(1)} GB`
  }

  const formatNumber = (num?: number) => {
    if (!num) return '-'
    if (num >= 1e9) {
      return `${(num / 1e9).toFixed(1)}B`
    }
    if (num >= 1e6) {
      return `${(num / 1e6).toFixed(1)}M`
    }
    if (num >= 1e3) {
      return `${(num / 1e3).toFixed(1)}K`
    }
    return num.toString()
  }

  const getStatusColor = (status: ModelStatus) => {
    const colorMap: Record<ModelStatus, string> = {
      uploaded: 'default',
      validated: 'processing',
      deployed: 'success',
      deprecated: 'warning',
      archived: 'error',
    }
    return colorMap[status]
  }

  // ==================== 图表配置 ====================

  const formatDistributionChart = () => {
    if (!statistics) return { data: [] }

    const data = Object.entries(statistics.by_format).map(([key, value]) => ({
      type: key,
      value,
    }))

    return {
      data,
      angleField: 'value',
      colorField: 'type',
      radius: 0.8,
      label: {
        type: 'inner',
        offset: '-30%',
        content: '{percentage}',
        style: {
          textAlign: 'center',
          fontSize: 14,
        },
      },
      interactions: [{ type: 'element-active' }],
    }
  }

  // ==================== 表格配置 ====================

  const modelColumns: ColumnsType<ModelEntry> = [
    {
      title: '模型名称',
      key: 'name',
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Button
            type="link"
            onClick={() => {
              loadModelDetails(record)
              setDetailModalVisible(true)
            }}
          >
            <strong>{record.metadata.name}</strong>
          </Button>
          <span style={{ color: '#666', fontSize: '12px' }}>
            v{record.metadata.version} | ID: {record.model_id}
          </span>
        </Space>
      ),
    },
    {
      title: '类型/格式',
      key: 'type',
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Tag color="blue">{record.metadata.model_type}</Tag>
          <Tag>{record.metadata.format}</Tag>
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: ModelStatus) => (
        <Badge
          status={status === 'deployed' ? 'success' : 'default'}
          text={
            <Tag color={getStatusColor(status)}>{status.toUpperCase()}</Tag>
          }
        />
      ),
    },
    {
      title: '规模',
      key: 'size',
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <span>{formatNumber(record.metadata.parameters_count)} 参数</span>
          <span style={{ fontSize: '12px', color: '#666' }}>
            {formatSize(record.metadata.model_size_mb)}
          </span>
        </Space>
      ),
    },
    {
      title: '部署/下载',
      key: 'usage',
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <span>
            <DeploymentUnitOutlined /> {record.deployment_count} 部署
          </span>
          <span style={{ fontSize: '12px', color: '#666' }}>
            <DownloadOutlined /> {record.download_count} 下载
          </span>
        </Space>
      ),
    },
    {
      title: '评分',
      key: 'scores',
      render: (_, record) => {
        if (!record.evaluation_scores) return '-'
        const scores = Object.entries(record.evaluation_scores)
        return (
          <Tooltip
            title={
              <div>
                {scores.map(([metric, value]) => (
                  <div key={metric}>
                    {metric}: {value.toFixed(3)}
                  </div>
                ))}
              </div>
            }
          >
            <span>
              {scores[0] && `${scores[0][0]}: ${scores[0][1].toFixed(3)}`}
              {scores.length > 1 && ` +${scores.length - 1}`}
            </span>
          </Tooltip>
        )
      },
    },
    {
      title: '标签',
      key: 'tags',
      render: (_, record) => (
        <Space wrap>
          {record.metadata.tags.slice(0, 3).map(tag => (
            <Tag key={tag} size="small">
              {tag}
            </Tag>
          ))}
          {record.metadata.tags.length > 3 && (
            <Tag size="small">+{record.metadata.tags.length - 3}</Tag>
          )}
        </Space>
      ),
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      render: (date: string) => new Date(date).toLocaleDateString(),
    },
    {
      title: '操作',
      key: 'actions',
      fixed: 'right',
      render: (_, record) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                loadModelDetails(record)
                setDetailModalVisible(true)
              }}
            />
          </Tooltip>
          <Tooltip title="部署">
            <Button
              size="small"
              icon={<RocketOutlined />}
              onClick={() => {
                setSelectedModel(record)
                setDeployModalVisible(true)
              }}
            />
          </Tooltip>
          <Tooltip title="下载">
            <Button
              size="small"
              icon={<DownloadOutlined />}
              onClick={() => handleDownload(record.model_id)}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => handleDelete(record.model_id)}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const versionColumns: ColumnsType<ModelVersion> = [
    {
      title: '版本',
      dataIndex: 'version',
      key: 'version',
      render: (version, record) => (
        <Space>
          {version}
          {record.is_latest && <Tag color="blue">最新</Tag>}
          {record.is_stable && <Tag color="green">稳定</Tag>}
        </Space>
      ),
    },
    {
      title: '大小',
      dataIndex: 'file_size_mb',
      key: 'file_size_mb',
      render: size => formatSize(size),
    },
    {
      title: '更新说明',
      dataIndex: 'changes',
      key: 'changes',
      ellipsis: true,
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: date => new Date(date).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button size="small" type="primary">
            激活
          </Button>
          <Button size="small">下载</Button>
        </Space>
      ),
    },
  ]

  const deploymentColumns: ColumnsType<ModelDeployment> = [
    {
      title: '环境',
      dataIndex: 'environment',
      key: 'environment',
      render: env => (
        <Tag
          color={
            env === 'production' ? 'red' : env === 'staging' ? 'orange' : 'blue'
          }
        >
          {env.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: '端点',
      dataIndex: 'endpoint_url',
      key: 'endpoint_url',
      ellipsis: true,
      render: url => (
        <Tooltip title={url}>
          <a href={url} target="_blank" rel="noopener noreferrer">
            {url}
          </a>
        </Tooltip>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: status => (
        <Badge
          status={
            status === 'active'
              ? 'success'
              : status === 'deploying'
                ? 'processing'
                : status === 'failed'
                  ? 'error'
                  : 'default'
          }
          text={status.toUpperCase()}
        />
      ),
    },
    {
      title: '性能',
      key: 'performance',
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <span>请求: {record.requests_count || 0}</span>
          <span style={{ fontSize: '12px', color: '#666' }}>
            延迟: {record.average_latency_ms || '-'}ms
          </span>
        </Space>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Button
          size="small"
          danger
          disabled={record.status !== 'active'}
          onClick={async () => {
            try {
              await modelRegistryService.stopDeployment(record.deployment_id)
              message.success('部署已停止')
              if (selectedModel) {
                loadModelDetails(selectedModel)
              }
            } catch (error) {
              message.error('停止部署失败')
            }
          }}
        >
          停止
        </Button>
      ),
    },
  ]

  // ==================== 渲染 ====================

  if (loading && models.length === 0) {
    return (
      <div style={{ padding: '24px', textAlign: 'center' }}>
        <Spin size="large" tip="加载模型注册表..." />
      </div>
    )
  }

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <Row align="middle" justify="space-between" style={{ marginBottom: 24 }}>
        <Col>
          <Space>
            <DatabaseOutlined style={{ fontSize: 24, color: '#1890ff' }} />
            <div>
              <h2 style={{ margin: 0 }}>模型注册表</h2>
              <span style={{ color: '#666' }}>管理、部署和评估AI模型</span>
            </div>
          </Space>
        </Col>
        <Col>
          <Space>
            <Button onClick={handleRefresh} loading={refreshing}>
              刷新
            </Button>
            <Button
              type="primary"
              icon={<CloudUploadOutlined />}
              onClick={() => setHubModalVisible(true)}
            >
              从Hub导入
            </Button>
            <Button
              type="primary"
              icon={<UploadOutlined />}
              onClick={() => setUploadModalVisible(true)}
            >
              上传模型
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 统计卡片 */}
      {statistics && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="模型总数"
                value={statistics.total_models}
                prefix={<FolderOpenOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="总存储"
                value={statistics.total_size_gb}
                suffix="GB"
                prefix={<DatabaseOutlined />}
                precision={1}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="活跃部署"
                value={statistics.active_deployments}
                prefix={<RocketOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="近期上传"
                value={statistics.recent_uploads}
                prefix={<ClockCircleOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* 主内容标签页 */}
      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="模型列表" key="models">
            {/* 过滤器 */}
            <Space style={{ marginBottom: 16 }} wrap>
              <Select
                placeholder="格式"
                style={{ width: 120 }}
                allowClear
                onChange={value => setFilters({ ...filters, format: value })}
              >
                <Option value="pytorch">PyTorch</Option>
                <Option value="tensorflow">TensorFlow</Option>
                <Option value="onnx">ONNX</Option>
                <Option value="huggingface">HuggingFace</Option>
              </Select>
              <Select
                placeholder="类型"
                style={{ width: 120 }}
                allowClear
                onChange={value =>
                  setFilters({ ...filters, model_type: value })
                }
              >
                <Option value="classification">分类</Option>
                <Option value="detection">检测</Option>
                <Option value="segmentation">分割</Option>
                <Option value="nlp">NLP</Option>
                <Option value="generative">生成</Option>
              </Select>
              <Select
                placeholder="状态"
                style={{ width: 120 }}
                allowClear
                onChange={value => setFilters({ ...filters, status: value })}
              >
                <Option value="uploaded">已上传</Option>
                <Option value="validated">已验证</Option>
                <Option value="deployed">已部署</Option>
                <Option value="deprecated">已弃用</Option>
              </Select>
              <Input.Search
                name="modelSearch"
                placeholder="搜索模型..."
                style={{ width: 300 }}
                onSearch={value => setFilters({ ...filters, search: value })}
                allowClear
              />
            </Space>

            {/* 模型表格 */}
            <Table
              columns={modelColumns}
              dataSource={models}
              rowKey="model_id"
              loading={loading}
              scroll={{ x: 1200 }}
            />
          </TabPane>

          <TabPane tab="格式分布" key="statistics">
            {statistics && (
              <Row gutter={[16, 16]}>
                <Col span={12}>
                  <Card title="模型格式分布">
                    <Pie {...formatDistributionChart()} height={300} />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="模型类型分布">
                    <Column
                      data={Object.entries(statistics.by_type).map(
                        ([k, v]) => ({ type: k, count: v })
                      )}
                      xField="type"
                      yField="count"
                      height={300}
                    />
                  </Card>
                </Col>
              </Row>
            )}
          </TabPane>
        </Tabs>
      </Card>

      {/* 上传模型模态框 */}
      <Modal
        title="上传模型"
        visible={uploadModalVisible}
        onCancel={() => setUploadModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={handleUpload}>
          <Form.Item
            name="name"
            label="模型名称"
            rules={[{ required: true, message: '请输入模型名称' }]}
          >
            <Input placeholder="例如: bert-base-chinese" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="version" label="版本" initialValue="1.0.0">
                <Input placeholder="1.0.0" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="author" label="作者">
                <Input placeholder="模型作者" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="format"
                label="模型格式"
                rules={[{ required: true, message: '请选择模型格式' }]}
              >
                <Select>
                  <Option value="pytorch">PyTorch</Option>
                  <Option value="tensorflow">TensorFlow</Option>
                  <Option value="onnx">ONNX</Option>
                  <Option value="huggingface">HuggingFace</Option>
                  <Option value="tensorrt">TensorRT</Option>
                  <Option value="mlflow">MLflow</Option>
                  <Option value="custom">自定义</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="model_type"
                label="模型类型"
                rules={[{ required: true, message: '请选择模型类型' }]}
              >
                <Select>
                  <Option value="classification">分类</Option>
                  <Option value="detection">检测</Option>
                  <Option value="segmentation">分割</Option>
                  <Option value="nlp">NLP</Option>
                  <Option value="generative">生成</Option>
                  <Option value="embedding">嵌入</Option>
                  <Option value="custom">自定义</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="description" label="描述">
            <TextArea rows={3} placeholder="模型描述..." />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="training_framework" label="训练框架">
                <Input placeholder="PyTorch 2.0" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="parameters_count" label="参数数量">
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="102000000"
                  formatter={value =>
                    `${value}`.replace(/\B(?=(\d{3})+(?!\d))/g, ',')
                  }
                />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="tags" label="标签">
            <Select mode="tags" placeholder="输入标签...">
              <Option value="nlp">NLP</Option>
              <Option value="vision">Vision</Option>
              <Option value="audio">Audio</Option>
              <Option value="multimodal">Multimodal</Option>
            </Select>
          </Form.Item>

          <Form.Item name="file" label="模型文件">
            <Upload beforeUpload={() => false} maxCount={1}>
              <Button icon={<UploadOutlined />}>选择文件</Button>
            </Upload>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                上传
              </Button>
              <Button onClick={() => setUploadModalVisible(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 从Hub导入模态框 */}
      <Modal
        title="从HuggingFace Hub导入"
        visible={hubModalVisible}
        onCancel={() => setHubModalVisible(false)}
        footer={null}
        width={500}
      >
        <Form form={hubForm} layout="vertical" onFinish={handleHubImport}>
          <Form.Item
            name="model_name"
            label="模型名称"
            rules={[{ required: true, message: '请输入模型名称' }]}
          >
            <Input
              placeholder="例如: bert-base-uncased 或 meta-llama/Llama-2-7b"
              prefix={<GlobalOutlined />}
            />
          </Form.Item>

          <Form.Item name="revision" label="版本/分支" initialValue="main">
            <Input placeholder="main" />
          </Form.Item>

          <Form.Item
            name="use_auth_token"
            label="需要认证"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Alert
            message="注意"
            description="导入大型模型可能需要较长时间，请耐心等待。"
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />

          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<CloudUploadOutlined />}
              >
                导入
              </Button>
              <Button onClick={() => setHubModalVisible(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 模型详情模态框 */}
      <Modal
        title={`模型详情: ${selectedModel?.metadata.name}`}
        visible={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        width={900}
        footer={[
          <Button key="close" onClick={() => setDetailModalVisible(false)}>
            关闭
          </Button>,
        ]}
      >
        {selectedModel && (
          <Tabs defaultActiveKey="info">
            <TabPane tab="基本信息" key="info">
              <Descriptions column={2} bordered>
                <Descriptions.Item label="模型ID">
                  {selectedModel.model_id}
                </Descriptions.Item>
                <Descriptions.Item label="名称">
                  {selectedModel.metadata.name}
                </Descriptions.Item>
                <Descriptions.Item label="版本">
                  {selectedModel.metadata.version}
                </Descriptions.Item>
                <Descriptions.Item label="格式">
                  {selectedModel.metadata.format}
                </Descriptions.Item>
                <Descriptions.Item label="类型">
                  {selectedModel.metadata.model_type}
                </Descriptions.Item>
                <Descriptions.Item label="状态">
                  <Tag color={getStatusColor(selectedModel.status)}>
                    {selectedModel.status.toUpperCase()}
                  </Tag>
                </Descriptions.Item>
                <Descriptions.Item label="作者">
                  {selectedModel.metadata.author || '-'}
                </Descriptions.Item>
                <Descriptions.Item label="许可证">
                  {selectedModel.metadata.license || '-'}
                </Descriptions.Item>
                <Descriptions.Item label="参数量">
                  {formatNumber(selectedModel.metadata.parameters_count)}
                </Descriptions.Item>
                <Descriptions.Item label="模型大小">
                  {formatSize(selectedModel.metadata.model_size_mb)}
                </Descriptions.Item>
                <Descriptions.Item label="训练框架">
                  {selectedModel.metadata.training_framework || '-'}
                </Descriptions.Item>
                <Descriptions.Item label="训练数据集">
                  {selectedModel.metadata.training_dataset || '-'}
                </Descriptions.Item>
                <Descriptions.Item label="创建时间">
                  {new Date(selectedModel.created_at).toLocaleString()}
                </Descriptions.Item>
                <Descriptions.Item label="更新时间">
                  {new Date(selectedModel.updated_at).toLocaleString()}
                </Descriptions.Item>
                <Descriptions.Item label="描述" span={2}>
                  {selectedModel.metadata.description || '-'}
                </Descriptions.Item>
                <Descriptions.Item label="标签" span={2}>
                  <Space wrap>
                    {selectedModel.metadata.tags.map(tag => (
                      <Tag key={tag}>{tag}</Tag>
                    ))}
                  </Space>
                </Descriptions.Item>
              </Descriptions>
            </TabPane>

            <TabPane
              tab={`版本 (${selectedModelVersions.length})`}
              key="versions"
            >
              <Table
                columns={versionColumns}
                dataSource={selectedModelVersions}
                rowKey="version"
                size="small"
              />
            </TabPane>

            <TabPane
              tab={`部署 (${selectedModelDeployments.length})`}
              key="deployments"
            >
              <Table
                columns={deploymentColumns}
                dataSource={selectedModelDeployments}
                rowKey="deployment_id"
                size="small"
              />
            </TabPane>

            <TabPane
              tab={`评估 (${selectedModelEvaluations.length})`}
              key="evaluations"
            >
              {selectedModelEvaluations.map(evaluation => (
                <Card
                  key={evaluation.evaluation_id}
                  size="small"
                  style={{ marginBottom: 16 }}
                >
                  <Descriptions column={3} size="small">
                    <Descriptions.Item label="数据集">
                      {evaluation.dataset_name}
                    </Descriptions.Item>
                    <Descriptions.Item label="版本">
                      {evaluation.model_version}
                    </Descriptions.Item>
                    <Descriptions.Item label="耗时">
                      {evaluation.evaluation_time_seconds}秒
                    </Descriptions.Item>
                  </Descriptions>
                  <Divider />
                  <Row gutter={16}>
                    {Object.entries(evaluation.metrics).map(
                      ([metric, value]) => (
                        <Col key={metric} span={6}>
                          <Statistic
                            title={metric.toUpperCase()}
                            value={value}
                            precision={3}
                          />
                        </Col>
                      )
                    )}
                  </Row>
                </Card>
              ))}
              {selectedModelEvaluations.length === 0 && (
                <Alert
                  message="暂无评估记录"
                  description="点击下方按钮开始评估模型"
                  type="info"
                  showIcon
                  action={
                    <Button
                      size="small"
                      type="primary"
                      onClick={() => {
                        setDetailModalVisible(false)
                        setEvaluateModalVisible(true)
                      }}
                    >
                      评估模型
                    </Button>
                  }
                />
              )}
            </TabPane>
          </Tabs>
        )}
      </Modal>

      {/* 部署模态框 */}
      <Modal
        title="部署模型"
        visible={deployModalVisible}
        onCancel={() => setDeployModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={deployForm}
          layout="vertical"
          onFinish={handleDeploy}
          initialValues={{
            environment: 'development',
            replicas: 1,
            cpu_request: '1',
            memory_request: '2Gi',
            auto_scaling: false,
          }}
        >
          <Form.Item
            name="environment"
            label="部署环境"
            rules={[{ required: true }]}
          >
            <Select>
              <Option value="development">开发环境</Option>
              <Option value="staging">预发布环境</Option>
              <Option value="production">生产环境</Option>
            </Select>
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="replicas" label="副本数">
                <InputNumber min={1} max={10} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="gpu_request" label="GPU数量">
                <InputNumber min={0} max={8} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="cpu_request" label="CPU请求">
                <Input placeholder="1" addonAfter="核" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="memory_request" label="内存请求">
                <Input placeholder="2Gi" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="auto_scaling"
            label="自动扩缩容"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<RocketOutlined />}
              >
                部署
              </Button>
              <Button onClick={() => setDeployModalVisible(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 评估模态框 */}
      <Modal
        title="评估模型"
        visible={evaluateModalVisible}
        onCancel={() => setEvaluateModalVisible(false)}
        footer={null}
        width={500}
      >
        <Form
          form={evaluateForm}
          layout="vertical"
          onFinish={handleEvaluate}
          initialValues={{
            batch_size: 32,
            device: 'cuda',
            metrics: ['accuracy', 'precision', 'recall', 'f1'],
          }}
        >
          <Form.Item
            name="dataset_name"
            label="评估数据集"
            rules={[{ required: true, message: '请选择数据集' }]}
          >
            <Select>
              <Option value="ImageNet">ImageNet</Option>
              <Option value="COCO">COCO</Option>
              <Option value="GLUE">GLUE</Option>
              <Option value="SQuAD">SQuAD</Option>
              <Option value="Custom">自定义数据集</Option>
            </Select>
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="batch_size" label="批次大小">
                <InputNumber min={1} max={256} style={{ width: '100%' }} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="device" label="设备">
                <Select>
                  <Option value="cuda">GPU</Option>
                  <Option value="cpu">CPU</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="metrics" label="评估指标">
            <Select mode="multiple">
              <Option value="accuracy">准确率</Option>
              <Option value="precision">精确率</Option>
              <Option value="recall">召回率</Option>
              <Option value="f1">F1分数</Option>
              <Option value="mAP">mAP</Option>
              <Option value="iou">IoU</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button
                type="primary"
                htmlType="submit"
                icon={<ExperimentOutlined />}
              >
                开始评估
              </Button>
              <Button onClick={() => setEvaluateModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default ModelRegistryPage
