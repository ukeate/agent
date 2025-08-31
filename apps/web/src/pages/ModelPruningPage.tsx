import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Form,
  Select,
  Input,
  InputNumber,
  Switch,
  Upload,
  Progress,
  Table,
  Space,
  Typography,
  Tabs,
  Modal,
  Alert,
  Steps,
  Divider,
  Tag,
  Tooltip,
  Slider,
  message,
  Statistic,
  List,
  Avatar,
  TreeSelect,
  Checkbox
} from 'antd'
import {
  ScissorOutlined,
  UploadOutlined,
  PlayCircleOutlined,
  SettingOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  PartitionOutlined,
  ClusterOutlined,
  AimOutlined,
  EyeOutlined,
  DownloadOutlined,
  DeleteOutlined,
  LineChartOutlined,
  BarChartOutlined,
  ThunderboltOutlined,
  DatabaseOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Step } = Steps
const { Option } = Select

interface PruningConfig {
  pruningType: 'structured' | 'unstructured'
  sparsityRatio: number
  importanceMetric: 'magnitude' | 'gradient' | 'l1_norm' | 'l2_norm' | 'taylor_expansion'
  structuredType?: 'channel' | 'filter' | 'layer'
  gradualPruning: boolean
  pruningSteps: number
  recoveryEpochs: number
  sensitivityAnalysis: boolean
  layerWiseSparsity: boolean
  layerSparsityRatios: Record<string, number>
}

interface PruningJob {
  id: string
  name: string
  modelPath: string
  config: PruningConfig
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  currentStep: number
  totalSteps: number
  originalParams: number
  prunedParams: number
  sparsityAchieved: number
  accuracyBefore: number
  accuracyAfter: number
  speedupRatio: number
  startTime: string
  endTime?: string
  logs: string[]
}

const ModelPruningPage: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('create')
  const [currentStep, setCurrentStep] = useState(0)
  const [selectedModel, setSelectedModel] = useState<any>(null)
  const [config, setConfig] = useState<PruningConfig>({
    pruningType: 'unstructured',
    sparsityRatio: 0.5,
    importanceMetric: 'magnitude',
    gradualPruning: true,
    pruningSteps: 3,
    recoveryEpochs: 5,
    sensitivityAnalysis: false,
    layerWiseSparsity: false,
    layerSparsityRatios: {}
  })
  const [jobs, setJobs] = useState<PruningJob[]>([])
  const [jobModalVisible, setJobModalVisible] = useState(false)
  const [selectedJob, setSelectedJob] = useState<PruningJob | null>(null)
  const [modelLayers, setModelLayers] = useState<string[]>([])

  useEffect(() => {
    // 模拟加载现有任务
    setJobs([
      {
        id: 'prune-001',
        name: 'ResNet50非结构化剪枝',
        modelPath: '/models/resnet50.pt',
        config: {
          pruningType: 'unstructured',
          sparsityRatio: 0.7,
          importanceMetric: 'magnitude',
          gradualPruning: true,
          pruningSteps: 3,
          recoveryEpochs: 5,
          sensitivityAnalysis: false,
          layerWiseSparsity: false,
          layerSparsityRatios: {}
        },
        status: 'completed',
        progress: 100,
        currentStep: 3,
        totalSteps: 3,
        originalParams: 25557032,
        prunedParams: 7667110,
        sparsityAchieved: 70.0,
        accuracyBefore: 76.2,
        accuracyAfter: 74.8,
        speedupRatio: 2.1,
        startTime: '2024-08-24 09:30:00',
        endTime: '2024-08-24 10:45:00',
        logs: [
          '开始非结构化剪枝任务...',
          '加载ResNet50模型...',
          '计算权重重要性（幅度）...',
          'Step 1/3: 剪枝23.3%权重...',
          '恢复训练5个epochs...',
          'Step 2/3: 剪枝46.7%权重...',
          '恢复训练5个epochs...',
          'Step 3/3: 剪枝70.0%权重...',
          '恢复训练5个epochs...',
          '最终稀疏度: 70.0%',
          '精度保持: 74.8% (损失1.4%)',
          '加速比: 2.1x',
          '剪枝完成！'
        ]
      },
      {
        id: 'prune-002',
        name: 'MobileNet结构化剪枝',
        modelPath: '/models/mobilenet_v2.pt',
        config: {
          pruningType: 'structured',
          sparsityRatio: 0.4,
          importanceMetric: 'l2_norm',
          structuredType: 'channel',
          gradualPruning: false,
          pruningSteps: 1,
          recoveryEpochs: 10,
          sensitivityAnalysis: true,
          layerWiseSparsity: false,
          layerSparsityRatios: {}
        },
        status: 'running',
        progress: 60,
        currentStep: 1,
        totalSteps: 1,
        originalParams: 3504872,
        prunedParams: 2103000,
        sparsityAchieved: 40.0,
        accuracyBefore: 72.1,
        accuracyAfter: 0,
        speedupRatio: 0,
        startTime: '2024-08-24 11:15:00',
        logs: [
          '开始结构化剪枝任务...',
          '加载MobileNet-V2模型...',
          '执行敏感性分析...',
          '基于L2范数计算通道重要性...',
          '剪枝40%低重要性通道...',
          '当前恢复训练进度: 6/10 epochs'
        ]
      }
    ])

    // 模拟模型层结构
    setModelLayers([
      'conv1',
      'layer1.0.conv1',
      'layer1.0.conv2',
      'layer1.1.conv1',
      'layer1.1.conv2',
      'layer2.0.conv1',
      'layer2.0.conv2',
      'layer2.1.conv1',
      'layer2.1.conv2',
      'layer3.0.conv1',
      'layer3.0.conv2',
      'layer3.1.conv1',
      'layer3.1.conv2',
      'layer4.0.conv1',
      'layer4.0.conv2',
      'layer4.1.conv1',
      'layer4.1.conv2',
      'fc'
    ])
  }, [])

  const pruningTypeOptions = [
    {
      value: 'unstructured',
      label: '非结构化剪枝',
      description: '剪除单个权重参数，保持网络结构',
      icon: <ClusterOutlined />
    },
    {
      value: 'structured',
      label: '结构化剪枝',
      description: '剪除整个通道/滤波器，改变网络结构',
      icon: <PartitionOutlined />
    }
  ]

  const importanceMetrics = [
    { value: 'magnitude', label: '权重幅度', description: '基于权重绝对值的重要性' },
    { value: 'gradient', label: '梯度信息', description: '基于梯度的重要性评估' },
    { value: 'l1_norm', label: 'L1范数', description: '基于L1范数的重要性' },
    { value: 'l2_norm', label: 'L2范数', description: '基于L2范数的重要性' },
    { value: 'taylor_expansion', label: '泰勒展开', description: '基于泰勒展开的重要性' }
  ]

  const structuredTypes = [
    { value: 'channel', label: '通道级剪枝', description: '剪除整个输出通道' },
    { value: 'filter', label: '滤波器剪枝', description: '剪除卷积滤波器' },
    { value: 'layer', label: '层级剪枝', description: '剪除整个网络层' }
  ]

  const handleStartPruning = async () => {
    try {
      const values = await form.validateFields()
      setLoading(true)

      // 创建新的剪枝任务
      const newJob: PruningJob = {
        id: `prune-${Date.now()}`,
        name: values.jobName,
        modelPath: values.modelPath || selectedModel?.name,
        config,
        status: 'pending',
        progress: 0,
        currentStep: 0,
        totalSteps: config.gradualPruning ? config.pruningSteps : 1,
        originalParams: Math.round(Math.random() * 50000000 + 1000000),
        prunedParams: 0,
        sparsityAchieved: 0,
        accuracyBefore: Math.round((Math.random() * 10 + 70) * 10) / 10,
        accuracyAfter: 0,
        speedupRatio: 0,
        startTime: new Date().toLocaleString(),
        logs: ['任务已创建，等待开始...']
      }

      setJobs(prev => [newJob, ...prev])
      message.success('剪枝任务创建成功！')
      
      // 模拟任务启动
      setTimeout(() => {
        setJobs(prev => prev.map(job => 
          job.id === newJob.id 
            ? { ...job, status: 'running' as const, logs: [...job.logs, '开始剪枝处理...'] }
            : job
        ))
      }, 1000)

      setActiveTab('jobs')
      form.resetFields()
      setCurrentStep(0)

    } catch (error) {
      console.error('创建任务失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: PruningJob['status']) => {
    switch (status) {
      case 'pending': return '#faad14'
      case 'running': return '#1890ff'
      case 'completed': return '#52c41a'
      case 'failed': return '#ff4d4f'
      default: return '#d9d9d9'
    }
  }

  const getStatusText = (status: PruningJob['status']) => {
    switch (status) {
      case 'pending': return '等待中'
      case 'running': return '剪枝中'
      case 'completed': return '已完成'
      case 'failed': return '失败'
      default: return '未知'
    }
  }

  const jobColumns = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: PruningJob) => (
        <div>
          <Text strong>{text}</Text>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.config.pruningType === 'structured' ? (
              <PartitionOutlined style={{ marginRight: '4px' }} />
            ) : (
              <ClusterOutlined style={{ marginRight: '4px' }} />
            )}
            {record.config.pruningType === 'structured' ? '结构化' : '非结构化'}剪枝
          </div>
        </div>
      )
    },
    {
      title: '剪枝类型',
      key: 'pruningType',
      render: (record: PruningJob) => (
        <Tag 
          color={record.config.pruningType === 'structured' ? 'purple' : 'blue'}
          icon={record.config.pruningType === 'structured' ? <PartitionOutlined /> : <ClusterOutlined />}
        >
          {record.config.pruningType === 'structured' ? '结构化' : '非结构化'}
        </Tag>
      )
    },
    {
      title: '目标稀疏度',
      key: 'sparsityRatio',
      render: (record: PruningJob) => (
        <Text strong style={{ color: '#722ed1' }}>
          {(record.config.sparsityRatio * 100).toFixed(0)}%
        </Text>
      )
    },
    {
      title: '状态',
      key: 'status',
      render: (record: PruningJob) => (
        <Tag color={getStatusColor(record.status)}>
          {getStatusText(record.status)}
        </Tag>
      )
    },
    {
      title: '剪枝进度',
      key: 'progress',
      render: (record: PruningJob) => (
        <div>
          <Progress 
            percent={record.progress} 
            size="small"
            status={record.status === 'failed' ? 'exception' : undefined}
          />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Step {record.currentStep}/{record.totalSteps}
          </Text>
        </div>
      )
    },
    {
      title: '实际稀疏度',
      key: 'sparsityAchieved',
      render: (record: PruningJob) => (
        record.sparsityAchieved > 0 ? 
          <Text type="success" strong>{record.sparsityAchieved.toFixed(1)}%</Text> : 
          <Text type="secondary">-</Text>
      )
    },
    {
      title: '加速比',
      key: 'speedupRatio',
      render: (record: PruningJob) => (
        record.speedupRatio > 0 ? 
          <Text type="success" strong>{record.speedupRatio.toFixed(1)}x</Text> : 
          <Text type="secondary">-</Text>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: PruningJob) => (
        <Space>
          <Button 
            type="link" 
            size="small" 
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedJob(record)
              setJobModalVisible(true)
            }}
          >
            查看
          </Button>
          {record.status === 'completed' && (
            <Button type="link" size="small" icon={<DownloadOutlined />}>
              下载
            </Button>
          )}
          {record.status === 'running' && (
            <Button type="link" size="small" danger icon={<DeleteOutlined />}>
              停止
            </Button>
          )}
        </Space>
      )
    }
  ]

  const renderCreateForm = () => (
    <Card title="创建剪枝任务">
      <Steps current={currentStep} style={{ marginBottom: '24px' }}>
        <Step title="选择模型" icon={<UploadOutlined />} />
        <Step title="剪枝配置" icon={<SettingOutlined />} />
        <Step title="高级设置" icon={<AimOutlined />} />
        <Step title="确认启动" icon={<PlayCircleOutlined />} />
      </Steps>

      <Form form={form} layout="vertical">
        {currentStep === 0 && (
          <div>
            <Form.Item
              name="jobName"
              label="任务名称"
              rules={[{ required: true, message: '请输入任务名称' }]}
            >
              <Input placeholder="例如: ResNet50非结构化剪枝" />
            </Form.Item>

            <Form.Item
              name="modelUpload"
              label="上传模型文件"
            >
              <Upload.Dragger
                multiple={false}
                showUploadList={false}
                beforeUpload={() => false}
                onChange={(info) => {
                  setSelectedModel(info.file)
                  message.success(`已选择模型: ${info.file.name}`)
                }}
              >
                <p className="ant-upload-drag-icon">
                  <DatabaseOutlined />
                </p>
                <p className="ant-upload-text">点击或拖拽上传模型文件</p>
                <p className="ant-upload-hint">
                  支持 .pt, .pth, .onnx 格式
                </p>
              </Upload.Dragger>
            </Form.Item>

            {selectedModel && (
              <Alert
                message={`已选择模型: ${selectedModel.name}`}
                type="success"
                showIcon
                style={{ marginBottom: '16px' }}
              />
            )}
          </div>
        )}

        {currentStep === 1 && (
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="剪枝类型" size="small">
                <Form.Item label="剪枝方式">
                  <Select
                    value={config.pruningType}
                    onChange={(value) => setConfig({ ...config, pruningType: value })}
                  >
                    {pruningTypeOptions.map(option => (
                      <Option key={option.value} value={option.value}>
                        <Space>
                          {option.icon}
                          <div>
                            <Text strong>{option.label}</Text>
                            <br />
                            <Text type="secondary" style={{ fontSize: '12px' }}>
                              {option.description}
                            </Text>
                          </div>
                        </Space>
                      </Option>
                    ))}
                  </Select>
                </Form.Item>

                {config.pruningType === 'structured' && (
                  <Form.Item label="结构化类型">
                    <Select
                      value={config.structuredType}
                      onChange={(value) => setConfig({ ...config, structuredType: value })}
                    >
                      {structuredTypes.map(option => (
                        <Option key={option.value} value={option.value}>
                          <div>
                            <Text strong>{option.label}</Text>
                            <br />
                            <Text type="secondary" style={{ fontSize: '12px' }}>
                              {option.description}
                            </Text>
                          </div>
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>
                )}

                <Form.Item label={`目标稀疏度 (${(config.sparsityRatio * 100).toFixed(0)}%)`}>
                  <Slider
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={config.sparsityRatio}
                    onChange={(value) => setConfig({ ...config, sparsityRatio: value })}
                    marks={{
                      0.1: '10%',
                      0.3: '30%',
                      0.5: '50%',
                      0.7: '70%',
                      0.9: '90%'
                    }}
                  />
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    稀疏度越高，模型越小但可能影响精度
                  </Text>
                </Form.Item>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="重要性评估" size="small">
                <Form.Item label="重要性指标">
                  <Select
                    value={config.importanceMetric}
                    onChange={(value) => setConfig({ ...config, importanceMetric: value })}
                  >
                    {importanceMetrics.map(option => (
                      <Option key={option.value} value={option.value}>
                        <div>
                          <Text strong>{option.label}</Text>
                          <br />
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            {option.description}
                          </Text>
                        </div>
                      </Option>
                    ))}
                  </Select>
                </Form.Item>

                <Form.Item>
                  <Checkbox
                    checked={config.sensitivityAnalysis}
                    onChange={(e) => setConfig({ ...config, sensitivityAnalysis: e.target.checked })}
                  >
                    启用敏感性分析
                  </Checkbox>
                  <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                    分析各层对剪枝的敏感程度
                  </div>
                </Form.Item>

                <Form.Item>
                  <Checkbox
                    checked={config.layerWiseSparsity}
                    onChange={(e) => setConfig({ ...config, layerWiseSparsity: e.target.checked })}
                  >
                    逐层稀疏度配置
                  </Checkbox>
                  <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                    为不同层设置不同的稀疏度
                  </div>
                </Form.Item>
              </Card>
            </Col>
          </Row>
        )}

        {currentStep === 2 && (
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="渐进式剪枝" size="small">
                <Form.Item>
                  <Switch
                    checked={config.gradualPruning}
                    onChange={(checked) => setConfig({ ...config, gradualPruning: checked })}
                  />
                  <span style={{ marginLeft: '8px' }}>启用渐进式剪枝</span>
                  <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                    分多个步骤逐步剪枝，减少精度损失
                  </div>
                </Form.Item>

                {config.gradualPruning && (
                  <>
                    <Form.Item label="剪枝步数">
                      <InputNumber
                        min={2}
                        max={10}
                        value={config.pruningSteps}
                        onChange={(value) => setConfig({ ...config, pruningSteps: value || 3 })}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>

                    <Form.Item label="恢复训练轮次">
                      <InputNumber
                        min={1}
                        max={20}
                        value={config.recoveryEpochs}
                        onChange={(value) => setConfig({ ...config, recoveryEpochs: value || 5 })}
                        style={{ width: '100%' }}
                      />
                    </Form.Item>
                  </>
                )}
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              {config.layerWiseSparsity && (
                <Card title="逐层稀疏度配置" size="small">
                  <div style={{ maxHeight: '300px', overflowY: 'auto' }}>
                    {modelLayers.map(layer => (
                      <div key={layer} style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        marginBottom: '8px' 
                      }}>
                        <Text style={{ 
                          width: '120px', 
                          fontSize: '12px',
                          overflow: 'hidden',
                          textOverflow: 'ellipsis'
                        }}>
                          {layer}
                        </Text>
                        <Slider
                          min={0}
                          max={0.9}
                          step={0.1}
                          value={config.layerSparsityRatios[layer] || config.sparsityRatio}
                          onChange={(value) => setConfig({
                            ...config,
                            layerSparsityRatios: {
                              ...config.layerSparsityRatios,
                              [layer]: value
                            }
                          })}
                          style={{ flex: 1, margin: '0 8px' }}
                        />
                        <Text style={{ width: '40px', fontSize: '12px' }}>
                          {((config.layerSparsityRatios[layer] || config.sparsityRatio) * 100).toFixed(0)}%
                        </Text>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </Col>
          </Row>
        )}

        {currentStep === 3 && (
          <div>
            <Alert
              message="确认剪枝配置"
              description="请检查以下配置信息，确认无误后点击开始剪枝"
              type="info"
              showIcon
              style={{ marginBottom: '24px' }}
            />

            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <Card title="剪枝配置" size="small">
                  <List size="small">
                    <List.Item>
                      <Text type="secondary">剪枝类型:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {config.pruningType === 'structured' ? '结构化' : '非结构化'}
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">目标稀疏度:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {(config.sparsityRatio * 100).toFixed(0)}%
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">重要性指标:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {importanceMetrics.find(m => m.value === config.importanceMetric)?.label}
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">渐进式剪枝:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {config.gradualPruning ? `是 (${config.pruningSteps}步)` : '否'}
                      </Text>
                    </List.Item>
                  </List>
                </Card>
              </Col>

              <Col xs={24} lg={12}>
                <Card title="高级选项" size="small">
                  <List size="small">
                    <List.Item>
                      <Text type="secondary">敏感性分析:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {config.sensitivityAnalysis ? '是' : '否'}
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">逐层配置:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {config.layerWiseSparsity ? '是' : '否'}
                      </Text>
                    </List.Item>
                    {config.gradualPruning && (
                      <List.Item>
                        <Text type="secondary">恢复训练:</Text>
                        <Text strong style={{ marginLeft: '8px' }}>
                          {config.recoveryEpochs} 轮次
                        </Text>
                      </List.Item>
                    )}
                  </List>
                </Card>
              </Col>
            </Row>
          </div>
        )}

        <div style={{ marginTop: '24px', textAlign: 'right' }}>
          <Space>
            {currentStep > 0 && (
              <Button onClick={() => setCurrentStep(currentStep - 1)}>
                上一步
              </Button>
            )}
            {currentStep < 3 ? (
              <Button 
                type="primary" 
                onClick={() => setCurrentStep(currentStep + 1)}
                disabled={currentStep === 0 && !selectedModel}
              >
                下一步
              </Button>
            ) : (
              <Button 
                type="primary" 
                loading={loading}
                onClick={handleStartPruning}
                icon={<PlayCircleOutlined />}
              >
                开始剪枝
              </Button>
            )}
          </Space>
        </div>
      </Form>
    </Card>
  )

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2} style={{ margin: 0, color: '#1a1a1a' }}>
          <ScissorOutlined style={{ marginRight: '12px', color: '#722ed1' }} />
          模型剪枝管理
        </Title>
        <Paragraph style={{ marginTop: '8px', color: '#666', fontSize: '16px' }}>
          通过结构化和非结构化剪枝技术，实现模型参数减少和推理加速
        </Paragraph>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="创建任务" key="create">
          {renderCreateForm()}
        </TabPane>
        
        <TabPane tab="任务列表" key="jobs">
          <Card 
            title="剪枝任务" 
            extra={
              <Button 
                type="primary" 
                onClick={() => setActiveTab('create')}
                icon={<PlayCircleOutlined />}
              >
                创建新任务
              </Button>
            }
          >
            <Table
              columns={jobColumns}
              dataSource={jobs}
              rowKey="id"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* 任务详情模态框 */}
      <Modal
        title="剪枝任务详情"
        open={jobModalVisible}
        onCancel={() => setJobModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setJobModalVisible(false)}>
            关闭
          </Button>
        ]}
        width={900}
      >
        {selectedJob && (
          <div>
            <Row gutter={[16, 16]}>
              <Col span={8}>
                <Card title="原始模型" size="small">
                  <div style={{ textAlign: 'center' }}>
                    <Avatar size={48} icon={<DatabaseOutlined />} style={{ backgroundColor: '#1890ff' }} />
                    <div style={{ marginTop: '8px' }}>
                      <Text strong>参数: {selectedJob.originalParams.toLocaleString()}</Text>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        精度: {selectedJob.accuracyBefore}%
                      </div>
                    </div>
                  </div>
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="剪枝过程" size="small">
                  <div style={{ textAlign: 'center' }}>
                    <Progress
                      type="circle"
                      percent={selectedJob.progress}
                      width={60}
                    />
                    <div style={{ marginTop: '8px' }}>
                      <Text>Step {selectedJob.currentStep}/{selectedJob.totalSteps}</Text>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        目标稀疏度: {(selectedJob.config.sparsityRatio * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="剪枝后模型" size="small">
                  <div style={{ textAlign: 'center' }}>
                    <Avatar size={48} icon={<ScissorOutlined />} style={{ backgroundColor: '#722ed1' }} />
                    <div style={{ marginTop: '8px' }}>
                      <Text strong>参数: {selectedJob.prunedParams.toLocaleString()}</Text>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        精度: {selectedJob.accuracyAfter > 0 ? `${selectedJob.accuracyAfter}%` : '-'}
                      </div>
                    </div>
                  </div>
                </Card>
              </Col>
            </Row>

            {selectedJob.status === 'completed' && (
              <>
                <Divider />
                <Title level={4}>剪枝结果</Title>
                <Row gutter={[16, 16]}>
                  <Col span={6}>
                    <Statistic
                      title="实际稀疏度"
                      value={selectedJob.sparsityAchieved}
                      precision={1}
                      suffix="%"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="参数减少"
                      value={((selectedJob.originalParams - selectedJob.prunedParams) / selectedJob.originalParams * 100)}
                      precision={1}
                      suffix="%"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="精度损失"
                      value={selectedJob.accuracyBefore - selectedJob.accuracyAfter}
                      precision={1}
                      suffix="%"
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="加速比"
                      value={selectedJob.speedupRatio}
                      precision={1}
                      suffix="x"
                    />
                  </Col>
                </Row>
              </>
            )}

            <Divider />
            <Title level={4}>执行日志</Title>
            <div style={{ 
              background: '#f5f5f5', 
              padding: '12px', 
              borderRadius: '4px',
              maxHeight: '200px',
              overflowY: 'auto',
              fontFamily: 'monospace'
            }}>
              {selectedJob.logs.map((log, index) => (
                <div key={index} style={{ margin: '4px 0', fontSize: '12px' }}>
                  <Text type="secondary">{new Date().toLocaleTimeString()}</Text>
                  <span style={{ margin: '0 8px' }}>|</span>
                  <Text>{log}</Text>
                </div>
              ))}
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default ModelPruningPage