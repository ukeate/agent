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
  Avatar
} from 'antd'
import {
  ExperimentFilled,
  UploadOutlined,
  PlayCircleOutlined,
  SettingOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  FireOutlined,
  TeamOutlined,
  EyeOutlined,
  DownloadOutlined,
  DeleteOutlined,
  LineChartOutlined,
  BulbOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Step } = Steps
const { Option } = Select

interface DistillationConfig {
  strategy: 'response_based' | 'feature_based' | 'attention_based' | 'self_distillation'
  temperature: number
  alpha: number
  beta: number
  numEpochs: number
  learningRate: number
  batchSize: number
  optimizer: 'adam' | 'sgd' | 'adamw'
  scheduler: string
  earlyStopping: boolean
  patience: number
}

interface DistillationJob {
  id: string
  name: string
  teacherModel: string
  studentModel: string
  config: DistillationConfig
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  currentEpoch: number
  totalEpochs: number
  teacherAccuracy: number
  studentAccuracy: number
  compressionRatio: number
  trainingLoss: number
  validationLoss: number
  startTime: string
  endTime?: string
  logs: string[]
}

const KnowledgeDistillationPage: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('create')
  const [currentStep, setCurrentStep] = useState(0)
  const [teacherModel, setTeacherModel] = useState<any>(null)
  const [studentModel, setStudentModel] = useState<any>(null)
  const [config, setConfig] = useState<DistillationConfig>({
    strategy: 'response_based',
    temperature: 3.0,
    alpha: 0.5,
    beta: 0.5,
    numEpochs: 10,
    learningRate: 0.0001,
    batchSize: 32,
    optimizer: 'adam',
    scheduler: 'cosine',
    earlyStopping: true,
    patience: 3
  })
  const [jobs, setJobs] = useState<DistillationJob[]>([])
  const [jobModalVisible, setJobModalVisible] = useState(false)
  const [selectedJob, setSelectedJob] = useState<DistillationJob | null>(null)

  useEffect(() => {
    // 模拟加载现有任务
    setJobs([
      {
        id: 'distill-001',
        name: 'BERT→DistilBERT',
        teacherModel: 'bert-base-uncased',
        studentModel: 'distilbert-base-uncased',
        config: {
          strategy: 'response_based',
          temperature: 3.0,
          alpha: 0.5,
          beta: 0.5,
          numEpochs: 10,
          learningRate: 0.0001,
          batchSize: 32,
          optimizer: 'adam',
          scheduler: 'cosine',
          earlyStopping: true,
          patience: 3
        },
        status: 'completed',
        progress: 100,
        currentEpoch: 10,
        totalEpochs: 10,
        teacherAccuracy: 92.1,
        studentAccuracy: 90.8,
        compressionRatio: 2.3,
        trainingLoss: 0.234,
        validationLoss: 0.267,
        startTime: '2024-08-24 09:30:00',
        endTime: '2024-08-24 11:15:00',
        logs: [
          '开始蒸馏任务...',
          '加载Teacher模型: BERT-Base',
          '加载Student模型: DistilBERT',
          '准备训练数据集...',
          'Epoch 1/10: Loss=0.521, Val_Loss=0.534',
          'Epoch 2/10: Loss=0.456, Val_Loss=0.478',
          '...',
          'Epoch 10/10: Loss=0.234, Val_Loss=0.267',
          '蒸馏完成，Student准确率: 90.8%',
          '压缩比: 2.3x',
          '保存蒸馏模型...',
          '任务完成！'
        ]
      },
      {
        id: 'distill-002',
        name: 'ResNet50→MobileNet',
        teacherModel: 'resnet50',
        studentModel: 'mobilenet-v2',
        config: {
          strategy: 'feature_based',
          temperature: 4.0,
          alpha: 0.3,
          beta: 0.7,
          numEpochs: 20,
          learningRate: 0.001,
          batchSize: 64,
          optimizer: 'sgd',
          scheduler: 'step',
          earlyStopping: false,
          patience: 5
        },
        status: 'running',
        progress: 45,
        currentEpoch: 9,
        totalEpochs: 20,
        teacherAccuracy: 76.2,
        studentAccuracy: 68.4,
        compressionRatio: 4.1,
        trainingLoss: 1.234,
        validationLoss: 1.456,
        startTime: '2024-08-24 14:20:00',
        logs: [
          '开始特征蒸馏任务...',
          '加载Teacher模型: ResNet50',
          '加载Student模型: MobileNet-V2',
          '准备ImageNet数据集...',
          'Epoch 1/20: Loss=2.145, Val_Loss=2.234',
          '...',
          'Epoch 9/20: Loss=1.234, Val_Loss=1.456',
          '当前Student准确率: 68.4%'
        ]
      }
    ])
  }, [])

  const strategyOptions = [
    {
      value: 'response_based',
      label: '响应式蒸馏',
      description: '基于输出概率分布的软标签蒸馏',
      icon: <BulbOutlined />
    },
    {
      value: 'feature_based',
      label: '特征式蒸馏',
      description: '基于中间层特征的知识迁移',
      icon: <LineChartOutlined />
    },
    {
      value: 'attention_based',
      label: '注意力蒸馏',
      description: '基于注意力机制的知识传递',
      icon: <EyeOutlined />
    },
    {
      value: 'self_distillation',
      label: '自蒸馏',
      description: '模型自我知识蒸馏',
      icon: <TeamOutlined />
    }
  ]

  const optimizerOptions = [
    { value: 'adam', label: 'Adam', description: '自适应学习率优化器' },
    { value: 'sgd', label: 'SGD', description: '随机梯度下降' },
    { value: 'adamw', label: 'AdamW', description: 'Adam with decoupled weight decay' }
  ]

  const handleStartDistillation = async () => {
    try {
      const values = await form.validateFields()
      setLoading(true)

      // 创建新的蒸馏任务
      const newJob: DistillationJob = {
        id: `distill-${Date.now()}`,
        name: values.jobName,
        teacherModel: teacherModel?.name || values.teacherModel,
        studentModel: studentModel?.name || values.studentModel,
        config,
        status: 'pending',
        progress: 0,
        currentEpoch: 0,
        totalEpochs: config.numEpochs,
        teacherAccuracy: Math.round((Math.random() * 10 + 85) * 10) / 10,
        studentAccuracy: 0,
        compressionRatio: Math.round((Math.random() * 3 + 2) * 10) / 10,
        trainingLoss: 0,
        validationLoss: 0,
        startTime: new Date().toLocaleString(),
        logs: ['任务已创建，等待开始...']
      }

      setJobs(prev => [newJob, ...prev])
      message.success('蒸馏任务创建成功！')
      
      // 模拟任务启动
      setTimeout(() => {
        setJobs(prev => prev.map(job => 
          job.id === newJob.id 
            ? { ...job, status: 'running' as const, logs: [...job.logs, '开始蒸馏训练...'] }
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

  const getStatusColor = (status: DistillationJob['status']) => {
    switch (status) {
      case 'pending': return '#faad14'
      case 'running': return '#1890ff'
      case 'completed': return '#52c41a'
      case 'failed': return '#ff4d4f'
      default: return '#d9d9d9'
    }
  }

  const getStatusText = (status: DistillationJob['status']) => {
    switch (status) {
      case 'pending': return '等待中'
      case 'running': return '训练中'
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
      render: (text: string, record: DistillationJob) => (
        <div>
          <Text strong>{text}</Text>
          <div style={{ fontSize: '12px', color: '#666' }}>
            <TeamOutlined style={{ marginRight: '4px' }} />
            {record.teacherModel} → {record.studentModel}
          </div>
        </div>
      )
    },
    {
      title: '蒸馏策略',
      key: 'strategy',
      render: (record: DistillationJob) => {
        const strategy = strategyOptions.find(s => s.value === record.config.strategy)
        return (
          <Tag color="orange" icon={strategy?.icon}>
            {strategy?.label}
          </Tag>
        )
      }
    },
    {
      title: '状态',
      key: 'status',
      render: (record: DistillationJob) => (
        <Tag color={getStatusColor(record.status)}>
          {getStatusText(record.status)}
        </Tag>
      )
    },
    {
      title: '训练进度',
      key: 'progress',
      render: (record: DistillationJob) => (
        <div>
          <Progress 
            percent={record.progress} 
            size="small"
            status={record.status === 'failed' ? 'exception' : undefined}
          />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            Epoch {record.currentEpoch}/{record.totalEpochs}
          </Text>
        </div>
      )
    },
    {
      title: '学生准确率',
      key: 'studentAccuracy',
      render: (record: DistillationJob) => (
        record.studentAccuracy > 0 ? 
          <Text type="success" strong>{record.studentAccuracy}%</Text> : 
          <Text type="secondary">-</Text>
      )
    },
    {
      title: '压缩比',
      key: 'compressionRatio',
      render: (record: DistillationJob) => (
        record.compressionRatio > 0 ? 
          <Text type="success" strong>{record.compressionRatio}x</Text> : 
          <Text type="secondary">-</Text>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: DistillationJob) => (
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
    <Card title="创建知识蒸馏任务">
      <Steps current={currentStep} style={{ marginBottom: '24px' }}>
        <Step title="选择模型" icon={<TeamOutlined />} />
        <Step title="配置蒸馏" icon={<SettingOutlined />} />
        <Step title="训练参数" icon={<SettingOutlined />} />
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
              <Input placeholder="例如: BERT→DistilBERT蒸馏" />
            </Form.Item>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="Teacher模型（教师模型）">
                  <Upload.Dragger
                    multiple={false}
                    showUploadList={false}
                    beforeUpload={() => false}
                    onChange={(info) => {
                      setTeacherModel(info.file)
                      message.success(`已选择Teacher模型: ${info.file.name}`)
                    }}
                  >
                    <p className="ant-upload-drag-icon">
                      <FireOutlined />
                    </p>
                    <p className="ant-upload-text">上传Teacher模型</p>
                    <p className="ant-upload-hint">
                      高精度的大型模型
                    </p>
                  </Upload.Dragger>
                  {teacherModel && (
                    <Alert
                      message={`Teacher: ${teacherModel.name}`}
                      type="success"
                      showIcon
                      style={{ marginTop: '8px' }}
                    />
                  )}
                </Form.Item>
              </Col>

              <Col span={12}>
                <Form.Item label="Student模型（学生模型）">
                  <Upload.Dragger
                    multiple={false}
                    showUploadList={false}
                    beforeUpload={() => false}
                    onChange={(info) => {
                      setStudentModel(info.file)
                      message.success(`已选择Student模型: ${info.file.name}`)
                    }}
                  >
                    <p className="ant-upload-drag-icon">
                      <BulbOutlined />
                    </p>
                    <p className="ant-upload-text">上传Student模型</p>
                    <p className="ant-upload-hint">
                      轻量级的小型模型
                    </p>
                  </Upload.Dragger>
                  {studentModel && (
                    <Alert
                      message={`Student: ${studentModel.name}`}
                      type="info"
                      showIcon
                      style={{ marginTop: '8px' }}
                    />
                  )}
                </Form.Item>
              </Col>
            </Row>
          </div>
        )}

        {currentStep === 1 && (
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="蒸馏策略" size="small">
                <Form.Item label="策略类型">
                  <Select
                    value={config.strategy}
                    onChange={(value) => setConfig({ ...config, strategy: value })}
                  >
                    {strategyOptions.map(option => (
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

                <Form.Item label={`温度参数 (${config.temperature})`}>
                  <Slider
                    min={1}
                    max={10}
                    step={0.1}
                    value={config.temperature}
                    onChange={(value) => setConfig({ ...config, temperature: value })}
                    marks={{
                      1: '1.0',
                      3: '3.0',
                      5: '5.0',
                      10: '10.0'
                    }}
                  />
                  <Text type="secondary" style={{ fontSize: '12px' }}>
                    温度越高，软标签越平滑
                  </Text>
                </Form.Item>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="损失权重" size="small">
                <Form.Item label={`蒸馏损失权重 α (${config.alpha})`}>
                  <Slider
                    min={0}
                    max={1}
                    step={0.1}
                    value={config.alpha}
                    onChange={(value) => setConfig({ ...config, alpha: value })}
                    marks={{
                      0: '0.0',
                      0.5: '0.5',
                      1: '1.0'
                    }}
                  />
                </Form.Item>

                <Form.Item label={`任务损失权重 β (${config.beta})`}>
                  <Slider
                    min={0}
                    max={1}
                    step={0.1}
                    value={config.beta}
                    onChange={(value) => setConfig({ ...config, beta: value })}
                    marks={{
                      0: '0.0',
                      0.5: '0.5',
                      1: '1.0'
                    }}
                  />
                </Form.Item>

                <Alert
                  message={`总权重: α + β = ${(config.alpha + config.beta).toFixed(1)}`}
                  type={config.alpha + config.beta === 1.0 ? 'success' : 'warning'}
                  showIcon
                  style={{ marginTop: '8px' }}
                />
              </Card>
            </Col>
          </Row>
        )}

        {currentStep === 2 && (
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="训练配置" size="small">
                <Form.Item label="训练轮次">
                  <InputNumber
                    min={1}
                    max={100}
                    value={config.numEpochs}
                    onChange={(value) => setConfig({ ...config, numEpochs: value || 10 })}
                    style={{ width: '100%' }}
                  />
                </Form.Item>

                <Form.Item label="学习率">
                  <InputNumber
                    min={0.00001}
                    max={0.01}
                    step={0.00001}
                    value={config.learningRate}
                    onChange={(value) => setConfig({ ...config, learningRate: value || 0.0001 })}
                    style={{ width: '100%' }}
                    formatter={value => `${value}e-4`}
                  />
                </Form.Item>

                <Form.Item label="批次大小">
                  <Select
                    value={config.batchSize}
                    onChange={(value) => setConfig({ ...config, batchSize: value })}
                  >
                    <Option value={16}>16</Option>
                    <Option value={32}>32</Option>
                    <Option value={64}>64</Option>
                    <Option value={128}>128</Option>
                  </Select>
                </Form.Item>
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="优化器配置" size="small">
                <Form.Item label="优化器">
                  <Select
                    value={config.optimizer}
                    onChange={(value) => setConfig({ ...config, optimizer: value })}
                  >
                    {optimizerOptions.map(option => (
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

                <Form.Item label="学习率调度器">
                  <Select
                    value={config.scheduler}
                    onChange={(value) => setConfig({ ...config, scheduler: value })}
                  >
                    <Option value="cosine">Cosine Annealing</Option>
                    <Option value="step">Step LR</Option>
                    <Option value="exponential">Exponential LR</Option>
                    <Option value="plateau">Reduce on Plateau</Option>
                  </Select>
                </Form.Item>

                <Form.Item>
                  <Switch
                    checked={config.earlyStopping}
                    onChange={(checked) => setConfig({ ...config, earlyStopping: checked })}
                  />
                  <span style={{ marginLeft: '8px' }}>启用早停</span>
                  {config.earlyStopping && (
                    <InputNumber
                      min={1}
                      max={10}
                      value={config.patience}
                      onChange={(value) => setConfig({ ...config, patience: value || 3 })}
                      addonBefore="耐心值"
                      style={{ marginTop: '8px', width: '100%' }}
                    />
                  )}
                </Form.Item>
              </Card>
            </Col>
          </Row>
        )}

        {currentStep === 3 && (
          <div>
            <Alert
              message="确认蒸馏配置"
              description="请检查以下配置信息，确认无误后点击开始蒸馏"
              type="info"
              showIcon
              style={{ marginBottom: '24px' }}
            />

            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <Card title="模型配置" size="small">
                  <List size="small">
                    <List.Item>
                      <Text type="secondary">Teacher模型:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {teacherModel?.name || 'bert-base-uncased'}
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">Student模型:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>
                        {studentModel?.name || 'distilbert-base-uncased'}
                      </Text>
                    </List.Item>
                    <List.Item>
                      <Text type="secondary">蒸馏策略:</Text>
                      <Tag color="orange" style={{ marginLeft: '8px' }}>
                        {strategyOptions.find(s => s.value === config.strategy)?.label}
                      </Tag>
                    </List.Item>
                  </List>
                </Card>
              </Col>

              <Col xs={24} lg={12}>
                <Card title="参数配置" size="small">
                  <Row gutter={[8, 8]}>
                    <Col span={12}>
                      <Text type="secondary">温度:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>{config.temperature}</Text>
                    </Col>
                    <Col span={12}>
                      <Text type="secondary">轮次:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>{config.numEpochs}</Text>
                    </Col>
                    <Col span={12}>
                      <Text type="secondary">学习率:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>{config.learningRate}</Text>
                    </Col>
                    <Col span={12}>
                      <Text type="secondary">批次:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>{config.batchSize}</Text>
                    </Col>
                    <Col span={12}>
                      <Text type="secondary">优化器:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>{config.optimizer.toUpperCase()}</Text>
                    </Col>
                    <Col span={12}>
                      <Text type="secondary">早停:</Text>
                      <Text strong style={{ marginLeft: '8px' }}>{config.earlyStopping ? '是' : '否'}</Text>
                    </Col>
                  </Row>
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
                disabled={currentStep === 0 && (!teacherModel || !studentModel)}
              >
                下一步
              </Button>
            ) : (
              <Button 
                type="primary" 
                loading={loading}
                onClick={handleStartDistillation}
                icon={<PlayCircleOutlined />}
              >
                开始蒸馏
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
          <ExperimentFilled style={{ marginRight: '12px', color: '#fa8c16' }} />
          知识蒸馏管理
        </Title>
        <Paragraph style={{ marginTop: '8px', color: '#666', fontSize: '16px' }}>
          通过Teacher-Student框架进行知识迁移，实现模型压缩和性能优化
        </Paragraph>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="创建任务" key="create">
          {renderCreateForm()}
        </TabPane>
        
        <TabPane tab="任务列表" key="jobs">
          <Card 
            title="蒸馏任务" 
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
        title="蒸馏任务详情"
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
                <Card title="Teacher模型" size="small">
                  <div style={{ textAlign: 'center' }}>
                    <Avatar size={48} icon={<FireOutlined />} style={{ backgroundColor: '#ff4d4f' }} />
                    <div style={{ marginTop: '8px' }}>
                      <Text strong>{selectedJob.teacherModel}</Text>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        准确率: {selectedJob.teacherAccuracy}%
                      </div>
                    </div>
                  </div>
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="蒸馏过程" size="small">
                  <div style={{ textAlign: 'center' }}>
                    <Progress
                      type="circle"
                      percent={selectedJob.progress}
                      width={60}
                    />
                    <div style={{ marginTop: '8px' }}>
                      <Text>{selectedJob.currentEpoch}/{selectedJob.totalEpochs} Epochs</Text>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        {selectedJob.config.strategy}
                      </div>
                    </div>
                  </div>
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="Student模型" size="small">
                  <div style={{ textAlign: 'center' }}>
                    <Avatar size={48} icon={<BulbOutlined />} style={{ backgroundColor: '#1890ff' }} />
                    <div style={{ marginTop: '8px' }}>
                      <Text strong>{selectedJob.studentModel}</Text>
                      <div style={{ fontSize: '12px', color: '#666' }}>
                        准确率: {selectedJob.studentAccuracy}%
                      </div>
                    </div>
                  </div>
                </Card>
              </Col>
            </Row>

            {selectedJob.status === 'running' && (
              <>
                <Divider />
                <Title level={4}>训练指标</Title>
                <Row gutter={[16, 16]}>
                  <Col span={6}>
                    <Statistic
                      title="训练损失"
                      value={selectedJob.trainingLoss}
                      precision={3}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="验证损失"
                      value={selectedJob.validationLoss}
                      precision={3}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="当前轮次"
                      value={selectedJob.currentEpoch}
                      suffix={`/ ${selectedJob.totalEpochs}`}
                    />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="压缩比"
                      value={selectedJob.compressionRatio}
                      suffix="x"
                      precision={1}
                    />
                  </Col>
                </Row>
              </>
            )}

            <Divider />
            <Title level={4}>训练日志</Title>
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

export default KnowledgeDistillationPage