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
  Spin,
  message,
  Statistic
} from 'antd'
import {
  CompressOutlined,
  UploadOutlined,
  PlayCircleOutlined,
  SettingOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined,
  DownloadOutlined,
  EyeOutlined,
  DeleteOutlined,
  EditOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Step } = Steps
const { Option } = Select

interface QuantizationConfig {
  method: 'ptq' | 'qat' | 'gptq' | 'awq'
  precision: 'int8' | 'int4' | 'fp16'
  calibrationSize: number
  preserveAccuracy: boolean
  targetAccuracyLoss: number
  optimizeForSpeed: boolean
  optimizeForSize: boolean
}

interface QuantizationJob {
  id: string
  name: string
  modelPath: string
  config: QuantizationConfig
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  originalSize: number
  compressedSize?: number
  compressionRatio?: number
  accuracyLoss?: number
  startTime: string
  endTime?: string
  logs: string[]
}

const QuantizationManagerPage: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('create')
  const [currentStep, setCurrentStep] = useState(0)
  const [selectedModel, setSelectedModel] = useState<any>(null)
  const [config, setConfig] = useState<QuantizationConfig>({
    method: 'ptq',
    precision: 'int8',
    calibrationSize: 512,
    preserveAccuracy: true,
    targetAccuracyLoss: 5.0,
    optimizeForSpeed: true,
    optimizeForSize: false
  })
  const [jobs, setJobs] = useState<QuantizationJob[]>([])
  const [jobModalVisible, setJobModalVisible] = useState(false)
  const [selectedJob, setSelectedJob] = useState<QuantizationJob | null>(null)

  useEffect(() => {
    // 模拟加载现有任务
    setJobs([
      {
        id: 'quant-001',
        name: 'BERT-Base-INT8',
        modelPath: '/models/bert-base-uncased.pt',
        config: {
          method: 'ptq',
          precision: 'int8',
          calibrationSize: 512,
          preserveAccuracy: true,
          targetAccuracyLoss: 3.0,
          optimizeForSpeed: true,
          optimizeForSize: false
        },
        status: 'completed',
        progress: 100,
        originalSize: 438,
        compressedSize: 110,
        compressionRatio: 4.0,
        accuracyLoss: 1.2,
        startTime: '2024-08-24 10:30:00',
        endTime: '2024-08-24 10:45:00',
        logs: [
          '开始量化任务...',
          '加载模型: BERT-Base',
          '准备校准数据集...',
          '执行后训练量化...',
          '量化完成，压缩比: 4.0x',
          '精度评估: 损失 1.2%',
          '保存量化模型...',
          '任务完成！'
        ]
      },
      {
        id: 'quant-002',
        name: 'ResNet50-INT4',
        modelPath: '/models/resnet50.pt',
        config: {
          method: 'gptq',
          precision: 'int4',
          calibrationSize: 1024,
          preserveAccuracy: false,
          targetAccuracyLoss: 8.0,
          optimizeForSpeed: false,
          optimizeForSize: true
        },
        status: 'running',
        progress: 67,
        originalSize: 102,
        startTime: '2024-08-24 11:15:00',
        logs: [
          '开始GPTQ量化任务...',
          '加载ResNet50模型...',
          '准备校准数据集 (1024 samples)...',
          '执行GPTQ算法...',
          '当前进度: 67%'
        ]
      }
    ])
  }, [])

  const methodOptions = [
    { value: 'ptq', label: '后训练量化 (PTQ)', description: '快速量化，无需重训练' },
    { value: 'qat', label: '量化感知训练 (QAT)', description: '精度最高，需要训练数据' },
    { value: 'gptq', label: 'GPTQ算法', description: '适用于大型语言模型' },
    { value: 'awq', label: 'AWQ算法', description: '激活加权量化' }
  ]

  const precisionOptions = [
    { value: 'int8', label: 'INT8', description: '8位整数，2-4x压缩' },
    { value: 'int4', label: 'INT4', description: '4位整数，4-8x压缩' },
    { value: 'fp16', label: 'FP16', description: '16位浮点，2x压缩' }
  ]

  const handleStartQuantization = async () => {
    try {
      const values = await form.validateFields()
      setLoading(true)

      // 创建新的量化任务
      const newJob: QuantizationJob = {
        id: `quant-${Date.now()}`,
        name: values.jobName,
        modelPath: values.modelPath || selectedModel?.name,
        config,
        status: 'pending',
        progress: 0,
        originalSize: Math.round(Math.random() * 500 + 100),
        startTime: new Date().toLocaleString(),
        logs: ['任务已创建，等待开始...']
      }

      setJobs(prev => [newJob, ...prev])
      message.success('量化任务创建成功！')
      
      // 模拟任务启动
      setTimeout(() => {
        setJobs(prev => prev.map(job => 
          job.id === newJob.id 
            ? { ...job, status: 'running' as const, logs: [...job.logs, '开始量化处理...'] }
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

  const getStatusColor = (status: QuantizationJob['status']) => {
    switch (status) {
      case 'pending': return '#faad14'
      case 'running': return '#1890ff'
      case 'completed': return '#52c41a'
      case 'failed': return '#ff4d4f'
      default: return '#d9d9d9'
    }
  }

  const getStatusText = (status: QuantizationJob['status']) => {
    switch (status) {
      case 'pending': return '等待中'
      case 'running': return '运行中'
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
      render: (text: string) => <Text strong>{text}</Text>
    },
    {
      title: '量化方法',
      key: 'method',
      render: (record: QuantizationJob) => (
        <Tag color="blue">
          {record.config.method.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '精度',
      key: 'precision',
      render: (record: QuantizationJob) => (
        <Tag color={record.config.precision === 'int4' ? 'red' : 
                   record.config.precision === 'int8' ? 'orange' : 'green'}>
          {record.config.precision.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '状态',
      key: 'status',
      render: (record: QuantizationJob) => (
        <Tag color={getStatusColor(record.status)}>
          {getStatusText(record.status)}
        </Tag>
      )
    },
    {
      title: '进度',
      key: 'progress',
      render: (record: QuantizationJob) => (
        <div style={{ width: 100 }}>
          <Progress 
            percent={record.progress} 
            size="small"
            status={record.status === 'failed' ? 'exception' : undefined}
          />
        </div>
      )
    },
    {
      title: '压缩比',
      key: 'compressionRatio',
      render: (record: QuantizationJob) => (
        record.compressionRatio ? 
          <Text type="success" strong>{record.compressionRatio}x</Text> : 
          <Text type="secondary">-</Text>
      )
    },
    {
      title: '精度损失',
      key: 'accuracyLoss',
      render: (record: QuantizationJob) => (
        record.accuracyLoss !== undefined ? 
          <Text style={{ color: record.accuracyLoss > 5 ? '#ff4d4f' : '#52c41a' }}>
            {record.accuracyLoss}%
          </Text> : 
          <Text type="secondary">-</Text>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: QuantizationJob) => (
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
    <Card title="创建量化任务">
      <Steps current={currentStep} style={{ marginBottom: '24px' }}>
        <Step title="选择模型" icon={<UploadOutlined />} />
        <Step title="配置量化" icon={<SettingOutlined />} />
        <Step title="确认并启动" icon={<PlayCircleOutlined />} />
      </Steps>

      <Form form={form} layout="vertical">
        {currentStep === 0 && (
          <div>
            <Form.Item
              name="jobName"
              label="任务名称"
              rules={[{ required: true, message: '请输入任务名称' }]}
            >
              <Input placeholder="例如: BERT-Base-INT8量化" />
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
                  <UploadOutlined />
                </p>
                <p className="ant-upload-text">点击或拖拽上传模型文件</p>
                <p className="ant-upload-hint">
                  支持 .pt, .pth, .onnx, .safetensors 格式
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
              <Card title="量化方法" size="small">
                <Form.Item name="method" label="量化算法">
                  <Select
                    value={config.method}
                    onChange={(value) => setConfig({ ...config, method: value })}
                  >
                    {methodOptions.map(option => (
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

                <Form.Item name="precision" label="量化精度">
                  <Select
                    value={config.precision}
                    onChange={(value) => setConfig({ ...config, precision: value })}
                  >
                    {precisionOptions.map(option => (
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
              </Card>
            </Col>

            <Col xs={24} lg={12}>
              <Card title="量化参数" size="small">
                <Form.Item name="calibrationSize" label="校准数据集大小">
                  <InputNumber
                    min={100}
                    max={5000}
                    value={config.calibrationSize}
                    onChange={(value) => setConfig({ ...config, calibrationSize: value || 512 })}
                    style={{ width: '100%' }}
                  />
                </Form.Item>

                <Form.Item name="targetAccuracyLoss" label="目标精度损失 (%)">
                  <InputNumber
                    min={0}
                    max={20}
                    step={0.1}
                    value={config.targetAccuracyLoss}
                    onChange={(value) => setConfig({ ...config, targetAccuracyLoss: value || 5.0 })}
                    style={{ width: '100%' }}
                  />
                </Form.Item>

                <Form.Item name="preserveAccuracy" valuePropName="checked">
                  <Switch
                    checked={config.preserveAccuracy}
                    onChange={(checked) => setConfig({ ...config, preserveAccuracy: checked })}
                  />
                  <span style={{ marginLeft: '8px' }}>保持精度优先</span>
                </Form.Item>
              </Card>
            </Col>

            <Col xs={24}>
              <Card title="优化目标" size="small">
                <Row gutter={16}>
                  <Col span={12}>
                    <Form.Item name="optimizeForSpeed" valuePropName="checked">
                      <Switch
                        checked={config.optimizeForSpeed}
                        onChange={(checked) => setConfig({ ...config, optimizeForSpeed: checked })}
                      />
                      <span style={{ marginLeft: '8px' }}>优化推理速度</span>
                    </Form.Item>
                  </Col>
                  <Col span={12}>
                    <Form.Item name="optimizeForSize" valuePropName="checked">
                      <Switch
                        checked={config.optimizeForSize}
                        onChange={(checked) => setConfig({ ...config, optimizeForSize: checked })}
                      />
                      <span style={{ marginLeft: '8px' }}>优化模型大小</span>
                    </Form.Item>
                  </Col>
                </Row>
              </Card>
            </Col>
          </Row>
        )}

        {currentStep === 2 && (
          <div>
            <Alert
              message="确认量化配置"
              description="请检查以下配置信息，确认无误后点击开始量化"
              type="info"
              showIcon
              style={{ marginBottom: '24px' }}
            />

            <Card title="配置摘要" size="small">
              <Row gutter={[16, 16]}>
                <Col span={8}>
                  <Text type="secondary">量化方法:</Text>
                  <div><Text strong>{config.method.toUpperCase()}</Text></div>
                </Col>
                <Col span={8}>
                  <Text type="secondary">量化精度:</Text>
                  <div><Text strong>{config.precision.toUpperCase()}</Text></div>
                </Col>
                <Col span={8}>
                  <Text type="secondary">校准数据:</Text>
                  <div><Text strong>{config.calibrationSize} 样本</Text></div>
                </Col>
                <Col span={8}>
                  <Text type="secondary">精度损失阈值:</Text>
                  <div><Text strong>{config.targetAccuracyLoss}%</Text></div>
                </Col>
                <Col span={8}>
                  <Text type="secondary">优化目标:</Text>
                  <div>
                    <Text strong>
                      {config.optimizeForSpeed && config.optimizeForSize ? '速度+大小' :
                       config.optimizeForSpeed ? '速度' :
                       config.optimizeForSize ? '大小' : '平衡'}
                    </Text>
                  </div>
                </Col>
                <Col span={8}>
                  <Text type="secondary">保持精度:</Text>
                  <div><Text strong>{config.preserveAccuracy ? '是' : '否'}</Text></div>
                </Col>
              </Row>
            </Card>
          </div>
        )}

        <div style={{ marginTop: '24px', textAlign: 'right' }}>
          <Space>
            {currentStep > 0 && (
              <Button onClick={() => setCurrentStep(currentStep - 1)}>
                上一步
              </Button>
            )}
            {currentStep < 2 ? (
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
                onClick={handleStartQuantization}
                icon={<PlayCircleOutlined />}
              >
                开始量化
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
          <CompressOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          量化管理中心
        </Title>
        <Paragraph style={{ marginTop: '8px', color: '#666', fontSize: '16px' }}>
          创建和管理模型量化任务，支持多种量化算法和精度配置
        </Paragraph>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="创建任务" key="create">
          {renderCreateForm()}
        </TabPane>
        
        <TabPane tab="任务列表" key="jobs">
          <Card 
            title="量化任务" 
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
        title="任务详情"
        open={jobModalVisible}
        onCancel={() => setJobModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setJobModalVisible(false)}>
            关闭
          </Button>
        ]}
        width={800}
      >
        {selectedJob && (
          <div>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Text type="secondary">任务名称:</Text>
                <div><Text strong>{selectedJob.name}</Text></div>
              </Col>
              <Col span={12}>
                <Text type="secondary">状态:</Text>
                <div>
                  <Tag color={getStatusColor(selectedJob.status)}>
                    {getStatusText(selectedJob.status)}
                  </Tag>
                </div>
              </Col>
              <Col span={12}>
                <Text type="secondary">开始时间:</Text>
                <div><Text>{selectedJob.startTime}</Text></div>
              </Col>
              <Col span={12}>
                <Text type="secondary">结束时间:</Text>
                <div><Text>{selectedJob.endTime || '-'}</Text></div>
              </Col>
            </Row>

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

            {selectedJob.status === 'completed' && (
              <>
                <Divider />
                <Title level={4}>压缩结果</Title>
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Statistic
                      title="原始大小"
                      value={selectedJob.originalSize}
                      suffix="MB"
                    />
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="压缩后大小"
                      value={selectedJob.compressedSize}
                      suffix="MB"
                    />
                  </Col>
                  <Col span={8}>
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
          </div>
        )}
      </Modal>
    </div>
  )
}

export default QuantizationManagerPage