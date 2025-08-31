import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Space,
  Typography,
  Tabs,
  Modal,
  Alert,
  Divider,
  Tag,
  Progress,
  Statistic,
  Timeline,
  List,
  Avatar,
  message,
  Tooltip,
  Drawer,
  Badge,
  Empty,
  Select
} from 'antd'
import {
  BranchesOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  EyeOutlined,
  SettingOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  SyncOutlined,
  DeleteOutlined,
  CompressOutlined,
  ScissorOutlined,
  ExperimentFilled,
  ThunderboltOutlined,
  LineChartOutlined,
  DatabaseOutlined,
  NodeIndexOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Option } = Select

interface PipelineJob {
  id: string
  name: string
  type: 'quantization' | 'pruning' | 'distillation' | 'mixed'
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused'
  priority: number
  progress: number
  currentStage: string
  stages: string[]
  completedStages: string[]
  estimatedTime: number
  elapsedTime: number
  startTime: string
  endTime?: string
  config: any
  results?: any
  logs: string[]
  dependencies: string[]
  resources: {
    cpu: number
    memory: number
    gpu: number
  }
}

interface PipelineStats {
  totalJobs: number
  runningJobs: number
  completedJobs: number
  failedJobs: number
  avgExecutionTime: number
  systemLoad: {
    cpu: number
    memory: number
    gpu: number
  }
  throughput: number
}

const CompressionPipelinePage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview')
  const [jobs, setJobs] = useState<PipelineJob[]>([])
  const [stats, setStats] = useState<PipelineStats | null>(null)
  const [selectedJob, setSelectedJob] = useState<PipelineJob | null>(null)
  const [jobDetailVisible, setJobDetailVisible] = useState(false)
  const [logDrawerVisible, setLogDrawerVisible] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [refreshInterval, setRefreshInterval] = useState(5000)

  useEffect(() => {
    // 模拟数据加载
    loadPipelineData()
    
    // 自动刷新
    const interval = autoRefresh ? setInterval(loadPipelineData, refreshInterval) : null
    
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [autoRefresh, refreshInterval])

  const loadPipelineData = () => {
    // 模拟流水线数据
    setStats({
      totalJobs: 45,
      runningJobs: 8,
      completedJobs: 32,
      failedJobs: 5,
      avgExecutionTime: 1.5,
      systemLoad: {
        cpu: 65,
        memory: 72,
        gpu: 88
      },
      throughput: 12.5
    })

    setJobs([
      {
        id: 'pipeline-001',
        name: 'BERT多策略压缩流水线',
        type: 'mixed',
        status: 'running',
        priority: 1,
        progress: 45,
        currentStage: '知识蒸馏',
        stages: ['量化预处理', '知识蒸馏', '结构化剪枝', '后处理优化'],
        completedStages: ['量化预处理'],
        estimatedTime: 120,
        elapsedTime: 54,
        startTime: '2024-08-24 10:30:00',
        config: {
          quantization: { precision: 'int8', method: 'ptq' },
          distillation: { temperature: 3.0, alpha: 0.5 },
          pruning: { sparsity: 0.3, type: 'structured' }
        },
        logs: [
          '流水线启动...',
          '阶段1: 量化预处理 - 开始',
          '加载BERT-Base模型...',
          '量化预处理完成',
          '阶段2: 知识蒸馏 - 开始',
          '初始化Teacher-Student架构...',
          'Epoch 3/10: Loss=0.234'
        ],
        dependencies: [],
        resources: { cpu: 45, memory: 60, gpu: 80 }
      },
      {
        id: 'pipeline-002',
        name: 'ResNet50剪枝流水线',
        type: 'pruning',
        status: 'completed',
        priority: 2,
        progress: 100,
        currentStage: '完成',
        stages: ['敏感性分析', '结构化剪枝', '精度恢复'],
        completedStages: ['敏感性分析', '结构化剪枝', '精度恢复'],
        estimatedTime: 90,
        elapsedTime: 87,
        startTime: '2024-08-24 08:15:00',
        endTime: '2024-08-24 09:42:00',
        config: {
          pruning: { sparsity: 0.5, type: 'structured', metric: 'l2_norm' }
        },
        results: {
          originalSize: 102,
          compressedSize: 51,
          compressionRatio: 2.0,
          accuracyLoss: 1.2
        },
        logs: [
          '流水线启动...',
          '阶段1: 敏感性分析 - 完成',
          '阶段2: 结构化剪枝 - 完成',
          '阶段3: 精度恢复 - 完成',
          '流水线执行成功'
        ],
        dependencies: [],
        resources: { cpu: 30, memory: 40, gpu: 65 }
      },
      {
        id: 'pipeline-003',
        name: 'GPT-2量化流水线',
        type: 'quantization',
        status: 'pending',
        priority: 3,
        progress: 0,
        currentStage: '等待中',
        stages: ['数据准备', 'INT8量化', '精度验证'],
        completedStages: [],
        estimatedTime: 180,
        elapsedTime: 0,
        startTime: '2024-08-24 12:00:00',
        config: {
          quantization: { precision: 'int8', method: 'qat', epochs: 5 }
        },
        logs: ['任务已排队，等待资源分配...'],
        dependencies: ['pipeline-001'],
        resources: { cpu: 0, memory: 0, gpu: 0 }
      },
      {
        id: 'pipeline-004',
        name: 'MobileNet蒸馏流水线',
        type: 'distillation',
        status: 'failed',
        priority: 2,
        progress: 25,
        currentStage: '失败',
        stages: ['模型加载', '蒸馏训练', '结果验证'],
        completedStages: ['模型加载'],
        estimatedTime: 60,
        elapsedTime: 15,
        startTime: '2024-08-24 11:45:00',
        endTime: '2024-08-24 12:00:00',
        config: {
          distillation: { temperature: 4.0, alpha: 0.3, epochs: 15 }
        },
        logs: [
          '流水线启动...',
          '阶段1: 模型加载 - 完成',
          '阶段2: 蒸馏训练 - 开始',
          'Epoch 2/15: Loss=1.234',
          'ERROR: GPU内存不足',
          '流水线执行失败'
        ],
        dependencies: [],
        resources: { cpu: 20, memory: 30, gpu: 95 }
      }
    ])
  }

  const getStatusColor = (status: PipelineJob['status']) => {
    switch (status) {
      case 'pending': return '#faad14'
      case 'running': return '#1890ff'
      case 'completed': return '#52c41a'
      case 'failed': return '#ff4d4f'
      case 'paused': return '#722ed1'
      default: return '#d9d9d9'
    }
  }

  const getStatusText = (status: PipelineJob['status']) => {
    switch (status) {
      case 'pending': return '等待中'
      case 'running': return '运行中'
      case 'completed': return '已完成'
      case 'failed': return '失败'
      case 'paused': return '已暂停'
      default: return '未知'
    }
  }

  const getStatusIcon = (status: PipelineJob['status']) => {
    switch (status) {
      case 'pending': return <ClockCircleOutlined />
      case 'running': return <SyncOutlined spin />
      case 'completed': return <CheckCircleOutlined />
      case 'failed': return <ExclamationCircleOutlined />
      case 'paused': return <PauseCircleOutlined />
      default: return null
    }
  }

  const getTypeIcon = (type: PipelineJob['type']) => {
    switch (type) {
      case 'quantization': return <CompressOutlined style={{ color: '#1890ff' }} />
      case 'pruning': return <ScissorOutlined style={{ color: '#722ed1' }} />
      case 'distillation': return <ExperimentFilled style={{ color: '#fa8c16' }} />
      case 'mixed': return <ThunderboltOutlined style={{ color: '#52c41a' }} />
      default: return null
    }
  }

  const getTypeText = (type: PipelineJob['type']) => {
    switch (type) {
      case 'quantization': return '量化'
      case 'pruning': return '剪枝'
      case 'distillation': return '蒸馏'
      case 'mixed': return '混合'
      default: return '未知'
    }
  }

  const handleJobAction = (job: PipelineJob, action: string) => {
    switch (action) {
      case 'start':
        message.success(`启动任务: ${job.name}`)
        break
      case 'pause':
        message.info(`暂停任务: ${job.name}`)
        break
      case 'stop':
        message.warning(`停止任务: ${job.name}`)
        break
      case 'restart':
        message.info(`重启任务: ${job.name}`)
        break
      case 'delete':
        message.error(`删除任务: ${job.name}`)
        break
    }
    // 更新任务状态的逻辑在这里实现
  }

  const pipelineColumns = [
    {
      title: '任务信息',
      key: 'info',
      render: (record: PipelineJob) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            {getTypeIcon(record.type)}
            <Text strong style={{ marginLeft: '8px' }}>{record.name}</Text>
            <Tag
              color={record.priority === 1 ? 'red' : record.priority === 2 ? 'orange' : 'default'}
              size="small"
              style={{ marginLeft: '8px' }}
            >
              P{record.priority}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.id} • {getTypeText(record.type)}
          </Text>
        </div>
      )
    },
    {
      title: '状态',
      key: 'status',
      render: (record: PipelineJob) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Badge 
              status={record.status === 'running' ? 'processing' : 
                     record.status === 'completed' ? 'success' :
                     record.status === 'failed' ? 'error' : 'default'}
            />
            <Text style={{ marginLeft: '4px' }}>{getStatusText(record.status)}</Text>
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.currentStage}
          </Text>
        </div>
      )
    },
    {
      title: '执行进度',
      key: 'progress',
      render: (record: PipelineJob) => (
        <div style={{ width: '120px' }}>
          <Progress
            percent={record.progress}
            size="small"
            status={record.status === 'failed' ? 'exception' : undefined}
          />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.completedStages.length}/{record.stages.length} 阶段
          </Text>
        </div>
      )
    },
    {
      title: '资源使用',
      key: 'resources',
      render: (record: PipelineJob) => (
        <div style={{ fontSize: '12px' }}>
          <div>CPU: {record.resources.cpu}%</div>
          <div>内存: {record.resources.memory}%</div>
          <div>GPU: {record.resources.gpu}%</div>
        </div>
      )
    },
    {
      title: '时间信息',
      key: 'time',
      render: (record: PipelineJob) => (
        <div style={{ fontSize: '12px' }}>
          <div>已用: {record.elapsedTime}min</div>
          <div>预计: {record.estimatedTime}min</div>
          {record.endTime && (
            <div>完成: {record.endTime.split(' ')[1]}</div>
          )}
        </div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: PipelineJob) => (
        <Space direction="vertical" size="small">
          <div>
            <Button
              type="link"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedJob(record)
                setJobDetailVisible(true)
              }}
            >
              详情
            </Button>
          </div>
          <Space size="small">
            {record.status === 'pending' && (
              <Button
                type="link"
                size="small"
                icon={<PlayCircleOutlined />}
                onClick={() => handleJobAction(record, 'start')}
              >
                启动
              </Button>
            )}
            {record.status === 'running' && (
              <>
                <Button
                  type="link"
                  size="small"
                  icon={<PauseCircleOutlined />}
                  onClick={() => handleJobAction(record, 'pause')}
                >
                  暂停
                </Button>
                <Button
                  type="link"
                  size="small"
                  danger
                  icon={<StopOutlined />}
                  onClick={() => handleJobAction(record, 'stop')}
                >
                  停止
                </Button>
              </>
            )}
            {(record.status === 'paused' || record.status === 'failed') && (
              <Button
                type="link"
                size="small"
                icon={<ReloadOutlined />}
                onClick={() => handleJobAction(record, 'restart')}
              >
                重启
              </Button>
            )}
          </Space>
        </Space>
      )
    }
  ]

  const renderOverview = () => (
    <div>
      {/* 系统统计 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="总任务数"
              value={stats?.totalJobs}
              prefix={<DatabaseOutlined style={{ color: '#1890ff' }} />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="运行中"
              value={stats?.runningJobs}
              prefix={<SyncOutlined style={{ color: '#faad14' }} />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="平均执行时间"
              value={stats?.avgExecutionTime}
              suffix="小时"
              precision={1}
              prefix={<ClockCircleOutlined style={{ color: '#722ed1' }} />}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="吞吐量"
              value={stats?.throughput}
              suffix="任务/天"
              precision={1}
              prefix={<ThunderboltOutlined style={{ color: '#52c41a' }} />}
            />
          </Card>
        </Col>
      </Row>

      {/* 系统负载 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col xs={24} lg={12}>
          <Card title="系统负载" size="small">
            <div style={{ marginBottom: '16px' }}>
              <Text type="secondary">CPU使用率</Text>
              <Progress percent={stats?.systemLoad.cpu} />
            </div>
            <div style={{ marginBottom: '16px' }}>
              <Text type="secondary">内存使用率</Text>
              <Progress percent={stats?.systemLoad.memory} />
            </div>
            <div>
              <Text type="secondary">GPU使用率</Text>
              <Progress percent={stats?.systemLoad.gpu} />
            </div>
          </Card>
        </Col>
        
        <Col xs={24} lg={12}>
          <Card title="任务状态分布" size="small">
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="已完成"
                  value={stats?.completedJobs}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="失败"
                  value={stats?.failedJobs}
                  valueStyle={{ color: '#ff4d4f' }}
                />
              </Col>
            </Row>
            
            <div style={{ marginTop: '16px' }}>
              <Progress
                percent={stats ? (stats.completedJobs / stats.totalJobs * 100) : 0}
                success={{ percent: stats ? (stats.completedJobs / stats.totalJobs * 100) : 0 }}
                type="line"
                showInfo={false}
              />
              <Text type="secondary" style={{ fontSize: '12px' }}>
                成功率: {stats ? ((stats.completedJobs / stats.totalJobs) * 100).toFixed(1) : 0}%
              </Text>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 正在运行的任务 */}
      <Card title="运行中的任务" size="small">
        {jobs.filter(job => job.status === 'running').length > 0 ? (
          <List
            dataSource={jobs.filter(job => job.status === 'running')}
            renderItem={(job) => (
              <List.Item
                actions={[
                  <Button
                    type="link"
                    size="small"
                    onClick={() => {
                      setSelectedJob(job)
                      setLogDrawerVisible(true)
                    }}
                  >
                    查看日志
                  </Button>
                ]}
              >
                <List.Item.Meta
                  avatar={<Avatar icon={getTypeIcon(job.type)} />}
                  title={
                    <div style={{ display: 'flex', alignItems: 'center' }}>
                      <Text strong>{job.name}</Text>
                      <Progress
                        percent={job.progress}
                        size="small"
                        style={{ marginLeft: '12px', width: '100px' }}
                      />
                    </div>
                  }
                  description={`当前阶段: ${job.currentStage} | 已用时间: ${job.elapsedTime}分钟`}
                />
              </List.Item>
            )}
          />
        ) : (
          <Empty description="暂无运行中的任务" />
        )}
      </Card>
    </div>
  )

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Title level={2} style={{ margin: 0, color: '#1a1a1a' }}>
              <BranchesOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
              压缩流水线管理
            </Title>
            <Paragraph style={{ marginTop: '8px', color: '#666', fontSize: '16px' }}>
              管理和监控模型压缩流水线，支持多阶段任务编排和资源调度
            </Paragraph>
          </div>
          
          <Space>
            <Select
              value={refreshInterval}
              onChange={setRefreshInterval}
              style={{ width: 120 }}
            >
              <Option value={2000}>2秒</Option>
              <Option value={5000}>5秒</Option>
              <Option value={10000}>10秒</Option>
              <Option value={30000}>30秒</Option>
            </Select>
            <Button
              type={autoRefresh ? 'primary' : 'default'}
              icon={<ReloadOutlined />}
              onClick={() => setAutoRefresh(!autoRefresh)}
            >
              {autoRefresh ? '停止刷新' : '自动刷新'}
            </Button>
          </Space>
        </div>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="总览" key="overview">
          {renderOverview()}
        </TabPane>
        
        <TabPane tab="所有任务" key="jobs">
          <Card>
            <Table
              columns={pipelineColumns}
              dataSource={jobs}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              size="middle"
            />
          </Card>
        </TabPane>
        
        <TabPane tab="任务依赖图" key="dependency">
          <Card title="任务依赖关系">
            <div style={{ textAlign: 'center', padding: '40px' }}>
              <NodeIndexOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
              <div style={{ marginTop: '16px' }}>
                <Text type="secondary">任务依赖图可视化功能开发中...</Text>
              </div>
            </div>
          </Card>
        </TabPane>
      </Tabs>

      {/* 任务详情模态框 */}
      <Modal
        title="任务详情"
        open={jobDetailVisible}
        onCancel={() => setJobDetailVisible(false)}
        footer={null}
        width={800}
      >
        {selectedJob && (
          <div>
            <Row gutter={[16, 16]} style={{ marginBottom: '16px' }}>
              <Col span={8}>
                <Card title="基本信息" size="small">
                  <div style={{ marginBottom: '8px' }}>
                    <Text type="secondary">任务名称:</Text>
                    <div><Text strong>{selectedJob.name}</Text></div>
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    <Text type="secondary">任务类型:</Text>
                    <div>
                      <Space>
                        {getTypeIcon(selectedJob.type)}
                        <Text>{getTypeText(selectedJob.type)}</Text>
                      </Space>
                    </div>
                  </div>
                  <div>
                    <Text type="secondary">优先级:</Text>
                    <div>
                      <Tag color={selectedJob.priority === 1 ? 'red' : selectedJob.priority === 2 ? 'orange' : 'default'}>
                        P{selectedJob.priority}
                      </Tag>
                    </div>
                  </div>
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="执行状态" size="small">
                  <div style={{ marginBottom: '8px' }}>
                    <Text type="secondary">当前状态:</Text>
                    <div>
                      <Badge 
                        status={selectedJob.status === 'running' ? 'processing' : 
                               selectedJob.status === 'completed' ? 'success' :
                               selectedJob.status === 'failed' ? 'error' : 'default'}
                      />
                      <Text style={{ marginLeft: '4px' }}>{getStatusText(selectedJob.status)}</Text>
                    </div>
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    <Text type="secondary">当前阶段:</Text>
                    <div><Text strong>{selectedJob.currentStage}</Text></div>
                  </div>
                  <div>
                    <Text type="secondary">整体进度:</Text>
                    <div style={{ marginTop: '8px' }}>
                      <Progress percent={selectedJob.progress} />
                    </div>
                  </div>
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="时间信息" size="small">
                  <div style={{ marginBottom: '8px' }}>
                    <Text type="secondary">开始时间:</Text>
                    <div><Text>{selectedJob.startTime}</Text></div>
                  </div>
                  <div style={{ marginBottom: '8px' }}>
                    <Text type="secondary">已用时间:</Text>
                    <div><Text strong>{selectedJob.elapsedTime}分钟</Text></div>
                  </div>
                  <div>
                    <Text type="secondary">预计时间:</Text>
                    <div><Text>{selectedJob.estimatedTime}分钟</Text></div>
                  </div>
                </Card>
              </Col>
            </Row>

            <Card title="执行阶段" size="small" style={{ marginBottom: '16px' }}>
              <Timeline>
                {selectedJob.stages.map((stage, index) => (
                  <Timeline.Item
                    key={index}
                    color={
                      selectedJob.completedStages.includes(stage) ? 'green' :
                      selectedJob.currentStage === stage ? 'blue' : 'gray'
                    }
                    dot={
                      selectedJob.completedStages.includes(stage) ? <CheckCircleOutlined /> :
                      selectedJob.currentStage === stage ? <SyncOutlined spin /> :
                      <ClockCircleOutlined />
                    }
                  >
                    <Text
                      strong={selectedJob.currentStage === stage}
                      type={selectedJob.completedStages.includes(stage) ? 'success' : 
                           selectedJob.currentStage === stage ? 'default' : 'secondary'}
                    >
                      {stage}
                    </Text>
                    {selectedJob.currentStage === stage && (
                      <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
                        正在执行中...
                      </div>
                    )}
                  </Timeline.Item>
                ))}
              </Timeline>
            </Card>

            <Card title="资源使用" size="small">
              <Row gutter={16}>
                <Col span={8}>
                  <Text type="secondary">CPU:</Text>
                  <Progress percent={selectedJob.resources.cpu} size="small" />
                </Col>
                <Col span={8}>
                  <Text type="secondary">内存:</Text>
                  <Progress percent={selectedJob.resources.memory} size="small" />
                </Col>
                <Col span={8}>
                  <Text type="secondary">GPU:</Text>
                  <Progress percent={selectedJob.resources.gpu} size="small" />
                </Col>
              </Row>
            </Card>

            {selectedJob.results && (
              <>
                <Divider />
                <Card title="执行结果" size="small">
                  <Row gutter={16}>
                    <Col span={6}>
                      <Statistic
                        title="原始大小"
                        value={selectedJob.results.originalSize}
                        suffix="MB"
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="压缩后大小"
                        value={selectedJob.results.compressedSize}
                        suffix="MB"
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="压缩比"
                        value={selectedJob.results.compressionRatio}
                        suffix="x"
                        precision={1}
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="精度损失"
                        value={selectedJob.results.accuracyLoss}
                        suffix="%"
                        precision={1}
                      />
                    </Col>
                  </Row>
                </Card>
              </>
            )}
          </div>
        )}
      </Modal>

      {/* 日志抽屉 */}
      <Drawer
        title="执行日志"
        placement="right"
        width={600}
        onClose={() => setLogDrawerVisible(false)}
        open={logDrawerVisible}
      >
        {selectedJob && (
          <div>
            <div style={{ marginBottom: '16px' }}>
              <Text strong>{selectedJob.name}</Text>
              <Tag color={getStatusColor(selectedJob.status)} style={{ marginLeft: '8px' }}>
                {getStatusText(selectedJob.status)}
              </Tag>
            </div>
            
            <div style={{ 
              background: '#f5f5f5', 
              padding: '12px', 
              borderRadius: '4px',
              maxHeight: '500px',
              overflowY: 'auto',
              fontFamily: 'monospace',
              fontSize: '12px'
            }}>
              {selectedJob.logs.map((log, index) => (
                <div key={index} style={{ margin: '4px 0' }}>
                  <Text type="secondary">{new Date().toLocaleTimeString()}</Text>
                  <span style={{ margin: '0 8px' }}>|</span>
                  <Text>{log}</Text>
                </div>
              ))}
            </div>
          </div>
        )}
      </Drawer>
    </div>
  )
}

export default CompressionPipelinePage