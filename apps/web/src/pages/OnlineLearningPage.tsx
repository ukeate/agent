import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  Progress,
  Statistic,
  Tabs,
  Timeline,
  Badge,
  Space,
  Alert,
  Descriptions,
  Tag,
  Tooltip,
  Popconfirm,
  Switch,
  Slider,
  message,
  List,
  Avatar,
  Divider,
  Empty,
} from 'antd'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import {
  ExperimentOutlined,
  RocketOutlined,
  StopOutlined,
  PlayCircleOutlined,
  BarChartOutlined,
  ReloadOutlined,
  TrophyOutlined,
  BulbOutlined,
  FireOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons'

const { TabPane } = Tabs
const { Option } = Select
const { TextArea } = Input

interface ABTest {
  id: string
  name: string
  description: string
  model_a_id: string
  model_a_name: string
  model_b_id: string
  model_b_name: string
  traffic_split: number
  status: 'pending' | 'running' | 'completed' | 'paused'
  start_time: string
  end_time?: string
  total_requests: number
  model_a_requests: number
  model_b_requests: number
  model_a_performance: {
    accuracy: number
    latency: number
    error_rate: number
  }
  model_b_performance: {
    accuracy: number
    latency: number
    error_rate: number
  }
  statistical_significance: number
  winner?: 'A' | 'B' | 'tie'
}

interface LearningJob {
  id: string
  name: string
  model_id: string
  model_name: string
  learning_type: 'lora' | 'qlora' | 'full' | 'prefix' | 'p_tuning'
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  batch_size: number
  start_time: string
  estimated_completion: string
  metrics: {
    current_loss: number
    best_loss: number
  }
}

interface FeedbackData {
  id: string
  model_id: string
  prediction: any
  ground_truth: any
  feedback_score: number
  user_feedback: string
  timestamp: string
  processed: boolean
}

const OnlineLearningPage: React.FC = () => {
  const [abTests, setAbTests] = useState<ABTest[]>([])
  const [learningJobs, setLearningJobs] = useState<LearningJob[]>([])
  const [feedbackData, setFeedbackData] = useState<FeedbackData[]>([])
  const [loading, setLoading] = useState(false)
  const [testModalVisible, setTestModalVisible] = useState(false)
  const [jobModalVisible, setJobModalVisible] = useState(false)
  const [selectedTest, setSelectedTest] = useState<ABTest | null>(null)
  const [testForm] = Form.useForm()
  const [jobForm] = Form.useForm()

  const systemStats = {
    active_tests: abTests.filter(t => t.status === 'running').length,
    active_jobs: learningJobs.filter(j => j.status === 'running').length,
    total_feedback: feedbackData.length,
    avg_loss:
      learningJobs.length > 0
        ? learningJobs.reduce(
            (sum, job) => sum + (job.metrics?.current_loss || 0),
            0
          ) / learningJobs.length
        : 0,
  }

  const performanceTrendData = learningJobs.map(job => ({
    time: job.start_time
      ? new Date(job.start_time).toLocaleDateString()
      : job.name,
    best_loss: job.metrics?.best_loss || 0,
    current_loss: job.metrics?.current_loss || 0,
  }))

  useEffect(() => {
    fetchABTests()
    fetchLearningJobs()
    fetchFeedbackData()
  }, [])

  const fetchABTests = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/experiments'))
      const data = await res.json()
      const experiments = Array.isArray(data?.experiments)
        ? data.experiments
        : Array.isArray(data)
          ? data
          : []
      const mapped = experiments.map((exp: any) => {
        const variants = Array.isArray(exp.variants) ? exp.variants : []
        const control =
          variants.find((v: any) => v.isControl) || variants[0] || {}
        const treatment =
          variants.find((v: any) => !v.isControl) || variants[1] || {}
        const controlRate = Number(control.conversionRate || 0)
        const treatmentRate = Number(treatment.conversionRate || 0)
        const winner =
          treatmentRate === controlRate
            ? 'tie'
            : treatmentRate > controlRate
              ? 'B'
              : 'A'
        return {
          id: String(exp.id),
          name: String(exp.name || ''),
          description: String(exp.description || exp.hypothesis || ''),
          model_a_id: String(control.id || ''),
          model_a_name: String(control.name || 'Control'),
          model_b_id: String(treatment.id || ''),
          model_b_name: String(treatment.name || 'Treatment'),
          traffic_split: Number(treatment.traffic ?? 0),
          status: exp.status || 'pending',
          start_time: exp.startDate || exp.created_at || '',
          end_time: exp.endDate || exp.updated_at || '',
          total_requests: Number(
            exp.participants || exp.sampleSize?.current || 0
          ),
          model_a_requests: Number(control.sampleSize || 0),
          model_b_requests: Number(treatment.sampleSize || 0),
          model_a_performance: {
            accuracy: controlRate,
            latency: 0,
            error_rate: Math.max(0, 1 - controlRate),
          },
          model_b_performance: {
            accuracy: treatmentRate,
            latency: 0,
            error_rate: Math.max(0, 1 - treatmentRate),
          },
          statistical_significance: Math.abs(Number(exp.lift || 0)),
          winner,
        } as ABTest
      })
      setAbTests(mapped)
    } catch (error) {
      message.error('加载A/B测试失败')
      setAbTests([])
    } finally {
      setLoading(false)
    }
  }

  const fetchLearningJobs = async () => {
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/fine-tuning/jobs'))
      const data = await res.json()
      const jobs = Array.isArray(data)
        ? data
        : Array.isArray(data?.jobs)
          ? data.jobs
          : []
      const mapped = jobs.map((job: any) => {
        const config = job.config || {}
        const status = String(job.status || 'pending')
        const trainingMode = String(config.training_mode || '').toLowerCase()
        const learningType =
          trainingMode === 'qlora'
            ? 'qlora'
            : trainingMode === 'full'
              ? 'full'
              : trainingMode === 'prefix'
                ? 'prefix'
                : trainingMode === 'p_tuning'
                  ? 'p_tuning'
                  : 'lora'
        const currentLoss = Number(job.current_loss ?? 0)
        const bestLoss = Number(job.best_loss ?? 0)
        return {
          id: String(job.job_id || ''),
          name: String(job.job_name || job.job_id || ''),
          model_id: String(config.model_name || ''),
          model_name: String(config.model_name || ''),
          learning_type: learningType,
          status: status as LearningJob['status'],
          progress: Number(job.progress || 0),
          batch_size: Number(config.per_device_train_batch_size || 0),
          start_time: job.started_at || job.created_at || '',
          estimated_completion:
            job.completed_at || job.started_at || job.created_at || '',
          metrics: {
            current_loss: currentLoss,
            best_loss: bestLoss,
          },
        } as LearningJob
      })
      setLearningJobs(mapped)
    } catch (error) {
      message.error('加载学习任务失败')
      setLearningJobs([])
    }
  }

  const fetchFeedbackData = async () => {
    try {
      const cookieId = document.cookie
        .split(';')
        .map(item => item.trim())
        .find(item => item.startsWith('client_id='))
        ?.split('=')[1]
      const userId = cookieId ? decodeURIComponent(cookieId) : ''
      if (!userId) {
        setFeedbackData([])
        return
      }
      const res = await apiFetch(buildApiUrl(`/api/v1/feedback/user/${userId}`))
      const data = await res.json()
      const items = Array.isArray(data?.data?.items) ? data.data.items : []
      const mapped = items.map((item: any) => {
        const value = item.value
        const score = typeof value === 'number' ? value : 0
        return {
          id: String(item.event_id || ''),
          model_id: String(item.item_id || ''),
          prediction: item.item_id || item.metadata?.prediction || '',
          ground_truth: item.raw_value || item.value,
          feedback_score: score,
          user_feedback: item.metadata?.comment || item.feedback_type || '',
          timestamp: item.timestamp || new Date().toISOString(),
          processed: false,
        } as FeedbackData
      })
      setFeedbackData(mapped)
    } catch (error) {
      message.error('加载反馈数据失败')
      setFeedbackData([])
    }
  }

  const handleCreateTest = async (values: any) => {
    try {
      const traffic = Number(values.traffic_split || 50)
      const payload = {
        name: values.name,
        description: values.description || values.name,
        type: 'A/B Testing',
        status: 'draft',
        variants: [
          {
            name: values.model_a_name || 'Control',
            traffic: 100 - traffic,
            isControl: true,
          },
          {
            name: values.model_b_name || 'Treatment',
            traffic: traffic,
            isControl: false,
          },
        ],
        metrics: ['conversion_rate'],
      }
      const res = await apiFetch(buildApiUrl('/api/v1/experiments'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      await fetchABTests()
      setTestModalVisible(false)
      testForm.resetFields()
      message.success('A/B测试已创建')
    } catch (error) {
      message.error('创建A/B测试失败')
    }
  }

  const handleCreateJob = async (values: any) => {
    try {
      const payload = {
        job_name: values.name,
        model_name: values.model_name,
        dataset_path: values.dataset_path,
        training_mode: values.learning_type,
      }
      const res = await apiFetch(buildApiUrl('/api/v1/fine-tuning/jobs'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      await fetchLearningJobs()
      setJobModalVisible(false)
      jobForm.resetFields()
      message.success('学习任务已创建')
    } catch (error) {
      message.error('创建学习任务失败')
    }
  }

  const handleTestAction = async (
    testId: string,
    action: 'start' | 'pause' | 'stop' | 'resume'
  ) => {
    try {
      await apiFetch(buildApiUrl(`/api/v1/experiments/${testId}/${action}`), {
        method: 'POST',
      })
      await fetchABTests()
      const actionText =
        action === 'start'
          ? '启动'
          : action === 'pause'
            ? '暂停'
            : action === 'resume'
              ? '恢复'
              : '停止'
      message.success(`A/B测试已${actionText}`)
    } catch (error) {
      message.error('操作失败')
    }
  }

  const getStatusColor = (status: string) => {
    const colors = {
      pending: 'orange',
      running: 'green',
      completed: 'blue',
      paused: 'default',
      failed: 'red',
    }
    return colors[status as keyof typeof colors] || 'default'
  }

  const getStatusText = (status: string) => {
    const texts = {
      pending: '待启动',
      running: '运行中',
      completed: '已完成',
      paused: '已暂停',
      failed: '失败',
    }
    return texts[status as keyof typeof texts] || status
  }

  const getLearningTypeText = (type: string) => {
    const texts = {
      lora: 'LoRA',
      qlora: 'QLoRA',
      full: '全量微调',
      prefix: 'Prefix Tuning',
      p_tuning: 'P-Tuning',
    }
    return texts[type as keyof typeof texts] || type
  }

  const testColumns = [
    {
      title: '测试名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: ABTest) => (
        <Space>
          <Avatar style={{ backgroundColor: '#1890ff' }}>
            {text.charAt(0)}
          </Avatar>
          <div>
            <div>{text}</div>
            <div style={{ fontSize: '12px', color: '#999' }}>
              {record.description}
            </div>
          </div>
        </Space>
      ),
    },
    {
      title: '模型对比',
      key: 'models',
      render: (_, record: ABTest) => (
        <div>
          <div>A: {record.model_a_name}</div>
          <div>B: {record.model_b_name}</div>
        </div>
      ),
    },
    {
      title: '流量分配',
      dataIndex: 'traffic_split',
      key: 'traffic_split',
      render: (split: number) => (
        <div>
          <div>A: {100 - split}%</div>
          <div>B: {split}%</div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge
          status={getStatusColor(status) as any}
          text={getStatusText(status)}
        />
      ),
    },
    {
      title: '总请求数',
      dataIndex: 'total_requests',
      key: 'total_requests',
      render: (requests: number) => requests.toLocaleString(),
    },
    {
      title: '胜出模型',
      key: 'winner',
      render: (_, record: ABTest) => {
        if (!record.winner) return <Tag>待确定</Tag>
        const winner =
          record.winner === 'A' ? record.model_a_name : record.model_b_name
        return (
          <Tag color="gold" icon={<TrophyOutlined />}>
            {winner}
          </Tag>
        )
      },
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: ABTest) => (
        <Space>
          {record.status === 'running' ? (
            <Button
              icon={<StopOutlined />}
              size="small"
              onClick={() => handleTestAction(record.id, 'pause')}
            >
              暂停
            </Button>
          ) : (
            <Button
              icon={<PlayCircleOutlined />}
              size="small"
              type="primary"
              onClick={() =>
                handleTestAction(
                  record.id,
                  record.status === 'paused' ? 'resume' : 'start'
                )
              }
            >
              {record.status === 'paused' ? '恢复' : '启动'}
            </Button>
          )}
          <Button
            icon={<BarChartOutlined />}
            size="small"
            onClick={() => setSelectedTest(record)}
          >
            详情
          </Button>
        </Space>
      ),
    },
  ]

  const jobColumns = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: LearningJob) => (
        <Space>
          <Avatar style={{ backgroundColor: '#52c41a' }}>
            <BulbOutlined />
          </Avatar>
          <div>
            <div>{text}</div>
            <div style={{ fontSize: '12px', color: '#999' }}>
              {record.model_name}
            </div>
          </div>
        </Space>
      ),
    },
    {
      title: '学习类型',
      dataIndex: 'learning_type',
      key: 'learning_type',
      render: (type: string) => (
        <Tag
          color={
            type === 'lora'
              ? 'blue'
              : type === 'qlora'
                ? 'green'
                : type === 'full'
                  ? 'purple'
                  : type === 'prefix'
                    ? 'orange'
                    : 'cyan'
          }
        >
          {getLearningTypeText(type)}
        </Tag>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Badge
          status={getStatusColor(status) as any}
          text={getStatusText(status)}
        />
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
      title: '批次大小',
      dataIndex: 'batch_size',
      key: 'batch_size',
      render: (batchSize: number) => batchSize.toLocaleString(),
    },
    {
      title: '当前Loss',
      key: 'current_loss',
      render: (_, record: LearningJob) => (
        <div
          style={{
            color: (record.metrics?.current_loss || 0) > 0 ? '#52c41a' : '#999',
          }}
        >
          {(record.metrics?.current_loss || 0).toFixed(4)}
        </div>
      ),
    },
    {
      title: '最近时间',
      dataIndex: 'estimated_completion',
      key: 'estimated_completion',
      render: (time: string) => {
        if (!time) return '-'
        const now = new Date()
        const timeDate = new Date(time)
        const diffMs = now.getTime() - timeDate.getTime()
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
        const diffDays = Math.floor(diffHours / 24)

        if (diffDays > 0) {
          return `${diffDays}天前`
        } else if (diffHours > 0) {
          return `${diffHours}小时前`
        } else {
          const diffMinutes = Math.floor(diffMs / (1000 * 60))
          return `${Math.max(1, diffMinutes)}分钟前`
        }
      },
    },
  ]

  // 生成性能对比数据
  const generatePerformanceData = (test: ABTest) => {
    return [
      {
        metric: '转化率(%)',
        modelA: (test.model_a_performance.accuracy || 0) * 100,
        modelB: (test.model_b_performance.accuracy || 0) * 100,
      },
      {
        metric: '样本量',
        modelA: test.model_a_requests,
        modelB: test.model_b_requests,
      },
    ]
  }

  return (
    <div style={{ padding: '24px' }}>
      {/* 统计概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃A/B测试"
              value={systemStats.active_tests}
              prefix={<ExperimentOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中任务"
              value={systemStats.active_jobs}
              prefix={<RocketOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="反馈数据"
              value={systemStats.total_feedback}
              prefix={<FireOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均Loss"
              value={systemStats.avg_loss}
              precision={4}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="ab-tests">
        <TabPane tab="A/B测试" key="ab-tests">
          <Card
            title="A/B测试管理"
            extra={
              <Button
                type="primary"
                icon={<ExperimentOutlined />}
                onClick={() => setTestModalVisible(true)}
              >
                创建A/B测试
              </Button>
            }
          >
            <Table
              columns={testColumns}
              dataSource={abTests}
              rowKey="id"
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="在线学习" key="learning">
          <Card
            title="在线学习任务"
            extra={
              <Button
                type="primary"
                icon={<BulbOutlined />}
                onClick={() => setJobModalVisible(true)}
              >
                创建学习任务
              </Button>
            }
          >
            <Table
              columns={jobColumns}
              dataSource={learningJobs}
              rowKey="id"
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="反馈管理" key="feedback">
          <Row gutter={16}>
            <Col span={16}>
              <Card title="用户反馈数据">
                <List
                  itemLayout="horizontal"
                  dataSource={feedbackData}
                  renderItem={feedback => (
                    <List.Item>
                      <List.Item.Meta
                        avatar={
                          <Avatar
                            style={{
                              backgroundColor: feedback.processed
                                ? '#52c41a'
                                : '#faad14',
                            }}
                          >
                            {feedback.processed ? (
                              <CheckCircleOutlined />
                            ) : (
                              <ClockCircleOutlined />
                            )}
                          </Avatar>
                        }
                        title={
                          <Space>
                            <span>预测: {String(feedback.prediction)}</span>
                            <span>→</span>
                            <span>实际: {String(feedback.ground_truth)}</span>
                            {feedback.feedback_score > 0 ? (
                              <Tag
                                color={
                                  feedback.feedback_score >= 4
                                    ? 'green'
                                    : feedback.feedback_score >= 3
                                      ? 'orange'
                                      : 'red'
                                }
                              >
                                评分: {feedback.feedback_score}/5
                              </Tag>
                            ) : (
                              <Tag>{feedback.user_feedback || '反馈'}</Tag>
                            )}
                          </Space>
                        }
                        description={
                          <div>
                            <div>{feedback.user_feedback}</div>
                            <div
                              style={{
                                fontSize: '12px',
                                color: '#999',
                                marginTop: '4px',
                              }}
                            >
                              {new Date(feedback.timestamp).toLocaleString(
                                'zh-CN',
                                {
                                  year: 'numeric',
                                  month: '2-digit',
                                  day: '2-digit',
                                  hour: '2-digit',
                                  minute: '2-digit',
                                  second: '2-digit',
                                }
                              )}
                            </div>
                          </div>
                        }
                      />
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
            <Col span={8}>
              <Card title="反馈统计">
                <Statistic
                  title="待处理反馈"
                  value={feedbackData.filter(f => !f.processed).length}
                  suffix="条"
                  style={{ marginBottom: '16px' }}
                />
                <Statistic
                  title="平均评分"
                  value={
                    feedbackData.reduce((sum, f) => sum + f.feedback_score, 0) /
                      feedbackData.length || 0
                  }
                  precision={1}
                  suffix="/5"
                />
              </Card>

              <Card title="反馈导入" style={{ marginTop: '16px' }}>
                <Alert
                  type="info"
                  message="请通过 /api/v1/feedback 接口写入反馈数据"
                  showIcon
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="性能分析" key="analysis">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="模型性能趋势">
                {performanceTrendData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={performanceTrendData}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="time" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="best_loss"
                        stroke="#faad14"
                        name="最优Loss"
                      />
                      <Line
                        type="monotone"
                        dataKey="current_loss"
                        stroke="#52c41a"
                        name="当前Loss"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <Empty description="暂无性能数据" />
                )}
              </Card>
            </Col>
            <Col span={12}>
              <Card title="学习效果统计">
                {learningJobs.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={learningJobs.map(job => ({
                        name: job.name,
                        best_loss: job.metrics?.best_loss || 0,
                        current_loss: job.metrics?.current_loss || 0,
                      }))}
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <RechartsTooltip />
                      <Legend />
                      <Bar dataKey="best_loss" fill="#faad14" name="最优Loss" />
                      <Bar
                        dataKey="current_loss"
                        fill="#52c41a"
                        name="当前Loss"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <Empty description="暂无学习数据" />
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* A/B测试创建模态框 */}
      <Modal
        title="创建A/B测试"
        visible={testModalVisible}
        onCancel={() => setTestModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form form={testForm} layout="vertical" onFinish={handleCreateTest}>
          <Form.Item
            name="name"
            label="测试名称"
            rules={[{ required: true, message: '请输入测试名称' }]}
          >
            <Input placeholder="请输入测试名称" />
          </Form.Item>

          <Form.Item name="description" label="测试描述">
            <TextArea rows={3} placeholder="请输入测试描述" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="model_a_name"
                label="对照组名称"
                rules={[{ required: true, message: '请输入对照组名称' }]}
              >
                <Input placeholder="例如 Control" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="model_b_name"
                label="实验组名称"
                rules={[{ required: true, message: '请输入实验组名称' }]}
              >
                <Input placeholder="例如 Treatment" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="traffic_split"
            label="流量分配 (模型B的流量百分比)"
            initialValue={50}
          >
            <Slider
              min={10}
              max={90}
              marks={{ 10: '10%', 50: '50%', 90: '90%' }}
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                创建测试
              </Button>
              <Button onClick={() => setTestModalVisible(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 学习任务创建模态框 */}
      <Modal
        title="创建学习任务"
        visible={jobModalVisible}
        onCancel={() => setJobModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form form={jobForm} layout="vertical" onFinish={handleCreateJob}>
          <Form.Item
            name="name"
            label="任务名称"
            rules={[{ required: true, message: '请输入任务名称' }]}
          >
            <Input placeholder="请输入任务名称" />
          </Form.Item>

          <Form.Item
            name="model_name"
            label="目标模型"
            rules={[{ required: true, message: '请输入模型名称' }]}
          >
            <Input placeholder="例如 hf-internal-testing/tiny-random-LlamaForCausalLM" />
          </Form.Item>

          <Form.Item
            name="dataset_path"
            label="数据集路径"
            rules={[{ required: true, message: '请输入数据集路径' }]}
          >
            <Input placeholder="例如 data/train.jsonl" />
          </Form.Item>

          <Form.Item
            name="learning_type"
            label="学习类型"
            rules={[{ required: true, message: '请选择学习类型' }]}
          >
            <Select placeholder="选择学习类型">
              <Option value="lora">LoRA</Option>
              <Option value="qlora">QLoRA</Option>
              <Option value="full">全量微调</Option>
              <Option value="prefix">Prefix Tuning</Option>
              <Option value="p_tuning">P-Tuning</Option>
            </Select>
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                创建任务
              </Button>
              <Button onClick={() => setJobModalVisible(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* A/B测试详情模态框 */}
      <Modal
        title="A/B测试详情"
        visible={!!selectedTest}
        onCancel={() => setSelectedTest(null)}
        footer={null}
        width={800}
      >
        {selectedTest && (
          <Tabs defaultActiveKey="overview">
            <TabPane tab="测试概览" key="overview">
              <Descriptions bordered column={2}>
                <Descriptions.Item label="测试名称">
                  {selectedTest.name}
                </Descriptions.Item>
                <Descriptions.Item label="状态">
                  {getStatusText(selectedTest.status)}
                </Descriptions.Item>
                <Descriptions.Item label="开始时间">
                  {selectedTest.start_time
                    ? new Date(selectedTest.start_time).toLocaleString(
                        'zh-CN',
                        {
                          year: 'numeric',
                          month: '2-digit',
                          day: '2-digit',
                          hour: '2-digit',
                          minute: '2-digit',
                          second: '2-digit',
                        }
                      )
                    : '-'}
                </Descriptions.Item>
                <Descriptions.Item label="总请求数">
                  {selectedTest.total_requests.toLocaleString()}
                </Descriptions.Item>
                <Descriptions.Item label="提升幅度">
                  {(selectedTest.statistical_significance * 100).toFixed(1)}%
                </Descriptions.Item>
                <Descriptions.Item label="胜出模型">
                  {selectedTest.winner
                    ? selectedTest.winner === 'A'
                      ? selectedTest.model_a_name
                      : selectedTest.model_b_name
                    : '待确定'}
                </Descriptions.Item>
              </Descriptions>
            </TabPane>
            <TabPane tab="性能对比" key="comparison">
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={generatePerformanceData(selectedTest)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="metric" />
                  <YAxis />
                  <RechartsTooltip />
                  <Legend />
                  <Bar
                    dataKey="modelA"
                    fill="#1890ff"
                    name={selectedTest.model_a_name}
                  />
                  <Bar
                    dataKey="modelB"
                    fill="#52c41a"
                    name={selectedTest.model_b_name}
                  />
                </BarChart>
              </ResponsiveContainer>
            </TabPane>
          </Tabs>
        )}
      </Modal>
    </div>
  )
}

export default OnlineLearningPage
