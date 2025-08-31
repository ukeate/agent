import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Progress,
  Statistic,
  Alert,
  Typography,
  Tabs,
  Space,
  Select,
  DatePicker,
  Switch,
  Tag,
  Tooltip,
  message,
  Modal,
  Form,
  InputNumber
} from 'antd'
import {
  ThunderboltOutlined,
  DashboardOutlined,
  LineChartOutlined,
  SettingOutlined,
  ReloadOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
  MemoryOutlined,
  HddOutlined
} from '@ant-design/icons'
// import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, AreaChart, Area } from 'recharts'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text } = Typography
const { Option } = Select
const { RangePicker } = DatePicker

interface PerformanceMetrics {
  cpu_usage: number
  memory_usage: number
  disk_usage: number
  network_io: number
  response_time: number
  throughput: number
  error_rate: number
  timestamp: string
}

interface Bottleneck {
  component: string
  type: 'cpu' | 'memory' | 'disk' | 'network' | 'database'
  severity: 'low' | 'medium' | 'high' | 'critical'
  description: string
  current_value: number
  threshold: number
  recommendations: string[]
}

interface OptimizationTask {
  task_id: string
  type: 'cache_optimization' | 'database_optimization' | 'memory_optimization' | 'query_optimization'
  status: 'pending' | 'running' | 'completed' | 'failed'
  progress: number
  description: string
  estimated_improvement: number
  started_at?: string
  completed_at?: string
}

const PerformanceOptimizationPage: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics[]>([])
  const [currentMetrics, setCurrentMetrics] = useState<PerformanceMetrics | null>(null)
  const [bottlenecks, setBottlenecks] = useState<Bottleneck[]>([])
  const [optimizationTasks, setOptimizationTasks] = useState<OptimizationTask[]>([])
  const [loading, setLoading] = useState(false)
  const [autoOptimization, setAutoOptimization] = useState(false)
  const [optimizationModalVisible, setOptimizationModalVisible] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchPerformanceData()
    const interval = setInterval(fetchPerformanceData, 10000)
    return () => clearInterval(interval)
  }, [])

  const fetchPerformanceData = async () => {
    try {
      const [metricsRes, bottlenecksRes, tasksRes] = await Promise.all([
        fetch('/api/v1/platform-integration/performance/metrics'),
        fetch('/api/v1/platform-integration/performance/bottlenecks'),
        fetch('/api/v1/platform-integration/performance/optimization-tasks')
      ])

      if (metricsRes.ok) {
        const data = await metricsRes.json()
        setMetrics(data.metrics || [])
        if (data.metrics?.length > 0) {
          setCurrentMetrics(data.metrics[data.metrics.length - 1])
        }
      }

      if (bottlenecksRes.ok) {
        const data = await bottlenecksRes.json()
        setBottlenecks(data.bottlenecks || [])
      }

      if (tasksRes.ok) {
        const data = await tasksRes.json()
        setOptimizationTasks(data.tasks || [])
      }
    } catch (error) {
      console.error('获取性能数据失败:', error)
    }
  }

  const handleStartOptimization = async (values: any) => {
    try {
      const response = await fetch('/api/v1/platform-integration/performance/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })

      if (response.ok) {
        message.success('优化任务已启动')
        setOptimizationModalVisible(false)
        form.resetFields()
        fetchPerformanceData()
      } else {
        message.error('启动优化失败')
      }
    } catch (error) {
      message.error('启动优化失败')
    }
  }

  const handleAutoOptimization = async (enabled: boolean) => {
    try {
      const response = await fetch('/api/v1/platform-integration/performance/auto-optimization', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ enabled })
      })

      if (response.ok) {
        setAutoOptimization(enabled)
        message.success(enabled ? '自动优化已启用' : '自动优化已禁用')
      } else {
        message.error('设置失败')
      }
    } catch (error) {
      message.error('设置失败')
    }
  }

  const getSeverityColor = (severity: string) => {
    const colors = {
      low: 'green',
      medium: 'orange',
      high: 'red',
      critical: 'purple'
    }
    return colors[severity] || 'default'
  }

  const getSeverityText = (severity: string) => {
    const texts = {
      low: '低',
      medium: '中',
      high: '高',
      critical: '严重'
    }
    return texts[severity] || severity
  }

  const getTypeIcon = (type: string) => {
    const icons = {
      cpu: <DashboardOutlined />,
      memory: <MemoryOutlined />,
      disk: <HddOutlined />,
      network: <CloudServerOutlined />,
      database: <DatabaseOutlined />
    }
    return icons[type] || <SettingOutlined />
  }

  const bottleneckColumns: ColumnsType<Bottleneck> = [
    {
      title: '组件',
      dataIndex: 'component',
      key: 'component'
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type) => (
        <Space>
          {getTypeIcon(type)}
          <span>{type.toUpperCase()}</span>
        </Space>
      )
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity) => (
        <Tag color={getSeverityColor(severity)}>
          {getSeverityText(severity)}
        </Tag>
      )
    },
    {
      title: '当前值',
      dataIndex: 'current_value',
      key: 'current_value',
      render: (value, record) => (
        <div>
          <div>{value.toFixed(2)}</div>
          <Progress
            percent={(value / record.threshold) * 100}
            size="small"
            status={value > record.threshold ? 'exception' : 'active'}
            showInfo={false}
          />
        </div>
      )
    },
    {
      title: '阈值',
      dataIndex: 'threshold',
      key: 'threshold'
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true
    }
  ]

  const taskColumns: ColumnsType<OptimizationTask> = [
    {
      title: '任务类型',
      dataIndex: 'type',
      key: 'type',
      render: (type) => {
        const typeTexts = {
          cache_optimization: '缓存优化',
          database_optimization: '数据库优化',
          memory_optimization: '内存优化',
          query_optimization: '查询优化'
        }
        return typeTexts[type] || type
      }
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const statusConfig = {
          pending: { color: 'default', text: '等待中' },
          running: { color: 'processing', text: '运行中' },
          completed: { color: 'success', text: '已完成' },
          failed: { color: 'error', text: '失败' }
        }
        const config = statusConfig[status]
        return <Tag color={config.color}>{config.text}</Tag>
      }
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress) => <Progress percent={progress} size="small" />
    },
    {
      title: '预期改善',
      dataIndex: 'estimated_improvement',
      key: 'estimated_improvement',
      render: (improvement) => `${improvement}%`
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true
    }
  ]

  // const formatMetricsData = (metrics: PerformanceMetrics[]) => {
  //   return metrics.map(metric => ({
  //     ...metric,
  //     time: new Date(metric.timestamp).toLocaleTimeString()
  //   }))
  // }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>性能优化</Title>
        <Space>
          <Text>自动优化:</Text>
          <Switch
            checked={autoOptimization}
            onChange={handleAutoOptimization}
            checkedChildren="开"
            unCheckedChildren="关"
          />
          <Button icon={<ReloadOutlined />} onClick={fetchPerformanceData}>
            刷新
          </Button>
          <Button type="primary" icon={<ThunderboltOutlined />} onClick={() => setOptimizationModalVisible(true)}>
            启动优化
          </Button>
        </Space>
      </div>

      {currentMetrics && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="CPU使用率"
                value={currentMetrics.cpu_usage}
                suffix="%"
                prefix={<DashboardOutlined />}
                valueStyle={{ 
                  color: currentMetrics.cpu_usage > 80 ? '#f5222d' : 
                         currentMetrics.cpu_usage > 60 ? '#faad14' : '#52c41a' 
                }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="内存使用率"
                value={currentMetrics.memory_usage}
                suffix="%"
                prefix={<MemoryOutlined />}
                valueStyle={{ 
                  color: currentMetrics.memory_usage > 80 ? '#f5222d' : 
                         currentMetrics.memory_usage > 60 ? '#faad14' : '#52c41a' 
                }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="响应时间"
                value={currentMetrics.response_time}
                suffix="ms"
                prefix={<LineChartOutlined />}
                valueStyle={{ 
                  color: currentMetrics.response_time > 1000 ? '#f5222d' : 
                         currentMetrics.response_time > 500 ? '#faad14' : '#52c41a' 
                }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="错误率"
                value={currentMetrics.error_rate}
                suffix="%"
                prefix={<ExclamationCircleOutlined />}
                valueStyle={{ 
                  color: currentMetrics.error_rate > 5 ? '#f5222d' : 
                         currentMetrics.error_rate > 1 ? '#faad14' : '#52c41a' 
                }}
              />
            </Card>
          </Col>
        </Row>
      )}

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="性能趋势" size="small">
            <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f5f5f5' }}>
              <Text type="secondary">图表组件临时禁用</Text>
            </div>
          </Card>
        </Col>
        <Col span={12}>
          <Card title="吞吐量与错误率" size="small">
            <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f5f5f5' }}>
              <Text type="secondary">图表组件临时禁用</Text>
            </div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card title="性能瓶颈" size="small">
            {bottlenecks.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px 0' }}>
                <CheckCircleOutlined style={{ fontSize: 48, color: '#52c41a' }} />
                <div style={{ marginTop: 16 }}>
                  <Text>当前没有检测到性能瓶颈</Text>
                </div>
              </div>
            ) : (
              <Table
                columns={bottleneckColumns}
                dataSource={bottlenecks}
                rowKey={(record, index) => `${record.component}-${index}`}
                pagination={false}
                size="small"
              />
            )}
          </Card>
        </Col>
        <Col span={12}>
          <Card title="优化任务" size="small">
            <Table
              columns={taskColumns}
              dataSource={optimizationTasks}
              rowKey="task_id"
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
      </Row>

      <Modal
        title="启动性能优化"
        open={optimizationModalVisible}
        onCancel={() => {
          setOptimizationModalVisible(false)
          form.resetFields()
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleStartOptimization}
        >
          <Form.Item
            name="optimization_type"
            label="优化类型"
            rules={[{ required: true, message: '请选择优化类型' }]}
          >
            <Select placeholder="选择优化类型" mode="multiple">
              <Option value="cache_optimization">缓存优化</Option>
              <Option value="database_optimization">数据库优化</Option>
              <Option value="memory_optimization">内存优化</Option>
              <Option value="query_optimization">查询优化</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="target_components"
            label="目标组件"
          >
            <Select placeholder="选择要优化的组件" mode="multiple">
              <Option value="api_service">API服务</Option>
              <Option value="database">数据库</Option>
              <Option value="cache">缓存</Option>
              <Option value="queue">消息队列</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="intensity"
            label="优化强度"
            initialValue={5}
          >
            <InputNumber
              min={1}
              max={10}
              style={{ width: '100%' }}
              placeholder="1-10，数值越高优化越激进"
            />
          </Form.Item>

          <Form.Item
            name="max_duration"
            label="最大执行时间 (分钟)"
            initialValue={30}
          >
            <InputNumber
              min={1}
              max={180}
              style={{ width: '100%' }}
              placeholder="优化任务的最大执行时间"
            />
          </Form.Item>

          <Alert
            message="注意"
            description="性能优化可能会临时影响系统性能，建议在低峰时段执行。"
            type="warning"
            style={{ marginBottom: 16 }}
          />

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => {
                setOptimizationModalVisible(false)
                form.resetFields()
              }}>
                取消
              </Button>
              <Button type="primary" htmlType="submit">
                开始优化
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default PerformanceOptimizationPage