import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Tag,
  Alert,
  Button,
  Select,
  DatePicker,
  Space,
} from 'antd'
import {
  DashboardOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  LineChartOutlined,
  ApiOutlined,
  ClockCircleOutlined,
  FireOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import { monitoringService } from '../services/monitoringService'

const { RangePicker } = DatePicker
const { Option } = Select

interface SystemMetrics {
  qps: number
  avgLatency: number
  errorRate: number
  cacheHitRate: number
  activeUsers: number
  recommendationAccuracy: number
}

interface AlgorithmPerformance {
  algorithm: string
  requests: number
  avgReward: number
  latency: number
  accuracy: number
  status: 'excellent' | 'good' | 'warning' | 'error'
}

interface RecentActivity {
  id: string
  timestamp: string
  event: string
  user: string
  algorithm: string
  result: 'success' | 'warning' | 'error'
  details: string
}

const RLSystemDashboardPage: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    qps: 0,
    avgLatency: 0,
    errorRate: 0,
    cacheHitRate: 0,
    activeUsers: 0,
    recommendationAccuracy: 0,
  })

  const [algorithmData, setAlgorithmData] = useState<AlgorithmPerformance[]>([])

  const [recentActivities, setRecentActivities] = useState<RecentActivity[]>([])

  const [refreshing, setRefreshing] = useState(false)

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      const dashboard = await monitoringService.getDashboardData()
      const m = dashboard.metrics || {}
      setMetrics({
        qps: m.qps?.current_value || 0,
        avgLatency: m.api_response_time?.current_value || 0,
        errorRate: m.error_rate?.current_value || 0,
        cacheHitRate: m.cache_hit_rate?.current_value || 0,
        activeUsers: m.active_users?.current_value || 0,
        recommendationAccuracy: m.recommendation_accuracy?.current_value || 0,
      })
      setAlgorithmData([])
      setRecentActivities([])
    } catch (e) {
      setMetrics({
        qps: 0,
        avgLatency: 0,
        errorRate: 0,
        cacheHitRate: 0,
        activeUsers: 0,
        recommendationAccuracy: 0,
      })
    } finally {
      setRefreshing(false)
    }
  }

  useEffect(() => {
    handleRefresh()
  }, [])

  const algorithmColumns: ColumnsType<AlgorithmPerformance> = [
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      render: text => <strong>{text}</strong>,
    },
    {
      title: '请求数',
      dataIndex: 'requests',
      key: 'requests',
      render: value => value.toLocaleString(),
    },
    {
      title: '平均奖励',
      dataIndex: 'avgReward',
      key: 'avgReward',
      render: value => (
        <Statistic
          value={value}
          precision={3}
          valueStyle={{ fontSize: '14px' }}
        />
      ),
    },
    {
      title: '延迟 (ms)',
      dataIndex: 'latency',
      key: 'latency',
      render: value => (
        <Tag color={value < 15 ? 'green' : value < 30 ? 'orange' : 'red'}>
          {value}ms
        </Tag>
      ),
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: value => (
        <Progress
          percent={value}
          size="small"
          status={value > 85 ? 'success' : value > 80 ? 'normal' : 'exception'}
        />
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: status => {
        const config = {
          excellent: { color: 'green', text: '优秀' },
          good: { color: 'blue', text: '良好' },
          warning: { color: 'orange', text: '警告' },
          error: { color: 'red', text: '错误' },
        }
        return <Tag color={config[status].color}>{config[status].text}</Tag>
      },
    },
  ]

  const activityColumns: ColumnsType<RecentActivity> = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150,
    },
    {
      title: '事件',
      dataIndex: 'event',
      key: 'event',
      width: 100,
    },
    {
      title: '用户/系统',
      dataIndex: 'user',
      key: 'user',
      width: 120,
    },
    {
      title: '算法',
      dataIndex: 'algorithm',
      key: 'algorithm',
      width: 120,
    },
    {
      title: '结果',
      dataIndex: 'result',
      key: 'result',
      width: 80,
      render: result => {
        const config = {
          success: { color: 'green', icon: <CheckCircleOutlined /> },
          warning: { color: 'orange', icon: <ExclamationCircleOutlined /> },
          error: { color: 'red', icon: <ExclamationCircleOutlined /> },
        }
        return (
          <Tag color={config[result].color} icon={config[result].icon}>
            {result}
          </Tag>
        )
      },
    },
    {
      title: '详情',
      dataIndex: 'details',
      key: 'details',
      ellipsis: true,
    },
  ]

  useEffect(() => {
    // 设置自动刷新
    const interval = setInterval(() => {
      handleRefresh()
    }, 30000) // 30秒刷新一次

    return () => clearInterval(interval)
  }, [])

  return (
    <div style={{ padding: '24px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '24px',
        }}
      >
        <h1 style={{ margin: 0, display: 'flex', alignItems: 'center' }}>
          <DashboardOutlined style={{ marginRight: '8px' }} />
          强化学习系统仪表板
        </h1>
        <Space>
          <Select defaultValue="realtime" style={{ width: 120 }}>
            <Option value="realtime">实时</Option>
            <Option value="5min">5分钟</Option>
            <Option value="1hour">1小时</Option>
            <Option value="1day">1天</Option>
          </Select>
          <RangePicker showTime />
          <Button
            type="primary"
            icon={<ThunderboltOutlined />}
            loading={refreshing}
            onClick={handleRefresh}
          >
            刷新
          </Button>
        </Space>
      </div>

      {/* 系统状态警告 */}
      {metrics.errorRate > 1.0 && (
        <Alert
          message="系统状态异常"
          description={`当前错误率为 ${metrics.errorRate.toFixed(2)}%，建议检查系统配置`}
          type="error"
          showIcon
          closable
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 核心指标卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="QPS"
              value={metrics.qps}
              prefix={<ApiOutlined />}
              suffix="req/s"
              valueStyle={{ color: metrics.qps > 1000 ? '#3f8600' : '#cf1322' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="平均延迟"
              value={metrics.avgLatency}
              prefix={<ClockCircleOutlined />}
              suffix="ms"
              valueStyle={{
                color: metrics.avgLatency < 50 ? '#3f8600' : '#cf1322',
              }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="错误率"
              value={metrics.errorRate}
              prefix={<ExclamationCircleOutlined />}
              suffix="%"
              precision={2}
              valueStyle={{
                color: metrics.errorRate < 1 ? '#3f8600' : '#cf1322',
              }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="缓存命中率"
              value={metrics.cacheHitRate}
              prefix={<FireOutlined />}
              suffix="%"
              precision={1}
              valueStyle={{
                color: metrics.cacheHitRate > 90 ? '#3f8600' : '#cf1322',
              }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={12}>
          <Card>
            <Statistic
              title="活跃用户数"
              value={metrics.activeUsers}
              prefix={<LineChartOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12}>
          <Card>
            <Statistic
              title="推荐准确率"
              value={metrics.recommendationAccuracy}
              prefix={<CheckCircleOutlined />}
              suffix="%"
              precision={1}
              valueStyle={{
                color:
                  metrics.recommendationAccuracy > 85 ? '#3f8600' : '#cf1322',
              }}
            />
          </Card>
        </Col>
      </Row>

      {/* 算法性能表格 */}
      <Card
        title="算法性能概览"
        style={{ marginBottom: '24px' }}
        extra={<Tag color="blue">实时数据</Tag>}
      >
        <Table
          dataSource={algorithmData}
          columns={algorithmColumns}
          rowKey="algorithm"
          pagination={false}
          size="middle"
        />
      </Card>

      {/* 最近活动 */}
      <Card title="最近活动" extra={<Tag color="green">自动更新</Tag>}>
        <Table
          dataSource={recentActivities}
          columns={activityColumns}
          rowKey="id"
          pagination={{ pageSize: 10 }}
          size="small"
        />
      </Card>
    </div>
  )
}

export default RLSystemDashboardPage
