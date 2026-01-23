import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import {
  Card,
  Row,
  Col,
  Select,
  DatePicker,
  Space,
  Button,
  Table,
  Statistic,
  Typography,
  message,
  Tag,
} from 'antd'
import { ReloadOutlined, LineChartOutlined } from '@ant-design/icons'

const { RangePicker } = DatePicker
const { Option } = Select

interface MetricRow {
  timestamp: string
  algorithm: string
  metric: string
  value: number
}

interface PerformanceRow {
  algorithm: string
  avg_reward: number
  conversion_rate: number
  ctr: number
  user_satisfaction: number
  performance: number
  trend: 'up' | 'down' | 'stable'
}

const RLMetricsAnalysisPage: React.FC = () => {
  const [metricData, setMetricData] = useState<MetricRow[]>([])
  const [performance, setPerformance] = useState<PerformanceRow[]>([])
  const [timeRange, setTimeRange] = useState<string>('24h')
  const [loading, setLoading] = useState(false)

  const loadData = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/qlearning/metrics'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ range: timeRange }),
      })
      const data = await res.json()
      setMetricData(data.metrics || [])
      setPerformance(data.performance || [])
    } catch (e: any) {
      message.error(e?.message || '加载RL指标失败')
      setMetricData([])
      setPerformance([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [timeRange])

  const metricColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (t: string) => new Date(t).toLocaleString(),
    },
    { title: '算法', dataIndex: 'algorithm', key: 'algorithm' },
    { title: '指标', dataIndex: 'metric', key: 'metric' },
    {
      title: '数值',
      dataIndex: 'value',
      key: 'value',
      render: (v: number) => v.toFixed(4),
    },
  ]

  const perfColumns = [
    { title: '算法', dataIndex: 'algorithm', key: 'algorithm' },
    {
      title: '平均奖励',
      dataIndex: 'avg_reward',
      key: 'avg_reward',
      render: (v: number) => v.toFixed(4),
    },
    {
      title: '转化率',
      dataIndex: 'conversion_rate',
      key: 'conversion_rate',
      render: (v: number) => `${(v * 100).toFixed(2)}%`,
    },
    {
      title: 'CTR',
      dataIndex: 'ctr',
      key: 'ctr',
      render: (v: number) => `${(v * 100).toFixed(2)}%`,
    },
    {
      title: '满意度',
      dataIndex: 'user_satisfaction',
      key: 'user_satisfaction',
      render: (v: number) => v.toFixed(2),
    },
    {
      title: '性能',
      dataIndex: 'performance',
      key: 'performance',
      render: (v: number) => v.toFixed(2),
    },
    {
      title: '趋势',
      dataIndex: 'trend',
      key: 'trend',
      render: (t: string) => (
        <Tag color={t === 'up' ? 'green' : t === 'down' ? 'red' : 'blue'}>
          {t}
        </Tag>
      ),
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      <Row justify="space-between" align="middle" style={{ marginBottom: 16 }}>
        <Col>
          <Space>
            <LineChartOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              RL 指标分析
            </Typography.Title>
          </Space>
          <Typography.Text type="secondary">
            数据来自 /api/v1/qlearning/metrics，实时后端返回
          </Typography.Text>
        </Col>
        <Col>
          <Space>
            <Select
              value={timeRange}
              onChange={setTimeRange}
              style={{ width: 120 }}
            >
              <Option value="1h">1小时</Option>
              <Option value="24h">24小时</Option>
              <Option value="7d">7天</Option>
              <Option value="30d">30天</Option>
            </Select>
            <Button
              icon={<ReloadOutlined />}
              onClick={loadData}
              loading={loading}
            >
              刷新
            </Button>
          </Space>
        </Col>
      </Row>

      <Card title="关键指标" style={{ marginBottom: 16 }}>
        <Table
          rowKey={r => `${r.timestamp}-${r.algorithm}-${r.metric}`}
          dataSource={metricData}
          columns={metricColumns}
          loading={loading}
          pagination={{ pageSize: 30 }}
        />
      </Card>

      <Card title="算法表现">
        <Table
          rowKey="algorithm"
          dataSource={performance}
          columns={perfColumns}
          loading={loading}
          pagination={false}
        />
      </Card>
    </div>
  )
}

export default RLMetricsAnalysisPage
