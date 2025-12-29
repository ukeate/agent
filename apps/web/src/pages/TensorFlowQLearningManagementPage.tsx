import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Row, 
  Col, 
  Button, 
  Space, 
  Table, 
  Select,
  Tag,
  Statistic,
  Alert,
  Typography,
  Divider,
  Tabs
} from 'antd'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Legend, 
  ResponsiveContainer
} from 'recharts'
import {
  ThunderboltOutlined,
  RocketOutlined,
  MonitorOutlined,
  BarChartOutlined
} from '@ant-design/icons'

const { Title, Text } = Typography

interface TensorFlowQLearningConfig {
  id: string
  name: string
  framework: string
  status: 'running' | 'stopped' | 'training'
  performance: number
  episodes: number
}

interface TensorFlowQLearningOverview {
  summary: {
    running: number
    training: number
    average_performance: number
    total_episodes: number
  }
  models: TensorFlowQLearningConfig[]
  trend: { episode: number; reward: number }[]
}

const TensorFlowQLearningManagementPage: React.FC = () => {
  const [configs, setConfigs] = useState<TensorFlowQLearningConfig[]>([])
  const [performanceData, setPerformanceData] = useState<
    { episode: number; reward: number }[]
  >([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [summary, setSummary] = useState<TensorFlowQLearningOverview['summary']>({
    running: 0,
    training: 0,
    average_performance: 0,
    total_episodes: 0
  })

  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      setError('')
      try {
        const response = await apiFetch(buildApiUrl('/api/v1/tensorflow-qlearning/overview'))
        const data: TensorFlowQLearningOverview = await response.json()
        setConfigs(data.models || [])
        setPerformanceData(data.trend || [])
        setSummary(
          data.summary || {
            running: 0,
            training: 0,
            average_performance: 0,
            total_episodes: 0
          }
        )
      } catch (e) {
        setError('加载TensorFlow Q-Learning数据失败')
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  const columns = [
    {
      title: '模型名称',
      dataIndex: 'name',
      key: 'name'
    },
    {
      title: '框架',
      dataIndex: 'framework', 
      key: 'framework'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          running: 'green',
          stopped: 'red', 
          training: 'blue'
        }
        return <Tag color={colors[status as keyof typeof colors]}>{status}</Tag>
      }
    },
    {
      title: '性能 (%)',
      dataIndex: 'performance',
      key: 'performance',
      render: (value: number) => <Text strong>{value}%</Text>
    },
    {
      title: 'Episodes',
      dataIndex: 'episodes',
      key: 'episodes'
    },
    {
      title: '操作',
      key: 'actions',
      render: () => (
        <Space>
          <Button size="small">训练</Button>
          <Button size="small">停止</Button>
          <Button size="small">配置</Button>
        </Space>
      )
    }
  ]

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <ThunderboltOutlined /> TensorFlow Q-Learning 管理
      </Title>

      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中的模型"
              value={summary.running}
              prefix={<RocketOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="训练中的模型"
              value={summary.training}
              prefix={<MonitorOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均性能"
              value={summary.average_performance}
              precision={1}
              suffix="%"
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总Episodes"
              value={summary.total_episodes}
            />
          </Card>
        </Col>
      </Row>

      {error && (
        <Alert
          message="数据加载异常"
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: '16px' }}
        />
      )}

      <Alert
        message="TensorFlow Q-Learning 集成"
        description="管理基于TensorFlow的Q-Learning模型训练和推理。支持DQN、Double DQN、Dueling DQN等算法。"
        type="info"
        showIcon
        style={{ marginBottom: '24px' }}
      />

      <Tabs
        items={[
          {
            key: '1',
            label: '模型管理',
            children: (
              <Card title="Q-Learning 模型配置">
                <Space style={{ marginBottom: '16px' }}>
                  <Button type="primary">新建模型</Button>
                  <Button>批量操作</Button>
                  <Button>导入配置</Button>
                </Space>
                <Table
                  columns={columns}
                  dataSource={configs}
                  rowKey="id"
                  loading={loading}
                />
              </Card>
            )
          },
          {
            key: '2',
            label: '性能监控',
            children: (
              <Card title="训练性能趋势">
                <ResponsiveContainer width="100%" height={400}>
                  <LineChart data={performanceData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="episode" />
                    <YAxis />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="reward"
                      stroke="#8884d8"
                      strokeWidth={2}
                      dot={{ fill: '#8884d8' }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            )
          }
        ]}
      />
    </div>
  )
}

export default TensorFlowQLearningManagementPage
