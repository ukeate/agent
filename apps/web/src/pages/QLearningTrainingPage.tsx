import React, { useEffect, useState } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Button,
  Space,
  Alert,
  Typography,
  Divider,
  Timeline,
  message,
} from 'antd'
import {
  ExperimentOutlined,
  MonitorOutlined,
  TrophyOutlined,
  LineChartOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts'
import apiClient from '../services/apiClient'

const { Title, Text } = Typography

interface TrendPoint {
  episode: number
  reward: number
  loss?: number
  averageReward?: number
}

const QLearningTrainingPage: React.FC = () => {
  const [summary, setSummary] = useState({
    running: 0,
    training: 0,
    average_performance: 0,
    total_episodes: 0,
  })
  const [trainingData, setTrainingData] = useState<TrendPoint[]>([])
  const [updatedAt, setUpdatedAt] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const fetchData = async () => {
    setLoading(true)
    try {
      const resp = await apiClient.get('/tensorflow-qlearning/overview')
      const data: any = resp.data || {}
      setSummary(data.summary || summary)
      const trend = Array.isArray(data.trend) ? data.trend : []
      const points: TrendPoint[] = trend.map((item: any, idx: number) => ({
        episode: item.episode || idx + 1,
        reward: item.reward || 0,
        loss: item.loss || 0,
        averageReward: item.averageReward || item.reward || 0,
      }))
      setTrainingData(points)
      setUpdatedAt(data.updated_at || null)
    } catch (err) {
      message.error('加载训练监控数据失败')
      setTrainingData([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  const currentPoint = trainingData[trainingData.length - 1]
  const currentEpisode = currentPoint?.episode || summary.total_episodes || 0

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <ExperimentOutlined /> Q-Learning训练监控
      </Title>
      <Text type="secondary">
        实时展示Q-Learning训练趋势、奖励与损失变化。数据来源后端API，无本地模拟。
      </Text>
      <div style={{ margin: '16px 0' }}>
        <Space>
          <Button
            icon={<ReloadOutlined />}
            onClick={fetchData}
            loading={loading}
          >
            刷新数据
          </Button>
          {updatedAt && <Text type="secondary">最近更新: {updatedAt}</Text>}
        </Space>
      </div>

      <Divider />

      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Alert
          message="训练概览"
          description={`运行中: ${summary.running} | 训练中: ${summary.training} | 总Episodes: ${summary.total_episodes}`}
          type="info"
          showIcon
        />

        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Card>
              <Statistic
                title="当前Episode"
                value={currentEpisode}
                prefix={<ExperimentOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="当前奖励"
                value={currentPoint?.reward || 0}
                precision={1}
                prefix={<TrophyOutlined />}
                valueStyle={{
                  color:
                    (currentPoint?.reward || 0) >= 0 ? '#3f8600' : '#cf1322',
                }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均奖励"
                value={
                  summary.average_performance ||
                  currentPoint?.averageReward ||
                  0
                }
                precision={1}
                prefix={<LineChartOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="总训练Episodes"
                value={summary.total_episodes}
                prefix={<MonitorOutlined />}
              />
            </Card>
          </Col>
        </Row>

        <Row gutter={16}>
          <Col span={12}>
            <Card title="奖励曲线" size="small">
              {trainingData.length ? (
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={trainingData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="episode" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="reward"
                      stroke="#1890ff"
                      name="奖励"
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="averageReward"
                      stroke="#52c41a"
                      name="平均奖励"
                      dot={false}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <Alert message="暂无奖励数据" type="info" showIcon />
              )}
            </Card>
          </Col>

          <Col span={12}>
            <Card title="损失曲线" size="small">
              {trainingData.length ? (
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={trainingData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="episode" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Area
                      type="monotone"
                      dataKey="loss"
                      stroke="#ff4d4f"
                      fill="#ff4d4f"
                      fillOpacity={0.3}
                      name="损失"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              ) : (
                <Alert message="暂无损失数据" type="info" showIcon />
              )}
            </Card>
          </Col>
        </Row>

        <Card title="训练日志" size="small">
          {trainingData.length ? (
            <Timeline
              items={trainingData.slice(-5).map(item => ({
                children: `Episode ${item.episode} - 奖励 ${item.reward}`,
                color: item.reward >= 0 ? 'green' : 'red',
              }))}
            />
          ) : (
            <Alert message="暂无训练日志" type="info" showIcon />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default QLearningTrainingPage
