import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Statistic, Progress, Tag, Timeline, Button, Space, Tabs, Badge, Alert, Table, Typography } from 'antd'
import { 
  ThunderboltOutlined, 
  RocketOutlined, 
  ClockCircleOutlined,
  UserOutlined,
  HeartOutlined,
  TrophyOutlined,
  LineChartOutlined,
  DatabaseOutlined,
  CloudServerOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  SyncOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs

interface FeatureData {
  key: string
  name: string
  value: number
  type: string
  updateTime: string
}

interface ModelInfo {
  name: string
  version: string
  status: 'online' | 'offline' | 'updating'
  accuracy: number
  latency: number
}

interface RecommendationItem {
  id: string
  item: string
  score: number
  reason: string
}

const PersonalizationEnginePage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [engineStatus, setEngineStatus] = useState<'running' | 'stopped' | 'error'>('running')
  const [metrics, setMetrics] = useState({
    latencyP99: 85,
    throughput: 1250,
    cacheHitRate: 82,
    activeUsers: 3421,
    totalRecommendations: 152000,
    errorRate: 0.2
  })

  const [features, setFeatures] = useState<FeatureData[]>([
    { key: '1', name: '用户活跃度', value: 0.85, type: '实时特征', updateTime: '2秒前' },
    { key: '2', name: '兴趣相似度', value: 0.72, type: '静态特征', updateTime: '5分钟前' },
    { key: '3', name: '点击率预测', value: 0.68, type: '模型特征', updateTime: '刚刚' },
    { key: '4', name: '会话时长', value: 12.5, type: '实时特征', updateTime: '1秒前' },
    { key: '5', name: '内容新鲜度', value: 0.91, type: '动态特征', updateTime: '30秒前' }
  ])

  const [models, setModels] = useState<ModelInfo[]>([
    { name: 'DNN推荐模型', version: 'v2.3.1', status: 'online', accuracy: 92.5, latency: 23 },
    { name: 'CF协同过滤', version: 'v1.8.0', status: 'online', accuracy: 88.2, latency: 15 },
    { name: 'Transformer模型', version: 'v3.0.0', status: 'updating', accuracy: 94.1, latency: 45 },
    { name: 'Bandit算法', version: 'v1.2.0', status: 'online', accuracy: 85.7, latency: 8 }
  ])

  const [recommendations, setRecommendations] = useState<RecommendationItem[]>([
    { id: '1', item: '机器学习入门教程', score: 0.95, reason: '基于学习历史' },
    { id: '2', item: 'Python高级编程', score: 0.89, reason: '相似用户推荐' },
    { id: '3', item: '深度学习实战', score: 0.86, reason: '热门趋势' },
    { id: '4', item: '数据结构与算法', score: 0.82, reason: '个性化匹配' },
    { id: '5', item: 'AI系统设计', score: 0.78, reason: '协同过滤' }
  ])

  useEffect(() => {
    // 模拟实时数据更新
    const interval = setInterval(() => {
      setMetrics(prev => ({
        ...prev,
        latencyP99: Math.max(50, Math.min(150, prev.latencyP99 + (Math.random() - 0.5) * 10)),
        throughput: Math.max(800, Math.min(2000, prev.throughput + (Math.random() - 0.5) * 100)),
        cacheHitRate: Math.max(70, Math.min(95, prev.cacheHitRate + (Math.random() - 0.5) * 2)),
        activeUsers: Math.max(2000, Math.min(5000, prev.activeUsers + Math.floor((Math.random() - 0.5) * 200))),
        totalRecommendations: prev.totalRecommendations + Math.floor(Math.random() * 100)
      }))
    }, 3000)

    return () => clearInterval(interval)
  }, [])

  const featureColumns: ColumnsType<FeatureData> = [
    {
      title: '特征名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '特征值',
      dataIndex: 'value',
      key: 'value',
      render: (value) => <Progress percent={value * 100} size="small" />
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type) => {
        const colors: Record<string, string> = {
          '实时特征': 'green',
          '静态特征': 'blue',
          '模型特征': 'purple',
          '动态特征': 'orange'
        }
        return <Tag color={colors[type]}>{type}</Tag>
      }
    },
    {
      title: '更新时间',
      dataIndex: 'updateTime',
      key: 'updateTime',
      render: (time) => <Text type="secondary">{time}</Text>
    }
  ]

  const handleRefreshEngine = () => {
    setLoading(true)
    setTimeout(() => {
      setLoading(false)
      setEngineStatus('running')
    }, 2000)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online': return 'success'
      case 'offline': return 'default'
      case 'updating': return 'processing'
      default: return 'default'
    }
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <RocketOutlined /> 实时个性化引擎
      </Title>
      <Paragraph type="secondary">
        毫秒级响应的智能推荐系统，支持实时特征计算、分布式模型服务和在线学习
      </Paragraph>

      {/* 系统状态 */}
      <Alert
        message={
          <Space>
            {engineStatus === 'running' ? <CheckCircleOutlined /> : <WarningOutlined />}
            <span>引擎状态: {engineStatus === 'running' ? '正常运行' : '异常'}</span>
          </Space>
        }
        type={engineStatus === 'running' ? 'success' : 'warning'}
        showIcon={false}
        action={
          <Button 
            size="small" 
            type="primary" 
            icon={<SyncOutlined spin={loading} />}
            onClick={handleRefreshEngine}
            loading={loading}
          >
            刷新状态
          </Button>
        }
        style={{ marginBottom: 24 }}
      />

      {/* 核心指标 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={8} lg={4}>
          <Card>
            <Statistic
              title="P99延迟"
              value={metrics.latencyP99}
              suffix="ms"
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: metrics.latencyP99 > 100 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={4}>
          <Card>
            <Statistic
              title="吞吐量"
              value={metrics.throughput}
              suffix="RPS"
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={4}>
          <Card>
            <Statistic
              title="缓存命中率"
              value={metrics.cacheHitRate}
              suffix="%"
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={4}>
          <Card>
            <Statistic
              title="活跃用户"
              value={metrics.activeUsers}
              prefix={<UserOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={4}>
          <Card>
            <Statistic
              title="今日推荐"
              value={metrics.totalRecommendations}
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={8} lg={4}>
          <Card>
            <Statistic
              title="错误率"
              value={metrics.errorRate}
              suffix="%"
              precision={2}
              prefix={<WarningOutlined />}
              valueStyle={{ color: metrics.errorRate > 1 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="1">
        <TabPane tab={<span><LineChartOutlined /> 实时特征</span>} key="1">
          <Card>
            <Title level={4}>实时特征计算</Title>
            <Table 
              columns={featureColumns} 
              dataSource={features}
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab={<span><CloudServerOutlined /> 模型服务</span>} key="2">
          <Row gutter={[16, 16]}>
            {models.map((model, index) => (
              <Col xs={24} sm={12} md={8} lg={6} key={index}>
                <Card>
                  <Badge status={getStatusColor(model.status)} text={model.status.toUpperCase()} />
                  <Title level={5} style={{ marginTop: 8 }}>{model.name}</Title>
                  <Text type="secondary">{model.version}</Text>
                  <div style={{ marginTop: 16 }}>
                    <div style={{ marginBottom: 8 }}>
                      <Text>准确率</Text>
                      <Progress percent={model.accuracy} size="small" />
                    </div>
                    <div>
                      <Text>延迟: {model.latency}ms</Text>
                    </div>
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </TabPane>

        <TabPane tab={<span><HeartOutlined /> 推荐结果</span>} key="3">
          <Card>
            <Title level={4}>实时推荐示例</Title>
            <Timeline>
              {recommendations.map(rec => (
                <Timeline.Item 
                  key={rec.id}
                  dot={<TrophyOutlined />}
                  color={rec.score > 0.9 ? 'green' : 'blue'}
                >
                  <Space direction="vertical" size="small">
                    <Space>
                      <Text strong>{rec.item}</Text>
                      <Tag color="gold">评分: {(rec.score * 100).toFixed(0)}%</Tag>
                    </Space>
                    <Text type="secondary">推荐理由: {rec.reason}</Text>
                  </Space>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </TabPane>

        <TabPane tab={<span><ApiOutlined /> API监控</span>} key="4">
          <Card>
            <Title level={4}>API端点状态</Title>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card size="small">
                  <Statistic
                    title="/api/v1/personalization/recommend"
                    value="正常"
                    valueStyle={{ color: '#52c41a' }}
                  />
                  <Text type="secondary">REST推荐接口</Text>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small">
                  <Statistic
                    title="/ws/personalization/stream"
                    value="在线"
                    valueStyle={{ color: '#52c41a' }}
                  />
                  <Text type="secondary">WebSocket实时流</Text>
                </Card>
              </Col>
            </Row>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default PersonalizationEnginePage