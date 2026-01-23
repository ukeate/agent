import React, { useState, useEffect } from 'react'
import { buildWsUrl } from '../utils/apiBase'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Tag,
  Timeline,
  Button,
  Space,
  Tabs,
  Badge,
  Alert,
  Table,
  Typography,
  message,
} from 'antd'
import { personalizationService } from '../services/personalizationService'
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
  SyncOutlined,
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
  const [engineStatus, setEngineStatus] = useState<
    'running' | 'stopped' | 'error'
  >('running')
  const [apiStatus, setApiStatus] = useState({
    metrics: 'unknown',
    features: 'unknown',
    models: 'unknown',
    recommend: 'unknown',
    websocket: 'unknown',
  })
  const [metrics, setMetrics] = useState({
    latencyP99: 0,
    throughput: 0,
    cacheHitRate: 0,
    activeUsers: 0,
    totalRecommendations: 0,
    errorRate: 0,
  })

  const [features, setFeatures] = useState<FeatureData[]>([])

  const [models, setModels] = useState<ModelInfo[]>([])

  const [recommendations, setRecommendations] = useState<RecommendationItem[]>(
    []
  )

  const updateApiStatus = (key: keyof typeof apiStatus, status: string) => {
    setApiStatus(prev => ({ ...prev, [key]: status }))
  }

  const checkWebSocket = async () => {
    if (typeof window === 'undefined') return false
    const userId = personalizationService.getClientId?.()
    if (!userId) return false
    const url = buildWsUrl('/personalization/stream')
    if (!url) return false
    return await new Promise<boolean>(resolve => {
      let settled = false
      const ws = new WebSocket(url)
      const timer = window.setTimeout(() => {
        if (settled) return
        settled = true
        ws.close()
        resolve(false)
      }, 5000)
      const finish = (result: boolean) => {
        if (settled) return
        settled = true
        window.clearTimeout(timer)
        ws.close()
        resolve(result)
      }
      ws.onopen = () => {
        ws.send(JSON.stringify({ user_id: userId }))
      }
      ws.onmessage = event => {
        try {
          const data = JSON.parse(event.data)
          if (data?.type === 'connected') {
            finish(true)
          }
        } catch {
          // 忽略解析错误
        }
      }
      ws.onerror = () => {
        finish(false)
      }
      ws.onclose = () => {
        finish(false)
      }
    })
  }

  const loadData = async () => {
    let hasError = false
    try {
      setLoading(true)
      setApiStatus({
        metrics: 'checking',
        features: 'checking',
        models: 'checking',
        recommend: 'checking',
        websocket: 'checking',
      })

      try {
        const overview = await personalizationService.getSystemOverview()
        if (overview) {
          setMetrics({
            latencyP99: overview.latency_p99 || 0,
            throughput: overview.throughput || 0,
            cacheHitRate: overview.cache_hit_rate || 0,
            activeUsers: overview.active_users || 0,
            totalRecommendations: overview.total_recommendations || 0,
            errorRate: overview.error_rate || 0,
          })
        }
        updateApiStatus('metrics', 'online')
      } catch {
        hasError = true
        updateApiStatus('metrics', 'error')
      }

      try {
        const featureList = await personalizationService.getFeatureStore()
        if (featureList && Array.isArray(featureList)) {
          setFeatures(
            featureList.map((f: any, idx: number) => ({
              key: f.id || String(idx),
              name: f.name,
              value: f.value ?? 0,
              type: f.type || '未知',
              updateTime: f.updated_at || '',
            }))
          )
        } else {
          setFeatures([])
        }
        updateApiStatus('features', 'online')
      } catch {
        hasError = true
        setFeatures([])
        updateApiStatus('features', 'error')
      }

      try {
        const modelList = await personalizationService.getModels()
        setModels(modelList || [])
        updateApiStatus('models', 'online')
      } catch {
        hasError = true
        setModels([])
        updateApiStatus('models', 'error')
      }

      try {
        const recs = await personalizationService.getRecommendations({
          limit: 5,
        })
        if (recs && Array.isArray(recs)) {
          setRecommendations(
            recs.map((r: any, idx: number) => ({
              id: r.id || String(idx),
              item: r.item || '',
              score: r.score || 0,
              reason: r.reason || '',
            }))
          )
        } else {
          setRecommendations([])
        }
        updateApiStatus('recommend', 'online')
      } catch {
        hasError = true
        setRecommendations([])
        updateApiStatus('recommend', 'error')
      }

      try {
        const wsOk = await checkWebSocket()
        updateApiStatus('websocket', wsOk ? 'online' : 'error')
        if (!wsOk) hasError = true
      } catch {
        hasError = true
        updateApiStatus('websocket', 'error')
      }
    } catch (err) {
      hasError = true
      message.error('加载个性化引擎数据失败')
    } finally {
      setEngineStatus(hasError ? 'error' : 'running')
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
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
      render: value => <Progress percent={value * 100} size="small" />,
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: type => {
        const colors: Record<string, string> = {
          实时特征: 'green',
          静态特征: 'blue',
          模型特征: 'purple',
          动态特征: 'orange',
        }
        return <Tag color={colors[type]}>{type}</Tag>
      },
    },
    {
      title: '更新时间',
      dataIndex: 'updateTime',
      key: 'updateTime',
      render: time => <Text type="secondary">{time}</Text>,
    },
  ]

  const handleRefreshEngine = async () => {
    setLoading(true)
    try {
      await loadData()
      setEngineStatus('running')
    } catch {
      setEngineStatus('error')
    } finally {
      setLoading(false)
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'online':
        return 'success'
      case 'offline':
        return 'default'
      case 'updating':
        return 'processing'
      default:
        return 'default'
    }
  }

  const getApiStatusDisplay = (status: string) => {
    switch (status) {
      case 'online':
        return { text: '正常', color: '#52c41a' }
      case 'error':
        return { text: '异常', color: '#ff4d4f' }
      case 'checking':
        return { text: '检测中', color: '#faad14' }
      default:
        return { text: '未检测', color: '#d9d9d9' }
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
            {engineStatus === 'running' ? (
              <CheckCircleOutlined />
            ) : (
              <WarningOutlined />
            )}
            <span>
              引擎状态: {engineStatus === 'running' ? '正常运行' : '异常'}
            </span>
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
              valueStyle={{
                color: metrics.latencyP99 > 100 ? '#ff4d4f' : '#52c41a',
              }}
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
              valueStyle={{
                color: metrics.errorRate > 1 ? '#ff4d4f' : '#52c41a',
              }}
            />
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="1">
        <TabPane
          tab={
            <span>
              <LineChartOutlined /> 实时特征
            </span>
          }
          key="1"
        >
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

        <TabPane
          tab={
            <span>
              <CloudServerOutlined /> 模型服务
            </span>
          }
          key="2"
        >
          <Row gutter={[16, 16]}>
            {models.map((model, index) => (
              <Col xs={24} sm={12} md={8} lg={6} key={index}>
                <Card>
                  <Badge
                    status={getStatusColor(model.status)}
                    text={model.status.toUpperCase()}
                  />
                  <Title level={5} style={{ marginTop: 8 }}>
                    {model.name}
                  </Title>
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

        <TabPane
          tab={
            <span>
              <HeartOutlined /> 推荐结果
            </span>
          }
          key="3"
        >
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
                      <Tag color="gold">
                        评分: {(rec.score * 100).toFixed(0)}%
                      </Tag>
                    </Space>
                    <Text type="secondary">推荐理由: {rec.reason}</Text>
                  </Space>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </TabPane>

        <TabPane
          tab={
            <span>
              <ApiOutlined /> API监控
            </span>
          }
          key="4"
        >
          <Card>
            <Title level={4}>API端点状态</Title>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card size="small">
                  <Statistic
                    title="/api/v1/personalization/recommend"
                    value={getApiStatusDisplay(apiStatus.recommend).text}
                    valueStyle={{
                      color: getApiStatusDisplay(apiStatus.recommend).color,
                    }}
                  />
                  <Text type="secondary">REST推荐接口</Text>
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small">
                  <Statistic
                    title="/ws/personalization/stream"
                    value={getApiStatusDisplay(apiStatus.websocket).text}
                    valueStyle={{
                      color: getApiStatusDisplay(apiStatus.websocket).color,
                    }}
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
