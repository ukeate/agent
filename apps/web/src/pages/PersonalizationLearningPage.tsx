import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Progress, Button, Space, Switch, Typography, Table, Tag, Statistic, Alert, Timeline, Slider, message, Empty } from 'antd'
import { 
import { logger } from '../utils/logger'
  BranchesOutlined,
  RiseOutlined,
  BulbOutlined,
  SyncOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  AimOutlined,
  RobotOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined
} from '@ant-design/icons'
import { Line, Gauge } from '@ant-design/plots'
import type { ColumnsType } from 'antd/es/table'
import { modelService } from '../services/modelService'

const { Title, Text, Paragraph } = Typography

interface LearningSession {
  session_id: string
  model_name: string
  model_version: string
  status: 'active' | 'paused' | 'completed' | 'failed'
  created_at: string
  updated_at: string
  config?: Record<string, any>
  feedback_count: number
  update_count: number
  performance_metrics: Record<string, number>
  pending_feedback: number
  buffer_usage: number
  buffer_capacity: number
}

interface OnlineMetric {
  timestamp: string
  loss: number
  mse: number
  mae: number
  accuracy: number
}

interface LearningHistory {
  timestamp: string
  update_count: number
  metrics: Record<string, any>
}

const PersonalizationLearningPage: React.FC = () => {
  const [sessions, setSessions] = useState<LearningSession[]>([])
  const [history, setHistory] = useState<LearningHistory[]>([])
  const [metrics, setMetrics] = useState<OnlineMetric[]>([])
  const activeSession = sessions.find((session) => session.status === 'active') || sessions[0]
  const hasConfig = Boolean(activeSession)
  const config = activeSession?.config || {}
  const learningRate = Number(config.learning_rate ?? 0)
  const explorationRate = Number(config.exploration_rate ?? 0)
  const batchSize = Number(config.batch_size ?? 0)
  const autoTuning = Boolean(config.auto_tuning ?? false)
  const isLearning = sessions.some((session) => session.status === 'active')

  const loadLearningData = async () => {
    try {
      const toNumber = (value: any) => {
        const n = Number(value)
        return Number.isFinite(n) ? n : 0
      }
      const sessionData = await modelService.getLearningSessions()
      setSessions(sessionData)

      const targetSession = sessionData.find((session) => session.status === 'active') || sessionData[0]
      if (!targetSession) {
        setHistory([])
        setMetrics([])
        return
      }

      const historyData = await modelService.getLearningHistory(targetSession.session_id)
      setHistory(historyData)
      const nextMetrics = historyData.map((item) => {
        const ts = String(item.timestamp)
        const values = item.metrics || {}
        return {
          timestamp: new Date(ts).toLocaleTimeString(),
          loss: toNumber(values.loss),
          mse: toNumber(values.mse),
          mae: toNumber(values.mae),
          accuracy: toNumber(values.accuracy),
        }
      })
      setMetrics(nextMetrics.slice(-60))
    } catch (error) {
      logger.error('加载在线学习数据失败:', error)
      message.error('加载在线学习数据失败')
    }
  }

  useEffect(() => {
    loadLearningData()
    const interval = setInterval(loadLearningData, 10000)
    return () => clearInterval(interval)
  }, [])

  const sessionColumns: ColumnsType<LearningSession> = [
    {
      title: '模型',
      dataIndex: 'model_name',
      key: 'model_name',
      render: (text, record) => <Text strong>{`${text} ${record.model_version}`}</Text>
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const config = {
          active: { color: 'processing', text: '运行中' },
          paused: { color: 'warning', text: '已暂停' },
          completed: { color: 'success', text: '已完成' },
          failed: { color: 'error', text: '失败' }
        }
        const target = config[status] || { color: 'default', text: String(status) }
        return <Tag color={target.color}>{target.text}</Tag>
      }
    },
    {
      title: '反馈数量',
      dataIndex: 'feedback_count',
      key: 'feedback_count',
      render: (value) => Number(value || 0).toLocaleString()
    },
    {
      title: '更新次数',
      dataIndex: 'update_count',
      key: 'update_count',
      render: (value) => Number(value || 0).toLocaleString()
    },
    {
      title: '最新损失',
      dataIndex: 'performance_metrics',
      key: 'performance_metrics',
      render: (_: Record<string, number>, record) => {
        const loss = record.performance_metrics?.loss
        if (typeof loss !== 'number') return '-'
        return (
          <Progress
            percent={Math.min(100, Math.max(0, loss * 100))}
            size="small"
            format={() => loss.toFixed(3)}
          />
        )
      }
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      render: (text) => <Text type="secondary">{text ? new Date(text).toLocaleString() : '-'}</Text>
    }
  ]

  // 损失趋势配置
  const lossConfig = {
    data: metrics,
    xField: 'timestamp',
    yField: 'loss',
    smooth: true,
    color: '#ff4d4f',
    yAxis: {
      title: { text: '损失值' }
    }
  }

  // MSE趋势配置
  const mseConfig = {
    data: metrics,
    xField: 'timestamp',
    yField: 'mse',
    smooth: true,
    color: '#52c41a',
    yAxis: {
      title: { text: 'MSE' }
    }
  }

  // 准确率仪表盘配置
  const latestMetric = metrics[metrics.length - 1]
  const accuracyValue = latestMetric ? Math.max(0, Math.min(1, latestMetric.accuracy)) : 0
  const accuracyGaugeConfig = {
    percent: accuracyValue,
    range: {
      color: 'l(0) 0:#ff4d4f 0.5:#faad14 1:#52c41a',
    },
    statistic: {
      content: {
        formatter: ({ percent }: any) => `准确率 ${(percent * 100).toFixed(1)}%`,
      },
    },
  }

  const handleToggleLearning = async () => {
    const activeSessions = sessions.filter((session) => session.status === 'active')
    const pausedSessions = sessions.filter((session) => session.status === 'paused')
    if (activeSessions.length === 0 && pausedSessions.length === 0) {
      message.info('暂无可操作的学习会话')
      return
    }

    try {
      if (activeSessions.length > 0) {
        await Promise.all(activeSessions.map((session) => modelService.pauseLearningSession(session.session_id)))
      } else {
        await Promise.all(pausedSessions.map((session) => modelService.resumeLearningSession(session.session_id)))
      }
      await loadLearningData()
    } catch (error) {
      logger.error('切换学习状态失败:', error)
      message.error('切换学习状态失败')
    }
  }

  const handleRefresh = () => {
    loadLearningData()
  }

  const totalFeedback = sessions.reduce((acc, session) => acc + (session.feedback_count || 0), 0)
  const totalUpdates = sessions.reduce((acc, session) => acc + (session.update_count || 0), 0)
  const efficiency = totalFeedback > 0 ? (totalUpdates / totalFeedback) * 100 : 0

  const timelineItems = [
    ...history.map((item) => ({
      key: `update-${item.update_count}-${item.timestamp}`,
      time: item.timestamp,
      title: `模型更新 #${item.update_count}`,
      detail: `loss=${Number(item.metrics?.loss || 0).toFixed(3)}，mse=${Number(item.metrics?.mse || 0).toFixed(3)}`,
      color: 'blue',
      icon: <BulbOutlined />,
    })),
    ...sessions.map((session) => ({
      key: `start-${session.session_id}`,
      time: session.created_at,
      title: '学习会话开始',
      detail: `${session.model_name} ${session.model_version}`,
      color: 'green',
      icon: <RobotOutlined />,
    })),
    ...sessions
      .filter((session) => session.status === 'completed')
      .map((session) => ({
        key: `complete-${session.session_id}`,
        time: session.updated_at,
        title: '学习会话完成',
        detail: `反馈 ${session.feedback_count}，更新 ${session.update_count}`,
        color: 'gray',
        icon: <ClockCircleOutlined />,
      })),
  ].sort((a, b) => new Date(b.time).getTime() - new Date(a.time).getTime())

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <BranchesOutlined /> 在线学习管理
      </Title>
      <Paragraph type="secondary">
        实时监控和管理个性化引擎的在线学习过程，支持动态调参和策略优化
      </Paragraph>

      {/* 学习状态告警 */}
      <Alert
        message={
          <Space>
            {isLearning ? <SyncOutlined spin /> : <PauseCircleOutlined />}
            <Text>学习状态: {isLearning ? '正在学习' : '已暂停'}</Text>
            <Text type="secondary">| 活跃会话: {sessions.filter(s => s.status === 'active').length}</Text>
          </Space>
        }
        type={isLearning ? 'success' : 'warning'}
        style={{ marginBottom: 24 }}
        action={
          <Space>
            <Button 
              type="primary" 
              icon={isLearning ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
              onClick={handleToggleLearning}
            >
              {isLearning ? '暂停' : '继续'}
            </Button>
            <Button 
              icon={<ReloadOutlined />}
              onClick={handleRefresh}
            >
              刷新
            </Button>
          </Space>
        }
      />

      {/* 学习配置 */}
      <Card title="学习参数配置" style={{ marginBottom: 24 }}>
        <Row gutter={[24, 16]}>
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>学习率: {hasConfig ? learningRate : '-'}</Text>
              <Slider
                min={0.001}
                max={0.1}
                step={0.001}
                value={Number.isFinite(learningRate) ? learningRate : 0}
                disabled
                marks={{
                  0.001: '0.001',
                  0.01: '0.01',
                  0.1: '0.1'
                }}
              />
            </Space>
          </Col>
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>探索率: {hasConfig ? explorationRate : '-'}</Text>
              <Slider
                min={0.01}
                max={0.5}
                step={0.01}
                value={Number.isFinite(explorationRate) ? explorationRate : 0}
                disabled
                marks={{
                  0.01: '1%',
                  0.1: '10%',
                  0.5: '50%'
                }}
              />
            </Space>
          </Col>
          <Col span={6}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>批大小: {hasConfig ? batchSize : '-'}</Text>
              <Slider
                min={8}
                max={128}
                step={8}
                value={Number.isFinite(batchSize) ? batchSize : 0}
                disabled
                marks={{
                  8: '8',
                  32: '32',
                  64: '64',
                  128: '128'
                }}
              />
            </Space>
          </Col>
          <Col span={6}>
            <Space direction="vertical">
              <Text>自动调参</Text>
              <Switch 
                checked={autoTuning}
                disabled
                checkedChildren="开启"
                unCheckedChildren="关闭"
              />
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 关键指标 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="最新损失"
              value={latestMetric ? latestMetric.loss : 0}
              precision={3}
              prefix={<RiseOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="反馈数量"
              value={totalFeedback}
              prefix={<SyncOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="更新次数"
              value={totalUpdates}
              prefix={<AimOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="更新效率"
              value={efficiency}
              suffix="%"
              precision={1}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 学习趋势图表 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card title="损失趋势">
            {metrics.length > 0 ? <Line {...lossConfig} height={200} /> : <Empty description="暂无数据" />}
          </Card>
        </Col>
        <Col span={8}>
          <Card title="MSE趋势">
            {metrics.length > 0 ? <Line {...mseConfig} height={200} /> : <Empty description="暂无数据" />}
          </Card>
        </Col>
        <Col span={8}>
          <Card title="当前准确率">
            {metrics.length > 0 ? <Gauge {...accuracyGaugeConfig} height={200} /> : <Empty description="暂无数据" />}
          </Card>
        </Col>
      </Row>

      {/* 学习会话表格 */}
      <Card title="学习会话" style={{ marginBottom: 24 }}>
        <Table 
          columns={sessionColumns}
          dataSource={sessions}
          pagination={false}
          size="middle"
        />
      </Card>

      {/* 学习历史时间轴 */}
      <Card title="学习历史">
        <Timeline>
          {timelineItems.length === 0 ? (
            <Timeline.Item>
              <Text type="secondary">暂无学习历史</Text>
            </Timeline.Item>
          ) : (
            timelineItems.map((item) => (
              <Timeline.Item key={item.key} color={item.color} dot={item.icon}>
                <Space direction="vertical" size="small">
                  <Text strong>{item.title}</Text>
                  <Text type="secondary">
                    {item.time ? new Date(item.time).toLocaleString() : '-'} {item.detail ? `- ${item.detail}` : ''}
                  </Text>
                </Space>
              </Timeline.Item>
            ))
          )}
        </Timeline>
      </Card>
    </div>
  )
}

export default PersonalizationLearningPage
