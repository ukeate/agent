import React, { useState, useEffect, useRef } from 'react'
import { buildWsUrl } from '../utils/apiBase'
import {
  Card,
  Tabs,
  Button,
  Badge,
  Input,
  Select,
  Slider,
  Progress,
  Row,
  Col,
  Typography,
  Space,
  Tag,
  Alert,
  Timeline,
  message,
} from 'antd'
import {
  ExperimentOutlined,
  LineChartOutlined,
  BarChartOutlined,
  AimOutlined,
  ClockCircleOutlined,
  StarOutlined,
  HeartOutlined,
  RadarChartOutlined,
} from '@ant-design/icons'
import { logger } from '../utils/logger'
const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { TabPane } = Tabs
const { Option } = Select

// 类型定义
interface EmotionState {
  id: string
  emotion: string
  intensity: number
  valence: number
  arousal: number
  dominance: number
  timestamp: string
  confidence: number
  triggers?: string[]
  context?: Record<string, any>
}

interface PersonalityProfile {
  user_id: string
  emotional_traits: Record<string, number>
  baseline_emotions: Record<string, number>
  emotion_volatility: number
  recovery_rate: number
  dominant_emotions: string[]
  sample_count: number
  confidence_score: number
  created_at: string
  updated_at: string
}

interface EmotionAnalytics {
  user_id: string
  period_days: number
  temporal_patterns?: {
    best_hours: Array<[number, number]>
    worst_hours: Array<[number, number]>
    weekly_patterns: Record<string, number>
    monthly_patterns: Record<string, number>
  }
  emotion_distribution?: Record<string, number>
  volatility?: {
    overall_volatility: number
    valence_volatility: number
    arousal_volatility: number
    dominance_volatility: number
  }
  clusters?: Array<{
    name: string
    emotions: string[]
    frequency: number
  }>
  patterns?: Array<{
    pattern_type: string
    description: string
    frequency: number
    confidence: number
  }>
  recovery_analysis?: {
    average_recovery_time: number
    recovery_rate: number
    triggers: Record<string, number>
  }
}

import { emotionalIntelligenceService } from '../services/emotionalIntelligenceService'

const EmotionModelingPage: React.FC = () => {
  const [currentEmotion, setCurrentEmotion] = useState<EmotionState | null>(
    null
  )
  const [emotionHistory, setEmotionHistory] = useState<EmotionState[]>([])
  const [personalityProfile, setPersonalityProfile] =
    useState<PersonalityProfile | null>(null)
  const [prediction, setPrediction] = useState<any>(null)
  const [analytics, setAnalytics] = useState<EmotionAnalytics | null>(null)
  const [loading, setLoading] = useState(false)
  const isMountedRef = useRef(true)

  // 新状态录入表单
  const [newEmotion, setNewEmotion] = useState({
    emotion: '',
    intensity: 0.5,
    triggers: '',
    context: '',
  })

  const emotions = [
    'happiness',
    'sadness',
    'anger',
    'fear',
    'surprise',
    'disgust',
    'neutral',
    'joy',
    'trust',
    'anticipation',
    'contempt',
    'shame',
    'guilt',
    'pride',
    'envy',
    'love',
    'gratitude',
    'hope',
    'anxiety',
    'depression',
  ]
  const userId = 'user-default'

  useEffect(() => {
    isMountedRef.current = true
    loadData()

    // 建立WebSocket连接
    let ws: WebSocket | null = null
    try {
      const wsUrl = buildWsUrl(`/ws/emotion/${userId}`)
      ws = new WebSocket(wsUrl)

      ws.onopen = () => {
        logger.log('WebSocket连接已建立')
      }

      ws.onmessage = event => {
        try {
          const payload = JSON.parse(event.data)
          if (!isMountedRef.current) return
          if (payload.type === 'emotion_update' && payload.emotion) {
            setCurrentEmotion(payload.emotion)
            setEmotionHistory(prev => [payload.emotion, ...prev].slice(0, 50))
          } else if (payload.type === 'emotion_result') {
            const state =
              payload.data?.emotional_state ||
              payload.data?.recognition_result?.fused_emotion
            if (state) {
              const nextState = {
                id: payload.message_id || state.timestamp,
                emotion: state.emotion,
                intensity: state.intensity,
                valence: state.valence,
                arousal: state.arousal,
                dominance: state.dominance,
                confidence: state.confidence,
                timestamp: state.timestamp || payload.timestamp,
              } as EmotionState
              setCurrentEmotion(nextState)
              setEmotionHistory(prev => [nextState, ...prev].slice(0, 50))
            }
          } else if (payload.type === 'profile_update' && payload.profile) {
            setPersonalityProfile(payload.profile)
          } else if (payload.type === 'analytics_update' && payload.analytics) {
            setAnalytics(payload.analytics)
          }
        } catch (err) {
          logger.error('解析WebSocket消息失败:', err)
        }
      }

      ws.onerror = error => {
        logger.error('WebSocket连接错误:', error)
        if (
          ws.readyState !== WebSocket.CLOSING &&
          ws.readyState !== WebSocket.CLOSED
        ) {
          ws.close()
        }
      }

      ws.onclose = () => {
        logger.log('WebSocket连接已断开')
      }
    } catch (error) {
      logger.error('建立WebSocket连接失败:', error)
    }

    // 清理函数
    return () => {
      isMountedRef.current = false
      if (ws && ws.readyState !== WebSocket.CLOSED) {
        ws.close()
      }
    }
  }, [])

  const loadData = async () => {
    if (!isMountedRef.current) return
    setLoading(true)
    try {
      const history = await loadEmotionHistory()
      if (!history.length) {
        if (!isMountedRef.current) return
        setPersonalityProfile(null)
        setAnalytics(null)
        return
      }
      await Promise.all([
        loadPersonalityProfile(history.length),
        loadAnalytics(history.length),
      ])
    } catch (error) {
      logger.error('加载数据失败:', error)
    } finally {
      if (isMountedRef.current) {
        setLoading(false)
      }
    }
  }

  const loadEmotionHistory = async (): Promise<EmotionState[]> => {
    try {
      const data = await emotionalIntelligenceService.getEmotionStateHistory(
        userId,
        50
      )
      const history = Array.isArray((data as any)?.history)
        ? (data as any).history
        : Array.isArray((data as any)?.states)
          ? (data as any).states
          : Array.isArray(data)
            ? (data as any)
            : []
      if (!isMountedRef.current) return history
      setEmotionHistory(history)
      setCurrentEmotion(history[0] || null)
      return history
    } catch (error) {
      logger.error('获取情感历史失败:', error)
      message.error('无法获取情感历史')
      if (isMountedRef.current) {
        setEmotionHistory([])
        setCurrentEmotion(null)
      }
      return []
    }
  }

  const loadPersonalityProfile = async (historyCount?: number) => {
    if (typeof historyCount === 'number' && historyCount < 10) {
      if (isMountedRef.current) {
        setPersonalityProfile(null)
      }
      return
    }
    try {
      const data = await emotionalIntelligenceService.getEmotionProfile(userId)
      const profile = (data as any)?.profile || data
      if (isMountedRef.current) {
        setPersonalityProfile(profile || null)
      }
    } catch (error) {
      const status = (error as any)?.response?.status
      if (status !== 404) {
        logger.error('获取个性画像失败:', error)
        message.error('无法获取个性画像')
      }
      if (isMountedRef.current) {
        setPersonalityProfile(null)
      }
    }
  }

  const loadAnalytics = async (historyCount?: number) => {
    if (typeof historyCount === 'number' && historyCount === 0) {
      if (isMountedRef.current) {
        setAnalytics(null)
      }
      return
    }
    try {
      const data = await emotionalIntelligenceService.performEmotionAnalytics({
        days_back: 30,
      })
      const analyticsData = (data as any)?.analytics || data
      const temporalPatterns =
        analyticsData?.temporal_analysis?.patterns ||
        analyticsData?.temporal_patterns
      const distribution =
        analyticsData?.basic_statistics?.emotion_distribution ||
        analyticsData?.emotion_distribution
      if (!isMountedRef.current) return
      setAnalytics(
        analyticsData
          ? {
              user_id: analyticsData.user_id ?? userId,
              period_days:
                analyticsData.analysis_period?.days ??
                analyticsData.period_days ??
                30,
              temporal_patterns: temporalPatterns
                ? {
                    best_hours: temporalPatterns.best_hours || [],
                    worst_hours: temporalPatterns.worst_hours || [],
                    weekly_patterns: temporalPatterns.weekly_patterns || {},
                    monthly_patterns: temporalPatterns.monthly_patterns || {},
                  }
                : undefined,
              emotion_distribution: distribution,
              volatility:
                analyticsData?.temporal_analysis?.volatility ||
                analyticsData?.volatility,
              clusters:
                analyticsData?.emotion_clusters || analyticsData?.clusters,
              patterns:
                analyticsData?.transition_patterns || analyticsData?.patterns,
              recovery_analysis: analyticsData?.recovery_analysis,
            }
          : null
      )
    } catch (error) {
      const status = (error as any)?.response?.status
      if (status !== 400) {
        logger.error('获取情感分析失败:', error)
        message.error('无法获取情感分析')
      }
      if (isMountedRef.current) {
        setAnalytics(null)
      }
    }
  }

  const recordEmotion = async () => {
    if (!newEmotion.emotion) {
      message.error('请选择情感类型')
      return
    }

    try {
      const emotionData = {
        emotion: newEmotion.emotion,
        intensity: newEmotion.intensity,
        triggers: newEmotion.triggers
          ? newEmotion.triggers
              .split(',')
              .map(t => t.trim())
              .filter(Boolean)
          : [],
        context: newEmotion.context ? { description: newEmotion.context } : {},
        source: 'manual',
        timestamp: new Date().toISOString(),
        user_id: userId,
      }

      await emotionalIntelligenceService.recordEmotionState(emotionData)

      message.success('情感状态记录成功')
      if (isMountedRef.current) {
        setNewEmotion({
          emotion: '',
          intensity: 0.5,
          triggers: '',
          context: '',
        })
        await loadData()
      }
    } catch (error) {
      logger.error('记录情感状态失败:', error)
      message.error('记录失败，请重试')
    }
  }

  const generatePrediction = async (timeHorizon: number = 1) => {
    if (!isMountedRef.current) return
    setLoading(true)
    try {
      const data = await emotionalIntelligenceService.predictEmotion({
        user_id: userId,
        time_horizon_hours: timeHorizon,
      })
      const predictionData = (data as any)?.prediction || data
      if (isMountedRef.current) {
        setPrediction(predictionData)
      }
      message.success('预测生成成功')
    } catch (error) {
      logger.error('生成预测失败:', error)
      message.error('预测失败，请重试')
    } finally {
      if (isMountedRef.current) {
        setLoading(false)
      }
    }
  }

  const getEmotionColor = (emotion: string) => {
    const colorMap: Record<string, string> = {
      happiness: 'gold',
      joy: 'orange',
      sadness: 'blue',
      anger: 'red',
      fear: 'purple',
      neutral: 'default',
      surprise: 'magenta',
      love: 'pink',
      gratitude: 'green',
    }
    return colorMap[emotion] || 'default'
  }

  const renderPersonalityTraits = () => {
    if (!personalityProfile) return null

    const traits = personalityProfile.emotional_traits
    const traitNames = {
      extraversion: '外向性',
      neuroticism: '神经质',
      agreeableness: '宜人性',
      conscientiousness: '尽责性',
      openness: '开放性',
    }

    return (
      <Space direction="vertical" style={{ width: '100%' }}>
        {Object.entries(traits).map(([trait, value]) => (
          <div key={trait}>
            <Text strong style={{ display: 'inline-block', width: '80px' }}>
              {traitNames[trait as keyof typeof traitNames]}
            </Text>
            <Progress
              percent={Math.round(
                (typeof value === 'number' ? value : 0) * 100
              )}
              size="small"
              style={{ flex: 1, marginLeft: 10 }}
              strokeColor="#1890ff"
            />
          </div>
        ))}
      </Space>
    )
  }

  const renderEmotionDistribution = () => {
    if (!analytics?.emotion_distribution) return null

    return (
      <Space direction="vertical" style={{ width: '100%' }}>
        {Object.entries(analytics.emotion_distribution).map(
          ([emotion, percentage]) => (
            <div
              key={emotion}
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
              }}
            >
              <Tag color={getEmotionColor(emotion)}>{emotion}</Tag>
              <Text strong>{((percentage as number) * 100).toFixed(1)}%</Text>
            </div>
          )
        )}
      </Space>
    )
  }

  const renderCurrentState = () => (
    <Row gutter={24}>
      <Col span={8}>
        <Card
          title={
            <span>
              <LineChartOutlined style={{ marginRight: 8 }} />
              当前情感状态
            </span>
          }
        >
          {currentEmotion ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ textAlign: 'center' }}>
                <Tag
                  color={getEmotionColor(currentEmotion.emotion)}
                  style={{ fontSize: '16px', padding: '8px 16px' }}
                >
                  {currentEmotion.emotion}
                </Tag>
              </div>
              <div>
                <Text>强度: </Text>
                <Progress
                  percent={Math.round(currentEmotion.intensity * 100)}
                  size="small"
                  strokeColor="#52c41a"
                />
              </div>
              <div>
                <Text>置信度: </Text>
                <Progress
                  percent={Math.round(currentEmotion.confidence * 100)}
                  size="small"
                  strokeColor="#1890ff"
                />
              </div>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                更新时间: {new Date(currentEmotion.timestamp).toLocaleString()}
              </Text>
            </Space>
          ) : (
            <Text type="secondary">暂无数据</Text>
          )}
        </Card>
      </Col>

      <Col span={8}>
        <Card
          title={
            <span>
              <AimOutlined style={{ marginRight: 8 }} />
              VAD空间坐标
            </span>
          }
        >
          {currentEmotion && (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>效价 (Valence):</Text>
                <Text code>{currentEmotion.valence.toFixed(2)}</Text>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>唤醒度 (Arousal):</Text>
                <Text code>{currentEmotion.arousal.toFixed(2)}</Text>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>支配性 (Dominance):</Text>
                <Text code>{currentEmotion.dominance.toFixed(2)}</Text>
              </div>
            </Space>
          )}
        </Card>
      </Col>

      <Col span={8}>
        <Card
          title={
            <span>
              <ClockCircleOutlined style={{ marginRight: 8 }} />
              快速操作
            </span>
          }
        >
          <Space direction="vertical" style={{ width: '100%' }}>
            <Button
              type="primary"
              block
              loading={loading}
              onClick={() => generatePrediction(1)}
            >
              生成情感预测
            </Button>
            <Button block onClick={loadData}>
              刷新数据
            </Button>
          </Space>
        </Card>
      </Col>
    </Row>
  )

  const renderEmotionRecord = () => (
    <Card title="记录新的情感状态" style={{ maxWidth: 800, margin: '0 auto' }}>
      <Row gutter={16}>
        <Col span={12}>
          <div style={{ marginBottom: 16 }}>
            <Text strong>情感类型</Text>
            <Select
              style={{ width: '100%', marginTop: 8 }}
              placeholder="选择情感类型"
              value={newEmotion.emotion}
              onChange={value =>
                setNewEmotion({ ...newEmotion, emotion: value })
              }
            >
              {emotions.map(emotion => (
                <Option key={emotion} value={emotion}>
                  {emotion}
                </Option>
              ))}
            </Select>
          </div>
        </Col>
        <Col span={12}>
          <div style={{ marginBottom: 16 }}>
            <Text strong>强度: {Math.round(newEmotion.intensity * 100)}%</Text>
            <Slider
              style={{ marginTop: 8 }}
              min={0}
              max={1}
              step={0.1}
              value={newEmotion.intensity}
              onChange={value =>
                setNewEmotion({ ...newEmotion, intensity: value })
              }
            />
          </div>
        </Col>
      </Row>

      <div style={{ marginBottom: 16 }}>
        <Text strong>触发因素 (逗号分隔)</Text>
        <Input
          style={{ marginTop: 8 }}
          value={newEmotion.triggers}
          onChange={e =>
            setNewEmotion({ ...newEmotion, triggers: e.target.value })
          }
          placeholder="工作压力, 家庭问题, 运动..."
        />
      </div>

      <div style={{ marginBottom: 16 }}>
        <Text strong>上下文描述</Text>
        <TextArea
          style={{ marginTop: 8 }}
          rows={3}
          value={newEmotion.context}
          onChange={e =>
            setNewEmotion({ ...newEmotion, context: e.target.value })
          }
          placeholder="描述当前情况或相关背景..."
        />
      </div>

      <Button
        type="primary"
        block
        size="large"
        disabled={!newEmotion.emotion}
        onClick={recordEmotion}
      >
        记录情感状态
      </Button>
    </Card>
  )

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <ExperimentOutlined style={{ marginRight: 12, color: '#1890ff' }} />
          情感状态建模系统
        </Title>
      </div>

      <Tabs defaultActiveKey="current">
        <TabPane tab="当前状态" key="current">
          {renderCurrentState()}
        </TabPane>

        <TabPane tab="记录情感" key="record">
          {renderEmotionRecord()}
        </TabPane>

        <TabPane tab="历史轨迹" key="history">
          <Card title="情感历史轨迹">
            <Timeline>
              {emotionHistory.slice(0, 8).map((state, index) => (
                <Timeline.Item
                  key={state.id}
                  color={index === 0 ? 'green' : 'blue'}
                >
                  <div
                    style={{ display: 'flex', alignItems: 'center', gap: 12 }}
                  >
                    <Tag color={getEmotionColor(state.emotion)}>
                      {state.emotion}
                    </Tag>
                    <Text>强度: {Math.round(state.intensity * 100)}%</Text>
                    <Text type="secondary">
                      {new Date(state.timestamp).toLocaleString()}
                    </Text>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </TabPane>

        <TabPane tab="个性画像" key="personality">
          <Row gutter={24}>
            <Col span={12}>
              <Card title="Big Five人格特质">{renderPersonalityTraits()}</Card>
            </Col>
            <Col span={12}>
              <Card title="情感特征">
                {personalityProfile && (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                      }}
                    >
                      <Text>情感波动性:</Text>
                      <Badge
                        color={
                          personalityProfile.emotion_volatility > 0.7
                            ? 'red'
                            : 'blue'
                        }
                        text={`${Math.round(personalityProfile.emotion_volatility * 100)}%`}
                      />
                    </div>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                      }}
                    >
                      <Text>恢复速度:</Text>
                      <Badge
                        color={
                          personalityProfile.recovery_rate > 0.7
                            ? 'green'
                            : 'blue'
                        }
                        text={`${Math.round(personalityProfile.recovery_rate * 100)}%`}
                      />
                    </div>
                    <div>
                      <Text strong>主导情感:</Text>
                      <div style={{ marginTop: 8 }}>
                        {personalityProfile.dominant_emotions.map(emotion => (
                          <Tag key={emotion} color={getEmotionColor(emotion)}>
                            {emotion}
                          </Tag>
                        ))}
                      </div>
                    </div>
                  </Space>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="情感预测" key="prediction">
          <Card
            title={
              <span>
                <StarOutlined style={{ marginRight: 8 }} />
                情感预测结果
              </span>
            }
          >
            {prediction ? (
              <Space direction="vertical" style={{ width: '100%' }}>
                <div
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                  }}
                >
                  <Text strong>预测置信度:</Text>
                  <Badge
                    count={`${Math.round(prediction.confidence * 100)}%`}
                    style={{ backgroundColor: '#52c41a' }}
                  />
                </div>

                <div>
                  <Title level={4}>
                    预测情感序列 ({prediction.time_horizon_hours}小时内):
                  </Title>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {prediction.predictions.map((pred: any, index: number) => (
                      <div
                        key={index}
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'space-between',
                        }}
                      >
                        <Tag color={getEmotionColor(pred.emotion)}>
                          {pred.emotion}
                        </Tag>
                        <div
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            gap: 8,
                            flex: 1,
                            marginLeft: 16,
                          }}
                        >
                          <Progress
                            percent={Math.round(pred.probability * 100)}
                            size="small"
                            style={{ flex: 1 }}
                            strokeColor="#1890ff"
                          />
                          <Text style={{ minWidth: '80px' }}>
                            {Math.round(pred.probability * 100)}%
                          </Text>
                          <Text
                            type="secondary"
                            style={{ minWidth: '120px', fontSize: '12px' }}
                          >
                            强度: {pred.intensity_range[0].toFixed(1)}-
                            {pred.intensity_range[1].toFixed(1)}
                          </Text>
                        </div>
                      </div>
                    ))}
                  </Space>
                </div>
              </Space>
            ) : (
              <div style={{ textAlign: 'center', padding: 40 }}>
                <Button
                  type="primary"
                  size="large"
                  loading={loading}
                  onClick={() => generatePrediction(1)}
                >
                  生成情感预测
                </Button>
              </div>
            )}
          </Card>
        </TabPane>

        <TabPane tab="数据分析" key="analytics">
          <Row gutter={24}>
            <Col span={12}>
              <Card
                title={
                  <span>
                    <BarChartOutlined style={{ marginRight: 8 }} />
                    情感分布
                  </span>
                }
              >
                {renderEmotionDistribution()}
              </Card>
            </Col>
            <Col span={12}>
              <Card title="时间模式分析">
                {analytics?.temporal_patterns && (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Title level={5}>最佳时段:</Title>
                      <Space direction="vertical">
                        {analytics.temporal_patterns.best_hours.map(
                          ([hour, score]: [number, number]) => (
                            <div
                              key={hour}
                              style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                              }}
                            >
                              <Text>{hour}:00</Text>
                              <Text style={{ color: '#52c41a' }}>
                                {Math.round(score * 100)}%
                              </Text>
                            </div>
                          )
                        )}
                      </Space>
                    </div>

                    <div>
                      <Title level={5}>低潮时段:</Title>
                      <Space direction="vertical">
                        {analytics.temporal_patterns.worst_hours.map(
                          ([hour, score]: [number, number]) => (
                            <div
                              key={hour}
                              style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                              }}
                            >
                              <Text>{hour}:00</Text>
                              <Text style={{ color: '#ff4d4f' }}>
                                {Math.round(score * 100)}%
                              </Text>
                            </div>
                          )
                        )}
                      </Space>
                    </div>
                  </Space>
                )}
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default EmotionModelingPage
