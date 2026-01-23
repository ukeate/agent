import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Tabs,
  Button,
  Input,
  Select,
  Row,
  Col,
  Typography,
  Space,
  Tag,
  Alert,
  Timeline,
  Progress,
  Divider,
  Badge,
  message,
  Table,
  Modal,
  Form,
  InputNumber,
} from 'antd'
import {
  TeamOutlined,
  LineChartOutlined,
  BarChartOutlined,
  NodeIndexOutlined,
  FireOutlined,
  ThunderboltOutlined,
  ExperimentOutlined,
  SyncOutlined,
} from '@ant-design/icons'
import { buildApiUrl, apiFetch } from '../utils/apiBase'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { TabPane } = Tabs
const { Option } = Select

// 群体情感状态类型定义
interface GroupEmotionalState {
  group_id: string
  timestamp: string
  participants: string[]
  dominant_emotion: string
  emotion_distribution: Record<string, number>
  consensus_level: number
  polarization_index: number
  emotional_volatility: number
  group_cohesion: string
  emotional_leaders: Array<{
    participant_id: string
    influence_score: number
    leadership_type: string
    influenced_participants: string[]
    dominant_emotions: string[]
    consistency_score: number
  }>
  influence_network: Record<string, string[]>
  contagion_patterns: Array<{
    source_participant: string
    target_participants: string[]
    emotion: string
    contagion_type: string
    strength: number
    propagation_speed: number
    timestamp: string
    duration_seconds: number
  }>
  contagion_velocity: number
  trend_prediction: string
  stability_score: number
  analysis_confidence: number
  data_completeness: number
}

// 情感传染事件
interface ContagionEvent {
  event_id: string
  source_participant: string
  target_participants: string[]
  emotion: string
  contagion_type: string
  strength: number
  timestamp: string
  propagation_speed?: number
  duration_seconds?: number
}

// 真实API客户端
const groupEmotionApi = {
  async analyzeGroupEmotion(participantEmotions: any, groupId?: string) {
    const participants = Object.entries(participantEmotions).map(
      ([userId, state]: [string, any]) => ({
        participant_id: userId,
        name: userId,
        emotion_data: {
          emotions: { [state.emotion]: state.intensity },
          intensity: state.intensity ?? 0.5,
          confidence: 0.8,
        },
        cultural_indicators: {},
        relationship_history: [],
      })
    )

    const payload = {
      session_id: groupId || `session_${Date.now()}`,
      participants,
      social_environment: {
        scenario: 'group_analysis',
        participants_count: participants.length,
        formality_level: 0.5,
        emotional_intensity: 0.5,
        time_pressure: 0.3,
        cultural_context: 'default',
      },
      analysis_types: ['group_emotion', 'contagion'],
      real_time: false,
    }

    const response = await apiFetch(
      buildApiUrl('/social-emotional-understanding/analyze/group-emotion'),
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      }
    )

    return await response.json()
  },

  async getGroupEmotionHistory(groupId: string, limit: number = 20) {
    const response = await apiFetch(
      buildApiUrl(
        `/social-emotional-understanding/group-emotion/history/${encodeURIComponent(groupId)}?limit=${limit}`
      )
    )
    return await response.json()
  },
}

const GroupEmotionAnalysisPage: React.FC = () => {
  const [groupState, setGroupState] = useState<GroupEmotionalState | null>(null)
  const [groupHistory, setGroupHistory] = useState<GroupEmotionalState[]>([])
  const [contagionEvents, setContagionEvents] = useState<ContagionEvent[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedGroupId, setSelectedGroupId] = useState('group_1')
  const [analysisParams, setAnalysisParams] = useState({
    timeWindow: 60,
    minParticipants: 3,
    confidenceThreshold: 0.6,
  })

  // 模拟分析表单
  const [form] = Form.useForm()
  const [showAnalysisModal, setShowAnalysisModal] = useState(false)
  const [participantEmotions, setParticipantEmotions] = useState<
    Record<string, any>
  >({})

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
    'excitement',
    'anxiety',
  ]

  const mapResponseToState = (resp: any): GroupEmotionalState | null => {
    const state = resp?.group_emotion_state
    if (!state) return null
    return {
      ...state,
      group_id: state.group_id || resp.session_id || 'group',
    }
  }

  const mapContagionEvents = (
    patterns: GroupEmotionalState['contagion_patterns'] = []
  ): ContagionEvent[] => {
    return patterns.map(pattern => ({
      event_id: `${pattern.source_participant}-${pattern.emotion}-${pattern.timestamp}`,
      source_participant: pattern.source_participant,
      target_participants: pattern.target_participants,
      emotion: pattern.emotion,
      contagion_type: pattern.contagion_type,
      strength: pattern.strength,
      timestamp: pattern.timestamp,
      propagation_speed: pattern.propagation_speed,
      duration_seconds: pattern.duration_seconds,
    }))
  }

  const groupCohesionColors = {
    high: '#52c41a',
    medium: '#1890ff',
    low: '#fa8c16',
    fragmented: '#f5222d',
  }

  const contagionTypeColors = {
    viral: '#f5222d',
    cascade: '#fa8c16',
    amplification: '#1890ff',
    dampening: '#52c41a',
  }

  useEffect(() => {
    loadData()
  }, [selectedGroupId])

  const loadData = async () => {
    setLoading(true)
    try {
      await loadGroupHistory()
    } catch (error) {
      logger.error('加载数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadGroupHistory = async () => {
    try {
      const response = await groupEmotionApi.getGroupEmotionHistory(
        selectedGroupId,
        20
      )
      const history = response?.history || []
      setGroupHistory(history)
      const latest = history[history.length - 1] || null
      setGroupState(latest)
      setContagionEvents(mapContagionEvents(latest?.contagion_patterns || []))
    } catch (error) {
      logger.error('获取群体历史失败:', error)
      setGroupHistory([])
      setGroupState(null)
      setContagionEvents([])
    }
  }

  const runGroupAnalysis = async () => {
    if (
      Object.keys(participantEmotions).length < analysisParams.minParticipants
    ) {
      message.error(
        `至少需要${analysisParams.minParticipants}个参与者的情感数据`
      )
      return
    }

    setLoading(true)
    try {
      const response = await groupEmotionApi.analyzeGroupEmotion(
        participantEmotions,
        selectedGroupId
      )
      const state = mapResponseToState(response)
      setGroupState(state)
      setContagionEvents(mapContagionEvents(state?.contagion_patterns || []))
      message.success('群体情感分析完成')
      setShowAnalysisModal(false)
      await loadData()
    } catch (error) {
      logger.error('分析失败:', error)
      message.error('分析失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  const addParticipant = () => {
    const participantId = `user${Object.keys(participantEmotions).length + 1}`
    setParticipantEmotions({
      ...participantEmotions,
      [participantId]: {
        emotion: 'neutral',
        intensity: 0.5,
        valence: 0.0,
        arousal: 0.5,
        dominance: 0.5,
      },
    })
  }

  const updateParticipantEmotion = (
    participantId: string,
    field: string,
    value: any
  ) => {
    setParticipantEmotions({
      ...participantEmotions,
      [participantId]: {
        ...participantEmotions[participantId],
        [field]: value,
      },
    })
  }

  const removeParticipant = (participantId: string) => {
    const newEmotions = { ...participantEmotions }
    delete newEmotions[participantId]
    setParticipantEmotions(newEmotions)
  }

  const getEmotionColor = (emotion: string) => {
    const colorMap: Record<string, string> = {
      happiness: 'gold',
      joy: 'orange',
      excitement: 'volcano',
      sadness: 'blue',
      anger: 'red',
      fear: 'purple',
      anxiety: 'magenta',
      neutral: 'default',
      trust: 'green',
      surprise: 'cyan',
    }
    return colorMap[emotion] || 'default'
  }

  const renderCurrentState = () => (
    <Row gutter={24}>
      <Col span={8}>
        <Card
          title={
            <span>
              <TeamOutlined style={{ marginRight: 8 }} />
              群体概况
            </span>
          }
        >
          {groupState ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ textAlign: 'center' }}>
                <Tag
                  color={getEmotionColor(groupState.dominant_emotion)}
                  style={{ fontSize: '16px', padding: '8px 16px' }}
                >
                  {groupState.dominant_emotion}
                </Tag>
                <div style={{ marginTop: 8 }}>
                  <Text type="secondary">主导情感</Text>
                </div>
              </div>

              <Divider />

              <div>
                <Text strong>参与者数量: </Text>
                <Badge
                  count={groupState.participants.length}
                  style={{ backgroundColor: '#1890ff' }}
                />
              </div>

              <div>
                <Text strong>群体凝聚力: </Text>
                <Tag
                  color={
                    groupCohesionColors[groupState.group_cohesion] || 'default'
                  }
                >
                  {groupState.group_cohesion}
                </Tag>
              </div>

              <div>
                <Text>一致性水平: </Text>
                <Progress
                  percent={Math.round(groupState.consensus_level * 100)}
                  size="small"
                  strokeColor="#52c41a"
                />
              </div>

              <div>
                <Text>极化指数: </Text>
                <Progress
                  percent={Math.round(groupState.polarization_index * 100)}
                  size="small"
                  strokeColor={
                    groupState.polarization_index > 0.6 ? '#f5222d' : '#1890ff'
                  }
                />
              </div>
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
              <BarChartOutlined style={{ marginRight: 8 }} />
              情感分布
            </span>
          }
        >
          {groupState?.emotion_distribution ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              {Object.entries(groupState.emotion_distribution)
                .sort(([, a], [, b]) => b - a)
                .slice(0, 6)
                .map(([emotion, percentage]) => (
                  <div
                    key={emotion}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <Tag color={getEmotionColor(emotion)}>{emotion}</Tag>
                    <div
                      style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        flex: 1,
                        marginLeft: 12,
                      }}
                    >
                      <Progress
                        percent={Math.round((percentage as number) * 100)}
                        size="small"
                        style={{ flex: 1 }}
                      />
                      <Text style={{ minWidth: '40px', fontSize: '12px' }}>
                        {((percentage as number) * 100).toFixed(1)}%
                      </Text>
                    </div>
                  </div>
                ))}
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
              <FireOutlined style={{ marginRight: 8 }} />
              情感领导者
            </span>
          }
        >
          {groupState?.emotional_leaders ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              {groupState.emotional_leaders.map((leader, index) => (
                <div
                  key={leader.participant_id}
                  style={{ display: 'flex', alignItems: 'center', gap: 12 }}
                >
                  <Badge
                    count={index + 1}
                    style={{
                      backgroundColor:
                        index === 0
                          ? '#f5222d'
                          : index === 1
                            ? '#fa8c16'
                            : '#1890ff',
                    }}
                  />
                  <Text strong>{leader.participant_id}</Text>
                  {index === 0 && <Tag color="red">主导</Tag>}
                </div>
              ))}

              <Divider />

              <div>
                <Text strong>传播速度: </Text>
                <Text type="secondary">
                  {(groupState.contagion_velocity || 0).toFixed(2)} / min
                </Text>
              </div>

              <div>
                <Text strong>稳定性: </Text>
                <Progress
                  percent={Math.round((groupState.stability_score || 0) * 100)}
                  size="small"
                  strokeColor="#13c2c2"
                />
              </div>
            </Space>
          ) : (
            <Text type="secondary">暂无数据</Text>
          )}
        </Card>
      </Col>
    </Row>
  )

  const renderContagionEvents = () => {
    const columns = [
      {
        title: '事件ID',
        dataIndex: 'event_id',
        key: 'event_id',
        width: 120,
        render: (id: string) => <Text code>{id.slice(-8)}</Text>,
      },
      {
        title: '源参与者',
        dataIndex: 'source_participant',
        key: 'source_participant',
        render: (participant: string) => <Tag color="blue">{participant}</Tag>,
      },
      {
        title: '情感类型',
        dataIndex: 'emotion',
        key: 'emotion',
        render: (emotion: string) => (
          <Tag color={getEmotionColor(emotion)}>{emotion}</Tag>
        ),
      },
      {
        title: '传染类型',
        dataIndex: 'contagion_type',
        key: 'contagion_type',
        render: (type: string) => (
          <Tag color={contagionTypeColors[type] || 'default'}>{type}</Tag>
        ),
      },
      {
        title: '传染强度',
        dataIndex: 'strength',
        key: 'strength',
        render: (strength: number) => (
          <Progress
            percent={Math.round(strength * 100)}
            size="small"
            strokeColor={
              strength > 0.7
                ? '#f5222d'
                : strength > 0.4
                  ? '#fa8c16'
                  : '#52c41a'
            }
            style={{ width: 80 }}
          />
        ),
      },
      {
        title: '目标数量',
        dataIndex: 'target_participants',
        key: 'target_count',
        render: (targets: string[]) => (
          <Badge
            count={targets.length}
            style={{ backgroundColor: '#722ed1' }}
          />
        ),
      },
      {
        title: '发生时间',
        dataIndex: 'timestamp',
        key: 'timestamp',
        render: (timestamp: string) => (
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {new Date(timestamp).toLocaleString()}
          </Text>
        ),
      },
    ]

    return (
      <Card
        title={
          <span>
            <ThunderboltOutlined style={{ marginRight: 8 }} />
            情感传染事件 ({contagionEvents.length})
          </span>
        }
      >
        {contagionEvents.length > 0 ? (
          <Table
            columns={columns}
            dataSource={contagionEvents}
            rowKey="event_id"
            pagination={{ pageSize: 10 }}
            size="small"
          />
        ) : (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Text type="secondary">暂无传染事件记录</Text>
          </div>
        )}
      </Card>
    )
  }

  const renderAnalysisModal = () => (
    <Modal
      title="群体情感分析"
      open={showAnalysisModal}
      onCancel={() => setShowAnalysisModal(false)}
      footer={[
        <Button key="cancel" onClick={() => setShowAnalysisModal(false)}>
          取消
        </Button>,
        <Button key="add" onClick={addParticipant}>
          添加参与者
        </Button>,
        <Button
          key="analyze"
          type="primary"
          loading={loading}
          onClick={runGroupAnalysis}
          disabled={
            Object.keys(participantEmotions).length <
            analysisParams.minParticipants
          }
        >
          开始分析
        </Button>,
      ]}
      width={800}
    >
      <Space direction="vertical" style={{ width: '100%' }}>
        <Alert
          message="群体情感分析"
          description={`请添加至少${analysisParams.minParticipants}个参与者的情感状态数据进行分析`}
          type="info"
          showIcon
        />

        {Object.entries(participantEmotions).map(([participantId, emotion]) => (
          <Card
            key={participantId}
            size="small"
            title={participantId}
            extra={
              <Button
                type="text"
                danger
                size="small"
                onClick={() => removeParticipant(participantId)}
              >
                删除
              </Button>
            }
          >
            <Row gutter={16}>
              <Col span={6}>
                <Text strong>情感:</Text>
                <Select
                  style={{ width: '100%', marginTop: 4 }}
                  value={emotion.emotion}
                  onChange={value =>
                    updateParticipantEmotion(participantId, 'emotion', value)
                  }
                >
                  {emotions.map(e => (
                    <Option key={e} value={e}>
                      {e}
                    </Option>
                  ))}
                </Select>
              </Col>
              <Col span={6}>
                <Text strong>强度: {Math.round(emotion.intensity * 100)}%</Text>
                <InputNumber
                  style={{ width: '100%', marginTop: 4 }}
                  min={0}
                  max={1}
                  step={0.1}
                  value={emotion.intensity}
                  onChange={value =>
                    updateParticipantEmotion(
                      participantId,
                      'intensity',
                      value || 0
                    )
                  }
                />
              </Col>
              <Col span={4}>
                <Text strong>效价:</Text>
                <InputNumber
                  style={{ width: '100%', marginTop: 4 }}
                  min={-1}
                  max={1}
                  step={0.1}
                  value={emotion.valence}
                  onChange={value =>
                    updateParticipantEmotion(
                      participantId,
                      'valence',
                      value || 0
                    )
                  }
                />
              </Col>
              <Col span={4}>
                <Text strong>唤醒:</Text>
                <InputNumber
                  style={{ width: '100%', marginTop: 4 }}
                  min={0}
                  max={1}
                  step={0.1}
                  value={emotion.arousal}
                  onChange={value =>
                    updateParticipantEmotion(
                      participantId,
                      'arousal',
                      value || 0
                    )
                  }
                />
              </Col>
              <Col span={4}>
                <Text strong>支配:</Text>
                <InputNumber
                  style={{ width: '100%', marginTop: 4 }}
                  min={0}
                  max={1}
                  step={0.1}
                  value={emotion.dominance}
                  onChange={value =>
                    updateParticipantEmotion(
                      participantId,
                      'dominance',
                      value || 0
                    )
                  }
                />
              </Col>
            </Row>
          </Card>
        ))}

        {Object.keys(participantEmotions).length === 0 && (
          <div style={{ textAlign: 'center', padding: 40 }}>
            <Text type="secondary">暂无参与者，请点击"添加参与者"开始</Text>
          </div>
        )}
      </Space>
    </Modal>
  )

  return (
    <div style={{ padding: '24px' }}>
      <div
        style={{
          marginBottom: 24,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Title level={2}>
          <TeamOutlined style={{ marginRight: 12, color: '#1890ff' }} />
          群体情感分析引擎
        </Title>
        <Space>
          <Select
            style={{ width: 200 }}
            value={selectedGroupId}
            onChange={setSelectedGroupId}
            placeholder="选择群体"
          >
            <Option value="group_1">群体 1</Option>
            <Option value="group_2">群体 2</Option>
            <Option value="group_3">群体 3</Option>
          </Select>
          <Button
            type="primary"
            icon={<ExperimentOutlined />}
            onClick={() => setShowAnalysisModal(true)}
          >
            运行分析
          </Button>
          <Button icon={<SyncOutlined />} loading={loading} onClick={loadData}>
            刷新
          </Button>
        </Space>
      </div>

      <Tabs defaultActiveKey="overview">
        <TabPane tab="群体概览" key="overview">
          {renderCurrentState()}

          <div style={{ marginTop: 24 }}>
            <Card
              title={
                <span>
                  <LineChartOutlined style={{ marginRight: 8 }} />
                  群体质量指标
                </span>
              }
            >
              {groupState && (
                <Row gutter={24}>
                  <Col span={6}>
                    <div style={{ textAlign: 'center' }}>
                      <Progress
                        type="circle"
                        percent={Math.round(groupState.data_completeness * 100)}
                        strokeColor="#52c41a"
                        width={100}
                      />
                      <div style={{ marginTop: 8 }}>
                        <Text strong>数据完整性</Text>
                      </div>
                    </div>
                  </Col>
                  <Col span={6}>
                    <div style={{ textAlign: 'center' }}>
                      <Progress
                        type="circle"
                        percent={Math.round(
                          groupState.analysis_confidence * 100
                        )}
                        strokeColor="#1890ff"
                        width={100}
                      />
                      <div style={{ marginTop: 8 }}>
                        <Text strong>分析置信度</Text>
                      </div>
                    </div>
                  </Col>
                  <Col span={6}>
                    <div style={{ textAlign: 'center' }}>
                      <Progress
                        type="circle"
                        percent={Math.round(
                          groupState.emotional_volatility * 100
                        )}
                        strokeColor="#fa8c16"
                        width={100}
                      />
                      <div style={{ marginTop: 8 }}>
                        <Text strong>情感波动性</Text>
                      </div>
                    </div>
                  </Col>
                  <Col span={6}>
                    <div style={{ textAlign: 'center', color: '#8c8c8c' }}>
                      <div
                        style={{
                          width: 100,
                          height: 100,
                          borderRadius: '50%',
                          border: '6px solid #f0f0f0',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          margin: '0 auto',
                          fontSize: '24px',
                          fontWeight: 'bold',
                        }}
                      >
                        {groupState.participants.length}
                      </div>
                      <div style={{ marginTop: 8 }}>
                        <Text strong>参与者数量</Text>
                      </div>
                    </div>
                  </Col>
                </Row>
              )}
            </Card>
          </div>
        </TabPane>

        <TabPane tab="传染事件" key="contagion">
          {renderContagionEvents()}
        </TabPane>

        <TabPane tab="历史趋势" key="history">
          <Card title="群体情感历史">
            <Timeline>
              {groupHistory.slice(0, 8).map((state, index) => (
                <Timeline.Item
                  key={`${state.group_id}_${state.timestamp}`}
                  color={index === 0 ? 'green' : 'blue'}
                >
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                      flexWrap: 'wrap',
                    }}
                  >
                    <Tag color={getEmotionColor(state.dominant_emotion)}>
                      {state.dominant_emotion}
                    </Tag>
                    <Text>凝聚力: </Text>
                    <Tag
                      color={
                        groupCohesionColors[state.group_cohesion] || 'default'
                      }
                    >
                      {state.group_cohesion}
                    </Tag>
                    <Text>
                      一致性: {Math.round(state.consensus_level * 100)}%
                    </Text>
                    <Text type="secondary">
                      {new Date(state.timestamp).toLocaleString()}
                    </Text>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </TabPane>

        <TabPane tab="分析配置" key="config">
          <Card title="分析参数配置" style={{ maxWidth: 600 }}>
            <Form layout="vertical">
              <Form.Item label="时间窗口 (分钟)">
                <InputNumber
                  style={{ width: '100%' }}
                  value={analysisParams.timeWindow}
                  onChange={value =>
                    setAnalysisParams({
                      ...analysisParams,
                      timeWindow: value || 60,
                    })
                  }
                  min={5}
                  max={1440}
                />
              </Form.Item>

              <Form.Item label="最少参与者数量">
                <InputNumber
                  style={{ width: '100%' }}
                  value={analysisParams.minParticipants}
                  onChange={value =>
                    setAnalysisParams({
                      ...analysisParams,
                      minParticipants: value || 3,
                    })
                  }
                  min={2}
                  max={50}
                />
              </Form.Item>

              <Form.Item label="置信度阈值">
                <InputNumber
                  style={{ width: '100%' }}
                  value={analysisParams.confidenceThreshold}
                  onChange={value =>
                    setAnalysisParams({
                      ...analysisParams,
                      confidenceThreshold: value || 0.6,
                    })
                  }
                  min={0.1}
                  max={1.0}
                  step={0.1}
                />
              </Form.Item>

              <Form.Item>
                <Button type="primary" block onClick={loadData}>
                  应用配置并刷新
                </Button>
              </Form.Item>
            </Form>
          </Card>
        </TabPane>
      </Tabs>

      {renderAnalysisModal()}
    </div>
  )
}

export default GroupEmotionAnalysisPage
