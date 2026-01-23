import { buildApiUrl, apiFetch } from '../utils/apiBase'
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
  Switch,
  Slider,
  List,
  Avatar,
  Rate,
  Checkbox,
  Tree,
  Tooltip,
} from 'antd'
import {
  GlobalOutlined,
  BookOutlined,
  UsergroupAddOutlined,
  LineChartOutlined,
  BarChartOutlined,
  NodeIndexOutlined,
  ExperimentOutlined,
  SyncOutlined,
  AlertOutlined,
  BulbOutlined,
  SettingOutlined,
  EyeOutlined,
  ThunderboltOutlined,
  FlagOutlined,
  CompassOutlined,
  TranslationOutlined,
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { TabPane } = Tabs
const { Option } = Select

// 文化背景类型定义
interface CulturalProfile {
  profile_id: string
  culture_id: string
  culture_name: string
  cultural_dimensions: {
    power_distance: number
    individualism_collectivism: number
    uncertainty_avoidance: number
    masculinity_femininity: number
    long_term_orientation: number
    indulgence_restraint: number
  }
  communication_patterns: {
    directness_level: number
    context_dependency: number
    silence_tolerance: number
    emotion_expression: number
    conflict_approach: string
  }
  social_norms: Array<{
    norm_type: string
    importance_level: number
    description: string
    violation_consequences: string
  }>
  behavioral_expectations: Record<string, number>
  taboo_behaviors: string[]
  preferred_interaction_styles: string[]
  time_orientation: string
  space_boundaries: Record<string, number>
  gift_giving_customs: Array<{
    occasion: string
    appropriate_gifts: string[]
    inappropriate_gifts: string[]
  }>
  business_etiquette: Record<string, string>
  created_timestamp: string
}

interface CulturalGap {
  gap_id: string
  user_culture: string
  target_culture: string
  dimension_differences: Record<string, number>
  communication_barriers: Array<{
    barrier_type: string
    severity: number
    description: string
    potential_solutions: string[]
  }>
  behavioral_conflicts: Array<{
    conflict_type: string
    risk_level: number
    description: string
    mitigation_strategies: string[]
  }>
  adaptation_priorities: Array<{
    priority_area: string
    importance: number
    urgency: number
    complexity: number
  }>
  success_probability: number
  estimated_adaptation_time: number
  analysis_timestamp: string
}

interface CulturalAdaptationPlan {
  plan_id: string
  user_id: string
  target_culture: string
  adaptation_phases: Array<{
    phase_number: number
    phase_name: string
    duration_weeks: number
    learning_objectives: string[]
    key_activities: string[]
    success_criteria: string[]
    resources_needed: string[]
  }>
  cultural_mentors: Array<{
    mentor_id: string
    expertise_areas: string[]
    availability: string
    rating: number
  }>
  progress_tracking: {
    current_phase: number
    completion_percentage: number
    skills_acquired: string[]
    remaining_challenges: string[]
  }
  adaptation_strategies: Record<string, any>
  created_timestamp: string
  last_updated: string
}

// 真实API客户端
const culturalApi = {
  async analyzeCulturalAdaptation(participants: any[], socialEnv: any) {
    const response = await apiFetch(
      buildApiUrl(
        '/api/v1/social-emotional-understanding/analyze/cultural-adaptation'
      ),
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: `session_${Date.now()}`,
          participants,
          social_environment: socialEnv,
          real_time: false,
        }),
      }
    )
    return await response.json()
  },
  async getCulturalProfile(cultureId: string) {
    const response = await apiFetch(
      buildApiUrl(
        `/api/v1/social-emotional-understanding/status?culture_id=${cultureId}`
      )
    )
    return await response.json()
  },
}

const CulturalAdaptationPage: React.FC = () => {
  const [culturalProfile, setCulturalProfile] =
    useState<CulturalProfile | null>(null)
  const [culturalGap, setCulturalGap] = useState<CulturalGap | null>(null)
  const [adaptationPlan, setAdaptationPlan] =
    useState<CulturalAdaptationPlan | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedUserCulture, setSelectedUserCulture] = useState('chinese')
  const [selectedTargetCulture, setSelectedTargetCulture] = useState('american')
  const [analysisResult, setAnalysisResult] = useState<any>(null)

  // 模态框状态
  const [showGapAnalysisModal, setShowGapAnalysisModal] = useState(false)
  const [showPlanModal, setShowPlanModal] = useState(false)
  const [gapAnalysisForm] = Form.useForm()
  const [planForm] = Form.useForm()

  // 文化列表
  const cultures = [
    { id: 'chinese', name: '中国文化', flag: '🇨🇳' },
    { id: 'american', name: '美国文化', flag: '🇺🇸' },
    { id: 'japanese', name: '日本文化', flag: '🇯🇵' },
    { id: 'german', name: '德国文化', flag: '🇩🇪' },
    { id: 'british', name: '英国文化', flag: '🇬🇧' },
    { id: 'french', name: '法国文化', flag: '🇫🇷' },
    { id: 'indian', name: '印度文化', flag: '🇮🇳' },
    { id: 'arabic', name: '阿拉伯文化', flag: '🇸🇦' },
    { id: 'brazilian', name: '巴西文化', flag: '🇧🇷' },
    { id: 'korean', name: '韩国文化', flag: '🇰🇷' },
  ]

  const culturalDimensions = {
    power_distance: '权力距离',
    individualism_collectivism: '个人主义-集体主义',
    uncertainty_avoidance: '不确定性规避',
    masculinity_femininity: '男性化-女性化',
    long_term_orientation: '长期导向',
    indulgence_restraint: '放纵-克制',
  }

  const communicationPatterns = {
    directness_level: '直接程度',
    context_dependency: '语境依赖',
    silence_tolerance: '沉默容忍',
    emotion_expression: '情感表达',
  }

  useEffect(() => {
    loadData()
  }, [selectedUserCulture, selectedTargetCulture])

  const loadData = async () => {
    setLoading(true)
    try {
      await analyze()
    } catch (error) {
      logger.error('加载数据失败:', error)
      message.error('加载失败')
    } finally {
      setLoading(false)
    }
  }
  const analyze = async () => {
    const participants = [
      {
        participant_id: 'user_1',
        name: 'User',
        emotion_data: {
          emotions: { joy: 0.4, neutral: 0.6 },
          intensity: 0.5,
          confidence: 0.9,
          context: 'conversation',
        },
        cultural_indicators: { culture_id: selectedTargetCulture },
      },
    ]

    const socialEnv = {
      scenario: 'business_meeting',
      participants_count: participants.length,
      formality_level: 0.6,
      emotional_intensity: 0.4,
      time_pressure: 0.3,
      cultural_context: selectedTargetCulture,
    }

    const response = await culturalApi.analyzeCulturalAdaptation(
      participants,
      socialEnv
    )
    setAnalysisResult(response)

    // 映射为现有视图所需结构
    const primaryCulture =
      response.cultural_analysis?.primary_culture || selectedTargetCulture
    setCulturalProfile({
      profile_id: `profile_${primaryCulture}`,
      culture_id: primaryCulture,
      culture_name:
        cultures.find(c => c.id === primaryCulture)?.name || primaryCulture,
      cultural_dimensions: {
        power_distance: 0.5,
        individualism_collectivism: 0.5,
        uncertainty_avoidance: 0.5,
        masculinity_femininity: 0.5,
        long_term_orientation: 0.5,
        indulgence_restraint: 0.5,
      },
      communication_patterns: {
        directness_level: 0.5,
        context_dependency: 0.5,
        silence_tolerance: 0.5,
        emotion_expression: 0.5,
        conflict_approach: 'balanced',
      },
      social_norms: [],
      behavioral_expectations: {},
      taboo_behaviors: [],
      preferred_interaction_styles: [],
      time_orientation: 'balanced',
      space_boundaries: {},
      gift_giving_customs: [],
      business_etiquette: {},
      created_timestamp: new Date().toISOString(),
    })

    setCulturalGap({
      gap_id: `gap_${selectedUserCulture}_${selectedTargetCulture}`,
      user_culture: selectedUserCulture,
      target_culture: selectedTargetCulture,
      dimension_differences: {},
      communication_barriers: [],
      behavioral_conflicts: [],
      adaptation_priorities: [],
      success_probability:
        response.cultural_analysis?.analysis_confidence || 0.5,
      estimated_adaptation_time: 8,
      analysis_timestamp: new Date().toISOString(),
    })

    setAdaptationPlan({
      plan_id: `plan_${Date.now()}`,
      user_id: 'user_1',
      target_culture: primaryCulture,
      adaptation_phases: [],
      cultural_mentors: [],
      progress_tracking: {
        current_phase: 0,
        completion_percentage: 0,
        skills_acquired: [],
        remaining_challenges: [],
      },
      adaptation_strategies: {
        recommendations: response.cross_cultural_recommendations || [],
      },
      created_timestamp: new Date().toISOString(),
      last_updated: new Date().toISOString(),
    })
  }

  const performGapAnalysis = async (values: any) => {
    setLoading(true)
    try {
      setSelectedUserCulture(values.user_culture)
      setSelectedTargetCulture(values.target_culture)
      await analyze()
      message.success('文化差异分析完成')
      setShowGapAnalysisModal(false)
    } catch (error) {
      logger.error('分析失败:', error)
      message.error('分析失败，请重试')
    } finally {
      setLoading(false)
    }
  }

  const getCultureFlag = (cultureId: string) => {
    return cultures.find(c => c.id === cultureId)?.flag || '🌍'
  }

  const getCultureName = (cultureId: string) => {
    return cultures.find(c => c.id === cultureId)?.name || cultureId
  }

  const getDimensionColor = (value: number) => {
    if (value >= 0.7) return '#f5222d'
    if (value >= 0.4) return '#fa8c16'
    return '#52c41a'
  }

  const renderCulturalProfile = () => (
    <Row gutter={24}>
      <Col span={12}>
        <Card
          title={
            <span>
              <FlagOutlined style={{ marginRight: 8 }} />
              文化维度分析
            </span>
          }
        >
          {culturalProfile ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              <div style={{ textAlign: 'center', marginBottom: 16 }}>
                <span style={{ fontSize: '32px' }}>
                  {getCultureFlag(culturalProfile.culture_id)}
                </span>
                <div style={{ marginTop: 8 }}>
                  <Text strong style={{ fontSize: '16px' }}>
                    {culturalProfile.culture_name}
                  </Text>
                </div>
              </div>

              <Divider />

              {Object.entries(culturalProfile.cultural_dimensions).map(
                ([dimension, value]) => (
                  <div key={dimension}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: 4,
                      }}
                    >
                      <Text strong style={{ fontSize: '13px' }}>
                        {
                          culturalDimensions[
                            dimension as keyof typeof culturalDimensions
                          ]
                        }
                        :
                      </Text>
                      <Text code>{(value * 100).toFixed(0)}%</Text>
                    </div>
                    <Progress
                      percent={Math.round(value * 100)}
                      size="small"
                      strokeColor={getDimensionColor(value)}
                      style={{ marginBottom: 8 }}
                    />
                  </div>
                )
              )}
            </Space>
          ) : (
            <Text type="secondary">暂无文化档案数据</Text>
          )}
        </Card>
      </Col>

      <Col span={12}>
        <Card
          title={
            <span>
              <TranslationOutlined style={{ marginRight: 8 }} />
              沟通模式
            </span>
          }
        >
          {culturalProfile ? (
            <Space direction="vertical" style={{ width: '100%' }}>
              {Object.entries(culturalProfile.communication_patterns)
                .filter(([key]) => key !== 'conflict_approach')
                .map(([pattern, value]) => (
                  <div key={pattern}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        marginBottom: 4,
                      }}
                    >
                      <Text strong style={{ fontSize: '13px' }}>
                        {
                          communicationPatterns[
                            pattern as keyof typeof communicationPatterns
                          ]
                        }
                        :
                      </Text>
                      <Text code>{((value as number) * 100).toFixed(0)}%</Text>
                    </div>
                    <Progress
                      percent={Math.round((value as number) * 100)}
                      size="small"
                      strokeColor="#1890ff"
                      style={{ marginBottom: 8 }}
                    />
                  </div>
                ))}

              <Divider />

              <div>
                <Text strong>冲突处理方式: </Text>
                <Tag
                  color={
                    culturalProfile.communication_patterns.conflict_approach ===
                    'direct'
                      ? 'red'
                      : 'blue'
                  }
                >
                  {culturalProfile.communication_patterns.conflict_approach}
                </Tag>
              </div>

              <div>
                <Text strong>时间观念: </Text>
                <Tag color="green">{culturalProfile.time_orientation}</Tag>
              </div>
            </Space>
          ) : (
            <Text type="secondary">暂无沟通模式数据</Text>
          )}
        </Card>
      </Col>
    </Row>
  )

  const renderCulturalGap = () => {
    if (!culturalGap) return null

    const barrierColumns = [
      {
        title: '障碍类型',
        dataIndex: 'barrier_type',
        key: 'barrier_type',
        render: (type: string) => <Tag color="orange">{type}</Tag>,
      },
      {
        title: '严重程度',
        dataIndex: 'severity',
        key: 'severity',
        render: (severity: number) => (
          <Progress
            percent={Math.round(severity * 100)}
            size="small"
            strokeColor={
              severity > 0.7
                ? '#f5222d'
                : severity > 0.4
                  ? '#fa8c16'
                  : '#52c41a'
            }
            style={{ width: 100 }}
          />
        ),
      },
      {
        title: '描述',
        dataIndex: 'description',
        key: 'description',
        ellipsis: true,
      },
      {
        title: '解决方案数',
        dataIndex: 'potential_solutions',
        key: 'solutions_count',
        render: (solutions: string[]) => (
          <Badge
            count={solutions.length}
            style={{ backgroundColor: '#1890ff' }}
          />
        ),
      },
    ]

    const conflictColumns = [
      {
        title: '冲突类型',
        dataIndex: 'conflict_type',
        key: 'conflict_type',
        render: (type: string) => <Tag color="red">{type}</Tag>,
      },
      {
        title: '风险等级',
        dataIndex: 'risk_level',
        key: 'risk_level',
        render: (level: number) => (
          <Progress
            percent={Math.round(level * 100)}
            size="small"
            strokeColor={level > 0.7 ? '#f5222d' : '#fa8c16'}
            style={{ width: 100 }}
          />
        ),
      },
      {
        title: '描述',
        dataIndex: 'description',
        key: 'description',
        ellipsis: true,
      },
      {
        title: '缓解策略数',
        dataIndex: 'mitigation_strategies',
        key: 'strategies_count',
        render: (strategies: string[]) => (
          <Badge
            count={strategies.length}
            style={{ backgroundColor: '#52c41a' }}
          />
        ),
      },
    ]

    return (
      <div>
        <Row gutter={24} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <div style={{ textAlign: 'center' }}>
                <Progress
                  type="circle"
                  percent={Math.round(culturalGap.success_probability * 100)}
                  strokeColor="#52c41a"
                  width={100}
                />
                <div style={{ marginTop: 8 }}>
                  <Text strong>成功概率</Text>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <div style={{ textAlign: 'center' }}>
                <div
                  style={{
                    fontSize: '36px',
                    fontWeight: 'bold',
                    color: '#1890ff',
                    lineHeight: 1,
                  }}
                >
                  {culturalGap.estimated_adaptation_time}
                </div>
                <div style={{ marginTop: 8 }}>
                  <Text strong>预计周数</Text>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <div style={{ textAlign: 'center' }}>
                <div
                  style={{
                    fontSize: '36px',
                    fontWeight: 'bold',
                    color: '#fa8c16',
                    lineHeight: 1,
                  }}
                >
                  {culturalGap.communication_barriers.length}
                </div>
                <div style={{ marginTop: 8 }}>
                  <Text strong>沟通障碍</Text>
                </div>
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <div style={{ textAlign: 'center' }}>
                <div
                  style={{
                    fontSize: '36px',
                    fontWeight: 'bold',
                    color: '#f5222d',
                    lineHeight: 1,
                  }}
                >
                  {culturalGap.behavioral_conflicts.length}
                </div>
                <div style={{ marginTop: 8 }}>
                  <Text strong>行为冲突</Text>
                </div>
              </div>
            </Card>
          </Col>
        </Row>

        <Row gutter={24}>
          <Col span={12}>
            <Card title="沟通障碍分析" size="small">
              <Table
                columns={barrierColumns}
                dataSource={culturalGap.communication_barriers}
                rowKey="barrier_type"
                pagination={false}
                size="small"
              />
            </Card>
          </Col>
          <Col span={12}>
            <Card title="行为冲突分析" size="small">
              <Table
                columns={conflictColumns}
                dataSource={culturalGap.behavioral_conflicts}
                rowKey="conflict_type"
                pagination={false}
                size="small"
              />
            </Card>
          </Col>
        </Row>
      </div>
    )
  }

  const renderAdaptationPlan = () => {
    if (!adaptationPlan) return null

    const currentPhase = adaptationPlan.adaptation_phases.find(
      phase =>
        phase.phase_number === adaptationPlan.progress_tracking.current_phase
    )

    return (
      <div>
        <Row gutter={24} style={{ marginBottom: 24 }}>
          <Col span={16}>
            <Card title="适应阶段进度">
              <div style={{ marginBottom: 16 }}>
                <Text strong>当前阶段: </Text>
                <Tag color="blue">
                  阶段 {adaptationPlan.progress_tracking.current_phase}:{' '}
                  {currentPhase?.phase_name}
                </Tag>
                <Text style={{ marginLeft: 16 }}>
                  总体进度:{' '}
                  {adaptationPlan.progress_tracking.completion_percentage}%
                </Text>
              </div>

              <Progress
                percent={adaptationPlan.progress_tracking.completion_percentage}
                strokeColor="#1890ff"
                style={{ marginBottom: 16 }}
              />

              <Timeline>
                {adaptationPlan.adaptation_phases.map(phase => (
                  <Timeline.Item
                    key={phase.phase_number}
                    color={
                      phase.phase_number <
                      adaptationPlan.progress_tracking.current_phase
                        ? 'green'
                        : phase.phase_number ===
                            adaptationPlan.progress_tracking.current_phase
                          ? 'blue'
                          : 'gray'
                    }
                  >
                    <div>
                      <Text strong>
                        阶段 {phase.phase_number}: {phase.phase_name}
                      </Text>
                      <Text type="secondary" style={{ marginLeft: 12 }}>
                        ({phase.duration_weeks} 周)
                      </Text>
                      <div style={{ marginTop: 8 }}>
                        <Text style={{ fontSize: '13px' }}>
                          学习目标:{' '}
                          {phase.learning_objectives.slice(0, 2).join(', ')}
                          {phase.learning_objectives.length > 2 && '...'}
                        </Text>
                      </div>
                    </div>
                  </Timeline.Item>
                ))}
              </Timeline>
            </Card>
          </Col>

          <Col span={8}>
            <Card title="技能掌握情况" style={{ marginBottom: 16 }}>
              <div style={{ marginBottom: 16 }}>
                <Text strong style={{ color: '#52c41a' }}>
                  已掌握技能:
                </Text>
                <div style={{ marginTop: 8 }}>
                  {adaptationPlan.progress_tracking.skills_acquired.map(
                    (skill, index) => (
                      <Tag key={index} color="green" style={{ margin: '2px' }}>
                        {skill}
                      </Tag>
                    )
                  )}
                </div>
              </div>

              <Divider />

              <div>
                <Text strong style={{ color: '#fa8c16' }}>
                  待改进领域:
                </Text>
                <div style={{ marginTop: 8 }}>
                  {adaptationPlan.progress_tracking.remaining_challenges.map(
                    (challenge, index) => (
                      <Tag key={index} color="orange" style={{ margin: '2px' }}>
                        {challenge}
                      </Tag>
                    )
                  )}
                </div>
              </div>
            </Card>

            <Card title="文化导师" size="small">
              {adaptationPlan.cultural_mentors.map(mentor => (
                <div
                  key={mentor.mentor_id}
                  style={{
                    marginBottom: 12,
                    padding: 8,
                    backgroundColor: '#fafafa',
                    borderRadius: 4,
                  }}
                >
                  <div
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                    }}
                  >
                    <Text strong>{mentor.mentor_id}</Text>
                    <Rate
                      disabled
                      value={mentor.rating}
                      style={{ fontSize: '12px' }}
                    />
                  </div>
                  <div style={{ marginTop: 4 }}>
                    <Text style={{ fontSize: '12px' }}>
                      专长: {mentor.expertise_areas.join(', ')}
                    </Text>
                  </div>
                  <div style={{ marginTop: 4 }}>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      可用时间: {mentor.availability}
                    </Text>
                  </div>
                </div>
              ))}
            </Card>
          </Col>
        </Row>
      </div>
    )
  }

  const renderGapAnalysisModal = () => (
    <Modal
      title="文化差异分析"
      open={showGapAnalysisModal}
      onCancel={() => setShowGapAnalysisModal(false)}
      footer={[
        <Button key="cancel" onClick={() => setShowGapAnalysisModal(false)}>
          取消
        </Button>,
        <Button
          key="analyze"
          type="primary"
          loading={loading}
          onClick={() => gapAnalysisForm.submit()}
        >
          开始分析
        </Button>,
      ]}
      width={600}
    >
      <Form
        form={gapAnalysisForm}
        layout="vertical"
        onFinish={performGapAnalysis}
        initialValues={{
          user_culture: selectedUserCulture,
          target_culture: selectedTargetCulture,
        }}
      >
        <Alert
          message="文化差异分析"
          description="分析您的文化背景与目标文化之间的差异，为制定适应策略提供依据"
          type="info"
          showIcon
          style={{ marginBottom: 24 }}
        />

        <Row gutter={16}>
          <Col span={12}>
            <Form.Item
              label="您的文化背景"
              name="user_culture"
              rules={[{ required: true, message: '请选择您的文化背景' }]}
            >
              <Select>
                {cultures.map(culture => (
                  <Option key={culture.id} value={culture.id}>
                    <span style={{ marginRight: 8 }}>{culture.flag}</span>
                    {culture.name}
                  </Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
          <Col span={12}>
            <Form.Item
              label="目标文化"
              name="target_culture"
              rules={[{ required: true, message: '请选择目标文化' }]}
            >
              <Select>
                {cultures.map(culture => (
                  <Option key={culture.id} value={culture.id}>
                    <span style={{ marginRight: 8 }}>{culture.flag}</span>
                    {culture.name}
                  </Option>
                ))}
              </Select>
            </Form.Item>
          </Col>
        </Row>

        <Form.Item label="主要交流场景" name="interaction_contexts">
          <Checkbox.Group>
            <Row>
              <Col span={12}>
                <Checkbox value="business">商务场合</Checkbox>
              </Col>
              <Col span={12}>
                <Checkbox value="social">社交场合</Checkbox>
              </Col>
              <Col span={12}>
                <Checkbox value="academic">学术环境</Checkbox>
              </Col>
              <Col span={12}>
                <Checkbox value="daily">日常生活</Checkbox>
              </Col>
              <Col span={12}>
                <Checkbox value="family">家庭环境</Checkbox>
              </Col>
              <Col span={12}>
                <Checkbox value="online">网络交流</Checkbox>
              </Col>
            </Row>
          </Checkbox.Group>
        </Form.Item>

        <Form.Item label="适应紧急程度" name="urgency_level" initialValue={5}>
          <Slider
            min={1}
            max={10}
            marks={{
              1: '不急',
              5: '一般',
              10: '非常急',
            }}
          />
        </Form.Item>

        <Form.Item label="特殊需求或关注点" name="special_requirements">
          <TextArea
            rows={3}
            placeholder="描述您在文化适应中的特殊需求或特别关注的方面..."
          />
        </Form.Item>
      </Form>
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
          <CompassOutlined style={{ marginRight: 12, color: '#1890ff' }} />
          文化背景适应引擎
        </Title>
        <Space>
          <Text>
            <span style={{ marginRight: 8 }}>
              {getCultureFlag(selectedUserCulture)}
            </span>
            {getCultureName(selectedUserCulture)}
          </Text>
          <Text type="secondary">→</Text>
          <Text>
            <span style={{ marginRight: 8 }}>
              {getCultureFlag(selectedTargetCulture)}
            </span>
            {getCultureName(selectedTargetCulture)}
          </Text>
          <Button
            type="primary"
            icon={<ExperimentOutlined />}
            onClick={() => setShowGapAnalysisModal(true)}
          >
            差异分析
          </Button>
          <Button icon={<SyncOutlined />} loading={loading} onClick={loadData}>
            刷新
          </Button>
        </Space>
      </div>

      <Tabs defaultActiveKey="profile">
        <TabPane tab="文化档案" key="profile">
          {renderCulturalProfile()}

          {culturalProfile && (
            <div style={{ marginTop: 24 }}>
              <Row gutter={24}>
                <Col span={12}>
                  <Card title="社会规范" size="small">
                    <List
                      size="small"
                      dataSource={culturalProfile.social_norms}
                      renderItem={norm => (
                        <List.Item>
                          <div style={{ width: '100%' }}>
                            <div
                              style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                alignItems: 'center',
                              }}
                            >
                              <Text strong>{norm.norm_type}</Text>
                              <Progress
                                percent={Math.round(
                                  norm.importance_level * 100
                                )}
                                size="small"
                                style={{ width: 100 }}
                              />
                            </div>
                            <Text style={{ fontSize: '12px', color: '#666' }}>
                              {norm.description}
                            </Text>
                          </div>
                        </List.Item>
                      )}
                    />
                  </Card>
                </Col>
                <Col span={12}>
                  <Card title="禁忌行为" size="small">
                    <List
                      size="small"
                      dataSource={culturalProfile.taboo_behaviors}
                      renderItem={taboo => (
                        <List.Item>
                          <AlertOutlined
                            style={{ color: '#f5222d', marginRight: 8 }}
                          />
                          <Text style={{ fontSize: '13px' }}>{taboo}</Text>
                        </List.Item>
                      )}
                    />
                  </Card>
                </Col>
              </Row>
            </div>
          )}
        </TabPane>

        <TabPane tab="差异分析" key="gap">
          {culturalGap ? (
            renderCulturalGap()
          ) : (
            <div style={{ textAlign: 'center', padding: 60 }}>
              <Text type="secondary">暂无差异分析数据，请先进行分析</Text>
              <div style={{ marginTop: 16 }}>
                <Button
                  type="primary"
                  icon={<ExperimentOutlined />}
                  onClick={() => setShowGapAnalysisModal(true)}
                >
                  开始差异分析
                </Button>
              </div>
            </div>
          )}
        </TabPane>

        <TabPane tab="适应计划" key="plan">
          {adaptationPlan ? (
            renderAdaptationPlan()
          ) : (
            <div style={{ textAlign: 'center', padding: 60 }}>
              <Text type="secondary">暂无适应计划，请先进行差异分析</Text>
            </div>
          )}
        </TabPane>

        <TabPane tab="资源库" key="resources">
          <Card title="学习资源" style={{ marginBottom: 16 }}>
            <Alert
              message="文化学习资源"
              description="这里将提供相关的文化学习材料、视频课程、实践指南等资源"
              type="info"
              showIcon
            />
          </Card>

          <Row gutter={24}>
            <Col span={12}>
              <Card title="在线课程" size="small">
                <div style={{ textAlign: 'center', padding: 40 }}>
                  <BookOutlined style={{ fontSize: '48px', color: '#ccc' }} />
                  <div style={{ marginTop: 12 }}>
                    <Text type="secondary">课程资源即将上线</Text>
                  </div>
                </div>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="实践机会" size="small">
                <div style={{ textAlign: 'center', padding: 40 }}>
                  <UsergroupAddOutlined
                    style={{ fontSize: '48px', color: '#ccc' }}
                  />
                  <div style={{ marginTop: 12 }}>
                    <Text type="secondary">实践机会即将推出</Text>
                  </div>
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {renderGapAnalysisModal()}
    </div>
  )
}

export default CulturalAdaptationPage
