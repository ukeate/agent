import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Row,
  Col,
  Statistic,
  Button,
  Table,
  Tag,
  Progress,
  Alert,
  Space,
  Modal,
  Form,
  Input,
  Select,
  Timeline,
  Divider,
  Typography,
  Spin,
  message,
} from 'antd'
import {
  BrainCircuitIcon,
  HeartHandshakeIcon,
  ShieldAlertIcon,
  TrendingUpIcon,
  MessageSquareIcon,
  UserCheckIcon,
  AlertTriangleIcon,
  ActivityIcon,
} from 'lucide-react'
import {
  emotionalIntelligenceService,
  type EmotionalDecision,
  type RiskAssessment,
  type SystemStats,
} from '../services/emotionalIntelligenceService'

const { Title, Text } = Typography
const { Option } = Select

const EmotionalIntelligenceDecisionEnginePage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [decisions, setDecisions] = useState<EmotionalDecision[]>([])
  const [riskAssessments, setRiskAssessments] = useState<RiskAssessment[]>([])
  const [decisionModalVisible, setDecisionModalVisible] = useState(false)
  const [riskModalVisible, setRiskModalVisible] = useState(false)
  const [currentDecision, setCurrentDecision] =
    useState<EmotionalDecision | null>(null)
  const [stats, setStats] = useState({
    totalDecisions: 0,
    averageConfidence: 0,
    highRiskUsers: 0,
    activeInterventions: 0,
  })

  // 新增状态 - 高级功能
  const [emotionPatterns, setEmotionPatterns] = useState<any>([])
  const [systemStatus, setSystemStatus] = useState<any>(null)
  const [activeTab, setActiveTab] = useState<
    'dashboard' | 'advanced' | 'analytics'
  >('dashboard')
  const [crisisPredictions, setCrisisPredictions] = useState<any>([])
  const [emotionStatistics, setEmotionStatistics] = useState<any>(null)
  const [selectedUserId, setSelectedUserId] = useState('')

  const [form] = Form.useForm()
  const [riskForm] = Form.useForm()

  // 加载数据
  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      // 加载系统统计
      const systemStats = await emotionalIntelligenceService.getSystemStats()
      setStats({
        totalDecisions: systemStats.total_decisions,
        averageConfidence: systemStats.average_confidence,
        highRiskUsers: systemStats.high_risk_users,
        activeInterventions: systemStats.active_interventions,
      })

      // 加载最近决策
      const recentDecisions =
        await emotionalIntelligenceService.getDecisionHistory('all', 10)
      setDecisions(recentDecisions)

      // 加载高风险用户
      const highRiskUsers =
        await emotionalIntelligenceService.getHighRiskUsers(0.6)

      // 加载风险评估
      const riskAssessments = await Promise.all(
        highRiskUsers.slice(0, 5).map(user =>
          emotionalIntelligenceService.assessRisk({
            user_id: user.user_id,
            emotion_history: [],
          })
        )
      )
      setRiskAssessments(riskAssessments)

      // 加载高级功能数据
      await loadAdvancedData(selectedUserId || undefined)
    } catch (error: any) {
      logger.error('加载数据失败:', error)
      message.error(error?.message || '加载数据失败')
    } finally {
      setLoading(false)
    }
  }

  // 加载高级功能数据
  const loadAdvancedData = async (userId?: string) => {
    try {
      // 获取系统状态
      const systemStatus =
        await emotionalIntelligenceService.getEmotionalIntelligenceSystemStatus()
      setSystemStatus(systemStatus)

      // 获取情感统计
      const emotionStats =
        await emotionalIntelligenceService.getEmotionStatistics()
      setEmotionStatistics(emotionStats)

      if (userId) {
        const patterns =
          await emotionalIntelligenceService.getEmotionalPatterns(userId)
        setEmotionPatterns([patterns])

        const crisisPred =
          await emotionalIntelligenceService.getCrisisPrediction(userId)
        setCrisisPredictions([crisisPred])
      } else {
        setEmotionPatterns([])
        setCrisisPredictions([])
      }
    } catch (error: any) {
      logger.error('加载高级数据失败:', error)
      message.error(error?.message || '加载高级数据失败')
    }
  }

  // 新增高级功能处理函数
  const handleAdvancedAction = async (action: string, params?: any) => {
    setLoading(true)
    try {
      let result
      switch (action) {
        case 'suicide_risk_assessment':
          result = await emotionalIntelligenceService.assessSuicideRisk(params)
          message.success('自杀风险评估完成')
          break
        case 'comprehensive_analysis':
          result =
            await emotionalIntelligenceService.performComprehensiveAnalysis(
              params
            )
          message.success('综合分析完成')
          break
        case 'emotion_prediction':
          result = await emotionalIntelligenceService.predictEmotion(params)
          message.success('情感预测完成')
          break
        case 'export_emotion_data':
          result = await emotionalIntelligenceService.exportEmotionData('json')
          message.success('情感数据导出完成')
          break
        case 'initialize_social_emotion':
          result =
            await emotionalIntelligenceService.initializeSocialEmotionSystem(
              params
            )
          message.success('社交情感系统初始化完成')
          break
        default:
          message.info('功能演示完成')
      }
      logger.log('高级功能执行结果:', result)

      // 重新加载数据
      await loadAdvancedData()
    } catch (error: any) {
      logger.error(`执行${action}失败:`, error)
      message.error(`执行失败: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  const handleMakeDecision = async (values: any) => {
    setLoading(true)
    try {
      // 构建情感状态
      const emotionState = {
        dominant_emotion: values.emotion_context || 'neutral',
        emotion_scores: {
          [values.emotion_context || 'neutral']: 0.8,
          neutral: 0.2,
        },
        valence:
          values.emotion_context === 'happy'
            ? 0.8
            : values.emotion_context === 'sad'
              ? -0.6
              : 0,
        arousal: values.emotion_context === 'anxious' ? 0.8 : 0.4,
        confidence: 0.85,
      }

      // 调用真实API
      const newDecision = await emotionalIntelligenceService.makeDecision({
        user_id: values.user_id,
        user_input: values.user_input,
        current_emotion_state: emotionState,
        emotion_history: [],
        environmental_factors: {},
      })

      setDecisions(prev => [newDecision, ...prev])
      setDecisionModalVisible(false)
      form.resetFields()
      message.success('决策制定成功')

      // 重新加载统计数据
      loadData()
    } catch (error: any) {
      logger.error('决策制定失败:', error)
      message.error(error.message || '决策制定失败')
    } finally {
      setLoading(false)
    }
  }

  const handleRiskAssessment = async (values: any) => {
    setLoading(true)
    try {
      const assessment = await emotionalIntelligenceService.assessRisk({
        user_id: values.user_id,
        emotion_history: [],
        context: values.assessment_type
          ? { assessment_type: values.assessment_type }
          : undefined,
      })
      setRiskAssessments(prev => [assessment, ...prev])
      message.success('风险评估完成')
      riskForm.resetFields()
      setRiskModalVisible(false)
    } catch (error: any) {
      logger.error('风险评估失败:', error)
      message.error(error?.message || '风险评估失败')
    } finally {
      setLoading(false)
    }
  }

  const decisionColumns = [
    {
      title: '决策ID',
      dataIndex: 'decision_id',
      key: 'decision_id',
      width: 120,
    },
    {
      title: '用户ID',
      dataIndex: 'user_id',
      key: 'user_id',
      width: 120,
    },
    {
      title: '选择策略',
      dataIndex: 'chosen_strategy',
      key: 'chosen_strategy',
      render: (strategy: string) => (
        <Tag color={strategy === 'supportive_strategy' ? 'green' : 'orange'}>
          {strategy}
        </Tag>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence_score',
      key: 'confidence_score',
      render: (score: number) => (
        <Progress
          percent={score * 100}
          size="small"
          status={
            score > 0.8 ? 'success' : score > 0.6 ? 'normal' : 'exception'
          }
        />
      ),
    },
    {
      title: '决策类型',
      dataIndex: 'decision_type',
      key: 'decision_type',
      render: (type: string) => {
        const colorMap: Record<string, string> = {
          supportive: 'green',
          corrective: 'orange',
          crisis: 'red',
        }
        return <Tag color={colorMap[type] || 'default'}>{type}</Tag>
      },
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: EmotionalDecision) => (
        <Space>
          <Button
            size="small"
            onClick={() => {
              setCurrentDecision(record)
              setDecisionModalVisible(true)
            }}
          >
            详情
          </Button>
        </Space>
      ),
    },
  ]

  const riskColumns = [
    {
      title: '评估ID',
      dataIndex: 'assessment_id',
      key: 'assessment_id',
      width: 120,
    },
    {
      title: '用户ID',
      dataIndex: 'user_id',
      key: 'user_id',
      width: 120,
    },
    {
      title: '风险等级',
      dataIndex: 'risk_level',
      key: 'risk_level',
      render: (level: string) => {
        const colorMap: Record<string, string> = {
          low: 'green',
          medium: 'orange',
          high: 'red',
          critical: 'purple',
        }
        return <Tag color={colorMap[level]}>{level.toUpperCase()}</Tag>
      },
    },
    {
      title: '风险分数',
      dataIndex: 'risk_score',
      key: 'risk_score',
      render: (score: number) => (
        <Progress
          percent={score * 100}
          size="small"
          status={
            score > 0.7 ? 'exception' : score > 0.4 ? 'active' : 'success'
          }
        />
      ),
    },
    {
      title: '预测置信度',
      dataIndex: 'prediction_confidence',
      key: 'prediction_confidence',
      render: (confidence: number) => `${(confidence * 100).toFixed(1)}%`,
    },
    {
      title: '推荐行动',
      dataIndex: 'recommended_actions',
      key: 'recommended_actions',
      render: (actions: string[]) => (
        <div>
          {actions.slice(0, 2).map((action, index) => (
            <Tag key={index} style={{ marginBottom: 4 }}>
              {action}
            </Tag>
          ))}
          {actions.length > 2 && (
            <Text type="secondary">+{actions.length - 2}更多</Text>
          )}
        </div>
      ),
    },
  ]

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card>
            <Space align="center">
              <BrainCircuitIcon size={32} />
              <div>
                <Title level={2} style={{ margin: 0 }}>
                  情感智能决策引擎
                </Title>
                <Text type="secondary">
                  基于情感状态的智能决策系统，提供个性化情感支持和风险预警
                </Text>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="决策总数"
              value={stats.totalDecisions}
              prefix={<ActivityIcon size={16} />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="平均置信度"
              value={(stats.averageConfidence * 100).toFixed(1)}
              suffix="%"
              prefix={<TrendingUpIcon size={16} />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="高风险用户"
              value={stats.highRiskUsers}
              prefix={<AlertTriangleIcon size={16} />}
              valueStyle={{ color: '#f5222d' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="活跃干预"
              value={stats.activeInterventions}
              prefix={<HeartHandshakeIcon size={16} />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 系统状态提醒 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Alert
            message={
              systemStatus?.system_status ? '系统状态已更新' : '系统状态未加载'
            }
            description={
              systemStatus?.system_status
                ? `决策记录 ${systemStatus.system_status.decision_history_count}，活跃干预 ${systemStatus.system_status.active_interventions}，过去24小时危机事件 ${systemStatus.system_status.crisis_events_24h}`
                : '请加载系统状态以查看运行指标'
            }
            type={systemStatus?.system_status ? 'success' : 'warning'}
            showIcon
            style={{ marginBottom: 16 }}
          />
        </Col>
      </Row>

      {/* 操作按钮 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Space>
            <Button
              type="primary"
              icon={<BrainCircuitIcon size={16} />}
              onClick={() => setDecisionModalVisible(true)}
            >
              新建决策
            </Button>
            <Button
              icon={<ShieldAlertIcon size={16} />}
              onClick={() => setRiskModalVisible(true)}
            >
              风险评估
            </Button>
            <Button icon={<UserCheckIcon size={16} />}>健康监测</Button>
            <Button icon={<MessageSquareIcon size={16} />}>干预管理</Button>
            <Input
              value={selectedUserId}
              onChange={e => setSelectedUserId(e.target.value)}
              placeholder="输入用户ID加载高级数据"
              style={{ width: 220 }}
            />
            <Button
              onClick={() => loadAdvancedData(selectedUserId || undefined)}
              disabled={!selectedUserId}
            >
              加载高级数据
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 主要内容区域 */}
      <Row gutter={[16, 16]}>
        {/* 决策历史 */}
        <Col span={24}>
          <Card
            title={
              <Space>
                <ActivityIcon size={20} />
                最近决策
              </Space>
            }
            extra={<Button type="link">查看全部</Button>}
          >
            <Table
              dataSource={decisions}
              columns={decisionColumns}
              rowKey="decision_id"
              pagination={{ pageSize: 5 }}
              size="small"
            />
          </Card>
        </Col>

        {/* 风险评估 */}
        <Col span={24}>
          <Card
            title={
              <Space>
                <ShieldAlertIcon size={20} />
                风险评估
              </Space>
            }
            extra={<Button type="link">查看全部</Button>}
          >
            <Table
              dataSource={riskAssessments}
              columns={riskColumns}
              rowKey="assessment_id"
              pagination={{ pageSize: 5 }}
              size="small"
            />
          </Card>
        </Col>
      </Row>

      {/* 新建决策模态框 */}
      <Modal
        title="新建情感决策"
        open={decisionModalVisible}
        onCancel={() => {
          setDecisionModalVisible(false)
          setCurrentDecision(null)
          form.resetFields()
        }}
        footer={null}
        width={600}
      >
        {currentDecision ? (
          // 显示决策详情
          <div>
            <Title level={4}>决策详情</Title>
            <Divider />
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Text strong>决策ID: </Text>
                <Text>{currentDecision.decision_id}</Text>
              </Col>
              <Col span={12}>
                <Text strong>用户ID: </Text>
                <Text>{currentDecision.user_id}</Text>
              </Col>
              <Col span={12}>
                <Text strong>选择策略: </Text>
                <Tag>{currentDecision.chosen_strategy}</Tag>
              </Col>
              <Col span={12}>
                <Text strong>置信度: </Text>
                <Progress
                  percent={currentDecision.confidence_score * 100}
                  size="small"
                  style={{ width: 100 }}
                />
              </Col>
              <Col span={24}>
                <Text strong>决策推理过程: </Text>
                <Timeline style={{ marginTop: 8 }}>
                  {currentDecision.reasoning.map((reason, index) => (
                    <Timeline.Item key={index}>{reason}</Timeline.Item>
                  ))}
                </Timeline>
              </Col>
            </Row>
          </div>
        ) : (
          // 新建决策表单
          <Form form={form} layout="vertical" onFinish={handleMakeDecision}>
            <Form.Item
              name="user_id"
              label="用户ID"
              rules={[{ required: true, message: '请输入用户ID' }]}
            >
              <Input placeholder="输入用户ID" />
            </Form.Item>

            <Form.Item
              name="user_input"
              label="用户输入"
              rules={[{ required: true, message: '请输入用户消息' }]}
            >
              <Input.TextArea placeholder="输入用户的消息内容..." rows={3} />
            </Form.Item>

            <Form.Item name="emotion_context" label="情感上下文">
              <Select placeholder="选择情感状态">
                <Option value="happy">开心</Option>
                <Option value="sad">难过</Option>
                <Option value="anxious">焦虑</Option>
                <Option value="angry">愤怒</Option>
                <Option value="neutral">中性</Option>
              </Select>
            </Form.Item>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" loading={loading}>
                  制定决策
                </Button>
                <Button onClick={() => setDecisionModalVisible(false)}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        )}
      </Modal>

      {/* 风险评估模态框 */}
      <Modal
        title="执行风险评估"
        open={riskModalVisible}
        onCancel={() => setRiskModalVisible(false)}
        footer={null}
      >
        <Spin spinning={loading}>
          <Form
            form={riskForm}
            layout="vertical"
            onFinish={handleRiskAssessment}
          >
            <Form.Item
              name="user_id"
              label="目标用户"
              rules={[{ required: true, message: '请输入用户ID' }]}
            >
              <Input placeholder="输入用户ID" />
            </Form.Item>
            <Form.Item name="assessment_type" label="评估类型">
              <Select placeholder="选择评估类型">
                <Option value="comprehensive">综合评估</Option>
                <Option value="crisis">危机评估</Option>
                <Option value="trend">趋势分析</Option>
              </Select>
            </Form.Item>
            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" loading={loading}>
                  开始评估
                </Button>
                <Button onClick={() => setRiskModalVisible(false)}>取消</Button>
              </Space>
            </Form.Item>
          </Form>
        </Spin>
      </Modal>
    </div>
  )
}

export default EmotionalIntelligenceDecisionEnginePage
