import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Switch,
  message,
  Tabs,
  Alert,
  Progress,
  Statistic,
  Row,
  Col,
  Timeline,
  Badge,
  Tooltip,
  Divider,
  Drawer,
  List,
  Empty,
  Spin,
  Descriptions,
  Result,
  Steps,
} from 'antd'
import {
  WarningOutlined,
  SafetyOutlined,
  ThunderboltOutlined,
  RollbackOutlined,
  MonitorOutlined,
  BarChartOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined,
  ExclamationCircleOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  HistoryOutlined,
  DashboardOutlined,
  SettingOutlined,
  SyncOutlined,
  AlertOutlined,
  FireOutlined,
  BugOutlined,
  RocketOutlined,
  CaretUpOutlined,
  CaretDownOutlined,
} from '@ant-design/icons'
import {
  riskAssessmentService,
  RiskLevel,
  RiskCategory,
  RollbackStrategy,
  RiskAssessment,
  RiskFactor,
  RollbackPlan,
  RollbackExecution,
  RiskLevelInfo,
  RiskCategoryInfo,
  RollbackStrategyInfo,
} from '../services/riskAssessmentService'
import { Gauge, Line, Column, Pie } from '@ant-design/plots'

const { TabPane } = Tabs
const { Option } = Select
const { TextArea } = Input
const { Step } = Steps

const RiskAssessmentDashboard: React.FC = () => {
  // 状态管理
  const [loading, setLoading] = useState(false)
  const [currentAssessment, setCurrentAssessment] =
    useState<RiskAssessment | null>(null)
  const [riskHistory, setRiskHistory] = useState<any[]>([])
  const [rollbackPlans, setRollbackPlans] = useState<any[]>([])
  const [rollbackExecutions, setRollbackExecutions] = useState<
    RollbackExecution[]
  >([])
  const [riskLevels, setRiskLevels] = useState<RiskLevelInfo[]>([])
  const [riskCategories, setRiskCategories] = useState<RiskCategoryInfo[]>([])
  const [rollbackStrategies, setRollbackStrategies] = useState<
    RollbackStrategyInfo[]
  >([])
  const [riskThresholds, setRiskThresholds] = useState<any>({})

  // 弹窗状态
  const [assessModalVisible, setAssessModalVisible] = useState(false)
  const [rollbackModalVisible, setRollbackModalVisible] = useState(false)
  const [monitorModalVisible, setMonitorModalVisible] = useState(false)
  const [thresholdModalVisible, setThresholdModalVisible] = useState(false)
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false)

  // 表单
  const [assessForm] = Form.useForm()
  const [rollbackForm] = Form.useForm()
  const [monitorForm] = Form.useForm()
  const [thresholdForm] = Form.useForm()

  // 其他状态
  const [selectedExperimentId, setSelectedExperimentId] = useState<string>('')
  const [selectedRollbackPlan, setSelectedRollbackPlan] = useState<any>(null)
  const [activeTab, setActiveTab] = useState('assessment')

  // 初始化加载
  useEffect(() => {
    loadInitialData()
  }, [])

  // 加载初始数据
  const loadInitialData = async () => {
    setLoading(true)
    try {
      await Promise.all([
        loadRiskLevels(),
        loadRiskCategories(),
        loadRollbackStrategies(),
        loadRiskThresholds(),
      ])
    } catch (error) {
      message.error('加载初始数据失败')
    } finally {
      setLoading(false)
    }
  }

  // 加载风险等级
  const loadRiskLevels = async () => {
    try {
      const result = await riskAssessmentService.listRiskLevels()
      if (result.success) {
        setRiskLevels(result.levels)
      }
    } catch (error) {
      logger.error('加载风险等级失败:', error)
    }
  }

  // 加载风险类别
  const loadRiskCategories = async () => {
    try {
      const result = await riskAssessmentService.listRiskCategories()
      if (result.success) {
        setRiskCategories(result.categories)
      }
    } catch (error) {
      logger.error('加载风险类别失败:', error)
    }
  }

  // 加载回滚策略
  const loadRollbackStrategies = async () => {
    try {
      const result = await riskAssessmentService.listRollbackStrategies()
      if (result.success) {
        setRollbackStrategies(result.strategies)
      }
    } catch (error) {
      logger.error('加载回滚策略失败:', error)
    }
  }

  // 加载风险阈值
  const loadRiskThresholds = async () => {
    try {
      const result = await riskAssessmentService.getRiskThresholds()
      if (result.success) {
        setRiskThresholds(result.thresholds)
      }
    } catch (error) {
      logger.error('加载风险阈值失败:', error)
    }
  }

  // 评估风险
  const handleAssessRisk = async (values: any) => {
    try {
      setLoading(true)
      const result = await riskAssessmentService.assessRisk(values)
      if (result.success) {
        setCurrentAssessment(result.assessment)
        message.success('风险评估完成')
        setAssessModalVisible(false)
        assessForm.resetFields()

        // 加载历史记录
        await loadRiskHistory(values.experiment_id)
      }
    } catch (error) {
      message.error('风险评估失败')
    } finally {
      setLoading(false)
    }
  }

  // 加载风险历史
  const loadRiskHistory = async (experimentId: string) => {
    try {
      const result = await riskAssessmentService.getRiskHistory(experimentId)
      if (result.success) {
        setRiskHistory(result.assessments)
      }
    } catch (error) {
      logger.error('加载风险历史失败:', error)
    }
  }

  // 创建回滚计划
  const handleCreateRollbackPlan = async (values: any) => {
    try {
      const result = await riskAssessmentService.createRollbackPlan(values)
      if (result.success) {
        message.success('回滚计划创建成功')
        setRollbackModalVisible(false)
        rollbackForm.resetFields()

        // 添加到计划列表
        setRollbackPlans([
          ...rollbackPlans,
          { id: result.plan_id, ...result.plan },
        ])
      }
    } catch (error) {
      message.error('创建回滚计划失败')
    }
  }

  // 执行回滚
  const handleExecuteRollback = async (
    planId: string,
    force: boolean = false
  ) => {
    try {
      const result = await riskAssessmentService.executeRollback({
        plan_id: planId,
        force,
      })
      if (result.success) {
        message.success('回滚执行已启动')
        setRollbackExecutions([...rollbackExecutions, result.execution])
      }
    } catch (error) {
      message.error('执行回滚失败')
    }
  }

  // 启动监控
  const handleStartMonitoring = async (values: any) => {
    try {
      const result = await riskAssessmentService.startMonitoring(values)
      if (result.success) {
        message.success('风险监控已启动')
        setMonitorModalVisible(false)
        monitorForm.resetFields()
      }
    } catch (error) {
      message.error('启动监控失败')
    }
  }

  // 更新阈值
  const handleUpdateThreshold = async (values: any) => {
    try {
      const result = await riskAssessmentService.updateRiskThreshold(
        values.category,
        values.metric,
        values.value
      )
      if (result.success) {
        message.success('阈值更新成功')
        setThresholdModalVisible(false)
        thresholdForm.resetFields()
        await loadRiskThresholds()
      }
    } catch (error) {
      message.error('更新阈值失败')
    }
  }

  // 获取风险等级颜色
  const getRiskLevelColor = (level: RiskLevel) => {
    const levelInfo = riskLevels.find(l => l.value === level)
    return levelInfo?.color || 'default'
  }

  // 获取风险等级图标
  const getRiskLevelIcon = (level: RiskLevel) => {
    switch (level) {
      case RiskLevel.MINIMAL:
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case RiskLevel.LOW:
        return <InfoCircleOutlined style={{ color: '#1890ff' }} />
      case RiskLevel.MEDIUM:
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />
      case RiskLevel.HIGH:
        return <WarningOutlined style={{ color: '#ff7a45' }} />
      case RiskLevel.CRITICAL:
        return <FireOutlined style={{ color: '#ff4d4f' }} />
      default:
        return <InfoCircleOutlined />
    }
  }

  // 风险仪表盘配置
  const gaugeConfig = {
    percent: currentAssessment ? currentAssessment.overall_risk_score : 0,
    range: {
      color: ['#52c41a', '#1890ff', '#faad14', '#ff7a45', '#ff4d4f'],
    },
    indicator: {
      pointer: { style: { stroke: '#D0D0D0' } },
      pin: { style: { stroke: '#D0D0D0' } },
    },
    statistic: {
      content: {
        formatter: ({ percent }: any) =>
          `风险分数: ${(percent * 100).toFixed(1)}%`,
      },
    },
  }

  // 风险因素表格列配置
  const riskFactorColumns = [
    {
      title: '类别',
      dataIndex: 'category',
      key: 'category',
      render: (category: RiskCategory) => {
        const categoryInfo = riskCategories.find(c => c.value === category)
        return <Tag>{categoryInfo?.name || category}</Tag>
      },
    },
    {
      title: '风险名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
    {
      title: '风险分数',
      dataIndex: 'risk_score',
      key: 'risk_score',
      render: (score: number) => (
        <Progress
          percent={score * 100}
          size="small"
          strokeColor={
            score > 0.7 ? '#ff4d4f' : score > 0.4 ? '#faad14' : '#52c41a'
          }
        />
      ),
    },
    {
      title: '严重性',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: string) => (
        <Tag
          color={
            severity === 'high'
              ? 'red'
              : severity === 'medium'
                ? 'orange'
                : 'green'
          }
        >
          {severity}
        </Tag>
      ),
    },
    {
      title: '缓解措施',
      dataIndex: 'mitigation',
      key: 'mitigation',
      ellipsis: true,
    },
  ]

  // 回滚计划表格列配置
  const rollbackPlanColumns = [
    {
      title: '计划ID',
      dataIndex: 'id',
      key: 'id',
      ellipsis: true,
    },
    {
      title: '实验ID',
      dataIndex: 'experiment_id',
      key: 'experiment_id',
      ellipsis: true,
    },
    {
      title: '策略',
      dataIndex: 'strategy',
      key: 'strategy',
      render: (strategy: RollbackStrategy) => {
        const strategyInfo = rollbackStrategies.find(s => s.value === strategy)
        return (
          <Tag
            color={
              strategy === RollbackStrategy.IMMEDIATE
                ? 'red'
                : strategy === RollbackStrategy.GRADUAL
                  ? 'orange'
                  : strategy === RollbackStrategy.PARTIAL
                    ? 'blue'
                    : 'default'
            }
          >
            {strategyInfo?.name || strategy}
          </Tag>
        )
      },
    },
    {
      title: '预计时长',
      dataIndex: 'estimated_duration_minutes',
      key: 'duration',
      render: (minutes: number) => `${minutes}分钟`,
    },
    {
      title: '自动执行',
      dataIndex: 'auto_execute',
      key: 'auto_execute',
      render: (auto: boolean) => (
        <Tag color={auto ? 'green' : 'default'}>{auto ? '是' : '否'}</Tag>
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: any, record: any) => (
        <Space>
          <Button
            type="link"
            icon={<PlayCircleOutlined />}
            onClick={() => handleExecuteRollback(record.id)}
          >
            执行
          </Button>
          <Button
            type="link"
            icon={<InfoCircleOutlined />}
            onClick={() => {
              setSelectedRollbackPlan(record)
              setDetailDrawerVisible(true)
            }}
          >
            详情
          </Button>
        </Space>
      ),
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      <Card
        title={
          <Space>
            <WarningOutlined />
            <span>风险评估与回滚管理</span>
          </Space>
        }
        extra={
          <Space>
            <Button
              type="primary"
              icon={<BarChartOutlined />}
              onClick={() => setAssessModalVisible(true)}
            >
              评估风险
            </Button>
            <Button
              icon={<RollbackOutlined />}
              onClick={() => setRollbackModalVisible(true)}
            >
              创建回滚计划
            </Button>
            <Button
              icon={<MonitorOutlined />}
              onClick={() => setMonitorModalVisible(true)}
            >
              启动监控
            </Button>
            <Button
              icon={<SettingOutlined />}
              onClick={() => setThresholdModalVisible(true)}
            >
              阈值设置
            </Button>
            <Button
              icon={<SyncOutlined />}
              onClick={loadInitialData}
              loading={loading}
            >
              刷新
            </Button>
          </Space>
        }
      >
        {/* 当前评估概览 */}
        {currentAssessment && (
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={8}>
              <Card>
                <div style={{ textAlign: 'center' }}>
                  <Gauge {...gaugeConfig} />
                  <Divider />
                  <Space direction="vertical">
                    <Space>
                      {getRiskLevelIcon(currentAssessment.overall_risk_level)}
                      <Tag
                        color={getRiskLevelColor(
                          currentAssessment.overall_risk_level
                        )}
                      >
                        {
                          riskLevels.find(
                            l =>
                              l.value === currentAssessment.overall_risk_level
                          )?.name
                        }
                      </Tag>
                    </Space>
                    {currentAssessment.requires_rollback && (
                      <Alert
                        message="建议执行回滚"
                        type="error"
                        showIcon
                        icon={<RollbackOutlined />}
                      />
                    )}
                  </Space>
                </div>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="风险统计">
                <Row gutter={16}>
                  <Col span={12}>
                    <Statistic
                      title="风险因素"
                      value={currentAssessment.risk_factors.length}
                      prefix={<BugOutlined />}
                    />
                  </Col>
                  <Col span={12}>
                    <Statistic
                      title="置信度"
                      value={currentAssessment.confidence * 100}
                      suffix="%"
                      prefix={<CheckCircleOutlined />}
                    />
                  </Col>
                </Row>
                <Divider />
                <div>
                  <strong>回滚策略:</strong>
                  <Tag color="blue" style={{ marginLeft: 8 }}>
                    {rollbackStrategies.find(
                      s => s.value === currentAssessment.rollback_strategy
                    )?.name || '无'}
                  </Tag>
                </div>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="建议措施">
                <List
                  size="small"
                  dataSource={currentAssessment.recommendations}
                  renderItem={(item: string) => (
                    <List.Item>
                      <Space>
                        <CheckCircleOutlined style={{ color: '#52c41a' }} />
                        {item}
                      </Space>
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
          </Row>
        )}

        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="风险因素" key="factors">
            {currentAssessment ? (
              <Table
                columns={riskFactorColumns}
                dataSource={currentAssessment.risk_factors}
                rowKey={(record, index) => index?.toString() || '0'}
                pagination={{ pageSize: 10 }}
              />
            ) : (
              <Empty description="请先进行风险评估" />
            )}
          </TabPane>

          <TabPane tab="评估历史" key="history">
            {riskHistory.length > 0 ? (
              <Timeline mode="left">
                {riskHistory.map((assessment, index) => (
                  <Timeline.Item
                    key={index}
                    color={getRiskLevelColor(assessment.risk_level)}
                    label={new Date(
                      assessment.assessment_time
                    ).toLocaleString()}
                  >
                    <Space direction="vertical">
                      <Space>
                        {getRiskLevelIcon(assessment.risk_level)}
                        <strong>
                          风险分数: {(assessment.risk_score * 100).toFixed(1)}%
                        </strong>
                      </Space>
                      <span>风险因素数量: {assessment.num_risk_factors}</span>
                      {assessment.requires_rollback && (
                        <Tag color="red">需要回滚</Tag>
                      )}
                    </Space>
                  </Timeline.Item>
                ))}
              </Timeline>
            ) : (
              <Empty description="暂无评估历史" />
            )}
          </TabPane>

          <TabPane tab="回滚计划" key="rollback">
            <Table
              columns={rollbackPlanColumns}
              dataSource={rollbackPlans}
              rowKey="id"
              pagination={{ pageSize: 10 }}
            />
          </TabPane>

          <TabPane tab="阈值配置" key="thresholds">
            <Row gutter={[16, 16]}>
              {Object.entries(riskThresholds).map(
                ([category, metrics]: [string, any]) => (
                  <Col span={8} key={category}>
                    <Card title={category} size="small">
                      <List
                        size="small"
                        dataSource={Object.entries(metrics)}
                        renderItem={([metric, value]: [string, any]) => (
                          <List.Item>
                            <span>
                              {metric}: {value}
                            </span>
                          </List.Item>
                        )}
                      />
                    </Card>
                  </Col>
                )
              )}
            </Row>
          </TabPane>
        </Tabs>
      </Card>

      {/* 评估风险弹窗 */}
      <Modal
        title="风险评估"
        visible={assessModalVisible}
        onCancel={() => setAssessModalVisible(false)}
        onOk={() => assessForm.submit()}
        width={500}
      >
        <Form form={assessForm} layout="vertical" onFinish={handleAssessRisk}>
          <Form.Item
            name="experiment_id"
            label="实验ID"
            rules={[{ required: true, message: '请输入实验ID' }]}
          >
            <Input placeholder="请输入实验ID" />
          </Form.Item>
          <Form.Item
            name="include_predictions"
            label="包含预测分析"
            valuePropName="checked"
            initialValue={true}
          >
            <Switch />
          </Form.Item>
        </Form>
      </Modal>

      {/* 创建回滚计划弹窗 */}
      <Modal
        title="创建回滚计划"
        visible={rollbackModalVisible}
        onCancel={() => setRollbackModalVisible(false)}
        onOk={() => rollbackForm.submit()}
        width={600}
      >
        <Form
          form={rollbackForm}
          layout="vertical"
          onFinish={handleCreateRollbackPlan}
        >
          <Form.Item
            name="experiment_id"
            label="实验ID"
            rules={[{ required: true, message: '请输入实验ID' }]}
          >
            <Input placeholder="请输入实验ID" />
          </Form.Item>
          <Form.Item name="strategy" label="回滚策略">
            <Select placeholder="请选择回滚策略">
              {rollbackStrategies.map(strategy => (
                <Option key={strategy.value} value={strategy.value}>
                  <Space>
                    <span>{strategy.name}</span>
                    <span style={{ color: '#999' }}>({strategy.duration})</span>
                  </Space>
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item
            name="auto_execute"
            label="自动执行"
            valuePropName="checked"
            initialValue={false}
          >
            <Switch />
          </Form.Item>
        </Form>
      </Modal>

      {/* 启动监控弹窗 */}
      <Modal
        title="启动风险监控"
        visible={monitorModalVisible}
        onCancel={() => setMonitorModalVisible(false)}
        onOk={() => monitorForm.submit()}
        width={500}
      >
        <Form
          form={monitorForm}
          layout="vertical"
          onFinish={handleStartMonitoring}
        >
          <Form.Item
            name="experiment_id"
            label="实验ID"
            rules={[{ required: true, message: '请输入实验ID' }]}
          >
            <Input placeholder="请输入实验ID" />
          </Form.Item>
          <Form.Item
            name="check_interval_minutes"
            label="检查间隔(分钟)"
            initialValue={5}
          >
            <InputNumber min={1} max={60} style={{ width: '100%' }} />
          </Form.Item>
        </Form>
      </Modal>

      {/* 阈值设置弹窗 */}
      <Modal
        title="更新风险阈值"
        visible={thresholdModalVisible}
        onCancel={() => setThresholdModalVisible(false)}
        onOk={() => thresholdForm.submit()}
        width={500}
      >
        <Form
          form={thresholdForm}
          layout="vertical"
          onFinish={handleUpdateThreshold}
        >
          <Form.Item
            name="category"
            label="风险类别"
            rules={[{ required: true, message: '请选择风险类别' }]}
          >
            <Select placeholder="请选择风险类别">
              {Object.keys(riskThresholds).map(category => (
                <Option key={category} value={category}>
                  {category}
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item
            name="metric"
            label="指标名称"
            rules={[{ required: true, message: '请输入指标名称' }]}
          >
            <Input placeholder="请输入指标名称" />
          </Form.Item>
          <Form.Item
            name="value"
            label="阈值"
            rules={[{ required: true, message: '请输入阈值' }]}
          >
            <InputNumber style={{ width: '100%' }} step={0.01} />
          </Form.Item>
        </Form>
      </Modal>

      {/* 详情抽屉 */}
      <Drawer
        title="回滚计划详情"
        placement="right"
        width={600}
        visible={detailDrawerVisible}
        onClose={() => setDetailDrawerVisible(false)}
      >
        {selectedRollbackPlan && (
          <Space direction="vertical" style={{ width: '100%' }} size="large">
            <Descriptions column={1} bordered>
              <Descriptions.Item label="计划ID">
                {selectedRollbackPlan.id}
              </Descriptions.Item>
              <Descriptions.Item label="实验ID">
                {selectedRollbackPlan.experiment_id}
              </Descriptions.Item>
              <Descriptions.Item label="触发原因">
                {selectedRollbackPlan.trigger_reason}
              </Descriptions.Item>
              <Descriptions.Item label="策略">
                {
                  rollbackStrategies.find(
                    s => s.value === selectedRollbackPlan.strategy
                  )?.name
                }
              </Descriptions.Item>
              <Descriptions.Item label="预计时长">
                {selectedRollbackPlan.estimated_duration_minutes}分钟
              </Descriptions.Item>
              <Descriptions.Item label="需要审批">
                {selectedRollbackPlan.approval_required ? '是' : '否'}
              </Descriptions.Item>
            </Descriptions>

            <Card title="执行步骤">
              <Steps direction="vertical" size="small">
                {selectedRollbackPlan.steps?.map(
                  (step: string, index: number) => (
                    <Step
                      key={index}
                      title={`步骤 ${index + 1}`}
                      description={step}
                    />
                  )
                )}
              </Steps>
            </Card>
          </Space>
        )}
      </Drawer>
    </div>
  )
}

export default RiskAssessmentDashboard
