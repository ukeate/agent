import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Tabs,
  Form,
  Input,
  Button,
  Select,
  InputNumber,
  Switch,
  Table,
  Statistic,
  Row,
  Col,
  Progress,
  Alert,
  Modal,
  Space,
  Tag,
  Descriptions,
  Timeline,
  Spin,
  message,
  Tooltip,
  Badge,
} from 'antd'
import {
  PlayCircleOutlined,
  PauseCircleOutlined,
  SettingOutlined,
  HistoryOutlined,
  BulbOutlined,
  ExperimentOutlined,
  PlusOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import {
  autoScalingService,
  ScalingMode,
  ScalingDirection,
  ScalingTrigger,
  ScalingRule,
  ScalingHistory,
  ScalingRecommendation,
  ScalingSimulation,
  CreateScalingRuleRequest,
  CreateConditionRequest,
} from '../services/autoScalingService'

const { TabPane } = Tabs
const { Option } = Select

const AutoScalingManagementPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [rules, setRules] = useState<ScalingRule[]>([])
  const [history, setHistory] = useState<ScalingHistory | null>(null)
  const [recommendations, setRecommendations] = useState<
    ScalingRecommendation[]
  >([])
  const [simulations, setSimulations] = useState<ScalingSimulation[]>([])
  const [scalingModes, setScalingModes] = useState<any[]>([])
  const [triggers, setTriggers] = useState<any[]>([])
  const [status, setStatus] = useState<any>({})

  const [createRuleForm] = Form.useForm()
  const [addConditionForm] = Form.useForm()
  const [configForm] = Form.useForm()

  const [selectedRule, setSelectedRule] = useState<string>('')
  const [selectedExperiment, setSelectedExperiment] = useState<string>('')
  const [addConditionVisible, setAddConditionVisible] = useState(false)
  const [conditionType, setConditionType] = useState<'scale_up' | 'scale_down'>(
    'scale_up'
  )

  useEffect(() => {
    loadInitialData()
  }, [])

  const loadInitialData = async () => {
    setLoading(true)
    try {
      await Promise.all([
        loadRules(),
        loadScalingModes(),
        loadTriggers(),
        loadStatus(),
      ])
    } catch (error) {
      message.error('加载数据失败')
    } finally {
      setLoading(false)
    }
  }

  const loadRules = async () => {
    try {
      const response = await autoScalingService.listRules()
      if (response.success) {
        setRules(response.rules)
      }
    } catch (error) {
      logger.error('加载规则失败:', error)
    }
  }

  const loadScalingModes = async () => {
    try {
      const response = await autoScalingService.listScalingModes()
      if (response.success) {
        setScalingModes(response.modes)
      }
    } catch (error) {
      logger.error('加载扩量模式失败:', error)
    }
  }

  const loadTriggers = async () => {
    try {
      const response = await autoScalingService.listTriggers()
      if (response.success) {
        setTriggers(response.triggers)
      }
    } catch (error) {
      logger.error('加载触发器失败:', error)
    }
  }

  const loadStatus = async () => {
    try {
      const response = await autoScalingService.getScalingStatus()
      if (response.success) {
        setStatus(response.status)
      }
    } catch (error) {
      logger.error('加载状态失败:', error)
    }
  }

  const loadHistory = async (experimentId: string) => {
    try {
      const response = await autoScalingService.getScalingHistory(experimentId)
      if (response.success) {
        setHistory(response.history)
      }
    } catch (error) {
      logger.error('加载历史失败:', error)
    }
  }

  const loadRecommendations = async (experimentId: string) => {
    try {
      const response = await autoScalingService.getRecommendations(experimentId)
      if (response.success) {
        setRecommendations(response.recommendations)
      }
    } catch (error) {
      logger.error('加载建议失败:', error)
    }
  }

  const simulateScaling = async (experimentId: string, days: number) => {
    setLoading(true)
    try {
      const response = await autoScalingService.simulateScaling(
        experimentId,
        days
      )
      if (response.success) {
        setSimulations(response.simulations)
      }
    } catch (error) {
      message.error('扩量模拟失败')
    } finally {
      setLoading(false)
    }
  }

  const handleCreateRule = async (values: any) => {
    setLoading(true)
    try {
      const request: CreateScalingRuleRequest = {
        experiment_id: values.experiment_id,
        name: values.name,
        mode: values.mode,
        variant: values.variant,
        description: values.description,
        scale_increment: values.scale_increment,
        scale_decrement: values.scale_decrement,
        min_percentage: values.min_percentage,
        max_percentage: values.max_percentage,
        cooldown_minutes: values.cooldown_minutes,
        enabled: values.enabled,
      }

      const response = await autoScalingService.createScalingRule(request)
      if (response.success) {
        message.success('规则创建成功')
        createRuleForm.resetFields()
        loadRules()
      }
    } catch (error) {
      message.error('创建规则失败')
    } finally {
      setLoading(false)
    }
  }

  const handleAddCondition = async (values: any) => {
    setLoading(true)
    try {
      const request: CreateConditionRequest = {
        trigger: values.trigger,
        metric_name: values.metric_name,
        operator: values.operator,
        threshold: values.threshold,
        confidence_level: values.confidence_level,
        min_sample_size: values.min_sample_size,
      }

      const response = await autoScalingService.addCondition(
        selectedRule,
        conditionType,
        request
      )
      if (response.success) {
        message.success('条件添加成功')
        addConditionForm.resetFields()
        setAddConditionVisible(false)
        loadRules()
      }
    } catch (error) {
      message.error('添加条件失败')
    } finally {
      setLoading(false)
    }
  }

  const handleStartScaling = async (ruleId: string) => {
    try {
      const response = await autoScalingService.startAutoScaling(ruleId)
      if (response.success) {
        message.success('自动扩量已启动')
        loadRules()
        loadStatus()
      }
    } catch (error) {
      message.error('启动扩量失败')
    }
  }

  const handleStopScaling = async (ruleId: string) => {
    try {
      const response = await autoScalingService.stopAutoScaling(ruleId)
      if (response.success) {
        message.success('自动扩量已停止')
        loadRules()
        loadStatus()
      }
    } catch (error) {
      message.error('停止扩量失败')
    }
  }

  const createTemplate = async (
    type: 'safe' | 'aggressive',
    experimentId: string
  ) => {
    try {
      const response =
        type === 'safe'
          ? await autoScalingService.createSafeTemplate(experimentId)
          : await autoScalingService.createAggressiveTemplate(experimentId)

      if (response.success) {
        message.success(`${type === 'safe' ? '安全' : '激进'}扩量模板创建成功`)
        loadRules()
      }
    } catch (error) {
      message.error('创建模板失败')
    }
  }

  const rulesColumns = [
    {
      title: '规则名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '实验ID',
      dataIndex: 'experiment_id',
      key: 'experiment_id',
    },
    {
      title: '扩量模式',
      dataIndex: 'mode',
      key: 'mode',
      render: (mode: ScalingMode) => {
        const modeInfo = scalingModes.find(m => m.value === mode)
        return <Tag color="blue">{modeInfo?.name || mode}</Tag>
      },
    },
    {
      title: '状态',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled: boolean) => (
        <Badge
          status={enabled ? 'success' : 'default'}
          text={enabled ? '启用' : '禁用'}
        />
      ),
    },
    {
      title: '扩量条件',
      dataIndex: 'scale_up_conditions_count',
      key: 'scale_up_conditions_count',
      render: (count: number, record: any) => (
        <span>
          ↑{count} / ↓{record.scale_down_conditions_count}
        </span>
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
      render: (_, record: ScalingRule) => (
        <Space>
          <Button
            type="primary"
            size="small"
            icon={<PlayCircleOutlined />}
            onClick={() => handleStartScaling(record.id)}
            disabled={!record.enabled}
          >
            启动
          </Button>
          <Button
            size="small"
            icon={<PauseCircleOutlined />}
            onClick={() => handleStopScaling(record.id)}
          >
            停止
          </Button>
          <Button
            size="small"
            icon={<PlusOutlined />}
            onClick={() => {
              setSelectedRule(record.id)
              setAddConditionVisible(true)
            }}
          >
            添加条件
          </Button>
        </Space>
      ),
    },
  ]

  const simulationColumns = [
    {
      title: '天数',
      dataIndex: 'day',
      key: 'day',
    },
    {
      title: '当前流量(%)',
      dataIndex: 'current_percentage',
      key: 'current_percentage',
    },
    {
      title: '新流量(%)',
      dataIndex: 'new_percentage',
      key: 'new_percentage',
    },
    {
      title: '操作',
      dataIndex: 'action',
      key: 'action',
      render: (action: string) => {
        const color =
          action === 'scale_up'
            ? 'green'
            : action === 'scale_down'
              ? 'red'
              : 'default'
        return <Tag color={color}>{action}</Tag>
      },
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => (
        <Progress percent={Math.round(confidence * 100)} size="small" />
      ),
    },
  ]

  return (
    <div style={{ padding: '24px' }}>
      <Card
        title="自动扩量管理"
        extra={
          <Space>
            <Button icon={<ReloadOutlined />} onClick={loadInitialData}>
              刷新
            </Button>
            <Tooltip title="扩量服务状态">
              <Badge
                status={status.active_rules > 0 ? 'processing' : 'default'}
                text={`活跃规则: ${status.active_rules}/${status.total_rules}`}
              />
            </Tooltip>
          </Space>
        }
      >
        <Tabs defaultActiveKey="rules">
          <TabPane
            tab={
              <span>
                <SettingOutlined />
                扩量规则
              </span>
            }
            key="rules"
          >
            <Row gutter={[16, 16]}>
              <Col span={24}>
                <Card title="创建扩量规则" size="small">
                  <Form
                    form={createRuleForm}
                    layout="inline"
                    onFinish={handleCreateRule}
                  >
                    <Form.Item
                      name="experiment_id"
                      rules={[{ required: true }]}
                    >
                      <Input
                        placeholder="实验ID"
                        name="autoscaling-experiment-id"
                        autoComplete="off"
                      />
                    </Form.Item>
                    <Form.Item name="name" rules={[{ required: true }]}>
                      <Input
                        placeholder="规则名称"
                        name="autoscaling-rule-name"
                        autoComplete="off"
                      />
                    </Form.Item>
                    <Form.Item name="mode" initialValue={ScalingMode.BALANCED}>
                      <Select placeholder="扩量模式" style={{ width: 120 }}>
                        {scalingModes.map(mode => (
                          <Option key={mode.value} value={mode.value}>
                            {mode.name}
                          </Option>
                        ))}
                      </Select>
                    </Form.Item>
                    <Form.Item name="variant" initialValue="treatment">
                      <Input
                        placeholder="变体"
                        name="autoscaling-variant"
                        autoComplete="off"
                        style={{ width: 100 }}
                      />
                    </Form.Item>
                    <Form.Item name="scale_increment" initialValue={10}>
                      <InputNumber placeholder="扩量增量%" min={1} max={50} />
                    </Form.Item>
                    <Form.Item name="scale_decrement" initialValue={5}>
                      <InputNumber placeholder="缩量减量%" min={1} max={50} />
                    </Form.Item>
                    <Form.Item name="min_percentage" initialValue={1}>
                      <InputNumber placeholder="最小流量%" min={0} max={100} />
                    </Form.Item>
                    <Form.Item name="max_percentage" initialValue={100}>
                      <InputNumber placeholder="最大流量%" min={0} max={100} />
                    </Form.Item>
                    <Form.Item name="cooldown_minutes" initialValue={30}>
                      <InputNumber placeholder="冷却时间(分钟)" min={5} />
                    </Form.Item>
                    <Form.Item
                      name="enabled"
                      valuePropName="checked"
                      initialValue={true}
                    >
                      <Switch checkedChildren="启用" unCheckedChildren="禁用" />
                    </Form.Item>
                    <Form.Item>
                      <Button
                        type="primary"
                        htmlType="submit"
                        loading={loading}
                      >
                        创建规则
                      </Button>
                    </Form.Item>
                  </Form>
                </Card>
              </Col>
              <Col span={24}>
                <Card
                  title="扩量规则列表"
                  size="small"
                  extra={
                    <Space>
                      <Button
                        onClick={() =>
                          createTemplate('safe', selectedExperiment)
                        }
                        disabled={!selectedExperiment}
                      >
                        创建安全模板
                      </Button>
                      <Button
                        onClick={() =>
                          createTemplate('aggressive', selectedExperiment)
                        }
                        disabled={!selectedExperiment}
                      >
                        创建激进模板
                      </Button>
                      <Input
                        placeholder="实验ID过滤"
                        style={{ width: 200 }}
                        onChange={e => setSelectedExperiment(e.target.value)}
                        name="autoscaling-filter-experiment"
                        autoComplete="off"
                      />
                    </Space>
                  }
                >
                  <Table
                    columns={rulesColumns}
                    dataSource={rules}
                    rowKey="id"
                    size="small"
                    loading={loading}
                  />
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane
            tab={
              <span>
                <HistoryOutlined />
                扩量历史
              </span>
            }
            key="history"
          >
            <Row gutter={[16, 16]}>
              <Col span={24}>
                <Input.Search
                  placeholder="输入实验ID查看历史"
                  enterButton="查询"
                  onSearch={value => {
                    setSelectedExperiment(value)
                    loadHistory(value)
                  }}
                />
              </Col>
              {history && (
                <>
                  <Col span={24}>
                    <Row gutter={16}>
                      <Col span={6}>
                        <Statistic
                          title="当前流量百分比"
                          value={history.current_percentage}
                          suffix="%"
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="总扩量次数"
                          value={history.total_scale_ups}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="总缩量次数"
                          value={history.total_scale_downs}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="最后扩量时间"
                          value={
                            history.last_scaled_at
                              ? new Date(
                                  history.last_scaled_at
                                ).toLocaleString()
                              : '无'
                          }
                        />
                      </Col>
                    </Row>
                  </Col>
                  <Col span={24}>
                    <Card title="最近扩量决策" size="small">
                      <Timeline>
                        {history.recent_decisions.map((decision, index) => (
                          <Timeline.Item
                            key={index}
                            color={
                              decision.direction === ScalingDirection.UP
                                ? 'green'
                                : decision.direction === ScalingDirection.DOWN
                                  ? 'red'
                                  : 'blue'
                            }
                          >
                            <div>
                              <strong>
                                {new Date(decision.timestamp).toLocaleString()}
                              </strong>
                              <br />
                              方向:{' '}
                              <Tag
                                color={
                                  decision.direction === ScalingDirection.UP
                                    ? 'green'
                                    : 'red'
                                }
                              >
                                {decision.direction}
                              </Tag>
                              <br />
                              流量变化: {decision.from}% → {decision.to}%
                              <br />
                              原因: {decision.reason}
                              <br />
                              置信度:{' '}
                              <Progress
                                percent={Math.round(decision.confidence * 100)}
                                size="small"
                              />
                            </div>
                          </Timeline.Item>
                        ))}
                      </Timeline>
                    </Card>
                  </Col>
                </>
              )}
            </Row>
          </TabPane>

          <TabPane
            tab={
              <span>
                <BulbOutlined />
                扩量建议
              </span>
            }
            key="recommendations"
          >
            <Row gutter={[16, 16]}>
              <Col span={24}>
                <Input.Search
                  placeholder="输入实验ID获取建议"
                  enterButton="获取建议"
                  onSearch={value => {
                    setSelectedExperiment(value)
                    loadRecommendations(value)
                  }}
                />
              </Col>
              <Col span={24}>
                <Row gutter={16}>
                  {recommendations.map((rec, index) => (
                    <Col span={8} key={index}>
                      <Card size="small" title={`建议 ${index + 1}`}>
                        <Descriptions column={1} size="small">
                          <Descriptions.Item label="操作">
                            {rec.action}
                          </Descriptions.Item>
                          <Descriptions.Item label="置信度">
                            <Progress
                              percent={Math.round(rec.confidence * 100)}
                              size="small"
                            />
                          </Descriptions.Item>
                          <Descriptions.Item label="原因">
                            {rec.reason}
                          </Descriptions.Item>
                          <Descriptions.Item label="风险等级">
                            <Tag
                              color={
                                rec.risk_level === 'low'
                                  ? 'green'
                                  : rec.risk_level === 'medium'
                                    ? 'orange'
                                    : 'red'
                              }
                            >
                              {rec.risk_level}
                            </Tag>
                          </Descriptions.Item>
                          {rec.suggested_percentage && (
                            <Descriptions.Item label="建议流量">
                              {rec.suggested_percentage}%
                            </Descriptions.Item>
                          )}
                          {rec.expected_impact && (
                            <Descriptions.Item label="预期影响">
                              {rec.expected_impact}
                            </Descriptions.Item>
                          )}
                        </Descriptions>
                      </Card>
                    </Col>
                  ))}
                </Row>
              </Col>
            </Row>
          </TabPane>

          <TabPane
            tab={
              <span>
                <ExperimentOutlined />
                扩量模拟
              </span>
            }
            key="simulation"
          >
            <Row gutter={[16, 16]}>
              <Col span={24}>
                <Card size="small" title="扩量模拟">
                  <Form
                    layout="inline"
                    onFinish={values =>
                      simulateScaling(values.experiment_id, values.days)
                    }
                  >
                    <Form.Item
                      name="experiment_id"
                      rules={[{ required: true }]}
                    >
                      <Input placeholder="实验ID" />
                    </Form.Item>
                    <Form.Item name="days" initialValue={7}>
                      <InputNumber placeholder="模拟天数" min={1} max={30} />
                    </Form.Item>
                    <Form.Item>
                      <Button
                        type="primary"
                        htmlType="submit"
                        loading={loading}
                      >
                        开始模拟
                      </Button>
                    </Form.Item>
                  </Form>
                </Card>
              </Col>
              {simulations.length > 0 && (
                <Col span={24}>
                  <Card title="模拟结果" size="small">
                    <Table
                      columns={simulationColumns}
                      dataSource={simulations}
                      rowKey="day"
                      size="small"
                      pagination={false}
                    />
                  </Card>
                </Col>
              )}
            </Row>
          </TabPane>
        </Tabs>
      </Card>

      <Modal
        title="添加扩量条件"
        open={addConditionVisible}
        onCancel={() => setAddConditionVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={addConditionForm}
          layout="vertical"
          onFinish={handleAddCondition}
        >
          <Form.Item label="条件类型">
            <Select value={conditionType} onChange={setConditionType}>
              <Option value="scale_up">扩量条件</Option>
              <Option value="scale_down">缩量条件</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="trigger"
            label="触发器类型"
            rules={[{ required: true }]}
          >
            <Select placeholder="选择触发器">
              {triggers.map(trigger => (
                <Option key={trigger.value} value={trigger.value}>
                  {trigger.name} - {trigger.description}
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item name="metric_name" label="指标名称">
            <Input placeholder="如: conversion_rate" />
          </Form.Item>
          <Form.Item
            name="operator"
            label="操作符"
            rules={[{ required: true }]}
          >
            <Select placeholder="选择操作符">
              <Option value=">">大于</Option>
              <Option value="<">小于</Option>
              <Option value=">=">大于等于</Option>
              <Option value="<=">小于等于</Option>
              <Option value="==">等于</Option>
            </Select>
          </Form.Item>
          <Form.Item name="threshold" label="阈值" rules={[{ required: true }]}>
            <InputNumber placeholder="阈值" style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item
            name="confidence_level"
            label="置信水平"
            initialValue={0.95}
          >
            <InputNumber
              min={0.5}
              max={0.999}
              step={0.01}
              style={{ width: '100%' }}
            />
          </Form.Item>
          <Form.Item
            name="min_sample_size"
            label="最小样本量"
            initialValue={1000}
          >
            <InputNumber min={100} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                添加条件
              </Button>
              <Button onClick={() => setAddConditionVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default AutoScalingManagementPage
