import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Progress, Button, Space, Switch, Typography, Table, Tag, Statistic, Alert, Timeline, List, Modal, Form, Input, Select, Slider, Badge } from 'antd'
import { 
  BellOutlined,
  AlertOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  EyeOutlined,
  SettingOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  ReloadOutlined,
  FilterOutlined,
  SyncOutlined,
  WarningOutlined,
  InfoCircleOutlined
} from '@ant-design/icons'
import { Line, Column, Gauge } from '@ant-design/plots'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text, Paragraph } = Typography
const { Option } = Select

interface AlertRule {
  id: string
  name: string
  description: string
  metric_name: string
  threshold: number
  comparison: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  duration: number
  enabled: boolean
  tags: { [key: string]: string }
}

interface AlertInstance {
  id: string
  rule_name: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  status: 'active' | 'resolved' | 'acknowledged' | 'suppressed'
  title: string
  description: string
  current_value: number
  threshold: number
  metric_name: string
  start_time: string
  last_update: string
  resolved_time?: string
  acknowledged_by?: string
  tags: { [key: string]: string }
}

const PersonalizationAlertsPage: React.FC = () => {
  const [alertRules, setAlertRules] = useState<AlertRule[]>([])
  const [activeAlerts, setActiveAlerts] = useState<AlertInstance[]>([])
  const [alertHistory, setAlertHistory] = useState<AlertInstance[]>([])
  const [selectedAlert, setSelectedAlert] = useState<AlertInstance | null>(null)
  const [ruleModalVisible, setRuleModalVisible] = useState(false)
  const [editingRule, setEditingRule] = useState<AlertRule | null>(null)
  const [alertFilter, setAlertFilter] = useState<string>('all')
  const [form] = Form.useForm()

  // 模拟数据
  useEffect(() => {
    // 模拟告警规则
    setAlertRules([
      {
        id: '1',
        name: 'high_recommendation_latency',
        description: '推荐响应延迟过高',
        metric_name: 'recommendation_latency_p99',
        threshold: 100.0,
        comparison: '>',
        severity: 'high',
        duration: 300,
        enabled: true,
        tags: { component: 'recommendation', metric_type: 'latency' }
      },
      {
        id: '2',
        name: 'low_cache_hit_rate',
        description: '缓存命中率过低',
        metric_name: 'cache_hit_rate',
        threshold: 0.8,
        comparison: '<',
        severity: 'medium',
        duration: 600,
        enabled: true,
        tags: { component: 'cache', metric_type: 'hit_rate' }
      },
      {
        id: '3',
        name: 'high_error_rate',
        description: '系统错误率过高',
        metric_name: 'error_rate',
        threshold: 0.01,
        comparison: '>',
        severity: 'critical',
        duration: 180,
        enabled: true,
        tags: { component: 'system', metric_type: 'error_rate' }
      }
    ])

    // 模拟活跃告警
    setActiveAlerts([
      {
        id: 'alert_1',
        rule_name: 'high_recommendation_latency',
        severity: 'high',
        status: 'active',
        title: '推荐延迟告警',
        description: '推荐响应延迟超过100ms阈值',
        current_value: 125.6,
        threshold: 100.0,
        metric_name: 'recommendation_latency_p99',
        start_time: '2024-01-15 14:30:00',
        last_update: '2024-01-15 14:35:00',
        tags: { component: 'recommendation' }
      },
      {
        id: 'alert_2',
        rule_name: 'low_cache_hit_rate',
        severity: 'medium',
        status: 'acknowledged',
        title: '缓存命中率低',
        description: '缓存命中率低于80%阈值',
        current_value: 0.75,
        threshold: 0.8,
        metric_name: 'cache_hit_rate',
        start_time: '2024-01-15 14:20:00',
        last_update: '2024-01-15 14:32:00',
        acknowledged_by: 'admin',
        tags: { component: 'cache' }
      }
    ])

    // 模拟告警历史
    setAlertHistory([
      {
        id: 'alert_3',
        rule_name: 'high_error_rate',
        severity: 'critical',
        status: 'resolved',
        title: '系统错误率过高',
        description: '系统错误率超过1%阈值',
        current_value: 0.015,
        threshold: 0.01,
        metric_name: 'error_rate',
        start_time: '2024-01-15 13:45:00',
        last_update: '2024-01-15 14:10:00',
        resolved_time: '2024-01-15 14:10:00',
        tags: { component: 'system' }
      }
    ])
  }, [])

  const getSeverityColor = (severity: string) => {
    const colors = {
      low: '#52c41a',
      medium: '#faad14',
      high: '#fa8c16',
      critical: '#ff4d4f'
    }
    return colors[severity] || '#d9d9d9'
  }

  const getStatusColor = (status: string) => {
    const colors = {
      active: 'red',
      resolved: 'green',
      acknowledged: 'orange',
      suppressed: 'default'
    }
    return colors[status] || 'default'
  }

  const alertRuleColumns: ColumnsType<AlertRule> = [
    {
      title: '规则名称',
      dataIndex: 'name',
      key: 'name',
      render: (text) => <Text strong>{text}</Text>
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description'
    },
    {
      title: '指标',
      dataIndex: 'metric_name',
      key: 'metric_name',
      render: (text) => <Tag color="blue">{text}</Tag>
    },
    {
      title: '阈值',
      key: 'threshold',
      render: (_, record) => `${record.comparison} ${record.threshold}`
    },
    {
      title: '严重级别',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity) => (
        <Tag color={getSeverityColor(severity)}>
          {severity.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled) => (
        <Switch checked={enabled} size="small" />
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button 
            type="link" 
            icon={<EditOutlined />} 
            onClick={() => handleEditRule(record)}
          >
            编辑
          </Button>
          <Button 
            type="link" 
            danger 
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteRule(record.id)}
          >
            删除
          </Button>
        </Space>
      )
    }
  ]

  const activeAlertColumns: ColumnsType<AlertInstance> = [
    {
      title: '告警',
      dataIndex: 'title',
      key: 'title',
      render: (text, record) => (
        <Space direction="vertical" size="small">
          <Text strong>{text}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.description}
          </Text>
        </Space>
      )
    },
    {
      title: '严重级别',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity) => (
        <Tag color={getSeverityColor(severity)}>
          {severity.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Tag color={getStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '当前值',
      key: 'current_value',
      render: (_, record) => (
        <Space direction="vertical" size="small">
          <Text>{record.current_value.toFixed(2)}</Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            阈值: {record.threshold}
          </Text>
        </Space>
      )
    },
    {
      title: '开始时间',
      dataIndex: 'start_time',
      key: 'start_time'
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button 
            type="link" 
            icon={<EyeOutlined />}
            onClick={() => setSelectedAlert(record)}
          >
            详情
          </Button>
          {record.status === 'active' && (
            <Button 
              type="link" 
              icon={<CheckCircleOutlined />}
              onClick={() => handleAcknowledgeAlert(record.id)}
            >
              确认
            </Button>
          )}
        </Space>
      )
    }
  ]

  const handleEditRule = (rule: AlertRule) => {
    setEditingRule(rule)
    form.setFieldsValue(rule)
    setRuleModalVisible(true)
  }

  const handleDeleteRule = (ruleId: string) => {
    setAlertRules(prev => prev.filter(rule => rule.id !== ruleId))
  }

  const handleAcknowledgeAlert = (alertId: string) => {
    setActiveAlerts(prev => 
      prev.map(alert => 
        alert.id === alertId 
          ? { ...alert, status: 'acknowledged', acknowledged_by: 'current_user' }
          : alert
      )
    )
  }

  const handleSaveRule = async (values: any) => {
    if (editingRule) {
      // 更新规则
      setAlertRules(prev => 
        prev.map(rule => 
          rule.id === editingRule.id 
            ? { ...rule, ...values }
            : rule
        )
      )
    } else {
      // 新增规则
      const newRule: AlertRule = {
        id: Date.now().toString(),
        ...values
      }
      setAlertRules(prev => [...prev, newRule])
    }
    
    setRuleModalVisible(false)
    setEditingRule(null)
    form.resetFields()
  }

  const filteredAlerts = activeAlerts.filter(alert => {
    if (alertFilter === 'all') return true
    return alert.severity === alertFilter
  })

  // 告警概览统计
  const alertStats = {
    total: activeAlerts.length,
    critical: activeAlerts.filter(a => a.severity === 'critical').length,
    high: activeAlerts.filter(a => a.severity === 'high').length,
    medium: activeAlerts.filter(a => a.severity === 'medium').length,
    low: activeAlerts.filter(a => a.severity === 'low').length
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <BellOutlined /> 告警管理系统
      </Title>
      <Paragraph type="secondary">
        实时监控个性化引擎告警，管理告警规则和通知配置
      </Paragraph>

      {/* 告警概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总告警数"
              value={alertStats.total}
              prefix={<AlertOutlined />}
              valueStyle={{ color: alertStats.total > 0 ? '#fa8c16' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="严重告警"
              value={alertStats.critical}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#ff4d4f' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="高级告警"
              value={alertStats.high}
              prefix={<WarningOutlined />}
              valueStyle={{ color: '#fa8c16' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="中级告警"
              value={alertStats.medium}
              prefix={<InfoCircleOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 活跃告警 */}
      <Card 
        title="活跃告警" 
        style={{ marginBottom: 24 }}
        extra={
          <Space>
            <Select
              value={alertFilter}
              onChange={setAlertFilter}
              style={{ width: 120 }}
            >
              <Option value="all">全部</Option>
              <Option value="critical">严重</Option>
              <Option value="high">高级</Option>
              <Option value="medium">中级</Option>
              <Option value="low">低级</Option>
            </Select>
            <Button icon={<ReloadOutlined />}>刷新</Button>
          </Space>
        }
      >
        <Table
          columns={activeAlertColumns}
          dataSource={filteredAlerts}
          rowKey="id"
          size="small"
          pagination={false}
        />
      </Card>

      {/* 告警规则管理 */}
      <Card 
        title="告警规则"
        extra={
          <Button 
            type="primary" 
            icon={<PlusOutlined />}
            onClick={() => {
              setEditingRule(null)
              form.resetFields()
              setRuleModalVisible(true)
            }}
          >
            新增规则
          </Button>
        }
      >
        <Table
          columns={alertRuleColumns}
          dataSource={alertRules}
          rowKey="id"
          size="small"
        />
      </Card>

      {/* 告警历史 */}
      <Card title="告警历史" style={{ marginTop: 24 }}>
        <Timeline>
          {alertHistory.map(alert => (
            <Timeline.Item
              key={alert.id}
              color={getSeverityColor(alert.severity)}
              dot={<CheckCircleOutlined />}
            >
              <Space direction="vertical" size="small">
                <Text strong>{alert.title}</Text>
                <Text type="secondary">{alert.description}</Text>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {alert.start_time} - {alert.resolved_time}
                </Text>
              </Space>
            </Timeline.Item>
          ))}
        </Timeline>
      </Card>

      {/* 规则编辑模态框 */}
      <Modal
        title={editingRule ? '编辑告警规则' : '新增告警规则'}
        visible={ruleModalVisible}
        onCancel={() => {
          setRuleModalVisible(false)
          setEditingRule(null)
          form.resetFields()
        }}
        onOk={() => form.submit()}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSaveRule}
        >
          <Form.Item
            name="name"
            label="规则名称"
            rules={[{ required: true, message: '请输入规则名称' }]}
          >
            <Input placeholder="规则名称" />
          </Form.Item>
          
          <Form.Item
            name="description"
            label="描述"
            rules={[{ required: true, message: '请输入描述' }]}
          >
            <Input.TextArea placeholder="规则描述" />
          </Form.Item>
          
          <Form.Item
            name="metric_name"
            label="监控指标"
            rules={[{ required: true, message: '请选择监控指标' }]}
          >
            <Select placeholder="选择监控指标">
              <Option value="recommendation_latency_p99">推荐延迟P99</Option>
              <Option value="feature_computation_latency_avg">特征计算延迟</Option>
              <Option value="cache_hit_rate">缓存命中率</Option>
              <Option value="error_rate">错误率</Option>
              <Option value="memory_usage_percent">内存使用率</Option>
              <Option value="cpu_usage_percent">CPU使用率</Option>
            </Select>
          </Form.Item>
          
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="comparison"
                label="比较操作"
                rules={[{ required: true, message: '请选择比较操作' }]}
              >
                <Select placeholder="比较操作">
                  <Option value=">">大于</Option>
                  <Option value="<">小于</Option>
                  <Option value=">=">大于等于</Option>
                  <Option value="<=">小于等于</Option>
                  <Option value="==">等于</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={16}>
              <Form.Item
                name="threshold"
                label="阈值"
                rules={[{ required: true, message: '请输入阈值' }]}
              >
                <Input type="number" placeholder="阈值" />
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item
            name="severity"
            label="严重级别"
            rules={[{ required: true, message: '请选择严重级别' }]}
          >
            <Select placeholder="严重级别">
              <Option value="low">低级</Option>
              <Option value="medium">中级</Option>
              <Option value="high">高级</Option>
              <Option value="critical">严重</Option>
            </Select>
          </Form.Item>
          
          <Form.Item
            name="duration"
            label="持续时间(秒)"
            rules={[{ required: true, message: '请输入持续时间' }]}
          >
            <Slider
              min={60}
              max={3600}
              marks={{
                60: '1分钟',
                300: '5分钟',
                900: '15分钟',
                1800: '30分钟',
                3600: '1小时'
              }}
            />
          </Form.Item>
          
          <Form.Item
            name="enabled"
            label="启用状态"
            valuePropName="checked"
            initialValue={true}
          >
            <Switch checkedChildren="启用" unCheckedChildren="禁用" />
          </Form.Item>
        </Form>
      </Modal>

      {/* 告警详情模态框 */}
      <Modal
        title="告警详情"
        visible={selectedAlert !== null}
        onCancel={() => setSelectedAlert(null)}
        footer={[
          <Button key="close" onClick={() => setSelectedAlert(null)}>
            关闭
          </Button>
        ]}
        width={600}
      >
        {selectedAlert && (
          <div>
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Text strong>告警标题:</Text>
                <br />
                <Text>{selectedAlert.title}</Text>
              </Col>
              <Col span={12}>
                <Text strong>严重级别:</Text>
                <br />
                <Tag color={getSeverityColor(selectedAlert.severity)}>
                  {selectedAlert.severity.toUpperCase()}
                </Tag>
              </Col>
            </Row>
            
            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              <Col span={24}>
                <Text strong>描述:</Text>
                <br />
                <Text>{selectedAlert.description}</Text>
              </Col>
            </Row>
            
            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              <Col span={12}>
                <Text strong>指标名称:</Text>
                <br />
                <Text>{selectedAlert.metric_name}</Text>
              </Col>
              <Col span={12}>
                <Text strong>当前值:</Text>
                <br />
                <Text>{selectedAlert.current_value}</Text>
              </Col>
            </Row>
            
            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              <Col span={12}>
                <Text strong>阈值:</Text>
                <br />
                <Text>{selectedAlert.threshold}</Text>
              </Col>
              <Col span={12}>
                <Text strong>状态:</Text>
                <br />
                <Tag color={getStatusColor(selectedAlert.status)}>
                  {selectedAlert.status.toUpperCase()}
                </Tag>
              </Col>
            </Row>
            
            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              <Col span={12}>
                <Text strong>开始时间:</Text>
                <br />
                <Text>{selectedAlert.start_time}</Text>
              </Col>
              <Col span={12}>
                <Text strong>最后更新:</Text>
                <br />
                <Text>{selectedAlert.last_update}</Text>
              </Col>
            </Row>
            
            {selectedAlert.acknowledged_by && (
              <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
                <Col span={24}>
                  <Text strong>确认人:</Text>
                  <br />
                  <Text>{selectedAlert.acknowledged_by}</Text>
                </Col>
              </Row>
            )}
          </div>
        )}
      </Modal>
    </div>
  )
}

export default PersonalizationAlertsPage