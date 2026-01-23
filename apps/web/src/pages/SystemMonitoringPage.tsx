import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Tag,
  Button,
  Space,
  Alert,
  Typography,
  Tabs,
  Select,
  DatePicker,
  Switch,
  Modal,
  Form,
  Input,
  InputNumber,
  message,
  Tooltip,
} from 'antd'
import {
  MonitorOutlined,
  DashboardOutlined,
  AlertOutlined,
  CloudServerOutlined,
  DatabaseOutlined,
  ApiOutlined,
  ReloadOutlined,
  SettingOutlined,
  BellOutlined,
  LineChartOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
} from '@ant-design/icons'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text } = Typography
const { TabPane } = Tabs
const { Option } = Select
const { RangePicker } = DatePicker

interface SystemMetrics {
  timestamp: string
  cpu_usage: number
  memory_usage: number
  disk_usage: number
  network_in: number
  network_out: number
  active_connections: number
  response_time: number
  throughput: number
  error_count: number
}

interface Alert {
  alert_id: string
  name: string
  severity: 'info' | 'warning' | 'error' | 'critical'
  status: 'active' | 'resolved' | 'silenced'
  description: string
  component: string
  triggered_at: string
  resolved_at?: string
  threshold_value: number
  current_value: number
}

interface MonitoringRule {
  rule_id: string
  name: string
  metric: string
  operator: 'gt' | 'lt' | 'eq' | 'ne'
  threshold: number
  duration: number
  enabled: boolean
  severity: 'info' | 'warning' | 'error' | 'critical'
  description: string
}

interface ServiceHealth {
  service_name: string
  status: 'healthy' | 'degraded' | 'unhealthy'
  response_time: number
  uptime: number
  last_check: string
  error_rate: number
}

const SystemMonitoringPage: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [rules, setRules] = useState<MonitoringRule[]>([])
  const [services, setServices] = useState<ServiceHealth[]>([])
  const [loading, setLoading] = useState(false)
  const [realTimeMode, setRealTimeMode] = useState(true)
  const [ruleModalVisible, setRuleModalVisible] = useState(false)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchMonitoringData()
    let interval: ReturnType<typeof setTimeout>
    if (realTimeMode) {
      interval = setInterval(fetchMonitoringData, 5000)
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [realTimeMode])

  const fetchMonitoringData = async () => {
    setLoading(true)
    try {
      const [metricsRes, alertsRes, rulesRes, servicesRes] =
        await Promise.allSettled([
          apiFetch(buildApiUrl('/api/v1/platform/monitoring/metrics')),
          apiFetch(buildApiUrl('/api/v1/platform/monitoring/alerts')),
          apiFetch(buildApiUrl('/api/v1/platform/monitoring/rules')),
          apiFetch(buildApiUrl('/api/v1/platform/monitoring/services')),
        ])

      if (metricsRes.status === 'fulfilled') {
        const data = await metricsRes.value.json()
        setMetrics(data.metrics || [])
      }

      if (alertsRes.status === 'fulfilled') {
        const data = await alertsRes.value.json()
        setAlerts(data.alerts || [])
      }

      if (rulesRes.status === 'fulfilled') {
        const data = await rulesRes.value.json()
        setRules(data.rules || [])
      }

      if (servicesRes.status === 'fulfilled') {
        const data = await servicesRes.value.json()
        setServices(data.services || [])
      }
    } catch (error) {
      logger.error('获取监控数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCreateRule = async (values: any) => {
    try {
      const response = await apiFetch(
        buildApiUrl('/api/v1/platform/monitoring/rules'),
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(values),
        }
      )
      await response.json().catch(() => null)
      message.success('监控规则创建成功')
      setRuleModalVisible(false)
      form.resetFields()
      fetchMonitoringData()
    } catch (error) {
      message.error('创建失败')
    }
  }

  const handleToggleRule = async (ruleId: string, enabled: boolean) => {
    try {
      const response = await apiFetch(
        buildApiUrl(`/api/v1/platform/monitoring/rules/${ruleId}`),
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ enabled }),
        }
      )
      await response.json().catch(() => null)
      message.success(enabled ? '规则已启用' : '规则已禁用')
      fetchMonitoringData()
    } catch (error) {
      message.error('操作失败')
    }
  }

  const handleResolveAlert = async (alertId: string) => {
    try {
      const response = await apiFetch(
        buildApiUrl(`/api/v1/platform/monitoring/alerts/${alertId}/resolve`),
        {
          method: 'POST',
        }
      )
      await response.json().catch(() => null)
      message.success('告警已解决')
      fetchMonitoringData()
    } catch (error) {
      message.error('操作失败')
    }
  }

  const getSeverityColor = (severity: string) => {
    const colors = {
      info: 'blue',
      warning: 'orange',
      error: 'red',
      critical: 'purple',
    }
    return colors[severity] || 'default'
  }

  const getStatusColor = (status: string) => {
    const colors = {
      healthy: 'success',
      degraded: 'warning',
      unhealthy: 'error',
      active: 'error',
      resolved: 'success',
      silenced: 'default',
    }
    return colors[status] || 'default'
  }

  const alertColumns: ColumnsType<Alert> = [
    {
      title: '告警名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: severity => (
        <Tag color={getSeverityColor(severity)}>{severity.toUpperCase()}</Tag>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: status => {
        const statusConfig = {
          active: { color: 'error', text: '活跃', icon: <AlertOutlined /> },
          resolved: {
            color: 'success',
            text: '已解决',
            icon: <CheckCircleOutlined />,
          },
          silenced: {
            color: 'default',
            text: '已静默',
            icon: <CloseCircleOutlined />,
          },
        }
        const config = statusConfig[status]
        return (
          <Tag color={config.color} icon={config.icon}>
            {config.text}
          </Tag>
        )
      },
    },
    {
      title: '组件',
      dataIndex: 'component',
      key: 'component',
    },
    {
      title: '当前值',
      key: 'current_value',
      render: (_, record) => (
        <div>
          <div>{record.current_value}</div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            阈值: {record.threshold_value}
          </Text>
        </div>
      ),
    },
    {
      title: '触发时间',
      dataIndex: 'triggered_at',
      key: 'triggered_at',
      render: time => new Date(time).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          {record.status === 'active' && (
            <Button
              size="small"
              type="primary"
              onClick={() => handleResolveAlert(record.alert_id)}
            >
              解决
            </Button>
          )}
        </Space>
      ),
    },
  ]

  const ruleColumns: ColumnsType<MonitoringRule> = [
    {
      title: '规则名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '监控指标',
      dataIndex: 'metric',
      key: 'metric',
    },
    {
      title: '条件',
      key: 'condition',
      render: (_, record) => `${record.operator} ${record.threshold}`,
    },
    {
      title: '持续时间',
      dataIndex: 'duration',
      key: 'duration',
      render: duration => `${duration}s`,
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: severity => (
        <Tag color={getSeverityColor(severity)}>{severity.toUpperCase()}</Tag>
      ),
    },
    {
      title: '状态',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled, record) => (
        <Switch
          size="small"
          checked={enabled}
          onChange={checked => handleToggleRule(record.rule_id, checked)}
        />
      ),
    },
  ]

  const serviceColumns: ColumnsType<ServiceHealth> = [
    {
      title: '服务名称',
      dataIndex: 'service_name',
      key: 'service_name',
      render: name => (
        <Space>
          <ApiOutlined />
          <span>{name}</span>
        </Space>
      ),
    },
    {
      title: '健康状态',
      dataIndex: 'status',
      key: 'status',
      render: status => {
        const statusConfig = {
          healthy: {
            color: 'success',
            text: '健康',
            icon: <CheckCircleOutlined />,
          },
          degraded: {
            color: 'warning',
            text: '降级',
            icon: <ExclamationCircleOutlined />,
          },
          unhealthy: {
            color: 'error',
            text: '异常',
            icon: <CloseCircleOutlined />,
          },
        }
        const config = statusConfig[status]
        return (
          <Tag color={config.color} icon={config.icon}>
            {config.text}
          </Tag>
        )
      },
    },
    {
      title: '响应时间',
      dataIndex: 'response_time',
      key: 'response_time',
      render: time => `${time}ms`,
    },
    {
      title: '运行时间',
      dataIndex: 'uptime',
      key: 'uptime',
      render: uptime => {
        const hours = Math.floor(uptime / 3600)
        const minutes = Math.floor((uptime % 3600) / 60)
        return `${hours}h ${minutes}m`
      },
    },
    {
      title: '错误率',
      dataIndex: 'error_rate',
      key: 'error_rate',
      render: rate => `${rate}%`,
    },
    {
      title: '最后检查',
      dataIndex: 'last_check',
      key: 'last_check',
      render: time => new Date(time).toLocaleString(),
    },
  ]

  const currentMetrics = metrics.length > 0 ? metrics[metrics.length - 1] : null
  const activeAlerts = alerts.filter(alert => alert.status === 'active')
  const criticalAlerts = activeAlerts.filter(
    alert => alert.severity === 'critical'
  )

  const chartData = metrics.slice(-20).map(metric => ({
    ...metric,
    time: new Date(metric.timestamp).toLocaleTimeString(),
  }))

  const serviceStatusData = [
    {
      name: '健康',
      value: services.filter(s => s.status === 'healthy').length,
      fill: '#52c41a',
    },
    {
      name: '降级',
      value: services.filter(s => s.status === 'degraded').length,
      fill: '#faad14',
    },
    {
      name: '异常',
      value: services.filter(s => s.status === 'unhealthy').length,
      fill: '#f5222d',
    },
  ]

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
        <Title level={2}>系统监控</Title>
        <Space>
          <Text>实时模式:</Text>
          <Switch
            checked={realTimeMode}
            onChange={setRealTimeMode}
            checkedChildren="开"
            unCheckedChildren="关"
          />
          <Button
            icon={<ReloadOutlined />}
            onClick={fetchMonitoringData}
            loading={loading}
          >
            刷新
          </Button>
          <Button
            type="primary"
            icon={<SettingOutlined />}
            onClick={() => setRuleModalVisible(true)}
          >
            添加规则
          </Button>
        </Space>
      </div>

      {criticalAlerts.length > 0 && (
        <Alert
          message={`检测到 ${criticalAlerts.length} 个严重告警`}
          description="请立即处理严重告警以确保系统稳定运行"
          type="error"
          showIcon
          closable
          style={{ marginBottom: 24 }}
        />
      )}

      {currentMetrics && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col span={4}>
            <Card>
              <Statistic
                title="CPU使用率"
                value={currentMetrics.cpu_usage}
                suffix="%"
                prefix={<DashboardOutlined />}
                valueStyle={{
                  color:
                    currentMetrics.cpu_usage > 80
                      ? '#f5222d'
                      : currentMetrics.cpu_usage > 60
                        ? '#faad14'
                        : '#52c41a',
                }}
              />
              <Progress
                percent={currentMetrics.cpu_usage}
                size="small"
                showInfo={false}
                status={currentMetrics.cpu_usage > 80 ? 'exception' : 'active'}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="内存使用率"
                value={currentMetrics.memory_usage}
                suffix="%"
                prefix={<DatabaseOutlined />}
                valueStyle={{
                  color:
                    currentMetrics.memory_usage > 80
                      ? '#f5222d'
                      : currentMetrics.memory_usage > 60
                        ? '#faad14'
                        : '#52c41a',
                }}
              />
              <Progress
                percent={currentMetrics.memory_usage}
                size="small"
                showInfo={false}
                status={
                  currentMetrics.memory_usage > 80 ? 'exception' : 'active'
                }
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="磁盘使用率"
                value={currentMetrics.disk_usage}
                suffix="%"
                prefix={<CloudServerOutlined />}
                valueStyle={{
                  color:
                    currentMetrics.disk_usage > 80
                      ? '#f5222d'
                      : currentMetrics.disk_usage > 60
                        ? '#faad14'
                        : '#52c41a',
                }}
              />
              <Progress
                percent={currentMetrics.disk_usage}
                size="small"
                showInfo={false}
                status={currentMetrics.disk_usage > 80 ? 'exception' : 'active'}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="活跃连接"
                value={currentMetrics.active_connections}
                prefix={<SyncOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="响应时间"
                value={currentMetrics.response_time}
                suffix="ms"
                prefix={<LineChartOutlined />}
                valueStyle={{
                  color:
                    currentMetrics.response_time > 1000
                      ? '#f5222d'
                      : currentMetrics.response_time > 500
                        ? '#faad14'
                        : '#52c41a',
                }}
              />
            </Card>
          </Col>
          <Col span={4}>
            <Card>
              <Statistic
                title="错误数量"
                value={currentMetrics.error_count}
                prefix={<ExclamationCircleOutlined />}
                valueStyle={{
                  color: currentMetrics.error_count > 0 ? '#f5222d' : '#52c41a',
                }}
              />
            </Card>
          </Col>
        </Row>
      )}

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={18}>
          <Card title="系统性能趋势" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <RechartsTooltip />
                <Line
                  type="monotone"
                  dataKey="cpu_usage"
                  stroke="#8884d8"
                  name="CPU%"
                />
                <Line
                  type="monotone"
                  dataKey="memory_usage"
                  stroke="#82ca9d"
                  name="内存%"
                />
                <Line
                  type="monotone"
                  dataKey="response_time"
                  stroke="#ffc658"
                  name="响应时间"
                />
              </LineChart>
            </ResponsiveContainer>
          </Card>
        </Col>
        <Col span={6}>
          <Card title="服务健康状态" size="small">
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={serviceStatusData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={80}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {serviceStatusData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <RechartsTooltip />
              </PieChart>
            </ResponsiveContainer>
            <div style={{ textAlign: 'center', marginTop: 16 }}>
              {serviceStatusData.map((item, index) => (
                <div key={index} style={{ marginBottom: 4 }}>
                  <span style={{ color: item.fill, marginRight: 8 }}>●</span>
                  <Text>
                    {item.name}: {item.value}
                  </Text>
                </div>
              ))}
            </div>
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="alerts">
        <TabPane tab={`告警 (${activeAlerts.length})`} key="alerts">
          <Table
            columns={alertColumns}
            dataSource={alerts}
            rowKey="alert_id"
            pagination={{ pageSize: 10 }}
            size="small"
          />
        </TabPane>
        <TabPane tab="监控规则" key="rules">
          <Table
            columns={ruleColumns}
            dataSource={rules}
            rowKey="rule_id"
            pagination={{ pageSize: 10 }}
            size="small"
          />
        </TabPane>
        <TabPane tab="服务状态" key="services">
          <Table
            columns={serviceColumns}
            dataSource={services}
            rowKey="service_name"
            pagination={{ pageSize: 10 }}
            size="small"
          />
        </TabPane>
      </Tabs>

      <Modal
        title="添加监控规则"
        open={ruleModalVisible}
        onCancel={() => {
          setRuleModalVisible(false)
          form.resetFields()
        }}
        footer={null}
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={handleCreateRule}>
          <Form.Item
            name="name"
            label="规则名称"
            rules={[{ required: true, message: '请输入规则名称' }]}
          >
            <Input placeholder="规则名称" />
          </Form.Item>

          <Form.Item
            name="metric"
            label="监控指标"
            rules={[{ required: true, message: '请选择监控指标' }]}
          >
            <Select placeholder="选择监控指标">
              <Option value="cpu_usage">CPU使用率</Option>
              <Option value="memory_usage">内存使用率</Option>
              <Option value="disk_usage">磁盘使用率</Option>
              <Option value="response_time">响应时间</Option>
              <Option value="error_rate">错误率</Option>
              <Option value="throughput">吞吐量</Option>
            </Select>
          </Form.Item>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="operator"
                label="操作符"
                rules={[{ required: true, message: '请选择操作符' }]}
              >
                <Select placeholder="选择操作符">
                  <Option value="gt">大于 (&gt;)</Option>
                  <Option value="lt">小于 (&lt;)</Option>
                  <Option value="eq">等于 (=)</Option>
                  <Option value="ne">不等于 (≠)</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="threshold"
                label="阈值"
                rules={[{ required: true, message: '请输入阈值' }]}
              >
                <InputNumber style={{ width: '100%' }} placeholder="阈值" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="duration"
                label="持续时间(秒)"
                rules={[{ required: true, message: '请输入持续时间' }]}
              >
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="秒"
                  min={1}
                />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="severity"
            label="严重程度"
            rules={[{ required: true, message: '请选择严重程度' }]}
          >
            <Select placeholder="选择严重程度">
              <Option value="info">信息</Option>
              <Option value="warning">警告</Option>
              <Option value="error">错误</Option>
              <Option value="critical">严重</Option>
            </Select>
          </Form.Item>

          <Form.Item name="description" label="描述">
            <Input.TextArea rows={3} placeholder="规则描述" />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button
                onClick={() => {
                  setRuleModalVisible(false)
                  form.resetFields()
                }}
              >
                取消
              </Button>
              <Button type="primary" htmlType="submit">
                创建
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default SystemMonitoringPage
