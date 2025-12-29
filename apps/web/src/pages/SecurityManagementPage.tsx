import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Tag,
  Button,
  Space,
  Row,
  Col,
  Statistic,
  Alert,
  Switch,
  Form,
  Input,
  Modal,
  Tabs,
  List,
  Avatar,
  Tooltip,
  Progress,
  Typography,
  message
} from 'antd'
import {
  SafetyOutlined as ShieldOutlined,
  UserOutlined,
  LockOutlined,
  UnlockOutlined,
  WarningOutlined,
  SafetyOutlined,
  AuditOutlined,
  SettingOutlined,
  PlusOutlined,
  DeleteOutlined,
  EditOutlined,
  EyeOutlined
} from '@ant-design/icons'
import apiClient from '../services/apiClient'

const { Text } = Typography
const { TabPane } = Tabs

interface User {
  id: string
  username: string
  role: string
  status: 'active' | 'suspended' | 'inactive'
  lastLogin: string
  permissions: string[]
}

interface SecurityEvent {
  id: string
  timestamp: string
  type: 'login' | 'access_denied' | 'permission_change' | 'suspicious_activity'
  user: string
  description: string
  severity: 'low' | 'medium' | 'high' | 'critical'
  ip: string
}

interface SecurityPolicy {
  id: string
  name: string
  description: string
  enabled: boolean
  rules: string[]
  configKey?: string
}

const SecurityManagementPage: React.FC = () => {
  const [users, setUsers] = useState<User[]>([])
  const [securityEvents, setSecurityEvents] = useState<SecurityEvent[]>([])
  const [securityPolicies, setSecurityPolicies] = useState<SecurityPolicy[]>([])
  const [securityMetrics, setSecurityMetrics] = useState<any>(null)

  const [isUserModalVisible, setIsUserModalVisible] = useState(false)
  const [form] = Form.useForm()

  const statusColors = {
    active: 'success',
    suspended: 'warning',
    inactive: 'default'
  }

  const statusTexts = {
    active: '活跃',
    suspended: '暂停',
    inactive: '非活跃'
  }

  const severityColors = {
    low: 'default',
    medium: 'warning',
    high: 'error',
    critical: 'error'
  }

  const severityTexts = {
    low: '低',
    medium: '中',
    high: '高',
    critical: '严重'
  }

  const eventTypeColors = {
    login: 'blue',
    access_denied: 'red',
    permission_change: 'orange',
    suspicious_activity: 'purple'
  }

  const eventTypeTexts = {
    login: '登录',
    access_denied: '访问拒绝',
    permission_change: '权限变更',
    suspicious_activity: '可疑活动'
  }

  useEffect(() => {
    const loadData = async () => {
      await Promise.all([
        loadUsers(),
        loadSecurityAlerts(),
        loadSecurityConfig(),
        loadSecurityMetrics(),
      ])
    }
    loadData()
  }, [])

  const loadUsers = async () => {
    try {
      const response = await apiClient.get('/auth/users')
      const list = Array.isArray(response.data?.users) ? response.data.users : []
      const mapped = list.map((user: any) => {
        const roles = Array.isArray(user.roles) ? user.roles : []
        const role = user.is_superuser ? '系统管理员' : roles[0] || '用户'
        return {
          id: String(user.id),
          username: String(user.username || ''),
          role,
          status: user.is_active ? 'active' : 'inactive',
          lastLogin: user.last_login ? new Date(user.last_login).toLocaleString() : '-',
          permissions: Array.isArray(user.permissions) ? user.permissions : [],
        }
      })
      setUsers(mapped)
    } catch (error) {
      setUsers([])
    }
  }

  const loadSecurityAlerts = async () => {
    try {
      const response = await apiClient.get('/security/alerts')
      const list = Array.isArray(response.data?.alerts) ? response.data.alerts : []
      const mapped = list.map((alert: any) => {
        const alertType = String(alert.alert_type || '')
        const type: SecurityEvent['type'] =
          alertType.includes('unauthorized') ? 'access_denied'
          : alertType.includes('permission') ? 'permission_change'
          : alertType.includes('login') ? 'login'
          : 'suspicious_activity'
        const severity = String(alert.threat_level || 'low') as SecurityEvent['severity']
        return {
          id: String(alert.id),
          timestamp: alert.timestamp ? new Date(alert.timestamp).toLocaleString() : '-',
          type,
          user: alert.user_id || 'unknown',
          description: alert.description || alert.alert_type || '',
          severity,
          ip: alert.source_ip || '-',
        }
      })
      setSecurityEvents(mapped)
    } catch (error) {
      setSecurityEvents([])
    }
  }

  const loadSecurityConfig = async () => {
    try {
      const response = await apiClient.get('/security/config')
      const config = response.data || {}
      setSecurityPolicies([
        {
          id: 'force_https',
          name: '强制HTTPS',
          description: '强制所有请求使用HTTPS',
          enabled: Boolean(config.force_https),
          rules: [`force_https=${Boolean(config.force_https)}`],
          configKey: 'force_https',
        },
        {
          id: 'csp_header',
          name: 'CSP头策略',
          description: '内容安全策略头配置',
          enabled: Boolean(config.csp_header),
          rules: [String(config.csp_header || '未配置')],
        },
        {
          id: 'rate_limit',
          name: '访问频率限制',
          description: '限制每分钟最大请求数',
          enabled: Number(config.max_requests_per_minute || 0) > 0,
          rules: [`max_requests_per_minute=${config.max_requests_per_minute ?? '-'}`],
        },
        {
          id: 'request_size',
          name: '请求大小限制',
          description: '限制单次请求最大大小',
          enabled: Number(config.max_request_size || 0) > 0,
          rules: [`max_request_size=${config.max_request_size ?? '-'}`],
        },
      ])
    } catch (error) {
      setSecurityPolicies([])
    }
  }

  const loadSecurityMetrics = async () => {
    try {
      const response = await apiClient.get('/security/metrics')
      setSecurityMetrics(response.data?.security_metrics || null)
    } catch (error) {
      setSecurityMetrics(null)
    }
  }

  const userColumns = [
    {
      title: '用户名',
      dataIndex: 'username',
      key: 'username',
      render: (text: string) => (
        <Space>
          <Avatar icon={<UserOutlined />} size="small" />
          <strong>{text}</strong>
        </Space>
      )
    },
    {
      title: '角色',
      dataIndex: 'role',
      key: 'role'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: keyof typeof statusColors) => (
        <Tag color={statusColors[status]}>{statusTexts[status]}</Tag>
      )
    },
    {
      title: '权限',
      dataIndex: 'permissions',
      key: 'permissions',
      render: (permissions: string[]) => (
        <Space wrap>
          {permissions.map(permission => (
            <Tag key={permission} size="small">{permission}</Tag>
          ))}
        </Space>
      )
    },
    {
      title: '最后登录',
      dataIndex: 'lastLogin',
      key: 'lastLogin',
      render: (time: string) => (
        <div className="text-xs text-gray-500">{time}</div>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: User) => (
        <Space>
          <Tooltip title="查看详情">
            <Button size="small" icon={<EyeOutlined />} disabled />
          </Tooltip>
          <Tooltip title="编辑">
            <Button size="small" icon={<EditOutlined />} disabled />
          </Tooltip>
          <Tooltip title={record.status === 'active' ? '暂停' : '激活'}>
            <Button 
              size="small" 
              icon={record.status === 'active' ? <LockOutlined /> : <UnlockOutlined />}
              disabled
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button size="small" danger icon={<DeleteOutlined />} disabled />
          </Tooltip>
        </Space>
      )
    }
  ]

  const eventColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 160,
      render: (time: string) => (
        <div className="text-xs text-gray-600">{time}</div>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: keyof typeof eventTypeColors) => (
        <Tag color={eventTypeColors[type]}>
          {eventTypeTexts[type]}
        </Tag>
      )
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: keyof typeof severityColors) => (
        <Tag color={severityColors[severity]}>
          {severityTexts[severity]}
        </Tag>
      )
    },
    {
      title: '用户',
      dataIndex: 'user',
      key: 'user'
    },
    {
      title: 'IP地址',
      dataIndex: 'ip',
      key: 'ip',
      render: (ip: string) => <Text code>{ip}</Text>
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description'
    }
  ]

  const togglePolicy = (policyId: string) => {
    const target = securityPolicies.find((policy) => policy.id === policyId)
    if (!target || !target.configKey) return
    const nextEnabled = !target.enabled
    apiClient
      .put('/security/config', { [target.configKey]: nextEnabled })
      .then(() => {
        setSecurityPolicies(prev =>
          prev.map(policy =>
            policy.id === policyId ? { ...policy, enabled: nextEnabled } : policy
          )
        )
        message.success('策略已更新')
      })
      .catch(() => {
        message.error('策略更新失败')
      })
  }

  const getSecurityScore = () => {
    if (!securityMetrics) {
      const enabledPolicies = securityPolicies.filter(p => p.enabled).length
      const totalPolicies = securityPolicies.length || 1
      return Math.round((enabledPolicies / totalPolicies) * 100)
    }
    const penalty =
      (securityMetrics.active_alerts || 0) * 10 +
      (securityMetrics.critical_alerts || 0) * 20 +
      (securityMetrics.blocked_ips || 0) * 2
    return Math.max(0, 100 - penalty)
  }

  const activeUsers = users.filter(u => u.status === 'active').length
  const suspendedUsers = users.filter(u => u.status === 'suspended').length
  const criticalEvents = securityEvents.filter(e => e.severity === 'critical' || e.severity === 'high').length
  const securityScore = getSecurityScore()

  return (
    <div className="p-6">
        <div className="mb-6">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-2xl font-bold">企业级安全管理</h1>
            <Space>
              <Button icon={<AuditOutlined />}>
                生成安全报告
              </Button>
              <Button icon={<SettingOutlined />}>
                安全设置
              </Button>
            </Space>
          </div>

          <Row gutter={16} className="mb-6">
            <Col span={6}>
              <Card>
                <Statistic
                  title="安全评分"
                  value={securityScore}
                  suffix="%"
                  valueStyle={{ color: securityScore >= 80 ? '#3f8600' : '#cf1322' }}
                  prefix={<ShieldOutlined />}
                />
                <Progress 
                  percent={securityScore} 
                  size="small" 
                  strokeColor={securityScore >= 80 ? '#52c41a' : '#ff4d4f'}
                  className="mt-2"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="活跃用户"
                  value={activeUsers}
                  suffix={`/ ${users.length}`}
                  valueStyle={{ color: '#1890ff' }}
                  prefix={<UserOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="暂停用户"
                  value={suspendedUsers}
                  valueStyle={{ color: suspendedUsers > 0 ? '#cf1322' : '#3f8600' }}
                  prefix={<LockOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="高危事件"
                  value={criticalEvents}
                  valueStyle={{ color: criticalEvents > 0 ? '#cf1322' : '#3f8600' }}
                  prefix={<WarningOutlined />}
                />
              </Card>
            </Col>
          </Row>

          {securityScore < 80 && (
            <Alert
              message="安全警告"
              description="当前安全评分较低，建议启用更多安全策略以提高系统安全性"
              type="warning"
              showIcon
              closable
              className="mb-4"
            />
          )}

          {criticalEvents > 0 && (
            <Alert
              message="安全提醒"
              description={`检测到 ${criticalEvents} 个高危安全事件，请及时处理`}
              type="error"
              showIcon
              closable
              className="mb-4"
            />
          )}
        </div>

        <Tabs defaultActiveKey="users">
          <TabPane tab="用户管理" key="users">
            <Card 
              title="用户列表" 
              extra={
                <Button 
                  type="primary" 
                  icon={<PlusOutlined />}
                  onClick={() => setIsUserModalVisible(true)}
                >
                  添加用户
                </Button>
              }
            >
              <Table
                columns={userColumns}
                dataSource={users}
                rowKey="id"
                pagination={false}
                size="middle"
              />
            </Card>
          </TabPane>

          <TabPane tab="安全事件" key="events">
            <Card title="安全事件日志">
              <Table
                columns={eventColumns}
                dataSource={securityEvents}
                rowKey="id"
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showTotal: (total) => `共 ${total} 条事件`
                }}
                size="small"
              />
            </Card>
          </TabPane>

          <TabPane tab="安全策略" key="policies">
            <Row gutter={16}>
              {securityPolicies.map(policy => (
                <Col span={12} key={policy.id} className="mb-4">
                  <Card 
                    title={
                      <Space>
                        <SafetyOutlined />
                        {policy.name}
                        <Switch 
                          checked={policy.enabled}
                          onChange={() => togglePolicy(policy.id)}
                          size="small"
                          disabled={!policy.configKey}
                        />
                      </Space>
                    }
                  >
                    <div className="mb-3">
                      <Text type="secondary">{policy.description}</Text>
                    </div>
                    <List
                      size="small"
                      dataSource={policy.rules}
                      renderItem={rule => (
                        <List.Item>
                          <Text style={{ fontSize: '12px' }}>• {rule}</Text>
                        </List.Item>
                      )}
                    />
                  </Card>
                </Col>
              ))}
            </Row>
          </TabPane>

          <TabPane tab="权限管理" key="permissions">
            <Card title="权限矩阵">
              <div className="text-center text-gray-500 py-8">
                <SettingOutlined style={{ fontSize: 48, marginBottom: 16 }} />
                <div>权限来自角色与API权限配置</div>
              </div>
            </Card>
          </TabPane>
        </Tabs>

        <Modal
          title="添加新用户"
          open={isUserModalVisible}
          onCancel={() => setIsUserModalVisible(false)}
          onOk={form.submit}
          okText="创建"
          cancelText="取消"
        >
          <Form
            form={form}
            layout="vertical"
          onFinish={(values) => {
            apiClient
              .post('/auth/register', values)
              .then(() => {
                message.success('用户已创建')
                setIsUserModalVisible(false)
                form.resetFields()
                loadUsers()
              })
              .catch(() => {
                message.error('用户创建失败')
              })
          }}
          >
            <Form.Item
              name="username"
              label="用户名"
              rules={[{ required: true, message: '请输入用户名' }]}
            >
              <Input placeholder="请输入用户名" />
            </Form.Item>
            <Form.Item
              name="password"
              label="密码"
              rules={[{ required: true, message: '请输入密码' }]}
            >
              <Input.Password placeholder="请输入密码" />
            </Form.Item>
            <Form.Item name="email" label="邮箱">
              <Input placeholder="请输入邮箱" />
            </Form.Item>
            <Form.Item name="full_name" label="姓名">
              <Input placeholder="请输入姓名" />
            </Form.Item>
          </Form>
        </Modal>
    </div>
  )
}

export default SecurityManagementPage
