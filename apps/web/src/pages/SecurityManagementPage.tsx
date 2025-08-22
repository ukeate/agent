import React, { useState } from 'react'
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
  Select,
  Modal,
  Tabs,
  List,
  Avatar,
  Tooltip,
  Progress,
  Typography
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

const { Option } = Select
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
}

const SecurityManagementPage: React.FC = () => {
  const [users] = useState<User[]>([
    {
      id: '1',
      username: 'admin',
      role: '系统管理员',
      status: 'active',
      lastLogin: '2024-01-15 14:30:25',
      permissions: ['read', 'write', 'delete', 'admin']
    },
    {
      id: '2',
      username: 'developer',
      role: '开发者',
      status: 'active',
      lastLogin: '2024-01-15 13:45:10',
      permissions: ['read', 'write']
    },
    {
      id: '3',
      username: 'analyst',
      role: '分析师',
      status: 'active',
      lastLogin: '2024-01-15 12:20:30',
      permissions: ['read']
    },
    {
      id: '4',
      username: 'guest',
      role: '访客',
      status: 'suspended',
      lastLogin: '2024-01-14 16:15:45',
      permissions: ['read']
    }
  ])

  const [securityEvents] = useState<SecurityEvent[]>([
    {
      id: '1',
      timestamp: '2024-01-15 14:35:20',
      type: 'suspicious_activity',
      user: 'unknown',
      description: '检测到异常登录尝试',
      severity: 'high',
      ip: '192.168.1.100'
    },
    {
      id: '2',
      timestamp: '2024-01-15 14:30:15',
      type: 'login',
      user: 'admin',
      description: '管理员登录成功',
      severity: 'low',
      ip: '192.168.1.50'
    },
    {
      id: '3',
      timestamp: '2024-01-15 14:25:45',
      type: 'access_denied',
      user: 'guest',
      description: '尝试访问受限资源',
      severity: 'medium',
      ip: '192.168.1.75'
    },
    {
      id: '4',
      timestamp: '2024-01-15 14:20:30',
      type: 'permission_change',
      user: 'admin',
      description: '修改了用户权限',
      severity: 'medium',
      ip: '192.168.1.50'
    }
  ])

  const [securityPolicies, setSecurityPolicies] = useState<SecurityPolicy[]>([
    {
      id: '1',
      name: '密码策略',
      description: '强制使用复杂密码',
      enabled: true,
      rules: ['最少8位字符', '包含大小写字母', '包含数字和特殊字符']
    },
    {
      id: '2',
      name: '会话超时',
      description: '自动注销非活跃用户',
      enabled: true,
      rules: ['30分钟无操作自动注销', '异地登录强制重新认证']
    },
    {
      id: '3',
      name: '访问控制',
      description: '基于角色的访问控制',
      enabled: true,
      rules: ['严格的权限分离', '最小权限原则', '定期权限审核']
    },
    {
      id: '4',
      name: '审计日志',
      description: '记录所有安全相关操作',
      enabled: false,
      rules: ['实时日志记录', '日志加密存储', '180天日志保留']
    }
  ])

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
            <Button size="small" icon={<EyeOutlined />} />
          </Tooltip>
          <Tooltip title="编辑">
            <Button size="small" icon={<EditOutlined />} />
          </Tooltip>
          <Tooltip title={record.status === 'active' ? '暂停' : '激活'}>
            <Button 
              size="small" 
              icon={record.status === 'active' ? <LockOutlined /> : <UnlockOutlined />}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button size="small" danger icon={<DeleteOutlined />} />
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
    setSecurityPolicies(prev => 
      prev.map(policy => 
        policy.id === policyId 
          ? { ...policy, enabled: !policy.enabled }
          : policy
      )
    )
  }

  const getSecurityScore = () => {
    const enabledPolicies = securityPolicies.filter(p => p.enabled).length
    const totalPolicies = securityPolicies.length
    return Math.round((enabledPolicies / totalPolicies) * 100)
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
              variant="warning"
              showIcon
              closable
              className="mb-4"
            />
          )}

          {criticalEvents > 0 && (
            <Alert
              message="安全提醒"
              description={`检测到 ${criticalEvents} 个高危安全事件，请及时处理`}
              variant="destructive"
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
                <div>权限管理功能开发中...</div>
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
              console.log('创建用户:', values)
              setIsUserModalVisible(false)
              form.resetFields()
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
              name="role"
              label="角色"
              rules={[{ required: true, message: '请选择角色' }]}
            >
              <Select placeholder="请选择角色">
                <Option value="admin">系统管理员</Option>
                <Option value="developer">开发者</Option>
                <Option value="analyst">分析师</Option>
                <Option value="guest">访客</Option>
              </Select>
            </Form.Item>
            <Form.Item
              name="permissions"
              label="权限"
              rules={[{ required: true, message: '请选择权限' }]}
            >
              <Select mode="multiple" placeholder="请选择权限">
                <Option value="read">读取</Option>
                <Option value="write">写入</Option>
                <Option value="delete">删除</Option>
                <Option value="admin">管理</Option>
              </Select>
            </Form.Item>
          </Form>
        </Modal>
    </div>
  )
}

export default SecurityManagementPage