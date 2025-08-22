import React, { useState, useEffect } from 'react'
import { 
  Card, 
  Row, 
  Col, 
  Button, 
  Space, 
  Table, 
  Form,
  Input,
  Select,
  Tag,
  Statistic,
  Alert,
  Typography,
  Divider,
  Tabs,
  Modal,
  Switch,
  DatePicker,
  Avatar,
  List,
  Progress,
  Tooltip,
  Badge
} from 'antd'
import { 
  UserOutlined,
  SafetyOutlined,
  KeyOutlined,
  LockOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  PlusOutlined,
  SafetyOutlined as ShieldOutlined,
  TeamOutlined,
  SettingOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ApiOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TabPane } = Tabs
const { RangePicker } = DatePicker

// 用户角色定义
const USER_ROLES = {
  SUPER_ADMIN: 'super_admin',
  ADMIN: 'admin', 
  USER: 'user',
  READONLY: 'readonly',
  API_CLIENT: 'api_client'
}

// 权限定义
const PERMISSIONS = [
  { key: 'users.read', name: '用户查看', category: '用户管理' },
  { key: 'users.write', name: '用户编辑', category: '用户管理' },
  { key: 'users.delete', name: '用户删除', category: '用户管理' },
  { key: 'agents.read', name: '智能体查看', category: '智能体管理' },
  { key: 'agents.write', name: '智能体编辑', category: '智能体管理' },
  { key: 'agents.execute', name: '智能体执行', category: '智能体管理' },
  { key: 'files.read', name: '文件查看', category: '文件管理' },
  { key: 'files.upload', name: '文件上传', category: '文件管理' },
  { key: 'files.delete', name: '文件删除', category: '文件管理' },
  { key: 'system.read', name: '系统监控', category: '系统管理' },
  { key: 'system.admin', name: '系统管理', category: '系统管理' }
]

// 生成用户数据
const generateUsers = () => {
  const users = []
  const roles = Object.values(USER_ROLES)
  
  for (let i = 0; i < 20; i++) {
    users.push({
      id: i + 1,
      username: `user_${i + 1}`,
      email: `user${i + 1}@example.com`,
      displayName: `用户 ${i + 1}`,
      role: roles[Math.floor(Math.random() * roles.length)],
      status: Math.random() > 0.2 ? 'active' : 'inactive',
      lastLogin: new Date(Date.now() - Math.random() * 7 * 24 * 3600 * 1000),
      createdAt: new Date(Date.now() - Math.random() * 180 * 24 * 3600 * 1000),
      loginCount: Math.floor(Math.random() * 1000),
      twoFactorEnabled: Math.random() > 0.5,
      permissions: PERMISSIONS.slice(0, Math.floor(Math.random() * PERMISSIONS.length) + 1)
        .map(p => p.key)
    })
  }
  
  return users
}

// 生成API密钥数据
const generateApiKeys = () => {
  const keys = []
  
  for (let i = 0; i < 10; i++) {
    keys.push({
      id: i + 1,
      name: `API Key ${i + 1}`,
      key: `ak_${Math.random().toString(36).substr(2, 20)}`,
      description: `用于${['内部服务', '第三方集成', '测试环境', '生产环境'][Math.floor(Math.random() * 4)]}`,
      createdAt: new Date(Date.now() - Math.random() * 90 * 24 * 3600 * 1000),
      lastUsed: Math.random() > 0.3 ? new Date(Date.now() - Math.random() * 7 * 24 * 3600 * 1000) : null,
      expiresAt: Math.random() > 0.5 ? new Date(Date.now() + Math.random() * 365 * 24 * 3600 * 1000) : null,
      status: Math.random() > 0.1 ? 'active' : 'revoked',
      usageCount: Math.floor(Math.random() * 10000),
      rateLimit: Math.floor(Math.random() * 1000) + 100,
      scopes: PERMISSIONS.slice(0, Math.floor(Math.random() * 5) + 1).map(p => p.key)
    })
  }
  
  return keys
}

// 生成会话数据
const generateSessions = () => {
  const sessions = []
  
  for (let i = 0; i < 15; i++) {
    sessions.push({
      id: i + 1,
      userId: Math.floor(Math.random() * 20) + 1,
      username: `user_${Math.floor(Math.random() * 20) + 1}`,
      ipAddress: `192.168.1.${Math.floor(Math.random() * 254) + 1}`,
      userAgent: ['Chrome/91.0', 'Firefox/89.0', 'Safari/14.0', 'Edge/91.0'][Math.floor(Math.random() * 4)],
      loginTime: new Date(Date.now() - Math.random() * 24 * 3600 * 1000),
      lastActivity: new Date(Date.now() - Math.random() * 3600 * 1000),
      status: Math.random() > 0.3 ? 'active' : 'expired',
      location: ['北京', '上海', '深圳', '杭州'][Math.floor(Math.random() * 4)],
      deviceType: ['Desktop', 'Mobile', 'Tablet'][Math.floor(Math.random() * 3)]
    })
  }
  
  return sessions.sort((a, b) => b.loginTime.getTime() - a.loginTime.getTime())
}

const AuthManagementPage: React.FC = () => {
  const [users, setUsers] = useState(() => generateUsers())
  const [apiKeys, setApiKeys] = useState(() => generateApiKeys())
  const [sessions, setSessions] = useState(() => generateSessions())
  const [userModalVisible, setUserModalVisible] = useState(false)
  const [apiKeyModalVisible, setApiKeyModalVisible] = useState(false)
  const [editingUser, setEditingUser] = useState<any>(null)
  const [editingApiKey, setEditingApiKey] = useState<any>(null)

  const getRoleColor = (role: string): string => {
    const colors = {
      [USER_ROLES.SUPER_ADMIN]: '#ff4d4f',
      [USER_ROLES.ADMIN]: '#fa8c16', 
      [USER_ROLES.USER]: '#1890ff',
      [USER_ROLES.READONLY]: '#52c41a',
      [USER_ROLES.API_CLIENT]: '#722ed1'
    }
    return colors[role] || '#666'
  }

  const getRoleName = (role: string): string => {
    const names = {
      [USER_ROLES.SUPER_ADMIN]: '超级管理员',
      [USER_ROLES.ADMIN]: '管理员',
      [USER_ROLES.USER]: '普通用户', 
      [USER_ROLES.READONLY]: '只读用户',
      [USER_ROLES.API_CLIENT]: 'API客户端'
    }
    return names[role] || role
  }

  // 统计信息
  const stats = {
    totalUsers: users.length,
    activeUsers: users.filter(u => u.status === 'active').length,
    totalApiKeys: apiKeys.length,
    activeApiKeys: apiKeys.filter(k => k.status === 'active').length,
    activeSessions: sessions.filter(s => s.status === 'active').length,
    twoFactorEnabled: users.filter(u => u.twoFactorEnabled).length
  }

  // 统计卡片
  const StatsCards = () => (
    <Row gutter={16}>
      <Col span={6}>
        <Card>
          <Statistic
            title="总用户数"
            value={stats.totalUsers}
            prefix={<UserOutlined />}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="活跃用户"
            value={stats.activeUsers}
            prefix={<CheckCircleOutlined />}
            valueStyle={{ color: '#52c41a' }}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="API密钥"
            value={stats.totalApiKeys}
            prefix={<KeyOutlined />}
          />
        </Card>
      </Col>
      <Col span={6}>
        <Card>
          <Statistic
            title="活跃会话"
            value={stats.activeSessions}
            prefix={<ShieldOutlined />}
          />
        </Card>
      </Col>
    </Row>
  )

  // 用户表格
  const UserTable = () => {
    const columns = [
      {
        title: '用户',
        key: 'user',
        render: (record: any) => (
          <Space>
            <Avatar icon={<UserOutlined />} />
            <div>
              <div>{record.displayName}</div>
              <Text type="secondary" style={{ fontSize: '12px' }}>{record.email}</Text>
            </div>
          </Space>
        )
      },
      {
        title: '角色',
        dataIndex: 'role',
        key: 'role',
        render: (role: string) => (
          <Tag color={getRoleColor(role)}>{getRoleName(role)}</Tag>
        )
      },
      {
        title: '状态',
        dataIndex: 'status',
        key: 'status',
        render: (status: string) => (
          <Badge 
            status={status === 'active' ? 'success' : 'default'} 
            text={status === 'active' ? '活跃' : '非活跃'}
          />
        )
      },
      {
        title: '双因子认证',
        dataIndex: 'twoFactorEnabled',
        key: 'twoFactor',
        render: (enabled: boolean) => (
          <Tag color={enabled ? 'green' : 'default'}>
            {enabled ? '已启用' : '未启用'}
          </Tag>
        )
      },
      {
        title: '最后登录',
        dataIndex: 'lastLogin',
        key: 'lastLogin',
        render: (time: Date) => (
          <Tooltip title={time.toLocaleString()}>
            <Text type="secondary">{time.toLocaleDateString()}</Text>
          </Tooltip>
        )
      },
      {
        title: '登录次数',
        dataIndex: 'loginCount',
        key: 'loginCount'
      },
      {
        title: '操作',
        key: 'actions',
        render: (record: any) => (
          <Space>
            <Button icon={<EyeOutlined />} size="small" />
            <Button 
              icon={<EditOutlined />} 
              size="small" 
              onClick={() => {
                setEditingUser(record)
                setUserModalVisible(true)
              }}
            />
            <Button icon={<DeleteOutlined />} size="small" danger />
          </Space>
        )
      }
    ]

    return (
      <Card 
        title="用户管理" 
        size="small"
        extra={
          <Button 
            type="primary" 
            icon={<PlusOutlined />}
            onClick={() => {
              setEditingUser(null)
              setUserModalVisible(true)
            }}
          >
            添加用户
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={users}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 10 }}
          scroll={{ x: 1000 }}
        />
      </Card>
    )
  }

  // API密钥表格
  const ApiKeyTable = () => {
    const columns = [
      {
        title: '密钥名称',
        dataIndex: 'name',
        key: 'name'
      },
      {
        title: '密钥',
        dataIndex: 'key',
        key: 'key',
        render: (key: string) => (
          <Text code>{key.substr(0, 20)}...</Text>
        )
      },
      {
        title: '描述',
        dataIndex: 'description',
        key: 'description',
        ellipsis: true
      },
      {
        title: '状态',
        dataIndex: 'status',
        key: 'status',
        render: (status: string) => (
          <Tag color={status === 'active' ? 'green' : 'red'}>
            {status === 'active' ? '活跃' : '已撤销'}
          </Tag>
        )
      },
      {
        title: '使用次数',
        dataIndex: 'usageCount',
        key: 'usageCount'
      },
      {
        title: '速率限制',
        dataIndex: 'rateLimit',
        key: 'rateLimit',
        render: (limit: number) => `${limit}/分钟`
      },
      {
        title: '最后使用',
        dataIndex: 'lastUsed',
        key: 'lastUsed',
        render: (time: Date | null) => 
          time ? time.toLocaleDateString() : '从未使用'
      },
      {
        title: '过期时间',
        dataIndex: 'expiresAt',
        key: 'expiresAt',
        render: (time: Date | null) => 
          time ? time.toLocaleDateString() : '永不过期'
      },
      {
        title: '操作',
        key: 'actions',
        render: (record: any) => (
          <Space>
            <Button icon={<EyeOutlined />} size="small" />
            <Button 
              icon={<EditOutlined />} 
              size="small"
              onClick={() => {
                setEditingApiKey(record)
                setApiKeyModalVisible(true)
              }}
            />
            <Button icon={<DeleteOutlined />} size="small" danger />
          </Space>
        )
      }
    ]

    return (
      <Card 
        title="API密钥管理" 
        size="small"
        extra={
          <Button 
            type="primary" 
            icon={<PlusOutlined />}
            onClick={() => {
              setEditingApiKey(null)
              setApiKeyModalVisible(true)
            }}
          >
            创建密钥
          </Button>
        }
      >
        <Table
          columns={columns}
          dataSource={apiKeys}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 8 }}
          scroll={{ x: 1200 }}
        />
      </Card>
    )
  }

  // 会话管理
  const SessionTable = () => {
    const columns = [
      {
        title: '用户',
        dataIndex: 'username',
        key: 'username',
        render: (username: string) => (
          <Space>
            <Avatar size="small" icon={<UserOutlined />} />
            <Text>{username}</Text>
          </Space>
        )
      },
      {
        title: 'IP地址',
        dataIndex: 'ipAddress',
        key: 'ipAddress'
      },
      {
        title: '位置',
        dataIndex: 'location',
        key: 'location'
      },
      {
        title: '设备类型',
        dataIndex: 'deviceType',
        key: 'deviceType',
        render: (type: string) => (
          <Tag>{type}</Tag>
        )
      },
      {
        title: '浏览器',
        dataIndex: 'userAgent',
        key: 'userAgent',
        ellipsis: true
      },
      {
        title: '登录时间',
        dataIndex: 'loginTime',
        key: 'loginTime',
        render: (time: Date) => time.toLocaleString()
      },
      {
        title: '最后活动',
        dataIndex: 'lastActivity',
        key: 'lastActivity',
        render: (time: Date) => (
          <Tooltip title={time.toLocaleString()}>
            <Text type="secondary">
              {Math.round((Date.now() - time.getTime()) / 60000)}分钟前
            </Text>
          </Tooltip>
        )
      },
      {
        title: '状态',
        dataIndex: 'status',
        key: 'status',
        render: (status: string) => (
          <Badge 
            status={status === 'active' ? 'processing' : 'default'} 
            text={status === 'active' ? '活跃' : '已过期'}
          />
        )
      },
      {
        title: '操作',
        key: 'actions',
        render: (record: any) => (
          <Button size="small" danger disabled={record.status !== 'active'}>
            强制下线
          </Button>
        )
      }
    ]

    return (
      <Card title="会话管理" size="small">
        <Table
          columns={columns}
          dataSource={sessions}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 8 }}
          scroll={{ x: 1200 }}
        />
      </Card>
    )
  }

  // 权限管理
  const PermissionManagement = () => {
    const groupedPermissions = PERMISSIONS.reduce((acc, perm) => {
      if (!acc[perm.category]) {
        acc[perm.category] = []
      }
      acc[perm.category].push(perm)
      return acc
    }, {} as Record<string, typeof PERMISSIONS>)

    return (
      <Card title="权限配置" size="small">
        <Row gutter={16}>
          {Object.entries(groupedPermissions).map(([category, perms]) => (
            <Col span={8} key={category}>
              <Card title={category} size="small" type="inner">
                <List
                  dataSource={perms}
                  renderItem={perm => (
                    <List.Item>
                      <Space>
                        <Text>{perm.name}</Text>
                        <Text type="secondary" style={{ fontSize: '12px' }}>
                          {perm.key}
                        </Text>
                      </Space>
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
          ))}
        </Row>
      </Card>
    )
  }

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <SafetyOutlined /> 认证授权管理
      </Title>
      <Paragraph type="secondary">
        管理系统用户、角色权限、API密钥和会话安全，提供全面的身份认证和访问控制功能
      </Paragraph>
      
      <Divider />

      <Tabs defaultActiveKey="1">
        <TabPane tab="总览" key="1">
          <Space direction="vertical" size="large" style={{ width: '100%' }}>
            <StatsCards />

            <Row gutter={16}>
              <Col span={12}>
                <Card title="双因子认证分布" size="small">
                  <div style={{ textAlign: 'center' }}>
                    <Progress
                      type="circle"
                      percent={Math.round((stats.twoFactorEnabled / stats.totalUsers) * 100)}
                      format={percent => (
                        <div>
                          <div>{percent}%</div>
                          <div style={{ fontSize: '12px', color: '#666' }}>
                            {stats.twoFactorEnabled}/{stats.totalUsers}
                          </div>
                        </div>
                      )}
                    />
                    <div style={{ marginTop: 16 }}>
                      <Text>双因子认证启用率</Text>
                    </div>
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="用户角色分布" size="small">
                  <List
                    dataSource={Object.values(USER_ROLES).map(role => ({
                      role,
                      count: users.filter(u => u.role === role).length
                    }))}
                    renderItem={item => (
                      <List.Item>
                        <List.Item.Meta
                          title={
                            <Space>
                              <Tag color={getRoleColor(item.role)}>
                                {getRoleName(item.role)}
                              </Tag>
                              <Text>{item.count} 人</Text>
                            </Space>
                          }
                        />
                      </List.Item>
                    )}
                  />
                </Card>
              </Col>
            </Row>
          </Space>
        </TabPane>

        <TabPane tab="用户管理" key="2">
          <UserTable />
        </TabPane>

        <TabPane tab="API密钥" key="3">
          <ApiKeyTable />
        </TabPane>

        <TabPane tab="会话管理" key="4">
          <SessionTable />
        </TabPane>

        <TabPane tab="权限配置" key="5">
          <PermissionManagement />
        </TabPane>
      </Tabs>

      {/* 用户编辑模态框 */}
      <Modal
        title={editingUser ? '编辑用户' : '添加用户'}
        visible={userModalVisible}
        onCancel={() => setUserModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form layout="vertical" initialValues={editingUser}>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="用户名" name="username" rules={[{ required: true }]}>
                <Input />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="邮箱" name="email" rules={[{ required: true, type: 'email' }]}>
                <Input />
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="显示名称" name="displayName">
                <Input />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="角色" name="role" rules={[{ required: true }]}>
                <Select>
                  {Object.values(USER_ROLES).map(role => (
                    <Option key={role} value={role}>{getRoleName(role)}</Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                {editingUser ? '更新' : '创建'}
              </Button>
              <Button onClick={() => setUserModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* API密钥编辑模态框 */}
      <Modal
        title={editingApiKey ? '编辑API密钥' : '创建API密钥'}
        visible={apiKeyModalVisible}
        onCancel={() => setApiKeyModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form layout="vertical" initialValues={editingApiKey}>
          <Form.Item label="密钥名称" name="name" rules={[{ required: true }]}>
            <Input />
          </Form.Item>
          <Form.Item label="描述" name="description">
            <Input.TextArea rows={3} />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="速率限制" name="rateLimit">
                <Input addonAfter="请求/分钟" type="number" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="过期时间" name="expiresAt">
                <DatePicker style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item label="权限范围" name="scopes">
            <Select mode="multiple" placeholder="选择权限">
              {PERMISSIONS.map(perm => (
                <Option key={perm.key} value={perm.key}>
                  {perm.name} ({perm.category})
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                {editingApiKey ? '更新' : '创建'}
              </Button>
              <Button onClick={() => setApiKeyModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default AuthManagementPage