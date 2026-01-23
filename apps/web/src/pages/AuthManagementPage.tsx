import React, { useEffect, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  Modal,
  Space,
  Table,
  Tag,
  Typography,
  message,
} from 'antd'
import {
  authService,
  type LoginRequest,
  type RegisterRequest,
  type User,
} from '../services/authService'

const { Title, Paragraph, Text } = Typography

type LoginHistoryItem = {
  timestamp: string
  ip_address: string
  user_agent: string
  location?: string | null
  success: boolean
}

type SessionItem = {
  session_id: string
  ip_address: string
  user_agent: string
  created_at: string
  last_activity: string
  is_current: boolean
}

const AuthManagementPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [user, setUser] = useState<User | null>(null)
  const [permissionSummary, setPermissionSummary] = useState<{
    user_id: string
    username: string
    roles: string[]
    permissions: string[]
  } | null>(null)
  const [loginHistory, setLoginHistory] = useState<LoginHistoryItem[]>([])
  const [sessions, setSessions] = useState<SessionItem[]>([])

  const [registerOpen, setRegisterOpen] = useState(false)
  const [profileOpen, setProfileOpen] = useState(false)
  const [passwordOpen, setPasswordOpen] = useState(false)

  const load = async () => {
    if (!authService.getToken()) {
      setUser(null)
      setPermissionSummary(null)
      setLoginHistory([])
      setSessions([])
      return
    }

    setLoading(true)
    try {
      const [u, p, h, s] = await Promise.all([
        authService.getCurrentUser(),
        authService.getPermissionSummary(),
        authService.getLoginHistory(20),
        authService.getActiveSessions(),
      ])
      setUser(u)
      setPermissionSummary(p)
      setLoginHistory(h)
      setSessions(s)
    } catch (e) {
      await authService.logout()
      setUser(null)
      setPermissionSummary(null)
      setLoginHistory([])
      setSessions([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  const handleLogin = async (values: LoginRequest) => {
    setLoading(true)
    try {
      await authService.login(values)
      message.success('登录成功')
      await load()
    } catch (e) {
      message.error((e as Error).message || '登录失败')
    } finally {
      setLoading(false)
    }
  }

  const handleRegister = async (values: RegisterRequest) => {
    setLoading(true)
    try {
      await authService.register(values)
      message.success('注册成功')
      setRegisterOpen(false)
    } catch (e) {
      message.error((e as Error).message || '注册失败')
    } finally {
      setLoading(false)
    }
  }

  const handleLogout = async () => {
    setLoading(true)
    try {
      await authService.logout()
    } finally {
      setUser(null)
      setPermissionSummary(null)
      setLoginHistory([])
      setSessions([])
      setLoading(false)
    }
  }

  const handleRefresh = async () => {
    setLoading(true)
    try {
      await authService.refreshToken()
      message.success('Token已刷新')
      await load()
    } catch (e) {
      message.error((e as Error).message || 'Token刷新失败')
    } finally {
      setLoading(false)
    }
  }

  const handleVerify = async () => {
    setLoading(true)
    try {
      const result = await authService.verifyToken()
      Modal.info({
        title: 'Token验证结果',
        content: (
          <Space direction="vertical" size={4}>
            <Text>有效: {String(result.valid)}</Text>
            <Text>用户: {result.username}</Text>
            <Text>过期时间: {result.expires_at || '-'}</Text>
            <Text>角色: {result.roles.join(', ') || '-'}</Text>
            <Text>权限: {result.permissions.join(', ') || '-'}</Text>
          </Space>
        ),
      })
    } catch (e) {
      message.error((e as Error).message || 'Token验证失败')
    } finally {
      setLoading(false)
    }
  }

  const handleCheckPermission = async (values: {
    resource: string
    action: string
  }) => {
    setLoading(true)
    try {
      const ok = await authService.checkPermission(
        values.resource,
        values.action
      )
      message.info(ok ? '有权限' : '无权限')
    } catch (e) {
      message.error((e as Error).message || '权限检查失败')
    } finally {
      setLoading(false)
    }
  }

  const handleRevokeSession = async (sessionId: string) => {
    setLoading(true)
    try {
      await authService.revokeSession(sessionId)
      message.success('会话已撤销')
      await load()
    } catch (e) {
      message.error((e as Error).message || '撤销会话失败')
    } finally {
      setLoading(false)
    }
  }

  const handleRevokeAllSessions = async () => {
    setLoading(true)
    try {
      await authService.revokeAllSessions()
    } finally {
      await authService.logout()
      setUser(null)
      setPermissionSummary(null)
      setLoginHistory([])
      setSessions([])
      setLoading(false)
    }
  }

  const handleUpdateProfile = async (values: Partial<User>) => {
    setLoading(true)
    try {
      await authService.updateProfile(values)
      message.success('资料已更新')
      setProfileOpen(false)
      await load()
    } catch (e) {
      message.error((e as Error).message || '资料更新失败')
    } finally {
      setLoading(false)
    }
  }

  const handleChangePassword = async (values: {
    current_password: string
    new_password: string
  }) => {
    setLoading(true)
    try {
      const result = await authService.changePassword(values)
      message.success(result.message || '密码已修改')
      setPasswordOpen(false)
    } catch (e) {
      message.error((e as Error).message || '密码修改失败')
    } finally {
      setLoading(false)
    }
  }

  const sessionColumns = [
    {
      title: '会话ID',
      dataIndex: 'session_id',
      key: 'session_id',
      ellipsis: true,
    },
    { title: 'IP', dataIndex: 'ip_address', key: 'ip_address' },
    { title: 'UA', dataIndex: 'user_agent', key: 'user_agent', ellipsis: true },
    { title: '创建时间', dataIndex: 'created_at', key: 'created_at' },
    { title: '最后活动', dataIndex: 'last_activity', key: 'last_activity' },
    {
      title: '当前',
      dataIndex: 'is_current',
      key: 'is_current',
      render: (v: boolean) =>
        v ? <Tag color="blue">当前</Tag> : <Tag>其他</Tag>,
    },
    {
      title: '操作',
      key: 'actions',
      render: (_: unknown, record: SessionItem) => (
        <Button
          danger
          size="small"
          onClick={() => handleRevokeSession(record.session_id)}
        >
          撤销
        </Button>
      ),
    },
  ]

  const historyColumns = [
    { title: '时间', dataIndex: 'timestamp', key: 'timestamp' },
    { title: 'IP', dataIndex: 'ip_address', key: 'ip_address' },
    { title: 'UA', dataIndex: 'user_agent', key: 'user_agent', ellipsis: true },
    {
      title: '结果',
      dataIndex: 'success',
      key: 'success',
      render: (v: boolean) =>
        v ? <Tag color="green">成功</Tag> : <Tag color="red">失败</Tag>,
    },
  ]

  const authed = !!user

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}>认证授权</Title>
      <Paragraph type="secondary">
        覆盖 /api/v1/auth 下的认证、权限、会话与安全相关接口。
      </Paragraph>

      {!authed && (
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Alert
            message="未登录：请先登录后再查看会话与权限信息"
            type="warning"
            showIcon
          />
          <Card title="登录" size="small">
            <Form layout="inline" onFinish={handleLogin}>
              <Form.Item
                name="username"
                rules={[{ required: true, message: '请输入用户名' }]}
              >
                <Input placeholder="用户名" autoComplete="username" />
              </Form.Item>
              <Form.Item
                name="password"
                rules={[{ required: true, message: '请输入密码' }]}
              >
                <Input.Password
                  placeholder="密码"
                  autoComplete="current-password"
                />
              </Form.Item>
              <Form.Item>
                <Space>
                  <Button type="primary" htmlType="submit" loading={loading}>
                    登录
                  </Button>
                  <Button onClick={() => setRegisterOpen(true)}>注册</Button>
                </Space>
              </Form.Item>
            </Form>
          </Card>
        </Space>
      )}

      {authed && (
        <Space direction="vertical" style={{ width: '100%' }} size="middle">
          <Card
            title="当前用户"
            size="small"
            extra={
              <Space>
                <Button onClick={handleVerify} loading={loading}>
                  验证Token
                </Button>
                <Button onClick={handleRefresh} loading={loading}>
                  刷新Token
                </Button>
                <Button onClick={() => setProfileOpen(true)}>编辑资料</Button>
                <Button onClick={() => setPasswordOpen(true)}>修改密码</Button>
                <Button danger onClick={handleLogout}>
                  退出
                </Button>
              </Space>
            }
          >
            <Space direction="vertical" size={8} style={{ width: '100%' }}>
              <Text>
                用户名: <Text strong>{user.username}</Text>
              </Text>
              <Text>用户ID: {user.id}</Text>
              <Space wrap>
                <Text>角色:</Text>
                {(permissionSummary?.roles || user.roles || []).map(r => (
                  <Tag key={r}>{r}</Tag>
                ))}
              </Space>
              <Space wrap>
                <Text>权限:</Text>
                {(permissionSummary?.permissions || user.permissions || []).map(
                  p => (
                    <Tag key={p}>{p}</Tag>
                  )
                )}
              </Space>
              <Card title="权限检查" size="small" type="inner">
                <Form
                  layout="inline"
                  onFinish={handleCheckPermission}
                  initialValues={{ resource: 'users', action: 'read' }}
                >
                  <Form.Item
                    name="resource"
                    rules={[{ required: true, message: '请输入资源' }]}
                  >
                    <Input placeholder="resource" style={{ width: 180 }} />
                  </Form.Item>
                  <Form.Item
                    name="action"
                    rules={[{ required: true, message: '请输入动作' }]}
                  >
                    <Input placeholder="action" style={{ width: 180 }} />
                  </Form.Item>
                  <Form.Item>
                    <Button htmlType="submit" loading={loading}>
                      检查
                    </Button>
                  </Form.Item>
                </Form>
              </Card>
            </Space>
          </Card>

          <Card
            title="会话"
            size="small"
            extra={
              <Button
                danger
                onClick={handleRevokeAllSessions}
                loading={loading}
                disabled={sessions.length === 0}
              >
                撤销全部会话
              </Button>
            }
          >
            <Table
              columns={sessionColumns}
              dataSource={sessions}
              rowKey="session_id"
              size="small"
              pagination={{ pageSize: 8 }}
            />
          </Card>

          <Card title="登录历史" size="small">
            <Table
              columns={historyColumns}
              dataSource={loginHistory}
              rowKey={r => `${r.timestamp}-${r.ip_address}`}
              size="small"
              pagination={{ pageSize: 8 }}
            />
          </Card>
        </Space>
      )}

      <Modal
        title="注册"
        open={registerOpen}
        onCancel={() => setRegisterOpen(false)}
        footer={null}
        destroyOnClose
      >
        <Form layout="vertical" onFinish={handleRegister}>
          <Form.Item
            label="用户名"
            name="username"
            rules={[{ required: true, message: '请输入用户名' }]}
          >
            <Input autoComplete="username" />
          </Form.Item>
          <Form.Item
            label="邮箱"
            name="email"
            rules={[
              { required: true, type: 'email', message: '请输入有效邮箱' },
            ]}
          >
            <Input autoComplete="email" />
          </Form.Item>
          <Form.Item
            label="密码"
            name="password"
            rules={[{ required: true, message: '请输入密码' }]}
          >
            <Input.Password autoComplete="new-password" />
          </Form.Item>
          <Form.Item label="全名" name="full_name">
            <Input />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading}>
              注册
            </Button>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="编辑资料"
        open={profileOpen}
        onCancel={() => setProfileOpen(false)}
        footer={null}
        destroyOnClose
      >
        <Form
          layout="vertical"
          onFinish={handleUpdateProfile}
          initialValues={user || {}}
        >
          <Form.Item
            label="邮箱"
            name="email"
            rules={[{ type: 'email', message: '请输入有效邮箱' }]}
          >
            <Input autoComplete="email" />
          </Form.Item>
          <Form.Item label="全名" name="full_name">
            <Input autoComplete="name" />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading}>
              保存
            </Button>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="修改密码"
        open={passwordOpen}
        onCancel={() => setPasswordOpen(false)}
        footer={null}
        destroyOnClose
      >
        <Form layout="vertical" onFinish={handleChangePassword}>
          <Form.Item
            label="当前密码"
            name="current_password"
            rules={[{ required: true, message: '请输入当前密码' }]}
          >
            <Input.Password autoComplete="current-password" />
          </Form.Item>
          <Form.Item
            label="新密码"
            name="new_password"
            rules={[{ required: true, message: '请输入新密码' }]}
          >
            <Input.Password autoComplete="new-password" />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading}>
              修改
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default AuthManagementPage
