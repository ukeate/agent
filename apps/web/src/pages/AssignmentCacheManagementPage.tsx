import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Button,
  Table,
  Form,
  Input,
  Select,
  Tabs,
  Statistic,
  Progress,
  Alert,
  Space,
  Typography,
  Row,
  Col,
  Modal,
  Divider,
  Badge,
  notification,
  Tooltip,
  Switch,
} from 'antd'
import {
  Database,
  Users,
  BarChart,
  Settings,
  RefreshCw,
  Plus,
  Download,
  Trash2,
  Clock,
  CheckCircle,
  AlertTriangle,
  TrendingUp,
} from 'lucide-react'
import assignmentCacheService, {
  CachedAssignment,
  CreateAssignmentRequest,
  CacheMetrics,
  HealthCheckResponse,
} from '../services/assignmentCacheService'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Option } = Select

interface AssignmentTableData {
  key: string
  user_id: string
  experiment_id: string
  variant_id: string
  assigned_at: string
  cache_status: string
  assignment_context: Record<string, any>
}

const AssignmentCacheManagementPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [metrics, setMetrics] = useState<CacheMetrics | null>(null)
  const [healthStatus, setHealthStatus] = useState<HealthCheckResponse | null>(
    null
  )
  const [assignments, setAssignments] = useState<AssignmentTableData[]>([])
  const [selectedUser, setSelectedUser] = useState<string>('')
  const [selectedExperiment, setSelectedExperiment] = useState<string>('')
  const [createModalVisible, setCreateModalVisible] = useState(false)
  const [batchModalVisible, setBatchModalVisible] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [form] = Form.useForm()
  const [batchForm] = Form.useForm()

  // 加载数据
  const loadData = async () => {
    setLoading(true)
    try {
      const [metricsData, healthData] = await Promise.all([
        assignmentCacheService.getCacheMetrics(),
        assignmentCacheService.healthCheck(),
      ])
      setMetrics(metricsData)
      setHealthStatus(healthData)
    } catch (error) {
      logger.error('加载数据失败:', error)
      notification.error({
        message: '数据加载失败',
        description: '无法获取缓存数据，请检查服务连接',
      })
    } finally {
      setLoading(false)
    }
  }

  // 查询用户分配
  const searchUserAssignment = async () => {
    if (!selectedUser || !selectedExperiment) {
      notification.warning({
        message: '参数不完整',
        description: '请输入用户ID和实验ID',
      })
      return
    }

    try {
      setLoading(true)
      const result = await assignmentCacheService.getUserAssignment(
        selectedUser,
        selectedExperiment
      )

      if (result.variant_id) {
        const assignment: AssignmentTableData = {
          key: `${result.user_id}-${result.experiment_id}`,
          user_id: result.user_id,
          experiment_id: result.experiment_id,
          variant_id: result.variant_id,
          assigned_at: result.assigned_at || new Date().toISOString(),
          cache_status: result.cache_status,
          assignment_context: result.assignment_context || {},
        }
        setAssignments([assignment])

        notification.success({
          message: '查询成功',
          description: `找到用户 ${selectedUser} 在实验 ${selectedExperiment} 中的分配`,
        })
      } else {
        setAssignments([])
        notification.info({
          message: '未找到分配',
          description: result.message || '该用户在指定实验中没有分配记录',
        })
      }
    } catch (error) {
      logger.error('查询失败:', error)
      notification.error({
        message: '查询失败',
        description: '无法查询用户分配，请检查输入参数',
      })
    } finally {
      setLoading(false)
    }
  }

  // 加载用户所有分配
  const loadUserAllAssignments = async (userId: string) => {
    try {
      setLoading(true)
      const result = await assignmentCacheService.getUserAllAssignments(userId)

      const assignmentData: AssignmentTableData[] = await Promise.all(
        result.assignments.map(async (assignment, index) => {
          const detail = await assignmentCacheService.getUserAssignment(
            userId,
            assignment.experiment_id
          )
          return {
            key: `${userId}-${assignment.experiment_id}-${index}`,
            user_id: userId,
            experiment_id: assignment.experiment_id,
            variant_id: assignment.variant_id,
            assigned_at: assignment.assigned_at,
            cache_status: detail.cache_status,
            assignment_context: assignment.assignment_context,
          }
        })
      )

      setAssignments(assignmentData)
      notification.success({
        message: '加载成功',
        description: `用户 ${userId} 共有 ${result.total_assignments} 个分配记录`,
      })
    } catch (error) {
      logger.error('加载用户分配失败:', error)
      notification.error({
        message: '加载失败',
        description: '无法加载用户的所有分配记录',
      })
    } finally {
      setLoading(false)
    }
  }

  // 创建分配
  const createAssignment = async (values: any) => {
    try {
      setLoading(true)
      const request: CreateAssignmentRequest = {
        user_id: values.user_id,
        experiment_id: values.experiment_id,
        variant_id: values.variant_id,
        assignment_context: values.assignment_context
          ? JSON.parse(values.assignment_context)
          : {},
        ttl: values.ttl ? parseInt(values.ttl) : undefined,
      }

      await assignmentCacheService.createAssignment(request)

      notification.success({
        message: '创建成功',
        description: `成功为用户 ${request.user_id} 创建分配`,
      })

      setCreateModalVisible(false)
      form.resetFields()
      await loadData()
    } catch (error) {
      logger.error('创建分配失败:', error)
      notification.error({
        message: '创建失败',
        description: '无法创建用户分配，请检查输入数据',
      })
    } finally {
      setLoading(false)
    }
  }

  // 清空所有缓存
  const clearAllCache = () => {
    Modal.confirm({
      title: '确认清空缓存',
      content: '这将清空所有用户分配缓存，此操作不可逆！',
      okText: '确认清空',
      cancelText: '取消',
      okType: 'danger',
      onOk: async () => {
        try {
          await assignmentCacheService.clearAllCache()
          notification.success({
            message: '缓存清空成功',
            description: '所有用户分配缓存已清空',
          })
          await loadData()
          setAssignments([])
        } catch (error) {
          logger.error('清空缓存失败:', error)
          notification.error({
            message: '清空失败',
            description: '无法清空缓存，请稍后重试',
          })
        }
      },
    })
  }

  // 预热缓存
  const warmupCache = async () => {
    try {
      const userIds = selectedUser
        .split(',')
        .map(s => s.trim())
        .filter(Boolean)
      if (userIds.length === 0) {
        notification.warning({
          message: '请输入用户ID',
          description: '支持用逗号分隔多个用户ID',
        })
        return
      }
      await assignmentCacheService.warmupCache(userIds)

      notification.success({
        message: '缓存预热已启动',
        description: `正在为 ${userIds.length} 个用户预热缓存`,
      })
    } catch (error) {
      logger.error('缓存预热失败:', error)
      notification.error({
        message: '预热失败',
        description: '无法启动缓存预热',
      })
    }
  }

  // 自动刷新
  useEffect(() => {
    let interval: ReturnType<typeof setTimeout>
    if (autoRefresh) {
      interval = setInterval(() => {
        loadData()
      }, 10000) // 10秒刷新一次
    }
    return () => {
      if (interval) clearInterval(interval)
    }
  }, [autoRefresh])

  useEffect(() => {
    loadData()
  }, [])

  // 表格列定义
  const columns = [
    {
      title: '用户ID',
      dataIndex: 'user_id',
      key: 'user_id',
      render: (text: string) => <Text code>{text}</Text>,
    },
    {
      title: '实验ID',
      dataIndex: 'experiment_id',
      key: 'experiment_id',
      render: (text: string) => <Text code>{text}</Text>,
    },
    {
      title: '变体ID',
      dataIndex: 'variant_id',
      key: 'variant_id',
      render: (text: string) => <Badge color="blue" text={text} />,
    },
    {
      title: '分配时间',
      dataIndex: 'assigned_at',
      key: 'assigned_at',
      render: (text: string) => (
        <Tooltip title={text}>
          <Text>{new Date(text).toLocaleString()}</Text>
        </Tooltip>
      ),
    },
    {
      title: '缓存状态',
      dataIndex: 'cache_status',
      key: 'cache_status',
      render: (status: string) => (
        <Badge
          color={
            status === 'hit' ? 'green' : status === 'miss' ? 'orange' : 'red'
          }
          text={status.toUpperCase()}
        />
      ),
    },
    {
      title: '上下文',
      dataIndex: 'assignment_context',
      key: 'assignment_context',
      render: (context: Record<string, any>) => (
        <Text code>{JSON.stringify(context)}</Text>
      ),
    },
  ]

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* 页面标题 */}
        <div style={{ marginBottom: '24px' }}>
          <Title level={2}>
            <Database style={{ marginRight: '8px', color: '#1890ff' }} />
            用户分配缓存管理
          </Title>
          <Paragraph>
            管理用户实验分配的缓存系统，提供高性能的分配查询和管理功能。
          </Paragraph>
        </div>

        {/* 控制面板 */}
        <Card style={{ marginBottom: '24px' }}>
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              marginBottom: '16px',
            }}
          >
            <Title level={4} style={{ margin: 0 }}>
              操作面板
            </Title>
            <Space>
              <Text>自动刷新:</Text>
              <Switch
                checked={autoRefresh}
                onChange={setAutoRefresh}
                checkedChildren="开启"
                unCheckedChildren="关闭"
              />
            </Space>
          </div>

          <Space size="middle" wrap>
            <Button
              type="primary"
              icon={<RefreshCw size={16} />}
              loading={loading}
              onClick={loadData}
            >
              刷新数据
            </Button>

            <Button
              icon={<Plus size={16} />}
              onClick={() => setCreateModalVisible(true)}
            >
              创建分配
            </Button>

            <Button icon={<TrendingUp size={16} />} onClick={warmupCache}>
              预热缓存
            </Button>

            <Button danger icon={<Trash2 size={16} />} onClick={clearAllCache}>
              清空缓存
            </Button>
          </Space>
        </Card>

        {/* 统计概览 */}
        {(metrics || healthStatus) && (
          <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="总分配数"
                  value={metrics?.total_assignments || 0}
                  prefix={<Database size={20} />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="总用户数"
                  value={metrics?.total_users || 0}
                  prefix={<Users size={20} />}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="缓存命中率"
                  value={metrics?.cache_hit_rate || 0}
                  precision={1}
                  suffix="%"
                  valueStyle={{
                    color:
                      (metrics?.cache_hit_rate || 0) > 80
                        ? '#3f8600'
                        : '#faad14',
                  }}
                />
              </Card>
            </Col>
            <Col xs={24} sm={12} md={6}>
              <Card>
                <Statistic
                  title="内存使用"
                  value={metrics?.memory_usage_mb || 0}
                  precision={1}
                  suffix="MB"
                  prefix={<BarChart size={20} />}
                />
              </Card>
            </Col>
          </Row>
        )}

        {/* 主要内容 */}
        <Tabs defaultActiveKey="query">
          {/* 查询分配 */}
          <TabPane tab="查询分配" key="query">
            <Card>
              <div style={{ marginBottom: '16px' }}>
                <Title level={4}>查询用户分配</Title>
                <Space size="middle" wrap>
                  <Input
                    placeholder="用户ID"
                    value={selectedUser}
                    onChange={e => setSelectedUser(e.target.value)}
                    name="assignment-filter-user"
                    style={{ width: 200 }}
                  />
                  <Input
                    placeholder="实验ID"
                    value={selectedExperiment}
                    onChange={e => setSelectedExperiment(e.target.value)}
                    name="assignment-filter-experiment"
                    style={{ width: 200 }}
                  />
                  <Button
                    type="primary"
                    onClick={searchUserAssignment}
                    loading={loading}
                  >
                    查询分配
                  </Button>
                  <Button
                    onClick={() =>
                      selectedUser && loadUserAllAssignments(selectedUser)
                    }
                    disabled={!selectedUser}
                  >
                    查询用户所有分配
                  </Button>
                </Space>
              </div>

              <Table
                columns={columns}
                dataSource={assignments}
                loading={loading}
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showQuickJumper: true,
                  showTotal: total => `共 ${total} 条记录`,
                }}
                scroll={{ x: 1000 }}
              />
            </Card>
          </TabPane>

          {/* 缓存监控 */}
          <TabPane tab="缓存监控" key="monitor">
            <Row gutter={[16, 16]}>
              <Col xs={24} lg={12}>
                <Card
                  title="健康状态"
                  extra={
                    <Badge
                      color={
                        healthStatus?.status === 'healthy' ? 'green' : 'red'
                      }
                      text={healthStatus?.status?.toUpperCase() || 'UNKNOWN'}
                    />
                  }
                >
                  {healthStatus && (
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                        }}
                      >
                        <Text>Redis连接:</Text>
                        <Badge
                          color={
                            healthStatus.redis_connection ? 'green' : 'red'
                          }
                          text={healthStatus.redis_connection ? '正常' : '异常'}
                        />
                      </div>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                        }}
                      >
                        <Text>缓存大小:</Text>
                        <Text>{healthStatus.cache_size}</Text>
                      </div>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                        }}
                      >
                        <Text>活跃键数:</Text>
                        <Text>{healthStatus.active_keys}</Text>
                      </div>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                        }}
                      >
                        <Text>内存使用:</Text>
                        <Text>{healthStatus.memory_usage}</Text>
                      </div>
                    </Space>
                  )}
                </Card>
              </Col>

              <Col xs={24} lg={12}>
                <Card title="缓存指标">
                  {metrics && (
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div>
                        <Text>命中率</Text>
                        <Progress
                          percent={Math.round(metrics.cache_hit_rate)}
                          strokeColor={{
                            '0%': '#ff4d4f',
                            '50%': '#faad14',
                            '100%': '#52c41a',
                          }}
                        />
                      </div>
                      <div>
                        <Text>失效率</Text>
                        <Progress
                          percent={Math.round(metrics.cache_miss_rate)}
                          strokeColor="#ff4d4f"
                        />
                      </div>
                      <Divider />
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                        }}
                      >
                        <Text>过期分配:</Text>
                        <Text strong>{metrics.expired_assignments}</Text>
                      </div>
                      <div
                        style={{
                          display: 'flex',
                          justifyContent: 'space-between',
                        }}
                      >
                        <Text>最近分配:</Text>
                        <Text strong>{metrics.recent_assignments}</Text>
                      </div>
                    </Space>
                  )}
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>

        {/* 创建分配模态框 */}
        <Modal
          title="创建用户分配"
          open={createModalVisible}
          onCancel={() => setCreateModalVisible(false)}
          footer={null}
        >
          <Form form={form} layout="vertical" onFinish={createAssignment}>
            <Form.Item
              label="用户ID"
              name="user_id"
              rules={[{ required: true, message: '请输入用户ID' }]}
            >
              <Input placeholder="例如: user_001" name="assignment-user-id" />
            </Form.Item>

            <Form.Item
              label="实验ID"
              name="experiment_id"
              rules={[{ required: true, message: '请输入实验ID' }]}
            >
              <Input
                placeholder="例如: exp_button_color"
                name="assignment-experiment-id"
              />
            </Form.Item>

            <Form.Item
              label="变体ID"
              name="variant_id"
              rules={[{ required: true, message: '请输入变体ID' }]}
            >
              <Input
                placeholder="例如: blue_button"
                name="assignment-variant-id"
              />
            </Form.Item>

            <Form.Item label="分配上下文 (JSON格式)" name="assignment_context">
              <Input.TextArea
                placeholder='{"device": "mobile", "source": "web"}'
                name="assignment-context"
                rows={3}
              />
            </Form.Item>

            <Form.Item label="TTL (秒)" name="ttl">
              <Input
                type="number"
                placeholder="3600"
                addonAfter="秒"
                name="assignment-ttl"
              />
            </Form.Item>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit" loading={loading}>
                  创建分配
                </Button>
                <Button onClick={() => setCreateModalVisible(false)}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Modal>
      </div>
    </div>
  )
}

export default AssignmentCacheManagementPage
