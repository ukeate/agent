import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Alert,
  Button,
  Space,
  Tabs,
  Tag,
  Timeline,
  Descriptions,
  Table,
  Badge,
  message,
  Modal,
  Form,
  Input,
} from 'antd'
import {
  SafetyOutlined,
  ExclamationCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  WarningOutlined,
  ReloadOutlined,
  SettingOutlined,
  MonitorOutlined,
  DatabaseOutlined,
  ClusterOutlined,
  HeartOutlined,
} from '@ant-design/icons'
import { faultToleranceService } from '../services/faultToleranceService'

interface SystemStatus {
  system_started: boolean
  health_summary: {
    total_components: number
    health_ratio: number
    active_faults: number
    status_counts: {
      healthy: number
      degraded: number
      unhealthy: number
    }
  }
  recovery_statistics: {
    success_rate: number
    recent_recoveries: any[]
  }
  consistency_statistics: {
    consistency_rate: number
  }
  backup_statistics: {
    total_backups: number
    components: Record<string, any>
  }
  active_faults: any[]
  last_updated: string
}

interface SystemMetrics {
  fault_detection_metrics: {
    total_components: number
    healthy_components: number
  }
  recovery_metrics: {
    success_rate: number
  }
  backup_metrics: {
    total_backups: number
  }
  consistency_metrics: {
    consistency_rate: number
  }
  system_availability: number
  last_updated: string
}

interface ComponentHealth {
  component_id: string
  status: 'healthy' | 'degraded' | 'unhealthy' | 'unknown'
  last_check: string
  response_time: number
  error_rate: number
  resource_usage: {
    cpu: number
    memory: number
  }
}

const FaultToleranceSystemPage: React.FC = () => {
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null)
  const [systemMetrics, setSystemMetrics] = useState<SystemMetrics | null>(null)
  const [components, setComponents] = useState<ComponentHealth[]>([])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')
  const [repairModalVisible, setRepairModalVisible] = useState(false)
  const [repairForm] = Form.useForm()

  const fetchSystemStatus = async () => {
    try {
      const data = await faultToleranceService.getSystemStatus()
      setSystemStatus(data)
    } catch (error) {
      logger.error('获取系统状态失败:', error)
    }
  }

  const fetchSystemMetrics = async () => {
    try {
      const data = await faultToleranceService.getMetrics()
      setSystemMetrics(data)
    } catch (error) {
      logger.error('获取系统指标失败:', error)
    }
  }

  const fetchComponentsList = async () => {
    try {
      const healthData = await faultToleranceService.getHealth()

      // 转换健康数据为组件列表格式
      const componentsList: ComponentHealth[] = healthData.services
        ? healthData.services.map((service: any) => ({
            component_id: service.service_id || service.component_id,
            status: service.status,
            last_check: service.last_check,
            response_time: service.response_time_ms
              ? service.response_time_ms / 1000
              : service.response_time || 0,
            error_rate: service.error_rate || 0,
            resource_usage: {
              cpu: service.resource_usage?.cpu || 0,
              memory: service.resource_usage?.memory || 0,
            },
          }))
        : []

      setComponents(componentsList)
    } catch (error) {
      logger.error('获取组件列表失败:', error)
    }
  }

  const startSystem = async () => {
    setLoading(true)
    try {
      const result = await faultToleranceService.startSystem()
      await fetchSystemStatus()
      message.success(
        result?.message ? `系统已启动: ${result.message}` : '系统已启动'
      )
    } catch (error) {
      logger.error('启动系统失败:', error)
      message.error('启动系统失败: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  const stopSystem = async () => {
    setLoading(true)
    try {
      const result = await faultToleranceService.stopSystem()
      await fetchSystemStatus()
      message.success(
        result?.message ? `系统已停止: ${result.message}` : '系统已停止'
      )
    } catch (error) {
      logger.error('停止系统失败:', error)
      message.error('停止系统失败: ' + error.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    const loadData = async () => {
      setLoading(true)
      await Promise.all([
        fetchSystemStatus(),
        fetchSystemMetrics(),
        fetchComponentsList(),
      ])
      setLoading(false)
    }

    loadData()
    const interval = setInterval(loadData, 10000) // 每10秒刷新
    return () => clearInterval(interval)
  }, [])

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy':
        return 'success'
      case 'degraded':
        return 'warning'
      case 'unhealthy':
        return 'error'
      default:
        return 'default'
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case 'degraded':
        return <WarningOutlined style={{ color: '#faad14' }} />
      case 'unhealthy':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
      default:
        return <ExclamationCircleOutlined style={{ color: '#d9d9d9' }} />
    }
  }

  const componentColumns = [
    {
      title: '组件ID',
      dataIndex: 'component_id',
      key: 'component_id',
      render: (id: string) => <Badge status="processing" text={id} />,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)} icon={getStatusIcon(status)}>
          {status.toUpperCase()}
        </Tag>
      ),
    },
    {
      title: '响应时间',
      dataIndex: 'response_time',
      key: 'response_time',
      render: (time: number) => `${time.toFixed(2)}s`,
    },
    {
      title: '错误率',
      dataIndex: 'error_rate',
      key: 'error_rate',
      render: (rate: number) => `${(rate * 100).toFixed(2)}%`,
    },
    {
      title: 'CPU使用率',
      dataIndex: ['resource_usage', 'cpu'],
      key: 'cpu',
      render: (cpu: number) => (
        <Progress
          percent={cpu}
          size="small"
          status={cpu > 80 ? 'exception' : cpu > 60 ? 'active' : 'success'}
        />
      ),
    },
    {
      title: '内存使用率',
      dataIndex: ['resource_usage', 'memory'],
      key: 'memory',
      render: (memory: number) => (
        <Progress
          percent={memory}
          size="small"
          status={
            memory > 80 ? 'exception' : memory > 60 ? 'active' : 'success'
          }
        />
      ),
    },
  ]

  const overviewTab = (
    <div>
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={24}>
          <Alert
            message="故障容错和恢复系统"
            description="实时监控分布式智能体系统的健康状态，提供自动故障检测、任务重分配和数据一致性保障"
            type="info"
            showIcon
            action={
              <Space>
                {systemStatus?.system_started ? (
                  <Button size="small" danger onClick={stopSystem}>
                    停止系统
                  </Button>
                ) : (
                  <Button size="small" type="primary" onClick={startSystem}>
                    启动系统
                  </Button>
                )}
                <Button
                  size="small"
                  icon={<ReloadOutlined />}
                  onClick={() => {
                    fetchSystemStatus()
                    fetchSystemMetrics()
                    fetchComponentsList()
                  }}
                  loading={loading}
                >
                  刷新
                </Button>
              </Space>
            }
          />
        </Col>
      </Row>

      {/* 系统状态指标 */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={6}>
          <Card>
            <Statistic
              title="系统可用性"
              value={(systemMetrics?.system_availability || 0) * 100}
              suffix="%"
              precision={2}
              valueStyle={{
                color:
                  (systemMetrics?.system_availability || 0) * 100 > 99
                    ? '#3f8600'
                    : '#cf1322',
              }}
              prefix={<SafetyOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="健康组件比率"
              value={
                systemStatus
                  ? systemStatus.health_summary.health_ratio * 100
                  : 0
              }
              suffix="%"
              precision={1}
              valueStyle={{
                color:
                  systemStatus && systemStatus.health_summary.health_ratio > 0.9
                    ? '#3f8600'
                    : '#cf1322',
              }}
              prefix={<HeartOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="恢复成功率"
              value={
                systemStatus
                  ? systemStatus.recovery_statistics.success_rate * 100
                  : 0
              }
              suffix="%"
              precision={1}
              valueStyle={{
                color:
                  systemStatus &&
                  systemStatus.recovery_statistics.success_rate > 0.95
                    ? '#3f8600'
                    : '#cf1322',
              }}
              prefix={<SyncOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃故障数"
              value={systemStatus?.active_faults.length || 0}
              valueStyle={{
                color:
                  (systemStatus?.active_faults.length || 0) === 0
                    ? '#3f8600'
                    : '#cf1322',
              }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 组件状态分布 */}
      <Row gutter={[16, 16]} className="mb-6">
        <Col span={8}>
          <Card title="组件状态分布" size="small">
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="flex items-center">
                  <CheckCircleOutlined
                    style={{ color: '#52c41a', marginRight: 8 }}
                  />
                  健康组件
                </span>
                <Badge
                  count={
                    systemStatus?.health_summary.status_counts.healthy || 0
                  }
                  style={{ backgroundColor: '#52c41a' }}
                />
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center">
                  <WarningOutlined
                    style={{ color: '#faad14', marginRight: 8 }}
                  />
                  降级组件
                </span>
                <Badge
                  count={
                    systemStatus?.health_summary.status_counts.degraded || 0
                  }
                  style={{ backgroundColor: '#faad14' }}
                />
              </div>
              <div className="flex justify-between items-center">
                <span className="flex items-center">
                  <CloseCircleOutlined
                    style={{ color: '#ff4d4f', marginRight: 8 }}
                  />
                  异常组件
                </span>
                <Badge
                  count={
                    systemStatus?.health_summary.status_counts.unhealthy || 0
                  }
                  style={{ backgroundColor: '#ff4d4f' }}
                />
              </div>
            </div>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="数据一致性" size="small">
            <div className="text-center">
              <Progress
                type="circle"
                percent={
                  systemStatus
                    ? systemStatus.consistency_statistics.consistency_rate * 100
                    : 0
                }
                format={percent => `${percent?.toFixed(1)}%`}
                status={
                  systemStatus &&
                  systemStatus.consistency_statistics.consistency_rate > 0.98
                    ? 'success'
                    : 'active'
                }
              />
              <div className="mt-2 text-gray-600">数据一致性率</div>
            </div>
          </Card>
        </Col>
        <Col span={8}>
          <Card title="备份统计" size="small">
            <Descriptions size="small" column={1}>
              <Descriptions.Item label="备份总数">
                {systemStatus?.backup_statistics.total_backups || 0}
              </Descriptions.Item>
              <Descriptions.Item label="备份组件">
                {
                  Object.keys(systemStatus?.backup_statistics.components || {})
                    .length
                }
              </Descriptions.Item>
              <Descriptions.Item label="最后更新">
                {systemStatus
                  ? new Date(systemStatus.last_updated).toLocaleString()
                  : '-'}
              </Descriptions.Item>
            </Descriptions>
          </Card>
        </Col>
      </Row>

      {/* 组件详情表格 */}
      <Card title="组件状态详情">
        <Table
          columns={componentColumns}
          dataSource={components}
          rowKey="component_id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: total => `共 ${total} 个组件`,
          }}
        />
      </Card>
    </div>
  )

  const faultHistoryTab = (
    <Card title="故障历史记录">
      <Timeline>
        {systemStatus?.active_faults.map((fault, index) => (
          <Timeline.Item
            key={index}
            color={
              fault.severity === 'high'
                ? 'red'
                : fault.severity === 'medium'
                  ? 'orange'
                  : 'blue'
            }
            dot={<ExclamationCircleOutlined />}
          >
            <div>
              <strong>{fault.fault_type}</strong> - {fault.description}
              <br />
              <span className="text-gray-500">
                影响组件: {fault.affected_components.join(', ')} | 严重程度:{' '}
                {fault.severity} | 检测时间:{' '}
                {new Date(fault.detected_at).toLocaleString()}
              </span>
            </div>
          </Timeline.Item>
        ))}
        {(!systemStatus?.active_faults ||
          systemStatus.active_faults.length === 0) && (
          <Timeline.Item color="green" dot={<CheckCircleOutlined />}>
            <div>
              <strong>系统正常</strong>
              <br />
              <span className="text-gray-500">当前没有活跃的故障事件</span>
            </div>
          </Timeline.Item>
        )}
      </Timeline>
    </Card>
  )

  // 新增：备份管理Tab - 使用未使用的API
  const backupManagementTab = (
    <div>
      <Row gutter={16} className="mb-4">
        <Col span={6}>
          <Card title="备份操作" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                type="primary"
                icon={<DatabaseOutlined />}
                onClick={async () => {
                  try {
                    setLoading(true)
                    if (!components.length) {
                      message.error('暂无可备份组件')
                      return
                    }
                    const componentIds = components.map(
                      component => component.component_id
                    )
                    const result =
                      await faultToleranceService.triggerManualBackup(
                        componentIds
                      )
                    message.success(
                      `备份完成，成功 ${result.success_count}/${result.total_count}`
                    )
                  } catch (error) {
                    message.error('启动备份失败: ' + error.message)
                  } finally {
                    setLoading(false)
                  }
                }}
                loading={loading}
                block
              >
                手动备份
              </Button>
              <Button
                icon={<SyncOutlined />}
                onClick={async () => {
                  try {
                    setLoading(true)
                    const stats =
                      await faultToleranceService.getBackupStatistics()
                    message.success(
                      `备份统计已刷新，共${stats.total_backups}个备份`
                    )
                  } catch (error) {
                    message.error('获取备份统计失败: ' + error.message)
                  } finally {
                    setLoading(false)
                  }
                }}
                loading={loading}
                block
              >
                刷新统计
              </Button>
            </Space>
          </Card>
        </Col>
        <Col span={6}>
          <Card title="一致性检查" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                type="primary"
                icon={<CheckCircleOutlined />}
                onClick={async () => {
                  try {
                    setLoading(true)
                    const result =
                      await faultToleranceService.checkDataConsistency()
                    message.success(`一致性检查已启动，ID: ${result.check_id}`)
                  } catch (error) {
                    message.error('启动一致性检查失败: ' + error.message)
                  } finally {
                    setLoading(false)
                  }
                }}
                loading={loading}
                block
              >
                数据一致性检查
              </Button>
              <Button
                icon={<WarningOutlined />}
                onClick={() => setRepairModalVisible(true)}
                loading={loading}
                block
              >
                强制修复
              </Button>
            </Space>
          </Card>
        </Col>
        <Col span={6}>
          <Card title="系统控制" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                type="primary"
                icon={<SyncOutlined />}
                onClick={startSystem}
                loading={loading}
                block
              >
                启动系统
              </Button>
              <Button
                danger
                icon={<CloseCircleOutlined />}
                onClick={stopSystem}
                loading={loading}
                block
              >
                停止系统
              </Button>
            </Space>
          </Card>
        </Col>
        <Col span={6}>
          <Card title="故障测试" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button
                type="dashed"
                icon={<ExclamationCircleOutlined />}
                onClick={async () => {
                  try {
                    setLoading(true)
                    if (!components.length) {
                      message.error('暂无可注入故障的组件')
                      return
                    }
                    const faultTypes =
                      await faultToleranceService.getFaultTypes()
                    if (!faultTypes.length) {
                      message.error('未获取到可用故障类型')
                      return
                    }
                    const target = components[0].component_id
                    const result = await faultToleranceService.injectFault(
                      faultTypes[0],
                      target,
                      5000
                    )
                    message.success(`故障注入已启动，ID: ${result.fault_id}`)
                  } catch (error) {
                    message.error('故障注入失败: ' + error.message)
                  } finally {
                    setLoading(false)
                  }
                }}
                loading={loading}
                block
              >
                注入测试故障
              </Button>
              <Button
                icon={<ReloadOutlined />}
                onClick={async () => {
                  try {
                    setLoading(true)
                    const faultTypes =
                      await faultToleranceService.getFaultTypes()
                    message.success(`支持的故障类型: ${faultTypes.join(', ')}`)
                  } catch (error) {
                    message.error('获取故障类型失败: ' + error.message)
                  } finally {
                    setLoading(false)
                  }
                }}
                loading={loading}
                block
              >
                查看故障类型
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>

      <Card title="系统报告">
        <Button
          type="primary"
          icon={<DatabaseOutlined />}
          onClick={async () => {
            try {
              setLoading(true)
              const report =
                await faultToleranceService.getFaultToleranceReport()
              message.success('系统报告获取成功')
              logger.log('容错系统报告:', report)
            } catch (error) {
              message.error('获取系统报告失败: ' + error.message)
            } finally {
              setLoading(false)
            }
          }}
          loading={loading}
        >
          生成容错系统报告
        </Button>
      </Card>
    </div>
  )

  const tabItems = [
    {
      key: 'overview',
      label: (
        <span>
          <MonitorOutlined />
          系统总览
        </span>
      ),
      children: overviewTab,
    },
    {
      key: 'faults',
      label: (
        <span>
          <ExclamationCircleOutlined />
          故障历史
        </span>
      ),
      children: faultHistoryTab,
    },
    {
      key: 'backup',
      label: (
        <span>
          <DatabaseOutlined />
          备份管理
        </span>
      ),
      children: backupManagementTab,
    },
  ]

  return (
    <div className="fault-tolerance-system-page p-6">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold mb-2">故障容错和恢复系统</h1>
          <p className="text-gray-600">
            智能体故障检测、任务重分配、数据备份和一致性保障
          </p>
        </div>
        <Space>
          <Button icon={<SettingOutlined />} href="/fault-tolerance/settings">
            系统设置
          </Button>
          <Button
            type="primary"
            icon={<DatabaseOutlined />}
            href="/fault-tolerance/backup"
          >
            备份管理
          </Button>
        </Space>
      </div>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={tabItems}
        size="large"
      />
      <Modal
        title="强制一致性修复"
        open={repairModalVisible}
        onCancel={() => {
          setRepairModalVisible(false)
          repairForm.resetFields()
        }}
        onOk={async () => {
          try {
            const values = await repairForm.validateFields()
            setLoading(true)
            const result = await faultToleranceService.forceRepairConsistency(
              values.dataKey,
              values.authoritativeComponentId
            )
            message.success(`强制修复完成: ${result.data_key}`)
            setRepairModalVisible(false)
            repairForm.resetFields()
          } catch (error) {
            message.error('强制修复失败: ' + error.message)
          } finally {
            setLoading(false)
          }
        }}
        confirmLoading={loading}
        destroyOnClose
      >
        <Form form={repairForm} layout="vertical">
          <Form.Item
            name="dataKey"
            label="数据键"
            rules={[{ required: true, message: '请输入数据键' }]}
          >
            <Input placeholder="例如: user_profile:123" />
          </Form.Item>
          <Form.Item
            name="authoritativeComponentId"
            label="权威组件ID"
            rules={[{ required: true, message: '请输入权威组件ID' }]}
          >
            <Input placeholder="例如: agent-1" />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default FaultToleranceSystemPage
