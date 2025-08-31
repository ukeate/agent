import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Button,
  Modal,
  Form,
  Input,
  Select,
  Tag,
  Space,
  Popconfirm,
  message,
  Drawer,
  Descriptions,
  Badge,
  Tooltip,
  Row,
  Col,
  Statistic,
  Timeline,
  Typography,
  Alert
} from 'antd'
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined,
  ApiOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ClockCircleOutlined,
  SettingOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text } = Typography
const { Option } = Select
const { TextArea } = Input

interface Component {
  component_id: string
  name: string
  component_type: 'ai_service' | 'data_processor' | 'model_service' | 'monitoring_tool' | 'integration_service'
  version: string
  status: 'healthy' | 'unhealthy'
  health_endpoint: string
  api_endpoint: string
  metadata: Record<string, any>
  last_check: string
  uptime: number
  created_at: string
  updated_at: string
}

interface ComponentHealth {
  status: 'healthy' | 'unhealthy'
  response_time: number
  error_message?: string
  checks: Array<{
    name: string
    status: 'pass' | 'fail'
    message?: string
  }>
}

const ComponentManagementPage: React.FC = () => {
  const [components, setComponents] = useState<Component[]>([])
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false)
  const [selectedComponent, setSelectedComponent] = useState<Component | null>(null)
  const [componentHealth, setComponentHealth] = useState<ComponentHealth | null>(null)
  const [form] = Form.useForm()

  useEffect(() => {
    fetchComponents()
  }, [])

  const fetchComponents = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/v1/platform-integration/components')
      if (response.ok) {
        const data = await response.json()
        setComponents(data.components || [])
      } else {
        message.error('获取组件列表失败')
      }
    } catch (error) {
      message.error('获取组件列表失败')
    } finally {
      setLoading(false)
    }
  }

  const fetchComponentHealth = async (componentId: string) => {
    try {
      const response = await fetch(`/api/v1/platform-integration/components/${componentId}/health`)
      if (response.ok) {
        const data = await response.json()
        setComponentHealth(data)
      }
    } catch (error) {
      console.error('获取组件健康状态失败:', error)
    }
  }

  const handleRegisterComponent = async (values: any) => {
    try {
      const response = await fetch('/api/v1/platform-integration/components/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ...values,
          metadata: values.metadata ? JSON.parse(values.metadata) : {}
        })
      })

      if (response.ok) {
        message.success('组件注册成功')
        setModalVisible(false)
        form.resetFields()
        fetchComponents()
      } else {
        const error = await response.json()
        message.error(`注册失败: ${error.detail}`)
      }
    } catch (error) {
      message.error('注册失败')
    }
  }

  const handleUnregisterComponent = async (componentId: string) => {
    try {
      const response = await fetch(`/api/v1/platform-integration/components/${componentId}/unregister`, {
        method: 'DELETE'
      })

      if (response.ok) {
        message.success('组件注销成功')
        fetchComponents()
      } else {
        message.error('注销失败')
      }
    } catch (error) {
      message.error('注销失败')
    }
  }

  const handleHealthCheck = async (componentId: string) => {
    try {
      const response = await fetch(`/api/v1/platform-integration/components/${componentId}/health-check`, {
        method: 'POST'
      })

      if (response.ok) {
        message.success('健康检查已启动')
        setTimeout(fetchComponents, 2000)
      } else {
        message.error('健康检查失败')
      }
    } catch (error) {
      message.error('健康检查失败')
    }
  }

  const showComponentDetail = async (component: Component) => {
    setSelectedComponent(component)
    await fetchComponentHealth(component.component_id)
    setDetailDrawerVisible(true)
  }

  const getTypeColor = (type: string) => {
    const colors = {
      ai_service: 'blue',
      data_processor: 'green',
      model_service: 'purple',
      monitoring_tool: 'orange',
      integration_service: 'cyan'
    }
    return colors[type] || 'default'
  }

  const getTypeText = (type: string) => {
    const texts = {
      ai_service: 'AI服务',
      data_processor: '数据处理',
      model_service: '模型服务',
      monitoring_tool: '监控工具',
      integration_service: '集成服务'
    }
    return texts[type] || type
  }

  const columns: ColumnsType<Component> = [
    {
      title: '组件名称',
      dataIndex: 'name',
      key: 'name',
      render: (text, record) => (
        <Space>
          <ApiOutlined />
          <div>
            <div>{text}</div>
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {record.component_id}
            </Text>
          </div>
        </Space>
      )
    },
    {
      title: '类型',
      dataIndex: 'component_type',
      key: 'component_type',
      render: (type) => <Tag color={getTypeColor(type)}>{getTypeText(type)}</Tag>
    },
    {
      title: '版本',
      dataIndex: 'version',
      key: 'version'
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => (
        <Badge
          status={status === 'healthy' ? 'success' : 'error'}
          text={status === 'healthy' ? '健康' : '异常'}
        />
      )
    },
    {
      title: '运行时间',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime) => {
        const hours = Math.floor(uptime / 3600)
        const minutes = Math.floor((uptime % 3600) / 60)
        return `${hours}h ${minutes}m`
      }
    },
    {
      title: '最后检查',
      dataIndex: 'last_check',
      key: 'last_check',
      render: (time) => new Date(time).toLocaleString()
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Tooltip title="查看详情">
            <Button
              type="text"
              icon={<EyeOutlined />}
              onClick={() => showComponentDetail(record)}
            />
          </Tooltip>
          <Tooltip title="健康检查">
            <Button
              type="text"
              icon={<ReloadOutlined />}
              onClick={() => handleHealthCheck(record.component_id)}
            />
          </Tooltip>
          <Popconfirm
            title="确认注销此组件？"
            onConfirm={() => handleUnregisterComponent(record.component_id)}
          >
            <Tooltip title="注销组件">
              <Button type="text" danger icon={<DeleteOutlined />} />
            </Tooltip>
          </Popconfirm>
        </Space>
      )
    }
  ]

  const summary = components.reduce((acc, comp) => {
    acc.total++
    if (comp.status === 'healthy') acc.healthy++
    else acc.unhealthy++
    return acc
  }, { total: 0, healthy: 0, unhealthy: 0 })

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Title level={2}>组件管理</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={fetchComponents}>
            刷新
          </Button>
          <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
            注册组件
          </Button>
        </Space>
      </div>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={8}>
          <Card>
            <Statistic
              title="总组件数"
              value={summary.total}
              prefix={<ApiOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="健康组件"
              value={summary.healthy}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="异常组件"
              value={summary.unhealthy}
              prefix={<ExclamationCircleOutlined />}
              valueStyle={{ color: '#f5222d' }}
            />
          </Card>
        </Col>
      </Row>

      <Card>
        <Table
          columns={columns}
          dataSource={components}
          rowKey="component_id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 个组件`
          }}
        />
      </Card>

      <Modal
        title="注册新组件"
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false)
          form.resetFields()
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleRegisterComponent}
        >
          <Form.Item
            name="component_id"
            label="组件ID"
            rules={[{ required: true, message: '请输入组件ID' }]}
          >
            <Input placeholder="唯一的组件标识符" />
          </Form.Item>

          <Form.Item
            name="name"
            label="组件名称"
            rules={[{ required: true, message: '请输入组件名称' }]}
          >
            <Input placeholder="组件显示名称" />
          </Form.Item>

          <Form.Item
            name="component_type"
            label="组件类型"
            rules={[{ required: true, message: '请选择组件类型' }]}
          >
            <Select placeholder="选择组件类型">
              <Option value="ai_service">AI服务</Option>
              <Option value="data_processor">数据处理</Option>
              <Option value="model_service">模型服务</Option>
              <Option value="monitoring_tool">监控工具</Option>
              <Option value="integration_service">集成服务</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="version"
            label="版本"
            rules={[{ required: true, message: '请输入版本号' }]}
          >
            <Input placeholder="如: 1.0.0" />
          </Form.Item>

          <Form.Item
            name="health_endpoint"
            label="健康检查端点"
            rules={[{ required: true, message: '请输入健康检查端点' }]}
          >
            <Input placeholder="如: http://service:8080/health" />
          </Form.Item>

          <Form.Item
            name="api_endpoint"
            label="API端点"
            rules={[{ required: true, message: '请输入API端点' }]}
          >
            <Input placeholder="如: http://service:8080/api/v1" />
          </Form.Item>

          <Form.Item
            name="metadata"
            label="元数据 (JSON格式)"
          >
            <TextArea
              rows={4}
              placeholder='{"description": "组件描述", "tags": ["tag1", "tag2"]}'
            />
          </Form.Item>

          <Form.Item style={{ marginBottom: 0 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => {
                setModalVisible(false)
                form.resetFields()
              }}>
                取消
              </Button>
              <Button type="primary" htmlType="submit">
                注册
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <Drawer
        title="组件详情"
        placement="right"
        onClose={() => setDetailDrawerVisible(false)}
        open={detailDrawerVisible}
        width={600}
      >
        {selectedComponent && (
          <div>
            <Descriptions title="基本信息" bordered column={1} size="small">
              <Descriptions.Item label="组件ID">
                {selectedComponent.component_id}
              </Descriptions.Item>
              <Descriptions.Item label="名称">
                {selectedComponent.name}
              </Descriptions.Item>
              <Descriptions.Item label="类型">
                <Tag color={getTypeColor(selectedComponent.component_type)}>
                  {getTypeText(selectedComponent.component_type)}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="版本">
                {selectedComponent.version}
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                <Badge
                  status={selectedComponent.status === 'healthy' ? 'success' : 'error'}
                  text={selectedComponent.status === 'healthy' ? '健康' : '异常'}
                />
              </Descriptions.Item>
              <Descriptions.Item label="健康检查端点">
                {selectedComponent.health_endpoint}
              </Descriptions.Item>
              <Descriptions.Item label="API端点">
                {selectedComponent.api_endpoint}
              </Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {new Date(selectedComponent.created_at).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="更新时间">
                {new Date(selectedComponent.updated_at).toLocaleString()}
              </Descriptions.Item>
            </Descriptions>

            {componentHealth && (
              <div style={{ marginTop: 24 }}>
                <Title level={4}>健康状态详情</Title>
                <Alert
                  type={componentHealth.status === 'healthy' ? 'success' : 'error'}
                  message={`状态: ${componentHealth.status === 'healthy' ? '健康' : '异常'}`}
                  description={
                    <div>
                      <div>响应时间: {componentHealth.response_time}ms</div>
                      {componentHealth.error_message && (
                        <div>错误信息: {componentHealth.error_message}</div>
                      )}
                    </div>
                  }
                  style={{ marginBottom: 16 }}
                />
                
                {componentHealth.checks?.length > 0 && (
                  <div>
                    <Text strong>检查项目:</Text>
                    <div style={{ marginTop: 8 }}>
                      {componentHealth.checks.map((check, index) => (
                        <div key={index} style={{ marginBottom: 8 }}>
                          <Badge
                            status={check.status === 'pass' ? 'success' : 'error'}
                            text={`${check.name}: ${check.message || check.status}`}
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {selectedComponent.metadata && Object.keys(selectedComponent.metadata).length > 0 && (
              <div style={{ marginTop: 24 }}>
                <Title level={4}>元数据</Title>
                <pre style={{ background: '#f5f5f5', padding: 12, borderRadius: 4 }}>
                  {JSON.stringify(selectedComponent.metadata, null, 2)}
                </pre>
              </div>
            )}
          </div>
        )}
      </Drawer>
    </div>
  )
}

export default ComponentManagementPage