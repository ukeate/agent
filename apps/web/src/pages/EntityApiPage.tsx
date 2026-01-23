import React, { useState, useCallback, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Typography,
  Row,
  Col,
  Space,
  Button,
  Table,
  Form,
  Input,
  Modal,
  Select,
  Tag,
  message,
  Tabs,
  Statistic,
  Alert,
  Badge,
  Drawer,
  Timeline,
  List,
} from 'antd'
import apiClient from '../services/apiClient'
import {
  NodeIndexOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  SearchOutlined,
  ApiOutlined,
  DatabaseOutlined,
  ExportOutlined,
  ImportOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { Option } = Select
const { TextArea } = Input

interface Entity {
  id: string
  uri: string
  type: string
  label: string
  properties: Record<string, any>
  created: string
  updated: string
  status: 'active' | 'inactive' | 'pending'
}

interface ApiCall {
  id: string
  method: 'GET' | 'POST' | 'PUT' | 'DELETE'
  endpoint: string
  timestamp: string
  status: number
  responseTime: number
  entity?: string
}

const EntityApiPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [entities, setEntities] = useState<Entity[]>([])
  const [loadingEntities, setLoadingEntities] = useState(true)

  const [apiCalls, setApiCalls] = useState<ApiCall[]>([])

  const [modalVisible, setModalVisible] = useState(false)
  const [drawerVisible, setDrawerVisible] = useState(false)
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null)
  const [form] = Form.useForm()

  // 加载实体列表
  const loadEntities = useCallback(async () => {
    setLoadingEntities(true)
    try {
      const response = await apiClient.get('/entities')
      logger.log('实体API调用成功:', response.data)
      if (response.data?.entities) {
        setEntities(response.data.entities)
      } else {
        setEntities([])
      }
    } catch (error) {
      logger.error('加载实体失败:', error)
      message.error('加载实体列表失败')
      setEntities([])
    } finally {
      setLoadingEntities(false)
    }
  }, [])

  // 获取单个实体
  const loadEntity = useCallback(async (entityId: string) => {
    try {
      const response = await apiClient.get(`/entities/${entityId}`)
      logger.log('获取实体详情成功:', response.data)
      return response.data
    } catch (error) {
      logger.error('获取实体详情失败:', error)
      message.error('获取实体详情失败')
      return null
    }
  }, [])

  useEffect(() => {
    loadEntities()
  }, [])

  const entityColumns = [
    {
      title: 'URI',
      dataIndex: 'uri',
      key: 'uri',
      ellipsis: true,
      render: (uri: string) => (
        <Text code style={{ fontSize: '12px' }}>
          {uri}
        </Text>
      ),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => <Tag color="blue">{type}</Tag>,
    },
    {
      title: '标签',
      dataIndex: 'label',
      key: 'label',
    },
    {
      title: '属性数',
      dataIndex: 'properties',
      key: 'properties',
      render: (properties: Record<string, any>) => (
        <Badge count={Object.keys(properties).length} color="green" />
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag
          color={
            status === 'active'
              ? 'green'
              : status === 'inactive'
                ? 'red'
                : 'orange'
          }
        >
          {status === 'active'
            ? '活跃'
            : status === 'inactive'
              ? '停用'
              : '待处理'}
        </Tag>
      ),
    },
    {
      title: '更新时间',
      dataIndex: 'updated',
      key: 'updated',
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: Entity) => (
        <Space size="small">
          <Button
            type="text"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedEntity(record)
              setDrawerVisible(true)
            }}
          />
          <Button
            type="text"
            icon={<EditOutlined />}
            onClick={() => {
              setSelectedEntity(record)
              form.setFieldsValue(record)
              setModalVisible(true)
            }}
          />
          <Button
            type="text"
            danger
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteEntity(record.id)}
          />
        </Space>
      ),
    },
  ]

  const apiCallColumns = [
    {
      title: '方法',
      dataIndex: 'method',
      key: 'method',
      render: (method: string) => (
        <Tag
          color={
            method === 'GET'
              ? 'blue'
              : method === 'POST'
                ? 'green'
                : method === 'PUT'
                  ? 'orange'
                  : 'red'
          }
        >
          {method}
        </Tag>
      ),
    },
    {
      title: '端点',
      dataIndex: 'endpoint',
      key: 'endpoint',
      ellipsis: true,
      render: (endpoint: string) => (
        <Text code style={{ fontSize: '12px' }}>
          {endpoint}
        </Text>
      ),
    },
    {
      title: '时间戳',
      dataIndex: 'timestamp',
      key: 'timestamp',
    },
    {
      title: '状态码',
      dataIndex: 'status',
      key: 'status',
      render: (status: number) => (
        <Tag color={status < 300 ? 'green' : status < 400 ? 'orange' : 'red'}>
          {status}
        </Tag>
      ),
    },
    {
      title: '响应时间',
      dataIndex: 'responseTime',
      key: 'responseTime',
      render: (time: number) => `${time}ms`,
    },
    {
      title: '实体',
      dataIndex: 'entity',
      key: 'entity',
      render: (entity: string) =>
        entity && (
          <Text code style={{ fontSize: '12px' }}>
            {entity}
          </Text>
        ),
    },
  ]

  const handleCreateEntity = useCallback(
    async (values: any) => {
      setLoading(true)
      try {
        const entityData = {
          uri: values.uri,
          type: values.type,
          label: values.label,
          properties: JSON.parse(values.properties || '{}'),
        }

        if (selectedEntity) {
          // 更新实体
          const response = await apiClient.put(
            `/entities/${selectedEntity.id}`,
            entityData
          )
          logger.log('更新实体成功:', response.data)
          message.success('实体更新成功')
        } else {
          // 创建新实体
          const response = await apiClient.post('/entities', entityData)
          logger.log('创建实体成功:', response.data)
          message.success('实体创建成功')
        }

        // 重新加载实体列表
        await loadEntities()
        setModalVisible(false)
        form.resetFields()
        setSelectedEntity(null)
      } catch (error) {
        logger.error('实体操作失败:', error)
        message.error(selectedEntity ? '实体更新失败' : '实体创建失败')
      } finally {
        setLoading(false)
      }
    },
    [form, selectedEntity, loadEntities]
  )

  const handleDeleteEntity = useCallback(
    async (id: string) => {
      Modal.confirm({
        title: '确认删除',
        content: '确定要删除这个实体吗？',
        onOk: async () => {
          try {
            const response = await apiClient.delete(`/entities/${id}`)
            logger.log('删除实体成功:', response.data)
            message.success('实体删除成功')
            // 重新加载实体列表
            await loadEntities()
          } catch (error) {
            logger.error('删除实体失败:', error)
            message.error('实体删除失败')
          }
        },
      })
    },
    [loadEntities]
  )

  const apiEndpoints = [
    {
      method: 'GET',
      endpoint: '/api/v1/entities',
      description: '获取所有实体列表',
      example: 'GET /api/v1/entities?type=foaf:Person&limit=10',
    },
    {
      method: 'POST',
      endpoint: '/api/v1/entities',
      description: '创建新实体',
      example: 'POST /api/v1/entities',
    },
    {
      method: 'GET',
      endpoint: '/api/v1/entities/{id}',
      description: '获取指定实体详情',
      example: 'GET /api/v1/entities/john-doe',
    },
    {
      method: 'PUT',
      endpoint: '/api/v1/entities/{id}',
      description: '更新实体信息',
      example: 'PUT /api/v1/entities/john-doe',
    },
    {
      method: 'DELETE',
      endpoint: '/api/v1/entities/{id}',
      description: '删除指定实体',
      example: 'DELETE /api/v1/entities/john-doe',
    },
  ]

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <NodeIndexOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          实体CRUD API管理
        </Title>
        <Paragraph>
          完整的实体生命周期管理API，支持RESTful操作和批量处理
        </Paragraph>
      </div>

      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card title="API统计概览" size="small">
            <Row gutter={16}>
              <Col span={6}>
                <Statistic
                  title="总实体数"
                  value={entities.length}
                  prefix={<DatabaseOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="今日API调用"
                  value={apiCalls.length}
                  prefix={<ApiOutlined />}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="平均响应时间"
                  value={
                    apiCalls.reduce((sum, call) => sum + call.responseTime, 0) /
                    apiCalls.length
                  }
                  suffix="ms"
                  precision={0}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="成功率"
                  value={
                    (apiCalls.filter(call => call.status < 400).length /
                      apiCalls.length) *
                    100
                  }
                  suffix="%"
                  precision={1}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={16}>
          <Tabs defaultActiveKey="entities" size="small">
            <TabPane tab="实体管理" key="entities">
              <Card
                size="small"
                title="实体列表"
                extra={
                  <Space>
                    <Button
                      type="primary"
                      icon={<PlusOutlined />}
                      onClick={() => {
                        setSelectedEntity(null)
                        form.resetFields()
                        setModalVisible(true)
                      }}
                    >
                      创建实体
                    </Button>
                    <Button icon={<ImportOutlined />}>批量导入</Button>
                    <Button icon={<ExportOutlined />}>批量导出</Button>
                  </Space>
                }
              >
                <Table
                  dataSource={entities}
                  columns={entityColumns}
                  rowKey="id"
                  size="small"
                  loading={loadingEntities}
                  pagination={{
                    showSizeChanger: true,
                    showQuickJumper: true,
                    showTotal: total => `共 ${total} 个实体`,
                  }}
                />
              </Card>
            </TabPane>

            <TabPane tab="API调用记录" key="api-calls">
              <Card size="small">
                <Table
                  dataSource={apiCalls}
                  columns={apiCallColumns}
                  rowKey="id"
                  size="small"
                  pagination={{
                    showSizeChanger: true,
                    showQuickJumper: true,
                    showTotal: total => `共 ${total} 次调用`,
                  }}
                />
              </Card>
            </TabPane>
          </Tabs>
        </Col>

        <Col span={8}>
          <Card title="API文档" size="small" style={{ marginBottom: '16px' }}>
            <List
              size="small"
              dataSource={apiEndpoints}
              renderItem={endpoint => (
                <List.Item>
                  <Space
                    direction="vertical"
                    size="small"
                    style={{ width: '100%' }}
                  >
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                      }}
                    >
                      <Tag
                        color={
                          endpoint.method === 'GET'
                            ? 'blue'
                            : endpoint.method === 'POST'
                              ? 'green'
                              : endpoint.method === 'PUT'
                                ? 'orange'
                                : 'red'
                        }
                      >
                        {endpoint.method}
                      </Tag>
                      <Text code style={{ fontSize: '12px' }}>
                        {endpoint.endpoint}
                      </Text>
                    </div>
                    <Text type="secondary" style={{ fontSize: '12px' }}>
                      {endpoint.description}
                    </Text>
                    <Text code style={{ fontSize: '11px', color: '#666' }}>
                      {endpoint.example}
                    </Text>
                  </Space>
                </List.Item>
              )}
            />
          </Card>

          <Card title="最近活动" size="small">
            <Timeline size="small">
              {apiCalls.slice(0, 5).map((call, index) => (
                <Timeline.Item
                  key={call.id}
                  color={call.status < 300 ? 'green' : 'red'}
                  dot={
                    call.status < 300 ? (
                      <CheckCircleOutlined />
                    ) : (
                      <ClockCircleOutlined />
                    )
                  }
                >
                  <div>
                    <Space>
                      <Tag
                        size="small"
                        color={
                          call.method === 'GET'
                            ? 'blue'
                            : call.method === 'POST'
                              ? 'green'
                              : call.method === 'PUT'
                                ? 'orange'
                                : 'red'
                        }
                      >
                        {call.method}
                      </Tag>
                      <Text style={{ fontSize: '12px' }}>{call.endpoint}</Text>
                    </Space>
                    <br />
                    <Text type="secondary" style={{ fontSize: '11px' }}>
                      {call.timestamp} - {call.responseTime}ms
                    </Text>
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </Col>
      </Row>

      <Modal
        title={selectedEntity ? '编辑实体' : '创建实体'}
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form form={form} layout="vertical" onFinish={handleCreateEntity}>
          <Form.Item
            name="uri"
            label="实体URI"
            rules={[{ required: true, message: '请输入实体URI' }]}
          >
            <Input placeholder="http://example.org/entity/name" />
          </Form.Item>

          <Form.Item
            name="type"
            label="实体类型"
            rules={[{ required: true, message: '请选择实体类型' }]}
          >
            <Select placeholder="选择实体类型">
              <Option value="foaf:Person">foaf:Person</Option>
              <Option value="org:Organization">org:Organization</Option>
              <Option value="schema:Product">schema:Product</Option>
              <Option value="owl:Class">owl:Class</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="label"
            label="实体标签"
            rules={[{ required: true, message: '请输入实体标签' }]}
          >
            <Input placeholder="实体的可读名称" />
          </Form.Item>

          <Form.Item name="properties" label="实体属性 (JSON格式)">
            <TextArea
              rows={6}
              placeholder={
                '{\n  "property1": "value1",\n  "property2": "value2"\n}'
              }
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                {selectedEntity ? '更新' : '创建'}
              </Button>
              <Button onClick={() => setModalVisible(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <Drawer
        title="实体详情"
        placement="right"
        open={drawerVisible}
        onClose={() => setDrawerVisible(false)}
        width={500}
      >
        {selectedEntity && (
          <Space direction="vertical" style={{ width: '100%' }}>
            <Card size="small" title="基本信息">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text strong>URI: </Text>
                  <Text code>{selectedEntity.uri}</Text>
                </div>
                <div>
                  <Text strong>类型: </Text>
                  <Tag color="blue">{selectedEntity.type}</Tag>
                </div>
                <div>
                  <Text strong>标签: </Text>
                  <Text>{selectedEntity.label}</Text>
                </div>
                <div>
                  <Text strong>状态: </Text>
                  <Tag
                    color={selectedEntity.status === 'active' ? 'green' : 'red'}
                  >
                    {selectedEntity.status === 'active' ? '活跃' : '停用'}
                  </Tag>
                </div>
              </Space>
            </Card>

            <Card size="small" title="属性信息">
              <Space direction="vertical" style={{ width: '100%' }}>
                {Object.entries(selectedEntity.properties).map(
                  ([key, value]) => (
                    <div key={key}>
                      <Text strong>{key}: </Text>
                      <Text>{String(value)}</Text>
                    </div>
                  )
                )}
              </Space>
            </Card>

            <Card size="small" title="时间信息">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text strong>创建时间: </Text>
                  <Text>{selectedEntity.created}</Text>
                </div>
                <div>
                  <Text strong>更新时间: </Text>
                  <Text>{selectedEntity.updated}</Text>
                </div>
              </Space>
            </Card>
          </Space>
        )}
      </Drawer>
    </div>
  )
}

export default EntityApiPage
