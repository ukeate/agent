import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Modal,
  Form,
  Input,
  Select,
  Tag,
  Alert,
  Tooltip,
  Row,
  Col,
  Statistic,
  Progress,
  Drawer,
  Typography,
  Divider,
  Badge,
  message
} from 'antd'
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  SearchOutlined,
  NodeIndexOutlined,
  InfoCircleOutlined,
  ThunderboltOutlined,
  SyncOutlined,
  ExportOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input

interface Entity {
  id: string
  canonical_form: string
  type: string
  properties: Record<string, any>
  confidence: number
  created_at: string
  updated_at: string
  source: string
  status: 'active' | 'pending' | 'merged'
}

const KnowledgeGraphEntityManagement: React.FC = () => {
  const [entities, setEntities] = useState<Entity[]>([])
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false)
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null)
  const [form] = Form.useForm()
  const [searchValue, setSearchValue] = useState('')
  const [filterType, setFilterType] = useState<string>('all')

  // 模拟数据
  const mockEntities: Entity[] = [
    {
      id: 'entity_001',
      canonical_form: '张三',
      type: 'PERSON',
      properties: {
        age: 30,
        occupation: '软件工程师',
        department: '技术部',
        email: 'zhangsan@example.com'
      },
      confidence: 0.95,
      created_at: '2025-01-15T10:30:00Z',
      updated_at: '2025-01-20T14:15:00Z',
      source: 'document_001',
      status: 'active'
    },
    {
      id: 'entity_002',
      canonical_form: '苹果公司',
      type: 'ORGANIZATION',
      properties: {
        industry: '科技',
        founded: 1976,
        headquarters: '加利福尼亚州库比蒂诺',
        stock_symbol: 'AAPL'
      },
      confidence: 0.98,
      created_at: '2025-01-12T09:20:00Z',
      updated_at: '2025-01-18T16:45:00Z',
      source: 'document_002',
      status: 'active'
    },
    {
      id: 'entity_003',
      canonical_form: '北京',
      type: 'LOCATION',
      properties: {
        country: '中国',
        population: 21540000,
        area: 16410.54,
        capital: true
      },
      confidence: 0.99,
      created_at: '2025-01-10T08:15:00Z',
      updated_at: '2025-01-19T11:30:00Z',
      source: 'document_003',
      status: 'active'
    },
    {
      id: 'entity_004',
      canonical_form: 'Python',
      type: 'TECHNOLOGY',
      properties: {
        category: '编程语言',
        paradigm: '多范式',
        first_appeared: 1991,
        developer: 'Guido van Rossum'
      },
      confidence: 0.92,
      created_at: '2025-01-08T14:20:00Z',
      updated_at: '2025-01-17T09:45:00Z',
      source: 'document_004',
      status: 'pending'
    }
  ]

  useEffect(() => {
    loadEntities()
  }, [])

  const loadEntities = async () => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      setEntities(mockEntities)
    } catch (error) {
      message.error('加载实体列表失败')
    } finally {
      setLoading(false)
    }
  }

  const handleCreateEntity = async (values: any) => {
    try {
      // 模拟创建实体API调用
      const newEntity: Entity = {
        id: `entity_${Date.now()}`,
        canonical_form: values.canonical_form,
        type: values.type,
        properties: JSON.parse(values.properties || '{}'),
        confidence: values.confidence / 100,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        source: values.source || 'manual',
        status: 'active'
      }
      
      setEntities([newEntity, ...entities])
      setModalVisible(false)
      form.resetFields()
      message.success('实体创建成功')
    } catch (error) {
      message.error('创建实体失败')
    }
  }

  const handleDeleteEntity = async (entityId: string) => {
    try {
      // 模拟删除API调用
      setEntities(entities.filter(e => e.id !== entityId))
      message.success('实体删除成功')
    } catch (error) {
      message.error('删除实体失败')
    }
  }

  const handleShowDetail = (entity: Entity) => {
    setSelectedEntity(entity)
    setDetailDrawerVisible(true)
  }

  const getTypeColor = (type: string) => {
    const colors = {
      'PERSON': 'blue',
      'ORGANIZATION': 'green',
      'LOCATION': 'orange',
      'TECHNOLOGY': 'purple',
      'EVENT': 'red',
      'CONCEPT': 'cyan'
    }
    return colors[type as keyof typeof colors] || 'default'
  }

  const getStatusBadge = (status: string) => {
    const statusMap = {
      'active': { color: 'success', text: '活跃' },
      'pending': { color: 'processing', text: '待处理' },
      'merged': { color: 'default', text: '已合并' }
    }
    const config = statusMap[status as keyof typeof statusMap]
    return <Badge status={config.color as any} text={config.text} />
  }

  const filteredEntities = entities.filter(entity => {
    const matchesSearch = entity.canonical_form.toLowerCase().includes(searchValue.toLowerCase()) ||
                         entity.type.toLowerCase().includes(searchValue.toLowerCase())
    const matchesType = filterType === 'all' || entity.type === filterType
    return matchesSearch && matchesType
  })

  const entityTypes = [...new Set(entities.map(e => e.type))]

  const columns = [
    {
      title: '实体名称',
      dataIndex: 'canonical_form',
      key: 'canonical_form',
      render: (text: string, record: Entity) => (
        <Button 
          type="link" 
          onClick={() => handleShowDetail(record)}
          style={{ padding: 0, height: 'auto' }}
        >
          <strong>{text}</strong>
        </Button>
      ),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={getTypeColor(type)}>{type}</Tag>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => (
        <Progress 
          percent={Math.round(confidence * 100)} 
          size="small"
          strokeColor={confidence > 0.9 ? '#52c41a' : confidence > 0.7 ? '#faad14' : '#ff4d4f'}
        />
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusBadge(status),
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      render: (date: string) => new Date(date).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: Entity) => (
        <Space>
          <Tooltip title="查看详情">
            <Button 
              type="text" 
              icon={<InfoCircleOutlined />} 
              onClick={() => handleShowDetail(record)}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button 
              type="text" 
              icon={<EditOutlined />} 
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button 
              type="text" 
              danger 
              icon={<DeleteOutlined />} 
              onClick={() => {
                Modal.confirm({
                  title: '确认删除',
                  content: `确定要删除实体"${record.canonical_form}"吗？`,
                  onOk: () => handleDeleteEntity(record.id),
                })
              }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const statistics = {
    total: entities.length,
    active: entities.filter(e => e.status === 'active').length,
    pending: entities.filter(e => e.status === 'pending').length,
    avgConfidence: entities.length > 0 
      ? entities.reduce((sum, e) => sum + e.confidence, 0) / entities.length 
      : 0
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <NodeIndexOutlined style={{ marginRight: '8px' }} />
          实体管理
        </Title>
        <Paragraph type="secondary">
          管理知识图谱中的实体，包括创建、编辑、删除和查看实体详情
        </Paragraph>
      </div>

      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总实体数"
              value={statistics.total}
              prefix={<NodeIndexOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃实体"
              value={statistics.active}
              valueStyle={{ color: '#3f8600' }}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="待处理实体"
              value={statistics.pending}
              valueStyle={{ color: '#cf1322' }}
              prefix={<SyncOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均置信度"
              value={Math.round(statistics.avgConfidence * 100)}
              suffix="%"
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 操作栏 */}
      <Card style={{ marginBottom: '16px' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Input
                placeholder="搜索实体名称或类型"
                prefix={<SearchOutlined />}
                value={searchValue}
                onChange={(e) => setSearchValue(e.target.value)}
                style={{ width: 250 }}
              />
              <Select
                placeholder="筛选类型"
                value={filterType}
                onChange={setFilterType}
                style={{ width: 150 }}
              >
                <Select.Option value="all">全部类型</Select.Option>
                {entityTypes.map(type => (
                  <Select.Option key={type} value={type}>{type}</Select.Option>
                ))}
              </Select>
            </Space>
          </Col>
          <Col>
            <Space>
              <Button icon={<ExportOutlined />}>导出</Button>
              <Button icon={<SyncOutlined />} onClick={loadEntities}>刷新</Button>
              <Button 
                type="primary" 
                icon={<PlusOutlined />}
                onClick={() => setModalVisible(true)}
              >
                新建实体
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 实体列表 */}
      <Card>
        <Table
          columns={columns}
          dataSource={filteredEntities}
          rowKey="id"
          loading={loading}
          pagination={{
            total: filteredEntities.length,
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 条记录`,
          }}
        />
      </Card>

      {/* 创建实体模态框 */}
      <Modal
        title="创建新实体"
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false)
          form.resetFields()
        }}
        onOk={() => form.submit()}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateEntity}
        >
          <Form.Item
            name="canonical_form"
            label="实体名称"
            rules={[{ required: true, message: '请输入实体名称' }]}
          >
            <Input placeholder="请输入实体的标准名称" />
          </Form.Item>

          <Form.Item
            name="type"
            label="实体类型"
            rules={[{ required: true, message: '请选择实体类型' }]}
          >
            <Select placeholder="请选择实体类型">
              <Select.Option value="PERSON">人物 (PERSON)</Select.Option>
              <Select.Option value="ORGANIZATION">组织 (ORGANIZATION)</Select.Option>
              <Select.Option value="LOCATION">地点 (LOCATION)</Select.Option>
              <Select.Option value="TECHNOLOGY">技术 (TECHNOLOGY)</Select.Option>
              <Select.Option value="EVENT">事件 (EVENT)</Select.Option>
              <Select.Option value="CONCEPT">概念 (CONCEPT)</Select.Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="properties"
            label="属性 (JSON格式)"
            rules={[
              {
                validator: (_, value) => {
                  if (!value) return Promise.resolve()
                  try {
                    JSON.parse(value)
                    return Promise.resolve()
                  } catch {
                    return Promise.reject(new Error('请输入有效的JSON格式'))
                  }
                }
              }
            ]}
          >
            <TextArea 
              rows={4}
              placeholder='{"key": "value", "key2": "value2"}'
            />
          </Form.Item>

          <Form.Item
            name="confidence"
            label="置信度"
            initialValue={90}
            rules={[{ required: true, message: '请设置置信度' }]}
          >
            <Input
              type="number"
              min={0}
              max={100}
              suffix="%"
              placeholder="0-100"
            />
          </Form.Item>

          <Form.Item
            name="source"
            label="数据源"
          >
            <Input placeholder="数据来源标识" />
          </Form.Item>
        </Form>
      </Modal>

      {/* 实体详情抽屉 */}
      <Drawer
        title="实体详情"
        placement="right"
        width={500}
        open={detailDrawerVisible}
        onClose={() => setDetailDrawerVisible(false)}
      >
        {selectedEntity && (
          <div>
            <Title level={4}>{selectedEntity.canonical_form}</Title>
            <Tag color={getTypeColor(selectedEntity.type)} style={{ marginBottom: '16px' }}>
              {selectedEntity.type}
            </Tag>

            <Divider />

            <div style={{ marginBottom: '16px' }}>
              <Text strong>置信度: </Text>
              <Progress 
                percent={Math.round(selectedEntity.confidence * 100)} 
                size="small"
                strokeColor={selectedEntity.confidence > 0.9 ? '#52c41a' : selectedEntity.confidence > 0.7 ? '#faad14' : '#ff4d4f'}
                style={{ width: '200px' }}
              />
            </div>

            <div style={{ marginBottom: '16px' }}>
              <Text strong>状态: </Text>
              {getStatusBadge(selectedEntity.status)}
            </div>

            <div style={{ marginBottom: '16px' }}>
              <Text strong>创建时间: </Text>
              <Text>{new Date(selectedEntity.created_at).toLocaleString()}</Text>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <Text strong>更新时间: </Text>
              <Text>{new Date(selectedEntity.updated_at).toLocaleString()}</Text>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <Text strong>数据源: </Text>
              <Text>{selectedEntity.source}</Text>
            </div>

            <Divider />

            <Title level={5}>属性信息</Title>
            <Card size="small">
              <pre style={{ whiteSpace: 'pre-wrap', fontSize: '12px' }}>
                {JSON.stringify(selectedEntity.properties, null, 2)}
              </pre>
            </Card>
          </div>
        )}
      </Drawer>
    </div>
  )
}

export default KnowledgeGraphEntityManagement