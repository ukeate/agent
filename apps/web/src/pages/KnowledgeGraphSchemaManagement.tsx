import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Alert,
  Tooltip,
  Row,
  Col,
  Statistic,
  Tag,
  Typography,
  Divider,
  Select,
  Input,
  Modal,
  Form,
  message,
  Tabs,
  Tree,
  Badge,
  Descriptions,
  Collapse,
  Switch,
  Progress,
  Timeline
} from 'antd'
import {
  SchemaOutlined,
  DatabaseOutlined,
  NodeIndexOutlined,
  ShareAltOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  ReloadOutlined,
  SearchOutlined,
  SettingOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ThunderboltOutlined,
  LinkOutlined,
  KeyOutlined,
  BranchesOutlined,
  FileTextOutlined,
  WarningOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input
const { TabPane } = Tabs
const { Panel } = Collapse
const { TreeNode } = Tree

interface SchemaNode {
  label: string
  type: string
  properties: SchemaProperty[]
  constraints: string[]
  indexes: string[]
  count: number
}

interface SchemaRelationship {
  type: string
  from_node: string
  to_node: string
  properties: SchemaProperty[]
  constraints: string[]
  count: number
}

interface SchemaProperty {
  name: string
  type: 'string' | 'integer' | 'float' | 'boolean' | 'datetime' | 'array'
  required: boolean
  unique: boolean
  indexed: boolean
  default_value?: any
}

interface SchemaConstraint {
  id: string
  name: string
  type: 'unique' | 'exists' | 'check' | 'key'
  target: string
  properties: string[]
  expression?: string
  enabled: boolean
  created_at: string
}

interface SchemaIndex {
  id: string
  name: string
  type: 'btree' | 'fulltext' | 'composite'
  target: string
  properties: string[]
  status: 'online' | 'building' | 'failed'
  size: number
  usage_count: number
  created_at: string
}

interface SchemaStatistics {
  total_nodes: number
  total_relationships: number
  node_types: number
  relationship_types: number
  total_properties: number
  total_constraints: number
  total_indexes: number
}

const KnowledgeGraphSchemaManagement: React.FC = () => {
  const [nodes, setNodes] = useState<SchemaNode[]>([])
  const [relationships, setRelationships] = useState<SchemaRelationship[]>([])
  const [constraints, setConstraints] = useState<SchemaConstraint[]>([])
  const [indexes, setIndexes] = useState<SchemaIndex[]>([])
  const [statistics, setStatistics] = useState<SchemaStatistics>({} as SchemaStatistics)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')
  const [modalVisible, setModalVisible] = useState(false)
  const [modalType, setModalType] = useState<'node' | 'relationship' | 'constraint' | 'index'>('node')
  const [selectedItem, setSelectedItem] = useState<any>(null)
  const [form] = Form.useForm()

  // 模拟节点类型数据
  const mockNodes: SchemaNode[] = [
    {
      label: 'Person',
      type: 'NODE',
      properties: [
        { name: 'id', type: 'string', required: true, unique: true, indexed: true },
        { name: 'name', type: 'string', required: true, unique: false, indexed: true },
        { name: 'age', type: 'integer', required: false, unique: false, indexed: false },
        { name: 'email', type: 'string', required: false, unique: true, indexed: true },
        { name: 'created_at', type: 'datetime', required: true, unique: false, indexed: true }
      ],
      constraints: ['UNIQUE (Person.id)', 'UNIQUE (Person.email)', 'EXISTS (Person.name)'],
      indexes: ['INDEX_Person_id', 'INDEX_Person_name', 'INDEX_Person_email'],
      count: 12450
    },
    {
      label: 'Organization',
      type: 'NODE',
      properties: [
        { name: 'id', type: 'string', required: true, unique: true, indexed: true },
        { name: 'name', type: 'string', required: true, unique: false, indexed: true },
        { name: 'industry', type: 'string', required: false, unique: false, indexed: true },
        { name: 'founded', type: 'integer', required: false, unique: false, indexed: false },
        { name: 'headquarters', type: 'string', required: false, unique: false, indexed: false }
      ],
      constraints: ['UNIQUE (Organization.id)', 'EXISTS (Organization.name)'],
      indexes: ['INDEX_Organization_id', 'INDEX_Organization_name', 'INDEX_Organization_industry'],
      count: 3200
    },
    {
      label: 'Location',
      type: 'NODE',
      properties: [
        { name: 'id', type: 'string', required: true, unique: true, indexed: true },
        { name: 'name', type: 'string', required: true, unique: false, indexed: true },
        { name: 'country', type: 'string', required: false, unique: false, indexed: true },
        { name: 'coordinates', type: 'array', required: false, unique: false, indexed: false }
      ],
      constraints: ['UNIQUE (Location.id)', 'EXISTS (Location.name)'],
      indexes: ['INDEX_Location_id', 'INDEX_Location_name'],
      count: 5600
    }
  ]

  // 模拟关系类型数据
  const mockRelationships: SchemaRelationship[] = [
    {
      type: 'WORKS_FOR',
      from_node: 'Person',
      to_node: 'Organization',
      properties: [
        { name: 'position', type: 'string', required: false, unique: false, indexed: false },
        { name: 'start_date', type: 'datetime', required: false, unique: false, indexed: true },
        { name: 'end_date', type: 'datetime', required: false, unique: false, indexed: false }
      ],
      constraints: [],
      count: 8900
    },
    {
      type: 'LOCATED_IN',
      from_node: 'Organization',
      to_node: 'Location',
      properties: [
        { name: 'is_headquarters', type: 'boolean', required: false, unique: false, indexed: false }
      ],
      constraints: [],
      count: 3100
    },
    {
      type: 'LIVES_IN',
      from_node: 'Person',
      to_node: 'Location',
      properties: [
        { name: 'since', type: 'datetime', required: false, unique: false, indexed: false }
      ],
      constraints: [],
      count: 11200
    }
  ]

  // 模拟约束数据
  const mockConstraints: SchemaConstraint[] = [
    {
      id: 'constraint_001',
      name: 'Person_id_unique',
      type: 'unique',
      target: 'Person',
      properties: ['id'],
      enabled: true,
      created_at: '2025-01-15T10:30:00Z'
    },
    {
      id: 'constraint_002',
      name: 'Person_email_unique',
      type: 'unique',
      target: 'Person',
      properties: ['email'],
      enabled: true,
      created_at: '2025-01-15T10:32:00Z'
    },
    {
      id: 'constraint_003',
      name: 'Person_name_exists',
      type: 'exists',
      target: 'Person',
      properties: ['name'],
      enabled: true,
      created_at: '2025-01-15T10:35:00Z'
    },
    {
      id: 'constraint_004',
      name: 'Organization_name_exists',
      type: 'exists',
      target: 'Organization',
      properties: ['name'],
      enabled: true,
      created_at: '2025-01-15T10:40:00Z'
    }
  ]

  // 模拟索引数据
  const mockIndexes: SchemaIndex[] = [
    {
      id: 'index_001',
      name: 'INDEX_Person_id',
      type: 'btree',
      target: 'Person',
      properties: ['id'],
      status: 'online',
      size: 15.6,
      usage_count: 89500,
      created_at: '2025-01-15T10:30:00Z'
    },
    {
      id: 'index_002',
      name: 'INDEX_Person_name',
      type: 'btree',
      target: 'Person',
      properties: ['name'],
      status: 'online',
      size: 8.9,
      usage_count: 45200,
      created_at: '2025-01-15T10:32:00Z'
    },
    {
      id: 'index_003',
      name: 'INDEX_Person_fulltext',
      type: 'fulltext',
      target: 'Person',
      properties: ['name', 'email'],
      status: 'building',
      size: 0,
      usage_count: 0,
      created_at: '2025-01-22T15:00:00Z'
    },
    {
      id: 'index_004',
      name: 'INDEX_Organization_composite',
      type: 'composite',
      target: 'Organization',
      properties: ['name', 'industry'],
      status: 'online',
      size: 4.2,
      usage_count: 12300,
      created_at: '2025-01-15T11:00:00Z'
    }
  ]

  const mockStatistics: SchemaStatistics = {
    total_nodes: 21250,
    total_relationships: 23200,
    node_types: 3,
    relationship_types: 3,
    total_properties: 15,
    total_constraints: 4,
    total_indexes: 8
  }

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      setNodes(mockNodes)
      setRelationships(mockRelationships)
      setConstraints(mockConstraints)
      setIndexes(mockIndexes)
      setStatistics(mockStatistics)
    } catch (error) {
      message.error('加载模式数据失败')
    } finally {
      setLoading(false)
    }
  }

  const handleCreateItem = async (values: any) => {
    try {
      // 模拟创建API调用
      message.success(`${modalType}创建成功`)
      setModalVisible(false)
      form.resetFields()
      loadData()
    } catch (error) {
      message.error(`${modalType}创建失败`)
    }
  }

  const handleDeleteItem = async (type: string, id: string) => {
    try {
      // 模拟删除API调用
      message.success(`${type}删除成功`)
      loadData()
    } catch (error) {
      message.error(`${type}删除失败`)
    }
  }

  const toggleConstraint = async (constraintId: string) => {
    try {
      const updatedConstraints = constraints.map(constraint => 
        constraint.id === constraintId 
          ? { ...constraint, enabled: !constraint.enabled } 
          : constraint
      )
      setConstraints(updatedConstraints)
      message.success('约束状态已更新')
    } catch (error) {
      message.error('更新约束状态失败')
    }
  }

  const rebuildIndex = async (indexId: string) => {
    try {
      const updatedIndexes = indexes.map(index => 
        index.id === indexId 
          ? { ...index, status: 'building' as const } 
          : index
      )
      setIndexes(updatedIndexes)
      message.success('索引重建已开始')
    } catch (error) {
      message.error('重建索引失败')
    }
  }

  const openCreateModal = (type: 'node' | 'relationship' | 'constraint' | 'index') => {
    setModalType(type)
    setSelectedItem(null)
    setModalVisible(true)
    form.resetFields()
  }

  const getConstraintTypeColor = (type: string) => {
    const colors = {
      'unique': 'blue',
      'exists': 'green',
      'check': 'orange',
      'key': 'purple'
    }
    return colors[type as keyof typeof colors] || 'default'
  }

  const getConstraintTypeName = (type: string) => {
    const names = {
      'unique': '唯一性',
      'exists': '存在性',
      'check': '检查',
      'key': '主键'
    }
    return names[type as keyof typeof names] || type
  }

  const getIndexTypeColor = (type: string) => {
    const colors = {
      'btree': 'blue',
      'fulltext': 'green',
      'composite': 'orange'
    }
    return colors[type as keyof typeof colors] || 'default'
  }

  const getIndexTypeName = (type: string) => {
    const names = {
      'btree': 'B树索引',
      'fulltext': '全文索引',
      'composite': '复合索引'
    }
    return names[type as keyof typeof names] || type
  }

  const getStatusBadge = (status: string) => {
    const statusMap = {
      'online': { color: 'success', text: '在线' },
      'building': { color: 'processing', text: '构建中' },
      'failed': { color: 'error', text: '失败' }
    }
    const config = statusMap[status as keyof typeof statusMap]
    return <Badge status={config.color as any} text={config.text} />
  }

  const nodeColumns = [
    {
      title: '节点类型',
      dataIndex: 'label',
      key: 'label',
      render: (text: string) => (
        <Space>
          <NodeIndexOutlined />
          <Text strong>{text}</Text>
        </Space>
      ),
    },
    {
      title: '属性数量',
      key: 'properties',
      render: (_, record: SchemaNode) => record.properties.length,
    },
    {
      title: '约束数量',
      key: 'constraints',
      render: (_, record: SchemaNode) => record.constraints.length,
    },
    {
      title: '索引数量',
      key: 'indexes',
      render: (_, record: SchemaNode) => record.indexes.length,
    },
    {
      title: '实例数量',
      dataIndex: 'count',
      key: 'count',
      render: (count: number) => count.toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: SchemaNode) => (
        <Space>
          <Tooltip title="编辑">
            <Button type="text" icon={<EditOutlined />} />
          </Tooltip>
          <Tooltip title="删除">
            <Button 
              type="text" 
              danger 
              icon={<DeleteOutlined />}
              onClick={() => {
                Modal.confirm({
                  title: '确认删除',
                  content: `确定要删除节点类型"${record.label}"吗？`,
                  onOk: () => handleDeleteItem('节点类型', record.label),
                })
              }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const relationshipColumns = [
    {
      title: '关系类型',
      dataIndex: 'type',
      key: 'type',
      render: (text: string) => (
        <Space>
          <ShareAltOutlined />
          <Text strong>{text}</Text>
        </Space>
      ),
    },
    {
      title: '源节点',
      dataIndex: 'from_node',
      key: 'from_node',
    },
    {
      title: '目标节点',
      dataIndex: 'to_node',
      key: 'to_node',
    },
    {
      title: '属性数量',
      key: 'properties',
      render: (_, record: SchemaRelationship) => record.properties.length,
    },
    {
      title: '实例数量',
      dataIndex: 'count',
      key: 'count',
      render: (count: number) => count.toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: SchemaRelationship) => (
        <Space>
          <Tooltip title="编辑">
            <Button type="text" icon={<EditOutlined />} />
          </Tooltip>
          <Tooltip title="删除">
            <Button 
              type="text" 
              danger 
              icon={<DeleteOutlined />}
              onClick={() => {
                Modal.confirm({
                  title: '确认删除',
                  content: `确定要删除关系类型"${record.type}"吗？`,
                  onOk: () => handleDeleteItem('关系类型', record.type),
                })
              }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const constraintColumns = [
    {
      title: '约束名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={getConstraintTypeColor(type)}>
          {getConstraintTypeName(type)}
        </Tag>
      ),
    },
    {
      title: '目标',
      dataIndex: 'target',
      key: 'target',
    },
    {
      title: '属性',
      dataIndex: 'properties',
      key: 'properties',
      render: (properties: string[]) => (
        <Space>
          {properties.map(prop => (
            <Tag key={prop} size="small">{prop}</Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'enabled',
      key: 'enabled',
      render: (enabled: boolean, record: SchemaConstraint) => (
        <Switch 
          checked={enabled} 
          onChange={() => toggleConstraint(record.id)}
          checkedChildren="启用"
          unCheckedChildren="禁用"
        />
      ),
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time: string) => new Date(time).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: SchemaConstraint) => (
        <Space>
          <Tooltip title="删除">
            <Button 
              type="text" 
              danger 
              icon={<DeleteOutlined />}
              onClick={() => {
                Modal.confirm({
                  title: '确认删除',
                  content: `确定要删除约束"${record.name}"吗？`,
                  onOk: () => handleDeleteItem('约束', record.id),
                })
              }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const indexColumns = [
    {
      title: '索引名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={getIndexTypeColor(type)}>
          {getIndexTypeName(type)}
        </Tag>
      ),
    },
    {
      title: '目标',
      dataIndex: 'target',
      key: 'target',
    },
    {
      title: '属性',
      dataIndex: 'properties',
      key: 'properties',
      render: (properties: string[]) => (
        <Space>
          {properties.map(prop => (
            <Tag key={prop} size="small">{prop}</Tag>
          ))}
        </Space>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusBadge(status),
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => `${size}MB`,
    },
    {
      title: '使用次数',
      dataIndex: 'usage_count',
      key: 'usage_count',
      render: (count: number) => count.toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: SchemaIndex) => (
        <Space>
          <Tooltip title="重建">
            <Button 
              type="text" 
              icon={<ReloadOutlined />}
              onClick={() => rebuildIndex(record.id)}
              disabled={record.status === 'building'}
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
                  content: `确定要删除索引"${record.name}"吗？`,
                  onOk: () => handleDeleteItem('索引', record.id),
                })
              }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <SchemaOutlined style={{ marginRight: '8px' }} />
          图模式管理
        </Title>
        <Paragraph type="secondary">
          管理知识图谱的模式定义，包括节点类型、关系类型、约束和索引
        </Paragraph>
      </div>

      {/* 统计概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={4}>
          <Card>
            <Statistic
              title="节点类型"
              value={statistics.node_types}
              prefix={<NodeIndexOutlined />}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="关系类型"
              value={statistics.relationship_types}
              prefix={<ShareAltOutlined />}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="总属性数"
              value={statistics.total_properties}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="约束数量"
              value={statistics.total_constraints}
              prefix={<KeyOutlined />}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="索引数量"
              value={statistics.total_indexes}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="总实例数"
              value={statistics.total_nodes + statistics.total_relationships}
              prefix={<DatabaseOutlined />}
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
                placeholder="搜索模式元素..."
                prefix={<SearchOutlined />}
                style={{ width: 250 }}
              />
            </Space>
          </Col>
          <Col>
            <Space>
              <Button icon={<SettingOutlined />}>模式配置</Button>
              <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>
                刷新
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="模式概览" key="overview">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="节点类型" extra={
                <Button 
                  type="primary" 
                  size="small" 
                  icon={<PlusOutlined />}
                  onClick={() => openCreateModal('node')}
                >
                  添加节点类型
                </Button>
              }>
                <Table
                  columns={nodeColumns}
                  dataSource={nodes}
                  rowKey="label"
                  size="small"
                  pagination={false}
                />
              </Card>
            </Col>
            <Col span={12}>
              <Card title="关系类型" extra={
                <Button 
                  type="primary" 
                  size="small" 
                  icon={<PlusOutlined />}
                  onClick={() => openCreateModal('relationship')}
                >
                  添加关系类型
                </Button>
              }>
                <Table
                  columns={relationshipColumns}
                  dataSource={relationships}
                  rowKey="type"
                  size="small"
                  pagination={false}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="节点类型" key="nodes">
          <Card title="节点类型管理" extra={
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => openCreateModal('node')}
            >
              添加节点类型
            </Button>
          }>
            <Table
              columns={nodeColumns}
              dataSource={nodes}
              rowKey="label"
              loading={loading}
              expandable={{
                expandedRowRender: (record) => (
                  <Descriptions title="属性详情" bordered size="small">
                    {record.properties.map(prop => (
                      <Descriptions.Item 
                        key={prop.name} 
                        label={prop.name}
                        span={1}
                      >
                        <Space>
                          <Tag color="blue">{prop.type}</Tag>
                          {prop.required && <Tag color="red">必需</Tag>}
                          {prop.unique && <Tag color="orange">唯一</Tag>}
                          {prop.indexed && <Tag color="green">索引</Tag>}
                        </Space>
                      </Descriptions.Item>
                    ))}
                  </Descriptions>
                ),
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="关系类型" key="relationships">
          <Card title="关系类型管理" extra={
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => openCreateModal('relationship')}
            >
              添加关系类型
            </Button>
          }>
            <Table
              columns={relationshipColumns}
              dataSource={relationships}
              rowKey="type"
              loading={loading}
              expandable={{
                expandedRowRender: (record) => (
                  <div>
                    <Title level={5}>关系流向</Title>
                    <div style={{ marginBottom: '16px' }}>
                      <Tag color="blue">{record.from_node}</Tag>
                      <span style={{ margin: '0 8px' }}>
                        <LinkOutlined />
                      </span>
                      <Tag color="green">{record.to_node}</Tag>
                    </div>
                    
                    {record.properties.length > 0 && (
                      <>
                        <Title level={5}>属性详情</Title>
                        <Descriptions bordered size="small">
                          {record.properties.map(prop => (
                            <Descriptions.Item 
                              key={prop.name} 
                              label={prop.name}
                              span={1}
                            >
                              <Space>
                                <Tag color="blue">{prop.type}</Tag>
                                {prop.required && <Tag color="red">必需</Tag>}
                                {prop.indexed && <Tag color="green">索引</Tag>}
                              </Space>
                            </Descriptions.Item>
                          ))}
                        </Descriptions>
                      </>
                    )}
                  </div>
                ),
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="约束管理" key="constraints">
          <Card title="约束管理" extra={
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => openCreateModal('constraint')}
            >
              添加约束
            </Button>
          }>
            <Table
              columns={constraintColumns}
              dataSource={constraints}
              rowKey="id"
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="索引管理" key="indexes">
          <Card title="索引管理" extra={
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => openCreateModal('index')}
            >
              添加索引
            </Button>
          }>
            <Table
              columns={indexColumns}
              dataSource={indexes}
              rowKey="id"
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="模式可视化" key="visualization">
          <Card title="模式可视化">
            <Alert
              message="模式可视化"
              description="此区域将显示知识图谱的模式结构图，包括节点类型、关系类型及其相互关系的可视化表示。"
              type="info"
              showIcon
              style={{ marginBottom: '16px' }}
            />
            <div style={{ 
              height: '500px', 
              border: '1px dashed #d9d9d9', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              backgroundColor: '#fafafa'
            }}>
              <div style={{ textAlign: 'center' }}>
                <BranchesOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
                <div style={{ marginTop: '16px' }}>
                  <Text type="secondary">模式结构图将在此显示</Text>
                </div>
              </div>
            </div>
          </Card>
        </TabPane>
      </Tabs>

      {/* 创建/编辑模态框 */}
      <Modal
        title={`${selectedItem ? '编辑' : '创建'}${modalType === 'node' ? '节点类型' : modalType === 'relationship' ? '关系类型' : modalType === 'constraint' ? '约束' : '索引'}`}
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
          onFinish={handleCreateItem}
        >
          {modalType === 'node' && (
            <>
              <Form.Item
                name="label"
                label="节点标签"
                rules={[{ required: true, message: '请输入节点标签' }]}
              >
                <Input placeholder="例如：Person, Organization" />
              </Form.Item>
              <Form.Item
                name="properties"
                label="属性定义 (JSON格式)"
              >
                <TextArea 
                  rows={6}
                  placeholder='[{"name":"id","type":"string","required":true,"unique":true}]'
                />
              </Form.Item>
            </>
          )}

          {modalType === 'relationship' && (
            <>
              <Form.Item
                name="type"
                label="关系类型"
                rules={[{ required: true, message: '请输入关系类型' }]}
              >
                <Input placeholder="例如：WORKS_FOR, LOCATED_IN" />
              </Form.Item>
              <Form.Item
                name="from_node"
                label="源节点类型"
                rules={[{ required: true, message: '请选择源节点类型' }]}
              >
                <Select placeholder="选择源节点类型">
                  {nodes.map(node => (
                    <Select.Option key={node.label} value={node.label}>
                      {node.label}
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>
              <Form.Item
                name="to_node"
                label="目标节点类型"
                rules={[{ required: true, message: '请选择目标节点类型' }]}
              >
                <Select placeholder="选择目标节点类型">
                  {nodes.map(node => (
                    <Select.Option key={node.label} value={node.label}>
                      {node.label}
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>
            </>
          )}

          {modalType === 'constraint' && (
            <>
              <Form.Item
                name="name"
                label="约束名称"
                rules={[{ required: true, message: '请输入约束名称' }]}
              >
                <Input placeholder="例如：Person_email_unique" />
              </Form.Item>
              <Form.Item
                name="type"
                label="约束类型"
                rules={[{ required: true, message: '请选择约束类型' }]}
              >
                <Select placeholder="选择约束类型">
                  <Select.Option value="unique">唯一性约束</Select.Option>
                  <Select.Option value="exists">存在性约束</Select.Option>
                  <Select.Option value="check">检查约束</Select.Option>
                  <Select.Option value="key">主键约束</Select.Option>
                </Select>
              </Form.Item>
              <Form.Item
                name="target"
                label="目标类型"
                rules={[{ required: true, message: '请选择目标类型' }]}
              >
                <Select placeholder="选择目标类型">
                  {nodes.map(node => (
                    <Select.Option key={node.label} value={node.label}>
                      {node.label}
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>
            </>
          )}

          {modalType === 'index' && (
            <>
              <Form.Item
                name="name"
                label="索引名称"
                rules={[{ required: true, message: '请输入索引名称' }]}
              >
                <Input placeholder="例如：INDEX_Person_name" />
              </Form.Item>
              <Form.Item
                name="type"
                label="索引类型"
                rules={[{ required: true, message: '请选择索引类型' }]}
              >
                <Select placeholder="选择索引类型">
                  <Select.Option value="btree">B树索引</Select.Option>
                  <Select.Option value="fulltext">全文索引</Select.Option>
                  <Select.Option value="composite">复合索引</Select.Option>
                </Select>
              </Form.Item>
              <Form.Item
                name="target"
                label="目标类型"
                rules={[{ required: true, message: '请选择目标类型' }]}
              >
                <Select placeholder="选择目标类型">
                  {nodes.map(node => (
                    <Select.Option key={node.label} value={node.label}>
                      {node.label}
                    </Select.Option>
                  ))}
                </Select>
              </Form.Item>
            </>
          )}
        </Form>
      </Modal>
    </div>
  )
}

export default KnowledgeGraphSchemaManagement