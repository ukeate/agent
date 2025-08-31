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
  message,
  Graph
} from 'antd'
import {
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  SearchOutlined,
  ShareAltOutlined,
  InfoCircleOutlined,
  ThunderboltOutlined,
  SyncOutlined,
  ExportOutlined,
  NodeIndexOutlined,
  ArrowRightOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { TextArea } = Input

interface Relation {
  id: string
  type: string
  source_entity_id: string
  target_entity_id: string
  source_entity_name: string
  target_entity_name: string
  properties: Record<string, any>
  confidence: number
  created_at: string
  updated_at: string
  source: string
  status: 'active' | 'pending' | 'deprecated'
}

const KnowledgeGraphRelationManagement: React.FC = () => {
  const [relations, setRelations] = useState<Relation[]>([])
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false)
  const [selectedRelation, setSelectedRelation] = useState<Relation | null>(null)
  const [form] = Form.useForm()
  const [searchValue, setSearchValue] = useState('')
  const [filterType, setFilterType] = useState<string>('all')

  // 模拟数据
  const mockRelations: Relation[] = [
    {
      id: 'relation_001',
      type: 'WORKS_FOR',
      source_entity_id: 'entity_001',
      target_entity_id: 'entity_002',
      source_entity_name: '张三',
      target_entity_name: '苹果公司',
      properties: {
        position: '高级软件工程师',
        department: '技术部',
        start_date: '2020-06-01',
        salary_range: '20-30万'
      },
      confidence: 0.95,
      created_at: '2025-01-15T10:30:00Z',
      updated_at: '2025-01-20T14:15:00Z',
      source: 'document_001',
      status: 'active'
    },
    {
      id: 'relation_002',
      type: 'LOCATED_IN',
      source_entity_id: 'entity_002',
      target_entity_id: 'entity_003',
      source_entity_name: '苹果公司',
      target_entity_name: '北京',
      properties: {
        address: '朝阳区建国门外大街',
        office_type: '分公司',
        employees: 500
      },
      confidence: 0.92,
      created_at: '2025-01-12T09:20:00Z',
      updated_at: '2025-01-18T16:45:00Z',
      source: 'document_002',
      status: 'active'
    },
    {
      id: 'relation_003',
      type: 'SKILLED_IN',
      source_entity_id: 'entity_001',
      target_entity_id: 'entity_004',
      source_entity_name: '张三',
      target_entity_name: 'Python',
      properties: {
        proficiency: '专家级',
        years_experience: 8,
        certifications: ['Python Institute PCAP']
      },
      confidence: 0.88,
      created_at: '2025-01-10T08:15:00Z',
      updated_at: '2025-01-19T11:30:00Z',
      source: 'document_003',
      status: 'active'
    },
    {
      id: 'relation_004',
      type: 'DEVELOPS',
      source_entity_id: 'entity_002',
      target_entity_id: 'entity_004',
      source_entity_name: '苹果公司',
      target_entity_name: 'Python',
      properties: {
        project_count: 15,
        primary_use: '数据分析和机器学习',
        team_size: 25
      },
      confidence: 0.78,
      created_at: '2025-01-08T14:20:00Z',
      updated_at: '2025-01-17T09:45:00Z',
      source: 'document_004',
      status: 'pending'
    }
  ]

  useEffect(() => {
    loadRelations()
  }, [])

  const loadRelations = async () => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      setRelations(mockRelations)
    } catch (error) {
      message.error('加载关系列表失败')
    } finally {
      setLoading(false)
    }
  }

  const handleCreateRelation = async (values: any) => {
    try {
      // 模拟创建关系API调用
      const newRelation: Relation = {
        id: `relation_${Date.now()}`,
        type: values.type,
        source_entity_id: values.source_entity_id,
        target_entity_id: values.target_entity_id,
        source_entity_name: values.source_entity_name || '实体A',
        target_entity_name: values.target_entity_name || '实体B',
        properties: JSON.parse(values.properties || '{}'),
        confidence: values.confidence / 100,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        source: values.source || 'manual',
        status: 'active'
      }
      
      setRelations([newRelation, ...relations])
      setModalVisible(false)
      form.resetFields()
      message.success('关系创建成功')
    } catch (error) {
      message.error('创建关系失败')
    }
  }

  const handleDeleteRelation = async (relationId: string) => {
    try {
      // 模拟删除API调用
      setRelations(relations.filter(r => r.id !== relationId))
      message.success('关系删除成功')
    } catch (error) {
      message.error('删除关系失败')
    }
  }

  const handleShowDetail = (relation: Relation) => {
    setSelectedRelation(relation)
    setDetailDrawerVisible(true)
  }

  const getTypeColor = (type: string) => {
    const colors = {
      'WORKS_FOR': 'blue',
      'LOCATED_IN': 'green',
      'SKILLED_IN': 'orange',
      'DEVELOPS': 'purple',
      'BELONGS_TO': 'red',
      'PARTNER_OF': 'cyan',
      'SIMILAR_TO': 'gold',
      'PART_OF': 'magenta'
    }
    return colors[type as keyof typeof colors] || 'default'
  }

  const getStatusBadge = (status: string) => {
    const statusMap = {
      'active': { color: 'success', text: '活跃' },
      'pending': { color: 'processing', text: '待处理' },
      'deprecated': { color: 'default', text: '已废弃' }
    }
    const config = statusMap[status as keyof typeof statusMap]
    return <Badge status={config.color as any} text={config.text} />
  }

  const filteredRelations = relations.filter(relation => {
    const searchText = searchValue.toLowerCase()
    const matchesSearch = 
      relation.type.toLowerCase().includes(searchText) ||
      relation.source_entity_name.toLowerCase().includes(searchText) ||
      relation.target_entity_name.toLowerCase().includes(searchText)
    const matchesType = filterType === 'all' || relation.type === filterType
    return matchesSearch && matchesType
  })

  const relationTypes = [...new Set(relations.map(r => r.type))]

  const columns = [
    {
      title: '源实体',
      dataIndex: 'source_entity_name',
      key: 'source_entity_name',
      render: (text: string) => (
        <Tag color="blue" icon={<NodeIndexOutlined />}>{text}</Tag>
      ),
    },
    {
      title: '关系类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string, record: Relation) => (
        <div style={{ textAlign: 'center' }}>
          <Tag color={getTypeColor(type)}>{type}</Tag>
          <div><ArrowRightOutlined style={{ color: '#999' }} /></div>
        </div>
      ),
    },
    {
      title: '目标实体',
      dataIndex: 'target_entity_name',
      key: 'target_entity_name',
      render: (text: string) => (
        <Tag color="green" icon={<NodeIndexOutlined />}>{text}</Tag>
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
      render: (_, record: Relation) => (
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
                  content: `确定要删除关系"${record.source_entity_name} -> ${record.type} -> ${record.target_entity_name}"吗？`,
                  onOk: () => handleDeleteRelation(record.id),
                })
              }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const statistics = {
    total: relations.length,
    active: relations.filter(r => r.status === 'active').length,
    pending: relations.filter(r => r.status === 'pending').length,
    avgConfidence: relations.length > 0 
      ? relations.reduce((sum, r) => sum + r.confidence, 0) / relations.length 
      : 0
  }

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <ShareAltOutlined style={{ marginRight: '8px' }} />
          关系管理
        </Title>
        <Paragraph type="secondary">
          管理知识图谱中实体间的关系，包括创建、编辑、删除和查看关系详情
        </Paragraph>
      </div>

      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总关系数"
              value={statistics.total}
              prefix={<ShareAltOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃关系"
              value={statistics.active}
              valueStyle={{ color: '#3f8600' }}
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="待处理关系"
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
                placeholder="搜索关系类型或实体名称"
                prefix={<SearchOutlined />}
                value={searchValue}
                onChange={(e) => setSearchValue(e.target.value)}
                style={{ width: 280 }}
              />
              <Select
                placeholder="筛选关系类型"
                value={filterType}
                onChange={setFilterType}
                style={{ width: 180 }}
              >
                <Select.Option value="all">全部类型</Select.Option>
                {relationTypes.map(type => (
                  <Select.Option key={type} value={type}>{type}</Select.Option>
                ))}
              </Select>
            </Space>
          </Col>
          <Col>
            <Space>
              <Button icon={<ExportOutlined />}>导出</Button>
              <Button icon={<SyncOutlined />} onClick={loadRelations}>刷新</Button>
              <Button 
                type="primary" 
                icon={<PlusOutlined />}
                onClick={() => setModalVisible(true)}
              >
                新建关系
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 关系列表 */}
      <Card>
        <Table
          columns={columns}
          dataSource={filteredRelations}
          rowKey="id"
          loading={loading}
          pagination={{
            total: filteredRelations.length,
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total) => `共 ${total} 条记录`,
          }}
        />
      </Card>

      {/* 创建关系模态框 */}
      <Modal
        title="创建新关系"
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
          onFinish={handleCreateRelation}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="source_entity_id"
                label="源实体ID"
                rules={[{ required: true, message: '请输入源实体ID' }]}
              >
                <Input placeholder="源实体的唯一标识符" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="source_entity_name"
                label="源实体名称"
                rules={[{ required: true, message: '请输入源实体名称' }]}
              >
                <Input placeholder="源实体的显示名称" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="type"
            label="关系类型"
            rules={[{ required: true, message: '请选择关系类型' }]}
          >
            <Select placeholder="请选择关系类型">
              <Select.Option value="WORKS_FOR">工作于 (WORKS_FOR)</Select.Option>
              <Select.Option value="LOCATED_IN">位于 (LOCATED_IN)</Select.Option>
              <Select.Option value="SKILLED_IN">擅长 (SKILLED_IN)</Select.Option>
              <Select.Option value="DEVELOPS">开发 (DEVELOPS)</Select.Option>
              <Select.Option value="BELONGS_TO">属于 (BELONGS_TO)</Select.Option>
              <Select.Option value="PARTNER_OF">合作伙伴 (PARTNER_OF)</Select.Option>
              <Select.Option value="SIMILAR_TO">相似于 (SIMILAR_TO)</Select.Option>
              <Select.Option value="PART_OF">部分 (PART_OF)</Select.Option>
            </Select>
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="target_entity_id"
                label="目标实体ID"
                rules={[{ required: true, message: '请输入目标实体ID' }]}
              >
                <Input placeholder="目标实体的唯一标识符" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="target_entity_name"
                label="目标实体名称"
                rules={[{ required: true, message: '请输入目标实体名称' }]}
              >
                <Input placeholder="目标实体的显示名称" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="properties"
            label="关系属性 (JSON格式)"
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

          <Row gutter={16}>
            <Col span={12}>
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
            </Col>
            <Col span={12}>
              <Form.Item
                name="source"
                label="数据源"
              >
                <Input placeholder="数据来源标识" />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>

      {/* 关系详情抽屉 */}
      <Drawer
        title="关系详情"
        placement="right"
        width={500}
        open={detailDrawerVisible}
        onClose={() => setDetailDrawerVisible(false)}
      >
        {selectedRelation && (
          <div>
            <div style={{ textAlign: 'center', marginBottom: '24px' }}>
              <Tag color="blue" size="large" icon={<NodeIndexOutlined />}>
                {selectedRelation.source_entity_name}
              </Tag>
              <div style={{ margin: '8px 0' }}>
                <ArrowRightOutlined style={{ fontSize: '16px', color: '#999' }} />
              </div>
              <Tag color={getTypeColor(selectedRelation.type)} size="large">
                {selectedRelation.type}
              </Tag>
              <div style={{ margin: '8px 0' }}>
                <ArrowRightOutlined style={{ fontSize: '16px', color: '#999' }} />
              </div>
              <Tag color="green" size="large" icon={<NodeIndexOutlined />}>
                {selectedRelation.target_entity_name}
              </Tag>
            </div>

            <Divider />

            <div style={{ marginBottom: '16px' }}>
              <Text strong>置信度: </Text>
              <Progress 
                percent={Math.round(selectedRelation.confidence * 100)} 
                size="small"
                strokeColor={selectedRelation.confidence > 0.9 ? '#52c41a' : selectedRelation.confidence > 0.7 ? '#faad14' : '#ff4d4f'}
                style={{ width: '200px' }}
              />
            </div>

            <div style={{ marginBottom: '16px' }}>
              <Text strong>状态: </Text>
              {getStatusBadge(selectedRelation.status)}
            </div>

            <div style={{ marginBottom: '16px' }}>
              <Text strong>源实体ID: </Text>
              <Text code>{selectedRelation.source_entity_id}</Text>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <Text strong>目标实体ID: </Text>
              <Text code>{selectedRelation.target_entity_id}</Text>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <Text strong>创建时间: </Text>
              <Text>{new Date(selectedRelation.created_at).toLocaleString()}</Text>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <Text strong>更新时间: </Text>
              <Text>{new Date(selectedRelation.updated_at).toLocaleString()}</Text>
            </div>

            <div style={{ marginBottom: '16px' }}>
              <Text strong>数据源: </Text>
              <Text>{selectedRelation.source}</Text>
            </div>

            <Divider />

            <Title level={5}>关系属性</Title>
            <Card size="small">
              <pre style={{ whiteSpace: 'pre-wrap', fontSize: '12px' }}>
                {JSON.stringify(selectedRelation.properties, null, 2)}
              </pre>
            </Card>
          </div>
        )}
      </Drawer>
    </div>
  )
}

export default KnowledgeGraphRelationManagement