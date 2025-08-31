import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Table,
  Tag,
  Tabs,
  Progress,
  Statistic,
  List,
  Modal,
  Form,
  Input,
  Select,
  DatePicker,
  Switch,
  message,
  Timeline,
  Alert,
  Popconfirm,
  Drawer,
  Descriptions,
  Badge
} from 'antd'
import {
  BarChartOutlined,
  SyncOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  ReloadOutlined,
  SettingOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  ExportOutlined,
  ImportOutlined,
  BellOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined
} from '@ant-design/icons'

const { Title, Text } = Typography
const { RangePicker } = DatePicker
const { Option } = Select

interface UpdateTask {
  id: string
  name: string
  type: string
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused'
  progress: number
  changes: number
  startTime?: string
  endTime?: string
  description?: string
  source?: string
  target?: string
}

interface IncrementalUpdate {
  id: string
  timestamp: string
  type: 'entity' | 'relation' | 'property'
  operation: 'create' | 'update' | 'delete'
  entityId?: string
  relationId?: string
  oldValue?: any
  newValue?: any
  status: 'success' | 'failed'
}

const KnowledgeGraphAnalysis: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [modalVisible, setModalVisible] = useState(false)
  const [drawerVisible, setDrawerVisible] = useState(false)
  const [selectedTask, setSelectedTask] = useState<UpdateTask | null>(null)
  const [form] = Form.useForm()

  // 模拟更新任务数据
  const updateTasks: UpdateTask[] = [
    {
      id: '1',
      name: '实体属性批量更新',
      type: '实体',
      status: 'completed',
      progress: 100,
      changes: 156,
      startTime: '2025-01-22 09:00:00',
      endTime: '2025-01-22 09:15:00',
      description: '批量更新Person实体的职业信息',
      source: '外部数据源A',
      target: 'Neo4j图数据库'
    },
    {
      id: '2',
      name: '关系增量同步',
      type: '关系',
      status: 'running',
      progress: 65,
      changes: 89,
      startTime: '2025-01-22 10:00:00',
      description: '同步WORKS_AT关系的变更',
      source: 'MySQL数据库',
      target: 'Neo4j图数据库'
    },
    {
      id: '3',
      name: '新增实体导入',
      type: '实体',
      status: 'pending',
      progress: 0,
      changes: 0,
      description: '导入新的公司实体数据',
      source: 'CSV文件',
      target: 'Neo4j图数据库'
    },
    {
      id: '4',
      name: '属性值校验更新',
      type: '属性',
      status: 'failed',
      progress: 25,
      changes: 12,
      startTime: '2025-01-22 08:30:00',
      description: '验证并更新实体属性的数据质量',
      source: '数据质量系统',
      target: 'Neo4j图数据库'
    }
  ]

  // 模拟增量更新记录
  const incrementalUpdates: IncrementalUpdate[] = [
    {
      id: '1',
      timestamp: '2025-01-22 10:30:15',
      type: 'entity',
      operation: 'update',
      entityId: 'person_001',
      oldValue: { name: '张三', title: '工程师' },
      newValue: { name: '张三', title: '高级工程师' },
      status: 'success'
    },
    {
      id: '2',
      timestamp: '2025-01-22 10:29:45',
      type: 'relation',
      operation: 'create',
      relationId: 'works_at_005',
      newValue: { from: 'person_002', to: 'company_003', type: 'WORKS_AT' },
      status: 'success'
    },
    {
      id: '3',
      timestamp: '2025-01-22 10:28:30',
      type: 'entity',
      operation: 'delete',
      entityId: 'person_999',
      oldValue: { name: '测试用户', title: '临时' },
      status: 'failed'
    }
  ]

  const statusColors = {
    pending: 'default',
    running: 'processing',
    completed: 'success',
    failed: 'error',
    paused: 'warning'
  }

  const statusText = {
    pending: '待运行',
    running: '运行中',
    completed: '已完成',
    failed: '失败',
    paused: '已暂停'
  }

  const taskColumns = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      render: (text: string, record: UpdateTask) => (
        <Space>
          <Text strong>{text}</Text>
          {record.status === 'running' && <SyncOutlined spin />}
        </Space>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={type === '实体' ? 'blue' : type === '关系' ? 'green' : 'orange'}>
          {type}
        </Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: keyof typeof statusColors) => (
        <Badge status={statusColors[status] as any} text={statusText[status]} />
      )
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      width: 120,
      render: (progress: number, record: UpdateTask) => (
        <Progress
          percent={progress}
          size="small"
          status={record.status === 'failed' ? 'exception' : undefined}
        />
      )
    },
    {
      title: '变更数',
      dataIndex: 'changes',
      key: 'changes',
      render: (changes: number) => (
        <Text type={changes > 0 ? 'success' : undefined}>{changes}</Text>
      )
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: UpdateTask) => (
        <Space size="small">
          <Button
            size="small"
            icon={<EyeOutlined />}
            onClick={() => {
              setSelectedTask(record)
              setDrawerVisible(true)
            }}
          />
          {record.status === 'running' && (
            <Popconfirm title="确定要暂停此任务吗？" onConfirm={() => handlePauseTask(record.id)}>
              <Button size="small" icon={<PauseCircleOutlined />} />
            </Popconfirm>
          )}
          {(record.status === 'pending' || record.status === 'paused') && (
            <Button
              size="small"
              icon={<PlayCircleOutlined />}
              type="primary"
              onClick={() => handleStartTask(record.id)}
            />
          )}
          {record.status === 'failed' && (
            <Button
              size="small"
              icon={<ReloadOutlined />}
              onClick={() => handleRetryTask(record.id)}
            />
          )}
        </Space>
      )
    }
  ]

  const handleStartTask = (taskId: string) => {
    message.success(`任务 ${taskId} 已开始运行`)
  }

  const handlePauseTask = (taskId: string) => {
    message.info(`任务 ${taskId} 已暂停`)
  }

  const handleRetryTask = (taskId: string) => {
    message.info(`任务 ${taskId} 正在重试`)
  }

  const handleCreateTask = () => {
    form.validateFields().then(values => {
      console.log('创建任务:', values)
      message.success('任务创建成功')
      setModalVisible(false)
      form.resetFields()
    })
  }

  const loadData = async () => {
    setLoading(true)
    try {
      await new Promise(resolve => setTimeout(resolve, 1000))
      message.success('数据刷新成功')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  const tabItems = [
    {
      key: 'tasks',
      label: '更新任务',
      children: (
        <div>
          <Card title="增量更新任务" extra={
            <Space>
              <Button icon={<ImportOutlined />}>批量导入</Button>
              <Button icon={<ExportOutlined />}>导出配置</Button>
              <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
                新建任务
              </Button>
            </Space>
          }>
            <Table
              columns={taskColumns}
              dataSource={updateTasks}
              loading={loading}
              pagination={{
                pageSize: 10,
                showTotal: (total) => `共 ${total} 个任务`
              }}
            />
          </Card>
        </div>
      )
    },
    {
      key: 'incremental',
      label: '增量更新记录',
      children: (
        <Card title="实时更新记录">
          <Timeline>
            {incrementalUpdates.map(update => (
              <Timeline.Item
                key={update.id}
                color={update.status === 'success' ? 'green' : 'red'}
                dot={update.status === 'success' ? <SyncOutlined /> : <StopOutlined />}
              >
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                    <Text strong>{update.timestamp}</Text>
                    <Tag color={update.type === 'entity' ? 'blue' : update.type === 'relation' ? 'green' : 'orange'}>
                      {update.type}
                    </Tag>
                    <Tag color={update.operation === 'create' ? 'green' : update.operation === 'update' ? 'orange' : 'red'}>
                      {update.operation.toUpperCase()}
                    </Tag>
                    <Badge status={update.status === 'success' ? 'success' : 'error'} text={update.status} />
                  </div>
                  <div style={{ fontSize: '12px', color: '#666' }}>
                    {update.entityId && `实体ID: ${update.entityId}`}
                    {update.relationId && `关系ID: ${update.relationId}`}
                  </div>
                </div>
              </Timeline.Item>
            ))}
          </Timeline>
        </Card>
      )
    },
    {
      key: 'monitoring',
      label: '同步监控',
      children: (
        <Row gutter={16}>
          <Col span={12}>
            <Card title="同步状态监控">
              <Space direction="vertical" style={{ width: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Text>数据源连接状态</Text>
                  <Badge status="success" text="正常" />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Text>图数据库连接状态</Text>
                  <Badge status="success" text="正常" />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Text>同步队列状态</Text>
                  <Badge status="processing" text="运行中" />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Text>错误队列</Text>
                  <Badge status="error" text="2个待处理" />
                </div>
              </Space>
            </Card>
          </Col>
          <Col span={12}>
            <Card title="性能指标">
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic title="今日处理量" value={1250} suffix="条" />
                </Col>
                <Col span={12}>
                  <Statistic title="平均处理时间" value={125} suffix="ms" />
                </Col>
                <Col span={12} style={{ marginTop: 16 }}>
                  <Statistic title="成功率" value={98.5} suffix="%" precision={1} />
                </Col>
                <Col span={12} style={{ marginTop: 16 }}>
                  <Statistic title="错误数" value={18} valueStyle={{ color: '#cf1322' }} />
                </Col>
              </Row>
            </Card>
          </Col>
        </Row>
      )
    }
  ]

  return (
    <div style={{ padding: '24px' }}>
      <Card>
        <Row justify="space-between" align="middle" style={{ marginBottom: '24px' }}>
          <Col>
            <Space>
              <BarChartOutlined style={{ fontSize: '24px' }} />
              <Title level={2} style={{ margin: 0 }}>图分析工具</Title>
              <Text type="secondary">知识图谱增量更新和实时同步管理工具</Text>
            </Space>
          </Col>
          <Col>
            <Space>
              <Button icon={<SettingOutlined />}>同步设置</Button>
              <Button icon={<BellOutlined />}>告警配置</Button>
              <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>
                刷新状态
              </Button>
            </Space>
          </Col>
        </Row>

        {/* 告警信息 */}
        <Alert
          message="系统提醒"
          description="有2个任务执行失败，请及时处理。同步队列中有18个错误记录待处理。"
          type="warning"
          showIcon
          closable
          style={{ marginBottom: '24px' }}
        />

        {/* 统计信息 */}
        <Row gutter={16} style={{ marginBottom: '24px' }}>
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="总更新任务"
                value={updateTasks.length}
                prefix={<ClockCircleOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="运行中任务"
                value={updateTasks.filter(t => t.status === 'running').length}
                prefix={<SyncOutlined />}
                valueStyle={{ color: '#52c41a' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="已完成任务"
                value={updateTasks.filter(t => t.status === 'completed').length}
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#fa8c16' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card size="small">
              <Statistic
                title="今日变更数"
                value={updateTasks.reduce((sum, task) => sum + task.changes, 0)}
                prefix={<EditOutlined />}
                valueStyle={{ color: '#722ed1' }}
              />
            </Card>
          </Col>
        </Row>
      </Card>

      <Tabs items={tabItems} defaultActiveKey="tasks" />

      {/* 创建任务模态框 */}
      <Modal
        title="创建增量更新任务"
        open={modalVisible}
        onOk={handleCreateTask}
        onCancel={() => {
          setModalVisible(false)
          form.resetFields()
        }}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item name="name" label="任务名称" rules={[{ required: true }]}>
            <Input placeholder="请输入任务名称" />
          </Form.Item>
          <Form.Item name="type" label="任务类型" rules={[{ required: true }]}>
            <Select placeholder="请选择任务类型">
              <Option value="entity">实体更新</Option>
              <Option value="relation">关系更新</Option>
              <Option value="property">属性更新</Option>
            </Select>
          </Form.Item>
          <Form.Item name="source" label="数据源" rules={[{ required: true }]}>
            <Input placeholder="请输入数据源" />
          </Form.Item>
          <Form.Item name="description" label="任务描述">
            <Input.TextArea rows={3} placeholder="请描述任务内容" />
          </Form.Item>
          <Form.Item name="schedule" label="调度设置">
            <Space>
              <Switch />
              <Text>启用自动调度</Text>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 任务详情抽屉 */}
      <Drawer
        title="任务详情"
        placement="right"
        onClose={() => setDrawerVisible(false)}
        open={drawerVisible}
        width={500}
      >
        {selectedTask && (
          <div>
            <Descriptions column={1} bordered size="small">
              <Descriptions.Item label="任务名称">{selectedTask.name}</Descriptions.Item>
              <Descriptions.Item label="任务类型">
                <Tag color={selectedTask.type === '实体' ? 'blue' : 'green'}>
                  {selectedTask.type}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                <Badge 
                  status={statusColors[selectedTask.status] as any} 
                  text={statusText[selectedTask.status]} 
                />
              </Descriptions.Item>
              <Descriptions.Item label="进度">
                <Progress percent={selectedTask.progress} size="small" />
              </Descriptions.Item>
              <Descriptions.Item label="变更数量">{selectedTask.changes}</Descriptions.Item>
              <Descriptions.Item label="数据源">{selectedTask.source}</Descriptions.Item>
              <Descriptions.Item label="目标">{selectedTask.target}</Descriptions.Item>
              <Descriptions.Item label="开始时间">{selectedTask.startTime || '-'}</Descriptions.Item>
              <Descriptions.Item label="结束时间">{selectedTask.endTime || '-'}</Descriptions.Item>
              <Descriptions.Item label="任务描述">{selectedTask.description}</Descriptions.Item>
            </Descriptions>

            <div style={{ marginTop: '16px' }}>
              <Space>
                {selectedTask.status === 'running' && (
                  <Button icon={<PauseCircleOutlined />} onClick={() => handlePauseTask(selectedTask.id)}>
                    暂停任务
                  </Button>
                )}
                {(selectedTask.status === 'pending' || selectedTask.status === 'paused') && (
                  <Button type="primary" icon={<PlayCircleOutlined />} onClick={() => handleStartTask(selectedTask.id)}>
                    启动任务
                  </Button>
                )}
                {selectedTask.status === 'failed' && (
                  <Button icon={<ReloadOutlined />} onClick={() => handleRetryTask(selectedTask.id)}>
                    重试任务
                  </Button>
                )}
                <Button icon={<EditOutlined />}>编辑任务</Button>
                <Button icon={<DeleteOutlined />} danger>删除任务</Button>
              </Space>
            </div>
          </div>
        )}
      </Drawer>
    </div>
  )
}

export default KnowledgeGraphAnalysis