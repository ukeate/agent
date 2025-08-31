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
  Progress,
  Tag,
  Typography,
  Divider,
  Select,
  Input,
  Upload,
  Modal,
  Form,
  message,
  Tabs,
  Steps,
  Descriptions,
  Timeline,
  Badge,
  List,
  Collapse,
  Switch,
  DatePicker
} from 'antd'
import {
  CloudUploadOutlined,
  CloudDownloadOutlined,
  SwapOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  UploadOutlined,
  DownloadOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  ReloadOutlined,
  SettingOutlined,
  InfoCircleOutlined,
  FolderOpenOutlined,
  FileSyncOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { Dragger } = Upload
const { Step } = Steps
const { TabPane } = Tabs
const { Panel } = Collapse
const { RangePicker } = DatePicker
const { TextArea } = Input

interface MigrationJob {
  id: string
  name: string
  type: 'import' | 'export' | 'migrate'
  source: string
  target: string
  format: 'csv' | 'json' | 'xml' | 'cypher' | 'graphml'
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused'
  progress: number
  created_at: string
  started_at?: string
  completed_at?: string
  file_path?: string
  file_size?: number
  records_total: number
  records_processed: number
  errors_count: number
  mapping_config?: any
}

interface DataSource {
  id: string
  name: string
  type: 'file' | 'database' | 'api' | 'graph'
  connection_string: string
  format: string
  status: 'connected' | 'disconnected' | 'error'
  last_sync: string
  size: number
}

interface MigrationTemplate {
  id: string
  name: string
  description: string
  source_format: string
  target_format: string
  mapping_rules: any
  validation_rules: string[]
  created_at: string
  usage_count: number
}

interface ValidationResult {
  field: string
  rule: string
  errors: number
  warnings: number
  status: 'pass' | 'warning' | 'error'
  sample_errors: string[]
}

const KnowledgeGraphDataMigration: React.FC = () => {
  const [migrationJobs, setMigrationJobs] = useState<MigrationJob[]>([])
  const [dataSources, setDataSources] = useState<DataSource[]>([])
  const [templates, setTemplates] = useState<MigrationTemplate[]>([])
  const [validationResults, setValidationResults] = useState<ValidationResult[]>([])
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('jobs')
  const [modalVisible, setModalVisible] = useState(false)
  const [modalType, setModalType] = useState<'import' | 'export' | 'migrate'>('import')
  const [currentStep, setCurrentStep] = useState(0)
  const [form] = Form.useForm()
  const [selectedJob, setSelectedJob] = useState<MigrationJob | null>(null)
  const [detailModalVisible, setDetailModalVisible] = useState(false)

  // 模拟迁移任务数据
  const mockJobs: MigrationJob[] = [
    {
      id: 'job_001',
      name: '员工数据导入',
      type: 'import',
      source: 'employees.csv',
      target: 'Person节点',
      format: 'csv',
      status: 'completed',
      progress: 100,
      created_at: '2025-01-22T10:30:00Z',
      started_at: '2025-01-22T10:31:00Z',
      completed_at: '2025-01-22T10:45:00Z',
      file_path: '/uploads/employees.csv',
      file_size: 2048576,
      records_total: 12500,
      records_processed: 12500,
      errors_count: 23
    },
    {
      id: 'job_002',
      name: '组织架构导出',
      type: 'export',
      source: 'Organization节点',
      target: 'organizations.json',
      format: 'json',
      status: 'running',
      progress: 65,
      created_at: '2025-01-22T14:00:00Z',
      started_at: '2025-01-22T14:01:00Z',
      records_total: 3200,
      records_processed: 2080,
      errors_count: 5
    },
    {
      id: 'job_003',
      name: '历史数据迁移',
      type: 'migrate',
      source: 'Legacy DB',
      target: 'Neo4j',
      format: 'cypher',
      status: 'failed',
      progress: 25,
      created_at: '2025-01-22T09:00:00Z',
      started_at: '2025-01-22T09:01:00Z',
      completed_at: '2025-01-22T09:15:00Z',
      records_total: 50000,
      records_processed: 12500,
      errors_count: 2156
    }
  ]

  // 模拟数据源数据
  const mockDataSources: DataSource[] = [
    {
      id: 'source_001',
      name: 'MySQL员工数据库',
      type: 'database',
      connection_string: 'mysql://user:pass@localhost:3306/employees',
      format: 'SQL',
      status: 'connected',
      last_sync: '2025-01-22T14:30:00Z',
      size: 156700000
    },
    {
      id: 'source_002',
      name: 'CSV文件批次',
      type: 'file',
      connection_string: '/data/csv_files/',
      format: 'CSV',
      status: 'connected',
      last_sync: '2025-01-22T12:00:00Z',
      size: 45600000
    },
    {
      id: 'source_003',
      name: 'Legacy图数据库',
      type: 'graph',
      connection_string: 'bolt://legacy:7687',
      format: 'Cypher',
      status: 'error',
      last_sync: '2025-01-21T16:00:00Z',
      size: 890000000
    }
  ]

  // 模拟迁移模板数据
  const mockTemplates: MigrationTemplate[] = [
    {
      id: 'template_001',
      name: 'CSV员工数据导入',
      description: '从CSV文件导入员工信息到Person节点',
      source_format: 'CSV',
      target_format: 'Neo4j',
      mapping_rules: {
        'name': 'name',
        'email': 'email',
        'department': 'department',
        'position': 'position'
      },
      validation_rules: ['非空验证', '邮箱格式验证', '重复性检查'],
      created_at: '2025-01-15T10:00:00Z',
      usage_count: 15
    },
    {
      id: 'template_002',
      name: 'JSON组织数据导入',
      description: '从JSON文件导入组织架构信息',
      source_format: 'JSON',
      target_format: 'Neo4j',
      mapping_rules: {
        'org_id': 'id',
        'org_name': 'name',
        'industry': 'industry',
        'parent_id': 'parent_organization'
      },
      validation_rules: ['必需字段检查', '层级关系验证'],
      created_at: '2025-01-16T14:00:00Z',
      usage_count: 8
    }
  ]

  // 模拟验证结果数据
  const mockValidationResults: ValidationResult[] = [
    {
      field: 'email',
      rule: '邮箱格式验证',
      errors: 15,
      warnings: 3,
      status: 'error',
      sample_errors: ['invalid_email@', 'test@.com', 'user@domain']
    },
    {
      field: 'name',
      rule: '非空验证',
      errors: 0,
      warnings: 0,
      status: 'pass',
      sample_errors: []
    },
    {
      field: 'department',
      rule: '重复性检查',
      errors: 0,
      warnings: 12,
      status: 'warning',
      sample_errors: ['多个"技术部"变体', '部门名称不统一']
    }
  ]

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      setMigrationJobs(mockJobs)
      setDataSources(mockDataSources)
      setTemplates(mockTemplates)
      setValidationResults(mockValidationResults)
    } catch (error) {
      message.error('加载数据失败')
    } finally {
      setLoading(false)
    }
  }

  const startMigrationJob = async (jobId: string) => {
    try {
      const updatedJobs = migrationJobs.map(job => 
        job.id === jobId 
          ? { ...job, status: 'running' as const, started_at: new Date().toISOString() }
          : job
      )
      setMigrationJobs(updatedJobs)
      message.success('迁移任务已启动')
    } catch (error) {
      message.error('启动任务失败')
    }
  }

  const pauseMigrationJob = async (jobId: string) => {
    try {
      const updatedJobs = migrationJobs.map(job => 
        job.id === jobId ? { ...job, status: 'paused' as const } : job
      )
      setMigrationJobs(updatedJobs)
      message.success('任务已暂停')
    } catch (error) {
      message.error('暂停任务失败')
    }
  }

  const deleteMigrationJob = async (jobId: string) => {
    try {
      const updatedJobs = migrationJobs.filter(job => job.id !== jobId)
      setMigrationJobs(updatedJobs)
      message.success('任务已删除')
    } catch (error) {
      message.error('删除任务失败')
    }
  }

  const validateData = async () => {
    setLoading(true)
    try {
      await new Promise(resolve => setTimeout(resolve, 2000))
      message.success('数据验证完成')
    } catch (error) {
      message.error('数据验证失败')
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = (info: any) => {
    if (info.file.status === 'done') {
      message.success(`${info.file.name} 文件上传成功`)
    } else if (info.file.status === 'error') {
      message.error(`${info.file.name} 文件上传失败`)
    }
  }

  const createMigrationJob = async (values: any) => {
    try {
      const newJob: MigrationJob = {
        id: `job_${Date.now()}`,
        name: values.name,
        type: modalType,
        source: values.source,
        target: values.target,
        format: values.format,
        status: 'pending',
        progress: 0,
        created_at: new Date().toISOString(),
        records_total: 0,
        records_processed: 0,
        errors_count: 0
      }
      setMigrationJobs([newJob, ...migrationJobs])
      setModalVisible(false)
      form.resetFields()
      setCurrentStep(0)
      message.success('迁移任务创建成功')
    } catch (error) {
      message.error('创建任务失败')
    }
  }

  const openCreateModal = (type: 'import' | 'export' | 'migrate') => {
    setModalType(type)
    setModalVisible(true)
    setCurrentStep(0)
    form.resetFields()
  }

  const getStatusBadge = (status: string) => {
    const statusMap = {
      'pending': { color: 'default', text: '等待中' },
      'running': { color: 'processing', text: '运行中' },
      'completed': { color: 'success', text: '已完成' },
      'failed': { color: 'error', text: '失败' },
      'paused': { color: 'warning', text: '已暂停' }
    }
    const config = statusMap[status as keyof typeof statusMap]
    return <Badge status={config.color as any} text={config.text} />
  }

  const getTypeColor = (type: string) => {
    const colors = {
      'import': 'blue',
      'export': 'green',
      'migrate': 'orange'
    }
    return colors[type as keyof typeof colors] || 'default'
  }

  const getTypeName = (type: string) => {
    const names = {
      'import': '导入',
      'export': '导出',
      'migrate': '迁移'
    }
    return names[type as keyof typeof names] || type
  }

  const getValidationStatusColor = (status: string) => {
    const colors = {
      'pass': 'green',
      'warning': 'orange',
      'error': 'red'
    }
    return colors[status as keyof typeof colors] || 'default'
  }

  const jobColumns = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: MigrationJob) => (
        <Button 
          type="link" 
          onClick={() => {
            setSelectedJob(record)
            setDetailModalVisible(true)
          }}
          style={{ padding: 0 }}
        >
          {text}
        </Button>
      ),
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={getTypeColor(type)}>
          {getTypeName(type)}
        </Tag>
      ),
    },
    {
      title: '数据源',
      dataIndex: 'source',
      key: 'source',
    },
    {
      title: '目标',
      dataIndex: 'target',
      key: 'target',
    },
    {
      title: '格式',
      dataIndex: 'format',
      key: 'format',
      render: (format: string) => <Tag>{format.toUpperCase()}</Tag>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusBadge(status),
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number, record: MigrationJob) => (
        <div>
          <Progress percent={progress} size="small" />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.records_processed.toLocaleString()} / {record.records_total.toLocaleString()}
          </Text>
        </div>
      ),
    },
    {
      title: '错误数',
      dataIndex: 'errors_count',
      key: 'errors_count',
      render: (count: number) => (
        <Text style={{ color: count > 0 ? '#ff4d4f' : '#52c41a' }}>
          {count.toLocaleString()}
        </Text>
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
      render: (_, record: MigrationJob) => (
        <Space>
          {record.status === 'pending' && (
            <Tooltip title="启动">
              <Button 
                type="text" 
                icon={<PlayCircleOutlined />}
                onClick={() => startMigrationJob(record.id)}
              />
            </Tooltip>
          )}
          {record.status === 'running' && (
            <Tooltip title="暂停">
              <Button 
                type="text" 
                icon={<PauseCircleOutlined />}
                onClick={() => pauseMigrationJob(record.id)}
              />
            </Tooltip>
          )}
          {record.status === 'paused' && (
            <Tooltip title="继续">
              <Button 
                type="text" 
                icon={<PlayCircleOutlined />}
                onClick={() => startMigrationJob(record.id)}
              />
            </Tooltip>
          )}
          <Tooltip title="详情">
            <Button 
              type="text" 
              icon={<InfoCircleOutlined />}
              onClick={() => {
                setSelectedJob(record)
                setDetailModalVisible(true)
              }}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button 
              type="text" 
              danger 
              icon={<CloseCircleOutlined />}
              onClick={() => {
                Modal.confirm({
                  title: '确认删除',
                  content: `确定要删除任务"${record.name}"吗？`,
                  onOk: () => deleteMigrationJob(record.id),
                })
              }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const sourceColumns = [
    {
      title: '数据源名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => {
        const colors = { 'file': 'blue', 'database': 'green', 'api': 'orange', 'graph': 'purple' }
        return <Tag color={colors[type as keyof typeof colors]}>{type.toUpperCase()}</Tag>
      },
    },
    {
      title: '连接信息',
      dataIndex: 'connection_string',
      key: 'connection_string',
      render: (text: string) => (
        <Text ellipsis={{ tooltip: text }} style={{ maxWidth: '200px' }}>
          {text}
        </Text>
      ),
    },
    {
      title: '格式',
      dataIndex: 'format',
      key: 'format',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = { 'connected': 'green', 'disconnected': 'orange', 'error': 'red' }
        const names = { 'connected': '已连接', 'disconnected': '未连接', 'error': '错误' }
        return <Tag color={colors[status as keyof typeof colors]}>{names[status as keyof typeof names]}</Tag>
      },
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => `${(size / 1024 / 1024).toFixed(1)} MB`,
    },
    {
      title: '最后同步',
      dataIndex: 'last_sync',
      key: 'last_sync',
      render: (time: string) => new Date(time).toLocaleString(),
    },
  ]

  const runningJobs = migrationJobs.filter(job => job.status === 'running')
  const completedJobs = migrationJobs.filter(job => job.status === 'completed')
  const failedJobs = migrationJobs.filter(job => job.status === 'failed')

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <SwapOutlined style={{ marginRight: '8px' }} />
          数据迁移工具
        </Title>
        <Paragraph type="secondary">
          管理知识图谱的数据导入、导出和迁移任务
        </Paragraph>
      </div>

      {/* 统计概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总任务数"
              value={migrationJobs.length}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中任务"
              value={runningJobs.length}
              valueStyle={{ color: '#1890ff' }}
              prefix={<PlayCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已完成任务"
              value={completedJobs.length}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="失败任务"
              value={failedJobs.length}
              valueStyle={{ color: '#ff4d4f' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 操作栏 */}
      <Card style={{ marginBottom: '16px' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Button 
                type="primary" 
                icon={<CloudUploadOutlined />}
                onClick={() => openCreateModal('import')}
              >
                数据导入
              </Button>
              <Button 
                icon={<CloudDownloadOutlined />}
                onClick={() => openCreateModal('export')}
              >
                数据导出
              </Button>
              <Button 
                icon={<SwapOutlined />}
                onClick={() => openCreateModal('migrate')}
              >
                数据迁移
              </Button>
            </Space>
          </Col>
          <Col>
            <Space>
              <Button icon={<SettingOutlined />}>配置</Button>
              <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>
                刷新
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="迁移任务" key="jobs">
          <Card title="迁移任务列表">
            <Table
              columns={jobColumns}
              dataSource={migrationJobs}
              rowKey="id"
              loading={loading}
              pagination={{
                pageSize: 10,
                showTotal: (total) => `共 ${total} 个任务`
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="数据源管理" key="sources">
          <Card title="数据源配置" extra={
            <Button type="primary" icon={<PlusOutlined />}>
              添加数据源
            </Button>
          }>
            <Table
              columns={sourceColumns}
              dataSource={dataSources}
              rowKey="id"
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="迁移模板" key="templates">
          <Card title="迁移模板">
            <List
              dataSource={templates}
              renderItem={(template) => (
                <List.Item
                  actions={[
                    <Button type="link">使用模板</Button>,
                    <Button type="link">编辑</Button>,
                    <Button type="link" danger>删除</Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={<FileTextOutlined style={{ fontSize: '24px' }} />}
                    title={
                      <Space>
                        <Text strong>{template.name}</Text>
                        <Tag color="blue">{template.source_format} → {template.target_format}</Tag>
                        <Badge count={template.usage_count} showZero={false} />
                      </Space>
                    }
                    description={
                      <div>
                        <Paragraph style={{ margin: 0 }}>{template.description}</Paragraph>
                        <Text type="secondary">
                          验证规则: {template.validation_rules.join(', ')} | 
                          创建时间: {new Date(template.created_at).toLocaleDateString()}
                        </Text>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>

        <TabPane tab="数据验证" key="validation">
          <Card title="数据验证结果" extra={
            <Button 
              type="primary" 
              icon={<ThunderboltOutlined />}
              onClick={validateData}
              loading={loading}
            >
              运行验证
            </Button>
          }>
            <Table
              dataSource={validationResults}
              rowKey="field"
              columns={[
                {
                  title: '字段',
                  dataIndex: 'field',
                  key: 'field',
                },
                {
                  title: '验证规则',
                  dataIndex: 'rule',
                  key: 'rule',
                },
                {
                  title: '状态',
                  dataIndex: 'status',
                  key: 'status',
                  render: (status: string) => (
                    <Tag color={getValidationStatusColor(status)}>
                      {status === 'pass' ? '通过' : status === 'warning' ? '警告' : '错误'}
                    </Tag>
                  ),
                },
                {
                  title: '错误数',
                  dataIndex: 'errors',
                  key: 'errors',
                  render: (count: number) => (
                    <Text style={{ color: count > 0 ? '#ff4d4f' : '#52c41a' }}>
                      {count}
                    </Text>
                  ),
                },
                {
                  title: '警告数',
                  dataIndex: 'warnings',
                  key: 'warnings',
                  render: (count: number) => (
                    <Text style={{ color: count > 0 ? '#faad14' : '#52c41a' }}>
                      {count}
                    </Text>
                  ),
                },
              ]}
              expandable={{
                expandedRowRender: (record) => (
                  record.sample_errors.length > 0 && (
                    <div>
                      <Text strong>错误示例:</Text>
                      <ul>
                        {record.sample_errors.map((error, index) => (
                          <li key={index}><Text type="secondary">{error}</Text></li>
                        ))}
                      </ul>
                    </div>
                  )
                ),
              }}
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* 创建任务模态框 */}
      <Modal
        title={`${modalType === 'import' ? '数据导入' : modalType === 'export' ? '数据导出' : '数据迁移'}`}
        open={modalVisible}
        onCancel={() => {
          setModalVisible(false)
          setCurrentStep(0)
          form.resetFields()
        }}
        footer={null}
        width={800}
      >
        <Steps current={currentStep} style={{ marginBottom: '24px' }}>
          <Step title="基本配置" />
          <Step title="映射设置" />
          <Step title="验证配置" />
          <Step title="确认执行" />
        </Steps>

        <Form
          form={form}
          layout="vertical"
          onFinish={createMigrationJob}
        >
          {currentStep === 0 && (
            <>
              <Form.Item
                name="name"
                label="任务名称"
                rules={[{ required: true, message: '请输入任务名称' }]}
              >
                <Input placeholder="输入任务名称" />
              </Form.Item>
              
              {modalType === 'import' && (
                <>
                  <Form.Item label="上传文件">
                    <Dragger
                      name="file"
                      multiple={false}
                      onChange={handleFileUpload}
                      showUploadList={false}
                    >
                      <p className="ant-upload-drag-icon">
                        <UploadOutlined />
                      </p>
                      <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
                      <p className="ant-upload-hint">
                        支持 CSV, JSON, XML, GraphML 格式
                      </p>
                    </Dragger>
                  </Form.Item>
                  
                  <Form.Item
                    name="target"
                    label="目标节点类型"
                    rules={[{ required: true, message: '请选择目标节点类型' }]}
                  >
                    <Select placeholder="选择目标节点类型">
                      <Select.Option value="Person">Person</Select.Option>
                      <Select.Option value="Organization">Organization</Select.Option>
                      <Select.Option value="Location">Location</Select.Option>
                    </Select>
                  </Form.Item>
                </>
              )}
              
              {modalType === 'export' && (
                <>
                  <Form.Item
                    name="source"
                    label="数据源"
                    rules={[{ required: true, message: '请选择数据源' }]}
                  >
                    <Select placeholder="选择要导出的数据源">
                      <Select.Option value="Person">Person节点</Select.Option>
                      <Select.Option value="Organization">Organization节点</Select.Option>
                      <Select.Option value="全部数据">全部数据</Select.Option>
                    </Select>
                  </Form.Item>
                  
                  <Form.Item
                    name="format"
                    label="导出格式"
                    rules={[{ required: true, message: '请选择导出格式' }]}
                  >
                    <Select placeholder="选择导出格式">
                      <Select.Option value="csv">CSV</Select.Option>
                      <Select.Option value="json">JSON</Select.Option>
                      <Select.Option value="xml">XML</Select.Option>
                      <Select.Option value="cypher">Cypher</Select.Option>
                      <Select.Option value="graphml">GraphML</Select.Option>
                    </Select>
                  </Form.Item>
                </>
              )}
              
              {modalType === 'migrate' && (
                <>
                  <Form.Item
                    name="source"
                    label="源数据库"
                    rules={[{ required: true, message: '请选择源数据库' }]}
                  >
                    <Select placeholder="选择源数据库">
                      {dataSources.map(source => (
                        <Select.Option key={source.id} value={source.id}>
                          {source.name}
                        </Select.Option>
                      ))}
                    </Select>
                  </Form.Item>
                  
                  <Form.Item
                    name="target"
                    label="目标数据库"
                    rules={[{ required: true, message: '请输入目标数据库' }]}
                  >
                    <Input placeholder="目标数据库连接信息" />
                  </Form.Item>
                </>
              )}
            </>
          )}

          {currentStep === 1 && (
            <div>
              <Title level={4}>字段映射配置</Title>
              <TextArea
                rows={8}
                placeholder="配置字段映射规则 (JSON格式)"
                defaultValue={JSON.stringify({
                  "name": "name",
                  "email": "email",
                  "department": "department"
                }, null, 2)}
              />
            </div>
          )}

          {currentStep === 2 && (
            <div>
              <Title level={4}>数据验证配置</Title>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text>选择要应用的验证规则:</Text>
                <Switch defaultChecked /> <Text>非空字段验证</Text><br />
                <Switch defaultChecked /> <Text>数据格式验证</Text><br />
                <Switch /> <Text>重复性检查</Text><br />
                <Switch /> <Text>引用完整性验证</Text>
              </Space>
            </div>
          )}

          {currentStep === 3 && (
            <div>
              <Title level={4}>任务确认</Title>
              <Descriptions bordered column={1}>
                <Descriptions.Item label="任务类型">
                  {getTypeName(modalType)}
                </Descriptions.Item>
                <Descriptions.Item label="任务名称">
                  {form.getFieldValue('name')}
                </Descriptions.Item>
                <Descriptions.Item label="数据源">
                  {form.getFieldValue('source')}
                </Descriptions.Item>
                <Descriptions.Item label="目标">
                  {form.getFieldValue('target')}
                </Descriptions.Item>
              </Descriptions>
            </div>
          )}

          <div style={{ marginTop: '24px', textAlign: 'right' }}>
            <Space>
              {currentStep > 0 && (
                <Button onClick={() => setCurrentStep(currentStep - 1)}>
                  上一步
                </Button>
              )}
              {currentStep < 3 && (
                <Button type="primary" onClick={() => setCurrentStep(currentStep + 1)}>
                  下一步
                </Button>
              )}
              {currentStep === 3 && (
                <Button type="primary" onClick={() => form.submit()}>
                  创建任务
                </Button>
              )}
            </Space>
          </div>
        </Form>
      </Modal>

      {/* 任务详情模态框 */}
      <Modal
        title="任务详情"
        open={detailModalVisible}
        onCancel={() => setDetailModalVisible(false)}
        width={800}
        footer={null}
      >
        {selectedJob && (
          <div>
            <Descriptions title={selectedJob.name} bordered>
              <Descriptions.Item label="任务ID">{selectedJob.id}</Descriptions.Item>
              <Descriptions.Item label="类型">
                <Tag color={getTypeColor(selectedJob.type)}>
                  {getTypeName(selectedJob.type)}
                </Tag>
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                {getStatusBadge(selectedJob.status)}
              </Descriptions.Item>
              <Descriptions.Item label="数据源">{selectedJob.source}</Descriptions.Item>
              <Descriptions.Item label="目标">{selectedJob.target}</Descriptions.Item>
              <Descriptions.Item label="格式">{selectedJob.format.toUpperCase()}</Descriptions.Item>
              <Descriptions.Item label="进度">{selectedJob.progress}%</Descriptions.Item>
              <Descriptions.Item label="总记录数">
                {selectedJob.records_total.toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="已处理">
                {selectedJob.records_processed.toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="错误数">
                {selectedJob.errors_count.toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="创建时间">
                {new Date(selectedJob.created_at).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="开始时间">
                {selectedJob.started_at 
                  ? new Date(selectedJob.started_at).toLocaleString() 
                  : '未开始'
                }
              </Descriptions.Item>
              <Descriptions.Item label="完成时间">
                {selectedJob.completed_at 
                  ? new Date(selectedJob.completed_at).toLocaleString() 
                  : '未完成'
                }
              </Descriptions.Item>
            </Descriptions>

            <Divider />

            <Progress 
              percent={selectedJob.progress} 
              status={selectedJob.status === 'failed' ? 'exception' : undefined}
              style={{ marginBottom: '16px' }}
            />

            {selectedJob.status === 'running' && (
              <Timeline>
                <Timeline.Item color="blue">任务已启动</Timeline.Item>
                <Timeline.Item color="blue">正在处理数据...</Timeline.Item>
                <Timeline.Item color="gray">等待完成</Timeline.Item>
              </Timeline>
            )}
          </div>
        )}
      </Modal>
    </div>
  )
}

export default KnowledgeGraphDataMigration