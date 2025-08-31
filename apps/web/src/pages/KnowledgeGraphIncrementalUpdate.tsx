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
  Badge,
  Timeline,
  message,
  Modal,
  Descriptions,
  Select,
  DatePicker,
  Switch,
  Drawer
} from 'antd'
import {
  SyncOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  SettingOutlined,
  BarChartOutlined,
  FileTextOutlined,
  ThunderboltOutlined,
  DatabaseOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { RangePicker } = DatePicker

interface UpdateJob {
  id: string
  source: string
  type: 'full' | 'incremental' | 'merge'
  status: 'pending' | 'running' | 'completed' | 'failed' | 'paused'
  started_at: string
  completed_at?: string
  progress: number
  entities_processed: number
  entities_total: number
  relations_processed: number
  relations_total: number
  conflicts_detected: number
  conflicts_resolved: number
  error_message?: string
  duration: number
}

interface ConflictResolution {
  id: string
  entity_id: string
  entity_name: string
  conflict_type: 'duplicate' | 'property_mismatch' | 'type_conflict' | 'reference_error'
  description: string
  strategy: 'auto_merge' | 'manual_review' | 'keep_existing' | 'use_new'
  status: 'pending' | 'resolved' | 'ignored'
  confidence: number
  created_at: string
}

interface UpdateMetrics {
  total_updates: number
  successful_updates: number
  failed_updates: number
  avg_duration: number
  entities_added: number
  entities_updated: number
  entities_merged: number
  relations_added: number
  relations_updated: number
}

const KnowledgeGraphIncrementalUpdate: React.FC = () => {
  const [updateJobs, setUpdateJobs] = useState<UpdateJob[]>([])
  const [conflicts, setConflicts] = useState<ConflictResolution[]>([])
  const [metrics, setMetrics] = useState<UpdateMetrics>({} as UpdateMetrics)
  const [loading, setLoading] = useState(false)
  const [autoUpdateEnabled, setAutoUpdateEnabled] = useState(true)
  const [selectedJob, setSelectedJob] = useState<UpdateJob | null>(null)
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false)
  const [configModalVisible, setConfigModalVisible] = useState(false)

  // 模拟更新任务数据
  const mockJobs: UpdateJob[] = [
    {
      id: 'job_001',
      source: 'document_batch_2025_01_22',
      type: 'incremental',
      status: 'completed',
      started_at: '2025-01-22T14:30:00Z',
      completed_at: '2025-01-22T14:45:00Z',
      progress: 100,
      entities_processed: 1250,
      entities_total: 1250,
      relations_processed: 3400,
      relations_total: 3400,
      conflicts_detected: 23,
      conflicts_resolved: 21,
      duration: 900
    },
    {
      id: 'job_002',
      source: 'real_time_stream',
      type: 'incremental',
      status: 'running',
      started_at: '2025-01-22T15:00:00Z',
      progress: 65,
      entities_processed: 820,
      entities_total: 1260,
      relations_processed: 1850,
      relations_total: 2900,
      conflicts_detected: 8,
      conflicts_resolved: 6,
      duration: 0
    },
    {
      id: 'job_003',
      source: 'manual_upload_xlsx',
      type: 'merge',
      status: 'failed',
      started_at: '2025-01-22T13:15:00Z',
      completed_at: '2025-01-22T13:20:00Z',
      progress: 15,
      entities_processed: 45,
      entities_total: 300,
      relations_processed: 12,
      relations_total: 450,
      conflicts_detected: 156,
      conflicts_resolved: 0,
      error_message: '数据格式不符合规范，存在大量重复实体',
      duration: 300
    }
  ]

  // 模拟冲突数据
  const mockConflicts: ConflictResolution[] = [
    {
      id: 'conflict_001',
      entity_id: 'entity_456',
      entity_name: '苹果公司',
      conflict_type: 'duplicate',
      description: '发现重复实体：苹果公司 vs Apple Inc.',
      strategy: 'auto_merge',
      status: 'resolved',
      confidence: 0.92,
      created_at: '2025-01-22T14:35:00Z'
    },
    {
      id: 'conflict_002', 
      entity_id: 'entity_789',
      entity_name: '张三',
      conflict_type: 'property_mismatch',
      description: '属性冲突：年龄字段存在不一致 (30 vs 32)',
      strategy: 'manual_review',
      status: 'pending',
      confidence: 0.75,
      created_at: '2025-01-22T15:05:00Z'
    },
    {
      id: 'conflict_003',
      entity_id: 'entity_012',
      entity_name: 'Python框架',
      conflict_type: 'type_conflict',
      description: '类型冲突：TECHNOLOGY vs CONCEPT',
      strategy: 'keep_existing',
      status: 'resolved',
      confidence: 0.88,
      created_at: '2025-01-22T14:42:00Z'
    }
  ]

  const mockMetrics: UpdateMetrics = {
    total_updates: 156,
    successful_updates: 142,
    failed_updates: 14,
    avg_duration: 680,
    entities_added: 12450,
    entities_updated: 8920,
    entities_merged: 1230,
    relations_added: 28900,
    relations_updated: 15600
  }

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      setUpdateJobs(mockJobs)
      setConflicts(mockConflicts)
      setMetrics(mockMetrics)
    } catch (error) {
      message.error('加载数据失败')
    } finally {
      setLoading(false)
    }
  }

  const handleJobAction = async (jobId: string, action: 'pause' | 'resume' | 'cancel' | 'retry') => {
    try {
      // 模拟API调用
      const updatedJobs = updateJobs.map(job => {
        if (job.id === jobId) {
          switch (action) {
            case 'pause':
              return { ...job, status: 'paused' as const }
            case 'resume':
              return { ...job, status: 'running' as const }
            case 'cancel':
              return { ...job, status: 'failed' as const, error_message: '用户取消' }
            case 'retry':
              return { ...job, status: 'pending' as const, error_message: undefined }
          }
        }
        return job
      })
      setUpdateJobs(updatedJobs)
      message.success(`任务${action}操作成功`)
    } catch (error) {
      message.error(`任务${action}操作失败`)
    }
  }

  const handleConflictResolve = async (conflictId: string, strategy: string) => {
    try {
      const updatedConflicts = conflicts.map(conflict => 
        conflict.id === conflictId 
          ? { ...conflict, strategy: strategy as any, status: 'resolved' as const }
          : conflict
      )
      setConflicts(updatedConflicts)
      message.success('冲突解决策略已应用')
    } catch (error) {
      message.error('冲突解决失败')
    }
  }

  const startNewUpdate = async () => {
    try {
      const newJob: UpdateJob = {
        id: `job_${Date.now()}`,
        source: 'manual_trigger',
        type: 'incremental',
        status: 'pending',
        started_at: new Date().toISOString(),
        progress: 0,
        entities_processed: 0,
        entities_total: 1000,
        relations_processed: 0,
        relations_total: 2500,
        conflicts_detected: 0,
        conflicts_resolved: 0,
        duration: 0
      }
      setUpdateJobs([newJob, ...updateJobs])
      message.success('新的增量更新任务已启动')
    } catch (error) {
      message.error('启动更新任务失败')
    }
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

  const getConflictTypeColor = (type: string) => {
    const colors = {
      'duplicate': 'orange',
      'property_mismatch': 'red',
      'type_conflict': 'purple',
      'reference_error': 'magenta'
    }
    return colors[type as keyof typeof colors] || 'default'
  }

  const getConflictTypeName = (type: string) => {
    const names = {
      'duplicate': '重复实体',
      'property_mismatch': '属性冲突',
      'type_conflict': '类型冲突',
      'reference_error': '引用错误'
    }
    return names[type as keyof typeof names] || type
  }

  const jobColumns = [
    {
      title: '任务ID',
      dataIndex: 'id',
      key: 'id',
      render: (text: string, record: UpdateJob) => (
        <Button 
          type="link" 
          onClick={() => { setSelectedJob(record); setDetailDrawerVisible(true) }}
          style={{ padding: 0 }}
        >
          {text}
        </Button>
      ),
    },
    {
      title: '数据源',
      dataIndex: 'source',
      key: 'source',
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={type === 'full' ? 'blue' : type === 'incremental' ? 'green' : 'orange'}>
          {type === 'full' ? '全量' : type === 'incremental' ? '增量' : '合并'}
        </Tag>
      ),
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
      render: (progress: number, record: UpdateJob) => (
        <div>
          <Progress percent={progress} size="small" />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            实体: {record.entities_processed}/{record.entities_total} | 
            关系: {record.relations_processed}/{record.relations_total}
          </Text>
        </div>
      ),
    },
    {
      title: '冲突',
      key: 'conflicts',
      render: (_, record: UpdateJob) => (
        <div>
          <Text>检测: {record.conflicts_detected}</Text><br />
          <Text type="secondary">解决: {record.conflicts_resolved}</Text>
        </div>
      ),
    },
    {
      title: '开始时间',
      dataIndex: 'started_at',
      key: 'started_at',
      render: (time: string) => new Date(time).toLocaleString(),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: UpdateJob) => (
        <Space>
          {record.status === 'running' && (
            <Tooltip title="暂停">
              <Button 
                type="text" 
                icon={<PauseCircleOutlined />}
                onClick={() => handleJobAction(record.id, 'pause')}
              />
            </Tooltip>
          )}
          {record.status === 'paused' && (
            <Tooltip title="继续">
              <Button 
                type="text" 
                icon={<PlayCircleOutlined />}
                onClick={() => handleJobAction(record.id, 'resume')}
              />
            </Tooltip>
          )}
          {record.status === 'failed' && (
            <Tooltip title="重试">
              <Button 
                type="text" 
                icon={<ReloadOutlined />}
                onClick={() => handleJobAction(record.id, 'retry')}
              />
            </Tooltip>
          )}
          <Tooltip title="详情">
            <Button 
              type="text" 
              icon={<InfoCircleOutlined />}
              onClick={() => { setSelectedJob(record); setDetailDrawerVisible(true) }}
            />
          </Tooltip>
        </Space>
      ),
    },
  ]

  const conflictColumns = [
    {
      title: '实体',
      dataIndex: 'entity_name',
      key: 'entity_name',
    },
    {
      title: '冲突类型',
      dataIndex: 'conflict_type',
      key: 'conflict_type',
      render: (type: string) => (
        <Tag color={getConflictTypeColor(type)}>
          {getConflictTypeName(type)}
        </Tag>
      ),
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      render: (text: string) => (
        <div style={{ maxWidth: '300px' }}>
          <Text ellipsis={{ tooltip: text }}>{text}</Text>
        </div>
      ),
    },
    {
      title: '解决策略',
      dataIndex: 'strategy',
      key: 'strategy',
      render: (strategy: string) => {
        const strategyMap = {
          'auto_merge': '自动合并',
          'manual_review': '人工审核',
          'keep_existing': '保留现有',
          'use_new': '使用新值'
        }
        return strategyMap[strategy as keyof typeof strategyMap] || strategy
      },
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => (
        <Progress 
          percent={Math.round(confidence * 100)} 
          size="small"
          strokeColor={confidence > 0.8 ? '#52c41a' : confidence > 0.6 ? '#faad14' : '#ff4d4f'}
        />
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = { 'pending': 'orange', 'resolved': 'green', 'ignored': 'gray' }
        const names = { 'pending': '待处理', 'resolved': '已解决', 'ignored': '已忽略' }
        return <Tag color={colors[status as keyof typeof colors]}>{names[status as keyof typeof names]}</Tag>
      },
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: ConflictResolution) => (
        <Space>
          {record.status === 'pending' && (
            <>
              <Button size="small" onClick={() => handleConflictResolve(record.id, 'auto_merge')}>
                自动合并
              </Button>
              <Button size="small" onClick={() => handleConflictResolve(record.id, 'keep_existing')}>
                保留现有
              </Button>
            </>
          )}
        </Space>
      ),
    },
  ]

  const pendingConflicts = conflicts.filter(c => c.status === 'pending')
  const successRate = metrics.total_updates > 0 ? (metrics.successful_updates / metrics.total_updates) * 100 : 0

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <SyncOutlined style={{ marginRight: '8px' }} />
          增量更新监控
        </Title>
        <Paragraph type="secondary">
          监控和管理知识图谱的增量更新任务，处理数据冲突和合并策略
        </Paragraph>
      </div>

      {/* 统计概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总更新任务"
              value={metrics.total_updates}
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="成功率"
              value={successRate}
              precision={1}
              suffix="%"
              valueStyle={{ color: successRate > 90 ? '#3f8600' : '#cf1322' }}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均耗时"
              value={metrics.avg_duration}
              suffix="秒"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="待处理冲突"
              value={pendingConflicts.length}
              valueStyle={{ color: pendingConflicts.length > 0 ? '#faad14' : '#3f8600' }}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 控制面板 */}
      <Card style={{ marginBottom: '16px' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Text strong>自动更新:</Text>
              <Switch 
                checked={autoUpdateEnabled} 
                onChange={setAutoUpdateEnabled}
                checkedChildren="开"
                unCheckedChildren="关"
              />
              <Divider type="vertical" />
              <Text type="secondary">下次更新: 15:30</Text>
            </Space>
          </Col>
          <Col>
            <Space>
              <Button icon={<SettingOutlined />} onClick={() => setConfigModalVisible(true)}>
                更新配置
              </Button>
              <Button icon={<ReloadOutlined />} onClick={loadData}>
                刷新
              </Button>
              <Button 
                type="primary" 
                icon={<PlayCircleOutlined />}
                onClick={startNewUpdate}
              >
                手动更新
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 更新任务列表 */}
      <Card title="更新任务" style={{ marginBottom: '16px' }}>
        <Table
          columns={jobColumns}
          dataSource={updateJobs}
          rowKey="id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showTotal: (total) => `共 ${total} 个任务`
          }}
        />
      </Card>

      {/* 冲突管理 */}
      <Card title="冲突管理" extra={
        <Badge count={pendingConflicts.length} showZero={false}>
          <Text>待处理冲突</Text>
        </Badge>
      }>
        <Table
          columns={conflictColumns}
          dataSource={conflicts}
          rowKey="id"
          pagination={{
            pageSize: 5,
            showTotal: (total) => `共 ${total} 个冲突`
          }}
        />
      </Card>

      {/* 任务详情抽屉 */}
      <Drawer
        title="任务详情"
        placement="right"
        width={600}
        open={detailDrawerVisible}
        onClose={() => setDetailDrawerVisible(false)}
      >
        {selectedJob && (
          <div>
            <Descriptions title={`任务 ${selectedJob.id}`} bordered column={2}>
              <Descriptions.Item label="数据源">{selectedJob.source}</Descriptions.Item>
              <Descriptions.Item label="类型">{selectedJob.type}</Descriptions.Item>
              <Descriptions.Item label="状态">{getStatusBadge(selectedJob.status)}</Descriptions.Item>
              <Descriptions.Item label="进度">{selectedJob.progress}%</Descriptions.Item>
              <Descriptions.Item label="开始时间">
                {new Date(selectedJob.started_at).toLocaleString()}
              </Descriptions.Item>
              <Descriptions.Item label="完成时间">
                {selectedJob.completed_at 
                  ? new Date(selectedJob.completed_at).toLocaleString() 
                  : '未完成'
                }
              </Descriptions.Item>
              <Descriptions.Item label="实体处理">
                {selectedJob.entities_processed} / {selectedJob.entities_total}
              </Descriptions.Item>
              <Descriptions.Item label="关系处理">
                {selectedJob.relations_processed} / {selectedJob.relations_total}
              </Descriptions.Item>
              <Descriptions.Item label="冲突检测">{selectedJob.conflicts_detected}</Descriptions.Item>
              <Descriptions.Item label="冲突解决">{selectedJob.conflicts_resolved}</Descriptions.Item>
            </Descriptions>

            {selectedJob.error_message && (
              <Alert
                message="错误信息"
                description={selectedJob.error_message}
                type="error"
                showIcon
                style={{ marginTop: '16px' }}
              />
            )}

            <Divider />

            <Title level={5}>处理进度</Title>
            <div style={{ marginBottom: '16px' }}>
              <Text>实体处理进度</Text>
              <Progress 
                percent={Math.round((selectedJob.entities_processed / selectedJob.entities_total) * 100)}
                status={selectedJob.status === 'failed' ? 'exception' : undefined}
              />
            </div>
            <div>
              <Text>关系处理进度</Text>
              <Progress 
                percent={Math.round((selectedJob.relations_processed / selectedJob.relations_total) * 100)}
                status={selectedJob.status === 'failed' ? 'exception' : undefined}
              />
            </div>
          </div>
        )}
      </Drawer>

      {/* 配置模态框 */}
      <Modal
        title="更新配置"
        open={configModalVisible}
        onCancel={() => setConfigModalVisible(false)}
        onOk={() => {
          setConfigModalVisible(false)
          message.success('配置已保存')
        }}
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>自动更新间隔</Text>
            <Select defaultValue="30min" style={{ width: '100%', marginTop: '8px' }}>
              <Select.Option value="15min">15分钟</Select.Option>
              <Select.Option value="30min">30分钟</Select.Option>
              <Select.Option value="1hour">1小时</Select.Option>
              <Select.Option value="6hour">6小时</Select.Option>
            </Select>
          </div>
          <div>
            <Text strong>冲突解决策略</Text>
            <Select defaultValue="auto_merge" style={{ width: '100%', marginTop: '8px' }}>
              <Select.Option value="auto_merge">自动合并</Select.Option>
              <Select.Option value="manual_review">人工审核</Select.Option>
              <Select.Option value="keep_existing">保留现有</Select.Option>
            </Select>
          </div>
          <div>
            <Text strong>并发任务数</Text>
            <Select defaultValue="3" style={{ width: '100%', marginTop: '8px' }}>
              <Select.Option value="1">1</Select.Option>
              <Select.Option value="3">3</Select.Option>
              <Select.Option value="5">5</Select.Option>
            </Select>
          </div>
        </Space>
      </Modal>
    </div>
  )
}

export default KnowledgeGraphIncrementalUpdate