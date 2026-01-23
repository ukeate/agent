import { buildApiUrl, apiFetch } from '../../utils/apiBase'
import React, { useState, useEffect } from 'react'
import { logger } from '../../utils/logger'
import {
  App,
  Card,
  Table,
  Button,
  Space,
  Tag,
  Progress,
  message,
  Row,
  Col,
  Statistic,
  Tooltip,
  Select,
  Input,
} from 'antd'
import {
  SaveOutlined,
  ReloadOutlined,
  DeleteOutlined,
  HistoryOutlined,
  CloudDownloadOutlined,
  ExclamationCircleOutlined,
  FileTextOutlined,
} from '@ant-design/icons'

const { Search } = Input
const { Option } = Select

interface CheckpointMetadata {
  checkpoint_id: string
  job_id: string
  created_at: string
  checkpoint_type: 'manual' | 'auto' | 'emergency'
  task_count: number
  completed_tasks: number
  failed_tasks: number
  file_size: number
  checksum: string
  tags: Record<string, string>
}

interface CheckpointStats {
  total_checkpoints: number
  total_size_bytes: number
  jobs_with_checkpoints: number
  checkpoint_types: Record<string, number>
  oldest_checkpoint?: string
  newest_checkpoint?: string
}

interface BatchJob {
  id: string
  name: string
  status: string
  progress: number
  total_tasks: number
  completed_tasks: number
  failed_tasks: number
}

const CheckpointManager: React.FC = () => {
  const { modal } = App.useApp()
  const [checkpoints, setCheckpoints] = useState<CheckpointMetadata[]>([])
  const [jobs, setJobs] = useState<BatchJob[]>([])
  const [stats, setStats] = useState<CheckpointStats>({
    total_checkpoints: 0,
    total_size_bytes: 0,
    jobs_with_checkpoints: 0,
    checkpoint_types: {},
  })
  const [loading, setLoading] = useState(false)
  const [selectedJobId, setSelectedJobId] = useState<string>('all')
  const [searchText, setSearchText] = useState('')

  const fetchCheckpoints = async (jobId?: string) => {
    setLoading(true)
    try {
      const url =
        jobId && jobId !== 'all'
          ? `/api/v1/batch/checkpoints?job_id=${jobId}`
          : '/api/v1/batch/checkpoints'

      const response = await apiFetch(buildApiUrl(url))
      const data = await response.json()
      setCheckpoints(data.checkpoints || [])
    } catch (error) {
      logger.error('获取检查点列表失败:', error)
      message.error('获取检查点列表失败')
    } finally {
      setLoading(false)
    }
  }

  const fetchStats = async () => {
    try {
      const response = await apiFetch(
        buildApiUrl('/api/v1/batch/checkpoints/stats')
      )
      const data = await response.json()
      setStats(data)
    } catch (error) {
      logger.error('获取检查点统计失败:', error)
    }
  }

  const fetchJobs = async () => {
    try {
      const response = await apiFetch(buildApiUrl('/api/v1/batch/jobs'))
      const data = await response.json()
      setJobs(data.jobs || [])
    } catch (error) {
      logger.error('获取作业列表失败:', error)
    }
  }

  const createCheckpoint = async (jobId: string) => {
    try {
      const response = await apiFetch(
        buildApiUrl(`/api/v1/batch/jobs/${jobId}/checkpoint`),
        {
          method: 'POST',
        }
      )

      const data = await response.json()
      message.success(`检查点创建成功: ${data.checkpoint_id}`)
      fetchCheckpoints(selectedJobId)
      fetchStats()
    } catch (error) {
      logger.error('创建检查点失败:', error)
      message.error('创建检查点失败')
    }
  }

  const restoreFromCheckpoint = async (
    checkpointId: string,
    _checkpointJobId: string
  ) => {
    modal.confirm({
      title: '确认恢复',
      content: `确定要从检查点 ${checkpointId.slice(0, 8)}... 恢复作业吗？`,
      icon: <ExclamationCircleOutlined />,
      onOk: async () => {
        try {
          const response = await apiFetch(
            buildApiUrl(`/api/v1/batch/checkpoints/${checkpointId}/restore`),
            {
              method: 'POST',
            }
          )

          const data = await response.json()
          message.success(`作业恢复成功: ${data.job_id}`)
          fetchJobs()
        } catch (error) {
          logger.error('恢复作业失败:', error)
          message.error('恢复作业失败')
        }
      },
    })
  }

  const deleteCheckpoint = async (checkpointId: string) => {
    modal.confirm({
      title: '确认删除',
      content: '确定要删除这个检查点吗？此操作不可恢复。',
      icon: <ExclamationCircleOutlined />,
      okButtonProps: { danger: true },
      onOk: async () => {
        try {
          const response = await apiFetch(
            buildApiUrl(`/api/v1/batch/checkpoints/${checkpointId}`),
            {
              method: 'DELETE',
            }
          )

          await response.json().catch(() => null)
          message.success('检查点删除成功')
          fetchCheckpoints(selectedJobId)
          fetchStats()
        } catch (error) {
          logger.error('删除检查点失败:', error)
          message.error('删除检查点失败')
        }
      },
    })
  }

  useEffect(() => {
    fetchCheckpoints(selectedJobId)
    fetchStats()
    fetchJobs()
  }, [selectedJobId])

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'manual':
        return 'blue'
      case 'auto':
        return 'green'
      case 'emergency':
        return 'red'
      default:
        return 'default'
    }
  }

  const filteredCheckpoints = checkpoints.filter(
    cp =>
      searchText === '' ||
      cp.checkpoint_id.toLowerCase().includes(searchText.toLowerCase()) ||
      cp.job_id.toLowerCase().includes(searchText.toLowerCase())
  )

  const columns = [
    {
      title: '检查点ID',
      dataIndex: 'checkpoint_id',
      key: 'checkpoint_id',
      render: (id: string) => (
        <Tooltip title={id}>
          <code>{id.slice(0, 12)}...</code>
        </Tooltip>
      ),
    },
    {
      title: '作业ID',
      dataIndex: 'job_id',
      key: 'job_id',
      render: (id: string) => (
        <Tooltip title={id}>
          <code>{id.slice(0, 8)}...</code>
        </Tooltip>
      ),
    },
    {
      title: '类型',
      dataIndex: 'checkpoint_type',
      key: 'checkpoint_type',
      render: (type: string) => (
        <Tag color={getTypeColor(type)}>{type.toUpperCase()}</Tag>
      ),
    },
    {
      title: '进度',
      key: 'progress',
      render: (_: any, record: CheckpointMetadata) => {
        const progress =
          record.task_count > 0
            ? Math.round((record.completed_tasks / record.task_count) * 100)
            : 0
        return (
          <div style={{ width: '120px' }}>
            <Progress
              percent={progress}
              size="small"
              format={() => `${record.completed_tasks}/${record.task_count}`}
            />
          </div>
        )
      },
    },
    {
      title: '文件大小',
      dataIndex: 'file_size',
      key: 'file_size',
      render: (size: number) => formatFileSize(size),
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
      render: (_: any, record: CheckpointMetadata) => (
        <Space>
          <Tooltip title="恢复作业">
            <Button
              size="small"
              icon={<CloudDownloadOutlined />}
              onClick={() =>
                restoreFromCheckpoint(record.checkpoint_id, record.job_id)
              }
            >
              恢复
            </Button>
          </Tooltip>
          <Tooltip title="删除检查点">
            <Button
              size="small"
              danger
              icon={<DeleteOutlined />}
              onClick={() => deleteCheckpoint(record.checkpoint_id)}
            >
              删除
            </Button>
          </Tooltip>
        </Space>
      ),
    },
  ]

  return (
    <div className="checkpoint-manager">
      {/* 统计卡片 */}
      <Row gutter={[16, 16]} className="mb-4">
        <Col span={6}>
          <Card>
            <Statistic
              title="总检查点数"
              value={stats.total_checkpoints}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="存储空间"
              value={formatFileSize(stats.total_size_bytes)}
              valueStyle={{ fontSize: '20px' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="覆盖作业数"
              value={stats.jobs_with_checkpoints}
              prefix={<HistoryOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <div className="text-center">
              <div className="text-gray-600 mb-2">检查点类型分布</div>
              <div className="space-y-1">
                {Object.entries(stats.checkpoint_types).map(([type, count]) => (
                  <div key={type} className="flex justify-between items-center">
                    <Tag color={getTypeColor(type)}>{type}</Tag>
                    <span className="text-sm">{count}</span>
                  </div>
                ))}
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      {/* 控制面板 */}
      <Card className="mb-4">
        <Row gutter={[16, 16]} align="middle">
          <Col span={8}>
            <Space>
              <span>作业筛选:</span>
              <Select
                value={selectedJobId}
                onChange={setSelectedJobId}
                style={{ width: 200 }}
                placeholder="选择作业"
              >
                <Option value="all">所有作业</Option>
                {jobs.map(job => (
                  <Option key={job.id} value={job.id}>
                    {job.name || job.id.slice(0, 8)}
                  </Option>
                ))}
              </Select>
            </Space>
          </Col>
          <Col span={8}>
            <Search
              placeholder="搜索检查点ID或作业ID"
              value={searchText}
              onChange={e => setSearchText(e.target.value)}
              style={{ width: '100%' }}
            />
          </Col>
          <Col span={8}>
            <div className="flex justify-end">
              <Space>
                {selectedJobId !== 'all' && (
                  <Button
                    type="primary"
                    icon={<SaveOutlined />}
                    onClick={() => createCheckpoint(selectedJobId)}
                  >
                    创建检查点
                  </Button>
                )}
                <Button
                  icon={<ReloadOutlined />}
                  onClick={() => {
                    fetchCheckpoints(selectedJobId)
                    fetchStats()
                  }}
                  loading={loading}
                >
                  刷新
                </Button>
              </Space>
            </div>
          </Col>
        </Row>
      </Card>

      {/* 检查点列表 */}
      <Card title={`检查点列表 (${filteredCheckpoints.length})`}>
        <Table
          columns={columns}
          dataSource={filteredCheckpoints}
          rowKey="checkpoint_id"
          loading={loading}
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: total => `共 ${total} 个检查点`,
          }}
          scroll={{ x: 'max-content' }}
        />
      </Card>
    </div>
  )
}

export default CheckpointManager
