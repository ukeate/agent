import React, { useEffect, useState } from 'react'
import {
  Badge,
  Button,
  Card,
  Col,
  Drawer,
  Empty,
  Modal,
  Row,
  Space,
  Statistic,
  Table,
  Tag,
  Typography,
  message,
  Select,
  Progress,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import {
  BranchesOutlined,
  DeleteOutlined,
  EyeOutlined,
  ReloadOutlined,
  StopOutlined,
} from '@ant-design/icons'
import apiClient from '../services/apiClient'

const { Title, Paragraph, Text } = Typography
const { Option } = Select

interface PipelineStatus {
  active_jobs: number
  queue_length: number
  max_concurrent_jobs: number
  device_info: any
  timestamp: string
}

interface CompressionJobListItem {
  job_id: string
  job_name: string
  status: string
  created_at: string
  message: string
}

interface JobStatus {
  job_id: string
  current_stage: string
  progress_percent: number
  estimated_time_remaining: number
  last_update: string
  recent_logs: string[]
}

interface CompressionResult {
  job_id: string
  compression_ratio: number
  speedup_ratio?: number
  memory_reduction?: number
  accuracy_retention?: number
  compressed_model_path: string
  evaluation_report_path: string
  compression_time: number
}

const CompressionPipelinePage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [refreshInterval, setRefreshInterval] = useState(5000)
  const [status, setStatus] = useState<PipelineStatus | null>(null)
  const [jobs, setJobs] = useState<CompressionJobListItem[]>([])
  const [selectedJob, setSelectedJob] = useState<CompressionJobListItem | null>(
    null
  )
  const [selectedJobStatus, setSelectedJobStatus] = useState<JobStatus | null>(
    null
  )
  const [selectedJobResult, setSelectedJobResult] =
    useState<CompressionResult | null>(null)
  const [detailVisible, setDetailVisible] = useState(false)
  const [logVisible, setLogVisible] = useState(false)

  const loadData = async () => {
    setLoading(true)
    try {
      const [statusResp, jobsResp] = await Promise.all([
        apiClient.get<PipelineStatus>('/model-compression/status'),
        apiClient.get<CompressionJobListItem[]>('/model-compression/jobs'),
      ])
      setStatus(statusResp.data || null)
      setJobs(Array.isArray(jobsResp.data) ? jobsResp.data : [])
    } catch (e: any) {
      message.error(e?.message || '加载流水线数据失败')
      setStatus(null)
      setJobs([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
  }, [])

  useEffect(() => {
    if (!autoRefresh) return
    const interval = setInterval(() => loadData(), refreshInterval)
    return () => clearInterval(interval)
  }, [autoRefresh, refreshInterval])

  const getBadgeStatus = (stage: string) => {
    const s = (stage || '').toLowerCase()
    if (s === 'completed') return 'success'
    if (s === 'failed') return 'error'
    if (s === 'cancelled') return 'warning'
    if (s === 'queued') return 'default'
    return 'processing'
  }

  const isTerminalStage = (stage: string) => {
    const s = (stage || '').toLowerCase()
    return s === 'completed' || s === 'failed' || s === 'cancelled'
  }

  const fetchJobStatus = async (jobId: string) => {
    const resp = await apiClient.get<JobStatus>(
      `/model-compression/jobs/${jobId}`
    )
    return resp.data || null
  }

  const openDetails = async (job: CompressionJobListItem) => {
    setSelectedJob(job)
    setSelectedJobStatus(null)
    setSelectedJobResult(null)
    setDetailVisible(true)
    try {
      setSelectedJobStatus(await fetchJobStatus(job.job_id))
    } catch (e: any) {
      message.error(e?.message || '加载任务状态失败')
    }
  }

  const openLogs = async (job: CompressionJobListItem) => {
    setSelectedJob(job)
    setSelectedJobStatus(null)
    setLogVisible(true)
    try {
      setSelectedJobStatus(await fetchJobStatus(job.job_id))
    } catch (e: any) {
      message.error(e?.message || '加载任务日志失败')
    }
  }

  const cancelJob = async (job: CompressionJobListItem) => {
    try {
      await apiClient.put(`/model-compression/jobs/${job.job_id}/cancel`)
      message.success('任务已取消')
      await loadData()
    } catch (e: any) {
      message.error(e?.message || '取消任务失败')
    }
  }

  const deleteJob = async (job: CompressionJobListItem) => {
    Modal.confirm({
      title: '确认删除任务',
      content: `确定要删除任务 "${job.job_id}" 吗？`,
      okText: '删除',
      okButtonProps: { danger: true },
      cancelText: '取消',
      onOk: async () => {
        try {
          await apiClient.delete(`/model-compression/jobs/${job.job_id}`, {
            params: { keep_result: true },
          })
          message.success('任务已删除')
          await loadData()
        } catch (e: any) {
          message.error(e?.message || '删除任务失败')
        }
      },
    })
  }

  const loadResult = async () => {
    if (!selectedJob) return
    try {
      const resp = await apiClient.get<CompressionResult>(
        `/model-compression/results/${selectedJob.job_id}`
      )
      setSelectedJobResult(resp.data || null)
    } catch (e: any) {
      message.error(e?.message || '加载任务结果失败')
      setSelectedJobResult(null)
    }
  }

  const columns: ColumnsType<CompressionJobListItem> = [
    {
      title: '任务',
      key: 'job',
      render: (_, record) => (
        <div>
          <div style={{ fontWeight: 500 }}>
            {record.job_name || record.job_id}
          </div>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {record.job_id}
          </Text>
        </div>
      ),
    },
    {
      title: '阶段',
      dataIndex: 'status',
      key: 'status',
      render: (v: string) => (
        <Badge status={getBadgeStatus(v) as any} text={v} />
      ),
    },
    {
      title: '更新时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (v: string) => <Text style={{ fontSize: 12 }}>{v}</Text>,
    },
    {
      title: '消息',
      dataIndex: 'message',
      key: 'message',
      render: (v: string) => <Text style={{ fontSize: 12 }}>{v}</Text>,
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button
            type="text"
            icon={<EyeOutlined />}
            onClick={() => openDetails(record)}
          />
          <Button
            type="text"
            icon={<ReloadOutlined />}
            onClick={() => openLogs(record)}
          />
          <Button
            type="text"
            icon={<StopOutlined />}
            onClick={() => cancelJob(record)}
            disabled={isTerminalStage(record.status)}
          />
          <Button
            type="text"
            danger
            icon={<DeleteOutlined />}
            onClick={() => deleteJob(record)}
          />
        </Space>
      ),
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: 16,
        }}
      >
        <div>
          <Title level={2} style={{ margin: 0 }}>
            <BranchesOutlined /> 压缩流水线管理
          </Title>
          <Paragraph style={{ marginBottom: 0 }}>
            数据来自 `/api/v1/model-compression/status` 与
            `/api/v1/model-compression/jobs`
          </Paragraph>
        </div>
        <Space>
          <Select
            value={refreshInterval}
            onChange={setRefreshInterval}
            style={{ width: 120 }}
          >
            <Option value={5000}>5秒</Option>
            <Option value={10000}>10秒</Option>
            <Option value={30000}>30秒</Option>
            <Option value={60000}>1分钟</Option>
          </Select>
          <Button
            icon={<ReloadOutlined />}
            onClick={() => loadData()}
            loading={loading}
          >
            刷新
          </Button>
          <Button onClick={() => setAutoRefresh(v => !v)}>
            {autoRefresh ? '停止刷新' : '开启刷新'}
          </Button>
        </Space>
      </div>

      <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="活跃任务" value={status?.active_jobs ?? 0} />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="队列长度" value={status?.queue_length ?? 0} />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic
              title="并发上限"
              value={status?.max_concurrent_jobs ?? 0}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Card>
            <Statistic title="更新时间" value={status?.timestamp || '-'} />
          </Card>
        </Col>
      </Row>

      <Card
        title={
          <Space>
            <Tag color="blue">任务列表</Tag>
            <Text type="secondary" style={{ fontSize: 12 }}>
              当前仅展示活跃任务
            </Text>
          </Space>
        }
        extra={<Tag>{jobs.length} 个</Tag>}
      >
        <Table
          rowKey="job_id"
          dataSource={jobs}
          columns={columns}
          loading={loading}
          pagination={{ pageSize: 10 }}
        />
        {!loading && jobs.length === 0 ? (
          <Empty description="暂无活跃任务" />
        ) : null}
      </Card>

      <Modal
        title={selectedJob ? `任务详情: ${selectedJob.job_id}` : '任务详情'}
        open={detailVisible}
        onCancel={() => setDetailVisible(false)}
        footer={[
          <Button
            key="result"
            type="primary"
            onClick={() => loadResult()}
            disabled={!selectedJob}
          >
            加载结果
          </Button>,
          <Button key="close" onClick={() => setDetailVisible(false)}>
            关闭
          </Button>,
        ]}
        width={800}
      >
        {selectedJobStatus ? (
          <Space direction="vertical" style={{ width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text>当前阶段</Text>
              <Text strong>{selectedJobStatus.current_stage}</Text>
            </div>
            <Progress
              percent={Math.round(selectedJobStatus.progress_percent)}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text>预计剩余(秒)</Text>
              <Text>{selectedJobStatus.estimated_time_remaining}</Text>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between' }}>
              <Text>更新时间</Text>
              <Text>{selectedJobStatus.last_update}</Text>
            </div>
          </Space>
        ) : (
          <Empty description="暂无状态数据" />
        )}
        {selectedJobResult ? (
          <pre
            style={{
              marginTop: 16,
              background: '#f5f5f5',
              padding: 12,
              borderRadius: 4,
            }}
          >
            {JSON.stringify(selectedJobResult, null, 2)}
          </pre>
        ) : null}
      </Modal>

      <Drawer
        title={selectedJob ? `任务日志: ${selectedJob.job_id}` : '任务日志'}
        placement="right"
        onClose={() => setLogVisible(false)}
        open={logVisible}
        width={560}
      >
        {selectedJobStatus?.recent_logs?.length ? (
          <Space direction="vertical" style={{ width: '100%' }}>
            {selectedJobStatus.recent_logs.map((line, index) => (
              <div
                key={index}
                style={{
                  background: '#f5f5f5',
                  padding: 8,
                  borderRadius: 4,
                  fontSize: 12,
                }}
              >
                {line}
              </div>
            ))}
          </Space>
        ) : (
          <Empty description="暂无日志" />
        )}
      </Drawer>
    </div>
  )
}

export default CompressionPipelinePage
