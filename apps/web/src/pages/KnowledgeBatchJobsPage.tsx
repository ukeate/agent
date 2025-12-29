import React, { useEffect, useState } from 'react'
import { Alert, Card, Table, Tag, Button, Space, Typography } from 'antd'
import { DatabaseOutlined, PlayCircleOutlined, PauseCircleOutlined } from '@ant-design/icons'
import { knowledgeExtractionService } from '../services/knowledgeExtractionService'

const { Title, Paragraph } = Typography

const KnowledgeBatchJobsPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [jobs, setJobs] = useState<Array<{
    batch_id: string
    status: string
    total_documents: number
    processed_documents: number
    successful_documents: number
    failed_documents: number
    progress: number
    created_at: string
    updated_at: string
  }>>([])

  const columns = [
    { title: '任务名称', dataIndex: 'name', key: 'name' },
    { 
      title: '状态', 
      dataIndex: 'status', 
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'processing' ? 'processing' : status === 'completed' ? 'success' : status === 'failed' ? 'error' : 'default'}>
          {status === 'processing' ? '运行中' : status === 'completed' ? '已完成' : status === 'failed' ? '失败' : '等待中'}
        </Tag>
      )
    },
    { title: '进度', dataIndex: 'progress', key: 'progress', render: (progress: number) => `${progress}%` },
    { title: '文档数', dataIndex: 'documents', key: 'documents' },
    {
      title: '操作',
      key: 'action',
      render: () => (
        <Space>
          <Button size="small" icon={<PauseCircleOutlined />}>暂停</Button>
          <Button size="small" type="link">详情</Button>
        </Space>
      )
    }
  ]

  useEffect(() => {
    const loadJobs = async () => {
      setLoading(true)
      setError(null)
      try {
        const data = await knowledgeExtractionService.listBatchJobs()
        setJobs(Array.isArray(data) ? data : [])
      } catch (err) {
        setError((err as Error).message || '加载批处理任务失败')
      } finally {
        setLoading(false)
      }
    }
    loadJobs()
  }, [])

  const dataSource = jobs.map(job => ({
    key: job.batch_id,
    name: job.batch_id,
    status: job.status,
    progress: Math.round(job.progress || 0),
    documents: job.total_documents
  }))

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <DatabaseOutlined style={{ marginRight: 8 }} />
          批处理作业管理
        </Title>
        <Paragraph type="secondary">
          管理知识图谱的批量处理任务
        </Paragraph>
      </div>

      <Card title="批处理任务" extra={<Button type="primary" icon={<PlayCircleOutlined />}>新建任务</Button>}>
        {error && <Alert type="error" message={error} showIcon style={{ marginBottom: 12 }} />}
        <Table columns={columns} dataSource={dataSource} loading={loading} />
      </Card>
    </div>
  )
}

export default KnowledgeBatchJobsPage
