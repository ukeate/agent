import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Alert, Button, Card, Descriptions, Space, message } from 'antd'
import { BookOutlined, FileTextOutlined, ReloadOutlined } from '@ant-design/icons'

type DocStatus = {
  status?: string
  started_at?: string
  completed_at?: string
  failed_at?: string
  error?: string
  result?: any
}

const DocumentationManagementPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [statusLoading, setStatusLoading] = useState(false)
  const [docStatus, setDocStatus] = useState<DocStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  const fetchStatus = async () => {
    setStatusLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/platform/documentation/status'))
      const data = await res.json()
      setDocStatus(data?.documentation || null)
    } catch (e: any) {
      setError(e?.message || '获取状态失败')
    } finally {
      setStatusLoading(false)
    }
  }

  useEffect(() => {
    fetchStatus()
    const interval = setInterval(fetchStatus, 5000)
    return () => clearInterval(interval)
  }, [])

  const start = async (type: 'documentation' | 'training') => {
    setLoading(true)
    setError(null)
    try {
      const url = buildApiUrl(
        type === 'documentation'
          ? '/api/v1/platform/documentation/generate'
          : '/api/v1/platform/documentation/training-materials'
      )
      const res = await apiFetch(url, { method: 'POST' })
      message.success(type === 'documentation' ? '已启动文档生成' : '已启动培训材料生成')
      await fetchStatus()
    } catch (e: any) {
      setError(e?.message || '启动失败')
      message.error('启动失败')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Card title="文档生成">
          <Space wrap>
            <Button
              type="primary"
              icon={<FileTextOutlined />}
              loading={loading}
              onClick={() => start('documentation')}
            >
              生成文档
            </Button>
            <Button icon={<BookOutlined />} loading={loading} onClick={() => start('training')}>
              生成培训材料
            </Button>
            <Button icon={<ReloadOutlined />} loading={statusLoading} onClick={fetchStatus}>
              刷新状态
            </Button>
          </Space>

          {error && <Alert style={{ marginTop: 12 }} type="error" message={error} />}

          <Descriptions size="small" column={1} style={{ marginTop: 12 }}>
            <Descriptions.Item label="状态">{docStatus?.status || 'unknown'}</Descriptions.Item>
            <Descriptions.Item label="开始时间">{docStatus?.started_at || '-'}</Descriptions.Item>
            <Descriptions.Item label="结束时间">{docStatus?.completed_at || docStatus?.failed_at || '-'}</Descriptions.Item>
          </Descriptions>

          {docStatus?.error && <Alert style={{ marginTop: 12 }} type="error" message={docStatus.error} />}
          {docStatus?.result && (
            <pre style={{ marginTop: 12, background: '#fafafa', padding: 12, overflow: 'auto' }}>
              {JSON.stringify(docStatus.result, null, 2)}
            </pre>
          )}
        </Card>
      </Space>
    </div>
  )
}

export default DocumentationManagementPage
