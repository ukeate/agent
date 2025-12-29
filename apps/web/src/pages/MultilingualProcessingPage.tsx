import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Tag, Alert } from 'antd'
import { ReloadOutlined, GlobalOutlined } from '@ant-design/icons'

type LanguageSupport = {
  code: string
  name: string
  tasks?: string[]
  status?: string
  coverage?: number
}

const MultilingualProcessingPage: React.FC = () => {
  const [languages, setLanguages] = useState<LanguageSupport[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/multilingual/supported-languages'))
      const data = await res.json()
      setLanguages(Array.isArray(data?.languages) ? data.languages : [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setLanguages([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const columns = [
    { title: '代码', dataIndex: 'code', key: 'code' },
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '任务', dataIndex: 'tasks', key: 'tasks', render: (t: string[]) => (t || []).join(', ') },
    { title: '状态', dataIndex: 'status', key: 'status', render: (s: string) => <Tag>{s}</Tag> },
    { title: '覆盖率', dataIndex: 'coverage', key: 'coverage' }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <GlobalOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              多语言处理
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message={error} />}

        <Card>
          <Table rowKey="code" dataSource={languages} columns={columns} loading={loading} />
        </Card>
      </Space>
    </div>
  )
}

export default MultilingualProcessingPage
