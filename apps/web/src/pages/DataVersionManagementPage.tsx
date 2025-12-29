import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Space, Typography, Button, Alert, Select, Table, Spin } from 'antd'
import { ReloadOutlined, DatabaseOutlined } from '@ant-design/icons'

type Dataset = { name: string }
type VersionRow = {
  version_id: string
  dataset_name: string
  description?: string
  created_at?: string
  record_count?: number
  size_mb?: number
}

const DataVersionManagementPage: React.FC = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selected, setSelected] = useState<string | undefined>()
  const [versions, setVersions] = useState<VersionRow[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadDatasets = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/training-data/datasets'))
      const data = await res.json()
      setDatasets(data?.datasets || [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setDatasets([])
    } finally {
      setLoading(false)
    }
  }

  const loadVersions = async (dataset: string) => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl(`/api/v1/training-data/datasets/${dataset}/versions`))
      const data = await res.json()
      setVersions(data?.versions || [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setVersions([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadDatasets()
  }, [])

  useEffect(() => {
    if (selected) loadVersions(selected)
  }, [selected])

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <DatabaseOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              数据版本管理
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={loadDatasets} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card title="选择数据集">
          <Select
            style={{ width: 240 }}
            placeholder="请选择数据集"
            value={selected}
            onChange={setSelected}
            loading={loading}
            options={(datasets || []).map((d) => ({ label: d.name || d, value: d.name || d }))}
          />
        </Card>

        <Card title="版本列表">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="version_id"
              dataSource={versions}
              pagination={{ pageSize: 10 }}
              columns={[
                { title: '版本ID', dataIndex: 'version_id' },
                { title: '数据集', dataIndex: 'dataset_name' },
                { title: '描述', dataIndex: 'description' },
                { title: '记录数', dataIndex: 'record_count' },
                { title: '大小(MB)', dataIndex: 'size_mb' },
                { title: '创建时间', dataIndex: 'created_at' },
              ]}
              locale={{ emptyText: selected ? '暂无版本' : '先选择数据集' }}
            />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default DataVersionManagementPage
