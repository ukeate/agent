import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState, useMemo } from 'react'
import { Card, Table, Typography, Space, Tag, Statistic, Row, Col, Alert, Spin, Select, Button } from 'antd'
import { ReloadOutlined, BarChartOutlined } from '@ant-design/icons'

const { Title, Text } = Typography
const { Option } = Select

interface ModelMeta {
  name: string
  version: string
  format?: string
  size_bytes?: number
  tags?: string[]
  created_at?: string
  updated_at?: string
  path?: string
}

interface ModelListResponse {
  models: ModelMeta[]
  total: number
}

const ModelPerformanceComparison: React.FC = () => {
  const [models, setModels] = useState<ModelMeta[]>([])
  const [loading, setLoading] = useState(false)
  const [selected, setSelected] = useState<string[]>([])

  const loadModels = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/models'))
      const data: ModelListResponse = await res.json()
      setModels(data.models || [])
      setSelected((data.models || []).slice(0, 3).map(m => `${m.name}:${m.version}`))
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadModels()
  }, [])

  const rows = useMemo(
    () =>
      models.map((m, idx) => ({
        key: `${m.name}:${m.version}:${idx}`,
        ...m,
      })),
    [models]
  )

  const selectedRows = rows.filter(r => selected.includes(`${r.name}:${r.version}`))

  const totalSize = selectedRows.reduce((sum, r) => sum + (r.size_bytes || 0), 0)

  const columns = [
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '版本', dataIndex: 'version', key: 'version' },
    {
      title: '格式',
      dataIndex: 'format',
      key: 'format',
      render: (v: string | undefined) => v || '-',
    },
    {
      title: '大小',
      dataIndex: 'size_bytes',
      key: 'size_bytes',
      render: (v: number | undefined) => (v ? `${(v / 1024 / 1024).toFixed(1)} MB` : '-'),
    },
    {
      title: '标签',
      dataIndex: 'tags',
      key: 'tags',
      render: (tags: string[] | undefined) =>
        tags && tags.length ? tags.map(t => <Tag key={t}>{t}</Tag>) : <Text type="secondary">无</Text>,
    },
    { title: '创建时间', dataIndex: 'created_at', key: 'created_at' },
    { title: '更新时间', dataIndex: 'updated_at', key: 'updated_at' },
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Title level={3} style={{ margin: 0 }}>
            <BarChartOutlined /> 模型性能对比（实时数据）
          </Title>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={loadModels} loading={loading}>
              刷新
            </Button>
          </Space>
        </Space>

        {loading ? (
          <Spin />
        ) : models.length === 0 ? (
          <Alert type="warning" message="暂无模型数据，请先在模型注册表中注册模型。" />
        ) : (
          <>
            <Card>
              <Space direction="vertical" style={{ width: '100%' }}>
                <Text>选择要对比的模型（最多 5 个）</Text>
                <Select
                  mode="multiple"
                  style={{ width: '100%' }}
                  value={selected}
                  onChange={setSelected}
                  maxTagCount={5}
                >
                  {rows.map(r => (
                    <Option key={`${r.name}:${r.version}`} value={`${r.name}:${r.version}`}>
                      {r.name}:{r.version}
                    </Option>
                  ))}
                </Select>
              </Space>
            </Card>

            <Card title="模型列表">
              <Table columns={columns} dataSource={rows} pagination={{ pageSize: 10 }} />
            </Card>

            {selectedRows.length > 0 && (
              <Card title="对比概要">
                <Row gutter={16}>
                  <Col span={6}>
                    <Statistic title="对比模型数" value={selectedRows.length} />
                  </Col>
                  <Col span={6}>
                    <Statistic
                      title="总大小"
                      value={totalSize / 1024 / 1024}
                      precision={1}
                      suffix="MB"
                    />
                  </Col>
                </Row>
              </Card>
            )}
          </>
        )}
      </Space>
    </div>
  )
}

export default ModelPerformanceComparison
