import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Space, Typography, Button, Alert, Spin, Tag } from 'antd'
import { ShieldOutlined, ReloadOutlined } from '@ant-design/icons'

type Policy = {
  id: string
  name: string
  status?: string
  regulation?: string
  updated_at?: string
}

type Audit = {
  id: string
  event: string
  timestamp: string
  user?: string
  result?: string
}

const PrivacyEthicsPage: React.FC = () => {
  const [policies, setPolicies] = useState<Policy[]>([])
  const [audits, setAudits] = useState<Audit[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [pRes, aRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/security/config'),
        apiFetch(buildApiUrl('/api/v1/security/mcp-tools/audit?limit=100'))
      ])
      const pData = await pRes.json()
      const aData = await aRes.json()
      setPolicies(
        Array.isArray(pData?.policies)
          ? pData.policies
          : [
              {
                id: 'security-config',
                name: 'Security Config',
                status: 'active',
                regulation: 'internal',
                updated_at: new Date().toISOString()
              }
            ]
      )
      setAudits(Array.isArray(aData?.logs) ? aData.logs : [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setPolicies([])
      setAudits([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <ShieldOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              隐私与伦理合规
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card title="策略配置">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="id"
              dataSource={policies}
              columns={[
                { title: '名称', dataIndex: 'name' },
                { title: '状态', dataIndex: 'status', render: (v) => <Tag>{v || '-'}</Tag> },
                { title: '法规', dataIndex: 'regulation' },
                { title: '更新时间', dataIndex: 'updated_at' }
              ]}
              locale={{ emptyText: '暂无策略数据，请在后端配置后查看。' }}
            />
          )}
        </Card>

        <Card title="审计事件">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey={(r) => r.id || r.event}
              dataSource={audits}
              columns={[
                { title: '事件', dataIndex: 'event' },
                { title: '时间', dataIndex: 'timestamp' },
                { title: '用户', dataIndex: 'user' },
                { title: '结果', dataIndex: 'result' }
              ]}
              locale={{ emptyText: '暂无审计记录，可通过实际操作产生。' }}
            />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default PrivacyEthicsPage
