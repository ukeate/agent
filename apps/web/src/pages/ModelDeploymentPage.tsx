import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Table, Space, Typography, Button, Alert, Spin, Tag } from 'antd'
import { CloudDownloadOutlined, ReloadOutlined } from '@ant-design/icons'

type ModelRow = { id?: string; name: string; version?: string; framework?: string; status?: string };
type DeploymentRow = { deployment_id: string; model_name: string; model_version?: string; deployment_type?: string; status: string; created_at?: string };

const ModelDeploymentPage: React.FC = () => {
  const [models, setModels] = useState<ModelRow[]>([])
  const [deployments, setDeployments] = useState<DeploymentRow[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const [mRes, dRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/model-service/models'),
        apiFetch(buildApiUrl('/api/v1/model-service/deployment/list'))
      ])
      const mData = await mRes.json()
      const dData = await dRes.json()
      setModels(Array.isArray(mData?.models) ? mData.models : [])
      setDeployments(Array.isArray(dData?.deployments) ? dData.deployments : [])
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setModels([])
      setDeployments([])
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
            <CloudDownloadOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              模型部署
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card title="模型列表">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey={(r) => r.id || r.name}
              dataSource={models}
              columns={[
                { title: '名称', dataIndex: 'name' },
                { title: '版本', dataIndex: 'version' },
                { title: '框架', dataIndex: 'framework' },
                { title: '状态', dataIndex: 'status' }
              ]}
              locale={{ emptyText: '暂无模型，先通过 /api/v1/model-service 上传注册。' }}
            />
          )}
        </Card>

        <Card title="部署列表">
          {loading ? (
            <Spin />
          ) : (
            <Table
              rowKey="deployment_id"
              dataSource={deployments}
              columns={[
                { title: '部署ID', dataIndex: 'deployment_id' },
                { title: '模型', render: (_, record) => `${record.model_name}${record.model_version ? `:${record.model_version}` : ''}` },
                { title: '类型', dataIndex: 'deployment_type' },
                { title: '状态', dataIndex: 'status', render: (v) => <Tag color={v === 'deployed' ? 'green' : 'default'}>{v}</Tag> },
                { title: '创建时间', dataIndex: 'created_at' }
              ]}
              locale={{ emptyText: '暂无部署，调用 /api/v1/model-service/deployment 请求创建。' }}
            />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default ModelDeploymentPage
