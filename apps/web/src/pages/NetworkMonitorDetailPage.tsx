import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Space, Typography, Button, Alert, Spin, Statistic } from 'antd'
import { WifiOutlined, ReloadOutlined } from '@ant-design/icons'

const NetworkMonitorDetailPage: React.FC = () => {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/offline/network'))
      const d = await res.json()
      setData(d)
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setData(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const metrics = data?.network || {}

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space
          align="center"
          style={{ justifyContent: 'space-between', width: '100%' }}
        >
          <Space>
            <WifiOutlined />
            <Typography.Title level={3} style={{ margin: 0 }}>
              网络监控详情
            </Typography.Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="加载失败" description={error} />}

        <Card title="实时指标">
          {loading ? (
            <Spin />
          ) : data ? (
            <Space size="large">
              <Statistic
                title="当前状态"
                value={metrics.current_status || '-'}
              />
              <Statistic
                title="当前延迟(ms)"
                value={metrics.current_latency_ms || 0}
              />
              <Statistic
                title="当前丢包率"
                value={metrics.current_packet_loss || 0}
              />
              <Statistic
                title="连接质量"
                value={metrics.connection_quality || 0}
              />
            </Space>
          ) : (
            <Alert type="info" message="暂无数据，确保后端网络监控已运行。" />
          )}
        </Card>
      </Space>
    </div>
  )
}

export default NetworkMonitorDetailPage
