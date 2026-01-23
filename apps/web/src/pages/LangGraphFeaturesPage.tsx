import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Space, Typography, Button, Alert, Input, Tabs, Spin, Table } from 'antd'
import { ApiOutlined, ReloadOutlined, PlayCircleOutlined } from '@ant-design/icons'

const { Title } = Typography
const { TextArea } = Input
const { TabPane } = Tabs

type DemoResponse = { success?: boolean; execution_time_ms?: number; result?: any; metadata?: any }
type CacheStats = { total_requests?: number; cache_hits?: number; cache_misses?: number; hit_rate?: number }
type HookStatus = { enabled?: boolean; hooks?: any }

const LangGraphFeaturesPage: React.FC = () => {
  const [message, setMessage] = useState('测试新Context API')
  const [contextResult, setContextResult] = useState<DemoResponse | null>(null)
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null)
  const [hookStatus, setHookStatus] = useState<HookStatus | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadStats = async () => {
    setLoading(true)
    setError(null)
    try {
      const [cacheRes, hookRes] = await Promise.all([
        apiFetch(buildApiUrl('/api/v1/langgraph/cache/stats')),
        apiFetch(buildApiUrl('/api/v1/langgraph/hooks/status'))
      ])
      setCacheStats(await cacheRes.json())
      setHookStatus(await hookRes.json())
    } catch (e: any) {
      setError(e?.message || '加载失败')
      setCacheStats(null)
      setHookStatus(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadStats()
  }, [])

  const runContextDemo = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/langgraph/context-api/demo'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message, use_new_api: true, user_id: 'demo_user', session_id: 'demo_session' })
      })
      const data = await res.json()
      setContextResult(data)
    } catch (e: any) {
      setError(e?.message || '执行失败')
      setContextResult(null)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Space>
            <ApiOutlined />
            <Title level={3} style={{ margin: 0 }}>
              LangGraph 特性
            </Title>
          </Space>
          <Button icon={<ReloadOutlined />} onClick={loadStats} loading={loading}>
            刷新
          </Button>
        </Space>

        {error && <Alert type="error" message="操作失败" description={error} />}

        <Tabs defaultActiveKey="context">
          <TabPane tab="Context API" key="context">
            <Card title="执行 Context Demo">
              <Space direction="vertical" style={{ width: '100%' }} size="middle">
                <TextArea rows={4} value={message} onChange={(e) => setMessage(e.target.value)} />
                <Button type="primary" icon={<PlayCircleOutlined />} onClick={runContextDemo} loading={loading}>
                  调用 /api/v1/langgraph/context-api/demo
                </Button>
                {loading ? (
                  <Spin />
                ) : contextResult ? (
                  <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(contextResult, null, 2)}</pre>
                ) : (
                  <Alert type="info" message="尚无执行结果，点击运行。" />
                )}
              </Space>
            </Card>
          </TabPane>

          <TabPane tab="缓存统计" key="cache">
            <Card>
              {loading ? (
                <Spin />
              ) : cacheStats ? (
                <Table
                  rowKey="key"
                  dataSource={[
                    { key: 'total_requests', value: cacheStats.total_requests },
                    { key: 'cache_hits', value: cacheStats.cache_hits },
                    { key: 'cache_misses', value: cacheStats.cache_misses },
                    { key: 'hit_rate', value: cacheStats.hit_rate }
                  ]}
                  columns={[
                    { title: '指标', dataIndex: 'key' },
                    { title: '值', dataIndex: 'value' }
                  ]}
                  pagination={false}
                />
              ) : (
                <Alert type="info" message="暂无缓存统计数据。" />
              )}
            </Card>
          </TabPane>

          <TabPane tab="Hooks 状态" key="hooks">
            <Card>
              {loading ? (
                <Spin />
              ) : hookStatus ? (
                <pre style={{ whiteSpace: 'pre-wrap' }}>{JSON.stringify(hookStatus, null, 2)}</pre>
              ) : (
                <Alert type="info" message="暂无 Hooks 数据。" />
              )}
            </Card>
          </TabPane>
        </Tabs>
      </Space>
    </div>
  )
}

export default LangGraphFeaturesPage
