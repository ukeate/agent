import { buildApiUrl, apiFetch } from '../../utils/apiBase'
import React, { useEffect, useState } from 'react'
import { Card, Space, Typography, Button, Table, Input, Select, Drawer, message, Tabs } from 'antd'
import { ReloadOutlined } from '@ant-design/icons'

type EventRow = {
  id: string
  event_id: string
  experiment_id: string
  variant_id?: string | null
  user_id: string
  session_id?: string | null
  event_type: string
  event_name: string
  event_category?: string | null
  event_timestamp: string
  server_timestamp: string
  status: string
  data_quality: string
  properties?: any
}

const EventTrackingPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [events, setEvents] = useState<EventRow[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(50)

  const [experimentIds, setExperimentIds] = useState('')
  const [userIds, setUserIds] = useState('')
  const [eventNames, setEventNames] = useState('')
  const [eventType, setEventType] = useState<string>('all')
  const [status, setStatus] = useState<string>('all')
  const [dataQuality, setDataQuality] = useState<string>('all')

  const [selected, setSelected] = useState<any>(null)
  const [drawerOpen, setDrawerOpen] = useState(false)

  const [createPayload, setCreatePayload] = useState(
    '{\n  "experiment_id": "exp_001",\n  "user_id": "user_001",\n  "event_type": "interaction",\n  "event_name": "button_click",\n  "properties": { "button_id": "cta" }\n}',
  )
  const [batchPayload, setBatchPayload] = useState(
    '{\n  "events": [\n    {\n      "experiment_id": "exp_001",\n      "user_id": "user_001",\n      "event_type": "interaction",\n      "event_name": "button_click"\n    }\n  ]\n}',
  )
  const [statsExperimentId, setStatsExperimentId] = useState('')
  const [statsHours, setStatsHours] = useState('24')
  const [statsResult, setStatsResult] = useState<string>('')

  const load = async (nextPage = page, nextPageSize = pageSize) => {
    setLoading(true)
    try {
      const params = new URLSearchParams({
        page: String(nextPage),
        page_size: String(nextPageSize),
        sort_by: 'event_timestamp',
        sort_order: 'desc',
      })

      experimentIds
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .forEach((v) => params.append('experiment_ids', v))
      userIds
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .forEach((v) => params.append('user_ids', v))
      eventNames
        .split(',')
        .map((s) => s.trim())
        .filter(Boolean)
        .forEach((v) => params.append('event_names', v))

      if (eventType !== 'all') params.append('event_types', eventType)
      if (status !== 'all') params.append('status', status)
      if (dataQuality !== 'all') params.append('data_quality', dataQuality)

      const res = await apiFetch(buildApiUrl(`/api/v1/event-tracking/events?${params.toString()}`))
      const data = await res.json().catch(() => null)

      setEvents(Array.isArray(data?.events) ? data.events : [])
      setTotal(Number(data?.total_count || 0))
      setPage(Number(data?.page || nextPage))
      setPageSize(Number(data?.page_size || nextPageSize))
    } catch (e: any) {
      message.error(e?.message || '加载失败')
      setEvents([])
      setTotal(0)
    } finally {
      setLoading(false)
    }
  }

  const createEvent = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/event-tracking/events'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: createPayload,
      })
      const data = await res.json().catch(() => null)
      message.success('事件已提交')
      await load(1, pageSize)
    } catch (e: any) {
      message.error(e?.message || '提交失败')
    } finally {
      setLoading(false)
    }
  }

  const createBatch = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(buildApiUrl('/api/v1/event-tracking/events/batch'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: batchPayload,
      })
      const data = await res.json().catch(() => null)
      message.success('批量事件已提交')
      await load(1, pageSize)
    } catch (e: any) {
      message.error(e?.message || '提交失败')
    } finally {
      setLoading(false)
    }
  }

  const loadStats = async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams({ hours: statsHours || '24' })
      if (statsExperimentId.trim()) params.set('experiment_id', statsExperimentId.trim())
      const res = await apiFetch(buildApiUrl(`/api/v1/event-tracking/stats?${params.toString()}`))
      const data = await res.json().catch(() => null)
      setStatsResult(JSON.stringify(data, null, 2))
    } catch (e: any) {
      message.error(e?.message || '加载失败')
      setStatsResult('')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load(1, pageSize)
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Typography.Title level={3} style={{ margin: 0 }}>
            事件追踪
          </Typography.Title>
          <Button icon={<ReloadOutlined />} onClick={() => load(1, pageSize)} loading={loading}>
            刷新
          </Button>
        </Space>

        <Tabs
          items={[
            {
              key: 'query',
              label: '查询',
              children: (
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                  <Card title="筛选">
                    <Space wrap>
                      <Input
                        value={experimentIds}
                        onChange={(e) => setExperimentIds(e.target.value)}
                        placeholder="experiment_ids (逗号分隔)"
                        style={{ width: 260 }}
                      />
                      <Input
                        value={userIds}
                        onChange={(e) => setUserIds(e.target.value)}
                        placeholder="user_ids (逗号分隔)"
                        style={{ width: 220 }}
                      />
                      <Input
                        value={eventNames}
                        onChange={(e) => setEventNames(e.target.value)}
                        placeholder="event_names (逗号分隔)"
                        style={{ width: 220 }}
                      />
                      <Select
                        value={eventType}
                        onChange={setEventType}
                        style={{ width: 160 }}
                        options={[
                          { value: 'all', label: 'event_type: all' },
                          { value: 'assignment', label: 'assignment' },
                          { value: 'exposure', label: 'exposure' },
                          { value: 'interaction', label: 'interaction' },
                          { value: 'conversion', label: 'conversion' },
                          { value: 'custom', label: 'custom' },
                          { value: 'system', label: 'system' },
                          { value: 'error', label: 'error' },
                        ]}
                      />
                      <Select
                        value={status}
                        onChange={setStatus}
                        style={{ width: 160 }}
                        options={[
                          { value: 'all', label: 'status: all' },
                          { value: 'pending', label: 'pending' },
                          { value: 'processed', label: 'processed' },
                          { value: 'aggregated', label: 'aggregated' },
                          { value: 'archived', label: 'archived' },
                          { value: 'failed', label: 'failed' },
                          { value: 'duplicate', label: 'duplicate' },
                        ]}
                      />
                      <Select
                        value={dataQuality}
                        onChange={setDataQuality}
                        style={{ width: 170 }}
                        options={[
                          { value: 'all', label: 'quality: all' },
                          { value: 'high', label: 'high' },
                          { value: 'medium', label: 'medium' },
                          { value: 'low', label: 'low' },
                          { value: 'invalid', label: 'invalid' },
                        ]}
                      />
                      <Button type="primary" onClick={() => load(1, pageSize)} loading={loading}>
                        查询
                      </Button>
                    </Space>
                  </Card>

                  <Card>
                    <Table
                      rowKey="event_id"
                      dataSource={events}
                      loading={loading}
                      pagination={{
                        current: page,
                        pageSize,
                        total,
                        showSizeChanger: true,
                        onChange: (p, ps) => load(p, ps),
                      }}
                      columns={[
                        { title: 'event_id', dataIndex: 'event_id' },
                        { title: 'experiment', dataIndex: 'experiment_id' },
                        { title: 'user', dataIndex: 'user_id' },
                        { title: 'type', dataIndex: 'event_type' },
                        { title: 'name', dataIndex: 'event_name' },
                        { title: 'status', dataIndex: 'status' },
                        { title: 'quality', dataIndex: 'data_quality' },
                        { title: 'ts', dataIndex: 'event_timestamp' },
                        {
                          title: '操作',
                          render: (_, r: EventRow) => (
                            <Button
                              size="small"
                              onClick={() => {
                                setSelected(r)
                                setDrawerOpen(true)
                              }}
                            >
                              详情
                            </Button>
                          ),
                        },
                      ]}
                    />
                  </Card>
                </Space>
              ),
            },
            {
              key: 'create',
              label: '创建事件',
              children: (
                <Card title="POST /api/v1/event-tracking/events">
                  <Input.TextArea rows={10} value={createPayload} onChange={(e) => setCreatePayload(e.target.value)} />
                  <Button type="primary" onClick={createEvent} loading={loading} style={{ marginTop: 12 }}>
                    提交
                  </Button>
                </Card>
              ),
            },
            {
              key: 'batch',
              label: '批量创建',
              children: (
                <Card title="POST /api/v1/event-tracking/events/batch">
                  <Input.TextArea rows={12} value={batchPayload} onChange={(e) => setBatchPayload(e.target.value)} />
                  <Button type="primary" onClick={createBatch} loading={loading} style={{ marginTop: 12 }}>
                    提交
                  </Button>
                </Card>
              ),
            },
            {
              key: 'stats',
              label: '统计',
              children: (
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                  <Card title="GET /api/v1/event-tracking/stats">
                    <Space wrap>
                      <Input
                        value={statsExperimentId}
                        onChange={(e) => setStatsExperimentId(e.target.value)}
                        placeholder="experiment_id (可选)"
                        style={{ width: 220 }}
                      />
                      <Input value={statsHours} onChange={(e) => setStatsHours(e.target.value)} placeholder="hours" style={{ width: 120 }} />
                      <Button type="primary" onClick={loadStats} loading={loading}>
                        查询
                      </Button>
                    </Space>
                  </Card>
                  <Card title="结果">
                    <pre style={{ whiteSpace: 'pre-wrap' }}>{statsResult || '暂无数据'}</pre>
                  </Card>
                </Space>
              ),
            },
          ]}
        />

        <Drawer title="事件详情" open={drawerOpen} onClose={() => setDrawerOpen(false)} width={720} destroyOnClose>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{selected ? JSON.stringify(selected, null, 2) : ''}</pre>
        </Drawer>
      </Space>
    </div>
  )
}

export default EventTrackingPage
