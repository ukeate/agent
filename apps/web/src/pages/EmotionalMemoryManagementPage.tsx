import React, { useEffect, useMemo, useState } from 'react'
import { Button, Card, Descriptions, Form, Input, InputNumber, Modal, Select, Space, Table, Typography, message } from 'antd'

import authService from '../services/authService'
import {
  emotionalMemoryService,
  EmotionType,
  StorageLayer,
  type EmotionalEvent,
  type EmotionalMemory,
  type MemoryStatistics,
  type TriggerPattern,
  type UserPreference,
} from '../services/emotionalMemoryService'

const { Title, Text } = Typography

export default function EmotionalMemoryManagementPage() {
  const [loading, setLoading] = useState(false)
  const [userId, setUserId] = useState<string | null>(null)
  const [username, setUsername] = useState<string | null>(null)

  const [stats, setStats] = useState<MemoryStatistics | null>(null)
  const [memories, setMemories] = useState<EmotionalMemory[]>([])
  const [events, setEvents] = useState<EmotionalEvent[]>([])
  const [preferences, setPreferences] = useState<UserPreference | null>(null)
  const [patterns, setPatterns] = useState<TriggerPattern[]>([])

  const [createOpen, setCreateOpen] = useState(false)
  const [predictOpen, setPredictOpen] = useState(false)
  const [predictResult, setPredictResult] = useState<Record<string, any> | null>(null)

  const [searchResults, setSearchResults] = useState<any[] | null>(null)

  const [form] = Form.useForm()
  const [predictForm] = Form.useForm()

  const emotionOptions = useMemo(
    () => Object.values(EmotionType).map((v) => ({ value: v, label: v })),
    []
  )
  const storageOptions = useMemo(
    () => Object.values(StorageLayer).map((v) => ({ value: v, label: v })),
    []
  )

  const loadAll = async (uid: string) => {
    setLoading(true)
    try {
      const [s, ms, es, ps, pats] = await Promise.all([
        emotionalMemoryService.getStatistics(uid),
        emotionalMemoryService.listMemories({ limit: 20 }),
        emotionalMemoryService.getEvents(uid, { limit: 10, offset: 0 }),
        emotionalMemoryService.getPreferences(uid),
        emotionalMemoryService.getTriggerPatterns(uid, { confidence_min: 0.0 }),
      ])
      setStats(s)
      setMemories(ms)
      setEvents(es)
      setPreferences(ps)
      setPatterns(pats)
      setSearchResults(null)
    } catch (e: any) {
      message.error(e?.message || '加载失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    ;(async () => {
      try {
        if (!authService.isAuthenticated()) {
          setUserId(null)
          setUsername(null)
          message.info('请先登录')
          return
        }
        const u = await authService.getCurrentUser()
        setUserId(u.id)
        setUsername(u.username)
        await loadAll(u.id)
      } catch (e: any) {
        if (e?.response?.status === 401) {
          setUserId(null)
          setUsername(null)
          message.info('请先登录')
          return
        }
        message.error(e?.message || '加载失败')
      }
    })()
  }, [])

  const memoryColumns = useMemo(
    () => [
      { title: '时间', dataIndex: 'timestamp', width: 180, render: (v: string) => (v ? new Date(v).toLocaleString() : '-') },
      { title: '情感', dataIndex: 'emotion_type', width: 120 },
      {
        title: '强度',
        dataIndex: 'intensity',
        width: 90,
        render: (v: number) => (typeof v === 'number' ? v.toFixed(2) : '-'),
      },
      { title: '内容', dataIndex: 'content', ellipsis: true },
      { title: '存储层', dataIndex: 'storage_layer', width: 100 },
      {
        title: '重要度',
        dataIndex: 'importance_score',
        width: 100,
        render: (v: number) => (typeof v === 'number' ? v.toFixed(2) : '-'),
      },
      {
        title: '操作',
        key: 'actions',
        width: 140,
        render: (_: any, row: EmotionalMemory) => (
          <Space>
            <Button
              size="small"
              danger
              onClick={async () => {
                if (!userId) return
                setLoading(true)
                try {
                  await emotionalMemoryService.deleteMemory(userId, row.id)
                  message.success('已删除')
                  await loadAll(userId)
                } catch (e: any) {
                  message.error(e?.message || '删除失败')
                } finally {
                  setLoading(false)
                }
              }}
            >
              删除
            </Button>
          </Space>
        ),
      },
    ],
    [userId]
  )

  const eventColumns = useMemo(
    () => [
      { title: '时间', dataIndex: 'timestamp', width: 180, render: (v: string) => (v ? new Date(v).toLocaleString() : '-') },
      { title: '类型', dataIndex: 'event_type', width: 140 },
      { title: '触发源', dataIndex: 'trigger_source', width: 140, ellipsis: true },
      { title: '影响', dataIndex: 'impact_score', width: 90 },
    ],
    []
  )

  const patternColumns = useMemo(
    () => [
      { title: '名称', dataIndex: 'pattern_name', ellipsis: true },
      { title: '类型', dataIndex: 'pattern_type', width: 140 },
      { title: '频率', dataIndex: 'frequency', width: 90 },
      { title: '置信度', dataIndex: 'confidence', width: 90 },
      { title: '最近触发', dataIndex: 'last_triggered', width: 180, render: (v: string) => (v ? new Date(v).toLocaleString() : '-') },
    ],
    []
  )

  const searchColumns = useMemo(
    () => [
      { title: '相关度', dataIndex: 'relevance_score', width: 90, render: (v: number) => (typeof v === 'number' ? v.toFixed(3) : '-') },
      {
        title: '情感',
        key: 'emotion',
        width: 120,
        render: (_: any, row: any) => row?.memory?.emotion_type || '-',
      },
      {
        title: '内容',
        key: 'content',
        ellipsis: true,
        render: (_: any, row: any) => row?.memory?.content || '-',
      },
    ],
    []
  )

  return (
    <div style={{ padding: 16 }}>
      <Space direction="vertical" size="large" style={{ width: '100%' }}>
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Title level={3} style={{ margin: 0 }}>
            情感记忆管理
          </Title>
          <Space>
            <Button disabled={!userId} loading={loading} onClick={() => userId && loadAll(userId)}>
              刷新
            </Button>
            <Button disabled={!userId} loading={loading} onClick={() => setCreateOpen(true)} type="primary">
              创建记忆
            </Button>
            <Button
              disabled={!userId}
              loading={loading}
              onClick={async () => {
                if (!userId) return
                setLoading(true)
                try {
                  await emotionalMemoryService.detectEvents(24)
                  message.success('已触发事件检测')
                  await loadAll(userId)
                } catch (e: any) {
                  message.error(e?.message || '事件检测失败')
                } finally {
                  setLoading(false)
                }
              }}
            >
              检测事件
            </Button>
            <Button
              disabled={!userId}
              loading={loading}
              onClick={async () => {
                if (!userId) return
                setLoading(true)
                try {
                  await emotionalMemoryService.learnPreferences()
                  message.success('已触发偏好学习')
                  await loadAll(userId)
                } catch (e: any) {
                  message.error(e?.message || '偏好学习失败')
                } finally {
                  setLoading(false)
                }
              }}
            >
              学习偏好
            </Button>
            <Button
              disabled={!userId}
              loading={loading}
              onClick={async () => {
                if (!userId) return
                setLoading(true)
                try {
                  await emotionalMemoryService.optimizeStorage()
                  message.success('已触发存储优化')
                } catch (e: any) {
                  message.error(e?.message || '存储优化失败')
                } finally {
                  setLoading(false)
                }
              }}
            >
              优化存储
            </Button>
            <Button
              disabled={!userId}
              loading={loading}
              onClick={async () => {
                if (!userId) return
                setLoading(true)
                try {
                  await emotionalMemoryService.identifyPatterns(userId, 3)
                  message.success('已触发模式识别')
                  await loadAll(userId)
                } catch (e: any) {
                  message.error(e?.message || '模式识别失败')
                } finally {
                  setLoading(false)
                }
              }}
            >
              识别模式
            </Button>
            <Button
              disabled={!userId}
              loading={loading}
              onClick={async () => {
                if (!userId) return
                setLoading(true)
                try {
                  const out = await emotionalMemoryService.exportMemories(userId, 'json')
                  message.success(`已导出：${out.count}`)
                } catch (e: any) {
                  message.error(e?.message || '导出失败')
                } finally {
                  setLoading(false)
                }
              }}
            >
              导出
            </Button>
            <Button
              disabled={!userId}
              onClick={() => {
                setPredictResult(null)
                setPredictOpen(true)
                predictForm.setFieldsValue({ context: '{}' })
              }}
            >
              风险预测
            </Button>
          </Space>
        </Space>

        <Text type="secondary">
          当前用户：{username || '-'}（{userId || '-'}）
        </Text>

        <Card title="统计" loading={loading}>
          {stats ? (
            <Descriptions size="small" column={4}>
              <Descriptions.Item label="总记忆数">{stats.total_memories}</Descriptions.Item>
              <Descriptions.Item label="平均强度">{stats.avg_intensity.toFixed(3)}</Descriptions.Item>
              <Descriptions.Item label="存储分布">{JSON.stringify(stats.storage_distribution)}</Descriptions.Item>
              <Descriptions.Item label="情感分布">{JSON.stringify(stats.emotion_distribution)}</Descriptions.Item>
            </Descriptions>
          ) : (
            <Text type="secondary">暂无数据</Text>
          )}
        </Card>

        <Card
          title="搜索"
          extra={
            <Input.Search
              allowClear
              name="memorySearch"
              placeholder="语义搜索（/memories/search）"
              style={{ width: 360 }}
              onSearch={async (v) => {
                if (!v.trim()) return
                setLoading(true)
                try {
                  const out = await emotionalMemoryService.searchMemories({ query: v.trim(), limit: 20 })
                  setSearchResults(out)
                  message.success(`结果：${out.length}`)
                } catch (e: any) {
                  message.error(e?.message || '搜索失败')
                } finally {
                  setLoading(false)
                }
              }}
            />
          }
        >
          {searchResults ? (
            <Table rowKey={(_, i) => String(i)} size="small" dataSource={searchResults} columns={searchColumns} pagination={{ pageSize: 10 }} />
          ) : (
            <Text type="secondary">未搜索</Text>
          )}
        </Card>

        <Card title="记忆列表" loading={loading}>
          <Table rowKey="id" size="small" dataSource={memories} columns={memoryColumns} pagination={{ pageSize: 10 }} />
        </Card>

        <Card title="事件列表" loading={loading}>
          <Table rowKey="id" size="small" dataSource={events} columns={eventColumns} pagination={{ pageSize: 10 }} />
        </Card>

        <Card title="用户偏好" loading={loading}>
          {preferences ? (
            <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{JSON.stringify(preferences, null, 2)}</pre>
          ) : (
            <Text type="secondary">暂无数据</Text>
          )}
        </Card>

        <Card title="触发模式" loading={loading}>
          <Table rowKey="id" size="small" dataSource={patterns} columns={patternColumns} pagination={{ pageSize: 10 }} />
        </Card>
      </Space>

      <Modal
        title="创建记忆"
        open={createOpen}
        destroyOnClose
        confirmLoading={loading}
        onCancel={() => setCreateOpen(false)}
        onOk={() => form.submit()}
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={{ intensity: 0.6, importance_score: 0.5, emotion_type: EmotionType.CALM, storage_layer: StorageLayer.HOT }}
          onFinish={async (values) => {
            if (!userId) return
            setLoading(true)
            try {
              await emotionalMemoryService.createMemory(userId, {
                emotion_type: values.emotion_type,
                intensity: values.intensity,
                context: values.context,
                importance_score: values.importance_score,
                storage_layer: values.storage_layer,
                tags: String(values.tags || '')
                  .split(',')
                  .map((s) => s.trim())
                  .filter(Boolean),
              })
              message.success('已创建')
              setCreateOpen(false)
              form.resetFields()
              await loadAll(userId)
            } catch (e: any) {
              message.error(e?.message || '创建失败')
            } finally {
              setLoading(false)
            }
          }}
        >
          <Form.Item name="emotion_type" label="情感类型" rules={[{ required: true, message: '必填' }]}>
            <Select options={emotionOptions} />
          </Form.Item>
          <Form.Item name="intensity" label="强度（0-1）" rules={[{ required: true, message: '必填' }]}>
            <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="importance_score" label="重要度（0-1）" rules={[{ required: true, message: '必填' }]}>
            <InputNumber min={0} max={1} step={0.01} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item name="storage_layer" label="存储层" rules={[{ required: true, message: '必填' }]}>
            <Select options={storageOptions} />
          </Form.Item>
          <Form.Item name="context" label="内容" rules={[{ required: true, message: '必填' }]}>
            <Input.TextArea rows={4} placeholder="context 字段（后端会作为 content 参与处理）" />
          </Form.Item>
          <Form.Item name="tags" label="标签（逗号分隔）">
            <Input placeholder="tag1,tag2" />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="风险预测"
        open={predictOpen}
        destroyOnClose
        confirmLoading={loading}
        onCancel={() => setPredictOpen(false)}
        onOk={() => predictForm.submit()}
      >
        <Form
          form={predictForm}
          layout="vertical"
          onFinish={async (values) => {
            if (!userId) return
            setLoading(true)
            try {
              const context = JSON.parse(values.context || '{}')
              const out = await emotionalMemoryService.predictRisk(userId, context)
              setPredictResult(out)
            } catch (e: any) {
              message.error(e?.message || '预测失败')
            } finally {
              setLoading(false)
            }
          }}
        >
          <Form.Item name="context" label="当前上下文（JSON）" rules={[{ required: true, message: '必填' }]}>
            <Input.TextArea rows={6} />
          </Form.Item>
        </Form>
        {predictResult ? <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{JSON.stringify(predictResult, null, 2)}</pre> : null}
      </Modal>
    </div>
  )
}
