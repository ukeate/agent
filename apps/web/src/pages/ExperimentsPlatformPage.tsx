import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Card,
  Descriptions,
  Divider,
  Form,
  Input,
  Modal,
  Select,
  Space,
  Table,
  Tag,
  Typography,
  Upload,
  message,
} from 'antd'
import type { UploadFile } from 'antd/es/upload/interface'
import {
  CloudDownloadOutlined,
  CommentOutlined,
  CopyOutlined,
  DeleteOutlined,
  PauseOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  ReloadOutlined,
  SettingOutlined,
  ShareAltOutlined,
  StopOutlined,
} from '@ant-design/icons'
import apiClient from '../services/apiClient'

const { Title, Text } = Typography

type ExperimentStatus =
  | 'draft'
  | 'running'
  | 'paused'
  | 'completed'
  | 'terminated'

interface ExperimentVariant {
  id: string
  name: string
  traffic: number
  isControl: boolean
  sampleSize?: number
  conversions?: number
  conversionRate?: number
}

interface Experiment {
  id: string
  name: string
  description: string
  type: string
  status: ExperimentStatus | string
  startDate?: string
  endDate?: string
  variants: ExperimentVariant[]
  metrics: any[]
  targetingRules: any[]
  sampleSize?: { current: number; required: number }
  confidenceLevel?: number
  tags?: string[]
  participants?: number
  total_conversions?: number
  conversion_rate?: number
  lift?: number
  created_at?: string
  updated_at?: string
}

interface ListExperimentsResponse {
  experiments: Experiment[]
  total: number
  page: number
  pageSize: number
}

function toPercent(value?: number) {
  if (typeof value !== 'number') return '-'
  return `${(value * 100).toFixed(2)}%`
}

function toNumberText(value?: number) {
  if (typeof value !== 'number') return '-'
  return value.toLocaleString()
}

export default function ExperimentsPlatformPage() {
  const [loading, setLoading] = useState(false)
  const [experiments, setExperiments] = useState<Experiment[]>([])
  const [selectedId, setSelectedId] = useState<string | null>(null)

  const [createOpen, setCreateOpen] = useState(false)
  const [createForm] = Form.useForm()

  const [updateOpen, setUpdateOpen] = useState(false)
  const [updateJson, setUpdateJson] = useState('{\n  \n}')

  const [settingsOpen, setSettingsOpen] = useState(false)
  const [settingsJson, setSettingsJson] = useState('{\n  \n}')

  const [apiTitle, setApiTitle] = useState<string>('输出')
  const [apiOutput, setApiOutput] = useState<any>(null)
  const [apiBusy, setApiBusy] = useState(false)

  const [monitoringMetric, setMonitoringMetric] = useState('conversion_rate')
  const [monitoringGranularity, setMonitoringGranularity] = useState<
    'hourly' | 'daily'
  >('hourly')

  const [sampleSizeBody, setSampleSizeBody] = useState({
    baselineRate: 0.1,
    minimumDetectableEffect: 0.02,
    confidenceLevel: 0.95,
    power: 0.8,
  })

  const [searchBody, setSearchBody] = useState({
    status: '' as string,
    owner: '' as string,
  })

  const [importFile, setImportFile] = useState<UploadFile | null>(null)

  const selectedExperiment = useMemo(
    () => experiments.find(e => e.id === selectedId) || null,
    [experiments, selectedId]
  )

  const fetchExperiments = async () => {
    setLoading(true)
    try {
      const res = await apiClient.get<ListExperimentsResponse>('/experiments', {
        params: {
          page: 1,
          pageSize: 50,
          sortBy: 'created_at',
          sortOrder: 'desc',
        },
      })
      const data: any = res.data || {}
      setExperiments(Array.isArray(data) ? data : data.experiments || [])
    } catch (e) {
      message.error((e as Error).message || '获取实验列表失败')
      setExperiments([])
      setSelectedId(null)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchExperiments()
  }, [])

  const callApi = async (title: string, fn: () => Promise<any>) => {
    setApiBusy(true)
    setApiTitle(title)
    setApiOutput(null)
    try {
      const data = await fn()
      setApiOutput(data)
    } catch (e) {
      setApiOutput({ error: (e as Error).message || String(e) })
    } finally {
      setApiBusy(false)
    }
  }

  const controlExperiment = async (id: string, action: string) => {
    await apiClient.post(`/experiments/${id}/${action}`)
    await fetchExperiments()
  }

  const onCreate = async (values: any) => {
    const tags = String(values.tags || '')
      .split(',')
      .map((s: string) => s.trim())
      .filter(Boolean)
    const metrics = String(values.metrics || '')
      .split(',')
      .map((s: string) => s.trim())
      .filter(Boolean)

    const res = await apiClient.post('/experiments/from-template', {
      templateId: 'ab_basic',
      overrides: {
        name: values.name,
        description: values.description,
        tags,
        metrics,
        confidenceLevel: values.confidenceLevel,
        power: values.power,
        sampleSize: values.sampleSize,
        status: 'draft',
      },
    })
    const created: any = res.data
    setCreateOpen(false)
    createForm.resetFields()
    await fetchExperiments()
    if (created?.id) setSelectedId(created.id)
  }

  const onDelete = async (id: string) => {
    await apiClient.delete(`/experiments/${id}`)
    if (selectedId === id) setSelectedId(null)
    await fetchExperiments()
  }

  const onClone = async (id: string) => {
    const name = window.prompt('克隆名称（可选）') || ''
    const res = await apiClient.post(
      `/experiments/${id}/clone`,
      name ? { name } : {}
    )
    const created: any = res.data
    await fetchExperiments()
    if (created?.id) setSelectedId(created.id)
  }

  const onShare = async (id: string) => {
    const raw = window.prompt('分享给用户ID（逗号分隔）')
    if (!raw) return
    const users = raw
      .split(',')
      .map(s => s.trim())
      .filter(Boolean)
    if (!users.length) return
    await apiClient.post(`/experiments/${id}/share`, { users })
    message.success('已提交分享')
  }

  const onComment = async (id: string) => {
    const text = window.prompt('评论内容')
    if (!text) return
    await apiClient.post(`/experiments/${id}/comments`, {
      text,
      type: 'comment',
    })
    message.success('已提交评论')
  }

  const openUpdate = (id: string) => {
    setSelectedId(id)
    setUpdateJson('{\n  \n}')
    setUpdateOpen(true)
  }

  const openSettings = (id: string) => {
    setSelectedId(id)
    setSettingsJson('{\n  \n}')
    setSettingsOpen(true)
  }

  const submitUpdate = async () => {
    if (!selectedExperiment) return
    const payload = JSON.parse(updateJson || '{}')
    await apiClient.put(`/experiments/${selectedExperiment.id}`, payload)
    setUpdateOpen(false)
    await fetchExperiments()
    message.success('已更新实验')
  }

  const submitSettings = async () => {
    if (!selectedExperiment) return
    const payload = JSON.parse(settingsJson || '{}')
    await apiClient.put(
      `/experiments/${selectedExperiment.id}/settings`,
      payload
    )
    setSettingsOpen(false)
    await fetchExperiments()
    message.success('已更新设置')
  }

  const exportOneGet = (id: string, format: 'json' | 'csv' | 'xlsx') => {
    window.open(`/api/v1/experiments/${id}/export?format=${format}`, '_blank')
  }

  const exportManyPost = async (ids: string[]) => {
    const res = await apiClient.post<Blob>(
      '/experiments/export',
      { ids },
      { responseType: 'blob' }
    )
    const url = window.URL.createObjectURL(res.data)
    const a = document.createElement('a')
    a.href = url
    a.download = 'experiments.json'
    a.click()
    window.URL.revokeObjectURL(url)
  }

  const importExperiments = async () => {
    if (!importFile?.originFileObj) {
      message.error('请选择要导入的文件')
      return
    }
    const formData = new FormData()
    formData.append('file', importFile.originFileObj)
    const res = await apiClient.post('/experiments/import', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    setImportFile(null)
    message.success(
      `导入完成：${res.data?.imported || 0} 成功，${res.data?.failed || 0} 失败`
    )
    await fetchExperiments()
  }

  const columns = [
    {
      title: '实验名称',
      dataIndex: 'name',
      key: 'name',
      render: (_: any, record: Experiment) => (
        <Button type="link" onClick={() => setSelectedId(record.id)}>
          {record.name}
        </Button>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          draft: 'default',
          running: 'processing',
          paused: 'warning',
          completed: 'success',
          terminated: 'default',
        }
        return (
          <Tag color={colorMap[status] || 'default'}>
            {String(status).toUpperCase()}
          </Tag>
        )
      },
    },
    {
      title: '变体数',
      dataIndex: 'variants',
      key: 'variants',
      render: (variants: ExperimentVariant[] = []) => variants?.length ?? 0,
    },
    {
      title: '参与人数',
      key: 'participants',
      render: (_: any, record: Experiment) =>
        toNumberText(record.participants ?? record.sampleSize?.current),
    },
    {
      title: '转化率',
      key: 'conversion_rate',
      render: (_: any, record: Experiment) => toPercent(record.conversion_rate),
    },
    {
      title: 'Lift',
      key: 'lift',
      render: (_: any, record: Experiment) =>
        typeof record.lift === 'number' ? `${record.lift.toFixed(2)}%` : '-',
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: Experiment) => (
        <Space size="small">
          {record.status === 'draft' && (
            <Button
              size="small"
              icon={<PlayCircleOutlined />}
              onClick={() => controlExperiment(record.id, 'start')}
            >
              启动
            </Button>
          )}
          {record.status === 'running' && (
            <>
              <Button
                size="small"
                icon={<PauseOutlined />}
                onClick={() => controlExperiment(record.id, 'pause')}
              >
                暂停
              </Button>
              <Button
                size="small"
                icon={<StopOutlined />}
                onClick={() => controlExperiment(record.id, 'stop')}
              >
                停止
              </Button>
            </>
          )}
          {record.status === 'paused' && (
            <>
              <Button
                size="small"
                icon={<PlayCircleOutlined />}
                onClick={() => controlExperiment(record.id, 'resume')}
              >
                恢复
              </Button>
              <Button
                size="small"
                icon={<StopOutlined />}
                onClick={() => controlExperiment(record.id, 'stop')}
              >
                停止
              </Button>
            </>
          )}
          <Button
            size="small"
            icon={<CopyOutlined />}
            onClick={() => onClone(record.id)}
          >
            克隆
          </Button>
          <Button
            size="small"
            icon={<ShareAltOutlined />}
            onClick={() => onShare(record.id)}
          >
            分享
          </Button>
          <Button
            size="small"
            icon={<CommentOutlined />}
            onClick={() => onComment(record.id)}
          >
            评论
          </Button>
          <Button
            size="small"
            icon={<SettingOutlined />}
            onClick={() => openSettings(record.id)}
          >
            设置
          </Button>
          <Button size="small" onClick={() => openUpdate(record.id)}>
            更新
          </Button>
          <Button
            size="small"
            danger
            icon={<DeleteOutlined />}
            onClick={() => onDelete(record.id)}
          >
            删除
          </Button>
        </Space>
      ),
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space style={{ width: '100%', justifyContent: 'space-between' }}>
        <Title level={2} style={{ margin: 0 }}>
          A/B测试实验平台
        </Title>
        <Space>
          <Button
            icon={<PlusOutlined />}
            type="primary"
            onClick={() => setCreateOpen(true)}
          >
            创建实验
          </Button>
          <Button
            icon={<ReloadOutlined />}
            onClick={fetchExperiments}
            loading={loading}
          >
            刷新
          </Button>
        </Space>
      </Space>

      <Divider />

      <Card title="实验列表">
        <Table
          rowKey="id"
          loading={loading}
          dataSource={experiments}
          columns={columns as any}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      <Divider />

      <Card
        title="实验详情"
        extra={
          selectedExperiment ? <Text code>{selectedExperiment.id}</Text> : null
        }
      >
        {selectedExperiment ? (
          <>
            <Descriptions bordered size="small" column={2}>
              <Descriptions.Item label="名称">
                {selectedExperiment.name}
              </Descriptions.Item>
              <Descriptions.Item label="状态">
                {String(selectedExperiment.status)}
              </Descriptions.Item>
              <Descriptions.Item label="类型">
                {selectedExperiment.type}
              </Descriptions.Item>
              <Descriptions.Item label="置信水平">
                {typeof selectedExperiment.confidenceLevel === 'number'
                  ? selectedExperiment.confidenceLevel
                  : '-'}
              </Descriptions.Item>
              <Descriptions.Item label="参与人数">
                {toNumberText(
                  selectedExperiment.participants ??
                    selectedExperiment.sampleSize?.current
                )}
              </Descriptions.Item>
              <Descriptions.Item label="总转化">
                {toNumberText(selectedExperiment.total_conversions)}
              </Descriptions.Item>
              <Descriptions.Item label="转化率">
                {toPercent(selectedExperiment.conversion_rate)}
              </Descriptions.Item>
              <Descriptions.Item label="Lift">
                {typeof selectedExperiment.lift === 'number'
                  ? `${selectedExperiment.lift.toFixed(2)}%`
                  : '-'}
              </Descriptions.Item>
            </Descriptions>

            <Divider />

            <Table
              size="small"
              rowKey="id"
              dataSource={selectedExperiment.variants || []}
              pagination={false}
              columns={[
                { title: '变体', dataIndex: 'name', key: 'name' },
                {
                  title: '对照组',
                  dataIndex: 'isControl',
                  key: 'isControl',
                  render: (v: boolean) => (v ? '是' : '否'),
                },
                {
                  title: '流量(%)',
                  dataIndex: 'traffic',
                  key: 'traffic',
                  render: (v: number) =>
                    typeof v === 'number' ? v.toFixed(2) : '-',
                },
                {
                  title: '样本量',
                  dataIndex: 'sampleSize',
                  key: 'sampleSize',
                  render: (v: number) => toNumberText(v),
                },
                {
                  title: '转化',
                  dataIndex: 'conversions',
                  key: 'conversions',
                  render: (v: number) => toNumberText(v),
                },
                {
                  title: '转化率',
                  dataIndex: 'conversionRate',
                  key: 'conversionRate',
                  render: (v: number) => toPercent(v),
                },
              ]}
            />

            <Divider />

            <Space wrap>
              <Button
                onClick={() =>
                  callApi(
                    '实验详情',
                    async () =>
                      (
                        await apiClient.get(
                          `/experiments/${selectedExperiment.id}`
                        )
                      ).data
                  )
                }
              >
                GET 详情
              </Button>
              <Button
                onClick={() =>
                  callApi(
                    '审计日志',
                    async () =>
                      (
                        await apiClient.get(
                          `/experiments/${selectedExperiment.id}/audit`
                        )
                      ).data
                  )
                }
              >
                GET 审计
              </Button>
              <Button
                onClick={() =>
                  callApi(
                    '事件流',
                    async () =>
                      (
                        await apiClient.get(
                          `/experiments/${selectedExperiment.id}/events`
                        )
                      ).data
                  )
                }
              >
                GET 事件
              </Button>
              <Button
                onClick={() =>
                  callApi(
                    '实时指标',
                    async () =>
                      (
                        await apiClient.get(
                          `/experiments/${selectedExperiment.id}/metrics`
                        )
                      ).data
                  )
                }
              >
                GET 指标
              </Button>
              <Button
                onClick={() =>
                  callApi(
                    '监控趋势',
                    async () =>
                      (
                        await apiClient.get(
                          `/experiments/${selectedExperiment.id}/monitoring?metric=${encodeURIComponent(
                            monitoringMetric
                          )}&granularity=${monitoringGranularity}`
                        )
                      ).data
                  )
                }
              >
                GET 监控
              </Button>
              <Select
                value={monitoringMetric}
                style={{ width: 180 }}
                onChange={v => setMonitoringMetric(v)}
                options={[
                  { value: 'conversion_rate', label: 'conversion_rate' },
                  {
                    value: 'conversion_rate_control',
                    label: 'conversion_rate_control',
                  },
                  {
                    value: 'conversion_rate_treatment',
                    label: 'conversion_rate_treatment',
                  },
                ]}
              />
              <Select
                value={monitoringGranularity}
                style={{ width: 120 }}
                onChange={v => setMonitoringGranularity(v)}
                options={[
                  { value: 'hourly', label: 'hourly' },
                  { value: 'daily', label: 'daily' },
                ]}
              />
              <Button
                onClick={() =>
                  callApi(
                    '统计分析',
                    async () =>
                      (
                        await apiClient.get(
                          `/experiments/${selectedExperiment.id}/analysis`
                        )
                      ).data
                  )
                }
              >
                GET 分析
              </Button>
              <Button
                onClick={() =>
                  callApi(
                    '成本分析',
                    async () =>
                      (
                        await apiClient.get(
                          `/experiments/${selectedExperiment.id}/cost-analysis`
                        )
                      ).data
                  )
                }
              >
                GET 成本
              </Button>
              <Button
                onClick={() =>
                  callApi(
                    '报告(json)',
                    async () =>
                      (
                        await apiClient.get(
                          `/experiments/${selectedExperiment.id}/report?format=json`
                        )
                      ).data
                  )
                }
              >
                GET 报告
              </Button>
              <Button
                icon={<CloudDownloadOutlined />}
                onClick={() => exportOneGet(selectedExperiment.id, 'json')}
              >
                导出 JSON
              </Button>
              <Button
                icon={<CloudDownloadOutlined />}
                onClick={() => exportOneGet(selectedExperiment.id, 'csv')}
              >
                导出 CSV
              </Button>
              <Button
                icon={<CloudDownloadOutlined />}
                onClick={() => exportOneGet(selectedExperiment.id, 'xlsx')}
              >
                导出 XLSX
              </Button>
              <Button
                icon={<CloudDownloadOutlined />}
                onClick={() => exportManyPost([selectedExperiment.id])}
              >
                批量导出(POST)
              </Button>
            </Space>
          </>
        ) : (
          <Alert message="请选择一个实验" type="info" showIcon />
        )}
      </Card>

      <Divider />

      <Card title="通用接口">
        <Space wrap>
          <Button
            onClick={() =>
              callApi(
                '模板列表',
                async () => (await apiClient.get('/experiments/templates')).data
              )
            }
          >
            GET /templates
          </Button>
          <Button
            onClick={() =>
              callApi('校验默认模板', async () => {
                const templates: any[] =
                  (await apiClient.get('/experiments/templates')).data || []
                const t = templates[0]
                const payload = {
                  ...(t?.default || {}),
                  name: 'validate',
                  description: 'validate',
                }
                return (await apiClient.post('/experiments/validate', payload))
                  .data
              })
            }
          >
            POST /validate
          </Button>
          <Button
            onClick={() =>
              callApi(
                '样本量计算',
                async () =>
                  (
                    await apiClient.post(
                      '/experiments/calculate-sample-size',
                      sampleSizeBody
                    )
                  ).data
              )
            }
          >
            POST /calculate-sample-size
          </Button>
          <Input
            style={{ width: 120 }}
            name="baselineRate"
            value={sampleSizeBody.baselineRate}
            onChange={e =>
              setSampleSizeBody({
                ...sampleSizeBody,
                baselineRate: Number(e.target.value),
              })
            }
            placeholder="baselineRate"
          />
          <Input
            style={{ width: 160 }}
            name="minimumDetectableEffect"
            value={sampleSizeBody.minimumDetectableEffect}
            onChange={e =>
              setSampleSizeBody({
                ...sampleSizeBody,
                minimumDetectableEffect: Number(e.target.value),
              })
            }
            placeholder="minimumDetectableEffect"
          />
          <Input
            style={{ width: 120 }}
            name="confidenceLevel"
            value={sampleSizeBody.confidenceLevel}
            onChange={e =>
              setSampleSizeBody({
                ...sampleSizeBody,
                confidenceLevel: Number(e.target.value),
              })
            }
            placeholder="confidenceLevel"
          />
          <Input
            style={{ width: 80 }}
            name="power"
            value={sampleSizeBody.power}
            onChange={e =>
              setSampleSizeBody({
                ...sampleSizeBody,
                power: Number(e.target.value),
              })
            }
            placeholder="power"
          />
        </Space>
        <Divider />
        <Space wrap>
          <Button
            onClick={() =>
              callApi('搜索实验', async () => {
                const filters: any = {}
                if (searchBody.status) filters.status = [searchBody.status]
                if (searchBody.owner) filters.owner = [searchBody.owner]
                return (
                  await apiClient.post('/experiments/search', {
                    filters,
                    pagination: { page: 1, limit: 20 },
                  })
                ).data
              })
            }
          >
            POST /search
          </Button>
          <Input
            style={{ width: 140 }}
            name="status"
            value={searchBody.status}
            onChange={e =>
              setSearchBody({ ...searchBody, status: e.target.value })
            }
            placeholder="status(draft/running/...)"
          />
          <Input
            style={{ width: 160 }}
            name="owner"
            value={searchBody.owner}
            onChange={e =>
              setSearchBody({ ...searchBody, owner: e.target.value })
            }
            placeholder="owner"
          />
        </Space>
        <Divider />
        <Space wrap>
          <Upload
            beforeUpload={file => {
              setImportFile(file as UploadFile)
              return false
            }}
            maxCount={1}
            fileList={importFile ? [importFile] : []}
            onRemove={() => {
              setImportFile(null)
            }}
          >
            <Button>选择导入文件</Button>
          </Upload>
          <Button type="primary" onClick={importExperiments}>
            导入(POST /import)
          </Button>
        </Space>
      </Card>

      <Divider />

      <Card title={apiTitle} loading={apiBusy}>
        <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
          {apiOutput ? JSON.stringify(apiOutput, null, 2) : '暂无输出'}
        </pre>
      </Card>

      <Modal
        title="创建实验（from-template: ab_basic）"
        open={createOpen}
        onCancel={() => setCreateOpen(false)}
        onOk={() => createForm.submit()}
        destroyOnClose
      >
        <Form
          form={createForm}
          layout="vertical"
          onFinish={onCreate}
          initialValues={{ confidenceLevel: 0.95, power: 0.8, sampleSize: 100 }}
        >
          <Form.Item
            name="name"
            label="实验名称"
            rules={[{ required: true, message: '请输入实验名称' }]}
          >
            <Input />
          </Form.Item>
          <Form.Item
            name="description"
            label="实验描述"
            rules={[{ required: true, message: '请输入实验描述' }]}
          >
            <Input.TextArea rows={3} />
          </Form.Item>
          <Form.Item name="metrics" label="指标（逗号分隔，可选）">
            <Input placeholder="conversion_rate, engagement" />
          </Form.Item>
          <Form.Item name="tags" label="标签（逗号分隔，可选）">
            <Input placeholder="tagA, tagB" />
          </Form.Item>
          <Form.Item name="confidenceLevel" label="置信水平">
            <Select
              options={[
                { value: 0.9, label: '90%' },
                { value: 0.95, label: '95%' },
                { value: 0.99, label: '99%' },
              ]}
            />
          </Form.Item>
          <Form.Item name="power" label="功效(power)">
            <Input />
          </Form.Item>
          <Form.Item name="sampleSize" label="最小样本量">
            <Input />
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="更新实验（PUT /experiments/{id}）"
        open={updateOpen}
        onCancel={() => setUpdateOpen(false)}
        onOk={submitUpdate}
        destroyOnClose
      >
        <Alert
          type="info"
          showIcon
          message="支持更新 name/description/endDate/metrics/targetingRules/tags 等字段"
        />
        <Divider />
        <Input.TextArea
          rows={12}
          value={updateJson}
          onChange={e => setUpdateJson(e.target.value)}
        />
      </Modal>

      <Modal
        title="更新设置（PUT /experiments/{id}/settings）"
        open={settingsOpen}
        onCancel={() => setSettingsOpen(false)}
        onOk={submitSettings}
        destroyOnClose
      >
        <Alert
          type="info"
          showIcon
          message="settings 会合并进 experiment.metadata_"
        />
        <Divider />
        <Input.TextArea
          rows={12}
          value={settingsJson}
          onChange={e => setSettingsJson(e.target.value)}
        />
      </Modal>
    </div>
  )
}
