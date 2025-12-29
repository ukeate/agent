import React, { useEffect, useState } from 'react'
import { Card, Table, Button, Space, Typography, Tag, Form, Input, Select, InputNumber, message } from 'antd'
import { ReloadOutlined, PlusOutlined } from '@ant-design/icons'
import { buildApiUrl, apiFetch } from '../utils/apiBase'

type EvaluationMetric = {
  id?: string
  name: string
  category?: string
  description?: string
  higherBetter?: boolean
  status?: string
  weight?: number
}

type MetricsConfiguration = {
  id?: string
  name: string
  description?: string
  metrics: Array<{ metricId: string; weight: number }>
  status?: string
}

const { Option } = Select

const EvaluationMetricsConfigPage: React.FC = () => {
  const [metrics, setMetrics] = useState<EvaluationMetric[]>([])
  const [configs, setConfigs] = useState<MetricsConfiguration[]>([])
  const [loading, setLoading] = useState(false)
  const [metricForm] = Form.useForm()
  const [configForm] = Form.useForm()

  const load = async () => {
    setLoading(true)
    try {
      const [mRes, cRes] = await Promise.all([
        apiFetch(buildApiUrl('/model-evaluation/metrics')),
        apiFetch(buildApiUrl('/model-evaluation/metrics/configurations'))
      ])
      setMetrics((await mRes.json()) || [])
      setConfigs((await cRes.json()) || [])
    } catch (e: any) {
      message.error(e?.message || '加载失败')
      setMetrics([])
      setConfigs([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  const addMetric = async (values: any) => {
    try {
      const res = await apiFetch(buildApiUrl('/model-evaluation/metrics'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })
      message.success('新增指标成功')
      metricForm.resetFields()
      load()
    } catch (e: any) {
      message.error(e?.message || '新增指标失败')
    }
  }

  const addConfig = async (values: any) => {
    try {
      const res = await apiFetch(buildApiUrl('/model-evaluation/metrics/configurations'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(values)
      })
      message.success('新增配置成功')
      configForm.resetFields()
      load()
    } catch (e: any) {
      message.error(e?.message || '新增配置失败')
    }
  }

  const metricColumns = [
    { title: '名称', dataIndex: 'name' },
    { title: '类别', dataIndex: 'category' },
    { title: '状态', dataIndex: 'status', render: (s: string) => <Tag>{s}</Tag> },
    { title: '权重', dataIndex: 'weight' }
  ]

  const configColumns = [
    { title: '配置名', dataIndex: 'name' },
    { title: '状态', dataIndex: 'status', render: (s: string) => <Tag>{s}</Tag> },
    { title: '指标数', dataIndex: 'metrics', render: (arr: any[]) => arr?.length || 0 }
  ]

  return (
    <div style={{ padding: 24 }}>
      <Space direction="vertical" style={{ width: '100%' }} size="large">
        <Space align="center" style={{ justifyContent: 'space-between', width: '100%' }}>
          <Typography.Title level={3} style={{ margin: 0 }}>
            评估指标配置
          </Typography.Title>
          <Button icon={<ReloadOutlined />} onClick={load} loading={loading}>
            刷新
          </Button>
        </Space>

        <Card title="新增指标">
          <Form layout="inline" form={metricForm} onFinish={addMetric}>
            <Form.Item name="name" rules={[{ required: true, message: '必填' }]}>
              <Input placeholder="accuracy" />
            </Form.Item>
            <Form.Item name="category">
              <Input placeholder="accuracy/fluency/..." />
            </Form.Item>
            <Form.Item name="weight" initialValue={1}>
              <InputNumber min={0} step={0.1} />
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit" icon={<PlusOutlined />} loading={loading}>
                新增
              </Button>
            </Form.Item>
          </Form>
        </Card>

        <Card title="新增配置">
          <Form layout="vertical" form={configForm} onFinish={addConfig}>
            <Form.Item name="name" label="名称" rules={[{ required: true }]}>
              <Input />
            </Form.Item>
            <Form.Item name="description" label="描述">
              <Input />
            </Form.Item>
            <Form.Item name="metrics" label="指标">
              <Select mode="multiple" placeholder="选择指标" style={{ width: '100%' }}>
                {metrics.map((m) => (
                  <Option key={m.id || m.name} value={m.id || m.name}>
                    {m.name}
                  </Option>
                ))}
              </Select>
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit" icon={<PlusOutlined />} loading={loading}>
                新增配置
              </Button>
            </Form.Item>
          </Form>
        </Card>

        <Card title="指标列表">
          <Table rowKey={(r) => r.id || r.name} dataSource={metrics} columns={metricColumns} loading={loading} />
        </Card>

        <Card title="配置列表">
          <Table rowKey={(r) => r.id || r.name} dataSource={configs} columns={configColumns} loading={loading} />
        </Card>
      </Space>
    </div>
  )
}

export default EvaluationMetricsConfigPage
