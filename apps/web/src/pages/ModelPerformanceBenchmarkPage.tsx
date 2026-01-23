import React, { useState } from 'react'
import {
  Button,
  Card,
  Form,
  Input,
  Space,
  Table,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import apiClient from '../services/apiClient'

const { Title, Paragraph, Text } = Typography

type BenchmarkRow = {
  device_name: string
  device_type: string
  batch_size: number
  sequence_length: number
  throughput: number
  latency_p50: number
  latency_p95: number
  latency_p99: number
  memory_usage: number
  power_consumption?: number | null
}

const parseIntList = (value?: string) => {
  const parts = (value || '')
    .split(',')
    .map(s => s.trim())
    .filter(Boolean)
  const nums = parts
    .map(p => Number(p))
    .filter(n => Number.isFinite(n))
    .map(n => Math.trunc(n))
  return nums.length ? nums : undefined
}

const ModelPerformanceBenchmarkPage: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [rows, setRows] = useState<BenchmarkRow[]>([])

  const columns: ColumnsType<BenchmarkRow> = [
    { title: '设备', dataIndex: 'device_name', key: 'device_name' },
    { title: '类型', dataIndex: 'device_type', key: 'device_type' },
    { title: 'batch', dataIndex: 'batch_size', key: 'batch_size' },
    { title: 'seq', dataIndex: 'sequence_length', key: 'sequence_length' },
    {
      title: '吞吐(tokens/s)',
      dataIndex: 'throughput',
      key: 'throughput',
      render: (v: number) => v.toFixed(2),
    },
    {
      title: 'P50(ms)',
      dataIndex: 'latency_p50',
      key: 'latency_p50',
      render: (v: number) => v.toFixed(2),
    },
    {
      title: 'P95(ms)',
      dataIndex: 'latency_p95',
      key: 'latency_p95',
      render: (v: number) => v.toFixed(2),
    },
    {
      title: 'P99(ms)',
      dataIndex: 'latency_p99',
      key: 'latency_p99',
      render: (v: number) => v.toFixed(2),
    },
    {
      title: '内存(MB)',
      dataIndex: 'memory_usage',
      key: 'memory_usage',
      render: (v: number) => v.toFixed(2),
    },
  ]

  const run = async (values: any) => {
    setLoading(true)
    try {
      const payload: any = {
        model_path: values.model_path,
      }
      if (values.device_name) payload.device_name = values.device_name
      const seqs = parseIntList(values.sequence_lengths)
      const batches = parseIntList(values.batch_sizes)
      if (seqs) payload.sequence_lengths = seqs
      if (batches) payload.batch_sizes = batches

      const resp = await apiClient.post<BenchmarkRow[]>(
        '/model-compression/benchmark',
        payload
      )
      setRows(Array.isArray(resp.data) ? resp.data : [])
      message.success('基准测试完成')
    } catch (e: any) {
      message.error(e?.message || '基准测试失败')
      setRows([])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <Title level={2} style={{ margin: 0 }}>
        性能基准测试
      </Title>
      <Paragraph style={{ marginTop: 8 }}>
        调用 `/api/v1/model-compression/benchmark` 执行硬件基准测试。
      </Paragraph>

      <Card style={{ marginBottom: 16 }} title="测试配置">
        <Form
          form={form}
          layout="vertical"
          onFinish={run}
          initialValues={{
            sequence_lengths: '128,256,512',
            batch_sizes: '1,2,4',
          }}
        >
          <Form.Item
            name="model_path"
            label="模型路径"
            rules={[{ required: true, message: '请输入模型路径' }]}
          >
            <Input placeholder="/abs/path/to/model.pt" />
          </Form.Item>
          <Form.Item name="device_name" label="设备名称（可选）">
            <Input placeholder="cpu / cuda:0" />
          </Form.Item>
          <Form.Item name="sequence_lengths" label="序列长度（逗号分隔，可选）">
            <Input placeholder="128,256,512" />
          </Form.Item>
          <Form.Item name="batch_sizes" label="batch sizes（逗号分隔，可选）">
            <Input placeholder="1,2,4" />
          </Form.Item>
          <Space>
            <Button type="primary" htmlType="submit" loading={loading}>
              开始测试
            </Button>
            <Button onClick={() => form.resetFields()} disabled={loading}>
              重置
            </Button>
          </Space>
        </Form>
        <Text type="secondary" style={{ fontSize: 12 }}>
          注意：模型需要能被后端 `torch.load` 正常加载。
        </Text>
      </Card>

      <Card title="测试结果">
        <Table
          rowKey={(r, i) =>
            `${r.device_name}-${r.batch_size}-${r.sequence_length}-${i}`
          }
          dataSource={rows}
          columns={columns}
          pagination={{ pageSize: 20 }}
        />
      </Card>
    </div>
  )
}

export default ModelPerformanceBenchmarkPage
