import React, { useEffect, useState } from 'react'
import {
  Button,
  Card,
  Form,
  Input,
  InputNumber,
  Select,
  Space,
  Table,
  Typography,
  message,
} from 'antd'
import type { ColumnsType } from 'antd/es/table'
import apiClient from '../services/apiClient'

const { Title, Paragraph, Text } = Typography
const { Option } = Select

type Strategy = {
  strategy_name: string
  description: string
  target_scenario: string
  compression_methods: string[]
  expected_compression_ratio: number
  expected_speedup: number
  expected_accuracy_retention: number
  hardware_compatibility: string[]
  config_template: any
}

const CompressionStrategyRecommendationPage: React.FC = () => {
  const [form] = Form.useForm()
  const [loading, setLoading] = useState(false)
  const [strategies, setStrategies] = useState<Strategy[]>([])
  const [recommended, setRecommended] = useState<Strategy[]>([])

  const loadStrategies = async () => {
    try {
      const resp = await apiClient.get<Strategy[]>(
        '/model-compression/strategies'
      )
      setStrategies(Array.isArray(resp.data) ? resp.data : [])
    } catch (e: any) {
      message.error(e?.message || '加载策略失败')
      setStrategies([])
    }
  }

  useEffect(() => {
    loadStrategies()
  }, [])

  const recommend = async (values: any) => {
    setLoading(true)
    try {
      const payload = {
        model_name: values.model_name,
        model_type: values.model_type,
        num_parameters: values.num_parameters,
        target_scenario: values.target_scenario,
        accuracy_tolerance: values.accuracy_tolerance,
        size_reduction_target: values.size_reduction_target,
      }
      const resp = await apiClient.post<Strategy[]>(
        '/model-compression/strategies/recommend',
        payload
      )
      setRecommended(Array.isArray(resp.data) ? resp.data : [])
      message.success('推荐完成')
    } catch (e: any) {
      message.error(e?.message || '推荐失败')
      setRecommended([])
    } finally {
      setLoading(false)
    }
  }

  const columns: ColumnsType<Strategy> = [
    { title: '名称', dataIndex: 'strategy_name', key: 'strategy_name' },
    { title: '场景', dataIndex: 'target_scenario', key: 'target_scenario' },
    {
      title: '方法',
      dataIndex: 'compression_methods',
      key: 'compression_methods',
      render: (v: string[]) => (Array.isArray(v) ? v.join(', ') : ''),
    },
    {
      title: '压缩比',
      dataIndex: 'expected_compression_ratio',
      key: 'expected_compression_ratio',
      render: (v: number) => v?.toFixed?.(2),
    },
    {
      title: '加速比',
      dataIndex: 'expected_speedup',
      key: 'expected_speedup',
      render: (v: number) => v?.toFixed?.(2),
    },
    {
      title: '精度保持',
      dataIndex: 'expected_accuracy_retention',
      key: 'expected_accuracy_retention',
      render: (v: number) => `${(v * 100).toFixed(1)}%`,
    },
  ]

  return (
    <div style={{ padding: 24 }}>
      <Title level={2} style={{ margin: 0 }}>
        压缩策略推荐
      </Title>
      <Paragraph style={{ marginTop: 8 }}>
        策略模板：`/api/v1/model-compression/strategies`，推荐：`/api/v1/model-compression/strategies/recommend`。
      </Paragraph>

      <Card title="策略推荐" style={{ marginBottom: 16 }}>
        <Form
          form={form}
          layout="inline"
          onFinish={recommend}
          initialValues={{
            target_scenario: 'cloud',
            accuracy_tolerance: 0.05,
            size_reduction_target: 0.5,
            model_type: 'transformer',
          }}
        >
          <Form.Item
            name="model_name"
            rules={[{ required: true, message: '请输入模型名称' }]}
          >
            <Input placeholder="model_name" style={{ width: 200 }} />
          </Form.Item>
          <Form.Item
            name="model_type"
            rules={[{ required: true, message: '请选择模型类型' }]}
          >
            <Select style={{ width: 160 }}>
              <Option value="transformer">transformer</Option>
              <Option value="cnn">cnn</Option>
              <Option value="rnn">rnn</Option>
              <Option value="other">other</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="num_parameters"
            rules={[{ required: true, message: '请输入参数数量' }]}
          >
            <InputNumber
              min={1}
              style={{ width: 160 }}
              placeholder="num_parameters"
            />
          </Form.Item>
          <Form.Item name="target_scenario" rules={[{ required: true }]}>
            <Select style={{ width: 140 }}>
              <Option value="cloud">cloud</Option>
              <Option value="edge">edge</Option>
              <Option value="mobile">mobile</Option>
            </Select>
          </Form.Item>
          <Form.Item name="accuracy_tolerance" rules={[{ required: true }]}>
            <InputNumber
              min={0}
              max={1}
              step={0.01}
              style={{ width: 140 }}
              placeholder="accuracy_tolerance"
            />
          </Form.Item>
          <Form.Item name="size_reduction_target" rules={[{ required: true }]}>
            <InputNumber
              min={0}
              max={1}
              step={0.01}
              style={{ width: 170 }}
              placeholder="size_reduction_target"
            />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                获取推荐
              </Button>
              <Button
                onClick={() => {
                  setRecommended([])
                  form.resetFields()
                }}
                disabled={loading}
              >
                重置
              </Button>
            </Space>
          </Form.Item>
        </Form>
        <Text type="secondary" style={{ fontSize: 12 }}>
          说明：推荐结果来自后端策略筛选逻辑，不做本地模拟与降级。
        </Text>
      </Card>

      <Card
        title="推荐结果"
        style={{ marginBottom: 16 }}
        extra={<Text type="secondary">{recommended.length} 条</Text>}
      >
        <Table
          rowKey="strategy_name"
          dataSource={recommended}
          columns={columns}
          pagination={{ pageSize: 10 }}
        />
      </Card>

      <Card
        title="预定义策略模板"
        extra={
          <Space>
            <Button onClick={() => loadStrategies()} size="small">
              刷新
            </Button>
            <Text type="secondary">{strategies.length} 条</Text>
          </Space>
        }
      >
        <Table
          rowKey="strategy_name"
          dataSource={strategies}
          columns={columns}
          pagination={{ pageSize: 10 }}
        />
      </Card>
    </div>
  )
}

export default CompressionStrategyRecommendationPage
