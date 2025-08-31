import React, { useState, useEffect } from 'react'
import {
  Card,
  Form,
  Input,
  Button,
  Select,
  Space,
  Alert,
  Spin,
  Typography,
  Divider,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Progress,
  message,
  Tabs,
  InputNumber,
  Switch
} from 'antd'
import {
  PlayCircleOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  ThunderboltOutlined,
  BarChartOutlined,
  DatabaseOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

const { TextArea } = Input
const { Title, Text } = Typography
const { Option } = Select
const { TabPane } = Tabs

interface LoadedModel {
  model_key: string
  name: string
  version: string
  format: string
  load_time: number
  last_used: string
  is_healthy: boolean
}

interface InferenceRequest {
  request_id: string
  model_name: string
  model_version: string
  inputs: any
  status: 'pending' | 'processing' | 'completed' | 'failed'
  created_at: string
  processing_time_ms?: number
  outputs?: any
  error?: string
}

interface ModelMetrics {
  request_count: number
  success_count: number
  error_count: number
  error_rate: number
  avg_latency_ms: number
  p95_latency_ms: number
  throughput_qps: number
}

const ModelInferencePage: React.FC = () => {
  const [form] = Form.useForm()
  const [batchForm] = Form.useForm()
  
  const [loading, setLoading] = useState(false)
  const [loadedModels, setLoadedModels] = useState<LoadedModel[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [inferenceResults, setInferenceResults] = useState<InferenceRequest[]>([])
  const [metrics, setMetrics] = useState<Record<string, ModelMetrics>>({})
  const [activeTab, setActiveTab] = useState('single')

  useEffect(() => {
    loadModels()
    loadMetrics()
    const interval = setInterval(loadMetrics, 30000) // 每30秒刷新指标
    return () => clearInterval(interval)
  }, [])

  const loadModels = async () => {
    try {
      // 模拟API调用
      const mockModels: LoadedModel[] = [
        {
          model_key: 'bert-base-chinese:1.0.0',
          name: 'bert-base-chinese',
          version: '1.0.0',
          format: 'huggingface',
          load_time: 2.3,
          last_used: '2024-01-15T10:30:00Z',
          is_healthy: true
        },
        {
          model_key: 'resnet50-classifier:2.1.0',
          name: 'resnet50-classifier',
          version: '2.1.0',
          format: 'pytorch',
          load_time: 1.1,
          last_used: '2024-01-15T09:45:00Z',
          is_healthy: true
        },
        {
          model_key: 'yolo-v8-detection:1.5.0',
          name: 'yolo-v8-detection',
          version: '1.5.0',
          format: 'onnx',
          load_time: 0.8,
          last_used: '2024-01-15T08:20:00Z',
          is_healthy: false
        }
      ]
      setLoadedModels(mockModels)
    } catch (error) {
      message.error('加载模型列表失败')
    }
  }

  const loadMetrics = async () => {
    try {
      // 模拟指标数据
      const mockMetrics = {
        'bert-base-chinese:1.0.0': {
          request_count: 1250,
          success_count: 1245,
          error_count: 5,
          error_rate: 0.004,
          avg_latency_ms: 145.2,
          p95_latency_ms: 280.5,
          throughput_qps: 8.5
        },
        'resnet50-classifier:2.1.0': {
          request_count: 850,
          success_count: 840,
          error_count: 10,
          error_rate: 0.012,
          avg_latency_ms: 65.8,
          p95_latency_ms: 120.3,
          throughput_qps: 12.3
        },
        'yolo-v8-detection:1.5.0': {
          request_count: 340,
          success_count: 320,
          error_count: 20,
          error_rate: 0.059,
          avg_latency_ms: 95.4,
          p95_latency_ms: 180.7,
          throughput_qps: 5.2
        }
      }
      setMetrics(mockMetrics)
    } catch (error) {
      console.error('加载指标失败:', error)
    }
  }

  const handleSingleInference = async (values: any) => {
    setLoading(true)
    try {
      const requestId = `req-${Date.now()}`
      const newRequest: InferenceRequest = {
        request_id: requestId,
        model_name: values.model.split(':')[0],
        model_version: values.model.split(':')[1],
        inputs: JSON.parse(values.inputs),
        status: 'pending',
        created_at: new Date().toISOString()
      }
      
      setInferenceResults(prev => [newRequest, ...prev])

      // 模拟推理过程
      setTimeout(() => {
        const completed: InferenceRequest = {
          ...newRequest,
          status: 'completed',
          processing_time_ms: Math.random() * 200 + 50,
          outputs: {
            prediction: [0.8, 0.1, 0.1],
            confidence: 0.85,
            labels: ['类别A', '类别B', '类别C']
          }
        }
        
        setInferenceResults(prev => 
          prev.map(req => req.request_id === requestId ? completed : req)
        )
        message.success('推理完成')
      }, 1000 + Math.random() * 2000)

    } catch (error) {
      message.error('推理请求失败')
    }
    setLoading(false)
  }

  const handleBatchInference = async (values: any) => {
    setLoading(true)
    try {
      const batchSize = parseInt(values.batch_size)
      const requests: InferenceRequest[] = []
      
      for (let i = 0; i < batchSize; i++) {
        requests.push({
          request_id: `batch-${Date.now()}-${i}`,
          model_name: values.model.split(':')[0],
          model_version: values.model.split(':')[1],
          inputs: JSON.parse(values.inputs),
          status: 'pending',
          created_at: new Date().toISOString()
        })
      }
      
      setInferenceResults(prev => [...requests, ...prev])
      
      // 模拟批量推理
      setTimeout(() => {
        const completedRequests = requests.map(req => ({
          ...req,
          status: 'completed' as const,
          processing_time_ms: Math.random() * 100 + 30,
          outputs: {
            prediction: [Math.random(), Math.random(), Math.random()],
            confidence: Math.random()
          }
        }))
        
        setInferenceResults(prev => 
          prev.map(req => 
            completedRequests.find(cr => cr.request_id === req.request_id) || req
          )
        )
        message.success(`批量推理完成，处理了${batchSize}个请求`)
      }, 2000 + Math.random() * 3000)

    } catch (error) {
      message.error('批量推理失败')
    }
    setLoading(false)
  }

  const loadModel = async (modelKey: string) => {
    try {
      message.loading('正在加载模型...', 2)
      setTimeout(() => {
        message.success('模型加载成功')
        loadModels()
      }, 2000)
    } catch (error) {
      message.error('模型加载失败')
    }
  }

  const unloadModel = async (modelKey: string) => {
    try {
      message.loading('正在卸载模型...', 1)
      setTimeout(() => {
        message.success('模型卸载成功')
        loadModels()
      }, 1000)
    } catch (error) {
      message.error('模型卸载失败')
    }
  }

  const modelColumns: ColumnsType<LoadedModel> = [
    {
      title: '模型',
      key: 'model',
      render: (_, record) => (
        <Space direction="vertical" size={0}>
          <Text strong>{record.name}</Text>
          <Text type="secondary">v{record.version}</Text>
        </Space>
      )
    },
    {
      title: '格式',
      dataIndex: 'format',
      key: 'format',
      render: (format) => (
        <Tag color="blue">{format.toUpperCase()}</Tag>
      )
    },
    {
      title: '状态',
      key: 'status',
      render: (_, record) => (
        <Space>
          {record.is_healthy ? (
            <Tag color="green" icon={<CheckCircleOutlined />}>健康</Tag>
          ) : (
            <Tag color="red" icon={<ExclamationCircleOutlined />}>异常</Tag>
          )}
        </Space>
      )
    },
    {
      title: '加载时间',
      dataIndex: 'load_time',
      key: 'load_time',
      render: (time) => `${time.toFixed(1)}s`
    },
    {
      title: '最后使用',
      dataIndex: 'last_used',
      key: 'last_used',
      render: (time) => new Date(time).toLocaleString()
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record) => (
        <Space>
          <Button
            type="link"
            size="small"
            onClick={() => setSelectedModel(record.model_key)}
          >
            选择
          </Button>
          <Button
            type="link"
            size="small"
            danger
            onClick={() => unloadModel(record.model_key)}
          >
            卸载
          </Button>
        </Space>
      )
    }
  ]

  const resultColumns: ColumnsType<InferenceRequest> = [
    {
      title: '请求ID',
      dataIndex: 'request_id',
      key: 'request_id',
      render: (id) => (
        <Text code style={{ fontSize: '12px' }}>
          {id.length > 20 ? `${id.substring(0, 20)}...` : id}
        </Text>
      )
    },
    {
      title: '模型',
      key: 'model',
      render: (_, record) => `${record.model_name}:${record.model_version}`
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const statusConfig = {
          pending: { color: 'orange', icon: <ClockCircleOutlined /> },
          processing: { color: 'blue', icon: <Spin size="small" /> },
          completed: { color: 'green', icon: <CheckCircleOutlined /> },
          failed: { color: 'red', icon: <ExclamationCircleOutlined /> }
        }
        const config = statusConfig[status as keyof typeof statusConfig]
        return (
          <Tag color={config.color} icon={config.icon}>
            {status.toUpperCase()}
          </Tag>
        )
      }
    },
    {
      title: '处理时间',
      dataIndex: 'processing_time_ms',
      key: 'processing_time',
      render: (time) => time ? `${time.toFixed(1)}ms` : '-'
    },
    {
      title: '创建时间',
      dataIndex: 'created_at',
      key: 'created_at',
      render: (time) => new Date(time).toLocaleTimeString()
    }
  ]

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>模型推理服务</Title>
        <Text type="secondary">高性能模型推理和批处理服务</Text>
      </div>

      {/* 已加载模型状态 */}
      <Card title="已加载模型" style={{ marginBottom: '24px' }}>
        <Table
          columns={modelColumns}
          dataSource={loadedModels}
          rowKey="model_key"
          size="small"
          pagination={false}
        />
      </Card>

      {/* 性能指标 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={8}>
          <Card>
            <Statistic
              title="活跃模型数"
              value={loadedModels.filter(m => m.is_healthy).length}
              suffix={`/ ${loadedModels.length}`}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="总请求数"
              value={Object.values(metrics).reduce((sum, m) => sum + m.request_count, 0)}
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
        <Col span={8}>
          <Card>
            <Statistic
              title="平均延迟"
              value={
                Object.values(metrics).length > 0
                  ? Object.values(metrics).reduce((sum, m) => sum + m.avg_latency_ms, 0) / Object.values(metrics).length
                  : 0
              }
              suffix="ms"
              prefix={<ThunderboltOutlined />}
              precision={1}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={16}>
        <Col span={12}>
          {/* 推理测试 */}
          <Card title="推理测试">
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              <TabPane tab="单次推理" key="single">
                <Form form={form} layout="vertical" onFinish={handleSingleInference}>
                  <Form.Item
                    name="model"
                    label="选择模型"
                    rules={[{ required: true, message: '请选择模型' }]}
                    initialValue={selectedModel}
                  >
                    <Select
                      placeholder="选择要使用的模型"
                      value={selectedModel}
                      onChange={setSelectedModel}
                    >
                      {loadedModels.filter(m => m.is_healthy).map(model => (
                        <Option key={model.model_key} value={model.model_key}>
                          {model.name}:{model.version} ({model.format})
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>

                  <Form.Item
                    name="inputs"
                    label="输入数据"
                    rules={[{ required: true, message: '请输入数据' }]}
                  >
                    <TextArea
                      rows={6}
                      placeholder='{"data": [[1, 2, 3, 4, 5]], "parameters": {"max_length": 128}}'
                    />
                  </Form.Item>

                  <Form.Item>
                    <Button
                      type="primary"
                      htmlType="submit"
                      loading={loading}
                      icon={<PlayCircleOutlined />}
                      block
                    >
                      执行推理
                    </Button>
                  </Form.Item>
                </Form>
              </TabPane>

              <TabPane tab="批量推理" key="batch">
                <Form form={batchForm} layout="vertical" onFinish={handleBatchInference}>
                  <Form.Item
                    name="model"
                    label="选择模型"
                    rules={[{ required: true, message: '请选择模型' }]}
                  >
                    <Select placeholder="选择要使用的模型">
                      {loadedModels.filter(m => m.is_healthy).map(model => (
                        <Option key={model.model_key} value={model.model_key}>
                          {model.name}:{model.version}
                        </Option>
                      ))}
                    </Select>
                  </Form.Item>

                  <Form.Item
                    name="batch_size"
                    label="批处理大小"
                    rules={[{ required: true, message: '请输入批处理大小' }]}
                    initialValue={10}
                  >
                    <InputNumber min={1} max={100} />
                  </Form.Item>

                  <Form.Item
                    name="inputs"
                    label="输入数据模板"
                    rules={[{ required: true, message: '请输入数据模板' }]}
                  >
                    <TextArea
                      rows={4}
                      placeholder='{"data": [[1, 2, 3, 4, 5]]}'
                    />
                  </Form.Item>

                  <Form.Item>
                    <Button
                      type="primary"
                      htmlType="submit"
                      loading={loading}
                      icon={<PlayCircleOutlined />}
                      block
                    >
                      批量推理
                    </Button>
                  </Form.Item>
                </Form>
              </TabPane>
            </Tabs>
          </Card>
        </Col>

        <Col span={12}>
          {/* 推理结果 */}
          <Card title={`推理结果 (${inferenceResults.length})`}>
            <Table
              columns={resultColumns}
              dataSource={inferenceResults.slice(0, 10)}
              rowKey="request_id"
              size="small"
              pagination={false}
              scroll={{ y: 400 }}
            />
          </Card>
        </Col>
      </Row>

      {/* 模型性能指标 */}
      {Object.keys(metrics).length > 0 && (
        <Card title="模型性能指标" style={{ marginTop: '24px' }}>
          <Row gutter={16}>
            {Object.entries(metrics).map(([modelKey, metric]) => (
              <Col span={8} key={modelKey}>
                <Card size="small" title={modelKey.replace(':', ' v')}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text strong>请求统计</Text>
                      <br />
                      <Text>总数: {metric.request_count}</Text>
                      <br />
                      <Text>成功: {metric.success_count}</Text>
                      <br />
                      <Text type="danger">失败: {metric.error_count}</Text>
                    </div>
                    
                    <div>
                      <Text strong>性能指标</Text>
                      <br />
                      <Text>平均延迟: {metric.avg_latency_ms.toFixed(1)}ms</Text>
                      <br />
                      <Text>P95延迟: {metric.p95_latency_ms.toFixed(1)}ms</Text>
                      <br />
                      <Text>吞吐量: {metric.throughput_qps.toFixed(1)} QPS</Text>
                    </div>
                    
                    <div>
                      <Text strong>错误率</Text>
                      <Progress
                        percent={metric.error_rate * 100}
                        size="small"
                        status={metric.error_rate > 0.05 ? 'exception' : 'success'}
                        format={() => `${(metric.error_rate * 100).toFixed(2)}%`}
                      />
                    </div>
                  </Space>
                </Card>
              </Col>
            ))}
          </Row>
        </Card>
      )}
    </div>
  )
}

export default ModelInferencePage