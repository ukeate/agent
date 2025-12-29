import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Space,
  Tag,
  Statistic,
  Typography,
  Alert,
  Tabs,
  Progress,
  Tooltip
} from 'antd'
import {
  ThunderboltOutlined,
  CodeOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  MonitorOutlined,
  RocketOutlined,
  DatabaseOutlined
} from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'
import { modelRegistryService } from '../services/modelRegistryService'
import { modelService } from '../services/modelService'
import apiClient from '../services/apiClient'

const { Title, Text } = Typography

interface TensorFlowModel {
  id: string
  name: string
  type: 'DNN' | 'CNN' | 'RNN' | 'Transformer'
  status: 'training' | 'deployed' | 'stopped' | 'loading' | 'unknown'
  version?: string
  accuracy?: number
  loss?: number
  epochs?: number
  batchSize?: number
  createdAt?: string
}

const TensorFlowManagementPage: React.FC = () => {
  const [models, setModels] = useState<TensorFlowModel[]>([])

  const [loading, setLoading] = useState(false)
  const [systemMetrics, setSystemMetrics] = useState<{ cpu?: number; memory?: number }>({})
  const navigate = useNavigate()

  const statusColors = {
    training: 'blue',
    deployed: 'green',
    stopped: 'red',
    loading: 'orange',
    unknown: 'default'
  }

  const typeColors = {
    DNN: 'purple',
    CNN: 'cyan',
    RNN: 'geekblue',
    Transformer: 'gold'
  }

  const columns = [
    {
      title: '模型名称',
      dataIndex: 'name',
      key: 'name'
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={typeColors[type as keyof typeof typeColors]}>{type}</Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={statusColors[status as keyof typeof statusColors]}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: 'TensorFlow版本',
      dataIndex: 'version',
      key: 'version'
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy?: number) => (
        accuracy !== undefined ? (
          <Tooltip title={`准确率: ${accuracy}%`}>
            <Progress 
              percent={accuracy} 
              size="small" 
              status={accuracy > 90 ? 'success' : 'active'}
            />
          </Tooltip>
        ) : (
          <Text type="secondary">-</Text>
        )
      ),
    },
    {
      title: '损失值',
      dataIndex: 'loss',
      key: 'loss',
      render: (loss?: number) => (
        loss !== undefined ? (
          <Text type={loss < 0.1 ? 'success' : 'warning'}>{loss}</Text>
        ) : (
          <Text type="secondary">-</Text>
        )
      )
    },
    {
      title: 'Epochs',
      dataIndex: 'epochs',
      key: 'epochs',
      render: (value?: number) => (value !== undefined ? value : '-')
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (value?: string) => value || '-'
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: TensorFlowModel) => (
        <Space>
          {record.status === 'loading' && (
            <Button size="small" icon={<PauseCircleOutlined />} disabled>加载中</Button>
          )}
          {record.status === 'stopped' && (
            <Button size="small" icon={<PlayCircleOutlined />} type="primary" disabled>启动</Button>
          )}
          <Button size="small" icon={<MonitorOutlined />} disabled>监控</Button>
        </Space>
      )
    }
  ]

  const handleCreateModel = () => {
    navigate('/model-registry')
  }

  const loadModels = async () => {
    setLoading(true)
    try {
      const [entries, loadedModels] = await Promise.all([
        modelRegistryService.listModels(),
        modelService.getLoadedModels()
      ])
      const loadedMap = new Map(loadedModels.map((item) => [`${item.name}:${item.version}`, item]))
      const nextModels = entries
        .filter((entry) => {
          const format = String(entry.metadata?.format || '').toLowerCase()
          const framework = String(entry.metadata?.training_framework || '').toLowerCase()
          return format === 'tensorflow' || framework.includes('tensorflow')
        })
        .map((entry) => {
          const metadata = entry.metadata
          const metrics = metadata?.performance_metrics || {}
          const modelKey = `${metadata?.name}:${metadata?.version}`
          const loaded = loadedMap.get(modelKey)
          const status = loaded?.status === 'loading'
            ? 'loading'
            : loaded?.status === 'ready'
              ? 'deployed'
              : 'stopped'
          const modelType = String(metadata?.model_type || '').toLowerCase()
          const type: TensorFlowModel['type'] =
            modelType === 'cnn' ? 'CNN'
            : modelType === 'rnn' ? 'RNN'
            : modelType === 'transformer' ? 'Transformer'
            : 'DNN'
          const accuracy = metrics.accuracy ?? metrics.acc ?? metrics.f1_score
          const loss = metrics.loss ?? metrics.train_loss
          return {
            id: modelKey,
            name: metadata?.name || '',
            type,
            status,
            version: metadata?.version,
            accuracy: typeof accuracy === 'number' ? accuracy : undefined,
            loss: typeof loss === 'number' ? loss : undefined,
            epochs: metadata?.training_epochs,
            createdAt: metadata?.created_at,
          }
        })
      setModels(nextModels)
    } catch (error) {
      setModels([])
    } finally {
      setLoading(false)
    }
  }

  const loadSystemMetrics = async () => {
    try {
      const response = await apiClient.get('/metrics')
      const data = response.data
      setSystemMetrics({
        cpu: data?.cpu?.usage,
        memory: data?.memory?.usage
      })
    } catch (error) {
      setSystemMetrics({})
    }
  }

  useEffect(() => {
    loadModels()
    loadSystemMetrics()
    const interval = setInterval(loadSystemMetrics, 30000)
    return () => clearInterval(interval)
  }, [])

  const accuracyValues = models
    .map((m) => m.accuracy)
    .filter((value): value is number => typeof value === 'number')
  const avgAccuracy = accuracyValues.length
    ? accuracyValues.reduce((acc, value) => acc + value, 0) / accuracyValues.length
    : 0

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <ThunderboltOutlined /> TensorFlow 模型管理
      </Title>

      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中模型"
              value={models.filter(m => m.status === 'training' || m.status === 'deployed').length}
              prefix={<RocketOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已部署模型"
              value={models.filter(m => m.status === 'deployed').length}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均准确率"
              value={avgAccuracy}
              precision={1}
              suffix="%"
              prefix={<ThunderboltOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总模型数"
              value={models.length}
              prefix={<CodeOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Alert
        message="TensorFlow 深度学习模型管理"
        description="管理TensorFlow模型的训练、部署和监控。支持多种神经网络架构，包括CNN、RNN、Transformer等。"
        type="info"
        showIcon
        style={{ marginBottom: '24px' }}
      />

      <Tabs
        items={[
          {
            key: '1',
            label: '模型列表',
            children: (
              <Card>
                <Space style={{ marginBottom: '16px' }}>
                  <Button 
                    type="primary" 
                    icon={<ThunderboltOutlined />}
                    onClick={handleCreateModel}
                  >
                    前往模型注册
                  </Button>
                  <Button icon={<ReloadOutlined />} onClick={loadModels}>刷新</Button>
                </Space>
                <Table
                  columns={columns}
                  dataSource={models}
                  rowKey="id"
                  loading={loading}
                />
              </Card>
            )
          },
          {
            key: '2',
            label: '运行监控',
            children: (
              <Card title="系统资源">
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card size="small" title="CPU 使用率">
                      <Progress percent={systemMetrics.cpu ?? 0} status="active" />
                      <Text type="secondary">来自 /api/v1/metrics</Text>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small" title="内存使用率">
                      <Progress percent={systemMetrics.memory ?? 0} status="active" />
                      <Text type="secondary">来自 /api/v1/metrics</Text>
                    </Card>
                  </Col>
                </Row>
              </Card>
            )
          }
        ]}
      />
    </div>
  )
}

export default TensorFlowManagementPage
