import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useEffect, useState } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Space,
  Button,
  message,
} from 'antd'
import {
  ReloadOutlined,
  Activity,
  ClockCircleOutlined,
} from '@ant-design/icons'

interface TrainingMetrics {
  current_step?: number
  total_steps?: number
  loss?: number
  accuracy?: number
  gpu_usage?: number
  gpu_memory?: number
  gpu_temp?: number
  updated_at?: string
}

const TrainingMonitorDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null)
  const [loading, setLoading] = useState(false)

  const loadData = async () => {
    setLoading(true)
    try {
      const res = await apiFetch(
        buildApiUrl('/api/v1/model-evaluation/metrics')
      )
      const data = await res.json()
      setMetrics(data)
    } catch (e: any) {
      setMetrics(null)
      message.error(e?.message || '获取训练监控数据失败')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadData()
    const timer = setInterval(loadData, 5000)
    return () => clearInterval(timer)
  }, [])

  return (
    <div style={{ padding: 24 }}>
      <Space style={{ marginBottom: 16 }}>
        <Activity />
        <h2 style={{ margin: 0 }}>训练监控仪表盘</h2>
      </Space>
      <Button
        icon={<ReloadOutlined />}
        onClick={loadData}
        loading={loading}
        style={{ marginBottom: 16 }}
      >
        刷新
      </Button>

      {metrics ? (
        <>
          <Row gutter={16}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="训练进度"
                  value={`${metrics.current_step ?? 0}/${metrics.total_steps ?? 0}`}
                />
                <Progress
                  percent={
                    metrics.total_steps
                      ? ((metrics.current_step || 0) / metrics.total_steps) *
                        100
                      : 0
                  }
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Loss"
                  value={metrics.loss ?? 0}
                  precision={4}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="Accuracy"
                  value={(metrics.accuracy ?? 0) * 100}
                  precision={2}
                  suffix="%"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="更新时间"
                  value={
                    metrics.updated_at
                      ? new Date(metrics.updated_at).toLocaleTimeString()
                      : '—'
                  }
                  prefix={<ClockCircleOutlined />}
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={16} style={{ marginTop: 16 }}>
            <Col span={8}>
              <Card title="GPU 使用率">
                <Progress percent={metrics.gpu_usage ?? 0} status="active" />
              </Card>
            </Col>
            <Col span={8}>
              <Card title="GPU 显存 (GB)">
                <Statistic value={metrics.gpu_memory ?? 0} precision={2} />
              </Card>
            </Col>
            <Col span={8}>
              <Card title="GPU 温度 (°C)">
                <Statistic value={metrics.gpu_temp ?? 0} precision={1} />
              </Card>
            </Col>
          </Row>
        </>
      ) : (
        <Card>
          <p style={{ color: '#888' }}>
            暂无训练数据，请确保后端 `/api/v1/model-evaluation/metrics` 可用。
          </p>
        </Card>
      )}
    </div>
  )
}

export default TrainingMonitorDashboard
