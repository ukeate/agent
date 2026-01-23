import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Row,
  Col,
  Button,
  Table,
  Statistic,
  Typography,
  Space,
  Tag,
  message,
} from 'antd'
import {
  ThunderboltOutlined,
  DashboardOutlined,
  LineChartOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloudServerOutlined,
  PieChartOutlined,
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'
import MetricChart from '../components/charts/MetricChart'

const { Title, Text } = Typography

interface PerformanceSample {
  cpu_percent: number
  memory_percent: number
  disk_read_bytes: number
  disk_write_bytes: number
  network_sent_bytes: number
  network_recv_bytes: number
  disk_mb_s: number
  network_mb_s: number
  bottlenecks: string[]
  timestamp: string
  time: string
}

interface OptimizationItem {
  optimization: string
  status: string
  timestamp: string
}

const PerformanceOptimizationPage: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceSample[]>([])
  const [optimizations, setOptimizations] = useState<OptimizationItem[]>([])
  const currentMetrics = metrics[metrics.length - 1] || null

  useEffect(() => {
    fetchPerformanceData()
    const interval = setInterval(fetchPerformanceData, 10000)
    return () => clearInterval(interval)
  }, [])

  const fetchPerformanceData = async () => {
    try {
      const res = await apiFetch(
        buildApiUrl('/api/v1/platform/optimization/metrics')
      )
      const data = await res.json()
      const m = data.metrics
      if (!m) return

      const readBytes = Number(m.disk_usage?.read_bytes || 0)
      const writeBytes = Number(m.disk_usage?.write_bytes || 0)
      const sentBytes = Number(m.network_usage?.bytes_sent || 0)
      const recvBytes = Number(m.network_usage?.bytes_recv || 0)
      const ts = String(m.timestamp)

      setMetrics(prev => {
        const last = prev[prev.length - 1]
        const dt = last
          ? (new Date(ts).getTime() - new Date(last.timestamp).getTime()) / 1000
          : 0
        const diskDelta = last
          ? Math.max(0, readBytes - last.disk_read_bytes) +
            Math.max(0, writeBytes - last.disk_write_bytes)
          : 0
        const netDelta = last
          ? Math.max(0, sentBytes - last.network_sent_bytes) +
            Math.max(0, recvBytes - last.network_recv_bytes)
          : 0

        const next = [
          ...prev,
          {
            cpu_percent: Number(m.cpu_percent || 0),
            memory_percent: Number(m.memory_percent || 0),
            disk_read_bytes: readBytes,
            disk_write_bytes: writeBytes,
            network_sent_bytes: sentBytes,
            network_recv_bytes: recvBytes,
            disk_mb_s: dt > 0 ? diskDelta / (1024 * 1024) / dt : 0,
            network_mb_s: dt > 0 ? netDelta / (1024 * 1024) / dt : 0,
            bottlenecks: Array.isArray(m.bottlenecks) ? m.bottlenecks : [],
            timestamp: ts,
            time: new Date(ts).toLocaleTimeString(),
          },
        ]
        return next.slice(-60)
      })
    } catch (error) {
      logger.error('获取性能数据失败:', error)
    }
  }

  const handleRunOptimization = async () => {
    try {
      const response = await apiFetch(
        buildApiUrl('/api/v1/platform/optimization/run'),
        { method: 'POST' }
      )

      message.success('优化任务已启动')
      const data = await response.json()
      const timestamp = data.optimization_results?.timestamp
      const list = Array.isArray(data.optimization_results?.optimizations)
        ? data.optimization_results.optimizations
        : []
      setOptimizations(
        list.map((item: any) => ({
          optimization: String(item.optimization || ''),
          status: String(
            item.results?.status ||
              data.optimization_results?.status ||
              'completed'
          ),
          timestamp: String(timestamp || ''),
        }))
      )
      fetchPerformanceData()
    } catch (error) {
      message.error('启动优化失败')
    }
  }

  const optimizationColumns: ColumnsType<OptimizationItem> = [
    {
      title: '优化项',
      dataIndex: 'optimization',
      key: 'optimization',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: status => {
        const statusConfig = {
          completed: { color: 'success', text: '已完成' },
          optimized: { color: 'success', text: '已优化' },
          disabled: { color: 'default', text: '未启用' },
          error: { color: 'error', text: '失败' },
        }
        const config = statusConfig[status] || {
          color: 'default',
          text: String(status),
        }
        return <Tag color={config.color}>{config.text}</Tag>
      },
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: v => (v ? new Date(v).toLocaleString() : '-'),
    },
  ]

  return (
    <div style={{ padding: '24px' }}>
      <div
        style={{
          marginBottom: 24,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Title level={2}>性能优化</Title>
        <Space>
          <Button icon={<ReloadOutlined />} onClick={fetchPerformanceData}>
            刷新
          </Button>
          <Button
            type="primary"
            icon={<ThunderboltOutlined />}
            onClick={handleRunOptimization}
          >
            运行优化
          </Button>
        </Space>
      </div>

      {currentMetrics && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="CPU使用率"
                value={currentMetrics.cpu_percent}
                suffix="%"
                prefix={<DashboardOutlined />}
                valueStyle={{
                  color:
                    currentMetrics.cpu_percent > 80
                      ? '#f5222d'
                      : currentMetrics.cpu_percent > 60
                        ? '#faad14'
                        : '#52c41a',
                }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="内存使用率"
                value={currentMetrics.memory_percent}
                suffix="%"
                prefix={<PieChartOutlined />}
                valueStyle={{
                  color:
                    currentMetrics.memory_percent > 80
                      ? '#f5222d'
                      : currentMetrics.memory_percent > 60
                        ? '#faad14'
                        : '#52c41a',
                }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="磁盘吞吐"
                value={currentMetrics.disk_mb_s}
                precision={2}
                suffix="MB/s"
                prefix={<LineChartOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="网络吞吐"
                value={currentMetrics.network_mb_s}
                precision={2}
                suffix="MB/s"
                prefix={<CloudServerOutlined />}
              />
            </Card>
          </Col>
        </Row>
      )}

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={12}>
          <Card title="性能趋势" size="small">
            <MetricChart
              type="line"
              series={[
                {
                  name: 'CPU',
                  data: metrics.map(m => ({ x: m.time, y: m.cpu_percent })),
                },
                {
                  name: '内存',
                  data: metrics.map(m => ({ x: m.time, y: m.memory_percent })),
                },
              ]}
              config={{ height: 300 }}
            />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="磁盘与网络吞吐" size="small">
            <MetricChart
              type="line"
              series={[
                {
                  name: '磁盘',
                  data: metrics.map(m => ({ x: m.time, y: m.disk_mb_s })),
                },
                {
                  name: '网络',
                  data: metrics.map(m => ({ x: m.time, y: m.network_mb_s })),
                },
              ]}
              config={{ height: 300 }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col span={12}>
          <Card title="性能瓶颈" size="small">
            {(currentMetrics?.bottlenecks || []).length === 0 ? (
              <div style={{ textAlign: 'center', padding: '40px 0' }}>
                <CheckCircleOutlined
                  style={{ fontSize: 48, color: '#52c41a' }}
                />
                <div style={{ marginTop: 16 }}>
                  <Text>当前没有检测到性能瓶颈</Text>
                </div>
              </div>
            ) : (
              <Space wrap>
                {(currentMetrics?.bottlenecks || []).map(b => (
                  <Tag key={b} color="red">
                    {b}
                  </Tag>
                ))}
              </Space>
            )}
          </Card>
        </Col>
        <Col span={12}>
          <Card title="优化记录" size="small">
            <Table
              columns={optimizationColumns}
              dataSource={optimizations}
              rowKey={record => `${record.timestamp}:${record.optimization}`}
              pagination={false}
              size="small"
            />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default PerformanceOptimizationPage
