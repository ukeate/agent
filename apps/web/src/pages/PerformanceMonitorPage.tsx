import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Button,
  Space,
  Table,
  Tag,
  Alert,
  Tabs,
  Select
} from 'antd'
import {
  DashboardOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  CloudOutlined,
  ReloadOutlined,
  DownloadOutlined,
} from '@ant-design/icons'

const { TabPane } = Tabs

interface SystemMetric {
  timestamp: string
  cpu: number
  memory: number
  disk: number
  network: number
}

interface ProcessInfo {
  id: string
  name: string
  cpu: number
  memory: number
  status: 'running' | 'idle' | 'error'
}

const PerformanceMonitorPage: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetric[]>([])
  const [processes, setProcesses] = useState<ProcessInfo[]>([])

  const [autoRefresh, setAutoRefresh] = useState(true)

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const res = await apiFetch(buildApiUrl('/api/v1/health/metrics'))
        const data = await res.json()
        const now = new Date().toISOString()
        const newMetric: SystemMetric = {
          timestamp: now,
          cpu: data.cpu_usage ?? 0,
          memory: data.memory_usage ?? 0,
          disk: data.disk_usage ?? 0,
          network: data.network_usage ?? 0
        }
        setMetrics(prev => [...prev.slice(-19), newMetric])
        if (Array.isArray(data.processes)) {
          const mapped = data.processes.map((p: any, idx: number) => ({
            id: p.id || String(idx),
            name: p.name || `process_${idx}`,
            cpu: p.cpu_usage ?? 0,
            memory: p.memory_usage ?? 0,
            status: (p.status || 'running') as ProcessInfo['status']
          }))
          setProcesses(mapped)
        }
      } catch (e) {
        // 保持原有数据，不填充假数据
      }
    }

    fetchMetrics()
    if (autoRefresh) {
      const interval = setInterval(fetchMetrics, 3000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const currentMetrics = metrics[metrics.length - 1] || {
    cpu: 0, memory: 0, disk: 0, network: 0
  }

  const processColumns = [
    {
      title: '进程名称',
      dataIndex: 'name',
      key: 'name'
    },
    {
      title: 'CPU使用率',
      dataIndex: 'cpu',
      key: 'cpu',
      render: (cpu: number) => (
        <div>
          <Progress 
            percent={cpu} 
            size="small" 
            strokeColor={cpu > 80 ? '#ff4d4f' : '#1890ff'}
          />
          <span className="ml-2">{cpu}%</span>
        </div>
      )
    },
    {
      title: '内存使用',
      dataIndex: 'memory',
      key: 'memory',
      render: (memory: number) => `${memory}MB`
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = { running: 'green', idle: 'blue', error: 'red' }
        const texts = { running: '运行中', idle: '空闲', error: '错误' }
        return <Tag color={colors[status as keyof typeof colors]}>{texts[status as keyof typeof texts]}</Tag>
      }
    }
  ]

  const avgCpu = Math.round(currentMetrics.cpu)
  const avgMemory = Math.round(currentMetrics.memory)
  const avgDisk = Math.round(currentMetrics.disk)
  const avgNetwork = Math.round(currentMetrics.network)

  return (
      <div className="p-6">
        <div className="mb-6">
          <div className="flex justify-between items-center mb-4">
            <h1 className="text-2xl font-bold">性能监控</h1>
            <Space>
              <Button 
                icon={<ReloadOutlined />}
                onClick={() => setAutoRefresh(!autoRefresh)}
                type={autoRefresh ? 'primary' : 'default'}
              >
                {autoRefresh ? '关闭' : '开启'}自动刷新
              </Button>
              <Button icon={<DownloadOutlined />}>
                导出报告
              </Button>
            </Space>
          </div>

          {(avgCpu > 80 || avgMemory > 80) && (
            <Alert
              message="性能警告"
              description="系统资源使用率过高，建议检查系统负载"
              variant="warning"
              showIcon
              closable
              className="mb-4"
            />
          )}

          <Row gutter={16} className="mb-6">
            <Col span={6}>
              <Card>
                <Statistic
                  title="CPU使用率"
                  value={avgCpu}
                  suffix="%"
                  valueStyle={{ color: avgCpu > 80 ? '#cf1322' : '#3f8600' }}
                  prefix={<ThunderboltOutlined />}
                />
                <Progress 
                  percent={avgCpu} 
                  size="small" 
                  strokeColor={avgCpu > 80 ? '#ff4d4f' : '#52c41a'}
                  className="mt-2"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="内存使用率"
                  value={avgMemory}
                  suffix="%"
                  valueStyle={{ color: avgMemory > 80 ? '#cf1322' : '#3f8600' }}
                  prefix={<DatabaseOutlined />}
                />
                <Progress 
                  percent={avgMemory} 
                  size="small" 
                  strokeColor={avgMemory > 80 ? '#ff4d4f' : '#52c41a'}
                  className="mt-2"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="磁盘使用率"
                  value={avgDisk}
                  suffix="%"
                  valueStyle={{ color: avgDisk > 80 ? '#cf1322' : '#3f8600' }}
                  prefix={<DashboardOutlined />}
                />
                <Progress 
                  percent={avgDisk} 
                  size="small" 
                  strokeColor={avgDisk > 80 ? '#ff4d4f' : '#52c41a'}
                  className="mt-2"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="网络使用率"
                  value={avgNetwork}
                  suffix="%"
                  valueStyle={{ color: avgNetwork > 80 ? '#cf1322' : '#3f8600' }}
                  prefix={<CloudOutlined />}
                />
                <Progress 
                  percent={avgNetwork} 
                  size="small" 
                  strokeColor={avgNetwork > 80 ? '#ff4d4f' : '#52c41a'}
                  className="mt-2"
                />
              </Card>
            </Col>
          </Row>
        </div>

        <Tabs defaultActiveKey="realtime">
          <TabPane tab="实时监控" key="realtime">
            <Row gutter={16}>
              <Col span={12}>
                <Card title="系统资源趋势" className="mb-4">
                  <div className="h-64 flex items-center justify-center text-gray-500">
                    <div className="text-center">
                      <DashboardOutlined style={{ fontSize: 48, marginBottom: 16 }} />
                      <div>实时图表组件 (需要图表库)</div>
                    </div>
                  </div>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="进程监控">
                  <Table
                    columns={processColumns}
                    dataSource={processes}
                    rowKey="id"
                    pagination={false}
                    size="small"
                  />
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="历史数据" key="history">
            <Card title="历史性能数据">
              <div className="h-96 flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <DashboardOutlined style={{ fontSize: 64, marginBottom: 16 }} />
                  <div>历史数据图表 (需要图表库)</div>
                  <div className="mt-2 text-sm">支持时间范围选择和数据对比</div>
                </div>
              </div>
            </Card>
          </TabPane>

          <TabPane tab="性能分析" key="analysis">
            <Row gutter={16}>
              <Col span={8}>
                <Card title="性能评分">
                  <div className="text-center">
                    <div className="text-4xl font-bold text-green-500 mb-2">85</div>
                    <Progress 
                      type="circle" 
                      percent={85} 
                      strokeColor="#52c41a"
                      size={120}
                    />
                    <div className="mt-4 text-gray-600">系统性能良好</div>
                  </div>
                </Card>
              </Col>
              <Col span={16}>
                <Card title="性能建议">
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Alert
                      message="CPU优化建议"
                      description="代码生成器CPU使用率较高，建议优化算法或增加实例数量"
                      variant="default"
                      showIcon
                    />
                    <Alert
                      message="内存管理"
                      description="向量数据库内存使用量较大，建议定期清理缓存"
                      variant="warning"
                      showIcon
                    />
                    <Alert
                      message="系统优化"
                      description="整体性能表现良好，继续保持当前配置"
                      type="success"
                      showIcon
                    />
                  </Space>
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </div>
  )
}

export default PerformanceMonitorPage
