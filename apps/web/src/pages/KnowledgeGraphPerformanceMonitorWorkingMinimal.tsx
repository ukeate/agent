import React from 'react'
import { Card, Row, Col, Statistic, Typography, Space, Button, Tabs } from 'antd'
import {
  DashboardOutlined,
  CloudServerOutlined,
  BankOutlined,
  HddOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  ClockCircleOutlined,
  WifiOutlined,
  ReloadOutlined
} from '@ant-design/icons'

const { Title, Text } = Typography

const KnowledgeGraphPerformanceMonitor: React.FC = () => {
  const systemMetrics = {
    cpu_usage: 78.5,
    memory_usage: 85.2,
    disk_usage: 67.8,
    cache_hit_rate: 92.3,
    database_connections: 156,
    query_response_time: 245,
    throughput: 1250,
    network_io: 24.5
  }

  const tabItems = [
    {
      key: 'overview',
      label: '系统概览',
      children: (
        <div>
          <Row gutter={16} style={{ marginBottom: '24px' }}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="CPU使用率"
                  value={systemMetrics.cpu_usage}
                  precision={1}
                  suffix="%"
                  prefix={<CloudServerOutlined />}
                  valueStyle={{ color: systemMetrics.cpu_usage > 80 ? '#ff4d4f' : '#52c41a' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="内存使用率"
                  value={systemMetrics.memory_usage}
                  precision={1}
                  suffix="%"
                  prefix={<BankOutlined />}
                  valueStyle={{ color: systemMetrics.memory_usage > 80 ? '#ff4d4f' : '#52c41a' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="磁盘使用率"
                  value={systemMetrics.disk_usage}
                  precision={1}
                  suffix="%"
                  prefix={<HddOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="缓存命中率"
                  value={systemMetrics.cache_hit_rate}
                  precision={1}
                  suffix="%"
                  prefix={<ThunderboltOutlined />}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="数据库连接数"
                  value={systemMetrics.database_connections}
                  prefix={<DatabaseOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="查询响应时间"
                  value={systemMetrics.query_response_time}
                  suffix="ms"
                  prefix={<ClockCircleOutlined />}
                  valueStyle={{ color: systemMetrics.query_response_time > 500 ? '#ff4d4f' : '#52c41a' }}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="系统吞吐量"
                  value={systemMetrics.throughput}
                  suffix="QPS"
                  prefix={<DatabaseOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="网络IO"
                  value={systemMetrics.network_io}
                  precision={1}
                  suffix="MB/s"
                  prefix={<WifiOutlined />}
                />
              </Card>
            </Col>
          </Row>
        </div>
      )
    },
    {
      key: 'queries',
      label: '查询性能',
      children: (
        <Card title="查询性能分析">
          <div style={{ 
            height: '200px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            border: '1px dashed #d9d9d9',
            backgroundColor: '#fafafa'
          }}>
            <Text type="secondary">查询性能分析功能</Text>
          </div>
        </Card>
      )
    },
    {
      key: 'cache',
      label: '缓存监控',
      children: (
        <Card title="缓存监控">
          <div style={{ 
            height: '200px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            border: '1px dashed #d9d9d9',
            backgroundColor: '#fafafa'
          }}>
            <Text type="secondary">缓存监控功能</Text>
          </div>
        </Card>
      )
    }
  ]

  return (
    <div style={{ padding: '24px' }}>
      <Card style={{ marginBottom: '24px' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <DashboardOutlined style={{ fontSize: '24px' }} />
              <Title level={2} style={{ margin: 0 }}>性能监控</Title>
              <Text type="secondary">监控知识图谱系统性能指标和系统资源使用情况</Text>
            </Space>
          </Col>
          <Col>
            <Button icon={<ReloadOutlined />}>刷新</Button>
          </Col>
        </Row>
      </Card>

      <Tabs items={tabItems} />
    </div>
  )
}

export default KnowledgeGraphPerformanceMonitor