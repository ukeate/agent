import React, { useState } from 'react'
import { 
  Card, 
  Typography, 
  Row, 
  Col, 
  Space, 
  Statistic,
  Progress,
  Table,
  Tag,
  Alert,
  Tabs,
  Select,
  Button
} from 'antd'
import { 
  MonitorOutlined, 
  ThunderboltOutlined,
  LineChartOutlined,
  DatabaseOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined
} from '@ant-design/icons'

const { Title, Paragraph } = Typography
const { TabPane } = Tabs
const { Option } = Select

const SparqlPerformance: React.FC = () => {
  const performanceMetrics = [
    { name: '查询吞吐量', value: 256, unit: 'QPS', trend: 'up', change: '+12%' },
    { name: '平均响应时间', value: 320, unit: 'ms', trend: 'down', change: '-8%' },
    { name: '峰值响应时间', value: 1240, unit: 'ms', trend: 'down', change: '-15%' },
    { name: '错误率', value: 0.8, unit: '%', trend: 'down', change: '-0.3%' }
  ]

  const queryPerformance = [
    {
      query: 'SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100',
      avgTime: 120,
      minTime: 85,
      maxTime: 180,
      executions: 1456,
      cacheHitRate: 78.5,
      status: 'good'
    },
    {
      query: 'SELECT ?person WHERE { ?person rdf:type foaf:Person }',
      avgTime: 350,
      minTime: 200,
      maxTime: 520,
      executions: 892,
      cacheHitRate: 65.2,
      status: 'warning'
    }
  ]

  const columns = [
    {
      title: '查询模式',
      dataIndex: 'query',
      key: 'query',
      ellipsis: true,
      render: (query: string) => (
        <code style={{ fontSize: '12px' }}>{query.substring(0, 50)}...</code>
      )
    },
    {
      title: '平均时间',
      dataIndex: 'avgTime',
      key: 'avgTime',
      render: (time: number) => `${time}ms`
    },
    {
      title: '执行次数',
      dataIndex: 'executions',
      key: 'executions'
    },
    {
      title: '缓存命中率',
      dataIndex: 'cacheHitRate',
      key: 'cacheHitRate',
      render: (rate: number) => `${rate}%`
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'good' ? 'green' : status === 'warning' ? 'orange' : 'red'}>
          {status === 'good' ? '良好' : status === 'warning' ? '警告' : '错误'}
        </Tag>
      )
    }
  ]

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <MonitorOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          SPARQL性能监控
        </Title>
        <Paragraph>
          实时监控SPARQL查询性能、响应时间和系统资源使用情况
        </Paragraph>
      </div>

      <Row gutter={[24, 24]}>
        <Col span={24}>
          <Card title="性能概览" size="small">
            <Row gutter={16}>
              {performanceMetrics.map((metric, index) => (
                <Col span={6} key={index}>
                  <Statistic
                    title={metric.name}
                    value={metric.value}
                    suffix={metric.unit}
                    prefix={index === 0 ? <ThunderboltOutlined /> : 
                           index === 1 ? <ClockCircleOutlined /> : 
                           index === 2 ? <ExclamationCircleOutlined /> : 
                           <CheckCircleOutlined />}
                    valueStyle={{ 
                      color: metric.trend === 'up' ? '#3f8600' : '#cf1322' 
                    }}
                  />
                  <div style={{ marginTop: '8px' }}>
                    <Tag color={metric.trend === 'up' ? 'green' : 'red'}>
                      {metric.change}
                    </Tag>
                  </div>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        <Col span={16}>
          <Tabs defaultActiveKey="queries" size="small">
            <TabPane tab="查询性能" key="queries">
              <Card size="small">
                <Table
                  dataSource={queryPerformance}
                  columns={columns}
                  pagination={false}
                  size="small"
                />
              </Card>
            </TabPane>

            <TabPane tab="资源使用" key="resources">
              <Card size="small">
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card size="small" title="CPU使用率">
                      <Progress percent={68} status="active" />
                      <p style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
                        当前: 68% | 平均: 72% | 峰值: 89%
                      </p>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small" title="内存使用率">
                      <Progress percent={45} status="active" strokeColor="#52c41a" />
                      <p style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
                        当前: 45% | 平均: 52% | 峰值: 76%
                      </p>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small" title="磁盘I/O">
                      <Progress percent={23} status="active" strokeColor="#1890ff" />
                      <p style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
                        读: 120MB/s | 写: 45MB/s
                      </p>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card size="small" title="网络I/O">
                      <Progress percent={15} status="active" strokeColor="#722ed1" />
                      <p style={{ marginTop: '8px', fontSize: '12px', color: '#666' }}>
                        入: 50MB/s | 出: 25MB/s
                      </p>
                    </Card>
                  </Col>
                </Row>
              </Card>
            </TabPane>
          </Tabs>
        </Col>

        <Col span={8}>
          <Card title="性能建议" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                message="查询优化建议"
                description="检测到复杂连接查询，建议启用查询重写优化"
                type="info"
                showIcon
              />
              <Alert
                message="缓存建议"
                description="频繁查询模式缓存命中率较低，建议调整缓存策略"
                type="warning"
                showIcon
              />
              <Alert
                message="索引建议"
                description="发现全表扫描操作，建议为常用谓词创建索引"
                type="error"
                showIcon
              />
            </Space>
          </Card>

          <Card title="快速操作" size="small" style={{ marginTop: '16px' }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button type="primary" block>
                生成性能报告
              </Button>
              <Button block>
                导出监控数据
              </Button>
              <Button block>
                配置告警规则
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default SparqlPerformance