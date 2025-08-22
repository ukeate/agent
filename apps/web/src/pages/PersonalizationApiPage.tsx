import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Table, Tag, Button, Space, Typography, Tabs, Badge, Alert, Input, Select, Switch, Statistic } from 'antd'
import { 
  ApiOutlined,
  CloudServerOutlined,
  SendOutlined,
  CodeOutlined,
  NetworkOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  WarningOutlined,
  GlobalOutlined,
  KeyOutlined,
  MonitorOutlined,
  FileTextOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined
} from '@ant-design/icons'
import type { ColumnsType } from 'antd/es/table'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { TextArea } = Input
const { Option } = Select

interface ApiEndpoint {
  key: string
  path: string
  method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'WEBSOCKET'
  description: string
  status: 'active' | 'deprecated' | 'maintenance'
  version: string
  latency: number
  qps: number
  successRate: number
}

interface ApiKey {
  id: string
  name: string
  key: string
  permissions: string[]
  usage: number
  limit: number
  status: 'active' | 'suspended' | 'expired'
  createdAt: string
}

const PersonalizationApiPage: React.FC = () => {
  const [selectedEndpoint, setSelectedEndpoint] = useState<string>('')
  const [testResponse, setTestResponse] = useState<string>('')
  const [isLoading, setIsLoading] = useState(false)
  const [apiKeys, setApiKeys] = useState<ApiKey[]>([
    {
      id: '1',
      name: 'Web Frontend',
      key: 'pk_live_51H7K2...',
      permissions: ['recommend', 'profile', 'feedback'],
      usage: 15420,
      limit: 100000,
      status: 'active',
      createdAt: '2024-01-10'
    },
    {
      id: '2',
      name: 'Mobile App',
      key: 'pk_live_94N8M1...',
      permissions: ['recommend', 'feedback'],
      usage: 8765,
      limit: 50000,
      status: 'active',
      createdAt: '2024-01-12'
    },
    {
      id: '3',
      name: 'Analytics Service',
      key: 'pk_test_38F2L5...',
      permissions: ['profile', 'analytics'],
      usage: 2340,
      limit: 10000,
      status: 'active',
      createdAt: '2024-01-14'
    }
  ])

  const [endpoints] = useState<ApiEndpoint[]>([
    {
      key: '1',
      path: '/api/v1/personalization/recommend',
      method: 'POST',
      description: '获取个性化推荐',
      status: 'active',
      version: 'v1.2.0',
      latency: 45,
      qps: 1250,
      successRate: 99.8
    },
    {
      key: '2',
      path: '/api/v1/personalization/profile',
      method: 'GET',
      description: '获取用户画像',
      status: 'active',
      version: 'v1.1.0',
      latency: 23,
      qps: 850,
      successRate: 99.9
    },
    {
      key: '3',
      path: '/api/v1/personalization/feedback',
      method: 'POST',
      description: '提交用户反馈',
      status: 'active',
      version: 'v1.0.0',
      latency: 15,
      qps: 420,
      successRate: 99.7
    },
    {
      key: '4',
      path: '/ws/personalization/stream',
      method: 'WEBSOCKET',
      description: '实时推荐流',
      status: 'active',
      version: 'v1.0.0',
      latency: 8,
      qps: 150,
      successRate: 99.5
    },
    {
      key: '5',
      path: '/api/v1/personalization/batch',
      method: 'POST',
      description: '批量推荐',
      status: 'active',
      version: 'v1.0.0',
      latency: 120,
      qps: 85,
      successRate: 99.2
    },
    {
      key: '6',
      path: '/api/v1/personalization/explain',
      method: 'POST',
      description: '推荐解释',
      status: 'deprecated',
      version: 'v0.9.0',
      latency: 78,
      qps: 25,
      successRate: 98.9
    }
  ])

  const endpointColumns: ColumnsType<ApiEndpoint> = [
    {
      title: 'API端点',
      dataIndex: 'path',
      key: 'path',
      render: (text, record) => (
        <Space direction="vertical" size="small">
          <Space>
            {getMethodTag(record.method)}
            <Text code>{text}</Text>
          </Space>
          <Text type="secondary" style={{ fontSize: '12px' }}>{record.description}</Text>
        </Space>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const config = {
          active: { color: 'success', text: '活跃' },
          deprecated: { color: 'warning', text: '已弃用' },
          maintenance: { color: 'processing', text: '维护中' }
        }
        return <Tag color={config[status].color}>{config[status].text}</Tag>
      }
    },
    {
      title: '版本',
      dataIndex: 'version',
      key: 'version',
      render: (version) => <Text type="secondary">{version}</Text>
    },
    {
      title: '延迟',
      dataIndex: 'latency',
      key: 'latency',
      render: (latency) => (
        <Text style={{ color: latency > 100 ? '#ff4d4f' : '#52c41a' }}>
          {latency}ms
        </Text>
      )
    },
    {
      title: 'QPS',
      dataIndex: 'qps',
      key: 'qps',
      render: (qps) => qps.toLocaleString()
    },
    {
      title: '成功率',
      dataIndex: 'successRate',
      key: 'successRate',
      render: (rate) => (
        <Text style={{ color: rate > 99 ? '#52c41a' : '#faad14' }}>
          {rate}%
        </Text>
      )
    }
  ]

  const keyColumns: ColumnsType<ApiKey> = [
    {
      title: '应用名称',
      dataIndex: 'name',
      key: 'name',
      render: (text) => <Text strong>{text}</Text>
    },
    {
      title: 'API密钥',
      dataIndex: 'key',
      key: 'key',
      render: (key) => <Text code>{key}</Text>
    },
    {
      title: '权限',
      dataIndex: 'permissions',
      key: 'permissions',
      render: (permissions) => (
        <Space wrap>
          {permissions.map(p => <Tag key={p} color="blue">{p}</Tag>)}
        </Space>
      )
    },
    {
      title: '使用量',
      dataIndex: 'usage',
      key: 'usage',
      render: (usage, record) => (
        <Space>
          <Text>{usage.toLocaleString()}</Text>
          <Text type="secondary">/ {record.limit.toLocaleString()}</Text>
        </Space>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status) => {
        const config = {
          active: { color: 'success', text: '活跃' },
          suspended: { color: 'error', text: '已暂停' },
          expired: { color: 'default', text: '已过期' }
        }
        return <Tag color={config[status].color}>{config[status].text}</Tag>
      }
    }
  ]

  const getMethodTag = (method: string) => {
    const colors = {
      GET: 'blue',
      POST: 'green',
      PUT: 'orange',
      DELETE: 'red',
      WEBSOCKET: 'purple'
    }
    return <Tag color={colors[method]}>{method}</Tag>
  }

  const handleTestApi = async () => {
    setIsLoading(true)
    
    // 模拟API调用
    setTimeout(() => {
      const mockResponse = {
        success: true,
        data: {
          recommendations: [
            { id: 1, title: "机器学习实战", score: 0.95 },
            { id: 2, title: "深度学习框架", score: 0.89 },
            { id: 3, title: "AI系统设计", score: 0.82 }
          ],
          user_id: "user_12345",
          request_id: "req_" + Date.now(),
          latency_ms: 45
        },
        timestamp: new Date().toISOString()
      }
      
      setTestResponse(JSON.stringify(mockResponse, null, 2))
      setIsLoading(false)
    }, 1000)
  }

  // 统计数据
  const totalQps = endpoints.reduce((acc, ep) => acc + ep.qps, 0)
  const avgLatency = endpoints.reduce((acc, ep) => acc + ep.latency, 0) / endpoints.length
  const avgSuccessRate = endpoints.reduce((acc, ep) => acc + ep.successRate, 0) / endpoints.length
  const activeEndpoints = endpoints.filter(ep => ep.status === 'active').length

  return (
    <div style={{ padding: '24px' }}>
      <Title level={2}>
        <ApiOutlined /> API接口管理
      </Title>
      <Paragraph type="secondary">
        管理个性化引擎的API接口、密钥和访问权限
      </Paragraph>

      {/* API概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃端点"
              value={activeEndpoints}
              suffix={`/ ${endpoints.length}`}
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总QPS"
              value={totalQps}
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均延迟"
              value={avgLatency}
              suffix="ms"
              precision={1}
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: avgLatency > 100 ? '#ff4d4f' : '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均成功率"
              value={avgSuccessRate}
              suffix="%"
              precision={1}
              prefix={<MonitorOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      <Tabs defaultActiveKey="1">
        <TabPane tab={<span><ApiOutlined /> API端点</span>} key="1">
          <Card>
            <Table
              columns={endpointColumns}
              dataSource={endpoints}
              pagination={false}
              onRow={(record) => ({
                onClick: () => setSelectedEndpoint(record.path)
              })}
              rowClassName={(record) => 
                selectedEndpoint === record.path ? 'ant-table-row-selected' : ''
              }
            />
          </Card>
        </TabPane>

        <TabPane tab={<span><KeyOutlined /> API密钥</span>} key="2">
          <Card
            extra={
              <Button type="primary" icon={<KeyOutlined />}>
                创建密钥
              </Button>
            }
          >
            <Table
              columns={keyColumns}
              dataSource={apiKeys}
              pagination={false}
            />
          </Card>
        </TabPane>

        <TabPane tab={<span><CodeOutlined /> API测试</span>} key="3">
          <Row gutter={[16, 16]}>
            <Col span={12}>
              <Card title="请求配置">
                <Space direction="vertical" style={{ width: '100%' }} size="middle">
                  <div>
                    <Text>选择端点:</Text>
                    <Select
                      style={{ width: '100%', marginTop: 8 }}
                      placeholder="选择要测试的API端点"
                      value={selectedEndpoint}
                      onChange={setSelectedEndpoint}
                    >
                      {endpoints.map(ep => (
                        <Option key={ep.path} value={ep.path}>
                          {getMethodTag(ep.method)} {ep.path}
                        </Option>
                      ))}
                    </Select>
                  </div>
                  
                  <div>
                    <Text>请求参数:</Text>
                    <TextArea
                      rows={8}
                      placeholder="输入JSON格式的请求参数"
                      defaultValue={JSON.stringify({
                        user_id: "user_12345",
                        scenario: "homepage",
                        n_recommendations: 10,
                        context: {
                          device: "mobile",
                          location: "beijing"
                        }
                      }, null, 2)}
                    />
                  </div>
                  
                  <Button 
                    type="primary" 
                    icon={<SendOutlined />}
                    loading={isLoading}
                    onClick={handleTestApi}
                    disabled={!selectedEndpoint}
                    block
                  >
                    发送请求
                  </Button>
                </Space>
              </Card>
            </Col>
            
            <Col span={12}>
              <Card title="响应结果">
                <TextArea
                  rows={16}
                  value={testResponse}
                  placeholder="API响应将显示在这里..."
                  readOnly
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={<span><FileTextOutlined /> 文档</span>} key="4">
          <Card>
            <Space direction="vertical" size="large" style={{ width: '100%' }}>
              <Alert
                message="API文档"
                description="完整的API文档可在开发者门户查看，包含详细的请求/响应示例、错误码说明等。"
                variant="default"
                showIcon
                action={
                  <Button size="small" type="primary" icon={<GlobalOutlined />}>
                    查看文档
                  </Button>
                }
              />
              
              <div>
                <Title level={4}>快速开始</Title>
                <Paragraph>
                  使用个性化引擎API的基本步骤：
                </Paragraph>
                <ol>
                  <li>获取API密钥</li>
                  <li>设置请求头：<Text code>Authorization: Bearer YOUR_API_KEY</Text></li>
                  <li>发送请求到相应端点</li>
                  <li>处理响应数据</li>
                </ol>
              </div>
              
              <div>
                <Title level={4}>示例代码</Title>
                <Text>JavaScript示例：</Text>
                <pre style={{ 
                  background: '#f5f5f5', 
                  padding: '16px', 
                  borderRadius: '4px',
                  overflow: 'auto'
                }}>
{`fetch('/api/v1/personalization/recommend', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'
  },
  body: JSON.stringify({
    user_id: 'user_123',
    scenario: 'homepage',
    n_recommendations: 10
  })
})
.then(response => response.json())
.then(data => console.log(data));`}
                </pre>
              </div>
            </Space>
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default PersonalizationApiPage