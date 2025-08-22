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
  Typography,
  List,
  Timeline,
  Tooltip
} from 'antd'
import {
  DatabaseOutlined,
  ThunderboltOutlined,
  ReloadOutlined,
  DeleteOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  FireOutlined
} from '@ant-design/icons'

const { Title, Text } = Typography
const { TabPane } = Tabs

interface CacheStats {
  hitRate: number
  missRate: number
  totalRequests: number
  totalHits: number
  totalMisses: number
  cacheSize: number
  memoryUsage: number
}

interface CacheNode {
  id: string
  name: string
  type: string
  status: 'active' | 'stale' | 'expired'
  hitCount: number
  lastAccessed: string
  size: string
  ttl: number
}

const CacheMonitorPage: React.FC = () => {
  const [stats, setStats] = useState<CacheStats>({
    hitRate: 87.5,
    missRate: 12.5,
    totalRequests: 15420,
    totalHits: 13492,
    totalMisses: 1928,
    cacheSize: 2048,
    memoryUsage: 75.3
  })

  const [nodes] = useState<CacheNode[]>([
    {
      id: '1',
      name: 'langgraph_node_cache_1',
      type: 'LangGraph节点',
      status: 'active',
      hitCount: 856,
      lastAccessed: '2分钟前',
      size: '128MB',
      ttl: 3600
    },
    {
      id: '2', 
      name: 'workflow_state_cache_2',
      type: '工作流状态',
      status: 'active',
      hitCount: 423,
      lastAccessed: '5分钟前',
      size: '64MB',
      ttl: 1800
    },
    {
      id: '3',
      name: 'agent_memory_cache_3',
      type: 'Agent内存',
      status: 'stale',
      hitCount: 234,
      lastAccessed: '1小时前',
      size: '32MB',
      ttl: 900
    },
    {
      id: '4',
      name: 'context_cache_4',
      type: '上下文缓存',
      status: 'expired',
      hitCount: 12,
      lastAccessed: '3小时前',
      size: '16MB',
      ttl: 0
    }
  ])

  const [autoRefresh, setAutoRefresh] = useState(true)

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        setStats(prev => ({
          ...prev,
          hitRate: Math.max(0, Math.min(100, prev.hitRate + (Math.random() - 0.5) * 2)),
          totalRequests: prev.totalRequests + Math.floor(Math.random() * 10),
          memoryUsage: Math.max(0, Math.min(100, prev.memoryUsage + (Math.random() - 0.5) * 5))
        }))
      }, 3000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const columns = [
    {
      title: '缓存节点',
      dataIndex: 'name',
      key: 'name',
      render: (name: string, record: CacheNode) => (
        <div>
          <Text strong>{name}</Text>
          <br />
          <Text type="secondary" className="text-xs">{record.type}</Text>
        </div>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = { 
          active: 'green', 
          stale: 'orange', 
          expired: 'red' 
        }
        const texts = { 
          active: '活跃', 
          stale: '过期中', 
          expired: '已过期' 
        }
        return <Tag color={colors[status as keyof typeof colors]}>
          {texts[status as keyof typeof texts]}
        </Tag>
      }
    },
    {
      title: '命中次数',
      dataIndex: 'hitCount',
      key: 'hitCount',
      render: (count: number) => (
        <Statistic 
          value={count} 
          valueStyle={{ fontSize: '14px' }}
          prefix={<ThunderboltOutlined />}
        />
      )
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size'
    },
    {
      title: '最后访问',
      dataIndex: 'lastAccessed',
      key: 'lastAccessed'
    },
    {
      title: 'TTL',
      dataIndex: 'ttl',
      key: 'ttl',
      render: (ttl: number) => ttl > 0 ? `${ttl}s` : '已过期'
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: CacheNode) => (
        <Space>
          <Tooltip title="刷新缓存">
            <Button 
              size="small" 
              icon={<ReloadOutlined />} 
              onClick={() => console.log('刷新缓存:', record.id)}
            />
          </Tooltip>
          <Tooltip title="清除缓存">
            <Button 
              size="small" 
              danger 
              icon={<DeleteOutlined />}
              onClick={() => console.log('清除缓存:', record.id)}
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  const performanceData = [
    { time: '10:00', hitRate: 85 },
    { time: '10:30', hitRate: 89 },
    { time: '11:00', hitRate: 92 },
    { time: '11:30', hitRate: 87 },
    { time: '12:00', hitRate: stats.hitRate }
  ]

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <Title level={2}>缓存监控 (LangGraph缓存系统)</Title>
          <Space>
            <Button 
              icon={<ReloadOutlined />}
              onClick={() => setAutoRefresh(!autoRefresh)}
              type={autoRefresh ? 'primary' : 'default'}
            >
              {autoRefresh ? '停止' : '开始'}自动刷新
            </Button>
            <Button 
              icon={<DeleteOutlined />} 
              danger
              onClick={() => console.log('清理所有缓存')}
            >
              清理所有缓存
            </Button>
            <Button 
              icon={<FireOutlined />}
              onClick={() => console.log('执行缓存预热')}
            >
              缓存预热
            </Button>
          </Space>
        </div>

        {stats.hitRate < 80 && (
          <Alert
            message="缓存命中率偏低"
            description="当前缓存命中率低于80%，建议检查缓存策略或增加缓存容量"
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
                title="缓存命中率"
                value={stats.hitRate}
                precision={1}
                suffix="%"
                valueStyle={{ color: stats.hitRate > 80 ? '#3f8600' : '#cf1322' }}
                prefix={<ThunderboltOutlined />}
              />
              <Progress 
                percent={stats.hitRate} 
                strokeColor={stats.hitRate > 80 ? '#52c41a' : '#ff4d4f'}
                size="small"
                className="mt-2"
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="总请求数"
                value={stats.totalRequests}
                prefix={<DatabaseOutlined />}
              />
              <div className="mt-2 text-xs text-gray-500">
                命中: {stats.totalHits} | 未命中: {stats.totalMisses}
              </div>
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="缓存大小"
                value={stats.cacheSize}
                suffix="MB"
                prefix={<DatabaseOutlined />}
              />
              <Progress 
                percent={Math.min(100, (stats.cacheSize / 4096) * 100)} 
                size="small"
                className="mt-2"
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="内存使用率"
                value={stats.memoryUsage}
                precision={1}
                suffix="%"
                valueStyle={{ color: stats.memoryUsage > 90 ? '#cf1322' : '#3f8600' }}
                prefix={<DatabaseOutlined />}
              />
              <Progress 
                percent={stats.memoryUsage} 
                strokeColor={stats.memoryUsage > 90 ? '#ff4d4f' : '#52c41a'}
                size="small"
                className="mt-2"
              />
            </Card>
          </Col>
        </Row>
      </div>

      <Tabs defaultActiveKey="nodes">
        <TabPane tab="缓存节点" key="nodes">
          <Card title="LangGraph节点缓存">
            <Table
              columns={columns}
              dataSource={nodes}
              rowKey="id"
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab="性能分析" key="performance">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="命中率趋势">
                <div className="h-64 flex items-center justify-center text-gray-500">
                  <div className="text-center">
                    <DatabaseOutlined style={{ fontSize: 48, marginBottom: 16 }} />
                    <div>命中率趋势图表 (需要图表库)</div>
                    <List
                      size="small"
                      dataSource={performanceData}
                      renderItem={item => (
                        <List.Item>
                          <Text>{item.time}: {item.hitRate}%</Text>
                        </List.Item>
                      )}
                      className="mt-4 text-left"
                    />
                  </div>
                </div>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="缓存操作日志">
                <Timeline
                  items={[
                    {
                      color: 'green',
                      children: (
                        <div>
                          <Text strong>缓存命中</Text>
                          <br />
                          <Text type="secondary">langgraph_node_cache_1 - 2分钟前</Text>
                        </div>
                      )
                    },
                    {
                      color: 'blue',
                      children: (
                        <div>
                          <Text strong>缓存更新</Text>
                          <br />
                          <Text type="secondary">workflow_state_cache_2 - 5分钟前</Text>
                        </div>
                      )
                    },
                    {
                      color: 'orange',
                      children: (
                        <div>
                          <Text strong>缓存过期</Text>
                          <br />
                          <Text type="secondary">agent_memory_cache_3 - 1小时前</Text>
                        </div>
                      )
                    },
                    {
                      color: 'red',
                      children: (
                        <div>
                          <Text strong>缓存清理</Text>
                          <br />
                          <Text type="secondary">context_cache_4 - 3小时前</Text>
                        </div>
                      )
                    }
                  ]}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="配置管理" key="config">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="缓存配置">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div>
                    <Text strong>默认TTL: </Text>
                    <Text>3600秒</Text>
                  </div>
                  <div>
                    <Text strong>最大缓存大小: </Text>
                    <Text>4096MB</Text>
                  </div>
                  <div>
                    <Text strong>清理策略: </Text>
                    <Text>LRU (最近最少使用)</Text>
                  </div>
                  <div>
                    <Text strong>压缩算法: </Text>
                    <Text>gzip</Text>
                  </div>
                  <div>
                    <Text strong>预热策略: </Text>
                    <Text>启动时预热常用节点</Text>
                  </div>
                </Space>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="健康状态">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <div className="flex items-center">
                    <CheckCircleOutlined className="text-green-500 mr-2" />
                    <Text>缓存服务正常运行</Text>
                  </div>
                  <div className="flex items-center">
                    <CheckCircleOutlined className="text-green-500 mr-2" />
                    <Text>Redis连接正常</Text>
                  </div>
                  <div className="flex items-center">
                    <ClockCircleOutlined className="text-orange-500 mr-2" />
                    <Text>存在过期缓存待清理</Text>
                  </div>
                  <div className="flex items-center">
                    <WarningOutlined className="text-yellow-500 mr-2" />
                    <Text>内存使用率较高 ({stats.memoryUsage.toFixed(1)}%)</Text>
                  </div>
                </Space>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default CacheMonitorPage