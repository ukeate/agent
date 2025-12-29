import React, { useState, useEffect } from 'react'
import {
import { logger } from '../utils/logger'
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
  Tooltip,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  message,
  Spin,
  Descriptions,
  Badge,
  Empty,
  Divider,
  Switch
} from 'antd'
import {
  DatabaseOutlined,
  ThunderboltOutlined,
  ReloadOutlined,
  DeleteOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ClockCircleOutlined,
  FireOutlined,
  SyncOutlined,
  InfoCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  RocketOutlined,
  FieldTimeOutlined,
  SettingOutlined,
  BulbOutlined
} from '@ant-design/icons'
import {
  cacheService,
  CacheStats,
  CacheHealth,
  CachePerformance,
  CacheStrategy
} from '../services/cacheService'

const { Title, Text } = Typography
const { TabPane } = Tabs
const { Option } = Select

const CacheMonitorPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [stats, setStats] = useState<CacheStats | null>(null)
  const [health, setHealth] = useState<CacheHealth | null>(null)
  const [performance, setPerformance] = useState<CachePerformance | null>(null)
  const [strategy, setStrategy] = useState<CacheStrategy | null>(null)
  const [cacheKeys, setCacheKeys] = useState<string[]>([])
  
  const [clearModalVisible, setClearModalVisible] = useState(false)
  const [invalidateModalVisible, setInvalidateModalVisible] = useState(false)
  const [strategyModalVisible, setStrategyModalVisible] = useState(false)
  const [warmModalVisible, setWarmModalVisible] = useState(false)
  
  const [clearForm] = Form.useForm()
  const [invalidateForm] = Form.useForm()
  const [strategyForm] = Form.useForm()
  const [warmForm] = Form.useForm()
  
  const [autoRefresh, setAutoRefresh] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')
  const [refreshInterval, setRefreshInterval] = useState<ReturnType<typeof setTimeout> | null>(null)

  // 加载数据
  useEffect(() => {
    loadCacheData()
  }, [])

  // 自动刷新
  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        loadCacheData()
      }, 5000)
      setRefreshInterval(interval)
    } else {
      if (refreshInterval) {
        clearInterval(refreshInterval)
        setRefreshInterval(null)
      }
    }
    return () => {
      if (refreshInterval) {
        clearInterval(refreshInterval)
      }
    }
  }, [autoRefresh])

  const loadCacheData = async () => {
    setLoading(true)
    try {
      const [statsData, healthData, perfData, strategyData] = await Promise.all([
        cacheService.getStats(),
        cacheService.checkHealth(),
        cacheService.getPerformance(),
        cacheService.getStrategy()
      ])
      setStats(statsData)
      setHealth(healthData)
      setPerformance(perfData)
      setStrategy(strategyData)
    } catch (error) {
      message.error('加载缓存数据失败')
    } finally {
      setLoading(false)
    }
  }

  const loadCacheKeys = async () => {
    try {
      const keys = await cacheService.listKeys('*', 100)
      setCacheKeys(keys)
    } catch (error) {
      logger.error('加载缓存键失败:', error)
    }
  }

  const handleClearCache = async (values: any) => {
    try {
      const result = await cacheService.clearCache(values.pattern)
      if (result.success) {
        message.success(`成功清理 ${result.cleared_count} 个缓存条目`)
        setClearModalVisible(false)
        clearForm.resetFields()
        loadCacheData()
      }
    } catch (error) {
      message.error('清理缓存失败')
    }
  }

  const handleInvalidateNode = async (values: any) => {
    try {
      const result = await cacheService.invalidateNodeCache(
        values.node_name,
        values.user_id,
        values.session_id,
        values.workflow_id
      )
      if (result.success) {
        message.success(result.message)
      } else {
        message.warning(result.message || '未找到可失效的缓存')
      }
      setInvalidateModalVisible(false)
      invalidateForm.resetFields()
      loadCacheData()
    } catch (error) {
      message.error('失效缓存失败')
    }
  }

  const handleUpdateStrategy = async (values: any) => {
    try {
      const success = await cacheService.updateStrategy(values)
      if (success) {
        message.success('缓存策略更新成功')
        setStrategyModalVisible(false)
        strategyForm.resetFields()
        loadCacheData()
      }
    } catch (error) {
      message.error('更新策略失败')
    }
  }

  const handleWarmCache = async (values: any) => {
    try {
      const keys = values.keys.split(',').map((k: string) => k.trim())
      const result = await cacheService.warmCache(keys)
      if (result.success) {
        message.success(`成功预热 ${result.warmed_count} 个缓存键`)
        setWarmModalVisible(false)
        warmForm.resetFields()
      }
    } catch (error) {
      message.error('预热缓存失败')
    }
  }

  // 获取健康状态图标
  const getHealthIcon = (status: string) => {
    switch (status) {
      case 'healthy':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />
      case 'degraded':
        return <ExclamationCircleOutlined style={{ color: '#faad14' }} />
      case 'unhealthy':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
      default:
        return <InfoCircleOutlined />
    }
  }

  // 节点表格列配置
  const nodeColumns = [
    {
      title: '节点名称',
      dataIndex: 'name',
      key: 'name',
      render: (name: string) => <Text strong>{name}</Text>
    },
    {
      title: '命中次数',
      dataIndex: 'hits',
      key: 'hits',
      sorter: (a: any, b: any) => a.hits - b.hits
    },
    {
      title: '未命中次数',
      dataIndex: 'misses',
      key: 'misses',
      sorter: (a: any, b: any) => a.misses - b.misses
    },
    {
      title: '命中率',
      key: 'hit_rate',
      render: (record: any) => {
        const rate = record.hits / (record.hits + record.misses) * 100
        return (
          <Progress
            percent={rate}
            size="small"
            strokeColor={rate > 80 ? '#52c41a' : '#faad14'}
          />
        )
      }
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
      render: (size: number) => `${(size / 1024 / 1024).toFixed(2)} MB`
    },
    {
      title: '条目数',
      dataIndex: 'items',
      key: 'items'
    },
    {
      title: '最后访问',
      dataIndex: 'last_accessed',
      key: 'last_accessed',
      render: (time: string) => new Date(time).toLocaleString()
    }
  ]

  return (
    <div className="p-6">
      <Spin spinning={loading}>
        <div className="mb-6">
          <div className="flex justify-between items-center mb-4">
            <Title level={2}>缓存监控 (LangGraph缓存系统)</Title>
            <Space>
              <Button 
                icon={<SyncOutlined spin={loading} />}
                onClick={loadCacheData}
                loading={loading}
              >
                刷新
              </Button>
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
                onClick={() => setClearModalVisible(true)}
              >
                清理缓存
              </Button>
              <Button 
                icon={<WarningOutlined />}
                onClick={() => setInvalidateModalVisible(true)}
              >
                失效节点缓存
              </Button>
              <Button 
                icon={<FireOutlined />}
                onClick={() => setWarmModalVisible(true)}
              >
                缓存预热
              </Button>
              <Button 
                icon={<SettingOutlined />}
                onClick={() => {
                  setStrategyModalVisible(true)
                  if (strategy) {
                    strategyForm.setFieldsValue(strategy)
                  }
                }}
              >
                策略设置
              </Button>
            </Space>
          </div>

          {health?.status === 'unhealthy' && (
            <Alert
              message="缓存服务异常"
              description={health.issues.join(', ')}
              type="error"
              showIcon
              closable
              className="mb-4"
            />
          )}

          {health?.status === 'degraded' && (
            <Alert
              message="缓存服务降级"
              description={health.recommendations.join(', ')}
              type="warning"
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
                  value={stats?.hit_rate || 0}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: (stats?.hit_rate || 0) > 80 ? '#3f8600' : '#cf1322' }}
                  prefix={<ThunderboltOutlined />}
                />
                <Progress 
                  percent={stats?.hit_rate || 0} 
                  strokeColor={(stats?.hit_rate || 0) > 80 ? '#52c41a' : '#ff4d4f'}
                  size="small"
                  className="mt-2"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="总请求数"
                  value={(stats?.total_hits || 0) + (stats?.total_misses || 0)}
                  prefix={<DatabaseOutlined />}
                />
                <div className="mt-2 text-xs text-gray-500">
                  命中: {stats?.total_hits || 0} | 未命中: {stats?.total_misses || 0}
                </div>
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="缓存大小"
                  value={stats?.memory_usage_mb || 0}
                  suffix="MB"
                  prefix={<DatabaseOutlined />}
                />
                <Progress 
                  percent={Math.min(100, ((stats?.memory_usage_mb || 0) / (stats?.max_memory_mb || 1024)) * 100)} 
                  size="small"
                  className="mt-2"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="健康状态"
                  value={health?.status || 'unknown'}
                  prefix={getHealthIcon(health?.status || 'unknown')}
                />
                <div className="mt-2 text-xs text-gray-500">
                  响应时间: {health?.metrics.avg_response_time_ms || 0}ms
                </div>
              </Card>
            </Col>
          </Row>

          <Card className="mb-6">
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              <TabPane tab="概览" key="overview">
                {stats && performance ? (
                  <>
                    <Row gutter={16}>
                      <Col span={12}>
                        <Card title="缓存命中率" size="small">
                          <Progress type="circle" percent={stats.hit_rate} />
                        </Card>
                      </Col>
                      <Col span={12}>
                        <Card title="响应时间分布" size="small">
                          <Descriptions column={2}>
                            <Descriptions.Item label="平均">
                              {performance.response_times.avg_ms}ms
                            </Descriptions.Item>
                            <Descriptions.Item label="P50">
                              {performance.response_times.p50_ms}ms
                            </Descriptions.Item>
                            <Descriptions.Item label="P95">
                              {performance.response_times.p95_ms}ms
                            </Descriptions.Item>
                            <Descriptions.Item label="P99">
                              {performance.response_times.p99_ms}ms
                            </Descriptions.Item>
                          </Descriptions>
                        </Card>
                      </Col>
                    </Row>
                    <Divider />
                    <Row gutter={16}>
                      <Col span={8}>
                        <Card title="吞吐量" size="small">
                          <Descriptions column={1}>
                            <Descriptions.Item label="读取/秒">
                              {performance.throughput.reads_per_second}
                            </Descriptions.Item>
                            <Descriptions.Item label="写入/秒">
                              {performance.throughput.writes_per_second}
                            </Descriptions.Item>
                          </Descriptions>
                        </Card>
                      </Col>
                      <Col span={8}>
                        <Card title="内存使用" size="small">
                          <Descriptions column={1}>
                            <Descriptions.Item label="已用">
                              {performance.memory.used_mb} MB
                            </Descriptions.Item>
                            <Descriptions.Item label="可用">
                              {performance.memory.available_mb} MB
                            </Descriptions.Item>
                            <Descriptions.Item label="碎片率">
                              {(performance.memory.fragmentation_ratio * 100).toFixed(1)}%
                            </Descriptions.Item>
                          </Descriptions>
                        </Card>
                      </Col>
                      <Col span={8}>
                        <Card title="缓存效率" size="small">
                          <Descriptions column={1}>
                            <Descriptions.Item label="效率">
                              {(stats.cache_efficiency * 100).toFixed(1)}%
                            </Descriptions.Item>
                            <Descriptions.Item label="驱逐次数">
                              {stats.evictions}
                            </Descriptions.Item>
                            <Descriptions.Item label="过期条目">
                              {stats.expired_items}
                            </Descriptions.Item>
                          </Descriptions>
                        </Card>
                      </Col>
                    </Row>
                  </>
                ) : (
                  <Empty description="暂无概览数据" />
                )}
              </TabPane>

              <TabPane tab="节点详情" key="nodes">
                {stats?.nodes && Object.keys(stats.nodes).length > 0 ? (
                  <Table
                    columns={nodeColumns}
                    dataSource={Object.entries(stats.nodes).map(([name, data]) => ({
                      name,
                      ...data,
                      key: name
                    }))}
                    pagination={{ pageSize: 10 }}
                  />
                ) : (
                  <Empty description="暂无节点数据" />
                )}
              </TabPane>

              <TabPane tab="性能指标" key="performance">
                {performance ? (
                  <Row gutter={[16, 16]}>
                    <Col span={12}>
                      <Card title="响应时间">
                        <Descriptions column={2}>
                          <Descriptions.Item label="平均">
                            {performance.response_times.avg_ms}ms
                          </Descriptions.Item>
                          <Descriptions.Item label="P50">
                            {performance.response_times.p50_ms}ms
                          </Descriptions.Item>
                          <Descriptions.Item label="P95">
                            {performance.response_times.p95_ms}ms
                          </Descriptions.Item>
                          <Descriptions.Item label="P99">
                            {performance.response_times.p99_ms}ms
                          </Descriptions.Item>
                        </Descriptions>
                      </Card>
                    </Col>
                    <Col span={12}>
                      <Card title="操作统计">
                        <Descriptions column={2}>
                          <Descriptions.Item label="总读取">
                            {performance.operations.total_reads}
                          </Descriptions.Item>
                          <Descriptions.Item label="总写入">
                            {performance.operations.total_writes}
                          </Descriptions.Item>
                          <Descriptions.Item label="总删除">
                            {performance.operations.total_deletes}
                          </Descriptions.Item>
                          <Descriptions.Item label="失败操作">
                            <Badge count={performance.operations.failed_operations} />
                          </Descriptions.Item>
                        </Descriptions>
                      </Card>
                    </Col>
                  </Row>
                ) : (
                  <Empty description="暂无性能数据" />
                )}
              </TabPane>

              <TabPane tab="健康检查" key="health">
                {health ? (
                  <div>
                    <Alert
                      message={`健康状态: ${health.status}`}
                      type={
                        health.status === 'healthy' ? 'success' :
                        health.status === 'degraded' ? 'warning' : 'error'
                      }
                      showIcon
                      style={{ marginBottom: 16 }}
                    />
                    <Row gutter={[16, 16]}>
                      <Col span={12}>
                        <Card title="检查项">
                          <List
                            dataSource={Object.entries(health.checks)}
                            renderItem={([key, value]) => (
                              <List.Item>
                                <Space>
                                  {value ? (
                                    <CheckCircleOutlined style={{ color: '#52c41a' }} />
                                  ) : (
                                    <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
                                  )}
                                  <span>{key.replace(/_/g, ' ').toUpperCase()}</span>
                                </Space>
                              </List.Item>
                            )}
                          />
                        </Card>
                      </Col>
                      <Col span={12}>
                        <Card title="建议">
                          <List
                            dataSource={health.recommendations}
                            renderItem={item => (
                              <List.Item>
                                <Space>
                                  <BulbOutlined style={{ color: '#faad14' }} />
                                  {item}
                                </Space>
                              </List.Item>
                            )}
                          />
                        </Card>
                      </Col>
                    </Row>
                  </div>
                ) : (
                  <Empty description="暂无健康检查数据" />
                )}
              </TabPane>
            </Tabs>
          </Card>
        </div>
      </Spin>

      {/* 清理缓存弹窗 */}
      <Modal
        title="清理缓存"
        visible={clearModalVisible}
        onCancel={() => setClearModalVisible(false)}
        onOk={() => clearForm.submit()}
        width={500}
      >
        <Form
          form={clearForm}
          layout="vertical"
          onFinish={handleClearCache}
          initialValues={{ pattern: '*' }}
        >
          <Alert
            message="警告"
            description="清理缓存可能会影响系统性能，请谨慎操作"
            type="warning"
            showIcon
            style={{ marginBottom: 16 }}
          />
          <Form.Item
            name="pattern"
            label="匹配模式"
            rules={[{ required: true, message: '请输入匹配模式' }]}
            extra="使用 * 匹配所有，支持通配符"
          >
            <Input placeholder="例如: workflow_*, *_cache" />
          </Form.Item>
        </Form>
      </Modal>

      {/* 失效节点缓存弹窗 */}
      <Modal
        title="失效节点缓存"
        visible={invalidateModalVisible}
        onCancel={() => setInvalidateModalVisible(false)}
        onOk={() => invalidateForm.submit()}
        width={500}
      >
        <Form
          form={invalidateForm}
          layout="vertical"
          onFinish={handleInvalidateNode}
        >
          <Form.Item
            name="node_name"
            label="节点名称"
            rules={[{ required: true, message: '请输入节点名称' }]}
          >
            <Input placeholder="例如: langgraph_node_1" />
          </Form.Item>
          <Form.Item
            name="user_id"
            label="用户ID（可选）"
          >
            <Input placeholder="特定用户的缓存" />
          </Form.Item>
          <Form.Item
            name="session_id"
            label="会话ID（可选）"
          >
            <Input placeholder="特定会话的缓存" />
          </Form.Item>
          <Form.Item
            name="workflow_id"
            label="工作流ID（可选）"
          >
            <Input placeholder="特定工作流的缓存" />
          </Form.Item>
        </Form>
      </Modal>

      {/* 策略设置弹窗 */}
      <Modal
        title="缓存策略设置"
        visible={strategyModalVisible}
        onCancel={() => setStrategyModalVisible(false)}
        onOk={() => strategyForm.submit()}
        width={500}
      >
        <Form
          form={strategyForm}
          layout="vertical"
          onFinish={handleUpdateStrategy}
        >
          <Form.Item
            name="eviction_policy"
            label="驱逐策略"
            rules={[{ required: true }]}
          >
            <Select>
              <Option value="LRU">LRU (最近最少使用)</Option>
              <Option value="LFU">LFU (最不经常使用)</Option>
              <Option value="FIFO">FIFO (先进先出)</Option>
              <Option value="TTL">TTL (基于过期时间)</Option>
              <Option value="NOEVICTION">NOEVICTION (不淘汰)</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="max_size_mb"
            label="最大缓存大小(MB)"
            rules={[{ required: true }]}
          >
            <InputNumber min={0} max={10240} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item
            name="default_ttl_seconds"
            label="默认TTL(秒)"
            rules={[{ required: true }]}
          >
            <InputNumber min={60} max={86400} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item
            name="compression_enabled"
            label="启用压缩"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
          <Form.Item
            name="warming_enabled"
            label="启用预热"
            valuePropName="checked"
          >
            <Switch />
          </Form.Item>
        </Form>
      </Modal>

      {/* 缓存预热弹窗 */}
      <Modal
        title="缓存预热"
        visible={warmModalVisible}
        onCancel={() => setWarmModalVisible(false)}
        onOk={() => warmForm.submit()}
        width={500}
      >
        <Form
          form={warmForm}
          layout="vertical"
          onFinish={handleWarmCache}
        >
          <Alert
            message="提示"
            description="预热缓存可以提前加载常用数据，提高访问速度"
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />
          <Form.Item
            name="keys"
            label="缓存键列表"
            rules={[{ required: true, message: '请输入缓存键' }]}
            extra="多个键用逗号分隔"
          >
            <Input.TextArea 
              rows={4} 
              placeholder="例如: user_1234, workflow_abc, session_xyz" 
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default CacheMonitorPage
