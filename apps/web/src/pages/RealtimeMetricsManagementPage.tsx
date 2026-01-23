import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Button,
  Table,
  Space,
  Typography,
  Tabs,
  message,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Progress,
  Tag,
  Statistic,
  Row,
  Col,
  Switch,
  Popconfirm,
  Tooltip,
  Divider,
  Alert,
  Timeline,
} from 'antd'
import {
  LineChartOutlined,
  BarChartOutlined,
  DashboardOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  PlusOutlined,
  SettingOutlined,
  EyeOutlined,
  DiffOutlined,
  ReloadOutlined,
  ExperimentOutlined,
  MonitorOutlined,
  AlertOutlined,
  RiseOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
} from '@ant-design/icons'
import { realtimeMetricsService } from '../services/realtimeMetricsService'
import { experimentService } from '../services/experimentService'
import MetricChart from '../components/charts/MetricChart'

const { Title, Text } = Typography
const { TabPane } = Tabs
const { Option } = Select

const RealtimeMetricsManagementPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('dashboard')
  const [metricsCatalog, setMetricsCatalog] = useState<any>(null)
  const [experiments, setExperiments] = useState<any[]>([])
  const [selectedExperiment, setSelectedExperiment] = useState<string>('')
  const [metricsData, setMetricsData] = useState<any>(null)
  const [comparisonData, setComparisonData] = useState<any>(null)
  const [trendsData, setTrendsData] = useState<any>(null)
  const [healthStatus, setHealthStatus] = useState<any>(null)
  const [monitoringStatus, setMonitoringStatus] = useState<string>('stopped')

  // 表单状态
  const [registerForm] = Form.useForm()
  const [calculationForm] = Form.useForm()
  const [comparisonForm] = Form.useForm()
  const [trendsForm] = Form.useForm()

  // Modal 状态
  const [registerModalVisible, setRegisterModalVisible] = useState(false)
  const [calculationModalVisible, setCalculationModalVisible] = useState(false)
  const [comparisonModalVisible, setComparisonModalVisible] = useState(false)
  const [trendsModalVisible, setTrendsModalVisible] = useState(false)

  // 获取指标目录
  const fetchMetricsCatalog = async () => {
    try {
      setLoading(true)
      const response = await realtimeMetricsService.getRealtimeMetricsCatalog()
      setMetricsCatalog(response)
    } catch (error) {
      message.error('获取指标目录失败')
      logger.error('获取指标目录失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 获取健康状态
  const fetchHealthStatus = async () => {
    try {
      const response = await realtimeMetricsService.realtimeHealthCheck()
      setHealthStatus(response)
    } catch (error) {
      message.error('获取健康状态失败')
      logger.error('获取健康状态失败:', error)
    }
  }

  // 获取实验摘要
  const fetchExperimentSummary = async (experimentId: string) => {
    try {
      setLoading(true)
      const response =
        await realtimeMetricsService.getRealtimeExperimentSummary(experimentId)
      setMetricsData(response)
    } catch (error) {
      message.error('获取实验摘要失败')
      logger.error('获取实验摘要失败:', error)
    } finally {
      setLoading(false)
    }
  }

  const fetchExperiments = async () => {
    try {
      const res = await experimentService.listExperiments({ pageSize: 50 })
      setExperiments(res.experiments || [])
      if (res.experiments?.length && !selectedExperiment) {
        setSelectedExperiment(
          res.experiments[0].id || res.experiments[0].experiment_id || ''
        )
      }
    } catch (error) {
      setExperiments([])
    }
  }

  useEffect(() => {
    fetchMetricsCatalog()
    fetchHealthStatus()
    fetchExperiments()
  }, [])

  // 注册指标
  const handleRegisterMetric = async (values: any) => {
    try {
      setLoading(true)
      await realtimeMetricsService.registerMetric(values)
      message.success('指标注册成功')
      setRegisterModalVisible(false)
      registerForm.resetFields()
      fetchMetricsCatalog()
    } catch (error) {
      message.error('指标注册失败')
      logger.error('指标注册失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 计算指标
  const handleCalculateMetrics = async (values: any) => {
    try {
      setLoading(true)
      const response =
        await realtimeMetricsService.calculateExperimentMetrics(values)
      setMetricsData(response)
      message.success('指标计算完成')
      setCalculationModalVisible(false)
    } catch (error) {
      message.error('指标计算失败')
      logger.error('指标计算失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 比较实验组
  const handleCompareGroups = async (values: any) => {
    try {
      setLoading(true)
      const response =
        await realtimeMetricsService.compareExperimentGroups(values)
      setComparisonData(response)
      message.success('分组比较完成')
      setComparisonModalVisible(false)
    } catch (error) {
      message.error('分组比较失败')
      logger.error('分组比较失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 获取趋势数据
  const handleGetTrends = async (values: any) => {
    try {
      setLoading(true)
      const response =
        await realtimeMetricsService.getExperimentMetricTrends(values)
      setTrendsData(response)
      message.success('趋势数据获取完成')
      setTrendsModalVisible(false)
    } catch (error) {
      message.error('获取趋势数据失败')
      logger.error('获取趋势数据失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 启动/停止监控
  const handleToggleMonitoring = async () => {
    try {
      if (monitoringStatus === 'stopped') {
        if (!selectedExperiment) {
          message.warning('请先选择实验')
          return
        }
        await realtimeMetricsService.startRealtimeMonitoring(selectedExperiment)
        setMonitoringStatus('running')
        message.success('实时监控已启动')
      } else {
        await realtimeMetricsService.stopRealtimeMonitoring()
        setMonitoringStatus('stopped')
        message.success('实时监控已停止')
      }
    } catch (error) {
      message.error('监控状态切换失败')
      logger.error('监控状态切换失败:', error)
    }
  }

  // 指标目录表格列
  const catalogColumns = [
    {
      title: '指标名称',
      dataIndex: 'display_name',
      key: 'display_name',
      render: (text: string, record: any) => (
        <Space>
          <Text strong>{text}</Text>
          <Tag
            color={
              record.category === 'primary'
                ? 'blue'
                : record.category === 'secondary'
                  ? 'green'
                  : record.category === 'guardrail'
                    ? 'orange'
                    : 'purple'
            }
          >
            {record.category}
          </Tag>
        </Space>
      ),
    },
    {
      title: '指标类型',
      dataIndex: 'metric_type',
      key: 'metric_type',
    },
    {
      title: '聚合方式',
      dataIndex: 'aggregation',
      key: 'aggregation',
    },
    {
      title: '单位',
      dataIndex: 'unit',
      key: 'unit',
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      ellipsis: true,
    },
  ]

  // 仪表板标签页
  const renderDashboardTab = () => (
    <div>
      {/* 系统状态卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="系统状态"
              value={healthStatus?.status === 'healthy' ? '正常' : '异常'}
              prefix={
                healthStatus?.status === 'healthy' ? (
                  <CheckCircleOutlined style={{ color: '#52c41a' }} />
                ) : (
                  <ExclamationCircleOutlined style={{ color: '#ff4d4f' }} />
                )
              }
              valueStyle={{
                color:
                  healthStatus?.status === 'healthy' ? '#52c41a' : '#ff4d4f',
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="注册指标数"
              value={metricsCatalog?.total_metrics || 0}
              prefix={<DashboardOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="监控状态"
              value={monitoringStatus === 'running' ? '运行中' : '已停止'}
              prefix={
                monitoringStatus === 'running' ? (
                  <PlayCircleOutlined style={{ color: '#52c41a' }} />
                ) : (
                  <PauseCircleOutlined style={{ color: '#faad14' }} />
                )
              }
              valueStyle={{
                color: monitoringStatus === 'running' ? '#52c41a' : '#faad14',
              }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Redis状态"
              value={healthStatus?.redis_status || 'unknown'}
              prefix={<MonitorOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 实验选择和控制 */}
      <Card title="实验控制" style={{ marginBottom: 24 }}>
        <Space size="large">
          <div>
            <Text>选择实验: </Text>
            <Select
              style={{ width: 300 }}
              placeholder="选择要监控的实验"
              value={selectedExperiment}
              onChange={setSelectedExperiment}
            >
              {experiments.map(exp => (
                <Option
                  key={exp.id || exp.experiment_id}
                  value={exp.id || exp.experiment_id}
                >
                  {exp.name ||
                    exp.experiment_name ||
                    exp.id ||
                    exp.experiment_id}
                </Option>
              ))}
            </Select>
          </div>
          <Button
            type={monitoringStatus === 'running' ? 'default' : 'primary'}
            icon={
              monitoringStatus === 'running' ? (
                <PauseCircleOutlined />
              ) : (
                <PlayCircleOutlined />
              )
            }
            onClick={handleToggleMonitoring}
            disabled={!selectedExperiment}
          >
            {monitoringStatus === 'running' ? '停止监控' : '启动监控'}
          </Button>
          <Button
            icon={<ReloadOutlined />}
            onClick={() =>
              selectedExperiment && fetchExperimentSummary(selectedExperiment)
            }
            disabled={!selectedExperiment}
          >
            刷新数据
          </Button>
        </Space>
      </Card>

      {/* 实验指标摘要 */}
      {metricsData && (
        <Card title="实验指标摘要" style={{ marginBottom: 24 }}>
          <Row gutter={[16, 16]}>
            {Object.entries(metricsData.summary?.groups || {}).map(
              ([groupId, group]: [string, any]) => (
                <Col span={8} key={groupId}>
                  <Card size="small" title={`分组: ${groupId}`}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Statistic title="用户数" value={group.user_count} />
                      <Statistic title="事件数" value={group.event_count} />
                      <Statistic title="指标数" value={group.metrics_count} />
                    </Space>
                  </Card>
                </Col>
              )
            )}
          </Row>
        </Card>
      )}
    </div>
  )

  // 指标管理标签页
  const renderMetricsTab = () => (
    <div>
      <Card
        title="指标目录"
        extra={
          <Button
            type="primary"
            icon={<PlusOutlined />}
            onClick={() => setRegisterModalVisible(true)}
          >
            注册新指标
          </Button>
        }
      >
        {metricsCatalog && (
          <Tabs defaultActiveKey="all">
            <TabPane tab="全部指标" key="all">
              <Table
                columns={catalogColumns}
                dataSource={[
                  ...metricsCatalog.catalog.primary,
                  ...metricsCatalog.catalog.secondary,
                  ...metricsCatalog.catalog.guardrail,
                  ...metricsCatalog.catalog.diagnostic,
                ]}
                rowKey="name"
                loading={loading}
                pagination={{ pageSize: 10 }}
              />
            </TabPane>
            <TabPane
              tab={`主要指标 (${metricsCatalog.catalog.primary.length})`}
              key="primary"
            >
              <Table
                columns={catalogColumns}
                dataSource={metricsCatalog.catalog.primary}
                rowKey="name"
                loading={loading}
                pagination={{ pageSize: 10 }}
              />
            </TabPane>
            <TabPane
              tab={`次要指标 (${metricsCatalog.catalog.secondary.length})`}
              key="secondary"
            >
              <Table
                columns={catalogColumns}
                dataSource={metricsCatalog.catalog.secondary}
                rowKey="name"
                loading={loading}
                pagination={{ pageSize: 10 }}
              />
            </TabPane>
            <TabPane
              tab={`护栏指标 (${metricsCatalog.catalog.guardrail.length})`}
              key="guardrail"
            >
              <Table
                columns={catalogColumns}
                dataSource={metricsCatalog.catalog.guardrail}
                rowKey="name"
                loading={loading}
                pagination={{ pageSize: 10 }}
              />
            </TabPane>
          </Tabs>
        )}
      </Card>
    </div>
  )

  // 分析工具标签页
  const renderAnalysisTab = () => (
    <div>
      <Row gutter={[16, 16]}>
        <Col span={8}>
          <Card
            title="指标计算"
            actions={[
              <Button
                key="calculate"
                type="primary"
                icon={<BarChartOutlined />}
                onClick={() => setCalculationModalVisible(true)}
              >
                开始计算
              </Button>,
            ]}
          >
            <Text>
              计算指定实验的所有指标数据，支持不同时间窗口的聚合分析。
            </Text>
          </Card>
        </Col>
        <Col span={8}>
          <Card
            title="分组比较"
            actions={[
              <Button
                key="compare"
                type="primary"
                icon={<DiffOutlined />}
                onClick={() => setComparisonModalVisible(true)}
              >
                开始比较
              </Button>,
            ]}
          >
            <Text>对比实验组和对照组的指标差异，计算统计显著性。</Text>
          </Card>
        </Col>
        <Col span={8}>
          <Card
            title="趋势分析"
            actions={[
              <Button
                key="trends"
                type="primary"
                icon={<RiseOutlined />}
                onClick={() => setTrendsModalVisible(true)}
              >
                查看趋势
              </Button>,
            ]}
          >
            <Text>分析指标随时间的变化趋势，识别模式和异常。</Text>
          </Card>
        </Col>
      </Row>

      {/* 比较结果 */}
      {comparisonData && (
        <Card title="分组比较结果" style={{ marginTop: 24 }}>
          <Alert
            message={`比较完成: ${comparisonData.summary?.significant_metrics} / ${comparisonData.summary?.total_metrics} 个指标具有统计显著性`}
            type="info"
            style={{ marginBottom: 16 }}
          />
          <Table
            columns={[
              {
                title: '指标名称',
                dataIndex: 'metric_name',
                key: 'metric_name',
              },
              {
                title: '对照组值',
                dataIndex: 'control_value',
                key: 'control_value',
                render: (val: number) => val.toFixed(4),
              },
              {
                title: '实验组值',
                dataIndex: 'treatment_value',
                key: 'treatment_value',
                render: (val: number) => val.toFixed(4),
              },
              {
                title: '绝对差异',
                dataIndex: 'absolute_difference',
                key: 'absolute_difference',
                render: (val: number) => val.toFixed(4),
              },
              {
                title: '相对差异',
                dataIndex: 'relative_difference',
                key: 'relative_difference',
                render: (val: number) => `${val.toFixed(2)}%`,
              },
              {
                title: '显著性',
                dataIndex: 'is_significant',
                key: 'is_significant',
                render: (significant: boolean) => (
                  <Tag color={significant ? 'green' : 'red'}>
                    {significant ? '显著' : '不显著'}
                  </Tag>
                ),
              },
            ]}
            dataSource={Object.values(comparisonData.comparisons || {})}
            rowKey="metric_name"
            pagination={false}
          />
        </Card>
      )}

      {/* 趋势图表 */}
      {trendsData && (
        <Card title="指标趋势" style={{ marginTop: 24 }}>
          <MetricChart
            type="line"
            series={[
              {
                name: trendsData.metric_name,
                data: (trendsData.trends || []).map((t: any) => ({
                  x: t.timestamp,
                  y: t.value,
                  yMin: t.confidence_interval?.[0],
                  yMax: t.confidence_interval?.[1],
                  metadata: { sample_size: t.sample_size },
                })),
              },
            ]}
            config={{
              showLegend: false,
              confidenceInterval: true,
              height: 300,
            }}
          />
        </Card>
      )}
    </div>
  )

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <Title level={2}>
          <LineChartOutlined style={{ marginRight: 12 }} />
          实时指标监控系统
        </Title>
        <Text type="secondary">
          企业级实时指标监控平台，支持A/B测试指标计算、分组比较和趋势分析
        </Text>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab} size="large">
        <TabPane
          tab={
            <span>
              <DashboardOutlined />
              实时仪表板
            </span>
          }
          key="dashboard"
        >
          {renderDashboardTab()}
        </TabPane>
        <TabPane
          tab={
            <span>
              <SettingOutlined />
              指标管理
            </span>
          }
          key="metrics"
        >
          {renderMetricsTab()}
        </TabPane>
        <TabPane
          tab={
            <span>
              <BarChartOutlined />
              分析工具
            </span>
          }
          key="analysis"
        >
          {renderAnalysisTab()}
        </TabPane>
      </Tabs>

      {/* 注册指标模态框 */}
      <Modal
        title="注册新指标"
        visible={registerModalVisible}
        onCancel={() => setRegisterModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form
          form={registerForm}
          layout="vertical"
          onFinish={handleRegisterMetric}
        >
          <Form.Item name="name" label="指标名称" rules={[{ required: true }]}>
            <Input placeholder="输入指标的唯一标识名称" />
          </Form.Item>
          <Form.Item
            name="display_name"
            label="显示名称"
            rules={[{ required: true }]}
          >
            <Input placeholder="输入指标的显示名称" />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="metric_type"
                label="指标类型"
                rules={[{ required: true }]}
              >
                <Select placeholder="选择指标类型">
                  <Option value="conversion">转化率</Option>
                  <Option value="continuous">连续值</Option>
                  <Option value="count">计数</Option>
                  <Option value="ratio">比率</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="category"
                label="指标类别"
                rules={[{ required: true }]}
              >
                <Select placeholder="选择指标类别">
                  <Option value="primary">主要指标</Option>
                  <Option value="secondary">次要指标</Option>
                  <Option value="guardrail">护栏指标</Option>
                  <Option value="diagnostic">诊断指标</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Form.Item
            name="aggregation"
            label="聚合方式"
            rules={[{ required: true }]}
          >
            <Select placeholder="选择聚合方式">
              <Option value="sum">求和</Option>
              <Option value="avg">平均值</Option>
              <Option value="count">计数</Option>
              <Option value="rate">比率</Option>
            </Select>
          </Form.Item>
          <Form.Item name="unit" label="单位">
            <Input placeholder="输入指标单位（可选）" />
          </Form.Item>
          <Form.Item name="description" label="描述">
            <Input.TextArea placeholder="输入指标描述" rows={3} />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="threshold_lower" label="下限阈值">
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="设置下限阈值"
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="threshold_upper" label="上限阈值">
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="设置上限阈值"
                />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                注册指标
              </Button>
              <Button onClick={() => setRegisterModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 指标计算模态框 */}
      <Modal
        title="计算实验指标"
        visible={calculationModalVisible}
        onCancel={() => setCalculationModalVisible(false)}
        footer={null}
      >
        <Form
          form={calculationForm}
          layout="vertical"
          onFinish={handleCalculateMetrics}
        >
          <Form.Item
            name="experiment_id"
            label="实验ID"
            rules={[{ required: true }]}
          >
            <Select placeholder="选择实验">
              {experiments.map(exp => (
                <Option key={exp.id} value={exp.id}>
                  {exp.name}
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item
            name="time_window"
            label="时间窗口"
            rules={[{ required: true }]}
          >
            <Select placeholder="选择时间窗口">
              <Option value="cumulative">累计</Option>
              <Option value="hourly">小时</Option>
              <Option value="daily">天</Option>
              <Option value="weekly">周</Option>
            </Select>
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                开始计算
              </Button>
              <Button onClick={() => setCalculationModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 分组比较模态框 */}
      <Modal
        title="分组比较分析"
        visible={comparisonModalVisible}
        onCancel={() => setComparisonModalVisible(false)}
        footer={null}
      >
        <Form
          form={comparisonForm}
          layout="vertical"
          onFinish={handleCompareGroups}
        >
          <Form.Item
            name="experiment_id"
            label="实验ID"
            rules={[{ required: true }]}
          >
            <Select placeholder="选择实验">
              {experiments.map(exp => (
                <Option key={exp.id} value={exp.id}>
                  {exp.name}
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="control_group"
                label="对照组"
                rules={[{ required: true }]}
              >
                <Input placeholder="输入对照组ID" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="treatment_group"
                label="实验组"
                rules={[{ required: true }]}
              >
                <Input placeholder="输入实验组ID" />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                开始比较
              </Button>
              <Button onClick={() => setComparisonModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 趋势分析模态框 */}
      <Modal
        title="指标趋势分析"
        visible={trendsModalVisible}
        onCancel={() => setTrendsModalVisible(false)}
        footer={null}
      >
        <Form form={trendsForm} layout="vertical" onFinish={handleGetTrends}>
          <Form.Item
            name="experiment_id"
            label="实验ID"
            rules={[{ required: true }]}
          >
            <Select placeholder="选择实验">
              {experiments.map(exp => (
                <Option key={exp.id} value={exp.id}>
                  {exp.name}
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item
            name="metric_name"
            label="指标名称"
            rules={[{ required: true }]}
          >
            <Input placeholder="输入指标名称" />
          </Form.Item>
          <Form.Item
            name="granularity"
            label="时间粒度"
            rules={[{ required: true }]}
          >
            <Select placeholder="选择时间粒度">
              <Option value="hourly">小时</Option>
              <Option value="daily">天</Option>
              <Option value="weekly">周</Option>
            </Select>
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                获取趋势
              </Button>
              <Button onClick={() => setTrendsModalVisible(false)}>取消</Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default RealtimeMetricsManagementPage
