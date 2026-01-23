import React, { useState, useEffect } from 'react'
import { logger } from '../../utils/logger'
import {
  Card,
  Row,
  Col,
  Statistic,
  Typography,
  Space,
  Button,
  Table,
  Tag,
  Progress,
  Tabs,
  Select,
  DatePicker,
  Modal,
  Form,
  Input,
  InputNumber,
  Switch,
  Alert,
  Timeline,
  Tooltip,
  Divider,
  message,
  List,
  Avatar,
  Badge,
  Descriptions,
  Drawer,
  Upload,
  Radio,
} from 'antd'
import {
  ExperimentOutlined,
  BarChartOutlined,
  TrophyOutlined,
  TeamOutlined,
  SettingOutlined,
  DownloadOutlined,
  ShareAltOutlined,
  BellOutlined,
  SecurityScanOutlined,
  RobotOutlined,
  DollarOutlined,
  CameraOutlined,
  CommentOutlined,
  LockOutlined,
  HistoryOutlined,
  BulbOutlined,
  CompareOutlined,
  ThunderboltOutlined,
  LineChartOutlined,
  FunnelPlotOutlined,
  UsergroupAddOutlined,
  AlertOutlined,
  FileTextOutlined,
  CopyOutlined,
} from '@ant-design/icons'
import {
  experimentServiceEnhanced,
  ExperimentMetadata,
  ExperimentAnalysis,
} from '../../services/experimentServiceEnhanced'

const { Title, Text, Paragraph } = Typography
const { TabPane } = Tabs
const { RangePicker } = DatePicker

const EnhancedExperimentAnalysisPage: React.FC = () => {
  const [experiments, setExperiments] = useState<ExperimentMetadata[]>([])
  const [selectedExperiment, setSelectedExperiment] = useState<string>('')
  const [analysis, setAnalysis] = useState<ExperimentAnalysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('analysis')

  // 模态框状态
  const [templateModalVisible, setTemplateModalVisible] = useState(false)
  const [optimizationModalVisible, setOptimizationModalVisible] =
    useState(false)
  const [segmentModalVisible, setSegmentModalVisible] = useState(false)
  const [recommendationModalVisible, setRecommendationModalVisible] =
    useState(false)
  const [automationModalVisible, setAutomationModalVisible] = useState(false)
  const [compareModalVisible, setCompareModalVisible] = useState(false)

  // 数据状态
  const [templates, setTemplates] = useState([])
  const [monitoring, setMonitoring] = useState<any>(null)
  const [recommendations, setRecommendations] = useState<any>(null)
  const [costAnalysis, setCostAnalysis] = useState<any>(null)
  const [comments, setComments] = useState<any[]>([])
  const [auditLog, setAuditLog] = useState<any[]>([])

  useEffect(() => {
    loadExperiments()
    loadTemplates()
  }, [])

  const loadExperiments = async () => {
    try {
      setLoading(true)
      const response = await experimentServiceEnhanced.searchExperiments({
        filters: { status: ['running', 'completed'] },
        pagination: { page: 1, limit: 20 },
      })
      setExperiments(response.experiments)
      if (response.experiments.length > 0) {
        setSelectedExperiment(response.experiments[0].id)
        loadExperimentData(response.experiments[0].id)
      }
    } catch (error) {
      message.error('加载实验列表失败')
    } finally {
      setLoading(false)
    }
  }

  const loadTemplates = async () => {
    try {
      const templates = await experimentServiceEnhanced.getExperimentTemplates()
      setTemplates(templates)
    } catch (error) {
      logger.error('加载模板失败:', error)
    }
  }

  const loadExperimentData = async (experimentId: string) => {
    try {
      setLoading(true)
      const [analysisData, monitoringData, costData, commentsData, auditData] =
        await Promise.all([
          experimentServiceEnhanced.getExperimentAnalysis(experimentId),
          experimentServiceEnhanced.getExperimentMonitoring(experimentId),
          experimentServiceEnhanced.getExperimentCostAnalysis(experimentId),
          experimentServiceEnhanced.getExperimentComments(experimentId),
          experimentServiceEnhanced.getExperimentAuditLog(experimentId),
        ])

      setAnalysis(analysisData)
      setMonitoring(monitoringData)
      setCostAnalysis(costData)
      setComments(commentsData)
      setAuditLog(auditData)
    } catch (error) {
      message.error('加载实验数据失败')
    } finally {
      setLoading(false)
    }
  }

  const handleExperimentChange = (experimentId: string) => {
    setSelectedExperiment(experimentId)
    loadExperimentData(experimentId)
  }

  const handleOptimizeConfig = async (values: any) => {
    try {
      const result =
        await experimentServiceEnhanced.optimizeExperimentConfig(values)
      message.success('配置优化完成')
      Modal.info({
        title: '优化建议',
        content: (
          <div>
            <p>推荐样本量: {result.recommended_sample_size}</p>
            <p>推荐持续时间: {result.recommended_duration}天</p>
            <p>流量分配: {JSON.stringify(result.traffic_allocation)}</p>
          </div>
        ),
      })
      setOptimizationModalVisible(false)
    } catch (error) {
      message.error('优化配置失败')
    }
  }

  const handleGetRecommendations = async () => {
    try {
      const result =
        await experimentServiceEnhanced.getExperimentRecommendations({
          context: 'current_analysis',
          current_experiments: [selectedExperiment],
        })
      setRecommendations(result)
      setRecommendationModalVisible(true)
    } catch (error) {
      message.error('获取推荐失败')
    }
  }

  const handleExportData = async (format: 'csv' | 'json' | 'xlsx') => {
    try {
      const result = await experimentServiceEnhanced.exportExperimentData(
        selectedExperiment,
        {
          format,
          include_raw_data: true,
        }
      )
      message.success(
        `数据导出成功，文件大小: ${(result.file_size / 1024 / 1024).toFixed(2)}MB`
      )
      // 模拟下载
      window.open(result.download_url)
    } catch (error) {
      message.error('数据导出失败')
    }
  }

  const variantColumns = [
    {
      title: '变量',
      dataIndex: 'variant_id',
      key: 'variant_id',
      render: (text: string) => <Tag color="blue">{text}</Tag>,
    },
    {
      title: '样本量',
      dataIndex: 'sample_size',
      key: 'sample_size',
      render: (value: number) => value.toLocaleString(),
    },
    {
      title: '转化率',
      dataIndex: 'conversion_rate',
      key: 'conversion_rate',
      render: (value: number) => (
        <Text strong>{(value * 100).toFixed(2)}%</Text>
      ),
    },
    {
      title: '相对提升',
      dataIndex: 'relative_improvement',
      key: 'relative_improvement',
      render: (value: number) => (
        <Text type={value > 0 ? 'success' : value < 0 ? 'danger' : 'secondary'}>
          {value > 0 ? '+' : ''}
          {(value * 100).toFixed(2)}%
        </Text>
      ),
    },
    {
      title: '统计显著性',
      dataIndex: 'statistical_significance',
      key: 'statistical_significance',
      render: (value: boolean) => (
        <Badge
          status={value ? 'success' : 'default'}
          text={value ? '显著' : '不显著'}
        />
      ),
    },
  ]

  return (
    <div style={{ padding: '24px' }}>
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card>
            <Row align="middle" justify="space-between">
              <Col span={12}>
                <Space>
                  <ExperimentOutlined
                    style={{ fontSize: '24px', color: '#1890ff' }}
                  />
                  <Title level={3} style={{ margin: 0 }}>
                    增强版实验分析平台
                  </Title>
                </Space>
              </Col>
              <Col span={12} style={{ textAlign: 'right' }}>
                <Space>
                  <Select
                    value={selectedExperiment}
                    onChange={handleExperimentChange}
                    style={{ width: 200 }}
                    placeholder="选择实验"
                  >
                    {experiments.map(exp => (
                      <Select.Option key={exp.id} value={exp.id}>
                        {exp.name}
                      </Select.Option>
                    ))}
                  </Select>
                  <Button
                    type="primary"
                    icon={<RobotOutlined />}
                    onClick={handleGetRecommendations}
                  >
                    智能推荐
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="实时样本量"
              value={
                monitoring?.real_time_metrics?.find(
                  (m: any) => m.metric === 'sample_size'
                )?.current || 0
              }
              prefix={<TeamOutlined />}
              suffix={`/ ${monitoring?.real_time_metrics?.find((m: any) => m.metric === 'sample_size')?.target || 0}`}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="当前转化率"
              value={
                monitoring?.real_time_metrics?.find(
                  (m: any) => m.metric === 'conversion_rate'
                )?.current * 100 || 0
              }
              precision={2}
              suffix="%"
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="统计功效"
              value={
                monitoring?.real_time_metrics?.find(
                  (m: any) => m.metric === 'statistical_power'
                )?.current * 100 || 0
              }
              precision={1}
              suffix="%"
              prefix={<BarChartOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="预计成本"
              value={costAnalysis?.projected_total_cost || 0}
              prefix={<DollarOutlined />}
              suffix="元"
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card>
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              <TabPane
                tab={
                  <span>
                    <BarChartOutlined />
                    深度分析
                  </span>
                }
                key="analysis"
              >
                <Row gutter={[16, 16]}>
                  <Col span={16}>
                    <Card title="变量性能对比" size="small">
                      <Table
                        dataSource={analysis?.variant_performance || []}
                        columns={variantColumns}
                        pagination={false}
                        size="small"
                      />
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card title="功效分析" size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>
                          <Text>当前功效</Text>
                          <Progress
                            percent={Math.round(
                              (analysis?.power_analysis.current_power || 0) *
                                100
                            )}
                            size="small"
                            status={
                              analysis?.power_analysis.current_power &&
                              analysis.power_analysis.current_power > 0.8
                                ? 'success'
                                : 'active'
                            }
                          />
                        </div>
                        <div>
                          <Text>
                            所需样本量:{' '}
                            {analysis?.power_analysis.required_sample_size?.toLocaleString()}
                          </Text>
                        </div>
                        <div>
                          <Text>
                            预计运行时间:{' '}
                            {analysis?.power_analysis.projected_runtime}天
                          </Text>
                        </div>
                      </Space>
                    </Card>
                  </Col>
                </Row>

                <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
                  <Col span={12}>
                    <Card title="智能建议" size="small">
                      <List
                        dataSource={analysis?.recommendations || []}
                        renderItem={(item: string) => (
                          <List.Item>
                            <BulbOutlined
                              style={{ color: '#faad14', marginRight: 8 }}
                            />
                            {item}
                          </List.Item>
                        )}
                      />
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="风险评估" size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div>
                          <Text>收入风险: </Text>
                          <Tag color="orange">
                            ¥
                            {analysis?.risk_assessment.revenue_risk?.toLocaleString()}
                          </Tag>
                        </div>
                        <div>
                          <Text>用户体验风险: </Text>
                          <Tag
                            color={
                              analysis?.risk_assessment.user_experience_risk ===
                              'high'
                                ? 'red'
                                : 'green'
                            }
                          >
                            {analysis?.risk_assessment.user_experience_risk}
                          </Tag>
                        </div>
                        <div>
                          <Text>技术风险: </Text>
                          <Tag
                            color={
                              analysis?.risk_assessment.technical_risk ===
                              'high'
                                ? 'red'
                                : 'green'
                            }
                          >
                            {analysis?.risk_assessment.technical_risk}
                          </Tag>
                        </div>
                      </Space>
                    </Card>
                  </Col>
                </Row>
              </TabPane>

              <TabPane
                tab={
                  <span>
                    <LineChartOutlined />
                    实时监控
                  </span>
                }
                key="monitoring"
              >
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card title="流量分布" size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        {Object.entries(
                          monitoring?.traffic_distribution || {}
                        ).map(([variant, percentage]) => (
                          <div key={variant}>
                            <Text>{variant}: </Text>
                            <Progress
                              percent={percentage as number}
                              size="small"
                            />
                          </div>
                        ))}
                      </Space>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="告警信息" size="small">
                      <List
                        dataSource={monitoring?.alerts || []}
                        renderItem={(alert: any) => (
                          <List.Item>
                            <Alert
                              message={alert.message}
                              type={
                                alert.severity === 'critical'
                                  ? 'error'
                                  : 'warning'
                              }
                              size="small"
                              showIcon
                            />
                          </List.Item>
                        )}
                      />
                    </Card>
                  </Col>
                </Row>
              </TabPane>

              <TabPane
                tab={
                  <span>
                    <FunnelPlotOutlined />
                    漏斗分析
                  </span>
                }
                key="funnel"
              >
                <Card title="转化漏斗分析">
                  <Button
                    type="primary"
                    onClick={() =>
                      experimentServiceEnhanced.getFunnelAnalysis(
                        selectedExperiment
                      )
                    }
                  >
                    生成漏斗分析
                  </Button>
                </Card>
              </TabPane>

              <TabPane
                tab={
                  <span>
                    <UsergroupAddOutlined />
                    分段分析
                  </span>
                }
                key="segments"
              >
                <Card title="用户分段分析">
                  <Space>
                    <Button
                      type="primary"
                      onClick={() => setSegmentModalVisible(true)}
                    >
                      配置分段
                    </Button>
                    <Button
                      onClick={() =>
                        experimentServiceEnhanced.getSegmentAnalysis(
                          selectedExperiment,
                          ['新用户', '老用户']
                        )
                      }
                    >
                      分析用户群体
                    </Button>
                  </Space>
                </Card>
              </TabPane>

              <TabPane
                tab={
                  <span>
                    <CommentOutlined />
                    协作讨论
                  </span>
                }
                key="collaboration"
              >
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <Card title="团队评论" size="small">
                      <List
                        dataSource={comments}
                        renderItem={(comment: any) => (
                          <List.Item>
                            <List.Item.Meta
                              avatar={<Avatar>{comment.user[0]}</Avatar>}
                              title={comment.user}
                              description={comment.comment}
                            />
                          </List.Item>
                        )}
                      />
                      <Button
                        type="dashed"
                        style={{ width: '100%', marginTop: 16 }}
                      >
                        添加评论
                      </Button>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="操作日志" size="small">
                      <Timeline size="small">
                        {auditLog.map((log: any, index: number) => (
                          <Timeline.Item key={index}>
                            <Text type="secondary">
                              {new Date(log.timestamp).toLocaleString()}
                            </Text>
                            <br />
                            <Text>
                              {log.user} {log.action}
                            </Text>
                            {log.reason && (
                              <Text type="secondary"> - {log.reason}</Text>
                            )}
                          </Timeline.Item>
                        ))}
                      </Timeline>
                    </Card>
                  </Col>
                </Row>
              </TabPane>

              <TabPane
                tab={
                  <span>
                    <DollarOutlined />
                    成本分析
                  </span>
                }
                key="cost"
              >
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Card title="成本统计" size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Statistic
                          title="设置成本"
                          value={costAnalysis?.setup_cost}
                          suffix="元"
                        />
                        <Statistic
                          title="每日运行成本"
                          value={costAnalysis?.running_cost_per_day}
                          suffix="元"
                        />
                        <Statistic
                          title="总成本"
                          value={costAnalysis?.total_cost_to_date}
                          suffix="元"
                        />
                      </Space>
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card title="ROI分析" size="small">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Statistic
                          title="预期ROI"
                          value={costAnalysis?.roi_analysis.projected_roi}
                          precision={2}
                          suffix="x"
                          valueStyle={{ color: '#3f8600' }}
                        />
                        <Text type="secondary">
                          盈亏平衡点:{' '}
                          {costAnalysis?.roi_analysis.break_even_point
                            ? new Date(
                                costAnalysis.roi_analysis.break_even_point
                              ).toLocaleDateString()
                            : '-'}
                        </Text>
                      </Space>
                    </Card>
                  </Col>
                  <Col span={8}>
                    <Card title="效率指标" size="small">
                      <Statistic
                        title="单次转化成本"
                        value={costAnalysis?.cost_per_conversion}
                        precision={2}
                        suffix="元"
                      />
                    </Card>
                  </Col>
                </Row>
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>

      {/* 功能按钮区 */}
      <Card style={{ marginTop: 16 }}>
        <Title level={4}>高级功能</Title>
        <Space wrap>
          <Button
            icon={<SettingOutlined />}
            onClick={() => setTemplateModalVisible(true)}
          >
            实验模板
          </Button>
          <Button
            icon={<ThunderboltOutlined />}
            onClick={() => setOptimizationModalVisible(true)}
          >
            配置优化
          </Button>
          <Button
            icon={<CompareOutlined />}
            onClick={() => setCompareModalVisible(true)}
          >
            实验对比
          </Button>
          <Button
            icon={<RobotOutlined />}
            onClick={() => setAutomationModalVisible(true)}
          >
            自动化规则
          </Button>
          <Button
            icon={<CameraOutlined />}
            onClick={() =>
              experimentServiceEnhanced.createExperimentSnapshot(
                selectedExperiment
              )
            }
          >
            创建快照
          </Button>
          <Button
            icon={<LockOutlined />}
            onClick={() =>
              experimentServiceEnhanced.getExperimentPermissions(
                selectedExperiment
              )
            }
          >
            权限管理
          </Button>
          <Button.Group>
            <Button
              icon={<DownloadOutlined />}
              onClick={() => handleExportData('csv')}
            >
              CSV
            </Button>
            <Button
              icon={<DownloadOutlined />}
              onClick={() => handleExportData('json')}
            >
              JSON
            </Button>
            <Button
              icon={<DownloadOutlined />}
              onClick={() => handleExportData('xlsx')}
            >
              Excel
            </Button>
          </Button.Group>
        </Space>
      </Card>

      {/* 模态框组件 */}
      <Modal
        title="配置优化"
        open={optimizationModalVisible}
        onCancel={() => setOptimizationModalVisible(false)}
        footer={null}
      >
        <Form onFinish={handleOptimizeConfig}>
          <Form.Item
            name="goal_metric"
            label="目标指标"
            initialValue="conversion_rate"
          >
            <Select>
              <Select.Option value="conversion_rate">转化率</Select.Option>
              <Select.Option value="revenue">收入</Select.Option>
              <Select.Option value="retention">留存率</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item name="baseline_value" label="基线值" initialValue={0.05}>
            <InputNumber min={0} max={1} step={0.01} />
          </Form.Item>
          <Form.Item
            name="minimum_effect_size"
            label="最小效应量"
            initialValue={0.02}
          >
            <InputNumber min={0} max={1} step={0.01} />
          </Form.Item>
          <Form.Item
            name="statistical_power"
            label="统计功效"
            initialValue={0.8}
          >
            <InputNumber min={0.5} max={0.99} step={0.01} />
          </Form.Item>
          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                优化配置
              </Button>
              <Button onClick={() => setOptimizationModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      <Modal
        title="智能推荐"
        open={recommendationModalVisible}
        onCancel={() => setRecommendationModalVisible(false)}
        footer={null}
        width={800}
      >
        <div>
          <Title level={5}>推荐实验</Title>
          <List
            dataSource={recommendations?.recommended_experiments || []}
            renderItem={(item: any) => (
              <List.Item>
                <List.Item.Meta
                  title={item.title}
                  description={
                    <div>
                      <Paragraph>{item.description}</Paragraph>
                      <Space>
                        <Tag>
                          预期影响: {(item.expected_impact * 100).toFixed(1)}%
                        </Tag>
                        <Tag
                          color={
                            item.effort_level === 'high'
                              ? 'red'
                              : item.effort_level === 'medium'
                                ? 'orange'
                                : 'green'
                          }
                        >
                          工作量: {item.effort_level}
                        </Tag>
                        <Tag>
                          优先级: {(item.priority_score * 100).toFixed(0)}
                        </Tag>
                      </Space>
                      <Paragraph type="secondary" style={{ marginTop: 8 }}>
                        {item.rationale}
                      </Paragraph>
                    </div>
                  }
                />
              </List.Item>
            )}
          />

          <Divider />

          <Title level={5}>优化建议</Title>
          <List
            dataSource={recommendations?.optimization_suggestions || []}
            renderItem={(suggestion: string) => (
              <List.Item>
                <BulbOutlined style={{ color: '#faad14', marginRight: 8 }} />
                {suggestion}
              </List.Item>
            )}
          />
        </div>
      </Modal>
    </div>
  )
}

export default EnhancedExperimentAnalysisPage
