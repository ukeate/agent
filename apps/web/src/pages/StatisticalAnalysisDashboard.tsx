import React, { useState, useEffect } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Table,
  Button,
  Space,
  Tag,
  Modal,
  Form,
  Input,
  Select,
  InputNumber,
  Switch,
  message,
  Tabs,
  Alert,
  Progress,
  Statistic,
  Row,
  Col,
  Badge,
  Tooltip,
  Divider,
  Upload,
  Radio,
  Slider,
  Empty,
  Spin,
  Descriptions,
  Result,
  Typography,
} from 'antd'
import {
  BarChartOutlined,
  LineChartOutlined,
  CalculatorOutlined,
  ExperimentOutlined,
  ThunderboltOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  InfoCircleOutlined,
  UploadOutlined,
  DownloadOutlined,
  SyncOutlined,
  FundOutlined,
  RiseOutlined,
  FallOutlined,
  QuestionCircleOutlined,
  BulbOutlined,
  WarningOutlined,
  FileTextOutlined,
  DotChartOutlined,
  AreaChartOutlined,
  PieChartOutlined,
} from '@ant-design/icons'
import {
  statisticalAnalysisService,
  DescriptiveStats,
  HypothesisTestResult,
  PowerAnalysisResult,
  TestType,
  SampleSizeCalculation,
  MinimumDetectableEffect,
} from '../services/statisticalAnalysisService'
import { analyticsService } from '../services/analyticsService'
import { Line, Column, Scatter, Area, Box } from '@ant-design/plots'

const { TabPane } = Tabs
const { Option } = Select
const { TextArea } = Input
const { Dragger } = Upload
const { Text } = Typography

const StatisticalAnalysisDashboard: React.FC = () => {
  // 状态管理
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('descriptive')

  // 数据状态
  const [controlData, setControlData] = useState<number[]>([])
  const [treatmentData, setTreatmentData] = useState<number[]>([])
  const [descriptiveStats, setDescriptiveStats] = useState<{
    control?: DescriptiveStats
    treatment?: DescriptiveStats
  }>({})
  const [testResult, setTestResult] = useState<HypothesisTestResult | null>(
    null
  )
  const [powerResult, setPowerResult] = useState<PowerAnalysisResult | null>(
    null
  )
  const [sampleSizeResult, setSampleSizeResult] =
    useState<SampleSizeCalculation | null>(null)
  const [mdeResult, setMdeResult] = useState<MinimumDetectableEffect | null>(
    null
  )
  const [statisticalSummary, setStatisticalSummary] = useState<any>(null)

  // 弹窗状态
  const [dataImportModalVisible, setDataImportModalVisible] = useState(false)
  const [hypothesisTestModalVisible, setHypothesisTestModalVisible] =
    useState(false)
  const [powerAnalysisModalVisible, setPowerAnalysisModalVisible] =
    useState(false)
  const [sampleSizeModalVisible, setSampleSizeModalVisible] = useState(false)
  const [mdeModalVisible, setMdeModalVisible] = useState(false)

  // 表单
  const [dataImportForm] = Form.useForm()
  const [hypothesisTestForm] = Form.useForm()
  const [powerAnalysisForm] = Form.useForm()
  const [sampleSizeForm] = Form.useForm()
  const [mdeForm] = Form.useForm()

  // 初始化加载
  useEffect(() => {
    loadStatisticalSummary()
  }, [])

  // 清空数据
  const resetData = () => {
    setControlData([])
    setTreatmentData([])
    setDescriptiveStats({})
    setTestResult(null)
    setPowerResult(null)
    setSampleSizeResult(null)
    setMdeResult(null)
  }

  // 加载统计摘要
  const loadStatisticalSummary = async () => {
    try {
      setLoading(true)
      const summary = await analyticsService.getStatisticalSummary()
      setStatisticalSummary(summary)
    } catch (error) {
      logger.error('加载统计摘要失败:', error)
      message.error('加载统计摘要失败')
    } finally {
      setLoading(false)
    }
  }

  // 计算描述性统计
  const calculateDescriptiveStats = async (
    control: number[],
    treatment: number[]
  ) => {
    try {
      setLoading(true)
      const [controlStats, treatmentStats] = await Promise.all([
        statisticalAnalysisService.getDescriptiveStats(control),
        statisticalAnalysisService.getDescriptiveStats(treatment),
      ])
      setDescriptiveStats({
        control: controlStats,
        treatment: treatmentStats,
      })
    } catch (error) {
      message.error('计算描述性统计失败')
    } finally {
      setLoading(false)
    }
  }

  // 执行假设检验
  const handleHypothesisTest = async (values: any) => {
    try {
      setLoading(true)
      const result = await statisticalAnalysisService.performHypothesisTest(
        controlData,
        treatmentData,
        values.test_type,
        values.confidence_level
      )
      setTestResult(result)
      message.success('假设检验完成')
      setHypothesisTestModalVisible(false)
      hypothesisTestForm.resetFields()
    } catch (error) {
      message.error('假设检验失败')
    } finally {
      setLoading(false)
    }
  }

  // 执行功效分析
  const handlePowerAnalysis = async (values: any) => {
    try {
      setLoading(true)
      const result = await statisticalAnalysisService.calculatePower(
        values.sample_size,
        values.effect_size,
        values.alpha,
        values.test_type
      )
      setPowerResult(result)
      message.success('功效分析完成')
      setPowerAnalysisModalVisible(false)
      powerAnalysisForm.resetFields()
    } catch (error) {
      message.error('功效分析失败')
    } finally {
      setLoading(false)
    }
  }

  // 计算样本量
  const handleSampleSizeCalculation = async (values: any) => {
    try {
      setLoading(true)
      const result = await statisticalAnalysisService.calculateSampleSize(
        values.effect_size,
        values.power,
        values.alpha,
        values.test_type
      )
      setSampleSizeResult(result)
      message.success('样本量计算完成')
      setSampleSizeModalVisible(false)
      sampleSizeForm.resetFields()
    } catch (error) {
      message.error('样本量计算失败')
    } finally {
      setLoading(false)
    }
  }

  // 计算MDE
  const handleMDECalculation = async (values: any) => {
    try {
      setLoading(true)
      const result = await statisticalAnalysisService.calculateMDE(
        values.sample_size,
        values.baseline_rate,
        values.power,
        values.alpha
      )
      setMdeResult(result)
      message.success('MDE计算完成')
      setMdeModalVisible(false)
      mdeForm.resetFields()
    } catch (error) {
      message.error('MDE计算失败')
    } finally {
      setLoading(false)
    }
  }

  // 导入数据
  const handleDataImport = (values: any) => {
    try {
      const parseValues = (input: string) => {
        const items = input
          .split(',')
          .map((value: string) => value.trim())
          .filter(Boolean)
        const numbers = items.map(value => Number(value))
        if (!numbers.length || numbers.some(value => Number.isNaN(value))) {
          throw new Error('数据格式错误')
        }
        return numbers
      }
      const control = parseValues(values.control_data)
      const treatment = parseValues(values.treatment_data)
      if (control.length < 2 || treatment.length < 2) {
        throw new Error('每组至少需要2个数据点')
      }
      setControlData(control)
      setTreatmentData(treatment)
      setTestResult(null)
      calculateDescriptiveStats(control, treatment)
      message.success('数据导入成功')
      setDataImportModalVisible(false)
      dataImportForm.resetFields()
    } catch (error) {
      message.error('数据格式错误')
    }
  }

  // 获取测试类型名称
  const getTestTypeName = (type: string) => {
    if (type.includes('paired')) return '配对T检验'
    if (type.includes('one_sample')) return '单样本T检验'
    if (type.includes('welch')) return 'Welch T检验'
    if (type.includes('two_sample') || type.includes('independent')) {
      return '双样本T检验'
    }
    if (type.includes('t_test')) return 'T检验'
    return type
  }

  // 箱线图配置
  const boxPlotConfig = {
    data: [
      ...controlData.map(v => ({ group: '对照组', value: v })),
      ...treatmentData.map(v => ({ group: '实验组', value: v })),
    ],
    xField: 'group',
    yField: 'value',
    boxStyle: {
      stroke: '#1890ff',
      fill: '#e6f7ff',
    },
    animation: {
      appear: {
        animation: 'scale-in-y',
        duration: 500,
      },
    },
  }

  // 分布直方图配置
  const histogramConfig = {
    data: [
      ...controlData.map(v => ({ group: '对照组', value: v })),
      ...treatmentData.map(v => ({ group: '实验组', value: v })),
    ],
    binField: 'value',
    stackField: 'group',
    binWidth: 5,
    color: ['#1890ff', '#52c41a'],
    columnStyle: {
      opacity: 0.6,
    },
  }

  return (
    <div style={{ padding: 24 }}>
      <Card
        title={
          <Space>
            <BarChartOutlined />
            <span>统计分析仪表板</span>
          </Space>
        }
        extra={
          <Space>
            <Button
              icon={<UploadOutlined />}
              onClick={() => setDataImportModalVisible(true)}
            >
              导入数据
            </Button>
            <Button
              type="primary"
              icon={<ExperimentOutlined />}
              onClick={() => setHypothesisTestModalVisible(true)}
              disabled={!controlData.length || !treatmentData.length}
            >
              假设检验
            </Button>
            <Button
              icon={<ThunderboltOutlined />}
              onClick={() => setPowerAnalysisModalVisible(true)}
            >
              功效分析
            </Button>
            <Button
              icon={<CalculatorOutlined />}
              onClick={() => setSampleSizeModalVisible(true)}
            >
              样本量计算
            </Button>
            <Button
              icon={<FundOutlined />}
              onClick={() => setMdeModalVisible(true)}
            >
              MDE计算
            </Button>
            <Button icon={<SyncOutlined />} onClick={resetData}>
              清空数据
            </Button>
            <Button
              icon={<DotChartOutlined />}
              onClick={loadStatisticalSummary}
            >
              刷新摘要
            </Button>
          </Space>
        }
      >
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="描述性统计" key="descriptive">
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Card title="对照组统计" size="small">
                  {descriptiveStats.control ? (
                    <Descriptions column={2} size="small">
                      <Descriptions.Item label="样本量">
                        {descriptiveStats.control.count}
                      </Descriptions.Item>
                      <Descriptions.Item label="均值">
                        {descriptiveStats.control.mean.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="中位数">
                        {descriptiveStats.control.median.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="标准差">
                        {descriptiveStats.control.std_dev.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="最小值">
                        {descriptiveStats.control.min_value.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="最大值">
                        {descriptiveStats.control.max_value.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="25分位">
                        {descriptiveStats.control.q25.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="75分位">
                        {descriptiveStats.control.q75.toFixed(2)}
                      </Descriptions.Item>
                    </Descriptions>
                  ) : (
                    <Empty description="暂无数据" />
                  )}
                </Card>
              </Col>
              <Col span={12}>
                <Card title="实验组统计" size="small">
                  {descriptiveStats.treatment ? (
                    <Descriptions column={2} size="small">
                      <Descriptions.Item label="样本量">
                        {descriptiveStats.treatment.count}
                      </Descriptions.Item>
                      <Descriptions.Item label="均值">
                        {descriptiveStats.treatment.mean.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="中位数">
                        {descriptiveStats.treatment.median.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="标准差">
                        {descriptiveStats.treatment.std_dev.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="最小值">
                        {descriptiveStats.treatment.min_value.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="最大值">
                        {descriptiveStats.treatment.max_value.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="25分位">
                        {descriptiveStats.treatment.q25.toFixed(2)}
                      </Descriptions.Item>
                      <Descriptions.Item label="75分位">
                        {descriptiveStats.treatment.q75.toFixed(2)}
                      </Descriptions.Item>
                    </Descriptions>
                  ) : (
                    <Empty description="暂无数据" />
                  )}
                </Card>
              </Col>
            </Row>

            {controlData.length > 0 && treatmentData.length > 0 && (
              <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
                <Col span={24}>
                  <Card title="数据分布">
                    <Box {...boxPlotConfig} />
                  </Card>
                </Col>
              </Row>
            )}
          </TabPane>

          <TabPane tab="假设检验" key="hypothesis">
            {testResult ? (
              <Row gutter={[16, 16]}>
                <Col span={8}>
                  <Card>
                    <Statistic
                      title="检验类型"
                      value={getTestTypeName(testResult.test_type)}
                      prefix={<ExperimentOutlined />}
                    />
                  </Card>
                </Col>
                <Col span={8}>
                  <Card>
                    <Statistic
                      title="P值"
                      value={testResult.p_value.toFixed(4)}
                      valueStyle={{
                        color: testResult.p_value < 0.05 ? '#52c41a' : '#000',
                      }}
                      prefix={
                        testResult.p_value < 0.05 ? (
                          <CheckCircleOutlined />
                        ) : (
                          <CloseCircleOutlined />
                        )
                      }
                    />
                  </Card>
                </Col>
                <Col span={8}>
                  <Card>
                    <Statistic
                      title="统计量"
                      value={testResult.statistic.toFixed(4)}
                      prefix={<CalculatorOutlined />}
                    />
                  </Card>
                </Col>
                <Col span={24}>
                  <Card title="检验结果">
                    <Alert
                      message={
                        testResult.is_significant
                          ? '拒绝原假设'
                          : '不拒绝原假设'
                      }
                      description={
                        testResult.is_significant
                          ? `在${((1 - testResult.alpha) * 100).toFixed(0)}%置信水平下，存在显著差异`
                          : `在${((1 - testResult.alpha) * 100).toFixed(0)}%置信水平下，不存在显著差异`
                      }
                      type={testResult.is_significant ? 'success' : 'info'}
                      showIcon
                    />
                    <Divider />
                    <Descriptions column={2}>
                      <Descriptions.Item label="对照组样本量">
                        {testResult.sample_sizes?.control}
                      </Descriptions.Item>
                      <Descriptions.Item label="实验组样本量">
                        {testResult.sample_sizes?.treatment}
                      </Descriptions.Item>
                      {testResult.means && (
                        <>
                          <Descriptions.Item label="对照组均值">
                            {testResult.means.control.toFixed(2)}
                          </Descriptions.Item>
                          <Descriptions.Item label="实验组均值">
                            {testResult.means.treatment.toFixed(2)}
                          </Descriptions.Item>
                        </>
                      )}
                      {testResult.effect_size && (
                        <Descriptions.Item label="效应量">
                          {testResult.effect_size.toFixed(3)}
                        </Descriptions.Item>
                      )}
                      {testResult.confidence_interval && (
                        <Descriptions.Item label="置信区间">
                          [{testResult.confidence_interval[0].toFixed(3)},{' '}
                          {testResult.confidence_interval[1].toFixed(3)}]
                        </Descriptions.Item>
                      )}
                    </Descriptions>
                  </Card>
                </Col>
              </Row>
            ) : (
              <Empty description="请先执行假设检验" />
            )}
          </TabPane>

          <TabPane tab="功效分析" key="power">
            <Row gutter={[16, 16]}>
              {powerResult && (
                <Col span={12}>
                  <Card title="功效分析结果">
                    <Statistic
                      title="统计功效"
                      value={(powerResult.power * 100).toFixed(1)}
                      suffix="%"
                      valueStyle={{
                        color: powerResult.power >= 0.8 ? '#52c41a' : '#faad14',
                      }}
                    />
                    <Divider />
                    <Descriptions column={1}>
                      <Descriptions.Item label="显著性水平">
                        {powerResult.alpha}
                      </Descriptions.Item>
                      <Descriptions.Item label="检验类型">
                        {getTestTypeName(powerResult.test_type)}
                      </Descriptions.Item>
                      <Descriptions.Item label="备择假设">
                        {powerResult.alternative}
                      </Descriptions.Item>
                    </Descriptions>
                    {powerResult.recommendations && (
                      <Alert
                        message="建议"
                        description={
                          <ul style={{ margin: 0, paddingLeft: 20 }}>
                            {powerResult.recommendations.map((rec, idx) => (
                              <li key={idx}>{rec}</li>
                            ))}
                          </ul>
                        }
                        type="info"
                        style={{ marginTop: 16 }}
                      />
                    )}
                  </Card>
                </Col>
              )}

              {sampleSizeResult && (
                <Col span={12}>
                  <Card title="样本量计算结果">
                    <Statistic
                      title="所需总样本量"
                      value={sampleSizeResult.total_sample_size}
                      prefix={<TeamOutlined />}
                    />
                    <Divider />
                    <Descriptions column={1}>
                      <Descriptions.Item label="样本量分配">
                        {Array.isArray(sampleSizeResult.sample_size)
                          ? `${sampleSizeResult.sample_size[0]} / ${sampleSizeResult.sample_size[1]}`
                          : sampleSizeResult.sample_size}
                      </Descriptions.Item>
                      <Descriptions.Item label="效应量">
                        {sampleSizeResult.effect_size.toFixed(3)}
                      </Descriptions.Item>
                      <Descriptions.Item label="统计功效">
                        {(sampleSizeResult.power * 100).toFixed(0)}%
                      </Descriptions.Item>
                      <Descriptions.Item label="显著性水平">
                        {sampleSizeResult.alpha}
                      </Descriptions.Item>
                    </Descriptions>
                  </Card>
                </Col>
              )}

              {mdeResult && (
                <Col span={12}>
                  <Card title="MDE计算结果">
                    <Statistic
                      title="最小可检测效应"
                      value={(mdeResult.relative_change * 100).toFixed(1)}
                      suffix="%"
                      prefix={<FundOutlined />}
                    />
                    <Divider />
                    <Descriptions column={1}>
                      <Descriptions.Item label="基准率">
                        {(mdeResult.baseline_rate * 100).toFixed(1)}%
                      </Descriptions.Item>
                      <Descriptions.Item label="绝对变化">
                        {(mdeResult.absolute_change * 100).toFixed(2)}%
                      </Descriptions.Item>
                      <Descriptions.Item label="MDE值">
                        {mdeResult.mde.toFixed(4)}
                      </Descriptions.Item>
                    </Descriptions>
                  </Card>
                </Col>
              )}

              {!powerResult && !sampleSizeResult && !mdeResult && (
                <Col span={24}>
                  <Empty description="请执行功效分析、样本量计算或MDE计算" />
                </Col>
              )}
            </Row>
          </TabPane>

          <TabPane tab="系统摘要" key="summary">
            {statisticalSummary ? (
              <Row gutter={[16, 16]}>
                {/* 总体统计 */}
                <Col span={24}>
                  <Card title="系统总体统计" size="small">
                    <Row gutter={[16, 16]}>
                      <Col span={6}>
                        <Statistic
                          title="数据集分析总数"
                          value={statisticalSummary.datasets_analyzed}
                          prefix={<FileTextOutlined />}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="统计测试执行次数"
                          value={statisticalSummary.statistical_tests_performed}
                          prefix={<CalculatorOutlined />}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="今日测试数量"
                          value={
                            statisticalSummary.recent_activity
                              ?.tests_last_24h || 0
                          }
                          prefix={<RiseOutlined />}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="活跃会话数"
                          value={
                            statisticalSummary.recent_activity
                              ?.active_sessions || 0
                          }
                          prefix={<ThunderboltOutlined />}
                        />
                      </Col>
                    </Row>
                  </Card>
                </Col>

                {/* 假设检验统计 */}
                <Col span={12}>
                  <Card title="假设检验分析" size="small">
                    <Row gutter={[8, 8]}>
                      <Col span={12}>
                        <Statistic
                          title="检验总数"
                          value={
                            statisticalSummary.hypothesis_tests?.total || 0
                          }
                          prefix={<ExperimentOutlined />}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="显著性水平"
                          value={
                            statisticalSummary.hypothesis_tests
                              ?.significance_level || 0
                          }
                          prefix={<InfoCircleOutlined />}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="拒绝原假设"
                          value={
                            statisticalSummary.hypothesis_tests
                              ?.rejected_null || 0
                          }
                          valueStyle={{ color: '#52c41a' }}
                          prefix={<CheckCircleOutlined />}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="接受原假设"
                          value={
                            statisticalSummary.hypothesis_tests
                              ?.accepted_null || 0
                          }
                          valueStyle={{ color: '#faad14' }}
                          prefix={<CloseCircleOutlined />}
                        />
                      </Col>
                    </Row>
                    {statisticalSummary.hypothesis_tests?.power_analysis && (
                      <>
                        <Divider />
                        <Descriptions title="功效分析" size="small" column={1}>
                          <Descriptions.Item label="平均功效">
                            {(
                              statisticalSummary.hypothesis_tests.power_analysis
                                .average_power * 100
                            ).toFixed(1)}
                            %
                          </Descriptions.Item>
                          <Descriptions.Item label="功效充足的测试">
                            {
                              statisticalSummary.hypothesis_tests.power_analysis
                                .tests_with_adequate_power
                            }
                          </Descriptions.Item>
                          <Descriptions.Item label="功效不足的测试">
                            {
                              statisticalSummary.hypothesis_tests.power_analysis
                                .underpowered_tests
                            }
                          </Descriptions.Item>
                        </Descriptions>
                      </>
                    )}
                  </Card>
                </Col>

                {/* 回归分析统计 */}
                <Col span={12}>
                  <Card title="回归分析" size="small">
                    <Row gutter={[8, 8]}>
                      <Col span={12}>
                        <Statistic
                          title="模型总数"
                          value={
                            statisticalSummary.regression_analysis
                              ?.models_fitted || 0
                          }
                          prefix={<LineChartOutlined />}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="预测准确率"
                          value={(
                            statisticalSummary.regression_analysis
                              ?.prediction_accuracy * 100
                          ).toFixed(1)}
                          suffix="%"
                          prefix={<FundOutlined />}
                        />
                      </Col>
                    </Row>
                    {statisticalSummary.regression_analysis
                      ?.r_squared_distribution && (
                      <>
                        <Divider />
                        <Descriptions title="R² 分布" size="small" column={2}>
                          <Descriptions.Item label="优秀 (>0.8)">
                            <Badge
                              count={
                                statisticalSummary.regression_analysis
                                  .r_squared_distribution.excellent
                              }
                              style={{ backgroundColor: '#52c41a' }}
                            />
                          </Descriptions.Item>
                          <Descriptions.Item label="良好 (0.6-0.8)">
                            <Badge
                              count={
                                statisticalSummary.regression_analysis
                                  .r_squared_distribution.good
                              }
                              style={{ backgroundColor: '#1890ff' }}
                            />
                          </Descriptions.Item>
                          <Descriptions.Item label="中等 (0.4-0.6)">
                            <Badge
                              count={
                                statisticalSummary.regression_analysis
                                  .r_squared_distribution.moderate
                              }
                              style={{ backgroundColor: '#faad14' }}
                            />
                          </Descriptions.Item>
                          <Descriptions.Item label="较差 (<0.4)">
                            <Badge
                              count={
                                statisticalSummary.regression_analysis
                                  .r_squared_distribution.poor
                              }
                              style={{ backgroundColor: '#f5222d' }}
                            />
                          </Descriptions.Item>
                        </Descriptions>
                      </>
                    )}
                  </Card>
                </Col>

                {/* 相关性分析 */}
                <Col span={12}>
                  <Card title="相关性分析" size="small">
                    <Row gutter={[8, 8]}>
                      <Col span={12}>
                        <Statistic
                          title="相关性计算总数"
                          value={
                            statisticalSummary.correlation_analysis
                              ?.correlations_computed || 0
                          }
                          prefix={<AreaChartOutlined />}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="显著相关数量"
                          value={
                            statisticalSummary.correlation_analysis
                              ?.significant_correlations || 0
                          }
                          valueStyle={{ color: '#52c41a' }}
                          prefix={<CheckCircleOutlined />}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="强相关数量"
                          value={
                            statisticalSummary.correlation_analysis
                              ?.strong_correlations || 0
                          }
                          valueStyle={{ color: '#1890ff' }}
                          prefix={<RiseOutlined />}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="平均相关强度"
                          value={
                            statisticalSummary.correlation_analysis?.average_correlation_strength?.toFixed(
                              3
                            ) || 0
                          }
                          prefix={<FundOutlined />}
                        />
                      </Col>
                    </Row>
                  </Card>
                </Col>

                {/* 异常值检测 */}
                <Col span={12}>
                  <Card title="异常值检测" size="small">
                    <Row gutter={[8, 8]}>
                      <Col span={12}>
                        <Statistic
                          title="检测到的异常值"
                          value={
                            statisticalSummary.outlier_detection
                              ?.outliers_detected || 0
                          }
                          valueStyle={{ color: '#faad14' }}
                          prefix={<WarningOutlined />}
                        />
                      </Col>
                      <Col span={12}>
                        <Statistic
                          title="异常值比例"
                          value={(
                            statisticalSummary.outlier_detection?.outlier_rate *
                            100
                          ).toFixed(2)}
                          suffix="%"
                          prefix={<PieChartOutlined />}
                        />
                      </Col>
                    </Row>
                    {statisticalSummary.outlier_detection?.methods_used && (
                      <>
                        <Divider />
                        <div>
                          <strong>检测方法: </strong>
                          {statisticalSummary.outlier_detection.methods_used.map(
                            (method: string, index: number) => (
                              <Tag key={index} color="blue">
                                {method}
                              </Tag>
                            )
                          )}
                        </div>
                      </>
                    )}
                  </Card>
                </Col>

                {/* 时间序列分析 */}
                {statisticalSummary.time_series_analysis && (
                  <Col span={24}>
                    <Card title="时间序列分析" size="small">
                      <Row gutter={[16, 16]}>
                        <Col span={6}>
                          <Statistic
                            title="序列分析数量"
                            value={
                              statisticalSummary.time_series_analysis
                                .series_analyzed
                            }
                            prefix={<AreaChartOutlined />}
                          />
                        </Col>
                        <Col span={6}>
                          <Statistic
                            title="预测准确率"
                            value={(
                              statisticalSummary.time_series_analysis
                                .forecasting_accuracy * 100
                            ).toFixed(1)}
                            suffix="%"
                            prefix={<FundOutlined />}
                          />
                        </Col>
                        <Col span={12}>
                          <Card title="趋势模式分布" size="small">
                            <Row gutter={[8, 8]}>
                              <Col span={6}>
                                <Statistic
                                  title="上升"
                                  value={
                                    statisticalSummary.time_series_analysis
                                      .trend_patterns?.increasing || 0
                                  }
                                />
                              </Col>
                              <Col span={6}>
                                <Statistic
                                  title="下降"
                                  value={
                                    statisticalSummary.time_series_analysis
                                      .trend_patterns?.decreasing || 0
                                  }
                                />
                              </Col>
                              <Col span={6}>
                                <Statistic
                                  title="稳定"
                                  value={
                                    statisticalSummary.time_series_analysis
                                      .trend_patterns?.stable || 0
                                  }
                                />
                              </Col>
                              <Col span={6}>
                                <Statistic
                                  title="季节性"
                                  value={
                                    statisticalSummary.time_series_analysis
                                      .trend_patterns?.seasonal || 0
                                  }
                                />
                              </Col>
                            </Row>
                          </Card>
                        </Col>
                      </Row>
                    </Card>
                  </Col>
                )}

                {/* 性能指标 */}
                <Col span={24}>
                  <Card title="系统性能指标" size="small">
                    <Row gutter={[16, 16]}>
                      <Col span={6}>
                        <Statistic
                          title="平均计算时间"
                          value={
                            statisticalSummary.performance_metrics
                              ?.average_computation_time_ms || 0
                          }
                          suffix="ms"
                          prefix={<ThunderboltOutlined />}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="缓存命中率"
                          value={(
                            statisticalSummary.performance_metrics
                              ?.cache_hit_rate * 100
                          ).toFixed(1)}
                          suffix="%"
                          valueStyle={{ color: '#52c41a' }}
                          prefix={<CheckCircleOutlined />}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="错误率"
                          value={(
                            statisticalSummary.performance_metrics?.error_rate *
                            100
                          ).toFixed(2)}
                          suffix="%"
                          valueStyle={{
                            color:
                              statisticalSummary.performance_metrics
                                ?.error_rate > 0.01
                                ? '#f5222d'
                                : '#52c41a',
                          }}
                          prefix={<WarningOutlined />}
                        />
                      </Col>
                      <Col span={6}>
                        <Statistic
                          title="今日实验分析"
                          value={
                            statisticalSummary.recent_activity
                              ?.experiments_analyzed_today || 0
                          }
                          prefix={<ExperimentOutlined />}
                        />
                      </Col>
                    </Row>
                  </Card>
                </Col>
              </Row>
            ) : (
              <div style={{ textAlign: 'center', padding: '40px' }}>
                <Spin size="large" />
                <div style={{ marginTop: 16 }}>加载统计摘要中...</div>
              </div>
            )}
          </TabPane>
        </Tabs>
      </Card>

      {/* 数据导入弹窗 */}
      <Modal
        title="导入数据"
        visible={dataImportModalVisible}
        onCancel={() => setDataImportModalVisible(false)}
        onOk={() => dataImportForm.submit()}
        width={600}
      >
        <Form
          form={dataImportForm}
          layout="vertical"
          onFinish={handleDataImport}
        >
          <Alert
            message="数据格式说明"
            description="请输入逗号分隔的数值，例如: 1.2, 3.4, 5.6, 7.8"
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />
          <Form.Item
            name="control_data"
            label="对照组数据"
            rules={[{ required: true, message: '请输入对照组数据' }]}
          >
            <TextArea rows={4} placeholder="输入逗号分隔的数值" />
          </Form.Item>
          <Form.Item
            name="treatment_data"
            label="实验组数据"
            rules={[{ required: true, message: '请输入实验组数据' }]}
          >
            <TextArea rows={4} placeholder="输入逗号分隔的数值" />
          </Form.Item>
        </Form>
      </Modal>

      {/* 假设检验弹窗 */}
      <Modal
        title="假设检验"
        visible={hypothesisTestModalVisible}
        onCancel={() => setHypothesisTestModalVisible(false)}
        onOk={() => hypothesisTestForm.submit()}
        width={500}
      >
        <Form
          form={hypothesisTestForm}
          layout="vertical"
          onFinish={handleHypothesisTest}
          initialValues={{
            test_type: TestType.T_TEST,
            confidence_level: 0.95,
          }}
        >
          <Form.Item
            name="test_type"
            label="检验类型"
            rules={[{ required: true }]}
          >
            <Select>
              <Option value={TestType.T_TEST}>T检验 (连续数据)</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="confidence_level"
            label="置信水平"
            rules={[{ required: true }]}
          >
            <Slider
              min={0.9}
              max={0.99}
              step={0.01}
              marks={{
                0.9: '90%',
                0.95: '95%',
                0.99: '99%',
              }}
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* 功效分析弹窗 */}
      <Modal
        title="功效分析"
        visible={powerAnalysisModalVisible}
        onCancel={() => setPowerAnalysisModalVisible(false)}
        onOk={() => powerAnalysisForm.submit()}
        width={500}
      >
        <Form
          form={powerAnalysisForm}
          layout="vertical"
          onFinish={handlePowerAnalysis}
          initialValues={{
            sample_size: 1000,
            effect_size: 0.2,
            alpha: 0.05,
            test_type: 'two_sample_t',
          }}
        >
          <Form.Item
            name="sample_size"
            label="样本量(单组)"
            rules={[{ required: true }]}
          >
            <InputNumber min={10} max={1000000} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item
            name="effect_size"
            label="效应量"
            rules={[{ required: true }]}
          >
            <InputNumber
              min={0.01}
              max={2}
              step={0.01}
              style={{ width: '100%' }}
            />
          </Form.Item>
          <Form.Item
            name="alpha"
            label="显著性水平 (α)"
            rules={[{ required: true }]}
          >
            <Select>
              <Option value={0.01}>0.01</Option>
              <Option value={0.05}>0.05</Option>
              <Option value={0.1}>0.10</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="test_type"
            label="检验类型"
            rules={[{ required: true }]}
          >
            <Select>
              <Option value="one_sample_t">单样本T检验</Option>
              <Option value="two_sample_t">双样本T检验</Option>
              <Option value="paired_t">配对T检验</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* 样本量计算弹窗 */}
      <Modal
        title="样本量计算"
        visible={sampleSizeModalVisible}
        onCancel={() => setSampleSizeModalVisible(false)}
        onOk={() => sampleSizeForm.submit()}
        width={500}
      >
        <Form
          form={sampleSizeForm}
          layout="vertical"
          onFinish={handleSampleSizeCalculation}
          initialValues={{
            effect_size: 0.2,
            power: 0.8,
            alpha: 0.05,
            test_type: 'two_sample_t',
          }}
        >
          <Form.Item
            name="effect_size"
            label="期望效应量"
            rules={[{ required: true }]}
            extra="小效应: 0.2, 中效应: 0.5, 大效应: 0.8"
          >
            <InputNumber
              min={0.01}
              max={2}
              step={0.01}
              style={{ width: '100%' }}
            />
          </Form.Item>
          <Form.Item name="power" label="统计功效" rules={[{ required: true }]}>
            <Slider
              min={0.5}
              max={0.99}
              step={0.01}
              marks={{
                0.5: '50%',
                0.8: '80%',
                0.99: '99%',
              }}
            />
          </Form.Item>
          <Form.Item
            name="alpha"
            label="显著性水平 (α)"
            rules={[{ required: true }]}
          >
            <Select>
              <Option value={0.01}>0.01</Option>
              <Option value={0.05}>0.05</Option>
              <Option value={0.1}>0.10</Option>
            </Select>
          </Form.Item>
          <Form.Item
            name="test_type"
            label="检验类型"
            rules={[{ required: true }]}
          >
            <Select>
              <Option value="one_sample_t">单样本T检验</Option>
              <Option value="two_sample_t">双样本T检验</Option>
              <Option value="paired_t">配对T检验</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* MDE计算弹窗 */}
      <Modal
        title="MDE计算"
        visible={mdeModalVisible}
        onCancel={() => setMdeModalVisible(false)}
        onOk={() => mdeForm.submit()}
        width={500}
      >
        <Form
          form={mdeForm}
          layout="vertical"
          onFinish={handleMDECalculation}
          initialValues={{
            sample_size: 1000,
            baseline_rate: 0.1,
            power: 0.8,
            alpha: 0.05,
          }}
        >
          <Form.Item
            name="sample_size"
            label="样本量"
            rules={[{ required: true }]}
          >
            <InputNumber min={10} max={1000000} style={{ width: '100%' }} />
          </Form.Item>
          <Form.Item
            name="baseline_rate"
            label="基准转化率"
            rules={[{ required: true }]}
            extra="输入0-1之间的值，如0.1表示10%"
          >
            <InputNumber
              min={0.001}
              max={1}
              step={0.001}
              style={{ width: '100%' }}
            />
          </Form.Item>
          <Form.Item name="power" label="统计功效" rules={[{ required: true }]}>
            <Slider
              min={0.5}
              max={0.99}
              step={0.01}
              marks={{
                0.5: '50%',
                0.8: '80%',
                0.99: '99%',
              }}
            />
          </Form.Item>
          <Form.Item
            name="alpha"
            label="显著性水平 (α)"
            rules={[{ required: true }]}
          >
            <Select>
              <Option value={0.01}>0.01</Option>
              <Option value={0.05}>0.05</Option>
              <Option value={0.1}>0.10</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  )
}

export default StatisticalAnalysisDashboard
