import React, { useState, useEffect } from 'react';
import {
import { logger } from '../utils/logger'
  Card,
  Button,
  Input,
  Select,
  Alert,
  Tabs,
  Row,
  Col,
  Form,
  message,
  InputNumber,
  Typography,
  Space,
  Progress,
  Table,
  Tag,
  Divider,
  Modal,
  Checkbox,
  Collapse,
  Statistic,
  Spin
} from 'antd';
import {
  FunctionOutlined,
  BarChartOutlined,
  ExperimentOutlined,
  LineChartOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  WarningOutlined,
  QuestionCircleOutlined,
  TrophyOutlined,
  ThunderboltOutlined,
  RocketOutlined
} from '@ant-design/icons';
import {
  hypothesisTestingService,
  HypothesisType,
  MetricType,
  HypothesisTestResponse,
  ABTestComparisonResponse,
  OneSampleTTestRequest,
  TwoSampleTTestRequest,
  PairedTTestRequest,
  ChiSquareGoodnessOfFitRequest,
  ChiSquareIndependenceRequest,
  TwoProportionTestRequest,
  ABTestComparisonRequest
} from '../services/hypothesisTestingService';

const { Option } = Select;
const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { Panel } = Collapse;

const HypothesisTestingPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [testResults, setTestResults] = useState<any[]>([]);
  const [selectedTest, setSelectedTest] = useState<any>(null);
  const [detailsModalVisible, setDetailsModalVisible] = useState(false);
  
  // T检验相关状态
  const [oneSampleData, setOneSampleData] = useState<string>('');
  const [populationMean, setPopulationMean] = useState<number>(0);
  const [twoSample1Data, setTwoSample1Data] = useState<string>('');
  const [twoSample2Data, setTwoSample2Data] = useState<string>('');
  const [equalVariances, setEqualVariances] = useState<boolean>(true);
  const [hypothesisType, setHypothesisType] = useState<HypothesisType>(HypothesisType.TWO_SIDED);
  const [alpha, setAlpha] = useState<number>(0.05);

  // 卡方检验相关状态
  const [observed, setObserved] = useState<string>('');
  const [expected, setExpected] = useState<string>('');
  const [contingencyTable, setContingencyTable] = useState<string>('');

  // 比例检验相关状态
  const [successes1, setSuccesses1] = useState<number>(0);
  const [total1, setTotal1] = useState<number>(0);
  const [successes2, setSuccesses2] = useState<number>(0);
  const [total2, setTotal2] = useState<number>(0);

  // A/B测试相关状态
  const [controlConversions, setControlConversions] = useState<number>(0);
  const [controlTotal, setControlTotal] = useState<number>(0);
  const [treatmentConversions, setTreatmentConversions] = useState<number>(0);
  const [treatmentTotal, setTreatmentTotal] = useState<number>(0);
  const [metricType, setMetricType] = useState<MetricType>(MetricType.CONVERSION);

  // 解析数字数组字符串
  const parseNumberArray = (input: string): number[] => {
    return input.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
  };

  // 解析二维数组字符串
  const parse2DNumberArray = (input: string): number[][] => {
    return input.split('\n').map(row => 
      row.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n))
    ).filter(row => row.length > 0);
  };

  // 执行单样本t检验
  const runOneSampleTTest = async () => {
    try {
      setLoading(true);
      const sample = parseNumberArray(oneSampleData);
      if (sample.length < 2) {
        message.error('样本数据至少需要2个数值');
        return;
      }

      const request: OneSampleTTestRequest = {
        sample,
        population_mean: populationMean,
        hypothesis_type: hypothesisType,
        alpha
      };

      const result = await hypothesisTestingService.oneSampleTTest(request);
      setTestResults(prev => [...prev, { id: Date.now(), type: '单样本t检验', result }]);
      
      setOneSampleData('');
      setPopulationMean(0);
      message.success('单样本t检验执行成功');
    } catch (error) {
      logger.error('单样本t检验失败:', error);
      message.error('检验失败，请检查输入数据');
    } finally {
      setLoading(false);
    }
  };

  // 执行双样本t检验
  const runTwoSampleTTest = async () => {
    try {
      setLoading(true);
      const sample1 = parseNumberArray(twoSample1Data);
      const sample2 = parseNumberArray(twoSample2Data);
      
      if (sample1.length < 2 || sample2.length < 2) {
        message.error('两个样本数据都至少需要2个数值');
        return;
      }

      const request: TwoSampleTTestRequest = {
        sample1,
        sample2,
        equal_variances: equalVariances,
        hypothesis_type: hypothesisType,
        alpha
      };

      const result = await hypothesisTestingService.twoSampleTTest(request);
      setTestResults(prev => [...prev, { id: Date.now(), type: '双样本t检验', result }]);
      
      setTwoSample1Data('');
      setTwoSample2Data('');
      message.success('双样本t检验执行成功');
    } catch (error) {
      logger.error('双样本t检验失败:', error);
      message.error('检验失败，请检查输入数据');
    } finally {
      setLoading(false);
    }
  };

  // 执行A/B测试比较
  const runABTestComparison = async () => {
    try {
      setLoading(true);

      if (controlTotal === 0 || treatmentTotal === 0) {
        message.error('样本总数不能为0');
        return;
      }

      const request: ABTestComparisonRequest = {
        control_group: {
          conversions: controlConversions,
          total_users: controlTotal
        },
        treatment_group: {
          conversions: treatmentConversions,
          total_users: treatmentTotal
        },
        metric_type: metricType,
        hypothesis_type: hypothesisType,
        alpha,
        equal_variances: true
      };

      const result = await hypothesisTestingService.abTestComparison(request);
      setTestResults(prev => [...prev, { id: Date.now(), type: 'A/B测试比较', result }]);
      
      setControlConversions(0);
      setControlTotal(0);
      setTreatmentConversions(0);
      setTreatmentTotal(0);
      message.success('A/B测试比较执行成功');
    } catch (error) {
      logger.error('A/B测试比较失败:', error);
      message.error('检验失败，请检查输入数据');
    } finally {
      setLoading(false);
    }
  };

  // 检查服务健康状态
  const checkServiceHealth = async () => {
    try {
      setLoading(true);
      const health = await hypothesisTestingService.healthCheck();
      message.success(`服务状态: ${health.status}\n${health.message}`);
    } catch (error) {
      logger.error('健康检查失败:', error);
      message.error('服务健康检查失败');
    } finally {
      setLoading(false);
    }
  };

  // 查看测试详情
  const viewTestDetails = (test: any) => {
    setSelectedTest(test);
    setDetailsModalVisible(true);
  };

  // 渲染检验结果卡片
  const renderTestResult = (result: any, resultType: string) => {
    if (!result) return null;

    const isSignificant = result.result?.is_significant;
    const tagColor = isSignificant ? 'success' : 'default';
    const icon = isSignificant ? <CheckCircleOutlined /> : <ExclamationCircleOutlined />;

    return (
      <Card 
        key={result.id} 
        style={{ marginBottom: 16 }}
        title={
          <Space>
            <Text strong>{resultType}</Text>
            <Tag color={tagColor} icon={icon}>
              {isSignificant ? '统计显著' : '不显著'}
            </Tag>
          </Space>
        }
        extra={
          <Button 
            type="link" 
            icon={<InfoCircleOutlined />}
            onClick={() => viewTestDetails(result)}
          >
            查看详情
          </Button>
        }
      >
        <Row gutter={[16, 16]}>
          <Col span={6}>
            <Statistic
              title="检验统计量"
              value={result.result?.statistic?.toFixed(4) || 'N/A'}
              precision={4}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="p值"
              value={result.result?.p_value?.toExponential(3) || 'N/A'}
            />
          </Col>
          <Col span={6}>
            <Statistic
              title="显著性水平"
              value={`α = ${result.result?.alpha || 0.05}`}
            />
          </Col>
          {result.result?.effect_size && (
            <Col span={6}>
              <Statistic
                title="效应量"
                value={result.result.effect_size.toFixed(4)}
                precision={4}
              />
            </Col>
          )}
        </Row>

        {result.interpretation && (
          <>
            <Divider />
            <div>
              <Text strong>结果解释：</Text>
              <div style={{ marginTop: 8 }}>
                {Object.entries(result.interpretation).map(([key, value]) => (
                  <Paragraph key={key} style={{ marginBottom: 4 }}>
                    {value as string}
                  </Paragraph>
                ))}
              </div>
            </div>
          </>
        )}

        {result.recommendations && result.recommendations.length > 0 && (
          <>
            <Divider />
            <div>
              <Text strong>建议：</Text>
              <div style={{ marginTop: 8 }}>
                {result.recommendations.map((rec: string, index: number) => (
                  <Alert 
                    key={index} 
                    message={rec} 
                    type="info" 
                    showIcon 
                    style={{ marginBottom: 8 }}
                  />
                ))}
              </div>
            </div>
          </>
        )}
      </Card>
    );
  };

  const tabItems = [
    {
      key: 'ttest',
      label: (
        <span>
          <FunctionOutlined />
          t检验
        </span>
      ),
      children: (
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <Collapse defaultActiveKey={['1']}>
            <Panel header="单样本t检验" key="1">
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <TextArea
                      rows={3}
                      placeholder="例如: 1.2, 2.3, 3.4, 4.5, 5.6"
                      value={oneSampleData}
                      onChange={(e) => setOneSampleData(e.target.value)}
                    />
                    <Text type="secondary">输入至少2个数值，用逗号分隔</Text>
                  </Col>
                  <Col span={6}>
                    <InputNumber
                      style={{ width: '100%' }}
                      placeholder="总体均值 (μ₀)"
                      value={populationMean}
                      onChange={(value) => setPopulationMean(value || 0)}
                    />
                    <Text type="secondary">零假设的总体均值</Text>
                  </Col>
                  <Col span={6}>
                    <Select
                      style={{ width: '100%' }}
                      placeholder="假设类型"
                      value={hypothesisType}
                      onChange={setHypothesisType}
                    >
                      <Option value={HypothesisType.TWO_SIDED}>双侧检验</Option>
                      <Option value={HypothesisType.LESS}>左侧检验</Option>
                      <Option value={HypothesisType.GREATER}>右侧检验</Option>
                    </Select>
                  </Col>
                </Row>
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <InputNumber
                      style={{ width: '100%' }}
                      placeholder="显著性水平 (α)"
                      value={alpha}
                      onChange={(value) => setAlpha(value || 0.05)}
                      min={0.001}
                      max={0.1}
                      step={0.01}
                    />
                    <Text type="secondary">通常为0.05或0.01</Text>
                  </Col>
                  <Col span={12}>
                    <Button
                      type="primary"
                      icon={<PlayCircleOutlined />}
                      onClick={runOneSampleTTest}
                      disabled={loading || !oneSampleData.trim()}
                      style={{ width: '100%', height: 40 }}
                    >
                      执行单样本t检验
                    </Button>
                  </Col>
                </Row>
              </Space>
            </Panel>

            <Panel header="双样本t检验" key="2">
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Row gutter={[16, 16]}>
                  <Col span={12}>
                    <TextArea
                      rows={3}
                      placeholder="例如: 1.2, 2.3, 3.4, 4.5, 5.6"
                      value={twoSample1Data}
                      onChange={(e) => setTwoSample1Data(e.target.value)}
                    />
                    <Text type="secondary">第一组样本数据</Text>
                  </Col>
                  <Col span={12}>
                    <TextArea
                      rows={3}
                      placeholder="例如: 2.1, 3.2, 4.3, 5.4, 6.5"
                      value={twoSample2Data}
                      onChange={(e) => setTwoSample2Data(e.target.value)}
                    />
                    <Text type="secondary">第二组样本数据</Text>
                  </Col>
                </Row>
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Checkbox
                      checked={equalVariances}
                      onChange={(e) => setEqualVariances(e.target.checked)}
                    >
                      假定等方差
                    </Checkbox>
                  </Col>
                  <Col span={8}>
                    <Select
                      style={{ width: '100%' }}
                      placeholder="假设类型"
                      value={hypothesisType}
                      onChange={setHypothesisType}
                    >
                      <Option value={HypothesisType.TWO_SIDED}>双侧检验</Option>
                      <Option value={HypothesisType.LESS}>左侧检验</Option>
                      <Option value={HypothesisType.GREATER}>右侧检验</Option>
                    </Select>
                  </Col>
                  <Col span={8}>
                    <Button
                      type="primary"
                      icon={<PlayCircleOutlined />}
                      onClick={runTwoSampleTTest}
                      disabled={loading || !twoSample1Data.trim() || !twoSample2Data.trim()}
                      style={{ width: '100%', height: 40 }}
                    >
                      执行双样本t检验
                    </Button>
                  </Col>
                </Row>
              </Space>
            </Panel>
          </Collapse>
        </Space>
      )
    },
    {
      key: 'chi-square',
      label: (
        <span>
          <BarChartOutlined />
          卡方检验
        </span>
      ),
      children: (
        <Alert 
          message="卡方检验功能正在开发中" 
          description="包括拟合优度检验和独立性检验" 
          type="info" 
          showIcon 
        />
      )
    },
    {
      key: 'proportion',
      label: (
        <span>
          <ExperimentOutlined />
          比例检验
        </span>
      ),
      children: (
        <Alert 
          message="两比例检验功能正在开发中" 
          description="用于比较两个群体的比例差异" 
          type="info" 
          showIcon 
        />
      )
    },
    {
      key: 'ab-test',
      label: (
        <span>
          <LineChartOutlined />
          A/B测试
        </span>
      ),
      children: (
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          <Alert 
            message="A/B测试比较分析" 
            description="比较对照组和实验组之间的指标差异" 
            type="info" 
            showIcon 
          />
          
          <Card title="转化率比较">
            <Row gutter={[16, 16]}>
              <Col span={6}>
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="对照组转化数"
                  value={controlConversions}
                  onChange={(value) => setControlConversions(value || 0)}
                  min={0}
                />
              </Col>
              <Col span={6}>
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="对照组总数"
                  value={controlTotal}
                  onChange={(value) => setControlTotal(value || 0)}
                  min={1}
                />
              </Col>
              <Col span={6}>
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="实验组转化数"
                  value={treatmentConversions}
                  onChange={(value) => setTreatmentConversions(value || 0)}
                  min={0}
                />
              </Col>
              <Col span={6}>
                <InputNumber
                  style={{ width: '100%' }}
                  placeholder="实验组总数"
                  value={treatmentTotal}
                  onChange={(value) => setTreatmentTotal(value || 0)}
                  min={1}
                />
              </Col>
            </Row>

            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              <Col span={8}>
                <Select
                  style={{ width: '100%' }}
                  placeholder="指标类型"
                  value={metricType}
                  onChange={setMetricType}
                  disabled
                >
                  <Option value={MetricType.CONVERSION}>转化率</Option>
                </Select>
              </Col>
              <Col span={8}>
                <Select
                  style={{ width: '100%' }}
                  placeholder="假设类型"
                  value={hypothesisType}
                  onChange={setHypothesisType}
                >
                  <Option value={HypothesisType.TWO_SIDED}>双侧检验</Option>
                  <Option value={HypothesisType.LESS}>左侧检验</Option>
                  <Option value={HypothesisType.GREATER}>右侧检验</Option>
                </Select>
              </Col>
              <Col span={8}>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={runABTestComparison}
                  disabled={loading || controlTotal === 0 || treatmentTotal === 0}
                  style={{ width: '100%', height: 40 }}
                >
                  执行A/B测试
                </Button>
              </Col>
            </Row>

            {/* 快速计算器 */}
            <Card 
              size="small" 
              style={{ marginTop: 16, backgroundColor: '#f0f9ff' }}
              title="快速统计计算"
            >
              <Row gutter={[16, 8]}>
                <Col span={12}>
                  <Text type="secondary">
                    对照组转化率: {controlTotal > 0 ? (controlConversions / controlTotal * 100).toFixed(2) : 0}%
                  </Text>
                </Col>
                <Col span={12}>
                  <Text type="secondary">
                    实验组转化率: {treatmentTotal > 0 ? (treatmentConversions / treatmentTotal * 100).toFixed(2) : 0}%
                  </Text>
                </Col>
                <Col span={24}>
                  <Text type="secondary">
                    相对提升: {controlTotal > 0 && treatmentTotal > 0 ? 
                      (((treatmentConversions / treatmentTotal) - (controlConversions / controlTotal)) / 
                       (controlConversions / controlTotal) * 100).toFixed(2) : 0}%
                  </Text>
                </Col>
              </Row>
            </Card>
          </Card>
        </Space>
      )
    },
    {
      key: 'results',
      label: (
        <span>
          <TrophyOutlined />
          检验结果
        </span>
      ),
      children: (
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {testResults.length === 0 ? (
            <Alert 
              message="还没有执行任何检验" 
              description="请在其他标签页中执行统计检验。" 
              type="info" 
              showIcon 
            />
          ) : (
            <>
              <Title level={4}>检验历史记录 ({testResults.length})</Title>
              {testResults.map((test) => renderTestResult(test.result, test.type))}
            </>
          )}
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <Spin spinning={loading}>
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {/* 页面标题 */}
          <Card>
            <Row justify="space-between" align="middle">
              <Col>
                <Space size="large" align="center">
                  <FunctionOutlined style={{ fontSize: 40, color: '#1890ff' }} />
                  <div>
                    <Title level={2} style={{ margin: 0 }}>
                      假设检验统计分析
                    </Title>
                    <Text type="secondary" style={{ fontSize: 16 }}>
                      t检验、卡方检验、A/B测试等统计推断功能
                    </Text>
                  </div>
                </Space>
              </Col>
              <Col>
                <Space>
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={checkServiceHealth}
                    disabled={loading}
                  >
                    服务状态
                  </Button>
                  <Button
                    icon={<QuestionCircleOutlined />}
                  >
                    帮助文档
                  </Button>
                </Space>
              </Col>
            </Row>
          </Card>

          {/* 核心统计卡片 */}
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="已执行检验"
                  value={testResults.length}
                  prefix={<ThunderboltOutlined style={{ color: '#1890ff' }} />}
                  suffix="次"
                />
                <Text type="secondary">历史检验记录</Text>
              </Card>
            </Col>
            
            <Col span={6}>
              <Card>
                <Statistic
                  title="显著结果"
                  value={testResults.filter(t => t.result?.result?.is_significant).length}
                  prefix={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
                  suffix="个"
                />
                <Text type="secondary">统计显著的检验</Text>
              </Card>
            </Col>
            
            <Col span={6}>
              <Card>
                <Statistic
                  title="平均效应量"
                  value={testResults.length > 0 ? (
                    testResults
                      .filter(t => t.result?.result?.effect_size)
                      .reduce((sum, t) => sum + t.result.result.effect_size, 0) / 
                    testResults.filter(t => t.result?.result?.effect_size).length
                  ).toFixed(3) : '0.000'}
                  prefix={<LineChartOutlined style={{ color: '#13c2c2' }} />}
                />
                <Text type="secondary">平均效应大小</Text>
              </Card>
            </Col>
            
            <Col span={6}>
              <Card>
                <Statistic
                  title="平均p值"
                  value={testResults.length > 0 ? (
                    testResults
                      .reduce((sum, t) => sum + (t.result?.result?.p_value || 0), 0) / 
                    testResults.length
                  ).toExponential(2) : '0.00e+0'}
                  prefix={<RocketOutlined style={{ color: '#fa8c16' }} />}
                />
                <Text type="secondary">所有检验平均</Text>
              </Card>
            </Col>
          </Row>

          {/* 主要功能标签页 */}
          <Card>
            <Tabs defaultActiveKey="ttest" items={tabItems} />
          </Card>

          {/* 检验详情模态框 */}
          <Modal
            title="检验详细结果"
            open={detailsModalVisible}
            onCancel={() => setDetailsModalVisible(false)}
            footer={[
              <Button key="close" onClick={() => setDetailsModalVisible(false)}>
                关闭
              </Button>
            ]}
            width={800}
          >
            {selectedTest && (
              <Space direction="vertical" size="middle" style={{ width: '100%' }}>
                <Text strong>消息: {selectedTest.message}</Text>
                <Divider />
                <Title level={5}>统计结果</Title>
                <pre style={{ 
                  background: '#f5f5f5', 
                  padding: 16, 
                  borderRadius: 4, 
                  fontSize: '12px',
                  overflow: 'auto',
                  maxHeight: 400
                }}>
                  {JSON.stringify(selectedTest.result, null, 2)}
                </pre>
              </Space>
            )}
          </Modal>
        </Space>
      </Spin>
    </div>
  );
};

export default HypothesisTestingPage;
