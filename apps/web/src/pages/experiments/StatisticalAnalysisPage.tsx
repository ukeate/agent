import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Typography,
  Space,
  Select,
  Button,
  Table,
  Progress,
  Tag,
  Divider,
  Alert,
  Input,
  Form,
  Modal,
  Tooltip,
  Tabs,
} from 'antd';
import {
  BarChartOutlined,
  LineChartOutlined,
  PieChartOutlined,
  FundOutlined,
  CalculatorOutlined,
  InfoCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ExperimentOutlined,
  TrophyOutlined,
  ThunderboltOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text, Paragraph } = Typography;
const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;

interface StatisticalTest {
  experiment_id: string;
  experiment_name: string;
  test_type: 'ttest' | 'chi_square' | 'mann_whitney' | 'fisher_exact';
  test_name: string;
  control_group: {
    name: string;
    sample_size: number;
    conversions: number;
    conversion_rate: number;
    mean: number;
    std_dev: number;
  };
  treatment_group: {
    name: string;
    sample_size: number;
    conversions: number;
    conversion_rate: number;
    mean: number;
    std_dev: number;
  };
  results: {
    p_value: number;
    confidence_level: number;
    effect_size: number;
    statistical_power: number;
    minimum_detectable_effect: number;
    is_significant: boolean;
    confidence_interval: {
      lower: number;
      upper: number;
    };
  };
  interpretation: string;
  recommendation: string;
}

interface PowerAnalysis {
  current_sample_size: number;
  required_sample_size: number;
  current_power: number;
  target_power: number;
  effect_size: number;
  alpha: number;
  days_to_significance: number;
}

const StatisticalAnalysisPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<string>('exp_001');
  const [selectedTest, setSelectedTest] = useState<string>('ttest');
  const [modalVisible, setModalVisible] = useState(false);
  const [calculatorVisible, setCalculatorVisible] = useState(false);
  const [form] = Form.useForm();

  // 模拟统计分析数据
  const statisticalTests: StatisticalTest[] = [
    {
      experiment_id: 'exp_001',
      experiment_name: '首页改版A/B测试',
      test_type: 'ttest',
      test_name: 'Welch\'s t-test',
      control_group: {
        name: '原版首页',
        sample_size: 7820,
        conversions: 1134,
        conversion_rate: 14.5,
        mean: 0.145,
        std_dev: 0.352,
      },
      treatment_group: {
        name: '新版首页',
        sample_size: 7600,
        conversions: 1254,
        conversion_rate: 16.5,
        mean: 0.165,
        std_dev: 0.371,
      },
      results: {
        p_value: 0.0032,
        confidence_level: 95,
        effect_size: 0.054,
        statistical_power: 84.2,
        minimum_detectable_effect: 0.03,
        is_significant: true,
        confidence_interval: {
          lower: 0.007,
          upper: 0.033,
        },
      },
      interpretation: '治疗组的转化率显著高于对照组，效应量为中等强度。',
      recommendation: '建议全量发布新版首页，预期可提升2.0%的转化率。',
    },
    {
      experiment_id: 'exp_002',
      experiment_name: '结算页面优化',
      test_type: 'chi_square',
      test_name: 'Pearson\'s Chi-square',
      control_group: {
        name: '原版结算页',
        sample_size: 4465,
        conversions: 1046,
        conversion_rate: 23.4,
        mean: 0.234,
        std_dev: 0.423,
      },
      treatment_group: {
        name: '优化结算页',
        sample_size: 4465,
        conversions: 1253,
        conversion_rate: 28.1,
        mean: 0.281,
        std_dev: 0.449,
      },
      results: {
        p_value: 0.0001,
        confidence_level: 99,
        effect_size: 0.101,
        statistical_power: 96.8,
        minimum_detectable_effect: 0.02,
        is_significant: true,
        confidence_interval: {
          lower: 0.025,
          upper: 0.069,
        },
      },
      interpretation: '优化版结算页的转化率显著高于原版，效应量大。',
      recommendation: '强烈建议立即全量发布优化版结算页，预期转化率提升4.7%。',
    },
  ];

  const powerAnalysisData: PowerAnalysis = {
    current_sample_size: 15420,
    required_sample_size: 18500,
    current_power: 84.2,
    target_power: 90.0,
    effect_size: 0.054,
    alpha: 0.05,
    days_to_significance: 12,
  };

  const confidenceLevels = [90, 95, 99];
  const testTypes = [
    { value: 'ttest', label: 'T检验 (连续变量)' },
    { value: 'chi_square', label: '卡方检验 (分类变量)' },
    { value: 'mann_whitney', label: 'Mann-Whitney U检验 (非参数)' },
    { value: 'fisher_exact', label: 'Fisher精确检验 (小样本)' },
  ];

  const getSignificanceColor = (pValue: number): string => {
    if (pValue < 0.001) return '#52c41a';
    if (pValue < 0.01) return '#73d13d';
    if (pValue < 0.05) return '#95de64';
    return '#ff4d4f';
  };

  const getEffectSizeInterpretation = (effectSize: number): { level: string; color: string } => {
    if (effectSize < 0.02) return { level: '小', color: '#faad14' };
    if (effectSize < 0.05) return { level: '中', color: '#1890ff' };
    return { level: '大', color: '#52c41a' };
  };

  const getPowerColor = (power: number): string => {
    if (power >= 90) return '#52c41a';
    if (power >= 80) return '#faad14';
    return '#ff4d4f';
  };

  const columns: ColumnsType<StatisticalTest> = [
    {
      title: '实验名称',
      dataIndex: 'experiment_name',
      key: 'experiment_name',
      width: 200,
      render: (text: string, record: StatisticalTest) => (
        <div>
          <Text strong>{text}</Text>
          <br />
          <Tag color="blue" size="small">{record.test_name}</Tag>
        </div>
      ),
    },
    {
      title: '对照组',
      key: 'control_group',
      width: 150,
      render: (_, record: StatisticalTest) => (
        <div>
          <div><Text>{record.control_group.name}</Text></div>
          <div><Text type="secondary">样本: {record.control_group.sample_size.toLocaleString()}</Text></div>
          <div><Text type="secondary">转化: {record.control_group.conversion_rate.toFixed(1)}%</Text></div>
        </div>
      ),
    },
    {
      title: '实验组',
      key: 'treatment_group',
      width: 150,
      render: (_, record: StatisticalTest) => (
        <div>
          <div><Text>{record.treatment_group.name}</Text></div>
          <div><Text type="secondary">样本: {record.treatment_group.sample_size.toLocaleString()}</Text></div>
          <div><Text type="secondary">转化: {record.treatment_group.conversion_rate.toFixed(1)}%</Text></div>
        </div>
      ),
    },
    {
      title: 'P值',
      key: 'p_value',
      width: 100,
      render: (_, record: StatisticalTest) => (
        <div style={{ textAlign: 'center' }}>
          <Text
            strong
            style={{ 
              color: getSignificanceColor(record.results.p_value),
              fontSize: '16px' 
            }}
          >
            {record.results.p_value < 0.001 ? '<0.001' : record.results.p_value.toFixed(4)}
          </Text>
          {record.results.is_significant && (
            <div><Tag color="green" size="small">显著</Tag></div>
          )}
        </div>
      ),
    },
    {
      title: '效应量',
      key: 'effect_size',
      width: 100,
      render: (_, record: StatisticalTest) => {
        const interpretation = getEffectSizeInterpretation(record.results.effect_size);
        return (
          <div style={{ textAlign: 'center' }}>
            <Text strong>{record.results.effect_size.toFixed(3)}</Text>
            <div>
              <Tag color={interpretation.color} size="small">
                {interpretation.level}效应
              </Tag>
            </div>
          </div>
        );
      },
    },
    {
      title: '统计功效',
      key: 'statistical_power',
      width: 120,
      render: (_, record: StatisticalTest) => (
        <div style={{ textAlign: 'center' }}>
          <Progress
            type="circle"
            size={60}
            percent={record.results.statistical_power}
            format={(percent) => `${percent}%`}
            strokeColor={getPowerColor(record.results.statistical_power)}
          />
        </div>
      ),
    },
    {
      title: '置信区间',
      key: 'confidence_interval',
      width: 120,
      render: (_, record: StatisticalTest) => (
        <div>
          <Text style={{ fontSize: '12px' }}>
            95% CI:
          </Text>
          <br />
          <Text strong style={{ fontSize: '12px' }}>
            [{record.results.confidence_interval.lower.toFixed(3)}, {record.results.confidence_interval.upper.toFixed(3)}]
          </Text>
        </div>
      ),
    },
  ];

  const handleRunAnalysis = () => {
    setModalVisible(true);
  };

  const handleCalculatePower = () => {
    setCalculatorVisible(true);
  };

  useEffect(() => {
    setLoading(true);
    // 模拟数据加载
    setTimeout(() => {
      setLoading(false);
    }, 500);
  }, [selectedExperiment, selectedTest]);

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Title level={2} style={{ margin: 0 }}>
              <BarChartOutlined /> 统计分析
            </Title>
            <Text type="secondary">深度统计分析和假设检验</Text>
          </div>
          <Space>
            <Button icon={<CalculatorOutlined />} onClick={handleCalculatePower}>
              功效计算器
            </Button>
            <Button type="primary" icon={<ThunderboltOutlined />} onClick={handleRunAnalysis}>
              运行分析
            </Button>
          </Space>
        </div>
      </div>

      {/* 筛选条件 */}
      <Card style={{ marginBottom: '16px' }}>
        <Row gutter={16} align="middle">
          <Col>
            <Text>实验：</Text>
            <Select
              value={selectedExperiment}
              onChange={setSelectedExperiment}
              style={{ width: 200, marginLeft: 8 }}
            >
              <Option value="exp_001">首页改版A/B测试</Option>
              <Option value="exp_002">结算页面优化</Option>
              <Option value="exp_003">推荐算法测试</Option>
            </Select>
          </Col>
          <Col>
            <Text>检验方法：</Text>
            <Select
              value={selectedTest}
              onChange={setSelectedTest}
              style={{ width: 200, marginLeft: 8 }}
            >
              {testTypes.map(type => (
                <Option key={type.value} value={type.value}>
                  {type.label}
                </Option>
              ))}
            </Select>
          </Col>
        </Row>
      </Card>

      {/* 分析结果概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="P值"
              value={statisticalTests[0].results.p_value}
              precision={4}
              valueStyle={{ 
                color: getSignificanceColor(statisticalTests[0].results.p_value) 
              }}
              prefix={<FundOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {statisticalTests[0].results.is_significant ? '统计显著' : '不显著'}
            </Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="效应量"
              value={statisticalTests[0].results.effect_size}
              precision={3}
              prefix={<TrophyOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {getEffectSizeInterpretation(statisticalTests[0].results.effect_size).level}效应强度
            </Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="统计功效"
              value={statisticalTests[0].results.statistical_power}
              precision={1}
              suffix="%"
              valueStyle={{ 
                color: getPowerColor(statisticalTests[0].results.statistical_power) 
              }}
              prefix={<LineChartOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              检测到真实效应的概率
            </Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="最小可检测效应"
              value={statisticalTests[0].results.minimum_detectable_effect}
              precision={3}
              prefix={<PieChartOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              当前样本量可检测的最小效应
            </Text>
          </Card>
        </Col>
      </Row>

      {/* 详细分析结果 */}
      <Row gutter={16}>
        <Col span={16}>
          <Card title="统计检验结果" style={{ marginBottom: '16px' }}>
            <Table
              columns={columns}
              dataSource={statisticalTests}
              rowKey="experiment_id"
              loading={loading}
              pagination={false}
              size="small"
            />
          </Card>

          {/* 功效分析 */}
          <Card title="功效分析 (Power Analysis)">
            <Row gutter={16}>
              <Col span={12}>
                <Statistic
                  title="当前样本量"
                  value={powerAnalysisData.current_sample_size}
                  suffix="人"
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="建议样本量"
                  value={powerAnalysisData.required_sample_size}
                  suffix="人"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
            </Row>
            <Divider />
            <Row gutter={16}>
              <Col span={8}>
                <div style={{ textAlign: 'center' }}>
                  <Progress
                    type="circle"
                    percent={powerAnalysisData.current_power}
                    format={(percent) => `${percent}%`}
                    strokeColor={getPowerColor(powerAnalysisData.current_power)}
                  />
                  <div style={{ marginTop: '8px' }}>
                    <Text>当前功效</Text>
                  </div>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center' }}>
                  <Progress
                    type="circle"
                    percent={powerAnalysisData.target_power}
                    format={(percent) => `${percent}%`}
                    strokeColor="#52c41a"
                  />
                  <div style={{ marginTop: '8px' }}>
                    <Text>目标功效</Text>
                  </div>
                </div>
              </Col>
              <Col span={8}>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#1890ff' }}>
                    {powerAnalysisData.days_to_significance}
                  </div>
                  <div style={{ fontSize: '32px', color: '#1890ff' }}>天</div>
                  <div style={{ marginTop: '8px' }}>
                    <Text>预计达到显著性</Text>
                  </div>
                </div>
              </Col>
            </Row>
          </Card>
        </Col>

        <Col span={8}>
          {/* 结果解释 */}
          <Card title="结果解释" size="small" style={{ marginBottom: '16px' }}>
            <Alert
              message="统计显著性"
              description={statisticalTests[0].interpretation}
              type={statisticalTests[0].results.is_significant ? 'success' : 'warning'}
              icon={statisticalTests[0].results.is_significant ? <CheckCircleOutlined /> : <WarningOutlined />}
              showIcon
            />
            <Divider style={{ margin: '12px 0' }} />
            <div>
              <Text strong>业务建议：</Text>
              <Paragraph style={{ marginTop: '8px', fontSize: '13px' }}>
                {statisticalTests[0].recommendation}
              </Paragraph>
            </div>
          </Card>

          {/* 统计知识 */}
          <Card title="统计概念解释" size="small">
            <Tabs size="small">
              <TabPane tab="P值" key="pvalue">
                <Text style={{ fontSize: '12px' }}>
                  P值表示在零假设为真的条件下，观察到当前或更极端结果的概率。通常：
                  <br />• P &lt; 0.001: 极显著
                  <br />• P &lt; 0.01: 高度显著
                  <br />• P &lt; 0.05: 显著
                  <br />• P ≥ 0.05: 不显著
                </Text>
              </TabPane>
              <TabPane tab="效应量" key="effect">
                <Text style={{ fontSize: '12px' }}>
                  效应量衡量实际差异的大小：
                  <br />• &lt; 0.02: 小效应
                  <br />• 0.02-0.05: 中等效应
                  <br />• &gt; 0.05: 大效应
                  <br />
                  效应量比P值更能反映实际意义。
                </Text>
              </TabPane>
              <TabPane tab="统计功效" key="power">
                <Text style={{ fontSize: '12px' }}>
                  统计功效是检测出真实效应的概率：
                  <br />• &gt; 90%: 极好
                  <br />• 80-90%: 良好
                  <br />• &lt; 80%: 不足
                  <br />
                  功效不足可能导致假阴性结果。
                </Text>
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>

      {/* 运行分析模态框 */}
      <Modal
        title="运行统计分析"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setModalVisible(false)}>
            取消
          </Button>,
          <Button key="submit" type="primary" onClick={() => setModalVisible(false)}>
            运行分析
          </Button>,
        ]}
      >
        <Form form={form} layout="vertical">
          <Form.Item label="选择实验" name="experiment">
            <Select placeholder="选择要分析的实验">
              <Option value="exp_001">首页改版A/B测试</Option>
              <Option value="exp_002">结算页面优化</Option>
              <Option value="exp_003">推荐算法测试</Option>
            </Select>
          </Form.Item>
          <Form.Item label="检验方法" name="test_type">
            <Select placeholder="选择统计检验方法">
              {testTypes.map(type => (
                <Option key={type.value} value={type.value}>
                  {type.label}
                </Option>
              ))}
            </Select>
          </Form.Item>
          <Form.Item label="置信水平" name="confidence_level">
            <Select placeholder="选择置信水平" defaultValue={95}>
              {confidenceLevels.map(level => (
                <Option key={level} value={level}>
                  {level}%
                </Option>
              ))}
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* 功效计算器模态框 */}
      <Modal
        title="统计功效计算器"
        open={calculatorVisible}
        onCancel={() => setCalculatorVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setCalculatorVisible(false)}>
            取消
          </Button>,
          <Button key="calculate" type="primary" onClick={() => setCalculatorVisible(false)}>
            计算
          </Button>,
        ]}
        width={600}
      >
        <Form layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="效应量 (Effect Size)">
                <Input placeholder="0.02" suffix={
                  <Tooltip title="预期的效应大小，通常基于历史数据或业务目标">
                    <InfoCircleOutlined style={{ color: 'rgba(0,0,0,.45)' }} />
                  </Tooltip>
                } />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="显著水平 (α)">
                <Select placeholder="0.05" defaultValue="0.05">
                  <Option value="0.01">0.01</Option>
                  <Option value="0.05">0.05</Option>
                  <Option value="0.1">0.10</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="目标功效 (1-β)">
                <Select placeholder="0.80" defaultValue="0.80">
                  <Option value="0.70">70%</Option>
                  <Option value="0.80">80%</Option>
                  <Option value="0.90">90%</Option>
                  <Option value="0.95">95%</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="双尾/单尾检验">
                <Select placeholder="双尾检验" defaultValue="two-tailed">
                  <Option value="two-tailed">双尾检验</Option>
                  <Option value="one-tailed">单尾检验</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>
    </div>
  );
};

export default StatisticalAnalysisPage;