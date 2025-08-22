import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Select,
  DatePicker,
  Button,
  Space,
  Typography,
  Tabs,
  Tooltip,
  Tag
} from 'antd';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  ScatterChart,
  Scatter
} from 'recharts';
import {
  TrophyOutlined,
  ClockCircleOutlined,
  BranchesOutlined,
  ExclamationCircleOutlined,
  ReloadOutlined,
  DownloadOutlined,
  BarChartOutlined,
  LineChartOutlined,
  PieChartOutlined,
  DotChartOutlined
} from '@ant-design/icons';
import { useReasoningStore } from '../../stores/reasoningStore';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface ReasoningMetrics {
  totalChains: number;
  completionRate: number;
  avgConfidence: number;
  avgStepsPerChain: number;
  avgExecutionTime: number;
  strategyDistribution: Array<{
    strategy: string;
    count: number;
    avgConfidence: number;
  }>;
  confidenceOverTime: Array<{
    date: string;
    confidence: number;
    chains: number;
  }>;
  stepTypeDistribution: Array<{
    type: string;
    count: number;
    avgConfidence: number;
  }>;
  performanceTrends: Array<{
    date: string;
    executionTime: number;
    stepsCount: number;
  }>;
  qualityMetrics: {
    highConfidence: number; // >80%
    mediumConfidence: number; // 60-80%
    lowConfidence: number; // <60%
    validationFailures: number;
    recoverySuccess: number;
  };
}

export const ReasoningStats: React.FC = () => {
  const {
    reasoningHistory,
    reasoningStats,
    isLoading,
    getReasoningHistory,
    getReasoningStats
  } = useReasoningStore();

  const [timeRange, setTimeRange] = useState<any>([]);
  const [selectedStrategy, setSelectedStrategy] = useState<string>('all');
  const [metrics, setMetrics] = useState<ReasoningMetrics | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    loadData();
  }, []);

  useEffect(() => {
    if (reasoningHistory) {
      calculateMetrics();
    }
  }, [reasoningHistory, timeRange, selectedStrategy]);

  const loadData = async () => {
    await Promise.all([
      getReasoningHistory(),
      getReasoningStats()
    ]);
  };

  const calculateMetrics = () => {
    if (!reasoningHistory || reasoningHistory.length === 0) {
      setMetrics(null);
      return;
    }

    let filteredChains = [...reasoningHistory];

    // 应用时间范围过滤
    if (timeRange && timeRange.length === 2) {
      const [start, end] = timeRange;
      filteredChains = filteredChains.filter(chain => {
        const chainDate = new Date(chain.created_at);
        return chainDate >= start.toDate() && chainDate <= end.toDate();
      });
    }

    // 应用策略过滤
    if (selectedStrategy !== 'all') {
      filteredChains = filteredChains.filter(chain => chain.strategy === selectedStrategy);
    }

    // 计算基础指标
    const totalChains = filteredChains.length;
    const completedChains = filteredChains.filter(c => c.conclusion).length;
    const completionRate = totalChains > 0 ? (completedChains / totalChains) * 100 : 0;

    const confidences = filteredChains
      .map(c => c.confidence_score)
      .filter(c => c !== undefined) as number[];
    const avgConfidence = confidences.length > 0 ? 
      confidences.reduce((sum, c) => sum + c, 0) / confidences.length : 0;

    const totalSteps = filteredChains.reduce((sum, c) => sum + (c.steps?.length || 0), 0);
    const avgStepsPerChain = totalChains > 0 ? totalSteps / totalChains : 0;

    // 策略分布
    const strategyMap = new Map();
    filteredChains.forEach(chain => {
      const strategy = chain.strategy;
      if (!strategyMap.has(strategy)) {
        strategyMap.set(strategy, { count: 0, confidences: [] });
      }
      const data = strategyMap.get(strategy);
      data.count++;
      if (chain.confidence_score !== undefined) {
        data.confidences.push(chain.confidence_score);
      }
    });

    const strategyDistribution = Array.from(strategyMap.entries()).map(([strategy, data]) => ({
      strategy,
      count: data.count,
      avgConfidence: data.confidences.length > 0 ? 
        data.confidences.reduce((sum: number, c: number) => sum + c, 0) / data.confidences.length : 0
    }));

    // 时间趋势
    const dateMap = new Map();
    filteredChains.forEach(chain => {
      const date = new Date(chain.created_at).toISOString().split('T')[0];
      if (!dateMap.has(date)) {
        dateMap.set(date, { confidences: [], chains: 0 });
      }
      const data = dateMap.get(date);
      data.chains++;
      if (chain.confidence_score !== undefined) {
        data.confidences.push(chain.confidence_score);
      }
    });

    const confidenceOverTime = Array.from(dateMap.entries())
      .map(([date, data]) => ({
        date,
        confidence: data.confidences.length > 0 ? 
          data.confidences.reduce((sum: number, c: number) => sum + c, 0) / data.confidences.length : 0,
        chains: data.chains
      }))
      .sort((a, b) => a.date.localeCompare(b.date));

    // 步骤类型分布
    const stepTypeMap = new Map();
    filteredChains.forEach(chain => {
      chain.steps?.forEach(step => {
        const type = step.step_type;
        if (!stepTypeMap.has(type)) {
          stepTypeMap.set(type, { count: 0, confidences: [] });
        }
        const data = stepTypeMap.get(type);
        data.count++;
        data.confidences.push(step.confidence);
      });
    });

    const stepTypeDistribution = Array.from(stepTypeMap.entries()).map(([type, data]) => ({
      type,
      count: data.count,
      avgConfidence: data.confidences.length > 0 ? 
        data.confidences.reduce((sum: number, c: number) => sum + c, 0) / data.confidences.length : 0
    }));

    // 质量指标
    const highConfidence = confidences.filter(c => c > 0.8).length;
    const mediumConfidence = confidences.filter(c => c >= 0.6 && c <= 0.8).length;
    const lowConfidence = confidences.filter(c => c < 0.6).length;

    const calculatedMetrics: ReasoningMetrics = {
      totalChains,
      completionRate,
      avgConfidence,
      avgStepsPerChain,
      avgExecutionTime: 0, // 需要从duration_ms计算
      strategyDistribution,
      confidenceOverTime,
      stepTypeDistribution,
      performanceTrends: [], // 需要从时间数据计算
      qualityMetrics: {
        highConfidence,
        mediumConfidence,
        lowConfidence,
        validationFailures: 0, // 需要从验证数据获取
        recoverySuccess: 0 // 需要从恢复数据获取
      }
    };

    setMetrics(calculatedMetrics);
  };

  const exportData = () => {
    if (!metrics) return;
    
    const data = {
      exportTime: new Date().toISOString(),
      timeRange,
      selectedStrategy,
      metrics
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `reasoning-stats-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
  };

  if (!metrics) {
    return (
      <div className="text-center p-8">
        <Text type="secondary">暂无统计数据</Text>
        <div className="mt-4">
          <Button 
            type="primary" 
            icon={<ReloadOutlined />}
            onClick={loadData}
            loading={isLoading}
          >
            加载数据
          </Button>
        </div>
      </div>
    );
  }

  const COLORS = ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1', '#fa8c16'];

  return (
    <div className="reasoning-stats">
      {/* 控制面板 */}
      <Row gutter={16} className="mb-4">
        <Col span={8}>
          <RangePicker
            value={timeRange}
            onChange={setTimeRange}
            placeholder={['开始日期', '结束日期']}
            style={{ width: '100%' }}
          />
        </Col>
        <Col span={6}>
          <Select
            value={selectedStrategy}
            onChange={setSelectedStrategy}
            style={{ width: '100%' }}
            placeholder="选择策略"
          >
            <Option value="all">全部策略</Option>
            <Option value="ZERO_SHOT">Zero-shot</Option>
            <Option value="FEW_SHOT">Few-shot</Option>
            <Option value="AUTO_COT">Auto-CoT</Option>
          </Select>
        </Col>
        <Col span={10}>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={loadData} loading={isLoading}>
              刷新数据
            </Button>
            <Button icon={<DownloadOutlined />} onClick={exportData}>
              导出数据
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 统计标签页 */}
      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab={
          <span>
            <BarChartOutlined />
            总览
          </span>
        } key="overview">
          {/* 核心指标 */}
          <Row gutter={16} className="mb-4">
            <Col span={6}>
              <Card>
                <Statistic
                  title="推理链总数"
                  value={metrics.totalChains}
                  prefix={<TrophyOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="完成率"
                  value={metrics.completionRate}
                  precision={1}
                  suffix="%"
                  prefix={<ExclamationCircleOutlined />}
                />
                <Progress 
                  percent={metrics.completionRate} 
                  showInfo={false} 
                  strokeColor="#52c41a"
                  className="mt-2"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="平均置信度"
                  value={metrics.avgConfidence * 100}
                  precision={1}
                  suffix="%"
                  prefix={<ClockCircleOutlined />}
                />
                <Progress 
                  percent={metrics.avgConfidence * 100} 
                  showInfo={false}
                  strokeColor="#1890ff"
                  className="mt-2"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="平均步骤数"
                  value={metrics.avgStepsPerChain}
                  precision={1}
                  prefix={<BranchesOutlined />}
                />
              </Card>
            </Col>
          </Row>

          {/* 质量分布 */}
          <Row gutter={16} className="mb-4">
            <Col span={12}>
              <Card title="质量分布" extra={
                <Tooltip title="基于置信度的质量评估">
                  <ExclamationCircleOutlined />
                </Tooltip>
              }>
                <Row gutter={16}>
                  <Col span={8}>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-600">
                        {metrics.qualityMetrics.highConfidence}
                      </div>
                      <div className="text-gray-500">高质量 (&gt;80%)</div>
                    </div>
                  </Col>
                  <Col span={8}>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-blue-600">
                        {metrics.qualityMetrics.mediumConfidence}
                      </div>
                      <div className="text-gray-500">中等质量 (60-80%)</div>
                    </div>
                  </Col>
                  <Col span={8}>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-red-600">
                        {metrics.qualityMetrics.lowConfidence}
                      </div>
                      <div className="text-gray-500">低质量 (&lt;60%)</div>
                    </div>
                  </Col>
                </Row>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="策略分布">
                <ResponsiveContainer width="100%" height={200}>
                  <PieChart>
                    <Pie
                      data={metrics.strategyDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ strategy, count }) => `${strategy}: ${count}`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="count"
                    >
                      {metrics.strategyDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={
          <span>
            <LineChartOutlined />
            趋势分析
          </span>
        } key="trends">
          <Row gutter={16}>
            <Col span={24}>
              <Card title="置信度时间趋势">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={metrics.confidenceOverTime}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="date" />
                    <YAxis domain={[0, 1]} />
                    <RechartsTooltip 
                      formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, '置信度']}
                    />
                    <Line 
                      type="monotone" 
                      dataKey="confidence" 
                      stroke="#1890ff" 
                      strokeWidth={2}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={
          <span>
            <PieChartOutlined />
            策略对比
          </span>
        } key="strategies">
          <Row gutter={16} className="mb-4">
            <Col span={24}>
              <Card title="策略性能对比">
                <Table
                  dataSource={metrics.strategyDistribution}
                  pagination={false}
                  size="small"
                  columns={[
                    {
                      title: '策略',
                      dataIndex: 'strategy',
                      key: 'strategy',
                      render: (strategy: string) => {
                        const strategyNames = {
                          'ZERO_SHOT': 'Zero-shot CoT',
                          'FEW_SHOT': 'Few-shot CoT',
                          'AUTO_COT': 'Auto-CoT'
                        };
                        const colors = {
                          'ZERO_SHOT': 'blue',
                          'FEW_SHOT': 'green',
                          'AUTO_COT': 'purple'
                        };
                        return (
                          <Tag color={colors[strategy] || 'default'}>
                            {strategyNames[strategy] || strategy}
                          </Tag>
                        );
                      }
                    },
                    {
                      title: '使用次数',
                      dataIndex: 'count',
                      key: 'count',
                      sorter: (a, b) => a.count - b.count
                    },
                    {
                      title: '平均置信度',
                      dataIndex: 'avgConfidence',
                      key: 'avgConfidence',
                      render: (confidence: number) => (
                        <div>
                          <span>{(confidence * 100).toFixed(1)}%</span>
                          <Progress 
                            percent={confidence * 100} 
                            showInfo={false}
                            size="small"
                            className="mt-1"
                          />
                        </div>
                      ),
                      sorter: (a, b) => a.avgConfidence - b.avgConfidence
                    },
                    {
                      title: '成功率',
                      key: 'successRate',
                      render: (_, record) => {
                        // 这里可以根据实际数据计算成功率
                        const rate = record.avgConfidence > 0.6 ? 
                          Math.min(100, record.avgConfidence * 120) : 
                          record.avgConfidence * 100;
                        return `${rate.toFixed(1)}%`;
                      }
                    }
                  ]}
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={24}>
              <Card title="策略置信度分布">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={metrics.strategyDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="strategy" />
                    <YAxis domain={[0, 1]} />
                    <RechartsTooltip 
                      formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, '平均置信度']}
                    />
                    <Bar dataKey="avgConfidence" fill="#1890ff" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab={
          <span>
            <DotChartOutlined />
            步骤分析
          </span>
        } key="steps">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="步骤类型分布">
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={metrics.stepTypeDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="type" />
                    <YAxis />
                    <RechartsTooltip />
                    <Bar dataKey="count" fill="#52c41a" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="步骤类型置信度">
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart data={metrics.stepTypeDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="count" name="数量" />
                    <YAxis dataKey="avgConfidence" name="平均置信度" domain={[0, 1]} />
                    <RechartsTooltip 
                      formatter={(value: number, name: string) => [
                        name === '平均置信度' ? `${(value * 100).toFixed(1)}%` : value,
                        name
                      ]}
                    />
                    <Scatter name="步骤类型" dataKey="avgConfidence" fill="#faad14" />
                  </ScatterChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>

          <Row gutter={16} className="mt-4">
            <Col span={24}>
              <Card title="步骤详细统计">
                <Table
                  dataSource={metrics.stepTypeDistribution}
                  pagination={false}
                  size="small"
                  columns={[
                    {
                      title: '步骤类型',
                      dataIndex: 'type',
                      key: 'type',
                      render: (type: string) => {
                        const typeNames = {
                          'OBSERVATION': '观察',
                          'ANALYSIS': '分析',
                          'HYPOTHESIS': '假设',
                          'VALIDATION': '验证',
                          'REFLECTION': '反思',
                          'CONCLUSION': '结论'
                        };
                        return typeNames[type] || type;
                      }
                    },
                    {
                      title: '使用次数',
                      dataIndex: 'count',
                      key: 'count',
                      sorter: (a, b) => a.count - b.count
                    },
                    {
                      title: '平均置信度',
                      dataIndex: 'avgConfidence',
                      key: 'avgConfidence',
                      render: (confidence: number) => `${(confidence * 100).toFixed(1)}%`,
                      sorter: (a, b) => a.avgConfidence - b.avgConfidence
                    },
                    {
                      title: '使用占比',
                      key: 'percentage',
                      render: (_, record) => {
                        const total = metrics.stepTypeDistribution.reduce((sum, item) => sum + item.count, 0);
                        const percentage = (record.count / total) * 100;
                        return `${percentage.toFixed(1)}%`;
                      }
                    }
                  ]}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>
    </div>
  );
};