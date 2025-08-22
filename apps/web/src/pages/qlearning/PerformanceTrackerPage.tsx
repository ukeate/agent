import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Typography, 
  Space, 
  Button, 
  Slider,
  Select,
  Alert,
  Progress,
  Statistic,
  Tag,
  Tabs,
  Table,
  Switch,
  Input,
  Divider
} from 'antd';
import { Line, Column, Radar, Heatmap } from '@ant-design/charts';
import {
  MonitorOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  LineChartOutlined,
  BarChartOutlined,
  RadarChartOutlined,
  DownloadOutlined,
  EyeOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const PerformanceTrackerPage: React.FC = () => {
  const [isTracking, setIsTracking] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [trackingInterval, setTrackingInterval] = useState(10);
  
  // 性能指标历史数据
  const [performanceData, setPerformanceData] = useState([
    { episode: 0, reward: 15, loss: 0.5, accuracy: 0.2, convergence: 0.1, exploration: 0.8, stability: 0.3 },
    { episode: 10, reward: 28, loss: 0.42, accuracy: 0.35, convergence: 0.25, exploration: 0.7, stability: 0.45 },
    { episode: 20, reward: 42, loss: 0.35, accuracy: 0.5, convergence: 0.4, exploration: 0.6, stability: 0.6 },
    { episode: 30, reward: 58, loss: 0.28, accuracy: 0.65, convergence: 0.55, exploration: 0.5, stability: 0.7 },
    { episode: 40, reward: 71, loss: 0.22, accuracy: 0.75, convergence: 0.7, exploration: 0.4, stability: 0.8 },
    { episode: 50, reward: 85, loss: 0.18, accuracy: 0.82, convergence: 0.8, exploration: 0.3, stability: 0.85 },
    { episode: 60, reward: 92, loss: 0.15, accuracy: 0.87, convergence: 0.85, exploration: 0.25, stability: 0.9 },
    { episode: 70, reward: 96, loss: 0.12, accuracy: 0.9, convergence: 0.9, exploration: 0.2, stability: 0.92 },
  ]);

  const [realTimeMetrics, setRealTimeMetrics] = useState({
    currentReward: 96,
    avgReward: 68.5,
    bestReward: 98,
    rewardTrend: '+2.3%',
    loss: 0.12,
    lossReduction: '76%',
    accuracy: 0.9,
    accuracyImprovement: '+350%',
    convergence: 0.9,
    stability: 0.92,
    exploration: 0.2,
  });

  const [performanceBreakdown, setPerformanceBreakdown] = useState([
    { metric: '奖励', score: 92, weight: 0.3, contribution: 27.6 },
    { metric: '收敛性', score: 90, weight: 0.25, contribution: 22.5 },
    { metric: '稳定性', score: 92, weight: 0.2, contribution: 18.4 },
    { metric: '探索效率', score: 75, weight: 0.15, contribution: 11.25 },
    { metric: '学习速度', score: 85, weight: 0.1, contribution: 8.5 },
  ]);

  const radarData = [
    { metric: '奖励获取', value: 0.92, fullMark: 1 },
    { metric: '收敛速度', value: 0.9, fullMark: 1 },
    { metric: '稳定性', value: 0.92, fullMark: 1 },
    { metric: '探索效率', value: 0.75, fullMark: 1 },
    { metric: '泛化能力', value: 0.85, fullMark: 1 },
    { metric: '鲁棒性', value: 0.88, fullMark: 1 },
  ];

  const heatmapData = [];
  const metrics = ['奖励', '损失', '准确率', '收敛', '稳定性'];
  const episodes = [10, 20, 30, 40, 50, 60, 70];
  
  episodes.forEach(ep => {
    metrics.forEach(metric => {
      heatmapData.push({
        episode: ep,
        metric: metric,
        value: Math.random() * 0.8 + 0.2, // 模拟数据
      });
    });
  });

  const lineConfig = {
    data: performanceData.flatMap(item => [
      { episode: item.episode, value: item.reward / 100, type: '奖励' },
      { episode: item.episode, value: 1 - item.loss, type: '损失(反向)' },
      { episode: item.episode, value: item.accuracy, type: '准确率' },
      { episode: item.episode, value: item.convergence, type: '收敛性' },
    ]),
    xField: 'episode',
    yField: 'value',
    seriesField: 'type',
    smooth: true,
    point: {
      size: 3,
    },
    color: ['#1890ff', '#52c41a', '#faad14', '#722ed1'],
    legend: {
      position: 'top',
    },
    tooltip: {
      showMarkers: false,
    },
  };

  const radarConfig = {
    data: radarData,
    xField: 'metric',
    yField: 'value',
    meta: {
      value: {
        alias: '得分',
        min: 0,
        max: 1,
      },
    },
    xAxis: {
      line: null,
      tickLine: null,
    },
    yAxis: {
      label: false,
      grid: {
        alternateColor: 'rgba(0, 0, 0, 0.04)',
      },
    },
    point: {
      size: 2,
    },
    area: {},
  };

  const heatmapConfig = {
    data: heatmapData,
    xField: 'episode',
    yField: 'metric',
    colorField: 'value',
    color: ['#174c83', '#7eb6d3', '#90d3e6', '#bae7f5', '#c7ecf7'],
    meta: {
      value: {
        alias: '性能值',
      },
    },
  };

  const performanceColumns = [
    {
      title: '性能指标',
      dataIndex: 'metric',
      key: 'metric',
    },
    {
      title: '得分',
      dataIndex: 'score',
      key: 'score',
      render: (score: number) => (
        <Space>
          <Progress
            type="circle"
            percent={score}
            width={40}
            strokeColor={score >= 90 ? '#52c41a' : score >= 70 ? '#faad14' : '#f5222d'}
          />
          <Text strong>{score}</Text>
        </Space>
      ),
    },
    {
      title: '权重',
      dataIndex: 'weight',
      key: 'weight',
      render: (weight: number) => <Tag color="blue">{(weight * 100).toFixed(0)}%</Tag>,
    },
    {
      title: '贡献度',
      dataIndex: 'contribution',
      key: 'contribution',
      render: (value: number) => <Text>{value.toFixed(1)}</Text>,
    },
  ];

  const startTracking = () => {
    setIsTracking(true);
    const interval = setInterval(() => {
      setEpisode(prev => {
        if (prev >= 100) {
          setIsTracking(false);
          clearInterval(interval);
          return prev;
        }
        return prev + 1;
      });
    }, 200);
  };

  const stopTracking = () => {
    setIsTracking(false);
  };

  const resetTracking = () => {
    setIsTracking(false);
    setEpisode(0);
  };

  const overallScore = performanceBreakdown.reduce((sum, item) => sum + item.contribution, 0);

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <MonitorOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          性能追踪监控系统
        </Title>
        <Text type="secondary">
          实时监控和分析智能体的学习性能，提供多维度的性能评估和优化建议
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        {/* 总体性能评分 */}
        <Col span={24}>
          <Card title="总体性能评分">
            <Row gutter={[16, 16]}>
              <Col span={6}>
                <Statistic
                  title="综合得分"
                  value={overallScore}
                  precision={1}
                  valueStyle={{ 
                    color: overallScore >= 90 ? '#52c41a' : overallScore >= 70 ? '#faad14' : '#f5222d',
                    fontSize: '36px'
                  }}
                  suffix="/100"
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="性能等级"
                  value={overallScore >= 90 ? '优秀' : overallScore >= 70 ? '良好' : '需改进'}
                  valueStyle={{ 
                    color: overallScore >= 90 ? '#52c41a' : overallScore >= 70 ? '#faad14' : '#f5222d' 
                  }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="排名百分位"
                  value={85}
                  suffix="%"
                  valueStyle={{ color: '#722ed1' }}
                />
              </Col>
              <Col span={6}>
                <Statistic
                  title="改进潜力"
                  value={100 - overallScore}
                  precision={1}
                  suffix="分"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* 跟踪控制 */}
        <Col span={8}>
          <Card title="跟踪控制">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic title="当前回合" value={episode} />
              <Statistic title="跟踪间隔" value={trackingInterval} suffix="ms" />
              <Progress 
                percent={episode} 
                status={isTracking ? 'active' : 'normal'} 
              />
              <Space>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={startTracking}
                  disabled={isTracking}
                >
                  开始跟踪
                </Button>
                <Button
                  icon={<PauseCircleOutlined />}
                  onClick={stopTracking}
                  disabled={!isTracking}
                >
                  暂停
                </Button>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={resetTracking}
                  disabled={isTracking}
                >
                  重置
                </Button>
              </Space>
            </Space>
          </Card>
        </Col>

        {/* 实时指标 */}
        <Col span={8}>
          <Card title="实时指标">
            <Row gutter={[8, 8]}>
              <Col span={12}>
                <Statistic
                  title="当前奖励"
                  value={realTimeMetrics.currentReward}
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="平均奖励"
                  value={realTimeMetrics.avgReward}
                  precision={1}
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="当前损失"
                  value={realTimeMetrics.loss}
                  precision={3}
                  valueStyle={{ color: '#faad14' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="准确率"
                  value={realTimeMetrics.accuracy}
                  precision={2}
                  valueStyle={{ color: '#722ed1' }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* 趋势分析 */}
        <Col span={8}>
          <Card title="趋势分析">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text>奖励趋势: </Text>
                <Tag color="green">{realTimeMetrics.rewardTrend}</Tag>
              </div>
              <div>
                <Text>损失降低: </Text>
                <Tag color="blue">{realTimeMetrics.lossReduction}</Tag>
              </div>
              <div>
                <Text>准确率提升: </Text>
                <Tag color="orange">{realTimeMetrics.accuracyImprovement}</Tag>
              </div>
              <Progress
                percent={realTimeMetrics.convergence * 100}
                format={percent => `收敛: ${percent}%`}
                strokeColor="#52c41a"
              />
              <Progress
                percent={realTimeMetrics.stability * 100}
                format={percent => `稳定: ${percent}%`}
                strokeColor="#1890ff"
              />
            </Space>
          </Card>
        </Col>

        {/* 性能曲线 */}
        <Col span={12}>
          <Card title="综合性能曲线">
            <div style={{ height: 300 }}>
              <Line {...lineConfig} />
            </div>
          </Card>
        </Col>

        {/* 性能雷达图 */}
        <Col span={12}>
          <Card title="多维性能分析">
            <div style={{ height: 300 }}>
              <Radar {...radarConfig} />
            </div>
          </Card>
        </Col>

        {/* 性能热力图 */}
        <Col span={24}>
          <Card title="性能热力图">
            <div style={{ height: 250 }}>
              <Heatmap {...heatmapConfig} />
            </div>
            <Alert
              message="热力图说明"
              description="颜色越深表示性能越好，可以直观看出不同阶段各指标的表现情况"
              variant="default"
              showIcon
              style={{ marginTop: 16 }}
            />
          </Card>
        </Col>

        {/* 性能分解表格 */}
        <Col span={24}>
          <Card title="性能指标分解" extra={
            <Space>
              <Button icon={<DownloadOutlined />} size="small">导出报告</Button>
              <Button icon={<EyeOutlined />} size="small">详细分析</Button>
            </Space>
          }>
            <Table
              dataSource={performanceBreakdown}
              columns={performanceColumns}
              pagination={false}
              size="middle"
              summary={() => (
                <Table.Summary.Row>
                  <Table.Summary.Cell index={0}><Text strong>总计</Text></Table.Summary.Cell>
                  <Table.Summary.Cell index={1}>
                    <Progress 
                      type="circle" 
                      percent={Math.round(overallScore)} 
                      width={40}
                      strokeColor="#52c41a"
                    />
                  </Table.Summary.Cell>
                  <Table.Summary.Cell index={2}>
                    <Tag color="green">100%</Tag>
                  </Table.Summary.Cell>
                  <Table.Summary.Cell index={3}>
                    <Text strong>{overallScore.toFixed(1)}</Text>
                  </Table.Summary.Cell>
                </Table.Summary.Row>
              )}
            />
          </Card>
        </Col>

        {/* 高级分析 */}
        <Col span={24}>
          <Card title="高级分析">
            <Tabs defaultActiveKey="1">
              <TabPane tab="性能对比" key="1">
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Statistic
                      title="与基准模型对比"
                      value={+23.5}
                      precision={1}
                      suffix="%"
                      valueStyle={{ color: '#52c41a' }}
                      prefix="+"
                    />
                    <Text type="secondary">相比标准Q-Learning算法</Text>
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="训练效率提升"
                      value={35}
                      suffix="%"
                      valueStyle={{ color: '#1890ff' }}
                    />
                    <Text type="secondary">相比无优化版本</Text>
                  </Col>
                  <Col span={8}>
                    <Statistic
                      title="收敛速度"
                      value={2.3}
                      precision={1}
                      suffix="x"
                      valueStyle={{ color: '#faad14' }}
                    />
                    <Text type="secondary">比传统方法快</Text>
                  </Col>
                </Row>
              </TabPane>
              <TabPane tab="瓶颈分析" key="2">
                <Alert
                  message="性能瓶颈识别"
                  description={
                    <ul>
                      <li>探索效率需要进一步优化，当前得分75分</li>
                      <li>学习速度可以通过调整学习率来改善</li>
                      <li>建议增加经验回放机制以提高稳定性</li>
                    </ul>
                  }
                  variant="warning"
                  showIcon
                />
              </TabPane>
              <TabPane tab="优化建议" key="3">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Alert
                    message="短期优化建议"
                    description="调整探索率衰减策略，建议使用指数衰减替代线性衰减"
                    variant="default"
                    showIcon
                  />
                  <Alert
                    message="中期优化建议"
                    description="考虑引入优先经验回放(PER)机制，可提升学习效率15-20%"
                    variant="default"
                    showIcon
                  />
                  <Alert
                    message="长期优化建议"
                    description="评估升级到Deep Q-Network(DQN)或更高级算法的可行性"
                    variant="default"
                    showIcon
                  />
                </Space>
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default PerformanceTrackerPage;