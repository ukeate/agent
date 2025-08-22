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
import { Line, Column, Gauge } from '@ant-design/charts';
import {
  CheckCircleOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  ReloadOutlined,
  StopOutlined,
  AlertOutlined,
  MonitorOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const EarlyStoppingPage: React.FC = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [episode, setEpisode] = useState(0);
  const [earlyStopTriggered, setEarlyStopTriggered] = useState(false);
  const [currentPatience, setCurrentPatience] = useState(10);
  
  // 早停参数
  const [patience, setPatience] = useState(10);
  const [minDelta, setMinDelta] = useState(0.001);
  const [monitorMetric, setMonitorMetric] = useState('loss');
  const [restoreBestWeights, setRestoreBestWeights] = useState(true);
  
  // 监控历史数据
  const [trainingHistory, setTrainingHistory] = useState([
    { episode: 0, loss: 0.5, accuracy: 0.2, reward: 15, validation_loss: 0.52 },
    { episode: 10, loss: 0.35, accuracy: 0.4, reward: 28, validation_loss: 0.38 },
    { episode: 20, loss: 0.28, accuracy: 0.55, reward: 42, validation_loss: 0.31 },
    { episode: 30, loss: 0.22, accuracy: 0.68, reward: 58, validation_loss: 0.25 },
    { episode: 40, loss: 0.18, accuracy: 0.75, reward: 71, validation_loss: 0.21 },
    { episode: 50, loss: 0.15, accuracy: 0.82, reward: 85, validation_loss: 0.18 },
    { episode: 60, loss: 0.12, accuracy: 0.85, reward: 92, validation_loss: 0.16 },
    { episode: 70, loss: 0.11, accuracy: 0.87, reward: 96, validation_loss: 0.15 },
    { episode: 80, loss: 0.105, accuracy: 0.875, reward: 97, validation_loss: 0.152 },
    { episode: 90, loss: 0.103, accuracy: 0.876, reward: 97.5, validation_loss: 0.154 },
  ]);

  const [earlyStopConditions, setEarlyStopConditions] = useState([
    {
      key: '1',
      condition: '验证损失停止改善',
      threshold: 0.001,
      patience: 10,
      status: '监控中',
      triggered: false,
      description: '验证集损失连续10个epoch没有改善超过0.001'
    },
    {
      key: '2',
      condition: '准确率收敛',
      threshold: 0.005,
      patience: 15,
      status: '监控中',
      triggered: false,
      description: '准确率连续15个epoch改善幅度小于0.005'
    },
    {
      key: '3',
      condition: '奖励饱和',
      threshold: 2.0,
      patience: 8,
      status: '监控中',
      triggered: false,
      description: '平均奖励连续8个epoch改善幅度小于2.0'
    },
    {
      key: '4',
      condition: '过拟合检测',
      threshold: 0.05,
      patience: 5,
      status: '警告',
      triggered: true,
      description: '训练损失和验证损失差距超过0.05且持续5个epoch'
    },
  ]);

  const lineConfig = {
    data: trainingHistory,
    xField: 'episode',
    yField: 'value',
    seriesField: 'type',
    smooth: true,
    point: {
      size: 4,
      shape: 'diamond',
    },
    color: ['#1890ff', '#52c41a', '#faad14', '#f5222d'],
    tooltip: {
      showMarkers: false,
    },
    legend: {
      position: 'top',
    },
  };

  const lossConfig = {
    data: trainingHistory,
    xField: 'episode',
    yField: 'loss',
    smooth: true,
    color: '#1890ff',
    point: {
      size: 4,
      shape: 'circle',
    },
    meta: {
      loss: {
        alias: '训练损失',
      },
    },
  };

  const validationLossConfig = {
    data: trainingHistory,
    xField: 'episode',
    yField: 'validation_loss',
    smooth: true,
    color: '#f5222d',
    point: {
      size: 4,
      shape: 'circle',
    },
    meta: {
      validation_loss: {
        alias: '验证损失',
      },
    },
  };

  const gaugeConfig = {
    percent: currentPatience / patience,
    range: {
      ticks: [0, 1/3, 2/3, 1],
      color: ['#30BF78', '#FAAD14', '#F4664A'],
    },
    indicator: {
      pointer: {
        style: {
          stroke: '#D0D0D0',
        },
      },
      pin: {
        style: {
          stroke: '#D0D0D0',
        },
      },
    },
    statistic: {
      content: {
        style: {
          fontSize: '36px',
          lineHeight: '36px',
        },
        formatter: () => `${currentPatience}/${patience}`,
      },
    },
  };

  const conditionsColumns = [
    {
      title: '触发条件',
      dataIndex: 'condition',
      key: 'condition',
    },
    {
      title: '阈值',
      dataIndex: 'threshold',
      key: 'threshold',
      render: (value: number) => <Tag color="blue">{value}</Tag>,
    },
    {
      title: '耐心值',
      dataIndex: 'patience',
      key: 'patience',
      render: (value: number) => <Text>{value} epochs</Text>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: any) => (
        <Tag color={record.triggered ? 'red' : status === '警告' ? 'orange' : 'green'}>
          {status}
        </Tag>
      ),
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
    },
  ];

  const startTraining = () => {
    setIsTraining(true);
    setEarlyStopTriggered(false);
    const interval = setInterval(() => {
      setEpisode(prev => {
        const newEpisode = prev + 1;
        
        // 模拟早停逻辑
        if (newEpisode > 50 && Math.random() < 0.1) {
          setEarlyStopTriggered(true);
          setIsTraining(false);
          clearInterval(interval);
          return newEpisode;
        }
        
        if (newEpisode >= 100) {
          setIsTraining(false);
          clearInterval(interval);
        }
        
        // 更新耐心值
        setCurrentPatience(Math.max(0, patience - Math.floor(newEpisode / 10)));
        
        return newEpisode;
      });
    }, 300);
  };

  const stopTraining = () => {
    setIsTraining(false);
    setEarlyStopTriggered(true);
  };

  const resetTraining = () => {
    setIsTraining(false);
    setEpisode(0);
    setEarlyStopTriggered(false);
    setCurrentPatience(patience);
  };

  const bestEpoch = trainingHistory.reduce((best, current) => 
    current.validation_loss < best.validation_loss ? current : best
  );

  return (
    <div style={{ padding: '24px', background: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <CheckCircleOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
          早停机制系统
        </Title>
        <Text type="secondary">
          智能监控训练过程，在模型性能停止改善时自动停止训练，防止过拟合
        </Text>
      </div>

      <Row gutter={[24, 24]}>
        {/* 早停状态监控 */}
        <Col span={24}>
          {earlyStopTriggered && (
            <Alert
              message="早停机制已触发"
              description={`训练在第${episode}个回合停止，模型已恢复到第${bestEpoch.episode}个回合的最佳权重`}
              variant="warning"
              showIcon
              style={{ marginBottom: 16 }}
              action={
                <Button size="small" onClick={resetTraining}>
                  重新开始
                </Button>
              }
            />
          )}
        </Col>

        {/* 训练控制 */}
        <Col span={8}>
          <Card title="训练控制">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic title="当前回合" value={episode} />
              <Statistic
                title="最佳回合"
                value={bestEpoch.episode}
                valueStyle={{ color: '#52c41a' }}
              />
              <Progress 
                percent={episode} 
                status={earlyStopTriggered ? 'exception' : isTraining ? 'active' : 'normal'} 
              />
              <Space>
                <Button
                  type="primary"
                  icon={<PlayCircleOutlined />}
                  onClick={startTraining}
                  disabled={isTraining}
                >
                  开始训练
                </Button>
                <Button
                  icon={<StopOutlined />}
                  onClick={stopTraining}
                  disabled={!isTraining}
                  danger
                >
                  手动停止
                </Button>
                <Button
                  icon={<ReloadOutlined />}
                  onClick={resetTraining}
                  disabled={isTraining}
                >
                  重置
                </Button>
              </Space>
            </Space>
          </Card>
        </Col>

        {/* 早停参数配置 */}
        <Col span={8}>
          <Card title="早停参数">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>监控指标</Text>
                <Select
                  value={monitorMetric}
                  onChange={setMonitorMetric}
                  style={{ width: '100%', marginTop: 8 }}
                >
                  <Option value="loss">验证损失</Option>
                  <Option value="accuracy">准确率</Option>
                  <Option value="reward">奖励</Option>
                </Select>
              </div>
              <div>
                <Text strong>耐心值: {patience}</Text>
                <Slider
                  min={5}
                  max={50}
                  value={patience}
                  onChange={setPatience}
                  marks={{ 5: '5', 10: '10', 20: '20', 50: '50' }}
                />
              </div>
              <div>
                <Text strong>最小改善: {minDelta}</Text>
                <Slider
                  min={0.0001}
                  max={0.01}
                  step={0.0001}
                  value={minDelta}
                  onChange={setMinDelta}
                  marks={{ 0.0001: '0.0001', 0.001: '0.001', 0.005: '0.005', 0.01: '0.01' }}
                />
              </div>
              <div>
                <Space>
                  <Text strong>恢复最佳权重</Text>
                  <Switch
                    checked={restoreBestWeights}
                    onChange={setRestoreBestWeights}
                  />
                </Space>
                <Text type="secondary" style={{ display: 'block', marginTop: 4 }}>
                  早停时恢复到验证集上表现最好的模型权重
                </Text>
              </div>
            </Space>
          </Card>
        </Col>

        {/* 耐心值仪表盘 */}
        <Col span={8}>
          <Card title="耐心值监控">
            <div style={{ height: 200 }}>
              <Gauge {...gaugeConfig} />
            </div>
            <div style={{ textAlign: 'center', marginTop: 16 }}>
              <Text type="secondary">
                剩余耐心: {currentPatience} / {patience}
              </Text>
            </div>
          </Card>
        </Col>

        {/* 训练损失曲线 */}
        <Col span={12}>
          <Card title="训练损失曲线">
            <div style={{ height: 300 }}>
              <Line {...lossConfig} />
            </div>
          </Card>
        </Col>

        {/* 验证损失曲线 */}
        <Col span={12}>
          <Card title="验证损失曲线">
            <div style={{ height: 300 }}>
              <Line {...validationLossConfig} />
            </div>
          </Card>
        </Col>

        {/* 最佳模型信息 */}
        <Col span={8}>
          <Card title="最佳模型信息">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Statistic
                title="最佳回合"
                value={bestEpoch.episode}
                valueStyle={{ color: '#52c41a' }}
              />
              <Statistic
                title="最佳验证损失"
                value={bestEpoch.validation_loss}
                precision={4}
                valueStyle={{ color: '#1890ff' }}
              />
              <Statistic
                title="对应准确率"
                value={bestEpoch.accuracy}
                precision={3}
                valueStyle={{ color: '#faad14' }}
              />
              <Statistic
                title="对应奖励"
                value={bestEpoch.reward}
                valueStyle={{ color: '#722ed1' }}
              />
            </Space>
          </Card>
        </Col>

        {/* 过拟合检测 */}
        <Col span={8}>
          <Card title="过拟合检测">
            <Space direction="vertical" style={{ width: '100%' }}>
              <div>
                <Text strong>训练-验证差距</Text>
                <Progress
                  percent={Math.abs(trainingHistory[trainingHistory.length - 1]?.loss - trainingHistory[trainingHistory.length - 1]?.validation_loss) * 1000}
                  status={Math.abs(trainingHistory[trainingHistory.length - 1]?.loss - trainingHistory[trainingHistory.length - 1]?.validation_loss) > 0.05 ? 'exception' : 'success'}
                  format={percent => `${(percent / 1000).toFixed(3)}`}
                />
              </div>
              <Alert
                message={Math.abs(trainingHistory[trainingHistory.length - 1]?.loss - trainingHistory[trainingHistory.length - 1]?.validation_loss) > 0.05 ? '检测到过拟合风险' : '模型泛化良好'}
                type={Math.abs(trainingHistory[trainingHistory.length - 1]?.loss - trainingHistory[trainingHistory.length - 1]?.validation_loss) > 0.05 ? 'warning' : 'success'}
                showIcon
              />
            </Space>
          </Card>
        </Col>

        {/* 实时统计 */}
        <Col span={8}>
          <Card title="实时统计">
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Statistic
                  title="已训练"
                  value={episode}
                  suffix="回合"
                  valueStyle={{ color: '#1890ff' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="无改善"
                  value={patience - currentPatience}
                  suffix="回合"
                  valueStyle={{ color: '#faad14' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="早停概率"
                  value={((patience - currentPatience) / patience * 100)}
                  precision={1}
                  suffix="%"
                  valueStyle={{ color: '#f5222d' }}
                />
              </Col>
              <Col span={12}>
                <Statistic
                  title="预计剩余"
                  value={currentPatience}
                  suffix="回合"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
            </Row>
          </Card>
        </Col>

        {/* 早停条件表格 */}
        <Col span={24}>
          <Card title="早停条件监控">
            <Table
              dataSource={earlyStopConditions}
              columns={conditionsColumns}
              pagination={false}
              size="middle"
            />
          </Card>
        </Col>

        {/* 高级配置 */}
        <Col span={24}>
          <Card title="高级配置">
            <Tabs defaultActiveKey="1">
              <TabPane tab="多指标监控" key="1">
                <Row gutter={[16, 16]}>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>损失监控</Text>
                      <Switch defaultChecked />
                      <Text type="secondary">监控训练和验证损失</Text>
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>准确率监控</Text>
                      <Switch defaultChecked />
                      <Text type="secondary">监控模型准确率变化</Text>
                    </Space>
                  </Col>
                  <Col span={8}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Text strong>奖励监控</Text>
                      <Switch />
                      <Text type="secondary">监控强化学习奖励变化</Text>
                    </Space>
                  </Col>
                </Row>
              </TabPane>
              <TabPane tab="自动调整" key="2">
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Alert
                    message="智能耐心值调整"
                    description="系统可根据训练进度自动调整耐心值，在训练初期使用较大耐心值，后期使用较小耐心值"
                    variant="default"
                    showIcon
                  />
                  <Space>
                    <Text strong>启用自动调整</Text>
                    <Switch />
                  </Space>
                </Space>
              </TabPane>
              <TabPane tab="历史记录" key="3">
                <Alert
                  message="早停历史"
                  description="共执行了12次训练，其中8次触发早停机制，平均节省训练时间35%，有效防止了过拟合"
                  type="success"
                  showIcon
                />
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default EarlyStoppingPage;