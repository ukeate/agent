import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Typography, 
  Statistic, 
  Progress, 
  Table,
  Alert,
  Timeline,
  Tabs,
  Select,
  Button,
  Space,
  Tag,
  Descriptions,
  Switch,
  Input,
  Slider
} from 'antd';
import {
  MonitorOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  ClockCircleOutlined,
  LineChartOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  WarningOutlined,
  SettingOutlined,
  ReloadOutlined,
  PauseOutlined,
  PlayCircleOutlined
} from '@ant-design/icons';
import { Line, Area, Gauge, Column } from '@ant-design/charts';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const TrainingMonitorDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [isRealTime, setIsRealTime] = useState(true);
  const [refreshInterval, setRefreshInterval] = useState(5);

  // 模拟实时数据
  const [metricsData, setMetricsData] = useState([
    { step: 0, loss: 2.45, lr: 2e-4, gpu: 78, memory: 14.2, accuracy: 0.23 },
    { step: 100, loss: 1.89, lr: 1.98e-4, gpu: 82, memory: 15.1, accuracy: 0.45 },
    { step: 200, loss: 1.56, lr: 1.96e-4, gpu: 85, memory: 15.8, accuracy: 0.67 },
    { step: 300, loss: 1.34, lr: 1.94e-4, gpu: 88, memory: 16.2, accuracy: 0.78 },
    { step: 400, loss: 1.18, lr: 1.92e-4, gpu: 87, memory: 16.5, accuracy: 0.82 },
    { step: 500, loss: 0.98, lr: 1.90e-4, gpu: 89, memory: 16.1, accuracy: 0.87 }
  ]);

  const [currentMetrics, setCurrentMetrics] = useState({
    loss: 0.8743,
    accuracy: 87.3,
    lr: 1.89e-4,
    gpu_usage: 87,
    gpu_memory: 16.2,
    gpu_temp: 73,
    step: 567,
    epoch: 2,
    progress: 67.8
  });

  // 实时数据更新模拟
  useEffect(() => {
    if (!isRealTime) return;
    
    const interval = setInterval(() => {
      setCurrentMetrics(prev => ({
        ...prev,
        loss: Math.max(0.1, prev.loss - Math.random() * 0.01),
        accuracy: Math.min(99, prev.accuracy + Math.random() * 0.5),
        gpu_usage: 85 + Math.random() * 10,
        gpu_temp: 70 + Math.random() * 15,
        step: prev.step + Math.floor(Math.random() * 5),
        progress: Math.min(100, prev.progress + Math.random() * 0.5)
      }));
    }, refreshInterval * 1000);

    return () => clearInterval(interval);
  }, [isRealTime, refreshInterval]);

  const lossConfig = {
    data: metricsData,
    xField: 'step',
    yField: 'loss',
    smooth: true,
    color: '#1890ff',
    point: { size: 3, shape: 'circle' },
    tooltip: { showMarkers: false },
    meta: { loss: { alias: '训练损失' } }
  };

  const accuracyConfig = {
    data: metricsData,
    xField: 'step',
    yField: 'accuracy',
    smooth: true,
    color: '#52c41a',
    point: { size: 3, shape: 'circle' },
    tooltip: { showMarkers: false },
    meta: { accuracy: { alias: '准确率' } }
  };

  const gpuConfig = {
    data: metricsData,
    xField: 'step',
    yField: 'gpu',
    smooth: true,
    color: '#faad14',
    areaStyle: { fillOpacity: 0.3 },
    tooltip: { showMarkers: false },
    meta: { gpu: { alias: 'GPU使用率 (%)' } }
  };

  const gaugeConfig = {
    percent: currentMetrics.gpu_usage / 100,
    range: { color: '#1890ff' },
    indicator: {
      pointer: { style: { stroke: '#D0D0D0' } },
      pin: { style: { stroke: '#D0D0D0' } }
    },
    statistic: {
      content: {
        style: { fontSize: '32px', lineHeight: '32px' },
        formatter: () => `${currentMetrics.gpu_usage.toFixed(1)}%`
      }
    }
  };

  const runningJobs = [
    { 
      id: '1', 
      name: 'llama2-lora-chat-v1', 
      status: '训练中', 
      progress: 67.8, 
      gpu: '0,1', 
      startTime: '14:30:00',
      eta: '04:32:15'
    },
    { 
      id: '2', 
      name: 'mistral-qlora-code-v2', 
      status: '队列中', 
      progress: 0, 
      gpu: '-', 
      startTime: '-',
      eta: '等待中'
    },
    { 
      id: '3', 
      name: 'qwen-lora-summary-v1', 
      status: '暂停', 
      progress: 23.5, 
      gpu: '2', 
      startTime: '12:15:30',
      eta: '暂停中'
    }
  ];

  const jobColumns = [
    { title: '任务名称', dataIndex: 'name', key: 'name' },
    { 
      title: '状态', 
      dataIndex: 'status', 
      key: 'status',
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          '训练中': 'processing',
          '队列中': 'default',
          '暂停': 'warning',
          '完成': 'success',
          '失败': 'error'
        };
        return <Tag color={colorMap[status]}>{status}</Tag>;
      }
    },
    {
      title: '进度',
      dataIndex: 'progress',
      key: 'progress',
      render: (progress: number) => (
        <div style={{ width: 100 }}>
          <Progress percent={progress} size="small" />
        </div>
      )
    },
    { title: 'GPU', dataIndex: 'gpu', key: 'gpu' },
    { title: '开始时间', dataIndex: 'startTime', key: 'startTime' },
    { title: '预计完成', dataIndex: 'eta', key: 'eta' },
    {
      title: '操作',
      key: 'action',
      render: (record: any) => (
        <Space>
          {record.status === '训练中' ? (
            <Button size="small" icon={<PauseOutlined />}>暂停</Button>
          ) : record.status === '暂停' ? (
            <Button size="small" icon={<PlayCircleOutlined />}>继续</Button>
          ) : null}
          <Button size="small" danger>停止</Button>
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Title level={2}>
              <MonitorOutlined style={{ marginRight: 8, color: '#1890ff' }} />
              训练监控仪表板
            </Title>
            <Text type="secondary">
              实时监控所有训练任务的状态、性能和资源使用情况
            </Text>
          </div>
          <Space>
            <Switch 
              checked={isRealTime} 
              onChange={setIsRealTime}
              checkedChildren="实时"
              unCheckedChildren="静态"
            />
            <Select 
              value={refreshInterval}
              onChange={setRefreshInterval}
              disabled={!isRealTime}
              style={{ width: 120 }}
            >
              <Option value={1}>1秒</Option>
              <Option value={5}>5秒</Option>
              <Option value={10}>10秒</Option>
              <Option value={30}>30秒</Option>
            </Select>
            <Button icon={<ReloadOutlined />}>刷新</Button>
            <Button icon={<SettingOutlined />}>设置</Button>
          </Space>
        </div>
      </div>

      {/* 实时状态概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={4}>
          <Card>
            <Statistic
              title="活跃任务"
              value={runningJobs.filter(j => j.status === '训练中').length}
              prefix={<PlayCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={5}>
          <Card>
            <Statistic
              title="实时损失"
              value={currentMetrics.loss}
              precision={4}
              prefix={<LineChartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={5}>
          <Card>
            <Statistic
              title="平均准确率"
              value={currentMetrics.accuracy}
              precision={1}
              suffix="%"
              prefix={<CheckCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={5}>
          <Card>
            <Statistic
              title="GPU使用率"
              value={currentMetrics.gpu_usage}
              precision={1}
              suffix="%"
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#faad14' }}
            />
          </Card>
        </Col>
        <Col span={5}>
          <Card>
            <Statistic
              title="显存使用"
              value={`${currentMetrics.gpu_memory}/24.0`}
              suffix="GB"
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="总览监控" key="overview">
            <Row gutter={16}>
              <Col span={8}>
                <Card title="训练损失曲线" size="small" style={{ height: 350, marginBottom: 16 }}>
                  <Line {...lossConfig} height={280} />
                </Card>
              </Col>
              <Col span={8}>
                <Card title="准确率趋势" size="small" style={{ height: 350, marginBottom: 16 }}>
                  <Line {...accuracyConfig} height={280} />
                </Card>
              </Col>
              <Col span={8}>
                <Card title="GPU使用率" size="small" style={{ height: 350, marginBottom: 16 }}>
                  <div style={{ height: 200, display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
                    <Gauge {...gaugeConfig} width={180} height={180} />
                  </div>
                  <div style={{ textAlign: 'center', marginTop: 16 }}>
                    <Text>GPU 温度: {currentMetrics.gpu_temp.toFixed(1)}°C</Text>
                  </div>
                </Card>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Card title="系统资源监控" size="small">
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <Text>GPU 0 使用率</Text>
                      <Text>{currentMetrics.gpu_usage.toFixed(1)}%</Text>
                    </div>
                    <Progress percent={currentMetrics.gpu_usage} strokeColor="#1890ff" size="small" />
                  </div>
                  
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <Text>GPU 0 显存</Text>
                      <Text>{currentMetrics.gpu_memory.toFixed(1)}/24.0 GB</Text>
                    </div>
                    <Progress percent={(currentMetrics.gpu_memory / 24) * 100} strokeColor="#52c41a" size="small" />
                  </div>
                  
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <Text>GPU 0 温度</Text>
                      <Text>{currentMetrics.gpu_temp.toFixed(1)}°C</Text>
                    </div>
                    <Progress percent={(currentMetrics.gpu_temp / 100) * 100} strokeColor="#faad14" size="small" />
                  </div>

                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <Text>系统内存</Text>
                      <Text>12.8/32.0 GB</Text>
                    </div>
                    <Progress percent={40} strokeColor="#722ed1" size="small" />
                  </div>
                </Card>
              </Col>
              
              <Col span={12}>
                <Card title="训练统计" size="small">
                  <Descriptions column={2} size="small">
                    <Descriptions.Item label="当前步数">{currentMetrics.step}/1500</Descriptions.Item>
                    <Descriptions.Item label="当前轮次">{currentMetrics.epoch}/5</Descriptions.Item>
                    <Descriptions.Item label="学习率">{currentMetrics.lr.toExponential(2)}</Descriptions.Item>
                    <Descriptions.Item label="批次大小">4</Descriptions.Item>
                    <Descriptions.Item label="梯度裁剪">1.0</Descriptions.Item>
                    <Descriptions.Item label="优化器">AdamW</Descriptions.Item>
                    <Descriptions.Item label="调度器">Cosine</Descriptions.Item>
                    <Descriptions.Item label="Warmup步数">150</Descriptions.Item>
                  </Descriptions>
                  
                  <div style={{ marginTop: 16, padding: '12px', backgroundColor: '#f6ffed', border: '1px solid #b7eb8f', borderRadius: 6 }}>
                    <div style={{ marginBottom: 8 }}>
                      <Text strong>训练进度</Text>
                    </div>
                    <Progress percent={currentMetrics.progress} strokeColor="#52c41a" />
                    <div style={{ marginTop: 8, display: 'flex', justifyContent: 'space-between' }}>
                      <Text type="secondary">预计剩余时间: 04:32:15</Text>
                      <Text type="secondary">完成时间: 2025-08-23 19:45</Text>
                    </div>
                  </div>
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="任务管理" key="jobs">
            <div style={{ marginBottom: 16 }}>
              <Space>
                <Button type="primary" icon={<PlayCircleOutlined />}>启动训练</Button>
                <Button icon={<PauseOutlined />}>暂停全部</Button>
                <Button icon={<ReloadOutlined />}>刷新状态</Button>
              </Space>
            </div>
            
            <Table 
              columns={jobColumns}
              dataSource={runningJobs}
              rowKey="id"
              size="small"
            />
            
            <Card title="队列统计" size="small" style={{ marginTop: 16 }}>
              <Row gutter={16}>
                <Col span={6}>
                  <Statistic title="运行中" value={1} valueStyle={{ color: '#52c41a' }} />
                </Col>
                <Col span={6}>
                  <Statistic title="等待中" value={1} valueStyle={{ color: '#faad14' }} />
                </Col>
                <Col span={6}>
                  <Statistic title="暂停" value={1} valueStyle={{ color: '#ff4d4f' }} />
                </Col>
                <Col span={6}>
                  <Statistic title="今日完成" value={3} valueStyle={{ color: '#1890ff' }} />
                </Col>
              </Row>
            </Card>
          </TabPane>

          <TabPane tab="告警中心" key="alerts">
            <Row gutter={16}>
              <Col span={16}>
                <Card title="实时告警" size="small" style={{ marginBottom: 16 }}>
                  <Timeline>
                    <Timeline.Item color="red" dot={<AlertOutlined />}>
                      <div>
                        <strong>15:45:23</strong> - GPU温度过高警告
                        <br />
                        <Text type="danger">GPU 0 温度达到82°C，超过设定阈值80°C</Text>
                        <br />
                        <Text type="secondary">任务: llama2-lora-chat-v1</Text>
                      </div>
                    </Timeline.Item>
                    
                    <Timeline.Item color="orange" dot={<WarningOutlined />}>
                      <div>
                        <strong>15:42:15</strong> - 内存使用率警告
                        <br />
                        <Text type="warning">显存使用率达到92%，接近上限</Text>
                        <br />
                        <Text type="secondary">任务: llama2-lora-chat-v1</Text>
                      </div>
                    </Timeline.Item>
                    
                    <Timeline.Item color="green" dot={<CheckCircleOutlined />}>
                      <div>
                        <strong>15:30:00</strong> - 训练检查点保存成功
                        <br />
                        <Text type="success">步数500检查点已保存到 /checkpoints/llama2_lora_500.pth</Text>
                        <br />
                        <Text type="secondary">任务: llama2-lora-chat-v1</Text>
                      </div>
                    </Timeline.Item>
                    
                    <Timeline.Item color="blue" dot={<CheckCircleOutlined />}>
                      <div>
                        <strong>14:30:00</strong> - 训练开始
                        <br />
                        <Text>LoRA微调任务正常启动</Text>
                        <br />
                        <Text type="secondary">任务: llama2-lora-chat-v1</Text>
                      </div>
                    </Timeline.Item>
                  </Timeline>
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="告警配置" size="small" style={{ marginBottom: 16 }}>
                  <div style={{ marginBottom: 16 }}>
                    <Text>GPU温度阈值</Text>
                    <Slider 
                      min={70} 
                      max={90} 
                      defaultValue={80}
                      marks={{ 70: '70°C', 80: '80°C', 90: '90°C' }}
                      tipFormatter={(value) => `${value}°C`}
                    />
                  </div>
                  
                  <div style={{ marginBottom: 16 }}>
                    <Text>显存使用阈值</Text>
                    <Slider 
                      min={70} 
                      max={95} 
                      defaultValue={90}
                      marks={{ 70: '70%', 90: '90%', 95: '95%' }}
                      tipFormatter={(value) => `${value}%`}
                    />
                  </div>
                  
                  <div style={{ marginBottom: 16 }}>
                    <Text>损失停滞检测</Text>
                    <Input.Group compact>
                      <Select defaultValue={100} style={{ width: '60%' }}>
                        <Option value={50}>50步</Option>
                        <Option value={100}>100步</Option>
                        <Option value={200}>200步</Option>
                      </Select>
                      <Input style={{ width: '40%' }} defaultValue="0.001" />
                    </Input.Group>
                  </div>
                  
                  <Button type="primary" block>保存配置</Button>
                </Card>
                
                <Card title="告警统计" size="small">
                  <Row gutter={8}>
                    <Col span={8}>
                      <Statistic
                        title="严重"
                        value={1}
                        valueStyle={{ color: '#ff4d4f' }}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="警告"
                        value={2}
                        valueStyle={{ color: '#faad14' }}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="信息"
                        value={5}
                        valueStyle={{ color: '#1890ff' }}
                      />
                    </Col>
                  </Row>
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="性能分析" key="performance">
            <Row gutter={16}>
              <Col span={12}>
                <Card title="训练效率分析" size="small" style={{ marginBottom: 16 }}>
                  <Alert
                    message="训练状态良好"
                    description="当前训练按预期进行，损失下降稳定，GPU利用率合理"
                    type="success"
                    showIcon
                    style={{ marginBottom: 16 }}
                  />
                  
                  <Descriptions bordered size="small">
                    <Descriptions.Item label="平均步时间">0.87秒</Descriptions.Item>
                    <Descriptions.Item label="吞吐量">138 tokens/s</Descriptions.Item>
                    <Descriptions.Item label="GPU效率">87.3%</Descriptions.Item>
                    <Descriptions.Item label="内存效率">67.5%</Descriptions.Item>
                    <Descriptions.Item label="收敛速度">优秀</Descriptions.Item>
                    <Descriptions.Item label="稳定性指数">0.92</Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
              
              <Col span={12}>
                <Card title="资源利用率历史" size="small" style={{ marginBottom: 16 }}>
                  <Area {...gpuConfig} height={200} />
                </Card>
              </Col>
            </Row>
            
            <Card title="性能优化建议" size="small">
              <Row gutter={16}>
                <Col span={8}>
                  <div style={{ padding: '16px', backgroundColor: '#f6ffed', border: '1px solid #b7eb8f', borderRadius: 6 }}>
                    <div style={{ color: '#52c41a', fontWeight: 'bold', marginBottom: 8 }}>
                      ✓ 运行良好
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 20, fontSize: '12px' }}>
                      <li>GPU利用率保持在85%以上</li>
                      <li>内存使用效率良好</li>
                      <li>训练损失稳定下降</li>
                      <li>无异常波动</li>
                    </ul>
                  </div>
                </Col>
                
                <Col span={8}>
                  <div style={{ padding: '16px', backgroundColor: '#fff7e6', border: '1px solid #ffd591', borderRadius: 6 }}>
                    <div style={{ color: '#faad14', fontWeight: 'bold', marginBottom: 8 }}>
                      ⚠ 注意监控
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 20, fontSize: '12px' }}>
                      <li>GPU温度偶尔接近阈值</li>
                      <li>显存使用率较高</li>
                      <li>建议适当调整批次大小</li>
                      <li>关注长期训练稳定性</li>
                    </ul>
                  </div>
                </Col>
                
                <Col span={8}>
                  <div style={{ padding: '16px', backgroundColor: '#e6f4ff', border: '1px solid #91caff', borderRadius: 6 }}>
                    <div style={{ color: '#1890ff', fontWeight: 'bold', marginBottom: 8 }}>
                      💡 优化建议
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 20, fontSize: '12px' }}>
                      <li>考虑启用混合精度训练</li>
                      <li>优化数据加载流水线</li>
                      <li>调整学习率调度策略</li>
                      <li>增加检查点保存频率</li>
                    </ul>
                  </div>
                </Col>
              </Row>
            </Card>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default TrainingMonitorDashboard;