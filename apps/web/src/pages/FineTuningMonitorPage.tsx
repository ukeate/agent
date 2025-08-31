import React, { useState } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Typography, 
  Progress, 
  Statistic, 
  Table, 
  Button,
  Space,
  Tag,
  Alert,
  Select,
  Timeline,
  Descriptions,
  Tabs
} from 'antd';
import {
  MonitorOutlined,
  LineChartOutlined,
  AlertOutlined,
  FileTextOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  ClockCircleOutlined,
  CheckCircleOutlined
} from '@ant-design/icons';
import { Line, Area, Column } from '@ant-design/charts';

const { Title, Text } = Typography;
const { TabPane } = Tabs;

const FineTuningMonitorPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('overview');

  // 模拟监控数据
  const metricsData = [
    { step: 0, loss: 2.34, lr: 2e-4, gpu: 85, memory: 16.2 },
    { step: 100, loss: 1.89, lr: 1.98e-4, gpu: 87, memory: 16.5 },
    { step: 200, loss: 1.56, lr: 1.96e-4, gpu: 89, memory: 16.3 },
    { step: 300, loss: 1.34, lr: 1.94e-4, gpu: 86, memory: 16.4 },
    { step: 400, loss: 1.18, lr: 1.92e-4, gpu: 88, memory: 16.6 },
    { step: 500, loss: 0.98, lr: 1.90e-4, gpu: 90, memory: 16.2 },
  ];

  const lossConfig = {
    data: metricsData,
    xField: 'step',
    yField: 'loss',
    smooth: true,
    color: '#1890ff',
    point: {
      size: 3,
      shape: 'circle',
    },
    tooltip: {
      showMarkers: false,
    },
    meta: {
      loss: {
        alias: '训练损失',
      },
    },
  };

  const gpuConfig = {
    data: metricsData,
    xField: 'step',
    yField: 'gpu',
    smooth: true,
    color: '#52c41a',
    areaStyle: {
      fillOpacity: 0.3,
    },
    tooltip: {
      showMarkers: false,
    },
    meta: {
      gpu: {
        alias: 'GPU使用率 (%)',
      },
    },
  };

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <MonitorOutlined style={{ marginRight: 8, color: '#52c41a' }} />
          训练监控中心
        </Title>
        <Text type="secondary">
          实时监控微调训练进度、系统资源使用和性能指标
        </Text>
      </div>

      {/* 实时状态概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="实时损失"
              value={0.8743}
              precision={4}
              prefix={<LineChartOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="GPU使用率"
              value={87}
              suffix="%"
              prefix={<ThunderboltOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="显存使用"
              value="16.2 / 24.0"
              suffix="GB"
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="训练进度"
              value={67.8}
              suffix="%"
              prefix={<ClockCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="监控概览" key="overview">
            <Row gutter={16}>
              <Col span={12}>
                <Card title="训练损失趋势" size="small" style={{ height: 350 }}>
                  <Line {...lossConfig} height={280} />
                </Card>
              </Col>
              <Col span={12}>
                <Card title="GPU使用率" size="small" style={{ height: 350 }}>
                  <Area {...gpuConfig} height={280} />
                </Card>
              </Col>
            </Row>

            <Row gutter={16} style={{ marginTop: 16 }}>
              <Col span={8}>
                <Card title="训练统计" size="small">
                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="当前步数">567 / 1500</Descriptions.Item>
                    <Descriptions.Item label="当前轮次">2 / 5</Descriptions.Item>
                    <Descriptions.Item label="学习率">1.89e-4</Descriptions.Item>
                    <Descriptions.Item label="批次大小">4</Descriptions.Item>
                    <Descriptions.Item label="梯度裁剪">1.0</Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
              <Col span={8}>
                <Card title="系统资源" size="small">
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <Text>GPU温度</Text>
                      <Text>73°C</Text>
                    </div>
                    <Progress percent={73} size="small" strokeColor="#faad14" />
                  </div>
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <Text>风扇转速</Text>
                      <Text>2847 RPM</Text>
                    </div>
                    <Progress percent={68} size="small" strokeColor="#1890ff" />
                  </div>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <Text>功耗</Text>
                      <Text>287W / 350W</Text>
                    </div>
                    <Progress percent={82} size="small" strokeColor="#52c41a" />
                  </div>
                </Card>
              </Col>
              <Col span={8}>
                <Card title="预测信息" size="small">
                  <Descriptions column={1} size="small">
                    <Descriptions.Item label="预计完成时间">
                      2025-08-23 19:45:00
                    </Descriptions.Item>
                    <Descriptions.Item label="剩余时间">
                      04:32:15
                    </Descriptions.Item>
                    <Descriptions.Item label="预计最终损失">
                      0.65 ± 0.08
                    </Descriptions.Item>
                    <Descriptions.Item label="收敛概率">
                      92.4%
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="异常检测" key="anomaly">
            <Row gutter={16}>
              <Col span={16}>
                <Card title="异常事件时间线" size="small">
                  <Timeline>
                    <Timeline.Item color="green" dot={<CheckCircleOutlined />}>
                      <div>
                        <strong>14:30:00</strong> - 训练正常开始
                        <br />
                        <Text type="secondary">所有系统检查通过</Text>
                      </div>
                    </Timeline.Item>
                    <Timeline.Item color="orange" dot={<AlertOutlined />}>
                      <div>
                        <strong>14:45:23</strong> - GPU温度预警
                        <br />
                        <Text type="warning">温度达到78°C，已自动调整风扇转速</Text>
                      </div>
                    </Timeline.Item>
                    <Timeline.Item color="red" dot={<AlertOutlined />}>
                      <div>
                        <strong>15:12:45</strong> - 内存使用峰值
                        <br />
                        <Text type="danger">显存使用达到22.8GB，接近上限</Text>
                      </div>
                    </Timeline.Item>
                    <Timeline.Item color="green" dot={<CheckCircleOutlined />}>
                      <div>
                        <strong>15:30:12</strong> - 系统状态恢复正常
                        <br />
                        <Text type="success">内存清理完成，训练继续</Text>
                      </div>
                    </Timeline.Item>
                  </Timeline>
                </Card>
              </Col>
              <Col span={8}>
                <Card title="告警配置" size="small" style={{ marginBottom: 16 }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div>
                      <Text>GPU温度阈值</Text>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Text type="secondary">75°C</Text>
                        <Tag color="orange">警告</Tag>
                      </div>
                    </div>
                    <div>
                      <Text>显存使用阈值</Text>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Text type="secondary">90%</Text>
                        <Tag color="red">严重</Tag>
                      </div>
                    </div>
                    <div>
                      <Text>损失停滞检测</Text>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Text type="secondary">100步</Text>
                        <Tag color="yellow">提示</Tag>
                      </div>
                    </div>
                  </Space>
                </Card>

                <Card title="告警统计" size="small">
                  <Row gutter={8}>
                    <Col span={8}>
                      <Statistic
                        title="警告"
                        value={3}
                        valueStyle={{ color: '#faad14' }}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="错误"
                        value={1}
                        valueStyle={{ color: '#ff4d4f' }}
                      />
                    </Col>
                    <Col span={8}>
                      <Statistic
                        title="恢复"
                        value={4}
                        valueStyle={{ color: '#52c41a' }}
                      />
                    </Col>
                  </Row>
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="性能报告" key="performance">
            <Row gutter={16}>
              <Col span={12}>
                <Card title="训练效率分析" size="small">
                  <div style={{ marginBottom: 16 }}>
                    <Alert
                      message="训练效率评估"
                      description="基于当前配置，训练效率良好，预计按时完成"
                      type="success"
                      showIcon
                    />
                  </div>
                  <Descriptions column={2} bordered size="small">
                    <Descriptions.Item label="平均步时间">0.85秒</Descriptions.Item>
                    <Descriptions.Item label="吞吐量">142 tokens/s</Descriptions.Item>
                    <Descriptions.Item label="GPU效率">87.3%</Descriptions.Item>
                    <Descriptions.Item label="内存效率">67.5%</Descriptions.Item>
                    <Descriptions.Item label="收敛速度">优秀</Descriptions.Item>
                    <Descriptions.Item label="稳定性">良好</Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
              <Col span={12}>
                <Card title="资源利用统计" size="small">
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <Text>GPU利用率</Text>
                      <Text strong>87.3%</Text>
                    </div>
                    <Progress percent={87.3} strokeColor="#1890ff" size="small" />
                  </div>
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <Text>显存利用率</Text>
                      <Text strong>67.5%</Text>
                    </div>
                    <Progress percent={67.5} strokeColor="#52c41a" size="small" />
                  </div>
                  <div style={{ marginBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <Text>CPU利用率</Text>
                      <Text strong>23.8%</Text>
                    </div>
                    <Progress percent={23.8} strokeColor="#faad14" size="small" />
                  </div>
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <Text>网络I/O</Text>
                      <Text strong>12.4%</Text>
                    </div>
                    <Progress percent={12.4} strokeColor="#722ed1" size="small" />
                  </div>
                </Card>
              </Col>
            </Row>

            <Card title="性能建议" size="small" style={{ marginTop: 16 }}>
              <Row gutter={16}>
                <Col span={8}>
                  <div style={{ padding: '16px', border: '1px dashed #52c41a', borderRadius: '6px' }}>
                    <div style={{ color: '#52c41a', marginBottom: 8, fontWeight: 'bold' }}>
                      ✓ 运行良好
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 20, fontSize: '12px' }}>
                      <li>GPU利用率在正常范围</li>
                      <li>内存使用效率高</li>
                      <li>收敛趋势稳定</li>
                    </ul>
                  </div>
                </Col>
                <Col span={8}>
                  <div style={{ padding: '16px', border: '1px dashed #faad14', borderRadius: '6px' }}>
                    <div style={{ color: '#faad14', marginBottom: 8, fontWeight: 'bold' }}>
                      ⚠ 注意事项
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 20, fontSize: '12px' }}>
                      <li>GPU温度偶有升高</li>
                      <li>内存峰值接近上限</li>
                      <li>建议降低批次大小</li>
                    </ul>
                  </div>
                </Col>
                <Col span={8}>
                  <div style={{ padding: '16px', border: '1px dashed #1890ff', borderRadius: '6px' }}>
                    <div style={{ color: '#1890ff', marginBottom: 8, fontWeight: 'bold' }}>
                      💡 优化建议
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 20, fontSize: '12px' }}>
                      <li>可适当增加学习率</li>
                      <li>考虑启用梯度累积</li>
                      <li>优化数据加载流水线</li>
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

export default FineTuningMonitorPage;