/**
 * 反馈质量监控页面
 * 监控反馈质量、异常检测和数据完整性
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Space,
  Typography,
  Tag,
  Alert,
  Progress,
  Row,
  Col,
  Statistic,
  Select,
  Switch,
  Timeline,
  Tabs,
  Drawer,
  Descriptions,
  message
} from 'antd';
import {
  EyeOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  DashboardOutlined,
  BarChartOutlined,
  BugOutlined,
  MonitorOutlined,
  ThunderboltOutlined,
  ClockCircleOutlined
} from '@ant-design/icons';
import { Line, Column, Gauge, Liquid } from '@ant-design/plots';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

interface QualityMetrics {
  total_feedbacks: number;
  high_quality_count: number;
  medium_quality_count: number;
  low_quality_count: number;
  anomaly_count: number;
  spam_count: number;
  duplicate_count: number;
  processing_latency: number;
  error_rate: number;
  data_completeness: number;
}

interface QualityAlert {
  id: string;
  type: 'error' | 'warning' | 'info';
  title: string;
  description: string;
  timestamp: string;
  status: 'active' | 'resolved' | 'investigating';
  severity: 'high' | 'medium' | 'low';
}

interface QualityLog {
  id: string;
  feedback_event_id: string;
  user_id: string;
  item_id: string;
  consistency_score: number;
  temporal_validity_score: number;
  anomaly_score: number;
  trust_score: number;
  sentiment_confidence: number;
  final_quality_score: number;
  flags: string[];
  evaluated_at: string;
}

const FeedbackQualityMonitorPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [metrics, setMetrics] = useState<QualityMetrics | null>(null);
  const [alerts, setAlerts] = useState<QualityAlert[]>([]);
  const [qualityLogs, setQualityLogs] = useState<QualityLog[]>([]);
  const [trendData, setTrendData] = useState<any[]>([]);
  const [selectedLog, setSelectedLog] = useState<QualityLog | null>(null);
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [realTimeMode, setRealTimeMode] = useState(true);
  const [alertFilter, setAlertFilter] = useState<string>('all');

  useEffect(() => {
    loadQualityData();
    
    if (realTimeMode) {
      const interval = setInterval(loadQualityData, 10000);
      return () => clearInterval(interval);
    }
  }, [realTimeMode]);

  const loadQualityData = async () => {
    setLoading(true);
    try {
      // 模拟加载质量监控数据
      await new Promise(resolve => setTimeout(resolve, 800));
      
      // 生成质量指标
      const mockMetrics: QualityMetrics = {
        total_feedbacks: 15420,
        high_quality_count: 10280,
        medium_quality_count: 4320,
        low_quality_count: 820,
        anomaly_count: 45,
        spam_count: 23,
        duplicate_count: 67,
        processing_latency: 120,
        error_rate: 0.012,
        data_completeness: 0.987
      };
      setMetrics(mockMetrics);

      // 生成告警数据
      const mockAlerts: QualityAlert[] = [
        {
          id: 'alert-1',
          type: 'error',
          title: '异常反馈激增',
          description: '检测到可疑用户 user-789 在短时间内提交大量低质量反馈',
          timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
          status: 'active',
          severity: 'high'
        },
        {
          id: 'alert-2',
          type: 'warning',
          title: '处理延迟增加',
          description: '反馈处理平均延迟超过阈值，当前延迟 120ms',
          timestamp: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
          status: 'investigating',
          severity: 'medium'
        },
        {
          id: 'alert-3',
          type: 'info',
          title: '质量分数下降',
          description: '过去1小时平均质量分数下降了5%',
          timestamp: new Date(Date.now() - 75 * 60 * 1000).toISOString(),
          status: 'resolved',
          severity: 'low'
        }
      ];
      setAlerts(mockAlerts);

      // 生成质量日志
      const mockLogs: QualityLog[] = Array.from({ length: 100 }, (_, i) => ({
        id: `log-${i + 1}`,
        feedback_event_id: `event-${i + 1}`,
        user_id: `user-${Math.floor(Math.random() * 1000) + 1}`,
        item_id: `item-${Math.floor(Math.random() * 100) + 1}`,
        consistency_score: 0.3 + Math.random() * 0.7,
        temporal_validity_score: 0.4 + Math.random() * 0.6,
        anomaly_score: Math.random() * 0.3,
        trust_score: 0.5 + Math.random() * 0.5,
        sentiment_confidence: 0.6 + Math.random() * 0.4,
        final_quality_score: 0.2 + Math.random() * 0.8,
        flags: Math.random() > 0.7 ? ['duplicate', 'spam'] : Math.random() > 0.4 ? ['low_consistency'] : [],
        evaluated_at: new Date(Date.now() - Math.random() * 24 * 60 * 60 * 1000).toISOString()
      }));
      setQualityLogs(mockLogs);

      // 生成趋势数据
      const mockTrends = Array.from({ length: 24 }, (_, i) => ({
        hour: `${i}:00`,
        quality_score: 0.6 + Math.random() * 0.3,
        anomaly_rate: Math.random() * 0.05,
        processing_latency: 80 + Math.random() * 60,
        error_rate: Math.random() * 0.02
      }));
      setTrendData(mockTrends);

    } catch (error) {
      console.error('加载质量监控数据失败:', error);
      message.error('加载质量监控数据失败');
    } finally {
      setLoading(false);
    }
  };

  const handleViewLog = (log: QualityLog) => {
    setSelectedLog(log);
    setDrawerVisible(true);
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'error': return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      case 'warning': return <ExclamationCircleOutlined style={{ color: '#faad14' }} />;
      case 'info': return <CheckCircleOutlined style={{ color: '#1890ff' }} />;
      default: return <ExclamationCircleOutlined />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'red';
      case 'investigating': return 'orange';
      case 'resolved': return 'green';
      default: return 'default';
    }
  };

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'high': return 'red';
      case 'medium': return 'orange';
      case 'low': return 'blue';
      default: return 'default';
    }
  };

  const getQualityLevel = (score: number) => {
    if (score >= 0.7) return { level: '高质量', color: '#52c41a' };
    if (score >= 0.4) return { level: '中质量', color: '#faad14' };
    return { level: '低质量', color: '#ff4d4f' };
  };

  // 告警表格列
  const alertColumns: ColumnsType<QualityAlert> = [
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => getAlertIcon(type),
      width: 60,
    },
    {
      title: '告警信息',
      key: 'alert_info',
      render: (_, record: QualityAlert) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{record.title}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.description}
          </div>
        </div>
      ),
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: string) => (
        <Tag color={getSeverityColor(severity)}>
          {severity === 'high' ? '高' : severity === 'medium' ? '中' : '低'}
        </Tag>
      ),
      filters: [
        { text: '高', value: 'high' },
        { text: '中', value: 'medium' },
        { text: '低', value: 'low' },
      ],
      onFilter: (value, record) => record.severity === value,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {status === 'active' ? '活跃' :
           status === 'investigating' ? '调查中' : '已解决'}
        </Tag>
      ),
    },
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      render: (timestamp: string) => (
        <span style={{ fontSize: '12px' }}>
          {new Date(timestamp).toLocaleString()}
        </span>
      ),
    },
  ];

  // 质量日志表格列
  const logColumns: ColumnsType<QualityLog> = [
    {
      title: '反馈ID',
      dataIndex: 'feedback_event_id',
      key: 'feedback_event_id',
      render: (id: string) => (
        <Text code style={{ fontSize: '12px' }}>{id}</Text>
      ),
    },
    {
      title: '用户ID',
      dataIndex: 'user_id',
      key: 'user_id',
      render: (id: string) => (
        <Text code style={{ fontSize: '12px' }}>{id}</Text>
      ),
    },
    {
      title: '质量分数',
      dataIndex: 'final_quality_score',
      key: 'final_quality_score',
      render: (score: number) => {
        const { level, color } = getQualityLevel(score);
        return (
          <div>
            <Progress
              percent={score * 100}
              size="small"
              strokeColor={color}
              format={() => ''}
              style={{ width: '60px', marginBottom: '4px' }}
            />
            <div>
              <Tag color={color} style={{ fontSize: '10px' }}>
                {level}
              </Tag>
            </div>
          </div>
        );
      },
      sorter: (a, b) => a.final_quality_score - b.final_quality_score,
    },
    {
      title: '标记',
      dataIndex: 'flags',
      key: 'flags',
      render: (flags: string[]) => (
        <div>
          {flags.map(flag => (
            <Tag key={flag} color="red" size="small">
              {flag === 'duplicate' ? '重复' :
               flag === 'spam' ? '垃圾' :
               flag === 'low_consistency' ? '不一致' : flag}
            </Tag>
          ))}
        </div>
      ),
    },
    {
      title: '评估时间',
      dataIndex: 'evaluated_at',
      key: 'evaluated_at',
      render: (time: string) => (
        <span style={{ fontSize: '12px' }}>
          {new Date(time).toLocaleString()}
        </span>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: QualityLog) => (
        <Button
          type="link"
          size="small"
          icon={<EyeOutlined />}
          onClick={() => handleViewLog(record)}
        >
          详情
        </Button>
      ),
    },
  ];

  const filteredAlerts = alerts.filter(alert => 
    alertFilter === 'all' || alert.status === alertFilter
  );

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面头部 */}
      <div style={{ marginBottom: '24px' }}>
        <Space align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
          <div>
            <Title level={2} style={{ margin: 0 }}>
              <EyeOutlined style={{ color: '#52c41a', marginRight: '8px' }} />
              反馈质量监控
            </Title>
            <Text type="secondary">
              实时监控反馈质量、检测异常数据和系统健康状况
            </Text>
          </div>
          <Space>
            <Switch
              checked={realTimeMode}
              onChange={setRealTimeMode}
              checkedChildren="实时"
              unCheckedChildren="手动"
            />
            <Button type="primary" onClick={loadQualityData} loading={loading}>
              刷新数据
            </Button>
          </Space>
        </Space>
      </div>

      {/* 系统状态告警 */}
      {alerts.filter(a => a.status === 'active').length > 0 && (
        <Alert
          message="系统告警"
          description={`检测到 ${alerts.filter(a => a.status === 'active').length} 个活跃告警，请及时处理`}
          variant="warning"
          showIcon
          style={{ marginBottom: '24px' }}
          action={
            <Button size="small" onClick={() => console.log('处理告警')}>
              查看详情
            </Button>
          }
        />
      )}

      {/* 核心指标卡片 */}
      {metrics && (
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="总反馈数"
                value={metrics.total_feedbacks}
                prefix={<DashboardOutlined />}
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="异常数量"
                value={metrics.anomaly_count}
                prefix={<WarningOutlined />}
                valueStyle={{ color: '#ff4d4f' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="处理延迟"
                value={metrics.processing_latency}
                suffix="ms"
                prefix={<ClockCircleOutlined />}
                valueStyle={{ color: '#faad14' }}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card>
              <Statistic
                title="数据完整性"
                value={metrics.data_completeness * 100}
                precision={1}
                suffix="%"
                prefix={<CheckCircleOutlined />}
                valueStyle={{ color: '#1890ff' }}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* 质量分布 */}
      {metrics && (
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={8}>
            <Card title="质量分布">
              <div style={{ textAlign: 'center' }}>
                <Liquid
                  percent={metrics.high_quality_count / metrics.total_feedbacks}
                  outline={{
                    border: 4,
                    distance: 8,
                  }}
                  wave={{
                    length: 128,
                  }}
                  height={150}
                />
                <div style={{ marginTop: '8px' }}>高质量反馈占比</div>
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card title="错误率">
              <div style={{ textAlign: 'center' }}>
                <Gauge
                  percent={metrics.error_rate}
                  range={{ color: ['#30BF78', '#FAAD14', '#F4664A'] }}
                  indicator={{
                    pointer: { style: { stroke: '#D0D0D0' } },
                    pin: { style: { stroke: '#D0D0D0' } },
                  }}
                  statistic={{
                    content: {
                      style: {
                        fontSize: '14px',
                        lineHeight: '14px',
                      },
                    },
                  }}
                  height={150}
                />
                <div style={{ marginTop: '8px' }}>系统错误率</div>
              </div>
            </Card>
          </Col>
          <Col xs={24} sm={8}>
            <Card title="异常检测">
              <div style={{ textAlign: 'center' }}>
                <Gauge
                  percent={metrics.anomaly_count / metrics.total_feedbacks}
                  range={{ color: ['#30BF78', '#FAAD14', '#F4664A'] }}
                  indicator={{
                    pointer: { style: { stroke: '#D0D0D0' } },
                    pin: { style: { stroke: '#D0D0D0' } },
                  }}
                  statistic={{
                    content: {
                      style: {
                        fontSize: '14px',
                        lineHeight: '14px',
                      },
                    },
                  }}
                  height={150}
                />
                <div style={{ marginTop: '8px' }}>异常比例</div>
              </div>
            </Card>
          </Col>
        </Row>
      )}

      {/* 主要内容区域 */}
      <Tabs defaultActiveKey="alerts" size="large">
        <TabPane tab={<span><WarningOutlined />告警管理</span>} key="alerts">
          <Card
            title="系统告警"
            extra={
              <Space>
                <Select
                  value={alertFilter}
                  onChange={setAlertFilter}
                  style={{ width: 120 }}
                >
                  <Option value="all">全部告警</Option>
                  <Option value="active">活跃</Option>
                  <Option value="investigating">调查中</Option>
                  <Option value="resolved">已解决</Option>
                </Select>
                <WarningOutlined />
              </Space>
            }
          >
            <Table
              columns={alertColumns}
              dataSource={filteredAlerts}
              rowKey="id"
              pagination={false}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab={<span><MonitorOutlined />质量日志</span>} key="logs">
          <Card title="质量评估日志" extra={<MonitorOutlined />}>
            <Table
              columns={logColumns}
              dataSource={qualityLogs}
              rowKey="id"
              pagination={{
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
              }}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab={<span><BarChartOutlined />趋势分析</span>} key="trends">
          <Card title="质量趋势分析" extra={<BarChartOutlined />}>
            <Line
              data={trendData}
              xField="hour"
              yField="quality_score"
              smooth
              point={{ size: 3 }}
              height={300}
              meta={{
                hour: { alias: '时间' },
                quality_score: { alias: '质量分数' }
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab={<span><BugOutlined />异常检测</span>} key="anomalies">
          <Row gutter={[16, 16]}>
            <Col xs={24} lg={12}>
              <Card title="异常类型分布">
                <Column
                  data={[
                    { type: '重复反馈', count: metrics?.duplicate_count || 0 },
                    { type: '垃圾内容', count: metrics?.spam_count || 0 },
                    { type: '异常行为', count: (metrics?.anomaly_count || 0) - (metrics?.duplicate_count || 0) - (metrics?.spam_count || 0) }
                  ]}
                  xField="type"
                  yField="count"
                  height={250}
                  meta={{
                    type: { alias: '异常类型' },
                    count: { alias: '数量' }
                  }}
                />
              </Card>
            </Col>
            <Col xs={24} lg={12}>
              <Card title="处理时间线">
                <Timeline mode="left">
                  <Timeline.Item color="green">
                    <Text strong>15:30</Text> - 检测到异常用户行为
                  </Timeline.Item>
                  <Timeline.Item color="blue">
                    <Text strong>15:32</Text> - 自动标记可疑反馈
                  </Timeline.Item>
                  <Timeline.Item color="orange">
                    <Text strong>15:35</Text> - 生成质量告警
                  </Timeline.Item>
                  <Timeline.Item color="red">
                    <Text strong>15:38</Text> - 人工审核介入
                  </Timeline.Item>
                </Timeline>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 质量日志详情抽屉 */}
      <Drawer
        title="质量评估详情"
        width={600}
        visible={drawerVisible}
        onClose={() => setDrawerVisible(false)}
      >
        {selectedLog && (
          <div>
            <Card title="基本信息" size="small" style={{ marginBottom: '16px' }}>
              <Descriptions column={1} size="small">
                <Descriptions.Item label="反馈事件ID">
                  {selectedLog.feedback_event_id}
                </Descriptions.Item>
                <Descriptions.Item label="用户ID">
                  {selectedLog.user_id}
                </Descriptions.Item>
                <Descriptions.Item label="推荐项ID">
                  {selectedLog.item_id}
                </Descriptions.Item>
                <Descriptions.Item label="评估时间">
                  {new Date(selectedLog.evaluated_at).toLocaleString()}
                </Descriptions.Item>
              </Descriptions>
            </Card>

            <Card title="质量因子" size="small" style={{ marginBottom: '16px' }}>
              <Space direction="vertical" style={{ width: '100%' }}>
                <div>
                  <Text>一致性分数</Text>
                  <Progress
                    percent={selectedLog.consistency_score * 100}
                    strokeColor="#52c41a"
                    style={{ marginLeft: '16px' }}
                  />
                </div>
                <div>
                  <Text>时效性分数</Text>
                  <Progress
                    percent={selectedLog.temporal_validity_score * 100}
                    strokeColor="#1890ff"
                    style={{ marginLeft: '16px' }}
                  />
                </div>
                <div>
                  <Text>异常分数</Text>
                  <Progress
                    percent={selectedLog.anomaly_score * 100}
                    strokeColor="#ff4d4f"
                    style={{ marginLeft: '16px' }}
                  />
                </div>
                <div>
                  <Text>信任分数</Text>
                  <Progress
                    percent={selectedLog.trust_score * 100}
                    strokeColor="#722ed1"
                    style={{ marginLeft: '16px' }}
                  />
                </div>
                <div>
                  <Text>情感置信度</Text>
                  <Progress
                    percent={selectedLog.sentiment_confidence * 100}
                    strokeColor="#fa8c16"
                    style={{ marginLeft: '16px' }}
                  />
                </div>
              </Space>
            </Card>

            <Card title="最终评估" size="small">
              <div style={{ textAlign: 'center' }}>
                <div style={{ marginBottom: '16px' }}>
                  <Text strong style={{ fontSize: '16px' }}>
                    最终质量分数
                  </Text>
                </div>
                <Gauge
                  percent={selectedLog.final_quality_score}
                  range={{ color: ['#FF4D4F', '#FAAD14', '#52C41A'] }}
                  height={200}
                />
                <div style={{ marginTop: '16px' }}>
                  {selectedLog.flags.length > 0 && (
                    <div>
                      <Text strong>质量标记: </Text>
                      {selectedLog.flags.map(flag => (
                        <Tag key={flag} color="red">
                          {flag === 'duplicate' ? '重复' :
                           flag === 'spam' ? '垃圾' :
                           flag === 'low_consistency' ? '不一致' : flag}
                        </Tag>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </Card>
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default FeedbackQualityMonitorPage;