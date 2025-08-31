import React, { useState, useEffect } from 'react';
import { 
  Card, 
  Row, 
  Col, 
  Statistic, 
  Table, 
  Alert, 
  Timeline, 
  Progress, 
  Badge,
  Select,
  DatePicker,
  Button,
  Space,
  Modal,
  Form,
  Input,
  Switch,
  Tabs,
  List,
  Avatar,
  Tag,
  Tooltip,
  Divider,
  message
} from 'antd';
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip as RechartsTooltip, 
  Legend, 
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import { 
  MonitorOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  SettingOutlined,
  ReloadOutlined,
  BellOutlined,
  TrendingUpOutlined,
  TrendingDownOutlined,
  DashboardOutlined,
  WarningOutlined
} from '@ant-design/icons';

const { TabPane } = Tabs;
const { Option } = Select;
const { RangePicker } = DatePicker;
const { TextArea } = Input;

interface MetricData {
  timestamp: string;
  latency: number;
  throughput: number;
  error_rate: number;
  cpu_usage: number;
  memory_usage: number;
  gpu_usage?: number;
  requests: number;
  accuracy?: number;
}

interface Alert {
  id: string;
  title: string;
  message: string;
  level: 'info' | 'warning' | 'error' | 'critical';
  timestamp: string;
  model_id?: string;
  deployment_id?: string;
  status: 'active' | 'resolved' | 'suppressed';
}

interface ModelPerformance {
  model_id: string;
  model_name: string;
  deployment_id: string;
  deployment_name: string;
  avg_latency: number;
  throughput: number;
  error_rate: number;
  uptime: number;
  last_request: string;
  health_score: number;
}

const ModelMonitoringPage: React.FC = () => {
  const [metricsData, setMetricsData] = useState<MetricData[]>([]);
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [modelPerformance, setModelPerformance] = useState<ModelPerformance[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [timeRange, setTimeRange] = useState<[Date, Date]>([
    new Date(Date.now() - 24 * 60 * 60 * 1000), // 1天前
    new Date()
  ]);
  const [alertModalVisible, setAlertModalVisible] = useState(false);
  const [alertForm] = Form.useForm();
  const [refreshInterval, setRefreshInterval] = useState<number>(30);
  const [autoRefresh, setAutoRefresh] = useState(true);

  const systemStats = {
    total_models: 12,
    active_deployments: 8,
    total_requests: 125680,
    avg_latency: 45.6,
    error_rate: 0.12,
    uptime: 99.97
  };

  useEffect(() => {
    fetchMetricsData();
    fetchAlerts();
    fetchModelPerformance();
  }, [selectedModel, timeRange]);

  useEffect(() => {
    if (autoRefresh) {
      const interval = setInterval(() => {
        fetchMetricsData();
        fetchAlerts();
        fetchModelPerformance();
      }, refreshInterval * 1000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh, refreshInterval, selectedModel, timeRange]);

  const fetchMetricsData = async () => {
    setLoading(true);
    try {
      // 模拟生成时间序列数据
      const data: MetricData[] = [];
      const now = new Date();
      for (let i = 23; i >= 0; i--) {
        const timeDate = new Date(now.getTime() - i * 60 * 60 * 1000);
        const timestamp = timeDate.toLocaleString('zh-CN', {
          year: 'numeric',
          month: '2-digit',
          day: '2-digit',
          hour: '2-digit',
          minute: '2-digit'
        }).replace(/\//g, '-');
        data.push({
          timestamp,
          latency: 30 + Math.random() * 40,
          throughput: 800 + Math.random() * 400,
          error_rate: Math.random() * 2,
          cpu_usage: 40 + Math.random() * 40,
          memory_usage: 50 + Math.random() * 30,
          gpu_usage: 30 + Math.random() * 50,
          requests: Math.floor(800 + Math.random() * 400),
          accuracy: 0.92 + Math.random() * 0.06
        });
      }
      setMetricsData(data);
    } catch (error) {
      message.error('获取监控数据失败');
    } finally {
      setLoading(false);
    }
  };

  const fetchAlerts = async () => {
    try {
      const mockAlerts: Alert[] = [
        {
          id: '1',
          title: '高延迟警告',
          message: 'text-classifier-prod 模型响应延迟超过阈值 (120ms)',
          level: 'warning',
          timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
          model_id: 'model_001',
          deployment_id: 'deploy_001',
          status: 'active'
        },
        {
          id: '2',
          title: '错误率异常',
          message: '情感分析模型错误率达到 5.2%，超过正常范围',
          level: 'error',
          timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
          model_id: 'model_002',
          deployment_id: 'deploy_002',
          status: 'active'
        },
        {
          id: '3',
          title: '资源使用警告',
          message: 'GPU内存使用率达到 95%',
          level: 'critical',
          timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
          status: 'active'
        },
        {
          id: '4',
          title: '服务恢复',
          message: '边缘检测模型服务已恢复正常',
          level: 'info',
          timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
          model_id: 'model_003',
          status: 'resolved'
        }
      ];
      setAlerts(mockAlerts);
    } catch (error) {
      message.error('获取告警信息失败');
    }
  };

  const fetchModelPerformance = async () => {
    try {
      const mockPerformance: ModelPerformance[] = [
        {
          model_id: 'model_001',
          model_name: 'BERT文本分类器',
          deployment_id: 'deploy_001',
          deployment_name: 'text-classifier-prod',
          avg_latency: 45.6,
          throughput: 1250,
          error_rate: 0.12,
          uptime: 99.97,
          last_request: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
          health_score: 98
        },
        {
          model_id: 'model_002',
          model_name: '情感分析模型',
          deployment_id: 'deploy_002',
          deployment_name: 'sentiment-analyzer-staging',
          avg_latency: 32.1,
          throughput: 890,
          error_rate: 5.2,
          uptime: 96.45,
          last_request: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
          health_score: 75
        },
        {
          model_id: 'model_003',
          model_name: '边缘检测模型',
          deployment_id: 'deploy_003',
          deployment_name: 'edge-detector-device',
          avg_latency: 28.3,
          throughput: 450,
          error_rate: 0.05,
          uptime: 99.9,
          last_request: new Date(Date.now() - 60 * 1000).toISOString(),
          health_score: 99
        }
      ];
      setModelPerformance(mockPerformance);
    } catch (error) {
      message.error('获取性能数据失败');
    }
  };

  const getAlertColor = (level: Alert['level']) => {
    const colors = {
      info: 'blue',
      warning: 'orange',
      error: 'red',
      critical: 'purple'
    };
    return colors[level];
  };

  const getAlertIcon = (level: Alert['level']) => {
    const icons = {
      info: <CheckCircleOutlined />,
      warning: <ExclamationCircleOutlined />,
      error: <CloseCircleOutlined />,
      critical: <WarningOutlined />
    };
    return icons[level];
  };

  const getHealthScoreColor = (score: number) => {
    if (score >= 90) return '#52c41a';
    if (score >= 80) return '#faad14';
    if (score >= 70) return '#fa8c16';
    return '#ff4d4f';
  };

  const performanceColumns = [
    {
      title: '模型',
      dataIndex: 'model_name',
      key: 'model_name',
      render: (text: string, record: ModelPerformance) => (
        <Space>
          <Avatar style={{ backgroundColor: '#1890ff' }}>
            {text.charAt(0)}
          </Avatar>
          <div>
            <div>{text}</div>
            <div style={{ fontSize: '12px', color: '#999' }}>{record.deployment_name}</div>
          </div>
        </Space>
      )
    },
    {
      title: '平均延迟',
      dataIndex: 'avg_latency',
      key: 'avg_latency',
      render: (latency: number) => `${latency.toFixed(1)}ms`,
      sorter: (a: ModelPerformance, b: ModelPerformance) => a.avg_latency - b.avg_latency,
    },
    {
      title: '吞吐量',
      dataIndex: 'throughput',
      key: 'throughput',
      render: (throughput: number) => `${throughput}/min`,
      sorter: (a: ModelPerformance, b: ModelPerformance) => a.throughput - b.throughput,
    },
    {
      title: '错误率',
      dataIndex: 'error_rate',
      key: 'error_rate',
      render: (rate: number) => (
        <span style={{ color: rate > 1 ? '#ff4d4f' : '#52c41a' }}>
          {rate.toFixed(2)}%
        </span>
      ),
      sorter: (a: ModelPerformance, b: ModelPerformance) => a.error_rate - b.error_rate,
    },
    {
      title: '可用性',
      dataIndex: 'uptime',
      key: 'uptime',
      render: (uptime: number) => `${uptime.toFixed(2)}%`,
      sorter: (a: ModelPerformance, b: ModelPerformance) => a.uptime - b.uptime,
    },
    {
      title: '健康评分',
      dataIndex: 'health_score',
      key: 'health_score',
      render: (score: number) => (
        <Progress 
          percent={score} 
          size="small" 
          strokeColor={getHealthScoreColor(score)}
          showInfo={false}
        />
      ),
      sorter: (a: ModelPerformance, b: ModelPerformance) => a.health_score - b.health_score,
    },
    {
      title: '最后请求',
      dataIndex: 'last_request',
      key: 'last_request',
      render: (time: string) => {
        const now = new Date();
        const timeDate = new Date(time);
        const diffMs = now.getTime() - timeDate.getTime();
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffDays = Math.floor(diffHours / 24);
        
        if (diffDays > 0) {
          return `${diffDays}天前`;
        } else if (diffHours > 0) {
          return `${diffHours}小时前`;
        } else {
          const diffMinutes = Math.floor(diffMs / (1000 * 60));
          return `${Math.max(1, diffMinutes)}分钟前`;
        }
      }
    }
  ];

  const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8'];

  const errorDistribution = [
    { name: '超时错误', value: 45, color: '#ff4d4f' },
    { name: '模型错误', value: 28, color: '#faad14' },
    { name: '网络错误', value: 18, color: '#52c41a' },
    { name: '其他错误', value: 9, color: '#1890ff' }
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* 控制面板 */}
      <Card style={{ marginBottom: '24px' }}>
        <Row gutter={16} align="middle">
          <Col span={6}>
            <Select 
              value={selectedModel} 
              onChange={setSelectedModel}
              style={{ width: '100%' }}
            >
              <Option value="all">所有模型</Option>
              <Option value="model_001">BERT文本分类器</Option>
              <Option value="model_002">情感分析模型</Option>
              <Option value="model_003">边缘检测模型</Option>
            </Select>
          </Col>
          <Col span={8}>
            <RangePicker
              value={timeRange}
              onChange={(dates) => setTimeRange(dates as [Date, Date])}
              showTime
              style={{ width: '100%' }}
            />
          </Col>
          <Col span={4}>
            <Space>
              <Switch 
                checked={autoRefresh} 
                onChange={setAutoRefresh}
                checkedChildren="自动刷新"
                unCheckedChildren="手动刷新"
              />
              <Button 
                icon={<ReloadOutlined />} 
                onClick={fetchMetricsData}
                loading={loading}
              />
            </Space>
          </Col>
          <Col span={6}>
            <Button 
              type="primary" 
              icon={<BellOutlined />}
              onClick={() => setAlertModalVisible(true)}
            >
              配置告警
            </Button>
          </Col>
        </Row>
      </Card>

      {/* 系统概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={4}>
          <Card>
            <Statistic
              title="模型总数"
              value={systemStats.total_models}
              prefix={<DashboardOutlined />}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="活跃部署"
              value={systemStats.active_deployments}
              prefix={<MonitorOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="总请求数"
              value={systemStats.total_requests}
              prefix={<TrendingUpOutlined />}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="平均延迟"
              value={systemStats.avg_latency}
              suffix="ms"
              precision={1}
              valueStyle={{ color: systemStats.avg_latency > 100 ? '#cf1322' : '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="错误率"
              value={systemStats.error_rate}
              suffix="%"
              precision={2}
              valueStyle={{ color: systemStats.error_rate > 1 ? '#cf1322' : '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={4}>
          <Card>
            <Statistic
              title="系统可用性"
              value={systemStats.uptime}
              suffix="%"
              precision={2}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 活跃告警 */}
      {alerts.filter(a => a.status === 'active').length > 0 && (
        <Alert
          message={`当前有 ${alerts.filter(a => a.status === 'active').length} 个活跃告警`}
          type="warning"
          showIcon
          style={{ marginBottom: '24px' }}
          action={
            <Button size="small" onClick={() => setAlertModalVisible(true)}>
              查看详情
            </Button>
          }
        />
      )}

      <Tabs defaultActiveKey="metrics">
        <TabPane tab="实时监控" key="metrics">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="响应时间 & 吞吐量" style={{ marginBottom: '16px' }}>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={metricsData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <RechartsTooltip />
                    <Legend />
                    <Line 
                      yAxisId="left"
                      type="monotone" 
                      dataKey="latency" 
                      stroke="#8884d8" 
                      name="延迟 (ms)"
                    />
                    <Line 
                      yAxisId="right"
                      type="monotone" 
                      dataKey="throughput" 
                      stroke="#82ca9d" 
                      name="吞吐量 (req/min)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="错误率" style={{ marginBottom: '16px' }}>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={metricsData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <RechartsTooltip />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="error_rate" 
                      stroke="#ff7300" 
                      fill="#ff7300" 
                      name="错误率 (%)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Card title="资源使用率">
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={metricsData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <RechartsTooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="cpu_usage" 
                      stroke="#1890ff" 
                      name="CPU使用率 (%)"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="memory_usage" 
                      stroke="#52c41a" 
                      name="内存使用率 (%)"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="gpu_usage" 
                      stroke="#fa8c16" 
                      name="GPU使用率 (%)"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="错误类型分布">
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={errorDistribution}
                      cx="50%"
                      cy="50%"
                      labelLine={false}
                      label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      outerRadius={80}
                      fill="#8884d8"
                      dataKey="value"
                    >
                      {errorDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <RechartsTooltip />
                  </PieChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>
        </TabPane>

        <TabPane tab="模型性能" key="performance">
          <Card title="模型性能对比">
            <Table
              columns={performanceColumns}
              dataSource={modelPerformance}
              rowKey="model_id"
              loading={loading}
            />
          </Card>
        </TabPane>

        <TabPane tab="告警管理" key="alerts">
          <Card title="告警历史">
            <List
              itemLayout="horizontal"
              dataSource={alerts}
              renderItem={alert => (
                <List.Item
                  actions={[
                    <Button 
                      type="link" 
                      size="small"
                      onClick={() => {
                        // 处理告警
                        message.success('告警已处理');
                      }}
                    >
                      {alert.status === 'active' ? '处理' : '已处理'}
                    </Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={
                      <Avatar 
                        style={{ backgroundColor: getAlertColor(alert.level) }}
                        icon={getAlertIcon(alert.level)}
                      />
                    }
                    title={
                      <Space>
                        <span>{alert.title}</span>
                        <Tag color={getAlertColor(alert.level)}>
                          {alert.level.toUpperCase()}
                        </Tag>
                        <Badge 
                          status={alert.status === 'active' ? 'error' : 'default'}
                          text={alert.status === 'active' ? '活跃' : '已解决'}
                        />
                      </Space>
                    }
                    description={
                      <div>
                        <div>{alert.message}</div>
                        <div style={{ fontSize: '12px', color: '#999', marginTop: '4px' }}>
                          {new Date(alert.timestamp).toLocaleString('zh-CN', {
                            year: 'numeric',
                            month: '2-digit',
                            day: '2-digit',
                            hour: '2-digit',
                            minute: '2-digit',
                            second: '2-digit'
                          })}
                        </div>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>

        <TabPane tab="性能分析" key="analysis">
          <Row gutter={16}>
            <Col span={24}>
              <Card title="模型准确率趋势" style={{ marginBottom: '16px' }}>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={metricsData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis domain={[0.8, 1.0]} />
                    <RechartsTooltip />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="accuracy" 
                      stroke="#52c41a" 
                      name="准确率"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Card title="请求量统计">
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={metricsData.slice(-12)}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="timestamp" />
                    <YAxis />
                    <RechartsTooltip />
                    <Bar dataKey="requests" fill="#1890ff" name="请求数" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="系统健康度">
                <div style={{ textAlign: 'center', padding: '40px' }}>
                  <Progress 
                    type="circle" 
                    percent={Math.round(systemStats.uptime)} 
                    width={160}
                    strokeColor={{
                      '0%': '#108ee9',
                      '100%': '#87d068',
                    }}
                    format={(percent) => `${percent}% 健康`}
                  />
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 告警配置模态框 */}
      <Modal
        title="告警规则配置"
        visible={alertModalVisible}
        onCancel={() => setAlertModalVisible(false)}
        footer={null}
        width={600}
      >
        <Form form={alertForm} layout="vertical">
          <Form.Item
            name="metric"
            label="监控指标"
            rules={[{ required: true, message: '请选择监控指标' }]}
          >
            <Select placeholder="选择监控指标">
              <Option value="latency">响应延迟</Option>
              <Option value="error_rate">错误率</Option>
              <Option value="throughput">吞吐量</Option>
              <Option value="cpu_usage">CPU使用率</Option>
              <Option value="memory_usage">内存使用率</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="condition"
            label="告警条件"
            rules={[{ required: true, message: '请选择告警条件' }]}
          >
            <Select placeholder="选择告警条件">
              <Option value="gt">大于</Option>
              <Option value="lt">小于</Option>
              <Option value="gte">大于等于</Option>
              <Option value="lte">小于等于</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="threshold"
            label="阈值"
            rules={[{ required: true, message: '请输入阈值' }]}
          >
            <Input placeholder="输入阈值" />
          </Form.Item>

          <Form.Item
            name="level"
            label="告警级别"
            rules={[{ required: true, message: '请选择告警级别' }]}
          >
            <Select placeholder="选择告警级别">
              <Option value="info">信息</Option>
              <Option value="warning">警告</Option>
              <Option value="error">错误</Option>
              <Option value="critical">严重</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="notification"
            label="通知方式"
          >
            <Select mode="multiple" placeholder="选择通知方式">
              <Option value="email">邮件</Option>
              <Option value="sms">短信</Option>
              <Option value="webhook">Webhook</Option>
              <Option value="slack">Slack</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
          >
            <TextArea rows={3} placeholder="输入告警描述" />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit">
                保存规则
              </Button>
              <Button onClick={() => setAlertModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ModelMonitoringPage;