import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Table,
  Tag,
  Alert,
  Input,
  Form,
  Modal,
  Select,
  Switch,
  Tooltip,
  Statistic,
  Timeline,
  List,
  Avatar,
  Tabs,
  Badge,
  Progress,
  message,
  Slider,
  InputNumber,
  Divider,
  Descriptions,
} from 'antd';
import {
  MonitorOutlined,
  SettingOutlined,
  PlusOutlined,
  BellOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  FireOutlined,
  SafetyCertificateOutlined,
  ThunderboltOutlined,
  EyeOutlined,
  EditOutlined,
  DeleteOutlined,
  MutedOutlined,
  SoundOutlined,
  ClockCircleOutlined,
  LineChartOutlined,
  BarChartOutlined,
  DashboardOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;
const { Option } = Select;
const { TabPane } = Tabs;
const { Search } = Input;

interface MonitoringAlert {
  id: string;
  name: string;
  description: string;
  experiment_id: string;
  experiment_name: string;
  alert_type: 'threshold' | 'anomaly' | 'trend' | 'custom';
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'active' | 'muted' | 'resolved' | 'acknowledged';
  metric: string;
  condition: {
    operator: 'greater_than' | 'less_than' | 'equal' | 'not_equal';
    threshold: number;
    duration: number; // minutes
  };
  current_value: number;
  trigger_count: number;
  last_triggered: string;
  notification_channels: ('email' | 'slack' | 'webhook' | 'sms')[];
  created_at: string;
  updated_at: string;
  created_by: string;
}

interface AlertInstance {
  id: string;
  alert_id: string;
  alert_name: string;
  triggered_at: string;
  resolved_at?: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  status: 'active' | 'acknowledged' | 'resolved' | 'false_positive';
  metric_value: number;
  threshold_value: number;
  message: string;
  assignee?: string;
  notes?: string;
}

interface SystemMetric {
  name: string;
  current_value: number;
  threshold: number;
  status: 'normal' | 'warning' | 'critical';
  trend: 'up' | 'down' | 'stable';
  change_percentage: number;
  unit: string;
}

const MonitoringAlertsPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [selectedTab, setSelectedTab] = useState<string>('alerts');
  const [alertModalVisible, setAlertModalVisible] = useState(false);
  const [instanceModalVisible, setInstanceModalVisible] = useState(false);
  const [selectedAlert, setSelectedAlert] = useState<string>('');
  const [searchText, setSearchText] = useState('');
  const [severityFilter, setSeverityFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [form] = Form.useForm();

  // 模拟监控告警数据
  const monitoringAlerts: MonitoringAlert[] = [
    {
      id: 'alert_001',
      name: '转化率异常下降',
      description: '当转化率连续10分钟低于13%时触发告警',
      experiment_id: 'exp_001',
      experiment_name: '首页改版A/B测试',
      alert_type: 'threshold',
      severity: 'high',
      status: 'active',
      metric: 'conversion_rate',
      condition: {
        operator: 'less_than',
        threshold: 13.0,
        duration: 10,
      },
      current_value: 12.3,
      trigger_count: 3,
      last_triggered: '2024-01-22 14:25:30',
      notification_channels: ['email', 'slack'],
      created_at: '2024-01-20',
      updated_at: '2024-01-22',
      created_by: 'Product Team',
    },
    {
      id: 'alert_002',
      name: '错误率超标',
      description: '当系统错误率超过1%时立即告警',
      experiment_id: 'exp_002',
      experiment_name: '结算页面优化',
      alert_type: 'threshold',
      severity: 'critical',
      status: 'resolved',
      metric: 'error_rate',
      condition: {
        operator: 'greater_than',
        threshold: 1.0,
        duration: 5,
      },
      current_value: 0.3,
      trigger_count: 1,
      last_triggered: '2024-01-21 09:15:45',
      notification_channels: ['email', 'slack', 'sms'],
      created_at: '2024-01-18',
      updated_at: '2024-01-21',
      created_by: 'DevOps Team',
    },
    {
      id: 'alert_003',
      name: '响应时间异常',
      description: '当平均响应时间超过300ms时触发',
      experiment_id: 'exp_003',
      experiment_name: '推荐算法测试',
      alert_type: 'anomaly',
      severity: 'medium',
      status: 'acknowledged',
      metric: 'response_time',
      condition: {
        operator: 'greater_than',
        threshold: 300,
        duration: 15,
      },
      current_value: 285,
      trigger_count: 5,
      last_triggered: '2024-01-22 11:30:12',
      notification_channels: ['email'],
      created_at: '2024-01-19',
      updated_at: '2024-01-22',
      created_by: 'Tech Team',
    },
    {
      id: 'alert_004',
      name: '用户参与度趋势下降',
      description: '当用户参与度连续下降超过48小时时告警',
      experiment_id: 'exp_001',
      experiment_name: '首页改版A/B测试',
      alert_type: 'trend',
      severity: 'low',
      status: 'muted',
      metric: 'engagement_rate',
      condition: {
        operator: 'less_than',
        threshold: 75,
        duration: 2880, // 48小时
      },
      current_value: 78.5,
      trigger_count: 0,
      last_triggered: '',
      notification_channels: ['email'],
      created_at: '2024-01-21',
      updated_at: '2024-01-22',
      created_by: 'UX Team',
    },
  ];

  const alertInstances: AlertInstance[] = [
    {
      id: 'instance_001',
      alert_id: 'alert_001',
      alert_name: '转化率异常下降',
      triggered_at: '2024-01-22 14:25:30',
      severity: 'high',
      status: 'active',
      metric_value: 12.3,
      threshold_value: 13.0,
      message: '转化率已连续10分钟低于阈值，当前值：12.3%，阈值：13.0%',
      assignee: 'Product Manager',
    },
    {
      id: 'instance_002',
      alert_id: 'alert_002',
      alert_name: '错误率超标',
      triggered_at: '2024-01-21 09:15:45',
      resolved_at: '2024-01-21 09:45:30',
      severity: 'critical',
      status: 'resolved',
      metric_value: 2.1,
      threshold_value: 1.0,
      message: '系统错误率达到2.1%，超过临界阈值1.0%',
      assignee: 'DevOps Lead',
      notes: '已修复API连接问题，错误率恢复正常',
    },
    {
      id: 'instance_003',
      alert_id: 'alert_003',
      alert_name: '响应时间异常',
      triggered_at: '2024-01-22 11:30:12',
      severity: 'medium',
      status: 'acknowledged',
      metric_value: 340,
      threshold_value: 300,
      message: '平均响应时间达到340ms，超过设定阈值300ms',
      assignee: 'Backend Developer',
    },
  ];

  const systemMetrics: SystemMetric[] = [
    {
      name: '平均转化率',
      current_value: 14.2,
      threshold: 13.0,
      status: 'normal',
      trend: 'up',
      change_percentage: 2.3,
      unit: '%',
    },
    {
      name: '系统错误率',
      current_value: 0.3,
      threshold: 1.0,
      status: 'normal',
      trend: 'down',
      change_percentage: -15.2,
      unit: '%',
    },
    {
      name: '平均响应时间',
      current_value: 285,
      threshold: 300,
      status: 'warning',
      trend: 'up',
      change_percentage: 8.5,
      unit: 'ms',
    },
    {
      name: '用户活跃度',
      current_value: 78.5,
      threshold: 75.0,
      status: 'normal',
      trend: 'stable',
      change_percentage: 0.8,
      unit: '%',
    },
    {
      name: '流量QPS',
      current_value: 1250,
      threshold: 1500,
      status: 'normal',
      trend: 'up',
      change_percentage: 12.4,
      unit: 'req/s',
    },
    {
      name: '内存使用率',
      current_value: 82,
      threshold: 85,
      status: 'warning',
      trend: 'up',
      change_percentage: 5.2,
      unit: '%',
    },
  ];

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'low': return '#52c41a';
      case 'medium': return '#faad14';
      case 'high': return '#ff7a45';
      case 'critical': return '#ff4d4f';
      default: return '#d9d9d9';
    }
  };

  const getSeverityText = (severity: string) => {
    switch (severity) {
      case 'low': return '低';
      case 'medium': return '中';
      case 'high': return '高';
      case 'critical': return '严重';
      default: return severity;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'error';
      case 'muted': return 'warning';
      case 'resolved': return 'success';
      case 'acknowledged': return 'processing';
      default: return 'default';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'active': return '活跃';
      case 'muted': return '静音';
      case 'resolved': return '已解决';
      case 'acknowledged': return '已确认';
      default: return status;
    }
  };

  const getMetricStatusColor = (status: string) => {
    switch (status) {
      case 'normal': return '#52c41a';
      case 'warning': return '#faad14';
      case 'critical': return '#ff4d4f';
      default: return '#d9d9d9';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'up': return <span style={{ color: '#ff4d4f' }}>↗</span>;
      case 'down': return <span style={{ color: '#52c41a' }}>↘</span>;
      case 'stable': return <span style={{ color: '#1890ff' }}>→</span>;
      default: return null;
    }
  };

  const alertsColumns: ColumnsType<MonitoringAlert> = [
    {
      title: '告警名称',
      key: 'name',
      width: 200,
      render: (_, record: MonitoringAlert) => (
        <div>
          <Text strong>{record.name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.description}
          </Text>
        </div>
      ),
    },
    {
      title: '实验',
      dataIndex: 'experiment_name',
      key: 'experiment_name',
      width: 150,
      render: (name: string, record: MonitoringAlert) => (
        <div>
          <Text>{name}</Text>
          <br />
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.experiment_id}
          </Text>
        </div>
      ),
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      width: 100,
      render: (severity: string) => (
        <Tag color={getSeverityColor(severity)}>
          {getSeverityText(severity)}
        </Tag>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status: string) => (
        <Badge status={getStatusColor(status)} text={getStatusText(status)} />
      ),
    },
    {
      title: '指标/条件',
      key: 'condition',
      width: 150,
      render: (_, record: MonitoringAlert) => (
        <div>
          <Text strong>{record.metric}</Text>
          <br />
          <Text style={{ fontSize: '12px' }}>
            {record.condition.operator === 'greater_than' ? '>' :
             record.condition.operator === 'less_than' ? '<' :
             record.condition.operator === 'equal' ? '=' : '≠'} {record.condition.threshold}
          </Text>
        </div>
      ),
    },
    {
      title: '当前值',
      dataIndex: 'current_value',
      key: 'current_value',
      width: 80,
      render: (value: number, record: MonitoringAlert) => (
        <Text 
          strong 
          style={{ 
            color: record.status === 'active' ? '#ff4d4f' : '#52c41a',
            fontSize: '14px'
          }}
        >
          {value}
        </Text>
      ),
    },
    {
      title: '触发次数',
      dataIndex: 'trigger_count',
      key: 'trigger_count',
      width: 80,
      align: 'center',
      render: (count: number) => (
        <Badge count={count} showZero />
      ),
    },
    {
      title: '通知渠道',
      dataIndex: 'notification_channels',
      key: 'notification_channels',
      width: 120,
      render: (channels: string[]) => (
        <div>
          {channels.map(channel => (
            <Tag key={channel} size="small" style={{ marginBottom: '2px' }}>
              {channel === 'email' ? '邮件' :
               channel === 'slack' ? 'Slack' :
               channel === 'webhook' ? 'Webhook' :
               channel === 'sms' ? '短信' : channel}
            </Tag>
          ))}
        </div>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (_, record: MonitoringAlert) => (
        <Space size="small">
          <Tooltip title="查看详情">
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedAlert(record.id);
                setInstanceModalVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button type="text" size="small" icon={<EditOutlined />} />
          </Tooltip>
          <Tooltip title={record.status === 'muted' ? '取消静音' : '静音'}>
            <Button 
              type="text" 
              size="small" 
              icon={record.status === 'muted' ? <SoundOutlined /> : <MutedOutlined />}
              onClick={() => message.success(record.status === 'muted' ? '已取消静音' : '已静音')}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const instancesColumns: ColumnsType<AlertInstance> = [
    {
      title: '触发时间',
      dataIndex: 'triggered_at',
      key: 'triggered_at',
      width: 120,
      render: (time: string) => (
        <Text style={{ fontSize: '12px' }}>{time}</Text>
      ),
    },
    {
      title: '告警名称',
      dataIndex: 'alert_name',
      key: 'alert_name',
      width: 180,
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      width: 80,
      render: (severity: string) => (
        <Tag color={getSeverityColor(severity)} size="small">
          {getSeverityText(severity)}
        </Tag>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status: string) => (
        <Badge 
          status={getStatusColor(status)} 
          text={
            status === 'active' ? '活跃' :
            status === 'acknowledged' ? '已确认' :
            status === 'resolved' ? '已解决' : '误报'
          }
        />
      ),
    },
    {
      title: '指标值',
      key: 'metric_values',
      width: 120,
      render: (_, record: AlertInstance) => (
        <div>
          <Text>当前: <Text strong style={{ color: '#ff4d4f' }}>{record.metric_value}</Text></Text>
          <br />
          <Text>阈值: <Text type="secondary">{record.threshold_value}</Text></Text>
        </div>
      ),
    },
    {
      title: '负责人',
      dataIndex: 'assignee',
      key: 'assignee',
      width: 100,
      render: (assignee: string) => assignee ? <Text>{assignee}</Text> : <Text type="secondary">未分配</Text>,
    },
    {
      title: '持续时间',
      key: 'duration',
      width: 100,
      render: (_, record: AlertInstance) => {
        if (record.resolved_at) {
          const duration = new Date(record.resolved_at).getTime() - new Date(record.triggered_at).getTime();
          const minutes = Math.floor(duration / (1000 * 60));
          return <Text>{minutes}分钟</Text>;
        }
        const duration = new Date().getTime() - new Date(record.triggered_at).getTime();
        const minutes = Math.floor(duration / (1000 * 60));
        return <Text style={{ color: '#ff4d4f' }}>{minutes}分钟</Text>;
      },
    },
  ];

  const filteredAlerts = monitoringAlerts.filter(alert => {
    const matchesSearch = alert.name.toLowerCase().includes(searchText.toLowerCase()) ||
                         alert.description.toLowerCase().includes(searchText.toLowerCase());
    const matchesSeverity = severityFilter === 'all' || alert.severity === severityFilter;
    const matchesStatus = statusFilter === 'all' || alert.status === statusFilter;
    return matchesSearch && matchesSeverity && matchesStatus;
  });

  const handleCreateAlert = () => {
    setAlertModalVisible(true);
  };

  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
    }, 500);
  }, [selectedTab]);

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Title level={2} style={{ margin: 0 }}>
              <MonitorOutlined /> 监控告警
            </Title>
            <Text type="secondary">实时监控实验指标并智能告警</Text>
          </div>
          <Space>
            <Button icon={<DashboardOutlined />}>
              监控面板
            </Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={handleCreateAlert}>
              创建告警
            </Button>
          </Space>
        </div>
      </div>

      {/* 告警统计概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃告警"
              value={monitoringAlerts.filter(a => a.status === 'active').length}
              valueStyle={{ color: '#ff4d4f' }}
              prefix={<FireOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="今日触发"
              value={24}
              prefix={<BellOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均响应时间"
              value={15.2}
              precision={1}
              suffix="分钟"
              prefix={<ClockCircleOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="告警覆盖率"
              value={87.5}
              precision={1}
              suffix="%"
              prefix={<SafetyCertificateOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 主要内容 */}
      <Tabs activeKey={selectedTab} onChange={setSelectedTab}>
        <TabPane tab="告警规则" key="alerts">
          {/* 筛选工具栏 */}
          <Card style={{ marginBottom: '16px' }}>
            <Row gutter={16} align="middle">
              <Col>
                <Search
                  placeholder="搜索告警名称"
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                  style={{ width: 200 }}
                  allowClear
                />
              </Col>
              <Col>
                <Select
                  value={severityFilter}
                  onChange={setSeverityFilter}
                  style={{ width: 120 }}
                >
                  <Option value="all">全部严重程度</Option>
                  <Option value="low">低</Option>
                  <Option value="medium">中</Option>
                  <Option value="high">高</Option>
                  <Option value="critical">严重</Option>
                </Select>
              </Col>
              <Col>
                <Select
                  value={statusFilter}
                  onChange={setStatusFilter}
                  style={{ width: 100 }}
                >
                  <Option value="all">全部状态</Option>
                  <Option value="active">活跃</Option>
                  <Option value="muted">静音</Option>
                  <Option value="resolved">已解决</Option>
                </Select>
              </Col>
            </Row>
          </Card>

          {/* 告警规则列表 */}
          <Card>
            <Table
              columns={alertsColumns}
              dataSource={filteredAlerts}
              rowKey="id"
              loading={loading}
              pagination={{
                total: filteredAlerts.length,
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) =>
                  `显示 ${range[0]}-${range[1]} 条，共 ${total} 条`,
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="告警实例" key="instances">
          <Card>
            <Table
              columns={instancesColumns}
              dataSource={alertInstances}
              rowKey="id"
              loading={loading}
              size="small"
              pagination={{
                pageSize: 15,
                showSizeChanger: true,
              }}
              expandable={{
                expandedRowRender: (record) => (
                  <div style={{ margin: 0 }}>
                    <Descriptions size="small" column={2}>
                      <Descriptions.Item label="告警消息">
                        {record.message}
                      </Descriptions.Item>
                      <Descriptions.Item label="处理备注">
                        {record.notes || '暂无备注'}
                      </Descriptions.Item>
                    </Descriptions>
                    <div style={{ marginTop: '12px' }}>
                      <Button size="small" type="primary" style={{ marginRight: '8px' }}>
                        确认告警
                      </Button>
                      <Button size="small" style={{ marginRight: '8px' }}>
                        分配给我
                      </Button>
                      <Button size="small">
                        标记为误报
                      </Button>
                    </div>
                  </div>
                ),
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="系统指标" key="metrics">
          <Row gutter={16}>
            {systemMetrics.map((metric, index) => (
              <Col span={8} key={index} style={{ marginBottom: '16px' }}>
                <Card size="small">
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                      <Text strong>{metric.name}</Text>
                      <div style={{ marginTop: '4px' }}>
                        <Text style={{ fontSize: '24px', fontWeight: 'bold' }}>
                          {metric.current_value}
                        </Text>
                        <Text style={{ marginLeft: '4px' }}>{metric.unit}</Text>
                      </div>
                      <div style={{ marginTop: '4px', display: 'flex', alignItems: 'center' }}>
                        {getTrendIcon(metric.trend)}
                        <Text 
                          style={{ 
                            marginLeft: '4px',
                            color: metric.change_percentage > 0 ? '#ff4d4f' : '#52c41a'
                          }}
                        >
                          {metric.change_percentage > 0 ? '+' : ''}{metric.change_percentage}%
                        </Text>
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <div style={{
                        width: '12px',
                        height: '12px',
                        borderRadius: '50%',
                        backgroundColor: getMetricStatusColor(metric.status),
                        marginBottom: '4px',
                        marginLeft: 'auto',
                      }}></div>
                      <Text style={{ fontSize: '12px' }}>
                        阈值: {metric.threshold}
                      </Text>
                    </div>
                  </div>
                  <div style={{ marginTop: '8px' }}>
                    <Progress
                      percent={Math.min((metric.current_value / metric.threshold) * 100, 100)}
                      size="small"
                      strokeColor={getMetricStatusColor(metric.status)}
                      showInfo={false}
                    />
                  </div>
                </Card>
              </Col>
            ))}
          </Row>
        </TabPane>

        <TabPane tab="告警历史" key="history">
          <Card>
            <Timeline>
              <Timeline.Item color="red" dot={<WarningOutlined />}>
                <div>
                  <Text strong>转化率异常下降</Text>
                  <Tag color="red" size="small" style={{ marginLeft: '8px' }}>严重</Tag>
                  <br />
                  <Text type="secondary">2024-01-22 14:25:30 - 转化率连续下降，当前12.3%，低于阈值13.0%</Text>
                </div>
              </Timeline.Item>
              <Timeline.Item color="green" dot={<CheckCircleOutlined />}>
                <div>
                  <Text strong>错误率告警已解决</Text>
                  <Tag color="green" size="small" style={{ marginLeft: '8px' }}>已解决</Tag>
                  <br />
                  <Text type="secondary">2024-01-21 09:45:30 - 系统错误率恢复正常，问题已修复</Text>
                </div>
              </Timeline.Item>
              <Timeline.Item color="orange" dot={<ClockCircleOutlined />}>
                <div>
                  <Text strong>响应时间告警已确认</Text>
                  <Tag color="orange" size="small" style={{ marginLeft: '8px' }}>已确认</Tag>
                  <br />
                  <Text type="secondary">2024-01-22 11:30:12 - 团队正在调查响应时间异常问题</Text>
                </div>
              </Timeline.Item>
              <Timeline.Item dot={<SyncOutlined />}>
                <div>
                  <Text strong>新告警规则创建</Text>
                  <br />
                  <Text type="secondary">2024-01-21 16:20:00 - 创建了用户参与度趋势监控告警</Text>
                </div>
              </Timeline.Item>
            </Timeline>
          </Card>
        </TabPane>
      </Tabs>

      {/* 创建告警模态框 */}
      <Modal
        title="创建监控告警"
        open={alertModalVisible}
        onCancel={() => setAlertModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setAlertModalVisible(false)}>
            取消
          </Button>,
          <Button key="submit" type="primary" onClick={() => {
            message.success('告警规则已创建');
            setAlertModalVisible(false);
          }}>
            创建告警
          </Button>,
        ]}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item label="告警名称" name="name" required>
            <Input placeholder="输入告警名称" />
          </Form.Item>
          
          <Form.Item label="描述" name="description">
            <Input.TextArea rows={2} placeholder="描述告警的目的和触发条件" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="关联实验" name="experiment_id" required>
                <Select placeholder="选择实验">
                  <Option value="exp_001">首页改版A/B测试</Option>
                  <Option value="exp_002">结算页面优化</Option>
                  <Option value="exp_003">推荐算法测试</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="严重程度" name="severity" required>
                <Select placeholder="选择严重程度">
                  <Option value="low">低</Option>
                  <Option value="medium">中</Option>
                  <Option value="high">高</Option>
                  <Option value="critical">严重</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item label="监控指标" name="metric" required>
                <Select placeholder="选择指标">
                  <Option value="conversion_rate">转化率</Option>
                  <Option value="error_rate">错误率</Option>
                  <Option value="response_time">响应时间</Option>
                  <Option value="engagement_rate">用户参与度</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="比较条件" name="operator" required>
                <Select placeholder="选择条件">
                  <Option value="greater_than">大于</Option>
                  <Option value="less_than">小于</Option>
                  <Option value="equal">等于</Option>
                  <Option value="not_equal">不等于</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item label="阈值" name="threshold" required>
                <InputNumber placeholder="输入阈值" style={{ width: '100%' }} />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item label="持续时间">
            <Slider
              min={1}
              max={60}
              marks={{
                1: '1分钟',
                5: '5分钟',
                10: '10分钟',
                30: '30分钟',
                60: '60分钟',
              }}
              defaultValue={10}
            />
          </Form.Item>

          <Form.Item label="通知渠道" name="notification_channels">
            <Select
              mode="multiple"
              placeholder="选择通知渠道"
              style={{ width: '100%' }}
            >
              <Option value="email">邮件</Option>
              <Option value="slack">Slack</Option>
              <Option value="webhook">Webhook</Option>
              <Option value="sms">短信</Option>
            </Select>
          </Form.Item>

          <Form.Item label="立即启用" name="active">
            <Switch defaultChecked />
          </Form.Item>
        </Form>
      </Modal>

      {/* 告警实例详情模态框 */}
      <Modal
        title="告警实例详情"
        open={instanceModalVisible}
        onCancel={() => setInstanceModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setInstanceModalVisible(false)}>
            关闭
          </Button>,
        ]}
        width={800}
      >
        <Table
          columns={instancesColumns}
          dataSource={alertInstances.filter(instance => 
            !selectedAlert || instance.alert_id === selectedAlert
          )}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 10 }}
        />
      </Modal>
    </div>
  );
};

export default MonitoringAlertsPage;