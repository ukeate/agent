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
  DatePicker,
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
} from 'antd';
import {
  FundViewOutlined,
  SettingOutlined,
  PlusOutlined,
  SearchOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  SyncOutlined,
  EyeOutlined,
  ApiOutlined,
  ShoppingCartOutlined,
  HeartOutlined,
  ShareAltOutlined,
  DownloadOutlined,
  EditOutlined,
  DeleteOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  BarChartOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;
const { Search } = Input;
const { Option } = Select;
const { RangePicker } = DatePicker;
const { TabPane } = Tabs;

interface TrackingEvent {
  id: string;
  name: string;
  display_name: string;
  category: 'page_view' | 'click' | 'conversion' | 'engagement' | 'error' | 'custom';
  description: string;
  status: 'active' | 'paused' | 'draft';
  trigger_conditions: string[];
  parameters: {
    name: string;
    type: string;
    required: boolean;
    description: string;
  }[];
  experiments: string[];
  total_events: number;
  events_24h: number;
  success_rate: number;
  created_at: string;
  updated_at: string;
}

interface EventLog {
  id: string;
  event_name: string;
  experiment_id: string;
  user_id: string;
  session_id: string;
  timestamp: string;
  parameters: Record<string, any>;
  status: 'success' | 'failed' | 'pending';
  processing_time: number;
}

interface EventStatistics {
  total_events: number;
  success_rate: number;
  average_processing_time: number;
  events_by_category: {
    category: string;
    count: number;
    percentage: number;
  }[];
  top_events: {
    name: string;
    count: number;
    change: number;
  }[];
}

const EventTrackingPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [selectedTab, setSelectedTab] = useState<string>('events');
  const [modalVisible, setModalVisible] = useState(false);
  const [logModalVisible, setLogModalVisible] = useState(false);
  const [selectedEvent, setSelectedEvent] = useState<string>('');
  const [searchText, setSearchText] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [form] = Form.useForm();

  // 模拟事件追踪数据
  const trackingEvents: TrackingEvent[] = [
    {
      id: 'event_001',
      name: 'homepage_view',
      display_name: '首页访问',
      category: 'page_view',
      description: '用户访问首页时触发',
      status: 'active',
      trigger_conditions: ['page_url = "/home"', 'page_load_complete = true'],
      parameters: [
        { name: 'page_url', type: 'string', required: true, description: '页面URL' },
        { name: 'user_agent', type: 'string', required: false, description: '用户代理' },
        { name: 'referrer', type: 'string', required: false, description: '来源页面' },
      ],
      experiments: ['exp_001', 'exp_003'],
      total_events: 234567,
      events_24h: 12345,
      success_rate: 99.2,
      created_at: '2024-01-10',
      updated_at: '2024-01-22',
    },
    {
      id: 'event_002',
      name: 'button_click',
      display_name: '按钮点击',
      category: 'click',
      description: '用户点击关键按钮时触发',
      status: 'active',
      trigger_conditions: ['element_type = "button"', 'click_event = true'],
      parameters: [
        { name: 'button_id', type: 'string', required: true, description: '按钮ID' },
        { name: 'button_text', type: 'string', required: false, description: '按钮文本' },
        { name: 'page_section', type: 'string', required: false, description: '页面区域' },
      ],
      experiments: ['exp_001', 'exp_002'],
      total_events: 89234,
      events_24h: 4567,
      success_rate: 98.7,
      created_at: '2024-01-12',
      updated_at: '2024-01-20',
    },
    {
      id: 'event_003',
      name: 'purchase_complete',
      display_name: '购买完成',
      category: 'conversion',
      description: '用户完成购买流程时触发',
      status: 'active',
      trigger_conditions: ['payment_status = "success"', 'order_created = true'],
      parameters: [
        { name: 'order_id', type: 'string', required: true, description: '订单ID' },
        { name: 'total_amount', type: 'number', required: true, description: '订单金额' },
        { name: 'payment_method', type: 'string', required: false, description: '支付方式' },
        { name: 'coupon_code', type: 'string', required: false, description: '优惠券代码' },
      ],
      experiments: ['exp_002'],
      total_events: 15678,
      events_24h: 234,
      success_rate: 97.8,
      created_at: '2024-01-15',
      updated_at: '2024-01-21',
    },
    {
      id: 'event_004',
      name: 'form_error',
      display_name: '表单错误',
      category: 'error',
      description: '表单验证失败时触发',
      status: 'paused',
      trigger_conditions: ['form_validation = false', 'error_occurred = true'],
      parameters: [
        { name: 'form_id', type: 'string', required: true, description: '表单ID' },
        { name: 'error_field', type: 'string', required: true, description: '错误字段' },
        { name: 'error_message', type: 'string', required: false, description: '错误信息' },
      ],
      experiments: ['exp_001'],
      total_events: 3456,
      events_24h: 45,
      success_rate: 94.5,
      created_at: '2024-01-18',
      updated_at: '2024-01-22',
    },
  ];

  const eventLogs: EventLog[] = [
    {
      id: 'log_001',
      event_name: 'homepage_view',
      experiment_id: 'exp_001',
      user_id: 'user_12345',
      session_id: 'session_67890',
      timestamp: '2024-01-22 14:30:25',
      parameters: {
        page_url: '/home',
        user_agent: 'Mozilla/5.0...',
        referrer: 'https://google.com',
      },
      status: 'success',
      processing_time: 125,
    },
    {
      id: 'log_002',
      event_name: 'button_click',
      experiment_id: 'exp_001',
      user_id: 'user_12345',
      session_id: 'session_67890',
      timestamp: '2024-01-22 14:30:45',
      parameters: {
        button_id: 'cta-button',
        button_text: '立即购买',
        page_section: 'hero',
      },
      status: 'success',
      processing_time: 98,
    },
    {
      id: 'log_003',
      event_name: 'purchase_complete',
      experiment_id: 'exp_002',
      user_id: 'user_54321',
      session_id: 'session_09876',
      timestamp: '2024-01-22 14:25:12',
      parameters: {
        order_id: 'order_789',
        total_amount: 299.99,
        payment_method: 'credit_card',
        coupon_code: 'SAVE10',
      },
      status: 'success',
      processing_time: 234,
    },
    {
      id: 'log_004',
      event_name: 'form_error',
      experiment_id: 'exp_001',
      user_id: 'user_11111',
      session_id: 'session_22222',
      timestamp: '2024-01-22 14:20:33',
      parameters: {
        form_id: 'checkout-form',
        error_field: 'email',
        error_message: 'Invalid email format',
      },
      status: 'failed',
      processing_time: 67,
    },
  ];

  const eventStatistics: EventStatistics = {
    total_events: 342935,
    success_rate: 98.4,
    average_processing_time: 142,
    events_by_category: [
      { category: '页面访问', count: 234567, percentage: 68.4 },
      { category: '用户交互', count: 89234, percentage: 26.0 },
      { category: '转化事件', count: 15678, percentage: 4.6 },
      { category: '错误事件', count: 3456, percentage: 1.0 },
    ],
    top_events: [
      { name: '首页访问', count: 234567, change: 12.3 },
      { name: '按钮点击', count: 89234, change: 8.7 },
      { name: '购买完成', count: 15678, change: -2.1 },
      { name: '表单错误', count: 3456, change: -15.4 },
    ],
  };

  const getCategoryIcon = (category: string) => {
    switch (category) {
      case 'page_view': return <EyeOutlined />;
      case 'click': return <ApiOutlined />;
      case 'conversion': return <ShoppingCartOutlined />;
      case 'engagement': return <HeartOutlined />;
      case 'error': return <CloseCircleOutlined />;
      default: return <InfoCircleOutlined />;
    }
  };

  const getCategoryColor = (category: string) => {
    switch (category) {
      case 'page_view': return 'blue';
      case 'click': return 'green';
      case 'conversion': return 'gold';
      case 'engagement': return 'purple';
      case 'error': return 'red';
      default: return 'default';
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'success';
      case 'paused': return 'warning';
      case 'draft': return 'default';
      default: return 'default';
    }
  };

  const getStatusText = (status: string) => {
    switch (status) {
      case 'active': return '活跃';
      case 'paused': return '暂停';
      case 'draft': return '草稿';
      default: return status;
    }
  };

  const eventsColumns: ColumnsType<TrackingEvent> = [
    {
      title: '事件名称',
      key: 'name',
      width: 200,
      render: (_, record: TrackingEvent) => (
        <div>
          <div style={{ marginBottom: '4px' }}>
            {getCategoryIcon(record.category)} <Text strong style={{ marginLeft: '4px' }}>
              {record.display_name}
            </Text>
          </div>
          <Text code style={{ fontSize: '11px' }}>{record.name}</Text>
        </div>
      ),
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      width: 100,
      render: (category: string) => (
        <Tag color={getCategoryColor(category)} icon={getCategoryIcon(category)}>
          {category === 'page_view' ? '页面访问' :
           category === 'click' ? '点击' :
           category === 'conversion' ? '转化' :
           category === 'engagement' ? '参与' :
           category === 'error' ? '错误' : '自定义'}
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
      title: '关联实验',
      dataIndex: 'experiments',
      key: 'experiments',
      width: 120,
      render: (experiments: string[]) => (
        <div>
          {experiments.slice(0, 2).map(exp => (
            <Tag key={exp} size="small" style={{ marginBottom: '2px' }}>
              {exp}
            </Tag>
          ))}
          {experiments.length > 2 && (
            <Tag size="small">+{experiments.length - 2}</Tag>
          )}
        </div>
      ),
    },
    {
      title: '24h事件数',
      dataIndex: 'events_24h',
      key: 'events_24h',
      width: 100,
      align: 'right',
      render: (count: number) => <Text strong>{count.toLocaleString()}</Text>,
    },
    {
      title: '成功率',
      dataIndex: 'success_rate',
      key: 'success_rate',
      width: 100,
      render: (rate: number) => (
        <div style={{ textAlign: 'center' }}>
          <Progress
            type="circle"
            size={50}
            percent={rate}
            format={() => `${rate}%`}
            strokeColor={rate >= 98 ? '#52c41a' : rate >= 95 ? '#faad14' : '#ff4d4f'}
          />
        </div>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (_, record: TrackingEvent) => (
        <Space size="small">
          <Tooltip title="查看日志">
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />}
              onClick={() => {
                setSelectedEvent(record.id);
                setLogModalVisible(true);
              }}
            />
          </Tooltip>
          <Tooltip title="编辑">
            <Button type="text" size="small" icon={<EditOutlined />} />
          </Tooltip>
          <Tooltip title={record.status === 'active' ? '暂停' : '启动'}>
            <Button 
              type="text" 
              size="small" 
              icon={record.status === 'active' ? <PauseCircleOutlined /> : <PlayCircleOutlined />}
            />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const logsColumns: ColumnsType<EventLog> = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 120,
      render: (timestamp: string) => (
        <Text style={{ fontSize: '12px' }}>{timestamp}</Text>
      ),
    },
    {
      title: '事件',
      dataIndex: 'event_name',
      key: 'event_name',
      width: 120,
      render: (name: string) => <Text code>{name}</Text>,
    },
    {
      title: '实验ID',
      dataIndex: 'experiment_id',
      key: 'experiment_id',
      width: 80,
      render: (id: string) => <Tag size="small">{id}</Tag>,
    },
    {
      title: '用户ID',
      dataIndex: 'user_id',
      key: 'user_id',
      width: 100,
      render: (id: string) => <Text style={{ fontSize: '12px' }}>{id}</Text>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 80,
      render: (status: string) => (
        <Badge 
          status={status === 'success' ? 'success' : status === 'failed' ? 'error' : 'processing'} 
          text={status === 'success' ? '成功' : status === 'failed' ? '失败' : '处理中'}
        />
      ),
    },
    {
      title: '处理时间',
      dataIndex: 'processing_time',
      key: 'processing_time',
      width: 80,
      render: (time: number) => <Text>{time}ms</Text>,
    },
  ];

  const filteredEvents = trackingEvents.filter(event => {
    const matchesSearch = event.display_name.toLowerCase().includes(searchText.toLowerCase()) ||
                         event.name.toLowerCase().includes(searchText.toLowerCase());
    const matchesCategory = categoryFilter === 'all' || event.category === categoryFilter;
    const matchesStatus = statusFilter === 'all' || event.status === statusFilter;
    return matchesSearch && matchesCategory && matchesStatus;
  });

  const handleCreateEvent = () => {
    setModalVisible(true);
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
              <FundViewOutlined /> 事件跟踪
            </Title>
            <Text type="secondary">监控和管理实验事件数据收集</Text>
          </div>
          <Space>
            <Button icon={<BarChartOutlined />}>
              事件分析
            </Button>
            <Button type="primary" icon={<PlusOutlined />} onClick={handleCreateEvent}>
              创建事件
            </Button>
          </Space>
        </div>
      </div>

      {/* 事件统计概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总事件数"
              value={eventStatistics.total_events}
              prefix={<FundViewOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              过去24小时
            </Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="成功率"
              value={eventStatistics.success_rate}
              precision={1}
              suffix="%"
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              事件处理成功率
            </Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均处理时间"
              value={eventStatistics.average_processing_time}
              suffix="ms"
              prefix={<SyncOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              事件处理延迟
            </Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃事件"
              value={trackingEvents.filter(e => e.status === 'active').length}
              prefix={<PlayCircleOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              正在跟踪的事件数
            </Text>
          </Card>
        </Col>
      </Row>

      {/* 主要内容 */}
      <Tabs activeKey={selectedTab} onChange={setSelectedTab}>
        <TabPane tab="事件配置" key="events">
          {/* 筛选工具栏 */}
          <Card style={{ marginBottom: '16px' }}>
            <Row gutter={16} align="middle">
              <Col>
                <Search
                  placeholder="搜索事件名称"
                  value={searchText}
                  onChange={(e) => setSearchText(e.target.value)}
                  style={{ width: 200 }}
                  allowClear
                />
              </Col>
              <Col>
                <Select
                  value={categoryFilter}
                  onChange={setCategoryFilter}
                  style={{ width: 120 }}
                >
                  <Option value="all">全部分类</Option>
                  <Option value="page_view">页面访问</Option>
                  <Option value="click">点击事件</Option>
                  <Option value="conversion">转化事件</Option>
                  <Option value="engagement">参与事件</Option>
                  <Option value="error">错误事件</Option>
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
                  <Option value="paused">暂停</Option>
                  <Option value="draft">草稿</Option>
                </Select>
              </Col>
            </Row>
          </Card>

          {/* 事件列表 */}
          <Card>
            <Table
              columns={eventsColumns}
              dataSource={filteredEvents}
              rowKey="id"
              loading={loading}
              pagination={{
                total: filteredEvents.length,
                pageSize: 10,
                showSizeChanger: true,
                showQuickJumper: true,
                showTotal: (total, range) =>
                  `显示 ${range[0]}-${range[1]} 条，共 ${total} 条`,
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="事件日志" key="logs">
          <Card>
            <div style={{ marginBottom: '16px' }}>
              <Row gutter={16} align="middle">
                <Col>
                  <RangePicker />
                </Col>
                <Col>
                  <Select placeholder="选择事件" style={{ width: 200 }}>
                    {trackingEvents.map(event => (
                      <Option key={event.id} value={event.name}>
                        {event.display_name}
                      </Option>
                    ))}
                  </Select>
                </Col>
                <Col>
                  <Select placeholder="状态筛选" style={{ width: 120 }}>
                    <Option value="success">成功</Option>
                    <Option value="failed">失败</Option>
                    <Option value="pending">处理中</Option>
                  </Select>
                </Col>
              </Row>
            </div>
            <Table
              columns={logsColumns}
              dataSource={eventLogs}
              rowKey="id"
              loading={loading}
              size="small"
              pagination={{
                total: eventLogs.length,
                pageSize: 20,
                showSizeChanger: true,
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="统计分析" key="analytics">
          <Row gutter={16}>
            <Col span={12}>
              <Card title="事件分类分布">
                <div style={{ marginBottom: '16px' }}>
                  {eventStatistics.events_by_category.map(item => (
                    <div key={item.category} style={{ marginBottom: '12px' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                        <Text>{item.category}</Text>
                        <Text strong>{item.count.toLocaleString()} ({item.percentage}%)</Text>
                      </div>
                      <Progress percent={item.percentage} size="small" />
                    </div>
                  ))}
                </div>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="热门事件排行">
                <List
                  itemLayout="horizontal"
                  dataSource={eventStatistics.top_events}
                  renderItem={(item, index) => (
                    <List.Item>
                      <List.Item.Meta
                        avatar={<Avatar style={{ backgroundColor: '#1890ff' }}>{index + 1}</Avatar>}
                        title={item.name}
                        description={
                          <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Text>{item.count.toLocaleString()} 次</Text>
                            <Text type={item.change > 0 ? 'success' : 'danger'}>
                              {item.change > 0 ? '+' : ''}{item.change}%
                            </Text>
                          </div>
                        }
                      />
                    </List.Item>
                  )}
                />
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 创建事件模态框 */}
      <Modal
        title="创建跟踪事件"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setModalVisible(false)}>
            取消
          </Button>,
          <Button key="submit" type="primary" onClick={() => {
            message.success('事件已创建');
            setModalVisible(false);
          }}>
            创建事件
          </Button>,
        ]}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="事件名称" name="name" required>
                <Input placeholder="例如: button_click" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="显示名称" name="display_name" required>
                <Input placeholder="例如: 按钮点击" />
              </Form.Item>
            </Col>
          </Row>
          <Form.Item label="事件分类" name="category" required>
            <Select placeholder="选择事件分类">
              <Option value="page_view">页面访问</Option>
              <Option value="click">点击事件</Option>
              <Option value="conversion">转化事件</Option>
              <Option value="engagement">参与事件</Option>
              <Option value="error">错误事件</Option>
              <Option value="custom">自定义事件</Option>
            </Select>
          </Form.Item>
          <Form.Item label="描述" name="description">
            <Input.TextArea rows={3} placeholder="描述事件的用途和触发条件" />
          </Form.Item>
          <Form.Item label="触发条件" name="trigger_conditions">
            <Select
              mode="tags"
              placeholder="输入触发条件，例如: element_type = 'button'"
              style={{ width: '100%' }}
            />
          </Form.Item>
          <Form.Item label="关联实验" name="experiments">
            <Select
              mode="multiple"
              placeholder="选择要关联的实验"
              style={{ width: '100%' }}
            >
              <Option value="exp_001">首页改版A/B测试</Option>
              <Option value="exp_002">结算页面优化</Option>
              <Option value="exp_003">推荐算法测试</Option>
            </Select>
          </Form.Item>
          <Form.Item label="启用事件" name="active">
            <Switch defaultChecked />
          </Form.Item>
        </Form>
      </Modal>

      {/* 事件日志详情模态框 */}
      <Modal
        title="事件日志详情"
        open={logModalVisible}
        onCancel={() => setLogModalVisible(false)}
        footer={[
          <Button key="close" onClick={() => setLogModalVisible(false)}>
            关闭
          </Button>,
          <Button key="export" type="primary" icon={<DownloadOutlined />}>
            导出日志
          </Button>,
        ]}
        width={800}
      >
        <Table
          columns={logsColumns}
          dataSource={eventLogs.filter(log => !selectedEvent || log.event_name.includes(selectedEvent))}
          rowKey="id"
          size="small"
          pagination={{
            pageSize: 10,
            showSizeChanger: false,
          }}
          expandable={{
            expandedRowRender: (record) => (
              <div style={{ margin: 0 }}>
                <Text strong>参数详情：</Text>
                <pre style={{ 
                  backgroundColor: '#f5f5f5', 
                  padding: '8px', 
                  marginTop: '8px',
                  fontSize: '12px',
                  borderRadius: '4px'
                }}>
                  {JSON.stringify(record.parameters, null, 2)}
                </pre>
              </div>
            ),
          }}
        />
      </Modal>
    </div>
  );
};

export default EventTrackingPage;