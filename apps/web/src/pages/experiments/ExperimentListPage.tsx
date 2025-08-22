import React, { useState, useEffect } from 'react';
import {
  Card,
  Table,
  Button,
  Tag,
  Space,
  Typography,
  Row,
  Col,
  Statistic,
  Progress,
  Dropdown,
  MenuProps,
  message,
  Input,
  Select,
  DatePicker,
  Modal,
} from 'antd';
import {
  PlusOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  MoreOutlined,
  ExperimentOutlined,
  RiseOutlined,
  UsergroupAddOutlined,
  BarChartOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text } = Typography;
const { Search } = Input;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface Experiment {
  id: string;
  name: string;
  description: string;
  status: 'draft' | 'running' | 'paused' | 'completed' | 'archived';
  variants: number;
  traffic_percentage: number;
  users_enrolled: number;
  conversion_rate: number;
  statistical_significance: boolean;
  start_date?: string;
  end_date?: string;
  created_at: string;
  updated_at: string;
  creator: string;
}

const ExperimentListPage: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const [searchText, setSearchText] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [dateRange, setDateRange] = useState<any[]>([]);

  // 模拟数据
  const mockExperiments: Experiment[] = [
    {
      id: 'exp_001',
      name: '首页改版A/B测试',
      description: '测试新版首页对用户转化率的影响',
      status: 'running',
      variants: 2,
      traffic_percentage: 50,
      users_enrolled: 15420,
      conversion_rate: 0.145,
      statistical_significance: true,
      start_date: '2024-01-15',
      end_date: '2024-02-15',
      created_at: '2024-01-10',
      updated_at: '2024-01-20',
      creator: 'Product Team',
    },
    {
      id: 'exp_002',
      name: '结算页面优化',
      description: '优化结算流程，减少用户流失',
      status: 'completed',
      variants: 3,
      traffic_percentage: 100,
      users_enrolled: 8930,
      conversion_rate: 0.234,
      statistical_significance: true,
      start_date: '2024-01-01',
      end_date: '2024-01-31',
      created_at: '2023-12-28',
      updated_at: '2024-01-31',
      creator: 'UX Team',
    },
    {
      id: 'exp_003',
      name: '推荐算法测试',
      description: '测试新的机器学习推荐算法效果',
      status: 'draft',
      variants: 2,
      traffic_percentage: 20,
      users_enrolled: 0,
      conversion_rate: 0,
      statistical_significance: false,
      created_at: '2024-01-22',
      updated_at: '2024-01-22',
      creator: 'ML Team',
    },
    {
      id: 'exp_004',
      name: '定价策略实验',
      description: '测试不同定价策略对销售的影响',
      status: 'paused',
      variants: 4,
      traffic_percentage: 25,
      users_enrolled: 3245,
      conversion_rate: 0.089,
      statistical_significance: false,
      start_date: '2024-01-18',
      created_at: '2024-01-15',
      updated_at: '2024-01-20',
      creator: 'Pricing Team',
    },
  ];

  useEffect(() => {
    setLoading(true);
    // 模拟API调用
    setTimeout(() => {
      setExperiments(mockExperiments);
      setLoading(false);
    }, 800);
  }, []);

  const getStatusColor = (status: string) => {
    const colors = {
      draft: 'default',
      running: 'processing',
      paused: 'warning',
      completed: 'success',
      archived: 'default',
    };
    return colors[status as keyof typeof colors];
  };

  const getStatusText = (status: string) => {
    const texts = {
      draft: '草稿',
      running: '运行中',
      paused: '已暂停',
      completed: '已完成',
      archived: '已归档',
    };
    return texts[status as keyof typeof texts];
  };

  const handleStatusChange = (experimentId: string, newStatus: string) => {
    setExperiments(prev =>
      prev.map(exp =>
        exp.id === experimentId
          ? { ...exp, status: newStatus as Experiment['status'], updated_at: new Date().toISOString().split('T')[0] }
          : exp
      )
    );
    message.success(`实验状态已更新为${getStatusText(newStatus)}`);
  };

  const handleDeleteExperiment = (experimentId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '删除实验后将无法恢复，确认要删除吗？',
      onOk: () => {
        setExperiments(prev => prev.filter(exp => exp.id !== experimentId));
        message.success('实验已删除');
      },
    });
  };

  const getActionMenu = (record: Experiment): MenuProps => ({
    items: [
      {
        key: 'view',
        icon: <EyeOutlined />,
        label: '查看详情',
        onClick: () => {
          message.info(`查看实验: ${record.name}`);
        },
      },
      {
        key: 'edit',
        icon: <EditOutlined />,
        label: '编辑',
        disabled: record.status === 'running',
        onClick: () => {
          message.info(`编辑实验: ${record.name}`);
        },
      },
      {
        type: 'divider',
      },
      ...(record.status === 'draft' || record.status === 'paused'
        ? [
            {
              key: 'start',
              icon: <PlayCircleOutlined />,
              label: '启动',
              onClick: () => handleStatusChange(record.id, 'running'),
            },
          ]
        : []),
      ...(record.status === 'running'
        ? [
            {
              key: 'pause',
              icon: <PauseCircleOutlined />,
              label: '暂停',
              onClick: () => handleStatusChange(record.id, 'paused'),
            },
          ]
        : []),
      ...(record.status === 'running' || record.status === 'paused'
        ? [
            {
              key: 'complete',
              icon: <StopOutlined />,
              label: '结束',
              onClick: () => handleStatusChange(record.id, 'completed'),
            },
          ]
        : []),
      {
        type: 'divider',
      },
      {
        key: 'delete',
        icon: <DeleteOutlined />,
        label: '删除',
        danger: true,
        onClick: () => handleDeleteExperiment(record.id),
      },
    ],
  });

  const columns: ColumnsType<Experiment> = [
    {
      title: '实验名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      render: (text: string, record: Experiment) => (
        <div>
          <Text strong style={{ display: 'block' }}>
            {text}
          </Text>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.description}
          </Text>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 100,
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>{getStatusText(status)}</Tag>
      ),
    },
    {
      title: '变体数量',
      dataIndex: 'variants',
      key: 'variants',
      width: 80,
      align: 'center',
      render: (variants: number) => <Text>{variants}</Text>,
    },
    {
      title: '流量分配',
      dataIndex: 'traffic_percentage',
      key: 'traffic_percentage',
      width: 120,
      render: (percentage: number) => (
        <div style={{ width: 80 }}>
          <Progress
            percent={percentage}
            size="small"
            format={(percent) => `${percent}%`}
          />
        </div>
      ),
    },
    {
      title: '参与用户',
      dataIndex: 'users_enrolled',
      key: 'users_enrolled',
      width: 100,
      align: 'right',
      render: (users: number) => (
        <Text>{users.toLocaleString()}</Text>
      ),
    },
    {
      title: '转化率',
      dataIndex: 'conversion_rate',
      key: 'conversion_rate',
      width: 100,
      align: 'right',
      render: (rate: number, record: Experiment) => (
        <div style={{ textAlign: 'right' }}>
          <Text strong={record.statistical_significance}>
            {(rate * 100).toFixed(1)}%
          </Text>
          {record.statistical_significance && (
            <div>
              <Tag size="small" color="green">显著</Tag>
            </div>
          )}
        </div>
      ),
    },
    {
      title: '实验时间',
      key: 'duration',
      width: 140,
      render: (_, record: Experiment) => (
        <div>
          {record.start_date && (
            <Text style={{ display: 'block', fontSize: '12px' }}>
              开始: {record.start_date}
            </Text>
          )}
          {record.end_date && (
            <Text style={{ display: 'block', fontSize: '12px' }}>
              结束: {record.end_date}
            </Text>
          )}
        </div>
      ),
    },
    {
      title: '创建者',
      dataIndex: 'creator',
      key: 'creator',
      width: 100,
      render: (creator: string) => <Text>{creator}</Text>,
    },
    {
      title: '操作',
      key: 'actions',
      width: 60,
      fixed: 'right',
      render: (_, record: Experiment) => (
        <Dropdown menu={getActionMenu(record)} trigger={['click']}>
          <Button type="text" size="small" icon={<MoreOutlined />} />
        </Dropdown>
      ),
    },
  ];

  const filteredExperiments = experiments.filter(exp => {
    const matchesSearch = exp.name.toLowerCase().includes(searchText.toLowerCase()) ||
                         exp.description.toLowerCase().includes(searchText.toLowerCase());
    const matchesStatus = statusFilter === 'all' || exp.status === statusFilter;
    return matchesSearch && matchesStatus;
  });

  const stats = {
    total: experiments.length,
    running: experiments.filter(e => e.status === 'running').length,
    completed: experiments.filter(e => e.status === 'completed').length,
    draft: experiments.filter(e => e.status === 'draft').length,
  };

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: '24px' }}>
        <Title level={2} style={{ margin: 0 }}>
          <ExperimentOutlined /> 实验管理
        </Title>
        <Text type="secondary">管理和监控A/B测试实验</Text>
      </div>

      {/* 统计卡片 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总实验数"
              value={stats.total}
              prefix={<ExperimentOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="运行中"
              value={stats.running}
              valueStyle={{ color: '#1890ff' }}
              prefix={<PlayCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已完成"
              value={stats.completed}
              valueStyle={{ color: '#52c41a' }}
              prefix={<StopOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="草稿"
              value={stats.draft}
              valueStyle={{ color: '#faad14' }}
              prefix={<EditOutlined />}
            />
          </Card>
        </Col>
      </Row>

      {/* 工具栏 */}
      <Card style={{ marginBottom: '16px' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Search
                placeholder="搜索实验名称或描述"
                value={searchText}
                onChange={(e) => setSearchText(e.target.value)}
                style={{ width: 250 }}
                allowClear
              />
              <Select
                value={statusFilter}
                onChange={setStatusFilter}
                style={{ width: 120 }}
              >
                <Option value="all">全部状态</Option>
                <Option value="draft">草稿</Option>
                <Option value="running">运行中</Option>
                <Option value="paused">已暂停</Option>
                <Option value="completed">已完成</Option>
                <Option value="archived">已归档</Option>
              </Select>
              <RangePicker
                value={dateRange}
                onChange={setDateRange}
                placeholder={['开始日期', '结束日期']}
              />
            </Space>
          </Col>
          <Col>
            <Space>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => message.info('创建新实验')}
              >
                创建实验
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      {/* 实验列表 */}
      <Card>
        <Table
          columns={columns}
          dataSource={filteredExperiments}
          rowKey="id"
          loading={loading}
          pagination={{
            total: filteredExperiments.length,
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) =>
              `显示 ${range[0]}-${range[1]} 条，共 ${total} 条`,
          }}
          rowSelection={{
            selectedRowKeys,
            onChange: setSelectedRowKeys,
            getCheckboxProps: (record) => ({
              disabled: record.status === 'running',
            }),
          }}
          scroll={{ x: 1200 }}
        />
      </Card>

      {/* 批量操作 */}
      {selectedRowKeys.length > 0 && (
        <Card style={{ position: 'fixed', bottom: '24px', left: '50%', transform: 'translateX(-50%)', zIndex: 1000 }}>
          <Space>
            <Text>已选择 {selectedRowKeys.length} 个实验</Text>
            <Button onClick={() => message.info('批量启动')}>批量启动</Button>
            <Button onClick={() => message.info('批量暂停')}>批量暂停</Button>
            <Button danger onClick={() => message.info('批量删除')}>批量删除</Button>
            <Button onClick={() => setSelectedRowKeys([])}>取消选择</Button>
          </Space>
        </Card>
      )}
    </div>
  );
};

export default ExperimentListPage;