import React, { useState, useEffect } from 'react';
import dayjs from 'dayjs';
import { logger } from '../../utils/logger';
import { useDebouncedValue } from '../../hooks/useDebouncedValue';
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
  Alert,
  Form,
  Checkbox,
} from 'antd';

const { TextArea } = Input;
import {
  PlusOutlined,
  PlayCircleOutlined,
  PauseCircleOutlined,
  StopOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  MoreOutlined,
  InboxOutlined,
  ExperimentOutlined,
  RiseOutlined,
  UsergroupAddOutlined,
  BarChartOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import { experimentService, type ExperimentData, type ListExperimentsParams, type ExperimentConfig } from '../../services/experimentService';

const { Title, Text } = Typography;
const { Search } = Input;
const { Option } = Select;
const { RangePicker } = DatePicker;

interface Experiment extends ExperimentData {
  traffic_percentage: number;
  users_enrolled: number;
  conversion_rate: number;
  statistical_significance: boolean;
  creator: string;
}

const STATUS_CONFIG = {
  draft: { text: '草稿', color: 'default' },
  running: { text: '运行中', color: 'processing' },
  paused: { text: '已暂停', color: 'warning' },
  completed: { text: '已完成', color: 'success' },
  terminated: { text: '已归档', color: 'default' },
} as const;

const STATUS_OPTIONS = [
  { value: 'all', label: '全部状态' },
  { value: 'draft', label: STATUS_CONFIG.draft.text },
  { value: 'running', label: STATUS_CONFIG.running.text },
  { value: 'paused', label: STATUS_CONFIG.paused.text },
  { value: 'completed', label: STATUS_CONFIG.completed.text },
  { value: 'terminated', label: STATUS_CONFIG.terminated.text },
];

const ExperimentListPage: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedRowKeys, setSelectedRowKeys] = useState<string[]>([]);
  const [searchText, setSearchText] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [dateRange, setDateRange] = useState<any[] | null>(null);
  const [total, setTotal] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const [error, setError] = useState<string | null>(null);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [form] = Form.useForm();
  const debouncedSearchText = useDebouncedValue(searchText, 300);

  const loadExperiments = async () => {
    setLoading(true);
    setError(null);
    try {
      const [startDate, endDate] = dateRange || [];
      const trimmedSearch = debouncedSearchText.trim();
      const params: ListExperimentsParams = {
        search: trimmedSearch || undefined,
        status: statusFilter !== 'all' ? statusFilter : undefined,
        startDateFrom: startDate
          ? startDate.startOf('day').toISOString()
          : undefined,
        startDateTo: endDate ? endDate.endOf('day').toISOString() : undefined,
        page: currentPage,
        pageSize: pageSize,
      };
      
      const response = await experimentService.listExperiments(params);
      
      // 转换数据格式以适应本地接口
      const experimentsWithExtendedData: Experiment[] = response.experiments.map(exp => {
        const trafficPercentage = Array.isArray(exp.variants)
          ? exp.variants.reduce(
              (sum, variant) =>
                sum + Number(variant?.traffic ?? variant?.traffic_percentage ?? 0),
              0
            )
          : 0;
        const participants = exp.participants ?? exp.sampleSize?.current ?? 0;
        const sampleReached = exp.sampleSize?.required
          ? participants >= exp.sampleSize.required
          : false;
        return {
          ...exp,
          traffic_percentage: Math.min(100, Math.max(0, trafficPercentage)),
          users_enrolled: participants,
          conversion_rate: exp.conversion_rate ?? 0,
          statistical_significance: sampleReached,
          creator: exp.owners?.[0] || exp.owner || 'Unknown',
        };
      });
      
      setExperiments(experimentsWithExtendedData);
      setTotal(response.total);
    } catch (error) {
      logger.error('加载实验列表失败:', error);
      setError('加载实验列表失败，请检查网络连接');
      message.error('加载实验列表失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadExperiments();
  }, [debouncedSearchText, statusFilter, dateRange, currentPage, pageSize]);

  const getStatusColor = (status: string) => {
    return (
      STATUS_CONFIG[status as keyof typeof STATUS_CONFIG]?.color || 'default'
    );
  };

  const getStatusText = (status: string) => {
    return STATUS_CONFIG[status as keyof typeof STATUS_CONFIG]?.text || status;
  };

  const handleStatusChange = async (experiment: Experiment, newStatus: string) => {
    try {
      // 根据状态调用相应的API
      if (newStatus === 'running') {
        if (experiment.status === 'paused') {
          await experimentService.resumeExperiment(experiment.id);
        } else {
          await experimentService.startExperiment(experiment.id);
        }
      } else if (newStatus === 'paused') {
        await experimentService.pauseExperiment(experiment.id);
      } else if (newStatus === 'completed') {
        await experimentService.stopExperiment(experiment.id);
      } else if (newStatus === 'terminated') {
        await experimentService.archiveExperiment(experiment.id);
      }
      
      // 重新加载数据
      await loadExperiments();
      message.success(`实验状态已更新为${getStatusText(newStatus)}`);
    } catch (error) {
      logger.error('更新实验状态失败:', error);
      message.error('更新实验状态失败');
    }
  };

  const handleDeleteExperiment = (experimentId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '删除实验后将无法恢复，确认要删除吗？',
      onOk: async () => {
        try {
          await experimentService.deleteExperiment(experimentId);
          await loadExperiments();
          message.success('实验已删除');
        } catch (error) {
          logger.error('删除实验失败:', error);
          message.error('删除实验失败');
        }
      },
    });
  };

  const getSelectedExperiments = () =>
    selectedRowKeys
      .map(key => experiments.find(exp => exp.id === key))
      .filter(Boolean) as Experiment[];

  const handleBatchAction = async (action: 'start' | 'pause' | 'delete') => {
    const selectedExperiments = getSelectedExperiments();
    if (selectedExperiments.length === 0) {
      message.warning('请先选择实验');
      return;
    }

    const actionConfig = {
      start: {
        label: '启动',
        allowStatuses: new Set(['draft', 'paused']),
        request: (exp: Experiment) =>
          exp.status === 'paused'
            ? experimentService.resumeExperiment(exp.id)
            : experimentService.startExperiment(exp.id),
      },
      pause: {
        label: '暂停',
        allowStatuses: new Set(['running']),
        request: (exp: Experiment) => experimentService.pauseExperiment(exp.id),
      },
      delete: {
        label: '删除',
        allowStatuses: new Set(['draft']),
        request: (exp: Experiment) => experimentService.deleteExperiment(exp.id),
      },
    } as const;

    const config = actionConfig[action];
    const eligible = selectedExperiments.filter(exp =>
      config.allowStatuses.has(exp.status)
    );
    const skippedCount = selectedExperiments.length - eligible.length;

    if (eligible.length === 0) {
      message.warning(`没有可${config.label}的实验`);
      return;
    }

    try {
      const results = await Promise.allSettled(
        eligible.map(exp => config.request(exp))
      );
      const successCount = results.filter(result => result.status === 'fulfilled')
        .length;
      const failedCount = results.length - successCount;

      if (successCount > 0) {
        message.success(`${config.label}成功 ${successCount} 个实验`);
      }
      if (failedCount > 0) {
        message.error(`${config.label}失败 ${failedCount} 个实验`);
      }
      if (skippedCount > 0) {
        message.info(`已跳过 ${skippedCount} 个状态不符合的实验`);
      }
    } catch (error) {
      logger.error(`批量${config.label}失败:`, error);
      message.error(`批量${config.label}失败`);
    } finally {
      setSelectedRowKeys([]);
      await loadExperiments();
    }
  };

  const handleBatchDelete = () => {
    Modal.confirm({
      title: '确认批量删除',
      content: '仅草稿状态允许删除，删除后无法恢复。',
      okText: '删除',
      cancelText: '取消',
      onOk: () => handleBatchAction('delete'),
    });
  };

  const handleCreateExperiment = async (values: any) => {
    try {
      const config: ExperimentConfig = {
        name: values.name,
        description: values.description,
        type: values.type || 'A/B Testing',
        status: 'draft',
        variants: [
          { name: 'Control', traffic: 50 },
          { name: 'Treatment', traffic: 50 }
        ],
        metrics: values.metrics ? values.metrics.split(',').map((m: string) => m.trim()) : ['conversion_rate'],
        targetingRules: [],
        confidenceLevel: 0.95,
        tags: values.tags ? values.tags.split(',').map((t: string) => t.trim()) : [],
        enableDataQualityChecks: values.enableDataQualityChecks || false,
        enableAutoStop: values.enableAutoStop || false,
      };

      await experimentService.createExperiment(config);
      message.success('实验创建成功');
      setCreateModalVisible(false);
      form.resetFields();
      await loadExperiments();
    } catch (error) {
      logger.error('创建实验失败:', error);
      message.error('创建实验失败');
    }
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
              label: record.status === 'paused' ? '恢复' : '启动',
              onClick: () => handleStatusChange(record, 'running'),
            },
          ]
        : []),
      ...(record.status === 'running'
        ? [
            {
              key: 'pause',
              icon: <PauseCircleOutlined />,
              label: '暂停',
              onClick: () => handleStatusChange(record, 'paused'),
            },
          ]
        : []),
      ...(record.status === 'running' || record.status === 'paused'
        ? [
            {
              key: 'complete',
              icon: <StopOutlined />,
              label: '结束',
              onClick: () => handleStatusChange(record, 'completed'),
            },
          ]
        : []),
      ...(record.status === 'completed'
        ? [
            {
              key: 'archive',
              icon: <InboxOutlined />,
              label: '归档',
              onClick: () => handleStatusChange(record, 'terminated'),
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
      render: (variants: any[]) => <Text>{variants?.length || 0}</Text>,
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
              <Tag size="small" color="green">样本量达标</Tag>
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
          {record.startDate && (
            <Text style={{ display: 'block', fontSize: '12px' }}>
              开始: {dayjs(record.startDate).format('YYYY-MM-DD')}
            </Text>
          )}
          {record.endDate && (
            <Text style={{ display: 'block', fontSize: '12px' }}>
              结束: {dayjs(record.endDate).format('YYYY-MM-DD')}
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

  const stats = {
    total: total,
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

      {/* 错误信息显示 */}
      {error && (
        <Alert
          message="加载失败"
          description={error}
          type="error"
          showIcon
          style={{ marginBottom: 16 }}
          action={
            <Button size="small" onClick={loadExperiments}>
              重新加载
            </Button>
          }
        />
      )}

      {/* 工具栏 */}
      <Card style={{ marginBottom: '16px' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Search
                placeholder="搜索实验名称或描述"
                value={searchText}
                onChange={(e) => {
                  setSearchText(e.target.value);
                  setCurrentPage(1);
                  setSelectedRowKeys([]);
                }}
                style={{ width: 250 }}
                allowClear
              />
              <Select
                value={statusFilter}
                onChange={(value) => {
                  setStatusFilter(value);
                  setCurrentPage(1);
                  setSelectedRowKeys([]);
                }}
                style={{ width: 120 }}
              >
                {STATUS_OPTIONS.map(option => (
                  <Option key={option.value} value={option.value}>
                    {option.label}
                  </Option>
                ))}
              </Select>
              <RangePicker
                value={dateRange}
                onChange={(value) => {
                  setDateRange(value);
                  setCurrentPage(1);
                  setSelectedRowKeys([]);
                }}
                placeholder={['开始日期', '结束日期']}
              />
            </Space>
          </Col>
          <Col>
            <Space>
              <Button
                type="primary"
                icon={<PlusOutlined />}
                onClick={() => setCreateModalVisible(true)}
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
          dataSource={experiments}
          rowKey="id"
          loading={loading}
          pagination={{
            current: currentPage,
            pageSize: pageSize,
            total: total,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) =>
              `显示 ${range[0]}-${range[1]} 条，共 ${total} 条`,
            onChange: (page, size) => {
              setCurrentPage(page);
              setPageSize(size || 10);
              setSelectedRowKeys([]);
            },
          }}
          rowSelection={{
            selectedRowKeys,
            onChange: setSelectedRowKeys,
          }}
          scroll={{ x: 1200 }}
        />
      </Card>

      {/* 批量操作 */}
      {selectedRowKeys.length > 0 && (
        <Card style={{ position: 'fixed', bottom: '24px', left: '50%', transform: 'translateX(-50%)', zIndex: 1000 }}>
          <Space>
            <Text>已选择 {selectedRowKeys.length} 个实验</Text>
            <Button disabled={loading} onClick={() => handleBatchAction('start')}>批量启动</Button>
            <Button disabled={loading} onClick={() => handleBatchAction('pause')}>批量暂停</Button>
            <Button danger disabled={loading} onClick={handleBatchDelete}>批量删除</Button>
            <Button onClick={() => setSelectedRowKeys([])}>取消选择</Button>
          </Space>
        </Card>
      )}

      {/* 创建实验模态框 */}
      <Modal
        title="创建新实验"
        open={createModalVisible}
        onCancel={() => {
          setCreateModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateExperiment}
        >
          <Form.Item
            label="实验名称"
            name="name"
            rules={[{ required: true, message: '请输入实验名称' }]}
          >
            <Input placeholder="输入实验名称" />
          </Form.Item>

          <Form.Item
            label="实验描述"
            name="description"
            rules={[{ required: true, message: '请输入实验描述' }]}
          >
            <TextArea 
              rows={3} 
              placeholder="描述实验的目标和预期结果"
            />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                label="实验类型"
                name="type"
                initialValue="A/B Testing"
              >
                <Select>
                  <Option value="A/B Testing">A/B测试</Option>
                  <Option value="Multi-variant">多变体测试</Option>
                  <Option value="Feature Flag">功能开关</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                label="指标"
                name="metrics"
                initialValue="conversion_rate,click_rate"
              >
                <Input placeholder="用逗号分隔多个指标" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            label="标签"
            name="tags"
            initialValue="新实验"
          >
            <Input placeholder="用逗号分隔多个标签" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="enableDataQualityChecks"
                valuePropName="checked"
              >
                <Checkbox>启用数据质量检查</Checkbox>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="enableAutoStop"
                valuePropName="checked"
              >
                <Checkbox>启用自动停止</Checkbox>
              </Form.Item>
            </Col>
          </Row>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                创建实验
              </Button>
              <Button onClick={() => {
                setCreateModalVisible(false);
                form.resetFields();
              }}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ExperimentListPage;
