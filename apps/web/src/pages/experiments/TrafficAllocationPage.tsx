import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Button,
  Table,
  Progress,
  Tag,
  Divider,
  Alert,
  Input,
  Form,
  Modal,
  Slider,
  Radio,
  Switch,
  Tooltip,
  Statistic,
  Select,
  message,
  Tabs,
} from 'antd';
import {
  ShareAltOutlined,
  SettingOutlined,
  PieChartOutlined,
  ThunderboltOutlined,
  InfoCircleOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  ReloadOutlined,
  UserOutlined,
  TeamOutlined,
  AimOutlined,
  BranchesOutlined,
  ControlOutlined,
  SyncOutlined,
} from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

interface TrafficAllocation {
  experiment_id: string;
  experiment_name: string;
  status: 'active' | 'paused' | 'completed';
  allocation_method: 'random' | 'hash' | 'sticky' | 'geo';
  total_traffic: number;
  variants: {
    id: string;
    name: string;
    allocation_percentage: number;
    current_users: number;
    expected_users: number;
    deviation: number;
  }[];
  targeting_rules: {
    id: string;
    name: string;
    condition: string;
    allocation_percentage: number;
    active: boolean;
  }[];
  quality_metrics: {
    allocation_accuracy: number;
    user_distribution_quality: number;
    hash_distribution_uniformity: number;
    sticky_session_rate: number;
  };
}

interface AllocationRule {
  id: string;
  name: string;
  type: 'geo' | 'device' | 'user_segment' | 'time' | 'custom';
  condition: string;
  traffic_percentage: number;
  priority: number;
  active: boolean;
}

const TrafficAllocationPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<string>('exp_001');
  const [modalVisible, setModalVisible] = useState(false);
  const [rulesModalVisible, setRulesModalVisible] = useState(false);
  const [rebalanceModalVisible, setRebalanceModalVisible] = useState(false);
  const [form] = Form.useForm();
  const [rulesForm] = Form.useForm();

  // 模拟流量分配数据
  const allocationData: TrafficAllocation[] = [
    {
      experiment_id: 'exp_001',
      experiment_name: '首页改版A/B测试',
      status: 'active',
      allocation_method: 'hash',
      total_traffic: 80,
      variants: [
        {
          id: 'control',
          name: '对照组 (原版首页)',
          allocation_percentage: 50,
          current_users: 7820,
          expected_users: 8000,
          deviation: -2.25,
        },
        {
          id: 'treatment',
          name: '实验组 (新版首页)',
          allocation_percentage: 50,
          current_users: 8180,
          expected_users: 8000,
          deviation: 2.25,
        },
      ],
      targeting_rules: [
        {
          id: 'rule_001',
          name: '桌面端用户',
          condition: 'device_type = "desktop"',
          allocation_percentage: 60,
          active: true,
        },
        {
          id: 'rule_002',
          name: '新用户',
          condition: 'user_type = "new"',
          allocation_percentage: 70,
          active: true,
        },
      ],
      quality_metrics: {
        allocation_accuracy: 97.75,
        user_distribution_quality: 94.2,
        hash_distribution_uniformity: 98.5,
        sticky_session_rate: 89.3,
      },
    },
    {
      experiment_id: 'exp_002',
      experiment_name: '结算页面优化',
      status: 'active',
      allocation_method: 'random',
      total_traffic: 100,
      variants: [
        {
          id: 'control',
          name: '对照组',
          allocation_percentage: 33.3,
          current_users: 2988,
          expected_users: 2977,
          deviation: 0.37,
        },
        {
          id: 'treatment_a',
          name: '实验组A',
          allocation_percentage: 33.3,
          current_users: 2965,
          expected_users: 2977,
          deviation: -0.40,
        },
        {
          id: 'treatment_b',
          name: '实验组B',
          allocation_percentage: 33.4,
          current_users: 2979,
          expected_users: 2983,
          deviation: -0.13,
        },
      ],
      targeting_rules: [],
      quality_metrics: {
        allocation_accuracy: 99.77,
        user_distribution_quality: 98.9,
        hash_distribution_uniformity: 0, // 不适用于随机分配
        sticky_session_rate: 92.1,
      },
    },
  ];

  const allocationRules: AllocationRule[] = [
    {
      id: 'rule_001',
      name: '地理位置定向',
      type: 'geo',
      condition: 'country IN ["CN", "US", "JP"]',
      traffic_percentage: 75,
      priority: 1,
      active: true,
    },
    {
      id: 'rule_002',
      name: '设备类型筛选',
      type: 'device',
      condition: 'device_type = "mobile"',
      traffic_percentage: 60,
      priority: 2,
      active: true,
    },
    {
      id: 'rule_003',
      name: '高价值用户',
      type: 'user_segment',
      condition: 'user_value_score > 80',
      traffic_percentage: 90,
      priority: 3,
      active: false,
    },
  ];

  const currentAllocation = allocationData.find(a => a.experiment_id === selectedExperiment);

  const getDeviationColor = (deviation: number): string => {
    const abs = Math.abs(deviation);
    if (abs <= 2) return '#52c41a';
    if (abs <= 5) return '#faad14';
    return '#ff4d4f';
  };

  const getQualityColor = (quality: number): string => {
    if (quality >= 95) return '#52c41a';
    if (quality >= 90) return '#faad14';
    return '#ff4d4f';
  };

  const variantColumns: ColumnsType<any> = [
    {
      title: '变体名称',
      dataIndex: 'name',
      key: 'name',
      width: 200,
      render: (text: string, record: any) => (
        <div>
          <Text strong>{text}</Text>
          <br />
          <Tag color={record.id === 'control' ? 'blue' : 'green'} size="small">
            {record.id === 'control' ? '对照组' : '实验组'}
          </Tag>
        </div>
      ),
    },
    {
      title: '分配比例',
      dataIndex: 'allocation_percentage',
      key: 'allocation_percentage',
      width: 120,
      render: (percentage: number) => (
        <div style={{ textAlign: 'center' }}>
          <div style={{ marginBottom: '8px' }}>
            <Text strong style={{ fontSize: '16px' }}>
              {percentage.toFixed(1)}%
            </Text>
          </div>
          <Progress
            percent={percentage}
            size="small"
            showInfo={false}
            strokeColor="#1890ff"
          />
        </div>
      ),
    },
    {
      title: '实际用户',
      dataIndex: 'current_users',
      key: 'current_users',
      width: 100,
      align: 'right',
      render: (users: number) => (
        <Text strong>{users.toLocaleString()}</Text>
      ),
    },
    {
      title: '期望用户',
      dataIndex: 'expected_users',
      key: 'expected_users',
      width: 100,
      align: 'right',
      render: (users: number) => (
        <Text type="secondary">{users.toLocaleString()}</Text>
      ),
    },
    {
      title: '偏差',
      dataIndex: 'deviation',
      key: 'deviation',
      width: 100,
      render: (deviation: number) => (
        <div style={{ textAlign: 'center' }}>
          <Text
            strong
            style={{ 
              color: getDeviationColor(deviation),
              fontSize: '14px' 
            }}
          >
            {deviation > 0 ? '+' : ''}{deviation.toFixed(2)}%
          </Text>
        </div>
      ),
    },
  ];

  const rulesColumns: ColumnsType<AllocationRule> = [
    {
      title: '规则名称',
      dataIndex: 'name',
      key: 'name',
      width: 150,
      render: (text: string, record: AllocationRule) => (
        <div>
          <Text strong>{text}</Text>
          <br />
          <Tag color="blue" size="small">{record.type}</Tag>
        </div>
      ),
    },
    {
      title: '条件',
      dataIndex: 'condition',
      key: 'condition',
      width: 200,
      render: (condition: string) => (
        <Text code style={{ fontSize: '12px' }}>
          {condition}
        </Text>
      ),
    },
    {
      title: '流量比例',
      dataIndex: 'traffic_percentage',
      key: 'traffic_percentage',
      width: 100,
      render: (percentage: number) => (
        <Text strong>{percentage}%</Text>
      ),
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      width: 80,
      align: 'center',
      render: (priority: number) => (
        <Tag color="orange">{priority}</Tag>
      ),
    },
    {
      title: '状态',
      dataIndex: 'active',
      key: 'active',
      width: 80,
      render: (active: boolean) => (
        <Switch size="small" checked={active} />
      ),
    },
  ];

  const handleRebalance = () => {
    setRebalanceModalVisible(true);
  };

  const handleManageRules = () => {
    setRulesModalVisible(true);
  };

  const handleAllocationChange = () => {
    setModalVisible(true);
  };

  useEffect(() => {
    setLoading(true);
    setTimeout(() => {
      setLoading(false);
    }, 500);
  }, [selectedExperiment]);

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <div style={{ marginBottom: '24px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div>
            <Title level={2} style={{ margin: 0 }}>
              <ShareAltOutlined /> 流量分配管理
            </Title>
            <Text type="secondary">精确控制实验流量分配和用户定向</Text>
          </div>
          <Space>
            <Button icon={<BranchesOutlined />} onClick={handleManageRules}>
              定向规则
            </Button>
            <Button icon={<ReloadOutlined />} onClick={handleRebalance}>
              重新平衡
            </Button>
            <Button type="primary" icon={<SettingOutlined />} onClick={handleAllocationChange}>
              调整分配
            </Button>
          </Space>
        </div>
      </div>

      {/* 实验选择 */}
      <Card style={{ marginBottom: '16px' }}>
        <Row gutter={16} align="middle">
          <Col>
            <Text>选择实验：</Text>
            <Select
              value={selectedExperiment}
              onChange={setSelectedExperiment}
              style={{ width: 250, marginLeft: 8 }}
            >
              <Option value="exp_001">首页改版A/B测试</Option>
              <Option value="exp_002">结算页面优化</Option>
              <Option value="exp_003">推荐算法测试</Option>
            </Select>
          </Col>
          <Col>
            <Text>分配方法：</Text>
            <Tag color="blue" style={{ marginLeft: 8 }}>
              {currentAllocation?.allocation_method === 'hash' ? 'Hash算法' : 
               currentAllocation?.allocation_method === 'random' ? '随机分配' :
               currentAllocation?.allocation_method === 'sticky' ? '粘性会话' : '地理位置'}
            </Tag>
          </Col>
          <Col>
            <Text>总流量：</Text>
            <Text strong style={{ marginLeft: 8 }}>
              {currentAllocation?.total_traffic}%
            </Text>
          </Col>
        </Row>
      </Card>

      {/* 分配质量指标 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="分配准确度"
              value={currentAllocation?.quality_metrics.allocation_accuracy}
              precision={2}
              suffix="%"
              valueStyle={{ 
                color: getQualityColor(currentAllocation?.quality_metrics.allocation_accuracy || 0) 
              }}
              prefix={<AimOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              实际vs期望流量分配的准确度
            </Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="用户分布质量"
              value={currentAllocation?.quality_metrics.user_distribution_quality}
              precision={1}
              suffix="%"
              valueStyle={{ 
                color: getQualityColor(currentAllocation?.quality_metrics.user_distribution_quality || 0) 
              }}
              prefix={<TeamOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              用户在各变体间的分布均匀性
            </Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="Hash分布均匀性"
              value={currentAllocation?.quality_metrics.hash_distribution_uniformity || 0}
              precision={1}
              suffix="%"
              valueStyle={{ 
                color: getQualityColor(currentAllocation?.quality_metrics.hash_distribution_uniformity || 0) 
              }}
              prefix={<ControlOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              Hash算法的分布均匀程度
            </Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="粘性会话率"
              value={currentAllocation?.quality_metrics.sticky_session_rate}
              precision={1}
              suffix="%"
              valueStyle={{ 
                color: getQualityColor(currentAllocation?.quality_metrics.sticky_session_rate || 0) 
              }}
              prefix={<UserOutlined />}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              用户保持在同一变体的比例
            </Text>
          </Card>
        </Col>
      </Row>

      {/* 主要内容 */}
      <Row gutter={16}>
        <Col span={16}>
          <Card title="变体流量分配" style={{ marginBottom: '16px' }}>
            <Table
              columns={variantColumns}
              dataSource={currentAllocation?.variants}
              rowKey="id"
              loading={loading}
              pagination={false}
              size="small"
            />
          </Card>

          {/* 分配偏差分析 */}
          <Card title="分配偏差分析">
            <Alert
              message="分配偏差检测"
              description="实验组的用户数量比期望值多出2.25%，但仍在可接受范围内（±5%）。"
              variant="default"
              showIcon
              style={{ marginBottom: '16px' }}
            />
            
            <Row gutter={16}>
              {currentAllocation?.variants.map((variant, index) => (
                <Col span={8} key={variant.id}>
                  <div style={{ textAlign: 'center', padding: '16px', border: '1px solid #f0f0f0', borderRadius: '6px' }}>
                    <div style={{ fontSize: '24px', fontWeight: 'bold', marginBottom: '8px' }}>
                      {variant.name}
                    </div>
                    <Progress
                      type="circle"
                      percent={Math.abs(variant.deviation) * 10} // 放大显示
                      format={() => `${variant.deviation > 0 ? '+' : ''}${variant.deviation.toFixed(2)}%`}
                      strokeColor={getDeviationColor(variant.deviation)}
                      size={80}
                    />
                    <div style={{ marginTop: '8px' }}>
                      <Text type="secondary">偏差率</Text>
                    </div>
                  </div>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        <Col span={8}>
          {/* 定向规则 */}
          <Card title="定向规则" size="small" style={{ marginBottom: '16px' }}>
            <Table
              columns={rulesColumns}
              dataSource={allocationRules}
              rowKey="id"
              pagination={false}
              size="small"
            />
          </Card>

          {/* 分配策略说明 */}
          <Card title="分配策略说明" size="small">
            <Tabs size="small">
              <TabPane tab="Hash分配" key="hash">
                <Text style={{ fontSize: '12px' }}>
                  使用Murmur3哈希算法基于用户ID进行稳定分配：
                  <br />• 用户始终分配到相同变体
                  <br />• 分配结果可重现
                  <br />• 适合长期实验
                  <br />• 避免学习效应干扰
                </Text>
              </TabPane>
              <TabPane tab="随机分配" key="random">
                <Text style={{ fontSize: '12px' }}>
                  完全随机分配用户到各个变体：
                  <br />• 每次访问可能分配到不同变体
                  <br />• 快速达到理论分配比例
                  <br />• 适合短期实验
                  <br />• 可能受随机性影响
                </Text>
              </TabPane>
              <TabPane tab="粘性分配" key="sticky">
                <Text style={{ fontSize: '12px' }}>
                  基于会话的粘性分配：
                  <br />• 会话期间保持相同变体
                  <br />• 跨会话可能改变分配
                  <br />• 平衡稳定性和灵活性
                  <br />• 适合中期实验
                </Text>
              </TabPane>
            </Tabs>
          </Card>
        </Col>
      </Row>

      {/* 调整分配模态框 */}
      <Modal
        title="调整流量分配"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setModalVisible(false)}>
            取消
          </Button>,
          <Button key="submit" type="primary" onClick={() => {
            message.success('流量分配已更新');
            setModalVisible(false);
          }}>
            应用更改
          </Button>,
        ]}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item label="分配方法">
            <Radio.Group defaultValue="hash">
              <Radio value="hash">Hash算法 (推荐)</Radio>
              <Radio value="random">随机分配</Radio>
              <Radio value="sticky">粘性会话</Radio>
            </Radio.Group>
          </Form.Item>
          
          <Form.Item label="总流量比例">
            <Slider
              min={10}
              max={100}
              defaultValue={80}
              marks={{
                10: '10%',
                25: '25%',
                50: '50%',
                75: '75%',
                100: '100%',
              }}
            />
          </Form.Item>

          <Divider>变体分配比例</Divider>
          
          <Form.Item label="对照组">
            <Slider
              min={0}
              max={100}
              defaultValue={50}
              marks={{
                0: '0%',
                25: '25%',
                50: '50%',
                75: '75%',
                100: '100%',
              }}
            />
          </Form.Item>

          <Form.Item label="实验组">
            <Slider
              min={0}
              max={100}
              defaultValue={50}
              marks={{
                0: '0%',
                25: '25%',
                50: '50%',
                75: '75%',
                100: '100%',
              }}
            />
          </Form.Item>
        </Form>
      </Modal>

      {/* 定向规则管理模态框 */}
      <Modal
        title="管理定向规则"
        open={rulesModalVisible}
        onCancel={() => setRulesModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setRulesModalVisible(false)}>
            关闭
          </Button>,
          <Button key="add" type="primary" onClick={() => message.success('规则已添加')}>
            添加规则
          </Button>,
        ]}
        width={800}
      >
        <Form rulesForm={rulesForm} layout="vertical">
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="规则名称" name="name">
                <Input placeholder="输入规则名称" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="规则类型" name="type">
                <Select placeholder="选择规则类型">
                  <Option value="geo">地理位置</Option>
                  <Option value="device">设备类型</Option>
                  <Option value="user_segment">用户分段</Option>
                  <Option value="time">时间条件</Option>
                  <Option value="custom">自定义</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          <Form.Item label="条件表达式" name="condition">
            <Input placeholder='例如: device_type = "mobile" AND country = "CN"' />
          </Form.Item>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item label="流量比例" name="traffic_percentage">
                <Slider min={0} max={100} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item label="优先级" name="priority">
                <Select placeholder="选择优先级">
                  <Option value={1}>1 (最高)</Option>
                  <Option value={2}>2</Option>
                  <Option value={3}>3</Option>
                  <Option value={4}>4</Option>
                  <Option value={5}>5 (最低)</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>

      {/* 重新平衡模态框 */}
      <Modal
        title="重新平衡流量"
        open={rebalanceModalVisible}
        onCancel={() => setRebalanceModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setRebalanceModalVisible(false)}>
            取消
          </Button>,
          <Button key="rebalance" type="primary" danger onClick={() => {
            message.success('流量已重新平衡');
            setRebalanceModalVisible(false);
          }}>
            执行重新平衡
          </Button>,
        ]}
      >
        <Alert
          message="重新平衡警告"
          description="重新平衡会影响正在进行的实验，可能导致统计功效下降。请确认您了解这一风险。"
          variant="warning"
          showIcon
          style={{ marginBottom: '16px' }}
        />
        
        <Form layout="vertical">
          <Form.Item label="平衡策略">
            <Radio.Group defaultValue="gradual">
              <Radio value="immediate">立即平衡</Radio>
              <Radio value="gradual">渐进平衡 (推荐)</Radio>
            </Radio.Group>
          </Form.Item>
          
          <Form.Item label="平衡时长 (仅渐进平衡)">
            <Select defaultValue="24" style={{ width: '100%' }}>
              <Option value="6">6小时</Option>
              <Option value="12">12小时</Option>
              <Option value="24">24小时</Option>
              <Option value="48">48小时</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default TrafficAllocationPage;