import React, { useState, useEffect } from 'react';
import {
import { logger } from '../utils/logger'
  Card,
  Button,
  Table,
  Form,
  Input,
  Select,
  Switch,
  Modal,
  Space,
  Typography,
  Divider,
  Alert,
  Tag,
  notification,
  Row,
  Col,
  Statistic,
  Tabs,
  InputNumber
} from 'antd';
import {
  Target,
  Plus,
  Edit,
  Trash2,
  Users,
  Settings,
  BarChart3,
  Play,
  Pause,
  RefreshCw,
  TestTube,
  Filter,
  CheckCircle
} from 'lucide-react';
import targetingRulesService, {
  TargetingRule,
  CreateRuleRequest,
  UpdateRuleRequest,
  RuleStatistics,
  OperatorsResponse,
  UserEvaluationResponse
} from '../services/targetingRulesService';

const { Title, Text, Paragraph } = Typography;
const { TabPane } = Tabs;

const TargetingRulesManagementPage: React.FC = () => {
  const [rules, setRules] = useState<TargetingRule[]>([]);
  const [loading, setLoading] = useState(false);
  const [statistics, setStatistics] = useState<RuleStatistics | null>(null);
  const [operators, setOperators] = useState<OperatorsResponse | null>(null);
  const [selectedRule, setSelectedRule] = useState<TargetingRule | null>(null);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [testModalVisible, setTestModalVisible] = useState(false);
  const [batchTestModalVisible, setBatchTestModalVisible] = useState(false);
  const [eligibilityModalVisible, setEligibilityModalVisible] = useState(false);
  const [templatesModalVisible, setTemplatesModalVisible] = useState(false);
  const [testResult, setTestResult] = useState<UserEvaluationResponse | null>(null);
  const [batchTestResult, setBatchTestResult] = useState<any>(null);
  const [eligibilityResult, setEligibilityResult] = useState<any>(null);
  const [ruleTemplates, setRuleTemplates] = useState<any>(null);
  const [activeTab, setActiveTab] = useState('rules');
  const [form] = Form.useForm();
  const [editForm] = Form.useForm();
  const [testForm] = Form.useForm();
  const [batchTestForm] = Form.useForm();
  const [eligibilityForm] = Form.useForm();

  // 加载规则列表
  const loadRules = async () => {
    setLoading(true);
    try {
      const response = await targetingRulesService.getRules();
      setRules(response.rules);
    } catch (error) {
      logger.error('加载规则失败:', error);
      notification.error({
        message: '加载失败',
        description: '无法加载定向规则列表'
      });
    } finally {
      setLoading(false);
    }
  };

  // 加载统计信息
  const loadStatistics = async () => {
    try {
      const stats = await targetingRulesService.getStatistics();
      setStatistics(stats);
    } catch (error) {
      logger.error('加载统计信息失败:', error);
    }
  };

  // 加载操作符信息
  const loadOperators = async () => {
    try {
      const ops = await targetingRulesService.getOperators();
      setOperators(ops);
    } catch (error) {
      logger.error('加载操作符失败:', error);
    }
  };

  // 加载规则模板
  const loadRuleTemplates = async () => {
    try {
      const templates = await targetingRulesService.getRuleTemplates();
      setRuleTemplates(templates);
    } catch (error) {
      logger.error('加载模板失败:', error);
    }
  };

  // 创建规则
  const createRule = async (values: any) => {
    try {
      const request: CreateRuleRequest = {
        rule_id: values.rule_id,
        name: values.name,
        description: values.description,
        rule_type: values.rule_type,
        condition: {
          field: values.field,
          operator: values.operator,
          value: values.value,
          case_sensitive: values.case_sensitive || false
        },
        priority: values.priority || 0,
        is_active: values.is_active !== false,
        experiment_ids: values.experiment_ids ? values.experiment_ids.split(',').map((id: string) => id.trim()) : [],
        variant_ids: values.variant_ids ? values.variant_ids.split(',').map((id: string) => id.trim()) : [],
        metadata: {}
      };

      await targetingRulesService.createRule(request);
      notification.success({
        message: '创建成功',
        description: '定向规则已创建'
      });
      setCreateModalVisible(false);
      form.resetFields();
      loadRules();
      loadStatistics();
    } catch (error) {
      logger.error('创建规则失败:', error);
      notification.error({
        message: '创建失败',
        description: '无法创建定向规则'
      });
    }
  };

  // 更新规则
  const updateRule = async (values: any) => {
    if (!selectedRule) return;

    try {
      const request: UpdateRuleRequest = {
        name: values.name,
        description: values.description,
        condition: values.field ? {
          field: values.field,
          operator: values.operator,
          value: values.value,
          case_sensitive: values.case_sensitive || false
        } : undefined,
        priority: values.priority,
        is_active: values.is_active,
        experiment_ids: values.experiment_ids ? values.experiment_ids.split(',').map((id: string) => id.trim()) : [],
        variant_ids: values.variant_ids ? values.variant_ids.split(',').map((id: string) => id.trim()) : []
      };

      await targetingRulesService.updateRule(selectedRule.rule_id, request);
      notification.success({
        message: '更新成功',
        description: '定向规则已更新'
      });
      setEditModalVisible(false);
      editForm.resetFields();
      setSelectedRule(null);
      loadRules();
      loadStatistics();
    } catch (error) {
      logger.error('更新规则失败:', error);
      notification.error({
        message: '更新失败',
        description: '无法更新定向规则'
      });
    }
  };

  // 删除规则
  const deleteRule = async (ruleId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '确定要删除这个定向规则吗？',
      onOk: async () => {
        try {
          await targetingRulesService.deleteRule(ruleId);
          notification.success({
            message: '删除成功',
            description: '定向规则已删除'
          });
          loadRules();
          loadStatistics();
        } catch (error) {
          logger.error('删除规则失败:', error);
          notification.error({
            message: '删除失败',
            description: '无法删除定向规则'
          });
        }
      }
    });
  };

  // 切换规则状态
  const toggleRuleStatus = async (rule: TargetingRule) => {
    try {
      await targetingRulesService.updateRule(rule.rule_id, {
        is_active: !rule.is_active
      });
      notification.success({
        message: '状态更新成功',
        description: `规则已${!rule.is_active ? '激活' : '停用'}`
      });
      loadRules();
      loadStatistics();
    } catch (error) {
      logger.error('切换规则状态失败:', error);
      notification.error({
        message: '状态更新失败',
        description: '无法更新规则状态'
      });
    }
  };

  // 测试规则
  const testRule = async (values: any) => {
    try {
      const result = await targetingRulesService.testUserScenario(
        values.user_id,
        JSON.parse(values.user_context || '{}')
      );
      setTestResult(result);
      notification.success({
        message: '测试完成',
        description: '用户定向测试已完成'
      });
    } catch (error) {
      logger.error('测试规则失败:', error);
      notification.error({
        message: '测试失败',
        description: '无法执行用户定向测试'
      });
    }
  };

  // 批量测试规则
  const batchTestRules = async (values: any) => {
    try {
      const userContexts = JSON.parse(values.user_contexts || '[]');
      const result = await targetingRulesService.batchEvaluateUsers({
        user_contexts: userContexts,
        experiment_id: values.experiment_id
      });
      setBatchTestResult(result);
      notification.success({
        message: '批量测试完成',
        description: `已测试 ${result.total_users} 个用户，匹配率 ${result.match_rate_percentage.toFixed(1)}%`
      });
    } catch (error) {
      logger.error('批量测试规则失败:', error);
      notification.error({
        message: '批量测试失败',
        description: '无法执行批量用户定向测试'
      });
    }
  };

  // 检查用户资格
  const checkEligibility = async (values: any) => {
    try {
      const result = await targetingRulesService.checkUserEligibility(
        values.user_id,
        values.experiment_id,
        JSON.parse(values.user_context || '{}')
      );
      setEligibilityResult(result);
      notification.success({
        message: '资格检查完成',
        description: `用户 ${result.is_eligible ? '符合' : '不符合'} 实验条件`
      });
    } catch (error) {
      logger.error('检查用户资格失败:', error);
      notification.error({
        message: '资格检查失败',
        description: '无法检查用户实验资格'
      });
    }
  };

  // 创建预设规则
  const createPresetRules = async () => {
    try {
      const result = await targetingRulesService.createPresetRules();
      notification.success({
        message: '预设规则创建成功',
        description: `成功创建 ${result.created_rules.length} 个预设规则`
      });
      loadRules();
      loadStatistics();
    } catch (error) {
      logger.error('创建预设规则失败:', error);
      notification.error({
        message: '预设规则创建失败',
        description: '无法创建预设规则'
      });
    }
  };

  // 清除所有规则
  const clearAllRules = () => {
    Modal.confirm({
      title: '确认清除',
      content: '确定要清除所有定向规则吗？此操作不可恢复！',
      danger: true,
      onOk: async () => {
        try {
          await targetingRulesService.clearRules();
          notification.success({
            message: '清除成功',
            description: '所有定向规则已清除'
          });
          loadRules();
          loadStatistics();
        } catch (error) {
          logger.error('清除规则失败:', error);
          notification.error({
            message: '清除失败',
            description: '无法清除规则'
          });
        }
      }
    });
  };

  useEffect(() => {
    loadRules();
    loadStatistics();
    loadOperators();
    loadRuleTemplates();
  }, []);

  // 编辑规则时填充表单
  const editRule = (rule: TargetingRule) => {
    setSelectedRule(rule);
    const condition = rule.condition as any;
    editForm.setFieldsValue({
      name: rule.name,
      description: rule.description,
      field: condition.field,
      operator: condition.operator,
      value: condition.value,
      case_sensitive: condition.case_sensitive,
      priority: rule.priority,
      is_active: rule.is_active,
      experiment_ids: rule.experiment_ids?.join(', '),
      variant_ids: rule.variant_ids?.join(', ')
    });
    setEditModalVisible(true);
  };

  // 表格列定义
  const columns = [
    {
      title: 'ID',
      dataIndex: 'rule_id',
      key: 'rule_id',
      width: 200,
      ellipsis: true
    },
    {
      title: '名称',
      dataIndex: 'name',
      key: 'name',
      width: 200
    },
    {
      title: '类型',
      dataIndex: 'rule_type',
      key: 'rule_type',
      width: 100,
      render: (type: string) => (
        <Tag color={
          type === 'blacklist' ? 'red' :
          type === 'whitelist' ? 'green' : 'blue'
        }>
          {type === 'blacklist' ? '黑名单' :
           type === 'whitelist' ? '白名单' : '定向'}
        </Tag>
      )
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
      width: 300,
      ellipsis: true
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      width: 80,
      sorter: (a: TargetingRule, b: TargetingRule) => a.priority - b.priority
    },
    {
      title: '状态',
      dataIndex: 'is_active',
      key: 'is_active',
      width: 100,
      render: (active: boolean, record: TargetingRule) => (
        <Switch
          checked={active}
          onChange={() => toggleRuleStatus(record)}
          checkedChildren="启用"
          unCheckedChildren="停用"
        />
      )
    },
    {
      title: '操作',
      key: 'actions',
      width: 200,
      render: (_: any, record: TargetingRule) => (
        <Space>
          <Button
            size="small"
            icon={<Edit size={14} />}
            onClick={() => editRule(record)}
          >
            编辑
          </Button>
          <Button
            size="small"
            danger
            icon={<Trash2 size={14} />}
            onClick={() => deleteRule(record.rule_id)}
          >
            删除
          </Button>
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
        {/* 页面标题 */}
        <div style={{ marginBottom: '24px' }}>
          <Title level={2}>
            <Target style={{ marginRight: '8px', color: '#1890ff' }} />
            定向规则管理系统
          </Title>
          <Paragraph>
            管理用户定向规则，控制实验参与资格和用户分组策略。
          </Paragraph>
        </div>

        {/* 统计信息 */}
        {statistics && (
          <Row gutter={16} style={{ marginBottom: '24px' }}>
            <Col span={4}>
              <Card>
                <Statistic
                  title="总规则数"
                  value={statistics.total_rules}
                  prefix={<Filter size={16} />}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card>
                <Statistic
                  title="活跃规则"
                  value={statistics.active_rules}
                  prefix={<Play size={16} />}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card>
                <Statistic
                  title="停用规则"
                  value={statistics.inactive_rules}
                  prefix={<Pause size={16} />}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card>
                <Statistic
                  title="平均优先级"
                  value={statistics.average_priority}
                  precision={1}
                  prefix={<BarChart3 size={16} />}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card>
                <Statistic
                  title="近期评估"
                  value={statistics.recent_evaluations}
                  prefix={<Users size={16} />}
                />
              </Card>
            </Col>
            <Col span={4}>
              <Card>
                <Button
                  type="primary"
                  icon={<RefreshCw size={16} />}
                  onClick={() => {
                    loadRules();
                    loadStatistics();
                  }}
                  loading={loading}
                >
                  刷新
                </Button>
              </Card>
            </Col>
          </Row>
        )}

        {/* 主要内容标签页 */}
        <Card>
          <Tabs activeKey={activeTab} onChange={setActiveTab}>
            <TabPane tab="规则列表" key="rules">
              <div style={{ marginBottom: '16px' }}>
                <Space>
                  <Button
                    type="primary"
                    icon={<Plus size={16} />}
                    onClick={() => setCreateModalVisible(true)}
                  >
                    创建规则
                  </Button>
                  <Button
                    icon={<Settings size={16} />}
                    onClick={createPresetRules}
                  >
                    创建预设规则
                  </Button>
                  <Button
                    icon={<TestTube size={16} />}
                    onClick={() => setTestModalVisible(true)}
                  >
                    单用户测试
                  </Button>
                  <Button
                    icon={<Users size={16} />}
                    onClick={() => setBatchTestModalVisible(true)}
                  >
                    批量测试
                  </Button>
                  <Button
                    icon={<CheckCircle size={16} />}
                    onClick={() => setEligibilityModalVisible(true)}
                  >
                    资格检查
                  </Button>
                  <Button
                    icon={<Settings size={16} />}
                    onClick={() => {
                      setTemplatesModalVisible(true);
                      loadRuleTemplates();
                    }}
                  >
                    查看模板
                  </Button>
                  <Button
                    danger
                    icon={<Trash2 size={16} />}
                    onClick={clearAllRules}
                  >
                    清除所有规则
                  </Button>
                </Space>
              </div>

              <Table
                columns={columns}
                dataSource={rules}
                rowKey="rule_id"
                loading={loading}
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showTotal: (total) => `共 ${total} 条规则`
                }}
                scroll={{ x: 1200 }}
              />
            </TabPane>

            <TabPane tab="规则类型统计" key="statistics">
              {statistics && (
                <Row gutter={16}>
                  <Col span={12}>
                    <Card title="规则类型分布">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        {Object.entries(statistics.rule_types).map(([type, count]) => (
                          <div key={type} style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <Tag color={
                              type === 'blacklist' ? 'red' :
                              type === 'whitelist' ? 'green' : 'blue'
                            }>
                              {type === 'blacklist' ? '黑名单' :
                               type === 'whitelist' ? '白名单' : '定向'}
                            </Tag>
                            <Text strong>{count}</Text>
                          </div>
                        ))}
                      </Space>
                    </Card>
                  </Col>
                  <Col span={12}>
                    <Card title="系统状态">
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>规则引擎状态</Text>
                          <Tag color="green">运行中</Tag>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <Text>最后更新</Text>
                          <Text>{new Date().toLocaleString()}</Text>
                        </div>
                      </Space>
                    </Card>
                  </Col>
                </Row>
              )}
            </TabPane>

            <TabPane tab="高级功能" key="advanced">
              <Row gutter={16}>
                <Col span={8}>
                  <Card title="批量操作">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Button
                        type="primary"
                        icon={<Users size={16} />}
                        onClick={() => setBatchTestModalVisible(true)}
                        block
                      >
                        批量用户测试
                      </Button>
                      <Button
                        icon={<CheckCircle size={16} />}
                        onClick={() => setEligibilityModalVisible(true)}
                        block
                      >
                        用户资格检查
                      </Button>
                    </Space>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card title="规则模板">
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <Button
                        icon={<Settings size={16} />}
                        onClick={() => {
                          setTemplatesModalVisible(true);
                          loadRuleTemplates();
                        }}
                        block
                      >
                        查看可用模板
                      </Button>
                      <Button
                        icon={<Plus size={16} />}
                        onClick={createPresetRules}
                        block
                      >
                        创建预设规则
                      </Button>
                    </Space>
                  </Card>
                </Col>
                <Col span={8}>
                  <Card title="运算符参考">
                    {operators && (
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Text strong>规则运算符: {operators.rule_operators.length}</Text>
                        <Text>逻辑运算符: {operators.logical_operators.length}</Text>
                        <Text>规则类型: {operators.rule_types.length}</Text>
                      </Space>
                    )}
                  </Card>
                </Col>
              </Row>
            </TabPane>
          </Tabs>
        </Card>

        {/* 创建规则模态框 */}
        <Modal
          title="创建定向规则"
          visible={createModalVisible}
          onCancel={() => {
            setCreateModalVisible(false);
            form.resetFields();
          }}
          footer={null}
          width={800}
        >
          <Form
            form={form}
            layout="vertical"
            onFinish={createRule}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="规则ID"
                  name="rule_id"
                  rules={[{ required: true, message: '请输入规则ID' }]}
                >
                  <Input placeholder="输入唯一的规则ID" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="规则名称"
                  name="name"
                  rules={[{ required: true, message: '请输入规则名称' }]}
                >
                  <Input placeholder="输入规则名称" />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item
              label="规则描述"
              name="description"
              rules={[{ required: true, message: '请输入规则描述' }]}
            >
              <Input.TextArea rows={3} placeholder="描述规则的用途和逻辑" />
            </Form.Item>

            <Row gutter={16}>
              <Col span={8}>
                <Form.Item
                  label="规则类型"
                  name="rule_type"
                  rules={[{ required: true, message: '请选择规则类型' }]}
                >
                  <Select placeholder="选择规则类型">
                    <Select.Option value="blacklist">黑名单</Select.Option>
                    <Select.Option value="whitelist">白名单</Select.Option>
                    <Select.Option value="targeting">定向</Select.Option>
                  </Select>
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item
                  label="字段"
                  name="field"
                  rules={[{ required: true, message: '请输入字段名' }]}
                >
                  <Input placeholder="例如: country, user_type" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item
                  label="操作符"
                  name="operator"
                  rules={[{ required: true, message: '请选择操作符' }]}
                >
                  <Select placeholder="选择操作符">
                    <Select.Option value="eq">等于</Select.Option>
                    <Select.Option value="neq">不等于</Select.Option>
                    <Select.Option value="in">包含</Select.Option>
                    <Select.Option value="not_in">不包含</Select.Option>
                    <Select.Option value="contains">含有</Select.Option>
                    <Select.Option value="gt">大于</Select.Option>
                    <Select.Option value="gte">大于等于</Select.Option>
                    <Select.Option value="lt">小于</Select.Option>
                    <Select.Option value="lte">小于等于</Select.Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="匹配值"
                  name="value"
                  rules={[{ required: true, message: '请输入匹配值' }]}
                >
                  <Input placeholder="输入匹配值(数组用逗号分隔)" />
                </Form.Item>
              </Col>
              <Col span={6}>
                <Form.Item
                  label="优先级"
                  name="priority"
                >
                  <InputNumber min={0} max={100} placeholder="0" />
                </Form.Item>
              </Col>
              <Col span={6}>
                <Form.Item
                  label="大小写敏感"
                  name="case_sensitive"
                  valuePropName="checked"
                >
                  <Switch />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="实验ID列表"
                  name="experiment_ids"
                >
                  <Input placeholder="多个ID用逗号分隔" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="变体ID列表"
                  name="variant_ids"
                >
                  <Input placeholder="多个ID用逗号分隔" />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit">
                  创建规则
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

        {/* 编辑规则模态框 */}
        <Modal
          title="编辑定向规则"
          visible={editModalVisible}
          onCancel={() => {
            setEditModalVisible(false);
            editForm.resetFields();
            setSelectedRule(null);
          }}
          footer={null}
          width={800}
        >
          <Form
            form={editForm}
            layout="vertical"
            onFinish={updateRule}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="规则名称"
                  name="name"
                >
                  <Input placeholder="输入规则名称" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="优先级"
                  name="priority"
                >
                  <InputNumber min={0} max={100} />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item
              label="规则描述"
              name="description"
            >
              <Input.TextArea rows={3} placeholder="描述规则的用途和逻辑" />
            </Form.Item>

            <Row gutter={16}>
              <Col span={8}>
                <Form.Item
                  label="字段"
                  name="field"
                >
                  <Input placeholder="例如: country, user_type" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item
                  label="操作符"
                  name="operator"
                >
                  <Select placeholder="选择操作符">
                    <Select.Option value="eq">等于</Select.Option>
                    <Select.Option value="neq">不等于</Select.Option>
                    <Select.Option value="in">包含</Select.Option>
                    <Select.Option value="not_in">不包含</Select.Option>
                    <Select.Option value="contains">含有</Select.Option>
                  </Select>
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item
                  label="匹配值"
                  name="value"
                >
                  <Input placeholder="输入匹配值" />
                </Form.Item>
              </Col>
            </Row>

            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="实验ID列表"
                  name="experiment_ids"
                >
                  <Input placeholder="多个ID用逗号分隔" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="激活状态"
                  name="is_active"
                  valuePropName="checked"
                >
                  <Switch checkedChildren="启用" unCheckedChildren="停用" />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit">
                  更新规则
                </Button>
                <Button onClick={() => {
                  setEditModalVisible(false);
                  editForm.resetFields();
                  setSelectedRule(null);
                }}>
                  取消
                </Button>
              </Space>
            </Form.Item>
          </Form>
        </Modal>

        {/* 测试规则模态框 */}
        <Modal
          title="测试用户定向"
          visible={testModalVisible}
          onCancel={() => {
            setTestModalVisible(false);
            testForm.resetFields();
            setTestResult(null);
          }}
          footer={null}
          width={800}
        >
          <Form
            form={testForm}
            layout="vertical"
            onFinish={testRule}
          >
            <Form.Item
              label="用户ID"
              name="user_id"
              rules={[{ required: true, message: '请输入用户ID' }]}
            >
              <Input placeholder="输入要测试的用户ID" />
            </Form.Item>

            <Form.Item
              label="用户上下文 (JSON格式)"
              name="user_context"
              rules={[{ required: true, message: '请输入用户上下文' }]}
            >
              <Input.TextArea
                rows={6}
                placeholder='{"country": "US", "user_type": "premium", "device_type": "mobile"}'
              />
            </Form.Item>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit">
                  开始测试
                </Button>
                <Button onClick={() => {
                  testForm.setFieldsValue({
                    user_id: 'test_user_001',
                    user_context: '{"country": "US", "user_type": "premium", "device_type": "mobile", "age": 25}'
                  });
                }}>
                  填充示例数据
                </Button>
              </Space>
            </Form.Item>
          </Form>

          {testResult && (
            <div style={{ marginTop: '24px' }}>
              <Divider />
              <Title level={4}>测试结果</Title>
              <Alert
                message={`匹配规则: ${testResult.matched_rules} / ${testResult.total_rules_evaluated}`}
                description={`用户 ${testResult.user_id} 的定向测试已完成`}
                type="info"
                showIcon
                style={{ marginBottom: '16px' }}
              />

              <div>
                <Text strong>详细结果:</Text>
                {testResult.results.map((result, index) => (
                  <Card key={index} size="small" style={{ marginTop: '8px' }}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Text strong>{result.rule_id}</Text>
                        <Tag color={result.matched ? 'green' : 'default'}>
                          {result.matched ? '匹配' : '不匹配'}
                        </Tag>
                      </div>
                      <Text type="secondary">{result.evaluation_reason}</Text>
                    </Space>
                  </Card>
                ))}
              </div>
            </div>
          )}
        </Modal>

        {/* 批量测试模态框 */}
        <Modal
          title="批量用户定向测试"
          visible={batchTestModalVisible}
          onCancel={() => {
            setBatchTestModalVisible(false);
            batchTestForm.resetFields();
            setBatchTestResult(null);
          }}
          footer={null}
          width={1000}
        >
          <Form
            form={batchTestForm}
            layout="vertical"
            onFinish={batchTestRules}
          >
            <Form.Item
              label="实验ID (可选)"
              name="experiment_id"
            >
              <Input placeholder="输入实验ID进行过滤" />
            </Form.Item>

            <Form.Item
              label="用户上下文列表 (JSON数组格式)"
              name="user_contexts"
              rules={[{ required: true, message: '请输入用户上下文列表' }]}
            >
              <Input.TextArea
                rows={10}
                placeholder={`[
  {"user_id": "user_001", "country": "US", "user_type": "premium"},
  {"user_id": "user_002", "country": "CN", "user_type": "basic"},
  {"user_id": "user_003", "country": "JP", "user_type": "premium"}
]`}
              />
            </Form.Item>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit">
                  开始批量测试
                </Button>
                <Button onClick={() => {
                  batchTestForm.setFieldsValue({
                    user_contexts: `[
  {"user_id": "user_001", "country": "US", "user_type": "premium", "device_type": "mobile"},
  {"user_id": "user_002", "country": "CN", "user_type": "basic", "device_type": "desktop"},
  {"user_id": "user_003", "country": "JP", "user_type": "premium", "device_type": "mobile"}
]`
                  });
                }}>
                  填充示例数据
                </Button>
              </Space>
            </Form.Item>
          </Form>

          {batchTestResult && (
            <div style={{ marginTop: '24px' }}>
              <Divider />
              <Title level={4}>批量测试结果</Title>
              <Alert
                message={`测试完成: ${batchTestResult.total_users} 个用户，匹配率 ${batchTestResult.match_rate_percentage.toFixed(1)}%`}
                description={`匹配用户数: ${batchTestResult.matched_users}`}
                type="success"
                showIcon
                style={{ marginBottom: '16px' }}
              />
              
              <div>
                <Text strong>详细结果:</Text>
                {batchTestResult.detailed_results?.slice(0, 5).map((result: any, index: number) => (
                  <Card key={index} size="small" style={{ marginTop: '8px' }}>
                    <Space direction="vertical" style={{ width: '100%' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Text strong>用户ID: {result.user_id}</Text>
                        <Tag color={result.matched_rules_count > 0 ? 'green' : 'default'}>
                          匹配 {result.matched_rules_count} 条规则
                        </Tag>
                      </div>
                      <Text type="secondary">
                        规则类型: {result.matched_rule_types?.join(', ') || '无匹配'}
                        {result.has_forced_variant && ' | 强制变体'}
                      </Text>
                    </Space>
                  </Card>
                ))}
                {batchTestResult.detailed_results?.length > 5 && (
                  <Text type="secondary" style={{ marginTop: '8px', display: 'block' }}>
                    显示前5个结果，总共 {batchTestResult.detailed_results.length} 个用户
                  </Text>
                )}
              </div>
            </div>
          )}
        </Modal>

        {/* 用户资格检查模态框 */}
        <Modal
          title="用户实验资格检查"
          visible={eligibilityModalVisible}
          onCancel={() => {
            setEligibilityModalVisible(false);
            eligibilityForm.resetFields();
            setEligibilityResult(null);
          }}
          footer={null}
          width={800}
        >
          <Form
            form={eligibilityForm}
            layout="vertical"
            onFinish={checkEligibility}
          >
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item
                  label="用户ID"
                  name="user_id"
                  rules={[{ required: true, message: '请输入用户ID' }]}
                >
                  <Input placeholder="输入用户ID" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item
                  label="实验ID"
                  name="experiment_id"
                  rules={[{ required: true, message: '请输入实验ID' }]}
                >
                  <Input placeholder="输入实验ID" />
                </Form.Item>
              </Col>
            </Row>

            <Form.Item
              label="用户上下文 (JSON格式)"
              name="user_context"
              rules={[{ required: true, message: '请输入用户上下文' }]}
            >
              <Input.TextArea
                rows={6}
                placeholder='{"country": "US", "user_type": "premium", "device_type": "mobile"}'
              />
            </Form.Item>

            <Form.Item>
              <Space>
                <Button type="primary" htmlType="submit">
                  检查资格
                </Button>
                <Button onClick={() => {
                  eligibilityForm.setFieldsValue({
                    user_id: 'test_user_001',
                    experiment_id: 'exp_premium_features',
                    user_context: '{"country": "US", "user_type": "premium", "device_type": "mobile", "age": 25}'
                  });
                }}>
                  填充示例数据
                </Button>
              </Space>
            </Form.Item>
          </Form>

          {eligibilityResult && (
            <div style={{ marginTop: '24px' }}>
              <Divider />
              <Title level={4}>资格检查结果</Title>
              <Alert
                message={eligibilityResult.is_eligible ? '用户符合实验条件' : '用户不符合实验条件'}
                description={eligibilityResult.eligibility_reason}
                type={eligibilityResult.is_eligible ? 'success' : 'warning'}
                showIcon
                style={{ marginBottom: '16px' }}
              />
              
              <div>
                <Text strong>详细信息:</Text>
                <Card size="small" style={{ marginTop: '8px' }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <Text>匹配规则:</Text>
                      <Text>{eligibilityResult.matched_rules?.join(', ') || '无'}</Text>
                    </div>
                    {eligibilityResult.forced_variant_id && (
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <Text>强制变体ID:</Text>
                        <Tag color="orange">{eligibilityResult.forced_variant_id}</Tag>
                      </div>
                    )}
                  </Space>
                </Card>
              </div>
            </div>
          )}
        </Modal>

        {/* 规则模板模态框 */}
        <Modal
          title="规则模板库"
          visible={templatesModalVisible}
          onCancel={() => setTemplatesModalVisible(false)}
          footer={[
            <Button key="close" onClick={() => setTemplatesModalVisible(false)}>
              关闭
            </Button>
          ]}
          width={1000}
        >
          {ruleTemplates ? (
            <div>
              <Alert
                message="可用模板"
                description="以下是系统提供的规则模板，可以参考这些模板创建自己的规则"
                type="info"
                showIcon
                style={{ marginBottom: '16px' }}
              />
              
              <Row gutter={16}>
                {Object.entries(ruleTemplates.templates || {}).map(([name, template]: [string, any]) => (
                  <Col span={8} key={name}>
                    <Card
                      title={template.name}
                      size="small"
                      style={{ marginBottom: '16px' }}
                      extra={
                        <Tag color={
                          template.rule_type === 'blacklist' ? 'red' :
                          template.rule_type === 'whitelist' ? 'green' : 'blue'
                        }>
                          {template.rule_type === 'blacklist' ? '黑名单' :
                           template.rule_type === 'whitelist' ? '白名单' : '定向'}
                        </Tag>
                      }
                    >
                      <Space direction="vertical" style={{ width: '100%' }}>
                        <Text type="secondary">{template.description}</Text>
                        <div>
                          <Text strong>条件:</Text>
                          <pre style={{ 
                            background: '#f5f5f5', 
                            padding: '8px', 
                            borderRadius: '4px',
                            fontSize: '12px',
                            marginTop: '4px'
                          }}>
                            {JSON.stringify(template.condition, null, 2)}
                          </pre>
                        </div>
                      </Space>
                    </Card>
                  </Col>
                ))}
              </Row>
            </div>
          ) : (
            <Alert
              message="加载中"
              description="正在加载规则模板..."
              type="info"
              showIcon
            />
          )}
        </Modal>
      </div>
    </div>
  );
};

export default TargetingRulesManagementPage;
