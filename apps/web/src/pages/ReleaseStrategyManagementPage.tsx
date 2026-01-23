import React, { useState, useEffect } from 'react';
import { Card, Button, Select, Input, Form, Table, Space, Tabs, Row, Col, Statistic, Alert, Switch, InputNumber, Tag, Modal, message } from 'antd';
import { PlusOutlined, PlayCircleOutlined, CheckCircleOutlined, ExclamationCircleOutlined, RocketOutlined, SettingOutlined } from '@ant-design/icons';
import { logger } from '../utils/logger'
import releaseStrategyService, {
  ReleaseStrategy,
  CreateStrategyRequest,
  CreateFromTemplateRequest,
  ApproveStageRequest,
  ReleaseTemplate,
  ReleaseType,
  ApprovalLevel,
  Environment,
  StrategyExecution
} from '../services/releaseStrategyService';

const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;

const ReleaseStrategyManagementPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'strategies' | 'templates' | 'execution' | 'approval' | 'monitoring'>('overview');
  const [strategies, setStrategies] = useState<any[]>([]);
  const [templates, setTemplates] = useState<ReleaseTemplate[]>([]);
  const [releaseTypes, setReleaseTypes] = useState<ReleaseType[]>([]);
  const [approvalLevels, setApprovalLevels] = useState<ApprovalLevel[]>([]);
  const [environments, setEnvironments] = useState<Environment[]>([]);
  const [executions, setExecutions] = useState<StrategyExecution[]>([]);
  const [healthData, setHealthData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [selectedStrategy, setSelectedStrategy] = useState<ReleaseStrategy | null>(null);
  const [modalVisible, setModalVisible] = useState(false);
  const [form] = Form.useForm();

  // 加载数据
  const loadData = async () => {
    setLoading(true);
    try {
      const [strategiesRes, templatesRes, typesRes, levelsRes, envsRes, healthRes] = await Promise.all([
        releaseStrategyService.listStrategies(),
        releaseStrategyService.listTemplates(),
        releaseStrategyService.listReleaseTypes(),
        releaseStrategyService.listApprovalLevels(),
        releaseStrategyService.listEnvironments(),
        releaseStrategyService.getHealthCheck()
      ]);

      setStrategies(strategiesRes.strategies || []);
      setTemplates(templatesRes.templates || []);
      setReleaseTypes(typesRes.release_types || []);
      setApprovalLevels(levelsRes.approval_levels || []);
      setEnvironments(envsRes.environments || []);
      setHealthData(healthRes);
    } catch (error) {
      logger.error('加载数据失败:', error);
      message.error('加载数据失败');
    }
    setLoading(false);
  };

  useEffect(() => {
    loadData();
  }, []);

  // 创建策略
  const handleCreateStrategy = async (values: any) => {
    try {
      const request: CreateStrategyRequest = {
        ...values,
        stages: values.stages || []
      };
      
      await releaseStrategyService.createStrategy(request);
      message.success('策略创建成功');
      setModalVisible(false);
      form.resetFields();
      loadData();
    } catch (error) {
      logger.error('创建策略失败:', error);
      message.error('创建策略失败');
    }
  };

  // 从模板创建
  const handleCreateFromTemplate = async (templateName: string, experimentId: string) => {
    try {
      const request: CreateFromTemplateRequest = {
        experiment_id: experimentId,
        template_name: templateName
      };
      
      await releaseStrategyService.createFromTemplate(request);
      message.success('从模板创建策略成功');
      loadData();
    } catch (error) {
      logger.error('从模板创建失败:', error);
      message.error('从模板创建失败');
    }
  };

  // 执行策略
  const handleExecuteStrategy = async (strategyId: string) => {
    try {
      const result = await releaseStrategyService.executeStrategy(strategyId);
      message.success(`策略开始执行，执行ID: ${result.exec_id}`);
      loadData();
    } catch (error) {
      logger.error('执行策略失败:', error);
      message.error('执行策略失败');
    }
  };

  // 审批阶段
  const handleApproveStage = async (values: any) => {
    try {
      const request: ApproveStageRequest = values;
      const result = await releaseStrategyService.approveStage(request);
      message.success(result.message);
      loadData();
    } catch (error) {
      logger.error('审批失败:', error);
      message.error('审批失败');
    }
  };

  // 渲染系统概览
  const renderOverview = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">发布策略管理概览</h2>
      <Row gutter={16}>
        <Col span={6}>
          <Card>
            <Statistic
              title="策略总数"
              value={healthData?.total_strategies || 0}
              prefix={<RocketOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃执行"
              value={healthData?.active_executions || 0}
              prefix={<PlayCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="模板数量"
              value={templates.length}
              prefix={<SettingOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="发布类型"
              value={releaseTypes.length}
              prefix={<ExclamationCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Card title="系统状态">
        {healthData && (
          <div className="space-y-4">
            <Alert
              message="服务状态"
              description={`发布策略服务运行正常，服务名称: ${healthData.service}`}
              type="success"
              showIcon
            />
            <Row gutter={16}>
              <Col span={12}>
                <Card size="small">
                  <Statistic title="服务状态" value={healthData.status} />
                </Card>
              </Col>
              <Col span={12}>
                <Card size="small">
                  <Statistic title="服务名称" value={healthData.service} />
                </Card>
              </Col>
            </Row>
          </div>
        )}
      </Card>
    </div>
  );

  // 渲染策略管理
  const renderStrategies = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">发布策略</h2>
        <Button type="primary" icon={<PlusOutlined />} onClick={() => setModalVisible(true)}>
          创建策略
        </Button>
      </div>

      <Card title="策略列表">
        <Table
          dataSource={strategies}
          loading={loading}
          rowKey="id"
          columns={[
            { title: '策略名称', dataIndex: 'name', key: 'name' },
            { title: '实验ID', dataIndex: 'experiment_id', key: 'experiment_id' },
            {
              title: '发布类型',
              dataIndex: 'release_type',
              key: 'release_type',
              render: (type: string) => <Tag color="blue">{type}</Tag>
            },
            { title: '阶段数量', dataIndex: 'num_stages', key: 'num_stages' },
            {
              title: '审批级别',
              dataIndex: 'approval_level',
              key: 'approval_level',
              render: (level: string) => <Tag color="orange">{level}</Tag>
            },
            { title: '创建时间', dataIndex: 'created_at', key: 'created_at' },
            {
              title: '操作',
              key: 'actions',
              render: (_, record) => (
                <Space>
                  <Button
                    type="link"
                    icon={<PlayCircleOutlined />}
                    onClick={() => handleExecuteStrategy(record.id)}
                  >
                    执行
                  </Button>
                  <Button type="link" onClick={() => setSelectedStrategy(record)}>
                    详情
                  </Button>
                </Space>
              )
            }
          ]}
        />
      </Card>

      <Modal
        title="创建发布策略"
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={null}
        width={800}
      >
        <Form form={form} layout="vertical" onFinish={handleCreateStrategy}>
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="experiment_id"
                label="实验ID"
                rules={[{ required: true, message: '请输入实验ID' }]}
              >
                <Input placeholder="输入实验ID" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="name"
                label="策略名称"
                rules={[{ required: true, message: '请输入策略名称' }]}
              >
                <Input placeholder="输入策略名称" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item name="description" label="策略描述">
            <TextArea rows={3} placeholder="输入策略描述" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="release_type"
                label="发布类型"
                rules={[{ required: true, message: '请选择发布类型' }]}
              >
                <Select placeholder="选择发布类型">
                  {releaseTypes.map(type => (
                    <Option key={type.value} value={type.value}>
                      {type.name} - {type.description}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="approval_level"
                label="审批级别"
                rules={[{ required: true, message: '请选择审批级别' }]}
              >
                <Select placeholder="选择审批级别">
                  {approvalLevels.map(level => (
                    <Option key={level.value} value={level.value}>
                      {level.name} - {level.description}
                    </Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="auto_promote" label="自动晋级" valuePropName="checked">
                <Switch />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="auto_rollback" label="自动回滚" valuePropName="checked">
                <Switch defaultChecked />
              </Form.Item>
            </Col>
          </Row>

          <div className="flex justify-end space-x-2">
            <Button onClick={() => setModalVisible(false)}>取消</Button>
            <Button type="primary" htmlType="submit">创建策略</Button>
          </div>
        </Form>
      </Modal>
    </div>
  );

  // 渲染模板管理
  const renderTemplates = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">策略模板</h2>

      <Card title="可用模板">
        <Row gutter={16}>
          {templates.map(template => (
            <Col span={8} key={template.name} className="mb-4">
              <Card
                size="small"
                title={template.display_name}
                extra={<Tag color="green">{template.release_type}</Tag>}
                actions={[
                  <Button
                    key="use"
                    type="link"
                    onClick={() => {
                      const experimentId = prompt('请输入实验ID:');
                      if (experimentId) {
                        handleCreateFromTemplate(template.name, experimentId);
                      }
                    }}
                  >
                    使用模板
                  </Button>
                ]}
              >
                <p>{template.description}</p>
                <div className="mt-2">
                  <div>阶段数量: {template.num_stages}</div>
                  <div>审批级别: {template.approval_level}</div>
                  <div>自动晋级: {template.auto_promote ? '是' : '否'}</div>
                  <div>自动回滚: {template.auto_rollback ? '是' : '否'}</div>
                </div>
              </Card>
            </Col>
          ))}
        </Row>
      </Card>
    </div>
  );

  // 渲染执行监控
  const renderExecution = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">执行监控</h2>

      <Card title="执行历史">
        <Table
          dataSource={executions}
          loading={loading}
          rowKey="exec_id"
          columns={[
            { title: '执行ID', dataIndex: 'exec_id', key: 'exec_id' },
            { title: '策略ID', dataIndex: 'strategy_id', key: 'strategy_id' },
            { title: '实验ID', dataIndex: 'experiment_id', key: 'experiment_id' },
            {
              title: '状态',
              dataIndex: 'status',
              key: 'status',
              render: (status: string) => {
                const color = status === 'completed' ? 'green' : status === 'failed' ? 'red' : 'blue';
                return <Tag color={color}>{status}</Tag>;
              }
            },
            { title: '当前阶段', dataIndex: 'current_stage', key: 'current_stage' },
            { title: '开始时间', dataIndex: 'started_at', key: 'started_at' }
          ]}
        />
      </Card>
    </div>
  );

  // 渲染审批管理
  const renderApproval = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">审批管理</h2>

      <Card title="待审批阶段">
        <Form layout="vertical" onFinish={handleApproveStage}>
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="exec_id"
                label="执行ID"
                rules={[{ required: true, message: '请输入执行ID' }]}
              >
                <Input placeholder="输入执行ID" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="stage_index"
                label="阶段索引"
                rules={[{ required: true, message: '请输入阶段索引' }]}
              >
                <InputNumber min={0} placeholder="输入阶段索引" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="approver"
                label="审批人"
                rules={[{ required: true, message: '请输入审批人' }]}
              >
                <Input placeholder="输入审批人ID" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="approved"
                label="审批结果"
                rules={[{ required: true, message: '请选择审批结果' }]}
              >
                <Select placeholder="选择审批结果">
                  <Option value={true}>通过</Option>
                  <Option value={false}>拒绝</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="comments" label="审批意见">
                <Input placeholder="输入审批意见" />
              </Form.Item>
            </Col>
          </Row>

          <Button type="primary" htmlType="submit" icon={<CheckCircleOutlined />}>
            提交审批
          </Button>
        </Form>
      </Card>
    </div>
  );

  // 渲染监控配置
  const renderMonitoring = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">监控配置</h2>

      <Row gutter={16}>
        <Col span={12}>
          <Card title="发布类型配置">
            {releaseTypes.map(type => (
              <div key={type.value} className="mb-4 p-3 border rounded">
                <div className="font-semibold">{type.name}</div>
                <div className="text-gray-600">{type.description}</div>
                <div className="text-sm text-blue-600">适用场景: {type.use_case}</div>
              </div>
            ))}
          </Card>
        </Col>

        <Col span={12}>
          <Card title="环境配置">
            {environments.map(env => (
              <div key={env.value} className="mb-4 p-3 border rounded">
                <div className="font-semibold">{env.name}</div>
                <div className="text-gray-600">{env.description}</div>
                <Tag color="blue">{env.value}</Tag>
              </div>
            ))}
          </Card>
        </Col>
      </Row>

      <Card title="审批级别说明">
        {approvalLevels.map(level => (
          <div key={level.value} className="mb-4 p-3 border rounded">
            <div className="font-semibold">{level.name}</div>
            <div className="text-gray-600">{level.description}</div>
          </div>
        ))}
      </Card>
    </div>
  );

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <h1 className="text-3xl font-bold mb-6">发布策略管理系统</h1>
      
      <Tabs activeKey={activeTab} onChange={(key) => setActiveTab(key as any)}>
        <TabPane tab="系统概览" key="overview">
          {renderOverview()}
        </TabPane>
        <TabPane tab="策略管理" key="strategies">
          {renderStrategies()}
        </TabPane>
        <TabPane tab="模板管理" key="templates">
          {renderTemplates()}
        </TabPane>
        <TabPane tab="执行监控" key="execution">
          {renderExecution()}
        </TabPane>
        <TabPane tab="审批管理" key="approval">
          {renderApproval()}
        </TabPane>
        <TabPane tab="监控配置" key="monitoring">
          {renderMonitoring()}
        </TabPane>
      </Tabs>
    </div>
  );
};

export default ReleaseStrategyManagementPage;
