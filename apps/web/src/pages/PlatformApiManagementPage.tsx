import React, { useEffect, useState } from 'react';
import { Button, Card, Form, Input, Modal, Select, Space, Table, Tabs, Tag, Typography, message } from 'antd';
import platformService, {
  type PlatformComponentInfo,
  type PlatformHealthStatus,
  type RegisterComponentRequest,
  type WorkflowRunRequest,
} from '../services/platformService';

const { Title, Paragraph, Text } = Typography;
const { TabPane } = Tabs;
const { TextArea } = Input;

type WorkflowItem = {
  workflow_id: string;
  workflow_type: string;
  last_status?: any;
};

const PlatformApiManagementPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState<PlatformHealthStatus | null>(null);
  const [components, setComponents] = useState<PlatformComponentInfo[]>([]);
  const [workflows, setWorkflows] = useState<WorkflowItem[]>([]);

  const [registerOpen, setRegisterOpen] = useState(false);
  const [registerForm] = Form.useForm<RegisterComponentRequest>();

  const [workflowForm] = Form.useForm<{
    workflow_type: string;
    parameters_json: string;
    priority: number;
  }>();

  const load = async () => {
    setLoading(true);
    try {
      const [h, c] = await Promise.all([platformService.getHealth(), platformService.getComponents()]);
      setHealth(h);
      setComponents(c);
    } catch (e) {
      message.error((e as Error).message || '加载平台数据失败');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void load();
  }, []);

  const handleOpenRegister = () => {
    registerForm.setFieldsValue({
      component_id: `component_${Date.now()}`,
      component_type: 'custom',
      version: '1.0.0',
      health_endpoint: '',
      api_endpoint: '',
      metadata: {},
    } as any);
    setRegisterOpen(true);
  };

  const handleRegister = async (values: any) => {
    setLoading(true);
    try {
      const metadata = values.metadata_json ? JSON.parse(values.metadata_json) : {};
      const payload: RegisterComponentRequest = {
        component_id: values.component_id,
        component_type: values.component_type,
        name: values.name,
        version: values.version,
        health_endpoint: values.health_endpoint,
        api_endpoint: values.api_endpoint,
        metadata,
      };
      const result = await platformService.registerComponent(payload);
      message.success(result.message || '组件注册成功');
      setRegisterOpen(false);
      await load();
    } catch (e) {
      message.error((e as Error).message || '组件注册失败');
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteComponent = (componentId: string) => {
    Modal.confirm({
      title: '确认删除组件',
      content: componentId,
      okText: '删除',
      okType: 'danger',
      cancelText: '取消',
      onOk: async () => {
        setLoading(true);
        try {
          const result = await platformService.deleteComponent(componentId);
          message.success(result.message || '组件已删除');
          await load();
        } catch (e) {
          message.error((e as Error).message || '删除组件失败');
        } finally {
          setLoading(false);
        }
      },
    });
  };

  const handleRunWorkflow = async (values: any) => {
    setLoading(true);
    try {
      const parameters = values.parameters_json ? JSON.parse(values.parameters_json) : {};
      const payload: WorkflowRunRequest = {
        workflow_type: values.workflow_type,
        parameters,
        priority: Number(values.priority) || 0,
      };
      const result = await platformService.runWorkflow(payload);
      message.success(result.message || '工作流已启动');
      setWorkflows((prev) => [{ workflow_id: result.workflow_id, workflow_type: result.workflow_type }, ...prev]);
      workflowForm.resetFields(['parameters_json']);
    } catch (e) {
      message.error((e as Error).message || '启动工作流失败');
    } finally {
      setLoading(false);
    }
  };

  const handleRefreshWorkflow = async (workflowId: string) => {
    setLoading(true);
    try {
      const status = await platformService.getWorkflowStatus(workflowId);
      setWorkflows((prev) =>
        prev.map((w) => (w.workflow_id === workflowId ? { ...w, last_status: status } : w))
      );
    } catch (e) {
      message.error((e as Error).message || '获取工作流状态失败');
    } finally {
      setLoading(false);
    }
  };

  const componentColumns = [
    { title: 'ID', dataIndex: 'component_id', key: 'component_id', width: 220, ellipsis: true },
    { title: '类型', dataIndex: 'component_type', key: 'component_type', render: (v: string) => <Tag>{v}</Tag> },
    { title: '名称', dataIndex: 'name', key: 'name' },
    { title: '版本', dataIndex: 'version', key: 'version', width: 100 },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 120,
      render: (v: string) => (
        <Tag color={v === 'healthy' ? 'green' : v === 'unhealthy' || v === 'error' ? 'red' : 'orange'}>{v}</Tag>
      ),
    },
    { title: '健康检查', dataIndex: 'health_endpoint', key: 'health_endpoint', ellipsis: true },
    { title: 'API端点', dataIndex: 'api_endpoint', key: 'api_endpoint', ellipsis: true },
    { title: '最后心跳', dataIndex: 'last_heartbeat', key: 'last_heartbeat', width: 200 },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (_: unknown, record: PlatformComponentInfo) => (
        <Button danger size="small" onClick={() => handleDeleteComponent(record.component_id)}>
          删除
        </Button>
      ),
    },
  ];

  const workflowColumns = [
    { title: '工作流ID', dataIndex: 'workflow_id', key: 'workflow_id', ellipsis: true },
    { title: '类型', dataIndex: 'workflow_type', key: 'workflow_type', width: 180 },
    {
      title: '状态',
      key: 'status',
      render: (_: unknown, record: WorkflowItem) => {
        const w = record.last_status?.workflow;
        const status = w?.status || record.last_status?.status;
        return <Text>{status || '-'}</Text>;
      },
    },
    {
      title: '操作',
      key: 'actions',
      width: 120,
      render: (_: unknown, record: WorkflowItem) => (
        <Button size="small" onClick={() => handleRefreshWorkflow(record.workflow_id)} loading={loading}>
          刷新
        </Button>
      ),
    },
  ];

  return (
    <div style={{ padding: 24 }}>
      <Title level={2}>平台集成 API</Title>
      <Paragraph type="secondary">覆盖 /api/v1/platform 的健康检查、组件注册/注销、工作流执行与状态接口。</Paragraph>

      <Space style={{ marginBottom: 16 }}>
        <Button onClick={load} loading={loading}>
          刷新
        </Button>
        <Button type="primary" onClick={handleOpenRegister}>
          注册组件
        </Button>
      </Space>

      <Card>
        <Tabs defaultActiveKey="health">
          <TabPane tab="健康检查" key="health">
            <Space direction="vertical" size={8} style={{ width: '100%' }}>
              <Text>
                总体状态: <Text strong>{health?.overall_status || '-'}</Text>
              </Text>
              <Text>
                组件健康: {health?.healthy_components ?? '-'} / {health?.total_components ?? '-'}
              </Text>
              <Text>时间: {health?.timestamp || '-'}</Text>
            </Space>
          </TabPane>

          <TabPane tab="组件" key="components">
            <Table columns={componentColumns} dataSource={components} rowKey="component_id" size="small" />
          </TabPane>

          <TabPane tab="工作流" key="workflows">
            <Card title="启动工作流" size="small" style={{ marginBottom: 16 }}>
              <Form
                form={workflowForm}
                layout="vertical"
                onFinish={handleRunWorkflow}
                initialValues={{ workflow_type: 'full_fine_tuning', parameters_json: '{}', priority: 0 }}
              >
                <Form.Item label="工作流类型" name="workflow_type" rules={[{ required: true, message: '请选择工作流类型' }]}>
                  <Select>
                    <Select.Option value="full_fine_tuning">full_fine_tuning</Select.Option>
                    <Select.Option value="model_optimization">model_optimization</Select.Option>
                    <Select.Option value="evaluation_only">evaluation_only</Select.Option>
                    <Select.Option value="data_processing">data_processing</Select.Option>
                  </Select>
                </Form.Item>
                <Form.Item label="参数(JSON)" name="parameters_json">
                  <TextArea rows={6} />
                </Form.Item>
                <Form.Item label="优先级" name="priority">
                  <Input />
                </Form.Item>
                <Form.Item>
                  <Button type="primary" htmlType="submit" loading={loading}>
                    启动
                  </Button>
                </Form.Item>
              </Form>
            </Card>

            <Card title="工作流列表" size="small">
              <Table columns={workflowColumns} dataSource={workflows} rowKey="workflow_id" size="small" />
            </Card>
          </TabPane>
        </Tabs>
      </Card>

      <Modal title="注册组件" open={registerOpen} onCancel={() => setRegisterOpen(false)} footer={null} destroyOnClose>
        <Form layout="vertical" onFinish={handleRegister} initialValues={{ component_type: 'custom', metadata_json: '{}' }}>
          <Form.Item label="组件ID" name="component_id" rules={[{ required: true, message: '请输入组件ID' }]}>
            <Input />
          </Form.Item>
          <Form.Item label="组件类型" name="component_type" rules={[{ required: true, message: '请选择组件类型' }]}>
            <Select>
              <Select.Option value="fine_tuning">fine_tuning</Select.Option>
              <Select.Option value="compression">compression</Select.Option>
              <Select.Option value="hyperparameter">hyperparameter</Select.Option>
              <Select.Option value="evaluation">evaluation</Select.Option>
              <Select.Option value="data_management">data_management</Select.Option>
              <Select.Option value="model_service">model_service</Select.Option>
              <Select.Option value="custom">custom</Select.Option>
            </Select>
          </Form.Item>
          <Form.Item label="名称" name="name" rules={[{ required: true, message: '请输入名称' }]}>
            <Input />
          </Form.Item>
          <Form.Item label="版本" name="version" rules={[{ required: true, message: '请输入版本' }]}>
            <Input />
          </Form.Item>
          <Form.Item label="健康检查端点" name="health_endpoint" rules={[{ required: true, message: '请输入健康检查端点' }]}>
            <Input />
          </Form.Item>
          <Form.Item label="API端点" name="api_endpoint" rules={[{ required: true, message: '请输入API端点' }]}>
            <Input />
          </Form.Item>
          <Form.Item label="元数据(JSON)" name="metadata_json">
            <TextArea rows={6} />
          </Form.Item>
          <Form.Item>
            <Button type="primary" htmlType="submit" loading={loading}>
              提交
            </Button>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default PlatformApiManagementPage;

