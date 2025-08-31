import React, { useState } from 'react';
import { 
  Card, 
  Table, 
  Button, 
  Space, 
  Typography, 
  Row, 
  Col, 
  Statistic, 
  Tag,
  Progress,
  Tabs,
  Form,
  Input,
  Select,
  Slider,
  Switch,
  Alert,
  Descriptions
} from 'antd';
import {
  DeploymentUnitOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  ExportOutlined,
  SettingOutlined,
  LineChartOutlined,
  DatabaseOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const ModelAdaptersPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('list');
  const [form] = Form.useForm();

  // 模拟适配器数据
  const adapters = [
    {
      id: '1',
      name: 'llama2-chat-lora-v1',
      baseModel: 'LLaMA 2 7B',
      type: 'LoRA',
      rank: 16,
      alpha: 32,
      status: '训练完成',
      accuracy: 87.5,
      size: '23MB',
      createdAt: '2025-08-20',
      description: '对话任务优化的LoRA适配器'
    },
    {
      id: '2',
      name: 'mistral-code-qlora-v2',
      baseModel: 'Mistral 7B',
      type: 'QLoRA',
      rank: 8,
      alpha: 16,
      status: '训练中',
      accuracy: 92.3,
      size: '12MB',
      createdAt: '2025-08-21',
      description: '代码生成专用QLoRA适配器'
    },
    {
      id: '3',
      name: 'qwen-summary-lora-v1',
      baseModel: 'Qwen 14B',
      type: 'LoRA',
      rank: 32,
      alpha: 64,
      status: '已部署',
      accuracy: 89.7,
      size: '45MB',
      createdAt: '2025-08-19',
      description: '文本摘要任务适配器'
    }
  ];

  const columns = [
    {
      title: '适配器名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: any) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{text}</div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.description}
          </Text>
        </div>
      ),
    },
    {
      title: '基座模型',
      dataIndex: 'baseModel',
      key: 'baseModel',
    },
    {
      title: '适配器类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color={type === 'LoRA' ? 'blue' : 'purple'}>
          {type}
        </Tag>
      ),
    },
    {
      title: '参数配置',
      key: 'params',
      render: (record: any) => (
        <div>
          <div>Rank: {record.rank}</div>
          <div>Alpha: {record.alpha}</div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          '训练完成': 'green',
          '训练中': 'processing',
          '已部署': 'success'
        };
        return <Tag color={colorMap[status]}>{status}</Tag>;
      },
    },
    {
      title: '性能',
      dataIndex: 'accuracy',
      key: 'accuracy',
      render: (accuracy: number) => (
        <div>
          <div>{accuracy}%</div>
          <Progress percent={accuracy} size="small" showInfo={false} />
        </div>
      ),
    },
    {
      title: '大小',
      dataIndex: 'size',
      key: 'size',
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: any) => (
        <Space>
          <Button size="small" icon={<EditOutlined />}>编辑</Button>
          <Button size="small" icon={<ExportOutlined />}>导出</Button>
          <Button danger size="small" icon={<DeleteOutlined />}>删除</Button>
        </Space>
      ),
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <DeploymentUnitOutlined style={{ marginRight: 8, color: '#1890ff' }} />
          模型适配器管理
        </Title>
        <Text type="secondary">
          管理和部署LoRA/QLoRA适配器，实现模型的快速定制化
        </Text>
      </div>

      {/* 统计概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="适配器总数"
              value={adapters.length}
              prefix={<DeploymentUnitOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已部署"
              value={adapters.filter(a => a.status === '已部署').length}
              prefix={<DatabaseOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均性能"
              value={87.8}
              suffix="%"
              prefix={<LineChartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总大小"
              value="80MB"
              prefix={<DatabaseOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="适配器列表" key="list">
            <div style={{ marginBottom: 16 }}>
              <Space>
                <Button type="primary" icon={<PlusOutlined />}>
                  创建适配器
                </Button>
                <Button icon={<ExportOutlined />}>
                  批量导出
                </Button>
                <Button icon={<SettingOutlined />}>
                  批量配置
                </Button>
              </Space>
            </div>
            <Table 
              columns={columns} 
              dataSource={adapters} 
              rowKey="id"
              pagination={{ pageSize: 10 }}
            />
          </TabPane>

          <TabPane tab="创建适配器" key="create">
            <Row gutter={24}>
              <Col span={12}>
                <Card title="基础配置" size="small">
                  <Form form={form} layout="vertical">
                    <Form.Item label="适配器名称" name="name" rules={[{ required: true }]}>
                      <Input placeholder="输入适配器名称" />
                    </Form.Item>
                    
                    <Form.Item label="基座模型" name="baseModel" rules={[{ required: true }]}>
                      <Select placeholder="选择基座模型">
                        <Option value="llama2-7b">LLaMA 2 7B</Option>
                        <Option value="mistral-7b">Mistral 7B</Option>
                        <Option value="qwen-14b">Qwen 14B</Option>
                        <Option value="chatglm3-6b">ChatGLM3 6B</Option>
                      </Select>
                    </Form.Item>

                    <Form.Item label="适配器类型" name="type" rules={[{ required: true }]}>
                      <Select placeholder="选择适配器类型">
                        <Option value="lora">LoRA</Option>
                        <Option value="qlora">QLoRA</Option>
                      </Select>
                    </Form.Item>

                    <Form.Item label="任务类型" name="taskType">
                      <Select placeholder="选择任务类型">
                        <Option value="chat">对话任务</Option>
                        <Option value="code">代码生成</Option>
                        <Option value="summary">文本摘要</Option>
                        <Option value="translation">机器翻译</Option>
                        <Option value="qa">问答系统</Option>
                      </Select>
                    </Form.Item>

                    <Form.Item label="描述" name="description">
                      <Input.TextArea rows={3} placeholder="输入适配器描述" />
                    </Form.Item>
                  </Form>
                </Card>
              </Col>

              <Col span={12}>
                <Card title="LoRA参数配置" size="small">
                  <Form form={form} layout="vertical">
                    <Form.Item label="Rank (r)" name="rank">
                      <div>
                        <Slider 
                          min={1} 
                          max={256} 
                          defaultValue={16}
                          marks={{ 1: '1', 16: '16', 64: '64', 256: '256' }}
                        />
                        <Text type="secondary">
                          控制适配器容量，值越大表达能力越强但参数量越多
                        </Text>
                      </div>
                    </Form.Item>

                    <Form.Item label="Alpha (α)" name="alpha">
                      <div>
                        <Slider 
                          min={1} 
                          max={128} 
                          defaultValue={32}
                          marks={{ 1: '1', 16: '16', 32: '32', 64: '64', 128: '128' }}
                        />
                        <Text type="secondary">
                          缩放因子，通常设置为Rank的2倍
                        </Text>
                      </div>
                    </Form.Item>

                    <Form.Item label="Dropout" name="dropout">
                      <div>
                        <Slider 
                          min={0} 
                          max={0.5} 
                          step={0.1}
                          defaultValue={0.1}
                          marks={{ 0: '0', 0.1: '0.1', 0.3: '0.3', 0.5: '0.5' }}
                        />
                        <Text type="secondary">
                          防止过拟合的正则化参数
                        </Text>
                      </div>
                    </Form.Item>

                    <Form.Item label="目标模块" name="targetModules">
                      <Select mode="multiple" placeholder="选择要应用LoRA的模块">
                        <Option value="q_proj">q_proj (查询投影)</Option>
                        <Option value="v_proj">v_proj (值投影)</Option>
                        <Option value="k_proj">k_proj (键投影)</Option>
                        <Option value="o_proj">o_proj (输出投影)</Option>
                        <Option value="gate_proj">gate_proj (门控投影)</Option>
                        <Option value="up_proj">up_proj (上投影)</Option>
                        <Option value="down_proj">down_proj (下投影)</Option>
                      </Select>
                    </Form.Item>

                    <Form.Item label="启用偏置" name="bias">
                      <Switch />
                    </Form.Item>
                  </Form>
                </Card>
              </Col>
            </Row>

            <div style={{ marginTop: 16, textAlign: 'center' }}>
              <Space>
                <Button type="primary" size="large">
                  创建适配器
                </Button>
                <Button size="large">
                  保存为模板
                </Button>
                <Button size="large">
                  重置配置
                </Button>
              </Space>
            </div>
          </TabPane>

          <TabPane tab="性能分析" key="performance">
            <Row gutter={16}>
              <Col span={12}>
                <Card title="适配器性能对比" size="small" style={{ marginBottom: 16 }}>
                  <div style={{ marginBottom: 16 }}>
                    <Alert
                      message="性能评估"
                      description="基于验证集的性能指标对比分析"
                      type="info"
                      showIcon
                    />
                  </div>
                  
                  {adapters.map(adapter => (
                    <div key={adapter.id} style={{ marginBottom: 16, padding: 16, border: '1px solid #f0f0f0', borderRadius: 6 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                        <Text strong>{adapter.name}</Text>
                        <Tag color={adapter.type === 'LoRA' ? 'blue' : 'purple'}>
                          {adapter.type}
                        </Tag>
                      </div>
                      <div style={{ marginBottom: 8 }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                          <Text>准确率</Text>
                          <Text>{adapter.accuracy}%</Text>
                        </div>
                        <Progress percent={adapter.accuracy} size="small" />
                      </div>
                      <Descriptions column={2} size="small">
                        <Descriptions.Item label="参数量">{adapter.size}</Descriptions.Item>
                        <Descriptions.Item label="Rank">{adapter.rank}</Descriptions.Item>
                      </Descriptions>
                    </div>
                  ))}
                </Card>
              </Col>

              <Col span={12}>
                <Card title="资源使用分析" size="small" style={{ marginBottom: 16 }}>
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <Text>总存储占用</Text>
                      <Text strong>80MB / 10GB</Text>
                    </div>
                    <Progress percent={0.8} strokeColor="#52c41a" />
                  </div>

                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <Text>内存使用</Text>
                      <Text strong>2.1GB / 16GB</Text>
                    </div>
                    <Progress percent={13.1} strokeColor="#1890ff" />
                  </div>

                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 8 }}>
                      <Text>推理延迟增加</Text>
                      <Text strong>+12ms</Text>
                    </div>
                    <Progress percent={24} strokeColor="#faad14" />
                  </div>

                  <Descriptions bordered size="small">
                    <Descriptions.Item label="平均Rank">18.7</Descriptions.Item>
                    <Descriptions.Item label="参数效率">97.8%</Descriptions.Item>
                    <Descriptions.Item label="推理吞吐量">142 tokens/s</Descriptions.Item>
                    <Descriptions.Item label="训练时间">平均3.2小时</Descriptions.Item>
                  </Descriptions>
                </Card>

                <Card title="优化建议" size="small">
                  <div style={{ padding: '12px', backgroundColor: '#f6ffed', border: '1px solid #b7eb8f', borderRadius: 6, marginBottom: 12 }}>
                    <Text strong style={{ color: '#52c41a' }}>✓ 性能良好</Text>
                    <ul style={{ margin: '8px 0 0 0', paddingLeft: 16, fontSize: '12px' }}>
                      <li>适配器大小合理，存储效率高</li>
                      <li>性能提升显著</li>
                      <li>推理延迟在可接受范围</li>
                    </ul>
                  </div>

                  <div style={{ padding: '12px', backgroundColor: '#fff7e6', border: '1px solid #ffd591', borderRadius: 6, marginBottom: 12 }}>
                    <Text strong style={{ color: '#faad14' }}>⚠ 注意事项</Text>
                    <ul style={{ margin: '8px 0 0 0', paddingLeft: 16, fontSize: '12px' }}>
                      <li>部分适配器Rank设置偏高</li>
                      <li>可考虑使用更小的Alpha值</li>
                      <li>建议启用梯度检查点节省内存</li>
                    </ul>
                  </div>

                  <div style={{ padding: '12px', backgroundColor: '#e6f4ff', border: '1px solid #91caff', borderRadius: 6 }}>
                    <Text strong style={{ color: '#1890ff' }}>💡 优化建议</Text>
                    <ul style={{ margin: '8px 0 0 0', paddingLeft: 16, fontSize: '12px' }}>
                      <li>尝试不同的目标模块组合</li>
                      <li>实验更精细的Dropout设置</li>
                      <li>考虑使用AdaLoRA动态调整Rank</li>
                    </ul>
                  </div>
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="部署管理" key="deployment">
            <Row gutter={16}>
              <Col span={16}>
                <Card title="部署状态" size="small">
                  <Table
                    columns={[
                      { title: '适配器', dataIndex: 'name', key: 'name' },
                      { title: '环境', dataIndex: 'env', key: 'env', render: (env: string) => <Tag>{env}</Tag> },
                      { title: '状态', dataIndex: 'deployStatus', key: 'deployStatus', 
                        render: (status: string) => (
                          <Tag color={status === '运行中' ? 'green' : status === '部署中' ? 'processing' : 'default'}>
                            {status}
                          </Tag>
                        )
                      },
                      { title: 'QPS', dataIndex: 'qps', key: 'qps' },
                      { title: '延迟', dataIndex: 'latency', key: 'latency' },
                      { title: '操作', key: 'action', render: () => (
                        <Space>
                          <Button size="small">监控</Button>
                          <Button size="small">更新</Button>
                          <Button size="small" danger>停止</Button>
                        </Space>
                      )}
                    ]}
                    dataSource={[
                      { name: 'llama2-chat-lora-v1', env: '生产', deployStatus: '运行中', qps: 12.5, latency: '45ms' },
                      { name: 'qwen-summary-lora-v1', env: '测试', deployStatus: '运行中', qps: 8.2, latency: '52ms' },
                      { name: 'mistral-code-qlora-v2', env: '开发', deployStatus: '部署中', qps: 0, latency: '-' }
                    ]}
                    rowKey="name"
                    size="small"
                  />
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="快速部署" size="small" style={{ marginBottom: 16 }}>
                  <Form layout="vertical">
                    <Form.Item label="选择适配器">
                      <Select placeholder="选择要部署的适配器">
                        {adapters.map(adapter => (
                          <Option key={adapter.id} value={adapter.id}>
                            {adapter.name}
                          </Option>
                        ))}
                      </Select>
                    </Form.Item>
                    
                    <Form.Item label="部署环境">
                      <Select placeholder="选择部署环境">
                        <Option value="dev">开发环境</Option>
                        <Option value="test">测试环境</Option>
                        <Option value="prod">生产环境</Option>
                      </Select>
                    </Form.Item>
                    
                    <Form.Item label="实例数量">
                      <Slider min={1} max={10} defaultValue={1} marks={{ 1: '1', 5: '5', 10: '10' }} />
                    </Form.Item>
                    
                    <Button type="primary" block>
                      立即部署
                    </Button>
                  </Form>
                </Card>

                <Card title="部署统计" size="small">
                  <Statistic title="总部署次数" value={23} style={{ marginBottom: 16 }} />
                  <Statistic title="成功率" value={95.7} suffix="%" valueStyle={{ color: '#52c41a' }} style={{ marginBottom: 16 }} />
                  <Statistic title="平均部署时间" value={3.2} suffix="分钟" />
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default ModelAdaptersPage;