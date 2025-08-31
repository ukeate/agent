import React, { useState } from 'react';
import { 
  Card, 
  Form, 
  Select, 
  Input, 
  Slider, 
  Switch, 
  Button, 
  Row, 
  Col, 
  Typography, 
  Space,
  Tabs,
  Alert,
  Descriptions,
  Tag,
  Divider,
  Table,
  Tooltip
} from 'antd';
import {
  SettingOutlined,
  SaveOutlined,
  ReloadOutlined,
  ExportOutlined,
  ImportOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined,
  FileTextOutlined,
  ThunderboltOutlined,
  DatabaseOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { TextArea } = Input;
const { Option } = Select;

const configTemplates = [
  {
    name: 'LoRA小模型快速微调',
    description: '适用于7B以下模型的高效微调配置',
    config: {
      loraRank: 16,
      loraAlpha: 32,
      loraDropout: 0.1,
      learningRate: 2e-4,
      batchSize: 4,
      epochs: 3,
      targetModules: ['q_proj', 'v_proj']
    }
  },
  {
    name: 'QLoRA大模型训练',
    description: '适用于13B+大模型的内存优化配置',
    config: {
      loraRank: 8,
      loraAlpha: 16,
      loraDropout: 0.05,
      learningRate: 1e-4,
      batchSize: 2,
      epochs: 5,
      quantization: '4-bit',
      targetModules: ['q_proj', 'v_proj', 'k_proj', 'o_proj']
    }
  }
];

const FineTuningConfigPage: React.FC = () => {
  const [form] = Form.useForm();
  const [activeTab, setActiveTab] = useState('basic');
  const [selectedTemplate, setSelectedTemplate] = useState<string>('');

  const handleTemplateSelect = (templateName: string) => {
    const template = configTemplates.find(t => t.name === templateName);
    if (template) {
      form.setFieldsValue(template.config);
      setSelectedTemplate(templateName);
    }
  };

  const handleSaveConfig = () => {
    form.validateFields().then(values => {
      console.log('保存配置:', values);
    });
  };

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <SettingOutlined style={{ marginRight: 8, color: '#1890ff' }} />
          微调配置管理
        </Title>
        <Text type="secondary">
          配置和管理LoRA/QLoRA微调参数，支持预设模板和自定义配置
        </Text>
      </div>

      <Row gutter={16}>
        <Col span={18}>
          <Card>
            <Tabs activeKey={activeTab} onChange={setActiveTab}>
              <TabPane tab="基础配置" key="basic">
                <Form form={form} layout="vertical">
                  <Alert
                    message="配置建议"
                    description="较小的rank值训练速度快但表达能力有限，较大的rank值表达能力强但需要更多计算资源"
                    type="info"
                    showIcon
                    style={{ marginBottom: 24 }}
                  />

                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item 
                        label="LoRA Rank" 
                        name="loraRank" 
                        initialValue={16}
                        tooltip="低秩分解的维度，控制适配器的容量"
                      >
                        <Slider
                          min={1}
                          max={128}
                          marks={{
                            1: '1',
                            8: '8',
                            16: '16',
                            32: '32',
                            64: '64',
                            128: '128'
                          }}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item 
                        label="LoRA Alpha" 
                        name="loraAlpha" 
                        initialValue={32}
                        tooltip="缩放因子，通常设置为rank的2倍"
                      >
                        <Slider
                          min={1}
                          max={256}
                          marks={{
                            1: '1',
                            16: '16',
                            32: '32',
                            64: '64',
                            128: '128',
                            256: '256'
                          }}
                        />
                      </Form.Item>
                    </Col>
                  </Row>

                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item 
                        label="学习率" 
                        name="learningRate" 
                        initialValue="2e-4"
                      >
                        <Select>
                          <Option value="1e-5">1e-5 (保守)</Option>
                          <Option value="5e-5">5e-5</Option>
                          <Option value="1e-4">1e-4</Option>
                          <Option value="2e-4">2e-4 (推荐)</Option>
                          <Option value="3e-4">3e-4</Option>
                          <Option value="5e-4">5e-4 (激进)</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item 
                        label="Dropout率" 
                        name="loraDropout" 
                        initialValue={0.1}
                      >
                        <Slider
                          min={0}
                          max={0.5}
                          step={0.05}
                          marks={{
                            0: '0',
                            0.1: '0.1',
                            0.2: '0.2',
                            0.3: '0.3',
                            0.5: '0.5'
                          }}
                        />
                      </Form.Item>
                    </Col>
                  </Row>

                  <Form.Item label="目标模块" name="targetModules">
                    <Select mode="multiple" placeholder="选择要应用LoRA的模块">
                      <Option value="q_proj">查询投影 (q_proj)</Option>
                      <Option value="k_proj">键投影 (k_proj)</Option>
                      <Option value="v_proj">值投影 (v_proj)</Option>
                      <Option value="o_proj">输出投影 (o_proj)</Option>
                      <Option value="gate_proj">门投影 (gate_proj)</Option>
                      <Option value="up_proj">上投影 (up_proj)</Option>
                      <Option value="down_proj">下投影 (down_proj)</Option>
                    </Select>
                  </Form.Item>
                </Form>
              </TabPane>

              <TabPane tab="训练参数" key="training">
                <Form form={form} layout="vertical">
                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item label="训练轮数" name="epochs" initialValue={3}>
                        <Select>
                          <Option value={1}>1轮</Option>
                          <Option value={3}>3轮</Option>
                          <Option value={5}>5轮</Option>
                          <Option value={10}>10轮</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="批次大小" name="batchSize" initialValue={4}>
                        <Select>
                          <Option value={1}>1</Option>
                          <Option value={2}>2</Option>
                          <Option value={4}>4</Option>
                          <Option value={8}>8</Option>
                          <Option value={16}>16</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item label="梯度累积步数" name="gradientAccumulation" initialValue={4}>
                        <Select>
                          <Option value={1}>1</Option>
                          <Option value={2}>2</Option>
                          <Option value={4}>4</Option>
                          <Option value={8}>8</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                  </Row>

                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="最大序列长度" name="maxLength" initialValue={2048}>
                        <Select>
                          <Option value={512}>512</Option>
                          <Option value={1024}>1024</Option>
                          <Option value={2048}>2048</Option>
                          <Option value={4096}>4096</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="warmup步数" name="warmupSteps" initialValue={100}>
                        <Select>
                          <Option value={0}>0 (无warmup)</Option>
                          <Option value={50}>50</Option>
                          <Option value={100}>100</Option>
                          <Option value={500}>500</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                  </Row>

                  <Row gutter={16}>
                    <Col span={8}>
                      <Form.Item name="fp16" valuePropName="checked">
                        <Space>
                          <Switch size="small" />
                          <Text>启用FP16混合精度</Text>
                        </Space>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item name="gradientCheckpointing" valuePropName="checked">
                        <Space>
                          <Switch size="small" />
                          <Text>启用梯度检查点</Text>
                        </Space>
                      </Form.Item>
                    </Col>
                    <Col span={8}>
                      <Form.Item name="dataloader_pin_memory" valuePropName="checked">
                        <Space>
                          <Switch size="small" />
                          <Text>固定内存</Text>
                        </Space>
                      </Form.Item>
                    </Col>
                  </Row>
                </Form>
              </TabPane>

              <TabPane tab="量化设置" key="quantization">
                <Form form={form} layout="vertical">
                  <Alert
                    message="量化配置"
                    description="QLoRA使用4-bit量化显著减少内存使用，推荐用于大模型微调"
                    type="success"
                    showIcon
                    style={{ marginBottom: 24 }}
                  />

                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item label="量化类型" name="quantizationType" initialValue="none">
                        <Select>
                          <Option value="none">不使用量化</Option>
                          <Option value="int8">8-bit量化</Option>
                          <Option value="int4">4-bit量化</Option>
                          <Option value="nf4">NF4量化 (推荐)</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="计算数据类型" name="computeDtype" initialValue="bfloat16">
                        <Select>
                          <Option value="float16">float16</Option>
                          <Option value="bfloat16">bfloat16 (推荐)</Option>
                          <Option value="float32">float32</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                  </Row>

                  <Row gutter={16}>
                    <Col span={12}>
                      <Form.Item name="doubleQuantization" valuePropName="checked">
                        <Space>
                          <Switch size="small" />
                          <Text>启用双量化</Text>
                          <Tooltip title="进一步压缩量化常数">
                            <InfoCircleOutlined style={{ marginLeft: 4, color: '#1890ff' }} />
                          </Tooltip>
                        </Space>
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item name="nestedQuantization" valuePropName="checked">
                        <Space>
                          <Switch size="small" />
                          <Text>嵌套量化</Text>
                        </Space>
                      </Form.Item>
                    </Col>
                  </Row>

                  <Form.Item label="量化块大小" name="blockSize" initialValue={64}>
                    <Slider
                      min={16}
                      max={256}
                      step={16}
                      marks={{
                        16: '16',
                        64: '64',
                        128: '128',
                        256: '256'
                      }}
                    />
                  </Form.Item>
                </Form>
              </TabPane>
            </Tabs>
          </Card>
        </Col>

        <Col span={6}>
          <Card title="配置模板" size="small" style={{ marginBottom: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              {configTemplates.map(template => (
                <Button
                  key={template.name}
                  type={selectedTemplate === template.name ? 'primary' : 'default'}
                  block
                  size="small"
                  onClick={() => handleTemplateSelect(template.name)}
                >
                  {template.name}
                </Button>
              ))}
            </Space>
          </Card>

          <Card title="操作" size="small" style={{ marginBottom: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button 
                type="primary" 
                icon={<SaveOutlined />} 
                block
                onClick={handleSaveConfig}
              >
                保存配置
              </Button>
              <Button icon={<ExportOutlined />} block>
                导出配置
              </Button>
              <Button icon={<ImportOutlined />} block>
                导入配置
              </Button>
              <Button icon={<ReloadOutlined />} block>
                重置配置
              </Button>
            </Space>
          </Card>

          <Card title="配置验证" size="small">
            <Space direction="vertical" style={{ width: '100%' }} size="small">
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>参数合理性</Text>
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>内存估算</Text>
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>硬件兼容</Text>
                <CheckCircleOutlined style={{ color: '#52c41a' }} />
              </div>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  );
};

export default FineTuningConfigPage;