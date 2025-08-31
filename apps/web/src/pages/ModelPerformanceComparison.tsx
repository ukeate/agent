import React, { useState } from 'react';
import { 
  Card, 
  Table, 
  Row, 
  Col, 
  Typography, 
  Statistic, 
  Tag,
  Button,
  Space,
  Select,
  Checkbox,
  Alert,
  Descriptions,
  Progress,
  Tabs,
  Radio,
  Input
} from 'antd';
import {
  BarChartOutlined,
  LineChartOutlined,
  FileTextOutlined,
  TrophyOutlined,
  CompressOutlined,
  DownloadOutlined,
  ShareAltOutlined,
  ReloadOutlined,
  FilterOutlined
} from '@ant-design/icons';
import { Column, Line, Radar, Scatter } from '@ant-design/charts';

const { Title, Text } = Typography;
const { TabPane } = Tabs;
const { Option } = Select;

const ModelPerformanceComparison: React.FC = () => {
  const [selectedModels, setSelectedModels] = useState(['model-1', 'model-2', 'model-3']);
  const [comparisonType, setComparisonType] = useState('accuracy');
  const [activeTab, setActiveTab] = useState('metrics');

  // 模拟模型性能数据
  const modelData = [
    {
      id: 'model-1',
      name: 'llama2-lora-chat-v1',
      type: 'LoRA',
      baseModel: 'LLaMA 2 7B',
      accuracy: 87.5,
      bleu: 34.2,
      rouge: 45.8,
      perplexity: 12.3,
      latency: 45,
      throughput: 142,
      parameters: '23M',
      size: '92MB',
      trainingTime: '3.2h',
      dataset: '对话数据集',
      rank: 16,
      alpha: 32,
      status: '部署中'
    },
    {
      id: 'model-2',
      name: 'mistral-qlora-code-v2',
      type: 'QLoRA',
      baseModel: 'Mistral 7B',
      accuracy: 92.3,
      bleu: 41.7,
      rouge: 52.1,
      perplexity: 8.9,
      latency: 38,
      throughput: 156,
      parameters: '12M',
      size: '48MB',
      trainingTime: '2.8h',
      dataset: '代码生成数据',
      rank: 8,
      alpha: 16,
      status: '训练完成'
    },
    {
      id: 'model-3',
      name: 'qwen-lora-summary-v1',
      type: 'LoRA',
      baseModel: 'Qwen 14B',
      accuracy: 89.7,
      bleu: 38.5,
      rouge: 48.9,
      perplexity: 10.1,
      latency: 52,
      throughput: 128,
      parameters: '45M',
      size: '180MB',
      trainingTime: '4.1h',
      dataset: '摘要数据集',
      rank: 32,
      alpha: 64,
      status: '已部署'
    },
    {
      id: 'model-4',
      name: 'chatglm3-lora-qa-v1',
      type: 'LoRA',
      baseModel: 'ChatGLM3 6B',
      accuracy: 85.1,
      bleu: 32.8,
      rouge: 43.2,
      perplexity: 14.7,
      latency: 41,
      throughput: 138,
      parameters: '18M',
      size: '72MB',
      trainingTime: '2.9h',
      dataset: '问答数据集',
      rank: 12,
      alpha: 24,
      status: '训练完成'
    }
  ];

  const benchmarkData = [
    { task: 'MMLU', 'llama2-lora': 68.2, 'mistral-qlora': 74.5, 'qwen-lora': 71.8, 'chatglm3-lora': 66.9 },
    { task: 'HellaSwag', 'llama2-lora': 82.1, 'mistral-qlora': 87.3, 'qwen-lora': 85.2, 'chatglm3-lora': 79.8 },
    { task: 'ARC', 'llama2-lora': 78.9, 'mistral-qlora': 82.4, 'qwen-lora': 80.6, 'chatglm3-lora': 76.3 },
    { task: 'GSM8K', 'llama2-lora': 45.7, 'mistral-qlora': 52.3, 'qwen-lora': 49.1, 'chatglm3-lora': 43.2 },
    { task: 'HumanEval', 'llama2-lora': 23.4, 'mistral-qlora': 31.8, 'qwen-lora': 27.6, 'chatglm3-lora': 21.5 }
  ];

  const performanceColumns = [
    {
      title: '模型名称',
      dataIndex: 'name',
      key: 'name',
      fixed: 'left' as const,
      width: 200,
      render: (text: string, record: any) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{text}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            <Tag size="small" color={record.type === 'LoRA' ? 'blue' : 'purple'}>
              {record.type}
            </Tag>
            {record.baseModel}
          </div>
        </div>
      ),
    },
    {
      title: '准确率',
      dataIndex: 'accuracy',
      key: 'accuracy',
      sorter: (a: any, b: any) => a.accuracy - b.accuracy,
      render: (value: number) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{value}%</div>
          <Progress percent={value} size="small" showInfo={false} />
        </div>
      ),
    },
    {
      title: 'BLEU',
      dataIndex: 'bleu',
      key: 'bleu',
      sorter: (a: any, b: any) => a.bleu - b.bleu,
      render: (value: number) => `${value}`
    },
    {
      title: 'ROUGE-L',
      dataIndex: 'rouge',
      key: 'rouge',
      sorter: (a: any, b: any) => a.rouge - b.rouge,
      render: (value: number) => `${value}`
    },
    {
      title: '困惑度',
      dataIndex: 'perplexity',
      key: 'perplexity',
      sorter: (a: any, b: any) => a.perplexity - b.perplexity,
      render: (value: number) => (
        <span style={{ color: value < 10 ? '#52c41a' : value < 15 ? '#faad14' : '#ff4d4f' }}>
          {value}
        </span>
      )
    },
    {
      title: '延迟',
      dataIndex: 'latency',
      key: 'latency',
      sorter: (a: any, b: any) => a.latency - b.latency,
      render: (value: number) => `${value}ms`
    },
    {
      title: '吞吐量',
      dataIndex: 'throughput',
      key: 'throughput',
      sorter: (a: any, b: any) => a.throughput - b.throughput,
      render: (value: number) => `${value} tok/s`
    },
    {
      title: '参数量',
      dataIndex: 'parameters',
      key: 'parameters',
    },
    {
      title: '模型大小',
      dataIndex: 'size',
      key: 'size',
    },
    {
      title: '训练时间',
      dataIndex: 'trainingTime',
      key: 'trainingTime',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colorMap: Record<string, string> = {
          '训练完成': 'green',
          '部署中': 'processing',
          '已部署': 'success'
        };
        return <Tag color={colorMap[status]}>{status}</Tag>;
      },
    }
  ];

  const getComparisonChart = () => {
    const selectedData = modelData.filter(m => selectedModels.includes(m.id));
    
    switch (comparisonType) {
      case 'accuracy':
        return (
          <Column
            data={selectedData}
            xField="name"
            yField="accuracy"
            color="#1890ff"
            meta={{
              accuracy: { alias: '准确率 (%)' }
            }}
            height={300}
          />
        );
      case 'performance':
        const perfData = selectedData.map(m => ({
          name: m.name,
          latency: m.latency,
          throughput: m.throughput
        }));
        return (
          <Scatter
            data={perfData}
            xField="latency"
            yField="throughput"
            colorField="name"
            size={5}
            meta={{
              latency: { alias: '延迟 (ms)' },
              throughput: { alias: '吞吐量 (tok/s)' }
            }}
            height={300}
          />
        );
      case 'radar':
        const radarData = selectedData.flatMap(m => [
          { name: m.name, metric: '准确率', value: m.accuracy },
          { name: m.name, metric: 'BLEU', value: m.bleu * 2 },
          { name: m.name, metric: 'ROUGE', value: m.rouge * 1.8 },
          { name: m.name, metric: '吞吐量', value: m.throughput * 0.6 },
          { name: m.name, metric: '效率', value: (1000 / m.latency) * 20 }
        ]);
        return (
          <Radar
            data={radarData}
            xField="metric"
            yField="value"
            seriesField="name"
            height={300}
            area={{}}
            point={{
              size: 2,
            }}
          />
        );
      default:
        return null;
    }
  };

  const bestModel = modelData.reduce((best, current) => 
    current.accuracy > best.accuracy ? current : best
  );

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <BarChartOutlined style={{ marginRight: 8, color: '#722ed1' }} />
          模型性能对比分析
        </Title>
        <Text type="secondary">
          全面对比和分析不同LoRA/QLoRA模型的性能表现和特征
        </Text>
      </div>

      {/* 快速概览 */}
      <Row gutter={16} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="参与对比模型"
              value={selectedModels.length}
              prefix={<CompressOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="最佳性能"
              value={bestModel.accuracy}
              suffix="%"
              prefix={<TrophyOutlined />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="平均延迟"
              value={Math.round(modelData.reduce((sum, m) => sum + m.latency, 0) / modelData.length)}
              suffix="ms"
              prefix={<LineChartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="总参数量"
              value="98M"
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Card>
        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="性能指标" key="metrics">
            <div style={{ marginBottom: 16 }}>
              <Alert
                message="最佳模型推荐"
                description={
                  <div>
                    <Text strong>{bestModel.name}</Text> 在准确率方面表现最佳 ({bestModel.accuracy}%)，
                    使用 {bestModel.type} 技术，基于 {bestModel.baseModel}，
                    训练时间 {bestModel.trainingTime}，推理延迟 {bestModel.latency}ms。
                  </div>
                }
                type="success"
                showIcon
                style={{ marginBottom: 16 }}
              />
              
              <Space style={{ marginBottom: 16 }}>
                <Text strong>模型筛选:</Text>
                <Checkbox.Group 
                  value={selectedModels}
                  onChange={setSelectedModels}
                >
                  {modelData.map(model => (
                    <Checkbox key={model.id} value={model.id}>
                      {model.name}
                    </Checkbox>
                  ))}
                </Checkbox.Group>
              </Space>
            </div>
            
            <Table 
              columns={performanceColumns}
              dataSource={modelData}
              rowKey="id"
              scroll={{ x: 1500 }}
              size="small"
              rowSelection={{
                selectedRowKeys: selectedModels,
                onChange: setSelectedModels,
              }}
            />
          </TabPane>

          <TabPane tab="可视化对比" key="visualization">
            <Row gutter={16}>
              <Col span={8}>
                <Card title="对比选项" size="small">
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>选择模型:</Text>
                    <Select
                      mode="multiple"
                      value={selectedModels}
                      onChange={setSelectedModels}
                      style={{ width: '100%', marginTop: 8 }}
                      placeholder="选择要对比的模型"
                    >
                      {modelData.map(model => (
                        <Option key={model.id} value={model.id}>
                          {model.name}
                        </Option>
                      ))}
                    </Select>
                  </div>
                  
                  <div style={{ marginBottom: 16 }}>
                    <Text strong>对比维度:</Text>
                    <Radio.Group 
                      value={comparisonType}
                      onChange={e => setComparisonType(e.target.value)}
                      style={{ marginTop: 8, width: '100%' }}
                    >
                      <Radio.Button value="accuracy">准确率</Radio.Button>
                      <Radio.Button value="performance">性能</Radio.Button>
                      <Radio.Button value="radar">雷达图</Radio.Button>
                    </Radio.Group>
                  </div>
                  
                  <Button type="primary" icon={<ReloadOutlined />} block>
                    更新图表
                  </Button>
                </Card>
              </Col>
              
              <Col span={16}>
                <Card title="模型对比图表" size="small">
                  {getComparisonChart()}
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="基准测试" key="benchmark">
            <Card title="标准基准测试结果" size="small" style={{ marginBottom: 16 }}>
              <Table
                columns={[
                  { title: '测试任务', dataIndex: 'task', key: 'task', width: 120 },
                  { 
                    title: 'LLaMA2-LoRA', 
                    dataIndex: 'llama2-lora', 
                    key: 'llama2-lora',
                    render: (value: number) => (
                      <div>
                        <Text>{value}</Text>
                        <Progress percent={value} size="small" showInfo={false} />
                      </div>
                    )
                  },
                  { 
                    title: 'Mistral-QLoRA', 
                    dataIndex: 'mistral-qlora', 
                    key: 'mistral-qlora',
                    render: (value: number) => (
                      <div>
                        <Text>{value}</Text>
                        <Progress percent={value} size="small" showInfo={false} />
                      </div>
                    )
                  },
                  { 
                    title: 'Qwen-LoRA', 
                    dataIndex: 'qwen-lora', 
                    key: 'qwen-lora',
                    render: (value: number) => (
                      <div>
                        <Text>{value}</Text>
                        <Progress percent={value} size="small" showInfo={false} />
                      </div>
                    )
                  },
                  { 
                    title: 'ChatGLM3-LoRA', 
                    dataIndex: 'chatglm3-lora', 
                    key: 'chatglm3-lora',
                    render: (value: number) => (
                      <div>
                        <Text>{value}</Text>
                        <Progress percent={value} size="small" showInfo={false} />
                      </div>
                    )
                  }
                ]}
                dataSource={benchmarkData}
                rowKey="task"
                pagination={false}
                size="small"
              />
            </Card>
            
            <Row gutter={16}>
              <Col span={12}>
                <Card title="任务类型分析" size="small">
                  <Descriptions bordered size="small">
                    <Descriptions.Item label="语言理解">
                      <Text>MMLU, HellaSwag - 测试模型的语言理解和常识推理能力</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="逻辑推理">
                      <Text>ARC - 测试科学推理和逻辑思维能力</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="数学计算">
                      <Text>GSM8K - 测试小学数学问题求解能力</Text>
                    </Descriptions.Item>
                    <Descriptions.Item label="代码生成">
                      <Text>HumanEval - 测试Python代码生成和理解能力</Text>
                    </Descriptions.Item>
                  </Descriptions>
                </Card>
              </Col>
              
              <Col span={12}>
                <Card title="综合评分" size="small">
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                      <Text>Mistral-QLoRA</Text>
                      <Text strong style={{ color: '#52c41a' }}>85.7</Text>
                    </div>
                    <Progress percent={85.7} strokeColor="#52c41a" size="small" />
                  </div>
                  
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                      <Text>Qwen-LoRA</Text>
                      <Text strong style={{ color: '#1890ff' }}>82.9</Text>
                    </div>
                    <Progress percent={82.9} strokeColor="#1890ff" size="small" />
                  </div>
                  
                  <div style={{ marginBottom: 16 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                      <Text>LLaMA2-LoRA</Text>
                      <Text strong style={{ color: '#faad14' }}>79.7</Text>
                    </div>
                    <Progress percent={79.7} strokeColor="#faad14" size="small" />
                  </div>
                  
                  <div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                      <Text>ChatGLM3-LoRA</Text>
                      <Text strong style={{ color: '#722ed1' }}>77.5</Text>
                    </div>
                    <Progress percent={77.5} strokeColor="#722ed1" size="small" />
                  </div>
                </Card>
              </Col>
            </Row>
          </TabPane>

          <TabPane tab="详细报告" key="report">
            <Row gutter={16}>
              <Col span={16}>
                <Card title="性能分析报告" size="small">
                  <div style={{ padding: '16px', backgroundColor: '#f9f9f9', borderRadius: 6, marginBottom: 16 }}>
                    <Title level={4}>执行摘要</Title>
                    <Text>
                      本次对比分析了4个不同的LoRA/QLoRA微调模型，涵盖了从7B到14B参数规模的不同基座模型。
                      通过多维度性能评估，Mistral-QLoRA模型在综合性能上表现最优，特别是在代码生成和数学推理任务上有显著优势。
                    </Text>
                  </div>
                  
                  <div style={{ marginBottom: 16 }}>
                    <Title level={5}>关键发现</Title>
                    <ul>
                      <li><strong>准确率排名:</strong> Mistral-QLoRA (92.3%) {'>'}  Qwen-LoRA (89.7%) {'>'} LLaMA2-LoRA (87.5%) {'>'} ChatGLM3-LoRA (85.1%)</li>
                      <li><strong>效率分析:</strong> QLoRA技术相比LoRA在保持性能的同时显著减少了参数量和存储需求</li>
                      <li><strong>延迟对比:</strong> Mistral-QLoRA在高性能的同时保持了较低的推理延迟(38ms)</li>
                      <li><strong>适用场景:</strong> 不同模型在特定任务上各有优势，需根据具体应用需求选择</li>
                    </ul>
                  </div>
                  
                  <div style={{ marginBottom: 16 }}>
                    <Title level={5}>技术分析</Title>
                    <Row gutter={16}>
                      <Col span={12}>
                        <Card size="small" title="LoRA vs QLoRA">
                          <ul style={{ fontSize: '12px' }}>
                            <li>QLoRA在相同性能下参数量减少50%</li>
                            <li>LoRA在大参数量下表现更稳定</li>
                            <li>QLoRA训练时间平均节省15%</li>
                            <li>内存使用QLoRA比LoRA节省30-40%</li>
                          </ul>
                        </Card>
                      </Col>
                      <Col span={12}>
                        <Card size="small" title="基座模型影响">
                          <ul style={{ fontSize: '12px' }}>
                            <li>14B模型在复杂推理任务上优势明显</li>
                            <li>7B模型在对话任务上性价比更高</li>
                            <li>Mistral架构对代码任务适应性更强</li>
                            <li>ChatGLM在中文处理上有特定优势</li>
                          </ul>
                        </Card>
                      </Col>
                    </Row>
                  </div>
                  
                  <div>
                    <Title level={5}>推荐建议</Title>
                    <Alert
                      type="info"
                      message="模型选择建议"
                      description={
                        <ul style={{ marginTop: 8 }}>
                          <li><strong>代码生成任务:</strong> 推荐使用Mistral-QLoRA，在HumanEval上得分31.8</li>
                          <li><strong>对话系统:</strong> 推荐LLaMA2-LoRA，平衡性能和资源消耗</li>
                          <li><strong>文本摘要:</strong> 推荐Qwen-LoRA，在ROUGE指标上表现优异</li>
                          <li><strong>资源受限环境:</strong> 推荐QLoRA技术，显著降低资源需求</li>
                        </ul>
                      }
                    />
                  </div>
                </Card>
              </Col>
              
              <Col span={8}>
                <Card title="报告操作" size="small" style={{ marginBottom: 16 }}>
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Button type="primary" icon={<DownloadOutlined />} block>
                      下载完整报告
                    </Button>
                    <Button icon={<ShareAltOutlined />} block>
                      分享报告
                    </Button>
                    <Button icon={<FilterOutlined />} block>
                      自定义报告
                    </Button>
                  </Space>
                </Card>
                
                <Card title="报告配置" size="small">
                  <div style={{ marginBottom: 12 }}>
                    <Text>报告格式:</Text>
                    <Select defaultValue="pdf" style={{ width: '100%', marginTop: 4 }}>
                      <Option value="pdf">PDF</Option>
                      <Option value="html">HTML</Option>
                      <Option value="excel">Excel</Option>
                    </Select>
                  </div>
                  
                  <div style={{ marginBottom: 12 }}>
                    <Text>包含内容:</Text>
                    <Checkbox.Group style={{ marginTop: 4, width: '100%' }} defaultValue={['metrics', 'charts', 'recommendations']}>
                      <div><Checkbox value="metrics">性能指标</Checkbox></div>
                      <div><Checkbox value="charts">图表分析</Checkbox></div>
                      <div><Checkbox value="benchmark">基准测试</Checkbox></div>
                      <div><Checkbox value="recommendations">推荐建议</Checkbox></div>
                    </Checkbox.Group>
                  </div>
                  
                  <div>
                    <Text>报告标题:</Text>
                    <Input 
                      defaultValue="LoRA/QLoRA模型性能对比分析报告"
                      style={{ marginTop: 4 }}
                    />
                  </div>
                </Card>
              </Col>
            </Row>
          </TabPane>
        </Tabs>
      </Card>
    </div>
  );
};

export default ModelPerformanceComparison;