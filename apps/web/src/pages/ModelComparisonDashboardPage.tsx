import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Table, Button, Select, DatePicker, Radio, Tabs, Space, Statistic, Tag, Progress, Tooltip, Switch, Modal, Form, Input, message, Divider } from 'antd';
import { CompareOutlined, BarChartOutlined, LineChartOutlined, RadarChartOutlined, TableOutlined, FileTextOutlined, DownloadOutlined, ShareAltOutlined, FilterOutlined, EyeOutlined } from '@ant-design/icons';
import { Line, Bar, Radar, Scatter } from 'react-chartjs-2';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';

const { RangePicker } = DatePicker;
const { Option } = Select;
const { TabPane } = Tabs;

// 数据接口定义
interface ModelComparison {
  id: string;
  name: string;
  models: string[];
  benchmarks: string[];
  metrics: string[];
  status: 'preparing' | 'running' | 'completed' | 'failed';
  createdAt: string;
  completedAt?: string;
  results?: ComparisonResult[];
}

interface ComparisonResult {
  modelId: string;
  modelName: string;
  benchmark: string;
  metric: string;
  value: number;
  rank: number;
  percentile: number;
  confidence: number;
}

interface ModelMetadata {
  id: string;
  name: string;
  version: string;
  type: 'transformer' | 'gpt' | 'bert' | 'llama' | 'custom';
  size: string;
  parameters: string;
  developer: string;
  releaseDate: string;
  description: string;
}

interface BenchmarkMetadata {
  id: string;
  name: string;
  category: string;
  description: string;
  metrics: string[];
  samples: number;
}

const ModelComparisonDashboardPage: React.FC = () => {
  // 状态管理
  const [comparisons, setComparisons] = useState<ModelComparison[]>([]);
  const [models, setModels] = useState<ModelMetadata[]>([]);
  const [benchmarks, setBenchmarks] = useState<BenchmarkMetadata[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedComparison, setSelectedComparison] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<'table' | 'chart'>('table');
  const [chartType, setChartType] = useState<'line' | 'bar' | 'radar' | 'scatter'>('bar');
  
  // 过滤状态
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>([]);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const [dateRange, setDateRange] = useState<[dayjs.Dayjs, dayjs.Dayjs] | null>(null);

  // 模态框状态
  const [isComparisonModalVisible, setIsComparisonModalVisible] = useState(false);
  const [isDetailModalVisible, setIsDetailModalVisible] = useState(false);
  
  // 表单
  const [comparisonForm] = Form.useForm();

  // 模拟数据加载
  useEffect(() => {
    loadMockData();
  }, []);

  const loadMockData = () => {
    const mockModels: ModelMetadata[] = [
      {
        id: '1',
        name: 'GPT-4',
        version: '0.0.1',
        type: 'gpt',
        size: '175B',
        parameters: '175B',
        developer: 'OpenAI',
        releaseDate: '2023-03-14',
        description: '大型语言模型，具备强大的理解和生成能力',
      },
      {
        id: '2',
        name: 'Claude-3.5-Sonnet',
        version: '20241022',
        type: 'transformer',
        size: 'Unknown',
        parameters: 'Unknown',
        developer: 'Anthropic',
        releaseDate: '2024-10-22',
        description: '高性能对话模型，在推理和创作方面表现优异',
      },
      {
        id: '3',
        name: 'Llama-2-70B',
        version: '70B-Chat',
        type: 'llama',
        size: '70B',
        parameters: '70B',
        developer: 'Meta',
        releaseDate: '2023-07-18',
        description: '开源大型语言模型，适合对话和指令遵循',
      },
    ];

    const mockBenchmarks: BenchmarkMetadata[] = [
      {
        id: '1',
        name: 'GLUE',
        category: 'NLU',
        description: '通用语言理解评估基准',
        metrics: ['accuracy', 'f1_score', 'matthews_correlation'],
        samples: 67349,
      },
      {
        id: '2',
        name: 'SuperGLUE',
        category: 'NLU',
        description: '更具挑战性的语言理解基准',
        metrics: ['accuracy', 'f1_score'],
        samples: 22200,
      },
      {
        id: '3',
        name: 'MMLU',
        category: 'Knowledge',
        description: '大规模多任务语言理解',
        metrics: ['accuracy'],
        samples: 15908,
      },
    ];

    const mockComparisons: ModelComparison[] = [
      {
        id: '1',
        name: '大型语言模型综合对比',
        models: ['GPT-4', 'Claude-3.5-Sonnet', 'Llama-2-70B'],
        benchmarks: ['GLUE', 'SuperGLUE', 'MMLU'],
        metrics: ['accuracy', 'f1_score'],
        status: 'completed',
        createdAt: '2024-01-15T10:00:00',
        completedAt: '2024-01-15T12:30:00',
        results: [
          { modelId: '1', modelName: 'GPT-4', benchmark: 'GLUE', metric: 'accuracy', value: 0.925, rank: 1, percentile: 95, confidence: 0.98 },
          { modelId: '2', modelName: 'Claude-3.5-Sonnet', benchmark: 'GLUE', metric: 'accuracy', value: 0.918, rank: 2, percentile: 88, confidence: 0.97 },
          { modelId: '3', modelName: 'Llama-2-70B', benchmark: 'GLUE', metric: 'accuracy', value: 0.883, rank: 3, percentile: 75, confidence: 0.95 },
          { modelId: '1', modelName: 'GPT-4', benchmark: 'SuperGLUE', metric: 'accuracy', value: 0.895, rank: 1, percentile: 92, confidence: 0.96 },
          { modelId: '2', modelName: 'Claude-3.5-Sonnet', benchmark: 'SuperGLUE', metric: 'accuracy', value: 0.887, rank: 2, percentile: 85, confidence: 0.94 },
          { modelId: '3', modelName: 'Llama-2-70B', benchmark: 'SuperGLUE', metric: 'accuracy', value: 0.845, rank: 3, percentile: 68, confidence: 0.92 },
        ],
      },
      {
        id: '2',
        name: '知识理解能力测试',
        models: ['GPT-4', 'Claude-3.5-Sonnet'],
        benchmarks: ['MMLU'],
        metrics: ['accuracy'],
        status: 'running',
        createdAt: '2024-01-20T09:15:00',
      },
    ];

    setModels(mockModels);
    setBenchmarks(mockBenchmarks);
    setComparisons(mockComparisons);
    setSelectedComparison(mockComparisons[0].id);
  };

  // 获取当前选择的对比结果
  const getCurrentComparison = () => {
    return comparisons.find(c => c.id === selectedComparison);
  };

  const getCurrentResults = () => {
    const comparison = getCurrentComparison();
    return comparison?.results || [];
  };

  // 处理新建对比
  const handleCreateComparison = async () => {
    try {
      const values = await comparisonForm.validateFields();
      
      const newComparison: ModelComparison = {
        id: Date.now().toString(),
        name: values.name,
        models: values.models,
        benchmarks: values.benchmarks,
        metrics: values.metrics,
        status: 'preparing',
        createdAt: new Date().toISOString(),
      };

      setComparisons([newComparison, ...comparisons]);
      message.success('模型对比任务已创建');
      setIsComparisonModalVisible(false);
      comparisonForm.resetFields();
      
      // 模拟开始运行
      setTimeout(() => {
        setComparisons(prev => prev.map(c => 
          c.id === newComparison.id 
            ? { ...c, status: 'running' }
            : c
        ));
      }, 1000);

    } catch (error) {
      message.error('创建失败，请检查输入');
    }
  };

  // 生成图表数据
  const generateChartData = () => {
    const results = getCurrentResults();
    const comparison = getCurrentComparison();
    
    if (!results.length || !comparison) return null;

    const models = Array.from(new Set(results.map(r => r.modelName)));
    const benchmarks = Array.from(new Set(results.map(r => r.benchmark)));
    const metrics = Array.from(new Set(results.map(r => r.metric)));

    if (chartType === 'bar') {
      return {
        labels: benchmarks,
        datasets: models.map((model, index) => ({
          label: model,
          data: benchmarks.map(benchmark => {
            const result = results.find(r => r.modelName === model && r.benchmark === benchmark);
            return result ? result.value : 0;
          }),
          backgroundColor: `hsl(${index * 137.5}, 70%, 60%)`,
          borderColor: `hsl(${index * 137.5}, 70%, 50%)`,
          borderWidth: 1,
        })),
      };
    }

    if (chartType === 'line') {
      return {
        labels: benchmarks,
        datasets: models.map((model, index) => ({
          label: model,
          data: benchmarks.map(benchmark => {
            const result = results.find(r => r.modelName === model && r.benchmark === benchmark);
            return result ? result.value : 0;
          }),
          borderColor: `hsl(${index * 137.5}, 70%, 50%)`,
          backgroundColor: `hsl(${index * 137.5}, 70%, 60%)`,
          tension: 0.4,
        })),
      };
    }

    if (chartType === 'radar') {
      return {
        labels: benchmarks,
        datasets: models.map((model, index) => ({
          label: model,
          data: benchmarks.map(benchmark => {
            const result = results.find(r => r.modelName === model && r.benchmark === benchmark);
            return result ? result.value * 100 : 0;
          }),
          borderColor: `hsl(${index * 137.5}, 70%, 50%)`,
          backgroundColor: `hsla(${index * 137.5}, 70%, 60%, 0.2)`,
        })),
      };
    }

    return null;
  };

  // 表格列定义
  const comparisonColumns: ColumnsType<ModelComparison> = [
    {
      title: '对比名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: ModelComparison) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            创建时间: {dayjs(record.createdAt).format('YYYY-MM-DD HH:mm')}
          </div>
        </div>
      ),
    },
    {
      title: '参与模型',
      dataIndex: 'models',
      key: 'models',
      render: (models: string[]) => (
        <div>
          {models.map(model => (
            <Tag key={model} style={{ marginBottom: '2px' }}>{model}</Tag>
          ))}
        </div>
      ),
    },
    {
      title: '测试基准',
      dataIndex: 'benchmarks',
      key: 'benchmarks',
      render: (benchmarks: string[]) => (
        <div>
          {benchmarks.map(benchmark => (
            <Tag key={benchmark} color="blue" style={{ marginBottom: '2px' }}>
              {benchmark}
            </Tag>
          ))}
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string, record: ModelComparison) => {
        const colors = {
          preparing: 'orange',
          running: 'blue',
          completed: 'green',
          failed: 'red',
        };
        const labels = {
          preparing: '准备中',
          running: '运行中',
          completed: '已完成',
          failed: '失败',
        };
        
        return (
          <div>
            <Tag color={colors[status as keyof typeof colors]}>
              {labels[status as keyof typeof labels]}
            </Tag>
            {record.completedAt && (
              <div style={{ fontSize: '10px', color: '#999', marginTop: '2px' }}>
                完成时间: {dayjs(record.completedAt).format('MM-DD HH:mm')}
              </div>
            )}
          </div>
        );
      },
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: ModelComparison) => (
        <Space>
          <Tooltip title="查看详情">
            <Button 
              type="text" 
              icon={<EyeOutlined />} 
              size="small"
              onClick={() => setSelectedComparison(record.id)}
            />
          </Tooltip>
          {record.status === 'completed' && (
            <>
              <Tooltip title="导出报告">
                <Button type="text" icon={<DownloadOutlined />} size="small" />
              </Tooltip>
              <Tooltip title="分享">
                <Button type="text" icon={<ShareAltOutlined />} size="small" />
              </Tooltip>
            </>
          )}
        </Space>
      ),
    },
  ];

  const resultColumns: ColumnsType<ComparisonResult> = [
    {
      title: '模型',
      dataIndex: 'modelName',
      key: 'modelName',
      render: (text: string, record: ComparisonResult) => (
        <div>
          <div style={{ fontWeight: 500 }}>{text}</div>
          <div style={{ fontSize: '10px', color: '#666' }}>
            排名: #{record.rank}
          </div>
        </div>
      ),
    },
    {
      title: '基准测试',
      dataIndex: 'benchmark',
      key: 'benchmark',
      render: (text: string) => <Tag color="blue">{text}</Tag>,
    },
    {
      title: '指标',
      dataIndex: 'metric',
      key: 'metric',
    },
    {
      title: '得分',
      dataIndex: 'value',
      key: 'value',
      render: (value: number) => (
        <div>
          <div style={{ fontWeight: 500 }}>
            {(value * 100).toFixed(2)}%
          </div>
        </div>
      ),
      sorter: (a, b) => a.value - b.value,
    },
    {
      title: '排名',
      dataIndex: 'rank',
      key: 'rank',
      render: (rank: number, record: ComparisonResult) => (
        <div>
          <Tag 
            color={rank === 1 ? 'gold' : rank === 2 ? 'silver' : rank === 3 ? 'orange' : 'default'}
          >
            #{rank}
          </Tag>
          <div style={{ fontSize: '10px', color: '#666', marginTop: '2px' }}>
            {record.percentile}th percentile
          </div>
        </div>
      ),
    },
    {
      title: '置信度',
      dataIndex: 'confidence',
      key: 'confidence',
      render: (confidence: number) => (
        <Progress 
          percent={confidence * 100}
          size="small"
          showInfo={false}
          strokeColor={confidence > 0.95 ? '#52c41a' : confidence > 0.9 ? '#faad14' : '#ff4d4f'}
        />
      ),
    },
  ];

  const chartData = generateChartData();
  const currentComparison = getCurrentComparison();
  const currentResults = getCurrentResults();

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h1>模型对比分析</h1>
          <p>多维度对比分析不同模型在各种基准测试中的表现</p>
        </div>
        <Button 
          type="primary" 
          icon={<CompareOutlined />}
          onClick={() => setIsComparisonModalVisible(true)}
        >
          新建对比分析
        </Button>
      </div>

      <Row gutter={[16, 16]}>
        {/* 左侧对比列表 */}
        <Col span={8}>
          <Card title="对比任务列表" size="small">
            <Table
              columns={comparisonColumns}
              dataSource={comparisons}
              rowKey="id"
              size="small"
              pagination={false}
              rowSelection={{
                type: 'radio',
                selectedRowKeys: selectedComparison ? [selectedComparison] : [],
                onChange: (selectedRowKeys) => {
                  setSelectedComparison(selectedRowKeys[0] as string);
                },
              }}
            />
          </Card>
        </Col>

        {/* 右侧详情面板 */}
        <Col span={16}>
          {currentComparison ? (
            <Card 
              title={currentComparison.name}
              extra={
                <Space>
                  <Radio.Group 
                    value={viewMode} 
                    onChange={(e) => setViewMode(e.target.value)}
                    size="small"
                  >
                    <Radio.Button value="table">
                      <TableOutlined /> 表格视图
                    </Radio.Button>
                    <Radio.Button value="chart">
                      <BarChartOutlined /> 图表视图
                    </Radio.Button>
                  </Radio.Group>
                </Space>
              }
            >
              {currentComparison.status === 'completed' ? (
                <>
                  {viewMode === 'table' ? (
                    <Table
                      columns={resultColumns}
                      dataSource={currentResults}
                      rowKey={(record) => `${record.modelId}-${record.benchmark}-${record.metric}`}
                      pagination={{ pageSize: 10 }}
                      size="small"
                    />
                  ) : (
                    <div>
                      <div style={{ marginBottom: '16px' }}>
                        <Space>
                          图表类型:
                          <Select
                            value={chartType}
                            onChange={setChartType}
                            style={{ width: 120 }}
                            size="small"
                          >
                            <Option value="bar">柱状图</Option>
                            <Option value="line">折线图</Option>
                            <Option value="radar">雷达图</Option>
                          </Select>
                        </Space>
                      </div>
                      
                      <div style={{ height: '400px' }}>
                        {chartData && chartType === 'bar' && (
                          <Bar 
                            data={chartData} 
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              plugins: {
                                legend: {
                                  position: 'top' as const,
                                },
                                title: {
                                  display: true,
                                  text: '模型性能对比 - 柱状图',
                                },
                              },
                              scales: {
                                y: {
                                  beginAtZero: true,
                                  max: 1,
                                },
                              },
                            }}
                          />
                        )}
                        
                        {chartData && chartType === 'line' && (
                          <Line 
                            data={chartData} 
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              plugins: {
                                legend: {
                                  position: 'top' as const,
                                },
                                title: {
                                  display: true,
                                  text: '模型性能对比 - 折线图',
                                },
                              },
                              scales: {
                                y: {
                                  beginAtZero: true,
                                  max: 1,
                                },
                              },
                            }}
                          />
                        )}
                        
                        {chartData && chartType === 'radar' && (
                          <Radar 
                            data={chartData} 
                            options={{
                              responsive: true,
                              maintainAspectRatio: false,
                              plugins: {
                                legend: {
                                  position: 'top' as const,
                                },
                                title: {
                                  display: true,
                                  text: '模型性能对比 - 雷达图',
                                },
                              },
                              scales: {
                                r: {
                                  beginAtZero: true,
                                  max: 100,
                                },
                              },
                            }}
                          />
                        )}
                      </div>
                    </div>
                  )}

                  <Divider />
                  
                  {/* 统计摘要 */}
                  <Row gutter={16}>
                    <Col span={6}>
                      <Statistic
                        title="参与模型"
                        value={currentComparison.models.length}
                        suffix="个"
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="测试基准"
                        value={currentComparison.benchmarks.length}
                        suffix="项"
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="评估指标"
                        value={currentComparison.metrics.length}
                        suffix="项"
                      />
                    </Col>
                    <Col span={6}>
                      <Statistic
                        title="测试结果"
                        value={currentResults.length}
                        suffix="条"
                      />
                    </Col>
                  </Row>
                </>
              ) : (
                <div style={{ textAlign: 'center', padding: '40px' }}>
                  <div style={{ fontSize: '16px', marginBottom: '16px' }}>
                    {currentComparison.status === 'preparing' && '正在准备对比任务...'}
                    {currentComparison.status === 'running' && '正在运行对比分析...'}
                    {currentComparison.status === 'failed' && '对比任务执行失败'}
                  </div>
                  {currentComparison.status !== 'failed' && (
                    <Progress percent={currentComparison.status === 'preparing' ? 25 : 60} />
                  )}
                </div>
              )}
            </Card>
          ) : (
            <Card>
              <div style={{ textAlign: 'center', padding: '40px', color: '#999' }}>
                请选择一个对比任务查看详情
              </div>
            </Card>
          )}
        </Col>
      </Row>

      {/* 新建对比模态框 */}
      <Modal
        title="创建模型对比分析"
        open={isComparisonModalVisible}
        onOk={handleCreateComparison}
        onCancel={() => {
          setIsComparisonModalVisible(false);
          comparisonForm.resetFields();
        }}
        width={800}
      >
        <Form
          form={comparisonForm}
          layout="vertical"
        >
          <Form.Item
            name="name"
            label="对比名称"
            rules={[{ required: true, message: '请输入对比名称' }]}
          >
            <Input placeholder="输入对比分析名称" />
          </Form.Item>

          <Form.Item
            name="models"
            label="选择模型"
            rules={[{ required: true, message: '请选择要对比的模型' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择要对比的模型"
              style={{ width: '100%' }}
            >
              {models.map(model => (
                <Option key={model.id} value={model.name}>
                  {model.name} ({model.type})
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="benchmarks"
            label="选择基准测试"
            rules={[{ required: true, message: '请选择基准测试' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择基准测试"
              style={{ width: '100%' }}
            >
              {benchmarks.map(benchmark => (
                <Option key={benchmark.id} value={benchmark.name}>
                  {benchmark.name} ({benchmark.category})
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="metrics"
            label="选择评估指标"
            rules={[{ required: true, message: '请选择评估指标' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择评估指标"
              style={{ width: '100%' }}
            >
              <Option value="accuracy">准确率 (Accuracy)</Option>
              <Option value="f1_score">F1分数</Option>
              <Option value="precision">精确率 (Precision)</Option>
              <Option value="recall">召回率 (Recall)</Option>
              <Option value="bleu">BLEU分数</Option>
              <Option value="rouge">ROUGE分数</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default ModelComparisonDashboardPage;