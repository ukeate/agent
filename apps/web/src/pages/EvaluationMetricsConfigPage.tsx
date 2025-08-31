import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Table, Button, Modal, Form, Input, Select, Tag, Space, Switch, message, Tooltip, Popconfirm, InputNumber, Slider, Tabs, Alert, Badge } from 'antd';
import { SlidersOutlined, PlusOutlined, EditOutlined, DeleteOutlined, SettingOutlined, CheckCircleOutlined, InfoCircleOutlined, ExperimentOutlined, FunctionOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';

const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;

// 评估指标定义
interface EvaluationMetric {
  id: string;
  name: string;
  displayName: string;
  category: 'accuracy' | 'fluency' | 'coherence' | 'diversity' | 'safety' | 'efficiency';
  description: string;
  formula: string;
  dataType: 'classification' | 'regression' | 'text_generation' | 'ranking' | 'multi_choice';
  range: {
    min: number;
    max: number;
  };
  higherBetter: boolean;
  status: 'active' | 'disabled' | 'deprecated';
  weight: number;
  threshold: {
    excellent: number;
    good: number;
    fair: number;
  };
  aggregationMethod: 'mean' | 'median' | 'weighted_mean' | 'max' | 'min';
  lastUpdated: string;
}

// 指标配置组合
interface MetricsConfiguration {
  id: string;
  name: string;
  description: string;
  metrics: Array<{
    metricId: string;
    weight: number;
    threshold?: number;
  }>;
  taskType: string[];
  status: 'active' | 'draft' | 'deprecated';
  createdAt: string;
  usageCount: number;
}

const EvaluationMetricsConfigPage: React.FC = () => {
  // 状态管理
  const [metrics, setMetrics] = useState<EvaluationMetric[]>([]);
  const [configurations, setConfigurations] = useState<MetricsConfiguration[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('metrics');

  // 模态框状态
  const [isMetricModalVisible, setIsMetricModalVisible] = useState(false);
  const [isConfigModalVisible, setIsConfigModalVisible] = useState(false);
  const [editingMetric, setEditingMetric] = useState<EvaluationMetric | null>(null);
  const [editingConfig, setEditingConfig] = useState<MetricsConfiguration | null>(null);

  // 表单
  const [metricForm] = Form.useForm();
  const [configForm] = Form.useForm();

  // 模拟数据加载
  useEffect(() => {
    loadMockData();
  }, []);

  const loadMockData = () => {
    const mockMetrics: EvaluationMetric[] = [
      {
        id: '1',
        name: 'accuracy',
        displayName: '准确率',
        category: 'accuracy',
        description: '正确预测样本数占总样本数的比例',
        formula: 'correct_predictions / total_predictions',
        dataType: 'classification',
        range: { min: 0, max: 1 },
        higherBetter: true,
        status: 'active',
        weight: 1.0,
        threshold: { excellent: 0.95, good: 0.85, fair: 0.7 },
        aggregationMethod: 'mean',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '2',
        name: 'f1_score',
        displayName: 'F1分数',
        category: 'accuracy',
        description: '精确率和召回率的调和平均数',
        formula: '2 * (precision * recall) / (precision + recall)',
        dataType: 'classification',
        range: { min: 0, max: 1 },
        higherBetter: true,
        status: 'active',
        weight: 1.0,
        threshold: { excellent: 0.9, good: 0.8, fair: 0.65 },
        aggregationMethod: 'mean',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '3',
        name: 'bleu_score',
        displayName: 'BLEU分数',
        category: 'fluency',
        description: '评估生成文本与参考文本的相似度',
        formula: 'geometric_mean(precision_n) * brevity_penalty',
        dataType: 'text_generation',
        range: { min: 0, max: 1 },
        higherBetter: true,
        status: 'active',
        weight: 0.8,
        threshold: { excellent: 0.4, good: 0.25, fair: 0.15 },
        aggregationMethod: 'mean',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '4',
        name: 'rouge_l',
        displayName: 'ROUGE-L',
        category: 'fluency',
        description: '基于最长公共子序列的文本相似度评估',
        formula: 'F_measure(LCS(reference, candidate))',
        dataType: 'text_generation',
        range: { min: 0, max: 1 },
        higherBetter: true,
        status: 'active',
        weight: 0.7,
        threshold: { excellent: 0.5, good: 0.35, fair: 0.2 },
        aggregationMethod: 'mean',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '5',
        name: 'perplexity',
        displayName: '困惑度',
        category: 'fluency',
        description: '衡量语言模型对文本序列的预测不确定性',
        formula: 'exp(-sum(log(p(w_i))) / N)',
        dataType: 'text_generation',
        range: { min: 1, max: 1000 },
        higherBetter: false,
        status: 'active',
        weight: 0.6,
        threshold: { excellent: 20, good: 50, fair: 100 },
        aggregationMethod: 'mean',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '6',
        name: 'bert_score',
        displayName: 'BERTScore',
        category: 'coherence',
        description: '基于BERT嵌入的语义相似度评估',
        formula: 'cosine_similarity(BERT(ref), BERT(cand))',
        dataType: 'text_generation',
        range: { min: 0, max: 1 },
        higherBetter: true,
        status: 'active',
        weight: 0.9,
        threshold: { excellent: 0.85, good: 0.75, fair: 0.6 },
        aggregationMethod: 'mean',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '7',
        name: 'diversity_score',
        displayName: '多样性得分',
        category: 'diversity',
        description: '评估生成文本的词汇和句法多样性',
        formula: 'distinct_n_grams / total_n_grams',
        dataType: 'text_generation',
        range: { min: 0, max: 1 },
        higherBetter: true,
        status: 'active',
        weight: 0.5,
        threshold: { excellent: 0.8, good: 0.6, fair: 0.4 },
        aggregationMethod: 'mean',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '8',
        name: 'safety_score',
        displayName: '安全性得分',
        category: 'safety',
        description: '评估生成内容的安全性和合规性',
        formula: 'weighted_sum(safety_classifiers)',
        dataType: 'text_generation',
        range: { min: 0, max: 1 },
        higherBetter: true,
        status: 'active',
        weight: 1.2,
        threshold: { excellent: 0.95, good: 0.85, fair: 0.7 },
        aggregationMethod: 'min',
        lastUpdated: '2024-01-15T10:00:00',
      },
      {
        id: '9',
        name: 'inference_time',
        displayName: '推理时间',
        category: 'efficiency',
        description: '模型完成推理任务所需的时间',
        formula: 'end_time - start_time',
        dataType: 'regression',
        range: { min: 0, max: 10000 },
        higherBetter: false,
        status: 'active',
        weight: 0.3,
        threshold: { excellent: 100, good: 500, fair: 1000 },
        aggregationMethod: 'median',
        lastUpdated: '2024-01-15T10:00:00',
      },
    ];

    const mockConfigurations: MetricsConfiguration[] = [
      {
        id: '1',
        name: '通用分类任务配置',
        description: '适用于大部分分类任务的标准指标配置',
        metrics: [
          { metricId: '1', weight: 0.4 }, // accuracy
          { metricId: '2', weight: 0.4 }, // f1_score
          { metricId: '9', weight: 0.2 }, // inference_time
        ],
        taskType: ['classification', 'multi_choice'],
        status: 'active',
        createdAt: '2024-01-10T10:00:00',
        usageCount: 145,
      },
      {
        id: '2',
        name: '文本生成质量评估',
        description: '专门用于评估文本生成质量的综合配置',
        metrics: [
          { metricId: '3', weight: 0.2 }, // bleu_score
          { metricId: '4', weight: 0.2 }, // rouge_l
          { metricId: '6', weight: 0.3 }, // bert_score
          { metricId: '7', weight: 0.15 }, // diversity_score
          { metricId: '8', weight: 0.15 }, // safety_score
        ],
        taskType: ['text_generation'],
        status: 'active',
        createdAt: '2024-01-12T14:30:00',
        usageCount: 89,
      },
      {
        id: '3',
        name: '高性能推理配置',
        description: '注重推理效率的轻量级评估配置',
        metrics: [
          { metricId: '1', weight: 0.6 }, // accuracy
          { metricId: '9', weight: 0.4 }, // inference_time
        ],
        taskType: ['classification', 'ranking'],
        status: 'active',
        createdAt: '2024-01-18T09:15:00',
        usageCount: 67,
      },
    ];

    setMetrics(mockMetrics);
    setConfigurations(mockConfigurations);
  };

  // 指标操作
  const handleCreateMetric = () => {
    setEditingMetric(null);
    metricForm.resetFields();
    setIsMetricModalVisible(true);
  };

  const handleEditMetric = (metric: EvaluationMetric) => {
    setEditingMetric(metric);
    metricForm.setFieldsValue({
      ...metric,
      thresholdExcellent: metric.threshold.excellent,
      thresholdGood: metric.threshold.good,
      thresholdFair: metric.threshold.fair,
      rangeMin: metric.range.min,
      rangeMax: metric.range.max,
    });
    setIsMetricModalVisible(true);
  };

  const handleDeleteMetric = (id: string) => {
    setMetrics(metrics.filter(metric => metric.id !== id));
    message.success('指标已删除');
  };

  const handleToggleMetric = (id: string, status: EvaluationMetric['status']) => {
    setMetrics(metrics.map(metric => 
      metric.id === id 
        ? { ...metric, status: status === 'active' ? 'disabled' : 'active' }
        : metric
    ));
    message.success(`指标已${status === 'active' ? '禁用' : '启用'}`);
  };

  const handleSaveMetric = async () => {
    try {
      const values = await metricForm.validateFields();
      
      const processedValues = {
        ...values,
        range: {
          min: values.rangeMin,
          max: values.rangeMax,
        },
        threshold: {
          excellent: values.thresholdExcellent,
          good: values.thresholdGood,
          fair: values.thresholdFair,
        },
      };
      
      if (editingMetric) {
        setMetrics(metrics.map(metric => 
          metric.id === editingMetric.id 
            ? { ...metric, ...processedValues, lastUpdated: new Date().toISOString() }
            : metric
        ));
        message.success('指标已更新');
      } else {
        const newMetric: EvaluationMetric = {
          ...processedValues,
          id: Date.now().toString(),
          lastUpdated: new Date().toISOString(),
        };
        setMetrics([...metrics, newMetric]);
        message.success('指标已创建');
      }
      
      setIsMetricModalVisible(false);
      metricForm.resetFields();
    } catch (error) {
      message.error('保存失败，请检查输入');
    }
  };

  // 配置操作
  const handleCreateConfig = () => {
    setEditingConfig(null);
    configForm.resetFields();
    setIsConfigModalVisible(true);
  };

  const handleEditConfig = (config: MetricsConfiguration) => {
    setEditingConfig(config);
    configForm.setFieldsValue(config);
    setIsConfigModalVisible(true);
  };

  const handleSaveConfig = async () => {
    try {
      const values = await configForm.validateFields();
      
      if (editingConfig) {
        setConfigurations(configurations.map(config => 
          config.id === editingConfig.id 
            ? { ...config, ...values }
            : config
        ));
        message.success('配置已更新');
      } else {
        const newConfig: MetricsConfiguration = {
          ...values,
          id: Date.now().toString(),
          status: 'draft',
          createdAt: new Date().toISOString(),
          usageCount: 0,
        };
        setConfigurations([...configurations, newConfig]);
        message.success('配置已创建');
      }
      
      setIsConfigModalVisible(false);
      configForm.resetFields();
    } catch (error) {
      message.error('保存失败，请检查输入');
    }
  };

  // 表格列定义
  const metricColumns: ColumnsType<EvaluationMetric> = [
    {
      title: '指标名称',
      key: 'name',
      render: (record: EvaluationMetric) => (
        <div>
          <div style={{ fontWeight: 500, display: 'flex', alignItems: 'center' }}>
            {record.displayName}
            <Tooltip title={record.description}>
              <InfoCircleOutlined style={{ marginLeft: '8px', color: '#999' }} />
            </Tooltip>
          </div>
          <div style={{ fontSize: '12px', color: '#666', fontFamily: 'monospace' }}>
            {record.name}
          </div>
        </div>
      ),
    },
    {
      title: '分类',
      dataIndex: 'category',
      key: 'category',
      render: (category: string) => {
        const colors = {
          accuracy: 'blue',
          fluency: 'green',
          coherence: 'orange',
          diversity: 'purple',
          safety: 'red',
          efficiency: 'cyan',
        };
        const labels = {
          accuracy: '准确性',
          fluency: '流畅性',
          coherence: '连贯性',
          diversity: '多样性',
          safety: '安全性',
          efficiency: '效率',
        };
        return (
          <Tag color={colors[category as keyof typeof colors]}>
            {labels[category as keyof typeof labels]}
          </Tag>
        );
      },
    },
    {
      title: '数据类型',
      dataIndex: 'dataType',
      key: 'dataType',
      render: (dataType: string) => {
        const labels = {
          classification: '分类',
          regression: '回归',
          text_generation: '文本生成',
          ranking: '排序',
          multi_choice: '多选',
        };
        return <Tag>{labels[dataType as keyof typeof labels] || dataType}</Tag>;
      },
    },
    {
      title: '取值范围',
      key: 'range',
      render: (record: EvaluationMetric) => (
        <div style={{ fontSize: '12px' }}>
          [{record.range.min}, {record.range.max}]
          <div style={{ color: record.higherBetter ? '#52c41a' : '#ff4d4f' }}>
            {record.higherBetter ? '↗ 越高越好' : '↘ 越低越好'}
          </div>
        </div>
      ),
    },
    {
      title: '阈值配置',
      key: 'threshold',
      render: (record: EvaluationMetric) => (
        <div style={{ fontSize: '12px' }}>
          <div><span style={{ color: '#52c41a' }}>优秀:</span> {record.threshold.excellent}</div>
          <div><span style={{ color: '#faad14' }}>良好:</span> {record.threshold.good}</div>
          <div><span style={{ color: '#ff4d4f' }}>及格:</span> {record.threshold.fair}</div>
        </div>
      ),
    },
    {
      title: '权重/聚合',
      key: 'weight',
      render: (record: EvaluationMetric) => (
        <div style={{ fontSize: '12px' }}>
          <div>权重: {record.weight}</div>
          <div>聚合: {record.aggregationMethod}</div>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          active: 'green',
          disabled: 'orange',
          deprecated: 'red',
        };
        const labels = {
          active: '启用',
          disabled: '禁用',
          deprecated: '已弃用',
        };
        return (
          <Tag color={colors[status as keyof typeof colors]}>
            {labels[status as keyof typeof labels]}
          </Tag>
        );
      },
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: EvaluationMetric) => (
        <Space>
          <Tooltip title="编辑">
            <Button 
              type="text" 
              icon={<EditOutlined />} 
              size="small"
              onClick={() => handleEditMetric(record)}
            />
          </Tooltip>
          <Tooltip title={record.status === 'active' ? '禁用' : '启用'}>
            <Switch
              size="small"
              checked={record.status === 'active'}
              onChange={() => handleToggleMetric(record.id, record.status)}
            />
          </Tooltip>
          <Popconfirm
            title="确定要删除这个指标吗？"
            onConfirm={() => handleDeleteMetric(record.id)}
          >
            <Tooltip title="删除">
              <Button type="text" icon={<DeleteOutlined />} size="small" danger />
            </Tooltip>
          </Popconfirm>
        </Space>
      ),
    },
  ];

  const configColumns: ColumnsType<MetricsConfiguration> = [
    {
      title: '配置名称',
      key: 'name',
      render: (record: MetricsConfiguration) => (
        <div>
          <div style={{ fontWeight: 500 }}>
            {record.name}
            <Badge 
              count={record.usageCount} 
              style={{ backgroundColor: '#52c41a', marginLeft: '8px' }}
              title="使用次数"
            />
          </div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.description}
          </div>
        </div>
      ),
    },
    {
      title: '包含指标',
      dataIndex: 'metrics',
      key: 'metrics',
      render: (configMetrics: Array<{metricId: string; weight: number}>) => (
        <div>
          <div style={{ marginBottom: '4px' }}>共{configMetrics.length}个指标</div>
          {configMetrics.slice(0, 3).map(cm => {
            const metric = metrics.find(m => m.id === cm.metricId);
            return metric ? (
              <Tag key={cm.metricId} size="small" style={{ marginBottom: '2px' }}>
                {metric.displayName} ({cm.weight})
              </Tag>
            ) : null;
          })}
          {configMetrics.length > 3 && (
            <Tag size="small">+{configMetrics.length - 3}个更多</Tag>
          )}
        </div>
      ),
    },
    {
      title: '适用任务',
      dataIndex: 'taskType',
      key: 'taskType',
      render: (taskTypes: string[]) => (
        <div>
          {taskTypes.map(type => (
            <Tag key={type} size="small" style={{ marginBottom: '2px' }}>
              {type}
            </Tag>
          ))}
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          active: 'green',
          draft: 'blue',
          deprecated: 'red',
        };
        const labels = {
          active: '启用',
          draft: '草稿',
          deprecated: '已弃用',
        };
        return (
          <Tag color={colors[status as keyof typeof colors]}>
            {labels[status as keyof typeof labels]}
          </Tag>
        );
      },
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: MetricsConfiguration) => (
        <Space>
          <Tooltip title="编辑">
            <Button 
              type="text" 
              icon={<EditOutlined />} 
              size="small"
              onClick={() => handleEditConfig(record)}
            />
          </Tooltip>
          <Tooltip title="复制配置">
            <Button type="text" icon={<ExperimentOutlined />} size="small" />
          </Tooltip>
          <Tooltip title="查看详情">
            <Button type="text" icon={<InfoCircleOutlined />} size="small" />
          </Tooltip>
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1><SlidersOutlined /> 评估指标配置</h1>
        <p>管理评估指标的定义、阈值配置和指标组合方案</p>
      </div>

      <Alert
        message="评估指标配置说明"
        description="评估指标是衡量模型性能的核心要素。您可以定义单个指标的计算方式、阈值和权重，也可以创建针对特定任务的指标组合配置。"
        type="info"
        showIcon
        style={{ marginBottom: '16px' }}
      />

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="指标定义管理" key="metrics">
          <div style={{ marginBottom: '16px' }}>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={handleCreateMetric}
            >
              新建评估指标
            </Button>
          </div>
          
          <Table
            columns={metricColumns}
            dataSource={metrics}
            rowKey="id"
            loading={loading}
            pagination={{ pageSize: 10 }}
            scroll={{ x: 1200 }}
          />
        </TabPane>

        <TabPane tab="指标组合配置" key="configurations">
          <div style={{ marginBottom: '16px' }}>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={handleCreateConfig}
            >
              新建指标配置
            </Button>
          </div>
          
          <Table
            columns={configColumns}
            dataSource={configurations}
            rowKey="id"
            loading={loading}
            pagination={{ pageSize: 10 }}
          />
        </TabPane>

        <TabPane tab="预设配置模板" key="templates">
          <Row gutter={[16, 16]}>
            <Col span={8}>
              <Card 
                title="分类任务标准配置"
                extra={<CheckCircleOutlined style={{ color: '#52c41a' }} />}
                actions={[
                  <Button key="use" type="link">使用模板</Button>,
                  <Button key="customize" type="link">自定义</Button>
                ]}
              >
                <p>适用于二分类和多分类任务</p>
                <div style={{ fontSize: '12px', color: '#666' }}>
                  <div>• 准确率 (40%)</div>
                  <div>• F1分数 (40%)</div>
                  <div>• 推理时间 (20%)</div>
                </div>
              </Card>
            </Col>
            
            <Col span={8}>
              <Card 
                title="文本生成质量配置"
                extra={<FunctionOutlined style={{ color: '#1890ff' }} />}
                actions={[
                  <Button key="use" type="link">使用模板</Button>,
                  <Button key="customize" type="link">自定义</Button>
                ]}
              >
                <p>专为文本生成任务优化</p>
                <div style={{ fontSize: '12px', color: '#666' }}>
                  <div>• BLEU分数 (25%)</div>
                  <div>• ROUGE-L (25%)</div>
                  <div>• BERTScore (30%)</div>
                  <div>• 多样性 (10%)</div>
                  <div>• 安全性 (10%)</div>
                </div>
              </Card>
            </Col>

            <Col span={8}>
              <Card 
                title="高效推理配置"
                extra={<ExperimentOutlined style={{ color: '#faad14' }} />}
                actions={[
                  <Button key="use" type="link">使用模板</Button>,
                  <Button key="customize" type="link">自定义</Button>
                ]}
              >
                <p>平衡准确性和推理效率</p>
                <div style={{ fontSize: '12px', color: '#666' }}>
                  <div>• 准确率 (60%)</div>
                  <div>• 推理时间 (40%)</div>
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 指标编辑模态框 */}
      <Modal
        title={editingMetric ? "编辑评估指标" : "新建评估指标"}
        open={isMetricModalVisible}
        onOk={handleSaveMetric}
        onCancel={() => {
          setIsMetricModalVisible(false);
          metricForm.resetFields();
        }}
        width={800}
      >
        <Form
          form={metricForm}
          layout="vertical"
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="指标名称 (英文)"
                rules={[{ required: true, message: '请输入指标名称' }]}
              >
                <Input placeholder="如: accuracy, f1_score" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="displayName"
                label="显示名称 (中文)"
                rules={[{ required: true, message: '请输入显示名称' }]}
              >
                <Input placeholder="如: 准确率, F1分数" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="description"
            label="指标描述"
            rules={[{ required: true, message: '请输入指标描述' }]}
          >
            <TextArea rows={2} placeholder="详细描述指标的含义和用途" />
          </Form.Item>

          <Form.Item
            name="formula"
            label="计算公式"
            rules={[{ required: true, message: '请输入计算公式' }]}
          >
            <Input placeholder="指标的计算公式或方法描述" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="category"
                label="指标分类"
                rules={[{ required: true, message: '请选择指标分类' }]}
              >
                <Select placeholder="选择分类">
                  <Option value="accuracy">准确性</Option>
                  <Option value="fluency">流畅性</Option>
                  <Option value="coherence">连贯性</Option>
                  <Option value="diversity">多样性</Option>
                  <Option value="safety">安全性</Option>
                  <Option value="efficiency">效率</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="dataType"
                label="数据类型"
                rules={[{ required: true, message: '请选择数据类型' }]}
              >
                <Select placeholder="选择数据类型">
                  <Option value="classification">分类</Option>
                  <Option value="regression">回归</Option>
                  <Option value="text_generation">文本生成</Option>
                  <Option value="ranking">排序</Option>
                  <Option value="multi_choice">多选</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="aggregationMethod"
                label="聚合方式"
                rules={[{ required: true, message: '请选择聚合方式' }]}
              >
                <Select placeholder="选择聚合方式">
                  <Option value="mean">平均值</Option>
                  <Option value="median">中位数</Option>
                  <Option value="weighted_mean">加权平均</Option>
                  <Option value="max">最大值</Option>
                  <Option value="min">最小值</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="rangeMin"
                label="最小值"
                rules={[{ required: true, message: '请输入最小值' }]}
              >
                <InputNumber style={{ width: '100%' }} placeholder="最小值" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="rangeMax"
                label="最大值"
                rules={[{ required: true, message: '请输入最大值' }]}
              >
                <InputNumber style={{ width: '100%' }} placeholder="最大值" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="weight"
                label="默认权重"
                rules={[{ required: true, message: '请输入默认权重' }]}
              >
                <InputNumber 
                  style={{ width: '100%' }} 
                  min={0} 
                  max={2} 
                  step={0.1}
                  placeholder="默认权重" 
                />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="thresholdExcellent"
                label="优秀阈值"
                rules={[{ required: true, message: '请输入优秀阈值' }]}
              >
                <InputNumber style={{ width: '100%' }} placeholder="优秀阈值" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="thresholdGood"
                label="良好阈值"
                rules={[{ required: true, message: '请输入良好阈值' }]}
              >
                <InputNumber style={{ width: '100%' }} placeholder="良好阈值" />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="thresholdFair"
                label="及格阈值"
                rules={[{ required: true, message: '请输入及格阈值' }]}
              >
                <InputNumber style={{ width: '100%' }} placeholder="及格阈值" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="higherBetter"
                label="数值倾向"
                valuePropName="checked"
              >
                <Switch
                  checkedChildren="越高越好"
                  unCheckedChildren="越低越好"
                />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="status"
                label="状态"
                rules={[{ required: true, message: '请选择状态' }]}
              >
                <Select placeholder="选择状态">
                  <Option value="active">启用</Option>
                  <Option value="disabled">禁用</Option>
                  <Option value="deprecated">已弃用</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>

      {/* 配置编辑模态框 */}
      <Modal
        title={editingConfig ? "编辑指标配置" : "新建指标配置"}
        open={isConfigModalVisible}
        onOk={handleSaveConfig}
        onCancel={() => {
          setIsConfigModalVisible(false);
          configForm.resetFields();
        }}
        width={600}
      >
        <Form
          form={configForm}
          layout="vertical"
        >
          <Form.Item
            name="name"
            label="配置名称"
            rules={[{ required: true, message: '请输入配置名称' }]}
          >
            <Input placeholder="输入指标配置名称" />
          </Form.Item>

          <Form.Item
            name="description"
            label="配置描述"
            rules={[{ required: true, message: '请输入配置描述' }]}
          >
            <TextArea rows={3} placeholder="描述这个指标配置的用途和特点" />
          </Form.Item>

          <Form.Item
            name="taskType"
            label="适用任务类型"
            rules={[{ required: true, message: '请选择适用任务类型' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择适用的任务类型"
            >
              <Option value="classification">分类任务</Option>
              <Option value="regression">回归任务</Option>
              <Option value="text_generation">文本生成</Option>
              <Option value="ranking">排序任务</Option>
              <Option value="multi_choice">多选任务</Option>
            </Select>
          </Form.Item>

          {/* 这里可以添加更复杂的指标选择和权重配置界面 */}
          <Alert
            message="指标选择和权重配置"
            description="请在指标定义管理中先创建所需的评估指标，然后在此处配置具体的指标组合和权重分配。"
            type="info"
            style={{ marginTop: '16px' }}
          />
        </Form>
      </Modal>
    </div>
  );
};

export default EvaluationMetricsConfigPage;