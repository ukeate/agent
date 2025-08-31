import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Button, Table, Tag, Space, Typography, Modal, Form, Input, Select, Switch, Drawer, Descriptions, Progress, Alert, Upload, message } from 'antd';
import { 
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  DownloadOutlined,
  UploadOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  FileTextOutlined,
  DatabaseOutlined,
  RocketOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { Option } = Select;
const { TextArea } = Input;

interface BenchmarkSuite {
  id: string;
  name: string;
  displayName: string;
  category: 'glue' | 'superglue' | 'mmlu' | 'humaneval' | 'hellaswag' | 'custom';
  status: 'active' | 'disabled' | 'draft';
  version: string;
  description: string;
  totalTasks: number;
  completedEvaluations: number;
  averageAccuracy: number;
  lastUsed: string;
  createdAt: string;
  tasks: BenchmarkTask[];
  config: {
    timeLimit: number;
    maxSamples: number;
    randomSeed: number;
    metrics: string[];
  };
}

interface BenchmarkTask {
  id: string;
  name: string;
  description: string;
  sampleCount: number;
  metrics: string[];
  difficulty: 'easy' | 'medium' | 'hard';
}

const BenchmarkSuiteManagementPage: React.FC = () => {
  const [suites, setSuites] = useState<BenchmarkSuite[]>([]);
  const [loading, setLoading] = useState(true);
  const [editModalVisible, setEditModalVisible] = useState(false);
  const [detailDrawerVisible, setDetailDrawerVisible] = useState(false);
  const [selectedSuite, setSelectedSuite] = useState<BenchmarkSuite | null>(null);
  const [form] = Form.useForm();

  useEffect(() => {
    loadSuitesData();
  }, []);

  const loadSuitesData = async () => {
    try {
      setLoading(true);
      
      // 模拟API调用
      const suitesData: BenchmarkSuite[] = [
        {
          id: 'glue_suite_001',
          name: 'glue',
          displayName: 'GLUE Benchmark Suite',
          category: 'glue',
          status: 'active',
          version: '1.0.0',
          description: 'General Language Understanding Evaluation benchmark for natural language understanding tasks.',
          totalTasks: 9,
          completedEvaluations: 156,
          averageAccuracy: 0.847,
          lastUsed: '2024-01-15 14:30:00',
          createdAt: '2024-01-01 09:00:00',
          tasks: [
            { id: 'cola', name: 'CoLA', description: 'Corpus of Linguistic Acceptability', sampleCount: 8551, metrics: ['accuracy', 'matthews_correlation'], difficulty: 'medium' },
            { id: 'sst2', name: 'SST-2', description: 'Stanford Sentiment Treebank', sampleCount: 67349, metrics: ['accuracy', 'f1'], difficulty: 'easy' },
            { id: 'mrpc', name: 'MRPC', description: 'Microsoft Research Paraphrase Corpus', sampleCount: 3668, metrics: ['accuracy', 'f1'], difficulty: 'medium' },
            { id: 'qqp', name: 'QQP', description: 'Quora Question Pairs', sampleCount: 363846, metrics: ['accuracy', 'f1'], difficulty: 'hard' },
            { id: 'stsb', name: 'STS-B', description: 'Semantic Textual Similarity Benchmark', sampleCount: 5749, metrics: ['pearson', 'spearman'], difficulty: 'medium' },
            { id: 'mnli', name: 'MNLI', description: 'Multi-Genre Natural Language Inference', sampleCount: 392702, metrics: ['accuracy'], difficulty: 'hard' },
            { id: 'qnli', name: 'QNLI', description: 'Question Natural Language Inference', sampleCount: 104743, metrics: ['accuracy'], difficulty: 'medium' },
            { id: 'rte', name: 'RTE', description: 'Recognizing Textual Entailment', sampleCount: 2490, metrics: ['accuracy'], difficulty: 'hard' },
            { id: 'wnli', name: 'WNLI', description: 'Winograd Natural Language Inference', sampleCount: 635, metrics: ['accuracy'], difficulty: 'hard' }
          ],
          config: {
            timeLimit: 7200,
            maxSamples: -1,
            randomSeed: 42,
            metrics: ['accuracy', 'f1', 'matthews_correlation']
          }
        },
        {
          id: 'superglue_suite_001',
          name: 'superglue',
          displayName: 'SuperGLUE Benchmark Suite',
          category: 'superglue',
          status: 'active',
          version: '1.0.3',
          description: 'A more challenging benchmark for general-purpose language understanding.',
          totalTasks: 8,
          completedEvaluations: 89,
          averageAccuracy: 0.723,
          lastUsed: '2024-01-15 13:45:00',
          createdAt: '2024-01-01 10:30:00',
          tasks: [
            { id: 'boolq', name: 'BoolQ', description: 'Boolean Questions', sampleCount: 9427, metrics: ['accuracy'], difficulty: 'medium' },
            { id: 'cb', name: 'CB', description: 'CommitmentBank', sampleCount: 250, metrics: ['accuracy', 'f1'], difficulty: 'hard' },
            { id: 'copa', name: 'COPA', description: 'Choice of Plausible Alternatives', sampleCount: 400, metrics: ['accuracy'], difficulty: 'medium' },
            { id: 'multirc', name: 'MultiRC', description: 'Multi-Sentence Reading Comprehension', sampleCount: 5100, metrics: ['f1a', 'em'], difficulty: 'hard' },
            { id: 'record', name: 'ReCoRD', description: 'Reading Comprehension with Commonsense Reasoning', sampleCount: 100730, metrics: ['f1', 'em'], difficulty: 'hard' },
            { id: 'rte', name: 'RTE', description: 'Recognizing Textual Entailment', sampleCount: 2490, metrics: ['accuracy'], difficulty: 'hard' },
            { id: 'wic', name: 'WiC', description: 'Word-in-Context', sampleCount: 5428, metrics: ['accuracy'], difficulty: 'medium' },
            { id: 'wsc', name: 'WSC', description: 'Winograd Schema Challenge', sampleCount: 104, metrics: ['accuracy'], difficulty: 'hard' }
          ],
          config: {
            timeLimit: 10800,
            maxSamples: -1,
            randomSeed: 42,
            metrics: ['accuracy', 'f1', 'em']
          }
        },
        {
          id: 'mmlu_suite_001',
          name: 'mmlu',
          displayName: 'MMLU Benchmark Suite',
          category: 'mmlu',
          status: 'active',
          version: '1.0.1',
          description: 'Massive Multitask Language Understanding benchmark across 57 subjects.',
          totalTasks: 57,
          completedEvaluations: 45,
          averageAccuracy: 0.681,
          lastUsed: '2024-01-15 12:20:00',
          createdAt: '2024-01-01 11:15:00',
          tasks: [
            { id: 'abstract_algebra', name: 'Abstract Algebra', description: 'College-level abstract algebra questions', sampleCount: 100, metrics: ['accuracy'], difficulty: 'hard' },
            { id: 'anatomy', name: 'Anatomy', description: 'High school anatomy questions', sampleCount: 135, metrics: ['accuracy'], difficulty: 'medium' },
            { id: 'astronomy', name: 'Astronomy', description: 'High school astronomy questions', sampleCount: 152, metrics: ['accuracy'], difficulty: 'medium' }
          ],
          config: {
            timeLimit: 14400,
            maxSamples: 1000,
            randomSeed: 42,
            metrics: ['accuracy']
          }
        },
        {
          id: 'humaneval_suite_001',
          name: 'humaneval',
          displayName: 'HumanEval Code Generation',
          category: 'humaneval',
          status: 'active',
          version: '1.0.0',
          description: 'Hand-written programming problems for evaluating code generation capabilities.',
          totalTasks: 164,
          completedEvaluations: 78,
          averageAccuracy: 0.734,
          lastUsed: '2024-01-15 11:30:00',
          createdAt: '2024-01-01 12:00:00',
          tasks: [
            { id: 'python_coding', name: 'Python Coding', description: 'Python programming problems', sampleCount: 164, metrics: ['pass@1', 'pass@10', 'pass@100'], difficulty: 'hard' }
          ],
          config: {
            timeLimit: 1800,
            maxSamples: 164,
            randomSeed: 0,
            metrics: ['pass@1', 'pass@10', 'pass@100']
          }
        },
        {
          id: 'custom_suite_001',
          name: 'custom_finance',
          displayName: '金融领域专业测试套件',
          category: 'custom',
          status: 'draft',
          version: '0.1.0',
          description: '针对金融领域专门设计的自定义基准测试套件。',
          totalTasks: 3,
          completedEvaluations: 2,
          averageAccuracy: 0.892,
          lastUsed: '2024-01-14 16:45:00',
          createdAt: '2024-01-14 14:20:00',
          tasks: [
            { id: 'risk_assessment', name: '风险评估', description: '金融风险评估能力测试', sampleCount: 500, metrics: ['accuracy', 'precision', 'recall'], difficulty: 'hard' },
            { id: 'market_analysis', name: '市场分析', description: '市场趋势分析能力测试', sampleCount: 300, metrics: ['accuracy', 'f1'], difficulty: 'medium' },
            { id: 'fraud_detection', name: '欺诈检测', description: '金融欺诈检测能力测试', sampleCount: 800, metrics: ['accuracy', 'auc', 'f1'], difficulty: 'hard' }
          ],
          config: {
            timeLimit: 5400,
            maxSamples: 1000,
            randomSeed: 123,
            metrics: ['accuracy', 'precision', 'recall', 'f1', 'auc']
          }
        }
      ];

      setSuites(suitesData);
    } catch (error) {
      console.error('加载基准测试套件数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleCreateSuite = () => {
    setSelectedSuite(null);
    form.resetFields();
    setEditModalVisible(true);
  };

  const handleEditSuite = (suite: BenchmarkSuite) => {
    setSelectedSuite(suite);
    form.setFieldsValue({
      name: suite.name,
      displayName: suite.displayName,
      description: suite.description,
      category: suite.category,
      timeLimit: suite.config.timeLimit,
      maxSamples: suite.config.maxSamples,
      randomSeed: suite.config.randomSeed,
      metrics: suite.config.metrics
    });
    setEditModalVisible(true);
  };

  const handleDeleteSuite = (suiteId: string) => {
    Modal.confirm({
      title: '确认删除',
      icon: <ExclamationCircleOutlined />,
      content: '删除后无法恢复，确定要删除这个基准测试套件吗？',
      okText: '确定',
      cancelText: '取消',
      onOk: async () => {
        try {
          console.log('删除套件:', suiteId);
          await loadSuitesData();
          message.success('删除成功');
        } catch (error) {
          message.error('删除失败');
        }
      }
    });
  };

  const handleViewDetail = (suite: BenchmarkSuite) => {
    setSelectedSuite(suite);
    setDetailDrawerVisible(true);
  };

  const handleSaveSuite = async () => {
    try {
      const values = await form.validateFields();
      console.log('保存套件:', values);
      setEditModalVisible(false);
      await loadSuitesData();
      message.success(selectedSuite ? '更新成功' : '创建成功');
    } catch (error) {
      message.error('保存失败');
    }
  };

  const handleRunEvaluation = (suiteId: string) => {
    Modal.confirm({
      title: '运行评估',
      content: '确定要使用这个基准测试套件运行评估吗？',
      onOk: () => {
        console.log('运行评估:', suiteId);
        message.success('评估任务已启动');
      }
    });
  };

  const getCategoryTag = (category: string) => {
    const categoryConfig = {
      glue: { color: 'blue', text: 'GLUE' },
      superglue: { color: 'green', text: 'SuperGLUE' },
      mmlu: { color: 'orange', text: 'MMLU' },
      humaneval: { color: 'purple', text: 'HumanEval' },
      hellaswag: { color: 'cyan', text: 'HellaSwag' },
      custom: { color: 'magenta', text: '自定义' }
    };
    const config = categoryConfig[category as keyof typeof categoryConfig];
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const getStatusTag = (status: string) => {
    const statusConfig = {
      active: { color: 'success', text: '激活' },
      disabled: { color: 'default', text: '停用' },
      draft: { color: 'processing', text: '草稿' }
    };
    const config = statusConfig[status as keyof typeof statusConfig];
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const getDifficultyTag = (difficulty: string) => {
    const difficultyConfig = {
      easy: { color: 'green', text: '简单' },
      medium: { color: 'orange', text: '中等' },
      hard: { color: 'red', text: '困难' }
    };
    const config = difficultyConfig[difficulty as keyof typeof difficultyConfig];
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const columns = [
    {
      title: '套件名称',
      dataIndex: 'displayName',
      key: 'displayName',
      width: 200,
      render: (text: string, record: BenchmarkSuite) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{text}</div>
          <Text type="secondary" style={{ fontSize: '12px' }}>{record.name}</Text>
        </div>
      )
    },
    {
      title: '类别',
      dataIndex: 'category',
      key: 'category',
      render: (category: string) => getCategoryTag(category),
      width: 100,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusTag(status),
      width: 80,
    },
    {
      title: '版本',
      dataIndex: 'version',
      key: 'version',
      width: 80,
    },
    {
      title: '任务数',
      dataIndex: 'totalTasks',
      key: 'totalTasks',
      width: 80,
    },
    {
      title: '平均准确率',
      dataIndex: 'averageAccuracy',
      key: 'averageAccuracy',
      render: (accuracy: number) => (
        <div>
          <Progress 
            percent={accuracy * 100} 
            size="small" 
            showInfo={false}
            strokeColor={{
              '0%': '#ff4d4f',
              '50%': '#faad14',
              '100%': '#52c41a',
            }}
          />
          <Text style={{ fontSize: '12px' }}>{(accuracy * 100).toFixed(1)}%</Text>
        </div>
      ),
      width: 120,
    },
    {
      title: '完成评估',
      dataIndex: 'completedEvaluations',
      key: 'completedEvaluations',
      width: 100,
    },
    {
      title: '最后使用',
      dataIndex: 'lastUsed',
      key: 'lastUsed',
      width: 150,
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: BenchmarkSuite) => (
        <Space size="small">
          <Button
            type="link"
            size="small"
            icon={<PlayCircleOutlined />}
            onClick={() => handleRunEvaluation(record.id)}
            disabled={record.status !== 'active'}
          >
            运行
          </Button>
          <Button
            type="link"
            size="small"
            icon={<EyeOutlined />}
            onClick={() => handleViewDetail(record)}
          >
            详情
          </Button>
          <Button
            type="link"
            size="small"
            icon={<EditOutlined />}
            onClick={() => handleEditSuite(record)}
          >
            编辑
          </Button>
          <Button
            type="link"
            size="small"
            icon={<DeleteOutlined />}
            onClick={() => handleDeleteSuite(record.id)}
            danger
            disabled={record.status === 'active'}
          >
            删除
          </Button>
        </Space>
      ),
      width: 200,
    },
  ];

  const taskColumns = [
    {
      title: '任务名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description',
    },
    {
      title: '样本数',
      dataIndex: 'sampleCount',
      key: 'sampleCount',
    },
    {
      title: '难度',
      dataIndex: 'difficulty',
      key: 'difficulty',
      render: (difficulty: string) => getDifficultyTag(difficulty),
    },
    {
      title: '评估指标',
      dataIndex: 'metrics',
      key: 'metrics',
      render: (metrics: string[]) => (
        <Space size={[0, 4]} wrap>
          {metrics.map(metric => <Tag key={metric}>{metric}</Tag>)}
        </Space>
      ),
    },
  ];

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>基准测试套件管理</Title>
        <Text type="secondary">
          管理和配置各种基准测试套件，包括GLUE、SuperGLUE、MMLU等标准测试集和自定义测试集
        </Text>
      </div>

      <Card 
        title="基准测试套件列表"
        extra={
          <Space>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={handleCreateSuite}
            >
              创建套件
            </Button>
            <Upload
              accept=".json"
              showUploadList={false}
              beforeUpload={() => {
                message.info('导入功能开发中...');
                return false;
              }}
            >
              <Button icon={<UploadOutlined />}>
                导入套件
              </Button>
            </Upload>
            <Button 
              icon={<ReloadOutlined />}
              onClick={loadSuitesData}
              loading={loading}
            >
              刷新
            </Button>
          </Space>
        }
      >
        <Table
          dataSource={suites}
          columns={columns}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 项 / 共 ${total} 项`
          }}
          loading={loading}
        />
      </Card>

      {/* 编辑模态框 */}
      <Modal
        title={selectedSuite ? '编辑基准测试套件' : '创建基准测试套件'}
        open={editModalVisible}
        onOk={handleSaveSuite}
        onCancel={() => setEditModalVisible(false)}
        width={600}
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="name"
            label="套件名称"
            rules={[{ required: true, message: '请输入套件名称' }]}
          >
            <Input placeholder="例如: glue, superglue, custom_finance" />
          </Form.Item>

          <Form.Item
            name="displayName"
            label="显示名称"
            rules={[{ required: true, message: '请输入显示名称' }]}
          >
            <Input placeholder="例如: GLUE Benchmark Suite" />
          </Form.Item>

          <Form.Item
            name="category"
            label="套件类别"
            rules={[{ required: true, message: '请选择套件类别' }]}
          >
            <Select placeholder="请选择套件类别">
              <Option value="glue">GLUE</Option>
              <Option value="superglue">SuperGLUE</Option>
              <Option value="mmlu">MMLU</Option>
              <Option value="humaneval">HumanEval</Option>
              <Option value="hellaswag">HellaSwag</Option>
              <Option value="custom">自定义</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="description"
            label="描述"
            rules={[{ required: true, message: '请输入描述' }]}
          >
            <TextArea rows={3} placeholder="请输入套件的详细描述" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="timeLimit"
                label="时间限制(秒)"
                rules={[{ required: true, message: '请输入时间限制' }]}
              >
                <Input type="number" min={300} max={86400} />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="maxSamples"
                label="最大样本数"
                rules={[{ required: true, message: '请输入最大样本数' }]}
              >
                <Input type="number" placeholder="-1表示无限制" />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="randomSeed"
                label="随机种子"
                rules={[{ required: true, message: '请输入随机种子' }]}
              >
                <Input type="number" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="metrics"
                label="评估指标"
                rules={[{ required: true, message: '请选择评估指标' }]}
              >
                <Select mode="multiple" placeholder="请选择评估指标">
                  <Option value="accuracy">Accuracy</Option>
                  <Option value="f1">F1 Score</Option>
                  <Option value="precision">Precision</Option>
                  <Option value="recall">Recall</Option>
                  <Option value="matthews_correlation">Matthews Correlation</Option>
                  <Option value="pearson">Pearson Correlation</Option>
                  <Option value="spearman">Spearman Correlation</Option>
                  <Option value="pass@1">Pass@1</Option>
                  <Option value="pass@10">Pass@10</Option>
                  <Option value="em">Exact Match</Option>
                  <Option value="auc">AUC</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
        </Form>
      </Modal>

      {/* 详情抽屉 */}
      <Drawer
        title="基准测试套件详情"
        placement="right"
        size="large"
        open={detailDrawerVisible}
        onClose={() => setDetailDrawerVisible(false)}
      >
        {selectedSuite && (
          <div>
            <Descriptions bordered column={2} style={{ marginBottom: '24px' }}>
              <Descriptions.Item label="套件名称">{selectedSuite.displayName}</Descriptions.Item>
              <Descriptions.Item label="类别">{getCategoryTag(selectedSuite.category)}</Descriptions.Item>
              <Descriptions.Item label="状态">{getStatusTag(selectedSuite.status)}</Descriptions.Item>
              <Descriptions.Item label="版本">{selectedSuite.version}</Descriptions.Item>
              <Descriptions.Item label="总任务数">{selectedSuite.totalTasks}</Descriptions.Item>
              <Descriptions.Item label="完成评估">{selectedSuite.completedEvaluations}</Descriptions.Item>
              <Descriptions.Item label="平均准确率">{(selectedSuite.averageAccuracy * 100).toFixed(2)}%</Descriptions.Item>
              <Descriptions.Item label="最后使用">{selectedSuite.lastUsed}</Descriptions.Item>
              <Descriptions.Item label="创建时间" span={2}>{selectedSuite.createdAt}</Descriptions.Item>
              <Descriptions.Item label="描述" span={2}>{selectedSuite.description}</Descriptions.Item>
            </Descriptions>

            <Card title="配置信息" size="small" style={{ marginBottom: '24px' }}>
              <Descriptions size="small" column={2}>
                <Descriptions.Item label="时间限制">{selectedSuite.config.timeLimit}秒</Descriptions.Item>
                <Descriptions.Item label="最大样本数">{selectedSuite.config.maxSamples === -1 ? '无限制' : selectedSuite.config.maxSamples}</Descriptions.Item>
                <Descriptions.Item label="随机种子">{selectedSuite.config.randomSeed}</Descriptions.Item>
                <Descriptions.Item label="评估指标">
                  <Space size={[0, 4]} wrap>
                    {selectedSuite.config.metrics.map(metric => <Tag key={metric}>{metric}</Tag>)}
                  </Space>
                </Descriptions.Item>
              </Descriptions>
            </Card>

            <Card title="包含任务" size="small">
              <Table
                dataSource={selectedSuite.tasks}
                columns={taskColumns}
                rowKey="id"
                pagination={false}
                size="small"
              />
            </Card>
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default BenchmarkSuiteManagementPage;