import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Table, Tag, Space, Typography, Button, Modal, Form, Input, Select, DatePicker, Upload, Progress, Drawer, Descriptions, Tabs, Alert, Tooltip, Divider } from 'antd';
import { 
  FileTextOutlined,
  DownloadOutlined,
  EyeOutlined,
  DeleteOutlined,
  ShareAltOutlined,
  UploadOutlined,
  FileExcelOutlined,
  FilePdfOutlined,
  FileImageOutlined,
  ReloadOutlined,
  PlusOutlined,
  SearchOutlined,
  FilterOutlined,
  BarChartOutlined,
  LineChartOutlined,
  PieChartOutlined,
  TrophyOutlined,
  ExperimentOutlined,
  CompareOutlined,
  CalendarOutlined,
  UserOutlined,
  TagOutlined,
  StarOutlined,
  SendOutlined
} from '@ant-design/icons';

const { Title, Text } = Typography;
const { RangePicker } = DatePicker;
const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;

interface EvaluationReport {
  id: string;
  title: string;
  type: 'single_model' | 'comparison' | 'benchmark_analysis' | 'trend_analysis' | 'custom';
  status: 'generating' | 'completed' | 'failed' | 'shared';
  format: 'html' | 'pdf' | 'excel' | 'json';
  description: string;
  createdBy: string;
  createdAt: string;
  updatedAt: string;
  fileSize: number;
  downloadCount: number;
  sharedWith: string[];
  tags: string[];
  models: string[];
  benchmarks: string[];
  metrics: {
    accuracy?: number;
    f1?: number;
    bleu?: number;
    rouge?: number;
  };
  charts: {
    type: 'bar' | 'line' | 'pie' | 'radar';
    title: string;
    data: any;
  }[];
  summary: {
    totalModels: number;
    totalBenchmarks: number;
    bestPerformingModel: string;
    averageAccuracy: number;
    keyFindings: string[];
  };
}

interface ReportTemplate {
  id: string;
  name: string;
  description: string;
  type: string;
  sections: string[];
  isDefault: boolean;
}

const EvaluationReportsCenterPage: React.FC = () => {
  const [reports, setReports] = useState<EvaluationReport[]>([]);
  const [templates, setTemplates] = useState<ReportTemplate[]>([]);
  const [loading, setLoading] = useState(true);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [previewDrawerVisible, setPreviewDrawerVisible] = useState(false);
  const [shareModalVisible, setShareModalVisible] = useState(false);
  const [selectedReport, setSelectedReport] = useState<EvaluationReport | null>(null);
  const [filterType, setFilterType] = useState<string>('all');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [searchText, setSearchText] = useState('');
  const [form] = Form.useForm();
  const [shareForm] = Form.useForm();

  useEffect(() => {
    loadReportsData();
    loadTemplatesData();
  }, []);

  const loadReportsData = async () => {
    try {
      setLoading(true);
      
      // 模拟API调用
      const reportsData: EvaluationReport[] = [
        {
          id: 'report_001',
          title: 'BERT模型GLUE基准测试综合报告',
          type: 'single_model',
          status: 'completed',
          format: 'html',
          description: 'BERT-Large-Uncased模型在GLUE基准测试上的详细评估报告，包含9个子任务的完整分析',
          createdBy: '张三',
          createdAt: '2024-01-15 16:30:00',
          updatedAt: '2024-01-15 16:45:00',
          fileSize: 2.5,
          downloadCount: 15,
          sharedWith: ['李四', '王五'],
          tags: ['BERT', 'GLUE', '自然语言理解'],
          models: ['BERT-Large-Uncased'],
          benchmarks: ['GLUE'],
          metrics: {
            accuracy: 0.847,
            f1: 0.834
          },
          charts: [
            {
              type: 'bar',
              title: 'GLUE任务性能对比',
              data: {
                labels: ['CoLA', 'SST-2', 'MRPC', 'QQP', 'STS-B', 'MNLI', 'QNLI', 'RTE', 'WNLI'],
                datasets: [{
                  label: 'Accuracy',
                  data: [0.85, 0.94, 0.89, 0.87, 0.89, 0.84, 0.91, 0.70, 0.56]
                }]
              }
            },
            {
              type: 'radar',
              title: '模型能力雷达图',
              data: {
                labels: ['语法接受度', '情感分析', '释义检测', '文本蕴含', '语义相似度'],
                datasets: [{
                  label: 'BERT-Large',
                  data: [0.85, 0.94, 0.89, 0.84, 0.89]
                }]
              }
            }
          ],
          summary: {
            totalModels: 1,
            totalBenchmarks: 1,
            bestPerformingModel: 'BERT-Large-Uncased',
            averageAccuracy: 0.847,
            keyFindings: [
              'BERT-Large在SST-2情感分析任务上表现最佳，准确率达94%',
              'WNLI任务是所有GLUE任务中最具挑战性的，准确率仅56%',
              '模型在需要常识推理的任务上表现相对较弱',
              '整体GLUE得分84.7，超过基线模型15个百分点'
            ]
          }
        },
        {
          id: 'report_002',
          title: '多模型SuperGLUE性能对比分析',
          type: 'comparison',
          status: 'completed',
          format: 'pdf',
          description: '对比分析GPT-3.5、Claude-3和BERT-Large在SuperGLUE基准测试上的表现差异',
          createdBy: '李四',
          createdAt: '2024-01-15 14:20:00',
          updatedAt: '2024-01-15 15:30:00',
          fileSize: 4.8,
          downloadCount: 23,
          sharedWith: ['张三', '王五', '赵六'],
          tags: ['模型对比', 'SuperGLUE', '性能分析'],
          models: ['GPT-3.5-Turbo', 'Claude-3-Sonnet', 'BERT-Large-Uncased'],
          benchmarks: ['SuperGLUE'],
          metrics: {
            accuracy: 0.723
          },
          charts: [
            {
              type: 'line',
              title: '模型性能趋势对比',
              data: {
                labels: ['BoolQ', 'CB', 'COPA', 'MultiRC', 'ReCoRD', 'RTE', 'WiC', 'WSC'],
                datasets: [
                  { label: 'GPT-3.5', data: [0.87, 0.71, 0.89, 0.68, 0.84, 0.72, 0.75, 0.83] },
                  { label: 'Claude-3', data: [0.91, 0.85, 0.94, 0.75, 0.89, 0.78, 0.82, 0.88] },
                  { label: 'BERT-Large', data: [0.79, 0.68, 0.71, 0.63, 0.76, 0.69, 0.70, 0.65] }
                ]
              }
            }
          ],
          summary: {
            totalModels: 3,
            totalBenchmarks: 1,
            bestPerformingModel: 'Claude-3-Sonnet',
            averageAccuracy: 0.723,
            keyFindings: [
              'Claude-3-Sonnet在大部分SuperGLUE任务中表现最佳',
              'GPT-3.5在COPA常识推理任务中展现出色表现',
              'BERT模型在需要上下文理解的复杂任务中存在明显劣势',
              'Claude-3的整体SuperGLUE得分比其他模型高8-12个百分点'
            ]
          }
        },
        {
          id: 'report_003',
          title: 'HumanEval代码生成能力评估报告',
          type: 'benchmark_analysis',
          status: 'generating',
          format: 'html',
          description: '专注分析各种大语言模型在HumanEval代码生成基准测试上的详细表现',
          createdBy: '王五',
          createdAt: '2024-01-15 17:00:00',
          updatedAt: '2024-01-15 17:00:00',
          fileSize: 0,
          downloadCount: 0,
          sharedWith: [],
          tags: ['代码生成', 'HumanEval', '编程能力'],
          models: ['GPT-4', 'Claude-3-Opus', 'Codex', 'CodeT5'],
          benchmarks: ['HumanEval'],
          metrics: {},
          charts: [],
          summary: {
            totalModels: 4,
            totalBenchmarks: 1,
            bestPerformingModel: '',
            averageAccuracy: 0,
            keyFindings: []
          }
        },
        {
          id: 'report_004',
          title: '2024年第一季度模型评估趋势分析',
          type: 'trend_analysis',
          status: 'completed',
          format: 'excel',
          description: '分析2024年第一季度各类模型在不同基准测试上的性能趋势和变化',
          createdBy: '赵六',
          createdAt: '2024-01-14 09:30:00',
          updatedAt: '2024-01-14 18:45:00',
          fileSize: 8.2,
          downloadCount: 31,
          sharedWith: ['张三', '李四', '王五', '孙七'],
          tags: ['趋势分析', '季度报告', '综合评估'],
          models: ['GPT系列', 'Claude系列', 'BERT系列', 'T5系列'],
          benchmarks: ['GLUE', 'SuperGLUE', 'MMLU', 'HumanEval'],
          metrics: {
            accuracy: 0.782
          },
          charts: [
            {
              type: 'line',
              title: '季度模型性能趋势',
              data: {
                labels: ['1月', '2月', '3月'],
                datasets: [
                  { label: '平均准确率', data: [0.751, 0.768, 0.782] },
                  { label: '新模型数量', data: [8, 12, 15] }
                ]
              }
            }
          ],
          summary: {
            totalModels: 35,
            totalBenchmarks: 4,
            bestPerformingModel: 'Claude-3-Opus',
            averageAccuracy: 0.782,
            keyFindings: [
              '第一季度整体模型性能提升了3.1个百分点',
              'Claude-3系列模型在推理任务上表现突出',
              '代码生成领域的模型性能提升最为显著',
              '多模态理解能力成为新的评估重点'
            ]
          }
        },
        {
          id: 'report_005',
          title: '金融领域专用模型评估报告',
          type: 'custom',
          status: 'failed',
          format: 'pdf',
          description: '针对金融领域定制化模型的专业评估，包括风险评估、欺诈检测、市场分析等任务',
          createdBy: '孙七',
          createdAt: '2024-01-15 11:15:00',
          updatedAt: '2024-01-15 11:30:00',
          fileSize: 0,
          downloadCount: 0,
          sharedWith: [],
          tags: ['金融', '领域专用', '风险评估'],
          models: ['FinanceBERT', 'FinanceGPT'],
          benchmarks: ['Custom-Finance'],
          metrics: {},
          charts: [],
          summary: {
            totalModels: 2,
            totalBenchmarks: 1,
            bestPerformingModel: '',
            averageAccuracy: 0,
            keyFindings: []
          }
        }
      ];

      setReports(reportsData);
    } catch (error) {
      console.error('加载报告数据失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const loadTemplatesData = async () => {
    try {
      const templatesData: ReportTemplate[] = [
        {
          id: 'template_001',
          name: '单模型评估模板',
          description: '适用于单个模型在多个基准测试上的评估报告',
          type: 'single_model',
          sections: ['模型概述', '评估配置', '基准测试结果', '性能分析', '结论与建议'],
          isDefault: true
        },
        {
          id: 'template_002',
          name: '模型对比分析模板',
          description: '适用于多个模型在相同基准测试上的对比分析',
          type: 'comparison',
          sections: ['对比概述', '模型信息', '评估结果对比', '性能差异分析', '推荐建议'],
          isDefault: true
        },
        {
          id: 'template_003',
          name: '基准测试分析模板',
          description: '深入分析特定基准测试的详细表现',
          type: 'benchmark_analysis',
          sections: ['基准测试介绍', '参与模型', '详细结果', '任务分析', '改进建议'],
          isDefault: false
        },
        {
          id: 'template_004',
          name: '趋势分析模板',
          description: '分析模型性能在时间维度上的变化趋势',
          type: 'trend_analysis',
          sections: ['趋势概览', '时间序列分析', '关键变化点', '影响因素', '未来预测'],
          isDefault: false
        }
      ];

      setTemplates(templatesData);
    } catch (error) {
      console.error('加载模板数据失败:', error);
    }
  };

  const handleCreateReport = async () => {
    try {
      const values = await form.validateFields();
      console.log('创建报告:', values);
      setCreateModalVisible(false);
      form.resetFields();
      await loadReportsData();
    } catch (error) {
      console.error('创建报告失败:', error);
    }
  };

  const handleDownloadReport = (reportId: string, format: string) => {
    console.log('下载报告:', reportId, format);
    // 在实际项目中实现下载逻辑
  };

  const handleShareReport = async (reportId: string) => {
    const report = reports.find(r => r.id === reportId);
    if (report) {
      setSelectedReport(report);
      shareForm.setFieldsValue({
        reportId: report.id,
        title: report.title,
        recipients: report.sharedWith
      });
      setShareModalVisible(true);
    }
  };

  const handlePreviewReport = (report: EvaluationReport) => {
    setSelectedReport(report);
    setPreviewDrawerVisible(true);
  };

  const handleDeleteReport = (reportId: string) => {
    Modal.confirm({
      title: '确认删除',
      content: '删除后无法恢复，确定要删除这个报告吗？',
      onOk: async () => {
        console.log('删除报告:', reportId);
        await loadReportsData();
      }
    });
  };

  const getStatusTag = (status: string) => {
    const statusConfig = {
      generating: { color: 'processing', text: '生成中' },
      completed: { color: 'success', text: '已完成' },
      failed: { color: 'error', text: '失败' },
      shared: { color: 'blue', text: '已分享' }
    };
    const config = statusConfig[status as keyof typeof statusConfig];
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const getTypeTag = (type: string) => {
    const typeConfig = {
      single_model: { color: 'blue', text: '单模型评估' },
      comparison: { color: 'green', text: '对比分析' },
      benchmark_analysis: { color: 'orange', text: '基准分析' },
      trend_analysis: { color: 'purple', text: '趋势分析' },
      custom: { color: 'magenta', text: '自定义报告' }
    };
    const config = typeConfig[type as keyof typeof typeConfig];
    return <Tag color={config.color}>{config.text}</Tag>;
  };

  const getFormatIcon = (format: string) => {
    const formatIcons = {
      html: <FileTextOutlined style={{ color: '#1890ff' }} />,
      pdf: <FilePdfOutlined style={{ color: '#ff4d4f' }} />,
      excel: <FileExcelOutlined style={{ color: '#52c41a' }} />,
      json: <FileTextOutlined style={{ color: '#722ed1' }} />
    };
    return formatIcons[format as keyof typeof formatIcons] || <FileTextOutlined />;
  };

  const filteredReports = reports.filter(report => {
    const typeMatch = filterType === 'all' || report.type === filterType;
    const statusMatch = filterStatus === 'all' || report.status === statusMatch;
    const searchMatch = searchText === '' || 
      report.title.toLowerCase().includes(searchText.toLowerCase()) ||
      report.description.toLowerCase().includes(searchText.toLowerCase()) ||
      report.tags.some(tag => tag.toLowerCase().includes(searchText.toLowerCase()));
    
    return typeMatch && statusMatch && searchMatch;
  });

  const columns = [
    {
      title: '报告标题',
      dataIndex: 'title',
      key: 'title',
      width: 250,
      render: (text: string, record: EvaluationReport) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            {getFormatIcon(record.format)}
            <span style={{ fontWeight: 'bold', marginLeft: '8px' }}>{text}</span>
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>{record.id}</Text>
        </div>
      )
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => getTypeTag(type),
      width: 120,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => getStatusTag(status),
      width: 100,
    },
    {
      title: '模型数量',
      dataIndex: 'models',
      key: 'models',
      render: (models: string[]) => models.length,
      width: 100,
    },
    {
      title: '基准测试',
      dataIndex: 'benchmarks',
      key: 'benchmarks',
      render: (benchmarks: string[]) => (
        <Space size={[0, 4]} wrap>
          {benchmarks.map(benchmark => (
            <Tag key={benchmark} size="small">{benchmark}</Tag>
          ))}
        </Space>
      ),
      width: 150,
    },
    {
      title: '标签',
      dataIndex: 'tags',
      key: 'tags',
      render: (tags: string[]) => (
        <Space size={[0, 4]} wrap>
          {tags.slice(0, 2).map(tag => (
            <Tag key={tag} size="small" icon={<TagOutlined />}>{tag}</Tag>
          ))}
          {tags.length > 2 && <Text type="secondary">+{tags.length - 2}</Text>}
        </Space>
      ),
      width: 150,
    },
    {
      title: '文件大小',
      dataIndex: 'fileSize',
      key: 'fileSize',
      render: (size: number) => size > 0 ? `${size}MB` : '-',
      width: 100,
    },
    {
      title: '下载次数',
      dataIndex: 'downloadCount',
      key: 'downloadCount',
      width: 100,
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      width: 150,
    },
    {
      title: '操作',
      key: 'action',
      render: (_, record: EvaluationReport) => (
        <Space size="small">
          <Tooltip title="预览">
            <Button
              type="link"
              size="small"
              icon={<EyeOutlined />}
              onClick={() => handlePreviewReport(record)}
            />
          </Tooltip>
          {record.status === 'completed' && (
            <Tooltip title="下载">
              <Button
                type="link"
                size="small"
                icon={<DownloadOutlined />}
                onClick={() => handleDownloadReport(record.id, record.format)}
              />
            </Tooltip>
          )}
          <Tooltip title="分享">
            <Button
              type="link"
              size="small"
              icon={<ShareAltOutlined />}
              onClick={() => handleShareReport(record.id)}
              disabled={record.status !== 'completed'}
            />
          </Tooltip>
          <Tooltip title="删除">
            <Button
              type="link"
              size="small"
              icon={<DeleteOutlined />}
              onClick={() => handleDeleteReport(record.id)}
              danger
            />
          </Tooltip>
        </Space>
      ),
      width: 150,
    },
  ];

  const completedReports = reports.filter(r => r.status === 'completed').length;
  const generatingReports = reports.filter(r => r.status === 'generating').length;
  const totalDownloads = reports.reduce((sum, r) => sum + r.downloadCount, 0);
  const sharedReports = reports.filter(r => r.sharedWith.length > 0).length;

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>评估报告中心</Title>
        <Text type="secondary">
          管理和生成各种类型的模型评估报告，支持多格式导出和团队协作分享
        </Text>
      </div>

      {/* 报告统计 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={24} sm={6}>
          <Card>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <FileTextOutlined style={{ fontSize: '32px', color: '#1890ff' }} />
              <div style={{ marginLeft: '16px' }}>
                <Text type="secondary">已完成报告</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{completedReports}</div>
              </div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <ExperimentOutlined style={{ fontSize: '32px', color: '#52c41a' }} />
              <div style={{ marginLeft: '16px' }}>
                <Text type="secondary">生成中</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{generatingReports}</div>
              </div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <DownloadOutlined style={{ fontSize: '32px', color: '#722ed1' }} />
              <div style={{ marginLeft: '16px' }}>
                <Text type="secondary">总下载量</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{totalDownloads}</div>
              </div>
            </div>
          </Card>
        </Col>
        <Col xs={24} sm={6}>
          <Card>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <ShareAltOutlined style={{ fontSize: '32px', color: '#fa8c16' }} />
              <div style={{ marginLeft: '16px' }}>
                <Text type="secondary">已分享</Text>
                <div style={{ fontSize: '24px', fontWeight: 'bold' }}>{sharedReports}</div>
              </div>
            </div>
          </Card>
        </Col>
      </Row>

      <Card
        title="评估报告列表"
        extra={
          <Space>
            <Input
              placeholder="搜索报告..."
              prefix={<SearchOutlined />}
              value={searchText}
              onChange={e => setSearchText(e.target.value)}
              style={{ width: 200 }}
            />
            <Select
              placeholder="报告类型"
              value={filterType}
              onChange={setFilterType}
              style={{ width: 150 }}
            >
              <Option value="all">全部类型</Option>
              <Option value="single_model">单模型评估</Option>
              <Option value="comparison">对比分析</Option>
              <Option value="benchmark_analysis">基准分析</Option>
              <Option value="trend_analysis">趋势分析</Option>
              <Option value="custom">自定义报告</Option>
            </Select>
            <Select
              placeholder="状态筛选"
              value={filterStatus}
              onChange={setFilterStatus}
              style={{ width: 120 }}
            >
              <Option value="all">全部状态</Option>
              <Option value="generating">生成中</Option>
              <Option value="completed">已完成</Option>
              <Option value="failed">失败</Option>
              <Option value="shared">已分享</Option>
            </Select>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setCreateModalVisible(true)}
            >
              创建报告
            </Button>
            <Button
              icon={<ReloadOutlined />}
              onClick={loadReportsData}
              loading={loading}
            >
              刷新
            </Button>
          </Space>
        }
      >
        <Table
          dataSource={filteredReports}
          columns={columns}
          rowKey="id"
          pagination={{
            pageSize: 10,
            showSizeChanger: true,
            showQuickJumper: true,
            showTotal: (total, range) => `第 ${range[0]}-${range[1]} 项 / 共 ${total} 项`
          }}
          loading={loading}
          scroll={{ x: 1400 }}
        />
      </Card>

      {/* 创建报告模态框 */}
      <Modal
        title="创建评估报告"
        open={createModalVisible}
        onOk={handleCreateReport}
        onCancel={() => setCreateModalVisible(false)}
        width={600}
        okText="创建"
        cancelText="取消"
      >
        <Form form={form} layout="vertical">
          <Form.Item
            name="title"
            label="报告标题"
            rules={[{ required: true, message: '请输入报告标题' }]}
          >
            <Input placeholder="请输入报告标题" />
          </Form.Item>

          <Form.Item
            name="type"
            label="报告类型"
            rules={[{ required: true, message: '请选择报告类型' }]}
          >
            <Select placeholder="请选择报告类型">
              <Option value="single_model">单模型评估</Option>
              <Option value="comparison">模型对比分析</Option>
              <Option value="benchmark_analysis">基准测试分析</Option>
              <Option value="trend_analysis">趋势分析</Option>
              <Option value="custom">自定义报告</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="template"
            label="使用模板"
            rules={[{ required: true, message: '请选择报告模板' }]}
          >
            <Select placeholder="请选择报告模板">
              {templates.map(template => (
                <Option key={template.id} value={template.id}>
                  {template.name} {template.isDefault && <Tag size="small" color="blue">默认</Tag>}
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="format"
            label="输出格式"
            rules={[{ required: true, message: '请选择输出格式' }]}
          >
            <Select placeholder="请选择输出格式">
              <Option value="html">HTML</Option>
              <Option value="pdf">PDF</Option>
              <Option value="excel">Excel</Option>
              <Option value="json">JSON</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="description"
            label="报告描述"
          >
            <TextArea rows={3} placeholder="请输入报告描述" />
          </Form.Item>

          <Form.Item
            name="models"
            label="包含模型"
            rules={[{ required: true, message: '请选择要包含的模型' }]}
          >
            <Select mode="multiple" placeholder="请选择要包含的模型">
              <Option value="BERT-Large-Uncased">BERT-Large-Uncased</Option>
              <Option value="GPT-3.5-Turbo">GPT-3.5-Turbo</Option>
              <Option value="Claude-3-Sonnet">Claude-3-Sonnet</Option>
              <Option value="T5-3B">T5-3B</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="benchmarks"
            label="基准测试"
            rules={[{ required: true, message: '请选择基准测试' }]}
          >
            <Select mode="multiple" placeholder="请选择基准测试">
              <Option value="GLUE">GLUE</Option>
              <Option value="SuperGLUE">SuperGLUE</Option>
              <Option value="MMLU">MMLU</Option>
              <Option value="HumanEval">HumanEval</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="tags"
            label="标签"
          >
            <Select mode="tags" placeholder="请添加标签">
              <Option value="自然语言理解">自然语言理解</Option>
              <Option value="代码生成">代码生成</Option>
              <Option value="性能对比">性能对比</Option>
              <Option value="趋势分析">趋势分析</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* 分享模态框 */}
      <Modal
        title="分享报告"
        open={shareModalVisible}
        onOk={async () => {
          try {
            const values = await shareForm.validateFields();
            console.log('分享报告:', values);
            setShareModalVisible(false);
            shareForm.resetFields();
          } catch (error) {
            console.error('分享失败:', error);
          }
        }}
        onCancel={() => setShareModalVisible(false)}
        okText="分享"
        cancelText="取消"
      >
        <Form form={shareForm} layout="vertical">
          <Form.Item
            name="recipients"
            label="分享给"
            rules={[{ required: true, message: '请选择分享对象' }]}
          >
            <Select mode="multiple" placeholder="请选择分享对象">
              <Option value="张三">张三</Option>
              <Option value="李四">李四</Option>
              <Option value="王五">王五</Option>
              <Option value="赵六">赵六</Option>
              <Option value="孙七">孙七</Option>
            </Select>
          </Form.Item>

          <Form.Item
            name="message"
            label="分享消息"
          >
            <TextArea rows={3} placeholder="添加分享消息（可选）" />
          </Form.Item>

          <Form.Item
            name="permissions"
            label="权限设置"
            initialValue="view"
          >
            <Select>
              <Option value="view">仅查看</Option>
              <Option value="download">查看和下载</Option>
              <Option value="edit">查看、下载和编辑</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* 预览抽屉 */}
      <Drawer
        title="报告预览"
        placement="right"
        size="large"
        open={previewDrawerVisible}
        onClose={() => setPreviewDrawerVisible(false)}
      >
        {selectedReport && (
          <div>
            <Descriptions bordered column={2} style={{ marginBottom: '24px' }}>
              <Descriptions.Item label="报告标题" span={2}>{selectedReport.title}</Descriptions.Item>
              <Descriptions.Item label="报告类型">{getTypeTag(selectedReport.type)}</Descriptions.Item>
              <Descriptions.Item label="状态">{getStatusTag(selectedReport.status)}</Descriptions.Item>
              <Descriptions.Item label="格式">{selectedReport.format.toUpperCase()}</Descriptions.Item>
              <Descriptions.Item label="文件大小">{selectedReport.fileSize > 0 ? `${selectedReport.fileSize}MB` : '-'}</Descriptions.Item>
              <Descriptions.Item label="创建者">{selectedReport.createdBy}</Descriptions.Item>
              <Descriptions.Item label="创建时间">{selectedReport.createdAt}</Descriptions.Item>
              <Descriptions.Item label="下载次数">{selectedReport.downloadCount}</Descriptions.Item>
              <Descriptions.Item label="分享状态">{selectedReport.sharedWith.length > 0 ? `已分享给${selectedReport.sharedWith.length}人` : '未分享'}</Descriptions.Item>
              <Descriptions.Item label="描述" span={2}>{selectedReport.description}</Descriptions.Item>
            </Descriptions>

            <Card title="包含模型" size="small" style={{ marginBottom: '16px' }}>
              <Space size={[0, 4]} wrap>
                {selectedReport.models.map(model => (
                  <Tag key={model} icon={<TrophyOutlined />}>{model}</Tag>
                ))}
              </Space>
            </Card>

            <Card title="基准测试" size="small" style={{ marginBottom: '16px' }}>
              <Space size={[0, 4]} wrap>
                {selectedReport.benchmarks.map(benchmark => (
                  <Tag key={benchmark} color="blue">{benchmark}</Tag>
                ))}
              </Space>
            </Card>

            {selectedReport.status === 'completed' && selectedReport.summary.keyFindings.length > 0 && (
              <Card title="关键发现" size="small" style={{ marginBottom: '16px' }}>
                <ul>
                  {selectedReport.summary.keyFindings.map((finding, index) => (
                    <li key={index} style={{ marginBottom: '8px' }}>{finding}</li>
                  ))}
                </ul>
              </Card>
            )}

            {selectedReport.sharedWith.length > 0 && (
              <Card title="分享对象" size="small">
                <Space size={[0, 4]} wrap>
                  {selectedReport.sharedWith.map(person => (
                    <Tag key={person} icon={<UserOutlined />} color="green">{person}</Tag>
                  ))}
                </Space>
              </Card>
            )}
          </div>
        )}
      </Drawer>
    </div>
  );
};

export default EvaluationReportsCenterPage;