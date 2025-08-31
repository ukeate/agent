import React, { useState, useEffect } from 'react';
import { Card, Row, Col, Table, Button, Modal, Form, Input, Select, Tag, Space, Progress, Statistic, Tabs, message, Tooltip, Popconfirm, Alert, Badge } from 'antd';
import { BookOutlined, PlusOutlined, EditOutlined, DeleteOutlined, PlayCircleOutlined, CheckCircleOutlined, InfoCircleOutlined, DownloadOutlined, UploadOutlined } from '@ant-design/icons';
import type { ColumnsType } from 'antd/es/table';
import dayjs from 'dayjs';

const { Option } = Select;
const { TextArea } = Input;
const { TabPane } = Tabs;

interface MmluSubject {
  id: string;
  name: string;
  displayName: string;
  category: 'stem' | 'humanities' | 'social_sciences' | 'other';
  description: string;
  samples: number;
  difficulty: 'undergraduate' | 'graduate' | 'professional';
  status: 'active' | 'disabled';
  averageScore?: number;
}

interface MmluBenchmarkConfig {
  id: string;
  name: string;
  description: string;
  selectedSubjects: string[];
  fewShotK: number;
  status: 'draft' | 'active' | 'running' | 'completed';
  createdAt: string;
  results?: MmluResult[];
  overallScore?: number;
}

interface MmluResult {
  subjectId: string;
  subjectName: string;
  accuracy: number;
  categoryAverage: number;
  difficulty: string;
}

const BenchmarkMmluPage: React.FC = () => {
  const [mmluSubjects, setMmluSubjects] = useState<MmluSubject[]>([]);
  const [benchmarkConfigs, setBenchmarkConfigs] = useState<MmluBenchmarkConfig[]>([]);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('subjects');

  const [isSubjectModalVisible, setIsSubjectModalVisible] = useState(false);
  const [isConfigModalVisible, setIsConfigModalVisible] = useState(false);
  const [editingSubject, setEditingSubject] = useState<MmluSubject | null>(null);

  const [subjectForm] = Form.useForm();
  const [configForm] = Form.useForm();

  useEffect(() => {
    loadMockData();
  }, []);

  const loadMockData = () => {
    const mockMmluSubjects: MmluSubject[] = [
      // STEM
      { id: '1', name: 'abstract_algebra', displayName: '抽象代数', category: 'stem', description: '抽象代数的基本概念和定理', samples: 100, difficulty: 'undergraduate', status: 'active', averageScore: 0.32 },
      { id: '2', name: 'astronomy', displayName: '天文学', category: 'stem', description: '天体物理学和天文观测', samples: 152, difficulty: 'graduate', status: 'active', averageScore: 0.68 },
      { id: '3', name: 'college_biology', displayName: '大学生物学', category: 'stem', description: '分子生物学、遗传学、生态学', samples: 144, difficulty: 'undergraduate', status: 'active', averageScore: 0.72 },
      { id: '4', name: 'college_chemistry', displayName: '大学化学', category: 'stem', description: '有机化学、无机化学、物理化学', samples: 100, difficulty: 'undergraduate', status: 'active', averageScore: 0.45 },
      { id: '5', name: 'college_computer_science', displayName: '大学计算机科学', category: 'stem', description: '算法、数据结构、计算理论', samples: 100, difficulty: 'undergraduate', status: 'active', averageScore: 0.58 },
      { id: '6', name: 'college_mathematics', displayName: '大学数学', category: 'stem', description: '微积分、线性代数、概率统计', samples: 100, difficulty: 'undergraduate', status: 'active', averageScore: 0.42 },
      { id: '7', name: 'college_physics', displayName: '大学物理学', category: 'stem', description: '经典力学、电磁学、量子力学', samples: 102, difficulty: 'undergraduate', status: 'active', averageScore: 0.48 },
      
      // Humanities
      { id: '8', name: 'formal_logic', displayName: '形式逻辑', category: 'humanities', description: '命题逻辑、谓词逻辑、推理规则', samples: 126, difficulty: 'graduate', status: 'active', averageScore: 0.39 },
      { id: '9', name: 'high_school_european_history', displayName: '高中欧洲史', category: 'humanities', description: '欧洲历史从中世纪到现代', samples: 165, difficulty: 'undergraduate', status: 'active', averageScore: 0.74 },
      { id: '10', name: 'high_school_us_history', displayName: '高中美国史', category: 'humanities', description: '美国历史从殖民地到现代', samples: 204, difficulty: 'undergraduate', status: 'active', averageScore: 0.78 },
      { id: '11', name: 'high_school_world_history', displayName: '高中世界史', category: 'humanities', description: '世界历史主要事件和发展', samples: 237, difficulty: 'undergraduate', status: 'active', averageScore: 0.79 },
      { id: '12', name: 'philosophy', displayName: '哲学', category: 'humanities', description: '西方哲学史、伦理学、形而上学', samples: 311, difficulty: 'graduate', status: 'active', averageScore: 0.71 },
      { id: '13', name: 'prehistory', displayName: '史前史', category: 'humanities', description: '人类起源和早期文明', samples: 324, difficulty: 'graduate', status: 'active', averageScore: 0.69 },
      
      // Social Sciences
      { id: '14', name: 'econometrics', displayName: '计量经济学', category: 'social_sciences', description: '经济数据分析和统计方法', samples: 114, difficulty: 'graduate', status: 'active', averageScore: 0.47 },
      { id: '15', name: 'high_school_geography', displayName: '高中地理', category: 'social_sciences', description: '自然地理和人文地理', samples: 198, difficulty: 'undergraduate', status: 'active', averageScore: 0.81 },
      { id: '16', name: 'high_school_government_and_politics', displayName: '高中政治学', category: 'social_sciences', description: '政府体制和政治理论', samples: 193, difficulty: 'undergraduate', status: 'active', averageScore: 0.89 },
      { id: '17', name: 'high_school_macroeconomics', displayName: '高中宏观经济学', category: 'social_sciences', description: '国民经济整体运行', samples: 390, difficulty: 'undergraduate', status: 'active', averageScore: 0.66 },
      { id: '18', name: 'high_school_psychology', displayName: '高中心理学', category: 'social_sciences', description: '认知心理学、发展心理学', samples: 545, difficulty: 'undergraduate', status: 'active', averageScore: 0.84 },
      { id: '19', name: 'human_sexuality', displayName: '人类性学', category: 'social_sciences', description: '性行为、性心理、性教育', samples: 131, difficulty: 'graduate', status: 'active', averageScore: 0.69 },
      
      // Other
      { id: '20', name: 'elementary_mathematics', displayName: '小学数学', category: 'other', description: '基础算术和几何', samples: 378, difficulty: 'undergraduate', status: 'active', averageScore: 0.42 },
      { id: '21', name: 'high_school_statistics', displayName: '高中统计学', category: 'other', description: '描述统计和推断统计基础', samples: 216, difficulty: 'undergraduate', status: 'active', averageScore: 0.53 },
      { id: '22', name: 'machine_learning', displayName: '机器学习', category: 'other', description: '监督学习、无监督学习、深度学习', samples: 112, difficulty: 'graduate', status: 'active', averageScore: 0.45 },
    ];

    const mockBenchmarkConfigs: MmluBenchmarkConfig[] = [
      {
        id: '1',
        name: 'MMLU完整基准测试',
        description: '包含所有57个学科的完整MMLU评估',
        selectedSubjects: mmluSubjects.slice(0, 10).map(s => s.name),
        fewShotK: 5,
        status: 'completed',
        createdAt: '2024-01-10T10:00:00',
        overallScore: 0.627,
        results: [
          { subjectId: '1', subjectName: '抽象代数', accuracy: 0.32, categoryAverage: 0.41, difficulty: 'undergraduate' },
          { subjectId: '2', subjectName: '天文学', accuracy: 0.68, categoryAverage: 0.53, difficulty: 'graduate' },
          { subjectId: '3', subjectName: '大学生物学', accuracy: 0.72, categoryAverage: 0.53, difficulty: 'undergraduate' },
        ],
      },
      {
        id: '2',
        name: 'STEM学科专项测试',
        description: '专注于STEM领域的知识评估',
        selectedSubjects: mmluSubjects.filter(s => s.category === 'stem').map(s => s.name),
        fewShotK: 5,
        status: 'running',
        createdAt: '2024-01-18T14:30:00',
      },
    ];

    setMmluSubjects(mockMmluSubjects);
    setBenchmarkConfigs(mockBenchmarkConfigs);
  };

  // 学科操作
  const handleCreateSubject = () => {
    setEditingSubject(null);
    subjectForm.resetFields();
    setIsSubjectModalVisible(true);
  };

  const handleEditSubject = (subject: MmluSubject) => {
    setEditingSubject(subject);
    subjectForm.setFieldsValue(subject);
    setIsSubjectModalVisible(true);
  };

  const handleSaveSubject = async () => {
    try {
      const values = await subjectForm.validateFields();
      
      if (editingSubject) {
        setMmluSubjects(mmluSubjects.map(subject => 
          subject.id === editingSubject.id 
            ? { ...subject, ...values }
            : subject
        ));
        message.success('学科已更新');
      } else {
        const newSubject: MmluSubject = {
          ...values,
          id: Date.now().toString(),
        };
        setMmluSubjects([...mmluSubjects, newSubject]);
        message.success('学科已创建');
      }
      
      setIsSubjectModalVisible(false);
      subjectForm.resetFields();
    } catch (error) {
      message.error('保存失败，请检查输入');
    }
  };

  // 基准配置操作
  const handleCreateConfig = () => {
    configForm.resetFields();
    setIsConfigModalVisible(true);
  };

  const handleRunBenchmark = (id: string) => {
    setBenchmarkConfigs(benchmarkConfigs.map(config => 
      config.id === id 
        ? { ...config, status: 'running' }
        : config
    ));
    message.success('MMLU基准测试已启动');
  };

  const handleSaveConfig = async () => {
    try {
      const values = await configForm.validateFields();
      
      const newConfig: MmluBenchmarkConfig = {
        ...values,
        id: Date.now().toString(),
        status: 'draft',
        createdAt: new Date().toISOString(),
      };
      setBenchmarkConfigs([...benchmarkConfigs, newConfig]);
      message.success('配置已创建');
      
      setIsConfigModalVisible(false);
      configForm.resetFields();
    } catch (error) {
      message.error('保存失败，请检查输入');
    }
  };

  // 表格列定义
  const subjectColumns: ColumnsType<MmluSubject> = [
    {
      title: '学科名称',
      key: 'name',
      render: (record: MmluSubject) => (
        <div>
          <div style={{ fontWeight: 500 }}>{record.displayName}</div>
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
          stem: 'blue',
          humanities: 'green',
          social_sciences: 'orange',
          other: 'purple',
        };
        const labels = {
          stem: 'STEM',
          humanities: '人文学科',
          social_sciences: '社会科学',
          other: '其他',
        };
        return (
          <Tag color={colors[category as keyof typeof colors]}>
            {labels[category as keyof typeof labels]}
          </Tag>
        );
      },
    },
    {
      title: '难度',
      dataIndex: 'difficulty',
      key: 'difficulty',
      render: (difficulty: string) => {
        const colors = {
          undergraduate: 'green',
          graduate: 'orange',
          professional: 'red',
        };
        const labels = {
          undergraduate: '本科',
          graduate: '研究生',
          professional: '专业',
        };
        return (
          <Tag color={colors[difficulty as keyof typeof colors]}>
            {labels[difficulty as keyof typeof labels]}
          </Tag>
        );
      },
    },
    {
      title: '题目数量',
      dataIndex: 'samples',
      key: 'samples',
      render: (samples: number) => samples.toLocaleString(),
    },
    {
      title: '平均得分',
      dataIndex: 'averageScore',
      key: 'averageScore',
      render: (score?: number) => score ? (
        <div>
          <Progress 
            percent={score * 100}
            size="small"
            format={(percent) => `${percent?.toFixed(1)}%`}
          />
        </div>
      ) : '暂无数据',
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'active' ? 'green' : 'orange'}>
          {status === 'active' ? '启用' : '禁用'}
        </Tag>
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: MmluSubject) => (
        <Space>
          <Tooltip title="编辑">
            <Button 
              type="text" 
              icon={<EditOutlined />} 
              size="small"
              onClick={() => handleEditSubject(record)}
            />
          </Tooltip>
          <Tooltip title="查看详情">
            <Button type="text" icon={<InfoCircleOutlined />} size="small" />
          </Tooltip>
        </Space>
      ),
    },
  ];

  const configColumns: ColumnsType<MmluBenchmarkConfig> = [
    {
      title: '配置名称',
      key: 'name',
      render: (record: MmluBenchmarkConfig) => (
        <div>
          <div style={{ fontWeight: 500 }}>
            {record.name}
            {record.overallScore && (
              <Badge 
                count={`${(record.overallScore * 100).toFixed(1)}%`} 
                style={{ backgroundColor: '#52c41a', marginLeft: '8px' }}
              />
            )}
          </div>
          <div style={{ fontSize: '12px', color: '#666' }}>
            {record.description}
          </div>
        </div>
      ),
    },
    {
      title: '包含学科',
      dataIndex: 'selectedSubjects',
      key: 'selectedSubjects',
      render: (subjects: string[]) => (
        <div>
          <div style={{ marginBottom: '4px' }}>共{subjects.length}个学科</div>
          <div>
            {subjects.slice(0, 3).map(subject => (
              <Tag key={subject} size="small" style={{ marginBottom: '2px' }}>
                {mmluSubjects.find(s => s.name === subject)?.displayName || subject}
              </Tag>
            ))}
            {subjects.length > 3 && (
              <Tag size="small">+{subjects.length - 3}个更多</Tag>
            )}
          </div>
        </div>
      ),
    },
    {
      title: 'Few-shot',
      dataIndex: 'fewShotK',
      key: 'fewShotK',
      render: (k: number) => `${k}-shot`,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = {
          draft: 'default',
          active: 'blue',
          running: 'orange',
          completed: 'green',
        };
        const labels = {
          draft: '草稿',
          active: '就绪',
          running: '运行中',
          completed: '已完成',
        };
        
        return (
          <Tag color={colors[status as keyof typeof colors]}>
            {labels[status as keyof typeof labels]}
          </Tag>
        );
      },
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt',
      render: (date: string) => dayjs(date).format('YYYY-MM-DD HH:mm'),
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: MmluBenchmarkConfig) => (
        <Space>
          {record.status === 'active' && (
            <Tooltip title="运行基准测试">
              <Button 
                type="text" 
                icon={<PlayCircleOutlined />} 
                size="small"
                onClick={() => handleRunBenchmark(record.id)}
              />
            </Tooltip>
          )}
          {record.status === 'completed' && (
            <Tooltip title="查看结果">
              <Button type="text" icon={<CheckCircleOutlined />} size="small" />
            </Tooltip>
          )}
          <Tooltip title="编辑">
            <Button type="text" icon={<EditOutlined />} size="small" />
          </Tooltip>
          <Tooltip title="下载报告">
            <Button type="text" icon={<DownloadOutlined />} size="small" />
          </Tooltip>
        </Space>
      ),
    },
  ];

  // 计算分类统计
  const categoryStats = mmluSubjects.reduce((stats, subject) => {
    stats[subject.category] = (stats[subject.category] || 0) + 1;
    return stats;
  }, {} as Record<string, number>);

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1><BookOutlined /> MMLU基准测试管理</h1>
        <p>管理MMLU（Massive Multitask Language Understanding）基准测试学科和配置</p>
      </div>

      <Alert
        message="MMLU基准测试说明"
        description="MMLU包含57个学科的选择题，涵盖STEM、人文、社会科学等领域，从小学到专业水平。支持0-shot到few-shot评估，是衡量模型世界知识和问题解决能力的重要基准。"
        type="info"
        showIcon
        style={{ marginBottom: '16px' }}
      />

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="学科管理" key="subjects">
          <div style={{ marginBottom: '16px' }}>
            <Space>
              <Button 
                type="primary" 
                icon={<PlusOutlined />}
                onClick={handleCreateSubject}
              >
                新增学科
              </Button>
              <Button icon={<UploadOutlined />}>
                导入数据集
              </Button>
              <Button icon={<DownloadOutlined />}>
                导出配置
              </Button>
            </Space>
          </div>
          
          <Table
            columns={subjectColumns}
            dataSource={mmluSubjects}
            rowKey="id"
            loading={loading}
            pagination={{ pageSize: 15 }}
          />
        </TabPane>

        <TabPane tab="基准配置管理" key="configs">
          <div style={{ marginBottom: '16px' }}>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={handleCreateConfig}
            >
              新建基准配置
            </Button>
          </div>
          
          <Table
            columns={configColumns}
            dataSource={benchmarkConfigs}
            rowKey="id"
            loading={loading}
            pagination={{ pageSize: 10 }}
            expandable={{
              expandedRowRender: (record) => (
                <div style={{ padding: '16px' }}>
                  {record.results ? (
                    <Row gutter={16}>
                      <Col span={24}>
                        <h4>评估结果 (整体得分: {(record.overallScore! * 100).toFixed(1)}%)</h4>
                        <Row gutter={16}>
                          {record.results.map(result => (
                            <Col span={8} key={result.subjectId} style={{ marginBottom: '16px' }}>
                              <Card size="small">
                                <div>
                                  <div style={{ fontWeight: 500 }}>{result.subjectName}</div>
                                  <div style={{ fontSize: '18px', color: '#1890ff', marginTop: '8px' }}>
                                    {(result.accuracy * 100).toFixed(1)}%
                                  </div>
                                  <div style={{ fontSize: '12px', color: '#666' }}>
                                    分类平均: {(result.categoryAverage * 100).toFixed(1)}%
                                  </div>
                                </div>
                              </Card>
                            </Col>
                          ))}
                        </Row>
                      </Col>
                    </Row>
                  ) : (
                    <div>该配置尚未运行或正在运行中</div>
                  )}
                </div>
              ),
              rowExpandable: (record) => record.status === 'completed',
            }}
          />
        </TabPane>

        <TabPane tab="数据集统计" key="statistics">
          <Row gutter={[16, 16]}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="总学科数"
                  value={mmluSubjects.length}
                  suffix="个"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="启用学科"
                  value={mmluSubjects.filter(s => s.status === 'active').length}
                  suffix="个"
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="总题目数"
                  value={mmluSubjects.reduce((sum, subject) => sum + subject.samples, 0)}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="配置数"
                  value={benchmarkConfigs.length}
                  suffix="个"
                />
              </Card>
            </Col>
          </Row>

          <Row gutter={16} style={{ marginTop: '16px' }}>
            <Col span={8}>
              <Card title="学科分类分布">
                <div>
                  <div>STEM: {categoryStats.stem || 0}个</div>
                  <div>人文学科: {categoryStats.humanities || 0}个</div>
                  <div>社会科学: {categoryStats.social_sciences || 0}个</div>
                  <div>其他: {categoryStats.other || 0}个</div>
                </div>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="难度分布">
                <div>
                  <div>本科: {mmluSubjects.filter(s => s.difficulty === 'undergraduate').length}个</div>
                  <div>研究生: {mmluSubjects.filter(s => s.difficulty === 'graduate').length}个</div>
                  <div>专业: {mmluSubjects.filter(s => s.difficulty === 'professional').length}个</div>
                </div>
              </Card>
            </Col>
            <Col span={8}>
              <Card title="表现最佳学科">
                <div>
                  {mmluSubjects
                    .filter(s => s.averageScore)
                    .sort((a, b) => (b.averageScore || 0) - (a.averageScore || 0))
                    .slice(0, 5)
                    .map(subject => (
                      <div key={subject.id} style={{ marginBottom: '4px' }}>
                        {subject.displayName}: {((subject.averageScore || 0) * 100).toFixed(1)}%
                      </div>
                    ))
                  }
                </div>
              </Card>
            </Col>
          </Row>
        </TabPane>
      </Tabs>

      {/* 学科编辑模态框 */}
      <Modal
        title={editingSubject ? "编辑MMLU学科" : "新增MMLU学科"}
        open={isSubjectModalVisible}
        onOk={handleSaveSubject}
        onCancel={() => {
          setIsSubjectModalVisible(false);
          subjectForm.resetFields();
        }}
        width={600}
      >
        <Form
          form={subjectForm}
          layout="vertical"
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item
                name="name"
                label="学科英文名"
                rules={[{ required: true, message: '请输入学科英文名' }]}
              >
                <Input placeholder="如: abstract_algebra" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="displayName"
                label="学科中文名"
                rules={[{ required: true, message: '请输入学科中文名' }]}
              >
                <Input placeholder="如: 抽象代数" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="description"
            label="学科描述"
            rules={[{ required: true, message: '请输入学科描述' }]}
          >
            <TextArea rows={3} placeholder="详细描述学科内容和范围" />
          </Form.Item>

          <Row gutter={16}>
            <Col span={8}>
              <Form.Item
                name="category"
                label="学科分类"
                rules={[{ required: true, message: '请选择学科分类' }]}
              >
                <Select placeholder="选择学科分类">
                  <Option value="stem">STEM</Option>
                  <Option value="humanities">人文学科</Option>
                  <Option value="social_sciences">社会科学</Option>
                  <Option value="other">其他</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="difficulty"
                label="难度等级"
                rules={[{ required: true, message: '请选择难度等级' }]}
              >
                <Select placeholder="选择难度等级">
                  <Option value="undergraduate">本科</Option>
                  <Option value="graduate">研究生</Option>
                  <Option value="professional">专业</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item
                name="samples"
                label="题目数量"
                rules={[{ required: true, message: '请输入题目数量' }]}
              >
                <Input type="number" min={1} placeholder="题目数量" />
              </Form.Item>
            </Col>
          </Row>

          <Form.Item
            name="status"
            label="状态"
            rules={[{ required: true, message: '请选择状态' }]}
          >
            <Select placeholder="选择状态">
              <Option value="active">启用</Option>
              <Option value="disabled">禁用</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>

      {/* 基准配置模态框 */}
      <Modal
        title="创建MMLU基准配置"
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
            <Input placeholder="输入基准配置名称" />
          </Form.Item>

          <Form.Item
            name="description"
            label="配置描述"
            rules={[{ required: true, message: '请输入配置描述' }]}
          >
            <TextArea rows={3} placeholder="描述这个基准配置的用途" />
          </Form.Item>

          <Form.Item
            name="selectedSubjects"
            label="选择学科"
            rules={[{ required: true, message: '请选择至少一个学科' }]}
          >
            <Select
              mode="multiple"
              placeholder="选择要包含的MMLU学科"
              showSearch
              filterOption={(input, option) =>
                (option?.children as string)?.toLowerCase().includes(input.toLowerCase())
              }
            >
              {mmluSubjects.filter(subject => subject.status === 'active').map(subject => (
                <Option key={subject.id} value={subject.name}>
                  {subject.displayName} ({subject.category})
                </Option>
              ))}
            </Select>
          </Form.Item>

          <Form.Item
            name="fewShotK"
            label="Few-shot示例数"
            rules={[{ required: true, message: '请选择Few-shot示例数' }]}
          >
            <Select placeholder="选择Few-shot示例数">
              <Option value={0}>0-shot (零样本)</Option>
              <Option value={1}>1-shot</Option>
              <Option value={3}>3-shot</Option>
              <Option value={5}>5-shot</Option>
              <Option value={10}>10-shot</Option>
            </Select>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default BenchmarkMmluPage;