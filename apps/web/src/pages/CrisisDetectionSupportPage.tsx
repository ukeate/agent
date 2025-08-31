import React, { useState, useEffect } from 'react';
import {
  Card,
  Row,
  Col,
  Statistic,
  Table,
  Tag,
  Button,
  Space,
  Alert,
  Modal,
  Form,
  Input,
  Select,
  Typography,
  Divider,
  Timeline,
  Badge,
  Tooltip,
  Progress,
  Avatar,
  List,
  Spin
} from 'antd';
import {
  AlertTriangleIcon,
  PhoneIcon,
  ShieldIcon,
  ClockIcon,
  UserIcon,
  MessageCircleIcon,
  HeartIcon,
  BrainIcon,
  ActivityIcon,
  SirenIcon
} from 'lucide-react';

const { Title, Text } = Typography;
const { TextArea } = Input;
const { Option } = Select;

interface CrisisIndicator {
  type: string;
  score: number;
  evidence: any;
  description: string;
}

interface CrisisAssessment {
  user_id: string;
  crisis_detected: boolean;
  severity_level: 'mild' | 'moderate' | 'severe' | 'critical';
  crisis_type: string;
  indicators: CrisisIndicator[];
  risk_score: number;
  confidence: number;
  immediate_actions: string[];
  professional_required: boolean;
  monitoring_level: string;
  check_frequency: string;
  timestamp: string;
}

interface EmergencyContact {
  id: string;
  name: string;
  type: 'hotline' | 'counselor' | 'emergency' | 'family';
  phone: string;
  availability: '24/7' | 'business_hours' | 'on_call';
  response_time: string;
}

interface SupportResource {
  id: string;
  title: string;
  type: 'article' | 'video' | 'audio' | 'interactive';
  description: string;
  crisis_type: string[];
  effectiveness_rating: number;
}

const CrisisDetectionSupportPage: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [crisisAssessments, setCrisisAssessments] = useState<CrisisAssessment[]>([]);
  const [emergencyContacts, setEmergencyContacts] = useState<EmergencyContact[]>([]);
  const [supportResources, setSupportResources] = useState<SupportResource[]>([]);
  const [detectionModalVisible, setDetectionModalVisible] = useState(false);
  const [responseModalVisible, setResponseModalVisible] = useState(false);
  const [selectedCrisis, setSelectedCrisis] = useState<CrisisAssessment | null>(null);
  
  const [stats, setStats] = useState({
    activeCrises: 0,
    criticalCases: 0,
    averageResponseTime: '0',
    interventionSuccess: 0
  });

  const [form] = Form.useForm();
  const [responseForm] = Form.useForm();

  // 模拟数据
  useEffect(() => {
    const mockCrisisAssessments: CrisisAssessment[] = [
      {
        user_id: 'user_001',
        crisis_detected: true,
        severity_level: 'severe',
        crisis_type: 'suicidal_ideation',
        indicators: [
          {
            type: 'language_indicator',
            score: 0.9,
            evidence: { keywords: ['不想活了', '结束一切'] },
            description: '检测到自杀相关语言表达'
          },
          {
            type: 'emotional_indicator',
            score: 0.85,
            evidence: { emotion: 'despair', intensity: 0.9 },
            description: '极度绝望情感状态'
          }
        ],
        risk_score: 0.92,
        confidence: 0.88,
        immediate_actions: [
          '立即联系紧急热线',
          '通知专业心理医生',
          '确保用户安全环境',
          '持续监控和陪伴'
        ],
        professional_required: true,
        monitoring_level: 'critical',
        check_frequency: '每小时',
        timestamp: new Date().toISOString()
      },
      {
        user_id: 'user_002',
        crisis_detected: true,
        severity_level: 'moderate',
        crisis_type: 'anxiety_attack',
        indicators: [
          {
            type: 'behavioral_indicator',
            score: 0.7,
            evidence: { rapid_responses: true, repetitive_patterns: true },
            description: '检测到焦虑发作行为模式'
          }
        ],
        risk_score: 0.65,
        confidence: 0.75,
        immediate_actions: [
          '提供呼吸练习指导',
          '播放舒缓音频',
          '联系心理咨询师'
        ],
        professional_required: false,
        monitoring_level: 'elevated',
        check_frequency: '每2小时',
        timestamp: new Date(Date.now() - 3600000).toISOString()
      }
    ];

    const mockEmergencyContacts: EmergencyContact[] = [
      {
        id: '1',
        name: '国家心理危机干预热线',
        type: 'hotline',
        phone: '400-161-9995',
        availability: '24/7',
        response_time: '立即'
      },
      {
        id: '2',
        name: '专业心理医生 - 张医生',
        type: 'counselor',
        phone: '138-0000-0000',
        availability: 'business_hours',
        response_time: '30分钟内'
      },
      {
        id: '3',
        name: '急救服务',
        type: 'emergency',
        phone: '120',
        availability: '24/7',
        response_time: '立即'
      }
    ];

    const mockSupportResources: SupportResource[] = [
      {
        id: '1',
        title: '呼吸放松练习',
        type: 'interactive',
        description: '通过引导式深呼吸帮助缓解焦虑和恐慌',
        crisis_type: ['anxiety_attack', 'panic_disorder'],
        effectiveness_rating: 4.8
      },
      {
        id: '2',
        title: '危机时刻生存指南',
        type: 'article',
        description: '实用的危机应对策略和自我保护技巧',
        crisis_type: ['suicidal_ideation', 'depression'],
        effectiveness_rating: 4.6
      },
      {
        id: '3',
        title: '正念冥想音频',
        type: 'audio',
        description: '10分钟平静心灵的引导冥想',
        crisis_type: ['anxiety_attack', 'stress_overload'],
        effectiveness_rating: 4.4
      }
    ];

    setCrisisAssessments(mockCrisisAssessments);
    setEmergencyContacts(mockEmergencyContacts);
    setSupportResources(mockSupportResources);
    setStats({
      activeCrises: mockCrisisAssessments.length,
      criticalCases: mockCrisisAssessments.filter(c => ['severe', 'critical'].includes(c.severity_level)).length,
      averageResponseTime: '3.2分钟',
      interventionSuccess: 94.5
    });
  }, []);

  const severityColor = (level: string) => {
    const colors: Record<string, string> = {
      'mild': '#52c41a',
      'moderate': '#faad14',
      'severe': '#f5222d',
      'critical': '#722ed1'
    };
    return colors[level] || '#d9d9d9';
  };

  const contactTypeIcon = (type: string) => {
    const icons: Record<string, React.ReactNode> = {
      'hotline': <PhoneIcon size={16} />,
      'counselor': <BrainIcon size={16} />,
      'emergency': <SirenIcon size={16} />,
      'family': <HeartIcon size={16} />
    };
    return icons[type] || <UserIcon size={16} />;
  };

  const handleCrisisDetection = async (values: any) => {
    setLoading(true);
    try {
      // 模拟危机检测API
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      // 模拟检测结果
      const hasKeywords = values.message.includes('想死') || values.message.includes('不想活');
      const newAssessment: CrisisAssessment = {
        user_id: values.user_id,
        crisis_detected: hasKeywords,
        severity_level: hasKeywords ? 'severe' : 'mild',
        crisis_type: hasKeywords ? 'suicidal_ideation' : 'general_distress',
        indicators: hasKeywords ? [
          {
            type: 'language_indicator',
            score: 0.9,
            evidence: { detected_keywords: ['想死', '不想活'] },
            description: '检测到危机相关语言'
          }
        ] : [],
        risk_score: hasKeywords ? 0.85 : 0.3,
        confidence: 0.82,
        immediate_actions: hasKeywords ? [
          '立即联系专业支持',
          '确保用户安全',
          '提供紧急资源'
        ] : ['提供情感支持', '监控状态变化'],
        professional_required: hasKeywords,
        monitoring_level: hasKeywords ? 'critical' : 'normal',
        check_frequency: hasKeywords ? '每30分钟' : '每日',
        timestamp: new Date().toISOString()
      };

      setCrisisAssessments(prev => [newAssessment, ...prev]);
      setDetectionModalVisible(false);
      form.resetFields();
      
      if (hasKeywords) {
        setSelectedCrisis(newAssessment);
        setResponseModalVisible(true);
      }
    } catch (error) {
      console.error('危机检测失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleEmergencyResponse = async (values: any) => {
    setLoading(true);
    try {
      // 模拟紧急响应
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setResponseModalVisible(false);
      responseForm.resetFields();
      setSelectedCrisis(null);
    } catch (error) {
      console.error('紧急响应失败:', error);
    } finally {
      setLoading(false);
    }
  };

  const crisisColumns = [
    {
      title: '用户',
      dataIndex: 'user_id',
      key: 'user_id',
      render: (userId: string) => (
        <Space>
          <Avatar icon={<UserIcon size={14} />} size="small" />
          {userId}
        </Space>
      )
    },
    {
      title: '严重程度',
      dataIndex: 'severity_level',
      key: 'severity_level',
      render: (level: string, record: CrisisAssessment) => (
        <Space>
          <Badge color={severityColor(level)} />
          <Tag color={severityColor(level)}>{level.toUpperCase()}</Tag>
          {record.professional_required && (
            <Tooltip title="需要专业干预">
              <BrainIcon size={14} color="#f5222d" />
            </Tooltip>
          )}
        </Space>
      )
    },
    {
      title: '危机类型',
      dataIndex: 'crisis_type',
      key: 'crisis_type',
      render: (type: string) => {
        const typeMap: Record<string, { label: string; color: string }> = {
          'suicidal_ideation': { label: '自杀倾向', color: 'red' },
          'anxiety_attack': { label: '焦虑发作', color: 'orange' },
          'depression': { label: '严重抑郁', color: 'purple' },
          'general_distress': { label: '一般困扰', color: 'blue' }
        };
        const config = typeMap[type] || { label: type, color: 'default' };
        return <Tag color={config.color}>{config.label}</Tag>;
      }
    },
    {
      title: '风险分数',
      dataIndex: 'risk_score',
      key: 'risk_score',
      render: (score: number) => (
        <Progress
          percent={score * 100}
          size="small"
          strokeColor={score > 0.8 ? '#f5222d' : score > 0.6 ? '#faad14' : '#52c41a'}
        />
      )
    },
    {
      title: '监控级别',
      dataIndex: 'monitoring_level',
      key: 'monitoring_level',
      render: (level: string) => {
        const levelMap: Record<string, string> = {
          'normal': 'default',
          'elevated': 'orange',
          'critical': 'red'
        };
        return <Tag color={levelMap[level]}>{level.toUpperCase()}</Tag>;
      }
    },
    {
      title: '检查频率',
      dataIndex: 'check_frequency',
      key: 'check_frequency',
      render: (frequency: string) => (
        <Space>
          <ClockIcon size={14} />
          {frequency}
        </Space>
      )
    },
    {
      title: '操作',
      key: 'actions',
      render: (record: CrisisAssessment) => (
        <Space>
          <Button 
            size="small"
            onClick={() => {
              setSelectedCrisis(record);
              setResponseModalVisible(true);
            }}
          >
            详情
          </Button>
          {record.crisis_detected && (
            <Button size="small" type="primary" danger>
              紧急响应
            </Button>
          )}
        </Space>
      )
    }
  ];

  return (
    <div style={{ padding: '24px' }}>
      {/* 页面标题 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Card>
            <Space align="center">
              <SirenIcon size={32} />
              <div>
                <Title level={2} style={{ margin: 0 }}>危机检测与紧急支持</Title>
                <Text type="secondary">24/7危机监测、即时响应和专业支持服务</Text>
              </div>
            </Space>
          </Card>
        </Col>
      </Row>

      {/* 统计卡片 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="活跃危机"
              value={stats.activeCrises}
              prefix={<AlertTriangleIcon size={16} />}
              valueStyle={{ color: '#f5222d' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="严重案例"
              value={stats.criticalCases}
              prefix={<SirenIcon size={16} />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="平均响应时间"
              value={stats.averageResponseTime}
              prefix={<ClockIcon size={16} />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card>
            <Statistic
              title="干预成功率"
              value={stats.interventionSuccess}
              suffix="%"
              prefix={<ShieldIcon size={16} />}
              valueStyle={{ color: '#52c41a' }}
            />
          </Card>
        </Col>
      </Row>

      {/* 紧急预警 */}
      {stats.criticalCases > 0 && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col span={24}>
            <Alert
              message={`紧急预警：检测到 ${stats.criticalCases} 个严重危机案例`}
              description="系统已识别出需要立即干预的高危情况，请优先处理严重级别的案例。"
              type="error"
              showIcon
              action={
                <Button size="small" type="primary" danger>
                  立即处理
                </Button>
              }
            />
          </Col>
        </Row>
      )}

      {/* 操作按钮 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col span={24}>
          <Space>
            <Button 
              type="primary" 
              icon={<AlertTriangleIcon size={16} />}
              onClick={() => setDetectionModalVisible(true)}
            >
              手动危机检测
            </Button>
            <Button 
              icon={<PhoneIcon size={16} />}
              type="primary"
              ghost
            >
              紧急联系人
            </Button>
            <Button icon={<HeartIcon size={16} />}>
              支持资源
            </Button>
            <Button icon={<ActivityIcon size={16} />}>
              实时监控
            </Button>
          </Space>
        </Col>
      </Row>

      {/* 主要内容区域 */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {/* 危机检测记录 */}
        <Col span={24}>
          <Card 
            title={
              <Space>
                <AlertTriangleIcon size={20} />
                危机检测记录
              </Space>
            }
            extra={<Button type="link">导出报告</Button>}
          >
            <Table
              dataSource={crisisAssessments}
              columns={crisisColumns}
              rowKey="user_id"
              pagination={{ pageSize: 10 }}
              rowClassName={(record) => record.severity_level === 'critical' ? 'critical-row' : ''}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        {/* 紧急联系人 */}
        <Col xs={24} lg={12}>
          <Card 
            title={
              <Space>
                <PhoneIcon size={20} />
                紧急联系人
              </Space>
            }
            extra={<Button type="link" size="small">管理</Button>}
          >
            <List
              itemLayout="horizontal"
              dataSource={emergencyContacts}
              renderItem={(contact) => (
                <List.Item
                  actions={[
                    <Button 
                      type="primary" 
                      size="small"
                      icon={<PhoneIcon size={14} />}
                    >
                      联系
                    </Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={
                      <Avatar icon={contactTypeIcon(contact.type)} />
                    }
                    title={contact.name}
                    description={
                      <Space direction="vertical" size="small">
                        <Text>{contact.phone}</Text>
                        <Space>
                          <Tag size="small">{contact.availability}</Tag>
                          <Text type="secondary" style={{ fontSize: '12px' }}>
                            {contact.response_time}
                          </Text>
                        </Space>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>

        {/* 支持资源 */}
        <Col xs={24} lg={12}>
          <Card 
            title={
              <Space>
                <HeartIcon size={20} />
                支持资源库
              </Space>
            }
            extra={<Button type="link" size="small">查看全部</Button>}
          >
            <List
              itemLayout="horizontal"
              dataSource={supportResources}
              renderItem={(resource) => (
                <List.Item
                  actions={[
                    <Button size="small">使用</Button>
                  ]}
                >
                  <List.Item.Meta
                    title={resource.title}
                    description={
                      <Space direction="vertical" size="small">
                        <Text style={{ fontSize: '12px' }}>
                          {resource.description}
                        </Text>
                        <Space>
                          <Tag size="small">{resource.type}</Tag>
                          <Progress
                            percent={resource.effectiveness_rating * 20}
                            size="small"
                            style={{ width: 60 }}
                            format={() => resource.effectiveness_rating.toFixed(1)}
                          />
                        </Space>
                      </Space>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
      </Row>

      {/* 危机检测模态框 */}
      <Modal
        title="手动危机检测"
        open={detectionModalVisible}
        onCancel={() => {
          setDetectionModalVisible(false);
          form.resetFields();
        }}
        footer={null}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCrisisDetection}
        >
          <Form.Item
            name="user_id"
            label="目标用户"
            rules={[{ required: true, message: '请输入用户ID' }]}
          >
            <Input placeholder="输入用户ID" />
          </Form.Item>

          <Form.Item
            name="message"
            label="用户消息/行为描述"
            rules={[{ required: true, message: '请输入需要检测的内容' }]}
          >
            <TextArea 
              placeholder="输入用户的消息内容或行为描述..."
              rows={4}
            />
          </Form.Item>

          <Form.Item
            name="context"
            label="上下文信息"
          >
            <TextArea 
              placeholder="输入相关背景信息（可选）..."
              rows={2}
            />
          </Form.Item>

          <Form.Item>
            <Space>
              <Button type="primary" htmlType="submit" loading={loading}>
                开始检测
              </Button>
              <Button onClick={() => setDetectionModalVisible(false)}>
                取消
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>

      {/* 危机详情/响应模态框 */}
      <Modal
        title={selectedCrisis ? "危机详情与响应" : "紧急响应"}
        open={responseModalVisible}
        onCancel={() => {
          setResponseModalVisible(false);
          setSelectedCrisis(null);
          responseForm.resetFields();
        }}
        footer={null}
        width={800}
      >
        {selectedCrisis && (
          <div>
            {/* 危机基本信息 */}
            <Card size="small" title="危机概况" style={{ marginBottom: 16 }}>
              <Row gutter={16}>
                <Col span={12}>
                  <Space direction="vertical">
                    <div>
                      <Text strong>用户ID: </Text>
                      <Text>{selectedCrisis.user_id}</Text>
                    </div>
                    <div>
                      <Text strong>严重程度: </Text>
                      <Tag color={severityColor(selectedCrisis.severity_level)}>
                        {selectedCrisis.severity_level.toUpperCase()}
                      </Tag>
                    </div>
                    <div>
                      <Text strong>风险分数: </Text>
                      <Progress 
                        percent={selectedCrisis.risk_score * 100} 
                        size="small"
                        style={{ width: 100 }}
                      />
                    </div>
                  </Space>
                </Col>
                <Col span={12}>
                  <Space direction="vertical">
                    <div>
                      <Text strong>检测时间: </Text>
                      <Text>{new Date(selectedCrisis.timestamp).toLocaleString()}</Text>
                    </div>
                    <div>
                      <Text strong>需要专业干预: </Text>
                      <Badge 
                        status={selectedCrisis.professional_required ? 'error' : 'default'}
                        text={selectedCrisis.professional_required ? '是' : '否'}
                      />
                    </div>
                    <div>
                      <Text strong>置信度: </Text>
                      <Text>{(selectedCrisis.confidence * 100).toFixed(1)}%</Text>
                    </div>
                  </Space>
                </Col>
              </Row>
            </Card>

            {/* 危机指标 */}
            <Card size="small" title="检测指标" style={{ marginBottom: 16 }}>
              {selectedCrisis.indicators.map((indicator, index) => (
                <Card.Grid key={index} style={{ width: '100%', padding: '12px' }}>
                  <Row>
                    <Col span={18}>
                      <Space direction="vertical" size="small">
                        <Text strong>{indicator.type}</Text>
                        <Text style={{ fontSize: '12px' }}>{indicator.description}</Text>
                      </Space>
                    </Col>
                    <Col span={6}>
                      <Progress
                        percent={indicator.score * 100}
                        size="small"
                        strokeColor={indicator.score > 0.7 ? '#f5222d' : '#faad14'}
                      />
                    </Col>
                  </Row>
                </Card.Grid>
              ))}
            </Card>

            {/* 即时行动建议 */}
            <Card size="small" title="即时行动建议" style={{ marginBottom: 16 }}>
              <Timeline>
                {selectedCrisis.immediate_actions.map((action, index) => (
                  <Timeline.Item 
                    key={index}
                    color={index === 0 ? 'red' : 'blue'}
                  >
                    {action}
                  </Timeline.Item>
                ))}
              </Timeline>
            </Card>

            {/* 响应操作 */}
            <Card size="small" title="响应操作">
              <Form
                form={responseForm}
                layout="vertical"
                onFinish={handleEmergencyResponse}
              >
                <Form.Item
                  name="response_type"
                  label="响应类型"
                  rules={[{ required: true }]}
                >
                  <Select placeholder="选择响应类型">
                    <Option value="immediate_contact">立即联系</Option>
                    <Option value="professional_referral">专业转介</Option>
                    <Option value="emergency_services">紧急服务</Option>
                    <Option value="family_notification">家属通知</Option>
                  </Select>
                </Form.Item>

                <Form.Item
                  name="notes"
                  label="处理备注"
                >
                  <TextArea 
                    placeholder="记录处理过程和结果..."
                    rows={3}
                  />
                </Form.Item>

                <Form.Item>
                  <Space>
                    <Button type="primary" htmlType="submit" loading={loading}>
                      执行响应
                    </Button>
                    <Button onClick={() => setResponseModalVisible(false)}>
                      稍后处理
                    </Button>
                  </Space>
                </Form.Item>
              </Form>
            </Card>
          </div>
        )}
      </Modal>

      <style jsx>{`
        .critical-row {
          background-color: #fff2f0 !important;
        }
      `}</style>
    </div>
  );
};

export default CrisisDetectionSupportPage;