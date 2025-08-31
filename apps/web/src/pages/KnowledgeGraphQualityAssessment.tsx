import React, { useState, useEffect } from 'react'
import {
  Card,
  Table,
  Button,
  Space,
  Alert,
  Tooltip,
  Row,
  Col,
  Statistic,
  Progress,
  Tag,
  Typography,
  Divider,
  Select,
  DatePicker,
  Charts,
  message,
  Modal,
  Tabs,
  List,
  Avatar,
  Badge,
  Collapse,
  Timeline
} from 'antd'
import {
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  CloseCircleOutlined,
  BarChartOutlined,
  PieChartOutlined,
  LineChartOutlined,
  FileSearchOutlined,
  SettingOutlined,
  ReloadOutlined,
  DownloadOutlined,
  BugOutlined,
  ThunderboltOutlined,
  DatabaseOutlined,
  LinkOutlined,
  NodeIndexOutlined,
  WarningOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { RangePicker } = DatePicker
const { TabPane } = Tabs
const { Panel } = Collapse

interface QualityMetrics {
  overall_score: number
  completeness: number
  accuracy: number
  consistency: number
  timeliness: number
  uniqueness: number
  validity: number
}

interface QualityIssue {
  id: string
  type: 'missing_property' | 'duplicate_entity' | 'invalid_format' | 'inconsistent_data' | 'outdated_info' | 'broken_reference'
  severity: 'critical' | 'high' | 'medium' | 'low'
  entity_id: string
  entity_name: string
  description: string
  suggestion: string
  detected_at: string
  status: 'open' | 'resolved' | 'ignored'
  auto_fixable: boolean
}

interface QualityRule {
  id: string
  name: string
  category: string
  description: string
  enabled: boolean
  severity: 'critical' | 'high' | 'medium' | 'low'
  auto_fix: boolean
  violation_count: number
}

interface QualityTrend {
  date: string
  overall_score: number
  completeness: number
  accuracy: number
  consistency: number
  issues_detected: number
  issues_resolved: number
}

const KnowledgeGraphQualityAssessment: React.FC = () => {
  const [metrics, setMetrics] = useState<QualityMetrics>({} as QualityMetrics)
  const [issues, setIssues] = useState<QualityIssue[]>([])
  const [rules, setRules] = useState<QualityRule[]>([])
  const [trends, setTrends] = useState<QualityTrend[]>([])
  const [loading, setLoading] = useState(false)
  const [selectedTimeRange, setSelectedTimeRange] = useState('7d')
  const [activeTab, setActiveTab] = useState('overview')
  const [rulesModalVisible, setRulesModalVisible] = useState(false)

  // 模拟质量指标数据
  const mockMetrics: QualityMetrics = {
    overall_score: 87.5,
    completeness: 92.3,
    accuracy: 89.7,
    consistency: 85.2,
    timeliness: 88.9,
    uniqueness: 94.1,
    validity: 83.6
  }

  // 模拟质量问题数据
  const mockIssues: QualityIssue[] = [
    {
      id: 'issue_001',
      type: 'missing_property',
      severity: 'high',
      entity_id: 'entity_123',
      entity_name: '张三',
      description: '缺少必需属性：email',
      suggestion: '补充邮箱地址信息',
      detected_at: '2025-01-22T14:30:00Z',
      status: 'open',
      auto_fixable: false
    },
    {
      id: 'issue_002',
      type: 'duplicate_entity',
      severity: 'critical',
      entity_id: 'entity_456',
      entity_name: '苹果公司',
      description: '发现重复实体：Apple Inc. 和 苹果公司',
      suggestion: '合并重复实体并统一引用',
      detected_at: '2025-01-22T13:45:00Z',
      status: 'open',
      auto_fixable: true
    },
    {
      id: 'issue_003',
      type: 'invalid_format',
      severity: 'medium',
      entity_id: 'entity_789',
      entity_name: '联系方式',
      description: '电话号码格式不正确：123-abc-456',
      suggestion: '使用标准电话号码格式',
      detected_at: '2025-01-22T12:20:00Z',
      status: 'resolved',
      auto_fixable: true
    },
    {
      id: 'issue_004',
      type: 'inconsistent_data',
      severity: 'high',
      entity_id: 'entity_012',
      entity_name: '员工信息',
      description: '年龄与出生日期不一致',
      suggestion: '核实并更正年龄或出生日期',
      detected_at: '2025-01-22T11:10:00Z',
      status: 'open',
      auto_fixable: false
    },
    {
      id: 'issue_005',
      type: 'broken_reference',
      severity: 'critical',
      entity_id: 'entity_345',
      entity_name: '部门关系',
      description: '引用的部门实体不存在',
      suggestion: '修复破损的引用关系',
      detected_at: '2025-01-22T10:30:00Z',
      status: 'open',
      auto_fixable: false
    }
  ]

  // 模拟质量规则数据
  const mockRules: QualityRule[] = [
    {
      id: 'rule_001',
      name: '必需属性完整性',
      category: '完整性',
      description: '检查核心实体是否包含所有必需属性',
      enabled: true,
      severity: 'high',
      auto_fix: false,
      violation_count: 23
    },
    {
      id: 'rule_002',
      name: '实体唯一性',
      category: '唯一性',
      description: '检测和标记重复的实体',
      enabled: true,
      severity: 'critical',
      auto_fix: true,
      violation_count: 8
    },
    {
      id: 'rule_003',
      name: '数据格式验证',
      category: '有效性',
      description: '验证数据字段格式是否符合预定义规范',
      enabled: true,
      severity: 'medium',
      auto_fix: true,
      violation_count: 156
    },
    {
      id: 'rule_004',
      name: '引用完整性',
      category: '一致性',
      description: '检查所有引用关系的目标实体是否存在',
      enabled: true,
      severity: 'critical',
      auto_fix: false,
      violation_count: 3
    },
    {
      id: 'rule_005',
      name: '时效性检查',
      category: '时效性',
      description: '识别过期或陈旧的数据',
      enabled: false,
      severity: 'low',
      auto_fix: false,
      violation_count: 45
    }
  ]

  // 模拟趋势数据
  const mockTrends: QualityTrend[] = [
    { date: '2025-01-15', overall_score: 85.2, completeness: 90.1, accuracy: 87.3, consistency: 82.5, issues_detected: 45, issues_resolved: 38 },
    { date: '2025-01-16', overall_score: 86.1, completeness: 91.2, accuracy: 88.1, consistency: 83.8, issues_detected: 42, issues_resolved: 41 },
    { date: '2025-01-17', overall_score: 85.8, completeness: 90.8, accuracy: 87.9, consistency: 83.2, issues_detected: 48, issues_resolved: 35 },
    { date: '2025-01-18', overall_score: 87.3, completeness: 92.5, accuracy: 89.2, consistency: 84.6, issues_detected: 38, issues_resolved: 42 },
    { date: '2025-01-19', overall_score: 86.9, completeness: 91.8, accuracy: 88.7, consistency: 84.1, issues_detected: 41, issues_resolved: 39 },
    { date: '2025-01-20', overall_score: 87.8, completeness: 92.7, accuracy: 89.5, consistency: 85.1, issues_detected: 35, issues_resolved: 44 },
    { date: '2025-01-21', overall_score: 87.5, completeness: 92.3, accuracy: 89.7, consistency: 85.2, issues_detected: 39, issues_resolved: 41 }
  ]

  useEffect(() => {
    loadData()
  }, [selectedTimeRange])

  const loadData = async () => {
    setLoading(true)
    try {
      // 模拟API调用
      await new Promise(resolve => setTimeout(resolve, 1000))
      setMetrics(mockMetrics)
      setIssues(mockIssues)
      setRules(mockRules)
      setTrends(mockTrends)
    } catch (error) {
      message.error('加载数据失败')
    } finally {
      setLoading(false)
    }
  }

  const runQualityAssessment = async () => {
    setLoading(true)
    try {
      await new Promise(resolve => setTimeout(resolve, 2000))
      message.success('质量评估已完成')
      loadData()
    } catch (error) {
      message.error('质量评估失败')
    } finally {
      setLoading(false)
    }
  }

  const fixIssue = async (issueId: string) => {
    try {
      const updatedIssues = issues.map(issue => 
        issue.id === issueId ? { ...issue, status: 'resolved' as const } : issue
      )
      setIssues(updatedIssues)
      message.success('问题已修复')
    } catch (error) {
      message.error('修复失败')
    }
  }

  const ignoreIssue = async (issueId: string) => {
    try {
      const updatedIssues = issues.map(issue => 
        issue.id === issueId ? { ...issue, status: 'ignored' as const } : issue
      )
      setIssues(updatedIssues)
      message.success('问题已忽略')
    } catch (error) {
      message.error('操作失败')
    }
  }

  const toggleRule = async (ruleId: string) => {
    try {
      const updatedRules = rules.map(rule => 
        rule.id === ruleId ? { ...rule, enabled: !rule.enabled } : rule
      )
      setRules(updatedRules)
      message.success('规则状态已更新')
    } catch (error) {
      message.error('更新失败')
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 90) return '#52c41a'
    if (score >= 80) return '#faad14'
    if (score >= 70) return '#fa8c16'
    return '#ff4d4f'
  }

  const getSeverityColor = (severity: string) => {
    const colors = {
      'critical': 'red',
      'high': 'orange',
      'medium': 'gold',
      'low': 'blue'
    }
    return colors[severity as keyof typeof colors] || 'default'
  }

  const getSeverityName = (severity: string) => {
    const names = {
      'critical': '严重',
      'high': '高',
      'medium': '中',
      'low': '低'
    }
    return names[severity as keyof typeof names] || severity
  }

  const getIssueTypeName = (type: string) => {
    const names = {
      'missing_property': '缺失属性',
      'duplicate_entity': '重复实体',
      'invalid_format': '格式错误',
      'inconsistent_data': '数据不一致',
      'outdated_info': '过期信息',
      'broken_reference': '破损引用'
    }
    return names[type as keyof typeof names] || type
  }

  const getIssueTypeIcon = (type: string) => {
    const icons = {
      'missing_property': <ExclamationCircleOutlined />,
      'duplicate_entity': <WarningOutlined />,
      'invalid_format': <BugOutlined />,
      'inconsistent_data': <CloseCircleOutlined />,
      'outdated_info': <CloseCircleOutlined />,
      'broken_reference': <LinkOutlined />
    }
    return icons[type as keyof typeof icons] || <ExclamationCircleOutlined />
  }

  const issueColumns = [
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Space>
          {getIssueTypeIcon(type)}
          <Text>{getIssueTypeName(type)}</Text>
        </Space>
      ),
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: string) => (
        <Tag color={getSeverityColor(severity)}>
          {getSeverityName(severity)}
        </Tag>
      ),
    },
    {
      title: '实体',
      dataIndex: 'entity_name',
      key: 'entity_name',
    },
    {
      title: '问题描述',
      dataIndex: 'description',
      key: 'description',
      render: (text: string) => (
        <div style={{ maxWidth: '250px' }}>
          <Text ellipsis={{ tooltip: text }}>{text}</Text>
        </div>
      ),
    },
    {
      title: '建议',
      dataIndex: 'suggestion',
      key: 'suggestion',
      render: (text: string) => (
        <div style={{ maxWidth: '200px' }}>
          <Text ellipsis={{ tooltip: text }} type="secondary">{text}</Text>
        </div>
      ),
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const colors = { 'open': 'red', 'resolved': 'green', 'ignored': 'gray' }
        const names = { 'open': '待处理', 'resolved': '已解决', 'ignored': '已忽略' }
        return <Tag color={colors[status as keyof typeof colors]}>{names[status as keyof typeof names]}</Tag>
      },
    },
    {
      title: '自动修复',
      dataIndex: 'auto_fixable',
      key: 'auto_fixable',
      render: (fixable: boolean) => (
        fixable ? <CheckCircleOutlined style={{ color: '#52c41a' }} /> : <CloseCircleOutlined style={{ color: '#ff4d4f' }} />
      ),
    },
    {
      title: '操作',
      key: 'actions',
      render: (_, record: QualityIssue) => (
        <Space>
          {record.status === 'open' && (
            <>
              {record.auto_fixable && (
                <Button 
                  size="small" 
                  type="primary"
                  onClick={() => fixIssue(record.id)}
                >
                  自动修复
                </Button>
              )}
              <Button 
                size="small"
                onClick={() => ignoreIssue(record.id)}
              >
                忽略
              </Button>
            </>
          )}
        </Space>
      ),
    },
  ]

  const openIssues = issues.filter(issue => issue.status === 'open')
  const criticalIssues = issues.filter(issue => issue.severity === 'critical')
  const autoFixableIssues = issues.filter(issue => issue.auto_fixable && issue.status === 'open')

  return (
    <div style={{ padding: '24px' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <FileSearchOutlined style={{ marginRight: '8px' }} />
          质量评估仪表板
        </Title>
        <Paragraph type="secondary">
          监控和评估知识图谱的数据质量，识别和修复质量问题
        </Paragraph>
      </div>

      {/* 控制面板 */}
      <Card style={{ marginBottom: '16px' }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Text strong>时间范围:</Text>
              <Select 
                value={selectedTimeRange} 
                onChange={setSelectedTimeRange}
                style={{ width: 120 }}
              >
                <Select.Option value="1d">1天</Select.Option>
                <Select.Option value="7d">7天</Select.Option>
                <Select.Option value="30d">30天</Select.Option>
                <Select.Option value="90d">90天</Select.Option>
              </Select>
            </Space>
          </Col>
          <Col>
            <Space>
              <Button icon={<SettingOutlined />} onClick={() => setRulesModalVisible(true)}>
                质量规则
              </Button>
              <Button icon={<DownloadOutlined />}>导出报告</Button>
              <Button icon={<ReloadOutlined />} onClick={loadData}>刷新</Button>
              <Button 
                type="primary" 
                icon={<ThunderboltOutlined />}
                onClick={runQualityAssessment}
                loading={loading}
              >
                运行评估
              </Button>
            </Space>
          </Col>
        </Row>
      </Card>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab="质量概览" key="overview">
          {/* 质量指标概览 */}
          <Row gutter={16} style={{ marginBottom: '24px' }}>
            <Col span={6}>
              <Card>
                <Statistic
                  title="总体质量评分"
                  value={metrics.overall_score}
                  precision={1}
                  suffix="/100"
                  valueStyle={{ color: getScoreColor(metrics.overall_score) }}
                  prefix={<BarChartOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="待处理问题"
                  value={openIssues.length}
                  valueStyle={{ color: openIssues.length > 0 ? '#ff4d4f' : '#52c41a' }}
                  prefix={<ExclamationCircleOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="严重问题"
                  value={criticalIssues.length}
                  valueStyle={{ color: criticalIssues.length > 0 ? '#ff4d4f' : '#52c41a' }}
                  prefix={<WarningOutlined />}
                />
              </Card>
            </Col>
            <Col span={6}>
              <Card>
                <Statistic
                  title="可自动修复"
                  value={autoFixableIssues.length}
                  valueStyle={{ color: '#1890ff' }}
                  prefix={<ThunderboltOutlined />}
                />
              </Card>
            </Col>
          </Row>

          {/* 质量维度详情 */}
          <Row gutter={16} style={{ marginBottom: '24px' }}>
            <Col span={12}>
              <Card title="质量维度评分">
                <div style={{ marginBottom: '16px' }}>
                  <div style={{ marginBottom: '8px' }}>
                    <Text>完整性 (Completeness)</Text>
                    <Text style={{ float: 'right' }}>{metrics.completeness}%</Text>
                  </div>
                  <Progress percent={metrics.completeness} strokeColor={getScoreColor(metrics.completeness)} />
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <div style={{ marginBottom: '8px' }}>
                    <Text>准确性 (Accuracy)</Text>
                    <Text style={{ float: 'right' }}>{metrics.accuracy}%</Text>
                  </div>
                  <Progress percent={metrics.accuracy} strokeColor={getScoreColor(metrics.accuracy)} />
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <div style={{ marginBottom: '8px' }}>
                    <Text>一致性 (Consistency)</Text>
                    <Text style={{ float: 'right' }}>{metrics.consistency}%</Text>
                  </div>
                  <Progress percent={metrics.consistency} strokeColor={getScoreColor(metrics.consistency)} />
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <div style={{ marginBottom: '8px' }}>
                    <Text>时效性 (Timeliness)</Text>
                    <Text style={{ float: 'right' }}>{metrics.timeliness}%</Text>
                  </div>
                  <Progress percent={metrics.timeliness} strokeColor={getScoreColor(metrics.timeliness)} />
                </div>
                <div style={{ marginBottom: '16px' }}>
                  <div style={{ marginBottom: '8px' }}>
                    <Text>唯一性 (Uniqueness)</Text>
                    <Text style={{ float: 'right' }}>{metrics.uniqueness}%</Text>
                  </div>
                  <Progress percent={metrics.uniqueness} strokeColor={getScoreColor(metrics.uniqueness)} />
                </div>
                <div>
                  <div style={{ marginBottom: '8px' }}>
                    <Text>有效性 (Validity)</Text>
                    <Text style={{ float: 'right' }}>{metrics.validity}%</Text>
                  </div>
                  <Progress percent={metrics.validity} strokeColor={getScoreColor(metrics.validity)} />
                </div>
              </Card>
            </Col>
            <Col span={12}>
              <Card title="问题分布">
                <div style={{ 
                  height: '300px', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  border: '1px dashed #d9d9d9',
                  backgroundColor: '#fafafa'
                }}>
                  <div style={{ textAlign: 'center' }}>
                    <PieChartOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
                    <div style={{ marginTop: '16px' }}>
                      <Text type="secondary">问题类型分布图表</Text>
                    </div>
                  </div>
                </div>
              </Card>
            </Col>
          </Row>

          {/* 近期质量问题 */}
          <Card title="近期质量问题" extra={
            <Space>
              <Badge count={openIssues.length} showZero={false}>
                <Text>待处理</Text>
              </Badge>
              <Badge count={criticalIssues.length} showZero={false} color="red">
                <Text>严重</Text>
              </Badge>
            </Space>
          }>
            <Table
              columns={issueColumns}
              dataSource={issues.slice(0, 5)}
              rowKey="id"
              pagination={false}
              size="small"
            />
            {issues.length > 5 && (
              <div style={{ textAlign: 'center', marginTop: '16px' }}>
                <Button type="link" onClick={() => setActiveTab('issues')}>
                  查看全部 {issues.length} 个问题
                </Button>
              </div>
            )}
          </Card>
        </TabPane>

        <TabPane tab="问题管理" key="issues">
          <Card title="质量问题列表">
            <Table
              columns={issueColumns}
              dataSource={issues}
              rowKey="id"
              loading={loading}
              pagination={{
                pageSize: 10,
                showTotal: (total) => `共 ${total} 个问题`
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab="质量趋势" key="trends">
          <Card title="质量趋势分析">
            <div style={{ 
              height: '400px', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              border: '1px dashed #d9d9d9',
              backgroundColor: '#fafafa'
            }}>
              <div style={{ textAlign: 'center' }}>
                <LineChartOutlined style={{ fontSize: '48px', color: '#d9d9d9' }} />
                <div style={{ marginTop: '16px' }}>
                  <Text type="secondary">质量趋势图表将在此显示</Text>
                </div>
              </div>
            </div>
          </Card>
        </TabPane>

        <TabPane tab="质量规则" key="rules">
          <Card title="质量检查规则">
            <List
              dataSource={rules}
              renderItem={(rule) => (
                <List.Item
                  actions={[
                    <Button 
                      type="link" 
                      onClick={() => toggleRule(rule.id)}
                    >
                      {rule.enabled ? '禁用' : '启用'}
                    </Button>
                  ]}
                >
                  <List.Item.Meta
                    avatar={
                      <Avatar 
                        icon={rule.enabled ? <CheckCircleOutlined /> : <CloseCircleOutlined />}
                        style={{ backgroundColor: rule.enabled ? '#52c41a' : '#d9d9d9' }}
                      />
                    }
                    title={
                      <Space>
                        <Text strong>{rule.name}</Text>
                        <Tag color={getSeverityColor(rule.severity)}>
                          {getSeverityName(rule.severity)}
                        </Tag>
                        {rule.auto_fix && <Tag color="blue">自动修复</Tag>}
                      </Space>
                    }
                    description={
                      <div>
                        <Paragraph style={{ margin: 0 }}>{rule.description}</Paragraph>
                        <Text type="secondary">类别: {rule.category} | 违规数: {rule.violation_count}</Text>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>
      </Tabs>

      {/* 质量规则配置模态框 */}
      <Modal
        title="质量规则配置"
        open={rulesModalVisible}
        onCancel={() => setRulesModalVisible(false)}
        width={800}
        footer={[
          <Button key="cancel" onClick={() => setRulesModalVisible(false)}>
            取消
          </Button>,
          <Button key="save" type="primary" onClick={() => {
            setRulesModalVisible(false)
            message.success('规则配置已保存')
          }}>
            保存
          </Button>
        ]}
      >
        <Collapse defaultActiveKey={['completeness']}>
          <Panel header="完整性规则" key="completeness">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>检查必需属性的完整性和覆盖率</Text>
              {/* 规则配置选项 */}
            </Space>
          </Panel>
          <Panel header="一致性规则" key="consistency">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>验证数据的内部一致性和逻辑正确性</Text>
              {/* 规则配置选项 */}
            </Space>
          </Panel>
          <Panel header="唯一性规则" key="uniqueness">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Text>检测重复实体和冗余数据</Text>
              {/* 规则配置选项 */}
            </Space>
          </Panel>
        </Collapse>
      </Modal>
    </div>
  )
}

export default KnowledgeGraphQualityAssessment