import React, { useState, useEffect, useRef } from 'react'
import {
  Card,
  Row,
  Col,
  Tabs,
  Button,
  Table,
  Tag,
  Alert,
  Modal,
  Form,
  Input,
  Select,
  Switch,
  Progress,
  Divider,
  Space,
  List,
  Timeline,
  Badge,
  Tooltip,
  Rate,
  Statistic,
  Steps,
  Tree,
  Collapse,
  message,
  Spin,
  Descriptions,
  Radio,
  Slider,
  CheckCard
} from 'antd'
import {
  ShieldOutlined,
  SafetyCertificateOutlined,
  UserOutlined,
  EyeOutlined,
  LockOutlined,
  AuditOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  FileProtectOutlined,
  GlobalOutlined,
  SettingOutlined,
  BellOutlined,
  HeartOutlined,
  TeamOutlined,
  BookOutlined,
  SecurityScanOutlined,
  ThunderboltOutlined,
  DashboardOutlined,
  HistoryOutlined,
  KeyOutlined
} from '@ant-design/icons'
import * as d3 from 'd3'

const { TabPane } = Tabs
const { Option } = Select
const { TextArea } = Input
const { Panel } = Collapse
const { Step } = Steps

interface PrivacyPolicy {
  id: string
  name: string
  category: string
  scope: string[]
  sensitivity_level: number
  retention_period: number
  access_control: string[]
  encryption_required: boolean
  audit_required: boolean
  consent_required: boolean
  third_party_sharing: boolean
  geographical_restrictions: string[]
  compliance_standards: string[]
  created_date: string
  last_updated: string
  status: 'active' | 'draft' | 'archived'
}

interface EthicalGuideline {
  id: string
  principle: string
  category: string
  description: string
  impact_level: number
  applicability: string[]
  implementation_steps: string[]
  monitoring_metrics: string[]
  violation_consequences: string[]
  cultural_considerations: string[]
  review_frequency: number
  responsible_parties: string[]
  status: 'active' | 'under_review' | 'deprecated'
}

interface ComplianceCheck {
  id: string
  check_type: string
  regulation: string
  requirement: string
  current_status: 'compliant' | 'non_compliant' | 'partial' | 'unknown'
  risk_level: 'low' | 'medium' | 'high' | 'critical'
  remediation_actions: string[]
  deadline: string
  responsible_team: string
  evidence_documents: string[]
  last_assessment: string
}

interface ConsentRecord {
  id: string
  user_id: string
  consent_type: string
  purpose: string
  granted_permissions: string[]
  withdrawn_permissions: string[]
  consent_timestamp: string
  expiry_date: string
  granularity_level: string
  withdrawal_method: string[]
  cultural_context: string
  language: string
  minor_consent: boolean
  guardian_info?: any
}

interface AuditEvent {
  id: string
  event_type: string
  timestamp: string
  user_id: string
  resource_accessed: string
  action_performed: string
  risk_score: number
  geographic_location: string
  device_info: any
  outcome: 'success' | 'failure' | 'blocked'
  privacy_implications: string[]
  ethical_concerns: string[]
}

const PrivacyEthicsPage: React.FC = () => {
  const [loading, setLoading] = useState(false)
  const [activeTab, setActiveTab] = useState('overview')
  const [privacyPolicies, setPrivacyPolicies] = useState<PrivacyPolicy[]>([])
  const [ethicalGuidelines, setEthicalGuidelines] = useState<EthicalGuideline[]>([])
  const [complianceChecks, setComplianceChecks] = useState<ComplianceCheck[]>([])
  const [consentRecords, setConsentRecords] = useState<ConsentRecord[]>([])
  const [auditEvents, setAuditEvents] = useState<AuditEvent[]>([])
  const [modalVisible, setModalVisible] = useState(false)
  const [modalType, setModalType] = useState<'policy' | 'guideline' | 'consent' | 'audit'>('policy')
  const [selectedItem, setSelectedItem] = useState<any>(null)
  const [assessmentMode, setAssessmentMode] = useState(false)
  const [realTimeMonitoring, setRealTimeMonitoring] = useState(true)
  
  const [form] = Form.useForm()
  const chartRef = useRef<HTMLDivElement>(null)
  const complianceChartRef = useRef<HTMLDivElement>(null)
  const riskChartRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    initializeData()
  }, [])

  useEffect(() => {
    if (activeTab === 'dashboard') {
      renderDashboardCharts()
    }
  }, [activeTab, privacyPolicies, complianceChecks])

  const initializeData = async () => {
    setLoading(true)
    try {
      // 模拟隐私政策数据
      const mockPolicies: PrivacyPolicy[] = [
        {
          id: 'policy_001',
          name: '用户情感数据保护政策',
          category: 'emotional_data',
          scope: ['emotion_recognition', 'empathy_modeling', 'social_analysis'],
          sensitivity_level: 9,
          retention_period: 365,
          access_control: ['authorized_researchers', 'system_administrators'],
          encryption_required: true,
          audit_required: true,
          consent_required: true,
          third_party_sharing: false,
          geographical_restrictions: ['EU', 'California'],
          compliance_standards: ['GDPR', 'CCPA', 'PIPEDA'],
          created_date: '2024-01-15',
          last_updated: '2024-11-20',
          status: 'active'
        },
        {
          id: 'policy_002',
          name: '社交互动数据处理政策',
          category: 'social_interaction',
          scope: ['conversation_analysis', 'relationship_mapping', 'behavioral_patterns'],
          sensitivity_level: 7,
          retention_period: 180,
          access_control: ['data_scientists', 'product_managers'],
          encryption_required: true,
          audit_required: false,
          consent_required: true,
          third_party_sharing: true,
          geographical_restrictions: [],
          compliance_standards: ['ISO27001', 'SOC2'],
          created_date: '2024-02-10',
          last_updated: '2024-10-15',
          status: 'active'
        }
      ]

      // 模拟伦理指导原则
      const mockGuidelines: EthicalGuideline[] = [
        {
          id: 'guideline_001',
          principle: '情感数据最小化原则',
          category: 'data_minimization',
          description: '仅收集和处理实现特定目的所必需的最少情感数据',
          impact_level: 9,
          applicability: ['emotion_recognition', 'empathy_response'],
          implementation_steps: [
            '定义明确的数据收集目的',
            '建立数据最小化审查流程',
            '定期评估数据使用的必要性',
            '自动化删除不必要的数据'
          ],
          monitoring_metrics: ['数据收集量', '数据保留时间', '目的达成率'],
          violation_consequences: ['数据使用权限暂停', '强制培训', '合规审查'],
          cultural_considerations: ['不同文化对隐私的理解差异', '情感表达的文化特性'],
          review_frequency: 90,
          responsible_parties: ['隐私官', '数据科学团队', '产品负责人'],
          status: 'active'
        },
        {
          id: 'guideline_002',
          principle: '算法公平性原则',
          category: 'algorithmic_fairness',
          description: '确保情感AI系统对不同群体和文化背景的公平对待',
          impact_level: 8,
          applicability: ['emotion_classification', 'bias_detection'],
          implementation_steps: [
            '建立多元化数据集',
            '实施偏见检测算法',
            '定期进行公平性测试',
            '建立反馈修正机制'
          ],
          monitoring_metrics: ['群体间准确率差异', '文化偏见指标', '用户满意度'],
          violation_consequences: ['算法调优要求', '重新训练模型', '暂停服务'],
          cultural_considerations: ['避免文化刻板印象', '尊重文化多样性'],
          review_frequency: 60,
          responsible_parties: ['AI伦理委员会', '算法团队'],
          status: 'active'
        }
      ]

      // 模拟合规检查
      const mockCompliance: ComplianceCheck[] = [
        {
          id: 'compliance_001',
          check_type: 'data_protection',
          regulation: 'GDPR',
          requirement: '数据主体权利实施',
          current_status: 'compliant',
          risk_level: 'low',
          remediation_actions: [],
          deadline: '2024-12-31',
          responsible_team: '隐私工程团队',
          evidence_documents: ['GDPR_compliance_report.pdf', 'user_rights_implementation.md'],
          last_assessment: '2024-11-15'
        },
        {
          id: 'compliance_002',
          check_type: 'consent_management',
          regulation: 'CCPA',
          requirement: '消费者隐私权告知',
          current_status: 'partial',
          risk_level: 'medium',
          remediation_actions: ['更新隐私政策', '实施用户控制面板'],
          deadline: '2024-12-01',
          responsible_team: '法务合规团队',
          evidence_documents: ['ccpa_gap_analysis.pdf'],
          last_assessment: '2024-10-20'
        }
      ]

      // 模拟同意记录
      const mockConsents: ConsentRecord[] = [
        {
          id: 'consent_001',
          user_id: 'user_12345',
          consent_type: 'emotion_data_processing',
          purpose: '情感分析和个性化体验',
          granted_permissions: ['emotion_recognition', 'mood_tracking', 'empathy_response'],
          withdrawn_permissions: [],
          consent_timestamp: '2024-11-01T10:30:00Z',
          expiry_date: '2025-11-01T10:30:00Z',
          granularity_level: 'detailed',
          withdrawal_method: ['app_settings', 'email_request'],
          cultural_context: 'western_individualistic',
          language: 'en-US',
          minor_consent: false
        }
      ]

      // 模拟审计事件
      const mockAudits: AuditEvent[] = [
        {
          id: 'audit_001',
          event_type: 'data_access',
          timestamp: '2024-11-29T14:22:00Z',
          user_id: 'researcher_001',
          resource_accessed: 'emotion_dataset_v2',
          action_performed: 'query_execution',
          risk_score: 3,
          geographic_location: 'US-CA',
          device_info: { type: 'workstation', ip: '192.168.1.100' },
          outcome: 'success',
          privacy_implications: ['personal_emotion_data_accessed'],
          ethical_concerns: []
        }
      ]

      setPrivacyPolicies(mockPolicies)
      setEthicalGuidelines(mockGuidelines)
      setComplianceChecks(mockCompliance)
      setConsentRecords(mockConsents)
      setAuditEvents(mockAudits)
    } catch (error) {
      console.error('初始化数据失败:', error)
      message.error('数据加载失败')
    } finally {
      setLoading(false)
    }
  }

  const renderDashboardCharts = () => {
    renderComplianceChart()
    renderRiskChart()
    renderPrivacyMetricsChart()
  }

  const renderComplianceChart = () => {
    if (!complianceChartRef.current) return

    const container = d3.select(complianceChartRef.current)
    container.selectAll('*').remove()

    const data = complianceChecks.reduce((acc, check) => {
      acc[check.current_status] = (acc[check.current_status] || 0) + 1
      return acc
    }, {} as Record<string, number>)

    const entries = Object.entries(data)
    const width = 300
    const height = 200
    const radius = Math.min(width, height) / 2

    const svg = container
      .append('svg')
      .attr('width', width)
      .attr('height', height)
      .append('g')
      .attr('transform', `translate(${width / 2}, ${height / 2})`)

    const color = d3.scaleOrdinal()
      .domain(entries.map(d => d[0]))
      .range(['#52c41a', '#faad14', '#f5222d', '#d9d9d9'])

    const pie = d3.pie<[string, number]>()
      .value(d => d[1])

    const arc = d3.arc<any>()
      .innerRadius(0)
      .outerRadius(radius)

    svg.selectAll('path')
      .data(pie(entries))
      .enter()
      .append('path')
      .attr('d', arc)
      .attr('fill', (d: any) => color(d.data[0]) as string)
      .attr('opacity', 0.8)

    svg.selectAll('text')
      .data(pie(entries))
      .enter()
      .append('text')
      .attr('transform', (d: any) => `translate(${arc.centroid(d)})`)
      .attr('text-anchor', 'middle')
      .text((d: any) => d.data[1])
      .attr('font-size', '12px')
      .attr('fill', 'white')
  }

  const renderRiskChart = () => {
    if (!riskChartRef.current) return

    const container = d3.select(riskChartRef.current)
    container.selectAll('*').remove()

    const riskData = complianceChecks.map(check => ({
      regulation: check.regulation,
      risk: check.risk_level === 'low' ? 1 : check.risk_level === 'medium' ? 2 : check.risk_level === 'high' ? 3 : 4
    }))

    const margin = { top: 20, right: 30, bottom: 40, left: 40 }
    const width = 300 - margin.left - margin.right
    const height = 200 - margin.top - margin.bottom

    const svg = container
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const xScale = d3.scaleBand()
      .domain(riskData.map(d => d.regulation))
      .range([0, width])
      .padding(0.1)

    const yScale = d3.scaleLinear()
      .domain([0, 4])
      .range([height, 0])

    svg.selectAll('.bar')
      .data(riskData)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.regulation)!)
      .attr('y', d => yScale(d.risk))
      .attr('width', xScale.bandwidth())
      .attr('height', d => height - yScale(d.risk))
      .attr('fill', d => {
        if (d.risk === 1) return '#52c41a'
        if (d.risk === 2) return '#faad14'
        if (d.risk === 3) return '#fa8c16'
        return '#f5222d'
      })
      .attr('opacity', 0.8)

    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale))

    svg.append('g')
      .call(d3.axisLeft(yScale))
  }

  const renderPrivacyMetricsChart = () => {
    if (!chartRef.current) return

    const container = d3.select(chartRef.current)
    container.selectAll('*').remove()

    const metricsData = [
      { metric: '数据加密率', value: 98 },
      { metric: '同意获取率', value: 85 },
      { metric: '审计覆盖率', value: 92 },
      { metric: '隐私政策遵循率', value: 89 }
    ]

    const margin = { top: 20, right: 30, bottom: 40, left: 100 }
    const width = 400 - margin.left - margin.right
    const height = 250 - margin.top - margin.bottom

    const svg = container
      .append('svg')
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    const xScale = d3.scaleLinear()
      .domain([0, 100])
      .range([0, width])

    const yScale = d3.scaleBand()
      .domain(metricsData.map(d => d.metric))
      .range([0, height])
      .padding(0.1)

    svg.selectAll('.bar')
      .data(metricsData)
      .enter()
      .append('rect')
      .attr('class', 'bar')
      .attr('x', 0)
      .attr('y', d => yScale(d.metric)!)
      .attr('width', d => xScale(d.value))
      .attr('height', yScale.bandwidth())
      .attr('fill', '#1890ff')
      .attr('opacity', 0.8)

    svg.selectAll('.label')
      .data(metricsData)
      .enter()
      .append('text')
      .attr('class', 'label')
      .attr('x', d => xScale(d.value) + 5)
      .attr('y', d => yScale(d.metric)! + yScale.bandwidth() / 2)
      .attr('dy', '0.35em')
      .text(d => `${d.value}%`)
      .attr('font-size', '12px')
      .attr('fill', '#666')

    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(xScale))

    svg.append('g')
      .call(d3.axisLeft(yScale))
  }

  const handleModalOpen = (type: 'policy' | 'guideline' | 'consent' | 'audit', item?: any) => {
    setModalType(type)
    setSelectedItem(item)
    setModalVisible(true)
  }

  const handleModalClose = () => {
    setModalVisible(false)
    setSelectedItem(null)
    form.resetFields()
  }

  const handleFormSubmit = async (values: any) => {
    try {
      // 模拟API调用
      console.log('提交表单数据:', values)
      message.success(`${modalType === 'policy' ? '隐私政策' : '伦理指导原则'}${selectedItem ? '更新' : '创建'}成功`)
      handleModalClose()
    } catch (error) {
      message.error('操作失败')
    }
  }

  const privacyPolicyColumns = [
    {
      title: '政策名称',
      dataIndex: 'name',
      key: 'name',
      render: (text: string, record: PrivacyPolicy) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{text}</div>
          <Tag color="blue">{record.category}</Tag>
        </div>
      )
    },
    {
      title: '敏感级别',
      dataIndex: 'sensitivity_level',
      key: 'sensitivity_level',
      render: (level: number) => (
        <div>
          <Progress 
            percent={(level / 10) * 100} 
            size="small" 
            strokeColor={level > 7 ? '#f5222d' : level > 5 ? '#faad14' : '#52c41a'}
            format={() => `${level}/10`}
          />
        </div>
      )
    },
    {
      title: '保留期限',
      dataIndex: 'retention_period',
      key: 'retention_period',
      render: (days: number) => `${days}天`
    },
    {
      title: '合规标准',
      dataIndex: 'compliance_standards',
      key: 'compliance_standards',
      render: (standards: string[]) => (
        <>
          {standards.map(standard => (
            <Tag key={standard} color="green">{standard}</Tag>
          ))}
        </>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const color = status === 'active' ? 'green' : status === 'draft' ? 'orange' : 'gray'
        return <Tag color={color}>{status.toUpperCase()}</Tag>
      }
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: PrivacyPolicy) => (
        <Space>
          <Button 
            type="link" 
            size="small" 
            onClick={() => handleModalOpen('policy', record)}
          >
            查看详情
          </Button>
          <Button type="link" size="small">
            编辑
          </Button>
        </Space>
      )
    }
  ]

  const ethicalGuidelineColumns = [
    {
      title: '伦理原则',
      dataIndex: 'principle',
      key: 'principle',
      render: (text: string, record: EthicalGuideline) => (
        <div>
          <div style={{ fontWeight: 'bold' }}>{text}</div>
          <Tag color="purple">{record.category}</Tag>
        </div>
      )
    },
    {
      title: '影响等级',
      dataIndex: 'impact_level',
      key: 'impact_level',
      render: (level: number) => (
        <Rate disabled value={level / 2} style={{ fontSize: '12px' }} />
      )
    },
    {
      title: '审查频率',
      dataIndex: 'review_frequency',
      key: 'review_frequency',
      render: (days: number) => `每${days}天`
    },
    {
      title: '负责方',
      dataIndex: 'responsible_parties',
      key: 'responsible_parties',
      render: (parties: string[]) => (
        <Tooltip title={parties.join(', ')}>
          <Badge count={parties.length} style={{ backgroundColor: '#1890ff' }}>
            <TeamOutlined />
          </Badge>
        </Tooltip>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => {
        const color = status === 'active' ? 'green' : status === 'under_review' ? 'orange' : 'gray'
        return <Tag color={color}>{status.replace('_', ' ').toUpperCase()}</Tag>
      }
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: EthicalGuideline) => (
        <Space>
          <Button 
            type="link" 
            size="small" 
            onClick={() => handleModalOpen('guideline', record)}
          >
            查看详情
          </Button>
          <Button type="link" size="small">
            编辑
          </Button>
        </Space>
      )
    }
  ]

  const complianceColumns = [
    {
      title: '法规',
      dataIndex: 'regulation',
      key: 'regulation',
      render: (regulation: string) => (
        <Tag color="geekblue" style={{ fontSize: '12px', fontWeight: 'bold' }}>
          {regulation}
        </Tag>
      )
    },
    {
      title: '要求',
      dataIndex: 'requirement',
      key: 'requirement'
    },
    {
      title: '合规状态',
      dataIndex: 'current_status',
      key: 'current_status',
      render: (status: string) => {
        const config = {
          compliant: { color: 'success', icon: <CheckCircleOutlined /> },
          partial: { color: 'warning', icon: <ExclamationCircleOutlined /> },
          non_compliant: { color: 'error', icon: <CloseCircleOutlined /> },
          unknown: { color: 'default', icon: <WarningOutlined /> }
        }
        const { color, icon } = config[status as keyof typeof config]
        return (
          <Tag color={color}>
            {icon} {status.replace('_', ' ').toUpperCase()}
          </Tag>
        )
      }
    },
    {
      title: '风险等级',
      dataIndex: 'risk_level',
      key: 'risk_level',
      render: (risk: string) => {
        const colors = { low: 'green', medium: 'orange', high: 'red', critical: 'purple' }
        return <Tag color={colors[risk as keyof typeof colors]}>{risk.toUpperCase()}</Tag>
      }
    },
    {
      title: '截止日期',
      dataIndex: 'deadline',
      key: 'deadline',
      render: (date: string) => {
        const isOverdue = new Date(date) < new Date()
        return (
          <span style={{ color: isOverdue ? '#f5222d' : '#666' }}>
            {date}
            {isOverdue && <WarningOutlined style={{ marginLeft: 4, color: '#f5222d' }} />}
          </span>
        )
      }
    },
    {
      title: '操作',
      key: 'action',
      render: (_: any, record: ComplianceCheck) => (
        <Space>
          <Button type="link" size="small">查看详情</Button>
          <Button type="link" size="small">更新状态</Button>
        </Space>
      )
    }
  ]

  const renderOverview = () => (
    <div>
      <Row gutter={[24, 24]}>
        <Col span={6}>
          <Card>
            <Statistic
              title="活跃隐私政策"
              value={privacyPolicies.filter(p => p.status === 'active').length}
              prefix={<FileProtectOutlined />}
              valueStyle={{ color: '#3f8600' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="伦理指导原则"
              value={ethicalGuidelines.filter(g => g.status === 'active').length}
              prefix={<HeartOutlined />}
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="合规检查项"
              value={complianceChecks.length}
              prefix={<AuditOutlined />}
              valueStyle={{ color: '#722ed1' }}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="高风险项"
              value={complianceChecks.filter(c => ['high', 'critical'].includes(c.risk_level)).length}
              prefix={<WarningOutlined />}
              valueStyle={{ color: '#cf1322' }}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={[24, 24]} style={{ marginTop: '24px' }}>
        <Col span={12}>
          <Card title="合规状态分布" size="small">
            <div ref={complianceChartRef}></div>
          </Card>
        </Col>
        <Col span={12}>
          <Card title="风险等级分析" size="small">
            <div ref={riskChartRef}></div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[24, 24]} style={{ marginTop: '24px' }}>
        <Col span={24}>
          <Card title="隐私保护指标" size="small">
            <div ref={chartRef}></div>
          </Card>
        </Col>
      </Row>

      <Row gutter={[24, 24]} style={{ marginTop: '24px' }}>
        <Col span={12}>
          <Card 
            title="近期合规提醒" 
            size="small"
            extra={
              <Badge count={3} size="small">
                <BellOutlined />
              </Badge>
            }
          >
            <List
              size="small"
              dataSource={[
                { text: 'GDPR年度评估即将到期', type: 'warning', date: '2024-12-15' },
                { text: '情感数据保留政策需要更新', type: 'info', date: '2024-12-20' },
                { text: 'CCPA合规状态待确认', type: 'error', date: '2024-12-01' }
              ]}
              renderItem={(item) => (
                <List.Item>
                  <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                    <span style={{ 
                      color: item.type === 'error' ? '#f5222d' : item.type === 'warning' ? '#faad14' : '#1890ff' 
                    }}>
                      {item.text}
                    </span>
                    <span style={{ fontSize: '12px', color: '#999' }}>{item.date}</span>
                  </div>
                </List.Item>
              )}
            />
          </Card>
        </Col>
        <Col span={12}>
          <Card title="快速操作" size="small">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Button 
                type="primary" 
                block 
                icon={<SecurityScanOutlined />}
                onClick={() => message.info('开始隐私影响评估')}
              >
                启动隐私影响评估
              </Button>
              <Button 
                block 
                icon={<AuditOutlined />}
                onClick={() => message.info('生成合规报告')}
              >
                生成合规报告
              </Button>
              <Button 
                block 
                icon={<WarningOutlined />}
                onClick={() => message.info('检查风险项目')}
              >
                风险项目检查
              </Button>
            </Space>
          </Card>
        </Col>
      </Row>
    </div>
  )

  return (
    <div style={{ padding: '24px', backgroundColor: '#f0f2f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <h1 style={{ fontSize: '28px', fontWeight: 'bold', color: '#1890ff', marginBottom: '8px' }}>
          <ShieldOutlined style={{ marginRight: '12px' }} />
          隐私保护与伦理管理系统
        </h1>
        <p style={{ fontSize: '16px', color: '#666', margin: 0 }}>
          全面的隐私保护和伦理合规管理，确保社交情感AI系统的负责任发展
        </p>
      </div>

      <div style={{ marginBottom: '24px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <Space>
          <Switch 
            checkedChildren="实时监控" 
            unCheckedChildren="手动刷新" 
            checked={realTimeMonitoring}
            onChange={setRealTimeMonitoring}
          />
          <Switch 
            checkedChildren="评估模式" 
            unCheckedChildren="查看模式" 
            checked={assessmentMode}
            onChange={setAssessmentMode}
          />
        </Space>
        
        <Space>
          <Button 
            type="primary" 
            icon={<SecurityScanOutlined />}
            onClick={() => handleModalOpen('policy')}
          >
            新建隐私政策
          </Button>
          <Button 
            icon={<HeartOutlined />}
            onClick={() => handleModalOpen('guideline')}
          >
            新建伦理指导原则
          </Button>
          <Button icon={<DashboardOutlined />} loading={loading}>
            刷新数据
          </Button>
        </Space>
      </div>

      <Tabs activeKey={activeTab} onChange={setActiveTab}>
        <TabPane tab={
          <span>
            <DashboardOutlined />
            总览面板
          </span>
        } key="overview">
          {renderOverview()}
        </TabPane>

        <TabPane tab={
          <span>
            <FileProtectOutlined />
            隐私政策
          </span>
        } key="policies">
          <Card title="隐私政策管理">
            <Table
              columns={privacyPolicyColumns}
              dataSource={privacyPolicies}
              rowKey="id"
              loading={loading}
              pagination={{ pageSize: 10 }}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab={
          <span>
            <HeartOutlined />
            伦理指导
          </span>
        } key="ethics">
          <Card title="伦理指导原则">
            <Table
              columns={ethicalGuidelineColumns}
              dataSource={ethicalGuidelines}
              rowKey="id"
              loading={loading}
              pagination={{ pageSize: 10 }}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab={
          <span>
            <AuditOutlined />
            合规检查
          </span>
        } key="compliance">
          <Card title="合规状态监控">
            <Alert
              message="合规性监控"
              description="实时跟踪各项法规要求的合规状态，及时发现和处理合规风险"
              type="info"
              showIcon
              style={{ marginBottom: '16px' }}
            />
            <Table
              columns={complianceColumns}
              dataSource={complianceChecks}
              rowKey="id"
              loading={loading}
              pagination={{ pageSize: 10 }}
              size="small"
            />
          </Card>
        </TabPane>

        <TabPane tab={
          <span>
            <UserOutlined />
            用户同意
          </span>
        } key="consent">
          <Card title="用户同意管理">
            <div style={{ marginBottom: '16px' }}>
              <Alert
                message="用户同意记录"
                description="管理用户对各项数据处理活动的同意状态，支持细粒度权限控制"
                type="info"
                showIcon
              />
            </div>
            
            <Row gutter={16}>
              <Col span={8}>
                <Statistic 
                  title="总同意记录" 
                  value={consentRecords.length} 
                  prefix={<CheckCircleOutlined />}
                />
              </Col>
              <Col span={8}>
                <Statistic 
                  title="同意率" 
                  value={85} 
                  suffix="%" 
                  valueStyle={{ color: '#3f8600' }}
                />
              </Col>
              <Col span={8}>
                <Statistic 
                  title="撤回请求" 
                  value={3} 
                  prefix={<CloseCircleOutlined />}
                  valueStyle={{ color: '#cf1322' }}
                />
              </Col>
            </Row>

            <div style={{ marginTop: '24px' }}>
              <List
                dataSource={consentRecords}
                renderItem={(record) => (
                  <List.Item
                    actions={[
                      <Button type="link" size="small">查看详情</Button>,
                      <Button type="link" size="small">撤回管理</Button>
                    ]}
                  >
                    <List.Item.Meta
                      title={
                        <div>
                          <Tag color="blue">{record.consent_type}</Tag>
                          <span style={{ marginLeft: 8 }}>{record.purpose}</span>
                        </div>
                      }
                      description={
                        <div>
                          <div>用户: {record.user_id}</div>
                          <div>同意时间: {new Date(record.consent_timestamp).toLocaleString()}</div>
                          <div>到期时间: {new Date(record.expiry_date).toLocaleString()}</div>
                        </div>
                      }
                    />
                  </List.Item>
                )}
              />
            </div>
          </Card>
        </TabPane>

        <TabPane tab={
          <span>
            <HistoryOutlined />
            审计日志
          </span>
        } key="audit">
          <Card title="隐私审计日志">
            <Alert
              message="审计追踪"
              description="记录所有涉及个人数据和隐私相关的操作，确保操作的可追溯性"
              type="info"
              showIcon
              style={{ marginBottom: '16px' }}
            />
            
            <Timeline>
              {auditEvents.map((event) => (
                <Timeline.Item 
                  key={event.id}
                  color={event.outcome === 'success' ? 'green' : event.outcome === 'blocked' ? 'red' : 'orange'}
                >
                  <div style={{ marginBottom: '8px' }}>
                    <Tag color="geekblue">{event.event_type}</Tag>
                    <span style={{ marginLeft: '8px', fontWeight: 'bold' }}>
                      {event.action_performed}
                    </span>
                  </div>
                  
                  <div style={{ fontSize: '12px', color: '#666' }}>
                    <div>用户: {event.user_id}</div>
                    <div>资源: {event.resource_accessed}</div>
                    <div>时间: {new Date(event.timestamp).toLocaleString()}</div>
                    <div>地点: {event.geographic_location}</div>
                    <div>风险评分: 
                      <Tag color={event.risk_score > 7 ? 'red' : event.risk_score > 4 ? 'orange' : 'green'}>
                        {event.risk_score}/10
                      </Tag>
                    </div>
                    
                    {event.privacy_implications.length > 0 && (
                      <div style={{ marginTop: '4px' }}>
                        隐私影响: {event.privacy_implications.map(impl => (
                          <Tag key={impl} size="small" color="purple">{impl}</Tag>
                        ))}
                      </div>
                    )}
                    
                    {event.ethical_concerns.length > 0 && (
                      <div style={{ marginTop: '4px' }}>
                        伦理关注: {event.ethical_concerns.map(concern => (
                          <Tag key={concern} size="small" color="orange">{concern}</Tag>
                        ))}
                      </div>
                    )}
                  </div>
                </Timeline.Item>
              ))}
            </Timeline>
          </Card>
        </TabPane>
      </Tabs>

      {/* 详情模态框 */}
      <Modal
        title={`${modalType === 'policy' ? '隐私政策' : modalType === 'guideline' ? '伦理指导原则' : modalType === 'consent' ? '用户同意' : '审计事件'}详情`}
        visible={modalVisible}
        onCancel={handleModalClose}
        width={800}
        footer={[
          <Button key="cancel" onClick={handleModalClose}>
            取消
          </Button>,
          <Button key="submit" type="primary" onClick={() => form.submit()}>
            {selectedItem ? '更新' : '创建'}
          </Button>
        ]}
      >
        {modalType === 'policy' && (
          <Form form={form} layout="vertical" onFinish={handleFormSubmit}>
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="政策名称" name="name" rules={[{ required: true }]}>
                  <Input placeholder="输入隐私政策名称" />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="政策类别" name="category" rules={[{ required: true }]}>
                  <Select placeholder="选择政策类别">
                    <Option value="emotional_data">情感数据</Option>
                    <Option value="social_interaction">社交互动</Option>
                    <Option value="behavioral_analysis">行为分析</Option>
                    <Option value="biometric_data">生物识别</Option>
                  </Select>
                </Form.Item>
              </Col>
            </Row>
            
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item label="敏感级别" name="sensitivity_level">
                  <Slider min={1} max={10} marks={{ 1: '低', 5: '中', 10: '高' }} />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item label="保留期限(天)" name="retention_period">
                  <InputNumber min={1} max={3650} style={{ width: '100%' }} />
                </Form.Item>
              </Col>
            </Row>
            
            <Form.Item label="适用范围" name="scope">
              <Select mode="multiple" placeholder="选择适用范围">
                <Option value="emotion_recognition">情感识别</Option>
                <Option value="empathy_modeling">共情建模</Option>
                <Option value="social_analysis">社交分析</Option>
                <Option value="behavior_prediction">行为预测</Option>
              </Select>
            </Form.Item>
            
            <Row gutter={16}>
              <Col span={8}>
                <Form.Item name="encryption_required" valuePropName="checked">
                  <Switch checkedChildren="需要加密" unCheckedChildren="无需加密" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="audit_required" valuePropName="checked">
                  <Switch checkedChildren="需要审计" unCheckedChildren="无需审计" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="consent_required" valuePropName="checked">
                  <Switch checkedChildren="需要同意" unCheckedChildren="无需同意" />
                </Form.Item>
              </Col>
            </Row>
            
            <Form.Item label="合规标准" name="compliance_standards">
              <Select mode="multiple" placeholder="选择合规标准">
                <Option value="GDPR">GDPR</Option>
                <Option value="CCPA">CCPA</Option>
                <Option value="PIPEDA">PIPEDA</Option>
                <Option value="ISO27001">ISO27001</Option>
                <Option value="SOC2">SOC2</Option>
              </Select>
            </Form.Item>
          </Form>
        )}
        
        {modalType === 'guideline' && (
          <Form form={form} layout="vertical" onFinish={handleFormSubmit}>
            <Form.Item label="伦理原则" name="principle" rules={[{ required: true }]}>
              <Input placeholder="输入伦理原则名称" />
            </Form.Item>
            
            <Form.Item label="原则类别" name="category" rules={[{ required: true }]}>
              <Select placeholder="选择原则类别">
                <Option value="data_minimization">数据最小化</Option>
                <Option value="algorithmic_fairness">算法公平</Option>
                <Option value="transparency">透明度</Option>
                <Option value="accountability">问责制</Option>
                <Option value="human_dignity">人类尊严</Option>
              </Select>
            </Form.Item>
            
            <Form.Item label="描述" name="description" rules={[{ required: true }]}>
              <TextArea rows={3} placeholder="详细描述伦理原则的内容和要求" />
            </Form.Item>
            
            <Form.Item label="影响等级" name="impact_level">
              <Slider min={1} max={10} marks={{ 1: '低', 5: '中', 10: '高' }} />
            </Form.Item>
            
            <Form.Item label="适用领域" name="applicability">
              <Select mode="multiple" placeholder="选择适用领域">
                <Option value="emotion_recognition">情感识别</Option>
                <Option value="empathy_response">共情回应</Option>
                <Option value="bias_detection">偏见检测</Option>
                <Option value="decision_making">决策制定</Option>
              </Select>
            </Form.Item>
            
            <Form.Item label="审查频率(天)" name="review_frequency">
              <InputNumber min={7} max={365} style={{ width: '100%' }} />
            </Form.Item>
          </Form>
        )}
        
        {selectedItem && modalType === 'policy' && (
          <Descriptions title="政策详情" bordered size="small" column={2}>
            <Descriptions.Item label="创建日期">{selectedItem.created_date}</Descriptions.Item>
            <Descriptions.Item label="最后更新">{selectedItem.last_updated}</Descriptions.Item>
            <Descriptions.Item label="地理限制">
              {selectedItem.geographical_restrictions.join(', ') || '无'}
            </Descriptions.Item>
            <Descriptions.Item label="第三方共享">
              <Tag color={selectedItem.third_party_sharing ? 'orange' : 'green'}>
                {selectedItem.third_party_sharing ? '允许' : '禁止'}
              </Tag>
            </Descriptions.Item>
            <Descriptions.Item label="访问控制" span={2}>
              {selectedItem.access_control.map((control: string) => (
                <Tag key={control} color="blue">{control}</Tag>
              ))}
            </Descriptions.Item>
          </Descriptions>
        )}
        
        {selectedItem && modalType === 'guideline' && (
          <div>
            <Descriptions title="指导原则详情" bordered size="small" column={1}>
              <Descriptions.Item label="实施步骤">
                <ol>
                  {selectedItem.implementation_steps.map((step: string, idx: number) => (
                    <li key={idx}>{step}</li>
                  ))}
                </ol>
              </Descriptions.Item>
              <Descriptions.Item label="监控指标">
                {selectedItem.monitoring_metrics.map((metric: string) => (
                  <Tag key={metric} color="cyan">{metric}</Tag>
                ))}
              </Descriptions.Item>
              <Descriptions.Item label="违规后果">
                {selectedItem.violation_consequences.map((consequence: string) => (
                  <Tag key={consequence} color="red">{consequence}</Tag>
                ))}
              </Descriptions.Item>
              <Descriptions.Item label="文化考量">
                {selectedItem.cultural_considerations.map((consideration: string) => (
                  <Tag key={consideration} color="purple">{consideration}</Tag>
                ))}
              </Descriptions.Item>
            </Descriptions>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default PrivacyEthicsPage