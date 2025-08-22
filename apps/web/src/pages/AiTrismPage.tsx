import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Statistic,
  Progress,
  Table,
  Tag,
  Button,
  Space,
  Alert,
  Tabs,
  Typography,
  List,
  Badge,
  Switch
} from 'antd'
import {
  SafetyOutlined,
  ExclamationCircleOutlined,
  CheckOutlined,
  AlertOutlined,
  CheckCircleOutlined,
  ReloadOutlined
} from '@ant-design/icons'

const { Title, Text } = Typography
const { TabPane } = Tabs

interface TrustMetric {
  component: string
  score: number
  last_evaluated: string
  status: 'high' | 'medium' | 'low'
  details: string
}

interface RiskAssessment {
  id: string
  category: 'bias' | 'privacy' | 'security' | 'fairness' | 'transparency'
  level: 'critical' | 'high' | 'medium' | 'low'
  description: string
  affected_models: string[]
  mitigation_status: 'planned' | 'in_progress' | 'completed' | 'pending'
  created_at: string
  updated_at: string
}

interface SecurityIncident {
  id: string
  type: 'data_breach' | 'model_poisoning' | 'adversarial_attack' | 'prompt_injection' | 'other'
  severity: 'critical' | 'high' | 'medium' | 'low'
  status: 'detected' | 'investigating' | 'mitigated' | 'resolved'
  description: string
  affected_components: string[]
  detection_time: string
  resolution_time?: string
  auto_response_triggered: boolean
}

interface ComplianceCheck {
  framework: string
  requirement: string
  status: 'compliant' | 'non_compliant' | 'partial' | 'unknown'
  last_check: string
  evidence: string[]
  action_required: boolean
}

const AiTrismPage: React.FC = () => {
  const [trustMetrics, setTrustMetrics] = useState<TrustMetric[]>([])
  const [riskAssessments, setRiskAssessments] = useState<RiskAssessment[]>([])
  const [securityIncidents, setSecurityIncidents] = useState<SecurityIncident[]>([])
  const [complianceChecks, setComplianceChecks] = useState<ComplianceCheck[]>([])
  const [autoResponseEnabled, setAutoResponseEnabled] = useState(true)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadTrustMetrics()
    loadRiskAssessments()
    loadSecurityIncidents()
    loadComplianceChecks()
  }, [])

  const loadTrustMetrics = async () => {
    // 模拟Trust信任指标数据
    const mockMetrics: TrustMetric[] = [
      {
        component: 'GPT-4o模型',
        score: 87,
        last_evaluated: '2024-01-16T10:30:00Z',
        status: 'high',
        details: '模型输出一致性良好，偏见检测通过'
      },
      {
        component: 'RAG检索系统',
        score: 92,
        last_evaluated: '2024-01-16T09:15:00Z',
        status: 'high',
        details: '检索准确性优秀，知识更新及时'
      },
      {
        component: '多模态处理器',
        score: 73,
        last_evaluated: '2024-01-16T08:45:00Z',
        status: 'medium',
        details: '图像理解准确性需要改进'
      },
      {
        component: '对话生成引擎',
        score: 65,
        last_evaluated: '2024-01-15T16:20:00Z',
        status: 'medium',
        details: '存在轻微偏见倾向，需要调优'
      }
    ]
    setTrustMetrics(mockMetrics)
  }

  const loadRiskAssessments = async () => {
    // 模拟Risk风险评估数据
    const mockRisks: RiskAssessment[] = [
      {
        id: 'risk-001',
        category: 'bias',
        level: 'high',
        description: '对话生成中检测到性别偏见倾向',
        affected_models: ['conversation-model-v2.1'],
        mitigation_status: 'in_progress',
        created_at: '2024-01-15T14:30:00Z',
        updated_at: '2024-01-16T09:15:00Z'
      },
      {
        id: 'risk-002',
        category: 'privacy',
        level: 'medium',
        description: '用户数据在向量化过程中可能存在泄露风险',
        affected_models: ['embedding-model-v1.3', 'rag-retriever-v2.0'],
        mitigation_status: 'planned',
        created_at: '2024-01-14T11:20:00Z',
        updated_at: '2024-01-16T08:30:00Z'
      },
      {
        id: 'risk-003',
        category: 'security',
        level: 'critical',
        description: '发现潜在的提示注入攻击漏洞',
        affected_models: ['chat-interface-v1.5'],
        mitigation_status: 'completed',
        created_at: '2024-01-13T16:45:00Z',
        updated_at: '2024-01-16T12:00:00Z'
      }
    ]
    setRiskAssessments(mockRisks)
  }

  const loadSecurityIncidents = async () => {
    // 模拟Security安全事件数据
    const mockIncidents: SecurityIncident[] = [
      {
        id: 'inc-001',
        type: 'prompt_injection',
        severity: 'high',
        status: 'mitigated',
        description: '检测到针对对话系统的提示注入攻击',
        affected_components: ['chat-api', 'prompt-processor'],
        detection_time: '2024-01-16T14:20:00Z',
        resolution_time: '2024-01-16T14:35:00Z',
        auto_response_triggered: true
      },
      {
        id: 'inc-002',
        type: 'adversarial_attack',
        severity: 'medium',
        status: 'investigating',
        description: '多模态模型遭受对抗性样本攻击',
        affected_components: ['multimodal-processor', 'image-classifier'],
        detection_time: '2024-01-16T13:45:00Z',
        auto_response_triggered: false
      },
      {
        id: 'inc-003',
        type: 'data_breach',
        severity: 'critical',
        status: 'resolved',
        description: '训练数据存储异常访问尝试',
        affected_components: ['training-data-store', 'model-registry'],
        detection_time: '2024-01-15T22:30:00Z',
        resolution_time: '2024-01-16T02:15:00Z',
        auto_response_triggered: true
      }
    ]
    setSecurityIncidents(mockIncidents)
  }

  const loadComplianceChecks = async () => {
    // 模拟合规检查数据
    const mockCompliance: ComplianceCheck[] = [
      {
        framework: 'GDPR',
        requirement: '数据主体权利保护',
        status: 'compliant',
        last_check: '2024-01-16T08:00:00Z',
        evidence: ['data-deletion-logs.pdf', 'privacy-policy-v2.3.pdf'],
        action_required: false
      },
      {
        framework: 'ISO27001',
        requirement: '信息安全管理体系',
        status: 'partial',
        last_check: '2024-01-15T16:30:00Z',
        evidence: ['security-audit-2024-q1.pdf'],
        action_required: true
      },
      {
        framework: 'NIST AI RMF',
        requirement: 'AI系统风险管理',
        status: 'compliant',
        last_check: '2024-01-16T10:15:00Z',
        evidence: ['ai-risk-assessment-v1.2.pdf', 'mitigation-plan-2024.pdf'],
        action_required: false
      }
    ]
    setComplianceChecks(mockCompliance)
  }

  const getTrustScoreColor = (score: number) => {
    if (score >= 80) return '#52c41a'
    if (score >= 60) return '#faad14'
    return '#ff4d4f'
  }

  const getRiskLevelColor = (level: string) => {
    const colors = {
      critical: 'red',
      high: 'orange',
      medium: 'gold',
      low: 'green'
    }
    return colors[level as keyof typeof colors]
  }

  const getStatusColor = (status: string) => {
    const colors = {
      compliant: 'green',
      non_compliant: 'red',
      partial: 'orange',
      unknown: 'default',
      completed: 'green',
      in_progress: 'blue',
      planned: 'orange',
      pending: 'default',
      detected: 'red',
      investigating: 'orange',
      mitigated: 'blue',
      resolved: 'green'
    }
    return colors[status as keyof typeof colors]
  }

  const trustColumns = [
    {
      title: '组件',
      dataIndex: 'component',
      key: 'component'
    },
    {
      title: '信任分数',
      dataIndex: 'score',
      key: 'score',
      render: (score: number) => (
        <div style={{ width: '120px' }}>
          <Progress 
            percent={score} 
            strokeColor={getTrustScoreColor(score)}
            format={() => `${score}/100`}
          />
        </div>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={status === 'high' ? 'green' : status === 'medium' ? 'orange' : 'red'}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '最后评估',
      dataIndex: 'last_evaluated',
      key: 'last_evaluated',
      render: (time: string) => new Date(time).toLocaleString()
    },
    {
      title: '详情',
      dataIndex: 'details',
      key: 'details'
    }
  ]

  const riskColumns = [
    {
      title: '风险ID',
      dataIndex: 'id',
      key: 'id',
      render: (id: string) => <code>{id}</code>
    },
    {
      title: '类别',
      dataIndex: 'category',
      key: 'category',
      render: (category: string) => (
        <Tag color="blue">{category.toUpperCase()}</Tag>
      )
    },
    {
      title: '风险等级',
      dataIndex: 'level',
      key: 'level',
      render: (level: string) => (
        <Tag color={getRiskLevelColor(level)}>
          {level.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '描述',
      dataIndex: 'description',
      key: 'description'
    },
    {
      title: '缓解状态',
      dataIndex: 'mitigation_status',
      key: 'mitigation_status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '更新时间',
      dataIndex: 'updated_at',
      key: 'updated_at',
      render: (time: string) => new Date(time).toLocaleString()
    }
  ]

  const securityColumns = [
    {
      title: '事件ID',
      dataIndex: 'id',
      key: 'id',
      render: (id: string) => <code>{id}</code>
    },
    {
      title: '类型',
      dataIndex: 'type',
      key: 'type',
      render: (type: string) => (
        <Tag color="red">{type.replace('_', ' ').toUpperCase()}</Tag>
      )
    },
    {
      title: '严重程度',
      dataIndex: 'severity',
      key: 'severity',
      render: (severity: string) => (
        <Tag color={getRiskLevelColor(severity)}>
          {severity.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (status: string) => (
        <Tag color={getStatusColor(status)}>
          {status.toUpperCase()}
        </Tag>
      )
    },
    {
      title: '自动响应',
      dataIndex: 'auto_response_triggered',
      key: 'auto_response_triggered',
      render: (triggered: boolean) => (
        <Tag color={triggered ? 'green' : 'default'}>
          {triggered ? '已触发' : '未触发'}
        </Tag>
      )
    },
    {
      title: '检测时间',
      dataIndex: 'detection_time',
      key: 'detection_time',
      render: (time: string) => new Date(time).toLocaleString()
    }
  ]

  return (
    <div className="p-6">
      <div className="mb-6">
        <div className="flex justify-between items-center mb-4">
          <Title level={2}>AI TRiSM框架 (Trust/Risk/Security)</Title>
          <Space>
            <Switch
              checked={autoResponseEnabled}
              onChange={setAutoResponseEnabled}
              checkedChildren="自动响应"
              unCheckedChildren="手动响应"
            />
            <Button 
              icon={<ReloadOutlined />}
              onClick={() => {
                loadTrustMetrics()
                loadRiskAssessments()
                loadSecurityIncidents()
                loadComplianceChecks()
              }}
              loading={loading}
            >
              刷新数据
            </Button>
          </Space>
        </div>

        <Row gutter={16} className="mb-6">
          <Col span={6}>
            <Card>
              <Statistic
                title="总体信任分数"
                value={79}
                suffix="/100"
                valueStyle={{ color: getTrustScoreColor(79) }}
                prefix={<SafetyOutlined />}
              />
              <Progress 
                percent={79} 
                strokeColor={getTrustScoreColor(79)}
                showInfo={false}
                className="mt-2"
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="高风险项目"
                value={riskAssessments.filter(r => r.level === 'critical' || r.level === 'high').length}
                valueStyle={{ color: '#ff4d4f' }}
                prefix={<ExclamationCircleOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="安全事件"
                value={securityIncidents.filter(i => i.status !== 'resolved').length}
                suffix="待处理"
                valueStyle={{ color: '#faad14' }}
                prefix={<AlertOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="合规率"
                value={Math.round((complianceChecks.filter(c => c.status === 'compliant').length / complianceChecks.length) * 100)}
                suffix="%"
                valueStyle={{ color: '#52c41a' }}
                prefix={<CheckCircleOutlined />}
              />
            </Card>
          </Col>
        </Row>

        <Alert
          message="AI TRiSM框架状态"
          description="当前系统整体信任度良好，存在1个高风险项目和2个待处理安全事件，建议优先处理提示注入攻击防护。"
          variant="warning"
          showIcon
          className="mb-6"
        />
      </div>

      <Tabs defaultActiveKey="trust">
        <TabPane tab={
          <span>
            <SafetyOutlined />
            Trust (信任)
          </span>
        } key="trust">
          <Card title="信任度评估">
            <Table
              columns={trustColumns}
              dataSource={trustMetrics}
              rowKey="component"
              pagination={false}
            />
          </Card>
        </TabPane>

        <TabPane tab={
          <span>
            <ExclamationCircleOutlined />
            Risk (风险)
          </span>
        } key="risk">
          <Card title="风险评估与管理">
            <Table
              columns={riskColumns}
              dataSource={riskAssessments}
              rowKey="id"
              pagination={{ pageSize: 10 }}
            />
          </Card>
        </TabPane>

        <TabPane tab={
          <span>
            <SafetyOutlined />
            Security (安全)
          </span>
        } key="security">
          <Card title="安全事件监控">
            <Table
              columns={securityColumns}
              dataSource={securityIncidents}
              rowKey="id"
              pagination={{ pageSize: 10 }}
              expandable={{
                expandedRowRender: (record) => (
                  <div className="p-4">
                    <Title level={5}>事件详情</Title>
                    <p><strong>描述:</strong> {record.description}</p>
                    <p><strong>影响组件:</strong> {record.affected_components.join(', ')}</p>
                    {record.resolution_time && (
                      <p><strong>解决时间:</strong> {new Date(record.resolution_time).toLocaleString()}</p>
                    )}
                  </div>
                )
              }}
            />
          </Card>
        </TabPane>

        <TabPane tab={
          <span>
            <CheckCircleOutlined />
            合规监控
          </span>
        } key="compliance">
          <Card title="合规性检查">
            <List
              dataSource={complianceChecks}
              renderItem={item => (
                <List.Item
                  actions={[
                    <Tag color={getStatusColor(item.status)}>
                      {item.status.replace('_', ' ').toUpperCase()}
                    </Tag>,
                    item.action_required && (
                      <Badge status="error" text="需要行动" />
                    )
                  ]}
                >
                  <List.Item.Meta
                    title={`${item.framework} - ${item.requirement}`}
                    description={
                      <div>
                        <Text type="secondary">
                          最后检查: {new Date(item.last_check).toLocaleString()}
                        </Text>
                        <br />
                        <Text type="secondary">
                          证据文件: {item.evidence.join(', ')}
                        </Text>
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </Card>
        </TabPane>
      </Tabs>
    </div>
  )
}

export default AiTrismPage