import React, { useState, useEffect } from 'react'
import {
  Card,
  Row,
  Col,
  Table,
  Button,
  Form,
  Input,
  Select,
  Space,
  Typography,
  Alert,
  Tag,
  Modal,
  Tabs,
  Badge,
  Progress,
  Statistic,
  Tree,
  Switch,
  Tooltip,
  Divider,
  Timeline,
  notification,
  Radio,
  Drawer,
  Steps
} from 'antd'
import {
  SafetyCertificateOutlined,
  UserOutlined,
  TeamOutlined,
  KeyOutlined,
  LockOutlined,
  UnlockOutlined,
  SettingOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
  EyeOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ExclamationCircleOutlined,
  MessageOutlined,
  ApiOutlined,
  ReloadOutlined,
  SearchOutlined,
  FilterOutlined,
  ShareAltOutlined,
  SecurityScanOutlined,
  AuditOutlined,
  BranchesOutlined
} from '@ant-design/icons'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TextArea } = Input
const { TabPane } = Tabs
const { TreeNode } = Tree
const { Step } = Steps

interface ACLRule {
  id: string
  name: string
  description: string
  principal: string
  principalType: 'agent' | 'group' | 'role'
  permission: 'allow' | 'deny'
  resource: string
  resourceType: 'subject' | 'stream' | 'queue' | 'topic'
  actions: string[]
  conditions?: string[]
  priority: number
  enabled: boolean
  createdAt: string
  updatedAt: string
  createdBy: string
  matchCount: number
  lastMatched?: string
}

interface ACLGroup {
  id: string
  name: string
  description: string
  members: string[]
  rules: string[]
  enabled: boolean
  createdAt: string
  parentGroup?: string
}

interface ACLRole {
  id: string
  name: string
  description: string
  permissions: string[]
  inheritedRoles: string[]
  assignedAgents: string[]
  enabled: boolean
  createdAt: string
}

interface ACLAuditLog {
  id: string
  timestamp: string
  event: 'rule_matched' | 'rule_created' | 'rule_updated' | 'rule_deleted' | 'access_granted' | 'access_denied'
  principal: string
  resource: string
  action: string
  result: 'allow' | 'deny'
  ruleId?: string
  ruleName?: string
  details: string
  sourceIp?: string
  userAgent?: string
}

interface SecurityMetrics {
  totalRules: number
  activeRules: number
  ruleMatches: number
  accessGranted: number
  accessDenied: number
  criticalViolations: number
  lastViolation?: string
}

const ACLProtocolManagementPage: React.FC = () => {
  const [form] = Form.useForm()
  const [activeTab, setActiveTab] = useState('rules')
  const [loading, setLoading] = useState(false)
  const [ruleModalVisible, setRuleModalVisible] = useState(false)
  const [groupModalVisible, setGroupModalVisible] = useState(false)
  const [roleModalVisible, setRoleModalVisible] = useState(false)
  const [auditDrawerVisible, setAuditDrawerVisible] = useState(false)
  const [selectedRule, setSelectedRule] = useState<ACLRule | null>(null)
  const [filterPrincipal, setFilterPrincipal] = useState<string>('')
  const [filterResource, setFilterResource] = useState<string>('')

  const [securityMetrics, setSecurityMetrics] = useState<SecurityMetrics>({
    totalRules: 47,
    activeRules: 43,
    ruleMatches: 15642,
    accessGranted: 14956,
    accessDenied: 686,
    criticalViolations: 3,
    lastViolation: '2025-08-26 11:45:23'
  })

  const [aclRules, setAclRules] = useState<ACLRule[]>([
    {
      id: 'rule-001',
      name: '任务处理智能体基础权限',
      description: '允许任务处理智能体访问任务相关主题',
      principal: 'task-agent-group',
      principalType: 'group',
      permission: 'allow',
      resource: 'agents.tasks.>',
      resourceType: 'subject',
      actions: ['subscribe', 'publish'],
      conditions: ['source_ip_whitelist', 'time_window'],
      priority: 100,
      enabled: true,
      createdAt: '2025-08-20 10:30:00',
      updatedAt: '2025-08-25 14:20:00',
      createdBy: 'admin',
      matchCount: 2547,
      lastMatched: '2025-08-26 12:44:30'
    },
    {
      id: 'rule-002',
      name: '系统管理权限控制',
      description: '限制系统管理主题只允许管理员访问',
      principal: 'admin-role',
      principalType: 'role',
      permission: 'allow',
      resource: 'system.management.>',
      resourceType: 'subject',
      actions: ['subscribe', 'publish', 'admin'],
      priority: 200,
      enabled: true,
      createdAt: '2025-08-18 09:15:00',
      updatedAt: '2025-08-24 16:45:00',
      createdBy: 'system',
      matchCount: 156,
      lastMatched: '2025-08-26 12:30:15'
    },
    {
      id: 'rule-003',
      name: '敏感数据访问拒绝',
      description: '拒绝未授权智能体访问敏感数据主题',
      principal: '*',
      principalType: 'agent',
      permission: 'deny',
      resource: 'system.sensitive.>',
      resourceType: 'subject',
      actions: ['*'],
      conditions: ['!has_sensitive_clearance'],
      priority: 300,
      enabled: true,
      createdAt: '2025-08-19 14:20:00',
      updatedAt: '2025-08-26 09:10:00',
      createdBy: 'security-admin',
      matchCount: 23,
      lastMatched: '2025-08-26 11:45:23'
    },
    {
      id: 'rule-004',
      name: '客户端直接通信限制',
      description: '限制客户端智能体只能发送直接消息',
      principal: 'client-*',
      principalType: 'agent',
      permission: 'allow',
      resource: 'agents.direct.>',
      resourceType: 'subject',
      actions: ['publish'],
      conditions: ['rate_limit_100_per_minute'],
      priority: 150,
      enabled: true,
      createdAt: '2025-08-21 11:40:00',
      updatedAt: '2025-08-25 08:30:00',
      createdBy: 'system-admin',
      matchCount: 894,
      lastMatched: '2025-08-26 12:45:12'
    }
  ])

  const [aclGroups, setAclGroups] = useState<ACLGroup[]>([
    {
      id: 'group-001',
      name: 'task-agent-group',
      description: '任务处理智能体组',
      members: ['task-agent-01', 'task-agent-02', 'task-agent-03'],
      rules: ['rule-001'],
      enabled: true,
      createdAt: '2025-08-20 10:00:00'
    },
    {
      id: 'group-002',
      name: 'worker-agent-group',
      description: '工作执行智能体组',
      members: ['worker-agent-01', 'worker-agent-02', 'worker-agent-03', 'worker-agent-04'],
      rules: ['rule-001'],
      enabled: true,
      createdAt: '2025-08-20 10:15:00',
      parentGroup: 'group-001'
    },
    {
      id: 'group-003',
      name: 'client-group',
      description: '客户端智能体组',
      members: ['client-agent-01', 'client-agent-02'],
      rules: ['rule-004'],
      enabled: true,
      createdAt: '2025-08-21 11:30:00'
    }
  ])

  const [aclRoles, setAclRoles] = useState<ACLRole[]>([
    {
      id: 'role-001',
      name: 'admin-role',
      description: '系统管理员角色',
      permissions: ['system.admin', 'user.manage', 'rule.manage'],
      inheritedRoles: [],
      assignedAgents: ['admin-agent', 'system-monitor-agent'],
      enabled: true,
      createdAt: '2025-08-18 09:00:00'
    },
    {
      id: 'role-002',
      name: 'operator-role',
      description: '系统操作员角色',
      permissions: ['system.read', 'task.manage', 'monitor.view'],
      inheritedRoles: [],
      assignedAgents: ['operator-agent-01', 'operator-agent-02'],
      enabled: true,
      createdAt: '2025-08-19 10:30:00'
    },
    {
      id: 'role-003',
      name: 'readonly-role',
      description: '只读访问角色',
      permissions: ['system.read', 'monitor.view'],
      inheritedRoles: [],
      assignedAgents: ['monitor-agent', 'audit-agent'],
      enabled: true,
      createdAt: '2025-08-20 14:15:00'
    }
  ])

  const [auditLogs, setAuditLogs] = useState<ACLAuditLog[]>([
    {
      id: 'audit-001',
      timestamp: '2025-08-26 12:45:30',
      event: 'access_granted',
      principal: 'task-agent-01',
      resource: 'agents.tasks.process',
      action: 'publish',
      result: 'allow',
      ruleId: 'rule-001',
      ruleName: '任务处理智能体基础权限',
      details: '规则匹配成功，允许发布消息',
      sourceIp: '192.168.1.100'
    },
    {
      id: 'audit-002',
      timestamp: '2025-08-26 11:45:23',
      event: 'access_denied',
      principal: 'unauthorized-agent',
      resource: 'system.sensitive.keys',
      action: 'subscribe',
      result: 'deny',
      ruleId: 'rule-003',
      ruleName: '敏感数据访问拒绝',
      details: '未授权访问敏感主题，拒绝请求',
      sourceIp: '10.0.0.50'
    },
    {
      id: 'audit-003',
      timestamp: '2025-08-26 10:30:15',
      event: 'rule_created',
      principal: 'security-admin',
      resource: 'rule-005',
      action: 'create',
      result: 'allow',
      details: '创建新的ACL规则：临时访问控制',
      sourceIp: '192.168.1.10',
      userAgent: 'ACL-Manager/1.0'
    }
  ])

  const ruleColumns = [
    {
      title: '规则信息',
      key: 'info',
      width: 300,
      render: (record: ACLRule) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Badge status={record.enabled ? 'success' : 'default'} />
            <Text strong style={{ marginLeft: '8px', fontSize: '13px' }}>{record.name}</Text>
            <Tag 
              color={record.permission === 'allow' ? 'green' : 'red'} 
              style={{ marginLeft: '8px', fontSize: '10px' }}
            >
              {record.permission === 'allow' ? '允许' : '拒绝'}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.description}
          </Text>
        </div>
      )
    },
    {
      title: '主体',
      key: 'principal',
      width: 150,
      render: (record: ACLRule) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Tag color={
              record.principalType === 'agent' ? 'blue' :
              record.principalType === 'group' ? 'green' : 'orange'
            } style={{ fontSize: '10px' }}>
              {record.principalType === 'agent' ? '智能体' :
               record.principalType === 'group' ? '组' : '角色'}
            </Tag>
          </div>
          <Text style={{ fontSize: '12px' }}>{record.principal}</Text>
        </div>
      )
    },
    {
      title: '资源',
      key: 'resource',
      width: 200,
      render: (record: ACLRule) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Tag color="purple" style={{ fontSize: '10px' }}>
              {record.resourceType}
            </Tag>
          </div>
          <Text code style={{ fontSize: '11px' }}>{record.resource}</Text>
        </div>
      )
    },
    {
      title: '操作',
      dataIndex: 'actions',
      key: 'actions',
      width: 120,
      render: (actions: string[]) => (
        <div>
          {actions.slice(0, 2).map((action, index) => (
            <Tag key={index} style={{ fontSize: '10px', marginBottom: '2px' }}>
              {action}
            </Tag>
          ))}
          {actions.length > 2 && (
            <Text type="secondary" style={{ fontSize: '10px' }}>
              +{actions.length - 2}
            </Text>
          )}
        </div>
      )
    },
    {
      title: '匹配统计',
      key: 'stats',
      width: 100,
      render: (record: ACLRule) => (
        <div>
          <div style={{ marginBottom: '2px' }}>
            <Text style={{ fontSize: '11px' }}>匹配: {record.matchCount}</Text>
          </div>
          {record.lastMatched && (
            <Text type="secondary" style={{ fontSize: '10px' }}>
              {record.lastMatched}
            </Text>
          )}
        </div>
      )
    },
    {
      title: '优先级',
      dataIndex: 'priority',
      key: 'priority',
      width: 80,
      render: (priority: number) => (
        <Tag color={priority >= 200 ? 'red' : priority >= 150 ? 'orange' : 'blue'}>
          {priority}
        </Tag>
      )
    },
    {
      title: '操作',
      key: 'operations',
      width: 120,
      render: (record: ACLRule) => (
        <Space>
          <Tooltip title="查看详情">
            <Button 
              type="text" 
              size="small" 
              icon={<EyeOutlined />}
              onClick={() => handleViewRule(record)}
            />
          </Tooltip>
          <Tooltip title="编辑规则">
            <Button 
              type="text" 
              size="small" 
              icon={<EditOutlined />}
              onClick={() => handleEditRule(record)}
            />
          </Tooltip>
          <Tooltip title="删除规则">
            <Button 
              type="text" 
              size="small" 
              icon={<DeleteOutlined />}
              danger
              onClick={() => handleDeleteRule(record)}
            />
          </Tooltip>
        </Space>
      )
    }
  ]

  const groupColumns = [
    {
      title: '组信息',
      key: 'info',
      render: (record: ACLGroup) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Badge status={record.enabled ? 'success' : 'default'} />
            <Text strong style={{ marginLeft: '8px' }}>{record.name}</Text>
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.description}
          </Text>
        </div>
      )
    },
    {
      title: '成员数量',
      key: 'members',
      render: (record: ACLGroup) => (
        <Statistic value={record.members.length} suffix="个" />
      )
    },
    {
      title: '关联规则',
      key: 'rules',
      render: (record: ACLGroup) => (
        <Text>{record.rules.length} 条规则</Text>
      )
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt'
    },
    {
      title: '操作',
      key: 'operations',
      render: (record: ACLGroup) => (
        <Space>
          <Button type="text" size="small" icon={<EyeOutlined />} />
          <Button type="text" size="small" icon={<EditOutlined />} />
          <Button type="text" size="small" icon={<DeleteOutlined />} danger />
        </Space>
      )
    }
  ]

  const roleColumns = [
    {
      title: '角色信息',
      key: 'info',
      render: (record: ACLRole) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Badge status={record.enabled ? 'success' : 'default'} />
            <Text strong style={{ marginLeft: '8px' }}>{record.name}</Text>
          </div>
          <Text type="secondary" style={{ fontSize: '12px' }}>
            {record.description}
          </Text>
        </div>
      )
    },
    {
      title: '权限',
      dataIndex: 'permissions',
      key: 'permissions',
      render: (permissions: string[]) => (
        <div>
          {permissions.slice(0, 2).map((perm, index) => (
            <Tag key={index} style={{ fontSize: '10px', marginBottom: '2px' }}>
              {perm}
            </Tag>
          ))}
          {permissions.length > 2 && (
            <Text type="secondary" style={{ fontSize: '10px' }}>
              +{permissions.length - 2}
            </Text>
          )}
        </div>
      )
    },
    {
      title: '分配的智能体',
      key: 'agents',
      render: (record: ACLRole) => (
        <Text>{record.assignedAgents.length} 个智能体</Text>
      )
    },
    {
      title: '创建时间',
      dataIndex: 'createdAt',
      key: 'createdAt'
    },
    {
      title: '操作',
      key: 'operations',
      render: () => (
        <Space>
          <Button type="text" size="small" icon={<EyeOutlined />} />
          <Button type="text" size="small" icon={<EditOutlined />} />
          <Button type="text" size="small" icon={<DeleteOutlined />} danger />
        </Space>
      )
    }
  ]

  const auditColumns = [
    {
      title: '时间',
      dataIndex: 'timestamp',
      key: 'timestamp',
      width: 150
    },
    {
      title: '事件',
      dataIndex: 'event',
      key: 'event',
      width: 120,
      render: (event: string) => (
        <Tag color={
          event.includes('granted') ? 'green' :
          event.includes('denied') ? 'red' :
          event.includes('created') || event.includes('updated') ? 'blue' :
          'orange'
        }>
          {event}
        </Tag>
      )
    },
    {
      title: '主体',
      dataIndex: 'principal',
      key: 'principal',
      width: 120
    },
    {
      title: '资源',
      dataIndex: 'resource',
      key: 'resource',
      width: 150,
      render: (resource: string) => <Text code style={{ fontSize: '11px' }}>{resource}</Text>
    },
    {
      title: '操作',
      dataIndex: 'action',
      key: 'action',
      width: 80
    },
    {
      title: '结果',
      dataIndex: 'result',
      key: 'result',
      width: 80,
      render: (result: string) => (
        <Badge 
          status={result === 'allow' ? 'success' : 'error'}
          text={result === 'allow' ? '允许' : '拒绝'}
        />
      )
    },
    {
      title: '详情',
      dataIndex: 'details',
      key: 'details',
      ellipsis: true
    }
  ]

  const handleViewRule = (rule: ACLRule) => {
    Modal.info({
      title: '规则详情',
      width: 800,
      content: (
        <div>
          <Divider>基本信息</Divider>
          <Row gutter={[16, 8]}>
            <Col span={12}>
              <Text strong>规则ID: </Text>
              <Text code>{rule.id}</Text>
            </Col>
            <Col span={12}>
              <Text strong>规则名称: </Text>
              <Text>{rule.name}</Text>
            </Col>
            <Col span={24}>
              <Text strong>描述: </Text>
              <Text>{rule.description}</Text>
            </Col>
            <Col span={12}>
              <Text strong>创建者: </Text>
              <Text>{rule.createdBy}</Text>
            </Col>
            <Col span={12}>
              <Text strong>创建时间: </Text>
              <Text>{rule.createdAt}</Text>
            </Col>
          </Row>

          <Divider>权限配置</Divider>
          <Row gutter={[16, 8]}>
            <Col span={8}>
              <Text strong>权限类型: </Text>
              <Tag color={rule.permission === 'allow' ? 'green' : 'red'}>
                {rule.permission === 'allow' ? '允许' : '拒绝'}
              </Tag>
            </Col>
            <Col span={8}>
              <Text strong>主体类型: </Text>
              <Tag color="blue">{rule.principalType}</Tag>
            </Col>
            <Col span={8}>
              <Text strong>优先级: </Text>
              <Tag>{rule.priority}</Tag>
            </Col>
            <Col span={12}>
              <Text strong>主体: </Text>
              <Text code>{rule.principal}</Text>
            </Col>
            <Col span={12}>
              <Text strong>资源: </Text>
              <Text code>{rule.resource}</Text>
            </Col>
          </Row>

          <Divider>操作和条件</Divider>
          <Row gutter={[16, 8]}>
            <Col span={12}>
              <Text strong>允许操作: </Text>
              <div style={{ marginTop: '4px' }}>
                {rule.actions.map((action, index) => (
                  <Tag key={index} style={{ marginBottom: '4px' }}>{action}</Tag>
                ))}
              </div>
            </Col>
            <Col span={12}>
              <Text strong>附加条件: </Text>
              <div style={{ marginTop: '4px' }}>
                {rule.conditions?.map((condition, index) => (
                  <Tag key={index} color="orange" style={{ marginBottom: '4px' }}>{condition}</Tag>
                )) || <Text type="secondary">无</Text>}
              </div>
            </Col>
          </Row>

          <Divider>使用统计</Divider>
          <Row gutter={[16, 8]}>
            <Col span={8}>
              <Statistic title="匹配次数" value={rule.matchCount} />
            </Col>
            <Col span={8}>
              <Text strong>最后匹配: </Text>
              <Text>{rule.lastMatched || '从未匹配'}</Text>
            </Col>
            <Col span={8}>
              <Text strong>状态: </Text>
              <Badge status={rule.enabled ? 'success' : 'default'} text={rule.enabled ? '启用' : '禁用'} />
            </Col>
          </Row>
        </div>
      )
    })
  }

  const handleEditRule = (rule: ACLRule) => {
    setSelectedRule(rule)
    form.setFieldsValue(rule)
    setRuleModalVisible(true)
  }

  const handleDeleteRule = (rule: ACLRule) => {
    Modal.confirm({
      title: '删除规则',
      content: `确定要删除规则 "${rule.name}" 吗？此操作不可撤销。`,
      onOk: () => {
        setAclRules(prev => prev.filter(r => r.id !== rule.id))
        notification.success({
          message: '删除成功',
          description: `规则 "${rule.name}" 已被删除`
        })
      }
    })
  }

  const handleCreateRule = (values: any) => {
    const newRule: ACLRule = {
      id: `rule-${Date.now()}`,
      name: values.name,
      description: values.description,
      principal: values.principal,
      principalType: values.principalType,
      permission: values.permission,
      resource: values.resource,
      resourceType: values.resourceType,
      actions: values.actions,
      conditions: values.conditions,
      priority: values.priority,
      enabled: values.enabled,
      createdAt: new Date().toLocaleString('zh-CN'),
      updatedAt: new Date().toLocaleString('zh-CN'),
      createdBy: 'current-user',
      matchCount: 0
    }

    setAclRules(prev => [newRule, ...prev])
    setRuleModalVisible(false)
    form.resetFields()
    notification.success({
      message: '创建成功',
      description: `ACL规则 "${values.name}" 已创建`
    })
  }

  const handleTestRule = () => {
    notification.info({
      message: '规则测试',
      description: '正在测试规则匹配逻辑...'
    })
    
    setTimeout(() => {
      notification.success({
        message: '测试完成',
        description: '规则测试通过，逻辑正确'
      })
    }, 2000)
  }

  const refreshData = () => {
    setLoading(true)
    setTimeout(() => {
      notification.success({
        message: '刷新成功',
        description: 'ACL配置已更新'
      })
      setLoading(false)
    }, 1000)
  }

  const filteredRules = aclRules.filter(rule => {
    const matchesPrincipal = !filterPrincipal || rule.principal.toLowerCase().includes(filterPrincipal.toLowerCase())
    const matchesResource = !filterResource || rule.resource.toLowerCase().includes(filterResource.toLowerCase())
    return matchesPrincipal && matchesResource
  })

  const securityLevel = securityMetrics.accessDenied / securityMetrics.ruleMatches
  const getSecurityStatus = () => {
    if (securityLevel < 0.01) return { status: 'success', text: '安全', color: 'green' }
    if (securityLevel < 0.05) return { status: 'warning', text: '警告', color: 'orange' }
    return { status: 'error', text: '风险', color: 'red' }
  }

  const securityStatus = getSecurityStatus()

  return (
    <div style={{ padding: '24px', background: '#f5f5f5', minHeight: '100vh' }}>
      <div style={{ marginBottom: '24px' }}>
        <Title level={2}>
          <SafetyCertificateOutlined style={{ marginRight: '12px', color: '#1890ff' }} />
          ACL协议管理
        </Title>
        <Paragraph>
          基于角色和属性的访问控制列表管理，确保智能体间通信的安全性和权限控制
        </Paragraph>
      </div>

      {/* 安全状态告警 */}
      {securityMetrics.criticalViolations > 0 && (
        <Alert
          message="安全警告"
          description={`检测到 ${securityMetrics.criticalViolations} 个严重安全违规事件，最后一次违规时间: ${securityMetrics.lastViolation}`}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={() => setAuditDrawerVisible(true)}>
              查看详情
            </Button>
          }
          style={{ marginBottom: '24px' }}
        />
      )}

      {/* 安全指标概览 */}
      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="活跃规则"
              value={securityMetrics.activeRules}
              suffix={`/ ${securityMetrics.totalRules}`}
              valueStyle={{ color: '#1890ff' }}
              prefix={<SafetyCertificateOutlined />}
            />
          </Card>
        </Col>
        
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="规则匹配"
              value={securityMetrics.ruleMatches}
              valueStyle={{ color: '#52c41a' }}
              prefix={<CheckCircleOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                今日新增: +{Math.floor(securityMetrics.ruleMatches * 0.15)}
              </Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="访问拒绝"
              value={securityMetrics.accessDenied}
              valueStyle={{ color: '#ff4d4f' }}
              prefix={<CloseCircleOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                拒绝率: {((securityMetrics.accessDenied / securityMetrics.ruleMatches) * 100).toFixed(2)}%
              </Text>
            </div>
          </Card>
        </Col>
        
        <Col xs={12} sm={6}>
          <Card>
            <Statistic
              title="安全等级"
              value={securityStatus.text}
              valueStyle={{ color: securityStatus.color }}
              prefix={<SecurityScanOutlined />}
            />
            <div style={{ marginTop: '8px' }}>
              <Progress 
                percent={Math.max(20, 100 - securityLevel * 2000)} 
                size="small" 
                showInfo={false}
                status={securityStatus.status as any}
              />
            </div>
          </Card>
        </Col>
      </Row>

      {/* 主管理界面 */}
      <Card>
        <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Space>
            <Button 
              type="primary" 
              icon={<PlusOutlined />}
              onClick={() => {
                setSelectedRule(null)
                form.resetFields()
                setRuleModalVisible(true)
              }}
            >
              创建规则
            </Button>
            <Button icon={<ReloadOutlined />} loading={loading} onClick={refreshData}>
              刷新
            </Button>
          </Space>
          
          <Space>
            <Input
              placeholder="筛选主体"
              prefix={<SearchOutlined />}
              style={{ width: 150 }}
              value={filterPrincipal}
              onChange={(e) => setFilterPrincipal(e.target.value)}
            />
            <Input
              placeholder="筛选资源"
              prefix={<FilterOutlined />}
              style={{ width: 150 }}
              value={filterResource}
              onChange={(e) => setFilterResource(e.target.value)}
            />
            <Button icon={<AuditOutlined />} onClick={() => setAuditDrawerVisible(true)}>
              审计日志
            </Button>
            <Button icon={<SettingOutlined />}>ACL设置</Button>
          </Space>
        </div>

        <Tabs activeKey={activeTab} onChange={setActiveTab}>
          <TabPane tab="访问规则" key="rules" icon={<LockOutlined />}>
            <Table
              columns={ruleColumns}
              dataSource={filteredRules}
              rowKey="id"
              size="small"
              pagination={{ pageSize: 15 }}
              scroll={{ x: 1400 }}
            />
          </TabPane>
          
          <TabPane tab="用户组" key="groups" icon={<TeamOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Button type="primary" icon={<PlusOutlined />} onClick={() => setGroupModalVisible(true)}>
                创建用户组
              </Button>
            </div>
            <Table
              columns={groupColumns}
              dataSource={aclGroups}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </TabPane>
          
          <TabPane tab="角色管理" key="roles" icon={<UserOutlined />}>
            <div style={{ marginBottom: '16px' }}>
              <Button type="primary" icon={<PlusOutlined />} onClick={() => setRoleModalVisible(true)}>
                创建角色
              </Button>
            </div>
            <Table
              columns={roleColumns}
              dataSource={aclRoles}
              rowKey="id"
              size="small"
              pagination={false}
            />
          </TabPane>
          
          <TabPane tab="权限继承" key="inheritance" icon={<BranchesOutlined />}>
            <Card title="权限继承关系图" size="small">
              <div style={{ textAlign: 'center', padding: '40px 0' }}>
                <Text type="secondary">权限继承关系可视化视图开发中...</Text>
              </div>
            </Card>
          </TabPane>
        </Tabs>
      </Card>

      {/* 创建/编辑规则Modal */}
      <Modal
        title={selectedRule ? '编辑ACL规则' : '创建ACL规则'}
        visible={ruleModalVisible}
        onCancel={() => setRuleModalVisible(false)}
        width={800}
        footer={[
          <Button key="test" onClick={handleTestRule}>
            测试规则
          </Button>,
          <Button key="cancel" onClick={() => setRuleModalVisible(false)}>
            取消
          </Button>,
          <Button key="submit" type="primary" onClick={() => form.submit()}>
            {selectedRule ? '更新' : '创建'}
          </Button>
        ]}
      >
        <Form
          form={form}
          layout="vertical"
          onFinish={handleCreateRule}
        >
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="name" label="规则名称" rules={[{ required: true }]}>
                <Input placeholder="请输入规则名称" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="priority" label="优先级" rules={[{ required: true }]} initialValue={100}>
                <Input type="number" placeholder="数值越大优先级越高" />
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item name="description" label="规则描述">
            <TextArea rows={2} placeholder="请输入规则描述" />
          </Form.Item>
          
          <Row gutter={16}>
            <Col span={8}>
              <Form.Item name="permission" label="权限类型" rules={[{ required: true }]} initialValue="allow">
                <Radio.Group>
                  <Radio.Button value="allow">允许</Radio.Button>
                  <Radio.Button value="deny">拒绝</Radio.Button>
                </Radio.Group>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="principalType" label="主体类型" rules={[{ required: true }]} initialValue="agent">
                <Select>
                  <Option value="agent">智能体</Option>
                  <Option value="group">用户组</Option>
                  <Option value="role">角色</Option>
                </Select>
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="resourceType" label="资源类型" rules={[{ required: true }]} initialValue="subject">
                <Select>
                  <Option value="subject">主题</Option>
                  <Option value="stream">数据流</Option>
                  <Option value="queue">队列</Option>
                  <Option value="topic">话题</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="principal" label="主体" rules={[{ required: true }]}>
                <Input placeholder="例如: task-agent-*, admin-group" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="resource" label="资源" rules={[{ required: true }]}>
                <Input placeholder="例如: agents.tasks.>, system.admin.*" />
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item name="actions" label="允许的操作" rules={[{ required: true }]}>
            <Select mode="multiple" placeholder="选择操作类型">
              <Option value="subscribe">订阅</Option>
              <Option value="publish">发布</Option>
              <Option value="admin">管理</Option>
              <Option value="read">读取</Option>
              <Option value="write">写入</Option>
              <Option value="delete">删除</Option>
            </Select>
          </Form.Item>
          
          <Form.Item name="conditions" label="附加条件">
            <Select mode="tags" placeholder="输入条件表达式">
              <Option value="source_ip_whitelist">IP白名单</Option>
              <Option value="time_window">时间窗口</Option>
              <Option value="rate_limit">速率限制</Option>
              <Option value="has_clearance">权限许可</Option>
            </Select>
          </Form.Item>
          
          <Form.Item name="enabled" label="启用规则" valuePropName="checked" initialValue={true}>
            <Switch />
          </Form.Item>
        </Form>
      </Modal>

      {/* 审计日志抽屉 */}
      <Drawer
        title="ACL审计日志"
        placement="right"
        width={1000}
        visible={auditDrawerVisible}
        onClose={() => setAuditDrawerVisible(false)}
      >
        <Table
          columns={auditColumns}
          dataSource={auditLogs}
          rowKey="id"
          size="small"
          pagination={{ pageSize: 20 }}
          scroll={{ x: 800 }}
        />
      </Drawer>
    </div>
  )
}

export default ACLProtocolManagementPage