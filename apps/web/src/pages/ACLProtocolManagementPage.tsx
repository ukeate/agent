import React, { useState, useEffect } from 'react'
import {
import { logger } from '../utils/logger'
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
  Steps,
  Upload
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
  BranchesOutlined,
  DownloadOutlined,
  UploadOutlined
} from '@ant-design/icons'
import { aclService, type ACLRule, type SecurityMetrics } from '../services/aclService'

const { Title, Text, Paragraph } = Typography
const { Option } = Select
const { TextArea } = Input
const { TabPane } = Tabs
const { TreeNode } = Tree
const { Step } = Steps

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

  const [securityMetrics, setSecurityMetrics] = useState<SecurityMetrics | null>(null)
  const [aclRules, setAclRules] = useState<ACLRule[]>([])
  const [aclGroups, setAclGroups] = useState<ACLGroup[]>([])
  const [aclRoles, setAclRoles] = useState<ACLRole[]>([])
  const [auditLogs, setAuditLogs] = useState<ACLAuditLog[]>([])

  // 加载ACL规则
  const loadACLRules = async () => {
    setLoading(true)
    try {
      const rules = await aclService.listRules()
      setAclRules(rules)
    } catch (error) {
      logger.error('加载ACL规则失败:', error)
      setAclRules([])
    } finally {
      setLoading(false)
    }
  }

  // 加载安全指标
  const loadSecurityMetrics = async () => {
    try {
      const metrics = await aclService.getSecurityMetrics()
      setSecurityMetrics(metrics)
    } catch (error) {
      logger.error('加载安全指标失败:', error)
      setSecurityMetrics(null)
    }
  }

  useEffect(() => {
    loadACLRules()
    loadSecurityMetrics()

    // 设置定时刷新
    const interval = setInterval(() => {
      loadSecurityMetrics()
    }, 30000)

    return () => clearInterval(interval)
  }, [])

  const ruleColumns = [
    {
      title: '规则信息',
      key: 'info',
      width: 300,
      render: (record: ACLRule) => (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
            <Badge status={record.status === 'active' ? 'success' : 'default'} />
            <Text strong style={{ marginLeft: '8px', fontSize: '13px' }}>{record.name}</Text>
            <Tag 
              color={record.action === 'allow' ? 'green' : 'red'} 
              style={{ marginLeft: '8px', fontSize: '10px' }}
            >
              {record.action === 'allow' ? '允许' : '拒绝'}
            </Tag>
          </div>
          <Text type="secondary" style={{ fontSize: '11px' }}>
            {record.description}
          </Text>
        </div>
      )
    },
    {
      title: '源',
      dataIndex: 'source',
      key: 'source',
      width: 150,
      render: (source: string) => (
        <Text style={{ fontSize: '12px' }}>{source}</Text>
      )
    },
    {
      title: '目标',
      dataIndex: 'target',
      key: 'target',
      width: 200,
      render: (target: string) => (
        <Text code style={{ fontSize: '11px' }}>{target}</Text>
      )
    },
    {
      title: '条件',
      dataIndex: 'conditions',
      key: 'conditions',
      width: 120,
      render: (conditions: string[]) => (
        <div>
          {conditions.slice(0, 2).map((condition, index) => (
            <Tag key={index} style={{ fontSize: '10px', marginBottom: '2px' }}>
              {condition}
            </Tag>
          ))}
          {conditions.length > 2 && (
            <Text type="secondary" style={{ fontSize: '10px' }}>
              +{conditions.length - 2}
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
              <Text strong>创建时间: </Text>
              <Text>{new Date(rule.created_at).toLocaleString()}</Text>
            </Col>
            <Col span={12}>
              <Text strong>更新时间: </Text>
              <Text>{new Date(rule.updated_at).toLocaleString()}</Text>
            </Col>
          </Row>

          <Divider>权限配置</Divider>
          <Row gutter={[16, 8]}>
            <Col span={8}>
              <Text strong>权限类型: </Text>
              <Tag color={rule.action === 'allow' ? 'green' : 'red'}>
                {rule.action === 'allow' ? '允许' : '拒绝'}
              </Tag>
            </Col>
            <Col span={8}>
              <Text strong>优先级: </Text>
              <Tag>{rule.priority}</Tag>
            </Col>
            <Col span={8}>
              <Text strong>状态: </Text>
              <Badge status={rule.status === 'active' ? 'success' : 'default'} text={rule.status} />
            </Col>
            <Col span={12}>
              <Text strong>源: </Text>
              <Text code>{rule.source}</Text>
            </Col>
            <Col span={12}>
              <Text strong>目标: </Text>
              <Text code>{rule.target}</Text>
            </Col>
          </Row>

          <Divider>条件</Divider>
          <Row gutter={[16, 8]}>
            <Col span={24}>
              <Text strong>附加条件: </Text>
              <div style={{ marginTop: '4px' }}>
                {rule.conditions?.map((condition, index) => (
                  <Tag key={index} color="orange" style={{ marginBottom: '4px' }}>{condition}</Tag>
                )) || <Text type="secondary">无</Text>}
              </div>
            </Col>
          </Row>
        </div>
      )
    })
  }

  const handleEditRule = (rule: ACLRule) => {
    setSelectedRule(rule)
    form.setFieldsValue({
      name: rule.name,
      description: rule.description,
      source: rule.source,
      target: rule.target,
      action: rule.action,
      conditions: rule.conditions,
      priority: rule.priority,
      status: rule.status
    })
    setRuleModalVisible(true)
  }

  const handleDeleteRule = async (rule: ACLRule) => {
    Modal.confirm({
      title: '删除规则',
      content: `确定要删除规则 "${rule.name}" 吗？此操作不可撤销。`,
      onOk: async () => {
        try {
          await aclService.deleteRule(rule.id)
          await loadACLRules()
          notification.success({
            message: '删除成功',
            description: `规则 "${rule.name}" 已被删除`
          })
        } catch (error) {
          notification.error({
            message: '删除失败',
            description: '无法删除规则，请稍后重试'
          })
        }
      }
    })
  }

  const handleCreateRule = async (values: any) => {
    try {
      if (selectedRule) {
        await aclService.updateRule(selectedRule.id, values)
        notification.success({
          message: '更新成功',
          description: `ACL规则 "${values.name}" 已更新`
        })
      } else {
        await aclService.createRule(values)
        notification.success({
          message: '创建成功',
          description: `ACL规则 "${values.name}" 已创建`
        })
      }
      await loadACLRules()
      setRuleModalVisible(false)
      form.resetFields()
      setSelectedRule(null)
    } catch (error) {
      notification.error({
        message: selectedRule ? '更新失败' : '创建失败',
        description: '操作失败，请稍后重试'
      })
    }
  }

  const handleTestRule = async () => {
    const values = form.getFieldsValue()
    try {
      const result = await aclService.validateRule({
        name: values.name,
        description: values.description,
        source: values.source,
        target: values.target,
        action: values.action,
        conditions: values.conditions,
        priority: values.priority
      })
      
      if (result.is_valid) {
        notification.success({
          message: '测试通过',
          description: '规则配置有效'
        })
      } else {
        notification.warning({
          message: '测试失败',
          description: result.errors.join(', ')
        })
      }
    } catch (error) {
      notification.error({
        message: '测试失败',
        description: '无法测试规则，请稍后重试'
      })
    }
  }

  const refreshData = async () => {
    setLoading(true)
    try {
      await Promise.all([loadACLRules(), loadSecurityMetrics()])
      notification.success({
        message: '刷新成功',
        description: 'ACL配置已更新'
      })
    } catch (error) {
      notification.error({
        message: '刷新失败',
        description: '无法刷新数据，请稍后重试'
      })
    } finally {
      setLoading(false)
    }
  }

  const handleExportRules = async () => {
    try {
      const blob = await aclService.exportRules('json')
      const url = window.URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.style.display = 'none'
      a.href = url
      a.download = `acl-rules-${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(a)
      a.click()
      window.URL.revokeObjectURL(url)
      notification.success({
        message: '导出成功',
        description: 'ACL规则已导出到文件'
      })
    } catch (error) {
      notification.error({
        message: '导出失败',
        description: '无法导出ACL规则，请稍后重试'
      })
    }
  }

  const handleImportRules = (file: File) => {
    const importRules = async () => {
      try {
        const result = await aclService.importRules(file)
        notification.success({
          message: '导入成功',
          description: `成功导入 ${result.imported} 条规则，失败 ${result.failed} 条`
        })
        if (result.errors.length > 0) {
          notification.warning({
            message: '导入警告',
            description: result.errors.join(', ')
          })
        }
        await loadACLRules()
      } catch (error) {
        notification.error({
          message: '导入失败',
          description: '无法导入ACL规则，请检查文件格式'
        })
      }
    }
    importRules()
    return false // 阻止默认上传行为
  }

  const filteredRules = aclRules.filter(rule => {
    const matchesPrincipal = !filterPrincipal || rule.source.toLowerCase().includes(filterPrincipal.toLowerCase())
    const matchesResource = !filterResource || rule.target.toLowerCase().includes(filterResource.toLowerCase())
    return matchesPrincipal && matchesResource
  })

  const getSecurityStatus = () => {
    if (!securityMetrics) return { status: 'success', text: '安全', color: 'green' }
    
    const totalRequests = securityMetrics.allowed_requests + securityMetrics.blocked_requests
    const blockRate = totalRequests > 0 ? securityMetrics.blocked_requests / totalRequests : 0
    
    if (blockRate < 0.01) return { status: 'success', text: '安全', color: 'green' }
    if (blockRate < 0.05) return { status: 'warning', text: '警告', color: 'orange' }
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
      {securityMetrics && securityMetrics.violation_count > 0 && (
        <Alert
          message="安全警告"
          description={`检测到 ${securityMetrics.violation_count} 个严重安全违规事件，最后一次违规时间: ${securityMetrics.last_violation_time ? new Date(securityMetrics.last_violation_time).toLocaleString() : '未知'}`}
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
      {securityMetrics && (
        <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
          <Col xs={12} sm={6}>
            <Card>
              <Statistic
                title="活跃规则"
                value={securityMetrics.active_rules}
                suffix={`/ ${securityMetrics.total_rules}`}
                valueStyle={{ color: '#1890ff' }}
                prefix={<SafetyCertificateOutlined />}
              />
            </Card>
          </Col>
          
          <Col xs={12} sm={6}>
            <Card>
              <Statistic
                title="允许请求"
                value={securityMetrics.allowed_requests}
                valueStyle={{ color: '#52c41a' }}
                prefix={<CheckCircleOutlined />}
              />
            </Card>
          </Col>
          
          <Col xs={12} sm={6}>
            <Card>
              <Statistic
                title="拒绝请求"
                value={securityMetrics.blocked_requests}
                valueStyle={{ color: '#ff4d4f' }}
                prefix={<CloseCircleOutlined />}
              />
              <div style={{ marginTop: '8px' }}>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  拒绝率: {securityMetrics.allowed_requests + securityMetrics.blocked_requests > 0 
                    ? ((securityMetrics.blocked_requests / (securityMetrics.allowed_requests + securityMetrics.blocked_requests)) * 100).toFixed(2) 
                    : 0}%
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
                  percent={securityMetrics.allowed_requests + securityMetrics.blocked_requests > 0
                    ? Math.max(20, 100 - (securityMetrics.blocked_requests / (securityMetrics.allowed_requests + securityMetrics.blocked_requests)) * 2000)
                    : 100} 
                  size="small" 
                  showInfo={false}
                  status={securityStatus.status as any}
                />
              </div>
            </Card>
          </Col>
        </Row>
      )}

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
              placeholder="筛选源"
              prefix={<SearchOutlined />}
              style={{ width: 150 }}
              value={filterPrincipal}
              onChange={(e) => setFilterPrincipal(e.target.value)}
              name="acl-filter-principal"
            />
            <Input
              placeholder="筛选目标"
              prefix={<FilterOutlined />}
              style={{ width: 150 }}
              value={filterResource}
              onChange={(e) => setFilterResource(e.target.value)}
              name="acl-filter-resource"
            />
            <Button icon={<AuditOutlined />} onClick={() => setAuditDrawerVisible(true)}>
              审计日志
            </Button>
            <Button icon={<DownloadOutlined />} onClick={handleExportRules}>
              导出规则
            </Button>
            <Upload
              accept=".json,.yaml,.xml"
              beforeUpload={handleImportRules}
              showUploadList={false}
            >
              <Button icon={<UploadOutlined />}>
                导入规则
              </Button>
            </Upload>
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
              scroll={{ x: 1200 }}
              loading={loading}
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
        onCancel={() => {
          setRuleModalVisible(false)
          setSelectedRule(null)
          form.resetFields()
        }}
        width={800}
        footer={[
          <Button key="test" onClick={handleTestRule}>
            测试规则
          </Button>,
          <Button key="cancel" onClick={() => {
            setRuleModalVisible(false)
            setSelectedRule(null)
            form.resetFields()
          }}>
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
                <Input placeholder="请输入规则名称" name="acl-rule-name" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="priority" label="优先级" rules={[{ required: true }]} initialValue={100}>
                <Input type="number" placeholder="数值越大优先级越高" name="acl-rule-priority" />
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item name="description" label="规则描述">
            <TextArea rows={2} placeholder="请输入规则描述" name="acl-rule-description" />
          </Form.Item>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="action" label="权限类型" rules={[{ required: true }]} initialValue="allow">
                <Radio.Group>
                  <Radio.Button value="allow">允许</Radio.Button>
                  <Radio.Button value="deny">拒绝</Radio.Button>
                </Radio.Group>
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="status" label="状态" initialValue="active">
                <Select name="acl-rule-status">
                  <Option value="active">激活</Option>
                  <Option value="inactive">未激活</Option>
                </Select>
              </Form.Item>
            </Col>
          </Row>
          
          <Row gutter={16}>
            <Col span={12}>
              <Form.Item name="source" label="源" rules={[{ required: true }]}>
                <Input placeholder="例如: task-agent-*, admin-group" name="acl-rule-source" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item name="target" label="目标" rules={[{ required: true }]}>
                <Input placeholder="例如: agents.tasks.>, system.admin.*" name="acl-rule-target" />
              </Form.Item>
            </Col>
          </Row>
          
          <Form.Item name="conditions" label="附加条件">
            <Select mode="tags" placeholder="输入条件表达式" name="acl-rule-conditions">
              <Option value="source_ip_whitelist">IP白名单</Option>
              <Option value="time_window">时间窗口</Option>
              <Option value="rate_limit">速率限制</Option>
              <Option value="has_clearance">权限许可</Option>
            </Select>
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
