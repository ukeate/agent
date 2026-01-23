// 安全管理页面

import React, { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Form,
  DatePicker,
  Select,
  Table,
  Tag,
  Space,
  notification,
  Modal,
  Input,
} from 'antd'
import { logger } from '../utils/logger'
import {
  FileText,
  CheckCircle,
  XCircle,
  Clock,
  Download,
  Shield,
  AlertTriangle,
  Eye,
} from 'lucide-react'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import { SecurityDashboard } from '../components/security/SecurityDashboard'
import { APIKeyManager } from '../components/security/APIKeyManager'
import { ToolPermissions } from '../components/security/ToolPermissions'
import { SecurityAlerts } from '../components/security/SecurityAlerts'
import { securityApi } from '../services/securityApi'
import dayjs from 'dayjs'

const SecurityPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState('dashboard')
  const [complianceData, setComplianceData] = useState<any>(null)
  const [pendingApprovals, setPendingApprovals] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [approvalModalVisible, setApprovalModalVisible] = useState(false)
  const [selectedApproval, setSelectedApproval] = useState<any>(null)
  const [form] = Form.useForm()

  // 加载待审批请求
  const loadPendingApprovals = async () => {
    try {
      setLoading(true)
      const approvals = await securityApi.getPendingApprovals()
      setPendingApprovals(approvals)
    } catch (error) {
      logger.error('加载待审批请求失败:', error)
      notification.error({
        message: '加载失败',
        description: '无法加载待审批请求',
      })
    } finally {
      setLoading(false)
    }
  }

  // 生成合规报告
  const generateComplianceReport = async (values: any) => {
    try {
      setLoading(true)
      const report = await securityApi.getComplianceReport(
        values.startDate.format('YYYY-MM-DD'),
        values.endDate.format('YYYY-MM-DD'),
        values.reportType
      )
      setComplianceData(report)
      notification.success({
        message: '报告生成成功',
        description: '合规报告已生成',
      })
    } catch (error) {
      logger.error('生成合规报告失败:', error)
      notification.error({
        message: '生成失败',
        description: '无法生成合规报告',
      })
    } finally {
      setLoading(false)
    }
  }

  // 处理审批
  const handleApproval = async (
    requestId: string,
    approved: boolean,
    reason?: string
  ) => {
    try {
      await securityApi.approveToolCall(requestId, approved, reason)
      notification.success({
        message: '处理成功',
        description: `请求已${approved ? '批准' : '拒绝'}`,
      })
      loadPendingApprovals() // 重新加载列表
      setApprovalModalVisible(false)
      setSelectedApproval(null)
    } catch (error) {
      logger.error('处理审批失败:', error)
      notification.error({
        message: '处理失败',
        description: '无法处理审批请求',
      })
    }
  }

  useEffect(() => {
    if (activeTab === 'approvals') {
      loadPendingApprovals()
    }
  }, [activeTab])

  return (
    <div className="container mx-auto p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">安全管理中心</h1>
        <p className="text-gray-600 mt-2">
          管理API安全、工具权限、安全告警和访问控制
        </p>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-6">
          <TabsTrigger value="dashboard">安全概览</TabsTrigger>
          <TabsTrigger value="api-keys">API密钥</TabsTrigger>
          <TabsTrigger value="tool-permissions">工具权限</TabsTrigger>
          <TabsTrigger value="alerts">安全告警</TabsTrigger>
          <TabsTrigger value="compliance">合规报告</TabsTrigger>
          <TabsTrigger value="approvals">待审批</TabsTrigger>
        </TabsList>

        <TabsContent value="dashboard" className="mt-6">
          <SecurityDashboard />
        </TabsContent>

        <TabsContent value="api-keys" className="mt-6">
          <APIKeyManager />
        </TabsContent>

        <TabsContent value="tool-permissions" className="mt-6">
          <ToolPermissions />
        </TabsContent>

        <TabsContent value="alerts" className="mt-6">
          <SecurityAlerts />
        </TabsContent>

        <TabsContent value="compliance" className="mt-6">
          <Card>
            <Card.Header>
              <div className="flex items-center space-x-2">
                <FileText className="h-5 w-5" />
                <span className="text-lg font-semibold">合规报告生成</span>
              </div>
            </Card.Header>
            <Card.Body>
              <Form
                form={form}
                layout="inline"
                onFinish={generateComplianceReport}
                style={{ marginBottom: '20px' }}
              >
                <Form.Item
                  name="startDate"
                  label="开始日期"
                  rules={[{ required: true, message: '请选择开始日期' }]}
                >
                  <DatePicker />
                </Form.Item>
                <Form.Item
                  name="endDate"
                  label="结束日期"
                  rules={[{ required: true, message: '请选择结束日期' }]}
                >
                  <DatePicker />
                </Form.Item>
                <Form.Item name="reportType" label="报告类型">
                  <Select placeholder="选择报告类型" style={{ width: 150 }}>
                    <Select.Option value="security">安全报告</Select.Option>
                    <Select.Option value="privacy">隐私报告</Select.Option>
                    <Select.Option value="audit">审计报告</Select.Option>
                    <Select.Option value="comprehensive">
                      综合报告
                    </Select.Option>
                  </Select>
                </Form.Item>
                <Form.Item>
                  <Button type="primary" htmlType="submit" loading={loading}>
                    生成报告
                  </Button>
                </Form.Item>
              </Form>

              {complianceData && (
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-lg font-semibold">合规报告结果</h3>
                    <Button
                      icon={<Download />}
                      onClick={() => {
                        // 下载报告逻辑
                        const blob = new Blob(
                          [JSON.stringify(complianceData, null, 2)],
                          { type: 'application/json' }
                        )
                        const url = URL.createObjectURL(blob)
                        const link = document.createElement('a')
                        link.href = url
                        link.download = `compliance-report-${dayjs().format('YYYY-MM-DD')}.json`
                        document.body.appendChild(link)
                        link.click()
                        document.body.removeChild(link)
                        URL.revokeObjectURL(url)
                      }}
                    >
                      下载报告
                    </Button>
                  </div>
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <pre className="whitespace-pre-wrap text-sm">
                      {JSON.stringify(complianceData, null, 2)}
                    </pre>
                  </div>
                </div>
              )}
            </Card.Body>
          </Card>
        </TabsContent>

        <TabsContent value="approvals" className="mt-6">
          <Card>
            <Card.Header>
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <Clock className="h-5 w-5" />
                  <span className="text-lg font-semibold">
                    待审批的工具调用请求
                  </span>
                </div>
                <Button onClick={loadPendingApprovals} loading={loading}>
                  刷新
                </Button>
              </div>
            </Card.Header>
            <Card.Body>
              <Table
                dataSource={pendingApprovals}
                loading={loading}
                rowKey="request_id"
                columns={[
                  {
                    title: '请求ID',
                    dataIndex: 'request_id',
                    key: 'request_id',
                    width: 200,
                    ellipsis: true,
                  },
                  {
                    title: '工具名称',
                    dataIndex: 'tool_name',
                    key: 'tool_name',
                    width: 150,
                  },
                  {
                    title: '用户',
                    dataIndex: 'user_id',
                    key: 'user_id',
                    width: 120,
                  },
                  {
                    title: '风险级别',
                    dataIndex: 'risk_level',
                    key: 'risk_level',
                    width: 100,
                    render: (risk: string) => (
                      <Tag
                        color={
                          risk === 'critical'
                            ? 'red'
                            : risk === 'high'
                              ? 'orange'
                              : risk === 'medium'
                                ? 'yellow'
                                : 'green'
                        }
                      >
                        {risk === 'critical'
                          ? '严重'
                          : risk === 'high'
                            ? '高'
                            : risk === 'medium'
                              ? '中'
                              : '低'}
                      </Tag>
                    ),
                  },
                  {
                    title: '请求时间',
                    dataIndex: 'requested_at',
                    key: 'requested_at',
                    width: 180,
                    render: (time: string) =>
                      dayjs(time).format('YYYY-MM-DD HH:mm:ss'),
                  },
                  {
                    title: '操作',
                    key: 'actions',
                    width: 200,
                    render: (_: any, record: any) => (
                      <Space>
                        <Button
                          size="small"
                          icon={<Eye />}
                          onClick={() => {
                            setSelectedApproval(record)
                            setApprovalModalVisible(true)
                          }}
                        >
                          查看详情
                        </Button>
                        <Button
                          size="small"
                          type="primary"
                          icon={<CheckCircle />}
                          onClick={() =>
                            handleApproval(record.request_id, true)
                          }
                        >
                          批准
                        </Button>
                        <Button
                          size="small"
                          danger
                          icon={<XCircle />}
                          onClick={() =>
                            handleApproval(record.request_id, false)
                          }
                        >
                          拒绝
                        </Button>
                      </Space>
                    ),
                  },
                ]}
                pagination={{
                  pageSize: 10,
                  showSizeChanger: true,
                  showTotal: total => `共 ${total} 条请求`,
                }}
              />
            </Card.Body>
          </Card>
        </TabsContent>
      </Tabs>

      {/* 审批详情模态框 */}
      <Modal
        title="审批请求详情"
        visible={approvalModalVisible}
        onCancel={() => {
          setApprovalModalVisible(false)
          setSelectedApproval(null)
        }}
        footer={[
          <Button key="cancel" onClick={() => setApprovalModalVisible(false)}>
            取消
          </Button>,
          <Button
            key="reject"
            danger
            onClick={() =>
              selectedApproval &&
              handleApproval(selectedApproval.request_id, false)
            }
          >
            拒绝
          </Button>,
          <Button
            key="approve"
            type="primary"
            onClick={() =>
              selectedApproval &&
              handleApproval(selectedApproval.request_id, true)
            }
          >
            批准
          </Button>,
        ]}
        width={800}
      >
        {selectedApproval && (
          <div>
            <div className="space-y-4">
              <div>
                <span className="font-semibold">请求ID: </span>
                <span>{selectedApproval.request_id}</span>
              </div>
              <div>
                <span className="font-semibold">工具名称: </span>
                <span>{selectedApproval.tool_name}</span>
              </div>
              <div>
                <span className="font-semibold">用户: </span>
                <span>{selectedApproval.user_id}</span>
              </div>
              <div>
                <span className="font-semibold">风险级别: </span>
                <Tag
                  color={
                    selectedApproval.risk_level === 'critical'
                      ? 'red'
                      : selectedApproval.risk_level === 'high'
                        ? 'orange'
                        : selectedApproval.risk_level === 'medium'
                          ? 'yellow'
                          : 'green'
                  }
                >
                  {selectedApproval.risk_level === 'critical'
                    ? '严重'
                    : selectedApproval.risk_level === 'high'
                      ? '高'
                      : selectedApproval.risk_level === 'medium'
                        ? '中'
                        : '低'}
                </Tag>
              </div>
              <div>
                <span className="font-semibold">请求参数: </span>
                <pre className="mt-2 p-3 bg-gray-100 rounded text-sm overflow-auto">
                  {JSON.stringify(selectedApproval.parameters, null, 2)}
                </pre>
              </div>
              <div>
                <span className="font-semibold">请求原因: </span>
                <p className="mt-1">
                  {selectedApproval.justification || '未提供'}
                </p>
              </div>
            </div>
          </div>
        )}
      </Modal>
    </div>
  )
}

export default SecurityPage
