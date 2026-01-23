import React, { useState, useEffect } from 'react'
import { logger } from '../../utils/logger'
import {
  Card,
  Button,
  Row,
  Col,
  Progress,
  Alert,
  Table,
  Tag,
  Typography,
  Space,
  Statistic,
  Select,
  Divider,
  Modal,
  Descriptions,
  List,
} from 'antd'
import {
  DatabaseOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  WarningOutlined,
  BugOutlined,
  ToolOutlined,
  SyncOutlined,
} from '@ant-design/icons'
import { pgvectorApi } from '../../services/pgvectorApi'

const { Title, Text } = Typography
const { Option } = Select

interface IntegrityReport {
  table_name: string
  total_records: number
  valid_vectors: number
  null_vectors: number
  invalid_vectors: number
  dimension_mismatches: number
  zero_vectors: number
  integrity_rate: number
  issues: IntegrityIssue[]
  timestamp: string
}

interface IntegrityIssue {
  record_id: string
  issue: string
  details: string
}

interface RepairResult {
  strategy: string
  processed_issues: number
  successful_repairs: number
  failed_repairs: number
  removed_records: number
}

interface SystemSummary {
  total_records: number
  non_null_embeddings: number
  null_embeddings: number
  null_rate: number
  indexes: Array<{ name: string; definition: string }>
  validation_stats: any
}

const DataIntegrityPanel: React.FC = () => {
  const [integrityReport, setIntegrityReport] =
    useState<IntegrityReport | null>(null)
  const [systemSummary, setSystemSummary] = useState<SystemSummary | null>(null)
  const [repairResult, setRepairResult] = useState<RepairResult | null>(null)
  const [checking, setChecking] = useState(false)
  const [repairing, setRepairing] = useState(false)
  const [selectedTable, setSelectedTable] = useState('documents')
  const [repairStrategy, setRepairStrategy] = useState<
    'remove_invalid' | 'set_null'
  >('remove_invalid')
  const [repairModalVisible, setRepairModalVisible] = useState(false)

  useEffect(() => {
    fetchSystemSummary()
  }, [])

  const fetchSystemSummary = async () => {
    try {
      const summary = await pgvectorApi.getIntegritySummary(selectedTable)
      setSystemSummary(summary)
    } catch (error) {
      logger.error('获取系统摘要失败:', error)
    }
  }

  const handleIntegrityCheck = async () => {
    try {
      setChecking(true)
      const report = await pgvectorApi.validateVectorDataIntegrity({
        table_name: selectedTable,
        batch_size: 1000,
      })
      setIntegrityReport(report)
      await fetchSystemSummary()
    } catch (error) {
      logger.error('完整性检查失败:', error)
    } finally {
      setChecking(false)
    }
  }

  const handleRepairData = async () => {
    if (!integrityReport || integrityReport.issues.length === 0) return

    try {
      setRepairing(true)
      const result = await pgvectorApi.repairVectorData(
        integrityReport,
        repairStrategy
      )
      setRepairResult(result)
      setRepairModalVisible(false)

      // 重新检查完整性
      await handleIntegrityCheck()
    } catch (error) {
      logger.error('数据修复失败:', error)
    } finally {
      setRepairing(false)
    }
  }

  const getIntegrityColor = (rate: number) => {
    if (rate >= 0.95) return '#52c41a'
    if (rate >= 0.8) return '#faad14'
    return '#f5222d'
  }

  const getIssueColor = (issue: string) => {
    const colorMap: { [key: string]: string } = {
      null: 'default',
      invalid: 'red',
      dimension_mismatch: 'orange',
      zero_vector: 'yellow',
    }
    return colorMap[issue] || 'default'
  }

  const getIssueText = (issue: string) => {
    const textMap: { [key: string]: string } = {
      null: '空向量',
      invalid: '无效向量',
      dimension_mismatch: '维度不匹配',
      zero_vector: '零向量',
    }
    return textMap[issue] || issue
  }

  const issueColumns = [
    {
      title: '记录ID',
      dataIndex: 'record_id',
      key: 'record_id',
      width: 120,
      render: (id: string) => <Text code>{id.substring(0, 8)}...</Text>,
    },
    {
      title: '问题类型',
      dataIndex: 'issue',
      key: 'issue',
      width: 120,
      render: (issue: string) => (
        <Tag color={getIssueColor(issue)}>{getIssueText(issue)}</Tag>
      ),
    },
    {
      title: '详细信息',
      dataIndex: 'details',
      key: 'details',
      ellipsis: true,
    },
  ]

  const indexColumns = [
    {
      title: '索引名称',
      dataIndex: 'name',
      key: 'name',
    },
    {
      title: '索引定义',
      dataIndex: 'definition',
      key: 'definition',
      ellipsis: true,
    },
  ]

  return (
    <div>
      {/* 控制面板 */}
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row justify="space-between" align="middle">
          <Col>
            <Space>
              <Text>表名:</Text>
              <Select
                value={selectedTable}
                onChange={setSelectedTable}
                style={{ width: 200 }}
              >
                <Option value="documents">documents</Option>
                <Option value="knowledge_items">knowledge_items</Option>
              </Select>

              <Button
                type="primary"
                icon={<BugOutlined />}
                onClick={handleIntegrityCheck}
                loading={checking}
              >
                检查数据完整性
              </Button>
            </Space>
          </Col>
          <Col>
            <Button onClick={fetchSystemSummary} icon={<SyncOutlined />}>
              刷新概览
            </Button>
          </Col>
        </Row>
      </Card>

      {/* 系统概览 */}
      <Card title="系统数据概览" style={{ marginBottom: 16 }}>
        {systemSummary ? (
          <Row gutter={16}>
            <Col span={6}>
              <Statistic
                title="总记录数"
                value={systemSummary.total_records}
                prefix={<DatabaseOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="有效向量"
                value={systemSummary.non_null_embeddings}
                valueStyle={{ color: '#3f8600' }}
                prefix={<CheckCircleOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="空向量"
                value={systemSummary.null_embeddings}
                valueStyle={{ color: '#cf1322' }}
                prefix={<ExclamationCircleOutlined />}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="空向量率"
                value={systemSummary.null_rate * 100}
                precision={1}
                suffix="%"
                valueStyle={{
                  color: systemSummary.null_rate < 0.05 ? '#3f8600' : '#cf1322',
                }}
                prefix={<WarningOutlined />}
              />
            </Col>
          </Row>
        ) : (
          <div style={{ textAlign: 'center', padding: '20px' }}>
            <Text type="secondary">选择表并点击刷新概览</Text>
          </div>
        )}
      </Card>

      {/* 完整性检查结果 */}
      {integrityReport && (
        <Card
          title={`完整性检查结果 - ${integrityReport.table_name}`}
          style={{ marginBottom: 16 }}
          extra={
            <Space>
              <Text type="secondary">
                {new Date(integrityReport.timestamp).toLocaleString()}
              </Text>
              {integrityReport.issues.length > 0 && (
                <Button
                  type="primary"
                  danger
                  icon={<ToolOutlined />}
                  onClick={() => setRepairModalVisible(true)}
                  disabled={repairing}
                >
                  修复数据
                </Button>
              )}
            </Space>
          }
        >
          {/* 完整性概览 */}
          <Row gutter={16} style={{ marginBottom: 24 }}>
            <Col span={8}>
              <Card size="small">
                <Statistic
                  title="完整性率"
                  value={integrityReport.integrity_rate * 100}
                  precision={1}
                  suffix="%"
                  valueStyle={{
                    color: getIntegrityColor(integrityReport.integrity_rate),
                  }}
                />
                <Progress
                  percent={integrityReport.integrity_rate * 100}
                  strokeColor={getIntegrityColor(
                    integrityReport.integrity_rate
                  )}
                  size="small"
                />
              </Card>
            </Col>

            <Col span={16}>
              <Row gutter={8}>
                <Col span={6}>
                  <Statistic
                    title="总记录"
                    value={integrityReport.total_records}
                    size="small"
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="有效向量"
                    value={integrityReport.valid_vectors}
                    valueStyle={{ color: '#52c41a' }}
                    size="small"
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="空向量"
                    value={integrityReport.null_vectors}
                    valueStyle={{ color: '#faad14' }}
                    size="small"
                  />
                </Col>
                <Col span={6}>
                  <Statistic
                    title="无效向量"
                    value={integrityReport.invalid_vectors}
                    valueStyle={{ color: '#f5222d' }}
                    size="small"
                  />
                </Col>
              </Row>

              <Row gutter={8} style={{ marginTop: 16 }}>
                <Col span={12}>
                  <Statistic
                    title="维度不匹配"
                    value={integrityReport.dimension_mismatches}
                    valueStyle={{ color: '#fa8c16' }}
                    size="small"
                  />
                </Col>
                <Col span={12}>
                  <Statistic
                    title="零向量"
                    value={integrityReport.zero_vectors}
                    valueStyle={{ color: '#faad14' }}
                    size="small"
                  />
                </Col>
              </Row>
            </Col>
          </Row>

          {/* 问题列表 */}
          {integrityReport.issues.length > 0 ? (
            <>
              <Divider />
              <Title level={5}>
                发现的问题 ({integrityReport.issues.length}个)
              </Title>
              <Table
                dataSource={integrityReport.issues.map((issue, index) => ({
                  ...issue,
                  key: index,
                }))}
                columns={issueColumns}
                size="small"
                pagination={{ pageSize: 10, showSizeChanger: false }}
                scroll={{ y: 300 }}
              />
            </>
          ) : (
            <Alert
              message="数据完整性良好"
              description="未发现向量数据问题"
              type="success"
              showIcon
            />
          )}
        </Card>
      )}

      {/* 索引信息 */}
      {systemSummary?.indexes && systemSummary.indexes.length > 0 && (
        <Card title="向量索引信息">
          <Table
            dataSource={systemSummary.indexes.map((index, i) => ({
              ...index,
              key: i,
            }))}
            columns={indexColumns}
            size="small"
            pagination={false}
          />
        </Card>
      )}

      {/* 修复确认模态框 */}
      <Modal
        title="数据修复确认"
        open={repairModalVisible}
        onCancel={() => setRepairModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setRepairModalVisible(false)}>
            取消
          </Button>,
          <Button
            key="repair"
            type="primary"
            danger
            loading={repairing}
            onClick={handleRepairData}
          >
            执行修复
          </Button>,
        ]}
      >
        <Alert
          message="警告：数据修复操作不可逆"
          description="请确认修复策略后再执行，建议先备份数据"
          type="warning"
          showIcon
          style={{ marginBottom: 16 }}
        />

        <Descriptions column={1} bordered size="small">
          <Descriptions.Item label="表名">
            {integrityReport?.table_name}
          </Descriptions.Item>
          <Descriptions.Item label="发现问题">
            {integrityReport?.issues.length}个
          </Descriptions.Item>
          <Descriptions.Item label="修复策略">
            <Select
              value={repairStrategy}
              onChange={setRepairStrategy}
              style={{ width: 200 }}
            >
              <Option value="remove_invalid">删除无效记录</Option>
              <Option value="set_null">设置为NULL</Option>
            </Select>
          </Descriptions.Item>
        </Descriptions>

        <div style={{ marginTop: 16 }}>
          <Title level={5}>修复策略说明：</Title>
          <List size="small">
            <List.Item>
              <Text strong>删除无效记录：</Text>
              <Text>将无效、空值、零向量的记录从数据库中删除</Text>
            </List.Item>
            <List.Item>
              <Text strong>设置为NULL：</Text>
              <Text>将有问题的向量字段设置为NULL，保留记录</Text>
            </List.Item>
          </List>
        </div>
      </Modal>

      {/* 修复结果 */}
      {repairResult && (
        <Card title="修复结果" style={{ marginTop: 16 }}>
          <Row gutter={16}>
            <Col span={6}>
              <Statistic
                title="处理问题数"
                value={repairResult.processed_issues}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="修复成功"
                value={repairResult.successful_repairs}
                valueStyle={{ color: '#52c41a' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="修复失败"
                value={repairResult.failed_repairs}
                valueStyle={{ color: '#f5222d' }}
              />
            </Col>
            <Col span={6}>
              <Statistic
                title="删除记录"
                value={repairResult.removed_records}
                valueStyle={{ color: '#faad14' }}
              />
            </Col>
          </Row>
        </Card>
      )}
    </div>
  )
}

export default DataIntegrityPanel
