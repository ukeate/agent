import { buildApiUrl, apiFetch } from '../utils/apiBase'
import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Card,
  Tabs,
  Typography,
  Row,
  Col,
  Button,
  Tag,
  Statistic,
  Space,
  List,
  message,
} from 'antd'
import { logger } from '../utils/logger'
import {
  WarningOutlined,
  CheckCircleOutlined,
  BranchesOutlined,
  DatabaseOutlined,
  FileTextOutlined,
  ReloadOutlined,
  SettingOutlined,
  UserOutlined,
} from '@ant-design/icons'

const { Title, Text } = Typography
const { TabPane } = Tabs

interface ConflictRecord {
  conflict_id: string
  conflict_type: string
  description: string
  involved_tasks: string[]
  involved_agents: string[]
  timestamp: string
  resolved: boolean
  resolution_strategy?: string
  resolution_result?: Record<string, any>
}

interface ResolutionStrategy {
  strategy: string
  description: string
}

const ConflictResolutionPage: React.FC = () => {
  const [conflicts, setConflicts] = useState<ConflictRecord[]>([])
  const [selectedConflict, setSelectedConflict] =
    useState<ConflictRecord | null>(null)
  const [resolutionStrategies, setResolutionStrategies] = useState<
    ResolutionStrategy[]
  >([])
  const [loading, setLoading] = useState(true)

  const stats = useMemo(() => {
    const total = conflicts.length
    const resolved = conflicts.filter(c => c.resolved).length
    const unresolved = total - resolved
    const rate = total ? (resolved / total) * 100 : 0
    const typeDist: Record<string, number> = {}
    for (const c of conflicts)
      typeDist[c.conflict_type] = (typeDist[c.conflict_type] || 0) + 1
    return { total, resolved, unresolved, rate, typeDist }
  }, [conflicts])

  const getConflictTypeIcon = (type: string) => {
    switch (type) {
      case 'resource_conflict':
        return <DatabaseOutlined />
      case 'state_conflict':
        return <WarningOutlined />
      case 'assignment_conflict':
        return <UserOutlined />
      case 'dependency_conflict':
        return <BranchesOutlined />
      default:
        return <WarningOutlined />
    }
  }

  const getConflictTypeLabel = (type: string) => {
    switch (type) {
      case 'resource_conflict':
        return '资源冲突'
      case 'state_conflict':
        return '状态冲突'
      case 'assignment_conflict':
        return '分配冲突'
      case 'dependency_conflict':
        return '依赖冲突'
      default:
        return type
    }
  }

  const generateResolutionStrategies = (): ResolutionStrategy[] => [
    {
      strategy: 'priority_based',
      description: '优先级策略（保留高优先级任务，重分配其余任务）',
    },
    {
      strategy: 'resource_optimization',
      description: '资源优化（重分配资源占用最高的任务）',
    },
    {
      strategy: 'load_balancing',
      description: '负载均衡（重分配同一智能体上的冲突任务）',
    },
    {
      strategy: 'fairness',
      description: '公平性（按时间顺序重分配最近创建的任务）',
    },
  ]

  const fetchConflicts = useCallback(async () => {
    try {
      setLoading(true)
      const res = await apiFetch(
        buildApiUrl('/api/v1/distributed-task/conflicts')
      )
      const data = await res.json()
      const list: ConflictRecord[] = Array.isArray(data) ? data : []
      setConflicts(list)
      setSelectedConflict(prev =>
        prev ? list.find(c => c.conflict_id === prev.conflict_id) || null : prev
      )
    } catch (error) {
      logger.error('获取冲突列表失败:', error)
      message.error('获取冲突列表失败')
      setConflicts([])
      setSelectedConflict(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const resolveConflict = async (conflictId: string, strategy: string) => {
    try {
      const res = await apiFetch(
        buildApiUrl(
          `/api/v1/distributed-task/conflicts/resolve/${encodeURIComponent(conflictId)}?strategy=${encodeURIComponent(strategy)}`
        ),
        {
          method: 'POST',
        }
      )
      message.success('已提交冲突解决')
      await fetchConflicts()
    } catch (error) {
      logger.error('解决冲突失败:', error)
      message.error('解决冲突失败')
    }
  }

  const handleConflictSelect = (conflict: ConflictRecord) => {
    setSelectedConflict(conflict)
    setResolutionStrategies(generateResolutionStrategies())
  }

  useEffect(() => {
    fetchConflicts()
    const interval = setInterval(fetchConflicts, 10000)
    return () => clearInterval(interval)
  }, [fetchConflicts])

  return (
    <div style={{ padding: 24 }}>
      <div
        style={{
          marginBottom: 24,
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Title level={2} style={{ margin: 0 }}>
          <WarningOutlined style={{ marginRight: 12 }} />
          冲突解决中心
        </Title>
        <Space>
          <Button
            onClick={fetchConflicts}
            icon={<ReloadOutlined />}
            loading={loading}
          >
            刷新
          </Button>
        </Space>
      </div>

      <Row gutter={24} style={{ marginBottom: 24 }}>
        <Col span={6}>
          <Card>
            <Statistic
              title="总冲突数"
              value={stats.total}
              prefix={<WarningOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="未解决"
              value={stats.unresolved}
              prefix={<FileTextOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="已解决"
              value={stats.resolved}
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
        <Col span={6}>
          <Card>
            <Statistic
              title="解决率"
              value={stats.rate.toFixed(1)}
              suffix="%"
              prefix={<CheckCircleOutlined />}
            />
          </Card>
        </Col>
      </Row>

      <Row gutter={24}>
        <Col span={8}>
          <Card
            title="冲突列表"
            style={{ height: 600, overflow: 'auto' }}
            loading={loading}
          >
            <List
              dataSource={conflicts}
              locale={{ emptyText: '暂无冲突' }}
              renderItem={conflict => (
                <List.Item
                  onClick={() => handleConflictSelect(conflict)}
                  style={{
                    cursor: 'pointer',
                    backgroundColor:
                      selectedConflict?.conflict_id === conflict.conflict_id
                        ? '#f0f9ff'
                        : 'white',
                    border:
                      selectedConflict?.conflict_id === conflict.conflict_id
                        ? '2px solid #1890ff'
                        : '1px solid #f0f0f0',
                    borderRadius: 8,
                    marginBottom: 8,
                    padding: 12,
                  }}
                >
                  <div style={{ width: '100%' }}>
                    <div
                      style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        alignItems: 'center',
                        marginBottom: 8,
                      }}
                    >
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 8,
                        }}
                      >
                        {getConflictTypeIcon(conflict.conflict_type)}
                        <Text strong>
                          {getConflictTypeLabel(conflict.conflict_type)}
                        </Text>
                      </div>
                      <div style={{ display: 'flex', gap: 6 }}>
                        {conflict.resolved ? (
                          <Tag color="success">已解决</Tag>
                        ) : (
                          <Tag color="warning">未解决</Tag>
                        )}
                      </div>
                    </div>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {conflict.description}
                    </Text>
                    <br />
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      {new Date(conflict.timestamp).toLocaleString()}
                    </Text>
                  </div>
                </List.Item>
              )}
            />
          </Card>
        </Col>

        <Col span={16}>
          {selectedConflict ? (
            <Card>
              <Tabs defaultActiveKey="details">
                <TabPane
                  tab={
                    <span>
                      <FileTextOutlined /> 冲突详情
                    </span>
                  }
                  key="details"
                >
                  <Row gutter={16}>
                    <Col span={12}>
                      <Text strong>冲突ID:</Text>
                      <div
                        style={{
                          fontFamily: 'monospace',
                          backgroundColor: '#f5f5f5',
                          padding: 6,
                          borderRadius: 6,
                          marginTop: 6,
                        }}
                      >
                        {selectedConflict.conflict_id}
                      </div>
                    </Col>
                    <Col span={12}>
                      <Text strong>类型:</Text>
                      <div style={{ marginTop: 6 }}>
                        <Tag>{selectedConflict.conflict_type}</Tag>
                        {selectedConflict.resolved ? (
                          <Tag color="success">已解决</Tag>
                        ) : (
                          <Tag color="warning">未解决</Tag>
                        )}
                      </div>
                    </Col>
                  </Row>

                  <div style={{ marginTop: 16 }}>
                    <Text strong>描述:</Text>
                    <div style={{ marginTop: 6 }}>
                      {selectedConflict.description}
                    </div>
                  </div>

                  <div style={{ marginTop: 16 }}>
                    <Text strong>涉及任务:</Text>
                    <div style={{ marginTop: 6 }}>
                      {(selectedConflict.involved_tasks || []).length ? (
                        <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                          {JSON.stringify(
                            selectedConflict.involved_tasks,
                            null,
                            2
                          )}
                        </pre>
                      ) : (
                        <Text type="secondary">无</Text>
                      )}
                    </div>
                  </div>

                  <div style={{ marginTop: 16 }}>
                    <Text strong>涉及智能体:</Text>
                    <div style={{ marginTop: 6 }}>
                      {(selectedConflict.involved_agents || []).length ? (
                        <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                          {JSON.stringify(
                            selectedConflict.involved_agents,
                            null,
                            2
                          )}
                        </pre>
                      ) : (
                        <Text type="secondary">无</Text>
                      )}
                    </div>
                  </div>

                  <div style={{ marginTop: 16 }}>
                    <Text strong>解决信息:</Text>
                    <div style={{ marginTop: 6 }}>
                      {selectedConflict.resolution_strategy ? (
                        <Tag color="blue">
                          {selectedConflict.resolution_strategy}
                        </Tag>
                      ) : (
                        <Text type="secondary">暂无</Text>
                      )}
                      {selectedConflict.resolution_result ? (
                        <pre style={{ marginTop: 8, whiteSpace: 'pre-wrap' }}>
                          {JSON.stringify(
                            selectedConflict.resolution_result,
                            null,
                            2
                          )}
                        </pre>
                      ) : null}
                    </div>
                  </div>
                </TabPane>

                <TabPane
                  tab={
                    <span>
                      <SettingOutlined /> 解决方案
                    </span>
                  }
                  key="resolution"
                >
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {resolutionStrategies.map(s => (
                      <Card key={s.strategy} size="small">
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'center',
                            gap: 12,
                          }}
                        >
                          <div>
                            <Text strong>{s.strategy}</Text>
                            <div>
                              <Text type="secondary">{s.description}</Text>
                            </div>
                          </div>
                          <Button
                            type="primary"
                            disabled={selectedConflict.resolved}
                            onClick={() =>
                              resolveConflict(
                                selectedConflict.conflict_id,
                                s.strategy
                              )
                            }
                          >
                            应用
                          </Button>
                        </div>
                      </Card>
                    ))}
                  </Space>
                </TabPane>

                <TabPane
                  tab={
                    <span>
                      <FileTextOutlined /> 原始数据
                    </span>
                  }
                  key="raw"
                >
                  <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
                    {JSON.stringify(selectedConflict, null, 2)}
                  </pre>
                </TabPane>
              </Tabs>
            </Card>
          ) : (
            <Card
              style={{
                height: 400,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
              }}
            >
              <div style={{ textAlign: 'center', color: '#8c8c8c' }}>
                <WarningOutlined style={{ fontSize: 48, marginBottom: 16 }} />
                <div>请从左侧列表选择一个冲突</div>
              </div>
            </Card>
          )}
        </Col>
      </Row>
    </div>
  )
}

export default ConflictResolutionPage
