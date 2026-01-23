import React, { useState, useEffect } from 'react'
import {
  Card,
  Button,
  Input,
  Select,
  Alert,
  Badge,
  Tabs,
  Space,
  Typography,
  Row,
  Col,
  Form,
  message,
  Modal,
  Checkbox,
} from 'antd'
import {
  ReloadOutlined,
  PlusOutlined,
  EditOutlined,
  DeleteOutlined,
} from '@ant-design/icons'

import { logger } from '../utils/logger'
const { Option } = Select
const { Title, Text } = Typography
import {
  layeredExperimentsService,
  type ExperimentLayer,
  LayerType,
  ConflictResolution,
} from '../services/layeredExperimentsService'

const LayeredExperimentsManagementPage: React.FC = () => {
  const [layers, setLayers] = useState<ExperimentLayer[]>([])
  const [conflicts, setConflicts] = useState<any[]>([])
  const [metrics, setMetrics] = useState<any>(null)
  const [activeTab, setActiveTab] = useState('layers')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  const [layerForm, setLayerForm] = useState<Partial<ExperimentLayer>>({
    layer_id: '',
    name: '',
    description: '',
    layer_type: LayerType.MUTUALLY_EXCLUSIVE,
    traffic_percentage: 100,
    priority: 0,
    is_active: true,
    conflict_resolution: ConflictResolution.PRIORITY_BASED,
  })

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    await Promise.all([loadLayers(), loadConflicts(), loadMetrics()])
  }

  const loadLayers = async () => {
    try {
      const response = await layeredExperimentsService.listLayers(false)
      if (response.layers) {
        setLayers(response.layers)
      }
    } catch (err) {
      logger.warn('加载实验层失败:', err)
    }
  }

  const loadConflicts = async () => {
    try {
      const conflictsData = await layeredExperimentsService.getConflicts()
      setConflicts(conflictsData)
    } catch (err) {
      logger.warn('加载冲突失败:', err)
    }
  }

  const loadMetrics = async () => {
    try {
      const metricsData = await layeredExperimentsService.getSystemMetrics()
      setMetrics(metricsData)
    } catch (err) {
      logger.warn('加载指标失败:', err)
    }
  }

  const showConflictDetails = (conflict: any) => {
    Modal.info({
      title: '冲突详情',
      width: 720,
      content: (
        <pre
          style={{ margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word' }}
        >
          {JSON.stringify(conflict, null, 2)}
        </pre>
      ),
    })
  }

  const resolveConflict = (conflict: any) => {
    Modal.confirm({
      title: '解决冲突',
      okText: '去创建实验层',
      cancelText: '取消',
      content: (
        <div>
          <div style={{ marginBottom: 8 }}>
            <Text strong>冲突：</Text>
            <Text>{conflict?.description || '未知'}</Text>
          </div>
          <div>
            <Text type="secondary">
              建议：创建/启用实验层并分配合理流量，然后刷新数据。
            </Text>
          </div>
        </div>
      ),
      onOk: () => {
        setActiveTab('create')
      },
    })
  }

  const handleCreateLayer = async () => {
    if (!layerForm.layer_id || !layerForm.name) {
      setError('请填写必填字段')
      return
    }

    try {
      setLoading(true)
      setError(null)

      const response = await layeredExperimentsService.createLayer(layerForm)
      setSuccess('实验层创建成功')

      // 重置表单
      setLayerForm({
        layer_id: '',
        name: '',
        description: '',
        layer_type: LayerType.MUTUALLY_EXCLUSIVE,
        traffic_percentage: 100,
        priority: 0,
        is_active: true,
        conflict_resolution: ConflictResolution.PRIORITY_BASED,
      })

      await loadData()
    } catch (err) {
      setError('创建实验层失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const handleUpdateLayer = async (
    layerId: string,
    updates: Partial<ExperimentLayer>
  ) => {
    try {
      setLoading(true)
      setError(null)

      await layeredExperimentsService.updateLayer(layerId, updates)
      setSuccess('实验层更新成功')

      await loadData()
    } catch (err) {
      setError('更新实验层失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteLayer = async (layerId: string) => {
    if (!confirm('确定要删除此实验层吗？')) return

    try {
      setLoading(true)
      setError(null)

      await layeredExperimentsService.deleteLayer(layerId)
      setSuccess('实验层删除成功')

      await loadData()
    } catch (err) {
      setError('删除实验层失败: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const getLayerTypeColor = (type: LayerType) => {
    switch (type) {
      case LayerType.MUTUALLY_EXCLUSIVE:
        return 'red'
      case LayerType.ORTHOGONAL:
        return 'blue'
      case LayerType.HOLDBACK:
        return 'green'
      default:
        return 'default'
    }
  }

  const getLayerTypeText = (type: LayerType) => {
    switch (type) {
      case LayerType.MUTUALLY_EXCLUSIVE:
        return '互斥'
      case LayerType.ORTHOGONAL:
        return '正交'
      case LayerType.HOLDBACK:
        return '保留'
      default:
        return '未知'
    }
  }

  return (
    <div style={{ padding: '24px' }}>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '24px',
        }}
      >
        <Title level={2}>分层实验管理</Title>
        <Button icon={<ReloadOutlined />} onClick={loadData} loading={loading}>
          刷新数据
        </Button>
      </div>

      {error && (
        <Alert
          message="错误"
          description={error}
          type="error"
          closable
          style={{ marginBottom: 16 }}
          onClose={() => setError(null)}
        />
      )}

      {success && (
        <Alert
          message="成功"
          description={success}
          type="success"
          closable
          style={{ marginBottom: 16 }}
          onClose={() => setSuccess(null)}
        />
      )}

      <Row gutter={[16, 16]} style={{ marginBottom: '24px' }}>
        <Col span={6}>
          <Card title="总实验层">
            <div
              style={{
                fontSize: '24px',
                fontWeight: 'bold',
                marginBottom: '8px',
              }}
            >
              {layers.length}
            </div>
            <Text type="secondary">
              活跃: {layers.filter(l => l.is_active).length}
            </Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card title="流量分配">
            <div
              style={{
                fontSize: '24px',
                fontWeight: 'bold',
                marginBottom: '8px',
              }}
            >
              {layers.reduce(
                (sum, l) => sum + (l.is_active ? l.traffic_percentage : 0),
                0
              )}
              %
            </div>
            <Text type="secondary">总分配流量</Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card title="冲突数量">
            <div
              style={{
                fontSize: '24px',
                fontWeight: 'bold',
                marginBottom: '8px',
                color: conflicts.length > 0 ? '#ff4d4f' : '#52c41a',
              }}
            >
              {conflicts.length}
            </div>
            <Text type="secondary">待解决冲突</Text>
          </Card>
        </Col>
        <Col span={6}>
          <Card title="系统状态">
            <Badge
              color={layers.some(l => l.is_active) ? 'green' : 'red'}
              text={layers.some(l => l.is_active) ? '运行中' : '已停止'}
            />
            <div style={{ marginTop: '8px' }}>
              <Text type="secondary">分层实验系统</Text>
            </div>
          </Card>
        </Col>
      </Row>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        type="card"
        items={[
          {
            key: 'layers',
            label: '实验层管理',
            children: (
              <Card title="实验层列表">
                {layers.length > 0 ? (
                  <Space
                    direction="vertical"
                    size="large"
                    style={{ width: '100%' }}
                  >
                    {layers.map(layer => (
                      <Card
                        key={layer.layer_id}
                        style={{
                          borderLeft: `4px solid ${layer.is_active ? '#52c41a' : '#d9d9d9'}`,
                        }}
                      >
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            alignItems: 'flex-start',
                          }}
                        >
                          <div style={{ flex: 1 }}>
                            <Space wrap style={{ marginBottom: '8px' }}>
                              <Title level={5} style={{ margin: 0 }}>
                                {layer.name}
                              </Title>
                              <Badge
                                color={getLayerTypeColor(layer.layer_type)}
                                text={getLayerTypeText(layer.layer_type)}
                              />
                              <Badge
                                color={layer.is_active ? 'green' : 'default'}
                                text={layer.is_active ? '活跃' : '非活跃'}
                              />
                            </Space>
                            <Text
                              type="secondary"
                              style={{ display: 'block', marginBottom: '12px' }}
                            >
                              {layer.description}
                            </Text>
                            <Row gutter={[16, 8]}>
                              <Col span={12}>
                                <Text>ID: {layer.layer_id}</Text>
                              </Col>
                              <Col span={12}>
                                <Text>优先级: {layer.priority}</Text>
                              </Col>
                              <Col span={12}>
                                <Text>流量: {layer.traffic_percentage}%</Text>
                              </Col>
                              <Col span={12}>
                                <Text>
                                  冲突策略: {layer.conflict_resolution}
                                </Text>
                              </Col>
                            </Row>
                            <Text
                              type="secondary"
                              style={{
                                fontSize: '12px',
                                display: 'block',
                                marginTop: '8px',
                              }}
                            >
                              创建时间:{' '}
                              {new Date(layer.created_at).toLocaleString(
                                'zh-CN'
                              )}
                            </Text>
                          </div>
                          <Space>
                            <Button
                              size="small"
                              onClick={() =>
                                handleUpdateLayer(layer.layer_id, {
                                  is_active: !layer.is_active,
                                })
                              }
                            >
                              {layer.is_active ? '停用' : '启用'}
                            </Button>
                            <Button
                              size="small"
                              danger
                              onClick={() => handleDeleteLayer(layer.layer_id)}
                            >
                              删除
                            </Button>
                          </Space>
                        </div>
                      </Card>
                    ))}
                  </Space>
                ) : (
                  <div
                    style={{
                      textAlign: 'center',
                      padding: '40px',
                      color: '#999',
                    }}
                  >
                    没有创建的实验层
                  </div>
                )}
              </Card>
            ),
          },

          {
            key: 'create',
            label: '创建实验层',
            children: (
              <Card title="创建实验层">
                <Form layout="vertical">
                  <Row gutter={[16, 16]}>
                    <Col span={12}>
                      <Form.Item label="实验层ID" required>
                        <Input
                          placeholder="实验层ID *"
                          value={layerForm.layer_id}
                          onChange={e =>
                            setLayerForm({
                              ...layerForm,
                              layer_id: e.target.value,
                            })
                          }
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="实验层名称" required>
                        <Input
                          placeholder="实验层名称 *"
                          value={layerForm.name}
                          onChange={e =>
                            setLayerForm({ ...layerForm, name: e.target.value })
                          }
                        />
                      </Form.Item>
                    </Col>
                    <Col span={24}>
                      <Form.Item label="描述">
                        <Input
                          placeholder="描述"
                          value={layerForm.description}
                          onChange={e =>
                            setLayerForm({
                              ...layerForm,
                              description: e.target.value,
                            })
                          }
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="实验层类型">
                        <Select
                          value={layerForm.layer_type}
                          onChange={value =>
                            setLayerForm({
                              ...layerForm,
                              layer_type: value as LayerType,
                            })
                          }
                          placeholder="实验层类型"
                        >
                          <Option value={LayerType.MUTUALLY_EXCLUSIVE}>
                            互斥型
                          </Option>
                          <Option value={LayerType.ORTHOGONAL}>正交型</Option>
                          <Option value={LayerType.HOLDBACK}>保留层</Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="冲突解决策略">
                        <Select
                          value={layerForm.conflict_resolution}
                          onChange={value =>
                            setLayerForm({
                              ...layerForm,
                              conflict_resolution: value as ConflictResolution,
                            })
                          }
                          placeholder="冲突解决策略"
                        >
                          <Option value={ConflictResolution.PRIORITY_BASED}>
                            基于优先级
                          </Option>
                          <Option
                            value={ConflictResolution.FIRST_COME_FIRST_SERVE}
                          >
                            先到先得
                          </Option>
                          <Option value={ConflictResolution.ROUND_ROBIN}>
                            轮转分配
                          </Option>
                          <Option value={ConflictResolution.RANDOM}>
                            随机选择
                          </Option>
                        </Select>
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="流量百分比">
                        <Input
                          type="number"
                          placeholder="流量百分比"
                          value={layerForm.traffic_percentage}
                          onChange={e =>
                            setLayerForm({
                              ...layerForm,
                              traffic_percentage: parseInt(e.target.value),
                            })
                          }
                          min={0}
                          max={100}
                        />
                      </Form.Item>
                    </Col>
                    <Col span={12}>
                      <Form.Item label="优先级">
                        <Input
                          type="number"
                          placeholder="优先级"
                          value={layerForm.priority}
                          onChange={e =>
                            setLayerForm({
                              ...layerForm,
                              priority: parseInt(e.target.value),
                            })
                          }
                        />
                      </Form.Item>
                    </Col>
                    <Col span={24}>
                      <Form.Item>
                        <Checkbox
                          checked={layerForm.is_active}
                          onChange={e =>
                            setLayerForm({
                              ...layerForm,
                              is_active: e.target.checked,
                            })
                          }
                        >
                          立即激活
                        </Checkbox>
                      </Form.Item>
                    </Col>
                  </Row>

                  <Button
                    type="primary"
                    icon={<PlusOutlined />}
                    onClick={handleCreateLayer}
                    loading={loading}
                  >
                    创建实验层
                  </Button>

                  <Card
                    style={{
                      marginTop: '16px',
                      background: '#f0f9ff',
                      border: '1px solid #0ea5e9',
                    }}
                  >
                    <Title level={5}>实验层类型说明</Title>
                    <div style={{ lineHeight: '1.6' }}>
                      <div>
                        <Text strong>互斥型:</Text>{' '}
                        同一用户只能参与该层中的一个实验
                      </div>
                      <div>
                        <Text strong>正交型:</Text>{' '}
                        与其他层独立，用户可同时参与多个层的实验
                      </div>
                      <div>
                        <Text strong>保留层:</Text>{' '}
                        预留对照流量，用于稳定性与基线对比
                      </div>
                    </div>
                  </Card>
                </Form>
              </Card>
            ),
          },

          {
            key: 'conflicts',
            label: '冲突管理',
            children: (
              <Card title={`冲突管理 (${conflicts.length})`}>
                {conflicts.length > 0 ? (
                  <Space
                    direction="vertical"
                    size="large"
                    style={{ width: '100%' }}
                  >
                    {conflicts.map((conflict, index) => (
                      <Card
                        key={index}
                        style={{ borderLeft: '4px solid #ff4d4f' }}
                      >
                        <div>
                          <div
                            style={{
                              display: 'flex',
                              justifyContent: 'space-between',
                              alignItems: 'flex-start',
                              marginBottom: '8px',
                            }}
                          >
                            <Title
                              level={5}
                              style={{ margin: 0, color: '#cf1322' }}
                            >
                              冲突 #{conflict.id || index + 1}
                            </Title>
                            <Badge
                              color="red"
                              text={conflict.type || '未知类型'}
                            />
                          </div>
                          <Text
                            style={{ display: 'block', marginBottom: '8px' }}
                          >
                            {conflict.description || '冲突描述'}
                          </Text>
                          <Text
                            type="secondary"
                            style={{
                              fontSize: '12px',
                              display: 'block',
                              marginBottom: '12px',
                            }}
                          >
                            影响层:{' '}
                            {conflict.affected_layers?.join(', ') || '未知'}
                          </Text>
                          <Space>
                            <Button
                              size="small"
                              onClick={() => showConflictDetails(conflict)}
                            >
                              查看详情
                            </Button>
                            <Button
                              size="small"
                              type="primary"
                              onClick={() => resolveConflict(conflict)}
                            >
                              解决冲突
                            </Button>
                          </Space>
                        </div>
                      </Card>
                    ))}
                  </Space>
                ) : (
                  <div
                    style={{
                      textAlign: 'center',
                      padding: '40px',
                      color: '#52c41a',
                    }}
                  >
                    没有检测到冲突
                  </div>
                )}
              </Card>
            ),
          },

          {
            key: 'metrics',
            label: '系统指标',
            children: (
              <Card title="系统指标">
                {metrics ? (
                  <Row gutter={[24, 24]}>
                    <Col span={12}>
                      <Title level={5}>流量分配</Title>
                      <Space
                        direction="vertical"
                        size="small"
                        style={{ width: '100%' }}
                      >
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                          }}
                        >
                          <Text>总分配流量:</Text>
                          <Text strong>
                            {metrics.total_traffic_allocated || 0}%
                          </Text>
                        </div>
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                          }}
                        >
                          <Text>平均层流量:</Text>
                          <Text strong>{metrics.avg_layer_traffic || 0}%</Text>
                        </div>
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                          }}
                        >
                          <Text>流量利用率:</Text>
                          <Text strong>
                            {metrics.traffic_utilization || 0}%
                          </Text>
                        </div>
                      </Space>
                    </Col>
                    <Col span={12}>
                      <Title level={5}>系统健康</Title>
                      <Space
                        direction="vertical"
                        size="small"
                        style={{ width: '100%' }}
                      >
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                          }}
                        >
                          <Text>活跃层数:</Text>
                          <Text strong>{metrics.active_layers || 0}</Text>
                        </div>
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                          }}
                        >
                          <Text>冲突数量:</Text>
                          <Text
                            strong
                            style={{
                              color:
                                conflicts.length > 0 ? '#ff4d4f' : '#52c41a',
                            }}
                          >
                            {conflicts.length}
                          </Text>
                        </div>
                        <div
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                          }}
                        >
                          <Text>系统状态:</Text>
                          <Badge color="green" text="正常" />
                        </div>
                      </Space>
                    </Col>
                  </Row>
                ) : (
                  <div
                    style={{
                      textAlign: 'center',
                      padding: '40px',
                      color: '#999',
                    }}
                  >
                    正在加载系统指标...
                  </div>
                )}
              </Card>
            ),
          },
        ]}
      />
    </div>
  )
}

export default LayeredExperimentsManagementPage
