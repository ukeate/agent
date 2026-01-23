/**
 * 知识图谱探索工具面板
 *
 * 功能包括：
 * - 路径查找工具，支持任意两实体间的关系路径发现
 * - 实体邻域探索功能，展示N跳范围内的相关实体
 * - 关系类型过滤和实体类型筛选功能
 * - 子图提取和局部视图生成
 * - 高级探索选项和参数配置
 */

import React, { useState, useCallback, useEffect } from 'react'
import {
  Card,
  Space,
  Typography,
  Row,
  Col,
  Select,
  Button,
  InputNumber,
  Slider,
  Checkbox,
  Collapse,
  Tag,
  Tree,
  List,
  Tooltip,
  Badge,
  message,
  Modal,
  Input,
} from 'antd'
import {
  SearchOutlined,
  RadarChartOutlined,
  FilterOutlined,
  ShareAltOutlined,
  SaveOutlined,
  BranchesOutlined,
  NodeExpandOutlined,
  SettingOutlined,
  ExportOutlined,
  EyeOutlined,
  DeleteOutlined,
} from '@ant-design/icons'

const { Title, Text } = Typography
const { Option } = Select
const { Panel } = Collapse
const { TreeNode } = Tree

// ==================== 类型定义 ====================

export interface PathFindingConfig {
  sourceEntity: string
  targetEntity: string
  maxDepth: number
  pathType: 'shortest' | 'all' | 'k_shortest'
  relationTypes: string[]
  excludeEntities: string[]
}

export interface NeighborhoodConfig {
  centerEntity: string
  depth: number
  entityTypes: string[]
  relationTypes: string[]
  minConfidence: number
  maxNodes: number
}

export interface FilterConfig {
  entityTypes: string[]
  relationTypes: string[]
  confidenceRange: [number, number]
  timeRange?: [Date, Date]
  properties: Record<string, any>
}

export interface SubgraphConfig {
  name: string
  description: string
  entities: string[]
  includeConnections: boolean
  depth: number
}

interface ExplorationToolPanelProps {
  availableEntityTypes?: string[]
  availableRelationTypes?: string[]
  selectedNodes?: string[]
  onPathFinding?: (config: PathFindingConfig) => void
  onNeighborhoodExploration?: (config: NeighborhoodConfig) => void
  onFilterChange?: (config: FilterConfig) => void
  onSubgraphExtract?: (config: SubgraphConfig) => void
  className?: string
}

// ==================== 默认配置 ====================

const defaultEntityTypes = [
  'Person',
  'Organization',
  'Location',
  'Concept',
  'Event',
  'Document',
  'Product',
  'Technology',
]

const defaultRelationTypes = [
  'works_at',
  'located_in',
  'related_to',
  'participated_in',
  'created_by',
  'owns',
  'collaborates_with',
  'mentions',
  'influences',
  'part_of',
]

// ==================== 主组件 ====================

const ExplorationToolPanel: React.FC<ExplorationToolPanelProps> = ({
  availableEntityTypes = defaultEntityTypes,
  availableRelationTypes = defaultRelationTypes,
  selectedNodes = [],
  onPathFinding,
  onNeighborhoodExploration,
  onFilterChange,
  onSubgraphExtract,
  className = '',
}) => {
  // ==================== 状态管理 ====================

  const [pathConfig, setPathConfig] = useState<PathFindingConfig>({
    sourceEntity: '',
    targetEntity: '',
    maxDepth: 5,
    pathType: 'shortest',
    relationTypes: [],
    excludeEntities: [],
  })

  const [neighborhoodConfig, setNeighborhoodConfig] =
    useState<NeighborhoodConfig>({
      centerEntity: '',
      depth: 2,
      entityTypes: [],
      relationTypes: [],
      minConfidence: 0.5,
      maxNodes: 100,
    })

  const [filterConfig, setFilterConfig] = useState<FilterConfig>({
    entityTypes: availableEntityTypes,
    relationTypes: availableRelationTypes,
    confidenceRange: [0, 1],
    properties: {},
  })

  const [activeFilters, setActiveFilters] = useState<string[]>([])
  const [savedSubgraphs, setSavedSubgraphs] = useState<SubgraphConfig[]>([])
  const [subgraphModalVisible, setSubgraphModalVisible] = useState(false)
  const [currentSubgraph, setCurrentSubgraph] = useState<
    Partial<SubgraphConfig>
  >({})

  // ==================== 路径查找功能 ====================

  const handlePathFinding = useCallback(() => {
    if (!pathConfig.sourceEntity || !pathConfig.targetEntity) {
      message.warning('请选择起始和目标实体')
      return
    }

    if (pathConfig.sourceEntity === pathConfig.targetEntity) {
      message.warning('起始实体和目标实体不能相同')
      return
    }

    onPathFinding?.(pathConfig)
    message.success('开始路径查找...')
  }, [pathConfig, onPathFinding])

  // ==================== 邻域探索功能 ====================

  const handleNeighborhoodExploration = useCallback(() => {
    if (!neighborhoodConfig.centerEntity) {
      message.warning('请选择中心实体')
      return
    }

    onNeighborhoodExploration?.(neighborhoodConfig)
    message.success(`开始探索 ${neighborhoodConfig.centerEntity} 的邻域...`)
  }, [neighborhoodConfig, onNeighborhoodExploration])

  // ==================== 过滤器功能 ====================

  const handleFilterChange = useCallback(() => {
    onFilterChange?.(filterConfig)

    // 更新活跃过滤器标记
    const filters: string[] = []
    if (filterConfig.entityTypes.length < availableEntityTypes.length) {
      filters.push('实体类型')
    }
    if (filterConfig.relationTypes.length < availableRelationTypes.length) {
      filters.push('关系类型')
    }
    if (
      filterConfig.confidenceRange[0] > 0 ||
      filterConfig.confidenceRange[1] < 1
    ) {
      filters.push('置信度')
    }

    setActiveFilters(filters)
    message.success('过滤器已应用')
  }, [
    filterConfig,
    availableEntityTypes.length,
    availableRelationTypes.length,
    onFilterChange,
  ])

  const resetFilters = useCallback(() => {
    setFilterConfig({
      entityTypes: availableEntityTypes,
      relationTypes: availableRelationTypes,
      confidenceRange: [0, 1],
      properties: {},
    })
    setActiveFilters([])
    onFilterChange?.({
      entityTypes: availableEntityTypes,
      relationTypes: availableRelationTypes,
      confidenceRange: [0, 1],
      properties: {},
    })
    message.success('过滤器已重置')
  }, [availableEntityTypes, availableRelationTypes, onFilterChange])

  // ==================== 子图提取功能 ====================

  const handleCreateSubgraph = useCallback(() => {
    if (selectedNodes.length === 0) {
      message.warning('请先选择要包含的节点')
      return
    }

    setCurrentSubgraph({
      entities: selectedNodes,
      includeConnections: true,
      depth: 1,
    })
    setSubgraphModalVisible(true)
  }, [selectedNodes])

  const handleSaveSubgraph = useCallback(() => {
    if (!currentSubgraph.name || !currentSubgraph.entities?.length) {
      message.warning('请填写子图名称并选择实体')
      return
    }

    const subgraph: SubgraphConfig = {
      name: currentSubgraph.name,
      description: currentSubgraph.description || '',
      entities: currentSubgraph.entities,
      includeConnections: currentSubgraph.includeConnections ?? true,
      depth: currentSubgraph.depth ?? 1,
    }

    setSavedSubgraphs(prev => [...prev, subgraph])
    onSubgraphExtract?.(subgraph)
    setSubgraphModalVisible(false)
    setCurrentSubgraph({})

    message.success(`子图 "${subgraph.name}" 已保存`)
  }, [currentSubgraph, onSubgraphExtract])

  const handleDeleteSubgraph = useCallback((index: number) => {
    setSavedSubgraphs(prev => prev.filter((_, i) => i !== index))
    message.success('子图已删除')
  }, [])

  // ==================== 预设实体选择 ====================

  const handleSetEntityFromSelection = useCallback(
    (target: 'source' | 'target' | 'center') => {
      if (selectedNodes.length === 0) {
        message.warning('请先在图中选择一个节点')
        return
      }

      if (selectedNodes.length > 1) {
        message.warning('请只选择一个节点')
        return
      }

      const entity = selectedNodes[0]

      if (target === 'source') {
        setPathConfig(prev => ({ ...prev, sourceEntity: entity }))
      } else if (target === 'target') {
        setPathConfig(prev => ({ ...prev, targetEntity: entity }))
      } else if (target === 'center') {
        setNeighborhoodConfig(prev => ({ ...prev, centerEntity: entity }))
      }
    },
    [selectedNodes]
  )

  // ==================== 生命周期 ====================

  useEffect(() => {
    // 当选中的节点变化时，自动更新实体选择
    if (selectedNodes.length === 1) {
      const entity = selectedNodes[0]
      if (!pathConfig.sourceEntity) {
        setPathConfig(prev => ({ ...prev, sourceEntity: entity }))
      } else if (
        !pathConfig.targetEntity &&
        entity !== pathConfig.sourceEntity
      ) {
        setPathConfig(prev => ({ ...prev, targetEntity: entity }))
      }

      if (!neighborhoodConfig.centerEntity) {
        setNeighborhoodConfig(prev => ({ ...prev, centerEntity: entity }))
      }
    }
  }, [
    selectedNodes,
    pathConfig.sourceEntity,
    pathConfig.targetEntity,
    neighborhoodConfig.centerEntity,
  ])

  // ==================== 渲染组件 ====================

  return (
    <Card
      className={`exploration-tool-panel ${className}`}
      title={
        <Space>
          <RadarChartOutlined />
          <Title level={4} style={{ margin: 0 }}>
            探索工具
          </Title>
          {activeFilters.length > 0 && (
            <Badge
              count={activeFilters.length}
              style={{ backgroundColor: '#52c41a' }}
            />
          )}
        </Space>
      }
    >
      <Collapse defaultActiveKey={['path-finding']} ghost>
        {/* 路径查找工具 */}
        <Panel
          header={
            <Space>
              <BranchesOutlined />
              <Text strong>路径查找</Text>
            </Space>
          }
          key="path-finding"
        >
          <Space direction="vertical" style={{ width: '100%' }}>
            {/* 实体选择 */}
            <Row gutter={[16, 16]}>
              <Col span={24}>
                <Text strong>起始实体:</Text>
                <Row gutter={8} align="middle" style={{ marginTop: 4 }}>
                  <Col flex="auto">
                    <Input
                      value={pathConfig.sourceEntity}
                      onChange={e =>
                        setPathConfig(prev => ({
                          ...prev,
                          sourceEntity: e.target.value,
                        }))
                      }
                      placeholder="请输入或选择起始实体"
                    />
                  </Col>
                  <Col>
                    <Tooltip title="使用当前选中的节点">
                      <Button
                        size="small"
                        onClick={() => handleSetEntityFromSelection('source')}
                        disabled={selectedNodes.length !== 1}
                      >
                        选中
                      </Button>
                    </Tooltip>
                  </Col>
                </Row>
              </Col>

              <Col span={24}>
                <Text strong>目标实体:</Text>
                <Row gutter={8} align="middle" style={{ marginTop: 4 }}>
                  <Col flex="auto">
                    <Input
                      value={pathConfig.targetEntity}
                      onChange={e =>
                        setPathConfig(prev => ({
                          ...prev,
                          targetEntity: e.target.value,
                        }))
                      }
                      placeholder="请输入或选择目标实体"
                    />
                  </Col>
                  <Col>
                    <Tooltip title="使用当前选中的节点">
                      <Button
                        size="small"
                        onClick={() => handleSetEntityFromSelection('target')}
                        disabled={selectedNodes.length !== 1}
                      >
                        选中
                      </Button>
                    </Tooltip>
                  </Col>
                </Row>
              </Col>
            </Row>

            {/* 路径配置 */}
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Text strong>路径类型:</Text>
                <Select
                  value={pathConfig.pathType}
                  onChange={value =>
                    setPathConfig(prev => ({ ...prev, pathType: value }))
                  }
                  style={{ width: '100%', marginTop: 4 }}
                >
                  <Option value="shortest">最短路径</Option>
                  <Option value="all">所有路径</Option>
                  <Option value="k_shortest">K条最短路径</Option>
                </Select>
              </Col>

              <Col span={12}>
                <Text strong>最大深度:</Text>
                <InputNumber
                  value={pathConfig.maxDepth}
                  onChange={value =>
                    setPathConfig(prev => ({ ...prev, maxDepth: value || 5 }))
                  }
                  min={1}
                  max={10}
                  style={{ width: '100%', marginTop: 4 }}
                />
              </Col>
            </Row>

            {/* 关系类型过滤 */}
            <div>
              <Text strong>允许的关系类型:</Text>
              <Select
                mode="multiple"
                value={pathConfig.relationTypes}
                onChange={value =>
                  setPathConfig(prev => ({ ...prev, relationTypes: value }))
                }
                placeholder="全部关系类型（留空表示不限制）"
                style={{ width: '100%', marginTop: 4 }}
              >
                {availableRelationTypes.map(type => (
                  <Option key={type} value={type}>
                    {type}
                  </Option>
                ))}
              </Select>
            </div>

            {/* 执行按钮 */}
            <Row>
              <Col span={24} style={{ textAlign: 'right' }}>
                <Button
                  type="primary"
                  icon={<SearchOutlined />}
                  onClick={handlePathFinding}
                  disabled={
                    !pathConfig.sourceEntity || !pathConfig.targetEntity
                  }
                >
                  查找路径
                </Button>
              </Col>
            </Row>
          </Space>
        </Panel>

        {/* 邻域探索工具 */}
        <Panel
          header={
            <Space>
              <NodeExpandOutlined />
              <Text strong>邻域探索</Text>
            </Space>
          }
          key="neighborhood"
        >
          <Space direction="vertical" style={{ width: '100%' }}>
            {/* 中心实体选择 */}
            <div>
              <Text strong>中心实体:</Text>
              <Row gutter={8} align="middle" style={{ marginTop: 4 }}>
                <Col flex="auto">
                  <Input
                    value={neighborhoodConfig.centerEntity}
                    onChange={e =>
                      setNeighborhoodConfig(prev => ({
                        ...prev,
                        centerEntity: e.target.value,
                      }))
                    }
                    placeholder="请输入或选择中心实体"
                  />
                </Col>
                <Col>
                  <Tooltip title="使用当前选中的节点">
                    <Button
                      size="small"
                      onClick={() => handleSetEntityFromSelection('center')}
                      disabled={selectedNodes.length !== 1}
                    >
                      选中
                    </Button>
                  </Tooltip>
                </Col>
              </Row>
            </div>

            {/* 探索参数 */}
            <Row gutter={[16, 16]}>
              <Col span={12}>
                <Text strong>探索深度:</Text>
                <Slider
                  value={neighborhoodConfig.depth}
                  onChange={value =>
                    setNeighborhoodConfig(prev => ({ ...prev, depth: value }))
                  }
                  min={1}
                  max={5}
                  marks={{ 1: '1', 2: '2', 3: '3', 4: '4', 5: '5' }}
                  style={{ marginTop: 8 }}
                />
              </Col>

              <Col span={12}>
                <Text strong>最大节点数:</Text>
                <InputNumber
                  value={neighborhoodConfig.maxNodes}
                  onChange={value =>
                    setNeighborhoodConfig(prev => ({
                      ...prev,
                      maxNodes: value || 100,
                    }))
                  }
                  min={10}
                  max={1000}
                  step={10}
                  style={{ width: '100%', marginTop: 4 }}
                />
              </Col>
            </Row>

            {/* 置信度阈值 */}
            <div>
              <Row gutter={8} align="middle">
                <Col span={8}>
                  <Text strong>最小置信度:</Text>
                </Col>
                <Col span={12}>
                  <Slider
                    value={neighborhoodConfig.minConfidence}
                    onChange={value =>
                      setNeighborhoodConfig(prev => ({
                        ...prev,
                        minConfidence: value,
                      }))
                    }
                    min={0}
                    max={1}
                    step={0.1}
                  />
                </Col>
                <Col span={4}>
                  <Text code>
                    {neighborhoodConfig.minConfidence.toFixed(1)}
                  </Text>
                </Col>
              </Row>
            </div>

            {/* 类型过滤 */}
            <Row gutter={16}>
              <Col span={12}>
                <Text strong>实体类型:</Text>
                <Select
                  mode="multiple"
                  value={neighborhoodConfig.entityTypes}
                  onChange={value =>
                    setNeighborhoodConfig(prev => ({
                      ...prev,
                      entityTypes: value,
                    }))
                  }
                  placeholder="全部类型"
                  style={{ width: '100%', marginTop: 4 }}
                  maxTagCount={2}
                >
                  {availableEntityTypes.map(type => (
                    <Option key={type} value={type}>
                      {type}
                    </Option>
                  ))}
                </Select>
              </Col>

              <Col span={12}>
                <Text strong>关系类型:</Text>
                <Select
                  mode="multiple"
                  value={neighborhoodConfig.relationTypes}
                  onChange={value =>
                    setNeighborhoodConfig(prev => ({
                      ...prev,
                      relationTypes: value,
                    }))
                  }
                  placeholder="全部类型"
                  style={{ width: '100%', marginTop: 4 }}
                  maxTagCount={2}
                >
                  {availableRelationTypes.map(type => (
                    <Option key={type} value={type}>
                      {type}
                    </Option>
                  ))}
                </Select>
              </Col>
            </Row>

            {/* 执行按钮 */}
            <Row>
              <Col span={24} style={{ textAlign: 'right' }}>
                <Button
                  type="primary"
                  icon={<SearchOutlined />}
                  onClick={handleNeighborhoodExploration}
                  disabled={!neighborhoodConfig.centerEntity}
                >
                  开始探索
                </Button>
              </Col>
            </Row>
          </Space>
        </Panel>

        {/* 过滤器面板 */}
        <Panel
          header={
            <Space>
              <FilterOutlined />
              <Text strong>过滤器</Text>
              {activeFilters.length > 0 && (
                <Badge count={activeFilters.length} size="small" />
              )}
            </Space>
          }
          key="filters"
        >
          <Space direction="vertical" style={{ width: '100%' }}>
            {/* 实体类型过滤 */}
            <div>
              <Text strong>实体类型:</Text>
              <Checkbox.Group
                value={filterConfig.entityTypes}
                onChange={value =>
                  setFilterConfig(prev => ({
                    ...prev,
                    entityTypes: value as string[],
                  }))
                }
                style={{ width: '100%', marginTop: 8 }}
              >
                <Row gutter={[8, 8]}>
                  {availableEntityTypes.map(type => (
                    <Col span={8} key={type}>
                      <Checkbox value={type}>{type}</Checkbox>
                    </Col>
                  ))}
                </Row>
              </Checkbox.Group>
            </div>

            {/* 关系类型过滤 */}
            <div>
              <Text strong>关系类型:</Text>
              <Checkbox.Group
                value={filterConfig.relationTypes}
                onChange={value =>
                  setFilterConfig(prev => ({
                    ...prev,
                    relationTypes: value as string[],
                  }))
                }
                style={{ width: '100%', marginTop: 8 }}
              >
                <Row gutter={[8, 8]}>
                  {availableRelationTypes.map(type => (
                    <Col span={8} key={type}>
                      <Checkbox value={type} style={{ fontSize: 12 }}>
                        <Text ellipsis style={{ maxWidth: 80 }} title={type}>
                          {type}
                        </Text>
                      </Checkbox>
                    </Col>
                  ))}
                </Row>
              </Checkbox.Group>
            </div>

            {/* 置信度范围 */}
            <div>
              <Text strong>置信度范围:</Text>
              <Row gutter={8} align="middle" style={{ marginTop: 8 }}>
                <Col span={16}>
                  <Slider
                    range
                    value={filterConfig.confidenceRange}
                    onChange={value =>
                      setFilterConfig(prev => ({
                        ...prev,
                        confidenceRange: value as [number, number],
                      }))
                    }
                    min={0}
                    max={1}
                    step={0.1}
                  />
                </Col>
                <Col span={8}>
                  <Text code>
                    {filterConfig.confidenceRange[0].toFixed(1)} -{' '}
                    {filterConfig.confidenceRange[1].toFixed(1)}
                  </Text>
                </Col>
              </Row>
            </div>

            {/* 操作按钮 */}
            <Row gutter={8}>
              <Col flex="auto">
                <Button
                  onClick={resetFilters}
                  disabled={activeFilters.length === 0}
                >
                  重置过滤器
                </Button>
              </Col>
              <Col>
                <Button type="primary" onClick={handleFilterChange}>
                  应用过滤器
                </Button>
              </Col>
            </Row>

            {/* 当前过滤器状态 */}
            {activeFilters.length > 0 && (
              <div>
                <Text strong>当前过滤器:</Text>
                <div style={{ marginTop: 4 }}>
                  {activeFilters.map(filter => (
                    <Tag key={filter} color="blue">
                      {filter}
                    </Tag>
                  ))}
                </div>
              </div>
            )}
          </Space>
        </Panel>

        {/* 子图管理面板 */}
        <Panel
          header={
            <Space>
              <ShareAltOutlined />
              <Text strong>子图管理</Text>
              {savedSubgraphs.length > 0 && (
                <Badge count={savedSubgraphs.length} size="small" />
              )}
            </Space>
          }
          key="subgraphs"
        >
          <Space direction="vertical" style={{ width: '100%' }}>
            {/* 创建子图 */}
            <Row gutter={8}>
              <Col flex="auto">
                <Text>
                  已选择 <Text strong>{selectedNodes.length}</Text> 个节点
                </Text>
              </Col>
              <Col>
                <Button
                  type="primary"
                  icon={<SaveOutlined />}
                  onClick={handleCreateSubgraph}
                  disabled={selectedNodes.length === 0}
                  size="small"
                >
                  创建子图
                </Button>
              </Col>
            </Row>

            {/* 已保存的子图列表 */}
            {savedSubgraphs.length > 0 && (
              <div>
                <Text strong>已保存的子图:</Text>
                <List
                  size="small"
                  dataSource={savedSubgraphs}
                  style={{ marginTop: 8 }}
                  renderItem={(subgraph, index) => (
                    <List.Item
                      actions={[
                        <Tooltip title="查看" key="view">
                          <Button
                            type="text"
                            icon={<EyeOutlined />}
                            size="small"
                            onClick={() => onSubgraphExtract?.(subgraph)}
                          />
                        </Tooltip>,
                        <Tooltip title="删除" key="delete">
                          <Button
                            type="text"
                            icon={<DeleteOutlined />}
                            size="small"
                            danger
                            onClick={() => handleDeleteSubgraph(index)}
                          />
                        </Tooltip>,
                      ]}
                    >
                      <List.Item.Meta
                        title={<Text strong>{subgraph.name}</Text>}
                        description={
                          <Space size="small">
                            <Tag>{subgraph.entities.length} 节点</Tag>
                            <Tag>深度 {subgraph.depth}</Tag>
                            {subgraph.description && (
                              <Text
                                type="secondary"
                                ellipsis
                                style={{ maxWidth: 150 }}
                              >
                                {subgraph.description}
                              </Text>
                            )}
                          </Space>
                        }
                      />
                    </List.Item>
                  )}
                />
              </div>
            )}
          </Space>
        </Panel>
      </Collapse>

      {/* 创建子图模态框 */}
      <Modal
        title="创建子图"
        open={subgraphModalVisible}
        onOk={handleSaveSubgraph}
        onCancel={() => {
          setSubgraphModalVisible(false)
          setCurrentSubgraph({})
        }}
        okText="保存"
        cancelText="取消"
      >
        <Space direction="vertical" style={{ width: '100%' }}>
          <div>
            <Text strong>子图名称:</Text>
            <Input
              value={currentSubgraph.name}
              onChange={e =>
                setCurrentSubgraph(prev => ({ ...prev, name: e.target.value }))
              }
              placeholder="请输入子图名称"
              style={{ marginTop: 4 }}
            />
          </div>

          <div>
            <Text strong>描述（可选）:</Text>
            <Input.TextArea
              value={currentSubgraph.description}
              onChange={e =>
                setCurrentSubgraph(prev => ({
                  ...prev,
                  description: e.target.value,
                }))
              }
              placeholder="请输入子图描述"
              style={{ marginTop: 4 }}
              rows={2}
            />
          </div>

          <div>
            <Text strong>包含的实体:</Text>
            <div
              style={{
                marginTop: 4,
                padding: 8,
                backgroundColor: '#f5f5f5',
                borderRadius: 4,
              }}
            >
              {currentSubgraph.entities?.map(entity => (
                <Tag key={entity} style={{ margin: 2 }}>
                  {entity}
                </Tag>
              ))}
            </div>
          </div>

          <Row gutter={16}>
            <Col span={12}>
              <Checkbox
                checked={currentSubgraph.includeConnections}
                onChange={e =>
                  setCurrentSubgraph(prev => ({
                    ...prev,
                    includeConnections: e.target.checked,
                  }))
                }
              >
                包含连接关系
              </Checkbox>
            </Col>

            <Col span={12}>
              <Text>扩展深度:</Text>
              <InputNumber
                value={currentSubgraph.depth}
                onChange={value =>
                  setCurrentSubgraph(prev => ({ ...prev, depth: value || 1 }))
                }
                min={0}
                max={3}
                style={{ width: '100%', marginTop: 4 }}
              />
            </Col>
          </Row>
        </Space>
      </Modal>
    </Card>
  )
}

export default ExplorationToolPanel
