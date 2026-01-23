/**
 * 记忆召回测试页面
 * 测试多维度召回机制：向量相似度、时间相关性、实体匹配
 */
import React, { useState } from 'react'
import { logger } from '../utils/logger'
import {
  Card,
  Input,
  Button,
  List,
  Tag,
  Progress,
  Space,
  Row,
  Col,
  Divider,
  Alert,
  Tabs,
  Badge,
  Timeline,
  Empty,
  Spin,
  message,
} from 'antd'
import {
  SearchOutlined,
  ExperimentOutlined,
  ClockCircleOutlined,
  TagsOutlined,
  RadarChartOutlined,
  LinkOutlined,
} from '@ant-design/icons'
import { Memory, MemoryType, MemoryFilters } from '@/types/memory'
import { memoryService } from '@/services/memoryService'
import MemoryGraphVisualization from '@/components/memory/MemoryGraphVisualization'

const { TextArea } = Input
const { TabPane } = Tabs

const MemoryRecallTestPage: React.FC = () => {
  const [query, setQuery] = useState('')
  const [searchResults, setSearchResults] = useState<Memory[]>([])
  const [relatedMemories, setRelatedMemories] = useState<Memory[]>([])
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null)
  const [loading, setLoading] = useState(false)
  const [searchMode, setSearchMode] = useState<
    'vector' | 'temporal' | 'entity' | 'hybrid'
  >('hybrid')
  const [filters, setFilters] = useState<MemoryFilters>({})

  // 执行记忆搜索
  const handleSearch = async () => {
    if (!query.trim()) return

    setLoading(true)
    try {
      // 根据搜索模式设置过滤器
      let searchFilters = { ...filters }

      if (searchMode === 'temporal') {
        // 时间相关搜索：获取最近的记忆
        const now = new Date()
        const dayAgo = new Date(now.getTime() - 24 * 60 * 60 * 1000)
        searchFilters.created_after = dayAgo.toISOString()
      }

      const results = await memoryService.searchMemories(
        query,
        searchFilters,
        20
      )
      setSearchResults(results)

      // 如果有结果，自动选择第一个并获取相关记忆
      if (results.length > 0) {
        handleMemorySelect(results[0])
      }
    } catch (error) {
      logger.error('搜索失败:', error)
    } finally {
      setLoading(false)
    }
  }

  // 选择记忆并获取相关记忆
  const handleMemorySelect = async (memory: Memory) => {
    setSelectedMemory(memory)

    try {
      const related = await memoryService.getRelatedMemories(memory.id, 2, 10)
      setRelatedMemories(related)
    } catch (error) {
      logger.error('获取相关记忆失败:', error)
    }
  }

  // 创建测试记忆
  const handleCreateTestMemory = async () => {
    const testMemories = [
      {
        type: MemoryType.WORKING,
        content: '用户询问了关于React Hooks的使用方法',
        importance: 0.6,
        tags: ['react', 'hooks', 'frontend'],
      },
      {
        type: MemoryType.EPISODIC,
        content: '上午10点，用户成功部署了应用到生产环境',
        importance: 0.8,
        tags: ['deployment', 'production', 'success'],
      },
      {
        type: MemoryType.SEMANTIC,
        content: 'React是一个用于构建用户界面的JavaScript库',
        importance: 0.9,
        tags: ['react', 'javascript', 'knowledge'],
      },
    ]

    try {
      for (const memory of testMemories) {
        await memoryService.createMemory(memory)
      }
      message.success('测试记忆创建成功')
    } catch (error) {
      logger.error('创建测试记忆失败:', error)
      message.error('创建测试记忆失败')
    }
  }

  // 渲染记忆卡片
  const renderMemoryCard = (memory: Memory, showScore: boolean = true) => (
    <Card
      size="small"
      hoverable
      onClick={() => handleMemorySelect(memory)}
      style={{
        marginBottom: 12,
        borderLeft: `4px solid ${getMemoryTypeColor(memory.type)}`,
        backgroundColor: selectedMemory?.id === memory.id ? '#f0f8ff' : 'white',
      }}
    >
      <Row gutter={12}>
        <Col span={16}>
          <div style={{ marginBottom: 8 }}>
            <strong>{memory.content.substring(0, 100)}</strong>
            {memory.content.length > 100 && '...'}
          </div>
          <Space size={8}>
            <Tag color={getMemoryTypeColor(memory.type)}>
              {getMemoryTypeLabel(memory.type)}
            </Tag>
            {memory.tags?.map(tag => (
              <Tag key={tag}>{tag}</Tag>
            ))}
          </Space>
        </Col>
        <Col span={8} style={{ textAlign: 'right' }}>
          {showScore && memory.relevance_score && (
            <div style={{ marginBottom: 8 }}>
              <span style={{ fontSize: 12, color: '#666' }}>相关性:</span>
              <Progress
                percent={memory.relevance_score * 100}
                size="small"
                style={{ width: 80 }}
              />
            </div>
          )}
          <Space size={4} style={{ fontSize: 12 }}>
            <span>
              <ClockCircleOutlined /> {memory.access_count}次
            </span>
            <span>重要性: {(memory.importance * 100).toFixed(0)}%</span>
          </Space>
        </Col>
      </Row>
    </Card>
  )

  const getMemoryTypeColor = (type: MemoryType) => {
    switch (type) {
      case MemoryType.WORKING:
        return 'green'
      case MemoryType.EPISODIC:
        return 'blue'
      case MemoryType.SEMANTIC:
        return 'purple'
      default:
        return 'default'
    }
  }

  const getMemoryTypeLabel = (type: MemoryType) => {
    switch (type) {
      case MemoryType.WORKING:
        return '工作记忆'
      case MemoryType.EPISODIC:
        return '情景记忆'
      case MemoryType.SEMANTIC:
        return '语义记忆'
      default:
        return '未知'
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <h1>
        <ExperimentOutlined /> 记忆召回测试
      </h1>
      <p style={{ color: '#666', marginBottom: 24 }}>
        测试多维度记忆召回机制：向量相似度、时间相关性、实体匹配、链式激活
      </p>

      {/* 搜索面板 */}
      <Card style={{ marginBottom: 24 }}>
        <Space direction="vertical" style={{ width: '100%' }}>
          <Alert
            message="召回策略说明"
            description={
              <ul style={{ margin: '8px 0', paddingLeft: 20 }}>
                <li>
                  <strong>语义匹配</strong>: 基于语义嵌入的相似度匹配
                </li>
                <li>
                  <strong>时间相关</strong>: 考虑时间衰减和最近访问
                </li>
                <li>
                  <strong>实体匹配</strong>: 基于关键实体和标签匹配
                </li>
                <li>
                  <strong>混合搜索</strong>: 融合多维度的智能召回
                </li>
              </ul>
            }
            type="info"
          />

          <div>
            <span>召回模式:</span>
            <Space style={{ marginLeft: 16 }}>
              <Button
                type={searchMode === 'hybrid' ? 'primary' : 'default'}
                onClick={() => setSearchMode('hybrid')}
                icon={<RadarChartOutlined />}
              >
                混合召回
              </Button>
              <Button
                type={searchMode === 'vector' ? 'primary' : 'default'}
                onClick={() => setSearchMode('vector')}
                icon={<SearchOutlined />}
              >
                向量搜索
              </Button>
              <Button
                type={searchMode === 'temporal' ? 'primary' : 'default'}
                onClick={() => setSearchMode('temporal')}
                icon={<ClockCircleOutlined />}
              >
                时间搜索
              </Button>
              <Button
                type={searchMode === 'entity' ? 'primary' : 'default'}
                onClick={() => setSearchMode('entity')}
                icon={<TagsOutlined />}
              >
                实体搜索
              </Button>
            </Space>
          </div>

          <TextArea
            name="memoryRecallQuery"
            placeholder="输入查询内容，测试记忆召回..."
            value={query}
            onChange={e => setQuery(e.target.value)}
            rows={3}
            onPressEnter={e => {
              if (e.shiftKey) return
              e.preventDefault()
              handleSearch()
            }}
          />

          <Space>
            <Button
              type="primary"
              icon={<SearchOutlined />}
              onClick={handleSearch}
              loading={loading}
            >
              执行召回
            </Button>
            <Button onClick={handleCreateTestMemory}>创建测试记忆</Button>
          </Space>
        </Space>
      </Card>

      {/* 结果展示 */}
      <Row gutter={16}>
        <Col span={12}>
          <Card
            title={
              <span>
                召回结果
                <Badge count={searchResults.length} style={{ marginLeft: 8 }} />
              </span>
            }
            style={{ height: '60vh', overflow: 'auto' }}
          >
            {loading ? (
              <div style={{ textAlign: 'center', padding: 50 }}>
                <Spin size="large" />
                <div style={{ marginTop: 16 }}>正在执行多维度记忆召回...</div>
              </div>
            ) : searchResults.length > 0 ? (
              <List
                dataSource={searchResults}
                renderItem={item => renderMemoryCard(item)}
              />
            ) : (
              <Empty description="暂无结果" />
            )}
          </Card>
        </Col>

        <Col span={12}>
          <Card
            title={
              <span>
                <LinkOutlined /> 关联记忆链
                {selectedMemory && (
                  <Badge
                    count={relatedMemories.length}
                    style={{ marginLeft: 8 }}
                  />
                )}
              </span>
            }
            style={{ height: '60vh', overflow: 'auto' }}
          >
            {selectedMemory ? (
              <div>
                <Alert
                  message="选中的记忆"
                  description={selectedMemory.content}
                  type="success"
                  style={{ marginBottom: 16 }}
                />

                <Timeline>
                  {relatedMemories.map((memory, index) => (
                    <Timeline.Item
                      key={memory.id}
                      color={getMemoryTypeColor(memory.type)}
                    >
                      <div
                        onClick={() => handleMemorySelect(memory)}
                        style={{ cursor: 'pointer' }}
                      >
                        <Tag color={getMemoryTypeColor(memory.type)}>
                          深度 {Math.floor(index / 3) + 1}
                        </Tag>
                        <div style={{ marginTop: 4 }}>
                          {memory.content.substring(0, 100)}...
                        </div>
                        <div
                          style={{ fontSize: 12, color: '#999', marginTop: 4 }}
                        >
                          相关度:{' '}
                          {((memory.relevance_score || 0) * 100).toFixed(1)}%
                        </div>
                      </div>
                    </Timeline.Item>
                  ))}
                </Timeline>
              </div>
            ) : (
              <Empty description="选择一个记忆查看关联链" />
            )}
          </Card>
        </Col>
      </Row>

      {/* 记忆网络可视化 */}
      {searchResults.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <MemoryGraphVisualization
            memories={searchResults}
            onNodeClick={handleMemorySelect}
          />
        </div>
      )}
    </div>
  )
}

export default MemoryRecallTestPage
