/**
 * 记忆层级可视化页面
 * 展示工作记忆、情景记忆、语义记忆的三层架构
 */
import React, { useState, useEffect } from 'react'
import { Card, Row, Col, Progress, Tag, List, Button, Tooltip, Badge, Statistic, Space, message, Input } from 'antd'
import { 
import { logger } from '../utils/logger'
  DatabaseOutlined, 
  ClockCircleOutlined, 
  BookOutlined,
  ThunderboltOutlined,
  RiseOutlined,
  SwapOutlined,
  DeleteOutlined,
  SyncOutlined,
  SearchOutlined,
  DownloadOutlined
} from '@ant-design/icons'
import { memoryService } from '@/services/memoryService'
import { Memory, MemoryType, MemoryAnalytics } from '@/types/memory'

const MemoryHierarchyPage: React.FC = () => {
  const [workingMemories, setWorkingMemories] = useState<Memory[]>([])
  const [episodicMemories, setEpisodicMemories] = useState<Memory[]>([])
  const [semanticMemories, setSemanticMemories] = useState<Memory[]>([])
  const [analytics, setAnalytics] = useState<MemoryAnalytics | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedMemory, setSelectedMemory] = useState<Memory | null>(null)
  const [searchResults, setSearchResults] = useState<Memory[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [searching, setSearching] = useState(false)

  useEffect(() => {
    loadMemories()
    loadAnalytics()
  }, [])

  const loadMemories = async () => {
    setLoading(true)
    try {
      // 获取当前会话ID（示例）
      const sessionId = localStorage.getItem('current_session_id') || 'default'
      
      // 加载各层级记忆
      const [working, episodic, semantic] = await Promise.all([
        memoryService.getSessionMemories(sessionId, MemoryType.WORKING, 20),
        memoryService.getSessionMemories(sessionId, MemoryType.EPISODIC, 20),
        memoryService.getSessionMemories(sessionId, MemoryType.SEMANTIC, 20)
      ])
      
      setWorkingMemories(working)
      setEpisodicMemories(episodic)
      setSemanticMemories(semantic)
    } catch (error) {
      logger.error('加载记忆失败:', error)
      message.error('加载记忆失败')
    } finally {
      setLoading(false)
    }
  }

  const loadAnalytics = async () => {
    try {
      const data = await memoryService.getMemoryAnalytics()
      setAnalytics(data)
    } catch (error) {
      logger.error('加载分析数据失败:', error)
    }
  }

  const handlePromoteMemory = async (memory: Memory) => {
    try {
      // 提升记忆层级
      const newType = memory.type === MemoryType.WORKING 
        ? MemoryType.EPISODIC 
        : MemoryType.SEMANTIC
      
      await memoryService.updateMemory(memory.id, { 
        status: memory.status,
        importance: Math.min(1.0, memory.importance * 1.2) 
      })
      
      message.success(`记忆已提升到${newType === MemoryType.EPISODIC ? '情景' : '语义'}记忆层`)
      loadMemories()
    } catch (error) {
      message.error('提升记忆失败')
    }
  }

  const handleConsolidateMemories = async () => {
    try {
      const sessionId = localStorage.getItem('current_session_id') || 'default'
      await memoryService.consolidateMemories(sessionId)
      message.success('记忆巩固完成')
      loadMemories()
    } catch (error) {
      message.error('记忆巩固失败')
    }
  }

  const handleCleanupOldMemories = async () => {
    try {
      await memoryService.cleanupOldMemories(30, 0.3)
      message.success('旧记忆清理完成')
      loadMemories()
      loadAnalytics()
    } catch (error) {
      logger.error('清理旧记忆失败:', error)
      message.error('清理旧记忆失败')
    }
  }

  const handleSearchMemories = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([])
      return
    }
    
    setSearching(true)
    try {
      const results = await memoryService.searchMemories(query, {}, 20)
      setSearchResults(results)
      message.success(`找到 ${results.length} 条相关记忆`)
    } catch (error) {
      logger.error('搜索记忆失败:', error)
      message.error('搜索记忆失败')
    } finally {
      setSearching(false)
    }
  }

  const handleExportMemories = async () => {
    try {
      const sessionId = localStorage.getItem('current_session_id') || 'default'
      const memories = await memoryService.exportMemories(sessionId)
      
      const blob = new Blob([JSON.stringify(memories, null, 2)], {
        type: 'application/json'
      })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url
      a.download = `memories-${sessionId}-${new Date().toISOString().split('T')[0]}.json`
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      URL.revokeObjectURL(url)
      
      message.success('记忆导出成功')
    } catch (error) {
      logger.error('导出记忆失败:', error)
      message.error('导出记忆失败')
    }
  }

  const renderMemoryCard = (memory: Memory, layerName: string, color: string) => (
    <Card
      key={memory.id}
      size="small"
      hoverable
      onClick={() => setSelectedMemory(memory)}
      style={{ 
        marginBottom: 8, 
        borderLeft: `3px solid ${color}`,
        cursor: 'pointer'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start' }}>
        <div style={{ flex: 1 }}>
          <div style={{ marginBottom: 4 }}>
            <Text style={{ fontSize: 12 }} ellipsis>
              {memory.content.substring(0, 100)}...
            </Text>
          </div>
          <Space size={4}>
            <Tag color={color} style={{ fontSize: 10 }}>
              {layerName}
            </Tag>
            <Tooltip title="重要性">
              <Progress 
                percent={memory.importance * 100} 
                size="small" 
                style={{ width: 60 }}
                strokeColor={memory.importance > 0.7 ? '#f5222d' : '#1890ff'}
                showInfo={false}
              />
            </Tooltip>
            <Tooltip title="访问次数">
              <Badge count={memory.access_count} style={{ fontSize: 10 }} />
            </Tooltip>
          </Space>
        </div>
        {memory.type !== MemoryType.SEMANTIC && (
          <Tooltip title="提升记忆层级">
            <Button 
              size="small" 
              icon={<RiseOutlined />}
              onClick={(e) => {
                e.stopPropagation()
                handlePromoteMemory(memory)
              }}
            />
          </Tooltip>
        )}
      </div>
    </Card>
  )

  const getMemoryTypeColor = (type: MemoryType) => {
    switch (type) {
      case MemoryType.WORKING:
        return '#52c41a'
      case MemoryType.EPISODIC:
        return '#1890ff'
      case MemoryType.SEMANTIC:
        return '#722ed1'
      default:
        return '#666'
    }
  }

  return (
    <div style={{ padding: 24 }}>
      <div style={{ marginBottom: 24 }}>
        <h1>
          <DatabaseOutlined /> 记忆层级系统
        </h1>
        <p style={{ color: '#666' }}>
          可视化展示三层记忆架构：工作记忆 → 情景记忆 → 语义记忆
        </p>
      </div>

      {/* 统计概览 */}
      {analytics && (
        <Row gutter={16} style={{ marginBottom: 24 }}>
          <Col span={6}>
            <Card>
              <Statistic
                title="总记忆数"
                value={analytics.total_memories}
                prefix={<DatabaseOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="平均重要性"
                value={analytics.avg_importance}
                precision={2}
                suffix="/ 1.0"
                valueStyle={{ color: '#3f8600' }}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="记忆增长率"
                value={analytics.memory_growth_rate}
                precision={2}
                suffix="条/天"
                prefix={<RiseOutlined />}
              />
            </Card>
          </Col>
          <Col span={6}>
            <Card>
              <Statistic
                title="存储使用"
                value={analytics.storage_usage_mb}
                precision={2}
                suffix="MB"
                prefix={<ThunderboltOutlined />}
              />
            </Card>
          </Col>
        </Row>
      )}

      {/* 搜索框 */}
      <div style={{ marginBottom: 16 }}>
        <Input.Search
          name="memorySearch"
          placeholder="搜索记忆内容..."
          allowClear
          enterButton={<SearchOutlined />}
          size="large"
          loading={searching}
          onSearch={handleSearchMemories}
          onChange={(e) => setSearchQuery(e.target.value)}
          style={{ maxWidth: 400 }}
        />
      </div>

      {/* 操作按钮 */}
      <div style={{ marginBottom: 16 }}>
        <Space>
          <Button 
            type="primary" 
            icon={<SwapOutlined />}
            onClick={handleConsolidateMemories}
          >
            巩固记忆
          </Button>
          <Button icon={<SyncOutlined />} onClick={loadMemories}>
            刷新
          </Button>
          <Button 
            icon={<DeleteOutlined />} 
            danger
            onClick={handleCleanupOldMemories}
          >
            清理旧记忆
          </Button>
          <Button 
            icon={<DownloadOutlined />}
            onClick={handleExportMemories}
          >
            导出记忆
          </Button>
        </Space>
      </div>

      {/* 搜索结果 */}
      {searchResults.length > 0 && (
        <Card 
          title={
            <span>
              <SearchOutlined style={{ color: '#1890ff' }} /> 搜索结果
              <Badge 
                count={searchResults.length} 
                style={{ marginLeft: 8 }}
                showZero
              />
            </span>
          }
          style={{ marginBottom: 16 }}
        >
          <List
            dataSource={searchResults}
            renderItem={(memory) => renderMemoryCard(
              memory, 
              memory.type || '未知', 
              getMemoryTypeColor(memory.type || MemoryType.WORKING)
            )}
            loading={searching}
            style={{ maxHeight: 300, overflow: 'auto' }}
          />
        </Card>
      )}

      {/* 三层记忆架构 */}
      <Row gutter={16}>
        {/* 工作记忆层 */}
        <Col span={8}>
          <Card 
            title={
              <span>
                <ThunderboltOutlined style={{ color: '#52c41a' }} /> 工作记忆
                <Badge 
                  count={workingMemories.length} 
                  style={{ marginLeft: 8 }}
                  showZero
                />
              </span>
            }
            extra={
              <Tag color="green">短期缓冲</Tag>
            }
            style={{ height: '70vh', overflow: 'auto' }}
          >
            <div style={{ fontSize: 12, color: '#666', marginBottom: 16 }}>
              容量: {workingMemories.length} / 100 | 
              特点: 快速访问，容量有限，自动淘汰
            </div>
            <List
              dataSource={workingMemories}
              renderItem={(memory) => renderMemoryCard(memory, '工作', '#52c41a')}
              loading={loading}
            />
          </Card>
        </Col>

        {/* 情景记忆层 */}
        <Col span={8}>
          <Card 
            title={
              <span>
                <ClockCircleOutlined style={{ color: '#1890ff' }} /> 情景记忆
                <Badge 
                  count={episodicMemories.length} 
                  style={{ marginLeft: 8 }}
                  showZero
                />
              </span>
            }
            extra={
              <Tag color="blue">具体事件</Tag>
            }
            style={{ height: '70vh', overflow: 'auto' }}
          >
            <div style={{ fontSize: 12, color: '#666', marginBottom: 16 }}>
              容量: {episodicMemories.length} / 10000 | 
              特点: 时序记录，情境相关，可召回
            </div>
            <List
              dataSource={episodicMemories}
              renderItem={(memory) => renderMemoryCard(memory, '情景', '#1890ff')}
              loading={loading}
            />
          </Card>
        </Col>

        {/* 语义记忆层 */}
        <Col span={8}>
          <Card 
            title={
              <span>
                <BookOutlined style={{ color: '#722ed1' }} /> 语义记忆
                <Badge 
                  count={semanticMemories.length} 
                  style={{ marginLeft: 8 }}
                  showZero
                />
              </span>
            }
            extra={
              <Tag color="purple">抽象知识</Tag>
            }
            style={{ height: '70vh', overflow: 'auto' }}
          >
            <div style={{ fontSize: 12, color: '#666', marginBottom: 16 }}>
              容量: {semanticMemories.length} / 5000 | 
              特点: 概念知识，永久存储，高度抽象
            </div>
            <List
              dataSource={semanticMemories}
              renderItem={(memory) => renderMemoryCard(memory, '语义', '#722ed1')}
              loading={loading}
            />
          </Card>
        </Col>
      </Row>

      {/* 记忆流转示意 */}
      <Card style={{ marginTop: 24 }}>
        <div style={{ textAlign: 'center' }}>
          <Space size="large">
            <div>
              <ThunderboltOutlined style={{ fontSize: 32, color: '#52c41a' }} />
              <div>工作记忆</div>
            </div>
            <RiseOutlined style={{ fontSize: 24 }} />
            <div>
              <ClockCircleOutlined style={{ fontSize: 32, color: '#1890ff' }} />
              <div>情景记忆</div>
            </div>
            <RiseOutlined style={{ fontSize: 24 }} />
            <div>
              <BookOutlined style={{ fontSize: 32, color: '#722ed1' }} />
              <div>语义记忆</div>
            </div>
          </Space>
          <div style={{ marginTop: 16, color: '#666' }}>
            记忆通过重要性评估和访问频率，逐级提升到更持久的存储层
          </div>
        </div>
      </Card>
    </div>
  )
}

// 修复Text组件导入
const Text: React.FC<{ style?: React.CSSProperties; ellipsis?: boolean; children: React.ReactNode }> = 
  ({ style, children }) => (
    <div style={{ ...style, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
      {children}
    </div>
  )

export default MemoryHierarchyPage
