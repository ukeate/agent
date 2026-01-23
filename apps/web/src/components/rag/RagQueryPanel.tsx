/**
 * 基础RAG查询面板
 *
 * 功能包括：
 * - 查询输入框和搜索按钮，支持Enter键提交
 * - 文件类型过滤器（代码/文档/全部）和高级搜索选项
 * - 查询历史记录展示和快速重用功能
 * - 查询参数配置（结果数量1-50、相关性阈值0-1）
 * - 查询语法提示和自动补全功能
 */

import React, { useState, useCallback, useEffect, useRef } from 'react'
import {
  Input,
  Button,
  Select,
  Slider,
  Collapse,
  Space,
  Tag,
  Tooltip,
  AutoComplete,
  Card,
  Typography,
  Row,
  Col,
  message,
} from 'antd'
import {
  SearchOutlined,
  SettingOutlined,
  HistoryOutlined,
  QuestionCircleOutlined,
  ClearOutlined,
} from '@ant-design/icons'
import { useRagStore } from '../../stores/ragStore'
import { ragService, QueryRequest } from '../../services/ragService'

const { TextArea } = Input
const { Option } = Select
const { Panel } = Collapse
const { Text, Title } = Typography

// ==================== 组件props类型 ====================

interface RagQueryPanelProps {
  onSearch?: (request: QueryRequest) => void
  onResults?: (results: any) => void
  disabled?: boolean
  className?: string
}

// ==================== 主组件 ====================

const RagQueryPanel: React.FC<RagQueryPanelProps> = ({
  onSearch,
  onResults,
  disabled = false,
  className = '',
}) => {
  // ==================== 状态管理 ====================

  const {
    currentQuery,
    queryHistory,
    searchPreferences,
    isQuerying,
    error,
    setCurrentQuery,
    setIsQuerying,
    setQueryResults,
    setError,
    addToHistory,
    updateSearchPreferences,
    clearErrors,
  } = useRagStore()

  // ==================== 本地状态 ====================

  const [searchType, setSearchType] = useState<
    'semantic' | 'keyword' | 'hybrid'
  >(searchPreferences.default_search_type)
  const [resultLimit, setResultLimit] = useState(
    searchPreferences.default_limit
  )
  const [scoreThreshold, setScoreThreshold] = useState(
    searchPreferences.default_score_threshold
  )
  const [fileTypeFilter, setFileTypeFilter] = useState<'all' | 'code' | 'docs'>(
    'all'
  )
  const [autoCompleteOptions, setAutoCompleteOptions] = useState<string[]>([])

  const inputRef = useRef<any>(null)

  // ==================== 查询语法提示 ====================

  const syntaxTips = [
    'AND、OR、NOT - 逻辑操作符',
    '"精确短语" - 精确匹配',
    'title:关键词 - 标题搜索',
    'ext:py - 文件扩展名',
    'path:/src/ - 路径搜索',
  ]

  // ==================== 自动补全逻辑 ====================

  const generateAutoComplete = useCallback(
    (value: string) => {
      if (!value.trim()) {
        setAutoCompleteOptions([])
        return
      }

      const suggestions = []

      // 基于历史查询的建议
      const historyMatches = queryHistory
        .filter(h => h.query.toLowerCase().includes(value.toLowerCase()))
        .slice(0, 3)
        .map(h => h.query)

      suggestions.push(...historyMatches)

      // 常用搜索模式建议
      if (value.length > 2) {
        const patterns = [
          `如何${value}`,
          `${value}示例`,
          `${value}文档`,
          `${value}API`,
          `${value}错误`,
        ]
        suggestions.push(...patterns)
      }

      // 去重并限制数量
      const uniqueSuggestions = [...new Set(suggestions)].slice(0, 8)
      setAutoCompleteOptions(uniqueSuggestions)
    },
    [queryHistory]
  )

  // ==================== 搜索执行逻辑 ====================

  const handleSearch = useCallback(async () => {
    if (!currentQuery.trim()) {
      message.warning('请输入搜索关键词')
      return
    }

    // 构建搜索请求
    const filters: Record<string, any> = {}

    if (fileTypeFilter === 'code') {
      filters.content_type = [
        'python',
        'javascript',
        'typescript',
        'java',
        'cpp',
        'c',
      ]
    } else if (fileTypeFilter === 'docs') {
      filters.content_type = ['markdown', 'txt', 'pdf', 'doc']
    }

    const searchRequest: QueryRequest = {
      query: currentQuery.trim(),
      search_type: searchType,
      limit: resultLimit,
      score_threshold: scoreThreshold,
      filters: Object.keys(filters).length > 0 ? filters : undefined,
    }

    try {
      setIsQuerying(true)
      setError(null)
      clearErrors()

      // 调用API
      const response = await ragService.query(searchRequest)

      if (response.success) {
        setQueryResults(response.results)

        // 添加到历史记录
        addToHistory(
          currentQuery,
          'basic',
          response.results.length,
          response.processing_time
        )

        // 回调通知
        onSearch?.(searchRequest)
        onResults?.(response)

        message.success(`找到 ${response.results.length} 个相关结果`)
      } else {
        throw new Error(response.error || '搜索失败')
      }
    } catch (err: any) {
      const errorMsg = err.message || '搜索请求失败'
      setError(errorMsg)
      message.error(errorMsg)
    } finally {
      setIsQuerying(false)
    }
  }, [
    currentQuery,
    searchType,
    resultLimit,
    scoreThreshold,
    fileTypeFilter,
    onSearch,
    onResults,
    setIsQuerying,
    setError,
    setQueryResults,
    addToHistory,
    clearErrors,
  ])

  // ==================== 事件处理 ====================

  const handleQueryChange = useCallback(
    (value: string) => {
      setCurrentQuery(value)
      generateAutoComplete(value)
    },
    [setCurrentQuery, generateAutoComplete]
  )

  const handleEnterPress = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault()
        handleSearch()
      }
    },
    [handleSearch]
  )

  const handleHistorySelect = useCallback(
    (query: string) => {
      setCurrentQuery(query)
      inputRef.current?.focus()
    },
    [setCurrentQuery]
  )

  const handleClearQuery = useCallback(() => {
    setCurrentQuery('')
    setError(null)
    clearErrors()
    inputRef.current?.focus()
  }, [setCurrentQuery, setError, clearErrors])

  const handlePreferencesChange = useCallback(() => {
    updateSearchPreferences({
      default_search_type: searchType,
      default_limit: resultLimit,
      default_score_threshold: scoreThreshold,
    })
    message.success('搜索偏好已保存')
  }, [searchType, resultLimit, scoreThreshold, updateSearchPreferences])

  // ==================== 生命周期 ====================

  useEffect(() => {
    // 组件加载时设置焦点
    inputRef.current?.focus()
  }, [])

  // ==================== 渲染组件 ====================

  return (
    <Card
      className={`rag-query-panel ${className}`}
      title={
        <Space>
          <SearchOutlined />
          <Title level={4} style={{ margin: 0 }}>
            RAG 混合搜索
          </Title>
        </Space>
      }
    >
      {/* 主搜索区域 */}
      <Space direction="vertical" style={{ width: '100%' }} size="middle">
        {/* 查询输入框 */}
        <AutoComplete
          options={autoCompleteOptions.map(opt => ({ value: opt }))}
          style={{ width: '100%' }}
          onSelect={handleQueryChange}
          disabled={disabled}
        >
          <TextArea
            ref={inputRef}
            placeholder='请输入搜索关键词... (支持 AND/OR 逻辑操作符, "精确短语", title:关键词等语法)'
            value={currentQuery}
            onChange={e => handleQueryChange(e.target.value)}
            onKeyDown={handleEnterPress}
            autoSize={{ minRows: 2, maxRows: 4 }}
            disabled={disabled}
            status={error ? 'error' : undefined}
          />
        </AutoComplete>

        {/* 错误提示 */}
        {error && (
          <Text type="danger" style={{ fontSize: 12 }}>
            {error}
          </Text>
        )}

        {/* 快速操作栏 */}
        <Row gutter={16} align="middle">
          <Col flex="auto">
            <Space>
              {/* 文件类型过滤 */}
              <Select
                value={fileTypeFilter}
                onChange={setFileTypeFilter}
                style={{ width: 100 }}
                size="small"
                disabled={disabled}
              >
                <Option value="all">全部</Option>
                <Option value="code">代码</Option>
                <Option value="docs">文档</Option>
              </Select>

              {/* 搜索类型 */}
              <Select
                value={searchType}
                onChange={setSearchType}
                style={{ width: 100 }}
                size="small"
                disabled={disabled}
              >
                <Option value="hybrid">混合</Option>
                <Option value="semantic">向量</Option>
                <Option value="keyword">BM25</Option>
              </Select>
            </Space>
          </Col>

          <Col>
            <Space>
              {/* 清空按钮 */}
              <Tooltip title="清空查询">
                <Button
                  icon={<ClearOutlined />}
                  size="small"
                  onClick={handleClearQuery}
                  disabled={disabled || !currentQuery}
                />
              </Tooltip>

              {/* 搜索按钮 */}
              <Button
                type="primary"
                icon={<SearchOutlined />}
                loading={isQuerying}
                onClick={handleSearch}
                disabled={disabled || !currentQuery.trim()}
              >
                搜索
              </Button>
            </Space>
          </Col>
        </Row>

        {/* 高级选项和历史记录 */}
        <Collapse ghost size="small">
          {/* 高级搜索选项 */}
          <Panel
            header={
              <Space>
                <SettingOutlined />
                <Text>高级选项</Text>
              </Space>
            }
            key="advanced"
          >
            <Space direction="vertical" style={{ width: '100%' }} size="middle">
              {/* 结果数量 */}
              <Row align="middle">
                <Col span={6}>
                  <Text>结果数量:</Text>
                </Col>
                <Col span={14}>
                  <Slider
                    min={1}
                    max={50}
                    value={resultLimit}
                    onChange={setResultLimit}
                    disabled={disabled}
                  />
                </Col>
                <Col span={4} style={{ textAlign: 'right' }}>
                  <Text code>{resultLimit}</Text>
                </Col>
              </Row>

              {/* 相关性阈值 */}
              <Row align="middle">
                <Col span={6}>
                  <Text>相关性阈值:</Text>
                </Col>
                <Col span={14}>
                  <Slider
                    min={0}
                    max={1}
                    step={0.1}
                    value={scoreThreshold}
                    onChange={setScoreThreshold}
                    disabled={disabled}
                  />
                </Col>
                <Col span={4} style={{ textAlign: 'right' }}>
                  <Text code>{scoreThreshold.toFixed(1)}</Text>
                </Col>
              </Row>

              {/* 保存偏好按钮 */}
              <Row>
                <Col span={24} style={{ textAlign: 'right' }}>
                  <Button
                    size="small"
                    onClick={handlePreferencesChange}
                    disabled={disabled}
                  >
                    保存为默认设置
                  </Button>
                </Col>
              </Row>
            </Space>
          </Panel>

          {/* 查询历史 */}
          <Panel
            header={
              <Space>
                <HistoryOutlined />
                <Text>查询历史 ({queryHistory.length})</Text>
              </Space>
            }
            key="history"
          >
            {queryHistory.length > 0 ? (
              <Space
                direction="vertical"
                style={{ width: '100%' }}
                size="small"
              >
                {queryHistory.slice(0, 10).map(item => (
                  <div
                    key={item.id}
                    style={{
                      padding: 8,
                      border: '1px solid #f0f0f0',
                      borderRadius: 4,
                      cursor: 'pointer',
                    }}
                    onClick={() => handleHistorySelect(item.query)}
                  >
                    <Space
                      direction="vertical"
                      size="small"
                      style={{ width: '100%' }}
                    >
                      <Text ellipsis style={{ maxWidth: '100%' }}>
                        {item.query}
                      </Text>
                      <Space size="small">
                        <Tag color={item.type === 'basic' ? 'blue' : 'green'}>
                          {item.type === 'basic' ? '基础' : '混合'}
                        </Tag>
                        <Text type="secondary" style={{ fontSize: 11 }}>
                          {item.results_count} 结果
                        </Text>
                        <Text type="secondary" style={{ fontSize: 11 }}>
                          {item.processing_time
                            ? item.processing_time.toFixed(2)
                            : '0.00'}
                          s
                        </Text>
                        <Text type="secondary" style={{ fontSize: 11 }}>
                          {new Date(item.timestamp).toLocaleTimeString()}
                        </Text>
                      </Space>
                    </Space>
                  </div>
                ))}
              </Space>
            ) : (
              <Text type="secondary">暂无查询历史</Text>
            )}
          </Panel>

          {/* 语法提示 */}
          <Panel
            header={
              <Space>
                <QuestionCircleOutlined />
                <Text>搜索语法</Text>
              </Space>
            }
            key="syntax"
          >
            <Space direction="vertical" size="small">
              {syntaxTips.map((tip, index) => (
                <Text key={index} code style={{ fontSize: 12 }}>
                  {tip}
                </Text>
              ))}
            </Space>
          </Panel>
        </Collapse>
      </Space>
    </Card>
  )
}

export default RagQueryPanel
