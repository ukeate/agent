import React, { useState, useEffect } from 'react'
import { Alert } from '../components/ui/alert'
import {
  memoryManagementService,
  MemoryType,
  MemoryStatus,
  MemoryResponse,
  MemoryAnalytics,
  MemoryPatterns,
  MemoryTrends,
  MemoryGraphStats,
  MemoryCreateRequest,
} from '../services/memoryManagementService'

const MemoryManagementMonitorPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<
    'overview' | 'memories' | 'analytics' | 'patterns' | 'cleanup'
  >('overview')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // 状态数据
  const [analytics, setAnalytics] = useState<MemoryAnalytics | null>(null)
  const [patterns, setPatterns] = useState<MemoryPatterns | null>(null)
  const [trends, setTrends] = useState<MemoryTrends | null>(null)
  const [graphStats, setGraphStats] = useState<MemoryGraphStats | null>(null)
  const [memories, setMemories] = useState<MemoryResponse[]>([])
  const [searchQuery, setSearchQuery] = useState('')
  const [selectedSession, setSelectedSession] = useState('')

  // 创建记忆表单
  const [createForm, setCreateForm] = useState<MemoryCreateRequest>({
    type: MemoryType.WORKING,
    content: '',
    importance: 0.5,
    tags: [],
    source: '',
  })

  // 过滤和搜索状态
  const [filters, setFilters] = useState({
    memoryTypes: [] as MemoryType[],
    status: [] as MemoryStatus[],
    minImportance: 0,
    maxImportance: 1,
    tags: [] as string[],
  })

  useEffect(() => {
    loadData()
  }, [])

  const loadData = async () => {
    setLoading(true)
    setError(null)
    try {
      const [analyticsData, patternsData, trendsData, graphData] =
        await Promise.all([
          memoryManagementService.getMemoryAnalytics(),
          memoryManagementService.getMemoryPatterns(),
          memoryManagementService.getMemoryTrends(),
          memoryManagementService.getMemoryGraphStats(),
        ])

      setAnalytics(analyticsData)
      setPatterns(patternsData)
      setTrends(trendsData)
      setGraphStats(graphData)
    } catch (err: any) {
      setError(`加载数据失败: ${err.response?.data?.detail || err.message}`)
    }
    setLoading(false)
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) return

    setLoading(true)
    try {
      const results = await memoryManagementService.searchMemories({
        query: searchQuery,
        memory_types:
          filters.memoryTypes.length > 0 ? filters.memoryTypes : undefined,
        status: filters.status.length > 0 ? filters.status : undefined,
        min_importance:
          filters.minImportance > 0 ? filters.minImportance : undefined,
        max_importance:
          filters.maxImportance < 1 ? filters.maxImportance : undefined,
        tags: filters.tags.length > 0 ? filters.tags : undefined,
        session_id: selectedSession || undefined,
        limit: 50,
      })
      setMemories(results)
    } catch (err: any) {
      setError(`搜索失败: ${err.response?.data?.detail || err.message}`)
    }
    setLoading(false)
  }

  const handleCreateMemory = async () => {
    if (!createForm.content.trim()) return

    setLoading(true)
    try {
      await memoryManagementService.createMemory(
        createForm,
        selectedSession || undefined
      )
      setCreateForm({
        type: MemoryType.WORKING,
        content: '',
        importance: 0.5,
        tags: [],
        source: '',
      })
      await loadData() // 刷新数据
      Alert.success('记忆创建成功')
    } catch (err: any) {
      setError(`创建记忆失败: ${err.response?.data?.detail || err.message}`)
    }
    setLoading(false)
  }

  const handleDeleteMemory = async (memoryId: string) => {
    if (!confirm('确认删除此记忆？')) return

    setLoading(true)
    try {
      await memoryManagementService.deleteMemory(memoryId)
      setMemories(memories.filter(m => m.id !== memoryId))
      Alert.success('记忆删除成功')
    } catch (err: any) {
      setError(`删除记忆失败: ${err.response?.data?.detail || err.message}`)
    }
    setLoading(false)
  }

  const handleConsolidate = async () => {
    if (!selectedSession) {
      setError('请先选择会话ID')
      return
    }

    setLoading(true)
    try {
      await memoryManagementService.consolidateSessionMemories(selectedSession)
      await loadData()
      Alert.success('记忆巩固完成')
    } catch (err: any) {
      setError(`记忆巩固失败: ${err.response?.data?.detail || err.message}`)
    }
    setLoading(false)
  }

  const handleCleanup = async () => {
    if (!confirm('确认清理旧记忆？此操作不可撤销。')) return

    setLoading(true)
    try {
      await memoryManagementService.cleanupOldMemories(30, 0.3)
      await loadData()
      Alert.success('记忆清理完成')
    } catch (err: any) {
      setError(`记忆清理失败: ${err.response?.data?.detail || err.message}`)
    }
    setLoading(false)
  }

  const renderOverview = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">记忆系统概览</h2>

      {analytics && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-blue-800">总记忆数</h3>
            <p className="text-2xl font-bold text-blue-900">
              {analytics.total_memories}
            </p>
          </div>
          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-green-800">平均重要性</h3>
            <p className="text-2xl font-bold text-green-900">
              {analytics.avg_importance.toFixed(2)}
            </p>
          </div>
          <div className="bg-purple-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-purple-800">访问总数</h3>
            <p className="text-2xl font-bold text-purple-900">
              {analytics.total_access_count}
            </p>
          </div>
          <div className="bg-orange-50 p-4 rounded-lg">
            <h3 className="text-lg font-semibold text-orange-800">
              存储使用(MB)
            </h3>
            <p className="text-2xl font-bold text-orange-900">
              {analytics.storage_usage_mb.toFixed(2)}
            </p>
          </div>
        </div>
      )}

      {analytics && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">记忆类型分布</h3>
            <div className="space-y-2">
              {Object.entries(analytics.memories_by_type).map(
                ([type, count]) => (
                  <div key={type} className="flex justify-between items-center">
                    <span className="capitalize">{type}</span>
                    <span className="font-bold">{count}</span>
                  </div>
                )
              )}
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">记忆状态分布</h3>
            <div className="space-y-2">
              {Object.entries(analytics.memories_by_status).map(
                ([status, count]) => (
                  <div
                    key={status}
                    className="flex justify-between items-center"
                  >
                    <span className="capitalize">{status}</span>
                    <span className="font-bold">{count}</span>
                  </div>
                )
              )}
            </div>
          </div>
        </div>
      )}

      {graphStats && (
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">记忆关联图统计</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-600">节点数</p>
              <p className="text-xl font-bold">
                {graphStats.graph_overview.total_nodes}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">连接数</p>
              <p className="text-xl font-bold">
                {graphStats.graph_overview.total_edges}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">图密度</p>
              <p className="text-xl font-bold">
                {graphStats.graph_overview.density.toFixed(3)}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">连通分量</p>
              <p className="text-xl font-bold">
                {graphStats.graph_overview.connected_components}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )

  const renderMemories = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">记忆管理</h2>

      {/* 创建记忆表单 */}
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">创建新记忆</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">记忆类型</label>
            <select
              value={createForm.type}
              onChange={e =>
                setCreateForm({
                  ...createForm,
                  type: e.target.value as MemoryType,
                })
              }
              className="w-full p-2 border rounded"
            >
              <option value={MemoryType.WORKING}>工作记忆</option>
              <option value={MemoryType.EPISODIC}>情景记忆</option>
              <option value={MemoryType.SEMANTIC}>语义记忆</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">
              重要性 (0-1)
            </label>
            <input
              type="number"
              min="0"
              max="1"
              step="0.1"
              value={createForm.importance}
              onChange={e =>
                setCreateForm({
                  ...createForm,
                  importance: parseFloat(e.target.value),
                })
              }
              className="w-full p-2 border rounded"
            />
          </div>
        </div>
        <div className="mt-4">
          <label className="block text-sm font-medium mb-2">记忆内容</label>
          <textarea
            value={createForm.content}
            onChange={e =>
              setCreateForm({ ...createForm, content: e.target.value })
            }
            placeholder="输入记忆内容..."
            className="w-full p-2 border rounded h-24"
          />
        </div>
        <div className="mt-4 flex gap-2">
          <input
            type="text"
            value={selectedSession}
            onChange={e => setSelectedSession(e.target.value)}
            placeholder="会话ID（可选）"
            className="flex-1 p-2 border rounded"
          />
          <button
            onClick={handleCreateMemory}
            disabled={loading || !createForm.content.trim()}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
          >
            创建记忆
          </button>
        </div>
      </div>

      {/* 搜索记忆 */}
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">搜索记忆</h3>
        <div className="flex gap-2 mb-4">
          <input
            type="text"
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            placeholder="输入搜索关键词..."
            className="flex-1 p-2 border rounded"
          />
          <button
            onClick={handleSearch}
            disabled={loading || !searchQuery.trim()}
            className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
          >
            搜索
          </button>
        </div>
      </div>

      {/* 记忆列表 */}
      {memories.length > 0 && (
        <div className="bg-white rounded-lg shadow">
          <h3 className="text-lg font-semibold p-4 border-b">搜索结果</h3>
          <div className="divide-y">
            {memories.map(memory => (
              <div key={memory.id} className="p-4">
                <div className="flex justify-between items-start mb-2">
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className={`px-2 py-1 rounded text-xs ${
                          memory.type === 'working'
                            ? 'bg-blue-100 text-blue-800'
                            : memory.type === 'episodic'
                              ? 'bg-green-100 text-green-800'
                              : 'bg-purple-100 text-purple-800'
                        }`}
                      >
                        {memory.type}
                      </span>
                      <span
                        className={`px-2 py-1 rounded text-xs ${
                          memory.status === 'active'
                            ? 'bg-green-100 text-green-800'
                            : memory.status === 'archived'
                              ? 'bg-yellow-100 text-yellow-800'
                              : 'bg-gray-100 text-gray-800'
                        }`}
                      >
                        {memory.status}
                      </span>
                      <span className="text-xs text-gray-500">
                        重要性: {memory.importance.toFixed(2)}
                      </span>
                      {memory.relevance_score && (
                        <span className="text-xs text-blue-600">
                          相关性: {memory.relevance_score.toFixed(2)}
                        </span>
                      )}
                    </div>
                    <p className="text-gray-800 mb-2">{memory.content}</p>
                    <div className="flex items-center gap-4 text-sm text-gray-500">
                      <span>访问次数: {memory.access_count}</span>
                      <span>
                        创建时间: {new Date(memory.created_at).toLocaleString()}
                      </span>
                      {memory.tags.length > 0 && (
                        <span>标签: {memory.tags.join(', ')}</span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => handleDeleteMemory(memory.id)}
                    className="px-3 py-1 text-red-600 hover:bg-red-50 rounded text-sm"
                  >
                    删除
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 记忆巩固 */}
      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">记忆巩固</h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={selectedSession}
            onChange={e => setSelectedSession(e.target.value)}
            placeholder="输入会话ID进行巩固"
            className="flex-1 p-2 border rounded"
          />
          <button
            onClick={handleConsolidate}
            disabled={loading || !selectedSession}
            className="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 disabled:opacity-50"
          >
            巩固记忆
          </button>
        </div>
      </div>
    </div>
  )

  const renderAnalytics = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">记忆分析</h2>

      {trends && (
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">趋势分析</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-600">
                {trends.summary.total_memories}
              </p>
              <p className="text-sm text-gray-600">总记忆数</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-green-600">
                {trends.summary.avg_daily_creation}
              </p>
              <p className="text-sm text-gray-600">日均创建数</p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-purple-600">
                {trends.summary.growth_rate.toFixed(2)}
              </p>
              <p className="text-sm text-gray-600">增长率</p>
            </div>
          </div>

          <h4 className="font-semibold mb-2">每日趋势</h4>
          <div className="space-y-2">
            {Object.entries(trends.daily_trends).map(([date, data]) => (
              <div
                key={date}
                className="flex justify-between items-center p-2 bg-gray-50 rounded"
              >
                <span className="font-medium">{date}</span>
                <div className="flex gap-4 text-sm">
                  <span>记忆: {data.memory_count}</span>
                  <span>重要性: {data.avg_importance.toFixed(2)}</span>
                  <span>访问: {data.total_access}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )

  const renderPatterns = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">使用模式</h2>

      {patterns && (
        <>
          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">时间模式</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-2">每日分布</h4>
                <div className="space-y-1">
                  {Object.entries(
                    patterns.time_patterns.daily_distribution
                  ).map(([date, count]) => (
                    <div key={date} className="flex justify-between">
                      <span className="text-sm">{date}</span>
                      <span className="font-medium">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <h4 className="font-medium mb-2">活跃天数</h4>
                <div className="space-y-1">
                  {patterns.usage_patterns.most_active_days.map(
                    ([date, count]) => (
                      <div key={date} className="flex justify-between">
                        <span className="text-sm">{date}</span>
                        <span className="font-medium">{count}</span>
                      </div>
                    )
                  )}
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white p-4 rounded-lg shadow">
            <h3 className="text-lg font-semibold mb-4">内容模式</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium mb-2">标签频率</h4>
                <div className="space-y-1">
                  {Object.entries(patterns.content_patterns.tag_frequency).map(
                    ([tag, count]) => (
                      <div key={tag} className="flex justify-between">
                        <span className="text-sm">{tag}</span>
                        <span className="font-medium">{count}</span>
                      </div>
                    )
                  )}
                </div>
              </div>
              <div>
                <h4 className="font-medium mb-2">高峰时段</h4>
                <div className="space-y-1">
                  {patterns.usage_patterns.peak_hours.map(([hour, count]) => (
                    <div key={hour} className="flex justify-between">
                      <span className="text-sm">{hour}:00</span>
                      <span className="font-medium">{count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )

  const renderCleanup = () => (
    <div className="space-y-6">
      <h2 className="text-2xl font-bold">记忆清理</h2>

      <div className="bg-white p-4 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">清理旧记忆</h3>
        <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 mb-4">
          <p className="text-yellow-800">
            清理功能将删除超过30天且重要性低于0.3的记忆。此操作不可撤销，请谨慎操作。
          </p>
        </div>
        <button
          onClick={handleCleanup}
          disabled={loading}
          className="px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50"
        >
          执行清理
        </button>
      </div>

      {analytics && (
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">存储统计</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-sm text-gray-600">活跃记忆</p>
              <p className="text-xl font-bold text-green-600">
                {analytics.memories_by_status.active || 0}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">已归档</p>
              <p className="text-xl font-bold text-yellow-600">
                {analytics.memories_by_status.archived || 0}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">已压缩</p>
              <p className="text-xl font-bold text-blue-600">
                {analytics.memories_by_status.compressed || 0}
              </p>
            </div>
            <div>
              <p className="text-sm text-gray-600">存储使用</p>
              <p className="text-xl font-bold text-purple-600">
                {analytics.storage_usage_mb.toFixed(2)} MB
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  )

  return (
    <div className="p-6">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">记忆管理监控</h1>
        <p className="text-gray-600">管理和监控AI智能体的记忆系统</p>
      </div>

      {error && (
        <Alert type="error" className="mb-6">
          {error}
        </Alert>
      )}

      <div className="mb-6">
        <nav className="flex space-x-1 bg-gray-100 p-1 rounded-lg">
          {[
            { key: 'overview', label: '概览' },
            { key: 'memories', label: '记忆管理' },
            { key: 'analytics', label: '分析统计' },
            { key: 'patterns', label: '使用模式' },
            { key: 'cleanup', label: '清理维护' },
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key as any)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === tab.key
                  ? 'bg-white text-blue-600 shadow-sm'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      {loading && (
        <div className="flex justify-center items-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        </div>
      )}

      <div className="bg-gray-50 rounded-lg p-6">
        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'memories' && renderMemories()}
        {activeTab === 'analytics' && renderAnalytics()}
        {activeTab === 'patterns' && renderPatterns()}
        {activeTab === 'cleanup' && renderCleanup()}
      </div>
    </div>
  )
}

export default MemoryManagementMonitorPage
