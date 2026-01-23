import React, { useEffect, useMemo, useState } from 'react'
import { apiClient } from '../services/apiClient'

interface CoverageFile {
  path: string
  coverage_percentage: number
  statements: number
  missing: number
}

interface CoverageReport {
  coverage_file?: string
  generated_at?: string
  overall_percentage: number
  total_files: number
  files: CoverageFile[]
}

const TestCoveragePage: React.FC = () => {
  const [report, setReport] = useState<CoverageReport | null>(null)
  const [filter, setFilter] = useState('')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false

    const load = async () => {
      try {
        setLoading(true)
        setError(null)
        const response =
          await apiClient.get<CoverageReport>('/testing/coverage')
        if (!cancelled) setReport(response.data)
      } catch (e: any) {
        if (!cancelled) setError(e?.message || '加载失败')
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    load()
    return () => {
      cancelled = true
    }
  }, [])

  const getCoverageColor = (percentage: number) => {
    if (percentage >= 95) return 'text-green-600'
    if (percentage >= 90) return 'text-yellow-600'
    if (percentage >= 80) return 'text-orange-600'
    return 'text-red-600'
  }

  const files = useMemo(() => {
    const all = report?.files || []
    const filtered = filter ? all.filter(f => f.path.includes(filter)) : all
    return [...filtered].sort(
      (a, b) => a.coverage_percentage - b.coverage_percentage
    )
  }, [report, filter])

  if (loading) {
    return <div className="p-6">加载测试覆盖率数据...</div>
  }

  if (error) {
    return <div className="p-6 text-red-600">加载失败：{error}</div>
  }

  if (!report || report.total_files === 0) {
    return (
      <div className="p-6">
        <h1 className="text-2xl font-bold mb-2">测试覆盖率</h1>
        <p className="text-gray-600">
          暂无覆盖率数据（未找到可用的 .coverage 文件）
        </p>
      </div>
    )
  }

  const generatedAt = report.generated_at
    ? new Date(report.generated_at).toLocaleString('zh-CN')
    : '未知'

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-6">
        <h1 className="text-3xl font-bold mb-2">测试覆盖率</h1>
        <p className="text-gray-600">
          数据来源：后端读取最新 .coverage 并计算文件级覆盖率
        </p>
      </div>

      <div className="bg-white rounded-lg shadow p-4 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <span className="block text-sm text-gray-500">整体覆盖率</span>
            <span
              className={`text-2xl font-bold ${getCoverageColor(report.overall_percentage)}`}
            >
              {report.overall_percentage.toFixed(2)}%
            </span>
          </div>
          <div>
            <span className="block text-sm text-gray-500">文件数量</span>
            <span className="text-2xl font-bold">{report.total_files}</span>
          </div>
          <div>
            <span className="block text-sm text-gray-500">生成时间</span>
            <span className="text-sm text-gray-700">{generatedAt}</span>
          </div>
        </div>
      </div>

      <div className="bg-white rounded-lg shadow p-4">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 mb-4">
          <h2 className="text-lg font-semibold">文件覆盖率（按覆盖率升序）</h2>
          <input
            value={filter}
            onChange={e => setFilter(e.target.value)}
            placeholder="按路径过滤"
            className="border rounded px-3 py-2 text-sm"
          />
        </div>

        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead className="text-left text-gray-500">
              <tr>
                <th className="py-2 pr-4">路径</th>
                <th className="py-2 pr-4 whitespace-nowrap">覆盖率</th>
                <th className="py-2 pr-4 whitespace-nowrap">missing/total</th>
              </tr>
            </thead>
            <tbody>
              {files.slice(0, 200).map(f => (
                <tr key={f.path} className="border-t">
                  <td className="py-2 pr-4 font-mono text-xs">{f.path}</td>
                  <td
                    className={`py-2 pr-4 whitespace-nowrap ${getCoverageColor(f.coverage_percentage)}`}
                  >
                    {f.coverage_percentage.toFixed(2)}%
                  </td>
                  <td className="py-2 pr-4 whitespace-nowrap">
                    {f.missing}/{f.statements}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {files.length > 200 && (
          <div className="text-xs text-gray-500 mt-3">
            仅显示前 200 条，可通过过滤缩小范围。
          </div>
        )}
      </div>
    </div>
  )
}

export default TestCoveragePage
