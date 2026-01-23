import React from 'react'

const KGReasoningPerformancePage: React.FC = () => {
  return (
    <div className="container mx-auto p-6">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-900 mb-6">
          知识图谱推理性能分析
        </h1>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* 推理性能监控 */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              推理性能监控
            </h2>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">平均推理时间:</span>
                <span className="font-medium">125ms</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">推理成功率:</span>
                <span className="font-medium text-green-600">98.5%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">缓存命中率:</span>
                <span className="font-medium text-blue-600">85.2%</span>
              </div>
            </div>
          </div>

          {/* 资源使用情况 */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              资源使用情况
            </h2>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">CPU使用率:</span>
                <span className="font-medium">45%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">内存使用:</span>
                <span className="font-medium">2.1GB</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">网络IO:</span>
                <span className="font-medium">15MB/s</span>
              </div>
            </div>
          </div>

          {/* 推理路径统计 */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              推理路径统计
            </h2>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600">平均路径长度:</span>
                <span className="font-medium">3.2跳</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">最大路径深度:</span>
                <span className="font-medium">8跳</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">路径发现数:</span>
                <span className="font-medium">1,247</span>
              </div>
            </div>
          </div>

          {/* 推理类型分布 */}
          <div className="bg-white rounded-lg shadow-md p-6 md:col-span-2">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              推理类型分布
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <div className="text-center p-4 bg-blue-50 rounded">
                <div className="text-2xl font-bold text-blue-600">65%</div>
                <div className="text-sm text-gray-600">演绎推理</div>
              </div>
              <div className="text-center p-4 bg-green-50 rounded">
                <div className="text-2xl font-bold text-green-600">25%</div>
                <div className="text-sm text-gray-600">归纳推理</div>
              </div>
              <div className="text-center p-4 bg-purple-50 rounded">
                <div className="text-2xl font-bold text-purple-600">8%</div>
                <div className="text-sm text-gray-600">类比推理</div>
              </div>
              <div className="text-center p-4 bg-orange-50 rounded">
                <div className="text-2xl font-bold text-orange-600">2%</div>
                <div className="text-sm text-gray-600">概率推理</div>
              </div>
            </div>
          </div>

          {/* 性能优化建议 */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              性能优化建议
            </h2>
            <ul className="space-y-2 text-sm">
              <li className="flex items-center text-green-600">
                <svg
                  className="w-4 h-4 mr-2"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                    clipRule="evenodd"
                  />
                </svg>
                索引优化完成
              </li>
              <li className="flex items-center text-yellow-600">
                <svg
                  className="w-4 h-4 mr-2"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  />
                </svg>
                考虑增加缓存层
              </li>
              <li className="flex items-center text-blue-600">
                <svg
                  className="w-4 h-4 mr-2"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
                    clipRule="evenodd"
                  />
                </svg>
                并行推理可提升30%
              </li>
            </ul>
          </div>
        </div>

        {/* 实时性能图表 */}
        <div className="mt-8 bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            实时性能趋势
          </h2>
          <div className="h-64 bg-gray-50 rounded-lg flex items-center justify-center">
            <p className="text-gray-500">
              性能趋势图表 (集成Chart.js或其他图表库)
            </p>
          </div>
        </div>

        {/* 详细统计表 */}
        <div className="mt-8 bg-white rounded-lg shadow-md p-6">
          <h2 className="text-xl font-semibold text-gray-800 mb-4">
            推理任务详细统计
          </h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    任务ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    推理类型
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    执行时间
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    置信度
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                    状态
                  </th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    #R001
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    演绎推理
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    89ms
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    0.95
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                      完成
                    </span>
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    #R002
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    归纳推理
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    234ms
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    0.87
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-blue-100 text-blue-800">
                      处理中
                    </span>
                  </td>
                </tr>
                <tr>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    #R003
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    类比推理
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    156ms
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    0.92
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                      完成
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  )
}

export default KGReasoningPerformancePage
