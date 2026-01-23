/**
 * 批处理系统页面
 */

import React from 'react'
import { BatchProcessingDashboard } from '../components/batch/BatchProcessingDashboard'

const BatchProcessingPage: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">批处理系统</h1>
        <p className="text-gray-600 mt-2">
          管理和监控批处理作业，实时追踪任务执行状态和系统性能指标
        </p>
      </div>

      <BatchProcessingDashboard />
    </div>
  )
}

export default BatchProcessingPage
