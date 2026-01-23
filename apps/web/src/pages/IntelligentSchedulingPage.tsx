/**
 * 智能调度监控页面
 */

import React from 'react'
import SchedulingMonitor from '../components/batch/SchedulingMonitor'

const IntelligentSchedulingPage: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-6">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">智能调度监控</h1>
        <p className="text-gray-600 mt-2">
          实时监控智能调度系统性能，包括工作者状态、SLA合规性和预测性调度建议
        </p>
      </div>

      <SchedulingMonitor />
    </div>
  )
}

export default IntelligentSchedulingPage
