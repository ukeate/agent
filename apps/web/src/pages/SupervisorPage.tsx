/**
 * Supervisor监控页面
 * 提供完整的Supervisor系统管理界面
 */

import React from 'react'
import { SupervisorDashboard } from '../components/supervisor'

export const SupervisorPage: React.FC = () => {
  return (
    <div className="supervisor-page min-h-screen bg-gray-50">
      <SupervisorDashboard 
        supervisorId="main_supervisor"
        className="max-w-7xl mx-auto"
      />
    </div>
  )
}

export default SupervisorPage