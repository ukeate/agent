import React from 'react'
import QLearningLiveView from './QLearningLiveView'

const PerformanceTrackerPage: React.FC = () => (
  <QLearningLiveView
    title="性能跟踪"
    subtitle="表格数据全部来自 /api/v1/qlearning"
  />
)

export default PerformanceTrackerPage
