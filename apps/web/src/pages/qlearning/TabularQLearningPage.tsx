import React from 'react'
import QLearningLiveView from './QLearningLiveView'

const TabularQLearningPage: React.FC = () => (
  <QLearningLiveView
    title="表格 Q-Learning"
    subtitle="数据来源于 /api/v1/qlearning 实时接口"
  />
)

export default TabularQLearningPage
