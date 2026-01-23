import React from 'react'
import QLearningLiveView from './QLearningLiveView'

const ThompsonSamplingPage: React.FC = () => (
  <QLearningLiveView
    title="Thompson Sampling"
    subtitle="直接使用后端真实会话数据"
  />
)

export default ThompsonSamplingPage
