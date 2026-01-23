import React from 'react'
import QLearningLiveView from './QLearningLiveView'

const LearningRateSchedulerPage: React.FC = () => (
  <QLearningLiveView title="学习率调度" subtitle="使用真实训练会话监控学习率" />
)

export default LearningRateSchedulerPage
