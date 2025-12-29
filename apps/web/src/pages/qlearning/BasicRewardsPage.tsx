import React from 'react';
import QLearningLiveView from './QLearningLiveView';

const BasicRewardsPage: React.FC = () => (
  <QLearningLiveView title="基础奖励" subtitle="直接使用 /api/v1/qlearning 实时数据" />
);

export default BasicRewardsPage;
