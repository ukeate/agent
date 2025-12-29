import React from 'react';
import QLearningLiveView from './QLearningLiveView';

const EarlyStoppingPage: React.FC = () => (
  <QLearningLiveView title="早停策略" subtitle="展示真实训练进度，无本地模拟" />
);

export default EarlyStoppingPage;
