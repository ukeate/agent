import React from 'react';
import QLearningLiveView from './qlearning/QLearningLiveView';

const QLearningPage: React.FC = () => (
  <QLearningLiveView title="Q-Learning 总览" subtitle="统一展示实时会话与算法支持，无随机假数据" />
);

export default QLearningPage;
