import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import { describe, it, expect, vi } from 'vitest';
import QLearningPage from '../../src/pages/QLearningPage';

// Mock navigation
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

// Mock child components
vi.mock('../../src/components/qlearning/QLearningAgentPanel', () => ({
  QLearningAgentPanel: ({ agents, loading, onCreateAgent }: any) => (
    <div data-testid="qlearning-agent-panel">
      <div>Agents: {agents.length}</div>
      <button onClick={() => onCreateAgent('dqn')} data-testid="create-agent-btn">
        创建智能体
      </button>
    </div>
  ),
}));

vi.mock('../../src/components/qlearning/QLearningTrainingPanel', () => ({
  QLearningTrainingPanel: ({ trainingSessions }: any) => (
    <div data-testid="qlearning-training-panel">
      <div>Training Sessions: {trainingSessions.length}</div>
    </div>
  ),
}));

vi.mock('../../src/components/qlearning/QLearningVisualization', () => ({
  QLearningVisualization: ({ agents }: any) => (
    <div data-testid="qlearning-visualization">
      <div>Visualization for {agents.length} agents</div>
    </div>
  ),
}));

vi.mock('../../src/components/qlearning/QLearningEnvironmentPanel', () => ({
  QLearningEnvironmentPanel: () => (
    <div data-testid="qlearning-environment-panel">Environment Panel</div>
  ),
}));

const renderWithRouter = (component: React.ReactElement) => {
  return render(<BrowserRouter>{component}</BrowserRouter>);
};

describe('QLearningPage', () => {
  it('应该正确渲染页面标题和基本结构', async () => {
    renderWithRouter(<QLearningPage />);
    
    expect(screen.getByText('Q-Learning策略优化系统')).toBeInTheDocument();
    expect(screen.getByText('强化学习智能体训练与策略优化平台')).toBeInTheDocument();
    
    // 检查统计卡片
    expect(screen.getByText('活跃智能体')).toBeInTheDocument();
    expect(screen.getByText('训练中会话')).toBeInTheDocument();
    expect(screen.getByText('已完成训练')).toBeInTheDocument();
    expect(screen.getByText('平均性能')).toBeInTheDocument();
  });

  it('应该显示算法技术栈标签', () => {
    renderWithRouter(<QLearningPage />);
    
    expect(screen.getByText('Classic Q-Learning - 表格型算法，适用于离散小状态空间')).toBeInTheDocument();
    expect(screen.getByText('Deep Q-Network (DQN) - 神经网络逼近，处理高维状态')).toBeInTheDocument();
    expect(screen.getByText('Double DQN - 减少Q值高估偏差')).toBeInTheDocument();
    expect(screen.getByText('Dueling DQN - 分离状态价值和优势函数')).toBeInTheDocument();
  });

  it('应该能够切换标签页', async () => {
    renderWithRouter(<QLearningPage />);
    
    // 切换到智能体管理标签
    const agentsTab = screen.getByRole('tab', { name: /智能体管理/ });
    fireEvent.click(agentsTab);
    
    await waitFor(() => {
      expect(screen.getByTestId('qlearning-agent-panel')).toBeInTheDocument();
    });
    
    // 切换到训练监控标签
    const trainingTab = screen.getByRole('tab', { name: /训练监控/ });
    fireEvent.click(trainingTab);
    
    await waitFor(() => {
      expect(screen.getByTestId('qlearning-training-panel')).toBeInTheDocument();
    });
  });

  it('应该正确处理导航点击', () => {
    renderWithRouter(<QLearningPage />);
    
    // 点击表格Q-Learning卡片
    const tabularCard = screen.getByText('表格Q-Learning').closest('.ant-card');
    fireEvent.click(tabularCard!);
    
    expect(mockNavigate).toHaveBeenCalledWith('/qlearning/tabular');
    
    // 点击DQN卡片
    const dqnCard = screen.getByText('Deep Q-Network').closest('.ant-card');
    fireEvent.click(dqnCard!);
    
    expect(mockNavigate).toHaveBeenCalledWith('/qlearning/dqn');
  });

  it('应该能够创建新智能体', async () => {
    renderWithRouter(<QLearningPage />);
    
    // 切换到智能体管理标签
    const agentsTab = screen.getByRole('tab', { name: /智能体管理/ });
    fireEvent.click(agentsTab);
    
    await waitFor(() => {
      const createBtn = screen.getByTestId('create-agent-btn');
      fireEvent.click(createBtn);
      
      // 验证agents数量更新
      expect(screen.getByText('Agents: 4')).toBeInTheDocument(); // 3个初始 + 1个新创建
    });
  });

  it('应该显示功能总览导航', () => {
    renderWithRouter(<QLearningPage />);
    
    // 检查各种功能按钮
    expect(screen.getByText('Epsilon-Greedy系列')).toBeInTheDocument();
    expect(screen.getByText('Upper Confidence Bound')).toBeInTheDocument();
    expect(screen.getByText('Thompson Sampling')).toBeInTheDocument();
    expect(screen.getByText('基础奖励函数')).toBeInTheDocument();
    expect(screen.getByText('状态空间设计')).toBeInTheDocument();
    expect(screen.getByText('训练调度管理')).toBeInTheDocument();
  });

  it('应该正确处理刷新数据操作', async () => {
    renderWithRouter(<QLearningPage />);
    
    const refreshBtn = screen.getByRole('button', { name: /刷新数据/ });
    fireEvent.click(refreshBtn);
    
    // 验证loading状态
    expect(refreshBtn).toHaveAttribute('disabled');
    
    // 等待加载完成
    await waitFor(() => {
      expect(refreshBtn).not.toHaveAttribute('disabled');
    }, { timeout: 2000 });
  });
});