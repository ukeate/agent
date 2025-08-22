import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import ExplorationStrategiesPage from '../../src/pages/qlearning/ExplorationStrategiesPage';

// Mock Ant Design Charts
vi.mock('@ant-design/charts', () => ({
  Line: ({ data, ...props }: any) => (
    <div data-testid="line-chart">
      <div>Chart with {data?.length || 0} data points</div>
      <div>{JSON.stringify(props)}</div>
    </div>
  ),
}));

describe('ExplorationStrategiesPage', () => {
  it('应该正确渲染页面标题和描述', () => {
    render(<ExplorationStrategiesPage />);
    
    expect(screen.getByText('探索策略系统 (Exploration Strategies)')).toBeInTheDocument();
    expect(screen.getByText(/平衡探索与利用的核心机制/)).toBeInTheDocument();
  });

  it('应该显示探索策略配置选项', () => {
    render(<ExplorationStrategiesPage />);
    
    expect(screen.getByText('策略类型:')).toBeInTheDocument();
    expect(screen.getByText(/探索率.*ε/)).toBeInTheDocument();
    
    // 检查策略选择器中的选项
    const strategySelect = screen.getByDisplayValue('Epsilon-Greedy');
    expect(strategySelect).toBeInTheDocument();
  });

  it('应该能够切换探索策略', async () => {
    render(<ExplorationStrategiesPage />);
    
    const strategySelect = screen.getByRole('combobox');
    
    // 模拟选择不同策略
    fireEvent.mouseDown(strategySelect);
    
    await waitFor(() => {
      const thompsonOption = screen.getByText('Thompson Sampling');
      fireEvent.click(thompsonOption);
    });
    
    expect(screen.getByText('当前策略: Thompson Sampling')).toBeInTheDocument();
  });

  it('应该显示训练状态监控', () => {
    render(<ExplorationStrategiesPage />);
    
    expect(screen.getByText('训练步数')).toBeInTheDocument();
    expect(screen.getByText('当前探索率')).toBeInTheDocument();
    expect(screen.getByText('总动作数')).toBeInTheDocument();
  });

  it('应该能够开始和暂停训练', async () => {
    render(<ExplorationStrategiesPage />);
    
    const trainBtn = screen.getByRole('button', { name: /开始训练/ });
    fireEvent.click(trainBtn);
    
    await waitFor(() => {
      expect(screen.getByText('暂停')).toBeInTheDocument();
    });
    
    // 再次点击应该暂停
    fireEvent.click(trainBtn);
    expect(screen.getByText('开始训练')).toBeInTheDocument();
  });

  it('应该显示探索vs利用平衡原则', () => {
    render(<ExplorationStrategiesPage />);
    
    expect(screen.getByText('探索 (Exploration):')).toBeInTheDocument();
    expect(screen.getByText('尝试未知的动作')).toBeInTheDocument();
    expect(screen.getByText('发现潜在的更优策略')).toBeInTheDocument();
    
    expect(screen.getByText('利用 (Exploitation):')).toBeInTheDocument();
    expect(screen.getByText('选择当前已知的最优动作')).toBeInTheDocument();
    expect(screen.getByText('最大化即时奖励')).toBeInTheDocument();
  });

  it('应该显示策略效果可视化图表', () => {
    render(<ExplorationStrategiesPage />);
    
    expect(screen.getByText('探索率变化曲线')).toBeInTheDocument();
    expect(screen.getByText('奖励收敛情况')).toBeInTheDocument();
    
    // 初始状态应该显示提示文字
    expect(screen.getByText('开始训练以查看探索率变化')).toBeInTheDocument();
    expect(screen.getByText('开始训练以查看奖励变化')).toBeInTheDocument();
  });

  it('应该显示动作选择统计', () => {
    render(<ExplorationStrategiesPage />);
    
    expect(screen.getByText('动作选择统计')).toBeInTheDocument();
    expect(screen.getByText('动作')).toBeInTheDocument();
    expect(screen.getByText('选择次数')).toBeInTheDocument();
    expect(screen.getByText('选择比例')).toBeInTheDocument();
  });

  it('应该显示各种探索策略的详细原理', () => {
    render(<ExplorationStrategiesPage />);
    
    // 检查标签页
    expect(screen.getByText('Epsilon-Greedy')).toBeInTheDocument();
    expect(screen.getByText('衰减Epsilon-Greedy')).toBeInTheDocument();
    expect(screen.getByText('Upper Confidence Bound')).toBeInTheDocument();
    expect(screen.getByText('Thompson Sampling')).toBeInTheDocument();
  });

  it('应该能够切换策略原理标签页', async () => {
    render(<ExplorationStrategiesPage />);
    
    // 点击UCB标签页
    const ucbTab = screen.getByRole('tab', { name: 'Upper Confidence Bound' });
    fireEvent.click(ucbTab);
    
    await waitFor(() => {
      expect(screen.getByText('Upper Confidence Bound (UCB)')).toBeInTheDocument();
      expect(screen.getByText(/UCB公式/)).toBeInTheDocument();
    });
  });

  it('应该能够重置训练数据', () => {
    render(<ExplorationStrategiesPage />);
    
    const resetBtn = screen.getByRole('button', { name: /重置/ });
    fireEvent.click(resetBtn);
    
    // 验证数据被重置
    expect(screen.getByText('训练步数')).toBeInTheDocument();
  });

  it('应该正确调整探索率参数', () => {
    render(<ExplorationStrategiesPage />);
    
    const epsilonSlider = screen.getByRole('slider');
    fireEvent.change(epsilonSlider, { target: { value: '0.3' } });
    
    expect(screen.getByText(/探索率.*ε.*: 0\.3/)).toBeInTheDocument();
  });
});